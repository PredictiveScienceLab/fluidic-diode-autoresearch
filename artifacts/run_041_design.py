"""
Fluidic Diode Design & Continuous Optimization Module.
Run 041: Refined Borda-Sting Interaction and Multi-Fidelity Hill Climbing.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 025 winner (Di = 1.974761)
# These are the current global best.
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9974176287651062, 
    0.05909641459584236, 0.22175829112529755, 0.561127245426178, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_trumpet_catcher_v3(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Refined Catcher Topology (v3):
    - Symmetric geometry focusing on steady-state inertia.
    - Nozzle ramps for forward streamlining.
    - Re-entrant Borda wall-lips optimized for reverse-jet generation.
    - Central island with a 'Trumpet' flare and a deep concave rear face.
    - Central splitter 'sting' with optimized width and taper.
    """
    # Mapping parameters (12 dims)
    sting_width_fac = 0.1 + 1.5 * params[0]
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]

    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    sting_taper_factor = 0.0 + 2.0 * params[8]
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    isl_flare       = 0.40 + 0.80 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Hooked Borda Lip
        lx = dx - ramp_len
        hook_curve = 3.5 * lip_hook 
        y_lip = throat - lip_hook * lx + hook_curve * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        # 3. Trumpet-Flared Island
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        
        # Profile: Streamlined nose (front) transitions to aggressive flare (back)
        nose_part = jnp.sqrt(jnp.clip(idx / 0.4, 0.0, 1.0)) * (idx < 0.4)
        flare_part = (1.0 + isl_flare * (jnp.maximum(0.0, idx - 0.4)**3)) * (idx >= 0.4)
        profile = jnp.where(idx < 0.4, nose_part, flare_part)
        
        # Taper the very ends to ensure a closed shape
        envelope = jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        half_w = (throat * 0.55) * profile * envelope
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # Deep Concavity + Splitter Sting
        cav_start = 1.0 - 0.45 * concavity
        cav_idx = (idx - cav_start) / (1.0 - cav_start + 1e-6)
        
        # The cavity itself
        cavity = (cav_idx > 0.0) & (ay < half_w * 0.85 * concavity)
        
        # The 'Sting' - central spike to split reverse flow
        base_sting_width = (throat * 0.05) * sting_width_fac 
        sting_cav_start = 0.2
        sting_cav_end = 0.8
        
        taper_progress = (cav_idx - sting_cav_start) / (sting_cav_end - sting_cav_start + 1e-6)
        current_sting_width = base_sting_width * jnp.clip(1.0 - sting_taper_factor * taper_progress, 0.0, 1.0)
        
        sting = (cav_idx > sting_cav_start) & (cav_idx < sting_cav_end) & (ay < current_sting_width)
        
        island_solid = (island_outer.astype(jnp.float32) * (1.0 - cavity.astype(jnp.float32)))
        island_solid = jnp.maximum(island_solid, sting.astype(jnp.float32))

        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_solid)

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_trumpet_catcher_v3(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_trumpet_catcher_v3(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    
    num_params = 12
    n_coarse = 64 
    
    # Population generation
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS 
    
    # Local search around previous best
    pop[1:24] = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.06, (23, num_params)), 0, 1)
    
    # Targeted exploration for sting and lip parameters
    for i in range(24, 40):
        p = PREV_BEST_PARAMS.copy()
        p[5] = rng.uniform(0.1, 0.3) # Lip hook focus
        p[8] = rng.uniform(0.0, 0.15) # Sting taper focus
        pop[i] = jnp.clip(p + rng.normal(0, 0.04, num_params), 0, 1)

    # Round 1: Coarse Screening (FAST_CONFIG)
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Select top candidates for fine evaluation
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop_1 = pop[top_idx]
    
    # Round 2: Fine Refinement (DEFAULT_CONFIG)
    masks_f1 = render_batch_fine(refine_pop_1)
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # Round 3: Final Hill-Climb on the best fine candidate
    best_f1_idx = np.argmax(np.asarray(scores_f1))
    best_f1_p = refine_pop_1[best_f1_idx]
    
    final_pop = jnp.tile(best_f1_p, (16, 1))
    final_pop = jnp.clip(final_pop + rng.normal(0, 0.015, (16, num_params)), 0, 1)
    final_pop = final_pop.at[0].set(best_f1_p) 
    
    masks_f2 = render_batch_fine(final_pop)
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_best_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = final_pop[final_best_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_best_idx])

if __name__ == "__main__":
    design = generate_design()
