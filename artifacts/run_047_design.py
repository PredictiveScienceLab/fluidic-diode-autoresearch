"""
Fluidic Diode Design & Continuous Optimization Module.
Run 047: Hammerhead Sting & Interleaving Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 025 winner (Di = 1.974761)
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9974176287651062, 
    0.05909641459584236, 0.22175829112529755, 0.561127245426178, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v3(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Advanced Catcher Topology with Hammerhead Sting:
    - Optimized nozzle ramps and re-entrant Borda wall-lips.
    - Trumpet-flared island with a deep concave rear face.
    - Multi-profile Sting: Seamlessly transitions from the Run 025 linear taper 
      to a 'Hammerhead' (flared tip) based on params[8].
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
    
    # params[8] logic: 
    # To maintain Run 025 compatibility (p8=0.059), we use a conditional profile.
    # If p8 < 0.15, use original linear taper. If p8 >= 0.15, allow Hammerhead flare.
    p8 = params[8]
    
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
        
        nose_part = jnp.sqrt(jnp.clip(idx / 0.4, 0.0, 1.0)) * (idx < 0.4)
        flare_part = (1.0 + isl_flare * (jnp.maximum(0.0, idx - 0.4)**3)) * (idx >= 0.4)
        profile = jnp.where(idx < 0.4, nose_part, flare_part)
        
        envelope = jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        half_w = (throat * 0.55) * profile * envelope
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # 4. Deep Concavity + Splitter Sting
        cav_start = 1.0 - 0.45 * concavity
        cav_idx = (idx - cav_start) / (1.0 - cav_start + 1e-6)
        cavity = (cav_idx > 0.0) & (ay < half_w * 0.85 * concavity)
        
        base_sting_width = (throat * 0.05) * sting_width_fac 
        t_prog = jnp.clip((cav_idx - 0.2) / 0.6, 0.0, 1.0)
        
        # Sting Profile: Linear Taper vs Hammerhead
        # If p8 is small (like 0.059), we match Run 025 exactly.
        alpha = jnp.where(p8 < 0.15, -2.0 * p8, -0.3)
        beta  = jnp.where(p8 < 0.15, 0.0, 5.0 * (p8 - 0.15))
        
        current_sting_width = base_sting_width * jnp.clip(1.0 + alpha * t_prog + beta * (t_prog**4), 0.01, 4.0)
        
        sting = (cav_idx > 0.2) & (cav_idx < 0.8) & (ay < current_sting_width)
        
        island_solid = (island_outer.astype(jnp.float32) * (1.0 - cavity.astype(jnp.float32)))
        island_solid = jnp.maximum(island_solid, sting.astype(jnp.float32))

        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_solid)

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v3(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v3(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    
    num_params = 12
    n_coarse = 96
    
    # Population Strategy
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS
    # Perturbations around current elite
    pop[1:32] = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.06, (31, num_params)), 0, 1)
    # Targeted search on the Hammerhead transition (params[8])
    pop[32:64] = PREV_BEST_PARAMS
    pop[32:64, 8] = rng.uniform(0.15, 0.75, 32)
    # Targeted search on stage interleaving (params[7] and params[10])
    pop[64:96] = PREV_BEST_PARAMS
    pop[64:96, 7] = rng.uniform(0.8, 1.0, 32)
    pop[64:96, 10] = rng.uniform(0.4, 0.7, 32)

    # Coarse screening on FAST_CONFIG
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Refinement Round 1 (DEFAULT_CONFIG)
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop_1 = pop[top_idx]
    best_coarse_p = pop[top_idx[0]]
    refine_pop_2 = jnp.clip(best_coarse_p + rng.normal(0, 0.025, (8, num_params)), 0, 1)
    refine_pop = jnp.concatenate([refine_pop_1, refine_pop_2], axis=0)
    
    masks_f1 = render_batch_fine(refine_pop)
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # Refinement Round 2 (Final Hill-Climb)
    best_f1_idx = np.argmax(np.asarray(scores_f1))
    best_f1_p = refine_pop[best_f1_idx]
    
    final_pop = jnp.tile(best_f1_p, (16, 1))
    final_pop = jnp.clip(final_pop + rng.normal(0, 0.009, (16, num_params)), 0, 1)
    final_pop = final_pop.at[0].set(best_f1_p) # Anchor
    
    masks_f2 = render_batch_fine(final_pop)
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_best_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = final_pop[final_best_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_best_idx])

if __name__ == "__main__":
    generate_design()
