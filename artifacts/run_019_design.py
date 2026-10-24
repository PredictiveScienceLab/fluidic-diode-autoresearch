"""
Fluidic Diode Design & Continuous Optimization Module.
Run 019: Trumpet-Flared Catcher Island with Variable Splitter Sting Width and Island Asymmetry.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 014 winner (Di = 1.860782)
PREV_BEST_PARAMS = np.array([
    0.847564160823822, 0.7750893235206604, 0.0, 0.7833242416381836, 0.7225937843322754,
    0.2202993482351303, 0.11345402896404266, 0.9863949418067932, 0.014233815483748913,
    0.1147463470697403, 0.566098690032959, 0.1662263721227646
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_trumpet_catcher_v2(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Advanced Catcher Topology (v2):
    - Nozzle walls for forward streamlining.
    - Re-entrant Borda wall-lips for reverse flow capture.
    - Central island with a 'Trumpet' flare and a deep concave rear face.
    - Features a central splitter 'sting' with its width controlled by params[0].
    - Introduces vertical asymmetry to the central islands via params[8].
    """
    # Mapping parameters (12 dims)
    sting_width_fac = 0.1 + 1.5 * params[0] # Scaling factor for central sting width
    throat          = 0.12 + 0.30 * params[1]  # Channel constriction
    ramp_len        = 0.15 + 0.40 * params[2]  # Length of nozzle ramp
    ramp_pow        = 0.40 + 1.60 * params[3]  # Curvature of nozzle ramp
    lip_size        = 0.04 + 0.16 * params[4]  # Length of the re-entrant lip
    lip_hook        = 0.05 + 1.45 * params[5]  # Curvature/depth of the lip hook

    isl_scale_x     = 0.15 + 0.30 * params[6]  # Island length scale
    isl_pos_x       = 0.02 + 0.20 * params[7]  # Island x-position relative to stage
    
    # params[8] now controls island asymmetry (vertical offset)
    # Maps [0,1] to [-0.3, 0.3] for y-offset relative to the channel height
    island_offset_factor = -0.3 + 0.6 * params[8] 
    
    concavity       = 0.30 + 0.65 * params[9]  # Depth of the island's rear concavity
    pitch           = 0.28 + 0.12 * params[10] # Spacing between stages
    isl_flare       = 0.40 + 0.80 * params[11] # Intensity of the rear trumpet flare

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    # Absolute y for symmetric wall features
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle Ramp (symmetric about y=0)
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Hooked Borda Lip (symmetric about y=0)
        lx = dx - ramp_len
        hook_curve = 3.5 * lip_hook 
        y_lip = throat - lip_hook * lx + hook_curve * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        # For island calculations, apply the vertical offset
        y_for_island = y - island_offset_factor
        ay_for_island = jnp.abs(y_for_island)

        # 3. Trumpet-Flared Island (now potentially asymmetric)
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        
        # Profile: Streamlined nose (front) transitions to aggressive flare (back)
        nose_part = jnp.sqrt(jnp.clip(idx / 0.4, 0.0, 1.0)) * (idx < 0.4)
        flare_part = (1.0 + isl_flare * (jnp.maximum(0.0, idx - 0.4)**3)) * (idx >= 0.4)
        profile = jnp.where(idx < 0.4, nose_part, flare_part)
        
        # Taper the very ends to ensure a closed shape
        envelope = jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        half_w = (throat * 0.55) * profile * envelope
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay_for_island < half_w)
        
        # Deep Concavity + Splitter Sting
        cav_start = 1.0 - 0.45 * concavity
        cav_idx = (idx - cav_start) / (1.0 - cav_start + 1e-6)
        
        # The cavity itself.
        cavity = (cav_idx > 0.0) & (ay_for_island < half_w * 0.85 * concavity)
        
        # The 'Sting' - a small central spike inside the cavity to split reverse flow
        sting_max_width = (throat * 0.05) * sting_width_fac 
        sting = (cav_idx > 0.2) & (cav_idx < 0.8) & (ay_for_island < sting_max_width)
        
        island_solid = (island_outer.astype(jnp.float32) * (1.0 - cavity.astype(jnp.float32)))
        island_solid = jnp.maximum(island_solid, sting.astype(jnp.float32))

        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_solid)

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    return render_trumpet_catcher_v2(params, nx_design, ny)

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    
    num_params = 12
    # 1. Global Screening (FAST_CONFIG) - ~12s
    n_coarse = 64
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    
    # Candidate 0: The previous best design from run_014
    # Its params[8] value (0.0142...) will now slightly shift the island downwards.
    pop[0] = PREV_BEST_PARAMS 
    
    # Local cluster around best
    pop[1:16] = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.08, (15, num_params)), 0, 1)
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Refinement Round 1 (DEFAULT_CONFIG) - ~18s
    # Take top 8 from coarse
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop_1 = pop[top_idx]
    
    # Add 8 small perturbations of the absolute best
    best_coarse_p = pop[top_idx[0]]
    refine_pop_2 = jnp.clip(best_coarse_p + rng.normal(0, 0.03, (8, num_params)), 0, 1)
    
    refine_pop = jnp.concatenate([refine_pop_1, refine_pop_2], axis=0)
    
    masks_f1 = render_batch_fine(refine_pop)
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # 3. Refinement Round 2 (Final Hill-Climb) - ~18s
    best_f1_idx = np.argmax(np.asarray(scores_f1))
    best_f1_p = refine_pop[best_f1_idx]
    
    final_pop = jnp.tile(best_f1_p, (16, 1))
    final_pop = jnp.clip(final_pop + rng.normal(0, 0.012, (16, num_params)), 0, 1)
    final_pop = final_pop.at[0].set(best_f1_p) # Keep parent
    
    masks_f2 = render_batch_fine(final_pop)
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_best_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = final_pop[final_best_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_best_idx])

if __name__ == "__main__":
    design = generate_design()
