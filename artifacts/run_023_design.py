"""
Fluidic Diode Design & Continuous Optimization Module.
Run 023: Asymmetric Central Island (Vertical Offset).
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 021 winner (Di = 1.945482)
# PREV_BEST_PARAMS now includes a neutral value (0.5) for params[12] (isl_offset_y_factor).
PREV_BEST_PARAMS = np.array([
    0.8434114456176758, 0.7999433279037476, 0.015717409551143646, 0.7680162787437439,
    0.7221002578735352, 0.2040959894657135, 0.11078418046236038, 0.9989439845085144,
    0.023696105927228928, 0.1080978587269783, 0.5505483150482178, 0.17034919559955597,
    0.5  # NEW: Neutral value for params[12] (isl_offset_y_factor) to maintain previous design
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_trumpet_catcher_v2(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Advanced Catcher Topology (v2) with Asymmetric Island:
    - Nozzle walls for forward streamlining.
    - Re-entrant Borda wall-lips for reverse flow capture.
    - Central island with a 'Trumpet' flare and a deep concave rear face.
    - Features a central splitter 'sting' with its width controlled by params[0]
      and its taper controlled by params[8].
    - Introduces a vertical offset for the central island, controlled by params[12].
    """
    # Mapping parameters (13 dims)
    sting_width_fac = 0.1 + 1.5 * params[0] # Scale for the base sting width
    throat          = 0.12 + 0.30 * params[1]  # Channel constriction
    ramp_len        = 0.15 + 0.40 * params[2]  # Length of nozzle ramp
    ramp_pow        = 0.40 + 1.60 * params[3]  # Curvature of nozzle ramp
    lip_size        = 0.04 + 0.16 * params[4]  # Length of the re-entrant lip
    lip_hook        = 0.05 + 1.45 * params[5]  # Curvature/depth of the lip hook

    isl_scale_x     = 0.15 + 0.30 * params[6]  # Island length scale
    isl_pos_x       = 0.02 + 0.20 * params[7]  # Island x-position relative to stage
    sting_taper_factor = 0.0 + 1.0 * params[8] # Taper magnitude for the central sting (0.0=no taper, 1.0=full taper to a point)
    concavity       = 0.30 + 0.65 * params[9]  # Depth of the island's rear concavity
    pitch           = 0.28 + 0.12 * params[10] # Spacing between stages
    isl_flare       = 0.40 + 0.80 * params[11] # Intensity of the rear trumpet flare
    isl_offset_y_factor = params[12]           # NEW: Factor for vertical island offset (0.0-1.0)

    # Calculate actual vertical offset for the island
    # params[12] = 0.5 -> 0 offset. params[12] = 0.0 -> -0.15 offset. params[12] = 1.0 -> +0.15 offset.
    # Offset range from -0.15 to +0.15 (relative to half-channel height, i.e., total channel height is 2 units)
    isl_y_offset = (isl_offset_y_factor - 0.5) * 0.3 

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y) # For wall geometry, which remains symmetric

    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle Ramp (symmetric)
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Hooked Borda Lip (symmetric)
        lx = dx - ramp_len
        hook_curve = 3.5 * lip_hook 
        y_lip = throat - lip_hook * lx + hook_curve * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        # 3. Trumpet-Flared Island (with vertical offset)
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        
        # Profile: Streamlined nose (front) transitions to aggressive flare (back)
        nose_part = jnp.sqrt(jnp.clip(idx / 0.4, 0.0, 1.0)) * (idx < 0.4)
        flare_part = (1.0 + isl_flare * (jnp.maximum(0.0, idx - 0.4)**3)) * (idx >= 0.4)
        profile = jnp.where(idx < 0.4, nose_part, flare_part)
        
        # Taper the very ends to ensure a closed shape
        envelope = jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        half_w = (throat * 0.55) * profile * envelope
        
        # Apply offset to the y-coordinate for the island and its features
        y_island_coord = jnp.abs(y - isl_y_offset) # Use absolute value relative to the offset center line
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (y_island_coord < half_w)
        
        # Deep Concavity + Splitter Sting (also with vertical offset)
        cav_start = 1.0 - 0.45 * concavity
        cav_idx = (idx - cav_start) / (1.0 - cav_start + 1e-6)
        
        # The cavity itself.
        cavity = (cav_idx > 0.0) & (y_island_coord < half_w * 0.85 * concavity)
        
        # The 'Sting' - a small central spike inside the cavity to split reverse flow
        base_sting_width = (throat * 0.05) * sting_width_fac 
        
        # Define sting's extent within the cavity index
        sting_cav_start = 0.2
        sting_cav_end = 0.8
        
        # Linear taper from max_width at sting_cav_start to 0 at sting_cav_end (if sting_taper_factor=1)
        taper_progress = (cav_idx - sting_cav_start) / (sting_cav_end - sting_cav_start + 1e-6)
        current_sting_width = base_sting_width * jnp.clip(1.0 - sting_taper_factor * taper_progress, 0.0, 1.0)
        
        sting = (cav_idx > sting_cav_start) & (cav_idx < sting_cav_end) & (y_island_coord < current_sting_width)
        
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
    
    num_params = 13 # Updated number of parameters to include isl_offset_y_factor
    
    # 1. Global Screening (FAST_CONFIG) - ~12s
    n_coarse = 64
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    
    # Candidate 0: The previous best design.
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
