"""
Fluidic Diode Design & Continuous Optimization Module.
Run 043: Targeted Exploration of Concavity, Lip Hook, and Island Position.
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
def render_trumpet_catcher_v2(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Advanced Catcher Topology (v2):
    - Nozzle walls for forward streamlining.
    - Re-entrant Borda wall-lips for reverse flow capture.
    - Central island with a 'Trumpet' flare and a deep concave rear face.
    - Features a central splitter 'sting' with its width controlled by params[0]
      and its taper controlled by params[8].
    """
    # Mapping parameters (12 dims)
    sting_width_fac = 0.1 + 1.5 * params[0] # Scale for the base sting width
    throat          = 0.12 + 0.30 * params[1]  # Channel constriction
    ramp_len        = 0.15 + 0.40 * params[2]  # Length of nozzle ramp
    ramp_pow        = 0.40 + 1.60 * params[3]  # Curvature of nozzle ramp
    lip_size        = 0.04 + 0.16 * params[4]  # Length of the re-entrant lip
    lip_hook        = 0.05 + 1.45 * params[5]  # Curvature/depth of the lip hook

    isl_scale_x     = 0.15 + 0.30 * params[6]  # Island length scale
    isl_pos_x       = 0.02 + 0.20 * params[7]  # Island x-position relative to stage
    sting_taper_factor = 0.0 + 2.0 * params[8] # Taper magnitude for the central sting (0.0=no taper, 2.0=full taper to a point)
    concavity       = 0.30 + 0.65 * params[9]  # Depth of the island's rear concavity
    pitch           = 0.28 + 0.12 * params[10] # Spacing between stages
    isl_flare       = 0.40 + 0.80 * params[11] # Intensity of the rear trumpet flare

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
        
        # The cavity itself.
        cavity = (cav_idx > 0.0) & (ay < half_w * 0.85 * concavity)
        
        # The 'Sting' - a small central spike inside the cavity to split reverse flow
        # Its width is now scaled by sting_width_fac and tapered by sting_taper_factor
        base_sting_width = (throat * 0.05) * sting_width_fac 
        
        # Define sting's extent within the cavity index
        sting_cav_start = 0.2
        sting_cav_end = 0.8
        
        # Linear taper from max_width at sting_cav_start to 0 at sting_cav_end (if sting_taper_factor=1)
        taper_progress = (cav_idx - sting_cav_start) / (sting_cav_end - sting_cav_start + 1e-6)
        current_sting_width = base_sting_width * jnp.clip(1.0 - sting_taper_factor * taper_progress, 0.0, 1.0)
        
        sting = (cav_idx > sting_cav_start) & (cav_idx < sting_cav_end) & (ay < current_sting_width)
        
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
    n_coarse = 64 # Number of candidates for coarse-grain screening
    
    # Initialize population with random values as a base for variety
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    
    # Strategy 1: Candidate 0 is always the previous best design (ensures monotonic improvement)
    pop[0] = PREV_BEST_PARAMS 
    
    # Strategy 2: Small perturbations around the previous best (15 candidates)
    pop[1:16] = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.08, (15, num_params)), 0, 1)
    
    # Strategy 3: Targeted exploration for specific parameters (10 candidates)
    # Focus on increasing lip_hook (params[5]), concavity (params[9]), and reducing isl_pos_x (params[7])
    for i in range(16, 26): # This will be candidates 16 to 25
        p_targeted = PREV_BEST_PARAMS.copy()
        
        # Increase params[5] (lip_hook) - bias towards higher values
        p_targeted[5] = np.clip(rng.uniform(0.4, 1.0), 0.0, 1.0) 
        # Decrease params[7] (isl_pos_x) - bias towards lower values
        p_targeted[7] = np.clip(rng.uniform(0.0, 0.6), 0.0, 1.0) 
        # Increase params[9] (concavity) - bias towards higher values
        p_targeted[9] = np.clip(rng.uniform(0.4, 1.0), 0.0, 1.0) 
        
        # Perturb other parameters slightly around the original best
        pop[i] = jnp.clip(p_targeted + rng.normal(0, 0.05, num_params), 0, 1)

    # Strategy 4: The remaining candidates (38) are left as random, ensuring global exploration.

    # Coarse-grain evaluation on FAST_CONFIG
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Refinement Round 1 (DEFAULT_CONFIG)
    # Take top 8 from coarse, plus 8 perturbations of the best coarse candidate
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop_1 = pop[top_idx]
    
    best_coarse_p = pop[top_idx[0]]
    refine_pop_2 = jnp.clip(best_coarse_p + rng.normal(0, 0.03, (8, num_params)), 0, 1)
    
    refine_pop = jnp.concatenate([refine_pop_1, refine_pop_2], axis=0) # Total 16 candidates
    
    masks_f1 = render_batch_fine(refine_pop)
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # Refinement Round 2 (Final Hill-Climb)
    # Take the best from Round 1 and generate 15 more small perturbations
    best_f1_idx = np.argmax(np.asarray(scores_f1))
    best_f1_p = refine_pop[best_f1_idx]
    
    final_pop = jnp.tile(best_f1_p, (16, 1)) # 16 candidates
    final_pop = jnp.clip(final_pop + rng.normal(0, 0.012, (16, num_params)), 0, 1)
    final_pop = final_pop.at[0].set(best_f1_p) # Ensure the parent is always included
    
    masks_f2 = render_batch_fine(final_pop)
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_best_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = final_pop[final_best_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_best_idx])

if __name__ == "__main__":
    design = generate_design()
