"""
Fluidic Diode Design & Continuous Optimization Module.
Run 011: Flared Concave-Catcher Island & Synchronized Borda Lips.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 007/010 winner (Di = 1.439455)
PREV_BEST_PARAMS = np.array([
    0.66164917, 0.75673449, 0.48251855, 0.17889510, 0.55028957,
    0.01179743, 0.83549881, 0.65504140, 0.72914433, 0.39249313,
    0.92340583, 0.87925982
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_flared_catcher_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Refined Borda-Catcher Topology:
    - Converging nozzles for forward flow.
    - Re-entrant (hooked) wall lips to trap reverse fluid.
    - Central island with a streamlined nose and a flared, concave rear "parachute".
    """
    throat = 0.15 + 0.30 * params[1]
    ramp_len = 0.20 + 0.45 * params[2]
    ramp_pow = 0.50 + 1.50 * params[3]
    lip_size = 0.05 + 0.15 * params[4]
    lip_hook = 0.10 + 1.40 * params[5] 

    isl_scale_x = 0.10 + 0.25 * params[6]
    isl_pos_x = 0.05 + 0.25 * params[7] # Relative to stage start
    isl_asym = params[8] # Nose vs Tail bias
    concavity_depth = 0.20 + 0.70 * params[9]
    pitch = 0.28 + 0.12 * params[10]
    isl_flare = 0.30 + 0.60 * params[11] # How much the back of the island widens

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle Ramp (Forward streamlining)
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Hooked Borda Lip (Reverse trapping)
        lx = dx - ramp_len
        hook_curve = 3.0 * lip_hook 
        y_lip = throat - lip_hook * lx + hook_curve * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.10)
        
        # 3. Flared Concave Island
        # Positioning: starts after the nozzle ramp
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        
        # Forward nose (streamlined)
        alpha = 1.0 + 4.0 * (1.0 - isl_asym)
        beta = 1.0 + 2.0 * isl_asym
        nose_prof = (jnp.maximum(0.0, idx)**alpha) * (jnp.maximum(0.0, 1.0 - idx)**beta)
        peak = (alpha/(alpha+beta))**alpha * (beta/(alpha+beta))**beta
        nose_prof = nose_prof / (peak + 1e-6)
        
        # Flared rear profile (Parachute effect)
        # Flare increases the width towards the back (idx -> 1.0)
        flare_w = throat * 0.5 * (1.0 + isl_flare * (jnp.maximum(0.0, idx - 0.4)**2))
        half_w = flare_w * nose_prof
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # Concavity subtraction (C-catcher)
        cav_idx = (idx - (1.0 - 0.4 * concavity_depth)) / (0.4 * concavity_depth + 1e-6)
        cavity = (cav_idx > 0.0) & (ay < half_w * 0.9 * concavity_depth)
        
        island_solid = island_outer.astype(jnp.float32) * (1.0 - cavity.astype(jnp.float32))

        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_solid)

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Focus purely on the high-performing Catcher topology
    return render_flared_catcher_mapped(params, nx_design, ny)

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    
    num_params = 12
    # 1. Global Screening (FAST_CONFIG) - Budget ~15s
    n_coarse = 64
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS 
    pop[1:12] = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.06, (11, num_params)), 0, 1)
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Refinement Rounds (DEFAULT_CONFIG) - Budget ~35s
    # Round 1: Elites from coarse + new local variations
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:6]
    elite_pop = pop[top_idx]
    
    # Augment elite_pop with local perturbations
    refine_pop = np.zeros((12, num_params), dtype=np.float32)
    refine_pop[:6] = elite_pop
    refine_pop[6:] = jnp.clip(elite_pop[0] + rng.normal(0, 0.03, (6, num_params)), 0, 1)
    
    masks_f1 = render_batch_fine(jnp.asarray(refine_pop))
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # Round 2: Final Hill-Climb around the best of Round 1
    best_idx = np.argmax(np.asarray(scores_f1))
    best_p = refine_pop[best_idx]
    
    final_pop = np.tile(best_p, (12, 1))
    final_pop[1:] = jnp.clip(best_p + rng.normal(0, 0.015, (11, num_params)), 0, 1)
    final_pop[0] = best_p # Anchor
    
    masks_f2 = render_batch_fine(jnp.asarray(final_pop))
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_best_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = final_pop[final_best_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_best_idx])

if __name__ == "__main__":
    design = generate_design()
