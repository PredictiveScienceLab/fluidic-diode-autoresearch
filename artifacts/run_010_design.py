"""
Fluidic Diode Design & Continuous Optimization Module.
Run 010: Concave-Catcher Island & Tuned Hooked Borda Lips.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 007 winner (stdout)
# Di = 1.439455
# BEST_FINE_PARAMS: [0.6616491675376892, 0.7567344903945923, 0.4825185537338257, 0.17889510095119476, 0.5502895712852478, 0.011797428131103516, 0.8354988098144531, 0.6550413966178894, 0.7291443347930908, 0.3924931287765503, 0.9234058260917664, 0.8792598247528076]
PREV_BEST_PARAMS = np.array([
    0.6616491675376892, # params[0] - selector
    0.7567344903945923, # params[1] - throat
    0.4825185537338257, # params[2] - ramp_len
    0.17889510095119476, # params[3] - ramp_pow
    0.5502895712852478, # params[4] - lip_size
    0.011797428131103516, # params[5] - lip_hook_magnitude (old lip_hook)
    0.8354988098144531, # params[6] - splitter_width_factor
    0.6550413966178894, # params[7] - splitter_offset_x
    0.7291443347930908, # params[8] - splitter_asymmetry
    0.3924931287765503, # params[9] - concavity_factor
    0.9234058260917664, # params[10] - pitch_factor
    0.8792598247528076 # params[11] - was unused by borda_catcher, now lip_hook_curvature
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_sawtooth_v1_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    throat_gap = 0.28 + 0.36 * params[1]
    ramp_len = 0.35 + 0.55 * params[2]
    ramp_pow = 0.60 + 1.20 * params[3]
    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    pitch = 0.84 / n_stages
    margin_x = 0.08
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** ramp_pow)
        tooth = ((dx >= 0.0) & (dx <= ramp_len) & (abs_y >= y_wall)).astype(jnp.float32)
        mask = jnp.maximum(mask, tooth)
    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_borda_catcher_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Advanced Borda-Sawtooth with:
    1. Hooked (re-entrant) wall lips.
    2. Concave-back central islands (C-catchers).
    3. Parameterized stage pitch.
    New: Independent control for lip hooking magnitude and curvature.
    """
    throat = 0.15 + 0.35 * params[1]
    ramp_len = 0.25 + 0.45 * params[2]
    ramp_pow = 0.50 + 1.50 * params[3]
    lip_size = 0.05 + 0.15 * params[4]
    
    # params[5] now controls lip_hook_magnitude (linear recession)
    lip_hook_magnitude = 0.20 + 1.20 * params[5] 
    # params[11] now controls lip_hook_curvature (quadratic re-entrance)
    lip_hook_curvature = 0.50 + 2.00 * params[11] # NEW parameter for curvature control

    splitter_width_factor = params[6]
    splitter_offset_x = params[7]
    splitter_asymmetry = params[8]
    concavity_factor = params[9]
    pitch_factor = 0.25 + 0.10 * params[10]
    
    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.05
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch_factor
        dx = (x - x0) / pitch_factor
        
        # 1. Main Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Hooked Borda Lip (now with independent magnitude and curvature)
        lx = dx - ramp_len
        # y decreases then increases quadratically to create a pocket
        y_lip = throat - lip_hook_magnitude * lx + lip_hook_curvature * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.12) # 0.12 is a small thickness
        
        # 3. Concave-Back Splitter Island
        isl_x_start = ramp_len + splitter_offset_x * (1.0 - ramp_len - lip_size)
        isl_len = 0.12 + 0.15 * splitter_width_factor
        idx = (dx - isl_x_start) / isl_len
        
        # Teardrop profile (forward)
        alpha = 1.0 + 3.0 * (1.0 - splitter_asymmetry)
        beta = 1.0 + 3.0 * splitter_asymmetry
        prof = (jnp.maximum(0.0, idx)**alpha) * (jnp.maximum(0.0, 1.0 - idx)**beta)
        # Normalize
        peak = (alpha/(alpha+beta))**alpha * (beta/(alpha+beta))**beta
        prof = prof / peak
        
        half_w = throat * 0.45 * splitter_width_factor * prof
        island = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # Subtraction for concavity (C-catcher effect)
        # Subtract a smaller ellipse/parabola from the back (idx near 1.0)
        cavity_idx = (idx - (1.0 - 0.3 * concavity_factor)) / (0.3 * concavity_factor + 1e-6)
        cavity = (cavity_idx > 0.0) & (ay < half_w * 0.85 * concavity_factor)
        
        island_solid = island.astype(jnp.float32) * (1.0 - cavity.astype(jnp.float32))

        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_solid)

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_tesla_v3_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    gap = 0.18 + 0.22 * params[1]
    isz = 0.04 + 0.12 * params[4]
    wt = 0.05 + 0.06 * params[5]
    bx = 0.15 + 0.35 * params[10]
    by = 0.60 + 0.25 * params[11]
    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)
    n_stages = 2
    pitch = 1.0 / n_stages
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for i in range(n_stages):
        xl = (x - i * pitch) / pitch
        side_v = (jnp.abs(xl - bx) < wt/2) & (ay > gap) & (ay < by)
        side_h = (xl > bx) & (xl < bx + 0.35) & (jnp.abs(ay - by) < wt/2)
        m = (by - gap) / 0.35
        fin_mask = (xl > bx) & (xl < bx + 0.35) & (ay < by - m*(xl-bx) + wt) & (ay > by - m*(xl-bx))
        island = (xl > bx - isz) & (xl < bx + isz) & (ay < gap)
        mask = jnp.maximum(mask, side_v.astype(jnp.float32))
        mask = jnp.maximum(mask, side_h.astype(jnp.float32))
        mask = jnp.maximum(mask, fin_mask.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    selector = params[0]
    return jax.lax.cond(
        selector < 0.3,
        lambda p: render_sawtooth_v1_mapped(p, nx_design, ny),
        lambda p: jax.lax.cond(
            selector < 0.85,
            lambda pp: render_borda_catcher_mapped(pp, nx_design, ny),
            lambda pp: render_tesla_v3_mapped(pp, nx_design, ny),
            p
        ),
        params
    )

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    
    num_params = 12
    # Ensure candidate 0 (PREV_BEST_PARAMS) exactly reproduces the previous best mask
    # by calculating the equivalent new params[11] for the new lip_hook_curvature.
    # Old: lip_hook = 0.20 + 1.20 * params[5], hook_curve = 2.5 * lip_hook
    # New: lip_hook_magnitude = 0.20 + 1.20 * params[5], lip_hook_curvature = 0.5 + 2.0 * params[11]
    # To match: lip_hook_magnitude = old_lip_hook, lip_hook_curvature = old_hook_curve
    # So, (0.5 + 2.0 * new_params[11]) = 2.5 * (0.20 + 1.20 * PREV_BEST_PARAMS[5])
    # new_params[11] = (2.5 * (0.20 + 1.20 * PREV_BEST_PARAMS[5]) - 0.5) / 2.0
    
    # Calculate equivalent params[11] for the previous best design
    old_lip_hook_val = 0.20 + 1.20 * PREV_BEST_PARAMS[5]
    new_params_11_equivalent = (2.5 * old_lip_hook_val - 0.5) / 2.0
    
    current_best_params = PREV_BEST_PARAMS.copy()
    current_best_params[11] = new_params_11_equivalent # Update params[11] for exact reproduction

    # 1. Global Screening (FAST_CONFIG)
    n_coarse = 64
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    pop[0] = current_best_params # Seed with previous winner, adjusted for new parameter
    
    # Perturb seed slightly to explore the immediate neighborhood in coarse mode
    pop[1:8] = jnp.clip(current_best_params + rng.normal(0, 0.05, (7, num_params)), 0, 1)
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Refinement Rounds (DEFAULT_CONFIG)
    # Round 1: Top 8 from coarse
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    elite_pop = pop[top_idx]
    
    masks_f1 = render_batch_fine(jnp.asarray(elite_pop))
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # Round 2: Local Search around the best of Round 1
    best_idx = np.argmax(np.asarray(scores_f1))
    best_p = elite_pop[best_idx]
    
    # 1+7 Evolution Strategy-like step
    local_pop = np.tile(best_p, (8, 1))
    local_pop[1:] = np.clip(best_p + rng.normal(0, 0.025, (7, num_params)), 0, 1)
    
    masks_f2 = render_batch_fine(jnp.asarray(local_pop))
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = local_pop[final_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_idx])

if __name__ == "__main__":
    design = generate_design()
