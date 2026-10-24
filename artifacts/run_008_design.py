"""
Fluidic Diode Design & Continuous Optimization Module.
Run 008: Deep-Cup Tesla-Borda Hybrid Refinement.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 007 winner (Di = 1.439455)
PREV_BEST_PARAMS = np.array([
    0.66164917, 0.75673449, 0.48251855, 0.17889510, 0.55028957,
    0.01179743, 0.83549881, 0.65504140, 0.72914433, 0.39249313,
    0.92340583, 0.87925982
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
    2. Deep Concave-back central islands (C-catchers).
    3. Parameterized stage pitch and island sharpening.
    """
    throat = 0.15 + 0.35 * params[1]
    ramp_len = 0.20 + 0.50 * params[2]
    ramp_pow = 0.40 + 1.60 * params[3]
    lip_size = 0.05 + 0.20 * params[4]
    lip_hook = 0.10 + 1.30 * params[5] 

    splitter_width_factor = 0.20 + 0.80 * params[6]
    splitter_offset_x = params[7]
    splitter_asymmetry = params[8] # Controls nose vs tail sharpness
    concavity_depth = 0.20 + 0.70 * params[9] 
    pitch_factor = 0.20 + 0.20 * params[10] 
    
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
        
        # 2. Hooked Borda Lip
        lx = dx - ramp_len
        hook_curve = 3.0 * lip_hook 
        y_lip = throat - lip_hook * lx + hook_curve * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.15)
        
        # 3. Concave-Back Splitter Island
        isl_x_start = ramp_len + splitter_offset_x * (1.0 - ramp_len - lip_size)
        isl_len = 0.10 + 0.25 * splitter_width_factor
        idx = (dx - isl_x_start) / isl_len
        
        # Asymmetric Profile (Forward-streamlined)
        alpha = 0.5 + 4.0 * (1.0 - splitter_asymmetry)
        beta = 0.5 + 4.0 * splitter_asymmetry
        prof = (jnp.maximum(0.0, idx)**alpha) * (jnp.maximum(0.0, 1.0 - idx)**beta)
        peak = (alpha/(alpha+beta))**alpha * (beta/(alpha+beta))**beta
        prof = prof / (peak + 1e-6)
        
        half_w = throat * 0.60 * splitter_width_factor * prof
        island = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # Deep Concavity (C-catcher effect)
        # We subtract an ellipse from the back
        cavity_x = (idx - 1.0) / (concavity_depth + 1e-6)
        cavity_y = ay / (half_w * 0.9 + 1e-6)
        cavity = (cavity_x**2 + cavity_y**2 < 1.0) & (idx > 1.0 - concavity_depth)
        
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
        selector < 0.2,
        lambda p: render_sawtooth_v1_mapped(p, nx_design, ny),
        lambda p: jax.lax.cond(
            selector < 0.90,
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
    # 1. Global Screening (FAST_CONFIG) - Larger batch
    n_coarse = 128
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS
    # Focus search around the successful catcher topology
    pop[1:32] = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.1, (31, num_params)), 0, 1)
    pop[1:32, 0] = jnp.clip(PREV_BEST_PARAMS[0] + rng.normal(0, 0.05, (31,)), 0.3, 0.85)
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Refinement Round 1 (DEFAULT_CONFIG)
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:12]
    elite_pop = pop[top_idx]
    
    masks_f1 = render_batch_fine(jnp.asarray(elite_pop))
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # 3. Refinement Round 2 (Local Search around the best of Round 1)
    best_idx = np.argmax(np.asarray(scores_f1))
    best_p = elite_pop[best_idx]
    
    local_pop = np.tile(best_p, (12, 1))
    # Small perturbations to optimize details
    local_pop[1:] = np.clip(best_p + rng.normal(0, 0.03, (11, num_params)), 0, 1)
    # Ensure we don't switch topology by accident during local search
    local_pop[1:, 0] = best_p[0] 
    
    masks_f2 = render_batch_fine(jnp.asarray(local_pop))
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = local_pop[final_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_idx])

if __name__ == "__main__":
    design = generate_design()
