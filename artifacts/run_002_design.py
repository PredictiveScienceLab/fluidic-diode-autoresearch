"""
Fluidic Diode Design & Continuous Optimization Module.
Run 002: Multi-fidelity search over a unified Sawtooth and Tesla-inspired topology space.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Previous best from run_000 baseline
PREV_BEST_PARAMS = [0.7611396908760071, 0.7860643267631531, 0.12811362743377686]

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_sawtooth(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Original sawtooth baseline renderer."""
    throat_gap = 0.28 + 0.36 * params[0]
    ramp_len = 0.35 + 0.55 * params[1]
    ramp_pow = 0.60 + 1.20 * params[2]

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
def render_tesla(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Tesla-inspired Valvular Conduit.
    Params: [throat_y, loop_y_offset, splitter_x, return_slope, wall_thick]
    """
    ty = 0.12 + 0.25 * params[0]     # Main channel half-width
    ly = ty + 0.18 + 0.35 * params[1] # Side loop Y position
    sx = 0.15 + 0.55 * params[2]    # Splitter start X
    rs = 0.80 + 3.20 * params[3]    # Return fin slope (steepness)
    th = 0.05 + 0.08 * params[4]    # Wall thickness

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 2
    pitch = 1.0 / n_stages
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)

    for i in range(n_stages):
        xl = (x - i * pitch) / pitch
        
        # 1. Outer side-loop wall
        l_wall = (xl > 0.1) & (xl < 0.9) & (ay > ly) & (ay < ly + th)
        
        # 2. Central splitter island (blocks the easy path)
        island = (xl > sx) & (xl < sx + 0.18) & (ay < ty)
        
        # 3. Re-entry 'diode' fin
        # For +x (forward), this is a guiding ramp.
        # For -x (reverse), this catches the flow and forces it into the side loop.
        x_ret = sx - 0.04
        # Linear ramp: ay = ty + slope * (x_ret - xl)
        fin_bound = ty + rs * (x_ret - xl)
        fin = (xl > x_ret - 0.22) & (xl < x_ret) & (ay > ty) & (ay < ly) & (ay < fin_bound)
        
        mask = jnp.maximum(mask, l_wall.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
        mask = jnp.maximum(mask, fin.astype(jnp.float32))

    # Channel boundary walls
    mask = jnp.where(ay > 0.97, 1.0, mask)
    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Unified parameter space for Sawtooth (selector < 0.4) and Tesla (selector >= 0.4)."""
    return jax.lax.cond(
        params[0] < 0.4,
        lambda p: render_sawtooth(p[1:4], nx_design, ny),
        lambda p: render_tesla(p[1:6], nx_design, ny),
        params
    )

render_batch_coarse = jax.jit(
    jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny))
)
render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)

def generate_design() -> np.ndarray:
    t0 = time.time()
    np.set_printoptions(precision=8, suppress=True)
    rng = np.random.default_rng(44)

    # Search space: 128 coarse candidates, 6 parameters [type, p1, p2, p3, p4, p5]
    pop = rng.uniform(0.0, 1.0, size=(128, 6)).astype(np.float32)
    
    # Invariant: Candidate 0 is the current best sawtooth from run_000
    pop[0] = np.array([0.1, PREV_BEST_PARAMS[0], PREV_BEST_PARAMS[1], PREV_BEST_PARAMS[2], 0.0, 0.0])

    # Phase 1: FAST_CONFIG Screening
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    scores_c_np = np.asarray(scores_c)

    # Phase 2: DEFAULT_CONFIG Verification (Top 12)
    top_idx = np.argsort(scores_c_np)[::-1][:12]
    pop_f1 = pop[top_idx]
    masks_f1 = render_batch_fine(jnp.asarray(pop_f1))
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    scores_f1_np = np.asarray(scores_f1)

    # Phase 3: Local Coordinate Refinement on DEFAULT_CONFIG
    best_idx_f1 = int(np.argmax(scores_f1_np))
    best_p = pop_f1[best_idx_f1]
    
    # Generate 11 neighbors with small perturbations
    neighbors = best_p + rng.normal(0, 0.04, size=(11, 6))
    pop_f2 = np.vstack([best_p[None, :], np.clip(neighbors, 0.0, 1.0)])
    
    masks_f2 = render_batch_fine(jnp.asarray(pop_f2))
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    scores_f2_np = np.asarray(scores_f2)

    best_idx_final = int(np.argmax(scores_f2_np))
    best_fine_params = pop_f2[best_idx_final]
    
    print(f"[design.py] Finished in {time.time()-t0:.1f}s | Best Fine Di = {scores_f2_np[best_idx_final]:.6f}")
    print("BEST_FINE_PARAMS:", best_fine_params.tolist())
    
    return np.asarray(masks_f2[best_idx_final], dtype=np.float32)

if __name__ == "__main__":
    # Test generation
    mask = generate_design()
    print("Mask shape:", mask.shape)
