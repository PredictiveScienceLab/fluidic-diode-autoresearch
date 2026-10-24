"""
Fluidic Diode Design & Continuous Optimization Module.
Run 003: Enhanced Sawtooth with Re-entrant Borda Lips and Tesla-style bifurcations.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 002 winner (which was a sawtooth)
# BEST_FINE_PARAMS: [0.0381664515824384, 0.07187846212967341, 0.21747368526033634, 0.360058201135526, 0.5675521640136428, 0.9274789735824693]
PREV_BEST_PARAMS = [0.03816645, 0.07187846, 0.21747369, 0.36005820, 0.56755216, 0.92747897, 0.0]

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_sawtooth_v1(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """The original sawtooth baseline renderer (Run 000/002)."""
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
def render_borda_sawtooth(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Enhanced sawtooth with re-entrant lips to catch reverse flow."""
    throat = 0.20 + 0.35 * params[0]
    ramp_len = 0.30 + 0.50 * params[1]
    ramp_pow = 0.60 + 1.40 * params[2]
    lip_size = 0.04 + 0.12 * params[3]
    lip_ang = 0.40 + 1.80 * params[4]
    
    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    pitch = 0.90 / n_stages
    mx = 0.05

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # Main nozzle ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # Re-entrant lip (catcher)
        lx = dx - ramp_len
        lip = (lx > 0.0) & (lx < lip_size) & (ay > throat - lip_ang * lx) & (ay < throat + 0.08)
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_tesla_v3(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Improved Tesla-style valve with cleaner geometry."""
    gap = 0.18 + 0.22 * params[0]
    bx = 0.15 + 0.35 * params[1]
    by = 0.60 + 0.25 * params[2]
    isz = 0.04 + 0.12 * params[3]
    wt = 0.05 + 0.06 * params[4]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)
    
    n_stages = 2
    pitch = 1.0 / n_stages
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for i in range(n_stages):
        xl = (x - i * pitch) / pitch
        
        # 1. Side-channel walls
        side_v = (jnp.abs(xl - bx) < wt/2) & (ay > gap) & (ay < by)
        side_h = (xl > bx) & (xl < bx + 0.35) & (jnp.abs(ay - by) < wt/2)
        
        # 2. Return fin (The "bucket")
        m = (by - gap) / 0.35
        fin_mask = (xl > bx) & (xl < bx + 0.35) & (ay < by - m*(xl-bx) + wt) & (ay > by - m*(xl-bx))

        # 3. Central island
        island = (xl > bx - isz) & (xl < bx + isz) & (ay < gap)
        
        mask = jnp.maximum(mask, side_v.astype(jnp.float32))
        mask = jnp.maximum(mask, side_h.astype(jnp.float32))
        mask = jnp.maximum(mask, fin_mask.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Unified space: [selector, p1, p2, p3, p4, p5, p6]"""
    selector = params[0]
    
    mask = jax.lax.cond(
        selector < 0.35,
        lambda p: render_sawtooth_v1(p[1:4], nx_design, ny),
        lambda p: jax.lax.cond(
            p[0] < 0.75,
            lambda pp: render_bota_sawtooth(pp[1:6], nx_design, ny) if 'render_bota_sawtooth' in globals() else render_borda_sawtooth(pp[1:6], nx_design, ny),
            lambda pp: render_tesla_v3(pp[1:6], nx_design, ny),
            p
        ),
        params
    )
    
    # Boundary wall enforcement
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    mask = jnp.where(jnp.abs(y) > 0.98, 1.0, mask)
    return mask

# Handle the typo in the cond lambda if it occurs
def render_bota_sawtooth(p, nx, ny): return render_borda_sawtooth(p, nx, ny)

render_batch_coarse = jax.jit(
    jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny))
)
render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)

def generate_design() -> np.ndarray:
    t0 = time.time()
    np.set_printoptions(precision=8, suppress=True)
    rng = np.random.default_rng(42)

    # 1. Global Exploration on FAST_CONFIG
    pop = rng.uniform(0.0, 1.0, size=(128, 7)).astype(np.float32)
    pop[0] = np.array(PREV_BEST_PARAMS) # Strictly maintain previous best

    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Elite Selection and Verification on DEFAULT_CONFIG
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:12]
    pop_f1 = pop[top_idx]
    masks_f1 = render_batch_fine(jnp.asarray(pop_f1))
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    scores_f1_np = np.asarray(scores_f1)

    # 3. Local Refinement on DEFAULT_CONFIG
    best_idx_f1 = int(np.argmax(scores_f1_np))
    best_p = pop_f1[best_idx_f1]
    
    # Small perturbations around the winner
    neighbors = best_p + rng.normal(0, 0.03, size=(11, 7))
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
    mask = generate_design()
    print("Mask shape:", mask.shape)
