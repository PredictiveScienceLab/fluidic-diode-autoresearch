"""
Fluidic Diode Design & Continuous Optimization Module.
Run 004: Bridge-Tesla Topology & Multi-Stage Local Refinement.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from Run 003 winner (Borda-Sawtooth)
# BEST_FINE_PARAMS: [0.40726313808519926, 0.27932513225170796, 0.08020598008569421, 0.0, 0.32392596559876763, 0.030945146263037977, 0.1821552140242117]
PREV_BEST_PARAMS = [0.40726314, 0.27932513, 0.08020598, 0.0, 0.32392597, 0.03094515, 0.18215521]

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_borda_sawtooth(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Enhanced sawtooth with re-entrant lips (Run 003 Elite)."""
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
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        lx = dx - ramp_len
        lip = (lx > 0.0) & (lx < lip_size) & (ay > throat - lip_ang * lx) & (ay < throat + 0.08)
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_bridge_tesla(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Curved Tesla-style valve with central teardrop islands and side loops."""
    gap = 0.15 + 0.30 * params[0]
    island_l = 0.15 + 0.35 * params[1]
    loop_r = 0.18 + 0.25 * params[2]
    loop_w = 0.04 + 0.12 * params[3]
    island_y_bias = 0.05 + 0.15 * params[4] # Controls island thickness curvature

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)
    
    n_stages = 2
    pitch = 0.48
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for i in range(n_stages):
        cx = 0.25 + i * pitch
        dx = (x - cx) / island_l
        
        # 1. Central Teardrop Island
        # Equation: (x/a)^2 + (y/b)^2 * (1 + k*x) <= 1
        is_island = (jnp.abs(dx) < 0.5) & (ay < gap * (1.0 - 4.0 * dx**2) * (1.0 - island_y_bias * dx))
        
        # 2. Side Loop Arcs (Bridge)
        dy_loop = ay - (gap + loop_r)
        dist_outer = jnp.sqrt((x - cx)**2 + dy_loop**2)
        is_loop = (dist_outer < loop_r + loop_w/2) & (dist_outer > loop_r - loop_w/2) & (x > cx - 0.1) & (ay > gap)
        
        # 3. Outer boundary reinforcement near loops
        mask = jnp.maximum(mask, is_island.astype(jnp.float32))
        mask = jnp.maximum(mask, is_loop.astype(jnp.float32))

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Unified space: [selector, p1, p2, p3, p4, p5, p6]"""
    selector = params[0]
    mask = jax.lax.cond(
        selector < 0.55,
        lambda p: render_borda_sawtooth(p[1:6], nx_design, ny),
        lambda p: render_bridge_tesla(p[1:6], nx_design, ny),
        params
    )
    # Wall enforcement
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    mask = jnp.where(jnp.abs(y) > 0.98, 1.0, mask)
    return mask

render_batch_coarse = jax.jit(
    jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny))
)
render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(44)

    # --- Phase 1: Global Screening (Coarse) ---
    batch_size_global = 128
    pop = rng.uniform(0.0, 1.0, size=(batch_size_global, 7)).astype(np.float32)
    pop[0] = np.array(PREV_BEST_PARAMS) # Seed with Run 003 winner
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    scores_c_np = np.asarray(scores_c)
    
    # --- Phase 2: High-Fidelity Verification ---
    top_k = 10
    elite_idx = np.argsort(scores_c_np)[::-1][:top_k]
    pop_f1 = pop[elite_idx]
    
    masks_f1 = render_batch_fine(jnp.asarray(pop_f1))
    scores_f1, qf1, qr1 = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    scores_f1_np = np.asarray(scores_f1)
    
    # --- Phase 3: Iterative Local Refinement (Fine) ---
    # Round A: Gaussian search around top 2 candidates
    best_idx_f1 = np.argsort(scores_f1_np)[::-1][:2]
    neighbors = []
    for idx in best_idx_f1:
        base_p = pop_f1[idx]
        neighbors.append(base_p) # Keep parent
        neighbors.extend(np.clip(base_p + rng.normal(0, 0.04, size=(7, 7)), 0, 1))
    
    pop_f2 = np.array(neighbors).astype(np.float32)
    masks_f2 = render_batch_fine(jnp.asarray(pop_f2))
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    scores_f2_np = np.asarray(scores_f2)
    
    # Final Selection
    winner_idx = np.argmax(scores_f2_np)
    best_params = pop_f2[winner_idx]
    best_score = scores_f2_np[winner_idx]
    
    print(f"[design.py] Finished in {time.time()-t_start:.1f}s | Best Fine Di = {best_score:.8f}")
    print("BEST_FINE_PARAMS:", best_params.tolist())
    
    return np.asarray(masks_f2[winner_idx], dtype=np.float32)

if __name__ == "__main__":
    mask = generate_design()
    print("Final Mask Shape:", mask.shape)
