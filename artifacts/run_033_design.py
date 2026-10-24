"""
Fluidic Diode Design & Continuous Optimization Module.
Run 033: Asymmetric Crescent-Lip + Dual-Vortex Trap Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from run_025 (Di = 1.974761)
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9974176287651062, 
    0.05909641459584236, 0.22175829112529755, 0.561127245426178, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_crescent_trap(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Topology: Asymmetric crescent-lip wall deflectors and dual-vortex traps.
    Uses params[8] for wall asymmetry/offset and params[9] for trap depth.
    """
    throat = 0.12 + 0.30 * params[1]
    ramp_len = 0.15 + 0.40 * params[2]
    lip_hook = 0.05 + 1.45 * params[5]
    isl_pos_x = 0.02 + 0.20 * params[7]
    wall_offset = -0.15 + 0.30 * params[8] # Asymmetry parameter
    trap_depth = 0.2 + 0.4 * params[9]
    pitch = 0.28 + 0.12 * params[10]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(3):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # Asymmetric walls
        y_upper = 1.0 - (throat + wall_offset) * (1.0 - jnp.exp(-5.0 * dx))
        y_lower = -1.0 + (throat - wall_offset) * (1.0 - jnp.exp(-5.0 * dx))
        wall = (dx > 0.0) & (dx < 0.8) & ((y > y_upper) | (y < y_lower))
        
        # Crescent lip (reverse flow check)
        lip = (dx > 0.3) & (dx < 0.5) & (y > throat + 0.1) & (y < throat + 0.1 + lip_hook)
        
        # Trap island
        ix = (dx - (0.4 + isl_pos_x)) / 0.2
        island = (ix**2 + (y/trap_depth)**2 < 0.1)
        
        mask = jnp.maximum(mask, wall.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_crescent_trap(p, 150, 60)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_crescent_trap(p, 250, 100)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    pop = np.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (64, 12)), 0, 1).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Refine top 16
    idx = np.argsort(scores_c)[-16:]
    pop_f = pop[idx]
    masks_f = render_batch_fine(jnp.asarray(pop_f))
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(scores_f)
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f}")
    print("BEST_FINE_PARAMS:", pop_f[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
