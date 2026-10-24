"""
Fluidic Diode Design & Continuous Optimization Module.
Run 031: Crescent-Lip Asymmetric Vortex-Trap Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Best parameters from Run 025 (Di = 1.974761)
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9974176287651062, 
    0.05909641459584236, 0.22175829112529755, 0.561127245426178, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_crescent_trap(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Advanced Catcher Topology (v3):
    - Implements asymmetric Crescent-Lip wall-hooks.
    - Staggered vertical positioning of islands to force tortuous reverse paths.
    """
    # Mapping parameters
    sting_width_fac = 0.1 + 1.5 * params[0]
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.20 * params[4]
    lip_curve       = 0.05 + 1.60 * params[5] # Crescent curvature
    
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    sting_taper     = 0.0 + 2.0 * params[8]
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    y_offset        = -0.1 + 0.2 * params[11] # Vertical asymmetry

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(3):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # Asymmetric walls
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        y_wall_top = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        y_wall_bot = -1.0 + (1.0 - throat) * (s ** ramp_pow)
        
        tooth = (dx >= 0.0) & (dx <= ramp_len) & ((y > y_wall_top) | (y < y_wall_bot))
        
        # Crescent Lips (Asymmetric)
        lx = dx - ramp_len
        lip = (lx > 0.0) & (lx < lip_size) & ((y > throat - lip_curve*lx) | (y < -throat + lip_curve*lx))
        
        # Offset Island
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        iy_c = y - y_offset
        island = (idx > 0.0) & (idx < 1.0) & (jnp.abs(iy_c) < (throat * 0.4 * (1.0 - idx**2)))
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
        
    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_crescent_trap(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_crescent_trap(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    pop = np.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (64, 12)), 0, 1).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS
    
    # Coarse search
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Fine refinement
    top_idx = np.argsort(scores_c)[-8:]
    pop_f = np.clip(pop[top_idx] + rng.normal(0, 0.02, (8, 12)), 0, 1)
    masks_f = render_batch_fine(jnp.asarray(pop_f))
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(scores_f)
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", pop_f[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
