"""
Fluidic Diode Design & Continuous Optimization Module.
Run 051: Asymmetric Vortex-Locking & S-Curve Serpentine Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite from Run 047
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9979028105735779, 
    0.05909641459584236, 0.22175829112529755, 0.5692214369773865, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v4(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Refined Topology: Adds a spanwise vertical stagger (params[10]) to each stage 
    to force S-curve reverse flow.
    """
    sting_width_fac = 0.1 + 1.5 * params[0] 
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    p8              = params[8]
    concavity       = 0.30 + 0.65 * params[9]
    stage_offset    = -0.2 + 0.4 * params[10] # Vertical shift per stage
    isl_flare       = 0.40 + 0.80 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    n_stages = 3
    pitch = 0.33 
    
    for k in range(n_stages):
        x0 = 0.05 + k * pitch
        dx = (x - x0) / pitch
        
        # Apply vertical offset y_shift to break symmetry
        y_shift = stage_offset * (k - 1)
        ay = jnp.abs(y - y_shift)
        
        # Geometry
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        lx = dx - ramp_len
        y_lip = throat - lip_hook * lx + (3.5 * lip_hook) * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        idx = (dx - (ramp_len + isl_pos_x)) / isl_scale_x
        half_w = (throat * 0.55) * (1.0 + isl_flare * (jnp.maximum(0.0, idx - 0.4)**3))
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w * jnp.clip(idx*10*(1-idx)*10, 0, 1))
        
        cav = (idx > 0.6) & (ay < half_w * 0.85 * concavity)
        sting = (idx > 0.3) & (idx < 0.9) & (ay < (throat * 0.08) * sting_width_fac)
        
        island = jnp.maximum(island_outer * (1.0 - cav), sting)
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, island)))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v4(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v4(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # 1. Broad Search
    pop = np.tile(PREV_BEST_PARAMS, (128, 1)).astype(np.float32)
    pop += rng.normal(0, 0.1, (128, 12))
    pop = np.clip(pop, 0, 1)
    pop[0] = PREV_BEST_PARAMS
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Refinement
    top_idx = np.argsort(scores_c)[-16:]
    refine_pop = pop[top_idx]
    masks_f = render_batch_fine(jnp.asarray(refine_pop))
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(scores_f)
    best_p = refine_pop[best_idx]
    
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_p.tolist())
    return np.asarray(masks_f[best_idx])
