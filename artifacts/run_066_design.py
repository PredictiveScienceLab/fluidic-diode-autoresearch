"""
Fluidic Diode Design & Continuous Optimization Module.
Run 066: Asymmetric Vortex-Locking with Dual-Lip Borda Phase-Shifting.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from Run 052 (Di = 2.013648)
BEST_PARAMS = np.array([
    0.94975233, 0.71391362, 0.09493200, 0.83849537, 
    0.81148946, 0.18793973, 0.17253507, 0.94670385, 
    0.0, 0.18118741, 0.57170140, 0.28217008
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v5(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Mapping parameters (12 dims)
    sting_w    = 0.15 + 1.2 * params[0] 
    throat     = 0.10 + 0.35 * params[1]
    ramp_len   = 0.20 + 0.35 * params[2]
    ramp_pow   = 0.50 + 1.50 * params[3]
    lip_size   = 0.05 + 0.15 * params[4]
    lip_hook   = 0.10 + 1.40 * params[5]
    isl_len    = 0.20 + 0.25 * params[6]
    isl_pos    = 0.05 + 0.15 * params[7]
    flare      = params[8] * 0.5
    concavity  = 0.20 + 0.70 * params[9]
    pitch      = 0.30 + 0.10 * params[10]
    y_offset   = -0.15 + 0.3 * params[11] # Spanwise island stagger

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    n_stages = 3
    
    for k in range(n_stages):
        x0 = 0.05 + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Nozzle
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (jnp.abs(y) >= yw)
        
        # 2. Borda Lips (Asymmetric placement)
        lx = dx - (ramp_len + 0.02)
        y_lip = throat - lip_hook * lx
        lip = (lx > 0.0) & (lx < lip_size) & (jnp.abs(y) > y_lip) & (jnp.abs(y) < y_lip + 0.1)
        
        # 3. Asymmetric Hammerhead (Spanwise shift y_offset)
        idx = (dx - (ramp_len + isl_pos)) / isl_len
        y_center = y - y_offset * ((k % 2) * 2 - 1)
        half_w = throat * 0.45 * (1.0 - jnp.abs(idx - 0.5) * 1.5)
        island = (idx > 0.0) & (idx < 1.0) & (jnp.abs(y_center) < half_w)
        
        # 4. Sting
        sting = (idx > 0.3) & (idx < 0.9) & (jnp.abs(y_center) < throat * 0.05 * sting_w)
        
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, jnp.maximum(island, sting))))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v5(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v5(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # Population-based refinement
    pop = jnp.clip(BEST_PARAMS + rng.normal(0, 0.06, (96, 12)), 0, 1)
    pop = pop.at[0].set(BEST_PARAMS)
    
    # Coarse pass
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Fine pass with elite survival
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[top_idx] + rng.normal(0, 0.01, (8, 12)), 0, 1)
    refine_pop = refine_pop.at[0].set(pop[top_idx[0]])
    
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    
    return np.asarray(masks_f[best_idx])
