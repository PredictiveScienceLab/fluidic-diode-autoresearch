"""
Fluidic Diode Design & Continuous Optimization Module.
Run 063: Tandem-Sting Asymmetry & Staggered Borda-Lip Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Best from Run 052 (Di = 2.013648)
BEST_FINE_PARAMS = np.array([
    0.94975233, 0.71391362, 0.09493200, 0.83849537, 
    0.81148946, 0.18793973, 0.17253508, 0.94670385, 
    0.0, 0.18118741, 0.57170141, 0.28217009
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_tandem_sting(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Mapping parameters (12 dims)
    sting_w    = 0.05 + 0.25 * params[0]
    throat     = 0.15 + 0.25 * params[1]
    ramp_len   = 0.15 + 0.35 * params[2]
    ramp_pow   = 0.50 + 1.50 * params[3]
    lip_size   = 0.05 + 0.15 * params[4]
    lip_hook   = 0.05 + 1.20 * params[5]
    isl_pos    = 0.10 + 0.20 * params[6]
    isl_gap    = 0.05 + 0.15 * params[7] # Gap between tandem elements
    p8         = params[8] # Second stage flare
    concavity  = 0.20 + 0.70 * params[9]
    pitch      = 0.30 + 0.10 * params[10]
    lip_shift  = -0.15 + 0.30 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.05
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Throat/Lip
        yw = 1.0 - (1.0 - throat) * (jnp.clip(dx / ramp_len, 0.0, 1.0) ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        lx = dx - (ramp_len + lip_shift)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > throat - lip_hook * lx) & (ay < throat + 0.1)
        
        # 2. Tandem Island
        ix_mid = ramp_len + isl_pos
        # Primary head
        head = (dx > ix_mid) & (dx < ix_mid + 0.15) & (ay < throat * 0.4)
        # Trailing sting (asymmetric)
        sting = (dx > ix_mid + 0.15 + isl_gap) & (dx < ix_mid + 0.45) & (ay < sting_w * (1.0 + p8))
        
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, jnp.maximum(head, sting))))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_tandem_sting(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_tandem_sting(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # Genetic optimization: Seed with elite
    pop = jnp.clip(BEST_FINE_PARAMS + rng.normal(0, 0.06, (96, 12)), 0, 1)
    pop = pop.at[0].set(BEST_FINE_PARAMS)
    
    # Multi-fidelity screening
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[top_idx] + rng.normal(0, 0.01, (8, 12)), 0, 1)
    
    # Final verification
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
