"""
Fluidic Diode Design & Continuous Optimization Module.
Run 062: Tandem-Island Vortex-Locking & Multi-Stage Stagnation.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Winner from Run 052
BEST_FINE_PARAMS = np.array([
    0.94975233, 0.71391362, 0.09493200, 0.83849537, 
    0.81148946, 0.18793973, 0.17253508, 0.94670385, 
    0.00000000, 0.18118741, 0.57170141, 0.28217009
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_tandem_vortex_v1(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Tandem-Island topology: Hammerhead (primary) + trailing wing (secondary) 
    to increase reverse flow stagnation.
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
    pitch           = 0.28 + 0.12 * params[10]
    lip_offset      = -0.1 + 0.2 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Nozzle
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Tandem Island (Primary + Wing)
        ix_s = ramp_len + isl_pos_x
        idx = (dx - ix_s) / isl_scale_x
        # Primary head
        primary = (idx > 0.0) & (idx < 0.4) & (ay < throat * 0.4 * (1.0 - idx*2))
        # Trailing wing
        wing = (idx > 0.5) & (idx < 0.85) & (ay < throat * 0.25)
        
        island = jnp.maximum(primary, wing)
        
        # 3. Borda Lip
        lx = dx - (ramp_len + lip_offset)
        y_lip = throat - lip_hook * lx
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.1)

        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(island, lip)))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_tandem_vortex_v1(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_tandem_vortex_v1(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    # Search around current elite
    pop = jnp.clip(BEST_FINE_PARAMS + rng.normal(0, 0.03, (128, 12)), 0, 1)
    pop = pop.at[0].set(BEST_FINE_PARAMS)
    
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[top_idx] + rng.normal(0, 0.008, (8, 12)), 0, 1)
    
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
