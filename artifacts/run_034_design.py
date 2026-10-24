"""
Fluidic Diode Design & Continuous Optimization Module.
Run 034: Asymmetric Serpentine-Trap Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from Run 025 (Di = 1.974761)
PREV_BEST_PARAMS = np.array([
    0.98273104, 0.72881252, 0.15233745, 0.84737521, 
    0.74325496, 0.16419895, 0.16507927, 0.99741763, 
    0.05909641, 0.22175829, 0.56112725, 0.18808439
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_asymmetric_serpentine(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Asymmetric Serpentine Topology:
    - Introduces `asym_offset` (params[11]) to stagger island positions vertically.
    - Maintains re-entrant Borda lips for reverse flow capture.
    """
    sting_width_fac = 0.1 + 1.5 * params[0]
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    sting_taper_f   = 0.0 + 2.0 * params[8]
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    asym_offset     = -0.2 + 0.4 * params[11] # Vertical stagger for islands

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Nozzle
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (jnp.abs(y) >= yw)
        
        # 2. Island (with vertical stagger)
        y_center = (k % 2) * asym_offset
        y_isl = y - y_center
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        
        half_w = (throat * 0.4) * (1.0 - (idx - 0.5)**2 * 4.0)
        island = (idx >= 0.0) & (idx <= 1.0) & (jnp.abs(y_isl) < half_w)
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_asymmetric_serpentine(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_asymmetric_serpentine(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    pop = np.tile(PREV_BEST_PARAMS, (128, 1)).astype(np.float32)
    pop[1:] += rng.normal(0, 0.1, (127, 12))
    pop = np.clip(pop, 0, 1)
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    best_c_idx = np.argsort(scores_c)[-16:]
    refine_pop = pop[best_c_idx]
    
    masks_f = render_batch_fine(jnp.asarray(refine_pop))
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(scores_f)
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    
    return np.asarray(masks_f[best_idx])
