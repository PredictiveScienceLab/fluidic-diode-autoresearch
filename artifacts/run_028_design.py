"""
Fluidic Diode Design & Continuous Optimization Module.
Run 028: Asymmetric Crescent-Lip Serpentine-Trap Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Winner of Run 025 (Di = 1.974761)
PREV_BEST_PARAMS = np.array([
    0.98273104, 0.72881252, 0.15233745, 0.84737521, 0.74325496, 0.16419895, 
    0.16507927, 0.99741763, 0.05909641, 0.22175829, 0.56112725, 0.18808439
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_crescent_lip_asym(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Topology: Staggered re-entrant lips and asymmetric island placement.
    params[0-11] control geometry, with params[10] modified to induce 
    vertical staggering of the wall-lips to force serpentine reverse-flow.
    """
    sting_width_fac = 0.1 + 1.5 * params[0]
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    sting_taper     = 0.0 + 2.0 * params[8]
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    isl_flare       = 0.40 + 0.80 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    n_stages = 3
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # Asymmetric lip offset based on stage index
        y_offset = 0.15 * ((-1)**k)
        
        # Main Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (jnp.abs(y - y_offset) >= yw)
        
        # Asymmetric Hooked Lip
        lx = dx - ramp_len
        y_lip = (throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2))
        lip = (lx > 0.0) & (lx < lip_size) & (jnp.abs(y - y_offset) > y_lip) & (jnp.abs(y - y_offset) < y_lip + 0.11)
        
        # Island
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        profile = jnp.where(idx < 0.4, jnp.sqrt(jnp.clip(idx / 0.4, 0.0, 1.0)), 1.0 + isl_flare * (jnp.maximum(0.0, idx - 0.4)**3))
        half_w = (throat * 0.55) * profile * jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        
        island = (idx >= 0.0) & (idx <= 1.0) & (jnp.abs(y - y_offset) < half_w)
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))

    return mask

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (64, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    masks_c = jax.vmap(lambda p: render_crescent_lip_asym(p, 250, 100))(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    best_idx = np.argsort(scores_c)[-8:]
    pop_f = pop[best_idx]
    
    masks_f = jax.vmap(lambda p: render_crescent_lip_asym(p, 250, 100))(pop_f)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_final_idx = np.argmax(scores_f)
    print(f"[design.py] Best Di: {scores_f[best_final_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", pop_f[best_final_idx].tolist())
    
    return np.asarray(masks_f[best_final_idx])
