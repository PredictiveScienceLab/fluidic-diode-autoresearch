"""
Fluidic Diode Design & Continuous Optimization Module.
Run 035: Asymmetric Serpentine-Trap & Differential Borda-Sting Offset.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Best from Run 025 (Di = 1.974761)
PREV_BEST_PARAMS = np.array([
    0.98273104, 0.72881252, 0.15233745, 0.84737521, 
    0.74325496, 0.16419895, 0.16507927, 0.99741763, 
    0.05909641, 0.22175829, 0.56112725, 0.18808439
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_asymmetric_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Param mapping
    sting_width_fac = 0.1 + 1.5 * params[0]
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    sting_taper     = 2.0 * params[8]
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    isl_offset_y    = -0.2 + 0.4 * params[11] # New: Vertical asymmetry offset

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    n_stages = 3
    for k in range(n_stages):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # Asymmetric Nozzle
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        y_w = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & ((y > y_w) | (y < -y_w))
        
        # Island with offset
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        y_centered = y - isl_offset_y
        
        # Shape
        half_w = (throat * 0.45) * jnp.sqrt(jnp.clip(idx * (1.0 - idx) * 4.0, 0.0, 1.0))
        island = (idx >= 0.0) & (idx <= 1.0) & (jnp.abs(y_centered) < half_w)
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    return render_asymmetric_diode(params, nx_design, ny)

render_batch = jax.jit(jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # Population-based refinement
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (32, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    masks = render_batch(pop)
    scores, _, _ = evaluate_batch_fast(masks, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores))
    print(f"[design.py] Best Di: {scores[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", pop[best_idx].tolist())
    
    return np.asarray(masks[best_idx])
