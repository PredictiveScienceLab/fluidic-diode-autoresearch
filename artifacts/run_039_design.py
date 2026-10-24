"""
Fluidic Diode Design & Continuous Optimization Module.
Run 039: Asymmetric Vortex-Wedge Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Winner from run_025
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9974176287651062, 
    0.05909641459584236, 0.22175829112529755, 0.561127245426178, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_asymmetric_wedge(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    throat, ramp_len, ramp_pow = 0.25 + 0.2*params[1], 0.15 + 0.4*params[2], 0.4 + 1.6*params[3]
    lip_size, lip_hook = 0.05 + 0.15*params[4], 0.05 + 1.45*params[5]
    isl_scale_x, isl_pos_x = 0.15 + 0.3*params[6], 0.02 + 0.2*params[7]
    sting_taper, concavity, pitch = 1.0 * params[8], 0.3 + 0.65*params[9], 0.28 + 0.12*params[10]
    y_offset = -0.15 + 0.3 * params[11] # Asymmetry parameter

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y - y_offset)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(3):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # Nozzle + Lip
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        lx = dx - ramp_len
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        # Island
        idx = (dx - (ramp_len + isl_pos_x)) / isl_scale_x
        half_w = (throat * 0.55) * jnp.sqrt(jnp.clip(idx / 0.4, 0.0, 1.0)) * (idx < 0.4)
        island = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # Asymmetric Wedge (Vortex trap)
        cav_idx = (idx - 0.5) / 0.5
        wedge = (cav_idx > -0.2) & (cav_idx < 0.6) & (ay < (throat * 0.2) * (1.0 - sting_taper * cav_idx))
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32) * (1.0 - wedge.astype(jnp.float32)))
    return mask

def generate_design() -> np.ndarray:
    rng = np.random.default_rng(42)
    pop = np.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (64, 12)), 0, 1)
    pop[0] = PREV_BEST_PARAMS
    
    masks_c = jax.vmap(lambda p: render_asymmetric_wedge(p, 250, 100))(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, DEFAULT_CONFIG)
    
    # Refine top 8
    best_idx = np.argsort(scores_c)[-8:]
    refine_pop = np.clip(pop[best_idx] + rng.normal(0, 0.02, (8, 12)), 0, 1)
    refine_pop[0] = pop[best_idx[-1]]
    
    masks_f = jax.vmap(lambda p: render_asymmetric_wedge(p, 250, 100))(jnp.asarray(refine_pop))
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    idx = np.argmax(scores_f)
    print(f"BEST_FINE_PARAMS: {refine_pop[idx].tolist()}")
    return np.asarray(masks_f[idx])
