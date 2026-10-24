"""
Fluidic Diode Design & Continuous Optimization Module.
Run 038: Sting-Vortex-Locking Asymmetry & Refined Taper Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Best from Run 025
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9974176287651062, 
    0.05909641459584236, 0.22175829112529755, 0.561127245426178, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_asymmetric_sting(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Parameter Mapping
    sting_width_fac = 0.1 + 1.5 * params[0]
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    sting_taper_fac = 0.0 + 2.0 * params[8]
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    y_offset        = -0.1 + 0.2 * params[11] # Asymmetry offset

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    n_stages = 3
    mx = 0.04
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # Asymmetric y-coordinate for island
        y_isl = y - (y_offset * ((-1)**k)) 
        ay = jnp.abs(y)
        ay_isl = jnp.abs(y_isl)
        
        # Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # Lip
        lx = dx - ramp_len
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        # Island
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        half_w = (throat * 0.5) * jnp.sqrt(jnp.clip(idx*10, 0, 1) * jnp.clip((1-idx)*10, 0, 1))
        island = (idx >= 0.0) & (idx <= 1.0) & (ay_isl < half_w)
        
        # Sting
        sting = (idx > 0.4) & (idx < 0.9) & (ay_isl < (half_w * 0.2 * sting_width_fac * (1.0 - sting_taper_fac * (idx-0.4))))
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
        mask = jnp.maximum(mask, sting.astype(jnp.float32))
        
    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_asymmetric_sting(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_asymmetric_sting(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    num_params = 12
    
    pop = np.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (64, num_params)), 0, 1).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS
    
    scores_c, _, _ = evaluate_batch_fast(render_batch_coarse(pop), FAST_CONFIG)
    top_idx = np.argsort(scores_c)[::-1][:8]
    
    refine_pop = np.clip(pop[top_idx[0]] + rng.normal(0, 0.02, (16, num_params)), 0, 1)
    refine_pop[0] = pop[top_idx[0]]
    
    scores_f, _, _ = evaluate_batch_fast(render_batch_fine(refine_pop), DEFAULT_CONFIG)
    best_idx = np.argmax(scores_f)
    
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    
    return np.asarray(render_batch_fine(refine_pop[best_idx:best_idx+1])[0])
