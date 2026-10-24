"""
Fluidic Diode Design & Continuous Optimization Module.
Run 094: Split-Wing Vortex Catcher & Asymmetric Vena-Contracta Bifurcation.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from run_084
BEST_PARAMS = np.array([
    0.9699504971504211, 0.7256813645362854, 0.09763211011886597, 0.9148319959640503, 
    0.7092326283454895, 0.14038322865962982, 0.0, 0.996536910533905, 
    0.08833111822605133, 0.25795435905456543, 0.5827779769897461, 0.28380316495895386
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v8(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Split-Wing Vortex Catcher: Enhances reverse flow resistance by splitting the 
    stagnation point into a bifurcated wake-trap.
    """
    sting_w    = 0.15 + 1.2 * params[0]
    throat     = 0.15 + 0.25 * params[1]
    ramp_len   = 0.20 + 0.30 * params[2]
    ramp_pow   = 0.50 + 1.50 * params[3]
    lip_size   = 0.05 + 0.15 * params[4]
    lip_hook   = 0.10 + 1.20 * params[5]
    isl_pos_x  = 0.05 + 0.25 * params[7]
    pin_str    = 0.10 + 0.80 * params[8]
    concavity  = 0.20 + 0.70 * params[9]
    pitch      = 0.30 + 0.10 * params[10]
    lip_off    = -0.05 + 0.15 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(3):
        x0 = 0.05 + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Nozzle
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Split-Wing Catcher (Vortex Trap)
        lx = dx - (ramp_len + isl_pos_x)
        wing_shape = (lx > 0) & (lx < 0.3) & (ay < throat * 0.4 * (1.0 - jnp.abs(lx/0.15 - 1.0)))
        
        # 3. Concave Collector
        collector = (lx > 0.15) & (lx < 0.4) & (ay < throat * concavity) & ~wing_shape
        
        # 4. Asymmetric Lip
        lx_lip = dx - (ramp_len + lip_off)
        lip = (lx_lip > 0) & (lx_lip < lip_size) & (ay > throat + lip_hook*lx_lip) & (ay < throat + lip_hook*lx_lip + 0.1)
        
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(wing_shape, jnp.maximum(collector, lip))))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v8(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v8(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # Evolutionary strategy: broaden search around the current elite
    pop = jnp.clip(BEST_PARAMS + rng.normal(0, 0.08, (128, 12)), 0, 1)
    pop = pop.at[0].set(BEST_PARAMS)
    
    scores_c, _, _ = evaluate_batch_fast(render_batch_coarse(pop), FAST_CONFIG)
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = pop[top_idx]
    
    scores_f, _, _ = evaluate_batch_fast(render_batch_fine(refine_pop), DEFAULT_CONFIG)
    best_idx = np.argmax(np.asarray(scores_f))
    best_params = refine_pop[best_idx]
    
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_params.tolist())
    
    return np.asarray(render_batch_fine(best_params[None])[0])
