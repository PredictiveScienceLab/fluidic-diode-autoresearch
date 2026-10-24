"""
Fluidic Diode Design & Continuous Optimization Module.
Run 068: Tapered-Sting-Splitter & Non-linear Borda-lip Phase Staggering.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from Run 067 (Di = 2.100043)
PREV_BEST_PARAMS = np.array([
    0.96912134, 0.72702110, 0.11894368, 0.87489581, 
    0.72909552, 0.18468273, 0.06188370, 0.96405077, 
    0.00375062, 0.23378712, 0.60700804, 0.28197339
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v6(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Refined Catcher Topology: Introducing non-linear Borda-lip phase staggering 
    and a Tapered-Sting-Splitter to maximize reverse flow impedance.
    """
    sting_w = 0.15 + 1.2 * params[0] 
    throat  = 0.15 + 0.25 * params[1]
    ramp_l  = 0.20 + 0.30 * params[2]
    ramp_p  = 0.50 + 1.50 * params[3]
    lip_sz  = 0.05 + 0.15 * params[4]
    lip_hk  = 0.10 + 1.00 * params[5]
    isl_sx  = 0.20 + 0.25 * params[6]
    isl_px  = 0.05 + 0.15 * params[7]
    p8      = params[8] 
    concav  = 0.35 + 0.50 * params[9]
    pitch   = 0.30 + 0.10 * params[10]
    lip_off = -0.15 + 0.3 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Nozzle Ramp
        s = jnp.clip(dx / ramp_l, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_p)
        tooth = (dx >= 0.0) & (dx <= ramp_l) & (ay >= yw)
        
        # 2. Phase-Staggered Hooked Borda Lip
        lx = dx - (ramp_l + lip_off + (k * 0.02))
        y_lip = throat - lip_hk * lx + 4.0 * lip_hk * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_sz) & (ay > y_lip) & (ay < y_lip + 0.15)
        
        # 3. Tapered-Sting-Splitter
        ix_start = ramp_l + isl_px
        idx = (dx - ix_start) / isl_sx
        half_w = (throat * 0.5) * jnp.sqrt(jnp.clip(idx * 2.0, 0, 1))
        
        cavity = (idx > (0.6 - 0.2 * concav)) & (ay < half_w * 0.7 * concav)
        sting = (idx > 0.3) & (idx < 0.85) & (ay < (throat * 0.06) * sting_w * (1.0 - 0.6*idx))
        
        island = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w) & ~cavity
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, jnp.maximum(island, sting))))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v6(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v6(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    # Search with higher mutation around the elite
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (128, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[top_idx] + rng.normal(0, 0.015, (8, 12)), 0, 1)
    
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
