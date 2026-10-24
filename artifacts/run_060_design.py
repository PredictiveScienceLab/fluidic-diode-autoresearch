"""
Fluidic Diode Design & Continuous Optimization Module.
Run 060: Asymmetric Vortex-Locking & Tapered Hammerhead Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from run_052
BEST_PARAMS = np.array([
    0.94975233, 0.71391362, 0.09493199, 0.83849537, 
    0.81148946, 0.18793973, 0.17253508, 0.94670385, 
    0.0, 0.18118741, 0.57170141, 0.28217009
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v5(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Optimized Hammerhead with Tapered Sting and Phase-Shifted Borda Lips.
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
        
        # 1. Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Phase-shifted Borda Lip
        lx = dx - (ramp_len + lip_offset)
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.12)
        
        # 3. Hammerhead with Tapered Sting
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        profile = jnp.where(idx < 0.4, jnp.sqrt(jnp.clip(idx / 0.4, 0, 1)), 1.0 + p8 * (jnp.maximum(0.0, idx - 0.4)**3))
        envelope = jnp.clip(idx * 10.0, 0, 1) * jnp.clip((1.0 - idx) * 10.0, 0, 1)
        half_w = (throat * 0.55) * profile * envelope
        
        cav_idx = (idx - (1.0 - 0.45 * concavity)) / (0.45 * concavity + 1e-6)
        cavity = (cav_idx > 0.0) & (ay < half_w * 0.85 * concavity)
        sting = (cav_idx > 0.2) & (cav_idx < 0.8) & (ay < (throat * 0.05) * sting_width_fac * (1.0 - 0.2 * idx))
        
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, jnp.maximum((idx >= 0.0) & (idx <= 1.0) & (ay < half_w) & ~cavity, sting))))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v5(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v5(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    # Search around current elite
    pop = jnp.clip(BEST_PARAMS + rng.normal(0, 0.04, (128, 12)), 0, 1)
    pop = pop.at[0].set(BEST_PARAMS)
    
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[top_idx] + rng.normal(0, 0.01, (8, 12)), 0, 1)
    
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
