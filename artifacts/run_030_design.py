"""
Fluidic Diode Design & Continuous Optimization Module.
Run 030: Crescent-Lip Asymmetry & Enhanced Vortex-Trap Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Best parameters from Run 025 (Di = 1.974761)
PREV_BEST_PARAMS = np.array([
    0.98273104, 0.72881252, 0.15233745, 0.84737521, 
    0.74325496, 0.16419895, 0.16507927, 0.99741763, 
    0.05909641, 0.22175829, 0.56112725, 0.18808439
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_crescent_catcher(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Advanced Catcher Topology (v3):
    - Staggered re-entrant lips (Crescent-Lip Asymmetry).
    - Maintains previous sting and island logic.
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
    lip_stagger     = 0.1 * params[11] # New: Vertical offset for upper/lower lips

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)
    sy = jnp.sign(y)

    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        lx = dx - ramp_len
        # Apply stagger: upper lip shifted by +lip_stagger, lower by -lip_stagger
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2) + (sy * lip_stagger)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        half_w = (throat * 0.55) * jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        cavity = (idx > (1.0 - 0.45 * concavity)) & (ay < half_w * 0.85 * concavity)
        
        sting = (idx > 0.6) & (idx < 0.9) & (ay < (throat * 0.05 * sting_width_fac) * jnp.clip(1.0 - sting_taper * (idx-0.6), 0.0, 1.0))
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_outer.astype(jnp.float32) * (1.0 - cavity.astype(jnp.float32)))
        mask = jnp.maximum(mask, sting.astype(jnp.float32))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_crescent_catcher(p, 250, 100)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_crescent_catcher(p, 250, 100)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (64, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    best_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[best_idx] + rng.normal(0, 0.02, (8, 12)), 0, 1)
    
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_final_idx = np.argmax(np.asarray(scores_f))
    best_final_p = refine_pop[best_final_idx]
    
    print(f"[design.py] Best Di: {scores_f[best_final_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f[best_final_idx])
