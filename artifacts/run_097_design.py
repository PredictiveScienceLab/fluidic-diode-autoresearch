"""
Fluidic Diode Design & Continuous Optimization Module.
Run 097: Enhanced Asymmetric Borda-Lip Bifurcation & Non-Linear Sting Curvature.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from run_095
BEST_PARAMS = np.array([
    0.9699504971504211, 0.7256813645362854, 0.09763211011886597, 0.9148319959640503, 
    0.7092326283454895, 0.14038322865962982, 0.0, 0.996536910533905, 
    0.08833111822605133, 0.25795435905456543, 0.5827779769897461, 0.28380316495895386
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v9(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Optimized Bifurcation Topology: Enhanced lip-to-pin coupling and non-linear stage tapering.
    Added non-linear sting curvature for improved wake interference.
    """
    sting_width_fac = 0.1 + 1.5 * params[0] 
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    pin_strength    = params[8] 
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
        stage_factor = 1.0 + 0.1 * k
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat * stage_factor) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Bifurcating Borda Lip
        lx = dx - (ramp_len + lip_offset)
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size * stage_factor) & (ay > y_lip) & (ay < y_lip + 0.12)
        
        # 3. Island and Sting (Curvature enhanced)
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        half_w = (throat * 0.55) * jnp.where(idx < 0.4, jnp.sqrt(jnp.clip(idx / 0.4, 0, 1)), 1.0)
        envelope = jnp.clip(idx * 10.0, 0, 1) * jnp.clip((1.0 - idx) * 10.0, 0, 1)
        cavity = (idx > (1.0 - 0.45 * concavity)) & (ay < half_w * 0.85 * concavity)
        # Non-linear sting curvature using squared idx term
        sting = (idx > 0.4) & (idx < 0.95) & (ay < (throat * 0.05) * sting_width_fac * (1.0 - 0.5 * (idx**1.5)))
        
        # 4. Phase-Locked Tandem Pin
        pin_x_rel = 1.15 + (k * 0.08)
        pin_idx = (dx - (ix_start + isl_scale_x * pin_x_rel)) / 0.09
        pin_solid = (pin_idx >= 0.0) & (pin_idx <= 1.0) & (ay < throat * 0.30 * pin_strength)
        
        island_solid = jnp.maximum((idx >= 0.0) & (idx <= 1.0) & (ay < half_w * envelope) & ~cavity, 
                                   jnp.maximum(sting, pin_solid))
        
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, island_solid)))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v9(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v9(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    pop = jnp.clip(BEST_PARAMS + rng.normal(0, 0.05, (128, 12)), 0, 1)
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
