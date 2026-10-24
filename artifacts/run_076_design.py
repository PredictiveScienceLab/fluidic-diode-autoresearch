"""
Fluidic Diode Design & Continuous Optimization Module.
Run 076: Phase-Locked Asymmetric Borda-Lip Sting & Vortex-Ratchet Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 072 winner (Di = 2.123762)
PREV_BEST_PARAMS = np.array([
    0.9699504971504211, 0.7256813645362854, 0.09763211011886597, 0.9148319959640503, 
    0.7092326283454895, 0.14038322865962982, 0.0, 0.996536910533905, 
    0.08833111822605133, 0.25795435905456543, 0.5827779769897461, 0.28380316495895386
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v7(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Optimized Tandem-Pin Topology with Phase-Locked Borda Lobe modulation.
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
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Phase-Locked Borda Lip (Offset modulated by stage index k)
        phase_k = lip_offset + 0.05 * (k - 1)
        lx = dx - (ramp_len + phase_k)
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.12)
        
        # 3. Hammerhead Island Base
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        half_w = (throat * 0.55) * jnp.where(idx < 0.4, jnp.sqrt(jnp.clip(idx / 0.4, 0, 1)), 1.0)
        envelope = jnp.clip(idx * 10.0, 0, 1) * jnp.clip((1.0 - idx) * 10.0, 0, 1)
        
        # 4. Tapered Sting & Cavity
        cavity = (idx > (1.0 - 0.45 * concavity)) & (ay < half_w * 0.85 * concavity)
        sting = (idx > 0.4) & (idx < 0.9) & (ay < (throat * 0.05) * sting_width_fac * (1.0 - 0.5*idx))
        
        # 5. Tandem Vortex Pin
        pin_x_rel = 1.15
        pin_idx = (dx - (ix_start + isl_scale_x * pin_x_rel)) / 0.08
        pin_solid = (pin_idx >= 0.0) & (pin_idx <= 1.0) & (ay < throat * 0.25 * pin_strength)
        
        island_solid = jnp.maximum((idx >= 0.0) & (idx <= 1.0) & (ay < half_w * envelope) & ~cavity, 
                                   jnp.maximum(sting, pin_solid))
        
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, island_solid)))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v7(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v7(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    pop_size = 128
    noise_scale = 0.04
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, noise_scale, (pop_size, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    top_k = 16
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:top_k]
    refine_pop = pop[top_idx]
    
    masks_f1 = render_batch_fine(refine_pop)
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    best_f_idx = np.argmax(np.asarray(scores_f1))
    best_params = refine_pop[best_f_idx]
    
    local_pop = jnp.clip(best_params + rng.normal(0, 0.01, (12, 12)), 0, 1)
    local_pop = local_pop.at[0].set(best_params)
    
    masks_f2 = render_batch_fine(local_pop)
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_best_idx = np.argmax(np.asarray(scores_f2))
    final_best_params = local_pop[final_best_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", final_best_params.tolist())
    
    return np.asarray(masks_f2[final_best_idx])
