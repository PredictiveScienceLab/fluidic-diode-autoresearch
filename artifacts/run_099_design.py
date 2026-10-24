"""
Fluidic Diode Design & Continuous Optimization Module.
Run 099: Asymmetric Vortex-Ratchet Coupling & Crescent-Cut Sting Optimization.
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
    Refined Topology: Crescent-Cut Sting & Tapered Vortex-Ratchet Coupling.
    """
    sting_w_fac    = 0.1 + 1.5 * params[0] 
    throat         = 0.12 + 0.30 * params[1]
    ramp_len       = 0.15 + 0.40 * params[2]
    ramp_pow       = 0.40 + 1.60 * params[3]
    lip_size       = 0.04 + 0.16 * params[4]
    lip_hook       = 0.05 + 1.45 * params[5]
    isl_scale_x    = 0.15 + 0.30 * params[6]
    isl_pos_x      = 0.02 + 0.20 * params[7]
    pin_strength   = params[8] 
    concavity      = 0.30 + 0.65 * params[9]
    pitch          = 0.28 + 0.12 * params[10]
    lip_offset     = -0.1 + 0.2 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(3):
        stage_factor = 1.0 + 0.05 * k
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat * stage_factor) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Borda Lip
        lx = dx - (ramp_len + lip_offset)
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size * stage_factor) & (ay > y_lip) & (ay < y_lip + 0.12)
        
        # 3. Island with Crescent-Cut
        idx = (dx - (ramp_len + isl_pos_x)) / isl_scale_x
        half_w = (throat * 0.55) * jnp.sin(jnp.pi * jnp.clip(idx, 0, 1))
        cavity = (idx > (0.6 - 0.2 * concavity)) & (ay < half_w * 0.7 * concavity)
        # Crescent cut: force flow path constriction
        crescent = (idx > 0.3) & (idx < 0.7) & (ay < (throat * 0.15) * (1.0 - 2.0*jnp.abs(idx-0.5)))
        
        island = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w) & ~cavity & ~crescent
        
        # 4. Phase-Locked Tandem Pin
        pin_idx = (dx - (ramp_len + isl_pos_x + isl_scale_x * (1.1 + 0.05*k))) / 0.09
        pin = (pin_idx >= 0.0) & (pin_idx <= 1.0) & (ay < throat * 0.30 * pin_strength)
        
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, jnp.maximum(island, pin))))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v9(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v9(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    pop = jnp.clip(BEST_PARAMS + rng.normal(0, 0.05, (128, 12)), 0, 1).at[0].set(BEST_PARAMS)
    
    scores_c, _, _ = evaluate_batch_fast(render_batch_coarse(pop), FAST_CONFIG)
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    scores_f, _, _ = evaluate_batch_fast(render_batch_fine(pop[top_idx]), DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    best_params = pop[top_idx][best_idx]
    
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_params.tolist())
    
    return np.asarray(render_batch_fine(best_params[None])[0])
