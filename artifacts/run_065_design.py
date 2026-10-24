"""
Fluidic Diode Design & Continuous Optimization Module.
Run 065: Tandem-Sting Asymmetry & Tapered Vortex-Locking.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from Run 052 (Di = 2.013648)
BEST_FINE_PARAMS = np.array([
    0.94975233, 0.71391362, 0.09493199, 0.83849537, 
    0.81148946, 0.18793973, 0.17253507, 0.94670385, 
    0.0, 0.18118740, 0.57170140, 0.28217008
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v5(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Tandem-Sting Vortex-Locking Topology with Asymmetric Phase-Shifting.
    """
    # Mapping parameters (12 dims)
    sting_w_base    = 0.1 + 1.2 * params[0] 
    throat          = 0.15 + 0.25 * params[1]
    ramp_len        = 0.20 + 0.35 * params[2]
    ramp_pow        = 0.50 + 1.50 * params[3]
    lip_size        = 0.05 + 0.15 * params[4]
    lip_hook        = 0.10 + 1.20 * params[5]
    isl_scale_x     = 0.20 + 0.25 * params[6]
    isl_pos_x       = 0.05 + 0.15 * params[7]
    flare_factor    = params[8] * 0.5
    concavity       = 0.40 + 0.50 * params[9]
    pitch           = 0.30 + 0.10 * params[10]
    lip_offset      = -0.15 + 0.3 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.05
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Asymmetric Hooked Borda Lip (Staggered per stage)
        lx = dx - (ramp_len + lip_offset * ((-1)**k))
        hook_curve = 3.0 * lip_hook 
        y_lip = throat - lip_hook * lx + hook_curve * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.15)
        
        # 3. Tandem-Sting Hammerhead
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        # Blunt head + tapered tail
        half_w = (throat * 0.5) * jnp.exp(-((idx-0.2)/0.3)**2)
        sting = (idx > 0.2) & (idx < 0.9) & (ay < (throat * 0.1) * sting_w_base * (1.0 - 0.5*idx))
        
        island_solid = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, jnp.maximum(island_solid, sting))))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_hammerhead_v5(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v5(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    # Search population around elite
    pop = jnp.clip(BEST_FINE_PARAMS + rng.normal(0, 0.04, (128, 12)), 0, 1)
    pop = pop.at[0].set(BEST_FINE_PARAMS)
    
    # Fidelity Screening
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Elitist Refinement
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[top_idx] + rng.normal(0, 0.01, (8, 12)), 0, 1)
    
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
