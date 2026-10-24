"""
Fluidic Diode Design & Continuous Optimization Module.
Run 061: Tandem Hammerhead & Asymmetric Borda-Lip Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from run_052 (Di = 2.013648)
PREV_BEST_PARAMS = np.array([
    0.94975233, 0.71391362, 0.09493200, 0.83849537, 
    0.81148946, 0.18793973, 0.17253508, 0.94670385, 
    0.0, 0.18118741, 0.57170141, 0.28217009
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_tandem_hammerhead(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Tandem Hammerhead Topology with dynamic throat gap and Borda-lip phase-shifting.
    """
    sting_width_fac = 0.1 + 1.2 * params[0] 
    throat          = 0.15 + 0.25 * params[1]
    ramp_len        = 0.20 + 0.30 * params[2]
    ramp_pow        = 0.50 + 1.50 * params[3]
    lip_size        = 0.05 + 0.15 * params[4]
    lip_hook        = 0.05 + 1.20 * params[5]
    isl_gap         = 0.05 + 0.10 * params[6] # Gap between tandem islands
    isl_pos_x       = 0.05 + 0.15 * params[7]
    p8              = params[8] # Flare strength
    concavity       = 0.20 + 0.70 * params[9]
    pitch           = 0.30 + 0.10 * params[10]
    lip_offset      = -0.15 + 0.3 * params[11]

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
        
        # 2. Asymmetric Hooked Borda Lip
        lx = dx - (ramp_len + lip_offset)
        y_lip = throat - lip_hook * lx + 2.0 * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.15)
        
        # 3. Tandem Hammerhead Islands
        ix_base = ramp_len + isl_pos_x
        # Island 1
        idx1 = (dx - ix_base) / 0.15
        i1 = (idx1 > 0.0) & (idx1 < 1.0) & (ay < 0.2 * throat)
        # Island 2 (The Catcher)
        idx2 = (dx - (ix_base + 0.15 + isl_gap)) / 0.25
        flare = (1.0 + p8 * (jnp.maximum(0.0, idx2 - 0.4)**2))
        half_w = (throat * 0.6) * flare * jnp.clip((1.0 - idx2)*4.0, 0.0, 1.0)
        cav_start = 1.0 - 0.5 * concavity
        cavity = (idx2 > cav_start) & (ay < half_w * 0.9 * concavity)
        i2 = (idx2 > 0.0) & (idx2 < 1.0) & (ay < half_w) & ~cavity
        
        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, jnp.maximum(i1, i2))))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_tandem_hammerhead(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_tandem_hammerhead(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    # Evolve population from current elite
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.06, (96, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Select top 8 and refine
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[top_idx] + rng.normal(0, 0.02, (8, 12)), 0, 1)
    
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
