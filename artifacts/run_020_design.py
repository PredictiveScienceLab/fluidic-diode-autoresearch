"""
Fluidic Diode Design & Continuous Optimization Module.
Run 020: Staggered Serpentine Vortex-Trap Diode.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite from Run 014 (Di = 1.860782)
BEST_ELITE = np.array([
    0.84756416, 0.77508932, 0.0, 0.78332424, 0.72259378, 
    0.22029935, 0.11345403, 0.98639494, 0.01423382, 0.11474635, 
    0.56609869, 0.16622637
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_staggered_serpentine(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Staggered Serpentine Topology:
    - params[8] now controls 'vertical_offset' of the island to break spanwise symmetry.
    - Uses the same base 'Trumpet Catcher' logic but with shifted island centers.
    """
    sting_width_fac = 0.1 + 1.5 * params[0] 
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    vert_offset     = (params[8] - 0.5) * 0.4 # New: Vertical shift [-0.2, 0.2]
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    isl_flare       = 0.40 + 0.80 * params[11]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(3):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # Geometry logic (Nozzle + Lip)
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # Staggered island vertical position
        y_center = ((-1)**k) * vert_offset
        y_isl = y - y_center
        
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        half_w = (throat * 0.55) * jnp.where(idx < 0.4, jnp.sqrt(jnp.clip(idx/0.4, 0, 1)), (1.0 + isl_flare * (jnp.maximum(0.0, idx-0.4)**3)))
        
        island = (idx >= 0.0) & (idx <= 1.0) & (jnp.abs(y_isl) < half_w)
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
    return mask

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # Population-based optimization
    pop = rng.uniform(0.0, 1.0, (64, 12)).astype(np.float32)
    pop[0] = BEST_ELITE
    
    # Coarse screen
    masks_c = jax.vmap(lambda p: render_staggered_serpentine(p, 250, 100))(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Fine refinement
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = pop[top_idx]
    
    masks_f = jax.vmap(lambda p: render_staggered_serpentine(p, 250, 100))(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    
    return np.asarray(masks_f[best_idx])
