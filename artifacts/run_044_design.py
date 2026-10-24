"""
Fluidic Diode Design & Continuous Optimization Module.
Run 044: Sting-Cavity Asymmetry & Vertical Staggering.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from run 025 (Di = 1.974761)
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9974176287651062, 
    0.05909641459584236, 0.22175829112529755, 0.561127245426178, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_asymmetric_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Mapping parameters (12 dims)
    sting_width_fac = 0.1 + 1.5 * params[0]
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    sting_taper_fac = 0.0 + 2.0 * params[8]
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    # Asymmetry parameter: vertical offset of island center
    vert_offset     = (params[11] - 0.5) * 0.4 

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(3):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # Asymmetric ramp (alternating y-offset)
        y_off = vert_offset if k % 2 == 0 else -vert_offset
        ay = jnp.abs(y - y_off)
        
        # Nozzle
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # Island
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        
        # Profile
        half_w = (throat * 0.5) * (1.0 - jnp.abs(idx - 0.5) * 1.5)
        island = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # Sting
        cav_idx = (idx - (1.0 - 0.3 * concavity))
        sting = (cav_idx > 0.0) & (cav_idx < 0.3) & (ay < (half_w * 0.2 * sting_width_fac))
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
        mask = jnp.maximum(mask, sting.astype(jnp.float32))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_asymmetric_diode(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_asymmetric_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # 1. Search population
    pop = np.tile(PREV_BEST_PARAMS, (64, 1))
    pop[1:] += rng.normal(0, 0.1, (63, 12))
    pop = np.clip(pop, 0, 1)
    
    # 2. Coarse screening
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 3. Fine refinement
    top_idx = np.argsort(np.asarray(scores_c))[-8:]
    refine_pop = np.clip(pop[top_idx] + rng.normal(0, 0.02, (8, 12)), 0, 1)
    refine_pop[0] = PREV_BEST_PARAMS
    
    masks_f = render_batch_fine(jnp.asarray(refine_pop))
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    best_p = refine_pop[best_idx]
    
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f}")
    print("BEST_FINE_PARAMS:", best_p.tolist())
    
    return np.asarray(masks_f[best_idx])
