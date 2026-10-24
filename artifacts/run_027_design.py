"""
Fluidic Diode Design & Continuous Optimization Module.
Run 027: Introducing asymmetric vertical island offsets to force tortuous reverse flow.
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
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9974176287651062, 
    0.05909641459584236, 0.22175829112529755, 0.561127245426178, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_asymmetric_catcher(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Param mapping (12 dims)
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
    isl_offset_y    = -0.15 + 0.30 * params[11] # New: Vertical asymmetry

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # Ramp & Lip
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        lx = dx - ramp_len
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        # Island with Y-Offset
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        
        # Asymmetric center
        y_center = isl_offset_y * (1.0 - 2.0 * jnp.abs(idx - 0.5))
        y_rel = y - y_center
        ay_rel = jnp.abs(y_rel)
        
        profile = jnp.sqrt(jnp.clip(idx / 0.4, 0.0, 1.0)) * (idx < 0.4) + (1.0 + 0.8 * (jnp.maximum(0.0, idx - 0.4)**3)) * (idx >= 0.4)
        envelope = jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        half_w = (throat * 0.55) * profile * envelope
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay_rel < half_w)
        
        # Sting
        cav_start = 1.0 - 0.45 * concavity
        cav_idx = (idx - cav_start) / (1.0 - cav_start + 1e-6)
        sting_width = (throat * 0.05) * sting_width_fac * jnp.clip(1.0 - sting_taper * ((cav_idx - 0.2) / 0.6), 0.0, 1.0)
        sting = (cav_idx > 0.2) & (cav_idx < 0.8) & (ay_rel < sting_width)
        
        island_solid = jnp.maximum(island_outer.astype(jnp.float32) * (1.0 - ((cav_idx > 0.0) & (ay_rel < half_w * 0.85 * concavity)).astype(jnp.float32)), sting.astype(jnp.float32))

        mask = jnp.maximum(mask, jnp.maximum(tooth, jnp.maximum(lip, island_solid)))

    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_asymmetric_catcher(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))
render_batch_coarse = jax.jit(jax.vmap(lambda p: render_asymmetric_catcher(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (64, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    # Coarse Screen
    masks_c = render_batch_coarse(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Fine Refinement
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop = jnp.clip(pop[top_idx] + rng.normal(0, 0.02, (8, 12)), 0, 1)
    
    masks_f = render_batch_fine(refine_pop)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores_f))
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", refine_pop[best_idx].tolist())
    return np.asarray(masks_f[best_idx])
