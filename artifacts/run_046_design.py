"""
Fluidic Diode Design & Continuous Optimization Module.
Run 046: Staggered Serpentine-Catcher with Borda-Lip Optimization.
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
def render_serpentine_catcher(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Mapping parameters
    sting_width_fac    = 0.1 + 1.5 * params[0]
    throat             = 0.12 + 0.30 * params[1]
    ramp_len           = 0.15 + 0.40 * params[2]
    ramp_pow           = 0.40 + 1.60 * params[3]
    lip_size           = 0.04 + 0.25 * params[4]
    lip_hook           = 0.05 + 1.60 * params[5]
    isl_scale_x        = 0.15 + 0.30 * params[6]
    isl_pos_x          = 0.02 + 0.20 * params[7]
    sting_taper_factor = 0.0 + 2.0 * params[8]
    concavity          = 0.30 + 0.65 * params[9]
    pitch              = 0.28 + 0.12 * params[10]
    y_offset           = -0.1 + 0.2 * params[11] # Vertical staggering for serpentine effect

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    n_stages = 3
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # Apply vertical offset to alternating stages to break symmetry
        offset = y_offset * ((-1) ** k)
        ay = jnp.abs(y - offset)
        
        # Nozzle
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # Borda Lip
        lx = dx - ramp_len
        y_lip = throat - lip_hook * lx + (3.5 * lip_hook) * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.12)
        
        # Island
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        half_w = (throat * 0.55) * jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        cav_start = 1.0 - 0.45 * concavity
        cavity = ((idx - cav_start) > 0.0) & (ay < half_w * 0.85 * concavity)
        
        # Tapered Sting
        taper_progress = (idx - (ix_start + 0.2)) / (0.8 - 0.2 + 1e-6)
        sting_w = (throat * 0.05 * sting_width_fac) * jnp.clip(1.0 - sting_taper_factor * taper_progress, 0.0, 1.0)
        sting = (idx > 0.2) & (idx < 0.8) & (ay < sting_w)
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, (island_outer * (1.0 - cavity)).astype(jnp.float32))
        mask = jnp.maximum(mask, sting.astype(jnp.float32))

    return mask

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # 64 candidates, 3 refinement rounds
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.06, (64, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    masks_c = jax.vmap(lambda p: render_serpentine_catcher(p, 250, 100))(pop)
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # Refinement
    top_p = pop[np.argsort(scores_c)[::-1][:16]]
    pop_f = jnp.clip(top_p + rng.normal(0, 0.02, (16, 12)), 0, 1)
    
    masks_f = jax.vmap(lambda p: render_serpentine_catcher(p, 250, 100))(pop_f)
    scores_f, _, _ = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    
    best_idx = np.argmax(scores_f)
    print(f"[design.py] Best Di: {scores_f[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", pop_f[best_idx].tolist())
    
    return np.asarray(masks_f[best_idx])
