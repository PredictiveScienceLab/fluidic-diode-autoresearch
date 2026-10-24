"""
Fluidic Diode Design & Continuous Optimization Module.
Run 049: Phase-Shifted Zigzag & Hammerhead Optimization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Best parameters from Run 047 (Di = 2.001761)
PREV_BEST_PARAMS = np.array([
    0.98273104429245, 0.7288125157356262, 0.152337446808815, 0.847375214099884, 
    0.7432549595832825, 0.16419894993305206, 0.165079265832901, 0.9979028105735779, 
    0.05909641459584236, 0.22175829112529755, 0.5692214369773865, 0.18808439373970032
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v4(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Mapping parameters
    sting_width_fac = 0.1 + 1.5 * params[0] 
    throat          = 0.12 + 0.30 * params[1]
    ramp_len        = 0.15 + 0.40 * params[2]
    ramp_pow        = 0.40 + 1.60 * params[3]
    lip_size        = 0.04 + 0.16 * params[4]
    lip_hook        = 0.05 + 1.45 * params[5]
    isl_scale_x     = 0.15 + 0.30 * params[6]
    isl_pos_x       = 0.02 + 0.20 * params[7]
    p8              = params[8] # Hammerhead flare factor
    concavity       = 0.30 + 0.65 * params[9]
    pitch           = 0.28 + 0.12 * params[10]
    zigzag_offset   = 0.10 * (params[11] - 0.5) # Vertical stagger

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    
    n_stages = 3
    mx = 0.04
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        # Apply vertical zigzag offset per stage
        y_shift = zigzag_offset * ((-1) ** k)
        ay = jnp.abs(y - y_shift)
        
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Borda Lip
        lx = dx - ramp_len
        y_lip = throat - lip_hook * lx + 3.5 * lip_hook * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11)
        
        # 3. Hammerhead Island
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        
        # Flare logic
        flare = 1.0 + p8 * (jnp.maximum(0.0, idx - 0.4)**3)
        half_w = (throat * 0.55) * jnp.sqrt(jnp.clip(idx*2.5, 0, 1)) * flare
        envelope = jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w * envelope)
        
        # 4. Concavity & Sting
        cavity = (idx > (1.0 - 0.4*concavity)) & (ay < half_w * 0.7)
        sting = (idx > 0.6) & (idx < 0.9) & (ay < (throat * 0.15))
        
        island_solid = (island_outer.astype(jnp.float32) * (1.0 - cavity.astype(jnp.float32)))
        island_solid = jnp.maximum(island_solid, sting.astype(jnp.float32))

        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_solid)

    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v4(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    pop = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.03, (32, 12)), 0, 1)
    pop = pop.at[0].set(PREV_BEST_PARAMS)
    
    masks = render_batch_fine(pop)
    scores, _, _ = evaluate_batch_fast(masks, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores))
    best_p = pop[best_idx]
    
    print(f"[design.py] Best Di: {scores[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_p.tolist())
    return np.asarray(masks[best_idx])
