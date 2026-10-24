"""
Fluidic Diode Design & Continuous Optimization Module.
Run 032: Crescent-Lip Asymmetry & Vortex-Trap Optimization.
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
    0.98273104, 0.72881252, 0.15233745, 0.84737521, 0.74325496, 
    0.16419895, 0.16507927, 0.99741763, 0.05909641, 0.22175829, 
    0.56112725, 0.18808439
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_crescent_trap(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Optimized Crescent-Trap Geometry:
    - Asymmetric wall-lips to create a 'swirl' effect in reverse flow.
    - Central island with a deep, non-linear concave rear face.
    - Splitter sting optimized for reverse flow impedance.
    """
    # Parameter map
    sting_width_fac = 0.1 + 1.5 * params[0]
    throat          = 0.15 + 0.25 * params[1]
    ramp_len        = 0.20 + 0.30 * params[2]
    ramp_pow        = 0.50 + 1.50 * params[3]
    lip_size        = 0.05 + 0.15 * params[4]
    lip_hook        = 0.10 + 1.00 * params[5]
    isl_scale_x     = 0.20 + 0.25 * params[6]
    isl_pos_x       = 0.05 + 0.15 * params[7]
    sting_taper_fac = 0.5 + 1.5 * params[8]
    concavity       = 0.40 + 0.50 * params[9]
    pitch           = 0.30 + 0.10 * params[10]
    lip_asym        = -0.1 + 0.2 * params[11] # Asymmetric lip offset

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)
    sy = jnp.sign(y)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    n_stages = 3
    mx = 0.04
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Asymmetric Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Crescent-Lip (Asymmetric Borda)
        lx = dx - ramp_len
        y_lip = throat - lip_hook * lx + 2.0 * (jnp.maximum(0.0, lx)**2)
        # Apply asymmetry based on sy
        lip = (lx > 0.0) & (lx < lip_size) & (y > y_lip + lip_asym * sy) & (y < y_lip + lip_asym * sy + 0.12)
        
        # 3. Flared Island
        ix_start = ramp_len + isl_pos_x
        idx = (dx - ix_start) / isl_scale_x
        nose_part = jnp.sqrt(jnp.clip(idx / 0.4, 0.0, 1.0)) * (idx < 0.4)
        flare_part = (1.0 + 0.5 * (jnp.maximum(0.0, idx - 0.4)**2)) * (idx >= 0.4)
        profile = jnp.where(idx < 0.4, nose_part, flare_part)
        half_w = (throat * 0.5) * profile * jnp.clip(idx * 10.0 * (1.0 - idx) * 10.0, 0.0, 1.0)
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        cavity = (idx > 1.0 - 0.4 * concavity) & (ay < half_w * 0.8)
        
        # Splitter Sting
        sting = (idx > 0.4) & (idx < 0.9) & (ay < (throat * 0.06) * sting_width_fac * (1.0 - sting_taper_fac * idx))
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, (island_outer.astype(jnp.float32) * (1.0 - cavity.astype(jnp.float32))))
        mask = jnp.maximum(mask, sting.astype(jnp.float32))

    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_crescent_trap(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # Elite-based search
    pop = [PREV_BEST_PARAMS] + [np.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, 12), 0, 1) for _ in range(15)]
    masks = render_batch_fine(jnp.asarray(pop))
    scores, _, _ = evaluate_batch_fast(masks, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores))
    best_p = pop[best_idx]
    
    print(f"[design.py] Best Di: {scores[best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_p.tolist())
    return np.asarray(masks[best_idx])
