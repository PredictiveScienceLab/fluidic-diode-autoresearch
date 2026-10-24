"""
Fluidic Diode Design & Continuous Optimization Module.
Run 050: Hammerhead Sting & Asymmetric Borda-Lip Phase-Shift.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Elite parameters from Run 047
PREV_BEST_PARAMS = np.array([
    0.98273104, 0.72881252, 0.15233745, 0.84737521, 
    0.74325496, 0.16419895, 0.16507927, 0.99790281, 
    0.05909641, 0.22175829, 0.56922144, 0.18808439
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_hammerhead_v4(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Mapping parameters (12 dims)
    s_w = 0.1 + 1.5 * params[0] 
    thr = 0.12 + 0.30 * params[1]
    r_l = 0.15 + 0.40 * params[2]
    r_p = 0.40 + 1.60 * params[3]
    l_s = 0.04 + 0.16 * params[4]
    l_h = 0.05 + 1.45 * params[5]
    isl_x = 0.15 + 0.30 * params[6]
    isl_p = 0.02 + 0.20 * params[7]
    p8 = params[8] # Hammerhead flare factor
    conc = 0.30 + 0.65 * params[9]
    pitch = 0.28 + 0.12 * params[10]
    lip_asym = 0.05 * params[11] # New: Phase shift for upper vs lower lip

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(3):
        x0 = 0.04 + k * pitch
        dx = (x - x0) / pitch
        
        # 1. Main Nozzle Ramp
        s = jnp.clip(dx / r_l, 0.0, 1.0)
        yw = 1.0 - (1.0 - thr) * (s ** r_p)
        tooth = (dx >= 0.0) & (dx <= r_l) & (ay >= yw)
        
        # 2. Asymmetric Borda Lip
        lx_u = dx - r_l - lip_asym
        lx_l = dx - r_l + lip_asym
        y_lip_u = thr - l_h * lx_u + 3.5 * l_h * (jnp.maximum(0.0, lx_u)**2)
        y_lip_l = thr - l_h * lx_l + 3.5 * l_h * (jnp.maximum(0.0, lx_l)**2)
        lip = ((lx_u > 0.0) & (lx_u < l_s) & (y > y_lip_u) & (y < y_lip_u + 0.11)) | \
              ((lx_l > 0.0) & (lx_l < l_s) & (y < -y_lip_l) & (y > -y_lip_l - 0.11))
        
        # 3. Hammerhead Island
        ix_start = r_l + isl_p
        idx = (dx - ix_start) / isl_x
        profile = jnp.where(idx < 0.4, jnp.sqrt(jnp.clip(idx/0.4, 0, 1)), 1.0 + 0.8 * (jnp.maximum(0.0, idx - 0.4)**3))
        half_w = (thr * 0.55) * profile * jnp.clip(idx * 10, 0, 1) * jnp.clip((1-idx) * 10, 0, 1)
        island = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # 4. Sting Flare
        cav_idx = (idx - (1.0 - 0.45 * conc)) / (0.45 * conc + 1e-6)
        sting_w = (thr * 0.05 * s_w) * jnp.clip(1.0 - 0.6 * cav_idx + 5.0 * (p8-0.15) * (cav_idx**4), 0.01, 4.0)
        sting = (cav_idx > 0.2) & (cav_idx < 0.8) & (ay < sting_w)
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, (island & ~((cav_idx > 0) & (ay < half_w * 0.85 * conc))).astype(jnp.float32))
        mask = jnp.maximum(mask, sting.astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_hammerhead_v4(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    rng = np.random.default_rng(42)
    pop = np.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (64, 12)), 0, 1)
    pop[0] = PREV_BEST_PARAMS
    
    masks = render_batch_fine(jnp.asarray(pop))
    scores, _, _ = evaluate_batch_fast(masks, DEFAULT_CONFIG)
    
    best_idx = np.argmax(np.asarray(scores))
    print(f"[design.py] Best Di: {scores[best_idx]:.6f}")
    print("BEST_FINE_PARAMS:", pop[best_idx].tolist())
    return np.asarray(masks[best_idx])
