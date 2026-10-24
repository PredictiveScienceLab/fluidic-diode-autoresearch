"""
Fluidic Diode Design & Continuous Optimization Module.
Multi-Fidelity Elite-Drift Strategy: Using FAST_CONFIG for population-based search
and DEFAULT_CONFIG for final candidate validation.
"""

from __future__ import annotations
import time
import jax
import jax.numpy as jnp
import numpy as np
from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Global Best Known Parameters from run_049 (Di = 383.239117)
BEST_PARAMS = np.array([
    0.34585416, 0.6224917, 0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191, 0.8370709, 0.0,
    0.6155642, 0.36107752, 0.84232432, 0.5733315, 0.0, 0.0, 0.0
], dtype=np.float32)

@jax.jit
def render_geometry(params: jnp.ndarray, nx: int, ny: int) -> jnp.ndarray:
    # (Same geometry logic as defined in the problem description)
    nozzle_len = 0.16 + 0.36 * params[0]
    throat_base = 0.27 + 0.32 * params[1]
    nozzle_pow = 0.60 + 1.10 * params[2]
    cup_offset = 0.32 + 0.38 * params[3]
    cup_ry_base = 0.12 + 0.18 * params[4]
    prow_stretch = 0.55 + 1.20 * params[5]
    cavity_len = 0.12 + 0.58 * params[6]
    wall_thick = 0.042 + 0.075 * params[7]
    nose_thick = 0.10 + 0.24 * params[8]
    pitch_scale = 0.78 + 0.16 * params[9]
    flare = -0.16 + 0.32 * params[10]
    undercut = 0.00 + 0.12 * params[11]
    ry_grade = -0.15 + 0.30 * params[12]
    th_grade = -0.15 + 0.30 * params[13]
    fin_len_scale = 0.0 + 0.8 * params[14]
    fin_width_scale = 0.0 + 0.8 * params[15]
    lip_start = 0.0 + 0.3 * params[16]
    lip_len = 0.0 + 0.15 * params[17]
    lip_depth = 0.0 + 0.25 * params[18]

    x = jnp.linspace(0.0, 1.0, nx)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * (k - 1.0)), 0.22, 0.68)
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len) & (abs_y >= y_wall)
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry_base * (1.0 + ry_grade * (k - 1.0)) + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = (ux_rel <= cavity_len) & (abs_y <= ry_local)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06)**2 + (abs_y/ry_inner)**2) <= 1.0)
        in_cup = in_outer & (~hollow)
        fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | fin).astype(jnp.float32))
    return mask

def generate_design() -> np.ndarray:
    rng = np.random.default_rng(42)
    best_params = BEST_PARAMS.copy()
    
    # Batch evaluate 64 candidates on FAST_CONFIG
    pop = [best_params] + [np.clip(best_params + rng.normal(0, 0.02, 19), 0, 1) for _ in range(63)]
    pop_arr = np.stack(pop)
    m_fast = jax.vmap(lambda p: render_geometry(p, FAST_CONFIG.nx - 2*FAST_CONFIG.pad_x, FAST_CONFIG.ny))(jnp.array(pop_arr))
    scores, _, _ = evaluate_batch_fast(m_fast, FAST_CONFIG)
    
    # Select top 8 and evaluate on DEFAULT_CONFIG
    top_indices = np.argsort(scores)[-8:]
    top_params = pop_arr[top_indices]
    m_fine = jax.vmap(lambda p: render_geometry(p, DEFAULT_CONFIG.nx - 2*DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))(jnp.array(top_params))
    fine_scores, _, _ = evaluate_batch_fast(m_fine, DEFAULT_CONFIG)
    
    best_idx = np.argmax(fine_scores)
    print(f"Final Selection Di: {fine_scores[best_idx]:.4f}")
    return np.array(m_fine[best_idx])
