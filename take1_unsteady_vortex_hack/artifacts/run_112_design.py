"""
Fluidic Diode Design & Continuous Optimization Module.
Optimizing the 19-parameter vortex-locking diode architecture.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# BEST_PARAMS_RUN049 (Di = 383.239117)
BEST_PARAMS = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0, 0.0, 0.0
], dtype=np.float32)

@jax.jit
def render_diode(params: jnp.ndarray, nx: int, ny: int) -> jnp.ndarray:
    # Unpack parameters
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
        stage_centered = k - 1.0
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * stage_centered), 0.10, 0.36)
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = (ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2) <= 1.0)
        in_cup = in_outer & (~hollow)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        
        is_lip = (lip_len > 1e-6) & (x >= x0 + pitch * (rear_edge + lip_start)) & (x <= x0 + pitch * (rear_edge + lip_start + lip_len)) & \
                 (abs_y >= y_wall - lip_depth * (1.0 - y_wall)) & (abs_y <= y_wall)
        
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin | is_lip).astype(jnp.float32))
    return mask

def generate_design() -> np.ndarray:
    rng = np.random.default_rng(42)
    # Generate population
    pop = [BEST_PARAMS.copy()]
    for _ in range(127):
        p = BEST_PARAMS.copy() + rng.normal(0, 0.015, size=19)
        pop.append(np.clip(p, 0.0, 1.0).astype(np.float32))
    
    # Coarse screening
    masks_fast = jax.vmap(lambda p: render_diode(p, FAST_CONFIG.nx - 2*FAST_CONFIG.pad_x, FAST_CONFIG.ny))(jnp.stack(pop))
    scores_fast, _, _ = evaluate_batch_fast(masks_fast, FAST_CONFIG)
    
    # Select top 8 for fine evaluation
    top_indices = np.argsort(np.array(scores_fast))[-8:]
    best_fine_params = np.stack([pop[i] for i in top_indices])
    masks_fine = jax.vmap(lambda p: render_diode(p, DEFAULT_CONFIG.nx - 2*DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))(jnp.stack(best_fine_params))
    scores_fine, _, _ = evaluate_batch_fast(masks_fine, DEFAULT_CONFIG)
    
    best_idx = int(np.argmax(scores_fine))
    print(f"Iteration complete. Best Di: {scores_fine[best_idx]:.6f}")
    print(f"Best Params: {best_fine_params[best_idx].tolist()}")
    
    return np.asarray(masks_fine[best_idx])
