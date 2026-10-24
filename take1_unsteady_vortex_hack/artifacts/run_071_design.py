"""
Fluidic Diode Design & Continuous Optimization Module (run_071).
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, evaluate_batch_fast

# Optimal parameters from run_049 (Di = 383.239117)
BEST_PARAMS = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0, 0.0, 0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Standard mapping
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
    
    # New parameter: fin_twist_angle (P19)
    fin_twist = -0.2 + 0.4 * params[18] 

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        stage_centered = k - 1.0
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * stage_centered), 0.10, 0.36)
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch

        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len) & (abs_y >= y_wall)

        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        in_outer = (ux_rel <= cavity_len) & (ux_rel >= -0.2) & (abs_y <= ry_local)
        
        # Apply fin twist: rotate the fin's y-bounds based on x-position
        fin_len_actual = fin_len_scale * cavity_len
        fin_width_actual = fin_width_scale * (ry_local - wall_thick)
        y_center_shift = fin_twist * (ux_rel / fin_len_actual)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_actual) & (jnp.abs(y - y_center_shift) <= fin_width_actual)

        stage = (wall_nozzle | (in_outer & ~((ux_rel > 0.0) & (abs_y <= ry_local - wall_thick)))) | is_fin
        mask = jnp.maximum(mask, stage.astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)

def generate_design() -> np.ndarray:
    t0 = time.time()
    best_params = BEST_PARAMS.copy()
    best_score = 383.239117
    
    # 3 rounds of targeted search
    for r in range(3):
        cands = [best_params.copy()]
        for _ in range(7):
            p = best_params.copy()
            p += np.random.normal(0, 0.015, size=p.shape)
            cands.append(np.clip(p, 0, 1))
        
        scores, _, _ = evaluate_batch_fast(render_batch_fine(jnp.array(cands)), DEFAULT_CONFIG)
        if np.max(scores) > best_score:
            best_score = float(np.max(scores))
            best_params = cands[int(np.argmax(scores))]
            
    return np.asarray(render_batch_fine(best_params[None, :])[0])
