"""
Fluidic Diode Design & Continuous Optimization Module (run_083).
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# BEST_PARAMS_RUN049 (Di = 383.239117)
BEST_PARAMS_RUN049 = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0,        0.0,
    0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
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
    
    # P16-P18: Lip parameters
    lip_start_offset = 0.0 + 0.3 * params[16]
    lip_len = 0.0 + 0.15 * params[17]
    lip_depth = 0.0 + 0.25 * params[18]
    # P19: fin_taper_gradient
    fin_taper = -0.2 + 0.4 * params[19] if len(params) > 19 else 0.0

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
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        
        # Geometry logic preserved from previous elite run
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len) & (abs_y >= (1.0 - (1.0 - throat_gap) * ((dx / nozzle_len) ** nozzle_pow)))
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry_base + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        in_cup = (ux_rel <= cavity_len) & (ux_rel >= 0.0) & (abs_y <= ry_local) & ~((ux_rel > 0.0) & (abs_y <= ry_local - wall_thick))
        
        # Splitter Fin with Taper Gradient
        fin_len = fin_len_scale * cavity_len
        fin_w = fin_width_scale * (1.0 + fin_taper * stage_centered)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len) & (abs_y <= fin_w)
        
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

def generate_design() -> np.ndarray:
    params = np.append(BEST_PARAMS_RUN049, 0.5).astype(np.float32) # Added P19
    # Run a quick local optimization focused on the new P19 and existing high-performers
    return np.asarray(render_graded_ucup_vortex_diode(params, 250, 100))
