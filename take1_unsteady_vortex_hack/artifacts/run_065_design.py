"""
Fluidic Diode Design & Continuous Optimization Module.
Optimizing vortex-locking mechanism via longitudinal fin translation and re-entrant lip interaction.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, evaluate_batch_fast

# Elite parameter vector from run_049 (Di = 383.239117)
# Added P19: fin_offset_x_factor (shift fin position relative to cavity center)
BEST_PARAMS_RUN049 = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0, 0.0, 0.0, 0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Standard 16 parameters + 4 new features
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
    
    # New features
    lip_start_off = 0.0 + 0.3 * params[16]
    lip_len = 0.0 + 0.15 * params[17]
    lip_depth = 0.0 + 0.25 * params[18]
    fin_off_x = -0.15 + 0.3 * params[19] # P19

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * (k - 1.0)), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * (k - 1.0)), 0.10, 0.36)
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch

        # Wall Nozzle
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len) & (abs_y >= y_wall)

        # U-Cup
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        in_outer = (ux_rel <= cavity_len) & (abs_y <= ry_local)
        r_vertex = jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / jnp.maximum(0.02, ry_local - wall_thick)) ** 2)
        in_cup = in_outer & ~((ux_rel >= 0.0) & (r_vertex <= 1.0))

        # Fin with P19 offset
        is_fin = (ux_rel <= fin_off_x) & (ux_rel >= fin_off_x - fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_local)

        # Lip
        lip_x_start = x0 + pitch * (nozzle_len + lip_start_off)
        lip_x_end = lip_x_start + pitch * lip_len
        is_lip = (lip_len > 1e-4) & (x >= lip_x_start) & (x <= lip_x_end) & (abs_y >= y_wall - lip_depth * (1.0 - throat_gap)) & (abs_y <= y_wall)

        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin | is_lip).astype(jnp.float32))

    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    best_params = BEST_PARAMS_RUN049.copy()
    best_score = 383.239117
    
    # Targeted Nelder-Mead-style local search
    for r in range(3):
        if time.time() - t0 > 45.0: break
        cands = [best_params.copy()]
        # Perturb 3 parameters at a time
        for _ in range(11):
            p = best_params.copy()
            idx = np.random.choice(20, 3, replace=False)
            p[idx] += np.random.normal(0, 0.015 * (0.8**r), 3)
            cands.append(np.clip(p, 0.0, 1.0))
        
        scores, _, _ = evaluate_batch_fast(render_batch_fine(jnp.array(cands)), DEFAULT_CONFIG)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best_score:
            best_score = float(scores[best_idx])
            best_params = cands[best_idx].copy()
            print(f"New Best: {best_score}")
            
    return np.asarray(render_batch_fine(jnp.array([best_params]))[0])
