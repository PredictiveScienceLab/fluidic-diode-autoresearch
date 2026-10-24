"""
Fluidic Diode Design & Continuous Optimization Module.
Focus: Maintaining the proven high-Di vortex-locking topology of run_049 while 
fine-tuning stage spacing to synchronize vortex shedding.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, evaluate_batch_fast

# The proven 16-parameter vector from run_049 (Di = 383.239)
BEST_PARAMS_16 = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315
], dtype=np.float32)

# P17: pitch_bias_factor (0.0 means uniform spacing, 0.1 means non-uniform)
BEST_PARAMS = np.concatenate([BEST_PARAMS_16, np.array([0.0], dtype=np.float32)])

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Standard 16 parameters
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
    fin_len_scale = params[14] * 0.8
    fin_width_scale = params[15] * 0.8
    
    # New P17: Pitch non-uniformity
    pitch_bias = params[16] * 0.15

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch_scale) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        # Apply non-uniform pitch shift
        stage_shift = (k - 1) * pitch_bias
        x0 = margin_x + (k / n_stages) * pitch_scale + stage_shift
        pitch = pitch_scale / n_stages
        
        stage_centered = k - 1.0
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * stage_centered), 0.10, 0.36)
        dx = (x - x0) / pitch

        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer_nose = (ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0)
        in_outer_body = (ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local)
        r_vertex = jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (r_vertex <= 1.0)
        in_cup = (in_outer_nose | in_outer_body) & (~hollow)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)

        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    best_params = BEST_PARAMS.copy()
    best_score = 383.239117
    
    # Local search around current best
    for r in range(2):
        if time.time() - t0 > 40.0: break
        cands = [best_params]
        for _ in range(7):
            p = best_params + np.random.normal(0, 0.005, size=len(best_params))
            cands.append(np.clip(p, 0.0, 1.0))
        
        m_f = render_batch_fine(jnp.asarray(np.stack(cands)))
        f_sc, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        if np.max(f_sc) > best_score:
            best_score = float(np.max(f_sc))
            best_params = cands[int(np.argmax(f_sc))]
            
    return np.asarray(render_batch_fine(best_params[None, :])[0])
