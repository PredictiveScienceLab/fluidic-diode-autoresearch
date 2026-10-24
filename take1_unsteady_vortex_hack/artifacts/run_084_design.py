"""
Fluidic Diode Design & Continuous Optimization Module.
Optimizing vortex-locking efficiency via 20-parameter manifold refinement.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Elite parameters from run_049 (Di = 383.239117)
BEST_PARAMS_ELITE = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0, 0.0, 0.0, 0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Baseline 16 parameters + 4 new control parameters
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, ry_grade, \
    th_grade, fin_len_scale, fin_width_scale = params[:16]
    
    lip_start, lip_len, lip_depth, fin_taper = params[16:20]

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
        in_outer = (ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch))**2 + (abs_y / ry_local)**2 <= 1.0) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06)/0.06)**2 + (abs_y/ry_inner)**2) <= 1.0)
        in_cup = in_outer & (~hollow)

        # Gradient-tapered fin
        fin_w = fin_width_scale * ry_inner * (1.0 - fin_taper * jnp.clip(ux_rel + fin_len_scale, 0.0, 1.0))
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale) & (abs_y <= fin_w)

        stage = (wall_nozzle | in_cup | is_fin).astype(jnp.float32)
        mask = jnp.maximum(mask, stage)
    return mask

render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)

def generate_design() -> np.ndarray:
    # Use the best known elite parameters as the starting point
    best_params = BEST_PARAMS_ELITE.copy()
    best_score = 383.239117
    
    # Perform a focused local search
    rng = np.random.default_rng(42)
    cands = [best_params]
    for _ in range(15):
        p = best_params.copy()
        # Perturb subset of parameters
        idx = rng.choice(20, size=2, replace=False)
        p[idx] = np.clip(p[idx] + rng.normal(0, 0.015, size=2), 0.0, 1.0)
        cands.append(p)
    
    m_f = render_batch_fine(jnp.stack(cands))
    scores, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    
    best_idx = int(np.argmax(scores))
    if scores[best_idx] > best_score:
        return np.asarray(m_f[best_idx])
    return np.asarray(m_f[0])
