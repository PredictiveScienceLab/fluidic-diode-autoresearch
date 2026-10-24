"""
Fluidic Diode Design & Continuous Optimization Module (run_050).
Focus: Optimization of stage axial spacing and vortex synchronization.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Elite parameters from run_049 (Di = 383.239)
# P0-P15: Original, P16-P18: Lip params, P19: New fin_pitch_bias
BEST_PARAMS_RUN049 = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.30195224, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.86073214, 0.5733315,  0.05, 0.08, 0.12, 0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # Unpack parameters
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, \
    ry_grade, th_grade, fin_len_scale, fin_width_scale, \
    lip_start, lip_len, lip_depth, fin_pitch_bias = params

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    
    n_stages = 3
    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    # Stage-specific centers with bias
    stage_centers = jnp.array([-1.0, 0.0, 1.0]) + jnp.array([-fin_pitch_bias, 0.0, fin_pitch_bias])
    
    for k in range(n_stages):
        stage_centered = stage_centers[k]
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * stage_centered * 0.5), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * stage_centered * 0.5), 0.10, 0.36)
        x0 = margin_x + (k / (n_stages - 1)) * pitch_scale if n_stages > 1 else margin_x
        dx = (x - x0) / pitch

        # Nozzle
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        # U-Cup
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = ((ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0)) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2) <= 1.0)
        in_cup = in_outer & (~hollow)

        # Splitter & Lip
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        s_nozzle_exit = jnp.clip(rear_edge / nozzle_len, 0.0, 1.0)
        y_w = 1.0 - (1.0 - throat_gap) * (s_nozzle_exit ** nozzle_pow)
        is_lip = (lip_len > 1e-6) & (x >= x0 + pitch * (rear_edge + lip_start)) & (x <= x0 + pitch * (rear_edge + lip_start + lip_len)) & \
                 (abs_y >= jnp.maximum(0.0, y_w - lip_depth * (1.0 - y_w))) & (abs_y <= y_w)

        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin | is_lip).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, 250, 100)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    best_params = BEST_PARAMS_RUN049.copy()
    best_score = 383.239
    
    # Focused local search
    for r in range(3):
        if time.time() - t0 > 45: break
        cands = [best_params]
        for _ in range(15):
            p = best_params.copy()
            idx = np.random.choice(len(p), 3, replace=False)
            p[idx] += np.random.normal(0, 0.01, 3)
            cands.append(np.clip(p, 0, 1))
        
        scores, _, _ = evaluate_batch_fast(render_batch_fine(jnp.array(cands)), DEFAULT_CONFIG)
        if np.max(scores) > best_score:
            best_score = float(np.max(scores))
            best_params = cands[int(np.argmax(scores))]
            
    return np.asarray(render_batch_fine(best_params[None])[0])
