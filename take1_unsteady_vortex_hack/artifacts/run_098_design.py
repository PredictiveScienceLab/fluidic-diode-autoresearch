"""
Fluidic Diode Design & Continuous Optimization Module (run_098).
Focus: Stochastic coordinate-ascent on sensitivity-ranked parameters.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# The 19-parameter vector from run_049 (Di = 383.239117)
BEST_PARAMS = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0, 0.0, 0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, ry_grade, \
    th_grade, fin_len_scale, fin_width_scale, lip_start, lip_len, lip_depth = params

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    pitch = (0.78 + 0.16 * pitch_scale) / n_stages
    margin_x = 0.5 * (1.0 - (0.78 + 0.16 * pitch_scale))
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        stage_centered = k - 1.0
        throat_gap = jnp.clip((0.27 + 0.32 * throat_base) * (1.0 + (-0.15 + 0.30 * th_grade) * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1.0 + (-0.15 + 0.30 * ry_grade) * stage_centered), 0.10, 0.36)
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch

        s = jnp.clip(dx / (0.16 + 0.36 * nozzle_len), 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** (0.60 + 1.10 * nozzle_pow))
        rear_edge = (0.16 + 0.36 * nozzle_len) - (0.0 + 0.12 * undercut) * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        ux_rel = (dx - (0.32 + 0.38 * cup_offset)) * aspect
        ry_local = jnp.clip(cup_ry + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0.0, (0.12 + 0.58 * cavity_len)), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - (0.042 + 0.075 * wall_thick))

        in_outer = ((ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, (0.10 + 0.24 * nose_thick) * (0.55 + 1.20 * prow_stretch))) ** 2 + (abs_y / ry_local) ** 2 <= 1.0)) | \
                   ((ux_rel > 0.0) & (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2) <= 1.0)
        in_cup = in_outer & (~hollow)

        is_fin = (ux_rel <= 0.0) & (ux_rel >= -(0.0 + 0.8 * fin_len_scale) * (0.12 + 0.58 * cavity_len)) & (abs_y <= (0.0 + 0.8 * fin_width_scale) * ry_inner)
        
        # Lip logic active with >0 params
        is_lip = (lip_len > 1e-4) & (x >= x0 + pitch * (rear_edge + 0.3 * lip_start)) & \
                 (x <= x0 + pitch * (rear_edge + 0.3 * lip_start + 0.15 * lip_len)) & \
                 (abs_y >= y_wall - 0.25 * lip_depth * (1.0 - y_wall)) & (abs_y <= y_wall)

        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin | is_lip).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, 250, 100)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    best_params, best_score = BEST_PARAMS.copy(), 383.239117
    
    # Sensitivity Indices: throat_base (1), fin_len (14), fin_width (15)
    target_idx = [1, 14, 15]
    
    for r in range(5):
        if time.time() - t0 > 40.0: break
        
        # Stochastic search: perturb only sensitivity-ranked params
        cands = [best_params.copy()]
        for _ in range(7):
            p = best_params.copy()
            p[target_idx] += np.random.normal(0, 0.02 * (0.9**r), size=3).astype(np.float32)
            cands.append(np.clip(p, 0.0, 1.0))
            
        masks = render_batch_fine(jnp.stack(cands))
        scores, _, _ = evaluate_batch_fast(masks, DEFAULT_CONFIG)
        
        if np.max(scores) > best_score:
            best_score = float(np.max(scores))
            best_params = cands[int(np.argmax(scores))]
            print(f"[run_098] New Best: {best_score:.6f} | Params: {best_params.tolist()}")
            
    return np.asarray(render_batch_fine(best_params[None, :])[0])
