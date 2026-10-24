"""
Fluidic Diode Design & Continuous Optimization Module (run_097).
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Global Best from run_049 (Di = 383.239117)
BEST_PARAMS_RUN049 = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0, 0.0, 0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, \
    ry_grade, th_grade, fin_len_scale, fin_width_scale, \
    lip_start, lip_len, lip_depth = params

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
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = ((ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch))**2 + (abs_y/ry_local)**2 <= 1.0)) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06)/0.06)**2 + (abs_y/ry_inner)**2) <= 1.0)
        in_cup = in_outer & (~hollow)
        
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        
        # Lip logic
        s_exit = jnp.clip(rear_edge / nozzle_len, 0.0, 1.0)
        y_exit = 1.0 - (1.0 - throat_gap) * (s_exit ** nozzle_pow)
        is_lip = (lip_len > 1e-4) & (x >= x0 + pitch*(rear_edge + lip_start)) & (x <= x0 + pitch*(rear_edge + lip_start + lip_len)) & \
                 (abs_y >= y_exit - lip_depth * (1.0 - y_exit)) & (abs_y <= y_exit)

        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin | is_lip).astype(jnp.float32))
    return mask

render_batch = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2*DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    best_p, best_s = BEST_PARAMS_RUN049.copy(), 383.239117
    
    # Population-based search
    for r in range(3):
        pop = [best_p]
        for _ in range(15):
            p = best_p.copy()
            # Mutate with decay sigma
            sigma = 0.04 * (0.5 ** r)
            idx = np.random.choice(len(best_p), 3, replace=False)
            p[idx] = np.clip(p[idx] + np.random.normal(0, sigma, 3), 0, 1)
            pop.append(p)
        
        scores, _, _ = evaluate_batch_fast(render_batch(jnp.array(pop)), DEFAULT_CONFIG)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best_s:
            best_s, best_p = float(scores[best_idx]), pop[best_idx]
            print(f"New best: {best_s:.6f}")
        if time.time() - t0 > 40: break
    
    return np.asarray(render_batch(jnp.array([best_p]))[0])
