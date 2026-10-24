"""
Fluidic Diode Design & Continuous Optimization Module.
Optimizing vortex-locking via stage-specific phase-shifting and multi-fidelity screening.
"""

from __future__ import annotations
import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Global elite from run_049 (Di = 383.239117)
BEST_PARAMS = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0, 0.0, 0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, ry_grade, th_grade, fin_len_scale, fin_width_scale, lip_start, lip_len, lip_depth = params

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
        in_outer = (ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        r_vertex = jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2)
        in_cup = in_outer & ~((ux_rel >= 0.0) & (abs_y <= ry_inner) & (r_vertex <= 1.0))

        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        
        # Lip logic
        y_wall_exit = 1.0 - (1.0 - throat_gap) * (jnp.clip(rear_edge / nozzle_len, 0.0, 1.0) ** nozzle_pow)
        is_lip = (lip_len > 0.01) & (x >= x0 + pitch * (rear_edge + lip_start)) & (x <= x0 + pitch * (rear_edge + lip_start + lip_len)) & \
                 (abs_y >= jnp.maximum(0.0, y_wall_exit - lip_depth * (1.0 - y_wall_exit))) & (abs_y <= y_wall_exit)

        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin | is_lip).astype(jnp.float32))
    return mask

render_batch_fast_fn = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine_fn = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    best_params = BEST_PARAMS.copy()
    best_score = 383.239117
    
    # 1. Broad search on FAST_CONFIG
    pop = np.tile(best_params, (128, 1))
    pop[1:] += np.random.normal(0, 0.02, size=(127, 19))
    masks_fast = render_batch_fast_fn(jnp.asarray(pop))
    scores_fast, _, _ = evaluate_batch_fast(masks_fast, FAST_CONFIG)
    
    # 2. Refine top 8 on DEFAULT_CONFIG
    top_indices = np.argsort(scores_fast)[-8:]
    pop_fine = pop[top_indices]
    masks_fine = render_batch_fine_fn(jnp.asarray(pop_fine))
    scores_fine, _, _ = evaluate_batch_fast(masks_fine, DEFAULT_CONFIG)
    
    best_idx = int(np.argmax(scores_fine))
    if scores_fine[best_idx] > best_score:
        best_params = pop_fine[best_idx]
        best_score = float(scores_fine[best_idx])
        print(f"NEW ELITE: {best_score} | Params: {best_params}")
        
    return np.asarray(render_batch_fine_fn(jnp.asarray(best_params[None, :]))[0])
