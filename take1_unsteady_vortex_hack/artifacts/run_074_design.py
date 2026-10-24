"""
Fluidic Diode Design & Continuous Optimization Module.
Focus: Introducing fin_tilt_angle to break spanwise symmetry of the vortex-locking mechanism.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, evaluate_batch_fast

# Optimal parameter vector from run_049 (Di = 383.239117)
BEST_PARAMS_RUN049 = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0, 0.0, 0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, cavity_len, wall_thick, \
    nose_thick, pitch_scale, flare, undercut, ry_grade, th_grade, fin_len_scale, fin_width_scale, \
    lip_start, lip_len, lip_depth = params

    fin_tilt = 0.0 # P19: New feature
    if len(params) > 19:
        fin_tilt = params[19]

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
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)

        in_outer = (ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        r_vertex = jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2)
        in_cup = in_outer & ~((ux_rel >= 0.0) & (abs_y <= ry_inner) & (r_vertex <= 1.0))

        # Splitter fin with tilt
        y_fin_center = fin_tilt * (ux_rel + fin_len_scale * cavity_len)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (jnp.abs(y - y_fin_center) <= fin_width_scale * ry_inner)

        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, 250, 100)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    # Start from run_049 elite + 1 new parameter (fin_tilt)
    best_params = np.append(BEST_PARAMS_RUN049, 0.0)
    best_score = 383.239117
    
    # Simple coordinate-wise descent on the new parameter
    for tilt in np.linspace(-0.2, 0.2, 8):
        p = best_params.copy()
        p[19] = tilt
        m = render_batch_fine(p[None, :])
        sc, _, _ = evaluate_batch_fast(m, DEFAULT_CONFIG)
        if sc[0] > best_score:
            best_score = float(sc[0])
            best_params = p.copy()
            print(f"NEW BEST: {best_score}")
            
    return np.asarray(render_batch_fine(best_params[None, :])[0])
