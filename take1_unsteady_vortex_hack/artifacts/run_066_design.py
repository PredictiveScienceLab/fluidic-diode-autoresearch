"""
Fluidic Diode Design & Continuous Optimization Module.
Focus: Non-uniform stage spacing to optimize vortex-locking phase-alignment.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, evaluate_batch_fast

# The winning parameter vector from run_049 (Di = 383.239117)
BEST_PARAMS_RUN049 = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0,        0.0,        0.0
], dtype=np.float32)

# P19: stage_spacing_bias (new)
BEST_PARAMS = np.concatenate([BEST_PARAMS_RUN049, np.array([0.0], dtype=np.float32)])

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, ry_grade, \
    th_grade, fin_len_scale, fin_width_scale, lip_start, lip_len, lip_depth, \
    spacing_bias = params

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    margin_x = 0.5 * (1.0 - pitch_scale)
    
    # Calculate non-uniform stage centers
    # spacing_bias shifts the relative distance between stages
    stage_centers = jnp.array([
        margin_x + (pitch_scale/3.0) * (1.0 - spacing_bias),
        margin_x + (pitch_scale/3.0) * 3.0,
        margin_x + (pitch_scale/3.0) * (5.0 + spacing_bias)
    ])

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * (k-1)), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * (k-1)), 0.10, 0.36)
        x0 = stage_centers[k]
        dx = (x - x0) / (pitch_scale / n_stages)

        # Geometry remains identical to run_049 for backward compatibility
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len) & (abs_y >= (1.0 - (1.0 - throat_gap) * (jnp.clip(dx / nozzle_len, 0.0, 1.0) ** nozzle_pow)))
        
        ux_rel = (dx - cup_offset) * (nx_design * pitch_scale / (ny * 0.5))
        in_outer = (ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch))**2 + (abs_y / cup_ry)**2 <= 1.0)
        in_cup = in_outer & ~((ux_rel >= 0.0) & (abs_y <= (cup_ry - wall_thick)) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06)/0.06)**2 + (abs_y/(cup_ry-wall_thick))**2) <= 1.0))
        
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * (cup_ry - wall_thick))
        
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

render_batch = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, 250, 100)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    best_params = BEST_PARAMS.copy()
    best_score = 383.239117
    
    # Nelder-Mead-lite on the new parameter space
    for r in range(3):
        if time.time() - t0 > 40.0: break
        cands = [best_params]
        for _ in range(7):
            p = best_params.copy()
            idx = np.random.choice(len(p), size=3, replace=False)
            p[idx] += np.random.normal(0, 0.015, size=3)
            cands.append(np.clip(p, 0.0, 1.0))
        
        scores, _, _ = evaluate_batch_fast(render_batch(jnp.stack(cands)), DEFAULT_CONFIG)
        if np.max(scores) > best_score:
            best_score = float(np.max(scores))
            best_params = cands[int(np.argmax(scores))]
            print(f"New best: {best_score}")
            
    print(f"Final Params: {best_params.tolist()}")
    return np.asarray(render_batch(jnp.stack([best_params]))[0])
