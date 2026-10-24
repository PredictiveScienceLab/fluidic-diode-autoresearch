"""
Fluidic Diode Design & Continuous Optimization Module (run_108).
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, SimConfig, evaluate_batch_fast

# Winning parameters from run_049 (Di = 383.239117)
BEST_PARAMS_RUN049 = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315, 0.0, 0.0, 0.0
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
        in_outer_nose = (ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0)
        in_outer_body = (ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local)
        in_cup = (in_outer_nose | in_outer_body) & ~((ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2) <= 1.0))

        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        stage = (wall_nozzle | in_cup | is_fin).astype(jnp.float32)
        mask = jnp.maximum(mask, stage)
    return mask

render_batch_fast_fn = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine_fn = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    rng = np.random.default_rng(42)
    best_params = BEST_PARAMS_RUN049.copy()
    
    # 1. Multi-fidelity screening: 128 candidates on FAST_CONFIG
    pop = [best_params] + [np.clip(best_params + rng.normal(0, 0.05, 19), 0, 1) for _ in range(127)]
    m_fast = render_batch_fast_fn(jnp.asarray(pop))
    scores_fast, _, _ = evaluate_batch_fast(m_fast, FAST_CONFIG)
    
    # 2. Refinement: Top 8 candidates on DEFAULT_CONFIG
    top_indices = np.argsort(scores_fast)[-8:]
    pop_fine = [pop[i] for i in top_indices]
    m_fine = render_batch_fine_fn(jnp.asarray(pop_fine))
    scores_fine, _, _ = evaluate_batch_fast(m_fine, DEFAULT_CONFIG)
    
    best_idx = int(np.argmax(scores_fine))
    print(f"[run_108] Best Fine Di: {scores_fine[best_idx]:.6f}")
    return np.asarray(m_fine[best_idx])
