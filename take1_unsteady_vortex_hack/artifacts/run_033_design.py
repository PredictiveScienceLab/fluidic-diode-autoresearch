"""
Fluidic Diode Design & Continuous Optimization Module (run_033).
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

MATCHED_FAST_CONFIG = SimConfig(
    nx=180, ny=60, pad_x=15, nu=0.0012, d_rho=0.03, n_steps=3200,
    avg_window=800, max_solid_fraction=0.60, min_open_ratio=0.12, q_open_ref=0.1585,
)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
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

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages, pitch = 3, pitch_scale / 3.0
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
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / 0.1, 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = ((ux_rel <= 0.0) & ((ux_rel/jnp.maximum(0.06, nose_thick*prow_stretch))**2 + (abs_y/ry_local)**2 <= 1.0)) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel-0.06)/0.06)**2 + (abs_y/ry_inner)**2) <= 1.0)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale*cavity_len) & (abs_y <= fin_width_scale*ry_inner)
        mask = jnp.maximum(mask, (wall_nozzle | (in_outer & (~hollow)) | is_fin).astype(jnp.float32))
    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, 150, 60)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, 250, 100)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(505)
    best_params = np.array([0.3458, 0.6225, 0.5552, 0.3020, 0.0474, 0.3896, 0.3069, 0.2458, 0.4469, 0.5715, 0.8371, 0.0, 0.6156, 0.3611, 0.8607, 0.5733], dtype=np.float32)
    
    pop = np.clip(best_params + rng.normal(0, 0.05, (128, 16)), 0, 1).astype(np.float32)
    pop[0] = best_params
    
    sc = np.concatenate([np.asarray(evaluate_batch_fast(render_batch_coarse(jnp.asarray(pop[i:i+32])), MATCHED_FAST_CONFIG)[0]) for i in range(0, 128, 32)])
    top_params = pop[np.argsort(sc)[-8:]]
    
    m_f = render_batch_fine(jnp.asarray(top_params))
    f_sc, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    elite = top_params[np.argsort(f_sc)[-3:]]
    scores = np.sort(f_sc)[-3:]
    
    for r in range(4):
        mutants = np.clip(elite[None, rng.choice(3, 24)] + rng.normal(0, 0.05*(0.8**r), (24, 16)), 0, 1)
        m_m = render_batch_fine(jnp.asarray(mutants))
        sc_m, _, _ = evaluate_batch_fast(m_m, DEFAULT_CONFIG)
        elite = np.concatenate([elite, mutants])[np.argsort(np.concatenate([scores, sc_m]))[-3:]]
        scores = np.sort(np.concatenate([scores, sc_m]))[-3:]
    
    print(f"Final Best Score: {scores[-1]:.4f} | Params: {elite[-1]}")
    return np.asarray(render_batch_fine(jnp.asarray(elite[-1][None]))[0])
