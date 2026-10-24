"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 038 Hypothesis:
Stabilize the "double-blockage" vortex-trap mechanism from `run_020` by fixing the 
high-performing geometry as the primary seed and performing a tightly constrained, 
low-sigma local search to avoid the "collapse" into low-diodicity configurations.
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
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, cavity_len, \
    wall_thick, nose_thick, pitch_scale, flare, undercut, ry_grade, th_grade, fin_len_scale, fin_width_scale = params

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
        rear_edge = nozzle_len - (undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0))
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = ((ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0)) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2) <= 1.0)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        mask = jnp.maximum(mask, (wall_nozzle | (in_outer & (~hollow)) | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    # Elite parameters from run_020
    best_params = np.array([0.4578, 0.5482, 0.7231, 0.3842, 0.1845, 0.7123, 0.3214, 0.0645, 0.2134, 0.8543, 0.1245, 0.0892, 0.0234, -0.0512, 0.5123, 0.4123], dtype=np.float32)
    
    rng = np.random.default_rng(505)
    pop = np.clip(best_params + rng.normal(0.0, 0.02, size=(8, 16)), 0.0, 1.0).astype(np.float32)
    pop[0] = best_params # Keep elite

    m_f = render_batch_fine(jnp.asarray(pop))
    f_sc, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    
    best_idx = int(np.argmax(f_sc))
    best_fine_score = float(f_sc[best_idx])
    best_fine_params = pop[best_idx]
    
    print(f"Initial Best Score: {best_fine_score:.4f} | Params: {best_fine_params}")
    return np.asarray(render_batch_fine(jnp.asarray(best_fine_params[None, :]))[0])
