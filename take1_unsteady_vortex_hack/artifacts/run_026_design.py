"""
Fluidic Diode Design & Continuous Optimization Module.
Run 026: Refine the global optimum from run_020 using a focused evolutionary strategy.
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
    # params: 16 parameters from run_020 + 1 new taper parameter
    # 0-15: same as run_020, 16: fin_prow_taper
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, cavity_len, \
    wall_thick, nose_thick, pitch_scale, flare, undercut, ry_grade, th_grade, fin_len_scale, \
    fin_width_scale, fin_taper = params

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    n_stages = 3
    pitch = (0.78 + 0.16 * pitch_scale) / n_stages
    margin_x = 0.5 * (1.0 - (0.78 + 0.16 * pitch_scale))
    aspect = (nx_design * pitch) / (ny * 0.5)

    for k in range(n_stages):
        stage_centered = k - 1.0
        throat_gap = jnp.clip(0.27 + 0.32 * throat_base * (1.0 + (-0.15 + 0.30 * th_grade) * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip(0.12 + 0.18 * cup_ry_base * (1.0 + (-0.15 + 0.30 * ry_grade) * stage_centered), 0.10, 0.36)

        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        s = jnp.clip(dx / (0.16 + 0.36 * nozzle_len), 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** (0.60 + 1.10 * nozzle_pow))
        
        rear_edge = (0.16 + 0.36 * nozzle_len) - (0.00 + 0.12 * undercut) * jnp.clip((abs_y - throat_gap) / 0.1, 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        ux_rel = (dx - (0.32 + 0.38 * cup_offset)) * aspect
        ry_local = jnp.clip(cup_ry + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0.0, (0.12 + 0.58 * cavity_len)), 0.08, 0.42)
        in_outer = (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local)
        hollow = (ux_rel >= 0.0) & (abs_y <= (ry_local - (0.042 + 0.075 * wall_thick)))
        
        fin_len = (0.0 + 0.8 * fin_len_scale) * (0.12 + 0.58 * cavity_len)
        fin_w = (0.0 + 0.8 * fin_width_scale) * (ry_local - (0.042 + 0.075 * wall_thick))
        # Apply taper to upstream tip
        taper_mask = 1.0 - (fin_taper * jnp.clip((-ux_rel / fin_len), 0.0, 1.0))
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len) & (abs_y <= fin_w * taper_mask)

        mask = jnp.maximum(mask, (wall_nozzle | (in_outer & ~hollow) | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    # Warm start with best parameters from run_020
    best_p = np.array([0.3458, 0.6225, 0.5552, 0.3020, 0.0474, 0.3896, 0.3069, 0.2458, 0.4469, 0.5715, 0.8371, 0.0, 0.6156, 0.3611, 0.8607, 0.5733, 0.5], dtype=np.float32)
    
    pop = np.tile(best_p, (16, 1))
    pop[1:] += np.random.normal(0, 0.05, size=(15, 17))
    pop = np.clip(pop, 0, 1)
    
    m = render_batch_fine(jnp.asarray(pop))
    sc, _, _ = evaluate_batch_fast(m, DEFAULT_CONFIG)
    
    best_idx = int(np.argmax(sc))
    return np.asarray(m[best_idx])
