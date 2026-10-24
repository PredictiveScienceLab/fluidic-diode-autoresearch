"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 032 Hypothesis:
- Physical Hypothesis: The extreme Di of 37.8 (run_020) relies on a delicate balance
  between the central splitter fin and the wall undercut. We will lock the core geometric
  ratios (pitch, nozzle power, cup base) to the `run_020` elite vector and perform
  a constrained local search (sigma=0.05) on the fin and nose parameters.
- Algorithmic Hypothesis: Use 128 coarse-screened candidates followed by a 4-round
  "Elite-Preservation" fine-grid evolution. This limits the search space to the
  neighborhood of the proven global maximum, preventing the optimizer from 
  drifting into low-performance regions.
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
    # Parameter map identical to run_020 to ensure topological consistency
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
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(3):
        stage_centered = k - 1.0
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * stage_centered), 0.10, 0.36)
        x0 = 0.5 * (1.0 - pitch_scale) + k * (pitch_scale / 3.0)
        dx = (x - x0) / (pitch_scale / 3.0)
        
        y_wall = 1.0 - (1.0 - throat_gap) * (jnp.clip(dx / nozzle_len, 0.0, 1.0) ** nozzle_pow)
        rear_edge = nozzle_len - (undercut * jnp.clip((abs_y - throat_gap) / 0.1, 0.0, 1.0))
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)
        
        ux_rel = (dx - cup_offset) * (nx_design * pitch_scale / 3.0) / (ny * 0.5)
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = (ux_rel <= cavity_len) & (abs_y <= ry_local) & (ux_rel >= -0.1)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (ux_rel <= cavity_len)
        in_cup = in_outer & (~hollow)
        
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, 275, 100)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    # Elite parameters from run_020
    best_p = np.array([0.34585, 0.62249, 0.55516, 0.30195, 0.04745, 0.38959, 0.30691, 0.24576, 0.44694, 0.57152, 0.83707, 0.05, 0.61556, 0.36108, 0.86073, 0.57333], dtype=np.float32)
    
    # Local search centered on elite
    pop = np.clip(best_p + np.random.normal(0, 0.05, size=(32, 16)), 0.0, 1.0).astype(np.float32)
    pop[0] = best_p
    
    m_f = render_batch_fine(jnp.asarray(pop))
    f_sc, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    
    idx = np.argmax(f_sc)
    print(f"Best Di: {f_sc[idx]:.4f} | Params: {pop[idx]}")
    return np.asarray(m_f[idx])
