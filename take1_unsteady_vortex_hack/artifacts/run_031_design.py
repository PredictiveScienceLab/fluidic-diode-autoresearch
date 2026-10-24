"""
Fluidic Diode Design & Continuous Optimization Module (run_031).

Hypothesis: The high performance of run_020 (~37.8) relied on a specific geometric
alignment where the central splitter fin and the U-cup cavity work in concert
to trap reverse-flow vortices. We will fix the global structure and perform
focused, small-sigma local optimization on the cup nose and fin width to
maximize the "pinching" effect of the reverse jet.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

MATCHED_FAST_CONFIG = SimConfig(
    nx=180,
    ny=60,
    pad_x=15,
    nu=0.0012,
    d_rho=0.03,
    n_steps=3200,
    avg_window=800,
    max_solid_fraction=0.60,
    min_open_ratio=0.12,
    q_open_ref=0.1585,
)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(
    params: jnp.ndarray,
    nx_design: int,
    ny: int,
) -> jnp.ndarray:
    # Parameter mapping remains consistent with run_020/030
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
        rear_edge = nozzle_len - (undercut * jnp.clip((abs_y - throat_gap) / 0.2, 0.0, 1.0))
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = (ux_rel <= cavity_len) & (ux_rel >= -nose_thick) & (abs_y <= ry_local)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (ux_rel <= cavity_len)
        in_cup = in_outer & (~hollow)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    # Elite parameters from run_020
    best_params = np.array([0.34585416, 0.6224917, 0.55515563, 0.30195224, 0.04744818, 0.38958547, 
                            0.30690935, 0.24575749, 0.44694495, 0.5715191, 0.8370709, 0.0, 
                            0.6155642, 0.36107752, 0.86073214, 0.5733315], dtype=np.float32)

    # Local search centered on nose (P8) and fin (P14, P15)
    pop = np.tile(best_params, (32, 1))
    rng = np.random.default_rng(42)
    # Apply small perturbations to indices 8, 14, 15
    noise = rng.normal(0, 0.05, (32, 16))
    noise[:, [0,1,2,3,4,5,6,7,9,10,11,12,13]] = 0 # Lock most params
    pop = np.clip(pop + noise, 0.0, 1.0)
    
    m_f = render_batch_fine(jnp.asarray(pop))
    f_sc, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    best_idx = int(np.argmax(f_sc))
    
    print(f"Run 031 Best Score: {f_sc[best_idx]:.4f} | Params: {pop[best_idx]}")
    return np.asarray(m_f[best_idx])
