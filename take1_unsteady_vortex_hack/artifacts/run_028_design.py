"""
Fluidic Diode Design & Continuous Optimization Module (run_028)

Hypothesis:
The record-breaking Di=37.8 (run_020) was achieved by a delicate synergy between 
the central splitter fin and the wall undercut. Subsequent runs failed due to 
over-aggressive optimization of the fin geometry, which destroyed the forward-flow 
conductance. We now perform a constrained optimization: we fix the core structural 
topology of the best-performing run_020 and apply a tighter mutation range (sigma=0.05) 
focused on fine-tuning the throat gaps and fin-to-cup positioning to maintain the 
vortex-locking effect while recovering forward-flow conductance.
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
    pitch = (0.78 + 0.16 * pitch_scale) / n_stages
    margin_x = 0.5 * (1.0 - (0.78 + 0.16 * pitch_scale))
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        stage_centered = k - 1.0
        throat_gap = jnp.clip((0.27 + 0.32 * throat_base) * (1.0 + (-0.15 + 0.30 * th_grade) * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1.0 + (-0.15 + 0.30 * ry_grade) * stage_centered), 0.10, 0.36)
        
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        s = jnp.clip(dx / (0.16 + 0.36 * nozzle_len), 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** (0.60 + 1.10 * nozzle_pow))
        
        rear_edge = (0.16 + 0.36 * nozzle_len) - (0.0 + 0.12 * undercut) * jnp.clip((abs_y - throat_gap) / 0.1, 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        ux_rel = (dx - (0.32 + 0.38 * cup_offset)) * aspect
        ry_local = jnp.clip(cup_ry + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0.0, (0.12 + 0.58 * cavity_len)), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - (0.042 + 0.075 * wall_thick))
        in_outer = (ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, (0.10 + 0.24 * nose_thick) * (0.55 + 1.20 * prow_stretch)))**2 + (abs_y / ry_local)**2 <= 1.0) | \
                   ((ux_rel > 0.0) & (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06)**2 + (abs_y / ry_inner)**2) <= 1.0)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -(0.0 + 0.8 * fin_len_scale) * (0.12 + 0.58 * cavity_len)) & (abs_y <= (0.0 + 0.8 * fin_width_scale) * ry_inner)
        mask = jnp.maximum(mask, (wall_nozzle | (in_outer & (~hollow)) | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    # Best params from run_020
    best_p = np.array([0.34585416, 0.6224917, 0.55515563, 0.30195224, 0.04744818, 0.38958547, 0.30690935, 0.24575749, 0.44694495, 0.5715191, 0.8370709, 0.5, 0.6155642, 0.36107752, 0.86073214, 0.5733315], dtype=np.float32)
    
    # Generate population centered on best_p
    rng = np.random.default_rng(42)
    pop = np.clip(best_p + rng.normal(0, 0.05, (32, 16)), 0, 1).astype(np.float32)
    pop[0] = best_p
    
    m_f = render_batch_fine(jnp.asarray(pop))
    scores, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    
    best_idx = int(np.argmax(scores))
    print(f"Run 028: Best Di = {scores[best_idx]:.4f}, Best Params = {pop[best_idx]}")
    return np.asarray(m_f[best_idx])
