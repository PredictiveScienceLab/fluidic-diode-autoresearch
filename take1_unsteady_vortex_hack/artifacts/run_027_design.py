"""
Fluidic Diode Design & Continuous Optimization Module.

Run 027 Strategy:
- Physical Hypothesis: The "double-blockage" (vortex trap + splitter fin) is highly sensitive.
  We will fix the central fin to be always present (P14, P15 > 0.2) and the wall undercut
  to be significant (P11 > 0.1) to prevent the optimizer from drifting into "open-pipe"
  local optima where diodicity collapses.
- Algorithmic: Use a "Survivor-Only" evolution. Start with the best known vector from
  run_020 and perform local hill-climbing + Gaussian mutation to refine the geometry
  without losing the high-resistance vortex trapping mechanism.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Configuration for multi-fidelity screening
MATCHED_FAST_CONFIG = SimConfig(
    nx=180, ny=60, pad_x=15, nu=0.0012, d_rho=0.03,
    n_steps=3200, avg_window=800, max_solid_fraction=0.60,
    min_open_ratio=0.12, q_open_ref=0.1585,
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
    undercut = 0.10 + 0.30 * params[11] # Force active undercut
    ry_grade = -0.15 + 0.30 * params[12]
    th_grade = -0.15 + 0.30 * params[13]
    fin_len_scale = 0.2 + 0.6 * params[14] # Force active fin
    fin_width_scale = 0.2 + 0.6 * params[15]

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
        in_outer = (ux_rel <= 0.0) & ((ux_rel/(nose_thick*prow_stretch))**2 + (abs_y/ry_local)**2 <= 1.0) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel-0.06)/0.06)**2 + (abs_y/ry_inner)**2) <= 1.0)
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_scale * cavity_len) & (abs_y <= fin_width_scale * ry_inner)
        mask = jnp.maximum(mask, (wall_nozzle | (in_outer & ~hollow) | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    # Warm start from run_020
    best_params = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    rng = np.random.default_rng(42)
    
    # Elitist evolution: 3 rounds of 16 candidates
    population = [best_params]
    for _ in range(15):
        population.append(np.clip(best_params + rng.normal(0, 0.05, 16), 0.0, 1.0))
    
    elite_params = best_params
    elite_score = 0.0
    
    for r in range(3):
        m = render_batch_fine(jnp.asarray(np.stack(population)))
        scores, _, _ = evaluate_batch_fast(m, DEFAULT_CONFIG)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > elite_score:
            elite_score = float(scores[best_idx])
            elite_params = population[best_idx]
            print(f"Round {r} new best: {elite_score:.4f} | Params: {elite_params}")
        
        population = [elite_params]
        for _ in range(15):
            population.append(np.clip(elite_params + rng.normal(0, 0.02 * (0.5**r), 16), 0.0, 1.0))

    return np.asarray(render_batch_fine(elite_params[None, :])[0])
