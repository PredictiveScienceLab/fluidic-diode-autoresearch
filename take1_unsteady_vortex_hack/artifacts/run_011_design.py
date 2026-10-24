"""
Fluidic Diode Design & Continuous Optimization Module (run_011).

Run 011 Hypothesis:
- Physical: Add 'fin_skew' to the central splitter fin. By offsetting the fin 
  longitudinally within the cavity, we break the symmetry of the vortex 
  trapping mechanism, aiming to force a more violent impingement during reverse flow.
- Algorithmic: Use a hybrid schedule: 10 generations of 32-pop FAST_CONFIG 
  (coarse screening) followed by 5 generations of 16-pop DEFAULT_CONFIG 
  (fine refinement) to maximize diodicity within the 45s budget.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Coarse configuration for rapid population screening
FAST_CONFIG = SimConfig(
    nx=180, ny=60, pad_x=15, nu=0.0012, d_rho=0.03, n_steps=2000,
    avg_window=500, max_solid_fraction=0.60, min_open_ratio=0.12, q_open_ref=0.1585,
)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Params (18): 0-13 (base), 14-15 (fin), 16 (tip_lip_angle), 17 (fin_skew)
    """
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, \
    ry_grade, th_grade, fin_len_s, fin_width_s, lip_angle, fin_skew = params

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    pitch = (0.78 + 0.16 * pitch_scale) / 3
    margin_x = 0.5 * (1.0 - (0.78 + 0.16 * pitch_scale))
    aspect = (nx_design * pitch) / (ny * 0.5)

    for k in range(3):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        ux_rel = (dx - (0.32 + 0.38 * cup_offset)) * aspect
        
        y_wall = 1.0 - (1.0 - jnp.clip(0.27 + 0.32 * throat_base * (1 + (-0.15 + 0.3*th_grade)*(k-1)), 0.22, 0.68)) * (jnp.clip(dx/ (0.16 + 0.36 * nozzle_len), 0, 1) ** (0.6 + 1.1 * nozzle_pow))
        wall_nozzle = (dx >= 0.0) & (dx <= (0.16 + 0.36 * nozzle_len) - (0.12 * undercut) * jnp.clip((abs_y - 0.2)/0.5, 0, 1)) & (abs_y >= y_wall)
        
        ry_local = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1 + (-0.15 + 0.3*ry_grade)*(k-1)) + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0, 0.7), 0.08, 0.42)
        in_cup = (ux_rel > 0) & (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local) & \
                 ~((ux_rel > (0.12 + 0.58 * cavity_len) - 0.1) & (abs_y > ry_local - (0.042 + 0.075 * wall_thick) - lip_angle*0.1))
        
        # Fin with skew
        ux_fin = ux_rel - (-0.2 + 0.4 * fin_skew)
        is_fin = (ux_fin <= 0.0) & (ux_fin >= -(0.8 * fin_len_s) * (0.12 + 0.58 * cavity_len)) & (abs_y <= (0.8 * fin_width_s) * (ry_local - (0.042 + 0.075 * wall_thick)))
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))
render_batch_fast = jax.jit(jax.vmap(lambda p: render_diode(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(505)
    best_p = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    
    pop = np.clip(best_p + rng.normal(0, 0.1, (32, 18)), 0, 1).astype(np.float32)
    pop[0] = best_p
    
    # Coarse Screening
    for r in range(10):
        m = render_batch_fast(jnp.asarray(pop))
        sc, _, _ = evaluate_batch_fast(m, FAST_CONFIG)
        idx = np.argsort(sc)[-8:]
        pop = np.clip(pop[idx[rng.integers(0, 8, 32)]] + rng.normal(0, 0.05, (32, 18)), 0, 1)
        
    # Fine Refinement
    for r in range(5):
        m = render_batch_fine(jnp.asarray(pop[:16]))
        sc, _, _ = evaluate_batch_fast(m, DEFAULT_CONFIG)
        idx = np.argsort(sc)[-8:]
        pop[:16] = np.clip(pop[idx[rng.integers(0, 8, 16)]] + rng.normal(0, 0.02, (16, 18)), 0, 1)
        
    return render_diode(pop[idx[-1]], DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)
