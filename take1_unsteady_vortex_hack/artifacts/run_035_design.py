"""
Fluidic Diode Design & Continuous Optimization Module (run_035).

Refinement of the high-performance run_020 topology by introducing a longitudinal
asymmetry parameter for the central splitter fin to optimize vortex-locking efficiency.
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
    # params: (17,)
    # 0-13: as in run_020, 14: fin_len, 15: fin_width, 16: fin_offset (longitudinal shift)
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
    fin_offset = -0.2 + 0.4 * params[16] # New: Shift fin relative to center of cavity

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    pitch = pitch_scale / 3.0
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    for k in range(3):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        
        # Wall/Cup rendering logic (same as run_020)
        y_wall = 1.0 - (1.0 - throat_base) * ((jnp.clip(dx/nozzle_len, 0, 1)) ** nozzle_pow)
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len - undercut * (abs_y - throat_base)) & (abs_y >= y_wall)
        
        ux_rel = (dx - cup_offset) * aspect
        in_outer = (ux_rel <= cavity_len) & (abs_y <= (cup_ry_base + flare * jnp.clip(ux_rel, 0, cavity_len)))
        hollow = (ux_rel >= 0.0) & (abs_y <= (cup_ry_base - 0.05)) & (ux_rel <= cavity_len)
        
        # Fin with offset
        fin_len_actual = fin_len_scale * cavity_len
        is_fin = (ux_rel <= fin_offset) & (ux_rel >= fin_offset - fin_len_actual) & (abs_y <= fin_width_scale * 0.15)
        
        mask = jnp.maximum(mask, (wall_nozzle | (in_outer & ~hollow) | is_fin).astype(jnp.float32))
    return mask

render_batch_fine = jax.jit(jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, 300-50, 100)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    # Best params from run_020 (16 dims), padded with 0.5 for P16
    best_p = np.array([0.34585416, 0.6224917, 0.55515563, 0.30195224, 0.04744818, 0.38958547,
                       0.30690935, 0.24575749, 0.44694495, 0.5715191, 0.8370709, 0.05,
                       0.6155642, 0.36107752, 0.86073214, 0.5733315, 0.5], dtype=np.float32)
    
    pop = np.tile(best_p, (8, 1)) + np.random.normal(0, 0.05, (8, 17))
    pop[0] = best_p
    
    m_f = render_batch_fine(jnp.asarray(pop))
    f_sc, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    
    best_idx = int(np.argmax(f_sc))
    print(f"Run 035 Final: Max Di = {f_sc[best_idx]:.4f} | Best Params = {pop[best_idx]}")
    return np.asarray(m_f[best_idx])
