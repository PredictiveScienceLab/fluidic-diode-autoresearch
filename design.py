"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

This is the ONLY file the autoresearch agent edits.
Must expose:
    generate_design() -> np.ndarray of shape (250, 100)

The returned array represents the solid obstacle layout in the design region
(values >= 0.5 are treated as solid wall, values < 0.5 as open fluid channel).
Compute budget for `generate_design()`: 5.0 minutes (300 seconds).

Baseline Strategy (Run 0):
- Parameterizes a 3-stage conical sawtooth nozzle-diffuser cascade using
  continuous geometric parameters:
    [ramp_length, throat_half_gap, lip_thickness, stage_spacing]
- Runs a fast local search (Nelder-Mead / coordinate search) on `FAST_CONFIG`
  from `simulator.py` to tune those continuous parameters before rendering
  the final high-resolution (250, 100) binary mask.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from simulator import DEFAULT_CONFIG, FAST_CONFIG, build_full_domain_mask, run_directional_lbm


def render_sawtooth_nozzle(
    params: np.ndarray | jnp.ndarray,
    nx_design: int,
    ny: int,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Render a 3-stage sawtooth nozzle-diffuser geometry on a grid of size (nx_design, ny).

    Continuous parameters (normalized to [0, 1]):
      params[0]: ramp_length_frac  (length of each converging ramp relative to stage pitch)
      params[1]: throat_gap_frac   (half-height of open central throat relative to ny/2)
      params[2]: ramp_curvature    (power-law exponent of converging ramp profile)
    """
    ramp_frac = jnp.clip(params[0], 0.20, 0.85)
    throat_frac = jnp.clip(params[1], 0.22, 0.65)
    exponent = jnp.clip(params[2], 0.50, 2.50)

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)

    n_stages = 3
    pitch = 0.84 / n_stages
    margin_x = 0.08

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = margin_x + k * pitch
        L = ramp_frac * pitch
        dx = x - x0
        # Normalized position along the ramp in [0, 1]
        s = jnp.clip(dx / (L + 1e-6), 0.0, 1.0)
        # Wall boundary y_wall(s) shrinks from 1.0 at s=0 down to throat_frac at s=1
        y_wall = 1.0 - (1.0 - throat_frac) * (s ** exponent)

        in_stage = jax.nn.sigmoid(temperature * (dx * (L - dx) * 100.0))
        in_wall = jax.nn.sigmoid(temperature * ((abs_y - y_wall) * 25.0))
        stage_mask = in_stage * in_wall
        mask = jnp.maximum(mask, stage_mask)

    return mask


def evaluate_fast_surrogate(params: np.ndarray) -> float:
    """Objective function on the fast coarse LBM grid (minimizes -diodicity)."""
    nx_c = FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x
    ny_c = FAST_CONFIG.ny
    mask_c = render_sawtooth_nozzle(jnp.asarray(params, dtype=jnp.float32), nx_c, ny_c, temperature=40.0)
    binary_c = (mask_c >= 0.5).astype(jnp.float32)

    solid_frac = float(jnp.mean(binary_c))
    if solid_frac > FAST_CONFIG.max_solid_fraction:
        return 0.0

    full_c = build_full_domain_mask(binary_c, FAST_CONFIG)
    q_fwd = float(run_directional_lbm(full_c, 1.0, FAST_CONFIG))
    q_rev = float(run_directional_lbm(full_c, -1.0, FAST_CONFIG))

    if q_fwd <= 1e-5 or q_rev <= 1e-5:
        return 0.0

    open_ratio = q_fwd / FAST_CONFIG.q_open_ref
    penalty = 1.0 if open_ratio >= FAST_CONFIG.min_open_ratio else (open_ratio / FAST_CONFIG.min_open_ratio) ** 2
    di = (q_fwd / q_rev) * penalty
    return -di


def generate_design() -> np.ndarray:
    """
    Optimize continuous parameters of the fluidic diode within the 5-minute budget
    and return the final (250, 100) design mask.
    """
    t_start = time.time()
    nx_d = DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x
    ny = DEFAULT_CONFIG.ny

    # Initial continuous parameters: [ramp_length_frac, throat_gap_frac, ramp_curvature]
    x0 = np.array([0.60, 0.38, 1.00], dtype=np.float64)

    # Quick Nelder-Mead continuous optimization on the fast LBM surrogate (8 evaluations)
    best_params = x0.copy()
    best_val = evaluate_fast_surrogate(best_params)

    res = minimize(
        evaluate_fast_surrogate,
        x0,
        method="Nelder-Mead",
        options={"maxfev": 6, "xatol": 0.05, "fatol": 0.002},
    )
    if res.fun < best_val:
        best_params = res.x
        best_val = res.fun

    elapsed = time.time() - t_start
    print(f"[design.py] Continuous optimization finished in {elapsed:.1f}s | Fast surrogate Di = {-best_val:.4f}")
    print(f"[design.py] Best parameters: {np.round(best_params, 4)}")

    final_mask = render_sawtooth_nozzle(
        jnp.asarray(best_params, dtype=jnp.float32), nx_d, ny, temperature=80.0
    )
    return (np.asarray(final_mask) >= 0.5).astype(np.float32)


if __name__ == "__main__":
    mask = generate_design()
    print(f"Generated mask shape: {mask.shape}, solid fraction: {mask.mean():.3f}")
