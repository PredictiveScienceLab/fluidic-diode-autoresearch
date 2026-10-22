"""
2D Lattice Boltzmann Method (D2Q9 Regularized Smagorinsky LBM) Fluidic Diode Simulator.

Evaluates a 2D channel geometry under a fixed macroscopic pressure drop Delta P
between the left port (x = 0) and right port (x = nx - 1) in both the forward (+x)
and reverse (-x) directions.

Physics formulation:
- Open pressure boundary conditions at x = 0 and x = nx - 1 with fixed density
  difference Delta rho = 2 * d_rho (Delta P = c_s^2 * Delta rho).
  - Forward flow (+1.0): rho(x=0) = 1 + d_rho, rho(x=nx-1) = 1 - d_rho.
  - Reverse flow (-1.0): rho(x=0) = 1 - d_rho, rho(x=nx-1) = 1 + d_rho.
- Top and bottom channel walls (y = 0 and y = ny - 1) have no-slip bounce-back BCs.
- Solid obstacles inside the design domain are modeled via Brinkman velocity
  attenuation + bounce-back reflection, supporting both binary masks in {0, 1}
  for ground-truth evaluation and continuous density fields in [0, 1] for
  differentiable / continuous optimization.
- Collision operator uses Regularized LBM (Latt & Chopard 2006) coupled with a
  Smagorinsky subgrid eddy-viscosity model (C_s = 0.12) for unconditional
  numerical stability around sharp geometric corners at moderate Reynolds numbers.
"""

from __future__ import annotations

import functools
from typing import Dict, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

# D2Q9 Lattice constants
CX_TUPLE = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY_TUPLE = (0, 0, 1, 0, -1, 1, 1, -1, -1)
CX = jnp.array(CX_TUPLE, dtype=jnp.int32)
CY = jnp.array(CY_TUPLE, dtype=jnp.int32)
CX_F = jnp.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0], dtype=jnp.float32)
CY_F = jnp.array([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0], dtype=jnp.float32)
WEIGHTS = jnp.array(
    [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0],
    dtype=jnp.float32,
)
# Opposite directions for bounce-back: [0, 3, 4, 1, 2, 7, 8, 5, 6]
OPP = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=jnp.int32)


class SimConfig(NamedTuple):
    nx: int = 300               # Total channel length
    ny: int = 100               # Total channel height
    pad_x: int = 25             # Fluid buffer at each port (design region is 250 x 100)
    nu: float = 0.002           # Kinematic viscosity in lattice units
    d_rho: float = 0.03         # Half-amplitude of imposed density drop (Delta rho = 0.06)
    n_steps: int = 8000         # Number of LBM timesteps for steady/quasi-steady convergence
    avg_window: int = 2000      # Final steps over which to time-average volumetric flow rate Q
    max_solid_fraction: float = 0.60  # Max allowed solid fraction in design region
    min_open_ratio: float = 0.12      # Min required Q_fwd / Q_open_pipe (prevents choking the pipe)
    q_open_ref: float = 0.1585        # Reference Q_open for DEFAULT_CONFIG


DEFAULT_CONFIG = SimConfig()

# Fast, lower-resolution surrogate configuration for inner-loop continuous optimization in design.py
FAST_CONFIG = SimConfig(
    nx=180,
    ny=60,
    pad_x=15,
    nu=0.0025,
    d_rho=0.03,
    n_steps=3200,
    avg_window=800,
    max_solid_fraction=0.60,
    min_open_ratio=0.12,
    q_open_ref=0.1550,
)


def compute_equilibrium(rho: jnp.ndarray, ux: jnp.ndarray, uy: jnp.ndarray) -> jnp.ndarray:
    """Compute D2Q9 equilibrium distribution f_eq(rho, ux, uy) of shape (..., 9)."""
    u_sq = ux ** 2 + uy ** 2
    cu = (
        ux[..., None] * CX_F
        + uy[..., None] * CY_F
    )
    feq = (
        rho[..., None]
        * WEIGHTS
        * (1.0 + 3.0 * cu + 4.5 * (cu ** 2) - 1.5 * u_sq[..., None])
    )
    return feq


def build_full_domain_mask(design_mask: jnp.ndarray, config: SimConfig = DEFAULT_CONFIG) -> jnp.ndarray:
    """
    Embed the design_mask of shape (nx - 2*pad_x, ny) into the full channel (nx, ny),
    enforcing solid top and bottom walls at y = 0 and y = ny - 1.
    """
    design_nx = config.nx - 2 * config.pad_x
    assert design_mask.shape == (design_nx, config.ny), (
        f"Expected design_mask shape {(design_nx, config.ny)}, got {design_mask.shape}"
    )
    clipped = jnp.clip(design_mask.astype(jnp.float32), 0.0, 1.0)
    pad = jnp.zeros((config.pad_x, config.ny), dtype=jnp.float32)
    full_mask = jnp.concatenate([pad, clipped, pad], axis=0)
    # Enforce top and bottom channel walls as solid (1.0)
    full_mask = full_mask.at[:, 0].set(1.0)
    full_mask = full_mask.at[:, -1].set(1.0)
    return full_mask


@functools.partial(jax.jit, static_argnames=("config", "return_fields"))
def run_directional_lbm(
    full_mask: jnp.ndarray,
    direction: float,
    config: SimConfig = DEFAULT_CONFIG,
    return_fields: bool = False,
):
    """
    Run D2Q9 Regularized Smagorinsky LBM under fixed pressure boundary conditions
    for a given flow direction (+1.0 for forward +x, -1.0 for reverse -x).
    """
    nx, ny = config.nx, config.ny
    solid = jnp.clip(full_mask, 0.0, 1.0)
    # Cubic Brinkman / bounce-back interpolation (exact 0 in fluid, 1 in solid, smooth for gradients)
    alpha = solid ** 3

    # Initialize with linear pressure (density) gradient between ports and zero velocity
    d_rho = config.d_rho * direction
    rho_left = 1.0 + d_rho
    rho_right = 1.0 - d_rho
    rho0 = jnp.linspace(rho_left, rho_right, nx, dtype=jnp.float32)[:, None] * jnp.ones((1, ny), dtype=jnp.float32)
    ux0 = jnp.zeros((nx, ny), dtype=jnp.float32)
    uy0 = jnp.zeros((nx, ny), dtype=jnp.float32)
    f0 = compute_equilibrium(rho0, ux0, uy0)

    start_avg = config.n_steps - config.avg_window
    tau_0 = 3.0 * config.nu + 0.5
    c_smag = 0.12

    q_xx = CX_F * CX_F - 1.0 / 3.0
    q_yy = CY_F * CY_F - 1.0 / 3.0
    q_xy = CX_F * CY_F

    def step_fn(carry, step_idx):
        f, q_accum = carry

        # 1. Macroscopic density and velocity
        rho = jnp.sum(f, axis=-1)
        inv_rho = 1.0 / jnp.maximum(rho, 0.5)
        ux = jnp.sum(f * CX_F[None, None, :], axis=-1) * inv_rho * (1.0 - alpha)
        uy = jnp.sum(f * CY_F[None, None, :], axis=-1) * inv_rho * (1.0 - alpha)

        # Clip velocity near Mach limit (u_max = 0.25) for numerical safety
        u_mag = jnp.sqrt(ux ** 2 + uy ** 2 + 1e-12)
        scale = jnp.minimum(1.0, 0.25 / u_mag)
        ux = ux * scale
        uy = uy * scale

        # 2. Regularized Smagorinsky BGK Collision
        feq = compute_equilibrium(rho, ux, uy)
        f_neq = f - feq
        pi_xx = jnp.sum(f_neq * (CX_F * CX_F)[None, None, :], axis=-1)
        pi_yy = jnp.sum(f_neq * (CY_F * CY_F)[None, None, :], axis=-1)
        pi_xy = jnp.sum(f_neq * (CX_F * CY_F)[None, None, :], axis=-1)
        pi_norm = jnp.sqrt(pi_xx ** 2 + pi_yy ** 2 + 2.0 * (pi_xy ** 2) + 1e-16)

        tau_eff = 0.5 * (
            tau_0 + jnp.sqrt(tau_0 ** 2 + 18.0 * (c_smag ** 2) * pi_norm / jnp.maximum(rho, 0.5))
        )
        omega_eff = (1.0 / tau_eff)[:, :, None]

        f_neq_reg = 4.5 * WEIGHTS[None, None, :] * (
            pi_xx[:, :, None] * q_xx[None, None, :]
            + pi_yy[:, :, None] * q_yy[None, None, :]
            + 2.0 * pi_xy[:, :, None] * q_xy[None, None, :]
        )

        f_col_fluid = feq + (1.0 - omega_eff) * f_neq_reg
        f_opp = f[:, :, OPP]
        f_col = (1.0 - alpha[:, :, None]) * f_col_fluid + alpha[:, :, None] * f_opp

        # 3. Streaming step
        f_stream = jnp.stack(
            [
                jnp.roll(f_col[:, :, i], shift=(CX_TUPLE[i], CY_TUPLE[i]), axis=(0, 1))
                for i in range(9)
            ],
            axis=-1,
        )

        # 4. Open pressure boundary conditions at x = 0 and x = nx - 1
        rho_1 = jnp.sum(f_stream[1, :, :], axis=-1)
        ux_1 = jnp.sum(f_stream[1, :, :] * CX_F[None, :], axis=-1) / jnp.maximum(rho_1, 0.5) * (1.0 - alpha[0, :])
        uy_1 = jnp.sum(f_stream[1, :, :] * CY_F[None, :], axis=-1) / jnp.maximum(rho_1, 0.5) * (1.0 - alpha[0, :])

        rho_m2 = jnp.sum(f_stream[-2, :, :], axis=-1)
        ux_m2 = jnp.sum(f_stream[-2, :, :] * CX_F[None, :], axis=-1) / jnp.maximum(rho_m2, 0.5) * (1.0 - alpha[-1, :])
        uy_m2 = jnp.sum(f_stream[-2, :, :] * CY_F[None, :], axis=-1) / jnp.maximum(rho_m2, 0.5) * (1.0 - alpha[-1, :])

        feq_left = compute_equilibrium(
            jnp.full((ny,), rho_left, dtype=jnp.float32), ux_1, uy_1
        )
        feq_right = compute_equilibrium(
            jnp.full((ny,), rho_right, dtype=jnp.float32), ux_m2, uy_m2
        )

        f_stream = f_stream.at[0, :, :].set(
            (1.0 - alpha[0, :, None]) * feq_left + alpha[0, :, None] * f_stream[0, :, :]
        )
        f_stream = f_stream.at[-1, :, :].set(
            (1.0 - alpha[-1, :, None]) * feq_right + alpha[-1, :, None] * f_stream[-1, :, :]
        )

        # 5. Accumulate directional volumetric flow rate Q(t) over the final avg_window steps
        q_inst = jnp.mean(direction * ux)
        weight = jnp.where(step_idx >= start_avg, 1.0 / config.avg_window, 0.0)
        q_accum = q_accum + weight * q_inst

        return (f_stream, q_accum), None

    (f_final, q_avg), _ = jax.lax.scan(
        step_fn, (f0, jnp.array(0.0, dtype=jnp.float32)), jnp.arange(config.n_steps)
    )

    if not return_fields:
        return q_avg

    rho = jnp.sum(f_final, axis=-1)
    ux = jnp.sum(f_final * CX_F[None, None, :], axis=-1) / jnp.maximum(rho, 0.5) * (1.0 - alpha)
    uy = jnp.sum(f_final * CY_F[None, None, :], axis=-1) / jnp.maximum(rho, 0.5) * (1.0 - alpha)
    return q_avg, rho, ux, uy


def evaluate_binary_design(
    design_mask: np.ndarray | jnp.ndarray,
    config: SimConfig = DEFAULT_CONFIG,
    return_fields: bool = False,
) -> Dict:
    """
    Ground-truth evaluation of a candidate fluidic diode design.
    Thresholds design_mask at 0.5 to enforce a strictly binary solid/fluid layout,
    simulates both forward (+x) and reverse (-x) pressure-driven flows, and computes:
      - q_fwd: Time-averaged forward volumetric flow rate
      - q_rev: Time-averaged reverse volumetric flow rate
      - raw_diodicity: q_fwd / q_rev
      - score: Effective diodicity (penalized if constraints are violated)
    """
    design_nx = config.nx - 2 * config.pad_x
    arr = np.asarray(design_mask, dtype=np.float32)
    if arr.shape != (design_nx, config.ny):
        raise ValueError(f"Expected design_mask shape {(design_nx, config.ny)}, got {arr.shape}")

    binary_design = (arr >= 0.5).astype(np.float32)
    solid_fraction = float(np.mean(binary_design))
    full_mask = build_full_domain_mask(jnp.asarray(binary_design), config)

    if return_fields:
        q_fwd_j, rho_f, ux_f, uy_f = run_directional_lbm(full_mask, 1.0, config, return_fields=True)
        q_rev_j, rho_r, ux_r, uy_r = run_directional_lbm(full_mask, -1.0, config, return_fields=True)
    else:
        q_fwd_j = run_directional_lbm(full_mask, 1.0, config, return_fields=False)
        q_rev_j = run_directional_lbm(full_mask, -1.0, config, return_fields=False)

    q_fwd = float(q_fwd_j)
    q_rev = float(q_rev_j)
    open_ratio = q_fwd / config.q_open_ref

    if q_fwd <= 1e-6 or q_rev <= 1e-6 or not np.isfinite(q_fwd) or not np.isfinite(q_rev):
        raw_diodicity = 0.0
    else:
        raw_diodicity = q_fwd / q_rev

    # Apply smooth engineering feasibility multipliers:
    # 1. Forward flow rate must be at least min_open_ratio (12%) of the open pipe flow rate
    #    so the valve is a true open conduit in the forward direction rather than a plugged wall.
    if open_ratio < config.min_open_ratio:
        flow_penalty = max(0.0, open_ratio / config.min_open_ratio) ** 2
    else:
        flow_penalty = 1.0

    # 2. Solid fraction must not exceed max_solid_fraction (60%)
    if solid_fraction > config.max_solid_fraction:
        solid_penalty = max(0.0, 1.0 - 5.0 * (solid_fraction - config.max_solid_fraction))
    else:
        solid_penalty = 1.0

    is_valid = (open_ratio >= config.min_open_ratio) and (solid_fraction <= config.max_solid_fraction) and (raw_diodicity > 0.0)
    score = raw_diodicity * flow_penalty * solid_penalty

    result = {
        "score": float(score),
        "raw_diodicity": float(raw_diodicity),
        "q_fwd": float(q_fwd),
        "q_rev": float(q_rev),
        "open_ratio": float(open_ratio),
        "solid_fraction": float(solid_fraction),
        "re_fwd": float(q_fwd * config.ny / config.nu),
        "re_rev": float(q_rev * config.ny / config.nu),
        "is_valid": bool(is_valid),
        "binary_design": binary_design,
        "full_mask": np.asarray(full_mask),
    }
    if return_fields:
        result.update({
            "rho_fwd": np.asarray(rho_f),
            "ux_fwd": np.asarray(ux_f),
            "uy_fwd": np.asarray(uy_f),
            "rho_rev": np.asarray(rho_r),
            "ux_rev": np.asarray(ux_r),
            "uy_rev": np.asarray(uy_r),
        })
    return result
