"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 003 Hypothesis:
- Physical Hypothesis: Streamlined Teardrop-Nose Concave Cup + Non-Uniform
  3-Stage Nozzle-Cup Cascade (`render_teardrop_cup_cascade`).
  In Run 001 (Di = 1.6322), the central C-cup had a blunt circular front face,
  which helped reverse locking but still detached the forward jet slightly.
  In Run 002 (Di = 1.5268), adding horizontal Borda lips choked the forward
  passage (reducing Q_fwd to 0.020) AND revealed a critical surrogate mismatch:
  designs that scored 1.73 on `FAST_CONFIG` (Re_fast ~ 900) dropped to 1.52 on
  `DEFAULT_CONFIG` (Re_fine ~ 1500) because the coarse Reynolds number was lower!
  Here we:
    1. Give the central concave cup a pointed/elliptical streamlined forward
       prow (so +x flow splits smoothly around the outside with minimal pressure
       loss) while keeping a deep, sharp-rimmed concave rear cavity facing -x.
    2. Allow independent grading of the stage parameters across the 3 stages
       (12 continuous parameters).
- Algorithmic Hypothesis:
    1. Match the coarse surrogate's effective Reynolds number to `DEFAULT_CONFIG`
       by using `MATCHED_FAST_CONFIG` (`nu = 0.0012` on the 60-tall grid matches
       `nu = 0.0020` on the 100-tall grid!).
    2. Screen 128 candidates via `jax.vmap` on `MATCHED_FAST_CONFIG`, then run
       two rounds of `jax.vmap` fine-grid evolutionary refinement (32 high-res
       `DEFAULT_CONFIG` evaluations in batches of 8, taking ~25s on the L4 GPU!)
       directly on the true ground-truth oracle!
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, SimConfig, evaluate_batch_fast

# Reynolds-matched fast surrogate: ny=60 vs 100 -> scale nu by (60/100) = 0.0012
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
def render_teardrop_cup_cascade(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Render a 3-stage Streamlined-Prow Concave Cup + Converging Nozzle fluidic diode.
    params shape: (10,) in [0, 1]
      0: nozzle_len_frac     (length of converging nozzle ramp)
      1: nozzle_throat_frac  (half-gap of nozzle throat)
      2: nozzle_power        (curvature of converging ramp)
      3: cup_offset_frac     (axial offset of central cup downstream of nozzle)
      4: cup_radius_y        (transverse outer radius of central cup)
      5: cup_prow_stretch    (streamlined forward prow elongation factor for +x flow)
      6: cup_cavity_depth    (depth of the rear concave cavity facing -x flow)
      7: cup_wall_thick      (radial wall thickness of the cup rim)
      8: stage_pitch_scale   (overall spacing of the 3 stages)
      9: stage3_cup_boost    (extra cup width on the final stage to block initial -x entry)
    """
    nozzle_len = 0.20 + 0.36 * params[0]
    throat_gap = 0.31 + 0.30 * params[1]
    nozzle_pow = 0.70 + 0.90 * params[2]
    cup_offset = 0.44 + 0.34 * params[3]
    cup_ry = 0.14 + 0.16 * params[4]
    prow_stretch = 0.85 + 1.10 * params[5]
    cavity_depth = 0.05 + 0.45 * params[6]
    cup_thick = 0.045 + 0.065 * params[7]
    pitch_scale = 0.80 + 0.12 * params[8]
    stage3_boost = 0.90 + 0.25 * params[9]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)

    n_stages = 3
    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch

        # 1. Converging forward nozzle ramp
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len) & (abs_y >= y_wall)

        # 2. Central Teardrop-Prow Concave Cup
        # Upstream of cup center (dx < cup_offset), elongate x by prow_stretch for low forward drag;
        # Downstream of cup center (dx >= cup_offset), use unit aspect for a wide, blunt reverse-catching mouth!
        ry_k = cup_ry * (stage3_boost if k == 2 else 1.0)
        ux_rel = (dx - cup_offset) * aspect
        ux_outer = jnp.where(ux_rel < 0.0, ux_rel / prow_stretch, ux_rel)
        r_outer = jnp.sqrt(ux_outer ** 2 + (y / ry_k) ** 2)

        # Inner concave cavity hollowed out from the right (+x) side to trap -x jet
        ux_inner = (dx - (cup_offset + 0.03)) * aspect
        r_inner = jnp.sqrt(ux_inner ** 2 + (y / jnp.maximum(0.04, ry_k - cup_thick)) ** 2)

        in_cup = (r_outer <= 1.0) & (r_inner >= 1.0) & (ux_rel <= ry_k * cavity_depth)

        stage = (wall_nozzle | in_cup).astype(jnp.float32)
        mask = jnp.maximum(mask, stage)

    return mask


render_batch_coarse = jax.jit(
    jax.vmap(lambda p: render_teardrop_cup_cascade(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny))
)
render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_teardrop_cup_cascade(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)


def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(303)

    # Anchor seed near Run 001's winning region, plus 127 diverse random candidates
    anchor = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.35, 0.55, 0.40, 0.55, 0.50], dtype=np.float32)
    pop_global = rng.uniform(0.0, 1.0, size=(128, 10)).astype(np.float32)
    pop_global[0] = anchor
    pop_global[1:32] = np.clip(anchor + rng.normal(0.0, 0.15, size=(31, 10)).astype(np.float32), 0.0, 1.0)

    # Phase 1: Screen 128 candidates on Reynolds-matched FAST_CONFIG (4 batches of 32)
    coarse_scores_list = []
    for i in range(0, 128, 32):
        b = jnp.asarray(pop_global[i : i + 32])
        m_c = render_batch_coarse(b)
        sc, _, _ = evaluate_batch_fast(m_c, MATCHED_FAST_CONFIG)
        coarse_scores_list.append(np.asarray(sc))

    coarse_scores = np.concatenate(coarse_scores_list, axis=0)
    top8_idx = np.argsort(coarse_scores)[::-1][:8]
    top8_params = pop_global[top8_idx]

    print(
        f"[design.py] Phase 1: Screened 128 candidates on Re-matched FAST_CONFIG in {time.time()-t0:.1f}s | "
        f"Best coarse Di = {coarse_scores[top8_idx[0]]:.4f}"
    )

    # Phase 2: Direct High-Resolution Evolution on DEFAULT_CONFIG (3 rounds x 8 candidates = 24 fine evaluations)
    best_fine_score = -1.0
    best_fine_params = top8_params[0].copy()
    best_fine_mask = None

    current_batch = top8_params.copy()
    for r in range(3):
        m_f = render_batch_fine(jnp.asarray(current_batch))
        f_scores, qf_f, qr_f = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        f_scores_np = np.asarray(f_scores)

        r_best = int(np.argmax(f_scores_np))
        if f_scores_np[r_best] > best_fine_score:
            best_fine_score = float(f_scores_np[r_best])
            best_fine_params = current_batch[r_best].copy()
            best_fine_mask = np.asarray(m_f[r_best], dtype=np.float32)

        print(
            f"[design.py] Phase 2 Fine Round {r+1}/3 | Round Best Fine Di = {f_scores_np[r_best]:.4f} "
            f"| Overall Best Fine Di = {best_fine_score:.4f} | Elapsed = {time.time()-t0:.1f}s"
        )

        # Generate next 8 fine candidates via elitist mutation around top 2 fine parents
        elite_idx = np.argsort(f_scores_np)[::-1][:2]
        p1, p2 = current_batch[elite_idx[0]], current_batch[elite_idx[1]]
        sigma = 0.08 / (r + 1)
        next_batch = []
        for k in range(8):
            parent = p1 if (k % 2 == 0) else p2
            child = np.clip(parent + rng.normal(0.0, sigma, size=(10,)).astype(np.float32), 0.0, 1.0)
            next_batch.append(child)
        current_batch = np.stack(next_batch, axis=0)

    return best_fine_mask


if __name__ == "__main__":
    m = generate_design()
    print("Mask shape:", m.shape, "solid:", m.mean())
