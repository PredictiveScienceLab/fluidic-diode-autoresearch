"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 004 Hypothesis:
- Physical Hypothesis: Visual inspection of `run_003_teardrop_cup_multifidelity.png`
  (Di = 1.6938) exposed a fascinating geometry bug! Because `ux_inner` was shifted by
  `+0.03 * pitch` (~2.2 pixels) while the cup prow was elongated by `prow_stretch > 1`,
  `r_inner < 1` broke through the pointed front nose of the cup (`y = 0`), creating a
  2-pixel hole right through the tip of every cup so the cup became a pair of open ribs!
  Furthermore, the split reverse jets at `y ~ +-0.35` were able to enter the next
  stage's converging nozzle throat without hitting an outer reverse scoop.
  Here we:
    1. Enforce a guaranteed solid septum / nose at the front of the cup (`ux_inner`
       offset by a parameterized `nose_solid_thickness` >= 0.14 in scaled coordinates)
       so the concave cup is a true, sealed deep parabolic reflector facing `-x`!
    2. Add an optional flared rim / reverse-scoop lip on the outer edge of the cup
       mouth (`flare_amp`) that angles the reflected `-x` jet directly into the
       recirculation pocket behind the upstream wall tooth.
    3. Explore both 3-stage and 4-stage cascades.
- Algorithmic Hypothesis:
    Run 128-candidate `jax.vmap` screening on `MATCHED_FAST_CONFIG` (64 3-stage +
    64 4-stage), followed by 5 rounds of direct high-resolution (`DEFAULT_CONFIG`)
    `jax.vmap` evolutionary optimization (40 fine-grid LBM evaluations in batches of 8,
    taking ~55s out of our 300s budget on the L4 GPU).
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


@functools.partial(jax.jit, static_argnames=("nx_design", "ny", "n_stages"))
def render_sealed_parabolic_cup_diode(
    params: jnp.ndarray,
    nx_design: int,
    ny: int,
    n_stages: int = 3,
) -> jnp.ndarray:
    """
    Render an n_stages Sealed-Nose Parabolic Cup + Converging Nozzle fluidic diode.
    params shape: (11,) in [0, 1]
      0: nozzle_len_frac     (length of converging nozzle ramp)
      1: nozzle_throat_frac  (half-gap of nozzle throat)
      2: nozzle_power        (curvature of converging ramp)
      3: cup_offset_frac     (axial center of cup nose downstream of stage start)
      4: cup_radius_y        (outer half-height of central cup)
      5: cup_prow_stretch    (streamlined forward prow elongation for +x flow)
      6: cup_cavity_len      (length of rear concave cavity arms extending toward +x)
      7: cup_wall_thick      (radial wall thickness of cup arms)
      8: cup_nose_thick      (axial thickness of solid sealed nose at y=0, preventing leak!)
      9: pitch_scale         (overall axial span of the cascade)
     10: flare_angle         (inward/outward flare of the cup arms toward +x)
    """
    nozzle_len = 0.18 + 0.36 * params[0]
    throat_gap = 0.28 + 0.32 * params[1]
    nozzle_pow = 0.65 + 1.00 * params[2]
    cup_offset = 0.34 + 0.36 * params[3]
    cup_ry = 0.13 + 0.17 * params[4]
    prow_stretch = 0.60 + 1.10 * params[5]
    cavity_len = 0.10 + 0.55 * params[6]
    wall_thick = 0.045 + 0.075 * params[7]
    nose_thick = 0.10 + 0.22 * params[8]
    pitch_scale = 0.80 + 0.14 * params[9]
    flare = -0.15 + 0.30 * params[10]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)

    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch

        # 1. Converging wall ramp
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len) & (abs_y >= y_wall)

        # 2. Sealed-Nose Parabolic / Teardrop Concave Cup
        # ux_rel = 0 is the vertex of the inner cavity; the outer nose extends to ux_rel = -nose_thick * prow_stretch
        ux_rel = (dx - cup_offset) * aspect
        # Flared local half-width as a function of ux_rel
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)

        # Outer boundary: for ux_rel < 0 (the nose), elliptical prow of length (nose_thick * prow_stretch);
        # for ux_rel in [0, cavity_len], outer cylinder of half-width ry_local
        in_outer_nose = (ux_rel <= 0.0) & (
            (ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0
        )
        in_outer_body = (ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local)
        in_outer = in_outer_nose | in_outer_body

        # Inner hollow cavity (open toward +x, strictly sealed for ux_rel < 0!)
        in_cavity = (ux_rel >= 0.0) & (
            ((ux_rel / jnp.maximum(0.05, nose_thick * 0.6)) ** 2 + (abs_y / ry_inner) ** 2 >= 1.0)
            is False
        )
        # More cleanly: cavity is inside abs_y < ry_inner for ux_rel > 0, with rounded inner vertex at ux_rel = 0
        r_vertex = jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (r_vertex <= 1.0)

        in_cup = in_outer & (~hollow)

        stage = (wall_nozzle | in_cup).astype(jnp.float32)
        mask = jnp.maximum(mask, stage)

    return mask


render_batch_coarse_3 = jax.jit(
    jax.vmap(lambda p: render_sealed_parabolic_cup_diode(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny, 3))
)
render_batch_coarse_4 = jax.jit(
    jax.vmap(lambda p: render_sealed_parabolic_cup_diode(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny, 4))
)
render_batch_fine_3 = jax.jit(
    jax.vmap(lambda p: render_sealed_parabolic_cup_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny, 3))
)
render_batch_fine_4 = jax.jit(
    jax.vmap(lambda p: render_sealed_parabolic_cup_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny, 4))
)


def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(404)

    # Phase 1: Screen 64 3-stage and 64 4-stage candidates on Re-matched FAST_CONFIG
    pop3 = rng.uniform(0.0, 1.0, size=(64, 11)).astype(np.float32)
    pop4 = rng.uniform(0.0, 1.0, size=(64, 11)).astype(np.float32)

    # Seed first 16 of pop3 near high-performing nozzle-cup proportions
    anchor = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50], dtype=np.float32)
    pop3[0] = anchor
    pop3[1:16] = np.clip(anchor + rng.normal(0.0, 0.16, size=(15, 11)).astype(np.float32), 0.0, 1.0)
    pop4[0] = anchor
    pop4[1:16] = np.clip(anchor + rng.normal(0.0, 0.16, size=(15, 11)).astype(np.float32), 0.0, 1.0)

    sc3_list, sc4_list = [], []
    for i in range(0, 64, 32):
        m3 = render_batch_coarse_3(jnp.asarray(pop3[i : i + 32]))
        s3, _, _ = evaluate_batch_fast(m3, MATCHED_FAST_CONFIG)
        sc3_list.append(np.asarray(s3))

        m4 = render_batch_coarse_4(jnp.asarray(pop4[i : i + 32]))
        s4, _, _ = evaluate_batch_fast(m4, MATCHED_FAST_CONFIG)
        sc4_list.append(np.asarray(s4))

    sc3 = np.concatenate(sc3_list, axis=0)
    sc4 = np.concatenate(sc4_list, axis=0)

    print(
        f"[design.py] Phase 1 Coarse Screening finished in {time.time()-t0:.1f}s | "
        f"Best 3-stage Coarse Di = {sc3.max():.4f} | Best 4-stage Coarse Di = {sc4.max():.4f}"
    )

    # Pick whichever stage count (3 or 4) dominates, and also test top 4 of each on DEFAULT_CONFIG first!
    top4_p3 = pop3[np.argsort(sc3)[::-1][:4]]
    top4_p4 = pop4[np.argsort(sc4)[::-1][:4]]

    mf3 = render_batch_fine_3(jnp.asarray(top4_p3))
    mf4 = render_batch_fine_4(jnp.asarray(top4_p4))
    init_fine_masks = jnp.concatenate([mf3, mf4], axis=0)
    init_f_scores, _, _ = evaluate_batch_fast(init_fine_masks, DEFAULT_CONFIG)
    init_f_np = np.asarray(init_f_scores)

    best_3_fine = float(init_f_np[:4].max())
    best_4_fine = float(init_f_np[4:].max())
    use_4_stages = best_4_fine > best_3_fine
    render_fine = render_batch_fine_4 if use_4_stages else render_batch_fine_3
    chosen_stages = 4 if use_4_stages else 3

    print(
        f"[design.py] Initial Fine Check: 3-stage max Di = {best_3_fine:.4f}, "
        f"4-stage max Di = {best_4_fine:.4f} -> Selecting {chosen_stages}-stage family!"
    )

    best_fine_score = float(init_f_np.max())
    best_idx0 = int(np.argmax(init_f_np))
    best_fine_mask = np.asarray(init_fine_masks[best_idx0], dtype=np.float32)

    elite_pool = top4_p4 if use_4_stages else top4_p3
    elite_scores = init_f_np[4:].copy() if use_4_stages else init_f_np[:4].copy()

    # Phase 2: 4 rounds of direct fine-grid (1+lambda) ES in batches of 8 on DEFAULT_CONFIG
    for r in range(4):
        order = np.argsort(elite_scores)[::-1]
        p1, p2 = elite_pool[order[0]], elite_pool[order[1]]
        sigma = 0.10 * (0.72 ** r)

        cand_batch = []
        for k in range(8):
            base = p1 if (k < 5) else p2
            step = rng.normal(0.0, sigma, size=(11,)).astype(np.float32)
            cand_batch.append(np.clip(base + step, 0.0, 1.0))
        cand_arr = np.stack(cand_batch, axis=0)

        m_f = render_fine(jnp.asarray(cand_arr))
        f_sc, qf_f, qr_f = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        f_sc_np = np.asarray(f_sc)

        r_best = int(np.argmax(f_sc_np))
        if f_sc_np[r_best] > best_fine_score:
            best_fine_score = float(f_sc_np[r_best])
            best_fine_mask = np.asarray(m_f[r_best], dtype=np.float32)

        # Merge into elite pool
        elite_pool = np.concatenate([elite_pool, cand_arr], axis=0)
        elite_scores = np.concatenate([elite_scores, f_sc_np], axis=0)

        print(
            f"[design.py] Fine Evolution Round {r+1}/4 | Round Best Di = {f_sc_np[r_best]:.4f} "
            f"| Overall Best Fine Di = {best_fine_score:.4f} | Elapsed = {time.time()-t0:.1f}s"
        )

    return best_fine_mask


if __name__ == "__main__":
    m = generate_design()
    print("Mask shape:", m.shape, "solid:", m.mean())
