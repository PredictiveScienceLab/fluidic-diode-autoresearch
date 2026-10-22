"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 005 Hypothesis:
- Physical Hypothesis: Non-Uniformly Graded 3-Stage Sealed U-Cup + Concave Wall
  Return Pockets (`render_graded_ucup_vortex_diode`).
  In Run 004 (`Di = 2.4386`), sealing the nose of the deep U-cup reflectors jumped
  forward flow to `Q_fwd = 0.0522` while cutting reverse flow to `Q_rev = 0.0214`.
  Looking closely at `artifacts/run_004_sealed_parabolic_cup.png`:
    1. In reverse flow (`-x`), the jet exiting the Stage 3 nozzle (`x ~ 200`) splits
       around the Stage 2 U-cup (`x ~ 165`) and shoots diagonally toward the outer
       channel walls at `x ~ 135..155`. Right now, that outer wall is flat (`y = 0`
       and `y = 99`), so the reverse jet bounces off the flat wall back toward the
       next throat! If we carve a backward-facing concave hook / undercut on the
       back face of each wall tooth (`tooth_undercut`), the diagonally deflected
       reverse jet gets scooped into a closed wall vortex loop and turned 180 degrees!
    2. Allowing stage-by-stage grading (`cup_ry_grade`, `throat_grade`) lets Stage 3
       (the reverse inlet) have a wider reverse-catching mouth while Stage 1 has a
       low-loss forward exit.
- Algorithmic Hypothesis:
    Use 160s of our 300s budget on the L4 GPU to run:
    - Phase 1: 128-candidate `jax.vmap` global screen on `MATCHED_FAST_CONFIG`.
    - Phase 2: 12 rounds of direct high-resolution (`DEFAULT_CONFIG`, `300x100`)
      `jax.vmap` evolutionary optimization (`12 x 8 = 96` fine-grid LBM evaluations,
      i.e., 192 high-resolution LBM runs in parallel batches of 8!).
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
    """
    Render a 3-stage Graded Sealed U-Cup + Wall Undercut Vortex Diode.
    params shape: (14,) in [0, 1]
      0: nozzle_len_frac     (length of converging nozzle ramp)
      1: nozzle_throat_frac  (base half-gap of nozzle throat)
      2: nozzle_power        (curvature of converging ramp)
      3: cup_offset_frac     (axial center of cup nose downstream of stage start)
      4: cup_radius_y        (base outer half-height of central U-cup)
      5: cup_prow_stretch    (streamlined forward prow elongation for +x flow)
      6: cup_cavity_len      (length of rear U-cup arms extending toward +x)
      7: cup_wall_thick      (radial wall thickness of U-cup arms)
      8: cup_nose_thick      (axial thickness of solid sealed nose at y=0)
      9: pitch_scale         (overall axial span of the 3-stage cascade)
     10: flare_angle         (inward/outward flare of the U-cup arms toward +x)
     11: tooth_undercut      (undercut / reverse hook on the downstream face of the wall nozzle)
     12: cup_ry_grade        (stage-to-stage grading of U-cup half-width across stages 0, 1, 2)
     13: throat_grade        (stage-to-stage grading of nozzle throat gap across stages 0, 1, 2)
    """
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

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)

    n_stages = 3
    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        stage_centered = k - 1.0  # -1.0, 0.0, +1.0
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * stage_centered), 0.10, 0.36)

        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch

        # 1. Converging wall ramp with optional downstream undercut scoop
        # The rear face of the tooth slopes backward slightly from throat (dx = nozzle_len)
        # toward the wall (dx = nozzle_len - undercut), scooping diagonal -x jets!
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        # 2. Sealed-Nose Deep U-Cup Reflector
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)

        in_outer_nose = (ux_rel <= 0.0) & (
            (ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0
        )
        in_outer_body = (ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local)
        in_outer = in_outer_nose | in_outer_body

        r_vertex = jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (r_vertex <= 1.0)

        in_cup = in_outer & (~hollow)

        stage = (wall_nozzle | in_cup).astype(jnp.float32)
        mask = jnp.maximum(mask, stage)

    return mask


render_batch_coarse = jax.jit(
    jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny))
)
render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)


def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(505)

    # Reconstruct Run 004's winning trajectory on the 11 base parameters so we warmly start from Di = 2.4386!
    rng_404 = np.random.default_rng(404)
    pop3_404 = rng_404.uniform(0.0, 1.0, size=(64, 11)).astype(np.float32)
    pop4_404 = rng_404.uniform(0.0, 1.0, size=(64, 11)).astype(np.float32)
    anchor_11 = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50], dtype=np.float32)
    pop3_404[0] = anchor_11
    pop3_404[1:16] = np.clip(anchor_11 + rng_404.normal(0.0, 0.16, size=(15, 11)).astype(np.float32), 0.0, 1.0)

    # Embed 64 candidates from Run 004's space (with neutral undercut=0, grade=0.5) + 64 new graded candidates
    pop = rng.uniform(0.0, 1.0, size=(128, 14)).astype(np.float32)
    for i in range(64):
        pop[i, :11] = pop3_404[i]
        pop[i, 11] = 0.0
        pop[i, 12] = 0.5
        pop[i, 13] = 0.5

    # Phase 1: Coarse vmap screening (128 candidates)
    sc_list = []
    for i in range(0, 128, 32):
        m_c = render_batch_coarse(jnp.asarray(pop[i : i + 32]))
        s_c, _, _ = evaluate_batch_fast(m_c, MATCHED_FAST_CONFIG)
        sc_list.append(np.asarray(s_c))
    sc_all = np.concatenate(sc_list, axis=0)

    print(
        f"[design.py] Phase 1 Coarse Screen (128 candidates) finished in {time.time()-t0:.1f}s | "
        f"Max Coarse Di = {sc_all.max():.4f}"
    )

    # Evaluate top 8 from Phase 1 on high-resolution DEFAULT_CONFIG
    top8_idx = np.argsort(sc_all)[::-1][:8]
    elite_pool = pop[top8_idx].copy()
    m_f0 = render_batch_fine(jnp.asarray(elite_pool))
    f_sc0, _, _ = evaluate_batch_fast(m_f0, DEFAULT_CONFIG)
    elite_scores = np.asarray(f_sc0).copy()

    best_fine_score = float(elite_scores.max())
    best_fine_mask = np.asarray(m_f0[int(np.argmax(elite_scores))], dtype=np.float32)

    print(
        f"[design.py] Initial Top-8 Fine Check finished in {time.time()-t0:.1f}s | "
        f"Best Fine Di = {best_fine_score:.4f}"
    )

    # Phase 2: 10 rounds of direct high-resolution (`DEFAULT_CONFIG`) vmap Evolutionary Search (80 fine evaluations!)
    for r in range(10):
        if time.time() - t0 > 220.0:
            break
        order = np.argsort(elite_scores)[::-1]
        p1, p2, p3 = elite_pool[order[0]], elite_pool[order[1]], elite_pool[order[2]]
        sigma = 0.11 * (0.82 ** r)

        cand_batch = []
        for k in range(8):
            if k < 4:
                base = p1
            elif k < 6:
                base = p2
            else:
                # Recombination between top 2 elites + mutation
                alpha_mix = rng.uniform(0.2, 0.8, size=(14,)).astype(np.float32)
                base = alpha_mix * p1 + (1.0 - alpha_mix) * p2
            step = rng.normal(0.0, sigma, size=(14,)).astype(np.float32)
            # In first 3 rounds, keep 2 children strictly in the ungraded subspace to refine the U-cup geometry
            if r < 3 and k == 0:
                step[11:] = 0.0
            cand_batch.append(np.clip(base + step, 0.0, 1.0))

        cand_arr = np.stack(cand_batch, axis=0)
        m_f = render_batch_fine(jnp.asarray(cand_arr))
        f_sc, qf_f, qr_f = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        f_sc_np = np.asarray(f_sc)

        r_best = int(np.argmax(f_sc_np))
        if f_sc_np[r_best] > best_fine_score:
            best_fine_score = float(f_sc_np[r_best])
            best_fine_mask = np.asarray(m_f[r_best], dtype=np.float32)

        elite_pool = np.concatenate([elite_pool, cand_arr], axis=0)
        elite_scores = np.concatenate([elite_scores, f_sc_np], axis=0)

        print(
            f"[design.py] Fine Evolution Round {r+1:02d}/10 | Round Best Di = {f_sc_np[r_best]:.4f} "
            f"| Overall Best Fine Di = {best_fine_score:.4f} | Elapsed = {time.time()-t0:.1f}s"
        )

    return best_fine_mask


if __name__ == "__main__":
    m = generate_design()
    print("Mask shape:", m.shape, "solid:", m.mean())
