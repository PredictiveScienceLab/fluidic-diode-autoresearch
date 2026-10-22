"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 001 Hypothesis:
- Physical Hypothesis: Replace the simple sawtooth nozzle with a 3-stage
  Nozzle-Plus-Central-Crescent-Cup (Borda / Umbrella) architecture. In +x flow,
  the central crescent cup presents a rounded nose in the wake of the converging
  nozzle; in -x flow, the open concave mouth of the cup faces the incoming reverse
  core jet and reflects it 180 degrees outward into the wall baffles.
- Algorithmic Hypothesis: Replace sequential Nelder-Mead with a GPU-vectorized
  (`jax.vmap`) Latin Hypercube / Random Population Search (N = 64 candidates in
  2 batches of 32) on `FAST_CONFIG`, followed by multi-fidelity verification of
  the top 4 candidates on `DEFAULT_CONFIG`.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast


@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_nozzle_cup_stage(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Render a 3-stage Nozzle + Central Concave Cup fluidic diode.
    params shape: (7,) in [0, 1]
      0: nozzle_len_frac     (length of wall converging ramp)
      1: nozzle_throat_frac  (half-gap of wall nozzle throat)
      2: cup_offset_frac     (axial position of central cup downstream of nozzle)
      3: cup_radius_frac     (outer radius of central concave cup)
      4: cup_thickness_frac  (wall thickness of central concave cup)
      5: cup_open_depth      (how deep the concave mouth opens toward +x)
      6: n_stages_float      (3 or 4 stages)
    """
    nozzle_len = 0.22 + 0.35 * params[0]
    throat_gap = 0.30 + 0.32 * params[1]
    cup_offset = 0.48 + 0.35 * params[2]
    cup_r = 0.14 + 0.18 * params[3]
    cup_thick = 0.05 + 0.07 * params[4]
    cup_mouth = -0.15 + 0.45 * params[5]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)

    n_stages = 3
    pitch = 0.86 / n_stages
    margin_x = 0.07

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch  # normalized stage coordinate in [0, 1]

        # 1. Converging wall ramp for dx in [0, nozzle_len]
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * s
        wall_nozzle = (dx >= 0.0) & (dx <= nozzle_len) & (abs_y >= y_wall)

        # 2. Central concave C-cup centered at (dx = cup_offset, y = 0)
        # Scaled so aspect ratio matches physical lattice coordinates
        aspect = (nx_design * pitch) / (ny * 0.5)
        dist = jnp.sqrt(((dx - cup_offset) * aspect) ** 2 + y ** 2)
        in_ring = (dist <= cup_r) & (dist >= jnp.maximum(0.02, cup_r - cup_thick))
        in_arc = ((dx - cup_offset) * aspect) <= (cup_r * cup_mouth)
        cup = in_ring & in_arc

        stage = (wall_nozzle | cup).astype(jnp.float32)
        mask = jnp.maximum(mask, stage)

    return mask


render_batch_coarse = jax.jit(
    jax.vmap(lambda p: render_nozzle_cup_stage(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny))
)
render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_nozzle_cup_stage(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)


def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(101)

    # Sample 64 parameter vectors in [0, 1]^7 using stratified Latin-like sampling
    n_candidates = 64
    batch_size = 32
    pop = rng.uniform(0.0, 1.0, size=(n_candidates, 7)).astype(np.float32)

    all_scores = []
    for i in range(0, n_candidates, batch_size):
        p_batch = jnp.asarray(pop[i : i + batch_size])
        masks_c = render_batch_coarse(p_batch)
        scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
        all_scores.append(np.asarray(scores_c))

    coarse_scores = np.concatenate(all_scores, axis=0)
    top_idx = np.argsort(coarse_scores)[::-1][:4]
    top_params = jnp.asarray(pop[top_idx])

    print(
        f"[design.py] Screened {n_candidates} candidates on L4 GPU (FAST_CONFIG) in {time.time()-t0:.1f}s | "
        f"Top coarse Di = {coarse_scores[top_idx[0]]:.4f}"
    )

    # Multi-fidelity check: evaluate top 4 candidates on high-resolution DEFAULT_CONFIG
    masks_f = render_batch_fine(top_params)
    fine_scores, qf_f, qr_f = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    fine_scores_np = np.asarray(fine_scores)
    best_k = int(np.argmax(fine_scores_np))

    print(
        f"[design.py] Multi-fidelity fine validation finished in {time.time()-t0:.1f}s | "
        f"Winner fine Di = {fine_scores_np[best_k]:.4f} (Q_fwd={float(qf_f[best_k]):.5f}, Q_rev={float(qr_f[best_k]):.5f})"
    )
    return np.asarray(masks_f[best_k], dtype=np.float32)


if __name__ == "__main__":
    m = generate_design()
    print("Mask shape:", m.shape, "solid:", m.mean())
