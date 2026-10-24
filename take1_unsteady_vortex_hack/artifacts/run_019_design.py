"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 019 Hypothesis:
- Physical Hypothesis: The existing parameterization of graded U-cups with a central splitter fin is effective.
  The key challenge is optimizing these parameters efficiently to maximize reverse flow impedance while
  maintaining forward flow conductance. No new geometric features are introduced in this iteration.
- Algorithmic Hypothesis: Optimize the multi-fidelity search strategy to strictly adhere to the 25-60 second
  time budget while maximizing the chance of finding high-diodicity designs.
    1.  **Corrected Coarse-Grid Reference Flow:** The `MATCHED_FAST_CONFIG`'s `q_open_ref` is explicitly
        set to a value scaled proportionally to `DEFAULT_CONFIG`'s `q_open_ref` based on differences in
        viscosity, pressure drop, and domain dimensions. This ensures that the `min_open_ratio` constraint
        is evaluated consistently across fidelity levels during optimization.
    2.  **Extensive Coarse-Grid Evolution:** Perform 8 rounds of evolutionary search on `MATCHED_FAST_CONFIG`
        with a population of 32 candidates per round. This allows for broad and rapid exploration of the
        parameter space, leveraging JAX's `vmap` for parallel simulation. Candidate generation is refined
        to focus mutations and recombinations around the top elites.
    3.  **Anchored Warm-Start:** The best parameter set from `run_007` (which effectively explored the central
        splitter fin) will be explicitly included and mutated in the initial coarse-grid population to
        ensure a strong starting point and prevent regression.
    4.  **Elite Preservation:** Elites (best-performing parameter sets) are maintained and used as mutation
        bases across coarse-grid rounds to guide the search towards promising regions.
    5.  **Final Fine-Grid Validation:** Only the top 8 candidates from the coarse-grid evolution will be
        evaluated *once* on the high-resolution `DEFAULT_CONFIG` at the very end. This significantly reduces
        expensive fine-grid simulations, keeping total design time within budget.
    6.  **Parameter Tracking:** The best parameter vector (`best_fine_params`) will be stored and printed
        along with the `best_fine_mask` to enable precise warm-starting for future iterations.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Calculate q_open_ref for MATCHED_FAST_CONFIG
# DEFAULT_CONFIG: d_rho=0.06, nu=0.002, ny=100, nx=300, q_open_ref=0.1585
# MATCHED_FAST_CONFIG: d_rho=0.03, nu=0.0012, ny=60, nx=180
# Q_open is proportional to (d_rho / nu) * H^3 / L
# Ratio of (d_rho / nu): (0.03 / 0.0012) / (0.06 / 0.002) = 25 / 30 = 0.8333
# Ratio of H^3: (60^3) / (100^3) = 0.6^3 = 0.216
# Ratio of L (inverse): 300 / 180 = 1.6667
# Combined ratio = 0.8333 * 0.216 * 1.6667 = 0.30
# q_open_ref_fast = 0.30 * DEFAULT_CONFIG.q_open_ref = 0.30 * 0.1585 = 0.04755
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
    q_open_ref=0.04755, # Corrected based on proportional scaling from DEFAULT_CONFIG
)


@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(
    params: jnp.ndarray,
    nx_design: int,
    ny: int,
) -> jnp.ndarray:
    """
    Render a 3-stage Graded Sealed U-Cup + Wall Undercut Vortex Diode with Central Splitter Fin.
    params shape: (16,) in [0, 1]
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
     14: fin_len_frac        (axial length of the central splitter fin, relative to cup cavity)
     15: fin_width_frac      (half-width of the central splitter fin, relative to inner cup radius)
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
    fin_len_scale = 0.0 + 0.8 * params[14] # Up to 80% of cup_cavity_len
    fin_width_scale = 0.0 + 0.8 * params[15] # Up to 80% of ry_inner

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
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        # 2. Sealed-Nose Deep U-Cup Reflector
        ux_rel = (dx - cup_offset) * aspect # Axial coordinate relative to cup nose (scaled by aspect ratio)
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

        # 3. Central Splitter Fin (extends upstream from cup nose)
        fin_len_actual = fin_len_scale * cavity_len
        fin_width_actual = fin_width_scale * ry_inner

        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_actual) & (abs_y <= fin_width_actual)

        stage = (wall_nozzle | in_cup | is_fin).astype(jnp.float32)
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
    rng = np.random.default_rng(505) # Keep fixed seed for reproducibility

    # Best parameters from run_007 (Di = 3.223864). This is the 'anchor_explore_fin_params'
    # from run_007's design.py, which was specifically initialized to explore fin parameters.
    # Since run_007's best_fine_params were not saved, we use this as our best-known starting point.
    best_known_params_run007 = np.array([
        0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, # P0-P10
        0.0,  0.5,  0.5,  # P11-P13 (undercut, cup_ry_grade, throat_grade)
        0.5,  0.5   # P14-P15 (fin_len_frac, fin_width_frac) - these were the exploration initial values
    ], dtype=np.float32)

    # Initialize elite pool with the best known design and some mutated versions
    elite_pool = np.zeros((8, 16), dtype=np.float32)
    elite_pool[0] = best_known_params_run007
    # Mutate the best known design to generate initial diversity around the optimum
    for i in range(1, 8):
        elite_pool[i] = np.clip(best_known_params_run007 + rng.normal(0.0, 0.1, size=(16,)).astype(np.float32), 0.0, 1.0)
    
    elite_scores = np.zeros(8, dtype=np.float32) # Will be populated after first evaluation

    # Phase 1: Extensive Coarse-Grid Evolutionary Search (8 rounds, 32 candidates per round)
    print(f"[design.py] Starting Phase 1: Coarse-Grid Evolution ({8} rounds of 32 candidates)")
    for r in range(8): # 8 rounds of evolution
        if time.time() - t0 > 35.0: # Roughly ~4.5s per round for 32 candidates. 8 rounds = 36s.
            print(f"[design.py] Early exit from coarse-grid evolution due to time budget at round {r+1}.")
            break

        cand_batch = []
        for k in range(32): # Generate 32 new candidates per round
            if k < 10: # Mutate top elite (10 candidates)
                base = elite_pool[0]
            elif k < 20: # Mutate second elite (10 candidates)
                base = elite_pool[1]
            elif k < 28: # Mutate third elite (8 candidates)
                base = elite_pool[2]
            else: # Recombination between top 2 elites + mutation (4 candidates)
                alpha_mix = rng.uniform(0.2, 0.8, size=(16,)).astype(np.float32)
                base = alpha_mix * elite_pool[0] + (1.0 - alpha_mix) * elite_pool[1]
            
            sigma = 0.1 * (0.85 ** r) # Decreasing mutation strength
            step = rng.normal(0.0, sigma, size=(16,)).astype(np.float32)
            cand_batch.append(np.clip(base + step, 0.0, 1.0))

        cand_arr = np.stack(cand_batch, axis=0)
        
        # Evaluate candidates on coarse grid
        m_c = render_batch_coarse(jnp.asarray(cand_arr))
        s_c, _, _ = evaluate_batch_fast(m_c, MATCHED_FAST_CONFIG)
        s_c_np = np.asarray(s_c)

        # Update elite pool: combine current elites with new candidates, keep top 8
        combined_pool = np.concatenate([elite_pool, cand_arr], axis=0)
        combined_scores = np.concatenate([elite_scores, s_c_np], axis=0)
        
        order_combined = np.argsort(combined_scores)[::-1]
        elite_pool = combined_pool[order_combined[:8]].copy()
        elite_scores = combined_scores[order_combined[:8]].copy()

        print(
            f"[design.py] Coarse Round {r+1:02d}/8 | Round Max Di = {s_c_np.max():.4f} "
            f"| Elite Max Di = {elite_scores[0]:.4f} | Elapsed = {time.time()-t0:.1f}s"
        )
    
    print(
        f"[design.py] Phase 1 Coarse-Grid Evolution finished in {time.time()-t0:.1f}s | "
        f"Top Coarse Di = {elite_scores[0]:.4f}"
    )

    # Phase 2: Final Fine-Grid Validation (Top 8 from coarse evolution)
    print(f"[design.py] Starting Phase 2: Final Fine-Grid Validation (8 candidates)")
    m_f = render_batch_fine(jnp.asarray(elite_pool))
    f_sc, qf_f, qr_f = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    f_sc_np = np.asarray(f_sc)

    best_idx = int(np.argmax(f_sc_np))
    final_best_score = float(f_sc_np[best_idx])
    final_best_mask = np.asarray(m_f[best_idx], dtype=np.float32)
    final_best_params = elite_pool[best_idx].copy() # Store the parameters

    print(
        f"[design.py] Final Fine Validation finished in {time.time()-t0:.1f}s | "
        f"Best Fine Di = {final_best_score:.4f}"
    )
    print(f"[design.py] Best Parameters: {final_best_params}")

    return final_best_mask


if __name__ == "__main__":
    m = generate_design()
    print("Mask shape:", m.shape, "solid:", m.mean())
