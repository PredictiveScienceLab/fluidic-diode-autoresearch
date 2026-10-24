"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 024 Hypothesis:
- Physical Hypothesis: Introduce a `fin_prow_taper_factor` (P16) to the central splitter fin.
  This parameter will control the sharpness of the fin's upstream (forward-facing) tip,
  allowing it to vary from a sharp triangular profile (taper_factor=0) to a blunt/rectangular
  profile (taper_factor=1). The goal is to reduce forward-flow drag by streamlining the fin
  while maintaining or enhancing reverse-flow impedance by forcing the reverse jet to interact
  more violently with the now-shaped fin.
- Algorithmic Hypothesis: The base parameters for `run_024` will be derived from `run_007`,
  but with `wall_undercut` (P11), `fin_len_frac` (P14), and `fin_width_frac` (P15) initialized
  to higher values (e.g., 0.9) to reflect the successful "double-blockage" mechanism discovered
  in `run_020`. The new `fin_prow_taper_factor` (P16) will be initialized to a neutral 0.5.
  The number of direct high-resolution (`DEFAULT_CONFIG`) `jax.vmap` evolutionary
  optimization rounds is reduced from 10 to 4 (`4 x 8 = 32` fine-grid LBM evaluations)
  to bring the total design time within the target 25-45 seconds, while still allowing
  sufficient exploration of the new parameter.
  The `best_fine_params` (raw [0,1] values) and `best_fine_raw_diodicity` will be
  explicitly stored and printed to ensure robust warm-starting for future iterations.
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
    Render a 3-stage Graded Sealed U-Cup + Wall Undercut Vortex Diode with Central Splitter Fin.
    params shape: (17,) in [0, 1]
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
     16: fin_prow_taper_factor (0=sharp triangular tip, 1=blunt/rectangular tip for fin's +x-facing prow)
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
    fin_len_scale = 0.0 + 0.8 * params[14]
    fin_width_scale = 0.0 + 0.8 * params[15]
    fin_prow_taper_factor = 0.0 + 1.0 * params[16] # 0 = sharp tip, 1 = blunt/rectangular tip

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
        
        rear_edge_base = nozzle_len
        rear_edge_undercut_profile = undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        rear_edge = rear_edge_base - rear_edge_undercut_profile
        
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

        # 3. Central Splitter Fin with Prow Taper
        fin_len_actual = fin_len_scale * cavity_len
        fin_width_base = fin_width_scale * ry_inner # Base half-width at the junction with the cup

        # x_fin_local goes from 0 (upstream tip, ux_rel = -fin_len_actual) to 1 (downstream end, ux_rel = 0.0)
        x_fin_local = jnp.clip((ux_rel + fin_len_actual) / jnp.maximum(1e-6, fin_len_actual), 0.0, 1.0)

        # Apply prow taper:
        # If fin_prow_taper_factor = 0.0, width tapers linearly from 0 at tip to fin_width_base at junction (sharp triangle).
        # If fin_prow_taper_factor = 1.0, width is fin_width_base along its entire length (rectangle).
        fin_width_at_x = fin_width_base * (fin_prow_taper_factor + (1.0 - fin_prow_taper_factor) * x_fin_local)
        
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_actual) & (abs_y <= fin_width_at_x)

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

    # Base parameters from run_007, now extended to 17 parameters.
    # We will adjust P11, P14, P15 to higher values to reflect run_020's success,
    # and initialize P16 (new) to 0.5.
    base_params_for_run024 = np.array([
        0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, # P0-P10 (from run_007)
        0.9,  0.5,  0.5,  # P11 (undercut, high), P12, P13 (grades, neutral from run_007)
        0.9,  0.9,        # P14, P15 (fin_len, fin_width, high to reflect run_020)
        0.5               # P16 (fin_prow_taper_factor, neutral)
    ], dtype=np.float32)

    # New anchor: Activate undercut with no fin (P14, P15 = 0.0)
    anchor_activate_undercut_nofin = base_params_for_run024.copy()
    anchor_activate_undercut_nofin[11] = 0.9 # High undercut
    anchor_activate_undercut_nofin[14] = 0.0 # No fin
    anchor_activate_undercut_nofin[15] = 0.0 # No fin
    anchor_activate_undercut_nofin[16] = 0.5 # Neutral taper for non-existent fin

    # New anchor: Ensure fin is active and undercut is active, with variable taper
    anchor_combined_undercut_fin_taper = base_params_for_run024.copy()
    anchor_combined_undercut_fin_taper[11] = 0.9 # High undercut
    anchor_combined_undercut_fin_taper[14] = 0.9 # High fin length
    anchor_combined_undercut_fin_taper[15] = 0.9 # High fin width
    anchor_combined_undercut_fin_taper[16] = 0.0 # Start with sharp taper

    # Initialize population (64 candidates for coarse screening)
    pop = rng.uniform(0.0, 1.0, size=(64, 17)).astype(np.float32) # P_count is 17
    
    # Embed the warm-start parameters and some mutations around them
    pop[0] = base_params_for_run024 # Primary warm-start
    pop[1] = anchor_activate_undercut_nofin
    pop[2] = anchor_combined_undercut_fin_taper
    
    # Mutate around the primary warm-start for further exploration
    pop[3:16] = np.clip(base_params_for_run024 + rng.normal(0.0, 0.16, size=(13, 17)).astype(np.float32), 0.0, 1.0)
    # Remaining pop[16:] are random uniform for broader exploration


    # Phase 1: Coarse vmap screening (64 candidates)
    sc_list = []
    qf_list = []
    qr_list = []
    for i in range(0, 64, 32): # Process in batches of 32 for efficiency
        m_c = render_batch_coarse(jnp.asarray(pop[i : i + 32]))
        s_c, qf_c, qr_c = evaluate_batch_fast(m_c, MATCHED_FAST_CONFIG)
        sc_list.append(np.asarray(s_c))
        qf_list.append(np.asarray(qf_c))
        qr_list.append(np.asarray(qr_c))
    sc_all = np.concatenate(sc_list, axis=0)
    qf_all = np.concatenate(qf_list, axis=0)
    qr_all = np.concatenate(qr_list, axis=0)

    print(
        f"[design.py] Phase 1 Coarse Screen (64 candidates) finished in {time.time()-t0:.1f}s | "
        f"Max Coarse Di = {sc_all.max():.4f}"
    )

    # Combine top coarse candidates with anchor points for initial fine evaluation
    top_coarse_params = pop[np.argsort(sc_all)[::-1][:6]] # Top 6 from coarse

    # Initial fine evaluation pool
    initial_fine_eval_list = []
    initial_fine_eval_list.append(base_params_for_run024)
    initial_fine_eval_list.append(anchor_activate_undercut_nofin)
    initial_fine_eval_list.append(anchor_combined_undercut_fin_taper)
    
    # Add top coarse params, avoiding duplicates
    for p_top in top_coarse_params:
        is_duplicate = False
        for p_exist in initial_fine_eval_list:
            if np.allclose(p_exist, p_top):
                is_duplicate = True
                break
        if not is_duplicate:
            initial_fine_eval_list.append(p_top)

    initial_fine_eval_params = np.stack(initial_fine_eval_list, axis=0)
    
    # Evaluate this expanded set on high-resolution DEFAULT_CONFIG
    m_f0_expanded = render_batch_fine(jnp.asarray(initial_fine_eval_params))
    f_sc0_expanded, f_qf0_expanded, f_qr0_expanded = evaluate_batch_fast(m_f0_expanded, DEFAULT_CONFIG)
    
    # Select the top 8 (or desired elite pool size) from this expanded set
    order_f_expanded = np.argsort(f_sc0_expanded)[::-1]
    
    elite_pool = initial_fine_eval_params[order_f_expanded[:8]].copy()
    elite_scores = np.asarray(f_sc0_expanded[order_f_expanded[:8]]).copy()
    elite_raw_dis = np.asarray(f_qf0_expanded[order_f_expanded[:8]] / f_qr0_expanded[order_f_expanded[:8]]).copy()


    best_fine_score = float(elite_scores.max())
    best_fine_raw_diodicity = float(elite_raw_dis.max())
    best_fine_params = elite_pool[np.argmax(elite_scores)].copy()
    best_fine_mask = np.asarray(m_f0_expanded[order_f_expanded[0]], dtype=np.float32)

    print(
        f"[design.py] Initial Fine Check ({len(initial_fine_eval_params)} candidates) finished in {time.time()-t0:.1f}s | "
        f"Best Fine Score = {best_fine_score:.4f} (Raw Di = {best_fine_raw_diodicity:.4f}) | Best Params = {best_fine_params}"
    )

    # Phase 2: 4 rounds of direct high-resolution (`DEFAULT_CONFIG`) vmap Evolutionary Search (targeting ~25-45s total)
    n_fine_rounds = 4 # Reduced rounds for faster execution
    for r in range(n_fine_rounds):
        if time.time() - t0 > 40.0: # Early exit if approaching budget limit (aim for 25-45s)
            print(f"[design.py] Early exit due to time budget at round {r+1}.")
            break
        
        # Select top 3 elites for mutation
        order = np.argsort(elite_scores)[::-1]
        p1, p2, p3 = elite_pool[order[0]], elite_pool[order[1]], elite_pool[order[2]]
        sigma = 0.11 * (0.85 ** r) # Slightly slower decay for mutation strength

        cand_batch = []
        for k in range(8): # Generate 8 new candidates per round
            if k < 4:
                base = p1
            elif k < 6:
                base = p2
            else:
                # Recombination between top 2 elites + mutation for some candidates
                alpha_mix = rng.uniform(0.2, 0.8, size=(17,)).astype(np.float32) # P_count is 17
                base = alpha_mix * p1 + (1.0 - alpha_mix) * p2
            step = rng.normal(0.0, sigma, size=(17,)).astype(np.float32) # P_count is 17
            
            cand_batch.append(np.clip(base + step, 0.0, 1.0))

        cand_arr = np.stack(cand_batch, axis=0)
        m_f = render_batch_fine(jnp.asarray(cand_arr))
        f_sc, qf_f, qr_f = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        f_sc_np = np.asarray(f_sc)
        f_raw_di_np = np.asarray(qf_f / qr_f)


        r_best_idx = int(np.argmax(f_sc_np))
        if f_sc_np[r_best_idx] > best_fine_score:
            best_fine_score = float(f_sc_np[r_best_idx])
            best_fine_raw_diodicity = float(f_raw_di_np[r_best_idx])
            best_fine_params = cand_arr[r_best_idx].copy()
            best_fine_mask = np.asarray(m_f[r_best_idx], dtype=np.float32)
            print(f"  NEW BEST PARAMS (raw [0,1]): {best_fine_params}")


        # Update elite pool: add new candidates, keep top 8 overall
        combined_pool = np.concatenate([elite_pool, cand_arr], axis=0)
        combined_scores = np.concatenate([elite_scores, f_sc_np], axis=0)
        combined_raw_dis = np.concatenate([elite_raw_dis, f_raw_di_np], axis=0)
        
        order_combined = np.argsort(combined_scores)[::-1]
        elite_pool = combined_pool[order_combined[:8]].copy()
        elite_scores = combined_scores[order_combined[:8]].copy()
        elite_raw_dis = combined_raw_dis[order_combined[:8]].copy()


        print(
            f"[design.py] Fine Evolution Round {r+1:02d}/{n_fine_rounds} | Round Best Score = {f_sc_np[r_best_idx]:.4f} "
            f"| Overall Best Fine Score = {best_fine_score:.4f} (Raw Di = {best_fine_raw_diodicity:.4f}) | Elapsed = {time.time()-t0:.1f}s"
        )
        

    # After all rounds, re-render the truly best parameters to ensure the returned mask is the best one.
    final_best_mask_from_params = render_batch_fine(jnp.asarray(best_fine_params[None, :]))[0]

    return np.asarray(final_best_mask_from_params)
