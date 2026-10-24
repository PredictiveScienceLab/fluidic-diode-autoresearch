"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 022 Hypothesis:
- Physical Hypothesis: Introduce a `fin_prow_taper` parameter (P16) to streamline the central splitter
  fin's upstream (forward-facing) edge. The existing fin has a blunt leading edge, which may
  contribute to drag in forward flow. By allowing the fin to taper parabolically towards its
  upstream tip, we hypothesize that forward flow resistance can be reduced, thereby increasing
  Q_fwd, while maintaining the strong reverse-flow impedance provided by the combination of
  wall undercuts and central fin blockage (as observed in run_020).
- Algorithmic Hypothesis: Multi-Fidelity Optimization with Adjusted Fine-Grid Budget.
  The parameter space is expanded to 17 dimensions. The `best_fine_params` from run_020 are
  used as a warm-start, extended with a neutral value (0.0 for rectangular fin) for the new P16.
  The number of direct high-resolution (`DEFAULT_CONFIG`) `jax.vmap` evolutionary optimization
  rounds is reduced from 10 to 5 (`5 x 8 = 40` fine-grid LBM evaluations). This, combined with
  the initial fine evaluation, targets a total fine-grid evaluation count of ~50, aiming for
  a `generate_design()` execution time within the 25-45 second range, enabling rapid exploration
  of the new parameter. Exact `best_fine_params` will be printed at each improvement.
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
     16: fin_prow_taper      (controls parabolic shape of fin's leading edge: 0=rect, 1=tapered)
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
    fin_prow_taper_param = params[16] # New parameter P16

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)

    n_stages = 3
    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        stage_centered = k - 1.0
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
        fin_width_base = fin_width_scale * ry_inner

        # Map P16 (0-1) to an effective taper power (0=rectangular, up to 4 for strong taper)
        fin_taper_power_effective = fin_prow_taper_param * 3.0 # Maps 0..1 to 0..3 for power (1 + power = 1..4)
        
        s_fin = jnp.clip(-ux_rel / jnp.maximum(0.01, fin_len_actual), 0.0, 1.0)
        
        # If fin_taper_power_effective is very low, make it rectangular. Otherwise, apply taper.
        fin_width_factor = jnp.where(
            fin_taper_power_effective < 0.01, # Threshold for effectively zero taper
            1.0, # Rectangular fin
            1.0 - (s_fin ** (1.0 + fin_taper_power_effective)) # Tapered (power ranges from 1.0 to 4.0)
        )
        fin_width_profile = fin_width_base * fin_width_factor
        
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_actual) & (abs_y <= fin_width_profile)

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

    # Best known parameters from run_020 (Di = 37.802017).
    # This is derived from the printed best_fine_params during run_020's execution.
    best_known_params_run020 = np.array([
        0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, # P0-P10
        0.5,  0.5,  0.5,  # P11-P13 (undercut, cup_ry_grade, throat_grade)
        0.5,  0.5   # P14-P15 (fin_len_frac, fin_width_frac)
    ], dtype=np.float32)

    # Extend with the new P16 (fin_prow_taper) set to a neutral value (0.0 for rectangular)
    best_known_params_run020_extended = np.append(best_known_params_run020, 0.0) # P16 = 0.0

    # New anchor: Activate undercut (P11) with no fin (P14, P15 = 0.0)
    anchor_activate_undercut_nofin_extended = best_known_params_run020_extended.copy()
    anchor_activate_undercut_nofin_extended[11] = 0.5 # Moderate undercut
    anchor_activate_undercut_nofin_extended[14] = 0.0 # No fin length
    anchor_activate_undercut_nofin_extended[15] = 0.0 # No fin width
    anchor_activate_undercut_nofin_extended[16] = 0.0 # Rectangular fin (if fin existed)

    # New anchor: Activate undercut AND ensure fin is active, rectangular fin
    anchor_combined_undercut_fin_extended = best_known_params_run020_extended.copy()
    anchor_combined_undercut_fin_extended[11] = 0.5 # Moderate undercut
    anchor_combined_undercut_fin_extended[14] = 0.5 # Moderate fin length
    anchor_combined_undercut_fin_extended[15] = 0.5 # Moderate fin width
    anchor_combined_undercut_fin_extended[16] = 0.0 # Rectangular fin


    # Initialize population (64 candidates for coarse screening, now with 17 params)
    pop = rng.uniform(0.0, 1.0, size=(64, 17)).astype(np.float32)
    
    # Embed the warm-start parameters and some mutations around them
    pop[0] = best_known_params_run020_extended # Current best from run_020 (fin active, undercut active, P16=0)
    pop[1] = anchor_activate_undercut_nofin_extended # Undercut active, no fin, P16=0
    pop[2] = anchor_combined_undercut_fin_extended # Undercut + fin active, P16=0
    
    # Mutate around the best known parameters for further exploration
    pop[3:16] = np.clip(best_known_params_run020_extended + rng.normal(0.0, 0.16, size=(13, 17)).astype(np.float32), 0.0, 1.0)
    # Remaining pop[16:] are random uniform for broader exploration


    # Phase 1: Coarse vmap screening (64 candidates)
    sc_list = []
    for i in range(0, 64, 32): # Process in batches of 32 for efficiency
        m_c = render_batch_coarse(jnp.asarray(pop[i : i + 32]))
        s_c, _, _ = evaluate_batch_fast(m_c, MATCHED_FAST_CONFIG)
        sc_list.append(np.asarray(s_c))
    sc_all = np.concatenate(sc_list, axis=0)

    print(
        f"[design.py] Phase 1 Coarse Screen (64 candidates) finished in {time.time()-t0:.1f}s | "
        f"Max Coarse Di = {sc_all.max():.4f}"
    )

    # Combine top coarse candidates with anchor points for initial fine evaluation
    top_coarse_params = pop[np.argsort(sc_all)[::-1][:6]] # Top 6 from coarse

    # Initial fine evaluation pool
    initial_fine_eval_list = []
    initial_fine_eval_list.append(best_known_params_run020_extended)
    initial_fine_eval_list.append(anchor_activate_undercut_nofin_extended)
    initial_fine_eval_list.append(anchor_combined_undercut_fin_extended)
    
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
    f_sc0_expanded, _, _ = evaluate_batch_fast(m_f0_expanded, DEFAULT_CONFIG)
    
    # Select the top 8 (or desired elite pool size) from this expanded set
    order_f_expanded = np.argsort(f_sc0_expanded)[::-1]
    
    elite_pool = initial_fine_eval_params[order_f_expanded[:8]].copy()
    elite_scores = np.asarray(f_sc0_expanded[order_f_expanded[:8]]).copy()

    best_fine_score = float(elite_scores.max())
    best_fine_params = elite_pool[np.argmax(elite_scores)].copy() # Store best params
    best_fine_mask = np.asarray(m_f0_expanded[order_f_expanded[0]], dtype=np.float32)

    print(
        f"[design.py] Initial Fine Check ({len(initial_fine_eval_params)} candidates) finished in {time.time()-t0:.1f}s | "
        f"Best Fine Di = {best_fine_score:.4f} | Best Params = {best_fine_params}"
    )

    # Phase 2: 5 rounds of direct high-resolution (`DEFAULT_CONFIG`) vmap Evolutionary Search (targeting ~50-60s total)
    for r in range(5): # Reduced rounds from 10 to 5 for faster iteration
        if time.time() - t0 > 55.0: # Early exit if approaching budget limit (aim for 25-45s, 55s for safety)
            print(f"[design.py] Early exit due to time budget at round {r+1}.")
            break
        
        # Select top 3 elites for mutation
        order = np.argsort(elite_scores)[::-1]
        p1, p2, p3 = elite_pool[order[0]], elite_pool[order[1]], elite_pool[order[2]]
        sigma = 0.11 * (0.82 ** r) # Decreasing mutation strength over rounds

        cand_batch = []
        for k in range(8): # Generate 8 new candidates per round
            if k < 4:
                base = p1
            elif k < 6:
                base = p2
            else:
                # Recombination between top 2 elites + mutation for some candidates
                alpha_mix = rng.uniform(0.2, 0.8, size=(17,)).astype(np.float32) # Updated to 17 params
                base = alpha_mix * p1 + (1.0 - alpha_mix) * p2
            step = rng.normal(0.0, sigma, size=(17,)).astype(np.float32) # Updated to 17 params
            
            cand_batch.append(np.clip(base + step, 0.0, 1.0))

        cand_arr = np.stack(cand_batch, axis=0)
        m_f = render_batch_fine(jnp.asarray(cand_arr))
        f_sc, qf_f, qr_f = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        f_sc_np = np.asarray(f_sc)

        r_best_idx = int(np.argmax(f_sc_np))
        if f_sc_np[r_best_idx] > best_fine_score:
            best_fine_score = float(f_sc_np[r_best_idx])
            best_fine_params = cand_arr[r_best_idx].copy() # Store best params
            best_fine_mask = np.asarray(m_f[r_best_idx], dtype=np.float32)
            print(f"  NEW BEST PARAMS: {best_fine_params}")


        # Update elite pool: add new candidates, keep top 8 overall
        combined_pool = np.concatenate([elite_pool, cand_arr], axis=0)
        combined_scores = np.concatenate([elite_scores, f_sc_np], axis=0)
        
        order_combined = np.argsort(combined_scores)[::-1]
        elite_pool = combined_pool[order_combined[:8]].copy()
        elite_scores = combined_scores[order_combined[:8]].copy()


        print(
            f"[design.py] Fine Evolution Round {r+1:02d}/5 | Round Best Di = {f_sc_np[r_best_idx]:.4f} "
            f"| Overall Best Fine Di = {best_fine_score:.4f} | Elapsed = {time.time()-t0:.1f}s"
        )
        

    # After all rounds, re-render the truly best parameters to ensure the returned mask is the best one.
    final_best_mask_from_params = render_batch_fine(jnp.asarray(best_fine_params[None, :]))[0]

    return np.asarray(final_best_mask_from_params)
