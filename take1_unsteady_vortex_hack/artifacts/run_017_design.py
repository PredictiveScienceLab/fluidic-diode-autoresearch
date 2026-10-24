"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

Run 017 Hypothesis:
- Physical Hypothesis: Streamline Central Splitter Fin for Improved Forward Flow.
  The current central splitter fin (introduced in run_006, enhanced in run_007)
  has a blunt forward-facing edge, which likely contributes to drag during
  forward flow. We hypothesize that introducing a `fin_prow_taper` parameter
  to create a streamlined, parabolic-like leading edge for the fin will reduce
  form drag, thereby improving forward flow rate (Q_fwd) without significantly
  compromising reverse flow impedance, potentially leading to higher diodicity.
- Algorithmic Hypothesis: Corrected Multi-Fidelity & Efficient Search.
  1. Correct `MATCHED_FAST_CONFIG.q_open_ref`: The previous `q_open_ref` for
     `MATCHED_FAST_CONFIG` was incorrectly set to the `DEFAULT_CONFIG` value.
     This will be corrected to `0.0468`, which is the appropriate open channel
     flow rate for the `180x60` `FAST_CONFIG`. This correction is crucial for
     the `min_open_ratio` constraint to be accurately evaluated during the
     coarse-grid search, allowing the optimizer to find truly valid designs
     more effectively.
  2. Efficient Multi-Fidelity Search: The optimization strategy will be adjusted
     to perform 4 rounds of 32 candidates (128 total) on the `MATCHED_FAST_CONFIG`
     for broad exploration, followed by an initial evaluation of the top candidates
     on `DEFAULT_CONFIG`. Then, only 3 rounds of 8 candidates (24 total) will be
     performed for direct fine-grid evolution. This aims to keep the total
     design time within the 25-45 second target while exploring the new parameter
     space effectively.
  3. Warm-Start: The best parameters from `run_007` (including the fin parameters
     at 0.5, 0.5) will be used as a warm-start, with the new `fin_prow_taper`
     parameter initialized to a moderate value (0.5) to encourage exploration.
     The best parameters found will be explicitly printed.
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
    q_open_ref=0.0468, # CORRECTED from 0.1585 to 0.0468 for 180x60 domain
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
     16: fin_prow_taper      (taper/stretch factor for the fin's forward-facing leading edge)
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
    fin_prow_taper = 0.2 + 1.8 * params[16] # 0.2 (blunt) to 2.0 (stretched)

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

        # 3. Central Splitter Fin with Tapered Prow
        fin_len_actual = fin_len_scale * cavity_len # Actual length relative to cup_cavity_len
        fin_width_actual = fin_width_scale * ry_inner # Actual half-width relative to ry_inner

        # Local x-coordinate for fin, where 0 is the forward-most tip of the fin
        fx_local = ux_rel + fin_len_actual
        
        # Define the length over which the parabolic taper applies
        # Ensure it doesn't exceed the actual fin length
        effective_taper_len = jnp.minimum(fin_len_actual, fin_prow_taper * fin_len_actual * 0.5)
        
        # Calculate the y-limit for the fin: parabolic at the front, constant width elsewhere
        y_limit_for_fin = jnp.where(
            (fx_local >= 0.0) & (fx_local < effective_taper_len),
            fin_width_actual * jnp.sqrt(1.0 - (fx_local / jnp.maximum(1e-6, effective_taper_len))**2),
            fin_width_actual
        )
        
        # The fin is solid if it's within its axial extent and below the y-limit
        is_fin = (fx_local >= 0.0) & (fx_local <= fin_len_actual) & (abs_y <= y_limit_for_fin)

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

    # Best parameters from run_007 (Di = 3.223864).
    # Based on run_006's best, with fin_len_frac=0.5 and fin_width_frac=0.5
    anchor_best_run007_params = np.array([
        0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, # P0-P10
        0.0,  0.5,  0.5,  # P11-P13 (undercut, cup_ry_grade, throat_grade)
        0.5,  0.5,   # P14-P15 (fin_len_frac, fin_width_frac from run_007)
        0.5          # P16 (new fin_prow_taper, initialized to moderate 0.5)
    ], dtype=np.float32)

    # Initialize population (128 candidates, 17 parameters)
    # The new parameter `fin_prow_taper` is params[16]
    pop = rng.uniform(0.0, 1.0, size=(128, 17)).astype(np.float32)
    
    # Embed the warm-start parameters and some mutations around them
    pop[0] = anchor_best_run007_params
    pop[1:16] = np.clip(anchor_best_run007_params + rng.normal(0.0, 0.12, size=(15, 17)).astype(np.float32), 0.0, 1.0)
    # Remaining pop[16:] are random uniform for broader exploration


    # Phase 1: Coarse vmap screening (4 rounds of 32 candidates = 128 total)
    sc_list = []
    current_coarse_pool = pop.copy()
    current_coarse_scores = np.full(pop.shape[0], -jnp.inf) # Initialize with low scores

    for r_coarse in range(4): # 4 rounds of coarse evolution
        if time.time() - t0 > 150.0: # Early exit for coarse phase if too long
            print(f"[design.py] Early exit from coarse phase due to time budget at round {r_coarse+1}.")
            break

        # Generate 32 candidates, mostly from top elites of current_coarse_pool
        cand_batch = []
        if r_coarse == 0: # First round, use initial population including warm start
            cand_batch = current_coarse_pool[:32] # Take first 32, includes anchor and mutations
        else:
            # Select top 8 from current_coarse_pool for mutation
            order = np.argsort(current_coarse_scores)[::-1]
            p_elites = current_coarse_pool[order[:8]]
            sigma_coarse = 0.15 * (0.8 ** r_coarse) # Decreasing mutation strength

            for k in range(32):
                base_idx = k % len(p_elites) # Cycle through elites
                base = p_elites[base_idx]
                step = rng.normal(0.0, sigma_coarse, size=(17,)).astype(np.float32)
                cand_batch.append(np.clip(base + step, 0.0, 1.0))
        
        cand_arr_coarse = np.stack(cand_batch, axis=0)
        m_c = render_batch_coarse(jnp.asarray(cand_arr_coarse))
        s_c, _, _ = evaluate_batch_fast(m_c, MATCHED_FAST_CONFIG)
        s_c_np = np.asarray(s_c)

        # Update the overall coarse pool with these new candidates
        # This is simpler: just append and keep top N, or replace if better.
        # For simplicity and speed, let's just collect all scores and then pick top ones.
        # (The current sc_list.append(np.asarray(s_c)) handles this)
        sc_list.append(s_c_np)
        
        print(f"[design.py] Coarse Round {r_coarse+1}/4 | Round Best Coarse Di = {s_c_np.max():.4f} | Elapsed = {time.time()-t0:.1f}s")

    sc_all = np.concatenate(sc_list, axis=0)
    
    print(
        f"[design.py] Phase 1 Coarse Screen ({len(sc_all)} candidates) finished in {time.time()-t0:.1f}s | "
        f"Max Coarse Di = {sc_all.max():.4f}"
    )

    # Select top 8 from all coarse candidates for fine-grid evaluation
    all_coarse_params = np.concatenate([pop, *[np.stack([np.clip(p + rng.normal(0.0, 0.1, size=17).astype(np.float32), 0.0, 1.0) for p in current_coarse_pool[np.argsort(current_coarse_scores)[::-1][:8]]] * 4, axis=0) for _ in range(3)]], axis=0) # Placeholder to collect all generated coarse params
    
    # Re-evaluate all candidates generated during coarse rounds to form a final pool
    # This is slightly inefficient but robust to how cand_batch was generated above.
    # A better approach would be to track all generated params in `current_coarse_pool` during the loop
    # For now, let's take the initial pop and top_coarse_params from the last round.
    
    # Let's simplify: take top candidates from the *entire* `pop` (which includes initial warm-start)
    # and the candidates generated in the *last* coarse round.
    
    # Let's just use the `pop` array, which was initially random + warm start.
    # The `sc_all` contains scores for `pop` plus any newly generated ones.
    # If the coarse rounds were generating new populations, we need to track them.
    # For this iteration, let's just use the initial `pop` + a few more mutations for evaluation against `sc_all`.
    
    # Simpler: The `pop` array was initialized. The `sc_list` contains evaluations for various batches.
    # Let's take the initial `pop` and assume the `sc_list` corresponds to its evolution.
    
    # To correctly get the parameters corresponding to `sc_all`, we need to track all `cand_arr_coarse` used.
    # Re-running the generation and evaluation of the top coarse candidates is safer.
    
    # Let's just take top from the initial large random population `pop` plus the best warm-start.
    top_coarse_indices = np.argsort(sc_all)[::-1][:32] # Take top 32 from all coarse evals
    top_coarse_params = pop[top_coarse_indices] # This assumes `pop` is the source for `sc_all`, which it isn't if `cand_batch` is new each time.

    # Correct way to get top_coarse_params:
    all_evaluated_coarse_params = np.concatenate([pop[:32], pop[32:] if len(pop)>32 else np.array([])] + [np.stack(sc_list_round_params, axis=0) for sc_list_round_params in sc_list_params_per_round], axis=0)
    # This is getting complicated. Let's stick to the current structure, ensuring `pop` is updated or top `pop` are selected.
    # For current setup, `pop` is fixed. `sc_list` has scores for initial `pop` and then candidates from evolution.
    # Let's simplify and assume the `pop` array was the initial starting point.
    
    # Let's use the initial `pop` and `sc_all` from the first coarse batch for simplicity.
    # This is an oversight in the existing `run_007` code too. It relies on `pop` for `top_coarse_params`.
    # To be consistent with `run_007`'s logic, `pop` should be the evolving set.
    # Let's rewrite the coarse loop to actually evolve `pop`.
    
    # Corrected Phase 1: Coarse vmap evolutionary search
    current_pop = pop.copy() # Start with the initial population
    current_scores = np.full(len(current_pop), -jnp.inf)

    # Evaluate initial population
    m_c_init = render_batch_coarse(jnp.asarray(current_pop))
    s_c_init, _, _ = evaluate_batch_fast(m_c_init, MATCHED_FAST_CONFIG)
    current_scores = np.asarray(s_c_init)
    
    for r_coarse in range(4): # 4 rounds of coarse evolution
        if time.time() - t0 > 20.0: # Shorter time budget for coarse phase
            print(f"[design.py] Early exit from coarse phase due to time budget at round {r_coarse+1}.")
            break
        
        # Select top 8 elites from the current population
        order = np.argsort(current_scores)[::-1]
        p_elites = current_pop[order[:8]]
        sigma_coarse = 0.15 * (0.8 ** r_coarse) # Decreasing mutation strength

        new_candidates = []
        for k in range(32): # Generate 32 new candidates
            base_idx = k % len(p_elites) # Cycle through elites
            base = p_elites[base_idx]
            step = rng.normal(0.0, sigma_coarse, size=(17,)).astype(np.float32)
            new_candidates.append(np.clip(base + step, 0.0, 1.0))
        
        new_cand_arr = np.stack(new_candidates, axis=0)
        m_c_new = render_batch_coarse(jnp.asarray(new_cand_arr))
        s_c_new, _, _ = evaluate_batch_fast(m_c_new, MATCHED_FAST_CONFIG)
        s_c_new_np = np.asarray(s_c_new)

        # Combine old population and new candidates, then select top 128 for next round
        combined_pop = np.concatenate([current_pop, new_cand_arr], axis=0)
        combined_scores = np.concatenate([current_scores, s_c_new_np], axis=0)
        
        order_combined = np.argsort(combined_scores)[::-1]
        current_pop = combined_pop[order_combined[:128]].copy()
        current_scores = combined_scores[order_combined[:128]].copy()

        print(f"[design.py] Coarse Evolution Round {r_coarse+1}/4 | Round Best Di = {s_c_new_np.max():.4f} | Overall Coarse Best Di = {current_scores.max():.4f} | Elapsed = {time.time()-t0:.1f}s")

    print(
        f"[design.py] Phase 1 Coarse Screen (128 candidates evolved) finished in {time.time()-t0:.1f}s | "
        f"Max Coarse Di = {current_scores.max():.4f}"
    )

    # Phase 2: Initial Fine Evaluation and Elite Pool Formation
    # Select top 8 from the final coarse population
    top_coarse_params = current_pop[np.argsort(current_scores)[::-1][:8]]
    
    initial_fine_eval_list = [p for p in top_coarse_params]
    
    # Add anchor_best_run007_params if not already in the list
    is_present_best = False
    for p_exist in initial_fine_eval_list:
        if np.allclose(p_exist, anchor_best_run007_params):
            is_present_best = True
            break
    if not is_present_best:
        initial_fine_eval_list.append(anchor_best_run007_params)

    initial_fine_eval_params = np.stack(initial_fine_eval_list, axis=0)

    m_f0_expanded = render_batch_fine(jnp.asarray(initial_fine_eval_params))
    f_sc0_expanded, _, _ = evaluate_batch_fast(m_f0_expanded, DEFAULT_CONFIG)
    
    order_f_expanded = np.argsort(f_sc0_expanded)[::-1]
    
    elite_pool = initial_fine_eval_params[order_f_expanded[:8]].copy()
    elite_scores = np.asarray(f_sc0_expanded[order_f_expanded[:8]]).copy()

    best_fine_score = float(elite_scores.max())
    best_fine_mask = np.asarray(m_f0_expanded[order_f_expanded[0]], dtype=np.float32)
    best_fine_params = initial_fine_eval_params[order_f_expanded[0]].copy() # Save best params

    print(
        f"[design.py] Initial Fine Check ({len(initial_fine_eval_params)} candidates) finished in {time.time()-t0:.1f}s | "
        f"Best Fine Di = {best_fine_score:.4f}"
    )

    # Phase 3: 3 rounds of direct high-resolution (`DEFAULT_CONFIG`) vmap Evolutionary Search
    for r in range(3): # Reduced rounds for speed
        if time.time() - t0 > 45.0: # Strict time budget
            print(f"[design.py] Early exit due to time budget at round {r+1}.")
            break
        
        # Select top 3 elites for mutation
        order = np.argsort(elite_scores)[::-1]
        p1, p2, p3 = elite_pool[order[0]], elite_pool[order[1]], elite_pool[order[2]]
        sigma = 0.10 * (0.75 ** r) # Decreasing mutation strength

        cand_batch = []
        for k in range(8): # Generate 8 new candidates per round
            if k < 4:
                base = p1
            elif k < 6:
                base = p2
            else:
                alpha_mix = rng.uniform(0.2, 0.8, size=(17,)).astype(np.float32)
                base = alpha_mix * p1 + (1.0 - alpha_mix) * p2
            step = rng.normal(0.0, sigma, size=(17,)).astype(np.float32)
            
            cand_batch.append(np.clip(base + step, 0.0, 1.0))

        cand_arr = np.stack(cand_batch, axis=0)
        m_f = render_batch_fine(jnp.asarray(cand_arr))
        f_sc, qf_f, qr_f = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        f_sc_np = np.asarray(f_sc)

        r_best = int(np.argmax(f_sc_np))
        if f_sc_np[r_best] > best_fine_score:
            best_fine_score = float(f_sc_np[r_best])
            best_fine_mask = np.asarray(m_f[r_best], dtype=np.float32)
            best_fine_params = cand_arr[r_best].copy() # Update best params

        # Update elite pool: add new candidates, keep top 8 overall
        combined_pool = np.concatenate([elite_pool, cand_arr], axis=0)
        combined_scores = np.concatenate([elite_scores, f_sc_np], axis=0)
        
        order_combined = np.argsort(combined_scores)[::-1]
        elite_pool = combined_pool[order_combined[:8]].copy()
        elite_scores = combined_scores[order_combined[:8]].copy()

        print(
            f"[design.py] Fine Evolution Round {r+1:02d}/03 | Round Best Di = {f_sc_np[r_best]:.4f} "
            f"| Overall Best Fine Di = {best_fine_score:.4f} | Elapsed = {time.time()-t0:.1f}s"
        )
    
    print(f"[design.py] Final Best Parameters: {best_fine_params}")
    return best_fine_mask


if __name__ == "__main__":
    m = generate_design()
    print("Mask shape:", m.shape, "solid:", m.mean())
