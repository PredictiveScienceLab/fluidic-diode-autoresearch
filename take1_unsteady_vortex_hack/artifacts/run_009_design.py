"""
Fluidic Diode Design & Continuous Optimization Module (run_009).

Run 009 Hypothesis:
- Physical: Introduce a `fin_prow_taper` parameter to control the shape of the central splitter fin's upstream (forward-facing) edge. A higher taper value will make the fin narrower towards the centerline at its upstream end, potentially reducing forward flow resistance while maintaining reverse flow impedance.
- Algorithmic: Implement a multi-fidelity optimization strategy. Perform the main evolutionary search (10 rounds of 32 candidates) using `MATCHED_FAST_CONFIG` for speed. Then, take the top 8 elite candidates from this coarse search, evaluate them on `DEFAULT_CONFIG` in a single batch, and return the best one. This should significantly reduce the total design time to the target 25-45 seconds.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

MATCHED_FAST_CONFIG = SimConfig(
    nx=180, ny=60, pad_x=15, nu=0.0012, d_rho=0.03, n_steps=3200,
    avg_window=800, max_solid_fraction=0.60, min_open_ratio=0.12, q_open_ref=0.1585,
)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Params (18): 0-13 (base), 14-15 (fin), 16 (tip_lip_angle), 17 (fin_prow_taper)
    """
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, \
    ry_grade, th_grade, fin_len_s, fin_width_s, lip_angle, fin_prow_taper = params

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    pitch = (0.78 + 0.16 * pitch_scale) / 3
    margin_x = 0.5 * (1.0 - (0.78 + 0.16 * pitch_scale))
    aspect = (nx_design * pitch) / (ny * 0.5)

    for k in range(3):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        ux_rel = (dx - (0.32 + 0.38 * cup_offset)) * aspect
        
        # Wall nozzle
        y_wall = 1.0 - (1.0 - jnp.clip(0.27 + 0.32 * throat_base * (1 + (-0.15 + 0.3*th_grade)*(k-1)), 0.22, 0.68)) * (jnp.clip(dx/ (0.16 + 0.36 * nozzle_len), 0, 1) ** (0.6 + 1.1 * nozzle_pow))
        wall_nozzle = (dx >= 0.0) & (dx <= (0.16 + 0.36 * nozzle_len) - (0.12 * undercut) * jnp.clip((abs_y - 0.2)/0.5, 0, 1)) & (abs_y >= y_wall)
        
        # U-cup with lip
        ry_local = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1 + (-0.15 + 0.3*ry_grade)*(k-1)) + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0, 0.7), 0.08, 0.42)
        in_cup = (ux_rel > 0) & (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local) & \
                 ~((ux_rel > (0.12 + 0.58 * cavity_len) - 0.1) & (abs_y > ry_local - (0.042 + 0.075 * wall_thick) - lip_angle*0.1))
        
        # Fin with prow taper
        fin_start_x_rel = -(0.8 * fin_len_s) * (0.12 + 0.58 * cavity_len)
        fin_end_x_rel = 0.0
        fin_max_width_y = (0.8 * fin_width_s) * (ry_local - (0.042 + 0.075 * wall_thick))

        # Calculate tapered width along the fin's length
        # `taper_progress` is 0 at `fin_end_x_rel` (downstream) and 1 at `fin_start_x_rel` (upstream)
        taper_progress = jnp.clip((ux_rel - fin_end_x_rel) / (fin_start_x_rel - fin_end_x_rel + 1e-6), 0, 1)
        fin_width_at_x = fin_max_width_y * (1.0 - fin_prow_taper * taper_progress)

        is_fin = (ux_rel >= fin_start_x_rel) & (ux_rel <= fin_end_x_rel) & (abs_y <= fin_width_at_x)
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

# Render batch for coarse search (MATCHED_FAST_CONFIG)
render_batch_coarse = jax.jit(jax.vmap(lambda p: render_diode(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny)))

# Render batch for fine evaluation (DEFAULT_CONFIG)
render_batch_fine = jax.jit(jax.vmap(lambda p: render_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(505) # Consistent seed

    # Current best parameters from run_008, with fin_prow_taper initialized to 0.5
    best_p_run008 = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    # Append the new parameter fin_prow_taper (18th parameter)
    current_best_params = np.append(best_p_run008, 0.5).astype(np.float32)
    
    # Initialize population for coarse search
    # Add a bit of noise around current_best_params for exploration
    pop = np.clip(current_best_params + rng.normal(0, 0.1, (32, len(current_best_params))), 0, 1).astype(np.float32)
    pop[0] = current_best_params # Ensure the current best is always part of the initial population for exploration

    elite_pool_coarse = pop[:8]
    elite_scores_coarse = np.zeros(8)
    
    # Coarse search loop using MATCHED_FAST_CONFIG
    num_coarse_rounds = 10
    for r in range(num_coarse_rounds):
        # Break early if exceeding time budget for coarse search
        # Aim for ~30-35s for coarse search (10 rounds * ~3.5s per batch for B=32)
        if time.time() - t0 > 30: 
            print(f"Coarse search cut short at round {r+1}/{num_coarse_rounds} due to time limit.")
            break
        
        m_coarse = render_batch_coarse(jnp.asarray(pop))
        sc_coarse, _, _ = evaluate_batch_fast(m_coarse, MATCHED_FAST_CONFIG)
        
        # Update elites for the coarse pool
        combined_coarse = np.concatenate([elite_pool_coarse, pop])
        scores_coarse = np.concatenate([elite_scores_coarse, np.asarray(sc_coarse)])
        idx_coarse = np.argsort(scores_coarse)[-8:] # Keep top 8 elites
        elite_pool_coarse, elite_scores_coarse = combined_coarse[idx_coarse], scores_coarse[idx_coarse]
        
        # Mutate population for next round based on elites
        pop = np.clip(elite_pool_coarse[rng.integers(0, 8, 32)] + rng.normal(0, 0.05 * (0.95**r), (32, len(current_best_params))), 0, 1)

    # Fine-grid evaluation of top elites from coarse search using DEFAULT_CONFIG
    # Ensure the globally best parameter set (current_best_params) is included
    # in the fine evaluation batch to prevent regression, alongside the top coarse elites.
    
    fine_eval_params_list = list(elite_pool_coarse)
    
    # Add current_best_params if it's not already in the elite_pool_coarse
    # This prevents regression from the best score so far.
    if not any(np.allclose(p, current_best_params) for p in fine_eval_params_list):
        fine_eval_params_list.append(current_best_params)

    # Ensure batch size for fine evaluation is reasonable (max 8-9)
    # If adding current_best_params made it >8, trim to 8 by taking top coarse elites
    if len(fine_eval_params_list) > 8:
        # Sort by coarse score to pick the best, then ensure current_best_params is included
        # (It's already in fine_eval_params_list if not in elite_pool_coarse)
        # We need to re-evaluate coarse scores for the `fine_eval_params_list` as `elite_scores_coarse` only
        # corresponds to `elite_pool_coarse`.
        # For simplicity, let's just use the top 8 from the coarse search + `current_best_params` if it's not already among them.
        fine_eval_params = elite_pool_coarse
        if not any(np.allclose(p, current_best_params) for p in elite_pool_coarse):
            fine_eval_params = np.vstack([elite_pool_coarse[:7], current_best_params]) # Take 7 best from coarse + current_best
        else:
            fine_eval_params = elite_pool_coarse # current_best_params is already in top 8 coarse
    else:
        fine_eval_params = np.array(fine_eval_params_list)
    
    # Evaluate the selected fine candidates
    m_fine = render_batch_fine(jnp.asarray(fine_eval_params))
    sc_fine, _, _ = evaluate_batch_fast(m_fine, DEFAULT_CONFIG)
    
    best_idx_fine = jnp.argmax(sc_fine)
    final_design_params = fine_eval_params[best_idx_fine]
    
    return render_diode(final_design_params, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)
