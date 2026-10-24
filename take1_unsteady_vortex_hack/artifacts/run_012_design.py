"""
Fluidic Diode Design & Continuous Optimization Module (run_012).

Run 012 Hypothesis:
- Physical: Introduce a `fin_prow_taper` parameter to streamline the forward-facing edge of the central splitter fin.
  This aims to reduce drag and improve forward flow conductance without compromising reverse flow impedance,
  potentially increasing diodicity.
- Algorithmic: Implement a robust multi-fidelity optimization strategy to achieve the 25-45s design time target.
  This involves a large population coarse search using `MATCHED_FAST_CONFIG` for broad exploration,
  followed by a focused fine-grid refinement using `DEFAULT_CONFIG` that explicitly includes the previous best elite
  to prevent regression.
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
        
        # Fin with taper
        fin_max_len = (0.8 * fin_len_s) * (0.12 + 0.58 * cavity_len)
        fin_max_width = (0.8 * fin_width_s) * (ry_local - (0.042 + 0.075 * wall_thick))
        
        # taper_progress is 0 at downstream end (ux_rel=0) and 1 at upstream end (ux_rel=-fin_max_len)
        taper_progress = jnp.clip(-ux_rel / fin_max_len, 0, 1) 
        current_width_factor = 1.0 - fin_prow_taper * taper_progress

        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_max_len) & (abs_y <= fin_max_width * current_width_factor)
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

render_batch = jax.jit(jax.vmap(lambda p: render_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))
render_batch_fast = jax.jit(jax.vmap(lambda p: render_diode(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(505) # Consistent seed for reproducibility
    
    # Best params from run_008 (17 params) + new fin_prow_taper (0.0 for no taper initially)
    best_p_008 = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    best_p = np.append(best_p_008, 0.0) # Add fin_prow_taper, initial value 0.0
    n_params = len(best_p)

    # --- Phase 1: Coarse Search with MATCHED_FAST_CONFIG ---
    B_COARSE = 64
    R_COARSE = 10
    
    # Initialize population around best_p, ensure best_p is included
    pop_coarse = np.clip(best_p + rng.normal(0, 0.1, (B_COARSE, n_params)), 0, 1).astype(np.float32)
    pop_coarse[0] = best_p 
    
    # Initialize elite pool for coarse search with random picks from initial population
    elite_pool_coarse = pop_coarse[rng.choice(B_COARSE, 8, replace=False)].copy()
    elite_scores_coarse = np.zeros(8) 

    print(f"Starting coarse search ({R_COARSE} rounds, {B_COARSE} pop) with MATCHED_FAST_CONFIG...")
    for r_coarse in range(R_COARSE):
        if time.time() - t0 > 20: # Budget for coarse search, ~20s
            print(f"Coarse search budget exceeded at round {r_coarse}. Breaking.")
            break
        
        m_coarse = render_batch_fast(jnp.asarray(pop_coarse))
        sc_coarse, _, _ = evaluate_batch_fast(m_coarse, MATCHED_FAST_CONFIG)
        
        # Update elites
        combined_coarse = np.concatenate([elite_pool_coarse, pop_coarse])
        scores_coarse = np.concatenate([elite_scores_coarse, np.asarray(sc_coarse)])
        idx_coarse = np.argsort(scores_coarse)[-8:] # Keep top 8 elites
        elite_pool_coarse, elite_scores_coarse = combined_coarse[idx_coarse], scores_coarse[idx_coarse]
        
        # Mutate for next round
        pop_coarse = np.clip(elite_pool_coarse[rng.integers(0, 8, B_COARSE)] + rng.normal(0, 0.08 * (0.9**(r_coarse)), (B_COARSE, n_params)), 0, 1)
        pop_coarse[0] = elite_pool_coarse[-1] # Always keep the best coarse elite in the population

    # --- Phase 2: Fine Refinement with DEFAULT_CONFIG ---
    B_FINE = 16
    R_FINE = 3 # Number of fine-tuning rounds
    K_CANDIDATES_FROM_COARSE = 8 # Take top 8 from coarse search for initial fine eval

    # Evaluate top coarse elites on DEFAULT_CONFIG to get true scores,
    # and explicitly include the original best_p from run_008 to prevent regression.
    print(f"Evaluating top {K_CANDIDATES_FROM_COARSE} coarse elites plus best_p_008 on DEFAULT_CONFIG...")
    initial_fine_eval_params = np.concatenate([elite_pool_coarse[-K_CANDIDATES_FROM_COARSE:], best_p[None, :]])
    
    m_fine_initial = render_batch(jnp.asarray(initial_fine_eval_params))
    sc_fine_initial, _, _ = evaluate_batch_fast(m_fine_initial, DEFAULT_CONFIG)
    
    # Find the overall best parameter set from this initial fine evaluation
    best_idx_overall = np.argmax(sc_fine_initial)
    current_best_fine_params = initial_fine_eval_params[best_idx_overall]
    
    # Create a new initial population for fine-tuning, centered around the best found so far
    pop_fine = np.clip(current_best_fine_params + rng.normal(0, 0.03, (B_FINE, n_params)), 0, 1).astype(np.float32)
    pop_fine[0] = current_best_fine_params # Ensure the absolute best is in the initial fine population

    # Initialize elite pool for fine search with random picks from initial population
    elite_pool_fine = pop_fine[rng.choice(B_FINE, 8, replace=False)].copy() 
    elite_scores_fine = np.zeros(8) 
    
    print(f"Starting fine refinement ({R_FINE} rounds, {B_FINE} pop) with DEFAULT_CONFIG...")
    for r_fine in range(R_FINE):
        if time.time() - t0 > 40: # Total budget for generate_design is 45s, leave some buffer
            print(f"Fine refinement budget exceeded at round {r_fine}. Breaking.")
            break

        m_fine = render_batch(jnp.asarray(pop_fine))
        sc_fine, _, _ = evaluate_batch_fast(m_fine, DEFAULT_CONFIG)
        
        # Update elites
        combined_fine = np.concatenate([elite_pool_fine, pop_fine])
        scores_fine = np.concatenate([elite_scores_fine, np.asarray(sc_fine)])
        idx_fine = np.argsort(scores_fine)[-8:]
        elite_pool_fine, elite_scores_fine = combined_fine[idx_fine], scores_fine[idx_fine]
        
        # Mutate for next round
        pop_fine = np.clip(elite_pool_fine[rng.integers(0, 8, B_FINE)] + rng.normal(0, 0.02 * (0.9**(r_fine)), (B_FINE, n_params)), 0, 1)
        pop_fine[0] = elite_pool_fine[-1] # Always keep the best fine elite in the population

    final_best_params = elite_pool_fine[-1]
    print(f"Total design time: {time.time() - t0:.2f} seconds.")
    return render_diode(final_best_params, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)
