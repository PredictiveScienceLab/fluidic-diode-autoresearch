"""
Fluidic Diode Design & Continuous Optimization Module (run_010).

Run 010 Hypothesis:
- Algorithmic: Reimplement multi-fidelity optimization to significantly reduce 
  design time. Perform the bulk of the evolutionary search (4 rounds, 32 population) 
  using `MATCHED_FAST_CONFIG` to quickly explore the parameter space within ~15 seconds. 
  Then, take the top 8 candidates from the fast search (including the run_008 best) 
  and evaluate them with `DEFAULT_CONFIG` for final selection, which should take 
  another ~24 seconds. This ensures the total design time is well within the 25-45s target.
- Physical: No new explicit physical parameter or structural change. The faster 
  search should allow for more thorough exploration of the existing parameter space, 
  potentially leading to finer-tuned geometry for both forward flow efficiency 
  and reverse flow impedance. The definition of `lip_angle` is preserved from run_008.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Define a fast configuration that matches the simulator's FAST_CONFIG for coarse evaluation
MATCHED_FAST_CONFIG = SimConfig(
    nx=180, ny=60, pad_x=15, nu=0.0012, d_rho=0.03, n_steps=3200,
    avg_window=800, max_solid_fraction=0.60, min_open_ratio=0.12, q_open_ref=0.1585,
)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Renders the 2D fluidic diode mask based on 17 continuous parameters.
    Params (17): 0-13 (base geometry), 14-15 (fin), 16 (tip_lip_angle)
    """
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, \
    ry_grade, th_grade, fin_len_s, fin_width_s, lip_angle = params

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    # Calculate pitch and margin based on pitch_scale
    pitch = (0.78 + 0.16 * pitch_scale) / 3
    margin_x = 0.5 * (1.0 - (0.78 + 0.16 * pitch_scale))
    aspect = (nx_design * pitch) / (ny * 0.5)

    # Render 3 stages of the diode
    for k in range(3):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        ux_rel = (dx - (0.32 + 0.38 * cup_offset)) * aspect
        
        # Wall nozzle geometry
        y_wall = 1.0 - (1.0 - jnp.clip(0.27 + 0.32 * throat_base * (1 + (-0.15 + 0.3*th_grade)*(k-1)), 0.22, 0.68)) * (jnp.clip(dx/ (0.16 + 0.36 * nozzle_len), 0, 1) ** (0.6 + 1.1 * nozzle_pow))
        wall_nozzle = (dx >= 0.0) & (dx <= (0.16 + 0.36 * nozzle_len) - (0.12 * undercut) * jnp.clip((abs_y - 0.2)/0.5, 0, 1)) & (abs_y >= y_wall)
        
        # U-cup with lip geometry (definition preserved from run_008)
        ry_local = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1 + (-0.15 + 0.3*ry_grade)*(k-1)) + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0, 0.7), 0.08, 0.42)
        in_cup = (ux_rel > 0) & (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local) & \
                 ~((ux_rel > (0.12 + 0.58 * cavity_len) - 0.1) & (abs_y > ry_local - (0.042 + 0.075 * wall_thick) - lip_angle*0.1))
        
        # Central splitter fin geometry
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -(0.8 * fin_len_s) * (0.12 + 0.58 * cavity_len)) & (abs_y <= (0.8 * fin_width_s) * (ry_local - (0.042 + 0.075 * wall_thick)))
        
        # Combine all solid parts
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

# Jitted and vmapped render functions for batch processing at different resolutions
render_batch_default = jax.jit(jax.vmap(lambda p: render_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))
render_batch_fast = jax.jit(jax.vmap(lambda p: render_diode(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny)))


def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(505) # Consistent RNG seed for reproducibility
    
    # Best parameters from run_008, used as a warm-start
    best_p_run008 = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    num_params = len(best_p_run008)

    # --- Coarse-grid evolutionary search using MATCHED_FAST_CONFIG ---
    COARSE_POP_SIZE = 32
    COARSE_ROUNDS = 4 # Each round takes ~3.5s. Aim for ~14s total for coarse search.
    COARSE_TIME_BUDGET = 15 # Seconds
    
    # Initialize population with mutations around best_p_run008
    pop_coarse = np.clip(best_p_run008 + rng.normal(0, 0.1, (COARSE_POP_SIZE, num_params)), 0, 1).astype(np.float32)
    pop_coarse[0] = best_p_run008 # Ensure best_p_run008 is always in the initial population

    elite_pool_coarse = np.empty((COARSE_POP_SIZE, num_params), dtype=np.float32)
    elite_scores_coarse = np.full(COARSE_POP_SIZE, -np.inf, dtype=np.float32)
    
    # Evaluate initial population and update elites
    m_initial_coarse = render_batch_fast(jnp.asarray(pop_coarse))
    sc_initial_coarse, _, _ = evaluate_batch_fast(m_initial_coarse, MATCHED_FAST_CONFIG)
    
    # Combine initial pop and scores, sort, and take top COARSE_POP_SIZE
    combined_initial = np.concatenate([elite_pool_coarse, pop_coarse])
    scores_initial = np.concatenate([elite_scores_coarse, np.asarray(sc_initial_coarse)])
    idx_initial = np.argsort(scores_initial)[-COARSE_POP_SIZE:]
    elite_pool_coarse, elite_scores_coarse = combined_initial[idx_initial], scores_initial[idx_initial]

    for r in range(COARSE_ROUNDS):
        if time.time() - t0 > COARSE_TIME_BUDGET:
            print(f"Coarse search timed out after {r} rounds ({time.time() - t0:.2f}s).")
            break
        
        # Mutate from current elites to form next population
        # Sample parents from current elite_pool_coarse
        parent_indices = rng.integers(0, COARSE_POP_SIZE, COARSE_POP_SIZE)
        # Mutation strength decays with rounds
        next_pop_coarse = np.clip(elite_pool_coarse[parent_indices] + rng.normal(0, 0.08 * (0.9**r), (COARSE_POP_SIZE, num_params)), 0, 1)
        next_pop_coarse[0] = elite_pool_coarse[-1] # Always include the best current coarse elite for robustness
        
        m_coarse = render_batch_fast(jnp.asarray(next_pop_coarse))
        sc_coarse, _, _ = evaluate_batch_fast(m_coarse, MATCHED_FAST_CONFIG)
        
        # Update elites
        combined_coarse = np.concatenate([elite_pool_coarse, next_pop_coarse])
        scores_coarse = np.concatenate([elite_scores_coarse, np.asarray(sc_coarse)])
        idx_coarse = np.argsort(scores_coarse)[-COARSE_POP_SIZE:]
        elite_pool_coarse, elite_scores_coarse = combined_coarse[idx_coarse], scores_coarse[idx_coarse]
        
    print(f"Coarse search finished. Best coarse score: {elite_scores_coarse[-1]:.4f} (took {time.time() - t0:.2f}s)")
    
    # --- Fine-grid evaluation using DEFAULT_CONFIG ---
    FINE_EVAL_COUNT = 8 # Evaluate top N designs on DEFAULT_CONFIG (Batch of 8 takes ~24s)
    
    # Select candidates for fine evaluation:
    # 1. The very best from run_008 (best_p_run008)
    # 2. The top (FINE_EVAL_COUNT - 1) candidates from the coarse search
    
    fine_candidates_list = [best_p_run008]
    
    # Add top coarse elites, ensuring we don't duplicate best_p_run008
    # Iterate from the best coarse elite downwards
    for p in reversed(elite_pool_coarse):
        if len(fine_candidates_list) >= FINE_EVAL_COUNT:
            break
        # Check if this candidate is already very close to best_p_run008 (using a small tolerance)
        if not np.allclose(p, best_p_run008, atol=1e-5):
            fine_candidates_list.append(p)
            
    fine_candidates = np.array(fine_candidates_list).astype(np.float32)
    
    print(f"Evaluating {len(fine_candidates)} candidates on DEFAULT_CONFIG...")
    m_fine = render_batch_default(jnp.asarray(fine_candidates))
    sc_fine, _, _ = evaluate_batch_fast(m_fine, DEFAULT_CONFIG)
    
    best_idx = np.argmax(sc_fine)
    final_best_p = fine_candidates[best_idx]
    
    print(f"Fine evaluation finished. Best fine score: {sc_fine[best_idx]:.4f}")
    print(f"Total design time: {time.time() - t0:.2f}s")
    
    # Return the best design found after fine-grid evaluation
    return render_diode(final_best_p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)
