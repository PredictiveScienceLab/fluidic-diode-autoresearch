"""
Fluidic Diode Design & Continuous Optimization Module (run_014).

Run 014 Hypothesis:
- Algorithmic: Reimplement a robust multi-fidelity optimization strategy to significantly
  reduce the design time. Perform the bulk of the evolutionary search (30 rounds, 32 population)
  using `MATCHED_FAST_CONFIG` to quickly identify promising parameter regions.
  Then, evaluate the top 8 elite candidates (ensuring the best known design from run_008 is included)
  using `DEFAULT_CONFIG` for final selection. This aims to keep the high diodicity of run_008
  while meeting the 25-45 second design time target. No new geometric parameters are introduced.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Configuration for the coarse-grid simulation (faster)
MATCHED_FAST_CONFIG = SimConfig(
    nx=180, ny=60, pad_x=15, nu=0.0012, d_rho=0.03, n_steps=3200,
    avg_window=800, max_solid_fraction=0.60, min_open_ratio=0.12, q_open_ref=0.1585,
)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Renders the 2D fluidic diode design based on a 17-parameter vector.
    
    Params (17 elements):
    0: nozzle_len      - Controls the length of the wall nozzle section.
    1: throat_base     - Base width of the main channel throat.
    2: nozzle_pow      - Exponent for the nozzle shape, affecting curvature.
    3: cup_offset      - Horizontal offset of the U-cup structure.
    4: cup_ry_base     - Base vertical radius of the U-cup.
    5: prow_stretch    - Stretches the 'prow' or leading edge of the cup.
    6: cavity_len      - Length of the U-cup cavity.
    7: wall_thick      - Thickness of the U-cup walls.
    8: nose_thick      - Thickness of the nose (forward-facing part) of the cup.
    9: pitch_scale     - Scales the overall pitch (spacing) of the stages.
    10: flare          - Flaring of the U-cup arms.
    11: undercut       - Undercut on the outer wall nozzle for reverse flow.
    12: ry_grade       - Grading of the U-cup vertical radius across stages.
    13: th_grade       - Grading of the throat thickness across stages.
    14: fin_len_s      - Scaled length of the central splitter fin.
    15: fin_width_s    - Scaled width of the central splitter fin.
    16: lip_angle      - Angle/depth of the vortex-trap lip on the U-cup arms.
    """
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, \
    ry_grade, th_grade, fin_len_s, fin_width_s, lip_angle = params

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    # Calculate pitch based on scale parameter
    total_active_len = (0.78 + 0.16 * pitch_scale)
    pitch = total_active_len / 3.0
    margin_x = 0.5 * (1.0 - total_active_len) # Center the design
    aspect = (nx_design * pitch) / (ny * 0.5) # Aspect ratio for local scaling

    for k in range(3): # Iterate over 3 stages
        x0 = margin_x + k * pitch # Starting x-position for the current stage
        dx = (x - x0) / pitch     # Normalized x-position within the current stage
        ux_rel = (dx - (0.32 + 0.38 * cup_offset)) * aspect # Relative x for cup geometry

        # Wall nozzle (outer channel walls)
        y_wall = 1.0 - (1.0 - jnp.clip(0.27 + 0.32 * throat_base * (1 + (-0.15 + 0.3*th_grade)*(k-1)), 0.22, 0.68)) * (jnp.clip(dx/ (0.16 + 0.36 * nozzle_len), 0, 1) ** (0.6 + 1.1 * nozzle_pow))
        wall_nozzle = (dx >= 0.0) & (dx <= (0.16 + 0.36 * nozzle_len) - (0.12 * undercut) * jnp.clip((abs_y - 0.2)/0.5, 0, 1)) & (abs_y >= y_wall)
        
        # U-cup with vortex-trap lip (central structures)
        ry_local = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1 + (-0.15 + 0.3*ry_grade)*(k-1)) + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0, 0.7), 0.08, 0.42)
        
        in_cup_base = (ux_rel > 0) & (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local)
        
        lip_fluid_region = (ux_rel > (0.12 + 0.58 * cavity_len) - 0.1) & \
                           (abs_y > ry_local - (0.042 + 0.075 * wall_thick) - lip_angle*0.1)
        
        in_cup = in_cup_base & ~lip_fluid_region
        
        # Central splitter fin
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -(0.8 * fin_len_s) * (0.12 + 0.58 * cavity_len)) & \
                 (abs_y <= (0.8 * fin_width_s) * (ry_local - (0.042 + 0.075 * wall_thick)))
        
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

# JIT-compiled vmap functions for batch rendering at different resolutions
render_batch_fast = jax.jit(jax.vmap(lambda p: render_diode(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny)))
render_batch_default = jax.jit(jax.vmap(lambda p: render_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(505) # Consistent RNG seed for reproducibility

    # Best parameters from run_008, which achieved Di = 18.5650
    best_p_run008 = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    # --- Coarse-grid evolutionary search using MATCHED_FAST_CONFIG ---
    
    # Initialize population by perturbing best_p_run008
    pop_size_fast = 32
    initial_std_dev = 0.1
    pop_fast = np.clip(best_p_run008 + rng.normal(0, initial_std_dev, (pop_size_fast, len(best_p_run008))), 0, 1).astype(np.float32)
    pop_fast[0] = best_p_run008 # Ensure the best known design is always in the initial population

    # Elite pool to track best designs during fast search
    elite_pool_fast = np.copy(pop_fast[:8]) # Initialize with first 8 from pop
    elite_scores_fast = np.zeros(8, dtype=np.float32)
    
    num_fast_rounds = 30 # Number of evolutionary rounds for the coarse search
    mutation_decay_rate = 0.95

    print(f"Starting {num_fast_rounds} rounds of fast-config search...")
    for r in range(num_fast_rounds):
        # Break early if exceeding a reasonable time budget for the fast search phase
        if time.time() - t0 > 20.0: 
            print(f"  Breaking fast search early at round {r} due to time budget.")
            break
        
        # Render and evaluate current population on fast config
        m_fast = render_batch_fast(jnp.asarray(pop_fast))
        sc_fast, _, _ = evaluate_batch_fast(m_fast, MATCHED_FAST_CONFIG)
        
        # Update elite pool
        combined_params = np.concatenate([elite_pool_fast, pop_fast])
        combined_scores = np.concatenate([elite_scores_fast, np.asarray(sc_fast)])
        
        # Select top 8 elites
        idx_top_elites = np.argsort(combined_scores)[-8:]
        elite_pool_fast, elite_scores_fast = combined_params[idx_top_elites], combined_scores[idx_top_elites]
        
        # Generate next population by mutating elites
        mutation_std_dev = initial_std_dev * (mutation_decay_rate**r)
        pop_fast = np.clip(elite_pool_fast[rng.integers(0, 8, pop_size_fast)] + rng.normal(0, mutation_std_dev, (pop_size_fast, len(best_p_run008))), 0, 1)
    
    print(f"Fast search completed in {time.time() - t0:.2f} seconds.")

    # --- Fine-grid evaluation of top candidates from fast search ---
    
    # Prepare candidates for final high-resolution evaluation
    # Always include the best_p_run008 to ensure no regression and a strong baseline
    num_fine_evals = 8
    
    # Sort elites from fast search by score (descending)
    sorted_elite_indices = np.argsort(elite_scores_fast)[::-1]
    top_fast_elites_params = elite_pool_fast[sorted_elite_indices]

    fine_eval_params_list = [best_p_run008] # Start with the run_008 best
    
    # Add unique top elites from fast search until we reach num_fine_evals
    for p in top_fast_elites_params:
        # Check if parameter set is already in the list (allow for small float differences)
        is_duplicate = False
        for existing_p in fine_eval_params_list:
            if np.allclose(p, existing_p, atol=1e-5): # Use a small tolerance for float comparison
                is_duplicate = True
                break
        
        if not is_duplicate and len(fine_eval_params_list) < num_fine_evals:
            fine_eval_params_list.append(p)
            
    # Pad with copies of the best_p_run008 if we don't have enough unique candidates
    while len(fine_eval_params_list) < num_fine_evals:
        fine_eval_params_list.append(best_p_run008)

    fine_eval_params_arr = np.array(fine_eval_params_list, dtype=np.float32)

    print(f"Evaluating {len(fine_eval_params_arr)} candidates on default config...")
    m_default = render_batch_default(jnp.asarray(fine_eval_params_arr))
    sc_default, _, _ = evaluate_batch_fast(m_default, DEFAULT_CONFIG)
    
    # Select the best design from the fine-grid evaluations
    best_idx_default = np.argmax(sc_default)
    final_best_p = fine_eval_params_arr[best_idx_default]
    final_best_score = sc_default[best_idx_default]

    print(f"Fine evaluation completed. Best score found: {final_best_score:.4f}.")
    print(f"Total design time: {time.time() - t0:.2f} seconds.")

    # Return the mask for the overall best design found
    return render_diode(final_best_p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)
