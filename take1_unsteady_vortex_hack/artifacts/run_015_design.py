"""
Fluidic Diode Design & Continuous Optimization Module (run_015).

Run 015 Hypothesis:
- Physical: Introduce a `fin_prow_taper` parameter to control the shape of the
  central splitter fin's upstream (forward-facing) edge. A higher taper value
  will make the fin narrower towards the upstream end, aiming to reduce form
  drag and improve forward flow efficiency, while maintaining its function in
  impeding reverse flow.
- Algorithmic: Reimplement a robust multi-fidelity optimization strategy to
  significantly reduce the design time. Perform the bulk of the evolutionary
  search (8 rounds, 32 population) using `MATCHED_FAST_CONFIG` to quickly
  explore the parameter space. Then, evaluate the top 8 elites using
  `DEFAULT_CONFIG` for final selection. This targets a total design time of
  30-40 seconds, allowing for more rapid iteration and exploration.
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
        
        # U-cup with lip (lip_angle currently makes the opening larger, effectively reducing trap)
        ry_local = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1 + (-0.15 + 0.3*ry_grade)*(k-1)) + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0, 0.7), 0.08, 0.42)
        in_cup = (ux_rel > 0) & (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local) & \
                 ~((ux_rel > (0.12 + 0.58 * cavity_len) - 0.1) & (abs_y > ry_local - (0.042 + 0.075 * wall_thick) - lip_angle*0.1))
        
        # Central Splitter Fin with Prow Taper
        fin_base_half_width = (0.8 * fin_width_s) * (ry_local - (0.042 + 0.075 * wall_thick))
        fin_max_len = (0.8 * fin_len_s) * (0.12 + 0.58 * cavity_len)

        # Tapering factor: 1.0 at ux_rel=0 (downstream end), decreases towards 0 at ux_rel=-fin_max_len (upstream end)
        # fin_prow_taper = 0 means no taper (blunt fin leading edge)
        # fin_prow_taper = 1 means sharp point at the upstream end
        taper_factor = 1.0 - fin_prow_taper * jnp.clip(-ux_rel / fin_max_len, 0.0, 1.0)
        
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_max_len) & (abs_y <= fin_base_half_width * taper_factor)
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

def make_render_batch(config: SimConfig):
    """Creates a vmap'd render function for a given config's design domain size."""
    return jax.jit(jax.vmap(lambda p: render_diode(p, config.nx - 2 * config.pad_x, config.ny)))

render_batch_fast = make_render_batch(MATCHED_FAST_CONFIG)
render_batch_default = make_render_batch(DEFAULT_CONFIG)

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(505)

    # Best parameters from run_008 (17 parameters) + new fin_prow_taper (0.5 for moderate taper)
    best_p_run_008 = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    initial_fin_prow_taper = 0.5 # Start with a moderate taper
    best_p = np.append(best_p_run_008, initial_fin_prow_taper).astype(np.float32)
    
    # Multi-fidelity optimization parameters
    N_POP_SIZE = 32          # Population size for each round (batch size for vmap)
    N_ROUNDS_FAST = 8        # Number of evolutionary rounds using FAST_CONFIG
    N_ELITES_FAST = 8        # Number of top designs to carry over and mutate from
    N_FINE_EVALS = 8         # Number of top designs from fast search to evaluate on DEFAULT_CONFIG
    
    # Initialize population for fast search
    pop = np.clip(best_p + rng.normal(0, 0.1, (N_POP_SIZE, len(best_p))), 0, 1).astype(np.float32)
    pop[0] = best_p # Ensure the current best design is always in the initial population
    
    elite_pool = pop[:N_ELITES_FAST].copy() # Initialize elite pool with initial candidates
    elite_scores = np.full(N_ELITES_FAST, -jnp.inf) # Initialize with very low scores

    # Fast search phase using MATCHED_FAST_CONFIG
    print(f"Starting fast search ({N_ROUNDS_FAST} rounds, {N_POP_SIZE} pop/round) on MATCHED_FAST_CONFIG...")
    for r in range(N_ROUNDS_FAST):
        # Allow up to 30 seconds for the fast search phase to leave budget for fine evaluation
        if time.time() - t0 > 30.0:
            print(f"Exiting fast search early after round {r} due to time budget.")
            break
        
        m_fast = render_batch_fast(jnp.asarray(pop))
        sc_fast, _, _ = evaluate_batch_fast(m_fast, MATCHED_FAST_CONFIG)
        
        # Update elites based on fast config scores
        combined_pool = np.concatenate([elite_pool, pop])
        combined_scores = np.concatenate([elite_scores, np.asarray(sc_fast)])
        
        # Sort by score and take the top N_ELITES_FAST
        idx = np.argsort(combined_scores)[-N_ELITES_FAST:]
        elite_pool, elite_scores = combined_pool[idx], combined_scores[idx]
        
        # Mutate population for the next round from the elite pool
        # Reduce mutation strength over rounds
        mutation_strength = 0.05 * (0.95**r)
        pop = np.clip(elite_pool[rng.integers(0, N_ELITES_FAST, N_POP_SIZE)] + 
                       rng.normal(0, mutation_strength, (N_POP_SIZE, len(best_p))), 0, 1)

    # Fine-grained evaluation phase using DEFAULT_CONFIG
    print(f"Starting fine evaluation ({N_FINE_EVALS} designs) on DEFAULT_CONFIG...")
    
    # Take the top N_FINE_EVALS from the elite pool found during fast search
    # Ensure the original best_p is included to prevent regression
    fine_eval_params = elite_pool[-N_FINE_EVALS:].copy()
    if not any(np.array_equal(p, best_p) for p in fine_eval_params):
        # If original best_p is not among the top elites, replace the worst with it
        fine_eval_params[0] = best_p
    
    m_fine = render_batch_default(jnp.asarray(fine_eval_params))
    sc_fine, _, _ = evaluate_batch_fast(m_fine, DEFAULT_CONFIG)
    
    # Select the best design from the fine evaluations
    best_idx_fine = np.argmax(sc_fine)
    final_best_p = fine_eval_params[best_idx_fine]
    
    print(f"Design time: {time.time() - t0:.2f}s")
    return render_diode(final_best_p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)
