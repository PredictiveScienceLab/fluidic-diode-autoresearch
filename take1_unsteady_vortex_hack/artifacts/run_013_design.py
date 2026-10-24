"""
Fluidic Diode Design & Continuous Optimization Module (run_013).

Run 013 Hypothesis:
- Physical: Introduce a `fin_prow_taper` parameter to give the central splitter fin a more streamlined, parabolic-like leading edge (facing +x). This aims to reduce form drag and improve forward flow conductance by allowing the fin to taper to a point upstream.
- Algorithmic: Continue the successful `DEFAULT_CONFIG` evolution strategy from `run_008`, but reduce the number of evolutionary rounds from 20 to 15 to accommodate the new parameter and keep the design time closer to ~200s (from ~275s in `run_008`), while staying well under the 300s budget. The mutation strength decay is also slightly increased to encourage faster convergence.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# MATCHED_FAST_CONFIG is not used in this run as we stick to DEFAULT_CONFIG for more precise evolution.
# It's kept here just in case, but the primary strategy is direct DEFAULT_CONFIG evolution as in run_008.
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
        # y_wall defines the inner boundary of the wall nozzle, it's symmetric.
        y_wall_base = 1.0 - jnp.clip(0.27 + 0.32 * throat_base * (1 + (-0.15 + 0.3*th_grade)*(k-1)), 0.22, 0.68)
        # The nozzle shape curves based on dx and nozzle_pow
        y_nozzle_profile = y_wall_base * (jnp.clip(dx/ (0.16 + 0.36 * nozzle_len), 0, 1) ** (0.6 + 1.1 * nozzle_pow))
        # Undercut applies a modification to the wall shape, making it thinner near the center
        wall_nozzle = (dx >= 0.0) & (dx <= (0.16 + 0.36 * nozzle_len) - (0.12 * undercut) * jnp.clip((abs_y - 0.2)/0.5, 0, 1)) & (abs_y >= y_wall_base * y_nozzle_profile)
        
        # U-cup with lip
        ry_local = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1 + (-0.15 + 0.3*ry_grade)*(k-1)) + (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0, 0.7), 0.08, 0.42)
        # The lip creates an inward-pointing hook for reverse flow, by making a section of the U-cup non-solid.
        # This condition `abs_y > ry_local - ... - lip_angle*0.1` means fluid is allowed *closer* to the centerline
        # at the trailing edge of the U-cup, forming a re-entrant lip.
        in_cup = (ux_rel > 0) & (ux_rel <= (0.12 + 0.58 * cavity_len)) & (abs_y <= ry_local) & \
                 ~((ux_rel > (0.12 + 0.58 * cavity_len) - 0.1) & (abs_y > ry_local - (0.042 + 0.075 * wall_thick) - lip_angle*0.1))
        
        # Central Splitter Fin with Streamlined Prow
        fin_y_max = (0.8 * fin_width_s) * (ry_local - (0.042 + 0.075 * wall_thick))
        fin_x_extent = (0.8 * fin_len_s) * (0.12 + 0.58 * cavity_len)
        
        # Normalized x-coordinate for shaping the prow (0 at upstream end, 1 at throat)
        # ux_rel is negative upstream from the throat (0).
        # We want the fin to be pointed at its upstream end (most negative ux_rel).
        prow_x_norm = jnp.clip((ux_rel + fin_x_extent) / fin_x_extent, 0, 1) 
        
        # prow_power_val ranges from 1.0 (triangle) to 3.0 (sharper parabola).
        # fin_prow_taper = 0 -> prow_power_val = 1 (linear taper to point)
        # fin_prow_taper = 1 -> prow_power_val = 3 (cubic taper to point)
        prow_power_val = 1.0 + 2.0 * fin_prow_taper 
        
        # The y-limit for the fin, creating a shape that tapers to a point upstream for forward flow.
        # When prow_x_norm = 0 (upstream end), fin_mask_y_limit = 0 (point).
        # When prow_x_norm = 1 (at throat), fin_mask_y_limit = fin_y_max (full width).
        fin_mask_y_limit = fin_y_max * (prow_x_norm ** prow_power_val)
        
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_x_extent) & (abs_y <= fin_mask_y_limit)
        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))
    return mask

render_batch = jax.jit(jax.vmap(lambda p: render_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(506) # Changed seed from 505 to 506 for a fresh exploration path
    
    # Current best parameters from run_008 (17 params)
    best_p_008 = np.array([0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    # Add new parameter: fin_prow_taper, initialized to 0.5 (for a parabolic prow shape, prow_power_val = 2.0)
    best_p = np.append(best_p_008, 0.5) # Now 18 parameters
    
    # Initialize population: 32 candidates, with the best_p as the first candidate
    pop = np.clip(best_p + rng.normal(0, 0.1, (32, 18)), 0, 1).astype(np.float32)
    pop[0] = best_p # Ensure the current best is always part of the population to avoid regression
    
    elite_pool = np.copy(pop[:8]) # Initialize elite pool with first 8 random candidates (or initial best)
    elite_scores = np.full(8, -1e9) # Initialize elite scores to a very low value to ensure they are replaced by first batch
    
    num_rounds = 15 # Reduced from 20 (in run_008) to save time, aiming for ~200s design time
    
    for r in range(num_rounds):
        # Stop early if approaching the 300s budget, allowing time for final rendering
        if time.time() - t0 > 240: 
            print(f"Stopping evolution at round {r+1}/{num_rounds} due to time limit.")
            break
        
        m = render_batch(jnp.asarray(pop))
        sc, _, _ = evaluate_batch_fast(m, DEFAULT_CONFIG) # Evaluate population on high-res config
        
        # Update elites: combine current elites with new population, sort by score, and select top 8
        combined_candidates = np.concatenate([elite_pool, pop])
        combined_scores = np.concatenate([elite_scores, np.asarray(sc)])
        idx = np.argsort(combined_scores)[-8:] # Get indices of top 8 scores
        elite_pool, elite_scores = combined_candidates[idx], combined_scores[idx]
        
        # Mutate for next generation: sample from current elites and add noise
        # Decrease mutation strength over rounds to encourage convergence
        mutation_strength = 0.05 * (0.90**r) # Slightly faster decay than 0.95**r to converge quicker
        pop = np.clip(elite_pool[rng.integers(0, 8, 32)] + rng.normal(0, mutation_strength, (32, 18)), 0, 1)
        
    # Return the best design found in the elite pool (highest score)
    return render_diode(elite_pool[-1], DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)
