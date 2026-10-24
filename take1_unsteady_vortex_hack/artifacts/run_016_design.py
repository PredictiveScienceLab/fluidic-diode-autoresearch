"""
Fluidic Diode Design & Continuous Optimization Module (run_016).

Run 016 Hypothesis:
- Physical: The "cross" island in Run 008 is extremely effective at blocking 
  reverse flow (Di > 18) but likely creates unnecessary drag in forward flow. 
  By introducing a 'prow_taper' parameter to the upstream (forward-facing) 
  part of the central island, we can streamline the forward path while 
  preserving the bluff-body characteristics that induce vortex locking in reverse.
- Algorithmic: Continue using direct high-resolution (DEFAULT_CONFIG) 
  evaluations in a 'ratcheted' evolutionary loop. Start exactly from the 
  Run 008 parameters to ensure no regression, then refine.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Params (18): 
    0-13: Base geometry (nozzle, cup, wall)
    14-15: Island (fin) dimensions
    16: Tip lip angle
    17: Prow taper (new)
    """
    nozzle_len, throat_base, nozzle_pow, cup_offset, cup_ry_base, prow_stretch, \
    cavity_len, wall_thick, nose_thick, pitch_scale, flare, undercut, \
    ry_grade, th_grade, fin_len_s, fin_width_s, lip_angle, prow_taper = params

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
        # ux_rel defines the coordinate relative to the island center/cup start
        ux_rel = (dx - (0.32 + 0.38 * cup_offset)) * aspect
        
        # 1. Wall Nozzle Logic
        y_wall = 1.0 - (1.0 - jnp.clip(0.27 + 0.32 * throat_base * (1 + (-0.15 + 0.3*th_grade)*(k-1)), 0.22, 0.68)) * \
                 (jnp.clip(dx / (0.16 + 0.36 * nozzle_len), 0, 1) ** (0.6 + 1.1 * nozzle_pow))
        
        # Wall is solid if it's outside the nozzle and within the stage length
        # Include the 'undercut' which carves into the wall to create a pocket
        wall_mask = (dx >= 0.0) & (dx <= (0.16 + 0.36 * nozzle_len) - (0.12 * undercut) * jnp.clip((abs_y - 0.2)/0.5, 0, 1)) & (abs_y >= y_wall)
        
        # 2. Island Logic (Split into Prow/Fin and Stern/Cup-Block)
        # ry_local is the base width of the island
        ry_local = jnp.clip((0.12 + 0.18 * cup_ry_base) * (1 + (-0.15 + 0.3*ry_grade)*(k-1)) + \
                            (-0.16 + 0.32 * flare) * jnp.clip(ux_rel, 0, 0.7), 0.08, 0.42)
        
        # Downstream part of the island (in_cup)
        # Nips the corners based on lip_angle to create a re-entrant shape
        stern_len = (0.12 + 0.58 * cavity_len)
        in_cup = (ux_rel > 0) & (ux_rel <= stern_len) & (abs_y <= ry_local) & \
                 ~((ux_rel > stern_len - 0.1) & (abs_y > ry_local - (0.042 + 0.075 * wall_thick) - lip_angle*0.1))
        
        # Upstream part of the island (is_fin)
        # prow_taper streamlines the leading edge (ux_rel = -fin_len)
        fin_len = (0.8 * fin_len_s) * stern_len
        fin_width = (0.8 * fin_width_s) * (ry_local - (0.042 + 0.075 * wall_thick))
        
        # Calculate tapering factor: 1.0 at ux_rel=0 (junction), (1-prow_taper) at ux_rel=-fin_len
        dist_from_junction = jnp.abs(ux_rel)
        taper_factor = 1.0 - prow_taper * jnp.clip(dist_from_junction / (fin_len + 1e-6), 0, 1)
        
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len) & (abs_y <= fin_width * taper_factor)
        
        mask = jnp.maximum(mask, (wall_mask | in_cup | is_fin).astype(jnp.float32))
        
    return mask

# Vectorized rendering for the population
render_batch = jax.jit(jax.vmap(lambda p: render_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(808)
    
    # Starting point: The optimized parameters from the record-breaking Run 008
    # Added 18th parameter: prow_taper (initialized to 0.0 to match Run 008 exactly)
    best_p = np.array([
        0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 
        0.50, 0.00, 0.50, 0.50, 0.50, 0.50, 0.50, 0.00
    ], dtype=np.float32)
    
    pop_size = 32
    num_rounds = 15
    
    # Initialize population around the best known point
    pop = np.clip(best_p + rng.normal(0, 0.05, (pop_size, 18)), 0, 1).astype(np.float32)
    pop[0] = best_p  # Ratchet: Always keep the current best
    
    elite_pool = pop[:8]
    elite_scores = np.zeros(8)
    
    print(f"Starting evolution from Run 008 baseline. Budget: 300s.")
    
    for r in range(num_rounds):
        elapsed = time.time() - t0
        if elapsed > 250:
            print(f"Time limit approaching ({elapsed:.1f}s). Stopping.")
            break
            
        # Evaluate current population using high-res simulator
        masks = render_batch(jnp.asarray(pop))
        scores, q_fwds, q_revs = evaluate_batch_fast(masks, DEFAULT_CONFIG)
        scores = np.asarray(scores)
        
        # Update elites
        combined_pop = np.concatenate([elite_pool, pop])
        combined_scores = np.concatenate([elite_scores, scores])
        
        idx = np.argsort(combined_scores)[-8:]
        elite_pool = combined_pop[idx]
        elite_scores = combined_scores[idx]
        
        best_round_score = np.max(scores)
        print(f"Round {r+1}/{num_rounds} | Best: {elite_scores[-1]:.4f} | Time: {elapsed:.1f}s")
        
        # Generate next population: mutate from elites
        # Use a decaying mutation rate to refine the search
        mutation_strength = 0.08 * (0.9 ** r)
        parent_indices = rng.integers(0, 8, pop_size)
        pop = np.clip(elite_pool[parent_indices] + rng.normal(0, mutation_strength, (pop_size, 18)), 0, 1)
        pop[0] = elite_pool[-1] # Ensure the absolute best is always carried forward
        
    # Final winner
    final_params = elite_pool[-1]
    return np.array(render_diode(final_params, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))

if __name__ == "__main__":
    # Test render
    design = generate_design()
    print(f"Design shape: {design.shape}, Solid fraction: {np.mean(design):.4f}")
