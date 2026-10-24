"""
Fluidic Diode Design & Continuous Optimization Module.
Run 009: Enhanced Concave-Catcher Island with Tunable Scoop.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 007 winner (stdout)
# Di = 1.439455
# BEST_FINE_PARAMS: [0.6616491675376892, 0.7567344903945923, 0.4825185537338257, 0.17889510095119476, 0.5502895712852478, 0.011797428131103516, 0.8354988098144531, 0.6550413966178894, 0.7291443347930908, 0.3924931287765503, 0.9234058260917664, 0.8792598247528076]
#
# Re-mapping for run_009:
# P0: selector (was P0)
# P1: throat (was P1)
# P2: ramp_len (was P2)
# P3: ramp_pow (was P3)
# P4: lip_size (was P4)
# P5: lip_hook (was P5)
# P6: splitter_width_factor (was P6)
# P7: splitter_offset_x (was P7)
# P8: splitter_asymmetry (was P8)
# P9: concavity_depth_factor (was P9, concavity_factor)
# P10: pitch_factor (was P10, pitch_factor)
# P11: concavity_mouth_width_factor (was P11, unused for run_007, now used)
# P12: concavity_peak_x_offset (NEW parameter for run_009)
PREV_BEST_PARAMS = np.array([
    0.6616491675376892, # P0: selector
    0.7567344903945923, # P1: throat
    0.4825185537338257, # P2: ramp_len
    0.17889510095119476, # P3: ramp_pow
    0.5502895712852478, # P4: lip_size
    0.011797428131103516, # P5: lip_hook
    0.8354988098144531, # P6: splitter_width_factor
    0.6550413966178894, # P7: splitter_offset_x
    0.7291443347930908, # P8: splitter_asymmetry
    0.3924931287765503, # P9: concavity_depth_factor (was old concavity_factor)
    0.9234058260917664, # P10: pitch_factor (was old pitch_factor)
    0.8792598247528076, # P11: concavity_mouth_width_factor (was old unused P11)
    0.5 # P12: concavity_peak_x_offset (NEW, neutral default)
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_sawtooth_v1_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    throat_gap = 0.28 + 0.36 * params[1]
    ramp_len = 0.35 + 0.55 * params[2]
    ramp_pow = 0.60 + 1.20 * params[3]
    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    pitch = 0.84 / n_stages
    margin_x = 0.08
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** ramp_pow)
        tooth = ((dx >= 0.0) & (dx <= ramp_len) & (abs_y >= y_wall)).astype(jnp.float32)
        mask = jnp.maximum(mask, tooth)
    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_borda_catcher_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Advanced Borda-Sawtooth with:
    1. Hooked (re-entrant) wall lips.
    2. Concave-back central islands (C-catchers) with tunable scoop.
    3. Parameterized stage pitch.
    """
    # Wall & Lip Parameters (P1-P5)
    throat = 0.15 + 0.35 * params[1]
    ramp_len = 0.25 + 0.45 * params[2]
    ramp_pow = 0.50 + 1.50 * params[3]
    lip_size = 0.05 + 0.15 * params[4]
    lip_hook = 0.20 + 1.20 * params[5] # Degree of re-entrant hooking

    # Island Base Shape Parameters (P6-P8)
    splitter_width_factor = params[6] # Controls overall island width
    splitter_offset_x = params[7]     # Controls island x-position within stage
    splitter_asymmetry = params[8]    # Controls base teardrop asymmetry (fwd sharpness vs rev bluntness)

    # Enhanced Concavity (Scoop) Parameters (P9, P11, P12)
    concavity_depth_factor = params[9] # How deep the cavity is (0.0 to 1.0)
    concavity_mouth_width_factor = params[11] # How wide the opening of the cavity is (0.0 to 1.0)
    concavity_peak_x_offset = params[12] # Where the deepest point of the cavity is (0.0 = cavity front, 1.0 = cavity back)

    # Stage Pitch Parameter (P10)
    pitch_factor = 0.25 + 0.10 * params[10] # Variable pitch

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.05 # Margin from left edge
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch_factor
        dx = (x - x0) / pitch_factor # Normalized x within current stage
        
        # 1. Main Nozzle Ramp (Upper and Lower Walls)
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Hooked Borda Lip (Re-entrant features on the walls)
        lx = dx - ramp_len # Relative x past the ramp
        hook_curve = 2.5 * lip_hook 
        y_lip = throat - lip_hook * lx + hook_curve * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.12)
        
        # 3. Central Island with Enhanced Concavity (Scoop)
        isl_x_start = ramp_len + splitter_offset_x * (1.0 - ramp_len - lip_size)
        isl_len = 0.12 + 0.15 * splitter_width_factor
        idx = (dx - isl_x_start) / isl_len # Normalized position along island (0 to 1)

        # Base teardrop profile for the island
        alpha = 1.0 + 3.0 * (1.0 - splitter_asymmetry) # Controls leading edge sharpness
        beta = 1.0 + 3.0 * splitter_asymmetry         # Controls trailing edge bluntness
        prof = (jnp.maximum(0.0, idx)**alpha) * (jnp.maximum(0.0, 1.0 - idx)**beta)
        # Normalize profile to prevent scaling issues
        peak_val = (alpha/(alpha+beta))**alpha * (beta/(alpha+beta))**beta if (alpha+beta) > 0 else 1.0
        prof = prof / (peak_val + 1e-6)
        
        half_w_base = throat * 0.45 * splitter_width_factor * prof
        
        # Enhanced Concavity (Scoop) at the rear of the island
        cavity_rel_len = 0.3 + 0.4 * concavity_mouth_width_factor # Relative length of the scoop region
        cavity_start_idx = 1.0 - cavity_rel_len # Normalized start point of the scoop on the island
        
        cavity_idx_local = (idx - cavity_start_idx) / (cavity_rel_len + 1e-6) # Normalized x within the scoop (0 to 1)
        
        # Create an asymmetric scoop profile: deepest at concavity_peak_x_offset
        scoop_profile_base = jnp.maximum(0.0, cavity_idx_local) * jnp.maximum(0.0, 1.0 - cavity_idx_local)
        
        # Shift the peak of the scoop profile
        shifted_scoop_profile = jnp.where(
            cavity_idx_local < concavity_peak_x_offset,
            scoop_profile_base * (cavity_idx_local / (concavity_peak_x_offset + 1e-6)),
            scoop_profile_base * ((1.0 - cavity_idx_local) / (1.0 - concavity_peak_x_offset + 1e-6))
        )
        shifted_scoop_profile = jnp.where(cavity_idx_local < 0.0, 0.0, shifted_scoop_profile)
        shifted_scoop_profile = jnp.where(cavity_idx_local > 1.0, 0.0, shifted_scoop_profile)

        cavity_depth_shape = half_w_base * concavity_depth_factor * shifted_scoop_profile
        
        half_w_final = jnp.maximum(0.0, half_w_base - cavity_depth_shape)

        island_solid = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w_final)
        
        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_solid.astype(jnp.float32))

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_tesla_v3_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    gap = 0.18 + 0.22 * params[1]
    isz = 0.04 + 0.12 * params[4]
    wt = 0.05 + 0.06 * params[5]
    bx = 0.15 + 0.35 * params[10] # This param now maps to pitch_factor in borda_catcher
    by = 0.60 + 0.25 * params[11] # This param now maps to concavity_mouth_width_factor in borda_catcher
    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)
    n_stages = 2
    pitch = 1.0 / n_stages
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for i in range(n_stages):
        xl = (x - i * pitch) / pitch
        side_v = (jnp.abs(xl - bx) < wt/2) & (ay > gap) & (ay < by)
        side_h = (xl > bx) & (xl < bx + 0.35) & (jnp.abs(ay - by) < wt/2)
        m = (by - gap) / 0.35
        fin_mask = (xl > bx) & (xl < bx + 0.35) & (ay < by - m*(xl-bx) + wt) & (ay > by - m*(xl-bx))
        island = (xl > bx - isz) & (xl < bx + isz) & (ay < gap)
        mask = jnp.maximum(mask, side_v.astype(jnp.float32))
        mask = jnp.maximum(mask, side_h.astype(jnp.float32))
        mask = jnp.maximum(mask, fin_mask.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    selector = params[0]
    return jax.lax.cond(
        selector < 0.3,
        lambda p: render_sawtooth_v1_mapped(p, nx_design, ny),
        lambda p: jax.lax.cond(
            selector < 0.85, # This will trigger for selector = 0.6616 from PREV_BEST_PARAMS
            lambda pp: render_borda_catcher_mapped(pp, nx_design, ny),
            lambda pp: render_tesla_v3_mapped(pp, nx_design, ny),
            p
        ),
        params
    )

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    
    num_params = 13 # Now 13 parameters
    n_coarse = 64 # Number of candidates for coarse screening
    
    # 1. Global Screening (FAST_CONFIG)
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS # Seed with previous winner (extended with default for new P12)
    
    # Perturb seed slightly to explore the immediate neighborhood in coarse mode
    # Ensure perturbations are within [0, 1]
    pop[1:8] = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (7, num_params)), 0, 1)
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Refinement Rounds (DEFAULT_CONFIG)
    # Round 1: Top 8 from coarse screening
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    elite_pop = pop[top_idx]
    
    masks_f1 = render_batch_fine(jnp.asarray(elite_pop))
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # Round 2: Local Search around the best of Round 1
    best_idx = np.argmax(np.asarray(scores_f1))
    best_p = elite_pop[best_idx]
    
    # 1+7 Evolution Strategy-like step: Parent + 7 slightly mutated children
    local_pop = np.tile(best_p, (8, 1))
    local_pop[1:] = np.clip(best_p + rng.normal(0, 0.025, (7, num_params)), 0, 1) # Smaller perturbation for fine-tuning
    
    masks_f2 = render_batch_fine(jnp.asarray(local_pop))
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = local_pop[final_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_idx])

if __name__ == "__main__":
    design = generate_design()
