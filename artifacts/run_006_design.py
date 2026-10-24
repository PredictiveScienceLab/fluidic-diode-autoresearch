"""
Fluidic Diode Design & Continuous Optimization Module.
Run 006: Enhanced Asymmetric Borda-Sawtooth with Teardrop Splitter and Recessed Lips.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 005 winner (stdout), re-mapped for 12 parameters.
# The selector (params[0]) is manually set to 0.5 to ensure it starts within
# the 'Borda Splitter Sawtooth' range [0.33, 0.66), as the description and image
# for run_005 clearly correspond to this geometry, despite the original selector value
# (0.929...) in the stdout that would have triggered the Tesla branch due to a bug in
# the `render_unified` selector logic (now fixed).
# New parameters (splitter asymmetry, borda recess) are initialized neutrally (0.5 or 0.0).
# Tesla parameters (params[8] and params[9] from run_005 output) are shifted to the end.
RUN_005_STDOUT_PARAMS = np.array([
    0.92901513, 0.61911958, 0.23334518, 0.60662361, 0.34372601,
    0.97497225, 0.45186559, 0.67096075, 0.05320767, 0.40298386
], dtype=np.float32)

PREV_BEST_PARAMS = np.array([
    0.5,                                    # selector (forced to Borda_Splitter range [0.33, 0.66))
    RUN_005_STDOUT_PARAMS[1],               # p_throat_ratio
    RUN_005_STDOUT_PARAMS[2],               # p_ramp_len_ratio
    RUN_005_STDOUT_PARAMS[3],               # p_ramp_pow_ratio
    RUN_005_STDOUT_PARAMS[4],               # p_lip_size_ratio
    RUN_005_STDOUT_PARAMS[5],               # p_lip_ang_ratio
    RUN_005_STDOUT_PARAMS[6],               # p_splitter_width_ratio
    RUN_005_STDOUT_PARAMS[7],               # p_splitter_offset_x_ratio
    0.5,                                    # p_splitter_asymmetry_ratio (new, 0.5 for neutral/symmetric init)
    0.0,                                    # p_borda_recess_ratio (new, 0.0 for neutral init)
    RUN_005_STDOUT_PARAMS[8],               # p_tesla_bx_ratio (shifted from old params[8])
    RUN_005_STDOUT_PARAMS[9]                # p_tesla_by_ratio (shifted from old params[9])
], dtype=np.float32)


@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_sawtooth_v1_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """The original sawtooth baseline renderer (Run 000/002), mapped to universal params."""
    # Uses params[1:4] for its core parameters
    throat_gap = 0.28 + 0.36 * params[1] # p_throat_ratio
    ramp_len = 0.35 + 0.55 * params[2]  # p_ramp_len_ratio
    ramp_pow = 0.60 + 1.20 * params[3]  # p_ramp_pow_ratio

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
def render_borda_splitter_sawtooth_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Enhanced sawtooth with re-entrant lips and a central splitter island, mapped to universal params."""
    # Uses params[1:10] for its core parameters
    throat = 0.20 + 0.35 * params[1] # p_throat_ratio
    ramp_len = 0.30 + 0.50 * params[2] # p_ramp_len_ratio
    ramp_pow = 0.60 + 1.40 * params[3] # p_ramp_pow_ratio
    lip_size = 0.04 + 0.12 * params[4] # p_lip_size_ratio
    lip_ang = 0.40 + 1.80 * params[5] # p_lip_ang_ratio

    splitter_width_factor = params[6] # p_splitter_width_ratio
    splitter_offset_x_factor = params[7] # p_splitter_offset_x_ratio
    splitter_asymmetry_ratio = params[8] # p_splitter_asymmetry_ratio (new)
    borda_recess_ratio = params[9] # p_borda_recess_ratio (new)

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    pitch = 0.90 / n_stages
    mx = 0.05

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch
        
        # Main nozzle ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # Re-entrant lip (catcher) with optional recess
        lx = dx - ramp_len
        
        recess_depth = 0.03 * borda_recess_ratio # Max recess depth scaled by parameter
        # Parabolic recess profile, max at lx=lip_size/2, zero at lx=0 and lx=lip_size
        recess_profile = jnp.where(
            (lx > 0.0) & (lx < lip_size),
            recess_depth * (4 * lx / lip_size) * (1 - lx / lip_size), # Standard parabola 0 to 1 back to 0
            0.0
        )
        lip = (lx > 0.0) & (lx < lip_size) & (ay > throat - lip_ang * lx - recess_profile) & (ay < throat + 0.08)
        
        # Central splitter island
        splitter_x_start_rel = ramp_len + splitter_offset_x_factor * (1.0 - ramp_len - lip_size)
        splitter_len_rel = 0.1 + 0.2 * splitter_width_factor # Make length scale with width
                    
        # Calculate island coordinates relative to normalized design domain
        island_x_start_norm = splitter_x_start_rel
        island_x_end_norm = splitter_x_start_rel + splitter_len_rel
        
        # Normalized x within island for profile (0 to 1)
        island_dx_profile = (dx - island_x_start_norm) / splitter_len_rel
        
        # Asymmetric profile for half-width (teardrop-like)
        max_half_width = throat * 0.4 * splitter_width_factor
        
        # Power law for asymmetry: (x^alpha) * ((1-x)^beta)
        # splitter_asymmetry_ratio (0..1): 0.0 -> blunter left (forward entry), 0.5 -> symmetric, 1.0 -> blunter right (forward entry)
        alpha_exp = 1.0 + 2.0 * (1.0 - splitter_asymmetry_ratio)
        beta_exp = 1.0 + 2.0 * splitter_asymmetry_ratio
        
        asymmetric_profile = jnp.where(
            (island_dx_profile >= 0.0) & (island_dx_profile <= 1.0),
            (island_dx_profile ** alpha_exp) * ((1.0 - island_dx_profile) ** beta_exp),
            0.0
        )
        
        # Normalize the asymmetric_profile to have a maximum of 1.0
        exponent_sum = alpha_exp + beta_exp
        x_peak_val = jnp.where(exponent_sum > 1e-6, alpha_exp / exponent_sum, 0.5) # Avoid div by zero, default to 0.5
        max_profile_val = (x_peak_val ** alpha_exp) * ((1.0 - x_peak_val) ** beta_exp)
        
        normalized_asymmetric_profile = jnp.where(max_profile_val > 1e-6, asymmetric_profile / max_profile_val, asymmetric_profile)
        
        island_half_width = max_half_width * normalized_asymmetric_profile
                    
        # Compute potential island_mask
        potential_island_mask = (dx >= island_x_start_norm) & (dx <= island_x_end_norm) & \
                                (ay < island_half_width)

        # Only activate island if splitter_width_factor is significantly positive
        island_mask = jnp.where(splitter_width_factor > 0.01, potential_island_mask.astype(jnp.float32), 0.0)

        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_mask) 

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_tesla_v3_mapped(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Improved Tesla-style valve with cleaner geometry, mapped to universal params."""
    # Uses params[1], params[4], params[5], params[10], params[11]
    gap_factor = params[1] # p_throat_ratio (repurposed)
    isz_factor = params[4] # p_lip_size_ratio (repurposed for island size)
    wt_factor = params[5] # p_lip_ang_ratio (repurposed for wall thickness)
    bx_factor = params[10] # p_tesla_bx_ratio
    by_factor = params[11] # p_tesla_by_ratio

    gap = 0.18 + 0.22 * gap_factor
    bx = 0.15 + 0.35 * bx_factor
    by = 0.60 + 0.25 * by_factor
    isz = 0.04 + 0.12 * isz_factor
    wt = 0.05 + 0.06 * wt_factor

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)
    
    n_stages = 2
    pitch = 1.0 / n_stages
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for i in range(n_stages):
        xl = (x - i * pitch) / pitch
        
        # 1. Side-channel walls
        side_v = (jnp.abs(xl - bx) < wt/2) & (ay > gap) & (ay < by)
        side_h = (xl > bx) & (xl < bx + 0.35) & (jnp.abs(ay - by) < wt/2)
        
        # 2. Return fin (The "bucket")
        m = (by - gap) / 0.35
        fin_mask = (xl > bx) & (xl < bx + 0.35) & (ay < by - m*(xl-bx) + wt) & (ay > by - m*(xl-bx))

        # 3. Central island
        island = (xl > bx - isz) & (xl < bx + isz) & (ay < gap)
        
        mask = jnp.maximum(mask, side_v.astype(jnp.float32))
        mask = jnp.maximum(mask, side_h.astype(jnp.float32))
        mask = jnp.maximum(mask, fin_mask.astype(jnp.float32))
        mask = jnp.maximum(mask, island.astype(jnp.float32))
    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """Unified space: [selector, p_throat_ratio, ..., p_tesla_by_ratio] (12 parameters)"""
    selector = params[0] # Extract selector once
    
    mask = jax.lax.cond(
        selector < 0.33, # Sawtooth v1
        lambda p_full: render_sawtooth_v1_mapped(p_full, nx_design, ny),
        lambda p_full: jax.lax.cond(
            selector < 0.66, # Borda Splitter Sawtooth
            lambda pp_full: render_borda_splitter_sawtooth_mapped(pp_full, nx_design, ny),
            lambda pp_full: render_tesla_v3_mapped(pp_full, nx_design, ny), # Tesla v3
            p_full # Pass the full params array to inner cond
        ),
        params # Pass the full params array to outer cond
    )
    return mask

render_batch_coarse = jax.jit(
    jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny))
)
render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)

def generate_design() -> np.ndarray:
    t0 = time.time()
    np.set_printoptions(precision=8, suppress=True)
    rng = np.random.default_rng(42)

    # 1. Global Exploration on FAST_CONFIG
    num_params = len(PREV_BEST_PARAMS)
    pop = rng.uniform(0.0, 1.0, size=(128, num_params)).astype(np.float32)
    pop[0] = PREV_BEST_PARAMS # Strictly maintain previous best as the first candidate

    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Elite Selection and Verification on DEFAULT_CONFIG
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:12] # Select top 12 candidates
    pop_f1 = pop[top_idx]
    masks_f1 = render_batch_fine(jnp.asarray(pop_f1))
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    scores_f1_np = np.asarray(scores_f1)

    # 3. Local Refinement on DEFAULT_CONFIG
    best_idx_f1 = int(np.argmax(scores_f1_np))
    best_p = pop_f1[best_idx_f1]
    
    # Small perturbations around the winner
    # Create 11 neighbors + 1 best_p = 12 total for the next batch
    neighbors = best_p + rng.normal(0, 0.03, size=(11, num_params)) 
    pop_f2 = np.vstack([best_p[None, :], np.clip(neighbors, 0.0, 1.0)])
    
    masks_f2 = render_batch_fine(jnp.asarray(pop_f2))
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    scores_f2_np = np.asarray(scores_f2)

    best_idx_final = int(np.argmax(scores_f2_np))
    best_fine_params = pop_f2[best_idx_final]
    
    print(f"[design.py] Finished in {time.time()-t0:.1f}s | Best Fine Di = {scores_f2_np[best_idx_final]:.6f}")
    print("BEST_FINE_PARAMS:", best_fine_params.tolist())
    
    return np.asarray(masks_f2[best_idx_final], dtype=jnp.float32)

if __name__ == "__main__":
    mask = generate_design()
    print("Mask shape:", mask.shape)
