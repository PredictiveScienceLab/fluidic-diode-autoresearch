"""
Fluidic Diode Design & Continuous Optimization Module.
Run 013: Refined Conical Splitter & Flexible Island Profile.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast

# Exact parameters from the Run 012 winner (Di = 1.813995)
PREV_BEST_PARAMS_12 = np.array([
    0.9399105906486511, 0.8422966003417969, 0.015307745896279812, 0.571638822555542, 
    0.7967522144317627, 0.32198548316955566, 0.16118746995925903, 0.969295859336853, 
    0.03991043195128441, 0.10686103999614716, 0.5164387822151184, 0.09080671519041061
], dtype=np.float32)

# New parameter set for run_013 (15 parameters)
# Map old 12 parameters and add 3 new ones.
# Old parameters (run_012 context):
# [0] throat      (0.12 + 0.30 * p[0])
# [1] ramp_len    (0.15 + 0.40 * p[1])
# [2] ramp_pow    (0.40 + 1.60 * p[2])
# [3] lip_size    (0.04 + 0.16 * p[3])
# [4] lip_hook    (0.05 + 1.45 * p[4])
# [5] isl_scale_x (0.15 + 0.30 * p[5]) -> island total x-length factor
# [6] isl_pos_x   (0.02 + 0.20 * p[6]) -> island start x-position offset
# [7] isl_asym    (params[7]) -> was unused in run_012, repurposed as isl_nose_curve_factor
# [8] concavity   (0.30 + 0.65 * p[8]) -> concavity depth factor
# [9] pitch       (0.28 + 0.12 * p[9]) -> stage pitch spacing factor
# [10] isl_flare  (0.40 + 0.80 * p[10]) -> intensity of rear flare, repurposed as isl_flare_max_width_ratio
# [11] (last element of PREV_BEST_PARAMS_12 is 0.09080671519041061, which was the value for isl_flare)

# New parameter mapping (15 parameters):
# p[0-6] map directly to old p[0-6]
# p[7]   = isl_nose_curve_factor (repurposes old p[7] 'isl_asym')
# p[8]   = concavity_depth_factor (repurposes old p[8] 'concavity')
# p[9]   = pitch_spacing_factor (repurposes old p[9] 'pitch')
# p[10]  = isl_flare_max_width_ratio (repurposes old p[10] 'isl_flare')
# p[11]  = sting_base_width_frac (new)
# p[12]  = sting_taper_factor (new, 0.0 for pointy, 1.0 for uniform width)
# p[13]  = sting_len_frac (new, length of sting within cavity)
# p[14]  = isl_flare_exp (new, exponent for the trumpet flare)

PREV_BEST_PARAMS = np.zeros(15, dtype=np.float32)
PREV_BEST_PARAMS[0:7] = PREV_BEST_PARAMS_12[0:7]
PREV_BEST_PARAMS[7] = PREV_BEST_PARAMS_12[7]  # isl_nose_curve_factor, was isl_asym
PREV_BEST_PARAMS[8] = PREV_BEST_PARAMS_12[8]  # concavity_depth_factor, was concavity
PREV_BEST_PARAMS[9] = PREV_BEST_PARAMS_12[9]  # pitch_spacing_factor, was pitch
PREV_BEST_PARAMS[10] = PREV_BEST_PARAMS_12[10] # isl_flare_max_width_ratio, was isl_flare

# Initialize new parameters to a neutral/mid-range value
PREV_BEST_PARAMS[11] = 0.5 # sting_base_width_frac
PREV_BEST_PARAMS[12] = 0.5 # sting_taper_factor (0.5 means tip is half of base width)
PREV_BEST_PARAMS[13] = 0.5 # sting_len_frac (half the cavity length)
PREV_BEST_PARAMS[14] = 0.5 # isl_flare_exp


@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_refined_catcher_with_conical_sting(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    """
    Refined Catcher Topology:
    - Optimized nozzle walls and re-entrant Borda wall-lips.
    - Central island with a more flexible, parameterized nose shape and a "trumpet" flare.
    - Deep concave rear face with a prominent, conical central splitter 'sting'.
    """
    # Mapping parameters (15 dims)
    # Wall geometry
    throat_frac           = params[0] 
    ramp_len_frac         = params[1] 
    ramp_pow_val          = params[2] 
    lip_len_frac          = params[3] 
    lip_hook_factor       = params[4] 

    # Island geometry
    isl_x_scale_frac      = params[5] # total x-length of island relative to pitch
    isl_x_offset_frac     = params[6] # x-pos of island nose relative to ramp end
    isl_nose_curve_factor = params[7] # exponent for island nose streamlining
    concavity_depth_frac  = params[8] 
    pitch_spacing_factor  = params[9] 
    isl_flare_max_width_ratio = params[10] # max width added by flare to island profile
    
    # Sting geometry
    sting_base_width_frac = params[11] # base width of sting relative to throat
    sting_taper_factor    = params[12] # tip width / base width ratio for sting
    sting_len_frac        = params[13] # length of sting relative to cavity length
    isl_flare_exp         = params[14] # exponent for the trumpet flare shape

    # Scaled parameters
    throat      = 0.12 + 0.30 * throat_frac
    ramp_len    = 0.15 + 0.40 * ramp_len_frac
    ramp_pow    = 0.40 + 1.60 * ramp_pow_val
    lip_size    = 0.04 + 0.16 * lip_len_frac
    lip_hook    = 0.05 + 1.45 * lip_hook_factor

    isl_scale_x = 0.15 + 0.30 * isl_x_scale_frac
    isl_pos_x   = 0.02 + 0.20 * isl_x_offset_frac
    isl_nose_curve = 0.2 + 1.8 * isl_nose_curve_factor # A broader range for nose curvature
    concavity   = 0.30 + 0.65 * concavity_depth_frac
    pitch       = 0.28 + 0.12 * pitch_spacing_factor
    isl_flare_max_width_add = 0.40 + 0.80 * isl_flare_max_width_ratio # Max width added to the profile by flare
    flare_exponent = 1.5 + 3.5 * isl_flare_exp # Exponent for the flare shape

    # Sting dimensions relative to throat
    sting_base_width = (0.02 + 0.08 * sting_base_width_frac) * throat
    sting_tip_width_ratio = jnp.clip(sting_taper_factor, 0.0, 1.0) # Ensure ratio is between 0 and 1
    sting_tip_width  = sting_base_width * sting_tip_width_ratio
    
    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    ay = jnp.abs(y)

    n_stages = 3
    mx = 0.04 # Margin from left edge
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    
    for k in range(n_stages):
        x0 = mx + k * pitch
        dx = (x - x0) / pitch # Normalized x-position within a stage
        
        # 1. Main Nozzle Ramp
        s = jnp.clip(dx / ramp_len, 0.0, 1.0)
        yw = 1.0 - (1.0 - throat) * (s ** ramp_pow)
        tooth = (dx >= 0.0) & (dx <= ramp_len) & (ay >= yw)
        
        # 2. Hooked Borda Lip
        lx = dx - ramp_len # x-position relative to ramp end
        hook_curve_factor = 3.5 * lip_hook # Original constant
        y_lip = throat - lip_hook * lx + hook_curve_factor * (jnp.maximum(0.0, lx)**2)
        lip = (lx > 0.0) & (lx < lip_size) & (ay > y_lip) & (ay < y_lip + 0.11) # Maintain original lip thickness logic
        
        # 3. Refined Trumpet-Flared Island
        # Island starts after the ramp_len, offset by isl_pos_x, and has length isl_scale_x
        ix_start_norm = ramp_len + isl_pos_x
        idx = (dx - ix_start_norm) / isl_scale_x # Normalized x-position within the island (0 to 1)
        
        # Define island profile: streamlined nose transitions to aggressive flare
        isl_nose_region_end = 0.4 # x-fraction of island where nose ends and flare potentially starts
        
        # Nose profile (0 to 1)
        nose_profile = jnp.power(jnp.clip(idx / isl_nose_region_end, 0.0, 1.0), isl_nose_curve)
        
        # Flare profile (0 to 1)
        flare_rel_idx = jnp.clip((idx - isl_nose_region_end) / (1.0 - isl_nose_region_end + 1e-6), 0.0, 1.0)
        flare_profile = jnp.power(flare_rel_idx, flare_exponent)
        
        # Combine nose and flare
        island_width_profile_base = jnp.where(idx < isl_nose_region_end, nose_profile, 1.0 + isl_flare_max_width_add * flare_profile)
        
        # Taper the very ends to ensure a closed shape
        envelope_taper = jnp.clip(idx * 10.0, 0.0, 1.0) * jnp.clip((1.0 - idx) * 10.0, 0.0, 1.0)
        
        # The overall half-width of the island
        # Keep 0.55 as a base multiplier for island width, allowing flare_max_width_add to scale it further
        half_w = (throat * 0.55) * island_width_profile_base * envelope_taper 
        
        island_outer = (idx >= 0.0) & (idx <= 1.0) & (ay < half_w)
        
        # Deep Concavity + Conical Splitter Sting
        cav_start_idx_in_isl = 1.0 - 0.45 * concavity # Start of the concave region within island's normalized x
        
        # Cavity mask: remove material from the island in the concave region
        cavity_mask = (idx > cav_start_idx_in_isl) & (ay < half_w * (1.0 - concavity)) 
        
        # Conical Sting within the cavity
        cavity_x_length = (1.0 - cav_start_idx_in_isl)
        sting_length_in_isl = cavity_x_length * sting_len_frac
        
        # Center the sting within the available cavity length
        sting_x_start_norm = cav_start_idx_in_isl + (cavity_x_length - sting_length_in_isl) * 0.5
        sting_x_end_norm = sting_x_start_norm + sting_length_in_isl
        
        sting_active = (idx >= sting_x_start_norm) & (idx <= sting_x_end_norm)
        
        # Normalized x-position within the sting's active length (0 to 1)
        sting_rel_x = jnp.clip((idx - sting_x_start_norm) / (sting_length_in_isl + 1e-6), 0.0, 1.0)
        
        # Conical width profile for the sting
        current_sting_width = sting_base_width * (1.0 - sting_rel_x) + sting_tip_width * sting_rel_x
        
        sting_solid = sting_active & (ay < current_sting_width * 0.5)
        
        island_solid = (island_outer.astype(jnp.float32) * (1.0 - cavity_mask.astype(jnp.float32)))
        island_solid = jnp.maximum(island_solid, sting_solid.astype(jnp.float32))

        mask = jnp.maximum(mask, tooth.astype(jnp.float32))
        mask = jnp.maximum(mask, lip.astype(jnp.float32))
        mask = jnp.maximum(mask, island_solid)

    return mask

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_unified(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    return render_refined_catcher_with_conical_sting(params, nx_design, ny)

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_unified(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_unified(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t_start = time.time()
    rng = np.random.default_rng(int(time.time() * 1000) % 10**9)
    
    num_params = 15 # Updated number of parameters
    
    # 1. Global Screening (FAST_CONFIG) - ~10-15s for 64 candidates
    n_coarse = 64
    pop = rng.uniform(0.0, 1.0, (n_coarse, num_params)).astype(np.float32)
    
    # Ensure the previous best design is always a candidate
    pop[0] = PREV_BEST_PARAMS 
    
    # Introduce tighter perturbations around the best candidate for local search
    pop[1:16] = jnp.clip(PREV_BEST_PARAMS + rng.normal(0, 0.05, (15, num_params)), 0, 1) # Smaller std dev
    
    masks_c = render_batch_coarse(jnp.asarray(pop))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    
    # 2. Refinement Round 1 (DEFAULT_CONFIG) - ~15-20s for 16 candidates
    # Take top 8 from coarse, plus 8 perturbations of the absolute best coarse
    top_idx = np.argsort(np.asarray(scores_c))[::-1][:8]
    refine_pop_1 = pop[top_idx]
    
    best_coarse_p = pop[top_idx[0]]
    refine_pop_2 = jnp.clip(best_coarse_p + rng.normal(0, 0.02, (8, num_params)), 0, 1) # Even smaller std dev
    
    refine_pop = jnp.concatenate([refine_pop_1, refine_pop_2], axis=0)
    
    masks_f1 = render_batch_fine(refine_pop)
    scores_f1, _, _ = evaluate_batch_fast(masks_f1, DEFAULT_CONFIG)
    
    # 3. Refinement Round 2 (Final Hill-Climb) - ~15-20s for 16 candidates
    best_f1_idx = np.argmax(np.asarray(scores_f1))
    best_f1_p = refine_pop[best_f1_idx]
    
    final_pop = jnp.tile(best_f1_p, (16, 1))
    final_pop = jnp.clip(final_pop + rng.normal(0, 0.008, (16, num_params)), 0, 1) # Very small std dev for fine-tuning
    final_pop = final_pop.at[0].set(best_f1_p) # Keep parent
    
    masks_f2 = render_batch_fine(final_pop)
    scores_f2, _, _ = evaluate_batch_fast(masks_f2, DEFAULT_CONFIG)
    
    final_best_idx = np.argmax(np.asarray(scores_f2))
    best_final_p = final_pop[final_best_idx]
    
    print(f"[design.py] Best Di: {scores_f2[final_best_idx]:.6f} | Time: {time.time()-t_start:.1f}s")
    print("BEST_FINE_PARAMS:", best_final_p.tolist())
    
    return np.asarray(masks_f2[final_best_idx])

if __name__ == "__main__":
    design = generate_design()
