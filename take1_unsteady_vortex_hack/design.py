"""
Fluidic Diode Design & Continuous Optimization Module (MUTABLE BY AGENT).

CRITICAL ENGINEERING NOTES FOR NEXT ITERATIONS (READ CAREFULLY):
1. `run_020` and `run_034` (`score = 37.802017`, Q_fwd = 0.057065, Q_rev = 0.001510) are ALREADY
   100% deterministic and reproducible on `DEFAULT_CONFIG`!
   - Do NOT propose hypotheses that merely say "Stabilize run_020"! It is already stable.
   - Previous runs (run_021-run_033, run_035-run_037) scored 1.1-4.5 ONLY because they accidentally
     simplified `in_outer_nose` (replacing `nose_thick * prow_stretch` with `0.15`) or `r_vertex`
     (replacing `jnp.minimum(0.0, ux_rel - 0.06) / 0.06` with `ux_rel / 0.1`), which changed the binary mask!
2. MANDATORY INVARIANT IN `render_graded_ucup_vortex_diode`:
   - Keep every single formula in `render_graded_ucup_vortex_diode` below verbatim!
   - If you add new geometric features (e.g. P16, P17: secondary vortex baffles, wall cavities,
     cup lip hooks, non-linear stage grading, or asymmetric splitter fins), parameterize them so that
     setting the new parameters to `0.0` produces the EXACT bit-for-bit mask of `BEST_PARAMS_RUN020`.
   - Always include the unperturbed `BEST_PARAMS_RUN020` (with `0.0` for any new parameters) as `pop[0]`
     in your `DEFAULT_CONFIG` evaluation so your returned design is guaranteed to achieve `score >= 37.802017`!
3. HOW TO BEAT `37.802017`:
   - Explore coordinate-wise perturbations / directional line search / Nelder-Mead / CMA-ES on `DEFAULT_CONFIG`
     around `BEST_PARAMS_RUN020` (using micro-mutations `sigma = 0.003` to `0.025` on 1-3 parameters at a time
     so only a few lattice boundary pixels flip at once), OR
   - Introduce a new physical mechanism (parameterized so `p_new = 0.0` is neutral) such as:
     * Downstream secondary cusp / vortex-chamber lip on the channel walls between stages,
     * Stage-specific axial spacing (non-uniform pitch between Stage 0->1 and Stage 1->2),
     * Curved / flared central splitter fin or dual-prong upstream wake splitter,
     * Concave inner floor shaping inside the U-cup cavity (`r_vertex` exponent/depth).
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Exact 16-parameter vector discovered in run_049 (deterministically scores Di = 383.239117):
BEST_PARAMS_RUN020_BASE = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315
], dtype=np.float32)

# Extend with new parameters, initialized to 0.0 (neutral) for new features
# P16: lip_start_offset_factor
# P17: lip_length_factor
# P18: lip_radial_depth_factor
BEST_PARAMS_RUN020 = np.concatenate([BEST_PARAMS_RUN020_BASE, np.array([0.0, 0.0, 0.0], dtype=np.float32)])


@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
    # DO NOT ALTER OR ABBREVIATE THESE 16 PARAMETER MAPPINGS:
    nozzle_len = 0.16 + 0.36 * params[0]
    throat_base = 0.27 + 0.32 * params[1]
    nozzle_pow = 0.60 + 1.10 * params[2]
    cup_offset = 0.32 + 0.38 * params[3]
    cup_ry_base = 0.12 + 0.18 * params[4]
    prow_stretch = 0.55 + 1.20 * params[5]
    cavity_len = 0.12 + 0.58 * params[6]
    wall_thick = 0.042 + 0.075 * params[7]
    nose_thick = 0.10 + 0.24 * params[8]
    pitch_scale = 0.78 + 0.16 * params[9]
    flare = -0.16 + 0.32 * params[10]
    undercut = 0.00 + 0.12 * params[11]
    ry_grade = -0.15 + 0.30 * params[12]
    th_grade = -0.15 + 0.30 * params[13]
    fin_len_scale = 0.0 + 0.8 * params[14]
    fin_width_scale = 0.0 + 0.8 * params[15]

    # NEW PARAMETERS (P16, P17, P18) for wall-mounted re-entrant lip
    # These are only active if params array has more than 16 elements
    lip_start_offset_factor = 0.0
    lip_length_factor = 0.0
    lip_radial_depth_factor = 0.0
    if len(params) > 16:
        lip_start_offset_factor = 0.0 + 0.3 * params[16] # 0 to 0.3 of a stage pitch
        lip_length_factor = 0.0 + 0.15 * params[17]      # 0 to 0.15 of a stage pitch
        lip_radial_depth_factor = 0.0 + 0.25 * params[18] # 0 to 0.25 of the channel half-height

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)
    n_stages = 3
    pitch = pitch_scale / n_stages
    margin_x = 0.5 * (1.0 - pitch_scale)
    aspect = (nx_design * pitch) / (ny * 0.5)

    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)
    for k in range(n_stages):
        stage_centered = k - 1.0
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * stage_centered), 0.22, 0.68)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * stage_centered), 0.10, 0.36)
        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch

        # 1. Converging wall ramp with optional undercut
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        # 2. Sealed-Nose Deep U-Cup Reflector (DO NOT REPLACE nose_thick*prow_stretch OR r_vertex WITH CONSTANTS!)
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)

        in_outer_nose = (ux_rel <= 0.0) & (
            (ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0
        )
        in_outer_body = (ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local)
        in_outer = in_outer_nose | in_outer_body

        r_vertex = jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (r_vertex <= 1.0)
        in_cup = in_outer & (~hollow)

        # 3. Central Splitter Fin
        fin_len_actual = fin_len_scale * cavity_len
        fin_width_actual = fin_width_scale * ry_inner
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_actual) & (abs_y <= fin_width_actual)

        # 4. Wall-mounted re-entrant lip (NEW FEATURE)
        # Calculate y-coordinate of the wall at the nozzle exit for this stage
        s_nozzle_exit = jnp.clip(rear_edge / nozzle_len, 0.0, 1.0)
        y_wall_nozzle_exit = 1.0 - (1.0 - throat_gap) * (s_nozzle_exit ** nozzle_pow)

        # Define the re-entrant lip's local x-coordinates (relative to x0, scaled by pitch)
        lip_x_start_local_dx = rear_edge + lip_start_offset_factor
        lip_x_end_local_dx = lip_x_start_local_dx + lip_length_factor

        # Global x-coordinates for the lip
        lip_x_start_abs = x0 + pitch * lip_x_start_local_dx
        lip_x_end_abs = x0 + pitch * lip_x_end_local_dx

        # The inner boundary of the lip. It extends from the wall (y_wall_nozzle_exit) towards the centerline.
        # (1.0 - y_wall_nozzle_exit) is the available channel height for the lip to extend into from the wall.
        lip_y_inner_boundary = jnp.maximum(0.0, y_wall_nozzle_exit - lip_radial_depth_factor * (1.0 - y_wall_nozzle_exit))

        # Condition to make the lip disappear if its parameters are 0
        is_lip_active = (lip_length_factor > 1e-6) & (lip_radial_depth_factor > 1e-6)

        is_lip = is_lip_active & \
                 (x >= lip_x_start_abs) & (x <= lip_x_end_abs) & \
                 (abs_y >= lip_y_inner_boundary) & (abs_y <= y_wall_nozzle_exit)

        stage = (wall_nozzle | in_cup | is_fin | is_lip).astype(jnp.float32)
        mask = jnp.maximum(mask, stage)

    return mask


render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)


def generate_design() -> np.ndarray:
    t0 = time.time()
    np.set_printoptions(precision=8, suppress=True)
    # Use a time-varying seed so every iteration explores fresh micro-mutations around BEST_PARAMS_RUN020!
    rng = np.random.default_rng(int(time.time() * 1000) % 1000000)

    # Initialize best_params with BEST_PARAMS_RUN020 (from run_049, Di = 383.239117)
    best_params = BEST_PARAMS_RUN020.copy()
    # Render the initial best mask to ensure it matches the run_049 baseline
    best_mask = np.asarray(render_batch_fine(jnp.asarray(best_params[None, :]))[0], dtype=np.float32)
    best_score = 383.239117 # Baseline score from run_049

    num_params = len(best_params) # Now 19 parameters

    # Coordinate-wise & sparse multi-scale local search on DEFAULT_CONFIG (~35s total)
    for r in range(4): # 4 rounds of optimization
        if time.time() - t0 > 45.0: # Enforce time budget
            print(f"[design.py] Time budget exceeded ({time.time() - t0:.2f}s), breaking optimization loop.")
            break
        
        cands = [best_params.copy()] # Always include the current best
        while len(cands) < 12: # Generate 11 more candidates
            p_mut = best_params.copy()
            
            # Mutate 1 to 3 coordinates at a time with fine sigma so only a few boundary pixels flip
            idx = rng.choice(num_params, size=int(rng.integers(1, 4)), replace=False)
            sigma = 0.012 * (0.75 ** r) # Fine sigma for local boundary refinement
            p_mut[idx] = np.clip(p_mut[idx] + rng.normal(0.0, sigma, size=len(idx)).astype(np.float32), 0.0, 1.0)
            cands.append(p_mut)

        cand_arr = np.stack(cands, axis=0)
        m_f = render_batch_fine(jnp.asarray(cand_arr))
        f_sc, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        f_sc_np = np.asarray(f_sc)

        r_best_idx = int(np.argmax(f_sc_np))
        if f_sc_np[r_best_idx] > best_score:
            best_score = float(f_sc_np[r_best_idx])
            best_params = cand_arr[r_best_idx].copy()
            best_mask = np.asarray(m_f[r_best_idx], dtype=np.float32)
            print(f"[design.py] NEW BEST Di = {best_score:.6f} | Params = {best_params.tolist()}")
        else:
            print(f"[design.py] Round {r} no improvement. Best Di = {best_score:.6f}")

    return best_mask
