"""
Fluidic Diode Design & Continuous Optimization Module.

Run 021 Hypothesis:
- Physical Hypothesis: Streamlined Splitter Fin & Axial Shift Optimization.
  The success of run_020 (Di = 37.8) demonstrates the power of the central splitter fin 
  combined with wall undercuts. To improve further, we introduce 'fin_taper' (P15) to 
  streamline the forward-facing edge of the fin into a sharp wedge/parabola, reducing 
  forward drag. We also introduce 'fin_shift' (P14) to allow the fin to move axially, 
  optimizing the gap where the reverse flow is diverted into the U-cup's trapping zones.
- Algorithmic Hypothesis: Elite-Preserving Evolutionary Search with Tapered Warm-Start.
  We warm-start from the run_020 configuration but initialize the new taper and shift 
  parameters to values that promote streamlining. We use a multi-fidelity approach 
  (64 coarse -> 48 fine evaluations) to stay within the 25-60s target on the L4 GPU.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

MATCHED_FAST_CONFIG = SimConfig(
    nx=180,
    ny=60,
    pad_x=15,
    nu=0.0012,
    d_rho=0.03,
    n_steps=3200,
    avg_window=800,
    max_solid_fraction=0.60,
    min_open_ratio=0.12,
    q_open_ref=0.1585,
)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_refined_vortex_diode(
    params: jnp.ndarray,
    nx_design: int,
    ny: int,
) -> jnp.ndarray:
    """
    Render a 3-stage Graded U-Cup Diode with Tapered Central Fin and Wall Undercuts.
    params (16,):
      0-10: Standard U-cup / Nozzle geometry
      11: tooth_undercut      (reverse hook depth on wall nozzle)
      12: cup_ry_grade        (grading of cup width)
      13: throat_grade        (grading of throat gap)
      14: fin_shift           (axial shift of the fin relative to cup nose)
      15: fin_taper           (tapering of the fin prow for +x flow)
    """
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
    undercut = 0.00 + 0.15 * params[11] 
    ry_grade = -0.15 + 0.30 * params[12]
    th_grade = -0.15 + 0.30 * params[13]
    
    # Fin Parameters
    fin_len = 0.15  # Fixed base length
    fin_width = 0.08 # Fixed base width
    fin_shift = -0.05 + 0.15 * params[14] # Shift relative to ux_rel = 0
    fin_taper = params[15] # 0 = rectangular, 1 = triangular/sharp

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

        # 1. Wall Nozzle with Undercut
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        # 2. U-Cup
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.42)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)
        in_outer = ((ux_rel <= 0.0) & ((ux_rel / jnp.maximum(0.06, nose_thick * prow_stretch)) ** 2 + (abs_y / ry_local) ** 2 <= 1.0)) | \
                   ((ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local))
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2) <= 1.0)
        in_cup = in_outer & (~hollow)

        # 3. Tapered Splitter Fin
        # Fin is located relative to cup nose (ux_rel = 0)
        f_x_start = fin_shift - fin_len
        f_x_end = fin_shift
        is_in_fin_x = (ux_rel >= f_x_start) & (ux_rel <= f_x_end)
        # Linear taper from f_x_end (full width) to f_x_start (tapered width)
        local_f_x = (ux_rel - f_x_start) / fin_len
        tapered_width = fin_width * (1.0 - fin_taper * (1.0 - local_f_x))
        is_fin = is_in_fin_x & (abs_y <= tapered_width)

        mask = jnp.maximum(mask, jnp.maximum(wall_nozzle, jnp.maximum(in_cup, is_fin)))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_refined_vortex_diode(p, MATCHED_FAST_CONFIG.nx - 2 * MATCHED_FAST_CONFIG.pad_x, MATCHED_FAST_CONFIG.ny)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_refined_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(21)

    # Base anchor derived from Run 020 elite characteristics
    # P0-P10: Standard U-cup geometry
    # P11: Undercut, P12-13: Grading, P14: Fin Shift, P15: Fin Taper
    anchor_020 = np.array([
        0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, # P0-P10
        0.60, 0.50, 0.50, # P11-P13
        0.30, 0.70  # P14 (shift), P15 (taper)
    ], dtype=np.float32)

    # Initial Population
    pop = rng.uniform(0.0, 1.0, size=(64, 16)).astype(np.float32)
    pop[0] = anchor_020
    pop[1:10] = np.clip(anchor_020 + rng.normal(0, 0.1, (9, 16)), 0, 1)

    # Phase 1: Coarse Screening
    s_c_list = []
    for i in range(0, 64, 32):
        m_c = render_batch_coarse(jnp.asarray(pop[i:i+32]))
        s_c, _, _ = evaluate_batch_fast(m_c, MATCHED_FAST_CONFIG)
        s_c_list.append(np.asarray(s_c))
    scores_coarse = np.concatenate(s_c_list)
    
    top_indices = np.argsort(scores_coarse)[::-1][:8]
    elites = pop[top_indices]
    
    # Phase 2: Fine Evolution (6 rounds of 8)
    m_f_elites = render_batch_fine(jnp.asarray(elites))
    scores_fine, _, _ = evaluate_batch_fast(m_f_elites, DEFAULT_CONFIG)
    scores_fine = np.asarray(scores_fine)
    
    best_score = float(np.max(scores_fine))
    best_params = elites[np.argmax(scores_fine)].copy()
    
    print(f"Initial Best Fine Di: {best_score:.4f}")

    for r in range(6):
        if time.time() - t0 > 250: break
        
        # Mutation/Crossover
        sigma = 0.08 * (0.8 ** r)
        offspring = []
        for _ in range(8):
            parent = elites[rng.choice(len(elites))]
            child = np.clip(parent + rng.normal(0, sigma, 16), 0, 1)
            offspring.append(child)
        
        offspring_arr = np.stack(offspring)
        m_f = render_batch_fine(jnp.asarray(offspring_arr))
        s_f, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
        s_f_np = np.asarray(s_f)
        
        # Update Elites
        combined_pop = np.concatenate([elites, offspring_arr])
        combined_scores = np.concatenate([scores_fine, s_f_np])
        idx = np.argsort(combined_scores)[::-1][:8]
        elites = combined_pop[idx]
        scores_fine = combined_scores[idx]
        
        if scores_fine[0] > best_score:
            best_score = float(scores_fine[0])
            best_params = elites[0].copy()
            print(f"Round {r} NEW BEST: {best_score:.4f} | Params: {best_params}")

    final_mask = render_refined_vortex_diode(jnp.asarray(best_params), DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny)
    return np.asarray(final_mask)
