"""
Fluidic Diode Design & Continuous Optimization Module.

Run 018 Hypothesis:
- Physical Hypothesis: Dual-Purpose Central Island Refinement.
  1. Internal vs. External Splitting: Introduce a `fin_shift` parameter that allows the 
     central fin to move from the upstream prow (where it streamlines forward flow) 
     to the internal cavity (where it splits the reverse-flow jet and promotes 
     vortex formation).
  2. Prow Curvature: Introduce a `prow_power` parameter to evolve the leading edge 
     from a blunt elliptical shape to a sharp, low-drag power-law profile.
- Algorithmic Hypothesis: Multi-Fidelity Evolutionary Ratchet.
  Use 128 coarse evaluations (MATCHED_FAST_CONFIG) to identify promising regions 
  of the 18-parameter space, followed by 5 rounds of high-resolution (DEFAULT_CONFIG) 
  evolution to fine-tune the geometry. Warm-start the population with the 
  Run 007 elite structure.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, SimConfig, evaluate_batch_fast

# Coarse surrogate for rapid screening
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
def render_refined_diode(
    params: jnp.ndarray,
    nx_design: int,
    ny: int,
) -> jnp.ndarray:
    """
    Render a 3-stage Graded U-Cup Diode with Shiftable Splitter Fin and Prow Taper.
    params shape: (18,) in [0, 1]
    """
    # 0-13: Inherited from Run 007
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
    
    # 14-17: New/Refined Parameters
    fin_len_scale = 0.0 + 0.8 * params[14]
    fin_width_scale = 0.0 + 0.7 * params[15]
    fin_shift = params[16]  # 0.0: upstream, 1.0: downstream (inside cavity)
    prow_power = 0.5 + 3.5 * params[17] # 0.5 (sharp) to 4.0 (blocky)

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
        throat_gap = jnp.clip(throat_base * (1.0 + th_grade * stage_centered), 0.20, 0.70)
        cup_ry = jnp.clip(cup_ry_base * (1.0 + ry_grade * stage_centered), 0.10, 0.40)

        x0 = margin_x + k * pitch
        dx = (x - x0) / pitch

        # 1. Converging wall ramp
        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / 0.1, 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

        # 2. Central Island (Prow + Cup)
        ux_rel = (dx - cup_offset) * aspect
        ry_local = jnp.clip(cup_ry + flare * jnp.clip(ux_rel, 0.0, cavity_len), 0.08, 0.45)
        ry_inner = jnp.maximum(0.02, ry_local - wall_thick)

        # Refined Prow with power-law taper
        prow_len = jnp.maximum(0.05, nose_thick * prow_stretch)
        in_prow = (ux_rel <= 0.0) & (ux_rel >= -prow_len) & (
            (jnp.abs(ux_rel) / prow_len)**prow_power + (abs_y / ry_local)**2 <= 1.0
        )
        
        in_outer_body = (ux_rel > 0.0) & (ux_rel <= cavity_len) & (abs_y <= ry_local)
        
        # Internal cavity (hollow)
        r_vertex = jnp.sqrt((jnp.minimum(0.0, ux_rel - 0.06) / 0.06) ** 2 + (abs_y / ry_inner) ** 2)
        hollow = (ux_rel >= 0.0) & (abs_y <= ry_inner) & (r_vertex <= 1.0)
        
        in_island = (in_prow | in_outer_body) & (~hollow)

        # 3. Shiftable Splitter Fin
        fin_len_actual = fin_len_scale * cavity_len
        fin_width_actual = fin_width_scale * ry_inner
        # fin_shift=0 -> fin is at [-fin_len, 0], fin_shift=1 -> fin is at [0, fin_len]
        fin_start = (fin_shift - 1.0) * fin_len_actual
        fin_end = fin_shift * fin_len_actual
        is_fin = (ux_rel >= fin_start) & (ux_rel <= fin_end) & (abs_y <= fin_width_actual)

        mask = jnp.maximum(mask, (wall_nozzle | in_island | is_fin).astype(jnp.float32))

    return mask

render_batch_coarse = jax.jit(jax.vmap(lambda p: render_refined_diode(p, 150, 60)))
render_batch_fine = jax.jit(jax.vmap(lambda p: render_refined_diode(p, 250, 100)))

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(707)

    # Base parameters inspired by Run 007
    # [0-13] are general shape, [14-17] are fin_len, fin_width, fin_shift, prow_power
    p007 = np.array([
        0.45, 0.42, 0.40, 0.45, 0.45, 0.50, 0.45, 0.40, 0.45, 0.55, 0.50, 0.0, 0.5, 0.5,
        0.5, 0.5, 0.0, 0.43 # fin_shift=0.0 (upstream), prow_power~2.0 (elliptical)
    ], dtype=np.float32)

    # Phase 1: Screen 128 candidates on Coarse Grid
    pop = rng.uniform(0.0, 1.0, size=(128, 18)).astype(np.float32)
    pop[0] = p007
    pop[1:10] = np.clip(p007 + rng.normal(0, 0.1, (9, 18)), 0, 1)
    
    scores_c = []
    for i in range(0, 128, 32):
        m_c = render_batch_coarse(pop[i:i+32])
        s_c, _, _ = evaluate_batch_fast(m_c, MATCHED_FAST_CONFIG)
        scores_c.append(np.array(s_c))
    scores_c = np.concatenate(scores_c)
    
    top_idx = np.argsort(scores_c)[::-1][:12]
    elite_params = pop[top_idx]
    
    # Phase 2: Fine-Grid Evolution (5 rounds of 8)
    # Start by evaluating the top 8 from coarse on the fine grid
    m_f = render_batch_fine(elite_params[:8])
    scores_f, _, _ = evaluate_batch_fast(m_f, DEFAULT_CONFIG)
    scores_f = np.array(scores_f)
    
    best_idx = np.argmax(scores_f)
    best_score = scores_f[best_idx]
    best_params = elite_params[best_idx]
    best_mask = np.array(m_f[best_idx])

    print(f"Initial fine best: {best_score:.4f}")

    for r in range(5):
        if time.time() - t0 > 260: break
        
        # Mutation around the current best
        sigma = 0.08 * (0.7**r)
        noise = rng.normal(0, sigma, (8, 18)).astype(np.float32)
        candidates = np.clip(best_params + noise, 0, 1)
        # Always keep the best
        candidates[0] = best_params 
        
        m_batch = render_batch_fine(candidates)
        s_batch, _, _ = evaluate_batch_fast(m_batch, DEFAULT_CONFIG)
        s_batch = np.array(s_batch)
        
        curr_best_idx = np.argmax(s_batch)
        if s_batch[curr_best_idx] > best_score:
            best_score = s_batch[curr_best_idx]
            best_params = candidates[curr_best_idx]
            best_mask = np.array(m_batch[curr_best_idx])
            print(f"Round {r} improvement: {best_score:.4f}")

    print(f"Final Design Time: {time.time()-t0:.1f}s | Best Di: {best_score:.4f}")
    print(f"Best Params: {best_params.tolist()}")
    
    return best_mask

if __name__ == "__main__":
    design = generate_design()
