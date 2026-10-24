"""
Fluidic Diode Design & Continuous Optimization Module.
Optimized for high-diodicity via vortex-locking geometry refinement.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, evaluate_batch_fast

# The proven 19-parameter vector from run_049/062/079 (Score: 383.239)
# P16, P17, P18 are kept at 0.0 to maintain the stable base topology.
BEST_PARAMS = np.array([
    0.34585416, 0.6224917,  0.55515563, 0.31620237, 0.04744818, 0.38958547,
    0.30690935, 0.24575749, 0.44694495, 0.5715191,  0.8370709,  0.0,
    0.6155642,  0.36107752, 0.84232432, 0.5733315,  0.0,        0.0,
    0.0
], dtype=np.float32)

@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_graded_ucup_vortex_diode(params: jnp.ndarray, nx_design: int, ny: int) -> jnp.ndarray:
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

        s = jnp.clip(dx / nozzle_len, 0.0, 1.0)
        y_wall = 1.0 - (1.0 - throat_gap) * (s ** nozzle_pow)
        rear_edge = nozzle_len - undercut * jnp.clip((abs_y - throat_gap) / jnp.maximum(0.05, 1.0 - throat_gap), 0.0, 1.0)
        wall_nozzle = (dx >= 0.0) & (dx <= rear_edge) & (abs_y >= y_wall)

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

        fin_len_actual = fin_len_scale * cavity_len
        fin_width_actual = fin_width_scale * ry_inner
        is_fin = (ux_rel <= 0.0) & (ux_rel >= -fin_len_actual) & (abs_y <= fin_width_actual)

        mask = jnp.maximum(mask, (wall_nozzle | in_cup | is_fin).astype(jnp.float32))

    return mask

render_batch = jax.jit(
    jax.vmap(lambda p: render_graded_ucup_vortex_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)

def generate_design() -> np.ndarray:
    t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_params = BEST_PARAMS.copy()
    best_score = 383.239117
    
    # Perform a tight local search around the vortex-locking parameters
    # Focusing on fin geometry (indices 14, 15) to optimize reverse-flow blockage
    for r in range(3):
        if time.time() - t0 > 45.0: break
        
        cands = [best_params.copy()]
        for _ in range(15):
            p = best_params.copy()
            # Mutate fin dimensions with very small sigma
            p[[14, 15]] = np.clip(p[[14, 15]] + rng.normal(0, 0.005, 2), 0.0, 1.0)
            cands.append(p)
            
        masks = render_batch(jnp.asarray(np.stack(cands)))
        scores, _, _ = evaluate_batch_fast(masks, DEFAULT_CONFIG)
        scores = np.asarray(scores)
        
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_params = cands[idx]
            print(f"Update: {best_score:.6f} | Params: {best_params}")
            
    return np.asarray(render_batch(jnp.asarray(best_params[None, :]))[0])
