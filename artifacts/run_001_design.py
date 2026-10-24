"""
Fluidic Diode Design & Continuous Optimization Module.
Run 001: Asymmetric Borda-Nozzle Cascade.
"""

from __future__ import annotations

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np

from simulator import DEFAULT_CONFIG, FAST_CONFIG, evaluate_batch_fast


@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))
def render_borda_nozzle_diode(
    params: jnp.ndarray,
    nx_design: int,
    ny: int,
) -> jnp.ndarray:
    """
    Renders a diode consisting of repeating stages with a converging nozzle
    and a re-entrant 'Borda' lip designed to catch reverse flow.
    
    Params (5): [throat_gap, nozzle_len, lip_height, lip_sharpness, pitch]
    """
    # Map [0, 1] params to physical ranges
    throat_gap = 0.20 + 0.35 * params[0]
    nozzle_len = 0.30 + 0.50 * params[1]
    lip_h      = 0.05 + 0.25 * params[2]
    lip_ang    = 0.10 + 0.80 * params[3]  # Controls the 'hook' of the lip
    pitch      = 0.25 + 0.15 * params[4]

    x = jnp.linspace(0.0, 1.0, nx_design)[:, None]
    y = jnp.linspace(-1.0, 1.0, ny)[None, :]
    abs_y = jnp.abs(y)

    n_stages = 3
    margin_x = 0.05
    mask = jnp.zeros((nx_design, ny), dtype=jnp.float32)

    for k in range(n_stages):
        x0 = margin_x + k * pitch
        # 1. Main Nozzle Ramp (Streamlined for +x)
        dx = (x - x0) / nozzle_len
        y_ramp = 1.0 - (1.0 - throat_gap) * jnp.clip(dx, 0.0, 1.0)
        ramp = ((dx >= 0.0) & (dx <= 1.0) & (abs_y >= y_ramp)).astype(jnp.float32)
        
        # 2. Re-entrant Borda Lip (Catches -x flow)
        # Positioned at the 'wide' entrance of the nozzle (x0)
        # We create a small 'hook' pointing into the flow
        lx = (x - x0) / 0.15
        ly = (abs_y - 1.0) # distance from wall
        # Rotated coordinate for the lip
        c, s = jnp.cos(lip_ang), jnp.sin(lip_ang)
        # Lip extends from (x0, 1.0) back towards +x and inward
        rx = (x - x0) * c - (1.0 - abs_y) * s
        ry = (x - x0) * s + (1.0 - abs_y) * c
        
        lip = ((rx >= 0.0) & (rx <= 0.08) & (jnp.abs(ry) <= lip_h)).astype(jnp.float32)
        
        mask = jnp.maximum(mask, ramp)
        mask = jnp.maximum(mask, lip)

    # Ensure the outer frame is solid to avoid leaks
    mask = jnp.where((abs_y > 0.98), 1.0, mask)
    
    return jnp.clip(mask, 0.0, 1.0)


render_batch_coarse = jax.jit(
    jax.vmap(lambda p: render_borda_nozzle_diode(p, FAST_CONFIG.nx - 2 * FAST_CONFIG.pad_x, FAST_CONFIG.ny))
)
render_batch_fine = jax.jit(
    jax.vmap(lambda p: render_borda_nozzle_diode(p, DEFAULT_CONFIG.nx - 2 * DEFAULT_CONFIG.pad_x, DEFAULT_CONFIG.ny))
)


def generate_design() -> np.ndarray:
    t0 = time.time()
    np.set_printoptions(precision=8, suppress=True)
    rng = np.random.default_rng(1337)

    # Phase 1: Broad screening on FAST_CONFIG
    # Batch size 64 for good coverage of the 5D space
    n_coarse = 64
    pop_coarse = rng.uniform(0.0, 1.0, size=(n_coarse, 5)).astype(np.float32)
    
    masks_c = render_batch_coarse(jnp.asarray(pop_coarse))
    scores_c, _, _ = evaluate_batch_fast(masks_c, FAST_CONFIG)
    scores_c_np = np.asarray(scores_c)

    # Phase 2: High-resolution refinement on DEFAULT_CONFIG
    # Take top 12 and perform a local search around the best coarse candidate
    top_idx = np.argsort(scores_c_np)[::-1][:6]
    best_coarse_params = pop_coarse[top_idx[0]]
    
    # Generate local variations around the best coarse candidate
    refine_pop = [best_coarse_params]
    for _ in range(11):
        noise = rng.normal(0, 0.05, size=5).astype(np.float32)
        refine_pop.append(np.clip(best_coarse_params + noise, 0.0, 1.0))
    
    pop_fine = np.array(refine_pop)
    masks_f = render_batch_fine(jnp.asarray(pop_fine))
    scores_f, qf_f, qr_f = evaluate_batch_fast(masks_f, DEFAULT_CONFIG)
    scores_f_np = np.asarray(scores_f)

    best_idx = int(np.argmax(scores_f_np))
    best_fine_params = pop_fine[best_idx]
    
    print(f"[design.py] Finished in {time.time()-t0:.1f}s | Best Fine Di = {scores_f_np[best_idx]:.6f}")
    print("BEST_FINE_PARAMS:", best_fine_params.tolist())
    
    return np.asarray(masks_f[best_idx], dtype=np.float32)

if __name__ == "__main__":
    # Test generation
    mask = generate_design()
    print("Mask shape:", mask.shape)
