"""
Read-Only Ground-Truth Evaluation & Visualization Harness for Fluidic Diode Autoresearch.

Usage:
    uv run python evaluate.py [--tag baseline] [--save-dir artifacts]

What it does:
1. Imports and executes `design.generate_design()` while enforcing the 5-minute
   (300s) compute budget.
2. Runs the ground-truth D2Q9 Regularized Smagorinsky LBM solver (`DEFAULT_CONFIG`)
   in both the forward (+x) and reverse (-x) directions under a fixed pressure drop.
3. Prints a standardized, machine-parseable summary block (`--- EVALUATION SUMMARY ---`)
   and appends a row to `results.tsv`.
4. Generates a publication-quality 3-panel comparison plot (`artifacts/latest_eval.png`
   and `artifacts/<tag>.png`) showing the solid geometry, forward velocity/streamlines,
   and reverse velocity/streamlines + recirculation zones.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulator import DEFAULT_CONFIG, evaluate_binary_design
import design


DESIGN_TIME_BUDGET_SECONDS = 300.0  # 5-minute compute budget for design.py


def render_diagnostic_plot(eval_res: dict, tag: str, out_path: Path, design_time: float):
    """Save a publication-grade 3-panel visualization of the fluidic diode flow fields."""
    full_mask = eval_res["full_mask"].T  # (ny, nx)
    ux_f = eval_res["ux_fwd"].T
    uy_f = eval_res["uy_fwd"].T
    ux_r = eval_res["ux_rev"].T
    uy_r = eval_res["uy_rev"].T

    speed_f = np.sqrt(ux_f ** 2 + uy_f ** 2)
    speed_r = np.sqrt(ux_r ** 2 + uy_r ** 2)
    vmax = max(float(np.percentile(speed_f, 99.5)), float(np.percentile(speed_r, 99.5)), 0.05)

    ny, nx = full_mask.shape
    x_grid = np.arange(nx)
    y_grid = np.arange(ny)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, constrained_layout=True)

    # Panel 1: Solid Geometry + Forward Streamlines
    ax0 = axes[0]
    ax0.imshow(
        full_mask,
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.2,
        aspect="equal",
    )
    ax0.axvline(DEFAULT_CONFIG.pad_x, color="#d95f02", linestyle="--", linewidth=1.0, alpha=0.8)
    ax0.axvline(nx - DEFAULT_CONFIG.pad_x, color="#d95f02", linestyle="--", linewidth=1.0, alpha=0.8)
    ax0.set_title(
        f"Fluidic Diode [{tag}] | Score (Di) = {eval_res['score']:.4f} "
        f"(Raw Di = {eval_res['raw_diodicity']:.4f}) | Solid = {eval_res['solid_fraction']*100:.1f}% "
        f"| Design = {design_time:.1f}s",
        fontsize=10.5,
        fontweight="bold",
    )
    ax0.set_ylabel("y (lattice)")
    fig.colorbar(im0 := ax0.images[0], ax=ax0, orientation="vertical", fraction=0.018, pad=0.01, label="solid")
    im0.colorbar.ax.set_visible(False)

    # Panel 2: Forward Flow (+x) Speed & Streamlines
    ax1 = axes[1]
    masked_speed_f = np.ma.masked_where(full_mask > 0.5, speed_f)
    im1 = ax1.imshow(
        masked_speed_f,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    ax1.imshow(
        np.ma.masked_where(full_mask <= 0.5, full_mask),
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    try:
        ax1.streamplot(
            x_grid,
            y_grid,
            ux_f,
            uy_f,
            color="white",
            linewidth=0.65,
            density=1.1,
            arrowsize=0.8,
        )
    except Exception:
        pass
    ax1.set_title(
        f"Forward Flow (+x, Low Resistance) | Q_fwd = {eval_res['q_fwd']:.5f} "
        f"({eval_res['open_ratio']*100:.1f}% of open pipe, Re = {eval_res['re_fwd']:.0f})",
        fontsize=10.5,
    )
    ax1.set_ylabel("y (lattice)")
    fig.colorbar(im1, ax=ax1, orientation="vertical", fraction=0.018, pad=0.01, label="|u|")

    # Panel 3: Reverse Flow (-x) Speed & Streamlines
    ax2 = axes[2]
    masked_speed_r = np.ma.masked_where(full_mask > 0.5, speed_r)
    im2 = ax2.imshow(
        masked_speed_r,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        aspect="equal",
    )
    ax2.imshow(
        np.ma.masked_where(full_mask <= 0.5, full_mask),
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        aspect="equal",
    )
    try:
        ax2.streamplot(
            x_grid,
            y_grid,
            ux_r,
            uy_r,
            color="white",
            linewidth=0.65,
            density=1.1,
            arrowsize=0.8,
        )
    except Exception:
        pass
    ax2.set_title(
        f"Reverse Flow (-x, High Resistance / Vortex Locking) | Q_rev = {eval_res['q_rev']:.5f} "
        f"(Re = {eval_res['re_rev']:.0f})",
        fontsize=10.5,
    )
    ax2.set_xlabel("x (lattice units)")
    ax2.set_ylabel("y (lattice)")
    fig.colorbar(im2, ax=ax2, orientation="vertical", fraction=0.018, pad=0.01, label="|u|")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    latest_path = out_path.parent / "latest_eval.png"
    if out_path != latest_path:
        fig.savefig(latest_path, dpi=160)
    plt.close(fig)


def log_to_tsv(tsv_path: Path, tag: str, eval_res: dict, design_time: float, sim_time: float, description: str):
    """Append evaluation metrics to results.tsv."""
    header = (
        "timestamp\ttag\tscore\traw_diodicity\tq_fwd\tq_rev\topen_ratio\t"
        "solid_fraction\tis_valid\tdesign_time_s\tsim_time_s\tdescription\n"
    )
    if not tsv_path.exists():
        tsv_path.write_text(header, encoding="utf-8")

    row = (
        f"{datetime.now().isoformat(timespec='seconds')}\t"
        f"{tag}\t"
        f"{eval_res['score']:.6f}\t"
        f"{eval_res['raw_diodicity']:.6f}\t"
        f"{eval_res['q_fwd']:.6f}\t"
        f"{eval_res['q_rev']:.6f}\t"
        f"{eval_res['open_ratio']:.4f}\t"
        f"{eval_res['solid_fraction']:.4f}\t"
        f"{eval_res['is_valid']}\t"
        f"{design_time:.2f}\t"
        f"{sim_time:.2f}\t"
        f"{description}\n"
    )
    with tsv_path.open("a", encoding="utf-8") as f:
        f.write(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fluidic diode design.")
    parser.add_argument("--tag", type=str, default="eval", help="Tag for artifacts and results.tsv")
    parser.add_argument("--description", type=str, default="manual_eval", help="Short description of experiment")
    parser.add_argument("--save-dir", type=str, default="artifacts", help="Directory to save plots and masks")
    parser.add_argument("--no-log", action="store_true", help="Do not append to results.tsv")
    args = parser.parse_args()

    print(f"=== Running design.generate_design() [Budget: {DESIGN_TIME_BUDGET_SECONDS:.0f}s] ===")
    t0 = time.time()
    candidate_mask = design.generate_design()
    design_time = time.time() - t0

    if design_time > DESIGN_TIME_BUDGET_SECONDS:
        print(
            f"WARNING: design.generate_design() took {design_time:.1f}s, "
            f"exceeding the {DESIGN_TIME_BUDGET_SECONDS:.0f}s budget!"
        )

    print("=== Running Ground-Truth D2Q9 LBM Evaluation (Forward & Reverse) ===")
    t1 = time.time()
    eval_res = evaluate_binary_design(candidate_mask, config=DEFAULT_CONFIG, return_fields=True)
    sim_time = time.time() - t1

    # Penalize score if design runtime exceeded budget
    if design_time > DESIGN_TIME_BUDGET_SECONDS:
        eval_res["score"] = 0.0
        eval_res["is_valid"] = False

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f"{args.tag}.png"
    mask_path = save_dir / f"{args.tag}_mask.npy"
    np.save(mask_path, eval_res["binary_design"])
    np.save(save_dir / "latest_mask.npy", eval_res["binary_design"])

    render_diagnostic_plot(eval_res, args.tag, plot_path, design_time)

    if not args.no_log:
        log_to_tsv(
            Path("results.tsv"),
            args.tag,
            eval_res,
            design_time,
            sim_time,
            args.description,
        )

    print("\n--- EVALUATION SUMMARY ---")
    print(f"tag:              {args.tag}")
    print(f"score:            {eval_res['score']:.6f}")
    print(f"raw_diodicity:    {eval_res['raw_diodicity']:.6f}")
    print(f"q_fwd:            {eval_res['q_fwd']:.6f}")
    print(f"q_rev:            {eval_res['q_rev']:.6f}")
    print(f"open_ratio:       {eval_res['open_ratio']:.4f} (min required: {DEFAULT_CONFIG.min_open_ratio:.2f})")
    print(f"solid_fraction:   {eval_res['solid_fraction']:.4f} (max allowed: {DEFAULT_CONFIG.max_solid_fraction:.2f})")
    print(f"re_fwd:           {eval_res['re_fwd']:.1f}")
    print(f"re_rev:           {eval_res['re_rev']:.1f}")
    print(f"is_valid:         {eval_res['is_valid']}")
    print(f"design_time_s:    {design_time:.2f}")
    print(f"sim_time_s:       {sim_time:.2f}")
    print(f"plot_saved:       {plot_path}")
    print("--------------------------")


if __name__ == "__main__":
    main()
