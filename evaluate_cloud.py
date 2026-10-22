"""
Cloud GPU Evaluation Harness (`evaluate_cloud.py`).

Synchronizes `design.py` to the Google Cloud NVIDIA L4 GPU VM (`smurf-l4-vm` in
`us-central1-b`), executes `evaluate.py` on the L4 GPU (exploiting `jax.vmap`
population batching), and pulls back the resulting `results.tsv` and diagnostic
streamline plots (`artifacts/<tag>.png` and `artifacts/latest_eval.png`).

Usage:
    uv run python evaluate_cloud.py --tag run_001 --description "Hypothesis description"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

VM_NAME = "smurf-l4-vm"
VM_ZONE = "us-central1-b"
REMOTE_DIR = "~/fluidic-diode"


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = f"/opt/homebrew/bin:{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
    return subprocess.run(cmd, env=env, check=check, text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate design.py on GCP NVIDIA L4 GPU.")
    parser.add_argument("--tag", type=str, required=True, help="Run tag (e.g., run_001)")
    parser.add_argument("--description", type=str, default="cloud_gpu_eval", help="Short hypothesis description")
    args = parser.parse_args()

    # 1. Push latest design.py (and simulator.py / evaluate.py if updated) to the L4 GPU VM
    print(f"[{args.tag}] Syncing design.py to GCP NVIDIA L4 GPU ({VM_NAME} @ {VM_ZONE})...")
    run_cmd([
        "gcloud", "compute", "scp",
        "--zone", VM_ZONE,
        "design.py",
        f"{VM_NAME}:{REMOTE_DIR}/design.py",
    ])

    # 2. Execute evaluate.py on the remote L4 GPU
    print(f"[{args.tag}] Running `evaluate.py` on NVIDIA L4 GPU (with jax.vmap acceleration)...")
    remote_cmd = (
        f"export PATH=$HOME/.local/bin:$PATH && "
        f"cd {REMOTE_DIR} && "
        f"uv run python evaluate.py --tag {args.tag} --description \"{args.description}\""
    )
    proc = run_cmd([
        "gcloud", "compute", "ssh",
        VM_NAME,
        "--zone", VM_ZONE,
        "--command", remote_cmd,
    ], check=False)

    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

    # 3. Pull back results.tsv and artifacts/<tag>.png + artifacts/latest_eval.png
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    run_cmd([
        "gcloud", "compute", "scp",
        "--zone", VM_ZONE,
        f"{VM_NAME}:{REMOTE_DIR}/results.tsv",
        "results.tsv",
    ], check=False)
    run_cmd([
        "gcloud", "compute", "scp",
        "--zone", VM_ZONE,
        f"{VM_NAME}:{REMOTE_DIR}/artifacts/{args.tag}.png",
        f"{VM_NAME}:{REMOTE_DIR}/artifacts/{args.tag}_mask.npy",
        f"{VM_NAME}:{REMOTE_DIR}/artifacts/latest_eval.png",
        f"{VM_NAME}:{REMOTE_DIR}/artifacts/latest_mask.npy",
        "artifacts/",
    ], check=False)


if __name__ == "__main__":
    main()
