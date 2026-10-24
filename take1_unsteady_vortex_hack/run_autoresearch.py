"""
Autonomous Research Loop Orchestrator (`run_autoresearch.py`).

Runs a Karpathy-style outer-loop autoresearch agent for N iterations (e.g., 50-100)
on a local machine or a Google Cloud / Colab GPU (T4 / L4 / A100).

At each iteration k:
1. Reads `program.md`, the current best `design.py`, and the full experiment history
   from `results.tsv`.
2. Optionally attaches the latest flow-field diagnostic image (`artifacts/latest_eval.png`)
   so multimodal LLMs (Gemini / Claude / GPT-4o) can visually inspect the streamlines
   and vortex locking in forward vs. reverse flow.
3. Prompts the LLM to formulate a new physical/algorithmic hypothesis and output a
   complete updated `design.py`.
4. Executes `evaluate.py` with a hard timeout enforcing the 5-minute design budget.
5. Applies the Ratchet Rule:
   - If `score` > `best_score` and `is_valid == True`, commits `design.py` and updates
     `best_design.py`.
   - Otherwise, restores `design.py` from `best_design.py` while keeping the failed
     experiment logged in `results.tsv`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def parse_eval_output(stdout: str) -> dict:
    """Parse the `--- EVALUATION SUMMARY ---` block from evaluate.py stdout."""
    info = {"score": 0.0, "raw_diodicity": 0.0, "is_valid": False}
    in_block = False
    for line in stdout.splitlines():
        if "--- EVALUATION SUMMARY ---" in line:
            in_block = True
            continue
        if in_block and line.startswith("---"):
            break
        if in_block and ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().split()[0]
            if k in ("score", "raw_diodicity", "q_fwd", "q_rev", "open_ratio", "solid_fraction", "design_time_s", "sim_time_s"):
                try:
                    info[k] = float(v)
                except ValueError:
                    pass
            elif k == "is_valid":
                info[k] = (v.lower() == "true")
    return info


def call_gemini_api(prompt: str, image_path: Path | None, model: str, api_key: str) -> str:
    """Call Google Gemini REST API with automatic multi-model fallback and retry on 503/429."""
    candidate_models = [
        model,
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash-lite",
    ]
    seen = []
    models_to_try = [m for m in candidate_models if not (m in seen or seen.append(m))]

    parts = [{"text": prompt}]
    if image_path is not None and image_path.exists():
        img_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": img_b64,
            }
        })
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.75},
    }

    last_err = None
    for attempt in range(4):
        for m in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"Empty Gemini response: {data}")
                text_parts = [p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])]
                return "\n".join(text_parts)
            except Exception as e:
                last_err = e
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"All Gemini models failed after retries: {last_err}")


def call_openai_compatible_api(prompt: str, model: str, api_key: str, base_url: str) -> str:
    """Call an OpenAI-compatible Chat Completions endpoint."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def extract_hypothesis_and_code(response_text: str) -> tuple[str, str]:
    """Extract a short hypothesis description and the python code block for design.py."""
    hyp_match = re.search(r"HYPOTHESIS:\s*(.+)", response_text)
    hypothesis = hyp_match.group(1).strip() if hyp_match else "Agent iteration"

    code_blocks = re.findall(r"```python\s*(.*?)```", response_text, flags=re.DOTALL)
    if not code_blocks:
        raise ValueError("No ```python ... ``` code block found in LLM response.")
    return hypothesis[:180], code_blocks[-1].strip() + "\n"


def plot_progress(tsv_path: Path, out_path: Path):
    """Plot the autoresearch progress curve (best score & iteration scores vs run index)."""
    if not tsv_path.exists():
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = tsv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return

    scores = []
    tags = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) >= 3:
            tags.append(cols[1])
            try:
                scores.append(float(cols[2]))
            except ValueError:
                scores.append(0.0)

    best_so_far = []
    cur_best = 0.0
    for s in scores:
        cur_best = max(cur_best, s)
        best_so_far.append(cur_best)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.55, 1.0]})
    iters = list(range(len(scores)))
    valid_iters = [i for i, s in zip(iters, scores) if s > 0.5]
    valid_scores = [s for s in scores if s > 0.5]

    # Panel 1: Full 117-iteration trajectory on log scale
    ax1.scatter(valid_iters, valid_scores, color="#66a61e", alpha=0.65, s=34, label="Valid Iteration Score (Di)")
    ax1.plot(iters, [max(1.0, b) for b in best_so_far], color="#d95f02", linewidth=2.4, label="Best Discovered Score (Ratchet)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Autoresearch Iteration (0 to 116)")
    ax1.set_ylabel("Effective Diodicity Score Di = Q_fwd / Q_rev (log scale)")
    ax1.set_title("Full Campaign (117 Runs): Vortex Mode-Locking Ratchet")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="upper left")

    # Annotate key milestones on Panel 1
    milestones = [
        (0, 1.0195, "run_000 (1.02)"),
        (4, 2.4386, "run_004 (2.44)"),
        (7, 3.2239, "run_007 (3.22)"),
        (8, 18.5624, "run_008 (18.56)"),
        (20, 37.8020, "run_020 (37.80)"),
        (49, 383.2391, "run_049 (383.24)"),
    ]
    for idx, val, label in milestones:
        if idx < len(scores):
            ax1.annotate(
                label,
                xy=(idx, val),
                xytext=(idx + 2.5, val * 1.18 if val < 100 else val * 0.72),
                fontsize=8.5,
                fontweight="bold",
                color="#8c2d04",
                arrowprops=dict(arrowstyle="->", color="#d95f02", lw=0.9),
            )

    # Panel 2: Early architectural discovery zoom (Runs 0 to 7)
    early_end = min(8, len(scores))
    ax2.scatter(range(early_end), scores[:early_end], color="#66a61e", s=50, zorder=3, label="Iteration Score")
    ax2.plot(range(early_end), best_so_far[:early_end], color="#d95f02", linewidth=2.4, zorder=2, label="Ratchet Best")
    for idx in range(early_end):
        if scores[idx] >= best_so_far[idx] - 1e-6:
            ax2.annotate(f"{scores[idx]:.2f}", (idx, scores[idx]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Early Iteration (Runs 0 to 7)")
    ax2.set_ylabel("Effective Diodicity Score (Linear Scale)")
    ax2.set_title("Early Topological Evolution (Runs 0–7)")
    ax2.set_ylim(0.85, 3.65)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run autonomous fluidic diode research loop.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of additional autoresearch iterations to run")
    parser.add_argument("--start-iter", type=int, default=0, help="Starting iteration number (0 = auto-detect from results.tsv)")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini", help="LLM API provider")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="Model name")
    parser.add_argument("--pro-model", type=str, default="gemini-3.1-pro-preview", help="Pro model name for every 5th deep architectural step")
    parser.add_argument("--base-url", type=str, default="https://api.openai.com/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--attach-image", action="store_true", default=True, help="Attach latest flow field image")
    args = parser.parse_args()

    program_md = Path("program.md").read_text(encoding="utf-8")
    design_path = Path("design.py")
    best_design_path = Path("artifacts/best_design.py")
    best_plot_path = Path("artifacts/best_eval.png")
    tsv_path = Path("results.tsv")

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    # Ensure baseline Run 0 has been evaluated
    if not tsv_path.exists():
        print("=== Running initial baseline evaluation (run_000_baseline) ===", flush=True)
        subprocess.run(
            [sys.executable, "evaluate.py", "--tag", "run_000_baseline", "--description", "Initial sawtooth baseline"],
            check=True,
        )

    # Determine current best score and highest run index from results.tsv
    best_score = 0.0
    max_run_idx = 0
    lines = tsv_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) >= 2:
            m = re.match(r"run_(\d+)", cols[1])
            if m:
                max_run_idx = max(max_run_idx, int(m.group(1)))
        if len(cols) >= 9 and cols[8].lower() == "true":
            best_score = max(best_score, float(cols[2]))

    start_it = args.start_iter if args.start_iter > 0 else (max_run_idx + 1)

    if not best_design_path.exists():
        shutil.copyfile(design_path, best_design_path)
    if Path("artifacts/latest_eval.png").exists() and not best_plot_path.exists():
        shutil.copyfile("artifacts/latest_eval.png", best_plot_path)

    print(f"Starting autoresearch loop at run_{start_it:03d} for {args.iterations} iterations (best_score = {best_score:.6f})", flush=True)

    for it in range(start_it, start_it + args.iterations):
        tag = f"run_{it:03d}"
        history_text = tsv_path.read_text(encoding="utf-8") if tsv_path.exists() else ""
        current_best_code = best_design_path.read_text(encoding="utf-8")

        active_model = args.pro_model if (it % 5 == 0 and args.pro_model) else args.model

        prompt = (
            f"{program_md}\n\n"
            f"## Current Best Score: {best_score:.6f}\n\n"
            f"## Experiment Log (`results.tsv`):\n```tsv\n{history_text[-8000:]}\n```\n\n"
            f"## Current Best `design.py`:\n```python\n{current_best_code}\n```\n\n"
            f"Formulate iteration `{tag}`. Critical engineering rules:\n"
            f"- Keep `generate_design()` execution time between **25 and 60 seconds** on the L4 GPU (e.g., 128 coarse `MATCHED_FAST_CONFIG` candidates in batches of 32 + at most 3 to 5 rounds of 8 fine `DEFAULT_CONFIG` candidates).\n"
            f"- **DO NOT** run 20 rounds of 32 `DEFAULT_CONFIG` simulations! And whenever you update `best_fine_score` or `best_fine_mask`, ALWAYS save and print the exact winning parameter vector `best_fine_params` so future runs can warm-start directly from it!\n"
            f"First write a single line:\n"
            f"HYPOTHESIS: <your concise physical and algorithmic hypothesis>\n"
            f"Then provide the complete, self-contained Python file for `design.py` inside a ```python ... ``` block."
        )

        print(f"\n[{tag}] Requesting hypothesis & design.py from {args.provider}:{active_model}...", flush=True)
        try:
            if args.provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
                try:
                    resp = call_gemini_api(
                        prompt,
                        best_plot_path if args.attach_image else None,
                        active_model,
                        api_key,
                    )
                except Exception as e_inner:
                    print(f"[{tag}] Primary model {active_model} error ({e_inner}), falling back to gemini-2.5-flash...", flush=True)
                    resp = call_gemini_api(
                        prompt,
                        best_plot_path if args.attach_image else None,
                        "gemini-2.5-flash",
                        api_key,
                    )
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "")
                resp = call_openai_compatible_api(prompt, active_model, api_key, args.base_url)

            hypothesis, new_code = extract_hypothesis_and_code(resp)
        except Exception as e:
            print(f"[{tag}] LLM generation error: {e}", flush=True)
            continue

        print(f"[{tag}] Hypothesis: {hypothesis}", flush=True)
        design_path.write_text(new_code, encoding="utf-8")
        shutil.copyfile(design_path, Path(f"artifacts/{tag}_design.py"))

        # Run evaluate.py with hard timeout (300s design + 90s eval margin = 390s)
        try:
            proc = subprocess.run(
                [sys.executable, "evaluate.py", "--tag", tag, "--description", hypothesis],
                capture_output=True,
                text=True,
                timeout=390,
            )
            print(proc.stdout, flush=True)
            if proc.returncode != 0:
                print(f"[{tag}] evaluate.py failed:\n{proc.stderr[-1200:]}", flush=True)
                # Log crash row in results.tsv so the agent learns from syntax/runtime errors!
                with tsv_path.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{tag}\t0.000000\t0.000000\t0.000000\t0.000000\t"
                        f"0.0000\t0.0000\tFalse\t0.00\t0.00\tCRASH: {hypothesis[:100]}\n"
                    )
                shutil.copyfile(best_design_path, design_path)
                continue
            eval_info = parse_eval_output(proc.stdout)
        except subprocess.TimeoutExpired:
            print(f"[{tag}] TIMEOUT (>390s). Reverting to best_design.py.", flush=True)
            with tsv_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{tag}\t0.000000\t0.000000\t0.000000\t0.000000\t"
                    f"0.0000\t0.0000\tFalse\t390.00\t0.00\tTIMEOUT: {hypothesis[:100]}\n"
                )
            shutil.copyfile(best_design_path, design_path)
            continue

        score = eval_info.get("score", 0.0)
        is_valid = eval_info.get("is_valid", False)

        if is_valid and score > best_score:
            print(f"[{tag}] NEW BEST SCORE: {score:.6f} (previous: {best_score:.6f})! Keeping design.", flush=True)
            best_score = score
            shutil.copyfile(design_path, best_design_path)
            if Path("artifacts/latest_eval.png").exists():
                shutil.copyfile("artifacts/latest_eval.png", best_plot_path)
            subprocess.run(["git", "add", "design.py", "results.tsv"], check=False)
            subprocess.run(["git", "commit", "-m", f"{tag}: score={score:.4f} | {hypothesis}"], check=False)
        else:
            print(f"[{tag}] Score {score:.6f} did not beat {best_score:.6f} (valid={is_valid}). Reverting design.py.", flush=True)
            shutil.copyfile(best_design_path, design_path)

        plot_progress(tsv_path, Path("artifacts/progress.png"))


if __name__ == "__main__":
    main()
