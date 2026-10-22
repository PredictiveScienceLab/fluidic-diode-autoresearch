# Autoresearch for Mechanical Engineering: Passive Fluidic Diode Discovery

This repository demonstrates a Karpathy-style **autoresearch** loop applied to a physical mechanical/thermo-fluids design problem: discovering a **2D passive fluidic diode** (a no-moving-parts valvular conduit) and its **inner-loop continuous optimization algorithm** from scratch.

---

## 1. The Physics Problem & Ground-Truth Evaluator

Under a fixed macroscopic pressure drop $\Delta P = c_s^2 \Delta \rho$ across a $300 \times 100$ channel, the goal is to maximize the steady-state volumetric flow-rate diodicity:

$$\mathrm{Di} = \frac{Q_{\text{fwd}}}{Q_{\text{rev}}}$$

subject to engineering feasibility constraints:
1. **Minimum Forward Conductance:** $Q_{\text{fwd}} / Q_{\text{open}} \ge 0.12$ (prevents trivially plugging the pipe).
2. **Maximum Solid Volume Fraction:** $\le 60\%$ of the $250 \times 100$ design zone.
3. **Compute Budget:** `design.generate_design()` has a **5.0-minute (300s)** wall-clock budget per iteration to run continuous parameter optimization (e.g., CMA-ES, Nelder-Mead, multi-fidelity surrogate screening, or JAX adjoint gradients) before returning the candidate geometry.

---

## 2. Repository Structure

| File | Role | Mutable by Agent? |
| :--- | :--- | :--- |
| [`simulator.py`](simulator.py) | 2D D2Q9 Regularized Smagorinsky Lattice Boltzmann (RLBM) solver in JAX (`DEFAULT_CONFIG` $300 \times 100$ and `FAST_CONFIG` $180 \times 60$). | **Read-Only** |
| [`evaluate.py`](evaluate.py) | Ground-truth evaluation harness, 5-minute budget enforcer, `results.tsv` logger, and 3-panel streamline plotter. | **Read-Only** |
| [`design.py`](design.py) | Parametric geometry generator + continuous optimization script. Baseline is a 3-stage sawtooth nozzle-diffuser ($\mathrm{Di} = 1.0218$). | **YES (Agent Workspace)** |
| [`program.md`](program.md) | System instructions, physical laws, and meta-level research strategy for the autonomous engineer. | **Read-Only** |
| [`run_autoresearch.py`](run_autoresearch.py) | Outer-loop orchestrator that queries an LLM (Gemini or OpenAI-compatible), runs `evaluate.py`, and applies the keep/revert ratchet. | **Read-Only** |

---

## 3. Quickstart (Local with `uv`)

```bash
# 1. Run the baseline evaluation (generates artifacts/run_000_baseline.png and results.tsv)
uv run python evaluate.py --tag run_000_baseline --description "Baseline 3-stage sawtooth nozzle"

# 2. Launch 50 autonomous research iterations (using Gemini 2.5 Pro with multimodal flow inspection)
export GEMINI_API_KEY="your-api-key"
uv run python run_autoresearch.py --iterations 50 --provider gemini --model gemini-2.5-pro --attach-image
```

---

## 4. Running on a Cheap Google Cloud / Colab GPU (T4 or L4)

On a single Google Cloud / Colab **T4** or **L4** GPU:
1. Install CUDA-enabled JAX inside the `uv` environment:
   ```bash
   uv add "jax[cuda12]"
   ```
2. Each `FAST_CONFIG` inner-loop evaluation takes **~0.25s** on a T4 GPU (allowing `design.py` to run 200+ continuous optimization evaluations inside its 5-minute budget), and each high-resolution `DEFAULT_CONFIG` ground-truth evaluation in `evaluate.py` takes **~2.5s**.
3. Run 50–100 iterations unattended:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   uv run python run_autoresearch.py --iterations 75 --provider gemini --model gemini-2.5-pro --attach-image
   ```
   Progress curves are automatically updated after every run in `artifacts/progress.png`, and every flow field visualization is saved in `artifacts/run_XXX.png`.
