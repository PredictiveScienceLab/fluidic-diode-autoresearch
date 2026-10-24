# Autoresearch for Mechanical Engineering: Passive Fluidic Diode Discovery

This repository contains the code, physics simulators, and complete experimental trajectories for our Substack post on repurposing the **autoresearch** pattern for mechanical engineering design: discovering a **2D passive fluidic diode** (a no-moving-parts check valve) and its **inner-loop GPU optimization algorithm**.

---

## 1. Two Campaigns Included in This Repository

We ran two 100+ iteration autonomous campaigns on a Google Cloud `g2-standard-8` VM (1× NVIDIA L4 GPU, 24 GB VRAM) using `gemini-2.5-pro` in multimodal mode (`--attach-image`):

1. **Take 1 — Unsteady Vortex Reward Hacking ([`take1_unsteady_vortex_hack/`](take1_unsteady_vortex_hack/)):**
   - Uses an 8,000-step 2D Lattice Boltzmann solver averaging flow over the final 1,000 steps.
   - Starting from a failing sawtooth nozzle ($\mathrm{Di} = 1.02$), the agent invented a legitimate cup-and-splitter diode up to $\mathrm{Di} = 3.22$ (`run_007`), and then used `jax.vmap` to exploit the finite 1,000-step measurement window by phase-locking a chaotic Kármán vortex street (`run_049`, $\mathrm{Di} = 383.24$).
2. **Take 2 — Steady-State Physics Oracle (Root Directory):**
   - Enforces three-plane incompressible mass conservation (inlet, outlet, and bulk flow agreement within $8\%$) over 14,000 steps and penalizes temporal oscillation ($\sigma_Q$) in the denominator.
   - Across 105 autonomous iterations ([`results.tsv`](results.tsv)), every unsteady vortex trick was rejected, and the loop made 18 steady-state improvements from $\mathrm{Di} = 1.02$ (`run_000_baseline`) to **$\mathrm{Di} = 2.18$** (`run_105`, corresponding to a $4.76\times$ pressure-drop asymmetry $\mathrm{Di}_{\Delta P} \approx \mathrm{Di}_Q^2$).

---

## 2. Repository Structure

| File | Role | Mutable by Agent? |
| :--- | :--- | :--- |
| [`simulator.py`](simulator.py) | 2D D2Q9 Regularized Smagorinsky Lattice Boltzmann (RLBM) solver in JAX (`DEFAULT_CONFIG` $300 \times 100$, `FAST_CONFIG` $180 \times 60$, and vectorized `evaluate_batch_fast` via `jax.vmap`). | **Read-Only** |
| [`evaluate.py`](evaluate.py) | Ground-truth evaluation harness, 5-minute budget enforcer, `results.tsv` logger, and 3-panel streamline plotter. | **Read-Only** |
| [`design.py`](design.py) | Parametric geometry generator + continuous GPU optimization script. Currently holds the winning Take 2 `run_105` design ($\mathrm{Di} = 2.18$). | **YES (Agent Workspace)** |
| [`program.md`](program.md) | System instructions, physical laws, and meta-level research strategy for the autonomous engineer. | **Read-Only** |
| [`run_autoresearch.py`](run_autoresearch.py) | Outer-loop orchestrator that queries an LLM (Gemini or OpenAI-compatible), runs `evaluate.py`, and applies the git keep/revert ratchet. | **Read-Only** |
| [`take1_unsteady_vortex_hack/`](take1_unsteady_vortex_hack/) | Complete archive of Take 1 (`simulator.py`, `design.py`, `results.tsv`, and `artifacts/`). | **Archive** |

---

## 3. Replicating the Results (Local or Cloud GPU with `uv`)

```bash
# 1. Install dependencies with uv
uv sync

# Optional: On a Linux machine with an NVIDIA GPU (e.g., GCP L4 or T4), install CUDA-enabled JAX:
uv pip install --upgrade "jax[cuda12]"

# 2. Evaluate the current winning Take 2 design (run_105, Di = 2.1806)
uv run python evaluate.py --tag test_run_105 --description "Verify winning Take 2 steady-state design"

# 3. Launch autonomous research iterations (using Gemini 2.5 Pro with multimodal flow inspection)
export GEMINI_API_KEY="your-api-key"
uv run python run_autoresearch.py --iterations 100 --provider gemini --model gemini-2.5-pro --attach-image
```
