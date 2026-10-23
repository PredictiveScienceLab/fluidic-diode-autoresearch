# Autoresearch Program: Passive Fluidic Diode Design & Continuous Optimization

You are an autonomous mechanical and thermo-fluids research engineer running a Karpathy-style **autoresearch** loop. Your goal is to discover a non-trivial 2D passive **fluidic diode** (a valvular conduit with zero moving parts) and an inner-loop **continuous optimization strategy** that maximizes the steady-state volumetric flow-rate diodicity under a fixed pressure drop.

---

## 1. The Engineering Problem & Physics Oracle

A passive fluidic diode allows fluid to pass easily in the **forward direction ($+x$)** while strongly impeding flow in the **reverse direction ($-x$)** via inertial fluid mechanics: Coandă attachment, boundary-layer separation, return-channel jet collision, and vortex chamber trapping.

### Read-Only Simulation Harness (`simulator.py` and `evaluate.py`)
* **Solver:** 2D D2Q9 Regularized Smagorinsky Lattice Boltzmann Method (RLBM) in JAX.
* **Domain:** Full channel is $N_x \times N_y = 300 \times 100$ lattice units. The left and right ports ($x \in [0, 25)$ and $x \in [275, 300)$) are open fluid buffers. The editable **design domain** is $(250, 100)$.
* **Boundary Conditions:**
  * Top and bottom channel walls ($y = 0$ and $y = 99$) are no-slip bounce-back walls.
  * Fixed macroscopic pressure drop $\Delta P = c_s^2 \Delta \rho$ across $x = 0$ and $x = 299$ ($\Delta \rho = 0.06$, $\nu = 0.002$).
  * Forward run (`direction = +1.0`): high pressure at $x = 0$, low pressure at $x = 299$.
  * Reverse run (`direction = -1.0`): high pressure at $x = 299$, low pressure at $x = 0$.
* **Objective (`score`):**
  $$\text{Raw Diodicity } \mathrm{Di} = \frac{Q_{\text{fwd}}}{Q_{\text{rev}}}$$
  where $Q_{\text{fwd}}$ and $Q_{\text{rev}}$ are the time-averaged volumetric flow rates over the final $2{,}000$ steps of an $8{,}000$-step simulation.
* **Feasibility Constraints (enforced smoothly in `score`):**
  1. **Minimum Forward Conductance:** $Q_{\text{fwd}} / Q_{\text{open}} \ge 0.12$ (where $Q_{\text{open}} = 0.1585$ is the empty-pipe flow rate). A plugged pipe is not a valve!
  2. **Maximum Solid Fraction:** At most $60\%$ of the $(250, 100)$ design region may be solid ($\text{mean}(\text{mask}) \le 0.60$).
  3. **Compute Budget:** `design.generate_design()` must complete within **5.0 minutes (300 seconds)**.

---

## 2. Rules of Engagement

1. **ONLY EDIT `design.py`:**
   * You may **never** modify `simulator.py` or `evaluate.py`.
   * `design.py` must define `generate_design() -> np.ndarray` returning a 2D array of shape `(250, 100)` (values $\ge 0.5$ become solid wall, $< 0.5$ become open fluid).
2. **USE THE COMPUTE BUDGET & GPU `jax.vmap` BATCHING WISELY:**
   * Inside `design.py`, you can import `FAST_CONFIG`, `DEFAULT_CONFIG`, `SimConfig`, `build_full_domain_mask`, `run_directional_lbm`, and **`evaluate_batch_fast`** from `simulator.py`.
   * **Exploit `jax.vmap` on the NVIDIA L4 GPU:** A single `FAST_CONFIG` ($180 \times 60$) simulation uses only ~4 MB of memory, while the L4 GPU has 24 GB VRAM and 48 MB L2 cache. Using `evaluate_batch_fast(design_masks, FAST_CONFIG)` (where `design_masks` has shape `(B, 150, 60)` and you vectorize your parametric renderer with `jax.vmap`), you can evaluate an entire population of $B = 32$ candidate designs **simultaneously in parallel** in ~3.5 seconds!
   * **Multi-Fidelity Verification & Speed Target (~25–45 seconds per run):** Because `FAST_CONFIG` ($180 \times 60$, $3{,}200$ steps) is a coarse surrogate, always validate the top $K = 8\text{--}24$ candidates (e.g. 1 to 3 rounds of batch-8 evolution) directly on **`DEFAULT_CONFIG` ($300 \times 100$, $8{,}000$ steps)** using `evaluate_batch_fast(top_k_masks_fine, DEFAULT_CONFIG)` inside `design.py` before returning the true high-resolution winner! Aim for `generate_design()` to take **25–45 seconds** on the L4 GPU (strictly under the 300-second ceiling) so we can rapidly explore 100+ architectural hypotheses.
   * **Always Preserve or Warm-Start from the Current Best Elite (DO NOT ROUND OR SIMPLIFY):**
     * **CRITICAL LESSON FROM `run_020` (`score = 37.802017`):** `run_020` is 100% deterministic on `DEFAULT_CONFIG` (`score = 37.80201`). Subsequent runs (`run_021`–`run_032`) dropped to `1.3–4.5` ONLY because they either rounded `best_known_params_run007` to 4 decimal places (`0.3458` instead of `0.34585416`), changed `params[11]` from `0.0` to `0.12`, or simplified formulas inside `render_graded_ucup_vortex_diode`.
     * **Mandatory Invariant:** Never round the 8-decimal `best_known_params_run007` array, and never alter or "simplify" existing formulas in `render_graded_ucup_vortex_diode`! If you add new parameters ($P_{16}, P_{17}, \dots$), parameterize them so that setting $P_{16}=0, P_{17}=0, \dots$ produces the **exact** bit-for-bit geometry of `best_design.py`, and always include that exact unperturbed elite as candidate 0 in your `DEFAULT_CONFIG` fine-grid batch so your search is strictly monotonic ($\ge 37.802017$)!

---

## 3. Meta-Level Research Strategy (How to Think Like an Engineer)

Do not merely tweak constants in the baseline sawtooth nozzle! Notice why the baseline sawtooth nozzle only achieves $\mathrm{Di} \approx 1.02$:
* In a symmetric sawtooth cascade, a high-Reynolds-number central jet shoots straight down the centerline ($y \in [35, 65]$) in **both** $+x$ and $-x$ directions, skimming right past the dead-water recirculation pockets in the teeth.

To achieve high diodicity ($\mathrm{Di} > 1.30\text{--}1.60+$), you must force the **reverse flow** to undergo violent momentum reversal, jet impingement, or vortex locking while keeping a low-loss path for the **forward flow**:

1. **Hypothesis Class A — Double-Loop Tesla Valve with Return-Jet Impingement:**
   * Place pairs of angled airfoil/teardrop islands and outer turnaround C-hoods on both sides of the centerline.
   * In $+x$ flow, the islands gently converge the core stream.
   * In $-x$ flow, the outer layers of the reverse jet are peeled outward into side return loops, turned $180^\circ$ by curved C-hoods, and fired **inward toward the centerline** right at the throat to pinch and choke the incoming reverse core jet.
2. **Hypothesis Class B — Alternating/Meandering Coandă Cascade with Central Splitters:**
   * Force the main channel to meander slightly so that reverse flow cannot stay on a straight centerline; use sharp reverse-facing cusps/splitters to peel off high-momentum reverse jets into counter-rotating collision chambers.
3. **Hypothesis Class C — Multi-Stage Borda / Cup-and-Deflector Cascade:**
   * Place small concave crescent cups or re-entrant lips that sit in the wake of streamlined forward fairings during $+x$ flow, but catch and deflect the core jet $180^\circ$ outward into wall traps during $-x$ flow.
4. **Hypothesis Class D — Continuous Hybrid Optimization (Parametric + CMA-ES / Adjoint Refinement):**
   * Parameterize a rich family of multi-stage valves with $8\text{--}20$ continuous parameters (stage non-uniformity, island angles, hood radii, throat widths, central pin offsets) and use `cma` (CMA-ES), Nelder-Mead, or multi-fidelity screening (coarse `FAST_CONFIG` search $\rightarrow$ fine `DEFAULT_CONFIG` selection) inside `design.py` within the 300s budget.

---

## 4. The Autoresearch Iteration Loop

For each iteration $k = 1, 2, \dots$:

1. **Inspect History & Flow Physics:**
   * Check `results.tsv` and visually inspect `artifacts/latest_eval.png` (or inspect velocity fields) to see *why* reverse flow is leaking through or where forward flow is choking.
2. **Formulate an Explicit Engineering Hypothesis:**
   * State what physical mechanism or continuous optimization algorithm you are introducing in `design.py`.
3. **Implement in `design.py` & Evaluate:**
   * Edit `design.py`.
   * Run:
     ```bash
     uv run python evaluate.py --tag run_XXX --description "Concise summary of hypothesis"
     ```
4. **Keep or Revert (Ratchet):**
   * If `score` strictly improves over the best score recorded so far (and `is_valid == True`), **commit** `design.py` (`git add design.py && git commit -m "run_XXX: score=..."`).
   * If `score` does not improve (or `design.py` crashes / exceeds the 300s budget), **revert** `design.py` to the last best commit (`git checkout -- design.py`) while keeping the row in `results.tsv` as a scientific record of what failed.
