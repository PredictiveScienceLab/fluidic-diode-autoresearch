# Autoresearch Program: Passive Fluidic Diode Design & Continuous Optimization (Hardened Steady-State Physics Oracle)

You are an autonomous mechanical and thermo-fluids research engineer running a Karpathy-style **autoresearch** loop. Your goal is to discover a non-trivial 2D passive **fluidic diode** (a valvular conduit with zero moving parts) and an inner-loop **continuous optimization strategy** that maximizes the **true steady-state** mass-flux diodicity under a fixed pressure drop.

---

## 1. The Engineering Problem & Hardened Physics Oracle

A passive fluidic diode allows fluid to pass easily in the **forward direction ($+x$)** while strongly impeding flow in the **reverse direction ($-x$)** via steady inertial fluid mechanics: Coandă attachment, boundary-layer separation, return-channel jet collision, and standing vortex-chamber trapping.

### Read-Only Simulation Harness (`simulator.py` and `evaluate.py`)
* **Solver:** 2D D2Q9 Regularized Smagorinsky Lattice Boltzmann Method (RLBM) in JAX with regularized non-equilibrium pressure boundary conditions.
* **Domain:** Full channel is $N_x \times N_y = 300 \times 100$ lattice units. The left and right ports ($x \in [0, 25)$ and $x \in [275, 300)$) are open fluid buffers. The editable **design domain** is $(250, 100)$.
* **Boundary Conditions & Laminar-Inertial Regime:**
  * Top and bottom channel walls ($y = 0$ and $y = 99$) are no-slip bounce-back walls.
  * Fixed macroscopic pressure drop $\Delta P = c_s^2 \Delta \rho$ across $x = 0$ and $x = 299$ ($\Delta \rho = 0.07$, $\nu = 0.0055$, open-pipe $\mathrm{Re}_{\text{open}} \approx 3{,}100$, valve $\mathrm{Re} \approx 400\text{--}1{,}200$).
  * Forward run (`direction = +1.0`): high pressure at $x = 0$, low pressure at $x = 299$.
  * Reverse run (`direction = -1.0`): high pressure at $x = 299$, low pressure at $x = 0$.
* **Conservative Steady-State Mass-Flux Objective (`score`):**
  Over the final $3{,}000$ steps of a $9{,}000$-step simulation, the solver measures cross-sectional mass flux $\dot{m} = \text{direction} \cdot \rho u_x$ across three control sections—inlet buffer ($\bar{Q}_{\text{in}}$), outlet buffer ($\bar{Q}_{\text{out}}$), and bulk channel ($\bar{Q}_{\text{bulk}}$)—along with the temporal standard deviation $\sigma_Q$:
  $$Q_{\text{fwd}} = \min(\bar{Q}_{\text{fwd,in}}, \bar{Q}_{\text{fwd,out}}, \bar{Q}_{\text{fwd,bulk}}) - 0.5\,\sigma_{Q_{\text{fwd}}}, \qquad Q_{\text{rev}} = \max(\bar{Q}_{\text{rev,in}}, \bar{Q}_{\text{rev,out}}, \bar{Q}_{\text{rev,bulk}}) + \sigma_{Q_{\text{rev}}}$$
  $$\text{Steady Diodicity } \mathrm{Di} = \frac{Q_{\text{fwd}}}{Q_{\text{rev}}}$$
  *(Because $Q_{\text{rev}}$ is bounded below by the maximum across all three sections plus $\sigma_{Q_{\text{rev}}}$, unsteady vortex sloshing is penalized rather than rewarded—only **true steady-state hydrodynamic resistance** increases $\mathrm{Di}$!)*
* **Feasibility Constraints (enforced smoothly in `score`):**
  1. **Minimum Forward Conductance:** $Q_{\text{fwd}} / Q_{\text{open}} \ge 0.15$ (where $Q_{\text{open}} = 0.1703$ is the empty-pipe mass flux). A plugged pipe is not a valve!
  2. **Maximum Solid Fraction:** At most $60\%$ of the $(250, 100)$ design region may be solid ($\text{mean}(\text{mask}) \le 0.60$).
  3. **Compute Budget:** `design.generate_design()` must complete within **5.0 minutes (300 seconds)**.

---

## 2. Rules of Engagement

1. **ONLY EDIT `design.py`:**
   * You may **never** modify `simulator.py` or `evaluate.py`.
   * `design.py` must define `generate_design() -> np.ndarray` returning a 2D array of shape `(250, 100)` (values $\ge 0.5$ become solid wall, $< 0.5$ become open fluid).
2. **USE THE COMPUTE BUDGET & GPU `jax.vmap` BATCHING WISELY:**
   * Inside `design.py`, you can import `FAST_CONFIG` (already Reynolds-matched at $180 \times 60, \nu = 0.0033, 4{,}500\text{ steps}$), `DEFAULT_CONFIG` ($300 \times 100, \nu = 0.0055, 9{,}000\text{ steps}$), `SimConfig`, `build_full_domain_mask`, `run_directional_lbm`, and **`evaluate_batch_fast`** from `simulator.py`.
   * **Exploit `jax.vmap` on the NVIDIA L4 GPU (Exact API Signature):**
     * `scores, q_fwds, q_revs = evaluate_batch_fast(design_masks, FAST_CONFIG)` where `design_masks` has shape `(B, 150, 60)` (`FAST_CONFIG.nx - 2*FAST_CONFIG.pad_x = 150, FAST_CONFIG.ny = 60`) returns a **3-tuple of 1D JAX arrays** `(scores, q_fwds, q_revs)` of shape `(B,)` (NOT a dict!).
     * On `DEFAULT_CONFIG`, `scores, q_fwds, q_revs = evaluate_batch_fast(design_masks_fine, DEFAULT_CONFIG)` takes `design_masks_fine` of shape `(B, 250, 100)` (`DEFAULT_CONFIG.nx - 2*DEFAULT_CONFIG.pad_x = 250, DEFAULT_CONFIG.ny = 100`).
     * Always decorate your single-mask renderer with `@functools.partial(jax.jit, static_argnames=("nx_design", "ny"))` and vectorize with `jax.vmap(lambda p: render_fn(p, nx_design, ny))` so `nx_design` and `ny` are concrete integers!
   * **Multi-Fidelity Verification & Speed Target (~25–55 seconds per run):** Always validate the top candidates (and perform 2 to 4 rounds of evolutionary / coordinate refinement) directly on **`DEFAULT_CONFIG` ($300 \times 100$, $9{,}000$ steps)** using `evaluate_batch_fast(top_k_masks_fine, DEFAULT_CONFIG)` inside `design.py` before returning the true high-resolution winner! Aim for `generate_design()` to take **25–55 seconds** on the L4 GPU.
   * **Strict Monotonicity Invariant (Never Lose Ground):**
     * Always store the exact 8-decimal parameter vector of the current best design in `design.py`, never round it to 4 decimals, and never alter or abbreviate existing geometry formulas without ensuring neutral defaults (`p_new = 0.0`) reproduce the exact previous best mask as candidate 0 in your `DEFAULT_CONFIG` batch!

---

## 3. Meta-Level Research Strategy (How to Think Like a Fluidic Engineer)

Notice why the baseline sawtooth nozzle (`run_000_baseline`) only achieves $\mathrm{Di} \approx 1.02$:
* In a symmetric sawtooth cascade, a central core jet shoots straight down the centerline ($y \in [35, 65]$) in **both** $+x$ and $-x$ directions, skimming right past the dead-water recirculation pockets in the wall teeth.

To achieve high **steady-state** diodicity ($\mathrm{Di} > 1.40 \to 2.50+$), you must force the **reverse flow** to undergo strong steady momentum reversal, jet collision, and vena-contracta choking while keeping a streamlined, low-loss path for the **forward flow**:

1. **Hypothesis Class A — Multi-Stage Nozzle + Sealed Concave U-Cup Collector + Upstream Splitter Sting:**
   * Place converging wall nozzles that guide $+x$ flow smoothly around a streamlined teardrop nose, while in $-x$ flow a deep sealed-nose concave U-cup on the centerline catches the reverse core jet, splits it with an upstream central sting, and turns it $180^\circ$ outward to collide with the incoming wall shear layers.
2. **Hypothesis Class B — Double-Loop Tesla Valve with Return-Jet Impingement & Borda Lips:**
   * Place angled airfoil/teardrop islands and outer turnaround C-hoods or re-entrant Borda lips on the channel walls so that reverse flow is peeled into side return channels, turned $180^\circ$, and fired transversely into the throat to pinch the reverse vena contracta.
3. **Hypothesis Class C — Alternating/Meandering Coandă Cascade (Asymmetric Serpentine Diode):**
   * Break spanwise symmetry ($y \leftrightarrow -y$) by staggering upper-wall and lower-wall deflectors and off-center cup collectors so reverse flow must make sharp $180^\circ$ S-turns around re-entrant cusps while forward flow follows a smooth meandering Coandă path.
4. **Hypothesis Class D — Continuous Hybrid Optimization (Multi-Fidelity `FAST_CONFIG` Screen $\rightarrow$ `DEFAULT_CONFIG` Evolution / Coordinate Line Search):**
   * Combine global exploration on `FAST_CONFIG` ($B = 64$) with multi-round elitist evolutionary / coordinate-wise refinement directly on `DEFAULT_CONFIG` ($B = 8\text{--}12$ per round), always seeding candidate 0 with the exact unperturbed parent elite.
