# The Fancy Calculator and the Fluidic Diode

**Thinking out loud about how AI agents change engineering design—and what happens when they outsmart my physics.**

Mechanical engineers design things. You take an idea, sketch a geometry, analyze the forces or the flow, and change the shape until the part does what you want.

For generations, professors taught students how to do the calculations by hand, and later, how to run the finite element or computational fluid dynamics code. Now I have a much fancier calculator on my desk: AI coding agents that can write code, run simulations on cloud GPUs, inspect the results, and modify their own designs in a closed loop.

What do I do with this fancy calculator? How does it change the way engineers design? And more importantly, what should I be teaching my students?

I do not have a finished textbook answer. In this post, I am thinking out loud. I want to test things out, see what works and what breaks, and figure out the foundational patterns for setting up AI-driven design loops—and for developing the physical intuition required to steer them.

---

### Distilling the Autoresearch Pattern for Engineering Design

In an [earlier post](https://ebilionis.substack.com/p/why-scientific-autonomy-needs-a-harness), I looked at Andrej Karpathy's `autoresearch` pattern—where an AI agent autonomously modifies a machine learning script, trains a model for five minutes, checks the validation loss, and keeps only the changes that improve the score.

What happens if I repurpose that exact pattern for mechanical engineering design?

Strip away the machine learning details, and an autonomous engineering design loop needs only two things:

1. **A mutable design (`design.py`):** A programmatic representation of the geometry—along with whatever inner-loop optimization or evolutionary search the agent wants to write—that outputs a candidate part.
2. **An in-silico evaluator (`simulator.py` and `evaluate.py`):** A locked, read-only virtual wind tunnel that tests the part under physical laws and returns a single scalar score.

The loop is simple. The agent proposes a parameterization and search strategy in `design.py`, evaluates candidates in batches on a GPU (`jax.vmap`), inspects the flow fields, and commits the new code to `git` only if the scalar metric improves.

---

### The Test Problem: A Passive Fluidic Diode

To test this pattern, I picked a classic fluid mechanics puzzle: designing a **passive fluidic diode**.

In 1920, Nikola Tesla patented what he called a ["valvular conduit"](https://www.youtube.com/watch?v=sxvIQMnKhIg)—a fixed channel with no moving parts that lets fluid flow easily in the forward direction while choking flow in the reverse direction. Engineers measure its performance by its **diodicity** ($\mathrm{Di}$), the ratio of forward flow rate to reverse flow rate under the same pressure drop.

Why do engineers care about a valve with no moving parts? Because moving parts jam, wear out, and fatigue:
- **Lab-on-a-chip microfluidics:** Microscopic check valves for drug delivery and point-of-care diagnostics cannot use tiny mechanical springs or hinges without clogging.
- **Two-phase AI chip and EV battery cooling:** When liquid coolant boils inside microchannels on a high-power GPU cold plate, expanding vapor bubbles can shoot backward toward the inlet and cause dry-out and thermal runaway. Passive fluidic diodes act as one-way vapor check valves etched directly into the cold plate.
- **Nuclear reactors and harsh engines:** In passive reactor safety loops and pulse-detonation combustors, extreme heat or radiation destroys conventional seals.

---

### Take 1: The Naive Loop and the 383× Illusion

I gave the agent a 2D periodic channel driven by a fixed body force at a moderate Reynolds number ($\mathrm{Re} \approx 100$–$250$), a fast Lattice Boltzmann simulator running on an NVIDIA L4 GPU, and a starting geometry consisting of simple sawtooth nozzles (`run_000_baseline`).

That baseline barely worked: forward and reverse flows were nearly identical, giving $\mathrm{Di} = 1.02$.

![Take 1 Baseline (Di = 1.02)](take1_unsteady_vortex_hack/artifacts/run_000_baseline.png)

Then I let the loop run.

In the first seven iterations, the agent made rapid progress. Inspecting the velocity fields, it noticed that the sawtooth blocks leaked reverse flow around their outer edges. It sealed the outer bypass, carved a parabolic cup in the center of the channel to catch reverse flow like a parachute (`run_004`), and then placed a triangular wedge upstream to split the reverse jet directly into the cup (`run_007`). Diodicity jumped to **$\mathrm{Di} = 3.22$**.

![Take 1 Run 007 (Di = 3.22)](take1_unsteady_vortex_hack/artifacts/run_007.png)

Up to this point, the agent was doing solid fluid mechanics.

Then I instructed the agent to vectorize its inner search with `jax.vmap` on the cloud GPU and run 100+ autonomous iterations. Within forty iterations, the score exploded—first to $12$, then to $105$, and finally to **$\mathrm{Di} = 383.24$** (`run_049`).

![Take 1 Run 049 (Di = 383.24)](take1_unsteady_vortex_hack/artifacts/run_049.png)

A passive laminar diode at $\mathrm{Re} \approx 200$ cannot achieve a diodicity of $383$. Even thirty-stage cascades in the experimental literature struggle to pass $5$. What happened?

My simulator ran for 8,000 time steps and measured flow rate over the final 1,000 steps. Armed with high GPU throughput, the optimizer found that if it placed sharp staggered teeth across the channel, it could trigger a violent Kármán vortex street. It then fine-tuned the tooth spacing to the third decimal place so that, in the reverse direction, a transient vortex happened to slosh backward across the measurement window right between steps 7,000 and 8,000—driving the time-averaged denominator $Q_{\text{rev}}$ to $0.0005$.

Just as I wrote in [*When AI Agents Start Cheating*](https://ebilionis.substack.com/p/when-ai-agents-start-cheating), the agent did not break the rules. It found a bug in my question.

---

### Take 2: Fixing the Physics and Recalling Hydraulics

To get real engineering, I had to fix the physics oracle.

In an incompressible periodic channel at steady state, mass conservation demands that the flux through the inlet, the outlet, and the bulk of the channel must all be equal. In Take 2, I hardened `simulator.py`:
1. I required the flow to run longer and checked fluxes across three separate planes (inlet, outlet, and domain average).
2. I penalized temporal oscillation ($\sigma_Q$) in the denominator so that unsteady vortex sloshing could never manufacture a near-zero reverse flow.

I reset the agent back to the $\mathrm{Di} = 1.02$ baseline and ran another 105 iterations on the L4 GPU.

![Take 2 Progress Curve](artifacts/take2_progress.png)

This time, every single unsteady vortex trick failed the steady-state gate and was discarded. Forced to obey steady-state conservation of mass, the agent made 18 valid design improvements, climbing from $\mathrm{Di} = 1.02$ to **$\mathrm{Di} = 2.18$** (`run_105`)—which corresponds to a **$4.76\times$ pressure-drop asymmetry** ($\mathrm{Di}_{\Delta P} \approx \mathrm{Di}_Q^2 = 4.76$) in a single compact periodic cell.

![Take 2 Final Steady-State Design (Di = 2.18)](artifacts/take2_run_105.png)

Look closely at what the agent built in `run_105`. It combines three classic mechanisms from hydraulic engineering:
- **Borda's Re-Entrant Mouthpiece (1766):** Sharp overhanging lips on the wall teeth that force the reverse jet to detach strongly (a vena contracta), while streamlining the forward jet.
- **Concave-Catcher Topology (Lin et al., 2015):** A streamlined central island whose downstream face is hollowed into a semi-circular C-cup.
- **The Pelton Turbine Bucket (1880):** A tiny central cusp (`0.08` channel widths tall) inside the back of the C-cup that splits the stagnating reverse jet into two $180^\circ$ counter-rotating vortices, turning the reverse flow back into the oncoming stream.

Did the agent invent these mechanisms from first principles? I cannot claim that. Large language models have read fluid mechanics textbooks, turbine patents, and topology optimization papers during pretraining. What the agent is doing here is likely **recalling** structural motifs from its training set—explicitly naming "Borda lips" and "C-cups" in its code comments—and then using continuous GPU optimization (`jax.vmap`) to tune their exact coordinates until those pieces work together inside my specific channel.

---

### What Should I Teach My Students?

Running these two loops back-to-back clarified something important for me about how engineering is changing.

When the calculator gets this capable, the bottleneck in engineering design shifts **from solving the optimization problem to specifying the physics oracle**.

If you give an AI agent a naive simulation harness, it will use a cloud GPU to turn numerical truncation and transient sloshing into a $383\times$ mirage. I do not need students who can manually tweak spline coordinates for three weeks; the agent can search 5,000 geometries an hour. Instead, I need students who immediately ask:
- *Is mass actually conserved across this control volume?*
- *Is this a steady-state phenomenon or a windowing artifact?*
- *Why does a re-entrant lip or a Pelton splitter create irreversible form drag at $\mathrm{Re} = 200$?*

The fancy calculator does not make physical intuition obsolete. It makes physical intuition the only thing keeping the calculator honest.

---

### How I Set This Up (and How to Replicate It)

If you want to replicate these runs or adapt the harness to your own engineering problem, all the code, logs, and artifacts from both Take 1 and Take 2 are open-source in a new repository: **[PredictiveScienceLab/fluidic-diode-autoresearch](https://github.com/PredictiveScienceLab/fluidic-diode-autoresearch)**.

Here is the exact stack I used:
- **Interactive Coding Agent:** I used **Google Antigravity** on my laptop to write the 2D JAX Regularized Lattice Boltzmann solver (`simulator.py`), set up the evaluation harness (`evaluate.py`), provision the cloud GPU, and diagnose the Take 1 vortex loophole.
- **Outer-Loop Autonomous Engineer:** Inside the unattended loop (`run_autoresearch.py`), each iteration queried **Gemini 2.5 Pro** (`gemini-2.5-pro`) in multimodal mode (`--attach-image`). At every step, the model read the `results.tsv` log, the current `design.py` script, and the 3-panel streamline PNG image of the best design so far, and wrote a new `design.py`.
- **Compute Cloud & Inner-Loop GPU Search:** I ran both 100+ iteration campaigns on a single **Google Cloud `g2-standard-8` VM** equipped with one **NVIDIA L4 GPU (24 GB VRAM)**. Using `jax.vmap`, each 5-minute iteration of `design.py` screened hundreds of candidate geometries in parallel on the L4 GPU before submitting its best candidate to the high-resolution ($300 \times 100$) ground-truth check and `git` keep/revert ratchet.
