# Legacy Results Errors Log
- status: active
- type: log
- id: rl_signaling.legacy_errors_log
- description: Per-figure / per-CSV honesty audit. For every saved artifact in results/, records whether the underlying data is correct, mislabeled, unreproducible, or contaminated by a code bug catalogued in LEGACY_BUGS_LOG.md. Read this before citing any figure from this repository.
- label: [agent, audit]
- injection: excluded
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

This log is the **results-level** companion to [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md). The bugs log catalogues *code* defects; this log answers the question:

> *Given the saved figures in [results/](results/), which ones can I trust as written, which are biased, which are mislabeled, and which are unreproducible?*

The audit was performed during the DEBUGGING_PLAN session on 2026-05-08, after the seven-phase refactor and the Phase 1–5 model-vs-implementation audit. The kernel-level math (entropy, MI, NMI, Q-update, TD-update, costly-signaling arithmetic, Roth–Erev sampling) was verified exact in Phase 4 against independent reference implementations under [analytics/scripts/](analytics/scripts/), so the errors below are all **structural / experimental**, never numerical.

## Severity scale

| Tag | Meaning |
|---|---|
| **WRONG** | Numerically incorrect under its label. Do not cite without re-running. |
| **BIASED-METRIC** | Values are honestly computed but the *summary statistic* (slice indices, sample window, etc.) covers a different range than the label implies. The qualitative direction of effects holds; magnitude estimates are biased. |
| **MISLABELED** | Correct content saved under a wrong filename or wrong axis label. The data inside is fine; the wrapper is misleading. |
| **UNREPRODUCIBLE** | The data appears correct but the producing notebook code has drifted such that re-running today does not regenerate it. Trust depends on trusting whoever ran the prior version. |
| **CLEAN** | No error. |

## Quick verdict by experiment family

| Experiment | Roth–Erev | Q-learning | TD-learning |
|---|---|---|---|
| Canonical (2-feature, fixed action sizes) | CLEAN | CLEAN | BIASED-METRIC |
| Costly signaling | UNREPRODUCIBLE (per-agent vs shared cost code drift) | CLEAN | (not run) |
| Complex / general games | UNREPRODUCIBLE | UNREPRODUCIBLE | UNREPRODUCIBLE + BIASED-METRIC + MISLABELED |
| Initialization study | (not run, see Bug 5) | WRONG | (not run) |
| Hyperparameter optimization | (not run) | UNREPRODUCIBLE | UNREPRODUCIBLE + BIASED-METRIC |

The rest of this file expands each cell. The detailed file-by-file inventory and column-level audit are in the [Detailed inventory](#detailed-inventory) section below.

---

## Error 1 — TD-learning canonical figures: BIASED-METRIC

**Affected files:**
- `results/td_learning_results_canonical.csv` (the `Agent_X_Initial_NMI` and `Agent_X_NMI` columns)
- `results/TD-learning_canonical_Agent_0_NMI.png`
- `results/TD-learning_canonical_Agent_0_avg_reward.png` (only the NMI-derived overlay; the reward histogram itself is fine)
- `results/TD-learning_canonical_Agent_0_final_reward.png` (reward part is clean; if any NMI overlay is present, that part is biased)
- `results/TD-learning_canonical_regression_signals_True_fullinfo_False.png`
- `results/TD-learning_canonical_regression_signals_True_fullinfo_True.png`

**Producer:** [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb), TD Agent canonical block. Drives `TempNetMultiAgentEnv` + `temp_simulation_function`.

**Underlying bug:** [Bug 2](LEGACY_BUGS_LOG.md#bug-2--tempnetmultiagentenvget_actions-runs-the-nmi-inner-loop-once-per-outer-loop-iteration) — the legacy `TempNetMultiAgentEnv.get_actions` runs the per-agent NMI inner loop **once per outer-loop iteration**, doubling the length of `signal_information_history[i]` to `2T` for the 2-agent setup.

**Why this matters at the result level.** The notebook aggregates the trajectory into CSV columns using positional slices:

```python
Agent_X_Initial_NMI = np.mean(info_hist[:10])    # first 10 entries
Agent_X_NMI         = np.mean(info_hist[-100:])  # last 100 entries
```

Under the 2× inflation:

| Slice | Intended episodes | Actual episodes |
|---|---|---|
| `info_hist[:10]` | first 10 | first **5** |
| `info_hist[-100:]` | last 100 | last **50** |

So both NMI summary statistics in `td_learning_results_canonical.csv` are averages over **half** the temporal window the column name implies. The values are real cumulative-NMI estimates; the window is just smaller than the label suggests. Reward columns (`Agent_X_avg_reward`, `Agent_X_final_reward`) are **unaffected** because they read from `rewards_history`, which is not nested-looped.

**Magnitude in practice.** The cumulative-NMI estimator is smooth late in training (the urns / Q-tables drift slowly in the last several hundred episodes), so the difference between "last 50" and "last 100" is typically in the second decimal. But it is non-zero, and the estimator's standard deviation under "last 50" is $\sqrt{2}\times$ larger than under "last 100" — comparing TD-learning's NMI distributional spread against Q-learning's or Roth–Erev's compares summary statistics with **different effective sample sizes**.

**Verdict.** The qualitative direction of TD-learning effects (does signaling help? does NMI rise during training?) is preserved. Quantitative claims based on these figures should be re-derived with the bug fixed; differences are likely small but real.

**Fix recipe.** Either patch [rl_signaling/env.py:733-760](rl_signaling/env.py#L733-L760) to lift the NMI inner loop out of the per-agent outer loop, or migrate the TD experiments to the canonical `MultiAgentEnv` + `run_simulation` API (which never had the bug). Then re-run the TD canonical block of `Run_Simulations.ipynb` and regenerate the affected figures.

---

## Error 2 — Complex experiment figures: UNREPRODUCIBLE (and TD blocks: + BIASED-METRIC)

**Affected files (12 PNGs + 3 CSVs):**

| Figure | Status |
|---|---|
| `results/Roth-Erev_complex_randomized_Agent_0_NMI.png` | UNREPRODUCIBLE |
| `results/Roth-Erev_complex_randomized_Agent_0_avg_reward.png` | UNREPRODUCIBLE |
| `results/Roth-Erev_complex_randomized_Agent_0_final_reward.png` | UNREPRODUCIBLE |
| `results/Roth-Erev_complex_randomized_regression_signals_*.png` (2 files) | UNREPRODUCIBLE |
| `results/Q-learning_complex_randomized_Agent_0_NMI.png` | UNREPRODUCIBLE |
| `results/Q-learning_complex_randomized_Agent_0_avg_reward.png` | UNREPRODUCIBLE |
| `results/Q-learning_complex_randomized_Agent_0_final_reward.png` | UNREPRODUCIBLE |
| `results/Q-learning_complex_randomized_regression_signals_*.png` (2 files) | UNREPRODUCIBLE + see Error 3 below (Bug 8) |
| `results/TD-learning_complex_randomized_Agent_0_NMI.png` | UNREPRODUCIBLE + BIASED-METRIC |
| `results/TD-learning_complex_randomized_Agent_0_avg_reward.png` | UNREPRODUCIBLE |
| `results/TD-learning_complex_randomized_Agent_0_final_reward.png` | UNREPRODUCIBLE |
| (no `TD-learning_complex_randomized_regression_*.png` files exist — see Error 3) | — |

CSVs:
- `results/urnagent_results_complex_randomized.csv`
- `results/qlearning_results_complex_randomized.csv`
- `results/td_learning_results_complex_randomized.csv`

**Producer:** **No current producer.** The matching block in [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb) writes `*_complex.csv` (without the `_randomized` suffix) using fixed action sizes `n_signaling_actions=4, n_final_actions=8`, while the saved `*_complex_randomized.csv` files reflect an earlier code variant where `n_signaling_actions = np.random.randint(2, 10)` and `n_final_actions = np.random.randint(2, 10)` were drawn per iteration.

**Underlying bug:** [Bug 6](LEGACY_BUGS_LOG.md#bug-6--run_simulationsipynb-writes-_complexcsv-but-plotting_resultsipynb-reads-_complex_randomizedcsv).

**What this means.** The data in the saved `_complex_randomized` CSVs was produced honestly at some point, but the producing code is no longer in the repository. Re-running `Run_Simulations.ipynb` today writes `*_complex.csv` (which `plotting_results.ipynb` does not consume), so the chain producer → CSV → figure is broken: figures are produced from stale CSVs that no current notebook regenerates.

**For the TD block specifically.** Whatever code variant produced `td_learning_results_complex_randomized.csv` *also* used `TempNetMultiAgentEnv` (the only TD env in the codebase), so it carries the same Bug 2 inflation as Error 1 — the TD complex figures are both UNREPRODUCIBLE *and* BIASED-METRIC.

**Verdict.** Trust depends on whether you trust the prior session's code. Independent reproduction is currently not possible. For papers / writeups, either re-run with a chosen direction (Phase 5 Bug 6 Option A or B) or annotate the figures as "produced by an earlier code variant" and document the variant's behavior.

**Fix recipe.** Phase 5 of [DEBUGGING_PLAN.md](DEBUGGING_PLAN.md) recommends Option A: reintroduce the per-iteration `np.random.randint(2, 10)` action-size draw in `Run_Simulations.ipynb`'s complex blocks and rename outputs to `*_complex_randomized.csv`. After fixing, regenerate every `_complex_randomized` figure.

---

## Error 3 — Complex regression PNGs: MISLABELED

**Affected files:**
- `results/Q-learning_complex_randomized_regression_signals_True_fullinfo_False.png` — **contains TD-learning data, not Q-learning data.**
- `results/Q-learning_complex_randomized_regression_signals_True_fullinfo_True.png` — same.
- `results/TD-learning_complex_randomized_regression_signals_*.png` — **do not exist** in the repository.

**Producer:** [notebooks/plotting_results.ipynb](notebooks/plotting_results.ipynb), final code cell.

**Underlying bug:** [Bug 8](LEGACY_BUGS_LOG.md#bug-8--plotting_resultsipynb-final-cell-uses-q-learning-filename-prefix-for-td-learning-data) — the final cell calls

```python
plot_regression(td_learning_complex, ..., filename_prefix='Q-learning_complex_randomized')
```

passing the TD-learning DataFrame but the Q-learning filename prefix. So the saved files have Q-learning names but TD-learning content. The cell that should have produced `TD-learning_complex_randomized_regression_*.png` ran but saved under the Q name; the original Q-learning regression cell (earlier in the notebook) had its output overwritten when the final cell ran.

**Compounding with Error 2.** Because the underlying CSV (`td_learning_results_complex_randomized.csv`) is itself unreproducible (Error 2) and biased (Bug 2 inflation), the misnamed Q-learning regression PNGs are TRIPLY problematic:
1. Wrong filename — TD content under Q name.
2. Underlying CSV from a code variant no longer in the repo.
3. NMI summary statistics in the underlying CSV reflect half the intended temporal window.

**Verdict.** If any writeup cites a "Q-learning regression in the complex / general-games regime" using one of these files, the figure shown is **TD-learning** regression. Re-run after fixing Bug 6 + Bug 8.

**Fix recipe.** One-line fix in the final cell: change `filename_prefix='Q-learning_complex_randomized'` → `filename_prefix='TD-learning_complex_randomized'`. Then re-run `plotting_results.ipynb` from the Q-learning General Games section onward (so the Q-learning regression PNG is regenerated with Q content first, then the TD regression PNG is created under its correct name).

---

## Error 4 — Initialization study figures: WRONG

**Affected files:**
- `results/initializations_rewards.png`
- `results/initializations_nmi.png`

**Producer:** [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb).

**Underlying bug:** [Bug 5](LEGACY_BUGS_LOG.md#bug-5--initializations_testipynb-overwrites-envagents-silently-dropping-the-initializetrue-state) — the experimental loop constructs `NetMultiAgentEnv(..., initialize=True, initialization_weights=init_weights)` (which honors the init request) but immediately replaces `env.agents` with fresh `QLearningAgent(...)` instances using the default `initialize=False`. So every iteration of the loop runs the **same** `initialize=False` configuration; the four labeled curves differ only in run-to-run randomness.

**Why this is WRONG, not BIASED-METRIC.** The figures' labels claim "init weights = [1,0]", "[1,1]", "[5,1]", "[100,1]" — and assert that the curves correspond to those four initialization regimes. They do not. The data IS real (each curve is an honest ε-greedy QLearning run), but the *labels are lies* about what configuration produced each curve.

If the figure is read as "four runs of the same configuration," the data is honest. If it's read at face value (different inits → different curves), the figure asserts something false.

**Compounding nuance.** The notebook's section header reads "# Urn Agent" but the code constructs `QLearningAgent`. `UrnAgent` is imported but never used. So even after Bug 5 is fixed, the section header remains a separate misleading claim. The figures are framed as testing UrnAgent's initialization behavior; they actually test QLearningAgent's (or, before the fix, neither).

**Verdict.** Do not cite. The hypothesis the figures appear to test ("does initialization strength accelerate convergence?") is unanswered by the saved data.

**Fix recipe.** Phase 5 Bug 5 Option B: migrate `Initializations_test.ipynb` to `MultiAgentEnv` + `agent_kwargs`, splitting into a UrnAgent block and a QLearningAgent block, with the section header and the agent type matching. Re-run.

---

## Error 5 — Costly-signaling figures: split verdict

The costly experiment file family contains two CSVs and two figure sets, with **different verdicts** for Roth-Erev versus Q-learning. They are split below.

### Error 5a — Roth-Erev costly figures: UNREPRODUCIBLE (cost-protocol drift)

**Affected files:**
- `results/Roth-Erev_canonical_costly_signal_Agent_0_NMI.png`
- `results/Roth-Erev_canonical_costly_signal_Agent_0_avg_reward.png`
- `results/Roth-Erev_canonical_costly_signal_Agent_0_final_reward.png`
- `results/q_costly_vs_reward.png`, `results/q_costs_vs_nmi.png`, `results/q_learning_costly_single_run*.png` (the Q-prefixed costly visualizations consume this same Roth-Erev CSV)

**Producer:** [notebooks/Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb), UrnAgent block.

**Source CSV:** `results/urnagent_results_canonical_costly_signal.csv` (the file `plotting_results.ipynb` actually reads — 1000 rows).

**Discovery.** This CSV's first column shows `Signal_Cost_A0 != Signal_Cost_A1` per row (e.g. row 0: `0.246, 0.0006`; row 1: `0.331, 0.439`) — the per-agent costs are drawn **independently**. The current notebook code at [notebooks/Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb) draws **shared** costs:

```python
rdn = np.random.uniform(0.0, 0.5)
signal_cost = [rdn, rdn]
```

Searching git history (`git log --all -p notebooks/Final_Costly_Signaling_Run_Simulations.ipynb`) finds **only** the shared-cost protocol — never an independent-cost variant. So the saved 1000-row CSV was produced by a code variant that has never been committed to the repository.

There is also a sibling file `results/urnagent_results_canonical_costly_signal (1).csv` (10000 rows, with `Signal_Cost_A0 == Signal_Cost_A1` per row) — this matches the current shared-cost code, but `plotting_results.ipynb` does **not** read it. So the figures show data from never-committed code; the data that matches the current code is sitting in an orphan file.

**Why this is UNREPRODUCIBLE.** The figures may be honest representations of a real costly-signaling experiment, but:
1. The cost protocol used to produce them (independent per-agent draws) is not in any version of the notebook in git history.
2. Re-running today produces 10× more iterations under a different cost protocol (shared draws).
3. Trust depends on trusting whoever ran the never-committed variant.

**Verdict.** Do not cite as "the costly Roth-Erev figures from the current `Final_Costly_Signaling_Run_Simulations.ipynb` notebook." If the independent-cost protocol is the desired experimental design, the notebook needs to be updated to match (and the saved figures kept); if the shared-cost protocol is the desired design, the figures need re-generation from the `(1)` sibling CSV (or a fresh re-run).

**Additional theoretical concern (independent of the protocol drift).** Costly signaling is **mathematically ill-defined** under the project's Roth-Erev rule, regardless of which cost protocol is used. The classical Roth-Erev urn (Roth & Erev 1995; Erev & Roth 1998) was specified for non-negative integer reinforcement: each play of action $a$ adds the reward to its urn count, and the probability $\mathbb{P}[a] = \mathrm{urn}[a]/\sum_{a'} \mathrm{urn}[a']$ rests on the urn-as-ball-counter interpretation. Costly signaling violates both assumptions:

1. **Negative rewards.** Whenever `game_reward < cost` and the agent emits a non-null signal, the net reward $r = G_i(\mathbf{v}, \alpha) - c_i$ is negative. The project's clamp `urn[a] = max(0, urn[a] + r)` handles this defensively but introduces an **absorbing barrier**: once `urn[a] = 0`, the action is never sampled again, so it cannot be re-updated to recover. A run of negative-reward episodes can permanently kill an action that would have recovered under Q-learning (where $Q$ can dip negative and exploration re-samples) or TD-learning (whose count-based learning rate eventually dominates).
2. **Real-valued rewards.** Even ignoring sign, the costly net reward is a real number (`0.75`, `-0.25`, etc.). Real-valued urns still admit the probability formula, but the urn-as-counter metaphor breaks — it is no longer Roth-Erev's classical model; it is a normalized weight scheme over accumulated reward.

Under `create_random_canonical_game(n=1, m=0)` with `c ∈ (0, 0.5)`, **both** failure modes fire: net rewards are real-valued and can be negative.

**Recommendation.** The costly-signaling experiment should not be reported for `UrnAgent`. Q-learning and TD-learning handle the necessary reward range natively. The Roth-Erev costly figures are best **retired** (not re-generated under either protocol), with a note in any writeup that the costly extension is not applied to UrnAgent for theoretical reasons. If the experiment is kept for comparison purposes, it should be labeled "Roth-Erev with non-negativity-clamped costly extension" so the deviation from the canonical Roth-Erev rule is visible to the reader.

**Note on cost-flow arithmetic.** Independent of the protocol question, the cost-flow arithmetic itself — `r = G_i(v, alpha) - c_i * 1[sigma != null]` — was verified exact in Phase 4 across eight cases under [analytics/scripts/verify_costly_signaling.py](analytics/scripts/verify_costly_signaling.py). The kernel is correct; the protocol selection is what differs.

**Fix recipe.** Decide which cost-draw protocol matches the user's intent. Option A: edit the notebook to draw `signal_cost = [np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)]` (independent per-agent), re-run, regenerate the figures from the new CSV. Option B: keep current shared-cost code, point `plotting_results.ipynb` at `urnagent_results_canonical_costly_signal (1).csv` (or rename it to drop the suffix), regenerate the figures.

### Error 5b — Q-learning costly figures: CLEAN

**Affected files (none — listed for completeness):**
- `results/QLearning_canonical_costly_signal_Agent_0_NMI.png`
- `results/QLearning_canonical_costly_signal_Agent_0_avg_reward.png`
- `results/QLearning_canonical_costly_signal_Agent_0_final_reward.png`

**Producer:** [notebooks/Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb), QLearningAgent block (currently commented out / `simulate=False`-gated, but was active when `qlearning_results_canonical_costly_signal.csv` was produced).

**Source CSV:** `results/qlearning_results_canonical_costly_signal.csv` (10000 rows, `Signal_Cost_A0 == Signal_Cost_A1` per row — shared-cost protocol matching current code).

**Verdict.** **Clean.** The CSV uses the same shared-cost protocol as the current code (rows show identical per-agent costs); QLearningAgent uses `NetMultiAgentEnv` + `simulation_function`, which is not affected by Bug 2. Cost-flow arithmetic is verified exact (Phase 4).

**Caveat.** The reproducibility caveat noted in the [Phase 3 cross-cutting findings](DEBUGGING_PLAN.md#cross-cutting-findings) applies to this and all other costly figures: `signal_cost` and `game_dicts` are constructed before the per-iteration seed reset, so individual rows of the saved CSVs are not row-reproducible from `iteration` alone. The population-level statistics (means, KDEs, regressions) are unaffected; if you re-run, you get a sample from the same distribution, not an identical row.

---

## Error 6 — Roth-Erev / Q-learning canonical (non-costly): CLEAN

**Affected files (none — listed for completeness):**
- `results/Roth-Erev_canonical_Agent_0_NMI.png`
- `results/Roth-Erev_canonical_Agent_0_avg_reward.png`
- `results/Roth-Erev_canonical_Agent_0_final_reward.png`
- `results/Roth-Erev_canonical_regression_signals_*.png` (2 files)
- `results/Q-learning_canonical_Agent_0_NMI.png`
- `results/Q-learning_canonical_Agent_0_avg_reward.png`
- `results/Q-learning_canonical_Agent_0_final_reward.png`
- `results/Q-learning_canonical_regression_signals_*.png` (2 files)

**Producer:** [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb), Roth-Erev canonical and Q-learning canonical blocks.

**Verdict.** **Clean.** Both blocks use `NetMultiAgentEnv` + `simulation_function`, which is not affected by Bug 2 (only `TempNetMultiAgentEnv` has the inflated NMI history). The kernel-level identities (`UrnAgent` Roth-Erev sampling, `QLearningAgent` Q-update closed form, NMI computation) are exact at atol = 1e-12 against [analytics/scripts/verify_q_learning.py](analytics/scripts/verify_q_learning.py) and [analytics/scripts/verify_urn_convergence.py](analytics/scripts/verify_urn_convergence.py).

The same `signal_cost` / `game_dicts` row-reproducibility caveat as Error 5 applies, but it does not affect the validity of the saved figures.

---

## Error 7 — Hyperparameter optimization figures: UNREPRODUCIBLE (and TD section: + BIASED-METRIC)

**Affected files:**
- `results/q_opt_canonical.png`, `results/q_opt_games.png`
- `results/td_opt_canonical.png`, `results/td_opt_games.png`

**Producer:** [notebooks/Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb).

**Underlying bug:** [Bug 7](LEGACY_BUGS_LOG.md#bug-7--parameter_optimization_wchoicesipynb-is-missing-imports-for-several-names-it-uses) — the imports cell omits `Categorical, Optimizer, Real, Integer` (from `skopt` / `skopt.space`), `Parallel, delayed` (from `joblib`), `multiprocessing`, and `datetime`. Restart-and-Run-All raises `NameError` immediately at the first `param_ranges = {... Categorical([...]) ...}` cell.

**Why this is UNREPRODUCIBLE not WRONG.** The saved Pareto-frontier figures are from a session where the user had imports loaded interactively. The data is honest research output. It is not regenerable from a fresh kernel without manual import patches.

**TD section also BIASED-METRIC.** The TD optimization function uses `TempNetMultiAgentEnv` and `temp_simulation_function`, so Bug 2 applies: the optimization objective `np.mean(agent_nmi[-n_episodes // 10:])` is computed over half the intended episode window. The Bayesian search is robust to a noisy objective (it explores the parameter space multiple times per region), so the chosen TD hyperparameters are likely close to optimal, but they were optimized against a slightly-noisier-than-intended metric. This indirectly affects every downstream TD-learning figure that uses these hyperparameters as inputs.

**Verdict.** The Pareto-frontier figures themselves are research records; they are not consumed by `plotting_results.ipynb`. The downstream TD figures inherit the bias. Magnitude is small (Bayesian search noise robustness + smooth NMI estimator at convergence).

**Fix recipe.** Add the missing imports, add `scikit-optimize` to `pyproject.toml`'s dev extras, then optionally re-run the TD optimization with Bug 2 fixed and update `Run_Simulations.ipynb` if the chosen hyperparameters shift materially.

---

## Detailed inventory

This section is the file-by-file, column-by-column expansion of the higher-level Errors 1–7 above. Read it when you need to verify the status of a specific saved artifact rather than a whole experiment family.

### A. Complete file inventory of `results/`

54 files total: 41 PNGs and 13 CSVs (one is a Mac-Finder duplicate). Status is the verdict for that specific file; "→" points at the section above that explains it.

#### A.1 — PNG files (41)

| Filename | Status | Producer notebook | Source CSV | Bug(s) | → Error |
|---|---|---|---|---|---|
| `Roth-Erev_canonical_Agent_0_NMI.png` | CLEAN | Run_Simulations.ipynb | `urnagent_results_canonical.csv` | (none) | Error 6 |
| `Roth-Erev_canonical_Agent_0_avg_reward.png` | CLEAN | Run_Simulations.ipynb | `urnagent_results_canonical.csv` | (none) | Error 6 |
| `Roth-Erev_canonical_Agent_0_final_reward.png` | CLEAN | Run_Simulations.ipynb | `urnagent_results_canonical.csv` | (none) | Error 6 |
| `Roth-Erev_canonical_regression_signals_True_fullinfo_False.png` | CLEAN | Run_Simulations.ipynb | `urnagent_results_canonical.csv` | (none) | Error 6 |
| `Roth-Erev_canonical_regression_signals_True_fullinfo_True.png` | CLEAN | Run_Simulations.ipynb | `urnagent_results_canonical.csv` | (none) | Error 6 |
| `Q-learning_canonical_Agent_0_NMI.png` | CLEAN | Run_Simulations.ipynb | `qlearning_results_canonical.csv` | (none) | Error 6 |
| `Q-learning_canonical_Agent_0_avg_reward.png` | CLEAN | Run_Simulations.ipynb | `qlearning_results_canonical.csv` | (none) | Error 6 |
| `Q-learning_canonical_Agent_0_final_reward.png` | CLEAN | Run_Simulations.ipynb | `qlearning_results_canonical.csv` | (none) | Error 6 |
| `Q-learning_canonical_regression_signals_True_fullinfo_False.png` | CLEAN | Run_Simulations.ipynb | `qlearning_results_canonical.csv` | (none) | Error 6 |
| `Q-learning_canonical_regression_signals_True_fullinfo_True.png` | CLEAN | Run_Simulations.ipynb | `qlearning_results_canonical.csv` | (none) | Error 6 |
| `TD-learning_canonical_Agent_0_NMI.png` | BIASED-METRIC | Run_Simulations.ipynb | `td_learning_results_canonical.csv` | Bug 2 | Error 1 |
| `TD-learning_canonical_Agent_0_avg_reward.png` | CLEAN | Run_Simulations.ipynb | `td_learning_results_canonical.csv` | (rewards unaffected) | Error 1 |
| `TD-learning_canonical_Agent_0_final_reward.png` | CLEAN | Run_Simulations.ipynb | `td_learning_results_canonical.csv` | (rewards unaffected) | Error 1 |
| `TD-learning_canonical_regression_signals_True_fullinfo_False.png` | BIASED-METRIC | Run_Simulations.ipynb | `td_learning_results_canonical.csv` | Bug 2 (NMI axis) | Error 1 |
| `TD-learning_canonical_regression_signals_True_fullinfo_True.png` | BIASED-METRIC | Run_Simulations.ipynb | `td_learning_results_canonical.csv` | Bug 2 (NMI axis) | Error 1 |
| `Roth-Erev_complex_randomized_Agent_0_NMI.png` | UNREPRODUCIBLE | (no current producer) | `urnagent_results_complex_randomized.csv` | Bug 6 | Error 2 |
| `Roth-Erev_complex_randomized_Agent_0_avg_reward.png` | UNREPRODUCIBLE | (no current producer) | `urnagent_results_complex_randomized.csv` | Bug 6 | Error 2 |
| `Roth-Erev_complex_randomized_Agent_0_final_reward.png` | UNREPRODUCIBLE | (no current producer) | `urnagent_results_complex_randomized.csv` | Bug 6 | Error 2 |
| `Roth-Erev_complex_randomized_regression_signals_True_fullinfo_False.png` | UNREPRODUCIBLE | (no current producer) | `urnagent_results_complex_randomized.csv` | Bug 6 | Error 2 |
| `Roth-Erev_complex_randomized_regression_signals_True_fullinfo_True.png` | UNREPRODUCIBLE | (no current producer) | `urnagent_results_complex_randomized.csv` | Bug 6 | Error 2 |
| `Q-learning_complex_randomized_Agent_0_NMI.png` | UNREPRODUCIBLE | (no current producer) | `qlearning_results_complex_randomized.csv` | Bug 6 | Error 2 |
| `Q-learning_complex_randomized_Agent_0_avg_reward.png` | UNREPRODUCIBLE | (no current producer) | `qlearning_results_complex_randomized.csv` | Bug 6 | Error 2 |
| `Q-learning_complex_randomized_Agent_0_final_reward.png` | UNREPRODUCIBLE | (no current producer) | `qlearning_results_complex_randomized.csv` | Bug 6 | Error 2 |
| `Q-learning_complex_randomized_regression_signals_True_fullinfo_False.png` | MISLABELED + UNREPRODUCIBLE + BIASED-METRIC | (no current producer) | `td_learning_results_complex_randomized.csv` (TD data under Q name) | Bugs 2, 6, 8 | Errors 2, 3 |
| `Q-learning_complex_randomized_regression_signals_True_fullinfo_True.png` | MISLABELED + UNREPRODUCIBLE + BIASED-METRIC | (no current producer) | `td_learning_results_complex_randomized.csv` | Bugs 2, 6, 8 | Errors 2, 3 |
| `TD-learning_complex_randomized_Agent_0_NMI.png` | UNREPRODUCIBLE + BIASED-METRIC | (no current producer) | `td_learning_results_complex_randomized.csv` | Bugs 2, 6 | Errors 1, 2 |
| `TD-learning_complex_randomized_Agent_0_avg_reward.png` | UNREPRODUCIBLE | (no current producer) | `td_learning_results_complex_randomized.csv` | Bug 6 (rewards unaffected by Bug 2) | Error 2 |
| `TD-learning_complex_randomized_Agent_0_final_reward.png` | UNREPRODUCIBLE | (no current producer) | `td_learning_results_complex_randomized.csv` | Bug 6 | Error 2 |
| (`TD-learning_complex_randomized_regression_signals_*.png` — **expected but missing** because Bug 8 saved them under the Q name) | NOT PRODUCED | — | — | Bug 8 | Error 3 |
| `Roth-Erev_canonical_costly_signal_Agent_0_NMI.png` | UNREPRODUCIBLE | Final_Costly_Signaling_Run_Simulations.ipynb | `urnagent_results_canonical_costly_signal.csv` (1000 rows; per-agent independent costs — never-committed protocol) | (no committed bug; protocol drift) | Error 5a |
| `Roth-Erev_canonical_costly_signal_Agent_0_avg_reward.png` | UNREPRODUCIBLE | Final_Costly_Signaling_Run_Simulations.ipynb | `urnagent_results_canonical_costly_signal.csv` | (protocol drift) | Error 5a |
| `Roth-Erev_canonical_costly_signal_Agent_0_final_reward.png` | UNREPRODUCIBLE | Final_Costly_Signaling_Run_Simulations.ipynb | `urnagent_results_canonical_costly_signal.csv` | (protocol drift) | Error 5a |
| `QLearning_canonical_costly_signal_Agent_0_NMI.png` | CLEAN | Final_Costly_Signaling_Run_Simulations.ipynb | `qlearning_results_canonical_costly_signal.csv` | (none) | Error 5b |
| `QLearning_canonical_costly_signal_Agent_0_avg_reward.png` | CLEAN | Final_Costly_Signaling_Run_Simulations.ipynb | `qlearning_results_canonical_costly_signal.csv` | (none) | Error 5b |
| `QLearning_canonical_costly_signal_Agent_0_final_reward.png` | CLEAN | Final_Costly_Signaling_Run_Simulations.ipynb | `qlearning_results_canonical_costly_signal.csv` | (none) | Error 5b |
| `q_costly_vs_reward.png` | UNREPRODUCIBLE | Final_Costly_Signaling_Run_Simulations.ipynb (or plotting_results) | `urnagent_results_canonical_costly_signal.csv` (Roth-Erev despite Q prefix in filename) | (protocol drift) | Error 5a |
| `q_costs_vs_nmi.png` | UNREPRODUCIBLE | (likely plotting helpers calling `plot_nmi_vs_cost`) | `urnagent_results_canonical_costly_signal.csv` | (protocol drift) | Error 5a |
| `q_learning_costly_single_run.png` | UNREPRODUCIBLE | (single-run trace, source unclear) | likely a single trajectory not in any CSV | (protocol drift) | Error 5a |
| `q_learning_costly_single_run_frequencies.png` | UNREPRODUCIBLE | (single-run trace) | likely a single trajectory | (protocol drift) | Error 5a |
| `initializations_nmi.png` | WRONG | Initializations_test.ipynb | (no CSV — written from in-memory dicts) | Bug 5 | Error 4 |
| `initializations_rewards.png` | WRONG | Initializations_test.ipynb | (no CSV) | Bug 5 | Error 4 |
| `q_opt_canonical.png` | UNREPRODUCIBLE | Parameter_Optimization_wchoices.ipynb | (Bayesian search records) | Bug 7 | Error 7 |
| `q_opt_games.png` | UNREPRODUCIBLE | Parameter_Optimization_wchoices.ipynb | (Bayesian search records) | Bug 7 | Error 7 |
| `td_opt_canonical.png` | UNREPRODUCIBLE + BIASED-METRIC | Parameter_Optimization_wchoices.ipynb | (Bayesian search records on bug-affected TD path) | Bugs 2, 7 | Error 7 |
| `td_opt_games.png` | UNREPRODUCIBLE + BIASED-METRIC | Parameter_Optimization_wchoices.ipynb | (Bayesian search records on bug-affected TD path) | Bugs 2, 7 | Error 7 |

#### A.2 — CSV files (13 — including 1 orphan)

| Filename | Rows | Status | Producer notebook (current) | Bug(s) | Notes |
|---|---|---|---|---|---|
| `urnagent_results_canonical.csv` | 40 000 | CLEAN | `Run_Simulations.ipynb` UrnAgent canonical block (gated `simulate=False`) | (none) | 4 cases × 10 000 iterations. Block currently inactive but produced this file when active. |
| `qlearning_results_canonical.csv` | 40 000 | CLEAN | `Run_Simulations.ipynb` QLearning canonical block | (none) | Same shape; 4 cases × 10 000. |
| `td_learning_results_canonical.csv` | 40 000 | BIASED-METRIC | `Run_Simulations.ipynb` TD canonical block | Bug 2 | NMI columns only; reward columns clean. |
| `urnagent_results_canonical_costly_signal.csv` | 1 000 | UNREPRODUCIBLE (per-agent independent cost protocol — never committed) | (none — code drift) | (no bug; protocol drift) | This is the file `plotting_results.ipynb` reads. Cost columns satisfy `Signal_Cost_A0 != Signal_Cost_A1`. |
| `urnagent_results_canonical_costly_signal (1).csv` | 10 000 | ORPHANED CLEAN | `Final_Costly_Signaling_Run_Simulations.ipynb` UrnAgent block (matches current code) | (none — but no consumer) | Mac-Finder duplicate naming. Cost columns satisfy `Signal_Cost_A0 == Signal_Cost_A1`. Not consumed by any plot. |
| `qlearning_results_canonical_costly_signal.csv` | 10 000 | CLEAN | `Final_Costly_Signaling_Run_Simulations.ipynb` QLearning block (commented out in current notebook but consistent with current code) | (none) | Cost columns satisfy `Signal_Cost_A0 == Signal_Cost_A1`. |
| `urnagent_results_complex.csv` | 8 000 | ORPHANED CLEAN | `Run_Simulations.ipynb` UrnAgent complex block | (none — no consumer) | 4 cases × 2 000 iterations. Bug 6: written by current code but plotting_results doesn't read it. |
| `qlearning_results_complex.csv` | 8 000 | ORPHANED CLEAN | `Run_Simulations.ipynb` QLearning complex block | (none — no consumer) | Same; 4 × 2 000. |
| `td_learning_results_complex.csv` | 8 000 | ORPHANED + BIASED-METRIC | `Run_Simulations.ipynb` TD complex block | Bug 2 | Same; 4 × 2 000. NMI columns only. |
| `urnagent_results_complex_randomized.csv` | 40 000 | UNREPRODUCIBLE | (no current producer) | Bug 6 | 4 × 10 000. From earlier code variant with randomized action sizes. |
| `qlearning_results_complex_randomized.csv` | 40 000 | UNREPRODUCIBLE | (no current producer) | Bug 6 | Same. |
| `td_learning_results_complex_randomized.csv` | 40 000 | UNREPRODUCIBLE + BIASED-METRIC | (no current producer) | Bugs 2, 6 | Same. NMI columns biased on top of unreproducibility. |

### B. CSV column-level status

For each saved CSV with summary columns, the per-column verdict:

#### B.1 — CSVs from `Run_Simulations.ipynb` canonical / complex blocks

Schema (13 columns): `iteration, n_signaling_actions, n_final_actions, full_information, with_signals, Agent_0_Initial_NMI, Agent_0_NMI, Agent_0_avg_reward, Agent_0_final_reward, Agent_1_*`.

| Column | UrnAgent canonical / complex | QLearning canonical / complex | TD canonical / complex |
|---|---|---|---|
| `iteration` | CLEAN | CLEAN | CLEAN |
| `n_signaling_actions` | CLEAN (constant) | CLEAN (constant) | CLEAN (constant) |
| `n_final_actions` | CLEAN (constant) | CLEAN (constant) | CLEAN (constant) |
| `full_information`, `with_signals` | CLEAN (boolean cell ID) | CLEAN | CLEAN |
| `Agent_X_Initial_NMI` | CLEAN | CLEAN | **BIASED-METRIC** — averages `info_hist[:10]` over 5 episodes (twice each) instead of 10 distinct episodes |
| `Agent_X_NMI` | CLEAN | CLEAN | **BIASED-METRIC** — averages `info_hist[-100:]` over last 50 episodes (twice each) instead of last 100 |
| `Agent_X_avg_reward` | CLEAN | CLEAN | CLEAN (`rewards_history` not nested-looped) |
| `Agent_X_final_reward` | CLEAN | CLEAN | CLEAN |

#### B.2 — Costly-signaling CSVs

Schema adds `Signal_Cost_A0, Signal_Cost_A1` columns.

| Column | UrnAgent costly (1000 rows; consumed) | UrnAgent costly (1) (10000 rows; orphan) | QLearning costly (10000 rows) |
|---|---|---|---|
| `Signal_Cost_A0`, `Signal_Cost_A1` | DRIFT — independent per-agent draws (never-committed protocol) | CLEAN — `A0 == A1` matches current shared-cost code | CLEAN — `A0 == A1` matches current code |
| `Agent_X_Initial_NMI`, `Agent_X_NMI` | UNREPRODUCIBLE (under DRIFT cost protocol) | UNREPRODUCIBLE (orphan; no consumer) | CLEAN |
| `Agent_X_avg_reward`, `Agent_X_final_reward` | UNREPRODUCIBLE | UNREPRODUCIBLE (orphan) | CLEAN |

### C. Notebook execution state

Several blocks in `Run_Simulations.ipynb` are gated by `simulate=False` (a flag set in the first cell of each block). The saved CSVs come from past runs when those flags were set to `True`. Current activation state by block:

| Notebook | Block | Current `simulate` flag | Last-known CSV source |
|---|---|---|---|
| `Run_Simulations.ipynb` | UrnAgent canonical | `simulate=False` | `urnagent_results_canonical.csv` (40 000 rows) — produced when `simulate=True` historically |
| `Run_Simulations.ipynb` | QLearning canonical | `simulate=False` | `qlearning_results_canonical.csv` (40 000 rows) — same |
| `Run_Simulations.ipynb` | TD canonical | `simulate=True` (uniquely active!) | `td_learning_results_canonical.csv` — would re-run on Run All |
| `Run_Simulations.ipynb` | UrnAgent complex | (no gate — top-level code) | `urnagent_results_complex.csv` — runs on Run All |
| `Run_Simulations.ipynb` | QLearning complex | (no gate) | `qlearning_results_complex.csv` — runs on Run All |
| `Run_Simulations.ipynb` | TD complex | (no gate) | `td_learning_results_complex.csv` — runs on Run All |
| `Final_Costly_Signaling_Run_Simulations.ipynb` | UrnAgent costly | `simulate=False` | `urnagent_results_canonical_costly_signal.csv` (1000) and `(1).csv` (10 000) historically |
| `Final_Costly_Signaling_Run_Simulations.ipynb` | QLearning costly | (commented out as a large block) | `qlearning_results_canonical_costly_signal.csv` historically |
| `Initializations_test.ipynb` | (single block, ungated) | (always runs on Run All) | Plot output files (no CSV) |
| `Parameter_Optimization_wchoices.ipynb` | All four optimization blocks | (no gate but Restart-and-Run-All fails immediately at Bug 7) | The `*_opt_*.png` were produced when imports were available interactively |

**Implication.** A user who runs `Run_Simulations.ipynb` end-to-end today will:
- Skip the canonical UrnAgent and QLearning blocks (gated off; existing CSVs preserved).
- Run the TD canonical block (active) — overwriting `td_learning_results_canonical.csv` with fresh data still subject to Bug 2.
- Run all three complex blocks (no gate) — overwriting `*_complex.csv` files (which `plotting_results.ipynb` doesn't consume; the saved figures are produced from the orphan `*_complex_randomized.csv`).

So a "clean re-run" today partially overwrites the saved data but does not refresh the figures, because the figures' CSV chain is broken (Bug 6).

### D. Saved test artifacts

#### D.1 — `tests/golden/baseline.json`

**Verdict:** CLEAN.

Contains the 100-episode reward and NMI fingerprints for all three agent types under `seed=12345, initialize=False`, captured against the **canonical** `MultiAgentEnv` + `run_simulation` API in Phase 5 of `REFACTOR_PLAN.md`. Note that the canonical API does **not** have Bug 2 — it computes NMI in a separate single-pass loop after `step_signal`. So the golden baseline is a true reference for the canonical path; it does not encode the buggy behavior of the legacy `TempNetMultiAgentEnv`.

Test [tests/test_golden.py](tests/test_golden.py) asserts post-refactor reproducibility against this baseline; passes.

#### D.2 — `notebooks/basic_unit_test.ipynb` saved cell outputs

**Verdict:** CLEAN.

The 4 MB notebook size is dominated by saved cell outputs (PNG plots inline). Phase 3 audit traced the code and confirmed it uses canonical `MultiAgentEnv` + `run_simulation` for all three agent types. No bug touches the canonical API. The plots are smoke-test illustrations and not consumed by any external figure.

Caveat: no seed is set, so `Restart-and-Run-All` produces different plot content each time. Acceptable for a smoke-test notebook.

#### D.3 — Other test files

All tests under [tests/](tests/) (50 from the refactor + 10 from Phase 4 numerical sanity = 60 total) pass. None of them assert behavior against any saved figure or CSV in `results/`; they assert against in-memory computations only. So the test suite is not affected by any of the errors above.

### E. Bug 2 magnitude analysis

The argument for "BIASED-METRIC, magnitude small in second decimal" is informal. Here is the analytical bound.

Let $\hat{N}_t$ denote the cumulative NMI estimator at episode $t$ (i.e. NMI computed from the cumulative `signal_usage[i]` after $t$ episodes). Then under the canonical env:

$$\text{Agent\_X\_NMI}_{\text{canonical}} = \frac{1}{100} \sum_{t = T-99}^{T} \hat{N}_t.$$

Under the legacy `TempNetMultiAgentEnv` 2× inflation, each $\hat{N}_t$ is appended twice, so `info_hist[-100:]` covers `t = T-49` through `T` with each value appearing twice:

$$\text{Agent\_X\_NMI}_{\text{legacy}} = \frac{1}{100} \cdot 2 \sum_{t=T-49}^{T} \hat{N}_t = \frac{1}{50} \sum_{t=T-49}^{T} \hat{N}_t.$$

The discrepancy is:

$$\text{Agent\_X\_NMI}_{\text{legacy}} - \text{Agent\_X\_NMI}_{\text{canonical}} = \frac{1}{50} \sum_{t=T-49}^{T} \hat{N}_t - \frac{1}{100} \sum_{t=T-99}^{T} \hat{N}_t.$$

Rewriting, this equals

$$\frac{1}{100} \sum_{t = T-49}^{T} \hat{N}_t - \frac{1}{100} \sum_{t = T-99}^{T-50} \hat{N}_t = \frac{1}{100} \big( \sum_{t=T-49}^{T} \hat{N}_t - \sum_{t=T-99}^{T-50} \hat{N}_t \big).$$

So the bias is the difference between the average over the **most recent 50 episodes** and the average over the **previous 50 episodes**. If the cumulative NMI estimator is increasing late in training (typical, as the policy converges), the legacy summary is **systematically higher** than the canonical by exactly this difference.

**Quantitative bound.** Each $\hat{N}_t \in [0, 1]$, so the per-episode change $|\hat{N}_{t+1} - \hat{N}_t| \le \frac{1}{t}$ in the worst case (one new sample shifts the estimator by at most $O(1/t)$). For $T = 10000$, the difference between the last-50 mean and the previous-50 mean is at most:

$$\bigg| \frac{1}{50} \sum_{t=T-49}^{T} \hat{N}_t - \frac{1}{50} \sum_{t=T-99}^{T-50} \hat{N}_t \bigg| \le 50 \cdot \max_{t \in [T-99, T]} \frac{1}{t} \approx \frac{50}{T-99} \approx 0.005.$$

So the worst-case bias on `Agent_X_NMI` for $T = 10000$ TD-canonical runs is about **0.005**. Empirically the bias will usually be **much smaller** (the estimator is smooth, not adversarial), often in the third decimal.

For `Agent_X_Initial_NMI` (first 10 entries / first 5 distinct episodes), the bound is much looser because $1/t$ is large at small $t$. The bias could be up to ~0.5 in the worst case, but the early-episode NMI is dominated by the H(O)=0 → NMI=0 convention (only one observation seen), so practical bias is small.

**Bottom line.** The bias on `Agent_X_NMI` is bounded above by ~0.005 for 10 000-episode TD-canonical runs and is much smaller in expectation. For comparing TD-learning to other agents, a difference of less than 0.005 in mean final NMI is below the noise floor; a difference of 0.05+ is robustly larger than the bias. Use this rule of thumb when reading TD-learning vs. UrnAgent / QLearning comparisons in the saved figures.

### F. Cross-error chain effects

Errors compound. The full chain when reading a TD-learning figure produced by the current code:

1. Hyperparameter optimization (`Parameter_Optimization_wchoices.ipynb` TD section) ran on the bug-affected path, so the saved best parameters (`learning_rate=0.1, gamma=0.99`, `choice='egreedy'`, etc.) were chosen against a slightly-noisier-than-intended NMI objective. **Magnitude:** small, because Bayesian search is robust to objective noise. (Error 7)
2. `Run_Simulations.ipynb` TD canonical block uses those slightly-suboptimal hyperparameters to produce `td_learning_results_canonical.csv`. (Independent of Bug 2 here.)
3. The summary statistics in that CSV (`Agent_X_Initial_NMI`, `Agent_X_NMI`) are biased by Bug 2 — averages over half the temporal window. **Magnitude:** bounded ≤ 0.005 by Section E. (Error 1)
4. `plotting_results.ipynb` consumes the biased columns to produce `TD-learning_canonical_*.png`. Bug 8 doesn't apply here (it only affects the complex regression figures). (Error 1, no Error 3)
5. Anyone comparing the TD-canonical NMI distribution against Q-canonical or Roth-Erev-canonical NMI sees a systematic **upward** offset of ≤ 0.005 plus $\sqrt{2}\times$ wider variance on the TD side. (Error 1)

For TD-complex figures the chain compounds further with Error 2 (CSV producer/consumer mismatch). For misnamed Q-complex regression PNGs the chain compounds with Error 3 (Bug 8 saves TD content under Q name).

### G. Predicted post-fix change direction

For each error, the direction the saved value will move after the fix lands and the experiment is re-run. Useful when sanity-checking a re-run.

| Error | Saved value | Post-fix direction | Magnitude |
|---|---|---|---|
| Error 1 | `td_learning_results_canonical.csv :: Agent_X_NMI` | **Decreases** (typically) — last 100 episodes' average is slightly lower than last 50's, since the cumulative NMI estimator is monotone-increasing late in training | ≤ 0.005 (Section E) |
| Error 1 | same column, but for runs where the policy is still drifting | could increase | bounded by same |
| Error 1 | `Initial_NMI` | usually increases — more episodes mean fewer H(O)=0 zero-padding hits | up to ~0.05 |
| Error 1 | reward columns | unchanged | exact |
| Error 2 | UrnAgent / Q-learning complex figures (Option A — restore randomized action sizes) | **Different distributions** — the action-size axis becomes a random rather than fixed parameter; the NMI / reward distributions widen | substantial — different experimental design |
| Error 2 | (Option B — retire `_randomized` figures) | the fixed-action-size figures from current `*_complex.csv` replace them — **smaller scale** (8 000 vs 40 000 rows) and tighter distributions | 5× fewer iterations |
| Error 3 | `Q-learning_complex_randomized_regression_*.png` | Replaced with actual Q-learning regression content | substantial — currently shows TD content |
| Error 4 | `initializations_*.png` curves | Each curve moves to its own trajectory — the four init weights produce **distinct** convergence rates instead of overlapping noise | substantial — currently zero between-curve effect |
| Error 5a | Roth-Erev costly figures | Either re-run with current shared-cost protocol (figure shape changes; cost axis becomes single-value per row) or with restored independent-cost protocol (matches current figures qualitatively) | depends on direction |
| Error 5b | Q-learning costly figures | unchanged after Bug 7 fix | (already clean) |
| Error 7 | `*_opt_*.png` | After Bug 7 fix and optional re-run with Bug 2 fixed for TD: the best TD hyperparameters might shift slightly | small (Bayesian search robust to objective noise) |

### H. Orphaned files

Files that exist in `results/` but no notebook consumes:

| File | Why orphaned | Recommended action |
|---|---|---|
| `urnagent_results_canonical_costly_signal (1).csv` | Mac-Finder duplicate naming (`(1)` suffix). Contains 10 000 rows under shared-cost protocol — matches current notebook. The plotting notebook reads the un-suffixed (1000-row, drift-protocol) sibling instead. | Delete this file once Error 5a is resolved (either by point plotting at it, in which case rename to drop the suffix, or by re-running). |
| `urnagent_results_complex.csv` | 8 000 rows from current `Run_Simulations.ipynb` UrnAgent complex block. Plotting notebook reads `urnagent_results_complex_randomized.csv` instead (Bug 6). | Delete after Bug 6 is resolved — Option A regenerates the `_randomized` version with current code; Option B retires `_randomized` and points plotting at this file. |
| `qlearning_results_complex.csv` | Same — 8 000 rows, no consumer. | Same. |
| `td_learning_results_complex.csv` | Same — 8 000 rows, no consumer. | Same. |

No orphaned PNGs; every PNG in `results/` is at least nominally referenced by some notebook (though in 5 cases the reference uses a wrong filename — Bug 8).

---

## Caveats on this audit

Three things this log does **not** establish:

1. **Magnitude of Bug 2's actual numerical drift on saved CSV columns.** I argued it's "in the second decimal" based on the smoothness of the cumulative NMI estimator late in training, but I have not run an A/B comparison with and without the bug. A targeted re-run on a single seed (or a paired-seed comparison across the bugged and fixed paths) would settle the magnitude exactly. This is a Phase 6 task.
2. **Whether the saved `*_complex_randomized.csv` files were produced by code that is otherwise identical to the current `Run_Simulations.ipynb`.** I have established that the *filename* differs and the *action-size handling* differs (fixed in current code, randomized in the saved data). I have not established that nothing else differs. If seeds, episode counts, or agent kwargs differed in the prior version, the figures may differ from a Bug 6 Option A re-run by more than just filename.
3. **Notebook narrative claims I did not trace.** Phase 3's audit covered the experimental loops (the cells that build `env`, run `simulation_function`, aggregate into CSV columns). It did not exhaustively read every markdown cell or every analysis cell. If a notebook's text asserts an interpretation that depends on a quantity not explicitly traced, that interpretation is outside this log's scope.

## Cross-references

| File | What it gives you |
|---|---|
| [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md) | Per-bug details — symptom, root cause, code locations, fix proposals |
| [DEBUGGING_PLAN.md](DEBUGGING_PLAN.md) | Phase 5 fix plan with batches, dependencies, regeneration list |
| [analytics/metrics_aggregation.md](analytics/metrics_aggregation.md) | The trajectory → CSV → figure pipeline traced bug-by-bug |
| [analytics/scripts/](analytics/scripts/) | Independent verification scripts confirming the kernel math is clean |
| [tests/test_numerical_sanity.py](tests/test_numerical_sanity.py) | Hand-derived numerical assertions backing the kernel-math correctness claim |

## Summary table

| Saved artifact set | Verdict | Action before citing |
|---|---|---|
| Roth-Erev canonical (5 figures) | CLEAN | None |
| Q-learning canonical (5 figures) | CLEAN | None |
| TD-learning canonical (5 figures — NMI-based) | BIASED-METRIC (Bug 2) | Note half-window caveat or re-run after Bug 2 fix |
| TD-learning canonical (reward-only figures within the same set) | CLEAN (rewards unaffected by Bug 2) | None |
| Costly-signaling Roth-Erev (3 figures + 4 q_costly_*) | UNREPRODUCIBLE (Error 5a — never-committed independent-cost protocol) | Decide cost-draw protocol; re-run |
| Costly-signaling Q-learning (3 figures) | CLEAN | None |
| Roth-Erev complex (5 figures) | UNREPRODUCIBLE (Bug 6) | Re-run after Bug 6 decision |
| Q-learning complex (3 histograms) | UNREPRODUCIBLE (Bug 6) | Re-run after Bug 6 decision |
| `Q-learning_complex_randomized_regression_*.png` (2 figures) | MISLABELED + UNREPRODUCIBLE + BIASED-METRIC (Bugs 6, 8 + TD content) | Re-run after Bugs 6, 8 fix |
| TD-learning complex (3 histograms; regression PNGs missing) | UNREPRODUCIBLE + BIASED-METRIC (Bugs 2, 6) | Re-run after Bugs 2, 6 fix |
| `initializations_*.png` (2 figures) | WRONG (Bug 5) — labels claim init effect that isn't there | Re-run after Bug 5 fix |
| Hyperparameter optimization Q-side (2 figures) | UNREPRODUCIBLE (Bug 7) | Re-run after Bug 7 fix if used as paper input |
| Hyperparameter optimization TD-side (2 figures) | UNREPRODUCIBLE + BIASED-METRIC (Bugs 2, 7) | Re-run after Bugs 2, 7 fix if used as paper input |
| `tests/golden/baseline.json` | CLEAN (canonical-API reference) | None |
| `notebooks/basic_unit_test.ipynb` saved outputs | CLEAN (smoke-test only) | None |
