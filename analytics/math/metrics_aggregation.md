# Metrics aggregation — from trajectories to saved CSVs and figures

- status: active
- type: explanation
- id: rl_signaling.analytics.metrics_aggregation
- description: Traces how the per-episode trajectory data structures (rewards_history, signal_information_history, signal_usage, histories, nature_history) are aggregated into the saved CSV columns by the experiment notebooks, and how plotting_results.ipynb consumes those CSVs to produce the figures in results/. Also explains how Bug 2's history inflation interacts with the slice-based summary statistics, and how Bug 6's filename mismatch breaks the producer/consumer chain.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

The `rl_signaling/env.py` envs and `rl_signaling/simulation.py` runners produce **trajectory-level** data — per-episode arrays. The experiment notebooks compress those trajectories into **summary scalars**, save them as rows in CSVs, and `plotting_results.ipynb` reads the CSVs to render the final figures. Understanding this aggregation chain is essential for interpreting the saved figures and for tracing the impact of any bug that touches a per-episode quantity.

This file walks through the chain end-to-end, gives the exact slice / mean / aggregation formulas, traces every saved figure back to its CSV and producer notebook, and connects the chain to the structural bugs filed during the DEBUGGING_PLAN audit (Bug 2 and Bug 6 in particular).

## The five trajectory data structures

Every env exposes a `report_metrics()` method returning a 5-tuple. The canonical [MultiAgentEnv.report_metrics](../../rl_signaling/env.py#L283-L293):

```python
return (
    self.signal_usage,
    self.rewards_history,
    self.signal_information_history,
    self.nature_history,
    self.histories,
)
```

| Element | Type | Shape | Per-episode appended at | Meaning |
|---|---|---|---|---|
| `signal_usage` | `list[dict]` (one per agent) | `dict[obs_tuple, np.ndarray[K]]` | [env.py:174-183](../../rl_signaling/env.py#L174-L183) inside `step_signal` | Cumulative count of `signal_usage[agent_i][obs][signal_index]` over all episodes. |
| `rewards_history` | `list[list[float]]` (one inner list per agent) | `(N_agents, T)` | [env.py:255](../../rl_signaling/env.py#L255) inside `update` | Per-episode net reward for each agent (already including any signaling cost). |
| `signal_information_history` | `list[list[float]]` (one per agent) | `(N_agents, T)` (canonical), `(N_agents, 2T)` (legacy `TempNetMultiAgentEnv`, see Bug 2) | [env.py:185-187](../../rl_signaling/env.py#L185-L187) inside `step_signal` | Per-episode NMI computed from the cumulative `signal_usage[agent_i]`. |
| `nature_history` | `list[tuple]` | `(T,)` | [env.py:149](../../rl_signaling/env.py#L149) inside `reset` | The full nature vector for each episode. |
| `histories` | `dict[agent_i, dict[str, list]]` | `histories[agent][channel][t] = deepcopy(usage[agent])` | [env.py:263-264](../../rl_signaling/env.py#L263-L264) inside `update` | Snapshot of `signal_usage` and `action_usage` after every episode. Memory-inefficient by design (used by `plot_simulation_summary` to render the urn-proportion-over-time panel). |

Indexing convention: $T$ is the number of episodes (`n_episodes`). The lists are zero-indexed in Python; episode $t$ (one-based) corresponds to slot `t-1`.

## How `signal_information_history[i][t]` is computed

After the signal step at episode $t$, each agent $i$'s `signal_usage[i]` has one extra count (the signal just emitted). The NMI is then computed on the **cumulative** usage:

$$\text{nmi}_t^{(i)} = \mathrm{NMI}\big( \text{signal\_usage}[i] \big),$$

where `compute_mutual_information` is the function described in [information_theory.md](information_theory.md). This is appended to `signal_information_history[i]`. So the history at index $t-1$ reflects the NMI computed **using the first $t$ episodes' signals**.

Note: the per-episode NMI is **not** the NMI of just episode $t$'s signal — that would be undefined (one observation, one signal). It is the cumulative-distribution NMI estimated up to and including episode $t$.

### Implication for time-series plots

When `plot_simulation_summary` (panel 2) plots `signal_information_history[i]` as a function of episode, it shows the **trajectory of cumulative NMI estimates**. Early in the run this is noisy (small sample size); late in the run it converges to the true NMI of the policy if the policy is stationary. If the policy is still drifting (early in training), the cumulative estimate lags the instantaneous policy.

### Bug 2 interaction (legacy `TempNetMultiAgentEnv`)

In the legacy two-step env, the inner NMI loop fires once per outer agent iteration, doubling the length of `signal_information_history[i]`. So:

- Canonical `MultiAgentEnv`: `len(signal_information_history[i]) == T` after $T$ episodes.
- Legacy `TempNetMultiAgentEnv` with $N$ agents: `len(signal_information_history[i]) == N · T`.

The values themselves are still NMI estimates — they are just **duplicated** within each signal phase (each agent's NMI is recomputed and appended $N$ times instead of once). This is Bug 2's actual symptom: same values, $N\times$ more entries.

## Per-episode reward computation

For non-costly runs, `rewards_history[i][t-1]` is simply $G_i(\mathbf{v}_t, \alpha_{i,t})$, the game-dict lookup at episode $t$.

For costly runs, [env.py:236-241](../../rl_signaling/env.py#L236-L241):

$$\text{rewards\_history}[i][t-1] = G_i(\mathbf{v}_t, \alpha_{i,t}) - c_i \cdot \mathbb{1}[\sigma_{i,t} \neq \nu].$$

So the saved per-episode reward is the **net** value (game reward minus cost). The gross reward is not separately tracked.

## How experiment notebooks aggregate trajectories into CSV rows

`Run_Simulations.ipynb`, `Final_Costly_Signaling_Run_Simulations.ipynb`, and `Initializations_test.ipynb` all follow the same aggregation pattern (with some variation in which regimes they iterate over). For a given `(iteration, agent_type, full_information, with_signals)` cell:

```python
signal_usage, rewards_history, signal_information_history, _, _ = simulation_function(...)

for agent_id in range(n_agents):
    info_hist = signal_information_history[agent_id]
    reward_hist = rewards_history[agent_id]
    results.extend([
        np.mean(info_hist[:10]),            # Agent_X_Initial_NMI
        np.mean(info_hist[-100:]),          # Agent_X_NMI
        np.mean(reward_hist),               # Agent_X_avg_reward
        np.mean(reward_hist[-100:])         # Agent_X_final_reward
    ])
```

(taken verbatim from [notebooks/Run_Simulations.ipynb](../../notebooks/Run_Simulations.ipynb) UrnAgent block).

So the four scalar columns per agent are:

| Column | Formula | Episodes covered (canonical) |
|---|---|---|
| `Agent_X_Initial_NMI` | $\dfrac{1}{10} \sum_{t=1}^{10} \text{nmi}_t^{(X)}$ | First 10 |
| `Agent_X_NMI` | $\dfrac{1}{100} \sum_{t=T-99}^{T} \text{nmi}_t^{(X)}$ | Last 100 |
| `Agent_X_avg_reward` | $\dfrac{1}{T} \sum_{t=1}^{T} r_{X, t}$ | All $T$ |
| `Agent_X_final_reward` | $\dfrac{1}{100} \sum_{t=T-99}^{T} r_{X, t}$ | Last 100 |

And for the costly experiments, two additional cost columns:

| Column | Source |
|---|---|
| `Signal_Cost_A0` | Iteration's $c_0$ value (set once per iteration, not derived from trajectory) |
| `Signal_Cost_A1` | Iteration's $c_1$ value |

### Bug 2 interaction with the slice indices

Because `info_hist[:10]` and `info_hist[-100:]` are positional slices (zero-based), they read **fixed indices** of the `signal_information_history[i]` array regardless of its length. Under the canonical env, `info_hist` has length $T$ and the slices cover episodes 1–10 and $(T-99)$–$T$. Under the legacy env (Bug 2's $2\times$ inflation), `info_hist` has length $2T$:

| Slice | Canonical | Legacy (`TempNetMultiAgentEnv`) |
|---|---|---|
| `info_hist[:10]` | first 10 entries = first 10 episodes | first 10 entries = first 5 episodes (each appended twice) |
| `info_hist[-100:]` | last 100 entries = last 100 episodes | last 100 entries = last 50 episodes |

So `Agent_X_Initial_NMI` and `Agent_X_NMI` in the legacy path average over **half** the intended sample. The values are still well-defined and qualitatively in the right direction (NMI is non-negative and bounded), but the variance of the estimator is $\sqrt{2}\times$ larger and the temporal window is half what was intended. This is the corrected experimental impact in Bug 2.

The `Agent_X_avg_reward` and `Agent_X_final_reward` columns are **unaffected** because they read from `rewards_history`, which is appended once per episode in both env variants (only the NMI inner loop is nested in the legacy path).

## How `plotting_results.ipynb` consumes the CSVs

`plotting_results.ipynb` reads each CSV, then dispatches to helpers in [rl_signaling/plotting.py](../../rl_signaling/plotting.py):

| Helper | Reads columns | Emits |
|---|---|---|
| `plot_all_histograms(df, filename_prefix=...)` | `Agent_0_final_reward`, `Agent_0_avg_reward`, `Agent_0_NMI` | 3 PNG histograms with KDE overlays per call |
| `plot_regression(df, x_var, y_var, filename_prefix=...)` | columns named in `x_var` and `y_var` (default `Agent_0_NMI` vs `Agent_0_final_reward`) | 1 PNG per `(with_signals, full_information)` pair |
| `plot_reward_vs_cost(df)` | `Signal_Cost_A0`, `Agent_0_final_reward` | 1 PNG (regression of reward vs cost) |
| `plot_nmi_vs_cost(df)` | `Signal_Cost_A0`, `Agent_0_NMI` | 1 PNG (regression of NMI vs cost) |
| `count_negative_nmi(file_path)` | every column with `"NMI"` in the name | dict of negative-value counts (diagnostic, not a figure) |

So every saved figure has a single CSV consumer, which has a single notebook producer. The chain is fragile under filename drift (see Bug 6).

## Producer ↔ consumer trace

The complete map of saved figures back to their producing notebook, organized by experiment.

### Canonical (2-feature, fixed action sizes)

| Figure | Consumer cell | CSV | Producer notebook | Affected by |
|---|---|---|---|---|
| `Roth-Erev_canonical_Agent_0_NMI.png` | `plot_all_histograms` | `urnagent_results_canonical.csv` | [Run_Simulations.ipynb](../../notebooks/Run_Simulations.ipynb) UrnAgent canonical block | (none) |
| `Roth-Erev_canonical_Agent_0_avg_reward.png` | same | same | same | (none) |
| `Roth-Erev_canonical_Agent_0_final_reward.png` | same | same | same | (none) |
| `Roth-Erev_canonical_regression_signals_*.png` | `plot_regression` | same | same | (none) |
| `Q-learning_canonical_*.png` | `plot_all_histograms` + `plot_regression` | `qlearning_results_canonical.csv` | [Run_Simulations.ipynb](../../notebooks/Run_Simulations.ipynb) QLearning canonical block | (none) |
| `TD-learning_canonical_*.png` | same | `td_learning_results_canonical.csv` | [Run_Simulations.ipynb](../../notebooks/Run_Simulations.ipynb) TDAgent canonical block | **Bug 2** (slices land on first 5 / last 50 instead of first 10 / last 100) |

### Costly signaling

| Figure | Consumer cell | CSV | Producer notebook | Affected by |
|---|---|---|---|---|
| `Roth-Erev_canonical_costly_signal_*.png` | `plot_all_histograms` + `plot_reward_vs_cost` + `plot_nmi_vs_cost` | `urnagent_results_canonical_costly_signal.csv` | [Final_Costly_Signaling_Run_Simulations.ipynb](../../notebooks/Final_Costly_Signaling_Run_Simulations.ipynb) UrnAgent block | (none) |
| `QLearning_canonical_costly_signal_*.png` | same | `qlearning_results_canonical_costly_signal.csv` | (same notebook, QLearning block — currently `simulate=False`-gated) | (none) |

### Initialization study

| Figure | Consumer cell | CSV | Producer notebook | Affected by |
|---|---|---|---|---|
| `initializations_rewards.png` | inline plt cell | (none — written from in-memory dicts) | [Initializations_test.ipynb](../../notebooks/Initializations_test.ipynb) | **Bug 5** (env.agents overwrite — every curve is the same configuration) |
| `initializations_nmi.png` | same | same | same | **Bug 5** (same reason) |

### Complex / general games

| Figure | Consumer cell | CSV | Producer notebook | Affected by |
|---|---|---|---|---|
| `Roth-Erev_complex_randomized_*.png` (5 PNGs) | `plot_all_histograms` + `plot_regression` | `urnagent_results_complex_randomized.csv` | **No current producer** — the matching block in `Run_Simulations.ipynb` writes `urnagent_results_complex.csv` (without `_randomized`) | **Bug 6** (filename mismatch; figures reflect stale data) |
| `Q-learning_complex_randomized_*.png` (5 PNGs) | same | `qlearning_results_complex_randomized.csv` | **No current producer** | **Bug 6** |
| `TD-learning_complex_randomized_*.png` (3 PNGs) | `plot_all_histograms` only — the regression PNGs do not exist | `td_learning_results_complex_randomized.csv` | **No current producer** | **Bug 6** |
| `Q-learning_complex_randomized_regression_signals_*.png` | `plot_regression` last cell | `td_learning_results_complex_randomized.csv` (TD data!) | **No current producer**, and the cell uses Q prefix for TD data | **Bug 6** + **Bug 8** (the Q PNG actually contains TD content; the TD regression PNG is never produced) |

The `_complex` (no `_randomized`) CSVs are written by [Run_Simulations.ipynb](../../notebooks/Run_Simulations.ipynb) but consumed by no figure. The `_complex_randomized` CSVs in `results/` are consumed by figures but produced by no current notebook. The two sides of the chain are disconnected — the substance of Bug 6.

### Hyperparameter optimization (research log)

| Output | Consumer | Producer notebook | Affected by |
|---|---|---|---|
| `q_opt_canonical.png`, `q_opt_games.png`, `td_opt_canonical.png`, `td_opt_games.png` | none (saved as a research record, not consumed by `plotting_results`) | [Parameter_Optimization_wchoices.ipynb](../../notebooks/Parameter_Optimization_wchoices.ipynb) | **Bug 7** (notebook fails Restart-and-Run-All due to missing imports; existing figures are from a prior run) |
| `q_costs_vs_reward.png`, `q_costs_vs_nmi.png` | none | (one of the costly notebooks; not currently in `plotting_results.ipynb`) | (none) |

## Cross-bug summary in CSV-trace form

After all open Phase 5 bugs land:

- **Bug 2 fix** (or migrate notebooks off legacy env) → `Agent_X_Initial_NMI` and `Agent_X_NMI` in `td_learning_results_*.csv` shift slightly (different episode windows averaged); downstream `TD-learning_canonical_*.png` and any TD complex figures regenerate.
- **Bug 5 fix** → `initializations_*.png` regenerate, this time showing the actual init effect.
- **Bug 6 fix** → either the `*_complex.csv` files are renamed (Option A) or the figures are retired (Option B); the chain reconnects.
- **Bug 8 fix** → the missing `TD-learning_complex_randomized_regression_*.png` files appear; the misnamed Q-prefix files get correct content via re-running the Q-block.
- **Bug 7 fix** → no figure changes; the hyperparameter notebook can be re-run from a fresh kernel.

## Cross-references

| File | Topic |
|---|---|
| [signaling_model.md](signaling_model.md) | What an episode produces (the sources of `rewards_history` etc.) |
| [agent_urn.md](agent_urn.md), [agent_q_learning.md](agent_q_learning.md), [agent_td_learning.md](agent_td_learning.md) | Agent-side updates that move the trajectories |
| [information_theory.md](information_theory.md) | The NMI computation that fills `signal_information_history` |
| [rl_signaling/plotting.py](../../rl_signaling/plotting.py) | The plotting helpers consuming the CSVs |
| [DEBUGGING_PLAN.md](../../docs/code-audit/DEBUGGING_PLAN.md) Phase 5 | Bug ledger and fix plan that this file's "affected by" columns trace to |
| [LEGACY_BUGS_LOG.md](../../docs/code-audit/LEGACY_BUGS_LOG.md) | Per-bug detail (Bug 2, Bug 5, Bug 6, Bug 7, Bug 8) |
