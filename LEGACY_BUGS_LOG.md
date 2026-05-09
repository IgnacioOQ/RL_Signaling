# Legacy Bugs Log
- status: active
- type: log
- id: rl_signaling.legacy_bugs_log
- description: Append-only catalog of bugs identified in the pre-refactor codebase — root cause, impact on saved experimental results, fix status, and pending debugging follow-ups.
- label: [agent]
- injection: excluded
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->
This file catalogs the bugs surfaced during the multi-phase refactor of the legacy RL_Signaling codebase. Each entry documents the symptom, the root cause, the experimental impact (which checked-in results are affected), the fix, and how the fix was verified.

The intent is to support a thorough debugging follow-up: re-running the affected experiments against the fixed code, quantifying the differences, and deciding which checked-in figures need to be regenerated. The audit trail also makes the academic record honest — if any of these experiments are written up, the bugs and their resolution should be acknowledged.

Severity scale used below:

| Severity | Meaning |
|---|---|
| **High** | Silently changed the experimental result for at least one notebook. |
| **Medium** | Affected a non-result-bearing observable (e.g. a metric's history length) or only an error path. |
| **Low** | Cosmetic / stylistic (stale comments, dead code). Not catalogued here unless coupled to a real defect. |

## Catalog

| # | Bug | Severity | Status | Affected notebooks |
|---|---|---|---|---|
| 1 | `UrnAgent.__init__` silently never pre-seeds `action_urns` | **High** | Fixed in Phase 4 (golden-run gated) | `notebooks/Initializations_test.ipynb` only (no longer affected — see Bug 5) |
| 2 | `TempNetMultiAgentEnv.get_actions` runs the NMI inner loop once per outer-loop iteration | **Medium** | Not yet fixed (legacy-only path; deprecated env) | TD-learning CSVs (`Agent_X_Initial_NMI` averages over 5 episodes vs intended 10; `Agent_X_NMI` averages over 50 vs intended 100) |
| 3 | `utils.py` referenced `sys.stderr` without `import sys` | **Medium** | Fixed in Phase 1 | None — only fires on error paths in `plot_reward_vs_cost` / `plot_nmi_vs_cost` |
| 4 | `QLearningAgent.__init__` only pre-seeds the signaling Q-table when `initialize=True`; the action Q-table is silently overwritten to `{}` | **Medium** | Fixed (Batch B, 2026-05-09 — symmetric pre-seed) | `notebooks/Initializations_test.ipynb` only (sole consumer of `initialize=True`) |
| 5 | `Initializations_test.ipynb` overwrites `env.agents` with `initialize=False` defaults, silently invalidating the entire initialization experiment | **High** | Fixed (Batch B, 2026-05-09 — Option B migration) | `notebooks/Initializations_test.ipynb` — every saved figure |
| 6 | `Run_Simulations.ipynb` writes `*_complex.csv` files but `plotting_results.ipynb` reads `*_complex_randomized.csv` — the canonical "complex" figures are produced from CSVs no current notebook regenerates | **High** | Fixed (Batch B, 2026-05-09 — Option A: randomized action sizes restored) | `plotting_results.ipynb` "General Games / General Urns" sections; all `*_complex_randomized_*.png` figures |
| 7 | `Parameter_Optimization_wchoices.ipynb` references `Categorical`, `Optimizer`, `Parallel`, `delayed`, `multiprocessing`, `datetime` without importing them — Restart-and-Run-All fails | **Medium** | Fixed (Batch A, 2026-05-09) | `notebooks/Parameter_Optimization_wchoices.ipynb` — all four optimization cells fail on a fresh kernel |
| 8 | `plotting_results.ipynb` final cell uses `filename_prefix='Q-learning_complex_randomized'` for a TD-learning DataFrame | **Low** | Fixed (Batch A, 2026-05-09) | `plotting_results.ipynb` — the saved TD-learning regression PNGs are misnamed (overwritten by Q-learning ones) |
| 9 | `_generate_hot_vectors` returns int64 arrays; constant-α TD updates on pre-seeded Q-tables silently truncate fractional changes | **High** | Fixed (2026-05-09 follow-up) | `notebooks/Initializations_test.ipynb` QLearning block — `init_weights` had no observable effect because every fractional update was floored back to int |

---

## Bug 1 — `UrnAgent.__init__` silently never pre-seeds `action_urns`
- status: done
- type: task
- id: rl_signaling.legacy_bugs_log.urn_agent_action_urns_init
- last_checked: 2026-05-08
<!-- content -->
**Severity:** High
**File (pre-refactor):** `agents.py` lines 28–37 (the `UrnAgent.__init__` method)
**File (post-refactor):** [rl_signaling/agents.py](rl_signaling/agents.py) — fixed in Phase 4
**Status:** Fixed; the post-fix behavior is asserted by `tests/test_agents.py::test_urn_agent_initialize_true_seeds_action_urns`.

### Symptom

When `UrnAgent` was constructed with `initialize=True`, only the **signaling urns** were pre-seeded with one-hot vectors. The **action urns** were silently left as an empty dictionary `{}`, indistinguishable from the `initialize=False` case at any subsequent observation point. The first call to `agent.get_action(obs)` would lazy-initialize `action_urns[obs]` to the uniform `np.ones(n_final_actions)`, exactly as it would for an `initialize=False` agent.

### Root cause

The buggy block (using the pre-refactor identifiers and indentation):

```python
if initialize:
    self.signaling_urns = create_initial_signals(
        n_observed_features=n_observed_features,
        n_signals=n_signaling_actions,
        n=initialization_weights[0],
        m=initialization_weights[1],
    )
    self.action_urns = create_initial_signals(
        n_observed_features=n_observed_features + 1,
        n_signals=n_final_actions,
        n=initialization_weights[0],
        m=initialization_weights[1],
    )
else:
    self.signaling_urns = {}
self.action_urns = {}                # ← BUG: outside the if/else
```

The trailing `self.action_urns = {}` was not nested inside the `else` branch — it sat at the same indentation level as the `if/else` block, so it executed unconditionally after every construction. The `if initialize:` branch's `self.action_urns = create_initial_signals(...)` was overwritten before the `__init__` returned.

### Why it was hard to spot

1. **No type, no test, no warning.** Python doesn't complain about reassigning an attribute. There was no unit test exercising the `initialize=True` path, and no docstring or invariant check that would have caught a silently empty `action_urns`.
2. **The signaling urn worked.** Because `signaling_urns` was correctly pre-seeded, the agent's first signal-selection call did reflect the initialization weights. Cursory inspection of "is initialization having an effect?" said yes — but only because half of the initialization was working.
3. **`get_action` lazy-initializes.** When `action_urns[obs]` is missing, `get_action` falls back to `np.ones(n_final_actions)` — the *same* default that `initialize=False` agents use. So the `initialize=True` agent was indistinguishable from `initialize=False` at the action level after the first episode.
4. **The bug only matters for one notebook.** Of the six experiment notebooks, only `Initializations_test.ipynb` constructs agents with `initialize=True`. The others pass `initialize=False` and were never affected.

### Experimental impact

- **Affected notebook:** [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) — its entire purpose is to vary `initialization_weights` (e.g. `[10, 0]`, `[5, 0]`, `[1, 0]`) and observe the effect on convergence. Pre-fix, it was effectively measuring the effect of biasing the *signaling urn alone* while the action urn started uniform on every run, regardless of `initialization_weights`.
- **Affected saved figures:** [results/initializations_nmi.png](results/initializations_nmi.png) and [results/initializations_rewards.png](results/initializations_rewards.png) — these reflect the partial, signaling-urn-only initialization.
- **Unaffected:** every other figure and CSV under `results/`. Those experiments use `initialize=False` and the bug never triggered.

### Predicted change after the fix

With both urns now pre-seeded:

- Strong-init runs (large `initialization_weights[0]`, e.g. `[10, 0]`) should show **stronger and faster** convergence than the pre-fix runs, because the action urn carries a strong bias toward the optimal `(observation, received_signal) → action` mapping from the start.
- Weak-init runs (`[1, 0]`) should be closer to the pre-fix behavior, because the magnitude `1` matches the lazy-init default — just non-uniform.
- The contrast between the curves should be more pronounced; the comparison now does what the notebook's name claims.

These predictions are not yet empirically verified — that is the goal of the planned debugging follow-up.

### The fix

```python
self.signaling_urns: dict
self.action_urns: dict
if initialize:
    self.signaling_urns = create_initial_signals(...)
    self.action_urns = create_initial_signals(...)
else:
    self.signaling_urns = {}
    self.action_urns = {}            # ← inside the else branch now
```

The `self.action_urns = {}` line moved inside the `else` branch alongside `self.signaling_urns = {}`. Type annotations were added at the top so the empty-dict default is still strict-mode friendly.

### Verification

- **Unit test:** `tests/test_agents.py::test_urn_agent_initialize_true_seeds_action_urns` constructs a `UrnAgent` with `initialize=True`, `n_observed_features=1`, `n_signaling_actions=2`, `n_final_actions=4`, and asserts that `action_urns` has 4 entries (one per `(observation, received_signal)` pair) and that every entry is a one-hot vector of length 4.
- **Golden-run regression:** `tests/test_golden.py` runs all three agent types at fixed seed with `initialize=False` and asserts byte-identical reproduction against `tests/golden/baseline.json`. This guarantees the fix did **not** regress the `initialize=False` path used by every other notebook.

### Pending debugging follow-up

- [ ] Re-run [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) against the fixed code.
- [ ] Diff the new `initializations_nmi.png` and `initializations_rewards.png` against the archived pre-fix versions.
- [ ] Quantify the difference: for each `init_weights` setting, report the change in (a) final mean reward, (b) episodes-to-convergence (some chosen threshold), (c) final NMI.
- [ ] If the `Initializations_test` writeup exists in any draft, add a footnote acknowledging the pre-fix bug and citing this entry.

---

## Bug 2 — `TempNetMultiAgentEnv.get_actions` runs the NMI inner loop once per outer-loop iteration
- status: todo
- type: task
- id: rl_signaling.legacy_bugs_log.temp_net_get_actions_nested_nmi_loop
- last_checked: 2026-05-08
<!-- content -->
**Severity:** Medium
**File (pre-refactor):** `environment.py` line ~330 (inside `TempNetMultiAgentEnv.get_actions`)
**File (post-refactor):** [rl_signaling/env.py:733-760](rl_signaling/env.py#L733-L760) — symptom preserved on the deprecated path; not yet fixed. The variable shadowing flagged in earlier framings was renamed away during the refactor (inner loop variable is now `j`), but the nested-loop structure that actually causes the duplicate writes was kept.
**Status:** Open. The legacy `TempNetMultiAgentEnv` is deprecated as of Phase 5; this bug exists only on the legacy path, which the experiment notebooks still use until they migrate to `MultiAgentEnv`. The new `MultiAgentEnv.step_signal` computes NMI in a separate, single-pass loop after all signals are chosen, so it does not have this bug.

### Symptom

For each signal phase, `signal_information_history[i]` is appended to `n_agents` times instead of once per agent. After `N` episodes with two agents, the history has `2 * N` entries rather than `N`.

### Root cause

The current code (post-refactor) at [rl_signaling/env.py:733-760](rl_signaling/env.py#L733-L760):

```python
def get_actions(self, observations):
    actions = []
    available_actions = self.get_available_actions()
    for i, (agent, obs) in enumerate(zip(self.agents, observations)):
        action = agent.get_action(obs, available_actions)
        actions.append(action)

        if self.step_type == "signal":
            if obs not in self.signal_usage[i]:
                self.signal_usage[i][obs] = np.zeros(self.n_signaling_actions)
            self.signal_usage[i][obs][action] += 1
            # Compute and record mutual information of signals
            for j in range(self.n_agents):                                   # ← inner loop INSIDE outer
                _, normalized_mutual_info = compute_mutual_information(self.signal_usage[j])
                self.signal_information_history[j].append(normalized_mutual_info)
        ...
```

The inner `for j in range(self.n_agents)` is nested inside the outer per-agent loop. So during one signal phase the inner loop fires `n_agents` times, and each firing appends NMI for **every** agent — leaving `n_agents` writes to each agent's history per signal phase instead of one. With two agents, that's a duplicate write per signal phase.

Earlier framings of this bug attributed it to variable shadowing — the inner loop used to be written `for i in range(self.n_agents)`, rebinding the outer `i`. The refactor renamed the inner variable to `j`, removing the shadowing, but did **not** restructure the loop to fire only once per signal phase. The shadowing was incidental; the inner loop's position inside the outer loop is the actual cause and is what would need to change to fix the symptom (e.g. lift the NMI computation into a second, single-pass loop after the per-agent action loop, mirroring `MultiAgentEnv.step_signal`).

### Experimental impact

- **None on numerics inside the agent.** `compute_mutual_information` is a pure function with no side effects on agent state. The bug only inflates the *length* of `signal_information_history`. Q-tables, urns, action selections, and rewards are all unaffected.
- **Saved CSVs from `temp_simulation_function` (TD-learning) ARE affected.** The Phase 3 audit corrected this. The TD-learning rows in [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb) extract `Agent_X_Initial_NMI = np.mean(info_hist[:10])` and `Agent_X_NMI = np.mean(info_hist[-100:])`. With the 2× inflation, those slices cover the first 5 episodes (not 10) and the last 50 episodes (not 100). The averages are still well-defined and the qualitative direction of the metric is preserved, but the effective sample size for these summary statistics is half what was intended. The mid-curve `_avg_reward` columns are unaffected because they're derived from `rewards_history`, not the inflated history.
- **Cosmetic on plots that use the full series directly.** When `signal_information_history` is plotted directly (e.g. in `plot_simulation_summary` panel 2 with `n_episodes` as the x-axis upper bound), the x-axis count would be inflated 2×.

### Why it doesn't need an urgent fix

The new `MultiAgentEnv.step_signal` does not have the bug. As notebooks migrate from the deprecated `TempNetMultiAgentEnv` to `MultiAgentEnv`, the bug becomes irrelevant. Until then, it's a known quirk of the deprecated path that doesn't affect any reported experimental result.

### Pending debugging follow-up

- [ ] Verify experimentally that `len(signal_information_history[i]) == 2 * n_episodes` on a small TempNetMultiAgentEnv run, confirming the diagnosis.
- [ ] Decide whether to patch the deprecated path (low priority) or just migrate the notebooks to the canonical API (preferred, planned).

---

## Bug 3 — Missing `import sys` in `utils.py` error paths
- status: done
- type: task
- id: rl_signaling.legacy_bugs_log.utils_missing_import_sys
- last_checked: 2026-05-08
<!-- content -->
**Severity:** Medium
**File (pre-refactor):** `utils.py` (the `plot_reward_vs_cost` and `plot_nmi_vs_cost` functions referenced `sys.stderr` without importing `sys`)
**File (post-refactor):** [rl_signaling/plotting.py](rl_signaling/plotting.py) — fixed in Phase 1
**Status:** Fixed.

### Symptom

If either `plot_reward_vs_cost(df)` or `plot_nmi_vs_cost(df)` was called with a DataFrame missing the `Signal_Cost_A0` column (or the corresponding y-axis column), the function tried to print an error to `sys.stderr` and would itself crash with `NameError: name 'sys' is not defined` — masking the real diagnostic.

### Root cause

```python
def plot_reward_vs_cost(df, ...):
    if 'Signal_Cost_A0' not in df.columns or 'Agent_0_final_reward' not in df.columns:
        print(f"Error: ...", file=sys.stderr)   # ← sys not imported
        return
```

The original `imports.py` did not import `sys`, and `utils.py` relied entirely on `from imports import *`. So `sys` was never in scope.

### Experimental impact

- **None on saved results.** The error path only fires when a DataFrame is malformed. In normal operation the path is never taken, so the missing import was latent.
- **Latent diagnostic loss.** Anyone debugging a malformed-DataFrame call would have seen `NameError` instead of the intended error message — a much harder failure to interpret.

### The fix

`import sys` was added to the module's explicit imports during Phase 1 (when `from imports import *` was replaced with explicit imports throughout). The error paths now print correctly.

### Verification

No dedicated test (the function only matters when its inputs are malformed, which is outside the scope of the unit suite). The fix is mechanical — a one-line `import sys` addition — and was caught by ruff's `F401` / explicit-import audit during Phase 1.

---

## Bug 4 — `QLearningAgent.__init__` only pre-seeds the signaling Q-table when `initialize=True`
- status: done
- type: task
- id: rl_signaling.legacy_bugs_log.qlearning_agent_action_q_table_init
- last_checked: 2026-05-09
<!-- content -->
**Severity:** Medium
**File (post-refactor):** [rl_signaling/agents.py:397-415](rl_signaling/agents.py#L397-L415) — fixed in Batch B on 2026-05-09 (symmetric pre-seed). See "Fix applied" section below.
**Status:** Fixed in Batch B on 2026-05-09. User confirmed the symmetric intent; the docstring previously said "pre-seed the signaling Q-table" was a refactor copy-paste artefact, not design intent.

### Symptom

When `QLearningAgent` is constructed with `initialize=True`, only the **signaling Q-table** (`q_table_signaling`) is pre-seeded with one-hot vectors. The **action Q-table** (`q_table_action`) is silently set to `{}`, indistinguishable from the `initialize=False` case at any subsequent observation point. The first call to `agent.get_action(state)` will lazy-initialize `q_table_action[state]` to `np.zeros(n_final_actions)`, exactly as it would for an `initialize=False` agent.

This is structurally identical to the pre-fix behavior of `UrnAgent.__init__` (see Bug 1), with one difference: the asymmetry is documented in the `QLearningAgent` docstring, whereas in `UrnAgent` the asymmetry was unintentional and was fixed during Phase 4 of the refactor.

### Root cause

```python
def __init__(self, ...):
    ...
    if initialize:
        self.q_table_signaling = create_initial_signals(
            n_observed_features=n_observed_features,
            n_signals=n_signaling_actions,
            n=initialization_weights[0],
            m=initialization_weights[1],
        )
        for state in self.q_table_signaling:
            self.signaling_counts[state] = np.zeros(self.n_signaling_actions)
    else:
        self.q_table_signaling = {}
    self.q_table_action: dict = {}                # ← always empty, regardless of `initialize`
```

The `q_table_action = {}` assignment sits outside the `if/else` block, so the action Q-table is reset to empty after every construction. Unlike Bug 1 (where the equivalent `action_urns = create_initial_signals(...)` line existed inside the `if initialize:` branch and was being overwritten), there is no `q_table_action = create_initial_signals(...)` call anywhere in the constructor — the action Q-table simply has no pre-seeding code path.

### Why it's hard to spot

Same three reasons as Bug 1, plus a fourth:

1. **No type, no test, no warning.** Constructing with `initialize=True` and observing `q_table_action == {}` is indistinguishable from `initialize=False` behavior unless the caller specifically inspects the dict.
2. **The signaling Q-table works.** `q_table_signaling` is correctly pre-seeded, so cursory inspection of "is initialization having an effect?" returns yes — but only on half the channel.
3. **`get_action` lazy-initializes.** Missing `q_table_action[state]` keys silently become `np.zeros(n_final_actions)` — the same default as `initialize=False`.
4. **The docstring says it's intentional.** The constructor's docstring explicitly states "If True, pre-seed the **signaling** Q-table with one-hot vectors" — singular, signaling only. So a reader checking the docstring would conclude the asymmetry is by design. Whether that aligns with the user's actual intent is the open question this entry surfaces.

### Experimental impact

- **Affected notebook:** [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) — the only notebook that constructs agents with `initialize=True`. If the user's intent matches the Bug 1 fix (both channels pre-seeded), this experiment is currently measuring the effect of biasing only the signaling Q-table while the action Q-table starts uniform on every run, regardless of `initialization_weights`. The same shape of partial measurement that Bug 1 caused for `UrnAgent` before its Phase 4 fix.
- **Affected saved figures:** [results/initializations_nmi.png](results/initializations_nmi.png) and [results/initializations_rewards.png](results/initializations_rewards.png) — these are already flagged as needing regeneration because of Bug 1; whether the regeneration should also include this fix depends on the user's clarification.
- **Unaffected:** every other notebook and CSV in `results/`. Those experiments use `initialize=False`.

### The fix (proposed; depends on user confirmation)

If the user wants both Q-tables pre-seeded (mirroring the Bug 1 fix for `UrnAgent`):

```python
if initialize:
    self.q_table_signaling = create_initial_signals(
        n_observed_features=n_observed_features,
        n_signals=n_signaling_actions,
        n=initialization_weights[0],
        m=initialization_weights[1],
    )
    self.q_table_action = create_initial_signals(
        n_observed_features=n_observed_features + 1,
        n_signals=n_final_actions,
        n=initialization_weights[0],
        m=initialization_weights[1],
    )
    for state in self.q_table_signaling:
        self.signaling_counts[state] = np.zeros(self.n_signaling_actions)
    for state in self.q_table_action:
        self.action_counts[state] = np.zeros(self.n_final_actions)
else:
    self.q_table_signaling = {}
    self.q_table_action = {}
```

The `n_observed_features + 1` argument to `create_initial_signals` for the action Q-table mirrors `UrnAgent.__init__` and is correct only for graphs where every agent has exactly one in-neighbour (the standard 2-agent fully-connected setup). For graphs with variable in-degree, the action-side observation length would vary across agents — this caveat is shared with `UrnAgent` and is not new to this entry, but should be flagged to the user before the fix lands.

If the user confirms the asymmetry is intentional, no code change is needed; instead, update the docstring to state explicitly that the action Q-table is intentionally not pre-seeded, and add a `Notes` section explaining the design choice (parallel with the `Notes` section already on `UrnAgent`).

### Verification (post-fix, whichever option lands)

- Add a unit test analogous to `tests/test_agents.py::test_urn_agent_initialize_true_seeds_action_urns` for `QLearningAgent`. Construct with `initialize=True, n_observed_features=1, n_signaling_actions=2, n_final_actions=4` and assert `q_table_action` has 4 entries (one per `(observation, received_signal)` pair) and each is a one-hot vector of length 4.
- Re-run the golden-run regression at `tests/test_golden.py` to confirm `initialize=False` runs are still byte-identical (the post-Phase-4 baseline used `initialize=False`, so this fix should not perturb it).

### Fix applied (2026-05-09, Batch B — symmetric pre-seed)

[rl_signaling/agents.py](rl_signaling/agents.py) `QLearningAgent.__init__` now pre-seeds **both** Q-tables when `initialize=True`, mirroring the Bug 1 fix on `UrnAgent`:

```python
self.q_table_signaling: dict
self.q_table_action: dict
if initialize:
    self.q_table_signaling = create_initial_signals(
        n_observed_features=n_observed_features,
        n_signals=n_signaling_actions,
        n=initialization_weights[0], m=initialization_weights[1],
    )
    self.q_table_action = create_initial_signals(
        n_observed_features=n_observed_features + 1,
        n_signals=n_final_actions,
        n=initialization_weights[0], m=initialization_weights[1],
    )
    for state in self.q_table_signaling:
        self.signaling_counts[state] = np.zeros(self.n_signaling_actions)
    for state in self.q_table_action:
        self.action_counts[state] = np.zeros(self.n_final_actions)
else:
    self.q_table_signaling = {}
    self.q_table_action = {}
```

The constructor docstring was updated to match: "pre-seed both the signaling Q-table and the action Q-table" (the previous wording said only signaling). Visit-count tables are pre-allocated for every pre-seeded state, matching the lazy-init contract used elsewhere in `_select_action`.

A new unit test `test_q_learning_initialize_true_seeds_both_q_tables` mirrors the UrnAgent assertion: `n_observed_features=1, n_signaling_actions=2, n_final_actions=4`, asserts `len(q_table_signaling)==2`, `len(q_table_action)==4`, every entry is a one-hot vector, and `signaling_counts` / `action_counts` cover every pre-seeded state. Full suite reports 61 passed (was 60); golden-run regression unchanged because `initialize=False` is the path it exercises.

The same `n_observed_features + 1` caveat as `UrnAgent` applies: the action-table key length assumes every agent has exactly one in-neighbour. For variable-in-degree graphs the action-side observation length would vary across agents — out of scope for this bug, shared with `UrnAgent`.

### Pending debugging follow-up

- [x] ~Confirm with the user whether the docstring's "signaling only" wording reflects the actual intent or was carried over inadvertently from the refactor.~ — confirmed: refactor copy-paste artefact.
- [x] Re-run [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) against the fixed code and diff against the pre-fix run (Phase 6 verification). — done in the 2026-05-09 follow-up; figures regenerated against the Bug 9 fix.
- [x] Note: this bug was masked by Bug 5; with Bug 5 also fixed in this batch (Option B migration), Bug 4's effect is now observable in the QLearning block of [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb).
- [x] Note (2026-05-09 follow-up): the Bug-4 fix alone was insufficient. The QLearning leg of `Initializations_test.ipynb` collapsed to the random-action baseline regardless of `init_weights` because of a separate dtype bug — see Bug 9. The Bug 4 symmetric pre-seed is a *prerequisite* for the experiment to be well-formed (both Q-tables must be pre-seeded for `init_weights` to be load-bearing on both signaling and action), but the Bug 9 dtype fix is what makes the pre-seed survive even one TD update. Both fixes are needed; neither alone is sufficient.

### Post-fix observation (Phase 6, 2026-05-09)

- New unit test `tests/test_agents.py::test_q_learning_initialize_true_seeds_both_q_tables` passes; full pytest suite reports 61 passed (was 60).
- In-memory inspection during the migration smoke test confirmed `q_table_action` is populated with 4 one-hot entries (one per `(obs, received_signal)` pair) when constructed with `n_observed_features=1, n_signaling_actions=2, n_final_actions=4, initialize=True`.
- Visit-count pre-allocation matches the golden-run baseline (`initialize=False` is unchanged → byte-identical reproduction against `tests/golden/baseline.json`, asserted by `test_golden`).
- The action-side initialization effect on convergence will appear in the regenerated `initializations_*.png` figures from the QLearning block of [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) (see Bug 5 post-fix observation for the figure-level result).

---

## Bug 5 — `Initializations_test.ipynb` overwrites `env.agents`, silently dropping the `initialize=True` state
- status: done
- type: task
- id: rl_signaling.legacy_bugs_log.initializations_test_env_agents_overwrite
- last_checked: 2026-05-09
<!-- content -->
**Severity:** High — invalidates the **entire** experiment the notebook claims to run.
**File (post-refactor):** [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb), rewritten in Batch B on 2026-05-09. See "Fix applied" section below.
**Status:** Fixed in Batch B on 2026-05-09 via Option B (canonical-API migration).

### Symptom

The notebook is named "Initializations_test" and the README describes its purpose as "Effect of urn/Q-table initialization strategies." It loops over `init_weights ∈ [[1,0],[1,1],[5,1],[100,1]]` and saves two figures (`results/initializations_rewards.png`, `results/initializations_nmi.png`) ostensibly comparing convergence across initialization strengths.

In reality, every iteration of the loop runs the **same** `initialize=False` configuration. The four labeled curves in each saved figure differ only in run-to-run randomness — there is no initialization effect.

### Root cause

```python
for init_weights in tqdm([[1,0],[1,1],[5,1],[100,1]]):
    env = NetMultiAgentEnv(
        ...
        agent_type=QLearningAgent,
        initialize=True, initialization_weights=init_weights,    # ← env constructs agents with init=True
        graph=G,
    )

    env.agents = [                                               # ← then overwrites them
        QLearningAgent(
            n_signaling_actions=n_signaling_actions,
            n_final_actions=n_final_actions,
            exploration_rate=0.9652628633727897,
            exploration_decay=0.9998122815486062,
            min_exploration_rate=1e-10,
            choice='ucb',
            # ← no `initialize=...`, so defaults to initialize=False
        ) for _ in range(n_agents)
    ]
    ...
```

The `NetMultiAgentEnv` constructor honors `initialize=True` and pre-seeds the agents' Q-tables. Immediately afterwards the notebook **replaces** `env.agents` with a fresh list of `QLearningAgent` instances constructed without `initialize=...` — so the new agents default to `initialize=False`. The pre-seeded Q-tables from the env constructor are garbage-collected.

The overwrite pattern itself is intentional — the notebook injects tuned hyperparameters (`exploration_rate=0.965…`, `choice='ucb'`) that the `NetMultiAgentEnv` constructor doesn't expose as kwargs. The bug is that the override doesn't preserve `initialize=True, initialization_weights=init_weights, n_observed_features=...`.

### Why it's hard to spot

1. **The variable `init_weights` is used as the dict key for results** (`rewards_histories[str(init_weights)] = rewards_history`), so the saved figures DO have four labeled curves — they just don't reflect what the labels claim.
2. **Run-to-run variance produces visibly different curves** in each iteration even when the configuration is identical, so a casual look at the figure suggests the variation is meaningful.
3. **The section header reads "# Urn Agent"** but the code constructs `QLearningAgent`. UrnAgent is imported but never used in the experimental loop. So a reader checking "is this testing the urn-init bug?" sees no UrnAgent code anywhere — but assumes the experiment must be doing what its filename says.

### Experimental impact

- **Affected figures:** [results/initializations_nmi.png](results/initializations_nmi.png) and [results/initializations_rewards.png](results/initializations_rewards.png). Pre-fix and post-Bug-1-fix re-runs of this notebook would both produce **noise**, not a convergence-vs-init-strength comparison.
- **Affected LEGACY_BUGS_LOG entries:**
  - **Bug 1's "affected notebook" claim is wrong.** Bug 1 is about `UrnAgent.__init__`, but this notebook does not construct any UrnAgent in its experimental loop. So Bug 1's predicted impact ("strong-init runs should show stronger and faster convergence than the pre-fix runs") cannot be validated by re-running this notebook unless Bug 5 is fixed first.
  - **Bug 4 is similarly masked.** Even if `QLearningAgent.__init__` were fixed to pre-seed both Q-tables, the override would still throw the pre-seeded state away.

### The fix (proposed)

Either:

**Option A — preserve initialization in the override.** Add the init kwargs to the manual `QLearningAgent(...)` construction:

```python
env.agents = [
    QLearningAgent(
        n_signaling_actions=n_signaling_actions,
        n_final_actions=n_final_actions,
        exploration_rate=0.9652628633727897,
        exploration_decay=0.9998122815486062,
        min_exploration_rate=1e-10,
        choice='ucb',
        initialize=True,
        initialization_weights=init_weights,
        n_observed_features=n_features,        # match the env's observation shape
    ) for _ in range(n_agents)
]
```

**Option B — drop the override and use `agent_kwargs`.** The canonical [rl_signaling.env.MultiAgentEnv](rl_signaling/env.py) constructor accepts `agent_kwargs={...}` so tuned hyperparameters can be threaded through without manual replacement. Migrate the notebook to `MultiAgentEnv` + `run_simulation` (matching the pattern in [notebooks/basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb)).

Option B also removes the deprecated-API dependency. Recommended.

Either option also requires fixing the section header "# Urn Agent" — either rename it to "# Q-Learning Agent" or add a parallel UrnAgent loop (which is what the section header originally promised).

### Verification (post-fix)

- After fixing, the four `init_weights` curves should visibly diverge: stronger init weights (e.g. `[100, 1]`) should converge faster and to higher NMI than weak init (`[1, 0]`).
- Diff the new `initializations_*.png` files against the archived pre-fix versions to demonstrate Bug 5 was real (post-fix curves separate; pre-fix curves overlap modulo noise).

### Fix applied (2026-05-09, Batch B — Option B canonical migration)

[notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) was rewritten end-to-end. Highlights:

1. **API migration.** Imports switched from `NetMultiAgentEnv` + `simulation_function` to canonical `MultiAgentEnv` + `run_simulation`. The deprecated `env.agents = [...]` override pattern is gone — tuned hyperparameters are threaded through `agent_kwargs={...}` instead.
2. **Section split.** The notebook now contains **two** experimental loops, mirroring what the original "# Urn Agent" section header originally promised:
    - `# Urn Agent` block: drives `UrnAgent` over `init_weights ∈ [[1,0], [1,1], [5,1], [100,1]]`. Saves to `results/initializations_urn_rewards.png` and `results/initializations_urn_nmi.png`.
    - `# Q-Learning Agent` block: drives `QLearningAgent` over the same `init_weights`, preserving the tuned hyperparameters from the prior code (`exploration_rate=0.965…`, `choice='ucb'`, etc.). Saves to `results/initializations_rewards.png` and `results/initializations_nmi.png` (existing filenames preserved for back-compat with the README's narrative).
3. **Paired comparison.** Each iteration of the loop now resets `np.random.seed(0)` and `random.seed(0)` so the four init-weight curves see identical RNG state — any difference in the curves is now attributable to `init_weights` only.
4. **Greyscale plots.** The two greyscale-styled cells at the bottom of the notebook now reference `qlearning_rewards_histories` / `qlearning_nmi_histories` (matching what the legacy code visualized).

`pytest tests/` reports 61 passed (60 + the new Bug 4 unit test). A 200-episode smoke test of both blocks ran end-to-end without errors and showed `init=[100,1]` and `init=[1,0]` already producing different last-5 reward means at the QLearning leg, confirming that initialization is now load-bearing.

The `n_observed_features=1` agent_kwarg matches the per-agent observation length (each agent sees one feature per `agents_observed_variables = {0:[0], 1:[1]}`), so the action-side urn / Q-table is keyed by 2-tuples (own observation + received signal) consistent with the symmetric Bug 1 / Bug 4 pre-seed.

### Pending debugging follow-up

- [x] ~Confirm with the user whether the section was meant to test QLearning, Urn, or both.~ — both, per Phase 5 plan recommendation.
- [x] ~Apply Option A or B as confirmed by the user.~ — Option B applied.
- [x] Re-run the notebook and regenerate the four figures — done in Phase 6, see Post-fix observation below.
- [x] Cross-validate Bug 1's predicted impact and Bug 4's predicted impact in the regenerated UrnAgent / QLearning blocks respectively — done; impact direction differs from prediction (see below).

### Post-fix observation (Phase 6, 2026-05-09)

`jupyter nbconvert --execute --inplace` ran the migrated notebook end-to-end in 98 s (`n_episodes=30000 × 4 init_weights × 2 agent types = 240 000` episodes total). All four figures regenerated; pre-fix versions backed up to `/tmp/rl_signaling_prefix_backup/` for diff.

**Pre-fix vs post-fix (QLearning rewards):**
- Pre-fix `initializations_rewards.png`: all four labeled curves overlap and converge sharply to reward ≈ 1.0 within the first ~500 episodes — the four labels are visually indistinguishable except for run-to-run noise. Consistent with Bug 5's symptom: same `initialize=False` configuration × 4 runs.
- Post-fix `initializations_rewards.png`: all four curves now bounce in a noisy band centered ~0.20–0.30 (random-action baseline for `n_final_actions=4`). The four labels are still close to each other but each curve traces a distinct trajectory — a sign that initialization is now actually load-bearing on the QLearning leg.

**Post-fix UrnAgent rewards (new figure):** the four curves separate decisively:
- `[1, 0]` → stuck at reward ≈ 0.25 throughout (strong purely-asymmetric prior cannot rebalance in 30 k episodes).
- `[1, 1]` → reward ≈ 0.85.
- `[5, 1]` → reward ≈ 0.85, plateauing around episode 5 000.
- `[100, 1]` → reward climbs from ≈ 0.20 at start to ≈ 0.95–1.0 by episode 20 000 (largest count → strongest pull toward the pre-seeded hot action, which on this random canonical game's seed happens to align with one of the four optimal `(obs, signal) → action` cells; the urn's positive-only updates let the wrong-cell agent rebalance).

**Post-fix UrnAgent NMI (new figure):** even more striking separation:
- `[1, 0]` → NMI ≈ 1.0 from episode ~100 onwards (perfect signaling).
- `[1, 1]` → NMI collapses to ≈ 0.05 (the symmetric `[1, 1]` prior produces near-uniform signaling — agents barely differentiate signals across observations).
- `[5, 1]` → NMI ≈ 0.93.
- `[100, 1]` → NMI ≈ 0.90 (slightly lower than `[5, 1]`, because the strong pre-seed locks onto a brittle protocol).

The `[1, 0]` UrnAgent case is particularly illuminating: NMI ≈ 1.0 (agents agree on a reliable signaling code) but reward ≈ 0.25 (the code happens to map to the wrong action on the receiver side). The fix surfaces a real scientific finding: **strong communication can coexist with poor task performance when the action prior is misaligned with the game's optimal action map.**

**Post-fix QLearning NMI (2026-05-09 first pass, pre-Bug-9):** all four curves spiked briefly (NMI ≈ 0.7 in the first ~100 episodes, from the one-hot pre-seed's natural separation) then collapsed to ≈ 0 by episode 500. This was originally interpreted as "the constant `α = 0.1` Q-update in conjunction with UCB exploration cannot maintain the pre-seeded signaling code." That interpretation was incomplete — the dominant cause turned out to be the int-dtype Bug 9 (see entry below), not constant-α TD-decay. After the Bug 9 fix, all four QLearning curves reach reward 1.0 by mid-run and NMI 0.93–0.98 by end. The four `init_weights` are now monotone in early NMI (`[1,1]: 0.02 < [1,0]: 0.18 < [5,1]: 0.24 < [100,1]: 0.62`) — the pre-seed survives long enough to provide an early signaling structure that converges to the equilibrium.

**Direction vs prediction:** the original LEGACY_BUGS_LOG Bug 1 prediction was "strong-init runs should show stronger and faster convergence than the pre-fix runs." The post-Bug-9 data confirms this for QLearning: stronger pre-seeds produce higher early NMI, even though all four converge to the same equilibrium by mid-run. For UrnAgent the picture is more interesting — strong arbitrary one-hot priors *hurt* convergence in the `[1, 0]` case because the urn's positive-only update rule preserves the (misaligned) bijection forever, locking the agent at NMI ≈ 1.0 / reward ≈ 0.25. The prediction was based on the assumption that the prior is informed (pointing at the optimal action), which is not what `create_initial_signals` actually does — it picks an arbitrary one-hot. With this clarified, the regenerated figures are honest: pre-seeded action priors are *biased* (toward an arbitrary action), not *informed* (toward the optimal action). For QLearning they accelerate the early signaling structure; for UrnAgent under `[1, 0]` they permanently lock in the (potentially misaligned) bijection. The asymmetry between agents is a real scientific finding, not a fix-correctness issue.

**Verification status:**
- 4 figures regenerated, sized 99 KB / 131 KB / 275 KB / 322 KB (post-fix), all visibly distinct from pre-fix.
- `jupyter nbconvert` exit code 0; no errors during 240 k-episode run.
- `pytest tests/` reports 61 passed.
- Unit-test cross-check: the new `test_q_learning_initialize_true_seeds_both_q_tables` confirms Bug 4's pre-seed is now in place.

---

## Bug 6 — `Run_Simulations.ipynb` writes `*_complex.csv` but `plotting_results.ipynb` reads `*_complex_randomized.csv`
- status: done
- type: task
- id: rl_signaling.legacy_bugs_log.complex_randomized_csv_filename_mismatch
- last_checked: 2026-05-09
<!-- content -->
**Severity:** High — the canonical "complex" / "general games" figures cannot be regenerated from the codebase as-is.
**Files:**
- Producer: [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb), "More Complex Model" section. Writes `urnagent_results_complex.csv`, `qlearning_results_complex.csv`, `td_learning_results_complex.csv`.
- Consumer: [notebooks/plotting_results.ipynb](notebooks/plotting_results.ipynb), "General Urns" / "General Games" sections. Reads `urnagent_results_complex_randomized.csv`, `qlearning_results_complex_randomized.csv`, `td_learning_results_complex_randomized.csv`.
- Existing files: only `*_complex_randomized.csv` versions remain in [results/](results/) after the Batch B fix; the orphaned `*_complex.csv` files were deleted.

**Status:** Fixed in Batch B on 2026-05-09 via Option A (restored randomized action sizes; producer renamed to match). See "Fix applied" section below.

### Symptom

The README "Reproducing the figures" section instructs the user to run the experiment notebooks, then run `plotting_results.ipynb` to regenerate figures. Following that procedure today silently fails for the "complex" experiments: `Run_Simulations.ipynb` produces `*_complex.csv` files, but `plotting_results.ipynb` reads `*_complex_randomized.csv` files. The plotting notebook's reads succeed (the `_randomized` files happen to be in `results/` already, from some earlier run) but the figures it produces reflect **stale, no-longer-regenerable data**, not the output of the complex blocks the user just ran.

### Root cause

The `_randomized` filename convention reflects an experiment design where `n_signaling_actions` and `n_final_actions` are drawn per-iteration from `np.random.randint(2, 10)` — varied across rows. The current `Run_Simulations.ipynb` "complex" blocks fix `n_signaling_actions=4, n_final_actions=8` and write to `*_complex.csv` (no `_randomized` suffix).

`plotting_results.ipynb` "General Games" markdown explicitly documents the variant it expects to read:

> - n_signaling_actions = np.random.randint(2, 10)
> - n_final_actions = np.random.randint(2, 10)

So the plotting notebook is matched to a different (older / parallel) version of `Run_Simulations.ipynb` that randomizes action sizes. That version is not in the current codebase. The `_complex_randomized.csv` files in [results/](results/) are orphans of a workflow that no current notebook regenerates.

The `qlearning_results_complex.csv`, `td_learning_results_complex.csv`, `urnagent_results_complex.csv` files in [results/](results/) are the inverse: produced by the current notebook but not consumed by anything.

### Experimental impact

- **Affected figures:** every `*_complex_randomized_*.png` in [results/](results/) — `Roth-Erev_complex_randomized_*.png`, `Q-learning_complex_randomized_*.png`, `TD-learning_complex_randomized_*.png`. These figures correspond to the saved `_complex_randomized.csv` data, which is not currently regenerable end-to-end.
- **Saved fixed-action `*_complex.csv` data is orphaned:** plotting_results doesn't consume it. So the data Run_Simulations produces today has no plotting consumer.

### The fix (proposed)

Two viable options, depending on what the user actually wants the experiment to be:

**Option A — restore the randomized-action workflow.** Reintroduce the per-iteration `n_signaling_actions = np.random.randint(2, 10)` / `n_final_actions = np.random.randint(2, 10)` draw inside `Run_Simulations.ipynb`'s complex blocks, and rename the output files to `*_complex_randomized.csv`. This matches what `plotting_results.ipynb` and the saved figures reflect.

**Option B — re-anchor plotting_results on fixed action sizes.** Update `plotting_results.ipynb` to read `*_complex.csv` (the current Run_Simulations output), and update its markdown narrative to match. Regenerate the `*_complex_*.png` figures from the fixed-action CSV. This means the saved `_complex_randomized` PNGs and CSVs are formally retired.

Option A preserves the current saved figures' meaning (more interesting experiment design — varying complexity); Option B is a smaller code change but discards the existing figures.

### Verification (post-fix)

- After whichever option lands, the README's "Reproducing the figures" recipe must run end-to-end and reproduce the saved PNGs (modulo seed differences) on a fresh checkout.
- Stale CSV files (the un-consumed side of whichever option is chosen) should be deleted from `results/` once the user is comfortable.

### Fix applied (2026-05-09, Batch B — Option A randomized action sizes restored)

[notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb) cells 15 (UrnAgent complex), 17 (QLearning complex), and 19 (TD complex) were rewritten:

1. **Per-iteration randomized action sizes.** `n_signaling_actions = np.random.randint(2, 10)` and `n_final_actions = np.random.randint(2, 10)` now happen inside `run_all_cases_for_iteration(...)`, replacing the previous hard-coded `n_signaling_actions=4, n_final_actions=8`. The two values are drawn once per iteration and threaded into `run_single_case(...)` as new positional arguments, so all four `(full_information, with_signals)` cases for a given iteration share the same action sizes — matching the schema documented in `plotting_results.ipynb`'s "General Games / General Urns" markdown.
2. **Output filenames renamed.** Each block now writes `urnagent_results_complex_randomized.csv` / `qlearning_results_complex_randomized.csv` / `td_learning_results_complex_randomized.csv` — the filenames `plotting_results.ipynb` already consumes.
3. **Orphans deleted.** `results/urnagent_results_complex.csv`, `results/qlearning_results_complex.csv`, `results/td_learning_results_complex.csv` (the orphaned 8 000-row CSVs that no consumer read) were removed.

`pytest tests/` reports 61 passed (60 + the Bug 4 unit test).

**Caveat — Colab dependency.** The notebook's cell 4 imports `from google.colab import drive` and sets `dump_path = '/content/drive/My Drive/...'`. To re-run the complex blocks locally, the user must (a) flip `simulate=True` in the UrnAgent block (cell 15 — currently gated `simulate=False` for compute reasons) and (b) replace cell 4's Colab `dump_path` with a local path such as `dump_path = '../results/'`. The fix itself does not require Colab; only the existing notebook scaffolding does.

**Caveat — multiprocessing seeding.** The action-size draws happen inside the worker subprocess and depend on the worker's startup RNG state, the same as the existing `game_dicts` construction (LEGACY_ERRORS_LOG cross-cutting finding #2). Population-level statistics are unaffected; individual rows are not row-reproducible from `iteration` alone. Migrating to `numpy.random.SeedSequence().spawn()` would close this gap and is tracked separately in the verify-reproducibility task.

### Pending debugging follow-up

- [x] ~Confirm with the user which option matches their intent — Option A (restore randomized) or Option B (retire randomized).~ — Option A confirmed.
- [x] ~Apply the chosen fix.~ — done.
- [ ] Regenerate the affected figures (Phase 6 verification — requires the Colab/local-path swap noted above; deferred to user-driven re-run).
- [x] ~Clean up orphaned CSV files in `results/`.~ — done.

### Post-fix observation (Phase 6, 2026-05-09)

- Producer (`Run_Simulations.ipynb` cells 15/17/19) now writes `urnagent_results_complex_randomized.csv`, `qlearning_results_complex_randomized.csv`, `td_learning_results_complex_randomized.csv` — exactly the filenames `plotting_results.ipynb` already reads under "General Urns" / "General Games". The producer/consumer chain is structurally consistent.
- The three orphaned `*_complex.csv` files (`urnagent_results_complex.csv`, `qlearning_results_complex.csv`, `td_learning_results_complex.csv`) were deleted from `results/`.
- Per-iteration `n_signaling_actions = np.random.randint(2, 10)` and `n_final_actions = np.random.randint(2, 10)` draws are threaded into `run_single_case(...)` consistently across all three blocks; the four `(full_information, with_signals)` cases for a given iteration share the same per-iteration action sizes.
- `pytest tests/` reports 61 passed; `tests/test_golden.py` is unaffected because it only exercises canonical-API paths.
- **Full figure regeneration is gated on Colab/local-path swap.** `Run_Simulations.ipynb` cell 4 imports `from google.colab import drive` and assigns `dump_path = '/content/drive/My Drive/...'`; running the complex blocks locally requires (a) flipping `simulate=True` in cell 15 (UrnAgent block, currently gated to limit compute) and (b) replacing cell 4 with `dump_path = '../results/'`. This is a notebook-scaffolding caveat, not a fix-correctness issue.
- **Multiprocessing seeding caveat.** The new action-size draws use the worker's startup RNG state — same pattern as the existing `game_dicts` construction. Population statistics are unaffected; individual rows are not row-reproducible from `iteration` alone. Migrating to `numpy.random.SeedSequence().spawn()` is tracked separately in `todo.verify_reproducibility`.

---

## Bug 7 — `Parameter_Optimization_wchoices.ipynb` is missing imports for several names it uses
- status: done
- type: task
- id: rl_signaling.legacy_bugs_log.parameter_optimization_missing_imports
- last_checked: 2026-05-09
<!-- content -->
**Severity:** Medium — Restart-and-Run-All fails immediately at the parameter-ranges cell.
**File:** [notebooks/Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb), cell 3 (the imports cell).
**Status:** Fixed in Batch A on 2026-05-09. See "Fix applied" section below.

### Symptom

Restart kernel → Run All → `NameError: name 'Categorical' is not defined` at the first `param_ranges = {... "choice": Categorical([...]) ...}` cell. The notebook does not currently produce its documented output on a fresh kernel. The user has been running it with implicit setup state from earlier sessions.

### Root cause

The imports cell loads the `rl_signaling` package, `random`, `matplotlib.pyplot`, `networkx`, `numpy`, `pandas`, `tqdm` — but does not load the names actually used by the four optimization functions:

- `Categorical`, `Real`, `Integer` — from `skopt.space`.
- `Optimizer` — from `skopt`.
- `Parallel`, `delayed` — from `joblib`.
- `multiprocessing` — stdlib.
- `datetime` — from `datetime`.

`skopt` (scikit-optimize) is not in the runtime's base image on either Colab or a stock pip install, so it also requires `!pip install scikit-optimize` before the import.

### Experimental impact

- **None on saved CSVs/figures** — the orphaned hyperparameter-search outputs in [results/](results/) (`q_opt_*.png`, `td_opt_*.png`) were produced by a prior run when the user had the imports loaded interactively.
- **Reproducibility hazard:** a fresh user following the README cannot reproduce the hyperparameter search without manually patching the imports cell.

### The fix (proposed)

Add the missing imports to cell 3:

```python
import multiprocessing
from datetime import datetime
from joblib import Parallel, delayed
from skopt import Optimizer
from skopt.space import Categorical, Integer, Real
```

Add `scikit-optimize` to `pyproject.toml`'s optional `dev` extras (or document the manual install in the notebook's first cell, since this notebook is research-only).

### Verification (post-fix)

- Restart kernel → Run All on the canonical Q-Learning section completes without NameError.
- Document the dependency in [README.md](README.md) Setup section if `scikit-optimize` is not added to extras.

### Fix applied (2026-05-09, Batch A)

Cell 3 of [notebooks/Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb) now imports `multiprocessing`, `datetime.datetime`, `joblib.Parallel`/`delayed`, `skopt.Optimizer`, and `skopt.space.Categorical`/`Integer`/`Real`. `scikit-optimize>=0.9` was added to `[project.optional-dependencies] dev` in `pyproject.toml` (line 41), so `pip install -e ".[dev]"` now provisions the dependency. `pytest tests/` still reports 60 passed.

### Pending debugging follow-up

- [x] ~Confirm whether this notebook is meant to be re-runnable end-to-end (research artifact) or if the saved CSVs / `q_opt_*.png` figures are the deliverable and the notebook is essentially "documentation of what was run."~ — implicitly resolved as re-runnable: imports added rather than banner added.
- [x] ~If re-runnable: apply the import fix and add `scikit-optimize` to extras.~ — done.
- [ ] If documentation only: add a markdown banner at the top stating "this notebook is a research log; the saved hyperparameter search CSVs are the deliverable." — not done; supersede only if a future session decides the research-log framing is preferred.

### Post-fix observation (Phase 6, 2026-05-09)

- `scikit-optimize 0.10.2` installed in the project venv via `pip install -e ".[dev]"` would now provision the dependency on a fresh checkout (verified: `from skopt.space import Categorical, Integer, Real; from skopt import Optimizer` succeeds in the active venv).
- Cell 3 of `notebooks/Parameter_Optimization_wchoices.ipynb` carries the five new imports (`multiprocessing`, `datetime.datetime`, `joblib.Parallel`/`delayed`, `skopt.Optimizer`, `skopt.space.Categorical`/`Integer`/`Real`).
- Restart-and-Run-All gating verified by inspection: the first `param_ranges = {... Categorical([...]) ...}` cell will resolve `Categorical` from the imported namespace. End-to-end Bayesian search re-run is gated on Colab/local-path swap (cell 4) plus user-controlled compute scale (`n_trials`); not executed in this session.
- `pytest tests/` reports 61 passed.

---

## Bug 8 — `plotting_results.ipynb` final cell uses Q-learning filename prefix for TD-learning data
- status: done
- type: task
- id: rl_signaling.legacy_bugs_log.plotting_results_td_filename_typo
- last_checked: 2026-05-09
<!-- content -->
**Severity:** Low — saved-file naming only; the figure content itself is correct.
**File:** [notebooks/plotting_results.ipynb](notebooks/plotting_results.ipynb), the final code cell.
**Status:** Fixed in Batch A on 2026-05-09. See "Fix applied" section below.

### Symptom

The TD-learning "General Games" regression plot is saved with `filename_prefix='Q-learning_complex_randomized'`. So the saved files are `Q-learning_complex_randomized_regression_signals_*.png`, the same names produced by the Q-learning General Games block earlier in the notebook. Running both blocks back-to-back **overwrites** the Q-learning regression PNGs with TD-learning content.

### Root cause

```python
# Last cell of plotting_results.ipynb
plot_regression(
    td_learning_complex,
    'Agent_0_NMI',
    'Agent_0_final_reward',
    filename_prefix='Q-learning_complex_randomized',   # ← copy-paste from Q-learning section
)
```

A copy-paste from the Q-learning block; the `filename_prefix` was not updated to `'TD-learning_complex_randomized'`.

### Experimental impact

- **Affected files:** `results/Q-learning_complex_randomized_regression_signals_*.png` — last regenerated by the TD-learning section, so they actually show TD-learning regression. The intended `TD-learning_complex_randomized_regression_signals_*.png` files are never produced (despite the directory containing the histogram-only `TD-learning_complex_randomized_*.png` files from `plot_all_histograms`).
- **No data loss** — the underlying CSV data is fine; it's purely a naming/save-path issue.

### The fix (proposed)

```python
plot_regression(
    td_learning_complex,
    'Agent_0_NMI',
    'Agent_0_final_reward',
    filename_prefix='TD-learning_complex_randomized',
)
```

### Verification (post-fix)

- Re-run the cell; verify `results/TD-learning_complex_randomized_regression_signals_*.png` exists and that `results/Q-learning_complex_randomized_regression_signals_*.png` reflects Q-learning data only (regenerate from the Q-learning cell if needed).

### Fix applied (2026-05-09, Batch A)

Final code cell of [notebooks/plotting_results.ipynb](notebooks/plotting_results.ipynb) (cell index 37) now reads:

```python
plot_regression(td_learning_complex,'Agent_0_NMI','Agent_0_final_reward',filename_prefix='TD-learning_complex_randomized')
```

`pytest tests/` still reports 60 passed. Re-run + figure regeneration is gated on Bug 6 resolution (Phase 6 verification) since both fixes share a re-run of the General Games block.

### Pending debugging follow-up

- [x] ~Apply the one-line fix.~ — done.
- [ ] Re-run the relevant cells (gated on Bug 6 resolution; runs in Phase 6 verification).
- [ ] Verify the file naming and content match (Phase 6).

### Post-fix observation (Phase 6, 2026-05-09)

- The corrected cell 37 of `notebooks/plotting_results.ipynb` (the only edit) is in place:

  ```python
  plot_regression(td_learning_complex, 'Agent_0_NMI', 'Agent_0_final_reward', filename_prefix='TD-learning_complex_randomized')
  ```

- Figure regeneration is gated on Bug 6's complex producer re-run, which itself is gated on the Colab/local-path swap (see Bug 6 post-fix observation). The expected post-fix outcome — `TD-learning_complex_randomized_regression_signals_*.png` exists and `Q-learning_complex_randomized_regression_signals_*.png` shows actual Q-learning content — has been verified as a code-change consistency check but the actual PNG regeneration is deferred.

---

## Bug 9 — `_generate_hot_vectors` returns int64; pre-seeded Q-tables silently truncate fractional TD updates
- status: done
- type: task
- id: rl_signaling.legacy_bugs_log.generate_hot_vectors_int_dtype
- last_checked: 2026-05-09
<!-- content -->
**Severity:** High — invalidates the entire QLearning leg of `notebooks/Initializations_test.ipynb`.
**File (post-refactor):** [rl_signaling/games.py:105-112](rl_signaling/games.py#L105-L112) — `_generate_hot_vectors`. Fixed in the 2026-05-09 follow-up; see "Fix applied" section below.
**Status:** Fixed.

### Symptom

In the QLearning leg of `notebooks/Initializations_test.ipynb` (post-Bug-4 / post-Bug-5 state), all four `init_weights` curves bounce around the random-action baseline (reward ≈ 0.20–0.30 = 1/`n_final_actions` for `n_final_actions=4`) for the entire 30 000-episode run. NMI starts at the pre-seed level (0.18–0.62 depending on `init_weights`) and collapses to ≈ 0 by episode 500. Even `init_weights=[100, 1]` shows no lock-in and no slow drift — the bias is washed out almost immediately. The pre-fix figures `results/initializations_rewards.png` and `results/initializations_nmi.png` reflect this degenerate regime.

By contrast, the UrnAgent leg of the same notebook (using the same `_generate_hot_vectors`-derived pre-seed) shows the expected separation across `init_weights`. So the symptom is QLearning-specific.

### Root cause

`_generate_hot_vectors` builds one-hot vectors from the `n` and `m` arguments without specifying a dtype:

```python
def _generate_hot_vectors(n_signals, n=1, m=0):
    return [
        np.array([n if i == j else m for i in range(n_signals)])
        for j in range(n_signals)
    ]
```

When the caller passes integer arguments — the standard `init_weights=[100, 1]`, `[5, 1]`, `[1, 1]`, `[1, 0]` cases all do — `np.array(...)` infers `dtype=int64`. Pre-seeded `q_table_signaling` and `q_table_action` therefore become int64 arrays, while the lazy-init path (executed when `initialize=False`) creates float64 arrays via `np.zeros(...)` — a silent dtype inconsistency between the two construction paths.

The constant-α TD update at [rl_signaling/agents.py:474](rl_signaling/agents.py#L474) and [:492](rl_signaling/agents.py#L492) is

```python
self.q_table_signaling[state][signal] += learning_rate * td_error  # learning_rate = 0.1
```

On a float64 array, this produces $Q \leftarrow Q + 0.1 \cdot (r - Q)$ — the canonical TD update. On an int64 array, NumPy in-place addition of a float into an int element converts the float increment back to int (truncation toward zero) before storing. Concretely:

- `Q[cold] = 1`, `r = 0`: $1 + 0.1 \cdot (0 - 1) = 0.9$ → cast to int = **0**. `Q[cold]` collapses to zero in **a single update**.
- `Q[hot] = 100`, `r = 0`: $100 + 0.1 \cdot (0 - 100) = 90.0$ → 90 (correct), but the cumulative trajectory diverges from the closed form because every step truncates 0.x of decay. Empirically, `Q[hot]` reaches 0 at $n \approx 30$ instead of the closed-form $n \approx 100$ where it crosses 1.0.
- For `init_weights=[1, 0]`: hot starts at 1, cold at 0; one reward-0 update on hot → 0, identical to a never-pre-seeded cell.
- For `init_weights=[1, 1]`: every cell starts at 1; one reward-0 update → 0; uniform pre-seed dies on the first visit.

So the pre-seed is erased far faster than the Hypothesis-1 ("constant-α TD-decay") closed form predicts, regardless of `init_weights` magnitude. The dtype bug is the dominant cause of the QLearning failure; H1 is real but subordinate.

### Why it was hard to spot

1. **Silent dtype promotion.** `np.array([100, 1])` infers int64 without warning. There is no error path; the truncation is invisible at the call site.
2. **The lazy-init path works.** `initialize=False` agents create float64 Q-tables via `np.zeros(...)`. So every notebook except `Initializations_test.ipynb` is unaffected, and the same `QLearningAgent` class behaves correctly in 5 of 6 notebooks.
3. **UrnAgent is incidentally robust to the bug.** Under the canonical game's integer-valued rewards (`{0, 1}`), UrnAgent's update `urn[a] = max(0, urn[a] + reward)` produces the same numerical values whether stored as int or float. Because `Initializations_test.ipynb` is structurally a Roth-Erev-style integer-reward experiment for UrnAgent, the UrnAgent figures are correct in both pre- and post-fix code. The bug masquerades as "QLearning-specific failure" rather than the structural dtype inconsistency it actually is.
4. **The Hypothesis-1 narrative is plausible.** The closed form $Q_n = r + (Q_0 - r)(1 - \alpha)^n$ predicts that with `α = 0.1`, `Q_0 = 100`, `r = 0`, `Q_n \approx 0.5` after 50 visits — a small remnant. So "the pre-seed decays" is a true statement, just not the dominant cause of the empirical zero-by-episode-500 collapse. A debugging session that landed on H1 alone would not have caught the dtype bug.

### Experimental impact

- **Affected notebook:** [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) — its QLearning block specifically. The UrnAgent block was unaffected (rewards in {0, 1} → int and float arithmetic are equivalent for `max(0, urn + reward)`).
- **Affected saved figures:** [results/initializations_rewards.png](results/initializations_rewards.png) and [results/initializations_nmi.png](results/initializations_nmi.png) — the QLearning figures regenerated on 2026-05-09 (post-Bug-4, post-Bug-5) reflect the int-truncated trajectory, not the intended experiment.
- **Unaffected:** every other notebook and CSV in `results/`. They use `initialize=False`; lazy-init creates float64 arrays via `np.zeros` / `np.ones`, bypassing the bug entirely. The golden-run baseline at [tests/golden/baseline.json](tests/golden/baseline.json) is uses `initialize=False` and is byte-identical pre- and post-fix.
- **Latent risk now closed:** an `initialize=True` costly-signaling experiment under the pre-fix code would have triggered the same int-truncation problem on UrnAgent (because $c_i \in (0, 0.5)$ produces fractional rewards). No current notebook exercises this path, but the dtype fix closes it before it ever fires.

### Three-hypothesis investigation (2026-05-09 follow-up session)

The investigation in `TODO_WORKFLOW.md::todo.investigate_qlearning_initialization` posed three hypotheses for QLearning's failure: H1 (TD-decay erases pre-seed), H2 (symmetric pre-seed adds noise), H3 (tuned hyperparameters wash out the bias). All three were empirically tested at notebook-scale (5000 episodes per cell, 5 seeds, paired comparison via `np.random.seed(0); random.seed(0)`). Results:

| Variant | rew@end ([1,0] / [1,1] / [5,1] / [100,1]) | nmi@end |
|---|---|---|
| V1 BASELINE — current code | 0.27 / 0.27 / 0.21 / 0.27 | 0 / 0 / 0 / 0 |
| V2 default hyperparameters (H3) | 0.27 / 0.27 / 0.21 / 0.27 (byte-identical to V1) | 0 / 0 / 0 / 0 |
| V3 asymmetric pre-seed (H2) | 0.39 / 0.36 / 0.46 / 0.39 | 0 / 0 / 0 / 0 |
| **V4 float dtype (Bug 9 fix)** | **1.00 / 1.00 / 1.00 / 1.00** | **0.91 / 0.89 / 0.83 / 0.81** |
| V5 positive-only-clamped Q (counter-test) | 0.62 / 0.27 / 0.25 / 0.25 | 0.31 / 0 / 0.99 / 0.99 |

H3 contributed nothing (V2 = V1 byte-identical, because UCB's high early bonus dominates ε in either schedule until counts equalize). H2 contributed a small amount (~0.1–0.2 reward improvement under asymmetric pre-seed, but NMI still 0). The dominant cause was the int dtype: V4 alone restores the experiment for all four `init_weights`, with `[1, 1]` showing higher variance because `[1, 1]` is by construction a no-bias pre-seed. V5 (positive-only-clamp) was brittle and is rejected.

### The fix

[rl_signaling/games.py:105-119](rl_signaling/games.py#L105-L119) `_generate_hot_vectors` now passes `dtype=np.float64` explicitly:

```python
def _generate_hot_vectors(
    n_signals: int, n: float = 1, m: float = 0
) -> list[NDArray[np.float64]]:
    return [
        np.array(
            [n if i == j else m for i in range(n_signals)],
            dtype=np.float64,
        )
        for j in range(n_signals)
    ]
```

The return-type annotation and the `SignalUrns` type alias at the top of the module were updated accordingly (`NDArray[np.int_]` → `NDArray[np.float64]`).

### Verification

- **Unit tests added** to [tests/test_agents.py](tests/test_agents.py):
  - `test_pre_seeded_q_tables_are_float_dtype` — asserts `np.issubdtype(vec.dtype, np.floating)` on every pre-seeded entry of `QLearningAgent.q_table_signaling`, `QLearningAgent.q_table_action`, `UrnAgent.signaling_urns`, `UrnAgent.action_urns`.
  - `test_q_learning_pre_seed_bias_persists_through_zero_reward_decay` — constructs a `QLearningAgent(initialize=True, init_weights=(100, 1))`, drives 50 reward-0 updates on each of `Q[hot]` and `Q[cold]` for the same state, and asserts the closed-form predictions match (`Q_hot ≈ 100·0.9⁵⁰ ≈ 0.5154`, `Q_cold ≈ 0.005154`) to relative tolerance `1e-6`, and that `Q_hot - Q_cold > 0.4` (the pre-seed advantage that survives the decay; pre-fix this gap was zero because both cells truncated to 0).
- **Golden-run regression** ([tests/test_golden.py](tests/test_golden.py)) still byte-identical because `initialize=False` is the path it exercises.
- `pytest tests/` reports **63 passed** (was 61).
- `jupyter nbconvert --execute --inplace` ran the notebook end-to-end against the fix; pre-fix QLearning PNGs backed up to `/tmp/rl_signaling_bug9_backup/`. Pre- and post-fix UrnAgent PNGs are visually identical (byte-identical numerics under integer rewards), confirming the fix is QLearning-specific in effect.

### Post-fix observation (2026-05-09)

A 30 000-episode reproduction (matching the notebook's protocol with paired seeds) yields:

**QLearning, post-fix:**

| init_weights | rew @ ep 1–100 | rew @ mid | rew @ end | nmi @ ep 1–100 | nmi @ mid | nmi @ end |
|---|---|---|---|---|---|---|
| [1, 0]    | 0.36 | 1.00 | 1.00 | 0.18 | 0.97 | 0.98 |
| [1, 1]    | 0.38 | 1.00 | 1.00 | 0.02 | 0.96 | 0.98 |
| [5, 1]    | 0.31 | 1.00 | 1.00 | 0.24 | 0.93 | 0.96 |
| [100, 1]  | 0.25 | 1.00 | 1.00 | 0.62 | 0.92 | 0.96 |

All four `init_weights` now reach optimal reward (1.00) by mid-run and sustain it. Early NMI (`ep 1–100`) is monotone in pre-seed strength (`[1,1]: 0.02 < [1,0]: 0.18 < [5,1]: 0.24 < [100,1]: 0.62`), confirming the pre-seed encodes initial signaling structure that survives the dtype fix. Late NMI converges to ≈ 0.96–0.98 across all four — QLearning learns the equilibrium independently of initialization, and the pre-seed accelerates the early signaling structure rather than determining the final state.

**UrnAgent, post-fix:** byte-identical to the 2026-05-09 figures (integer rewards make int and float arithmetic equivalent for the urn update), so the existing UrnAgent figures are valid as-is.

**Cross-reference to the original Bug 1 prediction.** The original prediction ("strong-init runs should show stronger and faster convergence") was based on the assumption of *informed* priors (pointing at the optimal action). The empirical post-Bug-9 data shows that for QLearning, the priors are *biased* (toward an arbitrary action) and survive long enough to provide an early signaling structure but do not determine the final convergence point — which is good. For UrnAgent, biased priors *do* determine the final state in the `[1, 0]` case (because the urn's positive-only update rule preserves the bijection forever when cold cells start at zero). This asymmetry is now visible in the regenerated figures and is a real scientific finding about agent-update structure, not a fix-correctness issue.

### Pending debugging follow-up

- [x] ~Diagnose why QLearning fails to lock into pre-seeded equilibria.~ — root cause identified as the int dtype bug; H1/H2/H3 magnitudes documented in the table above.
- [x] ~Implement the chosen fix.~ — done.
- [x] ~Add unit tests asserting dtype and bias-persistence.~ — done (2 new tests, suite 61 → 63).
- [x] ~Regenerate `results/initializations_*.png` against the fix.~ — done; pre-fix backed up to `/tmp/rl_signaling_bug9_backup/`.
- [x] ~Update Bug 4 / Bug 5 post-fix observations to point at Bug 9 as the actual cause of the QLearning collapse.~ — done (this commit).

---

## Adding a new bug entry

When a new bug is identified during the debugging follow-up, append a section using this template:

````markdown
## Bug N — {{One-line title}}
- status: {{todo | in-progress | done}}
- type: task
- id: rl_signaling.legacy_bugs_log.{{short_slug}}
- last_checked: {{YYYY-MM-DD}}
<!-- content -->
**Severity:** {{High | Medium | Low}}
**File (pre-refactor):** {{path:lines}}
**File (post-refactor):** {{path:lines}} — {{fixed in Phase X | not yet fixed}}
**Status:** {{Fixed | Open | Investigating}}

### Symptom
{{What the user / experimenter would observe.}}

### Root cause
{{Code snippet + explanation.}}

### Experimental impact
{{Which notebooks / CSVs / figures are affected, if any.}}

### The fix
{{Code snippet of the corrected version, or "not yet fixed".}}

### Verification
{{Unit test reference + golden-run reference.}}

### Pending debugging follow-up
- [ ] {{specific re-run / diff / measurement}}
````
