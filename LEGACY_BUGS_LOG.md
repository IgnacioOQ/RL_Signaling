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
| 1 | `UrnAgent.__init__` silently never pre-seeds `action_urns` | **High** | Fixed in Phase 4 (golden-run gated) | `notebooks/Initializations_test.ipynb` only |
| 2 | `TempNetMultiAgentEnv.get_actions` shadows the outer loop variable when computing NMI | **Medium** | Not yet fixed (legacy-only path; deprecated env) | None of the saved CSVs (NMI history length only, not values) |
| 3 | `utils.py` referenced `sys.stderr` without `import sys` | **Medium** | Fixed in Phase 1 | None — only fires on error paths in `plot_reward_vs_cost` / `plot_nmi_vs_cost` |

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

## Bug 2 — `TempNetMultiAgentEnv.get_actions` shadows the outer loop variable
- status: todo
- type: task
- id: rl_signaling.legacy_bugs_log.temp_net_get_actions_shadowing
- last_checked: 2026-05-08
<!-- content -->
**Severity:** Medium
**File (pre-refactor):** `environment.py` line ~330 (inside `TempNetMultiAgentEnv.get_actions`)
**File (post-refactor):** [rl_signaling/env.py](rl_signaling/env.py) — same code preserved on the deprecated path; not yet fixed.
**Status:** Open. The legacy `TempNetMultiAgentEnv` is deprecated as of Phase 5; this bug exists only on the legacy path, which the experiment notebooks still use until they migrate to `MultiAgentEnv`. The new `MultiAgentEnv.step_signal` does not have this bug.

### Symptom

For each signal phase, `signal_information_history[i]` is appended to `n_agents` times instead of once per agent. After `N` episodes with two agents, the history has `2 * N` entries rather than `N`.

### Root cause

```python
def get_actions(self, observations):
    actions = []
    available_actions = self.get_available_actions()
    for i, (agent, obs) in enumerate(zip(self.agents, observations)):
        action = agent.get_action(obs, available_actions)
        actions.append(action)
        if self.step_type == "signal":
            ...
            for i in range(self.n_agents):                                   # ← shadows outer i
                _, normalized_mutual_info = compute_mutual_information(self.signal_usage[i])
                self.signal_information_history[i].append(normalized_mutual_info)
        ...
```

The inner `for i in range(self.n_agents)` rebinds `i`, so the NMI for **every** agent gets appended every time the outer loop iterates. With two agents, that's a duplicate write per signal phase.

### Experimental impact

- **None on numerics.** `compute_mutual_information` is a pure function with no side effects on agent state. The bug only inflates the *length* of `signal_information_history`. Q-tables, urns, action selections, and rewards are all unaffected.
- **None on saved CSVs.** None of the saved CSVs in `results/` include the full `signal_information_history` series — they record final-NMI scalars (which are taken as `history[-1]` and reflect the most recent computation, which is correct).
- **Cosmetic on plots.** When `signal_information_history` is plotted directly (e.g. in some notebook cells), the x-axis count would be inflated 2×.

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
