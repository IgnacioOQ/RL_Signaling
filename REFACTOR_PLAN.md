# Refactor Plan

Multi-session plan for cleaning up RL_Signaling. Each phase is independently mergeable; verification step is mandatory before moving on. A fresh agent should be able to resume from any phase using only this document.

## Operating rules for any agent picking this up

1. Read this file end-to-end before doing anything.
2. Work on the `refactor` branch. Do not commit to `main` directly.
3. Do **not** stage or commit unless the user explicitly asks. Edit files; the user reviews `git diff` and commits.
4. Run the smoke test (see "Verification" under each phase) before declaring a phase done.
5. After completing a phase, update its status in this file (`Pending` → `Done`) and capture any new findings in the "Notes from execution" section of that phase.
6. If a phase produces decisions not captured here (e.g. you discover something during execution and choose an approach), write the decision into this file before moving on.

## Decisions already locked in

| Decision | Choice |
|---|---|
| Branch model | Single long-lived `refactor` branch, commits per phase |
| License | MIT, copyright "Anonymous" |
| Python floor | `>=3.10` |
| Dependency strategy | Unpinned ranges in `pyproject.toml` + pinned `requirements.txt` |
| Gitignore scope | Standard Python ignores only (no `.DS_Store` / `.vscode/` patterns) |
| Saved results | Regenerable — free to rename `cannonical` → `canonical` in filenames |
| Final layout | `rl_signaling/` package + `notebooks/` + `tests/` + `results/` |
| Urn-agent init bug | Fix in **Phase 4** alongside other agent changes (preserves checked-in CSV/PNG numerics until then) |
| Docstrings + type hints | Apply in **Phase 3.5** after the module split (avoids redoing) |

## Audit findings (Phase 0, frozen)

These are the substantive findings from the read-through. Treat them as the ground truth that motivates the later phases.

### Bug: `UrnAgent.__init__` always wipes `action_urns`

In `agents.py` (pre-refactor line numbers from the original flat module):

```python
if initialize:
    self.signaling_urns = create_initial_signals(...)
    self.action_urns    = create_initial_signals(...)   # initialized here
else:
    self.signaling_urns = {}
self.action_urns = {}   # <-- OUTSIDE the if/else; always overwrites
```

`UrnAgent.action_urns` is silently never initialized, even when `initialize=True`. Address this in **Phase 4** with a golden-run diff to surface impact on saved results.

### Truly dead code (already deleted in Phase 1)

- `utils.plot_hist` — superseded by `plot_histograms_with_kde`.
- `utils.create_directed_graph` — never called; notebooks build `nx.DiGraph` inline.
- Module-level scratch in `environment.py` (lines 7–19 in original) and `simulation_function.py` (lines 6–15 in original).
- `unittest` / `unittest.mock` imports in old `imports.py`.

### `utils.py` is a junk drawer

590 lines, 21 functions, three concerns mixed:

| Concern | Functions | Notebook usage |
|---|---|---|
| Game/signal generators | `create_random_game`, `create_random_canonical_game`, `generate_unique_dicts`, `generate_hot_vectors`, `create_initial_signals` | 2 + 5; some internal-only |
| Information theory | `compute_entropy`, `compute_mutual_information` | internal-only |
| Plotting + post-processing | 11 functions (`plot_*`, `compare_payoffs`, `calculate_proportions`, `smooth`, `count_negative_nmi`, `calculate_reward_difference`) | mostly only in `plotting_results.ipynb` |

Splits cleanly into `games.py`, `info_theory.py`, `plotting.py` in Phase 3.

### Duplication inside `agents.py`

The exploration-strategy block (`egreedy` / `softmax` / `ucb`) is written **three times**:
- inside `QLearningAgent.get_signal`
- inside `QLearningAgent.get_action`
- inside `TDLearningAgent.get_action`

Should consolidate into a `select_action(q_values, counts, exploration_rate, choice, available_actions=None)` helper in Phase 4.

### Two parallel pipelines because `TDLearningAgent` has a different interface

| Method | UrnAgent / QLearningAgent | TDLearningAgent |
|---|---|---|
| signal selection | `get_signal(state)` | `get_action(state, available_actions)` (role inferred from `step_type`) |
| action selection | `get_action(state)` | same |
| signal update | `update_signals(state, signal, reward)` | `update(state, action, reward, next_state, done)` |
| action update | `update_actions(state, action, reward)` | same |

Why `NetMultiAgentEnv` / `TempNetMultiAgentEnv` and `simulation_function` / `temp_simulation_function` both exist. Resolved in Phase 4 (unify agent interface) → Phase 5 (unify env + runner).

### `simulation_function` vs `temp_simulation_function`

The episode loops differ in 3 places (signal-step semantics, update call, costly-signaling logic), but the **plotting block at the bottom is ~120 lines duplicated near-verbatim**. Plotting must be extracted into `plotting.py` before unifying the loops.

### Notebook → module dependency map

| Notebook | UrnAgent | QLearningAgent | TDLearningAgent | NetEnv | TempNetEnv | sim_fn | temp_sim_fn |
|---|---|---|---|---|---|---|---|
| `basic_unit_test` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Run_Simulations` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Initializations_test` | ✓ | ✓ | – | ✓ | – | ✓ | – |
| `Parameter_Optimization_wchoices` | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `Final_Costly_Signaling_Run_Simulations` | ✓ | ✓ | – | ✓ | ✓ | ✓ | ✓ |
| `plotting_results` | – | – | – | – | – | – | – |

`plotting_results.ipynb` only consumes saved CSVs — easiest to migrate.

### `cannonical` typo: 62 occurrences across notebooks + 17 saved files in `plots_and_results/`

Plus references inside `agents.py`/`environment.py`/`simulation_function.py` are gone post-Phase-1, but the typo persists in notebooks and result filenames. Full rename happens in Phase 2.

## Target layout (post-refactor)

```
rl_signaling/
  __init__.py
  agents.py        # UrnAgent, QLearningAgent, TDLearningAgent, BaseAgent
  env.py           # MultiAgentEnv (unified)
  simulation.py    # run_simulation()
  games.py         # canonical & random game generators
  info_theory.py   # MI / NMI
  plotting.py      # all plot_* helpers
notebooks/
  basic_unit_test.ipynb
  run_simulations.ipynb
  initializations_test.ipynb
  parameter_optimization.ipynb
  costly_signaling_run.ipynb
  plotting_results.ipynb
tests/
  test_agents.py
  test_env.py
  test_info_theory.py
  test_smoke.py
results/             # renamed from plots_and_results/
pyproject.toml
README.md
LICENSE
.gitignore
requirements.txt
REFACTOR_PLAN.md     # this file
```

---

## Phase status

| Phase | Status | Branch checkpoint |
|---|---|---|
| 0. Audit | **Done** | findings frozen above |
| 1. Hygiene | **Done** | uncommitted on `refactor` |
| 2. Typo rename | **Done** | uncommitted on `refactor` |
| 3. Module split | Pending | – |
| 3.5. Docstrings + type hints | Pending | – |
| 4. Unified agent interface (+ urn-init bug fix) | Pending | – |
| 5. Unified Env + runner | Pending | – |
| 6. Tests | Pending | – |
| 7. Docs polish | Pending | – |

---

## Phase 1 — Hygiene (Done)

### Scope

- Add `LICENSE`, `.gitignore`, `pyproject.toml`, `requirements.txt`.
- Delete dead code: `plot_hist`, `create_directed_graph`, module-level scratch blocks, unused `unittest` imports.
- Replace `from imports import *` and `from utils import *` with explicit imports across modules and notebooks.
- Delete `imports.py`.

### Files touched

| File | Change |
|---|---|
| `LICENSE` | New (MIT, "Anonymous") |
| `.gitignore` | New |
| `pyproject.toml` | New |
| `requirements.txt` | New |
| `imports.py` | Deleted |
| `agents.py` | Star imports → explicit imports |
| `environment.py` | Star imports → explicit; module-level scratch deleted |
| `simulation_function.py` | Star imports → explicit; module-level scratch deleted; `n_agents`/`n_features`/etc. moved from module-level defaults to inline literal defaults |
| `utils.py` | Star imports → explicit; deleted `plot_hist` and `create_directed_graph`; **fixed missing `import sys`** (latent bug in `plot_reward_vs_cost` / `plot_nmi_vs_cost` error paths) |
| All 6 notebooks | Setup cell rewritten to per-notebook minimal explicit imports |

### Verification (run before declaring done)

```bash
python3 -c "
import sys, types
sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))
import utils, agents, environment, simulation_function
print('All modules import cleanly.')
"
```

Expected output: `All modules import cleanly.`

### Notes from execution

- Latent bug found and fixed: `utils.plot_reward_vs_cost` / `plot_nmi_vs_cost` use `sys.stderr` but the original `imports.py` did not import `sys`.
- All notebook calls to `simulation_function` / `temp_simulation_function` already passed `n_agents` / `n_features` / `n_signaling_actions` / `n_final_actions` explicitly, so removing module-level defaults was safe.

---

## Phase 2 — Typo rename (Done)

### Scope

Rename every occurrence of `cannonical` → `canonical` in:

- Code (function names, variable names, comments) — verify Phase 1 left none, but recheck.
- All 6 notebooks (62 occurrences distributed; see audit table).
- `plots_and_results/` filenames (17 files; rename via `git mv` is fine here since rename is the actual intent).
- README references if any.

### Files to touch

```
plots_and_results/Q-learning_cannonical_Agent_0_NMI.png
plots_and_results/Q-learning_cannonical_Agent_0_avg_reward.png
plots_and_results/Q-learning_cannonical_Agent_0_final_reward.png
plots_and_results/Q-learning_cannonical_regression_signals_True_fullinfo_False.png
plots_and_results/Q-learning_cannonical_regression_signals_True_fullinfo_True.png
plots_and_results/QLearning_cannonical_costly_signal_Agent_0_NMI.png
plots_and_results/QLearning_cannonical_costly_signal_Agent_0_avg_reward.png
plots_and_results/QLearning_cannonical_costly_signal_Agent_0_final_reward.png
plots_and_results/Roth-Erev_cannonical_Agent_0_NMI.png
plots_and_results/Roth-Erev_cannonical_Agent_0_avg_reward.png
plots_and_results/Roth-Erev_cannonical_Agent_0_final_reward.png
plots_and_results/Roth-Erev_cannonical_costly_signal_Agent_0_NMI.png
plots_and_results/Roth-Erev_cannonical_costly_signal_Agent_0_avg_reward.png
plots_and_results/Roth-Erev_cannonical_costly_signal_Agent_0_final_reward.png
plots_and_results/Roth-Erev_cannonical_regression_signals_True_fullinfo_False.png
plots_and_results/Roth-Erev_cannonical_regression_signals_True_fullinfo_True.png
plots_and_results/TD-learning_cannonical_Agent_0_NMI.png
plots_and_results/TD-learning_cannonical_Agent_0_avg_reward.png
plots_and_results/TD-learning_cannonical_Agent_0_final_reward.png
plots_and_results/TD-learning_cannonical_regression_signals_True_fullinfo_False.png
plots_and_results/TD-learning_cannonical_regression_signals_True_fullinfo_True.png
plots_and_results/qlearning_results_cannonical.csv
plots_and_results/qlearning_results_cannonical_costly_signal.csv
plots_and_results/td_learning_results_cannonical.csv
plots_and_results/urnagent_results_cannonical.csv
plots_and_results/urnagent_results_cannonical_costly_signal.csv
plots_and_results/urnagent_results_cannonical_costly_signal (1).csv
plots_and_results/q_opt_canonical.png   # already correct — leave alone
```

### Strategy

1. Grep verification (find every remaining occurrence):
   ```bash
   grep -RIn "cannonical" .
   ```
2. Edit code/notebooks via `Edit` tool with `replace_all=True`.
3. For result filenames, prefer `git mv` so the rename is tracked as a rename (not delete+add). Do **not** use bare `mv`.
4. Update any string literals in notebooks that reference the renamed result paths.

### Verification

```bash
grep -RIn "cannonical" . | grep -v REFACTOR_PLAN.md   # this file mentions it; ignore
# Should return zero results.
```

Plus the import smoke test from Phase 1.

### Notes from execution

- Total: 85 token replacements across the 6 notebooks (62 distinct lines).
- Both case variants present: `cannonical` (lowercase, in code/strings/filenames) and `Cannonical` (capitalized, in markdown headers).
- 27 files renamed under `plots_and_results/` via `git mv` (one filename had spaces and parens — `urnagent_results_cannonical_costly_signal (1).csv` — handled fine by shell glob expansion).
- String literals inside notebooks (e.g. `pd.read_csv(dump_path+'urnagent_results_cannonical.csv')`) and `filename_prefix='Roth-Erev_cannonical'` arguments were updated in lockstep with the filename renames, so notebook reads still resolve.
- All 6 notebooks parsed cleanly as JSON post-edit; module smoke test passes.

---

## Phase 3 — Module split (Pending)

### Scope

Reorganize the flat modules into the `rl_signaling/` package described in "Target layout."

### Steps

1. Create `rl_signaling/` directory with empty `__init__.py`.
2. Move modules:
   - `agents.py` → `rl_signaling/agents.py`
   - `environment.py` → `rl_signaling/env.py` (note: rename to `env.py` for brevity)
   - `simulation_function.py` → `rl_signaling/simulation.py` (rename for brevity)
3. Split `utils.py`:
   - `create_random_game`, `create_random_canonical_game`, `generate_unique_dicts`, `generate_hot_vectors`, `create_initial_signals` → `rl_signaling/games.py`
   - `compute_entropy`, `compute_mutual_information` → `rl_signaling/info_theory.py`
   - All `plot_*`, `compare_payoffs`, `calculate_reward_difference`, `calculate_proportions`, `smooth`, `count_negative_nmi` → `rl_signaling/plotting.py`
4. Make package importable: add the public surface to `rl_signaling/__init__.py`.
5. Mark internal helpers private (leading `_`):
   - `_generate_unique_dicts`, `_generate_hot_vectors` (called only by `create_initial_signals` / `create_random_canonical_game`)
   - `_compute_entropy` (called only by `compute_mutual_information`)
   - `_calculate_reward_difference` (called only by `compare_payoffs`)
6. Move notebooks into `notebooks/` directory.
7. Move `plots_and_results/` → `results/` (`git mv`).
8. Update notebook imports to use the new package: `from rl_signaling.agents import ...`, etc.
9. Update any path literals in notebooks/code that reference `plots_and_results/`.

### Verification

```bash
python3 -c "
import rl_signaling
from rl_signaling import agents, env, simulation, games, info_theory, plotting
print('Package OK')
"
```

Plus: launch one notebook (recommend `basic_unit_test.ipynb`) and run the first 3 cells. Imports must succeed.

### Risks

- Notebook JSON edits are noisy; preserve cell metadata.
- Path literals (e.g. `'./plots_and_results/'` defaults inside `utils.plot_histograms_with_kde` and similar) must be updated.

---

## Phase 3.5 — Docstrings + type hints (Pending)

### Scope

Add proper docstrings and type hints to the public API of every module in `rl_signaling/`.

### Style

- Numpy-style docstrings (matches what's already partially used in `agents.py`).
- `from __future__ import annotations` at the top of each module so PEP 604 union syntax (`int | None`) works on 3.10.
- Annotate function signatures and key class attributes; no need to annotate every local.
- Internal-only helpers get a one-line docstring; public functions get full Parameters / Returns sections.

### Verification

```bash
python3 -m pip install ruff && ruff check rl_signaling/ --select=D    # docstring lint
python3 -c "import rl_signaling; help(rl_signaling.agents.UrnAgent)"  # spot-check
```

---

## Phase 4 — Unified agent interface + urn-init bug fix (Pending)

### Scope

1. Define `BaseAgent` ABC in `rl_signaling/agents.py` with the canonical interface:
   ```
   get_signal(state) -> int
   get_action(state) -> int
   update_signals(state, signal, reward) -> None
   update_actions(state, action, reward) -> None
   ```
2. Refactor `TDLearningAgent` to match this interface (the current `get_action(state, available_actions)` and `update(state, action, reward, next_state, done)` shapes get split internally).
3. Extract the duplicated exploration-strategy block (`egreedy`, `softmax`, `ucb`) into a single helper:
   ```python
   def _select_action(q_values, counts, exploration_rate, choice, available_actions=None) -> int
   ```
   Replaces 3 near-identical blocks in `QLearningAgent.get_signal`, `QLearningAgent.get_action`, `TDLearningAgent.get_action`.
4. **Fix the urn-init bug** at `UrnAgent.__init__` — move the `self.action_urns = {}` assignment inside the `else` branch.

### Pre-flight: golden run

Before any changes, save a deterministic golden run:

```python
# scripts/golden_run.py — produces a JSON of (rewards_history, signal_usage, NMI) across all 3 agents at small n_episodes
import random, numpy as np, json
random.seed(0); np.random.seed(0)
# ... small n_agents=2 / n_episodes=200 sweep across UrnAgent / QLearningAgent / TDLearningAgent
# ... write to tests/golden/baseline.json
```

After Phase 4, re-run with the same seeds and diff against `baseline.json`. Acceptable diff: deterministic numerical drift only from the urn-init bug fix (which is the intended change). Anything else means the refactor altered behavior.

### Verification

- All notebooks still run without modification (since the agent classes keep their same names).
- Golden-run diff matches expected impact (urn-init change only).

---

## Phase 5 — Unified Env + runner (Pending)

### Scope

1. Collapse `MultiAgentEnv` (renamed from `NetMultiAgentEnv`) and `TempMultiAgentEnv` into a single class. The two-step semantics (signal step → action step) becomes a flag or implicit from a unified `step()` API.
2. Collapse `simulation_function` and `temp_simulation_function` into a single `run_simulation(env, n_episodes, ...)`.
3. Extract the ~120 lines of duplicated plotting logic from both old simulation functions into helpers in `rl_signaling/plotting.py` (e.g. `plot_simulation_summary(signal_usage, rewards_history, ...)`).
4. Keep the old class names (`NetMultiAgentEnv`, `TempNetMultiAgentEnv`, `simulation_function`, `temp_simulation_function`) as thin deprecated wrappers that emit a `DeprecationWarning`. Remove them in a later release once notebooks are migrated.

### Verification

- Golden-run diff matches Phase 4 baseline (new code reproduces old behavior).
- All notebooks run unmodified through the deprecated wrappers; one notebook (`basic_unit_test.ipynb`) is migrated to the new API as a reference example.

---

## Phase 6 — Tests (Pending)

### Scope

Convert `basic_unit_test.ipynb` into proper `pytest` tests under `tests/`:

- `tests/test_agents.py` — instantiation, signal/action selection within bounds, update math sanity.
- `tests/test_env.py` — environment init under each agent type, episode lifecycle invariants.
- `tests/test_info_theory.py` — `compute_mutual_information` on known cases (e.g. perfect correlation → NMI=1, independence → NMI=0).
- `tests/test_smoke.py` — end-to-end 100-episode run for each agent type, asserting rewards finite, signal-usage counts sum correctly, NMI in [0,1].
- `tests/test_golden.py` — load `tests/golden/baseline.json` and assert reproducibility against the post-refactor implementation (with the agreed urn-init bug fix accounted for).

### Verification

```bash
pip install -e .[dev]
pytest tests/ -v
```

All tests pass. Coverage on `rl_signaling/` ≥ 80% (informational; not a gating criterion).

---

## Phase 7 — Docs polish (Pending)

### Scope

- Add docstrings everywhere they're still missing.
- Optional: `docs/model.md` — formal writeup of the model (state, observations, signals, actions, payoff, costly extension), suitable for citing in a paper.
- Optional: `CITATION.cff` if the project will be cited.
- Update `README.md` to point at the new package layout.
- Add a short "Reproducing the figures" section to README pointing at the experiment notebooks → `plotting_results.ipynb`.

### Verification

- `README.md` instructions work end-to-end on a fresh checkout in a clean venv.

---

## Resume-here checklist for a new session

When picking this up in a new session:

1. `git status` — confirm you're on `refactor`. If not, `git checkout refactor`.
2. `git log main..HEAD --oneline` — see what's been committed.
3. Read this file's "Phase status" table.
4. Read the next pending phase's Scope + Strategy + Verification sections.
5. Run that phase's pre-condition smoke test (usually the import test from Phase 1).
6. Execute. Update the phase's status and "Notes from execution" when done.
