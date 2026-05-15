# Notebook Refactor Plan
- status: draft
- type: plan
- id: rl_signaling.notebook_refactor_plan
- description: Phased plan for migrating the six pre-refactor notebooks under `notebooks/` onto the canonical `rl_signaling` API (`MultiAgentEnv` + `run_simulation`), bringing them in line with the project's notebook conventions, and refreshing stale figures.
- label: [planning, notebooks, refactor]
- injection: informational
- volatility: initial_draft
- scope: project-specific
- last_checked: 2026-05-15
<!-- content -->

The six notebooks under [notebooks/](notebooks/) were authored before the seven-phase code refactor described in [REFACTOR_PLAN.md](REFACTOR_PLAN.md). They still call the legacy `NetMultiAgentEnv` / `TempNetMultiAgentEnv` classes and the legacy `simulation_function` / `temp_simulation_function` runners, which now emit `DeprecationWarning`. Two notebooks ([basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb) and [Initializations_test.ipynb](notebooks/Initializations_test.ipynb)) were already partially updated to the canonical API; the rest still target the deprecated surface and the original Colab/Drive workflow.

This plan moves every notebook to the canonical API, brings the surrounding scaffolding in line with the project conventions documented in [content/how-to/NOTEBOOK_WRITING_SKILL.md](../knowledge_base/content/how-to/NOTEBOOK_WRITING_SKILL.md) (KB), and regenerates the figures that the [REFACTOR_PLAN.md](REFACTOR_PLAN.md) Status section flags as stale.

**Execution model:** sequential phases. Phases 1–2 are mechanical (low risk) and can be done one notebook at a time. Phases 3–4 involve re-running experiments and overwriting saved artifacts (`results/*.csv`, `results/*.png`) — these are heavy and only one phase 4 sweep is needed.

**Reference documents:**
- [README.md](README.md) — canonical API surface and minimal example.
- [REFACTOR_PLAN.md](REFACTOR_PLAN.md) — finished seven-phase code refactor that produced the canonical API.
- [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md) — Bug 1 (UrnAgent action-urn init) invalidates [results/initializations_*.png](results/).
- KB skill `content/how-to/NOTEBOOK_WRITING_SKILL.md` — the authoritative checklist for what a "good" notebook looks like in this project.

---

## Goals

1. Every notebook imports and uses the canonical API (`MultiAgentEnv` + `run_simulation`) and emits no `DeprecationWarning`.
2. Every notebook satisfies the **Restart-and-Run-All** test on a fresh `rl_signaling` kernel, locally, without Colab.
3. Dual local / Colab execution where the original Colab workflow was load-bearing (parameter sweeps, costly-signaling sweeps) — gated by a single `RUNNING_LOCALLY` switch per the KB skill §6.
4. Stable cell IDs (`nbformat ≥ 4.5`) so future agent edits address cells by `id` rather than by index.
5. Stale figures in [results/](results/) caused by the UrnAgent initialization fix are regenerated.

## Non-goals

- No new experiments. The plan only re-runs what is already saved under `results/`.
- No re-derivation of the optimal hyperparameters in [Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb) — the saved best-params from the original sweep are kept.
- No commits. The agent prepares files; the user reviews diffs and commits.

---

## Current state (audit)

| Notebook | nbformat | Kernel | Legacy API used | Other gaps |
|---|---|---|---|---|
| [basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb) | 4.2 | `rl_signaling` | None — already on canonical API | No `SMOKE_TEST` flag; no markdown title/abstract |
| [Initializations_test.ipynb](notebooks/Initializations_test.ipynb) | 4.2 | `rl_signaling` | None — already on canonical API | nbformat needs bump to 4.5; greyscale block has no `plt.savefig` |
| [Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb) | 4.0 | `python3` | `NetMultiAgentEnv`, `TempNetMultiAgentEnv`, `simulation_function`, `temp_simulation_function` | Hard-mounts Colab Drive; `!git clone ANONYMIZED_REPO_URL`; no `RUNNING_LOCALLY` switch |
| [Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb) | 4.0 | `python3` | All four legacy entry points | Same Colab-only scaffolding; multiprocessing reseed pattern is fine but uncommented |
| [Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb) | 4.0 | `python3` | `NetMultiAgentEnv`, `simulation_function` (Roth–Erev block already retired in markdown) | Same Colab-only scaffolding |
| [plotting_results.ipynb](notebooks/plotting_results.ipynb) | 4.2 | `python3` | None (consumes CSVs via `rl_signaling.plotting`) | Kernel name mismatch (should be `rl_signaling`); nbformat needs bump |

The asymmetry — two notebooks already on the canonical API, four still on the legacy one — is the result of the seven-phase code refactor having updated `basic_unit_test.ipynb` and `Initializations_test.ipynb` as part of REFACTOR_PLAN's Phase 7 documentation pass, but not the other four.

---

## Target state

Each notebook follows the canonical setup template from the KB skill §6 (in order):

1. **Title + abstract** — markdown cell with the notebook's purpose in one paragraph.
2. **Env switch + paths** — `RUNNING_LOCALLY` flag plus `BASE_PATH` / `RESULTS_DIR` derivation. Local default: `RESULTS_DIR = Path("../results")` (the canonical results sink already used by [Initializations_test.ipynb](notebooks/Initializations_test.ipynb) and [plotting_results.ipynb](notebooks/plotting_results.ipynb)).
3. **Git-clone block** — guarded by `if not RUNNING_LOCALLY:`. Uses `os.chdir(...)` not `%cd`. Force-fresh clone per KB skill §5.
4. **Pip install block** — `subprocess.run(['pip', 'install', '-q', '-e', '.'])` inside the cloned repo, guarded by `if not RUNNING_LOCALLY:`. Avoids the `!pip install` magic that ignores `if` guards.
5. **Parameters cell** — every tunable knob at the top, including `SMOKE_TEST` and `SMOKE_TEST_N_ITER` / `SMOKE_TEST_N_EPISODES` for any notebook whose runtime is > 5 min on a developer laptop.
6. **Imports cell** — explicit, one per line, no wildcard from external libraries. Wildcard from the project's own `rl_signaling` package is acceptable per the skill, but every notebook in the audit currently uses explicit imports — keep that convention.
7. **Body** — compute / save / analyze sections, each preceded by a markdown header.
8. **Disconnect cell** — only for notebooks intended to run on Colab. Gated by `if AUTO_DISCONNECT and not RUNNING_LOCALLY:`.

Every notebook saves with `nbformat=4.5` and `kernelspec.name="rl_signaling"`.

---

## Legacy → canonical API mapping (use this as a refactor cheat-sheet)

| Legacy | Canonical | Notes |
|---|---|---|
| `from rl_signaling.env import NetMultiAgentEnv, TempNetMultiAgentEnv` | `from rl_signaling import MultiAgentEnv` | Single env class for all three agent types. |
| `from rl_signaling.simulation import simulation_function, temp_simulation_function` | `from rl_signaling import run_simulation` | Single runner; takes the env, not the env's constructor args. |
| `NetMultiAgentEnv(... initialize=False, agent_type=QLearningAgent, ...)` followed by manual `env.agents = [QLearningAgent(...)]` rebuild | `MultiAgentEnv(... agent_type=QLearningAgent, agent_kwargs={"exploration_rate": ..., "exploration_decay": ..., "choice": "ucb"}, ...)` | The `agent_kwargs` dict is forwarded to the agent constructor — no more `initialize=False` + manual rebuild. |
| `TempNetMultiAgentEnv(... learning_rate=..., exploration_rate=..., ...)` | `MultiAgentEnv(agent_type=TDLearningAgent, agent_kwargs={"learning_rate": ..., "gamma": ..., "choice": "ucb"}, ...)` | TD agent kwargs go in `agent_kwargs`. `env.max_actions` / `n_actions` no longer needs to be plumbed by hand. |
| `NetMultiAgentEnv(..., costly_signaling=True, ...)` with `effective_n_signaling_actions = n_signaling_actions + 1` and a hand-built agent list | `MultiAgentEnv(..., costly_signaling=True, agent_kwargs={...})` | The env now auto-appends the null-signal action when `costly_signaling=True`; pass the base `n_signaling_actions`. |
| `simulation_function(n_agents=..., n_features=..., ..., env=env, ...)` | `run_simulation(env, n_episodes=..., with_signals=..., plot=..., verbose=...)` | All the `n_*` / size arguments live on the env now. |
| `simulation_function(..., costly_signaling=True, signal_cost=[c, c])` | `run_simulation(env, ..., signal_cost=[c, c])` | `costly_signaling` is on the env; `signal_cost` is per-call. |
| `temp_simulation_function(...)` | `run_simulation(...)` | Same call shape. **Caveat:** see "TD divergence" below. |
| Return tuple: `(signal_usage, rewards_history, signal_information_history, nature_history, histories)` | Return tuple: `(signal_usage, rewards_history, nmi_history, histories, nature_history)` | **Order changed.** `histories` and `nature_history` swap. Notebooks that unpack with `_, rewards, nmi, _, _` are unaffected. Notebooks that unpack with names need to swap. |

### TD divergence caveat (from [README.md](README.md))

The canonical `MultiAgentEnv` + `run_simulation` flow with `TDLearningAgent` produces output that differs from `TempNetMultiAgentEnv` + `temp_simulation_function` on roughly 1 in 100 episodes, due to a subtle change in when `exploration_rate` decays relative to the action-phase `get_action` call. The Q-value math is unchanged. The agents `UrnAgent` and `QLearningAgent` are byte-identical across the two APIs.

**Consequence for the refactor:** TD-learning CSVs regenerated from the migrated notebooks will not be bit-identical to the saved [results/td_learning_results_*.csv](results/). The figures produced by [notebooks/plotting_results.ipynb](notebooks/plotting_results.ipynb) will be visually indistinguishable but not byte-identical. The plan acknowledges this; it does not attempt to preserve the old TD numerics.

---

## Phased plan

### Phase 0 — Tooling (this session)

**Deliverables:**

- [notebooks/NOTEBOOKS_README.md](notebooks/NOTEBOOKS_README.md) — short conventions doc summarizing what every notebook in this folder must look like, plus how to run them locally vs. on Colab.
- [notebooks/_tools/nb_migrate.py](notebooks/_tools/nb_migrate.py) — small Python utility (no third-party deps beyond `nbformat`) that:
  - Bumps `nbformat_minor` to 5.
  - Sets `kernelspec.name = "rl_signaling"` and `display_name = "Python (rl_signaling)"`.
  - Assigns a stable `id` to every cell missing one.
  - Reports legacy-API string matches (greps every code cell for `NetMultiAgentEnv`, `TempNetMultiAgentEnv`, `simulation_function`, `temp_simulation_function`, `!git clone`, `%cd`, `google.colab`) so the agent can confirm migration completion.
  - Validates round-trip JSON load per the KB skill §9.

The utility is intentionally **conservative**: it only performs metadata-level changes plus reporting. Source-level rewrites (the legacy → canonical mapping in the table above) are done per-notebook by hand or by `NotebookEdit` with `cell_id=...`, so the human reviewer can read every diff.

**Exit criterion:** `nb_migrate.py` runs cleanly on all six notebooks and prints a per-notebook legacy-API report.

### Phase 1 — Rename & metadata pass

**Deliverables:** renamed files with consistent ordering, all on nbformat ≥ 4.5, all on the `rl_signaling` kernel.

Proposed rename:

| Current | Renamed |
|---|---|
| `basic_unit_test.ipynb` | `01_basic_unit_test.ipynb` |
| `Initializations_test.ipynb` | `02_initializations_test.ipynb` |
| `Run_Simulations.ipynb` | `03_run_simulations.ipynb` |
| `Parameter_Optimization_wchoices.ipynb` | `04_parameter_optimization.ipynb` |
| `Final_Costly_Signaling_Run_Simulations.ipynb` | `05_costly_signaling_simulations.ipynb` |
| `plotting_results.ipynb` | `06_plotting_results.ipynb` |

The numeric prefixes encode the natural read order: sanity check → initialization study → main runs → hyperparameter sweep → costly-signaling runs → plots. This matches the KB skill §13.

**Two updates required after rename:**

1. [README.md](README.md) — the **Notebooks** and **Reproducing the figures** sections both link to the old filenames. Update those links (six occurrences).
2. Any notebook that hard-codes a relative path back to itself (none currently — verified by grep over all six files) needs an update.

**Caveat:** renaming a `.ipynb` is a `git mv`. The CODING_AGENT_MAIN_WORKFLOW rule 7 forbids the agent from running `git add` / `git mv`. The agent will write the new file and delete the old one via `Bash mv` — the user reviews and stages the rename.

**Exit criterion:** all six notebooks renamed, README links updated, `nb_migrate.py` confirms nbformat=4.5 and kernel="rl_signaling" on every file.

### Phase 2 — API migration

**One notebook at a time.** Each migration is a self-contained reviewable change.

For every notebook below, apply the mapping table from the previous section and the KB skill §6 setup template. Specific notes per notebook follow.

#### 2.1 — `01_basic_unit_test.ipynb`

Already on the canonical API. Polish only:
- Add a markdown title + abstract cell (currently missing).
- The existing imports cell already does the right thing — leave alone.
- No `SMOKE_TEST` needed; runtime is ~30 s.

#### 2.2 — `02_initializations_test.ipynb`

Already on the canonical API. Polish only:
- Add a markdown title + abstract cell.
- The greyscale block (cells 12–13) has no `plt.savefig` — either add saves to `results/initializations_grayscale_*.png` or remove the cells. Pick one with the user.
- Add a comment cell flagging that the saved [results/initializations_urn_rewards.png](results/initializations_urn_rewards.png) and [results/initializations_urn_nmi.png](results/initializations_urn_nmi.png) figures, once regenerated, will differ from the pre-fix versions documented in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md) Bug 1.

#### 2.3 — `03_run_simulations.ipynb`

Heaviest migration. Six top-level simulation blocks (canonical UrnAgent, Q-learning, TD; complex UrnAgent, Q-learning, TD).

For each block:
- Replace `NetMultiAgentEnv` → `MultiAgentEnv`, `TempNetMultiAgentEnv` → `MultiAgentEnv(agent_type=TDLearningAgent)`.
- Move the per-agent kwargs (exploration rate, decay, choice, gamma) into `agent_kwargs`.
- Replace `simulation_function(n_agents=..., n_features=..., ..., env=env, plot=False, verbose=False)` → `run_simulation(env, n_episodes=..., with_signals=..., plot=False, verbose=False)`.
- Drop the manual `env.agents = [...]` rebuild — `agent_kwargs` handles it.

For the setup section:
- Replace the `!git clone ANONYMIZED_REPO_URL` + `%cd RL_Signaling` cells with the dual `RUNNING_LOCALLY` block from KB skill §6.
- Add `SMOKE_TEST` parameters (default `True`, `N_ITERATIONS=10`, `N_EPISODES=500`). Six blocks × 10k iter × 10k episodes is roughly an overnight run — the smoke-test default must be small.
- Gate Colab Drive mount on `not RUNNING_LOCALLY`. Local default: `dump_path = Path('../results')`.

**Bug 1 awareness:** the existing `urnagent_results_canonical.csv` and `urnagent_results_complex_randomized.csv` were generated with the pre-fix UrnAgent. Re-running this notebook in Phase 4 will overwrite them with corrected numbers.

#### 2.4 — `04_parameter_optimization.ipynb`

Same legacy-API mapping as 2.3, applied to the four `bayesian_*_parameter_search*` function bodies. Each function instantiates the legacy env inside the inner `single_trial` closure; the only change inside the closure is the env / runner swap.

**Multiprocessing reseed bug check.** Per KB skill §10, joblib's `Parallel(n_jobs=...)` with the default backend on macOS/Linux uses `fork` and inherits the parent RNG state unless workers reseed. The notebook's inner `single_trial` already calls `np.random.seed(trial_seed)` + `random.seed(trial_seed)` at entry — that satisfies the requirement. **Leave the existing seed plumbing alone.**

**Decision (2026-05-15):** this notebook stays Colab-only. The sweeps are too heavy for a developer laptop and the Bayesian-optimization budget (`n_calls=200`, `n_trials=100`, `n_episodes=10000`) needs Colab's parallel CPUs. The setup section keeps the `!git clone` + `subprocess.run(['pip', 'install', ...])` + Drive-mount scaffolding from KB skill §6, but skips the `RUNNING_LOCALLY=True` branch. The `RUNNING_LOCALLY` flag itself is still introduced for consistency — it just doesn't have a useful local mode here.

#### 2.5 — `05_costly_signaling_simulations.ipynb`

The Roth–Erev block in this notebook was already retired in the markdown (see cell 7) on 2026-05-09; only the Q-learning costly block needs migration:

- `NetMultiAgentEnv(... costly_signaling=True ...)` + manual agent rebuild with `effective_n_signaling_actions = n_signaling_actions + 1` → `MultiAgentEnv(... costly_signaling=True, agent_kwargs={"choice": "ucb", "exp_smoothing": False, ...})`. The canonical env auto-appends the null signal — drop the `+ 1` arithmetic.
- `simulation_function(..., costly_signaling=True, signal_cost=signal_cost)` → `run_simulation(env, ..., signal_cost=signal_cost)`.

Same Colab → dual-environment scaffolding as 2.3.

#### 2.6 — `06_plotting_results.ipynb`

No API migration (the notebook consumes CSVs only). Polish:
- Set kernel to `rl_signaling`.
- Bump to nbformat 4.5.
- Add `RUNNING_LOCALLY` switch + `RESULTS_DIR` constant; replace every `dump_path+'...'` with `RESULTS_DIR / '...'`.
- Replace the literal `../results/` string with the constant.

**Exit criterion:** every notebook imports from the canonical surface only. Run `nb_migrate.py` and confirm the legacy-API grep returns 0 hits per notebook.

### Phase 3 — Validation

For each migrated notebook, in this order:

1. Restart-and-Run-All on a fresh `rl_signaling` kernel, with `SMOKE_TEST=True`.
2. Confirm no `DeprecationWarning` from `rl_signaling.*` is emitted.
3. Confirm no error cells.
4. Confirm `nbformat.validate(nb)` succeeds after running.

Run the test suite to confirm the package still passes:

```bash
pytest tests/ -v
```

The 63 tests should all pass; the notebooks don't change the package.

**Exit criterion:** all six notebooks pass Restart-and-Run-All under `SMOKE_TEST=True`, the test suite is green.

### Phase 4 — Regenerate stale artifacts (DEFERRED — out of scope for this refactor)

**Status (2026-05-15): out of scope.** The user has confirmed that re-running the experiments is wanted eventually, but should not block the notebook refactor. Phases 0–3 ship as one PR; Phase 4 is filed as a follow-up task in [TODO_WORKFLOW.md](TODO_WORKFLOW.md) and executed when a beefy machine (or an overnight slot) is available.

The procedure below stays in this plan as the reference for that follow-up task.

For each notebook whose outputs are now stale due to Bug 1 (UrnAgent init) or general drift:

1. Set `SMOKE_TEST=False` and run end-to-end.
2. Confirm the saved CSV has the same column names as before (the column schema is hard-coded in each `run_*_for_iteration` function and is preserved by the migration).
3. Run `06_plotting_results.ipynb` to regenerate every figure under [results/](results/).

Order of execution: 03 → 05 → 02 → 06. (04 is a hyperparameter sweep — regenerating it is a separate decision; the best-params it produces are already baked into 03 and 05 as constants.)

**Cost estimate.** With the current parameter sweep (10k iterations × 10k episodes × 6 blocks across notebook 03), this is multi-day on a single laptop. Reserve for a beefy machine or for an overnight run with a smaller `N_ITERATIONS`.

**Exit criterion:** [results/](results/) directory is consistent with the migrated notebooks; [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md) Bug 1 has its "figures need regeneration" caveat removed.

### Phase 5 — Documentation pass + nbstripout

**Adopt `nbstripout` (decision 2026-05-15).** Before merging the refactor, configure `nbstripout` so future commits track notebook *source* only, not embedded cell outputs. One-time setup:

```bash
pip install nbstripout
nbstripout --install               # installs the git filter for this repo
nbstripout --install --attributes .gitattributes   # commits the filter config
```

This adds a `.gitattributes` entry that tells git to pipe every `.ipynb` through `nbstripout` on add/diff/merge. The local working copy keeps its outputs (Jupyter/VS Code shows them as usual), but the committed version has every code cell's `outputs: []` and `execution_count: null`. The current heavyweight outputs in [basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb) (~4 MB) and [plotting_results.ipynb](notebooks/plotting_results.ipynb) (~3 MB) will shrink to source size in the first commit after the filter is installed.

Add `nbstripout` to the `[dev]` extras in [pyproject.toml](pyproject.toml) so new contributors get it from `pip install -e ".[dev]"`.

Other documentation updates:

- Update [README.md](README.md):
  - **Notebooks** table: new filenames (Phase 1), one-line refreshed purpose strings.
  - **Reproducing the figures** section: new notebook ordering.
  - **Setup** section: mention `nbstripout --install` as a one-time step after `pip install -e ".[dev]"`.
  - **Status and known limitations** section: drop the `notebooks/Initializations_test.ipynb` stale-figures caveat once Phase 4 (the deferred follow-up) completes.
- Update [notebooks/NOTEBOOKS_README.md](notebooks/NOTEBOOKS_README.md): note that the repo uses `nbstripout` so cell outputs aren't committed — viewers need to run a notebook to see plots, or look at [results/](results/).
- Append a [WORKLOG.md](WORKLOG.md) entry summarizing the notebook refactor.
- File a Phase 4 follow-up task in [TODO_WORKFLOW.md](TODO_WORKFLOW.md): "Regenerate `results/` from the migrated notebooks." Cross-reference this plan.
- Delete this `NOTEBOOK_REFACTOR_PLAN.md` once Phases 0–3 + Phase 5 are complete (Phase 4 is then tracked by its TODO_WORKFLOW entry, not by this plan).

---

## Tooling (Phase 0 deliverables in detail)

### `notebooks/_tools/nb_migrate.py`

A small CLI utility built on `nbformat`. Two subcommands:

```bash
python notebooks/_tools/nb_migrate.py upgrade <path/to/nb.ipynb>
python notebooks/_tools/nb_migrate.py audit <path/to/nb.ipynb>
```

- `upgrade` performs the metadata-level changes (nbformat bump, kernel, cell IDs). Safe and idempotent.
- `audit` greps every code cell for legacy-API tokens and reports the cells / line numbers that still need a source-level rewrite.

A convenience wrapper runs both commands across the whole folder:

```bash
python notebooks/_tools/nb_migrate.py upgrade notebooks/
python notebooks/_tools/nb_migrate.py audit   notebooks/
```

The script intentionally **does not** perform source rewrites. Per the CODING_AGENT_MAIN_WORKFLOW Phase 4 rules, mechanical rewrites this large should land as small, reviewable diffs — `NotebookEdit` with `cell_id=...` per the KB skill §8 is the right tool.

### `notebooks/NOTEBOOKS_README.md`

Short reference for anyone (or any future agent) opening this folder cold. Contents:

- One-paragraph orientation: what the notebooks do, what order to run them in, what they consume / produce.
- Local setup: kernel name, where `results/` lives, the `RUNNING_LOCALLY=True` default.
- Colab setup: the dual-env block, the Drive mount, the `RUNNING_LOCALLY=False` switch.
- Pointer to KB skill `content/how-to/NOTEBOOK_WRITING_SKILL.md` for the full convention reference.
- Pointer to `_tools/nb_migrate.py` for metadata / audit operations.

---

## Validation checklist

After each phase, run:

```bash
# Phase 0 — tooling
python notebooks/_tools/nb_migrate.py audit notebooks/*.ipynb     # legacy-API report

# Phase 1 — metadata
python -c "import json,glob; [json.load(open(p)) for p in glob.glob('notebooks/*.ipynb')]"   # JSON validity
python notebooks/_tools/nb_migrate.py audit notebooks/*.ipynb     # confirm kernel + nbformat

# Phase 2 — per-notebook API migration
python notebooks/_tools/nb_migrate.py audit notebooks/<file>.ipynb   # expect 0 legacy hits

# Phase 3 — re-runnability
jupyter nbconvert --to notebook --execute notebooks/<file>.ipynb --output /tmp/exec_check.ipynb

# Phase 4 — final
pytest tests/ -v
```

---

## Risks and open questions

1. **TD divergence.** The 1-in-100-episode mismatch between the canonical and legacy TD flows means TD CSVs will not be byte-identical after migration. Documented in the README; accepted. Phase 4 (deferred) will produce a fresh [results/TD-learning_*.png](results/) suite — until then the existing PNGs stay in place, even though they were generated by the legacy flow.
2. **Hyperparameter sweep constants.** The best-params used by `03_run_simulations` and `05_costly_signaling_simulations` (e.g., `exploration_rate=0.9652628633727897`) came from the original sweep in 04. The plan keeps them frozen — re-sweeping is part of the deferred Phase 4 work, not this refactor.
3. **Renames are git-mv events.** The agent does not stage them per CODING_AGENT_MAIN_WORKFLOW rule 7. The user runs `git mv` (or stages the delete + add) after reviewing.

---

## Decisions (resolved 2026-05-15)

1. **Rename mapping** — accepted as proposed (numeric prefixes `01_`–`06_`, snake_case).
2. **`nbstripout`** — adopt during Phase 5. One-time setup at the repo level; new contributors get it via the `[dev]` extras.
3. **Phase 4 scope** — deferred. Phases 0–3 + Phase 5 ship as the refactor PR; Phase 4 becomes a follow-up task in [TODO_WORKFLOW.md](TODO_WORKFLOW.md) scheduled when a beefy machine is available.
4. **04 notebook environment** — `04_parameter_optimization.ipynb` stays Colab-only. The `RUNNING_LOCALLY` flag is introduced for consistency but the notebook documents that the local mode is not intended to run the full sweep.
