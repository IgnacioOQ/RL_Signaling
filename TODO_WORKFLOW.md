# TODO Workflow
- status: active
- type: plan
- id: rl_signaling.todo_workflow
- description: Cross-session task backlog; each task is self-contained and can be picked up by a coding agent with kb_mcp MCP tool access.
- label: [planning, agent]
- injection: excluded
- volatility: evolving
- scope: project-specific
- owner: agent
- last_checked: 2026-05-08
<!-- content -->
Cross-session task backlog. Tasks are added here when work started in a session cannot be completed immediately. Each task must be fully self-contained — a fresh agent should be able to pick it up using only the task body and the kb_mcp tools, with no additional context required.

This file is the per-repository instance of the `TODO_WORKFLOW_TEMPLATE.md` pattern. It lives at the root of the working repository alongside `WORKLOG.md` and is intentionally **not registered with kb_mcp** — agents access it via the regular filesystem `Read`/`Edit` tools, not via `knowledge_base_*` calls.

**Agent rules (picking up tasks):**
1. Read each task in full before starting. If its preconditions are unmet, skip it and note the blocker.
2. After completing a task, delete its entire block from this file (from the `---` divider above the `##` header through the `---` divider below the last line of the task body).
3. After completing one or more tasks, assess whether a `WORKLOG.md` entry is warranted — see Phase 5 of `content/workflows/CODING_AGENT_MAIN_WORKFLOW.md`.
4. Confirm a task is still valid before executing; conditions may have changed since it was written.

**Adding tasks (session authors):**
- Copy the template at the bottom of this file (without fences), fill in all fields, and insert it as a new `##` block above the Template section, preceded and followed by `---`.
- Be precise: include target file paths, specific tool calls, expected outcomes, and a verification step.

---

## Run the Debugging Audit
- status: in-progress (Phases 0–5 complete; Phase 6 deferred to a follow-up session after Phase 5 fixes land)
- type: task
- id: todo.debugging_audit
- description: Execute the phased audit in DEBUGGING_PLAN.md to compare the rl_signaling/ implementation against the intended signaling model.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-08
<!-- content -->
**Context:** `DEBUGGING_PLAN.md` at the repo root carries a six-phase plan for auditing the implementation against the user's intended signaling model. Phase 1 is a model-specification handshake with the user; Phases 2–4 are the audit itself; Phase 5 produces a ranked fix plan; Phase 6 (separate session) verifies fixes after they land. Discrepancies surfaced during the audit are filed in `LEGACY_BUGS_LOG.md`.

**Status as of 2026-05-08:**
- Phase 0 (session boot) — done.
- Phase 1 (confirmed model specification) — done. The 24-axis model spec is recorded at the bottom of `DEBUGGING_PLAN.md`. The design-space catalog is in `MODELING_CHOICES_REF.md`.
- Phase 2 (module-level audit) — done. One new bug filed (Bug 4); one existing bug entry updated (Bug 2 root-cause framing).
- Phase 3 (notebook-level audit) — done. Four new bugs filed (Bug 5, 6, 7, 8).
- Phase 4 (numerical sanity) — done. 10 new tests in `tests/test_numerical_sanity.py`; full suite 60-passing. Independent verification scripts under `analytics/scripts/` all pass at atol=1e-12 (or documented Monte-Carlo tolerance).
- Phase 5 (synthesis + fix plan) — done. Three batches (A: quick wins, B: needs decisions, C: defer) ranked in DEBUGGING_PLAN.md.
- Phase 6 (verification re-run) — **pending**. Runs after Phase 5 fixes land.

**Phase 6 verification recipe (when fixes have landed):**
1. From a clean checkout, set up the venv and confirm `pytest tests/ -q` reports 60 passed.
2. Re-run each affected experiment notebook end-to-end on a fresh kernel.
3. Diff each regenerated figure / CSV against the archived pre-fix version. Predicted directions are in `LEGACY_ERRORS_LOG.md` Section G.
4. For each `LEGACY_BUGS_LOG.md` entry, append a "Post-fix observation" subsection recording the actual measured impact and whether it matched the prediction.
5. Update each Bug entry's `status` from `todo` / `open` to `done`.

**On completion:** After Phase 6 closes, delete this task block (from the `---` above the `##` header to the `---` below the last line) and add a final WORKLOG entry recording the audit's completion.

---

## Apply Phase 5 Fix Plan
- status: todo
- type: task
- id: todo.apply_phase_5_fix_plan
- description: Implement the Phase 5 fix plan from DEBUGGING_PLAN.md — Bugs 4, 5, 6, 7, 8 — in the recommended Batch A → Batch B order, with the Roth-Erev costly retirement folded in.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-08
<!-- content -->
**Context:** Phase 5 of `DEBUGGING_PLAN.md` (the section titled "Phase 5 — Fix plan") proposes three batches of fixes for the six open bugs surfaced during Phases 2–3 of the audit:

- **Batch A (quick wins, no decisions):** Bug 7 (missing imports in `Parameter_Optimization_wchoices.ipynb`) + Bug 8 (one-line `filename_prefix` typo in `plotting_results.ipynb`). Estimated ~30 lines across three files.
- **Batch B (needs user decisions):** Bug 5 (Initializations_test override), Bug 4 (QLearningAgent action-Q-table pre-seed), Bug 6 (Run_Simulations writes `*_complex.csv`, plotting_results reads `*_complex_randomized.csv`). Estimated 30–60 lines depending on Option A/B chosen for each.
- **Batch C (defer):** Bug 2 (TempNetMultiAgentEnv nested NMI loop) — becomes moot if Batch B Option B migrates notebooks off the legacy path.

`LEGACY_BUGS_LOG.md` carries the per-bug detail (symptom, root cause, file paths, fix proposals). `LEGACY_ERRORS_LOG.md` traces which saved figures change with each fix and predicts the direction of change.

**Preconditions:**
- `git status` shows clean working tree on the `debugging` branch (or the user's named fix branch). If on `refactor` or `main`, ask before proceeding.
- `.venv/python -m pytest tests/ -q` reports 60 passed.
- `DEBUGGING_PLAN.md`, `LEGACY_BUGS_LOG.md`, and `LEGACY_ERRORS_LOG.md` are at the repo root and read in full before any code change.

**Steps:**
1. Read `DEBUGGING_PLAN.md` Phase 5 section end-to-end. Confirm with the user which Option to take for each Batch B bug:
   - Bug 5: Option A (preserve init kwargs in `env.agents=` override) or Option B (migrate notebook to canonical `MultiAgentEnv` + `agent_kwargs`). Recommended: B.
   - Bug 4: confirm whether the docstring's "signaling-only" pre-seed intent is real or a refactor copy-paste artefact.
   - Bug 6: Option A (restore randomized action sizes; rename outputs to `*_complex_randomized.csv`) or Option B (re-anchor `plotting_results.ipynb` on the existing fixed-action `*_complex.csv` files; retire the `_randomized` figures). Recommended: A.
2. **Apply Batch A** (no decisions needed):
   - Bug 7: add the missing imports to cell 3 of `notebooks/Parameter_Optimization_wchoices.ipynb`:
     ```python
     import multiprocessing
     from datetime import datetime
     from joblib import Parallel, delayed
     from skopt import Optimizer
     from skopt.space import Categorical, Integer, Real
     ```
     Add `scikit-optimize` to `[project.optional-dependencies] dev` in `pyproject.toml`.
   - Bug 8: in `notebooks/plotting_results.ipynb`'s final code cell, change `filename_prefix='Q-learning_complex_randomized'` to `filename_prefix='TD-learning_complex_randomized'`.
3. **Apply Batch B** (after decisions in step 1):
   - Implement Option A or B for each bug as confirmed.
   - Update each `LEGACY_BUGS_LOG.md` Bug entry's `status` field to `done` and add a brief "Fix applied" sub-section linking to the commit / change.
4. **Defer Batch C** unless the user explicitly requests Bug 2 fix in this session.
5. **Address Error 5a (Roth-Erev costly retirement)** — see the separate `todo.retire_costly_urnagent` task block below; that one captures the theoretical-soundness rationale.
6. Run `.venv/bin/python -m pytest tests/ -q` after each batch. Expected: 60 passed.
7. After Batch B lands, prompt the user to re-run the affected experiment notebooks (or invoke the `todo.verify_reproducibility` task block) so saved figures regenerate.

**Verification:**
- All Bug entries in `LEGACY_BUGS_LOG.md` for the bugs addressed have `status: done` and a "Fix applied" sub-section.
- `pytest tests/` reports 60 passed.
- `notebooks/Parameter_Optimization_wchoices.ipynb` does not raise `NameError` on Restart-and-Run-All at the first `param_ranges = {... Categorical([...]) ...}` cell.
- `notebooks/plotting_results.ipynb` produces a `TD-learning_complex_randomized_regression_*.png` file when run end-to-end (after Batch B Bug 6 is also resolved).
- `notebooks/Initializations_test.ipynb` produces visibly distinct curves per `init_weights` after Bug 5 fix.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry summarizing which Bug entries were closed.

---

## Retire or Relabel the Costly UrnAgent Experiment
- status: todo
- type: task
- id: todo.retire_costly_urnagent
- description: Decide on whether to retire or relabel the costly UrnAgent figures and CSVs in light of the Roth-Erev × costly-signaling theoretical incompatibility surfaced in the analytics/ math reference.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-08
<!-- content -->
**Context:** The costly-signaling extension is **theoretically ill-defined** under the project's Roth-Erev urn rule (see `analytics/agent_urn.md` "Applicability constraints" section and `analytics/costly_signaling.md` "Compatibility with the project's three agent types" section). The classical Roth-Erev urn was specified for non-negative integer reinforcement; costly signaling produces real-valued and potentially negative rewards, both of which break the urn-as-counter interpretation. The clamp `urn[a] = max(0, urn[a] + r)` introduces an absorbing barrier — once `urn[a] = 0`, the action is never sampled again, so it cannot recover under negative-reward streaks.

This concern is **separate** from Error 5a's protocol drift (per-agent independent vs shared cost) noted in `LEGACY_ERRORS_LOG.md`. Even if the protocol drift were fixed, the experiment would still be measuring a degenerate variant of Roth-Erev, not the canonical model.

The affected files are:
- `results/Roth-Erev_canonical_costly_signal_*.png` (3 PNGs)
- `results/q_costly_*.png`, `results/q_learning_costly_single_run*.png` (4 PNGs — Q-prefixed but consume the Roth-Erev CSV per `plotting_results.ipynb`)
- `results/urnagent_results_canonical_costly_signal.csv` (1000 rows; never-committed independent-cost protocol)
- `results/urnagent_results_canonical_costly_signal (1).csv` (10 000 rows; shared-cost protocol matching current code; orphan)

The Q-learning costly figures (`results/QLearning_canonical_costly_signal_*.png` + `results/qlearning_results_canonical_costly_signal.csv`) are **clean** and unaffected — Q-learning's TD update is defined on $\mathbb{R}$ and handles the costly reward range natively.

**Preconditions:**
- `analytics/agent_urn.md` and `analytics/costly_signaling.md` are present and the "Applicability constraints" / "Compatibility" sections are read in full.
- `LEGACY_ERRORS_LOG.md` Error 5a is read.

**Steps:**
1. Confirm with the user which option to take:
   - **Option A — Retire.** Delete the affected files. Remove the Roth-Erev block (or its costly invocation) from `notebooks/Final_Costly_Signaling_Run_Simulations.ipynb`. Remove the costly Roth-Erev consumer cells from `notebooks/plotting_results.ipynb`. Delete the orphan `(1).csv`. Add a markdown note to the costly notebook's title cell: "Costly signaling is reported only for QLearningAgent; UrnAgent is not included due to the Roth-Erev × costly-signaling theoretical incompatibility (see `analytics/agent_urn.md`)."
   - **Option B — Relabel.** Keep the figures but relabel them and the CSV columns as "Roth-Erev with non-negativity-clamped costly extension" so the deviation from the canonical Roth-Erev rule is visible to readers. Add a paragraph to the figures' surrounding markdown citing `analytics/agent_urn.md` "Applicability constraints" for the formal treatment.
   - **Recommended:** Option A.
2. Apply the chosen option.
3. If Option A is chosen and the costly-signaling section of `Run_Simulations.ipynb` is impacted, confirm that path is not exercised.
4. Update `LEGACY_ERRORS_LOG.md` Error 5a to reflect resolution. Move the verdict from UNREPRODUCIBLE to either RETIRED (Option A) or RELABELED (Option B).
5. Run the test suite to confirm nothing regressed:
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```

**Verification:**
- The decision is recorded explicitly in `LEGACY_ERRORS_LOG.md` Error 5a.
- If Option A: the listed files are removed; `notebooks/Final_Costly_Signaling_Run_Simulations.ipynb` no longer references UrnAgent in its costly block.
- If Option B: every affected figure / column / markdown cell carries the "non-negativity-clamped costly extension" relabel.
- `pytest tests/` reports 60 passed.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry recording the chosen option.

---

## Verify Experiment Reproducibility End-to-End
- status: todo
- type: task
- id: todo.verify_reproducibility
- description: After Phase 5 fixes land, re-run every experiment notebook on a clean kernel and confirm every CSV in results/ regenerates and every PNG in results/ is reproducible from the regenerated CSVs. Document the actual reproducibility status.
- owner: agent
- blocked_by: [todo.apply_phase_5_fix_plan, todo.retire_costly_urnagent]
- last_checked: 2026-05-08
<!-- content -->
**Context:** `LEGACY_ERRORS_LOG.md` catalogues the saved-figure status (CLEAN, BIASED-METRIC, MISLABELED, UNREPRODUCIBLE, WRONG) for every artifact in `results/`. Several bugs surfaced in Phases 2–3 of the audit break end-to-end reproducibility:

- **Bug 6** — `Run_Simulations.ipynb` writes `*_complex.csv`, `plotting_results.ipynb` reads `*_complex_randomized.csv` — so the README's "Reproducing the figures" recipe does not currently work for the complex experiment family.
- **Error 5a** — costly Roth-Erev figures use an independent-cost protocol that's never been committed to git.
- **Bug 5** — `Initializations_test.ipynb` silently runs the same configuration four times instead of varying initialization weights.
- **Bug 7** — `Parameter_Optimization_wchoices.ipynb` fails Restart-and-Run-All due to missing imports.
- **Cross-cutting (not a bug, deferred)** — game_dicts and signal_cost are constructed before the per-iteration seed reset, so individual rows of the saved CSVs are not row-reproducible from `iteration` alone (population statistics are unaffected). Fix would migrate to `numpy.random.SeedSequence().spawn()`.

This task verifies, **after the Phase 5 fixes have landed**, that:
- A fresh checkout + clean venv reproduces every CSV in `results/` from scratch.
- A fresh checkout + clean venv reproduces every PNG in `results/` from those CSVs.
- The README's "Reproducing the figures" instructions execute without manual intervention.

**Preconditions:**
- `todo.apply_phase_5_fix_plan` is closed.
- `todo.retire_costly_urnagent` is closed (so the costly Roth-Erev artifacts are either retired or relabeled).
- All open bugs in `LEGACY_BUGS_LOG.md` for which Phase 5 batches were chosen show `status: done`.
- `pytest tests/` reports 60 passed.
- An archive of the pre-fix `results/` directory is preserved on a separate branch or backup so the post-fix diff is meaningful.

**Steps:**
1. From a clean checkout (or after `git stash` of the working tree), set up the venv:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   python -m ipykernel install --user --name rl_signaling --display-name "Python (rl_signaling)"
   ```
2. Run the test suite to confirm green baseline:
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```
   Expected: 60 passed (more if Phase 5 added regression tests).
3. Restart-and-Run-All each experiment notebook on a fresh kernel, in this order:
   - `notebooks/Run_Simulations.ipynb` — produces `urnagent_results_canonical.csv`, `qlearning_results_canonical.csv`, `td_learning_results_canonical.csv`, and (depending on Bug 6 outcome) `*_complex_randomized.csv` or `*_complex.csv`.
   - `notebooks/Initializations_test.ipynb` — produces `initializations_nmi.png`, `initializations_rewards.png` (now with real init effect after Bug 5 fix).
   - `notebooks/Final_Costly_Signaling_Run_Simulations.ipynb` — produces costly-signaling CSV(s). After `todo.retire_costly_urnagent`, only the Q-learning block (or none) remains.
   - `notebooks/Parameter_Optimization_wchoices.ipynb` — research log; verify Restart-and-Run-All no longer raises `NameError` (Bug 7 fix).
4. After each notebook completes, check that the expected CSV(s) under `results/` have been freshly written. Diff a few sample rows against the pre-fix archive to quantify the change. Predicted-direction predictions are in `LEGACY_ERRORS_LOG.md` Section G.
5. Run `notebooks/plotting_results.ipynb` on a fresh kernel via Restart-and-Run-All. Confirm every read succeeds without `FileNotFoundError`, and every PNG under `results/` has a fresh timestamp.
6. Compute and record the diff statistics: for each saved CSV, compare pre-fix and post-fix means / standard deviations of every numeric column. For each PNG, optionally use `scikit-image` SSIM or eyeball-compare against the pre-fix archive.
7. Write up the reproducibility audit as either:
   - A new `## YYYY-MM-DD — Reproducibility audit` entry in `WORKLOG.md`, or
   - A standalone `REPRODUCIBILITY.md` at the repo root if the audit is large enough to warrant its own document.
8. Update the README's "Reproducing the figures" section if any step requires extra manual setup that the current text does not document.
9. **Optional but recommended:** migrate the multiprocessing seeding pattern to `numpy.random.SeedSequence().spawn()` so individual rows of the saved CSVs are row-reproducible from `iteration` alone. See `content/how-to/NOTEBOOK_WRITING_SKILL.md` Section 8 ("Parallel processing — Seeds across workers") for the recommended pattern. If deferred, file a separate task.

**Verification:**
- `git status` after a fresh end-to-end run shows clean modifications only to expected files (CSVs in `results/`, PNGs in `results/`, optionally notebook output cells).
- A diff between pre-fix and post-fix figures is documented in `WORKLOG.md` or `REPRODUCIBILITY.md`.
- `pytest tests/` still passes.
- The README "Reproducing the figures" section reflects the current procedure with no inaccuracies.
- `LEGACY_ERRORS_LOG.md` is updated: every `UNREPRODUCIBLE` verdict is replaced with either `CLEAN` (if the post-fix re-run resolved it) or kept with a note explaining why reproducibility is still partial (e.g. multiprocessing-seed row-level non-reproducibility).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry recording the audit results.

---

## Task Template

Copy the block below (without the outer fences), fill in all fields, and insert it as a new `## [Task Title]` task block.

````markdown
## [Task Title]
- status: todo
- type: task
- id: todo.[short_id]
- description: One-sentence description of what this task accomplishes.
- owner: agent
- blocked_by: []
- last_checked: {{YYYY-MM-DD}}
<!-- content -->
**Context:** Why this task exists and what triggered it. Include the KB path or repo file path it operates on.

**Preconditions:** Any state that must be true before starting (prior tasks complete, files present, etc.). Write `none` if there are none.

**Steps:**
1. (Include specific tool calls where possible, e.g., `knowledge_base_read(path="content/...", sections=["..."])`)
2. ...

**Verification:** How to confirm the task is complete (e.g., a grep that should return one match, a status field that should read `done`).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).
````
