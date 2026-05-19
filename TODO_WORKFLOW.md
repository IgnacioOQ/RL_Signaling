---
status: active
type: plan
id: rl_signaling.todo_workflow
description: Cross-session task backlog; each task is self-contained and can be picked up by a coding agent with kb_mcp MCP tool access.
label: [planning, agent]
injection: excluded
volatility: evolving
scope: project-specific
owner: agent
last_checked: '2026-05-19'
---

# TODO Workflow

Cross-session task backlog. Tasks are added here when work started in a session cannot be completed immediately. Each task must be fully self-contained — a fresh agent should be able to pick it up using only the task body and the kb_mcp tools, with no additional context required.

This file is the per-repository instance of the `TODO_WORKFLOW_TEMPLATE.md` pattern. It lives at the root of the working repository alongside `WORKLOG.md` and is intentionally **not registered with kb_mcp** — agents access it via the regular filesystem `Read`/`Edit` tools, not via `knowledge_base_*` calls.

**Agent rules (picking up tasks):**
1. Read each task in full before starting. If its preconditions are unmet, skip it and note the blocker.
2. After completing a task, delete its entire block from this file (from the `---` divider above the `##` header through the `---` divider below the last line of the task body).
3. After completing one or more tasks, assess whether a `WORKLOG.md` entry is warranted — see Phase 5 of `content/workflows/CODING_AGENT_MAIN_WORKFLOW.md`.
4. Confirm a task is still valid before executing; conditions may have changed since it was written.

**Adding tasks (session authors):**
- Copy the template at the bottom of this file (without the outer fences), fill in all fields, and insert it as a new `##` block above the Template section, preceded and followed by `---`.
- Be precise: include target file paths, specific tool calls, expected outcomes, and a verification step.

---

## Verify Experiment Reproducibility End-to-End

```yaml
status: todo
type: task
id: todo.verify_reproducibility
description: Re-run every experiment notebook on a clean kernel and confirm every CSV in results/ regenerates and every PNG in results/ is reproducible from the regenerated CSVs. Document the actual reproducibility status.
owner: agent
blocked_by: []
last_checked: '2026-05-09'
```

**Context:** `LEGACY_ERRORS_LOG.md` catalogues the saved-figure status (CLEAN, BIASED-METRIC, MISLABELED, UNREPRODUCIBLE, WRONG, RETIRED) for every artifact in `results/`. As of the 2026-05-09 fix session, the Phase 5 fixes for Bugs 4, 5, 6, 7, 8 have all landed, the Roth-Erev costly experiment was retired, and `notebooks/Initializations_test.ipynb` was re-run end-to-end with its 4 figures regenerated. The remaining reproducibility work is to verify the **larger experiment notebooks** end-to-end on a fresh checkout + clean venv:

- `notebooks/Run_Simulations.ipynb` — Bug 6 fix renamed `*_complex.csv` → `*_complex_randomized.csv` and randomized action sizes per iteration. Producer / consumer chain is consistent by code review; full re-run is gated on (a) flipping `simulate=True` in the UrnAgent block (cell 15, currently gated for compute reasons) and (b) replacing cell 4's Colab `dump_path = '/content/drive/My Drive/...'` with `dump_path = '../results/'` for local execution.
- `notebooks/Final_Costly_Signaling_Run_Simulations.ipynb` — UrnAgent block retired (replaced with a note); only the Q-learning block remains active. Full re-run requires the same Colab/local-path swap.
- `notebooks/Parameter_Optimization_wchoices.ipynb` — Bug 7 fix in place; `scikit-optimize 0.10.2` provisioned via `pip install -e ".[dev]"`. Restart-and-Run-All gating verified by import resolution; full Bayesian-search re-run is research-log only and not required to keep `results/*.png` consistent.
- **Cross-cutting (not a bug, deferred)** — `game_dicts`, `signal_cost`, and (after Bug 6) per-iteration `n_signaling_actions`/`n_final_actions` are constructed inside the worker subprocess before the `np.random.seed(iteration)` call inside `run_single_case`, so individual rows of the saved CSVs are not row-reproducible from `iteration` alone. Population statistics are unaffected. Fix would migrate to `numpy.random.SeedSequence().spawn()`.

This task verifies that:
- A fresh checkout + clean venv reproduces every active CSV in `results/` from scratch.
- A fresh checkout + clean venv reproduces every active PNG in `results/` from those CSVs (excluding the retired Roth-Erev costly artifacts).
- The README's "Reproducing the figures" instructions execute without manual intervention beyond the documented Colab/local-path swap.

**Preconditions:**
- `pytest tests/` reports 61 passed (60 + the Bug 4 unit test).
- All open bugs in `LEGACY_BUGS_LOG.md` for which Phase 5 batches were chosen show `status: done`. As of 2026-05-09: Bugs 4, 5, 6, 7, 8 = done; Bug 2 = open by design (deferred per Phase 5 plan).
- An archive of the pre-fix `results/` directory is preserved on a separate branch or backup so the post-fix diff is meaningful. The 2026-05-09 fix session backed up `initializations_{rewards,nmi}.png` to `/tmp/rl_signaling_prefix_backup/`; for the rest of `results/`, recover from `git log` history before the 2026-05-09 commit if needed.

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
   Expected: 61 passed.
3. Restart-and-Run-All each experiment notebook on a fresh kernel, in this order:
   - `notebooks/Run_Simulations.ipynb` — produces `urnagent_results_canonical.csv`, `qlearning_results_canonical.csv`, `td_learning_results_canonical.csv`, and `*_complex_randomized.csv` (Bug 6 fix already applied; producer renamed to match the consumer). Before running locally, replace cell 4 (`from google.colab import drive` + `dump_path = '/content/drive/...'`) with `dump_path = '../results/'`, and flip `simulate=True` in cell 15 (UrnAgent complex block) if you want that block to execute.
   - `notebooks/Initializations_test.ipynb` — already re-run end-to-end on 2026-05-09; verify the regenerated `initializations_{rewards,nmi}.png` and `initializations_urn_{rewards,nmi}.png` figures still match the 2026-05-09 outputs (paired-comparison seed reset = deterministic re-run modulo joblib worker startup state — but this notebook uses no Parallel, so it should be fully reproducible).
   - `notebooks/Final_Costly_Signaling_Run_Simulations.ipynb` — UrnAgent block retired; only the Q-learning block remains. Same Colab/local-path swap as Run_Simulations.
   - `notebooks/Parameter_Optimization_wchoices.ipynb` — research log; verify Restart-and-Run-All no longer raises `NameError` (Bug 7 fix). Same Colab/local-path swap as Run_Simulations.
   - `notebooks/plotting_results.ipynb` — runs after the others; consumes the regenerated CSVs.
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

## Execute the notebook refactor plan (Phases 0–3 + Phase 5)

```yaml
status: todo
type: task
id: todo.notebook_refactor
description: Execute Phases 0–3 and Phase 5 of NOTEBOOK_REFACTOR_PLAN.md, migrating the six notebooks under notebooks/ from the legacy NetMultiAgentEnv / simulation_function surface to the canonical MultiAgentEnv + run_simulation API, with the metadata, kernel, and dual local/Colab scaffolding required by the project's notebook conventions. Phase 4 (re-running the experiments to refresh results/) is deferred and tracked separately.
owner: agent
blocked_by: []
last_checked: '2026-05-15'
```

**Context.** The six notebooks under [notebooks/](notebooks/) were authored before the seven-phase code refactor and still call the legacy API surface (`NetMultiAgentEnv`, `TempNetMultiAgentEnv`, `simulation_function`, `temp_simulation_function`), which now emits `DeprecationWarning`. Two of the six ([basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb), [Initializations_test.ipynb](notebooks/Initializations_test.ipynb)) were partly updated during the original refactor; the remaining four still target the legacy surface plus a Colab-only setup section.

[NOTEBOOK_REFACTOR_PLAN.md](docs/code-audit/NOTEBOOK_REFACTOR_PLAN.md) is the authoritative plan. It includes:

- A legacy → canonical API mapping cheat-sheet (read this before touching any notebook).
- Per-notebook refactor recipes (§2.1 through §2.6).
- Resolved decisions (2026-05-15): rename mapping accepted, `nbstripout` adopted in Phase 5, Phase 4 deferred, `04_parameter_optimization.ipynb` stays Colab-only.
- Tooling already in place: [notebooks/_tools/nb_migrate.py](notebooks/_tools/nb_migrate.py) (upgrade + audit subcommands) and [notebooks/NOTEBOOKS_README.md](notebooks/NOTEBOOKS_README.md).

**Preconditions:**
- The plan document and the helper tooling are committed (they were authored in the 2026-05-15 session that wrote this task).
- `pytest tests/ -v` passes on the current branch — the migration should not change package behavior, so the test suite is the regression net.

**Steps:**
1. Read [NOTEBOOK_REFACTOR_PLAN.md](docs/code-audit/NOTEBOOK_REFACTOR_PLAN.md) end-to-end. The §"Legacy → canonical API mapping" table is the cheat-sheet for every notebook edit.
2. Run the audit tool to confirm the starting state matches what the plan documents:
   ```bash
   python notebooks/_tools/nb_migrate.py audit notebooks/
   ```
   Expected: four notebooks show legacy-API hits and `NEEDS UPGRADE` metadata; two show clean state.
3. **Phase 0 (tooling) — already done** by the 2026-05-15 session. Skip.
4. **Phase 1 — rename + metadata pass.** Rename the six notebooks per the plan's table (numeric prefixes `01_` through `06_`, snake_case). Run `python notebooks/_tools/nb_migrate.py upgrade notebooks/` to bump every file to `nbformat=4.5`, set the `rl_signaling` kernel, and assign stable cell IDs. Update the six links in [README.md](README.md) (the **Notebooks** table and the **Reproducing the figures** section). The agent must **not** stage the renames with `git mv` per CODING_AGENT_MAIN_WORKFLOW rule 7 — write new files, delete old, let the user stage.
5. **Phase 2 — API migration**, one notebook at a time, per plan §2.1–§2.6. Use `NotebookEdit` with `cell_id=...` (see KB skill `content/how-to/NOTEBOOK_WRITING_SKILL.md` §8) so edits address cells by their stable IDs rather than by index. After each notebook, re-run `nb_migrate.py audit <file>` — it must report `legacy-API hits: none`. Pay attention to:
   - The return-tuple order change documented in the plan's cheat-sheet (`histories` and `nature_history` swap positions). Notebooks unpacking with `_, _, _, _, _` are unaffected; named-target unpacks need fixing.
   - The TD divergence caveat — TDLearningAgent under the canonical API drifts by ~1% from the legacy flow. This is expected and documented; do not chase byte-identical TD output.
   - `04_parameter_optimization.ipynb` stays Colab-only — introduce `RUNNING_LOCALLY` for consistency but do not pretend the full sweep runs on a laptop.
6. **Phase 3 — validation.** For each migrated notebook, Restart-and-Run-All on a fresh `rl_signaling` kernel with `SMOKE_TEST=True`. Confirm no errors, no `DeprecationWarning` from `rl_signaling.*`, and `nbformat.validate(nb)` succeeds. Run `pytest tests/ -v` — must stay at 63 passed.
7. **Phase 5 — documentation + nbstripout.** One-time setup at the repo level:
   ```bash
   pip install nbstripout
   nbstripout --install
   nbstripout --install --attributes .gitattributes
   ```
   Add `nbstripout` to the `[dev]` extras in [pyproject.toml](pyproject.toml). Update [README.md](README.md) to mention the `nbstripout --install` step in the Setup section. Update [notebooks/NOTEBOOKS_README.md](notebooks/NOTEBOOKS_README.md) to note the strip-on-commit convention. Append a `WORKLOG.md` entry summarizing the refactor.
8. **Phase 4 is out of scope for this task.** The plan file already records that decision. The separate `todo.verify_notebook_drive_paths` task (which depends on this one) covers the Drive-path verification needed before any Colab re-run.

**Verification:**
- `python notebooks/_tools/nb_migrate.py audit notebooks/` reports `legacy-API hits: none` and `nbformat=4.5 OK; kernel='rl_signaling' OK` for every notebook.
- `pytest tests/ -v` reports 63 passed.
- Each migrated notebook completes Restart-and-Run-All under `SMOKE_TEST=True` with no errors and no `DeprecationWarning` from the `rl_signaling.*` namespace.
- [README.md](README.md) **Notebooks** and **Reproducing the figures** sections reference the renamed files.
- [.gitattributes](.gitattributes) contains the `nbstripout` filter line and [pyproject.toml](pyproject.toml) `[dev]` extras include `nbstripout`.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry summarizing what changed and noting that `todo.verify_notebook_drive_paths` is now unblocked. Delete [NOTEBOOK_REFACTOR_PLAN.md](docs/code-audit/NOTEBOOK_REFACTOR_PLAN.md) from `docs/code-audit/` once Phase 5 is done (the plan was an in-flight document; the WORKLOG entry preserves the historical record).

---

## Editorial review of §2.3 in main_v2.pdf, then port main_v2.tex → main.tex

```yaml
status: todo
type: task
id: todo.editorial_review_and_port_main_v2
description: Work through the remaining open/partial reviewer checklist items in main_v2.tex, then do a holistic editorial pass on main_v2.pdf, and finally port main_v2.tex to main.tex. Folds in the closed todo.continue_reviewer_revisions task.
owner: user + agent (review is user; mechanical port is agent)
blocked_by: []
last_checked: '2026-05-19'
```

**Context.** On 2026-05-18 the V2 §2.3 prose was ported from [manuscript/section_2_3.tex](manuscript/section_2_3.tex) into [manuscript/main_v2.tex](manuscript/main_v2.tex) §2.3 (line 482). The PDF compiles cleanly (22 pages, ~2.9 MB). Companion [manuscript/Appendix.tex](manuscript/Appendix.tex) compiles standalone (7 pages, ~1 MB). See [LP_TEX_REF.md](LP_TEX_REF.md) at the repo root for the LaTeX conventions reference, figure mapping tables, and compile commands. See the 2026-05-18 entry in [WORKLOG.md](WORKLOG.md) for the full session record.

The §2.3 prose addresses Reviewer 2's R2·C1 and R2·C2 (existence/reliability triplet; attractor-vs-basin-reach distinction in main text) and Reviewer 3's R3·C2 (proof-of-concept labeled as conceptual + simulation, not a convergence theorem). The Argiento obstruction is in the main text. The §3 intro was tightened to two scenarios (matching + random) with costly signaling deferred to the Appendix.

Reviewer checklist state as of 2026-05-18: R2·C2 + R3·C3 fully `[x]`; R2·C1, R2·C3, R3·C2 `[~]` partial; R2·C4, R2·C5, R3·C1, R3·C4, R3·C5, R3·C6, R3·C7 `[ ]` open. Checklist at [manuscript/reviewers/Reviewers Responses Checklist.md](manuscript/reviewers/Reviewers%20Responses%20Checklist.md). Formal response narrative at [manuscript/reviewers/Generated Responses to Reviewers.md](manuscript/reviewers/Generated%20Responses%20to%20Reviewers.md).

**Steps:**

1. **Reviewer checklist pass.** Load `content/workflows/REVISION_WORKFLOW.md` for the per-item sub-loop. Work through the partial (`[~]`) items first (R2·C1, R2·C3, R3·C2), then the open (`[ ]`) items (R2·C4, R2·C5, R3·C1, R3·C4, R3·C5, R3·C6, R3·C7). For each: read the affected `.tex` passage; propose 2–3 edit options to the user; on selection, apply via `Edit`; compile (`cd manuscript/ && latexmk -pdf main_v2.tex`); verify no new errors beyond the pre-existing baseline (the ~1.3mm body-text micro-overflow in §2.3 is pre-existing); update the checkbox to `[x]` with the date; sync the formal response narrative for substantive edits.
2. **Holistic editorial pass on `main_v2.pdf`.** Open the PDF, re-read §2.3 (pages 13–14 area). Note wording changes or substantive concerns. Re-check that the four `\paragraph{...}` blocks (*The figure*, *Three observations*, *Reading*, *What this is, and what this is not*) read cleanly; if you'd prefer numbered §2.3.1–§2.3.4 instead, swap to `\subsubsection{...}` for all four.
3. **Address remaining flagged items in [LP_TEX_REF.md](LP_TEX_REF.md):**
   - The Argiento footnote on §2.3 (line 517 of `main_v2.tex`) currently reads "documented in a companion technical note" — generic phrasing. Replace with a forward reference to an appendix subsection, or drop entirely if the prose stands on its own.
   - The `manuscript/section_2_3.tex` standalone fragment — delete if you no longer want it as a reference copy, since its content is now in `main_v2.tex`.
4. **Port `main_v2.tex` → `main.tex`** (agent task, mechanical):
   ```bash
   rm "manuscript/main.tex"
   mv "manuscript/main_v2.tex" "manuscript/main.tex"
   cd manuscript/ && latexmk -pdf main.tex
   rm manuscript/main_v2.pdf  # if not already gone
   ```
   Verify the PDF rebuilds cleanly with the new filename.
5. **Optional follow-ups (independent of the port):**
   - **Bare-name aliases for canonical TD figures.** Add `td_canonical_reward.png`, `td_canonical_nmi.png`, `td_canonical_regression.png` to `results/legacy/plots/` (copies or symlinks of the existing `TD-learning_canonical_*` files), then revert the three long-name `\includegraphics{}` calls in `Appendix.tex` Section B to bare names. Removes the bare-vs-long inconsistency documented in `LP_TEX_REF.md`.
   - **`.gitignore` LaTeX block.** Add the block below to `.gitignore` so build artifacts don't show in `git status`. Template is in `LP_TEX_REF.md` "Build artifacts" section.
     ```gitignore
     # LaTeX build artifacts (keep .tex, .bib, and .pdf; ignore the rest)
     *.aux
     *.bbl
     *.blg
     *.fdb_latexmk
     *.fls
     *.log
     *.out
     *.synctex.gz
     ```

**Verification:**
- After the rename, `latexmk -pdf main.tex` produces `main.pdf` with no errors (font warnings OK). 22 pages, all 16 body figures + figure references resolve.
- All §2.3-related references to "main_v2" elsewhere in the repo (memory files, `LP_TEX_REF.md`, `WORKLOG.md`) are either updated to reflect the new filename or kept as historical markers.

**On completion:** Delete this entire task block. Append a one-line `WORKLOG.md` entry recording the rename and any editorial decisions made along the way.

---

## Sweep main_v2.tex and reviewer responses for excessive em-dashes

```yaml
status: todo
type: task
id: todo.dash_sweep
description: Reduce overuse of em-dashes ("---" in LaTeX source / "—" in markdown) in main_v2.tex and manuscript/reviewers/*.md; rewrite parenthetical-dash constructions with parentheses, commas, or sentence splits where the dashed clause is non-load-bearing.
owner: agent
blocked_by: []
last_checked: '2026-05-19'
```

**Context:** User feedback (2026-05-19) during the R2·C5 round flagged excessive em-dash use in newly-added prose (e.g. an earlier draft of the §1.2 minimal-rationality continuation used dashes around a parenthetical: "the minimal-rationality framing of the RL agents --- as cognitively shallow utility-maximizers rather than fully rational deliberators --- is inherited..."). The dash convention in main_v2.tex is `---` (LaTeX em-dash); current count as of 2026-05-19 is 38 occurrences in main_v2.tex (up from 33 pre-revision). Em-dashes should be reserved for genuinely abrupt breaks or strong parentheticals; routine parentheticals work better with commas or parentheses.

**Preconditions:** None.

**Steps:**

1. Grep main_v2.tex for `---` and inspect each occurrence with its surrounding context. Note: literal Unicode em-dashes (`—`) also exist (~3 occurrences); include those.
2. For each occurrence, decide: keep (genuinely abrupt break, or load-bearing rhetorical pause) or rewrite (replace with parentheses, commas, or split into two sentences). Lean rewrite.
3. Apply edits, preferring the lightest substitution that preserves meaning.
4. Repeat the survey on the reviewer-facing prose in manuscript/reviewers/Generated Responses to Reviewers.md (uses Unicode `—` in markdown, not LaTeX `---`). Apply the same review.
5. Recompile main_v2.tex; confirm no rendering issues.

**Verification:**

- `grep -c -- '---' manuscript/main_v2.tex` returns a meaningfully smaller number than 38 (the pre-sweep count noted in this task's Context).
- Each retained em-dash is reviewable and defensible against the "abrupt break or strong parenthetical only" criterion.
- `latexmk -pdf` rebuilds cleanly; no new errors.
- Response document prose flows naturally without leaning on dashes for cadence.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry noting the new em-dash count and the criteria applied.

---

## Task Template

Copy the block below (without the outer fences), fill in all fields, and insert it as a new `## [Task Title]` task block.

````markdown
## [Task Title]

```yaml
status: todo
type: task
id: todo.[short_id]
description: One-sentence description of what this task accomplishes.
owner: agent
blocked_by: []
last_checked: 'YYYY-MM-DD'
```

**Context:** Why this task exists and what triggered it. Include the KB path or repo file path it operates on.

**Preconditions:** Any state that must be true before starting (prior tasks complete, files present, etc.). Write `none` if there are none.

**Steps:**
1. (Include specific tool calls where possible, e.g., `knowledge_base_read(path="content/...", sections=["..."])`)
2. ...

**Verification:** How to confirm the task is complete (e.g., a grep that should return one match, a status field that should read `done`).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).
````
