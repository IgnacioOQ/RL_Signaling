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

## Editorial review of main_v2.pdf (port step deferred — main.tex preserved as original)

```yaml
status: todo
type: task
id: todo.editorial_review_main_v2
description: Work through the remaining open/partial reviewer checklist items in main_v2.tex and do a holistic editorial pass on main_v2.pdf. The port step (main_v2.tex → main.tex) is intentionally NOT in scope — main.tex must remain untouched as the original-for-sanity-check baseline. Folds in the closed todo.continue_reviewer_revisions task.
owner: user + agent (review is user; mechanical edits are agent)
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
4. **Port step intentionally omitted.** `main.tex` is preserved untouched as the original-for-sanity-check baseline. `main_v2.tex` remains the working revision. Do NOT delete, overwrite, or rename `main.tex`. (If a future session wants to retire `main.tex`, rename it to `original.tex` first — never `rm` it.)
5. **Optional follow-ups:**
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
- `latexmk -pdf main_v2.tex` produces `main_v2.pdf` with no errors (font warnings OK). Current state as of 2026-05-19: 30 pages, all reviewer checklist items at `[x]`, R2·C1–R3·C7 all done.
- `main.tex` is left untouched (no port performed); the file remains identical to its state at session start and can be used as the original-for-sanity-check baseline.

**On completion:** Delete this entire task block. Append a `WORKLOG.md` entry recording the editorial decisions made along the way and noting that the port step was deferred at user request.

---

## Word-count reduction pass on main_v2.tex

```yaml
status: todo
type: task
id: todo.word_count_reduction
description: Reduce main_v2.tex word count to meet the journal's word limit. The current revision is over the limit; identify candidate sections for trimming and execute trims under user direction. The journal word limit is TBD — user to fill in before starting.
owner: user + agent
blocked_by: []
last_checked: '2026-05-19'
```

**Context.** The 2026-05-19 R3·C4–C7 round expanded several sections (R3·C4 alignment paragraph at §1.1, R3·C5 four-paragraph §2.1 rewrite, R3·C6 (a)/(b)/(c) opener at §4, R3·C7 terminology parentheticals + structured roadmap), and current state is 30 pages, ~2.9 MB. The journal's word limit is exceeded. **Insert the actual word limit here before starting the pass** — currently TBD pending user input.

Likely highest-value trim candidates, in rough priority (flagged during the 2026-05-19 holistic editorial pass):
- §2.3 "Reading" paragraph at [main_v2.tex:514](manuscript/main_v2.tex#L514): the second half compares the current framing to a previous draft using the `init_weights=(1,0)` corner case; a reader unfamiliar with the previous draft does not need this. Trim or move to a footnote.
- §2.3 "The figure" paragraph at [main_v2.tex:502](manuscript/main_v2.tex#L502): some duplication with the figure caption. Could be tightened.
- §1.2 multi-paragraph Gilbert / Huttegger / shared-goal block at [main_v2.tex:188–197](manuscript/main_v2.tex#L188): three long paragraphs; some compression possible.
- §1.1 line 184 alignment paragraph: long after the R3·C4 expansion. Could split or compress the complete-conflict mechanism explanation.
- §4 meta-openers at [main_v2.tex:637, 641](manuscript/main_v2.tex#L637): two consecutive epistemic-status paragraphs (R2·C1 + R3·C6). Could merge or shorten.

**Preconditions:**
- All reviewer-checklist items at `[x]` (already true as of 2026-05-19).
- Journal word limit confirmed with user before starting.

**Steps:**
1. Record current word count (`texcount manuscript/main_v2.tex` or equivalent) and target limit; compute the gap.
2. With the user, prioritize the candidate sites above and any others they identify.
3. For each chosen site: propose 2–3 trim options per REVISION_WORKFLOW Phase 3, apply, recompile, re-verify that response-narrative quotations still align (especially R2·C2's verbatim §2.3 quotations — see [Reviewers Responses Checklist.md](manuscript/reviewers/Reviewers%20Responses%20Checklist.md) §"§2.3-dependent response entries" for the verbatim/paraphrase/pointer ledger).
4. Re-run `texcount` after each round; stop when below the limit with reasonable margin.
5. Update the response narrative entries that quote §2.3 verbatim if those passages are trimmed.

**Verification:**
- `texcount manuscript/main_v2.tex` reports a word count at or below the journal's stated limit.
- `latexmk -pdf main_v2.tex` still produces a clean build (currently 30 pages).
- All `verbatim` and `paraphrase` ledger entries in [Reviewers Responses Checklist.md](manuscript/reviewers/Reviewers%20Responses%20Checklist.md) §"§2.3-dependent response entries" still cite content that exists in §2.3.

**On completion:** Delete this entire task block. Append a `WORKLOG.md` entry recording the pre/post word counts and the sites that were trimmed.

---

## Authenticity / voice-revision pass on main_v2.tex and reviewer responses

```yaml
status: todo
type: task
id: todo.authenticity_voice_pass
description: Iterative pass(es) where the user reads main_v2.tex and the reviewer responses, identifies sentences that read as LLM-generated or off-voice, and works with the agent to rewrite them in the user's philosophical-essay voice. Spans multiple sessions; not a single-sitting task. Goal is to assert authenticity and style across the manuscript and the reviewer-facing prose.
owner: user + agent (user identifies sites; agent rewrites under user direction)
blocked_by: []
last_checked: '2026-05-19'
```

**Context.** The reviewer-response edits (R2·C1–C5, R3·C1–C7, completed 2026-05-19) were drafted with substantial agent assistance and, while accurate, may read in places as more uniform and structured than the user's natural philosophical-essay voice. The user wants iterative passes to (a) restore authorial voice, (b) reduce the "generated" feel, and (c) assert authenticity. Same applies to the reviewer-facing prose in [manuscript/reviewers/Generated Responses to Reviewers.md](manuscript/reviewers/Generated%20Responses%20to%20Reviewers.md). The standing feedback memory `feedback_paper_work.md` already records "didactic + philosophical voice" and "iterate slowly" as principles; this task is the operational vehicle for applying them across the revised paper.

LLM-isms to watch for during rewrites: frequent emphasis markers (`\emph{}` overload), parenthetical lists of three, formulaic "First, … Second, … Third, …" sequences, hedging stacks ("this suggests … which may indicate … such that it appears"), redundant "in other words" rephrasings, abstract-Latinate diction where Anglo-Saxon would do, and `---`-as-rhythm (already partly tracked by `todo.dash_sweep`).

**Preconditions:**
- All reviewer-checklist items at `[x]` (already true as of 2026-05-19).
- Coordinated with `todo.word_count_reduction` so revoicing isn't undone by later trims (and vice versa). Recommended ordering: voice pass first on the sites you most care about, then word-count reduction, then a quick re-voice on anything the trim flattened — but the order can be inverted if the word-count gap is urgent.

**Steps:**
1. Per session, the user picks a section or paragraph to revoice (often by reading the PDF and flagging sites that feel off).
2. Agent loads the affected `.tex` passage (or `.md` response passage), re-paste the surrounding prose, and proposes 2–3 rewrite options that keep the substance and trim the LLM-isms listed above.
3. User selects an option or supplies their own; agent applies.
4. Recompile (`latexmk -pdf main_v2.tex`) after each session's edits.
5. Repeat across sessions until the user is satisfied. Track sites already revoiced in a session-end `WORKLOG.md` entry so the next session can pick up without re-treading covered ground.

**Verification:**
- User satisfied with voice across the manuscript and the responses (subjective — no objective stopping criterion).
- `latexmk -pdf main_v2.tex` still produces a clean build.
- Reviewer-response checklist quotations still align with what's in the manuscript ([Reviewers Responses Checklist.md](manuscript/reviewers/Reviewers%20Responses%20Checklist.md) §"§2.3-dependent response entries" re-verified per round).

**On completion:** Delete this entire task block at user discretion (there is no objective stopping criterion beyond user satisfaction). Append a `WORKLOG.md` entry summarizing the revoicing work done across sessions.

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

## Build the revision-tooling toolkit (scripts + templates)

```yaml
status: todo
type: task
id: todo.revision_tooling
description: Implement the cross-paper revision-tooling toolkit identified in the 2026-05-19 meta-knowledge session — five items (four small Python scripts + one templates directory) that give future paper revisions short-loop drift checks, word-count tracking, dash auditing, bib hygiene, and drop-in starter files. The KB-skill component (item 1 in the original five-item list) is out of scope here; it lives at `content/how-to/REVISION_RESPONSE_SKILL.md` in the knowledge base and is captured separately by KB updates as session lessons accumulate.
owner: agent (under user direction)
blocked_by: []
last_checked: '2026-05-19'
```

**Context.** The 2026-05-19 PHOS-17993 revision exposed five recurring operational frictions that a small toolkit could remove for any future paper revision: (a) drift between checklist `[x]` items, the response-narrative, and the actual `.tex` source; (b) word-count tracking against a journal limit without an easy per-section breakdown; (c) em-dash auditing across the manuscript and reviewer-facing prose; (d) bibliography hygiene before submission; (e) a clean starting point for the two parallel revision artifacts (operational checklist + formal response document). This task builds all five.

**Naming, location, design constraints:**
- **Location**: scripts at `scripts/` at the repo root (top-level), templates at `templates/revision/`. The repo already has `analytics/scripts/` for paper analytics — that directory is for this paper's analytics only. The `scripts/` directory at top-level is for *cross-paper* tooling that operates on the manuscript/response orbit. Confirm this split with the user at the start of the task before creating the directories.
- **Language**: Python 3.10+. No third-party dependencies beyond what's already in `pyproject.toml` `[dev]` extras (currently `pytest`, `scikit-optimize`). The four scripts should be standalone — no shared internal module — so they can be copied into another paper's repo individually.
- **CLI style**: all scripts use `argparse`, accept `--help`, return non-zero exit code on validation failure (so they can be wired into CI later), and support `--json` for machine-readable output.
- **No coupling to PHOS-17993**: the scripts must work on any paper structured around `main_v2.tex` + `Appendix.tex` + `References.bib` + `manuscript/reviewers/*.md`. File paths are arguments, not hardcoded.
- **Testing**: include a `tests/test_revision_tooling.py` file with smoke tests per script. Use `pytest` matching the project style.

**Preconditions:**
- `todo.editorial_review_main_v2` is complete (PHOS-17993 R2/R3 work landed) — done as of 2026-05-19.
- User has confirmed the top-level `scripts/` location vs the existing `analytics/scripts/` (asked at task start).
- Python 3.10 venv + project dev dependencies installed (`pip install -e ".[dev]"`).

---

### Tool 2 — `scripts/response_align.py`

**Purpose.** Catch drift between the three coupled revision artifacts: the operational checklist, the formal response document, and the manuscript source.

**Inputs:**
- `--checklist <path>` — path to a markdown file with reviewer-checklist structure (the convention used in this repo: `### [R<n>·C<m>]` headings, `[x]`/`[ ]`/`[~]` checkboxes, optional sub-bullets).
- `--responses <path>` — path to the formal response document (markdown with `--- Comment <n>: ... ---` section markers, optionally containing italicized verbatim quotations as `*"..."*`).
- `--manuscript <path>` (one or more) — `.tex` source file(s). Multiple paths allowed (e.g., `main_v2.tex` + `Appendix.tex`).
- `--json` (optional) — emit machine-readable JSON instead of human-readable text.

**Outputs.** A drift report with four sections:
1. **Checklist items marked `[x]` without a corresponding response-narrative entry.** Lists `[R<n>·C<m>]` IDs.
2. **Response-narrative entries without a corresponding checklist item.** Inverse direction.
3. **Verbatim quotes in the response narrative that do not appear in the manuscript.** Extracts `*"..."*` strings from the response document, normalizes whitespace, greps the `.tex` source(s). Reports each missing quote with the response-document line number.
4. **Sub-bullet checklist items marked `[x]` whose notes reference line numbers (e.g., `[main_v2.tex:184]`) that no longer point at the documented content.** Reads the line, applies fuzzy match against the bullet's "now reads" / "has been replaced with" quoted strings.

**Exit code**: 0 if no drift, non-zero if any of the four checks fails.

**Sketch of CLI usage:**
```bash
python scripts/response_align.py \
  --checklist "manuscript/reviewers/Reviewers Responses Checklist.md" \
  --responses "manuscript/reviewers/Generated Responses to Reviewers.md" \
  --manuscript manuscript/main_v2.tex manuscript/Appendix.tex
```

**Implementation notes:**
- Parse the checklist with a small regex pass — `### \[R\d+·C\d+\]` for top-level items, `^- \[x\]` for sub-bullets.
- Parse the response document by splitting on `--- Comment \d+:` and `--- Overall comment ---` markers; extract italicized double-quoted strings via `\*"([^"]+)"\*`.
- For the manuscript grep, strip LaTeX commands (`\emph{}`, `\cite{}`, etc.) before substring matching; whitespace-normalize both sides. A simple regex pass is fine — `(\\\w+\{[^}]*\}|\s+)` collapsed to a single space.
- Surface false positives clearly (e.g., quotes that span multiple `\paragraph{}` blocks may fail to match if the source uses macros the script didn't strip). Report the original quote and the closest match.

---

### Tool 3 — `scripts/word_count.py`

**Purpose.** Thin wrapper around `texcount` that reports overall + per-section word counts, abstract / body / footnotes breakdown, and comparison to a target limit. Useful for `todo.word_count_reduction` and for ongoing tracking during a revision.

**Inputs:**
- `<tex_file>` (positional) — path to the `.tex` to count.
- `--target <int>` (optional) — journal word limit; reports overage/underage.
- `--include <body|footnotes|abstract|appendix|all>` (optional, default `body`) — what to count. Multiple allowed.
- `--per-section` (flag) — emit per-section breakdown (one row per `\section`/`\subsection`).
- `--json` (optional) — machine-readable output for diffing across revisions.

**Outputs.** Human-readable table with:
- Total word count (body, abstract, footnotes, appendix, sum).
- Per-section breakdown (if `--per-section`).
- Target comparison if `--target` set (e.g., "9,103 words — 103 over target").

**Implementation notes:**
- Wrap `texcount -sec -nosub` (or equivalent flags) via subprocess. Parse texcount's output.
- For the per-section breakdown, use `texcount -inc -sec`.
- Footnotes count via `texcount -relax`. Abstract via parsing `\begin{abstract}...\end{abstract}` explicitly (texcount can miscount inside abstract environments depending on flags).
- Exit 0 if under target, exit 1 if over.

**Sketch of CLI usage:**
```bash
python scripts/word_count.py manuscript/main_v2.tex --target 9000 --per-section
```

**Optional follow-on**: integrate into a pre-commit hook that warns when the body word count rises by more than 5% in a single commit (defer to a separate task if you want this).

---

### Tool 4 — `scripts/dash_audit.py`

**Purpose.** Print all `---` (LaTeX em-dash) and `—` (Unicode em-dash) occurrences in `.tex` and `.md` files with surrounding context. Used by `todo.dash_sweep` and during the authenticity-voice pass.

**Inputs:**
- File paths or globs (positional, one or more).
- `--exclude-pattern <regex>` (optional, repeatable) — skip occurrences matching this regex in their context (e.g., to skip dashes inside `\citep{...--...}` bibkeys).
- `--context <int>` (optional, default 50) — characters of surrounding context to display.
- `--count-only` (flag) — emit only per-file counts, no individual occurrences.
- `--json` (optional) — machine-readable output.

**Outputs.** For each file:
- Total dash count, split by type (`---` vs `—`).
- Each occurrence as `path:line:col` with ±context chars surrounding.

**Implementation notes:**
- Use Python's `re` for the scanner; iterate files via `pathlib.Path.glob` if globs are passed as positional args.
- For `.tex` files, optionally strip comments (`% ...\n`) before scanning, to avoid dashes in commented-out prose.
- Sort output by file then by line number.
- Provide a useful default exclude pattern for bibkeys like `^.{0,80}--.{0,80}$` matching inside `\citep{}` — but accept overrides.

**Sketch of CLI usage:**
```bash
python scripts/dash_audit.py manuscript/main_v2.tex "manuscript/reviewers/*.md"
python scripts/dash_audit.py manuscript/main_v2.tex --count-only
python scripts/dash_audit.py manuscript/main_v2.tex --exclude-pattern '\\citep\{[^}]*--[^}]*\}'
```

---

### Tool 5 — `templates/revision/` directory with skeleton files

**Purpose.** A drop-in skeleton set for a new paper revision. Copying `templates/revision/*` into a fresh paper's `manuscript/reviewers/` directory should give a structurally complete starting point that can be filled in.

**Contents:**
1. **`Reviewers_Responses_Checklist_TEMPLATE.md`** — skeleton for the operational tracker. Carries the structural conventions from PHOS-17993's checklist:
   - YAML frontmatter (TBD placeholders for description, repository).
   - "Status legend" block (`[x]` / `[~]` / `[ ]`).
   - "Top-line progress" table with one row per `R<n>·C<m>` comment (placeholder rows).
   - "Section-dependent response entries" subsection (the verbatim/paraphrase/pointer ledger pattern).
   - Per-comment template: heading + "Reviewer's concern" pull-quote + "Actions" subsection with sub-bullet checkboxes.
   - "Notation drift introduced during the revision" subsection (initially empty, filled in as the revision progresses).
   - "Build / verification" subsection with `latexmk -pdf -interaction=nonstopmode` recipe.

2. **`Generated_Responses_to_Reviewers_TEMPLATE.md`** — skeleton for the formal response document. Carries the structural conventions from PHOS-17993's response narrative:
   - Top-line "Response to Reviewers" header + manuscript ID placeholder.
   - Per-reviewer section markers (`===== Reviewer #<n> =====`).
   - Per-comment sub-section markers (`--- Comment <n>: <title> ---`).
   - "Reviewer:" / "Response:" / "Concretely, ... (before/after quotations)" three-paragraph pattern. Each paragraph includes a TBD placeholder explaining the intent.
   - Final "Paper Changes Checklist" section that mirrors the operational checklist's structure for easy cross-reference.

3. **`paper_TEX_REF_TEMPLATE.md`** — skeleton for a project-local LaTeX-conventions reference. Carries the structural conventions from this repo's `LP_TEX_REF.md`:
   - "File layout" section (placeholders for the per-paper file set).
   - "Section structure" table (placeholder; instructions to populate via the grep command).
   - "Authoring conventions" section (citation style, line discipline, figure paths — sub-headers as placeholders).
   - "Citations and bibliography" subsection.
   - "Figures" subsection.
   - "Math notation" subsection.
   - "Compile and display" subsection.
   - "Build artifacts" subsection (the `.gitignore` block recipe).
   - "Gotchas" subsection (initially empty, filled as paper-specific traps surface).

Each template should be self-contained (no `\input{}` to other templates) and carry a brief `<!-- TEMPLATE-INSTRUCTIONS: ... -->` HTML-comment block at the top explaining what to fill in and the order of operations.

**Implementation notes:**
- Strip everything paper-specific from the PHOS-17993 originals: no R2·C1 / R3·C4 etc. identifiers, no `main_v2.tex` references, no Argiento / Skyrms / Lewis content. Leave the *structure* and *normative shape* but replace prose with `[TBD: ...]` placeholders that describe what should go there.
- Include a short `README.md` in `templates/revision/` explaining the order of operations: (1) copy templates into the new paper's `manuscript/reviewers/`, (2) populate the YAML frontmatter, (3) write the formal narrative first (Phase 1 of REVISION_WORKFLOW), (4) extract the checklist from the narrative (Phase 2), (5) run the Phase 3 sub-loop, (6) use the response_align/word_count/dash_audit/bib_unused scripts as auditors throughout.

---

### Tool 6 — `scripts/bib_unused.py`

**Purpose.** Report bib entries in a `.bib` file that aren't `\cite`'d anywhere in the manuscript sources. Also report the reverse — `\cite` keys used in the manuscript that have no bib entry. Helps trim the bib before submission and catches typos in cite keys.

**Inputs:**
- `<bib_file>` (positional) — path to `.bib`.
- `<tex_files>` (positional, one or more) — `.tex` files to scan for cite keys.
- `--reverse` (flag) — also report unmatched cite keys (default: only unused bib entries).
- `--strict` (flag) — fail with non-zero exit if any drift exists in either direction.
- `--json` (optional).

**Outputs.**
- Section A: bib entries defined but never `\cite`'d. One per line.
- Section B (if `--reverse` or `--strict`): cite keys used but not defined in the bib. One per line.

**Implementation notes:**
- Parse the `.bib` file with a simple regex for `@\w+\{(\w+),` to extract keys. Don't try to do a full BibTeX parse; this is good-enough for the common case.
- Parse `\tex` files for `\cite[p]?\{...\}` and `\citet?\{...\}` calls; split on commas inside the braces; collect all keys.
- Case-sensitive matching (bibkeys are case-sensitive).
- Exit 0 unless `--strict` and either drift direction has hits.

**Sketch of CLI usage:**
```bash
python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex manuscript/Appendix.tex
python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex --reverse --strict
```

---

### Steps:

1. **Confirm directory layout with user** at task start: `scripts/` at the repo root for these cross-paper tools, vs the existing `analytics/scripts/` for this paper's analytics. Confirm `templates/revision/` for the template skeletons.
2. **Implement `response_align.py`** (Tool 2). Highest priority — catches the most insidious bug class (silent drift between the three docs).
3. **Implement `word_count.py`** (Tool 3). Wires into `todo.word_count_reduction` immediately.
4. **Implement `dash_audit.py`** (Tool 4). Wires into `todo.dash_sweep` immediately.
5. **Implement `bib_unused.py`** (Tool 6). Useful as a final-pre-submission check.
6. **Build the `templates/revision/` skeletons** (Tool 5). Most labor-intensive; lowest urgency since no second paper is in flight. Do this last unless a second paper appears.
7. **Write smoke tests for each script** in `tests/test_revision_tooling.py`. Each script: at least one positive test (clean input, no drift / under target / no dashes / no unused bib) and at least one negative test (one drift case, returns expected non-zero).
8. **Add a top-level `scripts/README.md`** documenting each tool's purpose, CLI, exit codes, and the recommended invocation order during a revision (response_align after every round; word_count and dash_audit on demand; bib_unused as a pre-submission check).
9. **Update [content/workflows/REVISION_WORKFLOW.md](file://content/workflows/REVISION_WORKFLOW.md)** in the KB to mention the tools — a one-paragraph section pointing at `scripts/` and noting which script supports which phase. Use `mcp__kb_mcp__knowledge_base_update` per the KB write protocol.
10. **Update [feedback_paper_work.md](.claude/projects/-Users-ignacio-Documents-VS-Code-GitHub-Repositories-RL-Signaling/memory/feedback_paper_work.md)** with a brief reference to the toolkit so future sessions invoke the scripts proactively.

**Verification:**
- `pytest tests/test_revision_tooling.py` reports all passing.
- Each script's `--help` renders cleanly.
- Running `python scripts/response_align.py` with the PHOS-17993 artifacts reports zero drift (current state of the manuscript and response narrative is the validation set).
- `python scripts/word_count.py manuscript/main_v2.tex` reports a sensible body word count (cross-check by hand on the abstract).
- `python scripts/dash_audit.py manuscript/main_v2.tex --count-only` reports the current dash count (≈38 as of 2026-05-19, may have changed by the time this task runs).
- `python scripts/bib_unused.py manuscript/References.bib manuscript/main_v2.tex manuscript/Appendix.tex` reports the current unused-bib-entries list.
- `templates/revision/` contains three template files + a README; opening any of them in an editor shows a clear structural skeleton with `[TBD: ...]` placeholders for paper-specific content.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry recording the toolkit's launch and noting the validation results from running the scripts against the PHOS-17993 artifacts as the first regression set.

**Design questions to surface to the user before starting:**
- Confirm `scripts/` location at repo root vs alternative (e.g., `tools/`, `revision_tools/`, `manuscript/scripts/`).
- Confirm Python 3.10+ as the minimum version (or specify a different floor).
- Confirm the `--json` output format is desired (vs YAML, vs plain text only).
- Confirm whether the scripts should warn or fail by default when drift is detected (i.e., default to exit-0 with a printed report, or exit non-zero — affects how easy it is to wire into CI later).
- Confirm whether `bib_unused.py` should integrate with the bib hygiene already done by `latex` build (some venues' bibstyles strip unused entries automatically, in which case the script is for the author's pre-submission sanity rather than a build-time check).

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
