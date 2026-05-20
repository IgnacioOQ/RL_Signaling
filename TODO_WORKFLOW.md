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

## Resolve R3·C3 narrative-vs-manuscript verbatim drift (response_align.py finding)

```yaml
status: todo
type: task
id: todo.r3c3_verbatim_drift
description: The R3·C3 response-narrative italic-quotes the four-step signaling-game episode as if it were a verbatim manuscript excerpt, but the wording differs stylistically from main_v2.tex line 137 (semicolons inline vs "First, ... Second, ..." with periods; "Sender" / "Receiver" spelled out vs `\textbf{S}` / `\textbf{R}` after first introduction). scripts/response_align.py surfaces this as the single remaining check-C drift hit on the post-revision artifacts. Either tighten the narrative italic-quote to match the manuscript exactly, or rephrase to mark it as a paraphrase rather than verbatim.
owner: agent + user
blocked_by: []
last_checked: '2026-05-19'
```

**Context.** The 2026-05-19 toolkit-build session ran `scripts/response_align.py` against the post-revision PHOS-17993 artifacts and got `Total drift items: 1` — zero in checks A (checklist→narrative), B (narrative→checklist), D (anchor staleness), and a single hit in check C (verbatim quotes in narrative not found in manuscript):

- **Comment R3·C3** at [Generated Responses to Reviewers.md](manuscript/reviewers/Generated%20Responses%20to%20Reviewers.md): the narrative italic-quotes the four-step signaling-game episode as

  > *"Nature samples a state $x \in X$ according to $P$ and reveals it to Sender; Sender chooses a signal $s \in Sig$ as a function of $x$, $f(x) = s$, and sends it to Receiver (who does not observe $x$); Receiver chooses an action $a \in Ac$ as a function of the received signal, $g(s) = a$; the payoff functions assign rewards $u_S(x, a)$ and $u_R(x, a)$ to sender and receiver."*

  but [main_v2.tex:137](manuscript/main_v2.tex#L137) actually reads "First, Nature samples a state $x \in X$ according to $P$ and reveals it to Sender \textbf{S}. Second, \textbf{S} chooses a signal $s \in Sig$ as a function of $x$, $f(x) = s$, and sends it to Receiver \textbf{R} (who does not observe $x$). Third, \textbf{R} chooses an action $a \in Ac$ as a function of the received signal, $g(s) = a$. Fourth, the payoff functions assign rewards $u_S(x, a)$ and $u_R(x, a)$ to sender and receiver."

  Differences:
  1. **Sentence structure**: manuscript uses ordinal markers "First, … Second, … Third, … Fourth," with periods; narrative uses semicolons inline.
  2. **Role names**: manuscript introduces \textbf{S} / \textbf{R} on first mention and then uses the bold abbreviations; narrative spells out "Sender" / "Receiver" each time.

  All other content (math notation, argument order, conjunction structure) matches. The differences are stylistic — the narrative reads better as a single sentence — but the `*"..."*` italic-quote convention is supposed to signal *verbatim* manuscript content for reviewers.

**Preconditions:**
- All other reviewer-checklist items at `[x]` (true as of 2026-05-19).
- `scripts/response_align.py` currently reports `Total drift items: 1` against the current artifacts; no other revisions should have landed between when this task was filed and when it is picked up (otherwise re-confirm the finding first by re-running the script).

**Steps:**

1. Re-confirm the finding by running:

   ```bash
   python scripts/response_align.py \
     --checklist "manuscript/reviewers/Reviewers Responses Checklist.md" \
     --responses "manuscript/reviewers/Generated Responses to Reviewers.md" \
     --manuscript manuscript/main_v2.tex manuscript/Appendix.tex
   ```

   Expect: 1 hit in check C, R3·C3 only.
2. Re-paste the reviewer's full R3·C3 comment to the user before proposing options — per `feedback_revision_decision_quoting.md`, the reviewer's verbatim comment must accompany every revision-decision moment in this paper. (R3·C3 reads: *"The formal description of the standard signalling game departs from the usual presentation in the literature. For example, the paper describes nature as 'playing a game' with the sender and receiver, whereas the more common formulation treats the signalling game as a tuple consisting of states, messages, actions, strategies, payoff functions, and a probability distribution over states. Using a more standard formulation would make the paper easier to situate within the existing literature."*)
3. Propose 2–3 options (per `feedback_paper_work.md` Rule 3):
   - **Option A — exact verbatim quote.** Replace the narrative italic-quote with the manuscript's "First, Nature... Sender \textbf{S}. Second, \textbf{S} chooses..." wording verbatim, periods and all. Passes `response_align.py` check C cleanly; reads less smoothly as a sentence in the narrative.
   - **Option B — drop italic-quote markers; rephrase as paraphrase.** Reframe the narrative prose as "The text spells out the four-step episode using the tuple notation: Nature samples $x \sim P$ and reveals it to Sender, Sender maps $f(x) = s$, Receiver maps $g(s) = a$, and the payoff functions assign $u_S(x, a)$ and $u_R(x, a)$." No `*"..."*` markers. Keeps the smooth reading; loses the "this is verbatim from the new manuscript" signal.
   - **Option C — verbatim quote of a tighter manuscript fragment.** Quote the parts that ARE word-for-word identical between narrative and manuscript (the closing payoff sentence, or a math-only fragment) inside `*"..."*`, and let the rest be ordinary narrative prose.
4. User picks an option (or supplies their own); agent applies the edit to [Generated Responses to Reviewers.md](manuscript/reviewers/Generated%20Responses%20to%20Reviewers.md).
5. Re-run `scripts/response_align.py`; confirm `Total drift items: 0`.

**Verification:**
- `python scripts/response_align.py [...]` exits with `Total drift items: 0`.
- The R3·C3 narrative either italic-quotes manuscript text verbatim, or makes no verbatim-quote claim about the four-step episode.

**On completion:** Delete this entire task block. Append a one-paragraph note to `WORKLOG.md` recording which option was chosen and confirming check C is clean.

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
