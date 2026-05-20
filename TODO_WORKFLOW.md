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
last_checked: '2026-05-20'
---

# TODO Workflow

Cross-session task backlog. Tasks are added here when work started in a session cannot be completed immediately. Each task must be fully self-contained — a fresh agent should be able to pick it up using only the task body and the kb_mcp tools, with no additional context required.

This file is the per-repository instance of the `TODO_WORKFLOW_TEMPLATE.md` pattern. It lives at the root of the working repository alongside `worklog.jsonl` and is intentionally **not registered with kb_mcp** — agents access it via the regular filesystem `Read`/`Edit` tools, not via `knowledge_base_*` calls.

**Agent rules (picking up tasks):**
1. Read each task in full before starting. If its preconditions are unmet, skip it and note the blocker.
2. After completing a task, delete its entire block from this file (from the `---` divider above the `##` header through the `---` divider below the last line of the task body).
3. After completing one or more tasks, assess whether a `worklog.jsonl` entry is warranted (schema and append protocol in the "Worklog" section below) — see Phase 6 of `content/workflows/CODING_AGENT_MAIN_WORKFLOW.md`.
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
   - A new entry in `worklog.jsonl`, or
   - A standalone `REPRODUCIBILITY.md` at the repo root if the audit is large enough to warrant its own document.
8. Update the README's "Reproducing the figures" section if any step requires extra manual setup that the current text does not document.
9. **Optional but recommended:** migrate the multiprocessing seeding pattern to `numpy.random.SeedSequence().spawn()` so individual rows of the saved CSVs are row-reproducible from `iteration` alone. See `content/how-to/NOTEBOOK_WRITING_SKILL.md` Section 8 ("Parallel processing — Seeds across workers") for the recommended pattern. If deferred, file a separate task.

**Verification:**
- `git status` after a fresh end-to-end run shows clean modifications only to expected files (CSVs in `results/`, PNGs in `results/`, optionally notebook output cells).
- A diff between pre-fix and post-fix figures is documented in `worklog.jsonl` or `REPRODUCIBILITY.md`.
- `pytest tests/` still passes.
- The README "Reproducing the figures" section reflects the current procedure with no inaccuracies.
- `LEGACY_ERRORS_LOG.md` is updated: every `UNREPRODUCIBLE` verdict is replaced with either `CLEAN` (if the post-fix re-run resolved it) or kept with a note explaining why reproducibility is still partial (e.g. multiprocessing-seed row-level non-reproducibility).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a `worklog.jsonl` entry recording the audit results.

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
   Add `nbstripout` to the `[dev]` extras in [pyproject.toml](pyproject.toml). Update [README.md](README.md) to mention the `nbstripout --install` step in the Setup section. Update [notebooks/NOTEBOOKS_README.md](notebooks/NOTEBOOKS_README.md) to note the strip-on-commit convention. Append a `worklog.jsonl` entry summarizing the refactor.
8. **Phase 4 is out of scope for this task.** The plan file already records that decision. The separate `todo.verify_notebook_drive_paths` task (which depends on this one) covers the Drive-path verification needed before any Colab re-run.

**Verification:**
- `python notebooks/_tools/nb_migrate.py audit notebooks/` reports `legacy-API hits: none` and `nbformat=4.5 OK; kernel='rl_signaling' OK` for every notebook.
- `pytest tests/ -v` reports 63 passed.
- Each migrated notebook completes Restart-and-Run-All under `SMOKE_TEST=True` with no errors and no `DeprecationWarning` from the `rl_signaling.*` namespace.
- [README.md](README.md) **Notebooks** and **Reproducing the figures** sections reference the renamed files.
- [.gitattributes](.gitattributes) contains the `nbstripout` filter line and [pyproject.toml](pyproject.toml) `[dev]` extras include `nbstripout`.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a `worklog.jsonl` entry summarizing what changed and noting that `todo.verify_notebook_drive_paths` is now unblocked. Delete [NOTEBOOK_REFACTOR_PLAN.md](docs/code-audit/NOTEBOOK_REFACTOR_PLAN.md) from `docs/code-audit/` once Phase 5 is done (the plan was an in-flight document; the `worklog.jsonl` entry preserves the historical record).

---

## Word-count reduction pass on main_v2.tex

```yaml
status: todo
type: task
id: todo.word_count_reduction
description: Reduce main_v2.tex word count to meet the journal's 9,000-word all-inclusive limit (see docs/JOURNAL_WORD_LIMIT.md). The current revision is substantially over; identify candidate sections for trimming and execute trims under user direction.
owner: user + agent
blocked_by: []
last_checked: '2026-05-19'
```

**Context.** The 2026-05-19 R3·C4–C7 round expanded several sections (R3·C4 alignment paragraph at §1.1, R3·C5 four-paragraph §2.1 rewrite, R3·C6 (a)/(b)/(c) opener at §4, R3·C7 terminology parentheticals + structured roadmap), and current state is 30 pages, ~2.9 MB.

The journal's limit is **9,000 words, all-inclusive** — confirmed 2026-05-20 from the editor emails archived in [docs/](docs/) and distilled in [docs/JOURNAL_WORD_LIMIT.md](docs/JOURNAL_WORD_LIMIT.md). The count includes body, abstract, footnotes, in-text citations, captions, and figures/tables (table = 167 words; normal figure = 167; side-by-side pair = 1 normal figure; extra-large figure = 334). References and online-only appendices are excluded, and the costly-signals/TD-learning appendix is filed **online-only** (decided 2026-05-20) so it does not count. The journal counts in Microsoft Word — which counts mathematics and so exceeds local `texcount`. At the start of the 2026-05-20 reduction pass `main_v2.tex` was ~10,239 by `texcount` sum plus ~1,500 in figure/table charges, a journal-style total well over 11,500.

**Progress (2026-05-20, session 38).** A first reduction pass cut `texcount` sum 10,239 → 8,336 (−1,903):
- §2.3 restructured — caption trimmed, the two figure-setup paragraphs merged, the three observations stripped of per-parameter numbers, the Reading/honest-version/What-this-is meta-paragraphs folded into one "What the proof of concept shows" closing paragraph (−909).
- §2.1 gutted and retitled "Aims and Learning Rules" — redundant intuition/Gricean paragraphs removed, dual-ambitions + learning-rule paragraphs kept (−232).
- §1.2 Goal-Sharing reduced — Gilbert paragraph compressed (footnote cut), the two RL-positioning paragraphs merged, thesis paragraph light-trimmed (−317).
- §4 Discussion pass, all 9 paragraphs — the two meta-openers merged into one lean taxonomy, the "I apologize" limitation de-fanged, philosophical/Q-vs-RE/future-work paragraphs trimmed (−445).

Reviewer responses re-synced for R2·C2, R3·C5, R2·C1, R2·C4, R3·C6.

**Still over.** Journal-style total ≈ `texcount` 8,336 + ~1,500 figures/tables + the Word-vs-`texcount` math gap ≈ ~10,200 — roughly 1,000–1,200 over 9,000. Remaining candidates:
- Figure 2 (the Q-learning example-run figure): cutting it removes a 167-word figure charge plus its caption — the largest single remaining win.
- §3 opener "A note on the kind of claim made in this section" — reviewer-reply-flavored meta paragraph; trim or cut (touches R3·C2).
- Global em-dash sweep; the defensive opening paragraph at §1 ([main_v2.tex:111](manuscript/main_v2.tex#L111)).
- If still over after these, consider a deeper structural cut or an author-comment note to the editor.

**Deferred:** a consolidated `[main_v2.tex:XXX]` anchor refresh across `responses_to_reviewers.md` and `responses_checklist.md` — the §2.3/§2.1/§1.2/§4 trims shifted line numbers, so anchors in both reviewer docs are knowingly stale until the pass ends.

**Preconditions:**
- All reviewer-checklist items at `[x]` (already true as of 2026-05-19).
- Journal word limit confirmed: **9,000 words all-inclusive** (see [docs/JOURNAL_WORD_LIMIT.md](docs/JOURNAL_WORD_LIMIT.md)); costly-signals/TD appendix filed online-only so excluded.

**Steps:**
1. Record current word count (`texcount manuscript/main_v2.tex` or equivalent) and target limit; compute the gap.
2. With the user, prioritize the candidate sites above and any others they identify.
3. For each chosen site: propose 2–3 trim options per REVISION_WORKFLOW Phase 3, apply, recompile, re-verify that response-narrative quotations still align (especially R2·C2's verbatim §2.3 quotations — see [responses_checklist.md](manuscript/reviewers/responses_checklist.md) §"§2.3-dependent response entries" for the verbatim/paraphrase/pointer ledger).
4. Re-run `texcount` after each round; stop when below the limit with reasonable margin.
5. Update the response narrative entries that quote §2.3 verbatim if those passages are trimmed.

**Verification:**
- The journal-style count (body + abstract + footnotes + captions + figures/tables per [docs/JOURNAL_WORD_LIMIT.md](docs/JOURNAL_WORD_LIMIT.md)) is at or below 9,000 with margin; note `texcount` alone understates this (no math, no per-figure charge).
- `latexmk -pdf main_v2.tex` still produces a clean build (currently 30 pages).
- All `verbatim` and `paraphrase` ledger entries in [responses_checklist.md](manuscript/reviewers/responses_checklist.md) §"§2.3-dependent response entries" still cite content that exists in §2.3.

**On completion:** Delete this entire task block. Append a `worklog.jsonl` entry recording the pre/post word counts and the sites that were trimmed.

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

**Context.** The reviewer-response edits (R2·C1–C5, R3·C1–C7, completed 2026-05-19) were drafted with substantial agent assistance and, while accurate, may read in places as more uniform and structured than the user's natural philosophical-essay voice. The user wants iterative passes to (a) restore authorial voice, (b) reduce the "generated" feel, and (c) assert authenticity. Same applies to the reviewer-facing prose in [manuscript/reviewers/responses_to_reviewers.md](manuscript/reviewers/responses_to_reviewers.md). The standing feedback memory `feedback_paper_work.md` already records "didactic + philosophical voice" and "iterate slowly" as principles; this task is the operational vehicle for applying them across the revised paper.

LLM-isms to watch for during rewrites: frequent emphasis markers (`\emph{}` overload), parenthetical lists of three, formulaic "First, … Second, … Third, …" sequences, hedging stacks ("this suggests … which may indicate … such that it appears"), redundant "in other words" rephrasings, abstract-Latinate diction where Anglo-Saxon would do, and `---`-as-rhythm (already partly tracked by `todo.dash_sweep`).

**Preconditions:**
- All reviewer-checklist items at `[x]` (already true as of 2026-05-19).
- Coordinated with `todo.word_count_reduction` so revoicing isn't undone by later trims (and vice versa). Recommended ordering: voice pass first on the sites you most care about, then word-count reduction, then a quick re-voice on anything the trim flattened — but the order can be inverted if the word-count gap is urgent.

**Steps:**
1. Per session, the user picks a section or paragraph to revoice (often by reading the PDF and flagging sites that feel off).
2. Agent loads the affected `.tex` passage (or `.md` response passage), re-paste the surrounding prose, and proposes 2–3 rewrite options that keep the substance and trim the LLM-isms listed above.
3. User selects an option or supplies their own; agent applies.
4. Recompile (`latexmk -pdf main_v2.tex`) after each session's edits.
5. Repeat across sessions until the user is satisfied. Track sites already revoiced in a session-end `worklog.jsonl` entry so the next session can pick up without re-treading covered ground.

**Verification:**
- User satisfied with voice across the manuscript and the responses (subjective — no objective stopping criterion).
- `latexmk -pdf main_v2.tex` still produces a clean build.
- Reviewer-response checklist quotations still align with what's in the manuscript ([responses_checklist.md](manuscript/reviewers/responses_checklist.md) §"§2.3-dependent response entries" re-verified per round).

**On completion:** Delete this entire task block at user discretion (there is no objective stopping criterion beyond user satisfaction). Append a `worklog.jsonl` entry summarizing the revoicing work done across sessions.

---

## Worklog (`worklog.jsonl`) — Schema & Append Protocol

Each session that does non-trivial work appends one JSON object as a new line to `worklog.jsonl` at this repository's root. The file is plain JSONL — one JSON object per line, **oldest first** (chronological append order). It lives at the repo root, outside any docs-discovery surface (kb_mcp, search indexers). There is no helper script; agents construct and append the JSON directly.

`worklog.jsonl` already exists in this repo: 32 entries were migrated from the former markdown `WORKLOG.md` on 2026-05-20, oldest first (`session_id` 1–32). The next session's `session_id` is one past the last line's — currently `33`.

### Schema (`schema_version: 1`)

```json
{
  "schema_version": 1,
  "entry_id":      "YYYY-MM-DD-s1",
  "date":          "YYYY-MM-DD",
  "session_id":    1,
  "summary":       "One-line task summary",
  "body_markdown": "- **Task:** ...\n- **Outcome:** ...\n- **Key decisions:** ...\n- **KB changes:** ...\n- **Follow-up:** ..."
}
```

| Field | Type | Notes |
|:--|:--|:--|
| `schema_version` | int | Currently `1`. Bump on breaking changes. |
| `entry_id` | string | Unique across the file. `YYYY-MM-DD-s{N}` when `session_id` is set; plain `YYYY-MM-DD` otherwise. Same-key collisions get `-b` / `-c` / `-d` suffixes. |
| `date` | string | ISO `YYYY-MM-DD`. |
| `session_id` | int \| null | Sequential session counter — last entry's `session_id` + 1. Use `null` if the repo does not track sessions. |
| `summary` | string | One-line heading — what the session accomplished. |
| `body_markdown` | string | Full narrative (Task / Outcome / Key decisions / KB changes / Follow-up) as one opaque markdown blob. The inner bullet structure is convention, not schema. Newlines inside the string must be JSON-escaped as `\n` — `json.dumps` does this automatically. |

### Append protocol

1. **Find the next `session_id`** — read the last line of `worklog.jsonl` (returns `1` if the file is empty):

   ```bash
   if [[ -s worklog.jsonl ]]; then
     tail -1 worklog.jsonl | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print((d.get('session_id') or 0)+1)"
   else
     echo 1
   fi
   ```

2. **Construct the entry as a single-line JSON object.** `json.dumps(entry, ensure_ascii=False)` handles all escaping. Verify `entry_id` is unique against existing entries — if it collides, append `-b` / `-c` / `-d`.
3. **Append the line** with Python (constructs, escapes, and appends in one shot):

   ```bash
   python3 - <<'PY'
   import json
   entry = {
       "schema_version": 1,
       "entry_id":      "YYYY-MM-DD-sN",   # N = the integer from step 1
       "date":          "YYYY-MM-DD",
       "session_id":    None,              # set to that same integer
       "summary":       "...",
       "body_markdown": "- **Task:** ...\n- **Outcome:** ...",
   }
   with open("worklog.jsonl", "a", encoding="utf-8") as f:
       f.write(json.dumps(entry, ensure_ascii=False) + "\n")
   PY
   ```

   A filesystem `Edit` append works too — add the new single-line JSON object after the last existing line.

4. **Skip the worklog append** for trivial one-line changes or purely exploratory sessions with no concrete output.

### Reading back

Render the latest N entries for context loading:

```bash
tail -3 worklog.jsonl | python3 -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"
```

Or list entry headlines with `jq`:

```bash
jq -r '"\(.entry_id): \(.summary)"' worklog.jsonl | tail -10
```

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
