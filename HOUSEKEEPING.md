# RL_Signaling Housekeeping
- status: active
- type: workflow
- id: rl_signaling.housekeeping
- description: Recurring sanity check for the RL_Signaling repository — runs static quality checks, unit tests, and repo-level health checks; appends an audit report.
- label: [core, agent]
- injection: excluded
- volatility: evolving
- scope: project-specific
- last_checked: 2026-08-13
<!-- content -->
Per-repository housekeeping workflow for `RL_Signaling`. Run periodically or after any significant batch of changes. The "Latest Report" section at the bottom is the baseline for the next run; demote it to "Previous Report" before appending a new one.

The workflow covers four concerns regardless of stack: (1) static quality of the codebase (format, lint), (2) unit test correctness, (3) repository-level health (dependency drift, dead code, documentation freshness), and (4) an append-only audit trail in this file.

**Execution model:** sequential — each phase has an explicit exit criterion and a remediation step.

**Prerequisites:**
- Python 3.10+ available; project dependencies installed via `pip install -e ".[dev]"` (the `dev` extras pull in `ruff` and `pytest`).

> The per-repo `worklog.jsonl` and `TODO_WORKFLOW.md` that this workflow used to write to were retired on 2026-07-31; tasks and session history now live in the central planner store. Steps below that referred to them record their findings in the "Latest Report" section of this file instead.

---

## Phase 1 — Context Load

**Goal:** Identify the current state of the codebase before running checks.

### Step 1 — Discover the toolchain

The canonical commands for this repo:

| Concern | Command |
|---|---|
| Format check | `ruff format --check .` |
| Lint (package) | `ruff check rl_signaling/` — **this is the gate; it must stay clean** |
| Lint (repo-wide) | `ruff check .` — informational only; see the note below |
| Type check | n/a — `mypy` is not a project dependency and there is no type gate |
| Unit tests | `pytest tests/` |
| Build smoke | n/a — this is a research codebase, not a build artifact |
| Dependency audit | `pip list --outdated` (or `pip-audit` if installed) |

**On repo-wide lint.** `ruff check .` reports several hundred findings, essentially all in `analytics/scripts/` and `notebooks/`-derived code — missing docstrings, imports below the top of the file, long lines. These are notebook-shaped by nature and are **not** treated as defects. The published package `rl_signaling/` is clean and is the only lint gate. Do not "fix" the auxiliary scripts into compliance; that is churn, not quality.

There is no type gate. `mypy` is not in the `dev` extras, and per the academic-repo convention this repository does not carry CI, merge gates, or type enforcement — the unit of correctness is the figure, and the honest test is a green `pytest` plus a Restart-and-Run-All of the notebooks.

### Step 2 — Read the prior baseline

Read the "Latest Report" section at the bottom of this file. Note the previous test counts, lint counts, and any unresolved follow-ups.

**Exit criterion:** Toolchain commands confirmed; prior baseline is loaded.

---

## Phase 2 — Static Quality Checks

**Goal:** Verify the codebase is clean before exercising it.

### Step 1 — Format check

```bash
ruff format --check .
```

### Step 2 — Lint

```bash
ruff check rl_signaling/     # the gate — must be clean
ruff check .                 # informational — record the count, do not chase it
```

### Step 3 — Type check

n/a — no type gate in this repository (see Phase 1). Record `n/a` in the report.

### Step 4 — Remediation

- **Format errors:** `ruff format --check .` currently reports most files as unformatted, because the notebook-derived scripts were never run through `ruff format`. Do **not** run a repo-wide `ruff format .` — it would produce a large diff across research code that is deliberately left in its original shape. Format only files you are already editing.
- **Lint errors in `rl_signaling/`:** fix in source. Do not silence with `# noqa` unless the disable is documented and justified in the same change.
- **Lint findings elsewhere:** leave them. See the Phase 1 note.

**Exit criterion:** `ruff format --check` and `ruff check rl_signaling/` are clean. Repo-wide `ruff check .` is informational; compare its count against the previous report and note any large jump, which usually means new notebook-derived code landed.

---

## Phase 3 — Tests

**Goal:** Verify the published model still behaves as the paper reports, and that the suite has not silently shrunk.

### Step 1 — Unit tests

```bash
pytest tests/ -v
```

### What the suite actually covers

This is not a generic smoke suite — it tests the **model the paper reports**, so a regression here is a threat to a published claim, not just a code defect. Treat any failure as blocking.

| File | Covers | Why it matters to the paper |
|---|---|---|
| `test_agents.py` | The three learning rules — `UrnAgent` (Roth–Erev), `QLearningAgent`, `TDLearningAgent` — plus `ε-greedy` / `softmax` / `UCB` selection, urn clamping at zero, and initialization seeding | These are the agents §3 compares; the initialization-seeding tests pin the `init_weights` behaviour the proof-of-concept section turns on |
| `test_env.py` | `MultiAgentEnv` lifecycle: graph validation, observation construction, the signal step, full- vs partial-information regimes, and the costly-signaling null-signal append | The three information regimes are the paper's independent variable |
| `test_info_theory.py` | NMI identities on hand-built tables: perfect correlation → 1, independence → 0, degenerate cases, and the unit interval | NMI is the paper's measure of informative signaling |
| `test_numerical_sanity.py` | Hand-derived analytical answers checked against the implementation — entropy in bits (log base 2), the Q-learning single-update and ten-update closed forms, TD one-step bootstrap with and without a terminal, costly-signal cost arithmetic, and urn convergence under full information | The strongest tests in the repo: each states the analytical answer first, so a silent change in the learning math cannot pass |
| `test_golden.py` | Deterministic regression against `tests/golden/baseline.json` (seed 12345) | Catches any change that perturbs a published run |
| `test_smoke.py` | End-to-end, 100 episodes per agent, with and without signals | Integration across env + agent + simulation |

**Known coverage gap:** the suite exercises the mechanism, not the paper's *findings*. No test asserts that signalling raises reward or NMI in the partial-information regime — that is a statistical claim over 10,000-run distributions, and the evidence for it is the committed datasets plus the figures traced in `results/MANIFEST.md`, not pytest.

### Step 2 — Integration / end-to-end tests

`test_smoke.py` covers the package end-to-end. Beyond that, the notebooks are the integration surface; the honest equivalent of "integration tests pass" is a Restart-and-Run-All. After any change to `rl_signaling/agents.py`, `env.py`, or `simulation.py`, spot-check `notebooks/basic_unit_test.ipynb`; after a structural change, sweep the notebooks:

```bash
for nb in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "$nb"
done
```

`notebooks/Parameter_Optimization_wchoices.ipynb` is excluded from that sweep in practice — it writes to a Google Drive path and was run on Colab (see `results/MANIFEST.md`, Gap 4).

### Step 3 — Test count comparison

**Expected counts: 63 tracked, 80 collected locally.** The 17-test difference is `tests/test_revision_tooling.py`, which exercises the LaTeX revision toolkit under `scripts/`. Both are gitignored on purpose — kept on disk as reusable tooling for a future paper, but not part of what this repository publishes (see `.gitignore`, "Manuscript-support tooling"). A local run of `pytest tests/` therefore reports **80**; a fresh clone reports **63**. Neither number is wrong, and the gap is not a finding — do not re-investigate it each run.

Use `git ls-files tests/` to distinguish the two when a count looks off.

Compare pass / fail / skipped counts against the prior "Latest Report." Any regression is a finding to investigate before closing the run. A count that *drops* is only acceptable when tests were deliberately removed with the code they covered — record the reason in the report, as with the 80 → 63 drop when the LaTeX revision toolkit left the repository.

**Exit criterion:** All tests pass. Test count is steady or higher than the prior report, or a drop is explained in the report.

---

## Phase 4 — Repository Health Checks

**Goal:** Catch slow-burning issues that lint and tests do not surface.

### Step 1 — Dependency drift

```bash
pip list --outdated
# Or, if installed:
pip-audit
```

Verify: `requirements.txt` pins are not stuck on EOL versions; no unaddressed CVE advisories.

### Step 2 — Dead code

```bash
ruff check . --select=F401,F841     # unused imports + unused vars
# Optional: pip install vulture && vulture rl_signaling/
```

### Step 3 — Build smoke

n/a — research codebase.

### Step 4 — Documentation freshness

- Does `README.md` still describe the actual entrypoints, module names, and notebook list?
- Does `results/MANIFEST.md` still match the figures actually present in `results/`?
- Do all relative markdown links still resolve? (A renamed file silently breaks them.)
- Does `.gitignore` still exclude everything listed in `PUBLICATION_CHECKLIST.md`? Run `git ls-files | grep -Ei 'manuscript|slides|reviewer'` — it must return nothing.
- Does this file's "Latest Report" history reference resolved work that should now be marked closed?

**Exit criterion:** No surprising drift. Anything actionable that is out of scope for this run is filed as a task in the planner store with a reproduction step.

---

## Phase 5 — Report & Close

**Goal:** Leave an auditable trail in this file.

### Step 1 — Demote the previous report

Rename the existing `## Latest Report` heading to `## Previous Report` and keep its body verbatim. Older "Previous Report" blocks stay in place, separated by `---` dividers.

### Step 2 — Append a new "Latest Report"

Use the template at the bottom of this file. Fill every section. If a phase did not run (e.g. tests are still `n/a`), record `n/a` rather than deleting the section.

### Step 3 — File follow-ups

If anything was found and not fixed, file it as a task in the planner store with enough context for a fresh agent to pick it up.

### Step 4 — Bump `last_checked`

Update the `last_checked` field in this file's metadata header to today's date.

**Exit criterion:** The "Latest Report" reflects today's run, deferred work is recorded in the planner store, and tests / linters are green or have explicit known-issue annotations.

---

## Quick Reference — Housekeeping Checklist

```
[ ] Phase 1: Toolchain identified, prior baseline read
[ ] Phase 2: Format / lint / type checks — clean
[ ] Phase 3: Unit tests — green; counts steady or improving
[ ] Phase 4: Dependency / dead-code / docs — no surprising drift
[ ] Phase 5: New "Latest Report" appended; deferrals filed in the planner store; last_checked bumped
```

---

## Latest Report Template

Copy the block below and fill it in for each housekeeping run. The most recent block is `## Latest Report`; older blocks are renamed to `## Previous Report`.

````markdown
## Latest Report

**Date:** {{YYYY-MM-DD}}
**Trigger:** {{What prompted this run — routine cadence, post-phase checkpoint, post-merge, etc.}}

### Artifact counts
- Source files: {{N}}
- Lines of code: {{N}}
- Registered tests: {{N | n/a}}

### Static quality
- Format: {{pass | N issues}}
- Lint: {{pass | N errors / M warnings}}
- Type check: {{pass | N errors | n/a}}

### Tests
- Unit: {{N passed / M failed / K skipped | n/a}}
- Comparison vs. previous report: {{steady | +N tests | regression}}

### Repository health
- Dependencies: {{in sync | N outdated | N advisories}}
- Dead code: {{none | N findings}}
- Docs: {{current | N drift items}}

### Notable events
- {{Surprises, root-caused issues, decisions made.}}

### Files modified this run
- {{Path: change}}

### Follow-ups recorded in the planner store
- {{Title — short reason | none}}
````

---

## Previous Report

**Date:** 2026-08-12
**Trigger:** Publication preparation — the article was accepted at *Philosophy of Science*, and the repository was reduced to code-only and made paper-ready. This is the first report; it establishes the baseline.

### Artifact counts
- Tracked files: 135
- Source files (tracked `.py`): 33 — of which 7 are the `rl_signaling/` package
- Lines of code: 3,019 in the package; 11,327 across all tracked Python
- Registered tests: 63

### Static quality
- Format: **41 of 49 files would be reformatted** — pre-existing, concentrated in notebook-derived code. Deliberately not addressed; see Phase 2 Step 4.
- Lint (`rl_signaling/`): **pass — all checks passed**
- Lint (repo-wide): 321 findings, informational. Top rules: D103 undocumented-public-function (80), E402 import-not-at-top (56), E702 multiple-statements (28), I001 unsorted-imports (27). All in `analytics/scripts/` and `notebooks/`-derived code.
- Type check: n/a

### Tests
- Unit: **63 passed / 0 failed / 0 skipped** (~4 s)
- Comparison vs. previous report: n/a — first report. Note the count dropped from 80 when the 17 revision-tooling tests left the repository alongside the LaTeX toolkit they exercised; this is a scope reduction, not a regression.

### Repository health
- Dependencies: several minor versions behind (asttokens, beautifulsoup4, bleach, coverage, debugpy, decorator, …). No EOL pins, no advisories. Not actioned — the committed results were produced under the current pins, and bumping them risks perturbing the golden-run baseline for no scientific gain.
- Dead code: 14 findings (F401 unused-import, F841 unused-variable), all outside the package.
- Docs: current. All relative markdown links across the 24 tracked markdown files resolve (0 broken); 45 pre-existing broken links inherited from the May `analytics/docs` → `analytics/math` migration were repaired in this pass.

### Notable events
- History rewritten with `git filter-repo` to remove the manuscript, referee correspondence, slides, and internal audit trail from all commits. The remote was deleted and recreated; the pre-scrub commits are confirmed unreachable. `.git` fell from 288 MB to 109 MB, tracked files from 226 to 135.
- The manuscript's pre-migration location `analytics/docs/` had to be scrubbed alongside `manuscript/` — it still held a referee response letter. Scrubbing only the current path would have missed it.
- Added `results/MANIFEST.md`, tracing all 27 published figures to source and data, and recording four reproducibility gaps.
- `pyproject.toml` still declares `authors = [{ name = "Anonymous" }]`, left over from blind review.

### Files modified this run
- `.gitignore`: publication boundary block; excludes `manuscript/`, `slides/`, `docs/`, `scripts/`, `templates/`, `LP_TEX_REF.md`, `PAPER_WRITING_SKILL.md`, `.vscode/`
- `README.md`: accepted-article framing, corrected layout, full notebook table, figure-reproduction section
- `HOUSEKEEPING.md`: this rewrite — toolchain table, lint policy, freshness checks, first baseline report
- `PUBLICATION_CHECKLIST.md`, `results/MANIFEST.md`: new
- 16 markdown files under `analytics/`: link repairs

### Follow-ups recorded in the planner store
- Set the real author name in `pyproject.toml` (currently `Anonymous`).
- Move `~/Desktop/RL_Signaling_backups/` to durable storage — it is now the only copy of the pre-scrub history.
- Add the article DOI and citation to `README.md`; check the journal's code-deposit policy before going public.

---

## Previous Report

**Date:** 2026-08-12
**Trigger:** Post-publication-prep verification run, requested after the repository was reduced to code-only. First run under the revised Phase 3, which now documents what the test suite covers relative to the paper's model.

### Artifact counts
- Tracked files: 135
- Source files (tracked `.py`): 35 — of which 7 are the `rl_signaling/` package
- Lines of code: 3,019 in the package; 9,664 across all tracked Python
- Registered tests: 63

### Static quality
- Format: 23 of 29 files would be reformatted — pre-existing, notebook-derived code. Deliberately not addressed (Phase 2 Step 4).
- Lint (`rl_signaling/`): **pass — all checks passed**
- Lint (repo-wide): 205 findings, informational. Top rules: E402 import-not-at-top (55), D103 undocumented-public-function (40), E501 line-too-long (24), F821 undefined-name (23), I001 unsorted-imports (17). All in `analytics/scripts/` and notebook-derived code.
- Type check: n/a — no type gate in this repository.

### Tests
- Unit: **63 passed / 0 failed / 0 skipped** (~17 s)
- Comparison vs. previous report: **steady at 63.**

### Repository health
- Dependencies: 33 packages behind latest. Of the 8 declared runtime dependencies, **none** are outdated; the drift is entirely in transitive/dev packages (`matplotlib-inline`, `tqdm`). No EOL pins, no advisories. Not actioned — the committed datasets were produced under the current pins, and bumping them risks perturbing the golden-run baseline for no scientific gain.
- Dead code: 12 findings (F401/F841), **none in the package** — `ruff check rl_signaling/ --select=F401,F841` passes.
- Docs: current. All markdown links across the 24 tracked documents resolve (**0 broken**). README's module list and notebook table match the package and `notebooks/` exactly.
- Publication boundary: **intact.** `git ls-files | grep -Ei 'manuscript|slides|reviewer|writing_comments|LP_TEX|PAPER_WRITING'` returns nothing.

### Notable events
- Phase 3 was rewritten this run. It previously described the suite only as `pytest tests/ -v`; it now documents, per file, which part of the paper's model each test covers — the three learning rules, the three information regimes, the NMI identities, the hand-derived analytical cases, and the golden-run regression — and states that a failure there threatens a published claim rather than merely a code path.
- A coverage gap is now recorded explicitly: the suite exercises the **mechanism**, not the paper's **findings**. Nothing asserts that signalling raises reward or NMI in the partial-information regime — that is a statistical claim over 10,000-run distributions, evidenced by the committed datasets and the figures traced in `results/MANIFEST.md`, not by pytest. This is a deliberate boundary, not a defect, but it should stay visible.
- Static-quality counts fell (format 41→23 files, lint 321→205) purely because the LaTeX revision toolkit and templates left the repository; no code was fixed.
- Local `.git` is 265 MB while a full clone from GitHub is 109 MB. The difference is pre-scrub objects held alive by reflog entries from verification fetches — local-only, no effect on the published repository. `git gc --prune=now --expire=now` would reclaim it, at the cost of destroying an incidental local copy of the pre-scrub history.

### Files modified this run
- `HOUSEKEEPING.md`: Phase 3 rewritten (suite-coverage table, integration/Restart-and-Run-All guidance, test-count-drop policy); prior report demoted; this report appended.

### Follow-ups recorded in the planner store
- `results/MANIFEST.md` names 32 of the 54 tracked figures literally; the other 22 are covered by grouped family rows in the exploratory table. All 27 **paper** figures are named individually, so traceability is complete — but a reader grepping for an exploratory filename will not find it. Low priority.
- Move `~/Desktop/RL_Signaling_backups/` to durable storage — still the only reliable copy of the pre-scrub history.
- Add the article DOI and citation to `README.md`; check the *Philosophy of Science* code-deposit policy before going public.

---

## Latest Report

**Date:** 2026-08-13
**Trigger:** User request — "make sure the models work well." Verification-focused run; scope was deliberately capped at the unit suite (notebook Restart-and-Run-All and the findings-level test were both offered and declined).

### Artifact counts
- Tracked files: **100** (down from 135 — see Notable events)
- Source files (tracked `.py`): **20** — of which 7 are the `rl_signaling/` package
- Lines of code: 3,019 in the package; **6,461** across all tracked Python (down from 9,664)
- Registered tests: **63 tracked** (80 collected — see Notable events)

### Static quality
- Format: 23 of 29 files would be reformatted — pre-existing, notebook-derived. Deliberately not addressed (Phase 2 Step 4). Unchanged from prior report.
- Lint (`rl_signaling/`): **pass — all checks passed**
- Lint (repo-wide): **205 findings**, informational — identical to the prior report. Top rules: E402 (55), D103 (40), E501 (24), F821 (23), I001 (17). The count did not fall when `analytics/` was untracked, because `ruff` walks the filesystem and `analytics/` is still on disk.
- Type check: n/a — no type gate in this repository.

### Tests
- Unit: **80 passed / 0 failed / 0 skipped** (9.1 s)
- Comparison vs. previous report: **steady on the tracked suite at 63.** The 17-test surplus is `tests/test_revision_tooling.py`, which is untracked-but-present on disk; see Notable events.

### Repository health
- Dependencies: 109 packages behind latest. None of the 8 declared runtime dependencies are outdated; the drift is transitive/dev. No EOL pins, no advisories. Not actioned — same reasoning as prior runs: the committed datasets were produced under the current pins, and bumping them risks perturbing the golden-run baseline for no scientific gain.
- Dead code: 12 findings (F401/F841), **none in the package** — `ruff check rl_signaling/ --select=F401,F841` passes.
- Docs: **current.** An earlier pass in this run flagged dangling `analytics/` links in `README.md` and `results/MANIFEST.md`; on re-verification against `HEAD` (`0cd1b7e`) they were already repaired. See Notable events.
- Publication boundary: **intact.** `git ls-files | grep -Ei 'manuscript|slides|reviewer|writing_comments|LP_TEX|PAPER_WRITING'` returns nothing.

### Notable events
- **The models are sound at the mechanism level.** Every test covering the paper's model passes: the three learning rules with all three selection strategies, the three information regimes, the NMI identities, the hand-derived analytical cases in `test_numerical_sanity.py`, and the seed-12345 golden regression in `test_golden.py`. Nothing has drifted since the baseline.
- **The findings-level gap recorded last run is still open and was not closed this run.** No test asserts that signalling raises reward or NMI under partial information. That remains evidenced by the committed datasets and the figures traced in `results/MANIFEST.md`, not by pytest. Writing such a test was offered and declined for this run; it stays a standing option, not a defect.
- Commit `11afe8a` ("Untrack analytics/ from the published repository") landed after the prior report and removed 36 files / 6,857 lines from tracking. This accounts for the entire 135 → 100 tracked-file and 9,664 → 6,461 LOC drop. The directory is intact on disk, and `0cd1b7e` then rewrote the `README.md` and `results/MANIFEST.md` references into unlinked prose. **Process note:** the Phase 4 docs check in this run read a working tree at `11afe8a` and reported the links as dangling; the follow-up filed against them was already satisfied by `0cd1b7e`. Verify the docs check against `git log --oneline -1` before filing link findings.
- `tests/test_revision_tooling.py` (17 tests) is gitignored but present on disk, so `pytest tests/` collects 80 against the tracked 63. This is **deliberate**, not leftover: `.gitignore` keeps the revision toolkit and its test file local as reusable tooling for a future paper. Phase 3 Step 3 now documents both numbers so future runs stop treating the gap as a finding.
- The planner MCP credential was expired at the start of the session (`cps_session_start` → `RefreshError`) and was re-authenticated mid-run via `uv run cps-ctrl login --no-browser` from `central_planner_storage`. Session `bf4aa1ed`, worklog `672ee7f6`.
- **The follow-ups claimed by both 2026-08-12 reports were never actually filed.** The `rl_signaling` repo held exactly two tasks in the planner store, both `done` and both from the July revision round. The carried-forward items below therefore exist only as prose in this file; they were reviewed this run and deliberately left unfiled.

### Files modified this run
- `HOUSEKEEPING.md`: prior report demoted; this report appended; `last_checked` bumped.

### Follow-ups recorded in the planner store
Both were filed and both were closed in the same run:
- `todo.docs.repoint-analytics-links` — **closed, already done.** Commit `0cd1b7e` had repaired the references before the task was filed; the finding was an artefact of checking a stale working tree.
- `todo.tests.resolve-untracked-revision-tooling` — **closed, resolved.** Not by deleting the file: `.gitignore` keeps it on disk deliberately. Phase 3 Step 3 now documents the expected 63-tracked / 80-collected split.

### Files not addressed
- `notebooks/poc_absorbing_states.py` is untracked and unexplained — it is neither gitignored nor committed. Left alone this run; decide whether it belongs in the repository or in the local-only `analytics/` tree.

### Follow-ups carried forward, not filed
These were listed as "recorded in the planner store" by the 2026-08-12 reports but never actually filed. They remain prose-only by explicit decision this run:
- `results/MANIFEST.md` names 32 of 54 tracked figures literally; all 27 paper figures are individually named, so traceability is complete. Low priority.
- Move `~/Desktop/RL_Signaling_backups/` to durable storage — still the only reliable copy of the pre-scrub history.
- Add the article DOI and citation to `README.md`; check the *Philosophy of Science* code-deposit policy before going public.
