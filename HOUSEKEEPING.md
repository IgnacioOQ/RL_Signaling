# RL_Signaling Housekeeping
- status: active
- type: workflow
- id: rl_signaling.housekeeping
- description: Recurring sanity check for the RL_Signaling repository — runs static quality checks, unit tests, and repo-level health checks; appends an audit report.
- label: [core, agent]
- injection: excluded
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
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

The canonical commands for this repo (post-Phase-1 of `REFACTOR_PLAN.md`):

| Concern | Command |
|---|---|
| Format check | `ruff format --check .` |
| Lint | `ruff check .` |
| Type check | `mypy rl_signaling/` *(applies once Phase 3.5 lands type hints)* |
| Unit tests | `pytest tests/` *(applies once Phase 6 builds the suite)* |
| Build smoke | n/a — this is a research codebase, not a build artifact |
| Dependency audit | `pip list --outdated` (or `pip-audit` if installed) |

If a command above is marked *applies once Phase X lands*, record `n/a` in the report until that phase is done.

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
ruff check .
```

### Step 3 — Type check

```bash
mypy rl_signaling/
# n/a until Phase 3.5 of REFACTOR_PLAN.md adds type hints across the package.
```

### Step 4 — Remediation

- **Format errors:** run `ruff format .` (without `--check`) and re-verify.
- **Lint errors:** fix in source. Do not silence with `# noqa` unless the disable is documented and justified in the same change.
- **Type errors:** fix in source. Do not widen with `Any` to bypass.

**Exit criterion:** All static checks return zero errors. Pre-existing warnings are flat or trending down vs. the previous report.

---

## Phase 3 — Tests

**Goal:** Verify behavior is unbroken and the test suite has not silently shrunk.

### Step 1 — Unit tests

```bash
pytest tests/ -v
# n/a until Phase 6 of REFACTOR_PLAN.md builds the test suite.
```

### Step 2 — Integration / end-to-end tests

n/a — this repo's only integration surface is the notebooks. Spot-check by running the first 5 cells of `notebooks/basic_unit_test.ipynb` after any change to `rl_signaling/agents.py`, `env.py`, or `simulation.py`.

### Step 3 — Test count comparison

Compare pass / fail / skipped counts against the prior "Latest Report." Any regression is a finding to investigate before closing the run.

**Exit criterion:** All tests pass. Test count is steady or higher than the prior report.

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

- Does `README.md` still describe the actual entrypoints and module names? (Especially after Phase 3 of `REFACTOR_PLAN.md` migrates modules into `rl_signaling/`.)
- Is `REFACTOR_PLAN.md`'s "Phase status" table accurate?
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
