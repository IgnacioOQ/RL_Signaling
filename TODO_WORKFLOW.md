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
- status: todo
- type: task
- id: todo.debugging_audit
- description: Execute the phased audit in DEBUGGING_PLAN.md to compare the rl_signaling/ implementation against the intended signaling model.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-08
<!-- content -->
**Context:** `DEBUGGING_PLAN.md` at the repo root carries a six-phase plan for auditing the implementation against the user's intended signaling model. Phase 1 is a model-specification handshake with the user; Phases 2–4 are the audit itself; Phase 5 produces a ranked fix plan; Phase 6 (separate session) verifies fixes after they land. Discrepancies surfaced during the audit are filed in `LEGACY_BUGS_LOG.md`, which already carries three entries from the refactor.

**Preconditions:**
- `git status` shows the working tree on the `refactor` branch (or the user's named debugging branch). If not, ask before proceeding.
- The `.venv/` exists and `pytest tests/` reports 50 passed.
- `DEBUGGING_PLAN.md` exists at the repo root with the "Phase status" table and the empty placeholder sections at the bottom.

**Steps:**
1. Read `DEBUGGING_PLAN.md` end-to-end, plus the README Model section, `LEGACY_BUGS_LOG.md`, and `REFACTOR_PLAN.md`.
2. Run the smoke test:
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```
   Expected: 50 passed.
3. **Phase 1 first.** Walk the user through every bullet in the Phase 1 checklist. Record each answer verbatim in the `## Phase 1 — Confirmed model specification` section at the bottom of `DEBUGGING_PLAN.md`. Do not start Phase 2 until that section is filled in.
4. Execute Phases 2 → 3 → 4 → 5 in order. Each phase has its own deliverable section in `DEBUGGING_PLAN.md`.
5. For every discrepancy found, append a new entry to `LEGACY_BUGS_LOG.md` using the template at its bottom. Do **not** fix bugs in the same session that finds them — fixes are deferred to a follow-up session per the plan's Operating Rule 4.

**Verification:**
- `## Phase 1 — Confirmed model specification` in `DEBUGGING_PLAN.md` is populated.
- `## Phase 2 — Findings`, `## Phase 3 — Findings`, and `## Phase 5 — Fix plan` in `DEBUGGING_PLAN.md` are populated.
- Every new bug surfaced has a corresponding `## Bug N — …` block in `LEGACY_BUGS_LOG.md`.
- `pytest tests/` still passes.

**On completion:** Do **not** delete this task block — keep it until Phase 6 (verification re-run) closes. At that point, delete the block (from the `---` above the `##` header to the `---` below the last line) and add a final WORKLOG entry recording the audit's completion.

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
