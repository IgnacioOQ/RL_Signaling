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

## Continue Repository Refactor
- status: todo
- type: task
- id: todo.refactor_continue
- description: Resume the multi-phase refactor described in REFACTOR_PLAN.md from the next pending phase.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-08
<!-- content -->
**Context:** The `refactor` branch carries Phase 0 (audit) and Phase 1 (hygiene) of a multi-session refactor. The full plan, decisions, per-phase scope, and verification steps live in `REFACTOR_PLAN.md` at the repository root. Phases 2 through 7 are pending. This task is a placeholder that points an incoming agent at that plan.

**Preconditions:**
- `git status` shows the working tree on the `refactor` branch. If not, run `git checkout refactor`.
- `REFACTOR_PLAN.md` exists at repo root and the "Phase status" table is current.

**Steps:**
1. Read `REFACTOR_PLAN.md` end-to-end (it is self-contained).
2. Run the Phase 1 verification smoke test to confirm the baseline:
   ```bash
   python3 -c "
   import sys, types
   sys.modules.setdefault('seaborn', types.ModuleType('seaborn'))
   import utils, agents, environment, simulation_function
   print('Modules import cleanly.')
   "
   ```
3. Identify the next pending phase from the "Phase status" table.
4. Execute that phase following its Scope / Strategy / Verification sections.
5. Update the phase's status (`Pending` → `Done`) and fill its "Notes from execution" section in `REFACTOR_PLAN.md` before stopping.
6. Append a `WORKLOG.md` entry summarizing the phase if it involved non-trivial work (any phase from 2 onward qualifies).

**Verification:**
- The phase's verification command (defined in its `REFACTOR_PLAN.md` section) succeeds.
- `REFACTOR_PLAN.md` reflects the new phase status.
- `WORKLOG.md` has a new dated entry for the work just done.

**On completion:** Do **not** delete this task block — keep it until every phase in `REFACTOR_PLAN.md` is marked `Done`. At that point delete the block (from the `---` above the `##` header to the `---` below the last line) and add a final WORKLOG entry recording the refactor's completion.

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
