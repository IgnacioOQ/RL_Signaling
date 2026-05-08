# Debugging Plan
- status: active
- type: plan
- id: rl_signaling.debugging_plan
- description: Phased audit of the rl_signaling/ implementation against the intended signaling model, designed for execution in a fresh session. Each phase has a clear deliverable that feeds the next.
- label: [planning, agent]
- injection: excluded
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->
This is a debugging plan for a future session. The objective is to compare the **intended system** (the signaling model described in the README and in the user's mental model) against the **current code implementation** under `rl_signaling/`, and surface any discrepancy.

The structure mirrors `REFACTOR_PLAN.md`: each phase has a scope, a verification step, and a deliverable. A discrepancy found in any phase should be recorded as a new entry in `LEGACY_BUGS_LOG.md` using the template at the bottom of that file.

## Operating rules for the agent picking this up

1. Read this file end-to-end before starting any phase.
2. Work on a `debugging` branch (or whichever the user has checked out). Do not commit unless explicitly instructed.
3. The first deliverable (Phase 1) is **the user's confirmation of the model description in this file**. Do not start the code audit until that handshake is done — every later phase compares against that ground truth.
4. When a discrepancy is found, do NOT fix it in the same session that found it. File it in `LEGACY_BUGS_LOG.md` with severity, location, and reproducibility. Fixes happen in a separate review pass after the audit is complete.
5. The verification harness (`pytest tests/`, `tests/golden/baseline.json`) is the safety net. Run it after any exploratory edit and before stopping.

## Decisions to lock in early

| Decision | Default | Locked? |
|---|---|---|
| Branch model | One long-lived `debugging` branch off `refactor` (or wherever the refactor lands). | No — confirm with user before starting Phase 2. |
| Where to record findings | `LEGACY_BUGS_LOG.md` per the template at its bottom. | Yes. |
| Whether to fix in the same pass | No — audit and fixes are separate sessions. | Yes — see Operating Rule 4. |
| Definition of "intended system" | The Model section of the README, plus any clarifications captured in Phase 1 of this plan. | Confirm in Phase 1. |
| Acceptable severity bar for "this is a bug, not a stylistic difference" | The implementation must match the model up to documented exceptions. Stylistic differences are not bugs. | Confirm in Phase 1. |

---

## Phase 0 — Session boot (Pending)

**Goal:** Bring the agent into the same context the prior session ended in, without redoing work.

### Steps

1. `git status` — confirm branch + uncommitted state.
2. `git log --oneline main..HEAD` — see what the refactor branch carries.
3. Read in order:
   - This file (`DEBUGGING_PLAN.md`).
   - `README.md` — especially the Model section.
   - `LEGACY_BUGS_LOG.md` — known bugs and their severity.
   - `REFACTOR_PLAN.md` — what the refactor changed and what's documented.
4. Run the smoke test:
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```
   Expected: 50 passed.
5. Confirm the user is ready for Phase 1.

### Deliverable

A one-paragraph "I'm ready" confirmation, naming the branch, the test count, and any open WORKLOG / LEGACY_BUGS_LOG entries that look load-bearing for the audit.

---

## Phase 1 — Ground truth: write down the intended model precisely (Pending)

**Goal:** Produce a precise, unambiguous specification of the signaling model the user *intends* to implement. Every later phase compares the code against this specification.

### Steps

1. Start from the README's Model section. Copy its bullets into a working buffer.
2. Walk through with the user. For each item, ask:
   - "Is this exactly what you intend, or is it a simplification?"
   - "Are there edge cases the README doesn't mention?"
   - "Does it match how you would describe the model in a paper draft?"
3. Cover at minimum the following axes (the user may add more):

   **State and observations**
   - Distribution of `nature_vector` (uniform binary? other?).
   - How `observed_variables` is interpreted (per-agent index lists? overlap allowed?).
   - Are observations always tuples of integers?

   **Signals**
   - Signal alphabet size; whether it can vary per agent.
   - When `costly_signaling=True`, is the null signal at the highest index by construction?
   - Should the null signal be receivable (i.e. appended to the neighbor's observation) or always suppressed?
   - Is the signaling step *simultaneous* (all agents pick before any are sent) or *sequential*?

   **Graph and message passing**
   - Directionality semantics: does an edge `(u, v)` mean "u sends to v" or "v listens to u"? Verify against `graph.predecessors(i)` usage in code.
   - Self-loops permitted?
   - Multi-edges (parallel edges) — undefined behavior?

   **Actions and payoffs**
   - Is the game dict's state key the *full* nature vector or the agent's observation?
   - When `full_information=True`, do agents share the same observation key, and does this affect Q-table key collisions?
   - Are rewards integers, floats, or mixed? Is the implicit reward floor zero?

   **Costly signaling**
   - Is `signal_cost` always per-agent, or could it be per-state?
   - Does the cost apply on the per-episode reward only, or also propagate into Q-bootstrap targets?

   **Information regimes**
   - Three regimes: full-info, partial-info-no-signals, partial-info-with-signals. The fourth combination (full-info + no signals) appears in some figures — is it considered the trivial baseline?

   **Agent learning rules**
   - UrnAgent: confirm Roth–Erev with reward-as-positive-reinforcement, clamped at zero. Does the user expect `reward = 0` to leave the urn unchanged (current behavior), or to add a small positive count to keep all actions live?
   - QLearningAgent: confirm `α = 0.1` constant; confirm `td_target = reward` (no bootstrap, episodes are single-step); confirm exploration-rate decay applied per-channel.
   - TDLearningAgent: confirm bootstrap from `next_state` Q-values; confirm count-based learning rate `1 / N(s,a)`; confirm shared Q-table across signal and action phases.

   **NMI**
   - Confirm formula: `NMI = I(S; O) / H(O)`. Some references use `I / sqrt(H(S) H(O))` (geometric NMI) — which is intended?
   - Confirm `H(O) == 0 → NMI := 0` is the user's preferred convention (alternatives: NaN, raise).

4. Write the result into a new section at the bottom of this file titled `## Phase 1 — Confirmed model specification`. Each item from the buffer above appears as either "Confirmed: …" or "Clarified: …" with the user's verbatim answer.

### Deliverable

A `## Phase 1 — Confirmed model specification` section in this file containing every item the user confirmed, with explicit yes/no answers and any clarifications. **Do not proceed past Phase 1 without this section in place.**

---

## Phase 2 — Module-level audit against the spec (Pending)

**Goal:** Walk through `rl_signaling/` module-by-module and check every behavior against the Phase 1 spec. One bug entry per discrepancy.

### Order

Audit in dependency order so you've already verified anything a module depends on before opening it:

1. [rl_signaling/games.py](rl_signaling/games.py) — random and canonical game generators; signal-urn initializers.
2. [rl_signaling/info_theory.py](rl_signaling/info_theory.py) — `_compute_entropy`, `compute_mutual_information`.
3. [rl_signaling/agents.py](rl_signaling/agents.py) — `BaseAgent`, `_select_action`, `UrnAgent`, `QLearningAgent`, `TDLearningAgent`.
4. [rl_signaling/env.py](rl_signaling/env.py) — `MultiAgentEnv` (canonical), `NetMultiAgentEnv` and `TempNetMultiAgentEnv` (deprecated).
5. [rl_signaling/simulation.py](rl_signaling/simulation.py) — `run_simulation` (canonical), `simulation_function` and `temp_simulation_function` (deprecated).
6. [rl_signaling/plotting.py](rl_signaling/plotting.py) — only checked for *correctness of the metric being plotted*, not for visual styling.

### Per-module checklist

For each module:

1. **Read the module top to bottom**, ignoring nothing. Treat docstrings and parameter defaults as claims that must be backed by code.
2. **Map every function / class to a corresponding bullet in the Phase 1 spec.** Functions that don't map to anything in the spec are either dead code (file in `LEGACY_BUGS_LOG.md`) or undocumented behavior (file under "Clarification needed").
3. **For every function:**
   - Is the math identical to the spec? Pay special attention to off-by-one indices, sign conventions, and reward direction (is reward added to Q or subtracted?).
   - Are the inputs validated, or does it rely on caller discipline? Both are fine — but note which.
   - Are the side effects explicit (e.g. counter increments inside `get_action`) or are they easy to miss?
4. **For every state-keyed dict** (`signaling_urns`, `q_table_signaling`, `q_table_action`, `q_table`, `signal_usage`, `action_usage`):
   - When is the key first created?
   - What is the initialization value?
   - Are there code paths that read the key without writing it first?
   - Could a key from one phase collide with a key from another phase?

### Specific things to look for

These are concrete possible bugs informed by what we already know about the codebase:

| Check | What to verify |
|---|---|
| **`UrnAgent.update_signals` / `update_actions` reward semantics** | The current code is `urn[s] = max(0, urn[s] + reward)`. With negative rewards (e.g. costly signaling making the net reward negative) this clamps; with very small positive rewards it adds. Is the clamping intended? Does the user mean to use the un-clamped Roth–Erev variant? |
| **`QLearningAgent` constant α = 0.1** | Hard-coded, not a constructor parameter. Was that the user's choice or an oversight? |
| **`QLearningAgent` "no bootstrap" assumption** | The code uses `td_target = reward` (no `+ γ max Q(s')`). Single-step episodes make this correct, but it's worth confirming the user agrees the canonical signaling game is single-step. |
| **`TDLearningAgent.update` next-state initialization** | `update` initializes `q_table[next_state]` as `np.zeros(n_actions)` if missing. In the deprecated path this happens *between* the signal-phase update and the action-phase get_action — verify it isn't masking a real "first-time-seen" event. |
| **`MultiAgentEnv._send_signals` null-signal suppression** | When `costly_signaling=True`, the null signal (highest index) is *not* appended to the receiver's observation. Is that correct, or should the null signal still be visible to receivers as a meaningful "I sent nothing" signal? |
| **`MultiAgentEnv.reward` cost application** | Currently the cost is deducted *after* the game-dict lookup, so the recorded reward is `game_reward - cost`. Verify this is what the user wants stored in `rewards_history` (vs storing the gross reward and tracking cost separately). |
| **`compute_mutual_information` H(O) == 0 branch** | Returns `NMI = 0`. Verify this matches the user's convention vs. NaN or undefined. |
| **`_select_action` ucb when counts are all zero** | Adds `1e-5` to avoid division by zero. Verify the resulting bonus magnitude matches the intended UCB1 / UCB-Tuned variant the user has in mind. |
| **Game-dict state key — full vs partial state** | The env looks up `game_dicts[i][tuple(self.nature_vector)][action]`. The state key is the full nature vector regardless of full vs partial information. Confirm. |
| **Signal selection vs action selection RNG ordering** | Within an episode, agents are iterated in agent-index order. Two agents share the global RNG. Does this match the user's expectation, or should each agent have an independent RNG? |
| **`agents_observed_variables` overlap** | Two agents could observe overlapping feature subsets (e.g. `{0:[0,1], 1:[1,2]}`). Confirm this is allowed and produces sensible behavior. |
| **`full_information=True` Q-table keys** | When all agents share the same observation, the Q-table keys collide across agents. Each agent has its own Q-table, so no actual collision — but worth verifying. |

### Deliverable

For each bug found:

1. A new `## Bug N — …` section appended to `LEGACY_BUGS_LOG.md`, following the template at the bottom of that file.
2. A one-line entry in this file's `## Phase 2 — Findings` section, linking to the LEGACY_BUGS_LOG entry.

If the audit finds zero discrepancies in a module, write "No discrepancies in `<module>` against Phase 1 spec." in the Findings section.

---

## Phase 3 — Notebook-level audit (Pending)

**Goal:** Check that the **experiment notebooks** encode what they claim. The module audit verifies the building blocks; this phase verifies how the blocks are assembled.

### Per-notebook checklist

For each of the six notebooks under `notebooks/`:

1. **What does the notebook claim to test?** Read its markdown headings and the user-facing description (if any). Write the claim in one sentence.
2. **What does the notebook actually do?** For each code cell, describe in one sentence what it constructs and runs.
3. **Are the three information regimes set up identically?** When a notebook runs all three regimes (full / partial-no-signals / partial-with-signals), it should:
   - Use the **same game dicts** across regimes.
   - Use the **same nature-vector seed** so the comparison is paired.
   - Differ only in the `full_information` and `with_signals` flags.
4. **Are the per-experiment seeds reset where the user expects?** A common pattern is to reset the seed at the top of each iteration so repeats are reproducible; verify or flag.
5. **Are the saved CSV column names consistent** with what `plotting_results.ipynb` reads downstream?
6. **Are the plots reading the columns they claim to read?** Mistakes here are silent — the figure renders, just from the wrong column.

### Per-notebook deliverable

Write a one-paragraph audit summary in this file's `## Phase 3 — Findings` section:

```
### notebooks/<name>.ipynb
- **Claim:** {{one sentence}}
- **Setup correctness:** {{OK | discrepancy + LEGACY_BUGS_LOG link}}
- **Regime symmetry:** {{OK | discrepancy + LEGACY_BUGS_LOG link}}
- **Output → plot consistency:** {{OK | discrepancy + LEGACY_BUGS_LOG link}}
- **Notes:** {{anything that doesn't fit the above but is worth recording}}
```

---

## Phase 4 — Numerical sanity (Pending)

**Goal:** Run hand-computed mini-cases to confirm the math. This is the strongest check: numerical agreement to many decimals against an analytically-known answer.

### Cases to construct

1. **NMI on a known distribution.** A 2×2 signal-usage table with rows `[10, 0]` and `[0, 10]` should give `NMI = 1`. Already covered by `tests/test_info_theory.py` — re-derive on paper, then re-derive against `_compute_entropy` to verify the log base.
2. **Single-step Q-update with α=0.1 and reward=1.** From `Q[s][a] = 0`, after one update should be `0.1`. After ten identical updates should converge geometrically; on paper compute the value at episode 10 and compare.
3. **TD update with γ=1, reward=0, next_state Q max = 1.** From `Q[s][a] = 0`, td_target = 0 + 1·1 = 1, td_error = 1 - 0 = 1, learning rate = 1/N(s,a) = 1. After one update Q should be `1`.
4. **Costly signaling with cost=0.25 and game reward=1.** Confirm rewards_history entry is `0.75` exactly. Then with `null_signal=True` confirm it's `1.0` (no cost).
5. **Two-agent canonical game, full information, no signals.** With `np.random.seed(0)` and a hand-constructed game dict where the optimal action for state `(0,0)` is action 2, after 1000 episodes both agents should pick action 2 with probability ≥ 0.95 (under any agent type with `exploration_rate → min`).

### Deliverable

A small script `tests/numerical_sanity.py` (or appended cases to `tests/test_smoke.py`) implementing each case, with a comment explaining the analytical answer. Each case asserts the implementation matches the analytical answer to a documented tolerance (exact for finite-step cases, asymptotic for the convergence case).

---

## Phase 5 — Synthesis and reporting (Pending)

**Goal:** Convert the per-phase findings into a coherent picture: how big is the bug surface, which experiments need to be re-run, what's the cost of fixing each.

### Steps

1. List every bug filed in `LEGACY_BUGS_LOG.md` during this debugging session.
2. For each, attach:
   - Severity (already in the LEGACY_BUGS_LOG entry).
   - Affected notebooks and saved figures.
   - Estimated fix effort: **trivial** (< 30 lines), **medium** (< 200 lines, contained), **large** (cross-cutting, > 200 lines).
   - Whether the fix changes any saved-result fingerprints — and if so, which ones need to be regenerated.
3. Group fixes into batches:
   - **Hot path:** bugs whose fix actively breaks current code (none expected, but possible).
   - **Result-affecting:** bugs whose fix changes one or more saved figures. List the figures.
   - **Latent:** bugs that only fire on error paths or unused code; deferred indefinitely.
4. Propose a fix order in this file's `## Phase 5 — Fix plan` section. The order should respect dependencies (don't fix downstream before upstream when both are buggy).

### Deliverable

A `## Phase 5 — Fix plan` section in this file ranking each bug by severity × impact × effort, with an explicit "fix in this batch / defer" decision per bug.

---

## Phase 6 — Verification re-run (Optional, only after fixes land in a separate session)

**Goal:** After bugs are fixed in a follow-up session, re-run the verification harness and the affected experiment notebooks; check that the new numbers match the predicted impact in the LEGACY_BUGS_LOG.

This phase is not part of the audit itself — it is the closing validation. It exists in this plan so a future agent reading the plan knows where the loop ends.

### Steps

1. `pytest tests/` → expected: 50+ passed (more if numerical sanity tests were added in Phase 4).
2. Run each affected notebook end-to-end.
3. Diff each regenerated figure / CSV against the archived pre-fix version.
4. For each LEGACY_BUGS_LOG entry, append a "Post-fix observation" subsection recording the actual measured impact and whether it matched the prediction.

---

## Resume-here checklist for a new session

```
[ ] Read this file end-to-end.
[ ] Read the README Model section, REFACTOR_PLAN.md, LEGACY_BUGS_LOG.md.
[ ] Confirm branch + run pytest (expect 50 passing).
[ ] Phase 1: walk the model with the user, fill in `## Phase 1 — Confirmed model specification`.
[ ] Phase 2: module-by-module audit; one LEGACY_BUGS_LOG entry per discrepancy.
[ ] Phase 3: per-notebook audit; record findings in `## Phase 3 — Findings`.
[ ] Phase 4: write numerical sanity cases under tests/.
[ ] Phase 5: synthesize fix plan in `## Phase 5 — Fix plan`.
[ ] Hand off to a separate session for fixes (Phase 6 closes the loop).
```

---

## Phase status

| Phase | Status |
|---|---|
| 0. Session boot | Pending |
| 1. Ground truth (model spec) | Pending |
| 2. Module-level audit | Pending |
| 3. Notebook-level audit | Pending |
| 4. Numerical sanity | Pending |
| 5. Synthesis + fix plan | Pending |
| 6. Verification re-run | Deferred (separate session) |

---

## Phase 1 — Confirmed model specification

*(To be filled in during Phase 1. Each bullet should read "Confirmed: …" or "Clarified: …" with the user's verbatim answer.)*

## Phase 2 — Findings

*(To be filled in during Phase 2. One line per module, linking to LEGACY_BUGS_LOG.md entries.)*

## Phase 3 — Findings

*(To be filled in during Phase 3. One block per notebook.)*

## Phase 5 — Fix plan

*(To be filled in during Phase 5. Ranked, batched, with per-bug fix/defer decisions.)*
