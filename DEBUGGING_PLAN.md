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
| 0. Session boot | Done (2026-05-08) |
| 1. Ground truth (model spec) | Done (2026-05-08) |
| 2. Module-level audit | Done (2026-05-08) |
| 3. Notebook-level audit | Done (2026-05-08) |
| 4. Numerical sanity | Done (2026-05-08) |
| 5. Synthesis + fix plan | Done (2026-05-08) |
| 6. Verification re-run | Deferred (separate session) |

---

## Phase 1 — Confirmed model specification

Locked in by the user during the Phase 1 handshake on 2026-05-08. Every later phase compares the code against this section. Items read either **Confirmed:** (matches what the user intends and what the code currently does) or **Clarified:** (the user's verbatim refinement of an axis the README left ambiguous).

### Operating decisions for the audit

- **Branch:** `debugging` branch off `refactor`. To be created at the top of Phase 2 before any LEGACY_BUGS_LOG / DEBUGGING_PLAN edits land.
- **Severity bar:** Implementation must match the spec; stylistic differences are not bugs. Discrepancies are filed only when behavior diverges from the items below.

### State and observations

- **Confirmed:** `nature_vector` is drawn i.i.d. uniform binary each episode — every feature is a fair coin, independent of the others. Equivalent to uniform over the 2^`n_features` states.
- **Confirmed:** `agents_observed_variables` is a per-agent list of feature indices; overlapping subsets are allowed; an agent's observation is a tuple of the corresponding bits in the order the indices appear in the list.
- **Confirmed:** observations are always `tuple[int, ...]`, ordered by feature-index position in `agents_observed_variables[i]`, used directly as dict keys for urns / Q-tables.

### Signals

- **Confirmed:** `n_signaling_actions` is a single global parameter; the alphabet does not vary per agent.
- **Confirmed:** when `costly_signaling=True`, the null signal sits at index `n_signaling_actions - 1` (highest index); senders pay no cost when they emit null; receivers do **not** see null in their observation — silence is silent. Receivers' observation length therefore varies with how many of their in-neighbours emitted null.
- **Confirmed:** the signaling step is **simultaneous** — every agent picks its signal from its pre-signal observation, then signals are delivered to neighbours. No agent's signal can influence another agent's signal in the same episode.

### Graph and message passing

- **Confirmed:** edge `(u, v)` means "u sends to v". Receiver `i` reads from `graph.predecessors(i)` to enumerate its senders.
- **Clarified:** self-loops are permitted by the env but are expected to be absent in practice; if a self-loop is present, the agent receives its own signal as part of its observation. The env does not filter or warn — caller discipline.
- **Clarified:** parallel/multi-edges are undefined behaviour. Use `nx.DiGraph` only; do **not** pass `nx.MultiDiGraph`. Code assumes one edge per `(u, v)`.

### Actions and payoffs

- **Confirmed:** the game-dict state key is the **full** `nature_vector` (as a tuple), regardless of `full_information`. Lookup is `game_dicts[i][tuple(nature_vector)][action]`. Payoff depending on the full state is what makes signaling necessary.
- **Confirmed:** when `full_information=True`, every agent observes the full nature vector. Each agent has its own urn / Q-table, so there is no key collision across agents even though they share the same observation key.
- **Confirmed:** rewards are floats (or ints; mixing is allowed in game dicts). There is **no** implicit non-negativity floor on the env-returned reward — costly signaling can drive net rewards negative.

### Costly signaling

- **Confirmed:** `signal_cost` is a per-agent scalar; same cost regardless of state or signal identity. Per-signal or per-state costs are not in scope.
- **Confirmed:** the cost is subtracted from the per-episode reward, so `rewards_history` records `game_reward - cost`. There is no separate gross-vs-net reward channel; for `TDLearningAgent`, the cost is therefore already inside the reward used for the signal-phase TD-bootstrap target.
- **Confirmed:** sending the null signal is free; only non-null signals incur the cost.

### Information regimes

- **Confirmed:** three primary regimes — full-information, partial-information-no-signals, partial-information-with-signals. The fourth combination (full-information + no-signals) is the trivial baseline that some figures show for completeness.
- **Confirmed:** when `full_information=True` AND `with_signals=True`, signals are **not** suppressed — agents go through the signaling step normally, and the signals are expected to be redundant (NMI low) but mechanically present.

### Agent learning rules

- **Confirmed (UrnAgent):** Roth–Erev with positive-reinforcement clamp. Update is `urn[s][a] = max(0, urn[s][a] + reward)`. Negative rewards (e.g. costly-signaling net-negative) clamp at zero; zero rewards leave the urn unchanged. The probabilistic interpretation of an urn is preserved (no negative weights).
- **Confirmed (QLearningAgent):** `α = 0.1` constant (hardcoded — accepted, not flagged as a bug); `td_target = reward` (no bootstrap; canonical signaling game is single-step, so `γ · max Q(s')` reduces to zero by terminality); exploration-rate decay is applied **per channel** — `get_signal` decays the signal-phase rate, `get_action` decays the action-phase rate, separately.
- **Confirmed (TDLearningAgent):** bootstrap from `next_state` Q-values (`γ · max`); count-based learning rate `1 / N(s, a)`; **shared** Q-table across signal and action phases. Phases are distinguishable by tuple length — signal-phase observations are the agent's nature observation only, action-phase observations have received signals appended — so signal-phase keys and action-phase keys do not collide in the shared dict.

### NMI and exploration details

- **Confirmed:** `NMI = I(S; O) / H(O)` — asymmetric, output-side normalization. Geometric NMI (`I / √(H(S)·H(O))`) is **not** the intended formula.
- **Confirmed:** when `H(O) == 0` (constant signal), `NMI := 0` by convention. Not NaN, not an exception.
- **Confirmed:** UCB tie-break when all action counts are zero adds a tiny epsilon (`1e-5`) to the count denominator to avoid division by zero. The first-pass-mandatory-exploration variant of UCB1 (visit each action exactly once before applying the UCB formula) is **not** the intended variant.
- **Confirmed:** all agents share the global `np.random` / `random` state. Iteration order within an episode is agent-index order — agent 0 always draws from the RNG before agent 1. Per-agent independent RNGs are not in scope.

---

## Phase 2 — Findings

Audit completed 2026-05-08 on the `debugging` branch (cut from `refactor`). Each module was read top-to-bottom and every public function mapped against the Phase 1 confirmed spec. Severity bar: implementation must match the spec; stylistic differences are not bugs.

### [rl_signaling/games.py](rl_signaling/games.py)

No discrepancies against Phase 1 spec. `create_random_game`, `create_random_canonical_game`, and `create_initial_signals` all use `product([0, 1], repeat=n)` to enumerate states — consistent with Axis 1 (i.i.d. uniform binary) and Axis 10 (full nature_vector as game-dict key). The canonical-game generator's `assert len(world_states) <= len(unique_dicts)` correctly guarantees a unique optimal action per state when `n_final_actions >= 2**n_features`. Internal helpers `_generate_unique_dicts` / `_generate_hot_vectors` operate as documented.

### [rl_signaling/info_theory.py](rl_signaling/info_theory.py)

No discrepancies against Phase 1 spec. `compute_mutual_information` divides by `H(O)` (Axis 21) and returns `NMI=0` when `H_O <= 0` (Axis 22, line 59). `_compute_entropy` correctly skips `p=0` terms via the `if p > 0` filter. One latent robustness gap: if `agent_signal_usage` contains an observation whose count row sums to zero (`{obs: [0, 0, ...]}`), line 52's per-observation normalization `count / sum(counts)` would raise `ZeroDivisionError`. Skipped per severity bar — not reachable in normal env operation because `signal_usage[i][obs]` is only created lazily on first emission, so the sum is always ≥ 1 once the key exists.

### [rl_signaling/agents.py](rl_signaling/agents.py)

**One discrepancy filed: Bug 4** — `QLearningAgent.__init__` only pre-seeds `q_table_signaling` when `initialize=True`; `q_table_action` is silently set to `{}` on every construction. Structurally identical to the pre-fix shape of Bug 1 (`UrnAgent.action_urns`), with the difference that the docstring at [rl_signaling/agents.py:354-358](rl_signaling/agents.py#L354-L358) explicitly states "pre-seed the **signaling** Q-table" — so this may be intentional. Filed in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md) as Bug 4 with a fix proposal contingent on user clarification. Affects only [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb).

Other axes verified clean:
- `_select_action` UCB epsilon is `1e-5` on counts plus a `+1` on `total_counts` (Axis 23 mechanism confirmed; the `+1` is an additional safeguard not in the spec narrative but does not change the documented "tiny epsilon" behavior).
- `UrnAgent.update_signals` / `update_actions` use `max(0, urn + reward)` (Axis 18).
- `QLearningAgent` constant `α = 0.1` (line 458, line 476), `td_target = reward` (no bootstrap), per-channel exploration decay (Axis 19).
- `TDLearningAgent` shared single Q-table, count-based `1/N(s,a)` learning rate, bootstrap from `next_state` with `γ=1` default (Axis 20). The `update_episode` signal-phase update uses `reward=0` and bootstraps from `action_state`; under `γ=1` this is mathematically equivalent to attributing `-cost` to the signal phase, so it is consistent with Axis 14 (cost subtracted from per-episode reward) without violating Axis 20's "bootstrap from next_state".
- All RNG calls go through Python's global `random` module or NumPy's global `np.random` state (Axis 24).

### [rl_signaling/env.py](rl_signaling/env.py)

**MultiAgentEnv (canonical):** No discrepancies. `reset` draws `nature_vector` via `np.random.randint(0, 2, size=n_features)` (Axis 1). Observation construction follows `agents_observed_variables[i]` index order (Axes 2–3). `step_signal` computes signals from frozen pre-signal observations and only then propagates them via `_send_signals` (Axis 6 — simultaneous). `_send_signals` iterates `graph.predecessors(i)` (Axis 7) and skips appending when `costly_signaling and signals[neig] == _null_signal_index` (Axis 5 — silence is silent). `reward` keys lookups by `tuple(self.nature_vector)` (Axis 10) and deducts cost only for non-null signals (Axes 14, 15). No multi-edge or self-loop guards (Axes 8–9 confirmed canonical).

**NetMultiAgentEnv (deprecated):** No spec discrepancies in the env itself; cost handling is delegated to `simulation_function`, which is consistent with the documented architecture.

**TempNetMultiAgentEnv (deprecated):** Bug 2 (already in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md)) remains open. The original "root cause" framing attributed the bug to variable shadowing in the inner NMI loop. The variable was renamed `i → j` during the refactor (no shadowing now), but the inner loop is still **nested inside** the outer per-agent loop at [rl_signaling/env.py:733-760](rl_signaling/env.py#L733-L760), so the symptom (each agent's `signal_information_history` getting `n_agents` writes per signal phase) persists. The Bug 2 entry has been updated in this session to reflect the corrected framing — the shadowing was incidental; the nested-loop structure is the actual cause.

### [rl_signaling/simulation.py](rl_signaling/simulation.py)

No discrepancies against Phase 1 spec.

- `run_simulation` (canonical) drives `MultiAgentEnv` through the `reset → step_signal → step_action → reward → update` cycle. The `with_signals=False` branch sets `signals=None` and `new_observations = copy.deepcopy(observations)`, which TDLearningAgent's `update_episode` handles correctly via the `if signal is not None` guard. One robustness gap: when `env.costly_signaling=True` and `with_signals=True` but the caller forgets to pass `signal_cost`, the env silently skips the cost (it short-circuits on `signal_cost is not None`). Skipped per severity bar — caller error rather than spec mismatch — but worth flagging as a Phase 5 candidate for a `ValueError` guard.
- `simulation_function` (deprecated) duplicates the cost-deduction logic from `MultiAgentEnv.reward`. Architectural concern only; the path is deprecated.
- `temp_simulation_function` (deprecated) does not support costly signaling — documented limitation, not used by the costly-signaling notebook.
- The legacy `temp_simulation_function`'s exploration-decay-before-action-phase ordering is the documented divergence already covered in [README.md](README.md) (Status section) and [REFACTOR_PLAN.md](REFACTOR_PLAN.md) Phase 6 notes. Not a new finding.

### [rl_signaling/plotting.py](rl_signaling/plotting.py)

No discrepancies against Phase 1 spec. Plotting functions consume metrics opaquely from upstream modules — `signal_usage`, `rewards_history`, `signal_information_history`, `histories` — without any metric-altering transformation. `calculate_proportions` returns `urn[state][0] / sum(urn[state])` (slot-0 share only); this is a deliberate diagnostic for binary signaling experiments and matches the docstring. `count_negative_nmi` is a purpose-built diagnostic that counts floating-point underflow in NMI columns; a non-zero return indicates `H_S < H_S_given_O` due to rounding, which is a known information-theoretic edge case rather than a bug. The `import sys` fix from Bug 3 is in place at line 11.

### Summary

| Module | Status |
|---|---|
| `games.py` | Clean against Phase 1 spec. |
| `info_theory.py` | Clean against Phase 1 spec. |
| `agents.py` | One open finding: **Bug 4** (filed). |
| `env.py` (canonical) | Clean against Phase 1 spec. |
| `env.py` (deprecated `TempNetMultiAgentEnv`) | **Bug 2** still open (existing entry; root-cause framing corrected this session). |
| `simulation.py` | Clean against Phase 1 spec. |
| `plotting.py` | Clean against Phase 1 spec. |

Net new bugs filed in this phase: 1 (Bug 4). Existing bug entries updated: 1 (Bug 2 root-cause clarification). Spec items verified hands-on: every axis 1–24.

## Phase 3 — Findings

Audit completed 2026-05-08. Six notebooks reviewed against (a) the Phase 1 confirmed model specification, (b) the plan's per-notebook checklist (claim vs actual, regime symmetry, seed reset, CSV ↔ plot consistency), and (c) the structural conventions in the KB's [content/how-to/NOTEBOOK_WRITING_SKILL.md](kb://content/how-to/NOTEBOOK_WRITING_SKILL.md) — restart-and-run-all viability, parameter-cell hygiene, multiprocessing seed correctness, stable cell IDs.

Net new bug entries filed: **Bugs 5, 6, 7, 8** in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md). One existing entry updated: **Bug 2** (the legacy `TempNetMultiAgentEnv` history inflation now confirmed to affect TD-learning saved CSV summary statistics — first 10 / last 100 slices land on first 5 / last 50 episodes after the 2× inflation).

### notebooks/basic_unit_test.ipynb

- **Claim:** Sanity check for each agent type on a small canonical game using the canonical `MultiAgentEnv` + `run_simulation` API.
- **Setup correctness:** OK. Three sections (`UrnAgent`, `QLearningAgent`, `TDLearningAgent`) share the same `graph`, `game_dicts`, and parameters (`N_AGENTS=2, N_FEATURES=2, N_SIGNALING_ACTIONS=2, N_FINAL_ACTIONS=4`). Each section constructs `MultiAgentEnv(...)` and calls `run_simulation(env, n_episodes=10000 or 15000, ..., plot=True)`.
- **Regime symmetry:** Not applicable — the notebook runs a single regime (`full_information=False, with_signals=True`) per agent type. Smoke-test by design.
- **Output → plot consistency:** No CSV outputs; plots are inline (consumed by the human reader).
- **Notes:** No seed is set, so each run is non-reproducible. Acceptable for a smoke-test notebook but noted against the skill's "set seeds early and log them" rule. `nbformat_minor=2` (no stable cell IDs).

### notebooks/Run_Simulations.ipynb

- **Claim:** Main runs — UrnAgent / QLearningAgent / TDLearningAgent on the canonical 2-feature game and the more-complex 3-feature game, all four `(full_information, with_signals)` cells per iteration.
- **Setup correctness:** OK on the canonical blocks (`n_features=2`, `n_signaling_actions=2`, `n_final_actions=4`, `obs_vars={0:[0],1:[1]}`). On the complex blocks (`n_features=3`, `obs_vars={0:[0,1],1:[1,2]}` — overlapping subsets per Phase 1 Axis 2), TD agent uses `gamma=0.99` rather than the canonical `γ=1` default; flag for user confirmation but not filed as a bug since the user controls the agent kwargs.
- **Regime symmetry:** Each iteration's `run_all_cases_for_iteration` constructs `game_dicts` and `obs_vars` once, then runs the same 4-cell `cases = [(False, False), (False, True), (True, False), (True, True)]` grid. `np.random.seed(iteration); random.seed(iteration)` is reset inside each `run_single_case` call, so each regime within an iteration sees the same nature-vector sequence. ✓ Paired comparison.
- **Output → plot consistency:** **Discrepancy** — the canonical blocks write `*_canonical.csv` and `plotting_results.ipynb` reads `*_canonical.csv` ✓. The **complex blocks** write `*_complex.csv` but `plotting_results.ipynb` reads `*_complex_randomized.csv`. → Filed as **Bug 6** in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md). The `*_complex.csv` files this notebook produces are orphaned; the `*_complex_randomized.csv` files plotting_results consumes are not regenerable from the current codebase.
- **Notes:** `game_dicts` are constructed inside `run_all_cases_for_iteration` *before* the per-case seed reset, so they depend on the worker's startup RNG state. With `joblib.Parallel(n_jobs=cpu_count())`, the loky backend assigns workers stochastically, which means individual rows of the saved CSV are not exactly reproducible from `iteration` alone. Population-level statistics are unaffected. Per the NOTEBOOK_WRITING_SKILL "silent bug" — consider migrating to `SeedSequence.spawn()`. Cell IDs missing.

### notebooks/Initializations_test.ipynb

- **Claim:** Effect of urn/Q-table initialization strategies. README links the saved figures (`results/initializations_*.png`) to this notebook, and the LEGACY_BUGS_LOG flags it as the sole notebook impacted by Bug 1.
- **Setup correctness:** **Critical discrepancy.** The experimental loop constructs `NetMultiAgentEnv(..., agent_type=QLearningAgent, initialize=True, initialization_weights=init_weights, ...)` (correct) but immediately overwrites `env.agents = [QLearningAgent(... no initialize ...) for _ in range(n_agents)]`. The replacement agents default to `initialize=False`, throwing away the env-constructor's pre-seeded Q-tables. → Filed as **Bug 5** in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md). High severity — the entire experiment is invalidated; the four `init_weights` curves in the saved figures show identical configurations modulo run-to-run noise. Bug 5 also masks Bug 1 and Bug 4: even after their fixes, the override still drops the initialized state.
- **Regime symmetry:** Not applicable — single regime (`with_signals=True, full_information=False`); the comparison axis is `init_weights`, which is what's broken.
- **Output → plot consistency:** Saved figures (`initializations_rewards.png`, `initializations_nmi.png`) are written from the in-memory `rewards_histories` / `signal_information_histories` dicts; no CSV intermediate. Filenames OK.
- **Notes:** The section header reads "# Urn Agent" but the loop constructs `QLearningAgent`. `UrnAgent` is imported but never used. So the LEGACY_BUGS_LOG Bug 1 "affected notebook" claim is inaccurate independent of Bug 5 — this notebook would not exercise UrnAgent's pre-seeded `action_urns` even if Bug 5 were fixed, because no UrnAgent is constructed in the loop. No seed setting. `nbformat_minor=2`.

### notebooks/Final_Costly_Signaling_Run_Simulations.ipynb

- **Claim:** Costly-signaling experiments — sweep `signal_cost ∈ U(0, 0.5)` per iteration on the partial-info-with-signals regime.
- **Setup correctness:** OK. `n_features=2`, `n_signaling_actions=2`, `n_final_actions=4`, `agents_observed_variables = {0:[0], 1:[1]}`. Constructs `NetMultiAgentEnv(..., costly_signaling=True, agent_type=UrnAgent, ...)` and uses `simulation_function(..., signal_cost=signal_cost, costly_signaling=True)`. Per-case seed reset inside `run_single_case_fixed` ✓.
- **Regime symmetry:** Single regime by design (`cases = [(False, True)]`). The variable axis is `signal_cost`, sampled uniformly per iteration.
- **Output → plot consistency:** Writes `urnagent_results_canonical_costly_signal.csv` ✓ matches what `plotting_results.ipynb` consumes.
- **Notes:** Same reproducibility caveat as Run_Simulations — `game_dicts` and `signal_cost` are generated outside the seeded scope (they use the worker's startup RNG state), so individual rows of the saved CSV are not reproducible from `iteration`. The notebook contains a large block of commented-out code (cell 8) that duplicates the active cell 4. Hygiene issue per the skill's "no hidden state / one concept per cell" rule. Cell IDs missing. Several empty cells (5, 7, 9 etc.) — cell-clutter.

### notebooks/Parameter_Optimization_wchoices.ipynb

- **Claim:** Bayesian hyperparameter search for QLearning and TDLearning, on canonical and complex models.
- **Setup correctness:** **Discrepancy in imports.** The four optimization functions use `Categorical`, `Real`, `Integer`, `Optimizer` (from `skopt` / `skopt.space`), `Parallel`, `delayed` (from `joblib`), `multiprocessing`, and `datetime` — none of which are in cell 3's imports. The first non-function cell that uses any of these (the `param_ranges = {... Categorical([...]) ...}` block) raises `NameError` on a fresh kernel. → Filed as **Bug 7** in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md). Medium severity — Restart-and-Run-All fails immediately.
- **Regime symmetry:** Single regime by design (`full_information=False, with_signals=True`); the variable axes are the hyperparameters.
- **Output → plot consistency:** Writes timestamped Bayesian-search CSVs and `q_opt_*.png` / `td_opt_*.png` figures. Outputs are not consumed by `plotting_results.ipynb`; this notebook is a research log feeding back into the tuned hyperparameter literals used in `Run_Simulations.ipynb` and `Initializations_test.ipynb`.
- **Notes:** Per-trial seed `seed = base_seed + len(results)` and `seeds = [seed + i*1000 for i in range(n_trials)]` produces deterministic but non-independent streams (the skill notes that adjacent / arithmetic seeds can correlate under some PRNGs; `SeedSequence.spawn` is the recommended alternative). Latent reproducibility issue, low severity. The TD complex search randomizes `n_signaling_actions` and `n_final_actions` per trial via `np.random.randint(2, 10)` — distinct from Run_Simulations' fixed-action complex setup, and unrelated to the Bug 6 mismatch (it writes `td_bayes_nmi_results_complex_randomized_*.csv`, not `td_learning_results_complex_randomized.csv`). Cell IDs missing.

### notebooks/plotting_results.ipynb

- **Claim:** Build the final figures from the saved CSVs in `results/`. Consumes `results/*.csv` and writes `results/*.png` via the helpers in [rl_signaling/plotting.py](rl_signaling/plotting.py).
- **Setup correctness:** OK on canonical blocks. Reads `urnagent_results_canonical.csv`, `qlearning_results_canonical.csv`, `td_learning_results_canonical.csv`, the corresponding costly-signal CSVs (Urn and QLearning), and the orphaned `*_complex_randomized.csv` files (see Bug 6). Helper calls (`plot_all_histograms`, `plot_regression`, `plot_reward_vs_cost`, `plot_nmi_vs_cost`, `count_negative_nmi`) all use the column names produced by Run_Simulations / Final_Costly_Signaling — matched.
- **Regime symmetry:** Filtering happens inside `plot_all_histograms` via `(with_signals, full_information)` pairs; consistent across the notebook.
- **Output → plot consistency:** **Two discrepancies.**
  - **Bug 6 (consumer side):** the "General Urns" / "General Games" blocks consume `*_complex_randomized.csv` files that no current notebook regenerates. The figures they emit (`Roth-Erev_complex_randomized_*.png`, etc.) are stale relative to any re-run of `Run_Simulations.ipynb`.
  - **Bug 8 (final cell):** the TD-learning "General Games" regression call uses `filename_prefix='Q-learning_complex_randomized'` — copy-paste from the Q-learning section. The saved figure path collides with the Q-learning regression PNGs and overwrites them with TD-learning content. Filed as **Bug 8** in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md), Low severity (naming only).
- **Notes:** Cell IDs missing (`nbformat_minor=2`).

### Summary

| Notebook | Status |
|---|---|
| `basic_unit_test.ipynb` | Clean (smoke test). |
| `Run_Simulations.ipynb` | **Bug 6** (complex CSV filename mismatch). |
| `Initializations_test.ipynb` | **Bug 5** (env.agents overwrite invalidates experiment); also masks Bugs 1 and 4. |
| `Final_Costly_Signaling_Run_Simulations.ipynb` | Clean against spec; cell-clutter (commented blocks, empty cells) noted. |
| `Parameter_Optimization_wchoices.ipynb` | **Bug 7** (missing imports — Restart-and-Run-All fails). |
| `plotting_results.ipynb` | **Bug 6** (consumer side) + **Bug 8** (filename_prefix typo). |

### Cross-cutting findings (not filed as individual bugs per severity bar)

1. **All six notebooks have `nbformat_minor` ≤ 2** — no stable cell IDs. The KB skill recommends nbformat ≥ 4.5 so programmatic edits via `cell_id` are stable across reorders. Repo-wide cleanup; not a numerical bug.
2. **Multiprocessing seed pattern.** `joblib.Parallel(n_jobs=cpu_count())(delayed(...))` runs each iteration in a worker subprocess. Per-case seeds (`np.random.seed(iteration)` inside the worker) reset Python and NumPy global state at the case boundary, but state used *outside* the seeded scope (game_dicts construction in `run_all_cases_for_iteration`, `signal_cost` draw in `Final_Costly_Signaling`) depends on the worker's startup RNG state. Population-level statistics are unaffected; individual rows are not row-reproducible from `iteration` alone. The KB skill's recommended `SeedSequence.spawn` pattern would close this gap.
3. **`env.agents = [...]` override pattern.** Used in 4 of 6 notebooks (Run_Simulations Q-learning + TD-learning blocks, Initializations_test, Parameter_Optimization Q + TD trials). Intent: inject tuned hyperparameters that the `NetMultiAgentEnv` constructor doesn't expose as kwargs. Side effect (caused Bug 5 in `Initializations_test`): anything passed to `NetMultiAgentEnv(..., initialize=True, ...)` is silently lost when the override doesn't preserve those kwargs. The canonical [rl_signaling.env.MultiAgentEnv](rl_signaling/env.py) accepts `agent_kwargs={...}` and would let these notebooks drop the override pattern.
4. **Code duplication across notebooks.** `run_single_case` / `run_all_cases_for_iteration` are near-verbatim duplicated across the canonical and complex sections of `Run_Simulations.ipynb` and across the four optimization functions in `Parameter_Optimization_wchoices.ipynb`. Per the KB skill's "logic placement: notebooks orchestrate, modules define" rule, these helpers should live in a `tests/experiment_helpers.py` (or similar) and be imported. Phase 5 candidate.

## Phase 5 — Fix plan

This section synthesizes the Phase 2–4 findings into a ranked, batched fix plan. Per the plan's Operating Rule 4, fixes are **not** applied in this session; this section is the proposal that a follow-up session executes.

### What Phase 4 already settled (don't re-debug)

The numerical sanity phase confirmed that every kernel-level identity in the codebase matches the math derived in [analytics/](analytics/):

- Shannon entropy is in bits (verified against `scipy.stats.entropy` to atol = 1e-12).
- Mutual information matches the by-hand derivation on six different signal-usage tables.
- `QLearningAgent` matches the closed form $Q_n = r(1 - (1-\alpha)^n)$ for $\alpha = 0.1$ across $n \in \{1, 2, 5, 10, 20, 50, 100\}$.
- `TDLearningAgent` reproduces the bootstrap and terminal updates exactly, and converges to $\mathbb{E}[r]$ at the Robbins-Monro rate.
- Costly signaling arithmetic is exact for every cost/null combination across two agents.
- `UrnAgent` Roth-Erev sampling matches the closed-form $(1 + nr/u_0)/(K + nr/u_0)$ to atol = 1e-12 and to a 5%-rel-tolerance Monte Carlo over 200 000 samples.

So all eight open bugs (1, 2 still open behaviorally, 4, 5, 6, 7, 8 — 1 and 3 are already fixed) are **structural / experimental**, not numerical. Fixes will not require touching `info_theory.py`, the agent update math, the env reward arithmetic, or the entropy formulas.

### Bug ledger

| # | Title | Severity | Status (after this session) | Affected results |
|---|---|---|---|---|
| 1 | `UrnAgent.action_urns` overwrite | High | **Fixed** in Phase 4 of refactor | None — masked by Bug 5 in the only consuming notebook |
| 2 | `TempNetMultiAgentEnv` nested NMI loop | Medium | Open | TD-learning CSVs (`Initial_NMI` averages over 5 episodes, `NMI` over 50, vs intended 10/100) |
| 3 | `utils.py` missing `import sys` | Medium | **Fixed** in Phase 1 of refactor | None |
| 4 | `QLearningAgent` `q_table_action` not pre-seeded | Medium | Open | None right now (Bug 5 masks it) |
| 5 | `Initializations_test.ipynb` `env.agents` overwrite | High | Open | `initializations_nmi.png`, `initializations_rewards.png` |
| 6 | `Run_Simulations` writes `*_complex.csv`, `plotting_results` reads `*_complex_randomized.csv` | High | Open | All 12 `*_complex_randomized_*.png` figures + 3 CSVs orphaned on each side |
| 7 | `Parameter_Optimization_wchoices` missing imports | Medium | Open | None on saved figures (research log only) |
| 8 | `plotting_results` final cell wrong filename_prefix | Low | Open | `Q-learning_complex_randomized_regression_*.png` (overwritten with TD content); `TD-learning_complex_randomized_regression_*.png` never produced |

Six open bugs total. Two are already closed (Bug 1 and Bug 3) and listed only for completeness.

### Severity × impact × effort

| # | Severity | Effort | Saved-fingerprint impact |
|---|---|---|---|
| 2 | Medium | **Trivial** if patching legacy env (~10 lines); or **moot** if notebooks migrate to canonical API | TD-learning CSV summary statistics in `td_learning_results_*.csv` |
| 4 | Medium | **Trivial** (~5 lines, mirror of Bug 1 fix) — but check user's design intent first | None (currently masked) |
| 5 | High | **Trivial** (Option A: ~10 lines, restore init kwargs in override) or **Medium** (Option B: ~50 lines, migrate notebook to `MultiAgentEnv` + `agent_kwargs`) | Both `initializations_*.png` need regeneration |
| 6 | High | **Medium** (~30–60 lines depending on direction; Option A reintroduces randomized action sizes; Option B retires the `_randomized` figures) | All 12 `*_complex_randomized_*.png` figures + 3 corresponding CSVs |
| 7 | Medium | **Trivial** (5 lines of imports + one `pyproject.toml` extras entry) | None (the saved `q_opt_*.png` and `td_opt_*.png` are from a prior run and the notebook is research-log) |
| 8 | Low | **Trivial** (1 line) | `Q-learning_complex_randomized_regression_*.png` and `TD-learning_complex_randomized_regression_*.png` need re-running together with the rest of the plotting notebook (depends on Bug 6 outcome) |

"Trivial" = under 30 lines of focused change; "Medium" = under 200 lines, contained; "Large" = cross-cutting / > 200 lines. No bug is "Large."

### Dependencies between fixes

```
Bug 5 ──┐  (Bug 5 masks Bug 4; until Bug 5 is fixed,
        │   any Bug 4 change has no observable effect)
Bug 4 ──┘

Bug 6  ──→  Bug 8  (the regression-PNG renaming Bug 8 fixes only matters
                    after the underlying *_complex_randomized.csv producer
                    is settled, since the consumer cell will be re-run)

Bug 5 fix Option B  ──→  Bug 2 becomes moot
                          (Option B migrates Initializations_test off the
                           legacy TempNetMultiAgentEnv path; if other
                           legacy-using notebooks also migrate, Bug 2 lives
                           only in deprecated code that no notebook calls)

Bug 7  (independent — orthogonal to all others)
```

### Hot path / Result-affecting / Latent classification

Per the plan's batching:

- **Hot path** (fix actively breaks current passing code): **none.** All six open bugs leave `pytest tests/` green; no test asserts the buggy behavior.
- **Result-affecting** (fix changes one or more saved figures): **Bug 2, Bug 5, Bug 6, Bug 8.** Each is itemized with the figure list above.
- **Latent** (fires only on error paths or unused code): **Bug 4** (currently masked by Bug 5; only fires once Bug 5 is fixed *and* a future re-run of `Initializations_test.ipynb` exercises `initialize=True`).
- **Notebook-only** (no figure / no result): **Bug 7** (Restart-and-Run-All gate; doesn't gate any saved CSV).

### Proposed fix order — three batches

#### Batch A — quick wins (dependency-free, no decisions required)

These four can land as a single follow-up commit. Total effort: ~30 lines across three files.

| # | Action | Files |
|---|---|---|
| 7 | Add missing imports (`scikit-optimize`, `joblib`, `multiprocessing`, `datetime`) to the imports cell of `Parameter_Optimization_wchoices.ipynb`. Add `scikit-optimize` to `[project.optional-dependencies] dev` in `pyproject.toml`. | `notebooks/Parameter_Optimization_wchoices.ipynb`, `pyproject.toml` |
| 8 | Change `filename_prefix='Q-learning_complex_randomized'` to `filename_prefix='TD-learning_complex_randomized'` in the final code cell of `plotting_results.ipynb`. | `notebooks/plotting_results.ipynb` |

After Batch A, Restart-and-Run-All works on `Parameter_Optimization_wchoices.ipynb` and the TD-learning regression PNG file naming is correct.

#### Batch B — structural fixes that need a user decision

These require an upfront product decision, then a focused fix.

| # | Decision needed | Recommended option | If chosen |
|---|---|---|---|
| 5 | Option A (preserve init kwargs in override) vs Option B (migrate `Initializations_test.ipynb` to `MultiAgentEnv` + `agent_kwargs`) | **Option B** — closes the env.agents-override pattern (which also caused subtle drift in three other notebooks) and brings the notebook in line with `basic_unit_test.ipynb`. | Resolves Bug 5; if the migration replaces the legacy `TempNetMultiAgentEnv` path, Bug 2 becomes moot for this notebook. Add a `UrnAgent` block alongside the QLearning block (the section header reads "# Urn Agent" but the code is QLearning-only — fix the header or split). |
| 4 | Whether the docstring's "pre-seed the **signaling** Q-table" intent is real or a refactor copy-paste artifact | Confirm with user; if intent is symmetric with UrnAgent fix, apply the analogous `q_table_action` pre-seed; if intent is signaling-only, update the docstring with a `Notes` section. | Trivial code change either way (~5 lines). Verify in `Initializations_test.ipynb` only after Batch B Bug 5 lands. |
| 6 | Option A (restore randomized action sizes in `Run_Simulations.ipynb`'s complex blocks; rename outputs to `*_complex_randomized.csv`) vs Option B (re-anchor `plotting_results.ipynb` on the existing fixed-action `*_complex.csv` files; retire the `_randomized` figures) | **Option A** — preserves the saved figures' meaning (varying action sizes is the more interesting experimental condition; the saved PNGs in `results/` reflect that). Option B discards 12 figures. | Modify the three complex-block functions in `Run_Simulations.ipynb` to draw `n_signaling_actions = np.random.randint(2, 10)` and `n_final_actions = np.random.randint(2, 10)` per iteration; rename outputs to `*_complex_randomized.csv`. Delete the orphaned `*_complex.csv` files. |

After Batch B, `notebooks/Initializations_test.ipynb` actually tests initialization, and the README's "Reproducing the figures" recipe runs end-to-end on a fresh checkout.

#### Batch C — defer / monitor

| # | Action | Justification |
|---|---|---|
| 2 | Defer indefinitely. | The legacy `TempNetMultiAgentEnv` is deprecated. If Batch B Option B migrates the active notebooks off the legacy path, Bug 2 lives only in code no notebook calls — making it dead-code rather than a live bug. The `Run_Simulations.ipynb` TD blocks still use the legacy path; if those are migrated to `MultiAgentEnv` (which already supports the same TD agent), Bug 2 becomes purely historical. |

If the user explicitly wants the legacy path to be kept callable (e.g. for someone reading the deprecated wrappers as a compatibility shim), the fix is trivial: lift the inner NMI loop out of the per-agent outer loop in `TempNetMultiAgentEnv.get_actions` ([rl_signaling/env.py:733-760](rl_signaling/env.py#L733-L760)) so it runs once per signal phase instead of per agent. Estimated ~10 lines.

### Saved-result regeneration list (after all batch A + B fixes land)

The follow-up session should re-run, in order:

1. `notebooks/Run_Simulations.ipynb` — produces `urnagent_results_canonical.csv`, `qlearning_results_canonical.csv`, `td_learning_results_canonical.csv`, and (after Batch B Bug 6) `*_complex_randomized.csv`.
2. `notebooks/Initializations_test.ipynb` — produces `initializations_nmi.png` and `initializations_rewards.png` (now with real init effect after Batch B Bug 5).
3. `notebooks/Final_Costly_Signaling_Run_Simulations.ipynb` — produces `urnagent_results_canonical_costly_signal.csv`, `qlearning_results_canonical_costly_signal.csv`. No fixes touch this notebook; re-run only if the canonical UrnAgent / QLearningAgent behavior has shifted (it has not, per Phase 4).
4. `notebooks/plotting_results.ipynb` — consumes everything from steps 1–3 and emits the figure PNGs in [results/](results/). After all batches, this should produce a complete and self-consistent figure set.

CSVs to delete after Batch B:

- `results/urnagent_results_canonical_costly_signal (1).csv` (Mac-Finder duplicate, no consumer).
- `results/urnagent_results_complex.csv`, `results/qlearning_results_complex.csv`, `results/td_learning_results_complex.csv` (orphaned by Bug 6 Option A; their replacements are the `*_complex_randomized.csv` versions).

### Cross-cutting cleanup (separate from bug fixes)

The Phase 3 cross-cutting findings warrant their own follow-up but are not gating any specific bug fix:

1. **`nbformat_minor` ≤ 2 across all six notebooks** — re-save them with a modern Jupyter (≥ 4.5) to get stable cell IDs. One-shot operation.
2. **Multiprocessing seed pattern** — the `joblib.Parallel(n_jobs=cpu_count())` workers don't see the per-iteration seed for `game_dicts` / `signal_cost` construction (which happen in `run_all_cases_for_iteration`, before the `np.random.seed(iteration)` call inside `run_single_case`). Population stats are unaffected; individual rows are not row-reproducible from `iteration`. Migrate to `numpy.random.SeedSequence().spawn()` for full reproducibility — recommended by the [NOTEBOOK_WRITING_SKILL](https://github.com/ignacioojea/knowledge-bases/blob/main/content/how-to/NOTEBOOK_WRITING_SKILL.md) but optional.
3. **`env.agents = [...]` override pattern** — repeated in 4 of 6 notebooks; caused Bug 5; would be eliminated by Batch B's migration to `agent_kwargs`. After Batch B, audit the remaining sites and either remove the overrides or document why they remain.
4. **Code duplication of `run_single_case` / `run_all_cases_for_iteration`** — extract into a `tests/experiment_helpers.py` (or a top-level `experiments/` module) to be imported by all four experiment notebooks.

### Final tally

- **Open bugs:** 6 (Bug 2, 4, 5, 6, 7, 8). Severity: 2 High, 3 Medium, 1 Low.
- **Closed bugs:** 2 (Bug 1, Bug 3 — fixed during the refactor).
- **Bugs that change saved figures:** 4 (Bug 2, 5, 6, 8).
- **Bugs that gate a notebook from running on a fresh kernel:** 1 (Bug 7).
- **Bug-fix dependencies:** Bug 5 must precede Bug 4. Bug 6 must precede Bug 8 (their fixes share a re-run of `plotting_results.ipynb`).
- **Recommended order:** Batch A (Bug 7, Bug 8) → Batch B (decisions on Bug 5, Bug 4, Bug 6 → fixes) → re-run notebooks → defer Bug 2 unless legacy path is kept canonically.
- **Phase 4 confirmed:** kernel-level math is correct. No fix touches the entropy / MI / Q-update / TD-update / cost-arithmetic kernels.

This Phase 5 plan completes the audit. The next session can pick it up at the Batch A entries and proceed top-down.
