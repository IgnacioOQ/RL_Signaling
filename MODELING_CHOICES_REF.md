# Modeling Choices Reference
- status: active
- type: reference
- id: rl_signaling.modeling_choices_ref
- description: Catalog of every modeling axis surfaced during the Phase 1 model-spec handshake — the options that were offered, the one that was picked, and the concrete code locations that would need to change to flip each option. Designed as a navigation aid for building variant models off the canonical implementation.
- label: [reference]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

This file is the design-space companion to [DEBUGGING_PLAN.md](DEBUGGING_PLAN.md). The Phase 1 confirmed spec there says **what the canonical model is**; this file enumerates the alternatives that were considered and rejected, plus pointers to the code that would have to change to adopt each alternative.

It is intentionally forward-looking. When a future variant ("what if signals were sequential instead of simultaneous?", "what if reward were clamped to zero?", "what if each agent had its own RNG?") needs scoping, this doc gives the option list and the touchpoints in one place — without having to re-walk the code.

The picked option for each axis is marked **(canonical)**. All file references point at the post-refactor canonical paths (`rl_signaling/`); the deprecated `NetMultiAgentEnv` / `TempNetMultiAgentEnv` / `simulation_function` / `temp_simulation_function` paths are not enumerated here.

---

## Axis 1 — Nature-vector distribution

**Question:** How is `nature_vector` distributed each episode?

**Options:**
1. **(canonical)** i.i.d. uniform binary — each of `n_features` bits drawn independently from `{0, 1}` with `p = 0.5`. Equivalent to "uniform over the 2^`n_features` states".
2. Other distribution — non-uniform marginals or correlated features.

**Code change to adopt option 2:** [rl_signaling/env.py:148](rl_signaling/env.py#L148) — `MultiAgentEnv.reset` calls `np.random.randint(0, 2, size=self.n_features)`. Replace this draw with the desired distribution (e.g. `np.random.binomial(1, p, size=...)` for non-uniform marginals; multivariate Bernoulli for correlated features). Also add a constructor parameter to plumb the distribution through. No other module needs to change — the rest of the pipeline treats `nature_vector` as opaque.

---

## Axis 2 — Observation interpretation

**Question:** How is `agents_observed_variables` interpreted?

**Options:**
1. **(canonical)** Per-agent list of feature indices; **overlap allowed**; observation is a tuple of the corresponding bits in agent-index order. e.g. `{0:[0,1], 1:[1,2]}` is legal.
2. Per-agent list, but overlap should be disallowed (partition only).
3. Other / clarify.

**Code change to adopt option 2:** [rl_signaling/env.py:75](rl_signaling/env.py#L75) — `MultiAgentEnv.__init__`. Add a validation pass: union the per-agent index sets and assert each feature index appears in at most one agent's list. Raise `ValueError` on overlap. No runtime path needs to change; this is a guard-only edit.

---

## Axis 3 — Observation type

**Question:** Are observations always tuples of integers (vs arrays, sorted, hashed)?

**Options:**
1. **(canonical)** `tuple[int, ...]` keyed by feature-index order in `agents_observed_variables[i]`.
2. Same tuples, but canonicalized (sorted) before keying.
3. Other.

**Code change to adopt option 2:** [rl_signaling/env.py:151-156](rl_signaling/env.py#L151-L156) — `MultiAgentEnv.reset` builds the per-agent observation tuple by iterating `self.observed_variables[i]`. Sort the index list at construction time (in `__init__`) so the iteration in `reset` is automatically order-canonical. Note that this changes the dict keys used by every agent's urn / Q-table — pre-existing saved Q-tables would be incompatible.

---

## Axis 4 — Signal alphabet shape

**Question:** Can `n_signaling_actions` vary per agent?

**Options:**
1. **(canonical)** No — single global `n_signaling_actions` for all agents.
2. Yes — per-agent alphabet sizes.

**Code change to adopt option 2:** broad surface — both env and agent constructors take a single `n_signaling_actions` scalar today. Touchpoints:
- [rl_signaling/env.py:75](rl_signaling/env.py#L75) — promote `n_signaling_actions` to `Sequence[int]`; null-signal index becomes per-agent (`n_signaling_actions[i] - 1`).
- [rl_signaling/agents.py:202](rl_signaling/agents.py#L202), [:371](rl_signaling/agents.py#L371), [:533](rl_signaling/agents.py#L533) — `UrnAgent`, `QLearningAgent`, `TDLearningAgent` constructors all take `n_signaling_actions` as a scalar; the env currently constructs them with the same value for every agent.
- [rl_signaling/env.py:269](rl_signaling/env.py#L269) — `_send_signals` would need per-sender null-index lookup.

---

## Axis 5 — Null-signal placement and receivability

**Question:** When `costly_signaling=True`, where does the null signal sit and is it receivable?

**Options:**
1. **(canonical)** Null at index `n_signaling_actions - 1`; senders pay no cost for null; receivers do **not** see null (silence is silent — receiver observation is shorter when neighbours emit null).
2. Null at highest index, **receivable** — receivers see null as a normal symbol.
3. Null at highest index, but observation length should be **constant** — pad with null for missing senders.
4. Other.

**Code change to adopt option 2:** [rl_signaling/env.py:269-281](rl_signaling/env.py#L269-L281) — `MultiAgentEnv._send_signals` skips appending the signal when it equals `self._null_signal_index`. Remove the conditional so every neighbour's signal is appended. Receiver tuple length becomes constant (= number of in-neighbours).

**Code change to adopt option 3:** Same location, but instead of removing the skip, replace it with an `else: append(null_token)` branch where `null_token` is a sentinel distinct from a "real" null (or use the null index itself but tag the position differently). This is functionally equivalent to option 2 unless you want a separate token to mark "no message" vs "explicit null".

---

## Axis 6 — Signal timing

**Question:** Is the signaling step simultaneous or sequential?

**Options:**
1. **(canonical)** Simultaneous — all agents pick from the same pre-signal observations, then signals are delivered.
2. Sequential by agent-index — agent 0 signals first, agent 1 sees agent 0's signal before signaling.

**Code change to adopt option 2:** [rl_signaling/env.py:160-191](rl_signaling/env.py#L160-L191) — `MultiAgentEnv.step_signal` currently iterates agents to compute signals from a frozen observation snapshot, then calls `_send_signals` once. To go sequential: interleave the loop — for each agent in agent-index order, compute its signal from its current (possibly-augmented) observation, then immediately deliver it to its successors before moving to the next agent. Note this also breaks the current "frozen snapshot" semantics that `step_signal`'s return tuple relies on — the signature would need a redesign.

---

## Axis 7 — Edge directionality

**Question:** What does edge `(u, v)` mean?

**Options:**
1. **(canonical)** "u sends to v"; receiver `i` reads from `graph.predecessors(i)`.
2. "v sends to u" / "u listens to v"; receiver `i` reads from `graph.successors(i)`.

**Code change to adopt option 2:** [rl_signaling/env.py:274](rl_signaling/env.py#L274) — `MultiAgentEnv._send_signals` uses `self.graph.predecessors(i)`. Swap to `self.graph.successors(i)`. No other site uses graph adjacency, but verify any future code added downstream uses the same convention.

---

## Axis 8 — Self-loops

**Question:** Are self-loops permitted (an agent receiving its own signal)?

**Options:**
1. **(canonical)** Permitted but expected to be absent; if present, the agent receives its own signal in its observation. Not filtered.
2. Disallowed — env should raise / filter them out.
3. Permitted and meaningful (e.g. as memory).

**Code change to adopt option 2:** Add a guard in [rl_signaling/env.py:75](rl_signaling/env.py#L75) — `MultiAgentEnv.__init__`. After accepting `graph`, run `assert not any(graph.has_edge(i, i) for i in graph.nodes)` and raise `ValueError` on self-loops. Or filter inline: `graph = nx.DiGraph((u,v) for u,v in graph.edges if u != v)`.

**Code change to adopt option 3:** No code change needed — option 3 is what the canonical code already does mechanically. The semantic "this is memory" is a documentation/interpretation choice.

---

## Axis 9 — Multi-edges

**Question:** Are parallel/multi-edges supported?

**Options:**
1. **(canonical)** Undefined / not supported — `nx.DiGraph` only.
2. Explicitly disallowed with a runtime check.

**Code change to adopt option 2:** [rl_signaling/env.py:75](rl_signaling/env.py#L75) — `MultiAgentEnv.__init__`. Add `assert not isinstance(graph, nx.MultiDiGraph)` and raise `TypeError` otherwise. One line of guard code.

---

## Axis 10 — Game-dict key

**Question:** What is the state key in `game_dicts[i]`?

**Options:**
1. **(canonical)** Full `nature_vector` (as a tuple) — same key for all agents, regardless of `full_information`.
2. Per-agent observation tuple — different key for different agents.
3. Other.

**Code change to adopt option 2:** [rl_signaling/env.py:223](rl_signaling/env.py#L223) — `MultiAgentEnv.reward` looks up `game_dicts[i][state_key]` where `state_key = tuple(self.nature_vector)`. Replace with the per-agent observation: `state_key = observations[i]` (passed in or recomputed). This fundamentally changes the model — payoff would no longer require signaling, since each agent's reward depends only on what it observes. Game-dict generators in [rl_signaling/games.py](rl_signaling/games.py) would also need to be regenerated against the per-agent observation space, not the full state space.

---

## Axis 11 — Full-information observation key collision

**Question:** When `full_information=True`, do all agents share the same observation key?

**Options:**
1. **(canonical)** Yes — every agent observes the full nature_vector; per-agent Q-tables / urns prevent any collision.
2. Other.

**Code change:** No change needed — the canonical answer is what the code does. Each agent owns its own urn / Q-table dict; cross-agent collisions are structurally impossible. This axis is here purely as a sanity check for future readers.

---

## Axis 12 — Reward type and floor

**Question:** What is the reward type, and is there a non-negativity floor?

**Options:**
1. **(canonical)** Floats (or ints; mixing is allowed in game dicts); no implicit floor — costly signaling can drive net rewards negative.
2. Should be strictly non-negative — clamp at zero.
3. Other.

**Code change to adopt option 2:** [rl_signaling/env.py:236-241](rl_signaling/env.py#L236-L241) — `MultiAgentEnv.reward` returns `r - signal_cost[i]` after the cost deduction. Wrap in `max(0, …)` to clamp. Note this would interact with [rl_signaling/agents.py:305](rl_signaling/agents.py#L305) (UrnAgent's `max(0, urn + reward)` clamp) — clamping the env reward to zero would make the urn clamp redundant for cost-driven negatives, but not for cases where the game dict itself contains negative payoffs.

---

## Axis 13 — Signal-cost shape

**Question:** Is `signal_cost` per-agent only, or could it be per-state / per-signal?

**Options:**
1. **(canonical)** Per-agent scalar — same cost regardless of state or signal.
2. Per-signal — different cost for different non-null signals.
3. Per-state — cost depends on the underlying nature_vector.

**Code change to adopt option 2:** [rl_signaling/env.py:214](rl_signaling/env.py#L214) — `MultiAgentEnv.reward` already takes `signal_cost: Sequence[float] | None` keyed per agent. Promote to `Sequence[Sequence[float]]` keyed per `(agent, signal)`. Update [rl_signaling/env.py:238](rl_signaling/env.py#L238) lookup to `signal_cost[i][signals[i]]`. Decide whether the null signal still costs zero (likely yes — keep the existing null guard).

**Code change to adopt option 3:** Same `reward()` method. The cost would need to depend on `state_key` — change the lookup to `signal_cost[i][state_key]` or a callable `signal_cost(i, state_key)`. State-keyed costs are unusual; consider whether a callable is cleaner than a precomputed dict.

---

## Axis 14 — Cost flow

**Question:** Where is the cost applied?

**Options:**
1. **(canonical)** Cost subtracted from per-episode reward (`rewards_history` records `game_reward - cost`); does not propagate into Q-bootstrap targets via a separate channel.
2. Cost recorded **separately** — `rewards_history` stores gross reward; cost tracked in a different series.

**Code change to adopt option 2:** Two-site change.
- [rl_signaling/env.py:236-241](rl_signaling/env.py#L236-L241) — return both the gross reward and the cost; signature becomes `tuple[list[float], list[float]]` or similar.
- [rl_signaling/simulation.py](rl_signaling/simulation.py) — the canonical `run_simulation` consumes `env.reward(...)` and writes to `rewards_history`. It would need a parallel `cost_history` accumulator and a decision about which series the agent learning rules see (bootstrap targets currently use the net reward; option 2 forces a choice).

---

## Axis 15 — Null-signal cost

**Question:** Does the null signal cost anything?

**Options:**
1. **(canonical)** No — null is free; only non-null signals incur cost.
2. Yes — null also costs (smaller amount).

**Code change to adopt option 2:** [rl_signaling/env.py:236-241](rl_signaling/env.py#L236-L241) — `MultiAgentEnv.reward` currently has `r - signal_cost[i] if signals[i] != self._null_signal_index else r`. Drop the conditional (or replace with a separate `null_cost` parameter), so null-signal episodes also pay a cost.

---

## Axis 16 — Number of information regimes

**Question:** How many information regimes are intended?

**Options:**
1. **(canonical)** Three primary regimes (full / partial-no-signals / partial-with-signals); full-info+no-signals is the trivial baseline some figures show for completeness.
2. Four equally important regimes (all 2×2 of `full_information` × `with_signals`).

**Code change:** No code change — both options use the same `full_information` and `with_signals` flags. The difference is purely interpretive (which regimes are first-class research conditions vs sanity baselines). Axis lives in the documentation and the experiment notebooks under [notebooks/](notebooks/), not the package.

---

## Axis 17 — Full-info + signals interaction

**Question:** When `full_information=True` AND `with_signals=True`, are signals suppressed?

**Options:**
1. **(canonical)** No — signals run normally; expected to be redundant (NMI low) but mechanically present.
2. Yes — env should auto-suppress signals when full information is on.

**Code change to adopt option 2:** [rl_signaling/env.py:75](rl_signaling/env.py#L75) — `MultiAgentEnv.__init__`. Add a guard: if `full_information=True`, force-disable the signal step (e.g. set an internal `_signals_active` flag; have `step_signal` return immediately). The runner [rl_signaling/simulation.py](rl_signaling/simulation.py) would need to handle the resulting empty signal-history shape, similar to its existing `with_signals=False` branch.

---

## Axis 18 — UrnAgent reward clamping

**Question:** What is the Roth–Erev update rule?

**Options:**
1. **(canonical)** `urn[s][a] = max(0, urn[s][a] + reward)` — positive-reinforcement clamp; negative rewards clamp to zero; zero rewards leave the urn unchanged.
2. Un-clamped: `urn[s][a] += reward`, allow negative counts.
3. Floor on zero rewards: `max(epsilon, urn + reward)` — keep all arms live.

**Code change to adopt option 2:** [rl_signaling/agents.py:303-313](rl_signaling/agents.py#L303-L313) — `UrnAgent.update_signals` and `update_actions` both wrap the assignment in `max(0, ...)`. Drop the `max(0, ...)` and assign `urn[s][a] += reward` directly. **Caution:** the urn-as-distribution interpretation breaks (negative weights) — `get_signal` / `get_action` would need a softmax or shift-and-renormalize step before sampling, or the urn semantics must change wholesale.

**Code change to adopt option 3:** Same two methods. Replace `max(0, ...)` with `max(epsilon, ...)` for some small epsilon. Keeps the distribution interpretation intact and prevents arms from dying.

---

## Axis 19 — Q-learning hyperparameters

**Question:** Learning rate, bootstrap, exploration decay.

**Options:**
1. **(canonical)** `α = 0.1` constant (hardcoded); `td_target = reward` (no bootstrap; episodes are single-step); exploration decay applied per-channel (signal phase and action phase have separate decay schedules).
2. `α` should be a constructor parameter (not hardcoded).
3. Bootstrap should be present (multi-step semantics).

**Code change to adopt option 2:** [rl_signaling/agents.py:447-477](rl_signaling/agents.py#L447-L477) — `QLearningAgent.update_signals` and `update_actions` both contain a hardcoded `alpha = 0.1` literal. Promote `alpha` to a constructor parameter at [rl_signaling/agents.py:371](rl_signaling/agents.py#L371) and reference `self.alpha` in the update.

**Code change to adopt option 3:** Same two methods. Replace `td_target = reward` with `td_target = reward + gamma * np.max(Q[next_state])`. Requires plumbing `next_state` into the update method (currently the env discards it because episodes are terminal). Episode loop in [rl_signaling/simulation.py](rl_signaling/simulation.py) and the `update_episode` signature in [rl_signaling/agents.py:484](rl_signaling/agents.py#L484) would both need to thread the next state through.

---

## Axis 20 — TD-learning structure

**Question:** Bootstrap, learning rate, table sharing.

**Options:**
1. **(canonical)** Bootstrap from `next_state` (`γ · max`); count-based learning rate `1 / N(s, a)`; **shared** Q-table across signal and action phases — phases distinguishable by tuple length (signal-phase obs is shorter; action-phase obs has received signals appended).
2. Bootstrap as above, but Q-tables should be **separate** per phase (`q_table_signaling`, `q_table_action`).
3. No bootstrap (terminal targets in both phases).

**Code change to adopt option 2:** [rl_signaling/agents.py:498-650](rl_signaling/agents.py#L498-L650) — `TDLearningAgent` currently uses one `self.q_table` dict. Split into `self.q_table_signaling` and `self.q_table_action`, route `get_signal` / `get_action` and `update_episode` to the appropriate dict. Note that this also closes a small theoretical concern: under the canonical shared-table approach, a bug that produced equal-length keys across phases (e.g. an agent with no in-neighbours) could collide.

**Code change to adopt option 3:** [rl_signaling/agents.py:641-650](rl_signaling/agents.py#L641-L650) — `TDLearningAgent.update_episode` performs the signal-phase TD bootstrap then the action-phase terminal update. Drop the `gamma * max(Q[next_state])` term in the signal-phase update — set `td_target = reward` for both phases. Makes TDLearningAgent equivalent to QLearningAgent up to the learning-rate schedule.

---

## Axis 21 — NMI formula

**Question:** What normalization is used?

**Options:**
1. **(canonical)** `NMI = I(S; O) / H(O)` — asymmetric, output-side normalization.
2. Geometric NMI: `I(S; O) / sqrt(H(S) · H(O))`.
3. Other.

**Code change to adopt option 2:** [rl_signaling/info_theory.py:59](rl_signaling/info_theory.py#L59) — `compute_mutual_information` divides by `H_O`. Replace with `np.sqrt(H_S * H_O)`. The function returns `(I, NMI)`; only the second value changes. Plotting code in [rl_signaling/plotting.py](rl_signaling/plotting.py) consumes the NMI series opaquely so it does not need to change. Existing saved CSVs that recorded NMI would mean a different thing post-change — flag in any writeup.

---

## Axis 22 — H(O)=0 convention

**Question:** What is returned when `H(O) == 0` (constant signal)?

**Options:**
1. **(canonical)** `NMI := 0`.
2. `NMI := NaN`.
3. Raise.

**Code change to adopt option 2:** [rl_signaling/info_theory.py:59](rl_signaling/info_theory.py#L59) — currently `NMI = I_S_O / H_O if H_O > 0 else 0`. Replace the `else 0` branch with `else float('nan')`. Downstream callers (plotting, golden tests) currently assume a numeric value — flag any that would silently propagate NaN.

**Code change to adopt option 3:** Same line. Replace with an explicit `if H_O == 0: raise ZeroDivisionError("H(O) is zero — NMI undefined")`. Forces every caller to guard or catch.

---

## Axis 23 — UCB tie-break with zero counts

**Question:** What does UCB do when all action counts are zero?

**Options:**
1. **(canonical)** Add tiny epsilon (`1e-5`) to counts to avoid div-by-zero; resulting bonus is large but finite; equivalent to picking near-uniformly on the very first call.
2. First-pass mandatory exploration: pick each action exactly once before applying UCB1.
3. Other.

**Code change to adopt option 2:** [rl_signaling/agents.py:106](rl_signaling/agents.py#L106) — `_select_action` adds `safe_counts = counts + 1e-5`. Replace with: if any count is zero, return `np.argmin(counts)` (i.e. pick an unvisited arm); otherwise apply the UCB1 formula. Removes the epsilon and matches textbook UCB1.

---

## Axis 24 — Per-agent vs shared RNG

**Question:** Do agents share the global RNG?

**Options:**
1. **(canonical)** Shared global `np.random` / `random` state; iteration order is agent-index order — agent 0 always draws from RNG before agent 1.
2. Each agent has an independent RNG.

**Code change to adopt option 2:** broad surface — every `random.uniform`, `np.random.choice`, `np.random.randint` call would need to route through a per-agent generator instead of the global state.
- [rl_signaling/agents.py:84](rl_signaling/agents.py#L84) (egreedy `random.uniform`), [:97](rl_signaling/agents.py#L97) (softmax `np.random.choice`), [:108](rl_signaling/agents.py#L108) (UCB `np.argmax`, no RNG) — `_select_action`. Add a `rng: np.random.Generator` parameter and use `rng.uniform(...)` / `rng.choice(...)`.
- [rl_signaling/agents.py:202](rl_signaling/agents.py#L202), [:371](rl_signaling/agents.py#L371), [:533](rl_signaling/agents.py#L533) — agent constructors. Take a `seed` or `rng` and store `self.rng`.
- [rl_signaling/env.py:148](rl_signaling/env.py#L148) — env's nature draw also uses global state; decide whether the env shares one of the agent RNGs or has its own.

This is the most invasive of the axes; probably warrants a dedicated branch.

---

## Cross-axis interaction notes

Some axes are coupled — flipping one without the matching counterpart can produce inconsistent semantics. Watch for these:

- **Axis 18 (urn clamping) ↔ Axis 12 (reward floor) ↔ costly signaling.** If you adopt option 2 of axis 18 (un-clamped urn), the urn-as-probability interpretation breaks for any reward stream that can go negative. Adopting axis 12 option 2 (clamp env reward at zero) would "fix" this by guaranteeing non-negative rewards reach the urn — but it also changes what the QLearning / TDLearning agents see, which may not be desired. **More fundamentally:** the canonical Roth-Erev urn is defined for non-negative *integer* rewards, while costly signaling produces *real-valued* and potentially *negative* rewards. The project's clamped variant is a workaround, not a principled extension. The costly-signaling experiment is therefore theoretically ill-defined under `UrnAgent`. Q-learning and TD-learning handle the real-valued, signed reward range natively. See [analytics/agent_urn.md, "Applicability constraints"](analytics/agent_urn.md#applicability-constraints--when-roth-erev-is-well-defined) and [analytics/costly_signaling.md, "Compatibility with the project's three agent types"](analytics/costly_signaling.md#compatibility-with-the-projects-three-agent-types) for the formal treatment, and [LEGACY_ERRORS_LOG.md Error 5a](LEGACY_ERRORS_LOG.md#error-5a--roth-erev-costly-figures-unreproducible-cost-protocol-drift) for the audit consequence.
- **Axis 5 (null receivability) ↔ Axis 6 (signal timing).** Sequential signaling (axis 6 option 2) only makes a difference if downstream agents can actually see the upstream signal — which depends on the receivability convention. Going sequential without changing axis 5 is fine (later agents see earlier non-null signals); going sequential with axis 5 option 2 (null is observable) gives later agents a "this neighbour explicitly chose to be silent" token.
- **Axis 13 (signal-cost shape) ↔ Axis 14 (cost flow).** Per-state or per-signal cost shapes (axis 13 options 2/3) compound the bookkeeping question raised by axis 14 — separate-channel tracking becomes more useful when the cost is structured rather than a flat scalar.
- **Axis 19 (Q-learning bootstrap) ↔ Axis 20 (TD structure).** Adopting axis 19 option 3 (Q-learning bootstrap) would make Q-learning and TD-learning differ only in their learning-rate schedule (constant α vs `1 / N`), undermining the experimental contrast between the two agents.

---

## How to use this file

When designing a variant model:

1. Identify which axes you want to flip from the canonical answer. The canonical column matches what the current code does.
2. Read the cross-axis interaction notes to spot hidden dependencies.
3. For each flipped axis, follow the code-change pointer to the relevant module.
4. Once changes land, update [DEBUGGING_PLAN.md](DEBUGGING_PLAN.md)'s Phase 1 spec section to reflect the new canonical, or fork it into a variant-specific spec doc. Either way, leave this file's option tree intact — it documents the design space, not the current default.
