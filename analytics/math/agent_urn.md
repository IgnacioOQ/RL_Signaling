# Roth–Erev urn agent

- status: active
- type: explanation
- id: rl_signaling.analytics.agent_urn
- description: Mathematical description of UrnAgent — the Roth–Erev reinforcement-learning rule used by rl_signaling/agents.py:UrnAgent. Covers the per-state urn data structure, the sampling probability, the reinforcement update with non-negativity clamp, lazy initialization, and the asymptotic relationship between accumulated reward and urn weights.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

The simplest of the three agents in [rl_signaling/agents.py](../../rl_signaling/agents.py) is `UrnAgent`, a Roth–Erev–style learner that samples actions in proportion to "urn counts" and reinforces the chosen action by the reward received. This file gives the formal definitions, derives the sampling distribution, walks through the update rule including the non-negativity clamp, and derives the asymptotic behavior under stationary feedback.

The implementation is at [rl_signaling/agents.py:162-327](../../rl_signaling/agents.py#L162-L327).

## State of the agent

Each `UrnAgent` instance maintains two collections of urns:

**Signaling urn.** A dictionary $\sigma\text{-urn}$ mapping each direct observation $\mathbf{o} \in \mathcal{V}_i$ to a non-negative integer-valued weight vector of length $K$ (the signaling alphabet size, possibly augmented by the null signal in the costly setting):

$$\sigma\text{-urn}[\mathbf{o}] \in \mathbb{R}_{\ge 0}^{K}.$$

The $a$-th entry $\sigma\text{-urn}[\mathbf{o}][a]$ is the "weight" of signal $a$ given observation $\mathbf{o}$.

**Action urn.** A dictionary $\alpha\text{-urn}$ mapping each post-signal observation $\tilde{\mathbf{o}}$ to a weight vector of length $M$ (the action alphabet size):

$$\alpha\text{-urn}[\tilde{\mathbf{o}}] \in \mathbb{R}_{\ge 0}^{M}.$$

Note the **distinct keys** for the two dictionaries: $\sigma\text{-urn}$ is keyed by the direct observation $\mathbf{o}$, while $\alpha\text{-urn}$ is keyed by the post-signal observation $\tilde{\mathbf{o}}$. The two are different lengths whenever signals are present and not all neighbours emit null.

Implementation: `self.signaling_urns: dict` and `self.action_urns: dict` at [rl_signaling/agents.py:221-238](../../rl_signaling/agents.py#L221-L238).

## Sampling probability

Given an urn vector $\mathbf{u} = (u_0, u_1, \dots, u_{K-1}) \in \mathbb{R}_{\ge 0}^{K}$ with $\sum_a u_a > 0$, the action is sampled in proportion to the entries:

$$\boxed{\; \mathbb{P}[\sigma = a \mid \mathbf{u}] \;=\; \frac{u_a}{\sum_{a'=0}^{K-1} u_{a'}}. \;}$$

The same formula holds for action selection by replacing the alphabet size and the urn role.

Implementation: `np.random.choice(self.n_signaling_actions, p=probability_weights)` at [rl_signaling/agents.py:273](../../rl_signaling/agents.py#L273) for signals and [rl_signaling/agents.py:302](../../rl_signaling/agents.py#L302) for actions.

### Edge case: empty urn

If at any point $\sum_a u_a = 0$ — for example, after a long run of negative rewards has clamped every entry to zero — the formula above is undefined (0/0). The code handles this defensively at [rl_signaling/agents.py:267-270](../../rl_signaling/agents.py#L267-L270):

```python
if total_sum <= 0:
    urn_values = np.ones(self.n_signaling_actions)
    self.signaling_urns[state] = urn_values
    total_sum = self.n_signaling_actions
```

The urn is reset to uniform $\mathbf{1}$, restoring a well-defined uniform distribution and resuming learning from there. This guard is rarely triggered in practice but ensures `get_signal` / `get_action` never return NaN.

## Lazy initialization

When the agent first observes a state $\mathbf{o}$ that is not yet a key in $\sigma\text{-urn}$, the urn is **lazy-initialized** to the all-ones vector:

$$\sigma\text{-urn}[\mathbf{o}] := \mathbf{1} = (1, 1, \dots, 1) \in \mathbb{R}^K.$$

This makes the very first signal selection at $\mathbf{o}$ uniform over the alphabet. Same logic for $\alpha\text{-urn}$.

Implementation: [rl_signaling/agents.py:259-260](../../rl_signaling/agents.py#L259-L260) and [rl_signaling/agents.py:290-291](../../rl_signaling/agents.py#L290-L291).

## Eager (one-hot) initialization

When the agent is constructed with `initialize=True`, the signaling urn (and action urn — post-Phase-4 fix) is pre-populated with one-hot vectors via `create_initial_signals`. For each $\mathbf{o} \in \mathcal{V}_i$, a unique one-hot vector $\mathbf{e}_{\pi(\mathbf{o})}$ is assigned:

$$\sigma\text{-urn}[\mathbf{o}] = n \cdot \mathbf{e}_{\pi(\mathbf{o})} + m \cdot (\mathbf{1} - \mathbf{e}_{\pi(\mathbf{o})}),$$

where $n$ and $m$ are the `initialization_weights` (default $(1, 0)$) and $\pi$ is a fixed bijection $\mathcal{V}_i \to \{0, \dots, K-1\}$ (chosen via `random.shuffle`).

With $(n, m) = (1, 0)$ the agent's initial signaling policy is **deterministic**: each observation maps to a unique signal with probability 1. With $(n, m) = (10, 1)$ the policy is mostly deterministic but allows occasional drift; the larger ratio $n/(n + (K-1)m)$ pins the dominant action more strongly.

This is the lever that the `Initializations_test.ipynb` experiment was designed to study. (See Bug 5 in [LEGACY_BUGS_LOG.md](../../docs/code-audit/LEGACY_BUGS_LOG.md) for why the experiment as written silently bypassed this lever.)

## The Roth–Erev update

The **update** after observing reward $r$ for choosing signal $a$ in state $\mathbf{o}$ is:

$$\boxed{\; \sigma\text{-urn}[\mathbf{o}][a] \;\leftarrow\; \max\big(0,\; \sigma\text{-urn}[\mathbf{o}][a] + r\big). \;}$$

Same rule for action urns with $a \to \alpha$ and $\sigma\text{-urn} \to \alpha\text{-urn}$. Implementation: [rl_signaling/agents.py:304-314](../../rl_signaling/agents.py#L304-L314).

The non-negativity clamp at zero (Phase 1 [Axis 18](../../docs/code-audit/DEBUGGING_PLAN.md#agent-learning-rules)) is what differentiates this from the **un-clamped** Roth–Erev variant. Both are in the literature; the un-clamped variant allows urn weights to go negative (which breaks the probabilistic interpretation of the urn) so it requires a softmax or shift-and-renormalize step in `get_signal`. The project's clamped variant preserves the urn-as-distribution interpretation directly.

### Properties of the clamped update

1. **Monotonicity in reward.** For non-negative rewards, the update strictly increases (or holds constant) the chosen action's weight. This is the basic positive-reinforcement principle.
2. **Floor at zero.** For sufficiently negative rewards, the chosen action's weight is clamped to zero. Once at zero, future updates with $r > 0$ recover the action; with $r \le 0$ the action stays at zero (and is never sampled, since $\mathbb{P}[a] = 0/(\text{sum})$). The defensive guard above prevents the *whole row* from being zero.
3. **Zero-reward updates are no-ops.** When $r = 0$, the update leaves the urn unchanged.
4. **Non-chosen actions are not updated.** Only the action that was sampled receives the reinforcement signal. This is the classical operant-conditioning shape.

## Closed form: single state, constant reward

Take a single observation $\mathbf{o}$, and assume the same action $a^\star$ is selected every time (e.g., because $a^\star$ is initially the only nonzero entry, so the lazy-init falls back on selection probability $1/K$ but quickly converges to $a^\star$ once reinforcement takes hold). Each pull yields reward $r > 0$.

After $n$ pulls, the urn entry for $a^\star$ has accumulated

$$\sigma\text{-urn}[\mathbf{o}][a^\star] = u_0 + n \cdot r,$$

where $u_0$ is the initial weight (1 if lazy-init, $n_{\text{init}}$ if eager). All other entries remain at $u_0$ (or $m_{\text{init}}$).

The sampling probability for $a^\star$ after $n$ pulls is

$$\mathbb{P}[\sigma = a^\star \mid n \text{ pulls}] = \frac{u_0 + n r}{u_0 + n r + (K - 1) u_0} = \frac{1 + n r / u_0}{K + n r / u_0}.$$

As $n \to \infty$ with $r > 0$ fixed:

$$\mathbb{P}[\sigma = a^\star] \to 1.$$

Quantitatively, for the test scenario $u_0 = 1$, $K = 4$, $r = 1$, $n = 250$ (one of four states encountered on average over 1000 episodes with uniform nature):

$$\mathbb{P}[\sigma = a^\star] = \frac{1 + 250}{4 + 250} = \frac{251}{254} \approx 0.988.$$

This is the basis for the convergence assertion in [tests/test_numerical_sanity.py::test_urn_agent_converges_to_optimal_action_in_full_information](../../tests/test_numerical_sanity.py#L186-L228) — the assertion bar is set at $\ge 0.95$ to leave headroom for stochastic exploration that picks suboptimal actions early in training.

The full derivation generalizes to time-varying reward streams. If the reward at episode $t$ is $r_t \ge 0$ and the chosen action is $a_t$:

$$\sigma\text{-urn}[\mathbf{o}][a] = u_0 + \sum_{t : a_t = a} r_t.$$

The sampling probability is the fraction of accumulated reinforcement on $a$ over the total. In words: **urn weight ≈ accumulated reward**.

## Applicability constraints — when Roth-Erev is well-defined

The classical Roth-Erev urn (Roth & Erev 1995; Erev & Roth 1998) was specified for **non-negative integer reinforcement**. The probability formula

$$\mathbb{P}[\sigma = a \mid \mathbf{u}] = \frac{u_a}{\sum_{a'} u_{a'}}$$

rests on the urn-as-ball-counter interpretation: each play of $a$ adds $r$ "balls" to the $a$-th compartment, and sampling proportional to ball counts is the canonical urn-scheme primitive (Pólya, Friedman, Hoppe — see standard probabilistic combinatorics references).

This project's UrnAgent extends the rule to real-valued rewards (the urn weights become real numbers $\ge 0$) and to negative rewards (via the clamp $\max(0, \cdot)$). Both extensions are workarounds, not principled generalizations. The constraints they violate:

### Constraint 1 — Non-integer rewards break the urn-as-counter metaphor

When $r \in \mathbb{R}$, the urn weights are real numbers and the probability formula still computes — it is just a normalized weight scheme. The classical "balls in an urn" interpretation no longer applies; one cannot sample from such an "urn" by physical analogy.

For the project's canonical experiments — `create_random_canonical_game(n=1, m=0)` returns rewards in $\{0, 1\}$ — this is **not** an issue: every $r$ is integer-valued and the classical interpretation holds.

For costly signaling — net reward $r = G_i(\mathbf{v}, \alpha) - c_i \cdot \mathbb{1}[\sigma \neq \nu]$, with $c_i \in (0, 0.5)$ — the rewards become real-valued (`0.75`, `0.5`, etc.). The urn weights drift from integer to real. The math still goes through; the conceptual model does not.

### Constraint 2 — Negative rewards introduce an absorbing barrier

When $r < 0$ and the clamp fires, $\mathrm{urn}[a]$ collapses to zero. Then $\mathbb{P}[a] = 0$ forever, which means action $a$ is never sampled again, which means $\mathrm{urn}[a]$ is never updated again — even if subsequent episodes would have $r > 0$.

This is structurally different from Q-learning (where $Q[a]$ can dip negative and exploration $\varepsilon > 0$ keeps re-sampling) and TD-learning (where the count-based learning rate $1/N(s,a)$ eventually dominates and pulls $Q$ toward $\mathbb{E}[r]$). For UrnAgent, an action killed by a streak of negative rewards stays dead.

Defensively, `get_signal` and `get_action` reset the *whole row* to ones if `total_sum <= 0` ([rl_signaling/agents.py:267-270](../../rl_signaling/agents.py#L267-L270)), so a complete row-collapse cannot occur. But individual actions hitting the absorbing barrier is reachable and not handled.

### Implication for costly signaling

Both constraints fire under costly signaling. With `create_random_canonical_game(n=1, m=0)` and per-agent cost $c_i \sim \mathrm{Uniform}(0, 0.5)$, the per-episode net reward is one of:

| Game outcome | Signal | Net reward |
|---|---|---|
| Optimal action | non-null | $1 - c_i \in (0.5, 1)$ — real, positive |
| Optimal action | null | $1$ — integer |
| Sub-optimal action | non-null | $0 - c_i \in (-0.5, 0)$ — real, **negative** |
| Sub-optimal action | null | $0$ — integer |

So roughly $\tfrac{3}{4}$ of episodes (sub-optimal actions are 3/4 of the action space at random) yield negative rewards under non-null signals, hitting the clamp. The dynamics that result are not the canonical Roth-Erev model — they are a degenerate variant with real-valued urns and absorbing barriers.

**Recommendation.** The costly-signaling extension should not be applied to UrnAgent in formal experiments. Q-learning and TD-learning handle real-valued and signed rewards natively (the TD update is a stochastic-approximation rule on $\mathbb{R}$, not a probability-counting scheme). If costly UrnAgent results must be reported, label them explicitly as "Roth-Erev with non-negativity-clamped costly extension" so the deviation from the canonical rule is visible to readers.

This concern is recorded as an additional note in [Error 5a of LEGACY_ERRORS_LOG.md](../../docs/code-audit/LEGACY_ERRORS_LOG.md#error-5a--roth-erev-costly-figures-unreproducible-cost-protocol-drift).

---

## Mixed-reward and rare-event behavior

In the costly-signaling setting, the per-episode reward can be negative (when $G_i(\mathbf{v}, \alpha_i) < c_i$ for a non-null signal). The clamped update means a single large negative episode reduces the urn weight on the responsible action, but never below zero. The sampling probability for that action falls accordingly; after enough negative episodes it approaches zero.

If the urn has weights $\mathbf{u} = (u_0, \dots, u_{K-1})$ and the chosen action $a^\star$ receives $r < 0$ at the next pull:

$$u_{a^\star}^{\text{new}} = \max(0, u_{a^\star} + r),$$

while other entries are unchanged. The new sampling distribution shifts mass *away* from $a^\star$ toward the unchanged actions:

$$\mathbb{P}^{\text{new}}[a^\star] = \frac{\max(0, u_{a^\star} + r)}{\max(0, u_{a^\star} + r) + \sum_{a \neq a^\star} u_a}.$$

This is monotone non-increasing in $|r|$ for negative $r$, hits zero precisely when $u_{a^\star} + r \le 0$, and stays at zero on subsequent negative-reward episodes that re-select $a^\star$ (which they cannot, since $\mathbb{P}[a^\star] = 0$). The probability of re-sampling $a^\star$ after it hits zero requires *some* other action to be punished into a tie, after which the defensive uniform reset may fire. In practice, the urn dynamics are well-behaved on a single state and the clamp acts as a soft stopping rule.

## Summary table — urn data and operations

| Operation | Code | Effect |
|---|---|---|
| Construct | [agents.py:202-238](../../rl_signaling/agents.py#L202-L238) | Initializes urns: lazy (`{}`) or eager (one-hot via `create_initial_signals`) |
| `get_signal(state)` | [agents.py:245-273](../../rl_signaling/agents.py#L245-L273) | Lazy-init if missing; sample $\sigma \sim \text{urn}/\text{sum}(\text{urn})$ |
| `get_action(state)` | [agents.py:275-302](../../rl_signaling/agents.py#L275-L302) | Same shape as `get_signal` for $\alpha\text{-urn}$ |
| `update_signals(state, sig, r)` | [agents.py:304-308](../../rl_signaling/agents.py#L304-L308) | $\sigma\text{-urn}[\text{state}][\sigma] \leftarrow \max(0, \sigma\text{-urn}[\text{state}][\sigma] + r)$ |
| `update_actions(state, act, r)` | [agents.py:310-314](../../rl_signaling/agents.py#L310-L314) | $\alpha\text{-urn}[\text{state}][\alpha] \leftarrow \max(0, \alpha\text{-urn}[\text{state}][\alpha] + r)$ |
| `update_episode(...)` | [agents.py:316-327](../../rl_signaling/agents.py#L316-L327) | Calls `update_signals` then `update_actions` (skipping signals if `signal is None`) |

## Cross-references

| Concept | Code | Spec axis | Test |
|---|---|---|---|
| Sampling proportional to urn | [agents.py:272-273](../../rl_signaling/agents.py#L272-L273) | Axis 18 | [test_agents.py::test_urn_agent_get_signal_in_range](../../tests/test_agents.py#L71-L75) |
| Clamp at zero | [agents.py:306, 312](../../rl_signaling/agents.py#L306) | Axis 18 | [test_agents.py::test_urn_agent_update_clamps_at_zero](../../tests/test_agents.py#L78-L82) |
| Eager initialization | [agents.py:223-235](../../rl_signaling/agents.py#L223-L235) | (constructor) | [test_agents.py::test_urn_agent_initialize_true_seeds_action_urns](../../tests/test_agents.py#L85-L104) |
| Convergence on stationary state | this file, "Closed form" | (asymptotic) | [test_numerical_sanity.py::test_urn_agent_converges_to_optimal_action_in_full_information](../../tests/test_numerical_sanity.py#L186-L228) |

## Independent verification

The script [scripts/verify_urn_convergence.py](scripts/verify_urn_convergence.py) constructs a single-state UrnAgent, runs $n = 1000$ updates with reward 1 on a fixed action, and compares the empirical sampling probability of that action against the closed-form $(1 + n r / u_0) / (K + n r / u_0)$.
