# Q-learning agent

- status: active
- type: explanation
- id: rl_signaling.analytics.agent_q_learning
- description: Mathematical description of QLearningAgent — single-step temporal-difference learning over two separate Q-tables (signaling, action), with constant learning rate alpha=0.1, no bootstrap (terminal episodes), and per-channel exploration decay. Includes the closed form Q_n = r·(1 - (1-alpha)^n) and the optional exponential-smoothing variant.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

The `QLearningAgent` in [rl_signaling/agents.py:330-495](../../rl_signaling/agents.py#L330-L495) implements a temporal-difference learner specialized to the project's single-step signaling game. This file walks through the data structures, the simplified Bellman update that arises from the terminal-episode assumption, the closed form for constant-reward learning, the exponential-smoothing variant, and the per-channel exploration-decay schedule.

The exploration kernel itself (ε-greedy / softmax / UCB) is shared with the TD agent and is documented separately in [exploration_strategies.md](exploration_strategies.md).

## State of the agent

`QLearningAgent` keeps **two** Q-tables — one for signaling, one for actions:

$$Q_{\text{sig}} : \mathcal{V}_i \to \mathbb{R}^K, \qquad Q_{\text{act}} : (\mathcal{V}_i \times \mathcal{A}_{\text{sig}}^{\le |\mathcal{N}_i|}) \to \mathbb{R}^M.$$

In code these are `q_table_signaling` and `q_table_action` ([rl_signaling/agents.py:394-408](../../rl_signaling/agents.py#L394-L408)). The keys are tuples (the direct observation $\mathbf{o}$ for signaling, the post-signal observation $\tilde{\mathbf{o}}$ for actions). New keys are lazy-initialized to the all-zeros vector:

$$Q_{\text{sig}}[\mathbf{o}] := \mathbf{0} \in \mathbb{R}^K \qquad \text{when } \mathbf{o} \text{ is first seen}.$$

Two parallel **count tables** $N_{\text{sig}}, N_{\text{act}}$ track per-action visit counts at each state. The counts are used by the UCB exploration strategy and are inert for ε-greedy / softmax (they are still incremented, but not read).

## Single-step temporal-difference assumption

The standard Q-learning update is

$$Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \big[\, r + \gamma \max_{a'} Q(s', a') \;-\; Q(s, a)\,\big].$$

In the project's signaling game (Phase 1 [Axis 19](../../docs/code-audit/DEBUGGING_PLAN.md#agent-learning-rules)), every episode is **terminal** — there is no next state $s'$ for the agent to bootstrap from. Mathematically, the bootstrap term $\gamma \max_{a'} Q(s', a')$ collapses to zero by the Bellman equation's terminal-state convention. So the update reduces to:

$$\boxed{\; Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \big[\, r \;-\; Q(s, a)\,\big]. \;}$$

This is a constant-step-size **stochastic approximation** to the expected reward $\mathbb{E}[r \mid s, a]$. It does not converge to the discounted optimal Q-value of a multi-step MDP because there is no multi-step structure.

## Constant learning rate $\alpha$

The learning rate is hardcoded:

$$\alpha = 0.1$$

at [rl_signaling/agents.py:458](../../rl_signaling/agents.py#L458) and [:476](../../rl_signaling/agents.py#L476). The Phase 1 spec [Axis 19](../../docs/code-audit/DEBUGGING_PLAN.md#agent-learning-rules) confirms this is intentional. A constant $\alpha$ keeps the update responsive to changes in the environment (in our case, changes in the *partner's* signaling policy) at the cost of asymptotic noise — Q never settles to a single value, it tracks a moving target with ~10% step size.

The alternative, decaying learning rate $\alpha_n = 1/n$, would satisfy the Robbins–Monro convergence condition. The TD agent uses it. The Q-agent does not, and that is a deliberate design choice.

## Update: derivation and closed form

Take a fixed $(s, a)$ pair, $\alpha = 0.1$, $Q_0 := Q^{(0)}(s, a)$, and assume the same reward $r$ is received every time the pair is updated. Let $Q_n$ denote the value after $n$ updates.

The recursion is

$$Q_{n+1} = Q_n + \alpha (r - Q_n) = (1 - \alpha) Q_n + \alpha r.$$

This is a first-order linear recursion. The fixed point $Q^\star$ satisfies $Q^\star = (1 - \alpha) Q^\star + \alpha r$, i.e. $Q^\star = r$. Defining the offset $\Delta_n := Q_n - r$:

$$\Delta_{n+1} = (1 - \alpha) \Delta_n.$$

This is geometric with ratio $1 - \alpha$, so

$$\Delta_n = (1 - \alpha)^n \Delta_0,$$

and

$$\boxed{\; Q_n \;=\; r + (Q_0 - r)(1 - \alpha)^n. \;}$$

For the canonical starting point $Q_0 = 0$ this simplifies to

$$\boxed{\; Q_n \;=\; r \cdot \big[1 - (1 - \alpha)^n\big]. \;}$$

### Numerical example

With $\alpha = 0.1$, $r = 1$, $Q_0 = 0$:

| $n$ | $Q_n = 1 - 0.9^n$ |
|---|---|
| 1 | $0.1$ |
| 2 | $0.19$ |
| 5 | $0.40951$ |
| 10 | $0.6513215599$ |
| 20 | $0.8784233454$ |
| 50 | $0.9948462732$ |
| 100 | $0.9999734386$ |

These values are exact rational fractions of $0.9^n$. The asymptote $Q_n \to 1$ happens at rate $0.9^n = e^{n \ln 0.9} \approx e^{-0.1054 n}$, so the half-life is $\ln(2) / 0.1054 \approx 6.58$ episodes.

The $n = 1$ and $n = 10$ values are tested in [tests/test_numerical_sanity.py::test_q_learning_single_update_is_exact_alpha_times_reward](../../tests/test_numerical_sanity.py#L71-L78) and [::test_q_learning_ten_updates_match_geometric_closed_form](../../tests/test_numerical_sanity.py#L81-L93).

## Implementation

The relevant block at [rl_signaling/agents.py:447-464](../../rl_signaling/agents.py#L447-L464):

```python
def update_signals(self, state, signal, reward):
    if self.exp_smoothing:
        alpha = 0.1
        self.q_table_signaling[state][signal] = (
            (1 - alpha) * self.q_table_signaling[state][signal] + alpha * reward
        )
    else:
        td_target = reward
        td_error = td_target - self.q_table_signaling[state][signal]
        learning_rate = 0.1
        self.q_table_signaling[state][signal] += learning_rate * td_error
    self.signal_exploration_rate = max(
        self.min_exploration_rate,
        self.signal_exploration_rate * self.exploration_decay,
    )
```

Two notes:

1. The TD-update branch (`else`) and the exponential-smoothing branch (`if self.exp_smoothing`) are **algebraically identical** — both are $Q \leftarrow (1-\alpha) Q + \alpha r$. The two branches differ only in how the formula is written, not in what it computes. (Algebraic check: $(1-\alpha) Q + \alpha r = Q + \alpha(r - Q)$ — same value, different form.) So the `exp_smoothing` parameter is a no-op in the canonical model; it would only matter if some other update were spliced in.
2. The exploration decay is applied **per call** to `update_signals`. The signal-channel rate `signal_exploration_rate` is independent of the action-channel rate `action_exploration_rate`. Each is decayed once per episode (since `update_episode` calls each of `update_signals` and `update_actions` exactly once per episode, modulo the `signal is None` no-signals branch).

## Per-channel exploration decay

Two **separate** exploration rates are tracked:

$$\varepsilon_{\text{sig}}^{(t)}, \quad \varepsilon_{\text{act}}^{(t)} \quad \text{(at episode } t\text{)}.$$

Both start at the same value `exploration_rate` (default 1.0) and decay multiplicatively after each per-channel update:

$$\varepsilon_{\text{sig}}^{(t+1)} = \max\big( \varepsilon_{\min}, \; \rho \cdot \varepsilon_{\text{sig}}^{(t)} \big),$$

$$\varepsilon_{\text{act}}^{(t+1)} = \max\big( \varepsilon_{\min}, \; \rho \cdot \varepsilon_{\text{act}}^{(t)} \big),$$

where $\rho$ is the `exploration_decay` parameter (default 0.995) and $\varepsilon_{\min}$ is `min_exploration_rate` (default 0.001).

After $t$ updates:

$$\varepsilon^{(t)} = \max\big( \varepsilon_{\min}, \; \rho^t \cdot \varepsilon^{(0)} \big).$$

With the defaults, $\varepsilon$ reaches the floor $\varepsilon_{\min} = 0.001$ at $t = \log(\varepsilon_{\min} / \varepsilon_0) / \log \rho \approx \log(1/1000) / \log 0.995 \approx 1379$. So in a 10000-episode run, exploration is at the floor for $\sim 8600$ of the episodes — most of the run is near-greedy.

The signal- and action-channel rates are decayed only when their respective updates fire. In particular, when `with_signals=False`, `update_signals` is never called and `signal_exploration_rate` stays at its initial value forever. This is fine in practice because nothing reads it in that regime.

## How exploration interacts with action selection

`get_signal(state)` reads the **current** `signal_exploration_rate` and dispatches to `_select_action(q_values, counts, exploration_rate, choice)`. The exploration kernel chooses an action under the named strategy (ε-greedy / softmax / UCB) using the rate as the strategy's free parameter. See [exploration_strategies.md](exploration_strategies.md) for the strategy formulas.

After `get_signal` returns, the visit count for the chosen signal is incremented at [agents.py:429](../../rl_signaling/agents.py#L429). The count is read by UCB to compute the exploration bonus.

The decay schedule then fires at the **next** `update_signals` call. So the order is:

```
get_signal(state) → counts[state][signal] += 1; sample using ε
update_signals(state, signal, reward) → Q ← Q + α(r − Q); ε ← ρε
```

Per episode, $(\varepsilon, Q)$ are read by `get_signal` and updated (decayed and shifted, respectively) by `update_signals`.

## Variance of the asymptotic Q-estimate

Because $\alpha$ is constant, $Q_n$ does not converge to a deterministic limit even when the rewards are i.i.d. Suppose $r_n \stackrel{\text{i.i.d.}}{\sim} \mathcal{D}$ with mean $\mu$ and variance $\sigma^2$. Iterating the update:

$$Q_n = (1-\alpha) Q_{n-1} + \alpha r_n = \alpha \sum_{k=1}^{n} (1-\alpha)^{n-k} r_k + (1-\alpha)^n Q_0.$$

Taking the limit $n \to \infty$ (the $Q_0$ term decays geometrically),

$$\mathbb{E}[Q_\infty] = \alpha \sum_{k=0}^{\infty} (1-\alpha)^k \mu = \alpha \cdot \frac{\mu}{\alpha} = \mu.$$

For the second moment,

$$\text{Var}(Q_\infty) = \alpha^2 \sum_{k=0}^{\infty} (1-\alpha)^{2k} \sigma^2 = \frac{\alpha^2}{1 - (1-\alpha)^2} \sigma^2 = \frac{\alpha}{2-\alpha} \sigma^2.$$

For $\alpha = 0.1$:

$$\text{Var}(Q_\infty) \approx 0.0526 \sigma^2.$$

So the asymptotic standard deviation of $Q_n$ around $\mu$ is $\sqrt{0.0526} \sigma \approx 0.229 \sigma$. The agent's Q-estimate **always** wobbles by ~23% of the reward standard deviation, even after infinite training. This is the price of constant $\alpha$ — and the gain is the ability to track a non-stationary reward distribution (e.g. the partner's evolving signaling policy).

## Summary table

| Operation | Code | Effect |
|---|---|---|
| Construct | [agents.py:371-408](../../rl_signaling/agents.py#L371-L408) | Initializes Q-tables, count tables, per-channel exploration rates |
| `get_signal(state)` | [agents.py:417-430](../../rl_signaling/agents.py#L417-L430) | Lazy-init; sample via `_select_action`; increment counts |
| `get_action(state)` | [agents.py:432-445](../../rl_signaling/agents.py#L432-L445) | Same shape as `get_signal` for $Q_{\text{act}}$ |
| `update_signals(state, sig, r)` | [agents.py:447-464](../../rl_signaling/agents.py#L447-L464) | $Q_{\text{sig}}[\text{state}][\sigma] \leftarrow Q_{\text{sig}}[\text{state}][\sigma] + 0.1 (r - Q_{\text{sig}}[\text{state}][\sigma])$; decay $\varepsilon_{\text{sig}}$ |
| `update_actions(state, act, r)` | [agents.py:466-482](../../rl_signaling/agents.py#L466-L482) | Same shape for $Q_{\text{act}}$, $\varepsilon_{\text{act}}$ |
| `update_episode(...)` | [agents.py:484-495](../../rl_signaling/agents.py#L484-L495) | Calls `update_signals` then `update_actions` (skipping signals if `signal is None`) |

## Cross-references

| Concept | Code | Spec axis | Test |
|---|---|---|---|
| Single-step terminal target | [agents.py:455](../../rl_signaling/agents.py#L455) | Axis 19 (no bootstrap) | [test_numerical_sanity.py::test_q_learning_single_update_is_exact_alpha_times_reward](../../tests/test_numerical_sanity.py#L71-L78) |
| Constant $\alpha = 0.1$ | [agents.py:458, 476](../../rl_signaling/agents.py#L458) | Axis 19 | (covered by both numerical_sanity tests) |
| Closed form $Q_n = r(1 - (1-\alpha)^n)$ | this file, "Update: derivation" | (analytical) | [test_numerical_sanity.py::test_q_learning_ten_updates_match_geometric_closed_form](../../tests/test_numerical_sanity.py#L81-L93) |
| Per-channel decay | [agents.py:461-464, 479-482](../../rl_signaling/agents.py#L461-L464) | Axis 19 | [test_agents.py::test_q_learning_exploration_decays_after_update](../../tests/test_agents.py#L118-L129) |

## Independent verification

The script [scripts/verify_q_learning.py](scripts/verify_q_learning.py) drives a `QLearningAgent` through 100 identical-reward updates, computes $Q_n$ from the closed form for each $n$, and asserts agreement at every step. It also computes the asymptotic variance from the formula above and compares it to the empirical variance over a long run.
