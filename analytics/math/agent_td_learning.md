# TD-learning agent

- status: active
- type: explanation
- id: rl_signaling.analytics.agent_td_learning
- description: Mathematical description of TDLearningAgent — temporal-difference learning over a single shared Q-table for both signaling and action phases, with count-based learning rate 1/N(s,a), one-step bootstrap from next_state, and two TD updates per episode (signal-phase bootstrap, action-phase terminal). Includes derivation of equivalence to QLearningAgent under gamma=1 in the costly setting.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

The `TDLearningAgent` in [rl_signaling/agents.py:498-672](../../rl_signaling/agents.py#L498-L672) is the only agent in the project that uses **bootstrapping** — its update at the signal phase reads the action-phase Q-values to compute a discounted future return. Unlike `QLearningAgent` it has a single Q-table shared across both phases, and a count-based learning rate $1/N(s,a)$ that satisfies the Robbins–Monro condition.

This file derives the bootstrap formula, walks through the two-phase update, derives the equivalence between the canonical and legacy update orderings, and connects the design to the classical RL convergence theory.

## State of the agent

A single Q-table:

$$Q : \mathcal{S} \to \mathbb{R}^{n_{\text{actions}}}, \qquad n_{\text{actions}} = \max(K, M),$$

where $\mathcal{S}$ ranges over both **signal-phase observations** ($\mathbf{o}_i \in \mathcal{V}_i$) and **action-phase observations** ($\tilde{\mathbf{o}}_i$). Concretely the same Python dict `q_table` holds both kinds of keys:

$$Q[\mathbf{o}] \in \mathbb{R}^{n_{\text{actions}}} \quad \text{for signal-phase rows},$$

$$Q[\tilde{\mathbf{o}}] \in \mathbb{R}^{n_{\text{actions}}} \quad \text{for action-phase rows}.$$

Phase 1 Axis 20 confirms this is intentional. The two phases never collide because $\mathbf{o}$ and $\tilde{\mathbf{o}}$ have **different tuple lengths** — $\tilde{\mathbf{o}}$ appends 0..$|\mathcal{N}_i|$ extra tokens to $\mathbf{o}$ — so they hash to different dict slots.

A parallel **count table** $N(s, a)$ counts how many times action $a$ has been selected from state $s$:

$$N : \mathcal{S} \to \mathbb{N}^{n_{\text{actions}}}, \qquad N[s][a] := \#\{ t : (s_t, a_t) = (s, a) \}.$$

Implementation: `self.q_table` and `self.action_counts` at [rl_signaling/agents.py:565-566](../../rl_signaling/agents.py#L565-L566). Both are lazy-initialized on first access.

The **action subset** for the two phases is encoded by `n_signaling_actions` and `n_final_actions`:

- `get_signal(state)` calls `_select_action(..., available_actions=range(n_signaling_actions))`.
- `get_action(state)` calls `_select_action(..., available_actions=range(n_final_actions))` by default.

So the same Q-row is read with two different action masks depending on the phase.

## The TD update

The standard TD(0) update for an episodic MDP is

$$Q(s, a) \;\leftarrow\; Q(s, a) + \alpha_n \big[\, r + \gamma \max_{a'} Q(s', a') \cdot \mathbb{1}[\neg \text{done}] \;-\; Q(s, a) \,\big]$$

where $\alpha_n$ is the learning rate at step $n$, $\gamma \in [0, 1]$ is the discount factor, $r$ is the immediate reward, and the bootstrap term is dropped on the terminal step (`done=True`).

The project's `TDLearningAgent` implements this with two specializations:

1. **Count-based learning rate.** $\alpha_n = 1 / N(s, a)$.
2. **Default discount $\gamma = 1$.** No discounting in the canonical experiments.

So the update is:

$$\boxed{\; Q(s, a) \;\leftarrow\; Q(s, a) + \frac{1}{N(s, a)} \big[\, r + \gamma \max_{a'} Q(s', a') \cdot \mathbb{1}[\neg \text{done}] \;-\; Q(s, a) \,\big]. \;}$$

Implementation:

```python
td_target = reward
if not done:
    td_target += self.gamma * np.max(self.q_table[next_state])
td_error = td_target - self.q_table[state][action]
self.q_table[state][action] += td_error / self.action_counts[state][action]
```

at [rl_signaling/agents.py:624-634](../../rl_signaling/agents.py#L624-L634).

The denominator `self.action_counts[state][action]` is **at least 1** when the update is called, because `get_action` (which incremented the count) has already run and the count is read after that increment.

## Why count-based $\alpha$ works

The Robbins–Monro convergence theorem states that a stochastic-approximation iterate

$$x_{n+1} = x_n + \alpha_n (T(x_n) - x_n + \xi_n)$$

with mean-zero noise $\xi_n$ converges to the fixed point of $T$ provided

$$\sum_{n=1}^{\infty} \alpha_n = \infty \quad \text{and} \quad \sum_{n=1}^{\infty} \alpha_n^2 < \infty.$$

The choice $\alpha_n = 1/n$ satisfies both: $\sum 1/n = \infty$ (harmonic series), $\sum 1/n^2 < \infty$ (Basel-style sum). So the count-based rule is the canonical Robbins–Monro choice for stochastic averaging.

In the Q-learning setting, the iterate at a fixed $(s, a)$ pair is:

$$Q_{N+1}(s, a) = Q_N(s, a) + \frac{1}{N+1} \big[ r_N + \gamma \max_{a'} Q_N(s', a') - Q_N(s, a) \big].$$

If $r_N$ and $s'$ are i.i.d. given $(s, a)$ (which holds for stationary policies), this is exactly the empirical Bellman backup. The classical Watkins–Dayan convergence proof (1992) shows it converges to $Q^\star(s, a)$ provided every $(s, a)$ is visited infinitely often.

The "infinitely often" condition is delivered by the project's exploration schedule: with `min_exploration_rate > 0`, every action retains a strictly positive probability forever (under ε-greedy or softmax). So the formal convergence theorem applies — though the asymptotic noise is Robbins-Monro slow ($\sigma^2/n$) versus the Q-learning agent's constant-step-size noise ($\alpha \sigma^2 / (2 - \alpha) \approx 0.05 \sigma^2$).

The trade-off: TD agent has stronger asymptotic guarantees but slower late-stage tracking of a non-stationary partner.

## Two-phase update

The **canonical** API exposes a single `update_episode(signal_state, signal, action_state, action, reward)` call per episode. Internally for `TDLearningAgent` this fires **two** calls to `update`:

```python
def update_episode(self, signal_state, signal, action_state, action, reward):
    if signal is not None:
        self.update(state=signal_state, action=signal,
                    reward=0, next_state=action_state, done=False)
    self.update(state=action_state, action=action,
                reward=reward, next_state=action_state, done=True)
```

at [rl_signaling/agents.py:641-672](../../rl_signaling/agents.py#L641-L672).

Decomposing the two calls:

### Signal-phase call

Inputs: `state=signal_state` ($\mathbf{o}$), `action=signal` ($\sigma$), `reward=0`, `next_state=action_state` ($\tilde{\mathbf{o}}$), `done=False`.

Update:

$$Q(\mathbf{o}, \sigma) \leftarrow Q(\mathbf{o}, \sigma) + \frac{1}{N(\mathbf{o}, \sigma)} \big[ 0 + \gamma \max_{a'} Q(\tilde{\mathbf{o}}, a') - Q(\mathbf{o}, \sigma) \big].$$

The signal phase **bootstraps** from the action-phase Q-row. The reward at this phase is hardcoded to 0 because the reward signal is delivered only at the terminal action phase — the per-episode reward $r$ from the env covers both phases.

### Action-phase call

Inputs: `state=action_state` ($\tilde{\mathbf{o}}$), `action=action` ($\alpha$), `reward=reward`, `next_state=action_state` (unused; bootstrap dropped), `done=True`.

Update:

$$Q(\tilde{\mathbf{o}}, \alpha) \leftarrow Q(\tilde{\mathbf{o}}, \alpha) + \frac{1}{N(\tilde{\mathbf{o}}, \alpha)} \big[ r + 0 - Q(\tilde{\mathbf{o}}, \alpha) \big].$$

The action phase is **terminal** — the bootstrap term is gated to zero by `done=True`. So the update is plain Robbins–Monro averaging of $r$:

$$Q(\tilde{\mathbf{o}}, \alpha) \to \mathbb{E}[r \mid \tilde{\mathbf{o}}, \alpha] \quad \text{as } N \to \infty.$$

## Why the signal-phase reward is zero (and why this is right)

A natural question: shouldn't the signal-phase update see *the cost* of signaling, not zero?

The answer: the cost is bundled into the per-episode reward $r$ at the env level (see [costly_signaling.md](costly_signaling.md)), and that net reward $r$ is delivered to the action phase. With $\gamma = 1$ and $\text{done}=\text{True}$ at the action phase:

$$\mathbb{E}\big[ \gamma \max_{a'} Q(\tilde{\mathbf{o}}, a') \big] \;\to\; \mathbb{E}[r \mid \tilde{\mathbf{o}}, \alpha^\star] = \mathbb{E}[G_i(\mathbf{v}, \alpha^\star) - c_i \cdot \mathbb{1}[\sigma \neq \nu] \mid \tilde{\mathbf{o}}],$$

where $\alpha^\star$ is the greedy action at $\tilde{\mathbf{o}}$. So the bootstrap target at the signal phase indirectly carries the cost via the action-phase Q-values — the cost is **already in $r$**, propagated to the signal phase by the bootstrap.

To see the equivalence concretely with $\gamma = 1$: define a "cost-attributed signal phase" alternative where the signal-phase reward is $-c_i \cdot \mathbb{1}[\sigma \neq \nu]$ and the action-phase reward is $G_i(\mathbf{v}, \alpha)$. Total return per episode (signal-phase reward plus action-phase reward, undiscounted) is the same:

$$\big[-c_i \mathbb{1}[\sigma \neq \nu]\big] + \big[G_i(\mathbf{v}, \alpha)\big] = G_i(\mathbf{v}, \alpha) - c_i \mathbb{1}[\sigma \neq \nu].$$

Under $\gamma = 1$ TD(0) only the *total* return matters — the per-step decomposition is invariant to how the total is sliced. So $r_{\text{signal}} = 0$ + $r_{\text{action}} = r$ produces the same Q-values asymptotically as $r_{\text{signal}} = -c$ + $r_{\text{action}} = G$. The chosen split is the cleaner one: it matches the env's API (single per-episode reward) and avoids duplicating the cost-tracking logic.

For $\gamma < 1$ the equivalence breaks. The current code uses $\gamma = 1$ default ([rl_signaling/agents.py:540](../../rl_signaling/agents.py#L540)) so this is fine; the parameter optimization notebook explores $\gamma < 1$ and should be aware of the asymmetry it introduces.

## Order of updates within an episode (canonical vs legacy)

The two-phase update can be sequenced in two ways:

**Canonical (`MultiAgentEnv` + `run_simulation`).** Both updates fire **at end of episode**, inside `update_episode`. The signal-phase update sees the **latest** action-state Q-values and the **current** exploration rate. The exploration rate is decayed twice (once per `update` call) at end of episode.

**Legacy (`TempNetMultiAgentEnv` + `temp_simulation_function`).** The two updates are **interleaved** with the action-phase `get_action` call:

```
get_signal → update_signal → get_action → update_action
```

So the signal-phase update fires *before* `get_action` runs; that means `get_action` sees an exploration rate that has already been decayed once.

The two orderings produce slightly different Q-trajectories. The README's [Status and known limitations](../../README.md#status-and-known-limitations) section documents this: "the action-phase `get_action` sees a slightly higher `exploration_rate` in the new flow, which causes roughly 1 in 100 episodes to take a different explore/exploit branch." The Q-value math itself (the formulas above) is identical between the two flows; only the ε-decay sequencing differs.

The golden-run baseline ([tests/golden/baseline.json](../../tests/golden/baseline.json)) is captured against the canonical flow.

## Bootstrap with $\gamma = 1$, reward = 0, $\max Q(\text{next}) = 1$

Test case [test_td_one_step_bootstrap_with_unit_count](../../tests/test_numerical_sanity.py#L101-L121):

Setup: $Q[(0,)][0] = 0$, $Q[(1,)] = (1, 0, 0, 0)$, $N[(0,)][0] = 1$, $\gamma = 1$. Call `update(state=(0,), action=0, reward=0, next_state=(1,), done=False)`.

Trace:

$$\text{td\_target} = 0 + 1 \cdot \max(1, 0, 0, 0) = 1,$$

$$\text{td\_error} = 1 - 0 = 1,$$

$$Q[(0,)][0] \leftarrow 0 + \frac{1}{1} \cdot 1 = 1.$$

So $Q[(0,)][0]$ jumps from 0 to 1 in a single step. This is the bootstrap propagating future value backward.

## Terminal update: $\gamma = 1$, reward = 1, done = True

Test case [test_td_one_step_terminal_no_bootstrap](../../tests/test_numerical_sanity.py#L124-L134):

Setup: $Q[(0,)][0] = 0$, $N[(0,)][0] = 1$. Call `update(state=(0,), action=0, reward=1, next_state=(0,), done=True)`.

Trace:

$$\text{td\_target} = 1 + 0 \quad \text{(bootstrap dropped because done)} = 1,$$

$$\text{td\_error} = 1,$$

$$Q[(0,)][0] \leftarrow 0 + 1 \cdot 1 = 1.$$

Same numerical result as the bootstrap case, but for a different reason: the bootstrap is *gated*, not *evaluated to zero*.

## Action-state-as-self bootstrap edge case

The action-phase update has `next_state=action_state` (same as `state`) — the next state field is unused because `done=True`, but the code still passes a value. Why?

Because the lazy-init guard at [agents.py:620-622](../../rl_signaling/agents.py#L620-L622) reads:

```python
if next_state not in self.q_table:
    self.q_table[next_state] = np.zeros(self.n_actions)
    self.action_counts[next_state] = np.zeros(self.n_actions)
```

so passing a valid (existing) state avoids spuriously creating a new key. Passing `next_state=state` is the cheapest way to ensure the key is already present.

## Convergence rate

For $(s, a)$ visited at episodes $t_1 < t_2 < \dots$ with i.i.d. rewards $r_{t_k} \sim \mathcal{D}$ of mean $\mu$ and variance $\sigma^2$, terminal updates only ($\gamma = 0$ effectively, or `done=True` always):

$$Q_{t_K}(s, a) = \frac{1}{K} \sum_{k=1}^{K} r_{t_k}.$$

This is the empirical mean. By the central limit theorem,

$$\sqrt{K} \cdot \big( Q_{t_K} - \mu \big) \;\xrightarrow{d}\; \mathcal{N}(0, \sigma^2).$$

So the Q-estimate is $\sqrt{K}$-consistent — its error shrinks like $\sigma / \sqrt{K}$ where $K$ is the number of times $(s, a)$ has been visited. After 1000 visits, $|Q - \mu| \approx 0.032 \sigma$ — much tighter than QLearningAgent's $0.23 \sigma$ asymptotic noise.

For bootstrap updates the rate is the same in the limit (Watkins–Dayan), but the constant depends on the magnitude of the bootstrap target and the visit-count distribution across $(s, a)$ pairs. Practical convergence is slower than the no-bootstrap case in absolute episode count.

## Summary table

| Operation | Code | Effect |
|---|---|---|
| Construct | [agents.py:533-566](../../rl_signaling/agents.py#L533-L566) | Initializes Q-table and counts (lazy); both legacy `n_actions` and canonical `n_signaling_actions / n_final_actions` constructor forms supported |
| `get_signal(state)` | [agents.py:568-570](../../rl_signaling/agents.py#L568-L570) | Calls `get_action(state, available_actions=range(n_signaling_actions))` |
| `get_action(state, available_actions)` | [agents.py:572-603](../../rl_signaling/agents.py#L572-L603) | Lazy-init; `_select_action` over the action mask; increment counts |
| `update(state, action, reward, next_state, done)` | [agents.py:605-639](../../rl_signaling/agents.py#L605-L639) | TD-bootstrap update with $1/N$ learning rate; decay exploration rate |
| `update_episode(...)` | [agents.py:641-672](../../rl_signaling/agents.py#L641-L672) | Two `update` calls: signal-phase bootstrap + action-phase terminal |

## Cross-references

| Concept | Code | Spec axis | Test |
|---|---|---|---|
| Single shared Q-table | [agents.py:565](../../rl_signaling/agents.py#L565) | Axis 20 | (structural) |
| Bootstrap from `next_state` | [agents.py:624-626](../../rl_signaling/agents.py#L624-L626) | Axis 20 | [test_numerical_sanity.py::test_td_one_step_bootstrap_with_unit_count](../../tests/test_numerical_sanity.py#L101-L121) |
| Count-based $1/N(s,a)$ | [agents.py:634](../../rl_signaling/agents.py#L634) | Axis 20 | (covered by both td tests) |
| Terminal target | [agents.py:625](../../rl_signaling/agents.py#L625) (the `if not done`) | Axis 20 | [test_numerical_sanity.py::test_td_one_step_terminal_no_bootstrap](../../tests/test_numerical_sanity.py#L124-L134) |
| Two updates per episode | [agents.py:641-672](../../rl_signaling/agents.py#L641-L672) | Axis 20 | [test_agents.py::test_td_learning_update_episode_runs_two_updates](../../tests/test_agents.py#L169-L183) |

## Independent verification

The script [scripts/verify_td_learning.py](scripts/verify_td_learning.py) drives a `TDLearningAgent` through several scenarios — terminal updates with i.i.d. rewards (compares to empirical mean), one-step bootstrap (compares to closed-form), and full two-phase update (verifies both rows shift correctly).
