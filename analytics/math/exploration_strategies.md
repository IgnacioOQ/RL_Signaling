# Exploration strategies

- status: active
- type: explanation
- id: rl_signaling.analytics.exploration_strategies
- description: Mathematical description of the three exploration strategies (epsilon-greedy, softmax/Boltzmann, UCB) used by QLearningAgent and TDLearningAgent — formulas, derivations, regret bounds, and the project-specific epsilon-on-counts implementation detail for UCB.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

The three exploration strategies — ε-greedy, softmax (Boltzmann), and UCB — are factored into a single helper [_select_action](../../rl_signaling/agents.py#L38-L116) and shared by both `QLearningAgent` and `TDLearningAgent`. This file gives the formal action-selection distribution for each, derives the relevant regret/optimality properties, and walks through the project-specific UCB implementation choice (small-epsilon on counts to avoid div-by-zero on the first call).

The selector's signature is

```python
_select_action(q_values, counts, exploration_rate, choice, available_actions=None) -> int
```

returning the index of the selected action. `q_values` and `counts` are aligned numpy arrays; `exploration_rate` is the strategy-specific free parameter (ε for ε-greedy, temperature τ for softmax, weight $c$ for UCB); `choice` is one of `"egreedy"`, `"softmax"`, `"ucb"`.

## ε-greedy

### Formula

With probability $\varepsilon$ pick uniformly at random over the action set; otherwise pick the greedy (argmax) action. Formally, for action $a$:

$$\boxed{\; \mathbb{P}[\sigma = a \mid Q] \;=\; \frac{\varepsilon}{|\mathcal{A}|} \;+\; (1 - \varepsilon) \cdot \mathbb{1}\big[a = \arg\max_{a'} Q[a']\big]. \;}$$

Where $\mathcal{A}$ is the action set (or `available_actions` if specified). When two actions tie for the argmax, the implementation breaks the tie deterministically by index (numpy's `argmax` returns the smallest index among ties).

### Implementation

```python
if choice == "egreedy":
    if random.uniform(0, 1) < exploration_rate:
        if available_actions is None:
            return random.randint(0, n_actions - 1)
        return random.choice(actions)
    if available_actions is None:
        return int(np.argmax(q_values))
    return max(actions, key=lambda a: q_values[a])
```

at [rl_signaling/agents.py:83-90](../../rl_signaling/agents.py#L83-L90).

### Properties

1. **Always explores.** With $\varepsilon > 0$, every action has probability $\ge \varepsilon / |\mathcal{A}|$, so every action is sampled infinitely often in an infinite run. The Watkins–Dayan convergence theorem applies under the project's exploration schedule (with `min_exploration_rate > 0`).
2. **Tunable greediness.** $\varepsilon = 0$ is purely greedy (no exploration), $\varepsilon = 1$ is purely uniform.
3. **Regret.** Under stationary rewards, ε-greedy with constant $\varepsilon$ has linear cumulative regret: $\mathbb{E}[\text{regret}_t] = \Theta(\varepsilon t)$. With decaying $\varepsilon_t \to 0$, the regret can be sublinear, but the rate depends on the schedule.

### When to use

ε-greedy is the default cognitive baseline — simple, robust, fully exploring. It is the project's default for `TDLearningAgent` (`choice="egreedy"` at [agents.py:541](../../rl_signaling/agents.py#L541)).

## Softmax (Boltzmann)

### Formula

Action probabilities are proportional to the exponentiated Q-values divided by a temperature parameter $\tau$:

$$\boxed{\; \mathbb{P}[\sigma = a \mid Q] \;=\; \frac{\exp(Q[a] / \tau)}{\sum_{a' \in \mathcal{A}} \exp(Q[a'] / \tau)}. \;}$$

### Numerical stability

Naive evaluation of the formula can overflow when $Q[a] / \tau$ is large. The standard trick is to subtract the maximum:

$$\frac{\exp(Q[a] / \tau)}{\sum_{a'} \exp(Q[a'] / \tau)} = \frac{\exp\big( (Q[a] - Q^\star) / \tau \big)}{\sum_{a'} \exp\big( (Q[a'] - Q^\star) / \tau \big)}, \quad Q^\star := \max_{a'} Q[a'].$$

The numerator is bounded above by 1 (when $a$ is the max); the denominator is bounded below by 1 (since at least one term equals $\exp(0) = 1$). So both numerator and denominator are in $[\exp(-(\max - \min) / \tau), 1]$ — finite and well-conditioned.

Implementation:

```python
tau = max(exploration_rate, 1e-6)
if available_actions is None:
    stable_q = q_values - np.max(q_values)
    exp_q = np.exp(stable_q / tau)
    probabilities = exp_q / np.sum(exp_q)
    return int(np.random.choice(n_actions, p=probabilities))
```

at [rl_signaling/agents.py:92-98](../../rl_signaling/agents.py#L92-L98). The lower bound `tau = max(exploration_rate, 1e-6)` prevents division by zero when `exploration_rate` decays to a very small value.

### Properties

1. **Limit cases.**
    - As $\tau \to 0^+$: softmax converges to greedy. The maximum-Q action gets probability 1; ties split uniformly.
    - As $\tau \to \infty$: softmax converges to uniform. All actions get probability $1/|\mathcal{A}|$.
2. **Smoothness.** Unlike ε-greedy, softmax is smooth in $Q$ — small changes in Q values produce small changes in action probabilities. This is helpful when Q-values are estimated from noisy data; it prevents the agent from flipping between actions on infinitesimal Q-difference changes.
3. **No upper bound on probability gap.** Softmax with $\tau \approx Q$ scale gives a moderate gap between the best and second-best action; with $\tau \ll$ gap, the gap approaches 1; with $\tau \gg$ gap, all actions are near-uniform.

### When to use

Softmax is appropriate when actions have a meaningful magnitude difference in Q-values that should translate to graded preference (rather than sharp argmax). The project includes it as a hyperparameter alternative; it is selected when the user passes `choice="softmax"`.

## Upper Confidence Bound (UCB)

### Formula

UCB1 (Auer, Cesa-Bianchi, Fischer 2002) selects the action that maximizes the **optimistic** Q-value — the empirical Q plus an exploration bonus that depends on visit count:

$$\boxed{\; \sigma_t \;=\; \arg\max_{a \in \mathcal{A}} \Bigg[ Q[a] + c \cdot \sqrt{\frac{\ln(\text{total counts})}{N(a)}} \, \Bigg], \;}$$

where $c$ is the exploration weight and $N(a)$ is the number of times $a$ has been chosen in the past. The classical UCB1 uses $c = \sqrt{2}$; the project's parameterization treats $c$ as the `exploration_rate`.

### Why this works (regret bound)

UCB1 is the canonical proof of $O(\log t)$ cumulative regret for stochastic multi-armed bandits with bounded rewards. Sketch (Auer et al., Theorem 1):

- For each suboptimal arm $a$ with mean gap $\Delta_a := \mu^\star - \mu_a > 0$, the expected number of times $a$ is pulled in $t$ rounds is bounded by

$$\mathbb{E}[N_t(a)] \le \frac{8 \ln t}{\Delta_a^2} + 1 + \frac{\pi^2}{3}.$$

- So the total expected regret is

$$\mathbb{E}[R_t] \le \sum_{a : \Delta_a > 0} \Big( \frac{8 \ln t}{\Delta_a} + \big(1 + \tfrac{\pi^2}{3}\big) \Delta_a \Big) = O(\log t).$$

This is **logarithmic** in $t$, beating ε-greedy's linear regret. The intuition: the bonus $c \sqrt{\ln t / N(a)}$ is large for under-visited arms and shrinks as $N(a) \to \infty$, so the algorithm explores aggressively early and exploits when confident.

The bound assumes stationary rewards, single-state setting, and reward in $[0, 1]$. In the project's multi-state setting it is a heuristic — it is applied per state independently — but the empirical performance is good and the implementation is included as one of three options.

### Implementation: epsilon-on-counts

The textbook formula has a problem on the first call: $N(a) = 0$ for all $a$, so $\ln t / N(a)$ is $\ln 0 / 0$, undefined.

The project's workaround at [rl_signaling/agents.py:105-114](../../rl_signaling/agents.py#L105-L114):

```python
if choice == "ucb":
    safe_counts = counts + 1e-5
    total_counts = np.sum(counts) + 1
    ucb_bonus = exploration_rate * np.sqrt(np.log(total_counts) / safe_counts)
    ucb_scores = q_values + ucb_bonus
    if available_actions is None:
        return int(np.argmax(ucb_scores))
    masked = np.full_like(ucb_scores, -np.inf)
    masked[actions] = ucb_scores[actions]
    return int(np.argmax(masked))
```

Two regularizers:

1. `safe_counts = counts + 1e-5` — adds a tiny epsilon to the visit counts. Prevents division by zero.
2. `total_counts = np.sum(counts) + 1` — adds 1 to the total. Prevents `log(0)`.

The standard UCB1 prescription handles the first-call issue differently: **play each arm once** before applying the formula. The project's epsilon-on-counts implementation skips this initialization phase and starts applying the formula immediately.

### First-step behavior

A subtle point. With all counts at zero on the very first call:

$$\text{safe\_counts} = (\varepsilon, \varepsilon, \dots, \varepsilon), \quad \text{total\_counts} = 1, \quad \ln(\text{total\_counts}) = 0.$$

So the bonus is

$$\text{bonus}(a) = c \sqrt{\frac{0}{\varepsilon}} = 0$$

for every action. The argmax over $Q + \text{bonus}$ degenerates to argmax over $Q$, and since $Q = 0$ initially, **every action is tied at zero** and `np.argmax` returns index 0.

So the first UCB call deterministically picks action 0. This is **not** uniform-at-random as one might intuit from "epsilon prevents div-by-zero." The bonus is wiped out by the $\ln 1 = 0$ in the numerator, and the tie-break defaults to index 0.

After action 0 is chosen and `counts[0]` increments to 1, the second call has

$$\text{safe\_counts} = (1 + \varepsilon, \varepsilon, \dots), \quad \text{total\_counts} = 2, \quad \ln(\text{total\_counts}) = 0.693,$$

so

$$\text{bonus}(0) = c \sqrt{\frac{0.693}{1 + \varepsilon}} \approx 0.832 c, \qquad \text{bonus}(j) = c \sqrt{\frac{0.693}{\varepsilon}} \approx 263 c \quad (j > 0).$$

The unvisited arms get a huge bonus, so the second call picks one of them. The third call picks the next, and so on — UCB ends up doing the "play each arm once" initialization implicitly, just with the order biased to "action 0 first, then any other in argmax order."

This is reflected in Axis 23 of the Phase 1 spec.

### Properties

1. **Logarithmic regret** in stochastic stationary bandits (Auer et al. 2002).
2. **Exploration tapers as $N(a)$ grows.** The bonus shrinks like $1/\sqrt{N(a)}$, so well-visited arms are exploited; rarely-visited arms continue to get explored.
3. **Counts grow without bound.** Unlike the decay-based exploration of ε-greedy / softmax, UCB has no minimum exploration rate parameter — its exploration is governed by counts and the weight $c$.

### When to use

UCB is the default for `QLearningAgent` (`choice="ucb"` at [agents.py:381](../../rl_signaling/agents.py#L381)) and is empirically the strongest performer in the project's hyperparameter optimization runs.

## Decay schedule (ε-greedy and softmax)

`QLearningAgent` and `TDLearningAgent` both apply multiplicative decay to the exploration rate:

$$\varepsilon^{(t+1)} = \max\big( \varepsilon_{\min},\, \rho \cdot \varepsilon^{(t)} \big),$$

with default $\rho = 0.995$ and $\varepsilon_{\min} = 0.001$ (and $\varepsilon^{(0)} = 1.0$). After $t$ updates,

$$\varepsilon^{(t)} = \max\big( \varepsilon_{\min},\; 0.995^t \big).$$

The floor is reached at

$$t^\star = \frac{\ln(\varepsilon_{\min} / \varepsilon^{(0)})}{\ln \rho} = \frac{\ln(0.001)}{\ln 0.995} \approx 1379.$$

So in a 10000-episode run, exploration reaches the floor after ~1379 episodes and stays there for the remaining ~8600.

For UCB the "exploration rate" is the weight $c$, not a probability, so the decay schedule has a different interpretation: it shrinks the bonus magnitude, which still forces eventual exploitation of high-Q actions over time.

The decay is applied **per channel** for `QLearningAgent` (separate rates for signaling and action) but **single-channel** for `TDLearningAgent` (one rate, decayed twice per episode by the two `update` calls).

## Cross-references

| Concept | Code | Spec axis | Test |
|---|---|---|---|
| ε-greedy formula | [agents.py:83-90](../../rl_signaling/agents.py#L83-L90) | (constructor) | [test_agents.py::test_select_action_egreedy_at_zero_exploration_is_greedy](../../tests/test_agents.py#L37-L42) |
| Softmax with stable subtraction | [agents.py:92-103](../../rl_signaling/agents.py#L92-L103) | (constructor) | (smoke-tested in `test_q_learning_get_signal_each_strategy`) |
| UCB with epsilon-on-counts | [agents.py:105-114](../../rl_signaling/agents.py#L105-L114) | Axis 23 | [test_agents.py::test_select_action_ucb_with_unit_counts](../../tests/test_agents.py#L45-L49) |
| Available-actions mask | [agents.py:79-81, 110-114](../../rl_signaling/agents.py#L79-L81) | (TD subset selection) | [test_agents.py::test_select_action_respects_available_actions](../../tests/test_agents.py#L52-L58) |
| Decay schedule | [agents.py:461-464, 479-482, 636-639](../../rl_signaling/agents.py#L461-L464) | Axis 19 (Q-learning, per-channel); Axis 20 (TD, single-channel) | [test_agents.py::test_q_learning_exploration_decays_after_update](../../tests/test_agents.py#L118-L129) |

## Independent verification

No dedicated script — the strategies are exercised by the agent-level scripts ([scripts/verify_q_learning.py](scripts/verify_q_learning.py), [scripts/verify_td_learning.py](scripts/verify_td_learning.py)) which run the agent end-to-end under each `choice` value and check that the action distribution converges to the expected limit (greedy under low ε, near-uniform under high ε / large τ).
