# Costly signaling

- status: active
- type: explanation
- id: rl_signaling.analytics.costly_signaling
- description: Mathematical treatment of the costly-signaling extension — null-signal augmentation of the alphabet, cost flow into the per-episode reward, and a brief equilibrium-theory bridge linking the project's setup to the classical signaling-game literature.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

The base signaling game in [signaling_model.md](signaling_model.md) assumes signals are free. The **costly** extension makes every non-null signal incur a per-agent cost; a special **null** signal lets an agent opt out of speaking at no charge. This file gives the formal definitions, derives the reward arithmetic exactly, and connects the setup to standard signaling-game equilibrium concepts.

The implementation is gated by the boolean `costly_signaling` on `MultiAgentEnv` and threaded through to [rl_signaling/env.py:236-241](../../rl_signaling/env.py#L236-L241).

## Augmenting the alphabet with null

When `costly_signaling=True`, the signal alphabet is extended by one symbol:

$$\tilde{\mathcal{A}}_{\text{sig}} := \mathcal{A}_{\text{sig}} \cup \{\nu\}, \qquad \nu := K,$$

where $K$ is the base alphabet size `n_signaling_actions`. The null symbol's index is the highest, so the augmented alphabet is $\{0, 1, \dots, K\}$ with $|\tilde{\mathcal{A}}_{\text{sig}}| = K + 1$.

Inside the env, the augmented size is what the agents see:

```python
self.n_signaling_actions = n_signaling_actions + 1 if costly_signaling else n_signaling_actions
self._null_signal_index   = self.n_signaling_actions - 1 if costly_signaling else None
```

(at [rl_signaling/env.py:107-110](../../rl_signaling/env.py#L107-L110)).

So if the user passes `n_signaling_actions=2, costly_signaling=True`, the agents learn over an alphabet of size 3 with the null at index 2.

## Per-agent cost

Each agent $i$ has a fixed cost $c_i \ge 0$ (Phase 1 [Axis 13](../../docs/code-audit/DEBUGGING_PLAN.md#costly-signaling)). The cost is per-agent — it does **not** depend on which non-null symbol was emitted, nor on the world state, nor on time.

In the experiment notebooks, $c_i$ is sampled uniformly per iteration:

$$c_0 = c_1 = c \sim \text{Uniform}(0, 0.5),$$

so both agents pay the same cost in a given iteration but the cost varies across iterations. See [notebooks/Final_Costly_Signaling_Run_Simulations.ipynb](../../notebooks/Final_Costly_Signaling_Run_Simulations.ipynb).

## Cost flow into the reward

The per-episode reward of agent $i$ is

$$\boxed{\; r_i \;=\; G_i(\mathbf{v}, \alpha_i) \;-\; c_i \cdot \mathbb{1}[\sigma_i \neq \nu] \;}$$

where $\mathbb{1}[\cdot]$ is the indicator (1 when the bracketed condition holds, 0 otherwise). In words: emit any non-null signal → pay $c_i$; emit null → pay nothing.

Implementation:

```python
rewards = [
    r - signal_cost[i] if signals[i] != self._null_signal_index else r
    for i, r in enumerate(rewards)
]
```

at [rl_signaling/env.py:236-241](../../rl_signaling/env.py#L236-L241), where `rewards` on the right-hand side is the pre-cost game-dict lookup $G_i(\mathbf{v}, \alpha_i)$.

The cost is **subtracted from the per-episode reward** (Phase 1 [Axis 14](../../docs/code-audit/DEBUGGING_PLAN.md#costly-signaling)). It is not tracked on a separate accumulator — `rewards_history` records the **net** value $r_i$. Recovering the gross reward and the cost separately would require adding parallel storage, which the env does not currently do.

### Worked arithmetic — three cases

Take $G_i(\mathbf{v}, \alpha_i) = 1$ and $c_i = 0.25$. Then:

| Signal $\sigma_i$ | Cost paid | Net reward $r_i$ |
|---|---|---|
| any non-null ($\sigma_i \in \{0, \dots, K-1\}$) | $0.25$ | $1 - 0.25 = 0.75$ exact |
| null ($\sigma_i = K$) | $0$ | $1 - 0 = 1.0$ exact |
| (mixed across two agents — agent 0 non-null, agent 1 null) | $(0.25, 0)$ | $(0.75, 1.0)$ exact |

These three cases are the assertions in [tests/test_numerical_sanity.py](../../tests/test_numerical_sanity.py#L114-L150).

## Suppression on the receiver side

The null signal carries an additional special role: it is **not appended** to the receiver's post-signal observation $\tilde{\mathbf{o}}_i$ (Phase 1 [Axis 5](../../docs/code-audit/DEBUGGING_PLAN.md#signals)). So if agent $j \in \mathcal{N}_i$ emits null, agent $i$ receives no token from $j$ — silence is silent.

Implementation: the suppression is at [rl_signaling/env.py:274-281](../../rl_signaling/env.py#L274-L281):

```python
for neig in self.graph.predecessors(i):
    if self.costly_signaling and signals[neig] == self._null_signal_index:
        continue
    new_observations[i] = new_observations[i] + (signals[neig],)
```

This has a subtle consequence on the action space: the **length** of $\tilde{\mathbf{o}}_i$ depends on how many in-neighbours emitted null. If $i$ has $|\mathcal{N}_i| = 2$ in-neighbours and one of them emits null, $|\tilde{\mathbf{o}}_i|$ is $|\mathbf{o}_i| + 1$ instead of $|\mathbf{o}_i| + 2$. So the action policy $\pi_i^{\text{act}}$ must handle observations of varying length. Practically, this means the urn / Q-table dictionary key is a tuple whose length varies — different keys for "neighbour 0 silent" vs "neighbour 0 sent something."

## Why a null signal at all?

Without a null option, every signaling policy must commit to *some* element of the base alphabet, paying $c_i$ on every episode. The total cost over $T$ episodes is $T \cdot c_i$, regardless of whether the signal is informative.

With null available, an agent can learn to *not speak* if speaking is unprofitable — e.g., when its observation is uninformative about the optimal action, or when $c_i$ exceeds the marginal payoff gain of communicating. In equilibrium, costly signaling models predict a separating equilibrium where signals are emitted only when they justify their cost; the null option encodes the outside option of staying quiet.

## Compatibility with the project's three agent types

The costly extension introduces real-valued and potentially negative rewards into the per-episode reward signal. This interacts with the three agent types differently:

| Agent | Reward range tolerated | Compatibility with costly signaling |
|---|---|---|
| `QLearningAgent` | $\mathbb{R}$ — the TD update $Q \leftarrow Q + \alpha (r - Q)$ is a stochastic-approximation rule defined on $\mathbb{R}$ | **Native.** No conceptual or numerical issue. |
| `TDLearningAgent` | $\mathbb{R}$ — same | **Native.** No issue. |
| `UrnAgent` | $\mathbb{Z}_{\ge 0}$ in the classical Roth-Erev formulation | **Theoretically ill-defined** — see below |

### Why UrnAgent is incompatible

The classical Roth-Erev urn (Roth & Erev 1995; Erev & Roth 1998) is built around two assumptions that costly signaling violates:

1. **Non-negative rewards.** Each play of action $a$ adds $r$ "balls" to its compartment. Negative rewards have no urn-counter interpretation.
2. **Integer rewards.** The urn-as-counter metaphor and the Friedman/Pólya urn-scheme combinatorics assume integer counts.

The project's `UrnAgent` extends the rule to real and negative rewards via real-valued weights and the non-negativity clamp $\mathrm{urn}[a] \leftarrow \max(0, \mathrm{urn}[a] + r)$. Both extensions break the canonical interpretation:

- **Real-valued urns** are mathematically fine — the probability formula still computes — but the urn-as-balls model is gone.
- **Clamping at zero** introduces an absorbing barrier: once $\mathrm{urn}[a] = 0$, action $a$ is never sampled again, so it can never recover, even if the environment would later produce positive reward for it.

Under `create_random_canonical_game(n=1, m=0)` and $c \in (0, 0.5)$, roughly $\tfrac{3}{4}$ of episodes (those where the agent picks a suboptimal action while emitting a non-null signal) yield negative rewards. So the clamp is not a rare-event guard — it fires regularly, and the dynamics that result are a degenerate variant of Roth-Erev rather than the model the literature defines.

For the formal treatment of these constraints, see [agent_urn.md, "Applicability constraints"](agent_urn.md#applicability-constraints--when-roth-erev-is-well-defined).

### Recommendation

The costly-signaling experiment should be reported only for `QLearningAgent` and `TDLearningAgent`. If `UrnAgent` results under costly signaling are produced for any reason, they should be labeled as "Roth-Erev with non-negativity-clamped costly extension" so the deviation from the canonical Roth-Erev rule is visible to readers.

The saved costly Roth-Erev figures in [results/](../../results/) (`Roth-Erev_canonical_costly_signal_*.png`, `q_costly_*.png`, `q_learning_costly_single_run*.png`) are flagged in [LEGACY_ERRORS_LOG.md, Error 5a](../../docs/code-audit/LEGACY_ERRORS_LOG.md#error-5a--roth-erev-costly-figures-unreproducible-cost-protocol-drift) for two compounding reasons: (i) the cost protocol drift surfaced during the audit, and (ii) the theoretical incompatibility documented in this section.

## Connection to the classical signaling-game literature

The setup here is a multi-agent extension of single-sender signaling games studied since Spence (1973, *Job Market Signaling*) and Crawford & Sobel (1982, *Strategic Information Transmission*). The classical setup has:

- Sender with private type $\theta \sim p(\theta)$.
- Sender chooses signal $s \in \mathcal{S}$ at cost $c(\theta, s)$.
- Receiver observes $s$ and chooses action $a \in \mathcal{A}$.
- Both get state-dependent payoffs $u_S(\theta, a)$, $u_R(\theta, a)$.

Equilibrium concepts:

- **Pooling equilibrium.** All sender types choose the same signal. The signal carries no information about $\theta$. Receiver's action is the prior-best.
- **Separating equilibrium.** Different types choose different signals. The signal reveals $\theta$ exactly; the receiver can compute $a^* = \arg\max_a u_R(\theta, a)$.
- **Partial-pooling / babbling equilibria.** Mixed informative content.

In the project's setup, every agent is *both* sender (it emits $\sigma_i$) and receiver (it observes $\tilde{\mathbf{o}}_i$), and the payoff structure is **independent** across agents — agent $i$'s payoff $G_i(\mathbf{v}, \alpha_i)$ does not depend on $\alpha_j$ for $j \neq i$. This is unlike the classical sender/receiver setup where one side's payoff depends on the other's action.

The interesting question is whether **self-interested** learning rules can find a separating equilibrium under partial information: agent $j$ learns to emit informative $\sigma_j$ even though doing so does not directly help $j$'s own payoff. Information emerges (or fails to emerge) entirely from the dynamics of the chosen learning rule (Roth–Erev, Q-learning, TD-learning) interacting with the environment.

The hypothesis stated in the [README](../../README.md#hypothesis):

> Each agent's payoff is independent of the others' actions, so there is no immediate incentive to communicate meaningfully. The hypothesis is that, despite this, there exists a region of the parameter space in which agents coordinate.

The costly extension lets the experiments locate that region as a function of $c_i$: if $c_i$ is too high, signaling is suppressed and partial information dominates; if $c_i$ is low, signaling emerges (under the right learning rule). The plots in [results/q_costs_vs_nmi.png](../../results/q_costs_vs_nmi.png) and [results/q_costly_vs_reward.png](../../results/q_costly_vs_reward.png) trace this trade-off empirically.

## Numerical worked example — full episode

Take $N = 2$, $n_{\text{features}} = 2$, $K = 2$ (so augmented size 3, null at index 2), $M = 4$, $c_0 = c_1 = 0.25$. Suppose the canonical game dict for both agents is

$$G_i\big((0,0)\big) = (0, 0, 1, 0), \quad G_i\big((0,1)\big) = (0, 1, 0, 0), \quad \dots$$

i.e. action 2 is optimal in state $(0,0)$.

In one episode:

1. **Nature.** $\mathbf{v} = (0, 0)$.
2. **Observations.** With $I_0 = (1)$, $I_1 = (2)$: $\mathbf{o}_0 = (0)$, $\mathbf{o}_1 = (0)$.
3. **Signals.** Both policies sample. Suppose $\sigma_0 = 1$ (non-null), $\sigma_1 = 2$ (null).
4. **Propagation.** With $\mathcal{N}_0 = \{1\}$ and $\mathcal{N}_1 = \{0\}$:
    - $\tilde{\mathbf{o}}_0 = (0)$ — agent 1 emitted null, so nothing appended.
    - $\tilde{\mathbf{o}}_1 = (0, 1)$ — agent 0 emitted 1, appended.
5. **Actions.** Suppose both policies choose action 2.
6. **Reward.**
    - Agent 0: $G_0((0,0), 2) - c_0 \cdot \mathbb{1}[\sigma_0 \neq 2] = 1 - 0.25 \cdot 1 = 0.75$.
    - Agent 1: $G_1((0,0), 2) - c_1 \cdot \mathbb{1}[\sigma_1 \neq 2] = 1 - 0.25 \cdot 0 = 1.0$.

This trace is the test case in [test_costly_signaling_mixed_signals_cost_only_non_null](../../tests/test_numerical_sanity.py#L142-L150).

## Cross-references

| Concept | Code | Spec axis |
|---|---|---|
| Alphabet augmentation | [env.py:107-110](../../rl_signaling/env.py#L107-L110) | Axis 5 |
| Cost flow | [env.py:236-241](../../rl_signaling/env.py#L236-L241) | Axis 14 |
| Per-agent scalar cost | constructor argument `signal_cost` to `run_simulation` | Axis 13 |
| Null is free | [env.py:238](../../rl_signaling/env.py#L238) (the `else r` branch) | Axis 15 |
| Null suppression on receiver side | [env.py:274-281](../../rl_signaling/env.py#L274-L281) | Axis 5 |
