# Signaling model

- status: active
- type: explanation
- id: rl_signaling.analytics.signaling_model
- description: Formal mathematical definition of the multi-agent signaling game implemented by rl_signaling/env.py — state space, observations, signals, message passing on the directed graph, action selection, and per-agent payoff.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

This file gives a precise mathematical description of the signaling game one episode at a time. The implementation lives at [rl_signaling/env.py](../../rl_signaling/env.py) (canonical class `MultiAgentEnv`); the deprecated `NetMultiAgentEnv` and `TempNetMultiAgentEnv` realize the same model up to bookkeeping differences and are not separately formalized here.

The game has six ingredients: world state, observations, signal alphabet, directed graph, action alphabet, and per-agent payoff function. The episode unrolls those ingredients in a fixed order. We formalize each in turn.

## Ingredient 1 — world state

Fix a positive integer $n_{\text{features}}$ (the parameter `n_features`). The world-state space is

$$\mathcal{V} := \{0, 1\}^{n_{\text{features}}}, \qquad \lvert\mathcal{V}\rvert = 2^{n_{\text{features}}}.$$

In each episode $t$, nature draws

$$\mathbf{v}_t = (v_{t,1}, \dots, v_{t,n_{\text{features}}}) \;\sim\; \text{Uniform}(\mathcal{V}),$$

i.i.d. across episodes. Equivalently, each component $v_{t,k}$ is an independent fair coin (Phase 1 [Axis 1](../../docs/code-audit/DEBUGGING_PLAN.md#state-and-observations)).

Implementation: [rl_signaling/env.py:148](../../rl_signaling/env.py#L148):

```python
self.nature_vector = np.random.randint(0, 2, size=self.n_features)
```

## Ingredient 2 — observations

Fix a number of agents $N$ (the parameter `n_agents`). For each agent $i \in \{0, \dots, N-1\}$, fix an ordered list of feature indices

$$I_i = (k_{i,1}, \dots, k_{i,m_i}), \qquad m_i := \lvert I_i\rvert, \qquad k_{i, \ell} \in \{1, \dots, n_{\text{features}}\}.$$

This is the entry `agents_observed_variables[i]` in the env constructor. The lists are not required to be disjoint — overlap between agents is allowed (Phase 1 [Axis 2](../../docs/code-audit/DEBUGGING_PLAN.md#state-and-observations)).

Agent $i$'s **direct observation** is the projection of $\mathbf{v}$ onto the indices in $I_i$, in the order they appear in $I_i$:

$$\mathbf{o}_i := \big(v_{k_{i,1}}, \, v_{k_{i,2}}, \, \dots, \, v_{k_{i,m_i}}\big) \in \{0,1\}^{m_i} =: \mathcal{V}_i.$$

When the env is constructed with `full_information=True`, the projection is the identity: every agent observes $\mathbf{v}$ in full, $\mathcal{V}_i = \mathcal{V}$ for all $i$.

Implementation: [rl_signaling/env.py:151-156](../../rl_signaling/env.py#L151-L156).

## Ingredient 3 — signal alphabet

Fix an alphabet size $K$ (the parameter `n_signaling_actions`). The base signal alphabet is

$$\mathcal{A}_{\text{sig}} := \{0, 1, \dots, K - 1\}.$$

When `costly_signaling=False`, this is the alphabet every agent draws signals from. When `costly_signaling=True`, the alphabet is **augmented** with a null signal at index $K$:

$$\tilde{\mathcal{A}}_{\text{sig}} := \{0, 1, \dots, K - 1, K\}, \quad \text{null index} = K.$$

Inside the env, the augmented alphabet's size is stored as `n_signaling_actions = K + 1` (so the parameter passed to the constructor is $K$ and the internal value is $K+1$). The null index is recorded as `_null_signal_index = self.n_signaling_actions - 1`.

The cost of emitting any non-null signal is fixed per agent at $c_i \ge 0$. See [costly_signaling.md](costly_signaling.md) for the full treatment.

## Ingredient 4 — directed graph

Fix a finite directed graph $G = (V, E)$ on the agent set $V = \{0, 1, \dots, N-1\}$. The convention (Phase 1 [Axis 7](../../docs/code-audit/DEBUGGING_PLAN.md#graph-and-message-passing)) is

$$(u, v) \in E \;\;\Longleftrightarrow\;\; \text{agent } u \text{ sends to agent } v.$$

So agent $i$'s in-neighbours — the set of agents whose signals $i$ receives — is

$$\mathcal{N}_i := \{ j : (j, i) \in E \},$$

equivalently `graph.predecessors(i)` in NetworkX.

**Self-loops** $(i, i) \in E$ are permitted by the env and not filtered, but assumed absent in practice (Phase 1 [Axis 8](../../docs/code-audit/DEBUGGING_PLAN.md#graph-and-message-passing)). **Parallel edges** are undefined behaviour: the env requires `nx.DiGraph` (single edge per ordered pair), not `nx.MultiDiGraph` (Phase 1 [Axis 9](../../docs/code-audit/DEBUGGING_PLAN.md#graph-and-message-passing)).

## Ingredient 5 — action alphabet

Fix a final-action alphabet size $M$ (the parameter `n_final_actions`):

$$\mathcal{A}_{\text{act}} := \{0, 1, \dots, M - 1\}.$$

Every agent draws its final action from the same alphabet. The optimal element of $\mathcal{A}_{\text{act}}$ depends on the world state through the payoff function (next ingredient).

## Ingredient 6 — payoff

Each agent $i$ has a private payoff dictionary $G_i$. Mathematically:

$$G_i : \mathcal{V} \times \mathcal{A}_{\text{act}} \to \mathbb{R}, \qquad G_i(\mathbf{v}, a) = \text{reward to agent } i \text{ when state is } \mathbf{v} \text{ and action is } a.$$

The state key is the **full** $\mathbf{v}$, regardless of `full_information` (Phase 1 [Axis 10](../../docs/code-audit/DEBUGGING_PLAN.md#actions-and-payoffs)). This is what makes signaling necessary in the partial-information regime: an agent's payoff depends on features it cannot directly observe, so coordination via signals is required to reach the optimum.

Implementation:

```python
state_key = tuple(self.nature_vector)
rewards.append(self.game_dicts[i][state_key][action])
```

at [rl_signaling/env.py:223-226](../../rl_signaling/env.py#L223-L226).

The package provides two canonical generators in [rl_signaling/games.py](../../rl_signaling/games.py):

- `create_random_game` — i.i.d. uniform integer rewards in $[0, 9]$ for every $(\mathbf{v}, a)$ pair. No structure across states.
- `create_random_canonical_game` — each $\mathbf{v}$ is paired with a distinct one-hot reward dictionary, so each state has a unique optimal action with reward $n$ (default 1) and all other actions reward $m$ (default 0).

The canonical generator is the workhorse of the experiments; its uniqueness condition $2^{n_{\text{features}}} \le M$ is asserted at construction time.

## The episode

An episode is parameterized by the agent policies $\pi_i^{\text{sig}}$ and $\pi_i^{\text{act}}$ (defined below) and unfolds in five steps:

### Step 1 — sample nature

Draw $\mathbf{v} \sim \text{Uniform}(\mathcal{V})$. Compute $\mathbf{o}_i$ for every agent.

### Step 2 — encode signals

Each agent $i$ chooses a signal independently, given its direct observation:

$$\sigma_i \sim \pi_i^{\text{sig}}(\cdot \mid \mathbf{o}_i) \in \mathcal{A}_{\text{sig}}.$$

The signal step is **simultaneous** (Phase 1 [Axis 6](../../docs/code-audit/DEBUGGING_PLAN.md#signals)): every agent samples from $\pi_i^{\text{sig}}(\cdot \mid \mathbf{o}_i)$ before any signal is delivered, so no agent's signal can depend on another agent's signal in the same episode.

### Step 3 — propagate signals

For each agent $i$, the **post-signal observation** $\tilde{\mathbf{o}}_i$ is built by appending received signals from in-neighbours:

$$\tilde{\mathbf{o}}_i := \big(\mathbf{o}_i, \sigma_{j_1}, \sigma_{j_2}, \dots\big),$$

where $j_1 < j_2 < \dots$ enumerate $\mathcal{N}_i$ in NetworkX's predecessor order. Concretely the implementation iterates in node-id order (which equals agent-index order for graphs constructed via `add_edges_from(...)`).

When `costly_signaling=True`, signals equal to the null index $K$ are **suppressed** during propagation (Phase 1 [Axis 5](../../docs/code-audit/DEBUGGING_PLAN.md#signals)): they are not appended to the receiver's observation. So if all of $i$'s in-neighbours emit null, $\tilde{\mathbf{o}}_i = \mathbf{o}_i$ (length unchanged); if some emit null and others don't, only the non-null ones are appended (variable length, dependent on the realization).

Implementation: [rl_signaling/env.py:269-281](../../rl_signaling/env.py#L269-L281):

```python
def _send_signals(self, signals, observations):
    new_observations = copy.deepcopy(observations)
    for i in range(self.n_agents):
        for neig in self.graph.predecessors(i):
            if self.costly_signaling and signals[neig] == self._null_signal_index:
                continue
            new_observations[i] = new_observations[i] + (signals[neig],)
    return new_observations
```

### Step 4 — choose final actions

Each agent $i$ chooses its final action conditional on $\tilde{\mathbf{o}}_i$:

$$\alpha_i \sim \pi_i^{\text{act}}(\cdot \mid \tilde{\mathbf{o}}_i) \in \mathcal{A}_{\text{act}}.$$

### Step 5 — collect payoff and update

The per-agent reward is

$$r_i = G_i(\mathbf{v}, \alpha_i) - c_i \cdot \mathbb{1}[\sigma_i \neq K]$$

where $c_i$ is the per-agent signaling cost (only nonzero in the costly setting; see [costly_signaling.md](costly_signaling.md)) and $\mathbb{1}$ is the indicator function. Implementation: [rl_signaling/env.py:236-241](../../rl_signaling/env.py#L236-L241).

Each agent's policies are then updated using $r_i$ — see the agent-specific files [agent_urn.md](agent_urn.md), [agent_q_learning.md](agent_q_learning.md), [agent_td_learning.md](agent_td_learning.md).

## Information regimes

The Phase 1 spec ([Axis 16](../../docs/code-audit/DEBUGGING_PLAN.md#information-regimes)) distinguishes three primary regimes by the boolean flags `(full_information, with_signals)`:

| Regime | $\mathbf{o}_i$ | $\tilde{\mathbf{o}}_i$ |
|---|---|---|
| Full info, no signals | $\mathbf{o}_i = \mathbf{v}$ | $\tilde{\mathbf{o}}_i = \mathbf{v}$ |
| Partial info, no signals | $\mathbf{o}_i \subsetneq \mathbf{v}$ | $\tilde{\mathbf{o}}_i = \mathbf{o}_i$ |
| Partial info + signals | $\mathbf{o}_i \subsetneq \mathbf{v}$ | $\tilde{\mathbf{o}}_i = (\mathbf{o}_i, \sigma_{j_1}, \dots)$ |

A trivial fourth regime (full info + signals) appears in some figures as an upper-bound baseline; the signals are mechanically present but informationally redundant given the full state.

## Joint distribution of an episode

Conditioning on policies, the joint distribution of one episode's variables factorizes as

$$\begin{aligned}
\mathbb{P}\big[\mathbf{v}, \boldsymbol{\sigma}, \boldsymbol{\tilde{\mathbf{o}}}, \boldsymbol{\alpha}\big] &= \mathbb{P}[\mathbf{v}] \cdot \prod_{i=0}^{N-1} \pi_i^{\text{sig}}(\sigma_i \mid \mathbf{o}_i(\mathbf{v})) \cdot \mathbb{1}\big[\boldsymbol{\tilde{\mathbf{o}}} = f(\boldsymbol{\sigma}, \mathbf{v}, G)\big] \cdot \prod_{i=0}^{N-1} \pi_i^{\text{act}}(\alpha_i \mid \tilde{\mathbf{o}}_i)
\end{aligned}$$

where $f$ is the deterministic propagation function defined in Step 3. The factorization makes explicit that:

1. Agents' signals are conditionally independent given $\mathbf{v}$.
2. The post-signal observations are deterministic given $(\boldsymbol{\sigma}, \mathbf{v}, G)$.
3. Agents' actions are conditionally independent given their respective $\tilde{\mathbf{o}}_i$.

The expected per-agent reward is then

$$\mathbb{E}[r_i] = \mathbb{E}_{\mathbf{v}, \boldsymbol{\sigma}, \boldsymbol{\alpha}} \big[ G_i(\mathbf{v}, \alpha_i) - c_i \cdot \mathbb{1}[\sigma_i \neq K] \big].$$

## Why signaling matters

Each agent's payoff depends on $\mathbf{v}$, but only $\mathbf{o}_i \subsetneq \mathbf{v}$ is directly observed. Without communication, the best $\pi_i^{\text{act}}$ can do is to maximize the **conditional expectation** over the unobserved features:

$$\pi_i^{\text{act, no-signals}}(\mathbf{o}_i) = \arg\max_{a \in \mathcal{A}_{\text{act}}} \;\mathbb{E}_{\mathbf{v} \mid \mathbf{o}_i}\big[ G_i(\mathbf{v}, a) \big].$$

With signaling, $\tilde{\mathbf{o}}_i$ is informative about features outside $\mathbf{o}_i$ via the in-neighbours' signals — provided the signaling policies $\pi_j^{\text{sig}}$ are *also* informative. So the question the project studies is whether *self-interested* learning rules (no shared reward, no central planner) can converge to signaling policies that are informative about $\mathbf{v}$ in a way the receivers can decode.

The information-theoretic measure of "informative signaling policy" is exactly the NMI between $\sigma_i$ and $\mathbf{o}_i$, computed empirically over episodes. See [information_theory.md](information_theory.md) for the formal definition and [agent_urn.md](agent_urn.md), [agent_q_learning.md](agent_q_learning.md), [agent_td_learning.md](agent_td_learning.md) for the dynamics.

## Cross-references

| Concept | Code | Spec axis |
|---|---|---|
| Nature vector | [env.py:148](../../rl_signaling/env.py#L148) | Axis 1 |
| Observation projection | [env.py:151-156](../../rl_signaling/env.py#L151-L156) | Axes 2, 3 |
| Simultaneous signal step | [env.py:160-191](../../rl_signaling/env.py#L160-L191) | Axis 6 |
| Null-signal suppression | [env.py:269-281](../../rl_signaling/env.py#L269-L281) | Axis 5 |
| Predecessor-as-sender convention | [env.py:274](../../rl_signaling/env.py#L274) | Axis 7 |
| Game-dict full-state key | [env.py:223](../../rl_signaling/env.py#L223) | Axis 10 |
| Reward + cost flow | [env.py:236-241](../../rl_signaling/env.py#L236-L241) | Axes 12–15 |
