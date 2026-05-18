# Initialization basins — the role of `init_weights = (n, m)`

- status: active
- type: explanation
- id: rl_signaling.analytics.initialization_basins
- description: Formal analysis of how `init_weights = (n, m)` controls the starting position of the signal-trading Markov chain on policy space. Quantifies distance-to-absorbing-set for each (n, m), explains the NMI = 1.0 / reward = 0.25 dissociation observed at [1, 0], and gives concrete drift rates for the four notebook initializations.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-09
<!-- content -->

This file extends [proof_of_concept_markov.md](proof_of_concept_markov.md) by focusing on the **initialization** axis. The notebook [notebooks/Initializations_test.ipynb](../../notebooks/Initializations_test.ipynb) varies `init_weights ∈ {(1, 0), (1, 1), (5, 1), (100, 1)}` while fixing everything else; the empirical pattern for `UrnAgent` is striking and seemingly counterintuitive — the strongest pre-seed `(1, 0)` produces the **worst** reward but the **best** NMI. This document explains why, by formalizing what `(n, m)` means as a starting point on the Markov chain studied in [proof_of_concept_markov.md](proof_of_concept_markov.md).

The companion document [proof_of_concept_markov.md](proof_of_concept_markov.md) defines the chain and its absorbing structure; it should be read first. Here we restrict attention to the initial state $\sigma_0$ and the trajectory of the first few hundred episodes, which is when the basin assignment is determined.

## What `init_weights = (n, m)` means

The implementation at [rl_signaling/games.py:115-160](../../rl_signaling/games.py#L115-L160) constructs each pre-seeded urn cell as

$$\mathbf{u}^{(\text{init})}[\mathbf{o}] \;=\; n \cdot \mathbf{e}_{\pi(\mathbf{o})} \;+\; m \cdot (\mathbf{1} - \mathbf{e}_{\pi(\mathbf{o})}),$$

where $\mathbf{e}_k$ is the $k$-th standard basis vector and $\pi$ is a uniformly-random bijection $\mathcal{V}_i \to \mathcal{A}_{\text{sig}}$ chosen by `random.shuffle`. Concretely, for each observation $\mathbf{o}$ exactly one signal coordinate is set to $n$ ("hot"); the others are set to $m$ ("cold").

**The induced sampling probability** at $\mathbf{o}$ for the hot signal is

$$\mathbb{P}\bigl[\sigma = \pi(\mathbf{o}) \mid \mathbf{o}\bigr] \;=\; \frac{n}{n + (K - 1) m},$$

where $K$ is the alphabet size. For $K = 2$ this is $n / (n + m)$. For the four notebook settings:

| $(n, m)$ | $n / (n + m)$ at $K = 2$ | concentration | regime |
|---|---:|---|---|
| $(1, 0)$ | $1.000$ | deterministic (one-hot) | absorbing state |
| $(1, 1)$ | $0.500$ | uniform | maximum entropy |
| $(5, 1)$ | $0.833$ | strong bias | non-absorbing |
| $(100, 1)$ | $0.990$ | near-deterministic | non-absorbing |

So `(n, m)` controls *where* on policy space the chain starts, not *how* it learns. The learning rule is fixed by the agent class.

## The (1, 0) case — initialization in an absorbing state

At $(n, m) = (1, 0)$ every cell of every urn is one-hot. The state $\sigma_0$ is therefore deterministic ([proof_of_concept_markov.md](proof_of_concept_markov.md), Proposition: deterministic ⇒ absorbing). The chain stays in the same policy forever. The realized policy is whichever uniformly-random profile `random.shuffle` picked for the four channels (the two signaling and two action urns).

There are $|\Sigma_{\text{abs}}| = 2304$ joint absorbing states. The shuffles are independent across the four channels, so the initial absorbing state is uniform on $\Sigma_{\text{abs}}$.

> **Observation.** Under `[1, 0]` the chain is *trivially convergent* — it never moves — but this is not the convergence the §2.3 proof of concept is hoping for. The chain converges to whatever absorbing state the random initialization picked; that absorbing state is rarely ideal.

The reward distribution over $\Sigma_{\text{abs}}$ is computed in [scripts/enumerate_absorbing_states.py](scripts/enumerate_absorbing_states.py). The per-agent marginal is:

$$\mathbb{P}_\sigma\bigl[\bar{r}_i(\sigma) = r\bigr] \;=\; \begin{cases} \tfrac{96}{2304} = 4.17\% & r = 1 \\ \tfrac{576}{2304} = 25.00\% & r = 0.5 \\ \tfrac{768}{2304} = 33.33\% & r = 0.25 \\ \tfrac{864}{2304} = 37.50\% & r = 0 \end{cases}$$

Mean: $\mathbb{E}_\sigma[\bar{r}_i(\sigma)] = 1 / M = 0.25$, the random-action baseline.

This is the formal explanation of the NMI = 1 / reward = 0.25 dissociation:

- **NMI = 1** because the policy is deterministic — agent $i$'s signal is a deterministic function of its observation, so $H(\sigma_i \mid \mathbf{o}_i) = 0$, hence $I(\mathbf{o}_i; \sigma_i) = H(\mathbf{o}_i)$ and $\mathrm{NMI} = 1$.
- **Reward $\approx 0.25$** because the chain is at a uniformly-random absorbing state, and the *mean* reward over $\Sigma_{\text{abs}}$ is the random-action baseline.

> **The dissociation is structural**, not a sample-size artifact. With more episodes, both NMI and reward stay at their initial absorbing-state values: the chain doesn't move.

## What the [1, 0] case is *not*

It is sometimes (informally) said that `[1, 0]` "fails to learn." That framing is misleading. More precisely:

- The agent's *signaling code* is fixed at initialization and is locally optimal in the sense that no individual cell can be improved without coordination from the partner.
- The agent's *task performance* is bounded above by what its initial bijection can achieve, regardless of how many episodes it gets.
- For agent $i$ to achieve reward $1$, the joint configuration of $(f_0, g_0, f_1, g_1)$ must satisfy $g_i(\mathbf{o}_i, f_{j(i)}(\mathbf{v}_{j(i)})) = \alpha_i^\star(\mathbf{v})$ for all $\mathbf{v}$. With $f_{j(i)}$ randomly chosen, this constraint determines a **unique** $g_i$. Conditional on agent $j(i)$'s signaling bijection being independently random, agent $i$'s action bijection has probability $\tfrac{1}{4!} = \tfrac{1}{24}$ of being the unique correct one.

So the failure isn't the agent's; it's the absorbing-state distribution. The agent has no levers to escape its initial assignment because the urn rule cannot grow a zero-mass cell.

## The (1, 1) case — uniform start

At $(1, 1)$ all urn cells start at $\mathbf{1}$ (uniform over signals/actions). The induced policy at $t = 0$ is uniform; both NMI and expected reward at $t = 0$ are at their respective minima ($\mathrm{NMI}_0 \approx 0$, $r_0 \approx 1/M = 0.25$).

Because $m = 1 > 0$, no cell is at the absorbing barrier. Every cell can grow in either direction — and, crucially, any reward-$1$ episode increments the chosen cell, biasing the policy toward whatever happened on that episode. This is the classical "drift" mechanism.

> **Toy bound.** Consider a single cell of agent $0$'s signaling urn at observation $v_1$. Suppose the partner $1$'s decoding map happens to be such that, when this cell would emit signal $a$, agent $0$'s expected reward conditional on emitting $a$ is some $\bar{r}(a)$. The recursion in [proof_of_concept_markov.md](proof_of_concept_markov.md) shows that, conditional on this partner state, the cell's sampling probability concentrates on $\arg\max_a \bar{r}(a)$.

The non-trivial issue is that the partner state is itself evolving. So the per-cell "hot signal" is a moving target. Under the right luck, the joint state walks into the basin of an ideal absorbing state; under the wrong luck, it walks into the basin of a sub-ideal one. The empirical mean reward at $t = 30{,}000$ for `(1, 1)` is approximately $0.85$ ([LEGACY_BUGS_LOG.md](../../LEGACY_BUGS_LOG.md) Bug 5 post-fix), and the empirical NMI is approximately $0.55$ — these are the long-run averages over the realization-by-realization basin assignments.

## The (5, 1) and (100, 1) cases — biased start with $m > 0$

These are intermediate. The starting policy is *biased* toward whatever bijection `random.shuffle` picked, but every cell has positive mass on every alternative, so the chain can move.

| $(n, m)$ | initial sampling probability of "hot" | initial $\mathrm{NMI}_0$ |
|---|---:|---:|
| $(5, 1)$ | $5/6 \approx 0.833$ | $\approx 0.42$ |
| $(100, 1)$ | $100/101 \approx 0.990$ | $\approx 0.92$ |

The toy single-state model in [scripts/study_toy_markov_chain.py](scripts/study_toy_markov_chain.py) lets us quantify the drift rate per cell:

| $(n, m)$ | $\mathbb{E}[\rho_{10}]$ | $\mathbb{E}[\rho_{50}]$ | median $t$ for $\rho_t > 0.99$ |
|---|---:|---:|---:|
| $(1, 1)$ | 0.888 | 0.979 | 104 |
| $(5, 1)$ | 0.933 | 0.981 | 98 |
| $(100, 1)$ | 0.991 | 0.993 | 0 (already above) |

Two takeaways from the toy model:

1. **The hitting time is dominated by $m$, not by $n$.** Increasing $n$ from $1$ to $100$ at fixed $m = 1$ moves the median hitting time from $104$ to $0$, while the asymptotic value of $\rho_t$ is the same. The reason: $u_{\text{cold}} = m$ is a constant along trajectories, so the only quantity that matters for $\rho_t \to 1$ is whether $u_{\text{hot}, t} \gg m$. Larger $n$ starts further along.
2. **For the joint chain, the toy bound is a lower bound on convergence speed but not directly on the final policy.** The toy model assumes the hot signal is the *correct* one. In the joint chain, whether a cell's pre-seeded hot is "correct" depends on the partner's decoding state — which is itself drifting. So even at `(100, 1)`, if the random bijection happens to be sub-ideal, the cell pulls toward the sub-ideal action quickly, and the agent has to overcome that bias before learning the correct one.

> **The empirical pattern at `(100, 1)` for `UrnAgent`** ([LEGACY_BUGS_LOG.md](../../LEGACY_BUGS_LOG.md) Bug 5 post-fix observation): reward climbs $0.20 \to 0.95\text{–}1.0$ across $30{,}000$ episodes, NMI stays around $0.9$. The slow climb in reward (compared to the toy hitting time of $0$ for $(100, 1)$) is precisely because the joint chain has to escape the partial bias toward the random-shuffle bijection, which the toy model abstracts away.

## "Distance to ideal set" framing

A useful way to summarize the four initializations is *distance to the ideal set* under a chosen metric.

Pick $d$ to be total-variation distance between the induced policy and the closest ideal policy:

$$d(\sigma) \;=\; \min_{\sigma^\star \in \Sigma^\star} \mathrm{TV}\bigl(\pi(\sigma), \pi(\sigma^\star)\bigr),$$

where $\Sigma^\star$ is the set of $4$ ideal absorbing states for the canonical game.

| $(n, m)$ | typical $d(\sigma_0)$ | $d_t \to ?$ as $t \to \infty$ |
|---|---|---|
| $(1, 0)$, aligned to $\Sigma^\star$ | $0$ (probability $4/2304$) | stays at $0$ |
| $(1, 0)$, misaligned (typical case) | strictly positive | stays positive |
| $(1, 1)$ | maximum | drifts toward $0$ stochastically |
| $(5, 1)$ | strictly less than $(1, 1)$, conditional on alignment of random shuffle | drifts toward $0$ |
| $(100, 1)$ | depends sharply on alignment | drifts toward $0$ if aligned; converges slowly otherwise |

The key qualitative feature: for $m > 0$, $d_t$ has positive probability of decreasing at every step; under $m = 0$, $d_t$ is non-decreasing only because the chain is stuck.

## Q-learning — deferred

Q-learning's joint chain is analyzed in stochastic-approximation language; deferred — see [TODO_WORKFLOW.md](../../TODO_WORKFLOW.md) `todo.qlearning_proof_of_concept`.

## Practical takeaway for the proof of concept

Section §2.3's "miracle drift" argument is **rigorously correct for $m > 0$ in the toy single-state model** (the closed-form recursion proves $\rho_t \to 1$ a.s.). It is **rigorously wrong for $m = 0$**: the chain is provably stuck at a uniformly-random absorbing state, with mean reward $1/M$.

The reason `[1, 0]` looks superficially "good" in the empirical NMI plots is that NMI is high precisely *because* the chain is stuck — a deterministic policy is by construction a maximum-information signaling code. But the rewards reveal the underlying structure: the deterministic code is not aligned with the games.

For improving §2.3, the cleanest version of the proof of concept is:

1. **Restrict to $m > 0$.** Then the chain is non-absorbing.
2. **Adopt the toy single-state reduction** to show per-cell concentration toward the locally-optimal action. This is the formal sub-martingale convergence.
3. **State the open problem** as: lift the per-cell convergence to the joint chain. Cite [Argiento et al., 2009](../../manuscript/submission/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf) reference [1] for the analogous lifting in the cooperative-payoff Lewis-Skyrms case.

The `[1, 0]` figure in §2.3 (Figure 1) should be **labeled as a control**: a configuration that *is* trivially convergent (it doesn't move), but reveals the absorbing-state geometry rather than the drift mechanism. The figure is informative *because* it is the limit case, not because it is a successful proof of concept.

## Cross-references

| Claim | Code / data |
|---|---|
| `(n, m)` initial sampling probability $n / (n + m)$ | [analytics/agent_urn.md](agent_urn.md), §"Eager (one-hot) initialization" |
| `[1, 0]` is an absorbing state | [proof_of_concept_markov.md](proof_of_concept_markov.md), Proposition 1 |
| 4 ideal / 2304 absorbing states | [scripts/enumerate_absorbing_states.py](scripts/enumerate_absorbing_states.py) §4 |
| `[1, 0]` reward histogram (200 seeds) | [scripts/study_urn_basin_drift.py](scripts/study_urn_basin_drift.py) §2 |
| Per-cell drift rates for $m > 0$ | [scripts/study_toy_markov_chain.py](scripts/study_toy_markov_chain.py) §5 |
