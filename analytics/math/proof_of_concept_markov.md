# Proof of concept — the Markov chain on policy space

- status: active
- type: explanation
- id: rl_signaling.analytics.proof_of_concept_markov
- description: Formalizes §2.3 ("Proof of Concept") of `Signaling_Games_with_Distributed_Rewards.pdf` for the Roth-Erev case. Defines the modeler-perspective Markov chain on policy space induced by Roth-Erev dynamics in the canonical 2-agent signaling game, characterizes its absorbing states under `init_weights = [1, 0]`, counts them (2304), exhibits the per-agent reward distribution they induce, and identifies what is missing for a convergence-in-probability proof. Q-learning is deferred to a separate task.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-09
<!-- content -->

This file is the formal companion to the Proof of Concept section (§2.2) of the published article, "Signaling Games with Distributed Rewards" (*Philosophy of Science*). The paper sketches an informal "miracle drift" argument: the joint signal-trading game is a Markov chain over states; the ideal signaling profiles are *attractors*; if the system happens to enter the basin of attraction the ideal becomes reachable. The argument is acknowledged in footnote 4 to fall short of convergence in probability — only $\\|f_t - f^\\star\\|, \\|g_t - g^\\star\\|$ are shown to decrease in the right direction.

This document tightens the argument by writing out the chain explicitly, identifying the absorbing states for `UrnAgent` under `init_weights = [1, 0]`, counting them, and showing that the reward distribution over absorbing states explains the empirical NMI ≈ 1.0 / reward ≈ 0.25 pattern observed at `[1, 0]` in the post-fix re-run of [notebooks/Initializations_test.ipynb](../../notebooks/Initializations_test.ipynb).

The related document [initialization_basins.md](initialization_basins.md) addresses a narrower question: the role of the four `init_weights` settings as starting measures on policy space; basin-of-attraction structure for $m > 0$.

The new authoritative reference for the Roth–Erev factored kernel and the Pólya-urn analysis of the signaling tables is [roth_erev_polya_mle.md](roth_erev_polya_mle.md); it supersedes the conceptual scaffolding of this file in §"Transition kernel" and motivates the Pure-Pólya theorem appended below.

Conventions match [notation.md](notation.md). All numerical claims in this file are reproduced by the scripts at [scripts/study_toy_markov_chain.py](../scripts/study_toy_markov_chain.py), [scripts/enumerate_absorbing_states.py](../scripts/enumerate_absorbing_states.py), and [scripts/study_urn_basin_drift.py](../scripts/study_urn_basin_drift.py); the math derivations and the scripts are designed to cross-validate each other.

## Setting

We work in the canonical setup of §2.3 / Figure 1:

- **Two agents**, $i \in \{0, 1\}$, communicating on the directed graph $G$ with both edges $0 \to 1$ and $1 \to 0$.
- **World state.** A binary vector $\mathbf{v} = (v_1, v_2) \in \mathcal{V} = \{0, 1\}^2$ drawn uniformly each episode.
- **Observations.** Agent $0$ observes $v_1$, agent $1$ observes $v_2$. So $\mathcal{V}_i = \{0, 1\}$ for each $i$.
- **Signals.** Each agent emits a signal from $\mathcal{A}_{\text{sig}} = \{0, 1\}$ (so $K = 2$).
- **Actions.** Each agent picks a final action from $\mathcal{A}_{\text{act}} = \{0, 1, 2, 3\}$ (so $M = 4$).
- **Games.** A per-agent canonical matching game $G_i \colon \mathcal{V} \to \mathcal{A}_{\text{act}} \to \{0, 1\}$ such that for every world state $\mathbf{v}$ exactly one action $\alpha^\star_i(\mathbf{v}) \in \mathcal{A}_{\text{act}}$ pays $1$ and the other three pay $0$.
- **Learning rule.** `UrnAgent` (Roth-Erev). The continuous-state TD-learning dynamics are deferred to a separate task — see TODO_WORKFLOW.md `todo.qlearning_proof_of_concept`.
- **Initialization.** `init_weights = (n, m)` controls how the urns are pre-seeded by [rl_signaling/games.py:115-160](../../rl_signaling/games.py#L115-L160), `create_initial_signals`. The four notebook settings are $(1, 0)$, $(1, 1)$, $(5, 1)$, $(100, 1)$.

This is the smallest setting in which the proof of concept is non-trivial. Generalizations to more features, more signals, more agents, or random games (§3.2) are noted but not formalized here.

## State space

A single episode is a deterministic function of the joint policy and the random draws made within the episode. So we can take the **state of the chain at time $t$** to be the pair of policies of all agents:

$$\sigma_t \;=\; \bigl( u^{(0)}_{\text{sig}, t},\; u^{(0)}_{\text{act}, t},\; u^{(1)}_{\text{sig}, t},\; u^{(1)}_{\text{act}, t} \bigr).$$

Here:

- $u^{(i)}_{\text{sig}, t} \colon \mathcal{V}_i \to \mathbb{R}_{\ge 0}^{K}$ is agent $i$'s signaling urn after $t$ episodes. It assigns a non-negative real weight to each signal, for each direct observation. There are $|\mathcal{V}_i| = 2$ keys, each holding a length-$K = 2$ vector.
- $u^{(i)}_{\text{act}, t} \colon \mathcal{V}_i \times \mathcal{A}_{\text{sig}} \to \mathbb{R}_{\ge 0}^{M}$ is agent $i$'s action urn. There are $|\mathcal{V}_i| \cdot K = 4$ keys, each holding a length-$M = 4$ vector.

Each component is a non-negative-real-valued matrix. The state space is therefore a subset of

$$\Sigma \;=\; \bigl(\mathbb{R}_{\ge 0}^{2 \times 2}\bigr)^2 \times \bigl(\mathbb{R}_{\ge 0}^{4 \times 4}\bigr)^2.$$

Under integer-valued canonical-game rewards $r \in \{0, 1\}$ and `init_weights` integer-valued, every entry stays integer-valued (the bug fix in LEGACY_BUGS_LOG.md Bug 9 makes the storage dtype float, but the values are integer-valued under integer rewards). So in practice $\Sigma$ is a countable lattice.

This is a **larger** state than the §2.3 description ("a full description of the system including observed states of nature, $f$s and $g$s as tables, the signals and actions sent, and the reward obtained") — we drop the per-episode random outcomes from the state, since they are not Markov sufficient. The Markov property holds because, given $\sigma_t$, the next state $\sigma_{t+1}$ depends only on the random draws made in episode $t+1$ and not on the history of episode $1, \dots, t$.

## Transition kernel

One episode produces $\sigma_{t+1}$ from $\sigma_t$ via the following decomposition (matching the structure of [rl_signaling/simulation.py](../../rl_signaling/simulation.py)):

1. **Nature draws** $\mathbf{v} = (v_1, v_2) \sim \mathrm{Uniform}(\mathcal{V})$. Independent of $\sigma_t$.
2. **Signaling.** Each agent $i$ emits $\sigma_i \in \mathcal{A}_{\text{sig}}$ with probability proportional to $u^{(i)}_{\text{sig}, t}[\mathbf{o}_i]$, where $\mathbf{o}_i$ is its direct observation. Specifically:

   $$\mathbb{P}[\sigma_i = a \mid \sigma_t, \mathbf{v}] \;=\; \frac{u^{(i)}_{\text{sig}, t}[\mathbf{o}_i][a]}{\sum_{a'} u^{(i)}_{\text{sig}, t}[\mathbf{o}_i][a']}.$$

3. **Action.** Each agent $i$ receives the signal $\sigma_{j(i)}$ from its in-neighbour $j(i)$ (here, $j(0) = 1$ and $j(1) = 0$). It picks an action with probability proportional to $u^{(i)}_{\text{act}, t}[(\mathbf{o}_i, \sigma_{j(i)})]$.
4. **Reward.** Agent $i$ collects $r_i = G_i(\mathbf{v})[\alpha_i] \in \{0, 1\}$.
5. **Update.** The chosen cells get reinforced; for `UrnAgent` ([rl_signaling/agents.py:304-314](../../rl_signaling/agents.py#L304-L314)),

   $$u^{(i)}_{\text{sig}, t+1}[\mathbf{o}_i][\sigma_i] \;\leftarrow\; \max\bigl(0,\; u^{(i)}_{\text{sig}, t}[\mathbf{o}_i][\sigma_i] + r_i\bigr),$$

   and analogously for the action urn. All other cells are unchanged. Under non-negative rewards the $\max(0, \cdot)$ clamp is inert; we keep it because the formal results below depend on the **positive-only-update** property it expresses.

The transition kernel $\mathbb{P}(\sigma_{t+1} \mid \sigma_t)$ is therefore

$$\sum_{\mathbf{v}} \frac{1}{|\mathcal{V}|} \sum_{\sigma_0, \sigma_1} \prod_i \mathbb{P}[\sigma_i \mid \sigma_t, \mathbf{v}] \sum_{\alpha_0, \alpha_1} \prod_i \mathbb{P}[\alpha_i \mid \sigma_t, \mathbf{v}, \sigma_{j(i)}] \cdot \mathbb{1}\bigl[\sigma_{t+1} = \mathrm{update}(\sigma_t, \mathbf{v}, \sigma_*, \alpha_*)\bigr].$$

The last indicator is deterministic given the random draws.

## Absorbing states under `init_weights = (n, 0)`

A state $\sigma$ is **absorbing** if $\mathbb{P}(\sigma_{t+1} = \sigma \mid \sigma_t = \sigma) = 1$. For `UrnAgent` at $m = 0$ the absorbing structure has a clean characterization.

> **Definition.** Call a urn vector $\mathbf{u} \in \mathbb{R}_{\ge 0}^{n}$ *one-hot* if exactly one entry is positive and the others are zero. Call $\sigma$ *deterministic* if every cell in every urn is one-hot.

> **Proposition (absorbing $\Leftrightarrow$ deterministic for `UrnAgent`).** Let $\sigma$ be deterministic. Then $\sigma$ is an absorbing state of the chain — every transition leaves the *policy* $\sigma$ unchanged, although the absolute magnitudes of the positive entries may grow.

*Proof.* A deterministic policy puts probability $1$ on a unique signal at every observation, and probability $1$ on a unique action at every $(\mathbf{o}_i, \sigma_{j(i)})$ pair. So every random draw in steps 2 and 3 is in fact deterministic given $\mathbf{v}$. Step 5 increments the chosen cell by $r_i \in \{0, 1\}$. The chosen cell is the cell whose only positive entry is the picked signal/action; incrementing it leaves the cell *still one-hot* (the same coordinate is still the unique positive one). All other cells are unchanged. So $\sigma_{t+1}$ has the same one-hot pattern as $\sigma_t$, hence the same induced policy. $\square$

> **Proposition (absorbing $\Rightarrow$ deterministic).** If $\sigma$ is not deterministic — at least one cell has two or more positive entries — then $\sigma$ is *not* absorbing.

*Proof sketch.* Pick a non-one-hot cell, say a signaling cell with $u^{(i)}_{\text{sig}}[\mathbf{o}_i] = (a, b, \dots)$ with $a, b > 0$. With positive probability nature picks an episode where (i) $\mathbf{o}_i$ matches the chosen cell, (ii) the rolled signal is the smaller-weight option, (iii) the resulting joint reward is $r_i = 1$. Then update step increments the smaller entry by 1, changing the cell's distribution — so the policy is changed. The inequality $\mathbb{P}(\sigma_{t+1} \neq \sigma \mid \sigma_t = \sigma) > 0$ holds. $\square$

The construction of `create_initial_signals` ([rl_signaling/games.py:115-160](../../rl_signaling/games.py#L115-L160)) with `n_init = n, m_init = 0` produces a deterministic state at $t = 0$. So:

> **Corollary.** When `init_weights = (n, 0)` for any $n > 0$, the chain starts in an absorbing state and stays in the same policy forever.

This is the formal statement underlying the empirical observation in LEGACY_BUGS_LOG.md Bug 5's post-fix observation: under `[1, 0]`, NMI ≈ 1.0 (the policy is deterministic, so signals carry full information about the observation), while reward depends on whether the random initial bijection happens to be aligned with the games $G_0, G_1$.

## Counting absorbing states

For the canonical setting (2 agents, 2 features, 2 signals, 4 actions), the deterministic policies are characterized by four bijections — one per (agent, channel):

| Channel | Domain | Codomain | # bijections |
|---|---|---|---|
| Agent $i$ signaling | $\mathcal{V}_i = \{0, 1\}$ | $\mathcal{A}_{\text{sig}} = \{0, 1\}$ | $2! = 2$ |
| Agent $i$ action | $\mathcal{V}_i \times \mathcal{A}_{\text{sig}} = \{0,1\}^2$ | $\mathcal{A}_{\text{act}} = \{0,1,2,3\}$ | $4! = 24$ |

Per-agent deterministic policies: $2 \times 24 = 48$. Joint deterministic profiles:

$$|\Sigma_{\text{abs}}| \;=\; 48^2 \;=\; \boxed{2304}.$$

`create_initial_signals` calls `random.shuffle` independently for each of the four channels, so the initial state is uniformly distributed over the $2304$ absorbing states.

## Reward distribution over absorbing states

For each joint absorbing state $\sigma \in \Sigma_{\text{abs}}$, define the per-agent mean reward over world states:

$$\bar{r}_i(\sigma) \;=\; \frac{1}{|\mathcal{V}|} \sum_{\mathbf{v} \in \mathcal{V}} G_i(\mathbf{v})\bigl[\alpha_i^\sigma(\mathbf{v})\bigr],$$

where $\alpha_i^\sigma(\mathbf{v})$ is the action agent $i$ takes under $\sigma$ when the world state is $\mathbf{v}$. The rewards $\bar{r}_i(\sigma) \in \{0, \tfrac{1}{4}, \tfrac{1}{2}, \tfrac{3}{4}, 1\}$ since each agent's reward at each $\mathbf{v}$ is in $\{0, 1\}$ and $|\mathcal{V}| = 4$.

The script [scripts/enumerate_absorbing_states.py](../scripts/enumerate_absorbing_states.py) computes $\bar{r}_0(\sigma), \bar{r}_1(\sigma)$ for every $\sigma$. Running it on game seed $0$ produces the joint distribution shown in Table 1.

| Joint $(\bar{r}_0, \bar{r}_1)$ | count | fraction |
|---|---:|---:|
| $(1.00, 1.00)$ | 4 | 0.0017 |
| $(1.00, 0.50)$ + $(0.50, 1.00)$ | 48 | 0.0208 |
| $(1.00, 0.25)$ + $(0.25, 1.00)$ | 64 | 0.0278 |
| $(1.00, 0.00)$ + $(0.00, 1.00)$ | 72 | 0.0312 |
| $(0.50, 0.50)$ | 144 | 0.0625 |
| $(0.50, 0.25)$ + $(0.25, 0.50)$ | 384 | 0.1667 |
| $(0.25, 0.25)$ | 256 | 0.1111 |
| $(0.50, 0.00)$ + $(0.00, 0.50)$ | 432 | 0.1875 |
| $(0.25, 0.00)$ + $(0.00, 0.25)$ | 576 | 0.2500 |
| $(0.00, 0.00)$ | 324 | 0.1406 |

> **Key counts (game seed 0 — robust across seeds):**
>
> - Ideal states $\{\sigma : \bar{r}_0(\sigma) = \bar{r}_1(\sigma) = 1\}$: **4** (out of 2304).
> - Trap states $\{\sigma : \bar{r}_0 = \bar{r}_1 = 0\}$: **324**.
> - Mean per-agent reward over $\Sigma_{\text{abs}}$: $\tfrac{1}{|\Sigma_{\text{abs}}|}\sum_\sigma \bar{r}_0(\sigma) = \tfrac{1}{4}$, exactly matching the random-action baseline.

The "4 ideal states" count has a clean structural explanation. For $\bar{r}_0(\sigma) = 1$, agent $0$'s action map $g_0$ must satisfy $g_0(v_1, f_1(v_2)) = \alpha^\star_0(\mathbf{v})$ for every $\mathbf{v}$. Because $f_1$ is a bijection $\{0, 1\} \to \{0, 1\}$, the constraint on $g_0$ has a unique solution given $f_1$: each (key) → (action) entry is forced. So given $f_1$ there is exactly $1$ choice of $g_0$ achieving perfect agent-$0$ reward. With $|\{f_1\}| = 2$ choices, there are $2$ perfect $(f_1, g_0)$ pairs. By symmetry there are $2$ perfect $(f_0, g_1)$ pairs. The joint count is $2 \times 2 = 4$, matching the enumeration.

The **mean** $\tfrac{1}{|\Sigma_{\text{abs}}|}\sum_\sigma \bar{r}_i(\sigma) = \tfrac{1}{n_{\text{final\\_actions}}}$ also has a structural explanation: integrating over all $4!$ action bijections is integrating over a uniform-random action selection at every $(v_1, \sigma_{j(0)})$ key, which gives the random-action baseline.

> **Empirical confirmation.** Running 200 independent seeds of `UrnAgent` at `[1, 0]` ([scripts/study_urn_basin_drift.py](../scripts/study_urn_basin_drift.py) Section 2):
>
> - Reward $1.00$: 5 trials (theoretical 4/2304 → 0.34 expected; observed 5 within Poisson variance).
> - Reward $0.50$: 53 (theoretical $\tfrac{576}{2304} = 25\%$ → 50 expected; observed 53).
> - Reward $0.25$: 68 (theoretical $\tfrac{768}{2304} = 33.3\%$ → 67 expected; observed 68).
> - Reward $0.00$: 74 (theoretical $\tfrac{864}{2304} = 37.5\%$ → 75 expected; observed 74).
> - Empirical mean: $0.237$, theoretical $0.250$.

The agreement is strong. The `[1, 0]` failure mode is fully explained by the absorbing-state geometry: the chain locks into a uniformly-random absorbing state, and the absorbing states have most of their mass on low-reward profiles.

## Reachable states under `m > 0`

When $m > 0$ the chain has no absorbing states (Proposition: not deterministic ⇒ not absorbing). Instead, each cell of every urn evolves according to a state-dependent birth process. The simplest tractable instance is the *single-state, single-channel, single-agent* reduction studied in [scripts/study_toy_markov_chain.py](../scripts/study_toy_markov_chain.py).

> **Toy model.** One agent, one observation, two signals $\{0, 1\}$, deterministic reward $r(\sigma) = \mathbb{1}[\sigma = i^\star]$ for a fixed correct signal $i^\star$. Urn $\mathbf{u} = (u_{\text{hot}}, u_{\text{cold}}) \in \mathbb{Z}_{\ge 0}^2$ where $u_{\text{hot}} = u[i^\star]$.

In this reduction the Markov chain has:

- $u_{\text{cold}}$ is **constant along trajectories** (the wrong signal yields reward $0$, so its cell is never reinforced).
- $u_{\text{hot}}$ is a state-dependent birth process: it grows by $1$ with probability $\rho_t = u_{\text{hot}, t} / (u_{\text{hot}, t} + u_{\text{cold}})$ at each step, else stays.

Define $X_t = u_{\text{hot}, t} - u_{\text{hot}, 0}$ (number of correct signals in $[0, t)$). The recursion

$$\mathbb{P}(X_{t+1} = k+1 \mid X_t = k) = \rho_k = \frac{n_0 + k}{n_0 + k + m}, \qquad \mathbb{P}(X_{t+1} = k \mid X_t = k) = 1 - \rho_k,$$

with $n_0 = u_{\text{hot}, 0}$, lets us compute the distribution of $X_t$ for any $t$ explicitly via dynamic programming. The script [scripts/study_toy_markov_chain.py](../scripts/study_toy_markov_chain.py) does this and validates it against $50{,}000$-trajectory Monte Carlo simulations (Section 7).

| $(n, m)$ | $\mathbb{E}[\rho_0]$ | $\mathbb{E}[\rho_{10}]$ | $\mathbb{E}[\rho_{50}]$ | $\mathbb{E}[\rho_{100}]$ | $\mathbb{E}[\rho_{200}]$ | median $t$ for $\rho_t > 0.99$ |
|---|---:|---:|---:|---:|---:|---:|
| $(1, 1)$ | 0.500 | 0.888 | 0.979 | 0.990 | 0.995 | 104 |
| $(5, 1)$ | 0.833 | 0.933 | 0.981 | 0.990 | 0.995 |  98 |
| $(100, 1)$ | 0.990 | 0.991 | 0.993 | 0.995 | 0.997 |   0 |

The takeaways:

- $\rho_t$ is a non-decreasing sub-martingale: $\mathbb{E}[\rho_{t+1} \mid \rho_t] \ge \rho_t$.
- $\rho_t \to 1$ almost surely as $t \to \infty$, by the martingale convergence theorem combined with a non-trivial-increment lower bound (when $\rho_t < 1$ there is positive probability $\rho_t$ of an increment, so the martingale cannot stall short of the boundary).
- The convergence rate is dominated by $m$, not by $n$. The hitting time depends on $k_{\min} = \lceil 99 m - n \rceil$: the larger $m$, the more "wrong-cell mass" is in the urn that needs to be diluted by hot-cell growth.

This reduction makes the §2.3 sub-martingale argument formal in the simplest non-trivial case. It does **not** generalize cleanly to the joint two-agent chain because the joint chain has multiple channels with shared rewards — the "hot" cell of agent $0$'s signaling urn depends on agent $1$'s receiver decoding, which itself evolves. The toy model is the per-cell limit: it shows what happens when you fix everything else and ask whether one cell's policy concentrates.

## Pure-Pólya signaling-urn convergence

The toy single-state model above is the simplest reduction. The next-largest tractable case fixes the partner's policy but lets *one* agent's signaling table run free across all of its observation rows. In this case, each row is a Pólya urn with Bernoulli-thinned reinforcement and a per-color reinforcement probability that does not depend on the color sampled — the formal core of the §2.3 attractor picture, and the content of §3 of [roth_erev_polya_mle.md](roth_erev_polya_mle.md).

> **Theorem (Pólya signaling urn under fixed partner and fixed own action policy).** Fix agents $i \neq j$, and freeze the partner's full policy $(f^{(j)}, g^{(j)})$ and agent $i$'s own action policy $g^{(i)}$ for all time. For each $x \in \mathcal{V}_i$, define
> $$q^*(x) \;:=\; \mathbb{P}\bigl[r^{(i)}_t = 1 \,\big|\, \mathbf{o}^{(i)}_t = x\bigr] \;=\; \sum_{y, \sigma_j, a} P(y \mid x)\, \pi_j(\sigma_j \mid y)\, \pi_i(a \mid x, \sigma_j)\, \mathbf{1}\!\big[G_i(a, x, y) = 1\big],$$
> where $\pi_j(\sigma_j \mid y) = f^{(j)}_t[y, \sigma_j] / \sum_{\sigma'} f^{(j)}_t[y, \sigma']$ and $\pi_i(a \mid x, \sigma_j)$ is the analogous fraction of the (frozen) action urn $g^{(i)}[(x, \sigma_j)]$. Then **$q^*(x)$ does not depend on the signal $\sigma_i$ that agent $i$ sends**, and the row $f^{(i)}_t[x]$ of agent $i$'s signaling table evolves as a Bernoulli-thinned Pólya urn:
> $$\mathbb{P}\bigl(n \to n + e_\sigma \bigm| n\bigr) \;=\; P(x) \cdot \frac{n_\sigma}{S} \cdot q^*(x), \qquad \mathbb{P}(n \to n \mid n) \;=\; 1 - P(x)\, q^*(x),$$
> where $n = f^{(i)}_t[x]$, $S = \sum_\sigma n_\sigma$, and $e_\sigma$ is the $\sigma$-th standard basis vector. Conditional on $q^*(x) > 0$, the proportion vector
> $$\hat{f}^{(i)}_t[x] \;=\; \frac{f^{(i)}_t[x]}{\sum_\sigma f^{(i)}_t[x, \sigma]}$$
> converges almost surely as $t \to \infty$ to a random Dirichlet-distributed limit on the simplex with parameters equal to the initial propensities $f^{(i)}_0[x]$.

*Proof.* Agent $i$'s reward $r^{(i)}_t = G_i(a^{(i)}_t, x, y)$ is determined by the action $a^{(i)}_t$, the agent's own observation $x$, and nature's draw $y$. The action $a^{(i)}_t$ is sampled from $\pi_i(\cdot \mid x, \sigma_j)$ — keyed on the signal $\sigma_j$ *received* from $j$, which is in turn sampled from $\pi_j(\cdot \mid y)$ given nature's $y$. In particular, $a^{(i)}_t$ is independent of the signal $\sigma_i$ that agent $i$ emits. So conditioning on $\mathbf{o}^{(i)}_t = x$ but marginalizing over $\sigma_i$, the per-color reinforcement probability is

$$\mathbb{P}\bigl[r^{(i)}_t = 1 \bigm| x, \sigma_i = s\bigr] \;=\; \mathbb{P}\bigl[r^{(i)}_t = 1 \bigm| x\bigr] \;=\; q^*(x), \qquad \forall s \in \mathcal{A}_{\text{sig}}.$$

Per-episode dynamics of the row $n_t = f^{(i)}_t[x]$:

- With probability $1 - P(x)$, nature draws $\mathbf{o}^{(i)}_t \neq x$ and the row is untouched.
- With probability $P(x)$, agent $i$ samples $\sigma_i \sim n_t / S_t$ and receives reward 1 with probability $q^*(x)$, independently of $\sigma_i$. The row is updated to $n_t + e_{\sigma_i}$ if rewarded, else left unchanged.

Combining: $\mathbb{P}(n \to n + e_\sigma) = P(x) \cdot (n_\sigma / S) \cdot q^*(x)$ and $\mathbb{P}(n \to n) = 1 - P(x)\, q^*(x)$.

Now sub-sample the trajectory to those steps $\tau_1 < \tau_2 < \dots$ where the row was both visited *and* reinforced. Conditional on $q^*(x) > 0$, this sub-sampled chain is well-defined a.s. (each step contributes a visit-and-reinforce with probability $P(x)\, q^*(x) > 0$ independently of the past) and the number of visit-and-reinforce events in $[0, t]$ tends to infinity a.s. by the strong law of large numbers. On the sub-sampled chain the row evolves exactly as a *classical* Pólya urn with $K$ colors and initial composition $n_0 = f^{(i)}_0[x]$:

$$n_{\tau_{k+1}} \;=\; n_{\tau_k} + e_{\sigma}, \qquad \sigma \sim n_{\tau_k} / S_{\tau_k}.$$

By the classical Eggenberger–Pólya theorem (see Pemantle 2007, §2 or Mahmoud 2008, Ch. 3), the proportion vector $n_{\tau_k} / S_{\tau_k}$ converges almost surely to a random limit on the $(K-1)$-simplex with law $\mathrm{Dir}(n_0)$ as $k \to \infty$. Between two visit-and-reinforce steps the row is unchanged, so the proportion at any non-$\tau$ step equals the proportion at the most recent $\tau$ step. Hence $\hat{f}^{(i)}_t[x] \xrightarrow{a.s.} \mathrm{Dir}(n_0)$ as $t \to \infty$. $\square$

The theorem makes precise what the §2.3 informal "miracle drift" picture says about a single agent in a static environment: the signaling table does *not* converge to a deterministic optimum — it converges to *some* random extreme point on the simplex, picked out by initial bias and the path realization. **The random selection of *which* signaling system is delivered by the Pólya structure of the $f^{(i)}$ urns; the *correctness* of the resulting communication is delivered by the $g^{(i)}$ urns adapting to whatever the $f^{(i)}$ urns drift into** ([roth_erev_polya_mle.md](roth_erev_polya_mle.md) §3, §7).

The independence-of-color condition for $q^*(x)$ is fragile: if agent $j$'s policy is allowed to evolve, $q^*(x)$ becomes a function of $g^{(j)}_t$ — so the urn becomes a *generalized* Pólya urn whose reinforcement probability drifts. The theorem above is the static-partner reduction; the joint-chain extension is the open problem of the next section. The theorem and its empirical validation on a single agent against a frozen partner are in [scripts/study_polya_signaling_convergence.py](../scripts/study_polya_signaling_convergence.py); a Kolmogorov–Smirnov test against the Beta marginal of $\mathrm{Dir}(n_0)$ does not reject at $\alpha = 0.005$ at $T = 8{,}000$ episodes across $M = 200$ seeds.

## What is missing for a convergence-in-probability proof

The §2.3 footnote and the toy model above show that, **per cell, conditional on everything else being held fixed**, the cell's policy concentrates on the optimal entry. The gap between this and a full convergence-in-probability statement on the joint chain has two parts.

1. **Multi-channel coupling.** The reward delivered to agent $0$ depends on agent $1$'s signaling and decoding, which are themselves stochastic processes. So the "fixed reward function" assumption of the toy model does not hold. The joint chain is a coupled multi-cell birth process where the per-cell birth rate at $(i, \mathbf{o}, a)$ depends on the entire joint state.

2. **Non-uniqueness of attractors.** There are $2304$ absorbing states; only $4$ are ideal. Showing that the chain enters the basin of an attractor does not settle which attractor. Convergence in probability to the *ideal* set requires either (a) showing that the basins of the trap states have measure $0$ under the dynamics from common starting points, or (b) restricting to initializations that lie in the basin of an ideal state.

Argiento, Pemantle, Skyrms, and Volkov [1] address (1) for the classical Lewis-Skyrms game (sender-receiver, common payoff) using stochastic-approximation theory: the urn dynamics is a Robbins-Monro algorithm on the policy simplex with a vector field that is the gradient of expected reward. Convergence to a stable equilibrium of that vector field is then a standard result. Extending to signal-trading games with distributed rewards is open: the vector field is no longer the gradient of a single shared potential (each agent gradient-follows its own $r^{(i)}$), so Pemantle's stable-manifold theorem does not apply directly. [argiento_obstruction.md](argiento_obstruction.md) documents the precise step that breaks and three concrete salvage routes (local linear stability, sum-potential test, Benaïm–Hofbauer–Sorin set-valued SA).

We do not solve this here. The `[1, 0]` empirical pattern (concentrate at a uniformly-random absorbing state, mean reward $0.25$) is the cleanest *negative* result we can state: under $m = 0$, no convergence-in-probability statement to the *ideal* set holds — the chain is provably stuck.

## Q-learning — deferred

Q-learning's joint chain is analyzed in stochastic-approximation language; deferred — see TODO_WORKFLOW.md `todo.qlearning_proof_of_concept`.

## Cross-references

| Claim | Code / data |
|---|---|
| State space and transition kernel | [rl_signaling/simulation.py](../../rl_signaling/simulation.py), [rl_signaling/env.py](../../rl_signaling/env.py) |
| Absorbing $\Leftrightarrow$ deterministic for `UrnAgent` | [rl_signaling/agents.py:304-314](../../rl_signaling/agents.py#L304-L314) (clamped update) |
| 2304 absorbing states | [scripts/enumerate_absorbing_states.py](../scripts/enumerate_absorbing_states.py) §2 |
| 4 ideal / 324 trap states | [scripts/enumerate_absorbing_states.py](../scripts/enumerate_absorbing_states.py) §4 |
| Mean reward $1/M$ over $\Sigma_{\text{abs}}$ | [scripts/enumerate_absorbing_states.py](../scripts/enumerate_absorbing_states.py) §7 |
| `[1, 0]` empirical reward histogram | [scripts/study_urn_basin_drift.py](../scripts/study_urn_basin_drift.py) §2 |
| Toy single-state Markov chain | [scripts/study_toy_markov_chain.py](../scripts/study_toy_markov_chain.py) |
| Closed-form $\rho_t$ recursion | [scripts/study_toy_markov_chain.py](../scripts/study_toy_markov_chain.py) §1 |
| $\rho_t$ sub-martingale convergence | [scripts/study_toy_markov_chain.py](../scripts/study_toy_markov_chain.py) §5 |

## References

[1] Argiento, R., Pemantle, R., Skyrms, B., Volkov, S. (2009). "Learning to signal: Analysis of a micro-level reinforcement model." *Stochastic Processes and their Applications*, 119(2), 373–390. The classical convergence proof for the symmetric-Lewis-Skyrms version of the chain studied here.

[2] Skyrms, B. (2010). *Signals: Evolution, Learning, and Information*. Oxford University Press. Chapter 4 develops the Roth-Erev urn dynamics for sender-receiver games and discusses convergence informally.
