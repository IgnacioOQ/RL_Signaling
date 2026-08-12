# Notation

- status: active
- type: reference
- id: rl_signaling.analytics.notation
- description: Symbols, sets, indexing conventions, and probability notation used throughout the analytics/ folder. Read this before any other math file.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-08
<!-- content -->

This file fixes the symbols used throughout [analytics/](.). Every other math file assumes these conventions.

## Indices

| Symbol | Range | Meaning |
|---|---|---|
| $i, j$ | $\{0, 1, \dots, N-1\}$ | Agent index. Zero-based to match Python iteration. $N$ is `n_agents`. |
| $t$ | $\{1, 2, \dots, T\}$ | Episode index (one-based when discussed as time). $T$ is `n_episodes`. |
| $k$ | $\{1, \dots, n_{\text{features}}\}$ | Feature index inside the world state. |
| $a$ | depends on context | Action index, used both for signals and final actions. |

When iterating over Python arrays, the corresponding zero-based index is named explicitly (e.g. "`history[t-1]` for episode $t$").

## Sets

| Symbol | Definition | Meaning |
|---|---|---|
| $\mathcal{V}$ | $\{0, 1\}^{n_{\text{features}}}$ | World-state space. $\lvert\mathcal{V}\rvert = 2^{n_{\text{features}}}$. |
| $\mathcal{V}_i$ | $\{0, 1\}^{\lvert I_i \rvert}$ | Per-agent observation space, where $I_i$ is the index list `agents_observed_variables[i]`. |
| $\mathcal{A}_{\text{sig}}$ | $\{0, 1, \dots, K-1\}$ | Signal alphabet. $K$ is `n_signaling_actions`. With costly signaling the effective alphabet has size $K + 1$ and the highest index is the **null signal**. |
| $\mathcal{A}_{\text{act}}$ | $\{0, 1, \dots, M-1\}$ | Final-action alphabet. $M$ is `n_final_actions`. |
| $\mathcal{N}_i$ | subset of $\{0, \dots, N-1\} \setminus \{i\}$ | In-neighbours of agent $i$ in the directed graph $G$, i.e. `graph.predecessors(i)`. |

## Variables

| Symbol | Type | Meaning |
|---|---|---|
| $\mathbf{v}$, $\mathbf{v}_t$ | $\mathcal{V}$ | World state in episode $t$ ("nature_vector"). $\mathbf{v} = (v_1, \dots, v_{n_{\text{features}}})$. |
| $\mathbf{o}_i$, $\mathbf{o}_{i,t}$ | $\mathcal{V}_i$ | Direct observation of agent $i$, equal to $\mathbf{v}$ projected onto the indices in $I_i$. |
| $\sigma_i$, $\sigma_{i,t}$ | $\mathcal{A}_{\text{sig}}$ | Signal emitted by agent $i$. |
| $\tilde{\mathbf{o}}_i$, $\tilde{\mathbf{o}}_{i,t}$ | $\mathcal{V}_i \times \mathcal{A}_{\text{sig}}^{\lvert\mathcal{N}_i\rvert}$ (modulo null suppression) | Post-signal observation: $\mathbf{o}_i$ concatenated with the signals received from $\mathcal{N}_i$. |
| $\alpha_i$, $\alpha_{i,t}$ | $\mathcal{A}_{\text{act}}$ | Final action of agent $i$. |
| $r_i$, $r_{i,t}$ | $\mathbb{R}$ | Per-episode reward of agent $i$. |
| $c_i$ | $\mathbb{R}_{\ge 0}$ | Per-agent signaling cost (only when costly signaling is on). |
| $G_i$ | $\mathcal{V} \to \mathcal{A}_{\text{act}} \to \mathbb{R}$ | Per-agent game dictionary. The state key is the **full** $\mathbf{v}$, regardless of `full_information`. |

## Functions

| Symbol | Signature | Meaning |
|---|---|---|
| $\pi_i^{\text{sig}}$ | $\mathcal{V}_i \to \mathcal{A}_{\text{sig}}$ | Signal policy of agent $i$ (typically stochastic). |
| $\pi_i^{\text{act}}$ | $\mathcal{V}_i \times \mathcal{A}_{\text{sig}}^{\lvert\mathcal{N}_i\rvert} \to \mathcal{A}_{\text{act}}$ | Action policy of agent $i$. |
| $H(\cdot)$ | distribution $\to \mathbb{R}_{\ge 0}$ | Shannon entropy in bits (base 2). |
| $I(\cdot;\cdot)$ | pair of random variables $\to \mathbb{R}_{\ge 0}$ | Mutual information in bits. |
| $\mathrm{NMI}(\cdot;\cdot)$ | pair of random variables $\to [0, 1]$ | Normalized mutual information. The variant used by this project is the **asymmetric, output-side normalization** $I/H(O)$. |

## Probability notation

- $p(x) = \mathbb{P}[X = x]$ for the probability mass function of a discrete random variable.
- $p(x \mid y) = \mathbb{P}[X = x \mid Y = y]$ for conditionals.
- $\mathbb{E}_{X \sim p}[f(X)] = \sum_x p(x) f(x)$ for expectations.
- The empty sum is $0$ and the empty product is $1$ (standard convention).
- The convention $0 \cdot \log 0 = 0$ is used wherever it appears (justified by $\lim_{p\to 0^+} p \log p = 0$).

## Updates

The arrow $\leftarrow$ denotes assignment, mirroring Python:

$$Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \big( r - Q(s, a) \big)$$

reads as "set $Q(s,a)$ to the right-hand side." The right-hand side is evaluated using the *pre-update* values of all variables.

## Episode-length conventions

A single episode is one full pass through:

$$\text{reset} \;\to\; \text{step\_signal} \;\to\; \text{step\_action} \;\to\; \text{reward} \;\to\; \text{update}$$

Episodes are **terminal** — there is no concept of a "next episode" feeding into the current one (in the Markov-decision-process sense, every episode is a one-step horizon). This is why `QLearningAgent` uses `td_target = reward` with no $\gamma \cdot \max_a Q(s', a)$ term: $s'$ does not exist.

The `TDLearningAgent` is an exception in spirit — its single-episode trajectory has *two* states (signal-phase and action-phase), so its update bootstraps from the action-phase Q-values during the signal-phase update. See [agent_td_learning.md](agent_td_learning.md).

## Cross-references

When this folder cites a code line, it uses the form:

> See [rl_signaling/info_theory.py:14-16](../../rl_signaling/info_theory.py#L14-L16).

The link points at the canonical implementation. Deprecated paths in the same module ([rl_signaling/env.py](../../rl_signaling/env.py)'s `NetMultiAgentEnv` and `TempNetMultiAgentEnv`) are noted only when their behavior diverges from the canonical implementation.

When this folder cites a Phase 1 confirmed-model axis from DEBUGGING_PLAN.md, the citation reads "Axis $n$" and refers to the numbered axes in the Phase 1 — Confirmed model specification section.
