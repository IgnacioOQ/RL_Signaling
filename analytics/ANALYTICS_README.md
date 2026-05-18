# Analytics — Mathematical Reference

- status: active
- type: reference
- id: rl_signaling.analytics.analytics_readme
- description: Index and conventions for the analytics/ folder — exhaustive mathematical descriptions of every quantity computed by the rl_signaling codebase, plus independent verification scripts.
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-09
<!-- content -->

This folder is the project's mathematical reference. It defines every quantity the [rl_signaling/](../rl_signaling/) package computes, derives the identities the code depends on, and provides numerical worked examples. The accompanying [scripts/](scripts/) verify the math by running **independent** implementations (numpy / scipy) against the package and asserting agreement.

The companion documents are:

- [DEBUGGING_PLAN.md](../DEBUGGING_PLAN.md) — Phase 1 confirmed model specification (the *what*).
- [MODELING_CHOICES_REF.md](../MODELING_CHOICES_REF.md) — design-space catalog (the *why this choice*).
- This folder — the math behind those choices and what the code actually computes (the *how*, formally).

## Reading order

Read top-to-bottom; each file builds on the previous.

| # | File | Topic | Depends on |
|---|---|---|---|
| 0 | [notation.md](math/notation.md) | Symbols, sets, indexing conventions | — |
| 1 | [information_theory.md](math/information_theory.md) | Shannon entropy, mutual information, NMI | notation |
| 2 | [signaling_model.md](math/signaling_model.md) | World state, observations, signals, payoff | notation |
| 3 | [costly_signaling.md](math/costly_signaling.md) | Null signal, cost flow | signaling_model |
| 4 | [agent_urn.md](math/agent_urn.md) | Roth–Erev urn dynamics | signaling_model |
| 5 | [agent_q_learning.md](math/agent_q_learning.md) | Single-step TD, exploration decay | signaling_model |
| 6 | [agent_td_learning.md](math/agent_td_learning.md) | Bootstrap, count-based α, two-phase update | signaling_model, agent_q_learning |
| 7 | [exploration_strategies.md](math/exploration_strategies.md) | ε-greedy, softmax, UCB | agent_q_learning |
| 8 | [metrics_aggregation.md](math/metrics_aggregation.md) | Trajectory → CSV column → figure pipeline; producer/consumer trace | all of the above |
| 9 | [proof_of_concept_markov.md](math/proof_of_concept_markov.md) | Modeler-perspective Markov chain on policy space; absorbing structure under `[1, 0]`; 2304-state enumeration; what's missing for a convergence proof (Roth-Erev only) | agent_urn, signaling_model, information_theory |
| 10 | [initialization_basins.md](math/initialization_basins.md) | Role of `init_weights = (n, m)` as a starting measure; per-cell drift rates; the [1,0] NMI/reward dissociation | proof_of_concept_markov |
| 11 | [math/roth_erev_polya_mle.md](math/roth_erev_polya_mle.md) | Authoritative reference: factored Roth-Erev transition kernel (computed, not estimated); pure-Pólya signaling-urn structure; coarse-grained MLE recipe | proof_of_concept_markov, agent_urn |
| 12 | [argiento_obstruction.md](math/argiento_obstruction.md) | Documents the specific step at which Argiento et al. (2009) fails to lift to distributed rewards; three concrete salvage routes | proof_of_concept_markov |
| — | [scripts/](scripts/) | Independent verification scripts | all of the above |

## Conventions

- **Math notation.** GitHub-flavored markdown with native LaTeX (`$inline$`, `$$display$$`). Identifiers in code are rendered with backticks (e.g. `q_table`).
- **Log base.** All entropies and mutual informations are in **bits**, i.e. base 2. The code uses `np.log2`. See [information_theory.md](math/information_theory.md) for the full convention.
- **Indexing.** Agent index $i \in \{0, 1, \dots, N-1\}$ (zero-based, matching Python). Episode index $t = 1, 2, \dots$ (one-based when discussing time, zero-based when slicing arrays — disambiguated explicitly where it matters).
- **Tuples vs vectors.** State and observation values are written as tuples $\mathbf{v} = (v_1, \dots, v_n)$. Distributions are written as row vectors $\mathbf{p} = (p_1, \dots, p_k)$.
- **Probability of an event.** $\mathbb{P}[\cdot]$ for events; $p(\cdot)$ for probability mass functions. $\mathbb{E}[\cdot]$ for expectation.
- **Code references.** When a formula maps to a specific code line, the file uses the form [rl_signaling/agents.py:447](../rl_signaling/agents.py#L447) so the reader can jump from the math to the implementation.

## Verification posture

Every non-trivial identity in this folder is checked at least once by either:

1. A unit test in [tests/](../tests/) (most are in [tests/test_numerical_sanity.py](../tests/test_numerical_sanity.py) and [tests/test_info_theory.py](../tests/test_info_theory.py)).
2. A standalone script in [scripts/](scripts/) that uses an independent reference implementation (e.g. `scipy.stats.entropy`, hand-coded summations) and asserts agreement with `rl_signaling`.

The two-source pattern is intentional. A single-source check (the code matches itself) cannot detect a derivation error in the docstring; a two-source check (code matches an independent reference) can. The scripts under [scripts/](scripts/) are the second source.

## How to run the scripts

```bash
.venv/bin/python -m analytics.scripts.verify_information_theory
.venv/bin/python -m analytics.scripts.verify_q_learning
.venv/bin/python -m analytics.scripts.verify_td_learning
.venv/bin/python -m analytics.scripts.verify_costly_signaling
.venv/bin/python -m analytics.scripts.verify_urn_convergence
.venv/bin/python -m analytics.scripts.study_toy_markov_chain
.venv/bin/python -m analytics.scripts.enumerate_absorbing_states
.venv/bin/python -m analytics.scripts.study_urn_basin_drift
.venv/bin/python -m analytics.scripts.study_factored_kernel
.venv/bin/python -m analytics.scripts.study_polya_signaling_convergence
.venv/bin/python -m analytics.scripts.study_coarse_grained_mle
```

Each script prints a one-line PASS/FAIL summary per check and exits non-zero if any check failed. They are deliberately not `pytest` tests — they are tutorials that double as regression checks, and you can run them individually while reading the corresponding math file.

## Proof-of-concept improvement work (2026-05-09)

Files 9–10 above and the new authoritative reference [math/roth_erev_polya_mle.md](math/roth_erev_polya_mle.md) are companions to §2.3 ("Proof of Concept") of [manuscript/submission/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf](../manuscript/submission/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf). They formalize the informal "miracle drift" argument and identify what is missing for a convergence-in-probability proof.

### Frame: modeler perspective only

§2.3 is a **modeler-perspective Markov chain analysis**. The modeler can observe the full joint state $(x, y, f^{(1)}, f^{(2)}, g^{(1)}, g^{(2)}, \sigma^{(1)}, \sigma^{(2)}, a^{(1)}, a^{(2)}, r^{(1)}, r^{(2)})$ and asks whether it converges to an ideal absorbing region. There is no "hidden" structure from the modeler's vantage — every component is observable. Receiver-side Bayesian decoding from a signal (posterior over partner's feature, etc.) is a **different problem** and is not what the proof of concept is doing.

### Files

- **[math/roth_erev_polya_mle.md](math/roth_erev_polya_mle.md)** is the new authoritative reference. It gives the **exact factored kernel** (computed, not estimated — every factor is a closed-form rational from the urn fractions), the **pure-Pólya signaling-urn observation** (agent $i$'s reward depends on the signal received from $j(i)$, not on the signal sent by $i$, so the per-color reinforcement probability $q^*(x)$ is constant across colors and the proportion vector converges almost surely to a Dirichlet limit), and the **coarse-grained MLE recipe** (project the trajectory onto a discrete feature — modal map, simplex bin, NMI bin — then count and row-normalize).

- **[proof_of_concept_markov.md](math/proof_of_concept_markov.md)** is the modeler-perspective formalization for `UrnAgent` (Roth-Erev). It writes out the joint Markov chain explicitly, characterizes the absorbing states under `init_weights = [1, 0]` (counted at $|\Sigma_{\text{abs}}| = 2 \times 24 \times 2 \times 24 = 2304$), exhibits the per-agent reward distribution they induce (4 ideal, 324 trap, mean $1/M$), and identifies what is missing for a full convergence-in-probability statement. References Argiento et al. 2009 for the analogous result in the cooperative-payoff Lewis-Skyrms case.

- **[initialization_basins.md](math/initialization_basins.md)** focuses on the four notebook `init_weights` settings as starting measures on policy space. Tabulates initial sampling probability $n / (n + m)$, explains the NMI = 1 / reward = 0.25 dissociation observed at `[1, 0]`, and gives concrete drift rates for $m > 0$ from the toy single-state model.

- **[argiento_obstruction.md](math/argiento_obstruction.md)** documents the specific step at which Argiento, Pemantle, Skyrms, Volkov (2009)'s convergence theorem for the cooperative-payoff Lewis–Skyrms game fails to lift to the distributed-reward signal-trading game (the joint vector field is no longer a gradient flow without $G_1 = G_2$, so Pemantle's stable-manifold theorem does not apply directly). Identifies three concrete salvage routes: local linear stability at the ideal equilibria, sum-potential test on $W^\Sigma = \sum_i W_i$, and the Benaïm–Hofbauer–Sorin (2005) set-valued / differential-inclusion framework.

### Scripts that validate these files

- **[scripts/study_toy_markov_chain.py](scripts/study_toy_markov_chain.py)** — exact recursion + $50{,}000$-trajectory Monte Carlo for the smallest tractable Markov chain (1 obs, 2 signals, 1 agent). Validates the closed-form for $\mathbb{E}[\rho_t]$ across three `(n, m)` settings.
- **[scripts/enumerate_absorbing_states.py](scripts/enumerate_absorbing_states.py)** — brute-force enumeration of all $2304$ deterministic policy profiles for the canonical setup. Confirms the 4-ideal / 324-trap counts and the mean-reward identity $\bar r = 1/M$.
- **[scripts/study_urn_basin_drift.py](scripts/study_urn_basin_drift.py)** — `UrnAgent`-only empirical drift snapshots across `init_weights`. Section 2 cross-validates the analytical absorbing-state distribution against $200$ random seeds at `[1, 0]`.
- **[scripts/study_factored_kernel.py](scripts/study_factored_kernel.py)** — ports the doc's `one_step_kernel_value` to the simulator's dict-of-array urn representation; validates the choice rule, single-urn transition factorization, and full-state kernel sum-to-one across all 256 candidate next states for a concrete $s_t$.
- **[scripts/study_polya_signaling_convergence.py](scripts/study_polya_signaling_convergence.py)** — empirical validation of the Pure-Pólya signaling-urn theorem with a frozen partner and frozen agent-$i$ action policy. KS test against the Beta marginal of $\mathrm{Dir}(n_0)$ across $M = 200$ seeds at $T = 8{,}000$ episodes.
- **[scripts/study_coarse_grained_mle.py](scripts/study_coarse_grained_mle.py)** — coarse-grained MLE on three feature projections (modal signaling map, simplex bins, NMI bins). Reports basin-reach probabilities $P(\mathrm{NMI} > \tau \mid \text{NMI in bin})$ for $K \in \{100, 1000, 10000\}$ and $\tau \in \{0.7, 0.9\}$ across `init_weights` $\in \{[1,1], [5,1], [100,1]\}$. The visit-time-in-basin fraction is monotone in pre-seed strength: $0.000 \to 0.000 \to 0.997$, quantitatively confirming the §2.3 informal claim.

### Key result for §2.3

> Under `init_weights = (n, 0)` for any $n > 0$, the chain starts in an absorbing state and stays in the same policy forever. The realized policy is a uniformly-random element of $\Sigma_{\text{abs}}$; per-agent reward is distributed as $\{1.00, 0.50, 0.25, 0.00\}$ with probabilities $\{96, 576, 768, 864\} / 2304$ and mean $1/M = 0.25$. This is *not* the convergence the §2.3 informal argument is aiming for — it is the limit case where the chain doesn't move, providing a control rather than a proof of concept.

For $m > 0$ the chain is non-absorbing and the toy model proves per-cell concentration $\rho_t \to 1$ a.s. via sub-martingale convergence. Lifting per-cell to joint convergence is open. The cleanest version of the §2.3 proof of concept restricts to $m > 0$, adopts the toy single-state reduction for the formal sub-martingale, and states the joint-chain lifting as the open problem (citing Argiento et al. 2009 for the Lewis-Skyrms analog).

### Roth-Erev first; Q-learning is a separate problem

The files above focus on `UrnAgent` (Roth-Erev) because it has a discrete state space ($\mathbb{Z}_{\ge 0}^{\dots}$ under integer rewards) with a clean Pólya-urn structure. The absorbing-state argument, the factored kernel of [math/roth_erev_polya_mle.md](math/roth_erev_polya_mle.md), and the toy model all leverage this discreteness. `QLearningAgent`'s state lives in $\mathbb{R}^{\dots}$ and the constant-$\alpha$ TD update is non-monotone — it requires a different mathematical apparatus (stochastic approximation theory, the ODE limit, Robbins-Monro / Kushner-Clark conditions). [agent_q_learning.md](math/agent_q_learning.md) gives the per-cell math; the joint-chain analysis is deferred to `TODO_WORKFLOW.md::todo.qlearning_proof_of_concept`.

## File metadata schema

Every `.md` file in this folder follows the metadata format used elsewhere in the repository (see [LEGACY_BUGS_LOG.md](../LEGACY_BUGS_LOG.md), [DEBUGGING_PLAN.md](../DEBUGGING_PLAN.md), etc.):

```markdown
# Title
- status: active
- type: explanation | reference
- id: rl_signaling.analytics.<slug>
- description: one-line summary
- label: [reference, math]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: YYYY-MM-DD
<!-- content -->
```
