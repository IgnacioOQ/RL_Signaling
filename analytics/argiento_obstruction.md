# Argiento et al. (2009) — extension obstruction for distributed rewards

- status: active
- type: explanation
- id: rl_signaling.analytics.argiento_obstruction
- description: Documents the specific step at which Argiento, Pemantle, Skyrms, and Volkov's (2009) stochastic-approximation convergence proof for the cooperative-payoff Lewis–Skyrms signaling game fails to lift to the distributed-reward signal-trading game studied here. The breaking step is the construction of the Lyapunov potential W(θ); without a global potential, the joint vector field is not a gradient flow and Pemantle's a.s.-convergence theorems do not apply directly. Identifies three concrete salvage routes (local linear stability, sum-potential check, ODE/differential-inclusion framework of Borkar 2008 ch. 9 and Benaïm–Hofbauer–Sorin 2005) that future work could pursue.
- label: [reference, math]
- injection: informational
- volatility: initial_draft
- scope: project-specific
- last_checked: 2026-05-09
<!-- content -->

This file documents the open theoretical step in §2.3 of [docs/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf](docs/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf) (Phase 3 of `TODO_WORKFLOW.md::todo.deepen_proof_of_concept`). Argiento, Pemantle, Skyrms, and Volkov [1] proved a.s. convergence to stable separating equilibria for the **cooperative-payoff** Lewis–Skyrms sender–receiver game; the open question is whether their argument lifts to the **distributed-reward** signal-trading game where each agent has its own payoff function $G_i$. The short answer is: it does not lift directly, and the precise step that breaks is the construction of the Lyapunov potential. This file states the obstruction and identifies three salvage routes.

The reader is assumed to have read [proof_of_concept_markov.md](proof_of_concept_markov.md) §"Pure-Pólya signaling-urn convergence" (the static-partner reduction) and [docs/roth_erev_polya_mle.md](docs/roth_erev_polya_mle.md) §3 (the Pólya structure of a single signaling table).

## 1. Argiento et al.'s argument in the cooperative case

The Lewis–Skyrms game has two agents with a *common* payoff $G \colon \mathcal{V} \times \mathcal{A}_{\text{act}} \to \{0, 1\}$. Both agents' Roth–Erev urns are reinforced by the same $r$. Let $\theta$ denote the joint vector of urn fractions on the product simplex $\Delta = \prod_{(i, x)} \Delta^{|\mathcal{A}_{\text{sig}}| - 1} \times \prod_{(i, x, \sigma)} \Delta^{|\mathcal{A}_{\text{act}}| - 1}$. The proof has three layers:

1. **Stochastic-approximation embedding.** Linearly interpolate the discrete urn trajectory in continuous time. The result is a Robbins–Monro process $\theta_t$ satisfying $d\theta = h(\theta)\, dt + \text{noise}$, where $h$ is the expected one-step displacement on the simplex. This is standard for Pólya-style urns ([2, §2]; [3, ch. 9]).
2. **Gradient identification.** In the cooperative case, the expected reward $W(\theta) = \mathbb{E}_{(\mathbf{v}, \sigma, a) \sim \theta}[G(\mathbf{v}, a)]$ is a single scalar function on $\Delta$. Each agent's update direction is, up to a positive scalar, the gradient of $W$ with respect to that agent's own parameters: $h_i(\theta) \propto \nabla_{\theta_i} W(\theta)$. Hence $h(\theta) \propto \nabla W(\theta)$ on $\Delta$ — the joint vector field is a *gradient flow*.
3. **A.s. convergence to stable equilibria.** Apply Pemantle's stable-manifold theorem for SA with a Lyapunov function ([2, §3.2]): from a neighborhood of any linearly stable critical point of $W$, the SA process converges to that point a.s. Argiento et al. then identify the linearly stable critical points with the *separating* signaling equilibria (those for which the signal carries full information about $\mathbf{v}$) and bound the basin of each one.

The Lyapunov function $W$ is what makes step 3 go through. It is also what lets Argiento et al. rule out limit cycles, saddle-loops, and other non-pointwise recurrent sets that could otherwise capture the SA process.

## 2. The obstruction for distributed rewards

In the signal-trading game studied here, each agent has its own payoff $G_i$. The expected reward is now an $N$-vector $W_i(\theta) = \mathbb{E}[r^{(i)} \mid \theta]$, $i \in \{1, \dots, N\}$, and the SA vector field is

$$h_i(\theta) \;\propto\; \nabla_{\theta_i} W_i(\theta), \qquad i = 1, \dots, N.$$

That is, agent $i$ gradient-follows *its own* expected reward, not a shared one. Step 2 of Argiento et al.'s argument breaks unless there exists a global potential $W$ on $\Delta$ with $\partial W / \partial \theta_i = \nabla_{\theta_i} W_i$ for every $i$. By the standard mixed-partials integrability condition this requires

$$\frac{\partial (\nabla_{\theta_i} W_i)}{\partial \theta_j} \;=\; \frac{\partial (\nabla_{\theta_j} W_j)}{\partial \theta_i}, \qquad i \neq j,$$

which fails for general distributed-reward games. (The condition can be checked by direct computation on the canonical 2-feature, 2-signal, 4-action setup: $W_i$ and $W_j$ depend on different cross-blocks of $\theta$ and the resulting cross-Hessians are not symmetric except in degenerate cases.)

Without $W$, Pemantle's stable-manifold theorem cited above does not apply. The joint vector field $h$ is still the mean field of an SA process — Borkar's framework ([3, ch. 9]) gives that, with probability one, the process tracks the *internally chain-recurrent set* of the ODE $\dot\theta = h(\theta)$. But that set need not consist of point equilibria: it can in principle be a limit cycle, a heteroclinic chain, or a more complex invariant set. Argiento et al.'s argument *cannot* rule out these alternatives in the distributed-reward case; doing so would require either a non-gradient Lyapunov-style argument or a direct ODE analysis.

This is the substantive open problem the §2.3 footnote alludes to: not "show convergence" but "rule out non-pointwise recurrent sets of $\dot\theta = h$, then characterize the stable equilibria."

## 3. Salvage routes

Three concrete approaches a future session could try, in increasing order of difficulty:

- **(a) Local linear stability at the ideal equilibria.** The 4 ideal absorbing states $\sigma^* \in \Sigma^*$ enumerated in [proof_of_concept_markov.md](proof_of_concept_markov.md) are the candidate attractors. Linearize $h$ at each $\sigma^*$ and check whether all non-tangential eigenvalues have negative real part. If yes, Pemantle's *local* stable-manifold theorem ([2, §3.2]) applies without needing a global $W$: from any neighborhood of $\sigma^*$, the SA reaches $\sigma^*$ with positive probability. This gives a basin-of-attraction statement that is strictly weaker than Argiento et al.'s but still publishable. The linearization is a finite-dimensional eigenvalue computation that can be done by hand or in `numpy`.

- **(b) Sum-potential test.** Check whether $W^\Sigma(\theta) := \sum_i W_i(\theta)$ is a Lyapunov function for the joint vector field — i.e., whether $\langle h(\theta), \nabla W^\Sigma(\theta) \rangle \ge 0$ on $\Delta$. If yes, $W^\Sigma$ plays the role $W$ played in Argiento et al., despite not being the *anti-derivative* of $h$. Whether this monotonicity holds for the matching-game family is a direct calculation; the answer may depend on the relationship between $G_1$ and $G_2$. Even a partial answer ("monotone in a neighborhood of $\Sigma^*$ but not globally") is a useful and citable result.

- **(c) Differential-inclusion / set-valued framework of Benaïm–Hofbauer–Sorin.** [4] develops SA for *non-gradient* mean fields, including game-theoretic dynamics with multiple agents. Convergence is to internally chain-recurrent sets of an associated *differential inclusion*, not necessarily to point equilibria. Their results are explicitly designed for the kind of multi-agent learning the signal-trading game falls under; verifying their hypotheses (regularity of $h$, set-valued continuity at the simplex boundary) is the cleanest path to a complete a.s. statement, but it is also the most technically demanding.

The static-partner reduction in [proof_of_concept_markov.md](proof_of_concept_markov.md) §"Pure-Pólya signaling-urn convergence" is consistent with route (a) — fixing the partner is the linearization-friendly limit — but is not itself a substitute for a joint-chain analysis. The next-largest tractable case is route (a) at a single ideal $\sigma^*$ in the canonical 2×2×4 setup.

## References

[1] Argiento, R., Pemantle, R., Skyrms, B., Volkov, S. (2009). "Learning to signal: Analysis of a micro-level reinforcement model." *Stochastic Processes and their Applications* 119(2), 373–390.

[2] Pemantle, R. (2007). "A survey of random processes with reinforcement." *Probability Surveys* 4, 1–79. §2 covers Pólya urns and SA embedding; §3.2 contains the stable-manifold theorem cited above.

[3] Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems Viewpoint.* Cambridge University Press. Ch. 9 ("Multiple time scales and avoidance of traps") covers SA convergence to internally chain-recurrent sets of the mean-field ODE without requiring a Lyapunov potential.

[4] Benaïm, M., Hofbauer, J., Sorin, S. (2005). "Stochastic approximations and differential inclusions." *SIAM J. Control Optim.* 44(1), 328–348. The set-valued / differential-inclusion extension of Borkar's framework, designed for multi-agent learning dynamics.
