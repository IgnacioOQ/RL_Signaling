# Q-learning: an informal proof of concept — the canonical signaling game

This is the Q-learning companion to [Informal Proof.md](./Informal%20Proof.md). The setup, the Markov chain, and the open problem all change in instructive ways when you swap `UrnAgent` for `QLearningAgent` — and empirically Q-learning reaches the canonical game's optimum far more reliably than Roth–Erev. The aim of this note is to lay out the basics: what changes about the dynamics, what tools are appropriate, why Q-learning is empirically better, and what is left open.

We restrict throughout to the canonical 2-agent / 2-feature / 2-signal / 4-action setup, exactly as in the urn-case companion.

## 1. Setup

The game is identical to the urn case (see [Informal Proof.md](./Informal%20Proof.md) §1): two agents, binary world state, partial observation, signal, action, per-agent matching reward $r_i \in \{0, 1\}$. The 4 ideal joint policies are the same.

What changes is the agent class. `QLearningAgent` keeps two real-valued Q-tables per agent — $Q^{(i)}_{\text{sig}}[\mathbf{o}]$ for signal selection and $Q^{(i)}_{\text{act}}[(\mathbf{o}, \sigma)]$ for action selection — and updates them with a constant-$\alpha$ TD rule:

$$Q[s, a] \;\leftarrow\; Q[s, a] \;+\; \alpha\bigl(r - Q[s, a]\bigr).$$

Two structural differences from the Roth–Erev urn update matter throughout this note.

- **Q-values can *decrease*.** When $r = 0$ on a cell with $Q[s, a] > 0$, the update pulls $Q[s, a]$ *down* by $\alpha\, Q[s, a]$. Roth–Erev's $u \leftarrow \max(0, u + r)$ never decreases. So in Q-learning, bad cells get unlearned over time; in Roth–Erev, they have to be drowned out by hot-cell growth.
- **Exploration is a separate hyperparameter, not a function of the Q-values.** The choice rule is ε-greedy, softmax (temperature $\tau$), or UCB (constant $c$). Each keeps a positive sampling rate on every action regardless of how peaked the Q-table is. Roth–Erev's exploration is whatever urn-fraction sampling delivers, and it collapses as the urns concentrate.

Together, these two changes destroy the integer-lattice Pólya structure that the urn analysis lived inside. They are also, as §4 argues, exactly why Q-learning is empirically better.

## 2. The Markov chain on Q-table space

As before, the right state at time $t$ is the joint table:

$$Q_t \;=\; \bigl(Q^{(0)}_{\text{sig}, t},\; Q^{(0)}_{\text{act}, t},\; Q^{(1)}_{\text{sig}, t},\; Q^{(1)}_{\text{act}, t}\bigr).$$

The state space is now $\mathbb{R}^{D}$ for some finite $D$ — *continuous*. This single fact rules out most of the urn-case toolkit:

- **No integer-lattice absorbing states.** A Q-table with one cell much larger than the others is not absorbing — the choice rule still mixes by $\varepsilon$ (or by the softmax temperature), so the dynamics keep moving. There is no analogue of the $2304$ deterministic policies the urn chain freezes onto.
- **No Pólya structure.** The Pure-Pólya signaling theorem from [proof_of_concept_markov.md](proof_of_concept_markov.md) §"Pure-Pólya signaling-urn convergence" depended on positive-only reinforcement; constant-$\alpha$ TD does not have that property.
- **No `(1,0)` paradox.** There is no choice of initialization that puts the Q-table into a non-moving state. The dynamics are *always* probing.

So the analogues of "absorbing" and "trap" exist only as approximate notions — regions of $\mathbb{R}^D$ that the chain visits frequently or rarely. Quantifying those regions is what §3 takes up.

## 3. The right framework: stochastic approximation

The constant-$\alpha$ TD update fits cleanly into stochastic approximation (Robbins–Monro / Borkar 2008). The picture is:

Time-rescale the discrete trajectory by the step size $\alpha$. Then $Q_{t+1} = Q_t + \alpha\, \xi_t$ with mean-zero noise around the conditional expectation, and the rescaled trajectory is shadowed (in the Borkar 2008 ch. 9 sense) by the ODE

$$\dot Q \;=\; h(Q),\qquad h(Q) \;:=\; \mathbb{E}\bigl[\Delta Q \,\big|\, Q\bigr].$$

Three pieces of structure follow.

- **Fixed points of $h$.** A point $Q^\star$ with $h(Q^\star) = 0$ is a *self-consistent Q-table*: every cell visited under $Q^\star$'s choice rule has $Q^\star[s, a] = \mathbb{E}[r \mid Q^\star,\, (s, a)\text{ visited}]$. Each of the 4 ideal joint policies of the canonical game corresponds to such a fixed point (the chosen action at each $(s, a)$ key gives reward 1, so $Q$ settles at 1 there; off-policy cells settle at whatever the choice rule's residual exploration delivers).
- **Local linear stability at each ideal $Q^\star$.** Linearize $h$ at $Q^\star$ and read off the eigenvalues. If all non-tangential eigenvalues have negative real part, Pemantle's local stable-manifold theorem (Pemantle 2007 §3.2) gives almost-sure convergence to $Q^\star$ from a neighborhood — *with positive probability on the SA path*. This is the Q-learning analogue of "basin → ideal convergence" (question (ii) in the urn case): the stable-manifold theorem says that *if* the chain enters a small enough neighborhood of $Q^\star$, it stays there a.s.
- **Basin reachability is harder.** Local stability at every ideal $Q^\star$ does not by itself imply that the chain *enters* one of those neighborhoods from a generic start. That is question (i), and it requires more than the local analysis.

This is where the Argiento obstruction carries over verbatim. The cooperative-payoff Lewis–Skyrms argument identifies $h$ as the gradient of a single shared expected reward $W(\theta)$, then uses Pemantle's *global* convergence-to-stable-equilibria result. With distributed rewards, each agent's update is proportional to $\nabla_{\theta_i} W_i(\theta)$, the cross-Hessian symmetry needed to glue these into a global $W$ fails, and the joint flow could in principle have non-pointwise recurrent sets that the cooperative argument rules out for free. The obstacle and three concrete salvage routes are in [argiento_obstruction.md](argiento_obstruction.md); they apply identically to Q-learning.

## 4. Why Q-learning empirically beats Roth–Erev

Across the four `init_weights` settings tried in the project's notebooks, Q-learning reaches reward $\approx 1.0$ much more reliably than Roth–Erev — including from the worst-case `(1, 0)` initialization, where Roth–Erev provably can't move at all. Three structural reasons explain this:

1. **Cells can be unlearned.** A Roth–Erev cell that starts at weight $0$ is permanently dead; it can never gain mass, because nothing samples it. A Q-learning cell that starts with high $Q$ but produces $r = 0$ when sampled shrinks at rate $\alpha\, Q$ per visit until it stops being the choice-rule winner. Misaligned initial bijections — which Roth–Erev cannot escape — are *temporary* under Q-learning. This is the most important difference and explains most of the empirical gap.

2. **No integer-lattice absorption.** Roth–Erev's `(1, 0)` setting puts the chain inside an absorbing state and pins it forever to a uniformly-random one of the 2304 deterministic policies (mean reward $0.25$, the random-action baseline). Q-learning has no equivalent: regardless of initialization, the choice rule's residual exploration plus the non-monotone TD update guarantee the table keeps moving. There is no Q-learning analogue of the `(1, 0)` paradox.

3. **Exploration is a decoupled knob.** Roth–Erev's exploration is whatever the urn fractions deliver. As the urns concentrate, it vanishes — the chain stops probing alternatives. Q-learning's ε-greedy keeps a uniform $\varepsilon$ on every action regardless of how peaked the Q-table is; softmax keeps a positive temperature; UCB keeps the exploration bonus. So Q-learning continues to test alternatives even from a near-deterministic policy, which gives the chain a chance to discover that a different action would have done better.

The combined effect is that Q-learning's basin-reachability probability — the answer to question (i) — is empirically much closer to $1$ than Roth–Erev's, and is much less sensitive to initialization. (The TODO `todo.qlearning_proof_of_concept` Phase 4 in `TODO_WORKFLOW.md` specifies the headline empirical comparison: visit-time-in-basin fraction at NMI > 0.9, side by side with [study_coarse_grained_mle.py](../scripts/study_coarse_grained_mle.py)'s urn results.)

A clean way to read this: the urn's hard cases (`(1, 0)`, sub-ideal aligned initial bijection) are exactly the cases where Q-learning's structural advantages bite hardest. The two methods are not just different parameterizations of "the same thing" — they are different dynamics that handle the same game very differently.

## 5. The honest open problem

The position for Q-learning, as of the present draft:

- **Empirically**, Q-learning reaches reward $\approx 1.0$ across all four `init_weights` settings on the canonical game. The simulations are clean and the gap to Roth–Erev is large.
- **Per-cell convergence with the partner frozen** is the cleanest analogue of the urn-case Pólya theorem. With $h(Q)$ a contraction toward $\mathbb{E}[r \mid Q]$ and the partner-induced reward probabilities constant in time, the partner-frozen TD trajectory $Q_t \to \mathbb{E}[r]$ exponentially in the mean-field sense. Strictly stronger than the urn version: the limit is *deterministic*, no Dirichlet randomness.
- **Local linear stability at each ideal $Q^\star$** is a finite-dimensional eigenvalue computation — open but tractable. Salvage route (a) in [argiento_obstruction.md](argiento_obstruction.md). It would establish the Q-learning version of question (ii).
- **Joint a.s. convergence to the ideal set** is open for the same reason as the urn case: no global Lyapunov potential, so Argiento et al.'s argument does not lift directly. The right machinery is Borkar 2008 ch. 9's no-Lyapunov SA framework or Benaïm–Hofbauer–Sorin (2005)'s differential-inclusion framework, neither of which has been applied here yet.

So the structural picture is honest about both halves. The reason Q-learning works better is *not* that the convergence problem is solved for it — both methods sit on the same Argiento obstruction — but that Q-learning's dynamics avoid the local pathologies (dead cells, absorbing states, vanishing exploration) that make basin reachability fail empirically for Roth–Erev. Q-learning is, in effect, doing well on (i) by a robustness-of-dynamics argument that is currently empirical but plausibly formalizable.

The four-phase plan to close these gaps is in `TODO_WORKFLOW.md` under `todo.qlearning_proof_of_concept`. The expected outcomes per that plan: an explicit ODE for $h(Q)$ in the canonical setup (Phase 1), eigenvalues at each ideal $Q^\star$ (Phase 2), either an extension or a documented obstruction to global a.s. convergence (Phase 3), and an empirical visit-time-in-basin comparison Q-learning vs. Roth–Erev (Phase 4).
