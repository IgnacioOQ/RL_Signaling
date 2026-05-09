# TODO Workflow
- status: active
- type: plan
- id: rl_signaling.todo_workflow
- description: Cross-session task backlog; each task is self-contained and can be picked up by a coding agent with kb_mcp MCP tool access.
- label: [planning, agent]
- injection: excluded
- volatility: evolving
- scope: project-specific
- owner: agent
- last_checked: 2026-05-09
<!-- content -->
Cross-session task backlog. Tasks are added here when work started in a session cannot be completed immediately. Each task must be fully self-contained — a fresh agent should be able to pick it up using only the task body and the kb_mcp tools, with no additional context required.

This file is the per-repository instance of the `TODO_WORKFLOW_TEMPLATE.md` pattern. It lives at the root of the working repository alongside `WORKLOG.md` and is intentionally **not registered with kb_mcp** — agents access it via the regular filesystem `Read`/`Edit` tools, not via `knowledge_base_*` calls.

**Agent rules (picking up tasks):**
1. Read each task in full before starting. If its preconditions are unmet, skip it and note the blocker.
2. After completing a task, delete its entire block from this file (from the `---` divider above the `##` header through the `---` divider below the last line of the task body).
3. After completing one or more tasks, assess whether a `WORKLOG.md` entry is warranted — see Phase 5 of `content/workflows/CODING_AGENT_MAIN_WORKFLOW.md`.
4. Confirm a task is still valid before executing; conditions may have changed since it was written.

**Adding tasks (session authors):**
- Copy the template at the bottom of this file (without fences), fill in all fields, and insert it as a new `##` block above the Template section, preceded and followed by `---`.
- Be precise: include target file paths, specific tool calls, expected outcomes, and a verification step.

---

## User review — verify the 2026-05-09 modeler-perspective formalization (Phases 0–4 of the closed `todo.deepen_proof_of_concept`)
- status: todo
- type: task
- id: todo.user_review_deepen_proof_of_concept
- description: User-side review of the analytics/ extension produced when the agent executed `todo.deepen_proof_of_concept` end-to-end on 2026-05-09. Walk through each phase's deliverables, sanity-check the math derivations and the script outputs, and either accept the work or note specific changes needed before it is folded into the §2.3 manuscript draft.
- owner: user
- blocked_by: []
- last_checked: 2026-05-09
<!-- content -->
**Context.** This task is **for the user, not for an agent**. On 2026-05-09 the agent executed `todo.deepen_proof_of_concept` end-to-end (Phases 0–5). The work touched ten files, deleted one, renamed one, and added four. The agent's verification (`pytest tests/` = 63 passed, all 11 analytics scripts PASS) only catches *implementation* bugs; it does not check whether the math derivations are correct, whether the framing matches what the user actually wants in the §2.3 manuscript, or whether the empirical results are interpreted correctly. This review task is the gate before the work is treated as canonical.

**The closed task in one paragraph.** §2.3 of [analytics/docs/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf](analytics/docs/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf) was previously formalized only at the level of the absorbing-state enumeration. The closed task adopted [analytics/docs/roth_erev_polya_mle.md](analytics/docs/roth_erev_polya_mle.md) as the new authoritative reference, cleaned up agent-perspective and Q-learning content from the previous session, ported the doc's exact factored kernel into a verification script, stated and proved the Pure-Pólya signaling-urn theorem (single agent, partner fixed) with an empirical Dirichlet-limit check, documented the obstruction to lifting Argiento, Pemantle, Skyrms, Volkov (2009)'s convergence theorem to the distributed-reward case, and produced an empirical coarse-grained-MLE diagnostic that confirms the §2.3 informal "stronger pre-seed → larger basin" claim. Q-learning was deferred to `todo.qlearning_proof_of_concept` (filed in this same TODO_WORKFLOW.md, immediately below this block).

**Where to find the full record:** [WORKLOG.md](WORKLOG.md) entry `2026-05-09 — todo.deepen_proof_of_concept executed end-to-end (Phases 0–5)` is the authoritative summary. Read that first; the bullets below are an index into the deliverables.

---

### Phase 0 — Cleanup pass (file-by-file)

**What was done:**

- Deleted [analytics/hmm_perspective.md](analytics/hmm_perspective.md) outright (per user confirmation: receiver-side Bayes-decoding framing, not what §2.3 is doing). No NMI-bound note folded into [analytics/information_theory.md](analytics/information_theory.md).
- [analytics/proof_of_concept_markov.md](analytics/proof_of_concept_markov.md): §"Q-learning is structurally different" replaced with one-line deferral note. Description, "Learning rule" bullet, "Initialization" bullet purged of `QLearning` references.
- [analytics/initialization_basins.md](analytics/initialization_basins.md): §"QLearning: continuous start, no absorbing barrier" + post-Bug-9 numerical table replaced with one-line deferral note.
- [analytics/scripts/study_basin_drift.py](analytics/scripts/study_basin_drift.py) → [analytics/scripts/study_urn_basin_drift.py](analytics/scripts/study_urn_basin_drift.py) (plain `mv`). §3 ("QLearningAgent") and §4 ("Cross-agent comparison") removed; `policy_concentration_q`, `QLearningAgent` import, `TUNED_QL` config gone.
- [analytics/ANALYTICS_README.md](analytics/ANALYTICS_README.md) and [analytics/scripts/SCRIPTS_README.md](analytics/scripts/SCRIPTS_README.md): updated to reflect the new file structure and the new authoritative reference.

**What you should review:** Confirm that the one-line deferral notes in `proof_of_concept_markov.md` and `initialization_basins.md` are the right framing for the §2.3 manuscript — i.e., that you genuinely want Q-learning *cited* in the modeler-perspective files (via the deferral note pointing at `todo.qlearning_proof_of_concept`) rather than entirely silent. If you would prefer the modeler-perspective files to make no mention of Q-learning at all, ask the agent to remove the deferral notes.

---

### Phase 1 — Exact factored kernel

**What was done:** New file [analytics/scripts/study_factored_kernel.py](analytics/scripts/study_factored_kernel.py). Ports the doc's `one_step_kernel_value` from §2 of [analytics/docs/roth_erev_polya_mle.md](analytics/docs/roth_erev_polya_mle.md) to the simulator's dict-of-array urn representation. Three validation layers all pass:

- **Choice rule** (100,000 MC samples): max abs deviation from `n / sum(n)` ≤ 5/√N. Observed maxima 0.0008–0.0015.
- **Single-urn transition** (instrumented 20,000-episode simulation at `init_weights = (5, 1)`): empirical visit fraction matches `P(x) = 0.5`; per-signal reinforcement frequencies match the urn-fraction integrand. Critically, $q^*(x = 0) = 0.7403$ for σ = 0 vs $0.7399$ for σ = 1 (gap 0.0003 ≪ 3·SE = 0.054), empirically confirming the doc's §3 boxed observation that $q^*(x)$ is constant across colors.
- **Full-state kernel sum** (256-candidate enumeration for a concrete `s_curr`): sum = $1.000000000000001$ (machine precision). Mismatched-reward and mismatched-update counterexamples both correctly return 0.

**What you should review:**
1. The dict-of-array adaptation — confirm the indexing convention `signaling_urns[(x,)]` for `f^(i)` and `action_urns[(x, sig_received)]` for `g^(i)` matches what you want.
2. The choice of `init_weights = (5, 1)` for the single-urn validation: does this represent a regime you care about for §2.3, or would you prefer a different setting?
3. The 256-candidate enumeration uses a hand-picked toy game `G_i(a, x, y) = 1 iff a == 2*x + y`. Confirm this is acceptable as a validation example (it's not the canonical game from `create_random_canonical_game`).

---

### Phase 2 — Pure-Pólya signaling-urn theorem

**What was done:** New `## Pure-Pólya signaling-urn convergence` section added to [analytics/proof_of_concept_markov.md](analytics/proof_of_concept_markov.md) (between §"Reachable states under m > 0" and §"What is missing for a convergence-in-probability proof"). Theorem statement, proof, and pointer to the validation script. New file [analytics/scripts/study_polya_signaling_convergence.py](analytics/scripts/study_polya_signaling_convergence.py) validates the Dirichlet limit empirically.

**Theorem informal statement.** Fix the partner's full policy $(f^{(j)}, g^{(j)})$ and agent $i$'s own action policy $g^{(i)}$ for all time. For each observation $x$, the row $f^{(i)}_t[x]$ evolves as a Bernoulli-thinned Pólya urn with constant per-color reinforcement probability $q^*(x)$; conditional on $q^*(x) > 0$, the proportion vector converges almost surely to $\mathrm{Dir}(n_0)$ where $n_0$ is the initial propensity vector.

**Empirical validation.** $M = 200$ seeds × $T = 8{,}000$ episodes with `INIT_F = [3, 2]`. Theoretical $\mathrm{Dir}(3, 2)$ marginal: mean 0.6, std 0.2. Empirical: row x=0 mean 0.5756 / std 0.2094 / KS p = 0.020; row x=1 mean 0.5935 / std 0.2039 / KS p = 0.530. Both rows pass at $\alpha = 0.005$.

**What you should review:**
1. **The proof.** Confirm the proof in [analytics/proof_of_concept_markov.md](analytics/proof_of_concept_markov.md) §"Pure-Pólya signaling-urn convergence" is mathematically correct and at the right level of formality for the §2.3 manuscript. Key steps to verify: (a) the "$q^*(x)$ does not depend on $\sigma_i$" derivation, (b) the sub-sampling argument that reduces the Bernoulli-thinned urn to a classical Pólya urn at the visit-and-reinforce times, (c) the citation to Eggenberger–Pólya / Pemantle 2007 §2 for the Dirichlet limit.
2. **The "freeze both partner AND agent-i's action policy" caveat.** The theorem freezes `g^(i)` so that $q^*(x)$ is genuinely time-invariant (not just constant across colors). The doc's §3 statement is more permissive: it lets `g^(i)` evolve, in which case $q^*(x)$ drifts and the urn becomes a *generalized* Pólya. The agent chose the stricter version because it gives a clean Dirichlet limit; the more permissive version requires Pemantle's stochastic-approximation framework. If you want the manuscript to state the more permissive version, ask for the proof to be reworked.

---

### Phase 3 — Argiento obstruction (scope A)

**What was done:** New file [analytics/argiento_obstruction.md](analytics/argiento_obstruction.md) (~900 words). Per the Phase 3 status check, scope was set to A ("obstruction documented") rather than B ("obstruction + salvage sketch") or C ("extension proven").

**The obstruction.** Argiento, Pemantle, Skyrms, Volkov (2009) prove a.s. convergence to stable separating equilibria for the *cooperative-payoff* Lewis–Skyrms game by (1) embedding the urn dynamics as a stochastic-approximation process, (2) identifying the SA vector field as the gradient of the expected reward $W(\theta) = \mathbb{E}[G \mid \theta]$, and (3) applying Pemantle 2007 §3.2's stable-manifold theorem. Step 2 breaks for distributed rewards: each agent gradient-follows its *own* expected reward $W_i(\theta)$, and there is no single scalar potential whose gradient gives the joint vector field unless the cross-Hessian symmetry $\partial(\nabla_{\theta_i} W_i)/\partial \theta_j = \partial(\nabla_{\theta_j} W_j)/\partial \theta_i$ holds — and it does not for arbitrary distributed-reward games.

**Three salvage routes documented:** (a) local linear stability at each ideal $\sigma^*$ (hardest argument: requires eigenvalue computation); (b) sum-potential test on $W^\Sigma = \sum_i W_i$ (might or might not be a Lyapunov function — a direct calculation); (c) Benaïm–Hofbauer–Sorin (2005) set-valued / differential-inclusion framework (most general but most technical).

**What you should review:**
1. **The obstruction as stated.** Confirm the cross-Hessian-symmetry argument is the right formal statement of why Argiento et al. don't lift. There's a related but distinct framing in terms of "non-cooperative game" SA dynamics; if you prefer that framing, ask for the doc to be reworked.
2. **The three salvage routes.** Confirm they're the right shortlist of next-step approaches. If you've seen a fourth approach in the literature (e.g., a moment-closure argument, a coupling argument, a specific recent paper), say so.
3. **The "either outcome is publishable" framing.** The doc currently scopes itself to A. If you want to attempt B (sum-potential test) inline rather than deferring it, ask for that; it's roughly 3–5 hours of additional work per the pre-Phase-3 status check.

---

### Phase 4 — Coarse-grained MLE

**What was done:** New file [analytics/scripts/study_coarse_grained_mle.py](analytics/scripts/study_coarse_grained_mle.py). Implements the doc's §5 `estimate_coarse_transition_matrix` and three feature projections (modal map, simplex bins at 4/8 grid, NMI bins at 10 grid). Run T = 15,000 episodes per `init_weights` setting in $\{(1,1), (5,1), (100,1)\}$.

**Empirical findings:**

- Each chain concentrates in 1–3 NMI bins over T = 15k episodes. The chains do not appreciably traverse intermediate bins, so the basin-reach question $P(\mathrm{NMI} > 0.9 \mid \mathrm{NMI in bin})$ is largely vacuous at the doc's nominal threshold of 0.9 — only `(100, 1)` ever visits the high-NMI basin in finite time.
- At $\tau = 0.7$, `(5, 1)` starting in bin 4 reaches the threshold with probability 1.0 within $K \ge 100$ steps but starting in bin 6 (its modal bin) reaches with probability 0.005 within $K = 10{,}000$. This shows the chain locks in fast: a small initial deviation stays small.
- The robust diagnostic is the **visit-time-in-basin fraction**: $P(\mathrm{NMI} > 0.9$ over the whole trajectory$) = 0.000 \to 0.000 \to 0.997$ as pre-seed $n$ goes $1 \to 5 \to 100$ (with $m = 1$ fixed). This is a quantitative confirmation of the §2.3 informal "stronger pre-seed → larger basin" claim.

**What you should review:**
1. **Whether T = 15,000 episodes is enough for §2.3.** The chains barely traverse over this horizon. If the §2.3 manuscript needs a "the chain reaches the high-NMI basin from $\mathrm{NMI} = 0.5$ within $K$ steps" claim, a longer simulation (T = 100k or seed-pooled across many runs) is needed.
2. **The two NMI thresholds (0.7 and 0.9).** The 0.9 threshold is the doc's nominal target; 0.7 was added as a finite-T-attainable proxy. Confirm both thresholds belong in the manuscript (or pick one).
3. **The visit-time fraction as the headline statistic.** If you want a different summary statistic (e.g., expected hitting time to high-NMI basin, mean reach probability conditional on visiting an intermediate bin), ask for it.

---

### Phase 5 — Wrap-up

**What was done:**

- New TODO `todo.qlearning_proof_of_concept` filed in this same `TODO_WORKFLOW.md` (immediately below this block) with full framing, four phases, and a pointer to [analytics/argiento_obstruction.md](analytics/argiento_obstruction.md) noting that the distributed-reward obstruction carries over to Q-learning verbatim.
- Closed `todo.deepen_proof_of_concept` block deleted from this file.
- WORKLOG entry appended ([WORKLOG.md](WORKLOG.md)).

**What you should review:** Confirm the Q-learning follow-up TODO scopes the next session correctly. If you'd like to adjust the four phases (e.g., merge Phase 3 into the obstruction reference, add a fifth phase comparing the SA ODE limit to the discrete trajectory empirically), ask now before a future session picks it up.

---

**Verification (the agent's, not yours):**

- `pytest tests/` = 63 passed.
- All 11 analytics scripts pass when run individually:
  - `verify_information_theory`, `verify_q_learning`, `verify_td_learning`, `verify_costly_signaling`, `verify_urn_convergence` (per-cell math, pre-existing)
  - `study_toy_markov_chain`, `enumerate_absorbing_states`, `study_urn_basin_drift` (modeler-perspective Roth-Erev, pre-existing)
  - `study_factored_kernel`, `study_polya_signaling_convergence`, `study_coarse_grained_mle` (this session's contribution)
- Phase 0 verification greps clean (HMM content gone outside `analytics/docs/`).

These pass-checks only verify that the implementation runs and the empirical claims reproduce. They do not validate the math derivations or the framing choices.

---

**On completion (when you finish the review):**
- If you accept the work: delete this entire task block from `TODO_WORKFLOW.md`. Optional: append a one-line "review accepted" entry to `WORKLOG.md`.
- If you want changes: leave this block in place (or downgrade `status: todo` → `status: in_progress`) and either (a) edit the corresponding files yourself, or (b) file a new TODO describing the changes for an agent session to apply.

---

## Q-learning version of the §2.3 proof of concept — continuous-state stochastic approximation
- status: todo
- type: task
- id: todo.qlearning_proof_of_concept
- description: Lift the modeler-perspective Markov-chain analysis of §2.3 to the QLearningAgent (continuous state space). Frame the constant-α TD update as a stochastic-approximation algorithm, identify the ODE limit on the joint Q-table simplex, characterize the fixed points and basins, and either prove convergence to ideal Q-tables or document the obstruction.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-09
<!-- content -->
**Context.** `todo.deepen_proof_of_concept` (closed 2026-05-09) formalized the §2.3 proof of concept for the Roth-Erev case (`UrnAgent`), where the chain has a discrete integer-lattice state space and a clean Pólya-urn structure. Q-learning is structurally different and was deferred to this follow-up.

**Why this is a separate problem:**

- **Continuous state space.** `QLearningAgent.q_table_signaling` and `q_table_action` live in $\mathbb{R}^{\dots}$, not $\mathbb{Z}^{\dots}$. The Pólya-urn machinery does not apply directly — there is no urn-fraction interpretation, no integer-lattice non-recurrence to flag, no closed-form factored kernel.
- **Non-monotone update.** Constant-$\alpha$ TD: $Q[s, a] \leftarrow Q[s, a] + \alpha (r - Q[s, a])$. Unlike Roth-Erev's $\max(0, u + r)$, this update can *decrease* a cell with reward 0. So the absorbing-state analysis of [analytics/proof_of_concept_markov.md](analytics/proof_of_concept_markov.md) does not apply: there are no deterministic-policy fixed points in the integer-lattice sense.
- **The right framework is stochastic approximation.** The constant-$\alpha$ TD chain is a Robbins-Monro algorithm with non-decaying step size; its mean field is an ODE on the joint Q-table simplex. Convergence is studied in Kushner-Yin / Borkar 2008 / Benaïm-Hofbauer-Sorin 2005. The distributed-reward obstruction documented in [analytics/argiento_obstruction.md](analytics/argiento_obstruction.md) — that the joint vector field is not a gradient flow without $G_1 = G_2$ — carries over verbatim.

**Preconditions:**

- [analytics/proof_of_concept_markov.md](analytics/proof_of_concept_markov.md) exists and contains the Pure-Pólya theorem (Roth-Erev).
- [analytics/argiento_obstruction.md](analytics/argiento_obstruction.md) exists and documents the distributed-reward obstruction to lifting Argiento et al. (2009).
- [analytics/agent_q_learning.md](analytics/agent_q_learning.md) provides the per-cell Q-learning math.
- `pytest tests/` reports 63 passed.

---

### Phase 1 — ODE limit

1. State the SA embedding precisely: with constant step size $\alpha$ and time rescaling $t / \alpha$, the discrete TD trajectory is shadowed (in the Borkar 2008 ch. 9 sense) by an ODE $\dot Q = h(Q)$ where $h(Q) = \mathbb{E}[\Delta Q \mid Q]$ is the expected one-step displacement of the joint Q-table.
2. Compute $h(Q)$ explicitly for the canonical 2-feature, 2-signal, 4-action setup. The expected-update factor depends on the choice rule (ε-greedy, softmax, UCB). Start with softmax (smoothest, gives a closed-form gradient-of-log-Z structure).
3. Verify the formula numerically against a Monte Carlo estimate of $\mathbb{E}[\Delta Q]$ over many independent draws from a fixed $Q$. Cross-validation pattern follows [analytics/scripts/study_factored_kernel.py](analytics/scripts/study_factored_kernel.py).

### Phase 2 — Fixed-point / linear-stability analysis

4. Identify the fixed points of $\dot Q = h(Q)$. The 4 ideal absorbing states $\sigma^* \in \Sigma^*$ enumerated in [analytics/proof_of_concept_markov.md](analytics/proof_of_concept_markov.md) are the natural candidates (with $Q$-table entries arbitrarily large at the ideal action and zero elsewhere); verify by direct substitution into $h$.
5. Linearize $h$ at each ideal $Q^*$ and compute the eigenvalues. If all non-tangential eigenvalues have negative real part, $Q^*$ is locally exponentially stable and Pemantle 2007 §3.2 applies: from a neighborhood, the SA reaches $Q^*$ a.s. with positive probability. This is salvage route (a) of [analytics/argiento_obstruction.md](analytics/argiento_obstruction.md).

### Phase 3 — Lift to a.s. convergence (open)

6. Without a global Lyapunov function (the obstruction in [analytics/argiento_obstruction.md](analytics/argiento_obstruction.md)), the global a.s.-convergence statement requires either Benaïm-Hofbauer-Sorin's set-valued SA framework, or a sum-potential check on $W^\Sigma(Q) = \sum_i \mathbb{E}[r_i \mid Q]$. Document either an extension or the obstruction.

### Phase 4 — Empirical complement

7. Adapt [analytics/scripts/study_coarse_grained_mle.py](analytics/scripts/study_coarse_grained_mle.py) to `QLearningAgent`: feature functions become "argmax over Q-table rows" rather than urn modal map; basin-reach probabilities at the same NMI thresholds. The expected outcome (per `WORKLOG.md` 2026-05-09 Bug 9 entry) is that QLearning at every `init_weights` reaches reward $\approx 1.0$ by mid-run — concretely test that.
8. Compare to the UrnAgent results from [analytics/scripts/study_coarse_grained_mle.py](analytics/scripts/study_coarse_grained_mle.py): does QLearning's visit-time fraction in the high-NMI basin exceed UrnAgent's at every `init_weights`?

---

**Verification:**

- Phase 1: ODE formula matches a Monte Carlo estimate of $\mathbb{E}[\Delta Q]$ to within MC tolerance ($\le 5 / \sqrt{N}$ over $N \ge 10{,}000$ samples).
- Phase 2: linearization eigenvalues at every ideal $Q^*$ are reported; sign pattern documented.
- Phase 3: a section in (new) `analytics/qlearning_ode_analysis.md` documenting either an extension or the obstruction.
- Phase 4: a new `analytics/scripts/study_qlearning_coarse_mle.py` with the visit-time-in-basin comparison.
- All scripts under `analytics/scripts/` pass; `pytest tests/` still passes.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry recording the Phase 1-3 outputs.

**References:**

- Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems Viewpoint*. Cambridge University Press. Ch. 2-3 (basic theory), Ch. 9 (no-Lyapunov case).
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction* (2nd ed.). Robbins-Monro conditions; constant-α TD discussion.
- Benaïm, M., Hofbauer, J., Sorin, S. (2005). "Stochastic approximations and differential inclusions." *SIAM J. Control Optim.* 44(1), 328-348.
- Pemantle, R. (2007). "A survey of random processes with reinforcement." *Probability Surveys* 4, 1-79. §3.2 (local stable-manifold theorem).
- [analytics/agent_q_learning.md](analytics/agent_q_learning.md) — per-cell closed-forms.
- [analytics/argiento_obstruction.md](analytics/argiento_obstruction.md) — distributed-reward obstruction analysis (carries over to Q-learning).

---

## Verify Experiment Reproducibility End-to-End
- status: todo
- type: task
- id: todo.verify_reproducibility
- description: Re-run every experiment notebook on a clean kernel and confirm every CSV in results/ regenerates and every PNG in results/ is reproducible from the regenerated CSVs. Document the actual reproducibility status.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-09
<!-- content -->
**Context:** `LEGACY_ERRORS_LOG.md` catalogues the saved-figure status (CLEAN, BIASED-METRIC, MISLABELED, UNREPRODUCIBLE, WRONG, RETIRED) for every artifact in `results/`. As of the 2026-05-09 fix session, the Phase 5 fixes for Bugs 4, 5, 6, 7, 8 have all landed, the Roth-Erev costly experiment was retired, and `notebooks/Initializations_test.ipynb` was re-run end-to-end with its 4 figures regenerated. The remaining reproducibility work is to verify the **larger experiment notebooks** end-to-end on a fresh checkout + clean venv:

- `notebooks/Run_Simulations.ipynb` — Bug 6 fix renamed `*_complex.csv` → `*_complex_randomized.csv` and randomized action sizes per iteration. Producer / consumer chain is consistent by code review; full re-run is gated on (a) flipping `simulate=True` in the UrnAgent block (cell 15, currently gated for compute reasons) and (b) replacing cell 4's Colab `dump_path = '/content/drive/My Drive/...'` with `dump_path = '../results/'` for local execution.
- `notebooks/Final_Costly_Signaling_Run_Simulations.ipynb` — UrnAgent block retired (replaced with a note); only the Q-learning block remains active. Full re-run requires the same Colab/local-path swap.
- `notebooks/Parameter_Optimization_wchoices.ipynb` — Bug 7 fix in place; `scikit-optimize 0.10.2` provisioned via `pip install -e ".[dev]"`. Restart-and-Run-All gating verified by import resolution; full Bayesian-search re-run is research-log only and not required to keep `results/*.png` consistent.
- **Cross-cutting (not a bug, deferred)** — `game_dicts`, `signal_cost`, and (after Bug 6) per-iteration `n_signaling_actions`/`n_final_actions` are constructed inside the worker subprocess before the `np.random.seed(iteration)` call inside `run_single_case`, so individual rows of the saved CSVs are not row-reproducible from `iteration` alone. Population statistics are unaffected. Fix would migrate to `numpy.random.SeedSequence().spawn()`.

This task verifies that:
- A fresh checkout + clean venv reproduces every active CSV in `results/` from scratch.
- A fresh checkout + clean venv reproduces every active PNG in `results/` from those CSVs (excluding the retired Roth-Erev costly artifacts).
- The README's "Reproducing the figures" instructions execute without manual intervention beyond the documented Colab/local-path swap.

**Preconditions:**
- `pytest tests/` reports 61 passed (60 + the Bug 4 unit test).
- All open bugs in `LEGACY_BUGS_LOG.md` for which Phase 5 batches were chosen show `status: done`. As of 2026-05-09: Bugs 4, 5, 6, 7, 8 = done; Bug 2 = open by design (deferred per Phase 5 plan).
- An archive of the pre-fix `results/` directory is preserved on a separate branch or backup so the post-fix diff is meaningful. The 2026-05-09 fix session backed up `initializations_{rewards,nmi}.png` to `/tmp/rl_signaling_prefix_backup/`; for the rest of `results/`, recover from `git log` history before the 2026-05-09 commit if needed.

**Steps:**
1. From a clean checkout (or after `git stash` of the working tree), set up the venv:
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   python -m ipykernel install --user --name rl_signaling --display-name "Python (rl_signaling)"
   ```
2. Run the test suite to confirm green baseline:
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```
   Expected: 61 passed.
3. Restart-and-Run-All each experiment notebook on a fresh kernel, in this order:
   - `notebooks/Run_Simulations.ipynb` — produces `urnagent_results_canonical.csv`, `qlearning_results_canonical.csv`, `td_learning_results_canonical.csv`, and `*_complex_randomized.csv` (Bug 6 fix already applied; producer renamed to match the consumer). Before running locally, replace cell 4 (`from google.colab import drive` + `dump_path = '/content/drive/...'`) with `dump_path = '../results/'`, and flip `simulate=True` in cell 15 (UrnAgent complex block) if you want that block to execute.
   - `notebooks/Initializations_test.ipynb` — already re-run end-to-end on 2026-05-09; verify the regenerated `initializations_{rewards,nmi}.png` and `initializations_urn_{rewards,nmi}.png` figures still match the 2026-05-09 outputs (paired-comparison seed reset = deterministic re-run modulo joblib worker startup state — but this notebook uses no Parallel, so it should be fully reproducible).
   - `notebooks/Final_Costly_Signaling_Run_Simulations.ipynb` — UrnAgent block retired; only the Q-learning block remains. Same Colab/local-path swap as Run_Simulations.
   - `notebooks/Parameter_Optimization_wchoices.ipynb` — research log; verify Restart-and-Run-All no longer raises `NameError` (Bug 7 fix). Same Colab/local-path swap as Run_Simulations.
   - `notebooks/plotting_results.ipynb` — runs after the others; consumes the regenerated CSVs.
4. After each notebook completes, check that the expected CSV(s) under `results/` have been freshly written. Diff a few sample rows against the pre-fix archive to quantify the change. Predicted-direction predictions are in `LEGACY_ERRORS_LOG.md` Section G.
5. Run `notebooks/plotting_results.ipynb` on a fresh kernel via Restart-and-Run-All. Confirm every read succeeds without `FileNotFoundError`, and every PNG under `results/` has a fresh timestamp.
6. Compute and record the diff statistics: for each saved CSV, compare pre-fix and post-fix means / standard deviations of every numeric column. For each PNG, optionally use `scikit-image` SSIM or eyeball-compare against the pre-fix archive.
7. Write up the reproducibility audit as either:
   - A new `## YYYY-MM-DD — Reproducibility audit` entry in `WORKLOG.md`, or
   - A standalone `REPRODUCIBILITY.md` at the repo root if the audit is large enough to warrant its own document.
8. Update the README's "Reproducing the figures" section if any step requires extra manual setup that the current text does not document.
9. **Optional but recommended:** migrate the multiprocessing seeding pattern to `numpy.random.SeedSequence().spawn()` so individual rows of the saved CSVs are row-reproducible from `iteration` alone. See `content/how-to/NOTEBOOK_WRITING_SKILL.md` Section 8 ("Parallel processing — Seeds across workers") for the recommended pattern. If deferred, file a separate task.

**Verification:**
- `git status` after a fresh end-to-end run shows clean modifications only to expected files (CSVs in `results/`, PNGs in `results/`, optionally notebook output cells).
- A diff between pre-fix and post-fix figures is documented in `WORKLOG.md` or `REPRODUCIBILITY.md`.
- `pytest tests/` still passes.
- The README "Reproducing the figures" section reflects the current procedure with no inaccuracies.
- `LEGACY_ERRORS_LOG.md` is updated: every `UNREPRODUCIBLE` verdict is replaced with either `CLEAN` (if the post-fix re-run resolved it) or kept with a note explaining why reproducibility is still partial (e.g. multiprocessing-seed row-level non-reproducibility).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry recording the audit results.

---

## Task Template

Copy the block below (without the outer fences), fill in all fields, and insert it as a new `## [Task Title]` task block.

````markdown
## [Task Title]
- status: todo
- type: task
- id: todo.[short_id]
- description: One-sentence description of what this task accomplishes.
- owner: agent
- blocked_by: []
- last_checked: {{YYYY-MM-DD}}
<!-- content -->
**Context:** Why this task exists and what triggered it. Include the KB path or repo file path it operates on.

**Preconditions:** Any state that must be true before starting (prior tasks complete, files present, etc.). Write `none` if there are none.

**Steps:**
1. (Include specific tool calls where possible, e.g., `knowledge_base_read(path="content/...", sections=["..."])`)
2. ...

**Verification:** How to confirm the task is complete (e.g., a grep that should return one match, a status field that should read `done`).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md (from the `---` above the `##` header to the `---` below the last line).
````
