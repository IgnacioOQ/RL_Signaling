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

## Execute the notebook refactor plan (Phases 0–3 + Phase 5)
- status: todo
- type: task
- id: todo.notebook_refactor
- description: Execute Phases 0–3 and Phase 5 of [NOTEBOOK_REFACTOR_PLAN.md](NOTEBOOK_REFACTOR_PLAN.md) — migrate the six notebooks under `notebooks/` from the legacy `NetMultiAgentEnv` / `simulation_function` surface to the canonical `MultiAgentEnv` + `run_simulation` API, with the metadata, kernel, and dual local/Colab scaffolding required by the project's notebook conventions. Phase 4 (re-running the experiments to refresh `results/`) is deferred and tracked separately.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-15
<!-- content -->
**Context.** The six notebooks under [notebooks/](notebooks/) were authored before the seven-phase code refactor and still call the legacy API surface (`NetMultiAgentEnv`, `TempNetMultiAgentEnv`, `simulation_function`, `temp_simulation_function`), which now emits `DeprecationWarning`. Two of the six ([basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb), [Initializations_test.ipynb](notebooks/Initializations_test.ipynb)) were partly updated during the original refactor; the remaining four still target the legacy surface plus a Colab-only setup section.

[NOTEBOOK_REFACTOR_PLAN.md](NOTEBOOK_REFACTOR_PLAN.md) is the authoritative plan. It includes:

- A legacy → canonical API mapping cheat-sheet (read this before touching any notebook).
- Per-notebook refactor recipes (§2.1 through §2.6).
- Resolved decisions (2026-05-15): rename mapping accepted, `nbstripout` adopted in Phase 5, Phase 4 deferred, `04_parameter_optimization.ipynb` stays Colab-only.
- Tooling already in place: [notebooks/_tools/nb_migrate.py](notebooks/_tools/nb_migrate.py) (upgrade + audit subcommands) and [notebooks/NOTEBOOKS_README.md](notebooks/NOTEBOOKS_README.md).

**Preconditions:**
- The plan document and the helper tooling are committed (they were authored in the 2026-05-15 session that wrote this task).
- `pytest tests/ -v` passes on the current branch — the migration should not change package behavior, so the test suite is the regression net.

**Steps:**
1. Read [NOTEBOOK_REFACTOR_PLAN.md](NOTEBOOK_REFACTOR_PLAN.md) end-to-end. The §"Legacy → canonical API mapping" table is the cheat-sheet for every notebook edit.
2. Run the audit tool to confirm the starting state matches what the plan documents:
   ```bash
   python notebooks/_tools/nb_migrate.py audit notebooks/
   ```
   Expected: four notebooks show legacy-API hits and `NEEDS UPGRADE` metadata; two show clean state.
3. **Phase 0 (tooling) — already done** by the 2026-05-15 session. Skip.
4. **Phase 1 — rename + metadata pass.** Rename the six notebooks per the plan's table (numeric prefixes `01_` through `06_`, snake_case). Run `python notebooks/_tools/nb_migrate.py upgrade notebooks/` to bump every file to `nbformat=4.5`, set the `rl_signaling` kernel, and assign stable cell IDs. Update the six links in [README.md](README.md) (the **Notebooks** table and the **Reproducing the figures** section). The agent must **not** stage the renames with `git mv` per CODING_AGENT_MAIN_WORKFLOW rule 7 — write new files, delete old, let the user stage.
5. **Phase 2 — API migration**, one notebook at a time, per plan §2.1–§2.6. Use `NotebookEdit` with `cell_id=...` (see KB skill `content/how-to/NOTEBOOK_WRITING_SKILL.md` §8) so edits address cells by their stable IDs rather than by index. After each notebook, re-run `nb_migrate.py audit <file>` — it must report `legacy-API hits: none`. Pay attention to:
   - The return-tuple order change documented in the plan's cheat-sheet (`histories` and `nature_history` swap positions). Notebooks unpacking with `_, _, _, _, _` are unaffected; named-target unpacks need fixing.
   - The TD divergence caveat — TDLearningAgent under the canonical API drifts by ~1% from the legacy flow. This is expected and documented; do not chase byte-identical TD output.
   - `04_parameter_optimization.ipynb` stays Colab-only — introduce `RUNNING_LOCALLY` for consistency but do not pretend the full sweep runs on a laptop.
6. **Phase 3 — validation.** For each migrated notebook, Restart-and-Run-All on a fresh `rl_signaling` kernel with `SMOKE_TEST=True`. Confirm no errors, no `DeprecationWarning` from `rl_signaling.*`, and `nbformat.validate(nb)` succeeds. Run `pytest tests/ -v` — must stay at 63 passed.
7. **Phase 5 — documentation + nbstripout.** One-time setup at the repo level:
   ```bash
   pip install nbstripout
   nbstripout --install
   nbstripout --install --attributes .gitattributes
   ```
   Add `nbstripout` to the `[dev]` extras in [pyproject.toml](pyproject.toml). Update [README.md](README.md) to mention the `nbstripout --install` step in the Setup section. Update [notebooks/NOTEBOOKS_README.md](notebooks/NOTEBOOKS_README.md) to note the strip-on-commit convention. Append a `WORKLOG.md` entry summarizing the refactor.
8. **Phase 4 is out of scope for this task.** The plan file already records that decision. The separate `todo.verify_notebook_drive_paths` task (which depends on this one) covers the Drive-path verification needed before any Colab re-run.

**Verification:**
- `python notebooks/_tools/nb_migrate.py audit notebooks/` reports `legacy-API hits: none` and `nbformat=4.5 OK; kernel='rl_signaling' OK` for every notebook.
- `pytest tests/ -v` reports 63 passed.
- Each migrated notebook completes Restart-and-Run-All under `SMOKE_TEST=True` with no errors and no `DeprecationWarning` from the `rl_signaling.*` namespace.
- [README.md](README.md) **Notebooks** and **Reproducing the figures** sections reference the renamed files.
- [.gitattributes](.gitattributes) contains the `nbstripout` filter line and [pyproject.toml](pyproject.toml) `[dev]` extras include `nbstripout`.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry summarizing what changed and noting that `todo.verify_notebook_drive_paths` is now unblocked. Delete [NOTEBOOK_REFACTOR_PLAN.md](NOTEBOOK_REFACTOR_PLAN.md) from the repo root once Phase 5 is done (the plan was an in-flight document; the WORKLOG entry preserves the historical record).

---

## Verify Google Drive dump paths match the migrated notebooks
- status: todo
- type: task
- id: todo.verify_notebook_drive_paths
- description: After the notebook refactor wires every notebook to a single `BASE_PATH` constant under `RUNNING_LOCALLY=False`, confirm that the Drive layout the new notebooks expect actually exists in the user's Google Drive and that the legacy CSVs are still where the new code looks for them.
- owner: user
- blocked_by: [todo.notebook_refactor]
- last_checked: 2026-05-15
<!-- content -->
**Context.** The notebook refactor planned in [NOTEBOOK_REFACTOR_PLAN.md](NOTEBOOK_REFACTOR_PLAN.md) replaces the scattered Colab path strings (each notebook currently hard-codes its own `dump_path = '/content/drive/My Drive/Colab Projects/Python ABMs/Communication/Plots and Datasets/'`) with a single `BASE_PATH` constant derived from the KB notebook skill §7 pattern. Three of the six notebooks ([Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb), [Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb), [Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb)) read from / write to Drive when running on Colab.

The risk is that the *new* `BASE_PATH` chosen by the refactor (whatever value lands in the migrated notebooks) does not match the *actual* folder where the legacy CSVs live on Drive, and the Colab branch will either:
- Write into the wrong folder and silently fragment the result set, or
- Fail to find the inputs that [plotting_results.ipynb](notebooks/plotting_results.ipynb) expects when run on Colab.

Only the user can verify this — the agent has no access to the user's personal Google Drive without explicit MCP auth.

**Preconditions:**
- The notebook refactor (Phases 0–3 of [NOTEBOOK_REFACTOR_PLAN.md](NOTEBOOK_REFACTOR_PLAN.md)) has landed, so the new `BASE_PATH` values are committed somewhere reviewable.
- You have access to the same Google account whose Drive backs the historical Colab runs.

**Steps:**
1. Identify the `BASE_PATH` (or `dump_path`) constant used in each of the three Colab-targeted notebooks after the refactor:
   - `03_run_simulations.ipynb`
   - `04_parameter_optimization.ipynb`
   - `05_costly_signaling_simulations.ipynb`
   Run `grep -n "BASE_PATH\|dump_path" notebooks/*.ipynb` to surface every path expression.
2. Open Google Drive in a browser, signed in to the same account that hosts the historical runs. Confirm that the folder pointed to by each `BASE_PATH` exists.
3. For each Drive folder, list its contents and verify the expected CSVs are present. The canonical input set that `06_plotting_results.ipynb` consumes is:
   - `urnagent_results_canonical.csv`
   - `urnagent_results_complex_randomized.csv`
   - `qlearning_results_canonical.csv`
   - `qlearning_results_complex_randomized.csv`
   - `qlearning_results_canonical_costly_signal.csv`
   - `td_learning_results_canonical.csv`
   - `td_learning_results_complex_randomized.csv`
4. If the Drive layout differs from what the migrated notebooks expect, choose one:
   - **Migrate Drive to match the new code** — move/rename the existing folder so its path matches the new `BASE_PATH`.
   - **Update the code to match Drive** — edit the `BASE_PATH` constants in the migrated notebooks to match the actual Drive folder.
5. Smoke-test by opening `06_plotting_results.ipynb` in Colab with `RUNNING_LOCALLY=False`, mounting Drive, and running the data-load cells only. Confirm every CSV resolves.

**Verification:**
- Every `BASE_PATH` in the three Colab-targeted notebooks points to a folder that exists in Drive.
- Every CSV listed in Step 3 is reachable under each notebook's `BASE_PATH` on Colab.
- A Colab smoke-run of `06_plotting_results.ipynb` reaches at least the first plot without `FileNotFoundError`.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. If the path mismatch required code edits in the migrated notebooks, mention them in the `WORKLOG.md` entry for the notebook refactor.

---

## Update analytics/ doc cross-references after results/ reorganization
- status: todo
- type: task
- id: todo.update_doc_paths_after_results_reorg
- description: After the 2026-05-15 reorganization of results/ into legacy/{datasets,plots} + new_code/{datasets,plots} + proof_of_concept/, several markdown files under analytics/ still reference the pre-reorg top-level paths. Update each reference to point at the new subfolder location.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-15
<!-- content -->
**Context.** The 2026-05-15 session moved 50 files out of `results/` root into three subfolders:
- `legacy/datasets/` — 7 CSVs from pre-refactor experiment runs.
- `legacy/plots/` — 35 PNGs derived from the legacy CSVs (canonical / complex / costly experiments + parameter-optimization frontiers).
- `new_code/plots/` — 1 PNG so far (`figure_ql_vs_re_canonical.png`); `new_code/datasets/` reserved for the post-refactor re-run of the experiment notebooks (currently empty).
- `proof_of_concept/` — 7 PNGs (`initializations_*`, `initializations_urn_*`, `poc_optionA/B/C_*`).

The producer scripts under [analytics/scripts/](analytics/scripts/) and two notebooks ([Initializations_test.ipynb](notebooks/Initializations_test.ipynb), [Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb)) were updated to write to the new subfolders. The remaining items are doc cross-references in markdown files that still reference the old top-level `results/foo.png` and `results/foo.csv` paths. Independent of `todo.notebook_refactor`.

**Preconditions:**
- The reorganization is complete. Verify with `ls results/` — only `legacy/`, `new_code/`, `proof_of_concept/` (plus `.DS_Store`) should be present at the top level.

**Steps:**
1. Update [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof of Concept (Paper Draft).md):
   - `results/initializations_urn_rewards.png` → `results/proof_of_concept/initializations_urn_rewards.png`
   - `results/initializations_urn_nmi.png` → `results/proof_of_concept/initializations_urn_nmi.png`
   - `results/figure_init_paradox_scatter.{csv,png}` → `results/proof_of_concept/figure_init_paradox_scatter.{csv,png}`
   - In the Figure 3 sketch script block: `results/qlearning_results_canonical.csv` → `results/legacy/datasets/qlearning_results_canonical.csv`; `results/urnagent_results_canonical.csv` → `results/legacy/datasets/urnagent_results_canonical.csv`; `results/figure_ql_vs_re_canonical.png` → `results/new_code/plots/figure_ql_vs_re_canonical.png`.
2. Update [analytics/metrics_aggregation.md](analytics/metrics_aggregation.md): audit every `results/*.png` and `results/*.csv` reference and replace each with the appropriate `results/legacy/{datasets,plots}/...` path.
3. Update [analytics/costly_signaling.md](analytics/costly_signaling.md): references to `results/Roth-Erev_canonical_costly_signal_*.png` → `results/legacy/plots/Roth-Erev_canonical_costly_signal_*.png` (these PNGs are flagged in [LEGACY_ERRORS_LOG.md](LEGACY_ERRORS_LOG.md) as retired/unreproducible — keep the references, just fix the paths). Also: `results/q_costs_vs_nmi.png` and `results/q_costly_vs_reward.png` do not exist in `results/` at all — verify whether these refer to never-produced figures and either remove the references or file a separate task to produce them.
4. Audit [README.md](README.md) for `results/` paths that need updating to reflect the new subfolder structure (the **Reproducing the figures** section in particular).
5. Final verification grep:
   ```bash
   grep -rIn "results/[A-Za-z0-9_-]\+\.\(png\|csv\)" analytics/ README.md \
     | grep -v "results/legacy\|results/new_code\|results/proof_of_concept"
   ```
   Should return no matches.

**Verification:**
- The grep in Step 5 returns no matches.
- A spot-check in [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof of Concept (Paper Draft).md) confirms the markdown links resolve when clicked from VS Code.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a one-line `WORKLOG.md` entry recording the doc cross-reference cleanup.

---

## Finalize §2.3 figure selection for the philosophical paper
- status: todo
- type: task
- id: todo.finalize_section_2_3_figure
- description: User-side decision about which of the four candidate figures to commit to in §2.3 of [analytics/docs/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf](analytics/docs/Signaling_Games_with_Distributed_Rewards__Shortened_.pdf). Once chosen, the paper-ready draft needs to be revised from "audit + sketches" to a committed selection.
- owner: user
- blocked_by: []
- last_checked: 2026-05-15
<!-- content -->
**Context.** During the 2026-05-15 session the §2.3 paper draft was rewritten ([analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof of Concept (Paper Draft).md)) and three new candidate figures were generated under [results/proof_of_concept/](results/proof_of_concept/):
- `poc_optionA_phase_portrait.png` — (NMI_t, reward_t) trajectories per init regime, single-seed close-ups.
- `poc_optionB_cell_concentration.png` — per-cell hot-fraction trajectories under (1,1) and (5,1).
- `poc_optionC_absorbing_distribution.png` — marginal + joint distribution of mean reward over the 2304 absorbing states.

In addition, the existing post-fix re-run figures from [Initializations_test.ipynb](notebooks/Initializations_test.ipynb) live at:
- [results/proof_of_concept/initializations_urn_rewards.png](results/proof_of_concept/initializations_urn_rewards.png)
- [results/proof_of_concept/initializations_urn_nmi.png](results/proof_of_concept/initializations_urn_nmi.png)

The agent's recommendation in [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof of Concept (Paper Draft).md) §"Plot audit and figure sketches" was: keep the existing init sweep as Figure 1, add Option C as Figure 2, drop Option A (visually weak as rendered) and Option B (too technical for a philosophy paper).

**Decision needed:**
- Which figure(s) to commit to as the §2.3 figure(s).
- Once chosen, the paper draft should be revised to bake in the choice. The current draft presents A/B/C as candidates rather than committing.

**On completion:**
- If accepting the agent's recommendation: instruct an agent session to revise [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof of Concept (Paper Draft).md) to commit to "existing init sweep as Figure 1 + Option C as Figure 2," remove the multi-option discussion, and update the corresponding caption text. Then delete this task block.
- If choosing differently: leave a note here with the chosen combination and instruct an agent session to revise the draft accordingly. Then delete this task block.
- If deferring further: keep this task block and update `last_checked`.

---

## Re-render §2.3 figure candidates from the bug-fixed notebook, then pick the figures and rewrite §2.3

- status: in_progress
- type: task
- id: todo.finalize_section_2_3_with_figures
- description: Pick up the §2.3 work where the 2026-05-16 session left off — re-render every plot from the bug-fixed `proof_of_concept_figures.ipynb` (the basin-sweep figures and several others were produced under a `build_env_from_spec` bug that has since been fixed), then collaborate with the user to commit to a final set of figures for §2.3, then rewrite §2.3 of `Signaling_Games_with_Distributed_Rewards.pdf` using those figures and the empirical findings recorded in the paper draft.
- owner: agent + user (figure choice is the user's; everything else is agent-pickable)
- blocked_by: []
- last_checked: 2026-05-17

**Progress as of 2026-05-17 evening session** (see WORKLOG entry `§2.3 figure decisions: RE confirmed as headline...`):
- ✅ Re-rendered Option F (RE) and Option G (QL) horizon-sweep plots with the bug-fixed notebook.
- ✅ Figure-choice decision: **Roth-Erev confirmed as the §2.3 headline** (Figure 1 + Option A + Option F). Q-learning Option G is at best a robustness check or appendix figure, not a co-headline. Saved as [[feedback-paper-work]] Rule 7.
- ✅ `QLEARN_PARAMS` switched from tuned UCB to textbook ε-greedy in `_final` and `_aggregate` (defensible defaults for a philosophy-paper audience; saved as [[feedback-paper-work]] Rule 6).
- ✅ Notebook tree consolidated 7→3: `_final` (active), `_backup` (snapshot), `_aggregate` (catalog of all candidates).
- ⏳ **Next**: user re-runs Option G with the new ε-greedy params and decides keep-in-§2.3 / appendix / drop. If kept, the QL NMI inversion at short horizons may need a one-line caveat in the caption (finite-sample bias in the plug-in NMI estimator is the leading hypothesis).
- ⏳ **Then**: agent rewrites §2.3 of the PDF using the chosen figures, addressing R2·C1, R2·C2, R3·C2 from [analytics/docs/Generated Responses to Reviewers.md](analytics/docs/Generated%20Responses%20to%20Reviewers.md). Current draft prose in [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof%20of%20Concept%20(Paper%20Draft).md) is largely usable but needs to drop the "candidate framing" and commit to the chosen figures.

The original task body below is retained for the broader scope (the rewrite phase is unchanged):

<!-- content -->
**Context.** The 2026-05-16 session built [notebooks/proof_of_concept_figures.ipynb](notebooks/proof_of_concept_figures.ipynb) (31 cells) — a single place that renders every candidate figure for §2.3 of *Signaling Games with Distributed Rewards*. The notebook produced rendered plots on Colab. Late in the session a bug was caught in `build_env_from_spec` (shared `action_urns` dict across both agents, fixed by creating a fresh table per agent inside the loop), so several already-rendered plots — including the headline Option D-β basin sweep — were produced with broken dynamics. The fix is in the current notebook; a re-run reproduces correct dynamics. See the [WORKLOG entry](WORKLOG.md) `2026-05-16 — Built the §2.3 proof-of-concept figures notebook` for the full session record.

The user explicitly deferred figure choice to a future session: "I have to do something else now, so I want to make sure all of the relevant information is stored for a next session. In that session we will examine the plots, and continue with them if necessary. We will make a choice on which plots to use. After that, we will draft again the proof of concept section."

This task supersedes the scope of `todo.finalize_section_2_3_figure` (which was filed before the new notebook existed and refers to older candidate figures). Delete that block once this one is acted on.

**Preconditions:**

- Notebook [notebooks/proof_of_concept_figures.ipynb](notebooks/proof_of_concept_figures.ipynb) builds (`nbformat.validate` passes) and runs end-to-end on Colab with `RUNNING_LOCALLY = False`. Build script at [notebooks/_tools/build_poc_notebook.py](notebooks/_tools/build_poc_notebook.py) is the round-trip source.
- The bug fix in `build_env_from_spec` (fresh `create_initial_signals` call per agent inside the loop) is in place — verify by grepping the setup cell for `new_action_table = create_initial_signals(`; it must appear *inside* the `for agent in env.agents` loop, not before it.
- The user has the rendered PNGs on Drive at `/content/drive/My Drive/Colab Projects/Python ABMs/Distributed Signaling/Plots and Datasets/Proof of Concept/`. The local laptop copies in [results/proof_of_concept/](results/proof_of_concept/) may be stale from the buggy run — the local ones called `Option D-beta 1.png` (pre-bug, correct shape) and `Option D-beta 2.png` (with-bug, flat reward) document the bug's empirical signature.

**Steps:**

1. **Re-render every plot from the bug-fixed notebook.** On Colab with `RUNNING_LOCALLY = False`, Restart-and-Run-All. The `BASIN_N_SEEDS = 200` Colab tier gives tight error bands on the basin sweep; bump higher if the histograms still look noisy. Expected total runtime: 15–25 minutes on a normal Colab box.

2. **Eyeball each candidate against the predictions in the paper draft note.** Open [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof%20of%20Concept%20(Paper%20Draft).md) — the "Observation — NMI vs reward dissociation in `sig=[5,1]` vs `sig=[1,1]` (2026-05-16)" section makes specific empirical predictions (orange bimodality, green wider than orange, Q-learning flatter than Roth–Erev). Confirm or refute each. The orange bimodality finding is the strongest piece of evidence for the §2.3 honest framing and should not be lost.

3. **Pick the figure(s) for §2.3.** Six candidates: Figure 1 (init sweep), Figure 2 (per-seed scatter), Option A (phase portrait), Option B (per-cell hot-fraction), Option C (absorbing-state distribution), Option D (α/β/γ — basin sweeps), Option E (Roth–Erev vs Q-learning). The user has expressed:
   - **Liked**: Option D-β (the basin reach with reward + NMI overlay was called "pretty good"), Option C ("Reward distribution over the 2304 absorbing states" — called "quite interesting"), Option A ("quite interesting").
   - **Maybe-not**: Figure 2 ("not sure I want to use figure 2", but the orange-bimodality finding should be mentioned in §2.3 prose if dropped — see the paper-draft note's "Figure 2 — status (undecided)" subsection).
   - **Notebook-only**: Option B was flagged as too technical for the philosophy paper.
   The user reads philosophy and game-theory contexts; figure budget should be conservative.

4. **Commit to chosen figures in the paper draft.** Edit [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof%20of%20Concept%20(Paper%20Draft).md): remove the "Plot audit and figure sketches" section's "candidate" framing, replace with the committed selection, write the final captions, and update the "Figure 2 — status (undecided)" subsection of the 2026-05-16 observation note to "committed" (or note that Figure 2 was dropped and the finding migrated to §2.3 prose).

5. **Rewrite §2.3 of `Signaling_Games_with_Distributed_Rewards.pdf`** using the committed figures. The current draft prose in [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof%20of%20Concept%20(Paper%20Draft).md) is largely usable but needs two updates:
   - **Anchor the philosophical content on the lock-in vs co-adaptation finding** (the continuous version of the old `(1, 0)` paradox) — this is the cleanest empirical evidence for "high NMI ≠ successful communication."
   - **Promote the orange bimodality** to a load-bearing observation — it's the strongest empirical witness for "attractors exist but the basin is not provably reached" (the framing the reviewer responses commit §2.3 to).
   - The reviewer-response checklist in [analytics/docs/Generated Responses to Reviewers.md](analytics/docs/Generated%20Responses%20to%20Reviewers.md) (R2·C1, R2·C2, R3·C2) is the audit list — every checklist item should be addressed in the new §2.3.

6. **Update [Proof of Concept (Paper Draft).md](analytics/docs/Proof%20of%20Concept%20(Paper%20Draft).md) header** with a date marker once the rewrite is incorporated into the PDF, so future sessions know which paragraphs are stale.

**Verification:**

- Every figure that ends up in §2.3 of the PDF has a matching PNG under [results/proof_of_concept/](results/proof_of_concept/) (or its Drive equivalent) and a citation in the paper draft.
- A grep of [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof%20of%20Concept%20(Paper%20Draft).md) for "candidate", "sketch", "undecided", "consider" returns no matches in the main §2.3 prose (acceptable in marginal notes only).
- The §2.3 PDF rewrite addresses each of the three reviewer concerns landing on §2.3 (R2·C1, R2·C2, R3·C2) — confirmed by a brief review against [analytics/docs/Generated Responses to Reviewers.md](analytics/docs/Generated%20Responses%20to%20Reviewers.md).

**On completion:**

- Delete this entire task block from `TODO_WORKFLOW.md`.
- Also delete `todo.finalize_section_2_3_figure` (the older user-side task that this supersedes).
- Append a `WORKLOG.md` entry recording the chosen figures, the §2.3 rewrite, and any further empirical findings that surfaced during the re-render.

**Pointers:**

- **Live notebook** — [notebooks/proof_of_concept_figures.ipynb](notebooks/proof_of_concept_figures.ipynb)
- **Build script (round-trip source)** — [notebooks/_tools/build_poc_notebook.py](notebooks/_tools/build_poc_notebook.py)
- **Paper draft (under active rewrite)** — [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof%20of%20Concept%20(Paper%20Draft).md)
- **Reviewer responses + checklist** — [analytics/docs/Generated Responses to Reviewers.md](analytics/docs/Generated%20Responses%20to%20Reviewers.md)
- **Empirical findings note (this session)** — [analytics/docs/Proof of Concept (Paper Draft).md](analytics/docs/Proof%20of%20Concept%20(Paper%20Draft).md) — section "Observation — NMI vs reward dissociation…"
- **Combinatorial companion** — [analytics/docs/Urn Absorbing States.md](analytics/docs/Urn%20Absorbing%20States.md)
- **Formal Markov-chain reference** — [analytics/proof_of_concept_markov.md](analytics/proof_of_concept_markov.md)
- **Drive root for Colab artifacts** — `/content/drive/My Drive/Colab Projects/Python ABMs/Distributed Signaling/Plots and Datasets/Proof of Concept/` (documented in [notebooks/NOTEBOOKS_README.md](notebooks/NOTEBOOKS_README.md))

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
