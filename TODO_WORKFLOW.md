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

## Investigate why QLearning fails to lock into pre-seeded equilibria
- status: todo
- type: task
- id: todo.investigate_qlearning_initialization
- description: The Initializations experiment is supposed to test whether agents systematically biased toward one of the multiple pure-strategy equilibria of the canonical signaling game lock in to that equilibrium under different bias strengths. UrnAgent shows the expected behavior; QLearningAgent does not. Diagnose why and propose a fix.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-09
<!-- content -->
**Context — what the study is for.** The canonical signaling game has many pure-strategy equilibria. Each equilibrium is a *bijection* from world-state observations to signals (the sender's encoding) composed with a bijection from `(own_obs, received_signal)` to final actions (the receiver's decoding) such that the realized action matches the game's optimal action for the full state. The `Initializations_test.ipynb` notebook varies `init_weights ∈ {[1,0], [1,1], [5,1], [100,1]}` to ask: *if we pre-seed agents with a strong systematic bias toward one of these candidate equilibria, does the bias persist through learning?* The pre-seed mechanism is `rl_signaling.games.create_initial_signals(n_observed_features, n_signals, n, m)` ([rl_signaling/games.py:115-160](rl_signaling/games.py#L115-L160)) which returns a deterministic bijection observation → unique one-hot vector with weight `n` (= `init_weights[0]`) at the hot position and `m` (= `init_weights[1]`) at the others. Increasing the magnitude of `n` should produce a stronger initial pull toward whatever bijection the random shuffle picked — *that is the intended mechanism of the experiment*, not a confounder.

**What works — UrnAgent.** The 2026-05-09 re-run of `notebooks/Initializations_test.ipynb` (post-Bug-5 / post-Bug-4 fix; figures at [results/initializations_urn_rewards.png](results/initializations_urn_rewards.png) and [results/initializations_urn_nmi.png](results/initializations_urn_nmi.png)) shows the expected pattern:

- `[1, 0]` → NMI ≈ 1.0 throughout, reward stuck at ≈ 0.25 — agents lock into a *strong* (deterministic) signaling protocol but the pre-seeded action map happens to be wrong on this seed; perfect signaling, mismatched receiver decoding.
- `[1, 1]` → NMI ≈ 0.05 with reward ≈ 0.85 — uniform pre-seed, ordinary learning, near-optimal payoff.
- `[5, 1]` → NMI ≈ 0.93, reward ≈ 0.85 — moderate bias.
- `[100, 1]` → NMI ≈ 0.90, reward climbs 0.20 → 0.95–1.0 across 30 000 episodes — strong but recoverable.

The `[1, 0]` case is the cleanest demonstration of the design working: the urn *cannot* assign positive count to any cold action (`m = 0`), so the pre-seeded bijection persists indefinitely. This is exactly the "stick with one of the candidate equilibria" phenomenon the experiment is meant to probe.

**What doesn't work — QLearning.** The same notebook's QLearning block produces [results/initializations_rewards.png](results/initializations_rewards.png) and [results/initializations_nmi.png](results/initializations_nmi.png). All four `init_weights` curves bounce around the random-action baseline (reward ≈ 0.20–0.30 = 1/`n_final_actions`) for the entire 30 000-episode run. NMI starts at ~0.7 (consistent with the pre-seeded bijection being briefly visible in the first ~100 episodes) then collapses to ≈ 0 by episode 500. Even `init_weights=[100,1]` shows no lock-in, no slow drift toward an equilibrium — the bias is washed out almost immediately. **This is the suspicious behavior** that this task investigates.

**Why this is suspicious — the asymmetry between agents.** Inspecting the update rules clarifies the structural reason but does *not* fully explain why the QL collapse is so fast and so total:

- `UrnAgent.update_signals` / `UrnAgent.update_actions` ([rl_signaling/agents.py:240-243+](rl_signaling/agents.py#L240-L243)): `urn[s][a] = max(0, urn[s][a] + reward)`. Updates are **positive-only and additive**. A pre-seeded `urn[s][hot] = 100` stays at 100 forever during reward-0 episodes; reward-1 episodes only grow it further. The cold actions can grow when explored and rewarded, but the hot action's bias never erodes.
- `QLearningAgent.update_signals` / `update_actions` ([rl_signaling/agents.py:462-479+](rl_signaling/agents.py#L462-L479)): `Q[s][a] += 0.1 * (reward - Q[s][a])`. Updates are **multiplicative TD** with a constant learning rate. A pre-seeded `Q[s][hot] = 100` decays toward the observed reward at rate $0.9^n$ per visit. With reward zero ~75 % of the time on a random pre-seed, `Q[hot]` decays to ≈ 0.25 within ~50 visits. The pre-seeded magnitude is **erased** by the TD update itself, regardless of UCB exploration.

So the asymmetry is real, but it's worth being precise about *which* observation it explains. The TD-decay of `Q[hot]` toward observed reward is one piece. Whether that alone is sufficient to explain the *complete* erasure across all `init_weights` is what this task should determine.

**Three live hypotheses for QLearning's failure:**

1. **TD-decay erases the pre-seed.** With `α = 0.1` hard-coded and no compensating mechanism, `Q[hot] = 100` is mathematically guaranteed to decay to ≈ `mean_reward` within ~50 visits regardless of which equilibrium that pre-seed was supposed to encode. The "magnitude" of the pre-seed (the 100 in `[100, 1]`) does not produce a persistent bias the way the urn count does — because Q values are bounded by the observed-reward range `[0, 1]` and the update pulls toward that range. Implication: if true, `QLearningAgent` as currently designed cannot exhibit the lock-in phenomenon UrnAgent shows. The fix would have to change either the update rule (e.g. exponential smoothing, or a positive-only-clamped variant), the learning rate (smaller `α` or count-based `1/N`), or the meaning of `init_weights` for QL (e.g. use it to set initial `signaling_counts` / `action_counts` rather than Q values, mimicking the count-as-bias structure of the urn).

2. **The Bug 4 fix (symmetric pre-seed of both Q-tables) breaks the experiment.** The 2026-05-09 session applied Bug 4's "symmetric" interpretation, mirroring the Bug 1 fix on `UrnAgent` so that `QLearningAgent.__init__` pre-seeds both `q_table_signaling` *and* `q_table_action`. With the action Q-table pre-seeded with random hot positions per `(own_obs, received_signal)` key, the action policy is initially noisy from the receiver's perspective: roughly 1 of 4 keys' pre-seeded action coincides with the optimal action for the corresponding world state. The other 3 keys produce reward 0, dragging all Q values toward the action-key-averaged reward of ≈ 0.25 and feeding noise back into the signal-phase Q-update (because the signaling reward is the same as the action reward in this single-step game). Implication: the asymmetric pre-seed (signaling Q-table only, action Q-table left empty / lazy-init to zero) might preserve the experiment's intent. The pre-Bug-4 docstring described the asymmetric behavior; the user confirmed "symmetric" during Batch B but may not have anticipated this interaction. Note: this is *not* a strict alternative to Hypothesis 1 — even with asymmetric pre-seed, TD-decay still erases the signaling Q-table's pre-seed at the same rate.

3. **`exploration_rate` interaction.** The notebook injects tuned hyperparameters from `Parameter_Optimization_wchoices.ipynb`: `exploration_rate=0.965, exploration_decay=0.9998, min_exploration_rate=1e-10, choice='ucb'`. UCB's bonus is `exploration_rate * sqrt(log(t)/(N + 1e-5))`, which is *enormous* on unvisited actions (`N=0` → bonus ≈ `0.965 * sqrt(log(t) / 1e-5)`). Even with `Q[hot] = 100`, the UCB bonus on a cold action with `N=0` reaches ≈ 254 by `t=2`, forcing an immediate explore-all-actions sweep. After all four actions are visited once, counts equalize and the bonus drops to ~1, but by then the pre-seed has already been touched by reward-0 updates that pull every cell toward zero. Implication: tuned hyperparameters were optimized against `initialize=False` (per the Bug 5 history) and may be incompatible with the `initialize=True` regime; running the QL block at QLearningAgent's *default* hyperparameters might preserve the bias.

These hypotheses are not mutually exclusive. The investigation should distinguish their magnitudes empirically.

**Cross-references:**
- Migrated notebook: [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) (rewritten 2026-05-09 to canonical `MultiAgentEnv` + `agent_kwargs`, Urn / QLearning split, paired-comparison seeding).
- Pre-seed function: [rl_signaling/games.py:115-160](rl_signaling/games.py#L115-L160) (`create_initial_signals`).
- Game generator: [rl_signaling/games.py:55-102](rl_signaling/games.py#L55-L102) (`create_random_canonical_game`).
- Agents: [rl_signaling/agents.py](rl_signaling/agents.py) — `UrnAgent.__init__` ([:202-238](rl_signaling/agents.py#L202-L238)), `UrnAgent.update_*` (positive-only-clamped urn updates, search "max(0,"); `QLearningAgent.__init__` ([:371-417](rl_signaling/agents.py#L371-L417)), `QLearningAgent.update_signals` / `update_actions` ([:462-499](rl_signaling/agents.py#L462-L499)) (constant α=0.1 TD updates).
- UCB selection: [rl_signaling/agents.py:105-114](rl_signaling/agents.py#L105-L114).
- Math reference: [analytics/agent_q_learning.md](analytics/agent_q_learning.md) (Q-update closed form $Q_n = Q_0(1-\alpha)^n + r(1-(1-\alpha)^n)$ derived and verified to atol=1e-12 in Phase 4).
- Prior session's write-up: `LEGACY_BUGS_LOG.md` Bug 5 → "Post-fix observation (Phase 6, 2026-05-09)" — note the framing there ("biased toward an arbitrary action vs informed toward the optimal action") was *one* candidate explanation; it is now superseded by the framing in this task (the bias **is** the experimental treatment, and the question is why QL doesn't preserve it).
- Pre-fix figure backup (if `/tmp` still has it): `/tmp/rl_signaling_prefix_backup/initializations_{rewards,nmi}.png`. If gone, recover the pre-fix figures from `git log` history before the 2026-05-09 commit.

**Preconditions:**
- `notebooks/Initializations_test.ipynb` is at the post-Bug-5 / post-Bug-4 state (2026-05-09 commit).
- `pytest tests/` reports 61 passed.
- The four post-fix figures `results/initializations_{,urn_}{rewards,nmi}.png` are present and reflect the 2026-05-09 re-run.

**Steps:**

1. **Read for context.** Before any code, read end-to-end: `LEGACY_BUGS_LOG.md` Bug 4 (Fix applied + Post-fix observation), Bug 5 (Fix applied + Post-fix observation), `analytics/agent_q_learning.md` (the constant-α closed form), `analytics/agent_urn.md` (the positive-only-clamped urn update). Hold the asymmetry firmly in mind: urn updates preserve high pre-seeded values through reward-0 episodes; TD updates do not.

2. **Test Hypothesis 1 in isolation — TD-decay alone.** Construct a minimal reproduction: a single `QLearningAgent` instance with `initialize=True, init_weights=[100, 1], n_observed_features=1, n_signaling_actions=2, n_final_actions=4`, default hyperparameters (no tuned overrides). Drive it with a deterministic environment that always returns reward 0 regardless of action. Assert that `Q[hot]` decays from 100 toward 0 at rate $0.9^n$ per visit (the closed-form prediction). Then drive it with a deterministic environment that always returns reward 1 *only* when the agent picks action `0`. Track `Q[hot]` for each `(obs, received_signal)` cell — does the bias persist when the pre-seeded hot action happens to be 0? Does it decay when it doesn't? This isolates the TD-decay mechanism from UCB exploration and from the canonical game's reward sparsity.

3. **Test Hypothesis 2 in isolation — symmetric vs asymmetric pre-seed.** Without modifying the codebase, write a temporary fork of `QLearningAgent.__init__` (or a subclass override) that pre-seeds **only** `q_table_signaling` and leaves `q_table_action` empty / lazy-init to zeros. Re-run the QL block of `Initializations_test.ipynb` using this asymmetric variant. Compare the four resulting curves to the symmetric-pre-seed ones in `results/initializations_rewards.png`. If the asymmetric variant produces visible separation across `init_weights` (similar to UrnAgent's pattern), Hypothesis 2 is confirmed and Bug 4's "symmetric" decision should be reversed. If both variants are stuck, Hypothesis 1 dominates.

4. **Test Hypothesis 3 in isolation — default hyperparameters.** Re-run the QL block of `Initializations_test.ipynb` with `QLearningAgent`'s constructor *defaults* instead of the tuned `Parameter_Optimization` values: `exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.001`. Compare to the tuned-hyperparameter run. If the defaults preserve the bias, Hypothesis 3 contributes; if not, the tuned hyperparameters are not the cause.

5. **Counter-test — match UrnAgent dynamics structurally.** Implement a "positive-only-clamped Q variant" by either (a) wrapping `update_signals` / `update_actions` to apply `Q[s][a] = max(Q[s][a], Q[s][a] + 0.1*(r - Q[s][a]))` (i.e. block decreases — TD only allowed to grow Q), or (b) using `exp_smoothing=True` (already supported by `QLearningAgent`) which gives `Q ← (1-α)*Q + α*r` — the *same* form as the standard TD update mathematically, so this is **not** a fix on its own; it just confirms `exp_smoothing` reproduces the broken behavior. Document whether (a) restores the bias-persistence pattern. If yes, the structural reason for QL's failure is the unconstrained TD-decay, and the fix space includes either a positive-only Q variant or a redesign of `init_weights` to seed something other than Q values.

6. **Reframe `init_weights` for QLearning if hypotheses 1 & 5 are confirmed.** If TD-decay is the proximate cause and a positive-only Q variant feels like a hack, consider alternative interpretations: pre-seed the *visit counts* `signaling_counts` / `action_counts` rather than (or in addition to) the Q values, biasing the UCB bonus toward the pre-seeded equilibrium without giving the agent unrealistic Q magnitudes; or pre-seed Q values at a more modest scale (e.g. `[1, 0]` only, treating `init_weights` as a probability-shaping factor that gets applied through softmax / boltzmann selection rather than directly stored as Q). Each of these is a redesign rather than a bug fix; document the tradeoffs.

7. **Land the chosen fix.** Whichever option survives the analysis, implement it as a small, focused change with:
   - A clear `### Why this fix` paragraph in the relevant code's docstring (`QLearningAgent.__init__`) explaining what `init_weights` now means for QL and how it relates to the experimental intent.
   - A unit test in `tests/test_agents.py` asserting the bias-persistence property (e.g. "after 50 reward-0 episodes on a single state, `Q[hot] - Q[cold]` is still ≥ X for `init_weights=[100, 1]`").
   - A re-run of `notebooks/Initializations_test.ipynb` and a fresh round of `### Post-fix observation` updates on `LEGACY_BUGS_LOG.md` Bug 4 and Bug 5.
   - A WORKLOG entry documenting the diagnosis, the fix, and the new figure outcomes.

**Verification:**
- One of the three live hypotheses is empirically confirmed (or all three are partially confirmed with documented relative magnitudes).
- The QLearning block of `Initializations_test.ipynb` produces curves that show meaningful separation across `init_weights` — at minimum, `[1, 0]` should lock into a deterministic policy (whether reward-1 or reward-0.25) the way UrnAgent's `[1, 0]` does, and `[100, 1]` should show a slow drift rather than instant collapse.
- The fix (whatever it is) does not regress the UrnAgent block.
- `pytest tests/` still passes.
- `tests/test_golden.py` still byte-identical (the golden baseline uses `initialize=False`, so any fix to the `initialize=True` path should leave it untouched).

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry recording the diagnosis, the chosen fix, and the regenerated figure outcomes.

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
