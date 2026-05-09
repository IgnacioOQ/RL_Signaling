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

## Sanity-check the Phase 6 Initializations finding
- status: todo
- type: task
- id: todo.sanity_check_initializations_finding
- description: Verify the surprising Phase 6 finding that init_weights=[1,0] yields UrnAgent NMI≈1.0 with reward stuck at ≈0.25, and that QLearning under all four init_weights settles at random-baseline reward.
- owner: agent
- blocked_by: []
- last_checked: 2026-05-09
<!-- content -->
**Context:** The 2026-05-09 Phase 6 verification re-ran the migrated `notebooks/Initializations_test.ipynb` end-to-end after Bugs 4 and 5 were fixed. The regenerated figures show:

- **UrnAgent `[1, 0]`** → NMI ≈ 1.0 throughout, reward stuck at ≈ 0.25 (random-baseline for `n_final_actions = 4`).
- **UrnAgent `[1, 1]`** → NMI ≈ 0.05, reward ≈ 0.85.
- **UrnAgent `[5, 1]`** → NMI ≈ 0.93, reward ≈ 0.85.
- **UrnAgent `[100, 1]`** → NMI ≈ 0.90, reward climbing 0.20 → 0.95–1.0 across 30 000 episodes.
- **QLearningAgent** all four init_weights → reward stuck at ≈ 0.20–0.30 throughout; NMI spikes briefly (~0.7) then collapses to ≈ 0 by episode 500.

The detailed write-up is in `LEGACY_BUGS_LOG.md` Bug 5 → "Post-fix observation (Phase 6, 2026-05-09)". The interpretation offered there — that `create_initial_signals` produces *biased* (toward an arbitrary action) rather than *informed* (toward the optimal action) priors, which inverts the original Bug 1 prediction — is a strong claim and was made on visual evidence alone. It bundles three separate hypotheses that should be teased apart before treating the framing as authoritative:

1. **The pre-seed pattern.** What does `rl_signaling.games.create_initial_signals(n_observed_features, n_signals, n, m)` actually produce? Is it deterministic per `(n_observed_features, n_signals, n, m)`? Does the "hot" position vary by observation key, or is it the same column for every key?
2. **The game's optimal action map.** For the seed used in `Initializations_test.ipynb` (`np.random.seed(0); random.seed(0)` per iteration), what `(state) → optimal_action` table does `create_random_canonical_game(2, 4, n=1, m=0)` produce? Does it coincide cell-wise with the pre-seeded "hot" action chosen by `create_initial_signals(n_observed_features+1, n_final_actions, ...)`?
3. **The QLearning stuck-at-baseline behavior.** Even granting the biased-prior framing, the closed-form $Q_n = Q_0 (1-\alpha)^n + r (1 - (1-\alpha)^n)$ at $\alpha = 0.1$ pulls $Q$ toward $r$ at rate $0.9^n$ — a Q[hot] = 100 prior should decay below 1.0 within ~44 reward-zero updates. The fact that *all four* QLearning curves are stuck suggests the agent isn't sampling cold actions enough times for that decay to apply, not that the decay itself is too slow. This is a UCB-exploration question, separate from the prior question.

**Preconditions:**
- `notebooks/Initializations_test.ipynb` is at the post-Bug-5 / post-Bug-4 state (committed in the 2026-05-09 fix session).
- `pytest tests/` reports 61 passed.
- The four post-fix figures `results/initializations_{,urn_}{rewards,nmi}.png` are present and reflect the 2026-05-09 re-run.
- Pre-fix backups live at `/tmp/rl_signaling_prefix_backup/initializations_{rewards,nmi}.png` (these are session-local; if the OS has cleared `/tmp` since 2026-05-09, recover them from `git log` history before the 2026-05-09 commit).

**Steps:**

1. **Pin down `create_initial_signals` behavior.** Read [rl_signaling/games.py](rl_signaling/games.py)'s `create_initial_signals` end-to-end. Document the actual pattern (deterministic vs RNG-driven? per-key hot-position selection?) and add a unit test in `tests/test_games.py` if one is not already there — mirror the assertion shape already in `tests/test_agents.py::test_urn_agent_initialize_true_seeds_action_urns`.

2. **Compare prior vs optimal map.** Construct the canonical game used by `Initializations_test.ipynb` exactly: `np.random.seed(0); random.seed(0); create_random_canonical_game(2, 4, n=1, m=0)` for each of the two agents. For each `(obs, received_signal)` key, compute the agent-side optimal action by enumerating the full nature_vector consistent with `obs`, looking up the rewards from the game dict, and identifying the action that maximizes expected reward. Compare cell-by-cell against the pre-seeded "hot" action chosen by `create_initial_signals(n_observed_features=2, n_signals=4, n=1, m=0)`. Tabulate matches vs mismatches.

3. **Dump the UrnAgent `[1, 0]` end-of-training state.** Re-run the `[1, 0]` UrnAgent leg with `n_episodes=30000, np.random.seed(0); random.seed(0)`. After training, dump:
   - `agents[0].signaling_urns` and `agents[1].signaling_urns` — are they concentrated on a single signal per observation? (They should be — NMI ≈ 1 implies a deterministic signaling code modulo rare exploration.)
   - `agents[0].action_urns` and `agents[1].action_urns` — are they concentrated on the pre-seeded "hot" action regardless of received signal? (Expected if the framing is correct.)
   - For each agent, the per-state action choice frequency averaged over the last 100 episodes, and compare to the game's optimal action per state.

4. **Dump the QLearning `[100, 1]` end-of-training state.** Re-run with `n_episodes=30000, np.random.seed(0); random.seed(0)`. After training, dump:
   - `agents[0].q_table_action` values per `(obs, received_signal)` key — quantify how many cells still carry the 100-valued pre-seed (i.e. were never updated) vs decayed to ≈ reward-mean.
   - `agents[0].action_counts` per key — for each key, count how many times each action was selected. If cold actions show $N \approx 0$ at episode 30 000 while the hot action shows $N \approx 7500$, the agent never explored — that confirms the UCB-exploration explanation.
   - Compute the UCB bonus $c \cdot \sqrt{\log(t) / N(a))}$ at $t = 30000$ for $N(\text{cold}) = 0$ vs $N(\text{hot}) = 7500$. Determine whether the bonus could plausibly flip the selection given the post-pre-seed Q-spread.

5. **Counter-experiment — informed prior.** Construct an `informed` pre-seed that points at the game's actual optimal action per `(obs, received_signal)` key (computed from the per-iteration `randomcanonical_game[i]`). Inject it via either (a) a custom `create_initial_signals`-shaped helper that takes the optimal-action map as an argument, or (b) direct `agents[i].q_table_action[key] = …` / `urns[key] = …` assignment after env construction. Re-run the `init_weights = [100, 1]` comparison for both UrnAgent and QLearning agents, against the `create_initial_signals` (uninformed/biased) version. If the informed-prior version converges fast to reward ≈ 1.0 while the biased version stays near 0.25, that is direct evidence the framing is correct and the original Bug 1 prediction implicitly assumed an informed prior.

6. **Document the verdict.** Write the findings as either:
   - A new `### Sanity-check follow-up (Phase 6, YYYY-MM-DD)` subsection appended to `LEGACY_BUGS_LOG.md` Bug 5's "Post-fix observation" section, with a clear verdict on the biased-vs-informed-prior question and on the UCB-exploration-stuck-at-cold-Ns question.
   - Or a new `analytics/initialization_priors.md` if the analysis grows large enough to warrant its own document (formal treatment of how `create_initial_signals` interacts with `create_random_canonical_game` under the QLearning vs UrnAgent dynamics).

**Verification:**
- The unit test for `create_initial_signals` structure passes.
- The biased-vs-informed prior verdict is recorded with concrete evidence (dumps of urns / Q-tables, per-action visit counts, per-state action-frequency comparisons against the game's optimal map).
- The counter-experiment outcome is recorded with paired-seed final-reward / final-NMI comparisons across `informed` vs `create_initial_signals` priors.
- If the framing is corroborated, the existing `LEGACY_BUGS_LOG.md` Bug 5 Post-fix observation either (a) stands as written (already framed as a hypothesis), or (b) is amended with the corroborated finding and concrete numbers.
- If the framing is refuted, the Phase 6 write-up is corrected: identify the actual cause of the stuck-at-baseline QLearning result (e.g. exploration-rate decay schedule, UCB tie-break behavior, action-count seeding) and retract the biased-prior framing.

**On completion:** Delete this entire task block from TODO_WORKFLOW.md. Append a WORKLOG entry recording the verdict and any code or document changes that resulted.

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
