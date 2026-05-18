# RL_Signaling
- status: active
- type: how-to
- id: rl_signaling.readme
- description: Reinforcement-learning study of emergent signaling between agents under partial observation; covers the model, repository layout, setup, and a minimal runnable example.
- label: [core]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-05-18
<!-- content -->

A reinforcement-learning study of emergent signaling between agents under partial observation. Two or more agents on a directed graph each observe a subset of the world state, exchange signals, then take an action whose payoff depends on the full state. The question is whether — and when — meaningful communication emerges, even though each agent's payoff is independent of the others' actions.

## Model

- **World state.** A binary vector of `n_features` random variables.
- **Observations.** Each agent sees a fixed subset of those features (`agents_observed_variables`).
- **Signals.** Agents on a directed graph emit a signal from a set of size `n_signaling_actions`. Signals are passed to in-neighbours.
- **Actions.** After receiving signals, each agent picks a final action from a set of size `n_final_actions`.
- **Payoff.** A per-agent game dictionary `G: state → action → reward`. Payoffs depend on the full state, not just on what the agent observed, so cooperative signaling is required to reach the optimum.

Three information regimes are compared in the experiments:

1. **Full information** — every agent observes the full state, no signaling needed.
2. **Partial information, no signals** — each agent sees only its subset.
3. **Partial information with signals** — each agent sees its subset plus the signals it receives.

### Costly signaling

When `costly_signaling=True`, an extra "null signal" action is added; sending any non-null signal incurs a per-agent cost (`signal_cost`). This setup is used to study when signaling survives a price tag.

## Repository layout

```
rl_signaling/                   # the package
  __init__.py                   # public surface
  agents.py                     # BaseAgent ABC + UrnAgent, QLearningAgent, TDLearningAgent
  env.py                        # MultiAgentEnv (canonical) + deprecated NetMultiAgentEnv, TempNetMultiAgentEnv
  simulation.py                 # run_simulation (canonical) + deprecated simulation_function, temp_simulation_function
  games.py                      # canonical & random game generators, signal-urn initializers
  info_theory.py                # mutual-information / NMI metrics
  plotting.py                   # plot helpers, post-processing utilities, plot_simulation_summary
notebooks/                      # experiment notebooks (see table below)
analytics/                      # mathematical reference (every quantity the package computes) + independent verification scripts
  math/                         # math files: notation, info_theory, signaling_model, agents, proof_of_concept_markov, the §2.3 markdown drafts, and the authoritative roth_erev_polya_mle.md reference
  scripts/                      # standalone PASS/FAIL verification scripts (see analytics/scripts/SCRIPTS_README.md)
manuscript/                     # paper sources for PHOS-17993 ("Signaling Games with Distributed Rewards")
  main.tex                      # original paper source (pre-§2.3-rewrite)
  main_v2.tex                   # working copy of main.tex with the new §2.3; compiles to main_v2.pdf
  Appendix.tex                  # standalone appendix document (own \documentclass); compiles to Appendix.pdf
  section_2_3.tex               # standalone LaTeX fragment of §2.3 prose (reference copy)
  References.bib                # BibTeX file (shared cross-paper library, ~28 cited from main.tex)
  reviewers/                    # reviewer responses (Generated Responses to Reviewers.md + the original .docx)
  submission/                   # original submitted PDF
results/                        # saved CSVs and PNG figures from each experiment
tests/                          # pytest suite (63 tests, ~5 s); includes a golden-run regression against tests/golden/baseline.json
README.md                       # this file
REFACTOR_PLAN.md                # phased refactor plan (Phases 1–7 complete)
WORKLOG.md                      # append-only history of significant changes
TODO_WORKFLOW.md                # cross-session task backlog
HOUSEKEEPING.md                 # recurring repo health check
LEGACY_BUGS_LOG.md              # catalog of bugs surfaced by the refactor
DEBUGGING_PLAN.md               # phased audit of code vs intended model (next session)
LP_TEX_REF.md                   # project-local LaTeX conventions for manuscript/
pyproject.toml, requirements.txt, LICENSE, .gitignore
```

| Module | Purpose |
|---|---|
| [rl_signaling/agents.py](rl_signaling/agents.py) | `BaseAgent` ABC + three concrete agents: `UrnAgent` (Roth–Erev), `QLearningAgent` (`egreedy` / `softmax` / `ucb`), `TDLearningAgent` |
| [rl_signaling/env.py](rl_signaling/env.py) | Canonical `MultiAgentEnv` (single-step shape, drives any `BaseAgent`); legacy `NetMultiAgentEnv` and `TempNetMultiAgentEnv` retained as deprecated wrappers |
| [rl_signaling/simulation.py](rl_signaling/simulation.py) | Canonical `run_simulation(env, n_episodes, …)`; legacy `simulation_function` and `temp_simulation_function` retained as deprecated wrappers |
| [rl_signaling/games.py](rl_signaling/games.py) | Random and canonical game generators; signal-urn initializers |
| [rl_signaling/info_theory.py](rl_signaling/info_theory.py) | Mutual information and normalized mutual information |
| [rl_signaling/plotting.py](rl_signaling/plotting.py) | KDE histograms, regression plots, reward/NMI-vs-cost plots, smoothing, CSV post-processing, the `plot_simulation_summary` helper used by `run_simulation` |
| [analytics/](analytics/) | Mathematical reference for every quantity the package computes, plus independent verification scripts. See [analytics/ANALYTICS_README.md](analytics/ANALYTICS_README.md) for the reading order; the Markov-chain analysis of the signal-trading game lives in [analytics/math/proof_of_concept_markov.md](analytics/math/proof_of_concept_markov.md), [analytics/math/initialization_basins.md](analytics/math/initialization_basins.md), [analytics/math/argiento_obstruction.md](analytics/math/argiento_obstruction.md), and the authoritative reference [analytics/math/roth_erev_polya_mle.md](analytics/math/roth_erev_polya_mle.md). |
| [manuscript/](manuscript/) | Paper sources for PHOS-17993. `main_v2.tex` is the active working copy (compiles to `main_v2.pdf`); `Appendix.tex` is a standalone companion. Reviewer-response material lives in [manuscript/reviewers/](manuscript/reviewers/). See [LP_TEX_REF.md](LP_TEX_REF.md) for the project-local LaTeX conventions. |

### Notebooks

| Notebook | Purpose |
|---|---|
| [notebooks/basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb) | Sanity check for each agent type on a small canonical game |
| [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb) | Main runs: canonical and complex models across the three agent types |
| [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) | Effect of urn/Q-table initialization strategies |
| [notebooks/Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb) | Hyperparameter tuning for Q-learning and TD-learning |
| [notebooks/Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb) | Costly-signaling experiments |
| [notebooks/plotting_results.ipynb](notebooks/plotting_results.ipynb) | Builds the final figures from the saved CSVs in `results/` |

## Setup

Python 3.10+. The recommended setup uses a project-local virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

The `dev` extras include `pytest`, `ruff`, and `ipykernel`. To run the notebooks in Jupyter / VS Code, register the kernel once:

```bash
python -m ipykernel install --user --name rl_signaling --display-name "Python (rl_signaling)"
```

Then select the **Python (rl_signaling)** kernel inside any notebook.

Alternative install paths:
- `pip install -e .` — runtime deps only, no test/lint tools.
- `pip install -r requirements.txt` — exact pinned versions matching the checked-in lockfile, useful for reproducibility audits.

## Minimal example

A 2-agent run on the canonical game with the urn agent:

```python
import networkx as nx
from rl_signaling import MultiAgentEnv, UrnAgent, run_simulation
from rl_signaling.games import create_random_canonical_game

# Fully connected directed graph between the two agents
graph = nx.DiGraph()
graph.add_nodes_from([0, 1])
graph.add_edges_from([(0, 1), (1, 0)])

game_dicts = {i: create_random_canonical_game(n_features=2, n_final_actions=4)
              for i in range(2)}

env = MultiAgentEnv(
    n_agents=2,
    n_features=2,
    n_signaling_actions=2,
    n_final_actions=4,
    full_information=False,
    game_dicts=game_dicts,
    observed_variables={0: [0], 1: [1]},
    agent_type=UrnAgent,
    graph=graph,
)

signal_usage, rewards_history, nmi_history, histories, nature_history = run_simulation(
    env, n_episodes=10000, with_signals=True, plot=True
)
```

The same scaffolding works for the other two agent types — just swap `agent_type=QLearningAgent` or `agent_type=TDLearningAgent`. See [notebooks/basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb) for the canonical reference.

The legacy `NetMultiAgentEnv` / `TempNetMultiAgentEnv` classes and `simulation_function` / `temp_simulation_function` runners are still available for backward compatibility with the experiment notebooks but emit a `DeprecationWarning`.

## Reproducing the figures

The CSVs and PNGs under [results/](results/) are the canonical outputs of the experiment notebooks. To regenerate them:

1. Run the experiment notebooks to produce per-experiment CSVs:
   - [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb) — canonical and complex models across the three agent types.
   - [notebooks/Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb) — costly-signaling experiments.
   - [notebooks/Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb) — hyperparameter sweeps.
   - [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) — urn-initialization variants.
2. Run [notebooks/plotting_results.ipynb](notebooks/plotting_results.ipynb) to consume the CSVs and emit the figures.

## Tests

Install the development extras and run the test suite:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The suite includes information-theoretic identities, agent contract checks, env lifecycle invariants, end-to-end smoke runs, and a deterministic golden-run regression against [tests/golden/baseline.json](tests/golden/baseline.json). To regenerate the baseline (only when the canonical implementation legitimately changes):

```bash
python tests/golden/save_baseline.py
```

## Hypothesis

Each agent's payoff is independent of the others' actions, so there is no immediate incentive to communicate meaningfully. The hypothesis is that, despite this, there exists a region of the parameter space in which agents coordinate — i.e. signal output and signal decoding align with the (partially hidden) state of the world.

## Status and known limitations

- The seven-phase refactor described in [REFACTOR_PLAN.md](REFACTOR_PLAN.md) is **complete**. The next active workstream is the model-vs-implementation audit in [DEBUGGING_PLAN.md](DEBUGGING_PLAN.md). See [TODO_WORKFLOW.md](TODO_WORKFLOW.md) for the active cross-session task and [WORKLOG.md](WORKLOG.md) for the full history.
- `signal_usage` history in [rl_signaling/env.py](rl_signaling/env.py) is appended every episode via `deepcopy`, which is memory-inefficient for long runs but kept for plotting compatibility.
- **Legacy and canonical APIs diverge slightly for `TDLearningAgent`.** In the legacy two-step flow (`TempNetMultiAgentEnv` + `temp_simulation_function`), the signal-phase update decays `exploration_rate` **before** the action-phase `get_action` runs. In the canonical `MultiAgentEnv` + `run_simulation` flow, both TD updates run at end-of-episode inside `update_episode`. As a result, the action-phase `get_action` sees a slightly higher `exploration_rate` in the new flow, which causes roughly 1 in 100 episodes to take a different explore/exploit branch and produce a different reward. The Q-value math is unchanged — this is purely an exploration-schedule ordering difference. The golden-run baseline at [tests/golden/baseline.json](tests/golden/baseline.json) is captured against the canonical API; the deprecated path is no longer the reference. `UrnAgent` and `QLearningAgent` produce byte-identical output across both APIs.
- Known legacy-code bugs are catalogued in [LEGACY_BUGS_LOG.md](LEGACY_BUGS_LOG.md). The `UrnAgent` action-urn initialization bug (Bug 1) was fixed during the refactor and changes the output of [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) — the saved `initializations_*.png` figures reflect the pre-fix behavior and need to be regenerated to reflect the corrected experiment.
