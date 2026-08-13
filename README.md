# RL_Signaling
- status: active
- type: how-to
- id: rl_signaling.readme
- description: Reinforcement-learning study of emergent signaling between agents under partial observation; companion code for "Signaling Games with Distributed Rewards" (Philosophy of Science). Covers the model, repository layout, setup, figure reproduction, and a minimal runnable example.
- label: [core]
- injection: informational
- volatility: evolving
- scope: project-specific
- last_checked: 2026-08-12
<!-- content -->

Companion code for **"Signaling Games with Distributed Rewards"**, accepted at *Philosophy of Science* (PHOS-17993).

A reinforcement-learning study of emergent signaling between agents under partial observation. Two or more agents on a directed graph each observe a subset of the world state, exchange signals, then take an action whose payoff depends on the full state. The question is whether — and when — meaningful communication emerges, even though each agent's payoff is independent of the others' actions.

> **This repository is code-only.** The manuscript sources, referee correspondence, and talk slides are deliberately not distributed here — the published article is under journal copyright and referee material is confidential. Every figure in the paper is traced back to the code and data that produced it in [results/MANIFEST.md](results/MANIFEST.md).

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
results/                        # saved CSVs and PNG figures from each experiment
  MANIFEST.md                   # figure -> notebook -> dataset traceability for every published figure
  legacy/datasets/              # the seven simulation outputs behind the paper (~24 MB, tracked deliberately)
  legacy/plots/                 # figures from the original run
  proof_of_concept/             # proof-of-concept figures
  new_code/plots/               # post-refactor verification figure
tests/                          # pytest suite (63 tests, ~4 s); includes a golden-run regression against tests/golden/baseline.json
README.md                       # this file
HOUSEKEEPING.md                 # recurring repo health check
pyproject.toml, requirements.txt, LICENSE, .gitignore
```

That is the whole repository. Everything relating to the manuscript — the LaTeX sources, the referee correspondence, the revision toolkit and its templates, the LaTeX style notes, and the internal audit trail — is excluded by design. The `analytics/` tree — the mathematical derivations behind every quantity the package computes, and the standalone verification scripts that check them — is likewise kept local and not distributed here, along with the working-material notebooks that depend on it and the generators that built them.

| Module | Purpose |
|---|---|
| [rl_signaling/agents.py](rl_signaling/agents.py) | `BaseAgent` ABC + three concrete agents: `UrnAgent` (Roth–Erev), `QLearningAgent` (`egreedy` / `softmax` / `ucb`), `TDLearningAgent` |
| [rl_signaling/env.py](rl_signaling/env.py) | Canonical `MultiAgentEnv` (single-step shape, drives any `BaseAgent`); legacy `NetMultiAgentEnv` and `TempNetMultiAgentEnv` retained as deprecated wrappers |
| [rl_signaling/simulation.py](rl_signaling/simulation.py) | Canonical `run_simulation(env, n_episodes, …)`; legacy `simulation_function` and `temp_simulation_function` retained as deprecated wrappers |
| [rl_signaling/games.py](rl_signaling/games.py) | Random and canonical game generators; signal-urn initializers |
| [rl_signaling/info_theory.py](rl_signaling/info_theory.py) | Mutual information and normalized mutual information |
| [rl_signaling/plotting.py](rl_signaling/plotting.py) | KDE histograms, regression plots, reward/NMI-vs-cost plots, smoothing, CSV post-processing, the `plot_simulation_summary` helper used by `run_simulation` |
| [results/MANIFEST.md](results/MANIFEST.md) | Traceability map from each published figure back to the notebook, module function, and dataset that produced it — including four recorded reproducibility gaps. |

### Notebooks

| Notebook | Purpose |
|---|---|
| [notebooks/basic_unit_test.ipynb](notebooks/basic_unit_test.ipynb) | Sanity check for each agent type on a small canonical game |
| [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb) | Main runs: canonical and complex models across the three agent types |
| [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) | Effect of urn/Q-table initialization strategies |
| [notebooks/Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb) | Hyperparameter tuning for Q-learning and TD-learning |
| [notebooks/Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb) | Costly-signaling experiments |
| [notebooks/plotting_results.ipynb](notebooks/plotting_results.ipynb) | Builds the final figures from the saved CSVs in `results/` |
| [notebooks/proof_of_concept_figures_final.ipynb](notebooks/proof_of_concept_figures_final.ipynb) | **Produces the paper's proof-of-concept figure** (§2.2). Roth–Erev and Q-learning candidates |
| [notebooks/poc_absorbing_states.py](notebooks/poc_absorbing_states.py) | `enumerate_absorbing_rewards` — mean reward over the 2304 absorbing states, used by the notebook above |

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

**[results/MANIFEST.md](results/MANIFEST.md) is the authoritative map** from each of the paper's 27 figures to the code and data behind it. Read it first — figure filenames are constructed at save time from a prefix plus a variable name, so grepping the codebase for a figure's filename will find nothing.

The seven datasets behind the published figures are committed under `results/legacy/datasets/`, so the figures can be rebuilt without re-running the simulations:

```bash
pip install -e .
jupyter nbconvert --to notebook --execute --inplace notebooks/plotting_results.ipynb
```

That regenerates 15 of the 27 published figures. The remaining 12 have documented gaps — hand-renamed files, figures saved with an explicit path in an interactive session, and one hyperparameter sweep that ran on Colab and whose raw output was never committed. All four gaps are described in the MANIFEST rather than glossed over.

To re-run the simulations from scratch instead of reusing the committed CSVs:

- [notebooks/Run_Simulations.ipynb](notebooks/Run_Simulations.ipynb) — canonical and complex models across the three agent types.
- [notebooks/Final_Costly_Signaling_Run_Simulations.ipynb](notebooks/Final_Costly_Signaling_Run_Simulations.ipynb) — costly-signaling experiments.
- [notebooks/Parameter_Optimization_wchoices.ipynb](notebooks/Parameter_Optimization_wchoices.ipynb) — hyperparameter sweeps (expects a Colab/Drive path; see MANIFEST Gap 4).
- [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb) — urn-initialization variants.

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

- A seven-phase refactor moved the code from a flat collection of scripts into the `rl_signaling/` package; it is **complete**. A model-vs-implementation audit and a notebook-migration cleanup were planned but not carried out. The working notes for all three are internal documents, not distributed with this repository.
- `signal_usage` history in [rl_signaling/env.py](rl_signaling/env.py) is appended every episode via `deepcopy`, which is memory-inefficient for long runs but kept for plotting compatibility.
- **Legacy and canonical APIs diverge slightly for `TDLearningAgent`.** In the legacy two-step flow (`TempNetMultiAgentEnv` + `temp_simulation_function`), the signal-phase update decays `exploration_rate` **before** the action-phase `get_action` runs. In the canonical `MultiAgentEnv` + `run_simulation` flow, both TD updates run at end-of-episode inside `update_episode`. As a result, the action-phase `get_action` sees a slightly higher `exploration_rate` in the new flow, which causes roughly 1 in 100 episodes to take a different explore/exploit branch and produce a different reward. The Q-value math is unchanged — this is purely an exploration-schedule ordering difference. The golden-run baseline at [tests/golden/baseline.json](tests/golden/baseline.json) is captured against the canonical API; the deprecated path is no longer the reference. `UrnAgent` and `QLearningAgent` produce byte-identical output across both APIs.
- An `UrnAgent` action-urn initialization bug was found and fixed during the refactor. It changes the output of [notebooks/Initializations_test.ipynb](notebooks/Initializations_test.ipynb): the saved `init_smooth_*.png` figures reflect the **pre-fix** behaviour and would need regenerating to reflect the corrected experiment. Those figures are exploratory and do not appear in the paper — see [results/MANIFEST.md](results/MANIFEST.md).
- Four of the paper's 27 figures cannot be reproduced byte-identically from a clean clone. The causes are documented as Gaps 1–4 in [results/MANIFEST.md](results/MANIFEST.md); the underlying data is committed in every case except the Colab hyperparameter sweep.
