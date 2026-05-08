# RL_Signaling

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

| File | Purpose |
|---|---|
| [agents.py](agents.py) | Agent classes: `UrnAgent` (Roth–Erev), `QLearningAgent` (`egreedy` / `softmax` / `ucb`), `TDLearningAgent` |
| [environment.py](environment.py) | `NetMultiAgentEnv` (main, single-step episodes) and `TempNetMultiAgentEnv` (two-step formulation used by the TD-learning agent) |
| [simulation_function.py](simulation_function.py) | `simulation_function` for `NetMultiAgentEnv`; `temp_simulation_function` for `TempNetMultiAgentEnv` |
| [utils.py](utils.py) | Game generators, signal-urn initializers, mutual-information metrics, plotting helpers |
| [imports.py](imports.py) | Shared third-party imports |
| [plots_and_results/](plots_and_results/) | Saved CSVs and PNG figures from each experiment |

### Notebooks

| Notebook | Purpose |
|---|---|
| [basic_unit_test.ipynb](basic_unit_test.ipynb) | Sanity check for each agent type on a small canonical game |
| [Run_Simulations.ipynb](Run_Simulations.ipynb) | Main runs: canonical and complex models across the three agent types |
| [Initializations_test.ipynb](Initializations_test.ipynb) | Effect of urn/Q-table initialization strategies |
| [Parameter_Optimization_wchoices.ipynb](Parameter_Optimization_wchoices.ipynb) | Hyperparameter tuning for Q-learning and TD-learning |
| [Final_Costly_Signaling_Run_Simulations.ipynb](Final_Costly_Signaling_Run_Simulations.ipynb) | Costly-signaling experiments |
| [plotting_results.ipynb](plotting_results.ipynb) | Builds the final figures from the saved CSVs |

## Setup

Dependencies (Python 3.10+):

```
numpy pandas matplotlib seaborn networkx scikit-learn tqdm joblib
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn networkx scikit-learn tqdm joblib
```

## Minimal example

A 2-agent run on the canonical game with the urn agent:

```python
import networkx as nx
from utils import create_random_canonical_game
from agents import UrnAgent
from environment import NetMultiAgentEnv
from simulation_function import simulation_function

n_agents, n_features = 2, 2
n_signaling_actions, n_final_actions = 2, 4
agents_observed_variables = {0: [0], 1: [1]}

# Fully connected directed graph between the two agents
G = nx.DiGraph()
G.add_nodes_from([0, 1])
G.add_edges_from([(0, 1), (1, 0)])

game_dicts = {i: create_random_canonical_game(n_features, n_final_actions)
              for i in range(n_agents)}

env = NetMultiAgentEnv(
    n_agents=n_agents, n_features=n_features,
    n_signaling_actions=n_signaling_actions,
    n_final_actions=n_final_actions,
    full_information=False,
    game_dicts=game_dicts,
    observed_variables=agents_observed_variables,
    agent_type=UrnAgent,
    initialize=False,
    costly_signaling=False,
    graph=G,
)

signal_usage, rewards_history, signal_information_history, histories, nature_history = \
    simulation_function(
        n_agents=n_agents, n_features=n_features,
        n_signaling_actions=n_signaling_actions,
        n_final_actions=n_final_actions,
        n_episodes=10000, with_signals=True,
        plot=True, env=env,
    )
```

For the TD-learning agent, swap in `TempNetMultiAgentEnv` and `temp_simulation_function` instead — see [basic_unit_test.ipynb](basic_unit_test.ipynb) for a working example.

To reproduce the published figures, run the experiment notebooks first to generate CSVs in `plots_and_results/`, then run [plotting_results.ipynb](plotting_results.ipynb).

## Hypothesis

Each agent's payoff is independent of the others' actions, so there is no immediate incentive to communicate meaningfully. The hypothesis is that, despite this, there exists a region of the parameter space in which agents coordinate — i.e. signal output and signal decoding align with the (partially hidden) state of the world.

## Status and known limitations

- `signal_usage` history in [environment.py](environment.py) is appended every episode via `deepcopy`, which is memory-inefficient for long runs but kept for plotting compatibility.
- The `TempNetMultiAgentEnv` / `temp_simulation_function` path is the one used for the TD-learning agent; the in-code comment claiming it is unused is stale.
