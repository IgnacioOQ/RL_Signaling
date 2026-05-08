"""Capture deterministic small-scale runs against the canonical API.

Runs a 100-episode simulation for each of the three agent types under a
fixed seed using :class:`rl_signaling.env.MultiAgentEnv` +
:func:`rl_signaling.simulation.run_simulation` (the new canonical API),
and writes the per-agent reward histories and final NMI to
``tests/golden/baseline.json``. ``tests/test_golden.py`` reads this file
and asserts byte-for-byte reproducibility against the canonical API.

The legacy ``NetMultiAgentEnv`` + ``simulation_function`` and
``TempNetMultiAgentEnv`` + ``temp_simulation_function`` paths are
deprecated and produce slightly different RNG sequences for
``TDLearningAgent`` (because signal-phase ``exploration_rate`` decay
happens before action-phase ``get_action`` in the legacy ordering, vs.
after both phases in the unified update_episode flow). The legacy
results are therefore not used as the golden reference.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_signaling.agents import (  # noqa: E402
    QLearningAgent,
    TDLearningAgent,
    UrnAgent,
)
from rl_signaling.env import MultiAgentEnv  # noqa: E402
from rl_signaling.games import create_random_canonical_game  # noqa: E402
from rl_signaling.simulation import run_simulation  # noqa: E402

SEED = 12345
N_EPISODES = 100
N_AGENTS = 2
N_FEATURES = 2
N_SIGNALING_ACTIONS = 2
N_FINAL_ACTIONS = 4


def _two_agent_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from([0, 1])
    g.add_edges_from([(0, 1), (1, 0)])
    return g


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _run(agent_type: type) -> dict:
    _seed(SEED)
    games = {
        i: create_random_canonical_game(N_FEATURES, N_FINAL_ACTIONS)
        for i in range(N_AGENTS)
    }
    env = MultiAgentEnv(
        n_agents=N_AGENTS,
        n_features=N_FEATURES,
        n_signaling_actions=N_SIGNALING_ACTIONS,
        n_final_actions=N_FINAL_ACTIONS,
        full_information=False,
        game_dicts=games,
        observed_variables={0: [0], 1: [1]},
        agent_type=agent_type,
        graph=_two_agent_graph(),
    )
    out = run_simulation(
        env, n_episodes=N_EPISODES, with_signals=True, plot=False
    )
    _, rewards_history, nmi_history, _, _ = out
    return {
        "rewards_history": [list(r) for r in rewards_history],
        "final_nmi": [h[-1] if h else None for h in nmi_history],
    }


def main() -> None:
    baseline = {
        "seed": SEED,
        "n_episodes": N_EPISODES,
        "urn_agent": _run(UrnAgent),
        "q_learning_agent": _run(QLearningAgent),
        "td_learning_agent": _run(TDLearningAgent),
    }
    out_path = Path(__file__).with_name("baseline.json")
    out_path.write_text(json.dumps(baseline, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
