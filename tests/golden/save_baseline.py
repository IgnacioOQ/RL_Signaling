"""Capture deterministic small-scale runs for each agent type.

Reads the current code, runs a 100-episode simulation for each of the three
agent types under a fixed seed, and writes the per-agent reward histories
and final NMI to ``tests/golden/baseline.json``.

The Phase 4 / Phase 5 refactors must produce a JSON with a matching
``baseline.json`` modulo the changes documented in REFACTOR_PLAN.md
(specifically: the UrnAgent action-urn initialization bug fix only affects
runs where ``initialize=True``; the baseline runs use ``initialize=False``,
so its outputs must be byte-identical pre- and post-fix).
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl_signaling.agents import QLearningAgent, TDLearningAgent, UrnAgent  # noqa: E402
from rl_signaling.env import NetMultiAgentEnv, TempNetMultiAgentEnv  # noqa: E402
from rl_signaling.games import create_random_canonical_game  # noqa: E402
from rl_signaling.simulation import simulation_function, temp_simulation_function  # noqa: E402


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


def run_urn_agent() -> dict:
    _seed(SEED)
    games = {i: create_random_canonical_game(N_FEATURES, N_FINAL_ACTIONS) for i in range(N_AGENTS)}
    env = NetMultiAgentEnv(
        n_agents=N_AGENTS,
        n_features=N_FEATURES,
        n_signaling_actions=N_SIGNALING_ACTIONS,
        n_final_actions=N_FINAL_ACTIONS,
        full_information=False,
        game_dicts=games,
        observed_variables={0: [0], 1: [1]},
        agent_type=UrnAgent,
        initialize=False,
        costly_signaling=False,
        graph=_two_agent_graph(),
    )
    out = simulation_function(
        n_agents=N_AGENTS,
        n_features=N_FEATURES,
        n_signaling_actions=N_SIGNALING_ACTIONS,
        n_final_actions=N_FINAL_ACTIONS,
        n_episodes=N_EPISODES,
        with_signals=True,
        plot=False,
        env=env,
    )
    signal_usage, rewards_history, nmi_history, _, _ = out
    return {
        "rewards_history": [list(r) for r in rewards_history],
        "final_nmi": [h[-1] if h else None for h in nmi_history],
    }


def run_q_learning_agent() -> dict:
    _seed(SEED)
    games = {i: create_random_canonical_game(N_FEATURES, N_FINAL_ACTIONS) for i in range(N_AGENTS)}
    env = NetMultiAgentEnv(
        n_agents=N_AGENTS,
        n_features=N_FEATURES,
        n_signaling_actions=N_SIGNALING_ACTIONS,
        n_final_actions=N_FINAL_ACTIONS,
        full_information=False,
        game_dicts=games,
        observed_variables={0: [0], 1: [1]},
        agent_type=QLearningAgent,
        initialize=False,
        costly_signaling=False,
        graph=_two_agent_graph(),
    )
    out = simulation_function(
        n_agents=N_AGENTS,
        n_features=N_FEATURES,
        n_signaling_actions=N_SIGNALING_ACTIONS,
        n_final_actions=N_FINAL_ACTIONS,
        n_episodes=N_EPISODES,
        with_signals=True,
        plot=False,
        env=env,
    )
    _, rewards_history, nmi_history, _, _ = out
    return {
        "rewards_history": [list(r) for r in rewards_history],
        "final_nmi": [h[-1] if h else None for h in nmi_history],
    }


def run_td_learning_agent() -> dict:
    _seed(SEED)
    games = {i: create_random_canonical_game(N_FEATURES, N_FINAL_ACTIONS) for i in range(N_AGENTS)}
    env = TempNetMultiAgentEnv(
        n_agents=N_AGENTS,
        n_features=N_FEATURES,
        n_signaling_actions=N_SIGNALING_ACTIONS,
        n_final_actions=N_FINAL_ACTIONS,
        full_information=False,
        game_dicts=games,
        observed_variables={0: [0], 1: [1]},
        agent_type=TDLearningAgent,
        graph=_two_agent_graph(),
    )
    out = temp_simulation_function(
        n_agents=N_AGENTS,
        n_features=N_FEATURES,
        n_signaling_actions=N_SIGNALING_ACTIONS,
        n_final_actions=N_FINAL_ACTIONS,
        n_episodes=N_EPISODES,
        with_signals=True,
        plot=False,
        env=env,
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
        "urn_agent": run_urn_agent(),
        "q_learning_agent": run_q_learning_agent(),
        "td_learning_agent": run_td_learning_agent(),
    }
    out_path = Path(__file__).with_name("baseline.json")
    out_path.write_text(json.dumps(baseline, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
