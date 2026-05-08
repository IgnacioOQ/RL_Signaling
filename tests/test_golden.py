"""Reproducibility regression: legacy and new APIs match the saved golden run.

The baseline at ``tests/golden/baseline.json`` is the deterministic
fingerprint captured at the end of Phase 4 / Phase 5 (seed 12345,
100 episodes, ``initialize=False``). Any future change that perturbs the
RNG sequence on this path will fail this test.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from rl_signaling.agents import QLearningAgent, TDLearningAgent, UrnAgent
from rl_signaling.env import MultiAgentEnv
from rl_signaling.games import create_random_canonical_game
from rl_signaling.simulation import run_simulation


BASELINE_PATH = Path(__file__).parent / "golden" / "baseline.json"


def _two_agent_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from([0, 1])
    g.add_edges_from([(0, 1), (1, 0)])
    return g


def _run(agent_cls, *, seed: int, n_episodes: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    games = {i: create_random_canonical_game(2, 4) for i in range(2)}
    env = MultiAgentEnv(
        n_agents=2,
        n_features=2,
        n_signaling_actions=2,
        n_final_actions=4,
        full_information=False,
        game_dicts=games,
        observed_variables={0: [0], 1: [1]},
        agent_type=agent_cls,
        graph=_two_agent_graph(),
    )
    out = run_simulation(env, n_episodes=n_episodes, with_signals=True, plot=False)
    _, rewards_history, nmi_history, _, _ = out
    return {
        "rewards_history": [list(r) for r in rewards_history],
        "final_nmi": [h[-1] if h else None for h in nmi_history],
    }


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(f"Baseline file missing: {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text())


@pytest.mark.parametrize(
    "agent_cls,key",
    [
        (UrnAgent, "urn_agent"),
        (QLearningAgent, "q_learning_agent"),
        (TDLearningAgent, "td_learning_agent"),
    ],
)
def test_new_api_reproduces_baseline_final_nmi(baseline, agent_cls, key):
    result = _run(agent_cls, seed=baseline["seed"], n_episodes=baseline["n_episodes"])
    assert result["final_nmi"] == baseline[key]["final_nmi"]


@pytest.mark.parametrize(
    "agent_cls,key",
    [
        (UrnAgent, "urn_agent"),
        (QLearningAgent, "q_learning_agent"),
        (TDLearningAgent, "td_learning_agent"),
    ],
)
def test_new_api_reproduces_baseline_rewards(baseline, agent_cls, key):
    result = _run(agent_cls, seed=baseline["seed"], n_episodes=baseline["n_episodes"])
    assert result["rewards_history"] == baseline[key]["rewards_history"]
