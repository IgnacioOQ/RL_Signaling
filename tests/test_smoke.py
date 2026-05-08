"""End-to-end smoke tests: 100 episodes per agent, with and without signals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rl_signaling.agents import QLearningAgent, TDLearningAgent, UrnAgent
from rl_signaling.env import MultiAgentEnv
from rl_signaling.simulation import run_simulation


N_EPISODES = 100


@pytest.mark.parametrize("agent_type", [UrnAgent, QLearningAgent, TDLearningAgent])
@pytest.mark.parametrize("with_signals", [True, False])
def test_end_to_end_invariants(
    two_agent_graph, small_game_dicts, agent_type, with_signals
):
    env = MultiAgentEnv(
        n_agents=2,
        n_features=2,
        n_signaling_actions=2,
        n_final_actions=4,
        full_information=False,
        game_dicts=small_game_dicts,
        observed_variables={0: [0], 1: [1]},
        agent_type=agent_type,
        graph=two_agent_graph,
    )

    signal_usage, rewards_history, nmi_history, _, nature_history = run_simulation(
        env,
        n_episodes=N_EPISODES,
        with_signals=with_signals,
        plot=False,
    )

    # Per-agent buffers all have the right length.
    assert len(rewards_history) == 2
    for i in range(2):
        assert len(rewards_history[i]) == N_EPISODES
    assert len(nature_history) == N_EPISODES

    # NMI history is populated only when with_signals=True (only the signal
    # step appends to signal_information_history).
    if with_signals:
        for i in range(2):
            assert len(nmi_history[i]) == N_EPISODES
            for nmi in nmi_history[i]:
                assert 0 <= nmi <= 1 + 1e-9
    else:
        for i in range(2):
            assert len(nmi_history[i]) == 0

    # Rewards are finite floats in the range produced by the canonical game (0 or 1).
    for i in range(2):
        for r in rewards_history[i]:
            assert math.isfinite(r)
            assert 0 <= r <= 1

    # When signals were sent, per-state signal counts sum to N_EPISODES.
    if with_signals:
        for i in range(2):
            total = sum(int(arr.sum()) for arr in signal_usage[i].values())
            assert total == N_EPISODES
    else:
        # No signal step ran → signal_usage is empty.
        for i in range(2):
            assert signal_usage[i] == {}
