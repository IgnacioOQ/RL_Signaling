"""Lifecycle and invariant tests for :class:`MultiAgentEnv`."""

from __future__ import annotations

import numpy as np
import pytest

from rl_signaling.agents import (
    BaseAgent,
    QLearningAgent,
    TDLearningAgent,
    UrnAgent,
)
from rl_signaling.env import MultiAgentEnv


# Keep test scaffolding compact and reused across cases.
def _make_env(
    two_agent_graph,
    small_game_dicts,
    *,
    agent_type=UrnAgent,
    full_information=False,
    costly_signaling=False,
):
    return MultiAgentEnv(
        n_agents=2,
        n_features=2,
        n_signaling_actions=2,
        n_final_actions=4,
        full_information=full_information,
        game_dicts=small_game_dicts,
        observed_variables={0: [0], 1: [1]},
        agent_type=agent_type,
        costly_signaling=costly_signaling,
        graph=two_agent_graph,
    )


def test_env_requires_graph(small_game_dicts):
    with pytest.raises(ValueError, match="graph"):
        MultiAgentEnv(
            n_agents=2,
            n_features=2,
            n_signaling_actions=2,
            n_final_actions=4,
            full_information=False,
            game_dicts=small_game_dicts,
            observed_variables={0: [0], 1: [1]},
            graph=None,
        )


def test_env_rejects_graph_node_count_mismatch(two_agent_graph, small_game_dicts):
    with pytest.raises(ValueError, match="Mismatch"):
        MultiAgentEnv(
            n_agents=3,  # graph has only 2 nodes
            n_features=2,
            n_signaling_actions=2,
            n_final_actions=4,
            full_information=False,
            game_dicts=small_game_dicts,
            observed_variables={0: [0], 1: [1], 2: [0]},
            graph=two_agent_graph,
        )


def test_env_rejects_non_baseagent_type(two_agent_graph, small_game_dicts):
    class NotAnAgent:
        def __init__(self, **_):
            pass

    with pytest.raises(ValueError, match="BaseAgent"):
        MultiAgentEnv(
            n_agents=2,
            n_features=2,
            n_signaling_actions=2,
            n_final_actions=4,
            full_information=False,
            game_dicts=small_game_dicts,
            observed_variables={0: [0], 1: [1]},
            agent_type=NotAnAgent,
            graph=two_agent_graph,
        )


@pytest.mark.parametrize("agent_type", [UrnAgent, QLearningAgent, TDLearningAgent])
def test_env_constructs_with_each_agent(agent_type, two_agent_graph, small_game_dicts):
    env = _make_env(two_agent_graph, small_game_dicts, agent_type=agent_type)
    assert len(env.agents) == 2
    assert all(isinstance(a, BaseAgent) for a in env.agents)


def test_costly_signaling_appends_null_signal(two_agent_graph, small_game_dicts):
    env = _make_env(two_agent_graph, small_game_dicts, costly_signaling=True)
    assert env.n_signaling_actions == 3  # 2 + null
    assert env._null_signal_index == 2


def test_reset_returns_nature_and_observations(two_agent_graph, small_game_dicts):
    env = _make_env(two_agent_graph, small_game_dicts)
    nat, obs = env.reset()
    assert nat.shape == (2,)
    assert len(obs) == 2
    # Partial-info: each agent sees only the feature listed in observed_variables.
    assert len(obs[0]) == 1
    assert len(obs[1]) == 1


def test_full_information_observations_carry_full_state(
    two_agent_graph, small_game_dicts
):
    env = _make_env(two_agent_graph, small_game_dicts, full_information=True)
    _, obs = env.reset()
    # In full_information mode each agent observes the full nature vector.
    assert len(obs[0]) == 2
    assert len(obs[1]) == 2


def test_step_signal_advances_step_and_appends_to_observation(
    two_agent_graph, small_game_dicts
):
    env = _make_env(two_agent_graph, small_game_dicts)
    _, obs = env.reset()
    sigs, new_obs = env.step_signal(obs)
    assert env.current_step == 2
    assert len(sigs) == 2
    # Each agent has in-degree 1 in the 2-agent fully-connected graph.
    assert all(0 <= s < env.n_signaling_actions for s in sigs)
    assert len(new_obs[0]) == len(obs[0]) + 1
    assert len(new_obs[1]) == len(obs[1]) + 1


def test_full_episode_increments_metric_buffers(two_agent_graph, small_game_dicts):
    env = _make_env(two_agent_graph, small_game_dicts)
    _, obs = env.reset()
    sigs, new_obs = env.step_signal(obs)
    acts = env.step_action(new_obs)
    rewards = env.reward(acts)
    env.update(obs, sigs, new_obs, acts, rewards)

    for i in range(2):
        assert len(env.rewards_history[i]) == 1
        assert len(env.signal_information_history[i]) == 1
        assert len(env.histories[i]["signal_history"]) == 1
        assert len(env.histories[i]["action_history"]) == 1
    assert env.current_step == 5


def test_reward_lookup_raises_on_missing_state(two_agent_graph, small_game_dicts):
    env = _make_env(two_agent_graph, small_game_dicts)
    env.nature_vector = np.array([1, 1])
    # game_dicts cover all 4 binary states, so to force an error we drop one.
    env.game_dicts = {0: {(0, 0): {0: 0}}, 1: small_game_dicts[1]}
    with pytest.raises(KeyError, match="Invalid state-action pair"):
        env.reward([0, 0])
