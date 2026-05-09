"""Agent contract tests: ABC enforcement, signal/action ranges, update math."""

from __future__ import annotations

import random

import numpy as np
import pytest

from rl_signaling.agents import (
    BaseAgent,
    QLearningAgent,
    TDLearningAgent,
    UrnAgent,
    _select_action,
)

# ----------------------------------------------------------- BaseAgent contract


def test_base_agent_is_not_directly_instantiable():
    with pytest.raises(TypeError):
        BaseAgent()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "agent_cls",
    [UrnAgent, QLearningAgent, TDLearningAgent],
)
def test_all_agents_subclass_base(agent_cls):
    assert issubclass(agent_cls, BaseAgent)


# -------------------------------------------------------- _select_action helper


def test_select_action_egreedy_at_zero_exploration_is_greedy():
    q = np.array([0.5, 1.0, 0.2, 0.8])
    counts = np.array([1.0, 1.0, 1.0, 1.0])
    random.seed(0)
    np.random.seed(0)
    assert _select_action(q, counts, 0.0, "egreedy") == 1


def test_select_action_ucb_with_unit_counts():
    q = np.array([0.5, 1.0, 0.2, 0.8])
    counts = np.array([1.0, 1.0, 1.0, 1.0])
    # Low exploration → tie-break by Q values; argmax is index 1.
    assert _select_action(q, counts, 0.0, "ucb") == 1


def test_select_action_respects_available_actions():
    q = np.array([0.5, 1.0, 0.2, 0.8])
    counts = np.array([1.0, 1.0, 1.0, 1.0])
    # Index 1 is the global greedy pick, but it is excluded.
    chosen = _select_action(q, counts, 0.0, "egreedy", available_actions=[0, 2, 3])
    assert chosen in {0, 2, 3}
    assert chosen == 3  # greedy over the available subset


def test_select_action_unknown_strategy_raises():
    q = np.array([0.5, 1.0])
    counts = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="Unknown choice strategy"):
        _select_action(q, counts, 0.5, "totally-not-a-strategy")


# ----------------------------------------------------------- UrnAgent specifics


def test_urn_agent_get_signal_in_range():
    a = UrnAgent(n_signaling_actions=3, n_final_actions=4)
    for _ in range(20):
        s = a.get_signal((0,))
        assert 0 <= s < 3


def test_urn_agent_update_clamps_at_zero():
    a = UrnAgent(n_signaling_actions=2, n_final_actions=2)
    a.get_signal((0,))  # populates signaling_urns[(0,)] = ones
    a.update_signals((0,), 0, reward=-100)
    assert a.signaling_urns[(0,)][0] == 0  # clamped, never negative


def test_urn_agent_initialize_true_seeds_action_urns():
    """Phase 4 bug fix: ``initialize=True`` must populate both urn dicts."""
    random.seed(0)
    a = UrnAgent(
        n_signaling_actions=2,
        n_final_actions=4,
        n_observed_features=1,
        initialize=True,
    )
    # signaling_urns indexed by 1-tuples (n_observed_features=1)
    assert len(a.signaling_urns) == 2
    # action_urns indexed by 2-tuples (observation + appended signal)
    assert len(a.action_urns) == 4
    # Every entry is a one-hot vector of the right length.
    for vec in a.signaling_urns.values():
        assert vec.shape == (2,)
        assert (vec == 0).sum() + (vec == 1).sum() == 2  # one-hot
    for vec in a.action_urns.values():
        assert vec.shape == (4,)
        assert (vec == 0).sum() + (vec == 1).sum() == 4


# --------------------------------------------------------- QLearningAgent tests


def test_q_learning_initialize_true_seeds_both_q_tables():
    """Bug 4 fix: ``initialize=True`` must populate both Q-table dicts."""
    random.seed(0)
    a = QLearningAgent(
        n_signaling_actions=2,
        n_final_actions=4,
        n_observed_features=1,
        initialize=True,
    )
    # q_table_signaling indexed by 1-tuples (n_observed_features=1)
    assert len(a.q_table_signaling) == 2
    # q_table_action indexed by 2-tuples (observation + appended signal)
    assert len(a.q_table_action) == 4
    # Every entry is a one-hot vector of the right length.
    for vec in a.q_table_signaling.values():
        assert vec.shape == (2,)
        assert (vec == 0).sum() + (vec == 1).sum() == 2  # one-hot
    for vec in a.q_table_action.values():
        assert vec.shape == (4,)
        assert (vec == 0).sum() + (vec == 1).sum() == 4
    # Visit counts are pre-allocated for every pre-seeded state.
    assert set(a.signaling_counts) == set(a.q_table_signaling)
    assert set(a.action_counts) == set(a.q_table_action)


def test_pre_seeded_q_tables_are_float_dtype():
    """Bug 9 fix: pre-seeded Q-tables must be float so TD updates do not truncate.

    Lazy-init creates float64 zeros; pre-seed used to create int64 one-hots,
    which silently floored fractional TD increments — Q[1] became 0 after a
    single reward-0 update because (1 + 0.1*(0-1)) = 0.9 cast back to int = 0.
    """
    a = QLearningAgent(
        n_signaling_actions=2,
        n_final_actions=4,
        n_observed_features=1,
        initialize=True,
        initialization_weights=(100, 1),
    )
    for vec in a.q_table_signaling.values():
        assert np.issubdtype(vec.dtype, np.floating)
    for vec in a.q_table_action.values():
        assert np.issubdtype(vec.dtype, np.floating)

    u = UrnAgent(
        n_signaling_actions=2,
        n_final_actions=4,
        n_observed_features=1,
        initialize=True,
        initialization_weights=(100, 1),
    )
    for vec in u.signaling_urns.values():
        assert np.issubdtype(vec.dtype, np.floating)
    for vec in u.action_urns.values():
        assert np.issubdtype(vec.dtype, np.floating)


def test_q_learning_pre_seed_bias_persists_through_zero_reward_decay():
    """Bug 9 verification: pre-seeded Q-table bias survives 50 reward-0 updates.

    Closed form: Q_n = r + (Q_0 - r)*(1 - alpha)^n. With alpha=0.1, r=0,
    Q_hot_0=100, Q_cold_0=1: after n=50 visits each, Q_hot ≈ 0.515 and
    Q_cold ≈ 0.0052, so Q_hot - Q_cold ≈ 0.51. Pre-fix int dtype collapsed
    both cells to 0 (Q_cold in one step; Q_hot via repeated truncation),
    leaving zero gap.
    """
    random.seed(0)
    a = QLearningAgent(
        n_signaling_actions=2,
        n_final_actions=4,
        n_observed_features=1,
        initialize=True,
        initialization_weights=(100, 1),
    )
    state = next(iter(a.q_table_signaling.keys()))
    hot = int(np.argmax(a.q_table_signaling[state]))
    cold = 1 - hot

    for _ in range(50):
        a.update_signals(state, hot, reward=0.0)
        a.update_signals(state, cold, reward=0.0)

    q_hot = a.q_table_signaling[state][hot]
    q_cold = a.q_table_signaling[state][cold]
    # Closed-form predictions: 100*0.9^50 ≈ 0.5154, 1*0.9^50 ≈ 0.005154.
    assert q_hot == pytest.approx(100 * 0.9 ** 50, rel=1e-6)
    assert q_cold == pytest.approx(1 * 0.9 ** 50, rel=1e-6)
    assert q_hot - q_cold > 0.4


@pytest.mark.parametrize("choice", ["egreedy", "softmax", "ucb"])
def test_q_learning_get_signal_each_strategy(choice):
    a = QLearningAgent(n_signaling_actions=3, n_final_actions=4, choice=choice)
    s = a.get_signal((0,))
    assert 0 <= s < 3
    assert a.signaling_counts[(0,)][s] == 1


def test_q_learning_exploration_decays_after_update():
    a = QLearningAgent(
        n_signaling_actions=2,
        n_final_actions=4,
        exploration_rate=1.0,
        exploration_decay=0.5,
        min_exploration_rate=0.0,
    )
    a.get_signal((0,))
    rate_before = a.signal_exploration_rate
    a.update_signals((0,), 0, reward=1.0)
    assert a.signal_exploration_rate == pytest.approx(rate_before * 0.5)


# --------------------------------------------------------- TDLearningAgent tests


def test_td_learning_legacy_constructor():
    a = TDLearningAgent(n_actions=4)
    # When only n_actions is given, both subsets default to it.
    assert a.n_signaling_actions == 4
    assert a.n_final_actions == 4
    assert a.n_actions == 4


def test_td_learning_canonical_constructor():
    a = TDLearningAgent(n_signaling_actions=2, n_final_actions=4)
    assert a.n_signaling_actions == 2
    assert a.n_final_actions == 4
    assert a.n_actions == 4  # max of the two


def test_td_learning_constructor_requires_either_form():
    with pytest.raises(ValueError, match="Provide either"):
        TDLearningAgent()


def test_td_learning_get_signal_uses_signaling_subset():
    a = TDLearningAgent(n_signaling_actions=2, n_final_actions=4, choice="egreedy")
    for _ in range(20):
        s = a.get_signal((0,))
        assert 0 <= s < 2


def test_td_learning_get_action_default_uses_final_subset():
    a = TDLearningAgent(n_signaling_actions=2, n_final_actions=4, choice="egreedy")
    for _ in range(20):
        act = a.get_action((0,))
        assert 0 <= act < 4


def test_td_learning_update_episode_runs_two_updates():
    a = TDLearningAgent(n_signaling_actions=2, n_final_actions=4, exploration_rate=0.0)
    # Pre-populate the action-state Q-row so the signal-phase TD bootstrap
    # has a non-zero target (otherwise td_error == 0 and the row would not move).
    a.q_table[(0, 0)] = np.array([0.0, 0.0, 5.0, 0.0])
    a.action_counts[(0, 0)] = np.array([0.0, 0.0, 0.0, 0.0])
    s = a.get_signal((0,))
    act = a.get_action((0, s))
    q_signal_before = a.q_table[(0,)].copy()
    q_action_before = a.q_table[(0, s)].copy()
    a.update_episode((0,), s, (0, s), act, reward=1.0)
    # Signal-phase row was 0; TD target = gamma*max(q_table[(0,s)]) > 0 → row moves.
    assert not np.array_equal(q_signal_before, a.q_table[(0,)])
    # Action-phase row: terminal td_target = reward = 1.0 → row moves.
    assert not np.array_equal(q_action_before, a.q_table[(0, s)])
