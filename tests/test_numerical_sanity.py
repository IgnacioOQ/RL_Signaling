"""Numerical sanity cases — hand-derived analytical answers verified against the implementation.

Each test states the analytical answer in a comment, then checks the implementation
matches to a documented tolerance. Exact for finite-step cases; asymptotic for the
convergence case.

The cases mirror Phase 4 of ``DEBUGGING_PLAN.md``:

1. NMI on a known distribution; verifies log base via ``_compute_entropy``.
2. Single-step Q-update with ``alpha=0.1`` and ``reward=1`` — geometric convergence.
3. TD update with ``gamma=1``, ``reward=0``, ``max Q(next)=1``, ``count=1`` → ``Q=1``.
4. Costly signaling with ``cost=0.25`` and ``game_reward=1`` — exact arithmetic.
5. Convergence to the optimal action under full information / no signals.
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pytest

from rl_signaling.agents import QLearningAgent, TDLearningAgent, UrnAgent
from rl_signaling.env import MultiAgentEnv
from rl_signaling.info_theory import _compute_entropy, compute_mutual_information
from rl_signaling.simulation import run_simulation


# =============================================================================
# Case 1 — NMI on a known distribution; verify log base via _compute_entropy.
# =============================================================================


def test_entropy_is_in_bits_log_base_2():
    """``H(Uniform_n) = log2(n)`` bits.

    Confirms the log base is 2 by checking three known points:
      - ``H([1.0]) = 0``
      - ``H([0.5, 0.5]) = 1`` bit
      - ``H([0.25, 0.25, 0.25, 0.25]) = 2`` bits
    """
    assert _compute_entropy([1.0]) == pytest.approx(0.0, abs=1e-12)
    assert _compute_entropy([0.5, 0.5]) == pytest.approx(1.0, abs=1e-12)
    assert _compute_entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(2.0, abs=1e-12)


def test_perfect_2x2_correlation_nmi_is_one_by_hand():
    """2x2 signal-usage table ``[[10, 0], [0, 10]]`` — derived by hand.

    ``P(O) = [0.5, 0.5]`` → ``H(O) = 1`` bit.
    ``P(S) = [0.5, 0.5]`` → ``H(S) = 1`` bit.
    ``P(S | O=o)`` is one-hot for each ``o`` → ``H(S | O) = 0``.
    ``I(S; O) = H(S) - H(S | O) = 1 - 0 = 1``.
    ``NMI = I / H(O) = 1 / 1 = 1.0`` exact.
    """
    usage = {
        (0,): np.array([10.0, 0.0]),
        (1,): np.array([0.0, 10.0]),
    }
    mi, nmi = compute_mutual_information(usage)
    assert mi == pytest.approx(1.0, abs=1e-12)
    assert nmi == pytest.approx(1.0, abs=1e-12)


# =============================================================================
# Case 2 — Q-learning update with alpha=0.1, reward=1, no bootstrap.
# =============================================================================


def test_q_learning_single_update_is_exact_alpha_times_reward():
    """``Q[s][a] = 0`` then update with ``alpha=0.1`` and ``reward=1``:

    ``td_target = reward = 1``, ``td_error = 1 - 0 = 1``,
    ``Q[s][a] += alpha * td_error = 0.1`` → ``Q[s][a] = 0.1`` exact.
    """
    a = QLearningAgent(n_signaling_actions=2, n_final_actions=2)
    state = (0,)
    a.get_signal(state)  # populate row with zeros and counts[state] with zeros
    assert a.q_table_signaling[state][0] == 0.0
    a.update_signals(state, 0, reward=1.0)
    assert a.q_table_signaling[state][0] == pytest.approx(0.1, abs=1e-12)


def test_q_learning_ten_updates_match_geometric_closed_form():
    """``Q_n = 1 - (1 - alpha)^n`` for constant ``alpha=0.1`` and constant ``reward=1``.

    Closed form: ``Q_{n+1} = (1 - alpha) Q_n + alpha * reward``, so
    ``Q_n = 1 - 0.9^n`` from ``Q_0 = 0``. After 10 updates,
    ``Q_10 = 1 - 0.9^10 = 0.6513215599...``.
    """
    a = QLearningAgent(n_signaling_actions=2, n_final_actions=2)
    state = (0,)
    a.get_signal(state)
    for _ in range(10):
        a.update_signals(state, 0, reward=1.0)
    expected = 1.0 - 0.9**10
    assert a.q_table_signaling[state][0] == pytest.approx(expected, abs=1e-12)


# =============================================================================
# Case 3 — TD update with gamma=1, reward=0, max Q(next)=1, count=1 → Q=1.
# =============================================================================


def test_td_one_step_bootstrap_with_unit_count():
    """``TDLearningAgent.update`` from ``Q[(0,)][0] = 0``:

    Pre-populated ``Q[(1,)] = [1, 0, ..., 0]`` so ``max(Q[(1,)]) = 1``.
    Pre-populated ``action_counts[(0,)][0] = 1`` (mimicking one prior ``get_action``).

    ``td_target = 0 + 1 * 1 = 1``, ``td_error = 1 - 0 = 1``,
    ``learning_rate = td_error / count = 1``,
    ``Q[(0,)][0] += 1`` → ``1.0`` exact.
    """
    a = TDLearningAgent(n_actions=4, gamma=1.0)
    a.q_table[(0,)] = np.zeros(4)
    a.q_table[(1,)] = np.array([1.0, 0.0, 0.0, 0.0])
    a.action_counts[(0,)] = np.array([1.0, 0.0, 0.0, 0.0])
    a.action_counts[(1,)] = np.zeros(4)

    a.update(state=(0,), action=0, reward=0.0, next_state=(1,), done=False)

    assert a.q_table[(0,)][0] == pytest.approx(1.0, abs=1e-12)


def test_td_one_step_terminal_no_bootstrap():
    """When ``done=True``, ``td_target = reward`` (no bootstrap term).

    ``Q[(0,)][0] = 0``, ``reward = 1``, ``count = 1`` →
    ``td_error = 1``, ``learning_rate = 1``, ``Q[(0,)][0] = 1`` exact.
    """
    a = TDLearningAgent(n_actions=4, gamma=1.0)
    a.q_table[(0,)] = np.zeros(4)
    a.action_counts[(0,)] = np.array([1.0, 0.0, 0.0, 0.0])

    a.update(state=(0,), action=0, reward=1.0, next_state=(0,), done=True)

    assert a.q_table[(0,)][0] == pytest.approx(1.0, abs=1e-12)


# =============================================================================
# Case 4 — Costly signaling: env.reward subtracts cost from non-null, leaves null free.
# =============================================================================


def _build_costly_env(two_agent_graph: nx.DiGraph) -> MultiAgentEnv:
    """Build a 2-feature, 4-action env with costly_signaling=True and a flat reward of 1."""
    game_dicts = {
        i: {(0, 0): {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}}
        for i in range(2)
    }
    env = MultiAgentEnv(
        n_agents=2,
        n_features=2,
        n_signaling_actions=2,
        n_final_actions=4,
        game_dicts=game_dicts,
        observed_variables={0: [0], 1: [1]},
        full_information=True,
        agent_type=UrnAgent,
        costly_signaling=True,
        graph=two_agent_graph,
    )
    env.nature_vector = np.array([0, 0])
    return env


def test_costly_signaling_subtracts_cost_for_non_null(two_agent_graph):
    """``game_reward = 1``, ``cost = 0.25``, both signals non-null → ``reward = 0.75`` exact."""
    env = _build_costly_env(two_agent_graph)
    rewards = env.reward(actions=[0, 0], signals=[0, 0], signal_cost=[0.25, 0.25])
    assert rewards == pytest.approx([0.75, 0.75], abs=1e-12)


def test_costly_signaling_null_signal_pays_no_cost(two_agent_graph):
    """Null signal is at index ``_null_signal_index``; emitting it skips the cost.

    ``game_reward = 1``, both signals null → ``reward = 1.0`` exact.
    """
    env = _build_costly_env(two_agent_graph)
    null_idx = env._null_signal_index
    assert null_idx == 2  # n_signaling_actions becomes base+1=3 internally; null = idx 2
    rewards = env.reward(
        actions=[0, 0], signals=[null_idx, null_idx], signal_cost=[0.25, 0.25]
    )
    assert rewards == pytest.approx([1.0, 1.0], abs=1e-12)


def test_costly_signaling_mixed_signals_cost_only_non_null(two_agent_graph):
    """Agent 0 sends non-null (pays cost), agent 1 sends null (free) → ``[0.75, 1.0]``."""
    env = _build_costly_env(two_agent_graph)
    null_idx = env._null_signal_index
    rewards = env.reward(
        actions=[0, 0], signals=[0, null_idx], signal_cost=[0.25, 0.25]
    )
    assert rewards == pytest.approx([0.75, 1.0], abs=1e-12)


# =============================================================================
# Case 5 — Convergence: full-info, no-signals, hand-crafted optimal-action game.
# =============================================================================


def test_urn_agent_converges_to_optimal_action_in_full_information(two_agent_graph):
    """Full-info, no-signals: each agent should learn the optimal action per state.

    Game: state ``(0, 0)`` → action 2 is optimal (reward 1, others 0). With
    ``full_information=True`` every agent sees the full state, so observations
    are tuples of length ``n_features``. After 1000 episodes (~250 per state on
    average), the Roth–Erev urn for state ``(0, 0)`` should be heavily biased
    toward action 2.

    Asymptotic check: ``urn[2] / sum(urn) >= 0.95`` for both agents at the end
    of the run.
    """
    np.random.seed(0)
    random.seed(0)

    # State -> {action -> reward}; one canonical optimum per state.
    # State (0, 0)'s optimum is action 2 per the Phase 4 plan.
    canonical = {
        (0, 0): {0: 0, 1: 0, 2: 1, 3: 0},
        (0, 1): {0: 0, 1: 1, 2: 0, 3: 0},
        (1, 0): {0: 1, 1: 0, 2: 0, 3: 0},
        (1, 1): {0: 0, 1: 0, 2: 0, 3: 1},
    }
    game_dicts = {i: dict(canonical) for i in range(2)}

    env = MultiAgentEnv(
        n_agents=2,
        n_features=2,
        n_signaling_actions=2,
        n_final_actions=4,
        game_dicts=game_dicts,
        observed_variables={0: [0], 1: [1]},
        full_information=True,
        agent_type=UrnAgent,
        graph=two_agent_graph,
    )

    run_simulation(env, n_episodes=1000, with_signals=False, plot=False)

    # Action urn is keyed by the post-signal observation. With full_information=True
    # and with_signals=False, that key is the full nature tuple — for state (0, 0)
    # it is the 2-tuple of zeros.
    target_state = (0, 0)
    for i in range(2):
        # Find the urn entry that compares equal to (0, 0). The env stores keys
        # built from `tuple(np.array(...))` so the ints are numpy.int64; equality
        # against a Python int tuple still works.
        urn = next(
            v for k, v in env.agents[i].action_urns.items() if tuple(int(x) for x in k) == target_state
        )
        prob_action_2 = urn[2] / urn.sum()
        assert prob_action_2 >= 0.95, (
            f"Agent {i} action_urns[{target_state}] = {urn}, "
            f"P(action 2) = {prob_action_2:.4f} (expected >= 0.95)"
        )
