"""Reinforcement-learning agents.

Three agent types share the role of selecting signals and final actions and
updating internal estimates from received rewards:

- :class:`UrnAgent` — Roth–Erev–style reinforcement of urn counts.
- :class:`QLearningAgent` — Q-learning with ``egreedy`` / ``softmax`` / ``ucb``
  exploration strategies.
- :class:`TDLearningAgent` — temporal-difference learning over a single Q-table
  with a unified action space (used with the two-step environment).

:class:`UrnAgent` and :class:`QLearningAgent` both inherit from
:class:`BaseAgent`, which defines the canonical
``get_signal`` / ``get_action`` / ``update_signals`` / ``update_actions``
interface. :class:`TDLearningAgent` exposes a different
``get_action(state, available_actions)`` / ``update(state, action, reward,
next_state, done)`` API because its update rule bootstraps off ``next_state``;
bringing it under the same base class is coupled to the env-unification
work in Phase 5 of ``REFACTOR_PLAN.md``.

The egreedy / softmax / ucb exploration strategy is implemented once in
:func:`_select_action` and reused across :class:`QLearningAgent` and
:class:`TDLearningAgent`.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from rl_signaling.games import create_initial_signals


def _select_action(
    q_values: NDArray[np.float64],
    counts: NDArray[np.float64],
    exploration_rate: float,
    choice: str,
    available_actions: Sequence[int] | None = None,
) -> int:
    """Pick an action under ``egreedy`` / ``softmax`` / ``ucb``.

    Shared exploration kernel for :class:`QLearningAgent` and
    :class:`TDLearningAgent`. When ``available_actions`` is ``None`` the
    full action set ``range(len(q_values))`` is considered.

    Parameters
    ----------
    q_values
        1-D array of Q-values, one entry per action in the full action set.
    counts
        1-D array of visit counts aligned with ``q_values`` (used by ``ucb``;
        ignored otherwise).
    exploration_rate
        Epsilon for ``egreedy``, temperature for ``softmax``, weight for
        ``ucb``.
    choice
        One of ``"egreedy"``, ``"softmax"``, ``"ucb"``.
    available_actions
        Optional subset of action indices that are valid in this step.

    Returns
    -------
    int
        Index of the chosen action, drawn from the full action set (so the
        caller can index ``q_values`` / ``counts`` directly with it).

    Raises
    ------
    ValueError
        If ``choice`` is not one of the three supported strategies.
    """
    n_actions = len(q_values)
    actions: list[int] = (
        list(range(n_actions)) if available_actions is None else list(available_actions)
    )

    if choice == "egreedy":
        if random.uniform(0, 1) < exploration_rate:
            if available_actions is None:
                return random.randint(0, n_actions - 1)
            return random.choice(actions)
        if available_actions is None:
            return int(np.argmax(q_values))
        return max(actions, key=lambda a: q_values[a])

    if choice == "softmax":
        tau = max(exploration_rate, 1e-6)
        if available_actions is None:
            stable_q = q_values - np.max(q_values)
            exp_q = np.exp(stable_q / tau)
            probabilities = exp_q / np.sum(exp_q)
            return int(np.random.choice(n_actions, p=probabilities))
        logits = np.array([q_values[a] for a in actions])
        stable_logits = logits - np.max(logits)
        exp_logits = np.exp(stable_logits / tau)
        probs = exp_logits / np.sum(exp_logits)
        return int(np.random.choice(actions, p=probs))

    if choice == "ucb":
        safe_counts = counts + 1e-5
        total_counts = np.sum(counts) + 1
        ucb_bonus = exploration_rate * np.sqrt(np.log(total_counts) / safe_counts)
        ucb_scores = q_values + ucb_bonus
        if available_actions is None:
            return int(np.argmax(ucb_scores))
        masked = np.full_like(ucb_scores, -np.inf)
        masked[actions] = ucb_scores[actions]
        return int(np.argmax(masked))

    raise ValueError(f"Unknown choice strategy: {choice}")


class BaseAgent(ABC):
    """Canonical agent interface used by :class:`rl_signaling.env.NetMultiAgentEnv`.

    Concrete agents implement four methods:

    - ``get_signal(state) -> int``
    - ``get_action(state) -> int``
    - ``update_signals(state, signal, reward) -> None``
    - ``update_actions(state, action, reward) -> None``

    Implemented by :class:`UrnAgent` and :class:`QLearningAgent`.
    :class:`TDLearningAgent` does not currently inherit from this ABC; its
    interface is bridged in Phase 5 alongside the env unification.
    """

    @abstractmethod
    def get_signal(self, state) -> int: ...

    @abstractmethod
    def get_action(self, state) -> int: ...

    @abstractmethod
    def update_signals(self, state, signal: int, reward: float) -> None: ...

    @abstractmethod
    def update_actions(self, state, action: int, reward: float) -> None: ...


class UrnAgent(BaseAgent):
    """Roth–Erev urn agent.

    Maintains one urn per observed state for both signaling and final actions.
    Selection probabilities are proportional to urn counts; updates add the
    reward to the chosen action's count, clamped at zero.

    Parameters
    ----------
    n_signaling_actions : int
        Number of possible signaling actions (already includes the null
        signal when ``costly_signaling=True``).
    n_final_actions : int
        Number of possible final actions.
    exploration_rate, exploration_decay, min_exploration_rate : float
        Unused for this agent — kept in the signature so all three agent
        types can be instantiated through the same factory in
        :class:`rl_signaling.env.NetMultiAgentEnv`.
    n_observed_features : int, default 1
        Number of binary features the agent observes; used by
        ``create_initial_signals`` when ``initialize=True``.
    initialize : bool, default False
        If True, pre-seed the signaling urns with one-hot vectors keyed by
        observation. Action urns are *not* currently pre-seeded — see the
        note below.
    initialization_weights : Sequence[float], default ``[1, 0]``
        ``(hot, cold)`` weights passed to ``create_initial_signals``.
    costly_signaling : bool, default False
        Stored for downstream use; this class does not enforce signaling
        cost itself (the simulation loop deducts the cost from the reward).

    Notes
    -----
    Phase 4 of ``REFACTOR_PLAN.md`` fixed the action-urn initialization bug
    that previously caused ``initialize=True`` runs to silently never
    pre-seed ``action_urns``. Behavior with ``initialize=False`` (the
    setting used by all checked-in result CSVs) is unchanged.
    """

    def __init__(
        self,
        n_signaling_actions: int,
        n_final_actions: int,
        # these are dummy parameters for the urn agent, but they help with
        # generalization across the QLearningAgent and TDLearningAgent factories
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.001,
        # these are not dummy
        n_observed_features: int = 1,
        initialize: bool = False,
        initialization_weights: Sequence[float] = (1, 0),
        costly_signaling: bool = False,
    ) -> None:
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.costly_signaling = costly_signaling

        self.signaling_urns: dict
        self.action_urns: dict
        if initialize:
            self.signaling_urns = create_initial_signals(
                n_observed_features=n_observed_features,
                n_signals=n_signaling_actions,
                n=initialization_weights[0],
                m=initialization_weights[1],
            )
            self.action_urns = create_initial_signals(
                n_observed_features=n_observed_features + 1,
                n_signals=n_final_actions,
                n=initialization_weights[0],
                m=initialization_weights[1],
            )
        else:
            self.signaling_urns = {}
            self.action_urns = {}

    def reset_urns(self) -> None:
        """Reset both signaling and action urns to empty dictionaries."""
        self.signaling_urns = {}
        self.action_urns = {}

    def get_signal(self, state) -> int:
        """Sample a signaling action proportional to the urn counts for ``state``.

        Parameters
        ----------
        state
            Observation tuple keying into the signaling urns.

        Returns
        -------
        int
            Index of the chosen signaling action.
        """
        if state not in self.signaling_urns:
            self.signaling_urns[state] = np.ones(self.n_signaling_actions)

        urn_values = self.signaling_urns[state]
        total_sum = np.sum(urn_values)

        # Safety check: if the urn is empty (sum is 0), reset to uniform to
        # avoid NaN / division-by-zero downstream.
        if total_sum <= 0:
            urn_values = np.ones(self.n_signaling_actions)
            self.signaling_urns[state] = urn_values
            total_sum = self.n_signaling_actions

        probability_weights = urn_values / total_sum
        return int(np.random.choice(self.n_signaling_actions, p=probability_weights))

    def get_action(self, state) -> int:
        """Sample a final action proportional to the urn counts for ``state``.

        Parameters
        ----------
        state
            Observation tuple (with received signals appended) keying into
            the action urns.

        Returns
        -------
        int
            Index of the chosen final action.
        """
        if state not in self.action_urns:
            self.action_urns[state] = np.ones(self.n_final_actions)

        urn_values = self.action_urns[state]
        total_sum = np.sum(urn_values)

        if total_sum <= 0:
            urn_values = np.ones(self.n_final_actions)
            self.action_urns[state] = urn_values
            total_sum = self.n_final_actions

        probability_weights = urn_values / total_sum
        return int(np.random.choice(self.n_final_actions, p=probability_weights))

    def update_signals(self, state, signal: int, reward: float) -> None:
        """Add ``reward`` to the chosen signal's urn count, clamped at zero."""
        self.signaling_urns[state][signal] = max(
            0, self.signaling_urns[state][signal] + reward
        )

    def update_actions(self, state, action: int, reward: float) -> None:
        """Add ``reward`` to the chosen action's urn count, clamped at zero."""
        self.action_urns[state][action] = max(
            0, self.action_urns[state][action] + reward
        )


class QLearningAgent(BaseAgent):
    """Q-learning agent with three exploration strategies.

    Holds two Q-tables — one for signaling, one for final actions — keyed
    by observation. Action selection follows ``choice``:

    - ``"egreedy"`` — with probability ``exploration_rate`` pick uniformly,
      else greedy.
    - ``"softmax"`` — Boltzmann sampling with temperature
      ``exploration_rate``.
    - ``"ucb"`` — upper-confidence-bound, with the bonus weight scaled by
      ``exploration_rate``.

    Parameters
    ----------
    n_signaling_actions, n_final_actions : int
        Action-space sizes (signaling already accounts for the null signal
        when ``costly_signaling=True``).
    exploration_rate : float, default 1
        Initial epsilon / temperature / UCB weight, depending on ``choice``.
    exploration_decay : float, default 0.995
        Multiplicative decay applied after every update.
    min_exploration_rate : float, default 0.001
        Floor for the decayed exploration rate.
    initialize : bool, default False
        If True, pre-seed the signaling Q-table with one-hot vectors via
        :func:`rl_signaling.games.create_initial_signals`.
    initialization_weights : Sequence[float], default ``[1, 0]``
        ``(hot, cold)`` weights passed to ``create_initial_signals``.
    n_observed_features : int, default 1
        Number of features the agent observes (used by initial seeding).
    choice : {"egreedy", "softmax", "ucb"}, default "ucb"
        Action-selection strategy.
    exp_smoothing : bool, default False
        If True, update Q-values via fixed-step exponential smoothing
        rather than constant-learning-rate TD.
    costly_signaling : bool, default False
        Stored for downstream use; cost is applied in the simulation loop.
    """

    def __init__(
        self,
        n_signaling_actions: int,
        n_final_actions: int,
        exploration_rate: float = 1,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.001,
        initialize: bool = False,
        initialization_weights: Sequence[float] = (1, 0),
        n_observed_features: int = 1,
        choice: str = "ucb",
        exp_smoothing: bool = False,
        costly_signaling: bool = False,
    ) -> None:
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.costly_signaling = costly_signaling
        self.choice = choice
        self.exp_smoothing = exp_smoothing
        self.signal_exploration_rate = exploration_rate
        self.action_exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.signaling_counts: dict = {}
        self.action_counts: dict = {}

        if initialize:
            self.q_table_signaling = create_initial_signals(
                n_observed_features=n_observed_features,
                n_signals=n_signaling_actions,
                n=initialization_weights[0],
                m=initialization_weights[1],
            )
            for state in self.q_table_signaling:
                self.signaling_counts[state] = np.zeros(self.n_signaling_actions)
        else:
            self.q_table_signaling = {}
        self.q_table_action: dict = {}

    def reset(self) -> None:
        """Reset both Q-tables and visit-count tables to empty."""
        self.q_table_signaling = {}
        self.q_table_action = {}
        self.signaling_counts = {}
        self.action_counts = {}

    def get_signal(self, state) -> int:
        """Choose a signaling action under the configured exploration strategy."""
        if state not in self.q_table_signaling:
            self.q_table_signaling[state] = np.zeros(self.n_signaling_actions)
            self.signaling_counts[state] = np.zeros(self.n_signaling_actions)

        signal = _select_action(
            q_values=self.q_table_signaling[state],
            counts=self.signaling_counts[state],
            exploration_rate=self.signal_exploration_rate,
            choice=self.choice,
        )
        self.signaling_counts[state][signal] += 1
        return signal

    def get_action(self, state) -> int:
        """Choose a final action under the configured exploration strategy."""
        if state not in self.q_table_action:
            self.q_table_action[state] = np.zeros(self.n_final_actions)
            self.action_counts[state] = np.zeros(self.n_final_actions)

        action = _select_action(
            q_values=self.q_table_action[state],
            counts=self.action_counts[state],
            exploration_rate=self.action_exploration_rate,
            choice=self.choice,
        )
        self.action_counts[state][action] += 1
        return action

    def update_signals(self, state, signal: int, reward: float) -> None:
        """Apply a TD update (or exponential smoothing) to the signaling Q-table."""
        if self.exp_smoothing:
            alpha = 0.1
            self.q_table_signaling[state][signal] = (
                (1 - alpha) * self.q_table_signaling[state][signal] + alpha * reward
            )
        else:
            td_target = reward
            td_error = td_target - self.q_table_signaling[state][signal]
            # A constant learning rate is more stable than one that decays to zero.
            learning_rate = 0.1
            self.q_table_signaling[state][signal] += learning_rate * td_error

        self.signal_exploration_rate = max(
            self.min_exploration_rate,
            self.signal_exploration_rate * self.exploration_decay,
        )

    def update_actions(self, state, action: int, reward: float) -> None:
        """Apply a TD update (or exponential smoothing) to the action Q-table."""
        if self.exp_smoothing:
            alpha = 0.1
            self.q_table_action[state][action] = (
                (1 - alpha) * self.q_table_action[state][action] + alpha * reward
            )
        else:
            td_target = reward
            td_error = td_target - self.q_table_action[state][action]
            learning_rate = 0.1
            self.q_table_action[state][action] += learning_rate * td_error

        self.action_exploration_rate = max(
            self.min_exploration_rate,
            self.action_exploration_rate * self.exploration_decay,
        )


class TDLearningAgent:
    """Temporal-difference learner with a single unified Q-table.

    Unlike :class:`UrnAgent` / :class:`QLearningAgent`, this agent uses the
    same Q-table for signal and action selection — the role is implicit in
    the ``available_actions`` argument passed to ``get_action``. The
    learning-rate schedule is ``1 / N(s, a)`` (count-based), which satisfies
    the Robbins–Monro condition under sufficient exploration.

    Parameters
    ----------
    n_actions : int
        Size of the unified action space, typically
        ``max(n_signaling_actions, n_final_actions)``.
    learning_rate : float, default 0.1
        Currently unused — the active update uses ``1 / N(s, a)``.
    exploration_rate : float, default 1.0
        Initial exploration weight.
    exploration_decay : float, default 0.995
        Multiplicative decay applied after every update.
    min_exploration_rate : float, default 0.001
        Floor for the decayed exploration rate.
    gamma : float, default 1
        Discount factor applied to bootstrapped returns.
    choice : {"egreedy", "softmax", "ucb"}, default "egreedy"
        Action-selection strategy.
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.001,
        gamma: float = 1,
        choice: str = "egreedy",
    ) -> None:
        self.n_actions = n_actions
        self.choice = choice
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.gamma = gamma
        self.q_table: dict = {}
        self.action_counts: dict = {}

    def get_action(self, state, available_actions: Sequence[int] | None = None) -> int:
        """Select an action from ``available_actions`` under ``self.choice`` strategy.

        Parameters
        ----------
        state
            Observation tuple keying into the Q-table.
        available_actions
            Subset of action indices that are valid in this step (e.g.
            signaling actions during the signal phase, final actions during
            the action phase). Defaults to all ``n_actions``.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
            self.action_counts[state] = np.zeros(self.n_actions)

        action = _select_action(
            q_values=self.q_table[state],
            counts=self.action_counts[state],
            exploration_rate=self.exploration_rate,
            choice=self.choice,
            available_actions=available_actions,
        )
        self.action_counts[state][action] += 1
        return action

    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> None:
        """Apply a one-step TD update; bootstrap from ``next_state`` unless ``done``.

        Uses a count-based learning rate ``1 / N(s, a)``.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
            self.action_counts[state] = np.zeros(self.n_actions)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.n_actions)
            self.action_counts[next_state] = np.zeros(self.n_actions)

        td_target = reward
        if not done:
            td_target += self.gamma * np.max(self.q_table[next_state])

        td_error = td_target - self.q_table[state][action]

        # ``self.action_counts[state][action] > 0`` because ``get_action``
        # already incremented it. Robbins–Monro condition is satisfied
        # provided exploration has a minimum rate and every state-action
        # pair is visited infinitely often.
        self.q_table[state][action] += td_error / self.action_counts[state][action]

        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay,
        )
