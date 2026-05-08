"""Multi-agent signaling environments.

Two environments share the role of running episodes over a directed graph
of agents:

- :class:`NetMultiAgentEnv` — single-step environment where signaling and
  the final action are chosen by separate methods on the agent
  (``get_signal`` / ``get_action``). Used with :class:`UrnAgent` and
  :class:`QLearningAgent`.
- :class:`TempNetMultiAgentEnv` — two-step environment that exposes a
  unified ``get_action(state, available_actions)`` API, used by
  :class:`TDLearningAgent`.

Unifying the two is tracked as Phase 5 in ``REFACTOR_PLAN.md``.
"""

from __future__ import annotations

import copy
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from rl_signaling.agents import QLearningAgent, TDLearningAgent, UrnAgent
from rl_signaling.info_theory import compute_mutual_information


class NetMultiAgentEnv:
    """Networked multi-agent environment with a single-step episode shape.

    The environment owns the agents and runs the canonical episode loop:
    sample nature → assign observations → encode signals → send signals
    along the graph → choose final actions → reward → update.

    Parameters
    ----------
    n_agents : int, default 2
        Number of agents (must match ``len(graph.nodes)``).
    n_features : int, default 2
        Number of binary features in the nature vector.
    n_signaling_actions : int, default 2
        Base size of the signaling action space. When ``costly_signaling``
        is True, an extra "null signal" action is appended internally —
        the agents are constructed with ``n_signaling_actions + 1``.
    n_final_actions : int, default 4
        Size of the final action space.
    exploration_rate, exploration_decay, min_exploration_rate : float
        Forwarded to the agent constructor.
    full_information : bool, default False
        If True, every agent observes the full nature vector.
    game_dicts : dict
        Mapping ``agent_index -> GameDict`` (one game per agent).
    observed_variables : dict
        Mapping ``agent_index -> list[int]`` selecting which features each
        agent observes when ``full_information=False``.
    agent_type : type, default :class:`UrnAgent`
        Agent class used to construct each player.
    initialize : bool, optional
        Forwarded to the agent constructor.
    initialization_weights : Sequence[float], default ``[1, 0]``
        Forwarded to the agent constructor.
    costly_signaling : bool, default False
        If True, append a null-signal action; the simulation loop is
        responsible for deducting the per-signal cost from rewards.
    graph : networkx.DiGraph
        Directed graph of agent connectivity. Required.

    Raises
    ------
    ValueError
        If ``graph is None`` or ``len(graph.nodes) != n_agents``.
    """

    def __init__(
        self,
        n_agents: int = 2,
        n_features: int = 2,
        n_signaling_actions: int = 2,
        n_final_actions: int = 4,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.001,
        full_information: bool = False,
        game_dicts: dict | None = None,
        observed_variables: dict | None = None,
        agent_type: type = UrnAgent,
        initialize: bool | None = None,
        initialization_weights=(1, 0),
        costly_signaling: bool = False,
        graph: nx.DiGraph | None = None,
    ) -> None:
        if graph is None:
            raise ValueError("Graph cannot be None. Please provide a valid graph structure.")

        num_nodes = len(graph.nodes)
        if num_nodes != n_agents:
            raise ValueError(
                f"Mismatch between number of agents ({n_agents}) "
                f"and number of nodes in graph ({num_nodes})."
            )

        self.n_agents = n_agents
        self.agent_type = agent_type

        # If signaling is costly, append a null-signal action.
        effective_n_signaling_actions = (
            n_signaling_actions + 1 if costly_signaling else n_signaling_actions
        )

        self.agents = [
            agent_type(
                n_signaling_actions=effective_n_signaling_actions,
                n_final_actions=n_final_actions,
                exploration_rate=exploration_rate,
                exploration_decay=exploration_decay,
                min_exploration_rate=min_exploration_rate,
                initialize=initialize,
                initialization_weights=initialization_weights,
            )
            for _ in range(self.n_agents)
        ]

        self.graph = graph

        # Environment parameters
        self.n_features = n_features
        self.n_signaling_actions = effective_n_signaling_actions
        self.n_final_actions = n_final_actions
        self.current_step = 0
        self.costly_signaling = costly_signaling
        self.full_information = full_information

        self.internal_game_dicts = game_dicts if game_dicts is not None else {}
        self.agents_observed_variables = (
            observed_variables if observed_variables is not None else {}
        )

        # Episode state
        self.nature_vector: NDArray[np.int_] | None = None
        self.signals: list[int] | None = None
        self.final_actions: list[int] | None = None

        # Per-agent history
        self.rewards_history: list[list[float]] = [[] for _ in range(self.n_agents)]
        self.signal_usage: list[dict] = [{} for _ in range(self.n_agents)]
        self.action_usage: list[dict] = [{} for _ in range(self.n_agents)]
        self.signal_information_history: list[list[float]] = [
            [] for _ in range(self.n_agents)
        ]
        self.nature_history: list[tuple[int, ...]] = []

        # Snapshots of usage dictionaries at the end of each episode.
        # NOTE: appending deepcopies every episode is memory-inefficient for
        # long runs; kept for plotting compatibility (REFACTOR_PLAN, Status).
        self.histories: dict[int, dict[str, list]] = {}
        for i in range(self.n_agents):
            self.histories[i] = {"signal_history": [], "action_history": []}

    def nature_sample(self) -> NDArray[np.int_]:
        """Sample a fresh binary nature vector and reset ``current_step``."""
        self.current_step = 0
        self.nature_vector = np.random.randint(0, 2, size=self.n_features)
        return self.nature_vector

    def encoding_signals(self, agents_observations: list[tuple]) -> list[int]:
        """Run the signal-selection step for every agent.

        Updates ``signal_usage`` and appends the per-agent normalized
        mutual information to ``signal_information_history``.

        Parameters
        ----------
        agents_observations
            Observation tuple per agent, in agent-index order.

        Returns
        -------
        list[int]
            Selected signal index per agent.
        """
        signals = [
            agent.get_signal(observation)
            for agent, observation in zip(self.agents, agents_observations)
        ]

        for i in range(self.n_agents):
            agent_observation = agents_observations[i]

            if agent_observation not in self.signal_usage[i]:
                self.signal_usage[i][agent_observation] = np.zeros(
                    self.n_signaling_actions
                )

            if not (0 <= signals[i] < self.n_signaling_actions):
                raise ValueError(
                    f"Signal {signals[i]} is out of range "
                    f"({self.n_signaling_actions}) for agent {i}"
                )
            self.signal_usage[i][agent_observation][signals[i]] += 1

        for i in range(self.n_agents):
            _, normalized_mutual_info = compute_mutual_information(self.signal_usage[i])
            self.signal_information_history[i].append(normalized_mutual_info)

        # encoding_signals is step 2, regardless of whether the signal is sent
        self.current_step = 2
        return signals

    def send_signals(
        self, signals: list[int], agents_observations: list[tuple]
    ) -> list[tuple]:
        """Append received signals to each agent's observation along graph edges.

        When ``costly_signaling=True`` the null signal (highest index) is
        suppressed: receivers do not append it.
        """
        new_observations = copy.deepcopy(agents_observations)
        for i in range(self.n_agents):
            in_neighbors = self.graph.predecessors(i)
            for neig in in_neighbors:
                if self.costly_signaling:
                    null_signal_index = self.n_signaling_actions - 1
                    if signals[neig] != null_signal_index:
                        new_observations[i] = new_observations[i] + (signals[neig],)
                else:
                    new_observations[i] = new_observations[i] + (signals[neig],)
        return new_observations

    def get_actions(self, agents_observations: list[tuple]) -> list[int]:
        """Run the final-action step for every agent and update ``action_usage``."""
        final_actions = [
            agent.get_action(observation)
            for agent, observation in zip(self.agents, agents_observations)
        ]

        for i in range(self.n_agents):
            agent_observation = agents_observations[i]
            if agent_observation not in self.action_usage[i]:
                self.action_usage[i][agent_observation] = np.zeros(self.n_final_actions)

            if not (0 <= final_actions[i] < self.n_final_actions):
                raise ValueError(
                    f"Action {final_actions[i]} is out of range for agent {i}"
                )
            self.action_usage[i][agent_observation][final_actions[i]] += 1

        self.current_step = 3
        return final_actions

    def play_step(self, final_actions: list[int]) -> tuple[list[float], bool]:
        """Look up rewards from each agent's game dict given the chosen action.

        Returns
        -------
        (rewards, done) : (list[float], bool)
            ``done`` is always True — episodes are single-step.
        """
        rewards: list[float] = []
        for i in range(self.n_agents):
            agent_action = final_actions[i]
            state_key = tuple(self.nature_vector)

            if (
                state_key in self.internal_game_dicts[i]
                and agent_action in self.internal_game_dicts[i][state_key]
            ):
                rewards.append(self.internal_game_dicts[i][state_key][agent_action])
            else:
                raise KeyError(
                    f"Invalid state-action pair ({state_key}, {agent_action}) for agent {i}"
                )

        self.current_step = 4
        return rewards, True

    def update_agents(
        self,
        nature_observations: list[tuple],
        new_observations: list[tuple],
        signals: list[int] | None,
        final_actions: list[int],
        rewards: list[float],
    ) -> None:
        """Apply per-agent signal/action updates and snapshot per-episode usage."""
        for i in range(self.n_agents):
            self.rewards_history[i].append(rewards[i])

        for i in range(self.n_agents):
            if signals is not None:
                self.agents[i].update_signals(
                    nature_observations[i], signals[i], rewards[i]
                )
            self.agents[i].update_actions(
                new_observations[i], final_actions[i], rewards[i]
            )

        # Snapshot usage at the end of the episode. This is memory-inefficient
        # because each entry is a deepcopy of the cumulative dictionaries;
        # kept for plotting compatibility — see REFACTOR_PLAN Status.
        for i in range(self.n_agents):
            self.histories[i]["signal_history"].append(copy.deepcopy(self.signal_usage[i]))
            self.histories[i]["action_history"].append(copy.deepcopy(self.action_usage[i]))

        self.current_step = 5

    def report_metrics(
        self,
    ) -> tuple[list[dict], list[list[float]], list[list[float]], list[tuple], dict]:
        """Return the canonical 5-tuple of per-episode metrics."""
        return (
            self.signal_usage,
            self.rewards_history,
            self.signal_information_history,
            self.nature_history,
            self.histories,
        )

    def render(self) -> None:
        """Print the current step, nature vector, signals, and final actions."""
        print(f"Step: {self.current_step}")
        print(f"Nature Vector: {self.nature_vector}")
        print(f"Signals: {self.signals}")
        print(f"Final Actions: {self.final_actions}")

    def assign_observations(self, nature_vector) -> list[tuple]:
        """Project the nature vector to per-agent observation tuples."""
        self.nature_history.append(tuple(nature_vector))
        agents_observations: list[tuple] = []
        if self.full_information:
            for _ in range(self.n_agents):
                agents_observations.append(tuple(nature_vector))
        else:
            for i in range(self.n_agents):
                observed_indexes = self.agents_observed_variables[i]
                subset = tuple(nature_vector[j] for j in observed_indexes)
                agents_observations.append(subset)
        self.current_step = 1
        return agents_observations


class TempNetMultiAgentEnv:
    """Two-step multi-agent environment used with :class:`TDLearningAgent`.

    Each episode unfolds as ``signal`` → ``act``. ``get_actions`` dispatches
    on ``self.step_type`` to expose either the signaling action set or the
    final action set, allowing one agent class with one ``get_action`` API
    to handle both phases.

    Parameters mirror :class:`NetMultiAgentEnv`. The only additional one is
    ``learning_rate``, forwarded to :class:`TDLearningAgent`.
    """

    def __init__(
        self,
        n_agents: int = 2,
        n_features: int = 2,
        n_signaling_actions: int = 2,
        n_final_actions: int = 4,
        learning_rate: float = 0.1,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration_rate: float = 0.001,
        full_information: bool = False,
        game_dicts: dict | None = None,
        observed_variables: dict | None = None,
        agent_type: type | None = None,
        graph: nx.DiGraph | None = None,
    ) -> None:
        if graph is None:
            raise ValueError("Graph cannot be None.")
        if len(graph.nodes) != n_agents:
            raise ValueError("Number of agents must match number of graph nodes.")
        if agent_type is None:
            raise ValueError("agent_type cannot be None.")

        self.n_agents = n_agents
        self.n_features = n_features
        self.n_signaling_actions = n_signaling_actions
        self.n_final_actions = n_final_actions
        self.full_information = full_information
        self.graph = graph
        self.game_dicts = game_dicts or {}
        self.observed_variables = observed_variables or {}
        self.agent_type = agent_type

        self.max_actions = max(n_signaling_actions, n_final_actions)
        self.agents = [
            agent_type(
                n_actions=self.max_actions,
                learning_rate=learning_rate,
                exploration_rate=exploration_rate,
                exploration_decay=exploration_decay,
                min_exploration_rate=min_exploration_rate,
            )
            for _ in range(n_agents)
        ]

        self.nature_vector: NDArray[np.int_] | None = None
        self.rewards_history: list[list[float]] = [[] for _ in range(n_agents)]
        self.action_usage: list[dict] = [{} for _ in range(n_agents)]
        self.signal_usage: list[dict] = [{} for _ in range(n_agents)]
        self.signal_information_history: list[list[float]] = [[] for _ in range(n_agents)]
        self.histories: dict[int, dict[str, list]] = {
            i: {"signal_history": [], "action_history": []} for i in range(n_agents)
        }
        self.nature_history: list[tuple[int, ...]] = []
        self.step_type: str = "signal"
        self.signals: list[Any] = [None] * n_agents

    def nature_sample(self) -> NDArray[np.int_]:
        """Sample a fresh binary nature vector."""
        self.nature_vector = np.random.randint(0, 2, size=self.n_features)
        return self.nature_vector

    def assign_observations(self, nature_vector) -> list[tuple]:
        """Project the nature vector to per-agent observation tuples."""
        self.nature_history.append(tuple(nature_vector))
        agents_observations: list[tuple] = []
        for i in range(self.n_agents):
            if self.full_information:
                obs = tuple(nature_vector)
            else:
                idxs = self.observed_variables[i]
                obs = tuple(nature_vector[j] for j in idxs)
            agents_observations.append(obs)
        return agents_observations

    def communicate(self, observations: list[tuple]) -> list[tuple]:
        """Append in-neighbours' last-step signals to each agent's observation."""
        new_obs = list(observations)
        for i in range(self.n_agents):
            for neighbor in self.graph.predecessors(i):
                signal = self.signals[neighbor]
                new_obs[i] = new_obs[i] + (signal,)
        return new_obs

    def get_available_actions(self) -> list[int]:
        """Return the action subset valid in the current step (signal vs. act)."""
        if self.step_type == "signal":
            return list(range(self.n_signaling_actions))
        elif self.step_type == "act":
            return list(range(self.n_final_actions))
        return []

    def get_actions(self, observations: list[tuple]) -> list[int]:
        """Choose actions for every agent given the current ``step_type``.

        Updates ``signal_usage`` (and ``signal_information_history``) during
        the signal step; updates ``action_usage`` during the act step.
        """
        actions: list[int] = []
        available_actions = self.get_available_actions()
        for i, (agent, obs) in enumerate(zip(self.agents, observations)):
            action = agent.get_action(obs, available_actions)
            actions.append(action)

            if self.step_type == "signal":
                if obs not in self.signal_usage[i]:
                    self.signal_usage[i][obs] = np.zeros(self.n_signaling_actions)
                self.signal_usage[i][obs][action] += 1
                # Compute and record mutual information of signals
                for j in range(self.n_agents):
                    _, normalized_mutual_info = compute_mutual_information(
                        self.signal_usage[j]
                    )
                    self.signal_information_history[j].append(normalized_mutual_info)
            else:
                if obs not in self.action_usage[i]:
                    self.action_usage[i][obs] = np.zeros(self.n_final_actions)
                self.action_usage[i][obs][action] += 1

        return actions

    def play_step(self, actions: list[int]) -> tuple[list[float], bool]:
        """Advance the two-step state machine and emit per-step rewards.

        ``signal`` step yields zero reward and ``done=False``;
        ``act`` step yields the game-dict reward and ``done=True``.
        """
        if self.step_type == "signal":
            self.signals = actions
            self.step_type = "act"
            rewards: list[float] = [0.0] * self.n_agents
            return rewards, False

        elif self.step_type == "act":
            rewards = []
            for i, action in enumerate(actions):
                key = tuple(self.nature_vector)
                reward = self.game_dicts[i].get(key, {}).get(action, 0)
                rewards.append(reward)
                self.rewards_history[i].append(reward)
            self.step_type = "done"
            return rewards, True

        raise RuntimeError("Environment is already done.")

    def update_agents(
        self,
        old_obs: list[tuple],
        actions: list[int],
        rewards: list[float],
        new_obs: list[tuple],
        done: bool,
    ) -> None:
        """Apply each agent's TD update and snapshot per-step usage."""
        for i in range(self.n_agents):
            self.agents[i].update(
                state=old_obs[i],
                action=actions[i],
                reward=rewards[i],
                next_state=new_obs[i],
                done=done,
            )
            self.histories[i]["signal_history"].append(
                copy.deepcopy(self.signal_usage[i])
            )
            self.histories[i]["action_history"].append(
                copy.deepcopy(self.action_usage[i])
            )

    def report_metrics(
        self,
    ) -> tuple[list[dict], list[list[float]], list[list[float]], list[tuple], dict]:
        """Return the canonical 5-tuple of per-episode metrics."""
        return (
            self.signal_usage,
            self.rewards_history,
            self.signal_information_history,
            self.nature_history,
            self.histories,
        )

    def render(self) -> None:
        """Print the current step type, nature vector, and last signals."""
        print(f"Step type: {self.step_type}")
        print(f"Nature Vector: {self.nature_vector}")
        print(f"Signals: {self.signals}")
