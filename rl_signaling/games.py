"""Game generators and signal-urn initializers."""

from __future__ import annotations

import random
from itertools import product

import numpy as np
from numpy.typing import NDArray

GameDict = dict[tuple[int, ...], dict[int, float]]
SignalUrns = dict[tuple[int, ...], NDArray[np.float64]]


def create_random_game(n_features: int = 3, n_final_actions: int = 5) -> GameDict:
    """Generate a random per-state, per-action payoff dictionary.

    Each binary state of length ``n_features`` is assigned an independent
    uniform integer reward in ``[0, 9]`` for every action — there is no
    structure across states or actions.

    Parameters
    ----------
    n_features : int, default 3
        Number of binary features defining the world state.
    n_final_actions : int, default 5
        Number of final actions an agent can take.

    Returns
    -------
    GameDict
        Mapping ``state -> {action -> reward}`` covering all ``2**n_features``
        binary states.

    """
    random_game_dict: GameDict = {}
    world_states = set(product([0, 1], repeat=n_features))
    for w in world_states:
        random_game_dict[w] = {}
        for a in range(n_final_actions):
            random_game_dict[w][a] = random.randint(0, 9)
    return random_game_dict


def _generate_unique_dicts(
    n_final_actions: int, n: float = 1, m: float = 0
) -> list[dict[int, float]]:
    """Generate one-hot payoff dictionaries: one action gets ``n``, others get ``m``."""
    return [
        {i: (n if i == j else m) for i in range(n_final_actions)}
        for j in range(n_final_actions)
    ]


def create_random_canonical_game(
    n_features: int,
    n_final_actions: int,
    n: float = 1,
    m: float = 0,
) -> GameDict:
    """Generate a canonical-form game where each state has a unique optimal action.

    Each world state is paired with a distinct one-hot payoff dict so the
    optimal action varies across states. This is the primary generator used
    in the experiments.

    Parameters
    ----------
    n_features : int
        Number of binary features defining the world state.
    n_final_actions : int
        Number of final actions; must be ``>= 2**n_features``.
    n : float, default 1
        Reward assigned to the unique optimal action for a given state.
    m : float, default 0
        Reward assigned to every non-optimal action.

    Returns
    -------
    GameDict
        Mapping ``state -> {action -> reward}`` where each state has exactly
        one action with reward ``n`` and all others with reward ``m``.

    Raises
    ------
    AssertionError
        If ``n_final_actions < 2**n_features`` (not enough one-hot dicts).

    """
    random_game_dict: GameDict = {}
    world_states = list(product([0, 1], repeat=n_features))
    unique_dicts = _generate_unique_dicts(n_final_actions, n, m)

    assert len(world_states) <= len(unique_dicts), (
        "Not enough unique dictionaries for the given states"
    )

    random.shuffle(unique_dicts)
    for w, unique_dict in zip(world_states, unique_dicts):
        random_game_dict[w] = unique_dict

    return random_game_dict


def _generate_hot_vectors(
    n_signals: int, n: float = 1, m: float = 0
) -> list[NDArray[np.float64]]:
    """Generate ``n_signals`` one-hot vectors of length ``n_signals``.

    Returns float64 arrays so downstream constant-α TD updates and
    fractional-reward urn updates do not silently truncate (in-place
    addition of a float into an int array casts the result back to int).
    """
    return [
        np.array(
            [n if i == j else m for i in range(n_signals)],
            dtype=np.float64,
        )
        for j in range(n_signals)
    ]


def create_initial_signals(
    n_observed_features: int,
    n_signals: int,
    n: float = 1,
    m: float = 0,
) -> SignalUrns:
    """Build a deterministic signal-urn map: each observation → unique one-hot vector.

    Used by agents that support pre-seeded signaling urns (``UrnAgent``,
    ``QLearningAgent`` when ``initialize=True``).

    Parameters
    ----------
    n_observed_features : int
        Number of binary features the agent observes.
    n_signals : int
        Number of signals available; must be ``>= 2**n_observed_features``.
    n : float, default 1
        Value at the "hot" position of each one-hot vector.
    m : float, default 0
        Value at the other positions.

    Returns
    -------
    SignalUrns
        Mapping ``observation -> one-hot signal vector``.

    Raises
    ------
    AssertionError
        If ``n_signals < 2**n_observed_features`` (not enough unique vectors).

    """
    signalling_urns: SignalUrns = {}
    observed_states = list(product([0, 1], repeat=n_observed_features))
    one_hot_vectors = _generate_hot_vectors(n_signals, n, m)

    assert len(observed_states) <= len(one_hot_vectors), (
        "Not enough unique vectors for the given states"
    )

    random.shuffle(one_hot_vectors)
    for o, vector in zip(observed_states, one_hot_vectors):
        signalling_urns[o] = vector

    return signalling_urns
