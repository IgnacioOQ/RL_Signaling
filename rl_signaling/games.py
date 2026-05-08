"""Game generators and signal-urn initializers."""

import random
from itertools import product

import numpy as np


def create_random_game(n_features=3, n_final_actions=5):
    """Generate a random per-state, per-action payoff dictionary.

    Each binary state of length `n_features` gets an independent uniform
    integer reward in [0, 9] for every action.
    """
    random_game_dict = dict()
    world_states = set(product([0, 1], repeat=n_features))
    for w in world_states:
        random_game_dict[w] = dict()
        for a in range(n_final_actions):
            random_game_dict[w][a] = random.randint(0, 9)
    return random_game_dict


def _generate_unique_dicts(n_final_actions, n=1, m=0):
    """Generate one-hot payoff dictionaries: one action gets `n`, others get `m`."""
    return [
        {i: (n if i == j else m) for i in range(n_final_actions)}
        for j in range(n_final_actions)
    ]


def create_random_canonical_game(n_features, n_final_actions, n=1, m=0):
    """Generate a canonical-form game where each state has a unique optimal action.

    Each world state is paired with a distinct one-hot payoff dict so the
    optimal action varies across states.
    """
    random_game_dict = dict()
    world_states = list(product([0, 1], repeat=n_features))
    unique_dicts = _generate_unique_dicts(n_final_actions, n, m)

    assert len(world_states) <= len(unique_dicts), (
        "Not enough unique dictionaries for the given states"
    )

    random.shuffle(unique_dicts)
    for w, unique_dict in zip(world_states, unique_dicts):
        random_game_dict[w] = unique_dict

    return random_game_dict


def _generate_hot_vectors(n_signals, n=1, m=0):
    """Generate `n_signals` one-hot vectors of length `n_signals`."""
    return [
        np.array([n if i == j else m for i in range(n_signals)])
        for j in range(n_signals)
    ]


def create_initial_signals(n_observed_features, n_signals, n=1, m=0):
    """Build a deterministic signal-urn map: each observation → unique one-hot vector.

    Used by agents that support pre-seeded signaling urns (UrnAgent, QLearningAgent
    when initialize=True).
    """
    signalling_urns = dict()
    observed_states = list(product([0, 1], repeat=n_observed_features))
    one_hot_vectors = _generate_hot_vectors(n_signals, n, m)

    assert len(observed_states) <= len(one_hot_vectors), (
        "Not enough unique vectors for the given states"
    )

    random.shuffle(one_hot_vectors)
    for o, vector in zip(observed_states, one_hot_vectors):
        signalling_urns[o] = vector

    return signalling_urns
