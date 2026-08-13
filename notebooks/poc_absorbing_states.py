"""Absorbing-state enumeration for the §2.2 proof-of-concept figure.

`enumerate_absorbing_rewards` returns the per-agent mean reward over every
absorbing state of the canonical two-agent signal-trading game. It backs the
Option C panel of `proof_of_concept_figures_final.ipynb`, which `results/MANIFEST.md`
records as the provenance for the paper's §2.2 figure.

This function previously lived in `analytics/scripts/figure_poc_options.py`. The
`analytics/` tree is kept local and is no longer distributed with this repository,
so the one helper the published notebook needs is vendored here. The derivation it
implements is documented in `analytics/math/`, alongside the rest of that
local-only mathematical reference.
"""

import itertools
import random

import numpy as np

from rl_signaling.games import create_random_canonical_game

# Canonical §2.2 game shape: 2 binary world features, 2 signals, 4 actions.
N_FEATURES = 2
N_SIG = 2
N_ACT = 4


def enumerate_absorbing_rewards(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return per-agent mean reward (r0, r1) over the 2304 absorbing states."""
    np.random.seed(seed)
    random.seed(seed)
    games = {i: create_random_canonical_game(N_FEATURES, N_ACT, n=1, m=0) for i in range(2)}
    world_states = list(itertools.product([0, 1], repeat=N_FEATURES))

    # bijection: 2! = 2 signaling maps; 4! = 24 action maps; per-agent = 48.
    sig_maps = list(itertools.permutations(range(N_SIG)))  # 2 of these
    act_keys = list(itertools.product([0, 1], range(N_SIG)))  # 4 (obs, sig) keys
    act_maps = list(itertools.permutations(range(N_ACT)))  # 24

    r0_list, r1_list = [], []
    for f0 in sig_maps:
        for f1 in sig_maps:
            for g0 in act_maps:
                for g1 in act_maps:
                    r0 = 0.0; r1 = 0.0
                    for (x, y) in world_states:
                        sig0 = f0[x]; sig1 = f1[y]
                        a0 = g0[act_keys.index((x, sig1))]
                        a1 = g1[act_keys.index((y, sig0))]
                        r0 += games[0][(x, y)][a0]
                        r1 += games[1][(x, y)][a1]
                    r0_list.append(r0 / 4); r1_list.append(r1 / 4)
    return np.array(r0_list), np.array(r1_list)
