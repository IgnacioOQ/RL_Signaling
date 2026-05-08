"""Information-theoretic metrics over an agent's signal-usage statistics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

SignalUsage = Mapping[tuple, NDArray[np.float64]]


def _compute_entropy(probabilities: Iterable[float]) -> float:
    """Shannon entropy (base 2) of a discrete probability distribution."""
    return -sum(p * np.log2(p) for p in probabilities if p > 0)


def compute_mutual_information(agent_signal_usage: SignalUsage) -> tuple[float, float]:
    """Compute mutual information and normalized MI between signals and observations.

    Parameters
    ----------
    agent_signal_usage : dict
        Mapping ``observation -> array of signal counts`` for a single agent.

    Returns
    -------
    (float, float)
        ``(I(S;O), NMI)`` where ``NMI = I(S;O) / H(O)``. Returns ``NMI=0``
        when ``H(O) == 0``.
    """
    total_signals = sum(sum(counts) for counts in agent_signal_usage.values())

    # P(S): overall signal probabilities
    signal_counts = defaultdict(int)
    for counts in agent_signal_usage.values():
        for s, count in enumerate(counts):
            signal_counts[s] += count
    P_S = {s: count / total_signals for s, count in signal_counts.items()}

    # P(O): observation probabilities
    P_O = {o: sum(counts) / total_signals for o, counts in agent_signal_usage.items()}

    # H(S): entropy of signals
    H_S = _compute_entropy(P_S.values())

    # H(S | O): conditional entropy of signals given observations
    H_S_given_O = 0
    for o, counts in agent_signal_usage.items():
        P_S_given_O = [count / sum(counts) for count in counts]
        H_S_given_O += P_O[o] * _compute_entropy(P_S_given_O)

    # H(O): entropy of observations
    H_O = _compute_entropy(P_O.values())

    I_S_O = H_S - H_S_given_O
    NMI = I_S_O / H_O if H_O > 0 else 0

    return I_S_O, NMI
