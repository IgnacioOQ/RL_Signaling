"""Information-theoretic identities on hand-built signal-usage tables."""

from __future__ import annotations

import numpy as np
import pytest

from rl_signaling.info_theory import compute_mutual_information


def test_perfect_correlation_gives_unit_nmi():
    """Each observation deterministically maps to a distinct signal → NMI = 1."""
    usage = {
        (0,): np.array([10.0, 0.0]),
        (1,): np.array([0.0, 10.0]),
    }
    mi, nmi = compute_mutual_information(usage)
    assert mi > 0
    assert nmi == pytest.approx(1.0, abs=1e-9)


def test_independence_gives_zero_nmi():
    """Uniform conditional → MI = 0 → NMI = 0."""
    usage = {
        (0,): np.array([5.0, 5.0]),
        (1,): np.array([5.0, 5.0]),
    }
    mi, nmi = compute_mutual_information(usage)
    assert mi == pytest.approx(0.0, abs=1e-9)
    assert nmi == pytest.approx(0.0, abs=1e-9)


def test_single_observation_gives_zero_nmi():
    """If only one observation occurs, ``H(O) = 0`` so NMI defaults to 0."""
    usage = {(0,): np.array([3.0, 7.0])}
    mi, nmi = compute_mutual_information(usage)
    assert nmi == 0


def test_partial_correlation_in_unit_interval():
    """For a noisy mapping, NMI lies strictly between 0 and 1."""
    usage = {
        (0,): np.array([8.0, 2.0]),
        (1,): np.array([3.0, 7.0]),
    }
    mi, nmi = compute_mutual_information(usage)
    assert 0 < nmi < 1
    assert mi > 0


def test_three_signals_three_observations():
    """Three deterministic observation→signal pairs → NMI = 1 across larger alphabets."""
    usage = {
        (0,): np.array([5.0, 0.0, 0.0]),
        (1,): np.array([0.0, 5.0, 0.0]),
        (2,): np.array([0.0, 0.0, 5.0]),
    }
    _, nmi = compute_mutual_information(usage)
    assert nmi == pytest.approx(1.0, abs=1e-9)
