"""Independent verification of rl_signaling.info_theory using scipy as the reference.

For each known case in analytics/math/information_theory.md, computes:
  (a) the quantity from `rl_signaling.info_theory`, and
  (b) the same quantity from `scipy.stats.entropy` (independent implementation),
asserts agreement to absolute tolerance 1e-12, and prints PASS/FAIL.

Run:
    .venv/bin/python -m analytics.scripts.verify_information_theory
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.stats import entropy as scipy_entropy

from rl_signaling.info_theory import _compute_entropy, compute_mutual_information

ATOL = 1e-12

failures: list[str] = []


def check(label: str, lhs: float, rhs: float, atol: float = ATOL) -> None:
    """Compare two scalars and record a PASS/FAIL line."""
    diff = abs(lhs - rhs)
    status = "PASS" if diff <= atol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: rl_signaling={lhs!r}, reference={rhs!r}, diff={diff:.3e}")


# -----------------------------------------------------------------------------
# Section 1 — Shannon entropy in bits (log base 2)
# -----------------------------------------------------------------------------
print("=== Shannon entropy (log base 2) ===")

cases: list[tuple[str, list[float]]] = [
    ("H([1.0]) = 0", [1.0]),
    ("H([0.5, 0.5]) = 1 bit", [0.5, 0.5]),
    ("H([0.25]*4) = 2 bits", [0.25] * 4),
    ("H([0.5, 0.25, 0.125, 0.125]) = 1.75 bits", [0.5, 0.25, 0.125, 0.125]),
    ("H(uniform 8) = 3 bits", [1 / 8] * 8),
    ("H([0.7, 0.3]) ≈ 0.881291…", [0.7, 0.3]),
]

for label, p in cases:
    rl_value = _compute_entropy(p)
    scipy_value = scipy_entropy(p, base=2)
    check(label, rl_value, scipy_value)


# -----------------------------------------------------------------------------
# Section 2 — Mutual information from a 2x2 signal-usage table
# -----------------------------------------------------------------------------
print("\n=== Mutual information / NMI ===")


def reference_nmi_asymmetric(usage: dict) -> tuple[float, float]:
    """Independent (S, O) -> (I, NMI) using scipy.stats.entropy and direct summation.

    Treats `usage` as: usage[obs][signal_index] = count.
    Computes I(S;O) and NMI = I(S;O) / H(O), where O is the observation marginal
    and S is the signal marginal. This mirrors the rl_signaling convention.
    """
    obs_keys = list(usage.keys())
    n_signals = len(usage[obs_keys[0]])
    counts = np.array([usage[o] for o in obs_keys], dtype=float)  # shape (|O|, |S|)
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0

    p_o = counts.sum(axis=1) / total                   # marginal over observations
    p_s = counts.sum(axis=0) / total                   # marginal over signals
    h_o = scipy_entropy(p_o, base=2)
    h_s = scipy_entropy(p_s, base=2)

    # H(S | O) = sum_o P(o) * H(S | O = o)
    h_s_given_o = 0.0
    for i, o in enumerate(obs_keys):
        row_total = counts[i].sum()
        if row_total == 0:
            continue
        cond = counts[i] / row_total
        h_s_given_o += p_o[i] * scipy_entropy(cond, base=2)

    mi = h_s - h_s_given_o
    nmi = mi / h_o if h_o > 0 else 0.0
    return mi, nmi


# Case A — perfect 2x2 correlation
usage_perfect = {
    (0,): np.array([10.0, 0.0]),
    (1,): np.array([0.0, 10.0]),
}
rl_mi, rl_nmi = compute_mutual_information(usage_perfect)
ref_mi, ref_nmi = reference_nmi_asymmetric(usage_perfect)
check("perfect 2x2 — I(S;O) = 1 bit", rl_mi, ref_mi)
check("perfect 2x2 — NMI = 1.0", rl_nmi, ref_nmi)

# Case B — independence
usage_indep = {
    (0,): np.array([5.0, 5.0]),
    (1,): np.array([5.0, 5.0]),
}
rl_mi, rl_nmi = compute_mutual_information(usage_indep)
ref_mi, ref_nmi = reference_nmi_asymmetric(usage_indep)
check("independence — I(S;O) = 0", rl_mi, ref_mi)
check("independence — NMI = 0", rl_nmi, ref_nmi)

# Case C — partial correlation
usage_partial = {
    (0,): np.array([8.0, 2.0]),
    (1,): np.array([3.0, 7.0]),
}
rl_mi, rl_nmi = compute_mutual_information(usage_partial)
ref_mi, ref_nmi = reference_nmi_asymmetric(usage_partial)
check("partial 2x2 — I(S;O) in (0, 1)", rl_mi, ref_mi)
check("partial 2x2 — NMI in (0, 1)", rl_nmi, ref_nmi)

# Case D — three-symbol alphabet
usage_3x3 = {
    (0,): np.array([5.0, 0.0, 0.0]),
    (1,): np.array([0.0, 5.0, 0.0]),
    (2,): np.array([0.0, 0.0, 5.0]),
}
rl_mi, rl_nmi = compute_mutual_information(usage_3x3)
ref_mi, ref_nmi = reference_nmi_asymmetric(usage_3x3)
check("perfect 3x3 — I(S;O) = log2(3)", rl_mi, ref_mi)
check("perfect 3x3 — NMI = 1.0", rl_nmi, ref_nmi)

# Case E — H(O) = 0 convention
usage_constant = {(0,): np.array([3.0, 7.0])}
rl_mi, rl_nmi = compute_mutual_information(usage_constant)
ref_mi, ref_nmi = reference_nmi_asymmetric(usage_constant)
check("H(O) = 0 — NMI := 0 by convention", rl_nmi, 0.0)
check("H(O) = 0 — reference NMI := 0", ref_nmi, 0.0)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n=== Summary ===")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"All checks passed (atol = {ATOL}).")
