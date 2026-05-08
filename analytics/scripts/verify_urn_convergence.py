"""Independent verification of UrnAgent — closed-form sampling probability and Monte-Carlo convergence.

Cross-checks the math derived in analytics/agent_urn.md:

  1. After n updates of action a* with reward r > 0 (constant) starting from urn = ones:
        urn[a*] = u_0 + n*r
        P[a*] = (u_0 + n*r) / (u_0 + n*r + (K-1)*u_0)
                = (1 + n*r/u_0) / (K + n*r/u_0)

  2. Monte-Carlo: compare empirical sampling frequency from many independent agents
     against the closed form above.

Run:
    .venv/bin/python -m analytics.scripts.verify_urn_convergence
"""

from __future__ import annotations

import sys

import numpy as np

from rl_signaling.agents import UrnAgent

ATOL_EXACT = 1e-12
RTOL_MC = 0.05  # 5% relative tolerance for Monte-Carlo sampling frequency

failures: list[str] = []


def check_exact(label: str, lhs: float, rhs: float, atol: float = ATOL_EXACT) -> None:
    diff = abs(lhs - rhs)
    status = "PASS" if diff <= atol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: rl_signaling={lhs!r}, expected={rhs!r}, diff={diff:.3e}")


def check_relative(label: str, lhs: float, rhs: float, rtol: float = RTOL_MC) -> None:
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)
    status = "PASS" if rel <= rtol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: empirical={lhs:.6g}, closed_form={rhs:.6g}, rel_diff={rel:.3e}")


# -----------------------------------------------------------------------------
# Section 1 — Urn weight closed form: urn[a*] = u_0 + n*r after n forced pulls
# -----------------------------------------------------------------------------
print("=== Urn weight closed form ===")

K = 4         # alphabet size
u_0 = 1       # initial weight (lazy-init = ones)
r = 1.0       # constant reward
A_STAR = 2    # the "optimal" action

a = UrnAgent(n_signaling_actions=K, n_final_actions=K)
state = (0,)
a.get_signal(state)  # populates urn with ones (lazy-init)

# Force-update action A_STAR n times. urn[a*] should increment by r each time.
for n in range(1, 21):
    a.update_signals(state, A_STAR, reward=r)
    expected_weight = u_0 + n * r
    actual_weight = a.signaling_urns[state][A_STAR]
    if n in {1, 5, 10, 20}:
        check_exact(f"urn[a*] after {n} updates", actual_weight, expected_weight)
    elif abs(actual_weight - expected_weight) > ATOL_EXACT:
        failures.append(f"urn weight silent mismatch at n={n}")


# -----------------------------------------------------------------------------
# Section 2 — Sampling probability closed form
# -----------------------------------------------------------------------------
print("\n=== Sampling probability closed form ===")

# After 250 forced updates of A_STAR (the average # of visits to one of 4 states
# in 1000 episodes), the sampling probability for A_STAR should be:
#   P = (1 + 250/1) / (4 + 250/1) = 251/254 ≈ 0.98818...

a = UrnAgent(n_signaling_actions=K, n_final_actions=K)
state = (0,)
a.get_signal(state)

N = 250
for _ in range(N):
    a.update_signals(state, A_STAR, reward=r)

urn_weights = a.signaling_urns[state]
total = urn_weights.sum()
analytical_prob = urn_weights[A_STAR] / total
closed_form_prob = (1 + N * r / u_0) / (K + N * r / u_0)
check_exact(f"P[a*] after {N} updates = (1 + N) / (K + N)", analytical_prob, closed_form_prob)


# -----------------------------------------------------------------------------
# Section 3 — Monte-Carlo: empirical sampling matches the closed form
# -----------------------------------------------------------------------------
print("\n=== Monte-Carlo sampling frequency ===")

# Take the urn at the state above (with 250 reinforcements on A_STAR) and sample
# many times. The empirical fraction of A_STAR should match closed_form_prob.

N_SAMPLES = 200_000

# Set a seed for reproducibility (np.random.choice in get_signal uses the global RNG).
np.random.seed(7)

samples = np.array([a.get_signal(state) for _ in range(N_SAMPLES)])
empirical_frac = float((samples == A_STAR).sum()) / N_SAMPLES
check_relative(f"empirical P[a*] over {N_SAMPLES} samples", empirical_frac, closed_form_prob)


# -----------------------------------------------------------------------------
# Section 4 — Clamp at zero: negative reward cannot push urn weight negative
# -----------------------------------------------------------------------------
print("\n=== Non-negativity clamp ===")

a = UrnAgent(n_signaling_actions=2, n_final_actions=2)
a.get_signal((0,))  # populates with ones
# Apply a large negative reward to action 0:
a.update_signals((0,), 0, reward=-100.0)
clamped = a.signaling_urns[(0,)][0]
check_exact("urn[0] after large negative reward (clamp at 0)", clamped, 0.0)

# Subsequent positive reward recovers from zero:
a.update_signals((0,), 0, reward=2.0)
recovered = a.signaling_urns[(0,)][0]
check_exact("urn[0] recovers from clamp after +2 reward", recovered, 2.0)


# -----------------------------------------------------------------------------
# Section 5 — Defensive uniform reset when total urn sum hits zero
# -----------------------------------------------------------------------------
print("\n=== Defensive uniform reset on empty urn ===")

a = UrnAgent(n_signaling_actions=3, n_final_actions=3)
state = (0,)
# Manually zero out the urn (skip lazy-init by writing the dict directly).
a.signaling_urns[state] = np.zeros(3)
# Calling get_signal on a zero-sum urn should reset to ones internally.
sampled = a.get_signal(state)
post_reset_sum = a.signaling_urns[state].sum()
check_exact(
    f"post-reset urn sum after sampling from empty (returned {sampled})",
    float(post_reset_sum),
    3.0,  # K * 1.0
)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n=== Summary ===")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"All checks passed (exact atol = {ATOL_EXACT}, MC rtol = {RTOL_MC}).")
