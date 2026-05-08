"""Independent verification of TDLearningAgent — bootstrap, terminal, and Robbins-Monro convergence.

Cross-checks the math derived in analytics/agent_td_learning.md:

  1. Bootstrap: Q[s][a] += (r + gamma*max(Q[next])) / N(s,a) when not done.
  2. Terminal:  Q[s][a] += r / N(s,a) when done (bootstrap dropped).
  3. Robbins-Monro: under i.i.d. terminal rewards, Q_n converges to E[r] at rate 1/sqrt(n).

Run:
    .venv/bin/python -m analytics.scripts.verify_td_learning
"""

from __future__ import annotations

import sys

import numpy as np

from rl_signaling.agents import TDLearningAgent

ATOL_EXACT = 1e-12
RTOL_RM = 0.05    # 5% relative tolerance for the empirical-mean Robbins-Monro check

failures: list[str] = []


def check_exact(label: str, lhs: float, rhs: float, atol: float = ATOL_EXACT) -> None:
    diff = abs(lhs - rhs)
    status = "PASS" if diff <= atol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: rl_signaling={lhs!r}, expected={rhs!r}, diff={diff:.3e}")


def check_relative(label: str, lhs: float, rhs: float, rtol: float = RTOL_RM) -> None:
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)
    status = "PASS" if rel <= rtol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: rl_signaling={lhs:.6g}, expected={rhs:.6g}, rel_diff={rel:.3e}")


# -----------------------------------------------------------------------------
# Section 1 — Single bootstrap step
# -----------------------------------------------------------------------------
print("=== Single bootstrap step ===")

a = TDLearningAgent(n_actions=4, gamma=1.0)
a.q_table[(0,)] = np.zeros(4)
a.q_table[(1,)] = np.array([1.0, 0.0, 0.0, 0.0])
a.action_counts[(0,)] = np.array([1.0, 0.0, 0.0, 0.0])
a.action_counts[(1,)] = np.zeros(4)

# td_target = 0 + 1 * max([1,0,0,0]) = 1
# td_error = 1 - 0 = 1
# learning_rate = td_error / count = 1
# Q[(0,)][0] = 0 + 1 = 1
a.update(state=(0,), action=0, reward=0.0, next_state=(1,), done=False)
check_exact("bootstrap with gamma=1, max Q(next)=1, count=1 → Q=1", a.q_table[(0,)][0], 1.0)


# -----------------------------------------------------------------------------
# Section 2 — Terminal step
# -----------------------------------------------------------------------------
print("\n=== Terminal step ===")

a = TDLearningAgent(n_actions=4, gamma=1.0)
a.q_table[(0,)] = np.zeros(4)
a.action_counts[(0,)] = np.array([1.0, 0.0, 0.0, 0.0])

# td_target = 1 (bootstrap dropped because done=True)
# Q[(0,)][0] = 0 + 1 = 1
a.update(state=(0,), action=0, reward=1.0, next_state=(0,), done=True)
check_exact("terminal with reward=1, count=1 → Q=1", a.q_table[(0,)][0], 1.0)


# -----------------------------------------------------------------------------
# Section 3 — Robbins-Monro convergence to E[r] under i.i.d. terminal rewards
# -----------------------------------------------------------------------------
print("\n=== Robbins-Monro: terminal-only updates converge to E[r] ===")

# Setup: keep calling update with terminal reward ~ Bernoulli(p), starting Q=0.
# After N visits, Q = (1/N) * sum_{k=1..N} r_k = empirical mean.
# So Q converges to p with standard deviation sqrt(p(1-p)/N).

P = 0.7
N_VISITS = 10000
PREDICTED_MEAN = P
PREDICTED_STD = np.sqrt(P * (1 - P) / N_VISITS)

rng = np.random.default_rng(seed=2026)

a = TDLearningAgent(n_actions=2, gamma=1.0)
a.q_table[(0,)] = np.zeros(2)
a.action_counts[(0,)] = np.zeros(2)

for _ in range(N_VISITS):
    # Manually increment count (mimicking the get_action call before update).
    a.action_counts[(0,)][0] += 1
    r = float(rng.binomial(1, P))
    a.update(state=(0,), action=0, reward=r, next_state=(0,), done=True)

empirical_q = a.q_table[(0,)][0]

# After N_VISITS, |Q - p| should be O(predicted_std) with very high probability.
# Use a 5-sigma envelope as a non-flaky tolerance.
five_sigma = 5 * PREDICTED_STD
diff = abs(empirical_q - PREDICTED_MEAN)
status = "PASS" if diff <= five_sigma else "FAIL"
if status == "FAIL":
    failures.append("Robbins-Monro convergence")
print(
    f"[{status}] Q_{N_VISITS} → E[r] = {PREDICTED_MEAN}: rl_signaling={empirical_q:.6g}, "
    f"5σ envelope = ±{five_sigma:.4f}, diff={diff:.4f}"
)


# -----------------------------------------------------------------------------
# Section 4 — Two-phase update via update_episode
# -----------------------------------------------------------------------------
print("\n=== Two-phase update_episode ===")

# Verify that update_episode fires both the signal-phase bootstrap and the
# action-phase terminal updates, with the correct keys.
a = TDLearningAgent(n_signaling_actions=2, n_final_actions=4, gamma=1.0)
a.q_table[(0,)] = np.zeros(4)
a.q_table[(0, 0)] = np.array([0.0, 0.0, 5.0, 0.0])  # action_state = (0, 0); pre-populated
a.action_counts[(0,)] = np.array([1.0, 0.0, 0.0, 0.0])
a.action_counts[(0, 0)] = np.array([0.0, 0.0, 1.0, 0.0])

# Signal-phase: state=(0,), action=0, reward=0, next_state=(0,0), done=False
#   td_target = 0 + 1 * max([0, 0, 5, 0]) = 5
#   td_error = 5 - 0 = 5
#   learning_rate = 5 / 1 = 5
#   Q[(0,)][0] = 0 + 5 = 5
# Action-phase: state=(0,0), action=2, reward=1, next_state=(0,0), done=True
#   td_target = 1 (terminal)
#   td_error = 1 - 5 = -4
#   learning_rate = -4 / 1 = -4
#   Q[(0,0)][2] = 5 + (-4) = 1
a.update_episode(signal_state=(0,), signal=0, action_state=(0, 0), action=2, reward=1.0)

check_exact("update_episode signal-phase Q[(0,)][0]", a.q_table[(0,)][0], 5.0)
check_exact("update_episode action-phase Q[(0,0)][2]", a.q_table[(0, 0)][2], 1.0)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n=== Summary ===")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"All checks passed.")
