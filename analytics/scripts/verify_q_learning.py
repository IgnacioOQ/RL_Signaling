"""Independent verification of QLearningAgent — closed-form Q_n vs implementation, plus asymptotic variance.

Cross-checks the math derived in analytics/agent_q_learning.md:

  1. Q_n = r * (1 - (1-alpha)^n) for constant reward r, starting from Q_0 = 0.
  2. Var(Q_inf) = alpha / (2-alpha) * sigma^2 for i.i.d. rewards with variance sigma^2.

Run:
    .venv/bin/python -m analytics.scripts.verify_q_learning
"""

from __future__ import annotations

import sys

import numpy as np

from rl_signaling.agents import QLearningAgent

ATOL_EXACT = 1e-12     # for closed-form per-step values
RTOL_VAR = 0.10        # 10% relative tolerance for asymptotic variance (Monte Carlo)

failures: list[str] = []


def check_exact(label: str, lhs: float, rhs: float, atol: float = ATOL_EXACT) -> None:
    """Exact-arithmetic check; fails if abs(lhs - rhs) > atol."""
    diff = abs(lhs - rhs)
    status = "PASS" if diff <= atol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: rl_signaling={lhs!r}, closed_form={rhs!r}, diff={diff:.3e}")


def check_relative(label: str, lhs: float, rhs: float, rtol: float = RTOL_VAR) -> None:
    """Relative-tolerance check, used for asymptotic / Monte-Carlo cases."""
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)
    status = "PASS" if rel <= rtol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: rl_signaling={lhs:.6g}, closed_form={rhs:.6g}, rel_diff={rel:.3e}")


# -----------------------------------------------------------------------------
# Section 1 — Q_n closed form, constant reward, alpha = 0.1
# -----------------------------------------------------------------------------
print("=== Q_n closed form for constant reward ===")

ALPHA = 0.1
REWARD = 1.0
N_STEPS = 100

agent = QLearningAgent(n_signaling_actions=2, n_final_actions=2)
state = (0,)
agent.get_signal(state)  # populate row with zeros

assert agent.q_table_signaling[state][0] == 0.0, "Q should start at zero"

for n in range(1, N_STEPS + 1):
    agent.update_signals(state, 0, reward=REWARD)
    expected = REWARD * (1.0 - (1.0 - ALPHA) ** n)
    actual = agent.q_table_signaling[state][0]
    if n in {1, 2, 5, 10, 20, 50, 100}:
        check_exact(f"Q_{n} = {expected:.10f}", actual, expected)
    else:
        # Silently assert for the in-between steps; only print landmark steps.
        if abs(actual - expected) > ATOL_EXACT:
            failures.append(f"Q_{n} silent mismatch")


# -----------------------------------------------------------------------------
# Section 2 — Asymptotic variance under i.i.d. Bernoulli rewards
# -----------------------------------------------------------------------------
print("\n=== Asymptotic variance Var(Q_inf) = alpha/(2-alpha) * sigma^2 ===")

# Reward distribution: Bernoulli(p=0.5), so mean = 0.5, variance = 0.25.
# Predicted asymptotic variance = 0.1 / 1.9 * 0.25 = 0.01316...
P = 0.5
SIGMA2 = P * (1 - P)
PREDICTED_VAR = ALPHA / (2 - ALPHA) * SIGMA2
PREDICTED_MEAN = P

print(f"Reward distribution: Bernoulli({P})")
print(f"  predicted asymptotic mean: {PREDICTED_MEAN}")
print(f"  predicted asymptotic Var(Q_inf): {PREDICTED_VAR:.6f}")

N_AGENTS_MC = 500
N_BURN_IN = 5000
N_SAMPLE = 5000

rng = np.random.default_rng(seed=12345)

agents = [QLearningAgent(n_signaling_actions=2, n_final_actions=2) for _ in range(N_AGENTS_MC)]
for a in agents:
    a.get_signal((0,))  # populate row

# Burn-in (let Q reach the asymptote)
for _ in range(N_BURN_IN):
    rewards = rng.binomial(1, P, size=N_AGENTS_MC).astype(float)
    for a, r in zip(agents, rewards):
        a.update_signals((0,), 0, reward=r)

# Sample Q at episode N_BURN_IN, N_BURN_IN+1, ..., N_BURN_IN+N_SAMPLE-1
samples = np.empty((N_AGENTS_MC, N_SAMPLE))
for t in range(N_SAMPLE):
    rewards = rng.binomial(1, P, size=N_AGENTS_MC).astype(float)
    for i, (a, r) in enumerate(zip(agents, rewards)):
        a.update_signals((0,), 0, reward=r)
        samples[i, t] = a.q_table_signaling[(0,)][0]

# Average across agents AND time samples (ergodicity):
#   E[Q_inf] = mean over (agent, t)
#   Var(Q_inf) = variance over (agent, t)
empirical_mean = samples.mean()
empirical_var = samples.var()

check_relative("E[Q_inf] = 0.5", empirical_mean, PREDICTED_MEAN)
check_relative("Var(Q_inf) = 0.01316", empirical_var, PREDICTED_VAR)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n=== Summary ===")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"All checks passed (exact atol = {ATOL_EXACT}, variance rtol = {RTOL_VAR}).")
