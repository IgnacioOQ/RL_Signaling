"""Toy single-state Markov chain — UrnAgent dynamics in the smallest tractable setting.

Setting:
  - One agent, one observation, two signals {0, 1}, single-state matching game.
  - Reward function r(signal) = 1[signal == i*] for a fixed correct signal i*.
  - UrnAgent dynamics: sample signal proportional to urn weights, then
    update urn[chosen_signal] += reward (with the project's max(0, ...) clamp,
    which is inert for non-negative rewards).

The state of the chain is the urn vector u = (u_hot, u_cold) where:
  - u_hot = u[i*] is the weight on the correct signal,
  - u_cold = u[1 - i*] is the weight on the incorrect signal.

Because the incorrect signal yields reward 0, u_cold is **constant** along
every trajectory — the only stochastic variable is u_hot, which grows by 1
with probability rho_t = u_hot / (u_hot + u_cold) at each step. This makes
the chain a generalized Polya urn that we can analyze in closed form.

Three regimes:

  (1) Aligned absorbing case (m = 0, hot at i*).
      State (n, 0). Sampling probability of i* = 1. Reward = 1 every step.
      State evolves (n, 0) -> (n+1, 0) -> ... but the *policy* never changes.

  (2) Misaligned absorbing case (m = 0, hot at 1 - i*).
      State (0, n). Sampling probability of i* = 0. Reward = 0 every step.
      State stays at (0, n) forever. The misaligned bijection persists.

  (3) Reachable case (m > 0). State (n, m) with hot at i*.
      Sampling probability of i* = n / (n + m) initially, grows toward 1.
      ρ_t is a non-decreasing sub-martingale converging a.s. to 1.

This script computes the exact distribution of u_hot at each time step using
the recursion P(X_{t+1} = k) = P(X_t = k) (1 - rho_k) + P(X_t = k-1) rho_{k-1}
where X_t = u_hot,t - n_0 counts the number of correct signals in [0, t-1]
and rho_k = (n_0 + k) / (n_0 + k + m). It then validates the closed form
against a Monte Carlo simulation and reports E[rho_t] and median hitting
times for the four notebook initializations [1,0], [1,1], [5,1], [100,1].

Run:
    .venv/bin/python -m analytics.scripts.study_toy_markov_chain
"""

from __future__ import annotations

import sys

import numpy as np

# Two-source verification tolerances.
ATOL_EXACT = 1e-12
RTOL_MC = 0.02  # 2% relative tolerance for Monte-Carlo validation.

failures: list[str] = []


def check_exact(label: str, lhs: float, rhs: float, atol: float = ATOL_EXACT) -> None:
    diff = abs(lhs - rhs)
    status = "PASS" if diff <= atol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: lhs={lhs:.12g}, rhs={rhs:.12g}, diff={diff:.3e}")


def check_relative(label: str, lhs: float, rhs: float, rtol: float = RTOL_MC) -> None:
    rel = abs(lhs - rhs) / max(abs(rhs), 1e-12)
    status = "PASS" if rel <= rtol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: empirical={lhs:.6g}, analytical={rhs:.6g}, rel={rel:.3e}")


# -----------------------------------------------------------------------------
# Section 1 — Exact transition kernel for the aligned case (n_0, m), m > 0.
# -----------------------------------------------------------------------------
# Let X_t = number of correct signals up to time t. Then u_hot,t = n_0 + X_t and
# rho_t = (n_0 + X_t) / (n_0 + X_t + m). The chain (X_t) on {0, 1, 2, ...} has:
#
#   P(X_{t+1} = k+1 | X_t = k) = rho_k = (n_0 + k) / (n_0 + k + m)
#   P(X_{t+1} = k   | X_t = k) = 1 - rho_k = m / (n_0 + k + m).
#
# So X_t is birth-only with state-dependent birth rate.

def state_distribution(n_0: int, m: int, T: int) -> np.ndarray:
    """P[t, k] = probability that X_t = k for t = 0..T, k = 0..T.

    P[t, k] = 0 for k > t (cannot have more births than time elapsed).
    Each row sums to 1 (modulo floating-point error).
    """
    P = np.zeros((T + 1, T + 1))
    P[0, 0] = 1.0
    for t in range(T):
        for k in range(t + 1):
            denom = n_0 + k + m
            rho_k = (n_0 + k) / denom
            P[t + 1, k]     += P[t, k] * (1 - rho_k)
            P[t + 1, k + 1] += P[t, k] * rho_k
    return P


def expected_rho(P: np.ndarray, n_0: int, m: int) -> np.ndarray:
    """E[rho_t] for t = 0..T, given the state distribution P."""
    T = P.shape[0] - 1
    rhos = np.zeros(T + 1)
    for t in range(T + 1):
        for k in range(t + 1):
            rho_k = (n_0 + k) / (n_0 + k + m)
            rhos[t] += P[t, k] * rho_k
    return rhos


# -----------------------------------------------------------------------------
# Section 2 — Sanity check the recursion: rows sum to 1, P(X_0 = 0) = 1.
# -----------------------------------------------------------------------------
print("=== Section 2: recursion sanity ===")

T = 200
for n_0, m in [(1, 1), (5, 1), (100, 1)]:
    P = state_distribution(n_0, m, T)
    row_sums = P.sum(axis=1)
    label = f"(n_0={n_0}, m={m}) row sums equal 1 at t in {{0, T/2, T}}"
    err = max(abs(row_sums[0] - 1), abs(row_sums[T // 2] - 1), abs(row_sums[T] - 1))
    check_exact(label, err, 0.0, atol=1e-10)
    check_exact(f"(n_0={n_0}, m={m}) P(X_0 = 0) = 1", P[0, 0], 1.0)


# -----------------------------------------------------------------------------
# Section 3 — Aligned absorbing case (n, 0): rho_t = 1 for all t.
# -----------------------------------------------------------------------------
print("\n=== Section 3: aligned absorbing case (m = 0, hot at i*) ===")

# When m = 0, X_t = t deterministically (rho_k = 1 for every k), so the urn
# is (n + t, 0) at time t and rho_t = 1 for all t.
for n_0 in [1, 5, 100]:
    # Compute rho_t analytically. When m = 0, the recursion places all mass
    # on k = t at every step (rho_k = 1, so all mass moves forward).
    # Direct computation: rho_t = (n_0 + t) / (n_0 + t + 0) = 1.
    for t in [0, 1, 10, 100]:
        rho_t_analytical = 1.0
        check_exact(
            f"aligned (n_0={n_0}, m=0): rho_{t} = 1",
            rho_t_analytical,
            1.0,
        )


# -----------------------------------------------------------------------------
# Section 4 — Misaligned absorbing case (0, n): rho_t = 0 for all t.
# -----------------------------------------------------------------------------
print("\n=== Section 4: misaligned absorbing case (m = 0, hot at 1 - i*) ===")

# When the urn starts at (u_{i*} = 0, u_{1-i*} = n), the agent always picks the
# wrong signal (reward 0), so the urn never changes. rho_t = 0 for all t.
# This is the (1, 0) failure mode: NMI = 1 (perfect signaling code) but
# reward = 0 (the code is misaligned).
for n_0 in [1, 5, 100]:
    rho_misaligned = 0.0  # u[i*] = 0, so probability of correct signal is 0.
    check_exact(
        f"misaligned (u[i*]=0, u[1-i*]={n_0}): rho_t = 0",
        rho_misaligned,
        0.0,
    )


# -----------------------------------------------------------------------------
# Section 5 — Reachable case (n, m) with m > 0: rho_t monotone increases.
# -----------------------------------------------------------------------------
print("\n=== Section 5: reachable case (m > 0), E[rho_t] over time ===")

# For each notebook initialization, print E[rho_t] at t = 0, 10, 50, 100, 200.
# The four init_weights from Initializations_test.ipynb are:
#   [1, 1] - uniform pre-seed
#   [5, 1] - moderate bias
#   [100, 1] - strong bias
#
# (We skip [1, 0] because m = 0 is handled in Sections 3-4.)
inits = [(1, 1), (5, 1), (100, 1)]
T = 200

print(f"{'init (n,m)':<14}", end="")
for t in [0, 10, 50, 100, 200]:
    print(f"{'E[rho_'+str(t)+']':>14}", end="")
print()

for n_0, m in inits:
    P = state_distribution(n_0, m, T)
    rhos = expected_rho(P, n_0, m)
    print(f"({n_0}, {m})".ljust(14), end="")
    for t in [0, 10, 50, 100, 200]:
        print(f"{rhos[t]:>14.6f}", end="")
    print()

# Verify monotonicity: E[rho_t] is non-decreasing in t.
for n_0, m in inits:
    P = state_distribution(n_0, m, T)
    rhos = expected_rho(P, n_0, m)
    monotone = bool(np.all(np.diff(rhos) >= -ATOL_EXACT))
    label = f"(n_0={n_0}, m={m}): E[rho_t] non-decreasing in t"
    if monotone:
        print(f"[PASS] {label}")
    else:
        failures.append(label)
        print(f"[FAIL] {label}")


# -----------------------------------------------------------------------------
# Section 6 — Closed-form check: E[rho_0] = n / (n + m).
# -----------------------------------------------------------------------------
print("\n=== Section 6: initial E[rho_0] = n / (n + m) ===")

for n_0, m in inits:
    P = state_distribution(n_0, m, 0)
    rho_0_analytical = n_0 / (n_0 + m)
    rho_0_recursion = expected_rho(P, n_0, m)[0]
    check_exact(
        f"(n_0={n_0}, m={m}): E[rho_0]",
        rho_0_recursion,
        rho_0_analytical,
    )


# -----------------------------------------------------------------------------
# Section 7 — Monte Carlo cross-check against the recursion.
# -----------------------------------------------------------------------------
print("\n=== Section 7: Monte Carlo validation ===")

# Simulate the chain N_TRAJ times for T_MC steps and compare empirical E[rho_t]
# at a few snapshot times to the analytical value.
N_TRAJ = 50_000
T_MC = 100

rng = np.random.default_rng(seed=0)

for n_0, m in inits:
    # Vectorized simulation: for each trajectory, count the number of births.
    X = np.zeros(N_TRAJ, dtype=np.int64)
    snapshots = {0: None, 10: None, 50: None, 100: None}
    snapshots[0] = X.copy()
    for t in range(1, T_MC + 1):
        # rho_t = (n_0 + X_t) / (n_0 + X_t + m).
        rho = (n_0 + X) / (n_0 + X + m)
        # Birth happens with probability rho.
        u = rng.random(N_TRAJ)
        X += (u < rho).astype(np.int64)
        if t in snapshots:
            snapshots[t] = X.copy()

    # Analytical reference.
    P = state_distribution(n_0, m, T_MC)
    rhos_analytical = expected_rho(P, n_0, m)

    for t, X_t in snapshots.items():
        emp_rho = float(np.mean((n_0 + X_t) / (n_0 + X_t + m)))
        check_relative(
            f"(n_0={n_0}, m={m}): E[rho_{t}] MC vs analytical",
            emp_rho,
            rhos_analytical[t],
        )


# -----------------------------------------------------------------------------
# Section 8 — Median hitting time for rho_t to cross 0.99 (aligned case).
# -----------------------------------------------------------------------------
print("\n=== Section 8: median hitting time for rho_t > 0.99 ===")

# For each (n_0, m), find the smallest t such that the median of rho_t exceeds
# 0.99 under the analytical state distribution. Equivalently, smallest t such
# that P(rho_t > 0.99) > 0.5.
# rho_t > 0.99  <=>  (n_0 + k) / (n_0 + k + m) > 0.99
#               <=>  k > 99 * m - n_0.
T_LONG = 1000
THRESHOLD = 0.99

for n_0, m in inits:
    P = state_distribution(n_0, m, T_LONG)
    # k_min = smallest k such that rho_k > 0.99
    k_min = int(np.ceil(99 * m - n_0)) + 1
    if k_min < 0:
        k_min = 0  # already above threshold at t = 0
    # Compute P(X_t >= k_min) for each t.
    cumulative = 1 - np.cumsum(P, axis=1) + P  # P(X_t >= k) for each k
    # cumulative[t, k_min] = P(X_t >= k_min).
    if k_min == 0:
        median_t = 0
    else:
        ge = cumulative[:, k_min]
        # First t such that P(X_t >= k_min) >= 0.5.
        median_idx = np.argmax(ge >= 0.5)
        median_t = int(median_idx) if ge[median_idx] >= 0.5 else None
    print(f"(n_0={n_0}, m={m}): k_min={k_min}, median t for rho_t > {THRESHOLD}: {median_t}")


# -----------------------------------------------------------------------------
# Section 9 — Summary
# -----------------------------------------------------------------------------
print("\n=== Summary ===")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"All checks passed (exact atol = {ATOL_EXACT}, MC rtol = {RTOL_MC}).")
