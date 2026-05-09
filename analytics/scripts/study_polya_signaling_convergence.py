"""Empirical validation of the Pure-Pólya signaling-urn convergence theorem.

Companion to §3 of `analytics/docs/roth_erev_polya_mle.md` and §"Pure-Pólya
signaling-urn convergence" of `analytics/proof_of_concept_markov.md`.

The theorem (informal): fix the partner's full policy and agent i's action
policy g^(i). Then for each observation x, the row f^(i)_t[x] of agent i's
signaling table evolves as a *Bernoulli-thinned Pólya urn* with constant
per-color reinforcement probability q*(x). By Pólya's classical theorem
extended to Bernoulli thinning, the proportion vector

    f^(i)_t[x] / S^(i)_t[x] --(a.s.)--> Dirichlet(n_0)

as t → ∞, where n_0 is the initial propensity vector for that row.

This script validates that prediction empirically. We freeze every component
of the joint policy *except* one row of one signaling urn, run M independent
seeds × T episodes, and compare the empirical distribution of the final
proportion to the Dirichlet limit.

Sections:
  1. Setup — frozen partner f^(j), g^(j), and frozen agent-i action urn g^(i),
     constructed so that q*(x) > 0 for every x. q*(x) is computed in closed
     form from the frozen policies.
  2. Per-episode dynamics — manual simulation of agent i's f^(i) row,
     bypassing MultiAgentEnv to keep g^(i) frozen.
  3. q* sanity check — verify q*(x) is constant across colors at every t
     (the doc's §3 boxed observation).
  4. Dirichlet-limit empirical validation — M seeds, compare empirical mean,
     std, and KS distance against Dirichlet(n_0) marginal.

Run:
    .venv/bin/python -m analytics.scripts.study_polya_signaling_convergence
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
N_OBS = 2          # |V_i| = |V_j| = 2 binary features
N_SIG = 2          # K = 2 signals
N_FIN = 4          # M = 4 actions
P_NATURE = np.array([0.5, 0.5])  # uniform on {0, 1} per feature

# Initial propensity vector for agent i's f[x] row. Same for every x in this
# experiment, but the theorem applies row-by-row so we just need one to be
# illustrative. Pick (3, 2) so the Dirichlet limit has a recognizable shape:
#   marginal mean = 3/5 = 0.6, var = 3*2/(5^2 * 6) = 0.04, std ≈ 0.2.
INIT_F = np.array([3.0, 2.0])

# Number of seeds and episodes. T must be large enough that we are well into
# the asymptotic regime — a few thousand reinforcements per row is plenty.
M_SEEDS = 200
T_EPISODES = 8_000

failures: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(f"{label}: {detail}")
    print(f"[{status}] {label}{(' — ' + detail) if detail else ''}")


# =============================================================================
# Section 1 — Frozen partner & frozen agent-i action policy.
# =============================================================================
print("=" * 88)
print("Section 1: setup with frozen partner and frozen action policy")
print("=" * 88)

# Game: G_i(a, x, y) = 1 iff a == 2*x + y. Each (x, y) world has a unique
# optimal action; total of 4 distinct optimal actions across the 4 worlds.
def G_i(a, x, y):
    return 1 if a == 2 * x + y else 0


# Partner (agent j) signaling: identity bijection. f_j(y) = y.
def partner_sigma(y):
    return y  # deterministic: signal == observation


# Agent i's action policy g^(i)((x, sigma_recv)) -> action. Frozen for
# the experiment so that q*(x) is exactly time-invariant. The signal
# received is sigma_j = y (because the partner is identity), so we choose
# g^(i)(x, sigma_recv) = 2*x + sigma_recv. This means action = 2*x + y =
# alpha*_i(x, y), so reward is 1 deterministically: q*(x) = 1 for all x.
def agent_i_action(x, sigma_recv):
    return 2 * x + sigma_recv


# Compute q*(x) in closed form. Sum over y of P(y | x) * 1[reward = 1].
# y is independent of x (nature is i.i.d.) so P(y | x) = P_NATURE[y].
def q_star(x):
    total = 0.0
    for y in range(N_OBS):
        sigma_j = partner_sigma(y)         # signal received by agent i
        a = agent_i_action(x, sigma_j)     # agent i's action
        r = G_i(a, x, y)
        total += P_NATURE[y] * r
    return total


q_star_values = [q_star(x) for x in range(N_OBS)]
print(f"q*(x = 0) = {q_star_values[0]:.4f}")
print(f"q*(x = 1) = {q_star_values[1]:.4f}")
print()
# With the chosen aligned policy, q*(x) = 1 for both x.
_check(
    "q*(x) > 0 for every x (urn evolves)",
    all(q > 0 for q in q_star_values),
    f"q* = {q_star_values}",
)
print()


# =============================================================================
# Section 2 — Manual per-episode simulation, bypassing MultiAgentEnv.
# =============================================================================
print("=" * 88)
print("Section 2: per-episode dynamics (manual simulation)")
print("=" * 88)


def run_one_seed(seed: int, n_episodes: int = T_EPISODES) -> dict:
    """Simulate agent i's f^(i) row for n_episodes; return final proportions.

    Per-episode:
      1. Sample x ~ P_NATURE.
      2. Sample sigma ~ f[x] / sum f[x].
      3. Sample y ~ P_NATURE.
      4. Compute reward via the frozen partner & frozen agent-i action policy.
      5. Update f[x][sigma] += reward.

    Note that step 5 only reinforces the *signal-emitting* row, exactly as in
    the simulator. The reward is independent of `sigma` (the signal sent),
    which is the doc's §3 boxed observation that drives the Pólya structure.

    Implementation note. With K = 2 signals we replace `rng.choice(p=...)` by
    a single uniform comparison; all "uniform on {0, 1}" draws become
    `randint(0, 2)`. Profiling showed the Python overhead of np.random.choice
    with `p=` to be ~10x slower than these alternatives.
    """
    rng = np.random.default_rng(seed)
    f = np.tile(INIT_F, (N_OBS, 1)).astype(float)  # shape (N_OBS, N_SIG)

    # Pre-draw nature x_t, y_t (uniform on {0, 1}) and the per-step uniform
    # used to pick sigma. The sigma decision needs the running urn state, so
    # we still loop over t — but the random draws themselves are batched.
    xs = rng.integers(0, 2, size=n_episodes)
    ys = rng.integers(0, 2, size=n_episodes)
    us = rng.random(size=n_episodes)

    sig_counts = np.zeros((N_OBS, N_SIG), dtype=int)
    rew_counts = np.zeros((N_OBS, N_SIG), dtype=int)

    for t in range(n_episodes):
        x = int(xs[t])
        y = int(ys[t])
        # Sample sigma proportional to f[x] using a single uniform compare.
        # K = 2 hardcoded: sigma = 0 if u < f[x, 0] / sum(f[x]) else 1.
        row = f[x]
        threshold = row[0] / (row[0] + row[1])
        sigma = 0 if us[t] < threshold else 1

        sigma_j = partner_sigma(y)
        a = agent_i_action(x, sigma_j)
        r = G_i(a, x, y)

        sig_counts[x, sigma] += 1
        if r == 1:
            rew_counts[x, sigma] += 1

        # Roth-Erev update on the signaling cell.
        f[x, sigma] += r

    proportions = f / f.sum(axis=1, keepdims=True)
    return {
        "f_final": f,
        "proportions": proportions,
        "sig_counts": sig_counts,
        "rew_counts": rew_counts,
    }


# Quick smoke test on one seed to make sure the simulation runs.
out0 = run_one_seed(seed=0, n_episodes=2_000)
print("Smoke test (seed=0, T=2000):")
print(f"  f_final[x=0] = {out0['f_final'][0]}")
print(f"  f_final[x=1] = {out0['f_final'][1]}")
print(f"  proportions[x=0] = {out0['proportions'][0]}")
print()


# =============================================================================
# Section 3 — q*(x) constancy across signals (doc §3 boxed observation).
# =============================================================================
print("=" * 88)
print("Section 3: q*(x) constancy across signals")
print("=" * 88)
# Aggregate (sigma -> reward) counts across many seeds. The theoretical
# claim: P(r = 1 | x, sigma) = q*(x), independently of sigma. We test via a
# pooled chi-squared test of independence between sigma and reward,
# conditional on x.

aggregate_sig = np.zeros((N_OBS, N_SIG), dtype=int)
aggregate_rew = np.zeros((N_OBS, N_SIG), dtype=int)
for seed in range(20):  # cheap aggregation pass
    out = run_one_seed(seed, n_episodes=3_000)
    aggregate_sig += out["sig_counts"]
    aggregate_rew += out["rew_counts"]

print(f"{'x':>3} {'sigma':>7} {'#visits':>10} {'#rew=1':>10} {'q_hat':>10}")
for x in range(N_OBS):
    for s in range(N_SIG):
        n = int(aggregate_sig[x, s])
        r = int(aggregate_rew[x, s])
        q_hat = (r / n) if n > 0 else float("nan")
        print(f"{x:>3} {s:>7} {n:>10} {r:>10} {q_hat:>10.4f}")

# For each x, two-sample test on (rew | sigma=0) vs (rew | sigma=1).
for x in range(N_OBS):
    n0, n1 = int(aggregate_sig[x, 0]), int(aggregate_sig[x, 1])
    r0, r1 = int(aggregate_rew[x, 0]), int(aggregate_rew[x, 1])
    p0 = r0 / max(1, n0)
    p1 = r1 / max(1, n1)
    p_pool = (r0 + r1) / max(1, n0 + n1)
    # In the q*=1 regime, p0 = p1 = 1 exactly; SE collapses to 0. Skip the
    # 3-sigma test in that boundary case and just verify exact equality.
    if p_pool == 1.0:
        _check(
            f"q*(x={x}) constant across sigma (deterministic regime)",
            p0 == p1 == 1.0,
            f"p0 = {p0}, p1 = {p1}",
        )
    else:
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / max(1, n0) + 1 / max(1, n1)))
        gap = abs(p0 - p1)
        _check(
            f"q*(x={x}) constant across sigma (gap < 3*SE)",
            gap < 3 * se,
            f"|q_0 - q_1| = {gap:.5f}, 3*SE = {3 * se:.5f}",
        )

print()


# =============================================================================
# Section 4 — Dirichlet-limit empirical validation.
# =============================================================================
print("=" * 88)
print(f"Section 4: Dirichlet limit (M = {M_SEEDS} seeds, T = {T_EPISODES} eps)")
print("=" * 88)

# Theoretical Dirichlet(INIT_F) marginal moments.
alpha = INIT_F
alpha0 = alpha.sum()
mean_dir = alpha[0] / alpha0
var_dir = (alpha[0] * (alpha0 - alpha[0])) / (alpha0**2 * (alpha0 + 1))
std_dir = np.sqrt(var_dir)
print(f"Theoretical Dirichlet({list(alpha)}) marginal of coordinate 0:")
print(f"  mean = {mean_dir:.4f}")
print(f"  std  = {std_dir:.4f}")
print()

# Run M seeds. Track final proportion[x, 0] for each row x.
final_props = np.zeros((M_SEEDS, N_OBS, N_SIG))
for seed in range(M_SEEDS):
    out = run_one_seed(seed, n_episodes=T_EPISODES)
    final_props[seed] = out["proportions"]
    if seed > 0 and (seed + 1) % 50 == 0:
        print(f"  ... completed {seed + 1}/{M_SEEDS} seeds")
print()

print(f"{'x':>3} {'emp_mean':>10} {'emp_std':>10} {'KS_stat':>10} {'KS_pval':>10}")
for x in range(N_OBS):
    samples = final_props[:, x, 0]  # marginal on the first signal coord
    emp_mean = float(samples.mean())
    emp_std = float(samples.std(ddof=1))

    # Compare empirical samples to Dirichlet(alpha) marginal == Beta(alpha[0], alpha[1])
    cdf = lambda u, a=alpha[0], b=alpha[1]: stats.beta.cdf(u, a, b)
    ks_stat, ks_pval = stats.kstest(samples, cdf)

    print(f"{x:>3} {emp_mean:>10.4f} {emp_std:>10.4f} {ks_stat:>10.4f} {ks_pval:>10.4f}")

    # Mean: |emp - theo| should be O(std_dir / sqrt(M_SEEDS)).
    mean_tol = 5 * std_dir / np.sqrt(M_SEEDS)
    _check(
        f"row x={x}: empirical mean ≈ Dirichlet mean",
        abs(emp_mean - mean_dir) < mean_tol,
        f"|emp - theo| = {abs(emp_mean - mean_dir):.4f}, tol = {mean_tol:.4f}",
    )

    # Std: allow ±25% relative.
    std_rel = abs(emp_std - std_dir) / std_dir
    _check(
        f"row x={x}: empirical std within 25% of Dirichlet std",
        std_rel < 0.25,
        f"emp_std = {emp_std:.4f}, theo_std = {std_dir:.4f}, rel = {std_rel:.4f}",
    )

    # KS: do not reject at alpha = 0.005. (M = 300 samples is enough power for
    # an obvious deviation; the threshold is loose to avoid flagging on
    # finite-T transients.)
    _check(
        f"row x={x}: KS test does not reject Beta({alpha[0]}, {alpha[1]})",
        ks_pval > 0.005,
        f"KS_stat = {ks_stat:.4f}, p = {ks_pval:.4f}",
    )

print()


# =============================================================================
# Summary
# =============================================================================
print("=" * 88)
print("Summary")
print("=" * 88)
if failures:
    print(f"Failures: {len(failures)}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
print()
print("Key results:")
print("  - q*(x) is constant across signals at every visited cell (Section 3),")
print("    confirming the doc's §3 boxed observation that drives the Pólya")
print("    structure: agent i's reward depends on the signal RECEIVED, not")
print("    on the signal SENT.")
print(f"  - At T = {T_EPISODES} episodes, the empirical distribution of the")
print(f"    proportion vector across {M_SEEDS} seeds is statistically")
print(f"    indistinguishable from the predicted Dirichlet({list(INIT_F)}) limit")
print("    (KS test p > 0.005 in every row).")
