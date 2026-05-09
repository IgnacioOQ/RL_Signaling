"""Enumerate the deterministic-policy absorbing states of the canonical signal-trading game.

For the §2.3 setup of `Signaling_Games_with_Distributed_Rewards.pdf`:
  - 2 agents
  - 2 features (X, Y), each binary; agent 0 observes X, agent 1 observes Y
  - 2 signals; 4 final actions
  - Two random canonical matching games (one per agent), with rewards in {0, 1}

Under UrnAgent dynamics with `init_weights = [1, 0]` (m = 0), every cell of
the urn is one-hot. The induced policy is deterministic and, because the
clamped urn update `urn[a] = max(0, urn[a] + r)` cannot grow a zero-weight
cell, the policy never changes. Every such state is an **absorbing state**
of the Markov chain on policy space.

Enumeration:
  - Per-agent signaling policy: bijection {0, 1} -> {0, 1}, so 2 per agent.
  - Per-agent action policy: bijection {0, 1} x {0, 1} -> {0, 1, 2, 3},
    so 4! = 24 per agent.
  - Per-agent total deterministic policies: 2 x 24 = 48.
  - Joint absorbing states: 48 x 48 = 2304.

For each joint profile, this script computes the per-agent mean reward over
the four (x, y) world states and reports:

  1. Distribution of mean reward across all 2304 absorbing states.
  2. Number of "ideal" absorbing states (mean reward = 1.0 for both agents).
  3. Number of "trap" absorbing states (mean reward = 0.0 for both agents).
  4. The breakdown by (mean_r0, mean_r1) to expose exploitation states where
     one agent does well at the other's expense.

Run:
    .venv/bin/python -m analytics.scripts.enumerate_absorbing_states
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import Counter

import numpy as np

from rl_signaling.games import create_random_canonical_game

failures: list[str] = []


def check_exact(label: str, lhs, rhs) -> None:
    status = "PASS" if lhs == rhs else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: lhs={lhs}, rhs={rhs}")


# -----------------------------------------------------------------------------
# Section 1 — Setup: generate the canonical matching games for two agents.
# -----------------------------------------------------------------------------
print("=== Section 1: setup ===")

N_FEATURES = 2
N_SIG = 2
N_FIN = 4
SEED = 0

random.seed(SEED)
np.random.seed(SEED)

games = {
    i: create_random_canonical_game(N_FEATURES, N_FIN, n=1, m=0) for i in range(2)
}
WORLD_STATES = list(itertools.product([0, 1], repeat=N_FEATURES))  # (x, y)

# Sanity: each (x, y) state has exactly one optimal action with reward 1.
for i in range(2):
    for state in WORLD_STATES:
        rewards = [games[i][state][a] for a in range(N_FIN)]
        n_optimal = sum(r == 1 for r in rewards)
        check_exact(
            f"agent {i}, state {state}: exactly one reward-1 action",
            n_optimal,
            1,
        )

# Print the optimal-action map for each agent: G_i^*(x, y).
def optimal_action(game, state):
    return max(game[state], key=game[state].get)


print("\nOptimal action maps under seed 0:")
for i in range(2):
    print(f"  Agent {i}: ", end="")
    for state in WORLD_STATES:
        print(f"  G*{state}={optimal_action(games[i], state)}", end="")
    print()


# -----------------------------------------------------------------------------
# Section 2 — Enumerate per-agent deterministic policies.
# -----------------------------------------------------------------------------
print("\n=== Section 2: per-agent deterministic policy enumeration ===")

# Each agent observes a single binary feature, so V_i = {0, 1}.
# Signaling policy is a function f: {0, 1} -> {0, 1}, but `_generate_hot_vectors`
# under `random.shuffle` produces a *bijection* — every observation maps to a
# distinct signal. There are 2! = 2 such bijections per agent.

sig_policies = [
    {0: 0, 1: 1},  # identity bijection
    {0: 1, 1: 0},  # swap bijection
]
check_exact("per-agent signaling bijections", len(sig_policies), 2)


# Action policy is a function g: {0, 1} x {0, 1} -> {0, 1, 2, 3}.
# Since the action key space (own_obs, received_signal) has 4 elements and
# the action space has 4 elements, the deterministic absorbing states correspond
# to the 4! = 24 bijections.
ACTION_KEYS = list(itertools.product([0, 1], [0, 1]))  # 4 keys
action_policies: list[dict[tuple[int, int], int]] = []
for perm in itertools.permutations(range(N_FIN)):
    policy = {key: action for key, action in zip(ACTION_KEYS, perm)}
    action_policies.append(policy)
check_exact("per-agent action bijections", len(action_policies), 24)

# Per-agent and joint counts.
n_per_agent = len(sig_policies) * len(action_policies)
n_joint = n_per_agent**2
check_exact("per-agent deterministic policies", n_per_agent, 48)
check_exact("joint absorbing states", n_joint, 2304)


# -----------------------------------------------------------------------------
# Section 3 — Compute the mean reward of every joint profile.
# -----------------------------------------------------------------------------
print("\n=== Section 3: mean reward of every joint absorbing state ===")


def mean_reward_pair(
    f0, g0, f1, g1,
    games,
) -> tuple[float, float]:
    """Mean per-agent reward over the 4 world states (x, y), for a deterministic
    joint policy profile (f0, g0, f1, g1).

    Agent 0 observes x, emits signal f0[x], receives signal f1[y] from agent 1,
    and acts via g0[(x, f1[y])]. Symmetric for agent 1.
    """
    r0_sum, r1_sum = 0.0, 0.0
    for x, y in WORLD_STATES:
        s_from_0 = f0[x]
        s_from_1 = f1[y]
        a0 = g0[(x, s_from_1)]
        a1 = g1[(y, s_from_0)]
        r0_sum += games[0][(x, y)][a0]
        r1_sum += games[1][(x, y)][a1]
    n = len(WORLD_STATES)
    return r0_sum / n, r1_sum / n


# Compute (r0, r1) for every joint profile.
all_rewards: list[tuple[float, float]] = []
for f0 in sig_policies:
    for g0 in action_policies:
        for f1 in sig_policies:
            for g1 in action_policies:
                all_rewards.append(mean_reward_pair(f0, g0, f1, g1, games))

check_exact("number of joint profiles evaluated", len(all_rewards), 2304)


# -----------------------------------------------------------------------------
# Section 4 — Distribution of (r0, r1) across absorbing states.
# -----------------------------------------------------------------------------
print("\n=== Section 4: reward distribution over absorbing states ===")

# Discrete histogram of (r0, r1).
counter = Counter(all_rewards)
# Sort by joint reward descending.
sorted_pairs = sorted(counter.items(), key=lambda kv: -(kv[0][0] + kv[0][1]))

print(f"{'(r0, r1)':<14} {'count':>6} {'fraction':>10}")
for (r0, r1), count in sorted_pairs:
    frac = count / len(all_rewards)
    print(f"({r0:.2f}, {r1:.2f})    {count:>6} {frac:>10.4f}")

# Specific quantities.
n_ideal = sum(1 for r in all_rewards if r == (1.0, 1.0))
n_one_perfect = sum(
    1 for r in all_rewards if (r[0] == 1.0) ^ (r[1] == 1.0)  # exactly one is 1
)
n_trap = sum(1 for r in all_rewards if r == (0.0, 0.0))
n_baseline = sum(1 for r in all_rewards if r[0] <= 0.25 and r[1] <= 0.25)

print(f"\n  Ideal states  (r0 = r1 = 1.0): {n_ideal}  ({n_ideal/2304:.2%})")
print(f"  Half-perfect  (exactly one r = 1.0): {n_one_perfect}  ({n_one_perfect/2304:.2%})")
print(f"  Trap states   (r0 = r1 = 0.0): {n_trap}  ({n_trap/2304:.2%})")
print(f"  At-or-below-baseline (max r <= 0.25): {n_baseline}  ({n_baseline/2304:.2%})")


# -----------------------------------------------------------------------------
# Section 5 — Per-agent reward marginals.
# -----------------------------------------------------------------------------
print("\n=== Section 5: per-agent reward marginals ===")

# How is the reward of agent 0 distributed when we marginalize over agent 1's policy?
r0_marginal = Counter(r[0] for r in all_rewards)
r1_marginal = Counter(r[1] for r in all_rewards)

print("Agent 0 reward marginal:")
for r, count in sorted(r0_marginal.items(), reverse=True):
    print(f"  r0 = {r:.2f}: {count}  ({count/2304:.4f})")

print("\nAgent 1 reward marginal:")
for r, count in sorted(r1_marginal.items(), reverse=True):
    print(f"  r1 = {r:.2f}: {count}  ({count/2304:.4f})")


# -----------------------------------------------------------------------------
# Section 6 — Connection to UrnAgent's `init_weights = [1, 0]` behavior.
# -----------------------------------------------------------------------------
print("\n=== Section 6: connection to init_weights = [1, 0] ===")

# `create_initial_signals` picks a random bijection. The fraction of joint
# profiles that are "ideal" (r0 = r1 = 1) is the empirical probability that
# random initialization on this seed lands on a perfect signaling+decoding
# system — assuming uniform over the 2304 joint absorbing states (which is
# what random.shuffle gives, modulo independence of the four shuffles).

prob_ideal = n_ideal / n_joint
prob_at_least_one_perfect = (n_ideal + n_one_perfect) / n_joint

print(f"  Under uniform initialization over 2304 absorbing states:")
print(f"    P(both agents at reward 1.0)         = {n_ideal}/{n_joint} = {prob_ideal:.4f}")
print(f"    P(at least one agent at reward 1.0)  = "
      f"{n_ideal + n_one_perfect}/{n_joint} = {prob_at_least_one_perfect:.4f}")
print(f"    P(reward 1/4 ≤ r ≤ 3/4 for both)     = "
      f"{sum(1 for r in all_rewards if 0.25 <= r[0] <= 0.75 and 0.25 <= r[1] <= 0.75)}/{n_joint}")

# Mean joint reward over absorbing states.
mean_r0 = sum(r[0] for r in all_rewards) / len(all_rewards)
mean_r1 = sum(r[1] for r in all_rewards) / len(all_rewards)
print(f"\n  Mean reward over absorbing states (uniform): r0 = {mean_r0:.4f}, r1 = {mean_r1:.4f}")


# -----------------------------------------------------------------------------
# Section 7 — Sanity check: 1/n_final_actions = 0.25 baseline.
# -----------------------------------------------------------------------------
print("\n=== Section 7: random-action baseline cross-check ===")

# With random uniform action selection, mean reward = 1 / n_final_actions.
# Under uniform sampling over absorbing states (which span all 24 action
# bijections), each action key has every action with equal frequency, so the
# mean per-state reward is also 1/n_final_actions.

# This is a sanity check: the per-agent mean reward across absorbing states
# should equal 1 / n_final_actions = 1/4 = 0.25.
expected_baseline = 1 / N_FIN
check_exact(
    f"mean r0 over absorbing states = 1/n_final_actions = {expected_baseline}",
    round(mean_r0, 6),
    expected_baseline,
)
check_exact(
    f"mean r1 over absorbing states = 1/n_final_actions = {expected_baseline}",
    round(mean_r1, 6),
    expected_baseline,
)


# -----------------------------------------------------------------------------
# Section 8 — Robustness check: same analysis under a few different game seeds.
# -----------------------------------------------------------------------------
print("\n=== Section 8: robustness across game seeds ===")

# The reward distribution shape should be invariant to the random seed used
# to generate the canonical games — only the *labelling* of which profiles
# are ideal changes. Verify by counting ideal states for several seeds.

print(f"{'seed':<6} {'n_ideal':>8} {'n_trap':>8}")
for seed in range(5):
    random.seed(seed)
    np.random.seed(seed)
    games_s = {
        i: create_random_canonical_game(N_FEATURES, N_FIN, n=1, m=0) for i in range(2)
    }
    n_ideal_s = 0
    n_trap_s = 0
    for f0 in sig_policies:
        for g0 in action_policies:
            for f1 in sig_policies:
                for g1 in action_policies:
                    r = mean_reward_pair(f0, g0, f1, g1, games_s)
                    if r == (1.0, 1.0):
                        n_ideal_s += 1
                    elif r == (0.0, 0.0):
                        n_trap_s += 1
    print(f"{seed:<6} {n_ideal_s:>8} {n_trap_s:>8}")


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n=== Summary ===")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
