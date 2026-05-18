"""Independent verification of MultiAgentEnv costly-signaling reward arithmetic.

Cross-checks the math derived in analytics/math/costly_signaling.md:

    r_i = G_i(v, alpha_i) - c_i * 1[sigma_i != null]

Tests every combination of {non-null, null} signals across two agents on a
flat-reward game (so the only variable is the cost flow).

Run:
    .venv/bin/python -m analytics.scripts.verify_costly_signaling
"""

from __future__ import annotations

import sys

import networkx as nx
import numpy as np

from rl_signaling.agents import UrnAgent
from rl_signaling.env import MultiAgentEnv

ATOL = 1e-12

failures: list[str] = []


def check_array(label: str, lhs, rhs, atol: float = ATOL) -> None:
    """Compare two same-length sequences and record a PASS/FAIL line."""
    arr_lhs = np.array(lhs, dtype=float)
    arr_rhs = np.array(rhs, dtype=float)
    if arr_lhs.shape != arr_rhs.shape:
        failures.append(label)
        print(f"[FAIL] {label}: shape mismatch {arr_lhs.shape} vs {arr_rhs.shape}")
        return
    diff = np.abs(arr_lhs - arr_rhs).max()
    status = "PASS" if diff <= atol else "FAIL"
    if status == "FAIL":
        failures.append(label)
    print(f"[{status}] {label}: rl_signaling={list(arr_lhs)}, expected={list(arr_rhs)}, max_diff={diff:.3e}")


# -----------------------------------------------------------------------------
# Build a minimal env with flat per-state per-action reward = GAME_REWARD.
# -----------------------------------------------------------------------------
GAME_REWARD = 1.0
COST = 0.25

graph = nx.DiGraph()
graph.add_nodes_from([0, 1])
graph.add_edges_from([(0, 1), (1, 0)])

# Flat game dict: every (state, action) gives GAME_REWARD; we only vary signals.
# n_signaling_actions=2 base, costly augments to 3 with null = index 2.
flat_dict = {(0, 0): {0: GAME_REWARD, 1: GAME_REWARD, 2: GAME_REWARD, 3: GAME_REWARD}}
game_dicts = {0: dict(flat_dict), 1: dict(flat_dict)}

env = MultiAgentEnv(
    n_agents=2,
    n_features=2,
    n_signaling_actions=2,
    n_final_actions=4,
    full_information=True,
    game_dicts=game_dicts,
    observed_variables={0: [0], 1: [1]},
    agent_type=UrnAgent,
    costly_signaling=True,
    graph=graph,
)
env.nature_vector = np.array([0, 0])  # set without calling reset()

null_idx = env._null_signal_index
print(f"Setup: GAME_REWARD={GAME_REWARD}, COST={COST}, null_signal_index={null_idx}")
print(f"  effective n_signaling_actions = {env.n_signaling_actions} (base 2 + 1 for null)")
assert null_idx == 2, "null signal must be at index 2 when base alphabet is 2"


# -----------------------------------------------------------------------------
# Case 1 — both signals non-null → both pay cost
# -----------------------------------------------------------------------------
print("\n=== Case 1: both non-null ===")

rewards = env.reward(actions=[0, 0], signals=[0, 0], signal_cost=[COST, COST])
expected = [GAME_REWARD - COST, GAME_REWARD - COST]
check_array("both non-null → both pay cost", rewards, expected)


# -----------------------------------------------------------------------------
# Case 2 — both signals null → no cost
# -----------------------------------------------------------------------------
print("\n=== Case 2: both null ===")

rewards = env.reward(actions=[0, 0], signals=[null_idx, null_idx], signal_cost=[COST, COST])
expected = [GAME_REWARD, GAME_REWARD]
check_array("both null → no cost", rewards, expected)


# -----------------------------------------------------------------------------
# Case 3 — agent 0 non-null, agent 1 null → mixed
# -----------------------------------------------------------------------------
print("\n=== Case 3: agent 0 non-null, agent 1 null ===")

rewards = env.reward(actions=[0, 0], signals=[1, null_idx], signal_cost=[COST, COST])
expected = [GAME_REWARD - COST, GAME_REWARD]
check_array("agent 0 non-null, agent 1 null", rewards, expected)


# -----------------------------------------------------------------------------
# Case 4 — agent 0 null, agent 1 non-null → mixed (other way)
# -----------------------------------------------------------------------------
print("\n=== Case 4: agent 0 null, agent 1 non-null ===")

rewards = env.reward(actions=[0, 0], signals=[null_idx, 0], signal_cost=[COST, COST])
expected = [GAME_REWARD, GAME_REWARD - COST]
check_array("agent 0 null, agent 1 non-null", rewards, expected)


# -----------------------------------------------------------------------------
# Case 5 — different costs per agent
# -----------------------------------------------------------------------------
print("\n=== Case 5: per-agent different costs ===")

rewards = env.reward(actions=[0, 0], signals=[0, 0], signal_cost=[0.1, 0.4])
expected = [GAME_REWARD - 0.1, GAME_REWARD - 0.4]
check_array("per-agent different costs", rewards, expected)


# -----------------------------------------------------------------------------
# Case 6 — cost flow when signals=None (no costly application)
# -----------------------------------------------------------------------------
print("\n=== Case 6: signals=None (no cost applied) ===")

rewards = env.reward(actions=[0, 0], signals=None, signal_cost=[COST, COST])
expected = [GAME_REWARD, GAME_REWARD]
check_array("signals=None → no cost", rewards, expected)


# -----------------------------------------------------------------------------
# Case 7 — cost flow when signal_cost=None (no costly application)
# -----------------------------------------------------------------------------
print("\n=== Case 7: signal_cost=None (no cost applied) ===")

rewards = env.reward(actions=[0, 0], signals=[0, 0], signal_cost=None)
expected = [GAME_REWARD, GAME_REWARD]
check_array("signal_cost=None → no cost", rewards, expected)


# -----------------------------------------------------------------------------
# Case 8 — net reward can go negative when cost > game_reward
# -----------------------------------------------------------------------------
print("\n=== Case 8: cost > game_reward → negative net reward ===")

rewards = env.reward(actions=[0, 0], signals=[0, 0], signal_cost=[1.5, 1.5])
expected = [GAME_REWARD - 1.5, GAME_REWARD - 1.5]
check_array("cost > game_reward → r < 0", rewards, expected)


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
