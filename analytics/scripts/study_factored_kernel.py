"""Exact factored Markov-chain kernel for the 2-agent Roth-Erev signal-trading game.

Modeler-perspective companion to §2 of `analytics/docs/roth_erev_polya_mle.md`.
The reference doc proves that the one-step kernel of the chain factorizes as a
product of urn-fraction terms times deterministic indicators, so the kernel is
*computed*, not estimated. This script ports the doc's `one_step_kernel_value`
to the simulator's dict-of-array representation and validates the kernel
against the simulator at three layers:

  Section 1.  The factored kernel function adapted to the simulator's data
              structures.
  Section 2.  `validate_choice_rule` — Monte Carlo of the choice rule alone,
              isolated from urn dynamics. 100,000 samples; max abs deviation
              from the closed form n / sum(n) must be O(1/sqrt(N)).
  Section 3.  Single-urn transition validation — instrument a 2-agent
              simulation and compare the empirical visit-and-reinforce rate
              for one specific cell against P(x) · (n_sigma/S) · q*(x), where
              q*(x) is itself estimated from the same trace.
  Section 4.  Full-state kernel sum check — pick a small concrete state, walk
              over the 256 candidate next states (2 nature × 4 signals × 16
              actions; rewards and updates are deterministic given those),
              compute one_step_kernel_value for each, verify sum = 1.

Notation matches `analytics/docs/roth_erev_polya_mle.md`. Q-learning analysis
is deferred — see `TODO_WORKFLOW.md::todo.qlearning_proof_of_concept`.

Run:
    .venv/bin/python -m analytics.scripts.study_factored_kernel
"""

from __future__ import annotations

import copy
import random
import sys
from itertools import product

import networkx as nx
import numpy as np

from rl_signaling.agents import UrnAgent
from rl_signaling.env import MultiAgentEnv
from rl_signaling.games import create_random_canonical_game


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
N_AGENTS = 2
N_FEATURES = 2
N_SIG = 2
N_FIN = 4
OBSERVED = {0: [0], 1: [1]}
P_NATURE = np.array([0.5, 0.5])  # nature is uniform on {0, 1} per feature

failures: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> None:
    """Print PASS/FAIL line and accumulate failures."""
    status = "PASS" if ok else "FAIL"
    if not ok:
        failures.append(f"{label}: {detail}")
    print(f"[{status}] {label}{(' — ' + detail) if detail else ''}")


# =============================================================================
# Section 1 — The factored kernel adapted to the simulator's dict urns.
# =============================================================================
print("=" * 88)
print("Section 1: factored kernel function")
print("=" * 88)


def choice_probs(propensity_vec):
    """Roth-Erev choice rule: P(option k) = n_k / sum(n)."""
    n = np.asarray(propensity_vec, dtype=float)
    total = n.sum()
    if total <= 0:
        raise ValueError("Roth-Erev urn requires strictly positive total mass.")
    return n / total


def one_step_kernel_value(s_curr, s_next, P_x=P_NATURE, P_y=P_NATURE,
                          G1=None, G2=None):
    """Exact one-step transition probability P(s_next | s_curr).

    The state is represented in the *simulator's* dict-of-array form:

      's_curr' / 's_next' : dict with keys
            'x', 'y'        : ints in {0, 1} — nature's draws this step
            'f1', 'f2'      : dict[(obs,), np.ndarray]  — signaling urns
                              (f1 is agent-0's, keyed by (x,);
                               f2 is agent-1's, keyed by (y,))
            'g1', 'g2'      : dict[(obs, sig_received), np.ndarray]
                              — action urns
            'sig1', 'sig2'  : ints — signals emitted at this step
            'a1', 'a2'      : ints — actions taken at this step
            'r1', 'r2'      : floats in {0, 1} — rewards delivered

      's_curr' has the urns BEFORE the t -> t+1 update.
      's_next' has the urns AFTER the update.

    P_x, P_y : marginal nature distributions. Uniform on {0, 1} by default.
    G1, G2   : reward functions G_i(action, x, y) -> {0, 1}. Required.

    The function returns 0 if any deterministic constraint is violated.
    """
    if G1 is None or G2 is None:
        raise ValueError("G1 and G2 are required.")

    x_n, y_n = s_next['x'], s_next['y']
    sig1, sig2 = s_next['sig1'], s_next['sig2']
    a1, a2 = s_next['a1'], s_next['a2']
    r1, r2 = s_next['r1'], s_next['r2']

    # --- Stochastic factor 1: nature -----------------------------------------
    p_nature = P_x[x_n] * P_y[y_n]

    # --- Stochastic factor 2: signal urns ------------------------------------
    # Agent-0's signaling urn is keyed by its observation (x,).
    # Agent-1's signaling urn is keyed by its observation (y,).
    p_sig1 = choice_probs(s_curr['f1'][(x_n,)])[sig1]
    p_sig2 = choice_probs(s_curr['f2'][(y_n,)])[sig2]

    # --- Stochastic factor 3: action urns ------------------------------------
    # Agent-0 receives sig2 from agent-1 (predecessor in the bidirectional
    # 2-cycle); agent-1 receives sig1.
    p_a1 = choice_probs(s_curr['g1'][(x_n, sig2)])[a1]
    p_a2 = choice_probs(s_curr['g2'][(y_n, sig1)])[a2]

    # --- Deterministic factor 1: rewards must match the matching games -------
    if r1 != G1(a1, x_n, y_n) or r2 != G2(a2, x_n, y_n):
        return 0.0

    # --- Deterministic factor 2: urns must update by the additive rule -------
    # Roth-Erev clamps at 0; with non-negative integer rewards the clamp is
    # inert, so the expected next urn is just s_curr's urn with one cell
    # incremented by the reward.
    def _expect(urn_dict, key, idx, increment):
        out = copy.deepcopy(urn_dict)
        out[key] = urn_dict[key].copy()
        out[key][idx] = max(0.0, out[key][idx] + increment)
        return out

    f1_expected = _expect(s_curr['f1'], (x_n,),       sig1, r1)
    f2_expected = _expect(s_curr['f2'], (y_n,),       sig2, r2)
    g1_expected = _expect(s_curr['g1'], (x_n, sig2),  a1,   r1)
    g2_expected = _expect(s_curr['g2'], (y_n, sig1),  a2,   r2)

    def _dict_eq(a, b):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(np.array_equal(a[k], b[k]) for k in a)

    if not (_dict_eq(f1_expected, s_next['f1'])
            and _dict_eq(f2_expected, s_next['f2'])
            and _dict_eq(g1_expected, s_next['g1'])
            and _dict_eq(g2_expected, s_next['g2'])):
        return 0.0

    return p_nature * p_sig1 * p_sig2 * p_a1 * p_a2


print("one_step_kernel_value: ported to dict-of-array urn representation.")
print("  signaling urn key:  (obs,)")
print("  action urn key:     (obs, sig_received)")
print()


# =============================================================================
# Section 2 — Choice-rule MC validation (isolated from urn dynamics).
# =============================================================================
print("=" * 88)
print("Section 2: validate_choice_rule")
print("=" * 88)


def validate_choice_rule(propensity_vec, n_samples=100_000, rng=None):
    """Empirical frequencies vs theoretical n / sum(n)."""
    if rng is None:
        rng = np.random.default_rng(0)
    p_theory = choice_probs(propensity_vec)
    samples = rng.choice(len(p_theory), size=n_samples, p=p_theory)
    p_empirical = np.bincount(samples, minlength=len(p_theory)) / n_samples
    max_abs_dev = float(np.max(np.abs(p_empirical - p_theory)))
    return p_empirical, p_theory, max_abs_dev


for prop in [[3, 1], [1, 1, 1, 1], [10, 1, 5, 2]]:
    n_samples = 100_000
    p_emp, p_th, dev = validate_choice_rule(prop, n_samples=n_samples,
                                            rng=np.random.default_rng(0))
    # 1 / sqrt(n_samples) ~= 0.00316; allow 5x slack for the worst-case bin.
    tol = 5.0 / np.sqrt(n_samples)
    _check(
        f"choice_rule({prop})  empirical vs n/sum(n)",
        dev < tol,
        f"max|emp - theory| = {dev:.5f}, tol = {tol:.5f}",
    )

print()


# =============================================================================
# Section 3 — Single-urn transition validation against the simulator.
# =============================================================================
print("=" * 88)
print("Section 3: single-urn transition validation against a 2-agent sim")
print("=" * 88)
# We instrument a 2-agent simulation and watch a single specific cell of
# agent-0's signaling urn — the (x=0,) row. At each episode we record:
#   - whether nature drew x = 0,
#   - if so, which signal sigma was sampled and what reward agent-0 got.
# Empirical events:
#   N_visit(x=0)       = #episodes with nature[0] = 0
#   N_sig(x=0, sig=s)  = #episodes with nature[0] = 0 and sigma_0 = s
#   N_rew(x=0, sig=s)  = #episodes with nature[0] = 0, sigma_0 = s, r_0 = 1
# Predictions from the factored kernel:
#   N_visit(x=0) / T                              -> P_nature[x=0]   = 0.5
#   N_sig(x=0, sig=s) / N_visit(x=0)              -> n[s] / sum(n) (urn fraction)
#   N_rew(x=0, sig=s) / N_sig(x=0, sig=s)         -> q(x=0, sigma=s)
# The doc's §3 "constant q*(x) across colors" claim says q(x=0, sigma=s) is
# the SAME for every s (because agent-0's reward depends on the signal RECEIVED
# from agent-1, not on sigma_0). We therefore also check that q(x=0, sigma=0)
# and q(x=0, sigma=1) agree to within Monte-Carlo noise.

random.seed(42)
np.random.seed(42)

N_EP_S3 = 20_000
INIT_W = (5, 1)  # m > 0 so the chain drifts and we get statistics.

games = {
    i: create_random_canonical_game(N_FEATURES, N_FIN, n=1, m=0)
    for i in range(N_AGENTS)
}
G_simul = nx.DiGraph()
G_simul.add_nodes_from([0, 1])
G_simul.add_edges_from([(0, 1), (1, 0)])

env = MultiAgentEnv(
    n_agents=N_AGENTS,
    n_features=N_FEATURES,
    n_signaling_actions=N_SIG,
    n_final_actions=N_FIN,
    full_information=False,
    game_dicts=games,
    observed_variables=OBSERVED,
    agent_type=UrnAgent,
    agent_kwargs={
        "n_observed_features": 1,
        "initialize": True,
        "initialization_weights": INIT_W,
    },
    graph=G_simul,
)

# Per-(x_obs, sigma) counters for agent-0's signaling cell at x=0.
# We track urn-snapshots BEFORE the step (so we can compute the predicted
# urn fraction on the fly).
visit_count = 0
sig_counts = np.zeros(N_SIG, dtype=int)
sig_reward_counts = np.zeros(N_SIG, dtype=int)
# Predicted vs observed for each cell — we use the time-averaged urn
# fraction, since the urn evolves over the trial.
predicted_sig_freq_sum = np.zeros(N_SIG)

for ep in range(N_EP_S3):
    _, observations = env.reset()
    nature_x = env.nature_vector[0]
    nature_y = env.nature_vector[1]

    # Snapshot agent-0's (x=0,) urn BEFORE this step.
    f1_x0 = env.agents[0].signaling_urns[(0,)].copy()
    if nature_x == 0:
        S = float(f1_x0.sum())
        predicted_sig_freq_sum += f1_x0 / S

    signals, new_observations = env.step_signal(observations)
    actions = env.step_action(new_observations)
    rewards = env.reward(actions)
    env.update(observations, signals, new_observations, actions, rewards)

    if nature_x == 0:
        visit_count += 1
        sig_counts[signals[0]] += 1
        if rewards[0] == 1:
            sig_reward_counts[signals[0]] += 1

# Empirical visit fraction.
emp_visit_frac = visit_count / N_EP_S3
_check(
    "P(x=0) — empirical vs P_nature[0] = 0.5",
    abs(emp_visit_frac - 0.5) < 5.0 / np.sqrt(N_EP_S3),
    f"empirical = {emp_visit_frac:.4f}, theoretical = 0.5",
)

# Empirical signal fractions conditional on x=0, against the time-averaged
# predicted fraction from the urn-fraction integrand.
emp_sig_freq = sig_counts / max(1, visit_count)
pred_sig_freq = predicted_sig_freq_sum / max(1, visit_count)
# Sample noise scales like 1/sqrt(visit_count) per bin.
tol_sig = 5.0 / np.sqrt(max(1, visit_count))
for s in range(N_SIG):
    _check(
        f"P(sigma_0={s} | x=0) — empirical vs urn-fraction integrand",
        abs(emp_sig_freq[s] - pred_sig_freq[s]) < tol_sig,
        f"empirical = {emp_sig_freq[s]:.4f}, predicted = {pred_sig_freq[s]:.4f}, "
        f"tol = {tol_sig:.4f}",
    )

# Doc's §3 claim: q*(x) is constant across signals. Estimate per-signal q.
print("\nq*(x=0) constancy check (doc §3 boxed observation):")
print(f"{'sigma':<8} {'#visits':>10} {'#rew=1':>10} {'q_hat':>10}")
for s in range(N_SIG):
    n_s = int(sig_counts[s])
    n_r = int(sig_reward_counts[s])
    q_hat = (n_r / n_s) if n_s > 0 else float("nan")
    print(f"{s:<8} {n_s:>10} {n_r:>10} {q_hat:>10.4f}")

# Two-sample test: for two binomials with rates p1, p2 and n1, n2 trials each,
# the standard error of (p1 - p2) is sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2)).
n1, n2 = int(sig_counts[0]), int(sig_counts[1])
p1 = sig_reward_counts[0] / max(1, n1)
p2 = sig_reward_counts[1] / max(1, n2)
p_pool = (sig_reward_counts.sum()) / max(1, n1 + n2)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / max(1, n1) + 1 / max(1, n2))) if (n1 + n2) > 0 else float("inf")
gap = abs(p1 - p2)
# 3-sigma allowance.
_check(
    "q*(x=0) constant across signals (gap < 3*SE)",
    gap < 3 * se,
    f"|q_0 - q_1| = {gap:.4f}, 3*SE = {3 * se:.4f}",
)

print()


# =============================================================================
# Section 4 — Full-state kernel sum = 1 over 256 candidate next states.
# =============================================================================
print("=" * 88)
print("Section 4: full-state kernel sum check")
print("=" * 88)
# Pick a concrete s_curr (small integer urns), enumerate every (x, y, sig1,
# sig2, a1, a2) combination, and verify the kernel values sum to 1. Rewards
# and post-update urns are deterministic given those choices, so the
# enumeration is 2 × 2 × 2 × 2 × 4 × 4 = 256 candidate next states.

# Build a small concrete s_curr.
def _mk_urn_dict(keys, vec):
    return {k: np.array(vec, dtype=float) for k in keys}


s_curr_f1 = _mk_urn_dict([(0,), (1,)], [3, 1])
s_curr_f2 = _mk_urn_dict([(0,), (1,)], [2, 2])
# Action urns are keyed by (obs, sig_received) — populate all 4 keys.
s_curr_g1 = _mk_urn_dict(
    [(x, s) for x in (0, 1) for s in (0, 1)], [1, 1, 1, 1]
)
s_curr_g2 = _mk_urn_dict(
    [(y, s) for y in (0, 1) for s in (0, 1)], [2, 1, 1, 1]
)

# A simple deterministic game: G_i(a, x, y) = 1 iff a == 2*x + y.
# This is the canonical "one optimal action per state" pattern.
def G1_demo(a, x, y):
    return 1 if a == 2 * x + y else 0


def G2_demo(a, x, y):
    return 1 if a == 2 * x + y else 0


s_curr = {
    'x': None, 'y': None,
    'f1': s_curr_f1, 'f2': s_curr_f2,
    'g1': s_curr_g1, 'g2': s_curr_g2,
    'sig1': None, 'sig2': None,
    'a1': None, 'a2': None,
    'r1': None, 'r2': None,
}

total_prob = 0.0
n_candidates = 0
for x_n, y_n, sig1, sig2, a1, a2 in product(
    range(2), range(2), range(N_SIG), range(N_SIG), range(N_FIN), range(N_FIN)
):
    r1 = G1_demo(a1, x_n, y_n)
    r2 = G2_demo(a2, x_n, y_n)

    # Apply the deterministic update to get s_next.
    next_f1 = {k: v.copy() for k, v in s_curr_f1.items()}
    next_f2 = {k: v.copy() for k, v in s_curr_f2.items()}
    next_g1 = {k: v.copy() for k, v in s_curr_g1.items()}
    next_g2 = {k: v.copy() for k, v in s_curr_g2.items()}
    next_f1[(x_n,)][sig1] += r1
    next_f2[(y_n,)][sig2] += r2
    next_g1[(x_n, sig2)][a1] += r1
    next_g2[(y_n, sig1)][a2] += r2

    s_next = {
        'x': x_n, 'y': y_n,
        'f1': next_f1, 'f2': next_f2,
        'g1': next_g1, 'g2': next_g2,
        'sig1': sig1, 'sig2': sig2,
        'a1': a1, 'a2': a2,
        'r1': r1, 'r2': r2,
    }
    p = one_step_kernel_value(s_curr, s_next, G1=G1_demo, G2=G2_demo)
    total_prob += p
    n_candidates += 1

_check(
    f"sum over {n_candidates} enumerated next states equals 1",
    abs(total_prob - 1.0) < 1e-12,
    f"sum = {total_prob:.15f}",
)

# Also verify that an INVALID s_next (mismatched reward) returns 0.
bad_s_next = copy.deepcopy(s_next)
bad_s_next['r1'] = 1 - bad_s_next['r1']  # flip the reward
bad_p = one_step_kernel_value(s_curr, bad_s_next, G1=G1_demo, G2=G2_demo)
_check(
    "kernel returns 0 on reward mismatch",
    bad_p == 0.0,
    f"got {bad_p}",
)

# And an INVALID s_next with mismatched urn update.
bad_s_next2 = copy.deepcopy(s_next)
bad_s_next2['f1'][(0,)][0] += 99.0  # corrupt the urn
bad_p2 = one_step_kernel_value(s_curr, bad_s_next2, G1=G1_demo, G2=G2_demo)
_check(
    "kernel returns 0 on urn-update mismatch",
    bad_p2 == 0.0,
    f"got {bad_p2}",
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
print("  - Choice rule: empirical agrees with n / sum(n) within MC tolerance.")
print("  - Single-urn transition: empirical visit/signal frequencies agree with")
print("    P(x) and the urn-fraction integrand. Per-signal reward rates are")
print("    statistically indistinguishable, supporting the doc's §3 q*(x)-")
print("    constant-across-colors observation.")
print("  - Full-state kernel: enumerating all 256 candidate next states for a")
print("    concrete s_curr yields sum_{s'} P(s' | s) = 1 to 1e-12.")
