"""Generate three candidate figures for §2.3 (Proof of Concept).

Produces three PNGs under results/proof_of_concept/:
  - poc_optionA_phase_portrait.png    — (NMI_t, reward_t) trajectories per init
  - poc_optionB_cell_concentration.png — per-cell hot-fraction trajectories
  - poc_optionC_absorbing_distribution.png — histogram of mean reward over the 2304 absorbing states

Each plot is meant to be DIDACTIC for §2.3 (illustrating the mechanism), not
aggregate empirical reliability (which is §3's job — Run_Simulations.ipynb).
"""

import copy
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rl_signaling import MultiAgentEnv, UrnAgent, run_simulation  # noqa: E402
from rl_signaling.games import create_random_canonical_game  # noqa: E402

RESULTS = REPO_ROOT / "results" / "proof_of_concept"
N_FEATURES = N_SIG = 2
N_ACT = 4
N_EPISODES = 30_000

INITS = [(1, 0), (1, 1), (5, 1), (100, 1)]
INIT_COLORS = {
    (1, 0): "tab:blue",
    (1, 1): "tab:orange",
    (5, 1): "tab:green",
    (100, 1): "tab:red",
}


def build_env(n_init: int, m_init: int, seed: int) -> MultiAgentEnv:
    np.random.seed(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    graph.add_edges_from([(0, 1), (1, 0)])
    games = {i: create_random_canonical_game(N_FEATURES, N_ACT) for i in range(2)}
    return MultiAgentEnv(
        2, N_FEATURES, N_SIG, N_ACT,
        full_information=False, game_dicts=games,
        observed_variables={0: [0], 1: [1]},
        agent_type=UrnAgent, graph=graph,
        agent_kwargs={"initialize": True, "initialization_weights": (n_init, m_init)},
    )


# ---------------------------------------------------------------------------
# Option A — phase-portrait trajectories in the (NMI, reward) plane
# ---------------------------------------------------------------------------

A_SEEDS = list(range(8))
A_SMOOTH_WIN = 500


def run_for_A(n_init: int, m_init: int, seed: int) -> dict:
    env = build_env(n_init, m_init, seed)
    _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
    r = pd.Series(rewards[0]).rolling(A_SMOOTH_WIN, min_periods=1).mean().to_numpy()
    n = pd.Series(nmi[0]).rolling(A_SMOOTH_WIN, min_periods=1).mean().to_numpy()
    return {"init": (n_init, m_init), "seed": seed, "reward": r, "nmi": n}


def plot_option_A(records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    for ax, init in zip(axes, INITS):
        for rec in [r for r in records if r["init"] == init]:
            cmap = plt.get_cmap("viridis")
            t = np.linspace(0, 1, len(rec["nmi"]))
            ax.scatter(rec["nmi"], rec["reward"], c=t, cmap=cmap, s=1, alpha=0.4)
            ax.scatter(rec["nmi"][-1], rec["reward"][-1], c="black", s=30,
                       marker="X", zorder=10)
        ax.axhline(0.25, ls="--", c="grey", alpha=0.5)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"init = {init}")
        ax.set_xlabel("NMI (smoothed)")
    axes[0].set_ylabel("Reward (smoothed)")
    fig.suptitle("Option A — phase-portrait trajectories in (NMI, reward) plane "
                 f"({len(A_SEEDS)} seeds per init; color = time; X = endpoint)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "poc_optionA_phase_portrait.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS}/poc_optionA_phase_portrait.png")


# ---------------------------------------------------------------------------
# Option B — per-cell hot-fraction trajectories
# ---------------------------------------------------------------------------

B_SEEDS = list(range(6))
B_INITS = [(1, 1), (5, 1)]
B_SNAPSHOT_EVERY = 50


def run_for_B(n_init: int, m_init: int, seed: int) -> dict:
    """Run manually so we can snapshot agent 0's signaling row 0 over time."""
    env = build_env(n_init, m_init, seed)
    snapshots = []
    for episode in range(N_EPISODES):
        _, observations = env.reset()
        signals, new_observations = env.step_signal(observations)
        actions = env.step_action(new_observations)
        rewards = env.reward(actions)
        env.update(observations, signals, new_observations, actions, rewards)
        if episode % B_SNAPSHOT_EVERY == 0:
            urn = env.agents[0].signaling_urns[(0,)]  # row 0 of agent 0's signaling
            total = float(urn.sum())
            hot_frac = float(urn.max() / total) if total > 0 else 0.5
            snapshots.append((episode, hot_frac))
    eps = np.array([s[0] for s in snapshots])
    rho = np.array([s[1] for s in snapshots])
    return {"init": (n_init, m_init), "seed": seed, "episodes": eps, "rho": rho}


def plot_option_B(records: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, init in zip(axes, B_INITS):
        subs = [r for r in records if r["init"] == init]
        for rec in subs:
            ax.plot(rec["episodes"], rec["rho"], alpha=0.7, lw=1.2)
        ax.axhline(0.5, ls=":", c="grey", alpha=0.5, label="uniform (ρ = 0.5)")
        ax.axhline(1.0, ls="--", c="black", alpha=0.3, label="one-hot (ρ = 1.0)")
        ax.set_xlabel("Episode")
        ax.set_title(f"init = {init}")
        ax.set_ylim(0.4, 1.05)
    axes[0].set_ylabel("Hot-cell fraction ρ_t = max(urn) / sum(urn)\nfor agent 0, signaling row 0")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Option B — per-cell hot-fraction concentration "
                 f"({len(B_SEEDS)} seeds per init; single signaling row of one agent)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS / "poc_optionB_cell_concentration.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS}/poc_optionB_cell_concentration.png")


# ---------------------------------------------------------------------------
# Option C — distribution of mean reward over the 2304 absorbing states
# ---------------------------------------------------------------------------


def enumerate_absorbing_rewards(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return per-agent mean reward (r0, r1) over the 2304 absorbing states."""
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)
    games = {i: create_random_canonical_game(N_FEATURES, N_ACT, n=1, m=0) for i in range(2)}
    world_states = list(itertools.product([0, 1], repeat=N_FEATURES))

    # bijection: 2! = 2 signaling maps; 4! = 24 action maps; per-agent = 48.
    sig_maps = list(itertools.permutations(range(N_SIG)))  # 2 of these
    act_keys = list(itertools.product([0, 1], range(N_SIG)))  # 4 (obs, sig) keys
    act_maps = list(itertools.permutations(range(N_ACT)))  # 24

    r0_list, r1_list = [], []
    for f0 in sig_maps:
        for f1 in sig_maps:
            for g0 in act_maps:
                for g1 in act_maps:
                    r0 = 0.0; r1 = 0.0
                    for (x, y) in world_states:
                        sig0 = f0[x]; sig1 = f1[y]
                        a0 = g0[act_keys.index((x, sig1))]
                        a1 = g1[act_keys.index((y, sig0))]
                        r0 += games[0][(x, y)][a0]
                        r1 += games[1][(x, y)][a1]
                    r0_list.append(r0 / 4); r1_list.append(r1 / 4)
    return np.array(r0_list), np.array(r1_list)


def plot_option_C() -> None:
    r0, r1 = enumerate_absorbing_rewards(seed=0)
    n_total = len(r0)
    n_ideal = int(np.sum((r0 == 1.0) & (r1 == 1.0)))
    n_trap = int(np.sum((r0 == 0.0) & (r1 == 0.0)))
    mean_marg = r0.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    bin_edges = np.array([0, 0.125, 0.375, 0.625, 0.875, 1.05])
    labels = ["0.00", "0.25", "0.50", "0.75", "1.00"]
    counts, _ = np.histogram(r0, bins=bin_edges)
    colors = ["#b9504e", "#c08e6b", "#c7b48f", "#a7b878", "#3a8a3a"]
    bars = ax.bar(labels, counts, color=colors, edgecolor="white")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 20, f"{c}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Mean reward over the 4 world states, per agent")
    ax.set_ylabel(f"Number of joint absorbing states (out of {n_total})")
    ax.set_title("Marginal distribution (one agent)")
    ax.axhline(0, color="black", lw=0.5)

    ax2 = axes[1]
    hist2d, xe, ye = np.histogram2d(
        r0, r1, bins=[bin_edges, bin_edges]
    )
    im = ax2.imshow(hist2d.T, origin="lower", cmap="magma_r",
                    extent=[0, 5, 0, 5], aspect="auto")
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = int(hist2d[i, j])
            ax2.text(i + 0.5, j + 0.5, f"{v}", ha="center", va="center",
                     color="white" if v > 200 else "black", fontsize=9)
    ax2.set_xticks(np.arange(len(labels)) + 0.5); ax2.set_xticklabels(labels)
    ax2.set_yticks(np.arange(len(labels)) + 0.5); ax2.set_yticklabels(labels)
    ax2.set_xlabel("Mean reward, agent 0")
    ax2.set_ylabel("Mean reward, agent 1")
    ax2.set_title("Joint distribution")
    plt.colorbar(im, ax=ax2, label="count")

    fig.suptitle(
        f"Option C — reward distribution over the {n_total} absorbing states "
        f"(game seed 0; {n_ideal} ideal, {n_trap} traps; mean = {mean_marg:.2f})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(RESULTS / "poc_optionC_absorbing_distribution.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS}/poc_optionC_absorbing_distribution.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Option A: phase-portrait trajectories ===")
    tasks_A = [(n, m, s) for (n, m) in INITS for s in A_SEEDS]
    records_A = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_for_A)(n, m, s) for (n, m, s) in tasks_A
    )
    plot_option_A(records_A)

    print("=== Option B: per-cell hot-fraction trajectories ===")
    tasks_B = [(n, m, s) for (n, m) in B_INITS for s in B_SEEDS]
    records_B = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_for_B)(n, m, s) for (n, m, s) in tasks_B
    )
    plot_option_B(records_B)

    print("=== Option C: absorbing-state reward distribution ===")
    plot_option_C()
    print("All three option PNGs written to results/proof_of_concept/.")
