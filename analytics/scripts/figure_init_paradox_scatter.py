import numpy as np, pandas as pd, matplotlib.pyplot as plt, networkx as nx
from joblib import Parallel, delayed
from rl_signaling import MultiAgentEnv, UrnAgent, run_simulation
from rl_signaling.games import create_random_canonical_game

INITS = [(1, 0), (1, 1), (5, 1), (100, 1)]
N_SEEDS = 200
N_EPISODES = 30_000
N_FEATURES = N_SIG = 2; N_ACT = 4


def run_one(n_init: int, m_init: int, seed: int) -> dict:
    np.random.seed(seed)
    graph = nx.DiGraph(); graph.add_nodes_from([0, 1]); graph.add_edges_from([(0, 1), (1, 0)])
    games = {i: create_random_canonical_game(N_FEATURES, N_ACT) for i in range(2)}
    env = MultiAgentEnv(2, N_FEATURES, N_SIG, N_ACT,
                        full_information=False, game_dicts=games,
                        observed_variables={0: [0], 1: [1]},
                        agent_type=UrnAgent, graph=graph,
                        agent_kwargs={"initialize": True, "initialization_weights": (n_init, m_init)})
    _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
    return {"init": f"({n_init},{m_init})",
            "seed": seed,
            "final_reward": float(np.mean(rewards[0][-1000:])),
            "final_nmi":    float(np.mean(nmi[0][-1000:]))}


if __name__ == "__main__":
    tasks = [(n, m, s) for (n, m) in INITS for s in range(N_SEEDS)]
    print(f"Running {len(tasks)} simulations across all CPU cores...")
    records = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one)(n, m, s) for (n, m, s) in tasks
    )

    df = pd.DataFrame(records)
    df.to_csv("results/proof_of_concept/figure_init_paradox_scatter.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"(1,0)": "tab:blue", "(1,1)": "tab:orange",
              "(5,1)": "tab:green", "(100,1)": "tab:red"}
    for init, sub in df.groupby("init"):
        ax.scatter(sub["final_nmi"], sub["final_reward"], s=14, alpha=0.6,
                   label=f"init = {init}", c=colors[init])
    ax.axhline(0.25, ls="--", c="grey", alpha=0.5, label="random-action baseline")
    ax.set_xlabel("Final NMI (mean over last 1000 episodes)")
    ax.set_ylabel("Final reward (mean over last 1000 episodes)")
    ax.set_title("The (1,0) paradox — Roth–Erev, canonical game, 200 seeds per init")
    ax.legend(loc="lower right"); ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig("results/proof_of_concept/figure_init_paradox_scatter.png", dpi=150)
    print("Saved results/figure_init_paradox_scatter.{csv,png}")
