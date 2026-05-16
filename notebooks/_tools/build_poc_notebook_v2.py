"""Build notebooks/proof_of_concept_figures_v2.ipynb as a shortlist subset
of proof_of_concept_figures.ipynb.

v2 drops:
  - Figure 2 (per-seed (NMI, reward) scatter)
  - Option B (per-cell hot-fraction concentration)
  - Option C (reward distribution over the 2304 absorbing states)
  - Option D-α (final-reward histograms across signaling pre-bias)

v2 keeps Figure 1, Option A, Option D-β, Option D-γ, Option E, Option F,
plus the env / params / setup / disconnect scaffolding.

Run after v1 is up to date:
    python notebooks/_tools/build_poc_notebook.py
    python notebooks/_tools/build_poc_notebook_v2.py
"""

import json
from pathlib import Path


REPO_ROOT = Path(
    "/Users/ignacio/Documents/VS Code/GitHub Repositories/RL_Signaling"
)
V1_PATH = REPO_ROOT / "notebooks" / "proof_of_concept_figures.ipynb"
V2_PATH = REPO_ROOT / "notebooks" / "proof_of_concept_figures_v2.ipynb"

# Cells to drop from v1, by stable id.
DROP_SLUGS = {
    "fig2-md", "fig2-code",
    "optB-md", "optB-code",
    "optC-md", "optC-code",
    "basin-alpha",
}


# ---------------------------------------------------------------------------
# Replacement cell sources
# ---------------------------------------------------------------------------

TITLE_V2 = """\
# §2.3 Proof of Concept — Figure Candidates (shortlist, v2)

This notebook is a curated subset of
[`proof_of_concept_figures.ipynb`](proof_of_concept_figures.ipynb). It
keeps only the candidates under active consideration for §2.3 of
*Signaling Games with Distributed Rewards* and drops Figure 2,
Option B, Option C, and Option D-α. Use v1 if you need any of the
dropped variants.

Runs **locally** or on **Google Colab**, controlled by the
`RUNNING_LOCALLY` switch in the first code cell:

- **Local** (`RUNNING_LOCALLY = True`): figures are displayed inline
  *and* saved as PNGs under `../results/proof_of_concept/`.
- **Colab** (`RUNNING_LOCALLY = False`): the bootstrap cells clone the
  repo, `pip install -e .` it, **mount Google Drive**, and save PNGs +
  CSVs to a project folder there. Use Colab when you want to crank up
  `N_SEEDS_OPT_A` / `BASIN_N_SEEDS` / `N_EPISODES` past what your laptop
  can comfortably run.

Each figure section follows the same shape:

1. **Markdown explainer** — what the figure shows, what to look for,
   the mechanism behind it, and any wrinkles.
2. **Code cell** — uses notebook-local helpers (asymmetric init is not
   supported by the analytics-script helpers as written).

## The shortlist at a glance

| # | Name | What it shows |
|---|---|---|
| 1 | Initialization sweep (rewards + NMI) | Time-series per init regime; the basin-reachability story. |
| A | Phase-portrait trajectories | Same runs as Fig. 1 but as motion in (NMI, reward) space. |
| D-β | Basin of attraction (mean ± std curves) | Continuous `sig_n` sweep, reward and NMI overlaid; the dissociation visible at a glance. |
| D-γ | 2D basin heatmap over `(sig_n, act_n)` | Joint dependence of basin reach on signaling and action pre-bias. |
| E | Roth–Erev vs Q-learning side-by-side | D-β-style plot for both agents on shared axes; the comparative basin question. |
| F | Time-horizon sweep | D-β-style plot at multiple horizons; reveals when initial bias matters vs washes out. |

Set `SMOKE_TEST = True` in the parameters cell for fast iteration; the
default reproduces paper-quality figures in roughly 4–6 minutes on a
4-core laptop (longer if Option F's full horizon sweep runs).
"""

BASIN_GAMMA_COMPUTE_V2 = '''\
%%time
"""Option D-γ — compute the (sig_n, act_n) grid.

Heavy: `GRID_SIG_N_VALUES × GRID_ACT_N_VALUES × GRID_N_SEEDS` simulations
(default 9 × 9 × 50 = 4,050). Skipped by default; flip `RUN_OPT_D_GAMMA`
below to True if you actually want the heatmap."""

RUN_OPT_D_GAMMA = False

if RUN_OPT_D_GAMMA:
    def run_grid_seed(sig_n, act_n, seed):
        spec = InitSpec(
            label=f"sig=({sig_n},1), act=({act_n},1)",
            sig=(sig_n, 1),
            act=(act_n, 1),
            color="tab:gray",
        )
        env = build_env_from_spec(spec, seed)
        _, rewards, _, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
        return {
            "sig_n": sig_n,
            "act_n": act_n,
            "seed": seed,
            "final_reward": float(np.mean(rewards[0][-1000:])),
        }

    tasks_G = [
        (sn, an, s)
        for sn in GRID_SIG_N_VALUES
        for an in GRID_ACT_N_VALUES
        for s in range(GRID_N_SEEDS)
    ]
    print(f"Running {len(tasks_G)} sims "
          f"({GRID_N_SEEDS} seeds × {len(GRID_SIG_N_VALUES)} sig_n × {len(GRID_ACT_N_VALUES)} act_n)...")
    with tqdm_joblib(tqdm(desc="(sig_n, act_n) grid", total=len(tasks_G))):
        records_G = Parallel(n_jobs=N_JOBS)(
            delayed(run_grid_seed)(sn, an, s) for (sn, an, s) in tasks_G
        )
    df_grid = pd.DataFrame(records_G)
    save_csv(df_grid, "basin_gamma_grid_data.csv")
    print(f"Collected {len(df_grid)} records over {len(GRID_SIG_N_VALUES)} × {len(GRID_ACT_N_VALUES)} grid.")
else:
    print("Option D-γ skipped (RUN_OPT_D_GAMMA = False). Set the flag to True to run.")
'''

BASIN_GAMMA_PLOT_V2 = '''\
"""Option D-γ — render the 2D success-rate heatmap. No-op if compute was skipped."""

if not RUN_OPT_D_GAMMA:
    print("Option D-γ plot skipped (compute was skipped).")
else:
    success_rate = (
        df_grid.assign(success=lambda d: d["final_reward"] > REWARD_THRESHOLD)
               .groupby(["sig_n", "act_n"])["success"]
               .mean()
               .unstack(level="act_n")
               .sort_index(ascending=True)
               .sort_index(axis=1, ascending=True)
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(success_rate.values, origin="lower", cmap="magma",
                   vmin=0, vmax=1, aspect="auto")

    for i in range(success_rate.shape[0]):
        for j in range(success_rate.shape[1]):
            val = success_rate.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.45 else "black", fontsize=9)

    ax.set_xticks(range(len(success_rate.columns)))
    ax.set_xticklabels(success_rate.columns)
    ax.set_yticks(range(len(success_rate.index)))
    ax.set_yticklabels(success_rate.index)
    ax.set_xlabel("Initial action bias")
    ax.set_ylabel("Initial signaling bias")
    ax.set_title(
        f"Probability of reaching high reward, by initial signaling and action biases  "
        f"(success = final reward > {REWARD_THRESHOLD}; {GRID_N_SEEDS} trials per cell)"
    )
    plt.colorbar(im, ax=ax, label=f"Probability of final reward > {REWARD_THRESHOLD}")
    plt.tight_layout()
    save_and_show("basin_gamma_heatmap.png")
'''


BASIN_MD_V2 = """\
## Option D-β — Basin of attraction (continuous signaling-bias sweep)

This section sweeps the signaling pre-bias parameter `n_sig` continuously
(holding `m_sig = 1` and `act = (1, 1)` fixed) to see how the basin of
attraction of high-reward joint policies depends on the amount of
initial signaling coordination.

**D-β (mean ± std curves)** — x-axis is `sig_n` on a log scale, y-axis
is the final value. Red curve = mean reward across trials, shaded band
= mean ± 1 std (clipped to `[0, 1]`). Green curve / band = same for
NMI. Reward and NMI are overlaid so the dissociation between them is
visible across the sweep — the dropping signaling-pre-bias forces the
chain through the basin boundary.

It answers: "how much signaling pre-bias is needed for the joint chain
to reach high reward reliably?"

The compute cell runs the full sweep once; the plot cell below renders
β from the resulting DataFrame.

**Runtime.** `len(BASIN_SIG_N_VALUES) × BASIN_N_SEEDS` simulations,
parallelized. Default: 9 × 50 = 450 sims, roughly 2 min on a 4-core
laptop. Bump `BASIN_N_SEEDS` (in the Parameters cell) on Colab if you
want tighter bands.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main() -> None:
    if not V1_PATH.exists():
        raise SystemExit(
            f"v1 notebook missing: {V1_PATH}. Run build_poc_notebook.py first."
        )
    notebook = json.loads(V1_PATH.read_text())

    new_cells = []
    for cell in notebook["cells"]:
        slug = cell.get("id", "")
        if slug in DROP_SLUGS:
            continue
        if slug == "title":
            cell = dict(cell)
            cell["source"] = TITLE_V2.splitlines(keepends=True)
        elif slug == "basin-md":
            cell = dict(cell)
            cell["source"] = BASIN_MD_V2.splitlines(keepends=True)
        elif slug == "basin-gamma-compute":
            cell = dict(cell)
            cell["source"] = BASIN_GAMMA_COMPUTE_V2.splitlines(keepends=True)
        elif slug == "basin-gamma-plot":
            cell = dict(cell)
            cell["source"] = BASIN_GAMMA_PLOT_V2.splitlines(keepends=True)
        new_cells.append(cell)

    notebook["cells"] = new_cells
    V2_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"Wrote {V2_PATH}  ({len(new_cells)} cells)")


if __name__ == "__main__":
    main()
