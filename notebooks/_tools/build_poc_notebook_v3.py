"""Build notebooks/proof_of_concept_figures_v3.ipynb as a Roth-Erev-focused
subset of proof_of_concept_figures_v2.ipynb.

v3 drops:
  - Option D-beta (basin of attraction, single-horizon sweep). Its H=10,000
    slice is already the brightest viridis curve in Option F's panels, so
    keeping it duplicated ~450 Roth-Erev sims with no additional signal.
  - Option D-gamma (2D heatmap over (sig_n, act_n)) entirely.
  - Option E (Roth-Erev vs Q-learning side-by-side) -- v3 is Roth-Erev only.
  - The Option D-beta + Option F combined side-by-side view (depended on
    Option D-beta).

v3 modifies:
  - Option F: Roth-Erev only; horizon set extended to include 30 and 50
    episodes; plot redesigned as a 1x2 grid (reward | NMI) with mean +/- std
    shadows, one curve per horizon. Q-learning branch removed (halves compute).

Run after v1 and v2 are up to date:
    python notebooks/_tools/build_poc_notebook.py
    python notebooks/_tools/build_poc_notebook_v2.py
    python notebooks/_tools/build_poc_notebook_v3.py
"""

import json
from pathlib import Path


REPO_ROOT = Path(
    "/Users/ignacio/Documents/VS Code/GitHub Repositories/RL_Signaling"
)
V2_PATH = REPO_ROOT / "notebooks" / "proof_of_concept_figures_v2.ipynb"
V3_PATH = REPO_ROOT / "notebooks" / "proof_of_concept_figures_v3.ipynb"

DROP_SLUGS = {
    "basin-md",
    "basin-compute",
    "basin-beta",
    "basin-gamma-md",
    "basin-gamma-compute",
    "basin-gamma-plot",
    "option-e-md",
    "option-e-ql-compute",
    "option-e-plot",
}


# ---------------------------------------------------------------------------
# Replacement cell sources
# ---------------------------------------------------------------------------

TITLE_V3 = """\
# §2.3 Proof of Concept — Figure Candidates (Roth-Erev)

A stand-alone notebook producing the candidate figures for §2.3 of
*Signaling Games with Distributed Rewards*. Every figure here uses the
Roth-Erev urn agent (`UrnAgent`) on the canonical 2-feature / 2-signal /
4-action signaling game.

Runs **locally** or on **Google Colab**, controlled by the
`RUNNING_LOCALLY` switch in the first code cell:

- **Local** (`RUNNING_LOCALLY = True`): figures are displayed inline
  *and* saved as PNGs under `../results/proof_of_concept/`.
- **Colab** (`RUNNING_LOCALLY = False`): the bootstrap cells clone the
  repo, `pip install -e .` it, **mount Google Drive**, and save PNGs +
  CSVs to a project folder there. Use Colab when you want to crank up
  `N_SEEDS_OPT_A` / `BASIN_N_SEEDS` / `N_EPISODES` past what your laptop
  can comfortably run.

## Figures at a glance

| # | Name | What it shows |
|---|---|---|
| 1 | Initialization sweep (rewards + NMI) | Time-series per init regime; the basin-reachability story. |
| A | Phase-portrait trajectories | Same runs as Fig. 1 but as motion in (NMI, reward) space. |
| F | Time-horizon sweep | Reward and NMI vs `sig_n` with one curve per horizon and std bands. |

Set `SMOKE_TEST = True` in the parameters cell for fast iteration; note
that Option F's max horizon is 10,000 episodes, so `N_EPISODES` must be
at least that — `SMOKE_TEST` clips `N_EPISODES` to 3,000 and will trip
Option F's assertion. Run Option F at the default `N_EPISODES` only.
"""

PARAMS_HEADER_V3 = """\
## Parameters

Every simulation knob lives in the cell below.

**Initialization regimes.** Each `InitSpec` carries independent `(n, m)`
weights for the **signaling urn** and the **action urn**. Across all
four regimes here, the action urn is always initialized uniformly to
`(1, 1)` — only the signaling urn varies. So every regime asks the same
question — "starting from a uniform action policy, how reliably does
learning find a high-reward joint policy?" — under different amounts
of initial signaling pre-coordination, from one-hot deterministic
(`sig=[1,0]`) to fully unbiased (`sig=[1,1]`) to strongly pre-biased
(`sig=[100,1]`). Labels show signaling weights only.

If you're running on Colab, this is where you'd bump up `N_SEEDS_OPT_A`
or `BASIN_N_SEEDS` to take advantage of the extra cores.
"""

PARAMS_V3 = '''\
"""Notebook-level parameters."""

from collections import namedtuple

# Flip to True for fast iteration: smaller seed counts, fewer episodes.
SMOKE_TEST = False

# Time horizon per run. (10k is enough for the trajectories to stabilize;
# 30k was overkill on a laptop.) Also: Option F's max horizon is 10k,
# so N_EPISODES must stay >= 10k for the full sweep to run.
N_EPISODES = 10_000 if not SMOKE_TEST else 3_000

# Per-figure seed counts.
N_SEEDS_OPT_A = 15 if not SMOKE_TEST else 3   # phase-portrait (Option A; excludes sig=[1,0])

# Basin / horizon sweep (Option F) — continuous sig_n sweep with m_sig=1 and act=(1,1) fixed.
BASIN_SIG_N_VALUES = [1, 2, 3, 5, 8, 13, 25, 50, 100] if not SMOKE_TEST else [1, 5, 50]
# More seeds on Colab where the parallelism is essentially free.
BASIN_N_SEEDS = 10 if SMOKE_TEST else (50 if RUNNING_LOCALLY else 500)

# Smoothing windows for the time-series plots (Figure 1).
WINDOW_REWARD = 100
WINDOW_NMI    = 100

# Parallel workers (-1 = all cores).
N_JOBS = -1

# Initialization regimes. Action urn is always initialized uniformly (1, 1);
# only the signaling urn varies. The label below shows only the signaling
# weights since act is invariant across regimes.
InitSpec = namedtuple("InitSpec", ["label", "sig", "act", "color"])
INITS = [
    InitSpec("sig=[1,0]",   sig=(1, 0),   act=(1, 1), color="tab:blue"),
    InitSpec("sig=[1,1]",   sig=(1, 1),   act=(1, 1), color="tab:orange"),
    InitSpec("sig=[5,1]",   sig=(5, 1),   act=(1, 1), color="tab:green"),
    InitSpec("sig=[100,1]", sig=(100, 1), act=(1, 1), color="tab:red"),
]

# Reader-friendly descriptions for the four initialization regimes.
# Used in figure legends and panel titles (paired with the spec.label
# in parens for traceability, e.g. "Frozen signaling (sig=[1,0])").
INIT_DESC = {
    "sig=[1,0]":   "Frozen signaling",
    "sig=[1,1]":   "Uniform start",
    "sig=[5,1]":   "Mild pre-bias",
    "sig=[100,1]": "Strong pre-bias",
}

print(f"SMOKE_TEST  = {SMOKE_TEST}")
print(f"N_EPISODES  = {N_EPISODES:,}")
print(f"INITS:")
for s in INITS:
    print(f"  {s.label:<14}  sig={s.sig}  act={s.act}")
'''

OPTA_MD_V3 = """\
## Option A — Phase-portrait trajectories in (NMI, reward)

**Three panels**, one per non-frozen init regime — `sig=[1,0]` is
dropped because its trajectories sit motionless on the right edge
(NMI = 1.0) and the phase portrait degenerates to vertical motion in
reward. The interesting contrast is among `sig=[1,1]`, `sig=[5,1]`, and
`sig=[100,1]`, which is where the dynamics is doing visible work.

`N_SEEDS_OPT_A` seeds per init. Each seed gives an `N_EPISODES`-episode
trajectory; we smooth both reward and NMI with a 500-episode rolling
mean, then plot each trajectory as a series of small dots in the
(NMI, reward) plane, **colored by episode** (viridis: purple = early,
yellow = late). The endpoint is marked with a black `X`.

### What the picture is doing

The same data as Figure 1, but reorganized: instead of "reward over time"
+ "NMI over time" as two parallel lines, we plot the pair
$(\\text{NMI}_t, \\text{reward}_t)$ as a point that *moves* through the
plane over time. Early-time positions are purple, late-time positions
yellow, and the `X` is where the chain ends up.

This is sometimes called a **phase portrait** — borrowing the term from
dynamical systems, where it shows trajectories of a state moving through
state space.

### What to look for

- `sig=[1,1]` orange: trajectories sweep across the (NMI, reward)
  plane. Some seeds end at high reward / partial NMI (co-adaptation
  succeeded); some end on the left edge near (NMI ≈ 0, reward ≈ 0.5)
  — the no-signaling failure mode.
- `sig=[5,1]` green: trajectories cover less ground than orange and
  end mostly with higher NMI (~0.5–0.95). Endpoint spread on the
  reward axis reflects lock-in to different bijections.
- `sig=[100,1]` red: tight cluster on the right — trajectories barely
  move (start close to where they end up).

### Wrinkle

At the current resolution the trajectories can look like noisy
scribbles because every dot is a 500-episode-smoothed snapshot.
Individual lines are still hard to follow through the cluster. Worth
deciding whether the trajectory *texture* (lots of overlapping paths)
is the right visual emphasis, or whether a smaller seed count with
labeled individual trajectories would communicate better.
"""

OPTION_F_MD_V3 = """\
## Option F — Time-horizon sweep (Roth-Erev): when does initial bias matter?

Sweep both the signaling bias `sig_n` and the horizon `H` so the
transient story is visible alongside the deep-asymptotic basin shape.
Each `sig_n` is run once at `N_EPISODES` episodes; every horizon is
read off the same trajectory, so adding horizons is free.

For each `sig_n` in `BASIN_SIG_N_VALUES` and each horizon
`H ∈ {30, 50, 100, 300, 1000, 3000, 10000}` we record the final-window
average reward and NMI. The window per horizon is `max(10, min(1000, H // 10))`
— short enough to follow the transient, long enough to suppress single-episode
noise.

### What to look for

- The **reward** curves should rise both with `sig_n` (basin reach) and
  with `H` (chain has time to climb into the basin). Roth-Erev's
  additive updates mean both effects compound; neither washes out.
- The **NMI** curves should be nearly horizon-independent at the
  extremes of `sig_n` (low bias → uniformly low NMI; high bias →
  near-saturation NMI even at short horizons because the initial bias
  is locked in by the urn) and most horizon-sensitive in the
  intermediate region where the chain is still moving.
- The 30- and 50-episode curves (the darkest viridis lines) sit close
  to the initial-policy baseline and answer "how quickly does the chain
  *start* climbing toward the basin?"

### Runtime

`len(BASIN_SIG_N_VALUES) × BASIN_N_SEEDS` Roth-Erev sims. Default:
9 × 50 = 450 sims, ~2 min on a 4-core laptop. Every horizon is a
slice of the same `N_EPISODES`-episode trajectory, so the horizon
ladder is free.
"""

OPTION_F_COMPUTE_V3 = '''\
%%time
"""Option F — Roth-Erev. Sweep both signaling bias and horizon. Each
simulation is run once at N_EPISODES; every horizon in HORIZON_VALUES
is a slice of that single trajectory, so adding horizons is free."""

HORIZON_VALUES = [30, 50, 100, 300, 1000, 3000, 10_000]
# Window per horizon: 10 episodes when horizon is tiny, otherwise H // 10
# capped at 1,000.
HORIZON_WINDOWS = {h: max(10, min(1000, h // 10)) for h in HORIZON_VALUES}

assert max(HORIZON_VALUES) <= N_EPISODES, (
    f"Max horizon {max(HORIZON_VALUES)} exceeds N_EPISODES {N_EPISODES}. "
    "Increase N_EPISODES in the Parameters cell or shorten HORIZON_VALUES."
)


def run_horizon_seed(sig_n, sig_m, seed):
    spec = InitSpec(label=f"sig=[{sig_n},{sig_m}]", sig=(sig_n, sig_m),
                    act=(1, 1), color="tab:gray")
    env = build_env_from_spec(spec, seed, agent_type=UrnAgent)
    _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
    # Match the agent-0 convention used by the other basin DataFrames.
    records = []
    for H in HORIZON_VALUES:
        W = HORIZON_WINDOWS[H]
        records.append({
            "agent": "UrnAgent",
            "sig_n": sig_n,
            "sig_m": sig_m,
            "seed": seed,
            "horizon": H,
            "window": W,
            "final_reward": float(np.mean(rewards[0][H - W : H])),
            "final_nmi":    float(np.mean(nmi[0][H - W : H])),
        })
    return records


tasks_F = [(n, 1, s) for n in BASIN_SIG_N_VALUES for s in range(BASIN_N_SEEDS)]
print(f"Running {len(tasks_F)} Roth-Erev horizon sims "
      f"({BASIN_N_SEEDS} seeds x {len(BASIN_SIG_N_VALUES)} sig_n)...")
with tqdm_joblib(tqdm(desc="time-horizon sweep (Roth-Erev)", total=len(tasks_F))):
    records_F_nested = Parallel(n_jobs=N_JOBS)(
        delayed(run_horizon_seed)(n, m, s) for (n, m, s) in tasks_F
    )
records_F = [rec for sublist in records_F_nested for rec in sublist]
df_horizon = pd.DataFrame(records_F)
save_csv(df_horizon, "horizon_sweep_data_roth_erev.csv")
print(f"Collected {len(df_horizon)} records over "
      f"sig_n = {BASIN_SIG_N_VALUES}, horizons = {HORIZON_VALUES}")
'''

OPTION_F_PLOT_V3 = '''\
"""Option F — Roth-Erev: 1x2 grid of final reward (left) and final NMI (right)
vs initial signaling bias, with one curve per horizon (mean +/- std shadow).
Horizons colour-coded with viridis from short (dark) to long (bright)."""

import matplotlib as mpl

metrics = [("final_reward", "Final reward"),
           ("final_nmi",    "Final NMI")]

cmap = mpl.colormaps["viridis"]
horizon_colors = {H: cmap(i / max(1, len(HORIZON_VALUES) - 1))
                  for i, H in enumerate(HORIZON_VALUES)}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True)
for ax, (metric_col, metric_label) in zip(axes, metrics):
    for H in HORIZON_VALUES:
        df_H = df_horizon[df_horizon["horizon"] == H]
        g = df_H.groupby("sig_n")[metric_col]
        mean = g.mean()
        std = g.std()
        ax.plot(mean.index, mean.values,
                color=horizon_colors[H], lw=2, marker="o",
                label=f"{H:,} episodes")
        ax.fill_between(mean.index,
                        np.clip(mean.values - std.values, 0, 1),
                        np.clip(mean.values + std.values, 0, 1),
                        color=horizon_colors[H], alpha=0.12)
    ax.axhline(0.5, ls=":", c="grey", alpha=0.6,
               label="No-signaling baseline (0.5)" if metric_col == "final_reward" else None)
    ax.set_xscale("log")
    ax.set_xlabel("Initial signaling bias (log scale)")
    ax.set_ylabel(metric_label)
    ax.set_ylim(0, 1.05)

axes[1].legend(title="Horizon (episodes)", loc="lower right",
               fontsize=8, framealpha=0.9, ncol=2)
fig.suptitle(
    f"Roth-Erev: final reward and NMI vs initial signaling bias, by horizon  "
    f"({BASIN_N_SEEDS} trials per value; bands = mean +/- std)",
    fontsize=12,
)
plt.tight_layout()
save_and_show("proof_of_concept_plot.png")
'''

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _md_cell(slug: str, src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": slug,
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def _code_cell(slug: str, src: str) -> dict:
    return {
        "cell_type": "code",
        "id": slug,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def main() -> None:
    if not V2_PATH.exists():
        raise SystemExit(
            f"v2 notebook missing: {V2_PATH}. Run build_poc_notebook_v2.py first."
        )
    notebook = json.loads(V2_PATH.read_text())

    new_cells = []
    for cell in notebook["cells"]:
        slug = cell.get("id", "")
        if slug in DROP_SLUGS:
            continue
        if slug == "title":
            cell = dict(cell)
            cell["source"] = TITLE_V3.splitlines(keepends=True)
        elif slug == "params-header":
            cell = dict(cell)
            cell["source"] = PARAMS_HEADER_V3.splitlines(keepends=True)
        elif slug == "params":
            cell = dict(cell)
            cell["source"] = PARAMS_V3.splitlines(keepends=True)
        elif slug == "optA-md":
            cell = dict(cell)
            cell["source"] = OPTA_MD_V3.splitlines(keepends=True)
        elif slug == "option-f-md":
            cell = dict(cell)
            cell["source"] = OPTION_F_MD_V3.splitlines(keepends=True)
        elif slug == "option-f-compute":
            cell = dict(cell)
            cell["source"] = OPTION_F_COMPUTE_V3.splitlines(keepends=True)
        elif slug == "option-f-plot":
            cell = dict(cell)
            cell["source"] = OPTION_F_PLOT_V3.splitlines(keepends=True)
        new_cells.append(cell)

    notebook["cells"] = new_cells
    V3_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"Wrote {V3_PATH}  ({len(new_cells)} cells)")


if __name__ == "__main__":
    main()
