"""Build notebooks/proof_of_concept_figures_v3.ipynb as a Roth-Erev-focused
subset of proof_of_concept_figures_v2.ipynb.

v3 drops:
  - Option D-gamma (2D heatmap over (sig_n, act_n)) entirely.

v3 modifies:
  - Option F: Roth-Erev only; horizon set extended to include 10 and 50
    episodes; plot redesigned as a 1x2 grid (reward | NMI) with mean +/- std
    shadows, one curve per horizon. Q-learning branch removed (halves compute).

v3 adds:
  - A new combined cell after Option F that places Option D-beta (deep
    asymptotic) and the new Option F (multi-horizon) side by side as a
    single 1x3 composite figure. No new compute -- reuses df_basin and
    df_horizon.

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
    "basin-gamma-md",
    "basin-gamma-compute",
    "basin-gamma-plot",
}


# ---------------------------------------------------------------------------
# Replacement cell sources
# ---------------------------------------------------------------------------

TITLE_V3 = """\
# §2.3 Proof of Concept — Figure Candidates (Roth-Erev focus, v3)

This notebook is a Roth-Erev-focused subset of
[`proof_of_concept_figures_v2.ipynb`](proof_of_concept_figures_v2.ipynb).
It drops Option D-γ (2D heatmap) entirely, refocuses Option F on Roth-Erev
alone (Q-learning's flat-curve story is deferred to §4), and adds a
combined view that places Option D-β and the new Option F side by side.

Runs **locally** or on **Google Colab**, controlled by the
`RUNNING_LOCALLY` switch in the first code cell:

- **Local** (`RUNNING_LOCALLY = True`): figures are displayed inline
  *and* saved as PNGs under `../results/proof_of_concept/`.
- **Colab** (`RUNNING_LOCALLY = False`): the bootstrap cells clone the
  repo, `pip install -e .` it, **mount Google Drive**, and save PNGs +
  CSVs to a project folder there. Use Colab when you want to crank up
  `N_SEEDS_OPT_A` / `BASIN_N_SEEDS` / `N_EPISODES` past what your laptop
  can comfortably run.

## The v3 shortlist at a glance

| # | Name | What it shows |
|---|---|---|
| 1 | Initialization sweep (rewards + NMI) | Time-series per init regime; the basin-reachability story. |
| A | Phase-portrait trajectories | Same runs as Fig. 1 but as motion in (NMI, reward) space. |
| D-β | Basin of attraction (mean ± std curves) | Continuous `sig_n` sweep at H=10,000, reward and NMI overlaid. |
| E | Roth–Erev vs Q-learning side-by-side | D-β-style plot for both agents on shared axes. |
| F | Time-horizon sweep (Roth-Erev) | Reward and NMI vs `sig_n` with one curve per horizon and std bands. |
| F + D-β combined | Side-by-side composite | Deep asymptotic (D-β) next to the horizon ladder (F). |

Set `SMOKE_TEST = True` in the parameters cell for fast iteration; note
that Option F's max horizon is 10,000 episodes, so `N_EPISODES` must be
at least that — `SMOKE_TEST` clips `N_EPISODES` to 3,000 and will trip
Option F's assertion. Run Option F at the default `N_EPISODES` only.
"""

OPTION_F_MD_V3 = """\
## Option F — Time-horizon sweep (Roth-Erev): when does initial bias matter?

Option D-β fixes the horizon at 10,000 episodes — a deep-asymptotic
measurement that hides the dynamics. Option F unfolds the same Roth-Erev
basin sweep across a horizon ladder so the transient story becomes
visible.

For each `sig_n` in `BASIN_SIG_N_VALUES` and each horizon
`H ∈ {10, 50, 100, 300, 1000, 3000, 10000}` we record the final-window
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

The 10- and 50-episode curves are the new additions: they sit close to
the initial-policy baseline and answer "how quickly does the chain
*start* climbing toward the basin?" — a question that the original
`H ∈ {100, 300, 1000, 3000, 10000}` set could not answer.

### Runtime

`len(BASIN_SIG_N_VALUES) × BASIN_N_SEEDS` Roth-Erev sims — same cost as
Option D-β (no extra horizons cost anything because every horizon is a
slice of the same 10,000-episode trajectory). Default: 9 × 50 = 450
sims, ~2 min on a 4-core laptop. Q-learning is intentionally dropped
from this section; see Option E for the Roth-Erev vs Q-learning
comparison and §4 (planned) for the Q-learning horizon story.
"""

OPTION_F_COMPUTE_V3 = '''\
%%time
"""Option F (v3) — Roth-Erev only. Sweep both signaling bias and horizon.
Each simulation is run once at N_EPISODES; every horizon in HORIZON_VALUES
is a slice of that single trajectory, so adding horizons is free."""

HORIZON_VALUES = [10, 50, 100, 300, 1000, 3000, 10_000]
# Window per horizon: 10 episodes when horizon is tiny, otherwise H // 10
# capped at 1,000. Matches the Option E / v2 Option F convention.
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
"""Option F (v3) — Roth-Erev: 1x2 grid of final reward (left) and final NMI
(right) vs initial signaling bias, with one curve per horizon (mean +/- std
shadow). Horizons colour-coded with viridis from short (dark) to long (bright)."""

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
save_and_show("horizon_sweep_roth_erev.png")
'''

COMBINED_MD = """\
## Combined view — Option D-β and Option F side by side

Same Roth-Erev sweep across `BASIN_SIG_N_VALUES`, two perspectives in one
figure:

- **Left** — Option D-β at the deep-asymptotic horizon (10,000 episodes).
  Reward and NMI overlaid so the dissociation between them is visible at a
  glance.
- **Middle / right** — Option F across the horizon ladder
  `H ∈ {10, 50, 100, 300, 1000, 3000, 10000}`. The H=10,000 curve in each
  panel is the same data as Option D-β's mean curve; the other horizons show
  how the basin sharpens as episodes accumulate.

Read together, the panels answer: *initial signaling bias raises the basin
ceiling, but it takes thousands of episodes for the chain to actually inhabit
that basin.*

(No new compute — both panels are rebuilt from `df_basin` and `df_horizon`.)
"""

COMBINED_CODE = '''\
"""Combined view — Option D-beta (deep asymptotic) on the left, Option F
(multi-horizon) split across the middle and right panels. Reuses df_basin
and df_horizon from the previous sections."""

import matplotlib as mpl

cmap = mpl.colormaps["viridis"]
horizon_colors = {H: cmap(i / max(1, len(HORIZON_VALUES) - 1))
                  for i, H in enumerate(HORIZON_VALUES)}

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharex=True, sharey=True)

# --- Panel 1: D-beta (single horizon, reward + NMI overlay) -----------------
ax = axes[0]
g_r = df_basin.groupby("sig_n")["final_reward"]
mean_r, std_r = g_r.mean(), g_r.std()
ax.plot(mean_r.index, mean_r.values,
        color="firebrick", lw=2, marker="o", label="Reward (mean)")
ax.fill_between(mean_r.index,
                np.clip(mean_r.values - std_r.values, 0, 1),
                np.clip(mean_r.values + std_r.values, 0, 1),
                color="firebrick", alpha=0.18, label="Reward (mean +/- std)")

g_n = df_basin.groupby("sig_n")["final_nmi"]
mean_n, std_n = g_n.mean(), g_n.std()
ax.plot(mean_n.index, mean_n.values,
        color="darkgreen", lw=2, marker="s", label="NMI (mean)")
ax.fill_between(mean_n.index,
                np.clip(mean_n.values - std_n.values, 0, 1),
                np.clip(mean_n.values + std_n.values, 0, 1),
                color="darkgreen", alpha=0.15, label="NMI (mean +/- std)")

ax.axhline(0.5, ls=":", c="grey", alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("Initial signaling bias (log scale)")
ax.set_ylabel("Final value (last 1000 episodes)")
ax.set_title("Option D-beta -- H = 10,000")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=7, ncol=2, framealpha=0.9)

# --- Panels 2 & 3: Option F (multi-horizon, std bands) ----------------------
for ax, (metric_col, metric_label) in zip(
    axes[1:],
    [("final_reward", "Final reward"), ("final_nmi", "Final NMI")],
):
    for H in HORIZON_VALUES:
        df_H = df_horizon[df_horizon["horizon"] == H]
        g = df_H.groupby("sig_n")[metric_col]
        mean = g.mean()
        std = g.std()
        ax.plot(mean.index, mean.values,
                color=horizon_colors[H], lw=2, marker="o",
                label=f"{H:,} ep")
        ax.fill_between(mean.index,
                        np.clip(mean.values - std.values, 0, 1),
                        np.clip(mean.values + std.values, 0, 1),
                        color=horizon_colors[H], alpha=0.10)
    ax.axhline(0.5, ls=":", c="grey", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Initial signaling bias (log scale)")
    ax.set_title(f"Option F -- {metric_label}")

axes[2].legend(title="Horizon", loc="lower right", fontsize=7,
               framealpha=0.9, ncol=2)

fig.suptitle(
    f"Roth-Erev -- basin of attraction (D-beta) and time-horizon sweep (F) side by side  "
    f"({BASIN_N_SEEDS} trials per value)",
    fontsize=12,
)
plt.tight_layout()
save_and_show("combined_basin_and_horizon_roth_erev.png")
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
        if slug == "option-f-plot":
            new_cells.append(_md_cell("combined-md", COMBINED_MD))
            new_cells.append(_code_cell("combined-code", COMBINED_CODE))

    notebook["cells"] = new_cells
    V3_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"Wrote {V3_PATH}  ({len(new_cells)} cells)")


if __name__ == "__main__":
    main()
