"""Build notebooks/proof_of_concept_figures.ipynb from cell definitions.

Run from anywhere:
    python /tmp/build_poc_notebook.py
"""

import json
from pathlib import Path


def md(slug: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": slug,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(slug: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "id": slug,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def code_timed(slug: str, source: str) -> dict:
    """Code cell with `%%time` prepended so Jupyter reports wall + CPU time."""
    return code(slug, "%%time\n" + source)


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

TITLE = """\
# §2.3 Proof of Concept — Figure Candidates

This notebook builds every candidate figure under discussion for §2.3 of
*Signaling Games with Distributed Rewards* (the philosophical paper). One
section per figure. The goal is a single place where you can read the
explanation, see the code, look at the plot, and decide.

Runs **locally** or on **Google Colab**, controlled by the
`RUNNING_LOCALLY` switch in the first code cell:

- **Local** (`RUNNING_LOCALLY = True`): figures are displayed inline
  *and* saved as PNGs under `../results/proof_of_concept/`.
- **Colab** (`RUNNING_LOCALLY = False`): the bootstrap cells clone the
  repo, `pip install -e .` it, **mount Google Drive**, and save PNGs +
  CSVs to a project folder there. Use Colab when you want to crank up
  `N_SEEDS_FIG2` / `N_SEEDS_OPT_A` / `N_EPISODES` past what your laptop
  can comfortably run.

Each figure section follows the same shape:

1. **Markdown explainer** — what the figure shows, what to look for,
   the mechanism behind it, and any wrinkles.
2. **Code cell** — uses notebook-local helpers (asymmetric init is not
   supported by the analytics-script helpers as written) plus
   `enumerate_absorbing_rewards` from
   [`../analytics/scripts/figure_poc_options.py`](../analytics/scripts/figure_poc_options.py)
   for Option C.

## The six candidates at a glance

| # | Name | What it shows |
|---|---|---|
| 1 | Initialization sweep (rewards + NMI) | Time-series per init regime; the basin-reachability story. |
| 2 | Per-seed (NMI, reward) scatter | 200 seeds per init projected onto a single 2-D cloud. |
| A | Phase-portrait trajectories | Same runs as Fig. 1 but as motion in (NMI, reward) space. |
| B | Per-cell hot-fraction | A single signaling row concentrating; the local Pólya story. |
| C | Absorbing-state distribution | Structural — the *space* of deterministic policies, mean = 0.25. |
| D | Basin of attraction | α = reward histograms across `sig_n`; β = mean ± std curves for reward and NMI; γ = 2D heatmap over `(sig_n, act_n)`. |
| E | Roth–Erev vs Q-learning side-by-side | D-β-style plot for both agents on shared axes; the comparative basin question. |

Set `SMOKE_TEST = True` in the parameters cell for fast iteration; the
default reproduces paper-quality figures in roughly 4–6 minutes on a
4-core laptop.
"""

# ---------------------------------------------------------------------------
# Environment switch
# ---------------------------------------------------------------------------

ENV_MD = """\
## Environment setup

Three small cells before anything else:

1. **Environment switch** — `RUNNING_LOCALLY` decides everything that
   follows. On local: notebook's parent is the repo root and is added
   to `sys.path`; `RESULTS_DIR` points at `../results/proof_of_concept/`
   so PNGs are saved there. On Colab: Drive is mounted and `RESULTS_DIR`
   points at a project folder under `My Drive`, so PNGs and CSVs persist
   across runtimes. The next two cells handle the clone + install.
2. **Git clone + chdir** — only fires on Colab. Force-fresh clone, then
   `os.chdir` into the clone and put it on `sys.path`. Uses Python
   builtins (`os.chdir`, `subprocess.run`) rather than line magics
   (`%cd`, `!pip`) so the `if not RUNNING_LOCALLY:` guard actually
   works (line magics fire regardless of the surrounding `if`).
3. **Pip install** — only fires on Colab. `pip install -q -e .` so the
   `rl_signaling` package and the `analytics.scripts.*` namespace
   become importable.

If you want to run on Colab, update `REPO_URL` in the clone cell to
match wherever this repo lives publicly.
"""

ENV_CODE = '''\
"""Environment switch — local vs Colab."""

import os
import sys
from pathlib import Path

# True  → laptop run; PNGs save to ../results/proof_of_concept/
# False → Google Colab; PNGs are NOT saved, only displayed inline
RUNNING_LOCALLY = False

if RUNNING_LOCALLY:
    # Notebook lives in <repo>/notebooks/; repo root is the parent.
    REPO_ROOT = Path(os.getcwd()).resolve().parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    RESULTS_DIR = REPO_ROOT / "results" / "proof_of_concept"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Local mode.")
    print(f"  REPO_ROOT   = {REPO_ROOT}")
    print(f"  RESULTS_DIR = {RESULTS_DIR}  (PNGs and CSVs will be saved here)")
else:
    # Colab: mount Drive and save artifacts to a project folder under My Drive.
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS_DIR = Path(
        "/content/drive/My Drive/Colab Projects/Python ABMs/"
        "Distributed Signaling/Plots and Datasets/Proof of Concept/"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Colab mode.")
    print(f"  RESULTS_DIR = {RESULTS_DIR}  (PNGs and CSVs will be saved to Drive)")

print(f"  CPU cores   = {os.cpu_count()}")
'''

CLONE_CODE = '''\
"""Git clone + chdir + sys.path — Colab only."""

REPO_URL = "https://github.com/IgnacioOQ/RL_Signaling"
REPO_BRANCH = "debugging"   # <-- change when this work merges to main
REPO_NAME = "RL_Signaling"

if not RUNNING_LOCALLY:
    import shutil
    import subprocess

    if os.path.exists(REPO_NAME):
        shutil.rmtree(REPO_NAME)
    subprocess.run(
        ["git", "clone", "-b", REPO_BRANCH, REPO_URL],
        check=True,
    )
    os.chdir(REPO_NAME)
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    print(f"Cloned {REPO_URL} (branch: {REPO_BRANCH})")
    print(f"  cwd = {os.getcwd()}")
'''

PIP_CODE = '''\
"""Pip install — Colab only."""

if not RUNNING_LOCALLY:
    import subprocess
    subprocess.run(["pip", "install", "-q", "-e", "."], check=True)
    print("Installed rl_signaling (editable).")
'''

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

PARAMS_MD = """\
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

If you're running on Colab, this is where you'd bump up `N_SEEDS_FIG2`
or `N_SEEDS_OPT_A` to take advantage of the extra cores.
"""

PARAMS_CODE = '''\
"""Notebook-level parameters."""

from collections import namedtuple

# Flip to True for fast iteration: smaller seed counts, fewer episodes.
SMOKE_TEST = False

# Time horizon per run. (10k is enough for the trajectories to stabilize;
# 30k was overkill on a laptop.)
N_EPISODES = 10_000 if not SMOKE_TEST else 3_000

# Per-figure seed counts.
N_SEEDS_FIG1   = 1                                # one trajectory per init
N_SEEDS_FIG2   = 200 if not SMOKE_TEST else 20    # per-seed scatter
N_SEEDS_OPT_A  = 15  if not SMOKE_TEST else 3     # phase-portrait (excludes sig=[1,0])
N_SEEDS_OPT_B  = 6   if not SMOKE_TEST else 3     # per-cell concentration
GAME_SEED_OPT_C = 0                               # enumeration is deterministic

# Basin sweep (Option D-α, D-β) — continuous sig_n sweep with m_sig=1 and act=(1,1) fixed.
BASIN_SIG_N_VALUES = [1, 2, 3, 5, 8, 13, 25, 50, 100] if not SMOKE_TEST else [1, 5, 50]
# More seeds on Colab where the parallelism is essentially free.
BASIN_N_SEEDS = 10 if SMOKE_TEST else (50 if RUNNING_LOCALLY else 500)

# Basin grid (Option D-γ) — 2D heatmap over (sig_n, act_n). Relaxes act=(1,1).
if SMOKE_TEST:
    GRID_SIG_N_VALUES = [1, 5, 50]
    GRID_ACT_N_VALUES = [1, 5, 50]
    GRID_N_SEEDS = 5
elif RUNNING_LOCALLY:
    GRID_SIG_N_VALUES = [1, 2, 5, 13, 50]
    GRID_ACT_N_VALUES = [1, 2, 5, 13, 50]
    GRID_N_SEEDS = 20
else:  # Colab
    GRID_SIG_N_VALUES = [1, 2, 3, 5, 8, 13, 25, 50, 100]
    GRID_ACT_N_VALUES = [1, 2, 3, 5, 8, 13, 25, 50, 100]
    GRID_N_SEEDS = 50
REWARD_THRESHOLD = 0.9   # cell color = P(final reward > this)

# Q-learning parameters for Option E (the Roth–Erev vs Q-learning comparison).
# Values come from the user's Bayesian-optimization sweep — see
# `notebooks/basic_unit_test.ipynb` for the same agent_kwargs.
QLEARN_PARAMS = {
    "exploration_rate":     0.9652628633727897,
    "exploration_decay":    0.9998122815486062,
    "min_exploration_rate": 1e-10,
    "choice":               "ucb",
    "exp_smoothing":        False,
}

# Smoothing windows for the time-series plots.
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

# ---------------------------------------------------------------------------
# Setup (imports + builder + save helper)
# ---------------------------------------------------------------------------

SETUP_MD = """\
## Setup — imports, env builder, save helper

This cell imports the canonical `rl_signaling` API plus the one helper
from the analytics scripts (`enumerate_absorbing_rewards` for Option C),
defines the asymmetric-init env builder, and defines a tiny
`save_and_show(filename)` helper that saves PNGs to `RESULTS_DIR` on
local runs and skips the save on Colab.

The other compute helpers (`build_env`, `run_for_A`, `run_for_B`,
`run_one`) under `analytics/scripts/` were written before this notebook
needed asymmetric initialization; using them here would require signature
changes, so we keep them out of the import list.
"""

SETUP_CODE = '''\
"""Imports, the asymmetric-init env builder, and save_and_show."""

import random
from collections import Counter
from contextlib import contextmanager

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from rl_signaling import MultiAgentEnv, UrnAgent, run_simulation
from rl_signaling.agents import QLearningAgent
from rl_signaling.games import create_random_canonical_game, create_initial_signals
from analytics.scripts.figure_poc_options import enumerate_absorbing_rewards

# Canonical §2.3 game shape.
N_FEATURES = 2
N_SIG = 2
N_ACT = 4

%matplotlib inline
plt.rcParams["figure.dpi"] = 110


@contextmanager
def tqdm_joblib(tqdm_obj):
    """Patch joblib's batch-completion callback to drive a tqdm progress bar.

    Usage:
        with tqdm_joblib(tqdm(desc="...", total=len(tasks))):
            results = Parallel(n_jobs=N_JOBS)(delayed(f)(x) for x in tasks)
    """
    class _Cb(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kw):
            tqdm_obj.update(n=self.batch_size)
            return super().__call__(*args, **kw)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Cb
    try:
        yield tqdm_obj
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_obj.close()


def build_env_from_spec(spec, seed: int,
                        agent_type=UrnAgent, extra_kwargs=None) -> MultiAgentEnv:
    """Build the canonical 2-agent signaling env with independent (n, m) weights
    for the signaling urn (`spec.sig`) and the action urn (`spec.act`).

    `agent_type` defaults to UrnAgent (Roth–Erev); pass QLearningAgent for the
    Q-learning sweep. `extra_kwargs` are merged into `agent_kwargs` — used to
    pass Q-learning-specific parameters (exploration_rate, choice, etc.)."""
    np.random.seed(seed)
    random.seed(seed)

    graph = nx.DiGraph()
    graph.add_nodes_from([0, 1])
    graph.add_edges_from([(0, 1), (1, 0)])
    games = {i: create_random_canonical_game(N_FEATURES, N_ACT) for i in range(2)}

    agent_kwargs = {"initialize": True, "initialization_weights": spec.sig}
    if extra_kwargs:
        agent_kwargs.update(extra_kwargs)

    env = MultiAgentEnv(
        2, N_FEATURES, N_SIG, N_ACT,
        full_information=False, game_dicts=games,
        observed_variables={0: [0], 1: [1]},
        agent_type=agent_type, graph=graph,
        agent_kwargs=agent_kwargs,
    )

    # If action weights differ from signaling weights, overwrite the action table.
    # Attribute name differs by agent: UrnAgent -> action_urns; QLearningAgent -> q_table_action.
    # IMPORTANT: build a fresh table inside the loop so each agent gets its own
    # dict + arrays — sharing would silently couple the two agents' learning.
    if spec.act != spec.sig:
        n_act, m_act = spec.act
        for agent in env.agents:
            new_action_table = create_initial_signals(
                n_observed_features=2,   # 1 own feature + 1 received signal
                n_signals=N_ACT,
                n=n_act,
                m=m_act,
            )
            if hasattr(agent, "action_urns"):
                agent.action_urns = new_action_table
            if hasattr(agent, "q_table_action"):
                agent.q_table_action = new_action_table
                # Reset visit counts so UCB doesn't read stale data.
                if hasattr(agent, "action_counts"):
                    agent.action_counts = {
                        state: np.zeros(N_ACT) for state in new_action_table
                    }

    return env


def save_and_show(filename: str, dpi: int = 150) -> None:
    """Save the current figure to RESULTS_DIR/filename and display inline.

    RESULTS_DIR points at the local results folder on laptop runs and at the
    mounted Drive folder on Colab — either way the artifact is saved."""
    if RESULTS_DIR is not None:
        path = RESULTS_DIR / filename
        plt.savefig(path, dpi=dpi)
        print(f"Saved {path}")
    plt.show()


def save_csv(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to RESULTS_DIR/filename (laptop or Drive, depending on mode)."""
    if RESULTS_DIR is not None:
        path = RESULTS_DIR / filename
        df.to_csv(path, index=False)
        print(f"Saved {path}")


print("Setup complete.")
'''

# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

FIG1_MD = """\
## Figure 1 — Initialization sweep (rewards + NMI)

Four regimes — one Roth–Erev run per regime, `N_EPISODES` episodes each.
Two panels: smoothed reward, smoothed NMI. **In every regime the action
urn is initialized uniformly to `(1, 1)`**; only the signaling urn
varies, and the labels show signaling weights only.

### What the four regimes mean

`init_weights = (n, m)` controls the per-cell pre-seeding of an urn: a
randomly chosen "hot" cell starts with weight `n`, every other cell
starts with weight `m`. Under Roth–Erev's positive-only update
$u[a] \\leftarrow \\max(0, u[a] + r)$, a cell starting at weight 0 can
**never** grow — so any urn initialized with `m = 0` is one-hot
*forever* (the cell pattern is frozen, even if the magnitudes drift).
This is the lever the four signaling regimes pull on:

- **`sig=[1,0]`** (blue) — signaling urns one-hot bijections from
  `t = 0`. Each agent's signal is a deterministic function of its
  observation forever (NMI = 1.0 from the outset). Action urns start
  uniform; the agent has to *learn* what each `(own_obs, received_signal)`
  key should map to.
- **`sig=[1,1]`** (orange) — signaling urns uniform; learning does all
  the work, both for signaling and for actions.
- **`sig=[5,1]`** (green) — signaling urns mildly pre-biased toward an
  arbitrary bijection (5 vs 1 on the hot cell). Actions still uniform.
- **`sig=[100,1]`** (red) — signaling urns strongly pre-biased; actions
  still uniform.

### Why this design is interesting

Every regime asks the *same* question — "starting from a uniform action
policy, how reliably does the joint chain reach high reward?" — under
different amounts of initial signaling pre-coordination. The varying
factor is the signaling channel's head start; the action channel always
starts from scratch.

- The blue (frozen-signaling) regime is the *upper bound*: signals are
  already a perfect deterministic language; the only thing to learn is
  the action mapping. Conditional on a fixed signal, each
  `(own_obs, received_signal)` key's action urn is a single Pólya urn
  with one correct action (reward 1) and three wrong (reward 0); it
  concentrates on the correct action over time.
- Red and green are *intermediate* cases: signaling can still adapt,
  but starts close to a bijection. Whether this *helps* (faster
  convergence) or *hurts* (locking into a bad bijection that the action
  channel then has to compensate for) is the empirical question.
- Orange is the *minimum coordination* case — pure from-scratch
  learning, both signals and actions starting uniform.

### What to look for

- The blue trajectory should rise rapidly to near 1.0 — easiest
  learning problem (only actions update).
- Blue NMI is pinned at 1.0 throughout.
- Green and orange may dissociate: green can end with *higher NMI* but
  *lower reward* than orange (lock-in to a random bijection vs
  co-adaptation to a useful one). See the
  [paper-draft note](../analytics/math/Proof%20of%20Concept%20(Paper%20Draft).md).
"""

FIG1_CODE = '''\
# One trajectory per init.
fig1_histories = {}
for spec in INITS:
    env = build_env_from_spec(spec, seed=0)
    _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
    fig1_histories[spec.label] = (spec, rewards[0], nmi[0])

# Rewards panel.
fig, ax = plt.subplots(figsize=(7, 4.5))
for label, (spec, r, _) in fig1_histories.items():
    smoothed = pd.Series(r).rolling(WINDOW_REWARD, min_periods=1).mean()
    ax.plot(smoothed, color=spec.color,
            label=f"{INIT_DESC[label]} ({label})", lw=1.2)
ax.set_xlabel("Episode")
ax.set_ylabel("Average reward per episode (smoothed)")
ax.set_title("Average reward over time, by initial signaling bias")
ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
plt.tight_layout()
save_and_show("initializations_urn_rewards.png")

# NMI panel.
fig, ax = plt.subplots(figsize=(7, 4.5))
for label, (spec, _, mi) in fig1_histories.items():
    smoothed = pd.Series(mi).rolling(WINDOW_NMI, min_periods=1).mean()
    ax.plot(smoothed, color=spec.color,
            label=f"{INIT_DESC[label]} ({label})", lw=1.2)
ax.set_xlabel("Episode")
ax.set_ylabel("NMI per episode (smoothed)")
ax.set_title("Signal informativeness (NMI) over time, by initial signaling bias")
ax.set_ylim(0, 1.05); ax.legend(loc="lower right")
plt.tight_layout()
save_and_show("initializations_urn_nmi.png")
'''

# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

FIG2_MD = """\
## Figure 2 — Per-seed (NMI, reward) scatter

200 independent seeds per init, `N_EPISODES` episodes each. For each
seed, record the final reward and final NMI (mean over the last 1000
episodes). Each point is one seed; color groups by init.

**Under the new asymmetric (1,0)** the original "(1,0) paradox" framing
(blue cluster pinned at `NMI = 1.0, reward = 0.25`) no longer applies.
The blue cluster should now sit in the high-NMI, high-reward corner —
similar to the red `sig=[100, 1]` cluster but reaching it via a
different mechanism (the signals are predetermined rather than just
strongly biased).

What the scatter shows: the **per-seed spread** within each init
regime. The width of each cluster is a measure of how reliably learning
reaches a high-reward policy from that starting condition. The
prediction (per the paper-draft note) is that the green `sig=[5,1]`
cluster will be visibly *wider* than the orange `sig=[1,1]` cluster
because lock-in to a random bijection produces high seed-to-seed
variance.

**Runtime.** 800 simulations, joblib-parallel. ~2–4 min on a 4-core
laptop with `SMOKE_TEST = False`; faster on Colab if you bump `N_JOBS`.
"""

FIG2_CODE = '''\
RUN_FIG2 = False   # set True to run the 800-sim scatter (~2-4 min on a 4-core laptop)

if not RUN_FIG2:
    print("Skipping Figure 2 (set RUN_FIG2 = True at the top of this cell to run it).")
else:
    def run_one_seed(spec, seed: int) -> dict:
        env = build_env_from_spec(spec, seed)
        _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
        return {
            "label": spec.label,
            "color": spec.color,
            "seed": seed,
            "final_reward": float(np.mean(rewards[0][-1000:])),
            "final_nmi": float(np.mean(nmi[0][-1000:])),
        }

    tasks = [(spec, s) for spec in INITS for s in range(N_SEEDS_FIG2)]
    print(f"Running {len(tasks)} sims ({N_SEEDS_FIG2} seeds × {len(INITS)} inits)...")
    with tqdm_joblib(tqdm(desc="per-seed scatter", total=len(tasks))):
        records = Parallel(n_jobs=N_JOBS)(
            delayed(run_one_seed)(spec, s) for (spec, s) in tasks
        )
    df_fig2 = pd.DataFrame(records)
    save_csv(df_fig2, "figure_init_paradox_scatter.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for spec in INITS:
        sub = df_fig2[df_fig2["label"] == spec.label]
        ax.scatter(sub["final_nmi"], sub["final_reward"], s=14, alpha=0.6,
                   label=f"{INIT_DESC[spec.label]} ({spec.label})", c=spec.color)
    ax.axhline(0.25, ls="--", c="grey", alpha=0.5, label="Random-action baseline (0.25)")
    ax.set_xlabel("Final NMI (averaged over last 1000 episodes)")
    ax.set_ylabel("Final average reward (over last 1000 episodes)")
    ax.set_title(f"Final reward vs final NMI, by initial signaling bias  "
                 f"({N_SEEDS_FIG2} trials per regime)")
    ax.legend(loc="lower right"); ax.set_xlim(-0.05, 1.05); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    save_and_show("figure_init_paradox_scatter.png")
'''

# ---------------------------------------------------------------------------
# Option A
# ---------------------------------------------------------------------------

OPTA_MD = """\
## Option A — Phase-portrait trajectories in (NMI, reward)

**Three panels**, one per non-frozen init regime — `sig=[1,0]` is
dropped because its trajectories sit motionless on the right edge
(NMI = 1.0) and the phase portrait degenerates to vertical motion in
reward. The interesting contrast is among `sig=[1,1]`, `sig=[5,1]`, and
`sig=[100,1]`, which is where the dynamics is doing visible work.

15 seeds per init. Each seed gives an `N_EPISODES`-episode trajectory;
we smooth both reward and NMI with a 500-episode rolling mean, then
plot each trajectory as a series of small dots in the (NMI, reward)
plane, **colored by episode** (viridis: purple = early, yellow = late).
The endpoint is marked with a black `X`.

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
  (the no-signaling failure mode revealed by Figure 2).
- `sig=[5,1]` green: trajectories cover less ground than orange and
  end mostly with higher NMI (~0.5–0.95). Endpoint spread on the
  reward axis reflects lock-in to different bijections.
- `sig=[100,1]` red: tight cluster on the right — trajectories barely
  move (start close to where they end up).

### Wrinkle

At the current resolution the trajectories can look like noisy
scribbles because every dot is a 500-episode-smoothed snapshot. With
15 seeds per panel the density is higher than the 8-seed version but
individual lines are still hard to follow through the cluster. Worth
deciding whether the trajectory *texture* (lots of overlapping paths)
is the right visual emphasis, or whether a smaller seed count with
labeled individual trajectories would communicate better.
"""

OPTA_CODE = '''\
def run_for_A(spec, seed: int) -> dict:
    env = build_env_from_spec(spec, seed)
    _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
    r = pd.Series(rewards[0]).rolling(500, min_periods=1).mean().to_numpy()
    n = pd.Series(nmi[0]).rolling(500, min_periods=1).mean().to_numpy()
    return {"spec": spec, "seed": seed, "reward": r, "nmi": n}

OPTA_SPECS = [s for s in INITS if s.label != "sig=[1,0]"]
tasks_A = [(spec, s) for spec in OPTA_SPECS for s in range(N_SEEDS_OPT_A)]
print(f"Running {len(tasks_A)} sims ({N_SEEDS_OPT_A} seeds × {len(OPTA_SPECS)} inits)...")
with tqdm_joblib(tqdm(desc="phase portrait trajectories", total=len(tasks_A))):
    records_A = Parallel(n_jobs=N_JOBS)(
        delayed(run_for_A)(spec, s) for (spec, s) in tasks_A
    )

fig, axes = plt.subplots(1, len(OPTA_SPECS), figsize=(4.3 * len(OPTA_SPECS), 4),
                         sharex=True, sharey=True)
for ax, spec in zip(axes, OPTA_SPECS):
    for rec in [r for r in records_A if r["spec"].label == spec.label]:
        t = np.linspace(0, 1, len(rec["nmi"]))
        ax.scatter(rec["nmi"], rec["reward"], c=t, cmap="viridis", s=1, alpha=0.4)
        ax.scatter(rec["nmi"][-1], rec["reward"][-1], c="black", s=30,
                   marker="X", zorder=10)
    ax.axhline(0.25, ls="--", c="grey", alpha=0.5)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"{INIT_DESC[spec.label]} ({spec.label})")
    ax.set_xlabel("NMI (smoothed)")
axes[0].set_ylabel("Average reward (smoothed)")
fig.suptitle(
    f"Learning trajectories through NMI × reward space  "
    f"({N_SEEDS_OPT_A} trials per panel; color: early (purple) → late (yellow); X = endpoint)",
    fontsize=12,
)
plt.tight_layout()
save_and_show("poc_optionA_phase_portrait.png")
'''

# ---------------------------------------------------------------------------
# Option B
# ---------------------------------------------------------------------------

OPTB_MD = """\
## Option B — Per-cell hot-fraction concentration

Two inits: `sig=[1,1]` and `sig=[5,1]`. The `sig=[1,0]` regime is
excluded here because its signaling urns are frozen one-hot from
`t = 0`, so the hot fraction is trivially `1.0` for every step and
there's nothing to plot.

Six seeds per init. Every 50 episodes we snapshot the **hot fraction**
of agent 0's signaling row for observation $v_1 = 0$:

$$
\\rho_t \\;=\\;
  \\frac{\\max_\\sigma f^{(0)}_t[0,\\,\\sigma]}
       {\\sum_\\sigma f^{(0)}_t[0,\\,\\sigma]}.
$$

We plot $\\rho_t$ over time, one curve per seed.

### What the picture is doing

It illustrates the **local attractor mechanism** in §2.3, focused on a
single cell of a single agent's signaling table.

Why does this cell concentrate? Roth–Erev's update for a signaling row
acts like a Pólya urn:

1. When the agent observes $v_1 = 0$, it samples a signal in proportion
   to the current weights of this row.
2. The signal is sent; the partner decodes and acts; the joint draw
   yields a binary reward $r \\in \\{0, 1\\}$.
3. The *sampled* cell of *this row* gets incremented by $r$. All other
   cells of this row are unchanged.

A higher-weight signal is more likely to be sampled, more likely to
collect a positive reinforcement, and thus more likely to grow further.
$\\rho_t$ is a sub-martingale that converges almost surely to 1 — the
row eventually becomes one-hot. **Which** signal wins is random, picked
out by initial bias and path.

### What to look for

- Under `sig=[1,1]`, $\\rho_t$ starts at 0.5 and drifts upward for every
  seed, but each seed reaches a *different* asymptote (some at 0.97,
  some at 0.55). The Pure-Pólya theorem says these asymptotes are
  samples from a Dirichlet distribution.
- Under `sig=[5,1]`, $\\rho_t$ starts at $\\approx 0.83$ and reaches its
  asymptote much faster.

### Why this matters for §2.3

This is the *positive* result behind "the ideal strategies are
attractors." Each individual cell of each individual signaling row
**provably concentrates**. The remaining question — the one §2.3 does
*not* settle as a theorem — is whether the *joint* chain concentrates
on an *ideal* policy rather than on a trap.

### Caveat

The most technical of the candidates. It assumes the reader knows what
"agent 0's signaling row 0" is. Probably best suited to the analytics
companion, not the philosophy paper — but worth seeing the live picture
before deciding.
"""

OPTB_CODE = '''\
# Only the non-frozen signaling regimes are meaningful here (sig=[1,0] is
# frozen so its hot fraction is trivially 1.0 throughout).
B_SPECS = [s for s in INITS if s.label in ("sig=[1,1]", "sig=[5,1]")]

def run_for_B(spec, seed: int) -> dict:
    """Step the env manually so we can snapshot agent 0's signaling row 0."""
    env = build_env_from_spec(spec, seed)
    snapshots = []
    for episode in range(N_EPISODES):
        _, observations = env.reset()
        signals, new_observations = env.step_signal(observations)
        actions = env.step_action(new_observations)
        rewards = env.reward(actions)
        env.update(observations, signals, new_observations, actions, rewards)
        if episode % 50 == 0:
            urn = env.agents[0].signaling_urns[(0,)]
            total = float(urn.sum())
            hot_frac = float(urn.max() / total) if total > 0 else 0.5
            snapshots.append((episode, hot_frac))
    eps = np.array([s[0] for s in snapshots])
    rho = np.array([s[1] for s in snapshots])
    return {"spec": spec, "seed": seed, "episodes": eps, "rho": rho}

tasks_B = [(spec, s) for spec in B_SPECS for s in range(N_SEEDS_OPT_B)]
print(f"Running {len(tasks_B)} sims ({N_SEEDS_OPT_B} seeds × {len(B_SPECS)} inits)...")
with tqdm_joblib(tqdm(desc="per-cell concentration", total=len(tasks_B))):
    records_B = Parallel(n_jobs=N_JOBS)(
        delayed(run_for_B)(spec, s) for (spec, s) in tasks_B
    )

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, spec in zip(axes, B_SPECS):
    subs = [r for r in records_B if r["spec"].label == spec.label]
    for rec in subs:
        ax.plot(rec["episodes"], rec["rho"], alpha=0.7, lw=1.2)
    ax.axhline(0.5, ls=":", c="grey", alpha=0.5, label="Uniform (any signal equally likely)")
    ax.axhline(1.0, ls="--", c="black", alpha=0.3, label="Deterministic (always same signal)")
    ax.set_xlabel("Episode")
    ax.set_title(f"{INIT_DESC[spec.label]} ({spec.label})")
    ax.set_ylim(0.4, 1.05)
axes[0].set_ylabel("Probability of agent's modal signal\\n(for one fixed observation, one agent)")
axes[0].legend(loc="lower right", fontsize=9)
fig.suptitle(
    f"Concentration of a single signaling probability over time  "
    f"({N_SEEDS_OPT_B} trials per panel)",
    fontsize=12,
)
plt.tight_layout()
save_and_show("poc_optionB_cell_concentration.png")
'''

# ---------------------------------------------------------------------------
# Option C
# ---------------------------------------------------------------------------

OPTC_MD = """\
## Option C — Reward distribution over the 2304 absorbing states

### Setup

**Enumeration, not simulation.** For a fixed game seed (here, seed 0),
compute the per-agent mean reward of every deterministic joint policy.

There are exactly $48 \\times 48 = 2304$ such profiles. Each agent has
$2! = 2$ signaling bijections and $4! = 24$ action bijections, so
$48 = 2 \\times 24$ deterministic per-agent policies; the joint space
is $48^2 = 2304$. For each profile, we compute the mean reward over the
four world states $(v_1, v_2) \\in \\{0, 1\\}^2$ — which gives values in
$\\{0, 0.25, 0.5, 0.75, 1.0\\}$.

Two panels:

- **Left**: marginal distribution of one agent's mean reward.
- **Right**: joint $(r_0, r_1)$ distribution as a count heatmap.

### Why this still matters for §2.3 (even without the old (1,0) regime)

Originally this figure was framed as the *structural* explanation for
the (symmetric) `(1, 0)` regime's empirical reward of 0.25: that regime
froze the chain at a uniformly random deterministic policy, and 0.25 is
the mean of this distribution.

Under the **new asymmetric** `sig=[1,0]` regime, that explanation no
longer applies as directly — the signaling tables are still frozen at
a random absorbing bijection, but the action urns are *not* frozen and
the agents *do* learn over time. The structural picture below still
shows the space of *fully deterministic* joint policies, but its
connection to the new Figure 1 blue line is now indirect.

It remains a useful figure for the *general* §2.3 claim that
distributed-reward absorbing states are bottom-heavy: most absorbing
policies give low reward; only 4 out of 2304 are ideal. That's the
counterweight to the "every cell concentrates" Pólya story — the
question is whether the joint dynamics concentrates on a *good*
absorbing state.

Key counts under game seed 0:
- **4** ideal states (both agents at mean reward 1.0).
- **324** joint traps (both at 0.0).
- Mean per-agent reward across all profiles: exactly $1/N_\\text{act} = 0.25$.

### Cost

~5 seconds, no simulations.
"""

OPTC_CODE = '''\
r0, r1 = enumerate_absorbing_rewards(seed=GAME_SEED_OPT_C)

n_total = len(r0)
n_ideal = int(np.sum((r0 == 1.0) & (r1 == 1.0)))
n_trap  = int(np.sum((r0 == 0.0) & (r1 == 0.0)))
print(f"Total absorbing states: {n_total}")
print(f"  ideal (r0 = r1 = 1.0): {n_ideal}")
print(f"  trap  (r0 = r1 = 0.0): {n_trap}")
print(f"  mean per-agent reward: {r0.mean():.4f}  (expected 1/{N_ACT} = {1/N_ACT:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Marginal.
ax = axes[0]
bin_edges = np.array([0, 0.125, 0.375, 0.625, 0.875, 1.05])
labels = ["0.00", "0.25", "0.50", "0.75", "1.00"]
counts, _ = np.histogram(r0, bins=bin_edges)
colors = ["#b9504e", "#c08e6b", "#c7b48f", "#a7b878", "#3a8a3a"]
bars = ax.bar(labels, counts, color=colors, edgecolor="white")
for bar, c in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, c + 20, f"{c}",
            ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Average reward over the 4 world states (per agent)")
ax.set_ylabel(f"Number of joint policies (out of {n_total})")
ax.set_title("One agent's reward distribution")

# Joint heatmap.
ax2 = axes[1]
hist2d, _, _ = np.histogram2d(r0, r1, bins=[bin_edges, bin_edges])
im = ax2.imshow(hist2d.T, origin="lower", cmap="magma_r",
                extent=[0, 5, 0, 5], aspect="auto")
for i in range(len(labels)):
    for j in range(len(labels)):
        v = int(hist2d[i, j])
        ax2.text(i + 0.5, j + 0.5, f"{v}", ha="center", va="center",
                 color="white" if v > 200 else "black", fontsize=9)
ax2.set_xticks(np.arange(len(labels)) + 0.5); ax2.set_xticklabels(labels)
ax2.set_yticks(np.arange(len(labels)) + 0.5); ax2.set_yticklabels(labels)
ax2.set_xlabel("Agent 1's average reward")
ax2.set_ylabel("Agent 2's average reward")
ax2.set_title("Joint reward distribution")
plt.colorbar(im, ax=ax2, label="Number of joint policies")

fig.suptitle(
    f"Reward distribution over the {n_total} deterministic joint policies  "
    f"(game seed {GAME_SEED_OPT_C}: {n_ideal} ideal, {n_trap} traps; mean per agent = {r0.mean():.2f})",
    fontsize=12,
)
plt.tight_layout()
save_and_show("poc_optionC_absorbing_distribution.png")
'''

# ---------------------------------------------------------------------------
# Option D — Basin of attraction sweep (α + β)
# ---------------------------------------------------------------------------

BASIN_MD = """\
## Option D — Basin of attraction (continuous signaling-bias sweep)

This section sweeps the signaling pre-bias parameter `n_sig` continuously
(holding `m_sig = 1` and `act = (1, 1)` fixed) to see how the basin of
attraction of high-reward joint policies depends on the amount of
initial signaling coordination.

Two views of the same data:

- **D-α (histograms)** — for each `sig_n` value, the histogram of final
  reward across many seeds. Makes the *bimodal* structure visible: some
  seeds reach the high-reward attractor, some fall into the no-signaling
  trap (reward ≈ 0.5). The transition from bimodal to unimodal as
  `sig_n` grows *is* the basin filling in.
- **D-β (mean ± std curves)** — the same data on a continuous axis:
  x-axis is `sig_n` on a log scale, y-axis is the final value. Red
  curve = mean reward across trials, shaded band = mean ± 1 std
  (clipped to `[0, 1]`). Green curve / band = same for NMI. Shows
  basin coverage as smooth summaries across the sweep, with reward
  and NMI overlaid so the dissociation between them is visible.

Together they answer: "how much signaling pre-bias is needed for the
joint chain to reach high reward reliably?"

The compute cell runs the full sweep once; the two plot cells then
render α and β from the same DataFrame, so you can re-run either plot
cheaply.

**Runtime.** `len(BASIN_SIG_N_VALUES) × BASIN_N_SEEDS` simulations,
parallelized. Default: 9 × 50 = 450 sims, roughly 2 min on a 4-core
laptop. Bump `BASIN_N_SEEDS` (in the Parameters cell) on Colab if you
want tighter histograms.
"""

BASIN_COMPUTE_CODE = '''\
"""Run the basin sweep: for each sig_n value, BASIN_N_SEEDS seeds."""

def run_basin_seed(sig_n, sig_m, seed):
    spec = InitSpec(label=f"sig=[{sig_n},{sig_m}]", sig=(sig_n, sig_m),
                    act=(1, 1), color="tab:gray")
    env = build_env_from_spec(spec, seed)
    _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
    return {
        "sig_n": sig_n,
        "sig_m": sig_m,
        "seed": seed,
        "final_reward": float(np.mean(rewards[0][-1000:])),
        "final_nmi":    float(np.mean(nmi[0][-1000:])),
    }

tasks_D = [(n, 1, s) for n in BASIN_SIG_N_VALUES for s in range(BASIN_N_SEEDS)]
print(f"Running {len(tasks_D)} sims ({BASIN_N_SEEDS} seeds × {len(BASIN_SIG_N_VALUES)} sig_n values)...")
with tqdm_joblib(tqdm(desc="basin sweep (Roth-Erev)", total=len(tasks_D))):
    records_D = Parallel(n_jobs=N_JOBS)(
        delayed(run_basin_seed)(n, m, s) for (n, m, s) in tasks_D
    )
df_basin = pd.DataFrame(records_D)
save_csv(df_basin, "basin_sweep_data.csv")
print(f"Collected {len(df_basin)} records over sig_n = {BASIN_SIG_N_VALUES}")
'''

BASIN_ALPHA_CODE = '''\
"""Option D-α — final-reward histograms across signaling pre-bias."""

n_vals = sorted(df_basin["sig_n"].unique())
fig, axes = plt.subplots(1, len(n_vals), figsize=(2.0 * len(n_vals), 3.2),
                         sharey=True, sharex=True)
if len(n_vals) == 1:
    axes = [axes]

bins = np.linspace(0, 1, 21)
for ax, n in zip(axes, n_vals):
    sub = df_basin[df_basin["sig_n"] == n]
    ax.hist(sub["final_reward"], bins=bins, color="steelblue", edgecolor="white")
    ax.axvline(sub["final_reward"].median(), color="firebrick", ls="--", lw=1.2,
               label=f"median = {sub['final_reward'].median():.2f}")
    ax.axvline(0.5, color="grey", ls=":", lw=1, alpha=0.5)
    ax.set_title(f"Bias = {n}")
    ax.set_xlim(0, 1.02)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Final average reward")
axes[0].set_ylabel(f"Number of trials (out of {BASIN_N_SEEDS})")
fig.suptitle(f"Final reward distribution by initial signaling bias  "
             f"({BASIN_N_SEEDS} trials per panel; grey dotted line = no-signaling baseline at 0.5)",
             fontsize=12)
plt.tight_layout()
save_and_show("basin_alpha_reward_histograms.png")
'''

BASIN_BETA_CODE = '''\
"""Option D-β — per-seed scatter (reward) plus reward and NMI summary curves."""

fig, ax = plt.subplots(figsize=(7.5, 4.8))

# Reward: mean ± std overlay (std band clipped to [0, 1]).
g_r = df_basin.groupby("sig_n")["final_reward"]
mean_r, std_r = g_r.mean(), g_r.std()
ax.plot(mean_r.index, mean_r.values,
        color="firebrick", lw=2, marker="o", label="Reward: mean")
ax.fill_between(mean_r.index,
                np.clip(mean_r.values - std_r.values, 0, 1),
                np.clip(mean_r.values + std_r.values, 0, 1),
                color="firebrick", alpha=0.18, label="Reward: mean ± std")

# NMI: mean ± std overlay (std band clipped to [0, 1]).
g_n = df_basin.groupby("sig_n")["final_nmi"]
mean_n, std_n = g_n.mean(), g_n.std()
ax.plot(mean_n.index, mean_n.values,
        color="darkgreen", lw=2, marker="s", label="NMI: mean")
ax.fill_between(mean_n.index,
                np.clip(mean_n.values - std_n.values, 0, 1),
                np.clip(mean_n.values + std_n.values, 0, 1),
                color="darkgreen", alpha=0.15, label="NMI: mean ± std")

ax.axhline(0.5, ls=":", c="grey", alpha=0.7,
           label="No-signaling reward baseline (≈ 0.5)")
ax.set_xscale("log")
ax.set_xlabel("Initial signaling bias (log scale)")
ax.set_ylabel("Final value (averaged over last 1000 episodes)")
ax.set_title(f"Final reward and NMI vs initial signaling bias  "
             f"({BASIN_N_SEEDS} trials per value)")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=8, ncol=2)
plt.tight_layout()
save_and_show("basin_beta_scatter.png")
'''

# ---------------------------------------------------------------------------
# Option D-γ — 2D heatmap over (sig_n, act_n)
# ---------------------------------------------------------------------------

BASIN_GAMMA_MD = """\
## Option D-γ — 2D basin heatmap over `(sig_n, act_n)`

The α and β plots fixed the action urn at uniform `(1, 1)` and swept
only the signaling pre-bias. γ relaxes that choice: both `sig_n` and
`act_n` are swept jointly on a grid, with `m_sig = m_act = 1` fixed.

For each cell `(sig_n, act_n)` we run `GRID_N_SEEDS` seeds and compute
$\\mathbb{P}(\\text{final reward} > \\text{REWARD\\_THRESHOLD})$ — that
probability is the cell's color in the heatmap.

### What γ answers that α and β don't

α and β are a *slice* of this 2D space along `act_n = 1`. γ is the
full picture, and it lets us ask:

- Does **action pre-bias matter at all**, or is signaling pre-bias the
  only thing that determines basin reach?
- Is the basin a **simple monotone region** (more bias in either channel
  → more reliable success), or does it have **structure** (some mixes
  of sig and act bias work better than others)?
- Is there a **diagonal trade-off** (e.g., low signaling bias
  compensated by high action bias)?

If γ is roughly constant across the `act_n` axis, the act=(1,1) choice
we made elsewhere in this notebook is *vindicated*: action pre-bias is
inert and we can ignore it.

If γ varies across `act_n`, the choice is *load-bearing* and the §2.3
narrative needs to acknowledge it.

### Runtime

`len(GRID_SIG_N_VALUES) × len(GRID_ACT_N_VALUES) × GRID_N_SEEDS`
simulations.

| Mode | Grid | Seeds/cell | Total sims | Estimated wall-clock |
|---|---|---:|---:|---|
| Smoke | 3 × 3 | 5 | 45 | ~30 sec |
| Local (`RUNNING_LOCALLY = True`) | 5 × 5 | 20 | 500 | ~5–10 min on 4 cores |
| Colab (`RUNNING_LOCALLY = False`) | 9 × 9 | 50 | 4 050 | ~60–80 min (empirically observed at ~1 s/task wall-clock under `n_jobs=-1` on a standard Colab box) |
"""

BASIN_GAMMA_COMPUTE_CODE = '''\
"""Option D-γ — compute the (sig_n, act_n) grid."""

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
'''

BASIN_GAMMA_PLOT_CODE = '''\
"""Option D-γ — render the 2D success-rate heatmap."""

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

# ---------------------------------------------------------------------------
# Option E — Roth–Erev vs Q-learning side-by-side basin comparison
# ---------------------------------------------------------------------------

OPTION_E_MD = """\
## Option E — Roth–Erev vs Q-learning: side-by-side basin comparison

The full Option D analysis runs the basin sweep with `UrnAgent` (Roth–Erev).
This section repeats the **D-β** plot for `QLearningAgent` and shows the two
on shared axes, side-by-side. The question is whether Q-learning's basin is
visibly wider than Roth–Erev's — a likely empirical answer to Reviewer 2's
"why does Q-learning outperform?" question.

### Setup

- **Same `sig_n` grid** as Option D-β: `BASIN_SIG_N_VALUES`.
- **Same `act = (1, 1)` invariant** as Option D-β.
- **Same `BASIN_N_SEEDS` per `sig_n` value.**
- **Q-learning parameters** are taken from the user's earlier Bayesian
  optimization (`QLEARN_PARAMS` in the Parameters cell): UCB choice rule,
  initial exploration ≈ 0.965, decay ≈ 0.9998, floor ≈ 1e-10, no
  exponential smoothing.

### Prediction

Q-learning's exploration bonus (UCB) drives the agent to try untried
signals/actions regardless of where the Q-table started. So even at
`sig_n = 1` (uniform signaling), Q-learning should reach high final
reward, and the spread across seeds should be tight.

In contrast Roth–Erev (left panel) has the lock-in / no-signaling failure
modes we already characterized: low `sig_n` → wide spread, sometimes
reward ≈ 0.5.

If that prediction holds, the side-by-side picture *visually demonstrates*
the §2.3 robustness gap: Roth–Erev's basin reach depends on initial
coordination; Q-learning's doesn't, because exploration substitutes for
coordination.

### Runtime

`len(BASIN_SIG_N_VALUES) × BASIN_N_SEEDS` Q-learning sims — same cost as
Option D-β. The compute cell runs the Q-learning sweep; the plot cell
overlays it next to the existing `df_basin` from Option D-β. **Run
Option D-β's compute cell first**, otherwise `df_basin` will not exist.
"""

OPTION_E_QL_COMPUTE_CODE = '''\
"""Option E — Q-learning basin sweep. Same sig_n grid and seed count as D-β."""

def run_basin_seed_ql(sig_n, sig_m, seed):
    spec = InitSpec(label=f"sig=[{sig_n},{sig_m}]", sig=(sig_n, sig_m),
                    act=(1, 1), color="tab:gray")
    env = build_env_from_spec(
        spec, seed,
        agent_type=QLearningAgent,
        extra_kwargs=QLEARN_PARAMS,
    )
    _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
    return {
        "sig_n": sig_n,
        "sig_m": sig_m,
        "seed": seed,
        "final_reward": float(np.mean(rewards[0][-1000:])),
        "final_nmi":    float(np.mean(nmi[0][-1000:])),
    }

tasks_E = [(n, 1, s) for n in BASIN_SIG_N_VALUES for s in range(BASIN_N_SEEDS)]
print(f"Running {len(tasks_E)} Q-learning sims "
      f"({BASIN_N_SEEDS} seeds × {len(BASIN_SIG_N_VALUES)} sig_n values)...")
with tqdm_joblib(tqdm(desc="basin sweep (Q-learning)", total=len(tasks_E))):
    records_E = Parallel(n_jobs=N_JOBS)(
        delayed(run_basin_seed_ql)(n, m, s) for (n, m, s) in tasks_E
    )
df_basin_ql = pd.DataFrame(records_E)
save_csv(df_basin_ql, "basin_sweep_data_ql.csv")
print(f"Collected {len(df_basin_ql)} Q-learning records over sig_n = {BASIN_SIG_N_VALUES}")
'''

OPTION_E_PLOT_CODE = '''\
"""Option E — render the side-by-side D-β-style plot for both agents."""

def render_basin_panel(ax, df, title):
    g_r = df.groupby("sig_n")["final_reward"]
    mean_r, std_r = g_r.mean(), g_r.std()
    ax.plot(mean_r.index, mean_r.values,
            color="firebrick", lw=2, marker="o", label="Reward: mean")
    ax.fill_between(mean_r.index,
                    np.clip(mean_r.values - std_r.values, 0, 1),
                    np.clip(mean_r.values + std_r.values, 0, 1),
                    color="firebrick", alpha=0.18, label="Reward: mean ± std")

    g_n = df.groupby("sig_n")["final_nmi"]
    mean_n, std_n = g_n.mean(), g_n.std()
    ax.plot(mean_n.index, mean_n.values,
            color="darkgreen", lw=2, marker="s", label="NMI: mean")
    ax.fill_between(mean_n.index,
                    np.clip(mean_n.values - std_n.values, 0, 1),
                    np.clip(mean_n.values + std_n.values, 0, 1),
                    color="darkgreen", alpha=0.15, label="NMI: mean ± std")

    ax.axhline(0.5, ls=":", c="grey", alpha=0.7,
               label="No-signaling reward baseline (≈ 0.5)")
    ax.set_xscale("log")
    ax.set_xlabel("Initial signaling bias (log scale)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
render_basin_panel(axes[0], df_basin,    "Roth–Erev")
render_basin_panel(axes[1], df_basin_ql, "Q-learning")
axes[0].set_ylabel("Final value (averaged over last 1000 episodes)")
axes[0].legend(loc="lower right", fontsize=8, ncol=2)
fig.suptitle(f"Final reward and NMI vs initial signaling bias: Roth–Erev vs Q-learning  "
             f"({BASIN_N_SEEDS} trials per value; identical setup except learning rule)",
             fontsize=12)
plt.tight_layout()
save_and_show("basin_e_comparison.png")
'''

# ---------------------------------------------------------------------------
# Option F — Time-horizon sweep: when does initial bias matter?
# ---------------------------------------------------------------------------

OPTION_F_MD = """\
## Option F — Time-horizon sweep: when does initial bias matter?

Option E's side-by-side plot averages reward and NMI over the **last 1000
of 10,000 episodes** — a deep-asymptotic measurement. The Roth–Erev panel
shows a clean basin-reach curve; the Q-learning panel is essentially flat.
A natural question is whether the flatness reflects a structural property
of the learning rule or an artifact of looking only at the long-horizon
limit.

### Mechanism (why we expect a difference)

- **Roth–Erev** updates urns *additively*: `urn[chosen] += reward`. Counts
  only grow. An initial bias of `[n, m]` persists for roughly `n + m`
  reinforcements before the additions overwhelm it — a *linear-time*
  decay.
- **Q-learning** updates Q-values *contractively*: with constant learning
  rate `α = 0.1`, `Q[chosen] ← 0.9 · Q[chosen] + 0.1 · reward`. The
  initial Q-value decays exponentially toward an EMA of reward, with time
  constant `τ ≈ -1 / log(0.9) ≈ 9.5` episodes per state-action visit.
  Whatever the magnitude of the initial bias, it falls below the reward
  scale within ~50-100 episodes (empirically verified — see the session
  `worklog.jsonl` entry for the decay table).

So Roth–Erev's initial bias survives for thousands of episodes; Q-learning's
survives for ~100. If we look at *both* agents at horizon 10,000, Q-learning
has long forgotten its initialization and the right panel reads as flat.

### Prediction

For each agent, plot the final-reward / final-NMI curve at horizons
`H ∈ {100, 300, 1000, 3000, 10000}`. Then:

- **Roth–Erev** curves should be *shape-stable* across horizons: the basin
  reach is monotone in the initial bias, and that gradient persists from
  short to long horizons (modulo overall slope changes as the chain
  approaches absorption).
- **Q-learning** curves should *flatten* with horizon: visible bias
  effect at `H = 100`, partial wash-out at `H = 300`, mostly flat at
  `H = 1000`, fully flat at `H = 10000`.

If that prediction holds, Option F is the cleanest visual evidence that
Q-learning's apparent robustness in Option E is a **consequence of
horizon choice** — the structural difference is real (contractive vs
additive updates), but framing it as "Q-learning always succeeds" hides
the mechanism. The §2.3 narrative anchors cleanly on Roth–Erev alone;
Q-learning's behavior is then a §4 story about update structure.

### Runtime

`len(BASIN_SIG_N_VALUES) × BASIN_N_SEEDS × 2` sims — twice Option E's
cost (Roth–Erev + Q-learning, run fresh so this cell is standalone). On
Colab at `BASIN_N_SEEDS = 200`, expect ~60 minutes wall-clock (at the
empirically observed ~1 s/task Colab rate). To shrink, lower
`BASIN_N_SEEDS` in the Parameters cell or set `SMOKE_TEST = True`.
"""

OPTION_F_COMPUTE_CODE = '''\
"""Option F — time-horizon sweep: run sims fresh and snapshot reward + NMI
at multiple horizons. Standalone (does not depend on df_basin / df_basin_ql)."""

HORIZON_VALUES = [100, 300, 1000, 3000, 10_000]
# Window for averaging at each horizon: 100 episodes when horizon allows,
# otherwise horizon // 10. Matches the Option E convention (last-1000 of 10000).
HORIZON_WINDOWS = {h: max(10, min(1000, h // 10)) for h in HORIZON_VALUES}

assert max(HORIZON_VALUES) <= N_EPISODES, (
    f"Max horizon {max(HORIZON_VALUES)} exceeds N_EPISODES {N_EPISODES}. "
    "Increase N_EPISODES in the Parameters cell or shorten HORIZON_VALUES."
)


def run_horizon_seed(sig_n, sig_m, seed, agent_type, agent_kwargs):
    spec = InitSpec(label=f"sig=[{sig_n},{sig_m}]", sig=(sig_n, sig_m),
                    act=(1, 1), color="tab:gray")
    env = build_env_from_spec(
        spec, seed,
        agent_type=agent_type,
        extra_kwargs=agent_kwargs,
    )
    _, rewards, nmi, _, _ = run_simulation(env, N_EPISODES, with_signals=True, plot=False)
    # rewards / nmi are lists of arrays, one per agent. Average across the
    # two agents at each horizon to match the Option D / E single-agent
    # convention used in their dataframes (rewards[0]).
    records = []
    for H in HORIZON_VALUES:
        W = HORIZON_WINDOWS[H]
        records.append({
            "agent": agent_type.__name__,
            "sig_n": sig_n,
            "sig_m": sig_m,
            "seed": seed,
            "horizon": H,
            "window": W,
            "final_reward": float(np.mean(rewards[0][H - W : H])),
            "final_nmi":    float(np.mean(nmi[0][H - W : H])),
        })
    return records


tasks_F = []
for n in BASIN_SIG_N_VALUES:
    for s in range(BASIN_N_SEEDS):
        tasks_F.append((n, 1, s, UrnAgent, None))
        tasks_F.append((n, 1, s, QLearningAgent, QLEARN_PARAMS))

print(f"Running {len(tasks_F)} time-horizon sims "
      f"({BASIN_N_SEEDS} seeds × {len(BASIN_SIG_N_VALUES)} sig_n × 2 agents)...")
with tqdm_joblib(tqdm(desc="time-horizon sweep (both agents)", total=len(tasks_F))):
    records_F_nested = Parallel(n_jobs=N_JOBS)(
        delayed(run_horizon_seed)(n, m, s, agent_type, extra)
        for (n, m, s, agent_type, extra) in tasks_F
    )
records_F = [rec for sublist in records_F_nested for rec in sublist]
df_horizon = pd.DataFrame(records_F)
save_csv(df_horizon, "horizon_sweep_data.csv")
print(f"Collected {len(df_horizon)} records over "
      f"sig_n = {BASIN_SIG_N_VALUES}, horizons = {HORIZON_VALUES}")
'''

OPTION_F_PLOT_CODE = '''\
"""Option F — 2×2 grid: rows = {reward, NMI}, cols = {Roth–Erev, Q-learning}.
Each panel shows one curve per horizon, color-coded with viridis."""

import matplotlib as mpl

agents = ["UrnAgent", "QLearningAgent"]
agent_titles = {"UrnAgent": "Roth–Erev", "QLearningAgent": "Q-learning"}
metrics = [("final_reward", "Final reward", "firebrick"),
           ("final_nmi",    "Final NMI",    "darkgreen")]

# Color each horizon with viridis from short (dark) to long (bright).
cmap = mpl.colormaps["viridis"]
horizon_colors = {H: cmap(i / (len(HORIZON_VALUES) - 1))
                  for i, H in enumerate(HORIZON_VALUES)}

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey="row")

for row, (metric_col, metric_label, _) in enumerate(metrics):
    for col, agent in enumerate(agents):
        ax = axes[row, col]
        df_sub = df_horizon[df_horizon["agent"] == agent]
        for H in HORIZON_VALUES:
            df_H = df_sub[df_sub["horizon"] == H]
            g = df_H.groupby("sig_n")[metric_col]
            mean = g.mean()
            ax.plot(mean.index, mean.values,
                    color=horizon_colors[H], lw=2, marker="o",
                    label=f"{H:,} episodes")
        ax.axhline(0.5, ls=":", c="grey", alpha=0.6)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        if row == 0:
            ax.set_title(agent_titles[agent])
        if col == 0:
            ax.set_ylabel(metric_label)
        if row == 1:
            ax.set_xlabel("Initial signaling bias (log scale)")

axes[0, 1].legend(title="Horizon (episodes)",
                  loc="lower right", fontsize=8, framealpha=0.9)
fig.suptitle(
    f"Final reward and NMI vs initial signaling bias, by horizon  "
    f"(Roth–Erev vs Q-learning; {BASIN_N_SEEDS} trials per value)",
    fontsize=12,
)
plt.tight_layout()
save_and_show("horizon_sweep_comparison.png")
'''

# ---------------------------------------------------------------------------
# Disconnect (Colab only)
# ---------------------------------------------------------------------------

DISCONNECT_MD = """\
## Disconnect Colab runtime

On Colab, the kernel keeps the runtime billed (or against quota) until
explicitly disconnected. The cell below disconnects automatically after
all figures are rendered. On local it just prints a message and exits.
"""

DISCONNECT_CODE = '''\
"""Disconnect Colab runtime — Colab only."""

from datetime import datetime

stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if not RUNNING_LOCALLY:
    from IPython.display import Javascript, display
    print(f"Run finished at {stamp} — disconnecting Colab runtime.")
    display(Javascript("google.colab.kernel.disconnect()"))
else:
    print(f"Run finished at {stamp}. Local mode — nothing to disconnect.")
'''


cells = [
    md("title", TITLE),
    md("env-header", ENV_MD),
    code("env-switch", ENV_CODE),
    code("env-clone", CLONE_CODE),
    code("env-pip", PIP_CODE),
    md("params-header", PARAMS_MD),
    code("params", PARAMS_CODE),
    md("setup-header", SETUP_MD),
    code("setup", SETUP_CODE),
    md("fig1-md", FIG1_MD),
    code_timed("fig1-code", FIG1_CODE),
    md("fig2-md", FIG2_MD),
    code_timed("fig2-code", FIG2_CODE),
    md("optA-md", OPTA_MD),
    code_timed("optA-code", OPTA_CODE),
    md("optB-md", OPTB_MD),
    code_timed("optB-code", OPTB_CODE),
    md("optC-md", OPTC_MD),
    code_timed("optC-code", OPTC_CODE),
    md("basin-md", BASIN_MD),
    code_timed("basin-compute", BASIN_COMPUTE_CODE),
    code("basin-alpha", BASIN_ALPHA_CODE),
    code("basin-beta", BASIN_BETA_CODE),
    md("basin-gamma-md", BASIN_GAMMA_MD),
    code_timed("basin-gamma-compute", BASIN_GAMMA_COMPUTE_CODE),
    code("basin-gamma-plot", BASIN_GAMMA_PLOT_CODE),
    md("option-e-md", OPTION_E_MD),
    code_timed("option-e-ql-compute", OPTION_E_QL_COMPUTE_CODE),
    code("option-e-plot", OPTION_E_PLOT_CODE),
    md("option-f-md", OPTION_F_MD),
    code_timed("option-f-compute", OPTION_F_COMPUTE_CODE),
    code("option-f-plot", OPTION_F_PLOT_CODE),
    md("disconnect-md", DISCONNECT_MD),
    code("disconnect-code", DISCONNECT_CODE),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (rl_signaling)",
            "language": "python",
            "name": "rl_signaling",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

repo_root = Path("/Users/ignacio/Documents/VS Code/GitHub Repositories/RL_Signaling")
target = repo_root / "notebooks" / "proof_of_concept_figures.ipynb"
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Wrote {target}  ({len(cells)} cells)")
