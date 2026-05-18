# Notebooks

This folder holds the experiment and reporting notebooks for the `rl_signaling` study. They orchestrate runs against the canonical `MultiAgentEnv` + `run_simulation` API defined in [`rl_signaling/`](../rl_signaling/); none of the heavy logic lives in the notebooks themselves.

The expected execution order is encoded by the filename prefix:

| Notebook | Purpose |
|---|---|
| `01_basic_unit_test.ipynb` | Sanity check for each of the three agent types on the canonical 2-feature game. |
| `02_initializations_test.ipynb` | Effect of urn / Q-table initialization strategies on convergence. |
| `03_run_simulations.ipynb` | Main runs: canonical and complex models across UrnAgent, QLearningAgent, TDLearningAgent. |
| `04_parameter_optimization.ipynb` | Bayesian hyperparameter sweep for Q-learning and TD-learning. |
| `05_costly_signaling_simulations.ipynb` | Q-learning costly-signaling experiments. |
| `06_plotting_results.ipynb` | Build all final figures from the CSVs under [`../results/`](../results/). |

03 and 05 write the CSVs in `../results/`; 06 reads them. 04 produces the hyperparameters that 03 and 05 hard-code.

## Conventions

All notebooks in this folder must:

1. Run **Restart-and-Run-All** on a fresh `rl_signaling` kernel without errors.
2. Use **only** the canonical API (`from rl_signaling import MultiAgentEnv, run_simulation, ...`). Legacy classes (`NetMultiAgentEnv`, `TempNetMultiAgentEnv`) and the legacy runners (`simulation_function`, `temp_simulation_function`) are deprecated and must not appear here.
3. Save with `nbformat=4.5` (stable cell IDs) and kernel `rl_signaling`.
4. Keep every tunable knob in a single Parameters cell near the top. No magic numbers later.
5. For any notebook that runs > 5 minutes end-to-end, expose `SMOKE_TEST` parameters so a developer iterating doesn't have to wait for the full sweep.

The full conventions reference is the knowledge-base document `content/how-to/NOTEBOOK_WRITING_SKILL.md` — read it before authoring a new notebook in this folder.

## Local run

```bash
# from the repo root
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
python -m ipykernel install --user --name rl_signaling --display-name "Python (rl_signaling)"
```

Then open any notebook in VS Code / Jupyter and select the **Python (rl_signaling)** kernel. The notebooks default to `RUNNING_LOCALLY = True` and read/write to `../results/` (relative to the notebook).

## Colab run

Set `RUNNING_LOCALLY = False` at the top of the notebook. The setup cells will:

1. Wipe and force-fresh-clone the repo.
2. `os.chdir(...)` into the clone and put it on `sys.path`.
3. `subprocess.run(['pip', 'install', '-q', '-e', '.'])` inside the clone.
4. Mount Google Drive and derive every result path from `BASE_PATH`.

The last cell of each Colab-targeted notebook disconnects the runtime so you stop being billed.

### Colab Drive root

All Colab-targeted notebooks in this project save their PNG and CSV artifacts under a single shared Drive root:

```text
/content/drive/My Drive/Colab Projects/Python ABMs/Distributed Signaling/Plots and Datasets/
```

Each notebook adds its own subfolder under this root so artifacts don't collide. Current convention:

| Notebook                          | Drive subfolder     |
|-----------------------------------|---------------------|
| `proof_of_concept_figures.ipynb`  | `Proof of Concept/` |

When adding a new Colab-targeted notebook, pick a fresh subfolder name under the root above and bake the full path into the notebook's env-switch cell. Update this table at the same time.

## Tooling

`_tools/nb_migrate.py` is a small helper used during the on-going refactor described in [`../NOTEBOOK_REFACTOR_PLAN.md`](../docs/code-audit/NOTEBOOK_REFACTOR_PLAN.md):

```bash
# Bump nbformat to 4.5, set the rl_signaling kernel, assign cell IDs:
python notebooks/_tools/nb_migrate.py upgrade notebooks/

# Report any cells that still call the legacy API or Colab-only magics:
python notebooks/_tools/nb_migrate.py audit notebooks/
```

The audit subcommand is the canonical check that a notebook has been migrated. Run it after any edit; it should report `legacy-API hits: none` on a fully-migrated file.
