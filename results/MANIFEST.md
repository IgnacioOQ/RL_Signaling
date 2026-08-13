# Figure Manifest
- status: active
- type: reference
- id: rl_signaling.results.manifest
- description: Traceability map from every published figure of "Signaling Games with Distributed Rewards" (PHOS-17993) back to the notebook, module function, and dataset that produced it; also classifies the exploratory figures that did not reach the paper.
- label: [core]
- injection: informational
- volatility: stable
- scope: project-specific
- last_checked: 2026-08-12
<!-- content -->

This file answers one question: **which code produced which figure in the paper?**

The manuscript sources are not part of this repository (see [PUBLICATION_CHECKLIST.md](../PUBLICATION_CHECKLIST.md)), so figures are located by their paper section and `\label`, not by file path. Section numbers refer to the published article.

`results/` holds 54 tracked PNGs. **27 appear in the paper**; the other 27 are exploratory variants retained as a record of what was examined. The two groups are separated below so a reader is never left guessing which is which.

## Naming convention

Most figures are not named literally anywhere in the codebase — they are **constructed at save time** from a prefix plus a variable name. Grepping for a figure's filename will therefore find nothing. The two constructors are:

| Constructor | Pattern | Defined at |
|:---|:---|:---|
| `plot_histograms_with_kde` | `{dump_path}/{filename_prefix}_{variable}.png` | [plotting.py:127](../rl_signaling/plotting.py#L127) |
| `plot_regression` | `{dump_path}/{filename_prefix}_regression_signals_{with_signals}_fullinfo_{full_information}.png` | [plotting.py:556](../rl_signaling/plotting.py#L556) |

Both are driven from [notebooks/plotting_results.ipynb](../notebooks/plotting_results.ipynb), which sets `dump_path = '../results/'` and iterates over seven `filename_prefix` values.

## Paper figures

### Main text

| Paper location | Figure file | Produced by | Input data | Traced |
|:---|:---|:---|:---|:---|
| §2.2 Proof of Concept, `fig:proof-of-concept` | `proof_of_concept/proof_of_concept_plot_RE.png` | [proof_of_concept_figures_final.ipynb](../notebooks/proof_of_concept_figures_final.ipynb) | generated in-notebook | inferred — see Gap 1 |
| §3.1 Matching Games, `fig:canonical_figures` | `legacy/plots/Roth-Erev_canonical_Agent_0_final_reward.png` | `plot_histograms_with_kde`, prefix `Roth-Erev_canonical` | `legacy/datasets/urnagent_results_canonical.csv` | confirmed |
| §3.1 | `legacy/plots/Q-learning_canonical_Agent_0_final_reward.png` | `plot_histograms_with_kde`, prefix `Q-learning_canonical` | `legacy/datasets/qlearning_results_canonical.csv` | confirmed |
| §3.1 | `legacy/plots/Roth-Erev_canonical_Agent_0_NMI.png` | `plot_histograms_with_kde`, prefix `Roth-Erev_canonical` | `legacy/datasets/urnagent_results_canonical.csv` | confirmed |
| §3.1 | `legacy/plots/Q-learning_canonical_Agent_0_NMI.png` | `plot_histograms_with_kde`, prefix `Q-learning_canonical` | `legacy/datasets/qlearning_results_canonical.csv` | confirmed |
| §3.1 | `legacy/plots/Roth-Erev_canonical_regression_signals_True_fullinfo_False.png` | `plot_regression`, prefix `Roth-Erev_canonical` | `legacy/datasets/urnagent_results_canonical.csv` | confirmed |
| §3.1 | `legacy/plots/Q-learning_canonical_regression_signals_True_fullinfo_False.png` | `plot_regression`, prefix `Q-learning_canonical` | `legacy/datasets/qlearning_results_canonical.csv` | confirmed |
| §3.2 Random Games, `fig:random_figures` | `legacy/plots/Roth-Erev_complex_randomized_Agent_0_final_reward.png` | `plot_histograms_with_kde`, prefix `Roth-Erev_complex_randomized` | `legacy/datasets/urnagent_results_complex_randomized.csv` | confirmed |
| §3.2 | `legacy/plots/Q-learning_complex_randomized_Agent_0_final_reward.png` | `plot_histograms_with_kde`, prefix `Q-learning_complex_randomized` | `legacy/datasets/qlearning_results_complex_randomized.csv` | confirmed |
| §3.2 | `legacy/plots/Roth-Erev_complex_randomized_Agent_0_NMI.png` | `plot_histograms_with_kde`, prefix `Roth-Erev_complex_randomized` | `legacy/datasets/urnagent_results_complex_randomized.csv` | confirmed |
| §3.2 | `legacy/plots/Q-learning_complex_randomized_Agent_0_NMI.png` | `plot_histograms_with_kde`, prefix `Q-learning_complex_randomized` | `legacy/datasets/qlearning_results_complex_randomized.csv` | confirmed |
| §3.2 | `legacy/plots/Roth-Erev_complex_randomized_regression_signals_True_fullinfo_False.png` | `plot_regression`, prefix `Roth-Erev_complex_randomized` | `legacy/datasets/urnagent_results_complex_randomized.csv` | confirmed |
| §3.2 | `legacy/plots/Q-learning_complex_randomized_regression_signals_True_fullinfo_False.png` | `plot_regression`, prefix `Q-learning_complex_randomized` | `legacy/datasets/qlearning_results_complex_randomized.csv` | confirmed |

`Roth-Erev` is the Roth–Erev urn agent (`UrnAgent`); its datasets are named `urnagent_*`.

### Appendix

| Paper location | Figure file | Produced by | Input data | Traced |
|:---|:---|:---|:---|:---|
| Costly Signals and Alarms, `fig:costlysignals` | `legacy/plots/q_rewards_costlysignal.png` | costly-signal run | `legacy/datasets/qlearning_results_canonical_costly_signal.csv` | inferred — Gap 2 |
| Costly Signals, `fig:costlysignals` | `legacy/plots/q_signalusage_costlysignal.png` | costly-signal run | same | inferred — Gap 2 |
| Costly Signals, `fig:rewardvscost` | `legacy/plots/q_rewardvscost_costlysignal.png` | `plot_regression_cost` family, [plotting.py:567](../rl_signaling/plotting.py#L567) | same | inferred — Gap 2 |
| Costly Signals, `fig:rewardvscost` | `legacy/plots/q_nmivscost_costlysignal.png` | NMI-vs-cost variant, [plotting.py:630](../rl_signaling/plotting.py#L630) | same | inferred — Gap 2 |
| TD Results, `fig:td_results` | `legacy/plots/TD-learning_canonical_Agent_0_final_reward.png` | `plot_histograms_with_kde`, prefix `TD-learning_canonical` | `legacy/datasets/td_learning_results_canonical.csv` | confirmed |
| TD Results | `legacy/plots/TD-learning_canonical_Agent_0_NMI.png` | `plot_histograms_with_kde`, prefix `TD-learning_canonical` | same | confirmed |
| TD Results | `legacy/plots/TD-learning_canonical_regression_signals_True_fullinfo_False.png` | `plot_regression`, prefix `TD-learning_canonical` | same | confirmed |
| TD Results | `legacy/plots/td_complex_reward.png` | TD complex-randomized run | `legacy/datasets/td_learning_results_complex_randomized.csv` | inferred — Gap 3 |
| TD Results | `legacy/plots/td_complex_nmi.png` | TD complex-randomized run | same | inferred — Gap 3 |
| TD Results | `legacy/plots/td_complex_random_regression.png` | TD complex-randomized run | same | inferred — Gap 3 |
| Optimization / Matching Games, `fig:canon_opt` | `legacy/plots/q_opt_canonical.png` | [Parameter_Optimization_wchoices.ipynb](../notebooks/Parameter_Optimization_wchoices.ipynb) | Colab sweep, 500 settings × 200 trials | inferred — Gap 4 |
| Optimization / Matching Games | `legacy/plots/td_opt_canonical.png` | same | same | inferred — Gap 4 |
| Optimization / Random Games, `fig:randomized_opt` | `legacy/plots/q_opt_games.png` | same | same | inferred — Gap 4 |
| Optimization / Random Games | `legacy/plots/td_opt_games.png` | same | same | inferred — Gap 4 |

## Known reproducibility gaps

These are recorded rather than papered over. "Inferred" above means the attribution follows from the filename, the surrounding paper text, and the available data, but no line of code in this repository writes that exact filename.

1. **Proof-of-concept renaming.** The notebooks emit `proof_of_concept_plot.png` and `proof_of_concept_plot_qlearning.png`. The paper uses `proof_of_concept_plot_RE.png` and `_QL.png`. The `_RE` / `_QL` files were renamed by hand; the rename is not scripted.
2. **Costly-signal figure names.** `q_*_costlysignal.png` do not follow the `filename_prefix` convention, and `plotting_results.ipynb` sets prefix `QLearning_canonical_costly_signal` — which produced the three `QLearning_canonical_costly_signal_*` files listed as exploratory below, *not* the four figures the appendix actually uses. The published four were saved with an explicit `file_path` argument in a session not preserved in the notebooks.
3. **TD complex-randomized figures.** `td_complex_*.png` likewise bypass the prefix convention, while the conventionally-named `TD-learning_complex_randomized_*` files went unused. Same cause as Gap 2.
4. **Optimization sweep is not reproducible from a clean clone.** `Parameter_Optimization_wchoices.ipynb` sets `dump_path` to a Google Drive path (`/content/drive/My Drive/Colab Projects/...`). The sweep ran on Colab and its raw output was never committed — only the four summary PNGs. Re-running requires re-executing the sweep and re-pointing `dump_path`.

Gaps 2 and 3 share one root cause: the paper's final figures were produced in an interactive session that set `file_path` directly, while the committed notebook drives the prefix-based API. The underlying **data** for both is committed, so the figures are regenerable in substance, but not byte-identically by re-running the notebook as committed.

## Regenerating the confirmed figures

```bash
pip install -e .
jupyter nbconvert --to notebook --execute --inplace notebooks/plotting_results.ipynb
```

This reads `results/legacy/datasets/*.csv` and rewrites the 15 "confirmed" figures into `results/legacy/plots/`. It does not regenerate the figures listed under Gaps 1–4.

## Exploratory figures (not in the paper)

Retained as a record of what was examined. Grouped by why they exist.

| Group | Files | Note |
|:---|:---|:---|
| `avg_reward` variants | `{Roth-Erev,Q-learning,TD-learning}_{canonical,complex_randomized}_Agent_0_avg_reward.png` (6) | Average rather than final reward; the paper reports final reward throughout. |
| `fullinfo_True` regressions | `{Roth-Erev,Q-learning,TD-learning}_*_regression_signals_True_fullinfo_True.png` (5) | Full-information control condition; the paper shows the partial-information cell. |
| Prefix-convention costly-signal | `QLearning_canonical_costly_signal_Agent_0_{NMI,avg_reward,final_reward}.png` (3) | Superseded by the hand-named appendix figures — see Gap 2. |
| Prefix-convention TD complex | `TD-learning_complex_randomized_Agent_0_{NMI,avg_reward,final_reward}.png` (3) | Superseded — see Gap 3. |
| TD complex alternates | `td_complex_random_{nmi,reward}.png`, `td_complex_regression.png` (3) | Near-duplicates of the appendix TD figures under different randomization. |
| Proof-of-concept alternates | `proof_of_concept_plot.png`, `proof_of_concept_plot_QL.png` (2) | `_QL` is the Q-learning companion; the paper prints only the Roth–Erev panel. |
| Initialization study | `init_smooth_{nmi,r}.png` (2) | From [Initializations_test.ipynb](../notebooks/Initializations_test.ipynb); the initialization-basin analysis lives in `analytics/math/initialization_basins.md`, which is kept local and not distributed with this repository. |
| Worked example | `example_process_{nmi,rewards}.png` (2) | Single-run illustration of one simulation trajectory. |
| Costly-signal NMI | `q_nmi_costlysignal.png` (1) | Cut from the appendix figure pair. |
| Post-refactor check | `new_code/plots/figure_ql_vs_re_canonical.png` (1) | From `analytics/scripts/figure_ql_vs_re_canonical.py` (kept local, not distributed) — the only figure whose filename appears literally in the source. Verifies the refactored package reproduces the legacy comparison. |

## Datasets

`results/legacy/datasets/` holds the seven simulation outputs behind the paper (~24 MB, tracked deliberately). Each is 10,000 simulations per condition.

| File | Agent | Game |
|:---|:---|:---|
| `urnagent_results_canonical.csv` | Roth–Erev urn | matching |
| `urnagent_results_complex_randomized.csv` | Roth–Erev urn | random |
| `qlearning_results_canonical.csv` | Q-learning | matching |
| `qlearning_results_complex_randomized.csv` | Q-learning | random |
| `qlearning_results_canonical_costly_signal.csv` | Q-learning | matching, costly signals |
| `td_learning_results_canonical.csv` | TD-learning | matching |
| `td_learning_results_complex_randomized.csv` | TD-learning | random |

These are kept in version control rather than regenerated: the runs are expensive, and they are the evidentiary basis for the published results.
