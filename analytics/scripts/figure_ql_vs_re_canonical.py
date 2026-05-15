import pandas as pd, matplotlib.pyplot as plt, numpy as np

ql = pd.read_csv("results/legacy/datasets/qlearning_results_canonical.csv")
re = pd.read_csv("results/legacy/datasets/urnagent_results_canonical.csv")
ql_signal = ql[(ql["with_signals"] == True) & (ql["full_information"] == False)]
re_signal = re[(re["with_signals"] == True) & (re["full_information"] == False)]

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
bins = np.linspace(0, 1, 25)
for ax, df, label in [
    (axes[0], re_signal, "Roth–Erev"),
    (axes[1], ql_signal, "Q-learning"),
]:
    ax.hist(df["Agent_0_final_reward"], bins=bins, color="steelblue", edgecolor="white")
    ax.axvline(df["Agent_0_final_reward"].mean(), color="firebrick", ls="--",
               label=f"mean = {df['Agent_0_final_reward'].mean():.2f}")
    ax.set_xlim(0, 1); ax.set_xlabel("Final reward (Agent 0)")
    ax.set_title(label); ax.legend()
axes[0].set_ylabel("Count of seeds")
plt.tight_layout(); plt.savefig("results/new_code/plots/figure_ql_vs_re_canonical.png", dpi=150)
