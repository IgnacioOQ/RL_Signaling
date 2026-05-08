"""Plotting helpers and post-processing utilities for simulation results."""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_histograms_with_kde(
    df,
    variable,
    bins=100,
    figsize=(5, 3),
    alpha=0.5,
    kde=True,
    variables=[(False, True), (True, False), (False, False)],
    dump_path="./results/",
    filename_prefix="test",
):
    plt.figure(figsize=figsize)

    colors = ["blue", "orange", "green", "red"]

    pretty_titles = {
        "Agent_0_final_reward": "Final Reward",
        "Agent_0_avg_reward": "Average Reward",
        "Agent_1_final_reward": "Final Reward",
        "Agent_1_avg_reward": "Average Reward",
        "Agent_0_NMI": "Final NMI",
        "Agent_1_NMI": "Final NMI",
        "Agent_0_NMI_Difference": "NMI Change (Post - Initial)",
        "Agent_1_NMI_Difference": "NMI Change (Post - Initial)",
    }

    for idx, (with_signals, full_information) in enumerate(variables):
        subset_df = df[
            (df["full_information"] == full_information)
            & (df["with_signals"] == with_signals)
        ]

        plt.hist(
            subset_df[variable],
            bins=bins,
            alpha=alpha,
            color=colors[idx],
            label=f"signals={with_signals}, full_info={full_information}",
            density=True,
        )

        if kde:
            sns.kdeplot(subset_df[variable], color=colors[idx], linewidth=2)

        mean_value = subset_df[variable].mean()
        std_dev = subset_df[variable].std()

        plt.axvline(
            mean_value,
            color=colors[idx],
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {mean_value:.2f}, Std Dev: {std_dev:.2f}",
        )

        vertical_offset = 0.07
        plt.text(
            mean_value,
            plt.gca().get_ylim()[1] * (0.9 - idx * vertical_offset),
            f"{mean_value:.2f}",
            color=colors[idx],
            fontsize=10,
            ha="center",
            bbox=dict(facecolor="white", edgecolor=colors[idx], boxstyle="round,pad=0.3"),
        )

    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{x:g}")
    )
    plt.yticks(rotation=45)

    plt.title(f"KDE-Histogram {pretty_titles[variable]} by Setup", fontsize=12)
    plt.xlabel(pretty_titles[variable], fontsize=10)
    plt.ylabel("Density (10k simulations)", fontsize=10)
    plt.legend(title="Setup", fontsize=9, title_fontsize=10)
    plt.gca().spines[["top", "right"]].set_visible(False)

    ylim = plt.gca().get_ylim()
    plt.ylim(top=ylim[1] * 0.95)

    plt.tight_layout()
    plt.savefig(f"{dump_path}/{filename_prefix}_{variable}.png", dpi=300)
    plt.show()


def plot_basecase_kde(
    df,
    variable,
    file_path="dummy_plot.png",
    bins=100,
    figsize=(6, 3),
    alpha=0.5,
    kde=True,
    variables=[(True, False), (False, True), (False, False)],
):
    plt.figure(figsize=figsize)

    subset_df = df[(df["full_information"] == False) & (df["with_signals"] == True)]

    plt.hist(
        subset_df[variable],
        bins=bins,
        alpha=alpha,
        color="orange",
        label=f"signals={True}, full_info={False}",
        density=True,
    )

    if kde:
        sns.kdeplot(subset_df[variable], color="orange", linewidth=2)

    mean_value = subset_df[variable].mean()
    std_dev = subset_df[variable].std()

    plt.axvline(
        mean_value,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_value:.2f}, Std Dev: {std_dev:.2f}",
    )

    vertical_offset = 0.07
    plt.text(
        mean_value,
        plt.gca().get_ylim()[1] * (0.9 - vertical_offset),
        f"{mean_value:.2f}",
        color="orange",
        fontsize=10,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="orange", boxstyle="round,pad=0.3"),
    )

    plt.title(f"Histogram and KDE of {variable} by Setup", fontsize=12)
    plt.xlabel(variable, fontsize=10)
    plt.ylabel("Density (10k simulations)", fontsize=10)
    plt.legend(title="Setup", fontsize=9, title_fontsize=10)
    plt.gca().spines[["top", "right"]].set_visible(False)

    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_all_histograms(
    df,
    bins=75,
    variables=[(False, True), (True, False), (False, False), (True, True)],
    filename_prefix="test",
):
    plot_histograms_with_kde(
        df,
        "Agent_0_final_reward",
        bins=75,
        variables=[(False, True), (True, False), (False, False), (True, True)],
        filename_prefix=filename_prefix,
    )
    plot_histograms_with_kde(
        df,
        "Agent_0_avg_reward",
        bins=75,
        variables=[(False, True), (True, False), (False, False), (True, True)],
        filename_prefix=filename_prefix,
    )
    plot_histograms_with_kde(
        df,
        "Agent_0_NMI",
        bins=75,
        variables=[(True, False), (True, True)],
        filename_prefix=filename_prefix,
    )


def _calculate_reward_difference(df, agent_col):
    """Reward delta (with_signals=True) - (with_signals=False) for one agent column."""
    return (
        df[df["with_signals"]][agent_col].values - df[~df["with_signals"]][agent_col].values
    )[0]


def compare_payoffs(df):
    iteration_indexes = df["iteration"].unique()

    columns = [
        "iteration",
        "n_signaling_actions",
        "n_final_actions",
        "A0_final_reward_signalvsnon_partialinfo",
        "A0_final_reward_signalvsnon_fullinfo",
        "A1_final_reward_signalvsnon_partialinfo",
        "A1_final_reward_signalvsnon_fullinfo",
    ]
    compared_payoff_df = pd.DataFrame(columns=columns)

    for i in iteration_indexes:
        iteration_df = df[df["iteration"] == i]
        n_signaling_actions = iteration_df["n_signaling_actions"].iloc[0]
        n_final_actions = iteration_df["n_final_actions"].iloc[0]

        full_info = iteration_df[iteration_df["full_information"]]
        partial_info = iteration_df[~iteration_df["full_information"]]

        A0_fullinfo_diff = _calculate_reward_difference(full_info, "Agent_0_final_reward")
        A0_partialinfo_diff = _calculate_reward_difference(partial_info, "Agent_0_final_reward")
        A1_fullinfo_diff = _calculate_reward_difference(full_info, "Agent_1_final_reward")
        A1_partialinfo_diff = _calculate_reward_difference(partial_info, "Agent_1_final_reward")

        compared_payoff_df.loc[len(compared_payoff_df)] = [
            i,
            n_signaling_actions,
            n_final_actions,
            A0_partialinfo_diff,
            A0_fullinfo_diff,
            A1_partialinfo_diff,
            A1_fullinfo_diff,
        ]
    return compared_payoff_df


def plot_payoff_comparison(df):
    compared_payoff_df = compare_payoffs(df)

    variables = [
        "A0_final_reward_signalvsnon_partialinfo",
        "A0_final_reward_signalvsnon_fullinfo",
        "A1_final_reward_signalvsnon_partialinfo",
        "A1_final_reward_signalvsnon_fullinfo",
    ]

    mean_colors = ["red", "blue", "green", "purple"]

    plt.figure(figsize=(10, 6))
    for idx, variable in enumerate(variables):
        compared_payoff_df[variable].plot(kind="hist", bins=50, alpha=0.6, label=variable)
        mean_value = compared_payoff_df[variable].mean()
        plt.axvline(
            mean_value,
            color=mean_colors[idx],
            linestyle="--",
            linewidth=1.5,
            label=f"{variable} Mean: {mean_value:.2f}",
        )

    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.title("Distributions of Difference Signaling vs Not")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def calculate_proportions(data, urn_type="signal_history"):
    proportions = {key: [] for key in data[urn_type][-1].keys()}
    for d in data[urn_type]:
        for key, value in d.items():
            total = np.sum(value)
            proportion = value[0] / total if total != 0 else 0
            proportions[key].append(proportion)
    return proportions


def smooth(values, window_size=3):
    pad_width = window_size // 2
    padded_values = np.pad(values, (pad_width, pad_width), mode="edge")
    smoothed = np.convolve(padded_values, np.ones(window_size) / window_size, mode="valid")
    return smoothed


def count_negative_nmi(file_path):
    """Count negative values in every NMI-bearing column of a results CSV."""
    df = pd.read_csv(file_path)
    nmi_columns = [col for col in df.columns if "NMI" in col]
    negative_counts = {col: (df[col] < 0).sum() for col in nmi_columns}
    return negative_counts


def plot_regression(
    df,
    x_var="Agent_0_NMI",
    y_var="Agent_0_final_reward",
    figsize=(6, 4),
    model_type="linear",
    filter_condition=[(True, True), (True, False)],
    dump_path="./results/",
    filename_prefix="test",
):
    """Plot a regression line between two variables and show coefficients and R²."""
    os.makedirs(dump_path, exist_ok=True)

    for with_signals, full_information in filter_condition:
        subset_df = df[
            (df["full_information"] == full_information)
            & (df["with_signals"] == with_signals)
        ].copy()
        subset_df = subset_df[[x_var, y_var]].dropna()

        X = subset_df[[x_var]].values
        y = subset_df[y_var].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        slope = model.coef_[0]
        intercept = model.intercept_

        plt.figure(figsize=figsize)
        sns.regplot(x=X.flatten(), y=y, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
        plt.title(
            f"Regression: NMI vs. Rewards (full info = {full_information}, signals = {with_signals})"
        )
        plt.xlabel("Final NMI")
        plt.ylabel("Final Reward")
        plt.grid(True)

        eq_str = f"$y = {intercept:.2f} + {slope:.2f}x$\n$R^2 = {r2:.3f}$"
        plt.text(
            0.05,
            0.95,
            eq_str,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )

        plt.tight_layout()

        filename = f"{filename_prefix}_regression_signals_{with_signals}_fullinfo_{full_information}.png"
        filepath = os.path.join(dump_path, filename)
        plt.savefig(filepath, dpi=300)
        print(f"[Saved] {filepath}")

        plt.show()


def plot_reward_vs_cost(df, plot_title="Final Reward vs. Signal Cost"):
    """Scatter + regression of Signal_Cost_A0 against Agent_0_final_reward."""
    if "Signal_Cost_A0" not in df.columns or "Agent_0_final_reward" not in df.columns:
        print(
            "Error: DataFrame is missing 'Signal_Cost_A0' or 'Agent_0_final_reward' column.",
            file=sys.stderr,
        )
        return

    temp_df = df[["Signal_Cost_A0", "Agent_0_final_reward"]].dropna()

    if temp_df.empty:
        print(
            f"Warning: No valid data for regression in '{plot_title}' after dropping NaNs.",
            file=sys.stderr,
        )
        slope, intercept, r2 = np.nan, np.nan, np.nan
    else:
        X = temp_df[["Signal_Cost_A0"]]
        y = temp_df["Agent_0_final_reward"]

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X, y)

    stats_text = (
        f"Slope: {slope:.4f}\n"
        f"Intercept: {intercept:.4f}\n"
        f"R-squared: {r2:.4f}"
    )

    plt.figure(figsize=(10, 6))
    ax = sns.regplot(
        data=df,
        x="Signal_Cost_A0",
        y="Agent_0_final_reward",
        scatter_kws={"alpha": 0.3, "s": 15},
        line_kws={"color": "red", "linewidth": 2},
    )

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )

    plt.title(plot_title, fontsize=16)
    plt.xlabel("Signal Cost", fontsize=12)
    plt.ylabel("Agent 0 Final Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_nmi_vs_cost(df, plot_title="Final NMI vs. Signal Cost"):
    """Scatter + regression of Signal_Cost_A0 against Agent_0_NMI."""
    if "Signal_Cost_A0" not in df.columns or "Agent_0_NMI" not in df.columns:
        print(
            "Error: DataFrame is missing 'Signal_Cost_A0' or 'Agent_0_NMI' column.",
            file=sys.stderr,
        )
        return

    temp_df = df[["Signal_Cost_A0", "Agent_0_NMI"]].dropna()

    if temp_df.empty:
        print(
            f"Warning: No valid data for regression in '{plot_title}' after dropping NaNs.",
            file=sys.stderr,
        )
        slope, intercept, r2 = np.nan, np.nan, np.nan
    else:
        X = temp_df[["Signal_Cost_A0"]]
        y = temp_df["Agent_0_NMI"]

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X, y)

    stats_text = (
        f"Slope: {slope:.4f}\n"
        f"Intercept: {intercept:.4f}\n"
        f"R-squared: {r2:.4f}"
    )

    plt.figure(figsize=(10, 6))
    ax = sns.regplot(
        data=df,
        x="Signal_Cost_A0",
        y="Agent_0_NMI",
        scatter_kws={"alpha": 0.3, "s": 15},
        line_kws={"color": "blue", "linewidth": 2},
    )

    ax.text(
        0.05,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
    )

    plt.title(plot_title, fontsize=16)
    plt.xlabel("Signal Cost", fontsize=12)
    plt.ylabel("Agent 0 Final NMI", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
