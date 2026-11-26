#!/usr/bin/env python3
"""Render revised Figure 3 from the frozen prospective-pair results table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


FEATURE_COLORS = {
    "Desc": "#4C72B0",
    "FP": "#DD8452",
    "FCFP": "#55A868",
    "Graph": "#8172B3",
}

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def parse_ci(text: str) -> tuple[float, float]:
    match = re.fullmatch(
        r"\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]",
        str(text),
    )
    if not match:
        raise ValueError(f"Cannot parse confidence interval: {text!r}")
    return float(match.group(1)), float(match.group(2))


def render(input_csv: Path, external_stats_csv: Path, output_png: Path) -> None:
    df = pd.read_csv(input_csv)
    observed = pd.read_csv(external_stats_csv)
    observed = observed[observed["analysis"] == "primary_all_six_cells"].set_index("pair")
    ethyl_col = "Δ N-ethyl (HC49−HC22)"
    propyl_col = "Δ N-propyl (HC46−HC52)"
    mean_col = "Mean Δ"
    ci_col = ("Mean Δ, uncertainty interval"
              if "Mean Δ, uncertainty interval" in df.columns
              else "Mean Δ, bootstrap 95% CI")
    required = {"Model", "Features", ethyl_col, propyl_col, mean_col, ci_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col in (ethyl_col, propyl_col, mean_col):
        df[col] = pd.to_numeric(df[col], errors="raise")
    df[["ci_low", "ci_high"]] = pd.DataFrame(
        df[ci_col].map(parse_ci).tolist(), index=df.index
    )
    df["label"] = df["Model"].replace({"BayesianRidge": "BayesRidge"}) + " [" + df["Features"] + "]"
    df = df.sort_values([mean_col, "Model", "Features"], ascending=[True, True, True]).reset_index(drop=True)

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(6.50, max(5.35, 0.195 * len(df) + 0.65)))

    ethyl_obs = observed.loc["N-ethyl"]
    propyl_obs = observed.loc["N-propyl"]
    ax.axvspan(ethyl_obs["delta_pIC50_ci95_low"], ethyl_obs["delta_pIC50_ci95_high"],
               color="0.84", alpha=0.45, zorder=0)
    ax.axvspan(propyl_obs["delta_pIC50_ci95_low"], propyl_obs["delta_pIC50_ci95_high"],
               color="#D9E2EC", alpha=0.55, zorder=0)
    ax.axvline(propyl_obs["delta_pIC50"], color="0.30", lw=0.8,
               ls=(0, (2, 2)), zorder=1)
    ax.axvline(ethyl_obs["delta_pIC50"], color="0.30", lw=0.8, ls="-", zorder=1)
    ax.axvline(0, color="0.15", lw=0.75, zorder=1)

    y = list(range(len(df)))
    for yi, row in df.iterrows():
        color = FEATURE_COLORS[row["Features"]]
        low, high, mean = row["ci_low"], row["ci_high"], row[mean_col]

        # Classical rows show bootstrap 95% intervals; graph rows show the
        # full range across the five predefined training seeds.
        ax.plot([low, high], [yi, yi], color=color, lw=1.25, alpha=0.48, zorder=2)
        ax.plot([low, low], [yi - 0.11, yi + 0.11], color=color, lw=0.7, alpha=0.60, zorder=2)
        ax.plot([high, high], [yi - 0.11, yi + 0.11], color=color, lw=0.7, alpha=0.60, zorder=2)
        ax.scatter(mean, yi, marker="D", s=13, facecolor="white", edgecolor=color,
                   linewidth=0.75, zorder=4)

        # The two prospective matched-pair predictions.
        ax.scatter(row[ethyl_col], yi + 0.13, marker="o", s=19, color=color,
                   edgecolor="white", linewidth=0.35, zorder=5)
        ax.scatter(row[propyl_col], yi - 0.13, marker="^", s=21, color=color,
                   edgecolor="white", linewidth=0.35, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    for tick, feat in zip(ax.get_yticklabels(), df["Features"]):
        tick.set_color(FEATURE_COLORS[feat])

    ax.set_xlim(-1.65, 1.95)
    ax.set_ylim(-0.65, len(df) - 0.12)
    ax.set_xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
    ax.set_xlabel("Signed within-pair difference, ΔpIC$_{50}$ (meta-Cl − para-Cl)")
    ax.set_title("Prospective matched-pair predictions", loc="left", pad=5)
    ax.grid(axis="x", color="0.88", lw=0.4, ls=":", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", pad=3)

    # Labels sit above the plotting region, clear of the top row.
    transform = ax.get_xaxis_transform()
    ax.text(propyl_obs["delta_pIC50"], 1.008,
            f"observed\nN-propyl  +{propyl_obs['delta_pIC50']:.2f}", transform=transform,
            ha="right", va="bottom", fontsize=6.2, color="0.25", clip_on=False)
    ax.text(ethyl_obs["delta_pIC50"], 1.008,
            f"observed\nN-ethyl  +{ethyl_obs['delta_pIC50']:.2f}", transform=transform,
            ha="left", va="bottom", fontsize=6.2, color="0.25", clip_on=False)

    pair_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.25",
               markeredgecolor="white", markersize=4.5,
               label="N-ethyl: HC49 − HC22"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="0.25",
               markeredgecolor="white", markersize=5,
               label="N-propyl: HC46 − HC52"),
        Line2D([0], [0], marker="D", color="0.45", markerfacecolor="white",
               markeredgecolor="0.45", lw=1.2, markersize=4,
               label="Mean Δ with uncertainty interval"),
        Patch(facecolor="0.84", alpha=0.45, edgecolor="none",
              label="Observed Δ: 95% CI from cell fits"),
    ]
    # Keep the two-row key and the interval note in separate figure-level
    # bands so they cannot collide after tight-bounding-box export.
    fig.legend(handles=pair_handles, loc="lower center", bbox_to_anchor=(0.60, 0.052),
               ncol=2, frameon=False, handletextpad=0.45, columnspacing=1.25)
    fig.text(0.60, 0.018,
             "Intervals: bootstrap 95% CI for non-graph models; seed range (n = 5) for graph models",
             ha="center", va="bottom", fontsize=6.1, color="0.35")

    # Feature space is encoded redundantly by the bracketed y-axis labels and
    # their colour, so a second legend would only compete with the endpoint key.
    fig.subplots_adjust(left=0.225, right=0.985, top=0.895, bottom=0.205)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_png.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_png.with_suffix(".tif"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    release_root = script_dir.parent if (script_dir.parent / "data").is_dir() else script_dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=release_root / "results" / "prospective_pair_deltas_thr55.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=release_root / "results" / "Figure_3_paired_deltas_threshold55.png",
    )
    parser.add_argument(
        "--external-stats",
        type=Path,
        default=release_root / "results" / "external_pair_statistics.csv",
    )
    args = parser.parse_args()
    render(args.input, args.external_stats, args.output)


if __name__ == "__main__":
    main()
