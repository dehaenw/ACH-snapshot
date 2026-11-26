#!/usr/bin/env python3
"""Render the SI Y-randomization figure from frozen repeat-level results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_COLORS = {"Desc": "#4C72B0", "FP": "#DD8452", "FCFP": "#55A868"}
ORDER = [
    ("XGB", "Desc"),
    ("RF", "FP"),
    ("SVM", "FP"),
    ("XGB", "FCFP"),
    ("TabPFN", "FCFP"),
]
RC = {
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
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


def render(repeats_path: Path, summary_path: Path, output: Path) -> None:
    repeats = pd.read_csv(repeats_path)
    summary = pd.read_csv(summary_path).set_index(["Model", "Features"])
    missing = [key for key in ORDER if key not in summary.index]
    if missing:
        raise ValueError(f"Missing Y-randomization configurations: {missing}")

    distributions = []
    labels = []
    observed = []
    colors = []
    for model, feature in ORDER:
        values = repeats.loc[
            (repeats["Model"] == model) & (repeats["Features"] == feature), "R2"
        ].to_numpy(float)
        if len(values) != 100:
            raise ValueError(f"Expected 100 null repeats for {model}/{feature}, found {len(values)}")
        distributions.append(values)
        labels.append(f"{model} [{feature}]")
        observed.append(float(summary.loc[(model, feature), "Observed_R2"]))
        colors.append(FEATURE_COLORS[feature])

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(6.70, 3.65))
    positions = np.arange(1, len(ORDER) + 1)
    violin = ax.violinplot(
        distributions,
        positions=positions,
        vert=False,
        widths=0.70,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        bw_method=0.25,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.22)
        body.set_linewidth(0.8)
    violin["cmedians"].set_color("0.25")
    violin["cmedians"].set_linewidth(1.0)

    rng = np.random.default_rng(20261125)
    for y, values, color, obs in zip(positions, distributions, colors, observed):
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(values, y + jitter, s=8, color="0.35", alpha=0.30,
                   edgecolors="none", zorder=2)
        ax.scatter(obs, y, marker="D", s=37, color=color, edgecolors="black",
                   linewidths=0.55, zorder=5)
        ax.text(obs + 0.018, y, f"{obs:.3f}", va="center", ha="left",
                fontsize=6.8, color="0.18")

    ax.axvline(0, color="0.45", lw=0.7, ls=":", zorder=1)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.73, 0.42)
    ax.set_xlabel("Out-of-fold $R^2$")
    ax.set_ylabel("")
    ax.grid(axis="x", color="0.90", lw=0.5, ls=":", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Y-randomization control (100 label permutations per configuration)", pad=4)

    legend = [
        mlines.Line2D([], [], marker="o", linestyle="none", markersize=4.0,
                      markerfacecolor="0.35", markeredgecolor="none", alpha=0.45,
                      label="permuted-label fit"),
        mlines.Line2D([], [], marker="D", linestyle="none", markersize=5.0,
                      markerfacecolor="white", markeredgecolor="black",
                      label="observed-label fit"),
    ]
    # Reserve a footer band so the key and inference note never obscure the
    # lowest violin (TabPFN) or the x-axis title in journal-size rendering.
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.50, 0.070),
               frameon=False, ncol=2, handletextpad=0.35, columnspacing=1.0)
    fig.text(0.50, 0.030,
             "No null repeat reached its observed value; empirical p = 0.0099 each",
             ha="center", va="bottom", fontsize=6.5, color="0.35")
    fig.subplots_adjust(left=0.20, right=0.985, top=0.90, bottom=0.34)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".tif"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeats",
        type=Path,
        default=root / "results" / "y_randomization" / "y_randomization_permutations.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=root / "results" / "y_randomization" / "y_randomization_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "Figure_S_Y_randomization.png",
    )
    args = parser.parse_args()
    render(args.repeats, args.summary, args.output)
    print(f"Wrote {args.output} and vector/TIFF variants")


if __name__ == "__main__":
    main()
