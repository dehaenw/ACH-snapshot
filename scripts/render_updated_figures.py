#!/usr/bin/env python3
"""Render Figures 3–5 after the deterministic graph-model rerun."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_COLORS = {
    "Desc": "#4C72B0",
    "FP": "#DD8452",
    "FCFP": "#55A868",
    "Graph": "#8172B3",
}
FEATURE_ORDER = ["Desc", "FP", "FCFP", "Graph"]
MODEL_DISPLAY = {"BayesianRidge": "BayesRidge"}

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


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".tif"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def render_figure4(results_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(results_dir / "cliff_directionality_pic50.csv")
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 2, figsize=(6.50, 4.25), sharex=True)
    axes = axes.ravel()
    for k, (ax, feat) in enumerate(zip(axes, FEATURE_ORDER)):
        sub = df[df["Features"] == feat].sort_values("pct_correct", ascending=True).reset_index(drop=True)
        color = FEATURE_COLORS[feat]
        y = np.arange(len(sub))
        ax.axvline(50, color="0.38", ls=":", lw=0.8, zorder=1)
        bars = ax.barh(y, sub["pct_correct"], height=0.68, color=color,
                       edgecolor="white", linewidth=0.3, zorder=3)
        for bar, value in zip(bars, sub["pct_correct"]):
            if value <= 50:
                bar.set_hatch("///")
                bar.set_edgecolor("white")
        if feat == "Graph":
            lo = sub["pct_correct_min"].to_numpy(float)
            hi = sub["pct_correct_max"].to_numpy(float)
            mid = sub["pct_correct"].to_numpy(float)
            ax.errorbar(mid, y, xerr=np.vstack([mid - lo, hi - mid]), fmt="none",
                        ecolor="0.20", elinewidth=0.75, capsize=2.0, capthick=0.7,
                        zorder=5)
        for row_idx, (yi, value) in enumerate(zip(y, sub["pct_correct"])):
            label_x = sub.loc[row_idx, "pct_correct_max"] + 1.4 if feat == "Graph" else value + 1.4
            ax.text(min(label_x, 97.5), yi, f"{value:.0f}", va="center",
                    ha="left", fontsize=6.2, color="0.25")
        labels = sub["Model"].replace(MODEL_DISPLAY)
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 100)
        ax.set_title(f"{feat}  (n = 140 pairs)", color=color, pad=3)
        ax.grid(axis="x", color="0.88", lw=0.4, ls=":", zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.text(-0.16, 1.06, "abcd"[k], transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", ha="left")
        if k >= 2:
            ax.set_xlabel("% cliff pairs ranked correctly")

    fig.text(0.5, 0.012,
             "Dotted line: descriptive 50% reference; Graph bars: mean and seed range (n = 5)",
             ha="center", va="bottom", fontsize=6.2, color="0.35")
    fig.subplots_adjust(left=0.18, right=0.985, top=0.94, bottom=0.13,
                        wspace=0.42, hspace=0.38)
    save(fig, out_dir / "Figure_4_cliff_directionality_threshold55.png")


def consistency_summary(df: pd.DataFrame) -> pd.DataFrame:
    cats = ["both_correct", "discordant", "both_wrong"]
    rows = []
    for (model, feat), group in df.groupby(["Model", "Features"], sort=False):
        if feat == "Graph":
            per_seed = []
            for seed, sg in group.groupby("Repeat_seed"):
                counts = sg["consistency"].value_counts()
                total = len(sg)
                per_seed.append({
                    "seed": int(seed),
                    **{f"pct_{c}": 100 * counts.get(c, 0) / total for c in cats},
                    "pct_correct_pic50": 100 * sg["correct_pic50"].mean(),
                    "pct_correct_pki": 100 * sg["correct_pki"].mean(),
                    "n_pairs": total,
                })
            seed_df = pd.DataFrame(per_seed)
            row = {"Model": model, "Features": feat, "n_pairs": int(seed_df["n_pairs"].iloc[0])}
            for col in [f"pct_{c}" for c in cats] + ["pct_correct_pic50", "pct_correct_pki"]:
                row[col] = seed_df[col].mean()
                row[f"{col}_min"] = seed_df[col].min()
                row[f"{col}_max"] = seed_df[col].max()
        else:
            counts = group["consistency"].value_counts()
            total = len(group)
            row = {
                "Model": model,
                "Features": feat,
                "n_pairs": total,
                **{f"pct_{c}": 100 * counts.get(c, 0) / total for c in cats},
                "pct_correct_pic50": 100 * group["correct_pic50"].mean(),
                "pct_correct_pki": 100 * group["correct_pki"].mean(),
            }
            for col in [f"pct_{c}" for c in cats] + ["pct_correct_pic50", "pct_correct_pki"]:
                row[f"{col}_min"] = row[col]
                row[f"{col}_max"] = row[col]
        rows.append(row)
    return pd.DataFrame(rows)


def render_figure5(results_dir: Path, out_dir: Path) -> None:
    raw = pd.read_csv(results_dir / "cross_endpoint_consistency.csv")
    grp = consistency_summary(raw)
    grp["label"] = grp["Model"].replace(MODEL_DISPLAY) + " [" + grp["Features"] + "]"
    grp = grp.sort_values(["pct_both_correct", "pct_correct_pic50"], ascending=True).reset_index(drop=True)

    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(6.50, 5.35),
                            gridspec_kw={"width_ratios": [1.08, 1.42], "wspace": 0.34})
    ax, ax2 = axes
    cats = ["both_correct", "discordant", "both_wrong"]
    labels = {"both_correct": "both endpoints", "discordant": "one endpoint", "both_wrong": "neither"}
    colors = {"both_correct": "#1B7837", "discordant": "#BDBDBD", "both_wrong": "#B2182B"}
    y = np.arange(len(grp))
    left = np.zeros(len(grp))
    for cat in cats:
        vals = grp[f"pct_{cat}"].to_numpy(float)
        ax.barh(y, vals, left=left, height=0.72, color=colors[cat],
                edgecolor="white", linewidth=0.3, label=labels[cat], zorder=3)
        left += vals
    ax.set_yticks(y, grp["label"])
    for tick, feat in zip(ax.get_yticklabels(), grp["Features"]):
        if feat == "Graph":
            tick.set_color(FEATURE_COLORS["Graph"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of matched cliff pairs")
    ax.set_title("Endpoints with correct direction", pad=3)
    ax.grid(axis="x", color="0.90", lw=0.4, ls=":", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.095), ncol=3,
              frameon=False, handlelength=1.0, handletextpad=0.35, columnspacing=0.8)

    # Small deterministic offsets only separate exact ties among point estimates.
    cells: dict[tuple[float, float], list[int]] = {}
    for idx, row in grp.iterrows():
        key = (round(row["pct_correct_pic50"], 8), round(row["pct_correct_pki"], 8))
        cells.setdefault(key, []).append(idx)
    offsets = {}
    for (cx, cy), members in cells.items():
        if len(members) == 1:
            offsets[members[0]] = (cx, cy)
        else:
            for j, idx in enumerate(members):
                a = 2 * np.pi * j / len(members)
                offsets[idx] = (cx + 1.5 * np.cos(a), cy + 1.5 * np.sin(a))
    for idx, row in grp.iterrows():
        x, yy = offsets[idx]
        color = FEATURE_COLORS[row["Features"]]
        if row["Features"] == "Graph":
            x0, y0 = row["pct_correct_pic50"], row["pct_correct_pki"]
            ax2.errorbar(
                x0, y0,
                # Clip machine-precision round-off when a five-seed range
                # collapses exactly onto its mean.
                xerr=np.maximum(0.0, np.array([[x0 - row["pct_correct_pic50_min"]],
                                              [row["pct_correct_pic50_max"] - x0]])),
                yerr=np.maximum(0.0, np.array([[y0 - row["pct_correct_pki_min"]],
                                              [row["pct_correct_pki_max"] - y0]])),
                fmt="o", ms=5.2, color=color, ecolor=color, elinewidth=0.8,
                capsize=2.0, markeredgecolor="white", markeredgewidth=0.45,
                alpha=0.95, zorder=5,
            )
        else:
            ax2.scatter(x, yy, s=27, color=color, edgecolors="white",
                        linewidths=0.45, alpha=0.90, zorder=3)
    ax2.plot([0, 100], [0, 100], color="0.35", ls="--", lw=0.8, zorder=1)
    ax2.set_xlim(-3, 103)
    ax2.set_ylim(-3, 103)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("% correct — pIC$_{50}$")
    ax2.set_ylabel("% correct — pK$_\\mathrm{i}$")
    ax2.set_title("Agreement between endpoints", pad=3)
    ax2.grid(color="0.90", lw=0.5, zorder=0)
    ax2.set_aspect("equal", adjustable="box")
    handles = [mlines.Line2D([0], [0], marker="o", color="w", markersize=4.5,
                             markerfacecolor=FEATURE_COLORS[f], label=f)
               for f in FEATURE_ORDER]
    ax2.legend(handles=handles, loc="upper right", title="Feature space",
               title_fontsize=6.5, frameon=False, handletextpad=0.25, labelspacing=0.25)
    ax2.text(0.98, 0.02,
             "dashed line = equal accuracy\nGraph whiskers = seed range (n = 5)",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=6.0, color="0.40")
    for axis, letter, xpos in [(ax, "a", -0.28), (ax2, "b", -0.17)]:
        axis.text(xpos, 1.045, letter, transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="top", ha="left")
    fig.subplots_adjust(left=0.205, right=0.985, top=0.96, bottom=0.115, wspace=0.34)
    save(fig, out_dir / "Figure_5_cross_endpoint_threshold55.png")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    release_root = script_dir.parent if (script_dir.parent / "data").is_dir() else script_dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=release_root / "results")
    parser.add_argument("--out-dir", type=Path, default=release_root / "results")
    args = parser.parse_args()
    render_figure4(args.results_dir, args.out_dir)
    render_figure5(args.results_dir, args.out_dir)
    print(f"Wrote Figures 4 and 5 to {args.out_dir}")


if __name__ == "__main__":
    main()
