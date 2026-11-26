#!/usr/bin/env python3
"""Descriptor-space PCA applicability-domain analysis for the pIC50 models.

The scaler and PCA are fitted on the training set only. External compounds are
then projected without refitting. The quantitative domain assessment combines
the PCA score distance (Hotelling T2) with the orthogonal residual (Q); empirical
95th-percentile training thresholds are used for both quantities.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, DataStructs
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")

SEED = 42
FP_RADIUS = 2
FP_SIZE = 2048
FIG_W2 = 6.50
FS_BASE = 8
FS_ANNOT = 7
FS_TICK = 6
FS_MIN = 6

EXT_SMILES = [
    "c1ccc(Cl)cc1C1(NCC)C(=O)CCCC1",
    "c1cc(Cl)ccc1C1(NCC)C(=O)CCCC1",
    "c1ccc(Cl)cc1C1(NCCC)C(=O)CCCC1",
    "c1cc(Cl)ccc1C1(NCCC)C(=O)CCCC1",
]
EXT_LABELS = ["meta-Cl/Et", "para-Cl/Et", "meta-Cl/Pr", "para-Cl/Pr"]
EXT_COLORS = ["#1b4332", "#6a0572", "#7b2d00", "#003566"]
EXT_MARKERS = ["*", "P", "X", "D"]

DESC_NAMES = [name for name, _ in Descriptors.descList]
DESC_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)
MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_SIZE
)

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": FS_BASE,
    "axes.titlesize": FS_BASE,
    "axes.labelsize": FS_BASE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_ANNOT,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


def molecule_descriptors(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return cleaned RDKit descriptor rows and a valid-structure mask."""
    rows = []
    valid = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid.append(False)
            continue
        values = np.asarray(DESC_CALC.CalcDescriptors(mol), dtype=float)
        rows.append(np.where(np.isfinite(values), values, 0.0))
        valid.append(True)
    return np.vstack(rows), np.asarray(valid, dtype=bool)


def ecfp4_max_similarity(
    train_smiles: list[str], external_smiles: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum binary ECFP4 Tanimoto and nearest training-row index."""
    train_fps = [MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles]
    maxima, indices = [], []
    for smi in external_smiles:
        fp = MORGAN.GetFingerprint(Chem.MolFromSmiles(smi))
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        idx = int(np.argmax(similarities))
        maxima.append(float(similarities[idx]))
        indices.append(idx)
    return np.asarray(maxima), np.asarray(indices, dtype=int)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Percent of reference observations less than or equal to each value."""
    reference = np.asarray(reference, dtype=float)
    return np.asarray([(reference <= value).mean() * 100.0 for value in values])


def fit_descriptor_ad(
    train_smiles: list[str], external_smiles: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit training-only PCA and return external summary and all plot points."""
    x_train, train_valid = molecule_descriptors(train_smiles)
    if not train_valid.all():
        train_smiles = list(np.asarray(train_smiles)[train_valid])
    x_ext, ext_valid = molecule_descriptors(external_smiles)
    if not ext_valid.all():
        raise ValueError("At least one external structure could not be parsed")

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_scaled = scaler.fit_transform(x_train)
    x_ext_scaled = scaler.transform(x_ext)

    pca = PCA(svd_solver="full")
    scores_train_all = pca.fit_transform(x_train_scaled)
    scores_ext_all = pca.transform(x_ext_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_pc = int(np.searchsorted(cumulative, 0.95) + 1)
    retained_variance = float(cumulative[n_pc - 1])

    scores_train = scores_train_all[:, :n_pc]
    scores_ext = scores_ext_all[:, :n_pc]
    eigenvalues = pca.explained_variance_[:n_pc]

    t2_train = np.sum(scores_train**2 / eigenvalues, axis=1)
    t2_ext = np.sum(scores_ext**2 / eigenvalues, axis=1)

    reconstruction_train = scores_train @ pca.components_[:n_pc]
    reconstruction_ext = scores_ext @ pca.components_[:n_pc]
    q_train = np.sum((x_train_scaled - reconstruction_train) ** 2, axis=1)
    q_ext = np.sum((x_ext_scaled - reconstruction_ext) ** 2, axis=1)

    t2_limit = float(np.quantile(t2_train, 0.95))
    q_limit = float(np.quantile(q_train, 0.95))

    train_distances = pairwise_distances(scores_train)
    np.fill_diagonal(train_distances, np.inf)
    train_nn = train_distances.min(axis=1)
    ext_distances = pairwise_distances(scores_ext, scores_train)
    ext_nn_idx = np.argmin(ext_distances, axis=1)
    ext_nn = ext_distances[np.arange(len(external_smiles)), ext_nn_idx]

    tanimoto, tanimoto_idx = ecfp4_max_similarity(train_smiles, external_smiles)

    summary = pd.DataFrame({
        "Compound": EXT_LABELS,
        "SMILES": external_smiles,
        "PC1": scores_ext_all[:, 0],
        "PC2": scores_ext_all[:, 1],
        "PC1_explained_variance_pct": 100.0 * pca.explained_variance_ratio_[0],
        "PC2_explained_variance_pct": 100.0 * pca.explained_variance_ratio_[1],
        "PCs_for_95pct_variance": n_pc,
        "Variance_retained": retained_variance,
        "Hotelling_T2": t2_ext,
        "Training_T2_95pct": t2_limit,
        "T2_fraction_of_limit": t2_ext / t2_limit,
        "T2_training_percentile": empirical_percentile(t2_train, t2_ext),
        "Q_residual": q_ext,
        "Training_Q_95pct": q_limit,
        "Q_fraction_of_limit": q_ext / q_limit,
        "Q_training_percentile": empirical_percentile(q_train, q_ext),
        "Inside_empirical_95pct_PCA_AD": (t2_ext <= t2_limit) & (q_ext <= q_limit),
        "PC95_nearest_neighbor_distance": ext_nn,
        "PC95_NN_training_percentile": empirical_percentile(train_nn, ext_nn),
        "Nearest_descriptor_training_SMILES": np.asarray(train_smiles)[ext_nn_idx],
        "Maximum_ECFP4_Tanimoto": tanimoto,
        "Nearest_ECFP4_training_SMILES": np.asarray(train_smiles)[tanimoto_idx],
    })

    train_points = pd.DataFrame({
        "Set": "Training",
        "Compound": [f"training_{i + 1}" for i in range(len(train_smiles))],
        "SMILES": train_smiles,
        "PC1": scores_train_all[:, 0],
        "PC2": scores_train_all[:, 1],
        "T2_fraction_of_limit": t2_train / t2_limit,
        "Q_fraction_of_limit": q_train / q_limit,
    })
    ext_points = pd.DataFrame({
        "Set": "External",
        "Compound": EXT_LABELS,
        "SMILES": external_smiles,
        "PC1": scores_ext_all[:, 0],
        "PC2": scores_ext_all[:, 1],
        "T2_fraction_of_limit": t2_ext / t2_limit,
        "Q_fraction_of_limit": q_ext / q_limit,
    })
    points = pd.concat([train_points, ext_points], ignore_index=True)

    two_pc_t2 = (
        scores_train_all[:, 0] ** 2 / pca.explained_variance_[0]
        + scores_train_all[:, 1] ** 2 / pca.explained_variance_[1]
    )
    two_pc_limit = float(np.quantile(two_pc_t2, 0.95))
    plot_info = {
        "pc1_variance_pct": 100.0 * pca.explained_variance_ratio_[0],
        "pc2_variance_pct": 100.0 * pca.explained_variance_ratio_[1],
        "pc1_variance": float(pca.explained_variance_[0]),
        "pc2_variance": float(pca.explained_variance_[1]),
        "two_pc_t2_limit": two_pc_limit,
        "n_pc": n_pc,
        "retained_variance": retained_variance,
    }
    return summary, points, plot_info


def plot_descriptor_ad(
    summary: pd.DataFrame, points: pd.DataFrame, plot_info: dict, output_path: str
) -> None:
    """Create the two-panel SI applicability-domain figure."""
    plt.rcParams.update(RC)
    fig, (ax_pca, ax_ad) = plt.subplots(1, 2, figsize=(FIG_W2, 3.0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.25, wspace=0.34)

    train = points[points["Set"] == "Training"]
    external = points[points["Set"] == "External"]

    ax_pca.scatter(
        train["PC1"], train["PC2"], s=8, color="0.62", alpha=0.55,
        linewidths=0, rasterized=True, zorder=1,
    )
    ellipse = Ellipse(
        (0.0, 0.0),
        width=2.0 * np.sqrt(plot_info["two_pc_t2_limit"] * plot_info["pc1_variance"]),
        height=2.0 * np.sqrt(plot_info["two_pc_t2_limit"] * plot_info["pc2_variance"]),
        facecolor="none", edgecolor="0.30", lw=0.8, ls="--", zorder=2,
    )
    ax_pca.add_patch(ellipse)

    handles = [mlines.Line2D(
        [], [], marker="o", linestyle="none", markersize=4,
        markerfacecolor="0.62", markeredgecolor="none", label="Training set"
    )]
    label_offsets = [(-11, 9), (-11, -10), (11, 9), (11, -10)]
    for index, (_, row) in enumerate(external.iterrows()):
        ax_pca.scatter(
            row["PC1"], row["PC2"], s=58, marker=EXT_MARKERS[index],
            color=EXT_COLORS[index], edgecolors="white", linewidths=0.6,
            zorder=4,
        )
        ax_pca.annotate(
            str(index + 1), xy=(row["PC1"], row["PC2"]),
            xytext=label_offsets[index], textcoords="offset points",
            fontsize=FS_MIN, color=EXT_COLORS[index], fontweight="bold",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=EXT_COLORS[index], lw=0.45),
            zorder=5,
        )
        handles.append(mlines.Line2D(
            [], [], marker=EXT_MARKERS[index], linestyle="none", markersize=5,
            markerfacecolor=EXT_COLORS[index], markeredgecolor="white",
            markeredgewidth=0.5, label=f"{index + 1} {row['Compound']}",
        ))
    handles.append(mlines.Line2D(
        [], [], color="0.30", ls="--", lw=0.8, label="95% training ellipse"
    ))
    ax_pca.set_xlabel(f"PC1 ({plot_info['pc1_variance_pct']:.1f}% variance)")
    ax_pca.set_ylabel(f"PC2 ({plot_info['pc2_variance_pct']:.1f}% variance)")
    ax_pca.set_title("Descriptor PCA")

    ax_ad.add_patch(Rectangle(
        (0, 0), 1, 1, facecolor="#2a9d8f", alpha=0.08, edgecolor="none", zorder=0
    ))
    ax_ad.scatter(
        train["T2_fraction_of_limit"], train["Q_fraction_of_limit"],
        s=8, color="0.62", alpha=0.45, linewidths=0, rasterized=True, zorder=1,
    )
    for index, (_, row) in enumerate(external.iterrows()):
        ax_ad.scatter(
            row["T2_fraction_of_limit"], row["Q_fraction_of_limit"],
            s=58, marker=EXT_MARKERS[index], color=EXT_COLORS[index],
            edgecolors="white", linewidths=0.6, zorder=4,
        )
        ax_ad.annotate(
            str(index + 1),
            xy=(row["T2_fraction_of_limit"], row["Q_fraction_of_limit"]),
            xytext=label_offsets[index], textcoords="offset points",
            fontsize=FS_MIN, color=EXT_COLORS[index], fontweight="bold",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="-", color=EXT_COLORS[index], lw=0.45),
            zorder=5,
        )
    ax_ad.axvline(1.0, color="0.30", lw=0.8, ls="--")
    ax_ad.axhline(1.0, color="0.30", lw=0.8, ls="--")
    ax_ad.set_xlabel(r"Score distance, $T^2/T^2_{95}$")
    ax_ad.set_ylabel(r"Orthogonal distance, $Q/Q_{95}$")
    ax_ad.set_title(
        f"PCA domain ({plot_info['n_pc']} PCs; "
        f"{100.0 * plot_info['retained_variance']:.1f}% variance)"
    )
    ax_ad.text(
        0.04, 0.96, "inside both empirical 95% limits",
        transform=ax_ad.transAxes, va="top", ha="left", fontsize=FS_MIN,
        color="#1b4332",
    )

    for ax, letter in [(ax_pca, "a"), (ax_ad, "b")]:
        ax.text(
            -0.15, 1.07, letter, transform=ax.transAxes,
            fontsize=FS_BASE + 1, fontweight="bold", ha="left", va="top",
        )

    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=3, fontsize=FS_MIN, frameon=False, handletextpad=0.35,
        columnspacing=1.1,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    base, _ = os.path.splitext(output_path)
    fig.savefig(base + ".svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/pic50_clean.csv")
    parser.add_argument("--out_dir", default="results/applicability_domain")
    parser.add_argument(
        "--deduplicate_structures", action="store_true",
        help="Sensitivity analysis: average pIC50 across repeated canonical SMILES "
             "before fitting the structural domain.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = ["Canonical SMILES", "pIC50"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    prepared = data[required].dropna().copy()
    if args.deduplicate_structures:
        prepared = (
            prepared.groupby("Canonical SMILES", as_index=False, sort=False)["pIC50"]
            .mean()
        )
    train_smiles = prepared["Canonical SMILES"].tolist()
    summary, points, plot_info = fit_descriptor_ad(
        train_smiles, EXT_SMILES
    )
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "descriptor_pca_ad_summary.csv")
    points_path = os.path.join(args.out_dir, "descriptor_pca_ad_points.csv")
    figure_path = os.path.join(args.out_dir, "descriptor_pca_ad.png")
    summary.to_csv(summary_path, index=False)
    points.to_csv(points_path, index=False)
    plot_descriptor_ad(summary, points, plot_info, figure_path)

    print(summary[[
        "Compound", "T2_fraction_of_limit", "Q_fraction_of_limit",
        "Inside_empirical_95pct_PCA_AD", "PC95_NN_training_percentile",
        "Maximum_ECFP4_Tanimoto",
    ]].to_string(index=False))
    print(f"\nOutputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
