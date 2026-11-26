#!/usr/bin/env python
"""
chemspace_viz.py  —  Chemical-space visualisation for NMDA antagonists
=======================================================================
Produces PCA, t-SNE and UMAP embeddings of the pKi and pIC50 training sets
across three feature representations (Desc, ECFP, FCFP).  For every
combination a figure is generated showing:

  • Training compounds coloured by activity value
  • 4 external test compounds overlaid (star markers, labelled)
  • SALI activity-cliff pairs connected by coloured lines
  • A zoom inset centred on the external compounds
  • Molecule-structure sub-panels for the nearest training neighbours
    of each external point

Usage
-----
  python chemspace_viz.py                  # uses same data paths as qsar_nmda.py
  python chemspace_viz.py --no_umap        # skip UMAP (faster, no umap-learn needed)
  python chemspace_viz.py --sali_threshold 6.0   # override cliff threshold

Outputs (written to ./results/chemspace/)
-----------------------------------------
  {pki,pic50}_{desc,fp,fcfp}_{pca,tsne,umap}.png  (+ .svg)
  — 3 methods × 3 representations × 2 datasets = up to 18 figures

Dependencies (in addition to qsar_nmda.py requirements)
--------------------------------------------------------
  pip install umap-learn          # optional, skip with --no_umap
"""

import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from io import BytesIO

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")

import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from external_data import EXT_LABELS, EXT_PIC50, EXT_SMILES

try:
    import umap
    import umap.umap_ as _umap_core
    import sklearn.utils.validation as _skval

    # umap does `from sklearn.utils.validation import check_array` so patching
    # the sklearn module attribute is not enough — we must patch the name
    # inside umap's own namespace after it has already been imported.
    _orig_check_array = _skval.check_array
    def _patched_check_array(array, *args, **kwargs):
        kwargs.pop("ensure_all_finite", None)
        return _orig_check_array(array, *args, **kwargs)

    _skval.check_array   = _patched_check_array   # for anything that re-imports later
    _umap_core.check_array = _patched_check_array  # patch the already-bound name in umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("✗ umap-learn not found — install with: pip install umap-learn")

try:
    from adjustText import adjust_text as _adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONFIGURATION  (mirrors qsar_nmda.py)
# ─────────────────────────────────────────────────────────────────────────────
SEED           = 42
FP_RADIUS      = 2
FP_SIZE        = 2048
SALI_THRESHOLD = 5.5      # overridable via --sali_threshold
OUT_DIR        = os.path.join("results", "chemspace")

# Colour palette for cliff lines (cycle through these)
CLIFF_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#8338ec",
                "#fb8500", "#023e8a", "#606c38", "#bc6c25"]

# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION FIGURE GEOMETRY  (mirrors qsar_nmda_last.py)
# -----------------------------------------------------------------------------
# Figures are drawn at the width they are actually placed at in the manuscript,
# so no rescaling happens in the word processor and text set at FS_BASE prints
# at FS_BASE.  Journal text width: 165 mm = 6.5 in.
# ─────────────────────────────────────────────────────────────────────────────
FIG_W2   = 6.50    # in — full text width (165 mm)
FIG_W1   = 3.15    # in — single column (80 mm)
FS_BASE  = 8       # titles, axis labels, series identity
FS_ANNOT = 7       # legend and annotation text
FS_TICK  = 6       # tick labels
FS_MIN   = 6       # hard floor — nothing is drawn smaller than this

_FIG_RC = {
    "figure.dpi":        150,
    "savefig.dpi":       600,
    "font.size":         FS_BASE,
    "axes.titlesize":    FS_BASE,
    "axes.labelsize":    FS_BASE,
    "xtick.labelsize":   FS_TICK,
    "ytick.labelsize":   FS_TICK,
    "legend.fontsize":   FS_ANNOT,
    "figure.titlesize":  FS_BASE,
    "axes.linewidth":    0.6,
    "grid.linewidth":    0.4,
    "lines.linewidth":   1.0,
    "patch.linewidth":   0.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "legend.frameon":    False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "svg.fonttype":      "none",   # keep SVG text editable
    "pdf.fonttype":      42,
}


def apply_print_style():
    """Apply the manuscript figure geometry to matplotlib's global rcParams."""
    plt.rcParams.update(_FIG_RC)


apply_print_style()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURISATION
# ─────────────────────────────────────────────────────────────────────────────
MFPGEN  = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
FCFPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_SIZE,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
_desc_names = [d[0] for d in Descriptors.descList]
_calc       = MoleculeDescriptors.MolecularDescriptorCalculator(_desc_names)


def mol_to_desc(mol):
    return np.array(_calc.CalcDescriptors(mol), dtype=float)

def mol_to_fp(mol):
    return MFPGEN.GetCountFingerprintAsNumPy(mol).astype(float)

def mol_to_fcfp(mol):
    return FCFPGEN.GetCountFingerprintAsNumPy(mol).astype(float)

def featurise(smiles_list, feat_fn):
    rows, mask = [], []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            rows.append(feat_fn(mol))
            mask.append(True)
        except Exception:
            rows.append(None)
            mask.append(False)
    good = np.array(mask, dtype=bool)
    X = np.vstack([r for r, g in zip(rows, good) if g])
    return X, good

def clean(X):
    return np.where(np.isfinite(X), X, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SALI
# ─────────────────────────────────────────────────────────────────────────────
def tanimoto_matrix(mols):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_SIZE)
           for m in mols if m is not None]
    n = len(fps)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            T[i, j] = T[j, i] = s
    return T

def compute_sali(smiles_list, y, threshold):
    """Return list of (i, j, sali_score) for pairs above threshold."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    T    = tanimoto_matrix(mols)
    y    = np.asarray(y)
    n    = len(mols)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - T[i, j]
            if d > 1e-6:
                score = abs(y[i] - y[j]) / d
                if score > threshold:
                    pairs.append((i, j, score))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_datasets():
    """Load pKi and pIC50 CSVs the same way qsar_nmda.py does."""
    # Try to find the data files relative to the script location
    search_dirs = [".", "..", "data"]
    pki_df = pic50_df = None
    for d in search_dirs:
        for fname in os.listdir(d) if os.path.isdir(d) else []:
            fl = fname.lower()
            full = os.path.join(d, fname)
            if "ki" in fl and fl.endswith(".csv") and pki_df is None:
                try:
                    pki_df = pd.read_csv(full)
                    print(f"  pKi  ← {full}")
                except Exception:
                    pass
            if "ic50" in fl and fl.endswith(".csv") and pic50_df is None:
                try:
                    pic50_df = pd.read_csv(full)
                    print(f"  pIC50← {full}")
                except Exception:
                    pass
    if pki_df is None or pic50_df is None:
        raise FileNotFoundError(
            "Could not find pKi / pIC50 CSV files. "
            "Run from the same directory as qsar_nmda.py.")
    return pki_df, pic50_df


def prep(df):
    """Mirror prep_dataset from qsar_nmda.py; returns (smiles, y, X_desc, X_fp, X_fcfp)."""
    smi_col = "Canonical SMILES"
    y_col   = [c for c in df.columns
               if c.lower().startswith("p") and c != smi_col][0]
    df = df[[smi_col, y_col]].dropna().copy()
    df.columns = ["smiles", "y"]

    X_fp_raw, mask = featurise(df["smiles"], mol_to_fp)
    df = df[mask].reset_index(drop=True)

    X_desc_raw, _ = featurise(df["smiles"], mol_to_desc)
    X_fcfp_raw, _ = featurise(df["smiles"], mol_to_fcfp)

    return (df["smiles"].tolist(), df["y"].values,
            clean(X_desc_raw), clean(X_fp_raw), clean(X_fcfp_raw))


# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────────────────
def scale_and_reduce(X_train, X_ext, method, seed=SEED):
    """
    Scale X_train, project X_ext into same space, then reduce to 2D.
    Returns (coords_train [N×2], coords_ext [M×2]).
    """
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X_train)
    Xe = scaler.transform(X_ext)

    if method == "PCA":
        red = PCA(n_components=2, random_state=seed)
        Zs  = red.fit_transform(Xs)
        Ze  = red.transform(Xe)

    elif method == "tSNE":
        # Fit t-SNE on train+ext together so externals are embedded consistently
        n_train = Xs.shape[0]
        combined = np.vstack([Xs, Xe])
        # First reduce dimensionality with PCA to speed up t-SNE
        n_pca = min(50, combined.shape[1], combined.shape[0] - 1)
        pca   = PCA(n_components=n_pca, random_state=seed)
        combined_pca = pca.fit_transform(combined)
        tsne  = TSNE(n_components=2, perplexity=min(30, n_train // 3),
                     random_state=seed, n_iter=1000, init="pca",
                     learning_rate="auto")
        Z_all = tsne.fit_transform(combined_pca)
        Zs, Ze = Z_all[:n_train], Z_all[n_train:]

    elif method == "UMAP":
        if not HAS_UMAP:
            return None, None
        n_train = Xs.shape[0]
        combined = np.vstack([Xs, Xe])
        n_pca = min(50, combined.shape[1], combined.shape[0] - 1)
        pca   = PCA(n_components=n_pca, random_state=seed)
        combined_pca = pca.fit_transform(combined)
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            random_state=seed)
        Z_all = reducer.fit_transform(combined_pca)
        Zs, Ze = Z_all[:n_train], Z_all[n_train:]

    else:
        raise ValueError(f"Unknown method: {method}")

    return Zs, Ze


# ─────────────────────────────────────────────────────────────────────────────
# MOLECULE RENDERING HELPER
# ─────────────────────────────────────────────────────────────────────────────
def mol_to_image(smi, size=(180, 140)):
    """Render a SMILES to a PIL image (RGB numpy array)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    from PIL import Image
    img = Image.open(BytesIO(drawer.GetDrawingText()))
    return np.array(img.convert("RGB"))


# ─────────────────────────────────────────────────────────────────────────────
# NEAREST-NEIGHBOUR FINDER  (Tanimoto on ECFP4)
# ─────────────────────────────────────────────────────────────────────────────
def nearest_neighbours(ext_smiles, train_smiles, train_y, k=2):
    """
    For each external SMILES find the k most similar training compounds
    by Tanimoto (ECFP4).  Returns list of lists of (smi, y, tanimoto).
    """
    train_fps = [AllChem.GetMorganFingerprintAsBitVect(
                     Chem.MolFromSmiles(s), FP_RADIUS, nBits=FP_SIZE)
                 for s in train_smiles]
    results = []
    for smi in ext_smiles:
        mol = Chem.MolFromSmiles(smi)
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_SIZE)
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        top_idx = np.argsort(sims)[::-1][:k]
        results.append([(train_smiles[i], train_y[i], sims[i]) for i in top_idx])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def savefig(path, fig=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    (fig or plt).savefig(path, dpi=600, bbox_inches="tight")
    base, _ = os.path.splitext(path)
    (fig or plt).savefig(base + ".svg", bbox_inches="tight")
    if fig:
        plt.close(fig)
    else:
        plt.close()
    print(f"  → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CORE PLOTTING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
_CMAP       = "plasma"
_EXT_MARKER = "*"
_EXT_SIZE   = 280


def make_embedding_figure(
    Zs, Ze,              # (N,2) train coords, (M,2) ext coords
    y_train,             # (N,) activity values for colour
    cliff_pairs,         # list of (i, j, score) from compute_sali
    train_smiles,        # for nearest-neighbour lookup
    ext_smiles,
    ext_labels,
    ext_y,
    nn_results,          # output of nearest_neighbours()
    feat_label,          # "Desc" / "FP" / "FCFP"
    method,              # "PCA" / "tSNE" / "UMAP"
    dataset_label,       # "pIC50" / "pKi"
    out_path,
    sali_threshold,
    max_cliff_lines=60,  # cap lines so figure stays readable
):
    # ── layout ────────────────────────────────────────────────────────────────
    # Main figure: [left big panel] + [right column of structure sub-panels]
    n_ext = len(ext_smiles)
    n_nn  = 2                          # nearest neighbours per external point
    n_struct_panels = n_ext * (1 + n_nn)   # ext structure + its nn structures

    fig = plt.figure(figsize=(18, 9))
    gs_outer = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[3, 1], wspace=0.05)

    # Left: main scatter + zoom inset (managed manually)
    ax_main = fig.add_subplot(gs_outer[0])

    # Right: grid of structure thumbnails  (n_ext rows × (1+n_nn) cols)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        n_ext, 1 + n_nn,
        subplot_spec=gs_outer[1],
        hspace=0.45, wspace=0.1)
    struct_axes = [[fig.add_subplot(gs_right[r, c])
                    for c in range(1 + n_nn)]
                   for r in range(n_ext)]

    # ── main scatter ──────────────────────────────────────────────────────────
    norm  = Normalize(vmin=y_train.min(), vmax=y_train.max())
    cmap  = plt.get_cmap(_CMAP)

    # SALI cliff lines (draw before points so they sit behind)
    sorted_pairs = sorted(cliff_pairs, key=lambda t: -t[2])
    for k_line, (i, j, score) in enumerate(sorted_pairs[:max_cliff_lines]):
        col = CLIFF_COLORS[k_line % len(CLIFF_COLORS)]
        ax_main.plot([Zs[i, 0], Zs[j, 0]], [Zs[i, 1], Zs[j, 1]],
                     color=col, lw=0.6, alpha=0.45, zorder=1)

    # Training scatter
    sc = ax_main.scatter(Zs[:, 0], Zs[:, 1],
                         c=y_train, cmap=_CMAP, norm=norm,
                         s=28, alpha=0.75, linewidths=0, zorder=2)

    # External points
    ext_colors = ["#1b4332", "#6a0572", "#7b2d00", "#003566"]
    for k_e, (ze, lbl, ey) in enumerate(zip(Ze, ext_labels, ext_y)):
        col = ext_colors[k_e % len(ext_colors)]
        ax_main.scatter(ze[0], ze[1],
                        marker=_EXT_MARKER, s=_EXT_SIZE,
                        color=col, edgecolors="white", linewidths=0.8,
                        zorder=5, label=lbl)
        ax_main.annotate(
            lbl,
            (ze[0], ze[1]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, fontweight="bold", color=col,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Colourbar
    cb = fig.colorbar(sc, ax=ax_main, fraction=0.03, pad=0.02)
    cb.set_label(dataset_label, fontsize=9)

    ax_main.set_xlabel(f"{method} 1", fontsize=10)
    ax_main.set_ylabel(f"{method} 2", fontsize=10)
    ax_main.set_title(
        f"{dataset_label} chemical space — {feat_label} / {method}\n"
        f"Colour = activity  |  ★ = external  |  "
        f"lines = SALI cliff pairs (>{sali_threshold:.0f})",
        fontsize=10)
    ax_main.legend(title="External", loc="upper left",
                   fontsize=8, title_fontsize=9, framealpha=0.85)

    # ── zoom inset centred on external compounds ───────────────────────────────
    if Ze.shape[0] > 0:
        x_pad = (Zs[:, 0].max() - Zs[:, 0].min()) * 0.07
        y_pad = (Zs[:, 1].max() - Zs[:, 1].min()) * 0.07
        ext_xmin, ext_xmax = Ze[:, 0].min() - x_pad, Ze[:, 0].max() + x_pad
        ext_ymin, ext_ymax = Ze[:, 1].min() - y_pad, Ze[:, 1].max() + y_pad

        # Expand if all externals are coincident
        if ext_xmax - ext_xmin < 1e-3:
            ext_xmin -= 0.5; ext_xmax += 0.5
        if ext_ymax - ext_ymin < 1e-3:
            ext_ymin -= 0.5; ext_ymax += 0.5

        # inset position in axes fraction (lower-right)
        ax_ins = ax_main.inset_axes([0.60, 0.01, 0.38, 0.38])

        # draw same cliff lines in inset
        for k_line, (i, j, score) in enumerate(sorted_pairs[:max_cliff_lines]):
            col = CLIFF_COLORS[k_line % len(CLIFF_COLORS)]
            ax_ins.plot([Zs[i, 0], Zs[j, 0]], [Zs[i, 1], Zs[j, 1]],
                        color=col, lw=0.5, alpha=0.35, zorder=1)

        ax_ins.scatter(Zs[:, 0], Zs[:, 1],
                       c=y_train, cmap=_CMAP, norm=norm,
                       s=14, alpha=0.7, linewidths=0, zorder=2)

        for k_e, (ze, lbl) in enumerate(zip(Ze, ext_labels)):
            col = ext_colors[k_e % len(ext_colors)]
            ax_ins.scatter(ze[0], ze[1],
                           marker=_EXT_MARKER, s=120,
                           color=col, edgecolors="white", linewidths=0.7,
                           zorder=5)
            ax_ins.annotate(lbl, (ze[0], ze[1]),
                            textcoords="offset points", xytext=(3, 2),
                            fontsize=6, fontweight="bold", color=col,
                            path_effects=[pe.withStroke(linewidth=1.5,
                                                        foreground="white")])

        ax_ins.set_xlim(ext_xmin, ext_xmax)
        ax_ins.set_ylim(ext_ymin, ext_ymax)
        ax_ins.set_xticks([]); ax_ins.set_yticks([])
        ax_ins.set_title("Zoom — ext. region", fontsize=7, pad=2)
        ax_ins.spines[:].set_linewidth(1.2)

        # Draw connector lines between the dashed rectangle (zoom region in
        # main axes) and the inset box.  For each of the four corner pairs,
        # pick the two with the shortest display-space distance — those are
        # the corners that genuinely bridge the gap.
        try:
            fig.canvas.draw()   # finalise layout so transforms are valid

            # Four corners of the zoomed region in main-axes data coords
            rect_corners_data = [
                [ext_xmin, ext_ymin],
                [ext_xmax, ext_ymin],
                [ext_xmax, ext_ymax],
                [ext_xmin, ext_ymax],
            ]

            # Same four corners of the inset axes in display coords,
            # then converted to main-axes data coords for the arrow target
            ins_bbox  = ax_ins.get_position()            # figure fraction
            fig_w, fig_h = fig.get_size_inches() * fig.dpi
            inset_corners_disp = [
                (ins_bbox.x0 * fig_w, ins_bbox.y0 * fig_h),   # BL
                (ins_bbox.x1 * fig_w, ins_bbox.y0 * fig_h),   # BR
                (ins_bbox.x1 * fig_w, ins_bbox.y1 * fig_h),   # TR
                (ins_bbox.x0 * fig_w, ins_bbox.y1 * fig_h),   # TL
            ]
            inset_corners_main = [
                ax_main.transData.inverted().transform(
                    ax_main.get_figure().transFigure.transform(
                        (d[0] / fig_w, d[1] / fig_h)))
                for d in inset_corners_disp
            ]

            # For every (rect_corner, inset_corner) pair at the same index,
            # compute display-space distance and pick the two shortest.
            rect_corners_disp = [
                ax_main.transData.transform(rc)
                for rc in rect_corners_data
            ]
            dists = [
                np.hypot(rc[0] - ic[0], rc[1] - ic[1])
                for rc, ic in zip(rect_corners_disp, inset_corners_disp)
            ]
            close_idx = np.argsort(dists)[:2]

            for idx in close_idx:
                rc_data = rect_corners_data[idx]
                ic_main = inset_corners_main[idx]
                con = matplotlib.patches.FancyArrowPatch(
                    rc_data, ic_main,
                    arrowstyle="-",
                    color="0.45", lw=0.9, ls="--",
                    transform=ax_main.transData,
                    zorder=6, clip_on=False)
                ax_main.add_patch(con)

            # Dashed rectangle around the zoomed region in main axes
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (ext_xmin, ext_ymin),
                ext_xmax - ext_xmin, ext_ymax - ext_ymin,
                linewidth=1.0, edgecolor="0.45", facecolor="none",
                ls="--", zorder=6)
            ax_main.add_patch(rect)

        except Exception:
            pass

    # ── structure sub-panels ──────────────────────────────────────────────────
    for r, (ext_smi, lbl, nn_list) in enumerate(
            zip(ext_smiles, ext_labels, nn_results)):
        # Column 0 — the external compound itself
        ax_e = struct_axes[r][0]
        img_e = mol_to_image(ext_smi, size=(160, 120))
        if img_e is not None:
            ax_e.imshow(img_e)
        ax_e.axis("off")
        ax_e.set_title(f"★ {lbl}", fontsize=7, fontweight="bold",
                       color=ext_colors[r % len(ext_colors)], pad=2)

        # Columns 1+ — nearest training neighbours
        for c, (nn_smi, nn_y, sim) in enumerate(nn_list):
            ax_n = struct_axes[r][1 + c]
            img_n = mol_to_image(nn_smi, size=(160, 120))
            if img_n is not None:
                ax_n.imshow(img_n)
            ax_n.axis("off")
            ax_n.set_title(
                f"NN{c+1}  sim={sim:.2f}\n{dataset_label}={nn_y:.2f}",
                fontsize=6.5, pad=2)

    # Column header for structure panel
    fig.text(0.73, 0.96,
             "External compounds & nearest training neighbours",
             ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    savefig(out_path, fig=fig)




# ─────────────────────────────────────────────────────────────────────────────
# MAIN-TEXT FIGURE 2 — three panels on one canvas
# -----------------------------------------------------------------------------
# (a) chemical-space embedding with the external compounds and SALI cliff pairs
# (b) distribution of per-compound maximum SALI, with the cliff threshold
# (c) activity landscape: max SALI against activity
# Drawn at FIG_W2 (165 mm) so it is placed in the manuscript without rescaling.
# The external-compound / nearest-neighbour structure grid that used to occupy
# the right-hand third of the embedding figure is now a separate SI figure
# (make_nn_grid_figure), which is what removes most of the white space.
# ─────────────────────────────────────────────────────────────────────────────
_EXT_COLORS = ["#1b4332", "#6a0572", "#7b2d00", "#003566"]
_CLIFF_LINE_COLOR = "#9aa0a6"     # single neutral grey: the lines mark pairs,
                                  # their colour carries no information


def sali_max_per_compound(smiles_list, y):
    """Maximum SALI score of each compound against all others."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    T = tanimoto_matrix(mols)
    y = np.asarray(y, dtype=float)
    n = len(mols)
    D = 1.0 - T
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.abs(y[:, None] - y[None, :]) / D
    S[~np.isfinite(S)] = 0.0
    np.fill_diagonal(S, 0.0)
    return S.max(axis=1)


def _panel_letter(ax, letter, dx=-0.11, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS_BASE + 1, fontweight="bold", va="top", ha="left")


def make_figure2_composite(
    Zs, Ze, y_train, cliff_pairs, ext_labels, ext_y,
    max_sali, cliff_mask, feat_label, method, dataset_label,
    out_path, sali_threshold, max_cliff_lines=60,
):
    apply_print_style()
    fig = plt.figure(figsize=(FIG_W2, 3.6))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.20, 0.72, 1.0], height_ratios=[1, 1],
        wspace=0.30, hspace=0.55,
        left=0.075, right=0.985, top=0.93, bottom=0.115)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_zoom = fig.add_subplot(gs[:, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 2])

    # ── (a) embedding ─────────────────────────────────────────────────────────
    norm = Normalize(vmin=float(np.min(y_train)), vmax=float(np.max(y_train)))
    sorted_pairs = sorted(cliff_pairs, key=lambda t: -t[2])[:max_cliff_lines]
    for i, j, _ in sorted_pairs:
        ax_a.plot([Zs[i, 0], Zs[j, 0]], [Zs[i, 1], Zs[j, 1]],
                  color=_CLIFF_LINE_COLOR, lw=0.4, alpha=0.7, zorder=1)

    sc = ax_a.scatter(Zs[:, 0], Zs[:, 1], c=y_train, cmap=_CMAP, norm=norm,
                      s=9, alpha=0.85, linewidths=0, zorder=2)

    # The four external compounds sit almost on top of one another in this
    # embedding, so they are marked (not labelled) in the main axes and named
    # individually in the zoom inset, where they are resolved.
    for k, ze in enumerate(Ze):
        ax_a.scatter(ze[0], ze[1], marker=_EXT_MARKER, s=70,
                     color=_EXT_COLORS[k % len(_EXT_COLORS)],
                     edgecolors="white", linewidths=0.6, zorder=5)

    ax_a.set_xlabel(f"{method} 1")
    ax_a.set_ylabel(f"{method} 2")
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for s in ("top", "right"):
        ax_a.spines[s].set_visible(False)

    # colour bar docked inside the axes so it costs no extra canvas width
    cax = ax_a.inset_axes([0.015, 0.90, 0.30, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=FS_MIN, length=1.5, pad=1)
    cb.ax.set_title(dataset_label, fontsize=FS_TICK, pad=2)
    cb.outline.set_linewidth(0.4)

    # ── (a) zoom inset on the external region ────────────────────────────────
    if Ze.shape[0] > 0:
        # Zoom window: scaled to the spread of the external points themselves
        # (with a floor, in case they are coincident) rather than to the full
        # embedding, so the inset shows resolved compounds and not padding.
        span = max(np.ptp(Ze[:, 0]), np.ptp(Ze[:, 1]))
        full = max(np.ptp(Zs[:, 0]), np.ptp(Zs[:, 1]))
        # The window starts from the spread of the external points (a fixed
        # fraction of the full embedding makes the inset almost all padding,
        # since the four compounds sit close together), then grows until enough
        # training points fall inside for the inset to show local context
        # rather than an empty box.
        # The window is sized to the external compounds themselves: they lie
        # within 1% of the full embedding span of one another, so a window set as
        # a fraction of that span (or grown until some number of training points
        # fall inside) collapses them into a single marker.  Their own spread
        # then fills a bit over half the window, and whichever training
        # neighbours lie that close come along for context.
        cx, cy = Ze[:, 0].mean(), Ze[:, 1].mean()
        pad = max(span * 0.90, full * 0.004)
        x0, x1 = cx - pad, cx + pad
        y0, y1 = cy - pad, cy + pad

        # Give the magnified region its own axes rather than drawing an inset on
        # top of the full embedding.  The complete UMAP therefore remains
        # visible, while the local external-compound geometry is still resolved.
        ax_ins = ax_zoom
        for i, j, _ in sorted_pairs:
            ax_ins.plot([Zs[i, 0], Zs[j, 0]], [Zs[i, 1], Zs[j, 1]],
                        color=_CLIFF_LINE_COLOR, lw=0.35, alpha=0.6, zorder=1)
        ax_ins.scatter(Zs[:, 0], Zs[:, 1], c=y_train, cmap=_CMAP, norm=norm,
                       s=18, alpha=0.8, linewidths=0, zorder=2)
        # The compounds are named once by a legend on the main axes rather than
        # by text inside the inset: four labels and four near-coincident markers
        # cannot both be legible in a box this size, whichever way the text is
        # placed.
        ext_handles = []
        for k, ze in enumerate(Ze):
            col = _EXT_COLORS[k % len(_EXT_COLORS)]
            ax_ins.scatter(ze[0], ze[1], marker=_EXT_MARKER, s=55,
                           color=col, edgecolors="white", linewidths=0.5,
                           zorder=5)
            ext_handles.append(mlines.Line2D(
                [], [], marker=_EXT_MARKER, ls="none", markersize=5,
                markerfacecolor=col, markeredgecolor="white",
                markeredgewidth=0.4, label=ext_labels[k]))
        ax_ins.set_xlim(x0, x1); ax_ins.set_ylim(y0, y1)
        ax_ins.legend(handles=ext_handles, loc="lower center",
                    bbox_to_anchor=(0.5, 0.015),
                    bbox_transform=ax_ins.transAxes,
                    ncol=1, fontsize=FS_MIN,
                    title="External compounds", title_fontsize=FS_MIN,
                    handletextpad=0.25, borderpad=0.35, labelspacing=0.28,
                    framealpha=0.85, frameon=True)
        ax_ins.set_xticks([]); ax_ins.set_yticks([])
        ax_ins.set_facecolor("white")
        for s in ax_ins.spines.values():
            s.set_visible(True); s.set_linewidth(0.6); s.set_color("0.35")
        ax_ins.set_title("External-compound region (zoom)",
                         fontsize=FS_TICK, pad=3)
        from matplotlib.patches import Rectangle
        ax_a.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            fill=False, edgecolor="0.35", linewidth=0.7,
            linestyle="--", zorder=4))
        ax_a.annotate("region at right", xy=(x1, y1), xytext=(3, 3),
                      textcoords="offset points", fontsize=FS_MIN,
                      color="0.35", ha="left", va="bottom")

    # ── (b) SALI distribution ─────────────────────────────────────────────────
    ax_b.hist(max_sali, bins=40, color="#4C72B0", edgecolor="white",
              linewidth=0.3)
    ax_b.axvline(sali_threshold, color="#C44E52", ls="--", lw=0.9)
    ax_b.annotate(f"cliff threshold = {sali_threshold:g}",
                  xy=(sali_threshold, ax_b.get_ylim()[1] * 0.92),
                  xytext=(4, 0), textcoords="offset points",
                  fontsize=FS_TICK, color="#C44E52", va="top")
    ax_b.set_xlabel("Max SALI per compound")
    ax_b.set_ylabel("Compounds")

    # ── (c) activity landscape ────────────────────────────────────────────────
    cm = np.asarray(cliff_mask, dtype=bool)
    ax_c.scatter(np.asarray(y_train)[~cm], np.asarray(max_sali)[~cm],
                 c="#4C72B0", s=8, alpha=0.65, linewidths=0,
                 label="non-cliff")
    ax_c.scatter(np.asarray(y_train)[cm], np.asarray(max_sali)[cm],
                 c="#C44E52", s=8, alpha=0.8, linewidths=0,
                 label=f"cliff (SALI > {sali_threshold:g})")
    ax_c.axhline(sali_threshold, color="#C44E52", ls="--", lw=0.9)
    ax_c.set_xlabel(dataset_label)
    ax_c.set_ylabel("Max SALI")
    ax_c.legend(loc="upper left", fontsize=FS_TICK, handletextpad=0.3,
                borderpad=0.1, labelspacing=0.25)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        _panel_letter(ax, letter, dx=-0.075 if ax is ax_a else -0.20)

    savefig(out_path, fig=fig)


def make_nn_grid_figure(ext_smiles, ext_labels, nn_results,
                        dataset_label, out_path):
    """SI figure: each external compound beside its nearest training neighbours."""
    apply_print_style()
    n_ext = len(ext_smiles)
    n_col = 1 + (len(nn_results[0]) if nn_results else 2)
    fig, axes = plt.subplots(n_ext, n_col,
                             figsize=(FIG_W2, 1.25 * n_ext))
    axes = np.atleast_2d(axes)

    for r, (smi, lbl, nn_list) in enumerate(zip(ext_smiles, ext_labels,
                                                nn_results)):
        col = _EXT_COLORS[r % len(_EXT_COLORS)]
        ax = axes[r, 0]
        img = mol_to_image(smi, size=(320, 240))
        if img is not None:
            ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{lbl}", fontsize=FS_ANNOT, fontweight="bold",
                     color=col, pad=2)

        for c, (nn_smi, nn_y, sim) in enumerate(nn_list):
            axn = axes[r, 1 + c]
            imgn = mol_to_image(nn_smi, size=(320, 240))
            if imgn is not None:
                axn.imshow(imgn)
            axn.axis("off")
            axn.set_title(f"NN{c+1}   T = {sim:.2f}   {dataset_label} = {nn_y:.2f}",
                          fontsize=FS_TICK, pad=2)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.05, hspace=0.30)
    savefig(out_path, fig=fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(run_umap=True, sali_threshold=SALI_THRESHOLD):
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[1] Loading datasets …")
    pki_df, pic50_df = load_datasets()

    datasets = {}
    print("\n[2] Featurising …")
    for label, df in [("pki", pki_df), ("pic50", pic50_df)]:
        print(f"  {label}:")
        smiles, y, X_desc, X_fp, X_fcfp = prep(df)
        datasets[label] = dict(
            smiles=smiles, y=y,
            X_desc=X_desc, X_fp=X_fp, X_fcfp=X_fcfp)
        print(f"    {len(smiles)} compounds, "
              f"desc={X_desc.shape[1]}  fp={X_fp.shape[1]}  fcfp={X_fcfp.shape[1]}")

    print("\n[3] Featurising external set …")
    ext_mols = [Chem.MolFromSmiles(s) for s in EXT_SMILES]
    X_ext = {
        "Desc": clean(np.vstack([mol_to_desc(m) for m in ext_mols])),
        "FP":   clean(np.vstack([mol_to_fp(m)   for m in ext_mols])),
        "FCFP": clean(np.vstack([mol_to_fcfp(m) for m in ext_mols])),
    }

    feat_map = {"Desc": "X_desc", "FP": "X_fp", "FCFP": "X_fcfp"}

    methods = ["PCA", "tSNE"]
    if run_umap and HAS_UMAP:
        methods.append("UMAP")

    n_total = len(datasets) * len(feat_map) * len(methods)
    done = 0

    for ds_label, ds in datasets.items():
        smiles  = ds["smiles"]
        y       = ds["y"]
        y_label = "pIC50" if "ic50" in ds_label else "pKi"

        print(f"\n[4] SALI pairs ({y_label}, threshold={sali_threshold}) …")
        cliff_pairs = compute_sali(smiles, y, sali_threshold)
        print(f"    {len(cliff_pairs)} cliff pairs found")

        print(f"  Nearest-neighbour lookup for {y_label} …")
        nn_results = nearest_neighbours(EXT_SMILES, smiles, y, k=2)

        for feat_label, xkey in feat_map.items():
            X_train = ds[xkey]
            X_e     = X_ext[feat_label]

            for method in methods:
                done += 1
                print(f"\n  [{done}/{n_total}] {y_label} / {feat_label} / {method} …")

                Zs, Ze = scale_and_reduce(X_train, X_e, method)
                if Zs is None:
                    print("    skipped (UMAP unavailable)")
                    continue

                fname   = f"{ds_label}_{feat_label.lower()}_{method.lower()}.png"
                out_path = os.path.join(OUT_DIR, fname)

                make_embedding_figure(
                    Zs=Zs, Ze=Ze,
                    y_train=y,
                    cliff_pairs=cliff_pairs,
                    train_smiles=smiles,
                    ext_smiles=EXT_SMILES,
                    ext_labels=EXT_LABELS,
                    ext_y=EXT_PIC50,
                    nn_results=nn_results,
                    feat_label=feat_label,
                    method=method,
                    dataset_label=y_label,
                    out_path=out_path,
                    sali_threshold=sali_threshold,
                )

    print(f"\nDone — figures written to {OUT_DIR}/")




def main_figure2(sali_threshold=SALI_THRESHOLD,
                 feat_label="FCFP", method="UMAP",
                 out_dir=None):
    """
    Build the main-text Figure 2 composite (and the SI nearest-neighbour grid)
    for the pIC50 dataset only.  Kept separate from main() so the manuscript
    figure can be regenerated without recomputing the full method x embedding
    sweep.
    """
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1] Loading datasets …")
    _, pic50_df = load_datasets()
    smiles, y, X_desc, X_fp, X_fcfp = prep(pic50_df)
    X_train = {"Desc": X_desc, "FP": X_fp, "FCFP": X_fcfp}[feat_label]
    print(f"    {len(smiles)} compounds, {feat_label} dim = {X_train.shape[1]}")

    ext_mols = [Chem.MolFromSmiles(s) for s in EXT_SMILES]
    X_e = {
        "Desc": clean(np.vstack([mol_to_desc(m) for m in ext_mols])),
        "FP":   clean(np.vstack([mol_to_fp(m)   for m in ext_mols])),
        "FCFP": clean(np.vstack([mol_to_fcfp(m) for m in ext_mols])),
    }[feat_label]

    print(f"\n[2] SALI (threshold = {sali_threshold}) …")
    cliff_pairs = compute_sali(smiles, y, sali_threshold)
    max_sali    = sali_max_per_compound(smiles, y)
    cliff_mask  = max_sali > sali_threshold
    print(f"    {len(cliff_pairs)} cliff pairs, "
          f"{int(cliff_mask.sum())}/{len(smiles)} compounds on a cliff")

    print(f"\n[3] Embedding ({feat_label} / {method}) …")
    Zs, Ze = scale_and_reduce(X_train, X_e, method)
    if Zs is None:
        raise RuntimeError(f"{method} unavailable — cannot build Figure 2")

    print("\n[4] Nearest neighbours …")
    nn_results = nearest_neighbours(EXT_SMILES, smiles, y, k=2)

    print("\n[5] Drawing …")
    make_figure2_composite(
        Zs=Zs, Ze=Ze, y_train=y, cliff_pairs=cliff_pairs,
        ext_labels=EXT_LABELS, ext_y=EXT_PIC50,
        max_sali=max_sali, cliff_mask=cliff_mask,
        feat_label=feat_label, method=method, dataset_label="pIC50",
        out_path=os.path.join(out_dir, "figure2_composite.png"),
        sali_threshold=sali_threshold)

    make_nn_grid_figure(
        EXT_SMILES, EXT_LABELS, nn_results, "pIC50",
        os.path.join(out_dir, "si_external_nearest_neighbours.png"))

    print(f"\nDone — Figure 2 written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chemical-space visualisation for NMDA antagonists")
    parser.add_argument("--no_umap", action="store_true",
                        help="Skip UMAP (faster; use if umap-learn is not installed)")
    parser.add_argument("--sali_threshold", type=float, default=SALI_THRESHOLD,
                        help=f"SALI cliff threshold (default: {SALI_THRESHOLD})")
    parser.add_argument("--figure2", action="store_true",
                        help="Build only the main-text Figure 2 composite "
                             "(pIC50 / FCFP / UMAP) and the SI "
                             "nearest-neighbour grid")
    parser.add_argument("--out_dir", default=None,
                        help=f"Output directory (default: {OUT_DIR})")
    args = parser.parse_args()

    if args.figure2:
        main_figure2(sali_threshold=args.sali_threshold,
                     out_dir=args.out_dir)
    else:
        main(run_umap=not args.no_umap,
             sali_threshold=args.sali_threshold)
