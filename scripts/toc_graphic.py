#!/usr/bin/env python
"""Table-of-contents / abstract graphic.

Left:  the matched pair that motivates the study -- two 2-aryl-2-amino-
       cyclohexanones identical in every respect except the ring position of
       the chlorine, with their measured pIC50 values.  Both are drawn on a
       common 2D template so that the only visual difference between the two
       panels is the chlorine; nothing else in the depiction moves.

Right: how often each molecular representation recovers the *direction* of an
       activity cliff, across the regressors benchmarked here.

Every number shown is read from the study's own result tables at build time --
the graphic contains no schematic or hand-drawn quantities.

Usage
-----
    python toc_graphic.py --out_dir results5x5_last
"""

import argparse
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from external_data import load_external_measurements

# ── canvas ──────────────────────────────────────────────────────────────────
# Prescribed by the publisher; the canvas must not be enlarged by a tight bbox.
FIG_W, FIG_H = 3.25, 1.75
DPI_TIF = 300

FS_TITLE = 8.0
FS_LABEL = 7.5
FS_TICK  = 6.5
FS_SMALL = 6.0

# ── the matched pair ────────────────────────────────────────────────────────
# Identical skeletons and identical N-ethyl substituents; the chlorine moves
# from the meta to the para position.  pIC50 values are the measured external
# values also used in the external-validation figures.
# Two tones per compound, taken from successive steps of the same RdBu
# diverging scheme so the palette family is unchanged and only lightness moves.
#   text -- the compound name, which has to stay readable as small bold type
#           (contrast against white 3.7 : 1 red, 3.4 : 1 blue)
#   fill -- the pastel atom highlight, chosen so a dark chlorine label reads
#           against it (10.6 : 1 red, 11.3 : 1 blue)
_EXTERNAL_BY_COMPOUND = load_external_measurements().set_index("compound")
PAIR = [
    (_EXTERNAL_BY_COMPOUND.loc["HC49", "smiles"], "meta-Cl",
     _EXTERNAL_BY_COMPOUND.loc["HC49", "mean_pIC50"], "#D6604D", "#F4A582"),
    (_EXTERNAL_BY_COMPOUND.loc["HC22", "smiles"], "para-Cl",
     _EXTERNAL_BY_COMPOUND.loc["HC22", "mean_pIC50"], "#4393C3", "#92C5DE"),
]
# RDKit labels chlorine in element green (#00cd00), which has almost the same
# luminance as these pastels -- 1.1 : 1 -- so the label would vanish into the
# highlight. Only chlorine is recoloured; O and N keep their element colours.
CL_LABEL_RGB = (0.10, 0.10, 0.10)

# ── molecule panel geometry ─────────────────────────────────────────────────
# Kept at module level because the atom-label size is derived from it: RDKit's
# fixedFontSize is in render-canvas pixels, so the size a reader actually sees
# depends on the scale factor from the render canvas into the panel.  Set as a
# pixel constant it silently came out at 2.8 pt, well under the 6 pt floor used
# for the article figures, which is what made the chlorine hard to read.
MOL_X, MOL_W = 0.012, 0.340
MOL_H = 0.390
LAB_X = MOL_X + MOL_W + 0.012
BANDS = [0.600, 0.020]                 # bottom edge of each structure panel
MOL_PX = (672, 400)                    # render canvas, ~2x the placed size
ATOM_LABEL_PT = 6.5                    # target effective size for O, NH, Cl


def atom_font_px(fig_w=None):
    """fixedFontSize, in render pixels, that puts atom labels at
    ATOM_LABEL_PT once the render canvas is scaled into the molecule panel.

    The panel is wider than it is tall relative to the render aspect, so the
    placed image is width-limited and the scale is set by the width alone.
    """
    fig_w = FIG_W if fig_w is None else fig_w
    placed_in = MOL_W * fig_w
    assert MOL_PX[0] / MOL_PX[1] > (MOL_W * FIG_W) / (MOL_H * FIG_H), \
        "render aspect must exceed panel aspect for the width-limited scale"
    return ATOM_LABEL_PT / 72.0 * MOL_PX[0] / placed_in
# Des-chloro skeleton -- the 2D template both members are drawn against.
PAIR_CORE = "CCNC1(c2ccccc2)CCCCC1=O"

REPR_ORDER  = ["Desc", "FP", "FCFP"]
REPR_LABEL  = {"Desc": "descriptors", "FP": "Morgan", "FCFP": "FCFP"}
REPR_COLOUR = {"Desc": "#4C72B0", "FP": "#DD8452", "FCFP": "#55A868"}


def draw_pair(size=MOL_PX):
    """Render both pair members against a shared 2D template.

    Returning a common template matters here: left to itself RDKit lays the
    N-alkyl chain out differently in each panel, which reads as a different
    substituent rather than the same one, and the meta -> para contrast the
    graphic exists to make is lost among spurious differences.
    """
    core = Chem.MolFromSmiles(PAIR_CORE)
    AllChem.Compute2DCoords(core)

    out = []
    for smi, _label, _pic50, _text_colour, fill_colour in PAIR:
        mol = Chem.MolFromSmiles(smi)
        AllChem.GenerateDepictionMatching2DStructure(mol, core)

        cl = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl"]
        rgb = matplotlib.colors.to_rgb(fill_colour)

        d = rdMolDraw2D.MolDraw2DCairo(*size)
        o = d.drawOptions()
        o.clearBackground = False
        o.fixedFontSize = int(round(atom_font_px()))
        # Bond width scales with the canvas so the drawing keeps its weight.
        o.bondLineWidth = 3
        o.highlightRadius = 0.42
        o.updateAtomPalette({17: CL_LABEL_RGB})
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, mol,
            highlightAtoms=cl,
            highlightAtomColors={i: rgb for i in cl})
        d.FinishDrawing()
        out.append(np.array(Image.open(io.BytesIO(d.GetDrawingText()))))
    return out


def direction_rates(out_dir):
    """Median share of cliff pairs whose direction each representation gets
    right, read from the study's directionality table."""
    path = os.path.join(out_dir, "cliff_directionality_pic50.csv")
    df = pd.read_csv(path)
    med = df.groupby("Features")["pct_correct"].median()
    n_models = int(df.groupby("Features")["Model"].nunique().max())
    n_pairs = int(df["n_pairs"].iloc[0])
    return [(r, float(med[r])) for r in REPR_ORDER], n_models, n_pairs


def build_figure(out_dir):
    rates, n_models, n_pairs = direction_rates(out_dir)
    imgs = draw_pair()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")

    # ── left: the matched pair ──────────────────────────────────────────────
    # The canvas is divided into three horizontal bands on this side: one per
    # structure, with a reserved band between them for the cliff caption.  The
    # caption is given its own band rather than being placed in the gap by eye,
    # which is how it ended up drawn over the lower structure.

    for (_smi, label, pic50, colour, _fill), im, y0 in zip(PAIR, imgs, BANDS):
        ax = fig.add_axes([MOL_X, y0, MOL_W, MOL_H])
        ax.imshow(im, interpolation="antialiased")
        ax.set_axis_off()
        # Name and potency ride together beside the structure, clear of it.
        fig.text(LAB_X, y0 + 0.245, label, fontsize=FS_LABEL,
                 fontweight="bold", color=colour, ha="left", va="center")
        fig.text(LAB_X, y0 + 0.115, f"pIC$_{{50}}$ {pic50:.2f}",
                 fontsize=FS_SMALL, color="0.30", ha="left", va="center")

    # The cliff itself, in the reserved band: one ring position, 1.8 log units.
    dlt = PAIR[0][2] - PAIR[1][2]
    cx = MOL_X + MOL_W / 2
    fig.text(cx, 0.560, "one ring position", fontsize=FS_SMALL - 0.5,
             color="0.35", ha="center", va="top", style="italic")
    fig.text(cx, 0.425, f"\u0394pIC$_{{50}}$ = {dlt:.2f}", fontsize=FS_LABEL,
             color="0.10", ha="center", va="bottom", fontweight="bold")

    # ── right: measured direction recovery ─────────────────────────────────
    # Horizontal bars: the representation names then sit as y tick labels at
    # body size with no rotation, and the chance reference can be labelled
    # above the axes instead of on top of a bar.
    ax = fig.add_axes([0.690, 0.345, 0.265, 0.430])
    order = sorted(rates, key=lambda kv: kv[1])          # best at the top
    ys = np.arange(len(order))
    vals = [v for _, v in order]
    ax.barh(ys, vals, color=[REPR_COLOUR[r] for r, _ in order],
            height=0.70, zorder=3)

    # 50 % is the coin-flip reference for getting a direction right.
    ax.axvline(50, color="0.35", lw=0.8, ls="--", zorder=4)
    ax.text(50, len(order) - 0.38, "chance", fontsize=FS_SMALL - 0.5,
            color="0.35", ha="center", va="bottom")

    for y, v in zip(ys, vals):
        ax.text(v + 2.5, y, f"{v:.0f}", fontsize=FS_TICK, ha="left",
                va="center", fontweight="bold", color="0.15")

    ax.set_yticks(ys)
    ax.set_yticklabels([REPR_LABEL[r] for r, _ in order], fontsize=FS_SMALL)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.tick_params(axis="x", labelsize=FS_SMALL, length=2, pad=1)
    ax.tick_params(axis="y", length=0, pad=1.5)
    # The panel is 0.86 in wide, so the axis label has to be short enough to
    # sit inside it; the sample size goes on its own lines underneath.
    ax.set_xlabel("% correct", fontsize=FS_SMALL - 0.5, labelpad=1.5,
                  color="0.25")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)

    pcx = 0.690 + 0.265 / 2
    fig.text(pcx, 0.975, "Do models get the\ndirection right?",
             fontsize=FS_TITLE - 1.0, fontweight="bold", ha="center",
             va="top", color="0.10", linespacing=1.25)
    fig.text(pcx, 0.130, f"median of {n_models} regressors",
             fontsize=FS_SMALL, color="0.45", ha="center", va="top")
    fig.text(pcx, 0.055, f"{n_pairs} cliff pairs",
             fontsize=FS_SMALL, color="0.45", ha="center", va="top")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results5x5_last")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig = build_figure(args.out_dir)
    stem = os.path.join(args.out_dir, "toc_graphic")
    # No tight bbox: the canvas size is prescribed.
    fig.savefig(f"{stem}.tif", dpi=DPI_TIF, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(f"{stem}.svg", facecolor="white")
    plt.close(fig)
    print(f"  -> {stem}.tif (300 dpi), .png, .svg  [{FIG_W} x {FIG_H} in]")


if __name__ == "__main__":
    main()
