#!/usr/bin/env python
"""
qsar_nmda.py  —  QSAR analysis of NMDA antagonists (Schultz et al. dataset)
=============================================================================
Usage
-----
  python qsar_nmda.py                     # default run
  python qsar_nmda.py --hyperopt          # + Optuna hyperparameter optimisation
  python qsar_nmda.py --sali_sweep        # + SALI threshold sensitivity plot
  python qsar_nmda.py --hyperopt --sali_sweep   # both
  python qsar_nmda.py --figures_only      # regenerate figures from existing CSVs
  python qsar_nmda.py --cv5x5            # 5×5 repeated CV + error bars + Tukey HSD

Outputs (written to ./results/)
-------
  cv_heatmap_pic50.png / cv_heatmap_pki.png
  desc_vs_fp_mae.png
  external_predictions.png          ← includes ChemProp
  metric_correlation_{desc,fp}.png
  sali_landscape.png
  sali_threshold_sweep.png          ← if --sali_sweep
  cv_vs_external_spearman.png
  cliff_mae_comparison.png
  cliff_directionality_bars.png      ← % correct per method
  mega_directionality_vs_mae.png     ← MAE vs % correct mega-scatter
  cliff_pair_heatmap.png             ← per-pair × per-model correctness (binary)
  cliff_directionality_pic50.csv     ← directionality table
  directionality_vs_sali_{Desc,FP,FCFP}.png   ← correctness vs SALI hardness (NEW)
  cross_endpoint_consistency.png     ← pIC50 ↔ pKi cliff-pair agreement (NEW)
  cross_endpoint_consistency.csv     ← (NEW)
  leave_cliff_out_cv.png             ← LCO generalisation gap (NEW)
  leave_cliff_out_pic50.csv          ← (NEW)
  repeated_dir_uncertainty_{feat}.png ← uncertainty vs correctness (--cv5x5, NEW)
  repeated_dir_summary_{feat}.png    ← graded heatmap with uncertainty (--cv5x5, NEW)
  ridge_desc_coefficients.png
  ridge_fp_coefficients.png
  ridge_fp_morgan_bits.png
  external_pearson_bar.png
  ridge_sim_map_*.png              ← ECFP Ridge similarity maps (in results/)
  ridge_fcfp_sim_map_*.png         ← FCFP Ridge similarity maps (in results/)
  ridge_heatmap_r0/r1/r2/all.png  ← per-radius ECFP Ridge coefficient bars
  ridge_fcfp_heatmap_r0/r1/r2/all.png  ← per-radius FCFP Ridge coefficient bars
  metric_correlation_fcfp.png
  results_pic50.csv / results_pki.csv
  external_results.csv
  cv5x5_pic50.csv                   ← if --cv5x5
  cv5x5_errorbars_pic50.png         ← if --cv5x5
  tukey_hsd_pic50.csv               ← if --cv5x5
  (All figures are also saved as .svg)
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse, logging, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ── sklearn compat shim for ChemProp 2.2.x (needs root_mean_squared_error) ───
import sklearn.metrics as _skm
if not hasattr(_skm, "root_mean_squared_error"):
    import numpy as _np
    def _rmse(y_true, y_pred, **kw):
        return _np.sqrt(_skm.mean_squared_error(y_true, y_pred, **kw))
    _skm.root_mean_squared_error = _rmse
    try:
        import sklearn.metrics._regression as _reg
        _reg.root_mean_squared_error = _rmse
    except Exception:
        pass

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless / script mode
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from tqdm import tqdm

from external_data import EXT_LABELS, EXT_PIC50, EXT_SMILES

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, DataStructs, Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
from rdkit.ML.Descriptors import MoleculeDescriptors
RDLogger.DisableLog("rdApp.*")

from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

try:
    from adjustText import adjust_text as _adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("✗ shap not found — pip install shap — SHAP figures will be skipped")

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    print("✗ TabPFN not found — skipping")

try:
    import chemprop
    from chemprop import data as cp_data
    from chemprop.nn import (BondMessagePassing, MeanAggregation, RegressionFFN,
                             UnscaleTransform)
    from chemprop.models import MPNN
    import torch
    import lightning.pytorch as pl
    HAS_CHEMPROP = True
    print(f"✓ ChemProp {chemprop.__version__}")
except ImportError as e:
    HAS_CHEMPROP = False
    print(f"✗ ChemProp not found ({e}) — skipping")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEED            = 42
N_FOLDS         = 5
N_REPEATS       = 5          # for --cv5x5 repeated CV
FP_RADIUS       = 2
FP_SIZE         = 2048
SALI_THRESHOLD  = 5.5          # primary threshold used for cliff reporting
TABPFN_MAX_FEAT = 150
TABPFN_MODEL     = Path(os.environ.get(
    "TABPFN_MODEL_PATH",
    Path.home() / ".cache" / "tabpfn" / "tabpfn-v2-regressor.ckpt",
))
CHEMPROP_EPOCHS = 40
OUT_DIR         = "results"
CHEMELEON_CKPT  = "checkpoints/chemeleon_mp.pt"   # set to None to disable

np.random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION FIGURE GEOMETRY
# -----------------------------------------------------------------------------
# Figures are drawn at the width they are actually placed at in the manuscript,
# so that no rescaling happens in the word processor.  Text set at FS_BASE then
# prints at FS_BASE.  (A figure authored 18 in wide but placed at 165 mm shrinks
# every label to ~0.36x its nominal size, which is what previously rendered the
# axis and point labels illegible.)  Journal text width: 165 mm = 6.5 in.
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


def check_text_overlaps(fig, ignore_ticklabels=True):
    """
    Geometric self-check.  Returns (overlaps, outside): pairs of visible text
    artists whose bounding boxes intersect, and text drawn off-canvas.  Two tick
    labels on the same axis are not counted.  Used to verify a figure is clean
    before writing it out.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    # Axes with the frame switched off (image panels, for instance) keep their
    # tick label artists, but those are never drawn, so they cannot collide with
    # anything.  Titles on such axes *are* drawn and stay in the check.
    undrawn = set()
    for ax in fig.axes:
        if not ax.axison:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                undrawn.add(id(t))

    items = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if id(t) in undrawn:
            continue
        try:
            items.append((t, t.get_window_extent(r)))
        except Exception:
            pass

    tick_ids = set()
    if ignore_ticklabels:
        for ax in fig.axes:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                tick_ids.add(id(t))

    overlaps = []
    for i, (ta, ba) in enumerate(items):
        for tb, bb in items[i + 1:]:
            if id(ta) in tick_ids and id(tb) in tick_ids:
                continue
            if ba.overlaps(bb):
                overlaps.append((ta.get_text()[:30], tb.get_text()[:30]))

    fb = fig.bbox
    outside = [t.get_text()[:30] for t, bb in items
               if bb.x1 < fb.x0 - 1 or bb.x0 > fb.x1 + 1
               or bb.y1 < fb.y0 - 1 or bb.y0 > fb.y1 + 1]
    return overlaps, outside


apply_print_style()


# ═════════════════════════════════════════════════════════════════════════════
# 1 · FEATURISATION
# ═════════════════════════════════════════════════════════════════════════════
MFPGEN      = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
# FCFP: same radius/size but pharmacophoric atom invariants (useFeatures=True equivalent)
FCFPGEN     = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_SIZE,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
_desc_names = [d[0] for d in Descriptors.descList]
_calc       = MoleculeDescriptors.MolecularDescriptorCalculator(_desc_names)
N_DESC      = len(_desc_names)


def smiles_to_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Cannot parse: {smi}")
    return mol


def mol_to_desc(mol):
    return np.array(_calc.CalcDescriptors(mol), dtype=float)


def mol_to_fp(mol):
    return MFPGEN.GetCountFingerprintAsNumPy(mol).astype(float)


def mol_to_fcfp(mol):
    """FCFP: Morgan counts with pharmacophoric (feature-based) atom invariants."""
    return FCFPGEN.GetCountFingerprintAsNumPy(mol).astype(float)


def featurise(smiles_list, feat_fn):
    rows, mask = [], []
    for smi in smiles_list:
        try:
            rows.append(feat_fn(smiles_to_mol(smi)))
            mask.append(True)
        except Exception:
            rows.append(None)
            mask.append(False)
    good = np.array(mask, dtype=bool)
    X = np.vstack([r for r, g in zip(rows, good) if g])
    return X, good


def clean(X):
    return np.where(np.isfinite(X), X, 0.0)


def prep_dataset(df, smiles_col="Canonical SMILES"):
    y_col = [c for c in df.columns
             if c.lower().startswith("p") and c != smiles_col][0]
    print(f"  y-column: {y_col}")
    df = df[[smiles_col, y_col]].dropna().copy()
    df.columns = ["smiles", "y"]
    X_fp_raw, mask = featurise(df["smiles"], mol_to_fp)
    df = df[mask].reset_index(drop=True)
    X_desc_raw,  _ = featurise(df["smiles"], mol_to_desc)
    X_fcfp_raw,  _ = featurise(df["smiles"], mol_to_fcfp)
    return df, clean(X_desc_raw), clean(X_fp_raw), clean(X_fcfp_raw)


# ═════════════════════════════════════════════════════════════════════════════
# 2 · SALI
# ═════════════════════════════════════════════════════════════════════════════

def tanimoto_matrix(mols):
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_SIZE)
           for m in mols]
    n = len(fps)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            T[i, j] = T[j, i] = s
    return T


def compute_sali(smiles_list, y):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    T    = tanimoto_matrix(mols)
    y    = np.asarray(y)
    n    = len(mols)
    S    = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - T[i, j]
            if d > 1e-6:
                v = abs(y[i] - y[j]) / d
                S[i, j] = S[j, i] = v
    return S


def sali_cliff_mask(smiles_list, y, threshold):
    return (compute_sali(smiles_list, y) > threshold).any(axis=1)


def compute_sali_pairs(sali_mat, threshold):
    """
    Return a list of (i, j, sali_val) for every upper-triangle pair whose
    SALI score exceeds *threshold*.  i < j always.
    """
    n = sali_mat.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sali_mat[i, j] > threshold:
                pairs.append((i, j, sali_mat[i, j]))
    return pairs  # list[(i, j, sali_val)]


# ═════════════════════════════════════════════════════════════════════════════
# 3 · CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def make_folds(n):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(kf.split(np.arange(n)))


def cv_evaluate(model_pipeline, X, y, cliff_mask, folds):
    all_yt, all_yp, all_ct, all_cp = [], [], [], []
    for tr_idx, val_idx in folds:
        model_pipeline.fit(X[tr_idx], y[tr_idx])
        yhat = model_pipeline.predict(X[val_idx])
        yval = y[val_idx]
        all_yt.extend(yval); all_yp.extend(yhat)
        cm = cliff_mask[val_idx]
        all_ct.extend(yval[cm]); all_cp.extend(yhat[cm])
    yt = np.array(all_yt); yp = np.array(all_yp)
    ct = np.array(all_ct); cp_ = np.array(all_cp)
    return {
        "MAE":       mean_absolute_error(yt, yp),
        "R2":        r2_score(yt, yp),
        "MAE_cliff": mean_absolute_error(ct, cp_) if len(ct) > 0 else np.nan,
        "R2_cliff":  r2_score(ct, cp_)            if len(ct) > 1 else np.nan,
    }


def cv_evaluate_return_preds(model_pipeline, X, y, cliff_mask, folds):
    """
    Like cv_evaluate but additionally returns index-aligned arrays so that
    yt_full[k] and yp_full[k] correspond to dataset index idx_full[k].
    Used for cliff-pair directionality scoring.
    """
    import copy
    all_yt, all_yp, all_ct, all_cp = [], [], [], []
    idx_full = np.full(len(y), -1, dtype=int)
    yt_full  = np.full(len(y), np.nan)
    yp_full  = np.full(len(y), np.nan)

    for tr_idx, val_idx in folds:
        pipe = copy.deepcopy(model_pipeline)
        pipe.fit(X[tr_idx], y[tr_idx])
        yhat = pipe.predict(X[val_idx])
        yval = y[val_idx]
        all_yt.extend(yval); all_yp.extend(yhat)
        cm = cliff_mask[val_idx]
        all_ct.extend(yval[cm]); all_cp.extend(yhat[cm])
        # store indexed predictions
        yt_full[val_idx] = yval
        yp_full[val_idx] = yhat

    yt = np.array(all_yt); yp = np.array(all_yp)
    ct = np.array(all_ct); cp_ = np.array(all_cp)
    metrics = {
        "MAE":       mean_absolute_error(yt, yp),
        "R2":        r2_score(yt, yp),
        "MAE_cliff": mean_absolute_error(ct, cp_) if len(ct) > 0 else np.nan,
        "R2_cliff":  r2_score(ct, cp_)            if len(ct) > 1 else np.nan,
    }
    return metrics, yt_full, yp_full


# ═════════════════════════════════════════════════════════════════════════════
# 4 · MODELS
# ═════════════════════════════════════════════════════════════════════════════

def make_sklearn_models(hyperopt=False, study_name="", X_tr=None, y_tr=None, folds=None):
    if hyperopt:
        return _make_hyperopt_models(X_tr, y_tr, folds, study_name)

    return {
        "Ridge": Pipeline([
            ("scl", StandardScaler(with_mean=False)),
            ("mdl", Ridge(alpha=1.0)),
        ]),
        "BayesianRidge": Pipeline([
            ("scl", StandardScaler(with_mean=False)),
            ("mdl", BayesianRidge()),
        ]),
        "ElasticNet": Pipeline([
            ("scl", StandardScaler(with_mean=False)),
            ("mdl", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ]),
        "RF": Pipeline([
            ("mdl", RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)),
        ]),
        "XGB": Pipeline([
            ("mdl", XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                 subsample=0.8, verbosity=0, random_state=SEED)),
        ]),
        "LGBM": Pipeline([
            ("mdl", LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  n_jobs=-1, verbose=-1, random_state=SEED)),
        ]),
        "SVM": Pipeline([
            ("scl", StandardScaler(with_mean=False)),
            ("mdl", SVR(C=10, epsilon=0.1, kernel="rbf")),
        ]),
        "MLP": Pipeline([
            ("scl", StandardScaler(with_mean=False)),
            ("mdl", MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                                 max_iter=500, early_stopping=True, random_state=SEED)),
        ]),
    }


def _make_hyperopt_models(X, y, folds, study_name):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ✗ Optuna not installed — pip install optuna — falling back to defaults")
        return make_sklearn_models(hyperopt=False)

    def _cv_score(pipeline, X, y, folds):
        maes = []
        for tr, val in folds:
            pipeline.fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[val], pipeline.predict(X[val])))
        return np.mean(maes)

    models = {}

    def obj_ridge(trial):
        alpha = trial.suggest_float("alpha", 1e-3, 1e3, log=True)
        return _cv_score(Pipeline([("scl", StandardScaler(with_mean=False)),
                                   ("mdl", Ridge(alpha=alpha))]), X, y, folds)
    s = optuna.create_study(direction="minimize", study_name=f"{study_name}_ridge")
    s.optimize(obj_ridge, n_trials=30, show_progress_bar=False)
    models["Ridge"] = Pipeline([("scl", StandardScaler(with_mean=False)),
                                 ("mdl", Ridge(**s.best_params))])
    print(f"    Ridge best alpha={s.best_params['alpha']:.4f}  MAE={s.best_value:.3f}")

    models["BayesianRidge"] = Pipeline([("scl", StandardScaler(with_mean=False)),
                                        ("mdl", BayesianRidge())])

    def obj_en(trial):
        alpha   = trial.suggest_float("alpha",    1e-3, 10.0, log=True)
        l1ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
        return _cv_score(Pipeline([("scl", StandardScaler(with_mean=False)),
                                   ("mdl", ElasticNet(alpha=alpha, l1_ratio=l1ratio,
                                                       max_iter=5000))]), X, y, folds)
    s = optuna.create_study(direction="minimize", study_name=f"{study_name}_en")
    s.optimize(obj_en, n_trials=30, show_progress_bar=False)
    models["ElasticNet"] = Pipeline([("scl", StandardScaler(with_mean=False)),
                                     ("mdl", ElasticNet(max_iter=5000, **s.best_params))])
    print(f"    ElasticNet best={s.best_params}  MAE={s.best_value:.3f}")

    def obj_rf(trial):
        p = dict(
            n_estimators = trial.suggest_int("n_estimators", 100, 600),
            max_depth    = trial.suggest_int("max_depth",    3,   20),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10),
        )
        return _cv_score(Pipeline([("mdl", RandomForestRegressor(
            **p, n_jobs=-1, random_state=SEED))]), X, y, folds)
    s = optuna.create_study(direction="minimize", study_name=f"{study_name}_rf")
    s.optimize(obj_rf, n_trials=30, show_progress_bar=False)
    models["RF"] = Pipeline([("mdl", RandomForestRegressor(
        **s.best_params, n_jobs=-1, random_state=SEED))])
    print(f"    RF best={s.best_params}  MAE={s.best_value:.3f}")

    def obj_xgb(trial):
        p = dict(
            n_estimators  = trial.suggest_int("n_estimators",   100, 600),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth     = trial.suggest_int("max_depth",       3,   10),
            subsample     = trial.suggest_float("subsample",      0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        )
        return _cv_score(Pipeline([("mdl", XGBRegressor(
            **p, verbosity=0, random_state=SEED))]), X, y, folds)
    s = optuna.create_study(direction="minimize", study_name=f"{study_name}_xgb")
    s.optimize(obj_xgb, n_trials=40, show_progress_bar=False)
    models["XGB"] = Pipeline([("mdl", XGBRegressor(
        **s.best_params, verbosity=0, random_state=SEED))])
    print(f"    XGB best={s.best_params}  MAE={s.best_value:.3f}")

    def obj_lgbm(trial):
        p = dict(
            n_estimators  = trial.suggest_int("n_estimators",   100, 600),
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves    = trial.suggest_int("num_leaves",      15,  127),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        )
        return _cv_score(Pipeline([("mdl", LGBMRegressor(
            **p, n_jobs=-1, verbose=-1, random_state=SEED))]), X, y, folds)
    s = optuna.create_study(direction="minimize", study_name=f"{study_name}_lgbm")
    s.optimize(obj_lgbm, n_trials=40, show_progress_bar=False)
    models["LGBM"] = Pipeline([("mdl", LGBMRegressor(
        **s.best_params, n_jobs=-1, verbose=-1, random_state=SEED))])
    print(f"    LGBM best={s.best_params}  MAE={s.best_value:.3f}")

    def obj_svm(trial):
        p = dict(
            C       = trial.suggest_float("C",       0.1, 100, log=True),
            epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        )
        return _cv_score(Pipeline([("scl", StandardScaler(with_mean=False)),
                                   ("mdl", SVR(kernel="rbf", **p))]), X, y, folds)
    s = optuna.create_study(direction="minimize", study_name=f"{study_name}_svm")
    s.optimize(obj_svm, n_trials=30, show_progress_bar=False)
    models["SVM"] = Pipeline([("scl", StandardScaler(with_mean=False)),
                               ("mdl", SVR(kernel="rbf", **s.best_params))])
    print(f"    SVM best={s.best_params}  MAE={s.best_value:.3f}")

    def obj_mlp(trial):
        nh1 = trial.suggest_categorical("nh1", [64, 128, 256])
        nh2 = trial.suggest_categorical("nh2", [32, 64, 128])
        lr  = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        return _cv_score(Pipeline([("scl", StandardScaler(with_mean=False)),
                                   ("mdl", MLPRegressor(
                                       hidden_layer_sizes=(nh1, nh2),
                                       learning_rate_init=lr,
                                       max_iter=500, early_stopping=True,
                                       random_state=SEED))]), X, y, folds)
    s = optuna.create_study(direction="minimize", study_name=f"{study_name}_mlp")
    s.optimize(obj_mlp, n_trials=30, show_progress_bar=False)
    bp = s.best_params
    models["MLP"] = Pipeline([("scl", StandardScaler(with_mean=False)),
                               ("mdl", MLPRegressor(
                                   hidden_layer_sizes=(bp["nh1"], bp["nh2"]),
                                   learning_rate_init=bp["lr"],
                                   max_iter=500, early_stopping=True,
                                   random_state=SEED))])
    print(f"    MLP best={bp}  MAE={s.best_value:.3f}")

    return models


def compress_for_tabpfn(X_train, X_other=None):
    if X_train.shape[1] > TABPFN_MAX_FEAT:
        pca = PCA(n_components=TABPFN_MAX_FEAT, random_state=SEED).fit(X_train)
        return pca.transform(X_train), (pca.transform(X_other) if X_other is not None else None)
    return X_train, X_other


def make_tabpfn_pipeline(device="cuda"):
    """TabPFN v2 with PCA fitted inside each CV training fold."""
    if not TABPFN_MODEL.exists():
        raise FileNotFoundError(
            f"TabPFN v2 checkpoint not found: {TABPFN_MODEL}. "
            "Set TABPFN_MODEL_PATH to the cached checkpoint."
        )
    return Pipeline([
        ("pca", PCA(n_components=TABPFN_MAX_FEAT, random_state=SEED)),
        ("mdl", TabPFNRegressor(
            device=device,
            model_path=TABPFN_MODEL,
            random_state=0,
            n_estimators=8,
            n_jobs=1,
            fit_mode="fit_preprocessors",
        )),
    ])


# ── ChemProp ─────────────────────────────────────────────────────────────────

def _make_cp_dataset(smiles, y=None):
    dps = []
    for i, smi in enumerate(smiles):
        dps.append(cp_data.MoleculeDatapoint(
            mol=Chem.MolFromSmiles(smi),
            y=np.array([y[i]], dtype=float) if y is not None else None,
        ))
    return cp_data.MoleculeDataset(dps)


def _build_mpnn(scaler, use_chemeleon=False):
    if use_chemeleon and os.path.exists(CHEMELEON_CKPT):
        ckpt = torch.load(CHEMELEON_CKPT, map_location="cpu", weights_only=True)
        # The public CheMeleon file is a bare message-passing encoder, not a
        # complete Chemprop MPNN.  Instantiate its 2048-wide/depth-6 geometry
        # from the saved hyperparameters and require an exact encoder load.
        mp_block = BondMessagePassing(**dict(ckpt["hyper_parameters"]))
        load_result = mp_block.load_state_dict(ckpt["state_dict"], strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(f"CheMeleon encoder load failed: {load_result}")
        print("    ✓ CheMeleon encoder loaded strictly")
    elif use_chemeleon:
        raise FileNotFoundError(
            f"CheMeleon checkpoint not found at {CHEMELEON_CKPT}; "
            "refusing to label a scratch model as pretrained"
        )
    else:
        mp_block = BondMessagePassing()
    predictor = RegressionFFN(
        input_dim=mp_block.output_dim,
        n_tasks=1,
        output_transform=UnscaleTransform.from_standard_scaler(scaler),
    )
    return MPNN(message_passing=mp_block, agg=MeanAggregation(), predictor=predictor)


def train_chemprop(smiles_train, y_train, use_chemeleon=False,
                   epochs=CHEMPROP_EPOCHS, seed=SEED):
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True, verbose=False)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dset   = _make_cp_dataset(smiles_train, y_train)
    scaler = dset.normalize_targets()
    mp     = _build_mpnn(scaler, use_chemeleon)
    trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False,
                         enable_model_summary=False, logger=False,
                         deterministic=True, enable_checkpointing=False)
    trainer.fit(mp, cp_data.build_dataloader(
        dset, batch_size=64, num_workers=0, shuffle=True, seed=seed))
    return mp, trainer, scaler


def predict_chemprop(mp, trainer, scaler, smiles_val):
    dset  = _make_cp_dataset(smiles_val)
    preds = trainer.predict(mp, cp_data.build_dataloader(
        dset, batch_size=64, num_workers=0, shuffle=False))
    # RegressionFFN's UnscaleTransform already returns predictions in the
    # original target units; applying the scaler again would double-unscale.
    return torch.cat(preds).detach().cpu().numpy().flatten()


def cv_chemprop(smiles_arr, y_arr, cliff_mask, folds, use_chemeleon=False):
    all_yt, all_yp, all_ct, all_cp = [], [], [], []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        mp, trainer, scaler = train_chemprop(
            smiles_arr[tr_idx], y_arr[tr_idx], use_chemeleon=use_chemeleon,
            seed=SEED * 100 + fold_idx)
        yhat = predict_chemprop(mp, trainer, scaler, smiles_arr[val_idx])
        yval = y_arr[val_idx]
        all_yt.extend(yval); all_yp.extend(yhat)
        cm = cliff_mask[val_idx]
        all_ct.extend(yval[cm]); all_cp.extend(yhat[cm])
    yt = np.array(all_yt); yp = np.array(all_yp)
    ct = np.array(all_ct); cp_ = np.array(all_cp)
    return {
        "MAE":       mean_absolute_error(yt, yp),
        "R2":        r2_score(yt, yp),
        "MAE_cliff": mean_absolute_error(ct, cp_) if len(ct) > 0 else np.nan,
        "R2_cliff":  r2_score(ct, cp_)            if len(ct) > 1 else np.nan,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5 · RESULTS HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def results_to_df(results):
    rows = [{"Model": m, "Features": f, **v} for (m, f), v in results.items()]
    return (pd.DataFrame(rows)
            .sort_values(["Features", "MAE"])
            .reset_index(drop=True))


def ext_metrics(ext_preds, y_ext_true):
    rows = []
    for (name, feat), preds in ext_preds.items():
        pr, _ = pearsonr(y_ext_true, preds)
        sr, _ = spearmanr(y_ext_true, preds)
        rows.append({"Model": name, "Features": feat,
                     "Pearson_r": pr, "Spearman_rho": sr,
                     "MAE_ext": mean_absolute_error(y_ext_true, preds)})
    # Sort by Pearson_r rather than Spearman
    return (pd.DataFrame(rows)
            .sort_values(["Features", "Pearson_r"], ascending=[True, False])
            .reset_index(drop=True))


def prospective_pair_table(ext_preds, X_dict, X_ext_dict, y, directionality,
                           n_bootstrap=200):
    """Signed external-pair predictions with percentile bootstrap intervals.

    The experimental values are not used in this table's model calculations;
    they are plotted separately with their cell-level uncertainty. Bootstrap
    intervals quantify training-sample variation for the conventional models.
    """
    rng = np.random.default_rng(SEED + 2026)
    dir_lookup = directionality.set_index(["Model", "Features"])["pct_correct"]
    rows = []
    for feat, X in X_dict.items():
        X_ext = X_ext_dict[feat]
        for model_name, template in make_sklearn_models().items():
            pred = np.asarray(ext_preds[(model_name, feat)], dtype=float)
            delta_ethyl = float(pred[0] - pred[1])
            delta_propyl = float(pred[2] - pred[3])
            boot_means = []
            for _ in range(n_bootstrap):
                sample = rng.integers(0, len(y), size=len(y))
                estimator = clone(template)
                estimator.fit(X[sample], y[sample])
                boot_pred = np.asarray(estimator.predict(X_ext), dtype=float)
                boot_means.append(float(
                    ((boot_pred[0] - boot_pred[1])
                     + (boot_pred[2] - boot_pred[3])) / 2.0
                ))
            low, high = np.percentile(boot_means, [2.5, 97.5])
            n_correct = int(delta_ethyl > 0) + int(delta_propyl > 0)
            rows.append({
                "Model": model_name,
                "Features": feat,
                "Δ N-ethyl (HC49−HC22)": delta_ethyl,
                "Δ N-propyl (HC46−HC52)": delta_propyl,
                "Mean Δ": (delta_ethyl + delta_propyl) / 2.0,
                "Mean Δ CI low": float(low),
                "Mean Δ CI high": float(high),
                "Mean Δ, uncertainty interval": f"[{low:.2f}, {high:.2f}]",
                "Both directions correct": "yes" if n_correct == 2 else "one" if n_correct == 1 else "no",
                "Training cliff pairs correct (%)": float(dir_lookup.loc[(model_name, feat)]),
                "Uncertainty interval": f"percentile bootstrap 95% CI ({n_bootstrap} refits)",
            })
    return (pd.DataFrame(rows)
            .sort_values(["Mean Δ", "Model", "Features"], ascending=[False, True, True])
            .reset_index(drop=True))


# ═════════════════════════════════════════════════════════════════════════════
# 6 · PLOTTING
# ═════════════════════════════════════════════════════════════════════════════
PAL = plt.cm.tab10.colors
METRICS = ["MAE", "R2", "MAE_cliff", "R2_cliff"]

# One colour per feature space, threaded through every main-text figure so the
# same representation is the same colour wherever it appears.  Colour-blind safe
# (Okabe-Ito derived) and distinguishable in greyscale by lightness.
FEAT_COLORS = {
    "Desc":  "#4C72B0",   # blue
    "FP":    "#DD8452",   # orange
    "FCFP":  "#55A868",   # green
    "Graph": "#8172B3",   # purple
}
FEAT_ORDER = ["Desc", "FP", "FCFP", "Graph"]

# Display names for estimators whose full name does not fit the space it has to
# occupy (a lattice cell, a tick label at FS_MIN).  Applied to every figure that
# names estimators so the manuscript is internally consistent; the caption should
# expand any abbreviation once.
MODEL_DISPLAY = {"BayesianRidge": "BayesRidge"}


def model_label(name):
    """Estimator name as it should be printed in a figure."""
    return MODEL_DISPLAY.get(name, name)


# Marker per estimator, so the mega-scatter can encode feature space in colour
# and estimator type in shape.  Fixed rather than assigned by enumeration order
# so the same estimator keeps the same marker if the model set changes.  Shapes
# are chosen to stay distinguishable at the ~2 mm printed marker size.
MODEL_MARKERS = {
    "Ridge":         "o",
    "BayesianRidge": "s",
    "ElasticNet":    "D",
    "SVM":           "^",
    "RF":            "v",
    "XGB":           "P",
    "LGBM":          "X",
    "MLP":           "*",
    "TabPFN":        "h",
}
_MARKER_FALLBACK = ["<", ">", "p", "8", "d"]


def model_marker(name, _cache={}):
    """Marker for an estimator, stable across figures and runs."""
    if name in MODEL_MARKERS:
        return MODEL_MARKERS[name]
    if name not in _cache:
        _cache[name] = _MARKER_FALLBACK[len(_cache) % len(_MARKER_FALLBACK)]
    return _cache[name]


def savefig(name, fig=None):
    # If name is already an absolute path or already starts with OUT_DIR, use as-is
    if os.path.isabs(name) or name.startswith(OUT_DIR + os.sep) or name.startswith(OUT_DIR + "/"):
        path = name
    else:
        path = os.path.join(OUT_DIR, name)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    target = fig if fig else plt
    target.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    # Also save as SVG automatically
    base, _ = os.path.splitext(path)
    svg_path = base + ".svg"
    target.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)
    if fig:
        plt.close(fig)
    else:
        plt.close()
    print(f"  → {path} (and .svg)")


def _annotate_scatter(ax, xs, ys, labels, fontsize=8):
    """Add text labels to a scatter plot, repelling overlaps when adjustText is available."""
    texts = []
    for x, y, lbl in zip(xs, ys, labels):
        texts.append(ax.annotate(lbl, (x, y),
                                 textcoords="offset points", xytext=(4, 3),
                                 fontsize=fontsize))
    if HAS_ADJUSTTEXT and texts:
        _adjust_text(texts, ax=ax,
                     arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))


def plot_cv_heatmap(tab, title, fname):
    tab = tab.copy()
    tab["Model_Feat"] = tab["Model"] + " / " + tab["Features"]
    heat = tab.set_index("Model_Feat")[METRICS].astype(float)
    norm = heat.copy()
    for col in ["MAE", "MAE_cliff"]:
        if col in norm:
            v = norm[col]; norm[col] = 1 - (v - v.min()) / (v.max() - v.min() + 1e-9)
    for col in ["R2", "R2_cliff"]:
        if col in norm:
            v = norm[col]; norm[col] = (v - v.min()) / (v.max() - v.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(8, max(5, len(heat) * 0.42)))
    sns.heatmap(norm, annot=heat.round(3), fmt=".3f",
                cmap="RdYlGn", ax=ax, linewidths=0.4,
                cbar_kws={"label": "Relative performance"})
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    savefig(fname)


def _std_lookup(df5x5, feat):
    """Return a Model→MAE_std Series from a df5x5 DataFrame, or an empty Series."""
    if df5x5 is None:
        return pd.Series(dtype=float)
    sub = df5x5[df5x5["Features"] == feat]
    if sub.empty or "MAE_std" not in sub.columns:
        return pd.Series(dtype=float)
    return sub.set_index("Model")["MAE_std"]


def plot_desc_vs_fp(tab, fname, df5x5=None):
    d_desc  = tab[tab["Features"] == "Desc"].set_index("Model")["MAE"]
    d_fp    = tab[tab["Features"] == "FP"].set_index("Model")["MAE"]
    std_desc = _std_lookup(df5x5, "Desc")
    std_fp   = _std_lookup(df5x5, "FP")
    common = d_desc.index.intersection(d_fp.index)
    fig, ax = plt.subplots(figsize=(6, 6))
    xs, ys, lbls = [], [], []
    for i, m in enumerate(common):
        xe = std_desc.get(m, np.nan)
        ye = std_fp.get(m, np.nan)
        if np.isfinite(xe) and np.isfinite(ye):
            ax.errorbar(d_desc[m], d_fp[m], xerr=xe, yerr=ye,
                        fmt="none", color=PAL[i % 10], capsize=3,
                        lw=1.2, alpha=0.6, zorder=2)
        ax.scatter(d_desc[m], d_fp[m], s=110, color=PAL[i % 10], zorder=3)
        xs.append(d_desc[m]); ys.append(d_fp[m]); lbls.append(m)
    _annotate_scatter(ax, xs, ys, lbls, fontsize=8)
    pad = 0.02
    lo = min(d_desc[common].min(), d_fp[common].min())
    hi = max(d_desc[common].max(), d_fp[common].max())
    span = hi - lo
    lo -= span * pad; hi += span * pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("MAE — Desc"); ax.set_ylabel("MAE — FP")
    eb_note = " (bars = ±1 SD over 5 repeats)" if df5x5 is not None else ""
    ax.set_title(f"Desc vs FP: 5-fold MAE (pIC50)\n(below diagonal = FP better){eb_note}")
    plt.tight_layout()
    savefig(fname)


def plot_external(ext_preds, y_ext_true, ext_labels, fname):
    """
    SI figure: predicted vs observed pIC50 on the four external compounds,
    one small panel per model x feature space.

    With only four points per panel — three of which are nearly coincident in
    observed pIC50 — per-point text labels are unreadable and adjustText pushes
    them far outside the axes.  The compounds are therefore identified once by
    a shared colour legend, and each panel carries only its correlation
    statistics.  Drawn at FIG_W2 so it needs no rescaling.
    """
    items   = list(ext_preds.items())
    n_total = len(items)
    ncols   = 4
    nrows   = int(np.ceil(n_total / ncols))
    panel_w = FIG_W2 / ncols

    apply_print_style()
    # Slightly wide-of-square panels keep the 6-row grid inside the usable
    # height of one portrait page (~8.6 in) instead of overflowing to 10.5 in.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIG_W2, nrows * panel_w * 0.85),
                             squeeze=False, sharex=True, sharey=True)

    ext_cols = [FEAT_COLORS[f] for f in FEAT_ORDER][:len(ext_labels)]

    for ax, ((name, feat), preds) in zip(axes.flat, items):
        preds = np.asarray(preds, dtype=float)
        for i, (t, p) in enumerate(zip(y_ext_true, preds)):
            ax.scatter(t, p, color=ext_cols[i % len(ext_cols)], s=16,
                       edgecolors="white", linewidths=0.3, zorder=3)
        lo = min(np.min(y_ext_true), np.min(preds)) - 0.35
        hi = max(np.max(y_ext_true), np.max(preds)) + 0.35
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.6, color="0.55",
                zorder=1)
        pr, _ = pearsonr(y_ext_true, preds)
        sr, _ = spearmanr(y_ext_true, preds)
        ax.set_title(f"{name} [{feat}]", fontsize=FS_TICK, pad=2)
        # Lower right: the four external compounds sit high-left of the parity
        # line in every panel, so the upper-left corner is occupied.
        ax.text(0.96, 0.04, f"r = {pr:.2f}\n\u03c1 = {sr:.2f}",
                transform=ax.transAxes, fontsize=FS_MIN,
                va="bottom", ha="right", color="0.3",
                linespacing=1.35)
        ax.tick_params(labelsize=FS_MIN)

    for ax in axes.flat[n_total:]:
        ax.axis("off")

    handles = [mlines.Line2D([], [], marker="o", ls="none",
                      markerfacecolor=ext_cols[i % len(ext_cols)],
                      markeredgecolor="white", markeredgewidth=0.3,
                      markersize=4, label=lbl)
               for i, lbl in enumerate(ext_labels)]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.0015), ncol=len(ext_labels),
               fontsize=FS_TICK, frameon=False, handletextpad=0.3,
               columnspacing=1.4)

    fig.supxlabel("Observed pIC$_{50}$", fontsize=FS_BASE, y=0.028)
    fig.supylabel("Predicted pIC$_{50}$", fontsize=FS_BASE, x=0.008)
    fig.tight_layout(rect=(0.022, 0.043, 1, 1), h_pad=0.7, w_pad=0.5)
    savefig(fname, fig=fig)


def plot_metric_correlation(tab, ext_corr, feat_lbl, fname):
    cv_sub  = tab[tab["Features"] == feat_lbl].set_index("Model")[METRICS]
    ext_sub = ext_corr[ext_corr["Features"] == feat_lbl].set_index("Model")[
        ["Pearson_r", "Spearman_rho", "MAE_ext"]]
    merged  = cv_sub.join(ext_sub, how="inner").astype(float).dropna()
    if len(merged) < 3:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(merged.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax, linewidths=0.4)
    ax.set_title(f"Metric correlation — pIC50 / {feat_lbl}", fontsize=11)
    plt.tight_layout()
    savefig(fname)


def plot_sali_landscape(smiles_list, y, cliff_mask, sali_mat, fname):
    max_sali = sali_mat.max(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(max_sali, bins=40, color="steelblue", edgecolor="white")
    axes[0].axvline(SALI_THRESHOLD, color="crimson", ls="--",
                    label=f"Cliff threshold ({SALI_THRESHOLD})")
    axes[0].set(xlabel="Max SALI per compound", ylabel="Count",
                title="SALI distribution — pIC50")
    axes[0].legend()
    colors = np.where(cliff_mask, "crimson", "steelblue")
    axes[1].scatter(y, max_sali, c=colors, alpha=0.6, s=25)
    axes[1].axhline(SALI_THRESHOLD, color="crimson", ls="--")
    axes[1].set(xlabel="pIC50", ylabel="Max SALI", title="Activity landscape — pIC50")
    plt.tight_layout()
    savefig(fname)


def plot_sali_threshold_sweep(smiles_list, y, cv_results_fn, folds,
                              X_desc, X_fp, fname,
                              thresholds=None):
    if thresholds is None:
        thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]

    SALI_mat = compute_sali(smiles_list, y)

    records = []
    for thr in tqdm(thresholds, desc="SALI threshold sweep"):
        cliff = (SALI_mat > thr).any(axis=1)
        n_cliff = cliff.sum()
        for feat_lbl, X in [("Desc", X_desc), ("FP", X_fp)]:
            for name, pipe in make_sklearn_models().items():
                res = cv_evaluate(pipe, X, y, cliff, folds)
                records.append({
                    "threshold":  thr,
                    "n_cliff":    n_cliff,
                    "pct_cliff":  100 * n_cliff / len(y),
                    "Features":   feat_lbl,
                    "Model":      name,
                    "MAE_cliff":  res["MAE_cliff"],
                    "MAE_all":    res["MAE"],
                })

    df_sweep = pd.DataFrame(records)
    df_sweep.to_csv(os.path.join(OUT_DIR, "sali_threshold_sweep.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, feat_lbl in zip(axes, ["Desc", "FP"]):
        sub = df_sweep[df_sweep["Features"] == feat_lbl]
        for i, name in enumerate(sub["Model"].unique()):
            msub = sub[sub["Model"] == name]
            ax.plot(msub["threshold"], msub["MAE_cliff"],
                    marker="o", color=PAL[i % 10], label=name, lw=1.5)
        ax.set_xlabel("SALI threshold")
        ax.set_ylabel("MAE on cliff compounds")
        ax.set_title(f"{feat_lbl}: cliff-MAE vs SALI threshold")
        ax.axvline(SALI_THRESHOLD, color="grey", ls="--", lw=0.8,
                   label=f"Default ({SALI_THRESHOLD})")
        ax.legend(fontsize=7, ncol=2)

    ax2 = axes[1].twinx()
    pct = df_sweep.groupby("threshold")["pct_cliff"].first()
    ax2.plot(pct.index, pct.values, "k:", lw=1.5, label="% cliff cpds")
    ax2.set_ylabel("% cliff compounds")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("SALI threshold sensitivity: cliff-MAE and cliff-compound fraction",
                 fontsize=12)
    plt.tight_layout()
    savefig(fname)
    return df_sweep


def plot_cv_vs_ext_spearman(tab, ext_corr, fname, df5x5=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, feat_lbl in zip(axes, ["Desc", "FP"]):
        cv_r2  = tab[tab["Features"] == feat_lbl].set_index("Model")["R2"]
        ext_sp = ext_corr[ext_corr["Features"] == feat_lbl].set_index("Model")["Spearman_rho"]
        std_mae = _std_lookup(df5x5, feat_lbl)   # MAE std as proxy for R² uncertainty
        common = cv_r2.index.intersection(ext_sp.index)
        xs, ys, lbls = [], [], []
        for i, m in enumerate(common):
            xe = std_mae.get(m, np.nan)
            if np.isfinite(xe):
                ax.errorbar(cv_r2[m], ext_sp[m], xerr=xe,
                            fmt="none", color=PAL[i % 10],
                            capsize=3, lw=1.2, alpha=0.6, zorder=2)
            ax.scatter(cv_r2[m], ext_sp[m], s=100, color=PAL[i % 10], zorder=3)
            xs.append(cv_r2[m]); ys.append(ext_sp[m]); lbls.append(m)
        _annotate_scatter(ax, xs, ys, lbls, fontsize=8)
        if feat_lbl == "FP":
            for cp_feat, marker in [("Graph", "^"), ("Graph+Desc", "s")]:
                cp_ext = ext_corr[ext_corr["Features"] == cp_feat]
                if cp_ext.empty:
                    continue
                for _, row in cp_ext.iterrows():
                    ax.scatter(np.nan, row["Spearman_rho"],
                               s=130, marker=marker, color="black",
                               label=f"{row['Model']} [{cp_feat}]", zorder=4)
        xvals = cv_r2[common].values if len(common) else np.array([0])
        yvals = ext_sp[common].values if len(common) else np.array([0])
        if xvals.min() < 0 < xvals.max():
            ax.axvline(0, color="grey", ls=":", lw=0.7)
        if yvals.min() < 0 < yvals.max():
            ax.axhline(0, color="grey", ls=":", lw=0.7)
        ax.set_xlabel(f"CV R² [{feat_lbl}]")
        ax.set_ylabel("Spearman ρ — external")
        eb_note = " (x-bars = ±1 SD MAE)" if df5x5 is not None else ""
        ax.set_title(f"{feat_lbl}: CV R² vs external ranking{eb_note}")
    plt.tight_layout()
    savefig(fname)


# ─ shared config for the two mega-scatter plots ──────────────────────────────
_FEAT_MARKERS  = {"Desc": "o", "FP": "s", "FCFP": "D", "Graph": "^"}
_FEAT_COLORS   = {"Desc": "#4e79a7", "FP": "#f28e2b", "FCFP": "#59a14f", "Graph": "#e15759"}


def _mega_scatter(ax, xs, ys, feat_labels, model_names,
                  model_color_map, xerrs=None, ylabel="", xlabel=""):
    """
    Plot one combined scatter where marker shape = feature space, colour = model.
    Annotates each point with its model name.
    """
    texts = []
    for x, y, feat, m in zip(xs, ys, feat_labels, model_names):
        marker = _FEAT_MARKERS.get(feat, "o")
        color  = model_color_map[m]
        xe = xerrs.get((m, feat), np.nan) if xerrs else np.nan
        if np.isfinite(xe):
            ax.errorbar(x, y, xerr=xe, fmt="none", color=color,
                        capsize=3, lw=1.0, alpha=0.55, zorder=2)
        ax.scatter(x, y, marker=marker, color=color, s=110,
                   edgecolors="white", linewidths=0.5, zorder=3)
        texts.append(ax.annotate(m, (x, y), textcoords="offset points",
                                 xytext=(4, 3), fontsize=7.5))
    if HAS_ADJUSTTEXT and texts:
        _adjust_text(texts, ax=ax,
                     arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    # reference lines only where meaningful
    if len(xs):
        if min(xs) < 0 < max(xs):
            ax.axvline(0, color="grey", ls=":", lw=0.7)
        if min(ys) < 0 < max(ys):
            ax.axhline(0, color="grey", ls=":", lw=0.7)

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

    # Feature-space legend (marker shapes)
    feat_seen = dict.fromkeys(feat_labels)   # preserves insertion order, deduped
    shape_handles = [
        mlines.Line2D([0], [0], marker=_FEAT_MARKERS.get(f, "o"),
                      color="w", markerfacecolor="grey",
                      markersize=9, label=f)
        for f in feat_seen
    ]
    # Model colour legend
    model_seen = dict.fromkeys(model_names)
    color_handles = [
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=model_color_map[m],
                      markersize=9, label=m)
        for m in model_seen
    ]
    leg1 = ax.legend(handles=shape_handles, title="Feature space",
                     loc="lower left", fontsize=7, title_fontsize=8,
                     framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=color_handles, title="Model",
              loc="upper right", fontsize=7, title_fontsize=8,
              framealpha=0.85, ncol=2)


def plot_mega_spearman(tab, ext_corr, fname, df5x5=None):
    """
    Single-panel mega-scatter: CV R² (x) vs external Spearman ρ (y),
    all feature spaces overlaid.  Marker = feature space, colour = model.
    """
    feat_order = [f for f in ["Desc", "FP", "FCFP", "Graph"]
                  if f in tab["Features"].unique() or f in ext_corr["Features"].unique()]

    all_models = sorted(set(tab["Model"]) | set(ext_corr["Model"]))
    model_color_map = {m: PAL[i % len(PAL)] for i, m in enumerate(all_models)}

    xs, ys, feats, models = [], [], [], []
    xerrs = {}
    if df5x5 is not None and "MAE_std" in df5x5.columns:
        for _, r5 in df5x5.iterrows():
            xerrs[(r5["Model"], r5["Features"])] = r5["MAE_std"]

    for feat in feat_order:
        cv_r2  = tab[tab["Features"] == feat].set_index("Model")["R2"]
        ext_sp = ext_corr[ext_corr["Features"] == feat].set_index("Model")["Spearman_rho"]
        # Graph models have no CV R² — plot on y-axis only at x=NaN; skip for this plot
        common = cv_r2.index.intersection(ext_sp.index)
        for m in common:
            xs.append(cv_r2[m]); ys.append(ext_sp[m])
            feats.append(feat);  models.append(m)

    fig, ax = plt.subplots(figsize=(9, 7))
    eb_note = " (x-bars = ±1 SD MAE)" if xerrs else ""
    _mega_scatter(ax, xs, ys, feats, models, model_color_map,
                  xerrs=xerrs if xerrs else None,
                  xlabel=f"CV R²{eb_note}",
                  ylabel="Spearman ρ — external set")
    ax.set_title("All representations: CV R² vs external Spearman ρ\n"
                 "(marker = feature space, colour = model)")
    plt.tight_layout()
    savefig(fname)


def plot_mega_pearson(tab, ext_corr, fname, df5x5=None):
    """
    Main-text Figure 3: CV R2 (x) against external Pearson r (y), every
    model x feature-space combination on one axes, plus a ranked bar of the
    external Pearson r.  Drawn at FIG_W2 so it needs no rescaling in the
    manuscript.  Colour encodes feature space (FEAT_COLORS); point labels are
    de-overlapped with adjustText and connected by leader lines.

    Graph models have no CV R2, so they are drawn in a narrow strip at the
    left of panel (a), separated by a break marker rather than being placed at
    a fabricated x position.
    """
    tab = tab.copy()
    ext_corr = ext_corr.copy()
    merged = tab.merge(ext_corr, on=["Model", "Features"], how="inner")
    has_cv = merged[merged["Features"] != "Graph"].dropna(subset=["R2", "Pearson_r"])
    # Graph models have external predictions but are intentionally absent from
    # the seeded classical-CV table.  Select them from external_results.csv
    # directly, otherwise the inner merge silently drops their left-strip points.
    graph = ext_corr[ext_corr["Features"] == "Graph"].dropna(
        subset=["Pearson_r"])

    xerrs = {}
    if df5x5 is not None and "MAE_std" in df5x5.columns:
        for _, r5 in df5x5.iterrows():
            xerrs[(r5["Model"], r5["Features"])] = r5["MAE_std"]

    apply_print_style()
    # Two legend rows are stacked under the panels (feature space, then
    # estimator), so the bottom margin is set from their combined height rather
    # than by eye.
    fig = plt.figure(figsize=(FIG_W2, 4.30))
    gs = fig.add_gridspec(
        1, 3, width_ratios=[0.26, 1.45, 0.92], wspace=0.10,
        # right edge leaves room for the longest bar tick label
        # ('ChemProp+CheMeleon [Graph]'), which is drawn outside the axes;
        # without this the canvas exceeds FIG_W2 and the journal shrinks it.
        left=0.075, right=0.760, top=0.93, bottom=0.285)
    ax_g = fig.add_subplot(gs[0, 0])     # graph-model strip
    ax_a = fig.add_subplot(gs[0, 1], sharey=ax_g)
    ax_b = fig.add_subplot(gs[0, 2])

    # ── (a) CV R2 vs external Pearson r ───────────────────────────────────────
    # Only the extremes are labelled here: 30 model x feature-space points cannot
    # be labelled legibly in a panel this size, and panel (b) names every one of
    # them anyway.  The four labelled points are the best and worst on each axis.
    to_label = set()
    for c, asc in (("R2", False), ("R2", True),
                   ("Pearson_r", False), ("Pearson_r", True)):
        idx = has_cv.sort_values(c, ascending=asc).index[:1]
        to_label.update(idx)

    texts = []
    for i, r in has_cv.iterrows():
        col = FEAT_COLORS.get(r["Features"], "0.4")
        xe = xerrs.get((r["Model"], r["Features"]), np.nan)
        if np.isfinite(xe):
            ax_a.errorbar(r["R2"], r["Pearson_r"], xerr=xe, fmt="none",
                          color=col, capsize=1.5, lw=0.5, alpha=0.5, zorder=2)
        mk = model_marker(r["Model"])
        ax_a.scatter(r["R2"], r["Pearson_r"], color=col,
                     marker=mk, s=34 if mk == "*" else 24,
                     edgecolors="white", linewidths=0.4, zorder=3)
        if i in to_label:
            texts.append(ax_a.text(r["R2"], r["Pearson_r"], r["Model"],
                                   fontsize=FS_MIN, color=col, zorder=4))
    if HAS_ADJUSTTEXT and texts:
        _adjust_text(texts, ax=ax_a, expand=(1.3, 1.6),
                     force_text=(0.5, 0.9), max_move=30,
                     arrowprops=dict(arrowstyle="-", color="0.55", lw=0.35))

    ax_a.axhline(0, color="0.6", ls=":", lw=0.6, zorder=1)
    ax_a.set_xlabel("Cross-validated R$^2$ (pIC$_{50}$)")

    # The claim this panel makes, stated as the title and tested here so the
    # two can never drift apart.
    if len(has_cv) > 2:
        _r, _p = pearsonr(has_cv["R2"], has_cv["Pearson_r"])
        ax_a.set_title("Internal CV accuracy vs external ranking",
                       fontsize=FS_BASE, pad=3)
        ax_a.text(0.015, 0.015,
                  f"r = {_r:.2f}, p = {_p:.2f} (n = {len(has_cv)})",
                  transform=ax_a.transAxes, fontsize=FS_TICK,
                  color="0.3", ha="left", va="bottom")

    handles = [mlines.Line2D([0], [0], marker="o", color="w", markersize=4,
                             markerfacecolor=FEAT_COLORS[f], label=f)
               for f in FEAT_ORDER
               if f in set(ext_corr["Features"])]
    # One legend for the whole figure: panel (b)'s bars are coloured by the same
    # feature space, and an in-panel legend collided with the model labels.
    feat_leg = fig.legend(handles=handles, loc="lower center",
                          bbox_to_anchor=(0.5, 0.095), ncol=len(handles),
                          fontsize=FS_TICK, frameon=False,
                          handletextpad=0.25, columnspacing=1.3,
                          title="feature space")
    feat_leg.get_title().set_fontsize(FS_TICK)
    feat_leg.get_title().set_color("0.35")

    # Estimator identity was previously readable only from panel (b)'s bar
    # labels, which forces the reader to match a bar to a point by eye.  Marker
    # shape now carries it directly in panel (a); this legend decodes it.  Drawn
    # in grey so shape is read as estimator and colour stays feature space.
    est_handles = [mlines.Line2D([0], [0], marker=model_marker(m), color="w",
                                 markersize=4.2, markerfacecolor="0.35",
                                 markeredgecolor="none", label=model_label(m))
                   for m in sorted(has_cv["Model"].unique())]
    est_leg = fig.legend(handles=est_handles, loc="lower center",
                         bbox_to_anchor=(0.5, 0.0), ncol=len(est_handles),
                         fontsize=FS_TICK, frameon=False,
                         handletextpad=0.2, columnspacing=0.75,
                         title="estimator")
    est_leg.get_title().set_fontsize(FS_TICK)
    est_leg.get_title().set_color("0.35")
    plt.setp(ax_a.get_yticklabels(), visible=False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.spines["left"].set_visible(False)

    # ── (a, left strip) graph models — no CV R2 exists for these ─────────────
    for k, (_, r) in enumerate(graph.iterrows()):
        col = FEAT_COLORS.get("Graph", "0.4")
        # Keep the two graph models visible without squeezing long model names
        # into the deliberately narrow no-CV strip.  Panel (b) supplies their
        # full names; the shared purple feature-space legend identifies them.
        ax_g.scatter(0.37 + 0.26 * k, r["Pearson_r"], color=col, s=24,
                     edgecolors="white", linewidths=0.4, zorder=3)
    ax_g.set_xlim(0, 1)
    ax_g.set_xticks([])
    ax_g.set_ylabel("Pearson r — external set")
    # The strip is only ~0.6 in wide, so an xlabel wide enough to name it spills
    # onto its own y tick labels.  Title it instead, and note the missing CV R2
    # inside the axes (Unicode superscript: mathtext would render at 0.7x = 4.2 pt).
    ax_g.set_title("Graph", fontsize=FS_BASE, pad=3)
    ax_g.text(0.5, 0.008, "external only", transform=ax_g.transAxes,
              fontsize=FS_MIN, color="0.45", ha="center", va="bottom")
    ax_g.axhline(0, color="0.6", ls=":", lw=0.6, zorder=1)
    ax_g.spines["bottom"].set_visible(False)

    # ── (b) ranked external Pearson r ─────────────────────────────────────────
    srt = ext_corr.sort_values("Pearson_r", ascending=True).copy()
    srt["label"] = (srt["Model"].replace(MODEL_DISPLAY)
                    + " [" + srt["Features"] + "]")
    bar_colors = [FEAT_COLORS.get(f, "0.4") for f in srt["Features"]]
    ax_b.barh(range(len(srt)), srt["Pearson_r"], color=bar_colors,
              edgecolor="white", linewidth=0.3, height=0.75)
    ax_b.set_yticks(range(len(srt)))
    ax_b.set_yticklabels(srt["label"], fontsize=FS_MIN)
    ax_b.yaxis.tick_right()          # keep labels clear of panel (a)
    ax_b.axvline(0, color="black", lw=0.6)
    ax_b.set_xlim(-1.05, 1.05)
    ax_b.set_ylim(-0.7, len(srt) - 0.3)
    ax_b.set_xlabel("Pearson r — external set")
    ax_b.set_title("Ranked by external r", fontsize=FS_BASE, pad=4)
    ax_b.tick_params(axis="y", length=0)
    ax_b.spines["left"].set_visible(False)
    ax_b.grid(axis="x", ls=":", lw=0.4, color="0.85", zorder=0)
    ax_b.set_axisbelow(True)

    # Panel letters in figure coordinates, at the left edge of each panel group.
    # An axes-relative offset has to be retuned whenever the width ratios change
    # and, on the narrow graph strip, lands on the shared y tick labels.
    for ax, letter in ((ax_g, "a"), (ax_b, "b")):
        bb = ax.get_position()
        fig.text(max(0.004, bb.x0 - 0.062), bb.y1 + 0.075, letter,
                 fontsize=FS_BASE + 1, fontweight="bold", va="top", ha="left")

    savefig(fname, fig=fig)


def plot_cliff_mae(tab, fname, df5x5=None):
    """Consolidated scatter plot of MAE vs Cliff MAE including Desc, FP, and Graph features."""
    fig, ax = plt.subplots(figsize=(9, 8))

    markers = {"Desc": "o", "FP": "s", "FCFP": "D", "Graph": "^"}
    models = tab["Model"].unique()
    color_map = {m: PAL[i % len(PAL)] for i, m in enumerate(models)}

    sub = tab.dropna(subset=["MAE", "MAE_cliff"])

    # Build per-(Model, Features) std lookup — only MAE_std available from 5×5
    std_map = {}
    if df5x5 is not None and "MAE_std" in df5x5.columns:
        for _, r5 in df5x5.iterrows():
            std_map[(r5["Model"], r5["Features"])] = r5["MAE_std"]

    xs, ys, lbls = [], [], []
    for _, row in sub.iterrows():
        feat = row["Features"]
        m    = row["Model"]
        std  = std_map.get((m, feat), np.nan)
        if np.isfinite(std):
            ax.errorbar(row["MAE"], row["MAE_cliff"],
                        xerr=std, yerr=std,
                        fmt="none", color=color_map.get(m, "black"),
                        capsize=3, lw=1.2, alpha=0.6, zorder=2)
        ax.scatter(row["MAE"], row["MAE_cliff"],
                   color=color_map.get(m, "black"), marker=markers.get(feat, "o"),
                   s=100, zorder=3)
        xs.append(row["MAE"]); ys.append(row["MAE_cliff"]); lbls.append(m)
    _annotate_scatter(ax, xs, ys, lbls, fontsize=8)

    lo = min(sub["MAE"].min(), sub["MAE_cliff"].min())
    hi = max(sub["MAE"].max(), sub["MAE_cliff"].max())
    pad = (hi - lo) * 0.05
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("MAE — all"); ax.set_ylabel("MAE — cliff")
    eb_note = "\n(bars = ±1 SD over 5 repeats, overall MAE only)" if std_map else ""
    ax.set_title(f"All Models: cliff difficulty (SALI > {SALI_THRESHOLD}){eb_note}")

    legend_elements = [mlines.Line2D([0], [0], marker=m, color='w', label=f,
                              markerfacecolor='gray', markersize=10)
                       for f, m in markers.items() if f in sub["Features"].unique()]
    ax.legend(handles=legend_elements, title="Features Space", loc="upper left")

    plt.tight_layout()
    savefig(fname)


# ═════════════════════════════════════════════════════════════════════════════
# 6a · CLIFF-PAIR DIRECTIONALITY
# ═════════════════════════════════════════════════════════════════════════════

def cliff_pair_directionality(pairs, y_true, y_pred):
    """
    For every SALI cliff pair (i, j, sali_val), determine whether the model
    correctly predicted the sign of the activity difference.

    A pair is *directionally correct* when
        sign(y_true[i] − y_true[j]) == sign(y_pred[i] − y_pred[j]).

    Parameters
    ----------
    pairs  : list of (i, j, sali_val) from compute_sali_pairs()
    y_true : np.ndarray, full-dataset true values (NaN for indices not in CV)
    y_pred : np.ndarray, full-dataset CV predictions (NaN for indices not in CV)

    Returns
    -------
    pair_df : pd.DataFrame  — one row per pair
    pct_correct : float     — percentage of evaluable pairs that are correct
    """
    rows = []
    for i, j, sali_val in pairs:
        yt_i, yt_j = y_true[i], y_true[j]
        yp_i, yp_j = y_pred[i], y_pred[j]
        if np.isnan(yp_i) or np.isnan(yp_j):
            continue   # fold boundary artefact — both must appear in validation
        true_diff = yt_i - yt_j
        pred_diff = yp_i - yp_j
        correct   = int(np.sign(true_diff) == np.sign(pred_diff))
        rows.append({
            "idx_i":      i,
            "idx_j":      j,
            "sali":       sali_val,
            "y_true_i":   yt_i,
            "y_true_j":   yt_j,
            "y_pred_i":   yp_i,
            "y_pred_j":   yp_j,
            "true_delta": true_diff,
            "pred_delta": pred_diff,
            "correct":    correct,
        })
    pair_df = pd.DataFrame(rows)
    pct = 100.0 * pair_df["correct"].mean() if len(pair_df) else np.nan
    return pair_df, pct


def compute_directionality_table(
        X_dict, y, sali_pairs, cliff_mask, folds, hyperopt=False):
    """
    Run CV with prediction storage for every (model, feature) combination and
    compute per-pair directionality.

    Returns
    -------
    dir_tab    : pd.DataFrame  — one row per (Model, Features) with pct_correct
    pair_store : dict          — (model_name, feat_label) → pair_df
    """
    dir_rows   = []
    pair_store = {}

    for feat_label, X in X_dict.items():
        print(f"\n  ── directionality / {feat_label} ──")
        for name, pipe in tqdm(make_sklearn_models(hyperopt=hyperopt).items(),
                               desc=f"dir/{feat_label}"):
            metrics, yt_full, yp_full = cv_evaluate_return_preds(
                pipe, X, y, cliff_mask, folds)
            pair_df, pct = cliff_pair_directionality(sali_pairs, yt_full, yp_full)
            pair_store[(name, feat_label)] = pair_df
            dir_rows.append({
                "Model":       name,
                "Features":    feat_label,
                "pct_correct": pct,
                "n_pairs":     len(pair_df),
                "MAE":         metrics["MAE"],
                "MAE_cliff":   metrics["MAE_cliff"],
            })
            print(f"    {name:14s}  pct_correct={pct:.1f}%  n_pairs={len(pair_df)}")

    dir_tab = (pd.DataFrame(dir_rows)
               .sort_values(["Features", "pct_correct"], ascending=[True, False])
               .reset_index(drop=True))
    return dir_tab, pair_store


def plot_cliff_directionality_bars(dir_tab, fname):
    """
    Main-text Figure 4: percentage of SALI cliff pairs whose rank order is
    predicted correctly, one panel per feature space, models sorted worst to
    best within each panel.  Drawn at FIG_W2 so it needs no rescaling.

    Bars are coloured by feature space (FEAT_COLORS) rather than by model:
    model identity is already carried by the y tick labels, so colour is free
    to encode the panel grouping and match the other main-text figures.  Bars
    at or below chance are hatched, since "worse than a coin flip" is the one
    distinction a reader must not miss.
    """
    feat_labels = [f for f in FEAT_ORDER if f in dir_tab["Features"].unique()]
    n_panels = len(feat_labels)
    n_rows   = max(int((dir_tab["Features"] == f).sum()) for f in feat_labels)

    apply_print_style()
    fig, axes = plt.subplots(
        1, n_panels, sharex=True,
        figsize=(FIG_W2, max(2.1, 0.155 * n_rows + 0.95)))
    if n_panels == 1:
        axes = [axes]

    for k, (ax, feat) in enumerate(zip(axes, feat_labels)):
        sub = (dir_tab[dir_tab["Features"] == feat]
               .sort_values("pct_correct", ascending=True))
        col = FEAT_COLORS.get(feat, "0.4")
        bars = ax.barh(range(len(sub)), sub["pct_correct"], height=0.7,
                       color=col, edgecolor="white", linewidth=0.3, zorder=3)
        # Mark at-or-below-chance bars so they cannot be misread as weak-positive
        for bar, val in zip(bars, sub["pct_correct"]):
            if val <= 50:
                bar.set_hatch("///")
                bar.set_edgecolor("white")

        for bar, val in zip(bars, sub["pct_correct"]):
            ax.text(bar.get_width() + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", ha="left",
                    fontsize=FS_MIN, color="0.25")

        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["Model"].replace(MODEL_DISPLAY), fontsize=FS_TICK)
        ax.axvline(50, color="0.45", ls=":", lw=0.7, zorder=2)
        n_p = int(sub["n_pairs"].iloc[0]) if len(sub) else 0
        ax.set_xlabel("% cliff pairs ranked correctly")
        ax.set_title(f"{feat}  (n = {n_p} pairs)", fontsize=FS_BASE,
                     color=col, pad=3)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.7, n_rows - 0.3)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", ls=":", lw=0.4, color="0.88", zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.30, 1.06, "abcdefg"[k], transform=ax.transAxes,
                fontsize=FS_BASE + 1, fontweight="bold", va="top", ha="left")

    # One annotation of the chance line, on the first panel only, placed below
    # the axis so it cannot collide with a bar or its value label
    axes[0].annotate("chance", xy=(50, -0.62), xytext=(2, 0),
                     textcoords="offset points", fontsize=FS_MIN,
                     color="0.45", va="center", ha="left")

    fig.tight_layout(w_pad=1.4)
    savefig(fname, fig=fig)


def plot_mega_directionality_vs_mae(dir_tab, fname):
    """
    Mega-scatter (same style as plot_mega_spearman / plot_mega_pearson):
    x = overall CV MAE,  y = % directionally correct cliff pairs.
    Marker shape = feature space,  colour = model.
    """
    all_models = sorted(dir_tab["Model"].unique())
    model_color_map = {m: PAL[i % len(PAL)] for i, m in enumerate(all_models)}

    xs, ys, feats, models = [], [], [], []
    for _, row in dir_tab.dropna(subset=["MAE", "pct_correct"]).iterrows():
        xs.append(row["MAE"])
        ys.append(row["pct_correct"])
        feats.append(row["Features"])
        models.append(row["Model"])

    fig, ax = plt.subplots(figsize=(9, 7))
    _mega_scatter(ax, xs, ys, feats, models, model_color_map,
                  xlabel="CV MAE (all compounds)",
                  ylabel="% cliff pairs predicted directionally correct")
    ax.axhline(50,  color="grey",   ls=":",  lw=1.0, alpha=0.7, label="Chance (50%)")
    ax.axhline(100, color="black",  ls="--", lw=0.8, alpha=0.7, label="Perfect (100%)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Cliff directionality vs overall MAE\n"
                 "(lower MAE + higher % correct = better at cliffs)\n"
                 "marker = feature space,  colour = model")
    plt.tight_layout()
    savefig(fname)


def plot_sali_pair_heatmap(dir_tab, pair_store, fname, top_n_pairs=40):
    """
    Heatmap: rows = SALI cliff pairs (sorted by SALI score, top N shown),
    columns = (Model, Features), cell = 1 (correct, green) / 0 (wrong, red).
    Provides a detailed view of which pairs trip up which models.
    """
    # Collect the union of top pairs across all models by SALI value
    all_pairs_seen = {}
    for pair_df in pair_store.values():
        for _, r in pair_df.iterrows():
            key = (int(r["idx_i"]), int(r["idx_j"]))
            if key not in all_pairs_seen:
                all_pairs_seen[key] = r["sali"]

    # Sort by SALI descending, keep top_n_pairs
    top_keys = sorted(all_pairs_seen, key=lambda k: all_pairs_seen[k], reverse=True)[:top_n_pairs]
    pair_labels = [f"({i},{j})  SALI={all_pairs_seen[(i,j)]:.1f}" for i, j in top_keys]

    col_labels = [f"{m}\n[{f}]" for (m, f) in pair_store.keys()]
    mat = np.full((len(top_keys), len(pair_store)), np.nan)

    for col_idx, ((m, f), pair_df) in enumerate(pair_store.items()):
        idx_map = {(int(r["idx_i"]), int(r["idx_j"])): r["correct"]
                   for _, r in pair_df.iterrows()}
        for row_idx, key in enumerate(top_keys):
            if key in idx_map:
                mat[row_idx, col_idx] = idx_map[key]

    fig_h = max(6, len(top_keys) * 0.28)
    fig_w = max(8, len(col_labels) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 ticks=[0, 1], label="0=wrong  1=correct")
    ax.set_title(f"Per-pair directionality: top {len(top_keys)} SALI cliff pairs\n"
                 "Green = model predicts correct sign  |  Red = wrong direction",
                 fontsize=10)
    plt.tight_layout()
    savefig(fname)


# ═════════════════════════════════════════════════════════════════════════════
# 6a-2 · DIRECTIONALITY vs SALI SCORE
# ═════════════════════════════════════════════════════════════════════════════

def plot_directionality_vs_sali(pair_store, fname, n_bins=5):
    """
    Two-panel figure per feature space:
      Left  — scatter: x=SALI score, y=correct(1)/wrong(0), one trace per model,
               with a LOWESS smoothed trend line.
      Right — binned bar chart: mean % correct per SALI-score bin, all models
               overlaid as lines. Answers: "do models struggle more on the hardest
               (highest-SALI) pairs or is failure randomly distributed?"
    """
    from scipy.stats import binned_statistic

    feat_labels = sorted({f for (_, f) in pair_store.keys()})
    all_models  = sorted({m for (m, _) in pair_store.keys()})
    color_map   = {m: PAL[i % len(PAL)] for i, m in enumerate(all_models)}

    for feat in feat_labels:
        keys = [(m, feat) for m in all_models if (m, feat) in pair_store]
        if not keys:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Collect global SALI range for shared bins
        all_sali = np.concatenate([pair_store[k]["sali"].values for k in keys])
        sali_min, sali_max = all_sali.min(), all_sali.max()
        bin_edges = np.linspace(sali_min, sali_max, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax_sc, ax_bin = axes

        for m in all_models:
            key = (m, feat)
            if key not in pair_store:
                continue
            df_p = pair_store[key].sort_values("sali")
            sali_vals = df_p["sali"].values
            correct   = df_p["correct"].values.astype(float)
            col       = color_map[m]

            # scatter (alpha low — lots of overlap at 0/1)
            ax_sc.scatter(sali_vals, correct + np.random.uniform(-0.03, 0.03, len(correct)),
                          color=col, alpha=0.25, s=18, zorder=2)

            # LOWESS trend
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(correct, sali_vals, frac=0.5, return_sorted=True)
                ax_sc.plot(smoothed[:, 0], smoothed[:, 1], color=col, lw=2.0,
                           label=m, zorder=3)
            except ImportError:
                # fallback: running mean
                window = max(3, len(sali_vals) // 5)
                rolled = pd.Series(correct).rolling(window, center=True, min_periods=1).mean()
                ax_sc.plot(sali_vals, rolled.values, color=col, lw=2.0, label=m, zorder=3)

            # binned means
            means, _, _ = binned_statistic(sali_vals, correct,
                                           statistic="mean", bins=bin_edges)
            valid = ~np.isnan(means)
            ax_bin.plot(bin_centres[valid], 100 * means[valid],
                        color=col, marker="o", ms=5, lw=1.5, label=m)

        ax_sc.axhline(0.5, color="grey", ls=":", lw=1.0, label="Chance")
        ax_sc.set_xlabel("SALI score"); ax_sc.set_ylabel("Directionally correct (jittered)")
        ax_sc.set_ylim(-0.15, 1.15)
        ax_sc.set_title(f"{feat}: directionality vs SALI score\n(LOWESS trend per model)")
        ax_sc.legend(fontsize=7, ncol=2)

        ax_bin.axhline(50, color="grey", ls=":", lw=1.0, label="Chance (50%)")
        ax_bin.set_xlabel(f"SALI score (binned, n={n_bins} bins)")
        ax_bin.set_ylabel("% directionally correct")
        ax_bin.set_title(f"{feat}: binned directionality vs SALI hardness")
        ax_bin.set_ylim(0, 105)
        ax_bin.legend(fontsize=7, ncol=2)

        fig.suptitle(f"Does model directional accuracy depend on cliff hardness? [{feat}]",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        savefig(f"directionality_vs_sali_{feat}.png")


# ═════════════════════════════════════════════════════════════════════════════
# 6a-3 · CROSS-ENDPOINT CONSISTENCY (pIC50 ↔ pKi)
# ═════════════════════════════════════════════════════════════════════════════

def compute_cross_endpoint_consistency(
        pair_store_pic50, pair_store_pki,
        smiles_pic50, smiles_pki):
    """
    Find cliff pairs that exist in *both* endpoints (matched by SMILES) and
    compare per-pair correctness.

    A pair that fails in both endpoints is a structural issue with the feature
    space / model; a pair that fails in only one may be assay noise or
    endpoint-specific SAR.

    Returns a DataFrame with columns:
        smi_i, smi_j, sali_pic50, sali_pki,
        (model, feat)_pic50_correct, (model, feat)_pki_correct,
        consistency  ('both_correct', 'both_wrong', 'discordant')
    """
    # Build SMILES→index maps
    smi2idx_pic50 = {s: i for i, s in enumerate(smiles_pic50)}
    smi2idx_pki   = {s: i for i, s in enumerate(smiles_pki)}

    # Find SMILES present in both datasets
    common_smiles = sorted(set(smiles_pic50) & set(smiles_pki))
    if len(common_smiles) < 2:
        print("  ✗ Cross-endpoint: fewer than 2 SMILES in common — skipping")
        return pd.DataFrame()

    print(f"  Cross-endpoint: {len(common_smiles)} SMILES in common across both endpoints")

    # For each (model, feat) key present in both stores, gather paired correctness
    common_keys = set(pair_store_pic50.keys()) & set(pair_store_pki.keys())
    if not common_keys:
        print("  ✗ No common (model, feat) keys across endpoints — skipping")
        return pd.DataFrame()

    rows = []
    for (m, feat) in sorted(common_keys):
        df_p50 = pair_store_pic50[(m, feat)].copy()
        df_pki = pair_store_pki[(m, feat)].copy()

        # Map dataset indices → SMILES, keep only common-SMILES pairs
        df_p50["smi_i"] = [smiles_pic50[int(r.idx_i)] for _, r in df_p50.iterrows()]
        df_p50["smi_j"] = [smiles_pic50[int(r.idx_j)] for _, r in df_p50.iterrows()]
        df_pki["smi_i"] = [smiles_pki[int(r.idx_i)]   for _, r in df_pki.iterrows()]
        df_pki["smi_j"] = [smiles_pki[int(r.idx_j)]   for _, r in df_pki.iterrows()]

        # Canonical pair key (sorted SMILES tuple)
        def pair_key(r):
            return tuple(sorted([r["smi_i"], r["smi_j"]]))

        df_p50["pair_key"] = df_p50.apply(pair_key, axis=1)
        df_pki["pair_key"] = df_pki.apply(pair_key, axis=1)

        # Filter to pairs where both SMILES are in common_smiles
        common_set = set(common_smiles)
        df_p50 = df_p50[df_p50["smi_i"].isin(common_set) & df_p50["smi_j"].isin(common_set)]
        df_pki = df_pki[df_pki["smi_i"].isin(common_set) & df_pki["smi_j"].isin(common_set)]

        p50_map = df_p50.set_index("pair_key")[["correct", "sali"]].to_dict("index")
        pki_map = df_pki.set_index("pair_key")[["correct", "sali"]].to_dict("index")

        shared_pairs = set(p50_map.keys()) & set(pki_map.keys())
        for pk in shared_pairs:
            c50 = p50_map[pk]["correct"]
            cki = pki_map[pk]["correct"]
            if c50 == 1 and cki == 1:
                consistency = "both_correct"
            elif c50 == 0 and cki == 0:
                consistency = "both_wrong"
            else:
                consistency = "discordant"
            rows.append({
                "Model":       m,
                "Features":    feat,
                "smi_i":       pk[0],
                "smi_j":       pk[1],
                "sali_pic50":  p50_map[pk]["sali"],
                "sali_pki":    pki_map[pk]["sali"],
                "correct_pic50": c50,
                "correct_pki":   cki,
                "consistency": consistency,
            })

    return pd.DataFrame(rows)


def plot_cross_endpoint_consistency(consistency_df, fname):
    """
    Main-text Figure 5: are cliff-pair directionality calls consistent between
    the pIC50 and pKi endpoints?

    (a) stacked bars — share of matched cliff pairs each model gets right on
        both endpoints, on exactly one, or on neither;
    (b) scatter — % correct on pIC50 against % correct on pKi, with the
        identity line marking perfect cross-endpoint agreement.

    Drawn at FIG_W2.  The three outcome categories use a sequential
    green-to-grey-to-red ramp rather than a red/green pair, so the ordering
    (both -> one -> neither) survives greyscale printing and colour-vision
    deficiency.
    """
    if consistency_df.empty:
        print("  ✗ Cross-endpoint consistency: no data — skipping plot")
        return

    cats    = ["both_correct", "discordant", "both_wrong"]
    cat_lbl = {"both_correct": "both endpoints",
               "discordant":   "one endpoint",
               "both_wrong":   "neither"}
    cat_col = {"both_correct": "#1B7837",     # dark green
               "discordant":   "#BDBDBD",     # neutral grey
               "both_wrong":   "#B2182B"}     # dark red

    grp = (consistency_df.groupby(["Model", "Features", "consistency"])
           .size().unstack(fill_value=0).reset_index())
    for c in cats:
        if c not in grp.columns:
            grp[c] = 0
    grp["total"] = grp[cats].sum(axis=1)
    for c in cats:
        grp[f"pct_{c}"] = 100.0 * grp[c] / grp["total"].replace(0, np.nan)

    grp["label"] = (grp["Model"].replace(MODEL_DISPLAY)
                    + " [" + grp["Features"] + "]")
    grp = grp.sort_values("pct_both_correct", ascending=True)

    apply_print_style()
    # Panel (b) is a conventional agreement scatter: its extra width makes the
    # endpoint relationship legible without turning coincident scores into a
    # text grid.
    fig_h = max(3.0, 0.135 * len(grp) + 1.6)
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W2, fig_h),
        gridspec_kw=dict(width_ratios=[1.0, 1.65], wspace=0.30))

    # ── (a) stacked outcome bars ─────────────────────────────────────────────
    ax = axes[0]
    left = np.zeros(len(grp))
    ypos = np.arange(len(grp))
    for c in cats:
        vals = grp[f"pct_{c}"].fillna(0).values
        ax.barh(ypos, vals, left=left, height=0.72, color=cat_col[c],
                label=cat_lbl[c], edgecolor="white", linewidth=0.3, zorder=3)
        left += vals
    ax.set_yticks(ypos)
    ax.set_yticklabels(grp["label"], fontsize=FS_MIN)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(grp) - 0.3)
    ax.set_xlabel("% of matched cliff pairs")
    # Self-contained: the legend that completes this sentence sits at the far
    # bottom-left of the figure, so a trailing ellipsis here reads as clipped
    # text rather than as a lead-in.
    ax.set_title("Endpoints with correct direction",
                 fontsize=FS_BASE, pad=3)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    # Anchored to the figure rather than the axes: an axes-relative offset has to
    # be retuned whenever the bar count changes, and drifted onto the x label.
    ax.legend(fontsize=FS_TICK, loc="lower left",
              bbox_to_anchor=(0.005, 0.0), bbox_transform=fig.transFigure,
              ncol=3, frameon=False, handlelength=1.1, handletextpad=0.4,
              columnspacing=1.0)

    # ── (b) endpoint agreement scatter ───────────────────────────────────────
    ax2 = axes[1]
    grp2 = (consistency_df.groupby(["Model", "Features"])
            .apply(lambda d: pd.Series({
                "pct_correct_pic50": 100 * d["correct_pic50"].mean(),
                "pct_correct_pki":   100 * d["correct_pki"].mean(),
            }), include_groups=False).reset_index())

    n_pairs_cd = consistency_df.groupby(["Model", "Features"]).size().max()
    step  = 100.0 / n_pairs_cd
    # Some configurations have exactly the same two endpoint scores.  Small,
    # deterministic radial offsets reveal their multiplicity without changing
    # the underlying scores or forcing the reader to decode a label lattice.
    cells = {}
    for idx, row in grp2.iterrows():
        cells.setdefault((row["pct_correct_pic50"], row["pct_correct_pki"]),
                         []).append(idx)
    offsets = {}
    for (cx, cy), members in cells.items():
        n_members = len(members)
        if n_members == 1:
            offsets[members[0]] = (cx, cy)
            continue
        radius = min(step * 0.20, 2.0)
        for j, idx in enumerate(sorted(members)):
            angle = 2 * np.pi * j / n_members
            offsets[idx] = (cx + radius * np.cos(angle),
                            cy + radius * np.sin(angle))
    for idx, row in grp2.iterrows():
        x, y = offsets[idx]
        ax2.scatter(x, y, s=32, color=FEAT_COLORS.get(row["Features"], "0.4"),
                    edgecolors="white", linewidths=0.45, alpha=0.90, zorder=3)

    ax2.plot([0, 100], [0, 100], color="0.35", ls="--", lw=0.8, zorder=1)
    ax2.grid(color="0.90", lw=0.5, zorder=0)
    axis_ticks = [0, 25, 50, 75, 100]
    ax2.set_xticks(axis_ticks); ax2.set_yticks(axis_ticks)
    ax2.set_xlim(-3, 103); ax2.set_ylim(-3, 103)
    ax2.set_xlabel("% correct — pIC$_{50}$")
    ax2.set_ylabel("% correct — pK$_\\mathrm{i}$")
    ax2.set_title("Agreement between endpoints", fontsize=FS_BASE, pad=3)
    ax2.tick_params(length=2)
    ax2.set_aspect("equal", adjustable="box")
    for sp in ax2.spines.values():
        sp.set_color("0.75")
    ax2.text(0.98, 0.02, "dashed line = equal accuracy\nsmall offsets separate ties",
             transform=ax2.transAxes, fontsize=FS_MIN, color="0.40",
             ha="right", va="bottom", zorder=5)

    handles = [mlines.Line2D([0], [0], marker="o", color="w", markersize=4,
                             markerfacecolor=FEAT_COLORS[f], label=f)
               for f in FEAT_ORDER if f in set(grp2["Features"])]
    ax2.legend(handles=handles, fontsize=FS_TICK, loc="upper right",
               title="Feature space", title_fontsize=FS_TICK,
               handletextpad=0.25, labelspacing=0.25, frameon=False)


    for ax_, letter in ((axes[0], "a"), (axes[1], "b")):
        ax_.text(-0.30 if letter == "a" else -0.14, 1.055, letter,
                 transform=ax_.transAxes, fontsize=FS_BASE + 1,
                 fontweight="bold", va="top", ha="left")

    fig.tight_layout(rect=(0, 0.055, 1, 1), w_pad=1.2)
    savefig(fname, fig=fig)

    overall = (consistency_df.groupby("consistency").size() /
               len(consistency_df) * 100).round(1)
    print("  Cross-endpoint consistency (pooled across models):")
    for k, v in overall.items():
        print(f"    {k:20s}  {v:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# 6a-4 · LEAVE-CLIFF-OUT CV
# ═════════════════════════════════════════════════════════════════════════════

def run_leave_cliff_out_cv(X_dict, y, cliff_mask, folds):
    """
    Train on non-cliff compounds only; evaluate on cliff compounds only.
    Uses the same fold indices as the standard CV so the splits are comparable.

    Returns a DataFrame with MAE_lco and pct_correct_lco per (Model, Features).
    Also returns a pair_store_lco with per-pair correctness.
    """
    from sklearn.base import clone

    # Build the SALI pair list restricted to cliff indices, using y
    # (we just need the cliff-compound predictions for directionality;
    #  the pairs are evaluated post-hoc from the LCO predictions)

    rows        = []
    pred_store  = {}   # (name, feat) → (yt_full, yp_full) over cliff-only val sets

    for feat_label, X in X_dict.items():
        print(f"\n  ── leave-cliff-out / {feat_label} ──")
        for name, pipe in tqdm(make_sklearn_models().items(),
                               desc=f"lco/{feat_label}"):
            yt_cliff_all, yp_cliff_all = [], []
            yt_full = np.full(len(y), np.nan)
            yp_full = np.full(len(y), np.nan)

            for tr_idx, val_idx in folds:
                # Only train on non-cliff in training fold
                non_cliff_tr = tr_idx[~cliff_mask[tr_idx]]
                if len(non_cliff_tr) < 5:
                    # Too few non-cliff training points — skip fold
                    continue
                # Evaluate on cliff compounds in validation fold
                cliff_val = val_idx[cliff_mask[val_idx]]
                if len(cliff_val) == 0:
                    continue

                p = clone(pipe)
                p.fit(X[non_cliff_tr], y[non_cliff_tr])
                yhat_cliff = p.predict(X[cliff_val])

                yt_cliff_all.extend(y[cliff_val])
                yp_cliff_all.extend(yhat_cliff)
                yt_full[cliff_val] = y[cliff_val]
                yp_full[cliff_val] = yhat_cliff

            if len(yt_cliff_all) < 2:
                mae_lco = np.nan
            else:
                mae_lco = mean_absolute_error(yt_cliff_all, yp_cliff_all)

            pred_store[(name, feat_label)] = (yt_full, yp_full)
            rows.append({
                "Model":    name,
                "Features": feat_label,
                "MAE_lco":  mae_lco,
                "n_cliff_preds": len(yt_cliff_all),
            })
            print(f"    {name:14s}  MAE_lco={mae_lco:.3f}  n={len(yt_cliff_all)}")

    return pd.DataFrame(rows), pred_store


def plot_leave_cliff_out(lco_tab, dir_tab, fname):
    """
    Two-panel figure:
      Left  — bar chart of MAE_lco per method, with MAE_cliff from standard CV
               overlaid as dots (to show the gap from "has seen cliffs in training"
               vs "never seen cliffs").
      Right — scatter: MAE_cliff (standard CV) vs MAE_lco, diagonal = no gap.
               Larger distance above diagonal = more the model relied on seeing
               cliffs in training.
    """
    merged = lco_tab.merge(
        dir_tab[["Model", "Features", "MAE", "MAE_cliff"]],
        on=["Model", "Features"], how="left")
    merged = merged.dropna(subset=["MAE_lco", "MAE_cliff"])

    feat_labels = [f for f in ["Desc", "FP", "FCFP"]
                   if f in merged["Features"].unique()]
    n_p = len(feat_labels)
    all_models = merged["Model"].unique()
    color_map  = {m: PAL[i % len(PAL)] for i, m in enumerate(all_models)}

    fig, axes = plt.subplots(1, n_p + 1,
                             figsize=(5 * (n_p + 1), max(5, len(merged) // n_p * 0.5 + 2)))

    # Left panels — per-feature bar+dot
    for ax, feat in zip(axes[:n_p], feat_labels):
        sub = merged[merged["Features"] == feat].sort_values("MAE_lco", ascending=True)
        colors = [color_map[m] for m in sub["Model"]]
        ax.barh(sub["Model"], sub["MAE_lco"], color=colors,
                edgecolor="white", alpha=0.85, height=0.6,
                label="MAE (LCO — never trained on cliffs)")
        ax.scatter(sub["MAE_cliff"], sub["Model"],
                   color="black", marker="D", s=55, zorder=4,
                   label="MAE_cliff (standard CV)")
        ax.set_xlabel("MAE")
        ax.set_title(f"{feat}: LCO vs standard CV\ncliff MAE")
        ax.legend(fontsize=7)

    # Right panel — scatter MAE_cliff vs MAE_lco
    ax_s = axes[-1]
    xs, ys, lbls = [], [], []
    for _, row in merged.iterrows():
        col    = color_map[row["Model"]]
        marker = _FEAT_MARKERS.get(row["Features"], "o")
        ax_s.scatter(row["MAE_cliff"], row["MAE_lco"],
                     color=col, marker=marker, s=100, zorder=3,
                     edgecolors="white", linewidths=0.5)
        xs.append(row["MAE_cliff"]); ys.append(row["MAE_lco"])
        lbls.append(row["Model"])
    _annotate_scatter(ax_s, xs, ys, lbls, fontsize=7)

    lo = min(min(xs), min(ys)) - 0.02
    hi = max(max(xs), max(ys)) + 0.02
    ax_s.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="No gap (y=x)")
    ax_s.set_xlim(lo, hi); ax_s.set_ylim(lo, hi)
    ax_s.set_xlabel("MAE_cliff — standard CV\n(trained on some cliffs)")
    ax_s.set_ylabel("MAE_lco — leave-cliff-out\n(never trained on cliffs)")
    ax_s.set_title("Cliff generalisation gap\n(above diagonal = relies on seeing cliffs)")
    feat_handles = [mlines.Line2D([0], [0], marker=v, color="w",
                                  markerfacecolor="grey", markersize=9, label=k)
                    for k, v in _FEAT_MARKERS.items()
                    if k in merged["Features"].unique()]
    ax_s.legend(handles=feat_handles, title="Feature space", fontsize=7, title_fontsize=8)

    fig.suptitle("Leave-Cliff-Out CV: can models generalise to cliffs\n"
                 "without seeing any cliff compounds in training?",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    savefig(fname)


# ═════════════════════════════════════════════════════════════════════════════
# 6a-5 · REPEATED DIRECTIONALITY UNCERTAINTY (--cv5x5 extension)
# ═════════════════════════════════════════════════════════════════════════════

def cv_repeated_directionality(model_pipeline, X, y, cliff_mask, sali_pairs,
                                n_splits=5, n_repeats=5, seed=SEED):
    """
    Run n_repeats × n_splits-fold CV, storing per-compound predictions each repeat.
    For every cliff pair compute:
      - mean_correct  : fraction of repeats the pair was directionally correct
      - std_correct   : std of that fraction (calibration proxy)
      - mean_abs_pred_delta : mean |ŷᵢ − ŷⱼ| across repeats (confidence magnitude)
      - std_pred_delta      : std of pred delta (spread = uncertainty)
    """
    import copy
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    # per_repeat[rep_idx][(i,j)] = correct (0/1)
    n = len(y)
    n_total_folds = n_splits * n_repeats
    # Store per-fold predictions indexed by fold number (folds within a repeat share indices)
    repeat_pred_store = []   # list of (yt_full, yp_full) per repeat

    yp_by_fold = []    # one entry per fold
    yt_by_fold = []
    fold_val_idxs = []

    for tr_idx, val_idx in rkf.split(X):
        p = copy.deepcopy(model_pipeline)
        p.fit(X[tr_idx], y[tr_idx])
        yhat = p.predict(X[val_idx])
        yp_by_fold.append((val_idx, yhat))
        yt_by_fold.append((val_idx, y[val_idx]))

    # Reassemble into per-repeat full arrays
    for rep in range(n_repeats):
        start = rep * n_splits
        yt_full = np.full(n, np.nan)
        yp_full = np.full(n, np.nan)
        for fold_idx in range(n_splits):
            val_idx, yhat = yp_by_fold[start + fold_idx]
            yt_full[val_idx] = yt_by_fold[start + fold_idx][1]
            yp_full[val_idx] = yhat
        repeat_pred_store.append((yt_full, yp_full))

    # Per-pair stats across repeats
    rows = []
    for (i, j, sali_val) in sali_pairs:
        per_rep_correct    = []
        per_rep_pred_delta = []
        for yt_full, yp_full in repeat_pred_store:
            if np.isnan(yp_full[i]) or np.isnan(yp_full[j]):
                continue
            true_delta = yt_full[i] - yt_full[j]
            pred_delta = yp_full[i] - yp_full[j]
            per_rep_correct.append(int(np.sign(true_delta) == np.sign(pred_delta)))
            per_rep_pred_delta.append(pred_delta)
        if len(per_rep_correct) == 0:
            continue
        rows.append({
            "idx_i":              i,
            "idx_j":              j,
            "sali":               sali_val,
            "mean_correct":       np.mean(per_rep_correct),
            "std_correct":        np.std(per_rep_correct, ddof=1) if len(per_rep_correct) > 1 else 0.0,
            "mean_pred_delta":    np.mean(per_rep_pred_delta),
            "std_pred_delta":     np.std(per_rep_pred_delta, ddof=1) if len(per_rep_pred_delta) > 1 else 0.0,
            "n_repeats_evaluated": len(per_rep_correct),
        })
    return pd.DataFrame(rows)


def run_repeated_directionality(X_dict, y, sali_pairs, cliff_mask,
                                 n_splits=5, n_repeats=5):
    """
    Run cv_repeated_directionality for all (model, feature) combinations.
    Returns a dict: (model_name, feat_label) → repeat_pair_df
    """
    store = {}
    for feat_label, X in X_dict.items():
        print(f"\n  ── repeated directionality / {feat_label} ──")
        for name, pipe in tqdm(make_sklearn_models().items(),
                               desc=f"rep_dir/{feat_label}"):
            rep_df = cv_repeated_directionality(
                pipe, X, y, cliff_mask, sali_pairs,
                n_splits=n_splits, n_repeats=n_repeats)
            store[(name, feat_label)] = rep_df
            n_amb = (rep_df["mean_correct"].between(0.2, 0.8)).sum()
            print(f"    {name:14s}  ambiguous pairs (20–80% correct): {n_amb}/{len(rep_df)}")
    return store


def plot_repeated_directionality_uncertainty(rep_dir_store, fname):
    """
    Two-panel figure per feature space:
      Left  — scatter: x = mean_correct (across repeats), y = std_pred_delta.
               Hypothesis: wrongly-directed pairs should show high prediction
               variance (std_pred_delta large) and low mean_correct.
      Right — violin/strip: std_pred_delta split by 'reliably correct' (mean≥0.8),
               'ambiguous' (0.2<mean<0.8), 'reliably wrong' (mean≤0.2).
    """
    feat_labels = sorted({f for (_, f) in rep_dir_store.keys()})
    all_models  = sorted({m for (m, _) in rep_dir_store.keys()})
    color_map   = {m: PAL[i % len(PAL)] for i, m in enumerate(all_models)}

    for feat in feat_labels:
        keys = [(m, feat) for m in all_models if (m, feat) in rep_dir_store]
        if not keys:
            continue

        # Pool all pairs across models for the right violin panel
        pool_rows = []
        for (m, feat_) in keys:
            df_r = rep_dir_store[(m, feat_)].copy()
            df_r["Model"] = m
            pool_rows.append(df_r)
        pool = pd.concat(pool_rows, ignore_index=True)

        def reliability_bucket(mc):
            if mc >= 0.8:   return "reliably\ncorrect"
            if mc <= 0.2:   return "reliably\nwrong"
            return "ambiguous"
        pool["reliability"] = pool["mean_correct"].apply(reliability_bucket)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left — scatter per model
        ax_sc = axes[0]
        for m in all_models:
            key = (m, feat)
            if key not in rep_dir_store:
                continue
            df_r = rep_dir_store[key]
            ax_sc.scatter(df_r["mean_correct"], df_r["std_pred_delta"],
                          color=color_map[m], alpha=0.55, s=30, label=m, zorder=2)

        ax_sc.axvline(0.5,  color="grey",  ls=":",  lw=1.0, label="50% correct")
        ax_sc.axvline(0.8,  color="green", ls="--", lw=0.7, alpha=0.6)
        ax_sc.axvline(0.2,  color="red",   ls="--", lw=0.7, alpha=0.6)
        ax_sc.set_xlabel("Mean directional correctness across repeats")
        ax_sc.set_ylabel("Std of predicted delta (ŷᵢ − ŷⱼ) across repeats")
        ax_sc.set_title(f"{feat}: prediction uncertainty vs correctness\n"
                        "(high spread + low correctness = genuinely confused)")
        ax_sc.legend(fontsize=7, ncol=2)

        # Right — violin by reliability bucket
        ax_v = axes[1]
        order = ["reliably\ncorrect", "ambiguous", "reliably\nwrong"]
        order_present = [o for o in order if o in pool["reliability"].unique()]
        palette = {"reliably\ncorrect": "#2ca02c",
                   "ambiguous":         "#ff7f0e",
                   "reliably\nwrong":   "#d62728"}
        sns.violinplot(data=pool, x="reliability", y="std_pred_delta",
                       order=order_present, palette=palette,
                       inner="quartile", ax=ax_v, cut=0)
        ax_v.set_xlabel("Reliability bucket (pooled across models)")
        ax_v.set_ylabel("Std of predicted delta across repeats")
        ax_v.set_title(f"{feat}: prediction spread by reliability\n"
                       "(well-calibrated model: wrong → high spread)")

        fig.suptitle(f"Repeated-CV uncertainty on cliff pairs [{feat}]\n"
                     f"({len(keys[0][0] if keys else '')} models, "
                     f"{pool['n_repeats_evaluated'].median():.0f} repeats median)",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        savefig(f"repeated_dir_uncertainty_{feat}.png")


def plot_repeated_directionality_summary(rep_dir_store, fname):
    """
    Summary heatmap: rows = cliff pairs (top 40 by SALI), columns = (model, feat),
    cell colour = mean_correct (0=always wrong, 1=always right).
    Overlaid with hatching for high-uncertainty pairs (std_correct > 0.4).
    This is a richer version of the binary cliff_pair_heatmap.
    """
    feat_labels = sorted({f for (_, f) in rep_dir_store.keys()})
    all_models  = sorted({m for (m, _) in rep_dir_store.keys()})

    # Union of pairs across all keys
    all_pairs_seen = {}
    for df_r in rep_dir_store.values():
        for _, r in df_r.iterrows():
            key = (int(r["idx_i"]), int(r["idx_j"]))
            if key not in all_pairs_seen or r["sali"] > all_pairs_seen[key]:
                all_pairs_seen[key] = r["sali"]

    top_keys = sorted(all_pairs_seen, key=lambda k: all_pairs_seen[k], reverse=True)[:40]
    pair_labels = [f"({i},{j})  SALI={all_pairs_seen[(i,j)]:.1f}" for i, j in top_keys]

    col_order = [(m, f) for f in feat_labels for m in all_models
                 if (m, f) in rep_dir_store]
    col_labels = [f"{m}\n[{f}]" for (m, f) in col_order]

    mat_mean = np.full((len(top_keys), len(col_order)), np.nan)
    mat_std  = np.full((len(top_keys), len(col_order)), np.nan)

    for col_idx, key_mf in enumerate(col_order):
        df_r = rep_dir_store[key_mf]
        idx_map = {(int(r["idx_i"]), int(r["idx_j"])): (r["mean_correct"], r["std_correct"])
                   for _, r in df_r.iterrows()}
        for row_idx, pair_key in enumerate(top_keys):
            if pair_key in idx_map:
                mat_mean[row_idx, col_idx] = idx_map[pair_key][0]
                mat_std[row_idx, col_idx]  = idx_map[pair_key][1]

    fig_h = max(7, len(top_keys) * 0.28)
    fig_w = max(10, len(col_labels) * 0.58)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat_mean, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Hatch high-uncertainty cells
    for row_idx in range(len(top_keys)):
        for col_idx in range(len(col_order)):
            if not np.isnan(mat_std[row_idx, col_idx]) and mat_std[row_idx, col_idx] > 0.4:
                rect = plt.Rectangle(
                    (col_idx - 0.5, row_idx - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="black", lw=0.4)
                ax.add_patch(rect)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01,
                 label="Mean directional correctness (0=always wrong, 1=always right)")
    ax.set_title(f"Repeated-CV directionality: top {len(top_keys)} SALI cliff pairs\n"
                 "Colour = mean correctness across repeats  |  Hatch = high uncertainty (std>0.4)",
                 fontsize=10)
    plt.tight_layout()
    savefig(fname)


def plot_ridge_desc(coef_df, fname):
    top30 = coef_df.head(30).sort_values("coef")
    colors = ["steelblue" if c > 0 else "salmon" for c in top30["coef"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top30["feature"], top30["coef"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardised coefficient")
    ax.set_title("Ridge (Desc) — top 30 features | pIC50")
    plt.tight_layout()
    savefig(fname)


def plot_ridge_fp_bars(coef_df, fname):
    top30 = coef_df.head(30).copy()
    top30["feature"] = "bit_" + top30["bit_idx"].astype(str)
    top30 = top30.sort_values("coef")
    colors = ["steelblue" if c > 0 else "salmon" for c in top30["coef"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top30["feature"], top30["coef"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardised coefficient")
    ax.set_title("Ridge (FP) — top 30 Morgan bits | pIC50")
    plt.tight_layout()
    savefig(fname)


def plot_ridge_fp_bits(coef_df, all_mols, bit_info_cache, fname, top_n=12):
    import io as _io
    from PIL import Image as _Image
    from rdkit.Chem.Draw import MolToImage

    top_rows = coef_df.head(top_n)[["bit_idx", "coef"]].values
    ncols = 4
    nrows = int(np.ceil(top_n / ncols))
    panel_w = FIG_W2 / ncols

    apply_print_style()
    # Row height = drawn image aspect (220/280 of the panel width) plus a strip
    # for the per-panel title.  Too small a height and tight_layout pushes the
    # bottom row's axes off the canvas, clipping the structures.
    row_h = panel_w * (220 / 280) + 0.20
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(FIG_W2, nrows * row_h + 0.40))
    sign_color = {True: "#2166ac", False: "#d6604d"}

    def find_best(bit_idx):
        best_mol, best_bi, best_cnt = None, {}, -1
        for mol, bi in zip(all_mols, bit_info_cache):
            if mol is None:
                continue
            if bit_idx in bi and len(bi[bit_idx]) > best_cnt:
                best_cnt = len(bi[bit_idx])
                best_mol, best_bi = mol, bi
        return best_mol, best_bi

    for ax, (bit_idx, coef_val) in zip(axes.flat, top_rows):
        bit_idx  = int(bit_idx)
        positive = coef_val > 0
        best_mol, best_bi = find_best(bit_idx)
        if best_mol is None:
            ax.axis("off")
            ax.set_title(f"bit {bit_idx}\n(absent)", fontsize=8)
            continue
        atom_idx, radius = best_bi[bit_idx][0]

        try:
            # useSVG=False returns encoded PNG bytes, not a PIL image, so the
            # bytes have to be decoded before they can be shown — passing them
            # straight to np.array() yields a 0-d array and silently falls
            # through to the plain-fragment branch below.
            png = Draw.DrawMorganEnv(
                best_mol, atom_idx, radius,
                molSize=(280, 220),
                useSVG=False,
            )
            ax.imshow(np.array(_Image.open(_io.BytesIO(png)).convert("RGB")))
        except Exception:
            env    = Chem.FindAtomEnvironmentOfRadiusN(best_mol, radius, atom_idx)
            amap   = {}
            submol = Chem.PathToSubmol(best_mol, env, atomMap=amap)
            ax.imshow(np.array(MolToImage(submol, size=(280, 220))))

        ax.axis("off")
        sign_str = "+" if positive else "−"
        ax.set_title(f"bit {bit_idx}   {sign_str}{abs(coef_val):.3f}",
                     fontsize=FS_ANNOT, color=sign_color[positive],
                     fontweight="bold", pad=2)

    for ax in axes.flat[top_n:]:
        ax.axis("off")
    # tight_layout already reserves room for a suptitle, so no rect is needed:
    # combining the two counts the title band twice and leaves a dead strip.
    fig.suptitle(f"Top {top_n} Morgan bit environments — Ridge (FP), pIC$_{{50}}$\n"
                 "blue: promotes activity     red: suppresses activity",
                 fontsize=FS_BASE)
    fig.tight_layout(h_pad=0.15, w_pad=0.2)
    savefig(fname, fig=fig)


def plot_external_pearson_bar(ext_corr, fname):
    """
    SI figure: ranked bar chart of Pearson r on the external set, all model x
    feature-space combinations side by side.  Drawn at single-column width with
    a height that scales with the number of bars, so the row labels stay legible
    without the figure having to be shrunk at typesetting.
    """
    srt = ext_corr.sort_values("Pearson_r", ascending=True).copy()
    srt["label"] = (srt["Model"].replace(MODEL_DISPLAY)
                    + " [" + srt["Features"] + "]")
    colors = ["#4878a8" if r >= 0 else "#c26a5a" for r in srt["Pearson_r"]]

    apply_print_style()
    fig, ax = plt.subplots(figsize=(FIG_W1 * 1.35,
                                    max(2.4, 0.135 * len(srt) + 0.75)))
    ax.barh(srt["label"], srt["Pearson_r"], color=colors,
            edgecolor="white", linewidth=0.4, height=0.78)
    ax.axvline(0, color="0.2", lw=0.7)
    ax.axvline(1, color="0.55", ls="--", lw=0.7)
    ax.text(1, len(srt) - 0.4, " perfect ranking", fontsize=FS_MIN,
            color="0.45", ha="left", va="top", rotation=90)
    ax.set_xlabel("Pearson $r$, external set", fontsize=FS_BASE)
    ax.set_xlim(min(-1.05, srt["Pearson_r"].min() - 0.08), 1.18)
    ax.tick_params(axis="y", labelsize=FS_MIN, length=0, pad=1)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.set_title("External-set ranking ability", fontsize=FS_BASE, pad=3)
    ax.grid(axis="x", lw=0.3, color="0.88", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    savefig(fname, fig=fig)


def _sim_map_atom_weights(mol, eff_coefs, radius_filter=None, use_fcfp=False):
    """
    Compute normalised per-atom Ridge weights for a molecule.

    Parameters
    ----------
    mol : RDKit Mol
    eff_coefs : np.ndarray  shape (FP_SIZE,)  — Ridge coefs / scaler.scale_
    radius_filter : int or None
        If set, only accumulate contributions from bitInfo hits at that radius.
    use_fcfp : bool
        If True, collect bitInfo via the FCFP (pharmacophoric) generator so that
        the bit indices match those the FCFP Ridge was trained on.

    Returns
    -------
    atom_weights : np.ndarray  shape (n_atoms,), normalised to [-1, 1]
    """
    bi = {}
    if use_fcfp:
        FCFPGEN.GetSparseCountFingerprint(mol)   # warm up (no-op)
        # FCFP bitInfo via the legacy API with useFeatures=True
        AllChem.GetMorganFingerprintAsBitVect(
            mol, FP_RADIUS, nBits=FP_SIZE, bitInfo=bi, useFeatures=True)
    else:
        AllChem.GetMorganFingerprintAsBitVect(
            mol, FP_RADIUS, nBits=FP_SIZE, bitInfo=bi)

    atom_weights = np.zeros(mol.GetNumAtoms())
    for bit, hits in bi.items():
        w = eff_coefs[bit]
        for atom_idx, hit_radius in hits:
            if radius_filter is None or hit_radius == radius_filter:
                atom_weights[atom_idx] += w

    max_w = np.max(np.abs(atom_weights))
    if max_w > 1e-6:
        atom_weights = atom_weights / max_w
    return atom_weights


def _write_sim_map_png(mol, atom_weights, out_path):
    """Render a SimilarityMap PNG with RdBu colouring and write to out_path."""
    drawer = rdMolDraw2D.MolDraw2DCairo(500, 500)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        atom_weights.tolist(),
        draw2d=drawer,
        colorMap="RdBu",
        alpha=0.3,
    )
    drawer.FinishDrawing()
    with open(out_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    print(f"  → {out_path}")


def plot_ridge_similarity_maps(pipe, ext_smiles, ext_labels,
                               fname_prefix="ridge_sim_map", use_fcfp=False):
    """
    Map Ridge coefficients for Morgan FP/FCFP bits onto 2D structures using RDKit
    SimilarityMaps.  Produces four PNGs per external compound:

      <prefix>_<label>.png        — all radii combined
      <prefix>_<label>_r0.png    — r=0 shell
      <prefix>_<label>_r1.png    — r=1 shell
      <prefix>_<label>_r2.png    — r=2 shell

    Blue = promotes activity, Red = suppresses activity.
    All written to OUT_DIR (results/).
    Set use_fcfp=True when pipe was trained on FCFP so bitInfo is collected
    with the pharmacophoric invariant generator.
    """
    scaler = pipe.named_steps["scl"]
    ridge  = pipe.named_steps["mdl"]
    scale  = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    eff_coefs = ridge.coef_ / scale

    for smi, lbl in zip(ext_smiles, ext_labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  ✗ Could not parse SMILES for {lbl}, skipping sim map")
            continue
        safe_lbl = lbl.replace("/", "_").replace("-", "_")

        # combined (all radii)
        weights_all = _sim_map_atom_weights(mol, eff_coefs,
                                            radius_filter=None, use_fcfp=use_fcfp)
        _write_sim_map_png(mol, weights_all,
                           os.path.join(OUT_DIR, f"{fname_prefix}_{safe_lbl}.png"))

        # per-radius maps
        for r in range(FP_RADIUS + 1):
            weights_r = _sim_map_atom_weights(mol, eff_coefs,
                                              radius_filter=r, use_fcfp=use_fcfp)
            if np.max(np.abs(weights_r)) < 1e-6:
                print(f"  ✗ No r={r} bits for {lbl}, skipping r={r} sim map")
                continue
            _write_sim_map_png(mol, weights_r,
                               os.path.join(OUT_DIR,
                                            f"{fname_prefix}_{safe_lbl}_r{r}.png"))



# ═════════════════════════════════════════════════════════════════════════════
# 6b · PER-RADIUS ECFP HEATMAPS
# ═════════════════════════════════════════════════════════════════════════════

def _fp_by_radius(mol, radius, fp_size=FP_SIZE):
    """Return a count-FP numpy array containing only bits whose ECFP *environment radius* == r."""
    gen_r = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    gen_rm1 = rdFingerprintGenerator.GetMorganGenerator(radius=max(0, radius - 1), fpSize=fp_size)
    fp_r   = gen_r.GetCountFingerprintAsNumPy(mol).astype(float)
    if radius == 0:
        return fp_r
    fp_rm1 = gen_rm1.GetCountFingerprintAsNumPy(mol).astype(float)
    diff = fp_r - fp_rm1
    diff[diff < 0] = 0.0
    return diff


def _coef_df_for_X(X, y, label=""):
    """Fit a Ridge model and return a coefficient DataFrame."""
    pipe = Pipeline([("scl", StandardScaler(with_mean=False)), ("mdl", Ridge(alpha=1.0))])
    pipe.fit(X, y)
    coefs = pipe.named_steps["mdl"].coef_
    df = (pd.DataFrame({"bit_idx": np.arange(len(coefs)), "coef": coefs})
          .assign(abs_coef=lambda d: d["coef"].abs())
          .sort_values("abs_coef", ascending=False)
          .reset_index(drop=True))
    return df


def plot_ridge_fp_heatmaps_by_radius(smiles_list, y, all_mols, bit_info_cache,
                                     fp_size=FP_SIZE, use_fcfp=False):
    """
    Produce per-radius Ridge coefficient bar charts (r=0, r=1, r=2, all).
    Set use_fcfp=True to operate on FCFP bits instead of ECFP.
    Output filenames are prefixed with 'ridge_heatmap_' (ECFP)
    or 'ridge_fcfp_heatmap_' (FCFP).
    """
    prefix = "ridge_fcfp_heatmap" if use_fcfp else "ridge_heatmap"
    fp_label = "FCFP" if use_fcfp else "FP"
    print(f"  Computing per-radius {fp_label} fingerprint matrices …")

    gen_full = (FCFPGEN if use_fcfp
                else rdFingerprintGenerator.GetMorganGenerator(
                    radius=FP_RADIUS, fpSize=fp_size))

    X_all = np.vstack([
        gen_full.GetCountFingerprintAsNumPy(m).astype(float)
        if m is not None else np.zeros(fp_size)
        for m in all_mols
    ])
    X_all = clean(X_all)

    # Build per-radius matrices by differencing successive-radius counts
    radii_Xs = {}
    for r in range(FP_RADIUS + 1):
        if use_fcfp:
            gen_r = rdFingerprintGenerator.GetMorganGenerator(
                radius=r, fpSize=fp_size,
                atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
            gen_rm1 = (rdFingerprintGenerator.GetMorganGenerator(
                radius=r - 1, fpSize=fp_size,
                atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
                       if r > 0 else None)
        else:
            gen_r   = rdFingerprintGenerator.GetMorganGenerator(radius=r,       fpSize=fp_size)
            gen_rm1 = (rdFingerprintGenerator.GetMorganGenerator(radius=r - 1,  fpSize=fp_size)
                       if r > 0 else None)

        rows = []
        for m in all_mols:
            if m is None:
                rows.append(np.zeros(fp_size))
                continue
            fp_r = gen_r.GetCountFingerprintAsNumPy(m).astype(float)
            if gen_rm1 is None:
                rows.append(fp_r)
            else:
                fp_rm1 = gen_rm1.GetCountFingerprintAsNumPy(m).astype(float)
                diff = fp_r - fp_rm1
                diff[diff < 0] = 0.0
                rows.append(diff)
        radii_Xs[r] = clean(np.vstack(rows))

    def _heatmap_for_coef(coef_df, title, fname, top_n=30):
        top = coef_df.head(top_n).copy()
        top["label"] = "bit_" + top["bit_idx"].astype(str)
        fig, ax = plt.subplots(figsize=(max(8, top_n * 0.35), 3))
        colors = ["steelblue" if c > 0 else "salmon" for c in top["coef"]]
        ax.barh(top["label"], top["coef"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Standardised Ridge coefficient")
        ax.set_title(title)
        plt.tight_layout()
        savefig(fname)

    for r, Xr in radii_Xs.items():
        if Xr.sum() == 0:
            print(f"    r={r}: all-zero matrix, skipping")
            continue
        cdf = _coef_df_for_X(Xr, y)
        _heatmap_for_coef(cdf,
                          f"Ridge ({fp_label}) — r={r} bits only | pIC50",
                          f"{prefix}_r{r}.png")

    cdf_all = _coef_df_for_X(X_all, y)
    _heatmap_for_coef(cdf_all,
                      f"Ridge ({fp_label}) — all bits | pIC50",
                      f"{prefix}_all.png")


# ═════════════════════════════════════════════════════════════════════════════
# 6b · SHAP EXPLANATION OF EXTERNAL PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
def _feature_names(feat_label, X_train=None):
    """
    Return canonical feature names for each feature space.
    """

    # RDKit descriptors
    if feat_label in ("Desc", "RDKit", "Descriptors"):
        return list(_desc_names)

    # Morgan / fingerprint bits
    if feat_label in ("FP", "ECFP", "Morgan"):
        n = X_train.shape[1]
        return [f"bit_{i}" for i in range(n)]

    # FCFP bits
    if feat_label in ("FCFP",):
        n = X_train.shape[1]
        return [f"fcfp_{i}" for i in range(n)]

    # dataframe fallback
    if hasattr(X_train, "columns"):
        return list(X_train.columns)

    # generic fallback
    return [f"f{i}" for i in range(X_train.shape[1])]

def _get_shap_explainer(pipe, X_train, feat_label, model_name):
    """
    Return (explainer, transform_fn, shap_kwargs) for the given fitted pipeline.

    Explainer strategy:
      - Tree models (RF, XGB, LGBM)  → TreeExplainer (exact, fast)
      - Linear models                 → LinearExplainer with the full
                                        transformed training set (exact and
                                        deterministic)
      - Other compatible models       → KernelExplainer, k-means background,
                                        nsamples capped at 128
      - TabPFN                        → (None, None, {}) — skipped entirely.
        KernelExplainer would need O(nsamples × 2048) forward passes through a
        GPU model, which stalls indefinitely at reasonable nsamples values.
    """
    mdl   = pipe.named_steps["mdl"]
    mtype = type(mdl).__name__

    # Hard skip for models incompatible with KernelExplainer at this input size
    if mtype in ("TabPFNRegressor",):
        return None, None, {}

    has_sc = "scl" in pipe.named_steps
    if has_sc:
        scaler = pipe.named_steps["scl"]
        def tx(X): return scaler.transform(X)
        X_bg = scaler.transform(X_train)
    else:
        def tx(X): return X
        X_bg = X_train

    tree_types = ("RandomForestRegressor", "XGBRegressor", "LGBMRegressor")
    linear_types = ("Ridge", "BayesianRidge", "ElasticNet")

    if mtype in tree_types:
        explainer   = shap.TreeExplainer(mdl, data=X_bg,
                                         feature_perturbation="interventional")
        shap_kwargs = {}
    elif mtype in linear_types:
        # Preserve every training observation in the background rather than
        # accepting the masker's default 100-row subsample.  With an
        # Independent masker, LinearExplainer gives the exact interventional
        # attributions beta_j * (x_j - E[X_j]) without Monte Carlo sampling.
        masker = shap.maskers.Independent(X_bg, max_samples=X_bg.shape[0])
        explainer = shap.LinearExplainer(mdl, masker)
        shap_kwargs = {}
    else:
        # Nonlinear, non-tree estimators (SVR and MLP):
        # KernelExplainer with a tiny background and hard nsamples cap.
        # nsamples=128 gives a good Shapley approximation and finishes in
        # seconds even for 2048-dim fingerprints.
        n_bg = min(20, X_bg.shape[0])
        bg   = shap.kmeans(X_bg, n_bg)
        def predict_fn(X_arr):
            return mdl.predict(X_arr)
        explainer   = shap.KernelExplainer(predict_fn, bg)
        shap_kwargs = {"nsamples": 128, "silent": True}

    return explainer, tx, shap_kwargs


def _shap_values_for_ext(pipe, X_train, X_ext, feat_label, model_name):
    """
    Compute SHAP values for X_ext (shape [4, p]) given a fitted pipe.
    Returns (shap_vals [4, p], X_ext_transformed [4, p], feature_names [p]),
    or None if the model type is skipped.
    """
    explainer, tx, shap_kwargs = _get_shap_explainer(
        pipe, X_train, feat_label, model_name)
    if explainer is None:
        return None
    Xe_t = tx(X_ext)
    sv   = explainer.shap_values(Xe_t, **shap_kwargs)
    if isinstance(sv, list):
        sv = sv[0]
    fnames = _feature_names(feat_label, X_train)
    return sv, Xe_t, fnames


def _top_features_union(shap_vals, feat_names, n=20):
    """Return indices of the top-n features by mean |SHAP| across external compounds."""
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    idx = np.argsort(mean_abs)[::-1][:n]
    return idx, [feat_names[i] for i in idx]


# ── 6b-1 · Waterfall plots (per compound × model) ────────────────────────────

def plot_shap_waterfalls(shap_store, ext_labels, out_dir, top_n=15):
    """
    One waterfall figure per (feat_label, model_name), with 4 sub-panels —
    one per external compound.
    """
    print("  SHAP waterfalls …")

    for (feat_label, model_name), (sv, Xe_t, fnames, base_val) in shap_store.items():
        n_ext = sv.shape[0]
        fig, axes = plt.subplots(1, n_ext, figsize=(5 * n_ext, 6), sharey=False)
        if n_ext == 1:
            axes = [axes]

        for ax, k in zip(axes, range(n_ext)):
            vals  = sv[k]
            order = np.argsort(np.abs(vals))[::-1][:top_n]
            v     = vals[order]
            names = [fnames[i] for i in order][::-1]
            v     = v[::-1]

            colors = ["#d62728" if x > 0 else "#1f77b4" for x in v]
            ax.barh(range(len(v)), v, color=colors, edgecolor="white", height=0.7)
            ax.set_yticks(range(len(v)))
            ax.set_yticklabels(names, fontsize=7)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP value", fontsize=8)
            ax.set_title(f"{ext_labels[k]}\n(base={base_val:.2f}  "
                         f"pred={base_val + vals.sum():.2f})",
                         fontsize=8)

        fig.suptitle(f"SHAP waterfall — {model_name} [{feat_label}] — pIC50",
                     fontsize=10, y=1.01)
        plt.tight_layout()
        safe = model_name.replace("+", "_")
        savefig(os.path.join(out_dir,
                             f"shap_waterfall_{feat_label}_{safe}.png"), fig=fig)


# ── 6b-2 · Beeswarm / dot-summary (per feat × model, all 4 externals) ────────

def plot_shap_beeswarm(shap_store, ext_labels, out_dir, top_n=20):
    """
    Dot-summary plot: each external compound is a dot, x=SHAP value, y=feature.
    """
    print("  SHAP beeswarm summaries …")

    ext_colors = ["#1b4332", "#6a0572", "#7b2d00", "#003566"]

    for (feat_label, model_name), (sv, Xe_t, fnames, base_val) in shap_store.items():
        # top features by mean |SHAP|
        top_idx, top_names = _top_features_union(sv, fnames, n=top_n)
        sv_top = sv[:, top_idx]   # [4, top_n]

        fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.32)))
        for k_e, lbl in enumerate(ext_labels):
            y_jitter = np.arange(top_n) + np.random.uniform(-0.15, 0.15, top_n)
            ax.scatter(sv_top[k_e], y_jitter,
                       color=ext_colors[k_e % len(ext_colors)],
                       s=55, alpha=0.85, label=lbl, zorder=3)

        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names, fontsize=7)
        ax.set_xlabel("SHAP value", fontsize=9)
        ax.set_title(f"SHAP summary — {model_name} [{feat_label}] — pIC50\n"
                     f"top {top_n} features by mean |SHAP|", fontsize=9)
        ax.legend(title="External", fontsize=7, title_fontsize=8,
                  loc="lower right", framealpha=0.85)
        plt.tight_layout()
        safe = model_name.replace("+", "_")
        savefig(os.path.join(out_dir,
                             f"shap_beeswarm_{feat_label}_{safe}.png"), fig=fig)


# ── 6b-3 · Cross-model SHAP heatmap (per external compound) ──────────────────

def plot_shap_crossmodel_heatmap(shap_store, ext_labels, out_dir, top_n=15):
    """
    For each external compound: heatmap of (model×feature) SHAP values.
    """
    print("  SHAP cross-model heatmaps …")

    # Group by feat_label
    from collections import defaultdict
    by_feat = defaultdict(dict)
    for (feat_label, model_name), (sv, Xe_t, fnames, base_val) in shap_store.items():
        by_feat[feat_label][model_name] = (sv, fnames)

    for feat_label, model_dict in by_feat.items():
        model_names = list(model_dict.keys())
        fnames      = next(iter(model_dict.values()))[1]

        # Union top features across all models for this feat space
        all_sv = np.vstack([model_dict[m][0] for m in model_names])
        top_idx, top_names = _top_features_union(all_sv, fnames, n=top_n)

        n_ext = next(iter(model_dict.values()))[0].shape[0]
        for k_e, ext_lbl in enumerate(ext_labels[:n_ext]):
            # Build matrix [n_models × top_n]
            mat = np.vstack([
                model_dict[m][0][k_e, top_idx]
                for m in model_names
            ])
            vmax = np.abs(mat).max() or 1.0

            fig, ax = plt.subplots(figsize=(max(8, top_n * 0.6),
                                            max(4, len(model_names) * 0.5)))
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto")
            ax.set_xticks(range(top_n))
            ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(model_names)))
            ax.set_yticklabels(model_names, fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                         label="SHAP value")
            ax.set_title(f"Cross-model SHAP — {feat_label} — {ext_lbl}\n"
                         f"top {top_n} features by pooled mean |SHAP|",
                         fontsize=9)
            plt.tight_layout()
            safe_lbl = ext_lbl.replace("/", "_")
            savefig(os.path.join(out_dir,
                                 f"shap_crossmodel_{feat_label}_{safe_lbl}.png"),
                    fig=fig)


# ── 6b-4 · Ridge weights vs SHAP scatter ─────────────────────────────────────

def plot_ridge_vs_shap(shap_store, ridge_pipes, ext_labels, out_dir, top_n=40):
    """
    Main-text Figure 6 (Desc) / SI (FP, FCFP): standardised Ridge coefficient
    against SHAP value, one sub-panel per external compound, for the top_n
    features by pooled mean |SHAP|.

    Drawn at FIG_W2 with square panels so it needs no rescaling in the
    manuscript.  Points are coloured by whether the two attributions agree in
    sign, and the disagreeing points additionally carry an open marker face so
    the distinction does not rely on hue alone.
    """
    print("  Ridge-vs-SHAP scatter …")

    from matplotlib.patches import Patch
    correlation_rows = []
    feature_rows = []

    for feat_label, ridge_pipe in ridge_pipes.items():
        key = (feat_label, "Ridge")
        if key not in shap_store:
            continue
        sv, Xe_t, fnames, base_val = shap_store[key]

        # The Ridge model is fitted to StandardScaler-transformed descriptors,
        # so coef_ already contains the standardised coefficients.  Dividing
        # these values by scaler.scale_ would instead convert them to raw-unit
        # coefficients and would contradict the axis label and manuscript.
        standardised_coef = ridge_pipe.named_steps["mdl"].coef_   # [p]

        n_ext = sv.shape[0]
        panel_w = FIG_W2 / max(n_ext, 1)

        apply_print_style()
        # Rank the features once, by pooled importance, and label points with
        # that rank rather than the descriptor name.  A panel this wide holds
        # about 32 characters of FS_MIN type, while RDKit descriptor names run
        # to 31 -- two names per panel is the most that can be placed without
        # overlap, which is what cost the figure its interpretability.  A rank
        # number is 1-2 characters, so every point of interest can be marked and
        # resolved through the key printed under the panels.
        mean_abs_all = np.mean(np.abs(sv), axis=0)
        rank_idx = np.argsort(mean_abs_all)[::-1][:top_n]
        rank_of  = {int(fi): r + 1 for r, fi in enumerate(rank_idx)}

        # The footer (shared x label + legend) needs a fixed height in inches;
        # expressing it as a figure fraction leaves a dead band whenever the
        # panel count — and so the figure height — changes.
        foot_in = 0.42
        key_line_in = 0.098          # one line of FS_MIN key text
        fig_h = panel_w + foot_in    # key band added once its height is known
        fig, axes = plt.subplots(
            1, n_ext, figsize=(FIG_W2, fig_h),
            sharex=True, sharey=True)
        if n_ext == 1:
            axes = [axes]

        used_ranks = set()
        mean_abs = np.mean(np.abs(sv), axis=0)          # [p] pooled importance
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]
        names    = [fnames[i] for i in top_idx]
        x_vals   = standardised_coef[top_idx]

        for k, ax in enumerate(axes):
            y_vals = sv[k][top_idx]
            agree  = np.sign(x_vals) == np.sign(y_vals)

            for pos, feature_idx in enumerate(top_idx):
                feature_rows.append({
                    "Features": feat_label,
                    "Compound": ext_labels[k],
                    "Rank": rank_of[int(feature_idx)],
                    "Feature": fnames[int(feature_idx)],
                    "Standardized_Ridge_coefficient": x_vals[pos],
                    "SHAP_value": y_vals[pos],
                    "Sign_agrees": bool(agree[pos]),
                })

            ax.scatter(x_vals[agree], y_vals[agree], s=14,
                       color="#2a9d8f", edgecolors="white", linewidths=0.3,
                       zorder=3, label="sign agrees")
            ax.scatter(x_vals[~agree], y_vals[~agree], s=16,
                       facecolors="none", edgecolors="#e63946", linewidths=0.7,
                       zorder=3, label="sign disagrees")

            # Mark the points that carry the interpretation: those furthest from
            # the origin, plus any sign disagreement (the figure's whole point,
            # and often not an extreme point).  Numbers are short enough that a
            # modest separation floor suffices to keep them apart.
            order = np.argsort(np.abs(x_vals) + np.abs(y_vals))[::-1]
            xr = max(np.ptp(x_vals), 1e-9)
            yr = max(np.ptp(y_vals), 1e-9)
            want = list(order[:6]) + [i for i in np.where(~agree)[0]
                                      if i not in set(order[:6])]
            chosen = []
            for i in want:
                if all(((x_vals[i] - x_vals[j]) / xr) ** 2 +
                       ((y_vals[i] - y_vals[j]) / yr) ** 2 > 0.022
                       for j in chosen):
                    chosen.append(int(i))
            used_ranks.update(rank_of[int(top_idx[i])] for i in chosen)
            texts = [ax.text(x_vals[i], y_vals[i],
                             str(rank_of[int(top_idx[i])]),
                             fontsize=FS_MIN, color="0.15", zorder=4,
                             ha="center", va="center",
                             bbox=dict(boxstyle="square,pad=0.10", fc="white",
                                       ec="none", alpha=0.80))
                     for i in chosen]
            ax.margins(x=0.14, y=0.30)
            obstacles = []

            ax.axhline(0, color="0.6", lw=0.5, ls=":", zorder=1)
            ax.axvline(0, color="0.6", lw=0.5, ls=":", zorder=1)

            # The correlation goes inside the axes: a title long enough to carry
            # it collides with the left-aligned panel letter.
            ax.set_title(ext_labels[k], fontsize=FS_BASE, pad=3)
            if len(x_vals) > 2:
                r, _ = pearsonr(x_vals, y_vals)
                correlation_rows.append({
                    "Features": feat_label,
                    "Compound": ext_labels[k],
                    "Pearson_r": r,
                    "N_features": len(x_vals),
                    "Sign_agreement_fraction": float(np.mean(agree)),
                    "Explainer": "LinearExplainer (interventional)",
                })
                # Put it in whichever corner the data leaves emptiest rather
                # than always bottom-left: with sign disagreement the cloud can
                # reach into any corner, and a fixed position collides with a
                # rank label that adjustText is not free to move off its point.
                xs = (np.asarray(x_vals) - min(x_vals)) / max(np.ptp(x_vals), 1e-9)
                ys = (np.asarray(y_vals) - min(y_vals)) / max(np.ptp(y_vals), 1e-9)
                corners = {(0.03, 0.03): (0.0, 0.0), (0.97, 0.03): (1.0, 0.0),
                           (0.03, 0.97): (0.0, 1.0), (0.97, 0.97): (1.0, 1.0)}
                cx, cy = max(corners, key=lambda c: min(
                    (xs - corners[c][0]) ** 2 + (ys - corners[c][1]) ** 2))
                corr_txt = ax.text(cx, cy, f"r = {r:.2f}",
                                   transform=ax.transAxes, fontsize=FS_MIN,
                                   color="0.3",
                                   ha="left" if cx < 0.5 else "right",
                                   va="bottom" if cy < 0.5 else "top")
                obstacles.append(corr_txt)

            if HAS_ADJUSTTEXT and texts:
                # A rank number is 1-2 characters, so it can be nudged in x as
                # well; the long feature names this replaced could only move in
                # y without running out of the panel.
                _adjust_text(texts, ax=ax, expand=(1.6, 1.6),
                             force_text=(0.8, 0.8), max_move=28,
                             objects=obstacles,
                             arrowprops=dict(arrowstyle="-", color="0.55",
                                             lw=0.3))

            if k == 0:
                ax.set_ylabel("SHAP value")
            # Panel letters are added after tight_layout (below), once the axes
            # have their final positions.

        # A single x label under the row, and one legend above it, so neither
        # can collide with a neighbouring panel or a labelled point.
        # fig.text, not supxlabel: tight_layout reserves space for a supxlabel
        # on top of the rect below, so the footer band gets counted twice and
        # opens a dead strip between the panels and the legend.
        xlab_artist = fig.text(0.5, 0.0, "Standardized Ridge coefficient",
                               fontsize=FS_BASE, ha="center", va="center")
        leg = [Patch(facecolor="#2a9d8f", edgecolor="white",
                     label="coefficient and SHAP have the same sign"),
               Patch(facecolor="none", edgecolor="#e63946",
                     label="coefficient and SHAP have opposite signs")]
        sign_leg = fig.legend(handles=leg, fontsize=FS_TICK, loc="lower center",
                              bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
                              handlelength=1.0, handletextpad=0.4,
                              columnspacing=1.4)

        # ── key: rank number -> descriptor name ─────────────────────────────
        # Wrapped to the figure width by measuring the rendered entries, so the
        # band height follows the number of marked features instead of being a
        # constant that goes wrong when the model or feature set changes.
        entries = [f"{r} {fnames[int(fi)]}"
                   for r, fi in enumerate(rank_idx, start=1)
                   if r in used_ranks]
        renderer = fig.canvas.get_renderer()
        probe = fig.text(0, -1, "", fontsize=FS_MIN)
        key_rows, row = [], []
        for e in entries:
            probe.set_text("   ".join(row + [e]))
            w = probe.get_window_extent(renderer=renderer).width / fig.dpi
            if row and w > FIG_W2 - 0.16:
                key_rows.append(row); row = [e]
            else:
                row.append(e)
        if row:
            key_rows.append(row)
        probe.remove()

        # Grow the canvas for the key rather than squeezing the panels: the
        # panels are square by construction and must stay so.
        key_in = len(key_rows) * key_line_in + 0.07
        fig_h += key_in
        fig.set_size_inches(FIG_W2, fig_h)

        for j, row in enumerate(key_rows):
            fig.text(0.5, (key_in - 0.05 - j * key_line_in) / fig_h,
                     "   ".join(row), fontsize=FS_MIN, color="0.30",
                     ha="center", va="center")

        xlab_artist.set_position((0.5, (key_in + 0.62 * foot_in) / fig_h))
        sign_leg.set_bbox_to_anchor((0.5, (key_in + 0.005) / fig_h),
                                    transform=fig.transFigure)

        fig.tight_layout(w_pad=0.9,
                         rect=(0, (foot_in + key_in) / fig_h, 1, 1))

        # Panel letters as left-aligned titles: any offset from the axes corner
        # large enough to clear the y tick labels lands on the panel next door.
        for k, ax in enumerate(axes):
            ax.set_title("abcdefgh"[k], loc="left",
                         fontsize=FS_BASE + 1, fontweight="bold", pad=3)
        savefig(os.path.join(out_dir, f"shap_ridge_vs_shap_{feat_label}.png"),
                fig=fig)

    # Persist the exact values used by the panels so that the manuscript text,
    # caption and figure can all be generated from one deterministic result.
    pd.DataFrame(correlation_rows).to_csv(
        os.path.join(out_dir, "ridge_shap_correlations.csv"), index=False)
    pd.DataFrame(feature_rows).to_csv(
        os.path.join(out_dir, "ridge_shap_feature_values.csv"), index=False)


# ── 6b · Master runner ───────────────────────────────────────────────────────

def run_shap_analysis(fitted_models, ext_labels, ridge_pipes, out_dir):
    """
    fitted_models : dict  (feat_label, model_name) →
                          (fitted_pipe, X_train, X_ext)
    ridge_pipes   : dict  feat_label → fitted Ridge pipeline
    out_dir       : str   output directory (created if absent)
    """
    if not HAS_SHAP:
        print("  ✗ shap not available — skipping SHAP analysis "
              "(pip install shap)")
        return

    print("\n[6b] SHAP explanation of external predictions")
    os.makedirs(out_dir, exist_ok=True)

    # ── Compute SHAP values for all (feat, model) combos ──────────────────────
    shap_store = {}   # (feat_label, model_name) → (sv, Xe_t, fnames, base_val)

    for (feat_label, model_name), (pipe, X_train, X_ext) in fitted_models.items():
        print(f"    SHAP: {model_name:16s} [{feat_label}] …", end=" ", flush=True)
        try:
            result = _shap_values_for_ext(
                pipe, X_train, X_ext, feat_label, model_name)
            if result is None:
                print("skipped (model not compatible with SHAP at this input size)")
                continue
            sv, Xe_t, fnames = result
            # Estimate base value
            explainer, tx, _ = _get_shap_explainer(
                pipe, X_train, feat_label, model_name)
            base_val = float(np.mean(explainer.expected_value)
                             if hasattr(explainer.expected_value, "__len__")
                             else explainer.expected_value)
            shap_store[(feat_label, model_name)] = (sv, Xe_t, fnames, base_val)
            print("✓")
        except Exception as e:
            print(f"✗  ({e})")

    if not shap_store:
        print("  No SHAP values computed — skipping figures")
        return

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_shap_waterfalls(shap_store, ext_labels, out_dir)
    plot_shap_beeswarm(shap_store, ext_labels, out_dir)
    plot_shap_crossmodel_heatmap(shap_store, ext_labels, out_dir)
    plot_ridge_vs_shap(shap_store, ridge_pipes, ext_labels, out_dir)

    # Persist mean SHAP values as CSV for downstream inspection
    rows = []
    for (feat_label, model_name), (sv, Xe_t, fnames, base_val) in shap_store.items():
        mean_abs = np.mean(np.abs(sv), axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:50]
        for i in top_idx:
            rows.append({
                "Features":   feat_label,
                "Model":      model_name,
                "Feature":    fnames[i],
                "MeanAbsSHAP": mean_abs[i],
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "shap_top_features.csv"), index=False)
    print(f"  SHAP top-features CSV → {out_dir}/shap_top_features.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 6c · 5×5 REPEATED CV + ERROR BARS + TUKEY HSD
# ═════════════════════════════════════════════════════════════════════════════

def cv_evaluate_repeated(model_pipeline, X, y, cliff_mask, n_splits=5, n_repeats=5, seed=SEED):
    """Run repeated k-fold CV and return per-repeat mean MAE values (for variance estimation)."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    repeat_maes = []
    fold_maes   = []
    for fold_i, (tr_idx, val_idx) in enumerate(rkf.split(X)):
        import copy
        pipe = copy.deepcopy(model_pipeline)
        pipe.fit(X[tr_idx], y[tr_idx])
        yhat = pipe.predict(X[val_idx])
        fold_maes.append(mean_absolute_error(y[val_idx], yhat))
        if (fold_i + 1) % n_splits == 0:
            repeat_maes.append(np.mean(fold_maes[-n_splits:]))
    return {
        "MAE_mean": float(np.mean(repeat_maes)),
        "MAE_std":  float(np.std(repeat_maes, ddof=1)),
        "MAE_repeats": [float(value) for value in repeat_maes],
    }


def run_cv5x5(results_dict, X_dict, y, cliff_mask, folds_base,
              n_splits=5, n_repeats=5):
    """
    Run 5×5 repeated CV for all sklearn models in results_dict keys.
    X_dict: {"Desc": X_desc, "FP": X_fp}
    Returns a DataFrame with columns: Model, Features, MAE_mean, MAE_std, MAE_repeats.
    """
    records = []
    for feat_label, X in X_dict.items():
        print(f"\n  ── {feat_label} (5×5 CV) ──")
        for name, pipe in tqdm(make_sklearn_models().items(), desc=f"5x5/{feat_label}"):
            res = cv_evaluate_repeated(pipe, X, y, cliff_mask,
                                       n_splits=n_splits, n_repeats=n_repeats)
            records.append({"Model": name, "Features": feat_label, **res})
            print(f"    {name:14s}  MAE={res['MAE_mean']:.3f} ± {res['MAE_std']:.3f}")
    return pd.DataFrame(records)


def plot_cv5x5_errorbars(df5x5, fname):
    """Grouped error-bar plot: MAE ± std across 5 repeats, grouped by feature space."""
    feat_labels = [f for f in ["Desc", "FP", "FCFP"]
                   if f in df5x5["Features"].unique()]
    n_panels = len(feat_labels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), sharey=False)
    if n_panels == 1:
        axes = [axes]
    for ax, feat_lbl in zip(axes, feat_labels):
        sub = df5x5[df5x5["Features"] == feat_lbl].sort_values("MAE_mean")
        colors = [PAL[i % 10] for i in range(len(sub))]
        ax.barh(sub["Model"], sub["MAE_mean"], xerr=sub["MAE_std"],
                color=colors, edgecolor="white",
                error_kw=dict(ecolor="black", capsize=4, lw=1.5))
        ax.set_xlabel("MAE (mean ± std over 5 repeats)")
        ax.set_title(f"{feat_lbl}: 5×5 repeated CV — pIC50")
        for i, (_, row) in enumerate(sub.iterrows()):
            for rep_mae in row["MAE_repeats"]:
                ax.scatter(rep_mae, row["Model"], color="black", s=18, alpha=0.5, zorder=4)
    plt.tight_layout()
    savefig(fname)


def compute_tukey_hsd(df5x5, alpha=0.05):
    """
    Pairwise Tukey HSD across all (Model, Features) combinations using the
    per-repeat MAE vectors.  Returns a long-form DataFrame of pairwise results.
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError:
        print("  ✗ statsmodels not found — pip install statsmodels — skipping Tukey HSD")
        return pd.DataFrame()

    groups, values = [], []
    for _, row in df5x5.iterrows():
        lbl = f"{row['Model']} [{row['Features']}]"
        for v in row["MAE_repeats"]:
            groups.append(lbl)
            values.append(v)

    result = pairwise_tukeyhsd(endog=values, groups=groups, alpha=alpha)
    df_out = pd.DataFrame(data=result._results_table.data[1:],
                          columns=result._results_table.data[0])
    return df_out


# ═════════════════════════════════════════════════════════════════════════════
# 6d · FIGURES-ONLY MODE (regenerate from saved CSVs)
# ═════════════════════════════════════════════════════════════════════════════

def _load_results_csv(path):
    """Load a results CSV produced by results_to_df() and restore numeric types."""
    df = pd.read_csv(path)
    for col in ["MAE", "R2", "MAE_cliff", "R2_cliff"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_figures_only():
    """Regenerate all standard figures from CSVs already in OUT_DIR."""
    print("\n[figures_only] Loading saved CSVs …")

    pic50_path = os.path.join(OUT_DIR, "results_pic50.csv")
    pki_path   = os.path.join(OUT_DIR, "results_pki.csv")
    ext_path   = os.path.join(OUT_DIR, "external_results.csv")

    missing = [p for p in [pic50_path, pki_path, ext_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing required CSVs in {OUT_DIR}/: {missing}\n"
            "Run without --figures_only first to generate them.")

    tab_pic50 = _load_results_csv(pic50_path)
    tab_pki   = _load_results_csv(pki_path)
    ext_corr  = _load_results_csv(ext_path)

    # Load 5×5 results if present — enables error bars on the scatter figures
    df5x5 = None
    cv5x5_path = os.path.join(OUT_DIR, "cv5x5_pic50.csv")
    if os.path.exists(cv5x5_path):
        import ast
        df5x5 = pd.read_csv(cv5x5_path)
        if "MAE_repeats" in df5x5.columns:
            df5x5["MAE_repeats"] = df5x5["MAE_repeats"].apply(
                lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
        print(f"  ✓ Loaded 5×5 results from {cv5x5_path} — error bars will be shown")
    else:
        print(f"  ✗ {cv5x5_path} not found — figures drawn without error bars")

    print("[figures_only] Regenerating figures …")
    plot_cv_heatmap(tab_pic50, "5-fold CV — pIC50 (Desc vs FP vs Graph)",
                    "cv_heatmap_pic50.png")
    plot_cv_heatmap(tab_pki,   "5-fold CV — pKi   (Desc vs FP vs Graph)",
                    "cv_heatmap_pki.png")
    plot_desc_vs_fp(tab_pic50, "desc_vs_fp_mae.png", df5x5=df5x5)
    plot_external_pearson_bar(ext_corr, "external_pearson_bar.png")
    plot_metric_correlation(tab_pic50, ext_corr, "Desc", "metric_correlation_desc.png")
    plot_metric_correlation(tab_pic50, ext_corr, "FP",   "metric_correlation_fp.png")
    plot_cv_vs_ext_spearman(tab_pic50, ext_corr, "cv_vs_external_spearman.png",
                            df5x5=df5x5)
    plot_mega_spearman(tab_pic50, ext_corr, "mega_cv_r2_vs_spearman.png", df5x5=df5x5)
    plot_mega_pearson( tab_pic50, ext_corr, "mega_cv_r2_vs_pearson.png",  df5x5=df5x5)
    plot_cliff_mae(tab_pic50, "cliff_mae_comparison.png", df5x5=df5x5)

    # Cliff-specific figures, from the per-pair CSVs written by a full run
    dir_path = os.path.join(OUT_DIR, "cliff_directionality_pic50.csv")
    if os.path.exists(dir_path):
        plot_cliff_directionality_bars(pd.read_csv(dir_path),
                                       "cliff_directionality_bars.png")
    else:
        print(f"  ✗ {dir_path} not found — skipping directionality bars")

    ce_path = os.path.join(OUT_DIR, "cross_endpoint_consistency.csv")
    if os.path.exists(ce_path):
        plot_cross_endpoint_consistency(pd.read_csv(ce_path),
                                        "cross_endpoint_consistency.png")
    else:
        print(f"  ✗ {ce_path} not found — skipping cross-endpoint figure")

    print(f"[figures_only] Done — figures written to {OUT_DIR}/")


# ═════════════════════════════════════════════════════════════════════════════

def run_shap_only():
    """
    Refit only what the SHAP and interpretation figures need — the sklearn
    models on the pIC50 endpoint for Desc/FP/FCFP, plus the three Ridge
    pipelines — and regenerate those figures.  This skips cross-validation,
    the pKi endpoint and the cliff analyses, so it is the cheap path for
    iterating on figure layout without retraining the whole study.
    """
    print("\n[shap_only] Loading and featurising pIC50 data …")
    pic50_df = pd.read_csv("data/pic50_clean.csv")
    df_pic50, X_desc, X_fp, X_fcfp = prep_dataset(pic50_df)
    y = df_pic50["y"].values

    X_desc_ext, _ = featurise(EXT_SMILES, mol_to_desc)
    X_fp_ext,   _ = featurise(EXT_SMILES, mol_to_fp)
    X_fcfp_ext, _ = featurise(EXT_SMILES, mol_to_fcfp)
    X_desc_ext = clean(X_desc_ext)
    X_fp_ext   = clean(X_fp_ext)
    X_fcfp_ext = clean(X_fcfp_ext)

    print("[shap_only] Fitting models on the full pIC50 set …")
    ext_preds, fitted_models = {}, {}
    for feat_label, X_train, X_ext_ in [("Desc", X_desc, X_desc_ext),
                                        ("FP",   X_fp,   X_fp_ext),
                                        ("FCFP", X_fcfp, X_fcfp_ext)]:
        for name, pipe in make_sklearn_models().items():
            pipe.fit(X_train, y)
            ext_preds[(name, feat_label)] = pipe.predict(X_ext_)
            fitted_models[(feat_label, name)] = (pipe, X_train, X_ext_)

    ridge_pipes = {}
    for feat_label, X_train in [("Desc", X_desc), ("FP", X_fp),
                                ("FCFP", X_fcfp)]:
        rp = Pipeline([("scl", StandardScaler(with_mean=False)),
                       ("mdl", Ridge(alpha=1.0))])
        rp.fit(X_train, y)
        ridge_pipes[feat_label] = rp

    plot_external(ext_preds, np.array(EXT_PIC50), EXT_LABELS,
                  "external_predictions.png")

    coef_fp_df = (pd.DataFrame({
        "bit_idx": np.arange(X_fp.shape[1]),
        "coef":    ridge_pipes["FP"].named_steps["mdl"].coef_})
        .reindex(np.argsort(
            -np.abs(ridge_pipes["FP"].named_steps["mdl"].coef_)))
        .reset_index(drop=True))
    all_mols = [Chem.MolFromSmiles(s) for s in df_pic50["smiles"]]
    bit_info_cache = []
    for mol in all_mols:
        bi = {}
        if mol is not None:
            AllChem.GetMorganFingerprintAsBitVect(
                mol, FP_RADIUS, nBits=FP_SIZE, bitInfo=bi)
        bit_info_cache.append(bi)
    plot_ridge_fp_bits(coef_fp_df, all_mols, bit_info_cache,
                       "ridge_fp_morgan_bits.png")

    shap_out = os.path.join(OUT_DIR, "shap")
    os.makedirs(shap_out, exist_ok=True)
    run_shap_analysis(fitted_models=fitted_models,
                      ext_labels=EXT_LABELS,
                      ridge_pipes=ridge_pipes,
                      out_dir=shap_out)
    print(f"[shap_only] Done — figures written to {OUT_DIR}/")

def main(hyperopt=False, sali_sweep=False, cv5x5=False, skip_graph=False):
    print("\n" + "═" * 65)
    print("QSAR NMDA — ligand-based modelling")
    print("═" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading data")
    pki_df   = pd.read_csv("data/pki_clean.csv")
    pic50_df = pd.read_csv("data/pic50_clean.csv")
    print(f"  pKi   shape: {pki_df.shape}")
    print(f"  pIC50 shape: {pic50_df.shape}")

    ext_df = pd.DataFrame({"smiles": EXT_SMILES, "pIC50": EXT_PIC50, "label": EXT_LABELS})
    y_ext  = np.array(EXT_PIC50)

    # ── Featurise ─────────────────────────────────────────────────────────────
    print("\n[2] Featurising")
    print("  pKi:")
    df_pki,   X_desc_pki,   X_fp_pki,   X_fcfp_pki   = prep_dataset(pki_df)
    print("  pIC50:")
    df_pic50, X_desc_pic50, X_fp_pic50, X_fcfp_pic50 = prep_dataset(pic50_df)

    X_fp_ext,   _ = featurise(EXT_SMILES, mol_to_fp)
    X_fcfp_ext, _ = featurise(EXT_SMILES, mol_to_fcfp)
    X_desc_ext, _ = featurise(EXT_SMILES, mol_to_desc)
    X_fp_ext   = clean(X_fp_ext);   X_fcfp_ext = clean(X_fcfp_ext)
    X_desc_ext = clean(X_desc_ext)

    # ── SALI ──────────────────────────────────────────────────────────────────
    print(f"\n[3] Computing SALI (threshold={SALI_THRESHOLD}) — pIC50")
    SALI_mat_pic50   = compute_sali(df_pic50["smiles"].tolist(), df_pic50["y"].values)
    cliff_mask_pic50 = (SALI_mat_pic50 > SALI_THRESHOLD).any(axis=1)
    print(f"  Cliff compounds (pIC50): {cliff_mask_pic50.sum()} / {len(cliff_mask_pic50)}")

    print(f"  Computing SALI — pKi")
    cliff_mask_pki = sali_cliff_mask(
        df_pki["smiles"].tolist(), df_pki["y"].values, SALI_THRESHOLD)
    print(f"  Cliff compounds (pKi):   {cliff_mask_pki.sum()} / {len(cliff_mask_pki)}")

    # ── Folds ─────────────────────────────────────────────────────────────────
    folds_pic50 = make_folds(len(df_pic50))
    folds_pki   = make_folds(len(df_pki))

    y_pic50      = df_pic50["y"].values
    smiles_pic50 = df_pic50["smiles"].values
    y_pki        = df_pki["y"].values
    smiles_pki   = df_pki["smiles"].values

    # ── CV — pIC50 ────────────────────────────────────────────────────────────
    print("\n[4] CV — pIC50")
    results_pic50 = {}

    for feat_label, X in [("Desc", X_desc_pic50), ("FP", X_fp_pic50),
                           ("FCFP", X_fcfp_pic50)]:
        print(f"\n  ── {feat_label} ──")
        model_factory_kwargs = {}
        if hyperopt:
            print(f"  Running Optuna hyperopt for {feat_label} …")
            model_factory_kwargs = dict(hyperopt=True,
                                        study_name=f"pic50_{feat_label}",
                                        X_tr=X, y_tr=y_pic50, folds=folds_pic50)

        for name, pipe in tqdm(make_sklearn_models(**model_factory_kwargs).items(),
                               desc=f"sklearn/{feat_label}"):
            res = cv_evaluate(pipe, X, y_pic50, cliff_mask_pic50, folds_pic50)
            results_pic50[(name, feat_label)] = res
            print(f"    {name:14s}  MAE={res['MAE']:.3f}  R²={res['R2']:.3f}  "
                  f"MAE_cliff={res['MAE_cliff']:.3f}")

        if HAS_TABPFN and feat_label in {"FP", "FCFP"}:
            pipe_t = make_tabpfn_pipeline(device="cuda")
            res = cv_evaluate(pipe_t, X, y_pic50, cliff_mask_pic50, folds_pic50)
            results_pic50[("TabPFN", feat_label)] = res
            print(f"    {'TabPFN':14s}  MAE={res['MAE']:.3f}  R²={res['R2']:.3f}")

    if HAS_CHEMPROP and not skip_graph:
        print("\n  ── ChemProp (graph only) ──")
        res = cv_chemprop(smiles_pic50, y_pic50, cliff_mask_pic50, folds_pic50)
        results_pic50[("ChemProp", "Graph")] = res
        print(f"    MAE={res['MAE']:.3f}  R²={res['R2']:.3f}  "
              f"MAE_cliff={res['MAE_cliff']:.3f}")

        print("  ── ChemProp + CheMeleon ──")
        res = cv_chemprop(smiles_pic50, y_pic50, cliff_mask_pic50, folds_pic50,
                          use_chemeleon=True)
        results_pic50[("ChemProp+CheMeleon", "Graph")] = res
        print(f"    MAE={res['MAE']:.3f}  R²={res['R2']:.3f}  "
              f"MAE_cliff={res['MAE_cliff']:.3f}")

    # ── CV — pKi ──────────────────────────────────────────────────────────────
    print("\n[5] CV — pKi")
    results_pki = {}

    for feat_label, X in [("Desc", X_desc_pki), ("FP", X_fp_pki),
                           ("FCFP", X_fcfp_pki)]:
        print(f"\n  ── {feat_label} ──")
        model_factory_kwargs = {}
        if hyperopt:
            model_factory_kwargs = dict(hyperopt=True,
                                        study_name=f"pki_{feat_label}",
                                        X_tr=X, y_tr=y_pki, folds=folds_pki)
        for name, pipe in tqdm(make_sklearn_models(**model_factory_kwargs).items(),
                               desc=f"sklearn/{feat_label}"):
            res = cv_evaluate(pipe, X, y_pki, cliff_mask_pki, folds_pki)
            results_pki[(name, feat_label)] = res
            print(f"    {name:14s}  MAE={res['MAE']:.3f}  R²={res['R2']:.3f}  "
                  f"MAE_cliff={res['MAE_cliff']:.3f}")

        if HAS_TABPFN and feat_label in {"FP", "FCFP"}:
            pipe_t = make_tabpfn_pipeline(device="cuda")
            res = cv_evaluate(pipe_t, X, y_pki, cliff_mask_pki, folds_pki)
            results_pki[("TabPFN", feat_label)] = res
            print(f"    {'TabPFN':14s}  MAE={res['MAE']:.3f}  R²={res['R2']:.3f}")

    if HAS_CHEMPROP and not skip_graph:
        print("\n  ── ChemProp (graph only) ──")
        res = cv_chemprop(smiles_pki, y_pki, cliff_mask_pki, folds_pki)
        results_pki[("ChemProp", "Graph")] = res
        print(f"    MAE={res['MAE']:.3f}  R²={res['R2']:.3f}  "
              f"MAE_cliff={res['MAE_cliff']:.3f}")

        print("  ── ChemProp + CheMeleon ──")
        res = cv_chemprop(smiles_pki, y_pki, cliff_mask_pki, folds_pki,
                          use_chemeleon=True)
        results_pki[("ChemProp+CheMeleon", "Graph")] = res
        print(f"    MAE={res['MAE']:.3f}  R²={res['R2']:.3f}  "
              f"MAE_cliff={res['MAE_cliff']:.3f}")

    # ── Summary tables ────────────────────────────────────────────────────────
    tab_pic50 = results_to_df(results_pic50)
    tab_pki   = results_to_df(results_pki)
    tab_pic50.to_csv(os.path.join(OUT_DIR, "results_pic50.csv"), index=False)
    tab_pki.to_csv(  os.path.join(OUT_DIR, "results_pki.csv"),   index=False)
    print("\n=== pIC50 CV results ===")
    print(tab_pic50.round(3).to_string(index=False))
    print("\n=== pKi CV results ===")
    print(tab_pki.round(3).to_string(index=False))

    # Δ(FP − Desc) and Δ(FCFP − Desc)
    d_desc = tab_pic50[tab_pic50["Features"] == "Desc"].set_index("Model")
    d_fp   = tab_pic50[tab_pic50["Features"] == "FP"].set_index("Model")
    d_fcfp = tab_pic50[tab_pic50["Features"] == "FCFP"].set_index("Model")
    both_fp   = d_desc.index.intersection(d_fp.index)
    both_fcfp = d_desc.index.intersection(d_fcfp.index)
    delta  = pd.DataFrame({
        "ΔMAE (FP−Desc)":   d_fp.loc[both_fp,     "MAE"] - d_desc.loc[both_fp,   "MAE"],
        "ΔR²  (FP−Desc)":   d_fp.loc[both_fp,     "R2"]  - d_desc.loc[both_fp,   "R2"],
        "ΔMAE (FCFP−Desc)": d_fcfp.loc[both_fcfp, "MAE"] - d_desc.loc[both_fcfp, "MAE"],
        "ΔR²  (FCFP−Desc)": d_fcfp.loc[both_fcfp, "R2"]  - d_desc.loc[both_fcfp, "R2"],
    }).round(3)
    print("\nΔ metrics vs Desc (negative ΔMAE = FP/FCFP is better):")
    print(delta.sort_values("ΔMAE (FP−Desc)").to_string())

    # ── Cliff-pair directionality ─────────────────────────────────────────────
    print("\n[4b] Cliff-pair directionality — pIC50")
    sali_pairs_pic50 = compute_sali_pairs(SALI_mat_pic50, SALI_THRESHOLD)
    print(f"  Total cliff pairs (SALI > {SALI_THRESHOLD}): {len(sali_pairs_pic50)}")

    dir_tab_pic50, pair_store_pic50 = compute_directionality_table(
        X_dict={"Desc": X_desc_pic50, "FP": X_fp_pic50, "FCFP": X_fcfp_pic50},
        y=y_pic50,
        sali_pairs=sali_pairs_pic50,
        cliff_mask=cliff_mask_pic50,
        folds=folds_pic50,
    )
    dir_tab_pic50.to_csv(
        os.path.join(OUT_DIR, "cliff_directionality_pic50.csv"), index=False)
    print("\n=== Cliff-pair directionality (pIC50) ===")
    print(dir_tab_pic50.round(2).to_string(index=False))

    # pKi directionality (needed for cross-endpoint analysis)
    print("\n[4c] Cliff-pair directionality — pKi")
    SALI_mat_pki     = compute_sali(df_pki["smiles"].tolist(), df_pki["y"].values)
    sali_pairs_pki   = compute_sali_pairs(SALI_mat_pki, SALI_THRESHOLD)
    print(f"  Total cliff pairs (SALI > {SALI_THRESHOLD}): {len(sali_pairs_pki)}")
    dir_tab_pki, pair_store_pki = compute_directionality_table(
        X_dict={"Desc": X_desc_pki, "FP": X_fp_pki, "FCFP": X_fcfp_pki},
        y=y_pki,
        sali_pairs=sali_pairs_pki,
        cliff_mask=cliff_mask_pki,
        folds=folds_pki,
    )
    dir_tab_pki.to_csv(
        os.path.join(OUT_DIR, "cliff_directionality_pki.csv"), index=False)

    # ── Leave-cliff-out CV ────────────────────────────────────────────────────
    print("\n[4d] Leave-cliff-out CV — pIC50")
    lco_tab_pic50, _lco_pred_store = run_leave_cliff_out_cv(
        X_dict={"Desc": X_desc_pic50, "FP": X_fp_pic50, "FCFP": X_fcfp_pic50},
        y=y_pic50,
        cliff_mask=cliff_mask_pic50,
        folds=folds_pic50,
    )
    lco_tab_pic50.to_csv(
        os.path.join(OUT_DIR, "leave_cliff_out_pic50.csv"), index=False)

    # ── External predictions ──────────────────────────────────────────────────
    print("\n[6] External set predictions")
    ext_preds     = {}
    fitted_models = {}   # (feat_label, model_name) → (pipe, X_train, X_ext)

    for feat_label, X_train, X_ext_ in [
            ("Desc", X_desc_pic50, X_desc_ext),
            ("FP",   X_fp_pic50,   X_fp_ext),
            ("FCFP", X_fcfp_pic50, X_fcfp_ext)]:
        for name, pipe in make_sklearn_models().items():
            pipe.fit(X_train, y_pic50)
            ext_preds[(name, feat_label)] = pipe.predict(X_ext_)
            fitted_models[(feat_label, name)] = (pipe, X_train, X_ext_)
        if HAS_TABPFN and feat_label in {"FP", "FCFP"}:
            pipe_t = make_tabpfn_pipeline(device="cuda")
            pipe_t.fit(X_train, y_pic50)
            ext_preds[("TabPFN", feat_label)] = pipe_t.predict(X_ext_)
            fitted_models[(feat_label, "TabPFN")] = (pipe_t, X_train, X_ext_)

    if HAS_CHEMPROP and not skip_graph:
        mp, trainer, scaler = train_chemprop(smiles_pic50, y_pic50)
        ext_preds[("ChemProp", "Graph")] = predict_chemprop(
            mp, trainer, scaler, EXT_SMILES)

        mp2, trainer2, scaler2 = train_chemprop(smiles_pic50, y_pic50,
                                                 use_chemeleon=True)
        ext_preds[("ChemProp+CheMeleon", "Graph")] = predict_chemprop(
            mp2, trainer2, scaler2, EXT_SMILES)

    ext_corr = ext_metrics(ext_preds, y_ext)
    ext_corr.to_csv(os.path.join(OUT_DIR, "external_results.csv"), index=False)
    print(ext_corr.round(3).to_string(index=False))

    prospective = prospective_pair_table(
        ext_preds=ext_preds,
        X_dict={"Desc": X_desc_pic50, "FP": X_fp_pic50, "FCFP": X_fcfp_pic50},
        X_ext_dict={"Desc": X_desc_ext, "FP": X_fp_ext, "FCFP": X_fcfp_ext},
        y=y_pic50,
        directionality=dir_tab_pic50,
    )
    prospective.to_csv(
        os.path.join(OUT_DIR, "prospective_pair_deltas_thr55.csv"), index=False)

    # ── Ridge interpretation ──────────────────────────────────────────────────
    print("\n[7] Ridge interpretation")
    ridge_desc_pipe = Pipeline([("scl", StandardScaler(with_mean=False)),
                                 ("mdl", Ridge(alpha=1.0))])
    ridge_desc_pipe.fit(X_desc_pic50, y_pic50)
    coef_desc_df = (pd.DataFrame({"feature": _desc_names,
                                   "coef": ridge_desc_pipe.named_steps["mdl"].coef_})
                    .assign(abs_coef=lambda d: d["coef"].abs())
                    .sort_values("abs_coef", ascending=False)
                    .reset_index(drop=True))
    print(f"  Top Ridge (Desc) feature: {coef_desc_df.iloc[0]['feature']}  "
          f"coef={coef_desc_df.iloc[0]['coef']:+.4f}")

    ridge_fp_pipe = Pipeline([("scl", StandardScaler(with_mean=False)),
                               ("mdl", Ridge(alpha=1.0))])
    ridge_fp_pipe.fit(X_fp_pic50, y_pic50)
    coef_fp_df = (pd.DataFrame({"bit_idx": np.arange(FP_SIZE),
                                 "coef": ridge_fp_pipe.named_steps["mdl"].coef_})
                  .assign(abs_coef=lambda d: d["coef"].abs())
                  .sort_values("abs_coef", ascending=False)
                  .reset_index(drop=True))
    print(f"  Top Ridge (FP)   bit:     {int(coef_fp_df.iloc[0]['bit_idx'])}  "
          f"coef={coef_fp_df.iloc[0]['coef']:+.4f}")

    ridge_fcfp_pipe = Pipeline([("scl", StandardScaler(with_mean=False)),
                                 ("mdl", Ridge(alpha=1.0))])
    ridge_fcfp_pipe.fit(X_fcfp_pic50, y_pic50)
    coef_fcfp_df = (pd.DataFrame({"bit_idx": np.arange(FP_SIZE),
                                   "coef": ridge_fcfp_pipe.named_steps["mdl"].coef_})
                    .assign(abs_coef=lambda d: d["coef"].abs())
                    .sort_values("abs_coef", ascending=False)
                    .reset_index(drop=True))
    print(f"  Top Ridge (FCFP) bit:     {int(coef_fcfp_df.iloc[0]['bit_idx'])}  "
          f"coef={coef_fcfp_df.iloc[0]['coef']:+.4f}")

    # ── SHAP explanation of external predictions ──────────────────────────────
    ridge_pipes = {
        "Desc": ridge_desc_pipe,
        "FP":   ridge_fp_pipe,
        "FCFP": ridge_fcfp_pipe,
    }
    shap_out = os.path.join(OUT_DIR, "shap")
    os.makedirs(shap_out, exist_ok=True)
    run_shap_analysis(
        fitted_models=fitted_models,
        ext_labels=EXT_LABELS,
        ridge_pipes=ridge_pipes,
        out_dir=shap_out,
    )

    # Cache Morgan bitInfo for all training mols
    all_mols_pic50 = [Chem.MolFromSmiles(s) for s in df_pic50["smiles"]]
    bit_info_cache = []
    for mol in all_mols_pic50:
        bi = {}
        if mol is not None:
            AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_SIZE, bitInfo=bi)
        bit_info_cache.append(bi)

    # ── Optional: 5×5 repeated CV + error bars + Tukey HSD ───────────────────
    # Run before figures so df5x5 is available for error-bar plots
    df5x5        = None
    rep_dir_store = None   # populated if cv5x5 enabled
    if cv5x5:
        print("\n[7b] 5×5 repeated CV — pIC50")
        df5x5 = run_cv5x5(
            results_pic50,
            {"Desc": X_desc_pic50, "FP": X_fp_pic50, "FCFP": X_fcfp_pic50},
            y_pic50, cliff_mask_pic50, folds_pic50,
            n_splits=N_FOLDS, n_repeats=N_REPEATS,
        )
        df5x5_save = df5x5.copy()
        df5x5_save["MAE_repeats"] = df5x5_save["MAE_repeats"].apply(str)
        df5x5_save.to_csv(os.path.join(OUT_DIR, "cv5x5_pic50.csv"), index=False)

        plot_cv5x5_errorbars(df5x5, "cv5x5_errorbars_pic50.png")

        print("\n  Tukey HSD pairwise comparisons …")
        tukey_df = compute_tukey_hsd(df5x5)
        if not tukey_df.empty:
            tukey_df.to_csv(os.path.join(OUT_DIR, "tukey_hsd_pic50.csv"), index=False)
            print(f"  Tukey HSD saved → {os.path.join(OUT_DIR, 'tukey_hsd_pic50.csv')}")
            sig = tukey_df[tukey_df["reject"].astype(str).str.lower() == "true"]
            print(f"  Significantly different pairs (α=0.05): {len(sig)}")

        # Repeated-CV directionality uncertainty
        print("\n[7c] Repeated-CV directionality uncertainty — pIC50")
        rep_dir_store = run_repeated_directionality(
            X_dict={"Desc": X_desc_pic50, "FP": X_fp_pic50, "FCFP": X_fcfp_pic50},
            y=y_pic50,
            sali_pairs=sali_pairs_pic50,
            cliff_mask=cliff_mask_pic50,
            n_splits=N_FOLDS, n_repeats=N_REPEATS,
        )

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[8] Generating figures")

    plot_cv_heatmap(tab_pic50, "5-fold CV — pIC50 (Desc vs FP vs Graph)",
                    "cv_heatmap_pic50.png")
    plot_cv_heatmap(tab_pki,   "5-fold CV — pKi   (Desc vs FP vs Graph)",
                    "cv_heatmap_pki.png")
    plot_desc_vs_fp(tab_pic50, "desc_vs_fp_mae.png", df5x5=df5x5)
    plot_external(ext_preds, y_ext, EXT_LABELS, "external_predictions.png")
    plot_external_pearson_bar(ext_corr, "external_pearson_bar.png")
    plot_metric_correlation(tab_pic50, ext_corr, "Desc",
                            "metric_correlation_desc.png")
    plot_metric_correlation(tab_pic50, ext_corr, "FP",
                            "metric_correlation_fp.png")
    plot_metric_correlation(tab_pic50, ext_corr, "FCFP",
                            "metric_correlation_fcfp.png")
    plot_sali_landscape(df_pic50["smiles"].tolist(), y_pic50,
                        cliff_mask_pic50, SALI_mat_pic50,
                        "sali_landscape.png")
    plot_cv_vs_ext_spearman(tab_pic50, ext_corr,
                             "cv_vs_external_spearman.png", df5x5=df5x5)
    plot_mega_spearman(tab_pic50, ext_corr,
                       "mega_cv_r2_vs_spearman.png", df5x5=df5x5)
    plot_mega_pearson(tab_pic50, ext_corr,
                      "mega_cv_r2_vs_pearson.png", df5x5=df5x5)
    plot_cliff_mae(tab_pic50, "cliff_mae_comparison.png", df5x5=df5x5)
    plot_cliff_directionality_bars(dir_tab_pic50, "cliff_directionality_bars.png")
    plot_mega_directionality_vs_mae(dir_tab_pic50, "mega_directionality_vs_mae.png")
    plot_sali_pair_heatmap(dir_tab_pic50, pair_store_pic50,
                           "cliff_pair_heatmap.png", top_n_pairs=50)
    plot_directionality_vs_sali(pair_store_pic50, "directionality_vs_sali.png")
    consistency_df = compute_cross_endpoint_consistency(
        pair_store_pic50, pair_store_pki,
        smiles_pic50.tolist(), smiles_pki.tolist())
    if not consistency_df.empty:
        consistency_df.to_csv(
            os.path.join(OUT_DIR, "cross_endpoint_consistency.csv"), index=False)
        plot_cross_endpoint_consistency(
            consistency_df, "cross_endpoint_consistency.png")
    plot_leave_cliff_out(lco_tab_pic50, dir_tab_pic50, "leave_cliff_out_cv.png")
    if rep_dir_store is not None:
        plot_repeated_directionality_uncertainty(
            rep_dir_store, "repeated_dir_uncertainty.png")
        plot_repeated_directionality_summary(
            rep_dir_store, "repeated_dir_summary.png")
    plot_ridge_fp_bars(coef_fp_df,   "ridge_fp_coefficients.png")
    plot_ridge_fp_bars(coef_fcfp_df, "ridge_fcfp_coefficients.png")
    plot_ridge_fp_bits(coef_fp_df, all_mols_pic50, bit_info_cache,
                       "ridge_fp_morgan_bits.png")

    # ── Similarity Maps (ECFP and FCFP) ──────────────────────────────────────
    print("  ECFP similarity maps …")
    plot_ridge_similarity_maps(ridge_fp_pipe,   EXT_SMILES, EXT_LABELS)
    print("  FCFP similarity maps …")
    plot_ridge_similarity_maps(ridge_fcfp_pipe, EXT_SMILES, EXT_LABELS,
                               fname_prefix="ridge_fcfp_sim_map", use_fcfp=True)

    # ── Ridge FP/FCFP heatmaps by environment radius ──────────────────────────
    print("\n  Ridge ECFP heatmaps by environment radius (r=0,1,2 + all)")
    plot_ridge_fp_heatmaps_by_radius(
        df_pic50["smiles"].tolist(), y_pic50,
        all_mols_pic50, bit_info_cache, use_fcfp=False)

    print("\n  Ridge FCFP heatmaps by environment radius (r=0,1,2 + all)")
    plot_ridge_fp_heatmaps_by_radius(
        df_pic50["smiles"].tolist(), y_pic50,
        all_mols_pic50, bit_info_cache, use_fcfp=True)

    # ── Optional: SALI threshold sweep ────────────────────────────────────────
    if sali_sweep:
        print("\n[9] SALI threshold sweep (this takes a few minutes)")
        plot_sali_threshold_sweep(
            df_pic50["smiles"].tolist(), y_pic50,
            cv_evaluate, folds_pic50,
            X_desc_pic50, X_fp_pic50,
            "sali_threshold_sweep.png",
            thresholds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30],
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("KEY FINDINGS")
    print("═" * 65)
    for feat_lbl in ["Desc", "FP", "Graph"]:
        sub = tab_pic50[tab_pic50["Features"] == feat_lbl].sort_values("MAE")
        if sub.empty:
            continue
        best = sub.iloc[0]
        print(f"\nBest pIC50 CV [{feat_lbl}]: {best['Model']}  "
              f"MAE={best['MAE']:.3f}  R²={best['R2']:.3f}  "
              f"MAE_cliff={best['MAE_cliff']:.3f}")

    print("\nBest external Pearson per feature space:")
    for feat_lbl in ["Desc", "FP", "Graph"]:
        sub = ext_corr[ext_corr["Features"] == feat_lbl].sort_values(
            "Pearson_r", ascending=False)
        if sub.empty:
            continue
        b = sub.iloc[0]
        print(f"  [{feat_lbl:10s}] {b['Model']:22s}  "
              f"r={b['Pearson_r']:.3f}  ρ={b['Spearman_rho']:.3f}  "
              f"MAE_ext={b['MAE_ext']:.3f}")

    print("\nBest cliff-pair directionality per feature space:")
    for feat_lbl in ["Desc", "FP", "FCFP"]:
        sub = dir_tab_pic50[dir_tab_pic50["Features"] == feat_lbl].sort_values(
            "pct_correct", ascending=False)
        if sub.empty:
            continue
        b = sub.iloc[0]
        print(f"  [{feat_lbl:10s}] {b['Model']:22s}  "
              f"pct_correct={b['pct_correct']:.1f}%  n_pairs={int(b['n_pairs'])}")

    print("\nLeave-cliff-out generalisation gap (pIC50):")
    for feat_lbl in ["Desc", "FP", "FCFP"]:
        sub = lco_tab_pic50[lco_tab_pic50["Features"] == feat_lbl].sort_values("MAE_lco")
        dir_sub = dir_tab_pic50[dir_tab_pic50["Features"] == feat_lbl].set_index("Model")
        if sub.empty:
            continue
        b = sub.iloc[0]
        std_cv = dir_sub["MAE_cliff"].get(b["Model"], np.nan)
        gap = b["MAE_lco"] - std_cv if np.isfinite(std_cv) else np.nan
        print(f"  [{feat_lbl:10s}] {b['Model']:22s}  "
              f"MAE_lco={b['MAE_lco']:.3f}  "
              f"MAE_cliff(CV)={std_cv:.3f}  gap={gap:+.3f}")

    print("\nAll outputs written to ./results/")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QSAR analysis of NMDA antagonists (Schultz et al.)")
    parser.add_argument("--hyperopt",   action="store_true",
                        help="Run Optuna hyperparameter optimisation (requires: pip install optuna)")
    parser.add_argument("--sali_sweep", action="store_true",
                        help="Run SALI threshold sensitivity sweep and plot")
    parser.add_argument("--figures_only", action="store_true",
                        help="Regenerate all figures from existing CSVs in results/ "
                             "(no model training needed)")
    parser.add_argument("--cv5x5", action="store_true",
                        help="Run 5×5 repeated CV with error bars and Tukey HSD "
                             "(slow — ~25× more model fits)")
    parser.add_argument("--shap_only", action="store_true",
                        help="Refit only the pIC50 sklearn models and "
                             "regenerate the SHAP / interpretation figures "
                             "(no cross-validation, no pKi endpoint)")
    parser.add_argument("--out_dir", default=None,
                        help=f"Read inputs from / write outputs to this "
                        f"directory instead of ./{OUT_DIR}/")
    parser.add_argument("--skip_graph", action="store_true",
                        help="Skip the legacy single-run graph branch; use "
                             "rerun_seeded_graph_models.py for graph models")
    args = parser.parse_args()

    if args.out_dir:
        OUT_DIR = args.out_dir
        os.makedirs(OUT_DIR, exist_ok=True)

    if args.shap_only:
        run_shap_only()
    elif args.figures_only:
        run_figures_only()
    else:
        main(hyperopt=args.hyperopt, sali_sweep=args.sali_sweep,
             cv5x5=args.cv5x5, skip_graph=args.skip_graph)
