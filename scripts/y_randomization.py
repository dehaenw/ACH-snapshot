#!/usr/bin/env python3
"""Deterministic Y-randomization control for representative pIC50 models.

The molecular features, five fixed folds, preprocessing, model settings, and
evaluation metrics are identical to the primary workflow.  For each null
repeat, the pIC50 labels are permuted across records before any model fitting.
No hyperparameter selection is repeated on the permuted labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import qsar_nmda as qn  # noqa: E402


SELECTED_CONFIGURATIONS = (
    ("XGB", "Desc", "descriptor representative"),
    ("RF", "FP", "highest overall fixed-fold pIC50 R2"),
    ("SVM", "FP", "highest cliff-subset pIC50 R2"),
    ("XGB", "FCFP", "highest FCFP cliff-subset pIC50 R2"),
    ("TabPFN", "FCFP", "tabular foundation-model representative"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_DIR / "data" / "pic50_clean.csv")
    parser.add_argument("--out-dir", type=Path, default=REPO_DIR / "results" / "y_randomization")
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--tabpfn-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--skip-tabpfn",
        action="store_true",
        help="Run only the four conventional configurations.",
    )
    return parser.parse_args()


def evaluate_oof(model_factory, X: np.ndarray, y: np.ndarray, folds) -> tuple[float, float]:
    prediction = np.full(len(y), np.nan, dtype=float)
    # Re-fitting the same estimator object is equivalent to constructing a new
    # object for these sklearn/TabPFN pipelines, and avoids reloading the large
    # TabPFN checkpoint five times per permutation.
    model = model_factory()
    for train_index, validation_index in folds:
        model.fit(X[train_index], y[train_index])
        prediction[validation_index] = np.asarray(model.predict(X[validation_index])).reshape(-1)
    del model
    if np.isnan(prediction).any():
        raise RuntimeError("Out-of-fold prediction vector is incomplete")
    return mean_absolute_error(y, prediction), r2_score(y, prediction)


def main() -> None:
    args = parse_args()
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1")

    # Avoid writing plotting caches into the user's home directory.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/nmda-y-randomization-mpl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    data, X_desc, X_fp, X_fcfp = qn.prep_dataset(raw)
    y = data["y"].to_numpy(dtype=float)
    X_by_feature = {"Desc": X_desc, "FP": X_fp, "FCFP": X_fcfp}
    folds = qn.make_folds(len(y))

    templates = qn.make_sklearn_models()
    configurations = list(SELECTED_CONFIGURATIONS)
    if args.skip_tabpfn:
        configurations = [item for item in configurations if item[0] != "TabPFN"]

    def factory(model_name: str):
        if model_name == "TabPFN":
            return lambda: qn.make_tabpfn_pipeline(device=args.tabpfn_device)
        return lambda: clone(templates[model_name])

    # Every model receives the same deterministic sequence of permuted labels.
    rng = np.random.default_rng(args.seed)
    permuted_targets = [rng.permutation(y) for _ in range(args.n_permutations)]

    published_path = REPO_DIR / "results" / "results_pic50.csv"
    published = pd.read_csv(published_path) if published_path.exists() else pd.DataFrame()
    long_rows: list[dict] = []
    summary_rows: list[dict] = []

    for model_name, feature_name, selection_reason in configurations:
        X = X_by_feature[feature_name]
        make_model = factory(model_name)
        observed_mae, observed_r2 = evaluate_oof(make_model, X, y, folds)
        print(
            f"{model_name}/{feature_name}: observed MAE={observed_mae:.4f}, "
            f"R2={observed_r2:.4f}"
        )

        null_mae = np.empty(args.n_permutations, dtype=float)
        null_r2 = np.empty(args.n_permutations, dtype=float)
        for repeat, permuted_y in enumerate(permuted_targets, start=1):
            mae, r2 = evaluate_oof(make_model, X, permuted_y, folds)
            null_mae[repeat - 1] = mae
            null_r2[repeat - 1] = r2
            long_rows.append(
                {
                    "Model": model_name,
                    "Features": feature_name,
                    "Permutation": repeat,
                    "Permutation_seed": args.seed,
                    "MAE": mae,
                    "R2": r2,
                }
            )
            if repeat == 1 or repeat % 10 == 0 or repeat == args.n_permutations:
                print(f"  permutation {repeat:3d}/{args.n_permutations}: R2={r2:.4f}")

        match = published[
            (published.get("Model") == model_name) &
            (published.get("Features") == feature_name)
        ] if not published.empty else pd.DataFrame()
        published_r2 = float(match.iloc[0]["R2"]) if len(match) == 1 else np.nan

        # Phipson-Smyth finite-simulation correction: the empirical p value can
        # never be exactly zero with a finite null sample.
        empirical_p = (1 + int(np.sum(null_r2 >= observed_r2))) / (args.n_permutations + 1)
        summary_rows.append(
            {
                "Model": model_name,
                "Features": feature_name,
                "Selection_reason": selection_reason,
                "N_records": len(y),
                "N_folds": len(folds),
                "N_permutations": args.n_permutations,
                "Observed_MAE": observed_mae,
                "Observed_R2": observed_r2,
                "Published_R2": published_r2,
                "Null_MAE_mean": float(np.mean(null_mae)),
                "Null_MAE_sd": float(np.std(null_mae, ddof=1)),
                "Null_R2_mean": float(np.mean(null_r2)),
                "Null_R2_sd": float(np.std(null_r2, ddof=1)),
                "Null_R2_95th_percentile": float(np.quantile(null_r2, 0.95)),
                "Null_R2_max": float(np.max(null_r2)),
                "Empirical_p_R2": empirical_p,
                "Null_repeats_at_or_above_observed_R2": int(np.sum(null_r2 >= observed_r2)),
            }
        )

    long_table = pd.DataFrame(long_rows)
    summary_table = pd.DataFrame(summary_rows)
    long_table.to_csv(args.out_dir / "y_randomization_permutations.csv", index=False)
    summary_table.to_csv(args.out_dir / "y_randomization_summary.csv", index=False)

    metadata = {
        "endpoint": "pIC50",
        "input": str(args.input),
        "n_records": int(len(y)),
        "folds": "KFold(n_splits=5, shuffle=True, random_state=42), unchanged from primary workflow",
        "permutation_seed": args.seed,
        "n_permutations": args.n_permutations,
        "procedure": (
            "The response vector was permuted across records before fitting. "
            "Features, folds, preprocessing and fixed model settings were unchanged; "
            "models were scored against the corresponding permuted validation labels."
        ),
        "empirical_p_definition": "(1 + count(null R2 >= observed R2)) / (1 + n_permutations)",
        "configurations": [
            {"model": model, "features": features, "selection_reason": reason}
            for model, features, reason in configurations
        ],
    }
    (args.out_dir / "y_randomization_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\nSummary")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
