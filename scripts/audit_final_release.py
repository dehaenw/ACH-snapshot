#!/usr/bin/env python3
"""Fail-fast audit for the final measured-IC50 analysis release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from external_data import EXPECTED_COMPOUNDS, EXT_LABELS, load_external_measurements


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

REQUIRED_ENVIRONMENT_PINS = (
    "python=3.12.12",
    "numpy==2.2.6",
    "pandas==2.3.3",
    "scipy==1.15.3",
    "scikit-learn==1.4.2",
    "rdkit==2025.9.3",
    "xgboost==3.2.0",
    "lightgbm==4.6.0",
    "shap==0.51.0",
    "umap-learn==0.5.7",
    "torch==2.9.1",
    "lightning==2.6.0",
    "chemprop==2.2.3",
    "tabpfn==2.1.0",
)

EXPECTED_PIC50 = np.array([
    6.2557368898022,
    4.68501598061177,
    6.10884566570212,
    4.94999831292921,
])

CORE_OUTPUTS = [
    "results_pic50.csv",
    "results_pki.csv",
    "cliff_directionality_pic50.csv",
    "cliff_directionality_pki.csv",
    "leave_cliff_out_pic50.csv",
    "external_results.csv",
    "prospective_pair_deltas_thr55.csv",
    "external_pair_statistics.csv",
    "sali_threshold_sweep.csv",
    "cross_endpoint_consistency.csv",
    "cv5x5_pic50.csv",
    "tukey_hsd_pic50.csv",
    "y_randomization/y_randomization_permutations.csv",
    "y_randomization/y_randomization_summary.csv",
    "y_randomization/y_randomization_metadata.json",
]

GRAPH_OUTPUTS = [
    "graph_metrics_per_repeat.csv",
    "graph_metrics_summary.csv",
    "graph_external_predictions_per_repeat.csv",
    "graph_external_predictions_summary.csv",
    "graph_external_metrics_per_repeat.csv",
    "graph_external_metrics_summary.csv",
    "graph_oof_predictions_per_repeat.csv",
    "graph_directionality_per_repeat.csv",
    "graph_rerun_metadata.csv",
    "graph_rerun_metadata.json",
]


def require_files(names: list[str]) -> None:
    missing = [name for name in names if not (RESULTS / name).is_file()]
    if missing:
        raise AssertionError(f"Missing required results: {missing}")


def audit_measurements() -> None:
    ext = load_external_measurements()
    assert ext["compound"].tolist() == EXPECTED_COMPOUNDS
    np.testing.assert_allclose(ext["mean_pIC50"], EXPECTED_PIC50, rtol=0, atol=5e-12)
    assert (ext["n"].to_numpy() == 6).all()

    stats = pd.read_csv(RESULTS / "external_pair_statistics.csv")
    primary = stats.loc[stats["analysis"] == "primary_all_six_cells"].set_index("pair")
    np.testing.assert_allclose(
        primary.loc[["N-ethyl", "N-propyl"], "delta_pIC50"],
        [EXPECTED_PIC50[0] - EXPECTED_PIC50[1], EXPECTED_PIC50[2] - EXPECTED_PIC50[3]],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        primary.loc[["N-ethyl", "N-propyl"], "ecfp4_sali"],
        [5.235736363968085, 4.152536347436277],
        rtol=0,
        atol=1e-12,
    )
    assert not primary["ecfp4_sali_gt_5_5"].astype(bool).any()


def audit_no_stale_literals() -> None:
    stale = ("6.67", "4.87", "4.97", "1.70", "1.80")
    allowed = {"audit_final_release.py"}
    hits: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for literal in stale:
            if literal in text:
                hits.append(f"{path.name}: {literal}")
    if hits:
        raise AssertionError(f"Possible stale apparent-IC50 literals remain: {hits}")


def audit_environment() -> None:
    path = ROOT / "environment.yml"
    if not path.is_file():
        raise AssertionError("Missing environment.yml")
    text = path.read_text(encoding="utf-8")
    if "name: nmda-qsar-final" not in text:
        raise AssertionError("Unexpected Conda environment name")
    missing = [pin for pin in REQUIRED_ENVIRONMENT_PINS if pin not in text]
    if missing:
        raise AssertionError(f"Missing required environment pins: {missing}")


def audit_graph() -> None:
    require_files(GRAPH_OUTPUTS)
    preds = pd.read_csv(RESULTS / "graph_external_predictions_per_repeat.csv")
    assert set(preds["compound"]) == set(EXT_LABELS)
    if "observed_pIC50" in preds:
        observed = (preds[["compound", "observed_pIC50"]]
                    .drop_duplicates().set_index("compound")
                    .loc[EXT_LABELS, "observed_pIC50"])
        np.testing.assert_allclose(observed, EXPECTED_PIC50, rtol=0, atol=5e-12)
    assert set(preds["Repeat_seed"]) == {42, 43, 44, 45, 46}
    assert set(preds["Model"]) == {"ChemProp", "ChemProp+CheMeleon"}

    metadata = json.loads((RESULTS / "graph_rerun_metadata.json").read_text())
    assert metadata["seeds"] == [42, 43, 44, 45, 46]


def audit_y_randomization() -> None:
    summary = pd.read_csv(RESULTS / "y_randomization" / "y_randomization_summary.csv")
    repeats = pd.read_csv(RESULTS / "y_randomization" / "y_randomization_permutations.csv")
    expected = {
        ("XGB", "Desc"),
        ("RF", "FP"),
        ("SVM", "FP"),
        ("XGB", "FCFP"),
        ("TabPFN", "FCFP"),
    }
    assert set(zip(summary["Model"], summary["Features"])) == expected
    assert (summary["N_records"] == 332).all()
    assert (summary["N_folds"] == 5).all()
    assert (summary["N_permutations"] == 100).all()
    assert len(repeats) == 500
    assert repeats.groupby(["Model", "Features"]).size().eq(100).all()

    published = pd.read_csv(RESULTS / "results_pic50.csv")
    for row in summary.itertuples(index=False):
        match = published[
            (published["Model"] == row.Model) &
            (published["Features"] == row.Features)
        ]
        assert len(match) == 1
        np.testing.assert_allclose(row.Observed_R2, match.iloc[0]["R2"], atol=1e-12)
        np.testing.assert_allclose(row.Published_R2, match.iloc[0]["R2"], atol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-graph", action="store_true")
    args = parser.parse_args()

    require_files(CORE_OUTPUTS)
    audit_environment()
    audit_measurements()
    audit_no_stale_literals()
    audit_y_randomization()
    if args.require_graph:
        audit_graph()
    print("Final release audit passed.")


if __name__ == "__main__":
    main()
