#!/usr/bin/env python3
"""Fail-fast integrity checks for the completed seeded graph-model rerun."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
RESULTS = ROOT / "results"
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["chemprop", "chemeleon"]
ENDPOINT_ROWS = {"pic50": 332, "pki": 511}


def main() -> None:
    checked = 0
    for endpoint, n_rows in ENDPOINT_ROWS.items():
        for model in MODELS:
            for seed in SEEDS:
                stem = f"{endpoint}_{model}_seed{seed}"
                metric_path = RESULTS / f"metrics_{stem}.json"
                pred_path = RESULTS / f"oof_{stem}.csv"
                direction_path = RESULTS / f"directionality_{stem}.csv"
                assert metric_path.exists(), metric_path
                assert pred_path.exists(), pred_path
                assert direction_path.exists(), direction_path
                payload = json.loads(metric_path.read_text())
                pred = pd.read_csv(pred_path)
                direction = pd.read_csv(direction_path)
                assert len(pred) == n_rows
                assert pred["y_pred"].notna().all()
                assert np.isfinite(pred["y_pred"]).all()
                assert payload["Repeat_seed"] == seed
                assert payload["n_folds"] == 5
                assert len(payload["folds"]) == 5
                assert payload["SALI_threshold"] == 5.5
                assert len(direction) == payload["n_cliff_pairs"]
                audit = payload["model_audit"]
                assert audit["all_parameters_trainable"] is True
                if model == "chemeleon":
                    assert audit["missing_keys"] == []
                    assert audit["unexpected_keys"] == []
                    assert audit["encoder_output_dim"] == 2048
                    assert audit["hyper_parameters"]["depth"] == 6
                else:
                    assert audit["encoder_output_dim"] == 300
                    assert audit["hyper_parameters"]["depth"] == 3
                if endpoint == "pic50":
                    ext = pd.read_csv(RESULTS / f"external_{stem}.csv")
                    assert len(ext) == 4
                    assert ext["predicted_pIC50"].notna().all()
                checked += 1

    summary = pd.read_csv(RESULTS / "graph_metrics_summary.csv")
    assert len(summary) == 4
    assert set(summary["n_repeats"]) == {5}
    metadata = json.loads((RESULTS / "graph_rerun_metadata.json").read_text())
    assert metadata["seeds"] == SEEDS
    assert metadata["n_folds"] == 5
    assert metadata["sali_threshold"] == 5.5
    print(f"PASS: {checked} endpoint/model/seed runs and aggregate metadata verified")


if __name__ == "__main__":
    main()
