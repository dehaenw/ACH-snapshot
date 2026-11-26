"""Authoritative concentration-response measurements for the external set.

All analysis scripts import the four measured pIC50 values from one CSV so a
stale hard-coded apparent-IC50 value cannot survive in an individual figure or
model branch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = PROJECT / "data" / "external_concentration_response.csv"
INDIVIDUAL_PATH = PROJECT / "data" / "external_individual_fits.csv"
EXPECTED_COMPOUNDS = ["HC49", "HC22", "HC46", "HC52"]


def load_external_measurements() -> pd.DataFrame:
    frame = pd.read_csv(SUMMARY_PATH).sort_values("analysis_order").reset_index(drop=True)
    if frame["compound"].tolist() != EXPECTED_COMPOUNDS:
        raise ValueError("External compounds are missing or not in the required matched-pair order")
    required = {
        "smiles", "label", "n", "ic50_geo_uM", "ic50_ci_low_uM",
        "ic50_ci_high_uM", "mean_pIC50", "pIC50_sem",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing external measurement columns: {sorted(missing)}")
    derived = 6.0 - np.log10(frame["ic50_geo_uM"].to_numpy(float))
    if not np.allclose(derived, frame["mean_pIC50"].to_numpy(float), atol=5e-12):
        raise ValueError("External mean pIC50 values do not match the geometric-mean IC50 values")
    return frame


def load_external_individual_fits() -> pd.DataFrame:
    frame = pd.read_csv(INDIVIDUAL_PATH, keep_default_na=False)
    counts = frame.groupby("compound").size().reindex(EXPECTED_COMPOUNDS)
    if counts.isna().any() or not (counts == 6).all():
        raise ValueError("Expected six individual fits for each external compound")
    derived = 6.0 - np.log10(frame["individual_ic50_uM"].to_numpy(float))
    if not np.allclose(derived, frame["pIC50"].to_numpy(float), atol=6e-5):
        raise ValueError("Individual pIC50 values do not match the individual IC50 values")
    return frame


EXTERNAL = load_external_measurements()
EXT_COMPOUNDS = EXTERNAL["compound"].tolist()
EXT_SMILES = EXTERNAL["smiles"].tolist()
EXT_LABELS = EXTERNAL["label"].tolist()
EXT_PIC50 = EXTERNAL["mean_pIC50"].to_numpy(float)
EXT_PIC50_SEM = EXTERNAL["pIC50_sem"].to_numpy(float)
