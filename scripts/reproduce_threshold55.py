"""Recompute threshold-dependent QSAR outputs at SALI > 5.5.

This script deliberately reuses the study pipeline's data preparation,
featurisation, model factory, CV folds, and SALI implementation.  The
threshold affects cliff reporting, directionality, cross-endpoint comparison,
and leave-cliff-out partitioning; it does not change ordinary CV model fits.
"""
from pathlib import Path
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
OUT = PROJECT / "results"
THRESHOLD = 5.5
sys.path.insert(0, str(SCRIPTS))
os.chdir(PROJECT)

import numpy as np
import pandas as pd
import qsar_nmda as Q


def evaluate(tag, feature_sets, y, cliff_mask, folds):
    rows = []
    for features, X in feature_sets.items():
        for model, pipeline in Q.make_sklearn_models().items():
            metrics = Q.cv_evaluate(pipeline, X, y, cliff_mask, folds)
            rows.append({"Model": model, "Features": features, **metrics})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"results_{tag}.csv", index=False)
    return table


def main():
    OUT.mkdir(exist_ok=True)
    pki_input = pd.read_csv(PROJECT / "data" / "pki_clean.csv")
    pic_input = pd.read_csv(PROJECT / "data" / "pic50_clean.csv")
    pki, pki_desc, pki_fp, pki_fcfp = Q.prep_dataset(pki_input)
    pic, pic_desc, pic_fp, pic_fcfp = Q.prep_dataset(pic_input)
    y_pki, y_pic = pki["y"].values, pic["y"].values
    sali_pki = Q.compute_sali(pki["smiles"].tolist(), y_pki)
    sali_pic = Q.compute_sali(pic["smiles"].tolist(), y_pic)
    mask_pki = (sali_pki > THRESHOLD).any(axis=1)
    mask_pic = (sali_pic > THRESHOLD).any(axis=1)
    pairs_pki = Q.compute_sali_pairs(sali_pki, THRESHOLD)
    pairs_pic = Q.compute_sali_pairs(sali_pic, THRESHOLD)
    folds_pki, folds_pic = Q.make_folds(len(pki)), Q.make_folds(len(pic))
    sets_pki = {"Desc": pki_desc, "FP": pki_fp, "FCFP": pki_fcfp}
    sets_pic = {"Desc": pic_desc, "FP": pic_fp, "FCFP": pic_fcfp}

    evaluate("pki", sets_pki, y_pki, mask_pki, folds_pki)
    evaluate("pic50", sets_pic, y_pic, mask_pic, folds_pic)
    dir_pic, store_pic = Q.compute_directionality_table(
        sets_pic, y_pic, pairs_pic, mask_pic, folds_pic)
    dir_pki, store_pki = Q.compute_directionality_table(
        sets_pki, y_pki, pairs_pki, mask_pki, folds_pki)
    dir_pic.to_csv(OUT / "cliff_directionality_pic50.csv", index=False)
    dir_pki.to_csv(OUT / "cliff_directionality_pki.csv", index=False)
    consistency = Q.compute_cross_endpoint_consistency(
        store_pic, store_pki, pic["smiles"].values, pki["smiles"].values)
    consistency.to_csv(OUT / "cross_endpoint_consistency.csv", index=False)
    lco, _ = Q.run_leave_cliff_out_cv(sets_pic, y_pic, mask_pic, folds_pic)
    lco.to_csv(OUT / "leave_cliff_out_pic50.csv", index=False)
    summary = {
        "sali_threshold": THRESHOLD,
        "pIC50": {"n_compounds": len(pic), "n_cliff_compounds": int(mask_pic.sum()),
                  "n_cliff_pairs": len(pairs_pic)},
        "pKi": {"n_compounds": len(pki), "n_cliff_compounds": int(mask_pki.sum()),
                "n_cliff_pairs": len(pairs_pki)},
    }
    (OUT / "summary_threshold55.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
