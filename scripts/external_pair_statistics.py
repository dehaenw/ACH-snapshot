#!/usr/bin/env python3
"""Compute matched-pair fold changes, uncertainty, similarity and SALI.

The primary analysis uses all six independently fitted cells per compound.
Because the source workbook flags one HC49 trace as a bad fit while retaining
it in the reported mean, a sensitivity row also excludes only that trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from scipy.stats import t

from external_data import load_external_individual_fits, load_external_measurements


THRESHOLD = 5.5
FP_RADIUS = 2
FP_SIZE = 2048
PAIRS = [("N-ethyl", "HC49", "HC22"), ("N-propyl", "HC46", "HC52")]


def similarity(smiles_a: str, smiles_b: str, feature: bool = False) -> float:
    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if feature:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=FP_RADIUS,
            fpSize=FP_SIZE,
            atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
        )
        fp_a, fp_b = generator.GetFingerprint(mol_a), generator.GetFingerprint(mol_b)
    else:
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, FP_RADIUS, nBits=FP_SIZE)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, FP_RADIUS, nBits=FP_SIZE)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def welch_difference(meta: np.ndarray, para: np.ndarray) -> tuple[float, float, float, float, float]:
    delta = float(meta.mean() - para.mean())
    v_meta = float(meta.var(ddof=1) / len(meta))
    v_para = float(para.var(ddof=1) / len(para))
    sem = float(np.sqrt(v_meta + v_para))
    df = float((v_meta + v_para) ** 2 / (
        v_meta ** 2 / (len(meta) - 1) + v_para ** 2 / (len(para) - 1)
    ))
    critical = float(t.ppf(0.975, df))
    return delta, delta - critical * sem, delta + critical * sem, sem, df


def analyse_pair(
    analysis: str,
    pair_name: str,
    meta_name: str,
    para_name: str,
    summary: pd.DataFrame,
    individual: pd.DataFrame,
) -> dict[str, object]:
    indexed = summary.set_index("compound")
    meta = individual.loc[individual["compound"] == meta_name, "pIC50"].to_numpy(float)
    para = individual.loc[individual["compound"] == para_name, "pIC50"].to_numpy(float)
    delta, low, high, sem, df = welch_difference(meta, para)
    sim_ecfp = similarity(indexed.loc[meta_name, "smiles"], indexed.loc[para_name, "smiles"])
    sim_fcfp = similarity(indexed.loc[meta_name, "smiles"], indexed.loc[para_name, "smiles"], True)
    denom_ecfp, denom_fcfp = 1.0 - sim_ecfp, 1.0 - sim_fcfp
    return {
        "analysis": analysis,
        "pair": pair_name,
        "meta_compound": meta_name,
        "para_compound": para_name,
        "n_meta": len(meta),
        "n_para": len(para),
        "meta_mean_pIC50": float(meta.mean()),
        "para_mean_pIC50": float(para.mean()),
        "delta_pIC50": delta,
        "delta_pIC50_sem": sem,
        "delta_welch_df": df,
        "delta_pIC50_ci95_low": low,
        "delta_pIC50_ci95_high": high,
        "potency_ratio_meta_over_para": 10.0 ** delta,
        "potency_ratio_ci95_low": 10.0 ** low,
        "potency_ratio_ci95_high": 10.0 ** high,
        "ecfp4_tanimoto": sim_ecfp,
        "ecfp4_sali": delta / denom_ecfp,
        "ecfp4_sali_ci95_low": low / denom_ecfp,
        "ecfp4_sali_ci95_high": high / denom_ecfp,
        "ecfp4_sali_gt_5_5": bool(delta / denom_ecfp > THRESHOLD),
        "fcfp4_tanimoto": sim_fcfp,
        "fcfp4_sali": delta / denom_fcfp,
        "fcfp4_sali_ci95_low": low / denom_fcfp,
        "fcfp4_sali_ci95_high": high / denom_fcfp,
        "fcfp4_sali_gt_5_5": bool(delta / denom_fcfp > THRESHOLD),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_external_measurements()
    individual = load_external_individual_fits()
    rows = [analyse_pair("primary_all_six_cells", *pair, summary, individual) for pair in PAIRS]
    sensitivity = individual.loc[
        ~((individual["compound"] == "HC49")
          & individual["source_note"].str.contains("bad fit", case=False, na=False))
    ].copy()
    rows.append(analyse_pair("sensitivity_exclude_flagged_HC49", *PAIRS[0], summary, sensitivity))
    output = pd.DataFrame(rows)
    output.to_csv(args.out_dir / "external_pair_statistics.csv", index=False)

    audit = {
        "source_summary": "data/external_concentration_response.csv",
        "source_individual": "data/external_individual_fits.csv",
        "sali_threshold": THRESHOLD,
        "primary_rows": output.loc[output["analysis"] == "primary_all_six_cells"].to_dict("records"),
        "flagged_trace_sensitivity": output.loc[
            output["analysis"] == "sensitivity_exclude_flagged_HC49"
        ].to_dict("records"),
    }
    (args.out_dir / "external_pair_statistics.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
