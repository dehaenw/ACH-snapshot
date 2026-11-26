#!/usr/bin/env python3
"""Merge corrected seeded graph-model results with the frozen study outputs.

The original non-graph rows are retained byte-for-value from the released
tables.  Only the two graph-model rows are added or replaced.  Additional
uncertainty columns are included so downstream figures and manuscript tables
can distinguish classical bootstrap intervals from graph seed ranges.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STANDALONE_RELEASE = (SCRIPT_DIR.parent / "data").is_dir()
ROOT = SCRIPT_DIR.parent if STANDALONE_RELEASE else SCRIPT_DIR
GRAPH = ROOT / "results"
# The release updates its own frozen non-graph tables in place. During the
# manuscript-working rerun, retain the original release as the frozen source
# and write a separate assembled directory for review.
BASE = Path(os.environ.get(
    "NMDA_QSAR_BASE_RESULTS",
    ROOT / ("results" if STANDALONE_RELEASE else "base_results"),
))
OUT = (ROOT / "results") if STANDALONE_RELEASE else (ROOT / "updated_results")
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["ChemProp", "ChemProp+CheMeleon"]


def graph_summary() -> pd.DataFrame:
    return pd.read_csv(GRAPH / "graph_metrics_summary.csv")


def graph_repeats() -> pd.DataFrame:
    return pd.read_csv(GRAPH / "graph_metrics_per_repeat.csv")


def add_graph_metric_rows(endpoint: str, summary: pd.DataFrame) -> None:
    base = pd.read_csv(BASE / f"results_{endpoint}.csv")
    base = base[base["Features"] != "Graph"].copy()
    g = summary[summary["Endpoint"] == endpoint].copy()
    primary = ["Model", "Features", "MAE", "R2", "MAE_cliff", "R2_cliff"]
    uncertainty = []
    for metric in ["MAE", "R2", "MAE_cliff", "R2_cliff"]:
        uncertainty.extend([f"{metric}_sd", f"{metric}_min", f"{metric}_max"])
    for col in uncertainty:
        if col not in base:
            base[col] = np.nan
    merged = pd.concat([base, g[primary + uncertainty]], ignore_index=True)
    merged.to_csv(OUT / f"results_{endpoint}.csv", index=False)


def add_directionality_rows(endpoint: str, summary: pd.DataFrame) -> None:
    base = pd.read_csv(BASE / f"cliff_directionality_{endpoint}.csv")
    base = base[base["Features"] != "Graph"].copy()
    g = summary[summary["Endpoint"] == endpoint].copy()
    rows = pd.DataFrame({
        "Model": g["Model"],
        "Features": "Graph",
        "pct_correct": g["directionality_pct"],
        "n_pairs": 140 if endpoint == "pic50" else 261,
        "MAE": g["MAE"],
        "MAE_cliff": g["MAE_cliff"],
        "pct_correct_sd": g["directionality_pct_sd"],
        "pct_correct_min": g["directionality_pct_min"],
        "pct_correct_max": g["directionality_pct_max"],
    })
    for col in ["pct_correct_sd", "pct_correct_min", "pct_correct_max"]:
        if col not in base:
            base[col] = np.nan
    pd.concat([base, rows], ignore_index=True).to_csv(
        OUT / f"cliff_directionality_{endpoint}.csv", index=False
    )


def prospective_table(summary: pd.DataFrame) -> None:
    base = pd.read_csv(BASE / "prospective_pair_deltas_thr55.csv")
    base = base[base["Features"] != "Graph"].copy()
    base = base.rename(columns={"Mean Δ, bootstrap 95% CI": "Mean Δ, uncertainty interval"})
    if "Uncertainty interval" not in base:
        base["Uncertainty interval"] = "bootstrap 95% CI"

    ext = pd.read_csv(GRAPH / "graph_external_metrics_summary.csv")
    g50 = summary[summary["Endpoint"] == "pic50"].set_index("Model")
    rows = []
    for _, row in ext.iterrows():
        model = row["Model"]
        n_correct = int(row["both_directions_correct_n"])
        if n_correct == len(SEEDS):
            both = "yes"
        elif n_correct == 0:
            both = "no"
        else:
            both = f"{n_correct}/{len(SEEDS)} seeds"
        rows.append({
            "Model": model,
            "Features": "Graph",
            "Δ N-ethyl (HC49−HC22)": row["delta_ethyl"],
            "Δ N-propyl (HC46−HC52)": row["delta_propyl"],
            "Mean Δ": row["mean_delta"],
            "Mean Δ CI low": row["mean_delta_min"],
            "Mean Δ CI high": row["mean_delta_max"],
            "Mean Δ, uncertainty interval": (
                f"[{row['mean_delta_min']:.2f}, {row['mean_delta_max']:.2f}]"
            ),
            "Both directions correct": both,
            "Training cliff pairs correct (%)": g50.loc[model, "directionality_pct"],
            "Uncertainty interval": "seed range (n=5)",
        })
    out = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values(["Mean Δ", "Model"], ascending=[False, True])
    out.to_csv(OUT / "prospective_pair_deltas_thr55.csv", index=False)


def external_table() -> None:
    base = pd.read_csv(BASE / "external_results.csv")
    base = base[base["Features"] != "Graph"].copy()
    ext = pd.read_csv(GRAPH / "graph_external_metrics_summary.csv")
    cols = ["Pearson_r", "Spearman_rho", "MAE_ext"]
    rows = []
    for _, r in ext.iterrows():
        row = {"Model": r["Model"], "Features": "Graph", "n_repeats": int(r["n_repeats"])}
        for metric in cols:
            row[metric] = r[metric]
            row[f"{metric}_sd"] = r[f"{metric}_sd"]
            row[f"{metric}_min"] = r[f"{metric}_min"]
            row[f"{metric}_max"] = r[f"{metric}_max"]
        rows.append(row)
    for col in ["n_repeats"] + [f"{m}_{s}" for m in cols for s in ["sd", "min", "max"]]:
        if col not in base:
            base[col] = np.nan
    pd.concat([base, pd.DataFrame(rows)], ignore_index=True).to_csv(
        OUT / "external_results.csv", index=False
    )


def repeat_table(repeats: pd.DataFrame) -> None:
    base = pd.read_csv(BASE / "cv5x5_pic50.csv")
    base = base[base["Features"] != "Graph"].copy()
    rows = []
    for model, group in repeats[repeats["Endpoint"] == "pic50"].groupby("Model", sort=False):
        vals = group.sort_values("Repeat_seed")["MAE"].to_numpy(float)
        rows.append({
            "Model": model,
            "Features": "Graph",
            "MAE_mean": vals.mean(),
            "MAE_std": vals.std(ddof=1),
            "MAE_repeats": json.dumps(vals.tolist()),
            "repeat_basis": "fixed folds; training seeds 42–46",
        })
    if "repeat_basis" not in base:
        base["repeat_basis"] = "independent fold assignments"
    pd.concat([base, pd.DataFrame(rows)], ignore_index=True).to_csv(
        OUT / "cv5x5_pic50.csv", index=False
    )


def pair_key_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pair_key"] = ["||".join(sorted((a, b))) for a, b in zip(df.smiles_i, df.smiles_j)]
    # Mirror the historical dictionary-based implementation, which keeps the
    # last row when duplicated canonical SMILES produce the same pair key.
    return df.drop_duplicates("pair_key", keep="last")


def cross_endpoint_table() -> None:
    classical = pd.read_csv(BASE / "cross_endpoint_consistency.csv")
    classical = classical[classical["Features"] != "Graph"].copy()
    classical["Repeat_seed"] = np.nan
    graph_rows = []
    for model_key, model_label in [("chemprop", "ChemProp"), ("chemeleon", "ChemProp+CheMeleon")]:
        for seed in SEEDS:
            p50 = pair_key_frame(GRAPH / f"directionality_pic50_{model_key}_seed{seed}.csv")
            pki = pair_key_frame(GRAPH / f"directionality_pki_{model_key}_seed{seed}.csv")
            shared = p50.merge(pki, on="pair_key", suffixes=("_pic50", "_pki"))
            for _, r in shared.iterrows():
                c50, cki = int(r["correct_pic50"]), int(r["correct_pki"])
                state = "both_correct" if c50 and cki else "both_wrong" if not c50 and not cki else "discordant"
                smi_i, smi_j = r["pair_key"].split("||", 1)
                graph_rows.append({
                    "Model": model_label,
                    "Features": "Graph",
                    "smi_i": smi_i,
                    "smi_j": smi_j,
                    "sali_pic50": r["SALI_pic50"],
                    "sali_pki": r["SALI_pki"],
                    "correct_pic50": c50,
                    "correct_pki": cki,
                    "consistency": state,
                    "Repeat_seed": seed,
                })
    out = pd.concat([classical, pd.DataFrame(graph_rows)], ignore_index=True)
    out.to_csv(OUT / "cross_endpoint_consistency.csv", index=False)


def combine_machine_readable_graph_outputs() -> None:
    oof_frames = []
    direction_frames = []
    for endpoint in ["pic50", "pki"]:
        for model_key in ["chemprop", "chemeleon"]:
            for seed in SEEDS:
                oof_frames.append(pd.read_csv(GRAPH / f"oof_{endpoint}_{model_key}_seed{seed}.csv"))
                direction_frames.append(
                    pd.read_csv(GRAPH / f"directionality_{endpoint}_{model_key}_seed{seed}.csv")
                )
    pd.concat(oof_frames, ignore_index=True).to_csv(
        OUT / "graph_oof_predictions_per_repeat.csv", index=False
    )
    pd.concat(direction_frames, ignore_index=True).to_csv(
        OUT / "graph_directionality_per_repeat.csv", index=False
    )
    for name in [
        "graph_metrics_per_repeat.csv",
        "graph_metrics_summary.csv",
        "graph_external_predictions_per_repeat.csv",
        "graph_external_predictions_summary.csv",
        "graph_external_metrics_per_repeat.csv",
        "graph_external_metrics_summary.csv",
    ]:
        (OUT / name).write_bytes((GRAPH / name).read_bytes())
    metadata = json.loads((GRAPH / "graph_rerun_metadata.json").read_text())
    metadata["seeds"] = ",".join(map(str, metadata["seeds"]))
    pd.DataFrame([metadata]).to_csv(OUT / "graph_rerun_metadata.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = graph_summary()
    repeats = graph_repeats()
    for endpoint in ["pic50", "pki"]:
        add_graph_metric_rows(endpoint, summary)
        add_directionality_rows(endpoint, summary)
    prospective_table(summary)
    external_table()
    repeat_table(repeats)
    cross_endpoint_table()
    combine_machine_readable_graph_outputs()
    for name in ["leave_cliff_out_pic50.csv", "sali_threshold_sweep.csv", "tukey_hsd_pic50.csv"]:
        (OUT / name).write_bytes((BASE / name).read_bytes())
    print(f"Wrote updated result tables to {OUT}")


if __name__ == "__main__":
    main()
