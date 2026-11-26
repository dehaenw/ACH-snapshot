#!/usr/bin/env python3
"""Deterministic Chemprop and CheMeleon reruns for the NMDA QSAR study.

The script intentionally keeps the original evaluation design (five shuffled,
row-wise folds; 40 epochs; package-default optimiser and batch size) while
fixing two defects in the historical graph-model branch:

1. every stochastic source is assigned an explicit, recorded seed; and
2. the CheMeleon encoder is instantiated from the checkpoint hyperparameters
   and its state dictionary is loaded strictly before a new regression head is
   attached.

Each endpoint/model/repetition is written separately, so a long CPU run can be
resumed without repeating completed work.  Aggregate CSV and JSON files are
created after all requested repetitions are present.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import lightning
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import rdkit
import sklearn
import torch
from chemprop import __version__ as chemprop_version
from chemprop import data as cp_data
from chemprop import nn as cp_nn
from chemprop.models import MPNN
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from external_data import EXT_LABELS, EXT_PIC50, EXT_SMILES


# Keep long seeded runs readable; RDKit otherwise emits one deprecation line
# for every molecule when reproducing the historical Morgan fingerprint setup.
RDLogger.DisableLog("rdApp.warning")


SCRIPT_DIR = Path(__file__).resolve().parent
# In the standalone release the script lives in ``scripts/``; in the working
# analysis directory it lives beside ``data/`` and ``results/``.
ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "data").is_dir() else SCRIPT_DIR
DATA_DIR = ROOT / "data"
CHECKPOINT_PATH = ROOT / "checkpoints" / "chemeleon_mp.pt"
RESULTS_DIR = ROOT / "results"

ENDPOINT_FILES = {
    "pic50": (DATA_DIR / "pic50_clean.csv", "pIC50"),
    "pki": (DATA_DIR / "pki_clean.csv", "pKi"),
}
MODEL_LABELS = {
    "chemprop": "ChemProp",
    "chemeleon": "ChemProp+CheMeleon",
}
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
N_FOLDS = 5
FOLD_SEED = 42
SALI_THRESHOLD = 5.5
FP_RADIUS = 2
FP_SIZE = 2048
BATCH_SIZE = 64

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", nargs="+", choices=sorted(ENDPOINT_FILES),
                        default=sorted(ENDPOINT_FILES))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_LABELS),
                        default=sorted(MODEL_LABELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--threads", type=int, default=0,
                        help="Torch CPU thread count; 0 keeps the runtime default.")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run one pIC50 fold for each requested model and stop.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True, verbose=False)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def trainer_accelerator(requested: str) -> str:
    if requested == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu requested but CUDA is unavailable")
        return "gpu"
    if requested == "cpu":
        return "cpu"
    return "gpu" if torch.cuda.is_available() else "cpu"


def load_endpoint(endpoint: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path, target_col = ENDPOINT_FILES[endpoint]
    raw = pd.read_csv(path)
    df = raw[["Canonical SMILES", target_col]].dropna().copy()
    df.columns = ["smiles", "y"]
    valid = df["smiles"].map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    df = df.loc[valid].reset_index(drop=True)
    return df, df["smiles"].to_numpy(), df["y"].to_numpy(dtype=float)


def make_dataset(smiles: np.ndarray | list[str], y: np.ndarray | None = None):
    datapoints = []
    for i, smi in enumerate(smiles):
        target = None if y is None else np.array([y[i]], dtype=float)
        datapoints.append(cp_data.MoleculeDatapoint(
            mol=Chem.MolFromSmiles(str(smi)), y=target
        ))
    return cp_data.MoleculeDataset(datapoints)


def load_chemeleon_encoder() -> tuple[cp_nn.BondMessagePassing, dict[str, Any]]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    hparams = dict(checkpoint["hyper_parameters"])
    encoder = cp_nn.BondMessagePassing(**hparams)
    load_result = encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"CheMeleon strict load failed: {load_result}")
    audit = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": sha256(CHECKPOINT_PATH),
        "hyper_parameters": hparams,
        "state_dict_keys": list(checkpoint["state_dict"].keys()),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "encoder_output_dim": int(encoder.output_dim),
    }
    return encoder, audit


def build_model(model_key: str, scaler) -> tuple[MPNN, dict[str, Any]]:
    if model_key == "chemeleon":
        message_passing, audit = load_chemeleon_encoder()
        source = "CheMeleon checkpoint; strict encoder load"
    else:
        message_passing = cp_nn.BondMessagePassing()
        audit = {
            "checkpoint": None,
            "checkpoint_sha256": None,
            "hyper_parameters": dict(message_passing.hparams),
            "state_dict_keys": None,
            "missing_keys": [],
            "unexpected_keys": [],
            "encoder_output_dim": int(message_passing.output_dim),
        }
        source = "random initialization"

    predictor = cp_nn.RegressionFFN(
        input_dim=message_passing.output_dim,
        n_tasks=1,
        output_transform=cp_nn.UnscaleTransform.from_standard_scaler(scaler),
    )
    model = MPNN(
        message_passing=message_passing,
        agg=cp_nn.MeanAggregation(),
        predictor=predictor,
    )
    audit.update({
        "initialization": source,
        "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "predictor_hyper_parameters": dict(predictor.hparams),
        "mpnn_hyper_parameters": {
            "batch_norm": bool(model.hparams["batch_norm"]),
            "warmup_epochs": int(model.hparams["warmup_epochs"]),
            "init_lr": float(model.hparams["init_lr"]),
            "max_lr": float(model.hparams["max_lr"]),
            "final_lr": float(model.hparams["final_lr"]),
        },
    })
    return model, audit


def fit_model(
    smiles_train: np.ndarray,
    y_train: np.ndarray,
    model_key: str,
    seed: int,
    epochs: int,
    accelerator: str,
) -> tuple[MPNN, pl.Trainer, dict[str, Any]]:
    set_seed(seed)
    dataset = make_dataset(smiles_train, y_train)
    scaler = dataset.normalize_targets()
    model, audit = build_model(model_key, scaler)
    loader = cp_data.build_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        seed=seed,
        drop_last=False,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, loader)
    audit.update({
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": BATCH_SIZE,
        "num_workers": 0,
        "shuffle": True,
        "dataloader_seed": int(seed),
        "target_mean": float(scaler.mean_[0]),
        "target_scale": float(scaler.scale_[0]),
        "accelerator": accelerator,
    })
    return model, trainer, audit


def predict(model: MPNN, trainer: pl.Trainer, smiles: np.ndarray | list[str]) -> np.ndarray:
    dataset = make_dataset(smiles)
    loader = cp_data.build_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )
    batches = trainer.predict(model, loader)
    return torch.cat(batches).detach().cpu().numpy().reshape(-1)


def compute_sali(smiles: np.ndarray, y: np.ndarray) -> np.ndarray:
    mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, FP_RADIUS, nBits=FP_SIZE) for m in mols]
    n = len(fps)
    sali = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            denominator = 1.0 - similarity
            if denominator > 1e-6:
                sali[i, j] = sali[j, i] = abs(y[i] - y[j]) / denominator
    return sali


def metrics(y: np.ndarray, pred: np.ndarray, cliff_mask: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "R2": float(r2_score(y, pred)),
        "MAE_cliff": float(mean_absolute_error(y[cliff_mask], pred[cliff_mask])),
        "R2_cliff": float(r2_score(y[cliff_mask], pred[cliff_mask])),
    }


def directionality_rows(
    endpoint: str,
    model_key: str,
    repeat_seed: int,
    smiles: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    sali: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, j in zip(*np.where(np.triu(sali > SALI_THRESHOLD, k=1))):
        true_delta = float(y[i] - y[j])
        pred_delta = float(pred[i] - pred[j])
        rows.append({
            "Endpoint": endpoint,
            "Model": MODEL_LABELS[model_key],
            "Features": "Graph",
            "Repeat_seed": int(repeat_seed),
            "idx_i": int(i),
            "idx_j": int(j),
            "smiles_i": str(smiles[i]),
            "smiles_j": str(smiles[j]),
            "SALI": float(sali[i, j]),
            "y_true_i": float(y[i]),
            "y_true_j": float(y[j]),
            "y_pred_i": float(pred[i]),
            "y_pred_j": float(pred[j]),
            "true_delta": true_delta,
            "pred_delta": pred_delta,
            "correct": int(np.sign(true_delta) == np.sign(pred_delta)),
        })
    return pd.DataFrame(rows)


def run_repetition(
    endpoint: str,
    model_key: str,
    repeat_seed: int,
    epochs: int,
    accelerator: str,
    benchmark: bool,
) -> None:
    model_label = MODEL_LABELS[model_key]
    safe_model = model_key
    run_stem = f"{endpoint}_{safe_model}_seed{repeat_seed}"
    prediction_path = RESULTS_DIR / f"oof_{run_stem}.csv"
    metrics_path = RESULTS_DIR / f"metrics_{run_stem}.json"
    directionality_path = RESULTS_DIR / f"directionality_{run_stem}.csv"
    external_path = RESULTS_DIR / f"external_{run_stem}.csv"

    df, smiles, y = load_endpoint(endpoint)
    sali = compute_sali(smiles, y)
    cliff_mask = (sali > SALI_THRESHOLD).any(axis=1)
    folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(smiles))
    if benchmark:
        folds = folds[:1]

    oof = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    audits = []
    started = time.perf_counter()
    for fold_idx, (train_idx, valid_idx) in enumerate(folds):
        fold_seed = int(repeat_seed * 100 + fold_idx)
        fold_started = time.perf_counter()
        model, trainer, audit = fit_model(
            smiles[train_idx], y[train_idx], model_key, fold_seed, epochs, accelerator
        )
        fold_pred = predict(model, trainer, smiles[valid_idx])
        oof[valid_idx] = fold_pred
        fold_rows.append({
            "fold": int(fold_idx),
            "fold_seed": fold_seed,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "MAE": float(mean_absolute_error(y[valid_idx], fold_pred)),
            "R2": float(r2_score(y[valid_idx], fold_pred)),
            "seconds": float(time.perf_counter() - fold_started),
        })
        audits.append(audit)
        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if benchmark:
        payload = {
            "benchmark": True,
            "endpoint": endpoint,
            "model": model_label,
            "repeat_seed": repeat_seed,
            "epochs": epochs,
            "folds": fold_rows,
            "audit": audits[0],
            "seconds": float(time.perf_counter() - started),
        }
        path = RESULTS_DIR / f"benchmark_{run_stem}_{epochs}epochs.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps(payload, indent=2, default=str), flush=True)
        return

    if np.isnan(oof).any():
        raise RuntimeError(f"Missing OOF predictions for {run_stem}")

    summary = metrics(y, oof, cliff_mask)
    directionality = directionality_rows(
        endpoint, model_key, repeat_seed, smiles, y, oof, sali
    )
    summary["directionality_pct"] = float(100 * directionality["correct"].mean())
    summary["n_cliff_pairs"] = int(len(directionality))
    summary.update({
        "Endpoint": endpoint,
        "Model": model_label,
        "Features": "Graph",
        "Repeat_seed": int(repeat_seed),
        "fold_seed": FOLD_SEED,
        "n_folds": N_FOLDS,
        "n_rows": int(len(df)),
        "n_cliff_compounds": int(cliff_mask.sum()),
        "SALI_threshold": SALI_THRESHOLD,
        "epochs": int(epochs),
        "seconds": float(time.perf_counter() - started),
        "folds": fold_rows,
        "model_audit": audits[0],
    })

    pd.DataFrame({
        "Endpoint": endpoint,
        "Model": model_label,
        "Features": "Graph",
        "Repeat_seed": repeat_seed,
        "row_index": np.arange(len(df), dtype=int),
        "smiles": smiles,
        "y_true": y,
        "y_pred": oof,
        "is_cliff_compound": cliff_mask,
    }).to_csv(prediction_path, index=False)
    directionality.to_csv(directionality_path, index=False)

    if endpoint == "pic50":
        full_seed = int(repeat_seed * 100 + 99)
        full_model, full_trainer, full_audit = fit_model(
            smiles, y, model_key, full_seed, epochs, accelerator
        )
        ext_pred = predict(full_model, full_trainer, EXT_SMILES)
        pd.DataFrame({
            "Model": model_label,
            "Features": "Graph",
            "Repeat_seed": repeat_seed,
            "training_seed": full_seed,
            "compound": EXT_LABELS,
            "smiles": EXT_SMILES,
            "observed_pIC50": EXT_PIC50,
            "predicted_pIC50": ext_pred,
        }).to_csv(external_path, index=False)
        summary["full_fit_audit"] = full_audit
        del full_model, full_trainer
        gc.collect()

    metrics_path.write_text(json.dumps(summary, indent=2, default=str))
    print(
        f"completed {endpoint} {model_label} seed={repeat_seed}: "
        f"MAE={summary['MAE']:.4f}, R2={summary['R2']:.4f}, "
        f"cliff MAE={summary['MAE_cliff']:.4f}, "
        f"directionality={summary['directionality_pct']:.1f}%",
        flush=True,
    )


def aggregate(requested_endpoints: list[str], requested_models: list[str], seeds: list[int]) -> None:
    metric_rows = []
    for endpoint in requested_endpoints:
        for model_key in requested_models:
            for seed in seeds:
                path = RESULTS_DIR / f"metrics_{endpoint}_{model_key}_seed{seed}.json"
                if not path.exists():
                    return
                payload = json.loads(path.read_text())
                metric_rows.append({
                    k: payload[k]
                    for k in [
                        "Endpoint", "Model", "Features", "Repeat_seed", "MAE", "R2",
                        "MAE_cliff", "R2_cliff", "directionality_pct", "n_cliff_pairs",
                        "seconds",
                    ]
                })

    per_repeat = pd.DataFrame(metric_rows)
    per_repeat.to_csv(RESULTS_DIR / "graph_metrics_per_repeat.csv", index=False)
    agg_rows = []
    metric_cols = ["MAE", "R2", "MAE_cliff", "R2_cliff", "directionality_pct"]
    for (endpoint, model, features), group in per_repeat.groupby(
        ["Endpoint", "Model", "Features"], sort=False
    ):
        row: dict[str, Any] = {
            "Endpoint": endpoint,
            "Model": model,
            "Features": features,
            "n_repeats": int(len(group)),
            "seeds": ",".join(str(x) for x in group["Repeat_seed"]),
        }
        for metric in metric_cols:
            values = group[metric].to_numpy(float)
            row[metric] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(RESULTS_DIR / "graph_metrics_summary.csv", index=False)

    ext_frames = []
    for model_key in requested_models:
        for seed in seeds:
            path = RESULTS_DIR / f"external_pic50_{model_key}_seed{seed}.csv"
            if path.exists():
                ext_frames.append(pd.read_csv(path))
    if ext_frames:
        external = pd.concat(ext_frames, ignore_index=True)
        external.to_csv(RESULTS_DIR / "graph_external_predictions_per_repeat.csv", index=False)
        ext_summary = (
            external.groupby(["Model", "Features", "compound", "observed_pIC50"], as_index=False)
            ["predicted_pIC50"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        ext_summary.to_csv(RESULTS_DIR / "graph_external_predictions_summary.csv", index=False)

        ext_metric_rows = []
        for (model, seed), group in external.groupby(["Model", "Repeat_seed"]):
            ordered = group.set_index("compound").loc[EXT_LABELS]
            observed = ordered["observed_pIC50"].to_numpy(float)
            predicted = ordered["predicted_pIC50"].to_numpy(float)
            ext_metric_rows.append({
                "Model": model,
                "Features": "Graph",
                "Repeat_seed": int(seed),
                "Pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
                "Spearman_rho": float(pd.Series(observed).corr(pd.Series(predicted), method="spearman")),
                "MAE_ext": float(mean_absolute_error(observed, predicted)),
                "delta_ethyl": float(predicted[0] - predicted[1]),
                "delta_propyl": float(predicted[2] - predicted[3]),
                "mean_delta": float(((predicted[0] - predicted[1]) + (predicted[2] - predicted[3])) / 2),
                "both_directions_correct": bool(predicted[0] > predicted[1] and predicted[2] > predicted[3]),
            })
        ext_metrics = pd.DataFrame(ext_metric_rows)
        ext_metrics.to_csv(RESULTS_DIR / "graph_external_metrics_per_repeat.csv", index=False)
        summary_rows = []
        for model, group in ext_metrics.groupby("Model", sort=False):
            row = {"Model": model, "Features": "Graph", "n_repeats": int(len(group))}
            for metric in ["Pearson_r", "Spearman_rho", "MAE_ext", "delta_ethyl", "delta_propyl", "mean_delta"]:
                values = group[metric].to_numpy(float)
                row[metric] = float(values.mean())
                row[f"{metric}_sd"] = float(values.std(ddof=1))
                row[f"{metric}_min"] = float(values.min())
                row[f"{metric}_max"] = float(values.max())
            row["both_directions_correct_n"] = int(group["both_directions_correct"].sum())
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            RESULTS_DIR / "graph_external_metrics_summary.csv", index=False
        )

    metadata = {
        "chemprop_version": chemprop_version,
        "lightning_version": lightning.__version__,
        "torch_version": torch.__version__,
        "rdkit_version": rdkit.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "seeds": seeds,
        "fold_seed": FOLD_SEED,
        "n_folds": N_FOLDS,
        "sali_threshold": SALI_THRESHOLD,
        "batch_size": BATCH_SIZE,
        "pic50_sha256": sha256(ENDPOINT_FILES["pic50"][0]),
        "pki_sha256": sha256(ENDPOINT_FILES["pki"][0]),
        "external_measurements_sha256": sha256(DATA_DIR / "external_concentration_response.csv"),
        "chemeleon_checkpoint_sha256": sha256(CHECKPOINT_PATH),
    }
    (RESULTS_DIR / "graph_rerun_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(4, args.threads)))
    accelerator = trainer_accelerator(args.device)

    if args.benchmark:
        endpoint = "pic50"
        seed = args.seeds[0]
        for model_key in args.models:
            run_repetition(
                endpoint, model_key, seed, args.epochs, accelerator, benchmark=True
            )
        return

    for endpoint in args.endpoints:
        for model_key in args.models:
            for seed in args.seeds:
                metrics_path = RESULTS_DIR / f"metrics_{endpoint}_{model_key}_seed{seed}.json"
                if metrics_path.exists() and not args.force:
                    print(f"skipping completed {metrics_path.name}", flush=True)
                    continue
                run_repetition(
                    endpoint, model_key, seed, args.epochs, accelerator, benchmark=False
                )
    aggregate(args.endpoints, args.models, args.seeds)


if __name__ == "__main__":
    main()
