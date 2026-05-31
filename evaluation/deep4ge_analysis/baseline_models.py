#!/usr/bin/env python3
"""Run lightweight baselines for Deep4ge.

Baselines (Data Showcase friendly):
- Fault detection: buggy vs. correct (binary)
- Fault diagnosis: predict fault_category among buggy logs (multi-class)

Key design choice: group-aware CV by StackOverflow ID (so_id) to reduce leakage.

Per-log features:
For each training log CSV, we use the *final-epoch* values of the 30 dynamic
features (i.e., all columns except the epoch index). This keeps the baselines
lightweight while still reflecting training dynamics at convergence.

This script is robust to two common layouts:

(A) You pass the *repo root* (recommended):
    repo_root/
      data/manifest.csv
      data/training_logs/...
(B) You pass the *data directory*:
    data/
      manifest.csv
      training_logs/...

Outputs (under <out_dir>/):
- baseline_results.json
- tables/baseline_results.tex

Example:
  python3 analysis/baseline_models.py --data-root . --out-dir output/analysis_artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# StratifiedGroupKFold is preferred when available (reduces label skew per fold)
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # older scikit-learn
    StratifiedGroupKFold = None


REQUIRED_MANIFEST_COLUMNS = {"subset", "so_id", "fault_category", "run_number", "num_epochs", "csv_path"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Deep4ge baseline models")
    p.add_argument("--data-root", type=Path, required=True, help="Repo root or data directory.")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-logs", type=int, default=None, help="Optional cap on number of logs to load (for quick iteration).")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--rf-estimators", type=int, default=200, help="Number of trees for RandomForest baseline.")
    return p.parse_args()


def infer_manifest_path(data_root: Path, manifest_arg: Optional[Path]) -> Path:
    if manifest_arg is not None:
        return manifest_arg
    c1 = data_root / "manifest.csv"
    c2 = data_root / "data" / "manifest.csv"
    if c1.exists():
        return c1
    if c2.exists():
        return c2
    raise FileNotFoundError(f"Could not find manifest.csv under {data_root} (tried {c1} and {c2}).")


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    df["subset"] = df["subset"].astype(str).str.lower()
    df["fault_category"] = df["fault_category"].astype(str).str.lower()
    df["so_id"] = df["so_id"].astype(str)
    df["csv_path"] = df["csv_path"].astype(str)

    return df


def infer_repo_root(manifest_path: Path, manifest_df: pd.DataFrame) -> Path:
    sample = None
    for v in manifest_df["csv_path"].head(50).tolist():
        if isinstance(v, str) and v.strip():
            sample = v
            break
    if sample is None:
        return manifest_path.parent

    candidates = [
        manifest_path.parent,
        manifest_path.parent.parent,
        manifest_path.parent.parent.parent,
    ]
    for c in candidates:
        if (c / sample).exists():
            return c
    return manifest_path.parent


def resolve_log_path(repo_root: Path, csv_path_value: str) -> Path:
    p = Path(csv_path_value)
    return p if p.is_absolute() else (repo_root / p)


def read_last_csv_row(path: Path) -> List[str]:
    """Read the last non-empty row from a CSV efficiently."""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        size = min(8192, end)  # logs are small; 8KB is sufficient
        f.seek(-size, os.SEEK_END)
        chunk = f.read(size)

    lines = chunk.splitlines()
    for line in reversed(lines):
        if line.strip():
            # csv.reader handles quoting/escaping
            return next(csv.reader([line.decode("utf-8", errors="ignore")]))

    # Fallback: read normally (should not happen for valid logs)
    last: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            last = row
    return last


def to_float(val: str) -> float:
    v = str(val).strip()
    if v == "":
        return float("nan")
    if v.lower() == "true":
        return 1.0
    if v.lower() == "false":
        return 0.0
    try:
        return float(v)
    except Exception:
        return float("nan")


def build_feature_matrix(
    df: pd.DataFrame,
    repo_root: Path,
    max_logs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return X, y_binary, y_category, groups(so_id), feature_names."""
    rows = df
    if max_logs is not None:
        rows = rows.sample(n=min(max_logs, len(rows)), random_state=7)

    # Discover schema from the first log (assumes consistent headers across logs).
    first_path = resolve_log_path(repo_root, str(rows.iloc[0]["csv_path"]))
    header = first_path.open("r", encoding="utf-8", errors="ignore").readline().strip().split(",")
    # Expect: epoch + 30 dynamic columns
    if len(header) < 2:
        raise ValueError(f"Unexpected CSV header in {first_path}: {header}")
    feature_names = [c for c in header if c.lower() != "epoch"]

    X = np.empty((len(rows), len(feature_names)), dtype=float)
    y_bin = np.empty(len(rows), dtype=int)
    y_cat = np.empty(len(rows), dtype=object)
    groups = np.empty(len(rows), dtype=object)

    for i, r in enumerate(rows.itertuples(index=False)):
        log_path = resolve_log_path(repo_root, str(getattr(r, "csv_path")))
        last = read_last_csv_row(log_path)

        # Map last-row values to floats, skipping epoch column (assumed first).
        vals = [to_float(v) for v in last[1:]]  # exclude epoch
        if len(vals) != len(feature_names):
            raise ValueError(f"Unexpected row width in {log_path} (got {len(vals)}, expected {len(feature_names)})")
        X[i, :] = vals

        subset = str(getattr(r, "subset")).lower()
        y_bin[i] = 1 if subset == "buggy" else 0
        y_cat[i] = str(getattr(r, "fault_category")).lower()
        groups[i] = str(getattr(r, "so_id"))

    # Replace inf with NaN for sklearn imputers.
    X[np.isinf(X)] = np.nan
    return X, y_bin, y_cat, groups, feature_names


def evaluate_model(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    is_multiclass: bool,
) -> Dict[str, object]:
    """Group-aware CV. Returns metrics plus raw per-fold predictions.

    The raw predictions (_y_true, _y_pred, _y_score) are kept so the
    bootstrap-CI step can reuse them without retraining.
    """
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=groups)

    avg = "macro" if is_multiclass else "binary"

    y_true_all = np.empty(0, dtype=y.dtype)
    y_pred_all = np.empty(0, dtype=y.dtype)
    y_score_all: Optional[np.ndarray] = None

    for train_idx, test_idx in split_iter:
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        y_true_all = np.concatenate([y_true_all, y[test_idx]])
        y_pred_all = np.concatenate([y_pred_all, pred])

        if not is_multiclass and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X[test_idx])[:, 1]
                y_score_all = proba if y_score_all is None else np.concatenate([y_score_all, proba])
            except Exception:
                y_score_all = None

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "precision": float(precision_score(y_true_all, y_pred_all, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true_all, y_pred_all, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true_all, y_pred_all, average=avg, zero_division=0)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "mcc": float(matthews_corrcoef(y_true_all, y_pred_all)),
    }

    if not is_multiclass and y_score_all is not None and len(np.unique(y_true_all)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true_all, y_score_all))
        metrics["auprc"] = float(average_precision_score(y_true_all, y_score_all))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    metrics["_y_true"] = y_true_all
    metrics["_y_pred"] = y_pred_all
    metrics["_y_score"] = y_score_all
    return metrics


def make_lr_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=7,
                ),
            ),
        ]
    )


def make_rf_pipeline(n_estimators: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=7,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def make_dummy_pipeline(strategy: str, constant=None) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DummyClassifier(strategy=strategy, constant=constant, random_state=7)),
        ]
    )


def write_latex_table(results: List[Dict[str, object]], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("% Auto-generated by analysis/baseline_models.py")
    lines.append("\\begin{tabular}{@{}llrrrrr@{}}\\toprule")
    lines.append("Task & Model & F1 & Macro-F1 & Bal.Acc & MCC & AUPRC \\\\ \\midrule")
    last_task = None
    for r in results:
        if last_task is not None and r["task"] != last_task:
            lines.append("\\midrule")
        last_task = r["task"]
        auprc = r.get("auprc", float("nan"))
        auprc_str = "--" if (isinstance(auprc, float) and np.isnan(auprc)) else f"{auprc:.3f}"
        lines.append(
            f"{r['task']} & {r['model']} & "
            f"{r['f1']:.3f} & {r['macro_f1']:.3f} & "
            f"{r['balanced_accuracy']:.3f} & {r['mcc']:.3f} & "
            f"{auprc_str} \\\\"
        )
    lines.append("\\bottomrule\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    manifest_path = infer_manifest_path(args.data_root, args.manifest)
    df = read_manifest(manifest_path)
    repo_root = infer_repo_root(manifest_path, df)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    X, y_binary, y_category, groups, feature_names = build_feature_matrix(df, repo_root=repo_root, max_logs=args.max_logs)

    results: List[Dict[str, object]] = []

    # Fault detection: naive baselines first, then real models.
    detection_models = [
        ("Always-faulty", make_dummy_pipeline("constant", constant=1)),
        ("Stratified", make_dummy_pipeline("stratified")),
        ("Uniform", make_dummy_pipeline("uniform")),
        ("LogReg", make_lr_pipeline()),
        ("RandForest", make_rf_pipeline(args.rf_estimators)),
    ]
    for name, pipe in detection_models:
        res = evaluate_model(pipe, X, y_binary, groups, n_splits=args.n_splits, is_multiclass=False)
        results.append({"task": "Fault detection", "model": name, **res})

    # Fault diagnosis (buggy only): stratified reference, then real models.
    buggy_mask = y_binary == 1
    Xb = X[buggy_mask]
    yb = y_category[buggy_mask]
    gb = groups[buggy_mask]

    diagnosis_models = [
        ("Stratified", make_dummy_pipeline("stratified")),
        ("LogReg", make_lr_pipeline()),
        ("RandForest", make_rf_pipeline(args.rf_estimators)),
    ]
    for name, pipe in diagnosis_models:
        res = evaluate_model(pipe, Xb, yb, gb, n_splits=args.n_splits, is_multiclass=True)
        results.append({"task": "Fault diagnosis", "model": name, **res})

    # Save the raw per-fold predictions so bootstrap_cis.py can reuse them.
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        name_safe = f"{r['task'].replace(' ', '_')}__{r['model'].replace(' ', '_')}"
        score = r["_y_score"]
        np.savez_compressed(
            preds_dir / f"{name_safe}.npz",
            y_true=r["_y_true"],
            y_pred=r["_y_pred"],
            y_score=score if score is not None else np.array([]),
        )

    # JSON gets only the serializable metric fields (drop the raw arrays).
    clean_results = [
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ]
    (out_dir / "baseline_results.json").write_text(
        json.dumps(clean_results, indent=2, default=float), encoding="utf-8"
    )
    write_latex_table(clean_results, tables_dir / "baseline_results.tex")

    print(f"Wrote {out_dir/'baseline_results.json'}")
    print(f"Wrote {tables_dir/'baseline_results.tex'}")
    print(f"Wrote per-fold predictions under {preds_dir}")


if __name__ == "__main__":
    main()
