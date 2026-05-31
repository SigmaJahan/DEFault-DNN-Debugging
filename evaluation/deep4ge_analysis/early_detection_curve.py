#!/usr/bin/env python3
"""Compute and plot fault-detection metrics vs first-k epochs.

For each k in K_VALUES, build a feature matrix from row k of every run's CSV
(or the last available row if the run has fewer than k+1 epochs), train a
group-aware-CV RF detector, and record F1 / Bal.Acc / MCC.

Outputs:
  output/analysis_artifacts/figures/early_detection.pdf
  output/analysis_artifacts/early_detection.json

Example:
  python3 analysis/early_detection_curve.py --data-root . \
      --out-dir output/analysis_artifacts
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline


K_VALUES = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]


def find_manifest(data_root: Path) -> Path:
    for c in [data_root / "data" / "manifest.csv", data_root / "manifest.csv"]:
        if c.exists():
            return c
    raise FileNotFoundError("manifest.csv not found")


def find_repo_root(manifest_path: Path, df: pd.DataFrame) -> Path:
    sample = next((v for v in df["csv_path"].head(50) if isinstance(v, str)), None)
    if sample is None:
        return manifest_path.parent
    for c in [manifest_path.parent, manifest_path.parent.parent]:
        if (c / sample).exists():
            return c
    return manifest_path.parent


def to_float(v: str) -> float:
    s = str(v).strip()
    if not s:
        return float("nan")
    if s.lower() == "true":
        return 1.0
    if s.lower() == "false":
        return 0.0
    try:
        return float(s)
    except Exception:
        return float("nan")


def read_row_at(path: Path, target_epoch: int):
    """Return (row_values, actual_epoch). Uses min(target_epoch, last_epoch)."""
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        rdr = csv.reader(f)
        next(rdr, None)  # skip header
        for r in rdr:
            if r and any(c.strip() for c in r):
                rows.append(r)
    if not rows:
        return None, -1
    idx = min(target_epoch, len(rows) - 1)
    return rows[idx], idx


def build_X_at_epoch(df, repo_root: Path, k: int):
    rows = list(df.itertuples(index=False))
    first_path = repo_root / rows[0].csv_path
    with first_path.open("r") as f:
        header = f.readline().strip().split(",")
    feat_idx = [i for i, c in enumerate(header) if c.lower() != "epoch"]

    X = np.full((len(rows), len(feat_idx)), np.nan)
    y = np.empty(len(rows), dtype=int)
    g = np.empty(len(rows), dtype=object)

    for i, r in enumerate(rows):
        p = repo_root / r.csv_path
        row, _ = read_row_at(p, k)
        if row is None:
            continue
        try:
            vals = [to_float(row[j]) for j in feat_idx]
        except IndexError:
            vals = [float("nan")] * len(feat_idx)
        X[i] = vals
        y[i] = 1 if str(r.subset).lower() == "buggy" else 0
        g[i] = str(r.so_id)
    X[np.isinf(X)] = np.nan
    return X, y, g


def evaluate_at_k(X, y, g, n_splits=5, n_estimators=200):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, random_state=7,
            n_jobs=-1, class_weight="balanced_subsample")),
    ])
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)

    y_true = np.empty(0, dtype=int)
    y_pred = np.empty(0, dtype=int)
    for tr, te in splitter.split(X, y, groups=g):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict(X[te])
        y_true = np.concatenate([y_true, y[te]])
        y_pred = np.concatenate([y_pred, p])
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--rf-estimators", type=int, default=200)
    args = ap.parse_args()

    mp = find_manifest(args.data_root)
    df = pd.read_csv(mp)
    df["subset"] = df["subset"].str.lower()
    df["so_id"] = df["so_id"].astype(str)
    df["csv_path"] = df["csv_path"].astype(str)
    repo_root = find_repo_root(mp, df)

    out_fig = args.out_dir / "figures"
    out_fig.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in K_VALUES:
        print(f"k = {k} ...", flush=True)
        X, y, g = build_X_at_epoch(df, repo_root, k)
        res = evaluate_at_k(X, y, g, args.n_splits, args.rf_estimators)
        rows.append({"k": k, **res})
        print(f"  F1={res['f1']:.3f}  BalAcc={res['balanced_accuracy']:.3f}  MCC={res['mcc']:.3f}")

    (args.out_dir / "early_detection.json").write_text(
        json.dumps(rows, indent=2, default=float), encoding="utf-8"
    )

    ks = [r["k"] for r in rows]
    f1 = [r["f1"] for r in rows]
    ba = [r["balanced_accuracy"] for r in rows]
    mc = [r["mcc"] for r in rows]

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.plot(ks, f1, marker="o", label="F1")
    ax.plot(ks, ba, marker="s", label="Balanced Accuracy")
    ax.plot(ks, mc, marker="^", label="MCC")
    ax.set_xlabel("Epochs available at decision time")
    ax.set_ylabel("Detection metric")
    ax.set_xticks(ks)
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_fig / "early_detection.pdf")
    print(f"Wrote {out_fig / 'early_detection.pdf'}")
    print(f"Wrote {args.out_dir / 'early_detection.json'}")


if __name__ == "__main__":
    main()
