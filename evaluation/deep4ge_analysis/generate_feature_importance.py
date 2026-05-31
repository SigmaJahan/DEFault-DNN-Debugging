#!/usr/bin/env python3
"""Generate RF feature importance figure and per-fault-category results table.

Outputs:
  figures/feature_importance_rf.pdf
  tables/per_fault_results.tex

Example:
  python3 analysis/generate_feature_importance.py --data-root . --out-dir output/analysis_artifacts
"""
from __future__ import annotations

import argparse
import os
import csv as _csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline


# ── feature group colouring ───────────────────────────────────────────────────
FEATURE_GROUPS: Dict[str, str] = {}

GRADIENT_FEATS = {
    "gradient_vanish", "gradient_explode", "nan_gradients_count",
    "mean_gradient", "gradient_std", "gradient_max", "gradient_min",
    "gradient_median", "mean_grad", "std_grad",
}
ACTIVATION_FEATS = {"dying_relu", "saturated_activation", "mean_activation", "std_activation"}
WEIGHT_FEATS = {
    "large_weight_count", "cons_mean_weight_count",
    "cons_std_weight_count", "nan_weight_count",
}
LOSS_ACC_FEATS = {
    "train_loss", "val_loss", "train_acc", "val_acc",
    "loss_oscillation", "acc_gap_too_big",
    "increase_loss_count", "decrease_acc_count",
}
SYSTEM_FEATS = {"cpu_utilization", "gpu_memory_utilization", "memory_usage", "adjusted_lr"}

GROUP_COLOR = {
    "Gradient":   "#4C78A8",
    "Activation": "#B279A2",
    "Weight":     "#54A24B",
    "Loss/Acc":   "#F58518",
    "System":     "#72B7B2",
    "Other":      "#BBBBBB",
}


def get_group(feat: str) -> str:
    f = feat.lower()
    if f in GRADIENT_FEATS:
        return "Gradient"
    if f in ACTIVATION_FEATS:
        return "Activation"
    if f in WEIGHT_FEATS:
        return "Weight"
    if f in LOSS_ACC_FEATS:
        return "Loss/Acc"
    if f in SYSTEM_FEATS:
        return "System"
    return "Other"


def configure_style() -> None:
    matplotlib.style.use("default")
    matplotlib.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": "#D8D8D8", "grid.linewidth": 0.5, "grid.alpha": 0.7,
        "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC", "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  Wrote {path}")


# ── data helpers ──────────────────────────────────────────────────────────────

def find_manifest(data_root: Path) -> Path:
    for cand in [data_root / "data" / "manifest.csv", data_root / "manifest.csv"]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"manifest.csv not found under {data_root}")


def find_repo_root(manifest_path: Path, df: pd.DataFrame) -> Path:
    sample = next((v for v in df["csv_path"].head(50) if isinstance(v, str) and v.strip()), None)
    if sample is None:
        return manifest_path.parent
    for cand in [manifest_path.parent, manifest_path.parent.parent]:
        if (cand / sample).exists():
            return cand
    return manifest_path.parent


def load_manifest(data_root: Path) -> Tuple[pd.DataFrame, Path]:
    mp = find_manifest(data_root)
    df = pd.read_csv(mp)
    df["subset"] = df["subset"].str.lower()
    df["fault_category"] = df["fault_category"].str.strip()
    df["so_id"] = df["so_id"].astype(str)
    df["csv_path"] = df["csv_path"].astype(str)
    repo_root = find_repo_root(mp, df)
    return df, repo_root


def resolve(repo_root: Path, csv_path: str) -> Path:
    p = Path(csv_path)
    return p if p.is_absolute() else (repo_root / p)


def _to_float(val: str) -> float:
    v = str(val).strip()
    if not v:
        return float("nan")
    if v.lower() == "true":
        return 1.0
    if v.lower() == "false":
        return 0.0
    try:
        return float(v)
    except Exception:
        return float("nan")


def _read_last_row(path: Path) -> List[str]:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = min(8192, f.tell())
        f.seek(-size, os.SEEK_END)
        chunk = f.read(size)
    for line in reversed(chunk.splitlines()):
        if line.strip():
            return next(_csv.reader([line.decode("utf-8", errors="ignore")]))
    return []


def build_feature_matrix(df: pd.DataFrame, repo_root: Path, max_logs: Optional[int] = None):
    if max_logs is not None:
        df = df.sample(n=min(max_logs, len(df)), random_state=7)

    first = resolve(repo_root, str(df.iloc[0]["csv_path"]))
    header = first.open("r", encoding="utf-8", errors="ignore").readline().strip().split(",")
    feat_names = [c for c in header if c.lower() != "epoch"]

    X = np.empty((len(df), len(feat_names)), dtype=float)
    y_bin = np.empty(len(df), dtype=int)
    y_cat = np.empty(len(df), dtype=object)
    groups = np.empty(len(df), dtype=object)

    for i, row in enumerate(df.itertuples(index=False)):
        p = resolve(repo_root, str(row.csv_path))
        last = _read_last_row(p)
        vals = [_to_float(v) for v in last[1:]]
        if len(vals) != len(feat_names):
            vals = vals[:len(feat_names)] + [float("nan")] * max(0, len(feat_names) - len(vals))
        X[i] = vals
        y_bin[i] = 1 if str(row.subset).lower() == "buggy" else 0
        y_cat[i] = str(row.fault_category)
        groups[i] = str(row.so_id)

    X[np.isinf(X)] = np.nan
    return X, y_bin, y_cat, groups, feat_names


# ── Figure B: feature importance ─────────────────────────────────────────────

def fig_feature_importance(X: np.ndarray, y: np.ndarray, feat_names: List[str],
                            out_path: Path, top_n: int = 15,
                            n_estimators: int = 300) -> None:
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=7, n_jobs=-1,
                                class_weight="balanced_subsample")
    rf.fit(X_imp, y)
    importances = rf.feature_importances_

    idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feat_names[i] for i in idx]
    top_imps = importances[idx]
    top_feats_rev = top_feats[::-1]
    top_imps_rev = top_imps[::-1]

    colors = [GROUP_COLOR[get_group(f)] for f in top_feats_rev]
    y_pos = np.arange(top_n)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(y_pos, top_imps_rev, color=colors, edgecolor="#333333", linewidth=0.4, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats_rev, fontsize=8)
    ax.set_xlabel("Mean decrease in impurity (RF feature importance)")
    ax.set_xlim(0, top_imps_rev.max() * 1.18)

    for i, v in enumerate(top_imps_rev):
        ax.text(v + top_imps_rev.max() * 0.01, i, f"{v:.3f}", va="center", ha="left", fontsize=7)

    # legend patches
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor="#333333", linewidth=0.4, label=g)
               for g, c in GROUP_COLOR.items() if g != "Other"]
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)

    save(fig, out_path)


# ── per-fault results table ───────────────────────────────────────────────────

def write_per_fault_table(df: pd.DataFrame, X: np.ndarray, y_bin: np.ndarray,
                           y_cat: np.ndarray, groups: np.ndarray,
                           out_path: Path, n_splits: int = 5,
                           n_estimators: int = 200) -> None:
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=7, n_jobs=-1,
                                       class_weight="balanced_subsample")),
    ])
    y_pred = np.full(len(y_bin), -1, dtype=int)
    for tr, te in splitter.split(X, y_bin, groups=groups):
        pipe.fit(X[tr], y_bin[tr])
        y_pred[te] = pipe.predict(X[te])

    categories = sorted(df[df["subset"] == "buggy"]["fault_category"].unique())
    n_neg = int((y_bin == 0).sum())  # number of correct baselines
    rows = []
    for cat in categories:
        mask = (y_cat == cat) | (y_bin == 0)
        n_cat = int((y_cat == cat).sum())
        y_t = y_bin[mask]
        y_p = y_pred[mask]
        prec = precision_score(y_t, y_p, pos_label=1, zero_division=0)
        rec = recall_score(y_t, y_p, pos_label=1, zero_division=0)
        f1 = f1_score(y_t, y_p, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(y_t, y_p) if len(np.unique(y_t)) == 2 else float("nan")
        # Trivial all-positive baseline F1 for this category.
        p = n_cat / (n_cat + n_neg) if (n_cat + n_neg) > 0 else 0.0
        trivial_f1 = 2 * p / (p + 1) if (p + 1) > 0 else 0.0
        rows.append((cat, n_cat, trivial_f1, f1, mcc))

    lines = [
        "% Auto-generated by analysis/generate_feature_importance.py",
        "\\begin{tabular}{@{}lrrrr@{}}\\toprule",
        "Fault Category & \\#Logs & Trivial F1 & F1 & MCC \\\\ \\midrule",
    ]
    for cat, n, trivial_f1, f1, mcc in rows:
        mcc_str = "--" if (isinstance(mcc, float) and np.isnan(mcc)) else f"{mcc:.3f}"
        lines.append(f"{cat} & {n:,} & {trivial_f1:.3f} & {f1:.3f} & {mcc_str} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--max-logs", type=int, default=None, help="Optional cap for quick smoke tests.")
    p.add_argument("--rf-estimators", type=int, default=300, help="Number of trees for the RF feature-importance model.")
    p.add_argument("--per-fault-estimators", type=int, default=200, help="Number of trees for the per-fault RF model.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    df, repo_root = load_manifest(args.data_root)

    fig_dir = args.out_dir / "figures"
    tab_dir = args.out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("Building feature matrix...")
    X, y_bin, y_cat, groups, feat_names = build_feature_matrix(df, repo_root, max_logs=args.max_logs)

    print("Generating feature importance figure…")
    fig_feature_importance(
        X, y_bin, feat_names, fig_dir / "feature_importance_rf.pdf",
        n_estimators=args.rf_estimators,
    )

    print("Generating per-fault-category results table…")
    write_per_fault_table(df, X, y_bin, y_cat, groups,
                          tab_dir / "per_fault_results.tex",
                          n_splits=args.n_splits,
                          n_estimators=args.per_fault_estimators)

    print("Done.")


if __name__ == "__main__":
    main()
