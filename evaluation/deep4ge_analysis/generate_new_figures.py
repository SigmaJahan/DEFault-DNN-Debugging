#!/usr/bin/env python3
"""Generate additional publication-quality figures for the Deep4ge paper.

Produces:
  figures/loss_curves_by_category.pdf  -- mean training loss per fault category
  figures/cat_arch_heatmap_improved.pdf -- row-normalised arch × category heatmap
  figures/per_fault_f1.pdf             -- RF per-fault-category detection F1

Example:
  python3 analysis/generate_new_figures.py --data-root . --out-dir output/analysis_artifacts
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
from matplotlib.ticker import MaxNLocator

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.pipeline import Pipeline


# ── colour palette ────────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Hyperparameter": "#4C78A8",
    "Loss":           "#F58518",
    "Weight":         "#54A24B",
    "Layer":          "#E45756",
    "Optimization":   "#72B7B2",
    "Activation":     "#B279A2",
    "Regularization": "#FF9DA7",
    "correct":        "#000000",
}
ARCH_ORDER = ["FNN", "CNN", "RNN"]


def configure_style() -> None:
    matplotlib.style.use("default")
    matplotlib.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#D8D8D8",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, facecolor="white", metadata={"Creator": "analysis/generate_new_figures.py"})
    plt.close(fig)
    print(f"  Wrote {path}")


# ── data loading ──────────────────────────────────────────────────────────────

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
    repo_root = find_repo_root(mp, df)
    return df, repo_root


def resolve(repo_root: Path, csv_path: str) -> Path:
    p = Path(csv_path)
    return p if p.is_absolute() else (repo_root / p)


def load_seed_metadata(data_root: Path) -> Optional[pd.DataFrame]:
    for cand in [
        data_root / "data" / "seed_programs" / "seed_metadata.csv",
        data_root / "seed_programs" / "seed_metadata.csv",
    ]:
        if cand.exists():
            sm = pd.read_csv(cand)
            sm["so_id"] = sm["so_id"].astype(str)
            sm["architecture"] = sm["architecture"].str.upper()
            return sm
    return None


# ── Figure A: per-category loss curves ───────────────────────────────────────

def fig_loss_curves_by_category(df: pd.DataFrame, repo_root: Path, out_path: Path,
                                 max_epoch: int = 50) -> None:
    """Mean training loss trajectory per fault category + correct baseline."""
    categories = list(df[df["subset"] == "buggy"]["fault_category"].unique())
    categories_sorted = sorted(categories, key=lambda c: -len(df[
        (df["subset"] == "buggy") & (df["fault_category"] == c)
    ]))
    groups = {c: df[(df["subset"] == "buggy") & (df["fault_category"] == c)] for c in categories_sorted}
    groups["correct"] = df[df["subset"] == "correct"]

    # Accumulate per-epoch sums
    sums: Dict[str, np.ndarray] = {}
    sumsq: Dict[str, np.ndarray] = {}
    cnts: Dict[str, np.ndarray] = {}
    pool: Dict[str, List[float]] = {}

    for key in list(categories_sorted) + ["correct"]:
        sums[key] = np.zeros(max_epoch)
        sumsq[key] = np.zeros(max_epoch)
        cnts[key] = np.zeros(max_epoch, dtype=int)
        pool[key] = []

    for key, sub_df in groups.items():
        for row in sub_df.itertuples(index=False):
            p = resolve(repo_root, str(row.csv_path))
            if not p.exists():
                continue
            try:
                log = pd.read_csv(p, usecols=["train_loss"])
            except Exception:
                continue
            vals = pd.to_numeric(log["train_loss"], errors="coerce").to_numpy(dtype=float)[:max_epoch]
            finite = vals[np.isfinite(vals)]
            pool[key].extend(finite.tolist())

    # winsorise per group
    clip: Dict[str, float] = {}
    for key in pool:
        if pool[key]:
            clip[key] = float(np.quantile(pool[key], 0.95))
        else:
            clip[key] = np.inf

    for key, sub_df in groups.items():
        cap = clip[key]
        for row in sub_df.itertuples(index=False):
            p = resolve(repo_root, str(row.csv_path))
            if not p.exists():
                continue
            try:
                log = pd.read_csv(p, usecols=["train_loss"])
            except Exception:
                continue
            vals = pd.to_numeric(log["train_loss"], errors="coerce").to_numpy(dtype=float)[:max_epoch]
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    v = min(v, cap)
                    sums[key][i] += v
                    sumsq[key][i] += v * v
                    cnts[key][i] += 1

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(max_epoch)
    all_keys = categories_sorted + ["correct"]

    for key in all_keys:
        c = cnts[key]
        valid = c > 0
        mean = np.full(max_epoch, np.nan)
        mean[valid] = sums[key][valid] / c[valid]

        ci = np.full(max_epoch, np.nan)
        v2 = c > 1
        var = np.full(max_epoch, np.nan)
        var[v2] = np.maximum(
            (sumsq[key][v2] - sums[key][v2]**2 / c[v2]) / (c[v2] - 1), 0
        )
        ci[v2] = 1.96 * np.sqrt(var[v2] / c[v2])

        color = CATEGORY_COLORS.get(key, "#888888")
        ls = "--" if key == "correct" else "-"
        lw = 2.2 if key == "correct" else 1.4
        n_logs = int(cnts[key][cnts[key] > 0][0]) if cnts[key].any() else 0
        label = f"{key} (n={len(groups[key]):,})"
        ax.plot(x, mean, color=color, linestyle=ls, linewidth=lw, label=label)
        mask = ~np.isnan(mean) & ~np.isnan(ci)
        if mask.any():
            ax.fill_between(x[mask], (mean - ci)[mask], (mean + ci)[mask],
                            color=color, alpha=0.10, linewidth=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean training loss (winsorized)")
    ax.set_xlim(0, max_epoch - 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    save(fig, out_path)


# ── Figure C: improved heatmap (row-normalised) ───────────────────────────────

def fig_heatmap_improved(df: pd.DataFrame, seed_meta: Optional[pd.DataFrame],
                          out_path: Path) -> None:
    if seed_meta is None:
        return
    so_to_arch = seed_meta.set_index("so_id")["architecture"].to_dict()
    buggy = df[df["subset"] == "buggy"].copy()
    buggy["architecture"] = buggy["so_id"].map(so_to_arch).fillna("UNKNOWN")
    buggy = buggy[buggy["architecture"].isin(ARCH_ORDER)]

    pivot = buggy.pivot_table(
        index="fault_category", columns="architecture",
        values="csv_path", aggfunc="count", fill_value=0
    )
    cols = [c for c in ARCH_ORDER if c in pivot.columns]
    rows = buggy["fault_category"].value_counts().index.tolist()
    rows = [r for r in rows if r in pivot.index]
    pivot = pivot.loc[rows, cols]

    # Row-normalise to percentages
    row_totals = pivot.sum(axis=1).replace(0, 1)
    pct = (pivot.div(row_totals, axis=0) * 100).round(1)

    fig_h = max(3.0, 0.55 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(5.5, fig_h))
    im = ax.imshow(pct.values, aspect="auto", interpolation="nearest",
                   cmap="Blues", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("% of fault-category logs", rotation=90, va="bottom", fontsize=8)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Fault category")

    for i in range(len(rows)):
        for j in range(len(cols)):
            v = pct.iat[i, j]
            txt_color = "white" if v > 60 else "black"
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=txt_color)

    # clean minor grid
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.grid(which="major", visible=False)

    save(fig, out_path)


# ── Figure D: per-fault detection F1 ─────────────────────────────────────────

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


def build_feature_matrix(df: pd.DataFrame, repo_root: Path):
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


def fig_per_fault_f1(df: pd.DataFrame, repo_root: Path, out_path: Path,
                     n_splits: int = 5) -> None:
    """RF binary detection F1 evaluated separately for each fault category."""
    print("  Building feature matrix for per-fault F1 (this takes a minute)…")
    X, y_bin, y_cat, groups, feat_names = build_feature_matrix(df, repo_root)

    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=7)
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=-1,
                                       class_weight="balanced_subsample")),
    ])

    # Collect predictions across folds
    y_pred_all = np.full(len(y_bin), -1, dtype=int)
    for tr, te in splitter.split(X, y_bin, groups=groups):
        pipe.fit(X[tr], y_bin[tr])
        y_pred_all[te] = pipe.predict(X[te])

    # Per-fault F1: for each category, subset to (that category | correct)
    categories = sorted(df[df["subset"] == "buggy"]["fault_category"].unique())
    f1s, counts = [], []
    for cat in categories:
        mask = (y_cat == cat) | (y_bin == 0)
        if mask.sum() < 10:
            f1s.append(0.0)
            counts.append(0)
            continue
        y_t = y_bin[mask]
        y_p = y_pred_all[mask]
        f1s.append(f1_score(y_t, y_p, pos_label=1, zero_division=0))
        counts.append(int(mask.sum()))

    # Plot
    colors = [CATEGORY_COLORS.get(c, "#888888") for c in categories]
    y_pos = np.arange(len(categories))
    labels = [f"{c} (n={cnt:,})" for c, cnt in zip(categories, counts)]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.barh(y_pos, f1s, color=colors, edgecolor="#2B2B2B", linewidth=0.5, height=0.65)
    for bar, v in zip(bars, f1s):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", ha="left", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Binary detection F1 (RF, 5-fold group-aware CV)")
    ax.set_xlim(0, 1.08)
    ax.axvline(0.5, color="#888888", linewidth=0.8, linestyle=":")
    ax.invert_yaxis()
    save(fig, out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-epoch", type=int, default=50)
    p.add_argument("--n-splits", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()

    df, repo_root = load_manifest(args.data_root)
    seed_meta = load_seed_metadata(args.data_root)

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Generating loss curves by category…")
    fig_loss_curves_by_category(df, repo_root,
                                 fig_dir / "loss_curves_by_category.pdf",
                                 max_epoch=args.max_epoch)

    print("Generating improved heatmap…")
    fig_heatmap_improved(df, seed_meta,
                          fig_dir / "cat_arch_heatmap_improved.pdf")

    print("Generating per-fault detection F1…")
    fig_per_fault_f1(df, repo_root,
                      fig_dir / "per_fault_f1.pdf",
                      n_splits=args.n_splits)

    print(f"Done. Figures written to {fig_dir}/")


if __name__ == "__main__":
    main()
