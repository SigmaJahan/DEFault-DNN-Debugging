#!/usr/bin/env python3
"""Shared feature builders for Deep4ge round-2 baselines.

Two representations per run:
  - snapshot: the 30 numeric values at the last epoch (round-1 baseline).
  - trajectory: per-feature summary statistics over all epochs of the run.

Also a manifest loader and an architecture map.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


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


def load_manifest(data_root: Path) -> Tuple[pd.DataFrame, Path]:
    mp = find_manifest(data_root)
    df = pd.read_csv(mp)
    df["subset"] = df["subset"].astype(str).str.lower()
    df["fault_category"] = df["fault_category"].astype(str).str.lower()
    df["so_id"] = df["so_id"].astype(str)
    df["csv_path"] = df["csv_path"].astype(str)
    return df, find_repo_root(mp, df)


def load_architecture_map(data_root: Path) -> dict:
    """Return {so_id: architecture} from seed_metadata.csv."""
    for c in [data_root / "data" / "seed_programs" / "seed_metadata.csv",
              data_root / "seed_programs" / "seed_metadata.csv"]:
        if c.exists():
            meta = pd.read_csv(c)
            meta["so_id"] = meta["so_id"].astype(str)
            return dict(zip(meta["so_id"], meta["architecture"].astype(str).str.upper()))
    raise FileNotFoundError("seed_metadata.csv not found")


def _to_float(v: str) -> float:
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


def _read_all_rows(path: Path) -> Tuple[List[str], np.ndarray]:
    """Return (header, array of shape [n_epochs, n_cols]) for one run CSV."""
    rows: List[List[float]] = []
    header: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        rdr = csv.reader(f)
        header = next(rdr, []) or []
        for r in rdr:
            if r and any(c.strip() for c in r):
                rows.append([_to_float(c) for c in r])
    if not rows:
        return header, np.empty((0, len(header)))
    width = len(header)
    fixed = [(row + [float("nan")] * width)[:width] for row in rows]
    return header, np.asarray(fixed, dtype=float)


def _summary_stats(col: np.ndarray) -> List[float]:
    """Six summary stats for one feature's trajectory: mean, std, min, max,
    last value, slope (linear-fit gradient over epoch index)."""
    valid = col[~np.isnan(col)]
    if valid.size == 0:
        return [np.nan] * 6
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    last = float(valid[-1])
    if valid.size >= 2:
        x = np.arange(valid.size, dtype=float)
        slope = float(np.polyfit(x, valid, 1)[0])
    else:
        slope = 0.0
    return [mean, std, vmin, vmax, last, slope]


def build_features(
    df: pd.DataFrame,
    repo_root: Path,
    representation: str,
    max_logs: Optional[int] = None,
):
    """Return X, y_bin, y_cat, groups(so_id), feature_names.

    representation:
      'snapshot'   -> last-epoch values, 30 features (round-1 baseline).
      'trajectory' -> 6 summary stats per feature, 180 features.
    """
    rows = df
    if max_logs is not None:
        rows = rows.sample(n=min(max_logs, len(rows)), random_state=7)
    rows = rows.reset_index(drop=True)

    first_path = repo_root / str(rows.iloc[0]["csv_path"])
    header, _ = _read_all_rows(first_path)
    feat_cols = [i for i, c in enumerate(header) if c.lower() != "epoch"]
    base_names = [header[i] for i in feat_cols]

    if representation == "snapshot":
        feature_names = list(base_names)
    elif representation == "trajectory":
        stat_suffix = ["mean", "std", "min", "max", "last", "slope"]
        feature_names = [f"{n}_{s}" for n in base_names for s in stat_suffix]
    else:
        raise ValueError(f"unknown representation: {representation}")

    X = np.full((len(rows), len(feature_names)), np.nan)
    y_bin = np.empty(len(rows), dtype=int)
    y_cat = np.empty(len(rows), dtype=object)
    groups = np.empty(len(rows), dtype=object)

    for i, r in enumerate(rows.itertuples(index=False)):
        path = repo_root / str(r.csv_path)
        _, arr = _read_all_rows(path)
        if arr.shape[0] == 0:
            pass  # leaves NaN row; imputer handles it
        elif representation == "snapshot":
            last = arr[-1]
            X[i] = [last[c] for c in feat_cols]
        else:  # trajectory
            vec: List[float] = []
            for c in feat_cols:
                vec.extend(_summary_stats(arr[:, c]))
            X[i] = vec
        y_bin[i] = 1 if str(r.subset).lower() == "buggy" else 0
        y_cat[i] = str(r.fault_category).lower()
        groups[i] = str(r.so_id)

    X[np.isinf(X)] = np.nan
    return X, y_bin, y_cat, groups, feature_names
