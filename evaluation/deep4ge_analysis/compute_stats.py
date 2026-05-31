#!/usr/bin/env python3
"""Compute descriptive statistics for the Deep4ge dataset.

This script reads the dataset manifest (and optional seed metadata) to compute
dataset-level descriptive statistics that are commonly expected in SE dataset
showcase papers.

The script is robust to two common layouts:

(A) You pass the *repo root* (recommended):
    repo_root/
      data/manifest.csv
      data/training_logs/...
(B) You pass the *data directory*:
    data/
      manifest.csv
      training_logs/...

Outputs (under <out_dir>/):
- stats.json
- tables/dataset_summary.tex
- tables/category_counts_buggy.tex
- tables/architecture_seeds_and_logs.tex
Optionally (if --scan-logs):
- stats.json includes per-feature summary stats (mean/std/min/max/nan_count)

Example:
  # From repo root:
  python3 analysis/compute_stats.py --data-root . --out-dir output/analysis_artifacts

  # Also works if you point directly at the data/ directory:
  python3 analysis/compute_stats.py --data-root data --out-dir output/analysis_artifacts --scan-logs
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_MANIFEST_COLUMNS = {
    "subset",  # buggy|correct
    "so_id",
    "fault_category",
    "run_number",
    "num_epochs",
    "csv_path",
}


@dataclass
class OnlineStats:
    """Online mean/variance/min/max with NaN tracking."""

    count: int = 0
    nan_count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = math.inf
    max_val: float = -math.inf

    def update_array(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        nan_mask = np.isnan(x)
        self.nan_count += int(nan_mask.sum())
        x = x[~nan_mask]
        if x.size == 0:
            return

        self.min_val = float(min(self.min_val, float(np.min(x))))
        self.max_val = float(max(self.max_val, float(np.max(x))))

        batch_count = int(x.size)
        batch_mean = float(np.mean(x))
        batch_m2 = float(np.sum((x - batch_mean) ** 2))

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + delta**2 * (self.count * batch_count / total)
        self.count = total

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return float("nan")
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        v = self.variance
        return float("nan") if math.isnan(v) else math.sqrt(v)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Deep4ge dataset statistics")
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Repo root or data directory (see module docstring).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit path to manifest.csv (otherwise inferred from --data-root).",
    )
    p.add_argument(
        "--seed-metadata",
        type=Path,
        default=None,
        help="Optional explicit seed metadata CSV (otherwise inferred from --data-root).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for stats.json and LaTeX tables",
    )
    p.add_argument(
        "--scan-logs",
        action="store_true",
        help="If set, scan all log CSVs to compute per-feature summary stats (can be slow).",
    )
    p.add_argument(
        "--max-logs",
        type=int,
        default=None,
        help="Optional cap on number of logs scanned (useful for quick iteration).",
    )
    return p.parse_args()


def infer_paths(data_root: Path, manifest_arg: Optional[Path], seed_meta_arg: Optional[Path]) -> Tuple[Path, Optional[Path]]:
    """Infer manifest and seed-metadata locations from --data-root."""
    if manifest_arg is not None:
        manifest_path = manifest_arg
    else:
        c1 = data_root / "manifest.csv"
        c2 = data_root / "data" / "manifest.csv"
        if c1.exists():
            manifest_path = c1
        elif c2.exists():
            manifest_path = c2
        else:
            raise FileNotFoundError(f"Could not find manifest.csv under {data_root} (tried {c1} and {c2}).")

    if seed_meta_arg is not None:
        seed_meta_path: Optional[Path] = seed_meta_arg
    else:
        c1 = data_root / "seed_programs" / "seed_metadata.csv"
        c2 = data_root / "data" / "seed_programs" / "seed_metadata.csv"
        if c1.exists():
            seed_meta_path = c1
        elif c2.exists():
            seed_meta_path = c2
        else:
            seed_meta_path = None  # optional

    return manifest_path, seed_meta_path


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    df["subset"] = df["subset"].astype(str).str.lower()
    df["fault_category"] = df["fault_category"].astype(str)
    df["so_id"] = df["so_id"].astype(str)
    df["csv_path"] = df["csv_path"].astype(str)

    return df


def read_seed_metadata(seed_meta_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if seed_meta_path is None or not seed_meta_path.exists():
        return None
    df = pd.read_csv(seed_meta_path)
    if "so_id" not in df.columns or "architecture" not in df.columns:
        return None
    df["so_id"] = df["so_id"].astype(str)
    df["architecture"] = df["architecture"].astype(str).str.upper()
    return df


def infer_repo_root(manifest_path: Path, manifest_df: pd.DataFrame) -> Path:
    """Infer the repo root used to resolve manifest csv_path values."""
    sample = None
    for v in manifest_df["csv_path"].head(50).tolist():
        if isinstance(v, str) and v.strip():
            sample = v
            break
    if sample is None:
        return manifest_path.parent

    candidates = [
        manifest_path.parent,           # e.g., repo_root/data
        manifest_path.parent.parent,    # e.g., repo_root
        manifest_path.parent.parent.parent,
    ]
    for c in candidates:
        try_path = c / sample
        if try_path.exists():
            return c
    # Fallback: assume paths are relative to manifest parent
    return manifest_path.parent


def compute_manifest_level_stats(df: pd.DataFrame) -> Dict[str, object]:
    total = int(len(df))
    subset_counts = df["subset"].value_counts(dropna=False).to_dict()
    unique_so_ids = int(df["so_id"].nunique())

    epochs_series = pd.to_numeric(df["num_epochs"], errors="coerce")
    total_epoch_records = int(epochs_series.fillna(0).sum())

    epochs_summary = {
        "count": int(epochs_series.notna().sum()),
        "min": int(epochs_series.min()) if epochs_series.notna().any() else None,
        "p25": float(epochs_series.quantile(0.25)) if epochs_series.notna().any() else None,
        "median": float(epochs_series.median()) if epochs_series.notna().any() else None,
        "p75": float(epochs_series.quantile(0.75)) if epochs_series.notna().any() else None,
        "max": int(epochs_series.max()) if epochs_series.notna().any() else None,
        "mean": float(epochs_series.mean()) if epochs_series.notna().any() else None,
        "std": float(epochs_series.std()) if epochs_series.notna().any() else None,
    }

    buggy_df = df[df["subset"] == "buggy"].copy()
    buggy_df["fault_category_norm"] = buggy_df["fault_category"].astype(str).str.strip()
    cat_counts_buggy = buggy_df["fault_category_norm"].value_counts(dropna=False).to_dict()

    return {
        "total_logs": total,
        "subset_counts": {str(k): int(v) for k, v in subset_counts.items()},
        "unique_so_ids": unique_so_ids,
        "total_epoch_records": total_epoch_records,
        "epochs_summary": epochs_summary,
        "fault_category_counts_buggy": {str(k): int(v) for k, v in cat_counts_buggy.items()},
    }


def resolve_log_path(repo_root: Path, csv_path_value: str) -> Path:
    p = Path(csv_path_value)
    return p if p.is_absolute() else (repo_root / p)


def scan_logs_for_feature_stats(
    df: pd.DataFrame,
    repo_root: Path,
    max_logs: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Scan log CSVs and compute per-feature stats (excluding epoch column)."""

    feature_stats: Dict[str, OnlineStats] = {}

    rows = df
    if max_logs is not None:
        rows = rows.head(max_logs)

    for i, row in enumerate(rows.itertuples(index=False), start=1):
        log_path = resolve_log_path(repo_root, str(getattr(row, "csv_path")))
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log CSV: {log_path}")

        log_df = pd.read_csv(log_path)

        for col in log_df.columns:
            if col.lower() == "epoch":
                continue
            if col not in feature_stats:
                feature_stats[col] = OnlineStats()
            arr = pd.to_numeric(log_df[col], errors="coerce").to_numpy(dtype=float)
            feature_stats[col].update_array(arr)

        if i % 1000 == 0:
            print(f"Scanned {i} logs...")

    out: Dict[str, Dict[str, float]] = {}
    for feat, st in feature_stats.items():
        out[feat] = {
            "count": int(st.count),
            "nan_count": int(st.nan_count),
            "mean": float(st.mean) if st.count > 0 else float("nan"),
            "std": float(st.std),
            "min": float(st.min_val) if st.min_val != math.inf else float("nan"),
            "max": float(st.max_val) if st.max_val != -math.inf else float("nan"),
        }
    return out


def compute_architecture_breakdown(
    manifest_df: pd.DataFrame,
    seed_meta_df: Optional[pd.DataFrame],
) -> Dict[str, object]:
    if seed_meta_df is None:
        return {}

    so_to_arch = seed_meta_df.set_index("so_id")["architecture"].to_dict()

    seed_counts = seed_meta_df["architecture"].value_counts().to_dict()

    tmp = manifest_df.copy()
    tmp["architecture"] = tmp["so_id"].map(so_to_arch).fillna("UNKNOWN")
    log_counts = tmp.groupby(["architecture", "subset"]).size().unstack(fill_value=0)
    log_counts["total"] = log_counts.sum(axis=1)

    arch_rows = []
    for arch in sorted(seed_counts.keys()):
        arch_rows.append(
            {
                "architecture": arch,
                "num_seeds": int(seed_counts.get(arch, 0)),
                "num_logs_total": int(log_counts.loc[arch, "total"]) if arch in log_counts.index else 0,
                "num_logs_buggy": int(log_counts.loc[arch, "buggy"]) if arch in log_counts.index and "buggy" in log_counts.columns else 0,
                "num_logs_correct": int(log_counts.loc[arch, "correct"]) if arch in log_counts.index and "correct" in log_counts.columns else 0,
            }
        )

    return {"seed_counts": {str(k): int(v) for k, v in seed_counts.items()}, "architecture_rows": arch_rows}


def write_latex_tables(stats: Dict[str, object], out_dir: Path) -> None:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    subset_counts: Dict[str, int] = {str(k): int(v) for k, v in (stats.get("subset_counts") or {}).items()}
    buggy = subset_counts.get("buggy", 0)
    correct = subset_counts.get("correct", 0)
    total = int(stats.get("total_logs", 0))
    unique_so = int(stats.get("unique_so_ids", 0))
    total_epoch_records = int(stats.get("total_epoch_records", 0))
    epochs_summary = stats.get("epochs_summary", {})
    median_epochs = epochs_summary.get("median", None)

    summary_tex = tables_dir / "dataset_summary.tex"
    with summary_tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by analysis/compute_stats.py\n")
        f.write("\\begin{tabular}{@{}lr@{}}\\toprule\n")
        f.write("Property & Value \\\\ \\midrule\n")
        f.write(f"Total logs & {total:,} \\\\ \n")
        f.write(f"Faulty logs (buggy) & {buggy:,} \\\\ \n")
        f.write(f"Correct baselines & {correct:,} \\\\ \n")
        f.write(f"Unique StackOverflow IDs (with logs) & {unique_so:,} \\\\ \n")
        f.write(f"Total epoch records & {total_epoch_records:,} \\\\ \n")
        if median_epochs is not None:
            f.write(f"Median epochs / log & {median_epochs:.0f} \\\\ \n")
        f.write("\\bottomrule\\end{tabular}\n")

    cat_counts: Dict[str, int] = {
        str(k): int(v) for k, v in (stats.get("fault_category_counts_buggy") or {}).items()
    }
    cat_tex = tables_dir / "category_counts_buggy.tex"
    with cat_tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by analysis/compute_stats.py\n")
        f.write("\\begin{tabular}{@{}lr@{}}\\toprule\n")
        f.write("Fault category & \\#buggy logs \\\\ \\midrule\n")
        for cat, cnt in sorted(cat_counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            f.write(f"{cat} & {cnt:,} \\\\ \n")
        f.write("\\bottomrule\\end{tabular}\n")

    arch_info = stats.get("architecture_breakdown") or {}
    arch_rows = arch_info.get("architecture_rows") or []
    if arch_rows:
        arch_tex = tables_dir / "architecture_seeds_and_logs.tex"
        with arch_tex.open("w", encoding="utf-8") as f:
            f.write("% Auto-generated by analysis/compute_stats.py\n")
            f.write("\\begin{tabular}{@{}lrrrr@{}}\\toprule\n")
            f.write("Architecture & \\#seeds & \\#logs (total) & \\#buggy & \\#correct \\\\ \\midrule\n")
            for r in arch_rows:
                f.write(
                    f"{r['architecture']} & {r['num_seeds']:,} & {r['num_logs_total']:,} & {r['num_logs_buggy']:,} & {r['num_logs_correct']:,} \\\\ \n"
                )
            f.write("\\bottomrule\\end{tabular}\n")


def main() -> None:
    args = parse_args()
    data_root: Path = args.data_root
    manifest_path, seed_meta_path = infer_paths(data_root, args.manifest, args.seed_metadata)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = read_manifest(manifest_path)
    repo_root = infer_repo_root(manifest_path, manifest_df)
    seed_meta_df = read_seed_metadata(seed_meta_path)

    stats: Dict[str, object] = compute_manifest_level_stats(manifest_df)
    stats["architecture_breakdown"] = compute_architecture_breakdown(manifest_df, seed_meta_df)

    if args.scan_logs:
        feature_stats = scan_logs_for_feature_stats(manifest_df, repo_root=repo_root, max_logs=args.max_logs)
        stats["feature_stats"] = feature_stats

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_latex_tables(stats, out_dir)

    print(f"Wrote {out_dir/'stats.json'}")
    print(f"Wrote tables under {out_dir/'tables'}")


if __name__ == "__main__":
    main()
