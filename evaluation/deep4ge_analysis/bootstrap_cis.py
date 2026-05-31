#!/usr/bin/env python3
"""Bootstrap 95% CIs for the metrics already produced by baseline_models.py.

Reads predictions/*.npz and reports a 95% CI for every metric. Writes
output/analysis_artifacts/bootstrap_results.json.

Example:
  python3 analysis/bootstrap_cis.py --pred-dir output/analysis_artifacts/predictions \
      --out-dir output/analysis_artifacts --n-boot 1000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
)


METRIC_FUNCS_BIN = {
    "f1": lambda y, p, s: f1_score(y, p, zero_division=0),
    "balanced_accuracy": lambda y, p, s: balanced_accuracy_score(y, p),
    "mcc": lambda y, p, s: matthews_corrcoef(y, p),
    "auroc": lambda y, p, s: roc_auc_score(y, s) if s is not None and len(s) else float("nan"),
    "auprc": lambda y, p, s: average_precision_score(y, s) if s is not None and len(s) else float("nan"),
}
METRIC_FUNCS_MULTI = {
    "macro_f1": lambda y, p, s: f1_score(y, p, average="macro", zero_division=0),
    "balanced_accuracy": lambda y, p, s: balanced_accuracy_score(y, p),
    "mcc": lambda y, p, s: matthews_corrcoef(y, p),
}


def bootstrap_ci(y, p, s, metric_fn, n_boot=1000, alpha=0.05, seed=7):
    rng = np.random.default_rng(seed)
    n = len(y)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        try:
            s_b = s[idx] if (s is not None and len(s)) else None
            estimates[b] = metric_fn(y[idx], p[idx], s_b)
        except Exception:
            estimates[b] = float("nan")
    estimates = estimates[~np.isnan(estimates)]
    if len(estimates) == 0:
        return float("nan"), float("nan"), float("nan")
    lo = float(np.quantile(estimates, alpha / 2))
    hi = float(np.quantile(estimates, 1 - alpha / 2))
    try:
        point = float(metric_fn(y, p, s if (s is not None and len(s)) else None))
    except Exception:
        point = float("nan")
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for npz_path in sorted(args.pred_dir.glob("*.npz")):
        # allow_pickle is required because diagnosis labels are string arrays.
        # These NPZ files are produced by baseline_models.py in this repo.
        d = np.load(npz_path, allow_pickle=True)
        y, p = d["y_true"], d["y_pred"]
        s = d["y_score"] if "y_score" in d.files and d["y_score"].size > 0 else None
        is_multi = len(np.unique(y)) > 2
        funcs = METRIC_FUNCS_MULTI if is_multi else METRIC_FUNCS_BIN

        row = {"file": npz_path.stem}
        for name, fn in funcs.items():
            point, lo, hi = bootstrap_ci(y, p, s, fn, n_boot=args.n_boot)
            row[name] = {"point": point, "lo": lo, "hi": hi}
        summary.append(row)
        print(
            f"{npz_path.stem}: "
            + ", ".join(
                f"{n}={row[n]['point']:.3f} [{row[n]['lo']:.3f},{row[n]['hi']:.3f}]"
                for n in funcs
            )
        )

    out_path = args.out_dir / "bootstrap_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
