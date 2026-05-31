# Deep4ge analysis scripts

This folder contains reproducible analysis scripts used to generate descriptive statistics, tables, figures, and baseline results for Deep4ge.

The scripts are robust to two common layouts:

(A) Pass the *repo root* (recommended):
- `./data/manifest.csv`
- `./data/training_logs/buggy/*.csv`
- `./data/training_logs/correct/*.csv`

(B) Pass the *data directory*:
- `./manifest.csv`
- `./training_logs/...`

Typical usage (from repo root):

1) Create a Python environment and install deps:

```
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib scikit-learn
```

2) Compute stats + tables:

```
python3 analysis/compute_stats.py --data-root . --out-dir output/analysis_artifacts
```

3) Generate paper figures:

```
python3 analysis/generate_figures.py --data-root . --out-dir output/analysis_artifacts
python3 analysis/generate_new_figures.py --data-root . --out-dir output/analysis_artifacts
```

4) Run the detection and diagnosis baselines (group-aware CV by `so_id`).
The scripts compare a final-epoch feature representation with a
trajectory-summary representation, report class-balanced metrics with
bootstrap confidence intervals, and add a per-category confusion analysis
and a cross-architecture transfer study.

```
# Final-epoch baselines plus naive floors and class-balanced metrics
python3 analysis/baseline_models.py --data-root . --out-dir output/analysis_artifacts
python3 analysis/generate_feature_importance.py --data-root . --out-dir output/analysis_artifacts

# Final-epoch vs. trajectory feature representations
PYTHONPATH=analysis python3 analysis/baseline_representations.py --data-root . --out-dir output/analysis_artifacts

# 95% bootstrap confidence intervals (reuses saved per-fold predictions)
python3 analysis/bootstrap_cis.py --pred-dir output/analysis_artifacts/predictions_r2 --out-dir output/analysis_artifacts

# Per-category diagnosis confusion matrix
PYTHONPATH=analysis python3 analysis/diagnosis_confusion.py --data-root . --out-dir output/analysis_artifacts

# Cross-architecture generalization (train on two families, test on the third)
PYTHONPATH=analysis python3 analysis/cross_architecture.py --data-root . --out-dir output/analysis_artifacts

# Detection from the first k epochs of a run
python3 analysis/early_detection_curve.py --data-root . --out-dir output/analysis_artifacts
```

`feature_builders.py` is a shared module imported by the
representation, confusion, and cross-architecture scripts. The three
scripts that import it are run with `PYTHONPATH=analysis` so the module
resolves from the repository root.

Outputs are written under `output/analysis_artifacts/`:
- `stats.json`, `baseline_results.json`, `representation_results.json`
- `bootstrap_results.json`, `cross_architecture.json`, `diagnosis_confusion.json`
- `early_detection.json`
- `tables/*.tex` and `figures/*.pdf`

For quick smoke tests, the baseline scripts support `--max-logs`.
