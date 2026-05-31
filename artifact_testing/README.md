# Artifact Testing: Reproduce the Case Study

This package reproduces DEFault's diagnosis of the PixelCNN case study (paper Section VI) on a laptop, in seconds, without retraining anything. It loads the pre-trained classifiers and feeds them dynamic and static features already extracted from the PixelCNN run, so you skip the slow training step. The PixelCNN program being diagnosed lives in [../case_study/](../case_study/).

## What's Here

| Folder | Contents |
|--------|----------|
| `evaluation_scripts/` | `testForCaseStudy_FD_FC.py` (fault detection + categorization), `testForCaseStudy_RCA.py` (root cause analysis) |
| `models/` | The 8 pre-trained Random Forest classifiers: detection plus 7 fault categories |
| `data/` | Pre-extracted features: `pixelcnn_buggy.csv` (dynamic), `static_features_df_test_file.csv` (static test sample) |
| `config/` | `config.ini` with the per-classifier decision thresholds |

## Environment

The classifiers were trained with scikit-learn 1.5.0. Use a matching environment (tested with Python 3.11).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install "scikit-learn==1.5.0" pandas numpy joblib   # enough for detection + categorization
pip install shap xgboost matplotlib                      # also needed for root cause analysis
```

## Step 1: Detection and Categorization (self-contained)

```bash
cd artifact_testing
python evaluation_scripts/testForCaseStudy_FD_FC.py
```

The script loads the 8 models, reads `data/pixelcnn_buggy.csv`, and prints a verdict per classifier. Expected result: the program is detected as faulty, with Loss, Hyperparameter, and Layer flagged among the categories. The three fault types match the paper's PixelCNN case study (wrong loss function, too few epochs, layer misconfigurations).

```
=== Detailed Results ===
         File Name  Avg. Probability (%) Bug Detected?   Bug Category
pixelcnn_buggy.csv                  74.1          Yes       Detection
pixelcnn_buggy.csv                  82.0          Yes  Hyperparameter
pixelcnn_buggy.csv                  63.5          Yes           Layer
pixelcnn_buggy.csv                  58.1          Yes            Loss
...
```

Exact probabilities can vary slightly across library versions, but the flagged categories are stable.

## Step 2: Root Cause Analysis (needs the dataset)

The root cause script trains a static-feature explainer, so it needs the labeled static-feature table from the Deep4ge dataset. The table is archived on Zenodo, not committed here.

> Deep4ge dataset. DOI [10.5281/zenodo.20337241](https://doi.org/10.5281/zenodo.20337241)

Download `static_features_df.csv` from the archive, then point the script at it:

```bash
export DEFAULT_STATIC_FEATURES_CSV=/path/to/static_features_df.csv
python evaluation_scripts/testForCaseStudy_RCA.py
```

The script trains an XGBoost model on the static features, runs SHAP on the PixelCNN test sample (`data/static_features_df_test_file.csv`), and prints the ranked root-cause features with readable insights. Without the table, the script stops with a message telling you where to get it.

## Notes

- Step 1 runs out of the box on the bundled data. Step 2 needs the one Zenodo download plus `shap`, `xgboost`, and `matplotlib`.
- PixelCNN source: [tf2-published-models](https://github.com/sarus-tech/tf2-published-models).
