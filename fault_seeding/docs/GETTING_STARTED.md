# Getting Started with the Deep4ge Dataset

## Installation

```bash
pip install pandas numpy
```

No other dependencies are required to use the dataset. TensorFlow is only needed if you want to run mutations or train models.

## Loading the Dataset

### 1. Load the Manifest

The manifest is the entry point for all data access:

```python
import pandas as pd

manifest = pd.read_csv("data/manifest.csv")
print(manifest.shape)          # (14227, 8)
print(manifest.columns.tolist())
# ['filename', 'subset', 'so_id', 'fault_category', 'is_faulty', 'run_number', 'num_epochs', 'csv_path']
```

### 2. Filter by Subset

```python
# Faulty training logs
buggy = manifest[manifest["subset"] == "buggy"]
print(f"Faulty programs: {len(buggy)}")   # 9,845

# Correct (baseline) training logs
correct = manifest[manifest["subset"] == "correct"]
print(f"Correct programs: {len(correct)}")  # 4,382
```

### 3. Filter by Fault Category

```python
# All Layer faults
layer_faults = manifest[manifest["fault_category"] == "Layer"]
print(f"Layer faults: {len(layer_faults)}")  # 1,335

# All categories
print(manifest[manifest["is_faulty"] == True]["fault_category"].value_counts())
```

### 4. Load a Training Log

```python
# Load a single training log CSV
row = manifest.iloc[0]
log = pd.read_csv(row["csv_path"])
print(f"Epochs: {len(log)}, Columns: {len(log.columns)}")
print(log.columns.tolist())
```

### 5. Batch Loading

```python
# Load all training logs for a specific SO question
so_id = "31556268"
so_files = manifest[manifest["so_id"] == so_id]
print(f"Files for SO#{so_id}: {len(so_files)}")

# Load all logs for this model
logs = {}
for _, row in so_files.iterrows():
    logs[row["filename"]] = pd.read_csv(row["csv_path"])
```

## Common Analysis Patterns

### Binary Classification: Faulty vs Correct

```python
# Build a simple feature matrix from final-epoch metrics
import numpy as np

features = []
labels = []

for _, row in manifest.iterrows():
    log = pd.read_csv(row["csv_path"])
    if len(log) == 0:
        continue
    # Use final epoch features
    final = log.iloc[-1]
    features.append(final.values)
    labels.append(1 if row["is_faulty"] else 0)

X = np.array(features, dtype=float)
y = np.array(labels)
print(f"Feature matrix: {X.shape}, Labels: {y.shape}")
```

### Fault Category Classification

```python
# Multi-class: predict which fault category
buggy = manifest[manifest["is_faulty"] == True]
category_labels = buggy["fault_category"].values
print(f"Categories: {np.unique(category_labels)}")
```

### Time-Series Analysis

```python
# Analyze training dynamics over epochs
log = pd.read_csv(manifest.iloc[0]["csv_path"])

# Plot loss curves
import matplotlib.pyplot as plt
plt.plot(log["epoch"], log["train_loss"], label="Train Loss")
plt.plot(log["epoch"], log["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

## Reproducing Mutations (Requires TensorFlow)

```bash
pip install -r requirements.txt

# List all 34 operators
python3 -m src.mutate --list-operators

# End-to-end demo: mutate → train → collect features → validate CSV
python3 scripts/demo_replication.py --operator HBS

# Dry-run: see the mutated source code without training
python3 scripts/demo_replication.py --operator FLC --dry-run

# Apply a single mutation operator to a seed program (CLI)
python3 -m src.mutate data/seed_programs/fnn/FNN_31556268_correct.py \
    --operator HBS --iterations 5

# (CNN example)
python3 -m src.mutate data/seed_programs/cnn/CNN_37624102_correct.py \
    --operator LCF --iterations 5

# (RNN example)
python3 -m src.mutate data/seed_programs/rnn/rnn_51971180_correct.py \
    --operator OCH --iterations 5
```

The `demo_replication.py` script verifies that the output CSV matches the same 31-column schema as all 14,227 files in the dataset.

For a reviewer-oriented TensorFlow smoke test, run:

```bash
python3 scripts/tensorflow_smoke_test.py
```

See [TENSORFLOW_RETRAINING_CHECK.md](TENSORFLOW_RETRAINING_CHECK.md) for the
tested TensorFlow version, environment setup, and expected output.

## Dataset Integrity and Regeneration Utilities

```bash
# Validate the released dataset (schema, naming, manifest/file consistency)
python3 scripts/validate_dataset.py

# Rebuild manifest.csv from data/training_logs/*
python3 scripts/build_manifest.py
```

Expected validation result for the current release: `PASS` with 14,227 files and 0 issues.

## File Naming Convention

### Training Logs

- **Buggy**: `{so_id}_{fault_category}_{run:04d}.csv`
  - Example: `31556268_Hyperparameter_0001.csv`
- **Correct**: `{so_id}_correct_{run:04d}.csv`
  - Example: `31556268_correct_0001.csv`

### Seed Programs

- **FNN**: `FNN_{so_id}_correct.py`
- **RNN**: `rnn_{so_id}_correct.py`

## CSV Column Reference

All training log CSVs contain 31 columns. See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for the complete column documentation.

Key columns for quick reference:
- `train_loss`, `val_loss`: Training and validation loss
- `train_acc`, `val_acc`: Training and validation accuracy
- `gradient_vanish`, `gradient_explode`: Binary gradient health indicators
- `dying_relu`: Dead neuron indicator
- `mean_gradient`, `gradient_std`: Gradient statistics
- `cpu_utilization`, `memory_usage`: Hardware metrics

## Operator Reference

All 34 mutation operators are documented in [OPERATOR_CATALOG.md](OPERATOR_CATALOG.md).
