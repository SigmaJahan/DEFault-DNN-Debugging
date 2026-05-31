# Deep4ge: A Dataset for Fault Detection and Diagnosis in Deep Neural Networks

> **Deep4ge** (pronounced "Deep Forge") - the **4** stands for the four pillars of this dataset: **F**aults, **F**eatures, DNN architecture families (**FNN/CNN/RNN**), and **F**ault categories.

A large dataset of **14,227 training logs** from mutated DNN programs (**9,845 faulty + 4,382 correct baselines**) with **31 columns logged per epoch** (epoch index, four train/val loss and accuracy metrics, and 26 dynamic indicators), designed for research on fault detection and diagnosis in deep learning systems.

## Dataset Summary

| Property | Value |
|----------|-------|
| Seed programs | 60 adapted DNN seed programs derived from StackOverflow posts (59 produce training logs) |
| Architectures (seed programs) | FNN (16), CNN (14), RNN (30) |
| Mutation operators | 27 used across 7 fault categories (**34 implemented** in the framework) |
| Faulty training logs | 9,845 |
| Correct (baseline) training logs | 4,382 |
| Total training logs | 14,227 |
| Dynamic features per epoch | 31 columns, including 11 novel/extended fields documented in the data dictionary |
| Unique StackOverflow IDs (with logs) | 59 |
| Total epoch records | 719,560 |
| Median epochs per log | 50 |
| Archive DOI | [10.5281/zenodo.20337241](https://doi.org/10.5281/zenodo.20337241) |

## Fault Categories

| Category | Operators | Count | Description |
|----------|-----------|-------|-------------|
| Hyperparameter | HBS, HLR, HNE, HDB | 2,571 | Batch size, learning rate, epochs, batching |
| Layer | LKS, LCF, LCP, LCS, LCD, LCN, LAD, LRM, LCT, LCO | 1,335 | Architecture mutations (new in Deep4ge; includes CNN-specific layer faults) |
| Loss | FLC | 2,148 | Loss function selection |
| Activation | ACH, ARM, AAL | 312 | Activation function mutations |
| Optimization | OCH, OCG | 1,240 | Optimizer and gradient clipping |
| Weight | WCI, WAB, WRB | 1,985 | Weight initialization and bias |
| Regularization | RCD, RAW, RCW, RRW | 254 | Dropout and regularization |

The seven categories and the per-category sums above are computed directly from `data/manifest.csv` and sum to 9,845 faulty logs.

## Relationship to the DEFault Paper

Deep4ge is the released, cleaned artifact behind the DEFault study (ICSE 2025, "Improved Detection and Diagnosis of Faults in Deep Neural Networks Using Hierarchical and Explainable Classification"). The paper reports the numbers measured at experiment time. The numbers here describe the public release after deduplication, filename normalization, and removal of crashed (header-only) runs. The two differ in the ways below, and the release values are the source of truth for anyone using these files.

| Quantity | Paper (experiment-time) | Deep4ge release (this repo) |
|----------|-------------------------|-----------------------------|
| Total training logs / mutants | 14,652 | 14,227 |
| Faulty logs | 9,855 | 9,845 |
| Correct (baseline) logs | 4,797 | 4,382 |
| Seed DNN programs | 60 repaired from 89 collected | 60 on disk (59 with logs) |
| New layer operators added to DeepCrime | 10 | 10 (LKS, LCF, LCP, LCS, LCD, LCN, LAD, LRM, LCT, LCO) |
| Dynamic features | 23 (Table III), 6 novel | 26 dynamic columns logged per epoch (31 total with epoch index and four train/val metrics), 11 novel/extended fields marked in the data dictionary |

The paper counts 23 dynamic features as the set fed to the classifiers. The callback in this repo logs a wider superset of 26 dynamic columns (plus the epoch index and four train/val loss and accuracy metrics, for 31 columns total). The extra logged columns are intermediate signals that the downstream tool either drops or merges before training. See [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for the full column schema and the novel-feature marks. The 954 crashed runs removed during cleanup account for most of the gap between the paper and release counts (see [docs/PROVENANCE.md](docs/PROVENANCE.md)).

## Quick Start

### Loading the Dataset (No TensorFlow Required)

```bash
pip install pandas
```

```python
import pandas as pd

# Load the manifest: the single entry point for all data access
manifest = pd.read_csv("data/manifest.csv")
print(manifest.shape)  # (14227, 8)

# Filter faulty logs
faulty = manifest[manifest["is_faulty"] == True]
print(f"Faulty: {len(faulty)}, Categories: {faulty['fault_category'].nunique()}")

# Load a single training log (31 columns x N epochs)
sample = pd.read_csv(faulty.iloc[0]["csv_path"])
print(f"Columns: {len(sample.columns)}, Epochs: {len(sample)}")
print(sample.columns.tolist())
```

### Reproducing a Mutation (Requires TensorFlow)

```bash
# Tested with Python 3.11 and TensorFlow 2.15.1.
pip install -r requirements.txt

# List all implemented mutation operators (34 in the framework)
python3 -m src.mutate --list-operators

# Apply a mutation and train (generates a 31-column CSV)
python3 scripts/demo_replication.py --operator HBS

# Dry-run: see the mutated source without training
python3 scripts/demo_replication.py --operator FLC --dry-run
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for more examples.

## Repository Structure

```
deep4ge/
├── data/
│   ├── manifest.csv              # Master index linking logs to metadata
│   ├── seed_programs/            # 60 adapted DNN seed programs from StackOverflow
│   │   ├── fnn/                  # 16 feed-forward (dense) programs
│   │   ├── cnn/                  # 14 convolutional programs (one has no logs)
│   │   ├── rnn/                  # 30 recurrent programs
│   │   ├── seed_metadata.csv     # so_id to architecture mapping + log counts
│   │   └── ATTRIBUTION.csv       # per-seed StackOverflow attribution/license metadata
│   ├── training_logs/            # Per-epoch training metrics (31 columns)
│   │   ├── buggy/                # 9,845 faulty model training logs
│   │   └── correct/              # 4,382 correct model training logs
├── analysis/                     # Reproducible stats, figures, and baselines
├── scripts/                      # Validation, manifest generation, and replication demos
├── src/                          # Mutation operators + dataset generation pipeline
├── docs/                         # Data dictionary, provenance, operator catalog, etc.
├── requirements.txt              # TensorFlow replication environment
└── CITATION.cff                  # Citation metadata
```

## Documentation

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md): how to load the dataset and reproduce mutations
- [docs/TENSORFLOW_RETRAINING_CHECK.md](docs/TENSORFLOW_RETRAINING_CHECK.md): TensorFlow-backed mutation/training smoke test
- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md): log CSV schema (31 columns)
- [docs/PROVENANCE.md](docs/PROVENANCE.md): dataset sources, pipeline, and design decisions
- [docs/SEED_ATTRIBUTION.md](docs/SEED_ATTRIBUTION.md): per-seed StackOverflow attribution and license boundaries
- [docs/OPERATOR_CATALOG.md](docs/OPERATOR_CATALOG.md): mutation operators (used + implemented)

## License

- **Framework/code**: [MIT](LICENSE)
- **Generated dataset files**: [CC BY 4.0](data/LICENSE_DATASET_CC_BY_4.0.md)
- **StackOverflow seed programs**: adapted source-post content; see [docs/SEED_ATTRIBUTION.md](docs/SEED_ATTRIBUTION.md) and [data/seed_programs/ATTRIBUTION.csv](data/seed_programs/ATTRIBUTION.csv)

## Citation

Use [CITATION.cff](CITATION.cff) for citation metadata. The archived release is
available at [10.5281/zenodo.20337241](https://doi.org/10.5281/zenodo.20337241).
