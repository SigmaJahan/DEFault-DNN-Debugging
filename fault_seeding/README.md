# Fault Seeding (Paper Section IV-D)

This stage injects faults into correct DNN programs to build the training data for DEFault. It holds three things: the original paper-time mutation scripts, the unified Deep4ge generation framework, and the dataset documentation.

## Layout

| Path | What it is |
|------|------------|
| [Part 1-DC/](Part%201-DC/) | DeepCrime operators as used in the paper, plus `mutation_score.py` for the killability check |
| [Part 2-EFI/](Part%202-EFI/) | Extended Fault Injection: the 10 new layer operators for CNN and RNN (`convolution_operators.py`, `recurrent_operators.py`) |
| [deep4ge_framework/](deep4ge_framework/) | The unified mutation framework behind the Deep4ge dataset release: 34 operators across 10 fault families, the logging callback, and the operator registry |
| [scripts/](scripts/) | Manifest building, dataset validation, statistics, and a replication demo |
| [docs/](docs/) | Dataset documentation: operator catalog, data dictionary, provenance, seed attribution |
| [Fault_Seeding_Parameters.md](Fault_Seeding_Parameters.md) | Mutation parameter bounds used during seeding |

## Operators

The paper supports seven fault categories for injection (Table I): Hyperparameter, Layer, Loss, Activation, Optimization, Weights, Regularization. The 10 layer operators are new to this work and cover faults that DeepCrime did not, including CNN-specific (kernel size, filter count, pooling, strides, padding) and RNN-specific (layer-type swap, output shape) mutations. See [docs/OPERATOR_CATALOG.md](docs/OPERATOR_CATALOG.md) for the full list.

## The Dataset

The generated training logs are published as the Deep4ge dataset and archived on Zenodo, not committed here.

> Deep4ge dataset. DOI [10.5281/zenodo.20337241](https://doi.org/10.5281/zenodo.20337241)

To regenerate a sample with the framework (needs TensorFlow):

```bash
pip install -r deep4ge_framework/requirements.txt
python -m deep4ge_framework.mutate --list-operators
python scripts/demo_replication.py --operator HBS --dry-run
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) and [docs/PROVENANCE.md](docs/PROVENANCE.md) for details on how the dataset was built. The released counts and how they relate to the paper are in [docs/DEEP4GE_DATASET_README.md](docs/DEEP4GE_DATASET_README.md).
