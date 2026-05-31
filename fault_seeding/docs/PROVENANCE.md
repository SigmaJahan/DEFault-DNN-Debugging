# Data Provenance

This document describes how the Deep4ge dataset was collected, filtered, and prepared.

## Seed Program Collection

60 real-world DNN programs were collected from StackOverflow (SO) questions. These are actual deep learning programs written by developers, covering diverse architectures and tasks. Of the 60, 59 produce training logs in the current release (one CNN seed, `so_id=50079585`, is retained as a seed but has no logs).

### Collection Criteria

1. **SO Questions**: Searched for questions tagged `[keras]`, `[tensorflow]`, or `[deep-learning]` that contained runnable DNN code
2. **Runnable**: Each program must train a model end-to-end (data loading, model definition, compilation, training)
3. **Reproducible**: Programs were corrected to run without errors and produce deterministic training outcomes
4. **Architecture coverage**: 16 Feedforward Neural Networks (FNN), 14 Convolutional Neural Networks (CNN), and 30 Recurrent Neural Networks (RNN) (60 total seed programs, 59 with logs).

### Seed Program SO IDs

The authoritative list of all 59 StackOverflow IDs and architecture labels is in:

- `data/seed_programs/seed_metadata.csv`
- `data/seed_programs/ATTRIBUTION.csv` (source URL, author display name,
  source license, and adaptation note for each seed)

This avoids duplication drift and keeps architecture assignments machine-checkable
(`FNN=16`, `CNN=14`, `RNN=30` in the current release).

Each SO ID maps to `https://stackoverflow.com/questions/{SO_ID}`.

Seed-program attribution and source-license metadata are documented separately
in `docs/SEED_ATTRIBUTION.md`.

### Corrections Applied

Each program was manually corrected to ensure:
- All required imports are present
- Data loading works (datasets are downloaded or synthesized)
- Model architecture is valid (compatible layer sizes, correct input/output shapes)
- Training completes without runtime errors
- The `EnhancedLoggingCallback` is integrated for dynamic feature collection

Corrected programs are stored in `data/seed_programs/{fnn,cnn,rnn}/`.

## Mutation Process

### DeepCrime Operators (24 operators)

The original 24 operators from DeepCrime (ISSTA 2021) were applied to generate faulty versions of the seed programs. These operators inject real faults observed in actual developer code, covering:

- **Training data faults**: label corruption, data deletion, class imbalance, noise injection, class overlap
- **Hyperparameter faults**: batch size, learning rate, epoch count, batching disabled
- **Activation faults**: changed, removed, or added activation functions
- **Loss function faults**: incompatible loss function substitution
- **Optimizer faults**: optimizer substitution, gradient clipping modification
- **Weight faults**: initializer changes, bias toggling
- **Regularization faults**: dropout rate changes, regularizer add/change/remove
- **Validation faults**: validation set removal
- **Training process faults**: early stopping patience modification

### DeepCrime++ Operators (10 operators, new in Deep4ge)

10 additional layer-fault operators were developed for this dataset to address the 21.67% of DNN faults that are layer-related (not covered by original DeepCrime):

- **CNN-specific**: kernel size, filter count, pooling size, strides, padding
- **Shared**: neuron count, layer addition, layer removal
- **RNN-specific**: layer type swap (LSTM/GRU), output shape modification

### Training and Feature Collection

Each mutated program (and each correct baseline) was trained with the `EnhancedLoggingCallback`, which records 31 columns per epoch (an epoch index, four core training/validation metrics, and 26 dynamic indicators):

- 5 training metrics (loss, accuracy for train and validation)
- 4 weight analysis features (large weights, constant weights, NaN weights)
- 4 accuracy/loss trend features (oscillation, gap, decrease/increase)
- 4 activation features (dying ReLU, saturation, mean/std activation)
- 8 gradient features (vanishing, exploding, NaN, statistics)
- 1 learning rate tracker
- 3 hardware utilization features (CPU, GPU memory, system memory)

Multiple runs (with different random seeds) were performed per operator-model combination.

## Preprocessing Pipeline

The raw CSV outputs from both DeepCrime and DeepCrime++ were preprocessed:

1. **Filename normalization**: Original filenames (e.g., `deepmultifixFNN_46642627_correct_change_batch_size_mutated0_MP_1_1_0.h5.csv`) were cleaned to standardized format (`{so_id}_{category}_{run:04d}.csv`)
2. **Operator code mapping**: Full operator names were mapped to 3-letter codes (e.g., `change_batch_size` -> `HBS`)
3. **Category labeling**: Operator codes were grouped into 7 fault categories
4. **Dying ReLU cleanup**: TensorFlow tensor strings (e.g., `tf.Tensor(False, ...)`) in the `dying_relu` column were parsed to binary 0/1
5. **Empty CSV removal**: Training runs that crashed before completing epoch 1 (header-only CSVs) were removed from the correct subset
6. **Manifest generation**: A master CSV index was built linking all files to their metadata

## Data Quality Notes

- **954 empty CSVs** were removed during preprocessing (392 correct, 562 dynamic). These represented training runs that crashed before completing the first epoch.
- All remaining CSVs contain the full 31-column header and at least one data row.
- The `dying_relu` column has been cleaned to binary 0/1 across all files.


## Architecture Mapping (FNN/CNN/RNN)

The original dataset release stored CNN-style seed programs inside the `seed_programs/fnn/` folder. For clarity and to make CNN coverage machine-checkable, seed programs are now separated into:

- `data/seed_programs/fnn/`
- `data/seed_programs/cnn/`
- `data/seed_programs/rnn/`

The file `data/seed_programs/seed_metadata.csv` records `(so_id, architecture, seed_file)` and summarizes how many logs were generated per seed. Note that one CNN seed program (`so_id=50079585`) is included as a seed but has no corresponding logs in the current manifest (hence 59 unique StackOverflow IDs with logs).
