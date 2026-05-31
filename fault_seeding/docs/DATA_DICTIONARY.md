# Data Dictionary: Training Log CSV Columns

All training log CSVs contain **31 columns** collected per epoch during DNN model training via the `EnhancedLoggingCallback`.

## Training Metrics (5 columns)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `epoch` | int | Training epoch number (0-indexed) |
| 2 | `train_loss` | float | Training loss at epoch end |
| 3 | `train_acc` | float | Training accuracy (or 1 - MAPE/100 for regression) |
| 4 | `val_loss` | float | Validation loss at epoch end |
| 5 | `val_acc` | float | Validation accuracy |

## Weight Analysis (4 columns)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 6 | `large_weight_count` | int | Count of weights with absolute value > 10.0 |
| 14 | `cons_mean_weight_count` | binary | 1 if mean weight unchanged from previous epoch |
| 15 | `cons_std_weight_count` | binary | 1 if std of weights unchanged from previous epoch |
| 16 | `nan_weight_count` | int | Count of NaN values in model weights |

## Accuracy and Loss Trends (4 columns)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 7 | `acc_gap_too_big` | binary | 1 if |train_acc - val_acc| > 0.1 (overfitting indicator) |
| 8 | `loss_oscillation` | binary | 1 if |loss[t-1] - loss[t]| > 0.01 |
| 12 | `decrease_acc_count` | binary | 1 if accuracy decreased from previous epoch |
| 13 | `increase_loss_count` | binary | 1 if loss increased from previous epoch |

## Activation Analysis (4 columns)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 9 | `dying_relu` | binary | 1 if >70% of ReLU outputs are <= 0 (dead neurons) |
| 18 | `saturated_activation` | bool | True if >50% of sigmoid/tanh outputs are saturated |
| 25 | `mean_activation`* | float | Mean activation value across layers with activation functions |
| 26 | `std_activation`* | float | Std of activation values across layers |

## Gradient Analysis (8 columns)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 10 | `gradient_vanish` | binary | 1 if mean absolute gradient < 1e-4 |
| 11 | `gradient_explode` | binary | 1 if max absolute gradient > 70 |
| 17 | `nan_gradients_count` | int | Count of NaN values in gradients |
| 19 | `mean_gradient` | float | Mean of per-layer gradient norms |
| 20 | `gradient_std` | float | Std of per-layer gradient norms |
| 21 | `gradient_max`* | float | Maximum per-layer gradient norm |
| 22 | `gradient_min`* | float | Minimum per-layer gradient norm |
| 23 | `gradient_median`* | float | Median per-layer gradient norm |
| 27 | `mean_grad`* | float | Mean of per-weight gradient values |
| 28 | `std_grad`* | float | Std of per-weight gradient values |

## Learning Rate (1 column)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 24 | `adjusted_lr`* | float | Learning rate value at the epoch (tracks scheduled/adaptive LR) |

## Hardware Utilization (3 columns)

| # | Column | Type | Description |
|---|--------|------|-------------|
| 29 | `cpu_utilization`* | float | CPU utilization percentage during training |
| 30 | `gpu_memory_utilization`* | float | GPU peak memory usage in MB |
| 31 | `memory_usage`* | float | System memory usage percentage |

*\* Novel features proposed by Deep4ge (not present in prior work)*

## Feature Sources

- **17 features** from existing literature (DeepFD, AutoTrainer, DeepDiagnosis, UMLAUT, DeepLocalize)
- **11 novel/extended features** proposed by Deep4ge and marked with `*` above: `gradient_max`, `gradient_min`, `gradient_median`, `adjusted_lr`, `mean_activation`, `std_activation`, `mean_grad`, `std_grad`, `cpu_utilization`, `gpu_memory_utilization`, `memory_usage`

## Parsing Notes

- All values are numeric (int, float, or boolean).
- The `dying_relu` column in raw CSVs may contain TensorFlow tensor strings (e.g., `tf.Tensor(False, shape=(), dtype=bool)`) — these have been cleaned to binary 0/1 in the preprocessed dataset.
