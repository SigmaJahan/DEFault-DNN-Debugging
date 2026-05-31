# Mutation Operator Catalog

Deep4ge implements **34 mutation operators** in the framework.

The **released dataset** uses **27 operators** across **7 primary fault categories** (Hyperparameter, Activation, Loss, Optimization, Weight, Regularization, Layer). Additional operators for training data, validation, and training process faults are implemented for extensibility but are **not used** in the current dataset release.

## Overview

| Category | Operators | Source | Architectures |
|----------|-----------|--------|---------------|
| Hyperparameter | HBS, HLR, HNE, HDB | DeepCrime | FNN, RNN, CNN |
| Activation | ACH, ARM, AAL | DeepCrime | FNN, RNN, CNN |
| Loss | FLC | DeepCrime | FNN, RNN, CNN |
| Optimization | OCH, OCG | DeepCrime | FNN, RNN, CNN |
| Weight | WCI, WAB, WRB | DeepCrime | FNN, RNN, CNN |
| Regularization | RCD, RAW, RCW, RRW | DeepCrime | FNN, RNN, CNN |
| Layer | LKS, LCF, LCP, LCS, LCD, LCN, LAD, LRM, LCT, LCO | **Deep4ge (new)** | Varies |

Additional operators from DeepCrime (training data, validation, training process) are registered but were not used in the primary dataset generation:
- Training Data: TCL, TRD, TUD, TAN, TCO
- Validation: VRM
- Training Process: RCP

---

## Hyperparameter Faults

| Code | Operator | Description |
|------|----------|-------------|
| **HBS** | Change Batch Size | Modifies the batch size to values: 1, 2, 4, 8, 16, 32 |
| **HLR** | Change Learning Rate | Modifies the optimizer learning rate |
| **HNE** | Change Epochs | Modifies the number of training epochs |
| **HDB** | Disable Batching | Switches to single-sample training (batch_size=1) |

## Activation Faults

| Code | Operator | Description |
|------|----------|-------------|
| **ACH** | Change Activation Function | Replaces a layer's activation with a random one from: elu, softmax, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, exponential, linear |
| **ARM** | Remove Activation Function | Replaces a non-linear activation with `linear` |
| **AAL** | Add Activation Function | Adds a random activation to layers with `linear` activation |

## Loss Function Faults

| Code | Operator | Description |
|------|----------|-------------|
| **FLC** | Change Loss Function | Replaces the loss function with a random one from 13 Keras losses |

## Optimization Faults

| Code | Operator | Description |
|------|----------|-------------|
| **OCH** | Change Optimisation Function | Replaces optimizer with: sgd, rmsprop, adagrad, adam, adamax, or nadam |
| **OCG** | Change Gradient Clip | Modifies gradient clipping value |

## Weight Faults

| Code | Operator | Description |
|------|----------|-------------|
| **WCI** | Change Weights Initialisation | Replaces kernel initializer with a random one from 13 Keras initializers |
| **WAB** | Add Bias | Enables bias on layers where it was disabled |
| **WRB** | Remove Bias | Disables bias on layers where it was enabled |

## Regularization Faults

| Code | Operator | Description |
|------|----------|-------------|
| **RCD** | Change Dropout Rate | Modifies dropout rate to: 0.125, 0.25, 0.75, or 1.0 |
| **RAW** | Add Weights Regularisation | Adds L1/L2/L1_L2 regularizer to non-regularized layers |
| **RCW** | Change Weights Regularisation | Changes existing regularizer type |
| **RRW** | Remove Weights Regularisation | Removes regularizer from layers |

## Layer Faults (Deep4ge Extension)

These 10 operators were added by Deep4ge to address the 21.67% of DNN faults that are layer-related (not covered by original DeepCrime).

### CNN-Specific

| Code | Operator | Description | Parameter Space |
|------|----------|-------------|----------------|
| **LKS** | Layer Kernel Size | Modifies Conv2D kernel_size | (2,2), (4,4), (6,6) |
| **LCF** | Layer Filter Count | Modifies Conv2D filters | 1, 2, 4, 8 |
| **LCP** | Layer Pooling Size | Modifies MaxPooling2D pool_size | (2,2), (4,4), (6,6) |
| **LCS** | Layer Strides | Modifies Conv2D strides | (1,1), (2,2), (3,3) |
| **LCD** | Layer Padding | Toggles Conv2D padding | valid ↔ same |

### Shared (FNN, RNN, CNN)

| Code | Operator | Description |
|------|----------|-------------|
| **LCN** | Layer Neuron Count | Modifies Dense layer units: 32, 64, 128, 256 |
| **LAD** | Layer Add | Inserts a Dense(32), Dropout(0.5), or Activation('tanh') layer |
| **LRM** | Layer Remove | Removes a random Dropout, Dense, or Activation layer |

### RNN-Specific

| Code | Operator | Description |
|------|----------|-------------|
| **LCT** | Layer Type Swap | Swaps LSTM ↔ GRU (including inside Bidirectional wrappers) |
| **LCO** | Layer Output Shape | Modifies LSTM/GRU unit count to random value in [1, 256] |
