"""All 34 mutation operators for DNN fault injection."""

from .base import MutationOperator

# Hyperparameter (4)
from .hyperparameter_ops import (
    ChangeBatchSize, ChangeLearningRate, ChangeEpochs, DisableBatching,
)

# Activation (3)
from .activation_ops import (
    ChangeActivationFunction, RemoveActivationFunction, AddActivationFunction,
)

# Loss (1)
from .loss_ops import ChangeLossFunction

# Optimization (2)
from .optimizer_ops import ChangeOptimisationFunction, ChangeGradientClip

# Weight (3)
from .weight_ops import ChangeWeightsInitialisation, AddBias, RemoveBias

# Regularization (4)
from .regularization_ops import (
    ChangeDropoutRate, AddWeightsRegularisation,
    ChangeWeightsRegularisation, RemoveWeightsRegularisation,
)

# Training data (5)
from .training_data_ops import (
    ChangeLabel, DeleteTrainingData, UnbalanceTrainingData,
    AddNoise, MakeOutputClassesOverlap,
)

# Validation (1)
from .validation_ops import RemoveValidationSet

# Training process (1)
from .training_process_ops import ChangeEarlyStoppingPatience

# Layer (10) — from DeepCrime++
from .layer_ops import (
    LayerKernelSize, LayerFilterCount, LayerPoolingSize, LayerStrides,
    LayerPadding, LayerNeuronCount, LayerAdd, LayerRemove,
    LayerTypeSwap, LayerOutputShape,
)

# Registry: code -> class
OPERATOR_CLASSES = {
    "HBS": ChangeBatchSize,
    "HLR": ChangeLearningRate,
    "HNE": ChangeEpochs,
    "HDB": DisableBatching,
    "ACH": ChangeActivationFunction,
    "ARM": RemoveActivationFunction,
    "AAL": AddActivationFunction,
    "FLC": ChangeLossFunction,
    "OCH": ChangeOptimisationFunction,
    "OCG": ChangeGradientClip,
    "WCI": ChangeWeightsInitialisation,
    "WAB": AddBias,
    "WRB": RemoveBias,
    "RCD": ChangeDropoutRate,
    "RAW": AddWeightsRegularisation,
    "RCW": ChangeWeightsRegularisation,
    "RRW": RemoveWeightsRegularisation,
    "TCL": ChangeLabel,
    "TRD": DeleteTrainingData,
    "TUD": UnbalanceTrainingData,
    "TAN": AddNoise,
    "TCO": MakeOutputClassesOverlap,
    "VRM": RemoveValidationSet,
    "RCP": ChangeEarlyStoppingPatience,
    "LKS": LayerKernelSize,
    "LCF": LayerFilterCount,
    "LCP": LayerPoolingSize,
    "LCS": LayerStrides,
    "LCD": LayerPadding,
    "LCN": LayerNeuronCount,
    "LAD": LayerAdd,
    "LRM": LayerRemove,
    "LCT": LayerTypeSwap,
    "LCO": LayerOutputShape,
}
