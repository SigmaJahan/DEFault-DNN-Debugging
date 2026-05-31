"""
Operator registry, fault categories, and Keras constants for Deep4ge.

This replaces the scattered configuration across deepcrime's utils/constants.py,
utils/properties.py, and all subject-specific property files.
"""

# ─── Fault Categories ─────────────────────────────────────────────────────────
# Maps 3-letter operator codes to (full_name, fault_category, applicable_architectures)

OPERATOR_REGISTRY = {
    # Hyperparameter faults (from DeepCrime)
    "HBS": ("change_batch_size",           "Hyperparameter", ["FNN", "RNN", "CNN"]),
    "HLR": ("change_learning_rate",        "Hyperparameter", ["FNN", "RNN", "CNN"]),
    "HNE": ("change_epochs",               "Hyperparameter", ["FNN", "RNN", "CNN"]),
    "HDB": ("disable_batching",            "Hyperparameter", ["FNN", "RNN", "CNN"]),

    # Activation faults (from DeepCrime)
    "ACH": ("change_activation_function",  "Activation",     ["FNN", "RNN", "CNN"]),
    "ARM": ("remove_activation_function",  "Activation",     ["FNN", "RNN", "CNN"]),
    "AAL": ("add_activation_function",     "Activation",     ["FNN", "RNN", "CNN"]),

    # Loss function faults (from DeepCrime)
    "FLC": ("change_loss_function",        "Loss",           ["FNN", "RNN", "CNN"]),

    # Optimization faults (from DeepCrime)
    "OCH": ("change_optimisation_function","Optimization",   ["FNN", "RNN", "CNN"]),
    "OCG": ("change_gradient_clip",        "Optimization",   ["FNN", "RNN", "CNN"]),

    # Weight faults (from DeepCrime)
    "WCI": ("change_weights_initialisation","Weight",        ["FNN", "RNN", "CNN"]),
    "WAB": ("add_bias",                    "Weight",         ["FNN", "RNN", "CNN"]),
    "WRB": ("remove_bias",                 "Weight",         ["FNN", "RNN", "CNN"]),

    # Regularization faults (from DeepCrime)
    "RCD": ("change_dropout_rate",         "Regularization", ["FNN", "RNN", "CNN"]),
    "RAW": ("add_weights_regularisation",  "Regularization", ["FNN", "RNN", "CNN"]),
    "RCW": ("change_weights_regularisation","Regularization",["FNN", "RNN", "CNN"]),
    "RRW": ("remove_weights_regularisation","Regularization",["FNN", "RNN", "CNN"]),

    # Training data faults (from DeepCrime)
    "TCL": ("change_label",                "Training_Data",  ["FNN", "RNN", "CNN"]),
    "TRD": ("delete_training_data",        "Training_Data",  ["FNN", "RNN", "CNN"]),
    "TUD": ("unbalance_train_data",        "Training_Data",  ["FNN", "RNN", "CNN"]),
    "TAN": ("add_noise",                   "Training_Data",  ["FNN", "RNN", "CNN"]),
    "TCO": ("make_output_classes_overlap",  "Training_Data",  ["FNN", "RNN", "CNN"]),

    # Validation faults (from DeepCrime)
    "VRM": ("remove_validation_set",       "Validation",     ["FNN", "RNN", "CNN"]),

    # Training process faults (from DeepCrime)
    "RCP": ("change_earlystopping_patience","Training_Process",["FNN", "RNN", "CNN"]),

    # Layer faults — CNN specific (from DeepCrime++)
    "LKS": ("layer_kernel_size",           "Layer",          ["CNN"]),
    "LCF": ("layer_filter_count",          "Layer",          ["CNN"]),
    "LCP": ("layer_pooling_size",          "Layer",          ["CNN"]),
    "LCS": ("layer_strides",              "Layer",          ["CNN"]),
    "LCD": ("layer_padding",              "Layer",          ["CNN"]),

    # Layer faults — shared (from DeepCrime++)
    "LCN": ("layer_neuron_count",          "Layer",          ["FNN", "RNN", "CNN"]),
    "LAD": ("layer_add",                   "Layer",          ["FNN", "RNN", "CNN"]),
    "LRM": ("layer_remove",               "Layer",          ["FNN", "RNN", "CNN"]),

    # Layer faults — RNN specific (from DeepCrime++)
    "LCT": ("layer_type_swap",            "Layer",          ["RNN"]),
    "LCO": ("layer_output_shape",         "Layer",          ["RNN"]),
}

# ─── Category Groupings ───────────────────────────────────────────────────────

CATEGORIES = {
    "Hyperparameter":   ["HBS", "HLR", "HNE", "HDB"],
    "Activation":       ["ACH", "ARM", "AAL"],
    "Loss":             ["FLC"],
    "Optimization":     ["OCH", "OCG"],
    "Weight":           ["WCI", "WAB", "WRB"],
    "Regularization":   ["RCD", "RAW", "RCW", "RRW"],
    "Training_Data":    ["TCL", "TRD", "TUD", "TAN", "TCO"],
    "Validation":       ["VRM"],
    "Training_Process": ["RCP"],
    "Layer":            ["LKS", "LCF", "LCP", "LCS", "LCD", "LCN", "LAD", "LRM", "LCT", "LCO"],
}

# Maps the category names used in the CSV filenames to canonical names
CSV_CATEGORY_NAMES = {
    "Hyperparameter": "Hyperparameter",
    "Activation":     "Activation",
    "Loss":           "Loss",
    "Optimization":   "Optimization",
    "Weight":         "Weight",
    "Regularizer":    "Regularization",   # Note: "Regularizer" in CSVs → "Regularization"
    "Regularization": "Regularization",
    "Layer":          "Layer",
    "Training_Data":  "Training_Data",
    "Validation":     "Validation",
    "Training_Process":"Training_Process",
}

# ─── Keras Constants ──────────────────────────────────────────────────────────

ACTIVATION_FUNCTIONS = [
    "elu", "softmax", "selu", "softplus", "softsign",
    "relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear",
]

KERAS_OPTIMIZERS = ["sgd", "rmsprop", "adagrad", "adam", "adamax", "nadam"]

KERAS_LOSSES = [
    "mean_squared_error", "mean_absolute_error",
    "mean_absolute_percentage_error", "mean_squared_logarithmic_error",
    "squared_hinge", "hinge", "categorical_hinge", "logcosh", "huber_loss",
    "categorical_crossentropy", "binary_crossentropy",
    "kullback_leibler_divergence", "poisson",
]

KERAS_INITIALIZERS = [
    "zeros", "ones", "constant", "random_normal", "random_uniform",
    "truncated_normal", "orthogonal", "lecun_uniform", "glorot_normal",
    "glorot_uniform", "he_normal", "lecun_normal", "he_uniform",
]

KERAS_REGULARIZERS = ["l1", "l2", "l1_l2"]

# ─── Mutation Parameter Spaces ────────────────────────────────────────────────

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DROPOUT_VALUES = [0.125, 0.25, 0.75, 1.0]
KERNEL_SIZES = [(2, 2), (4, 4), (6, 6)]
FILTER_SIZES = [1, 2, 4, 8]
POOLING_SIZES = [(2, 2), (4, 4), (6, 6)]
STRIDE_SIZES = [(1, 1), (2, 2), (3, 3)]
NEURON_COUNTS = [32, 64, 128, 256]

# ─── Training Log Schema ─────────────────────────────────────────────────────

# 31-column schema from EnhancedLoggingCallback
CSV_COLUMNS = [
    "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
    "large_weight_count", "acc_gap_too_big", "loss_oscillation", "dying_relu",
    "gradient_vanish", "gradient_explode", "decrease_acc_count", "increase_loss_count",
    "cons_mean_weight_count", "cons_std_weight_count",
    "nan_weight_count", "nan_gradients_count", "saturated_activation",
    "mean_gradient", "gradient_std", "gradient_max", "gradient_min",
    "gradient_median", "adjusted_lr", "mean_activation", "std_activation",
    "mean_grad", "std_grad", "cpu_utilization", "gpu_memory_utilization", "memory_usage",
]
