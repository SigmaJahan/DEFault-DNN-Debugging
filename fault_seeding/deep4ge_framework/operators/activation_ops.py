"""
Activation function mutation operators.

ACH - Change Activation Function
ARM - Remove Activation Function
AAL - Add Activation Function

These operators modify activation functions in model layers by operating on the
compiled Keras model's configuration dict.
"""

import copy
import random

from .base import MutationOperator

ACTIVATION_FUNCTIONS = [
    "elu", "softmax", "selu", "softplus", "softsign",
    "relu", "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear",
]


class ChangeActivationFunction(MutationOperator):
    """ACH: Replaces a layer's activation with a randomly selected different one.

    Operates on the compiled model config: picks a random layer with a
    non-linear activation and swaps it to a different activation function.
    """

    code = "ACH"
    name = "change_activation_function"
    category = "Activation"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        """Mutate model in-place via config manipulation."""
        from tensorflow.keras.models import model_from_config as _mfc
        config = model.get_config()
        layers = config["layers"]

        # Find eligible layers (those with an 'activation' config)
        eligible = [
            i for i, layer in enumerate(layers)
            if layer["config"].get("activation")
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)
        old_act = layers[idx]["config"]["activation"]

        candidates = [a for a in ACTIVATION_FUNCTIONS if a != old_act]
        new_act = random.choice(candidates)
        layers[idx]["config"]["activation"] = new_act

        return _rebuild_model(model, config)


class RemoveActivationFunction(MutationOperator):
    """ARM: Removes a non-linear activation by setting it to 'linear'.

    Finds a layer with a non-linear activation and replaces it with 'linear'.
    """

    code = "ARM"
    name = "remove_activation_function"
    category = "Activation"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        from tensorflow.keras.models import model_from_config as _mfc
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if layer["config"].get("activation")
            and layer["config"]["activation"] != "linear"
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)
        layers[idx]["config"]["activation"] = "linear"

        return _rebuild_model(model, config)


class AddActivationFunction(MutationOperator):
    """AAL: Adds a non-linear activation to a layer that currently has 'linear'.

    Finds a layer with 'linear' activation and sets it to a random non-linear one.
    """

    code = "AAL"
    name = "add_activation_function"
    category = "Activation"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        from tensorflow.keras.models import model_from_config as _mfc
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if layer["config"].get("activation") == "linear"
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)
        candidates = [a for a in ACTIVATION_FUNCTIONS if a != "linear"]
        layers[idx]["config"]["activation"] = random.choice(candidates)

        return _rebuild_model(model, config)


def _rebuild_model(model, config):
    """Rebuild a Keras model from a modified config, preserving weights."""
    from tensorflow.keras.models import model_from_config
    new_model = model_from_config(config)
    for old_layer, new_layer in zip(model.layers, new_model.layers):
        try:
            new_layer.set_weights(old_layer.get_weights())
        except ValueError:
            pass  # Shape mismatch — skip
    return new_model
