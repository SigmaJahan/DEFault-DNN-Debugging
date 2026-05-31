"""
Weight-related mutation operators.

WCI - Change Weights Initialisation
WAB - Add Bias
WRB - Remove Bias

These operators modify model layer configurations via Keras model.get_config().
"""

import copy
import random

from .base import MutationOperator

KERAS_INITIALIZERS = [
    "zeros", "ones", "constant", "random_normal", "random_uniform",
    "truncated_normal", "orthogonal", "lecun_uniform", "glorot_normal",
    "glorot_uniform", "he_normal", "lecun_normal", "he_uniform",
]


class ChangeWeightsInitialisation(MutationOperator):
    """WCI: Changes the kernel initializer of a random eligible layer."""

    code = "WCI"
    name = "change_weights_initialisation"
    category = "Weight"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if layer["config"].get("kernel_initializer")
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)

        old_init_cfg = layers[idx]["config"]["kernel_initializer"]
        if isinstance(old_init_cfg, dict):
            old_name = old_init_cfg.get("class_name", "").lower().replace("_", "")
        else:
            old_name = str(old_init_cfg).lower().replace("_", "")

        candidates = copy.copy(KERAS_INITIALIZERS)
        # Remove current initializer
        formatted = [c.lower().replace("_", "") for c in candidates]
        if old_name in formatted:
            candidates.pop(formatted.index(old_name))

        # Avoid 'identity' for 3D+ weight tensors
        if len(model.layers) > idx and len(model.layers[idx].weights) > 0:
            if len(model.layers[idx].weights[0].shape) > 2:
                if "identity" in candidates:
                    candidates.remove("identity")

        new_init = random.choice(candidates)
        layers[idx]["config"]["kernel_initializer"] = new_init

        return _rebuild_model(model, config)


class AddBias(MutationOperator):
    """WAB: Enables bias on a layer that has use_bias=False."""

    code = "WAB"
    name = "add_bias"
    category = "Weight"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if "use_bias" in layer["config"] and not layer["config"]["use_bias"]
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)
        layers[idx]["config"]["use_bias"] = True

        return _rebuild_model(model, config)


class RemoveBias(MutationOperator):
    """WRB: Disables bias on a layer that has use_bias=True."""

    code = "WRB"
    name = "remove_bias"
    category = "Weight"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if "use_bias" in layer["config"] and layer["config"]["use_bias"]
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)
        layers[idx]["config"]["use_bias"] = False

        return _rebuild_model(model, config)


def _rebuild_model(model, config):
    """Rebuild a Keras model from a modified config, preserving weights where possible."""
    from tensorflow.keras.models import model_from_config
    new_model = model_from_config(config)
    for old_layer, new_layer in zip(model.layers, new_model.layers):
        try:
            new_layer.set_weights(old_layer.get_weights())
        except ValueError:
            pass
    return new_model
