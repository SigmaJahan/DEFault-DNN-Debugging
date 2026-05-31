"""
Regularization mutation operators.

RCD - Change Dropout Rate
RAW - Add Weights Regularisation
RCW - Change Weights Regularisation
RRW - Remove Weights Regularisation

These operators modify dropout and kernel regularization via model config.
"""

import copy
import random

from .base import MutationOperator

KERAS_REGULARIZERS = ["l1", "l2", "l1_l2"]
DROPOUT_VALUES = [0.125, 0.25, 0.75, 1.0]


class ChangeDropoutRate(MutationOperator):
    """RCD: Modifies the dropout rate of a Dropout layer."""

    code = "RCD"
    name = "change_dropout_rate"
    category = "Regularization"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if layer["class_name"] == "Dropout"
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)
        layers[idx]["config"]["rate"] = random.choice(DROPOUT_VALUES)

        return _rebuild_model(model, config)


class AddWeightsRegularisation(MutationOperator):
    """RAW: Adds kernel regularization to a layer that has none."""

    code = "RAW"
    name = "add_weights_regularisation"
    category = "Regularization"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if "kernel_regularizer" in layer["config"]
            and layer["config"]["kernel_regularizer"] is None
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)
        layers[idx]["config"]["kernel_regularizer"] = random.choice(KERAS_REGULARIZERS)

        return _rebuild_model(model, config)


class ChangeWeightsRegularisation(MutationOperator):
    """RCW: Changes an existing kernel regularizer to a different type."""

    code = "RCW"
    name = "change_weights_regularisation"
    category = "Regularization"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if layer["config"].get("kernel_regularizer") is not None
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)

        # Detect current regularizer type
        reg_config = layers[idx]["config"]["kernel_regularizer"]
        if isinstance(reg_config, dict) and "config" in reg_config:
            l1_val = reg_config["config"].get("l1", 0)
            l2_val = reg_config["config"].get("l2", 0)
            if l1_val in (0, 0.0, "0", "0.0"):
                old_reg = "l2"
            elif l2_val in (0, 0.0, "0", "0.0"):
                old_reg = "l1"
            else:
                old_reg = "l1_l2"
        else:
            old_reg = str(reg_config)

        candidates = [r for r in KERAS_REGULARIZERS if r != old_reg]
        layers[idx]["config"]["kernel_regularizer"] = random.choice(candidates)

        return _rebuild_model(model, config)


class RemoveWeightsRegularisation(MutationOperator):
    """RRW: Removes kernel regularization from a layer."""

    code = "RRW"
    name = "remove_weights_regularisation"
    category = "Regularization"
    applicable_to = ["FNN", "RNN", "CNN"]

    @staticmethod
    def apply_to_model(model, layer_index=None):
        config = model.get_config()
        layers = config["layers"]

        eligible = [
            i for i, layer in enumerate(layers)
            if layer["config"].get("kernel_regularizer") is not None
        ]
        if not eligible:
            return model

        idx = layer_index if layer_index is not None else random.choice(eligible)
        layers[idx]["config"]["kernel_regularizer"] = None

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
