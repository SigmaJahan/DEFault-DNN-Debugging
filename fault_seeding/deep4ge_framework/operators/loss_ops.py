"""
Loss function mutation operator.

FLC - Change Loss Function

Modifies the loss function in model.compile() to a different Keras loss.
"""

import ast
import copy
import random

from .base import MutationOperator

KERAS_LOSSES = [
    "mean_squared_error", "mean_absolute_error",
    "mean_absolute_percentage_error", "mean_squared_logarithmic_error",
    "squared_hinge", "hinge", "categorical_hinge", "logcosh", "huber_loss",
    "categorical_crossentropy", "binary_crossentropy",
    "kullback_leibler_divergence", "poisson",
]


class ChangeLossFunction(MutationOperator):
    """FLC: Replaces the loss function in model.compile() with a random different one."""

    code = "FLC"
    name = "change_loss_function"
    category = "Loss"
    applicable_to = ["FNN", "RNN", "CNN"]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_compile_call(node):
            for kw in node.keywords:
                if kw.arg == "loss":
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        old_loss = kw.value.value
                        candidates = [l for l in KERAS_LOSSES if l != old_loss]
                        kw.value = ast.Constant(value=random.choice(candidates))
                    break
        return node

    @staticmethod
    def _is_compile_call(node):
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == "compile")
