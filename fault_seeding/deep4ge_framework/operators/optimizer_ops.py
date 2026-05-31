"""
Optimizer mutation operators.

OCH - Change Optimisation Function
OCG - Change Gradient Clip

Modifies optimizer selection and gradient clipping in model.compile().
"""

import ast
import random

from .base import MutationOperator

KERAS_OPTIMIZERS = ["sgd", "rmsprop", "adagrad", "adam", "adamax", "nadam"]


class ChangeOptimisationFunction(MutationOperator):
    """OCH: Replaces the optimizer in model.compile() with a random different one."""

    code = "OCH"
    name = "change_optimisation_function"
    category = "Optimization"
    applicable_to = ["FNN", "RNN", "CNN"]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_compile_call(node):
            for kw in node.keywords:
                if kw.arg == "optimizer":
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        old_opt = kw.value.value.lower()
                        candidates = [o for o in KERAS_OPTIMIZERS if o != old_opt]
                        kw.value = ast.Constant(value=random.choice(candidates))
                    break
        return node

    @staticmethod
    def _is_compile_call(node):
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == "compile")


class ChangeGradientClip(MutationOperator):
    """OCG: Modifies gradient clipping parameters (clipnorm/clipvalue) in the optimizer."""

    code = "OCG"
    name = "change_gradient_clip"
    category = "Optimization"
    applicable_to = ["FNN", "RNN", "CNN"]

    CLIP_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_optimizer_call(node):
            clip_val = random.choice(self.CLIP_VALUES)
            self._set_keyword(node, "clipnorm", ast.Constant(value=clip_val))
        return node

    @staticmethod
    def _is_optimizer_call(node):
        """Detect optimizer instantiation like Adam(...), SGD(...), etc."""
        if isinstance(node.func, ast.Name):
            return node.func.id.lower() in KERAS_OPTIMIZERS
        if isinstance(node.func, ast.Attribute):
            return node.func.attr.lower() in KERAS_OPTIMIZERS
        return False

    @staticmethod
    def _set_keyword(node, name, value):
        for kw in node.keywords:
            if kw.arg == name:
                kw.value = value
                return
        node.keywords.append(ast.keyword(arg=name, value=value))
