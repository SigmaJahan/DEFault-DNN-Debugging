"""
Hyperparameter mutation operators.

HBS - Change Batch Size
HLR - Change Learning Rate
HNE - Change Epochs
HDB - Disable Batching

These operators modify training hyperparameters by transforming the AST of
model.fit() and model.compile() calls.
"""

import ast
import random

from .base import MutationOperator


class ChangeBatchSize(MutationOperator):
    """HBS: Modifies batch_size in model.fit() to a random value."""

    code = "HBS"
    name = "change_batch_size"
    category = "Hyperparameter"
    applicable_to = ["FNN", "RNN", "CNN"]

    BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_fit_call(node):
            new_batch = random.choice(self.BATCH_SIZES)
            self._set_keyword(node, "batch_size", ast.Constant(value=new_batch))
        return node

    @staticmethod
    def _is_fit_call(node):
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit")

    @staticmethod
    def _set_keyword(node, name, value):
        for kw in node.keywords:
            if kw.arg == name:
                kw.value = value
                return
        node.keywords.append(ast.keyword(arg=name, value=value))


class ChangeLearningRate(MutationOperator):
    """HLR: Modifies the optimizer's learning rate via K.variable."""

    code = "HLR"
    name = "change_learning_rate"
    category = "Hyperparameter"
    applicable_to = ["FNN", "RNN", "CNN"]

    LR_RANGE = (1e-5, 1.0)

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_compile_call(node):
            import math
            log_low = math.log10(self.LR_RANGE[0])
            log_high = math.log10(self.LR_RANGE[1])
            new_lr = 10 ** random.uniform(log_low, log_high)
            # Wrap optimizer argument: operator_change_learning_rate(optimizer, new_lr)
            for kw in node.keywords:
                if kw.arg == "optimizer":
                    kw.value = self._wrap_lr(kw.value, new_lr)
                    break
            else:
                if node.args:
                    node.args[0] = self._wrap_lr(node.args[0], new_lr)
        return node

    @staticmethod
    def _is_compile_call(node):
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == "compile")

    @staticmethod
    def _wrap_lr(optimizer_node, new_lr):
        """Inject learning rate change after optimizer creation."""
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="__import__('tensorflow.keras.backend')", ctx=ast.Load()),
                attr="variable",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=new_lr)],
            keywords=[],
        )


class ChangeEpochs(MutationOperator):
    """HNE: Modifies epochs in model.fit()."""

    code = "HNE"
    name = "change_epochs"
    category = "Hyperparameter"
    applicable_to = ["FNN", "RNN", "CNN"]

    EPOCH_MULTIPLIERS = [0.25, 0.5, 2.0, 4.0]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_fit_call(node):
            for kw in node.keywords:
                if kw.arg == "epochs":
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                        multiplier = random.choice(self.EPOCH_MULTIPLIERS)
                        kw.value = ast.Constant(value=max(1, int(kw.value.value * multiplier)))
                    break
        return node

    @staticmethod
    def _is_fit_call(node):
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit")


class DisableBatching(MutationOperator):
    """HDB: Disables batching by removing batch_size or setting it very large."""

    code = "HDB"
    name = "disable_batching"
    category = "Hyperparameter"
    applicable_to = ["FNN", "RNN", "CNN"]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_fit_call(node):
            # Set batch_size to a very large value (effectively full-batch)
            self._set_keyword(node, "batch_size", ast.Constant(value=100000))
        return node

    @staticmethod
    def _is_fit_call(node):
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit")

    @staticmethod
    def _set_keyword(node, name, value):
        for kw in node.keywords:
            if kw.arg == name:
                kw.value = value
                return
        node.keywords.append(ast.keyword(arg=name, value=value))
