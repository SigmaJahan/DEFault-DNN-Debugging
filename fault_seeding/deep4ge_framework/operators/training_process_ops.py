"""
Training process mutation operator.

RCP - Change Early Stopping Patience

Modifies the EarlyStopping callback patience parameter.
"""

import ast
import random

from .base import MutationOperator


class ChangeEarlyStoppingPatience(MutationOperator):
    """RCP: Modifies the patience parameter of EarlyStopping callbacks.

    Finds EarlyStopping(...) instantiations in the AST and changes the
    patience keyword argument.
    """

    code = "RCP"
    name = "change_earlystopping_patience"
    category = "Training_Process"
    applicable_to = ["FNN", "RNN", "CNN"]

    PATIENCE_VALUES = [1, 2, 5, 10, 20, 50]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_early_stopping_call(node):
            new_patience = random.choice(self.PATIENCE_VALUES)
            self._set_keyword(node, "patience", ast.Constant(value=new_patience))
        return node

    @staticmethod
    def _is_early_stopping_call(node):
        """Detect EarlyStopping(...) instantiation."""
        if isinstance(node.func, ast.Name):
            return node.func.id == "EarlyStopping"
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "EarlyStopping"
        return False

    @staticmethod
    def _set_keyword(node, name, value):
        for kw in node.keywords:
            if kw.arg == name:
                kw.value = value
                return
        node.keywords.append(ast.keyword(arg=name, value=value))
