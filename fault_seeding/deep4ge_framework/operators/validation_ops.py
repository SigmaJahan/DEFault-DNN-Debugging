"""
Validation mutation operator.

VRM - Remove Validation Set

Modifies model.fit() to remove validation data.
"""

import ast

from .base import MutationOperator


class RemoveValidationSet(MutationOperator):
    """VRM: Removes validation data/split from model.fit() call.

    Sets validation_data=None and validation_split=0.0, effectively
    disabling validation during training.
    """

    code = "VRM"
    name = "remove_validation_set"
    category = "Validation"
    applicable_to = ["FNN", "RNN", "CNN"]

    def visit_Call(self, node):
        self.generic_visit(node)
        if self._is_fit_call(node):
            node.keywords = [
                kw for kw in node.keywords
                if kw.arg not in ("validation_data", "validation_split")
            ]
        return node

    @staticmethod
    def _is_fit_call(node):
        return (isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit")
