"""
Base class for all mutation operators.

All operators work by transforming the Python AST of a DNN program.
"""

import ast
import textwrap


class MutationOperator(ast.NodeTransformer):
    """Base class for all Deep4ge mutation operators.

    Subclasses must define:
        code: 3-letter operator abbreviation (e.g., "HBS")
        name: Full operator name (e.g., "change_batch_size")
        category: Fault category (e.g., "Hyperparameter")
        applicable_to: List of applicable architectures (e.g., ["FNN", "RNN", "CNN"])
    """

    code: str = ""
    name: str = ""
    category: str = ""
    applicable_to: list = []

    def mutate(self, source_code: str, params: dict = None) -> str:
        """Apply mutation to source code string, return mutated source code string."""
        tree = ast.parse(source_code)
        if params:
            self._params = params
        mutated_tree = self.visit(tree)
        ast.fix_missing_locations(mutated_tree)
        return ast.unparse(mutated_tree)

    def __repr__(self):
        return f"{self.code} ({self.name}) [{self.category}]"


class ModifySavePath(ast.NodeTransformer):
    """Rewrites model.save() path to include mutation info."""

    def __init__(self, new_suffix: str):
        self.new_suffix = new_suffix

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "save"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "model"):
            if node.args:
                node.args[0] = ast.Constant(value=f"models/{self.new_suffix}.h5")
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


class ModifyCallbackFilename(ast.NodeTransformer):
    """Rewrites the callback_filename variable to include mutation info."""

    def __init__(self, new_filename: str):
        self.new_filename = new_filename

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "callback_filename":
                node.value = ast.Constant(value=self.new_filename)
        ast.fix_missing_locations(node)
        return self.generic_visit(node)
