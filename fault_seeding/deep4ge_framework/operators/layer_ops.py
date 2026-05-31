"""
Layer fault mutation operators (from DeepCrime++ extension).

CNN-specific: LKS, LCF, LCP, LCS, LCD
Shared: LCN, LAD, LRM
RNN-specific: LCT, LCO
"""

import ast
import random
from .base import MutationOperator
from ..config import KERNEL_SIZES, FILTER_SIZES, POOLING_SIZES, STRIDE_SIZES, NEURON_COUNTS


# ─── CNN Layer Operators ──────────────────────────────────────────────────────

class LayerKernelSize(MutationOperator):
    """Modify Conv2D kernel_size to a random value from KERNEL_SIZES."""
    code = "LKS"
    name = "layer_kernel_size"
    category = "Layer"
    applicable_to = ["CNN"]

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "Conv2D":
                    for kw in arg.keywords:
                        if kw.arg == "kernel_size":
                            new = random.choice(KERNEL_SIZES)
                            kw.value = ast.Tuple(
                                elts=[ast.Constant(value=new[0]), ast.Constant(value=new[1])],
                                ctx=ast.Load(),
                            )
            ast.fix_missing_locations(node)
        return self.generic_visit(node)


class LayerFilterCount(MutationOperator):
    """Modify Conv2D filters to a random value from FILTER_SIZES."""
    code = "LCF"
    name = "layer_filter_count"
    category = "Layer"
    applicable_to = ["CNN"]

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "Conv2D":
                    for kw in arg.keywords:
                        if kw.arg == "filters":
                            kw.value = ast.Constant(value=random.choice(FILTER_SIZES))
            ast.fix_missing_locations(node)
        return self.generic_visit(node)


class LayerPoolingSize(MutationOperator):
    """Modify MaxPooling2D pool_size to a random value from POOLING_SIZES."""
    code = "LCP"
    name = "layer_pooling_size"
    category = "Layer"
    applicable_to = ["CNN"]

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "MaxPooling2D":
                    for kw in arg.keywords:
                        if kw.arg == "pool_size":
                            new = random.choice(POOLING_SIZES)
                            kw.value = ast.Tuple(
                                elts=[ast.Constant(value=new[0]), ast.Constant(value=new[1])],
                                ctx=ast.Load(),
                            )
            ast.fix_missing_locations(node)
        return self.generic_visit(node)


class LayerStrides(MutationOperator):
    """Modify Conv2D strides to a random value from STRIDE_SIZES."""
    code = "LCS"
    name = "layer_strides"
    category = "Layer"
    applicable_to = ["CNN"]

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "Conv2D":
                    for kw in arg.keywords:
                        if kw.arg == "strides":
                            new = random.choice(STRIDE_SIZES)
                            kw.value = ast.Tuple(
                                elts=[ast.Constant(value=new[0]), ast.Constant(value=new[1])],
                                ctx=ast.Load(),
                            )
            ast.fix_missing_locations(node)
        return self.generic_visit(node)


class LayerPadding(MutationOperator):
    """Toggle Conv2D padding between 'valid' and 'same'."""
    code = "LCD"
    name = "layer_padding"
    category = "Layer"
    applicable_to = ["CNN"]

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "Conv2D":
                    for kw in arg.keywords:
                        if kw.arg == "padding":
                            current = getattr(kw.value, "value", getattr(kw.value, "s", "valid"))
                            kw.value = ast.Constant(value="valid" if current == "same" else "same")
            ast.fix_missing_locations(node)
        return self.generic_visit(node)


# ─── Shared Layer Operators ───────────────────────────────────────────────────

class LayerNeuronCount(MutationOperator):
    """Modify Dense layer units to a random value from NEURON_COUNTS (skips output layer)."""
    code = "LCN"
    name = "layer_neuron_count"
    category = "Layer"
    applicable_to = ["FNN", "RNN", "CNN"]

    def __init__(self):
        super().__init__()
        self._dense_count = 0
        self._max_dense = None

    def mutate(self, source_code, params=None):
        tree = ast.parse(source_code)
        counter = _DenseLayerCounter()
        counter.visit(tree)
        self._max_dense = counter.dense_count - 1  # skip output layer
        self._dense_count = 0
        mutated = self.visit(tree)
        ast.fix_missing_locations(mutated)
        return ast.unparse(mutated)

    def visit_Call(self, node):
        if self._max_dense and self._dense_count < self._max_dense:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
                for arg in node.args:
                    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "Dense":
                        for kw in arg.keywords:
                            if kw.arg == "units":
                                kw.value = ast.Constant(value=random.choice(NEURON_COUNTS))
                                self._dense_count += 1
                ast.fix_missing_locations(node)
        return self.generic_visit(node)


class LayerAdd(MutationOperator):
    """Insert a random layer (Dense, Dropout, or Activation) after an existing layer."""
    code = "LAD"
    name = "layer_add"
    category = "Layer"
    applicable_to = ["FNN", "RNN", "CNN"]

    _layer_choices = [
        "model.add(layers.Dense(32, activation='relu'))",
        "model.add(layers.Dropout(0.5))",
        "model.add(layers.Activation('tanh'))",
    ]

    def visit_FunctionDef(self, node):
        present_types = []
        for stmt in node.body:
            self.generic_visit(stmt)
            if _is_model_add(stmt):
                layer_type = _get_layer_type(stmt)
                if layer_type in ("Dense", "Dropout", "Activation") and layer_type not in present_types:
                    present_types.append(layer_type)

        if not present_types:
            return node

        chosen_type = random.choice(present_types)
        eligible = [s for s in node.body if _is_model_add(s) and _get_layer_type(s) == chosen_type]

        if eligible:
            target = random.choice(eligible)
            idx = node.body.index(target) + 1
            new_code = random.choice(self._layer_choices)
            new_node = ast.parse(new_code).body[0].value
            node.body.insert(idx, ast.Expr(value=new_node))

        ast.fix_missing_locations(node)
        return node


class LayerRemove(MutationOperator):
    """Remove a random Dropout, Dense, or Activation layer."""
    code = "LRM"
    name = "layer_remove"
    category = "Layer"
    applicable_to = ["FNN", "RNN", "CNN"]

    def visit_FunctionDef(self, node):
        removable = [
            s for s in node.body
            if _is_model_add(s) and _get_layer_type(s) in ("Dropout", "Dense", "Activation")
        ]
        if removable:
            node.body.remove(random.choice(removable))
        ast.fix_missing_locations(node)
        return node


# ─── RNN Layer Operators ──────────────────────────────────────────────────────

class LayerTypeSwap(MutationOperator):
    """Swap a random LSTM layer to GRU or vice versa."""
    code = "LCT"
    name = "layer_type_swap"
    category = "Layer"
    applicable_to = ["RNN"]

    def visit_FunctionDef(self, node):
        lstm_nodes, gru_nodes = [], []
        for stmt in node.body:
            if _is_model_add(stmt):
                layer = stmt.value.args[0]
                _find_rnn_layers(layer, lstm_nodes, gru_nodes)

        if lstm_nodes:
            random.choice(lstm_nodes).func.attr = "GRU"
        if gru_nodes:
            random.choice(gru_nodes).func.attr = "LSTM"
        return node


class LayerOutputShape(MutationOperator):
    """Modify the unit count of a random LSTM/GRU layer."""
    code = "LCO"
    name = "layer_output_shape"
    category = "Layer"
    applicable_to = ["RNN"]

    def visit_FunctionDef(self, node):
        layer_nodes = []
        for stmt in node.body:
            if _is_model_add(stmt):
                layer = stmt.value.args[0]
                _find_rnn_layers(layer, layer_nodes, layer_nodes)

        if layer_nodes:
            target = random.choice(layer_nodes)
            if target.args:
                target.args[0] = ast.Constant(value=random.randint(1, 256))
        ast.fix_missing_locations(node)
        return node


# ─── Helpers ──────────────────────────────────────────────────────────────────

class _DenseLayerCounter(ast.NodeVisitor):
    def __init__(self):
        self.dense_count = 0

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "Dense":
            self.dense_count += 1
        self.generic_visit(node)


def _is_model_add(stmt):
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "add"
        and stmt.value.args
    )


def _get_layer_type(stmt):
    arg = stmt.value.args[0]
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
        return arg.func.attr
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
        return arg.func.id
    return None


def _find_rnn_layers(layer, lstm_list, gru_list):
    if isinstance(layer, ast.Call):
        if hasattr(layer.func, "attr"):
            if layer.func.attr == "LSTM":
                lstm_list.append(layer)
            elif layer.func.attr == "GRU":
                gru_list.append(layer)
        elif hasattr(layer.func, "id") and layer.func.id == "Bidirectional":
            if layer.args:
                _find_rnn_layers(layer.args[0], lstm_list, gru_list)
