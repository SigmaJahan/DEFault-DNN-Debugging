import ast
import random

kernel_replacement_sizes = [(1, 1), (2, 2), (7, 7)]
filter_replacement_sizes = [1, 3, 5, 10, 20]
pooling_replacement_sizes = [(1, 1), (3, 3), (5, 5)]
strides_replacement_sizes = [(1, 1), (3, 3), (5, 5)]
padding_replacements = ['valid', 'same', 'causal']
neuron_replacements = [1, 2, 5, 10, 50]


class KernelTransformer(ast.NodeTransformer):
    def __init__(self, replacement_sizes):
        self.replacement_sizes = replacement_sizes

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add':
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'Conv2D':
                    for keyword in arg.keywords:
                        if keyword.arg == 'kernel_size':
                            new_size = random.choice(kernel_replacement_sizes)
                            print(
                                f"Replacing {keyword.value.elts[0].value}x{keyword.value.elts[1].value} with {new_size[0]}x{new_size[1]}")
                            keyword.value = ast.Tuple(
                                elts=[ast.Constant(value=new_size[0]), ast.Constant(value=new_size[1])], ctx=ast.Load())
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


class FilterTransformer(ast.NodeTransformer):
    def __init__(self, replacement_sizes):
        self.replacement_sizes = replacement_sizes

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add':
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'Conv2D':
                    for keyword in arg.keywords:
                        if keyword.arg == 'filters':
                            new_size = random.choice(filter_replacement_sizes)
                            print(f"Replacing {keyword.value.n} with {new_size}")
                            keyword.value = ast.Constant(value=new_size)
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


class PoolingTransformer(ast.NodeTransformer):
    def __init__(self, replacement_sizes):
        self.replacement_sizes = replacement_sizes

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add':
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'MaxPooling2D':
                    for keyword in arg.keywords:
                        if keyword.arg == 'pool_size':
                            new_size = random.choice(pooling_replacement_sizes)
                            print(
                                f"Replacing {keyword.value.elts[0].value}x{keyword.value.elts[1].value} with {new_size[0]}x{new_size[1]}")
                            keyword.value = ast.Tuple(
                                elts=[ast.Constant(value=new_size[0]), ast.Constant(value=new_size[1])], ctx=ast.Load())
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


class StridesTransformer(ast.NodeTransformer):
    def __init__(self, replacement_sizes):
        self.replacement_sizes = replacement_sizes

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add':
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'Conv2D':
                    for keyword in arg.keywords:
                        if keyword.arg == 'strides':
                            new_size = random.choices(strides_replacement_sizes)
                            print(f"Replacing strides with {new_size[0]}x{new_size[1]}")
                            keyword.value = ast.Tuple(
                                elts=[ast.Constant(value=new_size[0]), ast.Constant(value=new_size[1])], ctx=ast.Load())
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


class PaddingTransformer(ast.NodeTransformer):
    def __init__(self, replacement_sizes):
        self.replacement_sizes = replacement_sizes

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add':
            for arg in node.args:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'Conv2D':
                    for keyword in arg.keywords:
                        if keyword.arg == 'padding':
                            new_size = random.choice(padding_replacements)
                            print(f"Replacing {keyword.value.s} with {new_size}")
                            keyword.value = ast.Constant(value=new_size)
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


class NeuronTransformer(ast.NodeTransformer):
    def __init__(self, replacement_sizes, counter):
        self.replacement_sizes = replacement_sizes
        self.counter = counter
        self.count = 0

    def visit_Call(self, node):
        if self.count < self.counter:
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'add':
                for arg in node.args:
                    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == 'Dense':
                        for keyword in arg.keywords:
                            if keyword.arg == 'units':
                                new_size = random.choice(neuron_replacements)
                                print(f"Replacing {keyword.value.n} with {new_size}")
                                self.count += 1
                                keyword.value = ast.Constant(value=new_size)
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


class DenseLayerCounter(ast.NodeVisitor):
    def __init__(self):
        self.dense_count = 0

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'Dense':
            self.dense_count += 1
        self.generic_visit(node)

    def count_dense_layers(self, node):
        self.visit(node)
        return self.dense_count


class CNNCheck(ast.NodeVisitor):
    def __init__(self):
        self.is_cnn = False

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ('Conv1D', 'Conv2D', 'Conv3D'):
            self.is_cnn = True
        self.generic_visit(node)

    def check_cnn(self, node):
        self.visit(node)
        return self.is_cnn
