import ast
import random
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class LayerInserter(ast.NodeTransformer):
    def __init__(self, layer_choices: List[str] = None):
        super().__init__()
        if layer_choices is None:
            layer_choices = [
                "model.add(layers.Dense(32, activation='relu'))",
                "model.add(layers.Dropout(0.5))",
                "model.add(layers.Activation('tanh'))",
                "model.add(layers.Dense(64, activation='relu'))",
                "model.add(layers.Dropout(0.3))",
                "model.add(layers.Activation('sigmoid'))",
                "model.add(layers.Dense(128, activation='relu))",
                "model.add(layers.Dropout(0.7))"
            ]
        self.layer_choices = layer_choices
        self.present_layer_types = []

    def visit_FunctionDef(self, node):
        try:
            self.present_layer_types = self.find_present_layer_types(node.body)
            if self.present_layer_types:
                chosen_layer_type = random.choice(self.present_layer_types)
                eligible_statements = [
                    stmt for stmt in node.body if self.is_layer_of_type(stmt, chosen_layer_type)
                ]
                if eligible_statements:
                    chosen_statement = random.choice(eligible_statements)
                    new_layer_code = random.choice(self.layer_choices)
                    new_layer_node = ast.parse(new_layer_code).body[0].value
                    index = node.body.index(chosen_statement) + 1
                    node.body.insert(index, ast.Expr(value=new_layer_node))

                ast.fix_missing_locations(node)
        except Exception as e:
            logging.error(f"Error in LayerInserter: {e}")
        return node

    def find_present_layer_types(self, body):
        present_layer_types = []
        for stmt in body:
            if self.is_add_layer_call(stmt):
                layer_type = stmt.value.args[0].func.attr
                if layer_type in ['Dense', 'Dropout', 'Activation'] and layer_type not in present_layer_types:
                    present_layer_types.append(layer_type)
        return present_layer_types

    def is_add_layer_call(self, stmt):
        return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and
                isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == 'add' and
                isinstance(stmt.value.args[0], ast.Call) and isinstance(stmt.value.args[0].func, ast.Attribute))

    def is_layer_of_type(self, stmt, layer_type):
        return self.is_add_layer_call(stmt) and stmt.value.args[0].func.attr == layer_type


class LayerRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        try:
            removable_nodes = [
                stmt for stmt in node.body if self.is_removable_layer(stmt)
            ]
            if removable_nodes:
                node_to_remove = random.choice(removable_nodes)
                node.body.remove(node_to_remove)
            ast.fix_missing_locations(node)
        except Exception as e:
            logging.error(f"Error in LayerRemover: {e}")
        return node

    def is_removable_layer(self, stmt):
        return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and
                isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == 'add' and
                isinstance(stmt.value.args[0], ast.Call) and isinstance(stmt.value.args[0].func, ast.Attribute) and
                stmt.value.args[0].func.attr in ['Dropout', 'Dense', 'Activation'])


class LayerTypeSwapper(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        try:
            lstm_nodes, gru_nodes = self.collect_layer_nodes(node.body)
            if lstm_nodes:
                random.choice(lstm_nodes).func.attr = 'GRU'
            if gru_nodes:
                random.choice(gru_nodes).func.attr = 'LSTM'
            ast.fix_missing_locations(node)
        except Exception as e:
            logging.error(f"Error in LayerTypeSwapper: {e}")
        return node

    def collect_layer_nodes(self, body):
        lstm_nodes = []
        gru_nodes = []
        for stmt in body:
            if self.is_add_layer_call(stmt):
                layer = stmt.value.args[0]
                self.find_layers(layer, lstm_nodes, gru_nodes)
        return lstm_nodes, gru_nodes

    def find_layers(self, layer, lstm_nodes, gru_nodes):
        if isinstance(layer, ast.Call):
            if hasattr(layer.func, 'attr'):
                if layer.func.attr == 'LSTM':
                    lstm_nodes.append(layer)
                elif layer.func.attr == 'GRU':
                    gru_nodes.append(layer)
            elif hasattr(layer.func, 'id') and layer.func.id == 'Bidirectional':
                self.find_layers(layer.args[0], lstm_nodes, gru_nodes)


class LayerUnitModifier(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        try:
            layer_nodes = self.collect_layer_nodes(node.body)
            if layer_nodes:
                random_layer = random.choice(layer_nodes)
                random_units = random.randint(1, 256)
                if 'args' in dir(random_layer) and len(random_layer.args) > 0:
                    random_layer.args[0] = ast.Constant(value=random_units)
            ast.fix_missing_locations(node)
        except Exception as e:
            logging.error(f"Error in LayerUnitModifier: {e}")
        return node

    def collect_layer_nodes(self, body):
        layer_nodes = []
        for stmt in body:
            if self.is_add_layer_call(stmt):
                layer = stmt.value.args[0]
                self.find_layers(layer, layer_nodes)
        return layer_nodes

    def find_layers(self, layer, layer_nodes):
        if isinstance(layer, ast.Call):
            if hasattr(layer.func, 'attr'):
                if layer.func.attr in ['LSTM', 'GRU']:
                    layer_nodes.append(layer)
            elif hasattr(layer.func, 'id') and layer.func.id == 'Bidirectional':
                self.find_layers(layer.args[0], layer_nodes)


def transform_code(source_code):
    try:
        tree = ast.parse(source_code)

        # Applying transformations
        transformers = [
            LayerInserter(),
            LayerRemover(),
            LayerTypeSwapper(),
            LayerUnitModifier()
        ]

        for transformer in transformers:
            tree = transformer.visit(tree)

        ast.fix_missing_locations(tree)

        transformed_code = ast.unparse(tree)
        return transformed_code

    except Exception as e:
        logging.error(f"Error in transforming code: {e}")
        return None
