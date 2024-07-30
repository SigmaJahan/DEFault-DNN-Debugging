import argparse
import ast
import sys
import logging
import warnings
from convolution_operators import (
    DenseLayerCounter, KernelTransformer, FilterTransformer,
    PoolingTransformer, StridesTransformer, PaddingTransformer,
    NeuronTransformer, CNNCheck
)
from rnn_operators import LayerInserter, LayerRemover, LayerTypeSwapper, LayerUnitModifier

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ModifySaveVisitor(ast.NodeTransformer):
    def __init__(self, new_suffix):
        self.new_suffix = new_suffix

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'save':
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'model':
                new_path = f"models/{self.new_suffix}.h5"
                node.args[0] = ast.Str(s=new_path) if sys.version_info < (3, 8) else ast.Constant(value=new_path)
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


class ModifyCallbackFilenameVisitor(ast.NodeTransformer):
    def __init__(self, new_filename):
        self.new_filename = new_filename

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'callback_filename':
                node.value = ast.Str(s=self.new_filename) if sys.version_info < (3, 8) else ast.Constant(
                    value=self.new_filename)
        ast.fix_missing_locations(node)
        return self.generic_visit(node)


def parse_args():
    parser = argparse.ArgumentParser(description="Parse and execute a Python file after AST reconstruction.")
    parser.add_argument("file", type=str, help="The Python file to be processed.")
    parser.add_argument("--modify_kernel_size", action='store_true', help="Modify kernel sizes.")
    parser.add_argument("--modify_filter_size", action='store_true', help="Modify filter sizes.")
    parser.add_argument("--modify_pooling_size", action='store_true', help="Modify pooling sizes.")
    parser.add_argument("--modify_strides_size", action='store_true', help="Modify strides sizes.")
    parser.add_argument("--modify_padding", action='store_true', help="Modify padding.")
    parser.add_argument("--modify_neurons", action='store_true', help="Modify neurons.")
    parser.add_argument('--all_cnn', action='store_true', help='Apply all CNN modifications.')
    parser.add_argument('--all_rnn', action='store_true', help='Apply all RNN modifications.')
    parser.add_argument('--add_layer', action='store_true', help='Add a layer to the model.')
    parser.add_argument('--remove_layer', action='store_true', help='Remove a layer from the model.')
    parser.add_argument('--change_layer_type', action='store_true', help='Change layer type in the model.')
    parser.add_argument('--change_output_shape', action='store_true',
                        help='Change output shape of a layer in the model.')
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations to run the model for.")
    return parser.parse_args()


def read_python_file(filename):
    with open(filename, "r") as file:
        return file.read()


def parse_code_to_ast(code):
    return ast.parse(code)


def execute_ast(ast_node):
    code = compile(ast_node, filename="<ast>", mode="exec")
    exec(code, globals())


def extract_main_argument(ast_node):
    for node in ast.walk(ast_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'main':
            if node.args and isinstance(node.args[0], (ast.Str, ast.Constant)):
                return node.args[0].s if isinstance(node.args[0], ast.Str) else node.args[0].value


def transform_and_execute(args, transformer, suffix, csv_suffix):
    for i in range(args.iterations):
        logging.info(f"Running iteration {i + 1}")
        code = read_python_file(args.file)
        ast_node = parse_code_to_ast(code)
        argument_passed_to_main = extract_main_argument(ast_node)
        argument_passed_to_main = argument_passed_to_main.split('.')[0]
        modify_save_visitor = ModifySaveVisitor(f'{argument_passed_to_main}_{suffix}_{i + 1}')
        ast_node = modify_save_visitor.visit(ast_node)
        modify_callback_visitor = ModifyCallbackFilenameVisitor(f'{argument_passed_to_main}_{csv_suffix}_{i + 1}.csv')
        ast_node = modify_callback_visitor.visit(ast_node)
        ast_node = transformer.visit(ast_node)
        execute_ast(ast_node)


def main():
    args = parse_args()
    try:
        if args.all_cnn:
            args.modify_kernel_size = True
            args.modify_filter_size = True
            args.modify_pooling_size = True
            args.modify_strides_size = True
            args.modify_padding = True
            args.modify_neurons = True
            args.add_layer = True
            args.remove_layer = True

        if args.all_rnn:
            args.modify_neurons = True
            args.add_layer = True
            args.remove_layer = True
            args.change_layer_type = True
            args.change_output_shape = True

        if args.modify_kernel_size:
            transform_and_execute(args, KernelTransformer(args.modify_kernel_size), 'LKS', 'LKS')

        if args.modify_filter_size:
            transform_and_execute(args, FilterTransformer(args.modify_filter_size), 'LCF', 'LCF')

        if args.modify_pooling_size:
            transform_and_execute(args, PoolingTransformer(args.modify_pooling_size), 'LCP', 'LCP')

        if args.modify_strides_size:
            transform_and_execute(args, StridesTransformer(args.modify_strides_size), 'LCS', 'LCS')

        if args.modify_padding:
            transform_and_execute(args, PaddingTransformer(args.modify_padding), 'LCD', 'LCD')

        if args.modify_neurons:
            for i in range(args.iterations):
                logging.info(f"Running iteration {i + 1}")
                code = read_python_file(args.file)
                ast_node = parse_code_to_ast(code)
                root = ast.parse(code)
                counter = DenseLayerCounter()
                argument_passed_to_main = extract_main_argument(ast_node)
                argument_passed_to_main = argument_passed_to_main.split('.')[0]
                modify_save_visitor = ModifySaveVisitor(f'{argument_passed_to_main}_LCN_{i + 1}')
                ast_node = modify_save_visitor.visit(ast_node)
                modify_callback_visitor = ModifyCallbackFilenameVisitor(f'{argument_passed_to_main}_LCN_{i + 1}.csv')
                ast_node = modify_callback_visitor.visit(ast_node)
                transformer = NeuronTransformer(args.modify_neurons, counter.count_dense_layers(root) - 1)
                ast_node = transformer.visit(ast_node)
                execute_ast(ast_node)

        if args.add_layer:
            transform_and_execute(args, LayerInserter(), 'LAD', 'LAD')

        if args.remove_layer:
            transform_and_execute(args, LayerRemover(), 'LRM', 'LRM')

        if args.change_layer_type:
            transform_and_execute(args, LayerTypeSwapper(), 'LCT', 'LCT')

        if args.change_output_shape:
            transform_and_execute(args, LayerUnitModifier(), 'LCO', 'LCO')

    except FileNotFoundError:
        logging.error(f"Error: The file '{args.file}' does not exist.")
        sys.exit(1)


if __name__ == "__main__":
    main()
