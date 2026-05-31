import argparse
import ast
import sys
import warnings
warnings.filterwarnings("ignore")
import uuid

class ModifySaveVisitor(ast.NodeTransformer):
    def __init__(self, new_suffix):
        self.new_suffix = new_suffix  # Store the new_suffix as an attribute

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'save':
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'model':
                    new_path = f"models/{self.new_suffix}.h5"
                    node.args[0] = ast.Str(s=new_path)
        ast.fix_missing_locations(node)
        return self.generic_visit(node)
    
class ModifyCallbackFilenameVisitor(ast.NodeTransformer):
    def __init__(self, new_filename):
        self.new_filename = new_filename  # Store the new filename as an attribute

    def visit_Assign(self, node):
        # Check if the target of the assignment is the variable 'callback_filename'
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'callback_filename':
                # Update the value to the new filename
                node.value = ast.Str(s=self.new_filename)  # Use ast.Constant(value=self.new_filename) for Python 3.8+
        ast.fix_missing_locations(node)
        return self.generic_visit(node)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Parse and execute a Python file after AST reconstruction.")
    parser.add_argument("file", type=str, help="The Python file to be processed.")
    parser.add_argument("--iterations", type=int, help="Number of iterations to run the model for")
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
            # Extract the argument passed to `main`
            if node.args and isinstance(node.args[0], (ast.Str, ast.Constant)):
                return node.args[0].s if isinstance(node.args[0], ast.Str) else node.args[0].value

def main():
    args = parse_args()
    try:
        for i in range(args.iterations):
            print (f"Running iteration {i+1}")
            code = read_python_file(args.file)
            ast_node = parse_code_to_ast(code)
            argument_passed_to_main = extract_main_argument(ast_node)
            argument_passed_to_main = argument_passed_to_main.split('.')[0]
            modifySaveVisitor = ModifySaveVisitor(f'{argument_passed_to_main}_correct_' + str(i+1))
            ast_node = modifySaveVisitor.visit(ast_node)
            modifyCallbackFilenameVisitor = ModifyCallbackFilenameVisitor(f'{argument_passed_to_main}_correct_' + str(i+1) + '.csv')
            ast_node = modifyCallbackFilenameVisitor.visit(ast_node)
            execute_ast(ast_node)
    except FileNotFoundError:
        print(f"Error: The file '{args.file}' does not exist.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()