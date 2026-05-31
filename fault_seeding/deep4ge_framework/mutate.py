"""
Single entry point for applying mutations to DNN programs.

Run from the fault_seeding/ directory.

Usage:
    python3 -m deep4ge_framework.mutate <seed_program.py> --operator <CODE> --iterations <N> [--output-dir <DIR>]
    python3 -m deep4ge_framework.mutate <seed_program.py> --all-layer --iterations <N>
    python3 -m deep4ge_framework.mutate <seed_program.py> --list-operators

Seed programs are part of the Deep4ge dataset on Zenodo (DOI 10.5281/zenodo.20337241).

Examples:
    python3 -m deep4ge_framework.mutate FNN_31556268_correct.py --operator HBS --iterations 5
    python3 -m deep4ge_framework.mutate CNN_37624102_correct.py --operator LCF --iterations 5
    python3 -m deep4ge_framework.mutate rnn_51971180_correct.py --operator OCH --iterations 3
    python3 -m deep4ge_framework.mutate --list-operators
"""

import argparse
import ast
import sys
import os
import importlib
import types

from .config import OPERATOR_REGISTRY, CATEGORIES
from .operators import OPERATOR_CLASSES
from .operators.base import ModifySavePath, ModifyCallbackFilename


# ─── Operator class lookup ────────────────────────────────────────────────────

def _get_operator_class(code: str):
    """Return the operator class for a given 3-letter code."""
    if code not in OPERATOR_CLASSES:
        print(f"Error: Unknown operator '{code}'.")
        print(f"Available operators: {', '.join(sorted(OPERATOR_CLASSES.keys()))}")
        sys.exit(1)
    return OPERATOR_CLASSES[code]


def _extract_model_name(source_code: str) -> str:
    """Extract the argument passed to main() in the source file."""
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "main"):
            if node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value).split(".")[0]
    return "unknown_model"


def _ensure_callback_importable():
    """Register deep4ge_framework/callback.py as the 'CustomCallback' module.

    All 60 seed programs do ``from CustomCallback import EnhancedLoggingCallback``.
    This function makes our callback.py importable under that name so the
    exec'd seed programs find the class without any filesystem hacks.
    """
    if "CustomCallback" not in sys.modules:
        callback_path = os.path.join(os.path.dirname(__file__), "callback.py")
        spec = importlib.util.spec_from_file_location("CustomCallback", callback_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["CustomCallback"] = mod
        spec.loader.exec_module(mod)


def run_mutation(filepath: str, operator_code: str, iterations: int, output_dir: str):
    """Apply a mutation operator to a seed program for N iterations."""
    _ensure_callback_importable()

    with open(filepath, "r") as f:
        source = f.read()

    model_name = _extract_model_name(source)
    os.makedirs(output_dir, exist_ok=True)

    OpClass = _get_operator_class(operator_code)

    for i in range(1, iterations + 1):
        print(f"[{operator_code}] Iteration {i}/{iterations} on {model_name}")
        tree = ast.parse(source)

        suffix = f"{model_name}_{operator_code}_{i}"
        tree = ModifySavePath(suffix).visit(tree)
        tree = ModifyCallbackFilename(os.path.join(output_dir, f"{suffix}.csv")).visit(tree)

        op = OpClass()
        tree = op.visit(tree)
        ast.fix_missing_locations(tree)

        code = compile(tree, filename="<ast>", mode="exec")
        exec(code, {"__name__": "__main__"})


def list_operators():
    """Print all registered operators."""
    print(f"{'Code':<6} {'Name':<35} {'Category':<18} {'Architectures'}")
    print("-" * 85)
    for code, (name, category, archs) in sorted(OPERATOR_REGISTRY.items()):
        print(f"{code:<6} {name:<35} {category:<18} {', '.join(archs)}")


def main():
    parser = argparse.ArgumentParser(
        description="Deep4ge: Apply mutation operators to DNN programs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file", nargs="?", help="Path to the seed DNN program (.py)")
    parser.add_argument("--operator", "-op", help="3-letter operator code (e.g., LKS, HBS)")
    parser.add_argument("--iterations", "-n", type=int, default=5, help="Number of mutation iterations (default: 5)")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory for CSVs and models")
    parser.add_argument("--all-layer", action="store_true", help="Run all layer operators")
    parser.add_argument("--list-operators", action="store_true", help="List all registered operators and exit")

    args = parser.parse_args()

    if args.list_operators:
        list_operators()
        return

    if not args.file:
        parser.error("A seed program file is required (unless using --list-operators)")

    if args.all_layer:
        for code in CATEGORIES["Layer"]:
            try:
                run_mutation(args.file, code, args.iterations, args.output_dir)
            except Exception as e:
                print(f"Warning: {code} failed on {args.file}: {e}")
    elif args.operator:
        run_mutation(args.file, args.operator, args.iterations, args.output_dir)
    else:
        parser.error("Either --operator <CODE> or --all-layer is required")


if __name__ == "__main__":
    main()
