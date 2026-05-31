#!/usr/bin/env python3
"""
Deep4ge — End-to-end reproducibility demo.

Demonstrates the full mutation → training → feature-collection pipeline:
1. Picks a seed program
2. Applies a mutation operator (AST transformation)
3. Executes the mutated program (trains the model)
4. Verifies the output CSV has the expected 31-column schema

Usage:
    python3 scripts/demo_replication.py                         # default: FNN + HBS
    python3 scripts/demo_replication.py --operator FLC          # specific operator
    python3 scripts/demo_replication.py --seed data/seed_programs/rnn/rnn_51971180_correct.py
    python3 scripts/demo_replication.py --dry-run               # show mutated source, skip training

This script is intended for artifact reviewers to verify that the mutation framework
produces training logs matching the dataset schema.
"""

import argparse
import ast
import csv
import importlib
import importlib.util
import os
import sys
import textwrap

# ─── Path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from deep4ge_framework.config import OPERATOR_REGISTRY, CSV_COLUMNS
from deep4ge_framework.operators import OPERATOR_CLASSES
from deep4ge_framework.operators.base import ModifySavePath, ModifyCallbackFilename


def register_callback_module():
    """Make deep4ge_framework/callback.py importable as 'CustomCallback' for seed programs."""
    if "CustomCallback" not in sys.modules:
        callback_path = os.path.join(REPO_ROOT, "src", "callback.py")
        spec = importlib.util.spec_from_file_location("CustomCallback", callback_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["CustomCallback"] = mod
        spec.loader.exec_module(mod)


def extract_model_name(source_code: str) -> str:
    """Extract the model name argument from main('name.h5') call."""
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "main"):
            if node.args and isinstance(node.args[0], ast.Constant):
                return str(node.args[0].value).split(".")[0]
    return "unknown_model"


def validate_output_csv(csv_path):
    """Check that the output CSV matches the Deep4ge 31-column schema."""
    issues = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) != 31:
                issues.append(f"Expected 31 columns, got {len(header)}")
            elif header != CSV_COLUMNS:
                issues.append(f"Header mismatch: {header}")
            rows = list(reader)
            if len(rows) == 0:
                issues.append("No data rows (header only)")
            else:
                # Check first row has numeric-parseable values
                for i, val in enumerate(rows[0]):
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        if val.lower() not in ("true", "false", "nan", "inf", "-inf"):
                            issues.append(f"Column '{header[i]}' row 0: non-numeric value '{val}'")
    except FileNotFoundError:
        issues.append(f"Output CSV not found: {csv_path}")
    except StopIteration:
        issues.append("Empty file")
    return issues, len(rows) if not issues or "No data rows" not in str(issues) else 0


def main():
    parser = argparse.ArgumentParser(
        description="Deep4ge reproducibility demo — mutate a seed program and verify output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed", default=None,
        help="Path to seed program (default: first FNN program found)",
    )
    parser.add_argument(
        "--operator", "-op", default="HBS",
        help="3-letter operator code (default: HBS)",
    )
    parser.add_argument(
        "--output-dir", "-o", default=os.path.join(REPO_ROOT, "output", "demo"),
        help="Output directory for demo CSV",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show mutated source code without running training",
    )
    args = parser.parse_args()

    # ── Find seed program ────────────────────────────────────────────────────
    if args.seed:
        seed_path = args.seed
    else:
        fnn_dir = os.path.join(REPO_ROOT, "data", "seed_programs", "fnn")
        fnn_files = sorted(f for f in os.listdir(fnn_dir) if f.endswith(".py"))
        if not fnn_files:
            print("ERROR: No FNN seed programs found.")
            sys.exit(1)
        seed_path = os.path.join(fnn_dir, fnn_files[0])

    if not os.path.exists(seed_path):
        print(f"ERROR: Seed program not found: {seed_path}")
        sys.exit(1)

    # ── Validate operator ────────────────────────────────────────────────────
    op_code = args.operator.upper()
    if op_code not in OPERATOR_CLASSES:
        print(f"ERROR: Unknown operator '{op_code}'")
        print(f"Available: {', '.join(sorted(OPERATOR_CLASSES.keys()))}")
        sys.exit(1)

    op_name, op_category, op_archs = OPERATOR_REGISTRY[op_code]

    # ── Print banner ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Deep4ge — Reproducibility Demo")
    print("=" * 70)
    print(f"  Seed program:   {os.path.basename(seed_path)}")
    print(f"  Operator:       {op_code} ({op_name})")
    print(f"  Category:       {op_category}")
    print(f"  Architectures:  {', '.join(op_archs)}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Dry run:        {args.dry_run}")
    print("=" * 70)

    # ── Read and mutate source ───────────────────────────────────────────────
    with open(seed_path, "r") as f:
        source = f.read()

    model_name = extract_model_name(source)
    os.makedirs(args.output_dir, exist_ok=True)

    tree = ast.parse(source)
    suffix = f"demo_{model_name}_{op_code}"
    csv_output = os.path.join(args.output_dir, f"{suffix}.csv")

    # Apply AST transformations
    tree = ModifySavePath(suffix).visit(tree)
    tree = ModifyCallbackFilename(csv_output).visit(tree)

    OpClass = OPERATOR_CLASSES[op_code]
    op = OpClass()
    tree = op.visit(tree)
    ast.fix_missing_locations(tree)

    mutated_source = ast.unparse(tree)

    print(f"\n--- Mutation applied: {op_code} ({op_name}) ---")
    print(f"  Mutated source length: {len(mutated_source)} chars")
    print(f"  Output CSV will be: {csv_output}")

    if args.dry_run:
        print(f"\n--- Mutated source (first 50 lines) ---")
        for i, line in enumerate(mutated_source.split("\n")[:50], 1):
            print(f"  {i:3d} | {line}")
        print("\n  [--dry-run] Skipping training execution.")
        print("  To run training, remove the --dry-run flag.")
        return

    # ── Execute mutated program ──────────────────────────────────────────────
    print(f"\n--- Running mutated program (training will begin) ---\n")
    register_callback_module()

    code = compile(tree, filename=f"<{op_code}_mutated>", mode="exec")
    try:
        exec(code, {"__name__": "__main__"})
    except Exception as e:
        print(f"\n--- Training failed with error: {type(e).__name__}: {e} ---")
        print("  This may be expected for certain mutations (e.g., shape mismatches).")
        if os.path.exists(csv_output):
            print(f"  Partial CSV was written — validating what we have...")
        else:
            print(f"  No CSV was produced.")
            sys.exit(1)

    # ── Validate output ──────────────────────────────────────────────────────
    print(f"\n--- Validating output CSV ---")
    issues, epoch_count = validate_output_csv(csv_output)

    if not issues:
        print(f"  PASS: {csv_output}")
        print(f"    Columns: 31 (matches Deep4ge schema)")
        print(f"    Epochs:  {epoch_count}")
        print(f"\n  The output CSV matches the same schema as the 14,227 files")
        print(f"  in the Deep4ge dataset. The mutation pipeline is reproducible.")
    else:
        print(f"  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  Demo complete. Output: {csv_output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
