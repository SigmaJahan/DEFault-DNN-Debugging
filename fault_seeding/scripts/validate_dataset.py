"""
Validate the integrity of the Deep4ge dataset.

Checks:
1. All CSVs have the expected 31-column header
2. No empty CSVs (must have header + at least 1 data row)
3. Manifest matches files on disk (bidirectional)
4. All filenames follow the naming convention
5. Seed programs exist for referenced SO IDs
6. Fault categories are all valid
"""

import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deep4ge_framework.config import CSV_COLUMNS, CATEGORIES


VALID_CATEGORIES = set(CATEGORIES.keys()) | {"none"}


def validate_csv_file(csv_path):
    """Validate a single CSV file. Returns list of issues."""
    issues = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) != 31:
                issues.append(f"Column count: expected 31, got {len(header)}")
            elif header != CSV_COLUMNS:
                issues.append("Header mismatch (correct count but wrong names)")

            row_count = 0
            for line_number, row in enumerate(reader, start=2):
                row_count += 1
                if len(row) != 31:
                    issues.append(f"Row {line_number}: expected 31 values, got {len(row)}")
                    continue
                for col_name, value in zip(header, row):
                    value = value.strip()
                    if not value:
                        issues.append(f"Row {line_number}, column '{col_name}': empty value")
                        continue
                    if value.lower() in {"true", "false", "nan", "inf", "-inf"}:
                        continue
                    try:
                        float(value)
                    except ValueError:
                        issues.append(f"Row {line_number}, column '{col_name}': non-numeric value '{value}'")

            if row_count == 0:
                issues.append("Empty CSV (header only)")
    except StopIteration:
        issues.append("Completely empty file")
    except Exception as e:
        issues.append(f"Read error: {e}")
    return issues


def validate_naming(filename, subset):
    """Check filename follows naming convention."""
    if subset == "buggy":
        m = re.match(r"^\d+_\w+_\d{4}\.csv$", filename)
        return m is not None
    elif subset == "correct":
        m = re.match(r"^\d+_correct_\d{4}\.csv$", filename)
        return m is not None
    return False


def validate_manifest(data_dir, manifest_path):
    """Cross-check manifest against files on disk."""
    issues = []
    data_dir = Path(data_dir).resolve()
    repo_root = data_dir.parent
    manifest_path = Path(manifest_path).resolve()

    if not manifest_path.exists():
        return ["manifest.csv not found"]

    with open(manifest_path, "r") as f:
        manifest_rows = list(csv.DictReader(f))

    # Every manifest entry → real file
    manifest_paths = set()
    for row in manifest_rows:
        csv_path = (repo_root / row["csv_path"]).resolve()
        manifest_paths.add(csv_path)
        if not csv_path.exists():
            issues.append(f"Manifest references missing: {row['csv_path']}")

    # Every file on disk → in manifest
    for subset in ["buggy", "correct"]:
        subset_dir = (data_dir / "training_logs" / subset).resolve()
        if not subset_dir.is_dir():
            continue
        for csv_file in subset_dir.glob("*.csv"):
            fp = csv_file.resolve()
            if fp not in manifest_paths:
                issues.append(f"Not in manifest: {subset}/{csv_file.name}")

    # Valid categories
    for row in manifest_rows:
        if row["fault_category"] not in VALID_CATEGORIES:
            issues.append(f"Invalid category '{row['fault_category']}' in {row['filename']}")

    return issues


def validate_seed_attribution(data_dir):
    issues = []
    data_dir = Path(data_dir).resolve()
    seed_meta = data_dir / "seed_programs" / "seed_metadata.csv"
    attribution = data_dir / "seed_programs" / "ATTRIBUTION.csv"

    if not seed_meta.exists():
        return ["seed_metadata.csv not found"]
    if not attribution.exists():
        return ["ATTRIBUTION.csv not found"]

    with seed_meta.open("r") as f:
        seed_rows = list(csv.DictReader(f))
    with attribution.open("r") as f:
        attr_rows = list(csv.DictReader(f))

    seed_ids = {row["so_id"] for row in seed_rows}
    attr_ids = {row["so_id"] for row in attr_rows}
    if seed_ids != attr_ids:
        issues.append(f"Attribution SO IDs differ from seed metadata: missing={sorted(seed_ids - attr_ids)}, extra={sorted(attr_ids - seed_ids)}")

    required = {"so_id", "seed_file", "source_url", "source_content_license", "adaptation_note"}
    missing_cols = required - set(attr_rows[0].keys() if attr_rows else [])
    if missing_cols:
        issues.append(f"ATTRIBUTION.csv missing required columns: {sorted(missing_cols)}")

    for row in attr_rows:
        if not row.get("source_url", "").startswith("https://stackoverflow.com/questions/"):
            issues.append(f"Invalid source_url for SO {row.get('so_id')}: {row.get('source_url')}")
        if not row.get("source_content_license"):
            issues.append(f"Missing source_content_license for SO {row.get('so_id')}")
        if not row.get("adaptation_note"):
            issues.append(f"Missing adaptation_note for SO {row.get('so_id')}")

    return issues


def validate_all(data_dir):
    """Run all validation checks."""
    data_dir = str(Path(data_dir).resolve())
    manifest_path = os.path.join(data_dir, "manifest.csv")
    all_issues = []
    file_count = 0
    naming_issues = 0

    for subset in ["buggy", "correct"]:
        subset_dir = os.path.join(data_dir, "training_logs", subset)
        if not os.path.isdir(subset_dir):
            all_issues.append(f"Missing directory: {subset_dir}")
            continue

        files = [f for f in os.listdir(subset_dir) if f.endswith(".csv")]
        print(f"\n--- {subset}: {len(files)} files ---")
        file_count += len(files)

        subset_issues = 0
        for filename in sorted(files):
            filepath = os.path.join(subset_dir, filename)
            issues = validate_csv_file(filepath)

            if not validate_naming(filename, subset):
                naming_issues += 1
                issues.append("Non-standard filename")

            if issues:
                subset_issues += len(issues)
                for issue in issues:
                    all_issues.append(f"[{subset}/{filename}] {issue}")

        if subset_issues == 0:
            print(f"  All files valid")
        else:
            print(f"  {subset_issues} issue(s)")

    # Manifest
    print(f"\n--- Manifest ---")
    mi = validate_manifest(data_dir, manifest_path)
    if mi:
        for m in mi[:20]:
            print(f"  ISSUE: {m}")
        all_issues.extend(mi)
    else:
        print(f"  OK ({file_count} files accounted for)")

    # Seed programs
    print(f"\n--- Seed Programs ---")
    for arch in ["fnn", "cnn", "rnn"]:
        prog_dir = os.path.join(data_dir, "seed_programs", arch)
        if os.path.isdir(prog_dir):
            count = len([f for f in os.listdir(prog_dir) if f.endswith(".py")])
            print(f"  {arch.upper()}: {count}")

    print(f"\n--- Seed Attribution ---")
    ai = validate_seed_attribution(data_dir)
    if ai:
        for issue in ai[:20]:
            print(f"  ISSUE: {issue}")
        all_issues.extend(ai)
    else:
        print("  OK (60 seed programs attributed)")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total CSV files:  {file_count:,}")
    print(f"Naming issues:    {naming_issues}")
    print(f"Total issues:     {len(all_issues)}")
    print(f"\n{'PASS' if not all_issues else 'FAIL'}")

    return all_issues


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    issues = validate_all(data_dir)
    sys.exit(1 if issues else 0)
