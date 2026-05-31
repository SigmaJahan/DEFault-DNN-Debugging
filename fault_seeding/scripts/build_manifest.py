"""
Build manifest.csv from the data/training_logs/ directory structure.

Scans all CSV files in buggy/ and correct/ subdirectories, parses the
standardized filenames, and produces a single manifest.csv.

Filename conventions:
  Buggy:   {so_id}_{fault_category}_{run_number:04d}.csv
  Correct: {so_id}_correct_{run_number:04d}.csv
"""

import csv
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deep4ge_framework.config import CSV_COLUMNS, CATEGORIES


VALID_CATEGORIES = set(CATEGORIES.keys()) | {"none"}


def count_epochs(csv_path):
    """Count data rows in a CSV (excluding header)."""
    try:
        with open(csv_path, "r") as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return -1


def parse_buggy_filename(filename):
    """Parse: {so_id}_{category}_{run:04d}.csv"""
    name = filename.replace(".csv", "")
    m = re.match(r"^(\d+)_(\w+)_(\d{4})$", name)
    if m:
        return {
            "so_id": m.group(1),
            "fault_category": m.group(2),
            "run_number": int(m.group(3)),
            "is_faulty": True,
        }
    return None


def parse_correct_filename(filename):
    """Parse: {so_id}_correct_{run:04d}.csv"""
    name = filename.replace(".csv", "")
    m = re.match(r"^(\d+)_correct_(\d{4})$", name)
    if m:
        return {
            "so_id": m.group(1),
            "fault_category": "none",
            "run_number": int(m.group(2)),
            "is_faulty": False,
        }
    return None


def build_manifest(data_dir, output_path):
    """Scan data/training_logs/ and build manifest.csv."""
    rows = []
    parse_failures = []

    for subset, parser in [("buggy", parse_buggy_filename), ("correct", parse_correct_filename)]:
        subset_dir = os.path.join(data_dir, "training_logs", subset)
        if not os.path.isdir(subset_dir):
            print(f"Warning: {subset_dir} not found, skipping")
            continue

        for filename in sorted(os.listdir(subset_dir)):
            if not filename.endswith(".csv"):
                continue

            csv_path = os.path.join(subset_dir, filename)
            rel_path = os.path.relpath(csv_path, os.path.join(data_dir, ".."))

            meta = parser(filename)
            if meta is None:
                parse_failures.append(filename)
                continue

            meta["filename"] = filename
            meta["subset"] = subset
            meta["csv_path"] = rel_path
            meta["num_epochs"] = count_epochs(csv_path)
            rows.append(meta)

    # Write manifest
    fieldnames = [
        "filename", "subset", "so_id", "fault_category",
        "is_faulty", "run_number", "num_epochs", "csv_path",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"Manifest written to {output_path}")
    print(f"  Buggy:   {sum(1 for r in rows if r['subset'] == 'buggy'):,}")
    print(f"  Correct: {sum(1 for r in rows if r['subset'] == 'correct'):,}")
    print(f"  Total:   {len(rows):,}")

    # Category distribution
    cats = Counter(r["fault_category"] for r in rows if r["is_faulty"])
    print(f"\n  Fault category distribution:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count:,}")

    # Unique SO IDs
    so_ids = set(r["so_id"] for r in rows)
    print(f"\n  Unique SO IDs: {len(so_ids)}")

    # Epoch stats
    epochs = [r["num_epochs"] for r in rows if r["num_epochs"] > 0]
    if epochs:
        print(f"\n  Epoch stats:")
        print(f"    Min: {min(epochs)}, Max: {max(epochs)}, Median: {sorted(epochs)[len(epochs)//2]}")

    if parse_failures:
        print(f"\n  WARNING: {len(parse_failures)} files failed to parse:")
        for f in parse_failures[:10]:
            print(f"    {f}")

    return rows


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output = os.path.join(data_dir, "manifest.csv")
    build_manifest(data_dir, output)
