"""
Compute and display summary statistics of the Deep4ge dataset.

Reads manifest.csv and prints tables suitable for inclusion in the paper/README.
"""

import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deep4ge_framework.config import CATEGORIES


def load_manifest(manifest_path):
    with open(manifest_path, "r") as f:
        return list(csv.DictReader(f))


def load_seed_architectures(data_dir):
    seed_metadata = os.path.join(data_dir, "seed_programs", "seed_metadata.csv")
    if os.path.exists(seed_metadata):
        with open(seed_metadata, "r") as f:
            return {
                row["so_id"]: row["architecture"].upper()
                for row in csv.DictReader(f)
                if row.get("so_id") and row.get("architecture")
            }

    # Fallback for older layouts without seed_metadata.csv.
    arch_by_so = {}
    for arch in ["fnn", "cnn", "rnn"]:
        prog_dir = os.path.join(data_dir, "seed_programs", arch)
        if not os.path.isdir(prog_dir):
            continue
        for filename in os.listdir(prog_dir):
            import re
            m = re.search(r"(\d{8,})", filename)
            if m:
                arch_by_so[m.group(1)] = arch.upper()
    return arch_by_so


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def compute_stats(data_dir):
    manifest_path = os.path.join(data_dir, "manifest.csv")
    rows = load_manifest(manifest_path)

    buggy = [r for r in rows if r["subset"] == "buggy"]
    correct = [r for r in rows if r["subset"] == "correct"]

    # ─── Overall Summary ─────────────────────────────────────────
    print_section("DATASET OVERVIEW")
    print(f"  Total CSV files:      {len(rows):,}")
    print(f"    Buggy (faulty):     {len(buggy):,}")
    print(f"    Correct (baseline): {len(correct):,}")

    # ─── Unique SO IDs ───────────────────────────────────────────
    so_ids = set(r["so_id"] for r in rows if r["so_id"])
    print_section("SEED PROGRAMS")
    print(f"  Unique SO IDs:       {len(so_ids)}")

    for arch in ["fnn", "cnn", "rnn"]:
        prog_dir = os.path.join(data_dir, "seed_programs", arch)
        if os.path.isdir(prog_dir):
            count = len([f for f in os.listdir(prog_dir) if f.endswith(".py")])
            print(f"  {arch.upper()} programs:       {count}")

    # ─── Fault Category Distribution ─────────────────────────────
    print_section("FAULT CATEGORY DISTRIBUTION (Buggy subset)")
    cat_counts = Counter(r["fault_category"] for r in buggy)
    print(f"  {'Category':<20} {'Count':>8} {'%':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8}")
    for cat, count in cat_counts.most_common():
        pct = count / len(buggy) * 100
        print(f"  {cat:<20} {count:>8,} {pct:>7.1f}%")

    # ─── Per-SO_ID Distribution ──────────────────────────────────
    print_section("TOP SO IDs BY FILE COUNT")
    so_counts = Counter(r["so_id"] for r in rows)
    print(f"  {'SO ID':<12} {'Total':>8} {'Buggy':>8} {'Correct':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    buggy_by_so = Counter(r["so_id"] for r in buggy)
    correct_by_so = Counter(r["so_id"] for r in correct)
    for so_id, total in so_counts.most_common(10):
        print(f"  {so_id:<12} {total:>8,} {buggy_by_so[so_id]:>8,} {correct_by_so[so_id]:>8,}")

    # ─── Epoch Distribution ──────────────────────────────────────
    print_section("EPOCH DISTRIBUTION")
    epochs = [int(r["num_epochs"]) for r in rows]
    print(f"  Total logs:           {len(epochs):,}")
    print(f"  Min epochs:           {min(epochs)}")
    print(f"  Max epochs:           {max(epochs)}")
    print(f"  Mean epochs:          {sum(epochs)/len(epochs):.1f}")
    sorted_e = sorted(epochs)
    mid = len(sorted_e) // 2
    median = sorted_e[mid] if len(sorted_e) % 2 else (sorted_e[mid-1] + sorted_e[mid]) / 2
    print(f"  Median epochs:        {median:.0f}")

    # Epoch distribution histogram
    buckets = [(1, 10), (11, 50), (51, 100), (101, 500), (501, 1000)]
    print(f"\n  {'Range':<15} {'Count':>8}")
    print(f"  {'-'*15} {'-'*8}")
    for low, high in buckets:
        count = sum(1 for e in epochs if low <= e <= high)
        print(f"  {f'{low}-{high}':<15} {count:>8,}")

    # ─── Category x SO cross-tab ────────────────────────────────
    print_section("FAULT CATEGORIES x ARCHITECTURES")
    arch_by_so = load_seed_architectures(data_dir)

    print(f"  {'Category':<20} {'CNN':>8} {'FNN':>8} {'RNN':>8} {'Total':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cat, count in cat_counts.most_common():
        cat_rows = [r for r in buggy if r["fault_category"] == cat]
        arch_counts = Counter(arch_by_so.get(r["so_id"], "UNKNOWN") for r in cat_rows)
        print(
            f"  {cat:<20} {arch_counts['CNN']:>8,} {arch_counts['FNN']:>8,} "
            f"{arch_counts['RNN']:>8,} {count:>8,}"
        )


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    compute_stats(data_dir)
