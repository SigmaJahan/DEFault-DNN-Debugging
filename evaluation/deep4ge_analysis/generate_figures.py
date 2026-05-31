#!/usr/bin/env python3
"""Generate publication-quality figures for the Deep4ge dataset paper.

The script is robust to two common layouts:

(A) You pass the *repo root* (recommended):
    repo_root/
      data/manifest.csv
      data/training_logs/...
(B) You pass the *data directory*:
    data/
      manifest.csv
      training_logs/...

Figures (PDF) written under <out_dir>/figures/:
- category_counts_buggy.pdf
- epochs_hist_buggy_vs_correct.pdf
- arch_log_counts_stacked.pdf
- cat_arch_heatmap_buggy.pdf
- mean_loss_curves_buggy_vs_correct.pdf

Example:
  python3 analysis/generate_figures.py --data-root . --out-dir output/analysis_artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


REQUIRED_MANIFEST_COLUMNS = {"subset", "so_id", "fault_category", "num_epochs", "csv_path"}
SUBSET_ORDER = ("buggy", "correct")
SUBSET_COLORS = {"buggy": "#D55E00", "correct": "#0072B2"}  # colorblind-safe
ARCH_ORDER = ("FNN", "CNN", "RNN", "UNKNOWN")


def configure_plot_style() -> None:
    matplotlib.style.use("default")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finalize_figure(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, metadata={"Creator": "analysis/generate_figures.py"}, facecolor="white")
    plt.close(fig)


def set_integer_count_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Deep4ge dataset figures")
    p.add_argument("--data-root", type=Path, required=True, help="Repo root or data directory.")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--seed-metadata", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-epoch-plot", type=int, default=50, help="Max epochs to plot for mean loss curves.")
    p.add_argument(
        "--loss-upper-quantile",
        type=float,
        default=0.95,
        help="Upper quantile for winsorizing loss values in mean curves (0,1).",
    )
    return p.parse_args()


def infer_paths(data_root: Path, manifest_arg: Optional[Path], seed_meta_arg: Optional[Path]) -> Tuple[Path, Optional[Path]]:
    if manifest_arg is not None:
        manifest_path = manifest_arg
    else:
        c1 = data_root / "manifest.csv"
        c2 = data_root / "data" / "manifest.csv"
        if c1.exists():
            manifest_path = c1
        elif c2.exists():
            manifest_path = c2
        else:
            raise FileNotFoundError(f"Could not find manifest.csv under {data_root} (tried {c1} and {c2}).")

    if seed_meta_arg is not None:
        seed_meta_path: Optional[Path] = seed_meta_arg
    else:
        c1 = data_root / "seed_programs" / "seed_metadata.csv"
        c2 = data_root / "data" / "seed_programs" / "seed_metadata.csv"
        if c1.exists():
            seed_meta_path = c1
        elif c2.exists():
            seed_meta_path = c2
        else:
            seed_meta_path = None

    return manifest_path, seed_meta_path


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    df["subset"] = df["subset"].astype(str).str.lower()
    df["fault_category"] = df["fault_category"].astype(str)
    df["so_id"] = df["so_id"].astype(str)
    df["csv_path"] = df["csv_path"].astype(str)
    return df


def read_seed_metadata(seed_meta_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if seed_meta_path is None or not seed_meta_path.exists():
        return None
    df = pd.read_csv(seed_meta_path)
    if "so_id" not in df.columns or "architecture" not in df.columns:
        return None
    df["so_id"] = df["so_id"].astype(str)
    df["architecture"] = df["architecture"].astype(str).str.upper()
    return df


def infer_repo_root(manifest_path: Path, manifest_df: pd.DataFrame) -> Path:
    sample = None
    for v in manifest_df["csv_path"].head(50).tolist():
        if isinstance(v, str) and v.strip():
            sample = v
            break
    if sample is None:
        return manifest_path.parent

    candidates = [
        manifest_path.parent,           # e.g., repo_root/data
        manifest_path.parent.parent,    # e.g., repo_root
        manifest_path.parent.parent.parent,
    ]
    for c in candidates:
        if (c / sample).exists():
            return c
    return manifest_path.parent


def resolve_log_path(repo_root: Path, csv_path_value: str) -> Path:
    p = Path(csv_path_value)
    return p if p.is_absolute() else (repo_root / p)


def fig_category_counts_buggy(df: pd.DataFrame, out_path: Path) -> None:
    buggy = df[df["subset"] == "buggy"].copy()
    counts = buggy["fault_category"].value_counts().sort_values(ascending=False)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(counts))
    bars = ax.bar(
        x,
        counts.values,
        color="#4C78A8",
        edgecolor="#2B2B2B",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index, rotation=18, ha="right")
    ax.set_xlabel("Fault category")
    ax.set_ylabel("Number of buggy logs")
    set_integer_count_axis(ax)
    ax.set_ylim(0, counts.max() * 1.12)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + counts.max() * 0.015,
            f"{int(val):,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    finalize_figure(fig, out_path)


def fig_epochs_hist(df: pd.DataFrame, out_path: Path) -> None:
    buggy_epochs = pd.to_numeric(df[df["subset"] == "buggy"]["num_epochs"], errors="coerce").dropna()
    corr_epochs = pd.to_numeric(df[df["subset"] == "correct"]["num_epochs"], errors="coerce").dropna()
    if buggy_epochs.empty and corr_epochs.empty:
        return

    all_epochs = pd.concat([buggy_epochs, corr_epochs], ignore_index=True)
    raw_upper = int(all_epochs.max())
    vis_upper = int(max(1, np.quantile(all_epochs, 0.995)))
    vis_upper = max(vis_upper, 50)
    upper = min(raw_upper, vis_upper)
    buggy_overflow = int((buggy_epochs > upper).sum())
    corr_overflow = int((corr_epochs > upper).sum())

    buggy_plot = buggy_epochs[buggy_epochs <= upper]
    corr_plot = corr_epochs[corr_epochs <= upper]
    bin_width = 5 if upper <= 150 else 10
    bins = np.arange(0, upper + bin_width + 1, bin_width)
    if bins.size < 3:
        bins = np.array([0.0, 1.0, 2.0])

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.hist(
        buggy_plot,
        bins=bins,
        alpha=0.50,
        color=SUBSET_COLORS["buggy"],
        edgecolor="white",
        linewidth=0.5,
        label=f"Buggy (n={len(buggy_epochs):,})",
    )
    ax.hist(
        corr_plot,
        bins=bins,
        alpha=0.50,
        color=SUBSET_COLORS["correct"],
        edgecolor="white",
        linewidth=0.5,
        label=f"Correct (n={len(corr_epochs):,})",
    )

    if not buggy_epochs.empty:
        ax.axvline(float(buggy_epochs.median()), color=SUBSET_COLORS["buggy"], linewidth=1.2, linestyle="-")
    if not corr_epochs.empty:
        ax.axvline(float(corr_epochs.median()), color=SUBSET_COLORS["correct"], linewidth=1.2, linestyle="-")

    ax.set_xlabel("Epochs per training log")
    ax.set_ylabel("Number of logs (log scale)")
    ax.set_xlim(0, upper)
    ax.set_yscale("log")
    ax.set_ylim(1, None)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    if buggy_overflow or corr_overflow:
        ax.text(
            0.01,
            0.97,
            f"Clipped for readability at {upper} epochs; overflow logs: buggy={buggy_overflow}, correct={corr_overflow}.",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="#444444",
        )

    finalize_figure(fig, out_path)


def fig_arch_log_counts(df: pd.DataFrame, seed_meta_df: Optional[pd.DataFrame], out_path: Path) -> None:
    if seed_meta_df is None:
        return

    so_to_arch = seed_meta_df.set_index("so_id")["architecture"].to_dict()
    tmp = df.copy()
    tmp["architecture"] = tmp["so_id"].map(so_to_arch).fillna("UNKNOWN")

    counts = tmp.groupby(["architecture", "subset"]).size().unstack(fill_value=0)
    if counts.empty:
        return

    arch_order = [a for a in ARCH_ORDER if a in counts.index] + [a for a in counts.index if a not in ARCH_ORDER]
    counts = counts.loc[arch_order]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(counts.index))
    bottom = np.zeros(len(counts), dtype=float)

    for subset in SUBSET_ORDER:
        vals = counts[subset].to_numpy(dtype=float) if subset in counts.columns else np.zeros(len(counts), dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.66,
            label=subset.capitalize(),
            color=SUBSET_COLORS[subset],
            edgecolor="#2B2B2B",
            linewidth=0.5,
            hatch="///" if subset == "correct" else None,
        )
        bottom += vals

    max_total = max(float(bottom.max()), 1.0)
    for x_i, total in zip(x, bottom):
        ax.text(x_i, total + max_total * 0.015, f"{int(total):,}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Number of logs")
    set_integer_count_axis(ax)
    ax.set_ylim(0, max_total * 1.14)
    ax.legend(loc="upper left")

    finalize_figure(fig, out_path)


def fig_cat_arch_heatmap_buggy(df: pd.DataFrame, seed_meta_df: Optional[pd.DataFrame], out_path: Path) -> None:
    if seed_meta_df is None:
        return

    so_to_arch = seed_meta_df.set_index("so_id")["architecture"].to_dict()
    buggy = df[df["subset"] == "buggy"].copy()
    buggy["architecture"] = buggy["so_id"].map(so_to_arch).fillna("UNKNOWN")

    pivot = buggy.pivot_table(
        index="fault_category",
        columns="architecture",
        values="csv_path",
        aggfunc="count",
        fill_value=0,
    )
    if pivot.empty:
        return

    cols = [c for c in ARCH_ORDER if c in pivot.columns] + [c for c in pivot.columns if c not in ARCH_ORDER]
    rows = buggy["fault_category"].value_counts().index.tolist()
    rows = [r for r in rows if r in pivot.index] + [r for r in pivot.index if r not in rows]
    pivot = pivot.loc[rows, cols]

    fig_h = max(3.2, 1.5 + 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(6.8, fig_h))

    vmax = float(max(1, int(pivot.to_numpy().max())))
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest", cmap="cividis", vmin=0, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.set_ylabel("Number of buggy logs", rotation=90, va="bottom")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Fault category")

    threshold = vmax * 0.55
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = int(pivot.iat[i, j])
            txt_color = "white" if val >= threshold else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", fontsize=7, color=txt_color)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    finalize_figure(fig, out_path)


def fig_mean_loss_curves(
    df: pd.DataFrame,
    repo_root: Path,
    out_path: Path,
    max_epoch: int = 50,
    loss_upper_quantile: float = 0.95,
) -> None:
    if max_epoch <= 0:
        return
    if not (0.0 < loss_upper_quantile < 1.0):
        raise ValueError("--loss-upper-quantile must be in (0,1)")

    sum_loss: Dict[str, Dict[str, np.ndarray]] = {
        "buggy": {"train_loss": np.zeros(max_epoch), "val_loss": np.zeros(max_epoch)},
        "correct": {"train_loss": np.zeros(max_epoch), "val_loss": np.zeros(max_epoch)},
    }
    sumsq_loss: Dict[str, Dict[str, np.ndarray]] = {
        "buggy": {"train_loss": np.zeros(max_epoch), "val_loss": np.zeros(max_epoch)},
        "correct": {"train_loss": np.zeros(max_epoch), "val_loss": np.zeros(max_epoch)},
    }
    cnt_loss: Dict[str, Dict[str, np.ndarray]] = {
        "buggy": {"train_loss": np.zeros(max_epoch, dtype=int), "val_loss": np.zeros(max_epoch, dtype=int)},
        "correct": {"train_loss": np.zeros(max_epoch, dtype=int), "val_loss": np.zeros(max_epoch, dtype=int)},
    }
    value_pool: Dict[str, Dict[str, list[np.ndarray]]] = {
        "buggy": {"train_loss": [], "val_loss": []},
        "correct": {"train_loss": [], "val_loss": []},
    }
    cached_logs: list[Tuple[str, np.ndarray, np.ndarray]] = []

    for r in df.itertuples(index=False):
        subset = str(getattr(r, "subset")).lower()
        if subset not in SUBSET_ORDER:
            continue
        log_path = resolve_log_path(repo_root, str(getattr(r, "csv_path")))
        if not log_path.exists():
            continue

        try:
            log_df = pd.read_csv(log_path, usecols=["train_loss", "val_loss"])
        except Exception:
            continue

        log_df = log_df.head(max_epoch)
        n = len(log_df)
        if n == 0:
            continue

        train = pd.to_numeric(log_df["train_loss"], errors="coerce").to_numpy(dtype=float, copy=False)
        val = pd.to_numeric(log_df["val_loss"], errors="coerce").to_numpy(dtype=float, copy=False)
        cached_logs.append((subset, train.copy(), val.copy()))

        for metric, values in (("train_loss", train), ("val_loss", val)):
            finite = values[np.isfinite(values)]
            if finite.size:
                value_pool[subset][metric].append(finite)

    if not cached_logs:
        return

    clip_upper: Dict[str, Dict[str, float]] = {
        "buggy": {"train_loss": np.inf, "val_loss": np.inf},
        "correct": {"train_loss": np.inf, "val_loss": np.inf},
    }
    for subset in SUBSET_ORDER:
        for metric in ("train_loss", "val_loss"):
            vals = value_pool[subset][metric]
            if not vals:
                continue
            all_vals = np.concatenate(vals)
            q = float(np.quantile(all_vals, loss_upper_quantile))
            clip_upper[subset][metric] = q if np.isfinite(q) else float("inf")

    for subset, train, val in cached_logs:
        for metric, values in (("train_loss", train), ("val_loss", val)):
            cap = clip_upper[subset][metric]
            clean = values.copy()
            finite_mask = np.isfinite(clean)
            if not finite_mask.any():
                continue
            clean[finite_mask] = np.minimum(clean[finite_mask], cap)

            idx = np.arange(clean.shape[0], dtype=int)
            i = idx[finite_mask]
            v = clean[finite_mask]
            sum_loss[subset][metric][i] += v
            sumsq_loss[subset][metric][i] += np.square(v)
            cnt_loss[subset][metric][i] += 1

    def mean_and_ci95(sum_arr: np.ndarray, sumsq_arr: np.ndarray, cnt_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.full(sum_arr.shape, np.nan, dtype=float)
        ci95 = np.full(sum_arr.shape, np.nan, dtype=float)

        valid_mean = cnt_arr > 0
        mean[valid_mean] = sum_arr[valid_mean] / cnt_arr[valid_mean]

        valid_var = cnt_arr > 1
        var = np.full(sum_arr.shape, np.nan, dtype=float)
        var[valid_var] = (
            sumsq_arr[valid_var] - (np.square(sum_arr[valid_var]) / cnt_arr[valid_var])
        ) / (cnt_arr[valid_var] - 1)
        var[valid_var] = np.maximum(var[valid_var], 0.0)

        sem = np.full(sum_arr.shape, np.nan, dtype=float)
        sem[valid_var] = np.sqrt(var[valid_var] / cnt_arr[valid_var])
        ci95[valid_var] = 1.96 * sem[valid_var]
        return mean, ci95

    x = np.arange(max_epoch, dtype=int)
    fig, ax = plt.subplots(figsize=(7.1, 4.0))

    plotted = False
    for subset in SUBSET_ORDER:
        for metric, linestyle, metric_name in (
            ("train_loss", "-", "Train"),
            ("val_loss", "--", "Validation"),
        ):
            mean, ci95 = mean_and_ci95(
                sum_loss[subset][metric],
                sumsq_loss[subset][metric],
                cnt_loss[subset][metric],
            )
            if np.isnan(mean).all():
                continue
            label = f"{subset.capitalize()} {metric_name}"
            ax.plot(x, mean, linestyle=linestyle, linewidth=1.8, color=SUBSET_COLORS[subset], label=label)

            mask = ~np.isnan(mean) & ~np.isnan(ci95)
            if mask.any():
                ax.fill_between(
                    x[mask],
                    (mean - ci95)[mask],
                    (mean + ci95)[mask],
                    color=SUBSET_COLORS[subset],
                    alpha=0.14,
                    linewidth=0,
                )
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Epoch (0-indexed)")
    ax.set_ylabel("Mean loss across logs (winsorized)")
    ax.set_xlim(0, max_epoch - 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right", ncol=2)

    finalize_figure(fig, out_path)


def main() -> None:
    args = parse_args()
    manifest_path, seed_meta_path = infer_paths(args.data_root, args.manifest, args.seed_metadata)
    configure_plot_style()

    out_dir: Path = args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = read_manifest(manifest_path)
    repo_root = infer_repo_root(manifest_path, df)
    seed_meta_df = read_seed_metadata(seed_meta_path)

    fig_category_counts_buggy(df, fig_dir / "category_counts_buggy.pdf")
    fig_epochs_hist(df, fig_dir / "epochs_hist_buggy_vs_correct.pdf")
    fig_arch_log_counts(df, seed_meta_df, fig_dir / "arch_log_counts_stacked.pdf")
    fig_cat_arch_heatmap_buggy(df, seed_meta_df, fig_dir / "cat_arch_heatmap_buggy.pdf")
    fig_mean_loss_curves(
        df,
        repo_root,
        fig_dir / "mean_loss_curves_buggy_vs_correct.pdf",
        max_epoch=args.max_epoch_plot,
        loss_upper_quantile=args.loss_upper_quantile,
    )

    print(f"Wrote figures under {fig_dir}")


if __name__ == "__main__":
    main()
