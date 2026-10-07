from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.results import (
    CONCURRENCIES,
    MODEL_LABELS,
    QUANTIZATIONS,
    join_quality_throughput,
    load_performance_comparison,
    load_quality_comparison,
    load_training_metrics,
    write_result_tables,
)


SIZES = ("270m", "e2b", "e4b")
COLORS = {"bf16": "#34495E", "fp8": "#E76F51", "nvfp4": "#159D91"}
QUANT_LABELS = {"bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4"}
SIZE_COLORS = {"270m": "#277DA1", "e2b": "#F8961E", "e4b": "#43AA8B"}
MARKERS = {"bf16": "o", "fp8": "s", "nvfp4": "^"}
NATURAL_FEN_TIMING_WARNING = "Archived/provisional: post-EOT generation may affect timing."


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.facecolor": "#FCFAF5",
            "figure.facecolor": "#F7F4ED",
            "axes.edgecolor": "#B8B1A4",
            "axes.grid": True,
            "grid.color": "#DED8CC",
            "grid.alpha": 0.72,
            "legend.frameon": False,
        }
    )


def add_header(figure: plt.Figure, title: str, subtitle: str) -> None:
    figure.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=19, fontweight="bold", color="#203040")
    subtitle_y = 0.985 - 30 / (72 * figure.get_figheight())
    figure.text(0.055, subtitle_y, subtitle, ha="left", va="top", fontsize=10.5, color="#59636B")


def save_plot(figure: plt.Figure, path: Path) -> None:
    for extension in ("png", "svg"):
        metadata = {"Date": None} if extension == "svg" else None
        figure.savefig(path.with_suffix(f".{extension}"), dpi=220, bbox_inches="tight", metadata=metadata)


def add_natural_fen_timing_warning(figure: plt.Figure) -> None:
    # Keep the caveat inside the exported image, even when shared without its caption.
    figure.text(
        0.055, -0.005, f"Natural-FEN timings | {NATURAL_FEN_TIMING_WARNING}", ha="left", va="top",
        fontsize=11, fontweight="bold", color="#9A441F",
        bbox={"facecolor": "#FFF0D8", "edgecolor": "#D4A369", "pad": 6},
    )


def plot_training_loss(history: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    add_header(figure, "SFT training loss across five epochs", "Teacher-forced loss for six BF16 LoRA runs.")
    for axis, size in zip(axes, SIZES):
        subset = history[history["size"] == size]
        for rank, line_style in ((16, "--"), (32, "-")):
            values = subset[subset["lora_rank"] == rank]
            axis.plot(values["epoch"], values["loss"], label=f"LoRA r={rank}", linestyle=line_style, linewidth=2)
        axis.set_title(MODEL_LABELS[size])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Training loss")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend()
    figure.tight_layout(rect=(0.04, 0.04, 0.99, 0.88), w_pad=2)
    save_plot(figure, output_dir / "training_loss_curves.png")
    plt.close(figure)


def plot_training_cost(training: pd.DataFrame, output_dir: Path) -> None:
    order = {(size, rank): index for index, (size, rank) in enumerate((s, r) for s in SIZES for r in (16, 32))}
    frame = training.copy()
    frame["order"] = [order[(size, rank)] for size, rank in zip(frame["size"], frame["lora_rank"])]
    frame = frame.sort_values("order")
    labels = [f"{model}\nr={rank}" for model, rank in zip(frame["model"], frame["lora_rank"])]

    figure, axis = plt.subplots(figsize=(12, 5.8))
    add_header(
        figure,
        "SFT compute and exact token accounting",
        "Training-loop hours; labels show processed input tokens across all epochs.",
    )
    bars = axis.bar(range(len(frame)), frame["training_seconds"] / 3600, color="#456990")
    for bar, tokens in zip(bars, frame["processed_input_tokens"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{tokens / 1e6:.1f}M tok",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(range(len(frame)), labels)
    axis.set_ylabel("Training-loop hours")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0.04, 0.05, 0.99, 0.86))
    save_plot(figure, output_dir / "training_cost.png")
    plt.close(figure)


def plot_throughput_scaling(performance: pd.DataFrame, output_dir: Path) -> None:
    mode_titles = {"natural_fen": "Real FEN prompts", "fixed_tokens": "Fixed 128 input / 128 output"}
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharex=True)
    add_header(
        figure,
        "vLLM concurrency scaling on Blackwell",
        "Output-token throughput; median of three repetitions, shaded band is min-max.",
    )
    for row, mode in enumerate(("natural_fen", "fixed_tokens")):
        for column, size in enumerate(SIZES):
            axis = axes[row, column]
            for quantization in QUANTIZATIONS:
                frame = performance[
                    (performance["mode"] == mode)
                    & (performance["size"] == size)
                    & (performance["quantization"] == quantization)
                ].sort_values("concurrency")
                axis.plot(
                    frame["concurrency"],
                    frame["output_tps"] / 1000,
                    color=COLORS[quantization],
                    marker="o",
                    linewidth=2.25,
                    markersize=5,
                    label=QUANT_LABELS[quantization],
                )
                axis.fill_between(
                    frame["concurrency"],
                    frame["output_tps_min"] / 1000,
                    frame["output_tps_max"] / 1000,
                    color=COLORS[quantization],
                    alpha=0.13,
                    linewidth=0,
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(CONCURRENCIES, [str(value) for value in CONCURRENCIES])
            axis.set_title(f"{MODEL_LABELS[size]} | {mode_titles[mode]}")
            axis.set_xlabel("Concurrent requests")
            if column == 0:
                axis.set_ylabel("Output throughput (k tokens/s)")
            axis.set_ylim(bottom=0)
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0 and column == 2:
                axis.legend(loc="upper left")
    figure.tight_layout(rect=(0.04, 0.04, 0.99, 0.91), h_pad=2, w_pad=1.5)
    add_natural_fen_timing_warning(figure)
    save_plot(figure, output_dir / "throughput_scaling.png")
    plt.close(figure)


def plot_parallelism_summary(performance: pd.DataFrame, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10))
    add_header(
        figure,
        "Parallel inference: throughput and latency",
        "Concurrency 256 throughput and tail latency plus NVFP4 request scaling.",
    )
    x = np.arange(len(SIZES))
    width = 0.24
    for axis, mode, title in (
        (axes[0, 0], "natural_fen", "Archived natural-FEN workload at concurrency 256"),
        (axes[0, 1], "fixed_tokens", "Controlled 128/128 workload at concurrency 256"),
    ):
        frame = performance[(performance["mode"] == mode) & (performance["concurrency"] == 256)]
        for offset, quantization in zip((-width, 0, width), QUANTIZATIONS):
            values = [
                frame[(frame["size"] == size) & (frame["quantization"] == quantization)]["output_tps"].iloc[0]
                / 1000
                for size in SIZES
            ]
            bars = axis.bar(x + offset, values, width, color=COLORS[quantization], label=QUANT_LABELS[quantization])
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        axis.set_xticks(x, [MODEL_LABELS[size] for size in SIZES])
        axis.set_ylabel("Output throughput (k tokens/s)")
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 1].legend(loc="upper right")

    natural = performance[performance["mode"] == "natural_fen"]
    for size in SIZES:
        frame = natural[(natural["size"] == size) & (natural["quantization"] == "nvfp4")]
        axes[1, 0].plot(
            frame["concurrency"], frame["request_rps"], marker="o", linewidth=2.3, label=MODEL_LABELS[size]
        )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(CONCURRENCIES, [str(value) for value in CONCURRENCIES])
    axes[1, 0].set_xlabel("Concurrent requests")
    axes[1, 0].set_ylabel("Completed requests/s")
    axes[1, 0].set_title("Natural-prompt request throughput with NVFP4")
    axes[1, 0].legend()
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    frame = natural[natural["concurrency"] == 256].copy()
    frame["label"] = frame["size"].map(MODEL_LABELS) + " " + frame["quantization"].map(QUANT_LABELS)
    frame = frame.sort_values("p95_e2e_ms")
    bars = axes[1, 1].barh(frame["label"], frame["p95_e2e_ms"] / 1000, color=frame["quantization"].map(COLORS))
    axes[1, 1].bar_label(bars, fmt="%.2fs", padding=3, fontsize=8)
    axes[1, 1].set_xlabel("p95 end-to-end latency (seconds)")
    axes[1, 1].set_title("Tail latency at concurrency 256")
    axes[1, 1].spines[["top", "right"]].set_visible(False)
    figure.tight_layout(rect=(0.045, 0.04, 0.99, 0.91), h_pad=2.8, w_pad=2.5)
    add_natural_fen_timing_warning(figure)
    save_plot(figure, output_dir / "parallelism_portfolio_summary.png")
    plt.close(figure)


def pareto_frontier(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    candidates = []
    for _, row in frame.iterrows():
        dominated = (
            (frame["output_tps"] >= row["output_tps"])
            & (frame[metric] >= row[metric])
            & ((frame["output_tps"] > row["output_tps"]) | (frame[metric] > row[metric]))
        ).any()
        if not dominated:
            candidates.append(row)
    return pd.DataFrame(candidates).sort_values("output_tps")


def plot_quality_throughput(joined: pd.DataFrame, mode: str, output_dir: Path) -> None:
    titles = {
        "natural_fen": ("Quality versus archived throughput: real FEN workload", "Requested EOT stopping, maximum 512 tokens"),
        "fixed_tokens": ("Quality-throughput frontier: controlled workload", "Exactly 128 input and output tokens"),
    }
    metrics = (
        ("macro_f1", "Move-set F1"),
        ("set_exact_rate", "Exact-position accuracy"),
        ("macro_recall", "Move-set recall"),
        ("macro_precision", "Move-set precision"),
    )
    title, workload = titles[mode]
    frame = joined[joined["mode"] == mode]
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10))
    add_header(
        figure,
        title,
        f"{workload}; throughput at concurrency 256 is median of three repetitions; quality uses 9,872 FENs.",
    )
    size_handles = [
        mlines.Line2D([], [], color=SIZE_COLORS[size], marker="o", linestyle="None", label=MODEL_LABELS[size])
        for size in SIZES
    ]
    quant_handles = [
        mlines.Line2D(
            [], [], color="#37474F", marker=MARKERS[quantization], markerfacecolor="white", linestyle="None",
            label=QUANT_LABELS[quantization]
        )
        for quantization in QUANTIZATIONS
    ]
    figure.legend(handles=size_handles + quant_handles, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=6)
    for axis, (metric, label) in zip(axes.flat, metrics):
        frontier = pareto_frontier(frame, metric)
        axis.plot(frontier["output_tps"] / 1000, frontier[metric] * 100, "--", color="#8A8175", linewidth=1.4)
        for _, row in frame.iterrows():
            x = row["output_tps"] / 1000
            error = np.array([[x - row["output_tps_min"] / 1000], [row["output_tps_max"] / 1000 - x]])
            axis.errorbar(
                x,
                row[metric] * 100,
                xerr=error,
                marker=MARKERS[row["quantization"]],
                markersize=9,
                markerfacecolor=SIZE_COLORS[row["size"]],
                markeredgecolor="white",
                color=SIZE_COLORS[row["size"]],
                capsize=3,
                linestyle="None",
            )
        axis.set_title(label)
        axis.set_xlabel("Output throughput (k tokens/s)")
        axis.set_ylabel(f"{label} (%)")
        axis.set_xlim(left=0)
        axis.spines[["top", "right"]].set_visible(False)
    filename = (
        "quality_vs_throughput_natural_fen.png"
        if mode == "natural_fen"
        else "quality_vs_throughput_fixed_128x128.png"
    )
    figure.tight_layout(rect=(0.04, 0.055, 0.99, 0.855), h_pad=2.1, w_pad=2)
    if mode == "natural_fen":
        add_natural_fen_timing_warning(figure)
    save_plot(figure, output_dir / filename)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all six README figures from raw publication outputs.")
    parser.add_argument("--quality-comparison", required=True, type=Path)
    parser.add_argument("--performance-csv", required=True, type=Path)
    parser.add_argument("--training-metrics", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument("--tables-dir", type=Path, default=Path("analysis/generated/tables"))
    args = parser.parse_args()

    quality = load_quality_comparison(args.quality_comparison)
    performance_raw, performance = load_performance_comparison(args.performance_csv)
    training, history = load_training_metrics(args.training_metrics)
    joined = join_quality_throughput(quality, performance)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_training_loss(history, args.output_dir)
    plot_training_cost(training, args.output_dir)
    plot_throughput_scaling(performance, args.output_dir)
    plot_parallelism_summary(performance, args.output_dir)
    plot_quality_throughput(joined, "natural_fen", args.output_dir)
    plot_quality_throughput(joined, "fixed_tokens", args.output_dir)
    write_result_tables(args.tables_dir, quality, performance_raw, performance, training, history)
    print(f"Regenerated six README figures in {args.output_dir.resolve()}")
    print(f"Wrote normalized tables to {args.tables_dir.resolve()}")


if __name__ == "__main__":
    main()
