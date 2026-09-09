"""Render tracked, audited evidence as tables, figures and a full experiment report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.build_readme_figures import (
    COLORS, MODEL_LABELS, SIZES, QUANTIZATIONS, configure_style, add_header, save_plot,
    plot_training_loss, plot_training_cost, plot_throughput_scaling, plot_parallelism_summary,
    plot_quality_throughput,
)
from analysis.results import parse_variant_name, load_performance_comparison, join_quality_throughput


def table(frame):
    def cell(value):
        return str(value).replace("|", "\\|")
    return "\n".join([
        "| " + " | ".join(map(cell, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def display_table(frame, columns, percent=(), decimals=2):
    view = frame[list(columns)].copy()
    for column in view:
        if column in percent:
            view[column] = view[column].map(lambda v: f"{100 * v:.{decimals}f}%")
        elif pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda v: f"{v:,.{decimals}f}")
    return table(view.rename(columns=columns))


def finish(fig, assets, name, top=0.89):
    fig.tight_layout(rect=(0.03, 0.03, 0.99, top), h_pad=2.8, w_pad=2)
    save_plot(fig, assets / name)
    plt.close(fig)


def quality_plots(q, diagnostics, complexity, assets):
    metrics = [("macro_precision", "Precision"), ("macro_recall", "Recall"),
               ("macro_f1", "Move-set F1"), ("set_exact_rate", "Exact-position accuracy")]
    labels = [f"{rank.upper()} / {fmt.upper()}" for rank in ("base", "r16", "r32") for fmt in QUANTIZATIONS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    add_header(fig, "All 27 deployment variants", "9,872 held-out positions; macro move metrics and whole-set exact accuracy (%).")
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = np.array([[q[(q['size'] == size) & (q.variant == rank) & (q.quantization == fmt)][metric].iloc[0] * 100
                            for size in SIZES] for rank in ("base", "r16", "r32") for fmt in QUANTIZATIONS])
        ax.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value > 65 else "#203040")
        ax.set_xticks(range(3), [MODEL_LABELS[s] for s in SIZES], fontsize=9)
        ax.set_yticks(range(9), labels, fontsize=9)
        ax.set_title(title)
        ax.grid(False)
    finish(fig, assets, "quality_matrix.png", 0.92)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    add_header(fig, "Quantization cost after BF16 LoRA training", "Change relative to the matching BF16 model; negative values are percentage-point losses.")
    names = [f"{s}-r{r}" for s in SIZES for r in (16, 32)]
    for ax, metric, title in zip(axes, ("macro_f1", "set_exact_rate"), ("Move-set F1", "Exact-position accuracy")):
        for shift, fmt in ((-0.18, "fp8"), (0.18, "nvfp4")):
            vals = [100 * (q.set_index('model').loc[f'{n}-{fmt}', metric] - q.set_index('model').loc[f'{n}-bf16', metric]) for n in names]
            bars = ax.bar(np.arange(6) + shift, vals, 0.34, label=fmt.upper(), color=COLORS[fmt])
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xticks(range(6), names, rotation=25, ha='right')
        ax.set_title(title)
        ax.set_ylabel("Change (percentage points)")
        ax.axhline(0, color="#34495E")
        ax.legend()
        ax.margins(y=0.22)
    finish(fig, assets, "quantization_quality_delta.png")

    frame = q[(q.quantization == 'bf16') & (q.variant != 'base')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    add_header(fig, "King-safety constraints remain difficult", "BF16 fine-tuned variants; 451 in-check positions and 9,421 non-check positions.")
    for ax, check, noncheck, title in zip(axes, ('check_f1', 'check_exact'), ('non_check_f1', 'non_check_exact'),
                                        ('Move-set F1', 'Exact-position accuracy')):
        x = np.arange(len(frame))
        for shift, field, color, label in ((-.18, noncheck, '#159D91', 'Not in check'), (.18, check, '#E76F51', 'In check')):
            bars = ax.bar(x + shift, frame[field] * 100, .34, color=color, label=label)
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
        ax.set_xticks(x, frame.model.str.replace('-bf16', ''), rotation=25, ha='right')
        ax.set_title(title)
        ax.set_ylim(0, 110)
        ax.set_ylabel('Percent')
    axes[1].legend(loc='upper right')
    finish(fig, assets, "check_status.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    add_header(fig, "Generation length limits", "Fraction ending at the 512-token cap; every capped response remains in the reported quality metrics.")
    merged = q.merge(diagnostics, on='model', validate='one_to_one')
    for ax, size in zip(axes, SIZES):
        for offset, fmt in zip((-.24, 0, .24), QUANTIZATIONS):
            vals = [merged[(merged['size'] == size) & (merged.variant == rank) & (merged.quantization == fmt)].length_rate.iloc[0] * 100
                    for rank in ('base', 'r16', 'r32')]
            bars = ax.bar(np.arange(3) + offset, vals, .23, color=COLORS[fmt], label=fmt.upper())
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
        ax.set_xticks(range(3), ['Base', 'Rank 16', 'Rank 32'])
        ax.set_title(MODEL_LABELS[size])
        ax.set_ylabel('Length-limited outputs (%)')
        ax.set_ylim(0, 112)
        ax.legend(fontsize=8)
    finish(fig, assets, "generation_limits.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    add_header(fig, "Accuracy by legal-set size", "Rank-32 BF16; descriptive bins have different board distributions, not controlled difficulty levels.")
    for ax, metric, title in zip(axes, ('f1', 'exact'), ('Move-set F1', 'Exact-position accuracy')):
        for size in SIZES:
            f = complexity[complexity.model == f'{size}-r32-bf16']
            ax.plot(f.legal_moves_bucket, 100 * f[metric], marker='o', label=MODEL_LABELS[size])
        ax.set_title(title)
        ax.set_ylabel('Percent')
        ax.set_xlabel('Number of legal moves')
        ax.set_ylim(0, 103)
        ax.legend()
    finish(fig, assets, "legal_set_size.png")


def serving_plots(p, q, evidence, assets):
    for mode, label in (('natural_fen', 'Natural FEN'), ('fixed_tokens', 'Fixed 128/128')):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        add_header(fig, f"{label}: client-observed latency", "Median across three runs of each per-run percentile; solid = p50, dashed = p95. These are not pooled percentiles.")
        for col, size in enumerate(SIZES):
            for row, fields, title in ((0, ('p50_ttft_ms', 'p95_ttft_ms'), 'Time to first token'),
                                       (1, ('p50_e2e_ms', 'p95_e2e_ms'), 'End-to-end latency')):
                ax = axes[row, col]
                for fmt in QUANTIZATIONS:
                    f = p[(p['size'] == size) & (p['mode'] == mode) & (p.quantization == fmt)]
                    for key, style in zip(fields, ('-', '--')):
                        ax.plot(f.concurrency, f[key], style, color=COLORS[fmt], marker='o',
                                label=fmt.upper() if style == '-' else None)
                ax.set_xscale('log', base=2)
                ax.set_yscale('log')
                ax.set_xticks([1, 8, 32, 128, 256], ['1', '8', '32', '128', '256'])
                ax.set_title(f'{MODEL_LABELS[size]} | {title}', fontsize=10)
                ax.set_xlabel('Concurrent requests')
                ax.set_ylabel('Milliseconds (log scale)')
                ax.legend(fontsize=8)
        finish(fig, assets, f"latency_{mode}.png", .91)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    add_header(fig, "Completed request throughput", "Median across three repetitions; natural responses have variable output lengths.")
    for row, mode in enumerate(('natural_fen', 'fixed_tokens')):
        for col, size in enumerate(SIZES):
            ax = axes[row, col]
            for fmt in QUANTIZATIONS:
                f = p[(p['size'] == size) & (p['mode'] == mode) & (p.quantization == fmt)]
                ax.plot(f.concurrency, f.request_rps, marker='o', color=COLORS[fmt], label=fmt.upper())
            ax.set_xscale('log', base=2)
            ax.set_xticks([1, 8, 32, 128, 256], ['1', '8', '32', '128', '256'])
            ax.set_title(f'{MODEL_LABELS[size]} | {mode}', fontsize=10)
            ax.set_xlabel('Concurrent requests')
            ax.set_ylabel('Requests/s')
            ax.legend(fontsize=8)
    finish(fig, assets, 'request_throughput.png', .91)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    add_header(fig, "Deployment footprint and startup", "Full checkpoint-directory GiB, not active model VRAM; one observed server startup per variant.")
    f = q[q.variant == 'r32']
    for offset, fmt in zip((-.24, 0, .24), QUANTIZATIONS):
        vals = [f[(f['size'] == size) & (f.quantization == fmt)].artifact_gib.iloc[0] for size in SIZES]
        bars = axes[0].bar(np.arange(3) + offset, vals, .23, color=COLORS[fmt], label=fmt.upper())
        axes[0].bar_label(bars, fmt='%.2f', padding=2, fontsize=8)
        vals = [next(r['server_startup_seconds'] for r in evidence['speed'] if r['name'] == f'{size}-r32-{fmt}') for size in SIZES]
        bars = axes[1].bar(np.arange(3) + offset, vals, .23, color=COLORS[fmt], label=fmt.upper())
        axes[1].bar_label(bars, fmt='%.0f', padding=2, fontsize=8)
    for ax in axes:
        ax.set_xticks(range(3), [MODEL_LABELS[s] for s in SIZES])
        ax.legend()
        ax.margins(y=.2)
    axes[0].set_ylabel('Checkpoint directory size (GiB)')
    axes[1].set_ylabel('Server startup seconds (one observation)')
    finish(fig, assets, 'deployment_footprint.png')


def make_report(root, evidence, q, p, training, diagnostics, assets):
    best = q.set_index('model').loc['e4b-r32-bf16']
    qtable = display_table(q, {'model': 'Variant', 'macro_precision': 'Precision', 'macro_recall': 'Recall',
                               'macro_f1': 'F1', 'set_exact_rate': 'Exact', 'illegal_move_rate': 'Illegal (micro)'},
                          ('macro_precision', 'macro_recall', 'macro_f1', 'set_exact_rate', 'illegal_move_rate'))
    bf16 = q[(q.quantization == 'bf16') & (q.variant != 'base')]
    subset_table = display_table(bf16, {'model': 'Variant', 'non_check_f1': 'Non-check F1', 'check_f1': 'In-check F1',
                                       'check_exact': 'In-check exact', 'check_illegal_share': 'Share of illegal extras from check'},
                                ('non_check_f1', 'check_f1', 'check_exact', 'check_illegal_share'))
    d = diagnostics.copy()
    diag_table = display_table(d, {'model': 'Variant', 'length_count': 'At cap', 'length_rate': 'At cap %',
                                   'mean_output_tokens': 'Mean tokens', 'p95_output_tokens': 'p95 tokens'}, ('length_rate',))
    perf = p[p.concurrency == 256].copy()
    perf['speedup'] = [row.output_tps / perf[(perf['size'] == row['size']) & (perf['mode'] == row['mode']) &
                                           (perf.quantization == 'bf16')].output_tps.iloc[0]
                       for _, row in perf.iterrows()]
    perf_table = display_table(perf, {'model': 'Variant', 'mode': 'Workload', 'request_rps': 'Requests/s',
                                     'output_tps': 'Output tok/s', 'speedup': 'vs BF16',
                                     'p95_ttft_ms': 'p95 TTFT ms', 'p95_e2e_ms': 'p95 E2E ms'})
    costs = training.copy()
    costs['minutes'] = costs.training_seconds / 60
    costs['input_m'] = costs.processed_input_tokens / 1e6
    cost_table = display_table(costs, {'model': 'Model', 'lora_rank': 'Rank', 'train_loss': 'Mean train loss',
                                      'minutes': 'Training minutes', 'input_m': 'Input M tokens',
                                      'input_tokens_per_second': 'Input tok/s'}, decimals=4)
    selection = pd.DataFrame(evidence['checkpoint_selection'])
    selection['old_checkpoint'] = selection.old_checkpoint.map(lambda p: p.rsplit('/', 1)[-1])
    selection['selected_checkpoint'] = selection.selected_checkpoint.map(lambda p: p.rsplit('/', 1)[-1])
    select_table = display_table(selection, {'run': 'Run', 'old_checkpoint': 'Old selection',
        'selected_checkpoint': 'Corrected selection', 'old_selection_f1': 'Old-contract F1',
        'corrected_selection_f1': 'Corrected-contract F1'}, ('old_selection_f1', 'corrected_selection_f1'))
    systems = []
    for t in evidence['training']:
        s = t['summary']
        systems.append({'run': t['config']['run']['project_name'], 'compile': s['torch_compile_strategy_resolved'],
                        'trainable_M': s['model_parameters_trainable'] / 1e6,
                        'trainable_pct': s['model_parameters_trainable_pct'],
                        'allocated_GiB': s['training_gpu_peak_allocated_bytes'] / 2**30,
                        'reserved_GiB': s['training_gpu_peak_reserved_bytes'] / 2**30,
                        'supervised_M': s['fen_sft_processed_supervised_tokens'] / 1e6})
    systems_table = display_table(pd.DataFrame(systems), {'run': 'Run', 'compile': 'Compile strategy',
        'trainable_M': 'Trainable M parameters', 'trainable_pct': 'Trainable %', 'supervised_M': 'Supervised M tokens',
        'allocated_GiB': 'Peak allocated GiB', 'reserved_GiB': 'Peak reserved GiB'})
    samples = evidence['audit']['measured_requests']
    source_table = table(pd.DataFrame(evidence['sources'])[['filename', 'bytes', 'sha256']])
    pngs = sorted(assets.glob('*.png'))
    gallery = '\n\n'.join(
        f'### {p.stem.replace("_", " ").capitalize()}\n\n![{p.stem}](<{Path(os.path.relpath(p, root)).as_posix()}>)'
        for p in pngs
    )
    text = f'''# Corrected experiment report (v2)

This report describes six BF16 LoRA SFT runs, 27 deployment-quality variants and nine rank-32 serving variants. Numerical tables are generated from the audited artifacts listed below. The quality experiment ran on September 8, 2026 and the serving benchmark on September 9, 2026.

## Research question and scope

Can small instruction-tuned language models enumerate every legal UCI move from a FEN, and how do model size, LoRA rank and post-training quantization change the accuracy/serving tradeoff? The model receives no rules engine during generation. `python-chess` supplies supervision and the evaluation oracle. This is a constrained sequence-generation study, not a measurement of chess-playing strength or search quality.

E4B rank-32 BF16 achieved **{best.macro_f1 * 100:.2f}% macro F1** and **{best.set_exact_rate * 100:.2f}% exact-position accuracy**. These are different outcomes: high move-level overlap can coexist with a wrong complete set. In these runs rank 32 exceeded rank 16 in BF16 for all three families. One seed per configuration does not establish a universal rank effect.

## Data and evaluation contract

Training used 60,000 positions for five epochs. The selection set is the first 128 positions of a 10,000-position validation file. Final scoring uses the remaining 9,872 positions, including 451 in check and 9,421 not in check. The saved split audit reports zero canonical-FEN and source-game collisions against training and the selection set. This report verifies the saved audit and held-out hash; it does not independently reconstruct the full original training split from these ZIPs.

The builder applies the first puzzle-line move to obtain the tactical position, samples rating/theme strata, and uses sorted complete legal sets. The all-moves converter rejects positions over the configured move-count cap instead of clipping labels. Lichess puzzle sampling is not a uniform sample of reachable chess states.

The exact prompt is generated by `make_all_legal_moves_instruction` and wrapped with the shared Gemma turn template. The prompt audit confirms evaluation matches the training instruction. Legacy validation-file instructions are ignored in favor of FEN-derived prompts. This is not an intentionally different prompt-robustness experiment.

Generation uses temperature 0, top-p 1, top-k -1, min-p 0, repetition penalty 1, seed 20260820 and a **512-output-token budget**. EOT uses string stopping plus verified token IDs where available; the parser independently discards post-EOT text. Gemma 4 tokenizes the marker differently, so an unknown-token ID must not become the stop ID. The saved per-model stop configurations are retained in `evidence.json`.

Precision, recall and F1 are computed per position, then averaged (macro). F1 is not obtained by taking the harmonic mean of the two dataset-average precision/recall values. Exact accuracy requires equality of the complete move sets. Valid UCI tokens are lowercased and deduplicated; malformed tokens are excluded from set scoring and tracked in the strict-format metric. Illegal rate is micro-aggregated illegal extras divided by valid unique predicted moves. Base scores therefore describe this specific prompt/parser/token budget and cannot measure all latent chess knowledge.

## Training techniques and what was measured

All six measured runs trained LoRA against BF16 base weights. Ranks 16/32 used alpha 32/64, dropout 0.05, attention and MLP projection targets, fused AdamW, learning rate 2e-4, weight decay 0.01 and 3% warmup (282 steps). Microbatch 8 with accumulation 4 gives effective batch 32; 9,375 optimizer steps cover five epochs. Loss is computed on completion tokens with prompt/padding labels masked. The low-rank update is scaled as `(alpha / rank) * B @ A`; the base is frozen while the adapters learn.

Training enabled TorchInductor compilation, dynamic shapes, TF32 for eligible FP32 operations and pinned/prefetched data loading. Gemma 3 resolved to full-model compilation; Gemma 4 resolved to regional decoder compilation. Wrapper resolution makes eligible inner linear layers accessible to PEFT. No compile-off control run was measured, so the report attributes no numerical speedup to compilation. Gradient checkpointing was disabled in these runs; this avoids recomputation at the cost of memory.

Resumable Trainer checkpoints preserve adapter/optimizer/scheduler/RNG state. Deferred generation-based checkpoint selection separates selection cost from the training loop. The loop timing can still include periodic checkpoint saving and compilation inside training; it is not isolated GPU kernel time.

{cost_table}

`Mean train loss` is the Trainer's whole-run average, not the final logged loss. Raw logged loss curves are shown in the gallery. No held-out cross-entropy loss series is available in these generation-based evaluation artifacts. Different tokenizers and model architectures prevent interpreting cross-family loss differences as direct measures of chess quality. Exact processed input tokens include prompt plus completion, exclude padding, and count repeated epochs. Supervised tokens refer only to labels contributing to loss. These are not unique corpus token counts.

{systems_table}

The supplied telemetry reports zero truncated training examples. Training hardware was RTX PRO 6000 Blackwell; exact package versions and configurations for each run are retained in `evidence.json`. Time measurements are individual runs, not statistical estimates of training-speed differences between ranks.

## Corrected checkpoint selection

{select_table}

Old and corrected selection scores use different evaluation contracts and sometimes different selected checkpoints. Their difference is **not** a measured improvement caused by additional training. The executed selection notebook is the source of this historical table; standalone full corrected selection JSON files were not supplied in the reporting archives. The comparison reevaluated 33 checkpoints per run according to that notebook. All final deployment claims use v2 evidence.

The changed selections are 270M-r32 (8700), 270M-r16 (9000) and E2B-r32 (8400). Those adapters were merged and quantized again. The other three selected adapters stayed the same. This report supersedes earlier v1 final-quality and serving summaries. Revisiting a held-out set while repairing evaluation can influence later decisions; this is an iterative benchmark, not a claim of a never-inspected blind test.

## Deployment precision

Every reported low-precision model starts from a BF16 LoRA merge, then undergoes post-training quantization. These results are not evidence of QLoRA, FP8 training or NVFP4 QAT performance. Optional training modes in the code are outside this experiment matrix.

FP8 uses the `FP8_DYNAMIC` compressed-tensors export, with stored quantized linear weights and dynamic activation scaling. NVFP4 uses NVIDIA ModelOpt W4A4: E2M1 values, groups of 16 with FP8 local scales and a higher-level scale. Calibration uses 512 training examples, maximum sequence length 512 and batch size 8. The actual quantizer metadata, excluded modules and tensor validation summaries are retained per model. Embeddings, heads and excluded components may remain higher precision; the whole artifact is not uniformly four or eight bits. See the [NVIDIA recipe reference](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/ptq.md).

The serving configuration requests FlashInfer CUTLASS for NVFP4 dense linears. A B12x MoE backend option in the command does not by itself demonstrate that MoE expert kernels were used. Kernel settings are recorded; no Nsight profiling or kernel-level FLOP measurements were performed.

## All quality results

{qtable}

FP8 and NVFP4 effects must be assessed against the same size/rank BF16 checkpoint. The quantization delta figure shows percentage-point differences. Quantization sometimes increases output-token throughput while decreasing exact accuracy substantially; neither bit width nor throughput alone selects a suitable deployment.

## Check versus non-check behavior

{subset_table}

For E4B-r32 BF16, check positions account for **{best.check_illegal_share * 100:.2f}%** of illegal extras while comprising only 4.57% of positions. The large check/non-check gap is consistent with difficulty enforcing king safety. It does not directly reveal an internal algorithm or prove that every extra move was caused by the same rule. Legal-set-size bins mix different board conditions, so their curves are descriptive rather than a causal difficulty ablation.

## Length limits and output diagnostics

{diag_table}

All length-limited generations remain in the denominators. Results measure quality under a fixed 512-token budget, not unconstrained quality. Base models often produce prose/repetition or incomplete lists under the strict output contract; this is a limitation of the baseline protocol as well as an observed task-compliance gap. The diagnostic CSV also includes post-EOT text counts, missed moves, and raw-prediction hashes. Post-EOT marker observation can be normalized by the worker and should not be interpreted as an independent stopping-rate measurement; use recorded finish reasons for the cap diagnostic.

## Serving experiment

Nine rank-32 exports were measured: three sizes times three precisions. Each ran natural FEN and fixed 128-input/128-output workloads at concurrency 1, 8, 32, 128 and 256, three repetitions each: 270 scenarios and **{samples:,} measured requests**, with zero failed requests. Each scenario also has 16 warmup requests, excluded from its measured request count. Request counts per repetition are 128, 128, 256, 512 and 1,024 respectively. Requests are not independent training seeds.

`vllm bench serve` measures a local client calling the local server over HTTP with streaming. Maximum concurrency caps in-flight requests; it is not a fixed training microbatch. vLLM schedules continuous batches. The benchmark uses infinite offered request rate, no prefix caching, chunked prefill, `max_num_seqs=256`, `max_num_batched_tokens=16384`, maximum context 1,024 and GPU memory utilization 0.90. See the [vLLM benchmark guide](https://docs.vllm.ai/en/stable/benchmarking/cli/).

Natural prompts use the held-out FEN prompt format and EOT stopping with a 512-token cap. Each concurrency level uses a subset of the same 9,872-position file, not the entire file; larger concurrency also changes sample count. Fixed-token scenarios ignore EOS and force 128 generated tokens, and all input/output lengths were verified in raw results. Fixed prompts are synthetic and their outputs have no chess-accuracy interpretation. Pairing full-test quality with fixed-token throughput compares two separate measurements of the same artifact.

Throughput is the median of three repetitions; bands are observed min/max, not confidence intervals. Latency cells are medians of three per-repetition p50/p95 values, not percentiles of pooled requests. Short fixed workloads have startup/steady-state effects and do not establish sustained production capacity. There is no WAN traffic, real-user arrival distribution or multi-GPU scaling experiment.

### Concurrency 256

{perf_table}

Full repetition data and medians at every concurrency are provided as CSV. No repetition is silently removed as an outlier. Natural output-token throughput depends on how many tokens the model produces, so requests/s and latency should be read alongside it. Small differences between three repetitions should not be presented as statistically established gains.

### Environment and memory

Serving used {evidence['speed_environment']['gpu']}. Package versions: `{json.dumps(evidence['speed_environment']['packages'], sort_keys=True)}`. This is the measured stack, not a promise that future unpinned dependencies behave identically.

The footprint chart reports full exported checkpoint-directory bytes (including tokenizer and potentially unused multimodal components), not active text-model weights or GPU memory. Quality reports include NVML peaks, which include allocator state, activations and KV cache. The 0.90 vLLM memory reservation must not be described as model-weight memory. Startup times are single observations and excluded from warm serving-throughput tables. Configured concurrency 256 is the maximum tested, not a proved capacity limit.

## Evidence validation and limits

The report builder independently rescored all {27 * 9872:,} saved completions against engine-generated legal moves, checked all aggregate metrics, test ordering and split hashes, checked all 270 raw scenario results for request completion and fixed lengths, and matched all nine serving checkpoint hashes to quality evidence. It checked the saved overlap audit; the original training dataset is a separate artifact. Unit tests cover invalid evidence rejection and parser contracts. CPU CI cannot establish GPU kernel correctness.

No new training, generation or performance measurements were run during report production. The report does not establish causality for model size/rank, prove general chess reasoning, demonstrate production SLAs, or benchmark optional QAT/RAG/RL modules. More training seeds, a new blind challenge set, format-flexible base prompting, longer generation budgets, and profiling are future work.

### Release notebook behavior

The release retains the established Colab-tested notebook execution behavior and v2 result paths. Reporting-time additions to subprocess streaming, nonempty-export rejection and quality-report preflight checks were deferred. The training, quantization, generation and scoring implementations were not changed for this report. CPU syntax/evidence tests support packaging checks, not a claim of a new GPU integration run. Publishing the completed v2 results does not require another Colab experiment.

Known operational limitations are documented rather than silently changed: existing export folders can be reused when overwrite is disabled, the quality subprocess can be quiet, and users should verify the selected quality comparison is complete before starting speed tests. For a future checkpoint change, back up old exports and use fresh names or explicit overwrite; an existing manifest alone is not proof of a new successful export. These cautions do not alter the supplied v2 measurements, whose quality/speed checkpoint hashes were audited.

## Sources and reproduction

{source_table}

The raw ZIPs remain external artifacts until uploaded to a release. This repository contains compact evidence and derived tables, not weights or the full raw prediction corpus. With all listed input files in one folder, run:

```bash
python -m pip install -e ".[analysis]"
python -m analysis.build_v2_report --input-dir /path/to/downloads
```

To redraw figures and this report from the already audited tracked evidence:

```bash
python -m analysis.build_v2_report
```

The first command validates raw evidence; the second does not claim to repeat that validation. Selected failure examples are stored in `evidence.json`; they are chosen as the lowest-F1 failures in each check subset, not representative random samples.

## Figure gallery

PNG files are suitable for GitHub; matching SVG files support scalable export.

{gallery}
'''
    (root / 'REPORT.md').write_text(text, encoding='utf-8')


def render(root, assets):
    evidence = json.loads((root / 'evidence.json').read_text(encoding='utf-8'))
    rows = []
    for r in evidence['quality']:
        size, variant, fmt = parse_variant_name(r['name'])
        rows.append({'model': r['name'], 'size': size, 'model_label': MODEL_LABELS[size],
                     'variant': variant, 'quantization': fmt, **r['quality']['overall'],
                     'check_f1': r['quality']['in_check']['macro_f1'],
                     'non_check_f1': r['quality']['not_in_check']['macro_f1'],
                     'check_exact': r['quality']['in_check']['set_exact_rate'],
                     'non_check_exact': r['quality']['not_in_check']['set_exact_rate'],
                     'check_illegal_share': r['quality']['illegal_moves_from_check_rate'],
                     'artifact_gib': r['checkpoint_directory_size_bytes'] / 2**30,
                     'checkpoint_content_sha256': r['checkpoint']['checkpoint_content_sha256'],
                     'evaluation_contract_sha256': r['evaluation_contract_sha256']})
    q = pd.DataFrame(rows)
    q.to_csv(root / 'quality_results.csv', index=False)
    raw, p = load_performance_comparison(root / 'performance_repetitions.csv')
    p.to_csv(root / 'performance_medians.csv', index=False)
    training = pd.read_csv(root / 'training_runs.csv')
    history = pd.read_csv(root / 'training_history.csv')
    diagnostics = pd.read_csv(root / 'generation_diagnostics.csv')
    complexity = pd.read_csv(root / 'position_complexity.csv')
    joined = join_quality_throughput(q, p)
    joined.to_csv(root / 'quality_throughput_joined.csv', index=False)
    assets.mkdir(parents=True, exist_ok=True)
    configure_style()
    plt.rcParams['svg.hashsalt'] = 'llm-chess-v2'
    plot_training_loss(history, assets)
    plot_training_cost(training, assets)
    plot_throughput_scaling(p, assets)
    plot_parallelism_summary(p, assets)
    for mode in ('natural_fen', 'fixed_tokens'):
        plot_quality_throughput(joined, mode, assets)
    quality_plots(q, diagnostics, complexity, assets)
    serving_plots(p, q, evidence, assets)
    make_report(root, evidence, q, p, training, diagnostics, assets)
    print(f'Report: {root / "REPORT.md"}; 15 figures saved as PNG and SVG', flush=True)
