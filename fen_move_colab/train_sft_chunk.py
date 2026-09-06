import argparse
import inspect
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List

import chess
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .build_all_legal_moves_dataset import make_all_legal_moves_instruction
from .chunk_state import resolve_chunk, update_stage_state
from .common import ensure_dir, load_config, resolve_drive_path, save_json, set_seed
from .fp8_lora import (
    get_model_load_kwargs,
    low_precision_alignment_enabled,
    maybe_prepare_fp8_lora_model,
    precision_mode,
    precision_runtime_metrics,
    resolve_adapter_source,
)
from .jsonl_data import load_jsonl_slice
from .publication_eval import parse_uci_moves, transformers_stop_kwargs
from .performance import (
    configure_torch_runtime,
    gradient_checkpointing_enabled,
    performance_metrics,
    prepare_model_for_compilation,
    training_argument_performance_kwargs,
)
from .run_telemetry import (
    TrainingTelemetryCallback,
    cuda_memory_metrics,
    directory_size_bytes,
    model_parameter_metrics,
    prefixed_trainer_metrics,
    process_peak_rss_bytes,
    reset_cuda_peak_memory,
    runtime_environment,
    utc_now_iso,
)


UCI_RE = re.compile(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", re.IGNORECASE)


def _resolve_warmup_steps(sft_cfg: Dict, dataset_size: int):
    per_device_batch = max(1, int(sft_cfg["per_device_train_batch_size"]))
    gradient_accumulation = max(1, int(sft_cfg["gradient_accumulation_steps"]))
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    epochs = float(sft_cfg.get("num_train_epochs", 1.0))
    updates_per_epoch = max(
        1,
        math.ceil(dataset_size / (per_device_batch * gradient_accumulation * world_size)),
    )
    planned_optimizer_steps = max(1, math.ceil(updates_per_epoch * epochs))

    explicit_steps = sft_cfg.get("warmup_steps")
    if explicit_steps is not None:
        warmup_steps = int(explicit_steps)
        if warmup_steps < 0:
            raise ValueError("sft.warmup_steps must be non-negative.")
        return warmup_steps, planned_optimizer_steps

    warmup_ratio = float(sft_cfg.get("warmup_ratio", 0.0))
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("sft.warmup_ratio must be between 0 and 1.")
    return math.ceil(planned_optimizer_steps * warmup_ratio), planned_optimizer_steps


def _parse_uci_move(text: str):
    match = UCI_RE.search((text or "").lower())
    if not match:
        return None
    try:
        return chess.Move.from_uci(match.group(0))
    except ValueError:
        return None


class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, response_template_ids, pad_to_multiple_of=None):
        super().__init__(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=pad_to_multiple_of)
        self.response_template_ids = response_template_ids

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        labels = batch["labels"].clone()
        tmpl = self.response_template_ids
        for i in range(len(labels)):
            seq = labels[i].tolist()
            found_idx = -1
            for j in range(max(0, len(seq) - len(tmpl) + 1)):
                if seq[j:j + len(tmpl)] == tmpl:
                    found_idx = j + len(tmpl)
                    break
            labels[i, :found_idx if found_idx >= 0 else len(labels[i])] = -100
        batch["labels"] = labels
        return batch


class SequentialTrainer(Trainer):
    """Sequential Trainer with low-overhead, runtime-observed token counts."""

    _fen_token_count_names = (
        "input_tokens",
        "supervised_tokens",
        "tensor_tokens",
        "examples",
        "microbatches",
    )

    def _observe_training_tokens(self, inputs) -> None:
        input_ids = inputs.get("input_ids")
        if input_ids is None or not torch.is_tensor(input_ids):
            return

        counts = torch.empty(len(self._fen_token_count_names), dtype=torch.int64, device=input_ids.device)
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        counts[0] = (
            attention_mask.detach().sum(dtype=torch.int64)
            if torch.is_tensor(attention_mask)
            else input_ids.numel()
        )
        counts[1] = (
            labels.detach().ne(-100).sum(dtype=torch.int64)
            if torch.is_tensor(labels)
            else 0
        )
        counts[2] = input_ids.numel()
        counts[3] = input_ids.shape[0]
        counts[4] = 1
        if getattr(self, "_fen_runtime_token_counts", None) is None:
            self._fen_runtime_token_counts = counts
        else:
            self._fen_runtime_token_counts.add_(counts)

    def compute_loss(self, model, inputs, *args, **kwargs):
        self._observe_training_tokens(inputs)
        return super().compute_loss(model, inputs, *args, **kwargs)

    def runtime_token_metrics(self, elapsed_seconds: float = 0.0) -> Dict[str, float]:
        counts = getattr(self, "_fen_runtime_token_counts", None)
        if counts is None:
            values = {name: 0 for name in self._fen_token_count_names}
        else:
            counts = counts.detach().clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            values = {
                name: int(counts[index].item())
                for index, name in enumerate(self._fen_token_count_names)
            }

        input_tokens = values["input_tokens"]
        supervised_tokens = values["supervised_tokens"]
        tensor_tokens = values["tensor_tokens"]
        metrics = {
            "fen_sft_runtime_invocation_input_tokens": input_tokens,
            "fen_sft_runtime_invocation_supervised_tokens": supervised_tokens,
            "fen_sft_runtime_invocation_tensor_tokens": tensor_tokens,
            "fen_sft_runtime_invocation_padding_tokens": max(0, tensor_tokens - input_tokens),
            "fen_sft_runtime_invocation_examples": values["examples"],
            "fen_sft_runtime_invocation_microbatches": values["microbatches"],
        }
        if elapsed_seconds > 0:
            metrics.update(
                {
                    "fen_sft_runtime_invocation_input_tokens_per_second": input_tokens / elapsed_seconds,
                    "fen_sft_runtime_invocation_supervised_tokens_per_second": supervised_tokens / elapsed_seconds,
                    "fen_sft_runtime_invocation_tensor_tokens_per_second": tensor_tokens / elapsed_seconds,
                }
            )
        return metrics

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        dataloader_kwargs = {
            "dataset": self.train_dataset,
            "batch_size": self._train_batch_size,
            "sampler": SequentialSampler(self.train_dataset),
            "collate_fn": self.data_collator,
            "drop_last": self.args.dataloader_drop_last,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = bool(
                getattr(self.args, "dataloader_persistent_workers", False)
            )
            dataloader_kwargs["prefetch_factor"] = int(
                getattr(self.args, "dataloader_prefetch_factor", 2) or 2
            )
        return DataLoader(
            **dataloader_kwargs,
        )


class LegalMoveValidationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        examples,
        eval_steps: int,
        max_new_tokens: int,
        batch_size: int,
        telemetry_callback=None,
    ):
        self.tokenizer = tokenizer
        self.examples = list(examples)
        self.eval_steps = max(1, int(eval_steps))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.batch_size = max(1, int(batch_size))
        self.telemetry_callback = telemetry_callback
        self.latest_metrics: Dict[str, float] = {}

    def _prompt(self, example) -> str:
        return f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n"

    def _evaluate(self, model) -> Dict[str, float]:
        if not self.examples:
            return {}

        was_training = model.training
        model.eval()
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        total = 0
        uci_shaped = 0
        legal = 0
        solution = 0
        completion_token_sum = 0
        completion_token_max = 0

        try:
            with torch.no_grad():
                for start in range(0, len(self.examples), self.batch_size):
                    batch_examples = self.examples[start:start + self.batch_size]
                    prompts = [self._prompt(example) for example in batch_examples]
                    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    device = getattr(model, "device", None) or next(model.parameters()).device
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
                    completions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    for completion, example in zip(completions, batch_examples):
                        total += 1
                        completion_tokens = len(self.tokenizer.encode(completion or "", add_special_tokens=False))
                        completion_token_sum += completion_tokens
                        completion_token_max = max(completion_token_max, completion_tokens)

                        move = _parse_uci_move(completion)
                        if move is None:
                            continue
                        uci_shaped += 1

                        board = chess.Board(example["fen"])
                        if move not in board.legal_moves:
                            continue
                        legal += 1
                        if move.uci() == example.get("solution_uci"):
                            solution += 1
        finally:
            self.tokenizer.padding_side = old_padding_side
            if was_training:
                model.train()

        denom = max(1, total)
        return {
            "fen_sft_val_samples": float(total),
            "fen_sft_val_uci_format_rate": uci_shaped / denom,
            "fen_sft_val_legal_move_rate": legal / denom,
            "fen_sft_val_illegal_move_rate": 1.0 - (legal / denom),
            "fen_sft_val_solution_move_rate": solution / denom,
            "fen_sft_val_completion_tokens_avg": completion_token_sum / denom,
            "fen_sft_val_completion_tokens_max": float(completion_token_max),
        }

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        metrics = self._evaluate(kwargs["model"])
        if metrics:
            self.latest_metrics = metrics
            if self.telemetry_callback is not None:
                self.telemetry_callback.record("validation", metrics, int(state.global_step), state.epoch)
            kwargs["model"].train()
            trainer = kwargs.get("trainer")
            if trainer is not None:
                trainer.log(metrics)
            else:
                print(metrics)


class AllLegalMovesValidationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        examples,
        eval_steps: int,
        max_new_tokens: int,
        batch_size: int,
        telemetry_callback=None,
    ):
        self.tokenizer = tokenizer
        self.examples = list(examples)
        self.eval_steps = max(1, int(eval_steps))
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.batch_size = max(1, int(batch_size))
        self.telemetry_callback = telemetry_callback
        self.latest_metrics: Dict[str, float] = {}
        self.stop_config, self.stop_kwargs = transformers_stop_kwargs(tokenizer)

    @staticmethod
    def _prompt(example) -> str:
        instruction = make_all_legal_moves_instruction(example["fen"])
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

    def _evaluate(self, model) -> Dict[str, float]:
        if not self.examples:
            return {}

        was_training = model.training
        model.eval()
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        total = 0
        exact_set = 0
        exact_ordered = 0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        predicted_move_sum = 0
        illegal_predicted_moves = 0

        try:
            with torch.no_grad():
                for start in range(0, len(self.examples), self.batch_size):
                    batch_examples = self.examples[start:start + self.batch_size]
                    prompts = [self._prompt(example) for example in batch_examples]
                    inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    device = getattr(model, "device", None) or next(model.parameters()).device
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    generation_kwargs = {
                        **self.stop_kwargs,
                        "max_new_tokens": self.max_new_tokens,
                        "do_sample": False,
                        "pad_token_id": pad_token_id,
                    }
                    try:
                        outputs = model.generate(**inputs, **generation_kwargs)
                    except TypeError:
                        # Older Transformers builds may not expose string
                        # stopping; parsing below remains the correctness
                        # backstop, and the stop contract is still recorded.
                        generation_kwargs.pop("stop_strings", None)
                        generation_kwargs.pop("tokenizer", None)
                        outputs = model.generate(**inputs, **generation_kwargs)
                    generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
                    completions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

                    for completion, example in zip(completions, batch_examples):
                        expected = sorted(set(example.get("legal_moves", [])))
                        if not expected:
                            raise ValueError("All-moves validation requires legal_moves in each validation record.")
                        predicted = parse_uci_moves(completion)["moves"]
                        predicted_set = set(predicted)
                        expected_set = set(expected)
                        correct = predicted_set & expected_set

                        total += 1
                        predicted_move_sum += len(predicted_set)
                        illegal_predicted_moves += len(predicted_set - expected_set)
                        precision = len(correct) / len(predicted_set) if predicted_set else 0.0
                        recall = len(correct) / len(expected_set)
                        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
                        precision_sum += precision
                        recall_sum += recall
                        f1_sum += f1
                        exact_set += int(predicted_set == expected_set)
                        exact_ordered += int(predicted == expected)
        finally:
            self.tokenizer.padding_side = old_padding_side
            if was_training:
                model.train()

        denom = max(1, total)
        predicted_denom = max(1, predicted_move_sum)
        return {
            "fen_sft_val_all_moves_samples": float(total),
            "fen_sft_val_all_moves_set_exact_rate": exact_set / denom,
            "fen_sft_val_all_moves_ordered_exact_rate": exact_ordered / denom,
            "fen_sft_val_all_moves_precision": precision_sum / denom,
            "fen_sft_val_all_moves_recall": recall_sum / denom,
            "fen_sft_val_all_moves_legal_moves_found_pct": 100.0 * recall_sum / denom,
            "fen_sft_val_all_moves_f1": f1_sum / denom,
            "fen_sft_val_all_moves_predicted_count_avg": predicted_move_sum / denom,
            "fen_sft_val_all_moves_illegal_move_rate": illegal_predicted_moves / predicted_denom,
        }

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        metrics = self._evaluate(kwargs["model"])
        if metrics:
            self.latest_metrics = metrics
            if self.telemetry_callback is not None:
                self.telemetry_callback.record("validation", metrics, int(state.global_step), state.epoch)
            kwargs["model"].train()
            trainer = kwargs.get("trainer")
            if trainer is not None:
                trainer.log(metrics)
            else:
                print(metrics)


def _format_example(example, tokenizer, max_seq_length: int):
    prompt = f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n"
    completion = f"{example['output']}<end_of_turn>"
    text = prompt + completion
    tokens = tokenizer(text, truncation=True, max_length=max_seq_length)
    original_length = len(tokenizer(text, truncation=False)["input_ids"])
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "_original_length": original_length,
        "_prompt_length": len(tokenizer.encode(prompt, add_special_tokens=False)),
        "_completion_length": len(tokenizer.encode(completion, add_special_tokens=False)),
    }


def _response_start(input_ids: List[int], response_template_ids: List[int]) -> int:
    for index in range(max(0, len(input_ids) - len(response_template_ids) + 1)):
        if input_ids[index:index + len(response_template_ids)] == response_template_ids:
            return index + len(response_template_ids)
    return -1


def _exact_token_metrics(
    tokenized,
    tokenizer,
    response_template_ids: List[int],
    per_device_batch_size: int,
    num_train_epochs: float,
    pad_to_multiple_of=None,
) -> Dict[str, float]:
    lengths = []
    input_tokens = 0
    supervised_tokens = 0
    prompt_sum = 0
    prompt_max = 0
    completion_sum = 0
    completion_max = 0
    original_tokens = 0
    truncated_samples = 0
    truncated_tokens = 0
    missing_response_template_samples = 0
    pad_token_id = tokenizer.pad_token_id

    for example in tokenized:
        input_ids = list(example["input_ids"])
        sequence_length = len(input_ids)
        original_length = int(example["_original_length"])
        prompt_length = int(example["_prompt_length"])
        completion_length = int(example["_completion_length"])
        response_start = _response_start(input_ids, response_template_ids)
        if response_start < 0:
            missing_response_template_samples += 1
            example_supervised_tokens = 0
        else:
            example_supervised_tokens = sum(
                1 for token_id in input_ids[response_start:]
                if pad_token_id is None or token_id != pad_token_id
            )

        lengths.append(sequence_length)
        input_tokens += sequence_length
        supervised_tokens += example_supervised_tokens
        prompt_sum += prompt_length
        prompt_max = max(prompt_max, prompt_length)
        completion_sum += completion_length
        completion_max = max(completion_max, completion_length)
        original_tokens += original_length
        if original_length > sequence_length:
            truncated_samples += 1
            truncated_tokens += original_length - sequence_length

    per_epoch_tensor_tokens = 0
    batch_size = max(1, int(per_device_batch_size))
    for start in range(0, len(lengths), batch_size):
        batch_lengths = lengths[start:start + batch_size]
        padded_length = max(batch_lengths, default=0)
        if pad_to_multiple_of and padded_length:
            padded_length = int(math.ceil(padded_length / pad_to_multiple_of) * pad_to_multiple_of)
        per_epoch_tensor_tokens += padded_length * len(batch_lengths)

    sample_count = len(lengths)
    denom = max(1, sample_count)
    masked_tokens = input_tokens - supervised_tokens
    per_epoch_padding_tokens = per_epoch_tensor_tokens - input_tokens
    rounded_epochs = round(num_train_epochs)
    integer_epoch_schedule = math.isclose(num_train_epochs, rounded_epochs, rel_tol=0.0, abs_tol=1e-9)
    processed_input_tokens = None
    processed_supervised_tokens = None
    processed_tensor_tokens = None
    processed_padding_tokens = None
    if integer_epoch_schedule:
        epoch_multiplier = int(rounded_epochs)
        processed_input_tokens = input_tokens * epoch_multiplier
        processed_supervised_tokens = supervised_tokens * epoch_multiplier
        processed_tensor_tokens = per_epoch_tensor_tokens * epoch_multiplier
        processed_padding_tokens = per_epoch_padding_tokens * epoch_multiplier

    return {
        "fen_sft_token_accounting_version": 1,
        "fen_sft_token_schedule_exact": integer_epoch_schedule,
        "fen_sft_unique_samples": sample_count,
        "fen_sft_unique_original_tokens": original_tokens,
        "fen_sft_unique_input_tokens": input_tokens,
        "fen_sft_unique_supervised_tokens": supervised_tokens,
        "fen_sft_unique_masked_tokens": masked_tokens,
        "fen_sft_input_tokens_per_epoch": input_tokens,
        "fen_sft_supervised_tokens_per_epoch": supervised_tokens,
        "fen_sft_tensor_tokens_per_epoch": per_epoch_tensor_tokens,
        "fen_sft_padding_tokens_per_epoch": per_epoch_padding_tokens,
        "fen_sft_processed_input_tokens": processed_input_tokens,
        "fen_sft_processed_supervised_tokens": processed_supervised_tokens,
        "fen_sft_processed_tensor_tokens": processed_tensor_tokens,
        "fen_sft_processed_padding_tokens": processed_padding_tokens,
        "fen_sft_processed_input_tokens_millions": (
            processed_input_tokens / 1_000_000 if processed_input_tokens is not None else None
        ),
        "fen_sft_processed_supervised_tokens_millions": (
            processed_supervised_tokens / 1_000_000 if processed_supervised_tokens is not None else None
        ),
        "fen_sft_truncated_samples": truncated_samples,
        "fen_sft_truncated_tokens": truncated_tokens,
        "fen_sft_missing_response_template_samples": missing_response_template_samples,
        "fen_sft_prompt_tokens_avg": prompt_sum / denom,
        "fen_sft_prompt_tokens_max": float(prompt_max),
        "fen_sft_completion_tokens_avg": completion_sum / denom,
        "fen_sft_completion_tokens_max": float(completion_max),
        "fen_sft_sequence_tokens_effective_avg": input_tokens / denom,
        "fen_sft_sequence_tokens_effective_max": float(max(lengths, default=0)),
    }


def _cumulative_token_metrics(state: Dict, current_metrics: Dict) -> Dict[str, float]:
    prior_input_tokens = 0
    prior_supervised_tokens = 0
    complete = True
    for item in state.get("history", []):
        metrics = item.get("metrics", {})
        input_count = metrics.get("fen_sft_processed_input_tokens")
        supervised_count = metrics.get("fen_sft_processed_supervised_tokens")
        if input_count is None or supervised_count is None:
            complete = False
            continue
        prior_input_tokens += int(input_count)
        prior_supervised_tokens += int(supervised_count)

    current_input_tokens = current_metrics.get("fen_sft_processed_input_tokens")
    current_supervised_tokens = current_metrics.get("fen_sft_processed_supervised_tokens")
    complete = complete and current_input_tokens is not None and current_supervised_tokens is not None
    cumulative_input_tokens = prior_input_tokens + int(current_input_tokens or 0) if complete else None
    cumulative_supervised_tokens = (
        prior_supervised_tokens + int(current_supervised_tokens or 0) if complete else None
    )
    return {
        "fen_sft_cumulative_token_accounting_complete": complete,
        "fen_sft_cumulative_processed_input_tokens": cumulative_input_tokens,
        "fen_sft_cumulative_processed_supervised_tokens": cumulative_supervised_tokens,
        "fen_sft_cumulative_processed_input_tokens_millions": (
            cumulative_input_tokens / 1_000_000 if cumulative_input_tokens is not None else None
        ),
        "fen_sft_cumulative_processed_supervised_tokens_millions": (
            cumulative_supervised_tokens / 1_000_000 if cumulative_supervised_tokens is not None else None
        ),
    }


def _load_sft_validation_examples(config: Dict, limit: int):
    data_cfg = config["data"]
    if limit <= 0:
        return []
    validation_path = data_cfg.get("validation_output")
    if not validation_path:
        validation_path = data_cfg.get("grpo_output") or resolve_drive_path(config, "datasets", "fen_move_grpo.jsonl")
    if not os.path.exists(validation_path):
        print(f"SFT validation skipped: {validation_path} does not exist yet.")
        return []
    return list(load_jsonl_slice(validation_path, 0, limit))


def _latest_trainer_checkpoint(save_dir: str):
    checkpoints = []
    for path in Path(save_dir).glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match and (path / "trainer_state.json").is_file():
            checkpoints.append((int(match.group(1)), str(path)))
    return max(checkpoints, default=(None, None))[1]


def run_sft_chunk(config: Dict, chunk_samples: int, start_sample=None, previous_total=None) -> str:
    stage_start_perf = time.perf_counter()
    stage_started_at_utc = utc_now_iso()
    set_seed(int(config["run"]["seed"]))
    configure_torch_runtime(config)
    sft_cfg = config["sft"]
    data_cfg = config["data"]
    if not data_cfg.get("sft_output"):
        data_cfg["sft_output"] = resolve_drive_path(config, "datasets", "fen_move_sft.jsonl")
    fp8_cfg = config.get("fp8", {})
    state_path = resolve_drive_path(config, "state", "sft_state.json")
    checkpoint_root = resolve_drive_path(config, "checkpoints", "sft")
    metrics_root = resolve_drive_path(config, "metrics", "sft")
    ensure_dir(checkpoint_root)
    ensure_dir(metrics_root)

    state, start, requested_count, requested_new_total = resolve_chunk(
        state_path, chunk_samples, start_sample, previous_total
    )
    dataset = load_jsonl_slice(data_cfg["sft_output"], start, requested_count)
    count = len(dataset)
    if count <= 0:
        raise ValueError(
            f"No SFT records are available from sample index {start}. "
            f"Requested {requested_count} records from {data_cfg['sft_output']}."
        )
    if count < requested_count:
        print(
            f"Requested {requested_count:,} SFT records but only {count:,} are available "
            f"from index {start:,}; checkpoint and state totals will use the actual count."
        )
    new_total = requested_new_total - requested_count + count
    save_dir = os.path.join(checkpoint_root, f"step_{new_total}")
    ensure_dir(save_dir)
    interrupted_checkpoint = _latest_trainer_checkpoint(save_dir)
    configured_resume = sft_cfg.get("resume_from_checkpoint")
    resume_checkpoint = (
        configured_resume
        or interrupted_checkpoint
        or state.get("latest_checkpoint")
        or sft_cfg["model_id"]
    )
    trainer_resume_checkpoint = None
    if resume_checkpoint and os.path.isdir(str(resume_checkpoint)):
        trainer_state_path = os.path.join(str(resume_checkpoint), "trainer_state.json")
        if os.path.isfile(trainer_state_path):
            trainer_resume_checkpoint = str(resume_checkpoint)
            print(f"Resuming optimizer, scheduler, and Trainer state from {trainer_resume_checkpoint}.")
    history_path = os.path.join(metrics_root, f"step_{new_total}_history.jsonl")
    config_snapshot_path = os.path.join(metrics_root, f"step_{new_total}_config.json")
    save_json(config_snapshot_path, config)
    environment_metrics = runtime_environment()
    telemetry_callback = TrainingTelemetryCallback(history_path, stage_start_perf)
    telemetry_callback.record(
        "stage_start",
        {
            "stage": "sft",
            "model_id": sft_cfg["model_id"],
            "sample_count": count,
            "environment": environment_metrics,
        },
    )

    max_seq_length = int(sft_cfg["max_seq_length"])
    if low_precision_alignment_enabled(config):
        max_seq_length = int(fp8_cfg.get("sft_max_seq_length", max_seq_length))

    model_setup_start = time.perf_counter()
    tokenizer_source, adapter_path = resolve_adapter_source(resume_checkpoint, sft_cfg["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_source, adapter_path = resolve_adapter_source(resume_checkpoint, sft_cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(model_source, **get_model_load_kwargs(config))
    model = maybe_prepare_fp8_lora_model(model, config, adapter_path=adapter_path)
    model = prepare_model_for_compilation(model, config)
    model_setup_seconds = time.perf_counter() - model_setup_start
    parameter_metrics = model_parameter_metrics(model)
    precision_metrics = precision_runtime_metrics(model, config)

    dataset_prepare_start = time.perf_counter()
    tokenized = dataset.map(
        lambda ex: _format_example(ex, tokenizer, max_seq_length),
        remove_columns=dataset.column_names,
    )

    response_template_ids = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)[1:]
    pad_to_multiple_of = 16 if low_precision_alignment_enabled(config) else None
    token_metrics = _exact_token_metrics(
        tokenized,
        tokenizer,
        response_template_ids,
        per_device_batch_size=int(sft_cfg["per_device_train_batch_size"]),
        num_train_epochs=float(sft_cfg.get("num_train_epochs", 1)),
        pad_to_multiple_of=pad_to_multiple_of,
    )
    tokenized = tokenized.remove_columns(
        ["_original_length", "_prompt_length", "_completion_length"]
    )
    collator = CompletionOnlyCollator(
        tokenizer,
        response_template_ids,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    dataset_prepare_seconds = time.perf_counter() - dataset_prepare_start

    checkpoint_cfg = sft_cfg.get("checkpointing", {})
    periodic_checkpointing = bool(checkpoint_cfg.get("enabled", True))
    save_steps = max(1, int(checkpoint_cfg.get("save_steps", 300)))
    warmup_steps, planned_optimizer_steps = _resolve_warmup_steps(sft_cfg, len(tokenized))
    if sft_cfg.get("warmup_steps") is None:
        print(
            f"Resolved warmup_ratio={float(sft_cfg.get('warmup_ratio', 0.0)):.4f} "
            f"to warmup_steps={warmup_steps} across {planned_optimizer_steps} planned optimizer steps."
        )
    else:
        print(f"Using explicit warmup_steps={warmup_steps}.")
    training_args_kwargs = {
        "output_dir": save_dir,
        "num_train_epochs": float(sft_cfg.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(sft_cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(sft_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(sft_cfg["learning_rate"]),
        "weight_decay": float(sft_cfg.get("weight_decay", 0.01)),
        "warmup_steps": warmup_steps,
        "logging_steps": int(sft_cfg.get("logging_steps", 20)),
        "save_strategy": "steps" if periodic_checkpointing else "no",
        "save_steps": save_steps,
        "report_to": [],
        "remove_unused_columns": False,
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "gradient_checkpointing": gradient_checkpointing_enabled(config),
        "seed": int(config["run"]["seed"]),
    }
    save_total_limit = checkpoint_cfg.get("save_total_limit")
    if save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = max(1, int(save_total_limit))
    training_args_kwargs.update(training_argument_performance_kwargs(config, TrainingArguments, model=model))
    args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized,
        "data_collator": collator,
        "callbacks": [telemetry_callback],
    }
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    val_cfg = sft_cfg.get("validation", {})
    validation_callback = None
    output_mode = str(sft_cfg.get("output_mode", "single_move"))
    if output_mode not in {"single_move", "all_legal_moves"}:
        raise ValueError("sft.output_mode must be 'single_move' or 'all_legal_moves'.")
    if bool(val_cfg.get("enabled", True)):
        default_samples = 64 if output_mode == "single_move" else 32
        default_max_new_tokens = 8 if output_mode == "single_move" else 384
        default_batch_size = 16 if output_mode == "single_move" else 4
        validation_examples = _load_sft_validation_examples(config, int(val_cfg.get("samples", default_samples)))
        if validation_examples:
            callback_cls = LegalMoveValidationCallback if output_mode == "single_move" else AllLegalMovesValidationCallback
            validation_callback = callback_cls(
                tokenizer=tokenizer,
                examples=validation_examples,
                eval_steps=int(val_cfg.get("steps", max(1, int(sft_cfg.get("logging_steps", 20))))),
                max_new_tokens=int(val_cfg.get("max_new_tokens", default_max_new_tokens)),
                batch_size=int(val_cfg.get("batch_size", default_batch_size)),
                telemetry_callback=telemetry_callback,
            )
            trainer_kwargs["callbacks"].append(validation_callback)

    trainer = SequentialTrainer(**trainer_kwargs)
    telemetry_callback.record(
        "training_ready",
        {
            "model_setup_seconds": model_setup_seconds,
            "dataset_prepare_seconds": dataset_prepare_seconds,
            **performance_metrics(config, model=model),
            **precision_metrics,
            **parameter_metrics,
            **token_metrics,
        },
    )
    reset_cuda_peak_memory()
    training_start = time.perf_counter()
    try:
        train_result = trainer.train(resume_from_checkpoint=trainer_resume_checkpoint)
    except Exception as exc:
        telemetry_callback.record(
            "train_error",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                **trainer.runtime_token_metrics(time.perf_counter() - training_start),
            },
            int(trainer.state.global_step),
            trainer.state.epoch,
        )
        raise
    training_loop_seconds = time.perf_counter() - training_start
    training_memory_metrics = cuda_memory_metrics("training_gpu")
    runtime_token_metrics = trainer.runtime_token_metrics(training_loop_seconds)
    runtime_token_metrics["fen_sft_runtime_invocation_scope"] = (
        "resumed_remainder" if trainer_resume_checkpoint else "full_training_schedule"
    )
    if token_metrics.get("fen_sft_processed_input_tokens") is not None:
        token_metrics["fen_sft_processed_token_count_source"] = "exact_tokenized_dataset_x_integer_epochs"
        token_metrics["fen_sft_runtime_matches_processed_schedule"] = (
            None
            if trainer_resume_checkpoint
            else runtime_token_metrics["fen_sft_runtime_invocation_input_tokens"]
            == token_metrics["fen_sft_processed_input_tokens"]
            and runtime_token_metrics["fen_sft_runtime_invocation_supervised_tokens"]
            == token_metrics["fen_sft_processed_supervised_tokens"]
        )
    else:
        token_metrics["fen_sft_processed_token_count_source"] = "runtime_observed_fractional_schedule"
        token_metrics["fen_sft_processed_input_tokens"] = runtime_token_metrics[
            "fen_sft_runtime_invocation_input_tokens"
        ]
        token_metrics["fen_sft_processed_supervised_tokens"] = runtime_token_metrics[
            "fen_sft_runtime_invocation_supervised_tokens"
        ]
        token_metrics["fen_sft_processed_tensor_tokens"] = runtime_token_metrics[
            "fen_sft_runtime_invocation_tensor_tokens"
        ]
        token_metrics["fen_sft_processed_padding_tokens"] = runtime_token_metrics[
            "fen_sft_runtime_invocation_padding_tokens"
        ]
        token_metrics["fen_sft_processed_input_tokens_millions"] = (
            token_metrics["fen_sft_processed_input_tokens"] / 1_000_000
        )
        token_metrics["fen_sft_processed_supervised_tokens_millions"] = (
            token_metrics["fen_sft_processed_supervised_tokens"] / 1_000_000
        )
    # Throughput always uses tokens observed during this invocation. Dividing
    # a full resumed schedule by only the remainder's wall time would overstate it.
    token_metrics["fen_sft_processed_input_tokens_per_second"] = runtime_token_metrics[
        "fen_sft_runtime_invocation_input_tokens_per_second"
    ]
    token_metrics["fen_sft_processed_supervised_tokens_per_second"] = runtime_token_metrics[
        "fen_sft_runtime_invocation_supervised_tokens_per_second"
    ]
    token_metrics["fen_sft_processed_tensor_tokens_per_second"] = runtime_token_metrics[
        "fen_sft_runtime_invocation_tensor_tokens_per_second"
    ]
    token_metrics.update(_cumulative_token_metrics(state, token_metrics))

    # Persist the final adapter before any optional generation work. A slow or
    # interrupted evaluation must never discard a completed training run.
    checkpoint_save_start = time.perf_counter()
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    checkpoint_save_seconds = time.perf_counter() - checkpoint_save_start
    checkpoint_bytes = directory_size_bytes(save_dir)

    final_validation_start = time.perf_counter()
    final_validation_metrics = validation_callback._evaluate(model) if validation_callback is not None else {}
    final_validation_seconds = time.perf_counter() - final_validation_start
    if final_validation_metrics:
        telemetry_callback.record(
            "final_validation",
            final_validation_metrics,
            int(trainer.state.global_step),
            trainer.state.epoch,
        )

    world_size = int(getattr(trainer.args, "world_size", 1) or 1)
    per_device_batch = int(sft_cfg["per_device_train_batch_size"])
    gradient_accumulation = int(sft_cfg["gradient_accumulation_steps"])
    lora_cfg = config.get("lora", {})

    metrics = {
        "stage": "sft",
        "status": "completed",
        "started_at_utc": stage_started_at_utc,
        "finished_at_utc": utc_now_iso(),
        "stage_wall_seconds": time.perf_counter() - stage_start_perf,
        "model_setup_seconds": model_setup_seconds,
        "dataset_prepare_seconds": dataset_prepare_seconds,
        "training_loop_seconds": training_loop_seconds,
        "final_validation_seconds": final_validation_seconds,
        "checkpoint_save_seconds": checkpoint_save_seconds,
        "start_sample_index": start,
        "sample_count": count,
        "saved_as_step": new_total,
        "model_id": sft_cfg["model_id"],
        "resume_checkpoint": resume_checkpoint,
        "checkpoint_path": save_dir,
        "checkpoint_size_bytes": checkpoint_bytes,
        "periodic_checkpointing_enabled": periodic_checkpointing,
        "periodic_checkpoint_save_steps": save_steps if periodic_checkpointing else None,
        "periodic_checkpoint_save_total_limit": save_total_limit,
        "trainer_resume_checkpoint": trainer_resume_checkpoint,
        "config_snapshot_path": config_snapshot_path,
        "history_path": history_path,
        "optimizer_global_step": float(trainer.state.global_step),
        "train_loss": float(train_result.training_loss) if train_result.training_loss is not None else None,
        "output_mode": output_mode,
        "precision_mode": precision_mode(config),
        "num_train_epochs": float(sft_cfg.get("num_train_epochs", 1)),
        "max_seq_length": max_seq_length,
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "world_size": world_size,
        "effective_train_batch_size": per_device_batch * gradient_accumulation * world_size,
        "learning_rate": float(sft_cfg["learning_rate"]),
        "warmup_ratio_requested": (
            float(sft_cfg.get("warmup_ratio", 0.0))
            if sft_cfg.get("warmup_steps") is None
            else None
        ),
        "warmup_steps": warmup_steps,
        "planned_optimizer_steps": planned_optimizer_steps,
        "lora_enabled": bool(lora_cfg.get("enabled", config.get("fp8", {}).get("use_lora", True))),
        "lora_rank": int(lora_cfg.get("r", config.get("fp8", {}).get("lora_r", 16))),
        "lora_alpha": int(lora_cfg.get("alpha", config.get("fp8", {}).get("lora_alpha", 32))),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "environment": environment_metrics,
        **performance_metrics(config, model=model),
        **parameter_metrics,
        **precision_metrics,
        **prefixed_trainer_metrics(train_result.metrics),
        **training_memory_metrics,
        **token_metrics,
        **runtime_token_metrics,
        **final_validation_metrics,
    }
    metrics_path = os.path.join(metrics_root, f"step_{new_total}.json")
    save_json(metrics_path, metrics)
    telemetry_callback.record(
        "run_summary",
        metrics,
        int(trainer.state.global_step),
        trainer.state.epoch,
    )
    update_stage_state(state_path, state, start, count, new_total, save_dir, metrics)
    print(f"Saved SFT chunk checkpoint: {save_dir}")
    print(f"Updated SFT state: {state_path}")
    print(metrics)
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Train one ordered Drive-backed SFT chunk.")
    parser.add_argument("--config", required=True, help="Path to the runtime training YAML generated by the notebook.")
    parser.add_argument("--chunk-samples", type=int, default=None)
    parser.add_argument("--start-sample", type=int, default=None)
    parser.add_argument("--previous-total", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    chunk_samples = args.chunk_samples or int(cfg["chunking"]["sft_chunk_samples"])
    run_sft_chunk(cfg, chunk_samples, args.start_sample, args.previous_total)


if __name__ == "__main__":
    main()
