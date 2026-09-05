from typing import Dict, Optional, Tuple

from .common import load_json, save_json


def load_stage_state(path: str) -> Dict:
    state = load_json(path, default={})
    state.setdefault("next_sample_index", 0)
    state.setdefault("total_samples_trained", 0)
    state.setdefault("latest_checkpoint", None)
    state.setdefault("history", [])
    return state


def resolve_chunk(
    state_path: str,
    chunk_samples: int,
    start_sample: Optional[int] = None,
    previous_total: Optional[int] = None,
) -> Tuple[Dict, int, int, int]:
    state = load_stage_state(state_path)
    start = int(state["next_sample_index"] if start_sample is None else start_sample)
    previous = int(state["total_samples_trained"] if previous_total is None else previous_total)
    count = int(chunk_samples)
    if count <= 0:
        raise ValueError("chunk_samples must be positive.")
    new_total = previous + count
    return state, start, count, new_total


def update_stage_state(
    state_path: str,
    state: Dict,
    start: int,
    count: int,
    new_total: int,
    checkpoint: str,
    metrics: Dict,
) -> None:
    state["next_sample_index"] = start + count
    state["total_samples_trained"] = new_total
    state["latest_checkpoint"] = checkpoint
    state.setdefault("history", []).append(
        {
            "start_sample_index": start,
            "sample_count": count,
            "saved_as_step": new_total,
            "checkpoint": checkpoint,
            "metrics": metrics,
        }
    )
    save_json(state_path, state)

