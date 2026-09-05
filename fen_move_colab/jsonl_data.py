import json
from itertools import islice
from typing import Dict, List, Optional

from datasets import Dataset


def load_jsonl_slice(path: str, start: int, count: Optional[int]) -> Dataset:
    records: List[Dict] = []
    stop = None if count is None else start + count
    with open(path, "r", encoding="utf-8") as f:
        for line in islice(f, start, stop):
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records loaded from {path} with start={start}, count={count}.")
    return Dataset.from_list(records)

