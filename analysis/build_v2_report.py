"""Audit archived experiments and regenerate the versioned publication evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.results import load_training_metrics
from fen_move_colab.publication_eval import aggregate_scores, score_completion, legal_moves_for_fen

QUALITY_ARCHIVE = "publication_evaluation_v2-20260909T121805Z-1-001.zip"
SPEED_ARCHIVE = "parallelism_benchmarks_v2-20260909T121800Z-1-001.zip"
TRAINING_ARCHIVES = [f"metrics-20260821T{time}Z-1-001.zip" for time in
                     ("082535", "082542", "082546", "082550", "082557", "082604")]
SELECTION_NOTEBOOK = "Checkpoint_Selection_EOT_Comparison_Colab (1).ipynb"
SIZES = ("270m", "e2b", "e4b")
FORMATS = ("bf16", "fp8", "nvfp4")
VARIANTS = ("base", "r16", "r32")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_member(archive, suffix):
    names = [n for n in archive.namelist() if n.endswith(suffix)]
    require(len(names) == 1, f"Expected one {suffix}, found {names}")
    return archive.read(names[0])


def read_json(archive, suffix):
    return json.loads(read_member(archive, suffix))


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_scenarios(scenarios, names):
    expected = set(itertools.product(names, ("natural_fen", "fixed_tokens"), (1, 8, 32, 128, 256), (1, 2, 3)))
    keys = [(s["model"], s["mode"], int(s["concurrency"]), int(s["repetition"])) for s in scenarios]
    require(len(keys) == len(set(keys)), "Duplicate benchmark scenarios")
    require(set(keys) == expected, "Missing or unexpected benchmark scenarios")
    for s in scenarios:
        require(s["completed"] == s["requested_prompts"] and s["failed"] == 0,
                f"Incomplete requests: {s['model']} {s['mode']} {s['concurrency']}")
        for key in ("duration", "output_throughput", "request_throughput", "p50_ttft_ms",
                    "p95_ttft_ms", "p50_e2el_ms", "p95_e2el_ms"):
            require(math.isfinite(s[key]) and s[key] > 0, f"Invalid {key}: {s}")


def selection_from_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "\n".join("".join(o.get("text", [])) for c in notebook["cells"] for o in c.get("outputs", []))
    pattern = (r"(gemma\S+): changed=(True|False) old_f1=([\d.]+) new_f1=([\d.]+)\s+"
               r"old: (\S+)\s+new: (\S+)")
    rows = [dict(run=m[0], changed=m[1] == "True", old_selection_f1=float(m[2]),
                 corrected_selection_f1=float(m[3]), old_checkpoint=m[4], selected_checkpoint=m[5])
            for m in re.findall(pattern, text)]
    require(len(rows) == 6, "Could not recover six selection results from notebook")
    return rows


def audit_inputs(input_dir, output_dir):
    paths = [input_dir / n for n in [QUALITY_ARCHIVE, SPEED_ARCHIVE, *TRAINING_ARCHIVES, SELECTION_NOTEBOOK]]
    sources = [{"filename": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in paths]
    selection = selection_from_notebook(paths[-1])
    diagnostic_rows, buckets, examples = [], [], []
    with zipfile.ZipFile(paths[0]) as archive:
        quality = read_json(archive, "/comparison.json")
        config = read_json(archive, "/evaluation_config.json")
        split = read_json(archive, "/testset/split_manifest.json")
        prompt = read_json(archive, "/prompt_audit.json")
        expected_names = {"-".join(v) for v in itertools.product(SIZES, VARIANTS, FORMATS)}
        require(len(quality["results"]) == 27 and {r["name"] for r in quality["results"]} == expected_names,
                "Quality matrix is incomplete")
        heldout_bytes = read_member(archive, "/" + Path(split["final_test_path"]).name)
        require(hashlib.sha256(heldout_bytes).hexdigest() == split["final_test_file_sha256"], "Test file hash mismatch")
        heldout = [json.loads(line) for line in heldout_bytes.splitlines() if line.strip()]
        require(len(heldout) == 9872, "Unexpected held-out size")
        require(split["checkpoint_selection_excluded_from_test"], "Selection set not excluded")
        for key in ("training_fen_collisions", "training_game_collisions",
                    "training_selection_fen_collisions", "training_selection_game_collisions"):
            require(split[key] == 0, f"Split audit failure: {key}")
        require(prompt["evaluation_prompt_matches_training"], "Prompt mismatch")
        oracle = {}
        for row in heldout:
            board, moves = legal_moves_for_fen(row["fen"])
            require(moves == sorted(row["legal_moves"]), "Incorrect ground truth")
            oracle[row["fen"]] = (row, board, moves)
        require(len(oracle) == len(heldout), "Duplicate test FENs")
        for result in quality["results"]:
            name = result["name"]
            print(f"Auditing predictions: {name}", flush=True)
            require(result["status"] == "completed", f"Failed model: {name}")
            raw = read_member(archive, f"/{name}/predictions.jsonl")
            predictions = [json.loads(line) for line in raw.splitlines() if line.strip()]
            require(len(predictions) == 9872, f"Prediction count mismatch: {name}")
            require([p["fen"] for p in predictions] == [r["fen"] for r in heldout], f"Test ordering mismatch: {name}")
            # Recompute from completion text and the engine, not from stored scores.
            rescored = [score_completion(oracle[p["fen"]][0], p["completion"]) for p in predictions]
            aggregate = aggregate_scores(rescored)
            for group in ("overall", "in_check", "not_in_check"):
                for metric, value in aggregate[group].items():
                    require(math.isclose(value, result["quality"][group][metric], abs_tol=1e-10),
                            f"Metric mismatch: {name} {group} {metric}")
            for saved, fresh in zip(predictions, rescored):
                require(saved["predicted_moves"] == fresh["predicted_moves"], f"Parser mismatch: {name}")
            finish = pd.Series([p.get("finish_reason", "unknown") for p in predictions]).value_counts().to_dict()
            diagnostic_rows.append({"model": name, "length_count": finish.get("length", 0),
                "length_rate": finish.get("length", 0) / len(predictions), "stop_count": finish.get("stop", 0),
                "mean_output_tokens": np.mean([p["completion_tokens"] for p in predictions]),
                "p95_output_tokens": np.percentile([p["completion_tokens"] for p in predictions], 95),
                "post_eot_text_count": sum(bool(p["text_after_end_of_turn"].strip()) for p in rescored),
                "missed_moves": sum(len(p["missed_moves"]) for p in rescored),
                "prediction_file_sha256": hashlib.sha256(raw).hexdigest()})
            for lo, hi, label in ((0, 10, "1-10"), (11, 20, "11-20"), (21, 30, "21-30"),
                                  (31, 40, "31-40"), (41, 1000, "41+")):
                subset = [p for p in rescored if lo <= p["expected_move_count"] <= hi]
                if subset:
                    buckets.append({"model": name, "legal_moves_bucket": label, "samples": len(subset),
                                    "f1": np.mean([p["f1"] for p in subset]),
                                    "exact": np.mean([p["set_exact"] for p in subset])})
            if name == "e4b-r32-bf16":
                for in_check in (True, False):
                    subset = sorted([p for p in rescored if p["in_check"] == in_check and not p["set_exact"]],
                                    key=lambda p: (p["f1"], p["fen"]))
                    for p in subset[:2]:
                        examples.append({k: p[k] for k in ("fen", "puzzle_id", "in_check", "f1",
                                                           "expected_moves", "missed_moves", "illegal_moves")})
        # Keep compact per-model evidence, including environments and content hashes.
        qresults = [{k: v for k, v in r.items() if k not in ("performance", "checkpoint", "source_checkpoint")}
                    | {"checkpoint": {k: v for k, v in r["checkpoint"].items() if k != "files"}}
                    for r in quality["results"]]
    qmap = {r["name"]: r for r in qresults}
    with zipfile.ZipFile(paths[1]) as archive:
        speed = read_json(archive, "/comparison.json")
        speed_config = read_json(archive, "/benchmark_config.json")
        require(speed["dataset"]["source_sha256"] == split["final_test_file_sha256"], "Benchmark split mismatch")
        scenarios = [s for r in speed["results"] for s in r["scenarios"]]
        names = {f"{s}-r32-{q}" for s, q in itertools.product(SIZES, FORMATS)}
        require(len(speed["results"]) == 9 and {r["name"] for r in speed["results"]} == names, "Speed matrix mismatch")
        validate_scenarios(scenarios, names)
        for result in speed["results"]:
            require(result["status"] == "completed", f"Failed benchmark: {result['name']}")
            require(result["serving_checkpoint"]["checkpoint_content_sha256"] ==
                    qmap[result["name"]]["checkpoint"]["checkpoint_content_sha256"], "Quality/speed checkpoint mismatch")
            for s in result["scenarios"]:
                scenario_path = f"/{s['model']}/scenarios/{s['mode']}_c{s['concurrency']}_r{s['repetition']}/result.json"
                raw = read_json(archive, scenario_path)
                require(raw["completed"] == s["requested_prompts"] and raw["failed"] == 0, "Raw request failure")
                for field in ("output_throughput", "request_throughput", "p50_ttft_ms", "p95_ttft_ms",
                              "p50_e2el_ms", "p95_e2el_ms", "total_input_tokens", "total_output_tokens"):
                    require(raw[field] == s[field], f"Raw scenario mismatch: {scenario_path} {field}")
                require(len(raw["output_lens"]) == raw["completed"], "Missing detailed request data")
                if s["mode"] == "fixed_tokens":
                    require(set(raw["input_lens"]) == {128} and set(raw["output_lens"]) == {128}, "Fixed token mismatch")
        csv = pd.read_csv(io.BytesIO(read_member(archive, "/performance_comparison.csv")))
        validate_scenarios(csv.to_dict("records"), names)
        speed_results = [{k: v for k, v in r.items() if k not in ("scenarios", "serving_checkpoint")}
                         for r in speed["results"]]
    training, history = load_training_metrics(paths[2:-1])
    require(len(training) == 6 and not training.duplicated(["size", "lora_rank"]).any(), "Training matrix mismatch")
    train_evidence = []
    for path in paths[2:-1]:
        with zipfile.ZipFile(path) as archive:
            train_evidence.append({"config": read_json(archive, "/step_60000_config.json"),
                                   "summary": read_json(archive, "/step_60000.json")})
    # The old training archive's selection scores use the superseded parser.
    training = training.drop(columns=["best_checkpoint", "selection_f1"])
    training = training.merge(pd.DataFrame(selection)[["run", "selected_checkpoint", "corrected_selection_f1"]],
                              on="run", validate="one_to_one")
    evidence = {"schema": "chess_report_v2", "sources": sources, "quality_config": config,
                "split": split, "prompt_audit": prompt, "quality": qresults,
                "speed_config": speed_config, "speed_environment": speed["environment"],
                "speed": speed_results, "training": train_evidence, "checkpoint_selection": selection,
                "audit": {"quality_models": 27, "rescored_predictions": 27 * 9872, "benchmark_models": 9,
                          "scenarios": len(scenarios), "measured_requests": sum(s["completed"] for s in scenarios),
                          "failed_requests": 0, "checkpoint_hash_matches": 9,
                          "fixed_input_output_lengths_verified": True,
                          "training_overlap": "saved split audit verified; original training data not reaudited"},
                "error_examples": examples}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evidence.json", evidence)
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "generation_diagnostics.csv", index=False)
    pd.DataFrame(buckets).to_csv(output_dir / "position_complexity.csv", index=False)
    pd.DataFrame(selection).to_csv(output_dir / "checkpoint_selection.csv", index=False)
    training.to_csv(output_dir / "training_runs.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    csv.to_csv(output_dir / "performance_repetitions.csv", index=False)
    print(json.dumps(evidence["audit"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Folder with the two v2 ZIPs, six metric ZIPs and selection notebook")
    parser.add_argument("--results-dir", type=Path, default=Path("results/v2"))
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    args = parser.parse_args()
    if args.input_dir:
        audit_inputs(args.input_dir, args.results_dir)
    from analysis.render_v2_report import render
    render(args.results_dir, args.assets_dir)


if __name__ == "__main__":
    main()
