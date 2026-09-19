from fen_move_colab.run_parallelism_benchmark import (
    _request_count,
    _scalar_result,
    _scenario_plan,
    _scenario_contract_payload,
    _signature_config,
    _validate_quality_provenance,
    _validate_benchmark_result,
)


def test_scenario_plan_covers_every_mode_concurrency_and_repetition():
    config = {
        "modes": ["natural_fen", "fixed_tokens"],
        "concurrencies": [1, 8],
        "repetitions": 3,
        "requests_per_concurrency": {"1": 16, "8": 32},
    }
    plan = _scenario_plan(config)
    assert len(plan) == 12
    assert plan[0] == {
        "mode": "natural_fen",
        "concurrency": 1,
        "repetition": 1,
        "requests": 16,
    }
    assert plan[-1] == {
        "mode": "fixed_tokens",
        "concurrency": 8,
        "repetition": 3,
        "requests": 32,
    }


def test_request_count_rejects_less_than_one_wave():
    config = {"requests_per_concurrency": {"8": 4}}
    try:
        _request_count(config, 8)
    except ValueError as exc:
        assert "at least concurrency" in str(exc)
    else:
        raise AssertionError("Expected request count validation to fail")


def test_scalar_result_keeps_summary_and_drops_detailed_arrays():
    payload = {
        "request_throughput": 12.5,
        "completed": 128,
        "ttfts": [0.1, 0.2],
        "metadata": {"model": "example"},
    }
    assert _scalar_result(payload) == {
        "request_throughput": 12.5,
        "completed": 128,
    }


def test_result_validation_rejects_partial_success():
    _validate_benchmark_result({"completed": 128, "failed": 0}, 128)
    try:
        _validate_benchmark_result({"completed": 127, "failed": 1}, 128)
    except ValueError as exc:
        assert "completed 127 of 128" in str(exc)
    else:
        raise AssertionError("Expected partial benchmark result to fail validation")


def test_parallelism_signature_ignores_transient_port():
    first = {"server": {"port": 12345, "host": "127.0.0.1"}, "run": {"resume": True}}
    second = {"server": {"port": 54321, "host": "127.0.0.1"}, "run": {"resume": True}}
    assert _signature_config(first) == _signature_config(second)


def test_scenario_contract_ignores_port_but_preserves_actual_command():
    first = {
        "server": {"port": 12345, "host": "127.0.0.1"},
        "command": ["vllm", "bench", "serve", "--port", "12345", "--seed", "7"],
    }
    second = {
        "server": {"port": 54321, "host": "127.0.0.1"},
        "command": ["vllm", "bench", "serve", "--port", "54321", "--seed", "7"],
    }

    assert _scenario_contract_payload(first) == _scenario_contract_payload(second)
    assert first["command"][4] == "12345"
    assert second["command"][4] == "54321"


def test_quality_provenance_requires_exact_checkpoint_hash():
    evidence = {"quality_status": "completed", "checkpoint_content_sha256": "abc"}
    manifest = {"checkpoint_content_sha256": "abc"}
    result = _validate_quality_provenance("demo", evidence, manifest)
    assert result["checkpoint_hash_match"] is True

    try:
        _validate_quality_provenance(
            "demo", evidence, {"checkpoint_content_sha256": "different"}
        )
    except ValueError as exc:
        assert "Checkpoint hash mismatch" in str(exc)
    else:
        raise AssertionError("Expected mismatched checkpoint hashes to fail")
