from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from scripts import finquant_expert_lora_training as training


def _dataset_contract(monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict]:
    monkeypatch.setattr(training, "_source_code_version", lambda: "commit-123")
    monkeypatch.setattr(training, "_sha256_file", lambda _path: "f" * 64)
    dataset = (
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": '{"task":"trade"}'},
                    {"role": "assistant", "content": '{"result":"ok"}'},
                ],
                "preference": {
                    "contract_version": training.PREFERENCE_CONTRACT_VERSION,
                    "objective": training.RETURN_OBJECTIVE_NAME,
                    "prompt": "prompt",
                    "chosen": '{"candidate":"positive_return"}',
                    "rejected": '{"candidate":"negative_return"}',
                },
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    return dataset, training._finalize_dataset_contract(
        dataset,
        {
            "source": "test",
            "objective_name": training.RETURN_OBJECTIVE_NAME,
            "objective_version": training.RETURN_OBJECTIVE_VERSION,
            "preference_contract_version": training.PREFERENCE_CONTRACT_VERSION,
            "preference_example_count": 1,
        },
    )


def test_dataset_contract_is_content_addressed_and_tamper_evident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, manifest = _dataset_contract(monkeypatch)

    training._validate_dataset_contract(dataset, manifest)

    assert manifest["dataset_schema_version"] == "bb_finquant_expert_sft.v2"
    assert manifest["dataset_version"].startswith(
        f"bb-finquant-sft-v2-{manifest['dataset_sha256'][:12]}-"
    )
    assert manifest["dataset_version"].endswith(manifest["dataset_lineage_sha256"][:8])
    assert manifest["source_code_version"] == "commit-123"
    assert manifest["base_model_identity"]["training_repo"] == "Qwen/Qwen3-14B"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        training._validate_dataset_contract(dataset + "{}\n", manifest)


def test_dataset_bytes_round_trip_without_platform_newline_translation(tmp_path) -> None:
    dataset = '{"row":1}\n{"row":2}\n'
    path = tmp_path / "dataset.jsonl"

    path.write_bytes(dataset.encode("utf-8"))

    assert training._sha256_file(path) == training._sha256_bytes(dataset)
    assert b"\r\n" not in path.read_bytes()


def test_compact_json_never_slices_invalid_json() -> None:
    value = {
        "task": "trade",
        "payload": {
            "symbol": "BTC/USDT",
            "raw_llm_response": {"reasoning": "x" * 6000},
        },
    }

    compacted = training._json_compact(value, limit=900)

    assert len(compacted) <= 900
    parsed = json.loads(compacted)
    assert isinstance(parsed, dict)


def test_dataset_contract_rejects_invalid_json_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training, "_source_code_version", lambda: "commit-123")
    monkeypatch.setattr(training, "_sha256_file", lambda _path: "f" * 64)
    dataset = (
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": '{"truncated":'},
                    {"role": "assistant", "content": "{}"},
                ],
                "preference": {
                    "contract_version": training.PREFERENCE_CONTRACT_VERSION,
                    "objective": training.RETURN_OBJECTIVE_NAME,
                    "prompt": "prompt",
                    "chosen": "{}",
                    "rejected": "{}",
                },
            }
        )
        + "\n"
    )
    manifest = training._finalize_dataset_contract(
        dataset,
        {
            "source": "test",
            "objective_name": training.RETURN_OBJECTIVE_NAME,
            "objective_version": training.RETURN_OBJECTIVE_VERSION,
            "preference_contract_version": training.PREFERENCE_CONTRACT_VERSION,
            "preference_example_count": 1,
        },
    )

    with pytest.raises(ValueError, match="invalid JSON message content"):
        training._validate_dataset_contract(dataset, manifest)


def test_adapter_version_and_paths_never_target_legacy_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dataset, manifest = _dataset_contract(monkeypatch)

    version = training._new_adapter_version(
        manifest,
        now=datetime(2026, 7, 12, 1, 2, 3, tzinfo=UTC),
    )
    adapter_dir, specialization_manifest, train_log = training._remote_adapter_paths(version)

    assert version == f"20260712T010203Z-{manifest['dataset_sha256'][:12]}"
    assert "/versions/" in adapter_dir
    assert "BB-FinQuant-Expert-14B-v1" not in adapter_dir
    assert specialization_manifest.startswith(adapter_dir)
    assert version in train_log


def test_trade_response_uses_authoritative_net_pnl_without_double_deducting_costs() -> None:
    response = training._trade_response(
        {
            "side": "long",
            "realized_pnl": 8.5,
            "gross_pnl": 10.0,
            "entry_fee": 0.4,
            "close_fee": 0.6,
            "funding_fee": -0.5,
            "liquidation_penalty": 0.0,
        }
    )

    assert response["net_pnl_after_all_costs_usdt"] == 8.5
    assert response["gross_pnl_usdt"] == 10.0
    assert response["entry_fee_usdt"] == 0.4
    assert response["close_fee_usdt"] == 0.6
    assert response["funding_fee_usdt_signed"] == -0.5
    assert response["total_cost_drag_usdt"] == 1.5
    assert "after_fee_pnl_usdt" not in response


def test_finquant_preference_counterexample_prefers_low_win_positive_expectancy() -> None:
    row = training._return_objective_counterexample()
    preference = row["preference"]
    user_payload = json.loads(row["messages"][1]["content"])["payload"]

    assert user_payload["candidate_a"]["win_rate"] == pytest.approx(0.35)
    assert user_payload["candidate_b"]["win_rate"] == pytest.approx(0.80)
    assert preference["metrics"]["chosen_net_return_after_all_cost_pct"] > 0
    assert preference["metrics"]["rejected_net_return_after_all_cost_pct"] < 0
    assert json.loads(preference["chosen"])["best_side"] == "candidate_a"


def test_shadow_response_chooses_fee_after_side_even_when_gross_points_elsewhere() -> None:
    response = training._shadow_response(
        {
            "gross_long_return_pct": 2.0,
            "gross_short_return_pct": -2.0,
            "long_net_return_after_all_cost_pct": -0.2,
            "short_net_return_after_all_cost_pct": 0.3,
            "missed_opportunity": True,
        }
    )

    assert response["best_side"] == "short"
    assert response["long_net_return_after_all_cost_pct"] == -0.2
    assert response["short_net_return_after_all_cost_pct"] == 0.3
    assert "long_return_pct" not in response

    with pytest.raises(ValueError, match="missing long_net_return_after_all_cost_pct"):
        training._shadow_response({"long_return_pct": 9.0, "short_return_pct": -9.0})


def test_remote_json_parser_returns_root_object_instead_of_last_nested_object() -> None:
    raw = 'remote-prefix\n{"verified":true,"adapter_path":"/adapter","manifest":{"training_config":{"rank":8}}}'

    parsed = training._json_object_from_remote_output(raw)

    assert parsed["verified"] is True
    assert parsed["adapter_path"] == "/adapter"
    assert parsed["manifest"]["training_config"] == {"rank": 8}


def test_8003_gateway_requires_a_real_adapter_and_does_not_rename_base_model() -> None:
    with pytest.raises(ValueError, match="verified BB-FinQuant adapter"):
        training._remote_service_update_script(adapter_path="")

    rendered = training._remote_service_update_script(
        adapter_path="/data/BB/models/finquant_lora/versions/20260712T010203Z-aaaaaaaaaaaa"
    )

    assert "--enable-lora" in rendered
    assert "--lora-modules BB-FinQuant-Expert-14B=" in rendered
    assert 'UPSTREAM_MODEL = "BB-FinQuant-Expert-14B"' in rendered
    assert 'payload["model"] = UPSTREAM_MODEL' not in rendered
    assert '"parent": FALLBACK_MODEL' not in rendered
    assert "finquant_adapter_not_loaded" in rendered
    assert "bb-phase3-llm-expert.service" in rendered
    assert "bb-phase3-llm-decision.service" in rendered
    assert "systemctl disable --now 'bb-finquant-expert-alias.service'" in rendered


def test_remote_trainer_and_registry_enforce_hashes_atomic_pointers_and_rollback() -> None:
    compile(training.REMOTE_TRAINER_CODE, "remote_finquant_trainer.py", "exec")
    compile(training.REMOTE_REGISTRY_TOOL_CODE, "remote_finquant_registry.py", "exec")

    assert "refusing to overwrite adapter version" in training.REMOTE_TRAINER_CODE
    assert "dataset SHA-256 does not match its manifest" in training.REMOTE_TRAINER_CODE
    assert '"adapter_files": adapter_files' in training.REMOTE_TRAINER_CODE
    assert '"held_out_eval_loss": eval_loss' in training.REMOTE_TRAINER_CODE
    assert "from trl import DPOConfig, DPOTrainer" in training.REMOTE_TRAINER_CODE
    assert '"training_stages": ["sft_format_domain", "trl_dpo_return_preference"]' in (
        training.REMOTE_TRAINER_CODE
    )
    assert '"preference_selection_accuracy": preference_selection_accuracy' in (
        training.REMOTE_TRAINER_CODE
    )
    assert '"training_config": {' in training.REMOTE_TRAINER_CODE
    assert '"trainer_code_sha256": sha256_file(Path(__file__))' in training.REMOTE_TRAINER_CODE
    assert "os.replace(staging_dir, output_dir)" in training.REMOTE_TRAINER_CODE
    assert "os.replace(temporary, path)" in training.REMOTE_REGISTRY_TOOL_CODE
    assert 'subparsers.add_parser("rollback")' in training.REMOTE_REGISTRY_TOOL_CODE
    assert 'RETIRED = ROOT / "retired"' in training.REMOTE_REGISTRY_TOOL_CODE
    assert '"can_influence_live": False' in training.REMOTE_REGISTRY_TOOL_CODE
    assert '"retired_incompatible_previous": retired_previous' in (
        training.REMOTE_REGISTRY_TOOL_CODE
    )
    assert '"retired_incompatible_rollback": retired_rollback' in (
        training.REMOTE_REGISTRY_TOOL_CODE
    )
    assert (
        'previous.get("adapter_path") == new_pointer.get("adapter_path")'
        in training.REMOTE_REGISTRY_TOOL_CODE
    )
    assert "ROLLBACK.unlink()" in training.REMOTE_REGISTRY_TOOL_CODE
    assert 'subparsers.add_parser("status")' in training.REMOTE_REGISTRY_TOOL_CODE
    assert "validate_pointer(target)" in training.REMOTE_REGISTRY_TOOL_CODE
    assert '"verification_status": verification_status' in training.REMOTE_REGISTRY_TOOL_CODE
    assert '"objective_name": manifest.get("objective_name")' in training.REMOTE_REGISTRY_TOOL_CODE
    assert '"objective_version": manifest.get("objective_version")' in (
        training.REMOTE_REGISTRY_TOOL_CODE
    )
    assert '"training_stages": manifest.get("training_stages")' in (
        training.REMOTE_REGISTRY_TOOL_CODE
    )


def test_training_refuses_to_run_while_conflicting_14b_services_remain_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    dataset, manifest = _dataset_contract(monkeypatch)

    with pytest.raises(ValueError, match="stopping all conflicting 14B services"):
        training.deploy_and_optionally_train(
            dataset_jsonl=dataset,
            manifest_json=json.dumps(manifest),
            train=True,
            switch_service=True,
            stop_inference_for_training=False,
            max_steps=1,
        )


def test_rollback_switch_uses_existing_remote_registry_without_dataset_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    closed: list[bool] = []

    class Connection:
        def close(self) -> None:
            closed.append(True)

    connection = Connection()
    monkeypatch.setattr(training, "load_model_server_info_from_platform", lambda _path: object())
    monkeypatch.setattr(training, "connect_remote_ssh", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(
        training,
        "_switch_verified_adapter",
        lambda actual_ssh, *, rollback_service: {
            "same_connection": actual_ssh is connection,
            "rollback": rollback_service,
        },
    )

    result = training.rollback_and_switch_service()

    assert result == {"same_connection": True, "rollback": True}
    assert closed == [True]


def test_retired_takeover_script_is_deleted() -> None:
    assert not (training.ROOT / "scripts" / "deploy_old_model_server_takeover.py").exists()
