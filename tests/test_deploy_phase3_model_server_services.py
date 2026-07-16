from __future__ import annotations

import json

import pytest

from scripts import deploy_phase3_model_server_services as deploy


def test_phase3_service_manifest_is_shadow_only_and_under_data_bb() -> None:
    payload = json.loads(deploy.render_manifest())

    assert payload["policy_id"] == deploy.POLICY_ID
    assert payload["phase3_root"] == "/data/BB"
    assert payload["shadow_only"] is True
    assert payload["live_routing_enabled"] is False
    assert payload["can_start_trading"] is False
    assert len(payload["services"]) == 3
    assert {item["slot"] for item in payload["services"]} == {
        "llm_decision_maker",
        "llm_expert_pool",
        "llm_high_risk_review",
    }
    assert {item["port"] for item in payload["services"]} == {8000, 8002, 8003}
    assert 8001 not in {item["port"] for item in payload["services"]}
    by_slot = {item["slot"]: item for item in payload["services"]}
    assert by_slot["llm_decision_maker"]["served_model_name"] == "qwen3-14b-trade"
    assert by_slot["llm_decision_maker"]["model_dir"].endswith("Qwen3-14B-AWQ")
    assert by_slot["llm_decision_maker"]["cuda_visible_devices"] == "0"
    assert by_slot["llm_decision_maker"]["tensor_parallel_size"] == 1
    assert by_slot["llm_expert_pool"]["served_model_name"] == "BB-FinQuant-Expert-14B"
    assert by_slot["llm_expert_pool"]["cuda_visible_devices"] == "0"
    for item in payload["services"]:
        assert item["model_dir"].startswith(("/data/BB/models/", "/data/trade_models/"))
        assert item["start_script_path"].startswith("/data/BB/scripts/")
        assert item["staged_service_path"].startswith("/data/BB/services/systemd/")
        assert item["log_path"].startswith("/data/BB/logs/services/")
        assert item["shadow_only"] is True
        assert item["live_routing_enabled"] is False


def test_direct_phase3_vllm_installer_is_retired() -> None:
    with pytest.raises(RuntimeError, match="migrate_phase3_model_service_identity"):
        deploy.install_services()


def test_phase3_systemd_services_do_not_reference_trading_service() -> None:
    for spec in deploy.PHASE3_SERVICE_SPECS:
        service = spec.render_systemd_service()

        assert "bb-paper-trading.service" not in service
        assert "run_paper_trading" not in service
        assert f"ExecStart={spec.start_script_path}" in service
        assert "BB_PHASE3_ROOT=/data/BB" in service


def test_phase3_plan_only_does_not_connect(monkeypatch, capsys) -> None:
    deploy.install_services(plan_only=True)

    output = capsys.readouterr().out
    assert "phase3_quant_model_services_shadow_only_2026_06_27" in output
    assert "bb-phase3-llm-decision.service" in output
