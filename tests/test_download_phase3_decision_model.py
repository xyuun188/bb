from __future__ import annotations

from scripts import download_phase3_decision_model as download


def test_phase3_decision_downloader_targets_data_bb_32b_model() -> None:
    script = download.render_remote_downloader()

    assert download.MODEL_REPO == "Qwen/Qwen3-32B-AWQ"
    assert download.TARGET_DIR == "/data/BB/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ"
    assert "/data/trade_models" not in script
    assert "llm_decision_maker" in script
    assert '"repo_id": MODEL_REPO' in script
    assert '"live_routing_enabled": False' in script
    assert "/data/BB/reports/inventory/phase3_model_download_manifest_latest.json" in script
    assert "/data/BB/reports/inventory/phase3_model_validation_latest.json" in script
    assert '"decision_maker": MODEL_REPO' in script
    assert '"llm_live_routing_enabled"] = False' in script
    assert "snapshot_download(" in script


def test_phase3_decision_downloader_start_command_records_pid() -> None:
    command = download._start_command()

    assert "nohup" in command
    assert "echo $!" in command
    assert download.PID_PATH in command
    assert download.LOG_PATH in command
