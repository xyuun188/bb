from __future__ import annotations

import json

import pytest

from core.phase3_model_contract import PHASE3_TARGET_MODEL_REPO_ID
from scripts import sync_phase3_model_inventory as sync


def _rendered_namespace() -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "phase3_inventory_test"}
    exec(  # noqa: S102 - repository-owned generated script under test.
        compile(sync.render_target_inventory_sync(), "<inventory-sync>", "exec"),
        namespace,
    )
    return namespace


def test_target_inventory_sync_writes_one_non_live_llm_identity() -> None:
    script = sync.render_target_inventory_sync()

    assert sync.REPORT_DOWNLOAD_MANIFEST in script
    assert sync.REPORT_VALIDATION_MANIFEST in script
    assert sync.TARGET_CANDIDATE_MANIFEST in script
    assert PHASE3_TARGET_MODEL_REPO_ID in script
    assert 'value.get("model_id") != "qwen3.8-27b"' in script
    assert '"slot": "llm_decision_and_expert_carrier"' in script
    assert '"topology_profile": "target_single_model"' in script
    assert '"live_routing_enabled": False' in script
    assert "http://127.0.0.1:8101/health" in script
    assert "HEALTH_RESPONSE_MAX_BYTES + 1" in script
    assert "BB-FinQuant-Expert-14B" not in script
    assert "deepseek-r1-14b-risk" not in script
    assert "qwen3-14b" not in script


def test_target_inventory_update_replaces_all_llm_rows() -> None:
    namespace = _rendered_namespace()
    update = namespace["update"]
    row = {
        "slot": "llm_decision_and_expert_carrier",
        "served_model_name": "qwen3.8-27b",
    }

    updated = update(
        {
            "models": [
                {"slot": "llm_decision_maker", "served_model_name": "retired"},
                {"slot": "timeseries", "served_model_name": "timesfm"},
            ]
        },
        row,
    )

    llm_rows = [
        item
        for item in updated["models"]
        if str(item.get("slot") or "").startswith("llm_")
    ]
    assert llm_rows == [row]
    assert updated["topology_profile"] == "target_single_model"
    assert updated["live_routing_enabled"] is False
    assert any(item.get("slot") == "timeseries" for item in updated["models"])


def test_target_inventory_rejects_unverified_candidate(tmp_path) -> None:
    namespace = _rendered_namespace()
    candidate_path = tmp_path / "target_model_candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "status": "pending",
                "model_id": "qwen3.8-27b",
                "repo_id": PHASE3_TARGET_MODEL_REPO_ID,
                "revision": "a" * 40,
                "model_path": "/data/models/qwen3.8-27b",
            }
        ),
        encoding="utf-8",
    )
    namespace["CANDIDATE_MANIFEST"] = candidate_path

    with pytest.raises(ValueError, match="not verified"):
        namespace["candidate"]()


def test_target_inventory_dry_run_does_not_connect(monkeypatch, capsys) -> None:
    def fail_connect(*_args, **_kwargs):
        raise AssertionError("dry-run must not connect to the model server")

    monkeypatch.setattr(sync, "connect_remote_ssh", fail_connect)

    assert sync.main(["--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "phase3_model_download_manifest_latest.json" in output
    assert "target_single_model" in output
    assert "llm_decision_and_expert_carrier" in output
    assert "target_model_candidate.json" in output


def test_target_inventory_rejects_removed_profile_argument() -> None:
    with pytest.raises(SystemExit):
        sync.main(["--dry-run", "--profile", "legacy_shadow"])
