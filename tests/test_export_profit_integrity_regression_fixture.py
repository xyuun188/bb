import json
from pathlib import Path

import pytest

from scripts import export_profit_integrity_regression_fixture as fixture_export

FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "profit_integrity"
    / "2026-07-14-root-cause-baseline.json"
)


def test_remote_template_compiles_and_is_read_only() -> None:
    template = fixture_export.REMOTE_SCRIPT_TEMPLATE.replace(
        "__DECISION_IDS__", "[79318]"
    ).replace("__LOCAL_GIT_HEAD__", "a" * 40)

    compile(template, "<profit-integrity-fixture>", "exec")
    assert "get_read_session_ctx" in template
    assert "MLSignalService().status()" in template
    assert "get_position_protection_orders(None)" in template
    assert '"audit_only": True' in template
    assert '"live_mutation": False' in template
    assert "cancel" not in template.lower()
    assert "commit()" not in template


def test_remote_command_is_scoped_and_includes_incident_ids() -> None:
    output_path = fixture_export._remote_result_path("abc123")
    command = fixture_export._build_remote_command(token="abc123", output_path=output_path)

    assert fixture_export.REMOTE_TMP_DIR in command
    assert "79318" in command
    assert "79568" in command
    assert "80855" in command
    assert "--property=User=bb" in command
    assert "chown bb:bb" in command
    assert output_path in command


def test_remote_command_rejects_injection_and_external_paths() -> None:
    with pytest.raises(ValueError):
        fixture_export._build_remote_command(token="bad;rm")
    with pytest.raises(ValueError):
        fixture_export._build_remote_command(token="safe", output_path="C:/outside/out.json")
    with pytest.raises(ValueError):
        fixture_export._build_remote_command((), token="safe")
    with pytest.raises(ValueError):
        fixture_export._build_remote_command(token="safe", git_head="not-a-sha")


def test_summary_exposes_artifact_and_protection_inventory() -> None:
    summary = fixture_export._summary(
        {
            "fixture_version": "fixture-v1",
            "decisions": [{"id": 1}],
            "shadow_backtests": [{"id": 2}],
            "orders": [],
            "positions": [{"id": 3}],
            "local_ml_artifact": {
                "live_ml_ready": False,
                "registry": {"version": "artifact-v1"},
            },
            "okx_protection_inventory": {
                "orphan_keys": [["TRX/USDT", "long"]],
                "duplicate_keys": [["IRYS/USDT", "long"]],
                "missing_protection_keys": [],
            },
        }
    )

    assert summary["artifact_version"] == "artifact-v1"
    assert summary["artifact_live_influence"] is False
    assert summary["orphan_protection_count"] == 1
    assert summary["duplicate_protection_count"] == 1


def test_parse_remote_payload_rejects_logs_without_fixture_and_accepts_prefix() -> None:
    payload = fixture_export._parse_remote_payload(
        b'2026-07-14 [info] initialized\n{"fixture_version":"fixture-v1","audit_only":true}'
    )

    assert payload == {"fixture_version": "fixture-v1", "audit_only": True}
    with pytest.raises(ValueError, match="marker is missing"):
        fixture_export._parse_remote_payload(b"only initialization logs")


def test_root_cause_fixture_preserves_incident_and_rollback_evidence() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert payload["audit_only"] is True
    assert payload["live_mutation"] is False
    assert payload["decision_ids"] == list(fixture_export.INCIDENT_DECISION_IDS)
    assert payload["git_head"] == "89550d916085395e5a61fbdf1a109324a8edfc46"
    assert all(len(value) == 64 for value in payload["source_sha256"].values())

    artifact = payload["local_ml_artifact"]
    assert artifact["registry"]["version"] == "20260714T095512798363Z-57c22201"
    assert artifact["live_ml_ready"] is True
    metrics = artifact["manifest"]["metrics"]
    assert metrics["top_short_profit_factor"] == 1_000_000.0
    assert metrics["top_short_avg_return_pct"] > 70.0
    assert metrics["bottom_long_avg_return_pct"] < -100.0

    decisions = {row["id"]: row for row in payload["decisions"]}
    icp = decisions[79318]
    icp_prediction = icp["production_evidence"]["ml_signal"]["predictions"][0]
    assert icp_prediction["short_lower_quantile_return_pct"] > icp_prediction[
        "short_expected_return_pct"
    ]
    assert icp["production_evidence"]["profit_risk_sizing"][
        "position_size_pct"
    ] == pytest.approx(0.52154988)

    robo_decisions = [decisions[row_id] for row_id in fixture_export.INCIDENT_DECISION_IDS[2:]]
    assert {row["market_fact"]["current_price"] for row in robo_decisions} == {0.10834}
    assert {row["market_fact"]["notional_24h_usdt"] for row in robo_decisions} == {0.0}
    assert all(row["market_fact"]["spread_pct"] > 28.0 for row in robo_decisions)
    robo_shadows = [
        row for row in payload["shadow_backtests"] if row["symbol"] == "ROBO/USDT"
    ]
    assert len(robo_shadows) == 18
    assert all(row["actual_price"] < 0.014 for row in robo_shadows)
    assert all(row["short_return_pct"] > 87.0 for row in robo_shadows)

    learning = payload["position_4879_learning"]
    assert len(learning["reflections"]) == 1
    assert learning["expert_memories"] == []
    inventory = payload["okx_protection_inventory"]
    assert inventory["missing_protection_keys"] == []
    assert inventory["duplicate_keys"] == [["IRYS/USDT", "short"]]
    assert inventory["orphan_keys"] == [
        ["BCH/USDT", "long"],
        ["SOL/USDT", "long"],
        ["TRX/USDT", "long"],
        ["TRX/USDT", "short"],
    ]
