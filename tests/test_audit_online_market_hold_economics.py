from __future__ import annotations

import io
import json
from unittest.mock import Mock

import pytest

from scripts import audit_online_market_hold_economics as audit


def test_remote_audit_compiles_and_reads_current_consultation_contract() -> None:
    compile(audit.REMOTE_SCRIPT, "<online-market-hold-economics>", "exec")

    assert 'raw.get("consultation")' in audit.REMOTE_SCRIPT
    assert 'raw.get("conflict_consultation")' in audit.REMOTE_SCRIPT
    assert 'consultation.get("consultation_attempts")' in audit.REMOTE_SCRIPT
    assert 'consultation.get("attempts")' in audit.REMOTE_SCRIPT
    assert "consultation_by_analysis_type" in audit.REMOTE_SCRIPT
    assert "production_permission" in audit.REMOTE_SCRIPT


def test_remote_audit_queries_market_economics_and_all_analysis_consultations() -> None:
    assert 'AIDecision.analysis_type == "market"' in audit.REMOTE_SCRIPT
    assert "consultation_rows" in audit.REMOTE_SCRIPT
    assert "non_market_consultation_rows" in audit.REMOTE_SCRIPT
    assert 'AIDecision.analysis_type != "market"' in audit.REMOTE_SCRIPT
    assert "AIDecision.raw_llm_response" in audit.REMOTE_SCRIPT
    assert "AIDecision.analysis_type," in audit.REMOTE_SCRIPT


def test_remote_audit_projects_only_required_columns_and_bounds_duplicate_scan() -> None:
    assert "select(AIDecision)" not in audit.REMOTE_SCRIPT
    assert "AIDecision.model_health_opinions," in audit.REMOTE_SCRIPT
    assert "AIDecision.model_health_timings," in audit.REMOTE_SCRIPT
    duplicate_scan = audit.REMOTE_SCRIPT.split("duplicate_rows =", 1)[1]
    assert ".where(\n                    *filters," in duplicate_scan


def test_remote_audit_reports_observation_starvation_economics() -> None:
    compile(audit.REMOTE_SCRIPT, "<online-market-hold-economics>", "exec")

    assert "preferred_paper_observation_side" in audit.REMOTE_SCRIPT
    assert "positive_mean_non_positive_lcb" in audit.REMOTE_SCRIPT
    assert "quality_permission_blockers" in audit.REMOTE_SCRIPT
    assert "gross_expected_return_pct" in audit.REMOTE_SCRIPT
    assert "execution_cost_pct" in audit.REMOTE_SCRIPT
    assert "lcb_penalty_pct" in audit.REMOTE_SCRIPT
    assert "directional_decision_examples" in audit.REMOTE_SCRIPT
    assert "profit_risk_sizing" in audit.REMOTE_SCRIPT
    assert "high_risk_review" in audit.REMOTE_SCRIPT
    assert "hard_review_required" in audit.REMOTE_SCRIPT
    assert "execution_result" in audit.REMOTE_SCRIPT
    assert "exchange_order_id" in audit.REMOTE_SCRIPT


def test_remote_audit_reads_large_json_without_log_truncation(monkeypatch) -> None:
    expected = {"row_count": 500, "evidence": "x" * 150_000}
    sftp = Mock()
    sftp.file.return_value = io.BytesIO(json.dumps(expected).encode("utf-8"))
    ssh = Mock()
    ssh.open_sftp.return_value = sftp
    run = Mock(return_value="diagnostic output is not the report")
    monkeypatch.setattr(audit, "run_remote_text", run)
    monkeypatch.setattr(audit.secrets, "token_hex", lambda _: "testtoken")

    assert audit._read_remote_report(ssh, 500, 90) == expected

    path = "/data/bb/app/tmp/codex-market-hold-economics/result_testtoken.json"
    command = run.call_args.args[1]
    assert command.endswith(f"> {path}")
    assert "umask 077" in command
    assert "install -d -m 0700" in command
    assert run.call_args.kwargs["max_output_chars"] == 4000
    sftp.file.assert_called_once_with(path, "r")
    sftp.remove.assert_called_once_with(path)
    sftp.close.assert_called_once()


@pytest.mark.parametrize("output", [b"not-json", b"[]"])
def test_remote_audit_cleans_up_invalid_report(monkeypatch, output) -> None:
    sftp = Mock()
    sftp.file.return_value = io.BytesIO(output)
    ssh = Mock()
    ssh.open_sftp.return_value = sftp
    monkeypatch.setattr(audit, "run_remote_text", Mock(return_value=""))

    with pytest.raises(ValueError):
        audit._read_remote_report(ssh, 500, 90)

    sftp.remove.assert_called_once()
    sftp.close.assert_called_once()


def test_remote_audit_preserves_execution_failure_during_cleanup(monkeypatch) -> None:
    sftp = Mock()
    sftp.remove.side_effect = FileNotFoundError
    ssh = Mock()
    ssh.open_sftp.return_value = sftp
    monkeypatch.setattr(
        audit, "run_remote_text", Mock(side_effect=RuntimeError("remote audit failed"))
    )

    with pytest.raises(RuntimeError, match="remote audit failed"):
        audit._read_remote_report(ssh, 500, 90)

    sftp.file.assert_not_called()
    sftp.remove.assert_called_once()
    sftp.close.assert_called_once()
