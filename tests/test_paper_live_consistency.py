from types import SimpleNamespace

import pytest

from services.paper_live_consistency import (
    assert_paper_live_decision_parity,
    compare_paper_live_decisions,
)


def _decision(**overrides: object) -> SimpleNamespace:
    values = {
        "model_name": "ensemble_trader",
        "action": "long",
        "confidence": 0.82,
        "position_size_pct": 0.15,
        "suggested_leverage": 3.0,
        "stop_loss_pct": 0.012,
        "take_profit_pct": 0.024,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_same_snapshot_has_identical_paper_live_contract() -> None:
    report = assert_paper_live_decision_parity(
        _decision(),
        _decision(),
        paper_model_sha256="a" * 64,
        live_model_sha256="a" * 64,
        paper_strategy_fingerprint="strategy-v1",
        live_strategy_fingerprint="strategy-v1",
    )

    assert report["ok"] is True
    assert report["decision_fields_match"] is True
    assert report["paper_decision_fingerprint"] == report["live_decision_fingerprint"]


def test_account_context_is_not_part_of_decision_parity() -> None:
    paper = {**vars(_decision()), "execution_account": "demo"}
    live = {**vars(_decision()), "execution_account": "okx-live"}

    report = assert_paper_live_decision_parity(
        paper,
        live,
        paper_model_sha256="b" * 64,
        live_model_sha256="b" * 64,
        paper_strategy_fingerprint="strategy-v2",
        live_strategy_fingerprint="strategy-v2",
    )

    assert report["ok"] is True


def test_strategy_or_decision_drift_fails_closed() -> None:
    report = compare_paper_live_decisions(
        _decision(position_size_pct=0.2),
        _decision(position_size_pct=0.15),
        paper_model_sha256="c" * 64,
        live_model_sha256="d" * 64,
        paper_strategy_fingerprint="strategy-v1",
        live_strategy_fingerprint="strategy-v2",
    )

    assert report["ok"] is False
    assert report["decision_fields_match"] is False
    assert report["model_sha256_match"] is False
    assert report["strategy_fingerprint_match"] is False
    with pytest.raises(ValueError, match="paper/live parity contract failed"):
        assert_paper_live_decision_parity(
            _decision(position_size_pct=0.2),
            _decision(position_size_pct=0.15),
            paper_model_sha256="c" * 64,
            live_model_sha256="d" * 64,
            paper_strategy_fingerprint="strategy-v1",
            live_strategy_fingerprint="strategy-v2",
        )
