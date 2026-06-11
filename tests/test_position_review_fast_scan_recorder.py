from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ai_brain.base_model import DecisionOutput
from services.position_review_fast_scan_hold import PositionReviewFastScanHoldPolicy
from services.position_review_fast_scan_recorder import PositionReviewFastScanRecorder
from services.position_review_outcome import PositionReviewOutcomePolicy
from services.position_review_result_recorder import PositionReviewResultRecorder


class _Skill:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name}


class _FeatureVector:
    def to_dict(self) -> dict[str, Any]:
        return {"current_price": 100.0}


def _result_recorder(calls: list[tuple[str, Any]]) -> PositionReviewResultRecorder:
    async def mark_reason(decision_id: int, reason: str) -> None:
        calls.append(("result_reason", decision_id, reason))

    async def mark_raw(decision_id: int, raw_response: dict[str, Any]) -> None:
        calls.append(("raw", decision_id, raw_response))

    async def log_risk(decision: DecisionOutput, model_name: str, reason: str) -> None:
        calls.append(("risk", decision.symbol, model_name, reason))

    return PositionReviewResultRecorder(
        outcome_policy=PositionReviewOutcomePolicy(),
        decision_reason_marker=mark_reason,
        decision_raw_response_marker=mark_raw,
        risk_result_logger=log_risk,
    )


def _recorder(
    calls: list[tuple[str, Any]],
    *,
    decision_id: int | None,
) -> PositionReviewFastScanRecorder:
    applied_defer_counts: dict[tuple[str, str], int] = {}

    async def log_decision(decision: DecisionOutput, is_paper: bool) -> int | None:
        calls.append(
            (
                "log",
                decision.model_name,
                decision.symbol,
                decision.feature_snapshot,
                is_paper,
            )
        )
        return decision_id

    async def mark_reason(decision_id: int, reason: str) -> None:
        calls.append(("reason", decision_id, reason))

    def apply_defer_count(key: tuple[str, str], value: int) -> None:
        applied_defer_counts[key] = value
        calls.append(("defer", key, value))

    return PositionReviewFastScanRecorder(
        default_model_name="ensemble_trader",
        normalize_symbol=lambda symbol: str(symbol).upper(),
        urgent_exit_checker=lambda scan: bool(scan and scan.get("urgent")),
        portfolio_symbol_context_provider=lambda _ctx, _model, _symbol, _positions: {
            "active": True,
            "is_focus": True,
        },
        position_skills_provider=lambda **_kwargs: [_Skill("risk")],
        agent_skills_summary_provider=lambda skills: {"count": len(skills)},
        defer_count_provider=lambda key: applied_defer_counts.get(key, 0),
        defer_count_applier=apply_defer_count,
        model_execution_mode_provider=lambda _model_name: "paper",
        decision_logger=log_decision,
        decision_reason_marker=mark_reason,
        result_recorder=_result_recorder(calls),
        hold_policy=PositionReviewFastScanHoldPolicy(
            clock=lambda: datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
        ),
    )


@pytest.mark.asyncio
async def test_fast_scan_recorder_logs_decision_and_result_row() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    round_decision_ids: set[int] = set()

    logged = await _recorder(calls, decision_id=42).record_many(
        skipped_items=[(("ensemble_trader", "BTC/USDT"), [{"side": "long"}])],
        fast_scan={
            ("ensemble_trader", "BTC/USDT"): {
                "priority_score": 88.0,
                "exit_score": 90.0,
                "add_score": 3.0,
                "reason": "profit_lock_candidate",
                "urgent": True,
            }
        },
        feature_vectors={"BTC/USDT": _FeatureVector()},
        portfolio_profit_context={"active": True},
        results=results,
        round_decision_ids=round_decision_ids,
        position_entry_pause_reason=None,
    )

    assert logged == 1
    assert round_decision_ids == {42}
    assert ("defer", ("ensemble_trader", "BTC/USDT"), 1) in calls
    assert ("log", "ensemble_trader", "BTC/USDT", {"current_price": 100.0}, True) in calls
    assert any(call[0] == "reason" and call[1] == 42 for call in calls)
    assert results["decisions"][0]["execution_status"] == "fast_position_scan"
    assert results["decisions"][0]["reason"]
    assert "profit_lock_candidate" in results["decisions"][0]["reason"]


@pytest.mark.asyncio
async def test_fast_scan_recorder_keeps_dashboard_row_when_decision_log_fails() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    round_decision_ids: set[int] = set()

    logged = await _recorder(calls, decision_id=None).record_many(
        skipped_items=[(("ensemble_trader", "ETH/USDT"), [])],
        fast_scan={("ensemble_trader", "ETH/USDT"): {"priority_score": 12.0}},
        feature_vectors={},
        portfolio_profit_context=None,
        results=results,
        round_decision_ids=round_decision_ids,
    )

    assert logged == 0
    assert round_decision_ids == set()
    assert not any(call[0] == "reason" for call in calls)
    assert results["decisions"][0]["symbol"] == "ETH/USDT"
    assert results["decisions"][0]["execution_status"] == "fast_position_scan"
