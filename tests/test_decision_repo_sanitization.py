from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from db.repositories.decision_repo import DecisionRepository

BAD_REASON = "AI 选择观望，未提交订单。?"
GOOD_REASON = "AI 选择观望，未提交订单。"


class FakeSession:
    def __init__(self, row: Any | None = None) -> None:
        self.row = row
        self.added: Any | None = None
        self.flush_count = 0

    def add(self, instance: Any) -> None:
        self.added = instance

    async def flush(self) -> None:
        self.flush_count += 1

    async def get(self, _model: Any, _row_id: int) -> Any:
        return self.row


@pytest.mark.asyncio
async def test_log_decision_sanitizes_text_and_json_payloads() -> None:
    session = FakeSession()
    repo = DecisionRepository(session)  # type: ignore[arg-type]

    decision = await repo.log_decision(
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "action": "hold",
            "confidence": 0.1,
            "reasoning": BAD_REASON,
            "execution_reason": BAD_REASON,
            "position_size_pct": 0.0,
            "suggested_leverage": 1.0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "feature_snapshot": {"reason": BAD_REASON},
            "raw_llm_response": {"notes": [BAD_REASON]},
            "analysis_type": "market",
            "is_paper": True,
        }
    )

    assert session.added is decision
    assert decision.reasoning == GOOD_REASON
    assert decision.execution_reason == GOOD_REASON
    assert decision.feature_snapshot == {"reason": GOOD_REASON}
    assert decision.raw_llm_response == {"notes": [GOOD_REASON]}
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_decision_repo_update_methods_sanitize_text_and_json() -> None:
    row = SimpleNamespace(
        execution_reason=None,
        raw_llm_response=None,
        position_size_pct=0.25,
        suggested_leverage=5.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    session = FakeSession(row=row)
    repo = DecisionRepository(session)  # type: ignore[arg-type]

    await repo.mark_execution_reason(123, BAD_REASON)
    await repo.update_raw_response(
        123,
        {
            "reason": BAD_REASON,
            "execution_parameters": {
                "position_size_pct": 0.004,
                "suggested_leverage": 2.0,
                "stop_loss_pct": 0.012,
                "take_profit_pct": 0.044,
            },
        },
    )

    assert row.execution_reason == GOOD_REASON
    assert row.raw_llm_response == {
        "reason": GOOD_REASON,
        "execution_parameters": {
            "position_size_pct": 0.004,
            "suggested_leverage": 2.0,
            "stop_loss_pct": 0.012,
            "take_profit_pct": 0.044,
        },
    }
    assert row.position_size_pct == 0.004
    assert row.suggested_leverage == 2.0
    assert row.stop_loss_pct == 0.012
    assert row.take_profit_pct == 0.044
    assert session.flush_count == 2
