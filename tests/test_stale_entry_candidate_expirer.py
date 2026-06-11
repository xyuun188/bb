from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from services.stale_entry_candidate_expirer import (
    StaleEntryCandidateExpirer,
    action_label,
    is_pending_execution_reason,
    pending_execution_failed_reason,
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row(
    *,
    row_id: int = 1,
    symbol: str = "BTC/USDT",
    action: str = "long",
    raw: Any | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=row_id,
        symbol=symbol,
        action=action,
        raw_llm_response=raw if raw is not None else {},
        execution_reason="",
    )


def test_stale_entry_candidate_reason_helpers() -> None:
    assert action_label("long") == "做多"
    assert action_label("short") == "做空"
    assert is_pending_execution_reason("")
    assert is_pending_execution_reason("正在提交 OKX：下单中")
    assert not is_pending_execution_reason("风险检查拦截")
    assert "45 秒内没有生成本地订单记录" in pending_execution_failed_reason(
        "BTC/USDT",
        "long",
    )


@pytest.mark.asyncio
async def test_stale_entry_candidate_expirer_marks_waiting_rows() -> None:
    waiting = [
        _row(
            raw={
                "opportunity_score": {
                    "score": 0.7,
                    "min_score_required": 0.8,
                    "expected_net_return_pct": 0.2,
                }
            }
        )
    ]
    flushed = False

    async def order_count_provider(_decision_id: int) -> int:
        return 0

    async def flush() -> None:
        nonlocal flushed
        flushed = True

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        waiting,
        [],
        order_count_provider=order_count_provider,
        flush_callback=flush,
    )

    assert expired == 1
    assert flushed
    assert "机会评分 0.7000 低于执行门槛 0.80" in waiting[0].execution_reason
    assert waiting[0].raw_llm_response["opportunity_score"]["selected_for_execution"] is False


@pytest.mark.asyncio
async def test_stale_entry_candidate_expirer_marks_pending_rows_by_order_state() -> None:
    pending_without_order = _row(row_id=11, symbol="ETH/USDT", action="short")
    pending_with_order = _row(row_id=12, symbol="SOL/USDT", action="long")

    async def order_count_provider(decision_id: int) -> int:
        return 1 if decision_id == 12 else 0

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        [],
        [pending_without_order, pending_with_order],
        order_count_provider=order_count_provider,
    )

    assert expired == 2
    assert "45 秒内没有生成本地订单记录" in pending_without_order.execution_reason
    assert "本地订单记录已生成" in pending_with_order.execution_reason
    assert (
        pending_with_order.raw_llm_response["opportunity_score"]["selected_for_execution"] is False
    )
