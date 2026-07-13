from types import SimpleNamespace
from typing import Any

import pytest

from services.position_protection_fallback import PositionProtectionFallbackPolicy


class _FakeResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar_one_or_none(self) -> Any:
        return self.value


class _FakeSession:
    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.execute_count = 0

    async def execute(self, _statement: Any) -> _FakeResult:
        self.execute_count += 1
        return _FakeResult(self.values.pop(0) if self.values else None)


def _decision(**kwargs: Any) -> SimpleNamespace:
    provenance = {
        "source": "dynamic_return_risk_plan",
        "observation_window": "current_decision",
        "sample_count": 3,
        "generated_at": "2026-07-12T00:00:00+00:00",
        "strategy_version": "test.v1",
        "fallback_reason": "",
    }
    defaults = {
        "id": 42,
        "raw_llm_response": {
            "production_return_policy": {"eligible": True},
            "profit_risk_sizing": {
                "production_eligible": True,
                "stress_stop_loss_pct": 0.05,
                "policy_provenance": provenance,
            },
        },
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_exact_order_dynamic_plan_recovers_stop_only() -> None:
    session = _FakeSession(_decision(id=101))
    result = await PositionProtectionFallbackPolicy().protection_from_decision(
        session,
        symbol="BTC/USDT",
        side="long",
        entry_price=100.0,
        order=SimpleNamespace(decision_id=101),
    )

    assert result["stop_loss_price"] == 95.0
    assert result["take_profit_price"] is None
    assert result["source"] == "exact_order_dynamic_risk_plan"
    assert result["decision_id"] == 101
    assert session.execute_count == 1


@pytest.mark.asyncio
async def test_missing_order_link_never_reuses_latest_symbol_decision() -> None:
    session = _FakeSession(_decision(id=202))
    result = await PositionProtectionFallbackPolicy().protection_from_decision(
        session,
        symbol="ETH/USDT",
        side="short",
        entry_price=100.0,
    )

    assert result == {}
    assert session.execute_count == 0


@pytest.mark.asyncio
async def test_missing_linked_decision_never_searches_for_another_decision() -> None:
    session = _FakeSession(None, _decision(id=303))
    result = await PositionProtectionFallbackPolicy().protection_from_decision(
        session,
        symbol="SOL/USDT",
        side="long",
        entry_price=50.0,
        order=SimpleNamespace(decision_id=999),
    )

    assert result == {}
    assert session.execute_count == 1


@pytest.mark.asyncio
async def test_incomplete_or_legacy_plan_is_rejected() -> None:
    legacy = _decision(raw_llm_response={}, stop_loss_pct=0.05, take_profit_pct=0.2)
    result = await PositionProtectionFallbackPolicy().protection_from_decision(
        _FakeSession(legacy),
        symbol="BTC/USDT",
        side="long",
        entry_price=100.0,
        order=SimpleNamespace(decision_id=legacy.id),
    )

    assert result == {}
