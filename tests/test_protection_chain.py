from datetime import UTC, datetime, timedelta

from core.reason_codes import ReasonCode
from services.protection_chain import (
    ProtectionChain,
    ProtectionObservation,
    ProtectionThresholds,
)


def test_protection_chain_reports_fixed_order_and_complete_evidence() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    report = ProtectionChain(
        ProtectionThresholds(
            max_consecutive_losses=3,
            max_drawdown_pct=8,
            min_pair_trade_count=5,
            min_pair_fee_adjusted_return_pct=0,
            max_spread_bps=20,
            min_liquidity_usdt=200_000,
        )
    ).evaluate(
        ProtectionObservation(
            scope="symbol",
            symbol="XRP/USDT",
            cooldown_until=now + timedelta(minutes=30),
            consecutive_losses=4,
            drawdown_pct=9,
            pair_trade_count=8,
            pair_fee_adjusted_return_pct=-0.2,
            spread_bps=25,
            liquidity_usdt=100_000,
            exchange_healthy=False,
            exchange_health_reason="ticker stale",
            source_event="test",
        ),
        now=now,
    )

    assert report["read_only"] is True
    assert report["is_entry_gate"] is False
    assert report["allowed_by_diagnostics"] is False
    assert [state["rule"] for state in report["states"]] == list(ProtectionChain.ORDER)
    assert report["primary_reason_code"] == ReasonCode.RISK_COOLDOWN
    assert all("observed_value" in state and "threshold" in state for state in report["states"])


def test_manual_override_preserves_original_trigger_evidence() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    report = ProtectionChain().evaluate(
        ProtectionObservation(
            scope="symbol",
            cooldown_until=now + timedelta(minutes=10),
            manual_overrides={
                "cooldown": {
                    "approved": True,
                    "approved_by": "risk_owner",
                    "approved_at": now.isoformat(),
                }
            },
        ),
        now=now,
    )
    state = report["states"][0]
    assert state["triggered"] is True
    assert state["active"] is False
    assert state["reason_evidence"]["release_at"] is not None
    assert state["manual_override"]["approved_by"] == "risk_owner"
