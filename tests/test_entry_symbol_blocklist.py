from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_brain.base_model import Action, DecisionOutput
from services.entry_symbol_blocklist import (
    TRANSIENT_ENTRY_BLOCK_MINUTES,
    EntrySymbolBlocklistPolicy,
)


def _normalize(symbol: str | None) -> str | None:
    if not symbol:
        return None
    return str(symbol).replace("-", "/")


def test_entry_symbol_blocklist_classifies_exchange_errors() -> None:
    policy = EntrySymbolBlocklistPolicy(_normalize)

    assert policy.is_untradable_exchange_error("OKX 51155 can't trade this pair")
    assert policy.is_untradable_exchange_error("local compliance restrictions")
    assert policy.is_untradable_exchange_error("OKX 提示该交易对当前不可交易")
    assert policy.is_untradable_exchange_error("交易对当前不可交易，请稍后重试")
    assert policy.is_untradable_exchange_error("instrument suspended")
    assert policy.is_untradable_exchange_error("not available for trading")
    assert policy.is_untradable_exchange_error("okx 51028 Contract under delivery.")
    assert not policy.is_untradable_exchange_error("okx does not have market symbol LAB/USDT:USDT")
    assert not policy.is_untradable_exchange_error("bad symbol")
    assert not policy.is_untradable_exchange_error(
        "OKX market symbol mismatch: requested WLFI/USDT, exchange instrument is H/USDT"
    )
    assert policy.is_transient_entry_exchange_error("51290 engine currently upgrading")
    assert policy.is_transient_entry_exchange_error(
        'Max retries exceeded: okx {"code":"50001","data":[],"msg":"Service temporarily unavailable. Please try again later."}'
    )
    assert policy.is_transient_entry_exchange_error(
        "open interest has reached the platform's limit"
    )
    assert policy.transient_entry_block_minutes("okx 50001 service temporarily unavailable") == 12.0
    assert (
        policy.transient_entry_block_minutes("open interest has reached the platform's limit")
        == 45.0
    )
    assert policy.transient_entry_block_minutes("OKX try again later") == (
        TRANSIENT_ENTRY_BLOCK_MINUTES
    )


def test_entry_symbol_blocklist_remembers_and_expires_temporary_blocks() -> None:
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    policy = EntrySymbolBlocklistPolicy(_normalize, clock=lambda: now)

    policy.remember_temporary_entry_block("BTC-USDT", "下单前价格变化太快", 3.0)

    reason = policy.blocked_symbol_reason("BTC-USDT")
    assert reason is not None
    assert "临时跳过新开仓" in reason
    assert "下单前价格变化太快" in reason

    policy.clock = lambda: now + timedelta(minutes=4)

    assert policy.blocked_symbol_reason("BTC-USDT") is None
    assert policy.blocked_symbols == {}


def test_entry_symbol_blocklist_remembers_untradable_symbol() -> None:
    now = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)
    policy = EntrySymbolBlocklistPolicy(_normalize, clock=lambda: now)

    policy.remember_untradable_symbol("ETH-USDT", "OKX 51155 can't trade this pair", 2.0)

    reason = policy.blocked_symbol_reason("ETH-USDT")
    assert reason is not None
    assert "51155" in reason


def test_entry_symbol_blocklist_recognizes_price_guard_skip_terms() -> None:
    policy = EntrySymbolBlocklistPolicy(_normalize)

    assert policy.is_entry_price_guard_skip("避免追高，系统跳过本次开仓")
    assert policy.is_entry_price_guard_skip("下单前行情质量复核未通过")
    assert not policy.is_entry_price_guard_skip("普通机会评分未达标")


def _entry_decision(action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.82,
        reasoning="recovery test",
        position_size_pct=0.08,
        suggested_leverage=5.0,
        feature_snapshot={
            "market_regime": "trend",
            "adx": 31.24,
            "timestamp": "volatile-value",
        },
        raw_response={"opportunity_score": {"score": 0.78}},
    )


def test_exchange_recovery_block_is_symbol_side_and_evidence_specific() -> None:
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    policy = EntrySymbolBlocklistPolicy(_normalize, clock=lambda: now)
    decision = _entry_decision()
    rejection = {
        "okx_rejection": True,
        "okx_error_code": "59247",
        "okx_error_payload": {
            "code": "0",
            "data": [{"sCode": "59247", "sMsg": "Operation failed"}],
        },
        "leverage_check": {"actual_leverage": 2},
    }

    policy.remember_exchange_rejection(decision, rejection)

    assert "59247" in str(policy.exchange_recovery_block_reason(decision, []))
    assert policy.exchange_recovery_block_reason(_entry_decision(Action.SHORT), []) is None

    decision.feature_snapshot["timestamp"] = "another-volatile-value"
    assert policy.exchange_recovery_block_reason(decision, []) is not None

    decision.feature_snapshot["market_regime"] = "range"
    assert policy.exchange_recovery_block_reason(decision, []) is None


def test_exchange_recovery_block_releases_when_position_leverage_changes() -> None:
    now = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)
    policy = EntrySymbolBlocklistPolicy(_normalize, clock=lambda: now)
    decision = _entry_decision()
    policy.remember_exchange_rejection(
        decision,
        {
            "okx_rejection": True,
            "okx_error_code": "50026",
            "okx_error_payload": {"code": "50026", "msg": "System error. Try again later."},
            "leverage_check": {"actual_leverage": 2},
        },
    )

    assert policy.exchange_recovery_block_reason(
        decision,
        [{"symbol": "BTC/USDT", "side": "long", "leverage": 2}],
    ) is not None
    assert policy.exchange_recovery_block_reason(
        decision,
        [{"symbol": "BTC/USDT", "side": "long", "leverage": 3}],
    ) is None
