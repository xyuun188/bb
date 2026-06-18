from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
