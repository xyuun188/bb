from __future__ import annotations

from services.entry_symbol_profit_quarantine import EntrySymbolProfitQuarantinePolicy


def test_symbol_profit_quarantine_returns_advisory_reason_for_cooldown_profile() -> None:
    policy = EntrySymbolProfitQuarantinePolicy(
        normalize_symbol=lambda symbol: str(symbol or "").replace("-", "/")
    )

    reason = policy.reason(
        "BTC-USDT",
        {
            "symbol_side_performance": {
                "BTC/USDT|all": {
                    "cooldown": True,
                    "count": 3,
                    "losses": 2,
                    "pnl": -128.75,
                    "cooldown_reason": "该币种最近滚动真实亏损过大",
                }
            }
        },
    )

    assert reason is not None
    assert "BTC/USDT 进入亏损隔离观察" in reason
    assert "最近 3 笔真实平仓累计 -128.75 U" in reason
    assert "不会直接拦截 AI 分析" in reason


def test_symbol_profit_quarantine_ignores_missing_or_inactive_profile() -> None:
    policy = EntrySymbolProfitQuarantinePolicy()

    assert policy.reason("BTC/USDT", None) is None
    assert policy.reason("BTC/USDT", {}) is None
    assert (
        policy.reason(
            "BTC/USDT",
            {"symbol_side_performance": {"BTC/USDT|all": {"cooldown": False}}},
        )
        is None
    )
