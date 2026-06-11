from __future__ import annotations

import pytest

from services.entry_suspicious_symbol import EntrySuspiciousSymbolPolicy


def _normalize(symbol: str | None) -> str:
    if not symbol:
        return ""
    normalized = str(symbol).split(":", 1)[0]
    if normalized.endswith("-SWAP"):
        normalized = normalized[:-5]
    if "/" not in normalized and "-" in normalized:
        parts = normalized.split("-")
        if len(parts) >= 2:
            normalized = f"{parts[0]}/{parts[1]}"
    return normalized


@pytest.mark.parametrize("symbol", [None, "", "   "])
def test_suspicious_symbol_policy_blocks_blank_symbols(symbol: str | None) -> None:
    assert (
        EntrySuspiciousSymbolPolicy(_normalize).reason(symbol) == "币种符号为空，跳过新开仓分析。"
    )


@pytest.mark.parametrize(
    ("symbol", "normalized"),
    [
        ("TEST-USDT-SWAP", "TEST/USDT"),
        ("demo/USDT", "DEMO/USDT"),
        ("mock-USDT", "MOCK/USDT"),
    ],
)
def test_suspicious_symbol_policy_blocks_test_and_placeholder_symbols(
    symbol: str,
    normalized: str,
) -> None:
    reason = EntrySuspiciousSymbolPolicy(_normalize).reason(symbol)

    assert reason is not None
    assert normalized in reason
    assert "测试/模拟/占位合约" in reason


def test_suspicious_symbol_policy_allows_normal_symbols() -> None:
    assert EntrySuspiciousSymbolPolicy(_normalize).reason("BTC-USDT-SWAP") is None
