from __future__ import annotations

from typing import Any

from services.entry_market_prefilter import EntryMarketLLMPrefilterPolicy


def test_entry_market_llm_prefilter_delegates_to_data_quality_reason() -> None:
    calls: list[tuple[Any, str]] = []

    def data_quality_reason(fv: Any, *, stage_label: str) -> str | None:
        calls.append((fv, stage_label))
        return "AI分析前没有有效价格，本次不执行新开仓。"

    reason = EntryMarketLLMPrefilterPolicy(data_quality_reason).skip_reason(
        {"symbol": "BTC/USDT"},
        local_ai_tools_context={"status": "ok"},
        open_positions=[{"symbol": "ETH/USDT"}],
    )

    assert reason == "AI分析前没有有效价格，本次不执行新开仓。"
    assert calls == [({"symbol": "BTC/USDT"}, "AI分析前")]


def test_entry_market_llm_prefilter_allows_ordinary_candidate_quality() -> None:
    def data_quality_reason(fv: Any, *, stage_label: str) -> str | None:
        return None

    reason = EntryMarketLLMPrefilterPolicy(data_quality_reason).skip_reason(
        {"symbol": "BTC/USDT"},
        local_ai_tools_context={"loss_probability": 0.7},
        open_positions=[],
    )

    assert reason is None
