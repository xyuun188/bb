"""Market-analysis prefilter for entry decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MarketDataQualityReason = Callable[..., str | None]


@dataclass(frozen=True, slots=True)
class EntryMarketLLMPrefilterPolicy:
    """Allow AI to see ordinary setups; block only clearly unusable market data."""

    market_data_quality_reason: MarketDataQualityReason
    stage_label: str = "AI分析前"

    def skip_reason(
        self,
        fv: Any,
        local_ai_tools_context: dict[str, Any] | None = None,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> str | None:
        _ = local_ai_tools_context, open_positions
        return self.market_data_quality_reason(fv, stage_label=self.stage_label)
