from __future__ import annotations

from datetime import UTC, datetime, timedelta

from services.position_group_aggregator import PositionGroupAggregator
from services.position_quality import PositionQualityScorer
from services.trading_service import TradingService


def _service() -> TradingService:
    service = TradingService.__new__(TradingService)
    service.position_group_aggregator = PositionGroupAggregator(
        normalize_symbol=lambda value: str(value or "").split(":")[0],
    )
    service.position_quality_scorer = PositionQualityScorer()
    service._current_capacity_context = {
        "open_group_count": 20,
        "effective_limit": 20,
        "low_quality_count": 1,
    }
    return service


def test_position_release_scan_forces_release_now_before_ai_review() -> None:
    opened_at = datetime.now(UTC) - timedelta(hours=7)

    scan = _service()._position_release_scan(
        model_name="ensemble_trader",
        symbol="AUCTION/USDT",
        normalized_symbol="AUCTION/USDT",
        positions=[
            {
                "model_name": "ensemble_trader",
                "symbol": "AUCTION/USDT",
                "side": "short",
                "quantity": 9.95,
                "contracts": 99.5,
                "contract_size": 0.1,
                "entry_price": 3.532,
                "current_price": 3.528,
                "notional": 35.1434,
                "unrealized_pnl": 0.0398,
                "created_at": opened_at.isoformat(),
                "is_open": True,
            }
        ],
        fast_scan={
            "priority_score": 0.0,
            "exit_score": 0.0,
            "reason": "",
        },
        feature_vector=None,
    )

    assert scan["force_exit_candidate"] is True
    assert scan["release_action"] == "close_short"
    assert scan["exit_score"] == 94.0
    assert scan["position_quality"]["bucket"] == "release_now"
    assert "stale_probe_capital_inefficient" in scan["release_reason"]
