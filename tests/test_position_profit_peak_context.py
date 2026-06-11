from typing import Any

from services.position_profit_peak_context import PositionProfitPeakContextPolicy
from services.position_profit_peaks import PositionProfitPeakTracker
from services.trading_service import TradingService


def _aggregate(
    positions: list[dict[str, Any]],
    model_name: str,
    symbol: str,
    side: str,
) -> dict[str, Any]:
    rows = [item for item in positions if item.get("side") == side]
    quantity = sum(float(item.get("quantity") or 0.0) for item in rows)
    notional = sum(
        float(item.get("entry_price") or 0.0) * float(item.get("quantity") or 0.0) for item in rows
    )
    return {
        "model_name": model_name,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "notional": notional,
        "unrealized_pnl": sum(float(item.get("unrealized_pnl") or 0.0) for item in rows),
    }


def _policy(peaks: dict[str, dict[str, Any]]) -> PositionProfitPeakContextPolicy:
    return PositionProfitPeakContextPolicy(
        normalize_symbol=lambda value: str(value or "").split(":")[0],
        aggregate_position_group=_aggregate,
        position_peak_key=lambda model, symbol, side: f"{model}|{symbol}|{side}",
        position_peaks_provider=lambda: peaks,
        default_model_name="ensemble_trader",
    )


def test_position_profit_peak_context_selects_largest_retrace() -> None:
    policy = _policy(
        {
            "ensemble_trader|BTC/USDT|long": {
                "peak_unrealized_pnl": 8.0,
                "peak_pnl_ratio": 0.08,
                "updated_at": "2026-06-10T01:00:00Z",
            },
            "ensemble_trader|BTC/USDT|short": {
                "peak_unrealized_pnl": 5.0,
                "peak_pnl_ratio": 0.05,
                "updated_at": "2026-06-10T01:02:00Z",
            },
        }
    )

    context = policy.context(
        "ensemble_trader",
        "BTC/USDT:USDT",
        [
            {"side": "long", "entry_price": 100.0, "quantity": 1.0, "unrealized_pnl": 3.0},
            {"side": "short", "entry_price": 100.0, "quantity": 1.0, "unrealized_pnl": 4.0},
        ],
    )

    assert context["side"] == "long"
    assert context["symbol"] == "BTC/USDT"
    assert context["peak_unrealized_pnl"] == 8.0
    assert context["current_unrealized_pnl"] == 3.0
    assert context["profit_retrace_abs"] == 5.0
    assert context["profit_retrace_ratio"] == 0.625
    assert context["peak_pnl_ratio"] == 0.08


def test_position_profit_peak_context_uses_current_profit_as_peak_floor() -> None:
    context = _policy({}).context(
        "ensemble_trader",
        "SOL/USDT",
        [
            {"side": "long", "entry_price": 20.0, "quantity": 2.0, "unrealized_pnl": 1.5},
        ],
    )

    assert context["peak_unrealized_pnl"] == 1.5
    assert context["profit_retrace_abs"] == 0.0
    assert context["rows"] == 1


def test_position_profit_peak_context_returns_empty_without_profit() -> None:
    context = _policy({}).context(
        "ensemble_trader",
        "SOL/USDT",
        [
            {"side": "long", "entry_price": 20.0, "quantity": 2.0, "unrealized_pnl": -1.5},
        ],
    )

    assert context == {}


def test_trading_service_position_profit_peak_context_delegates_to_policy(tmp_path) -> None:
    service = object.__new__(TradingService)
    service.position_profit_peaks = PositionProfitPeakTracker(
        path=tmp_path / "peaks.json",
        symbol_normalizer=service._normalize_position_symbol,
        float_parser=service._safe_float,
    )
    service.position_profit_peaks.peaks = {
        "ensemble_trader|BTC/USDT|long": {
            "peak_unrealized_pnl": 6.0,
            "peak_pnl_ratio": 0.06,
            "updated_at": "2026-06-10T01:00:00Z",
        }
    }

    context = service._position_profit_peak_context(
        "ensemble_trader",
        "BTC/USDT",
        [
            {
                "side": "long",
                "entry_price": 100.0,
                "current_price": 104.0,
                "quantity": 1.0,
                "unrealized_pnl": 2.0,
            }
        ],
    )

    assert context["peak_unrealized_pnl"] == 6.0
    assert context["current_unrealized_pnl"] == 2.0
    assert context["profit_retrace_abs"] == 4.0
