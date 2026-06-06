from datetime import datetime, timedelta, timezone

from models.decision import AIDecision
from models.learning import ShadowBacktest
from models.trade import Order, Position
from services.profit_attribution import build_profit_attribution, extract_signal_sides


def test_extract_signal_sides_reads_current_quant_tool_keys():
    signals = extract_signal_sides({
        "ml_signal": {
            "predictions": [
                {"best_side": "short", "best_expected_return_pct": 0.4235}
            ],
        },
        "local_ai_tools": {
            "profit_prediction": {
                "available": True,
                "trained": True,
                "best_side": "short",
                "expected_return_pct": 0.3566,
            },
            "time_series_prediction": {
                "available": True,
                "trained": True,
                "best_side": "short",
                "side": "short",
                "direction": "down",
                "expected_return_pct": -0.094,
            },
            "sentiment_analysis": {
                "available": True,
                "trained": True,
                "best_side": "long",
                "side": "long",
                "expected_return_pct": 0.5803,
            },
        },
    })

    assert signals["ml"]["available"] is True
    assert signals["ml"]["side"] == "short"
    assert signals["server_profit"]["available"] is True
    assert signals["server_profit"]["side"] == "short"
    assert signals["server_profit"]["expected_return_pct"] == 0.3566
    assert signals["timeseries"]["available"] is True
    assert signals["timeseries"]["side"] == "short"
    assert signals["timeseries"]["expected_return_pct"] == -0.094
    assert signals["sentiment"]["available"] is True
    assert signals["sentiment"]["side"] == "long"
    assert signals["sentiment"]["expected_return_pct"] == 0.5803


def test_profit_attribution_flags_direction_error_from_shadow_backtest():
    now = datetime(2026, 6, 5, 8, 0, tzinfo=timezone.utc)
    position = Position(
        id=1,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="short",
        quantity=1.0,
        entry_price=100.0,
        current_price=108.0,
        leverage=5.0,
        realized_pnl=-8.0,
        is_open=False,
        created_at=now,
        closed_at=now + timedelta(minutes=20),
    )
    decision = AIDecision(
        id=10,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="short",
        confidence=0.8,
        reasoning="看空。",
        position_size_pct=0.05,
        suggested_leverage=5.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        raw_llm_response={
            "ml_signal": {
                "predictions": [
                    {"best_side": "short", "best_expected_return_pct": 0.5}
                ]
            }
        },
        analysis_type="market",
        is_paper=True,
        was_executed=True,
        executed_at=now,
        created_at=now,
    )
    order = Order(
        id=100,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="sell",
        order_type="market",
        quantity=1.0,
        price=100.0,
        status="filled",
        decision_id=10,
        exchange_order_id="okx-1",
        filled_at=now,
        created_at=now,
    )
    shadow = ShadowBacktest(
        id=200,
        decision_id=10,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        analysis_type="market",
        decision_action="short",
        decision_confidence=0.8,
        entry_price=100.0,
        status="completed",
        due_at=now + timedelta(minutes=10),
        horizon_minutes=10,
        actual_price=108.0,
        long_return_pct=0.08,
        short_return_pct=-0.08,
        best_action="long",
        missed_opportunity=False,
        created_at=now,
    )

    result = build_profit_attribution([position], [order], [decision], [shadow])

    assert result["summary"]["trade_count"] == 1
    assert result["summary"]["total_closed_pnl"] == -8.0
    assert result["records"][0]["bucket"] == "ai_direction_error"
    assert result["records"][0]["main_reason"] == "AI方向判断偏差"
    assert result["records"][0]["shadow"]["best_action"] == "long"
