from datetime import UTC, datetime, timedelta

from models.decision import AIDecision
from models.learning import ShadowBacktest
from models.trade import Order, Position
from services.entry_signal_extraction import extract_entry_signal_sides
from services.profit_attribution import (
    build_profit_attribution,
    extract_signal_sides,
    match_entry_decisions_for_positions,
)


def test_extract_signal_sides_reads_current_quant_tool_keys():
    signals = extract_signal_sides(
        {
            "ml_signal": {
                "predictions": [{"best_side": "short", "best_expected_return_pct": 0.4235}],
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
        }
    )

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


def test_extract_signal_sides_reads_chinese_side_labels():
    signals = extract_signal_sides(
        {
            "local_ai_tools": {
                "profit_prediction": {
                    "available": True,
                    "side_label": "做空",
                    "expected_return_pct": 0.21,
                },
                "time_series_prediction": {
                    "available": True,
                    "action_label": "做多",
                    "expected_return_pct": 0.11,
                },
            },
        }
    )

    assert signals["server_profit"]["side"] == "short"
    assert signals["timeseries"]["side"] == "long"


def test_extract_signal_sides_reads_wrapped_server_quant_tools_payloads():
    signals = extract_signal_sides(
        {
            "server_quant_tools": {
                "profit_prediction": {
                    "ok": True,
                    "data": {
                        "prediction": {
                            "predicted_side": "long",
                            "expected_long_return_pct": 0.48,
                        }
                    },
                },
                "time_series_prediction": {
                    "status": "ok",
                    "result": {
                        "forecast_direction": "down",
                        "expected_move_pct": -0.22,
                    },
                },
                "sentiment_analysis": {
                    "available": True,
                    "payload": {"label": "bullish", "score": 0.36},
                },
            }
        }
    )

    assert signals["server_profit"]["available"] is True
    assert signals["server_profit"]["side"] == "long"
    assert signals["server_profit"]["expected_return_pct"] == 0.48
    assert signals["timeseries"]["available"] is True
    assert signals["timeseries"]["side"] == "short"
    assert signals["timeseries"]["expected_return_pct"] == -0.22
    assert signals["sentiment"]["available"] is True
    assert signals["sentiment"]["side"] == "long"
    assert signals["sentiment"]["score"] == 0.36


def test_extract_signal_sides_falls_back_to_opportunity_score_fields():
    signals = extract_signal_sides(
        {
            "opportunity_score": {
                "side": "long",
                "expected_return_pct": 0.42,
                "ml_aligned": True,
                "ml_influence_enabled": True,
                "server_profit_best_side": "short",
                "server_profit_expected_return_pct": 0.31,
                "server_profit_loss_probability": 0.28,
                "timeseries_expected_return_pct": 0.18,
                "timeseries_aligned": True,
            }
        }
    )

    assert signals["ml"]["available"] is True
    assert signals["ml"]["side"] == "long"
    assert signals["ml"]["expected_return_pct"] == 0.42
    assert signals["server_profit"]["available"] is True
    assert signals["server_profit"]["side"] == "short"
    assert signals["server_profit"]["expected_return_pct"] == 0.31
    assert signals["timeseries"]["available"] is True
    assert signals["timeseries"]["side"] == "long"
    assert signals["timeseries"]["expected_return_pct"] == 0.18


def test_extract_signal_sides_keeps_observe_only_ml_prediction_available():
    signals = extract_signal_sides(
        {
            "opportunity_score": {
                "side": "long",
                "expected_return_pct": 0.25,
                "ml_aligned": True,
                "ml_influence_enabled": False,
            }
        }
    )

    assert signals["ml"]["available"] is True
    assert signals["ml"]["influence_enabled"] is False
    assert signals["ml"]["side"] == "long"
    assert signals["ml"]["expected_return_pct"] == 0.25


def test_extract_signal_sides_keeps_ml_prediction_visible_when_influence_disabled():
    signals = extract_entry_signal_sides(
        {
            "ml_signal": {
                "predictions": [
                    {
                        "best_side": "short",
                        "best_expected_return_pct": 0.73,
                    }
                ]
            }
        },
        ml_influence_enabled=False,
    )

    assert signals["ml"]["available"] is True
    assert signals["ml"]["influence_enabled"] is False
    assert signals["ml"]["side"] == "short"
    assert signals["ml"]["expected_return_pct"] == 0.73


def test_profit_attribution_flags_direction_error_from_shadow_backtest():
    now = datetime(2026, 6, 5, 8, 0, tzinfo=UTC)
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
            "ml_signal": {"predictions": [{"best_side": "short", "best_expected_return_pct": 0.5}]}
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
    assert result["records"][0]["main_reason"] == "AI 方向判断偏差"
    assert result["records"][0]["side_label"] == "做空"
    assert result["records"][0]["entry_decision"]["action_label"] == "做空"
    assert "影子复盘显示更优方向是做多" in result["records"][0]["notes"]
    assert result["records"][0]["shadow"]["best_action"] == "long"
    assert result["records"][0]["shadow"]["best_action_label"] == "做多"


def test_match_entry_decisions_uses_order_decision_id_when_available():
    now = datetime(2026, 6, 4, 12, 49, 22, tzinfo=UTC)
    position = Position(
        id=1233,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="XPL/USDT",
        side="long",
        quantity=420.0,
        entry_price=0.0841,
        current_price=0.0919,
        leverage=1.0,
        realized_pnl=3.23,
        is_open=False,
        created_at=now + timedelta(seconds=5),
        closed_at=now + timedelta(hours=4),
    )
    entry_decision = AIDecision(
        id=38635,
        model_name="ensemble_trader",
        symbol="XPL/USDT",
        action="long",
        confidence=0.8,
        reasoning="entry",
        position_size_pct=0.05,
        suggested_leverage=2.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        raw_llm_response={"ml_signal": {"predictions": [{"best_side": "short"}]}},
        analysis_type="market",
        is_paper=True,
        was_executed=True,
        executed_at=now + timedelta(seconds=5),
        created_at=now,
    )
    order = Order(
        id=1778,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="XPL/USDT",
        side="buy",
        order_type="market",
        quantity=950.0,
        price=0.0841,
        status="filled",
        decision_id=38635,
        exchange_order_id="okx-xpl-entry",
        filled_at=now + timedelta(seconds=5),
        created_at=now + timedelta(seconds=5),
    )

    matched = match_entry_decisions_for_positions([position], [order], [entry_decision])

    assert [decision.id for decision in matched] == [38635]


def test_profit_attribution_matches_shadow_by_position_time_without_entry_decision():
    now = datetime(2026, 6, 5, 9, 0, tzinfo=UTC)
    position = Position(
        id=2,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ETH/USDT",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        current_price=96.0,
        leverage=3.0,
        realized_pnl=-4.0,
        is_open=False,
        created_at=now,
        closed_at=now + timedelta(minutes=18),
    )
    shadow = ShadowBacktest(
        id=201,
        decision_id=None,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ETH/USDT",
        analysis_type="market",
        decision_action="long",
        decision_confidence=0.7,
        entry_price=100.0,
        status="completed",
        due_at=now + timedelta(minutes=10),
        horizon_minutes=10,
        actual_price=96.0,
        long_return_pct=-0.04,
        short_return_pct=0.04,
        best_action="short",
        missed_opportunity=False,
        created_at=now + timedelta(minutes=2),
    )

    result = build_profit_attribution([position], [], [], [shadow])

    record = result["records"][0]
    assert record["entry_decision"] is None
    assert record["shadow"]["id"] == 201
    assert record["shadow"]["best_action"] == "short"
    assert record["evidence_status"]["ai"]["available"] is False
    assert "未匹配到开仓 AI 决策" in record["evidence_status"]["ai"]["missing_reason"]
    assert record["evidence_status"]["shadow"]["available"] is True
    assert result["summary"]["evidence_coverage"]["shadow"] == 1
    assert result["summary"]["evidence_coverage"]["all_core"] == 0


def test_profit_attribution_matches_decision_and_shadow_across_symbol_formats():
    now = datetime(2026, 6, 5, 10, 0, tzinfo=UTC)
    position = Position(
        id=3,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        current_price=103.0,
        leverage=2.0,
        realized_pnl=3.0,
        is_open=False,
        created_at=now,
        closed_at=now + timedelta(minutes=30),
    )
    decision = AIDecision(
        id=30,
        model_name="ensemble_trader",
        symbol="BTC-USDT-SWAP",
        action="long",
        confidence=0.82,
        reasoning="entry",
        position_size_pct=0.05,
        suggested_leverage=2.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        raw_llm_response={
            "local_ai_tools": {
                "profit_prediction": {
                    "best_side": "long",
                    "expected_return_pct": 0.42,
                }
            }
        },
        analysis_type="market",
        is_paper=True,
        was_executed=True,
        executed_at=now,
        created_at=now,
    )
    order = Order(
        id=300,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC-USDT-SWAP",
        side="buy",
        order_type="market",
        quantity=1.0,
        price=100.0,
        status="filled",
        decision_id=30,
        exchange_order_id="okx-btc-entry",
        filled_at=now,
        created_at=now,
    )
    shadow = ShadowBacktest(
        id=301,
        decision_id=None,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC-USDT-SWAP",
        analysis_type="market",
        decision_action="long",
        decision_confidence=0.82,
        entry_price=100.0,
        status="completed",
        due_at=now + timedelta(minutes=10),
        horizon_minutes=10,
        actual_price=103.0,
        long_return_pct=0.03,
        short_return_pct=-0.03,
        best_action="long",
        missed_opportunity=False,
        created_at=now + timedelta(minutes=2),
    )

    record = build_profit_attribution([position], [order], [decision], [shadow])["records"][0]

    assert record["entry_decision"]["id"] == 30
    assert record["signals"]["server_profit"]["available"] is True
    assert record["signals"]["server_profit"]["side"] == "long"
    assert record["shadow"]["id"] == 301


def test_profit_attribution_keeps_ai_evidence_when_model_payloads_are_missing():
    now = datetime(2026, 6, 5, 11, 0, tzinfo=UTC)
    position = Position(
        id=4,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="SOL/USDT",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        current_price=101.0,
        leverage=2.0,
        realized_pnl=1.0,
        is_open=False,
        created_at=now,
        closed_at=now + timedelta(minutes=20),
    )
    decision = AIDecision(
        id=40,
        model_name="ensemble_trader",
        symbol="SOL/USDT",
        action="long",
        confidence=0.77,
        reasoning="旧版本仅保存了 AI 入场结论，没有保存完整模型工具输出。",
        position_size_pct=0.05,
        suggested_leverage=2.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        raw_llm_response={},
        analysis_type="market",
        is_paper=True,
        was_executed=True,
        executed_at=now,
        created_at=now,
    )
    order = Order(
        id=400,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="SOL/USDT",
        side="buy",
        order_type="market",
        quantity=1.0,
        price=100.0,
        status="filled",
        decision_id=40,
        exchange_order_id="okx-sol-entry",
        filled_at=now,
        created_at=now,
    )

    record = build_profit_attribution([position], [order], [decision], [])["records"][0]

    assert record["entry_decision"]["id"] == 40
    assert record["entry_decision"]["action_label"] == "做多"
    assert record["entry_decision"]["matched_confidence"] == "high"
    assert record["signals"]["ml"]["available"] is False
    assert record["shadow"] is None
    assert record["evidence_status"]["ai"]["available"] is True
    assert record["evidence_status"]["ml"]["available"] is False
    assert "未保存本地 ML 证据" in record["evidence_status"]["ml"]["missing_reason"]
    assert record["evidence_status"]["shadow"]["available"] is False
    assert "未匹配到影子复盘样本" in record["evidence_status"]["shadow"]["missing_reason"]
