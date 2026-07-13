from __future__ import annotations

from services.trading_agent_skills import TradingAgentSkillBook


def test_market_skills_reject_ungoverned_wrapped_server_quant_tool_payloads() -> None:
    skills = TradingAgentSkillBook().market_skills(
        new_pair_pause_reason=None,
        ml_signal=None,
        local_ai_tools={
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
                "payload": {
                    "label": "bullish",
                    "score": 0.36,
                },
            },
        },
        market_regime=None,
        strategy_mode=None,
    )

    by_name = {skill.name: skill for skill in skills}

    assert by_name["server_profit_model"].status == "unavailable"
    assert by_name["server_profit_model"].decision == "neutral"
    assert by_name["time_series_model"].status == "unavailable"
    assert by_name["time_series_model"].decision == "neutral"
    assert by_name["sentiment_model"].status == "unavailable"
    assert by_name["sentiment_model"].decision == "neutral"
