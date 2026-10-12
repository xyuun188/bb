from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("status", ["deferred", "timeout", "error"])
def test_quant_skills_preserve_nonexecution_status_for_every_route(status: str) -> None:
    payload = {
        "available": False,
        "status": status,
        "error": "local AI tools inference queue exhausted the batch budget",
        "request_attempted": status != "deferred",
    }
    skills = TradingAgentSkillBook().position_skills(
        position_entry_pause_reason=None,
        ml_signal=None,
        local_ai_tools={
            name: dict(payload)
            for name in (
                "profit_prediction",
                "time_series_prediction",
                "sentiment_analysis",
                "exit_advice",
            )
        },
        portfolio_profit_protection=None,
    )
    by_name = {skill.name: skill for skill in skills}
    for name in (
        "server_profit_model",
        "time_series_model",
        "sentiment_model",
        "exit_advice_model",
    ):
        skill = by_name[name]
        assert skill.status == status
        assert skill.decision == "neutral"
        assert skill.blocks_entry is False
        assert skill.blocks_exit is False
        assert "已返回" not in skill.reason
        assert skill.data["request_attempted"] is (status != "deferred")
        if status == "deferred":
            assert "未发起请求" in skill.reason
        elif status == "timeout":
            assert "未取得响应" in skill.reason
