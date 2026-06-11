from __future__ import annotations

import json
from types import SimpleNamespace

from ai_brain.ensemble_coordinator import (
    MAX_NORMAL_ENTRY_SIZE,
    MIN_EXECUTABLE_ENTRY_CONFIDENCE,
    EnsembleCoordinator,
)
from ai_brain.prompts import build_batch_experts_user_prompt, build_decision_maker_user_prompt
from config.settings import settings
from services.trading_service import TradingService


def test_decision_prompts_ignore_daily_target_context() -> None:
    context = {
        "daily_target": {
            "enabled": True,
            "target_usdt": 999.0,
            "today_realized_pnl": 100.0,
        },
        "strategy_mode": {"strategy": "normal_capture"},
        "entry_candidate_evidence": {"expected_net_return_pct": 0.6},
    }

    decision_maker_prompt = build_decision_maker_user_prompt("market", context)
    batch_experts_prompt = build_batch_experts_user_prompt("market", context)

    assert "daily_target" not in decision_maker_prompt
    assert "daily_target" not in batch_experts_prompt
    assert "每日目标" not in decision_maker_prompt
    assert "每日目标" not in batch_experts_prompt


def test_decision_maker_prompt_is_compact_and_current_symbol_scoped() -> None:
    context = {
        "review_positions": True,
        "open_positions": [
            {"symbol": "BTC/USDT", "side": "long", "entry_price": 100, "unrealized_pnl": 1.2},
            {"symbol": "ETH/USDT", "side": "short", "entry_price": 200, "unrealized_pnl": -0.8},
        ],
        "preliminary_decision": {
            "action": "close_long",
            "confidence": 0.72,
            "reasoning": "规则层发现盈利保护触发，需要最终确认是否平仓。",
        },
        "expert_opinions": [
            {
                "model_name": "position_expert",
                "action": "close_long",
                "confidence": 0.73,
                "reasoning": "盈利回撤扩大，动能减弱，建议锁定利润。",
            }
        ],
        "close_evidence": {"should_close": True, "profit_protection": True},
        "position_review_policy": {"result": "close_long"},
        "entry_candidate_evidence": {"api_key": "must-not-leak", "long": {"score": 0.7}},
    }

    prompt = build_decision_maker_user_prompt("symbol=BTC/USDT; price=101", context)
    payload = json.loads(prompt.splitlines()[3])

    assert "STRICT_FINAL_DECISION_JSON_V2" in prompt
    assert len(prompt) < 2600
    assert payload["symbol"] == "BTC/USDT"
    assert payload["open_positions"] == [
        {"symbol": "BTC/USDT", "side": "long", "entry_price": 100, "unrealized_pnl": 1.2}
    ]
    assert payload["close_evidence"] == {"should_close": True, "profit_protection": True}
    assert payload["position_review_policy"] == {"result": "close_long"}
    assert "api_key" not in json.dumps(payload, ensure_ascii=False)


def test_ensemble_uses_entry_execution_gate_not_daily_target_gate() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())

    assert not hasattr(coordinator, "_daily_target_entry_gate")
    gate = coordinator._entry_execution_gate()

    assert gate == {
        "mode": "normal",
        "score_bonus": 0.0,
        "min_confidence": MIN_EXECUTABLE_ENTRY_CONFIDENCE,
        "min_quality_points": 2,
        "allow_probe": True,
        "max_position_size": MAX_NORMAL_ENTRY_SIZE,
        "max_leverage": settings.max_leverage,
    }


def test_trading_service_no_longer_owns_daily_target_private_rules() -> None:
    assert not hasattr(TradingService, "_daily_target_context")
    assert not hasattr(TradingService, "_configured_daily_target_usdt")
    assert not hasattr(TradingService, "_daily_profit_control_pause_reason")
    assert not hasattr(TradingService, "_daily_profit_control_state")
