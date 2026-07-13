from __future__ import annotations

from ai_brain.prompts import build_batch_experts_user_prompt, build_decision_maker_user_prompt
from services.trading_service import TradingService


def test_decision_prompts_ignore_daily_target_context() -> None:
    context = {
        "daily_target": {"enabled": True, "target_usdt": 999.0},
        "entry_candidate_evidence": {"expected_net_return_pct": 0.6},
    }
    assert "daily_target" not in build_decision_maker_user_prompt("market", context)
    assert "daily_target" not in build_batch_experts_user_prompt("market", context)


def test_trading_service_has_no_daily_target_or_fixed_ensemble_entry_gate() -> None:
    assert not hasattr(TradingService, "_daily_target_context")
    assert not hasattr(TradingService, "_daily_profit_control_pause_reason")
    assert not hasattr(TradingService, "_daily_profit_control_state")
