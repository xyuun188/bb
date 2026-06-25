from ai_brain.base_model import Action, DecisionOutput
from risk_manager.engine import RiskEngine


def _decision(symbol: str = "BTC/USDT", action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=action,
        confidence=0.8,
        reasoning="entry",
        position_size_pct=0.03,
        suggested_leverage=3.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.08,
        raw_response={},
    )


def test_risk_engine_blocks_new_symbol_using_capacity_snapshot_when_positions_are_stale() -> None:
    engine = RiskEngine(
        max_open_positions_provider=lambda: {
            "base_limit": 20,
            "effective_limit": 15,
            "entry_limit": 15,
            "open_group_count": 20,
            "reason": "over_capacity_release_first=1",
        }
    )

    result = engine.assess(
        decision=_decision("SOL/USDT"),
        current_positions=[
            {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"},
            {"model_name": "ensemble_trader", "symbol": "ETH/USDT", "side": "short"},
        ],
        account_balance=1000.0,
    )

    assert result.approved is False
    assert "容量快照 20 组" in result.rejection_reason
    assert "本次持仓列表 2 组" in result.rejection_reason
    assert "限制 15 组" in result.rejection_reason


def test_risk_engine_allows_same_symbol_add_when_capacity_snapshot_is_full() -> None:
    engine = RiskEngine(
        max_open_positions_provider=lambda: {
            "entry_limit": 1,
            "effective_limit": 1,
            "open_group_count": 4,
        }
    )

    result = engine.assess(
        decision=_decision("BTC/USDT"),
        current_positions=[
            {
                "model_name": "ensemble_trader",
                "symbol": "BTC/USDT",
                "side": "long",
                "quantity": 1.0,
                "entry_price": 100.0,
            }
        ],
        account_balance=1000.0,
    )

    assert result.approved is True


def test_risk_engine_blocks_same_symbol_opposite_entry_in_net_position_mode() -> None:
    engine = RiskEngine(
        max_open_positions_provider=lambda: {
            "entry_limit": 20,
            "effective_limit": 20,
            "open_group_count": 1,
        }
    )

    result = engine.assess(
        decision=_decision("MASK/USDT", Action.LONG),
        current_positions=[
            {
                "model_name": "ensemble_trader",
                "symbol": "MASK/USDT",
                "side": "short",
                "quantity": 47.0,
                "entry_price": 0.4103,
            }
        ],
        account_balance=1000.0,
    )

    assert result.approved is False
    assert "OKX 净持仓模式" in result.rejection_reason
    assert "先平掉或反转已有 short 仓位" in result.rejection_reason


def test_risk_engine_does_not_treat_closed_same_symbol_as_add() -> None:
    engine = RiskEngine(
        max_open_positions_provider=lambda: {
            "entry_limit": 1,
            "effective_limit": 1,
            "open_group_count": 1,
        }
    )

    result = engine.assess(
        decision=_decision("BTC/USDT"),
        current_positions=[
            {
                "model_name": "ensemble_trader",
                "symbol": "BTC/USDT",
                "side": "long",
                "is_open": False,
                "quantity": 0.0,
            }
        ],
        account_balance=1000.0,
    )

    assert result.approved is False
    assert "容量快照 1 组" in result.rejection_reason
