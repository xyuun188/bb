from ai_brain.base_model import Action, DecisionOutput
from services.exit_position_matcher import ExitPositionMatcher


def _decision(action: Action = Action.CLOSE_LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="测试平仓",
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


def _normalize_symbol(value) -> str:
    return str(value or "").replace("/", "-").replace("-SWAP", "")


def test_has_matching_position_requires_model_symbol_side_and_quantity() -> None:
    matcher = ExitPositionMatcher(_normalize_symbol)
    positions = [
        {"model_name": "other", "symbol": "BTC-USDT", "side": "long", "quantity": 1},
        {"model_name": "ensemble_trader", "symbol": "ETH-USDT", "side": "long", "quantity": 1},
        {"model_name": "ensemble_trader", "symbol": "BTC-USDT", "side": "short", "quantity": 1},
        {"model_name": "ensemble_trader", "symbol": "BTC-USDT", "side": "long", "quantity": 0},
        {"model_name": "ensemble_trader", "symbol": "BTC-USDT", "side": "long", "quantity": 2},
    ]

    assert matcher.has_matching_position(positions, "ensemble_trader", _decision()) is True


def test_has_matching_position_returns_true_for_non_exit_decision() -> None:
    matcher = ExitPositionMatcher(_normalize_symbol)

    assert matcher.has_matching_position([], "ensemble_trader", _decision(Action.LONG)) is True


def test_context_matching_allows_missing_model_name_and_contract_size_fields() -> None:
    matcher = ExitPositionMatcher(_normalize_symbol)
    positions = [
        {"symbol": "BTC-USDT-SWAP", "side": "long", "contracts": "3"},
        {"model_name": "other", "symbol": "BTC-USDT-SWAP", "side": "long", "contracts": "5"},
        {"model_name": "ensemble_trader", "symbol": "BTC-USDT", "side": "long", "sz": "-2"},
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC-USDT",
            "side": "long",
            "is_open": False,
            "sz": 9,
        },
    ]

    matches = matcher.matching_positions(
        positions,
        "ensemble_trader",
        _decision(),
        require_model_name=False,
    )

    assert matches == [positions[0], positions[2]]
