from ai_brain.base_model import Action, DecisionOutput
from services.entry_capacity import EntryCapacityPolicy


def _decision(symbol: str = "BTC/USDT", action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=action,
        confidence=0.8,
        reasoning="entry",
        position_size_pct=0.03,
        raw_response={},
    )


def _normalize(symbol) -> str | None:
    if symbol is None:
        return None
    return str(symbol).replace("/", "-").upper()


def test_entry_capacity_blocks_new_symbol_when_model_limit_reached() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 1)

    reason = policy.reason(
        "ensemble_trader",
        _decision("ETH/USDT"),
        [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}],
        {"symbol_side": {}, "model_totals": {}},
    )

    assert reason is not None
    assert "1" in reason


def test_entry_capacity_allows_same_symbol_add_even_at_model_limit() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 1)

    reason = policy.reason(
        "ensemble_trader",
        _decision("BTC/USDT"),
        [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}],
        {"symbol_side": {}, "model_totals": {}},
    )

    assert reason is None


def test_entry_capacity_counts_staged_entries() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 2)

    reason = policy.reason(
        "ensemble_trader",
        _decision("SOL/USDT"),
        [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}],
        {"symbol_side": {}, "model_totals": {"ensemble_trader": 1}},
    )

    assert reason is not None


def test_entry_capacity_reserves_staged_entry_slot() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 3)
    staged = policy.empty_staged_counts()

    policy.reserve_slot(
        "ensemble_trader",
        _decision("BTC/USDT", Action.LONG),
        staged,
    )

    assert staged["model_totals"] == {"ensemble_trader": 1}
    assert staged["side_totals"] == {"long": 1}
    assert staged["symbol_side"] == {("ensemble_trader", "BTC-USDT", "long"): 1}
