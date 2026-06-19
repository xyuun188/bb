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
    assert "真实持仓 1 组" in reason
    assert "执行容量 1 组" in reason


def test_entry_capacity_allows_same_symbol_add_even_at_model_limit() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 1)

    reason = policy.reason(
        "ensemble_trader",
        _decision("BTC/USDT"),
        [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}],
        {"symbol_side": {}, "model_totals": {}},
    )

    assert reason is None


def test_entry_capacity_counts_staged_entries_as_pending_not_real_positions() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 2)

    reason = policy.reason(
        "ensemble_trader",
        _decision("SOL/USDT"),
        [{"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"}],
        {"symbol_side": {}, "model_totals": {"ensemble_trader": 1}},
    )

    assert reason is not None
    assert "真实持仓 1 组" in reason
    assert "本轮待确认开仓 1 组" in reason
    assert "合计占用 2 组" in reason


def test_entry_capacity_ignores_closed_and_zero_quantity_positions() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 2)

    reason = policy.reason(
        "ensemble_trader",
        _decision("SOL/USDT"),
        [
            {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"},
            {
                "model_name": "ensemble_trader",
                "symbol": "ETH/USDT",
                "side": "long",
                "is_open": False,
            },
            {
                "model_name": "ensemble_trader",
                "symbol": "XRP/USDT",
                "side": "long",
                "quantity": 0.0,
            },
        ],
        {"symbol_side": {}, "model_totals": {}},
    )

    assert reason is None


def test_entry_capacity_counts_fragmented_positions_as_one_group() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 2)

    reason = policy.reason(
        "ensemble_trader",
        _decision("OP/USDT"),
        [
            {"model_name": "ensemble_trader", "symbol": "ARB/USDT", "side": "long"},
            {"model_name": "ensemble_trader", "symbol": "ARB/USDT", "side": "long"},
            {"model_name": "ensemble_trader", "symbol": "YGG/USDT", "side": "long"},
        ],
        {"symbol_side": {}, "model_totals": {}},
    )

    assert reason is not None
    assert "真实持仓 2 组" in reason


def test_entry_capacity_reserves_same_symbol_group_once_for_model_total() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 3)
    staged = policy.empty_staged_counts()

    policy.reserve_slot("ensemble_trader", _decision("BTC/USDT", Action.LONG), staged)
    policy.reserve_slot("ensemble_trader", _decision("BTC/USDT", Action.LONG), staged)

    assert staged["model_totals"] == {"ensemble_trader": 1}
    assert staged["side_totals"] == {"long": 2}
    assert staged["symbol_side"] == {("ensemble_trader", "BTC-USDT", "long"): 2}


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


def test_entry_capacity_releases_unconfirmed_staged_entry_slot() -> None:
    policy = EntryCapacityPolicy(_normalize, lambda: 3)
    staged = policy.empty_staged_counts()

    policy.reserve_slot("ensemble_trader", _decision("BTC/USDT", Action.LONG), staged)
    policy.reserve_slot("ensemble_trader", _decision("BTC/USDT", Action.LONG), staged)
    policy.reserve_slot("ensemble_trader", _decision("ETH/USDT", Action.LONG), staged)

    policy.release_slot("ensemble_trader", _decision("BTC/USDT", Action.LONG), staged)
    policy.release_slot("ensemble_trader", _decision("ETH/USDT", Action.LONG), staged)

    assert staged["model_totals"] == {"ensemble_trader": 1}
    assert staged["side_totals"] == {"long": 1}
    assert staged["symbol_side"] == {("ensemble_trader", "BTC-USDT", "long"): 1}


def test_entry_capacity_uses_dynamic_effective_limit_from_context() -> None:
    policy = EntryCapacityPolicy(
        _normalize,
        lambda: {
            "base_limit": 20,
            "effective_limit": 2,
            "reason": "low_quality_pressure=6",
            "factors": {"reason_codes": ["low_quality_pressure"]},
        },
    )

    reason = policy.reason(
        "ensemble_trader",
        _decision("SOL/USDT"),
        [
            {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"},
            {"model_name": "ensemble_trader", "symbol": "ETH/USDT", "side": "long"},
        ],
        {"symbol_side": {}, "model_totals": {}},
    )

    assert reason is not None
    assert "运行上限" in reason
    assert "低质量持仓压力较高" in reason


def test_entry_capacity_prefers_entry_limit_when_rotation_slots_are_open() -> None:
    policy = EntryCapacityPolicy(
        _normalize,
        lambda: {
            "base_limit": 4,
            "effective_limit": 4,
            "entry_limit": 5,
            "reason": "rotation_entry_expansion=1",
            "factors": {"reason_codes": ["rotation_entry_expansion", "release_rotation_slots"]},
        },
    )

    reason = policy.reason(
        "ensemble_trader",
        _decision("SOL/USDT"),
        [
            {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"},
            {"model_name": "ensemble_trader", "symbol": "ETH/USDT", "side": "long"},
            {"model_name": "ensemble_trader", "symbol": "XRP/USDT", "side": "long"},
            {"model_name": "ensemble_trader", "symbol": "DOGE/USDT", "side": "long"},
        ],
        {"symbol_side": {}, "model_totals": {}},
    )

    assert reason is None
