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


def _normalize(symbol: object) -> str | None:
    return None if symbol is None else str(symbol).replace("/", "-").upper()


def test_position_count_never_grants_or_blocks_entry() -> None:
    policy = EntryCapacityPolicy(_normalize)
    positions = [
        {
            "model_name": "ensemble_trader",
            "symbol": f"ASSET-{index}/USDT",
            "side": "long",
        }
        for index in range(500)
    ]

    assert policy.reason(
        "ensemble_trader",
        _decision("BTC/USDT"),
        positions,
        policy.empty_staged_counts(),
    ) is None


def test_reservations_only_track_current_round_deduplication() -> None:
    policy = EntryCapacityPolicy(_normalize)
    staged = policy.empty_staged_counts()

    policy.reserve_slot("ensemble_trader", _decision("BTC/USDT"), staged)
    policy.reserve_slot("ensemble_trader", _decision("BTC/USDT"), staged)
    policy.reserve_slot("ensemble_trader", _decision("ETH/USDT"), staged)

    assert staged["model_totals"] == {"ensemble_trader": 2}
    assert staged["side_totals"] == {"long": 3}
    assert staged["symbol_side"] == {
        ("ensemble_trader", "BTC-USDT", "long"): 2,
        ("ensemble_trader", "ETH-USDT", "long"): 1,
    }

    policy.release_slot("ensemble_trader", _decision("BTC/USDT"), staged)
    assert staged["model_totals"] == {"ensemble_trader": 2}
    assert staged["side_totals"] == {"long": 2}
