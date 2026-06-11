import pytest

from services.position_review_grouping import PositionReviewGroupingPolicy


def test_position_review_grouping_groups_by_model_and_symbol() -> None:
    btc_1 = {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "quantity": 1.0}
    btc_2 = {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "quantity": 2.0}
    eth = {"model_name": "ensemble_trader", "symbol": "ETH/USDT", "quantity": 3.0}
    other_model = {"model_name": "other", "symbol": "BTC/USDT", "quantity": 4.0}

    grouped = PositionReviewGroupingPolicy().group(
        [
            btc_1,
            {"model_name": "", "symbol": "IGNORED/USDT"},
            {"symbol": "NO_MODEL/USDT"},
            btc_2,
            eth,
            other_model,
        ]
    )

    assert grouped == {
        ("ensemble_trader", "BTC/USDT"): [btc_1, btc_2],
        ("ensemble_trader", "ETH/USDT"): [eth],
        ("other", "BTC/USDT"): [other_model],
    }


def test_position_review_grouping_items_preserves_group_order() -> None:
    items = PositionReviewGroupingPolicy().items(
        [
            {"model_name": "m1", "symbol": "A/USDT"},
            {"model_name": "m2", "symbol": "B/USDT"},
            {"model_name": "m1", "symbol": "A/USDT"},
        ]
    )

    assert [key for key, _positions in items] == [("m1", "A/USDT"), ("m2", "B/USDT")]


def test_position_review_grouping_keeps_missing_symbol_as_data_error() -> None:
    with pytest.raises(KeyError):
        PositionReviewGroupingPolicy().group([{"model_name": "ensemble_trader"}])
