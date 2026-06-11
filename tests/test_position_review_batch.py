from __future__ import annotations

from typing import Any

from services.position_review_batch import PositionReviewBatchPolicy


def _item(symbol: str) -> tuple[tuple[str, str], list[dict[str, Any]]]:
    return (("ensemble_trader", symbol), [{"symbol": symbol}])


def _policy() -> PositionReviewBatchPolicy:
    return PositionReviewBatchPolicy(
        urgent_exit_checker=lambda scan: bool(scan and scan.get("urgent")),
    )


def test_position_review_batch_prioritizes_urgent_and_deferred_exits() -> None:
    items = [
        _item(symbol) for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "OP/USDT"]
    ]
    fast_scan = {
        ("ensemble_trader", "BTC/USDT"): {
            "urgent": True,
            "exit_score": 96.0,
            "priority_score": 96.0,
        },
        ("ensemble_trader", "ETH/USDT"): {"exit_score": 72.0, "priority_score": 72.0},
        ("ensemble_trader", "SOL/USDT"): {"exit_score": 10.0, "priority_score": 10.0},
        ("ensemble_trader", "BNB/USDT"): {"exit_score": 5.0, "priority_score": 5.0},
        ("ensemble_trader", "OP/USDT"): {"exit_score": 4.0, "priority_score": 4.0},
    }

    result = _policy().select(
        items,
        fast_scan,
        max_groups_override=2,
        defer_count_provider=lambda key: 2 if key[1] == "ETH/USDT" else 0,
        cursor=0,
    )

    assert result.max_groups == 4
    assert result.urgent_exit_count == 2
    assert result.deferred_exit_count == 1
    assert ("ensemble_trader", "BTC/USDT") in result.selected_keys
    assert ("ensemble_trader", "ETH/USDT") in result.selected_keys
    assert len(result.selected_items) == 4
    assert len(result.skipped_items) == 1


def test_position_review_batch_rotates_normal_groups_by_cursor() -> None:
    items = [_item(symbol) for symbol in ["A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"]]
    fast_scan = {item[0]: {"exit_score": 0.0, "priority_score": 0.0} for item in items}

    result = _policy().select(
        items,
        fast_scan,
        max_groups_override=2,
        cursor=1,
    )

    assert [item[0][1] for item in result.selected_items] == ["B/USDT", "C/USDT"]
    assert result.next_cursor == 3
    assert [item[0][1] for item in result.skipped_items] == ["A/USDT", "D/USDT", "E/USDT"]


def test_position_review_batch_keeps_profit_exits_ahead_of_normal_rotation() -> None:
    items = [_item(symbol) for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]]
    fast_scan = {
        ("ensemble_trader", "BTC/USDT"): {
            "exit_score": 75.0,
            "priority_score": 75.0,
            "reason": "profit_retrace:3.00->1.50U/50%",
        },
        ("ensemble_trader", "ETH/USDT"): {"exit_score": 0.0, "priority_score": 0.0},
        ("ensemble_trader", "SOL/USDT"): {"exit_score": 0.0, "priority_score": 0.0},
        ("ensemble_trader", "BNB/USDT"): {"exit_score": 0.0, "priority_score": 0.0},
    }

    result = _policy().select(
        items,
        fast_scan,
        max_groups_override=2,
        cursor=1,
    )

    assert [item[0][1] for item in result.selected_items] == ["BTC/USDT", "ETH/USDT"]
    assert result.profit_exit_count == 1
    assert result.next_cursor == 2
