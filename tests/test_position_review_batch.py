from __future__ import annotations

from typing import Any

from services.position_review_batch import PositionReviewBatchPolicy


def _item(symbol: str) -> tuple[tuple[str, str], list[dict[str, Any]]]:
    return ("ensemble_trader", symbol), [{"symbol": symbol}]


def _scan(
    fraction: float,
    *,
    pnl: float = 0.0,
    retrace: float = 0.0,
    eligible: bool = True,
) -> dict[str, Any]:
    return {
        "dynamic_exit_eligible": eligible,
        "dynamic_exit_policy": {
            "eligible": eligible,
            "close_fraction": fraction,
            "fee_after_unrealized_pnl_usdt": pnl,
            "profit_retrace_ratio": retrace,
        },
    }


def _policy() -> PositionReviewBatchPolicy:
    return PositionReviewBatchPolicy(
        urgent_exit_checker=lambda scan: bool(
            scan
            and scan.get("dynamic_exit_eligible") is True
            and (scan.get("dynamic_exit_policy") or {}).get("eligible") is True
        )
    )


def test_position_review_batch_prioritizes_governed_dynamic_exits() -> None:
    items = [_item(symbol) for symbol in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]]
    fast_scan = {
        ("ensemble_trader", "BTC/USDT"): _scan(0.8, pnl=-4.0),
        ("ensemble_trader", "ETH/USDT"): _scan(0.4, pnl=2.0, retrace=0.4),
        ("ensemble_trader", "SOL/USDT"): _scan(0.0, eligible=False),
    }

    result = _policy().select(
        items,
        fast_scan,
        max_groups_override=1,
        defer_count_provider=lambda key: 1 if key[1] == "ETH/USDT" else 0,
    )

    assert result.max_groups == 2
    assert result.urgent_exit_count == 2
    assert result.deferred_exit_count == 1
    assert result.loss_watch_count == 1
    assert result.profit_exit_count == 1
    assert [item[0][1] for item in result.selected_items] == ["ETH/USDT", "BTC/USDT"]


def test_position_review_batch_rotates_noneligible_groups_by_cursor() -> None:
    items = [_item(symbol) for symbol in ["A/USDT", "B/USDT", "C/USDT", "D/USDT"]]
    fast_scan = {item[0]: _scan(0.0, eligible=False) for item in items}

    result = _policy().select(items, fast_scan, max_groups_override=2, cursor=1)

    assert [item[0][1] for item in result.selected_items] == ["B/USDT", "C/USDT"]
    assert result.next_cursor == 3
    assert result.priority_selected_count == 0


def test_position_review_batch_ignores_legacy_force_and_score_fields() -> None:
    items = [_item(symbol) for symbol in ["A/USDT", "B/USDT"]]
    fast_scan = {
        items[0][0]: {"force_exit_candidate": True, "exit_score": 999.0},
        items[1][0]: _scan(0.0, eligible=False),
    }

    result = _policy().select(items, fast_scan, max_groups_override=1, cursor=1)

    assert result.urgent_exit_count == 0
    assert result.selected_items == [items[1]]


def test_position_review_batch_counts_only_governed_priority() -> None:
    items = [_item(f"SYM{index}/USDT") for index in range(3)]
    fast_scan = {item[0]: _scan(0.2 + index * 0.1) for index, item in enumerate(items)}

    result = _policy().select(items, fast_scan, max_groups_override=3)

    assert len(result.selected_items) == 3
    assert result.priority_selected_count == 3
    assert result.skipped_items == []


def test_position_review_batch_hard_capacity_caps_urgent_slow_reviews() -> None:
    items = [_item(f"SYM{index}/USDT") for index in range(5)]
    fast_scan = {item[0]: _scan(1.0) for item in items}

    result = _policy().select(
        items,
        fast_scan,
        max_groups_override=5,
        hard_max_groups_override=2,
    )

    assert result.urgent_exit_count == 5
    assert result.max_groups == 2
    assert len(result.selected_items) == 2
    assert len(result.skipped_items) == 3


def test_capped_urgent_reviews_do_not_starve_previously_deferred_groups() -> None:
    items = [_item(f"SYM{index}/USDT") for index in range(6)]
    fast_scan = {item[0]: _scan(1.0) for item in items}
    deferred: dict[tuple[str, str], int] = {}
    reviewed: list[tuple[str, str]] = []

    for _ in range(3):
        result = _policy().select(
            items,
            fast_scan,
            hard_max_groups_override=2,
            defer_count_provider=lambda key: deferred.get(key, 0),
        )
        reviewed.extend(item[0] for item in result.selected_items)
        for key in result.selected_keys:
            deferred[key] = 0
        for key, _positions in result.skipped_items:
            deferred[key] = deferred.get(key, 0) + 1

    assert len(reviewed) == len(set(reviewed)) == 6
    assert set(reviewed) == {item[0] for item in items}


def test_explicit_hard_risk_precedes_deferred_ordinary_exit_reviews() -> None:
    items = [_item(symbol) for symbol in ["A/USDT", "B/USDT"]]
    fast_scan = {
        items[0][0]: _scan(1.0),
        items[1][0]: {**_scan(1.0), "dynamic_exit_hard_risk": True},
    }
    result = _policy().select(
        items,
        fast_scan,
        hard_max_groups_override=1,
        defer_count_provider=lambda key: 10 if key == items[0][0] else 0,
    )

    assert result.selected_items == [items[1]]
