from types import SimpleNamespace

import pytest

from scripts.repair_okx_exit_decision_lineage import _select_settlement_positions


def _position(position_id: int, close_order_ids: str):
    return SimpleNamespace(id=position_id, close_exchange_order_id=close_order_ids)


def test_exact_close_order_slice_wins_over_aggregated_history_row() -> None:
    exact = _position(4891, "3745160862257348608")
    aggregate = _position(
        4892,
        "3737273510193242112,3737646013847674880,3745160862257348608",
    )

    selected = _select_settlement_positions(
        [aggregate, exact],
        "3745160862257348608",
    )

    assert selected == [exact]


def test_multiple_aggregate_rows_without_exact_slice_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="refusing to aggregate duplicate"):
        _select_settlement_positions(
            [
                _position(1, "old-a,target"),
                _position(2, "old-b,target"),
            ],
            "target",
        )
