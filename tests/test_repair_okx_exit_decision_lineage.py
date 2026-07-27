from types import SimpleNamespace

import pytest

from scripts.repair_okx_exit_decision_lineage import (
    _audit_state,
    _fingerprint,
    _select_settlement_positions,
)


def _position(
    position_id: int,
    close_order_ids: str,
    *,
    settlement_status: str = "settled",
    settlement_raw: dict | None = None,
):
    return SimpleNamespace(
        id=position_id,
        close_exchange_order_id=close_order_ids,
        settlement_status=settlement_status,
        settlement_raw=settlement_raw or {},
    )


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


def test_multiple_active_exact_slices_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="must be deduplicated"):
        _select_settlement_positions(
            [_position(1, "target"), _position(2, "target")],
            "target",
        )


def test_superseded_exact_slice_is_excluded() -> None:
    canonical = _position(1, "target")
    superseded = _position(
        2,
        "target",
        settlement_status="superseded_position_residual",
    )

    assert _select_settlement_positions([superseded, canonical], "target") == [canonical]


def test_audit_state_bounds_raw_decision_output_without_weakening_fingerprint() -> None:
    state = {
        "close_order_id": "3780488864864108544",
        "decisions": [
            {
                "id": 132425,
                "was_executed": True,
                "raw_llm_response": {
                    "analysis": "x" * 50_000,
                    "execution_result": {
                        "exchange_order_id": "3780488864864108544"
                    },
                },
            }
        ],
        "positions": [],
    }

    audit = _audit_state(state)

    decision = audit["decisions"][0]
    assert "raw_llm_response" not in decision
    assert decision["raw_llm_response_bytes"] > 50_000
    assert decision["exit_exchange_order_ids"] == ["3780488864864108544"]
    assert _fingerprint(state) != _fingerprint(
        {
            **state,
            "decisions": [
                {
                    **state["decisions"][0],
                    "raw_llm_response": {"analysis": "changed"},
                }
            ],
        }
    )
