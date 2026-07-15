from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from services.exchange_exit_decision_lineage import (
    ExitDecisionLineageAmbiguous,
    apply_exit_decision_lineage,
    choose_exit_decision_lineage,
)


def _decision(decision_id: int, raw: dict, *, system_sync: bool = False):
    payload = dict(raw)
    if system_sync:
        payload.update({"system_sync": True, "source": "okx_position_reconcile"})
    return SimpleNamespace(
        id=decision_id,
        raw_llm_response=payload,
        was_executed=True,
        execution_reason=None,
        executed_at=datetime(2026, 7, 15, tzinfo=UTC),
        execution_price=20.0,
        outcome="loss",
        outcome_pnl_pct=-1.0,
    )


def test_exact_order_identity_reuses_dynamic_decision_and_retires_sync_duplicate() -> None:
    dynamic = _decision(
        89216,
        {
            "dynamic_exit_policy": {"eligible": True, "close_fraction": 0.66948936},
            "execution_result": {"exchange_order_id": "okx-etc-close"},
        },
    )
    synthetic = _decision(
        89217,
        {"close_fill": {"order_id": "okx-etc-close"}},
        system_sync=True,
    )
    order = SimpleNamespace(decision_id=89217)
    resolution = choose_exit_decision_lineage(
        [dynamic, synthetic],
        close_order_id="okx-etc-close",
        linked_decision=synthetic,
        linked_order=order,
    )

    result = apply_exit_decision_lineage(
        resolution,
        close_order_id="okx-etc-close",
        close_fill={"order_id": "okx-etc-close", "fee": 0.01},
        reconcile_origin="external_okx_sync",
        exit_price=19.0,
        realized_pnl=-0.5,
        closed_at=datetime(2026, 7, 15, 12, tzinfo=UTC),
        entry_notional=100.0,
    )

    assert result["authoritative_decision_id"] == 89216
    assert result["superseded_decision_ids"] == [89217]
    assert order.decision_id == 89216
    assert dynamic.was_executed is True
    assert dynamic.raw_llm_response["dynamic_exit_policy"]["close_fraction"] == 0.66948936
    assert dynamic.raw_llm_response["execution_result"]["exchange_order_id"] == "okx-etc-close"
    assert synthetic.was_executed is False
    assert synthetic.executed_at is None
    assert synthetic.raw_llm_response["reconciliation_superseded"][
        "authoritative_decision_id"
    ] == 89216

    second_pass = choose_exit_decision_lineage(
        [dynamic, synthetic],
        close_order_id="okx-etc-close",
        linked_decision=dynamic,
        linked_order=order,
    )
    assert second_pass.authoritative is dynamic
    assert second_pass.superseded == ()


def test_multiple_production_decisions_for_one_order_fail_closed() -> None:
    first = _decision(1, {"execution_result": {"exchange_order_id": "same-order"}})
    second = _decision(2, {"execution_result": {"exchange_order_id": "same-order"}})

    with pytest.raises(ExitDecisionLineageAmbiguous):
        choose_exit_decision_lineage(
            [first, second],
            close_order_id="same-order",
        )
