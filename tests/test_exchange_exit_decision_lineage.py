from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from services.exchange_exit_decision_lineage import (
    ExitDecisionLineageAmbiguous,
    ExitDecisionLineageResolution,
    _can_aggregate_close_positions,
    apply_exit_decision_lineage,
    choose_exit_decision_lineage,
    decision_exit_exchange_order_ids,
    recover_exit_decision_lineage_from_order_fact,
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


@pytest.mark.parametrize("status", ["open", "partial", "filled"])
@pytest.mark.parametrize("tracking_flag", ["existing_exit_order", "exit_replace_attempted"])
def test_tracking_existing_close_cannot_claim_its_execution(
    status: str, tracking_flag: str,
) -> None:
    observer = _decision(
        826226,
        {
            "execution_result": {
                "exchange_order_id": "okx-protection-close",
                "status": status,
                "quantity": 0.0,
                "exchange_confirmed": False,
                "raw_response": {
                    "exit_tracking": True,
                    tracking_flag: True,
                    "id": "okx-protection-close",
                    "info": {"ordId": "okx-protection-close"},
                },
            },
        },
    )
    observer.was_executed = False

    assert decision_exit_exchange_order_ids(observer) == set()
    resolution = choose_exit_decision_lineage(
        [observer],
        close_order_id="okx-protection-close",
        linked_decision=observer,
        linked_order=SimpleNamespace(decision_id=observer.id),
    )
    assert resolution.authoritative is None


def test_repeated_tracking_preserves_actual_submitter_and_external_protection_owner() -> None:
    observations = [
        _decision(
            decision_id,
            {
                "execution_result": {
                    "exchange_order_id": "okx-close",
                    "raw_response": {"exit_tracking": True, "existing_exit_order": True},
                },
            },
        )
        for decision_id in [826188, 826226]
    ]
    for owner in [
        _decision(1, {"execution_result": {"exchange_order_id": "okx-close"}}),
        _decision(2, {"close_fill": {"order_id": "okx-close"}}, system_sync=True),
    ]:
        resolution = choose_exit_decision_lineage(
            [*observations, owner], close_order_id="okx-close",
        )
        assert resolution.authoritative is owner
        assert resolution.matched_decision_ids == (owner.id,)
        assert resolution.superseded == ()


def test_own_native_close_tracking_still_claims_submitted_order() -> None:
    submitter = _decision(
        1,
        {
            "execution_result": {
                "exchange_order_id": "okx-own-close",
                "raw_response": {"exit_tracking": True, "native_full_close": True},
            },
        },
    )
    assert decision_exit_exchange_order_ids(submitter) == {"okx-own-close"}


def test_explicit_reconciliation_fact_survives_old_tracking_metadata() -> None:
    reconciled = _decision(
        1,
        {
            "close_fill": {"order_id": "okx-reconciled-close"},
            "execution_result": {
                "exchange_order_id": "old-observed-order",
                "raw_response": {"exit_tracking": True, "existing_exit_order": True},
            },
        },
        system_sync=True,
    )
    assert decision_exit_exchange_order_ids(reconciled) == {"okx-reconciled-close"}
    resolution = choose_exit_decision_lineage(
        [reconciled], close_order_id="okx-reconciled-close", linked_decision=reconciled,
    )
    assert resolution.authoritative is reconciled


def test_distinct_close_fragments_can_aggregate_against_authoritative_quantity() -> None:
    positions = [
        SimpleNamespace(
            id=101,
            execution_mode="paper",
            symbol="BTC/USDT",
            side="long",
            okx_pos_id="pos-a",
            entry_exchange_order_id="entry-a",
            entry_price=100.0,
            quantity=2.0,
        ),
        SimpleNamespace(
            id=102,
            execution_mode="paper",
            symbol="BTC/USDT",
            side="long",
            okx_pos_id="pos-b",
            entry_exchange_order_id="entry-b",
            entry_price=101.0,
            quantity=3.0,
        ),
    ]

    assert _can_aggregate_close_positions(positions, target_quantity=5.0) is True


def test_duplicate_close_projection_or_quantity_mismatch_stays_ambiguous() -> None:
    duplicate = SimpleNamespace(
        id=101,
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        okx_pos_id="pos-a",
        entry_exchange_order_id="entry-a",
        entry_price=100.0,
        quantity=2.0,
    )
    same_projection = SimpleNamespace(
        **{**duplicate.__dict__, "id": 102}
    )

    assert _can_aggregate_close_positions([duplicate, same_projection], target_quantity=4.0) is False
    assert _can_aggregate_close_positions([duplicate], target_quantity=3.0) is True
    assert _can_aggregate_close_positions(
        [
            duplicate,
            SimpleNamespace(
                **{**duplicate.__dict__, "id": 103, "entry_exchange_order_id": "entry-b"}
            ),
        ],
        target_quantity=9.0,
    ) is False


@pytest.mark.asyncio
async def test_native_order_fact_relinks_raced_reconciliation_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic = _decision(
        89216,
        {"execution_result": {"exchange_order_id": "okx-raced-close"}},
    )
    synthetic = _decision(
        89217,
        {"close_fill": {"order_id": "okx-raced-close"}},
        system_sync=True,
    )
    dynamic.action = "close_short"
    synthetic.action = "close_short"
    order = SimpleNamespace(
        id=6214,
        decision_id=89217,
        model_name="ensemble_trader",
        symbol="ETC/USDT",
        execution_mode="live",
        side="buy",
        exchange_order_id="okx-raced-close",
        price=19.0,
        filled_at=datetime(2026, 7, 15, 12, tzinfo=UTC),
        okx_raw_fills={},
    )

    class _Rows:
        def scalars(self):
            return self

        def all(self):
            return []

    class _Session:
        async def get(self, _model, decision_id):
            return {89217: synthetic}.get(int(decision_id))

        async def execute(self, _statement):
            return _Rows()

    async def fake_load(*_args, **_kwargs):
        return ExitDecisionLineageResolution(
            authoritative=dynamic,
            linked_order=order,
            superseded=(synthetic,),
            matched_decision_ids=(89216, 89217),
        )

    monkeypatch.setattr(
        "services.exchange_exit_decision_lineage.load_exit_decision_lineage",
        fake_load,
    )
    result = await recover_exit_decision_lineage_from_order_fact(
        _Session(),
        order=order,
        close_fill={"order_id": "okx-raced-close", "avg_price": 19.0, "pnl": -0.5},
        now=datetime(2026, 7, 15, 12, tzinfo=UTC),
    )

    assert result is not None
    assert result["authoritative_decision_id"] == 89216
    assert order.decision_id == 89216
    assert synthetic.was_executed is False
