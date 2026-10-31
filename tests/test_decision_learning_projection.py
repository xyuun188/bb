from copy import deepcopy

import pytest
from sqlalchemy import select

from db.session import get_read_session_ctx
from models.decision import AIDecision
from services.decision_learning_projection import (
    backfill_downside_return_projection,
    backfill_execution_risk_projection,
    restore_downside_return,
    restore_execution_risk_projection,
)


def test_restore_downside_preserves_original_contract_and_zero() -> None:
    source = {
        "normal_paper_trade": {"authorized": True, "contract_fingerprint": "original"},
        "profit_risk_sizing": {"expected_net_return_pct": 0.1},
    }
    before = deepcopy(source)
    cleaned = restore_downside_return(source, 0.0)
    assert cleaned["profit_risk_sizing"]["return_lcb_pct"] == 0.0
    assert cleaned["normal_paper_trade"] == source["normal_paper_trade"]
    assert source == before


@pytest.mark.parametrize("value", [None, True, "missing", float("nan"), float("inf")])
def test_restore_downside_never_fabricates_missing_evidence(value) -> None:
    assert restore_downside_return({"profit_risk_sizing": {}}, value) is None


def test_restore_execution_risk_projection_copies_only_missing_fields() -> None:
    snapshot = {
        "normal_paper_trade": {"selection_reason": "paper_training_entry"},
        "profit_risk_sizing": {
            "portfolio_risk_limit_applied": True,
        },
    }
    raw_sizing = {
        "portfolio_risk_limit_applied": False,
        "current_training_portfolio_stressed_loss_usdt": 0.8,
        "remaining_training_portfolio_risk_budget_usdt": 0.4,
    }

    repaired = restore_execution_risk_projection(snapshot, raw_sizing)

    assert repaired["profit_risk_sizing"] == {
        "portfolio_risk_limit_applied": True,
        "current_training_portfolio_stressed_loss_usdt": 0.8,
        "remaining_training_portfolio_risk_budget_usdt": 0.4,
    }


def test_restore_execution_risk_projection_reads_nested_audit_inputs() -> None:
    repaired = restore_execution_risk_projection(
        {
            "normal_paper_trade": {"selection_reason": "paper_training_entry"},
            "profit_risk_sizing": {},
        },
        {
            "portfolio_risk_limit_applied": True,
            "current_training_portfolio_stressed_loss_usdt": 0.8,
            "audit_inputs": {
                "remaining_training_portfolio_risk_budget_usdt": 0.4,
            },
        },
    )

    assert repaired["profit_risk_sizing"]["remaining_training_portfolio_risk_budget_usdt"] == 0.4


@pytest.mark.asyncio
async def test_backfill_uses_bounded_batches_and_is_idempotent(monkeypatch) -> None:
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import update

    from db.session import get_session_ctx

    now = datetime.now(UTC)
    async with get_session_ctx() as session:
        rows = [
            AIDecision(
                model_name="projection_test",
                symbol="BTC/USDT",
                action="long",
                confidence=0.7,
                reasoning="projection",
                created_at=now,
                raw_llm_response={"profit_risk_sizing": {"return_lcb_pct": -0.4}},
            ),
            AIDecision(
                model_name="projection_test",
                symbol="ETH/USDT",
                action="short",
                confidence=0.7,
                reasoning="projection",
                created_at=now,
                raw_llm_response={"profit_risk_sizing": {}},
            ),
        ]
        session.add_all(rows)
        await session.flush()
        ids = [row.id for row in rows]
    async with get_session_ctx() as session:
        for decision_id in ids:
            await session.execute(
                update(AIDecision)
                .where(AIDecision.id == decision_id)
                .values(
                    decision_learning_snapshot={"profit_risk_sizing": {"expected_net_return_pct": 0.1}},
                    decision_learning_snapshot_version=5,
                )
            )
    since = now - timedelta(seconds=1)
    plan = await backfill_downside_return_projection(since=since, batch_size=1)
    assert plan["status"] == "complete"
    assert plan["repairable"] == 1
    assert plan["original_field_unavailable"] == 1
    assert plan["applied"] == 0
    page = await backfill_downside_return_projection(since=since, batch_size=1, max_rows=1)
    assert page["status"] == "partial"
    remaining = await backfill_downside_return_projection(
        since=since, batch_size=1, before_id=page["last_scanned_id"]
    )
    assert page["scanned"] + remaining["scanned"] == plan["scanned"]
    assert page["repairable"] + remaining["repairable"] == plan["repairable"]
    applied = await backfill_downside_return_projection(since=since, batch_size=1, apply=True)
    assert applied["applied"] == 1
    repeated = await backfill_downside_return_projection(since=since, batch_size=1, apply=True)
    assert repeated["applied"] == 0
    async with get_read_session_ctx() as session:
        row = (await session.execute(select(AIDecision).where(AIDecision.id == ids[0]))).scalar_one()
        assert row.raw_llm_response["profit_risk_sizing"]["return_lcb_pct"] == -0.4
        assert row.decision_learning_snapshot["profit_risk_sizing"]["return_lcb_pct"] == -0.4
        assert row.decision_learning_snapshot_version == 6


@pytest.mark.asyncio
async def test_execution_risk_backfill_preserves_prior_downside_projection() -> None:
    from datetime import UTC, datetime

    from sqlalchemy import update

    from db.session import get_session_ctx

    now = datetime.now(UTC)
    raw_sizing = {
        "current_training_portfolio_stressed_loss_usdt": 0.8,
        "portfolio_risk_limit_applied": True,
        "remaining_training_portfolio_risk_budget_usdt": 0.4,
        "return_lcb_pct": -0.2,
    }
    async with get_session_ctx() as session:
        decision = AIDecision(
            model_name="projection_test",
            symbol="BTC/USDT",
            action="short",
            confidence=0.7,
            reasoning="projection",
            created_at=now,
            raw_llm_response={
                "normal_paper_trade": {"selection_reason": "paper_training_entry"},
                "profit_risk_sizing": raw_sizing,
            },
        )
        session.add(decision)
        await session.flush()
        decision_id = decision.id
    async with get_session_ctx() as session:
        await session.execute(
            update(AIDecision)
            .where(AIDecision.id == decision_id)
            .values(
                decision_learning_snapshot={
                    "normal_paper_trade": {"selection_reason": "paper_training_entry"},
                    "profit_risk_sizing": {},
                },
                decision_learning_snapshot_version=5,
            )
        )

    result = await backfill_execution_risk_projection(
        apply=True,
        since=now,
        batch_size=1,
    )

    assert result["status"] == "complete"
    assert result["applied"] >= 1
    async with get_read_session_ctx() as session:
        row = (
            await session.execute(select(AIDecision).where(AIDecision.id == decision_id))
        ).scalar_one()
        assert row.decision_learning_snapshot["profit_risk_sizing"] == raw_sizing
        assert row.decision_learning_snapshot_version == 7
