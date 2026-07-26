from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from config.settings import settings
from db.repositories.decision_repo import DecisionRepository
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Order
from web_dashboard.api import dashboard


class _EmptyMappingResult:
    def mappings(self) -> _EmptyMappingResult:
        return self

    def all(self) -> list[Any]:
        return []


class _CapturingSession:
    def __init__(self) -> None:
        self.statement: Any = None

    async def execute(self, statement: Any) -> _EmptyMappingResult:
        self.statement = statement
        return _EmptyMappingResult()


@pytest.mark.asyncio
async def test_decision_summary_query_projects_only_dashboard_fields() -> None:
    session = _CapturingSession()
    repo = DecisionRepository(session)  # type: ignore[arg-type]

    assert await repo.get_recent_decision_summaries(limit=5) == []

    selected = set(session.statement.selected_columns.keys())
    assert "feature_snapshot" not in selected
    assert "raw_llm_response" not in selected
    assert "model_health_timings" not in selected
    assert "decision_learning_snapshot" not in selected
    assert "raw_opportunity_score" in selected
    assert "raw_execution_result" in selected


@pytest.mark.asyncio
async def test_decisions_endpoint_preserves_list_contract_with_large_stored_payload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'decision-summary.db').as_posix()}",
    )
    await init_db()
    opportunity_score = {
        "score": 0.8123,
        "expected_net_return_pct": 0.42,
        "policy_provenance": {"source": "test-contract"},
    }
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="BTC/USDT",
                action="close_long",
                confidence=0.91,
                reasoning="risk reduced",
                position_size_pct=0.1,
                suggested_leverage=2.0,
                feature_snapshot={"unused_training_payload": "x" * 200_000},
                raw_llm_response={
                    "opportunity_score": opportunity_score,
                    "execution_result": {
                        "status": "rejected",
                        "raw_response": {
                            "sCode": "51028",
                            "sMsg": "Contract under delivery.",
                        },
                    },
                    "unused_model_transcript": "y" * 200_000,
                },
                execution_reason="原始说明已损坏，无法准确还原",
                was_executed=False,
                is_paper=True,
                created_at=datetime.now(UTC),
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="sell",
                    order_type="market",
                    quantity=0.25,
                    price=100.0,
                    status="rejected",
                    decision_id=decision.id,
                    okx_raw_fills={"unused_order_payload": "z" * 200_000},
                )
            )

        response = await dashboard.get_decisions(limit=5, is_paper=True)

        assert response["total"] == 1
        assert response["count"] == 1
        item = response["decisions"][0]
        assert item["symbol"] == "BTC/USDT"
        assert {
            key: item["opportunity_score"][key] for key in opportunity_score
        } == opportunity_score
        assert item["opportunity_score"]["selected_for_execution"] is False
        assert item["opportunity_score"]["execution_final_state"] == "skipped"
        assert item["order_quantity"] == 0.25
        assert item["order_status"] == "rejected"
        assert "51028" in item["execution_reason"]
    finally:
        await close_db()
