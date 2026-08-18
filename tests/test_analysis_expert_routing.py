from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from config.settings import (
    ENSEMBLE_TRADER_NAME,
    FIXED_AI_MODEL_SLOTS,
    MARKET_ANALYSIS_EXPERT_NAMES,
    POSITION_ANALYSIS_EXPERT_NAMES,
    settings,
)
from data_feed.feature_vector import FeatureVector
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from web_dashboard.api import dashboard


def test_ensemble_routes_position_exit_expert_only_for_position_analysis() -> None:
    assert EnsembleCoordinator._initial_expert_names({}) == MARKET_ANALYSIS_EXPERT_NAMES
    assert "position_expert" not in MARKET_ANALYSIS_EXPERT_NAMES

    assert (
        EnsembleCoordinator._initial_expert_names({"review_positions": True})
        == POSITION_ANALYSIS_EXPERT_NAMES
    )
    assert "position_expert" in POSITION_ANALYSIS_EXPERT_NAMES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("context", "expected_names"),
    [
        ({}, MARKET_ANALYSIS_EXPERT_NAMES),
        ({"review_positions": True}, POSITION_ANALYSIS_EXPERT_NAMES),
    ],
)
async def test_ensemble_applies_analysis_scope_to_registry_calls(
    context: dict,
    expected_names: tuple[str, ...],
) -> None:
    class CapturingRegistry:
        attempted: tuple[str, ...] = ()

        async def decide_all(self, features, expert_context):
            self.attempted = tuple(expert_context.get("_include_model_names") or ())
            expert_context["_attempted_models"] = list(self.attempted)
            expert_context["_model_failures"] = []
            expert_context["_model_timings"] = []
            return {
                name: DecisionOutput(
                    model_name=name,
                    symbol=features.symbol,
                    action=Action.HOLD,
                    confidence=0.5,
                    reasoning="scoped expert result",
                    position_size_pct=0.0,
                    suggested_leverage=1.0,
                    raw_response={},
                    feature_snapshot={},
                )
                for name in self.attempted
            }

    class CrossValidatorStub:
        async def validate_all(self, opinions, timing_context):
            return [], None

    registry = CapturingRegistry()
    coordinator = EnsembleCoordinator(registry)  # type: ignore[arg-type]
    coordinator.cross_validator = CrossValidatorStub()  # type: ignore[assignment]
    coordinator.combine = lambda features, context, opinions, *args: DecisionOutput(  # type: ignore[method-assign]
        model_name=ENSEMBLE_TRADER_NAME,
        symbol=features.symbol,
        action=Action.HOLD,
        confidence=0.5,
        reasoning="combined",
        position_size_pct=0.0,
        suggested_leverage=1.0,
        raw_response={},
        feature_snapshot={},
    )

    _decision, opinions = await coordinator.decide(FeatureVector(symbol="BTC/USDT"), context)

    assert registry.attempted == expected_names
    assert tuple(opinions) == expected_names


def _legacy_raw_payload(analysis_type: str) -> dict:
    expert_slots = [
        slot for slot in FIXED_AI_MODEL_SLOTS if str(slot.get("name")) != "decision_maker"
    ]
    expert_names = [str(slot["name"]) for slot in expert_slots]
    payload = {
        "analysis_type": analysis_type,
        "opinions": [
            {
                "model_name": slot["name"],
                "label": slot["label"],
                "role": slot["role"],
                "action": "hold",
                "confidence": 0.5,
                "weight": slot["weight"],
                "reasoning": f"{slot['name']} result",
            }
            for slot in expert_slots
        ],
        "attempted_experts": expert_names,
        "model_timings": [
            {"name": name, "duration_sec": index + 1.0, "status": "completed"}
            for index, name in enumerate(expert_names)
        ],
        "cross_validations": [
            {
                "expert_pair": ["position_expert", "risk_expert"],
                "validation_status": "completed",
                "consistency": "neutral",
            },
            {
                "expert_pair": ["trend_expert", "risk_expert"],
                "validation_status": "completed",
                "consistency": "aligned",
            },
        ],
    }
    if analysis_type == "market":
        payload["direction_competition"] = {
            "preferred_side": "short",
            "long": {"score": 0.2},
            "short": {"score": 0.4},
            "funding_projection": {
                "evidence_complete": True,
                "long": {"signed_cashflow_pct": -0.1},
                "short": {"signed_cashflow_pct": 0.1},
            },
        }
    else:
        payload["dynamic_exit_policy"] = {
            "settled_funding_fee": 12.5,
            "expected_future_funding_cashflow": 1.25,
            "current_lifecycle_net_pnl": 20.0,
            "projected_hold_net_pnl": 24.0,
            "funding_fee_included": True,
        }
    return payload


@pytest.mark.asyncio
async def test_dashboard_projects_experts_by_analysis_scope_for_legacy_records(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'expert-routing.db').as_posix()}",
    )
    monkeypatch.setattr(settings, "vector_memory_enabled", False)
    await init_db()
    try:
        async with get_session_ctx() as session:
            market = AIDecision(
                model_name=ENSEMBLE_TRADER_NAME,
                symbol="KAITO/USDT",
                action="hold",
                confidence=0.5,
                reasoning="market analysis",
                raw_llm_response=_legacy_raw_payload("market"),
                analysis_type="market",
                is_paper=True,
                created_at=datetime.now(UTC),
            )
            position = AIDecision(
                model_name=ENSEMBLE_TRADER_NAME,
                symbol="BTC/USDT",
                action="hold",
                confidence=0.5,
                reasoning="position analysis",
                raw_llm_response=_legacy_raw_payload("position"),
                analysis_type="position",
                is_paper=True,
                created_at=datetime.now(UTC),
            )
            session.add_all([market, position])
            await session.flush()
            market_id = market.id
            position_id = position.id

        market_response = await dashboard.get_analysis_records(
            decision_id=market_id,
            include_detail=True,
            is_paper=True,
        )
        market_record = market_response["records"][0]
        assert market_record["expected_expert_count"] == 4
        assert market_record["expert_count"] == 4
        assert "position_expert" not in market_record["attempted_experts"]
        assert "position_expert" not in {
            item["expert_name"] for item in market_record["experts"]
        }
        assert "position_expert" not in {
            item["name"] for item in market_record["model_timings"]
        }
        assert market_record["cross_summary"]["total"] == 1
        assert market_record["cross_summary"]["expected"] == 1
        assert market_record["direction_competition"]["funding_projection"][
            "evidence_complete"
        ] is True
        assert market_record["dynamic_exit_policy"] == {}

        position_response = await dashboard.get_analysis_records(
            decision_id=position_id,
            include_detail=True,
            is_paper=True,
        )
        position_record = position_response["records"][0]
        assert position_record["expected_expert_count"] == 5
        assert position_record["expert_count"] == 5
        assert "position_expert" in position_record["attempted_experts"]
        assert "position_expert" in {
            item["expert_name"] for item in position_record["experts"]
        }
        assert "position_expert" in {
            item["name"] for item in position_record["model_timings"]
        }
        assert position_record["cross_summary"]["total"] == 2
        assert position_record["cross_summary"]["expected"] == 2
        assert position_record["dynamic_exit_policy"]["settled_funding_fee"] == 12.5
        assert position_record["direction_competition"] == {}
    finally:
        await close_db()
