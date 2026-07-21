from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from models.learning import StrategyProfileSnapshot
from services.model_strategy_blueprint import build_model_strategy_blueprint
from services.paper_strategy_champion import (
    PaperStrategyChampionService,
    build_trained_model_strategy_candidates,
    compare_paper_strategy_challenger,
)


def _blueprint(
    model_version: str,
    *,
    comparison_reason: str = "strict_fee_after_improvement",
) -> dict:
    return {
        "strategy_id": f"trained_{model_version}",
        "model_version": model_version,
        "training_data_sha256": model_version * 8,
        "execution_scope": "paper_only",
        "eligible_sides": ["long"],
        "paper_execution_eligible": True,
        "live_execution_permission": False,
        "model_quality": {
            "comparison_accepted": True,
            "comparison_reason": comparison_reason,
        },
        "risk_policy": {
            "model_may_change_size_or_leverage": False,
            "model_may_bypass_order_deduplication": False,
        },
    }


def _candidate(
    candidate_id: str,
    *,
    version: int = 1,
    selector: dict | None = None,
    average: float = 0.4,
    lcb: float = 0.2,
    profit_factor: float = 1.5,
    drawdown: float = 0.3,
    rank: int = 1,
) -> dict:
    metrics = {
        "average_net_return_pct": average,
        "return_lcb_pct": lcb,
        "profit_factor": profit_factor,
        "max_drawdown": drawdown,
    }
    return {
        "id": candidate_id,
        "version": version,
        "label": candidate_id,
        "rank": rank,
        "params": {
            "selector": selector or {"scope": "side", "side": "long"},
            "historical_return_distribution": metrics,
            "policy_provenance": {
                "evidence_mode": "exact_trained_model_historical_replay",
            },
        },
        "promotion": {"production_influence_eligible": True},
        "backtest": {
            "status": "complete",
            "metrics": metrics,
            "evidence_partition": "strategy_development",
        },
        "shadow_validation": {
            "status": "complete",
            "metrics": metrics,
            "evidence_partition": "strategy_exam",
            "validation_method": "exact_current_model_on_immutable_shadow_snapshot",
        },
    }


def test_trained_artifact_generates_bounded_paper_only_blueprint() -> None:
    metadata = {
        "artifact_version": "model-v1",
        "training_data_sha256": "a" * 64,
        "training_shadow_sample_count": 24,
        "train_count": 12,
        "test_count": 12,
        "horizons": [10, 30, 60],
        "evaluation_group_policy": "chronological_disjoint_decision_groups",
        "oos_return_evaluation": {
            "long": {
                "avg_return_pct": 0.4,
                "return_lcb_pct": 0.2,
                "profit_factor": 1.5,
                "cvar_10_pct": -0.1,
                "max_drawdown_pct": 0.3,
            }
        },
    }
    readiness = {
        "paper_canary": {
            "authorized": True,
            "execution_scope": "paper_only",
            "eligible_sides": ["long"],
        }
    }
    activation = {
        "activation_stage": "canary",
        "paper_canary_authorized": True,
        "production_influence_authorized": False,
        "champion_comparison": {
            "accepted": True,
            "reason": "challenger_quality_improved",
        },
    }

    blueprint = build_model_strategy_blueprint(
        metadata=metadata,
        readiness=readiness,
        activation=activation,
    )

    assert blueprint["paper_execution_eligible"] is True
    assert blueprint["live_execution_permission"] is False
    assert blueprint["eligible_sides"] == ["long"]
    assert blueprint["training_evidence"]["holdout_sample_count"] == 12
    assert blueprint["exit_policy"]["historical_replay_horizon_minutes"] == 10
    assert blueprint["risk_policy"]["model_may_change_size_or_leverage"] is False
    assert blueprint["risk_policy"]["model_may_bypass_order_deduplication"] is False
    assert "position_size" not in blueprint
    assert "leverage" not in blueprint


def test_challenger_requires_strict_improvement_or_strictly_better_model() -> None:
    champion = build_trained_model_strategy_candidates(
        _blueprint("v1"),
        [_candidate("side_long")],
    )[0]
    unchanged = build_trained_model_strategy_candidates(
        _blueprint("v1"),
        [_candidate("symbol_long", selector={"symbol": "BTC/USDT", "side": "long"})],
    )[0]
    better_model = build_trained_model_strategy_candidates(
        _blueprint("v2"),
        [_candidate("side_long")],
    )[0]

    assert compare_paper_strategy_challenger(unchanged, champion)["accepted"] is False
    report = compare_paper_strategy_challenger(better_model, champion)
    assert report["accepted"] is True
    assert report["model_strict_improvement"] is True


def test_trained_strategy_rejects_legacy_selector_matched_shadow_evidence() -> None:
    candidate = _candidate("side_long")
    candidate["params"]["policy_provenance"] = {
        "evidence_mode": "authoritative_trade_outcomes"
    }
    candidate["shadow_validation"]["validation_method"] = (
        "legacy_selector_matched_shadow"
    )

    assert build_trained_model_strategy_candidates(
        _blueprint("v1"),
        [candidate],
    ) == []


@pytest.mark.asyncio
async def test_paper_champion_promotes_retains_and_rolls_back(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'champion.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(StrategyProfileSnapshot.__table__.create)
    service = PaperStrategyChampionService()

    async with sessions.begin() as session:
        initial = await service.reconcile(
            mode="paper",
            blueprint=_blueprint("v1"),
            candidates=[_candidate("side_long")],
            session=session,
        )
    assert initial["active"] is True
    assert initial["transition"] == "initial_champion_activated"

    async with sessions.begin() as session:
        retained = await service.reconcile(
            mode="paper",
            blueprint=_blueprint("v1"),
            candidates=[
                _candidate("side_long"),
                _candidate(
                    "symbol_long",
                    selector={"symbol": "BTC/USDT", "side": "long"},
                    average=0.3,
                    lcb=0.1,
                    profit_factor=1.3,
                    drawdown=0.4,
                    rank=2,
                ),
            ],
            session=session,
        )
    assert retained["profile_id"] == initial["profile_id"]
    assert retained["transition"] == "champion_retained"

    async with sessions.begin() as session:
        upgraded = await service.reconcile(
            mode="paper",
            blueprint=_blueprint("v2"),
            candidates=[_candidate("side_long")],
            session=session,
        )
    assert upgraded["active"] is True
    assert upgraded["model_version"] == "v2"
    assert upgraded["transition"] == "strictly_better_model_strategy_activated"

    async with sessions.begin() as session:
        rolled_back = await service.reconcile(
            mode="paper",
            blueprint=_blueprint("v2"),
            candidates=[
                _candidate(
                    "side_long",
                    version=2,
                    average=0.2,
                    lcb=0.05,
                    profit_factor=1.1,
                    drawdown=0.6,
                )
            ],
            session=session,
        )
    assert rolled_back["transition"] == "previous_champion_restored"
    assert rolled_back["model_version"] == "v1"
    assert rolled_back["model_rollback_required"] is True

    async with sessions() as session:
        active_rows = list(
            (
                await session.execute(
                    select(StrategyProfileSnapshot).where(
                        StrategyProfileSnapshot.is_active.is_(True)
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(active_rows) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_live_mode_never_activates_trained_strategy() -> None:
    result = await PaperStrategyChampionService().reconcile(
        mode="live",
        blueprint=_blueprint("v1"),
        candidates=[_candidate("side_long")],
    )

    assert result["active"] is False
    assert result["live_execution_permission"] is False
    assert result["reason"] == "live_strategy_activation_forbidden"


@pytest.mark.asyncio
async def test_legacy_active_profile_is_not_treated_as_model_strategy_champion(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(StrategyProfileSnapshot.__table__.create)
    async with sessions.begin() as session:
        session.add(
            StrategyProfileSnapshot(
                execution_mode="paper",
                profile_id="legacy_candidate",
                version=1,
                status="candidate",
                source="llm_structured_candidate",
                is_active=True,
                is_disabled=False,
            )
        )
    async with sessions.begin() as session:
        result = await PaperStrategyChampionService().reconcile(
            mode="paper",
            blueprint=_blueprint("v1"),
            candidates=[],
            session=session,
        )

    assert result["active"] is False
    assert result["reason"] == "no_validated_trained_model_strategy"
    await engine.dispose()
