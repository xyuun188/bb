from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from models.learning import StrategyProfileSnapshot
from services.model_strategy_blueprint import (
    build_model_strategy_blueprint,
    model_strategy_side_authorization,
    paper_strategy_authorization,
    strategy_evidence_contract,
)
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
        **strategy_evidence_contract(),
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
    model_version: str = "v1",
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
                **strategy_evidence_contract(),
                "strategy_id": f"trained_{model_version}",
                "evidence_mode": "exact_trained_model_historical_replay",
            },
        },
        "promotion": {"historical_prior_context_eligible": True},
        "backtest": {
            **strategy_evidence_contract(),
            "status": "complete",
            "metrics": metrics,
            "evidence_partition": "strategy_development",
        },
        "shadow_validation": {
            **strategy_evidence_contract(),
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
        "sample_count": 12,
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
        "live_ml_ready": False,
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


def test_negative_quality_shadow_artifact_can_collect_strict_paper_training_trades() -> None:
    blueprint = build_model_strategy_blueprint(
        metadata={
            "artifact_version": "shadow-v1",
            "trained_at": "2026-07-27T00:00:00+00:00",
            "test_count": 12,
            "oos_return_evaluation": {
                "long": {"return_lcb_pct": -0.2},
                "short": {"return_lcb_pct": -0.1},
            },
        },
        readiness={"paper_canary": {"authorized": False}},
        activation={
            "activation_stage": "shadow",
            "live_ml_ready": False,
            "paper_canary_authorized": False,
        },
    )

    assert blueprint["artifact_stage"] == "shadow"
    assert blueprint["paper_execution_eligible"] is True
    assert blueprint["live_execution_permission"] is False
    assert blueprint["eligible_sides"] == []
    assert blueprint["paper_training_sides"] == ["long", "short"]
    assert blueprint["paper_execution_sides"] == ["long", "short"]
    assert blueprint["model_quality"]["evaluated_sides"] == ["long", "short"]
    assert blueprint["entry_policy"][
        "require_current_fee_after_return_lcb_positive"
    ] is True
    assert blueprint["entry_policy"]["require_actual_trade_calibration"] is False
    assert "trained_model_has_no_positive_fee_after_side" in blueprint[
        "blocking_reasons"
    ]
    assert "trained_model_not_authorized_for_paper_strategy" not in blueprint[
        "blocking_reasons"
    ]
    authorization = model_strategy_side_authorization(
        {
            "model_version": "shadow-v1",
            "strategy_blueprint": blueprint,
        },
        execution_scope="paper",
        side="long",
    )
    assert authorization["eligible"] is True
    assert authorization["authorization_basis"] == "paper_current_signal_training"


def test_authorized_paper_canary_bootstraps_without_production_profitability() -> None:
    blueprint = build_model_strategy_blueprint(
        metadata={
            "artifact_version": "canary-v1",
            "trained_at": "2026-08-23T00:00:00+00:00",
            "test_count": 120,
            "oos_return_evaluation": {
                "long": {
                    "avg_return_pct": -0.04,
                    "return_lcb_pct": -0.12,
                    "profit_factor": 0.88,
                }
            },
        },
        readiness={
            "paper_canary": {
                "authorized": True,
                "state": "ready",
                "execution_scope": "paper_only",
                "eligible_sides": ["long"],
            }
        },
        activation={"activation_stage": "canary"},
    )

    assert blueprint["production_eligible_sides"] == []
    assert blueprint["paper_bootstrap_sides"] == ["long"]
    assert blueprint["paper_execution_eligible"] is True
    assert model_strategy_side_authorization(
        {"strategy_blueprint": blueprint},
        execution_scope="paper",
        side="long",
    )["authorization_basis"] == "paper_bootstrap_canary"


@pytest.mark.parametrize("stage", ["shadow", "canary"])
@pytest.mark.parametrize("compared_sides", [["long"], ["short"]])
def test_canary_replacement_executes_only_the_compared_sides(
    stage: str, compared_sides: list[str]
) -> None:
    blueprint = build_model_strategy_blueprint(
        metadata={
            "artifact_version": "side-scoped-canary",
            "trained_at": "2026-11-10T00:00:00+00:00",
            "test_count": 120,
            "oos_return_evaluation": {
                side: {
                    "avg_return_pct": 0.4,
                    "return_lcb_pct": 0.2,
                    "profit_factor": 1.5,
                }
                for side in ("long", "short")
            },
        },
        readiness={
            "paper_canary": {
                "authorized": True,
                "execution_scope": "paper_only",
                "eligible_sides": compared_sides,
            }
        },
        activation={
            "activation_stage": stage,
            "champion_comparison": {
                "accepted": True,
                "candidate_stage": "canary",
                "comparison_sides": compared_sides,
            },
        },
    )

    assert blueprint["paper_training_sides"] == ["long", "short"]
    assert blueprint["paper_execution_sides"] == compared_sides
    assert blueprint["live_execution_permission"] is False
    for side in ("long", "short"):
        permission = model_strategy_side_authorization(
            {"model_version": "side-scoped-canary", "strategy_blueprint": blueprint},
            execution_scope="paper",
            side=side,
        )
        assert permission["eligible"] is (side in compared_sides)


def test_activation_manifest_paper_canary_restores_runtime_blueprint_side() -> None:
    """Runtime metadata rebuilds must preserve the atomic canary permission."""

    blueprint = build_model_strategy_blueprint(
        metadata={
            "artifact_version": "runtime-canary-v1",
            "trained_at": "2026-08-23T00:00:00+00:00",
            "test_count": 120,
            "oos_return_evaluation": {
                "long": {
                    "avg_return_pct": -0.04,
                    "return_lcb_pct": -0.12,
                    "profit_factor": 0.88,
                }
            },
        },
        # This is the shape produced after readiness is rebuilt at runtime:
        # the persisted activation manifest remains the source of canary truth.
        readiness={"paper_canary": {"authorized": False}},
        activation={
            "activation_stage": "canary",
            "paper_canary_authorized": True,
            "paper_canary_report": {
                "authorized": True,
                "state": "ready",
                "execution_scope": "paper_only",
                "eligible_sides": ["long"],
            },
        },
    )

    assert blueprint["production_eligible_sides"] == []
    assert blueprint["paper_bootstrap_sides"] == ["long"]
    assert blueprint["paper_execution_sides"] == ["long"]
    assert blueprint["paper_execution_eligible"] is True


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
        [_candidate("side_long", model_version="v2")],
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


@pytest.mark.parametrize("field", list(strategy_evidence_contract()))
@pytest.mark.parametrize(
    "location", ["provenance", "backtest", "shadow_validation", "forward_validation"]
)
@pytest.mark.parametrize("value", [None, "legacy-evidence"])
def test_paper_candidate_cannot_borrow_another_evidence_contract(field, location, value) -> None:
    candidate = _candidate("side_long")
    candidate["forward_validation"] = {**strategy_evidence_contract(), "required": True}
    assert build_trained_model_strategy_candidates(_blueprint("v1"), [candidate])
    report = (
        candidate["params"]["policy_provenance"]
        if location == "provenance" else candidate[location]
    )
    if value is None:
        report.pop(field)
    else:
        report[field] = value
    assert build_trained_model_strategy_candidates(_blueprint("v1"), [candidate]) == []


def test_paper_candidate_cannot_borrow_another_strategy_identity() -> None:
    candidate = _candidate("side_long")
    candidate["params"]["policy_provenance"]["strategy_id"] = "another-strategy"
    assert build_trained_model_strategy_candidates(_blueprint("v1"), [candidate]) == []


def test_blueprint_identity_changes_when_scoring_changes(monkeypatch) -> None:
    from services import model_strategy_blueprint as module

    kwargs = {
        "metadata": {"artifact_version": "fixed-artifact", "test_count": 12},
        "readiness": {}, "activation": {"activation_stage": "shadow"},
    }
    before = build_model_strategy_blueprint(**kwargs)
    monkeypatch.setattr(module, "PREDICTION_SCORE_POLICY_VERSION", "another-score-policy")
    after = build_model_strategy_blueprint(**kwargs)
    assert before["model_version"] == after["model_version"]
    assert before["strategy_id"] != after["strategy_id"]


@pytest.mark.parametrize("location", ["champion", "signal", "blueprint"])
@pytest.mark.parametrize("field", [*strategy_evidence_contract(), "strategy_id"])
def test_paper_authority_rejects_same_artifact_with_stale_strategy_evidence(location, field) -> None:
    blueprint = _blueprint("v1")
    signal = {
        **strategy_evidence_contract(), "model_version": "v1", "artifact_lifecycle": "canary",
        "strategy_blueprint": blueprint,
        "prediction_quality": {"contract_complete": True, "paper_eligible": True},
    }
    champion = {
        **strategy_evidence_contract(), "active": True, "paper_execution_permission": True,
        "live_execution_permission": False, "model_version": "v1",
        "model_strategy_id": blueprint["strategy_id"],
    }
    context = {"execution_mode": "paper", "paper_strategy_champion": champion}
    assert paper_strategy_authorization(context, signal, symbol="BTC/USDT", side="long")[
        "eligible"
    ] is True
    if field == "strategy_id" and location == "signal":
        signal["strategy_blueprint"] = {**blueprint, "strategy_id": "another-strategy"}
    else:
        target = {"champion": champion, "signal": signal, "blueprint": blueprint}[location]
        target["model_strategy_id" if location == "champion" and field == "strategy_id" else field] = (
            "legacy-evidence"
        )
    result = paper_strategy_authorization(context, signal, symbol="BTC/USDT", side="long")
    assert result["eligible"] is False
    assert result["reason"] == "paper_strategy_evidence_contract_mismatch"


def test_persisted_champion_from_old_score_policy_loses_permission() -> None:
    from services.paper_strategy_champion import _row_payload

    blueprint = _blueprint("v1")
    blueprint.pop("prediction_score_policy_version")
    row = StrategyProfileSnapshot(
        profile_id="legacy", version=1, status="champion", source="trained_model_strategy_blueprint",
        is_active=True, is_disabled=False, params={"blueprint": blueprint},
        promotion={"paper_execution_permission": True}, backtest_metrics={}, probe_state={},
    )
    payload = _row_payload(row)
    assert payload["active"] is False
    assert payload["paper_execution_permission"] is False


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
            candidates=[_candidate("side_long", model_version="v2")],
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
                    model_version="v2",
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
async def test_model_version_change_without_validated_strategy_retires_stale_champion(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'stale.db'}")
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

    # A new artifact with no accepted strategy evidence must not leave the old
    # version's paper permission active. The old champion cannot match the
    # current model version and would otherwise block every local ML signal.
    async with sessions.begin() as session:
        retired = await service.reconcile(
            mode="paper",
            blueprint=_blueprint("v2"),
            candidates=[],
            session=session,
        )
    assert retired["active"] is False
    assert retired["transition"] == "stale_model_champion_retired"
    assert retired["retired_model_version"] == "v1"
    assert retired["reason"] == (
        "model_version_changed_without_validated_strategy_improvement"
    )

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
        previous = (
            await session.execute(
                select(StrategyProfileSnapshot).where(
                    StrategyProfileSnapshot.profile_id == initial["profile_id"]
                )
            )
        ).scalar_one()
    assert active_rows == []
    assert previous.status == "rolled_back"
    assert previous.scheduler_reason == (
        "model_version_changed_without_validated_strategy_improvement"
    )
    await engine.dispose()


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
