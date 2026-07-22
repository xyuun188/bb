from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from models.learning import StrategyProfileSnapshot
from services.continuous_strategy_routing import (
    CONTINUOUS_STRATEGY_SOURCE,
    ContinuousStrategyRoutingPolicy,
    ContinuousStrategyRoutingStore,
)


def _metrics(
    *,
    average: float,
    lcb: float,
    profit_factor: float,
    drawdown: float,
    tail: float,
    pnl: float,
    count: int = 30,
) -> dict:
    return {
        "sample_count": count,
        "average_net_return_pct": average,
        "return_lcb_pct": lcb,
        "profit_factor": profit_factor,
        "max_drawdown": drawdown,
        "tail_loss_pct": tail,
        "realized_net_pnl_usdt": pnl,
    }


def _candidate(
    profile_id: str,
    *,
    side: str,
    regime: str = "",
    scope: str = "side",
    development: dict | None = None,
    exam: dict | None = None,
    production_eligible: bool = False,
) -> dict:
    development = development or _metrics(
        average=0.4,
        lcb=0.2,
        profit_factor=1.6,
        drawdown=0.3,
        tail=-0.2,
        pnl=12.0,
    )
    exam = exam or _metrics(
        average=0.35,
        lcb=0.15,
        profit_factor=1.5,
        drawdown=0.35,
        tail=-0.25,
        pnl=10.5,
    )
    selector = {"scope": scope, "side": side}
    if regime:
        selector["market_regime"] = regime
    if scope.startswith("symbol"):
        selector["symbol"] = "BTC/USDT"
    return {
        "id": profile_id,
        "version": 1,
        "label": profile_id,
        "source": "trained_model_historical_replay_partition",
        "description": "test strategy",
        "rank": 1,
        "params": {
            "selector": selector,
            "prediction_horizon_minutes": 10,
            "historical_return_distribution": development,
        },
        "promotion": {
            "production_influence_eligible": production_eligible,
        },
        "backtest": {
            "status": "complete",
            "evidence_partition": "strategy_development",
            "metrics": development,
        },
        "shadow_validation": {
            "status": "complete",
            "evidence_partition": "strategy_exam",
            "validation_method": "exact_current_model_on_immutable_shadow_snapshot",
            "metrics": exam,
        },
    }


def _build(
    policy: ContinuousStrategyRoutingPolicy,
    candidates: list[dict],
    regime: str,
    *,
    mode: str = "paper",
) -> dict:
    return policy.build(
        execution_mode=mode,
        market_regime={"regime": regime},
        candidates=candidates,
    )


def test_negative_but_improving_strategy_can_remain_training_primary() -> None:
    negative = _candidate(
        "negative_improving",
        side="short",
        development=_metrics(
            average=-0.5,
            lcb=-0.7,
            profit_factor=0.7,
            drawdown=1.0,
            tail=-1.2,
            pnl=-15.0,
        ),
        exam=_metrics(
            average=-0.2,
            lcb=-0.3,
            profit_factor=0.9,
            drawdown=0.6,
            tail=-0.7,
            pnl=-6.0,
        ),
    )

    report = _build(ContinuousStrategyRoutingPolicy(), [negative], "range_bound")

    assert report["current_route"]["recommended_side"] == "short"
    assert report["current_route"]["primary"]["profile_id"] == "negative_improving"
    assert report["candidate_weights"][0]["effective_weight"] > 0.0
    assert report["order_creation_permission"] is False


def test_training_good_but_future_bad_strategy_cannot_be_primary() -> None:
    unstable = _candidate(
        "overfit",
        side="long",
        development=_metrics(
            average=1.2,
            lcb=0.9,
            profit_factor=4.0,
            drawdown=0.2,
            tail=-0.1,
            pnl=36.0,
        ),
        exam=_metrics(
            average=-1.0,
            lcb=-1.4,
            profit_factor=0.2,
            drawdown=4.0,
            tail=-3.0,
            pnl=-30.0,
        ),
    )

    report = _build(ContinuousStrategyRoutingPolicy(), [unstable], "trend_up")

    row = report["candidate_weights"][0]
    assert row["validated"] is True
    assert row["future_stable"] is False
    assert row["primary_eligible"] is False
    assert report["current_route"]["primary"] is None
    assert report["current_route"]["training_primary"]["profile_id"] == "overfit"


def test_market_regime_switches_primary_strategy() -> None:
    policy = ContinuousStrategyRoutingPolicy()
    candidates = [
        _candidate(
            "trend_up_long",
            side="long",
            regime="trend_up",
            scope="regime_side",
        ),
        _candidate(
            "trend_down_short",
            side="short",
            regime="trend_down",
            scope="regime_side",
        ),
    ]

    upward = _build(policy, candidates, "trend_up")
    downward = _build(policy, candidates, "trend_down")

    assert upward["current_route"]["primary"]["profile_id"] == "trend_up_long"
    assert upward["current_route"]["recommended_side"] == "long"
    assert downward["current_route"]["primary"]["profile_id"] == "trend_down_short"
    assert downward["current_route"]["recommended_side"] == "short"


def test_single_symbol_candidate_is_challenger_only() -> None:
    symbol = _candidate(
        "btc_only",
        side="long",
        scope="symbol_side",
    )
    global_short = _candidate("global_short", side="short")

    report = _build(
        ContinuousStrategyRoutingPolicy(),
        [symbol, global_short],
        "range_bound",
    )

    symbol_row = next(
        row for row in report["candidate_weights"] if row["profile_id"] == "btc_only"
    )
    assert symbol_row["primary_eligible"] is False
    assert symbol_row["route_role"] == "challenger"
    assert report["current_route"]["primary"]["profile_id"] == "global_short"


def test_live_strategy_route_is_unchanged_and_empty() -> None:
    report = _build(
        ContinuousStrategyRoutingPolicy(),
        [_candidate("paper_only", side="long")],
        "trend_up",
        mode="live",
    )

    assert report["applied"] is False
    assert report["live_strategy_unchanged"] is True
    assert report["candidate_weights"] == []
    assert report["current_route"] == {}


@pytest.mark.asyncio
async def test_strategy_routes_are_persisted_without_touching_champion_rows(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'routes.db'}")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(StrategyProfileSnapshot.__table__.create)
    candidates = [_candidate("global_long", side="long")]
    routing = _build(
        ContinuousStrategyRoutingPolicy(),
        candidates,
        "trend_up",
    )

    async with sessions.begin() as session:
        result = await ContinuousStrategyRoutingStore().persist(
            mode="paper",
            candidates=candidates,
            routing=routing,
            session=session,
        )
    async with sessions() as session:
        rows = list(
            (
                await session.execute(select(StrategyProfileSnapshot))
            ).scalars().all()
        )

    assert result["persisted"] is True
    assert len(rows) == 1
    assert rows[0].source == CONTINUOUS_STRATEGY_SOURCE
    assert rows[0].status == "primary"
    assert rows[0].promotion["live_execution_permission"] is False
    await engine.dispose()
