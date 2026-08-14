from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.experiment_contracts import (
    ExperimentContractError,
    build_dataset_manifest,
    build_experiment_result,
    build_experiment_spec,
    build_parameter_set,
    build_strategy_identity,
)
from services.experiment_registry import ExperimentRegistry


def _spec() -> dict:
    start = datetime(2026, 2, 1, tzinfo=UTC)
    end = start + timedelta(days=7)
    return build_experiment_spec(
        experiment_type="baseline_backtest",
        strategy=build_strategy_identity(
            strategy_id="registry.test",
            strategy_version="1.0.0",
            implementation="tests:RegistryStrategy",
            git_commit="1" * 40,
            source_sha256="2" * 64,
        ),
        parameters=build_parameter_set({"period": 14}),
        dataset=build_dataset_manifest(
            source="fixture",
            symbols=["ETH/USDT"],
            timeframe="1h",
            started_at=start,
            ended_at=end,
            timezone="UTC",
            row_count=168,
            data_sha256="3" * 64,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        ),
        execution_assumptions={"commission_rate": 0.001, "slippage_rate": 0.0},
        portfolio_assumptions={"initial_cash": 10_000},
        validation_windows=[{"role": "baseline", "started_at": start, "ended_at": end}],
        runner={"runner_id": "backtrader", "runner_version": "test.v1"},
        environment={"python": "3.12"},
        random_seed=0,
        authority_contract={
            "orders": "simulated",
            "fills": "simulated",
            "fees": "explicit",
            "settlement": "okx_production_only",
        },
    )


@pytest.mark.asyncio
async def test_experiment_registry_persists_and_protects_immutable_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from db import session as db_session
    from db.session import close_db, get_session_ctx, init_db

    await close_db()
    monkeypatch.setattr(
        db_session.settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'experiments.db').as_posix()}",
    )
    await init_db()
    try:
        spec = _spec()
        result = build_experiment_result(spec, status="complete", metrics={"net_profit": 2.0})
        async with get_session_ctx() as session:
            registry = ExperimentRegistry(session)
            row = await registry.register(spec, artifact_path="data/experiments/test")
            await registry.mark_running(spec["experiment_id"])
            completed = await registry.complete(spec["experiment_id"], result)
            assert completed.status == "complete"
            assert completed.result_sha256 == result["result_sha256"]

            invalidated = await registry.invalidate(
                spec["experiment_id"],
                reason="fixture provenance was intentionally invalidated",
            )
            assert invalidated.status == "invalidated"
            assert invalidated.result_sha256 == result["result_sha256"]

        async with get_session_ctx() as session:
            registry = ExperimentRegistry(session)
            row = await registry.get(spec["experiment_id"])
            assert row is not None
            row.strategy_version = "mutated"
            with pytest.raises(ExperimentContractError, match="immutable experiment fields"):
                await session.flush()
            await session.rollback()
    finally:
        await close_db()
