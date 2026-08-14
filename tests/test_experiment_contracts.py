from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from core.experiment_contracts import (
    ExperimentContractError,
    build_dataset_manifest,
    build_experiment_result,
    build_experiment_spec,
    build_parameter_set,
    build_strategy_identity,
    verify_experiment_result,
    verify_experiment_spec,
)


def _spec() -> dict:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=30)
    strategy = build_strategy_identity(
        strategy_id="test.strategy",
        strategy_version="1.0.0",
        implementation="tests:Strategy",
        git_commit="a" * 40,
        source_sha256="b" * 64,
        model_versions={},
    )
    parameters = build_parameter_set({"threshold": 0.25, "period": 14})
    dataset = build_dataset_manifest(
        source="test_fixture",
        symbols=["BTC/USDT"],
        timeframe="1h",
        started_at=start,
        ended_at=end,
        timezone="UTC",
        row_count=720,
        data_sha256="c" * 64,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        quality={"missing_value_count": 0},
    )
    return build_experiment_spec(
        experiment_type="baseline_backtest",
        strategy=strategy,
        parameters=parameters,
        dataset=dataset,
        execution_assumptions={"commission_rate": 0.001, "slippage_rate": 0.0002},
        portfolio_assumptions={"initial_cash": 10_000},
        validation_windows=[{"role": "baseline", "started_at": start, "ended_at": end}],
        runner={"runner_id": "backtrader", "runner_version": "test.v1"},
        environment={"python": "3.12"},
        random_seed=7,
        authority_contract={
            "orders": "simulated",
            "fills": "simulated",
            "fees": "explicit",
            "settlement": "okx_production_only",
        },
    )


def test_experiment_spec_is_content_addressed_and_deterministic() -> None:
    first = _spec()
    second = _spec()

    assert first == second
    assert first["experiment_id"].startswith("exp_")
    assert len(first["spec_sha256"]) == 64
    verify_experiment_spec(first)

    tampered = copy.deepcopy(first)
    tampered["execution_assumptions"]["commission_rate"] = 0.0
    with pytest.raises(ExperimentContractError, match="SHA-256 mismatch"):
        verify_experiment_spec(tampered)


def test_parameter_set_rejects_secret_material() -> None:
    with pytest.raises(ExperimentContractError, match="sensitive key"):
        build_parameter_set({"rsi_period": 14, "api_key": "must-not-be-stored"})


def test_dataset_requires_timezone_and_chronological_window() -> None:
    with pytest.raises(ExperimentContractError, match="timezone evidence"):
        build_dataset_manifest(
            source="fixture",
            symbols=["BTC/USDT"],
            timeframe="1h",
            started_at=datetime(2026, 1, 1),
            ended_at=datetime(2026, 1, 2, tzinfo=UTC),
            timezone="UTC",
            row_count=24,
            data_sha256=hashlib.sha256(b"fixture").hexdigest(),
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )


def test_experiment_result_is_bound_to_spec() -> None:
    spec = _spec()
    result = build_experiment_result(
        spec,
        status="complete",
        metrics={"fee_adjusted_net_profit": 12.5, "profit_factor": 1.4},
    )

    verify_experiment_result(result, spec)
    tampered = copy.deepcopy(result)
    tampered["metrics"]["fee_adjusted_net_profit"] = 999
    with pytest.raises(ExperimentContractError, match="result SHA-256 mismatch"):
        verify_experiment_result(tampered, spec)
