#!/usr/bin/env python3
"""Run or verify a content-addressed, reproducible BB backtest."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.data_replay import load_historical_from_db, load_historical_from_okx  # noqa: E402
from backtest.engine import AITradingStrategy, BacktestEngine  # noqa: E402
from backtest.reproducibility import (  # noqa: E402
    build_ohlcv_dataset_manifest,
    dataframe_from_snapshot,
    load_experiment_bundle,
    write_experiment_bundle,
)
from core.experiment_contracts import (  # noqa: E402
    build_experiment_result,
    build_experiment_spec,
    build_parameter_set,
    build_strategy_identity,
)
from core.safe_output import safe_print  # noqa: E402

DEFAULT_PARAMETERS = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "macd_threshold": 0,
}
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "experiments"


def _git_commit(explicit: str = "") -> str:
    configured = str(explicit or os.getenv("BB_EXPERIMENT_GIT_COMMIT") or "").strip().lower()
    if configured:
        if not re.fullmatch(r"[0-9a-f]{7,40}", configured):
            raise SystemExit("experiment Git commit must be a 7-40 character hexadecimal SHA")
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "Git metadata is unavailable; pass --git-commit or BB_EXPERIMENT_GIT_COMMIT"
        ) from exc
    return result.stdout.strip().lower()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "backtest/data_replay.py",
        "backtest/engine.py",
        "backtest/reproducibility.py",
        "scripts/run_reproducible_backtest.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(aliased=True),
        "packages": {
            name: _package_version(name)
            for name in ("backtrader", "numpy", "pandas", "sqlalchemy")
        },
    }


def _parse_parameters(value: str) -> dict[str, Any]:
    if not value:
        return dict(DEFAULT_PARAMETERS)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--parameters-json must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("--parameters-json must contain an object")
    return parsed


def _run_engine(frame: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    random_seed = int(spec["random_seed"])
    random.seed(random_seed)
    np.random.seed(random_seed)
    execution = spec["execution_assumptions"]
    portfolio = spec["portfolio_assumptions"]
    engine = BacktestEngine(
        initial_cash=float(portfolio["initial_cash"]),
        commission_rate=float(execution["commission_rate"]),
        slippage_rate=float(execution["slippage_rate"]),
        random_seed=random_seed,
    )
    engine.load_data(frame)
    engine.add_strategy(AITradingStrategy, **spec["parameters"]["values"])
    return engine.run()


def _build_spec(
    frame: pd.DataFrame,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    parameters: dict[str, Any],
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
    random_seed: int,
    git_commit: str = "",
) -> tuple[dict[str, Any], bytes]:
    _normalized, snapshot, dataset = build_ohlcv_dataset_manifest(
        frame,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
    )
    parameter_set = build_parameter_set(parameters)
    end = datetime.fromisoformat(dataset["ended_at"])
    start = datetime.fromisoformat(dataset["started_at"])
    strategy = build_strategy_identity(
        strategy_id="bb.ai_trading_strategy",
        strategy_version="ai-trading-strategy.v1",
        implementation="backtest.engine:AITradingStrategy",
        git_commit=_git_commit(git_commit),
        source_sha256=_source_sha256(),
        model_versions={"decision_mode": "indicator_rules_in_backtest"},
    )
    spec = build_experiment_spec(
        experiment_type="baseline_backtest",
        strategy=strategy,
        parameters=parameter_set,
        dataset=dataset,
        execution_assumptions={
            "commission_rate": float(commission_rate),
            "slippage_rate": float(slippage_rate),
            "funding_rate": 0.0,
            "latency_ms": 0,
            "fill_model": "backtrader_bar_execution",
            "minimum_order_notional": 0.0,
        },
        portfolio_assumptions={
            "initial_cash": float(initial_cash),
            "position_size_fraction": 0.1,
            "max_concurrent_positions": 1,
            "compounding": "broker_equity",
        },
        validation_windows=[
            {
                "role": "baseline",
                "started_at": start,
                "ended_at": end,
            }
        ],
        runner={
            "runner_id": "backtrader",
            "runner_version": BacktestEngine.RUNNER_VERSION,
            "engine_source_sha256": _source_sha256(),
        },
        environment=_environment(),
        random_seed=random_seed,
        authority_contract={
            "orders": "backtest_simulated_orders",
            "fills": "backtest_bar_execution",
            "fees": "explicit_commission_and_slippage_assumptions",
            "settlement": "OKX_authoritative_facts_for_production_only",
        },
    )
    return spec, snapshot


async def _load_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        frame = pd.read_csv(args.csv, parse_dates=["timestamp"])
        frame.attrs["bb_data_source"] = f"csv_snapshot:{Path(args.csv).name}"
        return frame
    if args.source == "okx":
        return await load_historical_from_okx(args.symbol, args.timeframe, args.limit)
    return await load_historical_from_db(args.symbol, args.timeframe, args.limit)


async def _persist_run(spec: dict[str, Any], result: dict[str, Any], artifact_path: Path) -> None:
    from db.session import get_session_ctx, init_db
    from services.experiment_registry import ExperimentRegistry

    await init_db()
    async with get_session_ctx() as session:
        registry = ExperimentRegistry(session)
        await registry.register(spec, artifact_path=str(artifact_path))
        await registry.mark_running(spec["experiment_id"])
        await registry.complete(spec["experiment_id"], result)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.verify_bundle:
        spec, stored_result, snapshot = load_experiment_bundle(Path(args.verify_bundle))
        frame = dataframe_from_snapshot(snapshot)
        metrics = _run_engine(frame, spec)
        expected = stored_result.get("metrics") if isinstance(stored_result, dict) else {}
        matches = metrics == expected
        if not matches:
            raise SystemExit(
                "reproducibility check failed: rerun metrics differ from stored result"
            )
        return {
            "verified": True,
            "experiment_id": spec["experiment_id"],
            "spec_sha256": spec["spec_sha256"],
            "result_sha256": stored_result["result_sha256"],
            "metrics_match": True,
        }

    frame = await _load_frame(args)
    actual_source = str(frame.attrs.get("bb_data_source") or "").strip()
    if not actual_source:
        raise SystemExit("historical data loader did not provide source provenance")
    spec, snapshot = _build_spec(
        frame,
        source=actual_source,
        symbol=args.symbol,
        timeframe=args.timeframe,
        parameters=_parse_parameters(args.parameters_json),
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        random_seed=args.random_seed,
        git_commit=args.git_commit,
    )
    metrics = _run_engine(frame, spec)
    result = build_experiment_result(
        spec,
        status="complete",
        metrics=metrics,
        artifacts={"dataset_snapshot": "dataset.csv", "spec": "spec.json"},
        diagnostics={"source": "reproducible_backtest_cli", "read_only_market_data": True},
    )
    artifact_dir = write_experiment_bundle(
        Path(args.output_root),
        spec=spec,
        result=result,
        dataset_snapshot=snapshot,
    )
    if args.persist:
        await _persist_run(spec, result, artifact_dir)
    return {
        "verified": True,
        "experiment_id": spec["experiment_id"],
        "spec_sha256": spec["spec_sha256"],
        "result_sha256": result["result_sha256"],
        "artifact_dir": str(artifact_dir),
        "persisted": bool(args.persist),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--source", choices=("db", "okx"), default="db")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--parameters-json", default="")
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--verify-bundle", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv and args.source != "db":
        raise SystemExit("--csv cannot be combined with --source okx")
    safe_print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
