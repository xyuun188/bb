#!/usr/bin/env python3
"""Run a resumable offline parameter search against the existing Backtrader engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.data_replay import load_historical_from_db, load_historical_from_okx  # noqa: E402
from backtest.engine import AITradingStrategy, BacktestEngine  # noqa: E402
from backtest.optimization import (  # noqa: E402
    ObjectiveConfig,
    ParameterSpace,
    ParameterSpec,
    build_walk_forward_windows,
    walk_forward_search,
)
from core.safe_output import (  # noqa: E402
    safe_error_text,
    safe_print,
)

DEFAULT_SPACE = {
    "rsi_oversold": {"kind": "int", "minimum": 25, "maximum": 35, "step": 5},
    "rsi_overbought": {"kind": "int", "minimum": 65, "maximum": 75, "step": 5},
    "macd_threshold": {"kind": "float", "minimum": -0.001, "maximum": 0.001, "step": 0.001},
}


def parse_parameter_space(value: str) -> ParameterSpace:
    try:
        raw = json.loads(value) if value else DEFAULT_SPACE
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--space-json must be valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("--space-json must contain a non-empty object")
    dimensions = []
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise SystemExit(f"parameter {name} must contain an object")
        kind = str(spec.get("kind") or "").lower()
        values = tuple(spec.get("values") or ())
        dimensions.append(
            ParameterSpec(
                name=str(name),
                kind=kind,
                minimum=spec.get("minimum"),
                maximum=spec.get("maximum"),
                step=spec.get("step"),
                values=values,
            )
        )
    return ParameterSpace(tuple(dimensions))


def evaluate_backtest(
    values: dict[str, Any],
    frame: pd.DataFrame,
    *,
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
    random_seed: int,
) -> dict[str, Any]:
    try:
        engine = BacktestEngine(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            random_seed=random_seed,
        )
        engine.load_data(frame)
        engine.add_strategy(AITradingStrategy, **dict(values))
        result = engine.run()
    except Exception as exc:
        return {
            "evaluation_status": "failed",
            "evaluation_error": safe_error_text(exc),
            "total_trades": 0,
            "net_profit": None,
            "profit_factor": None,
        }
    return {"evaluation_status": "complete", **result}


async def _load_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        frame = pd.read_csv(args.csv, parse_dates=["timestamp"])
        frame.attrs["bb_data_source"] = f"csv_snapshot:{Path(args.csv).name}"
        return frame
    if args.source == "okx":
        return await load_historical_from_okx(args.symbol, args.timeframe, args.limit)
    return await load_historical_from_db(args.symbol, args.timeframe, args.limit)


def run_search(args: argparse.Namespace, frame: pd.DataFrame) -> dict[str, Any]:
    space = parse_parameter_space(args.space_json)
    windows = build_walk_forward_windows(
        len(frame),
        train_rows=args.train_rows,
        validation_rows=args.validation_rows,
        oos_rows=args.oos_rows,
        step_rows=args.step_rows,
    )
    objective = ObjectiveConfig(
        min_trades=args.min_trades,
        min_profit_factor=args.min_profit_factor,
        max_drawdown_pct=args.max_drawdown_pct,
        max_tail_loss_pct=args.max_tail_loss_pct,
        min_oos_folds=args.min_oos_folds,
    )
    state_path = Path(args.state_path)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None

    def evaluator(values: dict[str, Any], window: pd.DataFrame, _role: str) -> dict[str, Any]:
        return evaluate_backtest(
            values,
            window,
            initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            slippage_rate=args.slippage_rate,
            random_seed=args.random_seed,
        )

    def checkpoint_writer(value: dict[str, Any]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = state_path.with_suffix(f"{state_path.suffix}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(state_path)

    report = walk_forward_search(
        frame,
        parameter_space=space,
        windows=windows,
        evaluator=evaluator,
        random_seed=args.random_seed,
        candidate_limit=args.candidate_limit,
        top_k=args.top_k,
        objective=objective,
        resume_state=state,
        checkpoint_writer=checkpoint_writer,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(output_path)
    report["state_path"] = str(state_path)
    return report


async def run(args: argparse.Namespace) -> dict[str, Any]:
    frame = await _load_frame(args)
    return run_search(args, frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--source", choices=("db", "okx"), default="db")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--space-json", default="")
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--train-rows", type=int, default=200)
    parser.add_argument("--validation-rows", type=int, default=100)
    parser.add_argument("--oos-rows", type=int, default=100)
    parser.add_argument("--step-rows", type=int, default=100)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--min-profit-factor", type=float, default=1.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=25.0)
    parser.add_argument("--max-tail-loss-pct", type=float, default=10.0)
    parser.add_argument("--min-oos-folds", type=int, default=2)
    parser.add_argument("--state-path", type=Path, default=ROOT / "data/optimization/search-state.json")
    parser.add_argument("--output-path", type=Path, default=ROOT / "data/optimization/search-report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv and args.source != "db":
        raise SystemExit("--csv cannot be combined with --source okx")
    safe_print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
