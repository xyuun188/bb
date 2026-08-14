#!/usr/bin/env python3
"""Audit a historical OHLCV snapshot for lookahead and recursive indicator bias."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.bias_analysis import (
    run_lookahead_analysis,
    run_recursive_warmup_analysis,
    validate_feature_availability,
)
from backtest.data_replay import load_historical_from_db, load_historical_from_okx
from backtest.reproducibility import normalize_ohlcv_dataframe
from core.safe_output import safe_print
from data_feed.technical_indicators import compute_all_indicators
from services.strategy_contract_adapter import timeframe_duration

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "quality" / "backtest_bias_audits"
AUDIT_FEATURES = (
    "rsi_14",
    "rsi_7",
    "macd",
    "macd_signal",
    "macd_diff",
    "stoch_k",
    "ema_12",
    "ema_26",
    "adx_14",
    "bb_width",
    "bb_pct",
    "atr_14",
    "volume_ratio",
    "returns_1",
    "returns_5",
    "returns_20",
    "volatility_20",
    "price_vs_sma20",
    "price_vs_sma50",
)


def build_bias_audit_report(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    checkpoint_count: int = 80,
    warmup_lengths: tuple[int, ...] = (100, 200),
    required_warmup_length: int = 200,
) -> dict[str, Any]:
    data = normalize_ohlcv_dataframe(frame)
    minimum_history = max(50, required_warmup_length)
    if len(data) <= minimum_history + 5:
        raise ValueError(
            f"bias audit needs more than {minimum_history + 5} OHLCV rows; got {len(data)}"
        )
    positions = np.linspace(
        minimum_history,
        len(data) - 1,
        num=min(checkpoint_count, len(data) - minimum_history),
        dtype=int,
    )
    checkpoints = sorted({int(item) for item in positions})
    lookahead = run_lookahead_analysis(
        data,
        compute_all_indicators,
        feature_columns=AUDIT_FEATURES,
        checkpoints=checkpoints,
    )
    recursive = run_recursive_warmup_analysis(
        data,
        compute_all_indicators,
        warmup_lengths=warmup_lengths,
        required_warmup_length=required_warmup_length,
        feature_columns=AUDIT_FEATURES,
        comparison_points=5,
    )
    duration = timeframe_duration(timeframe)
    availability = validate_feature_availability(
        [
            {
                "feature_available_at": timestamp.to_pydatetime() + duration,
                "decision_time": timestamp.to_pydatetime() + duration,
            }
            for timestamp in data.index
        ]
    )
    subreports = {"lookahead": lookahead, "recursive_warmup": recursive, "availability": availability}
    return {
        "audit_version": "bb.backtest-bias-audit.v1",
        "symbol": symbol,
        "timeframe": timeframe,
        "dataset": {
            "rows": len(data),
            "started_at": data.index[0].isoformat(),
            "ended_at": data.index[-1].isoformat(),
            "source": str(data.attrs.get("bb_data_source") or "unknown"),
        },
        "status": "pass" if all(item["status"] == "pass" for item in subreports.values()) else "fail",
        "reports": subreports,
        "checked_at": datetime.now(UTC).isoformat(),
    }


async def _load_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv:
        frame = pd.read_csv(args.csv, parse_dates=["timestamp"])
        frame.attrs["bb_data_source"] = f"csv_snapshot:{Path(args.csv).name}"
        return frame
    if args.source == "okx":
        return await load_historical_from_okx(args.symbol, args.timeframe, args.limit)
    return await load_historical_from_db(args.symbol, args.timeframe, args.limit)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    frame = await _load_frame(args)
    report = build_bias_audit_report(
        frame,
        symbol=args.symbol,
        timeframe=args.timeframe,
        checkpoint_count=args.checkpoint_count,
        warmup_lengths=tuple(args.warmup_lengths),
        required_warmup_length=args.required_warmup_length,
    )
    output_dir = Path(args.output_dir)
    await asyncio.to_thread(output_dir.mkdir, parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"bias-audit-{args.symbol.replace('/', '_')}-{stamp}.json"
    await asyncio.to_thread(
        output_path.write_text,
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["report_path"] = str(output_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--source", choices=("db", "okx"), default="db")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--checkpoint-count", type=int, default=80)
    parser.add_argument("--warmup-lengths", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--required-warmup-length", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv and args.source != "db":
        raise SystemExit("--csv cannot be combined with --source okx")
    safe_print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
