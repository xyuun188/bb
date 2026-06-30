#!/usr/bin/env python3
"""Warm Phase 3 market data caches without starting trading or model routing."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402
from core.safe_output import safe_error_text  # noqa: E402
from db.session import get_session_ctx  # noqa: E402
from models.market_data import Kline, Ticker  # noqa: E402
from services.crypto_feature_coverage import CryptoFeatureCoverageService  # noqa: E402
from services.data_service import DataService, KLINE_PERSIST_TIMEFRAME_LIMITS  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

DEFAULT_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")
REPORT_DIR = ROOT / "data" / "phase3_market_data_warmup"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_symbol_list(raw: list[str], *, limit: int) -> list[str]:
    normalized: list[str] = []
    for value in raw:
        if not value:
            continue
        if value.endswith("-SWAP"):
            parts = value.split("-")
            value = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else value
        if "/" not in value:
            value = f"{value}/USDT"
        if value not in normalized:
            normalized.append(value)
    return normalized[: max(limit, 1)]


def _parse_symbols(raw: list[str] | None, *, limit: int) -> list[str]:
    values: list[str] = []
    for item in raw or []:
        values.extend(part.strip() for part in str(item or "").split(","))
    return _normalize_symbol_list(values, limit=limit)


async def _discover_warmup_symbols(service: DataService, *, limit: int) -> list[str]:
    try:
        available = await service.rest_client.get_available_symbols()
    except Exception:
        available = []
    symbols = [
        str(item.get("symbol") or "").strip()
        for item in available
        if isinstance(item, dict) and str(item.get("symbol") or "").strip()
    ]
    if not symbols:
        symbols = [str(symbol or "").strip() for symbol in getattr(settings, "symbols", []) or []]
    if not symbols:
        symbols = list(DEFAULT_SYMBOLS)
    return _normalize_symbol_list(symbols, limit=limit)


async def _load_market_db_coverage(symbols: list[str]) -> dict[str, Any]:
    if not symbols:
        return {
            "available": True,
            "ticker_ready_count": 0,
            "kline_timeframe_ready_counts": {
                timeframe: 0 for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
            },
            "ticker_symbols": [],
            "kline_symbols_by_timeframe": {
                timeframe: [] for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
            },
        }

    async with get_session_ctx() as session:
        ticker_rows = list(
            (
                await session.execute(
                    select(Ticker.symbol, Ticker.last_price).where(Ticker.symbol.in_(symbols))
                )
            ).all()
        )
        kline_rows = list(
            (
                await session.execute(
                    select(
                        Kline.timeframe,
                        Kline.symbol,
                        func.count(Kline.id),
                        func.max(Kline.open_time),
                    )
                    .where(
                        Kline.symbol.in_(symbols),
                        Kline.timeframe.in_(tuple(KLINE_PERSIST_TIMEFRAME_LIMITS)),
                    )
                    .group_by(Kline.timeframe, Kline.symbol)
                )
            ).all()
        )

    ticker_symbols = sorted(
        {
            str(symbol or "")
            for symbol, price in ticker_rows
            if str(symbol or "") and float(price or 0.0) > 0.0
        }
    )
    kline_symbols_by_timeframe: dict[str, list[str]] = {
        timeframe: [] for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
    }
    kline_latest_by_timeframe: dict[str, dict[str, str | None]] = {
        timeframe: {} for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
    }
    for timeframe, symbol, count, latest_at in kline_rows:
        timeframe_key = str(timeframe or "")
        symbol_key = str(symbol or "")
        if (
            timeframe_key in kline_symbols_by_timeframe
            and symbol_key
            and int(count or 0) > 0
        ):
            kline_symbols_by_timeframe[timeframe_key].append(symbol_key)
            kline_latest_by_timeframe[timeframe_key][symbol_key] = (
                latest_at.isoformat() if hasattr(latest_at, "isoformat") else None
            )

    for timeframe in list(kline_symbols_by_timeframe):
        kline_symbols_by_timeframe[timeframe] = sorted(set(kline_symbols_by_timeframe[timeframe]))

    return {
        "available": True,
        "ticker_ready_count": len(ticker_symbols),
        "kline_timeframe_ready_counts": {
            timeframe: len(kline_symbols_by_timeframe.get(timeframe) or [])
            for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
        },
        "ticker_symbols": ticker_symbols,
        "kline_symbols_by_timeframe": kline_symbols_by_timeframe,
        "kline_latest_by_timeframe": kline_latest_by_timeframe,
    }


async def warm_market_data(
    *,
    symbols: list[str] | None = None,
    symbol_limit: int = 12,
    data_service: DataService | None = None,
    feature_service: CryptoFeatureCoverageService | None = None,
    db_coverage_loader: Any | None = None,
) -> dict[str, Any]:
    """Fetch and persist ticker/K-line facts only; never submits orders."""

    service = data_service or DataService()
    feature_service = feature_service or CryptoFeatureCoverageService()
    symbols = _normalize_symbol_list(symbols or [], limit=symbol_limit)
    if not symbols:
        symbols = await _discover_warmup_symbols(service, limit=symbol_limit)
    started_at = datetime.now(UTC)
    symbol_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        row: dict[str, Any] = {
            "symbol": symbol,
            "ticker_ok": False,
            "ticker_error": "",
            "klines": {},
        }
        try:
            ticker = await service._get_ticker_snapshot(symbol)
            row["ticker_ok"] = bool(ticker and float(ticker.get("last_price") or 0.0) > 0.0)
        except Exception as exc:
            row["ticker_error"] = safe_error_text(exc, limit=180)

        for timeframe, limit in KLINE_PERSIST_TIMEFRAME_LIMITS.items():
            try:
                _tf, rows = await service._fetch_and_persist_klines(symbol, timeframe, limit)
                row["klines"][timeframe] = {
                    "ok": bool(rows),
                    "rows": len(rows or []),
                }
            except Exception as exc:
                row["klines"][timeframe] = {
                    "ok": False,
                    "rows": 0,
                    "error": safe_error_text(exc, limit=180),
                }
        symbol_rows.append(row)

    try:
        feature_report = await feature_service.report(hours=24, limit=1000)
    except Exception as exc:
        feature_report = {
            "status": "error",
            "error": safe_error_text(exc, limit=180),
        }

    try:
        await service.rest_client.close()
    except Exception:
        pass

    fetched_timeframe_counts = {
        timeframe: sum(
            1 for row in symbol_rows if row.get("klines", {}).get(timeframe, {}).get("ok")
        )
        for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
    }
    fetched_ticker_count = sum(1 for row in symbol_rows if row.get("ticker_ok"))
    try:
        loader = db_coverage_loader or _load_market_db_coverage
        db_coverage = await loader(symbols)
    except Exception as exc:
        db_coverage = {
            "available": False,
            "error": safe_error_text(exc, limit=180),
            "ticker_ready_count": fetched_ticker_count,
            "kline_timeframe_ready_counts": fetched_timeframe_counts,
            "ticker_symbols": [],
            "kline_symbols_by_timeframe": {},
        }

    db_available = bool(db_coverage.get("available"))
    ticker_count = int(db_coverage.get("ticker_ready_count") or 0) if db_available else fetched_ticker_count
    timeframe_counts = (
        {
            timeframe: int(
                (db_coverage.get("kline_timeframe_ready_counts") or {}).get(timeframe) or 0
            )
            for timeframe in KLINE_PERSIST_TIMEFRAME_LIMITS
        }
        if db_available
        else fetched_timeframe_counts
    )
    status = "ready"
    if ticker_count <= 0 or any(count <= 0 for count in timeframe_counts.values()):
        status = "blocked"
    elif ticker_count < len(symbol_rows) or any(count < len(symbol_rows) for count in timeframe_counts.values()):
        status = "partial"

    return {
        "status": status,
        "audit_only": False,
        "market_data_mutation": True,
        "mutation_scope": ["market_tickers", "market_klines"],
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "changes_positions": False,
        "changes_orders": False,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "ticker_ready_count": ticker_count,
        "kline_timeframe_ready_counts": timeframe_counts,
        "fetched_ticker_ready_count": fetched_ticker_count,
        "fetched_kline_timeframe_ready_counts": fetched_timeframe_counts,
        "db_verification_available": db_available,
        "db_verification_error": db_coverage.get("error") or "",
        "db_ticker_symbols": db_coverage.get("ticker_symbols") or [],
        "db_kline_symbols_by_timeframe": db_coverage.get("kline_symbols_by_timeframe") or {},
        "feature_coverage_status_after_warmup": feature_report.get("status"),
        "feature_missing_after_warmup": feature_report.get("missing_features") or [],
        "feature_stale_after_warmup": feature_report.get("stale_features") or [],
        "symbol_results": symbol_rows,
        "started_at": started_at.isoformat(),
        "checked_at": _now_iso(),
        "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
    }


def _write_report(report: dict[str, Any], *, report_dir: Path = REPORT_DIR) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    report_path = report_dir / f"phase3-market-data-warmup-{stamp}.json"
    report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    report_path.write_text(report_text + "\n", encoding="utf-8")
    latest_path = report_dir / "latest.json"
    latest_path.write_text(report_text + "\n", encoding="utf-8")
    return report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", help="Symbol or comma-separated symbols.")
    parser.add_argument("--symbol-limit", type=int, default=12)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--no-write-report", action="store_true")
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbol_limit = _safe_int(args.symbol_limit, 12)
    symbols = _parse_symbols(args.symbol, limit=symbol_limit)
    report = await warm_market_data(symbols=symbols, symbol_limit=symbol_limit)
    if not args.no_write_report:
        report["report_path"] = str(_write_report(report))
        report["latest_path"] = str(REPORT_DIR / "latest.json")
    print(json.dumps(report, ensure_ascii=False, indent=args.json_indent, sort_keys=True))
    return 0 if report.get("status") in {"ready", "partial"} else 2


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
