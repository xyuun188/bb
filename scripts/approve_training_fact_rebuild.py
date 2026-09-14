#!/usr/bin/env python3
"""Audit preserved facts and approve their use in the active training epoch."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

import db.session as session_module  # noqa: E402
import models  # noqa: F401,E402
from config.settings import settings  # noqa: E402
from db.session import close_db, get_read_session_ctx, init_db  # noqa: E402
from models.market_data import Kline  # noqa: E402
from models.news import NewsArticle, SocialPost  # noqa: E402
from services.authoritative_trade_outcome import (  # noqa: E402
    load_authoritative_trade_outcomes,
)
from services.historical_shadow_rebuild import (  # noqa: E402
    rebuild_historical_shadow_samples,
)
from services.training_data_quality import annotate_samples  # noqa: E402
from services.training_epoch import (  # noqa: E402
    load_training_epoch,
    write_training_data_migration,
)

CONFIRMATION = "APPROVE_PRESERVED_FACT_REBUILD"
QUALITY_REPORT_FILENAME = "training_data_migration_quality.json"
REQUIRED_STOPPED_SERVICE = "bb-paper-trading.service"


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _service_is_stopped() -> bool:
    import subprocess

    result = subprocess.run(
        ["systemctl", "is-active", REQUIRED_STOPPED_SERVICE],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return (result.stdout or result.stderr or "").strip() in {
        "inactive",
        "failed",
        "unknown",
    }


async def _source_counts(since: datetime) -> dict[str, Any]:
    async with get_read_session_ctx() as session:
        kline = (
            await session.execute(
                select(func.count(Kline.id), func.min(Kline.open_time), func.max(Kline.open_time))
                .where(Kline.open_time >= since)
            )
        ).one()
        news_count = int(
            (
                await session.execute(
                    select(func.count(NewsArticle.id)).where(
                        func.coalesce(NewsArticle.published_at, NewsArticle.created_at) >= since
                    )
                )
            ).scalar_one()
            or 0
        )
        social_count = int(
            (
                await session.execute(
                    select(func.count(SocialPost.id)).where(SocialPost.created_at >= since)
                )
            ).scalar_one()
            or 0
        )
    return {
        "market_kline": int(kline[0] or 0),
        "market_kline_start": kline[1],
        "market_kline_end": kline[2],
        "news_article": news_count,
        "social_post": social_count,
    }


async def collect_report(*, since: datetime) -> dict[str, Any]:
    epoch = load_training_epoch()
    if since >= epoch["epoch_started_at"]:
        raise RuntimeError("historical rebuild start must predate the active epoch")
    trades = annotate_samples(
        await load_authoritative_trade_outcomes(
            since=since,
            compact=True,
            include_training_features=True,
        ),
        "trade",
    )
    approved_trades = [row for row in trades if not row.get("exclude_from_training")]
    excluded_reasons = Counter(
        reason
        for row in trades
        if row.get("exclude_from_training")
        for reason in row.get("quality_reasons") or ["unknown"]
    )
    sources = await _source_counts(since)
    approved_counts = {
        "authoritative_trade": len(approved_trades),
        "market_kline": int(sources["market_kline"]),
        "news_article": int(sources["news_article"]),
        "social_post": int(sources["social_post"]),
    }
    identity = {
        "version": "2026-09-14.preserved-training-facts.v1",
        "reset_id": epoch["reset_id"],
        "training_data_started_at": since.isoformat(),
        "approved_sample_counts": approved_counts,
        "market_kline_start": sources["market_kline_start"],
        "market_kline_end": sources["market_kline_end"],
        "authoritative_trade_outcome_ids": sorted(
            str(row.get("outcome_id") or "") for row in approved_trades
        ),
    }
    return {
        **identity,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_fact_fingerprint": _fingerprint(identity),
        "approved_sample_count_total": sum(approved_counts.values()),
        "authoritative_trade_input_count": len(trades),
        "authoritative_trade_excluded_count": len(trades) - len(approved_trades),
        "authoritative_trade_excluded_reasons": dict(excluded_reasons.most_common()),
        "live_routing_enabled": False,
    }


async def run(*, since: datetime, apply: bool, confirm: str) -> dict[str, Any]:
    await init_db()
    report = await collect_report(since=since)
    report["service_stopped"] = await asyncio.to_thread(_service_is_stopped)
    if not apply:
        return {**report, "status": "dry_run"}
    if confirm != CONFIRMATION:
        raise RuntimeError(f"--apply requires --confirm {CONFIRMATION}")
    if not report["service_stopped"]:
        raise RuntimeError(f"{REQUIRED_STOPPED_SERVICE} must be stopped")
    if report["approved_sample_count_total"] <= 0:
        raise RuntimeError("no preserved facts passed the current training contracts")
    shadow_rebuild = await rebuild_historical_shadow_samples(since=since)
    report["historical_shadow_rebuild"] = shadow_rebuild
    report["approved_sample_counts"]["historical_shadow"] = int(
        shadow_rebuild.get("created") or 0
    )
    report["approved_sample_count_total"] = sum(report["approved_sample_counts"].values())
    report["source_fact_fingerprint"] = _fingerprint(
        {
            "version": report["version"],
            "reset_id": report["reset_id"],
            "training_data_started_at": report["training_data_started_at"],
            "approved_sample_counts": report["approved_sample_counts"],
            "market_kline_start": report["market_kline_start"],
            "market_kline_end": report["market_kline_end"],
            "authoritative_trade_outcome_ids": report[
                "authoritative_trade_outcome_ids"
            ],
        }
    )
    quality_path = Path(settings.data_dir) / QUALITY_REPORT_FILENAME
    quality_sha256 = await asyncio.to_thread(_write_json, quality_path, report)
    migration = write_training_data_migration(
        {
            "approved_at": datetime.now(UTC).isoformat(),
            "training_data_started_at": report["training_data_started_at"],
            "source_fact_fingerprint": report["source_fact_fingerprint"],
            "quality_report_path": str(quality_path),
            "quality_report_sha256": quality_sha256,
            "approved_sample_counts": report["approved_sample_counts"],
            "approved_sample_count_total": report["approved_sample_count_total"],
        }
    )
    return {**report, "status": "approved", "migration": migration}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=(datetime.now(UTC) - timedelta(days=180)).isoformat())
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    try:
        result = await run(
            since=_parse_timestamp(args.since),
            apply=bool(args.apply),
            confirm=str(args.confirm or ""),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    finally:
        await close_db()
        session_module._engine = None
        session_module._sessionmaker = None


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
