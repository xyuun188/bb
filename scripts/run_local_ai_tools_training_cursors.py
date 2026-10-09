#!/usr/bin/env python3
"""Read Local AI training cursors in an isolated database process."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


def _limit_cursor_process_memory() -> None:
    """Keep the diagnostic cursor from reclaiming memory needed by trading."""

    if os.name == "nt":
        return
    try:
        import resource

        configured = int(os.environ.get("LOCAL_AI_CURSOR_MEMORY_LIMIT_BYTES", "1610612736"))
        limit = max(configured, 512 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError):
        # The cursor is diagnostic; inability to install a platform limit must
        # not prevent the normal training process from starting.
        return


_limit_cursor_process_memory()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db, get_read_session_ctx  # noqa: E402
from models.learning import ShadowBacktest  # noqa: E402
from scripts.train_local_ai_tools_models import (  # noqa: E402
    _LOCAL_AI_TOOLS_SHADOW_READ_PAGE_SIZE,
    _compact_training_shadow_sample,
    _completed_shadow_sample_count,
    _load_shadow_samples,
    _load_trade_samples,
    _shadow_sample_columns,
    _shadow_sample_from_mapping,
)
from services.local_ai_training_contract import (  # noqa: E402
    LOCAL_AI_TOOLS_TRAINING_TRANSPORT_VERSION,
    TRAINING_CURSOR_VERSION,
    TRAINING_DISTRIBUTION_PROFILE_VERSION,
    local_ai_training_cursor,
    market_training_identity,
)
from services.ml_training_dataset import probe_authoritative_trade_training_cursor  # noqa: E402
from services.training_data_quality import (  # noqa: E402
    annotate_sample,
    annotate_training_payload,
)
from services.training_epoch import load_training_data_start  # noqa: E402

CursorCounter = Callable[[], Awaitable[int]]
SampleLoader = Callable[[], Awaitable[list[dict[str, Any]]]]


def _compact_shadow_sample(mapping: Any) -> dict[str, Any] | None:
    """Build the same bounded shadow row used by the full training loader."""

    return _compact_training_shadow_sample(_shadow_sample_from_mapping(mapping))


async def _iter_shadow_samples_stream():
    """Yield bounded shadow rows page-by-page instead of materializing history."""

    epoch_start = load_training_data_start()
    before_id: int | None = None
    filters = (
        ShadowBacktest.status == "completed",
        ShadowBacktest.created_at >= epoch_start,
        ShadowBacktest.long_return_pct.is_not(None),
        ShadowBacktest.short_return_pct.is_not(None),
    )
    while True:
        async with get_read_session_ctx() as session:
            stmt = (
                select(*_shadow_sample_columns())
                .where(*filters)
                .order_by(ShadowBacktest.id.desc())
                .limit(_LOCAL_AI_TOOLS_SHADOW_READ_PAGE_SIZE)
            )
            if before_id is not None:
                stmt = stmt.where(ShadowBacktest.id < before_id)
            rows = list((await session.execute(stmt)).mappings().all())
        if not rows:
            return
        before_id = int(rows[-1].get("id") or 0) or before_id
        for mapping in rows:
            sample = _compact_shadow_sample(mapping)
            if sample is not None:
                yield sample
        if len(rows) < _LOCAL_AI_TOOLS_SHADOW_READ_PAGE_SIZE:
            return


def _duplicate_metadata(sample: dict[str, Any], seen: dict[tuple[int, int, str], int]) -> dict[str, Any]:
    result = dict(sample)
    decision_id = int(result.get("decision_id") or 0)
    horizon = int(result.get("horizon_minutes") or 0)
    version = str(result.get("label_version") or "")
    if decision_id > 0 and horizon > 0 and version:
        key = (decision_id, horizon, version)
        sample_id = int(result.get("id") or 0)
        if key in seen:
            result["is_duplicate"] = True
            result["duplicate_of"] = seen[key]
            result["duplicate_label_identity"] = {
                "decision_id": decision_id,
                "horizon_minutes": horizon,
                "label_version": version,
            }
        else:
            seen[key] = sample_id or len(seen) + 1
    return result


async def _streaming_cursor_probe() -> dict[str, Any]:
    """Compute the canonical cursor in one bounded-memory database pass."""

    profile_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    market_count = 0
    market_groups: set[str] = set()
    seen: dict[tuple[int, int, str], int] = {}
    async for raw in _iter_shadow_samples_stream():
        sample = annotate_sample(_duplicate_metadata(raw, seen), "shadow")
        identity = market_training_identity(sample)
        if identity is None:
            continue
        market_count += 1
        market_groups.add(str(identity["decision_group"]))
        features = identity.get("features") or {}
        for feature in ("returns_5", "returns_20", "volatility_20", "spread_pct", "orderbook_imbalance"):
            try:
                value = float(features.get(feature))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            stats = profile_stats[feature]
            stats[0] += 1.0
            stats[1] += value
            stats[2] += value * value
        for name, value in (("long_return_pct", identity["long_return_pct"]), ("short_return_pct", identity["short_return_pct"])):
            stats = profile_stats[name]
            numeric = float(value)
            stats[0] += 1.0
            stats[1] += numeric
            stats[2] += numeric * numeric

    # A cursor detects changed settlement facts; only the fitting loader may
    # reconstruct outcomes and report eligible trade/cost sample counts.
    trade_probe = await probe_authoritative_trade_training_cursor()

    profile: dict[str, dict[str, float | int]] = {}
    for key, (count, total, square_total) in profile_stats.items():
        if count <= 0:
            continue
        mean = total / count
        variance = max(square_total / count - mean * mean, 0.0)
        profile[key] = {"count": int(count), "mean": mean, "std": math.sqrt(variance)}
    return {
        "version": TRAINING_CURSOR_VERSION,
        "training_transport_version": LOCAL_AI_TOOLS_TRAINING_TRANSPORT_VERSION,
        "completed_market_sample_count": market_count,
        "completed_market_decision_group_count": len(market_groups),
        "completed_training_decision_group_count": len(market_groups),
        "authoritative_trade_training_probe": trade_probe,
        "cursor_count_scope": "clean_market_groups_and_settlement_fact_fingerprint",
        "training_distribution_profile": {
            "version": TRAINING_DISTRIBUTION_PROFILE_VERSION,
            "features": profile,
        },
    }


async def run_streaming_once() -> dict[str, Any]:
    try:
        shadow_count, cursor = await asyncio.gather(
            _completed_shadow_sample_count(),
            _streaming_cursor_probe(),
        )
        return {
            "trained": False,
            "reason": "cursor_probe_complete",
            "training_transport_version": LOCAL_AI_TOOLS_TRAINING_TRANSPORT_VERSION,
            "completed_shadow_sample_count": int(shadow_count),
            **cursor,
            "training_process_isolated": True,
            "cursor_policy": "canonical_clean_independent_decision_group_view",
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=500),
            "training_process_isolated": True,
        }
    finally:
        await close_db()


async def run_once(
    *,
    shadow_counter: CursorCounter = _completed_shadow_sample_count,
    shadow_loader: SampleLoader = _load_shadow_samples,
    trade_loader: SampleLoader = _load_trade_samples,
) -> dict[str, Any]:
    try:
        shadow_count, shadow_samples, trade_samples = await asyncio.gather(
            shadow_counter(),
            shadow_loader(),
            trade_loader(),
        )
        payload = annotate_training_payload(
            shadow_samples=shadow_samples,
            trade_samples=trade_samples,
            sequence_samples=[],
            text_sentiment_samples=[],
        )
        cursor = local_ai_training_cursor(
            shadow_samples=payload["shadow_samples"],
            trade_samples=payload["trade_samples"],
        )
        return {
            "trained": False,
            "reason": "cursor_probe_complete",
            "training_transport_version": LOCAL_AI_TOOLS_TRAINING_TRANSPORT_VERSION,
            "completed_shadow_sample_count": int(shadow_count),
            "completed_trade_sample_count": len(payload["trade_samples"]),
            **cursor,
            "training_process_isolated": True,
            "cursor_policy": "canonical_clean_independent_decision_group_view",
        }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=500),
            "training_process_isolated": True,
        }
    finally:
        await close_db()


async def _main() -> int:
    result = await run_streaming_once()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
