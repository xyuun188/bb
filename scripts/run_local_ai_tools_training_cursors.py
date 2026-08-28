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
    _compact_local_ai_tools_features,
    _completed_shadow_sample_count,
    _load_shadow_samples,
    _load_trade_samples,
    _shadow_sample_columns,
    _shadow_sample_from_mapping,
    _snapshot,
)
from services.local_ai_training_contract import (  # noqa: E402
    TRAINING_CURSOR_VERSION,
    TRAINING_DISTRIBUTION_PROFILE_VERSION,
    authoritative_cost_training_identity,
    local_ai_training_cursor,
    market_training_identity,
)
from services.training_data_quality import (  # noqa: E402
    annotate_sample,
    annotate_training_payload,
)
from services.training_epoch import load_training_epoch_start  # noqa: E402

CursorCounter = Callable[[], Awaitable[int]]
SampleLoader = Callable[[], Awaitable[list[dict[str, Any]]]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _compact_shadow_sample(mapping: Any) -> dict[str, Any] | None:
    """Build the same bounded shadow row used by the full training loader."""

    row = _shadow_sample_from_mapping(mapping)
    features = _snapshot(row.get("features"))
    if not features:
        return None
    features.setdefault("symbol", row.get("symbol"))
    features.setdefault("decision_confidence", _safe_float(row.get("decision_confidence")))
    features.setdefault("horizon_minutes", int(row.get("horizon_minutes") or 10))
    compact_features = _compact_local_ai_tools_features(features)
    if not compact_features:
        return None
    return {
        "id": int(row.get("id") or 0),
        "decision_id": int(row.get("decision_id") or 0) or None,
        "label_version": str(row.get("label_version") or ""),
        "symbol": row.get("symbol"),
        "analysis_type": row.get("analysis_type"),
        "decision_action": row.get("decision_action"),
        "decision_confidence": _safe_float(row.get("decision_confidence")),
        "horizon_minutes": int(row.get("horizon_minutes") or 10),
        "features": compact_features,
        "long_return_pct": _safe_float(row.get("long_return_pct")),
        "short_return_pct": _safe_float(row.get("short_return_pct")),
        "label_timestamp": row.get("label_timestamp"),
        "best_action": row.get("best_action"),
        "missed_opportunity": bool(row.get("missed_opportunity")),
    }


async def _iter_shadow_samples_stream():
    """Yield bounded shadow rows page-by-page instead of materializing history."""

    epoch_start = load_training_epoch_start()
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


def _shadow_group_key(sample: dict[str, Any]) -> str:
    decision_id = int(sample.get("decision_id") or 0)
    sample_id = int(sample.get("id") or 0)
    return f"shadow_decision:{decision_id or sample_id}"


async def _streaming_cursor_probe() -> dict[str, Any]:
    """Compute the canonical cursor with bounded memory and exact row counts."""

    group_counts: dict[str, int] = defaultdict(int)
    group_trainable_counts: dict[str, int] = defaultdict(int)
    group_weight_sums: dict[str, float] = defaultdict(float)
    group_budgets: dict[str, float] = defaultdict(float)
    first_pass_seen: dict[tuple[int, int, str], int] = {}

    async for raw in _iter_shadow_samples_stream():
        sample = annotate_sample(_duplicate_metadata(raw, first_pass_seen), "shadow")
        key = _shadow_group_key(sample)
        group_counts[key] += 1
        if not bool(sample.get("exclude_from_training")):
            weight = max(_safe_float(sample.get("sample_weight")), 0.0)
            group_trainable_counts[key] += 1
            group_weight_sums[key] += weight
            group_budgets[key] = max(group_budgets[key], weight)

    profile_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    market_count = 0
    market_groups: set[str] = set()
    second_pass_seen: dict[tuple[int, int, str], int] = {}
    async for raw in _iter_shadow_samples_stream():
        sample = annotate_sample(_duplicate_metadata(raw, second_pass_seen), "shadow")
        key = _shadow_group_key(sample)
        base = max(_safe_float(sample.get("sample_weight")), 0.0)
        total = group_weight_sums.get(key, 0.0)
        budget = group_budgets.get(key, 0.0)
        multiplier = budget / total if total > 0.0 else 0.0
        adjusted = 0.0 if sample.get("exclude_from_training") else base * multiplier
        sample["sample_weight"] = adjusted
        sample["correlation_weight"] = {
            "source": "shared_decision_or_authoritative_lifecycle_identity",
            "correlation_group": key,
            "group_sample_count": group_counts[key],
            "group_trainable_count": group_trainable_counts[key],
            "base_quality_weight": base,
            "group_effective_weight_budget": budget,
            "normalization_multiplier": multiplier,
            "effective_sample_weight": adjusted,
            "fixed_sampling_ratio": False,
        }
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

    trade_samples = await _load_trade_samples()
    trade_payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=trade_samples,
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    cost_count = 0
    cost_groups: set[str] = set()
    for sample in trade_payload["trade_samples"]:
        identity = authoritative_cost_training_identity(sample)
        if identity is None:
            continue
        cost_count += 1
        cost_groups.add(str(identity["decision_group"]))
        numeric = float(identity["execution_cost_pct"])
        stats = profile_stats["authoritative_execution_cost_pct"]
        stats[0] += 1.0
        stats[1] += numeric
        stats[2] += numeric * numeric

    profile: dict[str, dict[str, float | int]] = {}
    for key, (count, total, square_total) in profile_stats.items():
        if count <= 0:
            continue
        mean = total / count
        variance = max(square_total / count - mean * mean, 0.0)
        profile[key] = {"count": int(count), "mean": mean, "std": math.sqrt(variance)}
    return {
        "version": TRAINING_CURSOR_VERSION,
        "completed_market_sample_count": market_count,
        "completed_trade_sample_count": len(trade_payload["trade_samples"]),
        "completed_authoritative_cost_sample_count": cost_count,
        "completed_market_decision_group_count": len(market_groups),
        "completed_authoritative_cost_decision_group_count": len(cost_groups),
        "completed_training_decision_group_count": len(market_groups | cost_groups),
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
