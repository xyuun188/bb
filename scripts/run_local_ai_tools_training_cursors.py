#!/usr/bin/env python3
"""Read Local AI training cursors in an isolated database process."""

from __future__ import annotations

import asyncio
import json
import os
import sys
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

from sqlalchemy import func, select  # noqa: E402

from core.safe_output import safe_error_text  # noqa: E402
from db.session import close_db, get_read_session_ctx  # noqa: E402
from models.learning import ShadowBacktest  # noqa: E402
from scripts.train_local_ai_tools_models import (  # noqa: E402
    _REMOTE_TRAINING_MAX_SHADOW_SAMPLES,
    _completed_shadow_sample_count,
    _load_shadow_samples,
    _load_trade_samples,
)
from services.local_ai_training_contract import (  # noqa: E402
    LOCAL_AI_TOOLS_TRAINING_TRANSPORT_VERSION,
    TRAINING_CURSOR_VERSION,
    local_ai_training_cursor,
)
from services.ml_training_dataset import (  # noqa: E402
    probe_authoritative_trade_training_cursor,
    shadow_training_candidate_filters,
)
from services.training_data_quality import (  # noqa: E402
    annotate_training_payload,
)
from services.training_epoch import load_training_data_start  # noqa: E402

CursorCounter = Callable[[], Awaitable[int]]
SampleLoader = Callable[[], Awaitable[list[dict[str, Any]]]]


async def _completed_shadow_decision_group_count() -> int:
    """Count candidate decision groups in SQL without Python quality reprocessing."""

    epoch_start = load_training_data_start()
    group_key = func.coalesce(
        func.nullif(ShadowBacktest.decision_id, 0),
        -ShadowBacktest.id,
    )
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(func.count(func.distinct(group_key))).where(
                *shadow_training_candidate_filters(epoch_start),
            )
        )
        return int(result.scalar() or 0)


async def _streaming_cursor_probe() -> dict[str, Any]:
    """Fingerprint the same bounded shadow window used by the trainer."""

    shadow_count, group_count = await asyncio.gather(
        _completed_shadow_sample_count(),
        _completed_shadow_decision_group_count(),
    )
    shadow_samples, trade_probe = await asyncio.gather(
        _load_shadow_samples(sample_budget=_REMOTE_TRAINING_MAX_SHADOW_SAMPLES),
        probe_authoritative_trade_training_cursor(),
    )
    payload = annotate_training_payload(
        shadow_samples=shadow_samples,
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    cursor = local_ai_training_cursor(
        shadow_samples=payload["shadow_samples"],
        trade_samples=[],
    )
    return {
        "version": TRAINING_CURSOR_VERSION,
        "training_transport_version": LOCAL_AI_TOOLS_TRAINING_TRANSPORT_VERSION,
        "completed_shadow_sample_count": int(shadow_count),
        "completed_market_sample_count": int(cursor["completed_market_sample_count"]),
        "completed_market_decision_group_count": int(group_count),
        "completed_training_decision_group_count": int(group_count),
        "authoritative_trade_training_probe": trade_probe,
        "cursor_count_scope": (
            "bounded_training_sample_and_candidate_decision_groups_and_"
            "settlement_fact_fingerprint"
        ),
        "cursor_policy": "canonical_bounded_training_sample",
        "training_distribution_profile": cursor["training_distribution_profile"],
    }


async def run_streaming_once() -> dict[str, Any]:
    try:
        cursor = await _streaming_cursor_probe()
        return {
            "trained": False,
            "reason": "cursor_probe_complete",
            "training_transport_version": LOCAL_AI_TOOLS_TRAINING_TRANSPORT_VERSION,
            **cursor,
            "training_process_isolated": True,
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
