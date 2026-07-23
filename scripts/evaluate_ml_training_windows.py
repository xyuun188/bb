"""Dry-run local ML training-window variants against shadow outcomes.

This diagnostic is intentionally read-only. It compares candidate window
selection policies with the production trainer and readiness gates, but never
writes model artifacts or mutates shadow rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import get_read_session_ctx  # noqa: E402
from models.learning import ShadowBacktest  # noqa: E402
from services.ml_readiness import build_ml_readiness_report  # noqa: E402
from services.ml_signal_service import (  # noqa: E402
    _influence_policy,
    _shadow_action,
    _shadow_is_trainable_trade_opportunity,
    _shadow_quality_sample,
    _shadow_sort_key,
    _shadow_training_columns,
    _shadow_training_row_from_mapping,
    build_training_frame,
    count_shadow_training_rows,
    select_shadow_training_rows,
    shadow_training_quality_report,
    train_from_frame,
)
from services.training_data_quality import assess_shadow_sample  # noqa: E402

TRADE_ACTIONS = {"long", "short"}


@dataclass(frozen=True)
class WindowVariant:
    name: str
    selector: Callable[[list[Any], int], list[Any]]
    description: str


def _dedupe_rows(rows: list[Any]) -> list[Any]:
    deduped: dict[int, Any] = {}
    for row in rows:
        row_id = int(getattr(row, "id", 0) or 0)
        if row_id and row_id not in deduped:
            deduped[row_id] = row
    return sorted(deduped.values(), key=_shadow_sort_key, reverse=True)


def _is_trainable(row: Any) -> bool:
    return not assess_shadow_sample(_shadow_quality_sample(row)).exclude_from_training


def _is_best_trade(row: Any) -> bool:
    return _shadow_action(row, "best_action") in TRADE_ACTIONS


def _clean_trainable_rows(rows: list[Any]) -> list[Any]:
    return [
        row
        for row in _dedupe_rows(rows)
        if _is_trainable(row)
    ]


def _quality_sorted(rows: list[Any]) -> list[Any]:
    return _dedupe_rows(rows)


def _select_quality_first(rows: list[Any], *, limit: int) -> list[Any]:
    return _quality_sorted(_clean_trainable_rows(rows))[:limit]


def _select_current_policy(rows: list[Any], _limit: int) -> list[Any]:
    return select_shadow_training_rows(rows)


def _select_exclude_best_hold(rows: list[Any], limit: int) -> list[Any]:
    candidates = [row for row in _dedupe_rows(rows) if _shadow_is_trainable_trade_opportunity(row)]
    return _quality_sorted(candidates)[:limit]


def _select_cap_best_hold(rows: list[Any], limit: int, *, max_hold_share: float) -> list[Any]:
    cap = max(min(float(max_hold_share), 1.0), 0.0)
    best_hold_limit = int(max(int(limit), 1) * cap)
    selected: list[Any] = []
    selected_ids: set[int] = set()
    best_hold_count = 0

    for row in _quality_sorted(_clean_trainable_rows(rows)):
        row_id = int(getattr(row, "id", 0) or 0)
        if row_id in selected_ids:
            continue
        best_trade = _is_best_trade(row)
        if not best_trade:
            if best_hold_count >= best_hold_limit:
                continue
            best_hold_count += 1
        selected.append(row)
        selected_ids.add(row_id)
        if len(selected) >= limit:
            break
    return selected


def _select_balanced_action_exclude_best_hold(rows: list[Any], limit: int) -> list[Any]:
    candidates = [row for row in _dedupe_rows(rows) if _shadow_is_trainable_trade_opportunity(row)]
    grouped: dict[str, list[Any]] = {
        "long": [],
        "short": [],
        "hold": [],
        "other": [],
    }
    for row in _quality_sorted(candidates):
        action = _shadow_action(row, "decision_action")
        if action in {"long", "short", "hold"}:
            grouped[action].append(row)
        else:
            grouped["other"].append(row)

    safe_limit = max(int(limit), 1)
    targets = {
        "long": safe_limit // 3,
        "short": safe_limit // 3,
        "hold": safe_limit - (safe_limit // 3) * 2,
        "other": 0,
    }
    selected: list[Any] = []
    selected_ids: set[int] = set()

    def add_from(bucket: list[Any], target: int) -> None:
        for row in bucket:
            if len(selected) >= safe_limit or len(selected) >= target:
                return
            row_id = int(getattr(row, "id", 0) or 0)
            if row_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row_id)

    running_target = 0
    for key in ("long", "short", "hold", "other"):
        running_target += targets[key]
        add_from(grouped[key], running_target)
    for row in _quality_sorted(candidates):
        if len(selected) >= safe_limit:
            break
        row_id = int(getattr(row, "id", 0) or 0)
        if row_id not in selected_ids:
            selected.append(row)
            selected_ids.add(row_id)
    return selected


def _trainable_trade_rows(rows: list[Any]) -> list[Any]:
    return _quality_sorted(
        [row for row in _dedupe_rows(rows) if _shadow_is_trainable_trade_opportunity(row)]
    )


def _select_decision_equals_best(rows: list[Any], limit: int) -> list[Any]:
    candidates = [
        row
        for row in _trainable_trade_rows(rows)
        if _shadow_action(row, "decision_action") == _shadow_action(row, "best_action")
    ]
    return candidates[:limit]


def _select_decision_not_equals_best(rows: list[Any], limit: int) -> list[Any]:
    candidates = [
        row
        for row in _trainable_trade_rows(rows)
        if _shadow_action(row, "decision_action") != _shadow_action(row, "best_action")
    ]
    return candidates[:limit]


def _select_horizon(rows: list[Any], limit: int, horizon_minutes: int) -> list[Any]:
    candidates = [
        row
        for row in _trainable_trade_rows(rows)
        if int(getattr(row, "horizon_minutes", 0) or 0) == int(horizon_minutes)
    ]
    return candidates[:limit]


def _select_decision_side(rows: list[Any], limit: int, side: str) -> list[Any]:
    candidates = [
        row for row in _trainable_trade_rows(rows) if _shadow_action(row, "decision_action") == side
    ]
    return candidates[:limit]


def _select_best_side(rows: list[Any], limit: int, side: str) -> list[Any]:
    candidates = [
        row for row in _trainable_trade_rows(rows) if _shadow_action(row, "best_action") == side
    ]
    return candidates[:limit]


def _select_decision_equals_best_side(rows: list[Any], limit: int, side: str) -> list[Any]:
    candidates = [
        row
        for row in _trainable_trade_rows(rows)
        if _shadow_action(row, "decision_action") == side
        and _shadow_action(row, "best_action") == side
    ]
    return candidates[:limit]


def variants() -> list[WindowVariant]:
    return [
        WindowVariant(
            name="current_trade_opportunity_only",
            selector=_select_current_policy,
            description=(
                "Current production policy: decision_action and best_action are both long/short."
            ),
        ),
        WindowVariant(
            name="quality_first_trainable",
            selector=lambda rows, limit: _select_quality_first(rows, limit=limit),
            description="Quality-ranked trainable rows with low-confidence holds removed.",
        ),
        WindowVariant(
            name="exclude_best_hold",
            selector=_select_exclude_best_hold,
            description="Only rows with executable long/short decisions and long/short outcomes.",
        ),
        WindowVariant(
            name="cap_best_hold_30pct",
            selector=lambda rows, limit: _select_cap_best_hold(rows, limit, max_hold_share=0.30),
            description="Quality-ranked rows with non-trade best_action capped at 30%.",
        ),
        WindowVariant(
            name="cap_best_hold_20pct",
            selector=lambda rows, limit: _select_cap_best_hold(rows, limit, max_hold_share=0.20),
            description="Quality-ranked rows with non-trade best_action capped at 20%.",
        ),
        WindowVariant(
            name="balanced_action_exclude_best_hold",
            selector=_select_balanced_action_exclude_best_hold,
            description="Exclude best_action=hold and balance decision_action buckets.",
        ),
    ]


def extended_variants() -> list[WindowVariant]:
    """Diagnostic-only windows for side/horizon root-cause analysis.

    These variants intentionally do not imply production promotion. Some of them
    are biased views, such as decision=best, and are useful only to reveal where
    the current ML labels or features fail.
    """

    return [
        WindowVariant(
            name="diagnostic_decision_equals_best",
            selector=_select_decision_equals_best,
            description=(
                "Diagnostic only: rows where the original decision side matched hindsight "
                "best_action. Watch for survivorship bias before promoting anything."
            ),
        ),
        WindowVariant(
            name="diagnostic_decision_not_equals_best",
            selector=_select_decision_not_equals_best,
            description=(
                "Diagnostic only: rows where the original decision side disagreed with "
                "hindsight best_action; useful for direction-error analysis."
            ),
        ),
        WindowVariant(
            name="diagnostic_horizon_10",
            selector=lambda rows, limit: _select_horizon(rows, limit, 10),
            description="Diagnostic only: trade-opportunity rows with 10 minute labels.",
        ),
        WindowVariant(
            name="diagnostic_horizon_30",
            selector=lambda rows, limit: _select_horizon(rows, limit, 30),
            description="Diagnostic only: trade-opportunity rows with 30 minute labels.",
        ),
        WindowVariant(
            name="diagnostic_horizon_60",
            selector=lambda rows, limit: _select_horizon(rows, limit, 60),
            description="Diagnostic only: trade-opportunity rows with 60 minute labels.",
        ),
        WindowVariant(
            name="diagnostic_decision_long",
            selector=lambda rows, limit: _select_decision_side(rows, limit, "long"),
            description="Diagnostic only: rows whose original decision was long.",
        ),
        WindowVariant(
            name="diagnostic_decision_short",
            selector=lambda rows, limit: _select_decision_side(rows, limit, "short"),
            description="Diagnostic only: rows whose original decision was short.",
        ),
        WindowVariant(
            name="diagnostic_best_long",
            selector=lambda rows, limit: _select_best_side(rows, limit, "long"),
            description="Diagnostic only: rows whose hindsight best_action was long.",
        ),
        WindowVariant(
            name="diagnostic_best_short",
            selector=lambda rows, limit: _select_best_side(rows, limit, "short"),
            description="Diagnostic only: rows whose hindsight best_action was short.",
        ),
        WindowVariant(
            name="diagnostic_decision_equals_best_long",
            selector=lambda rows, limit: _select_decision_equals_best_side(rows, limit, "long"),
            description="Diagnostic only: long rows where decision_action matched best_action.",
        ),
        WindowVariant(
            name="diagnostic_decision_equals_best_short",
            selector=lambda rows, limit: _select_decision_equals_best_side(rows, limit, "short"),
            description="Diagnostic only: short rows where decision_action matched best_action.",
        ),
    ]


async def load_candidate_rows() -> list[Any]:
    columns = _shadow_training_columns()
    filters = (
        ShadowBacktest.status == "completed",
        ShadowBacktest.long_return_pct.is_not(None),
        ShadowBacktest.short_return_pct.is_not(None),
    )
    order_by = (ShadowBacktest.created_at.desc(), ShadowBacktest.id.desc())

    async with get_read_session_ctx() as session:

        async def load(stmt: Any) -> list[Any]:
            return [
                _shadow_training_row_from_mapping(row)
                for row in (await session.execute(stmt)).mappings().all()
            ]

        rows = await load(select(*columns).where(*filters).order_by(*order_by))
    return _dedupe_rows(rows)


def _counter(rows: list[Any], field: str) -> dict[str, int]:
    return dict(Counter(_shadow_action(row, field) or "unknown" for row in rows).most_common())


def _quality_counts(rows: list[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[assess_shadow_sample(_shadow_quality_sample(row)).status] += 1
    return dict(counts.most_common())


def _metric_subset(metadata: dict[str, Any]) -> dict[str, Any]:
    metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    keys = (
        "long_auc",
        "short_auc",
        "long_pr_auc",
        "short_pr_auc",
        "long_accuracy",
        "short_accuracy",
        "top_long_avg_return_pct",
        "bottom_long_avg_return_pct",
        "top_short_avg_return_pct",
        "bottom_short_avg_return_pct",
        "top_long_win_rate",
        "bottom_long_win_rate",
        "top_short_win_rate",
        "bottom_short_win_rate",
    )
    return {key: metrics.get(key) for key in keys}


def _compact_bucket_diagnostics(metadata: dict[str, Any]) -> dict[str, Any]:
    raw = metadata.get("score_bucket_diagnostics")
    if not isinstance(raw, dict):
        return {}
    compact: dict[str, Any] = {}
    for side in ("long", "short"):
        side_block = raw.get(side) if isinstance(raw.get(side), dict) else {}
        compact[side] = {}
        for bucket in ("top", "bottom"):
            item = side_block.get(bucket) if isinstance(side_block.get(bucket), dict) else {}
            compact[side][bucket] = {
                "count": item.get("count"),
                "avg_return_pct": item.get("avg_return_pct"),
                "win_rate": item.get("win_rate"),
                "action_counts": item.get("action_counts"),
                "best_action_counts": item.get("best_action_counts"),
                "data_quality_status_counts": item.get("data_quality_status_counts"),
            }
    return compact


def evaluate_variant(
    variant: WindowVariant,
    candidates: list[Any],
    *,
    limit: int,
    completed_count: int,
) -> dict[str, Any]:
    selected = variant.selector(candidates, limit)
    quality_state = shadow_training_quality_report(selected)
    frame = build_training_frame(selected)
    result: dict[str, Any] = {
        "variant": variant.name,
        "description": variant.description,
        "selected_row_count": len(selected),
        "frame_sample_count": int(len(frame)),
        "selection_counts": {
            "decision_action": _counter(selected, "decision_action"),
            "best_action": _counter(selected, "best_action"),
            "quality_status": _quality_counts(selected),
        },
    }
    metadata = train_from_frame(
        frame,
        completed_sample_count=completed_count,
        training_quality_report=quality_state["quality_report"],
        persist_artifact=False,
    )
    influence = _influence_policy(metadata)
    readiness = build_ml_readiness_report(metadata, influence)
    blocking = readiness.get("blocking_reasons")
    if not isinstance(blocking, list):
        blocking = []
    result.update(
        {
            "status": "evaluated",
            "readiness_state": readiness.get("state"),
            "live_ml_ready": bool(readiness.get("live_ml_ready")),
            "blocking_reason_codes": [
                str(item.get("code"))
                for item in blocking
                if isinstance(item, dict) and item.get("code")
            ],
            "readiness_thresholds": readiness.get("thresholds"),
            "readiness_metrics": readiness.get("metrics"),
            "metrics": _metric_subset(metadata),
            "training_window_composition": metadata.get("training_window_composition"),
            "quality_totals": (
                quality_state.get("quality_report", {}).get("totals")
                if isinstance(quality_state.get("quality_report"), dict)
                else {}
            ),
            "score_bucket_diagnostics": _compact_bucket_diagnostics(metadata),
        }
    )
    return result


async def run(*, include_extended: bool = False) -> dict[str, Any]:
    candidates = await load_candidate_rows()
    completed_count = await count_shadow_training_rows()
    selected_variants = variants()
    if include_extended:
        selected_variants = [*selected_variants, *extended_variants()]
    results = [
        evaluate_variant(
            variant,
            candidates,
            limit=len(candidates),
            completed_count=completed_count,
        )
        for variant in selected_variants
    ]
    ready_variants = [
        item["variant"] for item in results if item.get("live_ml_ready")
    ]
    return {
        "dry_run": True,
        "artifact_persisted": False,
        "database_mutated": False,
        "training_window_sample_count": len(candidates),
        "extended_diagnostics": bool(include_extended),
        "candidate_row_count": len(candidates),
        "completed_shadow_sample_count": completed_count,
        "ready_variants": ready_variants,
        "recommendation": (
            "A variant passes readiness; inspect diagnostics before promoting policy."
            if ready_variants
            else "No variant passed readiness; do not enable local ML live influence."
        ),
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Also evaluate side/horizon diagnostic windows. Slower and read-only.",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run(
            include_extended=bool(args.extended),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
