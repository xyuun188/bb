"""Compact, atomic observability for the isolated paper-strategy evaluator."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import settings

REPORT_RELATIVE_PATH = "strategy_replay/latest.json"
REPLAY_INTERVAL_SECONDS = 1800
REPLAY_TIMEOUT_SECONDS = 600


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def replay_report_path() -> Path:
    return settings.data_dir / REPORT_RELATIVE_PATH


def summarize_replay_governance(payload: dict[str, Any]) -> dict[str, Any]:
    schedule = _dict(payload.get("schedule"))
    blueprint = _dict(payload.get("model_strategy_blueprint"))
    replay = _dict(schedule.get("historical_model_replay"))
    candidates = schedule.get("candidates") or []
    return {
        "status": "ok" if payload.get("champion_reconciled") is True else "blocked",
        "generated_at": datetime.now(UTC).isoformat(),
        "execution_scope": "paper_only",
        "live_execution_permission": False,
        "submits_orders": False,
        "model_version": blueprint.get("model_version"),
        "champion_reconciled": payload.get("champion_reconciled") is True,
        "governance_blocker": payload.get("governance_blocker"),
        "scheduler_mode": schedule.get("scheduler_mode"),
        "candidate_count": len(candidates),
        "governed_candidate_count": schedule.get("governed_candidate_count"),
        "profitability_status": (
            "replay_incomplete"
            if replay.get("status") != "complete"
            else "validated_candidates_available"
            if schedule.get("governed_candidate_count")
            else "no_validated_fee_after_strategy"
        ),
        "candidate_rejection_counts": dict(Counter(
            str(reason)
            for candidate in candidates
            for reason in _dict(candidate.get("promotion")).get("rejection_reasons", [])
        )),
        "historical_model_replay": replay,
        "shadow_feedback": _dict(payload.get("feedback")).get("shadow_feedback"),
        "paper_strategy_champion": payload.get("paper_strategy_champion"),
        "leading_candidates": [
            {
                "selector": _dict(candidate.get("params")).get("selector"),
                "development": _dict(candidate.get("backtest")).get("metrics"),
                "exam": _dict(candidate.get("shadow_validation")).get("metrics"),
                "forward_exam": _dict(candidate.get("forward_validation")).get("metrics"),
                "rejections": _dict(candidate.get("promotion")).get("rejection_reasons"),
            }
            for candidate in candidates[:5]
        ],
    }


def write_replay_governance(report: dict[str, Any], *, path: Path | None = None) -> None:
    target = path or replay_report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def load_replay_governance(
    *, path: Path | None = None, now: datetime | None = None
) -> dict[str, Any]:
    try:
        report = json.loads((path or replay_report_path()).read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(str(report["generated_at"]).replace("Z", "+00:00"))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=UTC)
    except (OSError, KeyError, TypeError, ValueError):
        return {"status": "missing", "stale": True, "live_execution_permission": False}
    age = ((now or datetime.now(UTC)) - generated).total_seconds()
    return {
        **report,
        "age_seconds": round(age, 3),
        "stale": age < 0 or age > REPLAY_INTERVAL_SECONDS + REPLAY_TIMEOUT_SECONDS,
    }
