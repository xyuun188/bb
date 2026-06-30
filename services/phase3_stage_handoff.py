"""Read-only Phase 3 stage handoff evaluator.

This module turns the Phase 3 evidence reports into one operator-facing stage.
It never starts trading, submits orders, changes routing, or writes artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import settings

GO_NO_GO_REL = "phase3_go_no_go_reports/latest.json"
OBSERVATION_REL = "phase3_paper_resume_observation_reports/latest.json"
REBUILD_REL = "phase3_rebuild_preflight_reports/latest.json"
OKX_DAILY_REL = "okx_daily_reconciliation_reports/latest.json"
SPECIALIST_DATA_REL = "phase3/specialist_shadow_evaluation_latest.json"
SPECIALIST_REPORT_REL = "reports/phase3/specialist_shadow_evaluation_latest.json"
DEFAULT_REPORT_MAX_AGE_SECONDS = 3 * 3600


def _now() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            result = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _report_time(report: dict[str, Any]) -> datetime | None:
    return _parse_utc_datetime(
        report.get("checked_at")
        or report.get("generated_at")
        or report.get("created_at")
        or report.get("timestamp")
    )


def _age_seconds(report: dict[str, Any]) -> float | None:
    value = _report_time(report)
    if value is None:
        return None
    return max((_now() - value).total_seconds(), 0.0)


def _is_fresh(report: dict[str, Any], max_age_seconds: int) -> bool:
    age = _age_seconds(report)
    return age is not None and age <= max_age_seconds


def _blocker(code: str, message: str, *, evidence: Any | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "severity": "blocking", "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _warning(code: str, message: str, *, evidence: Any | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "severity": "warning", "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _specialist_tail_risk_models(models: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in models:
        row = _safe_dict(raw)
        if not row:
            continue
        tail_loss_count = _safe_int(row.get("tail_loss_count"))
        blockers = _safe_list(row.get("promotion_blockers"))
        if tail_loss_count <= 0 and "false_signal_loss_exceeds_floor" not in blockers:
            continue
        rows.append(
            {
                "model": row.get("model"),
                "tool": row.get("tool"),
                "promotion_blockers": blockers[:6],
                "tail_loss_count": tail_loss_count,
                "worst_realized_return_pct": row.get("worst_realized_return_pct"),
                "tail_loss_symbols": _safe_list(row.get("tail_loss_symbols"))[:6],
                "worst_samples": _safe_list(row.get("worst_samples"))[:3],
            }
        )
    rows.sort(
        key=lambda item: (
            _safe_int(item.get("tail_loss_count")),
            abs(_safe_float(item.get("worst_realized_return_pct"))),
        ),
        reverse=True,
    )
    return rows[:6]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    payload.setdefault("available", True)
    payload.setdefault("report_path", str(path))
    return payload


def _first_report(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        payload = _read_json(path)
        if payload:
            return payload
    return {"available": False, "candidate_paths": [str(path) for path in paths]}


def _report_status(report: dict[str, Any]) -> str:
    return str(report.get("status") or "missing").lower()


def evaluate_phase3_stage_handoff_inputs(
    *,
    go_no_go_report: dict[str, Any],
    observation_report: dict[str, Any],
    specialist_shadow_report: dict[str, Any],
    rebuild_preflight_report: dict[str, Any],
    okx_daily_report: dict[str, Any],
    report_max_age_seconds: int = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Evaluate the current Phase 3 operator handoff stage."""

    go_report = _safe_dict(go_no_go_report)
    gate = _safe_dict(go_report.get("go_no_go"))
    observation = _safe_dict(observation_report)
    specialist = _safe_dict(specialist_shadow_report)
    rebuild = _safe_dict(rebuild_preflight_report)
    okx_daily = _safe_dict(okx_daily_report)
    max_age = max(int(report_max_age_seconds or DEFAULT_REPORT_MAX_AGE_SECONDS), 60)

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    passed: list[str] = []

    if not go_report.get("available"):
        blockers.append(
            _blocker(
                "go_no_go_report_missing",
                "Latest Phase 3 Go/No-Go report is missing.",
                evidence=go_report.get("candidate_paths"),
            )
        )
    elif not _is_fresh(go_report, max_age):
        blockers.append(
            _blocker(
                "go_no_go_report_stale",
                "Latest Phase 3 Go/No-Go report is stale.",
                evidence={
                    "checked_at": go_report.get("checked_at"),
                    "age_seconds": _age_seconds(go_report),
                    "max_age_seconds": max_age,
                },
            )
        )
    else:
        passed.append("go_no_go_report_fresh")

    gate_blockers = _safe_list(gate.get("blockers"))
    if gate_blockers:
        blockers.append(
            _blocker(
                "go_no_go_has_blockers",
                "Phase 3 Go/No-Go still has hard blockers.",
                evidence=gate_blockers[:8],
            )
        )

    observation_status = _report_status(observation)
    if observation.get("available") and not _is_fresh(observation, max_age):
        warnings.append(
            _warning(
                "paper_observation_report_stale",
                "Latest paper observation report is stale.",
                evidence={
                    "checked_at": observation.get("checked_at"),
                    "age_seconds": _age_seconds(observation),
                    "max_age_seconds": max_age,
                },
            )
        )
    elif observation.get("available"):
        passed.append("paper_observation_report_fresh")
    else:
        warnings.append(
            _warning(
                "paper_observation_report_missing",
                "Latest paper observation report is missing.",
                evidence=observation.get("candidate_paths"),
            )
        )

    if observation_status == "critical":
        blockers.append(
            _blocker(
                "paper_observation_critical",
                "Post-resume paper observation is critical.",
                evidence=_safe_list(observation.get("blockers"))[:8],
            )
        )

    if specialist.get("available") and bool(specialist.get("live_mutation")):
        blockers.append(
            _blocker(
                "specialist_shadow_live_mutation",
                "Specialist shadow evidence must remain shadow-only.",
                evidence={"live_mutation": specialist.get("live_mutation")},
            )
        )
    elif specialist.get("available"):
        passed.append("specialist_shadow_report_available")
    else:
        warnings.append(
            _warning(
                "specialist_shadow_report_missing",
                "Specialist shadow evaluation report is missing.",
                evidence=specialist.get("candidate_paths"),
            )
        )

    okx_summary = _safe_dict(okx_daily.get("issue_ledger_summary"))
    okx_unresolved = int(okx_summary.get("unresolved") or 0) if okx_summary else 0
    if okx_daily.get("available") and okx_unresolved > 0:
        blockers.append(
            _blocker(
                "okx_daily_reconciliation_unresolved",
                "OKX daily reconciliation still has unresolved issues.",
                evidence=okx_summary,
            )
        )
    elif okx_daily.get("available"):
        passed.append("okx_daily_reconciliation_available")
    else:
        warnings.append(
            _warning(
                "okx_daily_reconciliation_report_missing",
                "OKX daily reconciliation report is missing.",
                evidence=okx_daily.get("candidate_paths"),
            )
        )

    if rebuild.get("available"):
        passed.append("rebuild_preflight_report_available")
    else:
        warnings.append(
            _warning(
                "rebuild_preflight_report_missing",
                "Phase 3 rebuild preflight report is missing.",
                evidence=rebuild.get("candidate_paths"),
            )
        )

    gate_status = str(gate.get("status") or go_report.get("status") or "missing").lower()
    can_start_paper = bool(gate.get("can_start_paper_with_operator_approval"))
    can_enter_canary = bool(gate.get("can_enter_canary_with_operator_approval"))
    gate_inputs = _safe_dict(gate.get("inputs"))
    rebuild_promotion = _safe_dict(rebuild.get("promotion_recommendation"))
    promotion_canary_ready_raw = (
        rebuild_promotion.get("canary_ready")
        if rebuild_promotion
        else gate_inputs.get("promotion_canary_ready")
    )
    promotion_canary_ready = (
        bool(promotion_canary_ready_raw)
        if promotion_canary_ready_raw is not None
        else None
    )
    if promotion_canary_ready is False:
        can_enter_canary = False
        warnings.append(
            _warning(
                "model_promotion_canary_not_ready",
                "Model promotion is still shadow-only.",
                evidence={
                    "recommended_stage": rebuild_promotion.get("recommended_stage"),
                    "canary_blocking_reasons": _safe_list(
                        rebuild_promotion.get("canary_blocking_reasons")
                    )[:8],
                },
            )
        )
    can_use_observation = bool(observation.get("can_use_for_promotion"))
    paper_active = bool(observation.get("paper_active"))
    specialist_summary = _safe_dict(specialist.get("summary"))
    specialist_promotion_ready_count = int(
        specialist_summary.get("promotion_ready_count")
        if specialist_summary.get("promotion_ready_count") is not None
        else specialist.get("promotion_ready_count") or 0
    )
    specialist_blocked_count = int(
        specialist_summary.get("blocked_count")
        if specialist_summary.get("blocked_count") is not None
        else specialist.get("blocked_count") or 0
    )
    specialist_models = _safe_list(specialist.get("models"))
    specialist_tail_risk_models = _specialist_tail_risk_models(specialist_models)
    specialist_canary_blocked = False
    if specialist.get("available") and specialist_promotion_ready_count <= 0:
        warnings.append(
            _warning(
                "specialist_shadow_no_promotion_ready_model",
                "Specialist shadow models are still collecting evidence and cannot enter canary yet.",
                evidence={
                    "promotion_ready_count": specialist_promotion_ready_count,
                    "blocked_count": specialist_blocked_count,
                    "top_blocked_reasons": _safe_list(
                        specialist_summary.get("top_blocked_reasons")
                    )[:8],
                    "tail_risk_models": specialist_tail_risk_models,
                },
            )
        )
        specialist_canary_blocked = gate_status == "paper_observation_healthy" and can_use_observation
    elif specialist.get("available"):
        passed.append("specialist_shadow_has_promotion_ready_model")

    if blockers:
        status = "blocked"
        stage = "fix_hard_blockers"
        next_action = "Fix the listed hard blockers before any paper/canary action."
    elif gate_status == "paper_observation_healthy" and can_use_observation:
        status = (
            "paper_observation_healthy"
            if specialist_canary_blocked or not can_enter_canary
            else "canary_review_ready"
        )
        stage = (
            "stay_shadow_improve_specialists"
            if specialist_canary_blocked or not can_enter_canary
            else "operator_review_for_canary"
        )
        next_action = (
            "Keep paper running in shadow and improve specialist false-signal loss before canary review."
            if specialist_canary_blocked or not can_enter_canary
            else "Review canary evidence manually; live remains disabled and requires a separate release."
        )
    elif paper_active or observation_status == "warming_up":
        status = "post_resume_observing"
        stage = "post_resume_observation_window"
        next_action = (
            "Keep collecting OKX, shadow, specialist, and trade-quality evidence until observation is healthy."
        )
    elif gate_status == "paper_resume_ready" and can_start_paper:
        status = "paper_start_ready"
        stage = "paper_start_pending_operator_approval"
        next_action = (
            "Operator may start paper only through scripts/start_phase3_paper_with_preflight.py "
            "--start-service --confirm-resume-paper CONFIRM_PHASE3_PAPER_RESUME."
        )
    else:
        status = "waiting_for_evidence"
        stage = "stay_shadow_collect_evidence"
        next_action = "Continue shadow-only evidence collection until Go/No-Go advances."

    return {
        "status": status,
        "stage": stage,
        "next_action": next_action,
        "checked_at": _now_iso(),
        "read_only": True,
        "audit_only": True,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "live_mutation": False,
        "can_start_paper_with_operator_approval": status == "paper_start_ready",
        "can_enter_canary_with_operator_approval": status == "canary_review_ready",
        "can_enter_live": False,
        "blockers": blockers,
        "warnings": warnings,
        "passed_checks": list(dict.fromkeys(passed)),
        "inputs": {
            "go_no_go_status": gate_status,
            "go_no_go_next_step": gate.get("next_step"),
            "paper_observation_status": observation_status,
            "paper_active": paper_active,
            "paper_can_use_for_promotion": can_use_observation,
            "promotion_canary_ready": promotion_canary_ready,
            "specialist_promotion_ready_count": specialist_promotion_ready_count,
            "specialist_blocked_count": specialist_blocked_count,
            "specialist_canary_blocked": specialist_canary_blocked,
            "specialist_tail_loss_model_count": len(specialist_tail_risk_models),
            "specialist_tail_loss_total": sum(
                _safe_int(item.get("tail_loss_count")) for item in specialist_tail_risk_models
            ),
            "okx_daily_status": okx_daily.get("status"),
            "okx_daily_unresolved": okx_unresolved,
            "specialist_eligible_shadow_count": specialist.get("eligible_shadow_count"),
            "rebuild_preflight_status": rebuild.get("status"),
        },
        "report_paths": {
            "go_no_go": go_report.get("report_path"),
            "paper_observation": observation.get("report_path"),
            "specialist_shadow": specialist.get("report_path"),
            "rebuild_preflight": rebuild.get("report_path"),
            "okx_daily": okx_daily.get("report_path"),
        },
        "operator_sequence": [
            "Start paper only through the confirmed preflight entrypoint.",
            "After start, let observation timers verify OKX native facts and shadow sample accumulation.",
            "Only if observation is healthy and promotion says canary_ready=true may an operator review canary.",
            "Live remains disabled from this handoff report.",
        ],
    }


@dataclass(slots=True)
class Phase3StageHandoffService:
    """Collect Phase 3 stage evidence without mutating runtime state."""

    root: Path | None = None
    report_max_age_seconds: int = DEFAULT_REPORT_MAX_AGE_SECONDS

    def _root(self) -> Path:
        return self.root or Path.cwd()

    def _data_dir(self) -> Path:
        return settings.data_dir

    def report(self) -> dict[str, Any]:
        root = self._root()
        data_dir = self._data_dir()
        go_no_go = _first_report(
            [
                data_dir / GO_NO_GO_REL,
                root / "data" / GO_NO_GO_REL,
            ]
        )
        observation = _first_report(
            [
                data_dir / OBSERVATION_REL,
                root / "data" / OBSERVATION_REL,
            ]
        )
        specialist = _first_report(
            [
                data_dir / SPECIALIST_DATA_REL,
                root / SPECIALIST_REPORT_REL,
            ]
        )
        rebuild = _first_report(
            [
                data_dir / REBUILD_REL,
                root / "data" / REBUILD_REL,
            ]
        )
        okx_daily = _first_report(
            [
                data_dir / OKX_DAILY_REL,
                root / "data" / OKX_DAILY_REL,
            ]
        )
        return evaluate_phase3_stage_handoff_inputs(
            go_no_go_report=go_no_go,
            observation_report=observation,
            specialist_shadow_report=specialist,
            rebuild_preflight_report=rebuild,
            okx_daily_report=okx_daily,
            report_max_age_seconds=self.report_max_age_seconds,
        )
