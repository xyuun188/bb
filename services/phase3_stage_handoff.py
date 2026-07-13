"""Read-only Phase 3 dynamic-return readiness observation."""

from __future__ import annotations

import json
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


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            result = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _age_seconds(report: dict[str, Any]) -> float | None:
    created = _parse_time(
        report.get("checked_at")
        or report.get("generated_at")
        or report.get("created_at")
        or report.get("timestamp")
    )
    return max((_now() - created).total_seconds(), 0.0) if created else None


def _is_fresh(report: dict[str, Any], max_age_seconds: int) -> bool:
    age = _age_seconds(report)
    return age is not None and age <= max_age_seconds


def _blocker(code: str, message: str, evidence: Any = None) -> dict[str, Any]:
    item = {"code": code, "severity": "blocking", "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _warning(code: str, message: str, evidence: Any = None) -> dict[str, Any]:
    item = {"code": code, "severity": "warning", "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(value, dict):
        return {}
    value.setdefault("available", True)
    value.setdefault("report_path", str(path))
    return value


def _first_report(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        report = _read_json(path)
        if report:
            return report
    return {"available": False, "candidate_paths": [str(path) for path in paths]}


def evaluate_phase3_stage_handoff_inputs(
    *,
    go_no_go_report: dict[str, Any],
    observation_report: dict[str, Any],
    specialist_shadow_report: dict[str, Any],
    rebuild_preflight_report: dict[str, Any],
    okx_daily_report: dict[str, Any],
    report_max_age_seconds: int = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Observe readiness without granting paper, canary, live, or model permissions."""

    go_report = _safe_dict(go_no_go_report)
    gate = _safe_dict(go_report.get("go_no_go")) or go_report
    observation = _safe_dict(observation_report)
    specialist = _safe_dict(specialist_shadow_report)
    rebuild = _safe_dict(rebuild_preflight_report)
    okx_daily = _safe_dict(okx_daily_report)
    max_age = max(int(report_max_age_seconds or DEFAULT_REPORT_MAX_AGE_SECONDS), 60)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    passed: list[str] = []

    if not go_report.get("available"):
        blockers.append(_blocker("go_no_go_report_missing", "Dynamic return gate report is missing."))
    elif not _is_fresh(go_report, max_age):
        blockers.append(
            _blocker(
                "go_no_go_report_stale",
                "Dynamic return gate report is stale.",
                {"age_seconds": _age_seconds(go_report), "max_age_seconds": max_age},
            )
        )
    else:
        passed.append("go_no_go_report_fresh")

    gate_blockers = _safe_list(gate.get("blockers"))
    gate_ready = bool(gate.get("ready")) and str(gate.get("status") or "") == "go"
    if gate_blockers or not gate_ready:
        blockers.append(
            _blocker(
                "dynamic_return_gate_not_ready",
                "Current fee-after return architecture has unresolved blockers.",
                gate_blockers or {"status": gate.get("status"), "ready": gate.get("ready")},
            )
        )
    else:
        passed.append("dynamic_return_gate_ready")

    unresolved = int(_safe_dict(okx_daily.get("issue_ledger_summary")).get("unresolved") or 0)
    if okx_daily.get("available") and unresolved > 0:
        blockers.append(
            _blocker(
                "okx_trade_facts_unresolved",
                "OKX authoritative trade facts still have unresolved issues.",
                _safe_dict(okx_daily.get("issue_ledger_summary")),
            )
        )
    elif okx_daily.get("available"):
        passed.append("okx_trade_facts_available")
    else:
        warnings.append(_warning("okx_report_missing", "OKX reconciliation report is missing."))

    if specialist.get("available") and (
        specialist.get("live_mutation") is True
        or specialist.get("production_permission") is True
    ):
        blockers.append(
            _blocker(
                "specialist_observation_boundary_violated",
                "Specialist shadow output attempted to claim production permission.",
            )
        )
    elif specialist.get("available"):
        passed.append("specialist_shadow_observation_only")
    else:
        warnings.append(_warning("specialist_report_missing", "Specialist observation report is missing."))

    if not observation.get("available"):
        warnings.append(_warning("observation_report_missing", "Paper observation report is missing."))
    if not rebuild.get("available"):
        warnings.append(_warning("rebuild_report_missing", "Rebuild report is missing."))

    ready = not blockers
    return {
        "checked_at": _now().isoformat(),
        "status": "dynamic_return_ready" if ready else "blocked",
        "stage": "observation_only",
        "ready": ready,
        "audit_only": True,
        "read_only": True,
        "production_permission": False,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "changes_strategy_weight": False,
        "changes_position_size": False,
        "changes_leverage": False,
        "changes_model_promotion": False,
        "live_mutation": False,
        "blockers": blockers,
        "warnings": warnings,
        "passed_checks": passed,
        "summary": {
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "dynamic_return_gate_ready": gate_ready,
            "okx_unresolved_count": unresolved,
        },
        "inputs": {
            "go_no_go": {"available": bool(go_report.get("available")), "status": gate.get("status")},
            "observation": {"available": bool(observation.get("available"))},
            "specialist": {"available": bool(specialist.get("available"))},
            "rebuild": {"available": bool(rebuild.get("available"))},
            "okx_daily": {"available": bool(okx_daily.get("available"))},
        },
        "policy": {
            "optimization_target": "realized_fee_after_return",
            "dynamic_return_gate_is_authoritative": True,
            "expert_memory_shadow_strategy_learning_are_observation_only": True,
        },
    }


class Phase3StageHandoffService:
    def __init__(
        self,
        *,
        data_dir: Path | None = None,
        report_max_age_seconds: int = DEFAULT_REPORT_MAX_AGE_SECONDS,
    ) -> None:
        self.data_dir = data_dir or settings.data_dir
        self.report_max_age_seconds = max(int(report_max_age_seconds or 60), 60)

    def report(self) -> dict[str, Any]:
        root = Path(self.data_dir)
        go_report = _first_report([root / GO_NO_GO_REL])
        observation = _first_report([root / OBSERVATION_REL])
        specialist = _first_report(
            [root / SPECIALIST_DATA_REL, Path.cwd() / SPECIALIST_REPORT_REL]
        )
        rebuild = _first_report([root / REBUILD_REL])
        okx_daily = _first_report([root / OKX_DAILY_REL])
        report = evaluate_phase3_stage_handoff_inputs(
            go_no_go_report=go_report,
            observation_report=observation,
            specialist_shadow_report=specialist,
            rebuild_preflight_report=rebuild,
            okx_daily_report=okx_daily,
            report_max_age_seconds=self.report_max_age_seconds,
        )
        report["report_paths"] = {
            "go_no_go": go_report.get("report_path"),
            "observation": observation.get("report_path"),
            "specialist": specialist.get("report_path"),
            "rebuild": rebuild.get("report_path"),
            "okx_daily": okx_daily.get("report_path"),
        }
        return report
