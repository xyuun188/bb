"""Read-only observation window after Phase 3 paper trading resumes."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import settings
from core.safe_output import safe_error_text
from services.okx_authoritative_sync import (
    DEFAULT_COLD_START_MARKER_PATH,
    OkxAuthoritativeSyncService,
)
from services.server_monitor_status import (
    collect_platform_runtime_status,
    collect_platform_server_status,
)

DEFAULT_OBSERVATION_HOURS = 2
DEFAULT_MIN_CREATED_SHADOW_SAMPLES = 5
DEFAULT_MIN_COMPLETED_SHADOW_SAMPLES = 1
DEFAULT_REPORT_MAX_AGE_SECONDS = 2 * 3600
DEFAULT_SAMPLE_LIMIT = 800

ReportProvider = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]


def _now() -> datetime:
    return datetime.now(UTC)


def _now_iso() -> str:
    return _now().isoformat()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _blocker_codes(blockers: list[Any]) -> set[str]:
    codes: set[str] = set()
    for item in blockers:
        if isinstance(item, dict):
            code = str(item.get("code") or "").strip()
        else:
            code = str(item or "").strip()
        if code:
            codes.add(code)
    return codes


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


def _age_seconds(value: Any) -> float | None:
    parsed = _parse_utc_datetime(value)
    if parsed is None:
        return None
    return max((_now() - parsed).total_seconds(), 0.0)


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


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def _service_by_name(platform_server: dict[str, Any], name: str) -> dict[str, Any]:
    for item in _safe_list(platform_server.get("services")):
        if isinstance(item, dict) and str(item.get("name") or "") == name:
            return item
    return {}


def _report_age_ok(report: dict[str, Any], max_age_seconds: int) -> tuple[bool, float | None]:
    age = _age_seconds(report.get("generated_at") or report.get("checked_at"))
    return age is not None and age <= max_age_seconds, age


def evaluate_phase3_paper_resume_observation_inputs(
    *,
    sample_summary: dict[str, Any],
    platform_server: dict[str, Any],
    platform_runtime: dict[str, Any],
    okx_authoritative_sync: dict[str, Any],
    specialist_shadow_evaluation: dict[str, Any],
    latest_preflight: dict[str, Any],
    min_created_shadow_samples: int = DEFAULT_MIN_CREATED_SHADOW_SAMPLES,
    min_completed_shadow_samples: int = DEFAULT_MIN_COMPLETED_SHADOW_SAMPLES,
    report_max_age_seconds: int = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Evaluate post-resume observation inputs without mutating runtime state."""

    samples = _safe_dict(sample_summary)
    platform = _safe_dict(platform_server)
    runtime = _safe_dict(platform_runtime)
    okx_sync = _safe_dict(okx_authoritative_sync)
    specialist = _safe_dict(specialist_shadow_evaluation)
    preflight = _safe_dict(latest_preflight)

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    passed: list[str] = []

    paper_service = _service_by_name(platform, "bb-paper-trading.service")
    paper_active = bool(paper_service.get("active"))
    local_tools = _safe_dict(runtime.get("local_ai_tools"))
    child_endpoints = _safe_dict(local_tools.get("child_endpoints"))
    created_shadow_count = _safe_int(samples.get("created_shadow_count"))
    completed_shadow_count = _safe_int(samples.get("completed_shadow_count"))
    specialist_eligible_count = _safe_int(specialist.get("eligible_shadow_count"))

    if not paper_active:
        warnings.append(
            _warning(
                "paper_trading_not_active",
                "Post-resume observation is waiting because paper trading is not active.",
                evidence=paper_service,
            )
        )
    else:
        passed.append("paper_trading_active_for_observation")

    preflight_consumed_after_resume = (
        bool(preflight)
        and preflight.get("can_resume_paper") is not True
        and paper_active
        and _blocker_codes(_safe_list(preflight.get("blockers"))) == {"paper_trading_already_active"}
    )
    if preflight_consumed_after_resume:
        passed.append("latest_preflight_consumed_after_resume")
    elif bool(preflight) and preflight.get("can_resume_paper") is not True:
        warnings.append(
            _warning(
                "latest_preflight_not_ready",
                "Latest paper-resume preflight is not ready; do not use this observation to justify resume.",
                evidence={
                    "status": preflight.get("status"),
                    "can_resume_paper": preflight.get("can_resume_paper"),
                },
            )
        )
    elif bool(preflight):
        passed.append("latest_preflight_ready")

    if okx_sync.get("okx_pull_available") is False or str(okx_sync.get("status") or "").lower() in {
        "critical",
        "error",
        "unavailable",
    }:
        blockers.append(
            _blocker(
                "okx_authoritative_sync_unhealthy_after_resume",
                "OKX native facts must remain clean during the post-resume observation window.",
                evidence={
                    "status": okx_sync.get("status"),
                    "fetch_errors": okx_sync.get("fetch_errors"),
                    "error": okx_sync.get("error"),
                },
            )
        )
    elif _safe_int(okx_sync.get("issue_count")) > 0:
        blockers.append(
            _blocker(
                "okx_authoritative_sync_has_post_resume_differences",
                "OKX/local differences appeared during the post-resume observation window.",
                evidence={
                    "issue_count": okx_sync.get("issue_count"),
                    "issues": _safe_list(okx_sync.get("issues"))[:8],
                },
            )
        )
    else:
        passed.append("okx_authoritative_sync_clean_after_resume")

    if not bool(local_tools.get("available")):
        blockers.append(
            _blocker(
                "phase3_quant_api_unavailable_after_resume",
                "Phase 3 quant API must remain reachable during paper observation.",
                evidence=local_tools,
            )
        )
    else:
        required_children = {
            "profit_prediction",
            "time_series_prediction",
            "sentiment_analysis",
            "exit_advice",
        }
        missing = [
            name
            for name in sorted(required_children)
            if not bool(_safe_dict(child_endpoints.get(name)).get("available"))
        ]
        if missing:
            blockers.append(
                _blocker(
                    "phase3_quant_api_child_endpoint_unhealthy_after_resume",
                    "All Phase 3 quant API child endpoints must stay healthy after resume.",
                    evidence={"missing": missing, "child_endpoints": child_endpoints},
                )
            )
        else:
            passed.append("phase3_quant_api_endpoints_healthy_after_resume")

    specialist_fresh, specialist_age = _report_age_ok(specialist, report_max_age_seconds)
    if not bool(specialist.get("available", bool(specialist))):
        warnings.append(
            _warning(
                "specialist_shadow_evaluation_missing_after_resume",
                "Specialist shadow evaluation report is missing.",
                evidence=specialist,
            )
        )
    elif not specialist_fresh:
        warnings.append(
            _warning(
                "specialist_shadow_evaluation_stale_after_resume",
                "Specialist shadow evaluation report is stale.",
                evidence={
                    "generated_at": specialist.get("generated_at"),
                    "checked_at": specialist.get("checked_at"),
                    "age_seconds": specialist_age,
                    "max_age_seconds": report_max_age_seconds,
                },
            )
        )
    elif bool(specialist.get("live_mutation")):
        blockers.append(
            _blocker(
                "specialist_shadow_live_mutation_after_resume",
                "Specialist shadow evaluation must remain shadow-only.",
                evidence={"live_mutation": specialist.get("live_mutation")},
            )
        )
    else:
        passed.append("specialist_shadow_evaluation_fresh_after_resume")

    if paper_active and created_shadow_count < min_created_shadow_samples:
        warnings.append(
            _warning(
                "post_resume_shadow_sample_floor_not_met",
                "Paper is active but new shadow samples have not reached the observation floor yet.",
                evidence={
                    "created_shadow_count": created_shadow_count,
                    "minimum": min_created_shadow_samples,
                },
            )
        )
    elif paper_active:
        passed.append("post_resume_shadow_samples_accumulating")

    if paper_active and completed_shadow_count < min_completed_shadow_samples:
        warnings.append(
            _warning(
                "post_resume_completed_shadow_floor_not_met",
                "Completed shadow outcomes are still warming up.",
                evidence={
                    "completed_shadow_count": completed_shadow_count,
                    "minimum": min_completed_shadow_samples,
                },
            )
        )
    elif paper_active:
        passed.append("post_resume_completed_shadow_samples_available")

    if paper_active and specialist_eligible_count <= 0:
        warnings.append(
            _warning(
                "specialist_shadow_evidence_not_accumulated_yet",
                "Specialist shadow evidence has not accumulated eligible completed samples yet.",
                evidence={"eligible_shadow_count": specialist_eligible_count},
            )
        )
    elif paper_active:
        passed.append("specialist_shadow_evidence_accumulating")

    if blockers:
        status = "critical"
    elif not paper_active:
        status = "waiting_for_resume"
    elif warnings:
        status = "warming_up"
    else:
        status = "healthy"

    return {
        "status": status,
        "read_only": True,
        "audit_only": True,
        "mutates_database": False,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "live_mutation": False,
        "paper_active": paper_active,
        "can_use_for_promotion": status == "healthy",
        "blockers": blockers,
        "warnings": warnings,
        "passed_checks": list(dict.fromkeys(passed)),
        "summary": {
            "created_shadow_count": created_shadow_count,
            "completed_shadow_count": completed_shadow_count,
            "created_order_count": _safe_int(samples.get("created_order_count")),
            "filled_order_count": _safe_int(samples.get("filled_order_count")),
            "open_position_count": _safe_int(samples.get("open_position_count")),
            "trade_reflection_count": _safe_int(samples.get("trade_reflection_count")),
            "specialist_eligible_shadow_count": specialist_eligible_count,
            "phase3_quant_child_endpoint_count": len(child_endpoints),
            "okx_issue_count": _safe_int(okx_sync.get("issue_count")),
            "specialist_report_age_seconds": (
                None if specialist_age is None else round(specialist_age, 3)
            ),
            "preflight_consumed_after_resume": preflight_consumed_after_resume,
        },
        "sample_summary": samples,
        "inputs": {
            "platform_server": platform,
            "platform_runtime": runtime,
            "okx_authoritative_sync": okx_sync,
            "specialist_shadow_evaluation": specialist,
            "latest_preflight": preflight,
        },
        "operator_sequence": [
            "If status is waiting_for_resume, keep the report as the baseline before paper is started.",
            "After paper starts, watch the first 30/60/120 minutes for OKX clean state and sample accumulation.",
            "Do not use specialist models for canary/live promotion until this observation is healthy and sample floors pass.",
        ],
        "checked_at": _now_iso(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_cold_start_reset_at(path: Path | None = DEFAULT_COLD_START_MARKER_PATH) -> datetime | None:
    if path is None or not Path(path).exists():
        return None
    payload = _read_json(Path(path))
    return _parse_utc_datetime(payload.get("reset_at"))


def _latest_report_from_candidates(paths: list[Path]) -> dict[str, Any]:
    for path in paths:
        payload = _read_json(path)
        if payload:
            payload.setdefault("available", True)
            payload.setdefault("report_path", str(path))
            return payload
    return {"available": False, "candidate_paths": [str(path) for path in paths]}


@dataclass(slots=True)
class Phase3PaperResumeObservationService:
    """Collect post-resume observation evidence without changing runtime state."""

    sample_summary_provider: ReportProvider | None = None
    platform_server_provider: ReportProvider | None = None
    platform_runtime_provider: ReportProvider | None = None
    okx_sync_provider: ReportProvider | None = None
    specialist_shadow_provider: ReportProvider | None = None
    latest_preflight_provider: ReportProvider | None = None
    observation_hours: int = DEFAULT_OBSERVATION_HOURS
    min_created_shadow_samples: int = DEFAULT_MIN_CREATED_SHADOW_SAMPLES
    min_completed_shadow_samples: int = DEFAULT_MIN_COMPLETED_SHADOW_SAMPLES
    report_max_age_seconds: int = DEFAULT_REPORT_MAX_AGE_SECONDS
    sample_limit: int = DEFAULT_SAMPLE_LIMIT

    async def report(self) -> dict[str, Any]:
        sample_summary = await self._collect(
            "sample_summary",
            self.sample_summary_provider,
            self._default_sample_summary,
        )
        platform_server = await self._collect(
            "platform_server",
            self.platform_server_provider,
            collect_platform_server_status,
        )
        platform_runtime = await self._collect(
            "platform_runtime",
            self.platform_runtime_provider,
            collect_platform_runtime_status,
        )
        okx_sync = await self._collect(
            "okx_authoritative_sync",
            self.okx_sync_provider,
            self._default_okx_sync,
        )
        specialist = await self._collect(
            "specialist_shadow_evaluation",
            self.specialist_shadow_provider,
            self._default_specialist_shadow,
        )
        preflight = await self._collect(
            "latest_preflight",
            self.latest_preflight_provider,
            self._default_latest_preflight,
        )
        return evaluate_phase3_paper_resume_observation_inputs(
            sample_summary=sample_summary,
            platform_server=platform_server,
            platform_runtime=platform_runtime,
            okx_authoritative_sync=okx_sync,
            specialist_shadow_evaluation=specialist,
            latest_preflight=preflight,
            min_created_shadow_samples=self.min_created_shadow_samples,
            min_completed_shadow_samples=self.min_completed_shadow_samples,
            report_max_age_seconds=self.report_max_age_seconds,
        )

    async def _collect(
        self,
        name: str,
        provider: ReportProvider | None,
        default_provider: ReportProvider,
    ) -> dict[str, Any]:
        try:
            result = await _maybe_await((provider or default_provider)())
            if isinstance(result, dict):
                return result
            return {
                "status": "unavailable",
                "read_only": True,
                "audit_only": True,
                "source": name,
                "error": f"{name} returned {type(result).__name__}",
            }
        except Exception as exc:
            return {
                "status": "unavailable",
                "read_only": True,
                "audit_only": True,
                "source": name,
                "error": safe_error_text(exc, limit=180),
            }

    async def _default_okx_sync(self) -> dict[str, Any]:
        return await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=max(int(self.observation_hours or 1), 1),
            limit=120,
            timeout_seconds=5.0,
        ).collect()

    async def _default_specialist_shadow(self) -> dict[str, Any]:
        return _latest_report_from_candidates(
            [
                settings.data_dir / "phase3" / "specialist_shadow_evaluation_latest.json",
                Path.cwd() / "reports" / "phase3" / "specialist_shadow_evaluation_latest.json",
            ]
        )

    async def _default_latest_preflight(self) -> dict[str, Any]:
        return _latest_report_from_candidates(
            [
                settings.data_dir / "phase3_paper_resume_preflight_reports" / "latest.json",
                Path.cwd() / "data" / "phase3_paper_resume_preflight_reports" / "latest.json",
            ]
        )

    async def _default_sample_summary(self) -> dict[str, Any]:
        from sqlalchemy import func, select

        from db.session import get_read_session_ctx
        from models.learning import ShadowBacktest, TradeReflection
        from models.trade import Order, Position

        window_start = _now() - timedelta(hours=max(int(self.observation_hours or 1), 1))
        cold_start = _load_cold_start_reset_at()
        if cold_start is not None and cold_start > window_start:
            window_start = cold_start
        since_naive = window_start.replace(tzinfo=None)
        async with get_read_session_ctx() as session:
            created_shadow_count = _safe_int(
                (
                    await session.execute(
                        select(func.count(ShadowBacktest.id)).where(
                            ShadowBacktest.execution_mode == "paper",
                            ShadowBacktest.created_at >= since_naive,
                        )
                    )
                ).scalar_one()
            )
            completed_shadow_count = _safe_int(
                (
                    await session.execute(
                        select(func.count(ShadowBacktest.id)).where(
                            ShadowBacktest.execution_mode == "paper",
                            ShadowBacktest.status == "completed",
                            ShadowBacktest.created_at >= since_naive,
                            ShadowBacktest.long_return_pct.is_not(None),
                            ShadowBacktest.short_return_pct.is_not(None),
                        )
                    )
                ).scalar_one()
            )
            created_order_count = _safe_int(
                (
                    await session.execute(
                        select(func.count(Order.id)).where(
                            Order.execution_mode == "paper",
                            Order.created_at >= since_naive,
                        )
                    )
                ).scalar_one()
            )
            filled_order_count = _safe_int(
                (
                    await session.execute(
                        select(func.count(Order.id)).where(
                            Order.execution_mode == "paper",
                            Order.status == "filled",
                            Order.created_at >= since_naive,
                        )
                    )
                ).scalar_one()
            )
            open_position_count = _safe_int(
                (
                    await session.execute(
                        select(func.count(Position.id)).where(
                            Position.execution_mode == "paper",
                            Position.is_open.is_(True),
                        )
                    )
                ).scalar_one()
            )
            trade_reflection_count = _safe_int(
                (
                    await session.execute(
                        select(func.count(TradeReflection.id)).where(
                            TradeReflection.execution_mode == "paper",
                            TradeReflection.created_at >= since_naive,
                        )
                    )
                ).scalar_one()
            )
            recent_rows = list(
                (
                    await session.execute(
                        select(ShadowBacktest)
                        .where(
                            ShadowBacktest.execution_mode == "paper",
                            ShadowBacktest.status == "completed",
                            ShadowBacktest.created_at >= since_naive,
                        )
                        .order_by(ShadowBacktest.id.desc())
                        .limit(max(1, min(int(self.sample_limit or DEFAULT_SAMPLE_LIMIT), 5000)))
                    )
                )
                .scalars()
                .all()
            )
        specialist_shadow_sample_count = 0
        for row in recent_rows:
            snapshot = row.feature_snapshot if isinstance(row.feature_snapshot, dict) else {}
            local_shadow = snapshot.get("local_ai_tools_shadow")
            if isinstance(local_shadow, dict) and local_shadow:
                specialist_shadow_sample_count += 1
        return {
            "status": "ok",
            "read_only": True,
            "audit_only": True,
            "window_start": window_start.isoformat(),
            "observation_hours": max(int(self.observation_hours or 1), 1),
            "cold_start_reset_at": cold_start.isoformat() if cold_start else None,
            "created_shadow_count": created_shadow_count,
            "completed_shadow_count": completed_shadow_count,
            "specialist_shadow_sample_count": specialist_shadow_sample_count,
            "created_order_count": created_order_count,
            "filled_order_count": filled_order_count,
            "open_position_count": open_position_count,
            "trade_reflection_count": trade_reflection_count,
            "sample_limit": max(1, min(int(self.sample_limit or DEFAULT_SAMPLE_LIMIT), 5000)),
        }
