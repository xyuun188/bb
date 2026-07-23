"""Read-only observation window after Phase 3 paper trading resumes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import settings
from core.safe_output import safe_error_text
from services.okx_authoritative_sync import OkxAuthoritativeSyncService
from services.server_monitor_status import (
    collect_platform_runtime_status,
    collect_platform_server_status,
)

DEFAULT_OBSERVATION_HOURS = 2
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
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


def _load_trading_runtime_status() -> dict[str, Any]:
    """Read the split-process trading heartbeat without touching the engine."""

    path = settings.data_dir / "trading_runtime_status.json"
    try:
        if not path.exists():
            return {"available": False, "reason": "missing_runtime_heartbeat"}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"available": False, "reason": "invalid_runtime_heartbeat"}
        heartbeat_at = _parse_utc_datetime(payload.get("heartbeat_at"))
        if heartbeat_at is not None:
            payload["heartbeat_age_seconds"] = round(_age_seconds(heartbeat_at) or 0.0, 3)
        else:
            payload["heartbeat_age_seconds"] = round(
                max(_now().timestamp() - path.stat().st_mtime, 0.0),
                3,
            )
        payload["available"] = True
        return payload
    except Exception as exc:
        return {
            "available": False,
            "reason": "runtime_heartbeat_read_failed",
            "error": safe_error_text(exc, limit=180),
        }


def _runtime_okx_sync_clean_for_entry(trading_runtime: dict[str, Any]) -> bool:
    """Return true when live runtime has recent clean OKX current-state evidence."""

    runtime = _safe_dict(trading_runtime)
    if not runtime.get("available") or not bool(runtime.get("running")):
        return False
    decision_interval = _safe_float(runtime.get("decision_interval"), 60.0)
    heartbeat_fresh_limit = max(decision_interval * 4.0, 180.0)
    heartbeat_age = runtime.get("heartbeat_age_seconds")
    if (
        heartbeat_age is None
        or _safe_float(heartbeat_age, heartbeat_fresh_limit + 1.0) > heartbeat_fresh_limit
    ):
        return False

    sync = _safe_dict(runtime.get("okx_authoritative_sync"))
    status = str(sync.get("status") or "").lower()
    if status not in {"ok", "degraded"}:
        return False
    if _safe_int(sync.get("last_requires_attention_count")) > 0:
        return False

    success_age = sync.get("last_success_age_seconds")
    stale_after = _safe_float(
        sync.get("stale_after_seconds"),
        max(decision_interval * 3.0, 180.0),
    )
    if isinstance(success_age, (int, float)):
        return float(success_age) <= stale_after
    success_at = _parse_utc_datetime(sync.get("last_success_at"))
    if success_at is None:
        return bool(sync.get("fresh_success_available"))
    return (_now() - success_at).total_seconds() <= stale_after


def evaluate_phase3_paper_resume_observation_inputs(
    *,
    sample_summary: dict[str, Any],
    platform_server: dict[str, Any],
    platform_runtime: dict[str, Any],
    okx_authoritative_sync: dict[str, Any],
    trading_runtime_status: dict[str, Any] | None = None,
    specialist_shadow_evaluation: dict[str, Any],
    latest_preflight: dict[str, Any],
    report_max_age_seconds: int = DEFAULT_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Evaluate post-resume observation inputs without mutating runtime state."""

    samples = _safe_dict(sample_summary)
    platform = _safe_dict(platform_server)
    runtime = _safe_dict(platform_runtime)
    okx_sync = _safe_dict(okx_authoritative_sync)
    trading_runtime = _safe_dict(trading_runtime_status)
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

    okx_issue_count = _safe_int(okx_sync.get("issue_count"))
    okx_audit_unhealthy = okx_sync.get("okx_pull_available") is False or str(
        okx_sync.get("status") or ""
    ).lower() in {
        "critical",
        "error",
        "unavailable",
    }
    runtime_okx_clean = _runtime_okx_sync_clean_for_entry(trading_runtime)
    if okx_issue_count > 0:
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
    elif okx_audit_unhealthy and runtime_okx_clean:
        runtime_sync = _safe_dict(trading_runtime.get("okx_authoritative_sync"))
        warnings.append(
            _warning(
                "okx_authoritative_audit_pull_degraded_runtime_clean",
                (
                    "Read-only OKX audit pull is degraded, but the live trading runtime has "
                    "recent clean OKX current-state evidence."
                ),
                evidence={
                    "audit_status": okx_sync.get("status"),
                    "okx_pull_available": okx_sync.get("okx_pull_available"),
                    "fetch_errors": okx_sync.get("fetch_errors"),
                    "error": okx_sync.get("error"),
                    "runtime_okx_status": runtime_sync.get("status"),
                    "runtime_last_success_at": runtime_sync.get("last_success_at"),
                },
            )
        )
        passed.append("okx_runtime_authoritative_sync_clean_after_resume")
    elif okx_audit_unhealthy:
        blockers.append(
            _blocker(
                "okx_authoritative_sync_unhealthy_after_resume",
                "OKX native facts must remain clean during the post-resume observation window.",
                evidence={
                    "status": okx_sync.get("status"),
                    "fetch_errors": okx_sync.get("fetch_errors"),
                    "error": okx_sync.get("error"),
                    "runtime_okx_sync_clean": runtime_okx_clean,
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

    if paper_active:
        passed.append("post_resume_shadow_counts_are_diagnostic_only")

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
            "runtime_okx_sync_clean": runtime_okx_clean,
            "specialist_report_age_seconds": (
                None if specialist_age is None else round(specialist_age, 3)
            ),
            "preflight_consumed_after_resume": preflight_consumed_after_resume,
        },
        "sample_summary": samples,
        "inputs": {
            "platform_server": platform,
            "platform_runtime": runtime,
            "trading_runtime_status": trading_runtime,
            "okx_authoritative_sync": okx_sync,
            "specialist_shadow_evaluation": specialist,
            "latest_preflight": preflight,
        },
        "operator_sequence": [
            "If status is waiting_for_resume, keep the report as the baseline before paper is started.",
            "After paper starts, watch the first 30/60/120 minutes for OKX clean state and sample accumulation.",
            "Use cost-complete fee-after return distributions, not fixed sample floors, for canary/live promotion.",
        ],
        "checked_at": _now_iso(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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
    trading_runtime_provider: ReportProvider | None = None
    okx_sync_provider: ReportProvider | None = None
    specialist_shadow_provider: ReportProvider | None = None
    latest_preflight_provider: ReportProvider | None = None
    observation_hours: int = DEFAULT_OBSERVATION_HOURS
    report_max_age_seconds: int = DEFAULT_REPORT_MAX_AGE_SECONDS
    sample_limit: int = DEFAULT_SAMPLE_LIMIT

    async def report(self) -> dict[str, Any]:
        (
            sample_summary,
            platform_server,
            platform_runtime,
            trading_runtime,
            specialist,
            preflight,
        ) = await asyncio.gather(
            self._collect(
                "sample_summary",
                self.sample_summary_provider,
                self._default_sample_summary,
            ),
            self._collect(
                "platform_server",
                self.platform_server_provider,
                collect_platform_server_status,
            ),
            self._collect(
                "platform_runtime",
                self.platform_runtime_provider,
                collect_platform_runtime_status,
            ),
            self._collect(
                "trading_runtime_status",
                self.trading_runtime_provider,
                _load_trading_runtime_status,
            ),
            self._collect(
                "specialist_shadow_evaluation",
                self.specialist_shadow_provider,
                self._default_specialist_shadow,
            ),
            self._collect(
                "latest_preflight",
                self.latest_preflight_provider,
                self._default_latest_preflight,
            ),
        )
        okx_sync = await self._collect(
            "okx_authoritative_sync",
            self.okx_sync_provider,
            self._default_okx_sync,
        )
        return evaluate_phase3_paper_resume_observation_inputs(
            sample_summary=sample_summary,
            platform_server=platform_server,
            platform_runtime=platform_runtime,
            okx_authoritative_sync=okx_sync,
            trading_runtime_status=trading_runtime,
            specialist_shadow_evaluation=specialist,
            latest_preflight=preflight,
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
            "created_shadow_count": created_shadow_count,
            "completed_shadow_count": completed_shadow_count,
            "specialist_shadow_sample_count": specialist_shadow_sample_count,
            "created_order_count": created_order_count,
            "filled_order_count": filled_order_count,
            "open_position_count": open_position_count,
            "trade_reflection_count": trade_reflection_count,
            "sample_limit": max(1, min(int(self.sample_limit or DEFAULT_SAMPLE_LIMIT), 5000)),
        }
