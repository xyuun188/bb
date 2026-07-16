"""Read-only hard gate before Phase 3 paper trading can be resumed."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.phase3_model_contract import PHASE3_REQUIRED_LLM_MODEL_IDS
from core.safe_output import safe_error_text
from executor.okx_executor import OKXExecutor
from services.okx_authoritative_sync import OkxAuthoritativeSyncService
from services.okx_trade_fact_integrity import OkxTradeFactIntegrityService
from services.phase3_model_server_readiness import Phase3ModelServerReadinessAuditService
from services.server_monitor_status import collect_platform_runtime_status

DEFAULT_OKX_LOOKBACK_HOURS = 24
DEFAULT_OKX_LIMIT = 120
DEFAULT_OKX_TIMEOUT_SECONDS = 5.0
DEFAULT_SPECIALIST_REPORT_MAX_AGE_SECONDS = 2 * 3600

ReportProvider = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    return max((datetime.now(UTC) - parsed).total_seconds(), 0.0)


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


def _platform_model_runtime_ready(runtime: dict[str, Any]) -> bool:
    rows = [
        item for item in _safe_list(runtime.get("ai_models")) if isinstance(item, dict)
    ]
    required = PHASE3_REQUIRED_LLM_MODEL_IDS
    available: set[str] = set()
    for row in rows:
        if not bool(row.get("available")):
            continue
        model = str(row.get("model") or "").strip().lower()
        if model:
            available.add(model)
        for served in _safe_list(row.get("models")):
            served_text = str(served or "").strip().lower()
            if served_text:
                available.add(served_text)
    return required.issubset(available)


def _account_equity_value(snapshot: dict[str, Any]) -> float:
    return max(
        _safe_float(snapshot.get("equity")),
        _safe_float(snapshot.get("total")),
        _safe_float(snapshot.get("cash")),
        _safe_float(snapshot.get("allocatable")),
        _safe_float(snapshot.get("free")),
    )


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def evaluate_phase3_paper_resume_preflight_inputs(
    *,
    okx_authoritative_sync: dict[str, Any],
    okx_trade_fact_integrity: dict[str, Any],
    model_server_readiness: dict[str, Any],
    platform_runtime: dict[str, Any],
    platform_server: dict[str, Any],
    specialist_shadow_evaluation: dict[str, Any],
    account_equity_truth: dict[str, Any],
    specialist_report_max_age_seconds: int = DEFAULT_SPECIALIST_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Evaluate read-only inputs into a single paper-resume go/no-go report."""

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    passed: list[str] = []

    okx_sync = _safe_dict(okx_authoritative_sync)
    okx_integrity = _safe_dict(okx_trade_fact_integrity)
    model_server = _safe_dict(model_server_readiness)
    runtime = _safe_dict(platform_runtime)
    platform = _safe_dict(platform_server)
    specialist = _safe_dict(specialist_shadow_evaluation)
    equity_truth = _safe_dict(account_equity_truth)

    if okx_sync.get("okx_pull_available") is False:
        blockers.append(
            _blocker(
                "okx_authoritative_pull_unavailable",
                "OKX native current-state pull must be available before paper trading resumes.",
                evidence=okx_sync.get("fetch_errors") or okx_sync.get("error"),
            )
        )
    elif str(okx_sync.get("status") or "").lower() in {"critical", "error", "unavailable"}:
        blockers.append(
            _blocker(
                "okx_authoritative_sync_not_clean",
                "OKX native facts are not clean enough to resume paper trading.",
                evidence={
                    "status": okx_sync.get("status"),
                    "issue_count": okx_sync.get("issue_count"),
                    "manual_review_count": okx_sync.get("manual_review_count"),
                },
            )
        )
    elif _safe_int(okx_sync.get("issue_count")) > 0:
        blockers.append(
            _blocker(
                "okx_authoritative_sync_has_differences",
                "OKX/local current facts still have unresolved differences.",
                evidence={
                    "issue_count": okx_sync.get("issue_count"),
                    "issues": _safe_list(okx_sync.get("issues"))[:8],
                },
            )
        )
    else:
        passed.append("okx_authoritative_sync_clean")

    if str(okx_integrity.get("status") or "").lower() == "critical":
        blockers.append(
            _blocker(
                "okx_trade_fact_integrity_critical",
                "Local order/position facts still contain critical OKX integrity issues.",
                evidence={
                    "critical_count": okx_integrity.get("critical_count"),
                    "issues": _safe_list(okx_integrity.get("issues"))[:8],
                },
            )
        )
    elif _safe_int(okx_integrity.get("critical_count")) > 0:
        blockers.append(
            _blocker(
                "okx_trade_fact_integrity_critical",
                "Critical OKX trade-fact issues must be cleared before resume.",
                evidence={"critical_count": okx_integrity.get("critical_count")},
            )
        )
    elif str(okx_integrity.get("status") or "").lower() == "warning":
        warnings.append(
            _warning(
                "okx_trade_fact_integrity_warning",
                "OKX trade-fact audit has warnings; paper may resume only if they are non-critical.",
                evidence={
                    "warning_count": okx_integrity.get("warning_count"),
                    "kind_counts": okx_integrity.get("kind_counts"),
                },
            )
        )
    else:
        passed.append("okx_trade_fact_integrity_no_critical_issues")

    equity_value = _account_equity_value(equity_truth)
    if equity_truth.get("available") is False or not equity_truth:
        blockers.append(
            _blocker(
                "okx_account_equity_unavailable",
                "OKX account-equity snapshot must be readable before paper trading resumes.",
                evidence={
                    "source": equity_truth.get("source"),
                    "error": equity_truth.get("error"),
                },
            )
        )
    elif str(equity_truth.get("source") or "") != "okx_snapshot":
        blockers.append(
            _blocker(
                "okx_account_equity_not_exchange_truth",
                "Account equity must come from an OKX snapshot, not local estimates or virtual balances.",
                evidence={
                    "source": equity_truth.get("source"),
                    "balance_source": equity_truth.get("balance_source"),
                },
            )
        )
    elif bool(equity_truth.get("stale")):
        blockers.append(
            _blocker(
                "okx_account_equity_snapshot_stale",
                "OKX account-equity snapshot is stale and cannot unlock paper resume.",
                evidence={
                    "stale_age_seconds": equity_truth.get("stale_age_seconds"),
                    "stale_reason": equity_truth.get("stale_reason"),
                },
            )
        )
    elif equity_value <= 0:
        blockers.append(
            _blocker(
                "okx_account_equity_non_positive",
                "OKX account-equity snapshot has no positive equity value.",
                evidence={
                    "equity": equity_truth.get("equity"),
                    "total": equity_truth.get("total"),
                    "cash": equity_truth.get("cash"),
                    "allocatable": equity_truth.get("allocatable"),
                    "free": equity_truth.get("free"),
                },
            )
        )
    else:
        passed.append("okx_account_equity_truth_available")

    platform_model_runtime_ready = _platform_model_runtime_ready(runtime)
    if not bool(model_server.get("runtime_ready")):
        if platform_model_runtime_ready:
            warnings.append(
                _warning(
                    "phase3_model_server_remote_audit_unverified",
                    "Remote model-server manifest audit is unavailable, but platform loopback model endpoints are healthy.",
                    evidence={
                        "status": model_server.get("status"),
                        "blockers": _safe_list(model_server.get("blockers"))[:8],
                    },
                )
            )
            passed.append("phase3_model_server_platform_endpoints_ready")
        else:
            blockers.append(
                _blocker(
                    "phase3_model_server_runtime_not_ready",
                    "Phase 3 quant model-server runtime must be ready before paper resumes.",
                    evidence={
                        "status": model_server.get("status"),
                        "artifact_ready": model_server.get("artifact_ready"),
                        "runtime_ready": model_server.get("runtime_ready"),
                        "blockers": _safe_list(model_server.get("blockers"))[:8],
                        "warnings": _safe_list(model_server.get("warnings"))[:8],
                    },
                )
            )
    elif bool(model_server.get("phase3_model_service_go_live_blocked")):
        blockers.append(
            _blocker(
                "phase3_model_server_go_live_blocked",
                "Phase 3 model server still reports service go-live blocked.",
                evidence={"status": model_server.get("status")},
            )
        )
    else:
        passed.append("phase3_model_server_runtime_ready")

    local_tools = _safe_dict(runtime.get("local_ai_tools"))
    child_endpoints = _safe_dict(local_tools.get("child_endpoints"))
    required_children = {
        "profit_prediction",
        "time_series_prediction",
        "sentiment_analysis",
        "exit_advice",
    }
    unavailable_children = [
        name
        for name in sorted(required_children)
        if not bool(_safe_dict(child_endpoints.get(name)).get("available"))
    ]
    if not bool(local_tools.get("available")):
        blockers.append(
            _blocker(
                "phase3_quant_api_unavailable",
                "Platform cannot reach the Phase 3 quant API through the approved tunnel.",
                evidence={
                    "api_base": local_tools.get("api_base"),
                    "health": local_tools.get("health"),
                    "tunnel_contract": local_tools.get("tunnel_contract"),
                },
            )
        )
    elif unavailable_children:
        blockers.append(
            _blocker(
                "phase3_quant_api_child_endpoints_unavailable",
                "All Phase 3 quant API child endpoints must be healthy before paper resumes.",
                evidence={
                    "unavailable": unavailable_children,
                    "child_endpoints": child_endpoints,
                },
            )
        )
    else:
        passed.append("phase3_quant_api_endpoints_ready")

    services = _safe_list(platform.get("services"))
    service_by_name = {
        str(item.get("name") or ""): _safe_dict(item)
        for item in services
        if isinstance(item, dict)
    }
    dashboard = service_by_name.get("bb-dashboard.service", {})
    tunnels = service_by_name.get("bb-model-tunnels.service", {})
    paper = service_by_name.get("bb-paper-trading.service", {})
    if platform.get("available") is False:
        blockers.append(
            _blocker(
                "platform_status_unavailable",
                "Platform service status must be readable before paper resumes.",
                evidence=platform.get("message") or platform.get("status"),
            )
        )
    if dashboard and not bool(dashboard.get("active")):
        blockers.append(
            _blocker(
                "dashboard_service_inactive",
                "Dashboard/API service must be active before paper resumes.",
                evidence=dashboard,
            )
        )
    elif dashboard:
        passed.append("dashboard_service_active")
    if tunnels and not bool(tunnels.get("active")):
        blockers.append(
            _blocker(
                "model_tunnel_service_inactive",
                "Model tunnel service must be active before paper resumes.",
                evidence=tunnels,
            )
        )
    elif tunnels:
        passed.append("model_tunnel_service_active")
    if bool(paper.get("active")):
        blockers.append(
            _blocker(
                "paper_trading_already_active",
                "Preflight must run while paper trading is stopped, before an operator starts it.",
                evidence=paper,
            )
        )
    elif paper:
        passed.append("paper_trading_stopped_before_resume")

    specialist_age = _age_seconds(
        specialist.get("generated_at") or specialist.get("checked_at")
    )
    if not bool(specialist.get("available", bool(specialist))):
        blockers.append(
            _blocker(
                "specialist_shadow_evaluation_missing",
                "Specialist shadow evaluation report must exist before paper resumes.",
                evidence={
                    "reason": specialist.get("reason"),
                    "candidate_paths": specialist.get("candidate_paths"),
                },
            )
        )
    elif specialist_age is None or specialist_age > specialist_report_max_age_seconds:
        blockers.append(
            _blocker(
                "specialist_shadow_evaluation_stale",
                "Specialist shadow evaluation report is missing a fresh timestamp.",
                evidence={
                    "generated_at": specialist.get("generated_at"),
                    "checked_at": specialist.get("checked_at"),
                    "age_seconds": specialist_age,
                    "max_age_seconds": specialist_report_max_age_seconds,
                },
            )
        )
    elif bool(specialist.get("live_mutation")):
        blockers.append(
            _blocker(
                "specialist_shadow_live_mutation_enabled",
                "Specialist evaluation must remain shadow-only before paper resumes.",
                evidence={"live_mutation": specialist.get("live_mutation")},
            )
        )
    else:
        passed.append("specialist_shadow_evaluation_fresh")

    status = "ready" if not blockers else "blocked"
    if status == "ready" and warnings:
        status = "ready_with_warnings"

    return {
        "status": status,
        "read_only": True,
        "audit_only": True,
        "mutates_database": False,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "live_mutation": False,
        "can_resume_paper": not blockers,
        "requires_operator_start": True,
        "target_service": "bb-paper-trading.service",
        "mode": "paper",
        "blockers": blockers,
        "warnings": warnings,
        "passed_checks": list(dict.fromkeys(passed)),
        "source_status": {
            "okx_authoritative_sync": okx_sync.get("status") or "unknown",
            "okx_trade_fact_integrity": okx_integrity.get("status") or "unknown",
            "model_server_readiness": model_server.get("status") or "unknown",
            "platform_runtime_local_ai_tools": local_tools.get("available"),
            "platform_server": platform.get("status") or "unknown",
            "specialist_shadow_evaluation": (
                "available" if bool(specialist.get("available", bool(specialist))) else "missing"
            ),
            "account_equity_truth": equity_truth.get("source") or "unknown",
        },
        "summary": {
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "okx_issue_count": _safe_int(okx_sync.get("issue_count")),
            "okx_integrity_critical_count": _safe_int(okx_integrity.get("critical_count")),
            "okx_account_equity": round(equity_value, 8) if equity_value > 0 else 0.0,
            "okx_account_equity_available": bool(equity_value > 0),
            "okx_account_equity_source": equity_truth.get("source") or "unknown",
            "model_server_runtime_ready": bool(model_server.get("runtime_ready")),
            "phase3_quant_api_available": bool(local_tools.get("available")),
            "phase3_quant_child_endpoint_count": len(child_endpoints),
            "specialist_shadow_report_age_seconds": (
                None if specialist_age is None else round(specialist_age, 3)
            ),
        },
        "inputs": {
            "okx_authoritative_sync": okx_sync,
            "okx_trade_fact_integrity": okx_integrity,
            "model_server_readiness": model_server,
            "platform_runtime": runtime,
            "platform_server": platform,
            "specialist_shadow_evaluation": specialist,
            "account_equity_truth": equity_truth,
        },
        "operator_sequence": [
            "Keep bb-paper-trading.service stopped while reviewing this report.",
            "Clear every blocker; warnings must be explicitly accepted by the operator.",
            "After can_resume_paper=true, start paper trading manually or through an approved release command.",
            "Watch OKX authoritative sync and specialist shadow evaluation after resume.",
        ],
        "checked_at": _now_iso(),
    }


async def _unavailable_report_provider(name: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "read_only": True,
        "audit_only": True,
        "error": safe_error_text(exc, limit=180),
        "source": name,
    }


@dataclass(slots=True)
class Phase3PaperResumePreflightService:
    """Collect a read-only go/no-go report before paper trading is restarted."""

    okx_sync_provider: ReportProvider | None = None
    okx_integrity_provider: ReportProvider | None = None
    model_server_provider: ReportProvider | None = None
    platform_runtime_provider: ReportProvider | None = None
    platform_server_provider: ReportProvider | None = None
    specialist_shadow_provider: ReportProvider | None = None
    account_equity_provider: ReportProvider | None = None
    okx_lookback_hours: int = DEFAULT_OKX_LOOKBACK_HOURS
    okx_limit: int = DEFAULT_OKX_LIMIT
    okx_timeout_seconds: float = DEFAULT_OKX_TIMEOUT_SECONDS
    model_server_timeout_seconds: int = 24
    specialist_report_max_age_seconds: int = DEFAULT_SPECIALIST_REPORT_MAX_AGE_SECONDS

    async def report(self) -> dict[str, Any]:
        okx_sync = await self._collect(
            "okx_authoritative_sync",
            self.okx_sync_provider,
            self._default_okx_sync,
        )
        okx_integrity = await self._collect(
            "okx_trade_fact_integrity",
            self.okx_integrity_provider,
            self._default_okx_integrity,
        )
        model_server, platform_runtime, platform_server, specialist = await asyncio.gather(
            self._collect(
                "phase3_model_server_readiness",
                self.model_server_provider,
                self._default_model_server,
            ),
            self._collect(
                "platform_runtime",
                self.platform_runtime_provider,
                collect_platform_runtime_status,
            ),
            self._collect(
                "platform_server",
                self.platform_server_provider,
                self._default_platform_server,
            ),
            self._collect(
                "specialist_shadow_evaluation",
                self.specialist_shadow_provider,
                self._default_specialist_shadow,
            ),
        )
        account_equity = await self._collect(
            "account_equity_truth",
            self.account_equity_provider,
            self._default_account_equity,
        )

        return evaluate_phase3_paper_resume_preflight_inputs(
            okx_authoritative_sync=okx_sync,
            okx_trade_fact_integrity=okx_integrity,
            model_server_readiness=model_server,
            platform_runtime=platform_runtime,
            platform_server=platform_server,
            specialist_shadow_evaluation=specialist,
            account_equity_truth=account_equity,
            specialist_report_max_age_seconds=self.specialist_report_max_age_seconds,
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
            return await _unavailable_report_provider(name, exc)

    async def _default_okx_sync(self) -> dict[str, Any]:
        return await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=self.okx_lookback_hours,
            limit=self.okx_limit,
            timeout_seconds=self.okx_timeout_seconds,
        ).collect()

    async def _default_okx_integrity(self) -> dict[str, Any]:
        return await OkxTradeFactIntegrityService(
            lookback_hours=max(self.okx_lookback_hours, 24),
            limit=max(self.okx_limit, 120),
        ).audit()

    async def _default_model_server(self) -> dict[str, Any]:
        return await Phase3ModelServerReadinessAuditService(
            timeout_seconds=self.model_server_timeout_seconds,
        ).report()

    async def _default_platform_server(self) -> dict[str, Any]:
        from services.server_monitor_status import collect_platform_server_status

        return collect_platform_server_status()

    async def _default_specialist_shadow(self) -> dict[str, Any]:
        from config.settings import settings

        candidates = [
            settings.data_dir / "phase3" / "specialist_shadow_evaluation_latest.json",
            Path.cwd() / "reports" / "phase3" / "specialist_shadow_evaluation_latest.json",
        ]
        for path in candidates:
            try:
                if not path.exists():
                    continue
                import json

                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(payload, dict):
                payload.setdefault("available", True)
                payload.setdefault("report_path", str(path))
                return payload
        return {
            "available": False,
            "status": "missing",
            "read_only": True,
            "audit_only": True,
            "live_mutation": False,
            "reason": "specialist_shadow_evaluation_report_missing",
            "candidate_paths": [str(path) for path in candidates],
        }

    async def _default_account_equity(self) -> dict[str, Any]:
        executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
        try:
            await executor.initialize()
            snapshot = await executor.get_balance_snapshot("USDT")
        finally:
            try:
                await executor.shutdown()
            except Exception as exc:
                snapshot = {
                    "error": f"executor shutdown failed: {safe_error_text(exc)}"
                }
        if not isinstance(snapshot, dict):
            return {
                "available": False,
                "status": "unavailable",
                "read_only": True,
                "audit_only": True,
                "source": "okx_snapshot",
                "error": f"balance snapshot returned {type(snapshot).__name__}",
            }
        result = dict(snapshot)
        result["available"] = not bool(result.get("error")) and _account_equity_value(result) > 0
        result["status"] = "ok" if result["available"] else "unavailable"
        result["read_only"] = True
        result["audit_only"] = True
        result["source"] = "okx_snapshot"
        return result
