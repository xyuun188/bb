"""Root-cause radar API for online system audits."""

from __future__ import annotations

import ast
import asyncio
import copy
import inspect
import json
from collections import Counter
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter
from sqlalchemy import and_, func, or_, select

from config.settings import settings
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol
from db.session import get_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest, TradeReflection
from models.market_data import Kline, Ticker
from models.trade import Order, Position
from scripts.audit_runtime_text_integrity import collect_runtime_text_integrity_report
from scripts.repair_okx_position_fact_links import (
    collect_scan_report as collect_position_fact_link_scan_report,
)
from services.crypto_feature_coverage import CryptoFeatureCoverageService
from services.exchange_position_state import (
    exchange_position_display_valuation,
    parse_exchange_position_snapshot,
)
from services.artifact_retirement_audit import ArtifactRetirementAuditService
from services.high_risk_review_audit import HighRiskReviewAuditService
from services.historical_trade_fact_audit import HistoricalTradeFactAuditService
from services.model_dynamic_routing import ModelDynamicRoutingService
from services.model_expert_competition import ModelExpertCompetitionService
from services.model_expert_health import ModelExpertHealthService
from services.okx_authoritative_sync import OkxAuthoritativeSyncService
from services.okx_trade_fact_integrity import OkxTradeFactIntegrityService
from services.phase3_go_no_go import evaluate_phase3_go_no_go_cards
from services.phase3_model_server_readiness import Phase3ModelServerReadinessAuditService
from services.phase3_paper_resume_observation import Phase3PaperResumeObservationService
from services.phase3_paper_resume_preflight import Phase3PaperResumePreflightService
from services.phase3_server_migration_audit import Phase3ServerMigrationAuditService
from services.phase3_stage_handoff import Phase3StageHandoffService
from services.phase3_rebuild_readiness import Phase3RebuildReadinessService
from services.position_capacity_release_audit import PositionCapacityReleaseAuditService
from services.profit_first_governance_report import ProfitFirstGovernanceReportService
from services.profit_first_ranking import ProfitFirstRankingService
from services.profit_first_recovery_blockers import build_profit_first_recovery_blockers
from services.server_monitor_status import collect_platform_runtime_status
from services.execution_reason_localizer import localize_execution_reason
from services.shadow_missed_opportunity_closed_loop import (
    ShadowMissedOpportunityClosedLoopService,
)
from services.strategy_signal_root_cause_audit import StrategySignalRootCauseAuditService
from services.strong_opportunity import StrongOpportunityService
from services.trade_execution_contract import TradeExecutionContractService
from services.trade_fact_trust import closed_position_trade_fact_trusted
from services.trading_params import DEFAULT_TRADING_PARAMS
from web_dashboard.api import data_collection as data_collection_api
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()
_skip_okx_daily_reconciliation_latest: ContextVar[bool] = ContextVar(
    "skip_okx_daily_reconciliation_latest",
    default=False,
)

AUDIT_WINDOWS = {"fast_minutes": 10, "trade_hours": 2, "strategy_hours": 24}
EXPECTED_KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h")
KLINE_STALE_LIMIT_SECONDS = {"1m": 180, "5m": 600, "15m": 1800, "1h": 7200}
STATUS_RANK = {"critical": 0, "warning": 1, "ok": 2, "info": 3}
SYSTEM_AUDIT_HISTORY_FILE = "system_audit_history.jsonl"
POSITION_PRICE_SPLIT_WARN_PCT = 0.03
POSITION_PNL_SPLIT_WARN_USDT = 0.5
OKX_RECONCILIATION_CACHE_TTL_SECONDS = 120
OKX_AUTHORITATIVE_SYNC_CACHE_TTL_SECONDS = 45
MODEL_RUNTIME_PROBE_TIMEOUT_SECONDS = 8.0
SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS = 20.0
SYSTEM_AUDIT_MAX_CONCURRENCY = 4
MODEL_EXPERT_AUDIT_HOURS = 24
MODEL_EXPERT_AUDIT_LIMIT = 200
SHADOW_MISSED_OPPORTUNITY_AUDIT_HOURS = 24
SHADOW_MISSED_OPPORTUNITY_AUDIT_LIMIT = 200
STRONG_OPPORTUNITY_AUDIT_HOURS = 24
STRONG_OPPORTUNITY_AUDIT_LIMIT = 500
POSITION_CAPACITY_RELEASE_AUDIT_HOURS = 24
POSITION_CAPACITY_RELEASE_AUDIT_LIMIT = 500
STRATEGY_SIGNAL_ROOT_CAUSE_AUDIT_HOURS = 24
STRATEGY_SIGNAL_ROOT_CAUSE_AUDIT_LIMIT = 500
OPTIONAL_TRAINING_SOURCE_STATUSES = {"disabled", "not_configured"}
TRADE_EXECUTION_CONTRACT_AUDIT_HOURS = 24
TRADE_EXECUTION_CONTRACT_AUDIT_LIMIT = 500
PROFIT_FIRST_RANKING_AUDIT_HOURS = 72
PROFIT_FIRST_RANKING_AUDIT_LIMIT = 800
PROFIT_FIRST_GOVERNANCE_AUDIT_HOURS = 24
PROFIT_FIRST_GOVERNANCE_AUDIT_LIMIT = 800
OKX_TRADE_FACT_INTEGRITY_AUDIT_HOURS = 72
OKX_TRADE_FACT_INTEGRITY_AUDIT_LIMIT = 500
OKX_AUTHORITATIVE_SYNC_AUDIT_HOURS = 24
OKX_AUTHORITATIVE_SYNC_AUDIT_LIMIT = 500
OKX_AUTHORITATIVE_SYNC_TIMEOUT_SECONDS = 5.0
RUNTIME_OKX_ENTRY_GATE_MIN_FRESH_SECONDS = 180.0
OKX_DAILY_RECONCILIATION_REPORT_MAX_AGE_SECONDS = 36 * 3600
OKX_DAILY_RECONCILIATION_REPORT_REL_PATH = "okx_daily_reconciliation_reports/latest.json"
SPECIALIST_SHADOW_EVALUATION_REL_PATH = "phase3/specialist_shadow_evaluation_latest.json"
SPECIALIST_SHADOW_EVALUATION_ALT_REL_PATH = (
    "reports/phase3/specialist_shadow_evaluation_latest.json"
)
PHASE3_GO_NO_GO_REPORT_REL_PATH = "phase3_go_no_go_reports/latest.json"
PHASE3_PAPER_RESUME_PREFLIGHT_REPORT_REL_PATH = "phase3_paper_resume_preflight_reports/latest.json"
PHASE3_OPERATOR_APPROVAL_REPORT_MAX_AGE_SECONDS = 3 * 3600
OKX_POSITION_FACT_LINK_AUDIT_DAYS = 14
OKX_POSITION_FACT_LINK_AUDIT_MAX_POSITIONS = 300
OKX_RECONCILIATION_AUDIT_MAX_CLOSE_ORDERS = 300
HISTORICAL_TRADE_FACT_AUDIT_DAYS = 180
HISTORICAL_TRADE_FACT_AUDIT_LIMIT = 2000
PHASE3_SERVER_MIGRATION_AUDIT_TIMEOUT_SECONDS = 45
PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS = 24
PHASE3_PAPER_RESUME_OBSERVATION_TIMEOUT_SECONDS = 70
PHASE3_PAPER_RESUME_PREFLIGHT_TIMEOUT_SECONDS = 70
PRIORITY_AUDIT_KEYS = ("okx_reconciliation", "trade_execution_contract")
DB_AUDIT_KEYS = (
    "trade_loop",
    "okx_trade_fact_integrity",
    "position_price_integrity",
    "market_data",
    "strategy_quality",
    "strategy_closed_loop",
    "strategy_signal_root_cause",
    "model_training",
    "model_dynamic_routing",
    "high_risk_review_audit",
    "crypto_feature_coverage",
    "shadow_missed_opportunity",
    "strong_opportunity",
    "position_capacity_release",
    "profit_first_governance",
    "profit_first_ranking",
)
HEAVY_AUDIT_KEYS = (
    "model_expert_health",
    "model_expert_competition",
    "runtime_text_integrity",
)
CARD_OWNER_PATHS = {
    "trade_loop": "services/trading_service.py",
    "okx_reconciliation": "scripts/repair_missing_closed_positions_from_orders.py",
    "okx_trade_fact_integrity": "services/okx_trade_fact_integrity.py",
    "phase3_server_migration": "services/phase3_server_migration_audit.py",
    "phase3_go_no_go": "services/phase3_go_no_go.py",
    "phase3_stage_handoff": "services/phase3_stage_handoff.py",
    "phase3_model_server_readiness": "services/phase3_model_server_readiness.py",
    "phase3_paper_resume_observation": "services/phase3_paper_resume_observation.py",
    "phase3_paper_resume_preflight": "services/phase3_paper_resume_preflight.py",
    "position_price_integrity": "web_dashboard/api/system_audit.py",
    "market_data": "models/market_data.py",
    "strategy_quality": "web_dashboard/api/system_audit.py",
    "strategy_closed_loop": "web_dashboard/api/system_audit.py",
    "strategy_signal_root_cause": "services/strategy_signal_root_cause_audit.py",
    "strategy_gate_contract": "services/runtime_entry_filters.py",
    "model_training": "web_dashboard/api/data_collection.py",
    "model_expert_health": "services/model_expert_health.py",
    "model_expert_competition": "services/model_expert_competition.py",
    "model_dynamic_routing": "services/model_dynamic_routing.py",
    "high_risk_review_audit": "services/high_risk_review_audit.py",
    "crypto_feature_coverage": "services/crypto_feature_coverage.py",
    "shadow_missed_opportunity": "services/shadow_missed_opportunity_closed_loop.py",
    "strong_opportunity": "services/strong_opportunity.py",
    "position_capacity_release": "services/position_capacity_release_audit.py",
    "trade_execution_contract": "services/trade_execution_contract.py",
    "profit_first_governance": "services/profit_first_governance_report.py",
    "profit_first_ranking": "services/profit_first_ranking.py",
    "profit_first_recovery_blockers": "services/profit_first_recovery_blockers.py",
    "visible_text_encoding": "web_dashboard/api/system_audit.py",
    "runtime_text_integrity": "scripts/audit_runtime_text_integrity.py",
}
NODE_OWNER_PATHS = {
    "runtime_loop": "services/trading_service.py",
    "market_data": "models/market_data.py",
    "crypto_feature_coverage": "services/crypto_feature_coverage.py",
    "model_training": "web_dashboard/api/data_collection.py",
    "model_expert_health": "services/model_expert_health.py",
    "model_expert_competition": "services/model_expert_competition.py",
    "model_dynamic_routing": "services/model_dynamic_routing.py",
    "high_risk_review_audit": "services/high_risk_review_audit.py",
    "shadow_missed_opportunity": "services/shadow_missed_opportunity_closed_loop.py",
    "strong_opportunity": "services/strong_opportunity.py",
    "position_capacity_release": "services/position_capacity_release_audit.py",
    "strategy_decision": "services/trading_policies.py",
    "strategy_closed_loop": "web_dashboard/api/system_audit.py",
    "strategy_signal_root_cause": "services/strategy_signal_root_cause_audit.py",
    "strategy_gate_contract": "services/runtime_entry_filters.py",
    "risk_guard": "services/trading_policies.py",
    "okx_execution": "services/execution_service.py",
    "position_sync": "services/position_sync_service.py",
    "server_migration": "services/phase3_server_migration_audit.py",
    "phase3_go_no_go": "services/phase3_go_no_go.py",
    "profit_first_ranking": "services/profit_first_ranking.py",
    "profit_first_governance": "services/profit_first_governance_report.py",
    "profit_first_recovery_blockers": "services/profit_first_recovery_blockers.py",
    "phase3_stage_handoff": "services/phase3_stage_handoff.py",
    "model_server_readiness": "services/phase3_model_server_readiness.py",
    "paper_resume_preflight": "services/phase3_paper_resume_preflight.py",
    "training_data": "services/okx_trade_fact_integrity.py",
    "dashboard_observability": "web_dashboard/static/js/dashboard.js",
    "visible_text_encoding": "web_dashboard/api/system_audit.py",
    "runtime_text_integrity": "scripts/audit_runtime_text_integrity.py",
}

_okx_reconciliation_cache: tuple[datetime, dict[str, Any]] | None = None
_okx_authoritative_sync_cache: tuple[datetime, dict[str, Any]] | None = None
_system_audit_collect_lock: asyncio.Lock | None = None


def _system_audit_lock() -> asyncio.Lock:
    global _system_audit_collect_lock
    if _system_audit_collect_lock is None:
        _system_audit_collect_lock = asyncio.Lock()
    return _system_audit_collect_lock


def _u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


SOURCE_MOJIBAKE_SCAN_TARGETS = (
    ("ai_brain", "*.py"),
    ("config", "*.py"),
    ("core", "*.py"),
    ("db", "*.py"),
    ("models", "*.py"),
    ("services", "*.py"),
    ("web_dashboard/api", "*.py"),
    ("web_dashboard/static/js", "*.js"),
    ("web_dashboard/static/css", "*.css"),
    ("web_dashboard/static", "*.html"),
    ("scripts", "*.py"),
)
SOURCE_MOJIBAKE_MARKERS = (
    _u("\\u951f"),
    _u("\\u951b"),
    _u("\\u9286"),
    _u("\\u95ab"),
    _u("\\u95b8"),
    _u("\\u95b9"),
    _u("\\u9227"),
    _u("\\u9225"),
    _u("\\u93c8"),
    _u("\\u93c3"),
    _u("\\u7487"),
    _u("\\u9352"),
    _u("\\u9359"),
    _u("\\u7459"),
    _u("\\u93b4"),
    _u("\\u93c1"),
    _u("\\u7edb"),
    _u("\\u6d5c\\u5fd4\\u5d2f"),
    _u("\\u9429"),
    _u("\\u7ee0\\u20ac"),
    _u("\\ufffd"),
)
STRATEGY_GATE_FORBIDDEN_PATTERNS = (
    "settings.min_entry_volume_ratio",
    "settings.min_entry_adx",
    "if False and",
)
STRATEGY_GATE_ALLOWED_PATHS = {"services/runtime_entry_filters.py"}
STRATEGY_ALLOWED_TOP_LEVEL_CONSTANTS = {"ACTION_SCORE"}
STRATEGY_PARAMETERIZED_TOKENS = (
    "DEFAULT_TRADING_PARAMS",
    "_PARAMS",
    "ENSEMBLE_ENTRY_DECISION_PARAMS",
    "ENSEMBLE_EXIT_DECISION_PARAMS",
    "ENSEMBLE_ML_PROBE_PARAMS",
    "ENTRY_RISK_SIZING_PARAMS",
)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _parse_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _age_seconds(value: Any) -> float | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return max((_now() - value.astimezone(UTC)).total_seconds(), 0.0)


def _status_from_counts(*, critical: bool = False, warning: bool = False) -> str:
    if critical:
        return "critical"
    if warning:
        return "warning"
    return "ok"


def _owner_path_for_card(key: str) -> str:
    return CARD_OWNER_PATHS.get(str(key or ""), "web_dashboard/api/system_audit.py")


def _owner_path_for_node(key: str, related_cards: list[dict[str, Any]]) -> str:
    node_owner = NODE_OWNER_PATHS.get(str(key or ""))
    if node_owner:
        return node_owner
    for card in related_cards:
        owner_path = str(card.get("owner_path") or "")
        if owner_path:
            return owner_path
    return "web_dashboard/api/system_audit.py"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _phase3_merge_clean_training_view_into_local_tools(
    local_tools: dict[str, Any],
    governance: dict[str, Any],
) -> dict[str, Any]:
    """Use the Phase 3 clean training view when live service status has no samples."""

    result = dict(local_tools)
    clean_view = _safe_dict(governance.get("local_ai_tools"))
    if not clean_view:
        return result

    clean_shadow_count = _safe_int_value(
        clean_view.get("phase3_clean_trainable_sample_count")
        or clean_view.get("trainable_sample_count")
        or clean_view.get("total_trainable_count")
    )
    clean_trade_count = _safe_int_value(clean_view.get("trade_sample_count"))
    current_shadow_count = _safe_int_value(
        result.get("shadow_sample_count")
        or result.get("trainable_sample_count")
        or result.get("total_trainable_count")
    )
    current_trade_count = _safe_int_value(
        result.get("trainable_trade_sample_count") or result.get("trade_sample_count")
    )

    if clean_shadow_count > current_shadow_count:
        result["shadow_sample_count"] = clean_shadow_count
        result["trainable_sample_count"] = clean_shadow_count
        result["total_trainable_count"] = clean_shadow_count
        result["phase3_clean_trainable_sample_count"] = clean_shadow_count
        result["training_sample_source"] = "phase3_clean_training_view"
    if clean_trade_count > current_trade_count:
        result["trade_sample_count"] = clean_trade_count
        result["trainable_trade_sample_count"] = clean_trade_count
        result["training_trade_sample_source"] = "phase3_clean_training_view"

    for key in (
        "sequence_sample_count",
        "text_sentiment_sample_count",
        "completed_shadow_sample_count",
        "completed_trade_sample_count",
        "raw_trade_sample_count",
        "quarantined_trade_sample_count",
    ):
        clean_value = _safe_int_value(clean_view.get(key))
        if clean_value > _safe_int_value(result.get(key)):
            result[key] = clean_value

    if not _safe_dict(result.get("quality_report")):
        quality = _safe_dict(clean_view.get("quality_report")) or _safe_dict(
            governance.get("local_ai_quality_report")
        )
        if quality:
            result["quality_report"] = quality
    if not _safe_dict(result.get("governance_report")):
        result["governance_report"] = governance

    result.setdefault("phase3_training_policy", "clean_training_view_only")
    result.setdefault("legacy_data_policy", "excluded_from_phase3_training")
    result["legacy_data_training_allowed"] = False
    result.setdefault("raw_records_preserved", True)
    return result


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


def _paper_service_active(platform_server: dict[str, Any]) -> bool:
    for item in _safe_list(platform_server.get("services")):
        if isinstance(item, dict) and str(item.get("name") or "") == "bb-paper-trading.service":
            return bool(item.get("active"))
    return False


def _load_trading_runtime_status_for_audit() -> dict[str, Any]:
    """Read the split-process trading heartbeat without touching the engine."""

    path = settings.data_dir / "trading_runtime_status.json"
    try:
        if not path.exists():
            return {"available": False, "reason": "missing_runtime_heartbeat"}
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
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


def _okx_runtime_entry_gate_summary(runtime_status: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether runtime OKX sync currently blocks new entries."""

    if not runtime_status.get("available"):
        return {
            "available": False,
            "status": "runtime_unavailable",
            "sync_status": "unknown",
            "entry_blocked": True,
            "blocker": "runtime_heartbeat_unavailable",
            "reason": (
                "交易运行时心跳不可用；暂停新开仓，直到运行时发布新的 OKX 同步心跳。"
            ),
            "heartbeat_age_seconds": runtime_status.get("heartbeat_age_seconds"),
            "heartbeat_fresh_limit_seconds": None,
            "running": runtime_status.get("running"),
        }
    sync = _safe_dict(runtime_status.get("okx_authoritative_sync"))
    sync_status = str(sync.get("status") or "unknown").lower()
    requires_attention = int(sync.get("last_requires_attention_count") or 0)
    last_error = str(sync.get("last_error") or "").strip()
    running = bool(runtime_status.get("running"))
    heartbeat_age = runtime_status.get("heartbeat_age_seconds")
    decision_interval = _safe_float(
        runtime_status.get("decision_interval"),
        float(settings.decision_interval_seconds or 60),
    )
    heartbeat_fresh_limit = max(
        decision_interval * 4.0,
        RUNTIME_OKX_ENTRY_GATE_MIN_FRESH_SECONDS,
    )
    heartbeat_stale = (
        heartbeat_age is None
        or _safe_float(heartbeat_age, heartbeat_fresh_limit + 1.0) > heartbeat_fresh_limit
    )
    entry_blocked = False
    reason = "OKX 运行态同步正常，允许新开仓。"
    blocker: str | None = None
    status = sync_status
    if not running:
        entry_blocked = True
        status = "runtime_inactive"
        blocker = "trading_runtime_inactive"
        reason = "交易运行时未运行；OKX 运行态同步无法授权新开仓。"
    elif heartbeat_stale:
        entry_blocked = True
        status = "runtime_heartbeat_stale"
        blocker = "trading_runtime_heartbeat_stale"
        reason = (
            "交易运行时心跳已过期；暂停新开仓，直到观察到新的 OKX 同步心跳。"
        )
    elif sync_status in {"warning", "stale"}:
        entry_blocked = True
        blocker = "okx_authoritative_sync_unhealthy"
        reason = (
            "OKX 运行态同步已过期；暂停新开仓。"
            if sync_status == "stale"
            else "OKX 运行态同步异常；暂停新开仓。"
        )
        if last_error:
            reason = f"{reason} 最近错误：{last_error}"
    elif requires_attention > 0:
        entry_blocked = True
        blocker = "okx_authoritative_sync_unhealthy"
        reason = (
            f"OKX 运行态同步发现 {requires_attention} 个当前状态差异；"
            "暂停新开仓，等待状态对齐后再恢复。"
        )
    reason = localize_execution_reason(reason) or reason
    return {
        "available": True,
        "status": status,
        "sync_status": sync_status,
        "entry_blocked": entry_blocked,
        "blocker": blocker,
        "reason": reason,
        "running": running,
        "heartbeat_age_seconds": heartbeat_age,
        "heartbeat_fresh_limit_seconds": round(heartbeat_fresh_limit, 3),
        "last_success_at": sync.get("last_success_at"),
        "last_failure_at": sync.get("last_failure_at"),
        "last_error": last_error or None,
        "last_result_count": sync.get("last_result_count"),
        "last_result_kinds": _safe_dict(sync.get("last_result_kinds")),
        "last_requires_attention_count": requires_attention,
        "last_samples": _safe_list(sync.get("last_samples"))[:8],
        "source": sync.get("source") or "okx_private_api_current_positions",
    }


def _load_okx_daily_reconciliation_report_summary() -> dict[str, Any]:
    path = settings.data_dir / OKX_DAILY_RECONCILIATION_REPORT_REL_PATH
    base: dict[str, Any] = {
        "available": False,
        "path": str(path),
        "max_age_seconds": OKX_DAILY_RECONCILIATION_REPORT_MAX_AGE_SECONDS,
        "read_only": True,
        "mutates_database": False,
    }
    if _skip_okx_daily_reconciliation_latest.get():
        return {
            **base,
            "status": "skipped",
            "stale": False,
            "requires_attention": False,
            "can_open_new_entries": False,
            "can_refresh_training": False,
            "entry_blocked": False,
            "training_blocked": False,
            "skip_reason": "daily_report_generation_avoids_self_referential_latest",
        }
    try:
        if not path.exists():
            return {**base, "status": "missing", "stale": True}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {**base, "status": "invalid", "stale": True}
        generated_at = _parse_utc_datetime(payload.get("generated_at"))
        age = _age_seconds(generated_at) if generated_at is not None else None
        stale = (
            age is None
            or age > OKX_DAILY_RECONCILIATION_REPORT_MAX_AGE_SECONDS
            or bool(payload.get("artifact_error"))
        )
        gates = _safe_dict(payload.get("operational_gates"))
        ledger = _safe_dict(payload.get("issue_ledger"))
        return {
            **base,
            "available": True,
            "status": payload.get("status") or "unknown",
            "generated_at": _iso(generated_at),
            "age_seconds": None if age is None else round(age, 3),
            "stale": stale,
            "requires_attention": bool(payload.get("requires_attention")),
            "can_open_new_entries": bool(payload.get("can_open_new_entries")),
            "can_refresh_training": bool(payload.get("can_refresh_training")),
            "entry_blocked": bool(gates.get("entry_blocked")),
            "training_blocked": bool(gates.get("training_blocked")),
            "attention_buckets": _safe_dict(gates.get("attention_buckets")),
            "issue_ledger_summary": _safe_dict(ledger.get("summary")),
            "entry_blockers": _safe_list(gates.get("entry_blockers"))[:8],
            "training_blockers": _safe_list(gates.get("training_blockers"))[:8],
            "attention_items": _safe_list(gates.get("attention_items"))[:8],
            "artifacts": _safe_dict(payload.get("artifacts")),
        }
    except Exception as exc:
        return {
            **base,
            "status": "read_failed",
            "stale": True,
            "error": safe_error_text(exc, limit=180),
        }


def _okx_position_snapshot_evidence(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "okx_pos_side": snapshot.get("raw_pos_side"),
        "okx_raw_pos": snapshot.get("raw_pos"),
        "okx_signed_position_size": round(_safe_float(snapshot.get("signed_position_size")), 8),
        "okx_side_inference": snapshot.get("side_inference"),
        "okx_ccxt_side": snapshot.get("raw_ccxt_side"),
    }


def _safe_crypto_feature_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["live_signal_mutation"] = False
    safe["can_missing_features_drive_live_entry"] = False
    safe["feature_defaults_are_neutral"] = True
    policy = _safe_dict(safe.get("feature_contribution_policy"))
    policy["missing_feature_policy"] = "neutral_blocked"
    policy["stale_feature_policy"] = "neutral_blocked"
    policy["low_confidence_event_policy"] = "shadow_only"
    safe["feature_contribution_policy"] = policy
    features = safe.get("features") if isinstance(safe.get("features"), list) else []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        if str(feature.get("status") or "") in {"missing", "stale", "low_confidence"}:
            feature["live_entry_influence"] = "blocked"
    return safe


def _safe_dynamic_routing_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["live_route_mutation"] = False
    safe["can_apply_live_route"] = False
    summary = _safe_dict(safe.get("summary"))
    safe["promotion_gate"] = {
        "canary_ready_count": int(summary.get("canary_ready_count") or 0),
        "live_ready_count": int(summary.get("live_ready_count") or 0),
        "live_blocked_count": int(summary.get("live_blocked_count") or 0),
        "live_route_mutation": False,
        "can_apply_live_route": False,
        "policy": "shadow/canary/live evidence is report-only until live mutation is explicitly enabled outside this audit.",
    }
    return safe


def _safe_high_risk_review_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["read_only"] = True
    safe["live_entry_mutation"] = False
    safe["can_bypass_risk_controls"] = False
    safe["can_force_open"] = False
    safe["hard_review_must_approve_before_execution"] = True
    for key in ("samples", "recent_reviews", "blocked", "unsafe_executed"):
        rows = safe.get(key) if isinstance(safe.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["can_bypass_risk_controls"] = False
            row["can_force_open"] = False
    return safe


def _safe_shadow_missed_opportunity_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["live_entry_mutation"] = False
    safe["can_bypass_risk_controls"] = False
    safe["weak_evidence_execution_allowed"] = False
    safe["global_missed_count_can_drive_entries"] = False
    for key in ("adopted", "probe_candidates", "blocked", "observe_only"):
        rows = safe.get(key) if isinstance(safe.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["can_force_open"] = False
            row["can_bypass_risk_controls"] = False
    return safe


def _safe_strong_opportunity_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["live_entry_mutation"] = False
    safe["live_sizing_mutation"] = False
    safe["can_bypass_risk_controls"] = False
    safe["can_force_open"] = False
    safe["can_apply_live_sizing"] = False
    for key in ("strong_candidates", "near_misses"):
        rows = safe.get(key) if isinstance(safe.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["can_bypass_risk_controls"] = False
            row["can_force_open"] = False
            row["can_apply_live_sizing"] = False
    return safe


def _safe_position_capacity_release_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["read_only"] = True
    safe["audit_only"] = True
    safe["live_exit_mutation"] = False
    safe["live_entry_mutation"] = False
    safe["live_sizing_mutation"] = False
    safe["can_force_close"] = False
    safe["can_close_winners"] = False
    safe["can_bypass_risk_controls"] = False
    for key in (
        "current_release_candidates",
        "old_profit_rotation_candidates",
        "unclosed_release_decisions",
        "protected_release_decisions",
        "exchange_blocked_release_decisions",
        "execution_link_gap_release_decisions",
        "stale_release_decisions",
        "crowded_blocks",
    ):
        rows = safe.get(key) if isinstance(safe.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["can_force_close"] = False
            row["can_close_winners"] = False
            row["can_bypass_risk_controls"] = False
    return safe


def _safe_trade_execution_contract_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["live_entry_mutation"] = False
    safe["live_exit_mutation"] = False
    safe["can_bypass_risk_controls"] = False
    policy = _safe_dict(safe.get("policy"))
    policy["entry_requires_positive_expected_net"] = True
    policy["entry_requires_structured_evidence"] = True
    policy["entry_requires_profit_first_trade_plan"] = True
    policy["profit_first_missing_plan_is_hard_violation"] = True
    policy["profit_first_shadow_lane_cannot_execute"] = True
    policy["position_size_requires_profit_risk_sizing"] = True
    policy["fast_loss_exit_requires_strong_exit_evidence"] = True
    policy["dust_fast_loss_requires_tiny_notional_and_tiny_abs_pnl"] = True
    policy["recent_loss_reentry_requires_strong_unlock"] = True
    policy["profit_first_probe_loss_brake_must_block_execution"] = True
    safe["policy"] = policy
    return safe


def _safe_profit_first_ranking_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["read_only"] = True
    safe["live_mutation"] = False
    safe["live_weight_mutation"] = False
    safe["live_sizing_mutation"] = False
    safe["can_change_model_routing"] = False
    safe["can_change_strategy_weight"] = False
    safe["can_increase_live_size"] = False
    for key in ("strategy_rankings", "source_rankings"):
        rows = safe.get(key) if isinstance(safe.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            stage = str(row.get("recommended_stage") or "").strip().lower()
            row["live_mutation"] = False
            row["live_weight_mutation"] = False
            row["can_increase_live_size"] = False
            if stage in {"shadow", "demote", "disable"}:
                row["can_increase_budget"] = False
                row["can_apply_live_weight"] = False
            if stage in {"demote", "disable"}:
                row["can_keep_live_size"] = False
    recommendations = safe.get("brain_recommendations")
    if isinstance(recommendations, dict):
        recommendations["live_mutation"] = False
        for key in (
            "strategy_actions",
            "source_weights",
            "lane_threshold_recommendations",
            "size_promotion_demotion",
            "no_entry_threshold_recommendations",
            "exit_policy_adjustments",
        ):
            rows = recommendations.get(key) if isinstance(recommendations.get(key), list) else []
            for row in rows:
                if isinstance(row, dict):
                    row["live_mutation"] = False
                    row["live_weight_mutation"] = False
                    row["can_increase_live_size"] = False
    if not isinstance(safe.get("blockers"), list):
        safe["blockers"] = []
    if not isinstance(safe.get("summary"), dict):
        safe["summary"] = {}
    if not isinstance(safe.get("policy"), dict):
        safe["policy"] = {}
    return safe


def _profit_first_ranking_observation_only(details: dict[str, Any]) -> bool:
    if not isinstance(details, dict):
        return False
    summary = _safe_dict(details.get("summary"))
    blockers = [_safe_dict(item) for item in _safe_list(details.get("blockers"))]
    unsafe_flags = (
        bool(details.get("live_mutation"))
        or bool(details.get("live_weight_mutation"))
        or bool(details.get("live_sizing_mutation"))
        or bool(details.get("can_change_model_routing"))
        or bool(details.get("can_change_strategy_weight"))
        or bool(details.get("can_increase_live_size"))
    )
    if (
        details.get("report_available") is False
        or not bool(details.get("audit_only"))
        or not bool(details.get("read_only"))
        or unsafe_flags
        or not bool(details.get("ranking_ready"))
        or int(summary.get("disable_count") or 0) > 0
    ):
        return False
    if any(str(item.get("severity") or "") == "blocking" for item in blockers):
        return False
    for key in ("strategy_rankings", "source_rankings"):
        rows = details.get(key) if isinstance(details.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            stage = str(row.get("recommended_stage") or "").strip().lower()
            if stage in {"shadow", "demote", "disable"} and bool(
                row.get("can_increase_budget")
            ):
                return False
            if stage in {"demote", "disable"} and bool(row.get("can_keep_live_size")):
                return False
            if stage in {"shadow", "demote", "disable"} and bool(
                row.get("can_apply_live_weight")
            ):
                return False
    return True


def _safe_profit_first_governance_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["read_only"] = True
    safe["live_mutation"] = False
    safe["live_entry_mutation"] = False
    safe["live_exit_mutation"] = False
    safe["live_weight_mutation"] = False
    safe["live_sizing_mutation"] = False
    safe["can_submit_orders"] = False
    safe["can_start_trading_service"] = False
    safe["can_change_model_routing"] = False
    safe["can_change_strategy_weight"] = False
    safe["can_increase_live_size"] = False
    for key in (
        "no_entry_governance",
        "losing_exit_governance",
        "policy",
        "summary",
        "ranking_summary",
        "trade_fact_report",
    ):
        if not isinstance(safe.get(key), dict):
            safe[key] = {}
    if not isinstance(safe.get("next_cycle_actions"), list):
        safe["next_cycle_actions"] = []
    if not isinstance(safe.get("missing_brain_outputs"), list):
        safe["missing_brain_outputs"] = []
    return safe


def _relative_gap(left: float, right: float) -> float:
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator


def _distribution(values: list[float]) -> dict[str, Any]:
    clean = sorted(value for value in values if isinstance(value, int | float))
    if not clean:
        return {"count": 0}

    def percentile(ratio: float) -> float:
        index = min(max(int((len(clean) - 1) * ratio), 0), len(clean) - 1)
        return round(float(clean[index]), 6)

    return {
        "count": len(clean),
        "min": round(float(clean[0]), 6),
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "max": round(float(clean[-1]), 6),
        "avg": round(float(sum(clean) / len(clean)), 6),
    }


def _decision_raw(row: AIDecision) -> dict[str, Any]:
    return row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {}


def _decision_opportunity(row: AIDecision) -> dict[str, Any]:
    raw = _decision_raw(row)
    opportunity = raw.get("opportunity_score")
    return opportunity if isinstance(opportunity, dict) else {}


def _decision_evidence(row: AIDecision) -> dict[str, Any]:
    evidence = _decision_opportunity(row).get("evidence_score")
    return evidence if isinstance(evidence, dict) else {}


def _decision_evidence_tier(row: AIDecision) -> str:
    return str(_decision_evidence(row).get("tier") or "")


def _decision_expected_net(row: AIDecision) -> float | None:
    opportunity = _decision_opportunity(row)
    if "expected_net_return_pct" not in opportunity:
        return None
    try:
        return float(opportunity.get("expected_net_return_pct"))
    except (TypeError, ValueError):
        return None


def _audit_card(
    key: str,
    title: str,
    status: str,
    summary: str,
    *,
    details: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    next_actions: list[str] | None = None,
    owner_path: str | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "title": title,
        "status": status,
        "summary": summary,
        "owner_path": owner_path or _owner_path_for_card(key),
        "details": details or {},
        "evidence": evidence or [],
        "next_actions": next_actions or [],
    }


def _read_json_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _report_checked_at(report: dict[str, Any]) -> datetime | None:
    return _parse_utc_datetime(
        report.get("checked_at")
        or report.get("generated_at")
        or report.get("created_at")
        or report.get("timestamp")
    )


def _report_fresh(report: dict[str, Any], *, max_age_seconds: int) -> bool:
    checked_at = _report_checked_at(report)
    if checked_at is None:
        return False
    age = _age_seconds(checked_at)
    return age is not None and age <= max_age_seconds


def _read_latest_phase3_report(relative_path: str) -> dict[str, Any]:
    report = _read_json_report(settings.data_dir / relative_path)
    if report:
        report.setdefault("report_path", str(settings.data_dir / relative_path))
        return report
    local_path = Path.cwd() / "data" / relative_path
    report = _read_json_report(local_path)
    if report:
        report.setdefault("report_path", str(local_path))
        return report
    return {}


def _phase3_paper_resume_pending_operator_approval() -> dict[str, Any]:
    go_no_go = _read_latest_phase3_report(PHASE3_GO_NO_GO_REPORT_REL_PATH)
    preflight = _read_latest_phase3_report(PHASE3_PAPER_RESUME_PREFLIGHT_REPORT_REL_PATH)
    go_no_go_fresh = _report_fresh(
        go_no_go,
        max_age_seconds=PHASE3_OPERATOR_APPROVAL_REPORT_MAX_AGE_SECONDS,
    )
    preflight_fresh = _report_fresh(
        preflight,
        max_age_seconds=PHASE3_OPERATOR_APPROVAL_REPORT_MAX_AGE_SECONDS,
    )
    go_no_go_details = _safe_dict(go_no_go.get("go_no_go"))
    go_status = str(go_no_go.get("status") or go_no_go_details.get("status") or "").strip()
    can_start_paper = bool(go_no_go_details.get("can_start_paper_with_operator_approval"))
    can_resume_paper = bool(preflight.get("can_resume_paper"))
    ready = (
        go_no_go_fresh
        and preflight_fresh
        and go_status == "paper_resume_ready"
        and can_start_paper
        and can_resume_paper
    )
    return {
        "ready": ready,
        "status": "paper_resume_ready" if ready else "not_ready",
        "go_no_go_status": go_status or "missing",
        "go_no_go_fresh": go_no_go_fresh,
        "preflight_fresh": preflight_fresh,
        "can_start_paper_with_operator_approval": can_start_paper,
        "can_resume_paper": can_resume_paper,
        "max_age_seconds": PHASE3_OPERATOR_APPROVAL_REPORT_MAX_AGE_SECONDS,
        "go_no_go_report_path": go_no_go.get("report_path"),
        "preflight_report_path": preflight.get("report_path"),
    }


def _specialist_shadow_latest_report() -> dict[str, Any]:
    candidates = [
        settings.data_dir / SPECIALIST_SHADOW_EVALUATION_REL_PATH,
        Path.cwd() / SPECIALIST_SHADOW_EVALUATION_ALT_REL_PATH,
    ]
    for path in candidates:
        report = _read_json_report(path)
        if report:
            report.setdefault("report_path", str(path))
            report.setdefault("available", True)
            return report
    return {
        "available": False,
        "ok": False,
        "live_mutation": False,
        "promotion_flow": "shadow_to_canary_to_live",
        "completed_count": 0,
        "eligible_shadow_count": 0,
        "model_count": 0,
        "models": [],
        "summary": {"promotion_ready_count": 0, "blocked_count": 0},
        "reason": "specialist_shadow_evaluation_report_missing",
        "candidate_paths": [str(path) for path in candidates],
    }


async def _audit_maybe_async(factory: Any) -> dict[str, Any]:
    result = factory()
    if inspect.isawaitable(result):
        result = await asyncio.wait_for(
            result,
            timeout=max(float(SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS or 20.0), 0.001),
        )
    return result


async def _run_audit_specs(
    specs: list[tuple[str, Any]],
    *,
    max_concurrency: int = SYSTEM_AUDIT_MAX_CONCURRENCY,
) -> dict[str, dict[str, Any] | Exception]:
    if not specs:
        return {}
    concurrency = max(1, int(max_concurrency or 1))
    if concurrency == 1:
        results: dict[str, dict[str, Any] | Exception] = {}
        for key, factory in specs:
            try:
                results[key] = await _audit_maybe_async(factory)
            except Exception as exc:
                results[key] = exc
        return results

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(key: str, factory: Any) -> tuple[str, dict[str, Any] | Exception]:
        async with semaphore:
            try:
                return key, await _audit_maybe_async(factory)
            except Exception as exc:
                return key, exc

    pairs = await asyncio.gather(*(run_one(key, factory) for key, factory in specs))
    return dict(pairs)


def _cached_okx_reconciliation_card() -> dict[str, Any] | None:
    cached = _okx_reconciliation_cache
    if cached is None:
        return None
    cached_at, payload = cached
    age_seconds = max((_now() - cached_at).total_seconds(), 0.0)
    if age_seconds > OKX_RECONCILIATION_CACHE_TTL_SECONDS:
        return None
    data = copy.deepcopy(payload)
    details = data.setdefault("details", {})
    details["cache"] = {
        "hit": True,
        "age_seconds": round(age_seconds, 3),
        "ttl_seconds": OKX_RECONCILIATION_CACHE_TTL_SECONDS,
    }
    return data


def _store_okx_reconciliation_card(payload: dict[str, Any]) -> dict[str, Any]:
    global _okx_reconciliation_cache
    data = copy.deepcopy(payload)
    details = data.setdefault("details", {})
    details["cache"] = {
        "hit": False,
        "age_seconds": 0.0,
        "ttl_seconds": OKX_RECONCILIATION_CACHE_TTL_SECONDS,
    }
    _okx_reconciliation_cache = (_now(), copy.deepcopy(data))
    return data


def _cached_okx_authoritative_sync_summary() -> dict[str, Any] | None:
    cached = _okx_authoritative_sync_cache
    if cached is None:
        return None
    cached_at, payload = cached
    age_seconds = max((_now() - cached_at).total_seconds(), 0.0)
    if age_seconds > OKX_AUTHORITATIVE_SYNC_CACHE_TTL_SECONDS:
        return None
    data = copy.deepcopy(payload)
    data["cache"] = {
        "hit": True,
        "age_seconds": round(age_seconds, 3),
        "ttl_seconds": OKX_AUTHORITATIVE_SYNC_CACHE_TTL_SECONDS,
    }
    return data


def _store_okx_authoritative_sync_summary(payload: dict[str, Any]) -> dict[str, Any]:
    global _okx_authoritative_sync_cache
    data = copy.deepcopy(payload)
    data["cache"] = {
        "hit": False,
        "age_seconds": 0.0,
        "ttl_seconds": OKX_AUTHORITATIVE_SYNC_CACHE_TTL_SECONDS,
    }
    _okx_authoritative_sync_cache = (_now(), copy.deepcopy(data))
    return data


def _okx_reconciliation_root_cause_summary(
    *,
    classification_counts: dict[str, Any],
    repairable_count: int,
    manual_review_count: int,
    skipped_candidate_count: int,
    unscanned_candidate_count: int,
    truncated: bool,
) -> dict[str, Any]:
    linked_count = int(classification_counts.get("linked") or 0)
    missing_count = max(int(repairable_count or 0) + int(manual_review_count or 0), 0)
    root_causes: list[dict[str, Any]] = []
    if repairable_count:
        root_causes.append(
            {
                "code": "deterministic_repair_available",
                "count": int(repairable_count),
                "owner": "scripts/repair_missing_closed_positions_from_orders.py",
                "training_policy": "quarantine_repaired_samples",
                "action": "Run dry-run, review exact symbol/order-id matches, then apply by explicit order id.",
            }
        )
    if manual_review_count:
        root_causes.append(
            {
                "code": "manual_review_required",
                "count": int(manual_review_count),
                "owner": "services/okx_trade_fact_integrity.py",
                "training_policy": "exclude_until_okx_backed",
                "action": "Inspect OKX close fill, local position lifecycle, symbol alias, side, quantity, and fee allocation.",
            }
        )
    if skipped_candidate_count:
        root_causes.append(
            {
                "code": "candidate_skipped_or_not_repairable",
                "count": int(skipped_candidate_count),
                "owner": "scripts/repair_missing_closed_positions_from_orders.py",
                "training_policy": "exclude_until_classified",
                "action": "Review skipped close orders; do not use inferred PnL for training until classified.",
            }
        )
    if unscanned_candidate_count or truncated:
        root_causes.append(
            {
                "code": "bounded_scan_incomplete",
                "count": int(unscanned_candidate_count),
                "owner": "web_dashboard/api/system_audit.py",
                "training_policy": "hold_training_refresh_until_full_scan",
                "action": "Run the full reconciliation script or scan by order-id batches before treating the window as clean.",
            }
        )
    status = (
        "dirty"
        if missing_count or skipped_candidate_count
        else "incomplete" if unscanned_candidate_count or truncated else "clean"
    )
    return {
        "status": status,
        "linked_close_order_count": linked_count,
        "missing_close_position_count": missing_count,
        "repairable_count": int(repairable_count),
        "manual_review_count": int(manual_review_count),
        "skipped_candidate_count": int(skipped_candidate_count),
        "unscanned_candidate_count": int(unscanned_candidate_count),
        "raw_records_preserved": True,
        "cleanup_mode": "quarantine_not_delete",
        "training_policy": (
            "only_okx_backed_clean_trade_facts"
            if status == "clean"
            else "exclude_dirty_or_unclassified_trade_facts"
        ),
        "requires_training_rebuild": status in {"dirty", "incomplete"},
        "root_causes": root_causes,
    }


def _split_training_source_warnings(
    sources: list[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    optional: list[dict[str, Any]] = []
    hard: list[dict[str, Any]] = []
    for row in sources or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "").strip()
        if status in {"active", "ok"}:
            continue
        if not bool(row.get("enabled")) and status in OPTIONAL_TRAINING_SOURCE_STATUSES:
            optional.append(row)
        else:
            hard.append(row)
    return optional, hard


def _load_trading_runtime_audit_window() -> dict[str, Any]:
    path = settings.data_dir / "trading_runtime_status.json"
    if not path.exists():
        return {"available": False, "started_at": None, "heartbeat_at": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "started_at": None, "heartbeat_at": None}
    if not isinstance(payload, dict):
        return {"available": False, "started_at": None, "heartbeat_at": None}
    started_at = _parse_utc_datetime(payload.get("started_at"))
    heartbeat_at = _parse_utc_datetime(
        payload.get("heartbeat_at") or payload.get("last_heartbeat_at")
    )
    last_market_round_started_at = _parse_utc_datetime(payload.get("last_market_round_started_at"))
    last_market_round_finished_at = _parse_utc_datetime(
        payload.get("last_market_round_finished_at")
    )
    return {
        "available": started_at is not None,
        "started_at": started_at,
        "started_at_iso": _iso(started_at),
        "heartbeat_at": heartbeat_at,
        "heartbeat_at_iso": _iso(heartbeat_at),
        "running": bool(payload.get("running", False)),
        "paused": bool(payload.get("paused", False)),
        "mode": payload.get("mode"),
        "scan_mode": payload.get("scan_mode"),
        "decision_interval": payload.get("decision_interval"),
        "current_stage": payload.get("current_stage"),
        "round_active": bool(payload.get("round_active", False)),
        "market_current_stage": payload.get("market_current_stage"),
        "market_round_active": bool(payload.get("market_round_active", False)),
        "last_market_round_started_at": last_market_round_started_at,
        "last_market_round_started_at_iso": _iso(last_market_round_started_at),
        "last_market_round_finished_at": last_market_round_finished_at,
        "last_market_round_finished_at_iso": _iso(last_market_round_finished_at),
    }


def _summarize_strategy_window(
    *,
    started_at: datetime | None,
    decisions: list[Any],
    positions: list[Any],
    high_quality_tiers: set[str],
    weak_tiers: set[str],
) -> dict[str, Any]:
    if started_at is None:
        return {
            "available": False,
            "started_at": None,
            "decision_count": 0,
            "entry_decision_count": 0,
            "executed_entry_count": 0,
            "weak_executed_count": 0,
            "fast_loss_under_15m_count": 0,
            "high_quality_entry_count": 0,
            "ml_usable_rate": 0.0,
            "historical_legacy_issues": False,
        }

    scoped_decisions = [
        row
        for row in decisions
        if (created_at := _parse_utc_datetime(getattr(row, "created_at", None))) is not None
        and created_at >= started_at
    ]
    scoped_positions = [
        row
        for row in positions
        if (created_at := _parse_utc_datetime(getattr(row, "created_at", None))) is not None
        and created_at >= started_at
    ]
    entry_decisions = [
        row for row in scoped_decisions if str(row.action or "").lower() in {"long", "short"}
    ]
    executed_entries = [row for row in entry_decisions if bool(row.was_executed)]
    high_quality_entries = [
        row for row in entry_decisions if _decision_evidence_tier(row) in high_quality_tiers
    ]
    weak_executed = [row for row in executed_entries if _decision_evidence_tier(row) in weak_tiers]
    component_stats: dict[str, Counter[str]] = {}
    for row in entry_decisions:
        components = _decision_evidence(row).get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            component_stats.setdefault(str(component.get("source") or "unknown"), Counter())[
                str(component.get("status") or "unknown")
            ] += 1
    ml_stats = dict(component_stats.get("ml", Counter()))
    ml_usable = sum(
        count
        for status, count in ml_stats.items()
        if status not in {"ignored", "missing", "unknown"}
    )
    ml_total = sum(ml_stats.values())
    fast_loss_count = 0
    for position in scoped_positions:
        if bool(position.is_open):
            continue
        opened = _parse_utc_datetime(getattr(position, "created_at", None))
        closed = _parse_utc_datetime(getattr(position, "closed_at", None))
        if opened is None or closed is None:
            continue
        hold_minutes = max((closed - opened).total_seconds() / 60.0, 0.0)
        if hold_minutes <= 15 and _safe_float(position.realized_pnl) < 0:
            fast_loss_count += 1
    return {
        "available": True,
        "started_at": _iso(started_at),
        "decision_count": len(scoped_decisions),
        "entry_decision_count": len(entry_decisions),
        "executed_entry_count": len(executed_entries),
        "weak_executed_count": len(weak_executed),
        "fast_loss_under_15m_count": fast_loss_count,
        "high_quality_entry_count": len(high_quality_entries),
        "ml_usable_rate": round(ml_usable / ml_total, 4) if ml_total else 0.0,
        "model_component_status_counts": {
            key: dict(value) for key, value in sorted(component_stats.items())
        },
        "historical_legacy_issues": False,
    }


def _ml_influence_reason_from_decisions(decisions: list[Any]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    influence_flags: Counter[str] = Counter()
    for row in decisions:
        opportunity = _decision_opportunity(row)
        if "ml_influence_enabled" in opportunity:
            influence_flags[str(bool(opportunity.get("ml_influence_enabled"))).lower()] += 1
        components = _decision_evidence(row).get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict) or component.get("source") != "ml":
                continue
            reason = str(component.get("reason") or component.get("status") or "unknown")
            reasons[reason[:160]] += 1
    if not reasons:
        summary = "最近开仓候选没有返回 ML 证据组件。"
    elif reasons.most_common(1)[0][0] == "ignored":
        summary = "ML 组件被标记为 ignored，但未写入具体原因，需要检查证据构建上下文。"
    else:
        summary = reasons.most_common(1)[0][0]
    return {
        "summary": summary,
        "top_reasons": [
            {"reason": reason, "count": count} for reason, count in reasons.most_common(5)
        ],
        "ml_influence_enabled_flags": dict(influence_flags),
    }


async def _trade_loop_audit() -> dict[str, Any]:
    now = _now()
    since_10m = now - timedelta(minutes=AUDIT_WINDOWS["fast_minutes"])
    since_2h = now - timedelta(hours=AUDIT_WINDOWS["trade_hours"])
    runtime_window = _load_trading_runtime_audit_window()
    async with get_session_ctx() as session:
        recent_decisions = (
            await session.execute(
                select(func.count(AIDecision.id), func.max(AIDecision.created_at)).where(
                    AIDecision.created_at >= since_10m
                )
            )
        ).one()
        decisions_2h = (
            await session.execute(
                select(func.count(AIDecision.id), func.max(AIDecision.created_at)).where(
                    AIDecision.created_at >= since_2h
                )
            )
        ).one()
        orders_2h = (
            await session.execute(
                select(func.count(Order.id), func.max(Order.created_at)).where(
                    Order.created_at >= since_2h
                )
            )
        ).one()
        open_positions = (
            await session.execute(select(func.count(Position.id)).where(Position.is_open.is_(True)))
        ).scalar()
    recent_count = int(recent_decisions[0] or 0)
    decisions_count = int(decisions_2h[0] or 0)
    orders_count = int(orders_2h[0] or 0)
    latest_decision_age = _age_seconds(recent_decisions[1])
    runtime_age_seconds = _age_seconds(runtime_window.get("started_at"))
    heartbeat_age_seconds = _age_seconds(runtime_window.get("heartbeat_at"))
    try:
        decision_interval = float(runtime_window.get("decision_interval") or 0)
    except (TypeError, ValueError):
        decision_interval = 0.0
    cold_start_grace_seconds = max(decision_interval * 3.0, 0.0)
    cold_start = bool(runtime_window.get("running")) and (
        runtime_age_seconds is not None
        and cold_start_grace_seconds > 0
        and runtime_age_seconds <= cold_start_grace_seconds
        and (heartbeat_age_seconds is None or heartbeat_age_seconds <= cold_start_grace_seconds)
    )
    heartbeat_fresh_seconds = max(cold_start_grace_seconds, 600.0)
    runtime_heartbeat_fresh = (
        heartbeat_age_seconds is not None and heartbeat_age_seconds <= heartbeat_fresh_seconds
    )
    runtime_running = bool(runtime_window.get("running")) and runtime_heartbeat_fresh
    stale_runtime_heartbeat = bool(runtime_window.get("running")) and not runtime_heartbeat_fresh
    market_analysis_paused = (
        runtime_running and bool(runtime_window.get("paused")) and runtime_heartbeat_fresh
    )
    paper_resume_pending = (
        _phase3_paper_resume_pending_operator_approval()
        if not runtime_running
        else {"ready": False}
    )
    stalled = (
        not bool(paper_resume_pending.get("ready"))
        and not market_analysis_paused
        and not cold_start
        and (recent_count == 0 or (latest_decision_age is not None and latest_decision_age > 600))
    )
    cold_start_no_orders = cold_start and orders_count == 0
    orderless_observation = (
        orders_count == 0
        and decisions_count > 30
        and recent_count > 0
        and not stalled
        and not cold_start
        and not market_analysis_paused
        and runtime_heartbeat_fresh
    )
    status = _status_from_counts(
        critical=stalled,
        warning=(
            bool(paper_resume_pending.get("ready"))
            or market_analysis_paused
            or cold_start_no_orders
            or orderless_observation
        ),
    )
    summary = (
        "交易服务刚重启，当前处于冷启动观察窗口，暂不判定为不开仓异常。"
        if cold_start_no_orders
        else (
            "最近 10 分钟没有新增分析，交易主循环可能卡住。"
            if stalled
            else (
                "最近 2 小时有分析但没有订单，需结合开仓漏斗判断是否策略正常观望。"
                if orders_count == 0 and decisions_count > 30
                else "分析心跳和订单链路有活动。"
            )
        )
    )
    if market_analysis_paused:
        summary = (
            "Market analysis is paused; treat zero new entries as an operator/runtime "
            "pause before diagnosing strategy thresholds."
        )
    if paper_resume_pending.get("ready"):
        summary = (
            "Phase 3 gates are paper_resume_ready; bb-paper-trading.service is still "
            "stopped pending explicit operator approval, so this is not a stalled loop."
        )
    return _audit_card(
        "trade_loop",
        "交易闭环",
        status,
        summary,
        details={
            "last_10m_decisions": recent_count,
            "last_2h_decisions": decisions_count,
            "last_2h_orders": orders_count,
            "open_positions": int(open_positions or 0),
            "latest_decision_at": _iso(recent_decisions[1]),
            "latest_order_at": _iso(orders_2h[1]),
            "cold_start": cold_start,
            "cold_start_no_orders": cold_start_no_orders,
            "orderless_observation": orderless_observation,
            "market_analysis_paused": market_analysis_paused,
            "paper_resume_pending_operator_approval": bool(paper_resume_pending.get("ready")),
            "paper_resume_gate": paper_resume_pending,
            "stale_runtime_heartbeat": stale_runtime_heartbeat,
            "runtime_heartbeat_fresh": runtime_heartbeat_fresh,
            "runtime_age_seconds": (
                round(runtime_age_seconds, 3) if runtime_age_seconds is not None else None
            ),
            "heartbeat_age_seconds": (
                round(heartbeat_age_seconds, 3) if heartbeat_age_seconds is not None else None
            ),
            "cold_start_grace_seconds": round(cold_start_grace_seconds, 3),
            "heartbeat_fresh_seconds": round(heartbeat_fresh_seconds, 3),
            "runtime_window": {
                "running": runtime_running,
                "reported_running": bool(runtime_window.get("running")),
                "paused": bool(runtime_window.get("paused")),
                "mode": runtime_window.get("mode"),
                "scan_mode": runtime_window.get("scan_mode"),
                "current_stage": runtime_window.get("current_stage"),
                "round_active": bool(runtime_window.get("round_active")),
                "market_current_stage": runtime_window.get("market_current_stage"),
                "market_round_active": bool(runtime_window.get("market_round_active")),
                "last_market_round_started_at": _iso(
                    runtime_window.get("last_market_round_started_at")
                ),
                "last_market_round_finished_at": _iso(
                    runtime_window.get("last_market_round_finished_at")
                ),
                "started_at": _iso(runtime_window.get("started_at")),
                "heartbeat_at": _iso(runtime_window.get("heartbeat_at")),
                "decision_interval": runtime_window.get("decision_interval"),
            },
        },
        evidence=[
            {"label": "10分钟决策", "value": recent_count},
            {"label": "2小时订单", "value": orders_count},
        ],
        next_actions=[
            "若处于冷启动观察窗口，先等交易服务完成至少 3 个调度周期再判定卡死。",
            "若 10 分钟决策为 0 且不是冷启动，先查交易服务心跳和当前 stage。",
            "若有大量分析但无订单，打开开仓漏斗看收益期望/风控/OKX 规则分布。",
        ],
    )


async def _okx_reconciliation_audit() -> dict[str, Any]:
    cached = _cached_okx_reconciliation_card()
    if cached is not None:
        return cached
    try:
        report = await asyncio.wait_for(
            _okx_reconciliation_light_scan(
                days=14,
                max_close_orders=OKX_RECONCILIATION_AUDIT_MAX_CLOSE_ORDERS,
            ),
            timeout=5.0,
        )
    except Exception as exc:
        timeout = isinstance(exc, TimeoutError)
        return _store_okx_reconciliation_card(
            _audit_card(
                "okx_reconciliation",
                "OKX 历史对账",
                "ok" if timeout else "warning",
                (
                    "OKX 历史对账完整 dry-run 超时；交易事实审计已正常，作为观察项稍后重试。"
                    if timeout
                    else "OKX 本地订单反推历史仓位 dry-run 执行失败。"
                ),
                details={
                    "error": safe_error_text(exc, limit=180),
                    "timeout": timeout,
                    "hard_failure": not timeout,
                    "window_days": 14,
                },
                next_actions=[
                    "dry-run 超时时不能直接补历史仓位；先确认是否仍在全量扫非平仓成交或数据库慢查询。",
                    "如果连续超时，运行对账脚本查看 candidate_order_count/scanned_order_count，再按订单 ID 精确核对。",
                ],
            )
        )
    plans = report.plans
    missing = len(plans)
    status = "warning" if missing else "warning" if report.truncated else "ok"
    classification_counts = dict(getattr(report, "classification_counts", {}) or {})
    plan_classifications = list(getattr(report, "plan_classifications", []) or [])
    repairable_count = int(getattr(report, "repairable_count", missing) or 0)
    manual_review_count = int(getattr(report, "manual_review_count", 0) or 0)
    skipped_candidate_count = int(getattr(report, "skipped_candidate_count", 0) or 0)
    unscanned_candidate_count = int(getattr(report, "unscanned_candidate_count", 0) or 0)
    root_cause_summary = _okx_reconciliation_root_cause_summary(
        classification_counts=classification_counts,
        repairable_count=repairable_count,
        manual_review_count=manual_review_count,
        skipped_candidate_count=skipped_candidate_count,
        unscanned_candidate_count=unscanned_candidate_count,
        truncated=bool(report.truncated),
    )
    classifications_by_close_order_id = {
        str(item.get("close_order_id") or ""): item
        for item in plan_classifications
        if isinstance(item, dict)
    }
    summary = (
        "存在可由 OKX 成交订单反推的缺失历史仓位。"
        if missing
        else (
            "14 天历史仓位 dry-run 已限量扫描；需运行完整脚本确认无缺失。"
            if report.truncated
            else "14 天历史仓位 dry-run 无缺失。"
        )
    )
    return _store_okx_reconciliation_card(
        _audit_card(
            "okx_reconciliation",
            "OKX 历史对账",
            status,
            summary,
            details={
                "window_days": report.lookback_days,
                "missing_closed_positions": missing,
                "candidate_close_order_count": report.candidate_order_count,
                "scanned_close_order_count": report.scanned_order_count,
                "truncated": report.truncated,
                "max_close_orders": report.max_close_orders,
                "duration_seconds": report.duration_seconds,
                "scan_mode": getattr(report, "scan_mode", "bounded_repair_dry_run"),
                "classification_counts": classification_counts,
                "repairable_count": repairable_count,
                "manual_review_count": manual_review_count,
                "skipped_candidate_count": skipped_candidate_count,
                "unscanned_candidate_count": unscanned_candidate_count,
                "root_cause_summary": root_cause_summary,
                "training_data_policy": {
                    "raw_records_preserved": True,
                    "cleanup_mode": "quarantine_not_delete",
                    "policy": root_cause_summary["training_policy"],
                    "requires_training_rebuild": root_cause_summary["requires_training_rebuild"],
                },
                "sample_plans": [
                    {
                        "symbol": plan.symbol,
                        "side": plan.side,
                        "quantity": plan.quantity,
                        "realized_pnl": round(float(plan.realized_pnl), 8),
                        "close_order_id": plan.close_order_id,
                        "exchange_order_id": getattr(plan, "exchange_order_id", None),
                        "closed_at": _iso(plan.closed_at),
                        "classification": classifications_by_close_order_id.get(
                            str(plan.close_order_id or ""), {}
                        ),
                    }
                    for plan in plans[:5]
                ],
            },
            evidence=[
                {"label": "候选平仓单", "value": report.candidate_order_count},
                {"label": "已扫描平仓单", "value": report.scanned_order_count},
                {"label": "缺失闭仓", "value": missing},
                {
                    "label": "可自动修复",
                    "value": repairable_count,
                },
                {"label": "Manual review", "value": manual_review_count},
                {"label": "Unscanned", "value": unscanned_candidate_count},
            ],
            next_actions=[
                "只允许先 dry-run 人工核对，再按 symbol/order-id 精确 apply。",
                "如果缺失不为 0，先不要做策略收益判断，避免训练和盈亏被脏账影响。",
                "如果 scanned_close_order_count 小于 candidate_close_order_count，先跑完整脚本或按订单 ID 分段复核。",
            ],
        )
    )


async def _okx_reconciliation_light_scan(
    *,
    days: int,
    max_close_orders: int | None = None,
) -> Any:
    """Return a fast read-only close-order link summary for dashboard audits.

    The full dry-run repair still lives in
    ``scripts/repair_missing_closed_positions_from_orders.py``.  This dashboard
    path intentionally avoids reconstructing every historical position so a slow
    repair scan does not look like a fresh OKX mismatch.
    """

    lookback_days = max(int(days or 14), 1)
    since = (_now() - timedelta(days=lookback_days)).replace(tzinfo=None)
    max_orders = int(max_close_orders) if max_close_orders is not None else None
    if max_orders is not None and max_orders <= 0:
        max_orders = None
    started_at = _now()
    close_long = and_(
        func.lower(AIDecision.action) == "close_long", func.lower(Order.side) == "sell"
    )
    close_short = and_(
        func.lower(AIDecision.action) == "close_short", func.lower(Order.side) == "buy"
    )
    conditions = [
        func.lower(Order.status) == "filled",
        Order.exchange_order_id.is_not(None),
        Order.exchange_order_id != "",
        Order.decision_id.is_not(None),
        Order.filled_at >= since,
        or_(close_long, close_short),
    ]

    plans: list[Any] = []
    plan_classifications: list[dict[str, Any]] = []
    linked_count = 0
    async with get_session_ctx() as session:
        candidate_order_count = int(
            (
                await session.execute(
                    select(func.count(Order.id))
                    .join(AIDecision, Order.decision_id == AIDecision.id)
                    .where(*conditions)
                )
            ).scalar_one()
            or 0
        )
        stmt = (
            select(Order, AIDecision.action)
            .join(AIDecision, Order.decision_id == AIDecision.id)
            .where(*conditions)
            .order_by(Order.filled_at.desc(), Order.created_at.desc())
        )
        if max_orders is not None:
            stmt = stmt.limit(max_orders)
        rows = list((await session.execute(stmt)).all())
        close_link_index = await _load_position_close_link_index(
            session,
            execution_modes={str(order.execution_mode or "") for order, _action in rows},
            exchange_order_ids={
                str(order.exchange_order_id or "").strip()
                for order, _action in rows
                if str(order.exchange_order_id or "").strip()
            },
        )
        for order, action in rows:
            exchange_order_id = str(order.exchange_order_id or "").strip()
            if not exchange_order_id:
                continue
            if (str(order.execution_mode or ""), exchange_order_id) in close_link_index:
                linked_count += 1
                continue
            close_order_id = int(order.id)
            classification = {
                "status": "manual_review",
                "reason": "close_order_has_no_position_close_exchange_link",
                "close_order_id": close_order_id,
                "close_exchange_order_id": exchange_order_id,
            }
            plan_classifications.append(classification)
            plans.append(
                SimpleNamespace(
                    symbol=normalize_trading_symbol(order.symbol),
                    side="long" if str(action or "").lower() == "close_long" else "short",
                    quantity=round(_safe_float(order.quantity), 8),
                    realized_pnl=0.0,
                    close_order_id=close_order_id,
                    closed_at=order.filled_at or order.created_at,
                    exchange_order_id=exchange_order_id,
                )
            )

    unscanned_count = max(candidate_order_count - len(rows), 0)
    classification_counts = {
        "linked": linked_count,
        "manual_review": len(plans),
        "unscanned": unscanned_count,
    }
    return SimpleNamespace(
        plans=plans,
        lookback_days=lookback_days,
        candidate_order_count=candidate_order_count,
        scanned_order_count=len(rows),
        truncated=bool(max_orders is not None and candidate_order_count > len(rows)),
        max_close_orders=max_orders,
        duration_seconds=round(max((_now() - started_at).total_seconds(), 0.0), 6),
        plan_classifications=plan_classifications,
        classification_counts=classification_counts,
        repairable_count=0,
        manual_review_count=len(plans),
        skipped_candidate_count=0,
        unscanned_candidate_count=unscanned_count,
        scan_mode="light_close_order_link_summary",
    )


async def _load_position_close_link_index(
    session: Any,
    *,
    execution_modes: set[str],
    exchange_order_ids: set[str],
) -> set[tuple[str, str]]:
    if not execution_modes or not exchange_order_ids:
        return set()

    rows = (
        await session.execute(
            select(Position.execution_mode, Position.close_exchange_order_id)
            .where(
                Position.execution_mode.in_(execution_modes),
                Position.close_exchange_order_id.is_not(None),
                Position.close_exchange_order_id != "",
            )
            .limit(5000)
        )
    ).all()
    linked: set[tuple[str, str]] = set()
    for mode, raw_link in rows:
        mode_text = str(mode or "")
        raw_text = str(raw_link or "")
        tokens = _exchange_order_link_tokens(raw_text)
        for exchange_order_id in exchange_order_ids:
            if exchange_order_id in tokens or exchange_order_id == raw_text:
                linked.add((mode_text, exchange_order_id))
    return linked


def _exchange_order_link_tokens(value: str) -> set[str]:
    separators = [",", ";", "|", "\n", "\t", " "]
    tokens = {value.strip()} if value.strip() else set()
    chunk = value
    for separator in separators:
        chunk = chunk.replace(separator, ",")
    tokens.update(part.strip() for part in chunk.split(",") if part.strip())
    return tokens


async def _position_close_link_exists(
    session: Any,
    *,
    execution_mode: str,
    exchange_order_id: str,
) -> bool:
    exact_row = await session.execute(
        select(Position.id)
        .where(
            Position.execution_mode == execution_mode,
            Position.close_exchange_order_id == exchange_order_id,
        )
        .limit(1)
    )
    if exact_row.scalar_one_or_none() is not None:
        return True

    loose_row = await session.execute(
        select(Position.id)
        .where(
            Position.execution_mode == execution_mode,
            Position.close_exchange_order_id.like(
                f"%{_escape_sql_like(exchange_order_id)}%",
                escape="\\",
            ),
        )
        .limit(1)
    )
    return loose_row.scalar_one_or_none() is not None


def _escape_sql_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def _position_fact_link_repair_summary() -> dict[str, Any]:
    try:
        report = await asyncio.wait_for(
            collect_position_fact_link_scan_report(
                days=OKX_POSITION_FACT_LINK_AUDIT_DAYS,
                max_positions=OKX_POSITION_FACT_LINK_AUDIT_MAX_POSITIONS,
            ),
            timeout=5.0,
        )
    except Exception as exc:
        return {
            "available": False,
            "error": safe_error_text(exc, limit=180),
            "read_only": True,
            "live_repair_mutation": False,
        }
    return {
        "available": True,
        "read_only": True,
        "live_repair_mutation": False,
        "lookback_days": report.lookback_days,
        "candidate_link_count": report.candidate_link_count,
        "repairable_count": report.repairable_count,
        "manual_review_count": report.manual_review_count,
        "classification_counts": dict(report.classification_counts),
        "scanned_position_count": report.scanned_position_count,
        "max_positions": report.max_positions,
        "truncated": report.truncated,
        "diagnostics": report.diagnostics[:20],
    }


async def _okx_authoritative_sync_summary() -> dict[str, Any]:
    cached = _cached_okx_authoritative_sync_summary()
    if cached is not None:
        return cached
    timeout_budget = max(
        OKX_AUTHORITATIVE_SYNC_TIMEOUT_SECONDS * 8.0 + 5.0,
        45.0,
    )
    try:
        report = await asyncio.wait_for(
            OkxAuthoritativeSyncService(
                mode="paper",
                lookback_hours=OKX_AUTHORITATIVE_SYNC_AUDIT_HOURS,
                limit=OKX_AUTHORITATIVE_SYNC_AUDIT_LIMIT,
                timeout_seconds=OKX_AUTHORITATIVE_SYNC_TIMEOUT_SECONDS,
            ).collect(),
            timeout=timeout_budget,
        )
        return _store_okx_authoritative_sync_summary(report)
    except Exception as exc:
        report = {
            "status": "warning",
            "read_only": True,
            "audit_only": True,
            "source": "okx_private_api",
            "mode": "paper",
            "okx_pull_available": False,
            "live_repair_mutation": False,
            "can_write_database": False,
            "error": safe_error_text(exc, limit=180),
            "cache": {
                "hit": False,
                "age_seconds": 0.0,
                "ttl_seconds": OKX_AUTHORITATIVE_SYNC_CACHE_TTL_SECONDS,
            },
            "apply_policy": {
                "can_write_database": False,
                "requires_allowlisted_apply": True,
                "requires_backup": True,
            },
        }
        return _store_okx_authoritative_sync_summary(report)


async def _okx_trade_fact_integrity_audit() -> dict[str, Any]:
    try:
        report = await asyncio.wait_for(
            OkxTradeFactIntegrityService(
                lookback_hours=OKX_TRADE_FACT_INTEGRITY_AUDIT_HOURS,
                limit=OKX_TRADE_FACT_INTEGRITY_AUDIT_LIMIT,
            ).audit(),
            timeout=8.0,
        )
    except Exception as exc:
        daily_report = _load_okx_daily_reconciliation_report_summary()
        return _audit_card(
            "okx_trade_fact_integrity",
            "OKX/本地交易事实一致性",
            "warning",
            "交易事实一致性巡检执行失败；当前不能证明口径已干净，需要先恢复只读审计。",
            details={
                "error": safe_error_text(exc, limit=180),
                "hard_failure": True,
                "read_only": True,
                "live_repair_mutation": False,
                "daily_reconciliation_report": daily_report,
            },
            evidence=[{"label": "审计错误", "value": 1}],
            next_actions=[
                "先恢复只读审计链路，再判断 orders、positions、OKX raw 回报和训练样本是否一致。",
                "审计不可用时不要直接批量修历史仓位，也不要把近期收益样本作为放大仓位依据。",
            ],
        )

    details = dict(report if isinstance(report, dict) else {})
    details["read_only"] = True
    details["live_repair_mutation"] = False
    details["can_apply_historical_repair"] = False
    link_repair = await _position_fact_link_repair_summary()
    details["position_fact_link_repair"] = link_repair
    authoritative_sync = await _okx_authoritative_sync_summary()
    authoritative_sync["live_repair_mutation"] = False
    authoritative_sync["can_write_database"] = False
    details["okx_authoritative_sync"] = authoritative_sync
    runtime_status = _load_trading_runtime_status_for_audit()
    runtime_entry_gate = _okx_runtime_entry_gate_summary(runtime_status)
    details["runtime_okx_entry_gate"] = runtime_entry_gate
    daily_report = _load_okx_daily_reconciliation_report_summary()
    details["daily_reconciliation_report"] = daily_report
    status = str(details.get("status") or "ok")
    critical_count = int(details.get("critical_count") or 0)
    warning_count = int(details.get("warning_count") or 0)
    link_candidate_count = (
        int(link_repair.get("candidate_link_count") or 0)
        if bool(link_repair.get("available"))
        else 0
    )
    unresolved_link_candidate_count = _okx_unresolved_link_candidate_count(details, link_repair)
    details["unresolved_position_fact_link_candidate_count"] = unresolved_link_candidate_count
    authoritative_issue_count = int(authoritative_sync.get("issue_count") or 0)
    authoritative_manual_review_count = int(authoritative_sync.get("manual_review_count") or 0)
    authoritative_repairable_count = int(authoritative_sync.get("repairable_count") or 0)
    authoritative_pull_available = bool(authoritative_sync.get("okx_pull_available"))
    if status == "ok" and unresolved_link_candidate_count > 0:
        status = "warning"
        warning_count = max(warning_count, 1)
    if authoritative_issue_count or not authoritative_pull_available:
        if status == "ok":
            status = "warning"
        warning_count = max(warning_count, authoritative_issue_count or 1)
    if runtime_entry_gate.get("entry_blocked") is True and status == "ok":
        status = "warning"
        warning_count = max(warning_count, 1)
    if bool(daily_report.get("stale")) and status == "ok":
        status = "warning"
        warning_count = max(warning_count, 1)
    elif bool(daily_report.get("requires_attention")):
        if status == "ok":
            status = "warning"
        warning_count = max(warning_count, 1)
    if critical_count:
        summary = (
            "发现 OKX 原始成交、订单、持仓之间存在关键口径不一致；需先完成备份和 dry-run 对账。"
        )
    elif runtime_entry_gate.get("entry_blocked") is True:
        summary = "OKX 自动同步当前阻断新开仓；需先恢复 OKX/本地当前状态一致，再允许新增风险。"
    elif warning_count:
        summary = "发现 OKX/本地交易事实存在需要关注项；暂不应自动写历史数据。"
    else:
        summary = "OKX 原始成交、订单和持仓口径在巡检窗口内一致。"
    return _audit_card(
        "okx_trade_fact_integrity",
        "OKX/本地交易事实一致性",
        status,
        summary,
        details=details,
        evidence=[
            {
                "label": "Daily report",
                "value": (
                    "stale"
                    if bool(daily_report.get("stale"))
                    else daily_report.get("status") or "unknown"
                ),
            },
            {
                "label": "Can train",
                "value": bool(daily_report.get("can_refresh_training")),
            },
            {
                "label": "OKX API facts",
                "value": (
                    int(authoritative_sync.get("okx_fill_order_count") or 0)
                    + int(authoritative_sync.get("okx_position_count") or 0)
                ),
            },
            {"label": "Manual review", "value": authoritative_manual_review_count},
            {"label": "Repairable", "value": authoritative_repairable_count},
            {
                "label": "Entry blocked",
                "value": bool(runtime_entry_gate.get("entry_blocked") is True),
            },
            {"label": "检查订单", "value": int(details.get("checked_orders") or 0)},
            {"label": "检查持仓", "value": int(details.get("checked_positions") or 0)},
            {"label": "关键问题", "value": critical_count},
            {"label": "关注项", "value": warning_count},
        ],
        next_actions=(
            [
                "先查看 runtime_okx_entry_gate.reason 与 last_samples，确认 OKX 当前持仓、挂单、成交与本地是否已恢复一致。",
                "阻断期间只允许平仓、止损止盈、仓位复核等降低风险动作，不允许新增开仓扩大错账。",
                "OKX 自动同步恢复 ok 且 requires_attention 清零后，再进入新开仓观察。",
            ]
            if runtime_entry_gate.get("entry_blocked") is True
            else (
                [
                    "按 issue 的 order_id、decision_id、position_id 对 OKX raw 回报、orders、positions 做备份和 dry-run 对账。",
                    "关键口径未清洁前，不使用相关收益样本训练 server_profit，也不把异常盈利单作为放大仓位模板。",
                    "确认 OKX instId、contract_size、filled_contracts、base quantity、entry/exit price 同源后，再制定精确修复脚本。",
                ]
                if critical_count
                or warning_count
                or unresolved_link_candidate_count
                or authoritative_issue_count
                else [
                    "保持只读巡检常开，后续历史修复或同步逻辑改动后必须先看这张卡是否仍为正常。",
                ]
            )
        ),
    )


def _okx_unresolved_link_candidate_count(
    details: dict[str, Any],
    link_repair: dict[str, Any],
) -> int:
    candidate_count = int(link_repair.get("candidate_link_count") or 0)
    if candidate_count <= 0:
        return 0
    covered_residual_positions = {
        int(issue.get("position_id"))
        for issue in _safe_list(details.get("issues"))
        if isinstance(issue, dict)
        and issue.get("kind") == "superseded_position_residual"
        and issue.get("severity") == "info"
        and issue.get("position_id") is not None
    }
    diagnostics = [
        item
        for item in _safe_list(link_repair.get("diagnostics"))
        if isinstance(item, dict)
    ]
    if diagnostics and all(
        int(item.get("position_id") or 0) in covered_residual_positions
        for item in diagnostics
    ):
        return 0
    return candidate_count


async def _position_price_integrity_audit() -> dict[str, Any]:
    from web_dashboard.api import dashboard as dashboard_api

    split_rows: list[dict[str, Any]] = []
    local_only_rows: list[dict[str, Any]] = []
    exchange_only_rows: list[dict[str, Any]] = []
    checked_modes: list[str] = []
    unavailable_modes: list[dict[str, str]] = []
    local_open_count = 0
    exchange_open_count = 0
    root_cause_counts: Counter[str] = Counter()
    okx_pos_side_counts: Counter[str] = Counter()
    okx_side_inference_counts: Counter[str] = Counter()

    for mode in ("paper", "live"):
        executor = dashboard_api._dashboard_okx_executor_for_mode(mode)
        if not executor:
            continue
        checked_modes.append(mode)
        try:
            exchange_positions = await asyncio.wait_for(
                executor.get_positions_strict(),
                timeout=1.8,
            )
        except Exception as exc:
            unavailable_modes.append({"mode": mode, "error": safe_error_text(exc, limit=120)})
            continue

        exchange_snapshots: dict[tuple[str, str], dict[str, Any]] = {}
        for raw_position in exchange_positions or []:
            snapshot = parse_exchange_position_snapshot(
                raw_position,
                symbol_normalizer=normalize_trading_symbol,
            )
            if not snapshot:
                continue
            exchange_snapshots[(str(snapshot["symbol"]), str(snapshot["side"]))] = snapshot
            okx_pos_side_counts[str(snapshot.get("raw_pos_side") or "unknown")] += 1
            okx_side_inference_counts[str(snapshot.get("side_inference") or "unknown")] += 1
        exchange_open_count += len(exchange_snapshots)

        async with get_session_ctx() as session:
            local_positions = list(
                (
                    await session.execute(
                        select(Position).where(
                            Position.execution_mode == mode,
                            Position.is_open.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            )
        local_open_count += len(local_positions)
        local_keys: set[tuple[str, str]] = set()

        for position in local_positions:
            key = (
                normalize_trading_symbol(position.symbol),
                str(position.side or "").lower(),
            )
            local_keys.add(key)
            snapshot = exchange_snapshots.get(key)
            if not snapshot:
                root_cause_counts["local_open_position_missing_on_okx"] += 1
                local_only_rows.append(
                    {
                        "mode": mode,
                        "position_id": int(position.id or 0),
                        "symbol": key[0],
                        "side": key[1],
                        "local_quantity": round(_safe_float(position.quantity), 8),
                        "local_entry_price": round(_safe_float(position.entry_price), 8),
                        "local_price": round(_safe_float(position.current_price), 8),
                        "local_unrealized_pnl": round(_safe_float(position.unrealized_pnl), 8),
                        "root_cause": "local_open_position_missing_on_okx",
                    }
                )
                continue
            valuation = exchange_position_display_valuation(
                snapshot,
                key[1],
                fallback_current_price=position.current_price,
                fallback_unrealized_pnl=position.unrealized_pnl,
                fallback_entry_price=position.entry_price,
                fallback_quantity=position.quantity,
            )
            local_price = _safe_float(position.current_price)
            okx_price = _safe_float(valuation.get("current_price"))
            local_pnl = _safe_float(position.unrealized_pnl)
            okx_pnl = _safe_float(valuation.get("unrealized_pnl"))
            price_gap = (
                _relative_gap(local_price, okx_price) if local_price > 0 and okx_price > 0 else 0.0
            )
            pnl_gap = abs(local_pnl - okx_pnl)
            if price_gap < POSITION_PRICE_SPLIT_WARN_PCT and pnl_gap < POSITION_PNL_SPLIT_WARN_USDT:
                continue
            root_cause = (
                "okx_upl_mismatch"
                if valuation.get("pnl_source") == "okx_position_upl"
                else "mark_price_recomputed_pnl_mismatch"
            )
            if price_gap >= POSITION_PRICE_SPLIT_WARN_PCT:
                root_cause = "mark_price_mismatch"
            root_cause_counts[root_cause] += 1
            split_rows.append(
                {
                    "mode": mode,
                    "symbol": key[0],
                    "side": key[1],
                    "local_price": round(local_price, 8),
                    "okx_price": round(okx_price, 8),
                    "price_gap_pct": round(price_gap * 100, 4),
                    "local_unrealized_pnl": round(local_pnl, 8),
                    "okx_unrealized_pnl": round(okx_pnl, 8),
                    "pnl_gap_usdt": round(pnl_gap, 8),
                    "pnl_source": valuation.get("pnl_source"),
                    "root_cause": root_cause,
                    "okx_entry_price": round(_safe_float(valuation.get("entry_price")), 8),
                    "okx_quantity": round(_safe_float(valuation.get("quantity")), 8),
                    "okx_contracts": round(_safe_float(snapshot.get("contracts")), 8),
                    "okx_contract_size": round(_safe_float(snapshot.get("contract_size")), 8),
                    "okx_raw_symbol": snapshot.get("raw_symbol"),
                    "okx_ccxt_symbol": snapshot.get("ccxt_symbol"),
                    **_okx_position_snapshot_evidence(snapshot),
                }
            )
        for key, snapshot in sorted(exchange_snapshots.items()):
            if key in local_keys:
                continue
            root_cause_counts["okx_open_position_missing_locally"] += 1
            valuation = exchange_position_display_valuation(
                snapshot,
                key[1],
                fallback_current_price=0.0,
                fallback_unrealized_pnl=0.0,
                fallback_entry_price=0.0,
                fallback_quantity=0.0,
            )
            exchange_only_rows.append(
                {
                    "mode": mode,
                    "symbol": key[0],
                    "side": key[1],
                    "okx_price": round(_safe_float(valuation.get("current_price")), 8),
                    "okx_entry_price": round(_safe_float(valuation.get("entry_price")), 8),
                    "okx_quantity": round(_safe_float(valuation.get("quantity")), 8),
                    "okx_unrealized_pnl": round(_safe_float(valuation.get("unrealized_pnl")), 8),
                    "pnl_source": valuation.get("pnl_source"),
                    "okx_contracts": round(_safe_float(snapshot.get("contracts")), 8),
                    "okx_contract_size": round(_safe_float(snapshot.get("contract_size")), 8),
                    "okx_raw_symbol": snapshot.get("raw_symbol"),
                    "okx_ccxt_symbol": snapshot.get("ccxt_symbol"),
                    **_okx_position_snapshot_evidence(snapshot),
                    "root_cause": "okx_open_position_missing_locally",
                }
            )

    mismatch_count = len(split_rows) + len(local_only_rows) + len(exchange_only_rows)
    root_cause_summary = {
        "status": "dirty" if mismatch_count else "incomplete" if unavailable_modes else "clean",
        "mismatch_count": mismatch_count,
        "split_count": len(split_rows),
        "local_only_count": len(local_only_rows),
        "exchange_only_count": len(exchange_only_rows),
        "root_cause_counts": dict(root_cause_counts),
        "okx_pos_side_counts": dict(okx_pos_side_counts),
        "okx_side_inference_counts": dict(okx_side_inference_counts),
        "read_only": True,
        "audit_only": True,
        "live_repair_mutation": False,
        "training_data_policy": "quarantine_untrusted_position_facts_until_okx_local_match",
    }
    status = _status_from_counts(critical=bool(mismatch_count), warning=bool(unavailable_modes))
    return _audit_card(
        "position_price_integrity",
        "持仓价格一致性",
        status,
        (
            "发现平台持仓价/浮盈与 OKX 持仓快照不一致，可能影响持仓分析、平仓和训练标签。"
            if split_rows
            else (
                "部分模式暂时无法读取 OKX 持仓快照。"
                if unavailable_modes
                else "平台持仓价格与 OKX 持仓快照一致。"
            )
        ),
        details={
            "checked_modes": checked_modes,
            "unavailable_modes": unavailable_modes,
            "local_open_positions": local_open_count,
            "exchange_open_positions": exchange_open_count,
            "mismatch_count": mismatch_count,
            "split_count": len(split_rows),
            "local_only_count": len(local_only_rows),
            "exchange_only_count": len(exchange_only_rows),
            "price_gap_warn_pct": POSITION_PRICE_SPLIT_WARN_PCT * 100,
            "pnl_gap_warn_usdt": POSITION_PNL_SPLIT_WARN_USDT,
            "root_cause_summary": root_cause_summary,
            "okx_pos_side_counts": dict(okx_pos_side_counts),
            "okx_side_inference_counts": dict(okx_side_inference_counts),
            "splits": split_rows[:12],
            "local_only_positions": local_only_rows[:12],
            "exchange_only_positions": exchange_only_rows[:12],
            "read_only": True,
            "audit_only": True,
            "live_repair_mutation": False,
        },
        evidence=[
            {"label": "价格/浮盈分裂", "value": len(split_rows)},
            {"label": "本地多余持仓", "value": len(local_only_rows)},
            {"label": "OKX多余持仓", "value": len(exchange_only_rows)},
            {"label": "本地开仓", "value": local_open_count},
            {"label": "OKX持仓", "value": exchange_open_count},
        ],
        next_actions=[
            "若出现分裂，先运行 OKX 同步并复查持仓页；不要基于分裂数据调整策略参数。",
            "若同一币种反复分裂，检查 OKX 字段解析、合约面值 ctVal、行情缓存和持仓同步任务。",
        ],
    )


async def _market_data_audit() -> dict[str, Any]:
    async with get_session_ctx() as session:
        kline_rows = (
            await session.execute(
                select(
                    Kline.timeframe,
                    func.count(Kline.id),
                    func.count(func.distinct(Kline.symbol)),
                    func.max(Kline.open_time),
                )
                .where(Kline.timeframe.in_(EXPECTED_KLINE_TIMEFRAMES))
                .group_by(Kline.timeframe)
            )
        ).all()
        ticker_row = (
            await session.execute(
                select(
                    func.count(Ticker.id),
                    func.max(func.coalesce(Ticker.updated_at, Ticker.created_at)),
                )
            )
        ).one()
    by_timeframe = {str(row[0]): row for row in kline_rows}
    rows: list[dict[str, Any]] = []
    stale_timeframes: list[str] = []
    missing_timeframes: list[str] = []
    for timeframe in EXPECTED_KLINE_TIMEFRAMES:
        row = by_timeframe.get(timeframe)
        count = int(row[1] or 0) if row else 0
        symbols = int(row[2] or 0) if row else 0
        latest = row[3] if row else None
        age = _age_seconds(latest)
        missing = count <= 0
        stale = bool(age is None or age > KLINE_STALE_LIMIT_SECONDS[timeframe])
        if missing:
            missing_timeframes.append(timeframe)
        elif stale:
            stale_timeframes.append(timeframe)
        rows.append(
            {
                "timeframe": timeframe,
                "rows": count,
                "symbols": symbols,
                "latest_at": _iso(latest),
                "age_seconds": round(age, 3) if age is not None else None,
                "missing": missing,
                "stale": stale,
            }
        )
    ticker_age = _age_seconds(ticker_row[1])
    ticker_stale = ticker_age is None or ticker_age > 600
    covered_timeframes = [
        row["timeframe"] for row in rows if not row["missing"] and not row["stale"]
    ]
    warmup_observing = bool(
        covered_timeframes and (missing_timeframes or stale_timeframes or ticker_stale)
    )
    status = _status_from_counts(
        critical=bool(missing_timeframes),
        warning=bool(stale_timeframes) or ticker_stale,
    )
    return _audit_card(
        "market_data",
        "行情与 K线",
        status,
        "行情/K线覆盖正常。" if status == "ok" else "行情或 K线覆盖存在缺失/过期。",
        details={
            "ticker_count": int(ticker_row[0] or 0),
            "ticker_latest_at": _iso(ticker_row[1]),
            "ticker_age_seconds": round(ticker_age, 3) if ticker_age is not None else None,
            "ticker_stale": ticker_stale,
            "klines": rows,
            "missing_timeframes": missing_timeframes,
            "stale_timeframes": stale_timeframes,
            "covered_timeframes": covered_timeframes,
            "warmup_observing": warmup_observing,
        },
        evidence=[{"label": f"{row['timeframe']} 币种", "value": row["symbols"]} for row in rows],
        next_actions=[
            "先查 DataService K线覆盖刷新任务和 OKX REST 错误。",
            "K线异常时不要先调整策略参数。",
        ],
    )


async def _strategy_quality_audit() -> dict[str, Any]:
    since = _now() - timedelta(hours=AUDIT_WINDOWS["strategy_hours"])
    async with get_session_ctx() as session:
        decisions = list(
            (
                await session.execute(
                    select(AIDecision)
                    .where(AIDecision.created_at >= since)
                    .order_by(AIDecision.created_at.desc())
                    .limit(500)
                )
            )
            .scalars()
            .all()
        )
        closed_positions = list(
            (
                await session.execute(
                    select(Position)
                    .where(Position.is_open.is_(False), Position.closed_at >= since)
                    .order_by(Position.closed_at.desc())
                    .limit(200)
                )
            )
            .scalars()
            .all()
        )
        recent_positions = list(
            (
                await session.execute(
                    select(Position)
                    .where(Position.created_at >= since)
                    .order_by(Position.created_at.desc())
                    .limit(300)
                )
            )
            .scalars()
            .all()
        )
    actions = Counter(str(row.action or "unknown").lower() for row in decisions)
    entry_decisions = [
        row for row in decisions if str(row.action or "").lower() in {"long", "short"}
    ]
    blocked_reasons: Counter[str] = Counter()
    zero_expected = 0
    negative_expected = 0
    weak_shadow_executed = []
    positive_weak_shadow_executed = []
    short_conservative_adjustments = []
    short_released_adjustments = []
    for row in entry_decisions:
        raw = row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {}
        opportunity = raw.get("opportunity_score") if isinstance(raw, dict) else {}
        if isinstance(opportunity, dict):
            net = opportunity.get("expected_net_return_pct")
            try:
                net_value = float(net)
                if abs(net_value) < 1e-9:
                    zero_expected += 1
                elif net_value < 0:
                    negative_expected += 1
            except (TypeError, ValueError):
                pass
            evidence_score = opportunity.get("evidence_score")
            if isinstance(evidence_score, dict):
                evidence_tier = str(evidence_score.get("tier") or "")
                short_adjustment = evidence_score.get("short_evidence_adjustment")
                if isinstance(short_adjustment, dict) and str(row.action or "").lower() == "short":
                    adjustment_sample = {
                        "decision_id": row.id,
                        "symbol": row.symbol,
                        "mode": short_adjustment.get("mode"),
                        "score_offset": round(_safe_float(short_adjustment.get("score_offset")), 6),
                        "size_multiplier": round(
                            _safe_float(short_adjustment.get("size_multiplier"), 1.0), 6
                        ),
                        "expected_net_return_pct": round(_safe_float(net), 6),
                        "created_at": _iso(row.created_at),
                    }
                    if short_adjustment.get("mode") == "strong_current_short_evidence":
                        short_released_adjustments.append(adjustment_sample)
                    elif short_adjustment.get("mode") == "conservative_short_evidence":
                        short_conservative_adjustments.append(adjustment_sample)
                if evidence_tier in {"weak_conflict_probe", "degraded_missing_probe"} and bool(
                    getattr(row, "was_executed", False)
                ):
                    sample = {
                        "decision_id": row.id,
                        "symbol": row.symbol,
                        "action": row.action,
                        "evidence_tier": evidence_tier,
                        "expected_net_return_pct": round(_safe_float(net), 6),
                        "position_size_pct": round(_safe_float(row.position_size_pct), 8),
                        "created_at": _iso(row.created_at),
                    }
                    weak_shadow_executed.append(sample)
                    if _safe_float(net) > 0:
                        positive_weak_shadow_executed.append(sample)
        reason = str(getattr(row, "execution_reason", "") or "").strip()
        if reason:
            blocked_reasons[reason[:80]] += 1
    position_notionals = sorted(
        abs(_safe_float(pos.quantity) * _safe_float(pos.entry_price))
        for pos in recent_positions
        if abs(_safe_float(pos.quantity) * _safe_float(pos.entry_price)) > 0
    )
    notional_stats: dict[str, Any] = {"count": len(position_notionals)}
    micro_notional_floor = 0.0
    micro_position_count = 0
    if position_notionals:
        median_index = len(position_notionals) // 2
        median_notional = position_notionals[median_index]
        micro_notional_floor = max(5.0, median_notional * 0.35)
        micro_position_count = sum(
            1 for value in position_notionals if value <= micro_notional_floor
        )
        notional_stats.update(
            {
                "min": round(position_notionals[0], 6),
                "avg": round(sum(position_notionals) / len(position_notionals), 6),
                "median": round(median_notional, 6),
                "max": round(position_notionals[-1], 6),
                "micro_observation_floor_usdt": round(micro_notional_floor, 6),
                "micro_position_count": micro_position_count,
                "audit_only": True,
            }
        )
    trusted_closed_positions = [
        pos for pos in closed_positions if closed_position_trade_fact_trusted(pos)
    ]
    quarantined_closed_position_count = len(closed_positions) - len(trusted_closed_positions)
    fast_loss_positions = []
    fast_loss_micro_positions = []
    for pos in trusted_closed_positions:
        created = pos.created_at
        closed = pos.closed_at
        if not isinstance(created, datetime) or not isinstance(closed, datetime):
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=UTC)
        hold_minutes = max((closed - created).total_seconds() / 60.0, 0.0)
        pnl = float(pos.realized_pnl or 0.0)
        if hold_minutes <= 10 and pnl < 0:
            notional = abs(_safe_float(pos.quantity) * _safe_float(pos.entry_price))
            sample = {
                "id": pos.id,
                "symbol": pos.symbol,
                "side": pos.side,
                "hold_minutes": round(hold_minutes, 3),
                "realized_pnl": round(pnl, 8),
                "notional_usdt": round(notional, 6),
                "closed_at": _iso(closed),
            }
            fast_loss_positions.append(sample)
            if micro_notional_floor > 0 and notional <= micro_notional_floor:
                fast_loss_micro_positions.append(sample)
    warning = bool(
        fast_loss_positions
        or weak_shadow_executed
        or (entry_decisions and negative_expected >= len(entry_decisions) * 0.7)
    )
    short_adjustment_evidence = [
        {"label": "做空保守修正", "value": len(short_conservative_adjustments)},
        {"label": "做空强证据放开", "value": len(short_released_adjustments)},
    ]
    short_adjustment_next_actions = [
        "做空保守修正占比高时，先看净收益、盈利质量、亏损概率、尾部风险和反向证据，不再盲目放大空单。"
    ]
    return _audit_card(
        "strategy_quality",
        "策略质量",
        "warning" if warning else "ok",
        (
            "存在快亏平、弱证据误执行或多数开仓候选净收益为负。"
            if warning
            else "最近策略质量未发现硬异常。"
        ),
        details={
            "window_hours": AUDIT_WINDOWS["strategy_hours"],
            "decision_count": len(decisions),
            "action_counts": dict(actions),
            "entry_decision_count": len(entry_decisions),
            "zero_expected_net_count": zero_expected,
            "negative_expected_net_count": negative_expected,
            "weak_shadow_executed_count": len(weak_shadow_executed),
            "positive_weak_shadow_executed_count": len(positive_weak_shadow_executed),
            "weak_shadow_executed_samples": weak_shadow_executed[:10],
            "short_conservative_adjustment_count": len(short_conservative_adjustments),
            "short_released_adjustment_count": len(short_released_adjustments),
            "short_conservative_adjustment_samples": short_conservative_adjustments[:10],
            "short_released_adjustment_samples": short_released_adjustments[:10],
            "position_notional_stats": notional_stats,
            "micro_position_count": micro_position_count,
            "closed_position_count": len(closed_positions),
            "trusted_closed_position_count": len(trusted_closed_positions),
            "quarantined_closed_position_count": quarantined_closed_position_count,
            "trade_fact_policy": "strategy_quality_fast_loss_uses_trusted_closed_facts_only",
            "fast_loss_positions": fast_loss_positions[:10],
            "fast_loss_micro_positions": fast_loss_micro_positions[:10],
            "top_blocked_reasons": [
                {"reason": reason, "count": count}
                for reason, count in blocked_reasons.most_common(8)
            ],
        },
        evidence=short_adjustment_evidence
        + [
            {"label": "开仓候选", "value": len(entry_decisions)},
            {"label": "负净收益", "value": negative_expected},
            {"label": "弱证据已执行", "value": len(weak_shadow_executed)},
            {"label": "微小仓观测", "value": micro_position_count},
            {"label": "快亏平", "value": len(fast_loss_positions)},
        ],
        next_actions=short_adjustment_next_actions
        + [
            "负净收益占比高时先查成本/滑点/点差，不直接放宽开仓。",
            "弱证据已执行不应为正；若出现，先查 entry_evidence 与执行器契约。",
            "微小仓快亏平出现时先查仓位 sizing 与新仓释放保护，不继续扩大样本污染。",
            "快亏平出现时先看执行详情的风控步骤和 OKX 平仓来源。",
        ],
    )


async def _model_expert_health_audit() -> dict[str, Any]:
    try:
        report = await ModelExpertHealthService().report(
            hours=MODEL_EXPERT_AUDIT_HOURS,
            limit=MODEL_EXPERT_AUDIT_LIMIT,
        )
    except Exception as exc:
        return _audit_card(
            "model_expert_health",
            "模型/专家体检",
            "warning",
            "模型/专家体检报告读取失败。",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
            next_actions=[
                "先检查 ai_decisions.raw_llm_response.model_timings 与专家返回结构是否正常写入。"
            ],
        )
    summary = _safe_dict(report.get("summary"))
    counts = _safe_dict(summary.get("recommended_state_counts"))
    components = _safe_dict(report.get("components"))
    concerning_states = {"reduce", "shadow_only", "disable", "replace"}
    concerning_count = sum(int(counts.get(state) or 0) for state in concerning_states)
    critical_count = int(counts.get("disable") or 0) + int(counts.get("replace") or 0)
    priority = {"disable": 0, "replace": 1, "reduce": 2, "shadow_only": 3}
    top_components = sorted(
        (
            {
                "name": name,
                "recommended_state": _safe_dict(component).get("recommended_state"),
                "state_reasons": _safe_dict(component).get("state_reasons") or [],
                "stability": _safe_dict(component).get("stability"),
            }
            for name, component in components.items()
            if _safe_dict(component).get("recommended_state") in concerning_states
        ),
        key=lambda item: (
            priority.get(str(item.get("recommended_state")), 9),
            str(item.get("name") or ""),
        ),
    )[:8]
    status = _status_from_counts(warning=bool(concerning_count))
    summary_text = (
        f"发现 {concerning_count} 个模型/专家需要降权、影子观察或禁用复核。"
        if concerning_count
        else "模型/专家体检暂未发现拖累项。"
    )
    return _audit_card(
        "model_expert_health",
        "模型/专家体检",
        status,
        summary_text,
        details={
            "audit_only": bool(report.get("audit_only", True)),
            "live_weight_mutation": bool(report.get("live_weight_mutation", False)),
            "component_count": int(summary.get("components") or len(components)),
            "recommended_state_counts": counts,
            "top_components": top_components,
            "windows_hours": report.get("windows_hours") or [],
        },
        evidence=[
            {"label": "组件数", "value": int(summary.get("components") or len(components))},
            {"label": "需降权", "value": int(counts.get("reduce") or 0)},
            {"label": "只影子", "value": int(counts.get("shadow_only") or 0)},
            {"label": "需禁用/替换复核", "value": critical_count},
        ],
        next_actions=[
            "本卡只读体检，不直接调整真实模型/专家权重。",
            "出现 reduce/shadow_only/disable 时，先看 24/72 小时参与、采纳、收益、JSON 错误和未返回率。",
            "样本不足的组件只能标记观察中，不因单次盈亏直接禁用或提权。",
        ],
    )


async def _model_expert_competition_audit() -> dict[str, Any]:
    try:
        report = await ModelExpertCompetitionService().report(
            hours=MODEL_EXPERT_AUDIT_HOURS,
            limit=MODEL_EXPERT_AUDIT_LIMIT,
        )
    except Exception as exc:
        return _audit_card(
            "model_expert_competition",
            "模型/专家竞赛",
            "warning",
            "模型/专家竞赛报告读取失败。",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
            next_actions=[
                "先确认 ai_decisions outcome 与 raw_llm_response.experts/model_timings 是否可读取。"
            ],
        )
    baseline = _safe_dict(report.get("baseline"))
    layers = _safe_dict(report.get("layers"))
    competitors = _safe_dict(report.get("competitors"))
    action_counts = Counter(
        str(_safe_dict(row).get("recommended_weight_action") or "unknown")
        for row in competitors.values()
    )
    blockers = (
        report.get("blocking_reasons") if isinstance(report.get("blocking_reasons"), list) else []
    )
    concerning = sum(
        int(action_counts.get(action) or 0) for action in ("reduce_shadow_weight", "pause_shadow")
    )
    top_competitors = sorted(
        (
            {
                "name": name,
                "recommended_weight_action": _safe_dict(row).get("recommended_weight_action"),
                "baseline_delta": _safe_dict(row).get("baseline_delta"),
                "can_apply_live_weight": False,
            }
            for name, row in competitors.items()
        ),
        key=lambda item: abs(
            _safe_float(_safe_dict(item.get("baseline_delta")).get("net_pnl_pct"), 0.0)
        ),
        reverse=True,
    )[:8]
    status = _status_from_counts(warning=bool(blockers or concerning))
    summary = (
        "竞赛缺少 baseline 样本，暂不能用于权重判断。"
        if blockers
        else (
            f"竞赛发现 {concerning} 个组件相对 baseline 落后或不稳定。"
            if concerning
            else "模型/专家竞赛已形成 baseline 对比，暂无拖累项。"
        )
    )
    return _audit_card(
        "model_expert_competition",
        "模型/专家竞赛",
        status,
        summary,
        details={
            "audit_only": bool(report.get("audit_only", True)),
            "live_weight_mutation": bool(report.get("live_weight_mutation", False)),
            "can_apply_live_weight": bool(report.get("can_apply_live_weight", False)),
            "baseline": baseline,
            "layers": layers,
            "blocking_reasons": blockers,
            "recommended_weight_action_counts": dict(action_counts),
            "top_competitors": top_competitors,
        },
        evidence=[
            {"label": "baseline样本", "value": int(baseline.get("sample_count") or 0)},
            {"label": "竞赛组件", "value": len(competitors)},
            {
                "label": "可提影子权重",
                "value": int(action_counts.get("increase_shadow_weight") or 0),
            },
            {"label": "需降/暂停影子", "value": concerning},
        ],
        next_actions=[
            "没有 baseline 对比前不得调整真实权重。",
            "C3 只输出 shadow/candidate 竞赛建议，不直接改主链路。",
            "后续 C5 动态路由必须消费本报告的 shadow/sim/live 分层统计。",
        ],
    )


async def _model_dynamic_routing_audit() -> dict[str, Any]:
    try:
        report = _safe_dynamic_routing_report(
            await ModelDynamicRoutingService().report(
                hours=MODEL_EXPERT_AUDIT_HOURS,
                limit=MODEL_EXPERT_AUDIT_LIMIT,
            )
        )
    except Exception as exc:
        return _audit_card(
            "model_dynamic_routing",
            "模型动态路由",
            "warning",
            "模型动态路由报告读取失败；保持全量专家主链路，不启用路由变更。",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
            next_actions=[
                "先确认 ai_decisions.raw_llm_response.dynamic_model_routing 是否正常写入。"
            ],
        )
    summary = _safe_dict(report.get("summary"))
    blockers = _safe_dict(report.get("blocking_reason_counts"))
    observations = _safe_dict(report.get("safety_observations"))
    unsafe_attempts = int(summary.get("unsafe_live_mutation_attempts") or 0)
    weak_executed = int(observations.get("weak_evidence_executed_count") or 0)
    route_count = int(summary.get("route_plan_count") or 0)
    promotion_gate = _safe_dict(report.get("promotion_gate"))
    warning = bool(unsafe_attempts or weak_executed or blockers or route_count == 0)
    return _audit_card(
        "model_dynamic_routing",
        "模型动态路由",
        "warning" if warning else "ok",
        (
            "动态路由仍处于影子/观察阶段，尚不能替换主链路。"
            if warning
            else "动态路由影子报告正常，未发现阻塞项。"
        ),
        details={
            "audit_only": True,
            "live_route_mutation": False,
            "can_apply_live_route": False,
            "summary": summary,
            "blocking_reason_counts": blockers,
            "safety_observations": observations,
            "unsafe_live_mutation_attempts": unsafe_attempts,
            "promotion_gate": promotion_gate,
        },
        evidence=[
            {"label": "路由计划", "value": route_count},
            {"label": "影子计划", "value": int(summary.get("shadow_only_count") or 0)},
            {"label": "理论少调用", "value": int(summary.get("estimated_call_reduction") or 0)},
            {"label": "弱证据已执行", "value": weak_executed},
        ],
        next_actions=[
            "没有 C2/C3 baseline 和线上观察前，不得把动态路由应用到真实专家调用。",
            "不得为了降延迟跳过 risk_expert 或必要风控复核。",
            "若弱证据执行或快亏平增加，继续保持 shadow_only 并先诊断执行质量。",
        ],
    )


async def _high_risk_review_audit() -> dict[str, Any]:
    try:
        report = _safe_high_risk_review_report(
            await HighRiskReviewAuditService().report(
                hours=MODEL_EXPERT_AUDIT_HOURS,
                limit=MODEL_EXPERT_AUDIT_LIMIT,
            )
        )
    except Exception as exc:
        return _audit_card(
            "high_risk_review_audit",
            "High-risk review",
            "warning",
            "High-risk review audit failed; keep hard-review gates fail-closed.",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
        )
    unsafe = int(report.get("executed_without_required_review_count") or 0)
    hard_required = int(report.get("hard_review_required_count") or 0)
    blocked = int(report.get("blocked_count") or 0)
    status = "critical" if unsafe else "warning" if hard_required and not blocked else "ok"
    summary = (
        "High-risk entries executed without completed approval; inspect gate wiring before live expansion."
        if unsafe
        else (
            "High-risk review gate is active and blocking or approving required reviews."
            if hard_required
            else "High-risk review audit found no required hard-review entries in the current window."
        )
    )
    return _audit_card(
        "high_risk_review_audit",
        "High-risk review",
        status,
        summary,
        details={
            "audit_only": True,
            "read_only": True,
            "live_entry_mutation": False,
            "can_bypass_risk_controls": False,
            "can_force_open": False,
            "summary": {
                "entry_decision_count": report.get("entry_decision_count"),
                "review_payload_count": report.get("review_payload_count"),
                "hard_review_required_count": hard_required,
                "blocked_count": blocked,
                "executed_without_required_review_count": unsafe,
            },
            "status_counts": report.get("status_counts") or {},
            "trigger_counts": report.get("trigger_counts") or {},
            "approved_counts": report.get("approved_counts") or {},
            "reason_counts": report.get("reason_counts") or {},
            "samples": report.get("samples") or [],
            "policy": report.get("policy") or {},
        },
        evidence=[
            {"label": "Hard reviews", "value": hard_required},
            {"label": "Blocked", "value": blocked},
            {"label": "Unsafe executed", "value": unsafe},
        ],
        next_actions=[
            "Hard-review-required entries must have approved=true before execution.",
            "Ordinary low-risk entries should not route through the high-risk reviewer.",
            "High-risk review failure must not bypass risk controls.",
        ],
    )


async def _shadow_missed_opportunity_audit() -> dict[str, Any]:
    try:
        report = _safe_shadow_missed_opportunity_report(
            await ShadowMissedOpportunityClosedLoopService().report(
                hours=SHADOW_MISSED_OPPORTUNITY_AUDIT_HOURS,
                limit=SHADOW_MISSED_OPPORTUNITY_AUDIT_LIMIT,
            )
        )
    except Exception as exc:
        return _audit_card(
            "shadow_missed_opportunity",
            "Shadow missed opportunity",
            "warning",
            "Shadow missed opportunity report failed; keep missed-opportunity feedback observe-only.",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
            next_actions=[
                "Check shadow_backtests completion rows and missed opportunity report inputs."
            ],
        )
    summary = _safe_dict(report.get("summary"))
    blocked_counts = _safe_dict(report.get("blocked_reason_counts"))
    weak_executed = int(summary.get("weak_evidence_executed_count") or 0)
    adopted = int(summary.get("adopted_count") or 0)
    probes = int(summary.get("probe_count") or 0)
    blocked = int(summary.get("blocked_count") or 0)
    warning = bool(weak_executed or blocked_counts)
    return _audit_card(
        "shadow_missed_opportunity",
        "Shadow missed opportunity",
        "warning" if warning else "ok",
        (
            "Missed opportunity loop is still observing or blocking weak evidence."
            if warning
            else "Missed opportunity loop has qualified same-symbol same-side evidence."
        ),
        details={
            "audit_only": True,
            "live_entry_mutation": False,
            "can_bypass_risk_controls": False,
            "weak_evidence_execution_allowed": False,
            "global_missed_count_can_drive_entries": False,
            "summary": summary,
            "blocked_reason_counts": blocked_counts,
            "probe_candidates": _safe_list(report.get("probe_candidates"))[:10],
            "adopted": _safe_list(report.get("adopted"))[:10],
            "blocked_examples": _safe_list(report.get("blocked"))[:10],
        },
        evidence=[
            {"label": "missed", "value": int(summary.get("missed_count") or 0)},
            {"label": "adopted", "value": adopted},
            {"label": "probe", "value": probes},
            {"label": "blocked", "value": blocked},
            {"label": "weak_executed", "value": weak_executed},
        ],
        next_actions=[
            "Use only same-symbol same-side repeated missed opportunities as learning evidence.",
            "Do not use global missed counts to force entries or bypass risk controls.",
            "Weak evidence execution must stay zero before promotion beyond observation.",
        ],
    )


async def _strong_opportunity_audit() -> dict[str, Any]:
    try:
        report = _safe_strong_opportunity_report(
            await StrongOpportunityService(
                lookback_hours=STRONG_OPPORTUNITY_AUDIT_HOURS,
                limit=STRONG_OPPORTUNITY_AUDIT_LIMIT,
            ).report()
        )
    except Exception as exc:
        return _audit_card(
            "strong_opportunity",
            "Strong opportunity",
            "warning",
            "Strong opportunity report failed; keep Phase 2 promotion shadow-only.",
            details={
                "error": safe_error_text(exc, limit=180),
                "audit_only": True,
                "live_entry_mutation": False,
                "live_sizing_mutation": False,
                "can_bypass_risk_controls": False,
                "can_force_open": False,
                "can_apply_live_sizing": False,
            },
            next_actions=[
                "Check recent entry decision raw opportunity_score and entry_candidate_evidence.",
                "Do not promote sizing or entry behavior while the report is unavailable.",
            ],
        )
    strong_count = int(report.get("strong_candidate_count") or 0)
    near_miss_count = int(report.get("near_miss_count") or 0)
    entry_decisions = int(report.get("entry_decisions") or 0)
    executed_strong = int(report.get("executed_strong_candidate_count") or 0)
    blockers = _safe_dict(report.get("blocker_counts"))
    warning = bool(strong_count == 0 or near_miss_count or blockers)
    return _audit_card(
        "strong_opportunity",
        "Strong opportunity",
        "warning" if warning else "ok",
        (
            "Strong opportunity classifier is shadow-only; no live sizing promotion is allowed."
            if warning
            else "Strong opportunity classifier found auditable candidates in shadow mode."
        ),
        details={
            "audit_only": True,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "can_bypass_risk_controls": False,
            "can_force_open": False,
            "can_apply_live_sizing": False,
            "lookback_hours": report.get("lookback_hours"),
            "checked_decisions": int(report.get("checked_decisions") or 0),
            "entry_decisions": entry_decisions,
            "strong_candidate_count": strong_count,
            "executed_strong_candidate_count": executed_strong,
            "near_miss_count": near_miss_count,
            "blocker_counts": blockers,
            "evidence_tier_counts": _safe_dict(report.get("evidence_tier_counts")),
            "side_counts": _safe_dict(report.get("side_counts")),
            "thresholds": _safe_dict(report.get("thresholds")),
            "strong_candidates": _safe_list(report.get("strong_candidates"))[:10],
            "near_misses": _safe_list(report.get("near_misses"))[:10],
            "diagnostic_boundary": report.get("diagnostic_boundary"),
        },
        evidence=[
            {"label": "entries", "value": entry_decisions},
            {"label": "strong", "value": strong_count},
            {"label": "executed_strong", "value": executed_strong},
            {"label": "near_miss", "value": near_miss_count},
        ],
        next_actions=[
            "Use this card only as Phase 2 shadow evidence; it cannot force open orders.",
            "Before live promotion, verify OKX fact integrity, selected-side expected net, profit quality, loss probability, tail risk and evidence tier.",
            "Do not increase leverage, position size, or bypass risk gates from this report alone.",
        ],
    )


async def _position_capacity_release_audit() -> dict[str, Any]:
    try:
        report = _safe_position_capacity_release_report(
            await PositionCapacityReleaseAuditService(
                lookback_hours=POSITION_CAPACITY_RELEASE_AUDIT_HOURS,
                limit=POSITION_CAPACITY_RELEASE_AUDIT_LIMIT,
            ).report()
        )
    except Exception as exc:
        return _audit_card(
            "position_capacity_release",
            "Position capacity release",
            "warning",
            "Position capacity release report failed; do not tune entries from capacity guesses.",
            details={
                "error": safe_error_text(exc, limit=180),
                "read_only": True,
                "audit_only": True,
                "live_exit_mutation": False,
                "live_entry_mutation": False,
                "live_sizing_mutation": False,
                "can_force_close": False,
                "can_close_winners": False,
                "can_bypass_risk_controls": False,
            },
            next_actions=[
                "Check position quality, release decisions, linked close orders, and crowded-side blocks.",
                "Do not force-close winners or open new capacity while release audit is unavailable.",
            ],
        )
    capacity = _safe_dict(report.get("capacity"))
    current_release_count = int(report.get("current_release_candidate_count") or 0)
    old_profit_count = int(report.get("old_profit_rotation_candidate_count") or 0)
    unclosed_release_count = int(report.get("unclosed_release_decision_count") or 0)
    protected_release_count = int(report.get("protected_release_decision_count") or 0)
    exchange_blocked_count = int(report.get("exchange_blocked_release_decision_count") or 0)
    execution_link_gap_count = int(report.get("execution_link_gap_release_decision_count") or 0)
    stale_release_count = int(report.get("stale_release_decision_count") or 0)
    crowded_block_count = int(report.get("crowded_block_count") or 0)
    open_group_count = int(report.get("open_group_count") or 0)
    entry_limit = int(capacity.get("entry_limit") or 0)
    over_capacity = bool(entry_limit and open_group_count >= entry_limit)
    warning = bool(
        current_release_count
        or old_profit_count
        or unclosed_release_count
        or exchange_blocked_count
        or execution_link_gap_count
        or crowded_block_count
        or over_capacity
    )
    return _audit_card(
        "position_capacity_release",
        "Position capacity release",
        "warning" if warning else "ok",
        (
            "Capacity release audit is observing release gaps, exchange blocks, execution-link gaps, old profit candidates, or crowded-side pressure."
            if warning
            else "Capacity release audit sees no current release backlog or crowded-side pressure."
        ),
        details={
            "read_only": True,
            "audit_only": True,
            "live_exit_mutation": False,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "can_force_close": False,
            "can_close_winners": False,
            "can_bypass_risk_controls": False,
            "lookback_hours": report.get("lookback_hours"),
            "checked_decisions": int(report.get("checked_decisions") or 0),
            "open_position_count": int(report.get("open_position_count") or 0),
            "open_group_count": open_group_count,
            "side_counts": _safe_dict(report.get("side_counts")),
            "quality_bucket_counts": _safe_dict(report.get("quality_bucket_counts")),
            "capacity": capacity,
            "current_release_candidate_count": current_release_count,
            "old_profit_rotation_candidate_count": old_profit_count,
            "release_decision_count": int(report.get("release_decision_count") or 0),
            "executed_release_decision_count": int(
                report.get("executed_release_decision_count") or 0
            ),
            "protected_release_decision_count": protected_release_count,
            "exchange_blocked_release_decision_count": exchange_blocked_count,
            "execution_link_gap_release_decision_count": execution_link_gap_count,
            "stale_release_decision_count": stale_release_count,
            "unclosed_release_decision_count": unclosed_release_count,
            "release_execution_state_counts": _safe_dict(
                report.get("release_execution_state_counts")
            ),
            "release_execution_block_counts": _safe_dict(
                report.get("release_execution_block_counts")
            ),
            "crowded_block_count": crowded_block_count,
            "current_release_candidates": _safe_list(report.get("current_release_candidates"))[:8],
            "old_profit_rotation_candidates": _safe_list(
                report.get("old_profit_rotation_candidates")
            )[:8],
            "unclosed_release_decisions": _safe_list(report.get("unclosed_release_decisions"))[:8],
            "protected_release_decisions": _safe_list(report.get("protected_release_decisions"))[
                :8
            ],
            "exchange_blocked_release_decisions": _safe_list(
                report.get("exchange_blocked_release_decisions")
            )[:8],
            "execution_link_gap_release_decisions": _safe_list(
                report.get("execution_link_gap_release_decisions")
            )[:8],
            "stale_release_decisions": _safe_list(report.get("stale_release_decisions"))[:8],
            "crowded_blocks": _safe_list(report.get("crowded_blocks"))[:8],
            "diagnostic_boundary": report.get("diagnostic_boundary"),
        },
        evidence=[
            {"label": "open_groups", "value": open_group_count},
            {"label": "entry_limit", "value": entry_limit},
            {"label": "release_candidates", "value": current_release_count},
            {"label": "old_profit_candidates", "value": old_profit_count},
            {"label": "unclosed_release", "value": unclosed_release_count},
            {"label": "crowded_blocks", "value": crowded_block_count},
        ],
        next_actions=[
            "Use this card to explain capacity pressure before changing entry thresholds or position size.",
            "If release decisions are unclosed, inspect linked execution reasons and close-order results.",
            "Old profitable positions require continuation evidence before live lock-profit or close policy promotion.",
        ],
    )


def _safe_strategy_signal_root_cause_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["read_only"] = True
    safe["audit_only"] = True
    safe["live_entry_mutation"] = False
    safe["live_sizing_mutation"] = False
    safe["live_leverage_mutation"] = False
    safe["can_force_open"] = False
    safe["can_override_thresholds"] = False
    safe["can_change_ml_readiness"] = False
    safe["can_bypass_risk_controls"] = False
    root_causes = safe.get("root_causes") if isinstance(safe.get("root_causes"), list) else []
    for row in root_causes:
        if not isinstance(row, dict):
            continue
        row["can_force_open"] = False
        row["can_override_thresholds"] = False
        row["can_change_ml_readiness"] = False
        row["can_bypass_risk_controls"] = False
    scheduler = safe.get("scheduler") if isinstance(safe.get("scheduler"), dict) else {}
    if scheduler:
        scheduler["read_only"] = True
        scheduler["audit_only"] = True
        scheduler["live_entry_mutation"] = False
        scheduler["live_sizing_mutation"] = False
        scheduler["live_leverage_mutation"] = False
        scheduler["can_force_open"] = False
        scheduler["can_override_thresholds"] = False
        scheduler["can_bypass_risk_controls"] = False
        scheduler_samples = (
            scheduler.get("latest_samples")
            if isinstance(scheduler.get("latest_samples"), list)
            else []
        )
        for row in scheduler_samples:
            if not isinstance(row, dict):
                continue
            row["can_force_open"] = False
            row["can_override_thresholds"] = False
            row["can_bypass_risk_controls"] = False
    return safe


async def _strategy_signal_root_cause_audit() -> dict[str, Any]:
    try:
        report = _safe_strategy_signal_root_cause_report(
            await StrategySignalRootCauseAuditService(
                lookback_hours=STRATEGY_SIGNAL_ROOT_CAUSE_AUDIT_HOURS,
                limit=STRATEGY_SIGNAL_ROOT_CAUSE_AUDIT_LIMIT,
            ).report()
        )
    except Exception as exc:
        return _audit_card(
            "strategy_signal_root_cause",
            "策略信号根因",
            "warning",
            "策略信号根因审计读取失败；先修审计链路，不改交易阈值。",
            details={
                "error": safe_error_text(exc, limit=180),
                "read_only": True,
                "audit_only": True,
                "live_entry_mutation": False,
                "live_sizing_mutation": False,
                "live_leverage_mutation": False,
                "can_force_open": False,
                "can_override_thresholds": False,
                "can_change_ml_readiness": False,
                "can_bypass_risk_controls": False,
            },
            next_actions=[
                "先恢复只读审计报告，再判断 ML、server_profit、shadow missed opportunity 卡点。",
                "审计失败期间不得通过降阈值、放大仓位或硬改 ML readiness 制造成交。",
            ],
            owner_path="services/strategy_signal_root_cause_audit.py",
        )

    ml = _safe_dict(report.get("ml"))
    server_profit = _safe_dict(report.get("server_profit"))
    shadow = _safe_dict(report.get("shadow_missed_opportunity"))
    root_causes = report.get("root_causes") if isinstance(report.get("root_causes"), list) else []
    status = str(report.get("status") or "warning")
    return _audit_card(
        "strategy_signal_root_cause",
        "策略信号根因",
        status if status in {"ok", "warning", "critical"} else "warning",
        str(report.get("summary") or "策略信号根因审计已完成。"),
        details=report,
        evidence=[
            {"label": "开仓候选", "value": int(report.get("entry_decision_count") or 0)},
            {"label": "高质量候选", "value": int(report.get("high_quality_entry_count") or 0)},
            {"label": "ML可用率", "value": ml.get("usable_rate", 0.0)},
            {
                "label": "server_profit反向/负向",
                "value": int(server_profit.get("negative_or_opposite_count") or 0),
            },
            {"label": "影子错过机会", "value": int(shadow.get("missed_count") or 0)},
            {"label": "根因数", "value": len(root_causes)},
        ],
        next_actions=(
            report.get("next_actions") if isinstance(report.get("next_actions"), list) else []
        ),
        owner_path="services/strategy_signal_root_cause_audit.py",
    )


def _trade_contract_violation_counts(summary: dict[str, Any]) -> tuple[int, int]:
    hard = (
        int(summary.get("weak_evidence_executed_count") or 0)
        + int(summary.get("negative_expected_executed_count") or 0)
        + int(summary.get("fast_loss_without_strong_exit_count") or 0)
        + int(summary.get("reentry_without_strong_unlock_count") or 0)
        + _profit_first_unresolved_count(summary, "profit_first_plan_missing_count")
        + _profit_first_unresolved_count(summary, "profit_first_plan_incomplete_count")
        + _profit_first_unresolved_count(summary, "shadow_lane_executed_count")
        + _profit_first_unresolved_count(
            summary, "profit_first_position_ladder_missing_count"
        )
        + _profit_first_unresolved_count(summary, "exit_plan_reference_missing_count")
        + _profit_first_unresolved_count(
            summary, "exit_plan_failure_reason_missing_count"
        )
        + _profit_first_unresolved_count(summary, "low_payoff_meaningful_size_count")
        + _profit_first_unresolved_count(
            summary, "profit_first_lane_size_above_max_count"
        )
        + _profit_first_unresolved_count(summary, "probe_loss_brake_bypassed_count")
        + _profit_first_unresolved_count(
            summary, "meaningful_lane_tiny_without_budget_reason_count"
        )
    )
    soft = (
        int(summary.get("missing_entry_explanation_count") or 0)
        + int(summary.get("missing_sizing_explanation_count") or 0)
        + int(summary.get("small_size_without_reason_count") or 0)
        + int(summary.get("profit_first_plan_derived_count") or 0)
    )
    return hard, soft


def _profit_first_unresolved_count(summary: dict[str, Any], key: str) -> int:
    if f"{key}_unresolved" in summary:
        return int(summary.get(f"{key}_unresolved") or 0)
    return int(summary.get(key) or 0)


def _trade_contract_current_window(
    *,
    runtime_window: dict[str, Any],
    current_summary: dict[str, Any],
    current_hard_violations: int,
    current_soft_violations: int,
    historical_has_violations: bool,
) -> dict[str, Any]:
    started_at = _parse_utc_datetime(runtime_window.get("started_at"))
    available = bool(runtime_window.get("available") and started_at is not None)
    return {
        "available": available,
        "started_at": runtime_window.get("started_at_iso") or _iso(started_at),
        "heartbeat_at": runtime_window.get("heartbeat_at_iso"),
        "running": bool(runtime_window.get("running")),
        "mode": runtime_window.get("mode"),
        "decision_interval": runtime_window.get("decision_interval"),
        "decision_count": int(current_summary.get("decision_count") or 0),
        "executed_entry_count": int(current_summary.get("executed_entry_count") or 0),
        "weak_evidence_executed_count": int(
            current_summary.get("weak_evidence_executed_count") or 0
        ),
        "negative_expected_executed_count": int(
            current_summary.get("negative_expected_executed_count") or 0
        ),
        "fast_loss_without_strong_exit_count": int(
            current_summary.get("fast_loss_without_strong_exit_count") or 0
        ),
        "reentry_without_strong_unlock_count": int(
            current_summary.get("reentry_without_strong_unlock_count") or 0
        ),
        "profit_first_plan_missing_count": int(
            current_summary.get("profit_first_plan_missing_count") or 0
        ),
        "profit_first_plan_missing_count_unresolved": _profit_first_unresolved_count(
            current_summary, "profit_first_plan_missing_count"
        ),
        "historical_recovery_quarantined_profit_first_plan_missing_count": int(
            current_summary.get(
                "historical_recovery_quarantined_profit_first_plan_missing_count"
            )
            or 0
        ),
        "profit_first_plan_incomplete_count": int(
            current_summary.get("profit_first_plan_incomplete_count") or 0
        ),
        "profit_first_plan_incomplete_count_unresolved": _profit_first_unresolved_count(
            current_summary, "profit_first_plan_incomplete_count"
        ),
        "historical_recovery_quarantined_profit_first_plan_incomplete_count": int(
            current_summary.get(
                "historical_recovery_quarantined_profit_first_plan_incomplete_count"
            )
            or 0
        ),
        "shadow_lane_executed_count": int(current_summary.get("shadow_lane_executed_count") or 0),
        "shadow_lane_executed_count_unresolved": _profit_first_unresolved_count(
            current_summary, "shadow_lane_executed_count"
        ),
        "historical_recovery_quarantined_shadow_lane_executed_count": int(
            current_summary.get("historical_recovery_quarantined_shadow_lane_executed_count")
            or 0
        ),
        "profit_first_position_ladder_missing_count": int(
            current_summary.get("profit_first_position_ladder_missing_count") or 0
        ),
        "profit_first_position_ladder_missing_count_unresolved": (
            _profit_first_unresolved_count(
                current_summary, "profit_first_position_ladder_missing_count"
            )
        ),
        "historical_recovery_quarantined_profit_first_position_ladder_missing_count": int(
            current_summary.get(
                "historical_recovery_quarantined_profit_first_position_ladder_missing_count"
            )
            or 0
        ),
        "exit_plan_reference_missing_count": int(
            current_summary.get("exit_plan_reference_missing_count") or 0
        ),
        "exit_plan_reference_missing_count_unresolved": _profit_first_unresolved_count(
            current_summary, "exit_plan_reference_missing_count"
        ),
        "historical_recovery_quarantined_exit_plan_reference_missing_count": int(
            current_summary.get(
                "historical_recovery_quarantined_exit_plan_reference_missing_count"
            )
            or 0
        ),
        "exit_plan_failure_reason_missing_count": int(
            current_summary.get("exit_plan_failure_reason_missing_count") or 0
        ),
        "exit_plan_failure_reason_missing_count_unresolved": _profit_first_unresolved_count(
            current_summary, "exit_plan_failure_reason_missing_count"
        ),
        "historical_recovery_quarantined_exit_plan_failure_reason_missing_count": int(
            current_summary.get(
                "historical_recovery_quarantined_exit_plan_failure_reason_missing_count"
            )
            or 0
        ),
        "low_payoff_meaningful_size_count": int(
            current_summary.get("low_payoff_meaningful_size_count") or 0
        ),
        "low_payoff_meaningful_size_count_unresolved": _profit_first_unresolved_count(
            current_summary, "low_payoff_meaningful_size_count"
        ),
        "historical_recovery_quarantined_low_payoff_meaningful_size_count": int(
            current_summary.get(
                "historical_recovery_quarantined_low_payoff_meaningful_size_count"
            )
            or 0
        ),
        "profit_first_lane_size_above_max_count": int(
            current_summary.get("profit_first_lane_size_above_max_count") or 0
        ),
        "profit_first_lane_size_above_max_count_unresolved": _profit_first_unresolved_count(
            current_summary, "profit_first_lane_size_above_max_count"
        ),
        "historical_recovery_quarantined_profit_first_lane_size_above_max_count": int(
            current_summary.get(
                "historical_recovery_quarantined_profit_first_lane_size_above_max_count"
            )
            or 0
        ),
        "probe_loss_brake_bypassed_count": int(
            current_summary.get("probe_loss_brake_bypassed_count") or 0
        ),
        "probe_loss_brake_bypassed_count_unresolved": _profit_first_unresolved_count(
            current_summary, "probe_loss_brake_bypassed_count"
        ),
        "historical_recovery_quarantined_probe_loss_brake_bypassed_count": int(
            current_summary.get(
                "historical_recovery_quarantined_probe_loss_brake_bypassed_count"
            )
            or 0
        ),
        "meaningful_lane_tiny_without_budget_reason_count": int(
            current_summary.get("meaningful_lane_tiny_without_budget_reason_count") or 0
        ),
        "meaningful_lane_tiny_without_budget_reason_count_unresolved": (
            _profit_first_unresolved_count(
                current_summary, "meaningful_lane_tiny_without_budget_reason_count"
            )
        ),
        "historical_recovery_quarantined_meaningful_lane_tiny_without_budget_reason_count": int(
            current_summary.get(
                "historical_recovery_quarantined_meaningful_lane_tiny_without_budget_reason_count"
            )
            or 0
        ),
        "profit_first_plan_derived_count": int(
            current_summary.get("profit_first_plan_derived_count") or 0
        ),
        "soft_violation_count": int(current_soft_violations),
        "hard_violation_count": int(current_hard_violations),
        "contract_violation_count": int(current_summary.get("contract_violation_count") or 0),
        "historical_recovery_quarantined_violation_count": int(
            current_summary.get("historical_recovery_quarantined_violation_count") or 0
        ),
        "historical_recovery_quarantine_unresolved_count": int(
            current_summary.get("historical_recovery_quarantine_unresolved_count")
            if "historical_recovery_quarantine_unresolved_count" in current_summary
            else current_hard_violations
        ),
        "historical_legacy_issues": bool(
            available
            and historical_has_violations
            and not current_hard_violations
            and not current_soft_violations
            and not int(current_summary.get("contract_violation_count") or 0)
        ),
    }


async def _trade_execution_contract_audit() -> dict[str, Any]:
    try:
        service = TradeExecutionContractService()
        report = _safe_trade_execution_contract_report(
            await service.report(
                hours=TRADE_EXECUTION_CONTRACT_AUDIT_HOURS,
                limit=TRADE_EXECUTION_CONTRACT_AUDIT_LIMIT,
            )
        )
        runtime_window = _load_trading_runtime_audit_window()
        runtime_started_at = _parse_utc_datetime(runtime_window.get("started_at"))
        current_report: dict[str, Any] | None = None
        if runtime_started_at is not None:
            current_report = _safe_trade_execution_contract_report(
                await service.report(
                    hours=TRADE_EXECUTION_CONTRACT_AUDIT_HOURS,
                    limit=TRADE_EXECUTION_CONTRACT_AUDIT_LIMIT,
                    since=runtime_started_at,
                )
            )
    except Exception as exc:
        return _audit_card(
            "trade_execution_contract",
            "Trade execution contract",
            "warning",
            "Trade execution contract report failed; keep entry, sizing and exit gates unchanged.",
            details={
                "error": safe_error_text(exc, limit=180),
                "report_available": False,
                "audit_only": True,
                "live_entry_mutation": False,
                "live_exit_mutation": False,
                "can_bypass_risk_controls": False,
                "summary": {
                    "decision_count": 0,
                    "executed_entry_count": 0,
                    "contract_violation_count": 0,
                    "report_available": False,
                },
                "current_summary": {},
                "violation_reason_counts": {},
                "policy": {
                    "entry_requires_profit_first_trade_plan": True,
                    "profit_first_missing_plan_is_hard_violation": True,
                    "profit_first_shadow_lane_cannot_execute": True,
                    "profit_first_probe_loss_brake_must_block_execution": True,
                    "report_available": False,
                },
            },
            next_actions=[
                "Check recent ai_decisions, orders and positions before changing live trading gates."
            ],
        )
    summary = _safe_dict(report.get("summary"))
    violation_counts = _safe_dict(report.get("violation_reason_counts"))
    current_summary = _safe_dict(current_report.get("summary")) if current_report else {}
    current_violation_counts = (
        _safe_dict(current_report.get("violation_reason_counts")) if current_report else {}
    )
    weak_executed = int(summary.get("weak_evidence_executed_count") or 0)
    negative_expected = int(summary.get("negative_expected_executed_count") or 0)
    fast_loss_without_exit = int(summary.get("fast_loss_without_strong_exit_count") or 0)
    dust_fast_loss = int(summary.get("dust_or_rounding_fast_loss_count") or 0)
    reentry_without_unlock = int(summary.get("reentry_without_strong_unlock_count") or 0)
    missing_sizing = int(summary.get("missing_sizing_explanation_count") or 0)
    hard_violations, soft_violations = _trade_contract_violation_counts(summary)
    current_hard_violations, current_soft_violations = _trade_contract_violation_counts(
        current_summary
    )
    current_window = _trade_contract_current_window(
        runtime_window=runtime_window,
        current_summary=current_summary,
        current_hard_violations=current_hard_violations,
        current_soft_violations=current_soft_violations,
        historical_has_violations=bool(hard_violations or soft_violations or violation_counts),
    )
    status = _status_from_counts(
        critical=bool(current_hard_violations if current_report else hard_violations),
        warning=bool(
            current_soft_violations
            or current_violation_counts
            or current_window.get("historical_legacy_issues")
            or (not current_report and (soft_violations or violation_counts))
        ),
    )
    summary_text = (
        "Trade execution has current hard contract violations in entry evidence, expected net, fast exits or loss re-entry."
        if (current_hard_violations if current_report else hard_violations)
        else (
            "24h historical trade execution contract violations remain; current runtime window has not reproduced them."
            if current_window.get("historical_legacy_issues")
            else (
                "Trade execution contract is missing explanations or sizing reasons; keep observing before tuning live logic."
                if (current_soft_violations if current_report else soft_violations)
                else "Trade execution contract passed for recent executed entries and closed positions."
            )
        )
    )
    return _audit_card(
        "trade_execution_contract",
        "Trade execution contract",
        status,
        summary_text,
        details={
            "audit_only": True,
            "live_entry_mutation": False,
            "live_exit_mutation": False,
            "can_bypass_risk_controls": False,
            "summary": summary,
            "violation_reason_counts": violation_counts,
            "current_summary": current_summary,
            "current_violation_reason_counts": current_violation_counts,
            "current_runtime_window": current_window,
            "entry_explanations": _safe_list(report.get("entry_explanations"))[:10],
            "fast_loss_samples": _safe_list(report.get("fast_loss_samples"))[:10],
            "dust_or_rounding_fast_loss_samples": _safe_list(
                report.get("dust_or_rounding_fast_loss_samples")
            )[:10],
            "violations": _safe_list(report.get("violations"))[:10],
            "current_entry_explanations": (
                _safe_list(current_report.get("entry_explanations"))[:10] if current_report else []
            ),
            "current_fast_loss_samples": (
                _safe_list(current_report.get("fast_loss_samples"))[:10] if current_report else []
            ),
            "current_dust_or_rounding_fast_loss_samples": (
                _safe_list(current_report.get("dust_or_rounding_fast_loss_samples"))[:10]
                if current_report
                else []
            ),
            "current_violations": (
                _safe_list(current_report.get("violations"))[:10] if current_report else []
            ),
            "current_historical_recovery_quarantined_violations": (
                _safe_list(current_report.get("historical_recovery_quarantined_violations"))[:10]
                if current_report
                else []
            ),
            "historical_recovery_quarantined_violations": _safe_list(
                report.get("historical_recovery_quarantined_violations")
            )[:10],
            "policy": _safe_dict(report.get("policy")),
            "query_policy": _safe_dict(report.get("query_policy")),
            "current_query_policy": (
                _safe_dict(current_report.get("query_policy")) if current_report else {}
            ),
        },
        evidence=[
            {"label": "执行开仓", "value": int(summary.get("executed_entry_count") or 0)},
            {"label": "弱证据执行", "value": weak_executed},
            {"label": "负期望执行", "value": negative_expected},
            {"label": "快亏缺强证据", "value": fast_loss_without_exit},
            {"label": "复开缺解锁", "value": reentry_without_unlock},
            {"label": "缺仓位解释", "value": missing_sizing},
        ],
        next_actions=[
            "Do not loosen entry thresholds to hide weak-evidence or non-positive expected-net executions.",
            "Fast loss exits must keep strong exit evidence before promotion beyond observation.",
            "Same-symbol same-side re-entry after loss must require explicit high-quality unlock evidence.",
        ],
    )


async def _profit_first_ranking_audit() -> dict[str, Any]:
    try:
        report = _safe_profit_first_ranking_report(
            await ProfitFirstRankingService().report(
                hours=PROFIT_FIRST_RANKING_AUDIT_HOURS,
                limit=PROFIT_FIRST_RANKING_AUDIT_LIMIT,
            )
        )
    except Exception as exc:
        return _audit_card(
            "profit_first_ranking",
            "Profit-First ranking",
            "warning",
            "Profit-First ranking report failed; keep model and strategy promotion shadow-only.",
            details={
                "error": safe_error_text(exc, limit=180),
                "report_available": False,
                "audit_only": True,
                "read_only": True,
                "live_mutation": False,
                "live_weight_mutation": False,
                "live_sizing_mutation": False,
                "can_change_model_routing": False,
                "can_change_strategy_weight": False,
                "can_increase_live_size": False,
                "ranking_ready": False,
                "status": "unavailable",
                "summary": {
                    "decision_count": 0,
                    "closed_position_count": 0,
                    "leaderboard_row_count": 0,
                    "source_row_count": 0,
                    "promote_candidate_count": 0,
                    "demote_count": 0,
                    "disable_count": 0,
                    "blocker_count": 0,
                    "report_available": False,
                },
                "blockers": [],
                "policy": {
                    "promotion_flow": "shadow_to_canary_to_live",
                    "trade_fact_policy": "okx_confirmed_closed_positions_only",
                    "report_available": False,
                },
            },
            next_actions=[
                "Rebuild the ranking report from ProfitFirstTradePlan and OKX-confirmed closed positions before resuming entries.",
            ],
        )
    summary = _safe_dict(report.get("summary"))
    blockers = _safe_list(report.get("blockers"))
    hard_blockers = [row for row in blockers if _safe_dict(row).get("severity") == "blocking"]
    demote_count = int(summary.get("demote_count") or 0)
    disable_count = int(summary.get("disable_count") or 0)
    ranking_ready = bool(report.get("ranking_ready"))
    if hard_blockers or disable_count:
        status = "critical"
        summary_text = "Profit-First ranking found model/strategy combinations that must be disabled before resume."
    elif demote_count or blockers:
        status = "warning"
        summary_text = "Profit-First ranking found losing model/source evidence; keep affected routes shadow or reduced."
    elif not ranking_ready:
        status = "warning"
        summary_text = "Profit-First ranking is still collecting realized-PnL evidence."
    else:
        status = "ok"
        summary_text = "Profit-First ranking is ready and promotion/demotion evidence is auditable."
    if status == "warning" and _profit_first_ranking_observation_only(report):
        report["observing"] = True
        report["observation_reason"] = (
            "ranking is read-only; demoted or weak rows cannot increase live budget"
        )
    return _audit_card(
        "profit_first_ranking",
        "Profit-First ranking",
        status,
        summary_text,
        details=report,
        evidence=[
            {"label": "Closed positions", "value": int(summary.get("closed_position_count") or 0)},
            {"label": "Leaderboard rows", "value": int(summary.get("leaderboard_row_count") or 0)},
            {
                "label": "Promote candidates",
                "value": int(summary.get("promote_candidate_count") or 0),
            },
            {"label": "Demote", "value": demote_count},
            {"label": "Disable", "value": disable_count},
            {"label": "Blockers", "value": len(blockers)},
        ],
        next_actions=[
            "Do not increase live budget for demoted or disabled model/strategy/lane combinations.",
            "Only canary/live-candidate rows with clean sample floors may receive more budget after operator approval.",
            "Keep source-weight changes audit-only until Phase 3 go/no-go and paper observation pass.",
        ],
        owner_path="services/profit_first_ranking.py",
    )


async def _profit_first_governance_audit() -> dict[str, Any]:
    try:
        report = _safe_profit_first_governance_report(
            await ProfitFirstGovernanceReportService().report(
                hours=PROFIT_FIRST_GOVERNANCE_AUDIT_HOURS,
                limit=PROFIT_FIRST_GOVERNANCE_AUDIT_LIMIT,
            )
        )
    except Exception as exc:
        return _audit_card(
            "profit_first_governance",
            "Profit-First governance",
            "warning",
            "Profit-First governance report failed; keep no-entry and losing-exit tuning shadow-only.",
            details={
                "error": safe_error_text(exc, limit=180),
                "report_available": False,
                "status": "unavailable",
                "audit_only": True,
                "read_only": True,
                "live_mutation": False,
                "live_entry_mutation": False,
                "live_exit_mutation": False,
                "live_weight_mutation": False,
                "live_sizing_mutation": False,
                "can_submit_orders": False,
                "can_start_trading_service": False,
                "can_change_model_routing": False,
                "can_change_strategy_weight": False,
                "can_increase_live_size": False,
                "summary": {
                    "no_entry_sample_count": 0,
                    "losing_exit_sample_count": 0,
                    "missing_brain_output_count": 0,
                    "report_available": False,
                },
            },
            next_actions=[
                "Rebuild Profit-First governance from ranking/brain outputs before resuming entries.",
            ],
            owner_path="services/profit_first_governance_report.py",
        )
    summary = _safe_dict(report.get("summary"))
    missing_outputs = _safe_list(report.get("missing_brain_outputs"))
    report_status = str(report.get("status") or "")
    if report_status == "unavailable":
        status = "warning"
        summary_text = "Profit-First governance is unavailable; keep entries paused or shadow-only."
    elif report_status == "incomplete" or missing_outputs:
        status = "warning"
        summary_text = "Profit-First governance is incomplete; brain outputs must be fixed before resume."
    else:
        status = "ok"
        summary_text = "Profit-First no-entry and losing-exit governance is ready and read-only."
    return _audit_card(
        "profit_first_governance",
        "Profit-First governance",
        status,
        summary_text,
        details=report,
        evidence=[
            {
                "label": "No-entry samples",
                "value": int(summary.get("no_entry_sample_count") or 0),
            },
            {
                "label": "Losing exits",
                "value": int(summary.get("losing_exit_sample_count") or 0),
            },
            {"label": "Diagnosis", "value": summary.get("no_entry_diagnosis") or ""},
            {"label": "Missing brain outputs", "value": len(missing_outputs)},
        ],
        next_actions=_safe_list(report.get("next_cycle_actions"))[:6]
        or [
            "Keep collecting no-entry and losing-exit evidence until the next governance window.",
        ],
        owner_path="services/profit_first_governance_report.py",
    )


def _profit_first_recovery_blockers_audit_from_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    cards_by_key = {str(card.get("key") or ""): card for card in cards}
    trade_contract = _safe_dict(
        _safe_dict(cards_by_key.get("trade_execution_contract")).get("details")
    )
    ranking = _safe_dict(_safe_dict(cards_by_key.get("profit_first_ranking")).get("details"))
    observation = _safe_dict(
        _safe_dict(cards_by_key.get("phase3_paper_resume_observation")).get("details")
    )
    report = build_profit_first_recovery_blockers(
        trade_contract=trade_contract,
        ranking=ranking,
        observation=observation,
    )
    summary = _safe_dict(report.get("summary"))
    blocking_count = int(report.get("blocking_item_count") or 0)
    status = "critical" if blocking_count else "ok"
    summary_text = (
        "Profit-First recovery has concrete blockers that must be repaired, disabled, or quarantined before resume."
        if blocking_count
        else "Profit-First recovery blocker checklist is clear."
    )
    return _audit_card(
        "profit_first_recovery_blockers",
        "Profit-First recovery blockers",
        status,
        summary_text,
        details=report,
        evidence=[
            {
                "label": "Contract blockers",
                "value": int(summary.get("contract_blocker_count") or 0),
            },
            {"label": "Ranking blockers", "value": int(summary.get("ranking_blocker_count") or 0)},
            {"label": "OKX blockers", "value": int(summary.get("okx_blocker_count") or 0)},
            {"label": "Blocking items", "value": blocking_count},
        ],
        next_actions=[
            "Repair or quarantine missing Profit-First plan/exit-reference history before resume.",
            "Keep disabled ranking combinations shadow-only unless an operator-approved state change is applied.",
            "Resolve OKX/local reconciliation differences before using paper observations for promotion.",
        ],
        owner_path="services/profit_first_recovery_blockers.py",
    )


async def _crypto_feature_coverage_audit() -> dict[str, Any]:
    try:
        report = _safe_crypto_feature_report(
            await CryptoFeatureCoverageService().report(hours=24, limit=1000)
        )
    except Exception as exc:
        return _audit_card(
            "crypto_feature_coverage",
            "数字货币特征覆盖",
            "warning",
            "数字货币特征覆盖报告读取失败。反查采集状态前不得把缺失特征当作有效信号。",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
            next_actions=["先检查行情、盘口、衍生品、新闻/社媒和事件日历采集链路。"],
        )
    missing = (
        report.get("missing_features") if isinstance(report.get("missing_features"), list) else []
    )
    stale = report.get("stale_features") if isinstance(report.get("stale_features"), list) else []
    neutralized = (
        report.get("neutralized_features")
        if isinstance(report.get("neutralized_features"), list)
        else []
    )
    features = report.get("features") if isinstance(report.get("features"), list) else []
    waiting_for_decision_samples = bool(report.get("waiting_for_decision_samples"))
    status = str(report.get("status") or "ok")
    if status not in {"ok", "warning", "critical"}:
        status = _status_from_counts(critical=bool(not features), warning=bool(missing or stale))
    summary = (
        "核心行情或特征快照缺失，缺失特征已被中性阻断。"
        if status == "critical"
        else (
            "核心行情已预热；paper 决策样本尚未生成，盘口/衍生品/新闻等特征先保持中性观察。"
            if waiting_for_decision_samples
            else (
                f"发现 {len(missing)} 类缺失、{len(stale)} 类过期特征；已按中性/只读处理。"
                if missing or stale
                else "数字货币特征覆盖未发现缺失或过期项。"
            )
        )
    )
    top_features = [
        {
            "key": _safe_dict(item).get("key"),
            "status": _safe_dict(item).get("status"),
            "source": _safe_dict(item).get("source"),
            "confidence": _safe_dict(item).get("confidence"),
            "live_entry_influence": _safe_dict(item).get("live_entry_influence"),
            "reasons": _safe_dict(item).get("reasons") or [],
        }
        for item in features
        if _safe_dict(item).get("status") in {"missing", "stale", "low_confidence"}
    ][:10]
    return _audit_card(
        "crypto_feature_coverage",
        "数字货币特征覆盖",
        status,
        summary,
        details={
            "audit_only": True,
            "live_signal_mutation": False,
            "can_missing_features_drive_live_entry": False,
            "feature_defaults_are_neutral": True,
            "decision_sample_count": int(report.get("decision_sample_count") or 0),
            "feature_snapshot_count": int(report.get("feature_snapshot_count") or 0),
            "waiting_for_decision_samples": waiting_for_decision_samples,
            "missing_features": missing,
            "stale_features": stale,
            "neutralized_features": neutralized,
            "symbols_observed": report.get("symbols_observed") or [],
            "feature_contribution_policy": report.get("feature_contribution_policy") or {},
            "top_feature_issues": top_features,
        },
        evidence=[
            {"label": "特征项", "value": len(features)},
            {"label": "缺失", "value": len(missing)},
            {"label": "过期", "value": len(stale)},
            {"label": "决策快照", "value": int(report.get("feature_snapshot_count") or 0)},
            {"label": "已中性阻断", "value": len(neutralized)},
        ],
        next_actions=[
            "缺失数据源不得静默当作正常，也不得填成利于开仓的默认值。",
            "低可信事件只允许影子观察，不得直接驱动真实开仓。",
            "特征时间戳缺失或过期时，先修复采集链路再评估策略参数。",
        ],
    )


async def _model_training_audit() -> dict[str, Any]:
    runtime_task = asyncio.create_task(
        asyncio.wait_for(
            collect_platform_runtime_status(),
            timeout=MODEL_RUNTIME_PROBE_TIMEOUT_SECONDS,
        )
    )
    try:
        data_status = await data_collection_api.get_data_collection_status(
            include_feature_coverage=False
        )
    except Exception as exc:
        data_status = exc
    try:
        historical_trade_facts = await HistoricalTradeFactAuditService(
            lookback_days=HISTORICAL_TRADE_FACT_AUDIT_DAYS,
            limit=HISTORICAL_TRADE_FACT_AUDIT_LIMIT,
        ).report()
    except Exception as exc:
        historical_trade_facts = exc
    try:
        artifact_retirement = await ArtifactRetirementAuditService().report()
    except Exception as exc:
        artifact_retirement = exc
    try:
        runtime_status = await runtime_task
    except Exception as exc:
        runtime_status = exc
    if isinstance(data_status, Exception):
        return _audit_card(
            "model_training",
            "模型与训练",
            "warning",
            "数据采集/训练状态读取失败。",
            details={"error": safe_error_text(data_status, limit=180)},
        )
    training = data_status.get("training") if isinstance(data_status, dict) else {}
    local_tools = training.get("local_ai_tools") if isinstance(training, dict) else {}
    governance = training.get("governance") if isinstance(training, dict) else {}
    if isinstance(local_tools, dict) and isinstance(governance, dict):
        local_tools = _phase3_merge_clean_training_view_into_local_tools(
            local_tools,
            governance,
        )
    historical_trade_fact_audit_warning = isinstance(historical_trade_facts, Exception)
    if historical_trade_fact_audit_warning:
        historical_trade_fact_report = {
            "status": "unavailable",
            "read_only": True,
            "audit_only": True,
            "error": safe_error_text(historical_trade_facts, limit=180),
            "cleanup_mode": "quarantine_not_delete",
            "training_policy": "clean_training_view_only",
            "can_delete_history": False,
            "can_apply_repair": False,
        }
    else:
        historical_trade_fact_report = (
            historical_trade_facts if isinstance(historical_trade_facts, dict) else {}
        )
    artifact_retirement_audit_warning = isinstance(artifact_retirement, Exception)
    if artifact_retirement_audit_warning:
        artifact_retirement_report = {
            "status": "unavailable",
            "read_only": True,
            "audit_only": True,
            "raw_artifacts_preserved": True,
            "can_delete_artifacts": False,
            "error": safe_error_text(artifact_retirement, limit=180),
            "next_required_action": "rerun_artifact_retirement_audit",
        }
    else:
        artifact_retirement_report = (
            artifact_retirement if isinstance(artifact_retirement, dict) else {}
        )
    artifact_retirement_required = str(artifact_retirement_report.get("status") or "") in {
        "retired_required",
        "untrusted",
        "blocked",
    }
    sources = data_status.get("sources") if isinstance(data_status, dict) else []
    optional_source_warnings, hard_source_warnings = _split_training_source_warnings(sources)
    runtime_probe: dict[str, Any] = {"status": "unknown"}
    model_critical: list[dict[str, Any]] = []
    if isinstance(runtime_status, Exception):
        timeout = isinstance(runtime_status, TimeoutError)
        runtime_probe = {
            "status": "warning",
            "error": safe_error_text(runtime_status, limit=180),
            "timeout": timeout,
        }
    elif isinstance(runtime_status, dict):
        runtime_models = (
            runtime_status.get("ai_models")
            if isinstance(runtime_status.get("ai_models"), list)
            else []
        )
        runtime_local_tools = (
            runtime_status.get("local_ai_tools")
            if isinstance(runtime_status.get("local_ai_tools"), dict)
            else {}
        )
        for row in runtime_models:
            if not isinstance(row, dict) or bool(row.get("available")):
                continue
            model_critical.append(
                {
                    "model": row.get("model") or row.get("name"),
                    "api_base": row.get("api_base"),
                    "endpoint_ok": bool(row.get("endpoint_ok")),
                    "model_available": bool(row.get("model_available")),
                    "status_code": row.get("status_code"),
                    "latency_ms": row.get("latency_ms"),
                    "error": row.get("error"),
                }
            )
        if runtime_local_tools and not bool(runtime_local_tools.get("available")):
            if bool(runtime_local_tools.get("configured", True)):
                model_critical.append(
                    {
                        "model": "local_ai_tools",
                        "api_base": runtime_local_tools.get("api_base"),
                        "health": runtime_local_tools.get("health"),
                        "status": runtime_local_tools.get("status"),
                        "child_endpoints": runtime_local_tools.get("child_endpoints"),
                    }
                )
        runtime_probe = {
            "status": "critical" if model_critical else "ok",
            "ai_model_count": len(runtime_models),
            "local_ai_tools_configured": (
                bool(runtime_local_tools.get("configured")) if runtime_local_tools else False
            ),
            "local_ai_tools_available": (
                bool(runtime_local_tools.get("available")) if runtime_local_tools else False
            ),
        }
    runtime_probe_timeout = bool(runtime_probe.get("timeout"))
    local_tools_status = str(local_tools.get("status") or "").lower()
    clean_training_view_available = (
        _safe_int_value(
            local_tools.get("phase3_clean_trainable_sample_count")
            or local_tools.get("trainable_sample_count")
            or local_tools.get("shadow_sample_count")
        )
        > 0
        and _safe_int_value(
            local_tools.get("trainable_trade_sample_count") or local_tools.get("trade_sample_count")
        )
        > 0
        and str(local_tools.get("phase3_training_policy") or "") == "clean_training_view_only"
    )
    local_tools_status_probe_slow = local_tools_status in {"timeout", "status_error"} and (
        bool(runtime_probe.get("local_ai_tools_available")) or clean_training_view_available
    )
    local_tools_unconfigured = (
        not bool(local_tools.get("available"))
        and local_tools_status in OPTIONAL_TRAINING_SOURCE_STATUSES
        and not bool(runtime_probe.get("local_ai_tools_configured"))
    )
    local_tools_hard_missing = (
        not bool(local_tools.get("available"))
        and not local_tools_unconfigured
        and not local_tools_status_probe_slow
    )
    runtime_probe_timeout_is_observing = runtime_probe_timeout and (
        bool(local_tools.get("available"))
        or local_tools_status_probe_slow
        or clean_training_view_available
    )
    runtime_probe_hard_failure = runtime_probe.get("status") == "warning" and not (
        runtime_probe_timeout_is_observing
    )
    hard_failure = (
        bool(model_critical)
        or bool(hard_source_warnings)
        or local_tools_hard_missing
        or runtime_probe_hard_failure
    )
    observing = not hard_failure and (
        bool(optional_source_warnings)
        or local_tools_status == "learning_only"
        or local_tools_status_probe_slow
        or local_tools_unconfigured
        or runtime_probe_timeout
        or historical_trade_fact_audit_warning
        or artifact_retirement_audit_warning
        or artifact_retirement_required
    )
    evaluation_policy = (
        local_tools.get("evaluation_policy")
        if isinstance(local_tools.get("evaluation_policy"), dict)
        else {}
    )
    promotion_flow = (
        local_tools.get("promotion_flow")
        or evaluation_policy.get("promotion_flow")
        or "shadow_to_canary_to_live"
    )
    phase3_training_governance = {
        "training_mode": local_tools.get("training_mode") or "shadow",
        "model_stage": local_tools.get("model_stage") or "shadow",
        "evaluation_policy": evaluation_policy,
        "promotion_flow": promotion_flow,
        "live_mutation": bool(
            local_tools.get("live_mutation") or evaluation_policy.get("live_mutation")
        ),
        "requires_walk_forward": bool(evaluation_policy.get("requires_walk_forward", True)),
        "policy": (
            "Phase 3 model changes must progress through shadow -> canary -> live; "
            "audit visibility must not mutate live trading weights by itself."
        ),
    }
    phase3_rebuild_readiness = Phase3RebuildReadinessService().report(
        local_ai_tools=local_tools,
        governance=governance if isinstance(governance, dict) else {},
        historical_trade_fact_audit=historical_trade_fact_report,
        artifact_retirement_audit=artifact_retirement_report,
        runtime_probe=runtime_probe,
        requested_persist_artifact=False,
        confirm_phase3_rebuild=False,
    )
    specialist_shadow_evaluation = _specialist_shadow_latest_report()
    specialist_summary = _safe_dict(specialist_shadow_evaluation.get("summary"))
    specialist_models = _safe_list(specialist_shadow_evaluation.get("models"))
    specialist_report_missing = not bool(specialist_shadow_evaluation.get("available"))
    status = _status_from_counts(
        critical=hard_failure
        and (bool(model_critical) or local_tools_hard_missing or runtime_probe_hard_failure),
        warning=hard_failure or observing or specialist_report_missing,
    )
    summary = "模型和训练数据状态正常。"
    if hard_failure:
        summary = "模型服务或训练数据源存在硬故障，需要处理。"
    elif observing:
        observing_reasons: list[str] = []
        if optional_source_warnings:
            observing_reasons.append("可选增强数据源未配置")
        if runtime_probe_timeout:
            observing_reasons.append("运行探针超时")
        if local_tools_status == "learning_only":
            observing_reasons.append("模型仍在学习观察")
        if local_tools_unconfigured:
            observing_reasons.append("本地量化工具未配置")
        if artifact_retirement_required:
            observing_reasons.append("Phase 3 artifact rebuild required")
        if artifact_retirement_audit_warning:
            observing_reasons.append("artifact retirement audit unavailable")
        reason_text = "、".join(observing_reasons) or "存在观察项"
        summary = f"模型服务可用；{reason_text}。"
    return _audit_card(
        "model_training",
        "模型与训练",
        status,
        summary,
        details={
            "local_ai_tools": {
                "available": bool(local_tools.get("available")),
                "status": local_tools.get("status"),
                "shadow_sample_count": local_tools.get("shadow_sample_count"),
                "trade_sample_count": local_tools.get("trade_sample_count"),
                "text_sentiment_sample_count": local_tools.get("text_sentiment_sample_count"),
                "training_mode": phase3_training_governance["training_mode"],
                "model_stage": phase3_training_governance["model_stage"],
                "promotion_flow": phase3_training_governance["promotion_flow"],
                "live_mutation": phase3_training_governance["live_mutation"],
                "evaluation_policy": phase3_training_governance["evaluation_policy"],
                "promotion_recommendation": (
                    local_tools.get("promotion_recommendation")
                    if isinstance(local_tools.get("promotion_recommendation"), dict)
                    else {}
                ),
                "models": (
                    local_tools.get("models") if isinstance(local_tools.get("models"), dict) else {}
                ),
            },
            "phase3_training_governance": phase3_training_governance,
            "phase3_rebuild_readiness": phase3_rebuild_readiness,
            "specialist_shadow_evaluation": {
                "available": bool(specialist_shadow_evaluation.get("available")),
                "generated_at": specialist_shadow_evaluation.get("generated_at"),
                "report_path": specialist_shadow_evaluation.get("report_path"),
                "completed_count": int(specialist_shadow_evaluation.get("completed_count") or 0),
                "eligible_shadow_count": int(
                    specialist_shadow_evaluation.get("eligible_shadow_count") or 0
                ),
                "model_count": int(specialist_shadow_evaluation.get("model_count") or 0),
                "promotion_ready_count": int(specialist_summary.get("promotion_ready_count") or 0),
                "blocked_count": int(specialist_summary.get("blocked_count") or 0),
                "summary": specialist_summary,
                "promotion_gate": (
                    specialist_shadow_evaluation.get("promotion_gate")
                    if isinstance(specialist_shadow_evaluation.get("promotion_gate"), dict)
                    else {}
                ),
                "top_blocked_reasons": _safe_list(specialist_summary.get("top_blocked_reasons"))[
                    :8
                ],
                "models": specialist_models[:8],
                "live_mutation": bool(specialist_shadow_evaluation.get("live_mutation")),
                "promotion_flow": specialist_shadow_evaluation.get("promotion_flow")
                or "shadow_to_canary_to_live",
                "reason": specialist_shadow_evaluation.get("reason"),
            },
            "historical_trade_fact_audit": historical_trade_fact_report,
            "artifact_retirement_audit": artifact_retirement_report,
            "governance_status": governance.get("status") if isinstance(governance, dict) else None,
            "runtime_probe": runtime_probe,
            "hard_failure": hard_failure,
            "observing": observing,
            "clean_training_view_available": clean_training_view_available,
            "local_tools_status_probe_slow": local_tools_status_probe_slow,
            "runtime_probe_timeout_is_observing": runtime_probe_timeout_is_observing,
            "source_warnings": hard_source_warnings[:8],
            "optional_source_warnings": optional_source_warnings[:8],
            "hard_source_warning_count": len(hard_source_warnings),
            "optional_source_warning_count": len(optional_source_warnings),
            "model_critical_items": model_critical[:8],
        },
        evidence=[
            {"label": "影子样本", "value": local_tools.get("shadow_sample_count") or 0},
            {"label": "交易样本", "value": local_tools.get("trade_sample_count") or 0},
            {"label": "文本样本", "value": local_tools.get("text_sentiment_sample_count") or 0},
            {"label": "训练阶段", "value": phase3_training_governance["model_stage"]},
            {"label": "晋级流程", "value": phase3_training_governance["promotion_flow"]},
            {
                "label": "Retired artifact",
                "value": artifact_retirement_report.get("retired_or_untrusted_count") or 0,
            },
            {
                "label": "Rebuild gate",
                "value": phase3_rebuild_readiness.get("status") or "unknown",
            },
            {
                "label": "Specialist shadow",
                "value": int(specialist_shadow_evaluation.get("eligible_shadow_count") or 0),
            },
        ],
        next_actions=[
            "模型 critical 时优先查端口契约 18000/18001/18002/18003 和 Phase 3 量化 API 健康状态。",
            "可选增强源未配置只影响新闻/事件覆盖，不应误判为模型训练硬故障。",
            "learning_only 表示模型可用但仍需效果验证，继续看高分组收益和样本质量。",
            "Retired or untrusted artifacts must be rebuilt from the Phase 3 clean training view before live influence.",
            "Phase 3 rebuild readiness must be ready before running --persist-artifact --confirm-phase3-rebuild.",
        ],
    )


def _source_scan_root() -> Path:
    return Path(__file__).resolve().parents[2]


async def _phase3_server_migration_audit() -> dict[str, Any]:
    report = await Phase3ServerMigrationAuditService(
        timeout_seconds=PHASE3_SERVER_MIGRATION_AUDIT_TIMEOUT_SECONDS
    ).report()
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    phase3_blocked = bool(report.get("phase3_go_live_blocked"))
    status = "warning" if phase3_blocked or warnings else "ok"
    summary = "Phase 3 model-server reset and migration gate is ready."
    if phase3_blocked:
        summary = (
            "Phase 3 model-server go-live is blocked until legacy resource release, "
            "/data/BB isolation, and whitelist migration are verified."
        )
    elif warnings:
        summary = "Phase 3 model-server gate is usable but has non-blocking migration warnings."
    return _audit_card(
        "phase3_server_migration",
        "Phase 3 server resource-release/migration gate",
        status,
        summary,
        details=report,
        evidence=[
            {"label": "Go-live blocked", "value": phase3_blocked},
            {"label": "Blockers", "value": len(blockers)},
            {"label": "Legacy data paths", "value": report.get("legacy_data_path_count") or 0},
            {"label": "Legacy services", "value": report.get("forbidden_service_count") or 0},
            {
                "label": "Migration items",
                "value": _safe_dict(report.get("migration_manifest")).get("item_count") or 0,
            },
        ],
        next_actions=[
            "Do not enable Phase 3 model-server production until this gate reports ready.",
            "Stop legacy services/processes/containers and keep old data isolated in place.",
            "Migrate only the approved whitelist manifest from the old server; never copy the old server wholesale.",
            "Keep Phase 3 model/cache/training/runtime/log data rooted under /data/BB.",
        ],
    )


async def _phase3_model_server_readiness_audit() -> dict[str, Any]:
    report = await Phase3ModelServerReadinessAuditService(
        timeout_seconds=PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS
    ).report()
    latest_report = _load_phase3_model_server_readiness_latest_report()
    if _phase3_model_readiness_report_verified(latest_report) and _phase3_model_readiness_probe_failed_before_remote(report):
        report = latest_report
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    artifact_ready = bool(report.get("artifact_ready"))
    runtime_ready = bool(report.get("runtime_ready"))
    service_go_live_blocked = bool(report.get("phase3_model_service_go_live_blocked"))
    if blockers:
        status = "critical"
    elif runtime_ready:
        status = "ok"
    else:
        status = "warning"
    summary = "Phase 3 quant model server artifacts and services are ready."
    if blockers:
        summary = (
            "Phase 3 quant model server is blocked by artifact, CUDA, GPU, or policy "
            "readiness failures."
        )
    elif not runtime_ready:
        summary = (
            "Phase 3 quant model artifacts are ready, but model-serving services/endpoints "
            "are not active yet."
        )
    elif warnings:
        summary = "Phase 3 quant model server is usable with non-blocking warnings."
    return _audit_card(
        "phase3_model_server_readiness",
        "Phase 3 quant model-server readiness",
        status,
        summary,
        details=report,
        evidence=[
            {"label": "Artifact ready", "value": artifact_ready},
            {"label": "Runtime ready", "value": runtime_ready},
            {"label": "Go-live blocked", "value": service_go_live_blocked},
            {"label": "GPU count", "value": report.get("gpu_count") or 0},
            {
                "label": "Required slots",
                "value": (
                    f"{report.get('required_slot_ready_count') or 0}/"
                    f"{report.get('required_slot_count') or 0}"
                ),
            },
            {
                "label": "Active endpoints",
                "value": report.get("active_endpoint_count") or 0,
            },
        ],
        next_actions=[
            "Do not route Phase 3 model calls to the new server until this gate reports runtime_ready=true.",
            "Keep LLM roles shadow/candidate-only until service health, latency, and promotion gates pass.",
            "Start or install audited Phase 3 model services from /data/BB only; do not reuse /data/trade_ai legacy services.",
            "After services are active, re-run this audit and then connect platform tunnels in shadow/canary mode.",
        ],
    )


def _load_phase3_model_server_readiness_latest_report() -> dict[str, Any]:
    path = settings.data_dir / "phase3_model_server_readiness_reports" / "latest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _phase3_model_readiness_report_verified(report: dict[str, Any]) -> bool:
    return bool(
        str(report.get("status") or "").lower() == "ready"
        and report.get("artifact_ready") is True
        and report.get("runtime_ready") is True
        and report.get("phase3_model_service_go_live_blocked") is False
    )


def _phase3_model_readiness_probe_failed_before_remote(report: dict[str, Any]) -> bool:
    if bool(report.get("remote_probe_available")):
        return False
    if str(report.get("status") or "").lower() not in {"unverified", "model_server_config_error"}:
        return False
    text = str(report.get("error") or "")
    return (
        "BB_SECURE_SETTINGS_KEY" in text
        or "Could not find server info file" in text
        or "Could not find model server info file" in text
    )


async def _phase3_paper_resume_preflight_audit() -> dict[str, Any]:
    report = await asyncio.wait_for(
        Phase3PaperResumePreflightService(
            okx_sync_provider=_okx_authoritative_sync_summary,
            model_server_timeout_seconds=PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS,
        ).report(),
        timeout=PHASE3_PAPER_RESUME_PREFLIGHT_TIMEOUT_SECONDS,
    )
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    can_resume = bool(report.get("can_resume_paper"))
    paper_active = _paper_service_active(
        _safe_dict(_safe_dict(report.get("inputs")).get("platform_server"))
    )
    consumed_after_resume = paper_active and not can_resume
    status = "ok" if can_resume and not warnings else "warning"
    if blockers:
        status = "critical"
    if consumed_after_resume:
        status = "warning"
    summary = "Phase 3 paper resume hard gate is ready; operator approval is still required."
    if consumed_after_resume:
        summary = "Phase 3 paper resume preflight has been consumed; post-resume observation is now authoritative."
    elif blockers:
        summary = "Phase 3 paper resume is blocked by hard preflight gates."
    elif warnings:
        summary = "Phase 3 paper resume gate is passable with warnings that need review."
    details = dict(report)
    details["consumed_after_resume"] = consumed_after_resume
    details["observing"] = consumed_after_resume
    return _audit_card(
        "phase3_paper_resume_preflight",
        "Phase 3 paper resume hard gate",
        status,
        summary,
        details=details,
        evidence=[
            {"label": "Can resume paper", "value": can_resume},
            {"label": "Blockers", "value": len(blockers)},
            {"label": "Warnings", "value": len(warnings)},
            {
                "label": "OKX issues",
                "value": _safe_dict(report.get("summary")).get("okx_issue_count") or 0,
            },
            {
                "label": "OKX equity",
                "value": bool(
                    _safe_dict(report.get("summary")).get("okx_account_equity_available")
                ),
            },
            {
                "label": "Model runtime",
                "value": bool(_safe_dict(report.get("summary")).get("model_server_runtime_ready")),
            },
            {
                "label": "Quant API",
                "value": bool(_safe_dict(report.get("summary")).get("phase3_quant_api_available")),
            },
        ],
        next_actions=[
            "Do not start bb-paper-trading.service until can_resume_paper=true.",
            "Clear OKX native sync, trade-fact integrity, model-server runtime, tunnel, and specialist-shadow blockers first.",
            "When the gate passes, resume paper only through an approved operator action and keep live trading disabled.",
            "After resume, watch OKX authoritative sync and specialist shadow evaluation for fresh samples.",
        ],
    )


async def _phase3_paper_resume_observation_audit() -> dict[str, Any]:
    report = await asyncio.wait_for(
        Phase3PaperResumeObservationService(
            okx_sync_provider=_okx_authoritative_sync_summary,
        ).report(),
        timeout=PHASE3_PAPER_RESUME_OBSERVATION_TIMEOUT_SECONDS,
    )
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    status_value = str(report.get("status") or "unknown")
    if blockers or status_value == "critical":
        status = "critical"
    elif status_value in {"waiting_for_resume", "warming_up"} or warnings:
        status = "warning"
    else:
        status = "ok"
    summary = "Phase 3 post-resume paper observation is healthy."
    if status_value == "waiting_for_resume":
        summary = (
            "Phase 3 post-resume observation is waiting because paper trading is still stopped."
        )
    elif status_value == "warming_up":
        summary = "Phase 3 paper has resumed but observation samples are still warming up."
    elif blockers:
        summary = "Phase 3 post-resume observation found hard blockers."
    elif warnings:
        summary = "Phase 3 post-resume observation has warnings that need review."
    details = dict(report)
    details["observing"] = status_value in {"waiting_for_resume", "warming_up"}
    return _audit_card(
        "phase3_paper_resume_observation",
        "Phase 3 post-resume paper observation",
        status,
        summary,
        details=details,
        evidence=[
            {"label": "Paper active", "value": bool(report.get("paper_active"))},
            {"label": "Blockers", "value": len(blockers)},
            {"label": "Warnings", "value": len(warnings)},
            {
                "label": "Shadow created",
                "value": _safe_dict(report.get("summary")).get("created_shadow_count") or 0,
            },
            {
                "label": "Shadow completed",
                "value": _safe_dict(report.get("summary")).get("completed_shadow_count") or 0,
            },
            {
                "label": "Specialist eligible",
                "value": _safe_dict(report.get("summary")).get("specialist_eligible_shadow_count")
                or 0,
            },
        ],
        next_actions=[
            "Before paper starts, use this card as the zero-sample baseline.",
            "After paper starts, watch the first 30/60/120 minutes for OKX clean state and sample accumulation.",
            "Do not promote specialist models until this observation is healthy and sample floors pass.",
        ],
    )


async def _phase3_stage_handoff_audit() -> dict[str, Any]:
    report = await asyncio.to_thread(Phase3StageHandoffService().report)
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    stage_status = str(report.get("status") or "waiting_for_evidence")
    if blockers or stage_status == "blocked":
        status = "critical"
        summary = "Phase 3 stage handoff is blocked by hard evidence gates."
    elif stage_status == "paper_start_ready":
        status = "warning"
        summary = "Phase 3 is paper-start ready, but bb-paper-trading.service still requires explicit operator approval."
    elif stage_status == "post_resume_observing":
        status = "warning"
        summary = "Phase 3 paper has entered the post-resume observation window."
    elif stage_status == "canary_review_ready":
        status = "warning"
        summary = (
            "Phase 3 evidence is ready for manual canary review; live routing remains disabled."
        )
    elif stage_status == "paper_observation_healthy":
        status = "warning"
        summary = (
            "Phase 3 paper observation is healthy; next promotion still requires manual review."
        )
    else:
        status = "warning"
        summary = "Phase 3 is still collecting shadow evidence before the next controlled stage."
    details = dict(report)
    details["observing"] = status == "warning"
    return _audit_card(
        "phase3_stage_handoff",
        "Phase 3 stage handoff",
        status,
        summary,
        details=details,
        evidence=[
            {"label": "Stage", "value": report.get("stage") or stage_status},
            {
                "label": "Can start paper",
                "value": bool(report.get("can_start_paper_with_operator_approval")),
            },
            {
                "label": "Can enter canary",
                "value": bool(report.get("can_enter_canary_with_operator_approval")),
            },
            {"label": "Can enter live", "value": bool(report.get("can_enter_live"))},
            {"label": "Blockers", "value": len(blockers)},
            {"label": "Warnings", "value": len(warnings)},
        ],
        next_actions=[
            "If blocked, fix the listed hard gates before any paper/canary/live action.",
            "If paper_start_ready, start paper only through scripts/start_phase3_paper_with_preflight.py with explicit confirmation.",
            "If post_resume_observing, keep collecting OKX native, shadow, specialist, and trade-quality evidence.",
            "If canary_review_ready, review evidence manually; live routing remains disabled from this handoff.",
        ],
    )


def _ensemble_top_level_parameter_audit(root: Path) -> dict[str, Any]:
    path = root / "ai_brain/ensemble_coordinator.py"
    hidden_constants: list[dict[str, Any]] = []
    top_level_constant_count = 0
    try:
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source)
    except Exception as exc:
        return {
            "top_level_constant_count": 0,
            "hidden_constants": [
                {
                    "name": "ensemble_coordinator_parse_error",
                    "line": 0,
                    "reason": safe_error_text(exc, limit=120),
                }
            ],
        }
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or not target.id.isupper():
                continue
            top_level_constant_count += 1
            if target.id in STRATEGY_ALLOWED_TOP_LEVEL_CONSTANTS:
                continue
            expression = ast.unparse(node.value)
            if any(token in expression for token in STRATEGY_PARAMETERIZED_TOKENS):
                continue
            hidden_constants.append(
                {
                    "name": target.id,
                    "line": node.lineno,
                    "expression": expression[:180],
                }
            )
    return {
        "top_level_constant_count": top_level_constant_count,
        "hidden_constants": hidden_constants,
    }


def _iter_source_scan_files() -> list[Path]:
    root = _source_scan_root()
    files: list[Path] = []
    for dirname, pattern in SOURCE_MOJIBAKE_SCAN_TARGETS:
        base = root / dirname
        if not base.exists():
            continue
        files.extend(path for path in base.rglob(pattern) if path.is_file())
    return sorted(set(files), key=lambda item: item.as_posix())


def _relative_source_path(path: Path) -> str:
    try:
        return path.relative_to(_source_scan_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _source_visible_text_audit() -> dict[str, Any]:
    offenders: list[dict[str, Any]] = []
    scanned = 0
    for path in _iter_source_scan_files():
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = sorted({marker for marker in SOURCE_MOJIBAKE_MARKERS if marker in text})
        if hits:
            offenders.append(
                {
                    "path": _relative_source_path(path),
                    "markers": [
                        marker.encode("unicode_escape").decode("ascii") for marker in hits[:8]
                    ],
                }
            )
    status = "critical" if offenders else "ok"
    return _audit_card(
        "visible_text_encoding",
        "中文显示与乱码回归",
        status,
        (
            "源码和前端静态资源未发现裸乱码。"
            if not offenders
            else "发现源码/前端静态资源重新出现裸乱码。"
        ),
        details={
            "scanned_files": scanned,
            "offender_count": len(offenders),
            "offenders": offenders[:20],
            "scope": [f"{dirname}/{pattern}" for dirname, pattern in SOURCE_MOJIBAKE_SCAN_TARGETS],
        },
        evidence=[
            {"label": "扫描文件", "value": scanned},
            {"label": "乱码文件", "value": len(offenders)},
        ],
        next_actions=[
            "若乱码文件不为 0，先定位来源是源码文案、模型返回还是历史数据，不要只在前端替换显示。",
            "历史数据修复样本必须使用 Unicode 转义，不允许裸乱码写入源码。",
        ],
    )


async def _runtime_text_integrity_audit() -> dict[str, Any]:
    try:
        report = await collect_runtime_text_integrity_report(
            hours=AUDIT_WINDOWS["strategy_hours"],
            limit_per_table=200,
            example_limit=12,
        )
    except Exception as exc:
        return _audit_card(
            "runtime_text_integrity",
            "运行时文本完整性",
            "warning",
            "运行时文本完整性巡检读取失败；本次未修改任何历史数据。",
            details={"error": safe_error_text(exc, limit=180), "dry_run": True},
            evidence=[
                {"label": "扫描记录", "value": 0},
                {"label": "疑似记录", "value": 0},
            ],
            next_actions=[
                "先检查数据库连接和 scripts/audit_runtime_text_integrity.py --help 输出。",
                "不要为了消除告警批量覆盖历史文本，先用 dry-run 示例定位来源。",
            ],
        )
    status = str(report.get("status") or "ok")
    suspected_records = int(report.get("suspected_records") or 0)
    summary = (
        "运行时写入文本未发现新增疑似乱码。"
        if suspected_records <= 0
        else "运行时记录中发现疑似乱码，需要定位写入边界或历史污染来源。"
    )
    return _audit_card(
        "runtime_text_integrity",
        "运行时文本完整性",
        "warning" if status == "warning" or suspected_records else "ok",
        summary,
        details=report,
        evidence=[
            {"label": "扫描记录", "value": int(report.get("scanned_records") or 0)},
            {"label": "疑似记录", "value": suspected_records},
            {"label": "疑似字段", "value": int(report.get("suspected_fields") or 0)},
            {"label": "可自动修复字段", "value": int(report.get("repairable_count") or 0)},
        ],
        next_actions=[
            "若疑似记录不为 0，先用审计脚本 dry-run 示例确认来源表和字段。",
            "优先修写入边界，禁止直接覆盖原始历史记录。",
            "可修复字段只生成修复报告，批量修复必须另走人工确认。",
        ],
    )


def _phase3_go_no_go_audit_from_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    report = evaluate_phase3_go_no_go_cards(cards)
    status_key = str(report.get("status") or "blocked")
    blockers = _safe_list(report.get("blockers"))
    warnings = _safe_list(report.get("warnings"))
    if blockers:
        card_status = "critical"
        summary = "Phase 3 cannot advance; foundation or promotion hard gates are blocked."
    elif status_key == "paper_resume_ready":
        card_status = "warning"
        summary = "Phase 3 can only resume paper through the controlled operator-approved path."
    elif status_key == "post_resume_observing":
        card_status = "warning"
        summary = (
            "Phase 3 paper is active; post-resume OKX and shadow evidence is still warming up."
        )
    elif status_key == "paper_observation_healthy":
        card_status = "warning"
        summary = (
            "Phase 3 paper observation is healthy; canary review still requires operator approval."
        )
    else:
        card_status = "warning"
        summary = "Phase 3 remains shadow-only while evidence continues to accumulate."
    return _audit_card(
        "phase3_go_no_go",
        "Phase 3 Go/No-Go 总闸门",
        card_status,
        summary,
        details=report,
        evidence=[
            {"label": "Next step", "value": report.get("next_step")},
            {
                "label": "Can start paper",
                "value": bool(report.get("can_start_paper_with_operator_approval")),
            },
            {
                "label": "Can enter canary",
                "value": bool(report.get("can_enter_canary_with_operator_approval")),
            },
            {"label": "Can enter live", "value": bool(report.get("can_enter_live"))},
            {"label": "Blockers", "value": len(blockers)},
            {"label": "Warnings", "value": len(warnings)},
        ],
        next_actions=[
            "If blocked, fix the listed hard gates before any paper/canary/live action.",
            "If paper_resume_ready, paper can only be started through the controlled preflight start command and explicit operator approval.",
            "If post_resume_observing, keep collecting OKX native, shadow, specialist, and trade-quality evidence.",
            "If paper_observation_healthy, review canary promotion evidence manually; do not enable live routing automatically.",
            "Live remains unavailable from this gate; it requires a separate operator-approved release step.",
        ],
    )


def _strategy_gate_contract_audit() -> dict[str, Any]:
    parameter_audit = _ensemble_top_level_parameter_audit(_source_scan_root())
    root = _source_scan_root()
    scan_paths = (
        root / "ai_brain",
        root / "services",
        root / "risk_manager",
        root / "web_dashboard/api/dashboard.py",
    )
    files: list[Path] = []
    for path in scan_paths:
        if path.is_file():
            files.append(path)
        elif path.exists():
            files.extend(candidate for candidate in path.rglob("*.py") if candidate.is_file())
    offenders: list[dict[str, Any]] = []
    for path in sorted(set(files), key=lambda item: item.as_posix()):
        rel_path = _relative_source_path(path)
        if rel_path in STRATEGY_GATE_ALLOWED_PATHS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [pattern for pattern in STRATEGY_GATE_FORBIDDEN_PATTERNS if pattern in text]
        if hits:
            offenders.append({"path": rel_path, "patterns": hits})
    try:
        runtime_source = ast.parse(
            (root / "services/runtime_entry_filters.py").read_text(encoding="utf-8")
        )
        runtime_contract_available = any(
            isinstance(node, ast.ClassDef) and node.name == "RuntimeEntryFilters"
            for node in ast.walk(runtime_source)
        )
    except Exception:
        runtime_contract_available = False
    hidden_strategy_constants = parameter_audit["hidden_constants"]
    status = (
        "critical"
        if offenders or hidden_strategy_constants or not runtime_contract_available
        else "ok"
    )
    return _audit_card(
        "strategy_gate_contract",
        "策略门槛契约",
        status,
        (
            "策略运行时门槛保持解释/排序/仓位参考，不是固定硬开仓门槛。"
            if status == "ok"
            else "发现固定门槛或死分支残留，可能重新把策略卡死。"
        ),
        details={
            "trading_parameter_version": DEFAULT_TRADING_PARAMS.version,
            "runtime_contract_available": runtime_contract_available,
            "forbidden_patterns": list(STRATEGY_GATE_FORBIDDEN_PATTERNS),
            "offender_count": len(offenders),
            "offenders": offenders[:20],
            "ensemble_top_level_constant_count": parameter_audit["top_level_constant_count"],
            "hidden_strategy_constant_count": len(hidden_strategy_constants),
            "hidden_strategy_constants": hidden_strategy_constants[:20],
            "allowed_top_level_constants": sorted(STRATEGY_ALLOWED_TOP_LEVEL_CONSTANTS),
        },
        evidence=[
            {"label": "策略参数版本", "value": DEFAULT_TRADING_PARAMS.version},
            {"label": "固定门槛残留", "value": len(offenders)},
            {"label": "隐藏策略常量", "value": len(hidden_strategy_constants)},
            {"label": "运行时契约", "value": "存在" if runtime_contract_available else "缺失"},
        ],
        next_actions=[
            "如发现 settings.min_entry_* 直接参与运行链路，必须改为 RuntimeEntryFilters 动态参考。",
            "低量比、ADX、置信度只能影响排序、仓位、杠杆和解释；不能作为硬开仓门槛。",
        ],
    )


async def _strategy_closed_loop_audit() -> dict[str, Any]:
    since = _now() - timedelta(hours=AUDIT_WINDOWS["strategy_hours"])
    runtime_window = _load_trading_runtime_audit_window()
    async with get_session_ctx() as session:
        decisions = list(
            (
                await session.execute(
                    select(
                        AIDecision.id,
                        AIDecision.symbol,
                        AIDecision.action,
                        AIDecision.position_size_pct,
                        AIDecision.raw_llm_response,
                        AIDecision.was_executed,
                        AIDecision.outcome_pnl_pct,
                        AIDecision.created_at,
                    )
                    .where(AIDecision.created_at >= since)
                    .order_by(AIDecision.created_at.desc())
                    .limit(500)
                )
            ).all()
        )
        orders = list(
            (
                await session.execute(
                    select(Order.decision_id, Order.status)
                    .where(Order.created_at >= since)
                    .order_by(Order.created_at.desc())
                    .limit(1000)
                )
            ).all()
        )
        positions = list(
            (
                await session.execute(
                    select(
                        Position.id,
                        Position.symbol,
                        Position.side,
                        Position.quantity,
                        Position.entry_price,
                        Position.realized_pnl,
                        Position.is_open,
                        Position.created_at,
                        Position.closed_at,
                        Position.entry_exchange_order_id,
                        Position.close_exchange_order_id,
                    )
                    .where(Position.created_at >= since)
                    .order_by(Position.created_at.desc())
                    .limit(800)
                )
            ).all()
        )
        trade_outcome_rows = (
            await session.execute(
                select(TradeReflection.outcome, func.count())
                .where(TradeReflection.created_at >= since)
                .group_by(TradeReflection.outcome)
            )
        ).all()
        shadow_action_rows = (
            await session.execute(
                select(ShadowBacktest.decision_action, func.count())
                .where(
                    ShadowBacktest.created_at >= since,
                    ShadowBacktest.status == "completed",
                )
                .group_by(ShadowBacktest.decision_action)
            )
        ).all()

    entry_decisions = [
        row for row in decisions if str(row.action or "").lower() in {"long", "short"}
    ]
    hold_decisions = [row for row in decisions if str(row.action or "").lower() == "hold"]
    executed_entries = [row for row in entry_decisions if bool(row.was_executed)]
    filled_orders_by_decision: dict[int, list[Order]] = {}
    for order in orders:
        if str(order.status or "").lower() != "filled" or order.decision_id is None:
            continue
        filled_orders_by_decision.setdefault(int(order.decision_id), []).append(order)

    tier_counts = Counter(_decision_evidence_tier(row) or "missing" for row in entry_decisions)
    high_quality_tiers = {"exploration", "small", "medium", "normal"}
    weak_tiers = {"weak_conflict_probe", "degraded_missing_probe"}
    high_quality_entries = [
        row for row in entry_decisions if _decision_evidence_tier(row) in high_quality_tiers
    ]
    weak_entries = [row for row in entry_decisions if _decision_evidence_tier(row) in weak_tiers]
    weak_executed = [row for row in executed_entries if _decision_evidence_tier(row) in weak_tiers]
    shadow_only_executed = [
        row
        for row in executed_entries
        if bool(_decision_raw(row).get("entry_evidence_shadow_only"))
    ]
    executed_without_order = [
        row for row in executed_entries if not filled_orders_by_decision.get(int(row.id))
    ]

    component_stats: dict[str, Counter[str]] = {}
    for row in entry_decisions:
        components = _decision_evidence(row).get("components")
        if not isinstance(components, list):
            continue
        for component in components:
            if not isinstance(component, dict):
                continue
            source = str(component.get("source") or "unknown")
            status = str(component.get("status") or "unknown")
            component_stats.setdefault(source, Counter())[status] += 1

    expected_net_values = [
        value for row in entry_decisions if (value := _decision_expected_net(row)) is not None
    ]
    positive_net_count = sum(1 for value in expected_net_values if value > 0)
    negative_net_count = sum(1 for value in expected_net_values if value < 0)
    closed_positions_all = [row for row in positions if not bool(row.is_open)]
    closed_positions = [
        row for row in closed_positions_all if closed_position_trade_fact_trusted(row)
    ]
    quarantined_closed_position_count = len(closed_positions_all) - len(closed_positions)
    realized_values = [_safe_float(row.realized_pnl) for row in closed_positions]
    win_count = sum(1 for value in realized_values if value > 0)
    loss_count = sum(1 for value in realized_values if value < 0)
    fast_loss_samples: list[dict[str, Any]] = []
    for position in closed_positions:
        if not isinstance(position.created_at, datetime) or not isinstance(
            position.closed_at, datetime
        ):
            continue
        opened = (
            position.created_at.replace(tzinfo=UTC)
            if position.created_at.tzinfo is None
            else position.created_at
        )
        closed = (
            position.closed_at.replace(tzinfo=UTC)
            if position.closed_at.tzinfo is None
            else position.closed_at
        )
        hold_minutes = max((closed - opened).total_seconds() / 60.0, 0.0)
        pnl = _safe_float(position.realized_pnl)
        if hold_minutes <= 15 and pnl < 0:
            fast_loss_samples.append(
                {
                    "id": position.id,
                    "symbol": position.symbol,
                    "side": position.side,
                    "hold_minutes": round(hold_minutes, 3),
                    "realized_pnl": round(pnl, 8),
                    "notional_usdt": round(
                        abs(_safe_float(position.quantity) * _safe_float(position.entry_price)),
                        6,
                    ),
                    "closed_at": _iso(closed),
                }
            )

    notional_values = [
        abs(_safe_float(row.quantity) * _safe_float(row.entry_price))
        for row in positions
        if abs(_safe_float(row.quantity) * _safe_float(row.entry_price)) > 0
    ]
    shadow_action_counts = {
        str(action or "unknown"): int(count or 0) for action, count in shadow_action_rows
    }
    trade_outcome_counts = {
        str(outcome or "unknown"): int(count or 0) for outcome, count in trade_outcome_rows
    }
    ml_stats = dict(component_stats.get("ml", Counter()))
    ml_influence_reason = _ml_influence_reason_from_decisions(entry_decisions)
    ml_usable = sum(
        count
        for status, count in ml_stats.items()
        if status not in {"ignored", "missing", "unknown"}
    )
    ml_total = sum(ml_stats.values())
    ml_usable_rate = round(ml_usable / ml_total, 4) if ml_total else 0.0
    executed_with_outcome = [row for row in executed_entries if row.outcome_pnl_pct is not None]
    high_quality_outcomes = [
        _safe_float(row.outcome_pnl_pct)
        for row in executed_with_outcome
        if _decision_evidence_tier(row) in high_quality_tiers
    ]
    weak_outcomes = [
        _safe_float(row.outcome_pnl_pct)
        for row in executed_with_outcome
        if _decision_evidence_tier(row) in weak_tiers
    ]
    current_window = _summarize_strategy_window(
        started_at=runtime_window.get("started_at"),
        decisions=decisions,
        positions=positions,
        high_quality_tiers=high_quality_tiers,
        weak_tiers=weak_tiers,
    )
    current_window["heartbeat_at"] = runtime_window.get("heartbeat_at_iso")
    current_window["running"] = bool(runtime_window.get("running"))
    current_window["mode"] = runtime_window.get("mode")
    current_window["decision_interval"] = runtime_window.get("decision_interval")
    current_window["historical_legacy_issues"] = bool(
        weak_executed or fast_loss_samples
    ) and not bool(
        current_window.get("weak_executed_count") or current_window.get("fast_loss_under_15m_count")
    )
    effectiveness_verdict = "样本不足，不能证明 ML/策略有效"
    if len(high_quality_outcomes) >= 5:
        high_avg = sum(high_quality_outcomes) / len(high_quality_outcomes)
        weak_avg = sum(weak_outcomes) / len(weak_outcomes) if weak_outcomes else 0.0
        if high_avg > max(weak_avg, 0.0):
            effectiveness_verdict = "高质量信号样本暂时优于弱证据样本"
        else:
            effectiveness_verdict = "高质量信号未表现出收益优势，需要降权复查"

    diagnostics = {
        "current_weak_executed": bool(current_window.get("weak_executed_count")),
        "historical_weak_executed": bool(weak_executed),
        "shadow_only_executed": bool(shadow_only_executed),
        "executed_without_order": bool(executed_without_order),
        "current_no_high_quality_entries": (
            int(current_window.get("entry_decision_count") or 0) >= 20
            and not int(current_window.get("high_quality_entry_count") or 0)
        ),
        "historical_no_high_quality_entries": (
            len(entry_decisions) >= 20 and not high_quality_entries
        ),
        "current_fast_loss_cluster": int(current_window.get("fast_loss_under_15m_count") or 0) >= 3,
        "historical_fast_loss_cluster": len(fast_loss_samples) >= 3,
        "current_ml_not_effective": (
            int(current_window.get("entry_decision_count") or 0) >= 10
            and float(current_window.get("ml_usable_rate") or 0.0) < 0.25
        ),
        "historical_ml_not_effective": ml_total >= 10 and ml_usable_rate < 0.25,
        "insufficient_effectiveness_samples": len(executed_with_outcome) < 10,
        "historical_legacy_issues": bool(current_window.get("historical_legacy_issues")),
    }
    critical = diagnostics["shadow_only_executed"] or diagnostics["executed_without_order"]
    current_execution_warning = any(
        diagnostics[key]
        for key in (
            "current_weak_executed",
            "current_no_high_quality_entries",
            "current_fast_loss_cluster",
        )
    )
    current_ml_warning = diagnostics["current_ml_not_effective"]
    current_warning = current_execution_warning or current_ml_warning
    historical_warning = any(
        diagnostics[key]
        for key in (
            "historical_weak_executed",
            "historical_no_high_quality_entries",
            "historical_fast_loss_cluster",
            "historical_ml_not_effective",
            "insufficient_effectiveness_samples",
        )
    )
    warning = current_warning or historical_warning
    if critical:
        status = "critical"
        summary = "策略闭环存在执行状态硬错误，需要先修执行契约。"
    elif current_execution_warning:
        status = "warning"
        summary = "当前运行窗口仍存在弱证据执行、高质量候选不足或快亏平风险。"
    elif current_ml_warning:
        status = "warning"
        summary = (
            "当前运行窗口 ML 仍未有效参与；执行硬错误暂未复现，需继续治理 ML readiness 与收益样本。"
        )
    elif warning:
        status = "warning"
        summary = "24小时历史窗口仍有遗留问题；当前运行窗口暂未复现硬执行错误，需继续观察新样本。"
    else:
        status = "ok"
        summary = "策略闭环关键节点暂未发现硬异常。"

    return _audit_card(
        "strategy_closed_loop",
        "策略闭环审计",
        status,
        summary,
        details={
            "window_hours": AUDIT_WINDOWS["strategy_hours"],
            "current_runtime_window": current_window,
            "decision_count": len(decisions),
            "entry_decision_count": len(entry_decisions),
            "hold_decision_count": len(hold_decisions),
            "executed_entry_count": len(executed_entries),
            "evidence_tier_counts": dict(tier_counts),
            "high_quality_entry_count": len(high_quality_entries),
            "weak_entry_count": len(weak_entries),
            "weak_executed_count": len(weak_executed),
            "shadow_only_executed_count": len(shadow_only_executed),
            "executed_without_filled_order_count": len(executed_without_order),
            "expected_net_distribution": _distribution(expected_net_values),
            "positive_expected_net_count": positive_net_count,
            "negative_expected_net_count": negative_net_count,
            "model_component_status_counts": {
                key: dict(value) for key, value in sorted(component_stats.items())
            },
            "ml_usable_rate": ml_usable_rate,
            "ml_influence_reason": ml_influence_reason,
            "position_notional_distribution": _distribution(notional_values),
            "closed_position_count": len(closed_positions),
            "closed_position_raw_count": len(closed_positions_all),
            "trade_fact_quarantined_closed_position_count": quarantined_closed_position_count,
            "closed_win_count": win_count,
            "closed_loss_count": loss_count,
            "realized_pnl_distribution": _distribution(realized_values),
            "fast_loss_under_15m_count": len(fast_loss_samples),
            "fast_loss_under_15m_samples": fast_loss_samples[:10],
            "sampled_decision_limit": 500,
            "sampled_order_limit": 1000,
            "sampled_position_limit": 800,
            "shadow_action_counts": shadow_action_counts,
            "trade_reflection_outcome_counts": trade_outcome_counts,
            "executed_outcome_sample_count": len(executed_with_outcome),
            "high_quality_outcome_distribution": _distribution(high_quality_outcomes),
            "weak_outcome_distribution": _distribution(weak_outcomes),
            "effectiveness_verdict": effectiveness_verdict,
            "diagnostics": diagnostics,
            "weak_executed_samples": [
                {
                    "id": row.id,
                    "symbol": row.symbol,
                    "action": row.action,
                    "tier": _decision_evidence_tier(row),
                    "expected_net_return_pct": _decision_expected_net(row),
                    "created_at": _iso(row.created_at),
                }
                for row in weak_executed[:10]
            ],
            "executed_without_filled_order_samples": [
                {
                    "id": row.id,
                    "symbol": row.symbol,
                    "action": row.action,
                    "created_at": _iso(row.created_at),
                }
                for row in executed_without_order[:10]
            ],
        },
        evidence=[
            {"label": "当前弱证据执行", "value": current_window.get("weak_executed_count") or 0},
            {"label": "当前快亏平", "value": current_window.get("fast_loss_under_15m_count") or 0},
            {"label": "高质量候选", "value": len(high_quality_entries)},
            {"label": "弱证据已执行", "value": len(weak_executed)},
            {"label": "快亏平", "value": len(fast_loss_samples)},
            {"label": "ML可用率", "value": ml_usable_rate},
            {"label": "收益样本", "value": len(executed_with_outcome)},
        ],
        next_actions=[
            "先确认高质量候选是否持续为 0；如果是，问题在上游模型/收益计算/证据融合，不要放开弱证据。",
            "弱证据已执行或 shadow-only 已执行不应出现；出现时先查执行绕过路径。",
            "ML 可用率低时先看 ml_influence_reason：如果仍是学习观察模式，要先解决训练质量/上线条件，而不是放宽开仓。",
            "快亏平集中时先查平仓原因和持仓时间，不把亏损探针继续喂给训练。",
        ],
    )


def _root_cause_findings(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    cards_by_key = {str(card.get("key") or ""): card for card in cards}
    for card in cards:
        status = str(card.get("status") or "info")
        if status == "ok":
            continue
        state, state_label = _issue_ledger_state(card, cards_by_key=cards_by_key)
        findings.append(
            {
                "key": card.get("key"),
                "title": card.get("title"),
                "severity": status,
                "state": state,
                "state_label": state_label,
                "owner_path": card.get("owner_path")
                or _owner_path_for_card(str(card.get("key") or "")),
                "summary": card.get("summary"),
                "evidence": card.get("evidence") or [],
                "next_actions": card.get("next_actions") or [],
            }
        )
    state_rank = {"unresolved": 0, "observing": 1, "fixed": 2}
    return sorted(
        findings,
        key=lambda row: (
            STATUS_RANK.get(str(row.get("severity")), 9),
            state_rank.get(str(row.get("state")), 9),
        ),
    )[:10]


_STRATEGY_CLOSED_LOOP_CURRENT_DIAGNOSTICS = (
    "current_weak_executed",
    "current_no_high_quality_entries",
    "current_fast_loss_cluster",
    "shadow_only_executed",
    "executed_without_order",
)

_STRATEGY_CLOSED_LOOP_OBSERVATION_DIAGNOSTICS = (
    "current_ml_not_effective",
    "historical_weak_executed",
    "historical_no_high_quality_entries",
    "historical_fast_loss_cluster",
    "historical_ml_not_effective",
    "insufficient_effectiveness_samples",
    "historical_legacy_issues",
)


def _strategy_closed_loop_observation_only(card: dict[str, Any]) -> bool:
    if str(card.get("key") or "") != "strategy_closed_loop":
        return False
    if str(card.get("status") or "info") == "ok":
        return False
    details = card.get("details") if isinstance(card.get("details"), dict) else {}
    current_window = (
        details.get("current_runtime_window")
        if isinstance(details.get("current_runtime_window"), dict)
        else {}
    )
    diagnostics = details.get("diagnostics") if isinstance(details.get("diagnostics"), dict) else {}
    if any(bool(diagnostics.get(key)) for key in _STRATEGY_CLOSED_LOOP_CURRENT_DIAGNOSTICS):
        return False
    return bool(current_window.get("historical_legacy_issues")) or any(
        bool(diagnostics.get(key)) for key in _STRATEGY_CLOSED_LOOP_OBSERVATION_DIAGNOSTICS
    )


def _card_historical_observation_only(card: dict[str, Any]) -> bool:
    if str(card.get("status") or "info") == "ok":
        return False
    details = card.get("details") if isinstance(card.get("details"), dict) else {}
    current_window = (
        details.get("current_runtime_window")
        if isinstance(details.get("current_runtime_window"), dict)
        else {}
    )
    if not bool(current_window.get("historical_legacy_issues")):
        return False
    diagnostics = details.get("diagnostics") if isinstance(details.get("diagnostics"), dict) else {}
    return not any(bool(diagnostics.get(key)) for key in _STRATEGY_CLOSED_LOOP_CURRENT_DIAGNOSTICS)


def _strategy_closed_loop_is_historical_only(cards_by_key: dict[str, dict[str, Any]]) -> bool:
    card = cards_by_key.get("strategy_closed_loop") or {}
    return _strategy_closed_loop_observation_only(card)


def _issue_ledger_state(
    card: dict[str, Any],
    *,
    cards_by_key: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, str]:
    status = str(card.get("status") or "info")
    key = str(card.get("key") or "")
    details = card.get("details") if isinstance(card.get("details"), dict) else {}
    observation_only = _strategy_closed_loop_observation_only(
        card
    ) or _card_historical_observation_only(card)
    if status == "ok":
        return "fixed", "已修复 / 当前验证通过"
    if observation_only:
        return "observing", "历史/样本观察 / 当前未复现硬错误"
    if (
        key == "profit_first_ranking"
        and status == "warning"
        and _profit_first_ranking_observation_only(details)
    ):
        return "observing", "Observation / Profit-First ranking budget guard"
    if status == "warning" and bool(details.get("observing")):
        return "observing", "Observation / controlled stage or warmup"
    if (
        key == "model_training"
        and status == "warning"
        and bool(details.get("observing"))
        and not bool(details.get("hard_failure"))
    ):
        return "observing", "观察项 / 可选增强或学习模式"
    if key == "okx_reconciliation" and status == "warning" and bool(details.get("timeout")):
        return "observing", "观察项 / 对账巡检超时"
    if key == "market_data" and status == "warning" and bool(details.get("warmup_observing")):
        return "observing", "Observation / market-data warmup coverage expanding"
    if key == "okx_trade_fact_integrity" and status == "warning":
        runtime_gate = _safe_dict(details.get("runtime_okx_entry_gate"))
        runtime_blocker = str(runtime_gate.get("blocker") or "")
        link_repair = _safe_dict(details.get("position_fact_link_repair"))
        authoritative_sync = _safe_dict(details.get("okx_authoritative_sync"))
        runtime_only_blocked = runtime_blocker in {
            "runtime_heartbeat_unavailable",
            "trading_runtime_inactive",
            "trading_runtime_heartbeat_stale",
        }
        has_data_integrity_issue = any(
            int(value or 0) > 0
            for value in (
                details.get("critical_count"),
                details.get("warning_count"),
                _okx_unresolved_link_candidate_count(details, link_repair),
                _safe_dict(authoritative_sync.get("severity_counts")).get("critical"),
                _safe_dict(authoritative_sync.get("severity_counts")).get("warning"),
                authoritative_sync.get("manual_review_count"),
                authoritative_sync.get("repairable_count"),
            )
        )
        authoritative_pull_failed = (
            "okx_pull_available" in authoritative_sync
            and authoritative_sync.get("okx_pull_available") is False
        )
        runtime_sync_healthy = (
            runtime_gate.get("entry_blocked") is False
            and runtime_gate.get("sync_status") == "ok"
            and int(runtime_gate.get("last_requires_attention_count") or 0) == 0
        )
        if (
            (runtime_only_blocked or authoritative_pull_failed)
            and not has_data_integrity_issue
            and (runtime_only_blocked or runtime_sync_healthy)
        ):
            return "observing", "Observation / OKX runtime sync healthy or runtime-only state"
        if not has_data_integrity_issue and runtime_sync_healthy:
            severity_counts = _safe_dict(details.get("severity_counts"))
            issues = _safe_list(details.get("issues"))
            info_only = (
                int(severity_counts.get("critical") or 0) == 0
                and int(severity_counts.get("warning") or 0) == 0
                and all(
                    not isinstance(issue, dict)
                    or str(issue.get("severity") or "info").lower() == "info"
                    for issue in issues
                )
            )
            if info_only:
                return "observing", "Observation / OKX integrity info-only residuals"
    if key == "phase3_go_no_go" and status == "warning":
        next_step = str(details.get("next_step") or "")
        status_key = str(details.get("status") or "")
        if (
            status_key == "paper_resume_ready"
            and next_step == "resume_paper_pending_operator_approval"
            and bool(details.get("can_start_paper_with_operator_approval"))
            and not bool(details.get("can_enter_live"))
            and not _safe_list(details.get("blockers"))
        ):
            return "observing", "Observation / paper resume pending operator approval"
        if (
            status_key == "post_resume_observing"
            and next_step == "continue_post_resume_observation"
            and not bool(details.get("can_start_paper_with_operator_approval"))
            and not bool(details.get("can_enter_live"))
            and not _safe_list(details.get("blockers"))
        ):
            return "observing", "Observation / post-resume paper evidence warming"
    if key == "phase3_stage_handoff" and status == "warning":
        stage_status = str(details.get("status") or "")
        if (
            stage_status
            in {
                "paper_start_ready",
                "post_resume_observing",
                "canary_review_ready",
                "paper_observation_healthy",
                "waiting_for_evidence",
            }
            and bool(details.get("audit_only"))
            and bool(details.get("read_only"))
            and not bool(details.get("starts_trading_service"))
            and not bool(details.get("submits_orders"))
            and not bool(details.get("changes_model_routing"))
            and not bool(details.get("live_mutation"))
            and not bool(details.get("can_enter_live"))
            and not _safe_list(details.get("blockers"))
        ):
            if stage_status == "paper_start_ready":
                return "observing", "Observation / paper start pending operator approval"
            if stage_status == "canary_review_ready":
                return "observing", "Observation / canary review pending operator approval"
            return "observing", "Observation / Phase 3 controlled stage handoff"
    if (
        key == "trade_loop"
        and status == "warning"
        and bool(details.get("paper_resume_pending_operator_approval"))
    ):
        return "observing", "Observation / paper resume pending operator approval"
    if key == "trade_loop" and status == "warning" and bool(details.get("cold_start")):
        return "observing", "观察项 / 服务冷启动"
    if key == "trade_loop" and status == "warning" and bool(details.get("market_analysis_paused")):
        return "observing", "观察项 / 新币种分析暂停"
    if key == "trade_loop" and status == "warning" and bool(details.get("orderless_observation")):
        return "observing", "观察项 / 有分析但当前未触发订单"
    if (
        key == "model_expert_health"
        and status == "warning"
        and bool(details.get("audit_only"))
        and not bool(details.get("live_weight_mutation"))
        and not bool(details.get("disable_or_replace_count"))
        and not bool(details.get("reduce_weight_count"))
    ):
        return "observing", "观察项 / 只读影子体检"
    if (
        key == "model_expert_competition"
        and status == "warning"
        and bool(details.get("audit_only"))
        and not bool(details.get("live_weight_mutation"))
        and not bool(details.get("can_apply_live_weight"))
    ):
        return "observing", "观察项 / baseline 或竞赛样本不足"
    if (
        key == "model_dynamic_routing"
        and status == "warning"
        and bool(details.get("audit_only"))
        and not bool(details.get("live_route_mutation"))
        and not bool(details.get("can_apply_live_route"))
        and not bool(details.get("unsafe_live_mutation_attempts"))
    ):
        return "observing", "观察项 / 动态路由影子阶段"
    if (
        key == "high_risk_review_audit"
        and status == "warning"
        and bool(details.get("audit_only"))
        and bool(details.get("read_only"))
        and not bool(details.get("live_entry_mutation"))
        and not bool(details.get("can_bypass_risk_controls"))
        and not bool(details.get("can_force_open"))
        and not bool(
            _safe_dict(details.get("summary")).get("executed_without_required_review_count")
        )
    ):
        return "observing", "Observation / high-risk review gate"
    if (
        key == "crypto_feature_coverage"
        and status == "warning"
        and bool(details.get("audit_only"))
        and not bool(details.get("live_signal_mutation"))
        and not bool(details.get("can_missing_features_drive_live_entry"))
        and bool(details.get("feature_defaults_are_neutral"))
    ):
        return "observing", "观察项 / 缺失特征已中性阻断"
    if (
        key == "shadow_missed_opportunity"
        and status == "warning"
        and bool(details.get("audit_only"))
        and not bool(details.get("live_entry_mutation"))
        and not bool(details.get("can_bypass_risk_controls"))
        and not bool(details.get("weak_evidence_execution_allowed"))
        and not bool(details.get("global_missed_count_can_drive_entries"))
    ):
        return "observing", "观察项 / missed opportunity 保守学习"
    if (
        key == "strong_opportunity"
        and status == "warning"
        and bool(details.get("audit_only"))
        and not bool(details.get("live_entry_mutation"))
        and not bool(details.get("live_sizing_mutation"))
        and not bool(details.get("can_bypass_risk_controls"))
        and not bool(details.get("can_force_open"))
        and not bool(details.get("can_apply_live_sizing"))
    ):
        return "observing", "Observation / strong opportunity shadow audit"
    if (
        key == "position_capacity_release"
        and status == "warning"
        and bool(details.get("audit_only"))
        and not bool(details.get("live_exit_mutation"))
        and not bool(details.get("live_entry_mutation"))
        and not bool(details.get("live_sizing_mutation"))
        and not bool(details.get("can_force_close"))
        and not bool(details.get("can_close_winners"))
        and not bool(details.get("can_bypass_risk_controls"))
    ):
        return "observing", "Observation / capacity release audit"
    if (
        key == "profit_first_governance"
        and status == "warning"
        and bool(details.get("audit_only"))
        and bool(details.get("read_only"))
        and not bool(details.get("live_mutation"))
        and not bool(details.get("live_entry_mutation"))
        and not bool(details.get("live_exit_mutation"))
        and not bool(details.get("live_weight_mutation"))
        and not bool(details.get("live_sizing_mutation"))
        and not bool(details.get("can_submit_orders"))
        and not bool(details.get("can_start_trading_service"))
        and not bool(details.get("can_change_model_routing"))
        and not bool(details.get("can_increase_live_size"))
    ):
        return "observing", "Observation / Profit-First governance audit"
    if (
        key == "phase3_paper_resume_observation"
        and status == "warning"
        and bool(details.get("audit_only"))
        and bool(details.get("read_only"))
        and bool(details.get("observing"))
        and not bool(details.get("starts_trading_service"))
        and not bool(details.get("submits_orders"))
        and not bool(details.get("changes_model_routing"))
        and not bool(details.get("live_mutation"))
    ):
        return "observing", "Observation / paper resume warming window"
    if (
        key == "strategy_signal_root_cause"
        and status == "warning"
        and bool(details.get("audit_only"))
        and bool(details.get("read_only"))
        and not bool(details.get("live_entry_mutation"))
        and not bool(details.get("live_sizing_mutation"))
        and not bool(details.get("live_leverage_mutation"))
        and not bool(details.get("can_force_open"))
        and not bool(details.get("can_override_thresholds"))
        and not bool(details.get("can_change_ml_readiness"))
        and not bool(details.get("can_bypass_risk_controls"))
    ):
        return "observing", "Observation / strategy signal root-cause audit"
    if (
        key == "strategy_quality"
        and status == "warning"
        and cards_by_key
        and _strategy_closed_loop_is_historical_only(cards_by_key)
    ):
        return "observing", "历史遗留 / 当前策略闭环未复现"
    return "unresolved", "未修复 / 当前仍需处理"


def _issue_ledger_from_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {"fixed": [], "unresolved": [], "observing": []}
    cards_by_key = {str(card.get("key") or ""): card for card in cards}
    for card in cards:
        state, label = _issue_ledger_state(card, cards_by_key=cards_by_key)
        buckets[state].append(
            {
                "key": card.get("key"),
                "title": card.get("title"),
                "status": card.get("status"),
                "state": state,
                "state_label": label,
                "owner_path": card.get("owner_path")
                or _owner_path_for_card(str(card.get("key") or "")),
                "summary": card.get("summary"),
                "evidence": card.get("evidence") or [],
                "next_actions": card.get("next_actions") or [],
            }
        )
    priority = {"critical": 0, "warning": 1, "ok": 2, "info": 3}
    for rows in buckets.values():
        rows.sort(key=lambda item: priority.get(str(item.get("status")), 9))
    return {
        "summary": {
            "fixed": len(buckets["fixed"]),
            "unresolved": len(buckets["unresolved"]),
            "observing": len(buckets["observing"]),
            "total": len(cards),
        },
        **buckets,
    }


def _worst_status(*statuses: Any) -> str:
    normalized = [str(status or "info") for status in statuses]
    return min(normalized or ["info"], key=lambda item: STATUS_RANK.get(item, 9))


def _node_state_from_cards(
    related_cards: list[dict[str, Any]],
    cards_by_key: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    if not related_cards:
        return "fixed", "No linked audit card."
    state_priority = {"unresolved": 0, "observing": 1, "fixed": 2}
    states = [_issue_ledger_state(card, cards_by_key=cards_by_key) for card in related_cards]
    return min(states, key=lambda item: state_priority.get(item[0], 9))


def _node_display_status(status: Any, state: Any) -> str:
    normalized_status = str(status or "info")
    normalized_state = str(state or "")
    if normalized_state == "fixed":
        return "ok"
    if normalized_state == "observing":
        return "warning"
    if normalized_state == "unresolved":
        return normalized_status if normalized_status in {"critical", "warning"} else "warning"
    return normalized_status if normalized_status in STATUS_RANK else "info"


def _card_map(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(card.get("key") or ""): card for card in cards if card.get("key")}


def _node_from_cards(
    key: str,
    title: str,
    layer: str,
    cards_by_key: dict[str, dict[str, Any]],
    card_keys: list[str],
    *,
    impact: str,
    upstream: list[str] | None = None,
    downstream: list[str] | None = None,
    checks: list[str] | None = None,
) -> dict[str, Any]:
    related_cards = [cards_by_key[item] for item in card_keys if item in cards_by_key]
    status = _worst_status(*(card.get("status") for card in related_cards))
    state, state_label = _node_state_from_cards(related_cards, cards_by_key)
    display_status = _node_display_status(status, state)
    summaries = [str(card.get("summary") or "") for card in related_cards if card.get("summary")]
    evidence: list[dict[str, Any]] = []
    next_actions: list[str] = []
    for card in related_cards:
        evidence.extend(card.get("evidence") or [])
        next_actions.extend(card.get("next_actions") or [])
    return {
        "key": key,
        "title": title,
        "layer": layer,
        "status": status,
        "display_status": display_status,
        "state": state,
        "state_label": state_label,
        "summary": "；".join(summaries[:2]) or "节点暂无异常。",
        "owner_path": _owner_path_for_node(key, related_cards),
        "impact": impact,
        "upstream": upstream or [],
        "downstream": downstream or [],
        "checks": checks or [],
        "card_keys": [card.get("key") for card in related_cards],
        "evidence": evidence[:6],
        "next_actions": list(dict.fromkeys(next_actions))[:6],
    }


def _build_audit_nodes(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards_by_key = _card_map(cards)
    return [
        _node_from_cards(
            "server_migration",
            "Phase 3 server resource-release/migration",
            "Infra layer",
            cards_by_key,
            ["phase3_server_migration"],
            impact=(
                "Blocks Phase 3 model-server go-live if legacy services/processes still consume "
                "resources, /data/BB isolation is missing, or old-server migration is not whitelist-only."
            ),
            downstream=["model_training", "model_expert_health", "strategy_decision"],
            checks=[
                "resource-release marker",
                "/data/BB isolation",
                "legacy service/process stopped",
                "whitelist migration manifest",
                "old data preserved as isolated history",
            ],
        ),
        _node_from_cards(
            "model_server_readiness",
            "Phase 3 quant model-server readiness",
            "Model infra layer",
            cards_by_key,
            ["phase3_model_server_readiness"],
            impact=(
                "Blocks Phase 3 model-server shadow/canary routing if model artifacts, CUDA/GPU "
                "validation, service contracts, or model endpoints are not ready."
            ),
            upstream=["server_migration"],
            downstream=["model_training", "model_expert_health", "model_dynamic_routing"],
            checks=[
                "download manifest",
                "validation manifest",
                "8 GPU CUDA validation",
                "required quant model slots",
                "model service endpoints",
            ],
        ),
        _node_from_cards(
            "phase3_stage_handoff",
            "Phase 3 controlled stage handoff",
            "Release gate layer",
            cards_by_key,
            ["phase3_stage_handoff", "phase3_go_no_go"],
            impact=(
                "Shows the only allowed next Phase 3 action: stay shadow-only, start paper "
                "with explicit approval, observe post-resume, or review canary. It never starts "
                "paper, promotes canary, or enables live routing by itself."
            ),
            upstream=["server_migration", "model_server_readiness", "okx_execution"],
            downstream=["runtime_loop", "strategy_closed_loop", "model_routing"],
            checks=[
                "Go/No-Go freshness",
                "paper start approval boundary",
                "post-resume observation",
                "specialist shadow evidence",
                "live disabled",
            ],
        ),
        _node_from_cards(
            "runtime_loop",
            "调度与心跳",
            "运行层",
            cards_by_key,
            ["trade_loop"],
            impact="决定系统是否持续分析、是否卡在某个阶段。",
            downstream=["market_data", "position_sync", "strategy_decision"],
            checks=["最近10分钟分析", "最近2小时订单", "当前持仓数量"],
        ),
        _node_from_cards(
            "market_data",
            "行情与K线",
            "数据层",
            cards_by_key,
            ["market_data"],
            impact="影响开仓候选、预期收益、止盈止损和训练特征。",
            upstream=["runtime_loop"],
            downstream=["strategy_decision", "risk_guard", "training_data"],
            checks=["Ticker新鲜度", "1m/5m/15m/1h K线覆盖", "币种覆盖"],
        ),
        _node_from_cards(
            "crypto_feature_coverage",
            "数字货币特征覆盖",
            "数据层",
            cards_by_key,
            ["crypto_feature_coverage"],
            impact="检查 K线、ticker、盘口、滑点、资金费率、未平仓量、新闻社媒和事件日历是否真实可用。",
            upstream=["runtime_loop", "market_data"],
            downstream=["model_training", "strategy_decision", "risk_guard"],
            checks=["缺失特征", "过期特征", "低可信事件", "默认值中性阻断"],
        ),
        _node_from_cards(
            "model_training",
            "模型与训练数据",
            "模型层",
            cards_by_key,
            ["model_training"],
            impact="影响盈利预测、时序预测、情绪预测、本地ML过滤和样本学习。",
            upstream=["market_data"],
            downstream=["strategy_decision", "training_data"],
            checks=["本地量化工具", "影子样本", "交易样本", "外部采集源"],
        ),
        _node_from_cards(
            "model_expert_health",
            "模型/专家体检",
            "模型层",
            cards_by_key,
            ["model_expert_health"],
            impact="评估模型/专家参与、采纳、收益、耗时、JSON 错误和未返回率，只输出体检建议。",
            upstream=["model_training"],
            downstream=["strategy_decision", "model_routing"],
            checks=["24/72小时贡献", "采纳后收益", "JSON错误率", "未返回率"],
        ),
        _node_from_cards(
            "model_expert_competition",
            "模型/专家竞赛",
            "模型层",
            cards_by_key,
            ["model_expert_competition"],
            impact="对模型/专家与 baseline 的离线/影子/模拟竞赛结果做证据化比较，不直接改真实权重。",
            upstream=["model_expert_health", "model_training"],
            downstream=["model_routing", "strategy_decision"],
            checks=["baseline 对比", "影子竞赛", "模拟A/B", "权重建议来源"],
        ),
        _node_from_cards(
            "model_dynamic_routing",
            "模型动态路由",
            "模型层",
            cards_by_key,
            ["model_dynamic_routing"],
            impact="根据候选质量、readiness、贡献、延迟、风险和事件状态生成影子路由计划；初期不替换主链路。",
            upstream=["model_expert_health", "model_expert_competition", "crypto_feature_coverage"],
            downstream=["strategy_decision", "risk_guard"],
            checks=["影子路由计划", "理论少调用", "阻塞原因", "弱证据/快亏观察"],
        ),
        _node_from_cards(
            "profit_first_ranking",
            "Profit-First ranking",
            "Learning layer",
            cards_by_key,
            ["profit_first_ranking"],
            impact="Ranks model, strategy, symbol, side, and lane combinations by realized net PnL before promotion or size increases.",
            upstream=["model_training", "trade_execution_contract"],
            downstream=["strategy_decision", "model_dynamic_routing", "risk_guard"],
            checks=[
                "realized net PnL",
                "profit factor",
                "consecutive losses",
                "tail loss",
                "fee drag",
                "shadow/canary/live stage",
            ],
        ),
        _node_from_cards(
            "profit_first_governance",
            "Profit-First no-entry / losing-exit governance",
            "Learning layer",
            cards_by_key,
            ["profit_first_governance"],
            impact=(
                "Classifies 24h no-entry causes and losing-exit attributions, then turns them "
                "into read-only next-cycle threshold, sizing, and exit-policy recommendations."
            ),
            upstream=["profit_first_ranking", "trade_execution_contract"],
            downstream=["strategy_decision", "model_dynamic_routing", "risk_guard"],
            checks=[
                "24h no-entry diagnosis",
                "losing-exit attribution counts",
                "brain output coverage",
                "next-cycle actions",
                "read-only live mutation guard",
            ],
        ),
        _node_from_cards(
            "profit_first_recovery_blockers",
            "Profit-First recovery blockers",
            "Release gate layer",
            cards_by_key,
            ["profit_first_recovery_blockers"],
            impact=(
                "Lists the exact trade-contract, ranking, and OKX reconciliation items that must "
                "be repaired, disabled, or quarantined before paper/canary recovery."
            ),
            upstream=[
                "profit_first_ranking",
                "profit_first_governance",
                "phase3_stage_handoff",
            ],
            downstream=["runtime_loop", "strategy_decision", "risk_guard"],
            checks=[
                "missing ProfitFirstTradePlan history",
                "missing exit-plan references",
                "ranking disable blockers",
                "OKX/local reconciliation differences",
            ],
        ),
        _node_from_cards(
            "high_risk_review_audit",
            "High-risk review",
            "Risk layer",
            cards_by_key,
            ["high_risk_review_audit"],
            impact="Audits independent high-risk review triggers, approvals, blocks, and unsafe executions without changing live gates.",
            upstream=["model_dynamic_routing", "strategy_decision"],
            downstream=["risk_guard", "okx_execution"],
            checks=["hard-review triggers", "approval status", "blocked count", "unsafe executed"],
        ),
        _node_from_cards(
            "shadow_missed_opportunity",
            "Shadow missed opportunity",
            "Learning layer",
            cards_by_key,
            ["shadow_missed_opportunity"],
            impact="Audits whether missed opportunities are usable only after repeated same-symbol same-side evidence.",
            upstream=["strategy_closed_loop", "model_training"],
            downstream=["strategy_decision", "training_data"],
            checks=[
                "same-symbol same-side repeats",
                "stable positive returns",
                "low risk evidence",
                "weak evidence execution",
            ],
        ),
        _node_from_cards(
            "strong_opportunity",
            "Strong opportunity",
            "Strategy layer",
            cards_by_key,
            ["strong_opportunity"],
            impact="Audits Phase 2 strong opportunity shape without changing live entries, sizing, leverage, or risk gates.",
            upstream=[
                "market_data",
                "model_training",
                "shadow_missed_opportunity",
                "okx_execution",
            ],
            downstream=["strategy_decision", "risk_guard", "training_data"],
            checks=[
                "selected-side expected net",
                "profit quality",
                "loss probability",
                "tail risk",
                "aligned sources",
                "read-only flags",
            ],
        ),
        _node_from_cards(
            "position_capacity_release",
            "Position capacity release",
            "Risk layer",
            cards_by_key,
            ["position_capacity_release"],
            impact="Audits capacity pressure, release candidates, old profit rotation candidates, and unclosed release decisions before entry thresholds are changed.",
            upstream=["position_sync", "strong_opportunity", "strategy_closed_loop"],
            downstream=["strategy_decision", "risk_guard", "okx_execution"],
            checks=[
                "current capacity",
                "release candidates",
                "old profit candidates",
                "unclosed release decisions",
                "crowded-side blocks",
                "read-only flags",
            ],
        ),
        _node_from_cards(
            "strategy_decision",
            "策略决策质量",
            "策略层",
            cards_by_key,
            [
                "strategy_quality",
                "strategy_signal_root_cause",
                "strong_opportunity",
                "position_capacity_release",
                "profit_first_ranking",
                "profit_first_governance",
                "high_risk_review_audit",
                "trade_execution_contract",
            ],
            impact="影响是否开仓、仓位大小、重复亏损复开和快进快出。",
            upstream=[
                "market_data",
                "model_training",
                "profit_first_ranking",
                "profit_first_governance",
                "position_sync",
                "strong_opportunity",
                "position_capacity_release",
            ],
            downstream=["risk_guard", "okx_execution"],
            checks=["负净收益候选", "零净收益候选", "快亏平样本", "拦截原因"],
        ),
        _node_from_cards(
            "strategy_closed_loop",
            "策略闭环有效性",
            "策略层",
            cards_by_key,
            ["strategy_closed_loop", "strategy_signal_root_cause"],
            impact="把数据、模型、决策、仓位、执行、平仓、训练反馈串起来，判断问题卡在哪一层。",
            upstream=["market_data", "model_training", "position_sync"],
            downstream=["risk_guard", "okx_execution", "training_data"],
            checks=["证据档位分布", "弱证据执行", "ML可用率", "快亏平", "收益样本"],
        ),
        _node_from_cards(
            "strategy_signal_root_cause",
            "策略信号根因",
            "策略层",
            cards_by_key,
            ["strategy_signal_root_cause"],
            impact="只读解释不开仓、小单、候选集中、ML/server_profit/shadow 证据未闭环的具体卡点。",
            upstream=["model_training", "shadow_missed_opportunity", "market_data"],
            downstream=["strategy_decision", "strategy_closed_loop"],
            checks=[
                "ML readiness",
                "server_profit contribution",
                "shadow missed conversion",
                "expected-net components",
                "candidate concentration",
            ],
        ),
        _node_from_cards(
            "strategy_gate_contract",
            "策略门槛契约",
            "策略层",
            cards_by_key,
            ["strategy_gate_contract"],
            impact="防止旧固定阈值、死分支、伪硬门槛重新卡住开仓。",
            upstream=["model_training", "strategy_decision", "strategy_closed_loop"],
            downstream=["risk_guard", "okx_execution"],
            checks=["RuntimeEntryFilters", "settings.min_entry_*残留", "if False死分支"],
        ),
        _node_from_cards(
            "risk_guard",
            "风控与守门",
            "风控层",
            cards_by_key,
            [
                "strategy_quality",
                "strategy_closed_loop",
                "trade_execution_contract",
                "position_capacity_release",
                "high_risk_review_audit",
                "okx_trade_fact_integrity",
                "position_price_integrity",
            ],
            impact="影响动态证据、低质量释放、快速平仓和下单前校验。",
            upstream=["strategy_decision", "position_sync"],
            downstream=["okx_execution", "position_sync"],
            checks=["持仓价一致性", "快亏平", "风险证据", "执行原因"],
        ),
        _node_from_cards(
            "okx_execution",
            "OKX执行与历史对账",
            "执行层",
            cards_by_key,
            ["okx_trade_fact_integrity", "okx_reconciliation", "position_price_integrity"],
            impact="影响下单、平仓、历史仓位、账户余额和盈亏记录。",
            upstream=["risk_guard"],
            downstream=["position_sync", "training_data"],
            checks=["缺失历史仓", "OKX持仓快照", "价格/PnL对齐"],
        ),
        _node_from_cards(
            "position_sync",
            "持仓同步与PnL",
            "同步层",
            cards_by_key,
            ["okx_trade_fact_integrity", "position_price_integrity", "okx_reconciliation"],
            impact="影响主面板余额、持仓分析、平仓判断和训练标签。",
            upstream=["okx_execution"],
            downstream=["strategy_decision", "training_data", "dashboard_observability"],
            checks=["平台价 vs OKX标记价", "平台浮盈 vs OKX upl", "合约面值ctVal"],
        ),
        _node_from_cards(
            "training_data",
            "训练标签与样本治理",
            "学习层",
            cards_by_key,
            [
                "model_training",
                "strategy_quality",
                "strategy_closed_loop",
                "okx_trade_fact_integrity",
                "position_price_integrity",
            ],
            impact="影响模型是否越学越聪明，避免错误价格/错误盈亏污染训练。",
            upstream=["market_data", "okx_execution", "position_sync"],
            downstream=["model_training", "strategy_decision"],
            checks=["样本数量", "数据源状态", "脏样本风险", "收益标签可信度"],
        ),
        _node_from_cards(
            "dashboard_observability",
            "页面与可观测性",
            "展示层",
            cards_by_key,
            ["trade_loop", "position_price_integrity", "model_training", "runtime_text_integrity"],
            impact="影响你能否从页面直接定位问题，而不是只看到泛化提示。",
            upstream=["position_sync", "model_training"],
            checks=["节点状态", "根因列表", "执行证据", "历史记录"],
        ),
        _node_from_cards(
            "visible_text_encoding",
            "中文显示与乱码",
            "展示层",
            cards_by_key,
            ["visible_text_encoding", "runtime_text_integrity"],
            impact="防止源码、页面或修复脚本重新出现裸乱码，影响排查和用户判断。",
            upstream=["dashboard_observability"],
            checks=["源码扫描", "前端静态资源", "脚本样本转义"],
        ),
        _node_from_cards(
            "runtime_text_integrity",
            "运行时文本完整性",
            "数据层",
            cards_by_key,
            ["runtime_text_integrity"],
            impact="防止模型输出、专家记忆、执行原因和策略学习事件把真乱码继续写入数据库。",
            upstream=["model_training", "strategy_decision"],
            downstream=["dashboard_observability", "training_data"],
            checks=["最近新增乱码记录", "来源表和字段", "样例", "可修复性"],
        ),
    ]


def _node_summary(nodes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "nodes": len(nodes),
        "critical": sum(1 for node in nodes if node.get("display_status") == "critical"),
        "warning": sum(1 for node in nodes if node.get("display_status") == "warning"),
        "ok": sum(1 for node in nodes if node.get("display_status") == "ok"),
    }


def _history_path() -> Path:
    return settings.data_dir / SYSTEM_AUDIT_HISTORY_FILE


def _history_record(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    root_causes = payload.get("root_causes") if isinstance(payload.get("root_causes"), list) else []
    return {
        "checked_at": payload.get("checked_at"),
        "source": source,
        "status": payload.get("status"),
        "status_label": payload.get("status_label"),
        "summary": payload.get("summary") or {},
        "node_summary": payload.get("node_summary") or {},
        "root_causes": root_causes[:8],
    }


def _read_history_records(limit: int = 50) -> list[dict[str, Any]]:
    path = _history_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records[-max(1, int(limit)) :][::-1]


def _append_history_record(payload: dict[str, Any], *, source: str) -> None:
    if not settings.system_audit_history_enabled:
        return
    path = _history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    max_records = max(50, min(int(settings.system_audit_history_max_records or 500), 5000))
    existing = list(reversed(_read_history_records(limit=max_records - 1)))
    existing.append(_history_record(payload, source=source))
    text = "\n".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        for item in existing[-max_records:]
    )
    path.write_text(text + "\n", encoding="utf-8")


async def collect_system_audit_status(
    *, record_history: bool = True, source: str = "api"
) -> dict[str, Any]:
    async with _system_audit_lock():
        return await _collect_system_audit_status_unlocked(
            record_history=record_history,
            source=source,
        )


async def _collect_system_audit_status_unlocked(
    *, record_history: bool = True, source: str = "api"
) -> dict[str, Any]:
    audit_specs = [
        ("trade_loop", _trade_loop_audit),
        ("okx_reconciliation", _okx_reconciliation_audit),
        ("okx_trade_fact_integrity", _okx_trade_fact_integrity_audit),
        ("phase3_server_migration", _phase3_server_migration_audit),
        ("phase3_model_server_readiness", _phase3_model_server_readiness_audit),
        ("phase3_paper_resume_preflight", _phase3_paper_resume_preflight_audit),
        ("phase3_paper_resume_observation", _phase3_paper_resume_observation_audit),
        ("phase3_stage_handoff", _phase3_stage_handoff_audit),
        ("position_price_integrity", _position_price_integrity_audit),
        ("market_data", _market_data_audit),
        ("strategy_quality", _strategy_quality_audit),
        ("strategy_closed_loop", _strategy_closed_loop_audit),
        ("strategy_signal_root_cause", _strategy_signal_root_cause_audit),
        ("model_training", _model_training_audit),
        ("model_expert_health", _model_expert_health_audit),
        ("model_expert_competition", _model_expert_competition_audit),
        ("model_dynamic_routing", _model_dynamic_routing_audit),
        ("high_risk_review_audit", _high_risk_review_audit),
        ("crypto_feature_coverage", _crypto_feature_coverage_audit),
        ("shadow_missed_opportunity", _shadow_missed_opportunity_audit),
        ("strong_opportunity", _strong_opportunity_audit),
        ("position_capacity_release", _position_capacity_release_audit),
        ("trade_execution_contract", _trade_execution_contract_audit),
        ("profit_first_ranking", _profit_first_ranking_audit),
        ("profit_first_governance", _profit_first_governance_audit),
        (
            "strategy_gate_contract",
            lambda: asyncio.to_thread(_strategy_gate_contract_audit),
        ),
        ("visible_text_encoding", lambda: asyncio.to_thread(_source_visible_text_audit)),
        ("runtime_text_integrity", _runtime_text_integrity_audit),
    ]
    priority_specs = [(key, factory) for key, factory in audit_specs if key in PRIORITY_AUDIT_KEYS]
    heavy_specs = [
        (key, factory)
        for key, factory in audit_specs
        if key in HEAVY_AUDIT_KEYS and key not in PRIORITY_AUDIT_KEYS
    ]
    db_specs = [
        (key, factory)
        for key, factory in audit_specs
        if key in DB_AUDIT_KEYS and key not in PRIORITY_AUDIT_KEYS and key not in HEAVY_AUDIT_KEYS
    ]
    regular_specs = [
        (key, factory)
        for key, factory in audit_specs
        if key not in PRIORITY_AUDIT_KEYS
        and key not in HEAVY_AUDIT_KEYS
        and key not in DB_AUDIT_KEYS
    ]
    result_by_key: dict[str, dict[str, Any] | Exception] = {}
    result_by_key.update(await _run_audit_specs(priority_specs, max_concurrency=1))
    result_by_key.update(await _run_audit_specs(db_specs, max_concurrency=1))
    result_by_key.update(
        await _run_audit_specs(
            regular_specs,
            max_concurrency=SYSTEM_AUDIT_MAX_CONCURRENCY,
        )
    )
    result_by_key.update(await _run_audit_specs(heavy_specs, max_concurrency=1))
    cards: list[dict[str, Any]] = []
    for section_key, _factory in audit_specs:
        result = result_by_key[section_key]
        if isinstance(result, Exception):
            cards.append(
                _audit_card(
                    section_key,
                    "巡检模块",
                    "warning",
                    "巡检模块执行失败。",
                    details={
                        "section_key": section_key,
                        "error": safe_error_text(result, limit=180),
                    },
                    owner_path=_owner_path_for_card(section_key),
                )
            )
        else:
            cards.append(result)
    cards.append(_profit_first_recovery_blockers_audit_from_cards(cards))
    cards.append(_phase3_go_no_go_audit_from_cards(cards))
    cards = sorted(cards, key=lambda item: STATUS_RANK.get(str(item.get("status")), 9))
    nodes = _build_audit_nodes(cards)
    findings = _root_cause_findings(cards)
    issue_ledger = _issue_ledger_from_cards(cards)
    status = "ok"
    if any(card.get("status") == "critical" for card in cards):
        status = "critical"
    elif any(card.get("status") == "warning" for card in cards):
        status = "warning"
    payload = sanitize_payload(
        {
            "status": status,
            "status_label": {"ok": "正常", "warning": "需关注", "critical": "异常"}.get(
                status, status
            ),
            "checked_at": _now().isoformat(),
            "windows": AUDIT_WINDOWS,
            "summary": {
                "cards": len(cards),
                "critical": sum(1 for card in cards if card.get("status") == "critical"),
                "warning": sum(1 for card in cards if card.get("status") == "warning"),
                "ok": sum(1 for card in cards if card.get("status") == "ok"),
                "findings": len(findings),
                "nodes": len(nodes),
            },
            "root_causes": findings,
            "issue_ledger": issue_ledger,
            "nodes": nodes,
            "node_summary": _node_summary(nodes),
            "cards": cards,
            "history": {
                "enabled": bool(settings.system_audit_history_enabled),
                "interval_seconds": int(settings.system_audit_history_interval_seconds or 300),
                "max_records": int(settings.system_audit_history_max_records or 500),
            },
            "safety_note": "根因雷达当前只读巡检；补历史仓位、重启服务、批量训练等动作必须人工确认。",
        }
    )
    if record_history:
        _append_history_record(payload, source=source)
    return payload


@router.get("/system-audit/status")
async def system_audit_status() -> dict[str, Any]:
    return await collect_system_audit_status(record_history=True, source="api")


@router.get("/model-expert-health/status")
async def model_expert_health_status(hours: int = 72, limit: int = 1200) -> dict[str, Any]:
    report = await ModelExpertHealthService().report(hours=hours, limit=limit)
    report["audit_only"] = True
    report["live_weight_mutation"] = False
    return sanitize_payload(report)


@router.get("/model-expert-competition/status")
async def model_expert_competition_status(hours: int = 72, limit: int = 1200) -> dict[str, Any]:
    report = await ModelExpertCompetitionService().report(hours=hours, limit=limit)
    report["audit_only"] = True
    report["live_weight_mutation"] = False
    report["can_apply_live_weight"] = False
    competitors = report.get("competitors") if isinstance(report.get("competitors"), dict) else {}
    for row in competitors.values():
        if isinstance(row, dict):
            row["can_apply_live_weight"] = False
    return sanitize_payload(report)


@router.get("/model-dynamic-routing/status")
async def model_dynamic_routing_status(hours: int = 72, limit: int = 1200) -> dict[str, Any]:
    report = await ModelDynamicRoutingService().report(hours=hours, limit=limit)
    return sanitize_payload(_safe_dynamic_routing_report(report))


@router.get("/high-risk-review-audit/status")
async def high_risk_review_audit_status(hours: int = 72, limit: int = 1200) -> dict[str, Any]:
    report = await HighRiskReviewAuditService().report(hours=hours, limit=limit)
    return sanitize_payload(_safe_high_risk_review_report(report))


@router.get("/shadow-missed-opportunity/status")
async def shadow_missed_opportunity_status(
    hours: int = SHADOW_MISSED_OPPORTUNITY_AUDIT_HOURS,
    limit: int = SHADOW_MISSED_OPPORTUNITY_AUDIT_LIMIT,
) -> dict[str, Any]:
    report = await ShadowMissedOpportunityClosedLoopService().report(hours=hours, limit=limit)
    return sanitize_payload(_safe_shadow_missed_opportunity_report(report))


@router.get("/strong-opportunity/status")
async def strong_opportunity_status(
    hours: int = STRONG_OPPORTUNITY_AUDIT_HOURS,
    limit: int = STRONG_OPPORTUNITY_AUDIT_LIMIT,
) -> dict[str, Any]:
    report = await StrongOpportunityService(lookback_hours=hours, limit=limit).report()
    return sanitize_payload(_safe_strong_opportunity_report(report))


@router.get("/position-capacity-release/status")
async def position_capacity_release_status(
    hours: int = POSITION_CAPACITY_RELEASE_AUDIT_HOURS,
    limit: int = POSITION_CAPACITY_RELEASE_AUDIT_LIMIT,
) -> dict[str, Any]:
    report = await PositionCapacityReleaseAuditService(
        lookback_hours=hours,
        limit=limit,
    ).report()
    return sanitize_payload(_safe_position_capacity_release_report(report))


@router.get("/trade-execution-contract/status")
async def trade_execution_contract_status(
    hours: int = TRADE_EXECUTION_CONTRACT_AUDIT_HOURS,
    limit: int = TRADE_EXECUTION_CONTRACT_AUDIT_LIMIT,
) -> dict[str, Any]:
    report = await TradeExecutionContractService().report(hours=hours, limit=limit)
    return sanitize_payload(_safe_trade_execution_contract_report(report))


@router.get("/profit-first-ranking/status")
async def profit_first_ranking_status(
    hours: int = PROFIT_FIRST_RANKING_AUDIT_HOURS,
    limit: int = PROFIT_FIRST_RANKING_AUDIT_LIMIT,
) -> dict[str, Any]:
    report = await ProfitFirstRankingService().report(hours=hours, limit=limit)
    return sanitize_payload(_safe_profit_first_ranking_report(report))


@router.get("/crypto-feature-coverage/status")
async def crypto_feature_coverage_status(hours: int = 24, limit: int = 1000) -> dict[str, Any]:
    report = await CryptoFeatureCoverageService().report(hours=hours, limit=limit)
    return sanitize_payload(_safe_crypto_feature_report(report))


@router.get("/system-audit/history")
async def system_audit_history(limit: int = 50) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or 50), 200))
    records = _read_history_records(limit=safe_limit)
    return sanitize_payload(
        {
            "enabled": bool(settings.system_audit_history_enabled),
            "interval_seconds": int(settings.system_audit_history_interval_seconds or 300),
            "records": records,
            "count": len(records),
        }
    )
