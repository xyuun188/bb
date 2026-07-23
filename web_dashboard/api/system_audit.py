"""Root-cause radar API for online system audits."""

from __future__ import annotations

import ast
import asyncio
import copy
import inspect
import json
import time
from collections import Counter
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import structlog
from fastapi import APIRouter
from sqlalchemy import and_, func, or_, select

from config.settings import settings
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol
from db.session import get_session_ctx
from models.decision import AIDecision
from models.market_data import Kline, Ticker
from models.trade import OkxPositionHistory, Order, Position
from scripts.audit_runtime_text_integrity import collect_runtime_text_integrity_report
from scripts.repair_okx_position_fact_links import (
    collect_scan_report as collect_position_fact_link_scan_report,
)
from services.artifact_retirement_audit import ArtifactRetirementAuditService
from services.crypto_feature_coverage import CryptoFeatureCoverageService
from services.exchange_position_state import (
    exchange_position_display_valuation,
    parse_exchange_position_snapshot,
)
from services.execution_reason_localizer import localize_execution_reason
from services.high_risk_review_audit import HighRiskReviewAuditService
from services.historical_trade_fact_audit import HistoricalTradeFactAuditService
from services.ml_signal_service import MLSignalService
from services.model_dynamic_routing import ModelDynamicRoutingService
from services.model_expert_competition import ModelExpertCompetitionService
from services.model_expert_health import ModelExpertHealthService
from services.model_training_registry import build_model_training_registry
from services.model_training_state import ModelTrainingStateStore
from services.okx_authoritative_sync import OkxAuthoritativeSyncService
from services.okx_trade_fact_integrity import OkxTradeFactIntegrityService
from services.phase3_go_no_go import evaluate_phase3_go_no_go_cards
from services.phase3_model_server_readiness import Phase3ModelServerReadinessAuditService
from services.phase3_paper_resume_observation import Phase3PaperResumeObservationService
from services.phase3_paper_resume_preflight import Phase3PaperResumePreflightService
from services.phase3_rebuild_readiness import Phase3RebuildReadinessService
from services.phase3_server_migration_audit import Phase3ServerMigrationAuditService
from services.phase3_stage_handoff import Phase3StageHandoffService
from services.position_capacity_release_audit import PositionCapacityReleaseAuditService
from services.production_source_health import ProductionSourceHealthService
from services.profit_training_contract import PROFIT_TRAINING_TARGET
from services.server_monitor_status import collect_platform_runtime_status
from services.shadow_missed_opportunity_closed_loop import (
    ShadowMissedOpportunityClosedLoopService,
)
from services.strategy_signal_root_cause_audit import StrategySignalRootCauseAuditService
from services.strong_opportunity import StrongOpportunityService
from services.trade_execution_contract import TradeExecutionContractService
from services.trading_params import DEFAULT_TRADING_PARAMS
from web_dashboard.api import data_collection as data_collection_api
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()
logger = structlog.get_logger(__name__)
_skip_okx_daily_reconciliation_latest: ContextVar[bool] = ContextVar(
    "skip_okx_daily_reconciliation_latest",
    default=False,
)

AUDIT_WINDOWS = {"fast_minutes": 10, "trade_hours": 2, "strategy_hours": 24}
EXPECTED_KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h")
KLINE_STALE_LIMIT_SECONDS = {"1m": 180, "5m": 600, "15m": 1800, "1h": 7200}
STATUS_RANK = {"critical": 0, "warning": 1, "ok": 2, "info": 3}
SYSTEM_AUDIT_HISTORY_FILE = "system_audit_history.jsonl"
SYSTEM_AUDIT_LATEST_FILE = "system_audit_latest.json"
POSITION_PRICE_SPLIT_WARN_PCT = 0.03
POSITION_PNL_SPLIT_WARN_USDT = 0.5
OKX_RECONCILIATION_CACHE_TTL_SECONDS = 120
OKX_AUTHORITATIVE_SYNC_CACHE_TTL_SECONDS = 45
MODEL_RUNTIME_PROBE_TIMEOUT_SECONDS = 8.0
SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS = 20.0
SYSTEM_AUDIT_MAX_CONCURRENCY = 4
MODEL_TRAINING_STATE_STORE = ModelTrainingStateStore(
    settings.data_dir / "model_training_scheduler_state.json"
)
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
SYSTEM_AUDIT_SECTION_TIMEOUT_OVERRIDES = {
    "phase3_server_migration": PHASE3_SERVER_MIGRATION_AUDIT_TIMEOUT_SECONDS + 5,
    "phase3_model_server_readiness": PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS + 5,
    "phase3_paper_resume_preflight": PHASE3_PAPER_RESUME_PREFLIGHT_TIMEOUT_SECONDS + 5,
    "phase3_paper_resume_observation": PHASE3_PAPER_RESUME_OBSERVATION_TIMEOUT_SECONDS + 5,
    "strategy_closed_loop": 60.0,
    "strategy_signal_root_cause": 60.0,
    "model_training": 180.0,
    "position_capacity_release": 60.0,
}
PRIORITY_AUDIT_KEYS = ("okx_reconciliation", "trade_execution_contract")
DB_AUDIT_KEYS = (
    "trade_loop",
    "okx_trade_fact_integrity",
    "position_price_integrity",
    "market_data",
    "strategy_quality",
    "strategy_closed_loop",
    "strategy_signal_root_cause",
    "production_source_health",
    "model_training",
    "model_dynamic_routing",
    "high_risk_review_audit",
    "crypto_feature_coverage",
    "shadow_missed_opportunity",
    "strong_opportunity",
    "position_capacity_release",
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
    "production_source_health": "services/production_source_health.py",
    "strategy_gate_contract": "services/live_ml_profit_contract.py",
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
    "production_source_health": "services/production_source_health.py",
    "strategy_gate_contract": "services/live_ml_profit_contract.py",
    "risk_guard": "services/trading_policies.py",
    "okx_execution": "services/execution_service.py",
    "position_sync": "services/position_sync_service.py",
    "server_migration": "services/phase3_server_migration_audit.py",
    "phase3_go_no_go": "services/phase3_go_no_go.py",
    "phase3_stage_handoff": "services/phase3_stage_handoff.py",
    "model_server_readiness": "services/phase3_model_server_readiness.py",
    "paper_resume_preflight": "services/phase3_paper_resume_preflight.py",
    "training_data": "services/okx_trade_fact_integrity.py",
    "dashboard_observability": "web_dashboard/static/js/dashboard.js",
    "visible_text_encoding": "web_dashboard/api/system_audit.py",
    "runtime_text_integrity": "scripts/audit_runtime_text_integrity.py",
}

_okx_reconciliation_cache: tuple[datetime, dict[str, Any]] | None = None
_system_audit_status_cache: tuple[datetime, dict[str, Any]] | None = None
_system_audit_refresh_task: asyncio.Task[Any] | None = None
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
    "runtime_entry_filters",
    "min_entry_volume_ratio_provider",
    "min_entry_adx_provider",
    "if False and",
)
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
        report_dir = path.parent
        candidates = [path]
        if report_dir.exists():
            candidates.extend(
                candidate
                for candidate in report_dir.glob("okx-daily-reconciliation-*.json")
                if candidate != path
            )
        completed_reports: list[tuple[datetime, Path, dict[str, Any]]] = []
        invalid_candidates: list[str] = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                invalid_candidates.append(str(candidate))
                continue
            if not isinstance(candidate_payload, dict):
                invalid_candidates.append(str(candidate))
                continue
            generated = _parse_utc_datetime(candidate_payload.get("generated_at"))
            completed = bool(candidate_payload.get("completed", True))
            if generated is None or not completed or candidate_payload.get("artifact_error"):
                invalid_candidates.append(str(candidate))
                continue
            completed_reports.append((generated, candidate, candidate_payload))
        if not completed_reports:
            if path.exists():
                return {
                    **base,
                    "status": "invalid",
                    "stale": True,
                    "invalid_candidates": invalid_candidates[:8],
                }
            return {**base, "status": "missing", "stale": True}
        generated_at, selected_path, payload = max(
            completed_reports,
            key=lambda item: item[0],
        )
        age = _age_seconds(generated_at) if generated_at is not None else None
        stale = (
            age is None
            or age > OKX_DAILY_RECONCILIATION_REPORT_MAX_AGE_SECONDS
        )
        gates = _safe_dict(payload.get("operational_gates"))
        ledger = _safe_dict(payload.get("issue_ledger"))
        return {
            **base,
            "available": True,
            "selected_path": str(selected_path),
            "latest_path": str(path),
            "latest_fallback_used": selected_path != path,
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
    safe["read_only"] = True
    safe["live_entry_mutation"] = False
    safe["can_bypass_risk_controls"] = False
    safe["global_missed_count_can_drive_entries"] = False
    for key in ("return_observations", "executed_return_contract_gaps"):
        rows = safe.get(key) if isinstance(safe.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["observation_only"] = True
            row["can_authorize_entry"] = False
            row["can_change_size_or_leverage"] = False
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
    safe["can_bypass_risk_controls"] = False
    for key in (
        "position_economics_incomplete",
        "executed_dynamic_exit_contract_gaps",
        "dynamic_exit_decisions",
    ):
        rows = safe.get(key) if isinstance(safe.get(key), list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["can_force_close"] = False
            row["can_bypass_risk_controls"] = False
    return safe


def _safe_trade_execution_contract_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["read_only"] = True
    safe["live_entry_mutation"] = False
    safe["live_exit_mutation"] = False
    safe["can_bypass_risk_controls"] = False
    policy = _safe_dict(safe.get("policy"))
    policy["optimization_target"] = PROFIT_TRAINING_TARGET
    policy["entry_requires_positive_fee_after_return"] = True
    policy["entry_requires_positive_return_lcb"] = True
    policy["entry_requires_live_execution_cost"] = True
    policy["entry_requires_dynamic_risk_budget"] = True
    policy["entry_requires_complete_provenance"] = True
    policy["exit_requires_position_economics"] = True
    policy["exit_requires_dynamic_close_fraction"] = True
    policy["filled_order_link_required"] = True
    safe["policy"] = policy
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


def _phase3_dynamic_return_gate_status() -> dict[str, Any]:
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
    can_resume_paper = bool(preflight.get("can_resume_paper"))
    ready = (
        go_no_go_fresh
        and preflight_fresh
        and go_status == "go"
        and go_no_go_details.get("ready") is True
        and can_resume_paper
    )
    return {
        "ready": ready,
        "status": "go" if ready else "no_go",
        "go_no_go_status": go_status or "missing",
        "go_no_go_fresh": go_no_go_fresh,
        "preflight_fresh": preflight_fresh,
        "can_resume_paper": can_resume_paper,
        "production_permission": False,
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
        "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        "completed_count": 0,
        "eligible_shadow_count": 0,
        "model_count": 0,
        "models": [],
        "summary": {"promotion_ready_count": 0, "blocked_count": 0},
        "reason": "specialist_shadow_evaluation_report_missing",
        "candidate_paths": [str(path) for path in candidates],
    }


def _system_audit_section_timeout_seconds(key: str) -> float:
    value = SYSTEM_AUDIT_SECTION_TIMEOUT_OVERRIDES.get(
        str(key or ""),
        SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS,
    )
    return max(float(value or SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS), 0.001)


async def _audit_maybe_async(
    factory: Any,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    result = factory()
    if inspect.isawaitable(result):
        effective_timeout = (
            SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        result = await asyncio.wait_for(
            result,
            timeout=max(float(effective_timeout or 20.0), 0.001),
        )
    return result


async def _run_audit_specs(
    specs: list[tuple[str, Any]],
    *,
    max_concurrency: int = SYSTEM_AUDIT_MAX_CONCURRENCY,
    timings: dict[str, float] | None = None,
) -> dict[str, dict[str, Any] | Exception]:
    if not specs:
        return {}
    concurrency = max(1, int(max_concurrency or 1))
    if concurrency == 1:
        results: dict[str, dict[str, Any] | Exception] = {}
        for key, factory in specs:
            started = time.perf_counter()
            try:
                results[key] = await _audit_maybe_async(
                    factory,
                    timeout_seconds=_system_audit_section_timeout_seconds(key),
                )
            except Exception as exc:
                results[key] = exc
            finally:
                if timings is not None:
                    timings[key] = round(time.perf_counter() - started, 4)
        return results

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(key: str, factory: Any) -> tuple[str, dict[str, Any] | Exception]:
        async with semaphore:
            started = time.perf_counter()
            try:
                return key, await _audit_maybe_async(
                    factory,
                    timeout_seconds=_system_audit_section_timeout_seconds(key),
                )
            except Exception as exc:
                return key, exc
            finally:
                if timings is not None:
                    timings[key] = round(time.perf_counter() - started, 4)

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
    dynamic_return_gate = (
        _phase3_dynamic_return_gate_status()
        if not runtime_running
        else {"ready": False}
    )
    stalled = (
        not bool(dynamic_return_gate.get("ready"))
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
            bool(dynamic_return_gate.get("ready"))
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
    if dynamic_return_gate.get("ready"):
        summary = (
            "The dynamic return gate is ready, but the trading service is stopped; "
            "this card remains an observation and grants no production permission."
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
            "dynamic_return_gate_ready": bool(dynamic_return_gate.get("ready")),
            "dynamic_return_gate": dynamic_return_gate,
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


async def _okx_reconciliation_audit(
    *,
    max_close_orders: int | None = OKX_RECONCILIATION_AUDIT_MAX_CLOSE_ORDERS,
) -> dict[str, Any]:
    use_dashboard_cache = max_close_orders == OKX_RECONCILIATION_AUDIT_MAX_CLOSE_ORDERS
    if use_dashboard_cache:
        cached = _cached_okx_reconciliation_card()
        if cached is not None:
            return cached

    def finalize(card: dict[str, Any]) -> dict[str, Any]:
        return _store_okx_reconciliation_card(card) if use_dashboard_cache else card

    try:
        report = await asyncio.wait_for(
            _okx_reconciliation_light_scan(
                days=14,
                max_close_orders=max_close_orders,
            ),
            timeout=5.0,
        )
    except Exception as exc:
        timeout = isinstance(exc, TimeoutError)
        return finalize(
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
    return finalize(
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
        official_history_link_index = await _load_official_history_close_link_index(
            session,
            execution_modes={str(order.execution_mode or "") for order, _action in rows},
            exchange_order_ids={
                str(order.exchange_order_id or "").strip()
                for order, _action in rows
                if str(order.exchange_order_id or "").strip()
            },
        )
        official_history_covered_count = 0
        for order, action in rows:
            exchange_order_id = str(order.exchange_order_id or "").strip()
            if not exchange_order_id:
                continue
            if (str(order.execution_mode or ""), exchange_order_id) in close_link_index:
                linked_count += 1
                continue
            if (
                str(order.execution_mode or "").strip().lower(),
                exchange_order_id,
            ) in official_history_link_index:
                official_history_covered_count += 1
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
        "official_history_covered": official_history_covered_count,
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
        official_history_covered_count=official_history_covered_count,
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


async def _load_official_history_close_link_index(
    session: Any,
    *,
    execution_modes: set[str],
    exchange_order_ids: set[str],
) -> set[tuple[str, str]]:
    modes = {
        str(mode or "").strip().lower()
        for mode in execution_modes
        if str(mode or "").strip()
    }
    if not modes or not exchange_order_ids:
        return set()
    rows = list(
        (
            await session.execute(
                select(OkxPositionHistory).where(OkxPositionHistory.mode.in_(sorted(modes)))
            )
        ).scalars().all()
    )
    return {
        (str(row.mode or "").strip().lower(), str(order_id or "").strip())
        for row in rows
        for order_id in (row.close_order_ids or [])
        if str(order_id or "").strip() in exchange_order_ids
    }


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
    concerning_states: set[str] = set()
    concerning_count = sum(int(counts.get(state) or 0) for state in concerning_states)
    critical_count = 0
    priority: dict[str, int] = {}
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
    summary_text = "模型/专家稳定性与收益指标仅供观察，不生成生产降权或禁用建议。"
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
            "expert_output_diversity": summary.get("expert_output_diversity") or {},
            "top_components": top_components,
            "windows_hours": report.get("windows_hours") or [],
        },
        evidence=[
            {"label": "组件数", "value": int(summary.get("components") or len(components))},
            {"label": "仅观察", "value": int(counts.get("observation_only") or 0)},
            {"label": "生产权重变更", "value": 0},
            {"label": "生产禁用", "value": critical_count},
        ],
        next_actions=[
            "本卡只读展示参与、费后收益、JSON 错误和未返回率。",
            "任何生产路由或权重变化必须经过独立模型治理流程。",
            "胜负率和稳定性指标不能直接授权交易、降权或禁用。",
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
    concerning = 0
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
        "竞赛缺少基线样本，仅记录观察。"
        if blockers
        else "模型/专家费后收益对比仅供观察，不生成生产权重建议。"
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
            {"label": "基线样本", "value": int(baseline.get("sample_count") or 0)},
            {"label": "竞赛组件", "value": len(competitors)},
            {
                "label": "仅观察",
                "value": int(action_counts.get("observation_only") or 0),
            },
            {"label": "生产权重变更", "value": concerning},
        ],
        next_actions=[
            "本卡只读展示基线和费后收益差异。",
            "模型晋升必须使用权威费后收益分布与完整治理证据。",
            "本报告不能改变生产专家集合、权重或路由。",
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
    ineligible_executed = int(
        observations.get("ineligible_return_contract_executed_count") or 0
    )
    route_count = int(summary.get("route_plan_count") or 0)
    promotion_gate = _safe_dict(report.get("promotion_gate"))
    warning = bool(unsafe_attempts or ineligible_executed or blockers or route_count == 0)
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
            {"label": "收益合同不合格却执行", "value": ineligible_executed},
        ],
        next_actions=[
            "没有 C2/C3 基线和线上观察前，不得把动态路由应用到真实专家调用。",
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
            "高风险独立复核",
            "warning",
            "高风险独立复核失败；硬复核门继续保持保守阻断。",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
        )
    unsafe = int(report.get("executed_without_required_review_count") or 0)
    hard_required = int(report.get("hard_review_required_count") or 0)
    blocked = int(report.get("blocked_count") or 0)
    status = "critical" if unsafe else "warning" if hard_required and not blocked else "ok"
    summary = (
        "存在未完成审批就执行的高风险开仓；扩大生产影响前必须检查复核门接线。"
        if unsafe
        else (
            "高风险复核门正在运行，并按要求阻断或批准复核。"
            if hard_required
            else "当前窗口没有需要硬复核的开仓。"
        )
    )
    return _audit_card(
        "high_risk_review_audit",
        "高风险独立复核",
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
            {"label": "需要硬复核", "value": hard_required},
            {"label": "已阻断", "value": blocked},
            {"label": "不安全执行", "value": unsafe},
        ],
        next_actions=[
            "需要硬复核的开仓必须在 approved=true 后才能执行。",
            "普通低风险开仓不应进入高风险复核器。",
            "高风险复核失败时不得绕过风险控制。",
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
            "影子错失机会复盘",
            "warning",
            "影子错失机会报告读取失败；错失机会反馈继续只作观察。",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
            next_actions=[
                "检查 shadow_backtests 完成记录和错失机会报告输入。"
            ],
        )
    summary = _safe_dict(report.get("summary"))
    blocked_counts = _safe_dict(report.get("blocked_reason_counts"))
    contract_gaps = int(summary.get("executed_return_contract_gap_count") or 0)
    warning = contract_gaps > 0
    return _audit_card(
        "shadow_missed_opportunity",
        "影子错失机会复盘",
        "warning" if warning else "ok",
        (
            "已执行开仓缺少正向费后收益契约。"
            if warning
            else "错失机会继续作为只读费后收益观察样本。"
        ),
        details={
            "audit_only": True,
            "read_only": True,
            "live_entry_mutation": False,
            "can_bypass_risk_controls": False,
            "global_missed_count_can_drive_entries": False,
            "summary": summary,
            "blocked_reason_counts": blocked_counts,
            "return_observations": _safe_list(report.get("return_observations"))[:10],
            "executed_return_contract_gaps": _safe_list(
                report.get("executed_return_contract_gaps")
            )[:10],
        },
        evidence=[
            {"label": "已完成", "value": int(summary.get("completed_count") or 0)},
            {"label": "已错失", "value": int(summary.get("missed_count") or 0)},
            {"label": "观察分组", "value": int(summary.get("observe_only_count") or 0)},
            {"label": "执行契约缺口", "value": contract_gaps},
        ],
        next_actions=[
            "错失机会收益只能用于观察。",
            "每次开仓都必须具备当前为正的收益置信下界、实时成本和来源证据。",
            "模型晋升前必须调查每一个已执行收益契约缺口。",
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
            "强机会识别",
            "warning",
            "强机会报告读取失败；二阶段晋升继续保持影子模式。",
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
                "检查最近开仓决策的原始 opportunity_score 和 entry_candidate_evidence。",
                "报告不可用时不得晋升仓位大小或开仓行为。",
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
        "强机会识别",
        "warning" if warning else "ok",
        (
            "强机会分类器当前只作影子观察；禁止晋升到生产仓位调整。"
            if warning
            else "强机会分类器在影子模式中找到了可审计候选。"
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
            "side_counts": _safe_dict(report.get("side_counts")),
            "thresholds": _safe_dict(report.get("thresholds")),
            "strong_candidates": _safe_list(report.get("strong_candidates"))[:10],
            "near_misses": _safe_list(report.get("near_misses"))[:10],
            "diagnostic_boundary": report.get("diagnostic_boundary"),
        },
        evidence=[
            {"label": "开仓决策", "value": entry_decisions},
            {"label": "强机会", "value": strong_count},
            {"label": "已执行强机会", "value": executed_strong},
            {"label": "接近达标", "value": near_miss_count},
        ],
        next_actions=[
            "本卡片只能作为二阶段影子证据，不能强制开仓。",
            "生产晋升前必须验证 OKX 事实完整性、选中方向收益分布、实时成本、来源证据和左尾质量。",
            "不得仅凭本报告提高杠杆或仓位，也不得绕过风险门。",
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
            "Position economics and dynamic exit",
            "warning",
            "Position economics audit is unavailable; production remains fail-closed.",
            details={
                "error": safe_error_text(exc, limit=180),
                "read_only": True,
                "audit_only": True,
                "live_exit_mutation": False,
                "live_entry_mutation": False,
                "live_sizing_mutation": False,
                "can_force_close": False,
                "can_bypass_risk_controls": False,
            },
            owner_path="services/position_capacity_release_audit.py",
        )

    economics_gaps = int(report.get("position_economics_incomplete_count") or 0)
    exit_gaps = int(report.get("executed_dynamic_exit_contract_gap_count") or 0)
    violation_count = economics_gaps + exit_gaps
    return _audit_card(
        "position_capacity_release",
        "持仓经济性与动态退出",
        "critical" if violation_count else "ok",
        (
            "持仓经济性或已执行动态退出契约不完整。"
            if violation_count
            else "当前持仓和已执行退出满足现行经济性契约。"
        ),
        details=report,
        evidence=[
            {"label": "当前持仓", "value": int(report.get("open_position_count") or 0)},
            {"label": "经济性完整", "value": int(report.get("position_economics_complete_count") or 0)},
            {"label": "经济性缺口", "value": economics_gaps},
            {"label": "动态退出", "value": int(report.get("dynamic_exit_decision_count") or 0)},
            {"label": "已执行退出契约缺口", "value": exit_gaps},
        ],
        next_actions=(
            ["阻断新增风险，并调查每个经济性或已执行退出契约缺口。"]
            if violation_count
            else ["继续审计费后持仓经济性和动态退出来源证据。"]
        ),
        owner_path="services/position_capacity_release_audit.py",
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


async def _strategy_quality_audit() -> dict[str, Any]:
    try:
        report = _safe_strategy_signal_root_cause_report(
            await StrategySignalRootCauseAuditService().report()
        )
    except Exception as exc:
        return _audit_card(
            "strategy_quality",
            "动态费后收益策略质量",
            "warning",
            "策略收益质量审计不可用；生产继续保持保守关闭。",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
            owner_path="services/strategy_signal_root_cause_audit.py",
        )
    blocked = int(report.get("live_ml_blocked_count") or 0)
    return _audit_card(
        "strategy_quality",
        "动态费后收益策略质量",
        "warning" if blocked else "ok",
        str(report.get("summary") or "动态费后收益策略审计已完成。"),
        details=report,
        evidence=[
            {"label": "开仓决策", "value": int(report.get("entry_decision_count") or 0)},
            {
                "label": "收益契约完整",
                "value": int(report.get("live_ml_ready_count") or 0),
            },
            {"label": "收益契约阻断", "value": blocked},
        ],
        next_actions=_safe_list(report.get("next_actions")),
        owner_path="services/strategy_signal_root_cause_audit.py",
    )


async def _production_source_health_audit() -> dict[str, Any]:
    try:
        report = await ProductionSourceHealthService().report(
            hours=AUDIT_WINDOWS["strategy_hours"],
            limit=5000,
            decision_interval_seconds=int(settings.decision_interval_seconds or 60),
        )
    except Exception as exc:
        return _audit_card(
            "production_source_health",
            "连续无生产收益源",
            "warning",
            "生产收益源连续性审计不可用；正式开仓继续失败关闭。",
            details={"error": safe_error_text(exc, limit=180)},
            owner_path="services/production_source_health.py",
        )
    status = str(report.get("status") or "warning")
    duration = report.get("continuous_no_source_seconds")
    if status == "ok":
        summary = "近期存在通过治理的生产收益源。"
    elif report.get("sampling_plan_alert_active"):
        summary = (
            "Paper bootstrap canary sampling plan is unreachable; the alert is active "
            "while controlled collection continues."
        )
    elif report.get("recovery_state") == "paper_normal_trading":
        summary = "正式生产收益源仍为空；模拟盘正常策略交易与持续训练正在运行，实盘继续保持关闭。"
    elif report.get("recovery_state") == "paper_bootstrap_collecting":
        summary = "正式生产收益源仍为空，模拟盘 bootstrap canary 正在采集恢复证据。"
    else:
        summary = "连续没有通过治理的生产收益源，系统已告警并保持正式开仓失败关闭。"
    return _audit_card(
        "production_source_health",
        "连续无生产收益源",
        status if status in {"ok", "warning", "critical"} else "warning",
        summary,
        details=report,
        evidence=[
            {"label": "连续无源秒数", "value": duration},
            {
                "label": "生产源决策",
                "value": int(report.get("production_source_decision_count") or 0),
            },
            {
                "label": "模拟策略已执行",
                "value": int(
                    report.get("paper_normal_executed_count")
                    or report.get("paper_bootstrap_executed_count")
                    or 0
                ),
            },
            {
                "label": "固定样本目标",
                "value": report.get("sample_target"),
            },
        ],
        next_actions=[
            "检查模拟盘正常策略的运行时风控、成交、费用结算和版本归因。",
            "所有完整平仓继续进入训练，不再以固定样本数量控制开仓。",
            "只有 walk-forward、样本外和权威成交费后收益下界转正后才恢复正式生产源。",
        ],
        owner_path="services/production_source_health.py",
    )


async def _strategy_closed_loop_audit() -> dict[str, Any]:
    try:
        root_report, contract_report = await asyncio.gather(
            StrategySignalRootCauseAuditService().report(),
            TradeExecutionContractService().report(
                hours=TRADE_EXECUTION_CONTRACT_AUDIT_HOURS,
                limit=TRADE_EXECUTION_CONTRACT_AUDIT_LIMIT,
            ),
        )
        root_report = _safe_strategy_signal_root_cause_report(root_report)
        contract_report = _safe_trade_execution_contract_report(contract_report)
    except Exception as exc:
        return _audit_card(
            "strategy_closed_loop",
            "动态费后收益闭环",
            "warning",
            "闭环审计不可用；当前不会修改任何生产策略。",
            details={"error": safe_error_text(exc, limit=180), "audit_only": True},
            owner_path="services/trade_execution_contract.py",
        )
    contract_summary = _safe_dict(contract_report.get("summary"))
    violations = int(contract_summary.get("contract_violation_count") or 0)
    blocked = int(root_report.get("live_ml_blocked_count") or 0)
    status = "critical" if violations else "warning" if blocked else "ok"
    return _audit_card(
        "strategy_closed_loop",
        "动态费后收益闭环",
        status,
        (
            "已执行决策违反收益契约。"
            if violations
            else "收益输入、执行契约和已实现盈亏均可审计。"
        ),
        details={
            "audit_only": True,
            "live_mutation": False,
            "strategy_signal": root_report,
            "trade_execution_contract": contract_report,
        },
        evidence=[
            {"label": "输入契约阻断", "value": blocked},
            {"label": "执行违规", "value": violations},
            {
                "label": "已实现净盈亏",
                "value": float(contract_summary.get("realized_net_pnl_usdt") or 0.0),
            },
        ],
        owner_path="services/trade_execution_contract.py",
    )


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
















async def _trade_execution_contract_audit() -> dict[str, Any]:
    try:
        report = _safe_trade_execution_contract_report(
            await TradeExecutionContractService().report(
                hours=TRADE_EXECUTION_CONTRACT_AUDIT_HOURS,
                limit=TRADE_EXECUTION_CONTRACT_AUDIT_LIMIT,
            )
        )
    except Exception as exc:
        return _audit_card(
            "trade_execution_contract",
            "动态费后收益执行契约",
            "warning",
            "收益契约审计不可用；生产策略继续保持保守关闭。",
            details={
                "error": safe_error_text(exc, limit=180),
                "audit_only": True,
                "live_entry_mutation": False,
                "live_exit_mutation": False,
                "can_bypass_risk_controls": False,
                "summary": {},
            },
            next_actions=[
                "修改任何生产收益策略前，先恢复只读审计。"
            ],
            owner_path="services/trade_execution_contract.py",
        )

    summary = _safe_dict(report.get("summary"))
    violation_count = int(summary.get("contract_violation_count") or 0)
    return _audit_card(
        "trade_execution_contract",
        "动态费后收益执行契约",
        "critical" if violation_count else "ok",
        (
            "已执行决策违反动态费后收益契约。"
            if violation_count
            else "已执行开仓和平仓满足动态费后收益契约。"
        ),
        details=report,
        evidence=[
            {"label": "已执行开仓", "value": int(summary.get("executed_entry_count") or 0)},
            {
                "label": "开仓契约完整",
                "value": int(summary.get("entry_contract_ready_count") or 0),
            },
            {"label": "已执行平仓", "value": int(summary.get("executed_exit_count") or 0)},
            {
                "label": "平仓契约完整",
                "value": int(summary.get("exit_contract_ready_count") or 0),
            },
            {"label": "违规项", "value": violation_count},
            {
                "label": "已实现净盈亏",
                "value": float(summary.get("realized_net_pnl_usdt") or 0.0),
            },
        ],
        next_actions=(
            [
                "阻断任何缺少正向费后收益置信下界、实时成本、风险预算或来源证据的执行路径。"
            ]
            if violation_count
            else ["继续审计已实现费后收益和左尾结果。"]
        ),
        owner_path="services/trade_execution_contract.py",
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


def _model_training_health_summary(
    *,
    local_tools: dict[str, Any],
    specialist_shadow_evaluation: dict[str, Any],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    local_stage = str(local_tools.get("model_stage") or local_tools.get("training_mode") or "shadow")
    trained_at = str(local_tools.get("trained_at") or "")
    trained_models_available = bool(
        local_tools.get("trained_models_available")
        or local_tools.get("model_bundle_available")
        or trained_at
    )
    promotion = (
        local_tools.get("promotion_recommendation")
        if isinstance(local_tools.get("promotion_recommendation"), dict)
        else {}
    )
    live_ready = bool(promotion.get("live_ready"))
    canary_ready = bool(promotion.get("canary_ready"))
    if trained_models_available:
        if live_ready or local_stage == "live":
            local_status = "trained_and_loaded"
        elif canary_ready or local_stage == "canary":
            local_status = "trained_canary_ready"
        elif local_stage in {"degraded", "shadow"}:
            local_status = "trained_but_not_promoted"
        else:
            local_status = "trained_shadow"
    elif bool(local_tools.get("available")):
        local_status = "service_online_without_training_artifact"
    else:
        local_status = "service_unavailable"
    items.append(
        {
            "model": "local_ai_tools_quant_bundle",
            "kind": "project_trained_bundle",
            "training_status": local_status,
            "trained_at": trained_at,
            "model_stage": local_stage,
            "live_ready": live_ready,
            "canary_ready": canary_ready,
            "sample_counts": {
                "shadow": _safe_int_value(local_tools.get("shadow_sample_count")),
                "trade": _safe_int_value(local_tools.get("trade_sample_count")),
                "text_sentiment": _safe_int_value(local_tools.get("text_sentiment_sample_count")),
            },
            "note": "本项目训练产物；未晋升时只能作为 shadow/观察证据。",
        }
    )
    for model in _safe_list(specialist_shadow_evaluation.get("models")):
        if not isinstance(model, dict):
            continue
        actual_count = _safe_int_value(model.get("actual_inference_count"))
        promotion_ready = bool(model.get("promotion_ready"))
        if actual_count <= 0:
            training_status = "installed_or_reported_but_actual_inference_missing"
        elif promotion_ready:
            training_status = "shadow_inference_evaluated_promotion_ready"
        else:
            training_status = "shadow_inference_evaluated_not_promoted"
        items.append(
            {
                "model": model.get("model"),
                "kind": "specialist_shadow_evidence_source",
                "tool": model.get("tool"),
                "training_status": training_status,
                "actual_inference_count": actual_count,
                "sample_count": _safe_int_value(model.get("sample_count")),
                "promotion_ready": promotion_ready,
                "promotion_blockers": _safe_list(model.get("promotion_blockers"))[:8],
                "note": "预训练/专业证据源；只有 actual inference 和影子表现达标后才可晋升。",
            }
        )
    if not bool(specialist_shadow_evaluation.get("available")):
        items.append(
            {
                "model": "timeseries_and_sentiment_specialist_shadow",
                "kind": "specialist_shadow_evaluation",
                "training_status": "evaluation_report_missing",
                "reason": specialist_shadow_evaluation.get("reason"),
                "note": "模型文件存在不等于进入训练闭环；缺少评估报告时必须按未闭环处理。",
            }
        )
    counts = Counter(str(item.get("training_status") or "unknown") for item in items)
    return {
        "policy": "distinguish_project_training_from_pretrained_or_alias_models",
        "items": items,
        "status_counts": dict(counts),
        "trained_and_loaded_count": counts.get("trained_and_loaded", 0),
        "trained_not_promoted_count": sum(
            counts.get(key, 0)
            for key in (
                "trained_but_not_promoted",
                "trained_shadow",
                "trained_canary_ready",
                "shadow_inference_evaluated_not_promoted",
            )
        ),
        "actual_inference_missing_count": counts.get(
            "installed_or_reported_but_actual_inference_missing",
            0,
        ),
        "service_without_artifact_count": counts.get(
            "service_online_without_training_artifact",
            0,
        ),
    }


def _consume_background_task_exception(task: asyncio.Task[Any]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:
        return


async def _model_training_audit() -> dict[str, Any]:
    runtime_task = asyncio.create_task(
        asyncio.wait_for(
            collect_platform_runtime_status(),
            timeout=MODEL_RUNTIME_PROBE_TIMEOUT_SECONDS,
        )
    )
    runtime_task.add_done_callback(_consume_background_task_exception)
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
    training_scheduler_state = MODEL_TRAINING_STATE_STORE.read()
    training_scheduler_stale = bool(training_scheduler_state.get("heartbeat_stale"))
    training_scheduler_error = training_scheduler_state.get("status") == "error"
    training_scheduler_unavailable = training_scheduler_state.get("status") == "unavailable"
    training_timeout_exceeded = bool(training_scheduler_state.get("training_timeout_exceeded"))
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
        or training_scheduler_stale
        or training_scheduler_error
        or training_scheduler_unavailable
        or training_timeout_exceeded
    )
    evaluation_policy = (
        local_tools.get("evaluation_policy")
        if isinstance(local_tools.get("evaluation_policy"), dict)
        else {}
    )
    promotion_flow = (
        local_tools.get("promotion_flow")
        or evaluation_policy.get("promotion_flow")
        or "candidate_to_shadow_to_canary_to_active"
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
            "三期模型变更必须按影子 -> 灰度 -> 生产的顺序推进；"
            "审计可见性本身不得修改生产交易权重。"
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
    training_health_summary = _model_training_health_summary(
        local_tools=local_tools if isinstance(local_tools, dict) else {},
        specialist_shadow_evaluation=specialist_shadow_evaluation,
    )
    try:
        local_ml_status = MLSignalService().status()
    except Exception as exc:
        local_ml_status = {
            "available": False,
            "status": "status_error",
            "error": safe_error_text(exc, limit=180),
        }
    model_registry = build_model_training_registry(
        local_ml_status=local_ml_status,
        local_tools_status=local_tools if isinstance(local_tools, dict) else {},
        specialist_report=specialist_shadow_evaluation,
        model_server_report=_load_phase3_model_server_readiness_latest_report(),
    )
    model_registry["scheduler_state"] = training_scheduler_state
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
            observing_reasons.append("需要按三期干净训练视图重建旧模型产物")
        if artifact_retirement_audit_warning:
            observing_reasons.append("旧模型产物退役巡检暂不可用")
        if training_scheduler_stale:
            observing_reasons.append("训练调度心跳过期")
        if training_scheduler_error:
            observing_reasons.append("训练调度状态不可读")
        if training_scheduler_unavailable:
            observing_reasons.append("训练调度尚无持久心跳")
        if training_timeout_exceeded:
            observing_reasons.append("模型训练运行超时")
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
            "training_health_summary": training_health_summary,
            "model_registry": model_registry,
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
                or "candidate_to_shadow_to_canary_to_active",
                "reason": specialist_shadow_evaluation.get("reason"),
            },
            "historical_trade_fact_audit": historical_trade_fact_report,
            "artifact_retirement_audit": artifact_retirement_report,
            "model_training_scheduler_state": training_scheduler_state,
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
            "已退役或不可信的模型产物必须使用三期干净训练视图重建，重建前不得影响生产交易。",
            "只有三期重建预检显示就绪，才能执行持久化模型产物的确认命令。",
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
    warning_codes = {
        str(item.get("code") or "")
        for item in warnings
        if isinstance(item, dict)
    }
    report["observing"] = bool(
        warnings
        and not phase3_blocked
        and not blockers
        and warning_codes.issubset({"legacy_data_paths_preserved"})
    )
    status = "warning" if phase3_blocked or warnings else "ok"
    summary = "三期模型服务器资源释放和迁移检查已通过。"
    if phase3_blocked:
        summary = "三期模型服务器仍被阻断：旧资源释放、/data/BB 隔离或白名单迁移尚未验证完成。"
    elif warnings:
        summary = "三期模型服务器可用；仅保留按策略隔离的旧数据路径观察项，不阻断运行。"
    return _audit_card(
        "phase3_server_migration",
        "三期模型服务器资源释放与迁移检查",
        status,
        summary,
        details=report,
        evidence=[
            {"label": "生产启用是否阻断", "value": phase3_blocked},
            {"label": "硬阻断", "value": len(blockers)},
            {"label": "已隔离旧数据路径", "value": report.get("legacy_data_path_count") or 0},
            {"label": "旧服务残留", "value": report.get("forbidden_service_count") or 0},
            {
                "label": "白名单迁移项",
                "value": _safe_dict(report.get("migration_manifest")).get("item_count") or 0,
            },
        ],
        next_actions=[
            "检查未显示就绪前，不得启用三期模型服务器的生产影响。",
            "旧服务、进程和容器必须保持停止；旧数据只读隔离保留。",
            "只允许迁移白名单清单中的内容，禁止整体复制旧服务器。",
            "三期模型、缓存、训练、运行和日志数据统一保存在 /data/BB。",
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
    summary = "三期量化模型服务器的产物和服务均已就绪。"
    if blockers:
        summary = (
            "三期量化模型服务器因产物、CUDA、GPU 或策略就绪检查失败而阻断。"
        )
    elif not runtime_ready:
        summary = (
            "三期量化模型产物已就绪，但模型服务或接口尚未运行。"
        )
    elif warnings:
        summary = "三期量化模型服务器可用，但仍有非阻断提示。"
    return _audit_card(
        "phase3_model_server_readiness",
        "三期量化模型服务器就绪检查",
        status,
        summary,
        details=report,
        evidence=[
            {"label": "模型产物就绪", "value": artifact_ready},
            {"label": "运行环境就绪", "value": runtime_ready},
            {"label": "生产启用阻断", "value": service_go_live_blocked},
            {"label": "GPU 数量", "value": report.get("gpu_count") or 0},
            {
                "label": "必需模型槽位",
                "value": (
                    f"{report.get('required_slot_ready_count') or 0}/"
                    f"{report.get('required_slot_count') or 0}"
                ),
            },
            {
                "label": "运行中接口",
                "value": report.get("active_endpoint_count") or 0,
            },
        ],
        next_actions=[
            "本检查返回 runtime_ready=true 前，不得把三期模型调用路由到新服务器。",
            "服务健康、延迟和晋升检查通过前，大模型角色只能处于影子或候选阶段。",
            "三期模型服务只能从 /data/BB 启动或安装，不得复用 /data/trade_ai 的旧服务。",
            "服务运行后重新执行本巡检，再以影子或灰度模式连接平台隧道。",
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
    summary = "三期模拟盘恢复硬检查已通过，但仍需要操作员批准。"
    if consumed_after_resume:
        summary = "三期模拟盘已恢复，恢复前检查已完成；现在以恢复后观察为准。"
    elif blockers:
        summary = "三期模拟盘恢复被硬性前置检查阻断。"
    elif warnings:
        summary = "三期模拟盘恢复检查可通过，但仍有提示需要复核。"
    details = dict(report)
    details["consumed_after_resume"] = consumed_after_resume
    details["observing"] = consumed_after_resume
    return _audit_card(
        "phase3_paper_resume_preflight",
        "三期模拟盘恢复硬检查",
        status,
        summary,
        details=details,
        evidence=[
            {"label": "可恢复模拟盘", "value": can_resume},
            {"label": "阻断项", "value": len(blockers)},
            {"label": "提示项", "value": len(warnings)},
            {
                "label": "OKX 问题",
                "value": _safe_dict(report.get("summary")).get("okx_issue_count") or 0,
            },
            {
                "label": "OKX 账户权益",
                "value": bool(
                    _safe_dict(report.get("summary")).get("okx_account_equity_available")
                ),
            },
            {
                "label": "模型运行环境",
                "value": bool(_safe_dict(report.get("summary")).get("model_server_runtime_ready")),
            },
            {
                "label": "量化接口",
                "value": bool(_safe_dict(report.get("summary")).get("phase3_quant_api_available")),
            },
        ],
        next_actions=[
            "can_resume_paper=true 前不得启动 bb-paper-trading.service。",
            "先清除 OKX 原生同步、交易事实完整性、模型服务器运行环境、隧道和专用模型影子评估阻断。",
            "检查通过后只能由获批的操作员动作恢复模拟盘，并继续关闭实盘交易。",
            "恢复后持续观察 OKX 权威同步和专用模型影子评估的新样本。",
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
    summary = "三期模拟盘恢复后观察正常。"
    if status_value == "waiting_for_resume":
        summary = (
            "模拟盘仍处于停止状态，三期恢复后观察正在等待。"
        )
    elif status_value == "warming_up":
        summary = "三期模拟盘已恢复，但观察样本仍在预热积累。"
    elif blockers:
        summary = "三期模拟盘恢复后观察发现硬阻断。"
    elif warnings:
        summary = "三期模拟盘恢复后观察有提示需要复核。"
    details = dict(report)
    details["observing"] = status_value in {"waiting_for_resume", "warming_up"}
    return _audit_card(
        "phase3_paper_resume_observation",
        "三期模拟盘恢复后观察",
        status,
        summary,
        details=details,
        evidence=[
            {"label": "模拟盘运行中", "value": bool(report.get("paper_active"))},
            {"label": "阻断项", "value": len(blockers)},
            {"label": "提示项", "value": len(warnings)},
            {
                "label": "已创建影子样本",
                "value": _safe_dict(report.get("summary")).get("created_shadow_count") or 0,
            },
            {
                "label": "已完成影子样本",
                "value": _safe_dict(report.get("summary")).get("completed_shadow_count") or 0,
            },
            {
                "label": "专用模型可评估样本",
                "value": _safe_dict(report.get("summary")).get("specialist_eligible_shadow_count")
                or 0,
            },
        ],
        next_actions=[
            "模拟盘启动前，把本卡片作为零样本基线。",
            "模拟盘启动后，观察前 30/60/120 分钟的 OKX 干净状态和样本积累。",
            "本观察正常且样本要求通过前，不得晋升专用模型。",
        ],
    )


async def _phase3_stage_handoff_audit() -> dict[str, Any]:
    report = await asyncio.to_thread(Phase3StageHandoffService().report)
    blockers = _safe_list(report.get("blockers"))
    warnings = _safe_list(report.get("warnings"))
    ready = bool(report.get("ready")) and str(report.get("status") or "") == "dynamic_return_ready"
    status = "critical" if blockers else "warning" if warnings else "ok"
    summary = (
        "三期动态费后收益观察被阻断。"
        if blockers
        else (
            "三期动态费后收益观察已就绪，但提示项不授予生产权限。"
            if warnings
            else "三期动态费后收益观察已就绪，并继续保持无生产权限。"
        )
    )
    details = dict(report)
    details["observing"] = not blockers
    return _audit_card(
        "phase3_stage_handoff",
        "三期动态费后收益观察",
        status,
        summary,
        details=details,
        evidence=[
            {"label": "已就绪", "value": ready},
            {"label": "生产权限", "value": False},
            {"label": "阻断项", "value": len(blockers)},
            {"label": "提示项", "value": len(warnings)},
        ],
        next_actions=(
            ["修复所有动态收益、OKX 事实或观察边界阻断。"]
            if blockers
            else ["保持本报告只读；它不能启动交易或晋升模型。"]
        ),
        owner_path="services/phase3_stage_handoff.py",
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
    blockers = _safe_list(report.get("blockers"))
    warnings = _safe_list(report.get("warnings"))
    ready = bool(report.get("ready")) and str(report.get("status") or "") == "go"
    card_status = "critical" if blockers else "warning" if warnings else "ok"
    summary = (
        "动态费后收益架构被阻断。"
        if blockers
        else (
            "动态费后收益架构已就绪，但仍有可观察提示。"
            if warnings
            else "动态费后收益架构满足全部必需契约。"
        )
    )
    report_summary = _safe_dict(report.get("summary"))
    return _audit_card(
        "phase3_go_no_go",
        "三期动态费后收益检查",
        card_status,
        summary,
        details=report,
        evidence=[
            {"label": "已就绪", "value": ready},
            {"label": "执行违规", "value": int(report_summary.get("current_contract_violation_count") or 0)},
            {"label": "持仓经济性缺口", "value": int(report_summary.get("position_economics_incomplete_count") or 0)},
            {"label": "动态退出缺口", "value": int(report_summary.get("executed_dynamic_exit_contract_gap_count") or 0)},
            {"label": "阻断项", "value": len(blockers)},
            {"label": "提示项", "value": len(warnings)},
        ],
        next_actions=(
            ["晋升前修复所有列出的收益、成本、来源、持仓经济性或训练阻断。"]
            if blockers
            else ["所有专家、记忆、影子和策略学习输出继续只作观察。"]
        ),
        owner_path="services/phase3_go_no_go.py",
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
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [pattern for pattern in STRATEGY_GATE_FORBIDDEN_PATTERNS if pattern in text]
        if hits:
            offenders.append({"path": rel_path, "patterns": hits})
    try:
        policy_source = ast.parse(
            (root / "services/dynamic_policy_values.py").read_text(encoding="utf-8")
        )
        return_source = ast.parse(
            (root / "services/live_ml_profit_contract.py").read_text(encoding="utf-8")
        )
        runtime_contract_available = any(
            isinstance(node, ast.ClassDef) and node.name == "DynamicPolicyValue"
            for node in ast.walk(policy_source)
        ) and any(
            isinstance(node, ast.ClassDef) and node.name == "LiveMLProfitContractAssessment"
            for node in ast.walk(return_source)
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
            "生产执行只接受动态费后收益、实时成本、风险预算和完整来源契约。"
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
            "删除任何重新引入 settings.min_entry_*、RuntimeEntryFilters 或固定策略比较的代码。",
            "缺收益、成本、有效期或来源时必须 fail-closed。",
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
    if status == "warning" and bool(details.get("observing")):
        return "observing", "观察项 / 受控阶段或预热中"
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
        return "observing", "观察项 / 行情数据预热覆盖扩展中"
    if key == "okx_trade_fact_integrity" and status == "warning":
        runtime_gate = _safe_dict(details.get("runtime_okx_entry_gate"))
        runtime_blocker = str(runtime_gate.get("blocker") or "")
        link_repair = _safe_dict(details.get("position_fact_link_repair"))
        authoritative_sync = _safe_dict(details.get("okx_authoritative_sync"))
        daily_report = _safe_dict(details.get("daily_reconciliation_report"))
        unresolved_link_candidate_count = _okx_unresolved_link_candidate_count(
            details,
            link_repair,
        )
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
        historical_links_isolated = bool(
            unresolved_link_candidate_count > 0
            and not has_data_integrity_issue
            and runtime_sync_healthy
            and link_repair.get("read_only") is True
            and not bool(link_repair.get("live_repair_mutation"))
            and daily_report.get("can_open_new_entries") is True
            and daily_report.get("can_refresh_training") is True
            and daily_report.get("requires_attention") is False
        )
        if historical_links_isolated:
            return "observing", "历史事实链接只读隔离 / 当前交易与训练口径正常"
        if (
            (runtime_only_blocked or authoritative_pull_failed)
            and not has_data_integrity_issue
            and (runtime_only_blocked or runtime_sync_healthy)
        ):
            return "observing", "观察项 / OKX 运行同步健康或仅运行态待观察"
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
                return "observing", "观察项 / OKX 一致性仅有信息级历史残留"
    if key == "phase3_go_no_go" and status == "warning":
        if (
            str(details.get("status") or "") == "go"
            and bool(details.get("ready"))
            and not _safe_list(details.get("blockers"))
        ):
            return "observing", "观察项 / 动态费后收益门存在非阻断提示"
    if key == "phase3_stage_handoff" and status == "warning":
        if (
            str(details.get("status") or "") == "dynamic_return_ready"
            and bool(details.get("audit_only"))
            and bool(details.get("read_only"))
            and not bool(details.get("production_permission"))
            and not bool(details.get("starts_trading_service"))
            and not bool(details.get("submits_orders"))
            and not bool(details.get("changes_model_routing"))
            and not bool(details.get("live_mutation"))
            and not _safe_list(details.get("blockers"))
        ):
            return "observing", "观察项 / 动态费后收益准备状态"
    if (
        key == "trade_loop"
        and status == "warning"
        and bool(details.get("dynamic_return_gate_ready"))
    ):
        return "observing", "观察项 / 收益门已就绪但交易服务当前停止"
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
        return "observing", "观察项 / 基线或竞赛样本不足"
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
        return "observing", "观察项 / 高风险复核门只读检查"
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
        return "observing", "观察项 / 强机会分类器处于影子审计阶段"
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
        return "observing", "观察项 / 仓位容量释放只读检查"
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
        return "observing", "观察项 / 模拟盘恢复预热窗口"
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
        return "observing", "观察项 / 策略信号根因只读检查"
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
            "三期服务器资源释放与迁移",
            "基础设施层",
            cards_by_key,
            ["phase3_server_migration"],
            impact=(
                "若旧服务或进程仍占用资源、/data/BB 未隔离，或旧服务器迁移未限定白名单，"
                "则阻断三期模型服务器启用。"
            ),
            downstream=["model_training", "model_expert_health", "strategy_decision"],
            checks=[
                "资源释放证明",
                "/data/BB 隔离",
                "旧服务与进程已停止",
                "白名单迁移清单",
                "旧数据只读隔离保留",
            ],
        ),
        _node_from_cards(
            "model_server_readiness",
            "三期量化模型服务器就绪检查",
            "模型基础设施层",
            cards_by_key,
            ["phase3_model_server_readiness"],
            impact=(
                "模型产物、CUDA/GPU 验证、服务契约或模型接口未就绪时，"
                "阻断三期模型服务器的影子或灰度路由。"
            ),
            upstream=["server_migration"],
            downstream=["model_training", "model_expert_health", "model_dynamic_routing"],
            checks=[
                "下载清单",
                "验证清单",
                "8 张 GPU 的 CUDA 验证",
                "必需量化模型槽位",
                "模型服务接口",
            ],
        ),
        _node_from_cards(
            "phase3_stage_handoff",
            "三期受控阶段交接",
            "发布检查层",
            cards_by_key,
            ["phase3_stage_handoff", "phase3_go_no_go"],
            impact=(
                "只展示三期下一步允许动作：保持影子、经明确批准启动模拟盘、观察恢复后状态或复核灰度。"
                "本节点不会自行启动模拟盘、晋升灰度或启用生产路由。"
            ),
            upstream=["server_migration", "model_server_readiness", "okx_execution"],
            downstream=["runtime_loop", "strategy_closed_loop", "model_routing"],
            checks=[
                "启用/不启用结论新鲜度",
                "模拟盘启动审批边界",
                "恢复后观察",
                "专用模型影子证据",
                "生产影响已关闭",
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
            impact="对模型/专家与基线的离线/影子/模拟竞赛结果做证据化比较，不直接改真实权重。",
            upstream=["model_expert_health", "model_training"],
            downstream=["model_routing", "strategy_decision"],
            checks=["基线对比", "影子竞赛", "模拟 A/B", "权重建议来源"],
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
            "high_risk_review_audit",
            "高风险独立复核",
            "风控层",
            cards_by_key,
            ["high_risk_review_audit"],
            impact="审计独立高风险复核的触发、审批、阻断和不安全执行，不修改生产门。",
            upstream=["model_dynamic_routing", "strategy_decision"],
            downstream=["risk_guard", "okx_execution"],
            checks=["硬复核触发", "审批状态", "阻断数量", "不安全执行"],
        ),
        _node_from_cards(
            "shadow_missed_opportunity",
            "影子错失机会复盘",
            "学习层",
            cards_by_key,
            ["shadow_missed_opportunity"],
            impact="审计错失机会是否只有在同币种同方向证据重复出现后才可用于学习。",
            upstream=["strategy_closed_loop", "model_training"],
            downstream=["strategy_decision", "training_data"],
            checks=[
                "同币种同方向重复证据",
                "稳定正收益",
                "低风险证据",
                "弱证据执行",
            ],
        ),
        _node_from_cards(
            "strong_opportunity",
            "强机会识别",
            "策略层",
            cards_by_key,
            ["strong_opportunity"],
            impact="审计二阶段强机会形态，不修改生产开仓、仓位、杠杆或风险门。",
            upstream=[
                "market_data",
                "model_training",
                "shadow_missed_opportunity",
                "okx_execution",
            ],
            downstream=["strategy_decision", "risk_guard", "training_data"],
            checks=[
                "选中方向预期净收益",
                "收益质量",
                "亏损概率",
                "尾部风险",
                "一致来源",
                "只读标记",
            ],
        ),
        _node_from_cards(
            "position_capacity_release",
            "持仓容量释放",
            "风控层",
            cards_by_key,
            ["position_capacity_release"],
            impact="在开仓策略参数变化前，审计容量压力、释放候选、旧盈利轮换候选和未闭环释放决策。",
            upstream=["position_sync", "strong_opportunity", "strategy_closed_loop"],
            downstream=["strategy_decision", "risk_guard", "okx_execution"],
            checks=[
                "当前容量",
                "释放候选",
                "旧盈利候选",
                "未闭环释放决策",
                "拥挤方向阻断",
                "只读标记",
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
                "high_risk_review_audit",
                "trade_execution_contract",
            ],
            impact="影响是否开仓、仓位大小、重复亏损复开和快进快出。",
            upstream=[
                "market_data",
                "model_training",
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
                "本地 ML 就绪状态",
                "服务器收益模型贡献",
                "影子错失机会转化",
                "预期净收益组成",
                "候选集中度",
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
            checks=["DynamicPolicyValue", "LiveMLProfitContractAssessment", "旧固定阈值残留"],
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


def _latest_audit_path() -> Path:
    return settings.data_dir / SYSTEM_AUDIT_LATEST_FILE


def _load_latest_audit_snapshot() -> tuple[datetime, dict[str, Any]] | None:
    path = _latest_audit_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    checked_at = _parse_utc_datetime(payload.get("checked_at"))
    if checked_at is None:
        return None
    return checked_at, payload


def _store_latest_audit_snapshot(payload: dict[str, Any]) -> None:
    path = _latest_audit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f"{path.suffix}.tmp")
    temp.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_audit_snapshot_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _audit_snapshot_json_default(value: Any) -> str:
    if isinstance(value, datetime):
        normalized = value if value.tzinfo else value.replace(tzinfo=UTC)
        return normalized.astimezone(UTC).isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _cached_system_audit_status() -> tuple[datetime, dict[str, Any]] | None:
    global _system_audit_status_cache
    if _system_audit_status_cache is None:
        _system_audit_status_cache = _load_latest_audit_snapshot()
    if _system_audit_status_cache is None:
        return None
    checked_at, payload = _system_audit_status_cache
    return checked_at, copy.deepcopy(payload)


async def _refresh_system_audit_status() -> None:
    global _system_audit_refresh_task
    try:
        await collect_system_audit_status(record_history=True, source="background_api_refresh")
    finally:
        _system_audit_refresh_task = None


def _schedule_system_audit_refresh() -> None:
    global _system_audit_refresh_task
    if _system_audit_refresh_task is None or _system_audit_refresh_task.done():
        _system_audit_refresh_task = asyncio.create_task(_refresh_system_audit_status())


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
    collection_started = time.perf_counter()
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
        ("production_source_health", _production_source_health_audit),
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
    section_timings: dict[str, float] = {}
    result_by_key: dict[str, dict[str, Any] | Exception] = {}
    result_by_key.update(
        await _run_audit_specs(
            priority_specs,
            max_concurrency=1,
            timings=section_timings,
        )
    )
    result_by_key.update(
        await _run_audit_specs(
            db_specs,
            max_concurrency=1,
            timings=section_timings,
        )
    )
    result_by_key.update(
        await _run_audit_specs(
            regular_specs,
            max_concurrency=SYSTEM_AUDIT_MAX_CONCURRENCY,
            timings=section_timings,
        )
    )
    result_by_key.update(
        await _run_audit_specs(
            heavy_specs,
            max_concurrency=1,
            timings=section_timings,
        )
    )
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
            "performance": {
                "total_seconds": round(time.perf_counter() - collection_started, 4),
                "section_seconds": dict(
                    sorted(section_timings.items(), key=lambda item: item[1], reverse=True)
                ),
                "group_concurrency": {
                    "priority": 1,
                    "database": 1,
                    "regular": SYSTEM_AUDIT_MAX_CONCURRENCY,
                    "heavy": 1,
                },
            },
            "safety_note": "根因雷达当前只读巡检；补历史仓位、重启服务、批量训练等动作必须人工确认。",
        }
    )
    if record_history:
        _append_history_record(payload, source=source)
    global _system_audit_status_cache
    checked_at = _parse_utc_datetime(payload.get("checked_at")) or _now()
    _system_audit_status_cache = (checked_at, copy.deepcopy(payload))
    try:
        _store_latest_audit_snapshot(payload)
    except OSError as exc:
        logger.warning("failed to store system audit snapshot", error=safe_error_text(exc))
    return payload


@router.get("/system-audit/status")
async def system_audit_status() -> dict[str, Any]:
    cached = _cached_system_audit_status()
    if cached is None:
        return await collect_system_audit_status(record_history=True, source="api_cold_start")
    checked_at, payload = cached
    age_seconds = max((_now() - checked_at).total_seconds(), 0.0)
    refresh_after = max(float(settings.system_audit_history_interval_seconds or 300), 60.0)
    if age_seconds >= refresh_after:
        _schedule_system_audit_refresh()
    payload["cache"] = {
        "hit": True,
        "age_seconds": round(age_seconds, 3),
        "refresh_after_seconds": round(refresh_after, 3),
        "refresh_in_background": age_seconds >= refresh_after,
    }
    return sanitize_payload(payload)


@router.get("/model-expert-health/status")
async def model_expert_health_status(hours: int = 72, limit: int = 1200) -> dict[str, Any]:
    report = await ModelExpertHealthService().report(hours=hours, limit=limit)
    report["audit_only"] = True
    report["live_weight_mutation"] = False
    components = report.get("components") if isinstance(report.get("components"), dict) else {}
    for row in components.values():
        if isinstance(row, dict):
            row["recommended_state"] = "observation_only"
            row["production_permission"] = False
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
            row["recommended_weight_action"] = "observation_only"
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
