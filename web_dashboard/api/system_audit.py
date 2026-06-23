"""Root-cause radar API for online system audits."""

from __future__ import annotations

import ast
import asyncio
import copy
import inspect
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from sqlalchemy import func, select

from config.settings import settings
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol
from db.session import get_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest, TradeReflection
from models.market_data import Kline, Ticker
from models.trade import Order, Position
from scripts.audit_runtime_text_integrity import collect_runtime_text_integrity_report
from scripts.repair_missing_closed_positions_from_orders import (
    collect_missing_closed_position_plans,
)
from services.crypto_feature_coverage import CryptoFeatureCoverageService
from services.exchange_position_state import (
    exchange_position_display_valuation,
    parse_exchange_position_snapshot,
)
from services.model_dynamic_routing import ModelDynamicRoutingService
from services.model_expert_competition import ModelExpertCompetitionService
from services.model_expert_health import ModelExpertHealthService
from services.server_monitor_status import collect_platform_runtime_status
from services.shadow_missed_opportunity_closed_loop import (
    ShadowMissedOpportunityClosedLoopService,
)
from services.trade_execution_contract import TradeExecutionContractService
from services.trading_params import DEFAULT_TRADING_PARAMS
from web_dashboard.api import data_collection as data_collection_api
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()

AUDIT_WINDOWS = {"fast_minutes": 10, "trade_hours": 2, "strategy_hours": 24}
EXPECTED_KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h")
KLINE_STALE_LIMIT_SECONDS = {"1m": 120, "5m": 600, "15m": 1800, "1h": 7200}
STATUS_RANK = {"critical": 0, "warning": 1, "ok": 2, "info": 3}
SYSTEM_AUDIT_HISTORY_FILE = "system_audit_history.jsonl"
POSITION_PRICE_SPLIT_WARN_PCT = 0.03
POSITION_PNL_SPLIT_WARN_USDT = 0.5
OKX_RECONCILIATION_CACHE_TTL_SECONDS = 120
MODEL_RUNTIME_PROBE_TIMEOUT_SECONDS = 8.0
SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS = 20.0
MODEL_EXPERT_AUDIT_HOURS = 24
MODEL_EXPERT_AUDIT_LIMIT = 200
SHADOW_MISSED_OPPORTUNITY_AUDIT_HOURS = 24
SHADOW_MISSED_OPPORTUNITY_AUDIT_LIMIT = 200
OPTIONAL_TRAINING_SOURCE_STATUSES = {"disabled", "not_configured"}
TRADE_EXECUTION_CONTRACT_AUDIT_HOURS = 24
TRADE_EXECUTION_CONTRACT_AUDIT_LIMIT = 500
PRIORITY_AUDIT_KEYS = ("trade_execution_contract",)
CARD_OWNER_PATHS = {
    "trade_loop": "services/trading_service.py",
    "okx_reconciliation": "scripts/repair_missing_closed_positions_from_orders.py",
    "position_price_integrity": "web_dashboard/api/system_audit.py",
    "market_data": "models/market_data.py",
    "strategy_quality": "web_dashboard/api/system_audit.py",
    "strategy_closed_loop": "web_dashboard/api/system_audit.py",
    "strategy_gate_contract": "services/runtime_entry_filters.py",
    "model_training": "web_dashboard/api/data_collection.py",
    "model_expert_health": "services/model_expert_health.py",
    "model_expert_competition": "services/model_expert_competition.py",
    "model_dynamic_routing": "services/model_dynamic_routing.py",
    "crypto_feature_coverage": "services/crypto_feature_coverage.py",
    "shadow_missed_opportunity": "services/shadow_missed_opportunity_closed_loop.py",
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
    "shadow_missed_opportunity": "services/shadow_missed_opportunity_closed_loop.py",
    "strategy_decision": "services/trading_policies.py",
    "strategy_closed_loop": "web_dashboard/api/system_audit.py",
    "strategy_gate_contract": "services/runtime_entry_filters.py",
    "risk_guard": "services/trading_policies.py",
    "okx_execution": "services/execution_service.py",
    "position_sync": "services/position_sync_service.py",
    "training_data": "services/training_data_quality.py",
    "dashboard_observability": "web_dashboard/static/js/dashboard.js",
    "visible_text_encoding": "web_dashboard/api/system_audit.py",
    "runtime_text_integrity": "scripts/audit_runtime_text_integrity.py",
}

_okx_reconciliation_cache: tuple[datetime, dict[str, Any]] | None = None


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


def _safe_trade_execution_contract_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report if isinstance(report, dict) else {})
    safe["audit_only"] = True
    safe["live_entry_mutation"] = False
    safe["live_exit_mutation"] = False
    safe["can_bypass_risk_controls"] = False
    policy = _safe_dict(safe.get("policy"))
    policy["entry_requires_positive_expected_net"] = True
    policy["entry_requires_structured_evidence"] = True
    policy["position_size_requires_profit_risk_sizing"] = True
    policy["fast_loss_exit_requires_strong_exit_evidence"] = True
    policy["recent_loss_reentry_requires_strong_unlock"] = True
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


async def _audit_maybe_async(factory: Any) -> dict[str, Any]:
    result = factory()
    if inspect.isawaitable(result):
        result = await asyncio.wait_for(
            result,
            timeout=max(float(SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS or 20.0), 0.001),
        )
    return result


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
    market_analysis_paused = (
        bool(runtime_window.get("running"))
        and bool(runtime_window.get("paused"))
        and runtime_heartbeat_fresh
    )
    stalled = (
        not market_analysis_paused
        and not cold_start
        and (recent_count == 0 or (latest_decision_age is not None and latest_decision_age > 600))
    )
    cold_start_no_orders = cold_start and orders_count == 0
    status = _status_from_counts(
        critical=stalled,
        warning=market_analysis_paused
        or cold_start_no_orders
        or (orders_count == 0 and decisions_count > 30),
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
            "market_analysis_paused": market_analysis_paused,
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
                "running": bool(runtime_window.get("running")),
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
        plans = await asyncio.wait_for(collect_missing_closed_position_plans(days=14), timeout=8.0)
    except Exception as exc:
        timeout = isinstance(exc, TimeoutError)
        return _store_okx_reconciliation_card(
            _audit_card(
                "okx_reconciliation",
                "OKX 历史对账",
                "warning",
                (
                    "OKX 历史对账 dry-run 超时；当前不能证明存在缺失仓位，先观察并重试。"
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
                    "dry-run 超时时先重试巡检或缩小窗口，不能直接补历史仓位。",
                    "如果连续超时，再查数据库慢查询、订单数量和对账索引。",
                ],
            )
        )
    missing = len(plans)
    status = "critical" if missing else "ok"
    return _store_okx_reconciliation_card(
        _audit_card(
            "okx_reconciliation",
            "OKX 历史对账",
            status,
            (
                "存在可由 OKX 成交订单反推的缺失历史仓位。"
                if missing
                else "14 天历史仓位 dry-run 无缺失。"
            ),
            details={
                "window_days": 14,
                "missing_closed_positions": missing,
                "sample_plans": [
                    {
                        "symbol": plan.symbol,
                        "side": plan.side,
                        "quantity": plan.quantity,
                        "realized_pnl": round(float(plan.realized_pnl), 8),
                        "close_order_id": plan.close_order_id,
                        "closed_at": _iso(plan.closed_at),
                    }
                    for plan in plans[:5]
                ],
            },
            evidence=[{"label": "缺失闭仓", "value": missing}],
            next_actions=[
                "只允许先 dry-run 人工核对，再按 symbol/order-id 精确 apply。",
                "如果缺失不为 0，先不要做策略收益判断，避免训练和盈亏被脏账影响。",
            ],
        )
    )


async def _position_price_integrity_audit() -> dict[str, Any]:
    from web_dashboard.api import dashboard as dashboard_api

    split_rows: list[dict[str, Any]] = []
    checked_modes: list[str] = []
    unavailable_modes: list[dict[str, str]] = []
    local_open_count = 0
    exchange_open_count = 0

    for mode in ("paper", "live"):
        executor = dashboard_api._dashboard_okx_executor_for_mode(mode)
        if not executor:
            continue
        checked_modes.append(mode)
        try:
            exchange_positions = await asyncio.wait_for(executor.get_positions(), timeout=1.8)
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

        for position in local_positions:
            key = (
                normalize_trading_symbol(position.symbol),
                str(position.side or "").lower(),
            )
            snapshot = exchange_snapshots.get(key)
            if not snapshot:
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
                }
            )

    status = _status_from_counts(critical=bool(split_rows), warning=bool(unavailable_modes))
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
            "split_count": len(split_rows),
            "price_gap_warn_pct": POSITION_PRICE_SPLIT_WARN_PCT * 100,
            "pnl_gap_warn_usdt": POSITION_PNL_SPLIT_WARN_USDT,
            "splits": split_rows[:12],
        },
        evidence=[
            {"label": "价格/浮盈分裂", "value": len(split_rows)},
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
    fast_loss_positions = []
    fast_loss_micro_positions = []
    for pos in closed_positions:
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


def _trade_contract_violation_counts(summary: dict[str, Any]) -> tuple[int, int]:
    hard = (
        int(summary.get("weak_evidence_executed_count") or 0)
        + int(summary.get("negative_expected_executed_count") or 0)
        + int(summary.get("fast_loss_without_strong_exit_count") or 0)
        + int(summary.get("reentry_without_strong_unlock_count") or 0)
    )
    soft = (
        int(summary.get("missing_entry_explanation_count") or 0)
        + int(summary.get("missing_sizing_explanation_count") or 0)
        + int(summary.get("small_size_without_reason_count") or 0)
    )
    return hard, soft


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
        "soft_violation_count": int(current_soft_violations),
        "hard_violation_count": int(current_hard_violations),
        "contract_violation_count": int(current_summary.get("contract_violation_count") or 0),
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
                "audit_only": True,
                "live_entry_mutation": False,
                "live_exit_mutation": False,
                "can_bypass_risk_controls": False,
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
            "violations": _safe_list(report.get("violations"))[:10],
            "current_entry_explanations": (
                _safe_list(current_report.get("entry_explanations"))[:10] if current_report else []
            ),
            "current_fast_loss_samples": (
                _safe_list(current_report.get("fast_loss_samples"))[:10] if current_report else []
            ),
            "current_violations": (
                _safe_list(current_report.get("violations"))[:10] if current_report else []
            ),
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
    status = str(report.get("status") or "ok")
    if status not in {"ok", "warning", "critical"}:
        status = _status_from_counts(critical=bool(not features), warning=bool(missing or stale))
    summary = (
        "核心行情或特征快照缺失，缺失特征已被中性阻断。"
        if status == "critical"
        else (
            f"发现 {len(missing)} 类缺失、{len(stale)} 类过期特征；已按中性/只读处理。"
            if missing or stale
            else "数字货币特征覆盖未发现缺失或过期项。"
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
            {"label": "已中性阻断", "value": len(neutralized)},
        ],
        next_actions=[
            "缺失数据源不得静默当作正常，也不得填成利于开仓的默认值。",
            "低可信事件只允许影子观察，不得直接驱动真实开仓。",
            "特征时间戳缺失或过期时，先修复采集链路再评估策略参数。",
        ],
    )


async def _model_training_audit() -> dict[str, Any]:
    data_status, runtime_status = await asyncio.gather(
        data_collection_api.get_data_collection_status(include_feature_coverage=False),
        asyncio.wait_for(
            collect_platform_runtime_status(),
            timeout=MODEL_RUNTIME_PROBE_TIMEOUT_SECONDS,
        ),
        return_exceptions=True,
    )
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
        }
    runtime_probe_timeout = bool(runtime_probe.get("timeout"))
    local_tools_status = str(local_tools.get("status") or "").lower()
    local_tools_unconfigured = (
        not bool(local_tools.get("available"))
        and local_tools_status in OPTIONAL_TRAINING_SOURCE_STATUSES
        and not bool(runtime_probe.get("local_ai_tools_configured"))
    )
    local_tools_hard_missing = (
        not bool(local_tools.get("available")) and not local_tools_unconfigured
    )
    runtime_probe_hard_failure = runtime_probe.get("status") == "warning" and not (
        runtime_probe_timeout and bool(local_tools.get("available"))
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
        or local_tools_unconfigured
        or runtime_probe_timeout
    )
    status = _status_from_counts(
        critical=bool(model_critical) or local_tools_hard_missing,
        warning=hard_failure or observing,
    )
    summary = "模型和训练数据状态正常。"
    if hard_failure:
        summary = "模型服务或训练数据源存在硬故障，需要处理。"
    elif observing:
        summary = "模型服务可用；可选增强数据源未配置、运行探针超时或模型仍在学习观察。"
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
            },
            "governance_status": governance.get("status") if isinstance(governance, dict) else None,
            "runtime_probe": runtime_probe,
            "hard_failure": hard_failure,
            "observing": observing,
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
        ],
        next_actions=[
            "模型 critical 时优先查端口契约 18000/18001/18002 和本地量化工具 API Key。",
            "可选增强源未配置只影响新闻/事件覆盖，不应误判为模型训练硬故障。",
            "learning_only 表示模型可用但仍需效果验证，继续看高分组收益和样本质量。",
        ],
    )


def _source_scan_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    closed_positions = [row for row in positions if not bool(row.is_open)]
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
    current_warning = any(
        diagnostics[key]
        for key in (
            "current_weak_executed",
            "current_no_high_quality_entries",
            "current_fast_loss_cluster",
            "current_ml_not_effective",
        )
    )
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
    elif current_warning:
        status = "warning"
        summary = "当前运行窗口仍存在弱证据执行、高质量候选不足、ML弱参与或快亏平风险。"
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
    return sorted(findings, key=lambda row: STATUS_RANK.get(str(row.get("severity")), 9))[:10]


def _strategy_closed_loop_is_historical_only(cards_by_key: dict[str, dict[str, Any]]) -> bool:
    card = cards_by_key.get("strategy_closed_loop") or {}
    state, _label = _issue_ledger_state(card, cards_by_key={})
    return state == "observing"


def _issue_ledger_state(
    card: dict[str, Any],
    *,
    cards_by_key: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, str]:
    status = str(card.get("status") or "info")
    key = str(card.get("key") or "")
    details = card.get("details") if isinstance(card.get("details"), dict) else {}
    current_window = (
        details.get("current_runtime_window")
        if isinstance(details.get("current_runtime_window"), dict)
        else {}
    )
    diagnostics = details.get("diagnostics") if isinstance(details.get("diagnostics"), dict) else {}
    historical_only = bool(current_window.get("historical_legacy_issues")) and not any(
        bool(diagnostics.get(key))
        for key in (
            "current_weak_executed",
            "current_no_high_quality_entries",
            "current_fast_loss_cluster",
            "current_ml_not_effective",
            "shadow_only_executed",
            "executed_without_order",
        )
    )
    if status == "ok":
        return "fixed", "已修复 / 当前验证通过"
    if historical_only:
        return "observing", "历史遗留 / 当前未复现"
    if (
        key == "model_training"
        and status == "warning"
        and bool(details.get("observing"))
        and not bool(details.get("hard_failure"))
    ):
        return "observing", "观察项 / 可选增强或学习模式"
    if key == "okx_reconciliation" and status == "warning" and bool(details.get("timeout")):
        return "observing", "观察项 / 对账巡检超时"
    if key == "trade_loop" and status == "warning" and bool(details.get("cold_start")):
        return "observing", "观察项 / 服务冷启动"
    if key == "trade_loop" and status == "warning" and bool(details.get("market_analysis_paused")):
        return "observing", "观察项 / 新币种分析暂停"
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
            "strategy_decision",
            "策略决策质量",
            "策略层",
            cards_by_key,
            ["strategy_quality", "trade_execution_contract"],
            impact="影响是否开仓、仓位大小、重复亏损复开和快进快出。",
            upstream=["market_data", "model_training", "position_sync"],
            downstream=["risk_guard", "okx_execution"],
            checks=["负净收益候选", "零净收益候选", "快亏平样本", "拦截原因"],
        ),
        _node_from_cards(
            "strategy_closed_loop",
            "策略闭环有效性",
            "策略层",
            cards_by_key,
            ["strategy_closed_loop"],
            impact="把数据、模型、决策、仓位、执行、平仓、训练反馈串起来，判断问题卡在哪一层。",
            upstream=["market_data", "model_training", "position_sync"],
            downstream=["risk_guard", "okx_execution", "training_data"],
            checks=["证据档位分布", "弱证据执行", "ML可用率", "快亏平", "收益样本"],
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
            ["okx_reconciliation", "position_price_integrity"],
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
            ["position_price_integrity", "okx_reconciliation"],
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
        "critical": sum(1 for node in nodes if node.get("status") == "critical"),
        "warning": sum(1 for node in nodes if node.get("status") == "warning"),
        "ok": sum(1 for node in nodes if node.get("status") == "ok"),
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
    audit_specs = [
        ("trade_loop", _trade_loop_audit),
        ("okx_reconciliation", _okx_reconciliation_audit),
        ("position_price_integrity", _position_price_integrity_audit),
        ("market_data", _market_data_audit),
        ("strategy_quality", _strategy_quality_audit),
        ("strategy_closed_loop", _strategy_closed_loop_audit),
        ("model_training", _model_training_audit),
        ("model_expert_health", _model_expert_health_audit),
        ("model_expert_competition", _model_expert_competition_audit),
        ("model_dynamic_routing", _model_dynamic_routing_audit),
        ("crypto_feature_coverage", _crypto_feature_coverage_audit),
        ("shadow_missed_opportunity", _shadow_missed_opportunity_audit),
        ("trade_execution_contract", _trade_execution_contract_audit),
        (
            "strategy_gate_contract",
            lambda: asyncio.to_thread(_strategy_gate_contract_audit),
        ),
        ("visible_text_encoding", lambda: asyncio.to_thread(_source_visible_text_audit)),
        ("runtime_text_integrity", _runtime_text_integrity_audit),
    ]
    results = await asyncio.gather(
        *(
            _audit_maybe_async(factory)
            for _key, factory in audit_specs
            if _key in PRIORITY_AUDIT_KEYS
        ),
        return_exceptions=True,
    )
    remaining_results = await asyncio.gather(
        *(
            _audit_maybe_async(factory)
            for _key, factory in audit_specs
            if _key not in PRIORITY_AUDIT_KEYS
        ),
        return_exceptions=True,
    )
    result_by_key = {
        key: result
        for key, result in zip(
            [key for key, _factory in audit_specs if key in PRIORITY_AUDIT_KEYS],
            results,
            strict=True,
        )
    }
    result_by_key.update(
        {
            key: result
            for key, result in zip(
                [key for key, _factory in audit_specs if key not in PRIORITY_AUDIT_KEYS],
                remaining_results,
                strict=True,
            )
        }
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


@router.get("/shadow-missed-opportunity/status")
async def shadow_missed_opportunity_status(
    hours: int = SHADOW_MISSED_OPPORTUNITY_AUDIT_HOURS,
    limit: int = SHADOW_MISSED_OPPORTUNITY_AUDIT_LIMIT,
) -> dict[str, Any]:
    report = await ShadowMissedOpportunityClosedLoopService().report(hours=hours, limit=limit)
    return sanitize_payload(_safe_shadow_missed_opportunity_report(report))


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
