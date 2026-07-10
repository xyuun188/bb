"""Expire stale entry candidates and pending entry submissions."""

from __future__ import annotations

import contextlib
import copy
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import structlog
from sqlalchemy import ARRAY, Text, bindparam, cast, func, literal, or_, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm.attributes import flag_modified

from config.settings import settings
from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order
from services.decision_freshness import ENTRY_DECISION_MAX_AGE_SECONDS
from services.decision_state import (
    DecisionStage,
    DecisionStageStatus,
    append_decision_stage,
    decision_state_from_raw,
    is_decision_terminal_state,
)
from services.entry_direction_metrics import entry_side_from_action, selected_side_evidence
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from web_dashboard.api.text_sanitize import sanitize_text

logger = structlog.get_logger(__name__)

ENTRY_PENDING_EXECUTION_MAX_SECONDS = 45.0
STALE_ENTRY_MAINTENANCE_BATCH_LIMIT = 250
STALE_ENTRY_MAINTENANCE_LOOKBACK = timedelta(hours=24)
_STALE_ENTRY_RAW_COLUMN_PREFIX = "stale_entry_raw__"
PENDING_EXECUTION_PREFIXES = (
    "正在提交 OKX",
    "本轮执行仍在处理中",
    "Execution still pending this round",
)

FloatParser = Callable[[Any, float], float]
OrderCountProvider = Callable[[int], Awaitable[int]]
FlushCallback = Callable[[], Awaitable[None]]


def _stale_entry_scan_columns() -> tuple[Any, ...]:
    raw = AIDecision.raw_llm_response
    return (
        AIDecision.id,
        AIDecision.symbol,
        AIDecision.action,
        AIDecision.execution_reason,
        AIDecision.created_at,
        AIDecision.updated_at,
        raw["opportunity_score"]["score"].label(f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}score"),
        raw["opportunity_score"]["min_score_required"].label(
            f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}min_score_required"
        ),
        raw["opportunity_score"]["expected_net_return_pct"].label(
            f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}expected_net_return_pct"
        ),
        raw["entry_candidate_evidence"]["long"]["expected_net_return_pct"].label(
            f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}long_expected_net_return_pct"
        ),
        raw["entry_candidate_evidence"]["short"]["expected_net_return_pct"].label(
            f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}short_expected_net_return_pct"
        ),
        raw["entry_candidate_evidence"]["side"].label(f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}side"),
        raw["entry_candidate_evidence"]["expected_net_return_pct"].label(
            f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}direct_expected_net_return_pct"
        ),
        raw["decision_state_machine"]["stages"].label(
            f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}stages"
        ),
    )


def _stale_entry_row_from_mapping(mapping: Any) -> SimpleNamespace:
    opportunity = {
        key: mapping.get(f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}{key}")
        for key in ("score", "min_score_required", "expected_net_return_pct")
        if mapping.get(f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}{key}") is not None
    }
    evidence: dict[str, Any] = {}
    for side in ("long", "short"):
        expected = mapping.get(f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}{side}_expected_net_return_pct")
        if expected is not None:
            evidence[side] = {"expected_net_return_pct": expected}
    direct_side = mapping.get(f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}side")
    if direct_side is not None:
        evidence["side"] = direct_side
    direct_expected = mapping.get(f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}direct_expected_net_return_pct")
    if direct_expected is not None:
        evidence["expected_net_return_pct"] = direct_expected
    stages = mapping.get(f"{_STALE_ENTRY_RAW_COLUMN_PREFIX}stages")
    raw: dict[str, Any] = {}
    if opportunity:
        raw["opportunity_score"] = opportunity
    if evidence:
        raw["entry_candidate_evidence"] = evidence
    if isinstance(stages, list):
        raw["decision_state_machine"] = {"stages": stages}
    return SimpleNamespace(
        id=mapping.get("id"),
        symbol=mapping.get("symbol"),
        action=mapping.get("action"),
        execution_reason=mapping.get("execution_reason"),
        created_at=mapping.get("created_at"),
        updated_at=mapping.get("updated_at"),
        raw_llm_response=raw,
    )


def _legacy_sql_like_patterns(pattern: str) -> tuple[str, ...]:
    """Match clean Chinese rows and old rows damaged by a wrong decode step."""

    damaged = pattern.encode("utf-8").decode("gbk", errors="replace")
    damaged_question = damaged.replace("\ufffd", "?")
    if damaged_question == damaged:
        return (pattern, damaged)
    return (pattern, damaged, damaged_question)


WAITING_ENTRY_PATTERNS = (
    *_legacy_sql_like_patterns("已进入本轮开仓候选排序%"),
    *_legacy_sql_like_patterns("本轮还在分析或排队中%"),
    *_legacy_sql_like_patterns("候选排序超时后复核%"),
)
PENDING_EXECUTION_PATTERNS = (
    *_legacy_sql_like_patterns("正在提交 OKX%"),
    *_legacy_sql_like_patterns("本轮执行仍在处理中%"),
    *_legacy_sql_like_patterns("%开仓信号已经进入 OKX 下单流程，但在 45 秒内%"),
    "Execution still pending this round%",
)


def action_label(action: str | None) -> str:
    value = str(action or "")
    if value in {"long", "open_long"}:
        return "做多"
    if value in {"short", "open_short"}:
        return "做空"
    if value == "close_long":
        return "平多"
    if value == "close_short":
        return "平空"
    return value or "交易"


def is_pending_execution_reason(reason: str | None) -> bool:
    text = str(reason or "")
    cleaned = str(sanitize_text(text) or text)
    return not cleaned or any(
        cleaned.startswith(prefix) or text.startswith(prefix)
        for prefix in PENDING_EXECUTION_PREFIXES
    )


def pending_execution_failed_reason(symbol: str, action: str | None = None) -> str:
    prefix = f"{action} " if action else ""
    return (
        f"{symbol} {prefix}开仓信号已经进入 OKX 下单流程，但在 "
        f"{ENTRY_PENDING_EXECUTION_MAX_SECONDS:.0f} 秒内没有生成本地订单记录，也没有拿到 OKX 成功或失败回报。"
        "系统已按下单流程异常处理，本次旧信号不再继续等待，下一轮会用最新行情重新分析。"
    )


@dataclass(slots=True)
class StaleEntryCandidateExpirer:
    """Clear stale entry candidates that can otherwise suppress fresh entries."""

    float_parser: FloatParser

    async def expire(self) -> int:
        """Expire one bounded batch without retaining a session during evaluation."""

        now = datetime.utcnow()
        waiting_cutoff = now - timedelta(seconds=ENTRY_DECISION_MAX_AGE_SECONDS)
        maintenance_since = now - STALE_ENTRY_MAINTENANCE_LOOKBACK
        try:
            async with get_session_ctx() as session:
                waiting_rows = await self._load_rows(
                    session,
                    since=maintenance_since,
                    cutoff=waiting_cutoff,
                    reason_patterns=WAITING_ENTRY_PATTERNS,
                )
                pending_rows = await self._load_rows(
                    session,
                    since=maintenance_since,
                    cutoff=None,
                    reason_patterns=PENDING_EXECUTION_PATTERNS,
                )
                open_state_rows = await self._load_stale_open_state_rows(
                    session,
                    since=maintenance_since,
                    cutoff=waiting_cutoff,
                )
                waiting_rows, pending_rows = _merge_open_state_repairs(
                    waiting_rows,
                    pending_rows,
                    open_state_rows,
                )
                pending_ids = [
                    int(getattr(row, "id", 0) or 0)
                    for row in pending_rows
                    if int(getattr(row, "id", 0) or 0) > 0
                ]
                order_counts = await self._load_order_counts(session, pending_ids)

            # These rows are detached after the read transaction.  Deciding which
            # stale candidates need a terminal state is CPU-only and must not keep
            # a database connection while it iterates a maintenance batch.
            candidate_rows = {
                int(getattr(row, "id", 0) or 0): row
                for row in [*waiting_rows, *pending_rows]
                if int(getattr(row, "id", 0) or 0) > 0
            }
            before = {
                row_id: (
                    str(getattr(row, "execution_reason", "") or ""),
                    copy.deepcopy(_safe_raw_response(getattr(row, "raw_llm_response", None))),
                )
                for row_id, row in candidate_rows.items()
            }

            async def order_count_provider(decision_id: int) -> int:
                return int(order_counts.get(int(decision_id or 0), 0))

            expired = await self.expire_rows(
                waiting_rows,
                pending_rows,
                now=now,
                order_count_provider=order_count_provider,
            )
            if not expired:
                return 0

            updates = {
                row_id: row
                for row_id, row in candidate_rows.items()
                if (
                    str(getattr(row, "execution_reason", "") or ""),
                    _safe_raw_response(getattr(row, "raw_llm_response", None)),
                )
                != before[row_id]
            }
            if not updates:
                return 0

            async with get_session_ctx() as session:
                if "postgresql" in settings.database_url:
                    await self._persist_postgres_updates(session, updates)
                else:
                    await self._persist_compatibility_updates(session, updates)
                await session.flush()

            logger.info(
                "expired stale entry candidates",
                expired=len(updates),
                waiting=len(waiting_rows),
                pending=len(pending_rows),
            )
            return len(updates)
        except Exception as exc:
            logger.warning("failed to expire stale entry candidates", error=safe_error_text(exc))
            return 0

    async def _persist_postgres_updates(self, session: Any, updates: dict[int, Any]) -> None:
        """Patch terminal state in PostgreSQL without fetching large decision JSON."""

        base_raw = func.coalesce(
            cast(AIDecision.raw_llm_response, JSONB),
            cast({}, JSONB),
        )
        top_patch = cast(bindparam("raw_patch", type_=JSONB), JSONB)
        opportunity_patch = cast(bindparam("opportunity_patch", type_=JSONB), JSONB)
        state_patch = cast(bindparam("state_patch", type_=JSONB), JSONB)
        opportunity = func.coalesce(
            base_raw.op("->")("opportunity_score"),
            cast({}, JSONB),
        ).op("||")(opportunity_patch)
        state_machine = func.coalesce(
            base_raw.op("->")("decision_state_machine"),
            cast({}, JSONB),
        ).op("||")(state_patch)
        merged_raw = func.jsonb_set(
            func.jsonb_set(
                base_raw.op("||")(top_patch),
                literal(["opportunity_score"], type_=ARRAY(Text)),
                opportunity,
                True,
            ),
            literal(["decision_state_machine"], type_=ARRAY(Text)),
            state_machine,
            True,
        )
        stmt = (
            update(AIDecision.__table__)
            .where(AIDecision.id == bindparam("target_decision_id"))
            .values(
                execution_reason=bindparam("execution_reason"),
                raw_llm_response=merged_raw,
            )
        )
        payloads = []
        for row_id, updated in updates.items():
            top, opportunity, state_machine = _stale_entry_update_patch(
                updated.raw_llm_response
            )
            payloads.append(
                {
                    "target_decision_id": int(row_id),
                    "execution_reason": updated.execution_reason,
                    "raw_patch": top,
                    "opportunity_patch": opportunity,
                    "state_patch": state_machine,
                }
            )
        if payloads:
            await session.execute(stmt, payloads)

    async def _persist_compatibility_updates(self, session: Any, updates: dict[int, Any]) -> None:
        """Keep SQLite and external adapters correct without PostgreSQL JSON operators."""

        result = await session.execute(select(AIDecision).where(AIDecision.id.in_(list(updates))))
        persisted_by_id = {
            int(getattr(row, "id", 0) or 0): row for row in result.scalars().all()
        }
        for row_id, updated in updates.items():
            persisted = persisted_by_id.get(row_id)
            if persisted is None:
                continue
            persisted.execution_reason = updated.execution_reason
            persisted.raw_llm_response = _merge_expired_raw_response(
                persisted.raw_llm_response,
                updated.raw_llm_response,
            )
            _mark_raw_response_modified(persisted)

    async def _load_order_counts(self, session: Any, decision_ids: list[int]) -> dict[int, int]:
        ids = sorted({int(decision_id or 0) for decision_id in decision_ids if decision_id})
        if not ids:
            return {}
        result = await session.execute(
            select(Order.decision_id, func.count(Order.id))
            .where(Order.decision_id.in_(ids))
            .group_by(Order.decision_id)
        )
        return {
            int(decision_id): int(count or 0)
            for decision_id, count in result.all()
            if decision_id is not None
        }

    async def expire_rows(
        self,
        waiting_rows: list[Any],
        pending_rows: list[Any],
        *,
        now: datetime | None = None,
        order_count_provider: OrderCountProvider,
        flush_callback: FlushCallback | None = None,
    ) -> int:
        """Apply stale-entry expiration rules to already-loaded decision rows."""

        current_time = now or datetime.utcnow()
        expired = 0
        for row in waiting_rows:
            if not _needs_terminal_state_repair(row):
                continue
            reason = self._waiting_expiration_reason(row)
            self._apply_reason(
                row,
                reason,
                stage=DecisionStage.RISK_CHECK,
                status=DecisionStageStatus.SKIPPED,
                skip_kind="stale_entry_candidate_expired",
                terminal=True,
            )
            expired += 1

        for row in pending_rows:
            if not pending_execution_is_stale(row, current_time):
                continue
            if not _needs_terminal_state_repair(row):
                continue
            order_count = await order_count_provider(int(row.id))
            if order_count > 0:
                reason = (
                    "本地订单记录已生成，但成交或撤单状态还没有最终确认。"
                    "请以执行记录中的最新订单状态为准。"
                )
                self._apply_reason(
                    row,
                    reason,
                    stage=DecisionStage.EXCHANGE_CONFIRM,
                    status=DecisionStageStatus.PENDING,
                    skip_kind="pending_exchange_order_status",
                    terminal=False,
                    selected_for_execution=True,
                )
            else:
                reason = pending_execution_failed_reason(row.symbol, row.action)
                self._apply_reason(
                    row,
                    reason,
                    stage=DecisionStage.EXCHANGE_SUBMIT,
                    status=DecisionStageStatus.FAILED,
                    skip_kind="pending_entry_execution_expired",
                    terminal=True,
                )
                row.raw_llm_response = append_decision_stage(
                    _safe_raw_response(row.raw_llm_response),
                    DecisionStage.LOCAL_SYNC,
                    DecisionStageStatus.SKIPPED,
                    "没有成交结果，本地持仓未改动。",
                    {
                        "skip_kind": "pending_entry_execution_expired",
                        "fallback_final_state": True,
                        "error_type": "missing_execution_result",
                    },
                )
            expired += 1

        if expired and flush_callback is not None:
            await flush_callback()
        return expired

    async def _load_rows(
        self,
        session: Any,
        *,
        since: datetime,
        cutoff: datetime | None,
        reason_patterns: tuple[str, ...],
    ) -> list[Any]:
        stmt = select(*_stale_entry_scan_columns()).where(
            AIDecision.was_executed.is_(False),
            AIDecision.action.in_(["long", "short", "open_long", "open_short"]),
            AIDecision.created_at >= since,
            or_(*[AIDecision.execution_reason.like(pattern) for pattern in reason_patterns]),
        )
        if cutoff is not None:
            stmt = stmt.where(AIDecision.created_at <= cutoff)
        stmt = stmt.order_by(AIDecision.created_at.asc(), AIDecision.id.asc()).limit(
            STALE_ENTRY_MAINTENANCE_BATCH_LIMIT
        )
        return [
            _stale_entry_row_from_mapping(row) for row in (await session.execute(stmt)).mappings().all()
        ]

    async def _load_stale_open_state_rows(
        self,
        session: Any,
        *,
        since: datetime,
        cutoff: datetime,
    ) -> list[Any]:
        stmt = (
            select(*_stale_entry_scan_columns())
            .where(
                AIDecision.was_executed.is_(False),
                AIDecision.action.in_(["long", "short", "open_long", "open_short"]),
                AIDecision.created_at >= since,
                AIDecision.created_at <= cutoff,
            )
            .order_by(AIDecision.created_at.asc(), AIDecision.id.asc())
            .limit(STALE_ENTRY_MAINTENANCE_BATCH_LIMIT)
        )
        rows = [
            _stale_entry_row_from_mapping(row)
            for row in (await session.execute(stmt)).mappings().all()
        ]
        return [row for row in rows if _needs_terminal_state_repair(row)]

    def _waiting_expiration_reason(self, row: Any) -> str:
        raw = _safe_raw_response(row.raw_llm_response)
        opportunity = raw.get("opportunity_score")
        if not isinstance(opportunity, dict):
            opportunity = {}
        score = self.float_parser(opportunity.get("score"), float("nan"))
        min_score = self.float_parser(
            opportunity.get("min_score_required"),
            MIN_ENTRY_OPPORTUNITY_SCORE,
        )
        expected_net = self.float_parser(
            opportunity.get("expected_net_return_pct"),
            0.0,
        )
        side = entry_side_from_action(str(row.action or ""))
        side_evidence = selected_side_evidence(raw, side)
        if side_evidence:
            expected_net = self.float_parser(
                side_evidence.get("expected_net_return_pct"),
                expected_net,
            )
        if expected_net <= 0:
            return (
                f"候选排序超时后复核：{row.symbol} 本次{action_label(row.action)}"
                f"预期净收益 {expected_net:.4f}% 不为正，旧信号不再执行，下一轮重新分析。"
            )
        if math.isfinite(score) and score <= min_score:
            return (
                f"候选排序超时后复核：{row.symbol} 本次{action_label(row.action)}"
                f"机会评分 {score:.4f} 低于执行门槛 {min_score:.2f}，旧信号不再执行，下一轮重新分析。"
            )
        return (
            f"候选排序等待超过 {ENTRY_DECISION_MAX_AGE_SECONDS:.0f} 秒，"
            "行情快照已经过期。为避免追单，本次旧信号不再执行，下一轮重新分析。"
        )

    def _apply_reason(
        self,
        row: Any,
        reason: str,
        *,
        stage: str,
        status: str,
        skip_kind: str,
        terminal: bool,
        selected_for_execution: bool = False,
    ) -> None:
        clean_reason = str(sanitize_text(reason) or reason)
        raw = _safe_raw_response(row.raw_llm_response)
        opportunity = raw.get("opportunity_score")
        if not isinstance(opportunity, dict):
            opportunity = {}
        opportunity["selected_for_execution"] = bool(selected_for_execution)
        opportunity["selection_reason"] = clean_reason
        opportunity["execution_final_state"] = status
        opportunity["execution_final_blocker"] = skip_kind
        raw["opportunity_score"] = opportunity
        raw["skip_kind"] = skip_kind
        raw["reason"] = clean_reason
        raw["execution_skipped"] = bool(terminal)
        raw["stale_entry_candidate_expired"] = bool(terminal)
        raw = append_decision_stage(
            raw,
            stage,
            status,
            clean_reason,
            {
                "skip_kind": skip_kind,
                "fallback_final_state": bool(terminal),
                "selected_for_execution": bool(selected_for_execution),
            },
        )
        row.raw_llm_response = raw
        _mark_raw_response_modified(row)
        row.execution_reason = clean_reason


def _safe_raw_response(value: Any) -> dict[str, Any]:
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    return raw if isinstance(raw, dict) else {}


def _merge_expired_raw_response(persisted: Any, updated: Any) -> dict[str, Any]:
    """Apply scan-time terminal-state changes without dropping the audit payload."""

    merged = dict(_safe_raw_response(persisted))
    top, opportunity, state_machine = _stale_entry_update_patch(updated)
    merged.update(top)
    if opportunity:
        existing_opportunity = merged.get("opportunity_score")
        if not isinstance(existing_opportunity, dict):
            existing_opportunity = {}
        merged["opportunity_score"] = {
            **existing_opportunity,
            **opportunity,
        }
    if state_machine:
        existing_state_machine = merged.get("decision_state_machine")
        if not isinstance(existing_state_machine, dict):
            existing_state_machine = {}
        merged["decision_state_machine"] = {
            **existing_state_machine,
            **state_machine,
        }
    return merged


def _stale_entry_update_patch(updated: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return only the terminal-state keys changed by stale-entry maintenance."""

    raw = _safe_raw_response(updated)
    top = {
        key: raw[key]
        for key in ("skip_kind", "reason", "execution_skipped", "stale_entry_candidate_expired")
        if key in raw
    }
    source_opportunity = raw.get("opportunity_score")
    opportunity = {
        key: source_opportunity[key]
        for key in (
            "selected_for_execution",
            "selection_reason",
            "execution_final_state",
            "execution_final_blocker",
        )
        if isinstance(source_opportunity, dict) and key in source_opportunity
    }
    state_machine = raw.get("decision_state_machine")
    return top, opportunity, state_machine if isinstance(state_machine, dict) else {}


def _needs_terminal_state_repair(row: Any) -> bool:
    raw = _safe_raw_response(getattr(row, "raw_llm_response", None))
    summary = decision_state_from_raw(raw).get("summary")
    if not isinstance(summary, dict):
        return True
    final_status = str(summary.get("final_status") or "")
    final_stage = str(summary.get("final_stage") or "")
    return not is_decision_terminal_state(final_stage, final_status)


def _merge_open_state_repairs(
    waiting_rows: list[Any],
    pending_rows: list[Any],
    open_state_rows: list[Any],
) -> tuple[list[Any], list[Any]]:
    waiting_by_id = {int(getattr(row, "id", 0) or 0): row for row in waiting_rows}
    pending_by_id = {int(getattr(row, "id", 0) or 0): row for row in pending_rows}
    for row in open_state_rows:
        row_id = int(getattr(row, "id", 0) or 0)
        if not row_id or row_id in waiting_by_id or row_id in pending_by_id:
            continue
        if _is_pending_submit_row(row):
            pending_by_id[row_id] = row
        else:
            waiting_by_id[row_id] = row
    return list(waiting_by_id.values()), list(pending_by_id.values())


def _is_pending_submit_row(row: Any) -> bool:
    reason = str(getattr(row, "execution_reason", "") or "").strip()
    if reason and is_pending_execution_reason(reason):
        return True
    return _pending_execution_started_at(row) is not None


def _mark_raw_response_modified(row: Any) -> None:
    with contextlib.suppress(Exception):
        flag_modified(row, "raw_llm_response")


def pending_execution_is_stale(row: Any, now: datetime | None = None) -> bool:
    current_time = _as_naive_utc(now or datetime.utcnow())
    pending_started_at = _pending_execution_started_at(row)
    if pending_started_at is None:
        pending_started_at = _as_naive_utc(getattr(row, "updated_at", None))
    if pending_started_at is None:
        pending_started_at = _as_naive_utc(getattr(row, "created_at", None))
    if pending_started_at is None:
        return True
    elapsed = (current_time - pending_started_at).total_seconds()
    return elapsed >= ENTRY_PENDING_EXECUTION_MAX_SECONDS


def _pending_execution_started_at(row: Any) -> datetime | None:
    raw = _safe_raw_response(getattr(row, "raw_llm_response", None))
    machine = raw.get("decision_state_machine")
    if not isinstance(machine, dict):
        return None
    stages = machine.get("stages")
    if not isinstance(stages, list):
        return None
    for event in reversed(stages):
        if not isinstance(event, dict):
            continue
        if str(event.get("stage") or "") != DecisionStage.EXCHANGE_SUBMIT:
            continue
        if str(event.get("status") or "") != DecisionStageStatus.PENDING:
            continue
        started_at = _as_naive_utc(event.get("at"))
        if started_at is not None:
            return started_at
    return None


def _as_naive_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return value
