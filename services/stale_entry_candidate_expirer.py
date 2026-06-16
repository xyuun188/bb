"""Expire stale entry candidates and pending entry submissions."""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import func, or_, select

from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order
from services.decision_freshness import ENTRY_DECISION_MAX_AGE_SECONDS
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from web_dashboard.api.text_sanitize import sanitize_text

logger = structlog.get_logger(__name__)

ENTRY_PENDING_EXECUTION_MAX_SECONDS = 45.0
PENDING_EXECUTION_PREFIXES = (
    "正在提交 OKX",
    "本轮执行仍在处理中",
    "Execution still pending this round",
)

FloatParser = Callable[[Any, float], float]
OrderCountProvider = Callable[[int], Awaitable[int]]
FlushCallback = Callable[[], Awaitable[None]]


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
)
PENDING_EXECUTION_PATTERNS = (
    *_legacy_sql_like_patterns("正在提交 OKX%"),
    *_legacy_sql_like_patterns("本轮执行仍在处理中%"),
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
        """Load and expire stale waiting/pending entry decisions from the DB."""

        now = datetime.utcnow()
        waiting_cutoff = now - timedelta(seconds=ENTRY_DECISION_MAX_AGE_SECONDS)
        pending_cutoff = now - timedelta(seconds=ENTRY_PENDING_EXECUTION_MAX_SECONDS)
        try:
            async with get_session_ctx() as session:
                waiting_rows = await self._load_rows(
                    session,
                    cutoff=waiting_cutoff,
                    reason_patterns=WAITING_ENTRY_PATTERNS,
                )
                pending_rows = await self._load_rows(
                    session,
                    cutoff=pending_cutoff,
                    reason_patterns=PENDING_EXECUTION_PATTERNS,
                )

                async def order_count_provider(decision_id: int) -> int:
                    count = (
                        await session.execute(
                            select(func.count(Order.id)).where(Order.decision_id == decision_id)
                        )
                    ).scalar() or 0
                    return int(count)

                expired = await self.expire_rows(
                    waiting_rows,
                    pending_rows,
                    order_count_provider=order_count_provider,
                    flush_callback=session.flush,
                )
                if expired:
                    logger.info(
                        "expired stale entry candidates",
                        waiting=len(waiting_rows),
                        pending=len(pending_rows),
                    )
                return expired
        except Exception as exc:
            logger.warning("failed to expire stale entry candidates", error=safe_error_text(exc))
            return 0

    async def expire_rows(
        self,
        waiting_rows: list[Any],
        pending_rows: list[Any],
        *,
        order_count_provider: OrderCountProvider,
        flush_callback: FlushCallback | None = None,
    ) -> int:
        """Apply stale-entry expiration rules to already-loaded decision rows."""

        expired = 0
        for row in waiting_rows:
            reason = self._waiting_expiration_reason(row)
            self._apply_reason(row, reason)
            expired += 1

        for row in pending_rows:
            order_count = await order_count_provider(int(row.id))
            if order_count > 0:
                reason = (
                    "本地订单记录已生成，但成交或撤单状态还没有最终确认。"
                    "请以执行记录中的最新订单状态为准。"
                )
            else:
                reason = pending_execution_failed_reason(row.symbol, row.action)
            self._apply_reason(row, reason)
            expired += 1

        if expired and flush_callback is not None:
            await flush_callback()
        return expired

    async def _load_rows(
        self,
        session: Any,
        *,
        cutoff: datetime,
        reason_patterns: tuple[str, ...],
    ) -> list[AIDecision]:
        stmt = select(AIDecision).where(
            AIDecision.was_executed.is_(False),
            AIDecision.action.in_(["long", "short", "open_long", "open_short"]),
            AIDecision.created_at <= cutoff,
            or_(*[AIDecision.execution_reason.like(pattern) for pattern in reason_patterns]),
        )
        return list((await session.execute(stmt)).scalars().all())

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

    def _apply_reason(self, row: Any, reason: str) -> None:
        clean_reason = str(sanitize_text(reason) or reason)
        raw = _safe_raw_response(row.raw_llm_response)
        opportunity = raw.get("opportunity_score")
        if not isinstance(opportunity, dict):
            opportunity = {}
        opportunity["selected_for_execution"] = False
        opportunity["selection_reason"] = clean_reason
        raw["opportunity_score"] = opportunity
        row.raw_llm_response = raw
        row.execution_reason = clean_reason


def _safe_raw_response(value: Any) -> dict[str, Any]:
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    return raw if isinstance(raw, dict) else {}
