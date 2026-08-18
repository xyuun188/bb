"""Persistence boundary for AI decision records.

TradingService should orchestrate trading flow, not own SQLAlchemy write
details for every decision state transition. This service keeps decision
logging, raw payload updates, execution reasons, duplicate-order checks, and
outcome marks behind one explicit boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import func, select

from ai_brain.base_model import DecisionOutput
from core.reason_codes import ReasonCode
from core.safe_output import safe_error_text
from db.repositories.decision_repo import DecisionRepository
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order
from services.decision_state import (
    DecisionStage,
    DecisionStageStatus,
    append_decision_stage,
    decision_state_from_raw,
    is_decision_terminal_state,
)
from services.text_integrity import looks_like_mojibake, sanitize_runtime_text

logger = structlog.get_logger(__name__)

SessionContextFactory = Callable[[], AbstractAsyncContextManager[Any]]
DecisionRepoFactory = Callable[[Any], DecisionRepository]
SymbolNormalizer = Callable[[str | None], str]
ReasonRecoverer = Callable[[AIDecision | None, Any], str | None]


class DecisionPersistenceService:
    """Persist decision records and state transitions."""

    def __init__(
        self,
        *,
        normalize_symbol: SymbolNormalizer,
        session_context_factory: SessionContextFactory = get_session_ctx,
        decision_repo_factory: DecisionRepoFactory = DecisionRepository,
    ) -> None:
        self._normalize_symbol = normalize_symbol
        self._session_context_factory = session_context_factory
        self._decision_repo_factory = decision_repo_factory

    async def log_decision(self, decision: DecisionOutput, is_paper: bool) -> int | None:
        """Insert one AI decision row and attach the initial AI-analysis stage."""

        try:
            async with self._session_context_factory() as session:
                repo = self._decision_repo_factory(session)
                raw_response = (
                    decision.raw_response if isinstance(decision.raw_response, dict) else {}
                )
                analysis_type = self.analysis_type(decision, raw_response)
                quality = raw_response.get("analysis_quality_contract")
                quality = quality if isinstance(quality, dict) else {}
                if not quality:
                    quality = self._runtime_failure_quality(raw_response)
                    if quality:
                        raw_response["analysis_quality_contract"] = quality
                analysis_complete = bool(quality.get("analysis_complete")) if quality else True
                if analysis_complete:
                    analysis_status = DecisionStageStatus.COMPLETED
                    analysis_reason = (
                        "AI 已完成分析、结构校验和交叉验证。"
                        if quality
                        else "AI 已完成分析并生成裁决。"
                    )
                else:
                    status_counts = quality.get("status_counts")
                    status_counts = status_counts if isinstance(status_counts, dict) else {}
                    only_skipped = bool(status_counts.get("skipped")) and not any(
                        status_counts.get(name)
                        for name in ("timeout", "parse_failed", "empty", "unavailable")
                    )
                    analysis_status = (
                        DecisionStageStatus.SKIPPED if only_skipped else DecisionStageStatus.FAILED
                    )
                    analysis_reason = str(
                        quality.get("reason") or "AI 分析证据不完整，本轮不可用于开仓。"
                    )
                raw_response = append_decision_stage(
                    raw_response,
                    DecisionStage.AI_ANALYSIS,
                    analysis_status,
                    analysis_reason,
                    data={
                        "analysis_type": analysis_type,
                        "action": decision.action.value,
                        "confidence": float(decision.confidence or 0.0),
                        "analysis_complete": analysis_complete,
                        "decision_eligible": bool(quality.get("decision_eligible", analysis_complete)),
                        "reason_code": quality.get("reason_code"),
                    },
                    reason_code=self._analysis_reason_code(quality),
                )
                decision.raw_response = raw_response
                display_symbol = self._normalize_symbol(decision.symbol) or decision.symbol
                idempotency_key = str(raw_response.get("analysis_idempotency_key") or "").strip()
                if idempotency_key:
                    existing = await repo.get_by_analysis_idempotency_key(idempotency_key)
                    if existing is not None:
                        decision.raw_response = {
                            **raw_response,
                            "duplicate_analysis": {
                                "skipped": True,
                                "existing_decision_id": int(existing.id),
                                "analysis_idempotency_key": idempotency_key,
                            },
                        }
                        logger.warning(
                            "duplicate analysis persistence skipped",
                            existing_decision_id=int(existing.id),
                            symbol=display_symbol,
                            analysis_type=analysis_type,
                        )
                        return None
                record = await repo.log_decision(
                    {
                        "model_name": decision.model_name,
                        "symbol": display_symbol,
                        "action": decision.action.value,
                        "confidence": decision.confidence,
                        "reasoning": sanitize_runtime_text(decision.reasoning),
                        "position_size_pct": decision.position_size_pct,
                        "suggested_leverage": decision.suggested_leverage,
                        "stop_loss_pct": decision.stop_loss_pct,
                        "take_profit_pct": decision.take_profit_pct,
                        "feature_snapshot": self.json_safe_payload(decision.feature_snapshot),
                        "raw_llm_response": self.json_safe_payload(decision.raw_response),
                        "analysis_type": analysis_type,
                        "analysis_idempotency_key": idempotency_key or None,
                        "is_paper": is_paper,
                    }
                )
                return record.id
        except Exception as exc:
            logger.error("failed to log decision", error=safe_error_text(exc))
            return None

    @staticmethod
    def _runtime_failure_quality(raw_response: dict[str, Any]) -> dict[str, Any]:
        """Keep explicit runtime failures from becoming completed legacy rows."""

        timeout = raw_response.get("market_model_timeout")
        failures = raw_response.get("expert_failures")
        failure_rows = failures if isinstance(failures, list) else []
        if not isinstance(timeout, dict) and not failure_rows:
            return {}

        attempted = raw_response.get("attempted_experts")
        attempted_names = attempted if isinstance(attempted, list) else []
        expected_count = max(len(attempted_names), len(failure_rows), 1)
        timed_out = isinstance(timeout, dict)
        reason = str(
            (timeout or {}).get("reason")
            or next(
                (
                    row.get("reason")
                    for row in failure_rows
                    if isinstance(row, dict) and row.get("reason")
                ),
                "",
            )
            or "AI runtime failure prevented a complete analysis."
        )
        return {
            "version": "2026-08-17.analysis-quality.v1",
            "expected_expert_count": expected_count,
            "attempted_expert_count": len(attempted_names),
            "returned_expert_count": 0,
            "successful_expert_count": 0,
            "status_counts": {
                "completed": 0,
                "timeout": expected_count if timed_out else 0,
                "parse_failed": 0,
                "empty": 0,
                "unavailable": 0 if timed_out else expected_count,
                "skipped": 0,
            },
            "experts": [],
            "expert_complete": False,
            "cross_validation_complete": False,
            "analysis_complete": False,
            "decision_eligible": False,
            "result": "unclear",
            "reason_code": "model_timeout" if timed_out else "model_unavailable",
            "reason": reason,
            "recovered_at_persistence_boundary": True,
        }

    @staticmethod
    def _analysis_reason_code(quality: dict[str, Any]) -> str | None:
        if not quality or quality.get("analysis_complete") is True:
            return None
        status_counts = quality.get("status_counts")
        counts = status_counts if isinstance(status_counts, dict) else {}
        reason_code = str(quality.get("reason_code") or "").lower()
        if counts.get("timeout") or reason_code == "model_timeout":
            return ReasonCode.MODEL_TIMEOUT
        if counts.get("parse_failed") or counts.get("empty"):
            return ReasonCode.MODEL_INVALID_OUTPUT
        if counts.get("unavailable") or reason_code == "model_unavailable":
            return ReasonCode.MODEL_UNAVAILABLE
        return ReasonCode.SIGNAL_UNAVAILABLE

    @staticmethod
    def analysis_type(decision: DecisionOutput, raw_response: dict[str, Any]) -> str:
        """Infer dashboard/reporting analysis type from decision payload."""

        raw_analysis_type = str(raw_response.get("analysis_type") or "").lower()
        if raw_analysis_type in {"position", "position_review", "holding", "holdings"}:
            return "position"
        if raw_analysis_type in {
            "entry_candidate",
            "market_entry_candidate",
        }:
            return "entry_candidate"
        if raw_analysis_type in {"market", "market_scan", "symbol_scan"}:
            return "market"
        if (
            raw_response.get("position_review_policy")
            or raw_response.get("position_review")
            or decision.action.value in {"close_long", "close_short"}
        ):
            return "position"
        return "market"

    def json_safe_payload(self, value: Any) -> Any:
        """Return a JSON-column-safe copy of model/feature payloads."""

        if value is None or isinstance(value, (int, bool)):
            return value
        if isinstance(value, str):
            return sanitize_runtime_text(value)
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): self.json_safe_payload(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self.json_safe_payload(item) for item in value]
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return self.json_safe_payload(item())
            except Exception as exc:
                logger.debug("failed to unwrap scalar payload item", error=safe_error_text(exc))
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception as exc:
                logger.debug("failed to serialize isoformat payload", error=safe_error_text(exc))
        return sanitize_runtime_text(str(value))

    def record_stage(
        self,
        decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str | None,
        data: dict[str, Any] | None = None,
        *,
        duration_sec: float | None = None,
    ) -> dict[str, Any]:
        """Append one state-machine event to a decision raw payload."""

        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw = append_decision_stage(
            raw,
            stage,
            status,
            str(sanitize_runtime_text(reason) or "") if reason else "",
            data=self.json_safe_payload(data or {}) if data else None,
            duration_sec=duration_sec,
        )
        decision.raw_response = raw
        return raw

    async def record_and_persist_stage(
        self,
        *,
        decision_id: int | None,
        decision: DecisionOutput,
        stage: str,
        status: str,
        reason: str | None,
        data: dict[str, Any] | None = None,
        duration_sec: float | None = None,
    ) -> dict[str, Any]:
        raw = self.record_stage(decision, stage, status, reason, data, duration_sec=duration_sec)
        if decision_id is not None:
            await self.update_raw_response(decision_id, raw)
        return raw

    async def mark_executed(self, decision_id: int, execution_price: float) -> None:
        try:
            async with self._session_context_factory() as session:
                repo = self._decision_repo_factory(session)
                await repo.mark_executed(decision_id, execution_price)
        except Exception as exc:
            logger.error("failed to mark decision executed", error=safe_error_text(exc))

    async def mark_reason(
        self,
        decision_id: int,
        reason: str | None,
        *,
        reason_recoverer: ReasonRecoverer | None = None,
    ) -> None:
        try:
            async with self._session_context_factory() as session:
                row = await session.get(AIDecision, int(decision_id))
                sanitized_reason = sanitize_runtime_text(reason)
                if self.execution_reason_is_unusable(sanitized_reason):
                    recovered = reason_recoverer(row, reason) if reason_recoverer else None
                    if recovered:
                        sanitized_reason = sanitize_runtime_text(recovered)
                repo = self._decision_repo_factory(session)
                await repo.mark_execution_reason(decision_id, sanitized_reason)
        except Exception as exc:
            logger.error("failed to mark decision reason", error=safe_error_text(exc))

    @staticmethod
    def execution_reason_is_unusable(reason: Any) -> bool:
        text = str(reason or "").strip()
        if not text:
            return False
        if looks_like_mojibake(text):
            return True
        unusable_markers = (
            "原始说明已损坏",
            "无法准确还原",
        )
        return any(marker in text for marker in unusable_markers)

    async def mark_pending_execution(self, decision_id: int, reason: str) -> None:
        """Mark an entry as in-flight without letting final fallback overwrite it."""

        await self.mark_reason(decision_id, f"正在提交 OKX：{reason}")

    async def duplicate_order_reason(
        self,
        decision_id: int,
        decision: DecisionOutput,
    ) -> str | None:
        """Prevent the same decision row from submitting more than one OKX order."""

        try:
            async with self._session_context_factory() as session:
                order_count = (
                    await session.execute(
                        select(func.count(Order.id)).where(Order.decision_id == int(decision_id))
                    )
                ).scalar() or 0
        except Exception as exc:
            logger.warning(
                "failed duplicate decision order check",
                decision_id=decision_id,
                error=safe_error_text(exc),
            )
            return None
        order_count = int(order_count)
        if order_count <= 0:
            return None
        if decision.is_exit:
            return (
                f"同一条平仓决策已经生成过 {order_count} 条订单，"
                "为避免重复平仓，本次重复进入执行流程已跳过。"
            )
        if decision.is_entry:
            return (
                f"同一条开仓决策已经生成过 {order_count} 条订单，"
                "为避免重复开仓，本次重复进入执行流程已跳过。"
            )
        return None

    async def update_raw_response(
        self,
        decision_id: int,
        raw_response: dict[str, Any] | None,
    ) -> None:
        try:
            async with self._session_context_factory() as session:
                repo = self._decision_repo_factory(session)
                await repo.update_raw_response(decision_id, self.json_safe_payload(raw_response))
        except Exception as exc:
            logger.error("failed to update decision raw response", error=safe_error_text(exc))

    async def fill_missing_reasons(
        self,
        decision_ids: set[int] | list[int],
        reason: str,
    ) -> None:
        ids = [int(decision_id) for decision_id in decision_ids if decision_id]
        if not ids:
            return
        try:
            sanitized_reason = str(sanitize_runtime_text(reason) or reason)
            async with self._session_context_factory() as session:
                repo = self._decision_repo_factory(session)
                await repo.fill_missing_execution_reasons(ids, sanitized_reason)
        except Exception as exc:
            logger.error("failed to fill missing decision reasons", error=safe_error_text(exc))

    async def finalize_unresolved_decisions(
        self,
        decisions: dict[int, DecisionOutput],
        reason: str,
    ) -> int:
        """Persist a skipped risk-check terminal state for unresolved non-executed decisions."""

        if not decisions:
            return 0
        try:
            sanitized_reason = str(sanitize_runtime_text(reason) or reason)
            updates: list[tuple[int, str, dict[str, Any]]] = []
            for decision_id, decision in decisions.items():
                if not decision_id:
                    continue
                terminal_reason = self._terminal_reason_from_decision(decision)
                if terminal_reason is not None:
                    # Preserve an already-recorded branch result when a persistence
                    # callback was interrupted just before the round finalizer runs.
                    raw = (
                        dict(decision.raw_response)
                        if isinstance(decision.raw_response, dict)
                        else {}
                    )
                    updates.append(
                        (int(decision_id), terminal_reason or sanitized_reason, raw)
                    )
                    continue
                raw = self.record_stage(
                    decision,
                    DecisionStage.RISK_CHECK,
                    DecisionStageStatus.SKIPPED,
                    sanitized_reason,
                    data={
                        "skip_kind": "round_unresolved_terminal_skip",
                        "fallback_final_state": True,
                    },
                )
                updates.append((int(decision_id), sanitized_reason, raw))
            if not updates:
                return 0
            async with self._session_context_factory() as session:
                repo = self._decision_repo_factory(session)
                return await repo.finalize_unresolved_decisions(updates)
        except Exception as exc:
            logger.error(
                "failed to finalize unresolved decision states",
                error=safe_error_text(exc),
            )
            return 0

    @staticmethod
    def _terminal_reason_from_decision(decision: DecisionOutput) -> str | None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        summary = decision_state_from_raw(raw).get("summary")
        if not isinstance(summary, dict) or not is_decision_terminal_state(
            summary.get("final_stage"),
            summary.get("final_status"),
        ):
            return None
        opportunity = raw.get("opportunity_score")
        selection_reason = (
            opportunity.get("selection_reason") if isinstance(opportunity, dict) else None
        )
        return str(
            summary.get("final_reason")
            or raw.get("reason")
            or selection_reason
            or ""
        ).strip()

    async def mark_outcome(self, decision_id: int, outcome: str, pnl_pct: float) -> None:
        try:
            async with self._session_context_factory() as session:
                repo = self._decision_repo_factory(session)
                await repo.mark_outcome(decision_id, outcome, pnl_pct)
        except Exception as exc:
            logger.error("failed to mark decision outcome", error=safe_error_text(exc))
