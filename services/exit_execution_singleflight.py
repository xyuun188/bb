"""Persistent single-flight coordination for exchange exit submissions."""

from __future__ import annotations

import secrets
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from executor.base_executor import ExecutionResult, OrderStatus
from services.okx_error_classifier import is_okx_temporary_service_error

EXIT_INTENT_KEY = "exit_execution_intent"
EXIT_INTENT_VERSION = "2026-08-20.exit-singleflight.v1"
SUBMIT_LEASE_SECONDS = 150.0
UNCONFIRMED_RETRY_SECONDS = 120.0
UNKNOWN_RESULT_RETRY_SECONDS = 180.0
TEMPORARY_OUTAGE_RETRY_SECONDS = 60.0
REJECTED_RETRY_SECONDS = 45.0
COMPLETED_CONFIRMATION_SECONDS = 300.0

SessionContextFactory = Callable[[], AbstractAsyncContextManager[Any]]
TradeRepoFactory = Callable[[Any], TradeRepository]


@dataclass(frozen=True)
class ExitExecutionLease:
    acquired: bool
    token: str
    key: str
    position_ids: tuple[int, ...]
    state: str
    attempt_count: int
    retry_after_seconds: float = 0.0
    reason: str = ""


class ExitExecutionSingleFlightService:
    """Store an exit lease on the open position lifecycle before OKX submission."""

    def __init__(
        self,
        *,
        session_context_factory: SessionContextFactory = get_session_ctx,
        trade_repo_factory: TradeRepoFactory = TradeRepository,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_context_factory = session_context_factory
        self._trade_repo_factory = trade_repo_factory
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    async def acquire(
        self,
        *,
        model_name: str,
        execution_mode: str,
        decision: DecisionOutput,
        decision_id: int | None,
    ) -> ExitExecutionLease:
        if not decision.is_exit:
            return ExitExecutionLease(
                acquired=True,
                token="",
                key="not_exit",
                position_ids=(),
                state="not_exit",
                attempt_count=0,
            )

        symbol = normalize_trading_symbol(decision.symbol)
        side = "long" if decision.action == Action.CLOSE_LONG else "short"
        mode = "live" if str(execution_mode).lower() == "live" else "paper"
        now = self._aware(self._now_provider())

        async with self._session_context_factory() as session:
            repo = self._trade_repo_factory(session)
            positions = await repo.get_matching_open_positions_for_update(
                model_name=model_name,
                symbol=symbol,
                side=side,
                execution_mode=mode,
            )
            if not positions:
                return ExitExecutionLease(
                    acquired=True,
                    token="",
                    key=f"{mode}:{symbol}:{side}:no_local_position",
                    position_ids=(),
                    state="no_local_position",
                    attempt_count=0,
                )

            position_ids = tuple(sorted(int(position.id) for position in positions))
            key = self._position_group_key(mode, symbol, side, positions)
            current = self._current_group_intent(positions)
            retry_at = self._parse_time(current.get("retry_after_at"))
            if current.get("key") == key and retry_at is not None and retry_at > now:
                retry_after = max((retry_at - now).total_seconds(), 0.0)
                return ExitExecutionLease(
                    acquired=False,
                    token=str(current.get("token") or ""),
                    key=key,
                    position_ids=position_ids,
                    state=str(current.get("state") or "waiting"),
                    attempt_count=self._safe_int(current.get("attempt_count"), 0),
                    retry_after_seconds=retry_after,
                    reason=(
                        "A previous exit submission for this position is still awaiting "
                        "exchange confirmation. No duplicate exit was submitted."
                    ),
                )

            token = secrets.token_hex(12)
            attempt_count = self._safe_int(current.get("attempt_count"), 0) + 1
            intent = {
                "version": EXIT_INTENT_VERSION,
                "key": key,
                "token": token,
                "state": "submitting",
                "attempt_count": attempt_count,
                "decision_id": decision_id,
                "position_ids": list(position_ids),
                "acquired_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "retry_after_at": (now + timedelta(seconds=SUBMIT_LEASE_SECONDS)).isoformat(),
                "last_error": None,
            }
            self._apply_intent(positions, intent)
            await session.flush()
            return ExitExecutionLease(
                acquired=True,
                token=token,
                key=key,
                position_ids=position_ids,
                state="submitting",
                attempt_count=attempt_count,
            )

    async def finish(
        self,
        lease: ExitExecutionLease,
        result: ExecutionResult | None,
    ) -> None:
        if not lease.acquired or not lease.token or not lease.position_ids:
            return
        now = self._aware(self._now_provider())
        state, delay, error = self._result_state(result)
        async with self._session_context_factory() as session:
            repo = self._trade_repo_factory(session)
            positions = await repo.get_positions_for_update(lease.position_ids)
            matching = [
                position
                for position in positions
                if self._position_intent(position).get("token") == lease.token
            ]
            if not matching:
                return
            intent = dict(self._position_intent(matching[0]))
            intent.update(
                {
                    "state": state,
                    "updated_at": now.isoformat(),
                    "retry_after_at": (now + timedelta(seconds=delay)).isoformat(),
                    "last_error": error or None,
                    "result_status": self._result_status(result),
                    "exchange_order_id": (
                        str(getattr(result, "exchange_order_id", "") or "").strip() or None
                    ),
                }
            )
            self._apply_intent(matching, intent)
            await session.flush()

    @staticmethod
    def waiting_result(
        decision: DecisionOutput,
        lease: ExitExecutionLease,
    ) -> ExecutionResult:
        side = "sell" if decision.action == Action.CLOSE_LONG else "buy"
        return ExecutionResult(
            order_id="exit_singleflight_wait",
            symbol=decision.symbol,
            side=side,
            order_type="market",
            quantity=0.0,
            price=0.0,
            status=OrderStatus.OPEN,
            raw_response={
                "exit_tracking": True,
                "exit_singleflight_wait": True,
                "do_not_persist_order": True,
                "singleflight_key": lease.key,
                "singleflight_state": lease.state,
                "attempt_count": lease.attempt_count,
                "retry_after_seconds": round(max(lease.retry_after_seconds, 0.0), 3),
                "message": lease.reason,
            },
        )

    @classmethod
    def _result_state(
        cls,
        result: ExecutionResult | None,
    ) -> tuple[str, float, str]:
        if result is None:
            return (
                "submitted_result_unknown",
                UNKNOWN_RESULT_RETRY_SECONDS,
                "Exchange submission returned no result; confirmation is required before retry.",
            )
        raw = result.raw_response if isinstance(result.raw_response, dict) else {}
        error = safe_error_text(raw.get("raw_error") or raw.get("error") or "", limit=300)
        rendered = " ".join(
            str(value or "")
            for value in (error, raw, result.order_id, result.exchange_order_id)
        )
        if is_okx_temporary_service_error(rendered):
            return "exchange_temporarily_unavailable", TEMPORARY_OUTAGE_RETRY_SECONDS, error
        if raw.get("execution_transport_unknown"):
            return "submitted_result_unknown", UNKNOWN_RESULT_RETRY_SECONDS, error
        if result.status in {OrderStatus.FILLED, OrderStatus.PARTIAL} and result.quantity > 0:
            return "exchange_progress_confirmed", COMPLETED_CONFIRMATION_SECONDS, error
        if result.order_id == "no_position":
            return "exchange_position_absent", COMPLETED_CONFIRMATION_SECONDS, error
        if result.status in {OrderStatus.OPEN, OrderStatus.PENDING}:
            return "submitted_unconfirmed", UNCONFIRMED_RETRY_SECONDS, error
        return "retry_wait", REJECTED_RETRY_SECONDS, error

    @staticmethod
    def _result_status(result: ExecutionResult | None) -> str | None:
        status = getattr(result, "status", None)
        value = getattr(status, "value", status)
        return str(value) if value is not None else None

    @staticmethod
    def _position_group_key(
        mode: str,
        symbol: str,
        side: str,
        positions: list[Any],
    ) -> str:
        lifecycle_tokens: list[str] = []
        for position in positions:
            entry_order_id = str(getattr(position, "entry_exchange_order_id", "") or "").strip()
            okx_pos_id = str(getattr(position, "okx_pos_id", "") or "").strip()
            if entry_order_id:
                lifecycle_tokens.append(f"entry:{entry_order_id}")
            elif okx_pos_id:
                created_at = getattr(position, "created_at", None)
                lifecycle_tokens.append(f"pos:{okx_pos_id}:{created_at or position.id}")
            else:
                lifecycle_tokens.append(f"local:{position.id}")
        return ":".join((mode, symbol, side, "|".join(sorted(lifecycle_tokens))))

    @classmethod
    def _current_group_intent(cls, positions: list[Any]) -> dict[str, Any]:
        intents = [cls._position_intent(position) for position in positions]
        intents = [intent for intent in intents if intent]
        if not intents:
            return {}
        return max(
            intents,
            key=lambda intent: cls._parse_time(intent.get("updated_at"))
            or datetime.min.replace(tzinfo=UTC),
        )

    @staticmethod
    def _position_intent(position: Any) -> dict[str, Any]:
        contract = getattr(position, "current_management_contract", None)
        if not isinstance(contract, dict):
            return {}
        intent = contract.get(EXIT_INTENT_KEY)
        return dict(intent) if isinstance(intent, dict) else {}

    @staticmethod
    def _apply_intent(positions: list[Any], intent: dict[str, Any]) -> None:
        for position in positions:
            contract = getattr(position, "current_management_contract", None)
            contract = dict(contract) if isinstance(contract, dict) else {}
            contract[EXIT_INTENT_KEY] = dict(intent)
            position.current_management_contract = contract

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return ExitExecutionSingleFlightService._aware(value)
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return ExitExecutionSingleFlightService._aware(
                datetime.fromisoformat(text.replace("Z", "+00:00"))
            )
        except ValueError:
            return None

    @staticmethod
    def _aware(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


def preserve_exit_execution_intent(
    previous_contract: Any,
    refreshed_contract: dict[str, Any],
) -> dict[str, Any]:
    """Keep the lifecycle lease when periodic position facts rebuild the contract."""

    merged = dict(refreshed_contract)
    previous = previous_contract if isinstance(previous_contract, dict) else {}
    intent = previous.get(EXIT_INTENT_KEY)
    if isinstance(intent, dict) and intent:
        merged[EXIT_INTENT_KEY] = dict(intent)
    return merged
