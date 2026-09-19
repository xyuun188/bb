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
from services.current_position_management import (
    PROFIT_LOCK_LEDGER_KEY,
    PROFIT_LOCK_LEDGER_VERSION,
)
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
                    # The position lifecycle is already closed (usually by a
                    # protection/full-close order).  This is a terminal skip,
                    # not a lease that may continue to the exchange submit.
                    acquired=False,
                    token="",
                    key=f"{mode}:{symbol}:{side}:no_local_position",
                    position_ids=(),
                    state="no_local_position",
                    attempt_count=0,
                    reason=(
                        "前一笔平仓已经完成，当前没有可平仓位；"
                        "本次重复平仓请求已跳过，未再次提交交易所。"
                    ),
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
            profit_lock_exit = self._is_profit_lock_exit(decision)
            intent = {
                "version": EXIT_INTENT_VERSION,
                "key": key,
                "token": token,
                "state": "submitting",
                "attempt_count": attempt_count,
                "decision_id": decision_id,
                "exit_reason_class": (
                    "profit_lock" if profit_lock_exit else "risk_or_other"
                ),
                "requested_close_fraction": self._decision_close_fraction(decision),
                "profit_lock_exit": profit_lock_exit,
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
            if (
                intent.get("profit_lock_exit") is True
                and result is not None
                and result.status in {OrderStatus.FILLED, OrderStatus.PARTIAL}
                and result.quantity > 0
            ):
                self._apply_profit_lock_fill(
                    matching,
                    intent=intent,
                    filled_quantity=float(result.quantity),
                    decision_id=self._safe_int(intent.get("decision_id"), 0) or None,
                    exchange_order_id=(
                        str(getattr(result, "exchange_order_id", "") or "").strip()
                        or None
                    ),
                    updated_at=now,
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
                "message": lease.reason
                or (
                    "前一笔平仓已经完成，当前没有可平仓位；"
                    "本次重复平仓请求已跳过，未再次提交交易所。"
                    if lease.state == "no_local_position"
                    else ""
                ),
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
    def _decision_close_fraction(decision: DecisionOutput) -> float:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        policy = raw.get("dynamic_exit_policy")
        policy = policy if isinstance(policy, dict) else {}
        value = policy.get("close_fraction", decision.position_size_pct)
        try:
            return round(min(max(float(value or 0.0), 0.0), 1.0), 8)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_profit_lock_exit(decision: DecisionOutput) -> bool:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        policy = raw.get("dynamic_exit_policy")
        policy = policy if isinstance(policy, dict) else {}
        try:
            lifecycle_net_pnl = float(policy.get("lifecycle_net_pnl_usdt") or 0.0)
            profit_lock_pressure = float(policy.get("profit_lock_pressure") or 0.0)
        except (TypeError, ValueError):
            return False
        return bool(
            policy.get("eligible") is True
            and policy.get("hard_risk") is not True
            and lifecycle_net_pnl > 0.0
            and profit_lock_pressure > 0.0
        )

    @classmethod
    def _apply_profit_lock_fill(
        cls,
        positions: list[Any],
        *,
        intent: dict[str, Any],
        filled_quantity: float,
        decision_id: int | None,
        exchange_order_id: str | None,
        updated_at: datetime,
    ) -> None:
        ledger = cls._current_group_profit_lock_ledger(positions)
        token = str(intent.get("token") or "").strip()
        intent_fills = {
            str(key): max(cls._safe_float(value, 0.0), 0.0)
            for key, value in (ledger.get("intent_fill_quantities") or {}).items()
            if str(key or "").strip()
        }
        previous_intent_fill = intent_fills.get(token, 0.0) if token else 0.0
        observed_intent_fill = max(float(filled_quantity or 0.0), 0.0)
        incremental_fill = max(observed_intent_fill - previous_intent_fill, 0.0)
        if incremental_fill <= 0.0:
            return
        realized_quantity = max(
            cls._safe_float(ledger.get("realized_quantity"), 0.0),
            0.0,
        ) + incremental_fill
        lifecycle_entry_quantity = max(
            (
                cls._safe_float(
                    getattr(position, "current_management_contract", {}).get(
                        "lifecycle_entry_quantity"
                    ),
                    0.0,
                )
                for position in positions
                if isinstance(getattr(position, "current_management_contract", None), dict)
            ),
            default=0.0,
        )
        if token:
            intent_fills[token] = observed_intent_fill
        if len(intent_fills) > 32:
            intent_fills = dict(list(intent_fills.items())[-32:])
        next_ledger = {
            "version": PROFIT_LOCK_LEDGER_VERSION,
            "realized_quantity": round(realized_quantity, 12),
            "realized_fraction": round(
                min(realized_quantity / lifecycle_entry_quantity, 1.0)
                if lifecycle_entry_quantity > 0.0
                else 0.0,
                8,
            ),
            "last_decision_id": decision_id,
            "last_exchange_order_id": exchange_order_id,
            "last_filled_quantity": round(observed_intent_fill, 12),
            "updated_at": updated_at.isoformat(),
            "intent_fill_quantities": intent_fills,
        }
        for position in positions:
            contract = getattr(position, "current_management_contract", None)
            contract = dict(contract) if isinstance(contract, dict) else {}
            contract[PROFIT_LOCK_LEDGER_KEY] = dict(next_ledger)
            position.current_management_contract = contract

    @classmethod
    def _current_group_profit_lock_ledger(cls, positions: list[Any]) -> dict[str, Any]:
        ledgers: list[dict[str, Any]] = []
        for position in positions:
            contract = getattr(position, "current_management_contract", None)
            if not isinstance(contract, dict):
                continue
            ledger = contract.get(PROFIT_LOCK_LEDGER_KEY)
            if isinstance(ledger, dict):
                ledgers.append(dict(ledger))
        if not ledgers:
            return {}
        return max(
            ledgers,
            key=lambda ledger: (
                cls._safe_float(ledger.get("realized_quantity"), 0.0),
                cls._parse_time(ledger.get("updated_at"))
                or datetime.min.replace(tzinfo=UTC),
            ),
        )

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

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


def preserve_exit_execution_intent(
    previous_contract: Any,
    refreshed_contract: dict[str, Any],
) -> dict[str, Any]:
    """Keep the lifecycle lease when periodic position facts rebuild the contract."""

    merged = dict(refreshed_contract)
    previous = previous_contract if isinstance(previous_contract, dict) else {}
    for key in (EXIT_INTENT_KEY, PROFIT_LOCK_LEDGER_KEY):
        value = previous.get(key)
        if isinstance(value, dict) and value:
            merged[key] = dict(value)
    return merged
