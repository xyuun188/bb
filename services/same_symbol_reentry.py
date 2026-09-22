"""Execution-time guard for recently closed symbol lifecycles."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from sqlalchemy import select

from ai_brain.base_model import DecisionOutput
from core.symbols import normalize_trading_symbol, trading_symbol_variants
from db.session import get_read_session_ctx
from models.trade import OkxPositionHistory, Position

SAME_SYMBOL_REENTRY_CONTRACT_VERSION = "2026-09-22.same-symbol-reentry.v1"
MIN_PROFITABLE_REENTRY_SECONDS = 10 * 60
MIN_NON_PROFITABLE_REENTRY_SECONDS = 20 * 60

SessionContextFactory = Callable[[], AbstractAsyncContextManager[Any]]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if isfinite(parsed) else None


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _aware(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 100_000_000_000:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return _aware(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        numeric = _safe_float(text)
        return _parse_time(numeric) if numeric is not None else None


@dataclass(frozen=True, slots=True)
class ClosedLifecycleFact:
    source: str
    source_id: str
    symbol: str
    side: str | None
    opened_at: datetime | None
    closed_at: datetime
    realized_net_pnl_usdt: float | None
    settlement_status: str | None = None


@dataclass(frozen=True, slots=True)
class SameSymbolReentryAssessment:
    allowed: bool
    reason: str
    symbol: str
    execution_mode: str
    latest_close_source: str | None
    latest_close_id: str | None
    latest_close_side: str | None
    latest_closed_at: str | None
    previous_realized_net_pnl_usdt: float | None
    previous_holding_minutes: float | None
    decision_evidence_at: str | None
    decision_evidence_source: str
    prediction_horizon_minutes: float
    required_cooldown_seconds: float
    elapsed_since_close_seconds: float | None
    remaining_cooldown_seconds: float
    selection_reason: str | None
    opportunity_score: float | None
    expected_net_return_pct: float | None
    opportunity_contract_consistent: bool
    policy_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SameSymbolReentryGuard:
    """Block churn until a completed symbol lifecycle has had time to reset."""

    def __init__(
        self,
        *,
        session_context_factory: SessionContextFactory = get_read_session_ctx,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_context_factory = session_context_factory
        self._now_provider = now_provider or (lambda: datetime.now(UTC))

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_mode: str,
    ) -> SameSymbolReentryAssessment:
        mode = "live" if str(model_mode or "").lower() == "live" else "paper"
        symbol = normalize_trading_symbol(decision.symbol)
        latest = await self._latest_closed_lifecycle(symbol, mode) if symbol else None
        assessment = self.assess(
            decision,
            execution_mode=mode,
            latest_closed_lifecycle=latest,
            now=self._now_provider(),
        )
        raw = dict(_safe_dict(decision.raw_response))
        raw["same_symbol_reentry_guard"] = assessment.to_dict()
        decision.raw_response = raw
        return assessment

    @classmethod
    def assess(
        cls,
        decision: DecisionOutput,
        *,
        execution_mode: str,
        latest_closed_lifecycle: ClosedLifecycleFact | None,
        now: datetime,
    ) -> SameSymbolReentryAssessment:
        symbol = normalize_trading_symbol(decision.symbol)
        raw = _safe_dict(decision.raw_response)
        normal_trade = _safe_dict(raw.get("normal_paper_trade"))
        opportunity = _safe_dict(raw.get("opportunity_score"))
        selection_reason = str(normal_trade.get("selection_reason") or "").strip() or None
        opportunity_score = _safe_float(opportunity.get("score"))
        expected_net = _safe_float(
            opportunity.get(
                "expected_realized_net_return_pct",
                opportunity.get("expected_net_return_pct"),
            )
        )
        opportunity_consistent = bool(
            expected_net is None
            or (
                expected_net > 0.0
                and (
                    opportunity_score is None
                    or opportunity_score > 0.0
                    or selection_reason == "paper_quality_observation"
                )
            )
        )
        evidence_at, evidence_source = cls._decision_evidence_time(decision)
        horizon_minutes = cls._prediction_horizon_minutes(raw)
        current = _aware(now)

        if not decision.is_entry:
            return cls._assessment(
                allowed=True,
                reason="not_entry",
                symbol=symbol,
                execution_mode=execution_mode,
                latest=latest_closed_lifecycle,
                evidence_at=evidence_at,
                evidence_source=evidence_source,
                horizon_minutes=horizon_minutes,
                required_seconds=0.0,
                elapsed_seconds=None,
                selection_reason=selection_reason,
                opportunity_score=opportunity_score,
                expected_net=expected_net,
                opportunity_consistent=opportunity_consistent,
                now=current,
            )

        if not opportunity_consistent:
            return cls._assessment(
                allowed=False,
                reason="entry_opportunity_contract_inconsistent",
                symbol=symbol,
                execution_mode=execution_mode,
                latest=latest_closed_lifecycle,
                evidence_at=evidence_at,
                evidence_source=evidence_source,
                horizon_minutes=horizon_minutes,
                required_seconds=0.0,
                elapsed_seconds=None,
                selection_reason=selection_reason,
                opportunity_score=opportunity_score,
                expected_net=expected_net,
                opportunity_consistent=False,
                now=current,
            )

        if latest_closed_lifecycle is None:
            return cls._assessment(
                allowed=True,
                reason="no_recent_closed_symbol_lifecycle",
                symbol=symbol,
                execution_mode=execution_mode,
                latest=None,
                evidence_at=evidence_at,
                evidence_source=evidence_source,
                horizon_minutes=horizon_minutes,
                required_seconds=0.0,
                elapsed_seconds=None,
                selection_reason=selection_reason,
                opportunity_score=opportunity_score,
                expected_net=expected_net,
                opportunity_consistent=True,
                now=current,
            )

        profitable = bool(
            latest_closed_lifecycle.realized_net_pnl_usdt is not None
            and latest_closed_lifecycle.realized_net_pnl_usdt > 0.0
        )
        minimum_seconds = (
            MIN_PROFITABLE_REENTRY_SECONDS
            if profitable
            else MIN_NON_PROFITABLE_REENTRY_SECONDS
        )
        horizon_seconds = max(horizon_minutes, 0.0) * 60.0
        required_seconds = max(float(minimum_seconds), horizon_seconds)
        elapsed_seconds = max(
            (current - latest_closed_lifecycle.closed_at).total_seconds(),
            0.0,
        )

        if evidence_at is None or evidence_source == "decision.timestamp":
            reason = "same_symbol_reentry_market_evidence_unavailable"
            allowed = False
        elif evidence_at <= latest_closed_lifecycle.closed_at:
            reason = "same_symbol_reentry_market_evidence_not_newer_than_close"
            allowed = False
        elif elapsed_seconds + 1e-9 < required_seconds:
            reason = "same_symbol_reentry_cooldown_active"
            allowed = False
        else:
            reason = "same_symbol_reentry_cooldown_elapsed"
            allowed = True

        return cls._assessment(
            allowed=allowed,
            reason=reason,
            symbol=symbol,
            execution_mode=execution_mode,
            latest=latest_closed_lifecycle,
            evidence_at=evidence_at,
            evidence_source=evidence_source,
            horizon_minutes=horizon_minutes,
            required_seconds=required_seconds,
            elapsed_seconds=elapsed_seconds,
            selection_reason=selection_reason,
            opportunity_score=opportunity_score,
            expected_net=expected_net,
            opportunity_consistent=True,
            now=current,
        )

    async def _latest_closed_lifecycle(
        self,
        symbol: str,
        execution_mode: str,
    ) -> ClosedLifecycleFact | None:
        variants = trading_symbol_variants(symbol) or {symbol}
        facts: list[ClosedLifecycleFact] = []
        async with self._session_context_factory() as session:
            local_result = await session.execute(
                select(Position)
                .where(
                    Position.execution_mode == execution_mode,
                    Position.symbol.in_(variants),
                    Position.is_open.is_(False),
                    Position.closed_at.is_not(None),
                )
                .order_by(Position.closed_at.desc())
                .limit(1)
            )
            local = local_result.scalar_one_or_none()
            if local is not None and local.closed_at is not None:
                facts.append(
                    ClosedLifecycleFact(
                        source="local_position",
                        source_id=str(local.id),
                        symbol=normalize_trading_symbol(local.symbol),
                        side=str(local.side or "").lower() or None,
                        opened_at=_parse_time(local.created_at),
                        closed_at=_aware(local.closed_at),
                        realized_net_pnl_usdt=_safe_float(local.realized_pnl),
                        settlement_status=str(local.settlement_status or "").strip() or None,
                    )
                )

            history_result = await session.execute(
                select(OkxPositionHistory)
                .where(
                    OkxPositionHistory.mode == execution_mode,
                    OkxPositionHistory.symbol.in_(variants),
                    OkxPositionHistory.close_status == "full",
                    OkxPositionHistory.updated_at_okx.is_not(None),
                )
                .order_by(OkxPositionHistory.updated_at_okx.desc())
                .limit(1)
            )
            history = history_result.scalar_one_or_none()
            if history is not None and history.updated_at_okx is not None:
                facts.append(
                    ClosedLifecycleFact(
                        source="okx_position_history",
                        source_id=str(history.row_identity or history.id),
                        symbol=normalize_trading_symbol(history.symbol),
                        side=str(history.pos_side or history.side or "").lower() or None,
                        opened_at=_parse_time(history.opened_at),
                        closed_at=_aware(history.updated_at_okx),
                        realized_net_pnl_usdt=_safe_float(history.realized_pnl),
                        settlement_status=str(history.sync_status or "").strip() or None,
                    )
                )

        if not facts:
            return None
        return max(
            facts,
            key=lambda item: (
                item.closed_at,
                item.source == "okx_position_history",
            ),
        )

    @staticmethod
    def _decision_evidence_time(decision: DecisionOutput) -> tuple[datetime | None, str]:
        snapshot = _safe_dict(decision.feature_snapshot)
        for key in (
            "market_timestamp",
            "feature_timestamp",
            "timestamp",
            "observed_at",
            "generated_at",
            "bar_closed_at",
        ):
            parsed = _parse_time(snapshot.get(key))
            if parsed is not None:
                return parsed, f"feature_snapshot.{key}"
        timing = _safe_dict(_safe_dict(decision.raw_response).get("timing"))
        for key in ("decision_completed_at", "analysis_started_at"):
            parsed = _parse_time(timing.get(key))
            if parsed is not None:
                return parsed, f"raw_response.timing.{key}"
        parsed = _parse_time(decision.timestamp)
        return parsed, "decision.timestamp" if parsed is not None else "unavailable"

    @staticmethod
    def _prediction_horizon_minutes(raw: dict[str, Any]) -> float:
        normal_trade = _safe_dict(raw.get("normal_paper_trade"))
        opportunity = _safe_dict(raw.get("opportunity_score"))
        distribution = _safe_dict(opportunity.get("return_distribution_contract"))
        for value in (
            normal_trade.get("prediction_horizon_minutes"),
            opportunity.get("selected_horizon_minutes"),
            distribution.get("horizon_minutes"),
        ):
            parsed = _safe_float(value)
            if parsed is not None and parsed > 0.0:
                return parsed
        return 0.0

    @staticmethod
    def _assessment(
        *,
        allowed: bool,
        reason: str,
        symbol: str,
        execution_mode: str,
        latest: ClosedLifecycleFact | None,
        evidence_at: datetime | None,
        evidence_source: str,
        horizon_minutes: float,
        required_seconds: float,
        elapsed_seconds: float | None,
        selection_reason: str | None,
        opportunity_score: float | None,
        expected_net: float | None,
        opportunity_consistent: bool,
        now: datetime,
    ) -> SameSymbolReentryAssessment:
        holding_minutes = None
        if latest is not None and latest.opened_at is not None:
            holding_minutes = max(
                (latest.closed_at - latest.opened_at).total_seconds(),
                0.0,
            ) / 60.0
        remaining = (
            max(required_seconds - elapsed_seconds, 0.0)
            if elapsed_seconds is not None
            else 0.0
        )
        return SameSymbolReentryAssessment(
            allowed=allowed,
            reason=reason,
            symbol=symbol,
            execution_mode=execution_mode,
            latest_close_source=latest.source if latest is not None else None,
            latest_close_id=latest.source_id if latest is not None else None,
            latest_close_side=latest.side if latest is not None else None,
            latest_closed_at=(latest.closed_at.isoformat() if latest is not None else None),
            previous_realized_net_pnl_usdt=(
                latest.realized_net_pnl_usdt if latest is not None else None
            ),
            previous_holding_minutes=(
                round(holding_minutes, 8) if holding_minutes is not None else None
            ),
            decision_evidence_at=evidence_at.isoformat() if evidence_at is not None else None,
            decision_evidence_source=evidence_source,
            prediction_horizon_minutes=round(max(horizon_minutes, 0.0), 8),
            required_cooldown_seconds=round(max(required_seconds, 0.0), 3),
            elapsed_since_close_seconds=(
                round(max(elapsed_seconds, 0.0), 3)
                if elapsed_seconds is not None
                else None
            ),
            remaining_cooldown_seconds=round(remaining, 3),
            selection_reason=selection_reason,
            opportunity_score=opportunity_score,
            expected_net_return_pct=expected_net,
            opportunity_contract_consistent=opportunity_consistent,
            policy_provenance={
                "source": "local_position_and_okx_full_close_history",
                "observation_window": "latest_complete_symbol_lifecycle",
                "sample_count": 1 if latest is not None else 0,
                "generated_at": now.isoformat(),
                "strategy_version": SAME_SYMBOL_REENTRY_CONTRACT_VERSION,
                "fallback_reason": "" if allowed else reason,
                "profitable_reentry_floor_seconds": MIN_PROFITABLE_REENTRY_SECONDS,
                "non_profitable_reentry_floor_seconds": (
                    MIN_NON_PROFITABLE_REENTRY_SECONDS
                ),
                "cooldown_basis": "max_outcome_floor_and_prediction_horizon",
                "production_permission": False,
            },
        )
