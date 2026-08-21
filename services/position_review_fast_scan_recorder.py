"""Persistence for position-review groups skipped by slow AI review."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.position_review_fast_scan_hold import PositionReviewFastScanHoldPolicy
from services.position_review_result_recorder import PositionReviewResultRecorder

NormalizeSymbol = Callable[[Any], str | None]
UrgentExitChecker = Callable[[dict[str, Any] | None], bool]
PortfolioSymbolContextProvider = Callable[
    [dict[str, Any], str, str, list[dict[str, Any]] | None],
    dict[str, Any],
]
PositionSkillsProvider = Callable[..., list[Any]]
AgentSkillsSummaryProvider = Callable[[list[Any]], dict[str, Any]]
DeferCountProvider = Callable[[tuple[str, str]], int]
DeferCountApplier = Callable[[tuple[str, str], int], None]
ModelExecutionModeProvider = Callable[[str], str]
DecisionLogger = Callable[[DecisionOutput, bool], Awaitable[int | None]]
DecisionReasonMarker = Callable[[int, str], Awaitable[None]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class PositionReviewFastScanRecorder:
    """Record HOLD rows for position groups deferred from slow review."""

    default_model_name: str
    normalize_symbol: NormalizeSymbol
    urgent_exit_checker: UrgentExitChecker
    portfolio_symbol_context_provider: PortfolioSymbolContextProvider
    position_skills_provider: PositionSkillsProvider
    agent_skills_summary_provider: AgentSkillsSummaryProvider
    defer_count_provider: DeferCountProvider
    defer_count_applier: DeferCountApplier
    model_execution_mode_provider: ModelExecutionModeProvider
    decision_logger: DecisionLogger
    decision_reason_marker: DecisionReasonMarker
    result_recorder: PositionReviewResultRecorder
    hold_policy: PositionReviewFastScanHoldPolicy
    no_action_record_interval_seconds: float = 300.0
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    _no_action_state: dict[tuple[str, str], tuple[tuple[Any, ...], datetime]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def reset_keys(self, keys: set[tuple[str, str]] | None) -> None:
        """Forget audit state after a group receives its slow governed review."""

        for key in keys or set():
            self._no_action_state.pop(key, None)

    async def record_many(
        self,
        *,
        skipped_items: list[tuple[tuple[str, str], list[dict[str, Any]]]],
        fast_scan: dict[tuple[str, str], dict[str, Any]],
        feature_vectors: dict[str, Any],
        portfolio_profit_context: dict[str, Any] | None,
        results: dict[str, Any] | None,
        round_decision_ids: set[int] | None,
        position_entry_pause_reason: str | None = None,
    ) -> int:
        """Persist fast-scan HOLD decisions and return the number logged."""

        if not skipped_items:
            return 0

        logged_count = 0
        for (model_name, symbol), positions in skipped_items:
            logged_count += await self._record_one(
                model_name=model_name,
                symbol=symbol,
                positions=positions,
                fast_scan=fast_scan,
                feature_vectors=feature_vectors,
                portfolio_profit_context=portfolio_profit_context,
                results=results,
                round_decision_ids=round_decision_ids,
                position_entry_pause_reason=position_entry_pause_reason,
            )
        return logged_count

    async def _record_one(
        self,
        *,
        model_name: str,
        symbol: str,
        positions: list[dict[str, Any]],
        fast_scan: dict[tuple[str, str], dict[str, Any]],
        feature_vectors: dict[str, Any],
        portfolio_profit_context: dict[str, Any] | None,
        results: dict[str, Any] | None,
        round_decision_ids: set[int] | None,
        position_entry_pause_reason: str | None,
    ) -> int:
        effective_model_name = model_name or self.default_model_name
        key = (model_name, symbol)
        normalized = self.normalize_symbol(symbol)
        scan = fast_scan.get((model_name, symbol), {})
        urgent_exit = self.urgent_exit_checker(scan)
        portfolio_symbol_context = self.portfolio_symbol_context_provider(
            portfolio_profit_context or {},
            effective_model_name,
            normalized or symbol,
            positions,
        )
        fast_scan_skills = self.position_skills_provider(
            position_entry_pause_reason=position_entry_pause_reason,
            ml_signal=None,
            local_ai_tools=None,
            portfolio_profit_protection=portfolio_symbol_context,
        )
        previous_defer_count = self.defer_count_provider(key)
        hold_plan = self.hold_policy.plan(
            scan,
            previous_defer_count=previous_defer_count,
            urgent_exit=urgent_exit,
            portfolio_symbol_context=portfolio_symbol_context,
            agent_skill_dicts=[skill.to_dict() for skill in fast_scan_skills],
            agent_skill_summary=self.agent_skills_summary_provider(fast_scan_skills),
        )
        self.defer_count_applier(key, hold_plan.defer_count)

        if not self._should_record_scan(
            key=key,
            scan=scan,
            positions=positions,
            urgent_exit=urgent_exit,
            hold_plan=hold_plan,
        ):
            return 0

        fv = feature_vectors.get(symbol) or feature_vectors.get(normalized)
        decision = DecisionOutput(
            model_name=effective_model_name,
            symbol=symbol,
            action=Action.HOLD,
            confidence=0.0,
            reasoning=hold_plan.reason,
            position_size_pct=0.0,
            suggested_leverage=1.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            raw_response=hold_plan.raw_response,
            feature_snapshot=fv.to_dict() if fv is not None and hasattr(fv, "to_dict") else {},
        )
        model_mode = self.model_execution_mode_provider(effective_model_name)
        decision_db_id = await self.decision_logger(decision, model_mode == "paper")
        self.result_recorder.append_fast_scan_result(
            results=results,
            model_name=effective_model_name,
            symbol=symbol,
            reason=hold_plan.reason,
            model_mode=model_mode,
        )
        if decision_db_id is None:
            return 0

        if round_decision_ids is not None:
            round_decision_ids.add(decision_db_id)
        await self.decision_reason_marker(decision_db_id, hold_plan.reason)
        return 1

    def _should_record_scan(
        self,
        *,
        key: tuple[str, str],
        scan: dict[str, Any],
        positions: list[dict[str, Any]],
        urgent_exit: bool,
        hold_plan: Any,
    ) -> bool:
        """Throttle only stable, no-action observations from the fast pass."""

        dynamic_exit = scan.get("dynamic_exit_policy")
        dynamic_exit = dynamic_exit if isinstance(dynamic_exit, dict) else {}
        eligible = bool(
            scan.get("dynamic_exit_eligible") is True and dynamic_exit.get("eligible") is True
        )
        hard_risk = bool(scan.get("dynamic_exit_hard_risk") or dynamic_exit.get("hard_risk"))
        if eligible or hard_risk or urgent_exit:
            return True

        fingerprint = (
            str(scan.get("reason") or ""),
            bool(scan.get("dynamic_exit_eligible")),
            bool(dynamic_exit.get("eligible")),
            round(_safe_float(dynamic_exit.get("close_fraction")), 4),
            tuple(
                sorted(
                    (
                        str(position.get("side") or "").lower(),
                        str(
                            position.get("id")
                            or position.get("pos_id")
                            or position.get("posId")
                            or ""
                        ),
                        round(_safe_float(position.get("contracts") or position.get("size")), 8),
                    )
                    for position in positions or []
                    if isinstance(position, dict)
                )
            ),
            str(getattr(hold_plan, "reason", "") or ""),
        )
        now = self.clock()
        previous = self._no_action_state.get(key)
        if previous is None:
            self._no_action_state[key] = (fingerprint, now)
            return True
        previous_fingerprint, previous_at = previous
        if previous_fingerprint != fingerprint:
            self._no_action_state[key] = (fingerprint, now)
            return True
        try:
            elapsed = (now - previous_at).total_seconds()
        except (AttributeError, TypeError):
            self._no_action_state[key] = (fingerprint, now)
            return True
        if elapsed < max(float(self.no_action_record_interval_seconds), 0.0):
            return False
        self._no_action_state[key] = (fingerprint, now)
        return True
