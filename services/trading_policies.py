"""Entry and exit policy gates.

These classes keep strategy/risk decisions out of the OKX submit section.  They
return one explicit blocker and Chinese reason; TradingService then records the
state-machine event and dashboard row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput


@dataclass(slots=True)
class PolicyGateResult:
    passed: bool
    blocker: str | None = None
    reason: str | None = None
    data: dict[str, Any] | None = None

    @classmethod
    def allow(cls, data: dict[str, Any] | None = None) -> "PolicyGateResult":
        return cls(True, data=data or {})

    @classmethod
    def block(
        cls,
        blocker: str,
        reason: str,
        data: dict[str, Any] | None = None,
    ) -> "PolicyGateResult":
        return cls(False, blocker=blocker, reason=reason, data=data or {})


class EntryPolicy:
    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_name: str,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> PolicyGateResult:
        if not decision.is_entry:
            return PolicyGateResult.allow({"intent": "not_entry"})

        stale_reason = self.orchestrator._stale_decision_reason(decision)
        if stale_reason:
            return PolicyGateResult.block("stale_decision", stale_reason)

        abnormal_wick_reason = self.orchestrator._abnormal_wick_entry_guard_reason(decision)
        if abnormal_wick_reason:
            return PolicyGateResult.block("abnormal_wick_entry_guard", abnormal_wick_reason)

        price_guard_reason = await self.orchestrator._pre_execution_price_guard_reason(decision)
        if price_guard_reason:
            return PolicyGateResult.block("pre_execution_price_guard", price_guard_reason)

        await self.orchestrator._apply_entry_profit_risk_sizing(
            decision,
            model_mode,
            open_positions=open_positions or [],
        )
        high_risk_reason = await self.orchestrator._high_risk_review_gate(
            decision,
            model_mode,
            open_positions or [],
        )
        if high_risk_reason:
            return PolicyGateResult.block("high_risk_review", high_risk_reason)

        return PolicyGateResult.allow({"intent": "entry"})


class ExitPolicy:
    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_name: str,
        open_positions: list[dict[str, Any]] | None,
    ) -> PolicyGateResult:
        if not decision.is_exit:
            return PolicyGateResult.allow({"intent": "not_exit"})

        await self.orchestrator.okx_sync_service.reconcile_positions("exit precheck")
        exit_positions = await self.orchestrator.okx_sync_service.get_open_positions_context()
        if open_positions is not None:
            open_positions[:] = exit_positions

        if not self.orchestrator._has_matching_exit_position(exit_positions, model_name, decision):
            exchange_has_position = await self.orchestrator.okx_sync_service.has_matching_exchange_exit_position(
                model_name,
                decision,
            )
            if not exchange_has_position:
                return PolicyGateResult.block(
                    "no_matching_exit_position",
                    self.orchestrator._no_matching_exit_position_reason(decision),
                )

        loss_partial_reason = self.orchestrator._loss_partial_exit_guard_reason(
            model_name,
            decision,
            exit_positions,
        )
        if loss_partial_reason:
            return PolicyGateResult.block("loss_partial_exit_guard", loss_partial_reason)

        recent_exit_reason = self.orchestrator._recent_exit_cooldown_reason(model_name, decision)
        if recent_exit_reason:
            return PolicyGateResult.block("recent_exit_cooldown", recent_exit_reason)

        profit_exit_guard_reason = await self.orchestrator._pre_execution_profit_exit_guard_reason(
            decision,
            exit_positions,
        )
        if profit_exit_guard_reason:
            return PolicyGateResult.block("profit_exit_precheck", profit_exit_guard_reason)

        guard_reason = await self.orchestrator._exit_fee_churn_guard_reason(model_name, decision)
        if guard_reason:
            return PolicyGateResult.block("exit_fee_churn_guard", guard_reason)

        stale_reason = self.orchestrator._stale_decision_reason(decision)
        if stale_reason:
            return PolicyGateResult.block("stale_decision", stale_reason)

        return PolicyGateResult.allow({
            "intent": "exit",
            "target_side": "long" if decision.action == Action.CLOSE_LONG else "short",
        })
