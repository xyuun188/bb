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

    def score_candidate(
        self,
        decision: DecisionOutput,
        strategy: dict[str, Any] | None = None,
    ) -> float:
        return self.orchestrator._candidate_opportunity_score(decision, strategy)

    def immediate_execution_reason(self, decision: DecisionOutput) -> str | None:
        return self.orchestrator._entry_immediate_execution_reason(decision)

    def wait_sort_reason(
        self,
        decision: DecisionOutput,
        *,
        rank: int | None = None,
        candidate_count: int | None = None,
    ) -> str:
        return self.orchestrator._entry_wait_sort_reason(
            decision,
            rank=rank,
            candidate_count=candidate_count,
        )

    def gate_reason(self, decision: DecisionOutput) -> str | None:
        return self.orchestrator._entry_opportunity_gate_reason(decision)

    async def pre_execution_price_guard_reason(self, decision: DecisionOutput) -> str | None:
        return await self.orchestrator._pre_execution_price_guard_reason(decision)

    async def apply_profit_risk_sizing(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        await self.orchestrator._apply_entry_profit_risk_sizing(
            decision,
            model_mode,
            open_positions=open_positions or [],
        )

    async def high_risk_review_gate(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> str | None:
        return await self.orchestrator._high_risk_review_gate(
            decision,
            model_mode,
            open_positions or [],
        )

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

        price_guard_reason = await self.pre_execution_price_guard_reason(decision)
        if price_guard_reason:
            return PolicyGateResult.block("pre_execution_price_guard", price_guard_reason)

        await self.apply_profit_risk_sizing(
            decision,
            model_mode,
            open_positions=open_positions or [],
        )
        high_risk_reason = await self.high_risk_review_gate(
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

    def has_matching_position(
        self,
        positions: list[dict[str, Any]] | None,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool:
        if not decision.is_exit:
            return True
        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        target_symbol = self.orchestrator._normalize_position_symbol(decision.symbol)
        for pos in positions or []:
            if str(pos.get("model_name") or "") != model_name:
                continue
            if self.orchestrator._normalize_position_symbol(pos.get("symbol")) != target_symbol:
                continue
            if str(pos.get("side") or "").lower() != target_side:
                continue
            if pos.get("is_open", True) is False:
                continue
            try:
                quantity = float(pos.get("quantity", pos.get("contracts", 0)) or 0)
            except (TypeError, ValueError):
                quantity = 0.0
            if quantity > 0:
                return True
        return False

    def no_matching_position_reason(self, decision: DecisionOutput) -> str:
        side_label = "多单" if decision.action == Action.CLOSE_LONG else "空单"
        return f"没有找到 {decision.symbol} 对应的可平{side_label}仓位，未向 OKX 提交平仓单。"

    def loss_partial_guard_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]] | None,
    ) -> str | None:
        return self.orchestrator._loss_partial_exit_guard_reason(
            model_name,
            decision,
            open_positions,
        )

    def recent_exit_cooldown_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> str | None:
        return self.orchestrator._recent_exit_cooldown_reason(model_name, decision)

    async def pre_execution_profit_guard_reason(
        self,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]] | None,
    ) -> str | None:
        return await self.orchestrator._pre_execution_profit_exit_guard_reason(
            decision,
            open_positions,
        )

    async def fee_churn_guard_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> str | None:
        return await self.orchestrator._exit_fee_churn_guard_reason(model_name, decision)

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

        if not self.has_matching_position(exit_positions, model_name, decision):
            exchange_has_position = await self.orchestrator.okx_sync_service.has_matching_exchange_exit_position(
                model_name,
                decision,
            )
            if not exchange_has_position:
                return PolicyGateResult.block(
                    "no_matching_exit_position",
                    self.no_matching_position_reason(decision),
                )

        loss_partial_reason = self.loss_partial_guard_reason(
            model_name,
            decision,
            exit_positions,
        )
        if loss_partial_reason:
            return PolicyGateResult.block("loss_partial_exit_guard", loss_partial_reason)

        recent_exit_reason = self.recent_exit_cooldown_reason(model_name, decision)
        if recent_exit_reason:
            return PolicyGateResult.block("recent_exit_cooldown", recent_exit_reason)

        profit_exit_guard_reason = await self.pre_execution_profit_guard_reason(
            decision,
            exit_positions,
        )
        if profit_exit_guard_reason:
            return PolicyGateResult.block("profit_exit_precheck", profit_exit_guard_reason)

        guard_reason = await self.fee_churn_guard_reason(model_name, decision)
        if guard_reason:
            return PolicyGateResult.block("exit_fee_churn_guard", guard_reason)

        stale_reason = self.orchestrator._stale_decision_reason(decision)
        if stale_reason:
            return PolicyGateResult.block("stale_decision", stale_reason)

        return PolicyGateResult.allow({
            "intent": "exit",
            "target_side": "long" if decision.action == Action.CLOSE_LONG else "short",
        })
