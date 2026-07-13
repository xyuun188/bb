"""Entry and exit policy gates.

These classes keep strategy/risk decisions out of the OKX submit section.  They
return one explicit blocker and Chinese reason; TradingService then records the
state-machine event and dashboard row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.dynamic_exit_policy import apply_dynamic_exit
from services.pipeline_context import EntryPipelineContext, ExitPipelineContext
from services.return_execution_policy import apply_production_entry_policy


@dataclass(slots=True)
class PolicyGateResult:
    passed: bool
    blocker: str | None = None
    reason: str | None = None
    data: dict[str, Any] | None = None

    @classmethod
    def allow(cls, data: dict[str, Any] | None = None) -> PolicyGateResult:
        return cls(True, data=data or {})

    @classmethod
    def block(
        cls,
        blocker: str,
        reason: str,
        data: dict[str, Any] | None = None,
    ) -> PolicyGateResult:
        return cls(False, blocker=blocker, reason=reason, data=data or {})


class EntryPolicy:
    def __init__(
        self,
        *,
        decision_freshness: Any | None = None,
        entry_priority: Any | None = None,
        entry_opportunity_score: Any | None = None,
        entry_profit_risk_sizing: Any | None = None,
        entry_price_guard: Any | None = None,
        entry_opportunity_gate: Any | None = None,
        high_risk_review_gate: Any | None = None,
    ) -> None:
        self.decision_freshness = decision_freshness
        self.entry_priority = entry_priority
        self.entry_opportunity_score = entry_opportunity_score
        self.entry_profit_risk_sizing = entry_profit_risk_sizing
        self.entry_price_guard = entry_price_guard
        self.entry_opportunity_gate = entry_opportunity_gate
        self.high_risk_review_gate_policy = high_risk_review_gate

    def score_candidate(
        self,
        decision: DecisionOutput,
        strategy: dict[str, Any] | None = None,
    ) -> float:
        if self.entry_opportunity_score is not None:
            return self.entry_opportunity_score.score_candidate(decision, strategy)
        raise RuntimeError("EntryPolicy requires entry_opportunity_score dependency")

    def ensure_opportunity_score(
        self,
        decision: DecisionOutput,
        strategy: dict[str, Any] | None = None,
    ) -> None:
        """Ensure every entry reaches execution with a computed opportunity score."""

        if not decision.is_entry:
            return
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score")
        if isinstance(opportunity, dict):
            try:
                if math.isfinite(float(opportunity.get("score"))):
                    return
            except (TypeError, ValueError):
                pass
        self.score_candidate(decision, strategy)

    @staticmethod
    def strategy_context_from_decision(decision: DecisionOutput) -> dict[str, Any]:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        context: dict[str, Any] = {}
        strategy_mode = raw.get("strategy_mode")
        if isinstance(strategy_mode, dict):
            context.update(strategy_mode)
        learning_context = raw.get("strategy_learning_context")
        if isinstance(learning_context, dict):
            context.update(learning_context)
        return context

    def immediate_execution_reason(self, decision: DecisionOutput) -> str | None:
        if self.entry_priority is None:
            return None
        return self.entry_priority.immediate_execution_reason(decision)

    def wait_sort_reason(
        self,
        decision: DecisionOutput,
        *,
        rank: int | None = None,
        candidate_count: int | None = None,
    ) -> str:
        if self.entry_priority is None:
            return "已进入开仓执行检查。"
        return self.entry_priority.wait_sort_reason(
            decision,
            rank=rank,
            candidate_count=candidate_count,
        )

    def gate_reason(self, decision: DecisionOutput) -> str | None:
        """Run exchange-safety checks without granting production permission."""

        if self.entry_opportunity_score is not None:
            self.ensure_opportunity_score(decision, self.strategy_context_from_decision(decision))
        if self.entry_opportunity_gate is not None:
            return self.entry_opportunity_gate.safety_reason(decision)
        return None

    def stale_decision_reason(self, decision: DecisionOutput) -> str | None:
        if self.decision_freshness is not None:
            return self.decision_freshness.stale_decision_reason(decision)
        return None

    async def pre_execution_price_guard_reason(self, decision: DecisionOutput) -> str | None:
        if self.entry_price_guard is None:
            return None
        return await self.entry_price_guard.guard_reason(decision)

    async def apply_profit_risk_sizing(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        if self.entry_profit_risk_sizing is not None:
            await self.entry_profit_risk_sizing.apply(
                decision,
                model_mode,
                open_positions=open_positions or [],
            )
            return
        raise RuntimeError("EntryPolicy requires entry_profit_risk_sizing dependency")

    async def high_risk_review_gate(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> str | None:
        if self.high_risk_review_gate_policy is None:
            return None
        return await self.high_risk_review_gate_policy.evaluate(
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
        context = EntryPipelineContext.from_inputs(
            decision=decision,
            model_name=model_name,
            model_mode=model_mode,
            open_positions=open_positions,
        )
        if not decision.is_entry:
            return PolicyGateResult.allow(
                {"intent": "not_entry", "pipeline_context": context.public_data()}
            )

        price_guard_reason = await self.pre_execution_price_guard_reason(decision)
        if price_guard_reason:
            return PolicyGateResult.block(
                "pre_execution_price_guard",
                price_guard_reason,
                {"pipeline_context": context.public_data()},
            )

        safety_reason = self.gate_reason(decision)
        if safety_reason:
            return PolicyGateResult.block(
                "entry_exchange_safety",
                safety_reason,
                {"pipeline_context": context.public_data()},
            )

        self.ensure_opportunity_score(decision, self.strategy_context_from_decision(decision))

        stale_reason = self.stale_decision_reason(decision)
        if stale_reason:
            return PolicyGateResult.block(
                "stale_decision",
                stale_reason,
                {"pipeline_context": context.public_data()},
            )

        await self.apply_profit_risk_sizing(
            decision,
            model_mode,
            open_positions=open_positions or [],
        )
        production_assessment = apply_production_entry_policy(decision)
        if not production_assessment.eligible:
            return PolicyGateResult.block(
                "production_return_policy",
                production_assessment.reason,
                {
                    "pipeline_context": context.public_data(),
                    "stage_status": "skipped",
                    "skip_kind": "production_return_policy",
                    "production_return_policy": production_assessment.to_dict(),
                },
            )

        return PolicyGateResult.allow(
            {
                "intent": "entry",
                "pipeline_context": context.public_data(),
                "production_return_policy": production_assessment.to_dict(),
            }
        )


class ExitPolicy:
    def __init__(
        self,
        *,
        exit_position_matcher: Any | None = None,
        exit_position_snapshot: Any | None = None,
    ) -> None:
        self.exit_position_matcher = exit_position_matcher
        self.exit_position_snapshot = exit_position_snapshot

    def has_matching_position(
        self,
        positions: list[dict[str, Any]] | None,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool:
        if self.exit_position_matcher is None:
            if not decision.is_exit:
                return True
            raise RuntimeError("ExitPolicy requires exit_position_matcher dependency")
        return self.exit_position_matcher.has_matching_position(
            positions,
            model_name,
            decision,
        )

    def no_matching_position_reason(self, decision: DecisionOutput) -> str:
        side_label = "多单" if decision.action == Action.CLOSE_LONG else "空单"
        return f"没有找到 {decision.symbol} 对应的可平{side_label}仓位，未向 OKX 提交平仓单。"

    async def evaluate(
        self,
        decision: DecisionOutput,
        model_name: str,
        open_positions: list[dict[str, Any]] | None,
        *,
        refresh_positions: bool = True,
    ) -> PolicyGateResult:
        context = ExitPipelineContext.from_inputs(
            decision=decision,
            model_name=model_name,
            open_positions=open_positions,
        )
        if not decision.is_exit:
            return PolicyGateResult.allow(
                {"intent": "not_exit", "pipeline_context": context.public_data()}
            )

        def gate_data() -> dict[str, Any]:
            return {
                "pipeline_context": context.public_data(),
            }

        exit_positions = open_positions or []
        if refresh_positions and self.exit_position_snapshot is not None:
            exit_positions = await self.exit_position_snapshot.refresh_positions(open_positions)
        context = context.with_refreshed_positions(exit_positions)
        if not self.has_matching_position(exit_positions, model_name, decision):
            exchange_has_position = False
            if self.exit_position_snapshot is not None:
                exchange_has_position = (
                    await self.exit_position_snapshot.has_matching_exchange_position(
                        model_name,
                        decision,
                    )
                )
            if exchange_has_position is None:
                return PolicyGateResult.block(
                    "exchange_position_snapshot_unavailable",
                    (
                        "OKX 持仓状态暂时查询失败，系统不能确认是否仍有可平仓仓位；"
                        "本轮不提交新的平仓单，等待下一轮同步确认。"
                    ),
                    gate_data(),
                )
            if exchange_has_position is False:
                return PolicyGateResult.block(
                    "no_matching_exit_position",
                    self.no_matching_position_reason(decision),
                    gate_data(),
                )

        dynamic_exit = apply_dynamic_exit(decision, exit_positions)
        if not dynamic_exit.eligible:
            return PolicyGateResult.block(
                "dynamic_exit_policy",
                dynamic_exit.reason,
                {**gate_data(), "dynamic_exit_policy": dynamic_exit.to_dict()},
            )

        return PolicyGateResult.allow(
            {
                "intent": "exit",
                "target_side": "long" if decision.action == Action.CLOSE_LONG else "short",
                "pipeline_context": context.public_data(),
                "dynamic_exit_policy": dynamic_exit.to_dict(),
            }
        )
