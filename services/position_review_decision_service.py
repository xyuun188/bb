"""Decision-context builder and model caller for position review."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_brain.base_model import DecisionOutput

ExpertMemoryContextProvider = Callable[[str], Awaitable[dict[str, Any]]]
MLSignalPredictor = Callable[[Any], dict[str, Any] | Awaitable[dict[str, Any]]]
LocalAIToolsContextProvider = Callable[..., Awaitable[dict[str, Any]]]
PositionSkillsProvider = Callable[..., list[Any]]
AgentSkillsAttacher = Callable[..., dict[str, Any]]
EnsembleDecider = Callable[[Any, dict[str, Any]], Awaitable[tuple[Any, Any]]]
ModelProvider = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class PositionReviewDecisionRequest:
    """Inputs required to ask a model for a position-review decision."""

    model_name: str
    symbol: str
    normalized_symbol: str
    feature_vector: Any
    open_positions: list[dict[str, Any]]
    trading_mode: str
    position_entry_pause_reason: str | None
    market_regime_context: dict[str, Any]
    strategy_mode_context: dict[str, Any]
    portfolio_symbol_context: dict[str, Any]
    position_profit_peak_context: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PositionReviewDecisionResult:
    """Result returned by the position-review model-calling boundary."""

    decision: Any
    analysis_started: datetime
    ml_signal_context: dict[str, Any]
    local_ai_tools_context: dict[str, Any]
    position_agent_skills: list[Any]


class PositionReviewDecisionService:
    """Build position-review context and call the appropriate model."""

    def __init__(
        self,
        *,
        default_model_name: str,
        expert_memory_context_provider: ExpertMemoryContextProvider,
        ml_signal_predictor: MLSignalPredictor,
        local_ai_tools_context_provider: LocalAIToolsContextProvider,
        position_skills_provider: PositionSkillsProvider,
        agent_skills_attacher: AgentSkillsAttacher,
        ensemble_decider: EnsembleDecider,
        model_provider: ModelProvider,
        pre_agent_skills_rollback: bool = False,
        local_quant_prompt_enabled: bool = True,
    ) -> None:
        self.default_model_name = default_model_name
        self.expert_memory_context_provider = expert_memory_context_provider
        self.ml_signal_predictor = ml_signal_predictor
        self.local_ai_tools_context_provider = local_ai_tools_context_provider
        self.position_skills_provider = position_skills_provider
        self.agent_skills_attacher = agent_skills_attacher
        self.ensemble_decider = ensemble_decider
        self.model_provider = model_provider
        self.pre_agent_skills_rollback = pre_agent_skills_rollback
        self.local_quant_prompt_enabled = local_quant_prompt_enabled

    async def decide(
        self,
        request: PositionReviewDecisionRequest,
    ) -> PositionReviewDecisionResult | None:
        """Build context, call the model, and attach position-review metadata."""

        memory_context = await self.expert_memory_context_provider(
            request.normalized_symbol or request.symbol
        )
        ml_signal_result = self.ml_signal_predictor(request.feature_vector)
        ml_signal_context = (
            await ml_signal_result if inspect.isawaitable(ml_signal_result) else ml_signal_result
        )
        local_ai_tools_context = await self.local_ai_tools_context_provider(
            request.feature_vector,
            ml_signal_context,
            open_positions=request.open_positions,
            include_exit_advice=True,
        )
        position_agent_skills = self.position_skills_provider(
            position_entry_pause_reason=request.position_entry_pause_reason,
            ml_signal=ml_signal_context,
            local_ai_tools=local_ai_tools_context,
            portfolio_profit_protection=request.portfolio_symbol_context,
        )
        analysis_started = datetime.now(UTC)
        decision = await self._call_model(
            request,
            memory_context=memory_context,
            ml_signal_context=ml_signal_context,
            local_ai_tools_context=local_ai_tools_context,
        )
        if decision is None:
            return None
        if isinstance(decision, DecisionOutput):
            self._attach_review_metadata(
                decision,
                request=request,
                position_agent_skills=position_agent_skills,
            )
        return PositionReviewDecisionResult(
            decision=decision,
            analysis_started=analysis_started,
            ml_signal_context=ml_signal_context,
            local_ai_tools_context=local_ai_tools_context,
            position_agent_skills=position_agent_skills,
        )

    async def _call_model(
        self,
        request: PositionReviewDecisionRequest,
        *,
        memory_context: dict[str, Any],
        ml_signal_context: dict[str, Any],
        local_ai_tools_context: dict[str, Any],
    ) -> Any:
        if request.model_name == self.default_model_name:
            decision, _opinions = await self.ensemble_decider(
                request.feature_vector,
                self._ensemble_context(
                    request,
                    memory_context=memory_context,
                    ml_signal_context=ml_signal_context,
                    local_ai_tools_context=local_ai_tools_context,
                ),
            )
            return decision

        model = self.model_provider(request.model_name)
        if model is None:
            return None
        return await model.decide(
            request.feature_vector,
            self._single_model_context(request),
        )

    def _ensemble_context(
        self,
        request: PositionReviewDecisionRequest,
        *,
        memory_context: dict[str, Any],
        ml_signal_context: dict[str, Any],
        local_ai_tools_context: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "open_positions": request.open_positions,
            "trading_mode": request.trading_mode,
            "review_positions": True,
            "position_entry_disabled": bool(request.position_entry_pause_reason),
            "position_entry_pause_reason": request.position_entry_pause_reason or "",
            **memory_context,
            "market_regime": request.market_regime_context,
            "strategy_mode": request.strategy_mode_context,
            "ml_signal": {} if self.pre_agent_skills_rollback else ml_signal_context,
            "local_ai_tools": ({} if self.pre_agent_skills_rollback else local_ai_tools_context),
            "ml_signal_prompt_enabled": self.local_quant_prompt_enabled,
            "local_ai_tools_prompt_enabled": self.local_quant_prompt_enabled,
            "portfolio_profit_protection": request.portfolio_symbol_context,
            "position_profit_peak": request.position_profit_peak_context,
        }

    @staticmethod
    def _single_model_context(
        request: PositionReviewDecisionRequest,
    ) -> dict[str, Any]:
        return {
            "open_positions": request.open_positions,
            "trading_mode": request.trading_mode,
            "review_positions": True,
            "position_entry_disabled": bool(request.position_entry_pause_reason),
            "position_entry_pause_reason": request.position_entry_pause_reason or "",
            "portfolio_profit_protection": request.portfolio_symbol_context,
            "position_profit_peak": request.position_profit_peak_context,
        }

    def _attach_review_metadata(
        self,
        decision: DecisionOutput,
        *,
        request: PositionReviewDecisionRequest,
        position_agent_skills: list[Any],
    ) -> None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["analysis_type"] = "position_review"
        raw["review_positions"] = True
        if request.portfolio_symbol_context.get("active"):
            raw["portfolio_profit_protection"] = request.portfolio_symbol_context
        if request.position_profit_peak_context:
            raw["position_profit_peak"] = request.position_profit_peak_context
        decision.raw_response = raw
        self.agent_skills_attacher(
            decision,
            phase="position_review",
            skills=position_agent_skills,
            note="持仓分析前的 Agent/Skills 证据快照。",
        )
