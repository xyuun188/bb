"""Decision-context builder and model caller for position review."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
    stronger_opportunity_context: dict[str, Any] = field(default_factory=dict)
    analysis_deadline_monotonic: float | None = None
    analysis_budget_seconds: float | None = None


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
        context = {
            "open_positions": request.open_positions,
            "trading_mode": request.trading_mode,
            "execution_mode": request.trading_mode,
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
            "stronger_opportunity": request.stronger_opportunity_context,
        }
        context["_consultation_reuse_key"] = self._consultation_reuse_key(request)
        context["_consultation_reuse_ttl_seconds"] = 120.0
        if request.analysis_deadline_monotonic is not None:
            context.update(
                {
                    "_analysis_deadline_monotonic": request.analysis_deadline_monotonic,
                    "_analysis_budget_scope": "position_review",
                    "_analysis_budget_seconds": request.analysis_budget_seconds,
                }
            )
        return context

    @staticmethod
    def _consultation_reuse_key(request: PositionReviewDecisionRequest) -> str:
        """Build a short-lived fingerprint for one position lifecycle state.

        The key deliberately contains lifecycle identity and fee/PnL state so a
        changed position cannot reuse a stale arbitration result. Prices are
        bucketed to avoid invalidating the result on every tiny tick.
        """

        def number(value: Any, default: float = 0.0) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return default
            return parsed if math.isfinite(parsed) else default

        def bucket_price(value: Any) -> float:
            price = number(value)
            if price <= 0:
                return 0.0
            # Use logarithmic buckets so the boundary is a true percentage
            # band.  Deriving the step from the same price would make
            # ``price / step`` constant and would silently disable bucketing.
            log_step = math.log1p(0.0025)
            bucket = round(math.log(price) / log_step)
            return round(math.exp(bucket * log_step), 12)

        def bucket_amount(value: Any, step: float) -> float:
            amount = number(value)
            if step <= 0:
                return round(amount, 8)
            return round(round(amount / step) * step, 8)

        def normalize_symbol(value: Any) -> str:
            text = str(value or "").upper().strip()
            if not text:
                return ""
            # CCXT may expose swap symbols as ``BTC/USDT:USDT`` while OKX
            # native records use ``BTC-USDT-SWAP``.  The suffix is a contract
            # namespace, not a different position lifecycle.
            text = text.split(":", 1)[0]
            text = text.replace("-SWAP", "")
            text = text.replace("-", "/")
            return text

        normalized = normalize_symbol(request.normalized_symbol or request.symbol)
        if not normalized:
            return ""

        candidates: list[dict[str, Any]] = []
        for item in request.open_positions:
            if not isinstance(item, dict):
                continue
            raw_info = item.get("info")
            info = raw_info if isinstance(raw_info, dict) else {}
            candidate_symbols = {
                normalize_symbol(value)
                for value in (
                    item.get("symbol"),
                    item.get("instId"),
                    item.get("instrument_id"),
                    info.get("instId"),
                )
            }
            candidate_symbols.discard("")
            if normalized in candidate_symbols:
                candidates.append(item)
        # A position review without an authoritative matching position must
        # not reuse another lifecycle's arbitration result.  It can still run
        # a fresh consultation; only the short-lived cache is disabled.
        if not candidates:
            return ""

        def field(
            position: dict[str, Any],
            info: dict[str, Any],
            management: dict[str, Any],
            *names: str,
        ) -> Any:
            for name in names:
                value = position.get(name)
                if value not in (None, ""):
                    return value
                value = info.get(name)
                if value not in (None, ""):
                    return value
                value = management.get(name)
                if value not in (None, ""):
                    return value
            return None

        position_states: list[dict[str, Any]] = []
        for position in candidates:
            raw_info = position.get("info")
            info = raw_info if isinstance(raw_info, dict) else {}
            raw_management = position.get("current_management_contract")
            management = raw_management if isinstance(raw_management, dict) else {}
            lifecycle = {
                key: field(position, info, management, key)
                for key in (
                    "lifecycle_id",
                    "position_id",
                    "local_position_id",
                    "id",
                    "pos_id",
                    "posId",
                    "okx_pos_id",
                    "okx_inst_id",
                    "open_time",
                    "opened_at",
                    "created_at",
                    "entry_time",
                    "open_order_id",
                    "entry_exchange_order_id",
                    "cTime",
                )
                if field(position, info, management, key) not in (None, "")
            }
            position_states.append(
                {
                    "side": str(
                        field(position, info, management, "side", "posSide") or ""
                    ).lower(),
                    "lifecycle": lifecycle,
                    "quantity": number(
                        field(position, info, management, "quantity", "base_quantity", "qty")
                    ),
                    "contracts": number(
                        field(position, info, management, "contracts", "pos", "size")
                    ),
                    "contract_size": number(
                        field(
                            position,
                            info,
                            management,
                            "contract_size",
                            "contractSize",
                            "ctVal",
                        )
                    ),
                    "entry_price": bucket_price(
                        field(position, info, management, "entry_price", "entryPrice", "avgPx")
                    ),
                    "current_price": bucket_price(
                        field(
                            position,
                            info,
                            management,
                            "current_price",
                            "mark_price",
                            "markPrice",
                            "markPx",
                        )
                    ),
                    # Money values are grouped into material bands so a tiny
                    # mark-price tick does not trigger a fresh deep consultation.
                    "unrealized_pnl": bucket_amount(
                        field(
                            position,
                            info,
                            management,
                            "unrealized_pnl",
                            "unrealized_pnl_usdt",
                            "unrealizedPnl",
                            "upl",
                        ),
                        0.05,
                    ),
                    "funding_fee": bucket_amount(
                        field(
                            position,
                            info,
                            management,
                            "funding_fee",
                            "funding_fee_amount",
                            "funding_fee_usdt",
                            "settled_funding_fee",
                        ),
                        0.01,
                    ),
                    "expected_future_funding": bucket_amount(
                        field(
                            position,
                            info,
                            management,
                            "expected_future_funding_cashflow",
                        ),
                        0.01,
                    ),
                    "fee": bucket_amount(
                        field(
                            position,
                            info,
                            management,
                            "fee",
                            "fees",
                            "entry_fee",
                            "entry_fee_usdt",
                        ),
                        0.01,
                    ),
                    "funding_evidence": {
                        key: field(position, info, management, key)
                        for key in (
                            "funding_fee_source",
                            "funding_bill_count",
                            "funding_evidence_complete",
                            "funding_evidence_eligible",
                            "funding_evidence_status",
                            "settled_funding_evidence_status",
                            "next_funding_time",
                        )
                        if field(position, info, management, key) not in (None, "")
                    },
                    "notional": bucket_amount(
                        field(
                            position,
                            info,
                            management,
                            "notional",
                            "notional_usd",
                            "notional_usdt",
                            "position_notional_usdt",
                            "position_value_usdt",
                            "notionalUsd",
                        ),
                        0.10,
                    ),
                    "liquidation_price": bucket_price(
                        field(
                            position,
                            info,
                            management,
                            "liquidation_price",
                            "liquidationPrice",
                            "liqPx",
                        )
                    ),
                }
            )
        position_states.sort(
            key=lambda state: json.dumps(
                state,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
        state = {
            "symbol": normalized,
            # Include every matching lifecycle (including hedged long/short
            # rows) so a change to any authoritative OKX row invalidates the
            # short-lived consultation reuse.
            "positions": position_states,
            "feature": {
                "current_price": bucket_price(getattr(request.feature_vector, "current_price", 0.0)),
                "funding_rate": round(number(getattr(request.feature_vector, "funding_rate", 0.0)), 8),
                "volatility_20": round(number(getattr(request.feature_vector, "volatility_20", 0.0)), 6),
                "atr_14": round(number(getattr(request.feature_vector, "atr_14", 0.0)), 8),
            },
            "market_regime": request.market_regime_context,
            "strategy_mode": request.strategy_mode_context,
        }
        payload = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _single_model_context(
        request: PositionReviewDecisionRequest,
    ) -> dict[str, Any]:
        return {
            "open_positions": request.open_positions,
            "trading_mode": request.trading_mode,
            "execution_mode": request.trading_mode,
            "review_positions": True,
            "position_entry_disabled": bool(request.position_entry_pause_reason),
            "position_entry_pause_reason": request.position_entry_pause_reason or "",
            "portfolio_profit_protection": request.portfolio_symbol_context,
            "position_profit_peak": request.position_profit_peak_context,
            "stronger_opportunity": request.stronger_opportunity_context,
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
        if request.stronger_opportunity_context:
            raw["stronger_opportunity"] = request.stronger_opportunity_context
        decision.raw_response = raw
        self.agent_skills_attacher(
            decision,
            phase="position_review",
            skills=position_agent_skills,
            note="持仓分析前的 Agent/Skills 证据快照。",
        )
