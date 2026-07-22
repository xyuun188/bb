"""Shadow backtest lifecycle and memory generation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from config.settings import FIXED_AI_MODEL_SLOTS, settings
from core.market_facts import (
    build_market_fact,
    build_shadow_market_fact_contract,
    compact_market_fact_contract,
    verify_market_fact_path,
)
from core.safe_output import safe_error_text
from core.training_contracts import (
    SHADOW_LABEL_VERSION,
    build_shadow_label_contract,
    compact_shadow_label_contract,
)
from db.repositories.memory_repo import MemoryRepository
from db.session import get_session_ctx
from services.execution_cost_model import execution_cost_estimate
from services.return_objective import RETURN_OBJECTIVE_NAME, RETURN_OBJECTIVE_VERSION
from services.shadow_training_quarantine import quarantine_completed_shadow_row

logger = structlog.get_logger(__name__)

SHADOW_BACKTEST_HORIZONS_MINUTES = (10, 30, 60)
SHADOW_LEVERAGE_SCENARIOS = (1, 2, 3, 5, 10)
SHADOW_LEVERAGE_COUNTERFACTUAL_VERSION = "2026-07-22.shadow-leverage-counterfactual.v1"

LatestPriceProvider = Callable[[str], Awaitable[float]]
LatestMarketFactProvider = Callable[[str], Awaitable[dict[str, Any]]]
PricePathProvider = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
SymbolNormalizer = Callable[[str | None], str]
FloatParser = Callable[[Any, float], float]
SessionFactory = Callable[[], Any]
RepositoryFactory = Callable[[Any], Any]
ExecutionCostFactsProvider = Callable[[str], Awaitable[dict[str, Any]]]

_SHADOW_TOOL_NAMES = (
    "profit_prediction",
    "time_series_prediction",
    "sentiment_analysis",
    "exit_advice",
)
_SHADOW_TOOL_KEYS = (
    "available",
    "status",
    "model",
    "primary_model",
    "challenger_model",
    "model_version",
    "route_mode",
    "fallback_reason",
    "best_side",
    "side",
    "direction",
    "expected_move_pct",
    "loss_probability",
    "profit_quality_score",
    "confidence",
    "specialist_inference_active",
    "specialist_primary_model",
    "specialist_challenger_model",
    "timesfm_shadow_expected_return_pct",
    "timesfm_shadow_expected_move_pct",
    "timesfm_shadow_side",
    "timesfm_shadow_confidence",
    "chronos_shadow_expected_return_pct",
    "chronos_shadow_expected_move_pct",
    "chronos_shadow_side",
    "chronos_shadow_confidence",
)
_SHADOW_PROFESSIONAL_KEYS = (
    "kind",
    "primary_model",
    "challenger_model",
    "artifacts_ready",
    "actual_inference",
    "baseline_response",
    "activation_blocker",
    "promotion_flow",
    "live_mutation",
)


def side_label(side: str) -> str:
    side_value = str(side).lower()
    if side_value == "long":
        return "做多"
    if side_value == "short":
        return "做空"
    return str(side)


def _safe_shadow_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _compact_shadow_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        number = _safe_shadow_number(value)
        return round(number, 8) if number is not None else None
    if isinstance(value, str):
        return value.strip()[:160]
    return None


def _compact_professional_shadow(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {}
    for key in _SHADOW_PROFESSIONAL_KEYS:
        if key not in value:
            continue
        item = _compact_shadow_value(value.get(key))
        if item is not None:
            compact[key] = item
    def compact_result_payload(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        compact_result = {}
        for key in (
            "model",
            "available",
            "actual_inference",
            "reason",
            "expected_return_pct",
            "expected_move_pct",
            "best_side",
            "direction",
            "confidence",
            "horizon_step",
            "sequence_length",
            "prediction_count",
        ):
            item = _compact_shadow_value(result.get(key))
            if item is not None:
                compact_result[key] = item
        return compact_result

    shadow_result = compact_result_payload(value.get("shadow_result"))
    if shadow_result:
        compact["shadow_result"] = shadow_result
    primary_shadow_result = compact_result_payload(value.get("primary_shadow_result"))
    if primary_shadow_result:
        compact["primary_shadow_result"] = primary_shadow_result
    challenger_shadow_result = compact_result_payload(value.get("challenger_shadow_result"))
    if challenger_shadow_result:
        compact["challenger_shadow_result"] = challenger_shadow_result
    predictions = value.get("predictions")
    if isinstance(predictions, dict):
        compact_predictions = {}
        for slot, prediction in list(predictions.items())[:4]:
            if not isinstance(prediction, dict):
                continue
            compact_prediction = {}
            for key in ("available", "reason", "score", "label", "text_count"):
                item = _compact_shadow_value(prediction.get(key))
                if item is not None:
                    compact_prediction[key] = item
            if compact_prediction:
                compact_predictions[str(slot)[:80]] = compact_prediction
        if compact_predictions:
            compact["predictions"] = compact_predictions
    return compact


def compact_local_ai_tools_shadow(local_ai_tools_context: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only auditable shadow evidence needed for later walk-forward scoring."""

    if not isinstance(local_ai_tools_context, dict):
        return {}
    compact: dict[str, Any] = {
        "status": str(local_ai_tools_context.get("status") or "")[:60],
        "captured_at": datetime.now(UTC).isoformat(),
    }
    for tool_name in _SHADOW_TOOL_NAMES:
        tool = local_ai_tools_context.get(tool_name)
        if not isinstance(tool, dict):
            continue
        item = {}
        for key in _SHADOW_TOOL_KEYS:
            if key not in tool:
                continue
            value = _compact_shadow_value(tool.get(key))
            if value is not None:
                item[key] = value
        professional = _compact_professional_shadow(tool.get("professional_model_shadow"))
        if professional:
            item["professional_model_shadow"] = professional
        if item:
            compact[tool_name] = item
    return compact if any(key in compact for key in _SHADOW_TOOL_NAMES) else {}


def shadow_fee_after_outcome(
    row: Any,
    *,
    long_return: float,
    short_return: float,
) -> dict[str, Any]:
    snapshot = getattr(row, "feature_snapshot", None)
    features = snapshot if isinstance(snapshot, dict) else {}
    execution_cost = execution_cost_estimate(features)
    funding_present = features.get("funding_data_available") is True
    funding_rate = _safe_shadow_number(features.get("funding_rate")) if funding_present else None
    funding_interval_minutes = _safe_shadow_number(features.get("funding_interval_minutes"))
    if funding_interval_minutes is None:
        funding_interval_hours = _safe_shadow_number(features.get("funding_interval_hours"))
        funding_interval_minutes = (
            funding_interval_hours * 60.0 if funding_interval_hours is not None else None
        )
    cost_complete = bool(
        execution_cost.production_eligible
        and funding_rate is not None
        and funding_present
        and funding_interval_minutes is not None
        and funding_interval_minutes > 0
    )
    horizon_minutes = max(int(getattr(row, "horizon_minutes", 0) or 0), 0)
    funding_drag_long_pct = (
        float(funding_rate or 0.0)
        * 100.0
        * horizon_minutes
        / float(funding_interval_minutes or 1.0)
    )
    funding_return_long_pct = -funding_drag_long_pct
    funding_return_short_pct = funding_drag_long_pct
    gross_long_pct = long_return * 100.0
    gross_short_pct = short_return * 100.0
    long_net_pct = (
        gross_long_pct
        - execution_cost.fee_pct
        - execution_cost.slippage_pct
        + funding_return_long_pct
        if cost_complete
        else None
    )
    short_net_pct = (
        gross_short_pct
        - execution_cost.fee_pct
        - execution_cost.slippage_pct
        + funding_return_short_pct
        if cost_complete
        else None
    )
    leverage_counterfactuals = [
        {
            "leverage": leverage,
            "long_fee_after_margin_return_pct": (
                round(float(long_net_pct) * leverage, 8)
                if long_net_pct is not None
                else None
            ),
            "short_fee_after_margin_return_pct": (
                round(float(short_net_pct) * leverage, 8)
                if short_net_pct is not None
                else None
            ),
            "long_gross_margin_return_pct": round(gross_long_pct * leverage, 8),
            "short_gross_margin_return_pct": round(gross_short_pct * leverage, 8),
            "approximate_full_margin_loss_move_pct": round(100.0 / leverage, 8),
            "creates_order": False,
        }
        for leverage in SHADOW_LEVERAGE_SCENARIOS
    ]
    return {
        "objective": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "cost_complete": cost_complete,
        "incomplete_reasons": [
            reason
            for condition, reason in (
                (execution_cost.production_eligible, execution_cost.reason),
                (funding_present, "funding_observation_missing"),
                (
                    funding_interval_minutes is not None
                    and funding_interval_minutes > 0,
                    "funding_interval_missing",
                ),
            )
            if not condition
        ],
        "cost_source": "dynamic_execution_estimate_from_shadow_snapshot",
        "fee_return_pct": execution_cost.fee_pct,
        "slippage_return_pct": execution_cost.slippage_pct,
        "funding_return_long_pct": funding_return_long_pct if funding_present else None,
        "funding_return_short_pct": funding_return_short_pct if funding_present else None,
        "funding_interval_minutes": funding_interval_minutes,
        "long_net_return_after_cost_pct": long_net_pct,
        "short_net_return_after_cost_pct": short_net_pct,
        "leverage_counterfactuals": leverage_counterfactuals,
        "leverage_counterfactual_policy": {
            "source": "one_cost_complete_shadow_path_without_duplicate_orders",
            "scenario_count": len(leverage_counterfactuals),
            "creates_order": False,
            "notional_return_is_shared_before_margin_leverage": True,
        },
        "execution_cost": execution_cost.to_dict(),
    }


def compact_shadow_leverage_counterfactuals(
    outcome: dict[str, Any] | None,
) -> dict[str, Any]:
    source = outcome if isinstance(outcome, dict) else {}
    rows = source.get("leverage_counterfactuals")
    rows = rows if isinstance(rows, list) else []
    compact: dict[str, Any] = {
        "version": SHADOW_LEVERAGE_COUNTERFACTUAL_VERSION,
        "source": "one_cost_complete_shadow_path_without_duplicate_orders",
        "creates_order": False,
        "scenario_count": len(rows),
    }
    for row in rows:
        item = row if isinstance(row, dict) else {}
        leverage = int(_safe_shadow_number(item.get("leverage")) or 0)
        if leverage not in SHADOW_LEVERAGE_SCENARIOS:
            continue
        prefix = f"leverage_{leverage}x"
        for target, source_key in (
            ("long_fee_after_margin_return_pct", "long_fee_after_margin_return_pct"),
            ("short_fee_after_margin_return_pct", "short_fee_after_margin_return_pct"),
            ("approximate_full_margin_loss_move_pct", "approximate_full_margin_loss_move_pct"),
        ):
            compact[f"{prefix}_{target}"] = _safe_shadow_number(item.get(source_key))
    return compact


@dataclass(slots=True)
class ShadowBacktestService:
    """Record delayed market outcomes and convert strong results into memory."""

    latest_price_provider: LatestPriceProvider
    symbol_normalizer: SymbolNormalizer
    float_parser: FloatParser
    session_factory: SessionFactory = get_session_ctx
    repository_factory: RepositoryFactory = MemoryRepository
    execution_cost_facts_provider: ExecutionCostFactsProvider | None = None
    latest_market_fact_provider: LatestMarketFactProvider | None = None
    price_path_provider: PricePathProvider | None = None
    horizons_minutes: tuple[int, ...] = SHADOW_BACKTEST_HORIZONS_MINUTES
    fixed_model_slots: list[dict[str, Any]] = field(
        default_factory=lambda: list(FIXED_AI_MODEL_SLOTS)
    )

    async def create(
        self,
        decision_id: int | None,
        decision: DecisionOutput,
        feature_vector: Any,
        execution_mode: str,
        analysis_type: str = "market",
        local_ai_tools_context: dict[str, Any] | None = None,
    ) -> None:
        """Record pending shadow samples for market-analysis decisions."""
        if analysis_type != "market":
            return
        entry_price = self.float_parser(
            getattr(feature_vector, "current_price", 0.0)
            or getattr(feature_vector, "close", 0.0)
            or (decision.feature_snapshot or {}).get("current_price"),
            0.0,
        )
        if entry_price <= 0:
            return

        now = datetime.now(UTC)
        try:
            async with self.session_factory() as session:
                repo = self.repository_factory(session)
                feature_snapshot = (
                    decision.feature_snapshot or getattr(feature_vector, "to_dict", lambda: {})()
                )
                if not isinstance(feature_snapshot, dict):
                    feature_snapshot = {}
                else:
                    feature_snapshot = dict(feature_snapshot)
                local_ai_shadow = compact_local_ai_tools_shadow(local_ai_tools_context)
                if local_ai_shadow:
                    feature_snapshot["local_ai_tools_shadow"] = local_ai_shadow
                entry_fact = feature_snapshot.get("market_fact")
                if not isinstance(entry_fact, dict):
                    entry_fact = build_market_fact(
                        decision.symbol,
                        {
                            **feature_snapshot,
                            "last_price": entry_price,
                            "source": "legacy_shadow_entry_snapshot",
                        },
                        contract_spec=feature_snapshot.get("contract_spec"),
                    )
                    feature_snapshot["market_fact"] = entry_fact
                market_contract = build_shadow_market_fact_contract(entry_fact, None, None)
                feature_snapshot["market_fact_contract"] = market_contract
                feature_snapshot["training_market_fact_contract"] = (
                    compact_market_fact_contract(market_contract)
                )
                for horizon in self.horizons_minutes:
                    await repo.create_shadow_backtest(
                        {
                            "decision_id": decision_id,
                            "model_name": decision.model_name,
                            "execution_mode": execution_mode,
                            "symbol": decision.symbol,
                            "analysis_type": analysis_type,
                            "decision_action": decision.action.value,
                            "decision_confidence": float(decision.confidence or 0.0),
                            "entry_price": entry_price,
                            "feature_snapshot": feature_snapshot,
                            "raw_llm_response": (
                                decision.raw_response
                                if isinstance(decision.raw_response, dict)
                                else {}
                            ),
                            "status": "pending",
                            "due_at": now + timedelta(minutes=int(horizon)),
                            "horizon_minutes": int(horizon),
                            "label_version": SHADOW_LABEL_VERSION,
                        }
                    )
        except Exception as exc:
            logger.debug(
                "failed to create shadow backtests",
                symbol=decision.symbol,
                error=safe_error_text(exc),
            )

    async def update_due(self, limit: int = 200) -> int:
        """Complete due samples without holding a database session during OKX reads."""
        try:
            async with self.session_factory() as session:
                repo = self.repository_factory(session)
                rows = await repo.get_due_shadow_backtests(limit=max(1, int(limit or 1)))
            if not rows:
                return 0

            execution_cost_facts: dict[str, dict[str, Any]] = {}
            if self.execution_cost_facts_provider is not None:
                execution_modes = sorted(
                    {
                        "live"
                        if str(getattr(row, "execution_mode", "paper")).lower() == "live"
                        else "paper"
                        for row in rows
                    }
                )
                for execution_mode in execution_modes:
                    try:
                        facts = await self.execution_cost_facts_provider(execution_mode)
                    except Exception as exc:
                        logger.warning(
                            "shadow execution cost fact refresh failed",
                            mode=execution_mode,
                            error=safe_error_text(exc),
                        )
                        facts = {}
                    execution_cost_facts[execution_mode] = (
                        dict(facts) if isinstance(facts, dict) else {}
                    )

            # Price collection can wait on an exchange request.  Keep it outside the
            # ORM context so low-priority shadow maintenance cannot exhaust the pool.
            market_fact_cache: dict[str, dict[str, Any]] = {}
            completions: dict[int, dict[str, Any]] = {}
            for row in rows:
                row_id = int(getattr(row, "id", 0) or 0)
                if row_id <= 0:
                    continue
                symbol = self.symbol_normalizer(row.symbol) or row.symbol
                if symbol not in market_fact_cache:
                    if self.latest_market_fact_provider is not None:
                        try:
                            fact = await self.latest_market_fact_provider(symbol)
                        except Exception as exc:
                            logger.warning(
                                "shadow result market fact unavailable",
                                symbol=symbol,
                                error=safe_error_text(exc),
                            )
                            fact = {}
                    else:
                        price = await self.latest_price_provider(symbol)
                        fact = build_market_fact(
                            symbol,
                            {
                                "last_price": price,
                                "bid": price,
                                "ask": price,
                                "timestamp": datetime.now(UTC),
                                "source": "legacy_price_only_observation",
                                "source_endpoint": "legacy_latest_price_provider",
                                "source_channel": "price_only",
                            },
                        )
                    market_fact_cache[symbol] = dict(fact) if isinstance(fact, dict) else {}
                result_fact = market_fact_cache.get(symbol, {})
                result_prices = (
                    result_fact.get("prices")
                    if isinstance(result_fact.get("prices"), dict)
                    else {}
                )
                actual_price = self.float_parser(result_prices.get("last"), 0.0)
                entry_price = self.float_parser(row.entry_price, 0.0)
                if actual_price <= 0 or entry_price <= 0:
                    continue

                execution_mode = (
                    "live"
                    if str(getattr(row, "execution_mode", "paper")).lower() == "live"
                    else "paper"
                )
                feature_snapshot = getattr(row, "feature_snapshot", None)
                feature_snapshot = (
                    dict(feature_snapshot) if isinstance(feature_snapshot, dict) else {}
                )
                entry_fact = feature_snapshot.get("market_fact")
                if not isinstance(entry_fact, dict):
                    entry_fact = build_market_fact(
                        symbol,
                        {
                            **feature_snapshot,
                            "last_price": entry_price,
                            "source": "legacy_shadow_entry_snapshot",
                        },
                        contract_spec=feature_snapshot.get("contract_spec"),
                    )
                    feature_snapshot["market_fact"] = entry_fact
                if self.price_path_provider is not None:
                    try:
                        price_path = await self.price_path_provider(entry_fact, result_fact)
                    except Exception as exc:
                        logger.warning(
                            "shadow native price path unavailable",
                            symbol=symbol,
                            error=safe_error_text(exc),
                        )
                        price_path = verify_market_fact_path(entry_fact, result_fact, [])
                else:
                    price_path = verify_market_fact_path(entry_fact, result_fact, [])
                market_contract = build_shadow_market_fact_contract(
                    entry_fact,
                    result_fact,
                    price_path,
                )
                feature_snapshot["market_fact_contract"] = market_contract
                feature_snapshot["training_market_fact_contract"] = (
                    compact_market_fact_contract(market_contract)
                )
                current_cost_facts = execution_cost_facts.get(execution_mode, {})
                if _safe_shadow_number(current_cost_facts.get("taker_fee_rate")):
                    feature_snapshot.update(current_cost_facts)
                row.feature_snapshot = feature_snapshot

                long_return = (actual_price - entry_price) / entry_price
                short_return = (entry_price - actual_price) / entry_price
                fee_after_outcome = shadow_fee_after_outcome(
                    row,
                    long_return=long_return,
                    short_return=short_return,
                )
                feature_snapshot["training_leverage_counterfactuals"] = (
                    compact_shadow_leverage_counterfactuals(fee_after_outcome)
                )
                long_net = fee_after_outcome.get("long_net_return_after_cost_pct")
                short_net = fee_after_outcome.get("short_net_return_after_cost_pct")
                best_action = "hold"
                if fee_after_outcome.get("cost_complete"):
                    if float(long_net) > 0.0 and float(long_net) >= float(short_net):
                        best_action = "long"
                    elif float(short_net) > 0.0 and float(short_net) > float(long_net):
                        best_action = "short"

                decision_action = str(row.decision_action or "hold")
                missed = decision_action == "hold" and best_action in {"long", "short"}
                feature_snapshot["training_label_contract"] = compact_shadow_label_contract(
                    build_shadow_label_contract(
                        shadow_backtest_id=row_id,
                        decision_id=getattr(row, "decision_id", None),
                        horizon_minutes=int(row.horizon_minutes),
                        long_return_pct=long_return * 100.0,
                        short_return_pct=short_return * 100.0,
                        best_action=best_action,
                        market_fact_contract=feature_snapshot.get(
                            "training_market_fact_contract"
                        ),
                        cost_facts=fee_after_outcome,
                        label_timestamp=getattr(row, "due_at", None),
                    )
                )
                completions[row_id] = {
                    "actual_price": actual_price,
                    "long_return": long_return,
                    "short_return": short_return,
                    "best_action": best_action,
                    "missed": missed,
                    "fee_after_outcome": fee_after_outcome,
                    "feature_snapshot": feature_snapshot,
                    "note": self._completion_note(
                        decision_action,
                        best_action,
                        int(row.horizon_minutes),
                        long_return,
                        short_return,
                        missed,
                    ),
                }

            if not completions:
                return 0

            memory_requests: list[tuple[Any, dict[str, Any]]] = []
            async with self.session_factory() as session:
                repo = self.repository_factory(session)
                reload_rows = getattr(repo, "get_pending_shadow_backtests_by_ids", None)
                if callable(reload_rows):
                    writable_rows = await reload_rows(list(completions))
                else:
                    # Keep isolated test doubles and external repository adapters working.
                    writable_rows = rows
                completed_count = 0
                for row in writable_rows:
                    completion = completions.get(int(getattr(row, "id", 0) or 0))
                    if completion is None:
                        continue
                    row.feature_snapshot = completion["feature_snapshot"]
                    await repo.complete_shadow_backtest(
                        row,
                        actual_price=completion["actual_price"],
                        long_return_pct=completion["long_return"] * 100,
                        short_return_pct=completion["short_return"] * 100,
                        best_action=completion["best_action"],
                        missed_opportunity=completion["missed"],
                        note=completion["note"],
                    )
                    completed_count += 1
                    quarantine_result = quarantine_completed_shadow_row(row)
                    if quarantine_result.get("applied"):
                        logger.info(
                            "shadow backtest quarantined from training",
                            shadow_backtest_id=getattr(row, "id", None),
                            symbol=getattr(row, "symbol", None),
                            reasons=quarantine_result.get("reasons"),
                        )
                        continue
                    if settings.shadow_memory_enabled:
                        memory_requests.append((row, completion))

            # Shadow outcomes are authoritative training facts. Observation-only
            # memory enrichment must never roll their completed transaction back.
            for row, completion in memory_requests:
                try:
                    async with self.session_factory() as memory_session:
                        memory_repo = self.repository_factory(memory_session)
                        await self._record_memory_in_session(
                            memory_repo,
                            row,
                            long_return=completion["long_return"],
                            short_return=completion["short_return"],
                            best_action=completion["best_action"],
                            fee_after_outcome=completion["fee_after_outcome"],
                        )
                except Exception as exc:
                    logger.warning(
                        "failed to record shadow observation memory",
                        shadow_backtest_id=getattr(row, "id", None),
                        error=safe_error_text(exc),
                    )
            logger.info("shadow backtests updated", count=completed_count)
            return completed_count
        except Exception as exc:
            logger.warning("failed to update shadow backtests", error=safe_error_text(exc))
            return 0

    def _completion_note(
        self,
        decision_action: str,
        best_action: str,
        horizon_minutes: int,
        long_return: float,
        short_return: float,
        missed: bool,
    ) -> str:
        if missed:
            return (
                f"当时观望，但 {horizon_minutes} 分钟后"
                f"{side_label(best_action)}方向收益约"
                f"{max(long_return, short_return) * 100:.2f}%。"
            )
        if (
            decision_action in {"long", "short"}
            and decision_action != best_action
            and best_action != "hold"
        ):
            return f"实际更优方向是 {side_label(best_action)}，用于后续复盘。"
        return ""

    async def _record_memory_in_session(
        self,
        repo: Any,
        row: Any,
        *,
        long_return: float,
        short_return: float,
        best_action: str,
        fee_after_outcome: dict[str, Any],
    ) -> None:
        """Turn shadow backtest outcomes into small, reusable expert memories."""
        decision_action = str(getattr(row, "decision_action", "") or "hold")
        symbol = str(getattr(row, "symbol", "") or "")
        horizon = int(getattr(row, "horizon_minutes", 0) or 0)
        if not symbol or horizon <= 0:
            return
        if fee_after_outcome.get("cost_complete") is not True:
            return

        if decision_action == "hold" and best_action in {"long", "short"}:
            realized_net_pct = self.float_parser(
                fee_after_outcome.get(f"{best_action}_net_return_after_cost_pct"),
                0.0,
            )
            if realized_net_pct <= 0.0:
                return
            memory_type = "shadow_missed_opportunity"
            side = best_action
            success_count = 0
            failure_count = 0
            outcome_text = (
                f"当时选择观望，但 {horizon} 分钟后"
                f"{side_label(side)}方向估算费后收益约 {realized_net_pct:.2f}%。"
            )
            recommended = "shadow_observation_only"
        elif decision_action in {"long", "short"}:
            side = decision_action
            realized_net_pct = self.float_parser(
                fee_after_outcome.get(f"{side}_net_return_after_cost_pct"),
                0.0,
            )
            if realized_net_pct > 0.0:
                memory_type = "shadow_good_signal"
                success_count = 0
                failure_count = 0
                outcome_text = (
                    f"影子复盘显示：{side_label(side)}信号在 {horizon} 分钟后"
                    f"估算费后收益约 {realized_net_pct:.2f}%，仅保留为待验证观察。"
                )
                recommended = "shadow_observation_only"
            elif realized_net_pct < 0.0:
                memory_type = "shadow_bad_signal"
                success_count = 0
                failure_count = 0
                opposite = "short" if side == "long" else "long"
                opposite_net_pct = self.float_parser(
                    fee_after_outcome.get(f"{opposite}_net_return_after_cost_pct"),
                    0.0,
                )
                outcome_text = (
                    f"影子复盘显示：{side_label(side)}信号在 {horizon} 分钟后"
                    f"估算费后亏损约 {abs(realized_net_pct):.2f}%，而"
                    f"{side_label(opposite)}方向估算费后收益约 {opposite_net_pct:.2f}%。"
                )
                recommended = "shadow_risk_observation_only"
            else:
                return
        else:
            return

        feature_snapshot = getattr(row, "feature_snapshot", None) or {}
        pattern = self._memory_pattern(feature_snapshot, symbol, side, horizon)
        labels = {slot["name"]: slot.get("label", slot["name"]) for slot in self.fixed_model_slots}
        for expert_name, lesson in self._expert_lessons(
            symbol=symbol,
            side=side,
            memory_type=memory_type,
            outcome_text=outcome_text,
        ).items():
            await repo.upsert_memory(
                {
                    "expert_name": expert_name,
                    "expert_label": labels.get(expert_name, expert_name),
                    "symbol": symbol,
                    "side": side,
                    "memory_type": memory_type,
                    "market_pattern": pattern,
                    "lesson": lesson,
                    "recommended_action": recommended,
                    "evidence_count": 1,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "memory_key": (
                        f"{expert_name}|shadow_correlated_path|"
                        f"{getattr(row, 'decision_id', None) or getattr(row, 'id', None)}|"
                        f"{symbol}|{side}|{self._feature_bucket(feature_snapshot)}"
                    ),
                    "extra": {
                        "source": "shadow_backtest",
                        "shadow_backtest_id": getattr(row, "id", None),
                        "decision_id": getattr(row, "decision_id", None),
                        "decision_action": decision_action,
                        "best_action": best_action,
                        "horizon_minutes": horizon,
                        "entry_price": getattr(row, "entry_price", None),
                        "actual_price": getattr(row, "actual_price", None),
                        "long_return_pct": long_return * 100,
                        "short_return_pct": short_return * 100,
                        "net_return_after_cost_pct": realized_net_pct,
                        "objective": RETURN_OBJECTIVE_NAME,
                        "objective_version": RETURN_OBJECTIVE_VERSION,
                        "cost_complete": True,
                        "production_evidence_eligible": False,
                        "correlation_group": (
                            f"shadow_decision:{getattr(row, 'decision_id', None) or getattr(row, 'id', None)}"
                        ),
                        "fee_after_outcome": fee_after_outcome,
                    },
                }
            )

    def _expert_lessons(
        self,
        *,
        symbol: str,
        side: str,
        memory_type: str,
        outcome_text: str,
    ) -> dict[str, str]:
        label = side_label(side)
        if memory_type == "shadow_missed_opportunity":
            return {
                "trend_expert": (
                    f"{symbol} {label}机会曾被观望错过。{outcome_text}"
                    "当方向结构、ADX、均线和 MACD 同向时，可以提高方向支持，但不能直接决定仓位。"
                ),
                "momentum_expert": (
                    f"{symbol} {label}机会曾被观望错过。{outcome_text}"
                    "如果预期净收益、手续费覆盖和亏损概率都合格，可以支持小仓位盈利质量试单。"
                ),
                "sentiment_expert": (
                    f"{symbol} {label}机会曾被观望错过。{outcome_text}"
                    "如果 1/5/10/30 分钟路径和事件冲击风险有利，可以支持更早执行。"
                ),
                "risk_expert": (
                    f"{symbol} {label}机会曾被观望错过。{outcome_text}"
                    "没有硬风险时，优先用仓位和杠杆控制风险，不要直接否决交易。"
                ),
            }
        if memory_type == "shadow_good_signal":
            return {
                "trend_expert": (
                    f"{symbol} {label}信号被影子复盘验证有效。{outcome_text}"
                    "下次出现相似方向结构时，可以适当提高方向信心。"
                ),
                "momentum_expert": (
                    f"{symbol} {label}信号被影子复盘验证有效。{outcome_text}"
                    "当扣费后预期净收益和盈亏质量仍为正时，可以支持执行。"
                ),
                "sentiment_expert": (
                    f"{symbol} {label}信号被影子复盘验证有效。{outcome_text}"
                    "短周期路径延续相似时，可以支持当前执行时机。"
                ),
                "risk_expert": (
                    f"{symbol} {label}信号被影子复盘验证有效。{outcome_text}"
                    "没有硬风险时，可以允许小仓位执行。"
                ),
            }
        return {
            "trend_expert": (
                f"{symbol} {label}信号在影子复盘中表现偏弱。{outcome_text}"
                "下次必须先看到趋势延续，再提高方向信心。"
            ),
            "momentum_expert": (
                f"{symbol} {label}信号在影子复盘中表现偏弱。{outcome_text}"
                "追单前要检查预期净收益、手续费覆盖和盈亏比是否过弱。"
            ),
            "sentiment_expert": (
                f"{symbol} {label}信号在影子复盘中表现偏弱。{outcome_text}"
                "执行前要确认短周期路径是否已经反转。"
            ),
            "risk_expert": (
                f"{symbol} {label}信号在影子复盘中表现偏弱。{outcome_text}"
                "相似条件下需要降低仓位/杠杆，必要时阻止新开仓。"
            ),
        }

    def _memory_pattern(
        self,
        feature_snapshot: dict[str, Any],
        symbol: str,
        side: str,
        horizon: int,
    ) -> str:
        return (
            f"{symbol} {side_label(side)}影子复盘 {horizon}分钟，"
            f"ADX={self.float_parser(feature_snapshot.get('adx_14'), 0.0):.1f}，"
            f"量比={self.float_parser(feature_snapshot.get('volume_ratio'), 0.0):.2f}，"
            f"5周期收益={self.float_parser(feature_snapshot.get('returns_5'), 0.0) * 100:.2f}%，"
            f"盘口倾斜={self.float_parser(feature_snapshot.get('orderbook_imbalance'), 0.0):.2f}"
        )

    def _feature_bucket(self, feature_snapshot: dict[str, Any]) -> str:
        del feature_snapshot
        return "continuous_market_features"
