"""Shadow backtest lifecycle and training-label generation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from core.market_facts import (
    build_market_fact,
    build_shadow_market_fact_contract,
    compact_market_fact_contract,
    market_fact_reasons,
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

SHADOW_BACKTEST_HORIZONS_MINUTES = (5, 15, 60, 240)
SHADOW_LEVERAGE_SCENARIOS = (1, 2, 3, 5, 10)
SHADOW_LEVERAGE_COUNTERFACTUAL_VERSION = "2026-07-22.shadow-leverage-counterfactual.v1"
# Shadow maintenance is deliberately bounded: it must make progress without
# competing with the trading loop for the exchange client or event-loop time.
SHADOW_RESULT_FACT_CONCURRENCY = 8
SHADOW_RESULT_FACT_TIMEOUT_SECONDS = 3.0
SHADOW_RESULT_FACT_BATCH_TIMEOUT_SECONDS = 6.0
# A timed-out market-fact build is kept alive by DataService single-flight.
# Do not hammer the same symbol every maintenance round while that refresh is
# still pending; retry after a short bounded cooldown instead.
SHADOW_RESULT_FACT_RETRY_COOLDOWN_SECONDS = 15.0
SHADOW_RESULT_COST_FACT_TIMEOUT_SECONDS = 5.0
SHADOW_RESULT_COST_FACT_BATCH_TIMEOUT_SECONDS = 6.0
SHADOW_RESULT_PATH_CONCURRENCY = 2
SHADOW_RESULT_PATH_TIMEOUT_SECONDS = 30.0
SHADOW_RESULT_PATH_BATCH_TIMEOUT_SECONDS = 60.0
SHADOW_RESULT_PATH_RETRY_COUNT = 1
SHADOW_RESULT_PATH_RETRY_DELAY_SECONDS = 0.05

LatestPriceProvider = Callable[[str], Awaitable[float]]
LatestMarketFactProvider = Callable[[str], Awaitable[dict[str, Any]]]
PricePathProvider = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]
SymbolNormalizer = Callable[[str | None], str]
FloatParser = Callable[[Any, float], float]
SessionFactory = Callable[[], Any]
RepositoryFactory = Callable[[Any], Any]
ExecutionCostFactsProvider = Callable[[str], Awaitable[dict[str, Any]]]


def _consume_async_task(task: asyncio.Task[Any]) -> None:
    """Drain a detached task so a slow provider cannot emit an unhandled error."""

    try:
        task.result()
    except BaseException:
        return


async def _bounded_task_gather(
    awaitables: Iterable[Awaitable[Any]],
    *,
    budget_seconds: float,
) -> tuple[list[Any], bool]:
    """Collect finished work without waiting for cancellation-resistant providers."""

    tasks = [asyncio.create_task(awaitable) for awaitable in awaitables]
    if not tasks:
        return [], False
    done, pending = await asyncio.wait(
        tasks,
        timeout=max(float(budget_seconds or 0.0), 0.0),
    )
    for task in pending:
        task.cancel()
        task.add_done_callback(_consume_async_task)
    results: list[Any] = []
    for task in done:
        try:
            results.append(task.result())
        except BaseException as exc:
            results.append(exc)
    return results, bool(pending)

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
)


def side_label(side: str) -> str:
    side_value = str(side).lower()
    if side_value == "long":
        return "做多"
    if side_value == "short":
        return "做空"
    return str(side)


def shadow_path_labels(
    *,
    entry_price: float,
    price_path: dict[str, Any] | None,
    stop_loss_fraction: float | None,
    take_profit_fraction: float | None = None,
) -> dict[str, Any]:
    """Derive direction-neutral path labels from the verified native 1m path."""

    path = price_path if isinstance(price_path, dict) else {}
    path_low = _safe_shadow_number(path.get("path_low"))
    path_high = _safe_shadow_number(path.get("path_high"))
    if entry_price <= 0 or path_low is None or path_high is None:
        return {}
    long_mfe = max((path_high - entry_price) / entry_price * 100.0, 0.0)
    long_mae = max((entry_price - path_low) / entry_price * 100.0, 0.0)
    short_mfe = long_mae
    short_mae = long_mfe
    stop = _safe_shadow_number(stop_loss_fraction)
    stop = stop if stop is not None and stop > 0.0 else None
    take = _safe_shadow_number(take_profit_fraction)
    take = take if take is not None and take > 0.0 else None

    def first_touch(side: str) -> tuple[bool, bool, str]:
        stop_triggered = False
        take_triggered = False
        first = "none"
        bars = path.get("_ordered_bar_ranges")
        if not isinstance(bars, list) or stop is None or take is None:
            return stop_triggered, take_triggered, first
        for raw_bar in bars:
            bar = raw_bar if isinstance(raw_bar, dict) else {}
            low = _safe_shadow_number(bar.get("low"))
            high = _safe_shadow_number(bar.get("high"))
            if low is None or high is None:
                continue
            if side == "long":
                touches_stop = low <= entry_price * (1.0 - stop)
                touches_take = high >= entry_price * (1.0 + take)
            else:
                touches_stop = high >= entry_price * (1.0 + stop)
                touches_take = low <= entry_price * (1.0 - take)
            stop_triggered = stop_triggered or touches_stop
            take_triggered = take_triggered or touches_take
            if first == "none" and touches_stop and touches_take:
                first = "path_uncertain"
            elif first == "none" and touches_stop:
                first = "stop_loss"
            elif first == "none" and touches_take:
                first = "take_profit"
        return stop_triggered, take_triggered, first

    long_stop, long_take, long_first = first_touch("long")
    short_stop, short_take, short_first = first_touch("short")
    return {
        "long_mfe_pct": long_mfe,
        "long_mae_pct": long_mae,
        "short_mfe_pct": short_mfe,
        "short_mae_pct": short_mae,
        "long_stop_loss_triggered": (
            long_stop
            if stop is not None and take is not None
            else path_low <= entry_price * (1.0 - stop)
            if stop is not None
            else False
        ),
        "short_stop_loss_triggered": (
            short_stop
            if stop is not None and take is not None
            else path_high >= entry_price * (1.0 + stop)
            if stop is not None
            else False
        ),
        "long_take_profit_triggered": long_take,
        "short_take_profit_triggered": short_take,
        "long_first_touch": long_first,
        "short_first_touch": short_first,
    }


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


def _model_shadow_action(
    raw_response: dict[str, Any],
    *,
    primary_action: str,
) -> tuple[str, str]:
    """Persist the model's explicit side so rules-canary returns stay separate."""

    if "model_shadow_decision" not in raw_response:
        action = str(primary_action or "").strip().lower()
        if action in {"buy", "open_long"}:
            action = "long"
        elif action in {"sell", "open_short"}:
            action = "short"
        return action if action in {"long", "short"} else "", "primary_model_decision"
    shadow = raw_response.get("model_shadow_decision")
    action = str(shadow.get("action") or "").strip().lower() if isinstance(shadow, dict) else ""
    if action in {"buy", "open_long"}:
        action = "long"
    elif action in {"sell", "open_short"}:
        action = "short"
    return action if action in {"long", "short"} else "", "explicit_model_shadow_decision"


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
        "long_net_return_after_all_cost_pct": long_net_pct,
        "short_net_return_after_all_cost_pct": short_net_pct,
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
    """Record delayed market outcomes for shadow training and evaluation."""

    latest_price_provider: LatestPriceProvider
    symbol_normalizer: SymbolNormalizer
    float_parser: FloatParser
    session_factory: SessionFactory = get_session_ctx
    repository_factory: RepositoryFactory = MemoryRepository
    execution_cost_facts_provider: ExecutionCostFactsProvider | None = None
    latest_market_fact_provider: LatestMarketFactProvider | None = None
    price_path_provider: PricePathProvider | None = None
    horizons_minutes: tuple[int, ...] = SHADOW_BACKTEST_HORIZONS_MINUTES
    _market_fact_retry_after: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    async def create(
        self,
        decision_id: int | None,
        decision: DecisionOutput,
        feature_vector: Any,
        execution_mode: str,
        analysis_type: str = "market",
        local_ai_tools_context: dict[str, Any] | None = None,
    ) -> bool:
        """Record pending shadow samples for market-analysis decisions."""
        if analysis_type != "market":
            return False
        entry_price = self.float_parser(
            getattr(feature_vector, "current_price", 0.0)
            or getattr(feature_vector, "close", 0.0)
            or (decision.feature_snapshot or {}).get("current_price"),
            0.0,
        )
        if entry_price <= 0:
            return False

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
                raw_response = (
                    decision.raw_response
                    if isinstance(decision.raw_response, dict)
                    else {}
                )
                shadow_action, shadow_source = _model_shadow_action(
                    raw_response,
                    primary_action=decision.action.value,
                )
                feature_snapshot["model_shadow_action"] = shadow_action
                feature_snapshot["model_shadow_action_source"] = shadow_source
                feature_snapshot["shadow_label_inputs"] = {
                    "version": SHADOW_LABEL_VERSION,
                    "stop_loss_fraction": _safe_shadow_number(
                        decision.stop_loss_pct
                    ),
                    "take_profit_fraction": _safe_shadow_number(
                        decision.take_profit_pct
                    ),
                    "horizons_minutes": list(self.horizons_minutes),
                }
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
                            "raw_llm_response": raw_response,
                            "status": "pending",
                            "due_at": now + timedelta(minutes=int(horizon)),
                            "horizon_minutes": int(horizon),
                            "label_version": SHADOW_LABEL_VERSION,
                        }
                    )
            return True
        except Exception as exc:
            logger.warning(
                "failed to create shadow backtests; caller will retry",
                symbol=decision.symbol,
                decision_id=decision_id,
                error=safe_error_text(exc),
            )
            raise

    async def recover_missing_market_samples(
        self,
        *,
        lookback_minutes: int = 180,
        limit: int = 200,
    ) -> dict[str, int]:
        """Rebuild missing/partial samples from durable AI decisions.

        Market decisions are persisted before this method is called.  A fresh
        session reads only decisions with fewer than the configured horizons and
        then reuses ``create`` so duplicate recovery is harmless.
        """

        since = datetime.now(UTC) - timedelta(minutes=max(int(lookback_minutes or 1), 1))
        async with self.session_factory() as session:
            repo = self.repository_factory(session)
            finder = getattr(repo, "get_market_decisions_missing_shadow_samples", None)
            if not callable(finder):
                return {"scanned": 0, "recovered": 0, "failed": 0}
            rows = await finder(
                since=since,
                horizon_count=len(self.horizons_minutes),
                limit=max(int(limit or 1), 1),
            )

        recovered = 0
        failed = 0
        for row in rows:
            raw_action = str(getattr(row, "action", "hold") or "hold")
            decision = DecisionOutput(
                model_name=str(getattr(row, "model_name", "") or "ensemble_trader"),
                symbol=str(getattr(row, "symbol", "") or ""),
                action=Action.from_string(raw_action),
                confidence=float(getattr(row, "confidence", 0.0) or 0.0),
                reasoning=str(getattr(row, "reasoning", "") or ""),
                position_size_pct=float(getattr(row, "position_size_pct", 0.0) or 0.0),
                suggested_leverage=float(getattr(row, "suggested_leverage", 1.0) or 1.0),
                stop_loss_pct=float(getattr(row, "stop_loss_pct", 0.0) or 0.0),
                take_profit_pct=float(getattr(row, "take_profit_pct", 0.0) or 0.0),
                timestamp=getattr(row, "created_at", None) or datetime.now(UTC),
                raw_response=(
                    dict(getattr(row, "raw_llm_response", {}) or {})
                    if isinstance(getattr(row, "raw_llm_response", None), dict)
                    else {}
                ),
                feature_snapshot=(
                    dict(getattr(row, "feature_snapshot", {}) or {})
                    if isinstance(getattr(row, "feature_snapshot", None), dict)
                    else {}
                ),
            )
            try:
                created = await self.create(
                    int(getattr(row, "id", 0) or 0),
                    decision,
                    decision.feature_snapshot or {},
                    "paper" if bool(getattr(row, "is_paper", True)) else "live",
                    analysis_type="market",
                )
                recovered += 1 if created else 0
            except Exception:
                failed += 1
        return {"scanned": len(rows), "recovered": recovered, "failed": failed}

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

                async def fetch_execution_cost_facts(
                    execution_mode: str,
                ) -> tuple[str, dict[str, Any]]:
                    try:
                        facts = await asyncio.wait_for(
                            self.execution_cost_facts_provider(execution_mode),
                            timeout=SHADOW_RESULT_COST_FACT_TIMEOUT_SECONDS,
                        )
                    except Exception as exc:
                        logger.warning(
                            "shadow execution cost fact refresh failed",
                            mode=execution_mode,
                            error=safe_error_text(exc),
                        )
                        facts = {}
                    return execution_mode, dict(facts) if isinstance(facts, dict) else {}

                cost_results, cost_timed_out = await _bounded_task_gather(
                    (fetch_execution_cost_facts(mode) for mode in execution_modes),
                    budget_seconds=SHADOW_RESULT_COST_FACT_BATCH_TIMEOUT_SECONDS,
                )
                if cost_timed_out:
                    logger.warning(
                        "shadow execution cost fact batch timed out",
                        mode_count=len(execution_modes),
                        timeout_seconds=SHADOW_RESULT_COST_FACT_BATCH_TIMEOUT_SECONDS,
                    )
                for item in cost_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        execution_mode, facts = item
                        execution_cost_facts[str(execution_mode)] = (
                            dict(facts) if isinstance(facts, dict) else {}
                        )

            # Price collection can wait on an exchange request. Keep it outside the
            # ORM context so low-priority shadow maintenance cannot exhaust the pool.
            # Fetch each symbol once and in parallel; the old row-by-row loop made a
            # 25-row batch exceed the 30-second trading-service maintenance budget.
            market_fact_cache: dict[str, dict[str, Any]] = {}
            symbols = {
                self.symbol_normalizer(getattr(row, "symbol", ""))
                or str(getattr(row, "symbol", "") or "")
                for row in rows
            }
            symbols.discard("")
            fact_semaphore = asyncio.Semaphore(max(1, SHADOW_RESULT_FACT_CONCURRENCY))

            async def fallback_market_fact(
                symbol: str,
                *,
                source: str,
            ) -> dict[str, Any]:
                if self.latest_price_provider is None:
                    return {}
                price = await asyncio.wait_for(
                    self.latest_price_provider(symbol),
                    timeout=SHADOW_RESULT_FACT_TIMEOUT_SECONDS,
                )
                return build_market_fact(
                    symbol,
                    {
                        "last_price": price,
                        "bid": price,
                        "ask": price,
                        "timestamp": datetime.now(UTC),
                        "source": source,
                        "source_endpoint": "legacy_latest_price_provider",
                        "source_channel": "price_only",
                    },
                )

            async def fetch_market_fact(symbol: str) -> tuple[str, dict[str, Any]]:
                async with fact_semaphore:
                    retry_after = self._market_fact_retry_after.get(symbol, 0.0)
                    if retry_after > asyncio.get_running_loop().time():
                        return symbol, {}
                    try:
                        if self.latest_market_fact_provider is not None:
                            fact = await asyncio.wait_for(
                                self.latest_market_fact_provider(symbol),
                                timeout=SHADOW_RESULT_FACT_TIMEOUT_SECONDS,
                            )
                        else:
                            fact = await fallback_market_fact(
                                symbol,
                                source="legacy_price_only_observation",
                            )
                        normalized = dict(fact) if isinstance(fact, dict) else {}
                        if self.latest_market_fact_provider is not None:
                            reasons = market_fact_reasons(normalized)
                            if reasons:
                                logger.info(
                                    "shadow result market fact not clean; keeping sample pending",
                                    symbol=symbol,
                                    reasons=reasons,
                                )
                                return symbol, {}
                        prices = normalized.get("prices")
                        last_price = (
                            self.float_parser(prices.get("last"), 0.0)
                            if isinstance(prices, dict)
                            else 0.0
                        )
                        # A websocket/cache price is a safe diagnostic fallback. The
                        # resulting market-fact contract remains marked legacy or
                        # incomplete and is quarantined from training as needed.
                        if (
                            self.latest_market_fact_provider is None
                            and last_price <= 0
                            and self.latest_price_provider is not None
                        ):
                            normalized = await fallback_market_fact(
                                symbol,
                                source="legacy_price_fallback_after_market_fact_timeout",
                            )
                        return symbol, normalized
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "shadow result market fact unavailable",
                            symbol=symbol,
                            error=safe_error_text(exc),
                        )
                        self._market_fact_retry_after[symbol] = (
                            asyncio.get_running_loop().time()
                            + SHADOW_RESULT_FACT_RETRY_COOLDOWN_SECONDS
                        )
                        if self.latest_market_fact_provider is not None:
                            return symbol, {}
                        try:
                            return symbol, await fallback_market_fact(
                                symbol,
                                source="legacy_price_fallback_after_market_fact_error",
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as fallback_exc:
                            logger.warning(
                                "shadow result market fact fallback unavailable",
                                symbol=symbol,
                                error=safe_error_text(fallback_exc),
                            )
                            return symbol, {}

            if symbols:
                fact_results, facts_timed_out = await _bounded_task_gather(
                    (fetch_market_fact(symbol) for symbol in sorted(symbols)),
                    budget_seconds=SHADOW_RESULT_FACT_BATCH_TIMEOUT_SECONDS,
                )
                if facts_timed_out:
                    logger.warning(
                        "shadow result market fact batch timed out",
                        symbol_count=len(symbols),
                        timeout_seconds=SHADOW_RESULT_FACT_BATCH_TIMEOUT_SECONDS,
                    )
                for item in fact_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        symbol, fact = item
                        market_fact_cache[str(symbol)] = (
                            dict(fact) if isinstance(fact, dict) else {}
                        )

            path_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
            path_semaphore = asyncio.Semaphore(max(1, SHADOW_RESULT_PATH_CONCURRENCY))

            async def fetch_path(
                row: Any,
                symbol: str,
                entry_fact: dict[str, Any],
                result_fact: dict[str, Any],
            ) -> tuple[tuple[str, int, int], dict[str, Any]]:
                entry_ms = int(entry_fact.get("source_timestamp_ms") or 0)
                result_ms = int(result_fact.get("source_timestamp_ms") or 0)
                cache_key = (symbol, entry_ms, result_ms)
                async with path_semaphore:
                    if self.price_path_provider is None:
                        return cache_key, verify_market_fact_path(entry_fact, result_fact, [])
                    last_error: Exception | None = None
                    for attempt in range(SHADOW_RESULT_PATH_RETRY_COUNT + 1):
                        try:
                            path = await asyncio.wait_for(
                                self.price_path_provider(entry_fact, result_fact),
                                timeout=SHADOW_RESULT_PATH_TIMEOUT_SECONDS,
                            )
                            return cache_key, dict(path) if isinstance(path, dict) else {}
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            last_error = exc
                            if attempt < SHADOW_RESULT_PATH_RETRY_COUNT:
                                await asyncio.sleep(SHADOW_RESULT_PATH_RETRY_DELAY_SECONDS)
                                continue
                    logger.warning(
                        "shadow native price path unavailable",
                        symbol=symbol,
                        shadow_backtest_id=getattr(row, "id", None),
                        attempts=SHADOW_RESULT_PATH_RETRY_COUNT + 1,
                        error=safe_error_text(last_error) if last_error else "unknown_error",
                    )
                    return cache_key, verify_market_fact_path(entry_fact, result_fact, [])

            path_tasks: list[Awaitable[tuple[tuple[str, int, int], dict[str, Any]]]] = []
            path_task_rows: list[tuple[Any, str, dict[str, Any], dict[str, Any], tuple[str, int, int]]] = []
            for row in rows:
                symbol = self.symbol_normalizer(row.symbol) or row.symbol
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
                entry_ms = int(entry_fact.get("source_timestamp_ms") or 0)
                result_ms = int(result_fact.get("source_timestamp_ms") or 0)
                cache_key = (symbol, entry_ms, result_ms)
                path_task_rows.append((row, symbol, entry_fact, result_fact, cache_key))
                if cache_key not in path_cache:
                    path_tasks.append(fetch_path(row, symbol, entry_fact, result_fact))
            if path_tasks:
                path_results, paths_timed_out = await _bounded_task_gather(
                    path_tasks,
                    budget_seconds=SHADOW_RESULT_PATH_BATCH_TIMEOUT_SECONDS,
                )
                if paths_timed_out:
                    logger.warning(
                        "shadow result price path batch timed out",
                        path_count=len(path_tasks),
                        timeout_seconds=SHADOW_RESULT_PATH_BATCH_TIMEOUT_SECONDS,
                    )
                for item in path_results:
                    if isinstance(item, tuple) and len(item) == 2:
                        cache_key, path = item
                        path_cache[cache_key] = dict(path) if isinstance(path, dict) else {}

            completions: dict[int, dict[str, Any]] = {}
            for row, _symbol, entry_fact, result_fact, cache_key in path_task_rows:
                row_id = int(getattr(row, "id", 0) or 0)
                if row_id <= 0:
                    continue
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
                feature_snapshot.setdefault("market_fact", entry_fact)
                price_path = path_cache.get(cache_key) or verify_market_fact_path(
                    entry_fact, result_fact, []
                )
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
                label_inputs = feature_snapshot.get("shadow_label_inputs")
                label_inputs = label_inputs if isinstance(label_inputs, dict) else {}
                path_labels = shadow_path_labels(
                    entry_price=entry_price,
                    price_path=price_path,
                    stop_loss_fraction=_safe_shadow_number(
                        label_inputs.get("stop_loss_fraction")
                    ),
                    take_profit_fraction=_safe_shadow_number(
                        label_inputs.get("take_profit_fraction")
                    ),
                )
                long_net = fee_after_outcome.get("long_net_return_after_all_cost_pct")
                short_net = fee_after_outcome.get("short_net_return_after_all_cost_pct")
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
                        **path_labels,
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
