"""Paper-only bootstrap lifecycle for collecting model-bound trade evidence.

This module never grants production permission.  It selects an already governed
canary artifact for OKX demo sampling, applies a small independent risk budget,
and fails closed when frequency, portfolio, or loss-streak guards are unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil, isclose, isfinite
from typing import Any

from sqlalchemy import select

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from db.session import get_read_session_ctx
from models.decision import AIDecision
from services.entry_profit_risk_sizing import (
    build_portfolio_risk_snapshot,
    select_okx_leverage_tier,
)

PAPER_BOOTSTRAP_CANARY_VERSION = "2026-07-19.paper-bootstrap-canary.v2"
PAPER_BOOTSTRAP_SIZING_VERSION = "2026-07-19.paper-bootstrap-sizing.v2"
PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION = "2026-07-19.paper-bootstrap-position-lifecycle.v1"
PAPER_BOOTSTRAP_MAX_OPEN_POSITIONS = 2
PAPER_BOOTSTRAP_MIN_DAILY_ENTRIES = 4
PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES = 24
PAPER_BOOTSTRAP_TARGET_AUTHORITATIVE_SAMPLES = 200
PAPER_BOOTSTRAP_COLLECTION_HORIZON_DAYS = 14
PAPER_BOOTSTRAP_SINGLE_TRADE_EQUITY_RISK = 0.0005
PAPER_BOOTSTRAP_PORTFOLIO_EQUITY_RISK = 0.001
PAPER_BOOTSTRAP_DAILY_LOSS_EQUITY_RISK = 0.003
PAPER_BOOTSTRAP_STRATUM_IMBALANCE_TOLERANCE = 2
PAPER_BOOTSTRAP_MIN_FILL_DRIFT_RESERVE_FRACTION = 0.0025
PAPER_BOOTSTRAP_PREFLIGHT_TIMEOUT_SECONDS = 3.0
PAPER_BOOTSTRAP_HISTORY_TIMEOUT_SECONDS = 2.5
PAPER_BOOTSTRAP_PREPARE_TIMEOUT_SECONDS = 15.0
PAPER_BOOTSTRAP_EXCHANGE_FACTS_TIMEOUT_SECONDS = 8.0
PAPER_BOOTSTRAP_BALANCE_TIMEOUT_SECONDS = 3.0
PAPER_BOOTSTRAP_DEADLINE_RESERVE_SECONDS = 0.05
PAPER_BOOTSTRAP_CANARY_HISTORY_START = datetime(2026, 7, 17, tzinfo=UTC)

BalanceProvider = Callable[[str, DecisionOutput], Awaitable[float | None]]
ExchangeFactsProvider = Callable[
    [str, DecisionOutput, list[dict[str, Any]]],
    Awaitable[dict[str, Any]],
]
HistoryProvider = Callable[[], Awaitable[list[Any]]]


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Retrieve late cancellation results without extending the caller deadline."""

    if task.cancelled():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


async def _bounded_policy_call(
    stage: str,
    provider: Callable[[], Awaitable[Any]],
    *,
    fallback: Any,
    timeout_seconds: float,
    deadline_monotonic: float,
    timeout_reason: str,
    budget_reason: str,
    unavailable_reason_prefix: str,
    message_zh: str,
) -> tuple[Any, dict[str, Any], str | None]:
    """Run one policy evidence provider without crossing its shared deadline."""

    loop = asyncio.get_running_loop()
    started = loop.time()
    requested_timeout = max(float(timeout_seconds or 0.0), 0.0)
    remaining_at_start = max(float(deadline_monotonic) - started, 0.0)
    allowed_timeout = min(
        requested_timeout,
        max(remaining_at_start - PAPER_BOOTSTRAP_DEADLINE_RESERVE_SECONDS, 0.0),
    )
    timing: dict[str, Any] = {
        "stage": stage,
        "status": "pending",
        "requested_timeout_seconds": round(requested_timeout, 6),
        "allowed_timeout_seconds": round(allowed_timeout, 6),
        "remaining_seconds_at_start": round(remaining_at_start, 6),
        "message_zh": message_zh,
    }
    reason: str | None = None
    task: asyncio.Task[Any] | None = None
    try:
        if allowed_timeout <= 0.0:
            timing["status"] = "budget_exhausted"
            reason = budget_reason
            return fallback, timing, reason

        task = asyncio.create_task(provider())
        done, pending = await asyncio.wait({task}, timeout=allowed_timeout)
        if pending:
            task.cancel()
            task.add_done_callback(_consume_task_result)
            timing["status"] = "timeout"
            reason = timeout_reason
            return fallback, timing, reason

        timing["status"] = "ok"
        return task.result(), timing, None
    except asyncio.CancelledError:
        if task is not None and not task.done():
            task.cancel()
            task.add_done_callback(_consume_task_result)
        raise
    except Exception as exc:
        timing["status"] = "error"
        timing["error_type"] = type(exc).__name__
        reason = f"{unavailable_reason_prefix}:{type(exc).__name__}"
        return fallback, timing, reason
    finally:
        timing["duration_seconds"] = round(max(loop.time() - started, 0.0), 6)
        timing["remaining_seconds_at_end"] = round(
            max(float(deadline_monotonic) - loop.time(), 0.0),
            6,
        )


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _positive(value: Any) -> float:
    result = _finite(value)
    return max(result or 0.0, 0.0)


def _normalized_ratio(value: Any) -> float:
    result = _positive(value)
    return result / 100.0 if result > 1.0 else result


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_raw(row: Any) -> dict[str, Any]:
    return _safe_dict(
        _row_value(row, "raw_llm_response")
        or _row_value(row, "raw_response")
        or _row_value(row, "decision_learning_snapshot")
    )


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _sampling_stratum(context: dict[str, Any], side: str) -> dict[str, str]:
    features = _safe_dict(context.get("sampling_features"))
    symbol = str(context.get("sampling_symbol") or features.get("symbol") or "unknown")
    volatility = _normalized_ratio(features.get("volatility_20"))
    if volatility <= 0:
        price = max(_positive(features.get("current_price")), _positive(features.get("close")))
        volatility = _positive(features.get("atr_14")) / price if price > 0 else 0.0
    volatility_bucket = (
        "high" if volatility >= 0.03 else "medium" if volatility >= 0.01 else "low"
    )
    strategy = _safe_dict(context.get("strategy_mode"))
    explicit_regime = str(
        context.get("market_regime")
        or strategy.get("market_regime")
        or strategy.get("regime")
        or ""
    ).strip().lower()
    if explicit_regime:
        regime = explicit_regime
    else:
        returns_20 = abs(_finite(features.get("returns_20")) or 0.0)
        regime = (
            "volatile"
            if volatility >= 0.03
            else "trending"
            if returns_20 >= 0.01
            else "ranging"
        )
    normalized_side = side if side in {"long", "short"} else "unknown"
    return {
        "symbol": symbol,
        "side": normalized_side,
        "volatility_bucket": volatility_bucket,
        "market_regime": regime,
        "key": "|".join((symbol, normalized_side, volatility_bucket, regime)),
    }


def build_paper_canary_position_lifecycle(decision: Any) -> dict[str, Any]:
    """Build an expiry contract for an actually executed paper canary entry."""

    raw = _row_raw(decision)
    contract = _safe_dict(raw.get("paper_bootstrap_canary"))
    action = str(_row_value(decision, "action") or "").lower()
    executed_at = _as_utc(_row_value(decision, "executed_at"))
    is_paper = bool(_row_value(decision, "is_paper"))
    was_executed = bool(_row_value(decision, "was_executed"))
    selected = _safe_dict(contract.get("selected_observation"))
    horizon_minutes = int(_positive(selected.get("horizon_minutes")) or 0)
    if (
        contract.get("version") != PAPER_BOOTSTRAP_CANARY_VERSION
        or contract.get("authorized") is not True
        or contract.get("requested") is not True
        or contract.get("execution_scope") != "paper_only"
        or contract.get("production_permission") is not False
        or not is_paper
        or not was_executed
        or action not in {"long", "short"}
        or executed_at is None
        or horizon_minutes <= 0
    ):
        return {}
    expires_at = executed_at + timedelta(minutes=horizon_minutes)
    return {
        "version": PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION,
        "kind": "paper_bootstrap_canary_position",
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "decision_id": _row_value(decision, "id"),
        "symbol": str(_row_value(decision, "symbol") or ""),
        "side": action,
        "executed_at": executed_at.isoformat(),
        "horizon_minutes": horizon_minutes,
        "expires_at": expires_at.isoformat(),
        "artifact_version": contract.get("artifact_version"),
        "source_contract_version": PAPER_BOOTSTRAP_CANARY_VERSION,
    }


def paper_canary_position_lifecycle(position: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable canary lifecycle from runtime or persisted state."""
    direct = _safe_dict(position.get("paper_canary_lifecycle"))
    if direct:
        return direct
    management = _safe_dict(position.get("current_management_contract"))
    return _safe_dict(management.get("paper_canary_lifecycle"))


def _is_open_paper_canary_position(position: dict[str, Any]) -> bool:
    lifecycle = paper_canary_position_lifecycle(position)
    return bool(
        position.get("is_open", True) is not False
        and _positive(position.get("quantity", 1.0)) > 0
        and str(position.get("execution_mode") or "paper").lower() == "paper"
        and lifecycle.get("version") == PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION
        and lifecycle.get("authorized") is True
        and lifecycle.get("execution_scope") == "paper_only"
        and lifecycle.get("production_permission") is False
    )


def assess_paper_canary_position_horizon(
    position: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess whether an attached paper canary position reached its model horizon."""

    lifecycle = paper_canary_position_lifecycle(position)
    current = _as_utc(now) or datetime.now(UTC)
    expires_at_value = lifecycle.get("expires_at")
    try:
        expires_at = datetime.fromisoformat(str(expires_at_value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        expires_at = None
    expires_at = _as_utc(expires_at)
    position_side = str(position.get("side") or "").lower()
    lifecycle_side = str(lifecycle.get("side") or "").lower()
    position_symbol = (
        str(position.get("symbol") or "")
        .upper()
        .replace("-", "")
        .replace("/", "")
        .replace(":USDT", "")
    )
    lifecycle_symbol = (
        str(lifecycle.get("symbol") or "")
        .upper()
        .replace("-", "")
        .replace("/", "")
        .replace(":USDT", "")
    )
    authorized = bool(
        lifecycle.get("version") == PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION
        and lifecycle.get("kind") == "paper_bootstrap_canary_position"
        and lifecycle.get("authorized") is True
        and lifecycle.get("execution_scope") == "paper_only"
        and lifecycle.get("production_permission") is False
        and str(position.get("execution_mode") or "").lower() == "paper"
        and position_side in {"long", "short"}
        and lifecycle_side == position_side
        and bool(position_symbol)
        and position_symbol == lifecycle_symbol
        and int(_positive(lifecycle.get("horizon_minutes")) or 0) > 0
        and expires_at is not None
    )
    return {
        "authorized": authorized,
        "elapsed": bool(authorized and expires_at is not None and current >= expires_at),
        "horizon_minutes": int(_positive(lifecycle.get("horizon_minutes")) or 0),
        "expires_at": expires_at.isoformat() if expires_at is not None else None,
        "decision_id": lifecycle.get("decision_id"),
        "version": lifecycle.get("version"),
    }


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_distribution(signal: dict[str, Any], side: str) -> dict[str, Any]:
    predictions = _safe_list(signal.get("predictions"))
    primary = _safe_dict(predictions[0] if predictions else {})
    contracts = _safe_dict(
        primary.get("return_distribution_contract") or signal.get("return_distribution_contract")
    )
    return _safe_dict(contracts.get(side))


def select_paper_bootstrap_candidate(context: dict[str, Any] | None) -> dict[str, Any]:
    """Select a paper sample by governed information value, not production PnL claims."""

    ctx = _safe_dict(context)
    blocked = {
        "version": PAPER_BOOTSTRAP_CANARY_VERSION,
        "authorized": False,
        "requested": False,
        "execution_scope": "paper_only",
        "production_permission": False,
    }
    if str(ctx.get("trading_mode") or "").lower() != "paper":
        return {**blocked, "reason": "paper_execution_mode_required"}

    candidate_evidence = _safe_dict(ctx.get("entry_candidate_evidence"))
    production_count = sum(
        int(_positive(_safe_dict(candidate_evidence.get(side)).get("production_source_count")))
        for side in ("long", "short")
    )
    if production_count > 0:
        return {**blocked, "reason": "production_return_source_available"}

    signal = _safe_dict(ctx.get("ml_signal"))
    readiness = _safe_dict(signal.get("paper_canary"))
    if (
        signal.get("paper_canary_authorized") is not True
        or signal.get("artifact_lifecycle") != "canary"
        or readiness.get("authorized") is not True
        or readiness.get("execution_scope") != "paper_only"
        or readiness.get("production_permission") is not False
    ):
        return {**blocked, "reason": "governed_paper_canary_artifact_unavailable"}

    eligible_sides = {
        str(side).lower()
        for side in _safe_list(readiness.get("eligible_sides"))
        if str(side).lower() in {"long", "short"}
    }
    observations: list[dict[str, Any]] = []
    for side in sorted(eligible_sides):
        distribution = _model_distribution(signal, side)
        raw_expected = _finite(distribution.get("raw_expected_return_pct"))
        objective_expected = _finite(distribution.get("objective_expected_return_pct"))
        lower_quantile = _finite(distribution.get("lower_quantile_return_pct"))
        dispersion = _positive(distribution.get("dispersion_pct"))
        if raw_expected is None or objective_expected is None or lower_quantile is None:
            continue
        execution_cost = _safe_dict(_safe_dict(candidate_evidence.get(side)).get("execution_cost"))
        current_cost = _positive(execution_cost.get("total_pct"))
        sampling_stratum = _sampling_stratum(ctx, side)
        observations.append(
            {
                "side": side,
                "raw_expected_return_pct": raw_expected,
                "objective_expected_return_pct": objective_expected,
                "lower_quantile_return_pct": lower_quantile,
                "dispersion_pct": dispersion,
                "current_execution_cost_pct": current_cost,
                "observed_net_return_pct": raw_expected - current_cost,
                "horizon_minutes": int(_positive(distribution.get("horizon_minutes")) or 10),
                "distribution_member_count": int(
                    _positive(distribution.get("distribution_member_count"))
                ),
                "source_authority": distribution.get("source_authority"),
                "sampling_stratum": sampling_stratum,
            }
        )
    if not observations:
        return {**blocked, "reason": "paper_canary_distribution_incomplete"}

    observations.sort(
        key=lambda item: (
            item["objective_expected_return_pct"],
            item["observed_net_return_pct"],
            item["lower_quantile_return_pct"],
        ),
        reverse=True,
    )
    selected = observations[0]
    if len(observations) > 1:
        score_gap = (
            selected["objective_expected_return_pct"]
            - observations[1]["objective_expected_return_pct"]
        )
        if score_gap <= 0:
            return {**blocked, "reason": "paper_canary_direction_not_identifiable"}
    else:
        score_gap = abs(selected["objective_expected_return_pct"])
    uncertainty = max(
        selected["dispersion_pct"],
        abs(selected["objective_expected_return_pct"] - selected["lower_quantile_return_pct"]),
    )
    confidence = score_gap / max(score_gap + uncertainty, 1e-12)
    now = datetime.now(UTC)
    return {
        "version": PAPER_BOOTSTRAP_CANARY_VERSION,
        "authorized": True,
        "requested": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "purpose": "collect_version_bound_authoritative_cost_and_return_samples",
        "selected_side": selected["side"],
        "selected_observation": selected,
        "sampling_stratum": selected["sampling_stratum"],
        "side_observations": observations,
        "direction_score_gap": score_gap,
        "confidence": max(0.0, min(confidence, 1.0)),
        "artifact_version": signal.get("model_version"),
        "artifact_lifecycle": signal.get("artifact_lifecycle"),
        "source_model": "local_ml_profit_quality",
        "source_sample_count": max(
            int(signal.get("trained_sample_count") or 0),
            int(selected.get("distribution_member_count") or 0),
        ),
        "generated_at": now.isoformat(),
        "policy_provenance": {
            "source": "governed_shadow_market_distribution_for_paper_bootstrap",
            "observation_window": "current_model_artifact_and_pre_order_market_snapshot",
            "sample_count": max(
                int(signal.get("trained_sample_count") or 0),
                int(selected.get("distribution_member_count") or 0),
            ),
            "generated_at": now.isoformat(),
            "strategy_version": PAPER_BOOTSTRAP_CANARY_VERSION,
            "fallback_reason": "",
        },
    }


def annotate_paper_bootstrap_opportunity(decision: DecisionOutput) -> float | None:
    """Persist the canary observation score without granting production eligibility."""

    raw = _safe_dict(decision.raw_response)
    contract = _safe_dict(raw.get("paper_bootstrap_canary"))
    if (
        contract.get("version") != PAPER_BOOTSTRAP_CANARY_VERSION
        or contract.get("requested") is not True
    ):
        return None

    selected = _safe_dict(contract.get("selected_observation"))
    score = _finite(selected.get("objective_expected_return_pct"))
    opportunity = dict(_safe_dict(raw.get("opportunity_score")))
    production_score = _finite(opportunity.get("score"))
    opportunity.update(
        {
            "contract_lifecycle": "paper_bootstrap_canary",
            "score_kind": "paper_canary_objective_expected_return",
            "score": round(score, 8) if score is not None else None,
            "production_score": production_score,
            "production_score_policy": opportunity.get("score_policy"),
            "production_eligible": False,
            "production_permission": False,
            "observation_only": True,
            "execution_scope": "paper_only",
            "canary_objective_expected_return_pct": score,
            "canary_observed_net_return_pct": _finite(selected.get("observed_net_return_pct")),
            "canary_lower_quantile_return_pct": _finite(selected.get("lower_quantile_return_pct")),
            "canary_artifact_version": contract.get("artifact_version"),
            "canary_policy_provenance": _safe_dict(contract.get("policy_provenance")),
        }
    )
    raw["opportunity_score"] = opportunity
    decision.raw_response = raw
    return score


@dataclass(frozen=True, slots=True)
class PaperBootstrapAssessment:
    eligible: bool
    reason: str
    details: dict[str, Any]


class PaperBootstrapCanaryPolicy:
    """Build and validate an independently bounded OKX-demo risk contract."""

    def __init__(
        self,
        *,
        allocated_order_balance: BalanceProvider,
        exchange_risk_facts: ExchangeFactsProvider,
        history_provider: HistoryProvider | None = None,
        preflight_timeout_seconds: float = PAPER_BOOTSTRAP_PREFLIGHT_TIMEOUT_SECONDS,
        history_timeout_seconds: float = PAPER_BOOTSTRAP_HISTORY_TIMEOUT_SECONDS,
        prepare_timeout_seconds: float = PAPER_BOOTSTRAP_PREPARE_TIMEOUT_SECONDS,
        exchange_facts_timeout_seconds: float = PAPER_BOOTSTRAP_EXCHANGE_FACTS_TIMEOUT_SECONDS,
        balance_timeout_seconds: float = PAPER_BOOTSTRAP_BALANCE_TIMEOUT_SECONDS,
    ) -> None:
        self.allocated_order_balance = allocated_order_balance
        self.exchange_risk_facts = exchange_risk_facts
        self.history_provider = history_provider or self._load_history
        self.preflight_timeout_seconds = max(float(preflight_timeout_seconds or 0.0), 0.05)
        self.history_timeout_seconds = max(float(history_timeout_seconds or 0.0), 0.05)
        self.prepare_timeout_seconds = max(float(prepare_timeout_seconds or 0.0), 0.05)
        self.exchange_facts_timeout_seconds = max(
            float(exchange_facts_timeout_seconds or 0.0),
            0.05,
        )
        self.balance_timeout_seconds = max(float(balance_timeout_seconds or 0.0), 0.05)

    @staticmethod
    def is_claimed(decision: DecisionOutput) -> bool:
        contract = _safe_dict(_safe_dict(decision.raw_response).get("paper_bootstrap_canary"))
        return bool(
            decision.is_entry
            and contract.get("version") == PAPER_BOOTSTRAP_CANARY_VERSION
            and contract.get("requested") is True
        )

    async def preflight(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]],
        *,
        deadline_monotonic: float | None = None,
        timing_scope: str = "runtime_preflight",
    ) -> PaperBootstrapAssessment:
        """Evaluate lifecycle guards before an entry action is persisted."""

        loop = asyncio.get_running_loop()
        started = loop.time()
        policy_deadline = started + self.preflight_timeout_seconds
        effective_deadline = (
            min(policy_deadline, float(deadline_monotonic))
            if deadline_monotonic is not None
            else policy_deadline
        )
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        reasons = PaperBootstrapCanaryPolicy._contract_reasons(
            decision,
            model_mode,
            contract,
        )
        runtime_guard = await self._runtime_guard(
            contract,
            open_positions,
            deadline_monotonic=effective_deadline,
        )
        reasons.extend(runtime_guard["blocking_reasons"])
        reasons = list(dict.fromkeys(reasons))
        eligible = not reasons
        contract = dict(contract)
        history_status = str(_safe_dict(runtime_guard.get("history_query")).get("status") or "")
        preflight_timing = {
            "stage": "paper_bootstrap_canary_preflight",
            "scope": timing_scope,
            "status": "completed" if history_status == "ok" else "failed_closed",
            "duration_seconds": round(max(loop.time() - started, 0.0), 6),
            "policy_timeout_seconds": round(self.preflight_timeout_seconds, 6),
            "deadline_remaining_seconds": round(
                max(effective_deadline - loop.time(), 0.0),
                6,
            ),
            "history_query_status": history_status,
            "authorized": eligible,
            "blocking_reasons": reasons,
            "message_zh": (
                "paper canary 运行时风控证据已在时限内完成。"
                if history_status == "ok"
                else "paper canary 运行时风控证据未在时限内完整取得，已按失败关闭处理。"
            ),
        }
        previous_timings = [
            item
            for item in _safe_list(contract.get("runtime_preflight_timings"))
            if isinstance(item, dict)
        ]
        contract["runtime_preflight_timings"] = [*previous_timings[-4:], preflight_timing]
        contract["runtime_preflight_timing"] = preflight_timing
        contract["runtime_guard"] = runtime_guard
        contract["runtime_preflight_authorized"] = eligible
        contract["runtime_preflight_blocking_reasons"] = reasons
        if not eligible:
            contract["runtime_authorized"] = False
            contract["runtime_blocking_reasons"] = reasons
            raw["profit_risk_sizing"] = {
                "production_eligible": False,
                "contract_lifecycle": "paper_bootstrap_canary",
                "reasons": reasons,
                "policy_provenance": {
                    **_safe_dict(contract.get("policy_provenance")),
                    "fallback_reason": ",".join(reasons),
                },
            }
            decision.position_size_pct = 0.0
            decision.suggested_leverage = 1.0
        if timing_scope == "market_decision_persistence":
            market_timings = [
                item
                for item in _safe_list(raw.get("market_context_timings"))
                if isinstance(item, dict)
            ]
            raw["market_context_timings"] = [*market_timings, preflight_timing]
        raw["paper_bootstrap_canary"] = contract
        decision.raw_response = raw
        return PaperBootstrapAssessment(
            eligible=eligible,
            reason=(
                "paper_bootstrap_canary_runtime_preflight_ready" if eligible else ",".join(reasons)
            ),
            details={
                "contract": contract,
                "runtime_guard": runtime_guard,
                "sizing": _safe_dict(raw.get("profit_risk_sizing")),
            },
        )

    @staticmethod
    def demote_blocked_candidate_to_hold(
        decision: DecisionOutput,
        assessment: PaperBootstrapAssessment,
    ) -> bool:
        """Persist a blocked canary as hold while retaining its shadow direction."""

        if assessment.eligible or not PaperBootstrapCanaryPolicy.is_claimed(decision):
            return False
        candidate_action = decision.action.value
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        contract["candidate_action"] = candidate_action
        contract["persisted_action"] = Action.HOLD.value
        contract["execution_intent"] = "observation_only_hold"
        contract["candidate_blocking_reason"] = assessment.reason
        raw["paper_bootstrap_canary"] = contract
        raw["paper_bootstrap_canary_observation"] = {
            "candidate_action": candidate_action,
            "persisted_action": Action.HOLD.value,
            "selected_side": contract.get("selected_side"),
            "reason": assessment.reason,
            "shadow_direction_preserved": True,
            "exchange_submission_allowed": False,
        }
        decision.action = Action.HOLD
        decision.position_size_pct = 0.0
        decision.suggested_leverage = 1.0
        decision.stop_loss_pct = 0.0
        decision.take_profit_pct = 0.0
        decision.reasoning = (
            "paper canary \u8fd0\u884c\u65f6\u98ce\u63a7\u672a\u6388\u6743\u672c\u8f6e\u5f00\u4ed3\uff0c"
            "\u5019\u9009\u65b9\u5411\u4ec5\u4fdd\u7559\u4e3a\u89c2\u5bdf\u8bc1\u636e\uff0c\u672c\u8f6e\u88c1\u51b3\u4e3a\u89c2\u671b\u3002"
        )
        decision.raw_response = raw
        return True

    async def prepare(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]],
    ) -> PaperBootstrapAssessment:
        loop = asyncio.get_running_loop()
        prepare_started = loop.time()
        prepare_deadline = prepare_started + self.prepare_timeout_seconds
        preflight = await self.preflight(
            decision,
            model_mode,
            open_positions,
            deadline_monotonic=prepare_deadline,
            timing_scope="risk_contract_prepare",
        )
        if not preflight.eligible:
            return preflight
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        runtime_guard = _safe_dict(preflight.details.get("runtime_guard"))
        reasons: list[str] = []
        facts: dict[str, Any] = {}
        allocated_margin = 0.0
        prepare_timings: list[dict[str, Any]] = []
        facts_value, facts_timing, facts_reason = await _bounded_policy_call(
            "exchange_risk_facts",
            lambda: self.exchange_risk_facts(model_mode, decision, open_positions),
            fallback={},
            timeout_seconds=self.exchange_facts_timeout_seconds,
            deadline_monotonic=prepare_deadline,
            timeout_reason="paper_canary_exchange_facts_timeout",
            budget_reason="paper_canary_prepare_budget_exhausted",
            unavailable_reason_prefix="paper_canary_exchange_facts_unavailable",
            message_zh="读取 OKX 账户、合约规格、杠杆档位和交易资格证据。",
        )
        prepare_timings.append(facts_timing)
        facts = _safe_dict(facts_value)
        if facts_reason:
            reasons.append(facts_reason)

        if not reasons:
            balance_value, balance_timing, balance_reason = await _bounded_policy_call(
                "allocated_order_balance",
                lambda: self.allocated_order_balance(model_mode, decision),
                fallback=None,
                timeout_seconds=self.balance_timeout_seconds,
                deadline_monotonic=prepare_deadline,
                timeout_reason="paper_canary_allocated_balance_timeout",
                budget_reason="paper_canary_prepare_budget_exhausted",
                unavailable_reason_prefix="paper_canary_allocated_balance_unavailable",
                message_zh="读取当前策略可分配的账户保证金证据。",
            )
            prepare_timings.append(balance_timing)
            allocated_margin = _positive(balance_value)
            if balance_reason:
                reasons.append(balance_reason)

        if not reasons:
            self._attach_risk_contract(
                decision,
                contract=contract,
                facts=facts,
                allocated_margin=allocated_margin,
                runtime_guard=runtime_guard,
                open_positions=open_positions,
            )
            sizing = _safe_dict(_safe_dict(decision.raw_response).get("profit_risk_sizing"))
            reasons.extend(str(item) for item in _safe_list(sizing.get("reasons")) if item)

        eligible = not reasons
        contract = dict(contract)
        prepare_timing = {
            "stage": "paper_bootstrap_canary_prepare",
            "status": "completed" if eligible else "failed_closed",
            "duration_seconds": round(max(loop.time() - prepare_started, 0.0), 6),
            "policy_timeout_seconds": round(self.prepare_timeout_seconds, 6),
            "deadline_remaining_seconds": round(
                max(prepare_deadline - loop.time(), 0.0),
                6,
            ),
            "authorized": eligible,
            "blocking_reasons": list(dict.fromkeys(reasons)),
            "stages": prepare_timings,
            "message_zh": (
                "paper canary 风险合同已在时限内完成。"
                if eligible
                else "paper canary 风险合同证据不完整，已按失败关闭处理。"
            ),
        }
        contract["runtime_prepare_timing"] = prepare_timing
        contract["runtime_guard"] = runtime_guard
        contract["runtime_authorized"] = eligible
        contract["runtime_blocking_reasons"] = list(dict.fromkeys(reasons))
        raw = _safe_dict(decision.raw_response)
        raw["paper_bootstrap_canary"] = contract
        if not eligible:
            raw["profit_risk_sizing"] = {
                "production_eligible": False,
                "contract_lifecycle": "paper_bootstrap_canary",
                "reasons": list(dict.fromkeys(reasons)),
                "policy_provenance": {
                    **_safe_dict(contract.get("policy_provenance")),
                    "fallback_reason": ",".join(dict.fromkeys(reasons)),
                },
            }
            decision.position_size_pct = 0.0
            decision.suggested_leverage = 1.0
        decision.raw_response = raw
        return PaperBootstrapAssessment(
            eligible=eligible,
            reason=(
                "paper_bootstrap_canary_contract_ready"
                if eligible
                else ",".join(dict.fromkeys(reasons))
            ),
            details={
                "contract": contract,
                "runtime_guard": runtime_guard,
                "sizing": _safe_dict(raw.get("profit_risk_sizing")),
            },
        )

    @staticmethod
    def assess(decision: DecisionOutput, model_mode: str) -> PaperBootstrapAssessment:
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        sizing = _safe_dict(raw.get("profit_risk_sizing"))
        reasons = PaperBootstrapCanaryPolicy._contract_reasons(
            decision,
            model_mode,
            contract,
        )
        if contract.get("runtime_authorized") is not True:
            reasons.append("paper_canary_runtime_guard_not_authorized")
        if sizing.get("contract_lifecycle") != "paper_bootstrap_canary":
            reasons.append("paper_canary_sizing_lifecycle_mismatch")
        if sizing.get("production_eligible") is not True:
            reasons.append("paper_canary_risk_contract_ineligible")
        risk_budget = _positive(sizing.get("risk_budget_usdt"))
        planned_loss = _positive(sizing.get("planned_stressed_loss_usdt"))
        stress = _positive(sizing.get("stressed_loss_fraction"))
        final_notional = _positive(sizing.get("final_notional_usdt"))
        target_notional = _positive(sizing.get("target_notional_usdt"))
        final_margin = _positive(sizing.get("final_margin_usdt"))
        position_size = _positive(sizing.get("position_size_pct"))
        fingerprint = str(
            _safe_dict(sizing.get("policy_provenance")).get("contract_fingerprint") or ""
        )
        if risk_budget <= 0 or planned_loss <= 0 or planned_loss > risk_budget + 1e-8:
            reasons.append("paper_canary_risk_budget_invalid")
        if stress <= 0 or not isclose(
            planned_loss,
            final_notional * stress,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            reasons.append("paper_canary_stressed_loss_algebra_mismatch")
        if final_notional <= 0 or final_notional > target_notional + 1e-8:
            reasons.append("paper_canary_notional_invalid")
        if final_margin <= 0 or position_size <= 0 or not fingerprint:
            reasons.append("paper_canary_sizing_identity_incomplete")
        if not isclose(position_size, _positive(decision.position_size_pct), abs_tol=1e-8):
            reasons.append("paper_canary_decision_size_mismatch")
        reasons = list(dict.fromkeys(reasons))
        return PaperBootstrapAssessment(
            eligible=not reasons,
            reason="paper_bootstrap_canary_contract_ready" if not reasons else ",".join(reasons),
            details={"contract": contract, "sizing": sizing, "blocking_reasons": reasons},
        )

    @staticmethod
    def _contract_reasons(
        decision: DecisionOutput,
        model_mode: str,
        contract: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        side = "long" if decision.action == Action.LONG else "short"
        if model_mode != "paper":
            reasons.append("paper_canary_live_execution_forbidden")
        if contract.get("authorized") is not True:
            reasons.append("paper_canary_not_authorized")
        if contract.get("execution_scope") != "paper_only":
            reasons.append("paper_canary_scope_invalid")
        if contract.get("production_permission") is not False:
            reasons.append("paper_canary_production_permission_invalid")
        if contract.get("artifact_lifecycle") != "canary":
            reasons.append("paper_canary_artifact_lifecycle_invalid")
        if contract.get("selected_side") != side:
            reasons.append("paper_canary_selected_side_mismatch")
        stratum = _safe_dict(contract.get("sampling_stratum"))
        if (
            stratum.get("side") != side
            or not stratum.get("symbol")
            or stratum.get("volatility_bucket") not in {"low", "medium", "high"}
            or not stratum.get("market_regime")
            or not stratum.get("key")
        ):
            reasons.append("paper_canary_sampling_stratum_incomplete")
        provenance = _safe_dict(contract.get("policy_provenance"))
        if (
            not provenance.get("source")
            or not provenance.get("observation_window")
            or int(_positive(provenance.get("sample_count"))) <= 0
            or not provenance.get("generated_at")
            or provenance.get("strategy_version") != PAPER_BOOTSTRAP_CANARY_VERSION
            or provenance.get("fallback_reason")
        ):
            reasons.append("paper_canary_provenance_incomplete")
        return reasons

    async def _runtime_guard(
        self,
        contract: dict[str, Any],
        open_positions: list[dict[str, Any]],
        *,
        deadline_monotonic: float,
    ) -> dict[str, Any]:
        account_open_positions = [
            item
            for item in open_positions
            if item.get("is_open", True) is not False and _positive(item.get("quantity", 1.0)) > 0
        ]
        reasons: list[str] = []
        history_value, history_timing, history_reason = await _bounded_policy_call(
            "canary_history_query",
            self.history_provider,
            fallback=[],
            timeout_seconds=self.history_timeout_seconds,
            deadline_monotonic=deadline_monotonic,
            timeout_reason="paper_canary_history_timeout",
            budget_reason="paper_canary_preflight_budget_exhausted",
            unavailable_reason_prefix="paper_canary_history_unavailable",
            message_zh="读取已执行 paper canary 的仓位、日频、连亏和冷却风控证据。",
        )
        history = list(history_value or [])
        if history_reason:
            reasons.append(history_reason)

        canary_rows = [
            row
            for row in history
            if (
                _row_value(row, "paper_canary_authorized") is True
                or _row_value(row, "paper_canary_authorized") == 1
                or _safe_dict(_row_raw(row).get("paper_bootstrap_canary")).get("authorized") is True
            )
        ]
        open_canary_rows = [
            row for row in canary_rows if not str(_row_value(row, "outcome") or "").strip()
        ]
        if len(open_canary_rows) >= PAPER_BOOTSTRAP_MAX_OPEN_POSITIONS:
            reasons.append("paper_canary_open_position_limit_reached")
        now = datetime.now(UTC)
        today_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        history_times = [
            timestamp
            for row in canary_rows
            if (
                timestamp := _as_utc(
                    _row_value(row, "executed_at") or _row_value(row, "created_at")
                )
            )
            is not None
        ]
        daily_rows = [
            row
            for row in canary_rows
            if (
                _as_utc(_row_value(row, "executed_at") or _row_value(row, "created_at"))
                or datetime.min.replace(tzinfo=UTC)
            )
            >= today_start
        ]
        daily_entries = sum(
            1 for row in daily_rows
        )
        completed = [row for row in canary_rows if _row_value(row, "outcome")]
        completed_sample_count = len(completed)
        remaining_sample_count = max(
            PAPER_BOOTSTRAP_TARGET_AUTHORITATIVE_SAMPLES - completed_sample_count,
            0,
        )
        collection_started_at = min(history_times) if history_times else now
        collection_deadline = collection_started_at + timedelta(
            days=PAPER_BOOTSTRAP_COLLECTION_HORIZON_DAYS
        )
        remaining_days = max(
            ceil(max((collection_deadline - now).total_seconds(), 0.0) / 86400.0),
            1,
        )
        required_daily_entries = (
            ceil(remaining_sample_count / remaining_days)
            if remaining_sample_count > 0
            else 0
        )
        adaptive_daily_limit = min(
            PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES,
            max(
                PAPER_BOOTSTRAP_MIN_DAILY_ENTRIES,
                required_daily_entries,
            ),
        )
        remaining_capacity = PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES * remaining_days
        sampling_plan_reachable = remaining_sample_count <= remaining_capacity
        if daily_entries >= adaptive_daily_limit:
            reasons.append("paper_canary_daily_entry_budget_exhausted")

        current_stratum = _safe_dict(contract.get("sampling_stratum"))
        current_stratum_key = str(current_stratum.get("key") or "")
        dimension_values = {
            "side": ("long", "short"),
            "volatility_bucket": ("low", "medium", "high"),
            "market_regime": ("ranging", "trending", "volatile"),
        }
        stratum_counts: dict[str, dict[str, int]] = {
            dimension: {value: 0 for value in values}
            for dimension, values in dimension_values.items()
        }
        symbol_counts: dict[str, int] = {}
        for row in daily_rows:
            row_side = str(_row_value(row, "action") or "").lower()
            row_symbol = str(_row_value(row, "symbol") or "")
            row_volatility = str(_row_value(row, "canary_volatility_bucket") or "")
            row_regime = str(_row_value(row, "canary_market_regime") or "")
            if row_side in stratum_counts["side"]:
                stratum_counts["side"][row_side] += 1
            if row_volatility in stratum_counts["volatility_bucket"]:
                stratum_counts["volatility_bucket"][row_volatility] += 1
            if row_regime in stratum_counts["market_regime"]:
                stratum_counts["market_regime"][row_regime] += 1
            if row_symbol:
                symbol_counts[row_symbol] = symbol_counts.get(row_symbol, 0) + 1
        current_symbol = str(current_stratum.get("symbol") or "")
        if current_symbol:
            symbol_counts.setdefault(current_symbol, 0)
        overrepresented_dimensions: list[str] = []
        if daily_entries >= PAPER_BOOTSTRAP_MIN_DAILY_ENTRIES:
            for dimension, counts in stratum_counts.items():
                current_value = str(current_stratum.get(dimension) or "")
                if current_value in counts and (
                    counts[current_value] - min(counts.values())
                    >= PAPER_BOOTSTRAP_STRATUM_IMBALANCE_TOLERANCE
                ):
                    overrepresented_dimensions.append(dimension)
            if len(symbol_counts) > 1 and current_symbol in symbol_counts and (
                symbol_counts[current_symbol] - min(symbol_counts.values())
                >= PAPER_BOOTSTRAP_STRATUM_IMBALANCE_TOLERANCE
            ):
                overrepresented_dimensions.append("symbol")
        if overrepresented_dimensions:
            reasons.append("paper_canary_stratum_quota_exhausted")

        daily_loss_fraction = 0.0
        for row in daily_rows:
            if str(_row_value(row, "outcome") or "").lower() != "loss":
                continue
            pnl_pct = _finite(_row_value(row, "outcome_pnl_pct"))
            notional = _positive(_row_value(row, "canary_final_notional_usdt"))
            account_equity = _positive(_row_value(row, "canary_account_equity_usdt"))
            if pnl_pct is not None and pnl_pct < 0 and notional > 0 and account_equity > 0:
                daily_loss_fraction += abs(pnl_pct) / 100.0 * notional / account_equity
            else:
                daily_loss_fraction += PAPER_BOOTSTRAP_SINGLE_TRADE_EQUITY_RISK
        daily_loss_budget_exhausted = (
            daily_loss_fraction >= PAPER_BOOTSTRAP_DAILY_LOSS_EQUITY_RISK
        )
        if daily_loss_budget_exhausted:
            reasons.append("paper_canary_daily_loss_budget_exhausted")

        last_executed = _as_utc(
            next(
                (
                    _row_value(row, "executed_at") or _row_value(row, "created_at")
                    for row in canary_rows
                    if str(_row_value(row, "canary_sampling_stratum_key") or "")
                    == current_stratum_key
                    if _row_value(row, "executed_at") or _row_value(row, "created_at")
                ),
                None,
            )
        )
        horizon_minutes = int(
            _positive(_safe_dict(contract.get("selected_observation")).get("horizon_minutes")) or 10
        )
        cooldown_seconds = max(
            horizon_minutes * 60,
            max(int(settings.decision_interval_seconds or 60), 1) * 5,
        )
        if isinstance(last_executed, datetime):
            if now - last_executed < timedelta(seconds=cooldown_seconds):
                reasons.append("paper_canary_cooldown_active")
        return {
            "blocking_reasons": list(dict.fromkeys(reasons)),
            "status": "available" if history_reason is None else "failed_closed",
            "history_query": history_timing,
            "open_position_count": len(open_canary_rows),
            "max_open_positions": PAPER_BOOTSTRAP_MAX_OPEN_POSITIONS,
            "account_open_position_count": len(account_open_positions),
            "open_position_source": "executed_canary_decisions_without_outcome",
            "daily_entry_count": daily_entries,
            "max_daily_entries": adaptive_daily_limit,
            "absolute_max_daily_entries": PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES,
            "min_daily_entries": PAPER_BOOTSTRAP_MIN_DAILY_ENTRIES,
            "completed_authoritative_sample_count": completed_sample_count,
            "target_authoritative_sample_count": PAPER_BOOTSTRAP_TARGET_AUTHORITATIVE_SAMPLES,
            "remaining_authoritative_sample_count": remaining_sample_count,
            "collection_started_at": collection_started_at.isoformat(),
            "collection_deadline": collection_deadline.isoformat(),
            "remaining_collection_days": remaining_days,
            "required_daily_entries": required_daily_entries,
            "remaining_collection_capacity": remaining_capacity,
            "sampling_plan_reachable": sampling_plan_reachable,
            "sampling_plan_alert_active": not sampling_plan_reachable,
            "daily_entry_limit_source": (
                "remaining_sample_deficit_over_remaining_collection_days"
            ),
            "sampling_stratum": current_stratum,
            "sampling_stratum_counts": stratum_counts,
            "sampling_symbol_counts": symbol_counts,
            "overrepresented_sampling_dimensions": overrepresented_dimensions,
            "daily_loss_fraction": daily_loss_fraction,
            "daily_loss_budget_fraction": PAPER_BOOTSTRAP_DAILY_LOSS_EQUITY_RISK,
            "daily_loss_budget_exhausted": daily_loss_budget_exhausted,
            "cooldown_seconds": cooldown_seconds,
            "last_executed_at": (
                last_executed.isoformat() if isinstance(last_executed, datetime) else None
            ),
        }

    def _attach_risk_contract(
        self,
        decision: DecisionOutput,
        *,
        contract: dict[str, Any],
        facts: dict[str, Any],
        allocated_margin: float,
        runtime_guard: dict[str, Any],
        open_positions: list[dict[str, Any]],
    ) -> None:
        raw = _safe_dict(decision.raw_response)
        snapshot = _safe_dict(decision.feature_snapshot)
        pre_order = _safe_dict(raw.get("pre_order_execution_facts"))
        available_from_facts = _positive(facts.get("available_margin_usdt"))
        available_margin = (
            min(allocated_margin, available_from_facts)
            if allocated_margin > 0 and available_from_facts > 0
            else available_from_facts
        )
        equity = _positive(facts.get("account_equity_usdt"))
        price = max(
            _positive(snapshot.get("current_price")),
            _positive(snapshot.get("close")),
        )
        atr_fraction = _positive(snapshot.get("atr_14")) / price if price > 0 else 0.0
        volatility_fraction = _normalized_ratio(snapshot.get("volatility_20"))
        wick_fraction = _positive(snapshot.get("abnormal_wick_max_pct")) / 100.0
        opportunity = _safe_dict(raw.get("opportunity_score"))
        execution_cost = _safe_dict(opportunity.get("execution_cost"))
        cost_pct = max(
            _positive(execution_cost.get("total_pct")),
            _positive(pre_order.get("total_cost_pct")),
        )
        stress = max(
            _normalized_ratio(decision.stop_loss_pct),
            atr_fraction,
            volatility_fraction,
            wick_fraction,
            cost_pct * 3.0 / 100.0,
        )
        reasons: list[str] = []
        if facts.get("production_eligible") is not True:
            reasons.append("paper_canary_exchange_risk_facts_ineligible")
        if pre_order and pre_order.get("production_eligible") is not True:
            reasons.append("paper_canary_pre_order_facts_ineligible")
        if equity <= 0 or available_margin <= 0:
            reasons.append("paper_canary_account_capacity_unavailable")
        if stress <= 0:
            reasons.append("paper_canary_stressed_loss_fraction_missing")

        single_trade_budget = equity * PAPER_BOOTSTRAP_SINGLE_TRADE_EQUITY_RISK
        portfolio_budget = equity * PAPER_BOOTSTRAP_PORTFOLIO_EQUITY_RISK
        contract_specs = _safe_dict(facts.get("contract_specs"))
        account_portfolio, account_portfolio_blockers = build_portfolio_risk_snapshot(
            open_positions,
            candidate_side=str(contract.get("selected_side") or ""),
            contract_specs=contract_specs,
        )
        account_portfolio["valuation_blockers"] = account_portfolio_blockers
        canary_positions = [
            position for position in open_positions if _is_open_paper_canary_position(position)
        ]
        canary_portfolio, canary_portfolio_blockers = build_portfolio_risk_snapshot(
            canary_positions,
            candidate_side=str(contract.get("selected_side") or ""),
            contract_specs=contract_specs,
        )
        canary_portfolio["scope"] = "paper_bootstrap_canary_positions_only"
        canary_portfolio["valuation_blockers"] = canary_portfolio_blockers
        runtime_open_count = int(_positive(runtime_guard.get("open_position_count")))
        conservative_open_count = max(runtime_open_count, len(canary_positions))
        current_canary_stressed_loss = max(
            _positive(canary_portfolio.get("current_stressed_loss_usdt")),
            conservative_open_count * single_trade_budget,
        )
        canary_portfolio["current_stressed_loss_usdt"] = current_canary_stressed_loss
        canary_portfolio["conservative_open_position_count"] = conservative_open_count
        remaining_portfolio_budget = max(
            portfolio_budget - current_canary_stressed_loss,
            0.0,
        )
        risk_budget = min(single_trade_budget, remaining_portfolio_budget)
        if risk_budget <= 0:
            reasons.append("paper_canary_portfolio_risk_budget_exhausted")
        fill_drift_reserve_fraction = max(
            cost_pct / 100.0,
            PAPER_BOOTSTRAP_MIN_FILL_DRIFT_RESERVE_FRACTION,
        )
        fill_notional_ceiling = risk_budget / stress if stress > 0 else 0.0
        target_notional = (
            fill_notional_ceiling / (1.0 + fill_drift_reserve_fraction)
            if fill_notional_ceiling > 0
            else 0.0
        )
        final_notional = min(target_notional, available_margin)
        final_margin = final_notional
        position_size = final_margin / available_margin if available_margin > 0 else 0.0
        planned_loss = final_notional * stress
        if final_notional <= 0 or planned_loss <= 0 or planned_loss > risk_budget + 1e-8:
            reasons.append("paper_canary_independent_risk_budget_zero")
        if current_canary_stressed_loss + planned_loss > portfolio_budget + 1e-8:
            reasons.append("paper_canary_portfolio_stressed_loss_exceeded")
        target_inst_id = str(facts.get("target_inst_id") or "").strip()
        target_contract_spec = _safe_dict(contract_specs.get(target_inst_id))
        leverage_tier_selection = select_okx_leverage_tier(
            facts.get("leverage_tiers"),
            target_notional_usdt=final_notional,
            mark_price=price,
            contract_spec=target_contract_spec,
        )
        if leverage_tier_selection.get("production_eligible") is not True:
            reasons.append(
                "paper_canary_leverage_tier_ineligible:"
                + str(leverage_tier_selection.get("reason") or "unknown")
            )

        generated_at = datetime.now(UTC).isoformat()
        selected_observation = _safe_dict(contract.get("selected_observation"))
        expected_net_return = _finite(selected_observation.get("observed_net_return_pct"))
        fingerprint_payload = {
            "artifact_version": contract.get("artifact_version"),
            "symbol": decision.symbol,
            "side": contract.get("selected_side"),
            "equity": round(equity, 8),
            "available_margin": round(available_margin, 8),
            "stress": round(stress, 8),
            "risk_budget": round(risk_budget, 8),
            "final_notional": round(final_notional, 8),
            "pre_order_fingerprint": pre_order.get("input_fingerprint"),
        }
        provenance = {
            "source": "paper_bootstrap_canary_independent_risk_budget",
            "observation_window": "current_okx_demo_account_market_and_canary_runtime_guard",
            "sample_count": max(int(contract.get("source_sample_count") or 0), 1),
            "generated_at": generated_at,
            "strategy_version": PAPER_BOOTSTRAP_SIZING_VERSION,
            "fallback_reason": ",".join(reasons),
            "contract_fingerprint": _canonical_sha256(fingerprint_payload),
        }
        sizing = {
            "contract_version": PAPER_BOOTSTRAP_SIZING_VERSION,
            "production_eligible": not reasons,
            "reason": (
                "paper_bootstrap_canary_independent_risk_budget_ready"
                if not reasons
                else ",".join(dict.fromkeys(reasons))
            ),
            "contract_lifecycle": "paper_bootstrap_canary",
            "execution_scope": "paper_only",
            "production_permission": False,
            "account_equity_usdt": round(equity, 8),
            "available_margin_usdt": round(available_margin, 8),
            "risk_budget_usdt": risk_budget,
            "single_trade_risk_budget_usdt": single_trade_budget,
            "portfolio_risk_budget_usdt": portfolio_budget,
            "remaining_portfolio_risk_budget_usdt": remaining_portfolio_budget,
            "current_portfolio_stressed_loss_usdt": current_canary_stressed_loss,
            "planned_stressed_loss_usdt": planned_loss,
            "stressed_loss_fraction": stress,
            "target_notional_usdt": target_notional,
            "fill_notional_ceiling_usdt": fill_notional_ceiling,
            "estimated_fill_drift_reserve_fraction": fill_drift_reserve_fraction,
            "final_notional_usdt": final_notional,
            "final_margin_usdt": final_margin,
            "position_size_pct": position_size,
            "expected_net_return_pct": expected_net_return,
            "expected_profit_usdt": (
                final_notional * expected_net_return / 100.0
                if expected_net_return is not None
                else None
            ),
            "portfolio_risk_snapshot": canary_portfolio,
            "account_portfolio_risk_snapshot": account_portfolio,
            "exchange_contract_specs": contract_specs,
            "exchange_risk_facts_provenance": facts.get("policy_provenance"),
            "entry_instrument_availability": facts.get("entry_instrument_availability"),
            "leverage_tier_selection": leverage_tier_selection,
            "runtime_guard": runtime_guard,
            "reasons": reasons,
            "units": {
                "money": "USDT",
                "returns": "percentage_points",
                "fractions": "decimal_ratio",
                "position_size_pct": "available_margin_fraction",
                "notional": "USDT",
            },
            "policy_provenance": provenance,
        }
        raw["profit_risk_sizing"] = sizing
        raw["execution_cost_sizing_pass"] = {
            "order_size_complete": bool(not reasons and final_notional > 0),
            "impact_basis_notional_usdt": final_notional,
            "final_notional_usdt": final_notional,
            "contract_lifecycle": "paper_bootstrap_canary",
        }
        decision.raw_response = raw
        decision.position_size_pct = position_size if not reasons else 0.0
        decision.suggested_leverage = 1.0
        if not reasons:
            decision.stop_loss_pct = stress
            decision.take_profit_pct = max(stress * 1.5, cost_pct * 2.0 / 100.0)

    @staticmethod
    def _history_statement() -> Any:
        canary = AIDecision.raw_llm_response["paper_bootstrap_canary"]
        sizing = AIDecision.raw_llm_response["profit_risk_sizing"]
        authorized = canary["authorized"].as_boolean()
        return (
            select(
                authorized.label("paper_canary_authorized"),
                AIDecision.symbol,
                AIDecision.action,
                AIDecision.outcome,
                AIDecision.outcome_pnl_pct,
                AIDecision.position_size_pct,
                canary["sampling_stratum"]["key"].as_string().label(
                    "canary_sampling_stratum_key"
                ),
                canary["sampling_stratum"]["volatility_bucket"].as_string().label(
                    "canary_volatility_bucket"
                ),
                canary["sampling_stratum"]["market_regime"].as_string().label(
                    "canary_market_regime"
                ),
                sizing["final_notional_usdt"].as_float().label(
                    "canary_final_notional_usdt"
                ),
                sizing["account_equity_usdt"].as_float().label(
                    "canary_account_equity_usdt"
                ),
                AIDecision.executed_at,
                AIDecision.created_at,
            )
            .where(
                AIDecision.is_paper.is_(True),
                AIDecision.action.in_(("long", "short")),
                AIDecision.was_executed.is_(True),
                AIDecision.created_at >= PAPER_BOOTSTRAP_CANARY_HISTORY_START,
                authorized.is_(True),
            )
            .order_by(AIDecision.created_at.desc(), AIDecision.id.desc())
            .limit(200)
        )

    @staticmethod
    async def _load_history() -> list[Any]:
        async with get_read_session_ctx() as session:
            result = await session.execute(PaperBootstrapCanaryPolicy._history_statement())
        return list(result.all())
