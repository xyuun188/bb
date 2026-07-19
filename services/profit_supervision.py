"""Separated supervision contracts for profit-oriented training.

Shadow observations describe market opportunity and counterfactual execution
cost. Only trusted OKX position lifecycles describe realized trade outcomes.
The contracts in this module keep those authorities separate through training,
evaluation, promotion, and production return composition.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from core.symbols import normalize_trading_symbol
from core.training_contracts import AUTHORITATIVE_TRADE_OUTCOME_SOURCES
from services.execution_cost_model import execution_cost_estimate

PROFIT_SUPERVISION_VERSION = "2026-07-14.separated-profit-supervision.v1"
MARKET_OPPORTUNITY_TASK = "market_opportunity_distribution"
COUNTERFACTUAL_EXECUTION_COST_TASK = "execution_cost_and_slippage_distribution"
AUTHORITATIVE_REALIZED_RETURN_TASK = "authoritative_realized_return_distribution"
PRODUCTION_RETURN_COMBINATION_VERSION = (
    "2026-07-15.standardized-return-distribution.v2"
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _weighted_quantile(pairs: list[tuple[float, float]], fraction: float) -> float | None:
    usable = sorted((value, weight) for value, weight in pairs if weight > 0)
    if not usable:
        return None
    total = sum(weight for _value, weight in usable)
    target = total * min(max(fraction, 0.0), 1.0)
    cumulative = 0.0
    for value, weight in usable:
        cumulative += weight
        if cumulative >= target:
            return value
    return usable[-1][0]


def weighted_distribution(
    values: Iterable[tuple[Any, Any]],
) -> dict[str, Any]:
    """Summarize a distribution without fixed market or strategy thresholds."""

    pairs: list[tuple[float, float]] = []
    for raw_value, raw_weight in values:
        value = _safe_float(raw_value)
        weight = _safe_float(raw_weight, 0.0) or 0.0
        if value is None or weight <= 0:
            continue
        pairs.append((value, weight))
    if not pairs:
        return {
            "count": 0,
            "effective_sample_size": 0.0,
            "expected": None,
            "median": None,
            "lower_hinge": None,
            "upper_hinge": None,
            "minimum": None,
            "maximum": None,
        }
    weight_sum = sum(weight for _value, weight in pairs)
    square_sum = sum(weight * weight for _value, weight in pairs)
    expected = sum(value * weight for value, weight in pairs) / weight_sum
    return {
        "count": len(pairs),
        "effective_sample_size": round(
            weight_sum * weight_sum / square_sum if square_sum > 0 else 0.0,
            8,
        ),
        "expected": round(expected, 8),
        "median": _weighted_quantile(pairs, 0.5),
        "lower_hinge": _weighted_quantile(pairs, 0.25),
        "upper_hinge": _weighted_quantile(pairs, 0.75),
        "minimum": min(value for value, _weight in pairs),
        "maximum": max(value for value, _weight in pairs),
    }


def _shadow_cost_labels(sample: dict[str, Any]) -> dict[str, Any]:
    features = _safe_dict(sample.get("features"))
    horizon = _safe_int(sample.get("horizon_minutes") or features.get("horizon_minutes"))
    estimate = execution_cost_estimate(features)
    funding_rate = _safe_float(features.get("funding_rate"))
    funding_interval = _safe_float(features.get("funding_interval_minutes"))
    if funding_interval is None:
        interval_hours = _safe_float(features.get("funding_interval_hours"))
        funding_interval = interval_hours * 60.0 if interval_hours is not None else None
    funding_drag = (
        funding_rate * 100.0 * horizon / funding_interval
        if funding_rate is not None
        and funding_interval is not None
        and funding_interval > 0
        and horizon > 0
        else None
    )
    eligible = bool(
        estimate.production_eligible and funding_drag is not None and horizon > 0
    )
    common_cost = estimate.fee_pct + estimate.slippage_pct
    return {
        "eligible": eligible,
        "source_authority": "shadow_counterfactual_live_microstructure",
        "horizon_minutes": horizon,
        "fee_pct": estimate.fee_pct if estimate.production_eligible else None,
        "slippage_pct": estimate.slippage_pct if estimate.production_eligible else None,
        "funding_drag_pct": funding_drag,
        "long_total_cost_pct": common_cost + funding_drag if eligible else None,
        "short_total_cost_pct": common_cost - funding_drag if eligible else None,
        "cost_model": estimate.to_dict(),
        "reason": "" if eligible else "counterfactual_execution_cost_incomplete",
    }


def _trade_cost_labels(
    labels: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    fee = _safe_float(labels.get("fee_return_pct"))
    slippage = _safe_float(labels.get("slippage_return_pct"))
    slippage_source = "okx_entry_and_exit_fill_slippage"
    if (
        slippage is None
        and sample.get("protection_execution_supervision_ready") is True
        and _safe_text(sample.get("stop_loss_slippage_source"))
        == "okx_configured_stop_trigger_to_fills_vwap"
    ):
        slippage = _safe_float(sample.get("stop_loss_slippage_pct"))
        slippage_source = "okx_stop_trigger_to_fill_slippage"
    funding = _safe_float(labels.get("funding_return_pct"))
    eligible = fee is not None and slippage is not None and funding is not None
    # Positive funding is income and therefore reduces realized execution cost.
    total_cost = fee + slippage - funding if eligible else None
    return {
        "eligible": eligible,
        "source_authority": "okx_fills_fees_funding",
        "fee_pct": fee,
        "slippage_pct": slippage,
        "slippage_source": slippage_source if slippage is not None else "missing",
        "funding_return_pct": funding,
        "total_cost_pct": total_cost,
        "reason": "" if eligible else "authoritative_execution_cost_incomplete",
    }


def build_profit_supervision_contract(
    sample: dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    """Build one immutable three-task supervision contract."""

    quality_eligible = not bool(sample.get("exclude_from_training"))
    weight = _safe_float(sample.get("sample_weight"), 0.0) or 0.0
    symbol = normalize_trading_symbol(
        sample.get("symbol") or _safe_dict(sample.get("features")).get("symbol")
    )
    decision_id = _safe_int(sample.get("decision_id")) or None
    horizon = _safe_int(sample.get("horizon_minutes")) or None
    labels = _safe_dict(sample.get("profit_learning_labels"))

    if kind == "shadow":
        long_return = _safe_float(sample.get("long_return_pct"))
        short_return = _safe_float(sample.get("short_return_pct"))
        market_eligible = bool(
            quality_eligible
            and horizon is not None
            and horizon > 0
            and long_return is not None
            and short_return is not None
        )
        market_task = {
            "eligible": market_eligible,
            "source_authority": "shadow_native_market_path",
            "actual_execution": False,
            "horizon_minutes": horizon,
            "long_gross_market_return_pct": long_return,
            "short_gross_market_return_pct": short_return,
            "reason": "" if market_eligible else "shadow_market_opportunity_incomplete",
        }
        cost_task = _shadow_cost_labels(sample)
        cost_task["eligible"] = bool(quality_eligible and cost_task.get("eligible"))
        realized_task = {
            "eligible": False,
            "source_authority": "none",
            "actual_execution": False,
            "reason": "shadow_cannot_supervise_realized_trade_return",
        }
    elif kind == "trade":
        market_task = {
            "eligible": False,
            "source_authority": "none",
            "actual_execution": True,
            "reason": "closed_trade_is_not_two_sided_market_counterfactual",
        }
        cost_task = _trade_cost_labels(labels, sample)
        cost_task["eligible"] = bool(quality_eligible and cost_task.get("eligible"))
        net_return = _safe_float(labels.get("net_return_after_cost_pct"))
        source = _safe_text(sample.get("source"))
        trusted = bool(
            quality_eligible
            and source in AUTHORITATIVE_TRADE_OUTCOME_SOURCES
            and sample.get("trade_fact_trusted") is True
            and _safe_text(sample.get("lifecycle_key"))
            and net_return is not None
        )
        realized_task = {
            "eligible": trusted,
            "source_authority": source if trusted else "none",
            "actual_execution": True,
            "lifecycle_key": _safe_text(sample.get("lifecycle_key")),
            "side": _safe_text(sample.get("side")).lower(),
            "realized_net_return_pct": net_return,
            "realized_net_pnl_usdt": _safe_float(labels.get("realized_net_pnl_usdt")),
            "gross_market_return_pct": _safe_float(
                labels.get("gross_return_on_notional_pct")
            ),
            "hold_minutes": _safe_float(sample.get("hold_minutes")),
            "stop_loss_slippage_pct": (
                _safe_float(sample.get("stop_loss_slippage_pct"))
                if sample.get("protection_execution_supervision_ready") is True
                and _safe_text(sample.get("stop_loss_slippage_source"))
                == "okx_configured_stop_trigger_to_fills_vwap"
                else None
            ),
            "protection_actual_side": _safe_text(
                sample.get("protection_actual_side")
            ),
            "trigger_to_first_fill_ms": _safe_float(
                sample.get("trigger_to_first_fill_ms")
            ),
            "execution_actual_over_budget_loss_usdt": _safe_float(
                sample.get("execution_actual_over_budget_loss_usdt")
            ),
            "exit_timing_label": _safe_text(labels.get("exit_timing_label")),
            "exit_quality_label": _safe_text(labels.get("payoff_profile_label")),
            "losing_exit_attribution": _safe_text(
                labels.get("losing_exit_attribution")
            ),
            "reason": "" if trusted else "authoritative_trade_outcome_incomplete",
        }
    else:
        market_task = {"eligible": False, "reason": "sample_kind_not_applicable"}
        cost_task = {"eligible": False, "reason": "sample_kind_not_applicable"}
        realized_task = {"eligible": False, "reason": "sample_kind_not_applicable"}

    tasks = {
        MARKET_OPPORTUNITY_TASK: market_task,
        COUNTERFACTUAL_EXECUTION_COST_TASK: cost_task,
        AUTHORITATIVE_REALIZED_RETURN_TASK: realized_task,
    }
    contract = {
        "version": PROFIT_SUPERVISION_VERSION,
        "immutable": True,
        "sample_kind": kind,
        "symbol": symbol,
        "decision_id": decision_id,
        "horizon_minutes": horizon,
        "sample_weight": weight,
        "correlation_weight": _safe_dict(sample.get("correlation_weight")),
        "tasks": tasks,
        "authority_invariant": {
            "shadow_realized_trade_weight": 0.0,
            "trade_market_counterfactual_weight": 0.0,
        },
    }
    contract["contract_fingerprint"] = _fingerprint(contract)
    return contract


def apply_correlation_group_weights(
    samples: list[dict[str, Any]],
    *,
    kind: str,
) -> list[dict[str, Any]]:
    """Keep correlated horizons from multiplying one decision's evidence."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if kind == "shadow":
            decision_id = _safe_int(sample.get("decision_id"))
            sample_id = _safe_int(sample.get("id"))
            key = f"shadow_decision:{decision_id or sample_id}"
        elif kind == "trade":
            lifecycle = _safe_text(sample.get("lifecycle_key"))
            position_id = _safe_int(sample.get("position_id"))
            sample_id = _safe_int(sample.get("id"))
            key = f"okx_lifecycle:{lifecycle or position_id or sample_id}"
        else:
            key = f"{kind}:{_safe_int(sample.get('id')) or id(sample)}"
        groups[key].append(sample)

    for key, group in groups.items():
        trainable = [row for row in group if not bool(row.get("exclude_from_training"))]
        base_weights = [
            max(_safe_float(row.get("sample_weight"), 0.0) or 0.0, 0.0)
            for row in trainable
        ]
        weight_sum = sum(base_weights)
        group_budget = max(base_weights, default=0.0)
        multiplier = group_budget / weight_sum if weight_sum > 0 else 0.0
        for row in group:
            base = max(_safe_float(row.get("sample_weight"), 0.0) or 0.0, 0.0)
            adjusted = 0.0 if row.get("exclude_from_training") else base * multiplier
            row["sample_weight"] = adjusted
            row["correlation_weight"] = {
                "source": "shared_decision_or_authoritative_lifecycle_identity",
                "correlation_group": key,
                "group_sample_count": len(group),
                "group_trainable_count": len(trainable),
                "base_quality_weight": base,
                "group_effective_weight_budget": group_budget,
                "normalization_multiplier": multiplier,
                "effective_sample_weight": adjusted,
                "fixed_sampling_ratio": False,
            }
    return samples


def _task_pairs(
    samples: list[dict[str, Any]],
    task_name: str,
    field: str,
) -> list[tuple[Any, Any]]:
    pairs: list[tuple[Any, Any]] = []
    for sample in samples:
        contract = _safe_dict(sample.get("profit_supervision"))
        task = _safe_dict(_safe_dict(contract.get("tasks")).get(task_name))
        if task.get("eligible") is not True:
            continue
        pairs.append((task.get(field), sample.get("sample_weight")))
    return pairs


def profit_supervision_report(
    shadow_samples: list[dict[str, Any]],
    trade_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report shadow and actual supervision separately."""

    all_samples = [*shadow_samples, *trade_samples]
    task_counts: dict[str, int] = {}
    for task_name in (
        MARKET_OPPORTUNITY_TASK,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        AUTHORITATIVE_REALIZED_RETURN_TASK,
    ):
        task_counts[task_name] = sum(
            1
            for sample in all_samples
            if _safe_dict(
                _safe_dict(sample.get("profit_supervision")).get("tasks")
            ).get(task_name, {}).get("eligible")
            is True
        )
    horizon_counts = Counter(
        _safe_int(sample.get("horizon_minutes"))
        for sample in shadow_samples
        if not sample.get("exclude_from_training")
    )
    decision_groups = {
        _safe_text(
            _safe_dict(sample.get("correlation_weight")).get("correlation_group")
        )
        for sample in all_samples
        if _safe_text(
            _safe_dict(sample.get("correlation_weight")).get("correlation_group")
        )
    }
    shadow_market_count = len(
        _task_pairs(
            shadow_samples,
            MARKET_OPPORTUNITY_TASK,
            "long_gross_market_return_pct",
        )
    )
    shadow_cost_count = len(
        _task_pairs(
            shadow_samples,
            COUNTERFACTUAL_EXECUTION_COST_TASK,
            "long_total_cost_pct",
        )
    )
    actual_cost_count = len(
        _task_pairs(
            trade_samples,
            COUNTERFACTUAL_EXECUTION_COST_TASK,
            "total_cost_pct",
        )
    )
    actual_return_count = len(
        _task_pairs(
            trade_samples,
            AUTHORITATIVE_REALIZED_RETURN_TASK,
            "realized_net_return_pct",
        )
    )
    report = {
        "version": PROFIT_SUPERVISION_VERSION,
        "task_counts": task_counts,
        "shadow_market_sample_count": shadow_market_count,
        "shadow_counterfactual_cost_sample_count": shadow_cost_count,
        "actual_execution_cost_sample_count": actual_cost_count,
        "actual_realized_return_sample_count": actual_return_count,
        "shadow_samples_are_actual_returns": False,
        "market_opportunity": {
            "long_gross_return_pct": weighted_distribution(
                _task_pairs(
                    shadow_samples,
                    MARKET_OPPORTUNITY_TASK,
                    "long_gross_market_return_pct",
                )
            ),
            "short_gross_return_pct": weighted_distribution(
                _task_pairs(
                    shadow_samples,
                    MARKET_OPPORTUNITY_TASK,
                    "short_gross_market_return_pct",
                )
            ),
            "horizon_counts": {
                str(key): value for key, value in sorted(horizon_counts.items()) if key > 0
            },
        },
        "counterfactual_execution_cost": {
            "long_total_cost_pct": weighted_distribution(
                _task_pairs(
                    shadow_samples,
                    COUNTERFACTUAL_EXECUTION_COST_TASK,
                    "long_total_cost_pct",
                )
            ),
            "short_total_cost_pct": weighted_distribution(
                _task_pairs(
                    shadow_samples,
                    COUNTERFACTUAL_EXECUTION_COST_TASK,
                    "short_total_cost_pct",
                )
            ),
        },
        "authoritative_realized_trade": {
            "net_return_after_cost_pct": weighted_distribution(
                _task_pairs(
                    trade_samples,
                    AUTHORITATIVE_REALIZED_RETURN_TASK,
                    "realized_net_return_pct",
                )
            ),
            "hold_minutes": weighted_distribution(
                _task_pairs(
                    trade_samples,
                    AUTHORITATIVE_REALIZED_RETURN_TASK,
                    "hold_minutes",
                )
            ),
            "stop_loss_slippage_pct": weighted_distribution(
                _task_pairs(
                    trade_samples,
                    AUTHORITATIVE_REALIZED_RETURN_TASK,
                    "stop_loss_slippage_pct",
                )
            ),
        },
        "correlation_group_count": len(decision_groups),
        "production_combination": {
            "version": PRODUCTION_RETURN_COMBINATION_VERSION,
            "formula": (
                "market_opportunity_distribution-live_execution_cost_distribution-"
                "authoritative_slippage_tail_excess"
            ),
            "shadow_point_prediction_is_realized_return": False,
        },
    }
    report["provenance"] = {
        "source": "separated_shadow_market_cost_and_okx_trade_supervision",
        "observation_window": "provided_immutable_training_view",
        "sample_count": len(all_samples),
        "effective_sample_size": round(
            sum(_safe_float(sample.get("sample_weight"), 0.0) or 0.0 for sample in all_samples),
            8,
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy_version": PROFIT_SUPERVISION_VERSION,
        "fallback_reason": (
            ""
            if shadow_market_count and actual_return_count
            else "market_or_authoritative_trade_supervision_missing"
        ),
        "data_fingerprint": _fingerprint(
            sorted(
                _safe_text(
                    _safe_dict(sample.get("profit_supervision")).get(
                        "contract_fingerprint"
                    )
                )
                for sample in all_samples
            )
        ),
    }
    return report


def authoritative_trade_calibration(
    trade_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build transparent symbol/side and global actual-trade distributions."""

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in trade_samples:
        contract = _safe_dict(sample.get("profit_supervision"))
        tasks = _safe_dict(contract.get("tasks"))
        realized = _safe_dict(tasks.get(AUTHORITATIVE_REALIZED_RETURN_TASK))
        if realized.get("eligible") is not True:
            continue
        side = _safe_text(realized.get("side")).lower()
        symbol = normalize_trading_symbol(sample.get("symbol"))
        if side not in {"long", "short"}:
            continue
        buckets[f"*|{side}"].append(sample)
        if symbol:
            buckets[f"{symbol}|{side}"].append(sample)

    profiles: dict[str, Any] = {}
    for key, samples in buckets.items():
        side = key.rsplit("|", 1)[-1]
        net_pairs = _task_pairs(
            samples,
            AUTHORITATIVE_REALIZED_RETURN_TASK,
            "realized_net_return_pct",
        )
        cost_pairs = _task_pairs(
            samples,
            COUNTERFACTUAL_EXECUTION_COST_TASK,
            "total_cost_pct",
        )
        slippage_pairs = _task_pairs(
            samples,
            COUNTERFACTUAL_EXECUTION_COST_TASK,
            "slippage_pct",
        )
        stop_pairs = _task_pairs(
            samples,
            AUTHORITATIVE_REALIZED_RETURN_TASK,
            "stop_loss_slippage_pct",
        )
        profiles[key] = {
            "source_authority": "okx_position_history",
            "symbol": key.rsplit("|", 1)[0],
            "side": side,
            "net_return_after_cost_pct": weighted_distribution(net_pairs),
            "execution_cost_pct": weighted_distribution(cost_pairs),
            "slippage_pct": weighted_distribution(slippage_pairs),
            "stop_loss_slippage_pct": weighted_distribution(stop_pairs),
            "hold_minutes": weighted_distribution(
                _task_pairs(
                    samples,
                    AUTHORITATIVE_REALIZED_RETURN_TASK,
                    "hold_minutes",
                )
            ),
            "exit_quality_counts": dict(
                Counter(
                    _safe_text(
                        _safe_dict(
                            _safe_dict(
                                _safe_dict(sample.get("profit_supervision")).get(
                                    "tasks"
                                )
                            ).get(AUTHORITATIVE_REALIZED_RETURN_TASK)
                        ).get("exit_quality_label")
                    )
                    or "unknown"
                    for sample in samples
                )
            ),
        }
    result = {
        "version": PROFIT_SUPERVISION_VERSION,
        "profiles": profiles,
        "actual_realized_return_sample_count": sum(
            1
            for sample in trade_samples
            if _safe_dict(
                _safe_dict(
                    _safe_dict(sample.get("profit_supervision")).get("tasks")
                ).get(AUTHORITATIVE_REALIZED_RETURN_TASK)
            ).get("eligible")
            is True
        ),
    }
    result["data_fingerprint"] = _fingerprint(result)
    return result


def select_trade_calibration(
    calibration: dict[str, Any] | None,
    *,
    symbol: str,
    side: str,
) -> dict[str, Any]:
    profiles = _safe_dict(_safe_dict(calibration).get("profiles"))
    normalized = normalize_trading_symbol(symbol)
    exact_key = f"{normalized}|{side}"
    global_key = f"*|{side}"
    if exact_key in profiles:
        return {**_safe_dict(profiles[exact_key]), "profile_source": "symbol_side"}
    if global_key in profiles:
        return {**_safe_dict(profiles[global_key]), "profile_source": "global_side"}
    return {
        "profile_source": "missing",
        "fallback_reason": "authoritative_trade_calibration_missing",
    }
