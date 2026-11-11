"""Round-trip execution cost from the account's current executable book."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from math import floor, isfinite
from typing import Any

from core.training_contracts import SHADOW_LABEL_MAX_SOURCE_LAG_SECONDS

EXECUTION_COST_FACT_FIELDS = (
    "taker_fee_rate",
    "maker_fee_rate",
    "entry_fee_rate",
    "exit_fee_rate",
    "round_trip_fee_pct",
    "fee_rate_source",
    "fee_rate_observed_at",
    "account_fee_received_at",
    "account_fee_source_observed_at",
    # The entry-leg policy travels with the fee facts so the cost model prices
    # the same execution the executor will actually use.
    "entry_execution_style",
    "post_only_fill_probability",
)
SHADOW_SETTLEMENT_COST_VERSION = "2026-11-11.observed-shadow-exit-cost.v2"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def execution_cost_provenance_blocker(snapshot: dict[str, Any]) -> str:
    """Explicit market-only provenance overrides copied microstructure numbers."""

    if snapshot.get("historical_market_path_only") is True:
        return "historical_market_path_has_no_executable_cost_evidence"
    for key in ("training_market_fact_contract", "market_fact_contract"):
        contract = _safe_dict(snapshot.get(key))
        assertions = _safe_dict(contract.get("assertions"))
        if (
            contract.get("status") == "historical_ohlcv_only"
            or contract.get("executable_market_fact_verified") is False
            or assertions.get("executable_market_fact_verified") is False
        ):
            return "market_fact_not_executable"
    return ""


def attach_execution_cost_facts(feature_vector: Any, facts: dict[str, Any] | None) -> Any:
    """Attach account fee facts to one live market snapshot without guessing values."""

    payload = facts if isinstance(facts, dict) else {}
    for key in EXECUTION_COST_FACT_FIELDS:
        value = payload.get(key)
        if value in {None, ""}:
            continue
        if isinstance(feature_vector, dict):
            feature_vector[key] = value
        else:
            setattr(feature_vector, key, value)
    provenance = payload.get("policy_provenance")
    if isinstance(provenance, dict):
        if isinstance(feature_vector, dict):
            feature_vector["fee_policy_provenance"] = dict(provenance)
        else:
            feature_vector.fee_policy_provenance = dict(provenance)
    return feature_vector


@dataclass(frozen=True, slots=True)
class ExecutionCostEstimate:
    """Round-trip fee and dynamic slippage estimate in percentage points."""

    fee_pct: float
    slippage_pct: float
    total_pct: float
    spread_pct: float
    spread_source: str
    bid_depth_usdt: float
    ask_depth_usdt: float
    orderbook_imbalance: float
    liquidity_penalty_pct: float
    imbalance_penalty_pct: float
    market_impact_pct: float
    order_notional_usdt: float
    order_side: str
    order_size_complete: bool
    observed_side_depth_usdt: float
    depth_consumption_fraction: float
    estimated_vwap: float
    reference_price: float
    book_levels_consumed: int
    slippage_source: str
    production_eligible: bool
    reason: str
    policy_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FundingCostEstimate:
    """Direction-aware adverse funding cost over one prediction horizon."""

    side: str
    funding_rate: float | None
    funding_interval_minutes: float | None
    horizon_minutes: float | None
    estimated_settlement_count: int
    next_funding_time: str | None
    funding_rate_observed_at: str | None
    signed_cashflow_pct: float | None
    adverse_cost_pct: float | None
    signed_return_impact_pct: float | None
    production_eligible: bool
    reason: str
    policy_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def funding_cost_estimate(
    feature_snapshot: dict[str, Any] | None,
    *,
    side: str,
    horizon_minutes: Any,
) -> FundingCostEstimate:
    """Project signed funding cashflow using the actual upcoming settlements."""

    snapshot = feature_snapshot if isinstance(feature_snapshot, dict) else {}
    normalized_side = str(side or "").strip().lower()
    horizon = _optional_finite_float(horizon_minutes)
    rate = _optional_finite_float(snapshot.get("funding_rate"))
    interval = _optional_finite_float(snapshot.get("funding_interval_minutes"))
    if interval is None:
        interval_hours = _optional_finite_float(snapshot.get("funding_interval_hours"))
        interval = interval_hours * 60.0 if interval_hours is not None else None

    reason = ""
    next_funding_raw = str(snapshot.get("next_funding_time") or "").strip()
    observed_raw = str(snapshot.get("funding_rate_observed_at") or "").strip()
    next_funding = _parse_timestamp(next_funding_raw)
    observed_at = _parse_timestamp(observed_raw)
    as_of_raw = (
        snapshot.get("original_decision_timestamp")
        or snapshot.get("timestamp")
        or snapshot.get("funding_as_of")
    )
    parsed_as_of = _parse_timestamp(str(as_of_raw or ""))
    as_of = parsed_as_of or datetime.now(UTC)
    legacy_projection = not bool(as_of_raw)
    if normalized_side not in {"long", "short"}:
        reason = "funding_side_invalid"
    elif snapshot.get("funding_data_available") is not True:
        reason = "funding_data_unavailable"
    elif rate is None:
        reason = "funding_rate_missing"
    elif interval is None or interval <= 0.0:
        reason = "funding_interval_missing"
    elif horizon is None or horizon <= 0.0:
        reason = "prediction_horizon_missing"
    elif as_of_raw and parsed_as_of is None:
        reason = "funding_as_of_invalid"
    elif not next_funding and not legacy_projection:
        reason = "next_funding_time_missing"
    elif not observed_at and not legacy_projection:
        reason = "funding_rate_observed_at_missing"
    elif observed_at and observed_at > as_of and not legacy_projection:
        reason = "funding_rate_observed_after_decision"
    elif observed_at and as_of > observed_at and not legacy_projection:
        max_age = max(interval * 2.0 if interval and interval > 0 else 480.0, 30.0)
        if (as_of - observed_at).total_seconds() > max_age * 60.0:
            reason = "funding_rate_stale"

    eligible = not reason
    signed_impact = None
    adverse_cost = None
    settlement_count = 0
    if eligible and rate is not None and interval is not None and horizon is not None:
        if next_funding is not None:
            # Count (decision, exit] settlements directly. Iterating from an
            # old next-funding timestamp can otherwise stall historical replay.
            step_seconds = interval * 60.0
            first = max(floor((as_of - next_funding).total_seconds() / step_seconds) + 1, 0)
            last = floor(
                ((as_of - next_funding).total_seconds() + horizon * 60.0) / step_seconds
            )
            settlement_count = max(last - first + 1, 0)
            if settlement_count > 1000:
                eligible = False
                reason = "funding_settlement_count_exceeds_limit"
            else:
                signed_impact = (
                    (-rate if normalized_side == "long" else rate)
                    * 100.0
                    * settlement_count
                )
        else:
            # Old stored/test snapshots predate next_funding_time. Keep their
            # result readable, but mark the method so production snapshots can
            # never silently fall back to continuous prorating.
            signed_impact = (
                (-rate if normalized_side == "long" else rate)
                * 100.0
                * horizon
                / interval
            )
        if signed_impact is not None:
            adverse_cost = max(-signed_impact, 0.0)

    signed_impact = round(signed_impact, 10) if signed_impact is not None else None
    return FundingCostEstimate(
        side=normalized_side,
        funding_rate=rate,
        funding_interval_minutes=interval,
        horizon_minutes=horizon,
        estimated_settlement_count=settlement_count,
        next_funding_time=next_funding_raw or None,
        funding_rate_observed_at=observed_raw or None,
        signed_cashflow_pct=signed_impact,
        adverse_cost_pct=(round(adverse_cost, 10) if adverse_cost is not None else None),
        signed_return_impact_pct=signed_impact,
        production_eligible=eligible,
        reason=(
            "current_direction_funding_cashflow_ready"
            if eligible
            else reason
        ),
        policy_provenance={
            "source": "okx_current_perpetual_funding_rate",
            "observation_window": "current_prediction_horizon",
            "sample_count": int(eligible),
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-11-10.decision-clock-funding-settlements.v2",
            "fallback_reason": "" if eligible else reason,
            "funding_rate_observed_at": observed_raw,
            "income_credit_policy": "signed_cashflow_positive_income_negative_cost",
            "estimated_settlement_count": settlement_count,
            "projection_method": (
                "next_funding_time_and_interval"
                if next_funding is not None
                else "legacy_interval_fallback"
            ),
            "funding_evidence_status": (
                "complete" if eligible and next_funding is not None else "legacy_degraded"
                if eligible
                else "unavailable"
            ),
        },
    )


def _parse_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        epoch = float(text)
    except ValueError:
        epoch = 0.0
    if epoch >= 1_000_000_000.0:
        if epoch >= 1_000_000_000_000_000.0:
            epoch /= 1_000_000.0
        elif epoch >= 1_000_000_000_000.0:
            epoch /= 1_000.0
        try:
            return datetime.fromtimestamp(epoch, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def execution_cost_estimate(feature_snapshot: dict[str, Any] | None) -> ExecutionCostEstimate:
    """Estimate round-trip execution costs from the current market snapshot."""

    snapshot = feature_snapshot if isinstance(feature_snapshot, dict) else {}
    spread_pct, spread_source = _normalized_spread_pct(snapshot)
    bid_depth = max(_safe_float(snapshot.get("orderbook_bid_depth"), 0.0), 0.0)
    ask_depth = max(_safe_float(snapshot.get("orderbook_ask_depth"), 0.0), 0.0)
    imbalance = min(max(_safe_float(snapshot.get("orderbook_imbalance"), 0.0), -1.0), 1.0)

    half_spread_pct = max(spread_pct / 2.0, 0.0)
    depth_asymmetry = (
        abs(bid_depth - ask_depth) / (bid_depth + ask_depth)
        if bid_depth > 0 and ask_depth > 0
        else 0.0
    )
    liquidity_penalty_pct = half_spread_pct * depth_asymmetry
    imbalance_penalty_pct = half_spread_pct * abs(imbalance)
    impact = _orderbook_market_impact(snapshot)
    exit_snapshot = {
        **snapshot,
        "planned_order_side": (
            "short" if snapshot.get("planned_order_side") == "long" else "long"
        ),
    }
    exit_impact = _orderbook_market_impact(exit_snapshot)
    round_trip_impact_ready = bool(
        impact.get("eligible") is True and exit_impact.get("eligible") is True
    )
    market_impact_pct = max(_safe_float(impact.get("market_impact_pct"), 0.0), 0.0)
    order_size_slippage_pct = max(
        _safe_float(impact.get("execution_slippage_pct"), 0.0),
        0.0,
    )
    raw_slippage_pct = (
        order_size_slippage_pct
        + max(_safe_float(exit_impact.get("execution_slippage_pct"), 0.0), 0.0)
        if round_trip_impact_ready
        else 2.0 * (half_spread_pct + liquidity_penalty_pct + imbalance_penalty_pct)
    )
    slippage_pct = raw_slippage_pct
    fee_pct, fee_source = round_trip_fee_pct(snapshot)
    order_notional = max(_safe_float(snapshot.get("planned_order_notional_usdt"), 0.0), 0.0)
    order_size_complete = order_notional > 0 and round_trip_impact_ready
    base_eligible = bool(spread_source != "missing" and spread_pct > 0 and fee_pct > 0)
    provenance_blocker = execution_cost_provenance_blocker(snapshot)
    production_eligible = bool(
        not provenance_blocker
        and base_eligible
        and (order_notional <= 0 or order_size_complete)
    )
    failure_reason = (
        provenance_blocker
        if provenance_blocker
        else "fee_rate_missing"
        if fee_pct <= 0
        else "live_spread_missing"
        if spread_source == "missing" or spread_pct <= 0
        else str(
            impact.get("reason")
            if impact.get("eligible") is not True
            else "planned_close_exceeds_observed_orderbook_depth"
        )
        if order_notional > 0 and not order_size_complete
        else ""
    )
    slippage_pct = round(slippage_pct, 6)
    return ExecutionCostEstimate(
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
        total_pct=fee_pct + slippage_pct,
        spread_pct=round(spread_pct, 6),
        spread_source=spread_source,
        bid_depth_usdt=round(bid_depth, 6),
        ask_depth_usdt=round(ask_depth, 6),
        orderbook_imbalance=round(imbalance, 6),
        liquidity_penalty_pct=round(liquidity_penalty_pct, 6),
        imbalance_penalty_pct=round(imbalance_penalty_pct, 6),
        market_impact_pct=round(market_impact_pct, 6),
        order_notional_usdt=round(order_notional, 8),
        order_side=str(snapshot.get("planned_order_side") or "").lower(),
        order_size_complete=order_size_complete,
        observed_side_depth_usdt=round(
            max(_safe_float(impact.get("observed_side_depth_usdt"), 0.0), 0.0),
            8,
        ),
        depth_consumption_fraction=round(
            max(_safe_float(impact.get("depth_consumption_fraction"), 0.0), 0.0),
            8,
        ),
        estimated_vwap=round(max(_safe_float(impact.get("estimated_vwap"), 0.0), 0.0), 12),
        reference_price=round(
            max(_safe_float(impact.get("reference_price"), 0.0), 0.0),
            12,
        ),
        book_levels_consumed=max(int(_safe_float(impact.get("book_levels_consumed"), 0.0)), 0),
        slippage_source=(
            "dynamic_live_spread_depth_imbalance_and_order_vwap"
            if order_size_complete and production_eligible
            else "pre_sizing_live_spread_depth_imbalance"
            if production_eligible
            else "observation_only_nonexecutable_market_fact"
            if provenance_blocker
            else "observation_only_missing_live_spread"
        ),
        production_eligible=production_eligible,
        reason=(
            "live_order_size_microstructure_cost_ready"
            if order_size_complete and production_eligible
            else "pre_sizing_microstructure_cost_ready"
            if production_eligible
            else failure_reason
        ),
        policy_provenance={
            "source": f"{fee_source}+live_bid_ask_depth_imbalance_and_order_vwap",
            "observation_window": "current_pre_order_native_orderbook_snapshot",
            "sample_count": (
                int(spread_source != "missing")
                + int(bid_depth > 0 and ask_depth > 0)
                + max(int(_safe_float(impact.get("book_levels_consumed"), 0.0)), 0)
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-10-22.round-trip-executable-provenance.v4",
            "execution_mode": snapshot.get("execution_mode"),
            "entry_slippage_pct": (
                impact.get("execution_slippage_pct") if round_trip_impact_ready else None
            ),
            "exit_slippage_pct": (
                exit_impact.get("execution_slippage_pct") if round_trip_impact_ready else None
            ),
            "fallback_reason": "" if production_eligible else failure_reason,
            "orderbook_fingerprint": str(
                _safe_dict(snapshot.get("pre_order_execution_facts")).get(
                    "input_fingerprint"
                )
                or ""
            ),
        },
    )


def shadow_execution_cost_estimate(
    feature_snapshot: dict[str, Any] | None,
    *,
    side: str,
) -> ExecutionCostEstimate:
    """Price label costs with an observed exit; never use this for live inputs."""

    snapshot = _safe_dict(feature_snapshot)
    entry_snapshot = {**snapshot, "planned_order_side": side}
    # A counterfactual book observation cannot prove a passive fill.
    entry_snapshot.pop("entry_execution_style", None)
    entry = execution_cost_estimate(entry_snapshot)
    observation = _safe_dict(snapshot.get("shadow_settlement_cost_observation"))
    basis = "entry_snapshot_round_trip_projection"
    if not observation or observation.get("version") == "2026-10-22.immutable-shadow-entry-cost.v1":
        return replace(entry, policy_provenance={
            **entry.policy_provenance,
            "label_cost_basis": basis,
            "execution_replay_complete": False,
        })

    exit_snapshot = dict(_safe_dict(observation.get("market_snapshot")))
    entry_price = _optional_finite_float(observation.get("entry_price"))
    exit_price = _optional_finite_float(observation.get("exit_price"))
    reason = ""
    source_at = _parse_timestamp(str(observation.get("source_timestamp_ms") or ""))
    received_at = _parse_timestamp(str(observation.get("received_at") or ""))
    due_at = _parse_timestamp(str(observation.get("scheduled_label_at") or ""))
    if observation.get("version") != SHADOW_SETTLEMENT_COST_VERSION:
        reason = "shadow_exit_cost_version_unknown"
    elif side not in {"long", "short"}:
        reason = "shadow_cost_side_invalid"
    elif observation.get("used_as_decision_time_features") is not False:
        reason = "shadow_exit_observation_not_label_only"
    elif observation.get("market_fact_verified") is not True:
        reason = "shadow_exit_market_fact_unverified"
    elif not observation.get("result_fact_id"):
        reason = "shadow_exit_fact_identity_missing"
    elif source_at is None or received_at is None or due_at is None:
        reason = "shadow_exit_clock_missing"
    elif not (due_at <= source_at <= received_at):
        reason = "shadow_exit_clock_invalid"
    elif (source_at - due_at).total_seconds() > SHADOW_LABEL_MAX_SOURCE_LAG_SECONDS:
        reason = "shadow_exit_observation_late"
    elif entry_price is None or entry_price <= 0 or exit_price is None or exit_price <= 0:
        reason = "shadow_exit_reference_price_missing"
    lineage = _safe_dict(
        snapshot.get("training_market_fact_contract") or snapshot.get("market_fact_contract")
    )
    expected_result_id = lineage.get("result_fact_id") or _safe_dict(
        lineage.get("result_fact")
    ).get("fact_id")
    if not reason and expected_result_id and expected_result_id != observation.get("result_fact_id"):
        reason = "shadow_exit_fact_identity_mismatch"

    # The exit closes the same base quantity, not the original quote notional.
    exit_snapshot["planned_order_side"] = "short" if side == "long" else "long"
    notional = _optional_finite_float(snapshot.get("planned_order_notional_usdt"))
    if notional is not None and notional > 0:
        entry_mid = entry.reference_price
        exit_bid = _optional_finite_float(exit_snapshot.get("bid"))
        exit_ask = _optional_finite_float(exit_snapshot.get("ask"))
        exit_mid = (
            (exit_bid + exit_ask) / 2.0
            if exit_bid is not None and exit_ask is not None else 0.0
        )
        exit_snapshot["planned_order_notional_usdt"] = (
            notional * exit_mid / entry_mid if entry_mid > 0 else notional
        )
    entry_impact = _orderbook_market_impact(entry_snapshot)
    exit_impact = _orderbook_market_impact(exit_snapshot)
    exit_estimate = execution_cost_estimate(exit_snapshot)
    entry_rate, _ = entry_fee_rate(entry_snapshot)
    exit_rate = _normalized_fee_rate(
        exit_snapshot.get("exit_fee_rate", exit_snapshot.get("taker_fee_rate"))
    )
    sized = notional is not None and notional > 0
    entry_ready = (
        not execution_cost_provenance_blocker(entry_snapshot)
        and entry.fee_pct > 0 and entry.spread_pct > 0
        and entry_impact.get("eligible") is True
    ) if sized else entry.production_eligible
    exit_ready = (
        exit_estimate.fee_pct > 0 and exit_estimate.spread_pct > 0
        and exit_impact.get("eligible") is True
    ) if sized else exit_estimate.production_eligible
    if not reason and not entry_ready:
        reason = entry.reason
    if not reason and not exit_ready:
        reason = f"shadow_exit:{exit_estimate.reason}"
    if not reason and (entry_rate <= 0 or exit_rate <= 0):
        reason = "shadow_per_leg_fee_rate_missing"
    ratio = exit_price / entry_price if entry_price and exit_price else 1.0
    entry_slippage = (
        _safe_float(entry_impact.get("execution_slippage_pct"), 0.0)
        if sized else entry.slippage_pct / 2.0
    )
    exit_slippage = (
        _safe_float(exit_impact.get("execution_slippage_pct"), 0.0)
        if sized else exit_estimate.slippage_pct / 2.0
    )
    fee_pct = (entry_rate + exit_rate * ratio) * 100.0
    slippage_pct = entry_slippage + exit_slippage * ratio
    return replace(
        entry,
        fee_pct=round(fee_pct, 8),
        slippage_pct=round(slippage_pct, 8),
        total_pct=round(fee_pct + slippage_pct, 8),
        order_size_complete=bool(sized and entry_ready and exit_ready),
        production_eligible=not reason,
        reason=reason or "shadow_observed_exit_microstructure_cost_ready",
        policy_provenance={
            **entry.policy_provenance,
            "label_cost_basis": "entry_and_observed_exit_microstructure",
            "strategy_version": SHADOW_SETTLEMENT_COST_VERSION,
            "entry_slippage_pct": entry_slippage,
            "exit_slippage_pct": exit_slippage,
            "exit_notional_ratio": ratio,
            "exit_fact_id": observation.get("result_fact_id"),
            "exit_source_timestamp_ms": observation.get("source_timestamp_ms"),
            "maker_fill_policy": "taker_counterfactual_without_assumed_passive_fill",
            "actual_execution": False,
            "execution_replay_complete": False,
            "fallback_reason": reason,
        },
    )


def _orderbook_market_impact(snapshot: dict[str, Any]) -> dict[str, Any]:
    order_notional = max(_safe_float(snapshot.get("planned_order_notional_usdt"), 0.0), 0.0)
    side = str(snapshot.get("planned_order_side") or "").lower()
    if order_notional <= 0:
        return {"eligible": False, "reason": "planned_order_notional_missing"}
    if side not in {"long", "short"}:
        return {"eligible": False, "reason": "planned_order_side_missing"}

    bid = max(_safe_float(snapshot.get("bid"), 0.0), 0.0)
    ask = max(_safe_float(snapshot.get("ask"), 0.0), 0.0)
    reference = (bid + ask) / 2.0 if bid > 0 and ask >= bid else 0.0
    contract_base = max(
        _safe_float(snapshot.get("contract_value_base"), 0.0),
        0.0,
    )
    levels = snapshot.get("orderbook_asks" if side == "long" else "orderbook_bids")
    levels = levels if isinstance(levels, list) else []
    if reference <= 0:
        return {"eligible": False, "reason": "orderbook_reference_price_missing"}
    if contract_base <= 0:
        return {"eligible": False, "reason": "contract_value_base_missing"}
    if not levels:
        return {"eligible": False, "reason": "executable_orderbook_levels_missing"}

    target_base = order_notional / reference
    remaining_base = target_base
    filled_base = 0.0
    quote_cost = 0.0
    observed_depth = 0.0
    consumed = 0
    best_price = 0.0
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price = max(_safe_float(level[0], 0.0), 0.0)
        contracts = max(_safe_float(level[1], 0.0), 0.0)
        level_base = contracts * contract_base
        if price <= 0 or level_base <= 0:
            continue
        if best_price <= 0:
            best_price = price
        observed_depth += level_base * price
        if remaining_base <= 0:
            continue
        used_base = min(remaining_base, level_base)
        filled_base += used_base
        quote_cost += used_base * price
        remaining_base -= used_base
        consumed += 1

    if filled_base <= 0 or remaining_base > 1e-12:
        return {
            "eligible": False,
            "reason": "planned_order_exceeds_observed_orderbook_depth",
            "observed_side_depth_usdt": observed_depth,
            "depth_consumption_fraction": (
                order_notional / observed_depth if observed_depth > 0 else 0.0
            ),
            "book_levels_consumed": consumed,
            "reference_price": reference,
        }

    vwap = quote_cost / filled_base
    adverse_impact = (
        max(vwap - best_price, 0.0)
        if side == "long"
        else max(best_price - vwap, 0.0)
    )
    return {
        "eligible": True,
        "reason": "order_size_vwap_ready",
        "market_impact_pct": adverse_impact / reference * 100.0,
        "execution_slippage_pct": (
            max(vwap - reference, 0.0) / reference * 100.0
            if side == "long"
            else max(reference - vwap, 0.0) / reference * 100.0
        ),
        "observed_side_depth_usdt": observed_depth,
        "depth_consumption_fraction": (
            order_notional / observed_depth if observed_depth > 0 else 0.0
        ),
        "estimated_vwap": vwap,
        "reference_price": reference,
        "book_levels_consumed": consumed,
    }


def entry_fee_rate(snapshot: dict[str, Any]) -> tuple[float, str]:
    """Return the fee rate the entry leg is expected to pay.

    A post-only entry only earns the maker rate for the share of orders that
    actually rest and fill; the rest pay taker (plus slippage). The fill rate
    is an explicit measured input, so with no measured evidence the entry leg
    stays on the taker basis and the model cannot invent a maker discount.
    """

    taker_rate = _normalized_fee_rate(
        snapshot.get("taker_fee_rate", snapshot.get("fee_rate"))
    )
    # An explicit entry rate outranks the account taker rate, matching the
    # historical precedence. It is still the basis a maker fill improves on.
    explicit_entry = _normalized_fee_rate(snapshot.get("entry_fee_rate"))
    base_entry = explicit_entry if explicit_entry > 0 else taker_rate
    if base_entry <= 0:
        return 0.0, "missing_exchange_fee_rate"
    if snapshot.get("entry_execution_style") != "post_only_with_market_fallback":
        return base_entry, "taker_entry"
    maker_rate = _normalized_fee_rate(snapshot.get("maker_fee_rate"))
    if maker_rate <= 0:
        return base_entry, "taker_entry_maker_rate_missing"
    probability = _safe_float(snapshot.get("post_only_fill_probability"), 0.0)
    if probability <= 0.0:
        return base_entry, "taker_entry_no_measured_maker_fill_rate"
    probability = min(max(probability, 0.0), 1.0)
    blended = probability * maker_rate + (1.0 - probability) * base_entry
    return blended, "post_only_measured_maker_share"


def round_trip_fee_pct(snapshot: dict[str, Any]) -> tuple[float, str]:
    explicit = _safe_float(snapshot.get("round_trip_fee_pct"), 0.0)
    if explicit > 0:
        return explicit, "snapshot_round_trip_fee_pct"
    exit_rate = _normalized_fee_rate(
        snapshot.get("exit_fee_rate", snapshot.get("taker_fee_rate", snapshot.get("fee_rate")))
    )
    entry_rate, entry_source = entry_fee_rate(snapshot)
    if entry_rate <= 0 or exit_rate <= 0:
        return 0.0, "missing_exchange_fee_rate"
    return (
        round((entry_rate + exit_rate) * 100.0, 6),
        f"snapshot_exchange_fee_rates.{entry_source}",
    )


def _normalized_fee_rate(value: Any) -> float:
    # Exchange fee-rate fields use decimal ratios. Percent values must use
    # the explicit round_trip_fee_pct field instead of unit guessing here.
    return max(_safe_float(value, 0.0), 0.0)


def _normalized_spread_pct(snapshot: dict[str, Any]) -> tuple[float, str]:
    spread = _safe_float(snapshot.get("spread_pct"), 0.0)
    if spread > 0:
        return spread, "spread_pct"

    bid = _safe_float(snapshot.get("bid"), 0.0)
    ask = _safe_float(snapshot.get("ask"), 0.0)
    if bid > 0 and ask >= bid:
        mid = (bid + ask) / 2.0
        if mid > 0:
            return (ask - bid) / mid * 100.0, "bid_ask"
    return 0.0, "missing"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
        return number if isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _optional_finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None
