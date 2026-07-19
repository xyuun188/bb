"""Build model-training samples from authoritative OKX SWAP lifecycles."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_list(item))
        return list(dict.fromkeys(token for token in result if token))
    text = _text(value)
    if not text:
        return []
    tokens = {text}
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return sorted(tokens)


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _raw_contract_spec(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("_bb_contract_spec")
    return dict(value) if isinstance(value, dict) else {}


def _order_trade_ids(orders: Iterable[Any]) -> list[str]:
    return list(
        dict.fromkeys(
            trade_id
            for order in orders
            for trade_id in _list(_value(order, "okx_trade_ids"))
            if trade_id
        )
    )


def _entry_fill_contracts(entry_orders: Iterable[Any]) -> float | None:
    values = [
        _safe_float(_value(order, "okx_fill_contracts"), None) for order in entry_orders
    ]
    valid = [value for value in values if value is not None and value > 0]
    return sum(valid) if valid else None


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _protection_execution(order: Any) -> dict[str, Any]:
    raw = _dict(_value(order, "okx_raw_fills", {}))
    execution = _dict(raw.get("protection_execution"))
    if (
        execution.get("lifecycle_complete") is True
        and _text(execution.get("source_authority"))
        == "okx_algo_history_plus_fills_history"
        and _text(execution.get("actual_side")).lower() in {"sl", "tp"}
    ):
        return execution
    return {}


def _protection_submission(order: Any) -> dict[str, Any]:
    raw = _dict(_value(order, "okx_raw_fills", {}))
    submission = _dict(raw.get("protection_submission"))
    if (
        submission.get("exchange_confirmation_recorded") is True
        and _text(submission.get("source_authority"))
        == "local_submit_plus_okx_create_order_response"
    ):
        return submission
    return {}


def _first_protection_execution(orders: Iterable[Any]) -> dict[str, Any]:
    return next(
        (execution for order in orders if (execution := _protection_execution(order))),
        {},
    )


def _first_protection_submission(orders: Iterable[Any]) -> dict[str, Any]:
    return next(
        (submission for order in orders if (submission := _protection_submission(order))),
        {},
    )


def _iso_from_ms(value: Any) -> str | None:
    timestamp_ms = _safe_float(value, None)
    if timestamp_ms is None or timestamp_ms <= 0:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).isoformat()


def _execution_budget_facts(
    *,
    raw_llm_response: dict[str, Any],
    realized_pnl: float,
) -> dict[str, float | None]:
    sizing = _dict(raw_llm_response.get("profit_risk_sizing"))
    risk_budget = _safe_float(sizing.get("risk_budget_usdt"), None)
    planned_loss = _safe_float(sizing.get("planned_stressed_loss_usdt"), None)
    actual_loss = max(-realized_pnl, 0.0)
    return {
        "risk_budget_usdt": risk_budget if risk_budget is not None and risk_budget > 0 else None,
        "planned_stressed_loss_usdt": (
            planned_loss if planned_loss is not None and planned_loss >= 0 else None
        ),
        "actual_loss_usdt": actual_loss,
        "actual_over_budget_loss_usdt": (
            max(actual_loss - risk_budget, 0.0)
            if risk_budget is not None and risk_budget > 0
            else None
        ),
    }


def _official_ratio_pct(raw: dict[str, Any], fallback: Any) -> float | None:
    ratio = _safe_float(raw.get("pnlRatio"), None)
    if ratio is None:
        ratio = _safe_float(fallback, None)
    return ratio * 100.0 if ratio is not None else None


def _has_raw_key(raw: dict[str, Any], *keys: str) -> bool:
    return any(key in raw and raw.get(key) not in (None, "") for key in keys)


def _canonical_execution_mode(value: Any) -> str:
    mode = _text(value).lower()
    if mode in {"paper", "demo", "sim", "simulation"}:
        return "paper"
    if mode in {"live", "real", "production"}:
        return "live"
    return ""


def build_okx_history_training_sample(
    history: Any,
    *,
    positions_by_id: dict[int, Any] | None = None,
    orders_by_exchange_id: dict[str, Any] | None = None,
    decision_raw_by_position_id: dict[int, dict[str, Any]] | None = None,
    decision_raw_by_order_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert one mirrored OKX positions-history lifecycle into one sample."""

    positions_by_id = positions_by_id or {}
    orders_by_exchange_id = orders_by_exchange_id or {}
    decision_raw_by_position_id = decision_raw_by_position_id or {}
    decision_raw_by_order_id = decision_raw_by_order_id or {}
    raw = dict(_value(history, "raw_row", {}) or {})
    position_ids = [int(value) for value in _list(_value(history, "position_ids")) if value.isdigit()]
    entry_order_ids = _list(_value(history, "entry_order_ids"))
    close_order_ids = _list(_value(history, "close_order_ids"))
    linked_order_ids = list(dict.fromkeys([*entry_order_ids, *close_order_ids]))
    entry_orders = [orders_by_exchange_id[value] for value in entry_order_ids if value in orders_by_exchange_id]
    close_orders = [
        orders_by_exchange_id[value]
        for value in close_order_ids
        if value in orders_by_exchange_id
    ]
    linked_orders = [orders_by_exchange_id[value] for value in linked_order_ids if value in orders_by_exchange_id]
    local_positions = [positions_by_id[value] for value in position_ids if value in positions_by_id]
    local_position = local_positions[0] if local_positions else None

    opened_at = _as_utc(_value(history, "opened_at"))
    closed_at = _as_utc(_value(history, "updated_at_okx"))
    hold_minutes = (
        max((closed_at - opened_at).total_seconds() / 60.0, 0.0)
        if opened_at and closed_at
        else 0.0
    )
    entry_price = _safe_float(_value(history, "open_avg_px"), 0.0) or 0.0
    exit_price = _safe_float(_value(history, "close_avg_px"), 0.0) or 0.0
    open_contracts = _safe_float(_value(history, "open_max_pos"), 0.0) or 0.0
    fill_contracts = _entry_fill_contracts(entry_orders)
    contracts = fill_contracts or open_contracts
    spec = _raw_contract_spec(raw)
    ct_val = _safe_float(spec.get("ctVal"), None)
    ct_mult = _safe_float(spec.get("ctMult"), None)
    lot_size = _safe_float(spec.get("lotSz"), None)
    notional = (
        abs(contracts * ct_val * ct_mult * entry_price)
        if contracts > 0 and ct_val and ct_mult and entry_price > 0
        else None
    )

    realized_pnl = _safe_float(_value(history, "realized_pnl"), 0.0) or 0.0
    gross_pnl = _safe_float(_value(history, "pnl"), 0.0) or 0.0
    fee_signed = _safe_float(_value(history, "fee"), 0.0) or 0.0
    funding_fee = _safe_float(_value(history, "funding_fee"), 0.0) or 0.0
    liquidation_penalty = _safe_float(
        raw.get("liqPenalty") or raw.get("liquidationPenalty"), 0.0
    ) or 0.0
    settlement_expected = gross_pnl + fee_signed + funding_fee + liquidation_penalty
    settlement_tolerance = max(1e-6, abs(realized_pnl) * 1e-5)
    source_execution_mode = _text(_value(history, "mode")).lower()
    execution_mode = _canonical_execution_mode(source_execution_mode)

    gaps: list[str] = []
    if not execution_mode:
        gaps.append("missing_or_invalid_execution_mode")
    if _text(_value(history, "sync_status")).lower() != "synced":
        gaps.append("history_sync_not_confirmed")
    if _text(_value(history, "close_status")).lower() != "full":
        gaps.append("lifecycle_not_fully_closed")
    if not _text(_value(history, "pos_id")):
        gaps.append("missing_okx_pos_id")
    if _text(_value(history, "side")).lower() not in {"long", "short"}:
        gaps.append("missing_position_side")
    if entry_price <= 0:
        gaps.append("missing_open_average_price")
    if exit_price <= 0:
        gaps.append("missing_close_average_price")
    if not _has_raw_key(raw, "realizedPnl", "realized_pnl"):
        gaps.append("missing_official_realized_pnl")
    if not _has_raw_key(raw, "fee", "totalFee", "total_fee"):
        gaps.append("missing_official_fee")
    if not _has_raw_key(raw, "fundingFee", "funding_fee"):
        gaps.append("missing_official_funding_fee")
    if ct_val is None or ct_val <= 0:
        gaps.append("missing_contract_ct_val")
    if ct_mult is None or ct_mult <= 0:
        gaps.append("missing_contract_ct_mult")
    if lot_size is None or lot_size <= 0:
        gaps.append("missing_contract_lot_size")
    if contracts <= 0:
        gaps.append("missing_fill_or_open_contracts")
    if abs(realized_pnl - settlement_expected) > settlement_tolerance:
        gaps.append("settlement_algebra_mismatch")
    gaps = list(dict.fromkeys(gaps))

    position_id = position_ids[0] if position_ids else 0
    raw_llm_response: dict[str, Any] = {}
    decision_id = 0
    decision_lineage_source = "missing"
    for order_id in entry_order_ids:
        order = orders_by_exchange_id.get(order_id)
        order_decision_id = int(_value(order, "decision_id", 0) or 0)
        if decision_id <= 0 and order_decision_id > 0:
            decision_id = order_decision_id
            decision_lineage_source = "exact_entry_order_decision_id"
        candidate = decision_raw_by_order_id.get(order_id)
        if isinstance(candidate, dict) and candidate:
            raw_llm_response = candidate
            decision_lineage_source = "exact_entry_order_decision_payload"
            break
    if not raw_llm_response:
        raw_llm_response = next(
            (
                decision_raw_by_position_id[value]
                for value in position_ids
                if value in decision_raw_by_position_id
            ),
            {},
        )
        if raw_llm_response:
            decision_lineage_source = "position_time_fallback_payload"

    stop_loss_price = _safe_float(_value(local_position, "stop_loss_price"), None)
    take_profit_price = _safe_float(_value(local_position, "take_profit_price"), None)
    side = _text(_value(history, "side")).lower()
    protection_execution = _first_protection_execution(close_orders)
    protection_submission = _first_protection_submission(entry_orders)
    stop_loss_fill_confirmed = bool(
        protection_execution
        and _text(protection_execution.get("actual_side")).lower() == "sl"
    )
    stop_loss_slippage_pct = (
        _safe_float(protection_execution.get("stop_loss_slippage_pct"), None)
        if stop_loss_fill_confirmed
        and _text(protection_execution.get("stop_loss_slippage_source"))
        == "okx_configured_stop_trigger_to_fills_vwap"
        else None
    )
    protection_execution_gaps: list[str] = []
    if protection_execution:
        if not protection_submission:
            protection_execution_gaps.append("missing_client_protection_submission_confirmation")
        if protection_execution.get("actual_trigger_market_price_available") is not True:
            protection_execution_gaps.append("actual_trigger_market_price_unavailable")
        if protection_execution.get("trigger_path_extrema_available") is not True:
            protection_execution_gaps.append("trigger_path_extrema_unavailable")
        if protection_execution.get("trigger_orderbook_snapshot_available") is not True:
            protection_execution_gaps.append("trigger_orderbook_snapshot_unavailable")
    budget_facts = _execution_budget_facts(
        raw_llm_response=raw_llm_response,
        realized_pnl=realized_pnl,
    )
    lineage_gaps: list[str] = []
    if not entry_order_ids:
        lineage_gaps.append("missing_position_history_entry_orders")
    elif not entry_orders:
        lineage_gaps.append("missing_loaded_entry_order_facts")
    if not close_order_ids:
        lineage_gaps.append("missing_position_history_close_orders")
    elif not any(order_id in orders_by_exchange_id for order_id in close_order_ids):
        lineage_gaps.append("missing_loaded_close_order_facts")
    if decision_id <= 0:
        lineage_gaps.append("missing_exact_entry_order_decision_link")
    if decision_lineage_source != "exact_entry_order_decision_payload":
        lineage_gaps.append("missing_exact_entry_order_decision_payload")
    if local_position is None:
        lineage_gaps.append("missing_local_position_strategy_lineage")
    if stop_loss_price is None or stop_loss_price <= 0:
        lineage_gaps.append("missing_planned_stop_loss_lineage")
    if take_profit_price is None or take_profit_price <= 0:
        lineage_gaps.append("missing_planned_take_profit_lineage")
    lineage_gaps = list(dict.fromkeys(lineage_gaps))
    model_name = _text(_value(local_position, "model_name")) if local_position else ""
    official_ratio_pct = _official_ratio_pct(raw, _value(history, "pnl_ratio"))
    lifecycle_key = _text(_value(history, "row_identity"))
    history_source = _text(_value(history, "source"))
    verified_execution_pair = history_source == "okx_verified_execution_pair_settlement"
    sample_source = (
        "okx_verified_execution_pair" if verified_execution_pair else "okx_position_history"
    )
    return {
        "source": sample_source,
        "id": int(_value(history, "id", 0) or 0),
        "lifecycle_key": lifecycle_key,
        "position_id": position_id,
        "decision_id": decision_id,
        "decision_lineage_source": decision_lineage_source,
        "position_ids": position_ids,
        "okx_pos_id": _text(_value(history, "pos_id")),
        "entry_order_ids": entry_order_ids,
        "close_order_ids": close_order_ids,
        "linked_order_ids": linked_order_ids,
        "okx_trade_ids": _order_trade_ids(linked_orders),
        "model_name": model_name,
        "execution_mode": execution_mode,
        "source_execution_mode": source_execution_mode,
        "symbol": _text(_value(history, "symbol")),
        "inst_id": _text(_value(history, "inst_id")),
        "side": side,
        "close_status": _text(_value(history, "close_status")).lower(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": contracts,
        "quantity_unit": "contracts",
        "fill_contracts": fill_contracts,
        "contract_ct_val": ct_val,
        "contract_ct_mult": ct_mult,
        "contract_lot_size": lot_size,
        "notional_usdt": notional,
        "authoritative_pnl_ratio_pct": official_ratio_pct,
        "realized_pnl": realized_pnl,
        "gross_pnl": gross_pnl,
        "fee": fee_signed,
        "fee_estimate": abs(fee_signed),
        "funding_fee": funding_fee,
        "liquidation_penalty": liquidation_penalty,
        "settlement_components_total": settlement_expected,
        "hold_minutes": hold_minutes,
        "leverage": _safe_float(_value(history, "leverage"), 1.0) or 1.0,
        "planned_stop_loss_price": stop_loss_price,
        "planned_take_profit_price": take_profit_price,
        "stop_loss_fill_confirmed": stop_loss_fill_confirmed,
        "stop_loss_slippage_pct": stop_loss_slippage_pct,
        "stop_loss_slippage_source": (
            protection_execution.get("stop_loss_slippage_source")
            if stop_loss_fill_confirmed
            else "not_authoritatively_confirmed"
        ),
        "protection_execution_supervision_ready": bool(protection_execution),
        "protection_lifecycle_complete": bool(
            protection_execution and protection_submission
        ),
        "protection_execution_gaps": protection_execution_gaps,
        "protection_algo_id": _text(protection_execution.get("algo_id")) or None,
        "protection_generated_order_id": (
            _text(protection_execution.get("generated_order_id")) or None
        ),
        "protection_actual_side": (
            _text(protection_execution.get("actual_side")) or None
        ),
        "exchange_configured_trigger_price": _safe_float(
            protection_execution.get("configured_trigger_price"),
            None,
        ),
        "actual_trigger_market_price": _safe_float(
            protection_execution.get("actual_trigger_market_price"),
            None,
        ),
        "actual_trigger_market_price_available": (
            protection_execution.get("actual_trigger_market_price_available") is True
        ),
        "protection_exchange_confirmed_at": (
            protection_submission.get("exchange_confirmed_at")
            or _iso_from_ms(protection_execution.get("exchange_confirmed_at_ms"))
        ),
        "protection_triggered_at": _iso_from_ms(
            protection_execution.get("triggered_at_ms")
        ),
        "protection_fill_started_at": _iso_from_ms(
            protection_execution.get("fill_started_at_ms")
        ),
        "protection_fill_completed_at": _iso_from_ms(
            protection_execution.get("fill_completed_at_ms")
        ),
        "trigger_to_first_fill_ms": _safe_float(
            protection_execution.get("trigger_to_first_fill_ms"),
            None,
        ),
        "protection_fill_mark_price": _safe_float(
            protection_execution.get("fill_mark_price"),
            None,
        ),
        "protection_fill_index_price": _safe_float(
            protection_execution.get("fill_index_price"),
            None,
        ),
        "protection_fill_path_min_price": _safe_float(
            protection_execution.get("fill_path_min_price"),
            None,
        ),
        "protection_fill_path_max_price": _safe_float(
            protection_execution.get("fill_path_max_price"),
            None,
        ),
        "protection_fill_mark_slippage_pct": _safe_float(
            protection_execution.get("fill_mark_slippage_pct"),
            None,
        ),
        "execution_risk_budget_usdt": budget_facts["risk_budget_usdt"],
        "execution_planned_stressed_loss_usdt": budget_facts[
            "planned_stressed_loss_usdt"
        ],
        "execution_actual_loss_usdt": budget_facts["actual_loss_usdt"],
        "execution_actual_over_budget_loss_usdt": budget_facts[
            "actual_over_budget_loss_usdt"
        ],
        "close_order_types": sorted(
            {
                _text(_value(order, "order_type")).lower()
                for order in close_orders
                if _text(_value(order, "order_type"))
            }
        ),
        "raw_llm_response": raw_llm_response,
        "outcome": "profit" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat",
        "pnl_source": (
            _text(raw.get("_bb_pnl_source"))
            if verified_execution_pair
            else "okx_position_history_realized_pnl"
        ),
        "settlement_source": (
            "okx_verified_execution_pair_settlement"
            if verified_execution_pair
            else "okx_position_history_realized_pnl"
        ),
        "funding_fee_source": (
            _text(raw.get("_bb_funding_fee_source"))
            if verified_execution_pair
            else "okx_positions_history.fundingFee"
        ),
        "fee_source": (
            _text(raw.get("_bb_fee_source"))
            if verified_execution_pair
            else "okx_positions_history.fee"
        ),
        "trade_fact_trusted": not gaps,
        "trade_fact_trust_reason": gaps[0] if gaps else "",
        "strategy_lineage_complete": not lineage_gaps,
        "strategy_lineage_gaps": lineage_gaps,
        "training_evidence_gaps": list(dict.fromkeys([*gaps, *lineage_gaps])),
        "label_timestamp": closed_at.isoformat() if closed_at else None,
    }
