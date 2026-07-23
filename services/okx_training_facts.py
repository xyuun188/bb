"""Build model-training samples from authoritative OKX SWAP lifecycles."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from services.paper_exploration import paper_exploration_contract_reasons
from services.paper_training import paper_training_contract_reasons
from services.profit_training_contract import validate_profit_training_sample


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


def _account_contract_size_from_orders(orders: Iterable[Any]) -> dict[str, Any]:
    values: list[tuple[float, str]] = []
    quarantined_reasons: list[str] = []
    for order in orders:
        raw = _dict(_value(order, "okx_raw_fills", {}))
        source = _text(raw.get("contract_size_source"))
        if source == "okx_account_position_evidence_quarantined":
            quarantined_reasons.append(
                _text(raw.get("contract_size_evidence_gap"))
                or "account_contract_size_evidence_quarantined"
            )
            continue
        if (
            raw.get("contract_size_verified") is True
            and source.startswith("okx_account_position_")
        ):
            size = _safe_float(raw.get("contract_size") or raw.get("contractSize"), None)
            if size is not None and size > 0:
                values.append((size, source))
    if quarantined_reasons:
        return {
            "contract_size": None,
            "source": "okx_account_position_evidence_quarantined",
            "conflict": True,
            "reason": quarantined_reasons[0],
            "values": [value for value, _source in values],
        }
    if not values:
        return {
            "contract_size": None,
            "source": "",
            "conflict": False,
            "reason": "",
            "values": [],
        }
    reference = values[0][0]
    conflict = any(
        not math.isclose(value, reference, rel_tol=0.02, abs_tol=1e-12)
        for value, _source in values[1:]
    )
    return {
        "contract_size": None if conflict else sum(value for value, _source in values) / len(values),
        "source": values[0][1] if not conflict else "okx_account_position_evidence_conflict",
        "conflict": conflict,
        "reason": "linked_order_account_contract_sizes_disagree" if conflict else "",
        "values": [value for value, _source in values],
    }


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


def _decision_authority(
    *,
    raw_llm_response: dict[str, Any],
    execution_mode: str,
    valid_paper_exploration: bool,
    valid_paper_training: bool,
    strategy_training_role: str,
 ) -> str:
    gate = _dict(raw_llm_response.get("production_trade_gate"))
    authority = _text(gate.get("decision_authority")).lower()
    if authority in {"rules", "model", "manual", "system"}:
        return authority
    if execution_mode == "paper" and (valid_paper_exploration or valid_paper_training):
        return "system"
    if strategy_training_role != "entry_strategy":
        return "system"
    return "model"


def _model_shadow_prediction(
    raw_llm_response: dict[str, Any],
    *,
    decision_authority: str,
) -> dict[str, Any]:
    if decision_authority != "rules":
        return {}
    gate = _dict(raw_llm_response.get("production_trade_gate"))
    signal = _dict(raw_llm_response.get("live_rules_canary_signal"))
    shadow = _dict(raw_llm_response.get("model_shadow_decision"))
    if (
        gate.get("mode") != "live_rules_canary"
        or gate.get("decision_authority") != "rules"
        or gate.get("model_can_influence") is not False
        or signal.get("production_eligible") is not True
        or signal.get("decision_authority") != "rules"
        or signal.get("model_can_influence") is not False
        or signal.get("action") not in {"long", "short"}
        or shadow.get("observation_only") is not True
        or shadow.get("can_authorize_entry") is not False
        or shadow.get("can_change_size_or_leverage") is not False
    ):
        return {}
    action = _text(shadow.get("action")).lower()
    if action in {"buy", "open_long"}:
        action = "long"
    elif action in {"sell", "open_short"}:
        action = "short"
    if action not in {"long", "short"}:
        return {}
    return {
        "action": action,
        "confidence": _safe_float(shadow.get("confidence"), None),
        "source": "live_rules_canary_model_shadow_decision",
        "observation_only": True,
        "can_authorize_entry": False,
        "rules_execution_action": signal.get("action"),
        "signal_version": signal.get("version"),
    }


def _directional_price_return_pct(
    *,
    side: str,
    entry_price: float,
    exit_price: float,
) -> float | None:
    if entry_price <= 0 or exit_price <= 0 or side not in {"long", "short"}:
        return None
    raw_return = (exit_price - entry_price) / entry_price * 100.0
    return raw_return if side == "long" else -raw_return


def _authoritative_notional_facts(
    *,
    inst_id: str,
    side: str,
    entry_price: float,
    exit_price: float,
    gross_pnl: float,
    contract_notional_usdt: float | None,
) -> dict[str, Any]:
    """Prefer the official gross-PnL price path over stale contract multipliers."""

    price_return_pct = _directional_price_return_pct(
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
    )
    pnl_implied_notional = None
    if (
        inst_id.upper().endswith("-USDT-SWAP")
        and price_return_pct is not None
        and abs(price_return_pct) > 1e-9
        and abs(gross_pnl) > 1e-9
    ):
        pnl_implied_notional = abs(gross_pnl) / (abs(price_return_pct) / 100.0)
    notional = (
        pnl_implied_notional
        if pnl_implied_notional is not None and pnl_implied_notional > 0
        else contract_notional_usdt
    )
    relative_error = None
    if (
        pnl_implied_notional is not None
        and pnl_implied_notional > 0
        and contract_notional_usdt is not None
        and contract_notional_usdt > 0
    ):
        relative_error = abs(contract_notional_usdt - pnl_implied_notional) / max(
            pnl_implied_notional,
            1e-9,
        )
    gross_return_pct = (
        gross_pnl / notional * 100.0
        if notional is not None and notional > 0
        else None
    )
    return_consistent = bool(
        price_return_pct is not None
        and gross_return_pct is not None
        and math.isclose(
            gross_return_pct,
            price_return_pct,
            rel_tol=0.01,
            abs_tol=0.05,
        )
    )
    return {
        "notional_usdt": notional,
        "notional_source": (
            "okx_gross_pnl_and_average_price_path"
            if pnl_implied_notional is not None and pnl_implied_notional > 0
            else "okx_contract_spec_and_entry_fills"
        ),
        "contract_spec_notional_usdt": contract_notional_usdt,
        "pnl_implied_notional_usdt": pnl_implied_notional,
        "contract_notional_relative_error": relative_error,
        "contract_notional_corrected": bool(
            relative_error is not None and relative_error > 0.05
        ),
        "gross_price_return_pct": price_return_pct,
        "gross_return_on_notional_pct": gross_return_pct,
        "gross_return_price_consistent": return_consistent,
    }


def build_okx_history_training_sample(
    history: Any,
    *,
    positions_by_id: dict[int, Any] | None = None,
    orders_by_exchange_id: dict[str, Any] | None = None,
    decision_raw_by_position_id: dict[int, dict[str, Any]] | None = None,
    decision_raw_by_order_id: dict[str, dict[str, Any]] | None = None,
    decision_execution_by_order_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert one mirrored OKX positions-history lifecycle into one sample."""

    positions_by_id = positions_by_id or {}
    orders_by_exchange_id = orders_by_exchange_id or {}
    decision_raw_by_position_id = decision_raw_by_position_id or {}
    decision_raw_by_order_id = decision_raw_by_order_id or {}
    decision_execution_by_order_id = decision_execution_by_order_id or {}
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
    side = _text(_value(history, "side")).lower()
    open_contracts = _safe_float(_value(history, "open_max_pos"), 0.0) or 0.0
    fill_contracts = _entry_fill_contracts(entry_orders)
    contracts = fill_contracts or open_contracts
    spec = _raw_contract_spec(raw)
    public_or_stored_ct_val = _safe_float(spec.get("ctVal"), None)
    ct_mult = _safe_float(spec.get("ctMult"), None)
    lot_size = _safe_float(spec.get("lotSz"), None)

    realized_pnl = _safe_float(_value(history, "realized_pnl"), 0.0) or 0.0
    gross_pnl = _safe_float(_value(history, "pnl"), 0.0) or 0.0
    fee_signed = _safe_float(_value(history, "fee"), 0.0) or 0.0
    funding_fee = _safe_float(_value(history, "funding_fee"), 0.0) or 0.0
    liquidation_penalty = _safe_float(
        raw.get("liqPenalty") or raw.get("liquidationPenalty"), 0.0
    ) or 0.0
    source_execution_mode = _text(_value(history, "mode")).lower()
    execution_mode = _canonical_execution_mode(source_execution_mode)
    account_contract_size = _account_contract_size_from_orders(entry_orders)
    history_contract_source = _text(raw.get("_bb_contract_spec_source"))
    history_account_ct_val = (
        public_or_stored_ct_val
        if history_contract_source.startswith("okx_account_position_")
        else None
    )
    linked_account_ct_val = _safe_float(
        account_contract_size.get("contract_size"),
        None,
    )
    account_values = [
        value
        for value in (history_account_ct_val, linked_account_ct_val)
        if value is not None and value > 0
    ]
    account_contract_conflict = bool(account_contract_size.get("conflict"))
    if len(account_values) > 1 and not math.isclose(
        account_values[0],
        account_values[1],
        rel_tol=0.02,
        abs_tol=1e-12,
    ):
        account_contract_conflict = True
    verified_account_ct_val = (
        sum(account_values) / len(account_values)
        if account_values and not account_contract_conflict
        else None
    )
    pnl_implied_ct_val = None
    price_delta = abs(exit_price - entry_price)
    if (
        contracts > 0
        and ct_mult is not None
        and ct_mult > 0
        and price_delta > 1e-12
        and abs(gross_pnl) > 1e-12
        and _text(_value(history, "inst_id")).upper().endswith("-USDT-SWAP")
    ):
        pnl_implied_ct_val = abs(gross_pnl) / contracts / price_delta / ct_mult
    if (
        verified_account_ct_val is not None
        and pnl_implied_ct_val is not None
        and not math.isclose(
            verified_account_ct_val,
            pnl_implied_ct_val,
            rel_tol=0.02,
            abs_tol=1e-12,
        )
    ):
        account_contract_conflict = True

    ct_val = public_or_stored_ct_val
    contract_ct_val_source = "okx_contract_spec"
    contract_ct_val_corrected = False
    if verified_account_ct_val is not None:
        ct_val = verified_account_ct_val
        contract_ct_val_source = (
            history_contract_source
            or _text(account_contract_size.get("source"))
            or "okx_account_position_verified"
        )
        contract_ct_val_corrected = bool(
            public_or_stored_ct_val is not None
            and not math.isclose(
                public_or_stored_ct_val,
                verified_account_ct_val,
                rel_tol=0.02,
                abs_tol=1e-12,
            )
        )
    elif (
        execution_mode == "paper"
        and pnl_implied_ct_val is not None
        and (
            public_or_stored_ct_val is None
            or not math.isclose(
                public_or_stored_ct_val,
                pnl_implied_ct_val,
                rel_tol=0.05,
                abs_tol=1e-12,
            )
        )
    ):
        ct_val = pnl_implied_ct_val
        contract_ct_val_source = "okx_gross_pnl_contract_size_crosscheck"
        contract_ct_val_corrected = True
    public_contract_notional = (
        abs(contracts * public_or_stored_ct_val * ct_mult * entry_price)
        if contracts > 0
        and public_or_stored_ct_val
        and ct_mult
        and entry_price > 0
        else None
    )
    effective_contract_notional = (
        abs(contracts * ct_val * ct_mult * entry_price)
        if contracts > 0 and ct_val and ct_mult and entry_price > 0
        else None
    )
    contract_notional = (
        effective_contract_notional
        if verified_account_ct_val is not None
        else public_contract_notional
    )
    notional_facts = _authoritative_notional_facts(
        inst_id=_text(_value(history, "inst_id")),
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_pnl=gross_pnl,
        contract_notional_usdt=contract_notional,
    )
    notional = _safe_float(notional_facts.get("notional_usdt"), None)
    settlement_expected = gross_pnl + fee_signed + funding_fee + liquidation_penalty
    settlement_tolerance = max(1e-6, abs(realized_pnl) * 1e-5)
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
    if account_contract_conflict:
        gaps.append("account_contract_size_evidence_conflict")
    if notional is None or notional <= 0:
        gaps.append("missing_authoritative_notional")
    if notional_facts.get("gross_return_price_consistent") is not True:
        gaps.append("gross_return_price_path_mismatch")
    if abs(realized_pnl - settlement_expected) > settlement_tolerance:
        gaps.append("settlement_algebra_mismatch")
    gaps = list(dict.fromkeys(gaps))

    position_id = position_ids[0] if position_ids else 0
    entry_decision_ids = sorted(
        {
            int(_value(orders_by_exchange_id.get(order_id), "decision_id", 0) or 0)
            for order_id in entry_order_ids
            if int(
                _value(orders_by_exchange_id.get(order_id), "decision_id", 0) or 0
            )
            > 0
        }
    )
    raw_llm_response: dict[str, Any] = {}
    decision_id = entry_decision_ids[0] if len(entry_decision_ids) == 1 else 0
    decision_lineage_source = (
        "exact_entry_order_decision_id"
        if len(entry_decision_ids) == 1
        else "multiple_entry_decisions"
        if len(entry_decision_ids) > 1
        else "missing"
    )
    if len(entry_decision_ids) == 1:
        for order_id in entry_order_ids:
            order = orders_by_exchange_id.get(order_id)
            if int(_value(order, "decision_id", 0) or 0) != decision_id:
                continue
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
    exact_execution = next(
        (
            decision_execution_by_order_id[order_id]
            for order_id in entry_order_ids
            if order_id in decision_execution_by_order_id
        ),
        {},
    )
    if entry_price > 0 and side in {"long", "short"}:
        stop_loss_pct = _safe_float(exact_execution.get("stop_loss_pct"), None)
        take_profit_pct = _safe_float(exact_execution.get("take_profit_pct"), None)
        if (stop_loss_price is None or stop_loss_price <= 0) and stop_loss_pct and stop_loss_pct > 0:
            stop_loss_price = (
                entry_price * (1 - stop_loss_pct)
                if side == "long"
                else entry_price * (1 + stop_loss_pct)
            )
        if (
            take_profit_price is None or take_profit_price <= 0
        ) and take_profit_pct and take_profit_pct > 0:
            take_profit_price = (
                entry_price * (1 + take_profit_pct)
                if side == "long"
                else entry_price * (1 - take_profit_pct)
            )
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
    if len(entry_decision_ids) > 1:
        lineage_gaps.append("multiple_entry_decision_lineage")
    if decision_lineage_source != "exact_entry_order_decision_payload":
        lineage_gaps.append("missing_exact_entry_order_decision_payload")
    if local_position is None:
        lineage_gaps.append("missing_local_position_strategy_lineage")
    if stop_loss_price is None or stop_loss_price <= 0:
        lineage_gaps.append("missing_planned_stop_loss_lineage")
    if take_profit_price is None or take_profit_price <= 0:
        lineage_gaps.append("missing_planned_take_profit_lineage")
    paper_canary = _dict(raw_llm_response.get("paper_bootstrap_canary"))
    obsolete_sampling_entry = bool(
        paper_canary
        and (
            _text(paper_canary.get("trade_kind")) != "normal_strategy_trade"
            or paper_canary.get("continuous_training_after_settlement") is not True
        )
    )
    if obsolete_sampling_entry:
        lineage_gaps.append("obsolete_sampling_entry_not_strategy_trainable")
    paper_exploration = _dict(raw_llm_response.get("paper_exploration"))
    paper_exploration_gaps = (
        paper_exploration_contract_reasons(paper_exploration)
        if paper_exploration
        else []
    )
    paper_training = _dict(raw_llm_response.get("paper_training"))
    paper_training_gaps = (
        paper_training_contract_reasons(paper_training)
        if paper_training
        else []
    )
    if paper_training and execution_mode != "paper":
        paper_training_gaps.append("paper_training_non_paper_execution_mode")
    if paper_training and (paper_exploration or paper_canary):
        paper_training_gaps.append("paper_training_conflicting_entry_contract")
    paper_training_gaps = list(dict.fromkeys(paper_training_gaps))
    valid_paper_exploration = bool(
        paper_exploration and not paper_exploration_gaps
    )
    valid_paper_training = bool(paper_training and not paper_training_gaps)
    if paper_exploration_gaps:
        lineage_gaps.append("invalid_paper_exploration_contract")
    if paper_training_gaps:
        lineage_gaps.append("invalid_paper_training_contract")
    lineage_gaps = list(dict.fromkeys(lineage_gaps))
    model_name = _text(_value(local_position, "model_name")) if local_position else ""
    official_ratio_pct = _official_ratio_pct(raw, _value(history, "pnl_ratio"))
    lifecycle_key = _text(_value(history, "row_identity"))
    history_source = _text(_value(history, "source"))
    verified_execution_pair = history_source == "okx_verified_execution_pair_settlement"
    sample_source = (
        "okx_verified_execution_pair" if verified_execution_pair else "okx_position_history"
    )
    strategy_training_role = (
        "aggregate_position_research_only"
        if len(entry_decision_ids) > 1
        else "obsolete_sampling_research_only"
        if obsolete_sampling_entry
        else "invalid_exploration_research_only"
        if paper_exploration_gaps
        else "invalid_paper_training_research_only"
        if paper_training_gaps
        else "entry_strategy"
    )
    sample = {
        "source": sample_source,
        "id": int(_value(history, "id", 0) or 0),
        "lifecycle_key": lifecycle_key,
        "position_id": position_id,
        "decision_id": decision_id,
        "entry_decision_ids": entry_decision_ids,
        "entry_decision_count": len(entry_decision_ids),
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
        "contract_ct_val_source": contract_ct_val_source,
        "contract_ct_val_corrected": contract_ct_val_corrected,
        "public_or_stored_contract_ct_val": public_or_stored_ct_val,
        "pnl_implied_contract_ct_val": pnl_implied_ct_val,
        "account_contract_size_evidence": account_contract_size,
        "contract_ct_mult": ct_mult,
        "contract_lot_size": lot_size,
        "notional_usdt": notional,
        **notional_facts,
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
        "strategy_entry_kind": (
            "loss_tolerant_paper_training"
            if valid_paper_training
            else "bounded_risk_paper_exploration"
            if valid_paper_exploration
            else "normal_strategy_trade"
        ),
        "strategy_selection_reason": (
            _text(paper_training.get("selection_reason"))
            if valid_paper_training
            else _text(paper_exploration.get("selection_reason"))
            if valid_paper_exploration
            else _text(
                _dict(raw_llm_response.get("entry_permission_policy")).get("source")
            )
            or "governed_fee_after_return_strategy"
        ),
        "paper_exploration_evidence": (
            {
                "version": paper_exploration.get("version"),
                "selected_side": paper_exploration.get("selected_side"),
                "expected_net_return_pct": paper_exploration.get(
                    "expected_net_return_pct"
                ),
                "return_lcb_pct": paper_exploration.get("return_lcb_pct"),
                "information_value_score": paper_exploration.get(
                    "information_value_score"
                ),
                "single_trade_risk_fraction_cap": paper_exploration.get(
                    "single_trade_risk_fraction_cap"
                ),
                "portfolio_risk_fraction_cap": paper_exploration.get(
                    "portfolio_risk_fraction_cap"
                ),
                "sample_target": paper_exploration.get("sample_target"),
                "daily_sample_quota": paper_exploration.get("daily_sample_quota"),
            }
            if valid_paper_exploration
            else {}
        ),
        "paper_training_evidence": (
            {
                "version": paper_training.get("version"),
                "trade_kind": paper_training.get("trade_kind"),
                "selected_side": paper_training.get("selected_side"),
                "signal_source": paper_training.get("signal_source"),
                "expected_net_return_pct": paper_training.get(
                    "expected_net_return_pct"
                ),
                "return_lcb_pct": paper_training.get("return_lcb_pct"),
                "loss_tolerant_for_training": paper_training.get(
                    "loss_tolerant_for_training"
                ),
                "continuous_training_after_settlement": paper_training.get(
                    "continuous_training_after_settlement"
                ),
                "sample_target": paper_training.get("sample_target"),
                "daily_sample_quota": paper_training.get("daily_sample_quota"),
            }
            if valid_paper_training
            else {}
        ),
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
        "strategy_entry_supervision_eligible": bool(
            len(entry_decision_ids) <= 1
            and not obsolete_sampling_entry
            and not paper_exploration_gaps
            and not paper_training_gaps
        ),
        "strategy_training_role": strategy_training_role,
        "training_evidence_gaps": list(dict.fromkeys([*gaps, *lineage_gaps])),
        "label_timestamp": closed_at.isoformat() if closed_at else None,
    }
    decision_authority = _decision_authority(
        raw_llm_response=raw_llm_response,
        execution_mode=execution_mode,
        valid_paper_exploration=valid_paper_exploration,
        valid_paper_training=valid_paper_training,
        strategy_training_role=strategy_training_role,
    )
    sample["decision_authority"] = decision_authority
    model_shadow_prediction = _model_shadow_prediction(
        raw_llm_response,
        decision_authority=decision_authority,
    )
    if model_shadow_prediction:
        sample["model_shadow_prediction"] = model_shadow_prediction
    if notional is not None and notional > 0:
        sample["net_return_after_all_cost_pct"] = realized_pnl / notional * 100.0
    sample["profit_training_contract"] = validate_profit_training_sample(sample).to_dict()
    return sample
