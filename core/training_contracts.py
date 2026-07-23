"""Immutable identities for versioned training labels."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

SHADOW_LABEL_VERSION = "2026-07-14.native-shadow-label.v1"
SHADOW_FEE_AFTER_LABEL_VERSION = "2026-07-22.shadow-fee-after-return.v1"
AUTHORITATIVE_TRADE_OUTCOME_VERSION = "2026-07-19.authoritative-trade-outcome.v2"
AUTHORITATIVE_TRADE_LABEL_VERSION = "2026-07-23.net-return-after-all-cost.v1"
AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY = "okx_settlement_and_execution"
AUTHORITATIVE_EXPERT_MEMORY_SOURCE = "authoritative_trade_outcome"
AUTHORITATIVE_TRADE_OUTCOME_SOURCES = frozenset(
    {
        "okx_position_history",
        "okx_verified_execution_pair",
    }
)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(_text(value).replace("Z", "+00:00"))
        except ValueError:
            return ""
    else:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def is_authoritative_expert_memory_extra(value: Any) -> bool:
    """Return whether an expert-memory payload is a complete OKX outcome fact."""

    extra = _dict(value)
    return bool(
        _text(extra.get("source")) == AUTHORITATIVE_EXPERT_MEMORY_SOURCE
        and _text(extra.get("authority_level")) == AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY
        and _text(extra.get("outcome_version")) == AUTHORITATIVE_TRADE_OUTCOME_VERSION
        and extra.get("production_evidence_eligible") is True
        and extra.get("cost_complete") is True
        and _text(extra.get("outcome_id"))
        and _text(extra.get("outcome_fingerprint"))
    )


def build_shadow_label_contract(
    *,
    shadow_backtest_id: int,
    decision_id: int | None,
    horizon_minutes: int,
    long_return_pct: float,
    short_return_pct: float,
    best_action: str,
    market_fact_contract: Mapping[str, Any] | None,
    cost_facts: Mapping[str, Any] | None,
    label_timestamp: Any,
    version: str = SHADOW_LABEL_VERSION,
) -> dict[str, Any]:
    market_contract = _dict(market_fact_contract)
    market_provenance = _dict(market_contract.get("provenance"))
    costs = _dict(cost_facts)
    long_net_return = _float(costs.get("long_net_return_after_all_cost_pct"))
    short_net_return = _float(costs.get("short_net_return_after_all_cost_pct"))
    fee_after_complete = bool(
        costs.get("cost_complete") is True
        and long_net_return is not None
        and short_net_return is not None
    )
    labels = {
        "long_return_pct": float(long_return_pct),
        "short_return_pct": float(short_return_pct),
        "best_action": _text(best_action).lower(),
        "label_timestamp": _iso(label_timestamp),
    }
    if fee_after_complete:
        labels.update(
            {
                "fee_after_label_version": SHADOW_FEE_AFTER_LABEL_VERSION,
                "fee_after_complete": True,
                "long_gross_return_pct": float(long_return_pct),
                "short_gross_return_pct": float(short_return_pct),
                "long_net_return_after_all_cost_pct": long_net_return,
                "short_net_return_after_all_cost_pct": short_net_return,
            }
        )
    payload = {
        "version": version,
        "immutable": True,
        "identity": {
            "shadow_backtest_id": int(shadow_backtest_id or 0),
            "decision_id": int(decision_id or 0),
            "horizon_minutes": int(horizon_minutes or 0),
        },
        "labels": labels,
        "market_fact_lineage": {
            "contract_version": market_contract.get("version"),
            "data_fingerprint": market_provenance.get("data_fingerprint")
            or market_contract.get("data_fingerprint"),
            "entry_fact_id": market_contract.get("entry_fact_id"),
            "result_fact_id": market_contract.get("result_fact_id"),
            "path_fingerprint": market_contract.get("path_fingerprint"),
        },
        "cost_facts": costs,
    }
    contract = {
        **payload,
        "provenance": {
            "source": "completed_shadow_native_market_fact_and_fee_after_cost",
            "observation_window": {
                "label_timestamp": _iso(label_timestamp),
                "horizon_minutes": int(horizon_minutes or 0),
            },
            "sample_count": 1,
            "effective_sample_size": 1.0,
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": version,
            "fallback_reason": "",
        },
    }
    contract["label_fingerprint"] = _fingerprint(payload)
    return contract


def compact_shadow_label_contract(
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    value = _dict(contract)
    identity = _dict(value.get("identity"))
    labels = _dict(value.get("labels"))
    lineage = _dict(value.get("market_fact_lineage"))
    provenance = _dict(value.get("provenance"))
    compact = {
        "version": value.get("version"),
        "immutable": value.get("immutable"),
        "shadow_backtest_id": identity.get("shadow_backtest_id"),
        "decision_id": identity.get("decision_id"),
        "horizon_minutes": identity.get("horizon_minutes"),
        "long_return_pct": labels.get("long_return_pct"),
        "short_return_pct": labels.get("short_return_pct"),
        "best_action": labels.get("best_action"),
        "label_timestamp": labels.get("label_timestamp"),
        "market_fact_contract_version": lineage.get("contract_version"),
        "market_fact_data_fingerprint": lineage.get("data_fingerprint"),
        "entry_fact_id": lineage.get("entry_fact_id"),
        "result_fact_id": lineage.get("result_fact_id"),
        "path_fingerprint": lineage.get("path_fingerprint"),
        "cost_facts_fingerprint": _fingerprint(_dict(value.get("cost_facts"))),
        "label_fingerprint": value.get("label_fingerprint"),
        "source": provenance.get("source"),
        "generated_at": provenance.get("generated_at"),
        "strategy_version": provenance.get("strategy_version"),
        "fallback_reason": provenance.get("fallback_reason"),
    }
    if labels.get("fee_after_complete") is True:
        compact.update(
            {
                "fee_after_label_version": labels.get("fee_after_label_version"),
                "fee_after_complete": True,
                "long_gross_return_pct": labels.get("long_gross_return_pct"),
                "short_gross_return_pct": labels.get("short_gross_return_pct"),
                "long_net_return_after_all_cost_pct": labels.get(
                    "long_net_return_after_all_cost_pct"
                ),
                "short_net_return_after_all_cost_pct": labels.get(
                    "short_net_return_after_all_cost_pct"
                ),
            }
        )
    compact["compact_fingerprint"] = _fingerprint(compact)
    return compact


def shadow_fee_after_label_values(
    contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return explicit fee-after labels without treating gross returns as net."""

    value = _dict(contract)
    labels = _dict(value.get("labels")) or value
    long_net = _float(labels.get("long_net_return_after_all_cost_pct"))
    short_net = _float(labels.get("short_net_return_after_all_cost_pct"))
    complete = bool(
        labels.get("fee_after_complete") is True
        and labels.get("fee_after_label_version") == SHADOW_FEE_AFTER_LABEL_VERSION
        and long_net is not None
        and short_net is not None
    )
    return {
        "version": labels.get("fee_after_label_version"),
        "complete": complete,
        "long_net_return_after_all_cost_pct": long_net if complete else None,
        "short_net_return_after_all_cost_pct": short_net if complete else None,
    }


def shadow_label_contract_reasons(
    contract: Mapping[str, Any] | None,
    *,
    decision_id: Any = None,
    horizon_minutes: Any = None,
    label_version: Any = None,
) -> list[str]:
    value = _dict(contract)
    if not value:
        return ["shadow_label_contract_missing"]
    reasons: list[str] = []
    if value.get("version") != SHADOW_LABEL_VERSION:
        reasons.append("shadow_label_version_missing_or_stale")
    expected_version = _text(label_version)
    if expected_version and value.get("version") != expected_version:
        reasons.append("shadow_label_row_version_mismatch")
    if value.get("immutable") is not True:
        reasons.append("shadow_label_not_immutable")
    compact = bool(value.get("compact_fingerprint"))
    identity = _dict(value.get("identity")) or value
    labels = _dict(value.get("labels")) or value
    contract_decision_id = int(_float(identity.get("decision_id")) or 0)
    contract_horizon = int(_float(identity.get("horizon_minutes")) or 0)
    if int(_float(identity.get("shadow_backtest_id")) or 0) <= 0:
        reasons.append("shadow_backtest_id_missing")
    if contract_decision_id <= 0:
        reasons.append("shadow_decision_id_missing")
    if contract_horizon <= 0:
        reasons.append("shadow_horizon_missing")
    expected_decision_id = int(_float(decision_id) or 0)
    expected_horizon = int(_float(horizon_minutes) or 0)
    if expected_decision_id > 0 and contract_decision_id != expected_decision_id:
        reasons.append("shadow_label_decision_id_mismatch")
    if expected_horizon > 0 and contract_horizon != expected_horizon:
        reasons.append("shadow_label_horizon_mismatch")
    if _float(labels.get("long_return_pct")) is None:
        reasons.append("shadow_long_return_label_missing")
    if _float(labels.get("short_return_pct")) is None:
        reasons.append("shadow_short_return_label_missing")
    if _text(labels.get("best_action")).lower() not in {"long", "short", "hold"}:
        reasons.append("shadow_best_action_label_invalid")
    if not _iso(labels.get("label_timestamp")):
        reasons.append("shadow_label_timestamp_missing")

    fee_after_declared = any(
        key in labels
        for key in (
            "fee_after_label_version",
            "fee_after_complete",
            "long_net_return_after_all_cost_pct",
            "short_net_return_after_all_cost_pct",
        )
    )
    if fee_after_declared:
        fee_after = shadow_fee_after_label_values(value)
        if fee_after.get("complete") is not True:
            reasons.append("shadow_fee_after_label_incomplete")
        else:
            long_net = float(fee_after["long_net_return_after_all_cost_pct"])
            short_net = float(fee_after["short_net_return_after_all_cost_pct"])
            expected_best = (
                "long"
                if long_net > 0.0 and long_net >= short_net
                else "short"
                if short_net > 0.0 and short_net > long_net
                else "hold"
            )
            if _text(labels.get("best_action")).lower() != expected_best:
                reasons.append("shadow_best_action_not_fee_after_optimal")

    if compact:
        for key in (
            "market_fact_data_fingerprint",
            "cost_facts_fingerprint",
            "label_fingerprint",
            "source",
            "generated_at",
            "strategy_version",
        ):
            if not _text(value.get(key)):
                reasons.append(f"shadow_label_lineage_missing:{key}")
        fingerprint_payload = dict(value)
        fingerprint = fingerprint_payload.pop("compact_fingerprint", None)
        if fingerprint != _fingerprint(fingerprint_payload):
            reasons.append("shadow_label_compact_fingerprint_mismatch")
    else:
        lineage = _dict(value.get("market_fact_lineage"))
        if not _text(lineage.get("data_fingerprint")):
            reasons.append("shadow_label_lineage_missing:market_fact_data_fingerprint")
        if not _dict(value.get("cost_facts")):
            reasons.append("shadow_label_lineage_missing:cost_facts")
        payload = {
            key: value.get(key)
            for key in (
                "version",
                "immutable",
                "identity",
                "labels",
                "market_fact_lineage",
                "cost_facts",
            )
        }
        if value.get("label_fingerprint") != _fingerprint(payload):
            reasons.append("shadow_label_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))
