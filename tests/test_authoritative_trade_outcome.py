from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from services.authoritative_trade_outcome import (
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
    build_authoritative_trade_outcome,
)
from services.training_data_quality import annotate_training_payload


def _sample(**overrides):
    sample = {
        "source": "okx_position_history",
        "id": 88,
        "lifecycle_key": "paper|ICP-USDT-SWAP|pos-icp|short|1",
        "position_id": 4879,
        "position_ids": [4879],
        "decision_id": 79318,
        "decision_lineage_source": "exact_entry_order_decision_payload",
        "okx_pos_id": "pos-icp",
        "entry_order_ids": ["entry-icp"],
        "close_order_ids": ["close-icp"],
        "linked_order_ids": ["entry-icp", "close-icp"],
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "ICP/USDT",
        "side": "short",
        "close_status": "full",
        "entry_price": 2.167,
        "exit_price": 2.285,
        "quantity": 60194.1,
        "quantity_unit": "contracts",
        "notional_usdt": 1304.5,
        "authoritative_pnl_ratio_pct": -5.51319027,
        "realized_pnl": -71.83582759,
        "gross_pnl": -70.0,
        "fee": -1.2,
        "fee_estimate": 1.2,
        "funding_fee": -0.63582759,
        "liquidation_penalty": 0.0,
        "hold_minutes": 292.4,
        "planned_stop_loss_price": 2.21,
        "stop_loss_fill_confirmed": True,
        "stop_loss_slippage_pct": 3.393665,
        "stop_loss_slippage_source": "okx_configured_stop_trigger_to_fills_vwap",
        "trigger_to_first_fill_ms": 1060.0,
        "execution_actual_over_budget_loss_usdt": 14.0,
        "outcome": "loss",
        "pnl_source": "okx_position_history_realized_pnl",
        "settlement_source": "okx_position_history_realized_pnl",
        "funding_fee_source": "okx_positions_history.fundingFee",
        "fee_source": "okx_positions_history.fee",
        "trade_fact_trusted": True,
        "trade_fact_trust_reason": "",
        "training_evidence_gaps": [],
        "label_timestamp": "2026-07-14T04:33:27.688000+00:00",
    }
    sample.update(overrides)
    return sample


def _reflection():
    return SimpleNamespace(
        id=5432,
        position_id=4879,
        source="authoritative_trade_outcome",
        outcome="loss",
        mistake_summary="authoritative loss",
        improvement_summary="recalibrate uncertainty and tail loss",
        created_at=datetime(2026, 7, 14, 4, 34, tzinfo=UTC),
    )


def test_real_outcome_has_stable_identity_and_shadow_is_counterfactual_only() -> None:
    shadow = SimpleNamespace(
        id=50798,
        decision_id=79318,
        status="completed",
        horizon_minutes=10,
        long_return_pct=0.138,
        short_return_pct=-0.138,
        best_action="long",
    )

    first = build_authoritative_trade_outcome(
        _sample(), reflection=_reflection(), shadow_rows=[shadow]
    )
    second = build_authoritative_trade_outcome(
        _sample(), reflection=_reflection(), shadow_rows=[shadow]
    )

    assert first["outcome_version"] == AUTHORITATIVE_TRADE_OUTCOME_VERSION
    assert first["outcome_id"] == second["outcome_id"]
    assert first["outcome_fingerprint"] == second["outcome_fingerprint"]
    rebuilt = build_authoritative_trade_outcome(first, reflection=_reflection())
    assert rebuilt["outcome_fingerprint"] == first["outcome_fingerprint"]
    assert first["outcome_complete"] is True
    assert first["counterfactual_production_weight"] == 0.0
    assert first["counterfactual_evidence"][0]["may_override_actual_outcome"] is False


def test_outcome_attribution_preserves_unknowns_and_measures_tail_execution() -> None:
    outcome = build_authoritative_trade_outcome(_sample(), reflection=_reflection())
    attribution = outcome["attribution"]

    assert attribution["direction_error"]["status"] == "unavailable"
    assert attribution["direction_error"]["contribution_usdt"] is None
    assert attribution["unknown_components_are_zero"] is False
    assert attribution["position_size_excess"]["contribution_usdt"] == -14.0
    assert attribution["stop_execution_slippage"]["contribution_usdt"] == pytest.approx(
        -1304.5 * 3.393665 / 100.0
    )


def test_missing_reflection_blocks_training_until_outcome_is_complete() -> None:
    outcome = build_authoritative_trade_outcome(_sample())
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[outcome],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert outcome["outcome_complete"] is False
    assert "missing_trade_reflection_link" in outcome["outcome_evidence_gaps"]
    assert payload["trade_samples"] == []
    record = payload["authoritative_outcome_manifest"]["records"][0]
    assert record["training_status"] == "excluded"


def test_complete_outcome_is_the_training_manifest_identity() -> None:
    outcome = build_authoritative_trade_outcome(_sample(), reflection=_reflection())
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[outcome],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert len(payload["trade_samples"]) == 1
    manifest = payload["authoritative_outcome_manifest"]
    assert manifest["record_count"] == 1
    assert manifest["included_count"] == 1
    assert manifest["records"][0]["outcome_id"] == outcome["outcome_id"]
