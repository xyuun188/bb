from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.okx_training_facts import build_okx_history_training_sample
from services.training_data_quality import annotate_training_payload


def _history(**overrides):
    opened = datetime(2026, 7, 11, 1, tzinfo=UTC)
    raw = {
        "instId": "BTC-USDT-SWAP",
        "posId": "pos-1",
        "posSide": "long",
        "realizedPnl": "8.5",
        "pnl": "10",
        "fee": "-1",
        "fundingFee": "-0.5",
        "pnlRatio": "0.085",
        "_bb_contract_spec": {
            "ctVal": "0.01",
            "ctMult": "1",
            "lotSz": "1",
        },
    }
    values = {
        "id": 1,
        "mode": "paper",
        "row_identity": "paper|BTC-USDT-SWAP|pos-1|long|1",
        "inst_id": "BTC-USDT-SWAP",
        "symbol": "BTC/USDT",
        "pos_id": "pos-1",
        "side": "long",
        "close_status": "full",
        "opened_at": opened,
        "updated_at_okx": opened + timedelta(hours=1),
        "open_avg_px": 100_000.0,
        "close_avg_px": 100_850.0,
        "open_max_pos": 2.0,
        "leverage": 2.0,
        "realized_pnl": 8.5,
        "pnl": 10.0,
        "pnl_ratio": 0.085,
        "funding_fee": -0.5,
        "fee": -1.0,
        "entry_order_ids": ["entry-1"],
        "close_order_ids": ["close-1"],
        "linked_order_ids": ["entry-1", "close-1"],
        "position_ids": [7],
        "evidence_gaps": [],
        "raw_row": raw,
        "sync_status": "synced",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _complete_lineage() -> dict:
    return {
        "positions_by_id": {
            7: SimpleNamespace(
                model_name="ensemble_trader",
                stop_loss_price=98_000.0,
                take_profit_price=104_000.0,
            )
        },
        "orders_by_exchange_id": {
            "entry-1": SimpleNamespace(
                okx_fill_contracts=2.0,
                okx_trade_ids="trade-entry",
                decision_id=91,
            ),
            "close-1": SimpleNamespace(
                okx_fill_contracts=2.0,
                okx_trade_ids="trade-close",
                decision_id=92,
            ),
        },
        "decision_raw_by_order_id": {
            "entry-1": {"opportunity_score": {"expected_net_return_pct": 0.8}}
        },
    }


def test_authoritative_okx_lifecycle_builds_one_contract_aware_sample() -> None:
    sample = build_okx_history_training_sample(
        _history(),
        **_complete_lineage(),
    )

    assert sample["source"] == "okx_position_history"
    assert sample["quantity"] == 2.0
    assert sample["quantity_unit"] == "contracts"
    assert sample["notional_usdt"] == 2000.0
    assert sample["authoritative_pnl_ratio_pct"] == 8.5
    assert sample["okx_trade_ids"] == ["trade-entry", "trade-close"]
    assert sample["trade_fact_trusted"] is True
    assert sample["training_evidence_gaps"] == []
    assert sample["strategy_lineage_complete"] is True


def test_authoritative_sample_uses_exact_entry_order_decision_evidence() -> None:
    entry = SimpleNamespace(
        okx_fill_contracts=2.0,
        okx_trade_ids="trade-entry",
        decision_id=91,
    )
    raw = {"local_ai_tools": {"time_series_prediction": {"model": "timesfm"}}}

    sample = build_okx_history_training_sample(
        _history(position_ids=[7]),
        orders_by_exchange_id={"entry-1": entry},
        decision_raw_by_position_id={7: {"local_ai_tools": {"wrong": True}}},
        decision_raw_by_order_id={"entry-1": raw},
    )

    assert sample["decision_id"] == 91
    assert sample["raw_llm_response"] == raw


def test_missing_official_funding_and_contract_spec_are_quarantined_with_reasons() -> None:
    history = _history()
    history.raw_row = {
        key: value
        for key, value in history.raw_row.items()
        if key not in {"fundingFee", "_bb_contract_spec"}
    }

    sample = build_okx_history_training_sample(history)
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert "missing_official_funding_fee" in sample["training_evidence_gaps"]
    assert "missing_contract_ct_val" in sample["training_evidence_gaps"]
    assert payload["trade_samples"] == []
    reasons = {item["reason"] for item in payload["quality_report"]["top_reasons"]}
    assert "trade:incomplete_okx_lifecycle:missing_official_funding_fee" in reasons


def test_training_report_blocks_pnl_return_sign_mismatch() -> None:
    sample = build_okx_history_training_sample(_history(), **_complete_lineage())
    sample["authoritative_pnl_ratio_pct"] = -8.5
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    consistency = payload["quality_report"]["training_label_consistency"]
    assert consistency["status"] == "blocked"
    assert consistency["promotion_blocked"] is True
    assert consistency["errors"][0]["reason"] == "pnl_return_sign_mismatch"


def test_authoritative_loss_with_exact_entry_lineage_remains_supervision_ready() -> None:
    history = _history(realized_pnl=-8.5, pnl=-7.0, pnl_ratio=-0.085)
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": "-8.5",
        "pnl": "-7",
        "pnlRatio": "-0.085",
    }
    sample = build_okx_history_training_sample(history, **_complete_lineage())

    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert len(payload["trade_samples"]) == 1
    trade = payload["trade_samples"][0]
    assert trade["data_quality_status"] == "included"
    labels = trade["profit_learning_labels"]
    assert labels["training_supervision_ready"] is True
    assert labels["exit_attribution_supervision_ready"] is True
    assert labels["losing_exit_attribution"] == "fee_after_return_forecast_error"
    assert labels["realized_net_pnl_usdt"] == -8.5


def test_entry_order_decision_id_is_preserved_when_raw_payload_is_empty() -> None:
    sample = build_okx_history_training_sample(
        _history(),
        orders_by_exchange_id={
            "entry-1": SimpleNamespace(decision_id=91, okx_fill_contracts=2.0),
            "close-1": SimpleNamespace(decision_id=92),
        },
    )

    assert sample["decision_id"] == 91
    assert sample["decision_lineage_source"] == "exact_entry_order_decision_id"
    assert "missing_exact_entry_order_decision_link" not in sample["strategy_lineage_gaps"]
    assert "missing_exact_entry_order_decision_payload" in sample["strategy_lineage_gaps"]


def test_position_fallback_payload_is_not_misreported_as_exact_entry_lineage() -> None:
    sample = build_okx_history_training_sample(
        _history(),
        decision_raw_by_position_id={7: {"opportunity_score": {"score": 1.0}}},
    )

    assert sample["decision_id"] == 0
    assert sample["decision_lineage_source"] == "position_time_fallback_payload"
    assert "missing_exact_entry_order_decision_link" in sample["strategy_lineage_gaps"]
