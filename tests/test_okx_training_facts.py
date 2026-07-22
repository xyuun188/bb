from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.authoritative_trade_outcome import build_authoritative_trade_outcome
from services.okx_training_facts import build_okx_history_training_sample
from services.paper_exploration import build_paper_exploration_contract
from services.paper_training import build_paper_training_contract
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
        "pnlRatio": "0.0085",
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
        "close_avg_px": 100_500.0,
        "open_max_pos": 2.0,
        "leverage": 2.0,
        "realized_pnl": 8.5,
        "pnl": 10.0,
        "pnl_ratio": 0.0085,
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


def _outcome(sample: dict) -> dict:
    reflection = SimpleNamespace(
        id=501,
        position_id=7,
        source="authoritative_trade_outcome",
        outcome=sample.get("outcome"),
        mistake_summary="fact",
        improvement_summary="recalibrate distribution",
        created_at=datetime(2026, 7, 11, 2, tzinfo=UTC),
    )
    return build_authoritative_trade_outcome(sample, reflection=reflection)


def test_authoritative_okx_lifecycle_builds_one_contract_aware_sample() -> None:
    sample = build_okx_history_training_sample(
        _history(),
        **_complete_lineage(),
    )

    assert sample["source"] == "okx_position_history"
    assert sample["quantity"] == 2.0
    assert sample["quantity_unit"] == "contracts"
    assert sample["notional_usdt"] == 2000.0
    assert sample["notional_source"] == "okx_gross_pnl_and_average_price_path"
    assert sample["gross_return_price_consistent"] is True
    assert sample["authoritative_pnl_ratio_pct"] == pytest.approx(0.85)
    assert sample["okx_trade_ids"] == ["trade-entry", "trade-close"]
    assert sample["trade_fact_trusted"] is True
    assert sample["training_evidence_gaps"] == []
    assert sample["strategy_lineage_complete"] is True

    outcome = _outcome(sample)
    label = outcome["training_label_contract"]
    assert label["execution_mode"] == "paper"
    assert label["realized_fee_after_return_pct"] == pytest.approx(8.5 / 2000.0 * 100.0)
    assert label["realized_net_pnl_usdt"] == 8.5


def test_paper_training_prefers_verified_account_contract_size_over_public_spec() -> None:
    history = _history(pnl=100.0, realized_pnl=98.5)
    lineage = _complete_lineage()
    lineage["orders_by_exchange_id"]["entry-1"].okx_raw_fills = {
        "contract_size": 0.1,
        "contract_size_verified": True,
        "contract_size_source": "okx_account_position_margin_notional_crosscheck",
    }
    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["contract_ct_val"] == pytest.approx(0.1)
    assert sample["contract_ct_val_source"].startswith("okx_account_position_")
    assert sample["contract_ct_val_corrected"] is True
    assert sample["notional_usdt"] == pytest.approx(20_000.0)
    assert "account_contract_size_evidence_conflict" not in sample["training_evidence_gaps"]
    assert sample["trade_fact_trusted"] is True


def test_valid_paper_exploration_is_a_normal_trainable_trade_with_selection_reason() -> None:
    provenance = {
        "source": "test_cost_complete_return_distribution",
        "observation_window": "current_test_candidate",
        "sample_count": 3,
        "generated_at": "2026-07-21T00:00:00+00:00",
        "strategy_version": "test.v1",
        "fallback_reason": "",
    }
    selected = {
        "eligible": True,
        "side": "long",
        "expected_net_return_pct": 0.3,
        "return_lcb_pct": -0.1,
        "lcb_gap_ratio": 1.0 / 3.0,
        "loss_probability": 0.3,
        "tail_risk_score": 0.2,
        "return_source_count": 3,
        "historical_evidence_count": 0,
        "exploration_allocation_multiplier": 1.0,
        "prediction_horizon_minutes": 30.0,
        "valid_for_seconds": 1800.0,
        "feature_opportunity_score": 8.0,
        "information_value_score": 0.04,
        "policy_provenance": provenance,
    }
    evidence = {
        "preferred_exploration_side": "long",
        "paper_exploration": {
            "preferred_side": "long",
            "selected": selected,
            "reason": "bounded_paper_exploration_side_selected",
        },
    }
    contract = build_paper_exploration_contract(evidence, symbol="BTC/USDT")
    lineage = _complete_lineage()
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "entry_candidate_evidence": evidence,
        "paper_exploration": contract,
    }

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert sample["strategy_entry_supervision_eligible"] is True
    assert sample["strategy_training_role"] == "entry_strategy"
    assert sample["strategy_entry_kind"] == "bounded_risk_paper_exploration"
    assert sample["strategy_selection_reason"] == (
        "bounded_paper_exploration_side_selected"
    )
    assert sample["paper_exploration_evidence"]["sample_target"] is None
    assert sample["paper_exploration_evidence"]["daily_sample_quota"] is None


def test_paper_training_loss_is_a_normal_authoritative_training_sample() -> None:
    lineage = _complete_lineage()
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "paper_training": build_paper_training_contract(
            symbol="BTC/USDT",
            selected_side="long",
            signal_source="local_ml_observation",
            expected_net_return_pct=-0.5,
            return_lcb_pct=-0.8,
            horizon_minutes=10.0,
        ),
        "paper_training_mode": "bootstrap",
    }
    history = _history(
        close_avg_px=99_650.0,
        realized_pnl=-8.5,
        pnl=-7.0,
        pnl_ratio=-0.0085,
    )
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": "-8.5",
        "pnl": "-7",
        "pnlRatio": "-0.0085",
    }

    sample = _outcome(build_okx_history_training_sample(history, **lineage))
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert sample["strategy_entry_supervision_eligible"] is True
    assert sample["strategy_training_role"] == "entry_strategy"
    assert sample["strategy_entry_kind"] == "loss_tolerant_paper_training"
    assert sample["paper_training_evidence"]["loss_tolerant_for_training"] is True
    assert sample["paper_training_evidence"]["sample_target"] is None
    assert sample["paper_training_evidence"]["daily_sample_quota"] is None
    assert len(payload["trade_samples"]) == 1
    assert payload["trade_samples"][0]["profit_learning_labels"][
        "realized_net_pnl_usdt"
    ] == -8.5


def test_paper_training_contract_is_never_trainable_as_a_live_trade() -> None:
    lineage = _complete_lineage()
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "paper_training": build_paper_training_contract(
            symbol="BTC/USDT",
            selected_side="long",
            signal_source="local_ml_observation",
            horizon_minutes=10.0,
        )
    }

    sample = build_okx_history_training_sample(
        _history(mode="live"),
        **lineage,
    )

    assert sample["strategy_entry_supervision_eligible"] is False
    assert sample["strategy_training_role"] == "invalid_paper_training_research_only"
    assert "invalid_paper_training_contract" in sample["training_evidence_gaps"]


def test_stale_contract_multiplier_uses_authoritative_gross_pnl_price_path() -> None:
    history = _history(
        inst_id="LIT-USDT-SWAP",
        symbol="LIT/USDT",
        open_avg_px=2.5,
        close_avg_px=2.35,
        realized_pnl=-4.55,
        pnl=-4.5,
        fee=-0.05,
        funding_fee=0.0,
        pnl_ratio=-0.0606666667,
    )
    history.raw_row = {
        **history.raw_row,
        "instId": "LIT-USDT-SWAP",
        "realizedPnl": "-4.55",
        "pnl": "-4.5",
        "fee": "-0.05",
        "fundingFee": "0",
        "pnlRatio": "-0.0606666667",
        "_bb_contract_spec": {"ctVal": "1", "ctMult": "1", "lotSz": "1"},
    }
    lineage = _complete_lineage()
    lineage["orders_by_exchange_id"]["entry-1"].okx_fill_contracts = 3.0

    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["contract_spec_notional_usdt"] == pytest.approx(7.5)
    assert sample["notional_usdt"] == pytest.approx(75.0)
    assert sample["contract_notional_corrected"] is True
    assert sample["gross_price_return_pct"] == pytest.approx(-6.0)
    assert sample["gross_return_on_notional_pct"] == pytest.approx(-6.0)
    assert sample["gross_return_price_consistent"] is True
    assert "gross_return_price_path_mismatch" not in sample["training_evidence_gaps"]


def test_multiple_entry_decisions_are_quarantined_from_strategy_training() -> None:
    lineage = _complete_lineage()
    lineage["orders_by_exchange_id"]["entry-2"] = SimpleNamespace(
        okx_fill_contracts=1.0,
        okx_trade_ids="trade-entry-2",
        decision_id=93,
    )
    history = _history(entry_order_ids=["entry-1", "entry-2"])

    sample = _outcome(build_okx_history_training_sample(history, **lineage))
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert sample["entry_decision_ids"] == [91, 93]
    assert sample["decision_id"] == 0
    assert sample["strategy_training_role"] == "aggregate_position_research_only"
    assert "multiple_entry_decision_lineage" in sample["training_evidence_gaps"]
    assert payload["trade_samples"] == []


def test_obsolete_sampling_entry_is_research_only() -> None:
    lineage = _complete_lineage()
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "paper_bootstrap_canary": {
            "trade_kind": "observation_only_probe",
            "continuous_training_after_settlement": False,
        }
    }

    sample = _outcome(build_okx_history_training_sample(_history(), **lineage))
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert sample["strategy_training_role"] == "obsolete_sampling_research_only"
    assert "obsolete_sampling_entry_not_strategy_trainable" in sample[
        "training_evidence_gaps"
    ]
    assert payload["trade_samples"] == []


def test_okx_demo_alias_normalizes_to_paper_and_invalid_mode_is_quarantined() -> None:
    demo = build_okx_history_training_sample(
        _history(mode="demo"),
        **_complete_lineage(),
    )
    invalid = build_okx_history_training_sample(
        _history(mode="unknown"),
        **_complete_lineage(),
    )

    assert demo["execution_mode"] == "paper"
    assert demo["source_execution_mode"] == "demo"
    assert "missing_or_invalid_execution_mode" not in demo["training_evidence_gaps"]
    assert invalid["execution_mode"] == ""
    assert "missing_or_invalid_execution_mode" in invalid["training_evidence_gaps"]


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


def test_exact_entry_decision_recovers_missing_planned_protection_prices() -> None:
    lineage = _complete_lineage()
    lineage["positions_by_id"][7].stop_loss_price = None
    lineage["positions_by_id"][7].take_profit_price = None
    lineage["decision_execution_by_order_id"] = {
        "entry-1": {
            "decision_id": 91,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }
    }

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert sample["planned_stop_loss_price"] == pytest.approx(98_000.0)
    assert sample["planned_take_profit_price"] == pytest.approx(104_000.0)
    assert "missing_planned_stop_loss_lineage" not in sample["strategy_lineage_gaps"]
    assert "missing_planned_take_profit_lineage" not in sample["strategy_lineage_gaps"]


def test_missing_official_funding_and_contract_spec_are_quarantined_with_reasons() -> None:
    history = _history()
    history.raw_row = {
        key: value
        for key, value in history.raw_row.items()
        if key not in {"fundingFee", "_bb_contract_spec"}
    }

    sample = _outcome(build_okx_history_training_sample(history))
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
    sample = _outcome(build_okx_history_training_sample(_history(), **_complete_lineage()))
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
    history = _history(
        close_avg_px=99_650.0,
        realized_pnl=-8.5,
        pnl=-7.0,
        pnl_ratio=-0.0085,
    )
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": "-8.5",
        "pnl": "-7",
        "pnlRatio": "-0.0085",
    }
    sample = _outcome(build_okx_history_training_sample(history, **_complete_lineage()))

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
    assert labels["losing_exit_attribution"] == "authoritative_multi_factor_outcome"
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


def test_stop_slippage_uses_exchange_algo_trigger_not_local_planned_stop() -> None:
    lineage = _complete_lineage()
    lineage["orders_by_exchange_id"]["entry-1"].okx_raw_fills = {
        "protection_submission": {
            "source_authority": "local_submit_plus_okx_create_order_response",
            "exchange_confirmation_recorded": True,
            "exchange_confirmed_at": "2026-07-11T01:00:01+00:00",
            "algo_ids": ["algo-stop-1"],
        }
    }
    lineage["orders_by_exchange_id"]["close-1"].okx_raw_fills = {
        "protection_execution": {
            "source_authority": "okx_algo_history_plus_fills_history",
            "lifecycle_complete": True,
            "algo_id": "algo-stop-1",
            "generated_order_id": "close-1",
            "actual_side": "sl",
            "configured_trigger_price": 97_500.0,
            "actual_trigger_market_price": None,
            "actual_trigger_market_price_available": False,
            "exchange_confirmed_at_ms": 1783731601000,
            "triggered_at_ms": 1783735200000,
            "fill_started_at_ms": 1783735200025,
            "fill_completed_at_ms": 1783735200030,
            "trigger_to_first_fill_ms": 25.0,
            "fill_mark_price": 97_450.0,
            "fill_index_price": 97_460.0,
            "fill_path_min_price": 96_950.0,
            "fill_path_max_price": 97_100.0,
            "fill_mark_slippage_pct": 0.461775,
            "trigger_path_extrema_available": False,
            "trigger_orderbook_snapshot_available": False,
            "stop_loss_slippage_pct": (97_500.0 - 97_000.0) / 97_500.0 * 100.0,
            "stop_loss_slippage_source": "okx_configured_stop_trigger_to_fills_vwap",
        }
    }
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "profit_risk_sizing": {
            "risk_budget_usdt": 5.0,
            "planned_stressed_loss_usdt": 4.5,
        }
    }
    history = _history(
        close_avg_px=97_000.0,
        realized_pnl=-8.5,
        pnl=-7.0,
        pnl_ratio=-0.085,
    )
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": "-8.5",
        "pnl": "-7",
        "pnlRatio": "-0.085",
    }

    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["stop_loss_fill_confirmed"] is True
    assert sample["stop_loss_slippage_pct"] == pytest.approx(
        (97_500.0 - 97_000.0) / 97_500.0 * 100.0
    )
    assert sample["stop_loss_slippage_pct"] != pytest.approx(
        (98_000.0 - 97_000.0) / 98_000.0 * 100.0
    )
    assert sample["stop_loss_slippage_source"] == (
        "okx_configured_stop_trigger_to_fills_vwap"
    )
    assert sample["actual_trigger_market_price"] is None
    assert sample["protection_lifecycle_complete"] is True
    assert sample["trigger_to_first_fill_ms"] == pytest.approx(25.0)
    assert sample["execution_actual_over_budget_loss_usdt"] == pytest.approx(3.5)
    assert "actual_trigger_market_price_unavailable" in sample["protection_execution_gaps"]


def test_legacy_stop_order_type_cannot_recreate_planned_price_slippage() -> None:
    lineage = _complete_lineage()
    lineage["orders_by_exchange_id"]["close-1"].order_type = "stop_loss"

    sample = build_okx_history_training_sample(
        _history(close_avg_px=97_000.0),
        **lineage,
    )

    assert sample["stop_loss_fill_confirmed"] is False
    assert sample["stop_loss_slippage_pct"] is None
    assert sample["stop_loss_slippage_source"] == "not_authoritatively_confirmed"
