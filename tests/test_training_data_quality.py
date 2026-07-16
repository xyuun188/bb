from __future__ import annotations

from datetime import UTC, datetime

from core.market_facts import MARKET_FACT_CONTRACT_VERSION
from core.training_contracts import (
    SHADOW_LABEL_VERSION,
    build_shadow_label_contract,
    compact_shadow_label_contract,
)
from services import training_data_quality
from services.authoritative_trade_outcome import AUTHORITATIVE_TRADE_OUTCOME_VERSION
from services.model_promotion_policy import build_return_objective_report
from services.training_data_quality import (
    DATA_QUALITY_VERSION,
    annotate_training_payload,
    assess_sequence_sample,
    assess_shadow_sample,
    assess_text_sentiment_sample,
    assess_trade_sample,
    governance_report,
)


def _clean_market_fact_contract() -> dict:
    return {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "clean",
        "violation_count": 0,
        "violation_reason_codes": "",
        "native_instrument_identity_verified": True,
        "same_contract_price_path_verified": True,
        "executable_market_fact_verified": True,
        "data_fingerprint": "test-shadow-market-facts",
    }


def _shadow_sample(**overrides):
    sample = {
        "id": 1,
        "decision_id": 1001,
        "label_version": SHADOW_LABEL_VERSION,
        "symbol": "BTC/USDT",
        "decision_action": "long",
        "decision_confidence": 0.72,
        "horizon_minutes": 30,
        "features": {
            "symbol": "BTC/USDT",
            "current_price": 100.0,
            "spread_pct": 0.03,
            "round_trip_fee_pct": 0.08,
            "funding_rate": 0.0,
            "funding_interval_minutes": 480.0,
            "training_market_fact_contract": _clean_market_fact_contract(),
        },
        "long_return_pct": 0.42,
        "short_return_pct": -0.31,
        "best_action": "long",
        "missed_opportunity": False,
        "label_timestamp": datetime(2026, 7, 14, 0, 30, tzinfo=UTC),
    }
    override_features = overrides.pop("features", None) if "features" in overrides else None
    sample.update(overrides)
    if override_features is not None:
        features = dict(override_features)
        if features:
            features.setdefault(
                "training_market_fact_contract",
                _clean_market_fact_contract(),
            )
        sample["features"] = features
    features = sample.get("features")
    if isinstance(features, dict) and features and "training_label_contract" not in features:
        features["training_label_contract"] = compact_shadow_label_contract(
            build_shadow_label_contract(
                shadow_backtest_id=int(sample.get("id") or 0),
                decision_id=int(sample.get("decision_id") or 0) or None,
                horizon_minutes=int(sample.get("horizon_minutes") or 0),
                long_return_pct=float(sample.get("long_return_pct") or 0.0),
                short_return_pct=float(sample.get("short_return_pct") or 0.0),
                best_action=str(sample.get("best_action") or "hold"),
                market_fact_contract=features.get("training_market_fact_contract"),
                cost_facts={"round_trip_fee_pct": features.get("round_trip_fee_pct")},
                label_timestamp=sample.get("label_timestamp"),
            )
        )
    return sample


def _trade_sample(**overrides):
    sample = {
        "source": "closed_position",
        "symbol": "BTC/USDT",
        "side": "long",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "quantity": 0.1,
        "realized_pnl": 1.0,
        "fee_estimate": 0.05,
        "funding_fee": 0.0,
        "funding_fee_source": "okx_positions_history.fundingFee",
        "pnl_source": "okx_position_history_realized_pnl",
        "settlement_source": "okx_position_history_settlement",
        "settlement_status": "reconciled",
        "position_size_pct": 0.03,
        "hold_minutes": 35.0,
        "outcome": "profit",
    }
    sample.update(overrides)
    return sample


def test_profit_learning_report_does_not_invent_profit_factor_without_losses() -> None:
    report = training_data_quality._profit_learning_report(
        [
            {
                "profit_learning_labels": {
                    "sample_kind": "trade",
                    "training_supervision_ready": True,
                    "realized_net_pnl_usdt": 2.5,
                    "net_return_after_cost_pct": 0.4,
                }
            }
        ]
    )["after_fee_quality"]

    assert report["net_realized_pnl_usdt"] == 2.5
    assert report["profit_factor"] is None
    assert "profit_factor_undefined_without_losses" in report["quality_warnings"]


def test_shadow_hold_samples_are_not_penalized_by_action_type() -> None:
    assessment = assess_shadow_sample(_shadow_sample(decision_action="hold"))

    assert assessment.status == "included"
    assert assessment.exclude_from_training is False
    assert assessment.weight == 1.0
    assert assessment.reasons == ()


def test_shadow_missing_features_are_excluded() -> None:
    assessment = assess_shadow_sample(_shadow_sample(features={}))

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert "missing_features" in assessment.reasons


def test_shadow_missing_funding_interval_is_excluded_from_training() -> None:
    sample = _shadow_sample()
    sample["features"] = dict(sample["features"])
    sample["features"].pop("funding_interval_minutes")

    assessment = assess_shadow_sample(sample)

    assert assessment.status == "excluded"
    assert assessment.reasons == ("cost_incomplete:funding_interval_missing",)


def test_shadow_mojibake_payload_is_excluded() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={"current_price": 100.0, "spread_pct": 0.01, "note": "鏈轰細璇勫垎"}
        )
    )

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert "mojibake_text" in assessment.reasons


def test_duplicate_training_sample_is_excluded() -> None:
    assessment = assess_shadow_sample(_shadow_sample(duplicate_count=2))

    assert assessment.status == "excluded"
    assert "duplicate_sample" in assessment.reasons


def test_trade_hold_duration_does_not_apply_fixed_fast_loss_penalty() -> None:
    assessment = assess_trade_sample(_trade_sample(realized_pnl=-0.8, hold_minutes=1.5))

    assert "fast_loss_exit_requires_review" not in assessment.reasons


def test_manual_trade_samples_are_excluded() -> None:
    assessment = assess_trade_sample(_trade_sample(source="manual"))

    assert assessment.status == "excluded"
    assert "manual_or_test_trade" in assessment.reasons


def test_trade_missing_fee_is_excluded_but_size_and_old_tier_are_not_gates() -> None:
    no_fee = assess_trade_sample(_trade_sample(fee_estimate=None))
    micro_probe = assess_trade_sample(
        _trade_sample(position_size_pct=0.0003, evidence_tier="weak_conflict_probe")
    )

    assert no_fee.status == "excluded"
    assert "missing_fee_estimate" in no_fee.reasons
    assert micro_probe.status == "included"
    assert "weak_evidence_micro_probe" not in micro_probe.reasons


def test_trade_mode_mixing_and_failed_close_are_excluded() -> None:
    mode_mixed = assess_trade_sample(_trade_sample(execution_mode="shadow"))
    bad_close = assess_trade_sample(_trade_sample(close_status="failed"))

    assert mode_mixed.status == "excluded"
    assert "execution_mode_mismatch" in mode_mixed.reasons
    assert bad_close.status == "excluded"
    assert "failed_close_status" in bad_close.reasons


def test_trade_historical_repair_and_untrusted_fact_are_excluded() -> None:
    repaired = assess_trade_sample(
        _trade_sample(
            source="trade_reflection",
            reflection_source="okx_order_pair_repair",
            trade_fact_repair_source="missing_closed_position_repair",
        )
    )
    untrusted = assess_trade_sample(
        _trade_sample(
            trade_fact_trusted=False,
            trade_fact_trust_reason="missing_close_exchange_order_id",
        )
    )

    assert repaired.status == "excluded"
    assert (
        "historical_reconciliation_repair:missing_closed_position_repair"
        in repaired.reasons
    )
    assert untrusted.status == "excluded"
    assert (
        "untrusted_trade_fact:missing_close_exchange_order_id"
        in untrusted.reasons
    )


def test_closed_position_training_requires_authoritative_pnl_and_funding_sources() -> None:
    local_pnl = assess_trade_sample(
        _trade_sample(
            pnl_source="position_realized_pnl",
            settlement_source="",
        )
    )
    missing_funding = assess_trade_sample(
        _trade_sample(
            funding_fee=None,
            funding_fee_source="",
        )
    )
    explicit_untrusted = assess_trade_sample(
        _trade_sample(
            trade_fact_trusted=False,
            trade_fact_trust_reason="",
        )
    )

    assert local_pnl.status == "excluded"
    assert "untrusted_realized_pnl_source:position_realized_pnl" in local_pnl.reasons
    assert missing_funding.status == "excluded"
    assert "missing_or_untrusted_funding_fee_source" in missing_funding.reasons
    assert explicit_untrusted.status == "excluded"
    assert "untrusted_trade_fact" in explicit_untrusted.reasons


def test_trade_unknown_losing_exit_attribution_is_excluded() -> None:
    assessment = assess_trade_sample(
        _trade_sample(
            realized_pnl=-0.7,
            hold_minutes=16.0,
            fee_estimate=0.02,
            raw_llm_response={},
        )
    )

    assert assessment.status == "excluded"
    assert "unknown_losing_exit_attribution" in assessment.reasons


def test_sequence_future_leakage_is_excluded() -> None:
    assessment = assess_sequence_sample(
        {
            "close_sequence": [100 + idx for idx in range(30)],
            "future_return_pct": 0.3,
            "timeframe": "1m",
            "feature_timestamp": datetime(2026, 6, 23, 1, 10, tzinfo=UTC),
            "label_timestamp": datetime(2026, 6, 23, 1, 0, tzinfo=UTC),
        }
    )

    assert assessment.status == "excluded"
    assert "future_leakage" in assessment.reasons


def test_shadow_feature_snapshot_future_leakage_is_excluded() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={
                "current_price": 100.0,
                "spread_pct": 0.01,
                "feature_timestamp": datetime(2026, 6, 23, 1, 10, tzinfo=UTC).isoformat(),
            },
            label_timestamp=datetime(2026, 6, 23, 1, 0, tzinfo=UTC).isoformat(),
        )
    )

    assert assessment.status == "excluded"
    assert "future_leakage" in assessment.reasons


def test_text_sentiment_mojibake_and_duplicate_samples_are_excluded() -> None:
    mojibake = assess_text_sentiment_sample(
        {
            "source": "news",
            "platform": "scrapling:okx",
            "text": "閺堣桨绱扮拠鍕瀻 market event summary",
            "sentiment_score": 0.2,
        }
    )
    duplicate = assess_text_sentiment_sample(
        {
            "source": "news",
            "platform": "scrapling:okx",
            "text": "OKX listing update with enough text for training",
            "sentiment_score": 0.2,
            "duplicate_of": "news:1",
        }
    )

    assert mojibake.status == "excluded"
    assert "mojibake_text" in mojibake.reasons
    assert duplicate.status == "excluded"
    assert "duplicate_sample" in duplicate.reasons


def test_training_payload_returns_trainable_samples_and_quality_report() -> None:
    payload = annotate_training_payload(
        shadow_samples=[_shadow_sample(), _shadow_sample(features={})],
        trade_samples=[_trade_sample(), _trade_sample(source="manual")],
        sequence_samples=[],
        text_sentiment_samples=[
            {
                "source": "news",
                "platform": "scrapling:ethereum_blog",
                "text": "Ethereum upgrade launch with ecosystem impact",
                "sentiment_score": 0.2,
            }
        ],
    )

    assert len(payload["shadow_samples"]) == 1
    assert len(payload["trade_samples"]) == 1
    report = payload["quality_report"]
    assert report["data_quality_version"] == DATA_QUALITY_VERSION
    assert report["totals"]["excluded"] == 2
    assert report["totals"]["effective_weight"] > 0
    assert report["totals"]["effective_weight_ratio"] > 0
    assert report["by_kind"]["shadow"]["total"] == 2
    assert report["by_kind"]["trade"]["total"] == 2
    assert report["by_kind"]["shadow"]["actions"]["long"] == 2
    assert report["by_kind"]["shadow"]["trainable_actions"]["long"] == 1
    assert report["by_kind"]["trade"]["sources"]["closed_position"] == 1
    assert report["by_kind"]["trade"]["trainable_sources"]["closed_position"] == 1
    assert report["by_kind"]["text_sentiment"]["sources"]["scrapling:ethereum_blog"] == 1
    assert report["by_kind"]["text_sentiment"]["trainable_sources"]["scrapling:ethereum_blog"] == 1
    assert report["specialist_shadow_models"] == {}
    market_contract = report["market_fact_contract"]
    assert market_contract["status"] == "clean"
    assert market_contract["violation_count"] == 0
    assert market_contract["assertions"]["native_instrument_identity_verified"] is True
    shadow_contract = payload["shadow_samples"][0]["training_sample_contract"]
    assert shadow_contract["immutable"] is True
    assert len(shadow_contract["sample_fingerprint"]) == 64
    assert shadow_contract["market_fact_contract"]["status"] == "clean"
    assert "hold_observation_penalty" not in report["policy"]
    assert "include_score_threshold" not in report["policy"]
    governance = payload["governance_report"]
    assert governance["cleanup_mode"] == "quarantine_not_delete"
    assert governance["training_policy"] == "clean_training_view_only"
    assert governance["raw_records_preserved"] is True
    assert governance["quarantine_applied"] is True
    assert governance["requires_artifact_refresh"] is True
    assert "local_ai_tools" in governance["refresh_targets"]


def test_training_payload_enriches_trade_profit_learning_labels() -> None:
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[
            _trade_sample(
                source="closed_position",
                realized_pnl=-0.12,
                fee_estimate=0.08,
                hold_minutes=18.0,
                leverage=1.0,
                loss_attribution="position_too_small_fee_drag",
                raw_llm_response={
                    "production_return_policy": {
                        "position_size_pct": 0.01,
                        "production_source_count": 2,
                        "policy_provenance": {
                            "strategy_version": "return-test-v1",
                        },
                    },
                    "profit_risk_sizing": {
                        "position_size_pct": 0.01,
                        "final_notional_usdt": 12.0,
                    },
                },
            )
        ],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    trade = payload["trade_samples"][0]
    labels = trade["profit_learning_labels"]
    assert labels["version"] == "separated-profit-supervision-v4"
    assert labels["training_supervision_ready"] is True
    assert labels["losing_exit_attribution"] == "position_too_small_fee_drag"
    assert labels["trade_profit_class"] == "cost_drag_loss"
    assert "size_efficiency_label" not in labels
    assert labels["cost_basis_label"] == "fee_plus_funding"
    assert labels["realized_net_pnl_usdt"] == -0.12
    assert labels["return_after_cost_pct"] == -1.0
    assert labels["net_return_after_cost_pct"] == -1.0
    assert labels["return_on_margin_pct"] == -1.0
    assert labels["return_after_cost_pct_deprecated"] is True
    assert labels["strategy_context"]["return_policy_version"] == "return-test-v1"
    assert labels["strategy_context"]["return_policy_source_count"] == 2
    assert "decision_lane" not in labels["strategy_context"]
    assert trade["losing_exit_attribution"] == "position_too_small_fee_drag"
    report = payload["quality_report"]["by_kind"]["trade"]["profit_learning"]
    assert report["supervision_ready_count"] == 1
    assert report["after_fee_quality"] == {
        "trade_count": 1,
        "win_count": 0,
        "loss_count": 1,
        "flat_count": 0,
        "win_rate": 0.0,
        "net_realized_pnl_usdt": -0.12,
        "gross_profit_usdt": 0.0,
        "gross_loss_usdt": 0.12,
        "profit_factor": 0.0,
        "avg_net_pnl_usdt": -0.12,
        "avg_win_usdt": 0.0,
        "avg_loss_usdt": 0.12,
        "avg_return_after_cost_pct": -1.0,
        "small_win_big_loss_ratio": 0.0,
        "quality_warnings": ["gross_loss_not_covered_by_profit"],
    }
    assert report["label_counts"]["losing_exit_attribution"][0]["value"] == (
        "position_too_small_fee_drag"
    )


def test_training_payload_trade_contract_feeds_return_objective_report() -> None:
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[
            _trade_sample(
                source="okx_position_history",
                event_type="AuthoritativeTradeOutcome",
                outcome_version=AUTHORITATIVE_TRADE_OUTCOME_VERSION,
                outcome_id="ato:test-1",
                outcome_fingerprint="fingerprint-test-1",
                trade_fact_trusted=True,
                lifecycle_key="okx-position:test-1",
                execution_slippage_usdt=0.0,
            )
        ],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    report = build_return_objective_report(trade_samples=payload["trade_samples"])

    assert report["available"] is True
    assert report["sample_count"] == 1
    assert report["average_net_return_after_cost_pct"] == 10.0
    assert "authoritative_realized_return_distribution_missing" not in report[
        "blocking_reasons"
    ]


def test_training_payload_enriches_shadow_missed_opportunity_labels() -> None:
    payload = annotate_training_payload(
        shadow_samples=[
            _shadow_sample(
                decision_action="hold",
                best_action="long",
                long_return_pct=0.9,
                short_return_pct=-0.3,
                missed_opportunity=True,
            )
        ],
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    shadow = payload["shadow_samples"][0]
    labels = shadow["profit_learning_labels"]
    assert labels["missed_opportunity_label"] == "missed_positive_entry"
    assert labels["shadow_outcome_label"] == "positive_shadow_edge"
    summary = payload["quality_report"]["profit_learning_summary"]["shadow"]
    assert summary["label_counts"]["missed_opportunity_label"][0]["value"] == (
        "missed_positive_entry"
    )


def test_training_payload_reports_specialist_shadow_model_quality() -> None:
    payload = annotate_training_payload(
        shadow_samples=[
            _shadow_sample(
                best_action="long",
                features={
                    "symbol": "BTC/USDT",
                    "current_price": 100.0,
                    "spread_pct": 0.03,
                    "local_ai_tools_shadow": {
                        "time_series_prediction": {
                            "timesfm_shadow_side": "long",
                            "timesfm_shadow_expected_return_pct": 0.42,
                            "specialist_inference_active": True,
                            "professional_model_shadow": {
                                "actual_inference": True,
                                "primary_shadow_result": {
                                    "model": "chronos-2-shadow-primary",
                                    "actual_inference": True,
                                    "expected_return_pct": 0.21,
                                    "best_side": "long",
                                    "sequence_length": 60,
                                },
                                "challenger_shadow_result": {
                                    "model": "timesfm-2.5-shadow-challenger",
                                    "actual_inference": True,
                                    "expected_return_pct": 0.42,
                                    "best_side": "long",
                                    "sequence_length": 60,
                                },
                            },
                        },
                        "sentiment_analysis": {
                            "best_side": "short",
                            "expected_return_pct": -0.12,
                            "specialist_inference_active": True,
                            "professional_model_shadow": {"actual_inference": True},
                        },
                    },
                },
            )
        ],
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    specialist = payload["quality_report"]["specialist_shadow_models"]
    chronos = specialist["time_series_prediction:chronos-2-shadow-primary"]
    timesfm = specialist["time_series_prediction:timesfm-2.5-shadow-challenger"]
    sentiment = specialist["sentiment_analysis"]

    assert chronos["tool"] == "time_series_prediction"
    assert chronos["model"] == "chronos-2-shadow-primary"
    assert chronos["sample_count"] == 1
    assert chronos["actual_inference_count"] == 1
    assert chronos["direction_hit_count"] == 1
    assert chronos["direction_hit_rate"] == 1.0
    assert chronos["avg_shadow_expected_return_pct"] == 0.21
    assert chronos["avg_realized_return_pct"] == 0.42
    assert chronos["return_lower_hinge_pct"] == 0.42
    assert chronos["observation_policy"]["promotion_authority"] is False
    assert timesfm["tool"] == "time_series_prediction"
    assert timesfm["model"] == "timesfm-2.5-shadow-challenger"
    assert timesfm["sample_count"] == 1
    assert timesfm["actual_inference_count"] == 1
    assert timesfm["specialist_inference_count"] == 1
    assert timesfm["direction_hit_count"] == 1
    assert timesfm["direction_hit_rate"] == 1.0
    assert timesfm["avg_shadow_expected_return_pct"] == 0.42
    assert timesfm["avg_realized_return_pct"] == 0.42
    assert timesfm["return_lower_hinge_pct"] == 0.42
    assert sentiment["direction_hit_count"] == 0


def test_training_payload_reports_specialist_tail_loss_quality() -> None:
    payload = annotate_training_payload(
        shadow_samples=[
            _shadow_sample(
                symbol="ACT/USDT",
                best_action="short",
                long_return_pct=-0.25,
                short_return_pct=0.18,
                features={
                    "symbol": "ACT/USDT",
                    "current_price": 100.0,
                    "spread_pct": 0.03,
                    "local_ai_tools_shadow": {
                        "time_series_prediction": {
                            "specialist_inference_active": True,
                            "professional_model_shadow": {
                                "actual_inference": True,
                                "primary_shadow_result": {
                                    "model": "chronos-2-shadow-primary",
                                    "actual_inference": True,
                                    "expected_return_pct": 0.31,
                                    "best_side": "long",
                                    "sequence_length": 60,
                                },
                            },
                        },
                    },
                },
            )
            for _ in range(34)
        ],
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    model = payload["quality_report"]["specialist_shadow_models"][
        "time_series_prediction:chronos-2-shadow-primary"
    ]

    assert model["actual_inference_count"] == 34
    assert model["direction_count"] == 34
    assert model["direction_hit_rate"] == 0.0
    assert model["false_signal_count"] == 34
    assert model["avg_realized_return_pct"] == -0.25
    assert model["worst_realized_return_pct"] == -0.25
    assert model["return_lower_hinge_pct"] == -0.25
    assert model["return_distribution_provenance"]["sample_count"] == 34
    assert model["worst_samples"][0]["predicted_side"] == "long"
    assert model["worst_samples"][0]["actual_best_side"] == "short"
    assert model["observation_policy"]["promotion_authority"] is False
    assert "promotion_blockers" not in model


def test_training_payload_quarantines_legacy_mixed_timeseries_shadow() -> None:
    payload = annotate_training_payload(
        shadow_samples=[
            _shadow_sample(
                best_action="long",
                features={
                    "symbol": "BTC/USDT",
                    "current_price": 100.0,
                    "spread_pct": 0.03,
                    "local_ai_tools_shadow": {
                        "time_series_prediction": {
                            "timesfm_shadow_side": "long",
                            "timesfm_shadow_expected_return_pct": 0.42,
                            "specialist_inference_active": True,
                            "professional_model_shadow": {
                                "actual_inference": True,
                                "shadow_result": {
                                    "model": "chronos-2-shadow-primary",
                                    "actual_inference": True,
                                    "expected_return_pct": 0.42,
                                    "best_side": "long",
                                    "sequence_length": 4,
                                },
                            },
                        },
                    },
                },
            )
        ],
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    specialist = payload["quality_report"]["specialist_shadow_models"]
    legacy = specialist["time_series_prediction:chronos-2-shadow-primary"]

    assert legacy["legacy_mixed_shadow_count"] == 1
    assert legacy["legacy_quarantined_count"] == 1
    assert legacy["actual_inference_count"] == 0
    assert legacy["direction_count"] == 0
    assert legacy["return_lower_hinge_pct"] is None
    assert legacy["return_distribution_provenance"]["production_eligible"] is False
    assert legacy["observation_policy"]["promotion_authority"] is False
    assert "promotion_ready" not in legacy


def test_training_payload_skips_baseline_only_profit_shadow_model_quality() -> None:
    payload = annotate_training_payload(
        shadow_samples=[
            _shadow_sample(
                best_action="long",
                features={
                    "symbol": "BTC/USDT",
                    "current_price": 100.0,
                    "spread_pct": 0.03,
                    "local_ai_tools_shadow": {
                        "profit_prediction": {
                            "best_side": "long",
                            "expected_return_pct": 0.42,
                            "specialist_inference_active": False,
                            "professional_model_shadow": {
                                "kind": "profit",
                                "actual_inference": False,
                                "baseline_response": True,
                            },
                        }
                    },
                },
            )
        ],
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert "profit_prediction" not in payload["quality_report"]["specialist_shadow_models"]


def test_training_payload_skips_non_specialist_heuristic_shadow_model_quality() -> None:
    payload = annotate_training_payload(
        shadow_samples=[
            _shadow_sample(
                best_action="long",
                features={
                    "symbol": "BTC/USDT",
                    "current_price": 100.0,
                    "spread_pct": 0.03,
                    "local_ai_tools_shadow": {
                        "profit_prediction": {
                            "model": "local-profit-heuristic-v1",
                            "best_side": "long",
                            "expected_return_pct": 0.42,
                            "specialist_inference_active": False,
                            "professional_model_shadow": {
                                "kind": "profit",
                                "actual_inference": False,
                                "baseline_response": False,
                            },
                        }
                    },
                },
            )
        ],
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert "profit_prediction" not in payload["quality_report"]["specialist_shadow_models"]


def test_shadow_market_data_quality_issue_is_excluded_from_training() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={
                "symbol": "BTC/USDT",
                "current_price": 100.0,
                "spread_pct": 0.03,
                "market_data_quality": {
                    "code": "price_source_split",
                    "exclude_from_training": True,
                    "training_quality_reason": "market_data_quality:price_source_split",
                },
            }
        )
    )

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert "market_data_quality:price_source_split" in assessment.reasons


def test_shadow_label_contract_is_required_and_tamper_evident() -> None:
    missing = _shadow_sample()
    missing["features"]["training_label_contract"] = {}

    missing_assessment = assess_shadow_sample(missing)

    assert missing_assessment.status == "excluded"
    assert (
        "shadow_label_contract:shadow_label_contract_missing"
        in missing_assessment.reasons
    )

    tampered = _shadow_sample(id=2, decision_id=1002)
    tampered["features"]["training_label_contract"]["long_return_pct"] = 99.0

    tampered_assessment = assess_shadow_sample(tampered)

    assert tampered_assessment.status == "excluded"
    assert (
        "shadow_label_contract:shadow_label_compact_fingerprint_mismatch"
        in tampered_assessment.reasons
    )

    stale_row_version = _shadow_sample(id=3, decision_id=1003)
    stale_row_version["label_version"] = "legacy-row-3"

    stale_assessment = assess_shadow_sample(stale_row_version)

    assert stale_assessment.status == "excluded"
    assert (
        "shadow_label_contract:shadow_label_row_version_mismatch"
        in stale_assessment.reasons
    )


def test_same_decision_horizon_label_version_has_one_trainable_identity() -> None:
    first = _shadow_sample(id=11, decision_id=5011)
    second = _shadow_sample(id=12, decision_id=5011)

    payload = annotate_training_payload(
        shadow_samples=[first, second],
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert len(payload["shadow_samples"]) == 1
    diagnostics = payload["quality_report"]["training_view_diagnostics"]
    assert diagnostics["raw_sample_count"] == 2
    assert diagnostics["trainable_sample_count"] == 1
    assert diagnostics["quarantined_sample_count"] == 1
    assert payload["quality_report"]["top_reasons"][0] == {
        "reason": "shadow:duplicate_decision_horizon_label_version",
        "count": 1,
    }


def test_training_view_reports_leave_one_symbol_out_and_time_influence() -> None:
    samples = [
        _shadow_sample(
            id=21,
            decision_id=6021,
            symbol="ROBO/USDT",
            long_return_pct=-95.0,
            short_return_pct=95.0,
            best_action="short",
        ),
        _shadow_sample(
            id=22,
            decision_id=6022,
            symbol="BTC/USDT",
            long_return_pct=0.4,
            short_return_pct=-0.3,
            best_action="long",
        ),
        _shadow_sample(
            id=23,
            decision_id=6023,
            symbol="ETH/USDT",
            long_return_pct=0.2,
            short_return_pct=-0.1,
            best_action="long",
            label_timestamp=datetime(2026, 7, 15, 0, 30, tzinfo=UTC),
        ),
    ]

    payload = annotate_training_payload(
        shadow_samples=samples,
        trade_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    diagnostics = payload["quality_report"]["training_view_diagnostics"]

    assert diagnostics["raw_sample_count"] == 3
    assert diagnostics["effective_sample_size"] == 3.0
    assert diagnostics["max_single_symbol_influence"]["symbol"] == "ROBO/USDT"
    assert {item["date"] for item in diagnostics["time_influence"]} == {
        "2026-07-14",
        "2026-07-15",
    }


def test_shadow_price_reconciliation_warning_is_excluded_from_training() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={
                "symbol": "PROS/USDT",
                "current_price": 0.5666,
                "close": 0.5666,
                "indicator_close_price": 0.3902,
                "indicator_price_gap_pct": 45.18,
                "price_reconciliation_warning": (
                    "ticker_current_price_kept_indicator_close_diverged"
                ),
                "spread_pct": 0.03,
            }
        )
    )

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert (
        "price_reconciliation:ticker_current_price_kept_indicator_close_diverged"
        in assessment.reasons
    )


def test_shadow_price_range_does_not_apply_fixed_tolerance() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={
                "symbol": "PROS/USDT",
                "current_price": 0.3902,
                "low_24h": 0.5491,
                "high_24h": 0.5707,
                "spread_pct": 0.03,
                "round_trip_fee_pct": 0.08,
                "funding_rate": 0.0,
                "funding_interval_minutes": 480.0,
            }
        )
    )

    assert assessment.status == "included"
    assert assessment.weight == 1.0
    assert assessment.reasons == ()


def test_training_governance_report_preserves_raw_records_and_targets_refresh() -> None:
    report = governance_report(
        {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {
                "total": 10,
                "included": 6,
                "downweighted": 3,
                "excluded": 1,
                "effective_weight_ratio": 0.72,
            },
            "top_reasons": [{"reason": "trade:manual_or_test_trade", "count": 1}],
        }
    )

    assert report["status"] == "quarantined"
    assert report["raw_records_preserved"] is True
    assert report["excluded_sample_count"] == 1
    assert report["downweighted_sample_count"] == 3
    assert report["contamination_risk"] == "unknown"
    assert report["contamination_classification_complete"] is False
    assert "vector_memory_reindex" in report["refresh_targets"]


def test_training_governance_treats_quarantined_contamination_as_low_residual_risk() -> None:
    report = governance_report(
        {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {
                "total": 9951,
                "included": 9938,
                "downweighted": 0,
                "benign_downweighted": 0,
                "contamination_downweighted": 0,
                "excluded": 13,
                "effective_weight_ratio": 0.9987,
            },
            "top_reasons": [{"reason": "sequence:abnormal_future_return", "count": 13}],
        }
    )

    assert report["status"] == "quarantined"
    assert report["contamination_risk"] == "low"
    assert report["contamination_risk_basis"] == "all_identified_contamination_quarantined"
    assert report["contamination_classification_complete"] is True
    assert report["excluded_ratio"] == 0.001306
    assert report["blocked_reason_count"] == 13
    assert report["requires_artifact_refresh"] is True

    refreshed = governance_report(
        {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {
                "total": 9951,
                "included": 9938,
                "downweighted": 0,
                "benign_downweighted": 0,
                "contamination_downweighted": 0,
                "excluded": 13,
                "effective_weight_ratio": 0.9987,
            },
            "top_reasons": [{"reason": "sequence:abnormal_future_return", "count": 13}],
        },
        artifact_quality_fingerprint=report["quality_fingerprint"],
    )
    assert refreshed["artifact_matches_quality"] is True
    assert refreshed["requires_artifact_refresh"] is False


def test_training_governance_blocks_contamination_that_remains_trainable() -> None:
    report = governance_report(
        {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {
                "total": 4,
                "included": 2,
                "downweighted": 1,
                "benign_downweighted": 0,
                "contamination_downweighted": 1,
                "excluded": 1,
                "effective_weight_ratio": 0.625,
            },
            "top_reasons": [{"reason": "shadow:invalid_market_fact", "count": 2}],
        }
    )

    assert report["contamination_risk"] == "high"
    assert report["contamination_risk_basis"] == "trainable_contamination_present"
    assert report["contamination_classification_complete"] is True


def test_artifact_bound_governance_report_marks_new_artifact_current() -> None:
    quality = training_data_quality.quality_report(
        {
            "shadow": [
                {
                    "data_quality_status": "downweighted",
                    "sample_weight": 0.5,
                    "quality_reasons": ["wide_spread_feature"],
                }
            ]
        }
    )

    report = training_data_quality.artifact_bound_governance_report(
        quality,
        persist_artifact=True,
    )

    assert report["artifact_quality_fingerprint"] == report["quality_fingerprint"]
    assert report["artifact_matches_quality"] is True
    assert report["requires_artifact_refresh"] is False
    assert report["contamination_risk"] == "high"


def test_trade_return_uses_valid_derived_notional_over_tiny_placeholder() -> None:
    payload = annotate_training_payload(
        trade_samples=[
            _trade_sample(
                quantity=2.0,
                entry_price=100.0,
                realized_pnl=4.0,
                notional_usdt=0.000001,
            )
        ],
        shadow_samples=[],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    labels = payload["trade_samples"][0]["profit_learning_labels"]
    assert labels["notional_usdt"] == 200.0
    assert labels["return_after_cost_pct"] == 2.0
    assert labels["net_return_after_cost_pct"] == 2.0


def test_trade_return_is_missing_for_zero_or_malformed_notional() -> None:
    assert (
        training_data_quality._trade_notional_usdt(
            {"quantity": "bad", "entry_price": "bad", "notional_usdt": 0}
        )
        is None
    )
