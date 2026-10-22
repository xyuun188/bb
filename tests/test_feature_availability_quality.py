from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from core.feature_availability import (
    FROZEN_FEATURE_FIELDS,
    availability_timestamp,
    feature_availability_reasons,
    seal_feature_availability,
)
from core.market_facts import MARKET_FACT_CONTRACT_VERSION
from core.training_contracts import (
    SHADOW_LABEL_VERSION,
    build_shadow_label_contract,
    compact_shadow_label_contract,
)
from data_feed.feature_vector import FeatureVector
from services import training_data_quality
from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.training_data_quality import (
    DATA_QUALITY_VERSION,
    annotate_sample,
    assess_shadow_sample,
    assess_trade_sample,
)


def test_availability_timestamp_accepts_rfc2822_news_dates() -> None:
    assert availability_timestamp("Wed, 21 Oct 2026 12:00:00 GMT") == datetime(
        2026, 10, 21, 12, 0, tzinfo=UTC
    )
    assert availability_timestamp("Wed, 21 Oct 2026 12:00:00") is None


def _shadow_sample() -> dict:
    label_at = datetime(2026, 10, 22, 0, 30, tzinfo=UTC)
    market_contract = {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "clean",
        "violation_count": 0,
        "violation_reason_codes": "",
        "native_instrument_identity_verified": True,
        "same_contract_price_path_verified": True,
        "executable_market_fact_verified": True,
        "data_fingerprint": "availability-quality-market-facts",
        "result_source_timestamp_ms": int(label_at.timestamp() * 1000),
        "result_received_at": label_at.isoformat(),
    }
    features = {
        "symbol": "BTC/USDT",
        "current_price": 100.0,
        "spread_pct": 0.03,
        "round_trip_fee_pct": 0.08,
        "funding_rate": 0.0,
        "funding_interval_minutes": 480.0,
        "training_market_fact_contract": market_contract,
    }
    features["training_label_contract"] = compact_shadow_label_contract(
        build_shadow_label_contract(
            shadow_backtest_id=1,
            decision_id=1001,
            horizon_minutes=30,
            long_return_pct=0.42,
            short_return_pct=-0.42,
            best_action="long",
            long_mfe_pct=0.42,
            long_mae_pct=0.0,
            short_mfe_pct=0.0,
            short_mae_pct=0.42,
            long_stop_loss_triggered=False,
            short_stop_loss_triggered=False,
            long_take_profit_triggered=False,
            short_take_profit_triggered=False,
            long_first_touch="none",
            short_first_touch="none",
            market_fact_contract=market_contract,
            cost_facts={"round_trip_fee_pct": 0.08},
            label_timestamp=label_at,
        )
    )
    return {
        "id": 1,
        "decision_id": 1001,
        "label_version": SHADOW_LABEL_VERSION,
        "symbol": "BTC/USDT",
        "decision_action": "long",
        "decision_confidence": 0.72,
        "horizon_minutes": 30,
        "features": features,
        "long_return_pct": 0.42,
        "short_return_pct": -0.42,
        "best_action": "long",
        "missed_opportunity": False,
        "label_timestamp": label_at,
    }


def _trade_sample() -> dict:
    return {
        "source": "closed_position",
        "symbol": "BTC/USDT",
        "side": "long",
        "entry_price": 100.0,
        "close_price": 101.0,
        "quantity": 0.1,
        "notional": 10.0,
        "realized_pnl": 1.0,
        "entry_fee": 0.02,
        "close_fee": 0.03,
        "entry_fee_source": "okx_fills_history",
        "close_fee_source": "okx_fills_history",
        "funding_fee": 0.0,
        "funding_fee_source": "okx_positions_history.fundingFee",
        "slippage": 0.01,
        "slippage_source": "okx_fills_history_fill_mark_round_trip",
        "net_return_after_all_cost_pct": 10.0,
        "pnl_source": "okx_position_history_realized_pnl",
        "settlement_source": "okx_position_history_settlement",
        "settlement_status": "reconciled",
        "position_size_pct": 0.03,
        "holding_minutes": 35.0,
        "outcome": "profit",
    }


def _seal_valid_shadow_availability(
    sample: dict,
    *,
    technical_received_at: str = "2026-10-22T00:00:02+00:00",
    news_received_at: str | None = None,
) -> None:
    features = sample["features"]
    features.update(
        {
            "original_decision_timestamp": "2026-10-22T00:00:10+00:00",
            "technical_availability": {
                "version": "2026-10-22.observed-component.v1",
                "available_at": "2026-10-22T00:00:03+00:00",
                "timeframes": "1m,15m",
                "source_delivery_observed": True,
                "1m_closed_at": "2026-10-22T00:00:01+00:00",
                "1m_received_at": technical_received_at,
                "15m_closed_at": "2026-10-22T00:00:01+00:00",
                "15m_received_at": technical_received_at,
                "input_fingerprint": "test-technical-inputs",
            },
            "sentiment_availability": {
                "version": "2026-10-22.observed-component.v1",
                "available_at": "2026-10-22T00:00:03+00:00",
                "news_input_count": 1 if news_received_at else 0,
                "news_clock_errors": 0,
                "news_input_fingerprint": "test-news-inputs",
                **(
                    {"news_received_at": news_received_at}
                    if news_received_at is not None
                    else {}
                ),
                "social_input_count": 0,
                "social_clock_errors": 0,
                "social_input_fingerprint": "test-social-inputs",
            },
            "funding_rate_observed_at": "2026-10-22T00:00:01+00:00",
            "funding_rate_received_at": "2026-10-22T00:00:02+00:00",
            "account_fee_source_observed_at": "2026-10-22T00:00:01+00:00",
            "account_fee_received_at": "2026-10-22T00:00:02+00:00",
        }
    )
    for name, second in (
        ("market_fact", "01"),
        ("orderbook_fact", "01"),
        ("mark_price_fact", "02"),
        ("index_price_fact", "03"),
    ):
        features[name] = {
            "timestamp": f"2026-10-22T00:00:{second}+00:00",
            "received_at": "2026-10-22T00:00:04+00:00",
        }
    seal_feature_availability(
        features,
        frozen_at=datetime(2026, 10, 22, 0, 0, 5, tzinfo=UTC),
    )


@pytest.mark.parametrize("kind", ["shadow", "trade"])
def test_legacy_sample_cannot_train_without_original_component_availability(kind: str) -> None:
    sample = _shadow_sample() if kind == "shadow" else _trade_sample()
    assessment = (
        assess_shadow_sample(sample) if kind == "shadow" else assess_trade_sample(sample)
    )

    assert assessment.exclude_from_training is True
    assert assessment.weight == 0.0
    assert any(reason.startswith("feature_availability:") for reason in assessment.reasons)


def test_excluding_legacy_trade_preserves_authoritative_monetary_audit() -> None:
    sample = _trade_sample()
    annotated = annotate_sample(sample, "trade")

    assert annotated["exclude_from_training"] is True
    assert annotated["realized_pnl"] == sample["realized_pnl"]
    assert annotated["entry_fee"] == sample["entry_fee"]
    assert annotated["close_fee"] == sample["close_fee"]
    assert annotated["profit_learning_labels"]["realized_net_pnl_usdt"] == 1.0
    assert annotated["profit_learning_labels"]["training_supervision_ready"] is False


def test_quality_checks_keep_structural_failure_before_component_availability() -> None:
    sample = _shadow_sample()
    sample["features"].pop("funding_interval_minutes")

    assessment = assess_shadow_sample(sample)

    assert assessment.reasons == ("cost_incomplete:funding_interval_missing",)


def test_quality_passes_original_decision_completion_without_timestamp_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _shadow_sample()
    sample["features"]["timestamp"] = "2026-10-22T00:00:00+00:00"
    sample["features"]["original_decision_timestamp"] = "2026-10-22T00:00:20+00:00"
    sample["original_decision_timestamp"] = "2026-10-22T00:00:10+00:00"
    checked: list[object] = []

    def check(features: dict, *, decision_at: object = None) -> list[str]:
        assert features is sample["features"]
        checked.append(decision_at)
        return ["frozen_after_decision"]

    monkeypatch.setattr(training_data_quality, "feature_availability_reasons", check)
    assessment = assess_shadow_sample(sample)

    assert checked == ["2026-10-22T00:00:10+00:00"]
    assert assessment.reasons == ("feature_availability:frozen_after_decision",)
    sample.pop("original_decision_timestamp")
    sample["features"].pop("original_decision_timestamp")
    assess_shadow_sample(sample)
    assert checked[-1] is None


def test_training_fingerprint_binds_original_clock_and_component_evidence() -> None:
    sample = _shadow_sample()
    sample["features"].update(
        {
            "feature_snapshot_frozen_at": "2026-10-22T00:00:01+00:00",
            "original_decision_timestamp": "2026-10-22T00:00:02+00:00",
            "feature_availability_contract": {
                "version": "test-original-availability",
                "frozen_at": "2026-10-22T00:00:01+00:00",
                "input_fingerprint": "original",
            },
            "technical_availability": {
                "available_at": "2026-10-22T00:00:00+00:00",
            },
            "sentiment_availability": {
                "available_at": "2026-10-21T23:59:00+00:00",
            },
        }
    )
    original = training_data_quality._training_sample_contract(sample, kind="shadow")
    for path, replacement in (
        (("original_decision_timestamp",), "2026-10-22T00:00:03+00:00"),
        (("feature_snapshot_frozen_at",), "2026-10-22T00:00:04+00:00"),
        (("technical_availability", "available_at"), "2026-10-21T23:59:59+00:00"),
        (("sentiment_availability", "available_at"), "2026-10-21T23:58:59+00:00"),
        (("feature_availability_contract", "input_fingerprint"), "rewritten"),
    ):
        changed = deepcopy(sample)
        target = changed["features"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        changed_contract = training_data_quality._training_sample_contract(
            changed, kind="shadow"
        )
        assert changed_contract["sample_fingerprint"] != original["sample_fingerprint"]

    assert original["lineage"]["feature_snapshot_frozen_at"] == (
        "2026-10-22T00:00:01+00:00"
    )
    assert original["feature_availability"]["technical_availability"] == {
        "available_at": "2026-10-22T00:00:00+00:00"
    }


def test_decision_completion_metadata_added_after_freeze_does_not_change_input_fingerprint() -> None:
    sample = _shadow_sample()
    _seal_valid_shadow_availability(sample)
    features = sample["features"]
    decision_at = features.pop("original_decision_timestamp")
    features.pop("feature_snapshot_frozen_at")
    features.pop("feature_availability_contract")
    seal_feature_availability(
        features,
        frozen_at=datetime(2026, 10, 22, 0, 0, 5, tzinfo=UTC),
    )

    features["original_decision_timestamp"] = decision_at

    reasons = feature_availability_reasons(features, decision_at=decision_at)

    assert "feature_inputs_changed_after_freeze" not in reasons


def test_component_availability_contract_retires_earlier_training_versions() -> None:
    assert DATA_QUALITY_VERSION == "2026-10-22.component-feature-availability.v8"
    assert PROFIT_SUPERVISION_VERSION == (
        "2026-10-22.separated-profit-component-availability.v4"
    )


def test_every_non_text_feature_vector_input_is_bound_by_the_availability_fingerprint() -> None:
    non_input_metadata = {
        "feature_snapshot_frozen_at",
        "feature_availability_contract",
        "recent_headlines",
        "recent_news_items",
    }

    assert set(FeatureVector.__dataclass_fields__) - non_input_metadata <= set(
        FROZEN_FEATURE_FIELDS
    )


def test_fresh_shadow_with_complete_observed_availability_is_eligible() -> None:
    sample = _shadow_sample()
    _seal_valid_shadow_availability(sample)

    assessment = assess_shadow_sample(sample)

    assert assessment.exclude_from_training is False
    assert assessment.status == "included"


def test_late_component_receipt_after_freeze_is_excluded() -> None:
    sample = _shadow_sample()
    _seal_valid_shadow_availability(
        sample,
        technical_received_at="2026-10-22T00:00:06+00:00",
    )
    assessment = assess_shadow_sample(sample)

    assert assessment.exclude_from_training is True
    assert any("technical:1m:received_after_freeze" in r for r in assessment.reasons)


def test_ambiguous_naive_component_clock_is_excluded() -> None:
    sample = _shadow_sample()
    _seal_valid_shadow_availability(
        sample,
        news_received_at="2026-10-22T00:00:02",
    )
    assessment = assess_shadow_sample(sample)

    assert assessment.exclude_from_training is True
    assert any("news:receipt_missing_or_ambiguous" in r for r in assessment.reasons)


def test_changed_clock_after_seal_is_excluded_as_tampered_input() -> None:
    sample = _shadow_sample()
    _seal_valid_shadow_availability(sample)
    sample["features"]["technical_availability"]["1m_received_at"] = (
        "2026-10-22T00:00:03+00:00"
    )

    assessment = assess_shadow_sample(sample)

    assert assessment.exclude_from_training is True
    assert "feature_availability:feature_inputs_changed_after_freeze" in (
        assessment.reasons
    )
