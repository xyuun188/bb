from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from scripts.quarantine_dirty_shadow_training_samples import (
    QUARANTINE_STATUS,
    _note_with_quarantine_reason,
    _quality_sample,
)
from services.shadow_training_quarantine import quarantine_completed_shadow_row
from services.training_data_quality import assess_shadow_sample


def _row(**overrides):
    row = SimpleNamespace(
        symbol="PROS/USDT",
        analysis_type="market",
        decision_action="long",
        decision_confidence=0.72,
        horizon_minutes=30,
        feature_snapshot={
            "symbol": "PROS/USDT",
            "current_price": 0.3902,
            "low_24h": 0.5491,
            "high_24h": 0.5707,
            "spread_pct": 0.03,
            "round_trip_fee_pct": 0.08,
            "funding_rate": 0.0,
            "funding_interval_minutes": 480.0,
        },
        long_return_pct=0.4,
        short_return_pct=-0.4,
        best_action="long",
        missed_opportunity=False,
        status="completed",
        due_at=datetime(2026, 6, 23, 1, 0, tzinfo=UTC),
        note="",
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def test_quality_sample_does_not_apply_fixed_price_range_quarantine() -> None:
    assessment = assess_shadow_sample(_quality_sample(_row()))

    assert assessment.exclude_from_training is False
    assert assessment.status == "included"
    assert "price_outside_24h_range" not in assessment.reasons


def test_quality_sample_marks_feature_snapshot_future_leakage_for_quarantine() -> None:
    row = _row(
        feature_snapshot={
            "symbol": "PROS/USDT",
            "current_price": 0.5600,
            "low_24h": 0.5491,
            "high_24h": 0.5707,
            "spread_pct": 0.03,
            "feature_timestamp": datetime(2026, 6, 23, 1, 5, tzinfo=UTC).isoformat(),
        },
        due_at=datetime(2026, 6, 23, 1, 0, tzinfo=UTC),
    )

    assessment = assess_shadow_sample(_quality_sample(row))

    assert assessment.exclude_from_training is True
    assert assessment.status == "excluded"
    assert "future_leakage" in assessment.reasons


def test_note_with_quarantine_reason_is_idempotent() -> None:
    note = _note_with_quarantine_reason("old note", ("future_leakage",))
    second = _note_with_quarantine_reason(note, ("future_leakage",))

    assert "[training_quarantine] future_leakage" in note
    assert second == note


def test_quarantine_status_constant_is_not_completed() -> None:
    assert QUARANTINE_STATUS == "quarantined"
    assert QUARANTINE_STATUS != "completed"


def test_quarantine_completed_shadow_row_ignores_old_price_range_rule() -> None:
    row = _row()

    result = quarantine_completed_shadow_row(row)

    assert result["applied"] is False
    assert row.status == "completed"
    assert "price_outside_24h_range" not in row.note


def test_clean_shadow_row_is_not_quarantined() -> None:
    row = _row(
        feature_snapshot={
            "symbol": "PROS/USDT",
            "current_price": 0.5600,
            "low_24h": 0.5491,
            "high_24h": 0.5707,
            "spread_pct": 0.03,
            "round_trip_fee_pct": 0.08,
            "funding_rate": 0.0,
            "funding_interval_minutes": 480.0,
        },
    )

    result = quarantine_completed_shadow_row(row)

    assert result["applied"] is False
    assert row.status == "completed"
