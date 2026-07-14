from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from config.settings import settings
from core.market_facts import MARKET_FACT_CONTRACT_VERSION
from core.training_contracts import (
    SHADOW_LABEL_VERSION,
    build_shadow_label_contract,
    compact_shadow_label_contract,
)
from db.session import close_db, get_session_ctx, init_db
from models.learning import ShadowBacktest
from scripts.quarantine_dirty_shadow_training_samples import (
    QUARANTINE_STATUS,
    _note_with_quarantine_reason,
    _quality_sample,
)
from services.shadow_training_quarantine import (
    quarantine_completed_shadow_row,
    quarantine_dirty_shadow_samples,
)
from services.training_data_quality import assess_shadow_sample


def _row(**overrides):
    due_at = datetime(2026, 6, 23, 1, 0, tzinfo=UTC)
    row = SimpleNamespace(
        id=101,
        decision_id=1001,
        label_version=SHADOW_LABEL_VERSION,
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
        due_at=due_at,
        note="",
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    features = row.feature_snapshot
    if isinstance(features, dict) and features:
        market_contract = {
            "version": MARKET_FACT_CONTRACT_VERSION,
            "status": "clean",
            "violation_count": 0,
            "violation_reason_codes": "",
            "native_instrument_identity_verified": True,
            "same_contract_price_path_verified": True,
            "executable_market_fact_verified": True,
            "data_fingerprint": "shadow-quarantine-test",
        }
        features.setdefault("training_market_fact_contract", market_contract)
        features.setdefault(
            "training_label_contract",
            compact_shadow_label_contract(
                build_shadow_label_contract(
                    shadow_backtest_id=row.id,
                    decision_id=row.decision_id,
                    horizon_minutes=row.horizon_minutes,
                    long_return_pct=row.long_return_pct,
                    short_return_pct=row.short_return_pct,
                    best_action=row.best_action,
                    market_fact_contract=market_contract,
                    cost_facts={"round_trip_fee_pct": features.get("round_trip_fee_pct")},
                    label_timestamp=row.due_at,
                )
            ),
        )
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


@pytest.mark.asyncio
async def test_dry_run_audits_completed_and_already_quarantined_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    await close_db()
    db_path = tmp_path / "shadow-audit.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()
    clean_features = _row().feature_snapshot
    due_at = datetime(2026, 7, 14, 1, 0, tzinfo=UTC)

    def db_row(row_id: int, status: str, features: dict) -> ShadowBacktest:
        feature_payload = dict(features)
        if feature_payload:
            feature_payload["training_label_contract"] = compact_shadow_label_contract(
                build_shadow_label_contract(
                    shadow_backtest_id=row_id,
                    decision_id=9000 + row_id,
                    horizon_minutes=30,
                    long_return_pct=0.4,
                    short_return_pct=-0.4,
                    best_action="long",
                    market_fact_contract=feature_payload.get(
                        "training_market_fact_contract"
                    ),
                    cost_facts={"round_trip_fee_pct": 0.08},
                    label_timestamp=due_at,
                )
            )
        return ShadowBacktest(
            id=row_id,
            decision_id=9000 + row_id,
            label_version=SHADOW_LABEL_VERSION,
            model_name="ensemble_trader",
            execution_mode="paper",
            symbol=f"TEST{row_id}/USDT",
            analysis_type="market",
            decision_action="hold",
            decision_confidence=0.7,
            entry_price=100.0,
            feature_snapshot=feature_payload,
            status=status,
            due_at=due_at,
            horizon_minutes=30,
            actual_price=101.0,
            long_return_pct=0.4,
            short_return_pct=-0.4,
            best_action="long",
            missed_opportunity=True,
            note="",
        )

    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    db_row(1, "completed", dict(clean_features)),
                    db_row(2, "completed", {}),
                    db_row(3, "quarantined", {}),
                ]
            )

        report = await quarantine_dirty_shadow_samples(
            dry_run=True,
            newest_first=False,
            batch_size=10,
            max_batches=2,
        )
    finally:
        await close_db()

    assert report["total_candidate_count"] == 3
    assert report["scanned"] == 3
    assert report["coverage_complete"] is True
    assert report["trainable"] == 1
    assert report["quarantined"] == 2
    assert report["already_quarantined"] == 1
    assert report["new_quarantine_candidates"] == 1
