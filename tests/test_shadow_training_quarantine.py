from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
from services.historical_shadow_rebuild import build_historical_shadow_sample
from services.shadow_training_quarantine import (
    quarantine_completed_shadow_row,
    quarantine_dirty_shadow_samples,
    requalify_verified_historical_shadow_samples,
)
from services.training_data_quality import assess_shadow_sample
from tests.feature_availability_fixtures import seal_synthetic_feature_availability


@pytest.fixture(autouse=True)
def _current_training_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.shadow_training_quarantine.load_training_data_start",
        lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


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
            "result_source_timestamp_ms": int(row.due_at.timestamp() * 1000),
            "result_received_at": row.due_at.isoformat(),
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
                    long_mfe_pct=max(float(row.long_return_pct or 0.0), 0.0),
                    long_mae_pct=max(-float(row.long_return_pct or 0.0), 0.0),
                    short_mfe_pct=max(float(row.short_return_pct or 0.0), 0.0),
                    short_mae_pct=max(-float(row.short_return_pct or 0.0), 0.0),
                    long_stop_loss_triggered=False,
                    short_stop_loss_triggered=False,
                    long_take_profit_triggered=False,
                    short_take_profit_triggered=False,
                    long_first_touch="none",
                    short_first_touch="none",
                    market_fact_contract=market_contract,
                    cost_facts={"round_trip_fee_pct": features.get("round_trip_fee_pct")},
                    label_timestamp=row.due_at,
                )
            ),
        )
        seal_synthetic_feature_availability(
            features,
            decision_at=row.due_at - timedelta(minutes=int(row.horizon_minutes or 30)),
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
            feature_payload["training_market_fact_contract"] = {
                **feature_payload["training_market_fact_contract"],
                "result_source_timestamp_ms": int(due_at.timestamp() * 1000),
                "result_received_at": due_at.isoformat(),
            }
            feature_payload["training_label_contract"] = compact_shadow_label_contract(
                build_shadow_label_contract(
                    shadow_backtest_id=row_id,
                    decision_id=9000 + row_id,
                    horizon_minutes=30,
                    long_return_pct=0.4,
                        short_return_pct=-0.4,
                        best_action="long",
                        long_mfe_pct=0.4,
                        long_mae_pct=0.0,
                        short_mfe_pct=0.0,
                        short_mae_pct=0.4,
                        long_stop_loss_triggered=False,
                        short_stop_loss_triggered=False,
                        long_take_profit_triggered=False,
                        short_take_profit_triggered=False,
                        long_first_touch="none",
                        short_first_touch="none",
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


@pytest.mark.asyncio
async def test_requalify_only_verified_historical_shadow_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    await close_db()
    db_path = tmp_path / "shadow-requalify.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()

    created_at = datetime(2026, 7, 14, 1, 0, 25, tzinfo=UTC)
    decision = SimpleNamespace(
        id=9001,
        model_name="ensemble_trader",
        symbol="PROS/USDT",
        action="hold",
        confidence=0.7,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        is_paper=True,
        created_at=created_at,
        feature_snapshot={
            "current_price": 100.0,
            "market_data_quality": {
                "code": "orderbook_depth_invalid",
                "exclude_from_training": True,
            },
            "training_quality_reason": "market_data_quality:orderbook_depth_invalid",
        },
    )
    bars = [
        {
            "open_time": datetime(2026, 7, 14, 1, minute, tzinfo=UTC),
            "open": 100.0 + minute,
            "high": 101.0 + minute,
            "low": 99.0 + minute,
            "close": 100.0 + minute,
            "volume": 1000.0 + minute,
        }
        for minute in range(6)
    ]

    def db_row(
        row_id: int,
        *,
        note: str,
        execution_mode: str = "paper",
        analysis_type: str = "market",
        historical: bool = True,
        valid_path: bool = True,
    ) -> ShadowBacktest:
        row_decision = SimpleNamespace(**vars(decision))
        row_decision.id = 9000 + row_id
        sample = build_historical_shadow_sample(
            row_decision,
            horizon_minutes=5,
            bars=bars,
            shadow_backtest_id=row_id,
        )
        assert sample is not None
        feature_payload = dict(sample["feature_snapshot"])
        feature_payload["historical_market_path_only"] = historical
        if not valid_path:
            feature_payload["training_market_fact_contract"] = {
                **feature_payload["training_market_fact_contract"],
                "path_status": "invalid",
            }
        return ShadowBacktest(
            id=row_id,
            decision_id=sample["decision_id"],
            label_version=sample["label_version"],
            model_name=sample["model_name"],
            execution_mode=execution_mode,
            symbol=sample["symbol"],
            analysis_type=analysis_type,
            decision_action=sample["decision_action"],
            decision_confidence=sample["decision_confidence"],
            entry_price=sample["entry_price"],
            feature_snapshot=feature_payload,
            status="quarantined",
            created_at=created_at,
            updated_at=created_at,
            due_at=sample["due_at"],
            horizon_minutes=sample["horizon_minutes"],
            actual_price=sample["actual_price"],
            long_return_pct=sample["long_return_pct"],
            short_return_pct=sample["short_return_pct"],
            best_action=sample["best_action"],
            missed_opportunity=sample["missed_opportunity"],
            note=note,
        )

    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    db_row(
                        1,
                        note=(
                            "[training_quarantine] "
                            "market_data_quality:orderbook_depth_invalid"
                        ),
                    ),
                    db_row(
                        2,
                        note=(
                            "[training_quarantine] "
                            "market_data_quality:orderbook_depth_invalid"
                        ),
                        valid_path=False,
                    ),
                    db_row(
                        3,
                        note=(
                            "[training_quarantine] "
                            "market_data_quality:orderbook_depth_invalid,future_leakage"
                        ),
                    ),
                    db_row(
                        4,
                        note=(
                            "[training_quarantine] "
                            "market_data_quality:orderbook_depth_invalid"
                        ),
                        historical=False,
                    ),
                    db_row(
                        5,
                        note=(
                            "[training_quarantine] "
                            "market_data_quality:orderbook_depth_invalid"
                        ),
                        execution_mode="live",
                    ),
                    db_row(
                        6,
                        note=(
                            "[training_quarantine] "
                            "market_data_quality:orderbook_depth_invalid"
                        ),
                        analysis_type="position",
                    ),
                    db_row(
                        7,
                        note=(
                            "[training_quarantine] "
                            "market_data_quality:orderbook_depth_invalid\n"
                            "[training_quarantine] future_leakage"
                        ),
                    ),
                ]
            )

        dry_report = await requalify_verified_historical_shadow_samples(
            batch_size=1,
            max_batches=10,
            dry_run=True,
        )
        assert dry_report["eligible"] == 0
        assert dry_report["requalified"] == 0
        assert dry_report["candidate_ids_sample"] == []
        assert dry_report["scanned"] == 4
        assert dry_report["coverage_complete"] is True
        assert dry_report["skipped_reasons"].get(
            "feature_availability:feature_availability_contract_missing", 0
        ) >= 1

        async with get_session_ctx() as session:
            stored = await session.get(ShadowBacktest, 1)
            assert stored is not None
            assert stored.status == "quarantined"
            compact = stored.training_feature_snapshot
            assert compact["historical_market_path_only"] is True
            assert compact["historical_shadow_rebuild_version"]
            assert compact["training_market_fact_contract"]["path_fingerprint"]
            assert compact["training_label_contract"]["label_fingerprint"]

        bounded_report = await requalify_verified_historical_shadow_samples(
            batch_size=1,
            max_batches=1,
            only_up_to_id=1,
        )
        assert bounded_report["eligible"] == 0
        assert bounded_report["last_scanned_id"] == 1
        assert bounded_report["coverage_complete"] is False
        resumed_report = await requalify_verified_historical_shadow_samples(
            batch_size=1,
            max_batches=10,
            only_newer_than_id=bounded_report["last_scanned_id"],
            only_up_to_id=3,
        )
        assert resumed_report["scanned"] == 2
        assert resumed_report["eligible"] == 0
        assert resumed_report["last_scanned_id"] == 3
        assert resumed_report["coverage_complete"] is True
        assert resumed_report["only_newer_than_id"] == 1
        assert resumed_report["only_up_to_id"] == 3

        apply_report = await requalify_verified_historical_shadow_samples(
            batch_size=1,
            max_batches=10,
            dry_run=False,
        )
        assert apply_report["eligible"] == 0
        assert apply_report["requalified"] == 0

        async with get_session_ctx() as session:
            rows = {
                row_id: await session.get(ShadowBacktest, row_id)
                for row_id in range(1, 8)
            }
        assert rows[1] is not None and rows[1].status == "quarantined"
        assert all(rows[row_id].status == "quarantined" for row_id in range(1, 8))

        repeat_report = await requalify_verified_historical_shadow_samples(
            batch_size=1,
            max_batches=10,
            dry_run=False,
        )
        assert repeat_report["eligible"] == repeat_report["requalified"] == 0
    finally:
        await close_db()
