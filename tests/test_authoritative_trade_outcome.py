from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import services.authoritative_trade_outcome as authoritative_trade_outcome
from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import OkxPositionHistory
from services.authoritative_trade_outcome import (
    AUTHORITATIVE_TRADE_LABEL_VERSION,
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
    build_authoritative_trade_outcome,
    load_authoritative_trade_outcomes,
)
from services.okx_execution_slippage import OKX_ROUND_TRIP_SLIPPAGE_SOURCE
from services.okx_position_history_store import load_okx_position_history_records
from services.training_data_quality import annotate_training_payload


@pytest.mark.asyncio
async def test_stale_compact_outcomes_return_immediately_and_refresh_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class PendingTask:
        def done(self) -> bool:
            return False

        def add_done_callback(self, _callback: object) -> None:
            return None

    def create_task(coroutine: object) -> PendingTask:
        coroutine.close()  # type: ignore[attr-defined]
        created.append(coroutine)
        return PendingTask()

    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test")
    monkeypatch.setattr(authoritative_trade_outcome.asyncio, "create_task", create_task)
    monkeypatch.setattr(
        authoritative_trade_outcome,
        "_compact_outcome_cache",
        (
            time.monotonic()
            - authoritative_trade_outcome._COMPACT_OUTCOME_CACHE_TTL_SECONDS
            - 1,
            [{"outcome_id": "cached", "execution_mode": "paper"}],
        ),
    )
    monkeypatch.setattr(
        authoritative_trade_outcome,
        "_compact_outcome_refresh_task",
        None,
    )

    first = await load_authoritative_trade_outcomes(mode="paper", compact=True)
    second = await load_authoritative_trade_outcomes(mode="paper", compact=True)

    assert first == second == [{"outcome_id": "cached", "execution_mode": "paper"}]
    assert len(created) == 1


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
        "entry_order_id": "entry-icp",
        "close_order_id": "close-icp",
        "linked_order_ids": ["entry-icp", "close-icp"],
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "ICP/USDT",
        "side": "short",
        "close_status": "full",
        "entry_price": 2.167,
        "close_price": 2.285,
        "quantity": 60194.1,
        "quantity_unit": "contracts",
        "notional": 1304.5,
        "notional_source": "okx_entry_fill_base_quantity_and_average_price",
        "realized_pnl": -71.83582759,
        "gross_pnl": -70.0,
        "entry_fee": 0.5,
        "close_fee": 0.7,
        "entry_fee_source": "okx_fills_history",
        "close_fee_source": "okx_fills_history",
        "funding_fee": -0.63582759,
        "liquidation_penalty": 0.0,
        "holding_minutes": 292.4,
        "planned_stop_loss_price": 2.21,
        "stop_loss_fill_confirmed": True,
        "slippage": 3.393665,
        "slippage_source": OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
        "entry_execution_slippage_usdt": 18.0,
        "close_execution_slippage_usdt": 26.270858925,
        "execution_slippage_usdt": 44.270858925,
        "trigger_to_first_fill_ms": 1060.0,
        "execution_actual_over_budget_loss_usdt": 14.0,
        "outcome": "loss",
        "pnl_source": "okx_position_history_realized_pnl",
        "settlement_source": "okx_position_history_realized_pnl",
        "funding_fee_source": "okx_positions_history.fundingFee",
        "decision_authority": "model",
        "net_return_after_all_cost_pct": -71.83582759 / 1304.5 * 100.0,
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
    without_reflection = build_authoritative_trade_outcome(
        _sample(), shadow_rows=[shadow]
    )

    assert first["outcome_version"] == AUTHORITATIVE_TRADE_OUTCOME_VERSION
    assert first["outcome_id"] == second["outcome_id"]
    assert first["outcome_fingerprint"] == second["outcome_fingerprint"]
    assert first["outcome_fingerprint"] == without_reflection["outcome_fingerprint"]
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
    assert attribution["execution_slippage"]["contribution_usdt"] == pytest.approx(
        -44.270858925
    )


def test_missing_optional_reflection_does_not_block_authoritative_label() -> None:
    outcome = build_authoritative_trade_outcome(_sample())
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[outcome],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert outcome["outcome_complete"] is True
    assert outcome["reflection_status"] == "pending_optional"
    assert "missing_trade_reflection_link" not in outcome["outcome_evidence_gaps"]
    assert len(payload["trade_samples"]) == 1
    record = payload["authoritative_outcome_manifest"]["records"][0]
    assert record["training_status"] == "included"


def test_demo_and_live_share_one_fee_after_label_schema() -> None:
    paper = build_authoritative_trade_outcome(_sample(execution_mode="simulation"))
    live = build_authoritative_trade_outcome(
        _sample(
            execution_mode="live",
            lifecycle_key="live|ICP-USDT-SWAP|pos-live|short|1",
            okx_pos_id="pos-live",
        )
    )

    paper_label = paper["training_label_contract"]
    live_label = live["training_label_contract"]
    assert paper["execution_mode"] == "paper"
    assert live["execution_mode"] == "live"
    assert paper_label["version"] == AUTHORITATIVE_TRADE_LABEL_VERSION
    assert set(paper_label) == set(live_label)
    assert paper_label["net_return_after_all_cost_pct"] == live_label[
        "net_return_after_all_cost_pct"
    ]
    assert paper_label["realized_net_pnl_usdt"] == live_label[
        "realized_net_pnl_usdt"
    ]


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


def test_missing_fee_after_label_contract_is_quarantined() -> None:
    outcome = build_authoritative_trade_outcome(_sample())
    outcome.pop("training_label_contract")

    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[outcome],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert payload["trade_samples"] == []
    record = payload["authoritative_outcome_manifest"]["records"][0]
    assert record["training_status"] == "excluded"


@pytest.mark.asyncio
async def test_authoritative_loader_deduplicates_legacy_and_canonical_history_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'authoritative-dedupe.db').as_posix()}",
    )
    await init_db()
    raw_row = {
        "instId": "ICP-USDT-SWAP",
        "posId": "dedupe-pos",
        "posSide": "short",
        "type": "2",
        "cTime": "1782579311213",
        "uTime": "1782580080851",
        "openAvgPx": "2.1",
        "closeAvgPx": "2.0",
        "openMaxPos": "10",
        "closeTotalPos": "10",
        "realizedPnl": "0.9",
        "pnl": "1.0",
        "fee": "-0.1",
        "fundingFee": "0",
        "ctVal": "1",
        "ctMult": "1",
        "lotSz": "1",
    }
    opened_at = datetime.fromtimestamp(1782579311213 / 1000, tz=UTC)
    closed_at = datetime.fromtimestamp(1782580080851 / 1000, tz=UTC)
    try:
        async with get_session_ctx() as session:
            common = {
                "mode": "paper",
                "inst_id": "ICP-USDT-SWAP",
                "symbol": "ICP/USDT",
                "pos_id": "dedupe-pos",
                "pos_side": "short",
                "side": "short",
                "close_type": "2",
                "close_status": "full",
                "opened_at": opened_at,
                "updated_at_okx": closed_at,
                "open_avg_px": 2.1,
                "close_avg_px": 2.0,
                "open_max_pos": 10.0,
                "close_total_pos": 10.0,
                "realized_pnl": 0.9,
                "pnl": 1.0,
                "fee": -0.1,
                "funding_fee": 0.0,
                "source": "okx_position_history_settlement",
                "raw_row": raw_row,
                "sync_status": "synced",
            }
            session.add_all(
                [
                    OkxPositionHistory(
                        **common,
                        row_identity=(
                            "paper|ICP-USDT-SWAP|dedupe-pos|short|2|"
                            "1782579311213|1782580080851|10|10"
                        ),
                    ),
                    OkxPositionHistory(
                        **common,
                        row_identity=(
                            "paper|ICP-USDT-SWAP|dedupe-pos|short|1782579311213"
                        ),
                    ),
                ]
            )

        outcomes = await load_authoritative_trade_outcomes(mode="paper")

        assert len(outcomes) == 1
        assert outcomes[0]["lifecycle_key"] == (
            "paper|ICP-USDT-SWAP|dedupe-pos|short|1782579311213"
        )
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_history_loader_filters_evaluation_window_in_sql(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'history-window.db').as_posix()}",
    )
    await init_db()
    cutoff = datetime(2026, 8, 18, 12, tzinfo=UTC)

    def history(
        *,
        pos_id: str,
        opened_at: datetime,
        updated_at_okx: datetime | None,
    ) -> OkxPositionHistory:
        opened_ms = int(opened_at.timestamp() * 1000)
        updated_ms = int(updated_at_okx.timestamp() * 1000) if updated_at_okx else 0
        raw_row = {
            "instId": "BTC-USDT-SWAP",
            "posId": pos_id,
            "posSide": "long",
            "type": "2",
            "cTime": str(opened_ms),
            "uTime": str(updated_ms) if updated_ms else "",
            "openMaxPos": "1",
            "closeTotalPos": "1",
        }
        return OkxPositionHistory(
            mode="paper",
            row_identity=f"paper|BTC-USDT-SWAP|{pos_id}|long|{opened_ms}",
            inst_id="BTC-USDT-SWAP",
            symbol="BTC/USDT",
            pos_id=pos_id,
            pos_side="long",
            side="long",
            opened_at=opened_at,
            updated_at_okx=updated_at_okx,
            raw_row=raw_row,
        )

    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    history(
                        pos_id="old",
                        opened_at=cutoff - timedelta(days=20),
                        updated_at_okx=cutoff - timedelta(days=10),
                    ),
                    history(
                        pos_id="recent-update",
                        opened_at=cutoff - timedelta(days=20),
                        updated_at_okx=cutoff + timedelta(hours=1),
                    ),
                    history(
                        pos_id="recent-open-no-update",
                        opened_at=cutoff + timedelta(hours=2),
                        updated_at_okx=None,
                    ),
                ]
            )

        async with get_session_ctx() as session:
            records = await load_okx_position_history_records(
                session,
                mode="paper",
                since=cutoff,
            )

        assert {record.pos_id for record in records} == {
            "recent-update",
            "recent-open-no-update",
        }
    finally:
        await close_db()
