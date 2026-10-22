from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import services.shadow_backtest_service as shadow_backtest_module
from ai_brain.base_model import Action, DecisionOutput
from core.market_facts import build_market_fact, verify_market_fact_path
from core.training_contracts import (
    SHADOW_FEE_AFTER_LABEL_VERSION,
    SHADOW_LABEL_VERSION,
    shadow_label_contract_reasons,
)
from services.shadow_backtest_service import (
    SHADOW_BACKTEST_HORIZONS_MINUTES,
    SHADOW_SAMPLE_RECOVERY_MAX_BATCH,
    ShadowBacktestService,
    _bounded_task_gather,
    shadow_fee_after_outcome,
    shadow_path_labels,
    side_label,
)


def test_market_label_horizons_match_the_versioned_training_contract() -> None:
    assert SHADOW_BACKTEST_HORIZONS_MINUTES == (5, 15, 60, 240)


def test_shadow_path_labels_capture_mfe_mae_and_stop_events() -> None:
    labels = shadow_path_labels(
        entry_price=100.0,
        price_path={"path_low": 97.0, "path_high": 102.0},
        stop_loss_fraction=0.025,
    )

    assert labels == {
        "long_mfe_pct": 2.0,
        "long_mae_pct": 3.0,
        "short_mfe_pct": 3.0,
        "short_mae_pct": 2.0,
        "long_stop_loss_triggered": True,
        "short_stop_loss_triggered": False,
        "long_take_profit_triggered": False,
        "short_take_profit_triggered": False,
        "long_first_touch": "none",
        "short_first_touch": "none",
    }


def test_shadow_path_labels_preserve_first_touch_uncertainty() -> None:
    labels = shadow_path_labels(
        entry_price=100.0,
        price_path={
            "path_low": 97.0,
            "path_high": 103.0,
            "_ordered_bar_ranges": [
                {"open_time_ms": 1, "low": 99.0, "high": 101.0},
                {"open_time_ms": 2, "low": 97.0, "high": 103.0},
            ],
        },
        stop_loss_fraction=0.02,
        take_profit_fraction=0.02,
    )

    assert labels["long_stop_loss_triggered"] is True
    assert labels["long_take_profit_triggered"] is True
    assert labels["long_first_touch"] == "path_uncertain"
    assert labels["short_stop_loss_triggered"] is True
    assert labels["short_take_profit_triggered"] is True
    assert labels["short_first_touch"] == "path_uncertain"


class _SessionCtx:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeRepo:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.due_rows: list[Any] = []
        self.completed: list[dict[str, Any]] = []

    async def create_shadow_backtest(self, data: dict[str, Any]) -> None:
        self.created.append(data)

    async def get_due_shadow_backtests(
        self, limit: int = 200, *, exclude_ids: list[int] | None = None,
    ) -> list[Any]:
        assert limit == 200
        return [row for row in self.due_rows if row.id not in (exclude_ids or [])]

    async def complete_shadow_backtest(self, row: Any, **data: Any) -> None:
        for key, value in data.items():
            setattr(row, key, value)
        self.completed.append(data)

_ENTRY_TIMESTAMP_MS = 1_783_990_800_000
_RESULT_TIMESTAMP_MS = _ENTRY_TIMESTAMP_MS + 60 * 60_000


def _market_fact(
    symbol: str,
    price: float,
    timestamp_ms: int,
    *,
    notional_24h_usdt: float = 100_000.0,
) -> dict[str, Any]:
    normalized = symbol.upper()
    base = normalized.split("/")[0]
    return build_market_fact(
        normalized,
        {
            "symbol": normalized,
            "inst_id": f"{base}-USDT-SWAP",
            "inst_type": "SWAP",
            "source": "websocket",
            "source_endpoint": "okx_ws_public",
            "source_channel": "tickers",
            "timestamp": timestamp_ms,
            "last_price": price,
            "bid": price * 0.9999,
            "ask": price * 1.0001,
            "notional_24h_usdt": notional_24h_usdt,
            "volume_24h_contracts": 100_000.0,
            "volume_24h_base": 100_000.0,
            "orderbook_bid_depth": 100_000.0,
            "orderbook_ask_depth": 100_000.0,
        },
        contract_spec={
            "instId": f"{base}-USDT-SWAP",
            "instType": "SWAP",
            "uly": f"{base}-USDT",
            "instFamily": f"{base}-USDT",
            "ctType": "linear",
            "ctVal": "1",
            "ctMult": "1",
            "ctValCcy": base,
            "settleCcy": "USDT",
            "lotSz": "0.1",
            "minSz": "0.1",
            "tickSz": "0.0001",
            "state": "live",
        },
        received_at=datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC),
    )


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _service(
    repo: _FakeRepo,
    latest_price: float = 101.0,
    execution_cost_facts_provider=None,
) -> ShadowBacktestService:
    async def latest(_symbol: str) -> float:
        return latest_price

    for row in repo.due_rows:
        if getattr(row, "due_at", None) is None:
            row.due_at = datetime.fromtimestamp(_RESULT_TIMESTAMP_MS / 1000.0, tz=UTC)
        snapshot = dict(getattr(row, "feature_snapshot", None) or {})
        snapshot.setdefault(
            "market_fact",
            _market_fact(
                str(row.symbol),
                float(row.entry_price),
                _ENTRY_TIMESTAMP_MS,
            ),
        )
        row.feature_snapshot = snapshot

    async def latest_fact(symbol: str) -> dict[str, Any]:
        return _market_fact(symbol, latest_price, _RESULT_TIMESTAMP_MS)

    async def price_path(
        entry_fact: dict[str, Any],
        result_fact: dict[str, Any],
    ) -> dict[str, Any]:
        entry_price = float(entry_fact["prices"]["last"])
        result_price = float(result_fact["prices"]["last"])
        low = min(entry_price, result_price) * 0.999
        high = max(entry_price, result_price) * 1.001
        bars = [
            [timestamp, entry_price, high, low, result_price, 1000.0]
            for timestamp in range(
                _ENTRY_TIMESTAMP_MS,
                _RESULT_TIMESTAMP_MS + 1,
                60_000,
            )
        ]
        return verify_market_fact_path(entry_fact, result_fact, bars)

    return ShadowBacktestService(
        latest_price_provider=latest,
        symbol_normalizer=lambda symbol: str(symbol or "").upper(),
        float_parser=_float,
        session_factory=_SessionCtx,
        repository_factory=lambda _session: repo,
        execution_cost_facts_provider=execution_cost_facts_provider,
        latest_market_fact_provider=latest_fact,
        price_path_provider=price_path,
        horizons_minutes=(10, 30),
    )


def _decision() -> DecisionOutput:
    return DecisionOutput(
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.7,
        reasoning="test shadow backtest",
        position_size_pct=0.05,
        model_name="ensemble_trader",
        feature_snapshot={
            "current_price": 100.0,
            "adx_14": 28.0,
            "market_fact": _market_fact("BTC/USDT", 100.0, _ENTRY_TIMESTAMP_MS),
        },
        raw_response={"reason": "test"},
    )


def test_shadow_fee_after_label_uses_funding_when_it_changes_the_best_side() -> None:
    row = SimpleNamespace(
        horizon_minutes=480,
        feature_snapshot={
            "bid": 99.99,
            "ask": 100.01,
            "orderbook_bid_depth": 100_000.0,
            "orderbook_ask_depth": 100_000.0,
            "taker_fee_rate": 0.0004,
            "funding_rate": 0.002,
            "funding_data_available": True,
            "funding_interval_minutes": 480.0,
        },
    )

    outcome = shadow_fee_after_outcome(
        row,
        long_return=0.001,
        short_return=-0.001,
    )

    assert outcome["cost_complete"] is True
    assert outcome["long_net_return_after_all_cost_pct"] < 0.0
    assert outcome["short_net_return_after_all_cost_pct"] > 0.0
    scenarios = outcome["leverage_counterfactuals"]
    assert [item["leverage"] for item in scenarios] == [1, 2, 3, 5, 10]
    assert scenarios[-1]["short_fee_after_margin_return_pct"] == pytest.approx(
        outcome["short_net_return_after_all_cost_pct"] * 10
    )
    assert all(item["creates_order"] is False for item in scenarios)
    assert outcome["leverage_counterfactual_policy"]["creates_order"] is False


@pytest.mark.asyncio
async def test_shadow_backtest_service_creates_pending_horizons() -> None:
    repo = _FakeRepo()
    feature_vector = SimpleNamespace(current_price=100.0, close=99.0)

    await _service(repo).create(123, _decision(), feature_vector, "paper")

    assert [item["horizon_minutes"] for item in repo.created] == [10, 30]
    assert all(item["status"] == "pending" for item in repo.created)
    assert repo.created[0]["decision_id"] == 123
    assert repo.created[0]["decision_action"] == "long"
    assert repo.created[0]["entry_price"] == 100.0
    assert repo.created[0]["feature_snapshot"]["market_fact"]["fact_id"].startswith(
        "sha256:"
    )
    assert repo.created[0]["feature_snapshot"]["training_market_fact_contract"][
        "status"
    ] == "quarantined"
    compact_contract = repo.created[0]["feature_snapshot"][
        "training_market_fact_contract"
    ]
    assert len(json.dumps(compact_contract).encode("utf-8")) < 2048


@pytest.mark.asyncio
async def test_shadow_backtest_service_recovers_missing_horizons_from_persisted_decision() -> None:
    repo = _FakeRepo()
    finder_kwargs: dict[str, Any] = {}
    row = SimpleNamespace(
        id=456,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="long",
        confidence=0.8,
        reasoning="durable recovery",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        is_paper=True,
        created_at=datetime.now(UTC),
        raw_llm_response={"reason": "recovery"},
        feature_snapshot={"current_price": 100.0},
    )

    async def finder(**_kwargs: Any) -> list[Any]:
        finder_kwargs.update(_kwargs)
        return [row]

    repo.get_market_decisions_missing_shadow_samples = finder  # type: ignore[attr-defined]
    result = await _service(repo).recover_missing_market_samples(limit=10)

    assert result == {"scanned": 1, "recovered": 1, "failed": 0}
    assert finder_kwargs["limit"] == SHADOW_SAMPLE_RECOVERY_MAX_BATCH
    assert [item["horizon_minutes"] for item in repo.created] == [10, 30]
    assert all(item["decision_id"] == 456 for item in repo.created)


@pytest.mark.asyncio
async def test_shadow_backtest_recovery_defers_a_timed_out_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=457,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="hold",
        confidence=0.8,
        reasoning="slow recovery",
        position_size_pct=0.05,
        suggested_leverage=1.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        is_paper=True,
        created_at=datetime.now(UTC),
        raw_llm_response={},
        feature_snapshot={"current_price": 100.0},
    )

    async def finder(**_kwargs: Any) -> list[Any]:
        return [row]

    repo.get_market_decisions_missing_shadow_samples = finder  # type: ignore[attr-defined]
    service = _service(repo)

    async def hanging_create(*_args: Any, **_kwargs: Any) -> bool:
        await asyncio.sleep(1.0)
        return True

    monkeypatch.setattr(ShadowBacktestService, "create", hanging_create)
    monkeypatch.setattr(
        shadow_backtest_module,
        "SHADOW_SAMPLE_RECOVERY_PER_DECISION_BUDGET_SECONDS",
        0.01,
    )

    result = await service.recover_missing_market_samples(limit=10)

    assert result == {"scanned": 1, "recovered": 0, "failed": 1}


@pytest.mark.asyncio
async def test_shadow_backtest_recovery_defers_when_remaining_budget_is_too_small(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=458,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="hold",
        confidence=0.8,
        reasoning="deferred recovery",
        position_size_pct=0.05,
        suggested_leverage=1.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        is_paper=True,
        created_at=datetime.now(UTC),
        raw_llm_response={},
        feature_snapshot={"current_price": 100.0},
    )

    async def finder(**_kwargs: Any) -> list[Any]:
        return [row]

    repo.get_market_decisions_missing_shadow_samples = finder  # type: ignore[attr-defined]
    monkeypatch.setattr(
        shadow_backtest_module,
        "SHADOW_SAMPLE_RECOVERY_TOTAL_BUDGET_SECONDS",
        0.005,
    )
    monkeypatch.setattr(
        shadow_backtest_module,
        "SHADOW_SAMPLE_RECOVERY_PER_DECISION_BUDGET_SECONDS",
        0.01,
    )

    result = await _service(repo).recover_missing_market_samples(limit=10)

    assert result == {"scanned": 0, "recovered": 0, "failed": 0}


@pytest.mark.asyncio
async def test_shadow_backtest_persists_explicit_model_side_for_rules_canary() -> None:
    repo = _FakeRepo()
    feature_vector = SimpleNamespace(current_price=100.0, close=99.0)
    decision = _decision()
    decision.raw_response = {
        "production_trade_gate": {
            "mode": "live_rules_canary",
            "decision_authority": "rules",
            "model_can_influence": False,
        },
        "model_shadow_decision": {
            "action": "short",
            "observation_only": True,
        },
    }

    await _service(repo).create(123, decision, feature_vector, "live")

    snapshot = repo.created[0]["feature_snapshot"]
    assert snapshot["model_shadow_action"] == "short"
    assert snapshot["model_shadow_action_source"] == "explicit_model_shadow_decision"


@pytest.mark.asyncio
async def test_shadow_backtest_service_captures_local_ai_tools_shadow_evidence() -> None:
    repo = _FakeRepo()
    feature_vector = SimpleNamespace(current_price=100.0, close=99.0)
    local_ai_tools_context = {
        "status": "completed",
        "time_series_prediction": {
            "available": True,
            "model": "local-timeseries-ensemble-v1",
            "expected_return_pct": 0.11,
            "timesfm_shadow_expected_return_pct": 0.42,
            "timesfm_shadow_side": "long",
            "specialist_inference_active": True,
            "professional_model_shadow": {
                "kind": "timeseries",
                "actual_inference": True,
                "baseline_response": True,
                "shadow_result": {
                    "model": "timesfm-2.5-shadow-challenger",
                    "actual_inference": True,
                    "expected_return_pct": 0.42,
                    "best_side": "long",
                    "confidence": 0.73,
                    "raw_predictions": list(range(200)),
                },
            },
            "raw_huge_payload": list(range(1000)),
        },
        "sentiment_analysis": {
            "available": True,
            "model": "finbert-shadow-ensemble-v1",
            "specialist_inference_active": True,
            "professional_model_shadow": {
                "kind": "sentiment",
                "actual_inference": True,
                "baseline_response": False,
                "predictions": {
                    "sentiment_primary": {
                        "available": True,
                        "score": 0.61,
                        "label": "positive",
                        "text_count": 3,
                    },
                    "sentiment_challenger": {
                        "available": False,
                        "reason": "no_text_inputs",
                    },
                },
            },
        },
    }

    await _service(repo).create(
        123,
        _decision(),
        feature_vector,
        "paper",
        local_ai_tools_context=local_ai_tools_context,
    )

    snapshot = repo.created[0]["feature_snapshot"]
    shadow = snapshot["local_ai_tools_shadow"]
    timeseries = shadow["time_series_prediction"]
    sentiment = shadow["sentiment_analysis"]

    assert shadow["status"] == "completed"
    assert timeseries["timesfm_shadow_expected_return_pct"] == 0.42
    assert timeseries["timesfm_shadow_side"] == "long"
    assert timeseries["professional_model_shadow"]["shadow_result"] == {
        "model": "timesfm-2.5-shadow-challenger",
        "actual_inference": True,
        "expected_return_pct": 0.42,
        "best_side": "long",
        "confidence": 0.73,
    }
    assert sentiment["professional_model_shadow"]["kind"] == "sentiment"
    assert sentiment["professional_model_shadow"]["predictions"] == {
        "sentiment_primary": {
            "available": True,
            "score": 0.61,
            "label": "positive",
            "text_count": 3,
        },
        "sentiment_challenger": {
            "available": False,
            "reason": "no_text_inputs",
        },
    }
    assert "raw_huge_payload" not in timeseries
    assert "raw_predictions" not in timeseries["professional_model_shadow"]["shadow_result"]


@pytest.mark.asyncio
async def test_shadow_backtest_cost_incomplete_move_keeps_gross_label_only() -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=7,
        decision_id=123,
        model_name="ensemble_trader",
        symbol="btc/usdt",
        decision_action="hold",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={
            "adx_14": 28.0,
            "volume_ratio": 1.4,
            "returns_5": 0.004,
            "orderbook_imbalance": 0.15,
        },
    )
    repo.due_rows = [row]

    await _service(repo, latest_price=101.0).update_due()

    assert repo.completed == [
        {
            "actual_price": 101.0,
            "long_return_pct": pytest.approx(1.0),
            "short_return_pct": pytest.approx(-1.0),
            "best_action": "hold",
            "missed_opportunity": False,
            "note": "",
        }
    ]


@pytest.mark.asyncio
async def test_shadow_backtest_records_fee_after_training_label_only() -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=8,
        decision_id=123,
        model_name="ensemble_trader",
        symbol="btc/usdt",
        decision_action="hold",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={
            "adx_14": 28.0,
            "volume_ratio": 1.4,
            "returns_5": 0.004,
            "orderbook_imbalance": 0.15,
            "bid": 99.99,
            "ask": 100.01,
            "orderbook_bid_depth": 100_000.0,
            "orderbook_ask_depth": 100_000.0,
            "taker_fee_rate": 0.0005,
            "funding_rate": 0.0,
            "funding_data_available": True,
            "funding_interval_minutes": 480.0,
        },
    )
    repo.due_rows = [row]

    await _service(repo, latest_price=101.0).update_due()

    label_contract = row.feature_snapshot["training_label_contract"]
    leverage_evidence = row.feature_snapshot["training_leverage_counterfactuals"]
    assert label_contract["version"] == SHADOW_LABEL_VERSION
    assert label_contract["decision_id"] == 123
    assert label_contract["horizon_minutes"] == 10
    assert label_contract["fee_after_label_version"] == SHADOW_FEE_AFTER_LABEL_VERSION
    assert label_contract["fee_after_complete"] is True
    assert label_contract["long_net_return_after_all_cost_pct"] < label_contract[
        "long_return_pct"
    ]
    assert shadow_label_contract_reasons(
        label_contract,
        decision_id=123,
        horizon_minutes=10,
    ) == []
    assert leverage_evidence["scenario_count"] == 5
    assert leverage_evidence["creates_order"] is False
    assert leverage_evidence["leverage_10x_long_fee_after_margin_return_pct"] == (
        pytest.approx(label_contract["long_net_return_after_all_cost_pct"] * 10)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_fee", [None, 0.0004])
async def test_due_shadow_observes_current_fee_without_replacing_decision_facts(
    entry_fee: float | None,
) -> None:
    repo = _FakeRepo()
    repo.due_rows = [
        SimpleNamespace(
            id=81,
            decision_id=987,
            model_name="ensemble_trader",
            execution_mode="paper",
            symbol="BTC/USDT",
            decision_action="hold",
            entry_price=100.0,
            horizon_minutes=10,
            status="pending",
            note="",
            feature_snapshot={
                "bid": 99.99,
                "ask": 100.01,
                "orderbook_bid_depth": 100_000.0,
                "orderbook_ask_depth": 100_000.0,
                "funding_rate": 0.0,
                "funding_data_available": True,
                "funding_interval_minutes": 480.0,
            },
        )
    ]
    original_observed_at = "2026-07-13T07:50:00+00:00"
    if entry_fee is not None:
        repo.due_rows[0].feature_snapshot.update({
            "taker_fee_rate": entry_fee,
            "fee_rate_source": "original_account_trade_fee",
            "fee_rate_observed_at": original_observed_at,
        })
    calls: list[str] = []

    async def fee_facts(mode: str) -> dict[str, Any]:
        calls.append(mode)
        return {
            "taker_fee_rate": 0.0005,
            "entry_fee_rate": 0.0005,
            "exit_fee_rate": 0.0005,
            "fee_rate_source": "okx_account_trade_fee.takerU",
            "fee_rate_observed_at": "2026-07-13T08:00:00+00:00",
        }

    await _service(
        repo,
        latest_price=101.0,
        execution_cost_facts_provider=fee_facts,
    ).update_due()

    assert calls == ["paper"]
    snapshot = repo.due_rows[0].feature_snapshot
    settlement = snapshot["shadow_settlement_cost_observation"]
    assert settlement["taker_fee_rate"] == pytest.approx(0.0005)
    assert settlement["fee_rate_observed_at"] == "2026-07-13T08:00:00+00:00"
    assert settlement["used_as_decision_time_features"] is False
    if entry_fee is None:
        assert "taker_fee_rate" not in snapshot
        assert "fee_rate_observed_at" not in snapshot
        assert snapshot["training_label_contract"].get("fee_after_complete") is not True
        assert repo.due_rows[0].status == "quarantined"
        assert "cost_incomplete:fee_rate_missing" in repo.due_rows[0].note
    else:
        assert snapshot["taker_fee_rate"] == pytest.approx(entry_fee)
        assert snapshot["fee_rate_source"] == "original_account_trade_fee"
        assert snapshot["fee_rate_observed_at"] == original_observed_at
        assert "entry_fee_rate" not in snapshot
        assert "exit_fee_rate" not in snapshot
        assert snapshot["training_label_contract"]["fee_after_complete"] is True


@pytest.mark.asyncio
async def test_shadow_backtest_price_collection_does_not_hold_database_session(
) -> None:
    repo = _FakeRepo()
    repo.due_rows = [
        SimpleNamespace(
            id=70,
            decision_id=123,
            model_name="ensemble_trader",
            symbol="BTC/USDT",
            decision_action="hold",
            entry_price=100.0,
            horizon_minutes=10,
            status="pending",
            note="",
            feature_snapshot={},
            due_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    ]

    class TrackingSessionCtx:
        active = 0
        entries = 0

        async def __aenter__(self) -> object:
            type(self).active += 1
            type(self).entries += 1
            return object()

        async def __aexit__(self, *_args: object) -> None:
            type(self).active -= 1

    async def latest(_symbol: str) -> float:
        assert TrackingSessionCtx.active == 0
        return 101.0

    service = ShadowBacktestService(
        latest_price_provider=latest,
        symbol_normalizer=lambda symbol: str(symbol or "").upper(),
        float_parser=_float,
        session_factory=TrackingSessionCtx,
        repository_factory=lambda _session: repo,
    )

    assert await service.update_due() == 1
    assert TrackingSessionCtx.entries == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("source_lag", [-1, 0, 61])
async def test_shadow_source_clock_gates_maturity_and_late_labels(source_lag) -> None:
    repo = _FakeRepo()
    result_at = datetime.fromtimestamp(_RESULT_TIMESTAMP_MS / 1000.0, tz=UTC)
    row = SimpleNamespace(
        id=71, decision_id=123, symbol="BTC/USDT", decision_action="hold",
        entry_price=100.0, horizon_minutes=10, status="pending", note="",
        due_at=result_at - timedelta(seconds=source_lag),
        feature_snapshot={
            "bid": 99.99, "ask": 100.01, "taker_fee_rate": 0.0005,
            "funding_rate": 0.0, "funding_interval_minutes": 480,
            "orderbook_bid_depth": 100_000, "orderbook_ask_depth": 100_000,
        },
    )
    repo.due_rows = [row]
    completed = await _service(repo).update_due()
    if source_lag < 0:
        assert completed == 0
        assert row.status == "pending"
        assert repo.completed == []
    else:
        assert completed == 1
        label = row.feature_snapshot["training_label_contract"]
        assert label["scheduled_label_at"] == row.due_at.isoformat()
        assert label["label_timestamp"] == result_at.isoformat()
        if source_lag > 60:
            assert row.status == "quarantined"
            assert "result_source_lag_exceeds_one_minute" in row.note


@pytest.mark.asyncio
async def test_shadow_backtest_service_does_not_apply_fixed_price_range_tolerance(
) -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=8,
        decision_id=124,
        model_name="ensemble_trader",
        symbol="PROS/USDT",
        decision_action="long",
        entry_price=0.3902,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={
            "current_price": 0.3902,
            "low_24h": 0.5491,
            "high_24h": 0.5707,
            "spread_pct": 0.03,
            "round_trip_fee_pct": 0.08,
            "funding_rate": 0.0,
            "funding_interval_minutes": 480.0,
        },
    )
    repo.due_rows = [row]

    await _service(repo, latest_price=0.3910).update_due()

    assert row.status != "quarantined"
    assert "price_outside_24h_range" not in row.note
    assert repo.completed[0]["actual_price"] == pytest.approx(0.3910)


@pytest.mark.asyncio
async def test_shadow_backtest_quarantines_zero_turnover_robo_entry_fact(
) -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=88,
        decision_id=79757,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ROBO/USDT",
        decision_action="short",
        entry_price=0.10834,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={
            "market_fact": _market_fact(
                "ROBO/USDT",
                0.10834,
                _ENTRY_TIMESTAMP_MS,
                notional_24h_usdt=0.0,
            ),
            "bid": 0.01293,
            "ask": 0.01295,
            "orderbook_bid_depth": 100_000.0,
            "orderbook_ask_depth": 100_000.0,
            "taker_fee_rate": 0.0005,
            "funding_rate": 0.0,
            "funding_data_available": True,
            "funding_interval_minutes": 480.0,
        },
    )
    repo.due_rows = [row]

    await _service(repo, latest_price=0.01294).update_due()

    assert row.status == "quarantined"
    assert "entry:zero_notional_turnover" in row.note


@pytest.mark.asyncio
async def test_shadow_backtest_market_fact_error_keeps_authoritative_sample_pending() -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=90,
        decision_id=900,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        decision_action="hold",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={},
    )
    repo.due_rows = [row]
    service = _service(repo, latest_price=101.5)

    async def broken_market_fact(_symbol: str) -> dict[str, Any]:
        raise TimeoutError("market fact timeout")

    service.latest_market_fact_provider = broken_market_fact

    assert await service.update_due() == 0
    assert repo.completed == []
    assert row.status == "pending"


@pytest.mark.asyncio
async def test_shadow_backtest_incomplete_authoritative_fact_stays_pending() -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=901,
        decision_id=9001,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        decision_action="hold",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={},
    )
    repo.due_rows = [row]
    service = _service(repo, latest_price=101.5)

    async def incomplete_market_fact(symbol: str) -> dict[str, Any]:
        return _market_fact(
            symbol,
            101.5,
            _RESULT_TIMESTAMP_MS,
            notional_24h_usdt=0.0,
        )

    service.latest_market_fact_provider = incomplete_market_fact

    assert await service.update_due() == 0
    assert repo.completed == []
    assert row.status == "pending"


@pytest.mark.asyncio
async def test_shadow_backtest_incomplete_fact_sets_symbol_retry_cooldown() -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=902,
        decision_id=9002,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        decision_action="hold",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={},
    )
    repo.due_rows = [row]
    service = _service(repo, latest_price=101.5)
    calls = 0

    async def incomplete_market_fact(symbol: str) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _market_fact(symbol, 101.5, _RESULT_TIMESTAMP_MS, notional_24h_usdt=0.0)

    service.latest_market_fact_provider = incomplete_market_fact

    assert await service.update_due() == 0
    assert await service.update_due() == 0
    assert calls == 1
    assert "BTC/USDT" in service._market_fact_retry_after


@pytest.mark.asyncio
async def test_shadow_backtest_market_facts_are_deduplicated_per_symbol() -> None:
    repo = _FakeRepo()
    repo.due_rows = [
        SimpleNamespace(
            id=row_id,
            decision_id=1000 + row_id,
            model_name="ensemble_trader",
            symbol="BTC/USDT",
            decision_action="hold",
            entry_price=100.0,
            horizon_minutes=10,
            status="pending",
            note="",
            feature_snapshot={},
        )
        for row_id in (91, 92, 93)
    ]
    calls: list[str] = []
    service = _service(repo, latest_price=101.0)

    async def market_fact(symbol: str) -> dict[str, Any]:
        calls.append(symbol)
        await asyncio.sleep(0)
        return _market_fact(symbol, 101.0, _RESULT_TIMESTAMP_MS)

    service.latest_market_fact_provider = market_fact

    assert await service.update_due() == 3
    assert calls == ["BTC/USDT"]
    assert len(repo.completed) == 3


@pytest.mark.asyncio
async def test_shadow_backtest_uses_historical_due_at_path_without_current_quote() -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=97,
        decision_id=997,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        decision_action="long",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        due_at=datetime.fromtimestamp(_RESULT_TIMESTAMP_MS / 1000.0, tz=UTC),
        feature_snapshot={
            "market_fact": _market_fact("BTC/USDT", 100.0, _ENTRY_TIMESTAMP_MS),
        },
    )
    repo.due_rows = [row]
    current_fact_calls = 0
    historical_calls: list[tuple[str, datetime, int]] = []

    async def current_fact(_symbol: str) -> dict[str, Any]:
        nonlocal current_fact_calls
        current_fact_calls += 1
        return _market_fact("BTC/USDT", 999.0, _RESULT_TIMESTAMP_MS)

    async def historical_path(
        symbol: str,
        _entry_fact: dict[str, Any],
        due_at: datetime,
        horizon: int,
    ) -> dict[str, Any]:
        historical_calls.append((symbol, due_at, horizon))
        entry_fact = _market_fact(symbol, 100.0, _ENTRY_TIMESTAMP_MS)
        result_fact = _market_fact(symbol, 101.0, _RESULT_TIMESTAMP_MS)
        result_fact["source_available_at"] = (
            due_at + timedelta(minutes=1)
        ).isoformat()
        return {
            "result_fact": result_fact,
            "price_path": {
                "status": "clean",
                "path_low": 99.0,
                "path_high": 101.0,
                "_ordered_bar_ranges": [],
            },
            "market_fact_contract": {
                "version": "2026-09-14",
                "status": "historical_ohlcv_only",
                "provenance": {"strategy_version": "historical-v1"},
                "entry_fact": entry_fact,
                "result_fact": result_fact,
            },
            "training_market_fact_contract": {
                "version": "2026-09-14",
                "status": "historical_ohlcv_only",
                "path_status": "clean",
            },
        }

    service = ShadowBacktestService(
        latest_price_provider=lambda _symbol: _completed_float(999.0),
        symbol_normalizer=lambda symbol: str(symbol or "").upper(),
        float_parser=_float,
        session_factory=_SessionCtx,
        repository_factory=lambda _session: repo,
        latest_market_fact_provider=current_fact,
        historical_market_path_provider=historical_path,
        horizons_minutes=(10,),
    )

    assert await service.update_due() == 1
    assert current_fact_calls == 0
    assert historical_calls == [
        ("BTC/USDT", datetime.fromtimestamp(_RESULT_TIMESTAMP_MS / 1000.0, tz=UTC), 10)
    ]
    assert repo.completed[0]["actual_price"] == pytest.approx(101.0)
    assert row.feature_snapshot["historical_market_path_only"] is True


async def _completed_float(value: float) -> float:
    return value


@pytest.mark.asyncio
async def test_unavailable_historical_rows_yield_queue_capacity_until_cooldown_expires():
    class QueueRepo(_FakeRepo):
        async def get_due_shadow_backtests(self, limit=1, *, exclude_ids=None):
            return [
                row for row in self.due_rows if row.id not in (exclude_ids or [])
            ][:limit]

    repo = QueueRepo()
    repo.due_rows = [
        SimpleNamespace(
            id=row_id, symbol="BTC/USDT", execution_mode="paper",
            due_at=datetime.fromtimestamp(_RESULT_TIMESTAMP_MS / 1000, tz=UTC),
            horizon_minutes=5, entry_price=100.0, status="pending", feature_snapshot={},
        )
        for row_id in (101, 102)
    ]
    calls = []

    async def unavailable(_symbol, _entry, _due, _horizon):
        calls.append(len(calls))
        return {}

    service = _service(repo)
    service.historical_market_path_provider = unavailable
    assert await service.update_due(limit=1) == 0
    assert set(service._historical_retry_after) == {101}
    assert await service.update_due(limit=1) == 0
    assert set(service._historical_retry_after) == {101, 102}
    assert len(calls) == 2
    assert await service.update_due(limit=1) == 0
    assert len(calls) == 2

    service._historical_retry_after[101] = 0
    assert await service.update_due(limit=1) == 0
    assert len(calls) == 3
    assert all(row.status == "pending" for row in repo.due_rows)
    assert repo.completed == []


@pytest.mark.asyncio
async def test_shadow_backtest_path_timeout_falls_back_to_endpoint_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=94,
        decision_id=994,
        model_name="ensemble_trader",
        symbol="ETH/USDT",
        decision_action="long",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={},
    )
    repo.due_rows = [row]
    service = _service(repo, latest_price=101.0)

    async def slow_path(
        _entry_fact: dict[str, Any],
        _result_fact: dict[str, Any],
    ) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {}

    monkeypatch.setattr(
        "services.shadow_backtest_service.SHADOW_RESULT_PATH_TIMEOUT_SECONDS",
        0.001,
    )
    service.price_path_provider = slow_path

    assert await service.update_due() == 1
    assert repo.completed[0]["actual_price"] == pytest.approx(101.0)
    assert row.feature_snapshot["market_fact_contract"]["assertions"][
        "same_contract_price_path_verified"
    ] is False


@pytest.mark.asyncio
async def test_shadow_backtest_market_fact_batch_timeout_leaves_rows_for_next_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=95,
        decision_id=995,
        model_name="ensemble_trader",
        symbol="SOL/USDT",
        decision_action="hold",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={},
    )
    repo.due_rows = [row]
    service = _service(repo, latest_price=101.0)

    async def slow_market_fact(_symbol: str) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return _market_fact("SOL/USDT", 101.0, _RESULT_TIMESTAMP_MS)

    monkeypatch.setattr(
        "services.shadow_backtest_service.SHADOW_RESULT_FACT_BATCH_TIMEOUT_SECONDS",
        0.001,
    )
    service.latest_market_fact_provider = slow_market_fact

    assert await service.update_due() == 0
    assert repo.completed == []
    assert row.status == "pending"


@pytest.mark.asyncio
async def test_shadow_backtest_batch_timeout_does_not_wait_for_stubborn_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo()
    row = SimpleNamespace(
        id=96,
        decision_id=996,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        decision_action="hold",
        entry_price=100.0,
        horizon_minutes=10,
        status="pending",
        note="",
        feature_snapshot={},
    )
    repo.due_rows = [row]
    service = _service(repo, latest_price=101.0)
    cancelled = asyncio.Event()

    async def stubborn_market_fact(_symbol: str) -> dict[str, Any]:
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            cancelled.set()
            return _market_fact("BTC/USDT", 101.0, _RESULT_TIMESTAMP_MS)
        raise AssertionError("provider should not finish normally")

    monkeypatch.setattr(
        "services.shadow_backtest_service.SHADOW_RESULT_FACT_BATCH_TIMEOUT_SECONDS",
        0.001,
    )
    service.latest_market_fact_provider = stubborn_market_fact

    started = asyncio.get_running_loop().time()
    assert await service.update_due() == 0
    assert asyncio.get_running_loop().time() - started < 0.2
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    assert repo.completed == []
    assert row.status == "pending"


@pytest.mark.asyncio
async def test_bounded_gather_cancels_provider_when_parent_is_cancelled() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def provider() -> None:
        started.set()
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(
        _bounded_task_gather((provider(),), budget_seconds=30.0)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_shadow_collection_stages_share_deadline_and_preserve_completed_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _FakeRepo()
    repo.due_rows = [
        SimpleNamespace(
            id=row_id,
            decision_id=1000 + row_id,
            model_name="ensemble_trader",
            symbol=symbol,
            decision_action="hold",
            entry_price=100.0,
            horizon_minutes=10,
            status="pending",
            note="",
            feature_snapshot={},
        )
        for row_id, symbol in ((101, "BTC/USDT"), (102, "ETH/USDT"))
    ]
    cost_finished = asyncio.Event()
    fact_finished = asyncio.Event()
    path_cancelled = asyncio.Event()

    async def cost_facts(_mode: str) -> dict[str, Any]:
        await asyncio.sleep(0.04)
        cost_finished.set()
        return {}

    service = _service(repo, execution_cost_facts_provider=cost_facts)
    original_path = service.price_path_provider
    assert original_path is not None

    async def market_fact(symbol: str) -> dict[str, Any]:
        assert cost_finished.is_set()
        await asyncio.sleep(0.04)
        fact_finished.set()
        return _market_fact(symbol, 101.0, _RESULT_TIMESTAMP_MS)

    async def path(
        entry: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        assert fact_finished.is_set()
        if entry["symbol"] == "ETH/USDT":
            try:
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                path_cancelled.set()
                raise
        return await original_path(entry, result)

    monkeypatch.setattr(
        "services.shadow_backtest_service.SHADOW_RESULT_COLLECTION_BUDGET_SECONDS",
        0.15,
    )
    service.latest_market_fact_provider = market_fact
    service.price_path_provider = path
    started = asyncio.get_running_loop().time()

    assert await asyncio.wait_for(service.update_due(), timeout=0.6) == 2
    assert asyncio.get_running_loop().time() - started < 0.6
    await asyncio.wait_for(path_cancelled.wait(), timeout=1.0)
    assert len(repo.completed) == 2
    assert repo.due_rows[0].feature_snapshot["market_fact_contract"]["assertions"][
        "same_contract_price_path_verified"
    ] is True


def test_shadow_backtest_side_label() -> None:
    assert side_label("long") == "做多"
    assert side_label("short") == "做空"
    assert side_label("hold") == "hold"
