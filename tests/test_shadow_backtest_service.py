from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.shadow_backtest_service import ShadowBacktestService, side_label


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
        self.memories: list[dict[str, Any]] = []

    async def create_shadow_backtest(self, data: dict[str, Any]) -> None:
        self.created.append(data)

    async def get_due_shadow_backtests(self, limit: int = 200) -> list[Any]:
        assert limit == 200
        return self.due_rows

    async def complete_shadow_backtest(self, row: Any, **data: Any) -> None:
        for key, value in data.items():
            setattr(row, key, value)
        self.completed.append(data)

    async def upsert_memory(self, data: dict[str, Any]) -> None:
        self.memories.append(data)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _service(repo: _FakeRepo, latest_price: float = 101.0) -> ShadowBacktestService:
    async def latest(_symbol: str) -> float:
        return latest_price

    return ShadowBacktestService(
        latest_price_provider=latest,
        symbol_normalizer=lambda symbol: str(symbol or "").upper(),
        float_parser=_float,
        session_factory=_SessionCtx,
        repository_factory=lambda _session: repo,
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
        feature_snapshot={"current_price": 100.0, "adx_14": 28.0},
        raw_response={"reason": "test"},
    )


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
                "live_mutation": False,
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
                "live_mutation": False,
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
async def test_shadow_backtest_cost_incomplete_positive_move_does_not_create_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "shadow_memory_enabled", True)
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
    assert repo.memories == []


@pytest.mark.asyncio
async def test_shadow_backtest_records_fee_after_observation_without_probe_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "shadow_memory_enabled", True)
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
            "funding_rate": 0.0,
            "funding_interval_minutes": 480.0,
        },
    )
    repo.due_rows = [row]

    await _service(repo, latest_price=101.0).update_due()

    assert len(repo.memories) == 4
    assert {item["expert_name"] for item in repo.memories} == {
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "risk_expert",
    }
    assert all(item["memory_type"] == "shadow_missed_opportunity" for item in repo.memories)
    assert repo.memories[0]["extra"]["actual_price"] == 101.0
    assert all(item["confidence_adjustment"] == 0.0 for item in repo.memories)
    assert all(item["position_size_multiplier"] == 1.0 for item in repo.memories)
    assert all(item["success_count"] == 0 for item in repo.memories)
    assert all(item["failure_count"] == 0 for item in repo.memories)
    assert all(item["extra"]["cost_complete"] is True for item in repo.memories)
    assert all(
        item["extra"]["production_evidence_eligible"] is False for item in repo.memories
    )
    assert all(item["recommended_action"] == "shadow_observation_only" for item in repo.memories)
    assert repo.memories[0]["extra"]["net_return_after_cost_pct"] < 1.0


@pytest.mark.asyncio
async def test_shadow_horizons_share_one_correlated_memory_key_per_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "shadow_memory_enabled", True)
    repo = _FakeRepo()
    feature_snapshot = {
        "bid": 99.99,
        "ask": 100.01,
        "orderbook_bid_depth": 100_000.0,
        "orderbook_ask_depth": 100_000.0,
        "funding_rate": 0.0,
        "funding_interval_minutes": 480.0,
    }
    repo.due_rows = [
        SimpleNamespace(
            id=row_id,
            decision_id=456,
            model_name="ensemble_trader",
            symbol="BTC/USDT",
            decision_action="hold",
            entry_price=100.0,
            horizon_minutes=horizon,
            status="pending",
            note="",
            feature_snapshot=feature_snapshot,
        )
        for row_id, horizon in ((10, 10), (11, 30))
    ]

    await _service(repo, latest_price=101.0).update_due()

    assert len(repo.memories) == 8
    assert len({item["memory_key"] for item in repo.memories}) == 4
    assert {
        item["extra"]["correlation_group"] for item in repo.memories
    } == {"shadow_decision:456"}


@pytest.mark.asyncio
async def test_shadow_backtest_price_collection_does_not_hold_database_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "shadow_memory_enabled", False)
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
async def test_shadow_backtest_service_quarantines_dirty_completed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from config.settings import settings

    monkeypatch.setattr(settings, "shadow_memory_enabled", True)
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
        },
    )
    repo.due_rows = [row]

    await _service(repo, latest_price=0.3910).update_due()

    assert row.status == "quarantined"
    assert "[training_quarantine] price_outside_24h_range" in row.note
    assert repo.memories == []
    assert repo.completed[0]["actual_price"] == pytest.approx(0.3910)


def test_shadow_backtest_side_label() -> None:
    assert side_label("long") == "做多"
    assert side_label("short") == "做空"
    assert side_label("hold") == "hold"
