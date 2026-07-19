from __future__ import annotations

from typing import Any

import pytest

from services import competition_service as competition_module
from services.competition_service import CompetitionService


class _FailingDecisionRepo:
    async def get_recent_decisions(self, model_name: str, limit: int) -> list[Any]:
        raise RuntimeError("Authorization: Bearer sharpe-secret-value failed")


class _FailingTradeRepo:
    async def get_recent_orders(self, model_name: str, limit: int) -> list[Any]:
        raise RuntimeError("password=max-drawdown-secret failed")


class _FakeLogger:
    def __init__(self) -> None:
        self.warning_events: list[dict[str, Any]] = []
        self.info_events: list[dict[str, Any]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warning_events.append({"event": event, **fields})

    def info(self, event: str, **fields: Any) -> None:
        self.info_events.append({"event": event, **fields})


@pytest.mark.asyncio
async def test_competition_metric_fallback_logs_redacted_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()
    monkeypatch.setattr(competition_module, "logger", fake_logger)
    service = CompetitionService()
    trade_repo: Any = _FailingTradeRepo()
    decision_repo: Any = _FailingDecisionRepo()

    sharpe = await service._calculate_sharpe(
        trade_repo,
        decision_repo,
        "unit-model",
    )
    max_drawdown = await service._calculate_max_drawdown(
        trade_repo,
        "unit-model",
        1000.0,
    )

    assert sharpe == 0.0
    assert max_drawdown == 0.0
    assert fake_logger.warning_events == [
        {
            "event": "model sharpe calculation failed",
            "model_name": "unit-model",
            "error": "Authorization: *** failed",
        },
        {
            "event": "model max drawdown calculation failed",
            "model_name": "unit-model",
            "error": "password=*** failed",
        },
    ]
    assert "sharpe-secret-value" not in str(fake_logger.warning_events)
    assert "max-drawdown-secret" not in str(fake_logger.warning_events)


@pytest.mark.asyncio
async def test_auto_promote_best_model_is_recommendation_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = _FakeLogger()
    service = CompetitionService()
    switched: list[str] = []

    class FakeModeManager:
        active_model_name = "current-active-model"

        async def switch_to_live(self, model_name: str) -> None:
            switched.append(model_name)

    async def fake_select_best_model() -> str:
        return "candidate-model"

    monkeypatch.setattr(competition_module, "logger", fake_logger)
    monkeypatch.setattr(service, "select_best_model", fake_select_best_model)
    monkeypatch.setattr(competition_module, "mode_manager", FakeModeManager())

    best = await service.auto_promote_best_model()

    assert best == "candidate-model"
    assert switched == []
    assert fake_logger.info_events == [
        {
            "event": "best model promotion recommendation recorded without active switch",
            "model": "candidate-model",
            "current_active_model": "current-active-model",
            "live_mutation": False,
            "policy": "phase3_observe_only",
        }
    ]
