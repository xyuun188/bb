from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.shadow_missed_opportunity_closed_loop import (
    ShadowMissedOpportunityClosedLoopService,
    summarize_shadow_missed_opportunities,
)


def _shadow(
    symbol: str,
    side: str,
    return_pct: float,
    *,
    index: int = 0,
    risk: str = "low",
    model_side: str | None = None,
    structure: str = "trend_up_momentum",
) -> SimpleNamespace:
    now = datetime.now(UTC)
    feature_snapshot = {
        "market_structure": structure,
        "price_vs_sma20": 0.035 if "up" in structure else -0.035,
        "returns_20": 0.024 if "up" in structure else -0.024,
        "volume_ratio": 1.65,
        "adx": 31.0,
        "volatility_20": 0.035,
        "loss_probability": 0.34,
        "tail_risk_score": 0.52,
        "abnormal_wick_max_pct": 0.0,
    }
    if risk == "high":
        feature_snapshot.update(
            {
                "loss_probability": 0.72,
                "tail_risk_score": 1.12,
                "abnormal_wick_max_pct": 7.5,
            }
        )
    raw = {
        "ml_signal": {
            "best_side": model_side if model_side is not None else side,
            "confidence": 0.72,
        },
        "opportunity_score": {
            "loss_probability": feature_snapshot["loss_probability"],
            "tail_risk_score": feature_snapshot["tail_risk_score"],
        },
    }
    return SimpleNamespace(
        id=index + 1,
        status="completed",
        missed_opportunity=True,
        decision_action="hold",
        symbol=symbol,
        best_action=side,
        long_return_pct=return_pct if side == "long" else -return_pct,
        short_return_pct=return_pct if side == "short" else -return_pct,
        feature_snapshot=feature_snapshot,
        raw_llm_response=raw,
        horizon_minutes=10,
        created_at=now - timedelta(minutes=index * 10),
        due_at=now - timedelta(minutes=index * 10),
    )


def test_repeated_same_symbol_side_creates_controlled_probe() -> None:
    report = summarize_shadow_missed_opportunities(
        [
            _shadow("BTC/USDT", "long", 0.44, index=1),
            _shadow("BTC/USDT", "long", 0.51, index=2),
            _shadow("BTC/USDT", "long", 0.58, index=3),
        ]
    )

    assert report["audit_only"] is True
    assert report["live_entry_mutation"] is False
    assert report["can_bypass_risk_controls"] is False
    assert report["weak_evidence_execution_allowed"] is False
    assert report["summary"]["probe_count"] == 1
    assert report["summary"]["adopted_count"] == 0
    candidate = report["probe_candidates"][0]
    assert candidate["symbol"] == "BTC/USDT"
    assert candidate["side"] == "long"
    assert candidate["allow_controlled_probe"] is True
    assert candidate["probe_rules"]["max_position_size_pct"] <= 0.015
    assert candidate["probe_rules"]["exit_rules"]
    assert "same_symbol_same_side_repeated" in candidate["adoption_reasons"]


def test_strong_repeated_evidence_is_adopted_for_learning_without_live_bypass() -> None:
    report = summarize_shadow_missed_opportunities(
        [
            _shadow("ETH/USDT", "short", value, index=index)
            for index, value in enumerate([0.70, 0.73, 0.76, 0.78, 0.80])
        ]
    )

    assert report["summary"]["adopted_count"] == 1
    adopted = report["adopted"][0]
    assert adopted["status"] == "adopted_learning"
    assert adopted["can_enter_training_positive"] is True
    assert adopted["allow_controlled_probe"] is True
    assert adopted["can_force_open"] is False
    assert adopted["can_bypass_risk_controls"] is False


def test_global_missed_count_across_symbols_never_qualifies() -> None:
    report = summarize_shadow_missed_opportunities(
        [_shadow(f"SYM{index}/USDT", "long", 0.70, index=index) for index in range(8)]
    )

    assert report["summary"]["missed_count"] == 8
    assert report["summary"]["adopted_count"] == 0
    assert report["summary"]["probe_count"] == 0
    assert report["summary"]["blocked_count"] == 8
    assert report["blocked_reason_counts"]["insufficient_repeated_same_symbol_side"] == 8
    assert report["global_missed_count_can_drive_entries"] is False


@pytest.mark.parametrize(
    ("risk", "model_side", "expected_reason"),
    [
        ("high", "long", "high_risk_evidence"),
        ("low", "short", "model_direction_not_aligned"),
    ],
)
def test_high_risk_or_model_conflict_blocks_repeated_misses(
    risk: str, model_side: str, expected_reason: str
) -> None:
    report = summarize_shadow_missed_opportunities(
        [
            _shadow("SOL/USDT", "long", 0.54, index=1, risk=risk, model_side=model_side),
            _shadow("SOL/USDT", "long", 0.57, index=2, risk=risk, model_side=model_side),
            _shadow("SOL/USDT", "long", 0.59, index=3, risk=risk, model_side=model_side),
        ]
    )

    assert report["summary"]["adopted_count"] == 0
    assert report["summary"]["probe_count"] == 0
    assert report["blocked_reason_counts"][expected_reason] == 1
    blocked = report["blocked"][0]
    assert blocked["allow_controlled_probe"] is False
    assert blocked["can_enter_training_positive"] is False


def test_one_off_move_is_observe_only_not_probe() -> None:
    report = summarize_shadow_missed_opportunities(
        [
            _shadow("ARB/USDT", "long", 0.08, index=1),
            _shadow("ARB/USDT", "long", 0.11, index=2),
            _shadow("ARB/USDT", "long", 2.40, index=3),
        ]
    )

    assert report["summary"]["adopted_count"] == 0
    assert report["summary"]["probe_count"] == 0
    assert report["blocked_reason_counts"]["one_off_move"] == 1
    assert report["blocked"][0]["status"] == "blocked"


@pytest.mark.asyncio
async def test_report_uses_recent_primary_key_window_for_online_read_only_path() -> None:
    class FakeScalarResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def all(self) -> list[SimpleNamespace]:
            return self._rows

    class FakeResult:
        def __init__(self, rows: list[SimpleNamespace]) -> None:
            self._rows = rows

        def scalars(self) -> FakeScalarResult:
            return FakeScalarResult(self._rows)

    class FakeSession:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def execute(self, statement: object) -> FakeResult:
            compiled = str(statement)
            self.statements.append(compiled)
            if "shadow_backtests" in compiled:
                return FakeResult(
                    [
                        _shadow("BTC/USDT", "long", 0.44, index=1),
                        SimpleNamespace(status="pending", missed_opportunity=True),
                    ]
                )
            return FakeResult([])

    class FakeSessionContext:
        def __init__(self, session: FakeSession) -> None:
            self._session = session

        async def __aenter__(self) -> FakeSession:
            return self._session

        async def __aexit__(self, *_exc: object) -> None:
            return None

    fake_session = FakeSession()

    def session_context_factory() -> FakeSessionContext:
        return FakeSessionContext(fake_session)

    service = ShadowMissedOpportunityClosedLoopService(
        session_context_factory=session_context_factory
    )

    report = await service.report()

    assert report["window_hours"] == 24
    assert report["summary"]["completed_count"] == 1
    assert any("ORDER BY shadow_backtests.id DESC" in item for item in fake_session.statements)
    assert not any("WHERE shadow_backtests.status" in item for item in fake_session.statements)
    assert any("ORDER BY ai_decisions.id DESC" in item for item in fake_session.statements)
