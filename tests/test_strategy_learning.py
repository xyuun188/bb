from __future__ import annotations

import ast
import inspect
import json
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import services.strategy_learning as strategy_learning_module
from services.strategy_learning import (
    StrategyCandidateGenerator,
    StrategyFeedback,
    StrategyLearningEngine,
    StrategyLearningService,
    _build_historical_replay_observations,
    _json_safe,
    _regime_label,
    _runtime_prior_usage,
)


def test_strategy_samples_use_feature_based_market_regime() -> None:
    assert _regime_label(
        {
            "market_regime": {"mode": "return_distribution_observation"},
            "adx_14": 30.0,
            "returns_20": 0.005,
            "price_vs_sma20": 0.004,
            "price_vs_sma50": 0.002,
        }
    ) == "trend_up"


def test_strategy_learning_json_payload_replaces_nested_non_finite_values() -> None:
    payload = _json_safe(
        {
            "score": float("-inf"),
            "nested": [float("nan"), {"upside": float("inf"), "valid": 0.25}],
            "generated_at": datetime(2026, 7, 15, tzinfo=UTC),
        }
    )

    assert payload == {
        "score": None,
        "nested": [None, {"upside": None, "valid": 0.25}],
        "generated_at": "2026-07-15T00:00:00+00:00",
    }
    json.dumps(payload, allow_nan=False)


def test_historical_replay_observations_build_in_background_helper() -> None:
    epoch_start = datetime(2026, 7, 1, tzinfo=UTC)
    row = SimpleNamespace(
        id=1,
        decision_id=2,
        symbol="BTC/USDT",
        training_feature_snapshot={
            "adx_14": 25.0,
            "funding_data_available": True,
            "funding_rate": 0.0001,
            "funding_interval_minutes": 480,
            "taker_fee_rate": 0.0005,
            "spread_pct": 0.01,
            "market_fact_quality_status": "verified",
        },
        feature_snapshot={},
        horizon_minutes=10,
        long_return_pct=1.0,
        short_return_pct=-1.0,
        created_at=datetime(2026, 7, 2, tzinfo=UTC),
        updated_at=datetime(2026, 7, 2, 0, 10, tzinfo=UTC),
        due_at=datetime(2026, 7, 2, 0, 10, tzinfo=UTC),
        decision_action="long",
        best_action="long",
        missed_opportunity=False,
    )

    observations, excluded = _build_historical_replay_observations(
        [row],
        epoch_start=epoch_start,
    )

    assert len(observations) == 1
    assert observations[0]["symbol"] == "BTC/USDT"
    assert excluded == {}


@pytest.mark.asyncio
async def test_runtime_feedback_limits_and_orders_historical_replay_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self, rows: list[Any]) -> None:
            self.rows = rows

        def scalars(self) -> FakeResult:
            return self

        def all(self) -> list[Any]:
            return list(self.rows)

    class FakeReadSession:
        def __init__(self) -> None:
            self.statements: list[Any] = []

        async def execute(self, statement: Any) -> FakeResult:
            self.statements.append(statement)
            if len(self.statements) == 3:
                return FakeResult([SimpleNamespace(id=2), SimpleNamespace(id=1)])
            return FakeResult([])

    class FakeReadContext:
        def __init__(self, session: FakeReadSession) -> None:
            self.session = session

        async def __aenter__(self) -> FakeReadSession:
            return self.session

        async def __aexit__(self, *_args: object) -> None:
            return None

    async def empty_outcomes(**_kwargs: Any) -> list[dict[str, Any]]:
        return []

    replay_row_ids: list[int] = []

    def capture_replay_rows(
        rows: list[Any],
        *,
        epoch_start: datetime,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        del epoch_start
        replay_row_ids.extend(int(row.id) for row in rows)
        return [], {}

    session = FakeReadSession()
    monkeypatch.setattr(
        strategy_learning_module,
        "load_authoritative_trade_outcomes",
        empty_outcomes,
    )
    monkeypatch.setattr(
        strategy_learning_module,
        "load_training_epoch_start",
        lambda: datetime(2026, 7, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(
        strategy_learning_module,
        "get_read_session_ctx",
        lambda: FakeReadContext(session),
    )
    monkeypatch.setattr(
        strategy_learning_module,
        "_build_historical_replay_observations",
        capture_replay_rows,
    )

    await StrategyLearningService()._feedback(
        mode="paper",
        hours=24,
        limit=7,
        include_historical_replay=True,
    )

    replay_statement = session.statements[2]
    assert replay_statement._limit_clause.value == 7
    assert replay_row_ids == [1, 2]


@pytest.mark.asyncio
async def test_realtime_strategy_context_does_not_run_model_historical_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feedback_calls: list[dict[str, Any]] = []
    build_calls: list[dict[str, Any]] = []

    class FakeEngine:
        def build_from_feedback(self, feedback: StrategyFeedback, **kwargs: Any) -> dict:
            del feedback
            build_calls.append(kwargs)
            return {
                "schedule": {
                    "candidates": [],
                    "continuous_strategy_routing": {"applied": False},
                }
            }

        def apply_to_context(
            self,
            strategy_context: dict[str, Any],
            _payload: dict[str, Any],
            *,
            paper_strategy_champion: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                **strategy_context,
                "paper_strategy_champion": paper_strategy_champion,
            }

    class FakeChampionService:
        current_calls = 0
        reconcile_calls = 0

        async def current(self, _mode: str) -> dict[str, Any]:
            self.current_calls += 1
            return {"status": "base_strategy"}

        async def reconcile(self, **_kwargs: Any) -> dict[str, Any]:
            self.reconcile_calls += 1
            return {"status": "reconciled"}

    async def fake_feedback(**kwargs: Any) -> StrategyFeedback:
        feedback_calls.append(kwargs)
        return _feedback()

    champion = FakeChampionService()
    service = StrategyLearningService(
        engine=FakeEngine(),
        champion_service=champion,
    )
    monkeypatch.setattr(service, "_feedback", fake_feedback)
    monkeypatch.setattr(
        strategy_learning_module,
        "paper_strategy_replay_available",
        lambda _blueprint: True,
    )

    result = await service.apply_to_strategy_context(
        mode="paper",
        strategy_context={"source": "trading_loop"},
        open_positions=[],
        model_strategy_blueprint={"model_version": "model-v1"},
        model_predictor=lambda *_args, **_kwargs: {},
    )

    assert feedback_calls[0]["include_historical_replay"] is False
    assert build_calls[0]["model_strategy_blueprint"] is None
    assert build_calls[0]["model_predictor"] is None
    assert champion.current_calls == 1
    assert champion.reconcile_calls == 0
    assert result["paper_strategy_champion"]["status"] == "base_strategy"


def _sample(
    source_id: int,
    *,
    side: str,
    return_pct: float,
    symbol: str = "BTC/USDT",
    regime: str = "trend",
) -> dict:
    return {
        "source_id": source_id,
        "source_row_id": source_id,
        "position_id": source_id,
        "symbol": symbol,
        "side": side,
        "market_regime": regime,
        "net_return_after_all_cost_pct": return_pct,
        "net_pnl_after_all_costs_usdt": return_pct,
        "timestamp": f"2026-07-12T{source_id:02d}:00:00+00:00",
    }


def _feedback(
    *,
    long_returns: list[float] | None = None,
    short_returns: list[float] | None = None,
    shadow_long_returns: list[float] | None = None,
    shadow_short_returns: list[float] | None = None,
) -> StrategyFeedback:
    long_returns = long_returns or [4.0, -1.0, 5.0, -0.5, 4.0, 5.0]
    short_returns = short_returns or [0.1, 0.1, -2.0]
    shadow_long_returns = shadow_long_returns or [3.0, -0.1, 2.0, 3.0]
    shadow_short_returns = shadow_short_returns or [0.1, -2.0]
    authoritative = [
        _sample(index, side="long", return_pct=value)
        for index, value in enumerate(long_returns, start=1)
    ] + [
        _sample(index, side="short", return_pct=value)
        for index, value in enumerate(short_returns, start=len(long_returns) + 1)
    ]
    shadow = [
        _sample(index, side="long", return_pct=value)
        for index, value in enumerate(shadow_long_returns, start=101)
    ] + [
        _sample(index, side="short", return_pct=value)
        for index, value in enumerate(shadow_short_returns, start=201)
    ]
    return StrategyFeedback(
        mode="paper",
        window_hours=168,
        generated_at="2026-07-12T00:00:00+00:00",
        totals={"sample_count": len(authoritative)},
        side_performance={},
        open_position_pressure={},
        decision_quality={},
        shadow_feedback={},
        expert_memory={},
        manual_intervention={},
        trade_fact_quarantine={},
        reflection_feedback={},
        event_feedback={},
        authoritative_return_observation={"sample_count": len(authoritative)},
        problems=[],
        root_causes=[],
        training_policy={},
        authoritative_return_samples=authoritative,
        shadow_return_samples=shadow,
    )


def test_strategy_candidates_are_generated_from_observed_partitions() -> None:
    profiles = StrategyCandidateGenerator().generate(_feedback())

    selectors = [profile.params["selector"] for profile in profiles]
    assert len(profiles) == len(selectors) == 6
    assert {selector["scope"] for selector in selectors} == {
        "side",
        "symbol_side",
        "regime_side",
    }
    assert {selector["side"] for selector in selectors} == {"long", "short"}
    assert all(profile.params["objective"] == "maximize_authoritative_fee_after_return_rate" for profile in profiles)
    assert all(profile.params["current_return_contract_required"] is True for profile in profiles)


def test_portfolio_strategy_requires_cross_symbol_generalization() -> None:
    payload = StrategyLearningEngine().build_from_feedback(_feedback())
    long_side = next(
        row
        for row in payload["schedule"]["candidates"]
        if row["params"]["selector"] == {"scope": "side", "side": "long"}
    )

    development = long_side["backtest"]["cross_symbol_generalization"]
    exam = long_side["shadow_validation"]["cross_symbol_generalization"]
    assert development["stable"] is False
    assert exam["stable"] is False
    assert "cross_symbol_coverage_insufficient" in development["blocking_reasons"]
    assert "walk_forward_cross_symbol_generalization_failed" in long_side["promotion"][
        "rejection_reasons"
    ]
    assert "shadow_cross_symbol_generalization_failed" in long_side["promotion"][
        "rejection_reasons"
    ]


def test_runtime_prior_usage_reports_actual_matches_not_ranked_candidates() -> None:
    entry_candidate_evidence = {
        "long": {
            "scheduled_return_prior": {
                "available": True,
                "profile_id": "btc_long_prior",
                "profile_version": 7,
                "rank": 2,
                "selector": {
                    "scope": "symbol_side",
                    "symbol": "BTC/USDT",
                    "side": "long",
                },
                "can_authorize_entry": False,
            }
        },
        "short": {"scheduled_return_prior": {"available": False}},
    }
    newer = SimpleNamespace(
        id=12,
        symbol="BTC/USDT",
        action="hold",
        created_at=datetime(2026, 7, 14, 5, 0, tzinfo=UTC),
        entry_candidate_evidence=entry_candidate_evidence,
        raw_llm_response={},
    )
    older_same_route = SimpleNamespace(
        id=11,
        symbol="BTC/USDT",
        action="short",
        created_at=datetime(2026, 7, 14, 4, 0, tzinfo=UTC),
        raw_llm_response={"entry_candidate_evidence": entry_candidate_evidence},
    )

    usage = _runtime_prior_usage([newer, older_same_route])

    assert usage["inspected_decision_count"] == 2
    assert usage["matched_decision_count"] == 2
    assert usage["matched_evaluation_count"] == 2
    assert usage["matched_profile_count"] == 1
    assert usage["latest_matches"] == [
        {
            "decision_id": 12,
            "matched_at": "2026-07-14T05:00:00+00:00",
            "symbol": "BTC/USDT",
            "decision_action": "hold",
            "evaluated_side": "long",
            "profile_id": "btc_long_prior",
            "profile_version": 7,
            "rank": 2,
            "selector": {
                "scope": "symbol_side",
                "symbol": "BTC/USDT",
                "side": "long",
            },
            "role": "historical_prior_only",
            "can_authorize_entry": False,
        }
    ]
    assert usage["decision_records"][0]["decision_id"] == 12
    assert usage["decision_records"][0]["side_evaluations"][0][
        "evaluation_status"
    ] == "matched_historical_prior"
    assert usage["decision_records"][0]["side_evaluations"][0][
        "context_fields_influenced"
    ] == ["scheduled_return_prior"]


def test_scheduler_uses_walk_forward_and_cost_complete_shadow_governance() -> None:
    payload = StrategyLearningEngine().build_from_feedback(_feedback())
    schedule = payload["schedule"]

    assert schedule["scheduler_mode"] == "governed_dynamic_return"
    assert schedule["candidate_count"] == len(schedule["candidates"])
    assert schedule["governed_candidate_count"] > 0
    assert schedule["runtime"]["historical_prior_context_enabled"] is True
    assert schedule["runtime"]["can_authorize_entry"] is False
    assert schedule["runtime"]["can_change_size_or_leverage"] is False
    production = schedule["current_production_strategy"]
    assert production["id"] == "dynamic_fee_after_return_execution"
    assert production["enabled"] is True
    assert production["historical_prior_can_authorize_entry"] is False
    assert "active_profile" not in schedule
    assert all(row["partition_policy"] == "sqrt_cardinality_expanding_walk_forward" for row in schedule["backtest"]["rows"])
    assert schedule["shadow_validation"]["cost_complete_required"] is True
    assert all(row["rows"] == [] for row in schedule["shadow_validation"]["rows"])
    assert all(
        row["row_detail_included"] is False
        for row in schedule["shadow_validation"]["rows"]
    )


def test_full_detail_expands_shadow_evidence_without_changing_candidate_count() -> None:
    engine = StrategyLearningEngine()
    feedback = _feedback()
    summary = engine.build_from_feedback(feedback, detail="summary")
    full = engine.build_from_feedback(feedback, detail="full")

    assert full["schedule"]["candidate_count"] == summary["schedule"]["candidate_count"]
    assert any(row["rows"] for row in full["schedule"]["shadow_validation"]["rows"])
    assert all(
        row["row_detail_included"] is True
        for row in full["schedule"]["shadow_validation"]["rows"]
    )


def test_low_win_high_return_policy_outranks_high_win_negative_return_policy() -> None:
    low_win_high_return = [-1.0, 4.0, -1.0, 4.0] * 2
    high_win_negative_return = [0.1, 0.1, -1.0] * 4
    feedback = _feedback(
        long_returns=low_win_high_return,
        short_returns=high_win_negative_return,
        shadow_long_returns=low_win_high_return,
        shadow_short_returns=high_win_negative_return,
    )
    for samples in (
        feedback.authoritative_return_samples,
        feedback.shadow_return_samples,
    ):
        rows_by_side = {
            side: [row for row in samples if row["side"] == side]
            for side in ("long", "short")
        }
        for rows in rows_by_side.values():
            midpoint = len(rows) // 2
            for index, row in enumerate(rows):
                row["symbol"] = "BTC/USDT" if index < midpoint else "ETH/USDT"

    payload = StrategyLearningEngine().build_from_feedback(feedback)
    side_candidates = {
        row["params"]["selector"]["side"]: row
        for row in payload["schedule"]["candidates"]
        if row["params"]["selector"]["scope"] == "side"
    }

    assert side_candidates["long"]["rank"] < side_candidates["short"]["rank"]
    assert side_candidates["long"]["promotion"]["historical_prior_context_eligible"] is True
    assert side_candidates["short"]["promotion"]["historical_prior_context_eligible"] is False


def test_strategy_candidate_cannot_promote_with_undefined_profit_factor() -> None:
    all_win = [1.0, 1.2, 0.8, 1.1]
    payload = StrategyLearningEngine().build_from_feedback(
        _feedback(long_returns=all_win, shadow_long_returns=all_win)
    )
    long_candidate = next(
        row
        for row in payload["schedule"]["candidates"]
        if row["params"]["selector"] == {"scope": "side", "side": "long"}
    )

    assert long_candidate["backtest"]["metrics"]["profit_factor"] is None
    assert long_candidate["shadow_validation"]["metrics"]["profit_factor"] is None
    assert long_candidate["promotion"]["historical_prior_context_eligible"] is False
    assert "walk_forward_profit_factor_undefined" in long_candidate["promotion"][
        "rejection_reasons"
    ]
    assert "shadow_profit_factor_undefined" in long_candidate["promotion"][
        "rejection_reasons"
    ]


def test_missing_shadow_evidence_fails_closed_without_numeric_fallback() -> None:
    feedback = _feedback()
    feedback.shadow_return_samples.clear()
    payload = StrategyLearningEngine().build_from_feedback(feedback)

    assert payload["schedule"]["scheduler_mode"] == "shadow_validation"
    assert payload["schedule"]["governed_candidate_count"] == 0
    assert payload["schedule"]["runtime"]["historical_prior_context_enabled"] is False
    assert "active_profile" not in payload["schedule"]
    assert payload["schedule"]["current_production_strategy"]["status"] == "running"
    assert payload["schedule"]["leading_candidate"] == payload["schedule"]["candidates"][0]
    assert all(
        "no_cost_complete_shadow_samples" in row["promotion"]["rejection_reasons"]
        for row in payload["schedule"]["candidates"]
    )
    assert all(
        row["shadow_validation"]["metrics"]["return_lcb_pct"] is None
        for row in payload["schedule"]["candidates"]
    )


def test_blocked_leading_candidate_is_not_attached_as_active_strategy() -> None:
    feedback = _feedback()
    feedback.shadow_return_samples.clear()
    engine = StrategyLearningEngine()
    payload = engine.build_from_feedback(feedback)

    result = engine.apply_to_context({}, payload)

    assert "strategy_profile_id" not in result
    assert "strategy_profile_version" not in result
    assert "active_profile" not in result["strategy_learning"]
    assert result["current_production_strategy"]["id"] == (
        "dynamic_fee_after_return_execution"
    )
    assert (
        result["strategy_learning"]["leading_candidate"]["id"]
        == payload["schedule"]["leading_candidate"]["id"]
    )
    assert result["strategy_learning"]["production_permission"] is False


def test_scheduler_has_no_external_profile_injection_interface() -> None:
    parameters = inspect.signature(
        StrategyLearningEngine.build_from_feedback
    ).parameters

    assert "extra_profiles" not in parameters


def test_strategy_learning_context_cannot_mutate_execution_fields() -> None:
    engine = StrategyLearningEngine()
    original = {
        "entry_threshold": "sentinel",
        "position_fraction": "sentinel",
        "leverage": "sentinel",
        "exit_fraction": "sentinel",
        "production_permission": "sentinel",
    }
    payload = engine.build_from_feedback(
        _feedback(),
        current_context={"account_equity": 100.0, "market_regime": {"mode": "trend"}},
    )
    result = engine.apply_to_context(dict(original), payload)

    for key, value in original.items():
        assert result[key] == value
    learning = result["strategy_learning"]
    assert learning["advisory_prior_only"] is True
    assert learning["production_permission"] is False
    assert result["current_production_strategy"] == payload[
        "current_production_strategy"
    ]
    assert result["scheduler_reason"] == payload["schedule"]["reason"]


def test_schedule_runtime_keeps_execution_fields_out() -> None:
    runtime = StrategyLearningEngine().build_from_feedback(_feedback())["schedule"]["runtime"]
    for field_name in (
        "entry_threshold",
        "position_fraction",
        "position_size_pct",
        "leverage",
        "stop_loss_pct",
        "take_profit_pct",
        "exit_fraction",
        "capacity",
    ):
        assert field_name not in runtime


def test_strategy_scheduler_has_no_fixed_promotion_gate_or_win_rate_branch() -> None:
    source_path = Path(__file__).resolve().parents[1] / "services/strategy_learning.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden_names = {
        "MIN_PROMOTION_SAMPLES",
        "MIN_PROFIT_FACTOR",
        "MIN_WIN_RATE",
        "CANDIDATE_COUNT",
        "PROMOTION_THRESHOLD",
    }
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    assert forbidden_names.isdisjoint(assigned_names)
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.While)):
            assert "win_rate" not in ast.unparse(node.test).lower()
    assert "observation_only" not in source


def test_strategy_feedback_shadow_query_omits_unused_large_payload_columns() -> None:
    source_path = Path(__file__).resolve().parents[1] / "services/strategy_learning.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    feedback_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_feedback"
    )
    shadow_assignment = next(
        node
        for node in ast.walk(feedback_method)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "shadow_rows"
            for target in node.targets
        )
    )
    query_source = ast.unparse(shadow_assignment.value)

    assert "select(" in query_source
    assert "ShadowBacktest.training_feature_snapshot" in query_source
    assert "ShadowBacktest.long_return_pct" in query_source
    assert "ShadowBacktest.short_return_pct" in query_source
    for unused_column in (
        "ShadowBacktest.raw_llm_response",
        "ShadowBacktest.feature_snapshot",
        "ShadowBacktest.note",
    ):
        assert unused_column not in query_source


def test_feedback_contract_carries_authoritative_audit_and_evaluation_samples() -> None:
    names = {item.name for item in fields(StrategyFeedback)}
    assert {
        "totals",
        "trade_fact_quarantine",
        "reflection_feedback",
        "training_policy",
        "runtime_prior_usage",
        "authoritative_return_samples",
        "shadow_return_samples",
    } <= names


def test_feedback_summary_reports_prior_record_count_without_embedding_records() -> None:
    feedback = _feedback()
    records = [
        {"decision_id": 12, "side_evaluations": [{"side": "long"}]},
        {"decision_id": 11, "side_evaluations": [{"side": "short"}]},
    ]
    feedback.runtime_prior_usage["decision_records"] = records

    summary = feedback.to_dict()["runtime_prior_usage"]
    audit = feedback.to_dict(include_runtime_prior_records=True)[
        "runtime_prior_usage"
    ]

    assert summary["decision_record_count"] == 2
    assert summary["decision_records_included"] is False
    assert "decision_records" not in summary
    assert audit["decision_record_count"] == 2
    assert audit["decision_records_included"] is True
    assert audit["decision_records"] == records
