from __future__ import annotations

import ast
import inspect
import json
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from services.strategy_learning import (
    StrategyCandidateGenerator,
    StrategyFeedback,
    StrategyLearningEngine,
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
        "net_return_after_cost_pct": return_pct,
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
    assert schedule["runtime"]["production_influence_enabled"] is True
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
    low_win_high_return = [-1.0, -1.0, 4.0] * 3
    high_win_negative_return = [0.1, 0.1, -1.0] * 3
    payload = StrategyLearningEngine().build_from_feedback(
        _feedback(
            long_returns=low_win_high_return,
            short_returns=high_win_negative_return,
            shadow_long_returns=low_win_high_return,
            shadow_short_returns=high_win_negative_return,
        )
    )
    side_candidates = {
        row["params"]["selector"]["side"]: row
        for row in payload["schedule"]["candidates"]
        if row["params"]["selector"]["scope"] == "side"
    }

    assert side_candidates["long"]["rank"] < side_candidates["short"]["rank"]
    assert side_candidates["long"]["promotion"]["production_influence_eligible"] is True
    assert side_candidates["short"]["promotion"]["production_influence_eligible"] is False


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
    assert long_candidate["promotion"]["production_influence_eligible"] is False
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
    assert payload["schedule"]["runtime"]["production_influence_enabled"] is False
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
