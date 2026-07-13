from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from services.strategy_learning import (
    StrategyCandidateGenerator,
    StrategyFeedback,
    StrategyLearningEngine,
    StrategyProfile,
)


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
    long_returns = long_returns or [4.0, -1.0, 5.0]
    short_returns = short_returns or [0.1, 0.1, -2.0]
    shadow_long_returns = shadow_long_returns or [3.0, 2.0]
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


def test_scheduler_uses_walk_forward_and_cost_complete_shadow_governance() -> None:
    payload = StrategyLearningEngine().build_from_feedback(_feedback())
    schedule = payload["schedule"]

    assert schedule["scheduler_mode"] == "governed_dynamic_return"
    assert schedule["candidate_count"] == len(schedule["candidates"])
    assert schedule["governed_candidate_count"] > 0
    assert schedule["runtime"]["production_influence_enabled"] is True
    assert schedule["runtime"]["can_authorize_entry"] is False
    assert schedule["runtime"]["can_change_size_or_leverage"] is False
    assert schedule["active_profile"]["promotion"]["production_permission"] is False
    assert schedule["active_profile"]["params"]["selector"]["scope"] != "symbol_side"
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


def test_missing_shadow_evidence_fails_closed_without_numeric_fallback() -> None:
    feedback = _feedback()
    feedback.shadow_return_samples.clear()
    payload = StrategyLearningEngine().build_from_feedback(feedback)

    assert payload["schedule"]["scheduler_mode"] == "shadow_validation"
    assert payload["schedule"]["governed_candidate_count"] == 0
    assert payload["schedule"]["runtime"]["production_influence_enabled"] is False
    assert payload["schedule"]["active_profile"] is None
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

    assert result["strategy_profile_id"] is None
    assert result["strategy_profile_version"] is None
    assert result["strategy_learning"]["active_profile"] == {}
    assert (
        result["strategy_learning"]["leading_candidate"]["id"]
        == payload["schedule"]["leading_candidate"]["id"]
    )
    assert result["strategy_learning"]["production_permission"] is False


def test_external_profile_cannot_reenter_scheduler() -> None:
    candidate = StrategyProfile(
        profile_id="external_execution_override",
        version=1,
        label="external override",
        status="candidate",
        source="external",
        description="must be ignored",
        params={"entry_threshold": -1, "position_fraction": 1, "leverage": 99},
    )
    payload = StrategyLearningEngine().build_from_feedback(
        _feedback(),
        extra_profiles=[candidate],
    )

    assert "external_execution_override" not in {
        row["id"] for row in payload["schedule"]["candidates"]
    }


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
    assert result["strategy_profile_id"] == payload["schedule"]["active_profile"]["id"]
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
        "authoritative_return_samples",
        "shadow_return_samples",
    } <= names
