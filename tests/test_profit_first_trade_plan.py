from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.profit_first_trade_plan import (
    attach_profit_first_trade_plan,
    build_profit_first_trade_plan,
    normalize_losing_exit_attribution,
    normalize_no_entry_reason,
    summarize_model_strategy_realized_pnl,
    summarize_probe_loop_health,
)


def _complete_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.84,
        reasoning="strong aligned entry",
        position_size_pct=0.06,
        suggested_leverage=4.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        feature_snapshot={"close": 100.0},
        raw_response={
            "analysis_type": "entry_candidate",
            "strategy_learning_context": {"strategy_profile_id": "balanced_probe"},
            "opportunity_score": {
                "score": 3.4,
                "side": "long",
                "expected_return_pct": 1.05,
                "expected_net_return_pct": 0.9,
                "fee_pct": 0.05,
                "slippage_pct": 0.04,
                "expected_loss_pct": 0.20,
                "profit_quality_ratio": 1.25,
                "reward_risk_ratio": 2.5,
                "server_profit_loss_probability": 0.38,
                "tail_risk_score": 0.62,
                "side_realized_pnl_usdt": 2.0,
                "ml_aligned": True,
                "local_profit_aligned": True,
                "timeseries_aligned": True,
                "expert_aligned": True,
                "evidence_score": {
                    "tier": "normal",
                    "effective_score": 88,
                    "components": [
                        {"source": "sentiment", "status": "aligned"},
                        {"source": "shadow_memory", "status": "aligned"},
                    ],
                },
            },
            "profit_risk_sizing": {
                "quality_tier": "high_profit",
                "position_size_pct": 0.06,
                "final_notional_usdt": 120.0,
                "planned_stop_loss_usdt": 2.8,
                "max_stop_loss_usdt": 4.0,
                "expected_profit_usdt": 1.08,
            },
        },
    )


def test_complete_entry_builds_profit_first_plan_with_lane_and_exit_plan() -> None:
    plan = build_profit_first_trade_plan(
        _complete_decision(),
        now=datetime(2026, 6, 29, tzinfo=UTC),
    )

    assert plan.plan_version.startswith("profit-first-v3")
    assert plan.analysis_type == "entry_candidate"
    assert plan.strategy_profile_id == "balanced_probe"
    assert plan.expected_net_return_pct == pytest.approx(0.9)
    assert plan.expected_profit_usdt == pytest.approx(1.08)
    assert plan.decision_lane == "meaningful_entry"
    assert plan.is_complete_for_real_trade is True
    assert plan.missing_required_fields == []
    assert plan.exit_plan_id.startswith("pfep-")
    assert plan.partial_exit_plan
    assert plan.full_exit_plan["take_profit_pct"] == pytest.approx(0.05)
    assert set(plan.model_sources) >= {
        "decision_llm",
        "local_ml",
        "server_profit",
        "timeseries",
        "expert_alignment",
    }


def test_missing_profit_and_exit_fields_force_shadow_only() -> None:
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="ETH/USDT",
        action=Action.SHORT,
        confidence=0.58,
        reasoning="weak candidate",
        position_size_pct=0.01,
        suggested_leverage=2.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        raw_response={"analysis_type": "entry_candidate"},
    )

    plan = build_profit_first_trade_plan(decision)

    assert plan.decision_lane == "shadow_only"
    assert plan.is_complete_for_real_trade is False
    assert "expected_net_return_pct" in plan.missing_required_fields
    assert "exit_plan_id" in plan.missing_required_fields
    assert plan.no_entry_reason == "shadow_only_missing_plan_fields"


def test_attach_profit_first_trade_plan_writes_raw_response_snapshot() -> None:
    decision = _complete_decision()

    raw = attach_profit_first_trade_plan(decision)

    assert raw is decision.raw_response
    assert raw["profit_first_trade_plan"]["decision_lane"] == "meaningful_entry"
    assert raw["profit_first_trade_plan"]["is_complete_for_real_trade"] is True
    assert raw["profit_first_exit_plan"]["exit_plan_id"].startswith("pfep-")
    assert raw["profit_first_entry_exit_binding"]["exit_decisions_must_reference_plan"] is True


def test_no_entry_reason_taxonomy_uses_canonical_categories() -> None:
    assert (
        normalize_no_entry_reason(
            {"opportunity_score": {"expected_net_return_pct": -0.04}}
        )
        == "profit_insufficient"
    )
    assert (
        normalize_no_entry_reason({}, execution_reason="OKX rejected entry order")
        == "okx_unavailable_or_rejected"
    )
    assert (
        normalize_no_entry_reason({}, execution_reason="same_side crowded cap")
        == "same_side_crowded"
    )
    assert (
        normalize_no_entry_reason({}, plan_missing_fields=["expected_net_return_pct"])
        == "shadow_only_missing_plan_fields"
    )


def test_losing_exit_attribution_taxonomy_covers_common_loss_modes() -> None:
    base = SimpleNamespace(
        side="long",
        realized_pnl=-0.7,
        created_at=datetime(2026, 6, 29, 8, 0, tzinfo=UTC),
        closed_at=datetime(2026, 6, 29, 8, 3, tzinfo=UTC),
    )

    assert (
        normalize_losing_exit_attribution(
            base,
            shadow={"best_action": "short"},
        )
        == "entry_wrong_direction"
    )
    assert (
        normalize_losing_exit_attribution(
            SimpleNamespace(
                side="long",
                realized_pnl=-0.1,
                created_at=datetime(2026, 6, 29, 8, 0, tzinfo=UTC),
                closed_at=datetime(2026, 6, 29, 8, 20, tzinfo=UTC),
            ),
            entry_raw={"profit_risk_sizing": {"position_size_pct": 0.01}},
        )
        == "position_too_small_fee_drag"
    )
    assert (
        normalize_losing_exit_attribution(
            SimpleNamespace(side="short", realized_pnl=-1.0, reason="capital release rotation")
        )
        == "capital_release_forced_loss"
    )


def test_probe_loop_health_flags_recent_all_loss_probe_window() -> None:
    now = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
    rows = [
        SimpleNamespace(
            realized_pnl=-0.4,
            closed_at=now - timedelta(minutes=20),
            created_at=now - timedelta(minutes=30),
            position_size_pct=0.01,
        ),
        SimpleNamespace(
            realized_pnl=-0.3,
            closed_at=now - timedelta(minutes=5),
            created_at=now - timedelta(minutes=12),
            raw_llm_response={
                "profit_first_trade_plan": {"decision_lane": "tiny_probe"}
            },
        ),
    ]

    summary = summarize_probe_loop_health(rows, now=now, window_hours=8)

    assert summary["probe_closed_count"] == 2
    assert summary["probe_loss_count"] == 2
    assert summary["all_recent_probes_losing"] is True
    assert summary["recommended_action"] == "shadow_new_tiny_probes_until_validated_upgrade"


def test_model_strategy_realized_pnl_leaderboard_uses_profit_first_dimensions() -> None:
    rows = [
        SimpleNamespace(
            model_name="ensemble_trader",
            symbol="BTC/USDT",
            side="long",
            realized_pnl=2.0,
            raw_llm_response={
                "profit_first_trade_plan": {
                    "strategy_profile_id": "balanced_probe",
                    "decision_lane": "validated_probe",
                }
            },
        ),
        SimpleNamespace(
            model_name="ensemble_trader",
            symbol="BTC/USDT",
            side="long",
            realized_pnl=-1.0,
            raw_llm_response={
                "profit_first_trade_plan": {
                    "strategy_profile_id": "balanced_probe",
                    "decision_lane": "validated_probe",
                }
            },
        ),
    ]

    leaderboard = summarize_model_strategy_realized_pnl(rows)

    assert leaderboard["count"] == 1
    row = leaderboard["rows"][0]
    assert row["strategy_profile_id"] == "balanced_probe"
    assert row["decision_lane"] == "validated_probe"
    assert row["realized_net_pnl"] == pytest.approx(1.0)
    assert row["profit_factor"] == pytest.approx(2.0)
