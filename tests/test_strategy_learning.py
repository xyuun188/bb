from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx

from services.strategy_learning import (
    AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS,
    StrategyLearningEngine,
    StrategyLearningService,
    StrategyLearningStateStore,
    StrategyProfile,
)
from services.trading_params import DEFAULT_TRADING_PARAMS

COMPLETED_EXPERT_TIMINGS = [
    {"name": "trend_expert", "status": "completed", "provider_model": "qwen", "seconds": 1.2},
    {"name": "momentum_expert", "status": "completed", "provider_model": "qwen", "seconds": 1.1},
    {"name": "sentiment_expert", "status": "completed", "provider_model": "qwen", "seconds": 1.0},
    {"name": "position_expert", "status": "completed", "provider_model": "qwen", "seconds": 1.3},
    {"name": "risk_expert", "status": "completed", "provider_model": "qwen", "seconds": 1.4},
]


FALLBACK_EXPERT_TIMINGS = [
    {"name": "trend_expert", "status": "completed", "provider_model": "qwen"},
    {"name": "momentum_expert", "status": "completed", "provider_model": "qwen"},
    {
        "name": "sentiment_expert",
        "status": "partial_batch_fallback",
        "provider_model": "qwen",
    },
    {"name": "position_expert", "status": "completed", "provider_model": "qwen"},
    {"name": "risk_expert", "status": "completed", "provider_model": "qwen"},
]


def _position(
    *,
    side: str,
    pnl: float,
    created_hours_ago: float = 5.0,
    closed_hours_ago: float = 1.0,
    position_id: int = 1,
    entry_raw: dict[str, Any] | None = None,
) -> SimpleNamespace:

    now = datetime.now(UTC)

    return SimpleNamespace(
        id=position_id,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="BTC/USDT",
        side=side,
        realized_pnl=pnl,
        unrealized_pnl=0.0,
        created_at=now - timedelta(hours=created_hours_ago),
        closed_at=now - timedelta(hours=closed_hours_ago),
        entry_raw=entry_raw,
    )


def _profit_first_entry_raw(
    *,
    side: str,
    strategy: str = "balanced_probe",
    source: str = "server_profit",
) -> dict[str, Any]:
    return {
        "profit_first_trade_plan": {
            "plan_version": "profit-first-v3.1",
            "symbol": "BTC/USDT",
            "side": side,
            "action": side,
            "strategy_profile_id": strategy,
            "decision_lane": "validated_probe",
            "expected_net_return_pct": 0.8,
            "position_size_pct": 0.04,
            "exit_plan_id": f"pfep-{side}",
            "model_sources": ["decision_llm", source],
            "model_contributions": [
                {"source": "decision_llm", "field_path": "decision.model_name"},
                {
                    "source": source,
                    "field_path": "opportunity_score.expected_net_return_pct",
                },
            ],
        },
        "profit_first_exit_plan": {"exit_plan_id": f"pfep-{side}"},
    }


def test_strategy_learning_expert_integrity_reads_duration_sec() -> None:
    compiler = StrategyLearningEngine().compiler
    timings = [
        {**item, "duration_sec": item.pop("seconds")}
        for item in (dict(row) for row in COMPLETED_EXPERT_TIMINGS)
    ]

    fallback, zero_second, missing = compiler._expert_integrity_flags({"model_timings": timings})

    assert fallback is False
    assert zero_second is False
    assert missing == []


def test_strategy_learning_excludes_untrusted_closed_position_facts() -> None:
    compiler = StrategyLearningEngine().compiler
    now = datetime.now(UTC)
    trusted = _position(side="long", pnl=3.0, position_id=101)
    trusted.entry_exchange_order_id = "entry-ok"
    trusted.close_exchange_order_id = "close-ok"
    dirty = _position(side="short", pnl=12.0, position_id=102)
    dirty.created_at = now - timedelta(minutes=30)
    dirty.closed_at = now
    dirty.entry_exchange_order_id = "entry-dirty"
    dirty.close_exchange_order_id = ""

    feedback = compiler.compile(
        mode="paper",
        window_hours=24,
        positions=[trusted, dirty],
        open_positions=[],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        reflections=[],
    ).to_dict()

    assert feedback["totals"]["closed_trade_count"] == 2
    assert feedback["totals"]["training_trade_count"] == 1
    assert feedback["totals"]["net_pnl"] == 3.0
    assert feedback["trade_fact_quarantine"]["excluded_position_count"] == 1
    assert feedback["trade_fact_quarantine"]["reason_counts"] == {
        "missing_close_exchange_order_id": 1
    }


def test_strategy_learning_deduplicates_authoritative_closed_position_pairs() -> None:
    compiler = StrategyLearningEngine().compiler
    first = _position(side="short", pnl=-0.7, position_id=201)
    first.symbol = "GOOGL/USDT"
    first.entry_exchange_order_id = "entry-dup"
    first.close_exchange_order_id = "close-dup"
    second = _position(side="short", pnl=-0.7, position_id=202)
    second.symbol = "GOOGL/USDT"
    second.entry_exchange_order_id = "entry-dup"
    second.close_exchange_order_id = "close-dup"

    feedback = compiler.compile(
        mode="paper",
        window_hours=24,
        positions=[first, second],
        open_positions=[],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(position_id=201, pnl=-0.7),
            _reflection(position_id=202, pnl=-0.7),
        ],
    ).to_dict()

    assert feedback["totals"]["closed_trade_count"] == 2
    assert feedback["totals"]["training_trade_count"] == 1
    assert feedback["totals"]["net_pnl"] == -0.7
    assert feedback["side_performance"]["short"]["losses"] == 1
    quarantine = feedback["trade_fact_quarantine"]
    assert quarantine["duplicate_position_count"] == 1
    assert quarantine["duplicate_group_count"] == 1
    assert quarantine["duplicate_position_ids"] == [201]
    assert feedback["reflection_feedback"]["training_count"] == 1


def _open_position(symbol: str, side: str, pnl: float) -> dict[str, Any]:

    return {
        "model_name": "ensemble_trader",
        "symbol": symbol,
        "side": side,
        "unrealized_pnl": pnl,
    }


def _open_position_with_okx_time(symbol: str, side: str, hours: float) -> dict[str, Any]:
    opened_at = datetime.now(UTC) - timedelta(hours=hours)
    return {
        "model_name": "ensemble_trader",
        "symbol": symbol,
        "side": side,
        "entry_price": 100.0,
        "current_price": 99.95,
        "quantity": 10.0,
        "unrealized_pnl": -0.5,
        "info": {"cTime": str(int(opened_at.timestamp() * 1000))},
    }


def _old_flat_open_position(symbol: str, side: str = "long") -> dict[str, Any]:

    return {
        "model_name": "ensemble_trader",
        "symbol": symbol,
        "side": side,
        "entry_price": 100.0,
        "current_price": 100.02,
        "quantity": 10.0,
        "unrealized_pnl": 0.05,
        "created_at": datetime.now(UTC) - timedelta(hours=14),
        "strategy_profile_id": "old_profile",
    }


def _decision(action: str, *, executed: bool = False, reason: str = "") -> SimpleNamespace:

    return SimpleNamespace(
        action=action,
        analysis_type="market",
        was_executed=executed,
        execution_reason=reason,
        raw_llm_response={"model_timings": FALLBACK_EXPERT_TIMINGS},
    )


def _healthy_decision(action: str, *, executed: bool = False) -> SimpleNamespace:

    return SimpleNamespace(
        action=action,
        analysis_type="market",
        was_executed=executed,
        execution_reason="",
        created_at=datetime.now(UTC),
        raw_llm_response={"model_timings": COMPLETED_EXPERT_TIMINGS},
    )


def _no_entry_decision(
    action: str = "long",
    *,
    reason: str = "profit_insufficient",
    shadow_return_pct: float = 1.2,
    missed_opportunity_count: int = 6,
) -> SimpleNamespace:

    raw = _profit_first_entry_raw(side="long" if action == "long" else "short")
    raw["shadow_outcome"] = {"shadow_return_pct": shadow_return_pct}
    raw["review_feedback"] = {"missed_opportunity_count": missed_opportunity_count}
    raw["no_entry_reason"] = reason
    raw["model_timings"] = COMPLETED_EXPERT_TIMINGS
    return SimpleNamespace(
        action=action,
        analysis_type="market",
        was_executed=False,
        execution_reason=reason,
        created_at=datetime.now(UTC),
        raw_llm_response=raw,
    )


def _strategy_event(
    event_type: str,
    *,
    status: str = "recorded",
    profile_id: str = "balanced_probe",
    reason: str = "",
    exclude: bool = False,
    order_id: int | None = None,
    position_id: int | None = None,
    action: str = "long",
    attribution: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> SimpleNamespace:

    return SimpleNamespace(
        id=1,
        created_at=created_at or datetime.now(UTC),
        event_type=event_type,
        event_status=status,
        severity="warn" if status in {"blocked", "failed", "rejected"} else "info",
        symbol="BTC/USDT",
        side="long",
        action=action,
        order_id=order_id,
        position_id=position_id,
        profile_id=profile_id,
        reason=reason,
        attribution=attribution if attribution is not None else {"blocker": reason},
        exclude_from_training=exclude,
    )


def _reflection(
    *,
    position_id: int,
    pnl: float,
    hold_minutes: float = 60.0,
    outcome: str | None = None,
    mistake: str = "",
    improvement: str = "",
    source: str = "system",
) -> SimpleNamespace:

    return SimpleNamespace(
        id=position_id,
        position_id=position_id,
        closed_at=datetime.now(UTC),
        symbol="BTC/USDT",
        side="long",
        realized_pnl=pnl,
        fee_estimate=0.08,
        hold_minutes=hold_minutes,
        outcome=outcome or ("win" if pnl > 0 else "loss" if pnl < 0 else "flat"),
        mistake_summary=mistake,
        improvement_summary=improvement,
        source=source,
    )


def _missed_shadow(symbol: str, side: str, return_pct: float) -> SimpleNamespace:
    return SimpleNamespace(
        status="completed",
        missed_opportunity=True,
        decision_action="hold",
        symbol=symbol,
        best_action=side,
        long_return_pct=return_pct if side == "long" else -0.2,
        short_return_pct=return_pct if side == "short" else -0.2,
        feature_snapshot={
            "market_structure": "trend_up_momentum" if side == "long" else "trend_down_momentum",
            "loss_probability": 0.34,
            "tail_risk_score": 0.52,
            "volume_ratio": 1.7,
            "adx": 31.0,
        },
        raw_llm_response={"ml_signal": {"best_side": side, "confidence": 0.72}},
        horizon_minutes=10,
    )


def test_strategy_learning_global_missed_count_does_not_select_probe_profile(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(
                side="long" if index % 2 == 0 else "short",
                pnl=0.8,
                position_id=index + 100,
            )
            for index in range(80)
        ],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("hold") for _ in range(20)],
        shadows=[_missed_shadow(f"SYM{index}/USDT", "long", 0.8) for index in range(8)],
        memories=[],
        max_open_positions=14,
    )

    feedback = payload["feedback"]
    closed_loop = feedback["shadow_feedback"]["missed_opportunity_closed_loop"]
    problem_keys = {item["key"] for item in feedback["problems"]}

    assert feedback["shadow_feedback"]["missed_opportunity_count"] == 8
    assert closed_loop["global_missed_count_can_drive_entries"] is False
    assert closed_loop["summary"]["probe_count"] == 0
    assert closed_loop["summary"]["adopted_count"] == 0
    assert "missed_opportunities" not in problem_keys
    assert payload["schedule"]["active_profile"]["id"] == "baseline_current"


def test_strategy_learning_uses_qualified_missed_loop_for_probe_feedback(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(
                side="long" if index % 2 == 0 else "short",
                pnl=0.8,
                position_id=index + 300,
            )
            for index in range(80)
        ],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("hold") for _ in range(20)],
        shadows=[
            _missed_shadow("BTC/USDT", "long", 0.44),
            _missed_shadow("BTC/USDT", "long", 0.52),
            _missed_shadow("BTC/USDT", "long", 0.58),
        ],
        memories=[],
        max_open_positions=14,
    )

    feedback = payload["feedback"]
    closed_loop = feedback["shadow_feedback"]["missed_opportunity_closed_loop"]
    problem_keys = {item["key"] for item in feedback["problems"]}
    validation_row = next(
        row
        for row in payload["schedule"]["shadow_validation"]["rows"]
        if row["profile_id"] == "balanced_probe"
    )

    assert closed_loop["summary"]["probe_count"] == 1
    assert closed_loop["usable_group_count"] == 1
    assert "missed_opportunities" in problem_keys
    assert validation_row["missed_opportunities_used"] == 1
    assert validation_row["missed_opportunity_raw_count"] == 3
    assert validation_row["missed_opportunity_closed_loop"]["usable_group_count"] == 1
    assert payload["schedule"]["probe"]["closed_loop_probe_rules"]


def test_strategy_learning_builds_full_feedback_and_schedules_loss_release(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=-4.0, position_id=1),
            _position(side="long", pnl=-5.0, position_id=2),
            _position(side="long", pnl=-3.0, position_id=3),
            _position(side="short", pnl=2.5, position_id=4),
        ],
        open_positions=[
            _open_position("BTC/USDT", "long", -8.0),
            _open_position("ETH/USDT", "long", -3.0),
            _open_position("SOL/USDT", "short", 1.5),
        ],
        orders=[],
        decisions=[
            _decision("long", reason="专家分析完整性保护[expert_integrity]"),
            _decision("hold"),
        ],
        shadows=[
            SimpleNamespace(
                status="completed",
                missed_opportunity=True,
                decision_action="hold",
                long_return_pct=0.8,
                short_return_pct=-0.2,
            )
        ],
        memories=[SimpleNamespace(is_active=True, memory_type="shadow_missed_opportunity")],
        reflections=[
            _reflection(
                position_id=1,
                pnl=-4.2,
                hold_minutes=260.0,
                mistake="亏损仓拖延过久",
                improvement="满仓时优先释放低质量亏损仓",
            )
        ],
        max_open_positions=3,
    )

    feedback = payload["feedback"]

    schedule = payload["schedule"]

    problem_keys = {item["key"] for item in feedback["problems"]}

    assert "negative_realized_pnl" in problem_keys

    assert "long_side_degraded" in problem_keys

    assert "full_position_loss_pressure" in problem_keys

    assert feedback["training_policy"]["manual_close_excluded"] is True

    assert feedback["reflection_feedback"]["training_count"] == 1

    assert feedback["reflection_feedback"]["avg_loss_hold_minutes"] == 260.0

    assert "reflection_loss_hold_too_long" in problem_keys

    assert schedule["active_profile"]["id"] == "loss_release"

    assert schedule["runtime"]["full_position_release"] is True

    assert schedule["runtime"]["rotation_slots"] >= 1
    assert (
        schedule["runtime"]["target_position_groups"]
        > feedback["open_position_pressure"]["open_group_count"]
    )
    assert schedule["runtime"]["release_target_groups"] >= 1
    assert (
        schedule["runtime"]["position_review_max_groups"]
        >= feedback["open_position_pressure"]["open_group_count"]
    )
    assert (
        schedule["runtime"]["analysis_budget"]["position_max_groups"]
        == schedule["runtime"]["position_review_max_groups"]
    )

    assert schedule["backtest"]["rows"]

    assert schedule["shadow_validation"]["rows"]

    assert schedule["probe"]["small_position_first"] is True


def test_strategy_learning_context_applies_profile_overrides(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="专家分析完整性保护[expert_integrity]")],
        shadows=[
            SimpleNamespace(
                status="completed",
                missed_opportunity=True,
                decision_action="hold",
                long_return_pct=0.9,
                short_return_pct=-0.1,
            )
        ],
        memories=[],
        max_open_positions=14,
    )

    context = engine.apply_to_context(
        {"min_opportunity_score": 1.0, "side_quality": {}},
        payload,
    )

    assert context["strategy_profile_id"] == "balanced_probe"

    assert context["min_opportunity_score"] < 1.0

    assert context["expert_integrity_mode"] == "balanced_probe_allow_one_non_core_missing"

    assert (
        context["position_size_multiplier"]
        == DEFAULT_TRADING_PARAMS.entry_risk_sizing.balanced_probe_position_size_multiplier
    )

    assert context["probe_fraction"] == 0.08

    assert (
        context["max_probe_size_pct"]
        == DEFAULT_TRADING_PARAMS.entry_risk_sizing.balanced_probe_max_position_size_pct
    )

    assert context["strategy_learning_sizing"]["profile_id"] == "balanced_probe"

    assert context["strategy_learning"]["low_trade_count_penalized"] is True


def test_strategy_learning_applies_profit_first_runtime_side_feedback(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    positions = [
        *[
            _position(
                side="long",
                pnl=1.2,
                position_id=100 + idx,
                entry_raw=_profit_first_entry_raw(side="long"),
            )
            for idx in range(4)
        ],
        *[
            _position(
                side="short",
                pnl=-3.0,
                position_id=200 + idx,
                entry_raw=_profit_first_entry_raw(
                    side="short",
                    strategy="short_loss_loop",
                    source="timeseries",
                ),
            )
            for idx in range(4)
        ],
    ]

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=positions,
        open_positions=[],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        max_open_positions=20,
    )
    context = engine.apply_to_context({"min_opportunity_score": 1.0}, payload)

    feedback = payload["feedback"]["profit_first_runtime_feedback"]
    assert feedback["objective"] == "maximize_realized_net_pnl"
    assert feedback["side_feedback"]["short"]["recommended_stage"] == "demote"
    assert feedback["side_feedback"]["short"]["hard_ban"] is False
    assert context["profit_first_runtime_feedback_applied"] is True
    assert context["side_weights"]["short"] < 1.0
    assert context["side_weights"]["long"] >= 1.0
    assert context["profit_first_runtime_feedback"]["exit_plan_reference"]["missing_count"] == 0
    assert context["strategy_learning_sizing"]["profit_first_runtime_feedback_applied"] is True


def test_strategy_learning_profit_first_runtime_guidance_drives_candidates_and_runtime(
    tmp_path,
) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    early_exit = _position(
        side="long",
        pnl=-0.9,
        position_id=301,
        entry_raw=_profit_first_entry_raw(side="long"),
    )
    early_exit.reason = "early exit after shallow noise"
    tiny_probe = _position(
        side="long",
        pnl=-0.4,
        position_id=302,
        entry_raw=_profit_first_entry_raw(side="long"),
    )
    tiny_probe.entry_raw["profit_first_trade_plan"]["position_size_pct"] = 0.01

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[early_exit, tiny_probe],
        open_positions=[],
        orders=[],
        decisions=[
            _no_entry_decision("long"),
            _no_entry_decision("long", shadow_return_pct=0.9, missed_opportunity_count=8),
            _no_entry_decision("long", shadow_return_pct=1.5, missed_opportunity_count=7),
        ],
        shadows=[],
        memories=[],
        max_open_positions=20,
    )
    context = engine.apply_to_context({"min_opportunity_score": 1.0}, payload)
    candidate_ids = {
        row["id"] for row in payload["schedule"]["candidates"] if isinstance(row, dict)
    }
    profit_first_context = context["strategy_learning"]["profit_first_context"]

    assert "quality_entry_recovery" in candidate_ids
    assert "winner_hold" in candidate_ids
    assert profit_first_context["missed_opportunity_feedback"]["diagnosis"] == (
        "system_over_conservative_review"
    )
    assert "missed_positive_shadow_relaxed_entry_reference" in profit_first_context[
        "applied_reasons"
    ]
    assert "exit_feedback_requests_longer_winner_hold" in profit_first_context["applied_reasons"]
    assert context["winner_hold_extension"] == "high"
    assert context["profit_lock_min_usdt_multiplier"] > 1.0
    assert context["strategy_learning_sizing"]["profit_first_context"]["applied_reasons"]


def test_strategy_learning_low_quality_open_positions_trigger_loss_release(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    old_flat_position = _old_flat_open_position("LOW/USDT")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[
            old_flat_position,
            _old_flat_open_position("LOW2/USDT"),
            _old_flat_open_position("LOW3/USDT"),
        ],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        max_open_positions=20,
    )
    context = engine.apply_to_context({}, payload)

    assert payload["feedback"]["open_position_pressure"]["low_quality_open_count"] == 3
    assert payload["schedule"]["active_profile"]["id"] == "loss_release"
    assert context["full_position_release"] is True
    assert context["release_losing_positions_first"] is True
    assert context["loss_exit_aggressiveness"] == "high"
    assert context["position_review_priority_boost"] >= 1.35
    assert context["strategy_learning_release_pressure_active"] is True
    assert context["portfolio_roster"]["policy_source"] == "strategy_learning_runtime"
    assert (
        context["target_position_groups"] == context["portfolio_roster"]["target_position_groups"]
    )
    assert context["position_review_max_groups"] >= 1
    assert (
        context["strategy_learning"]["runtime"]["analysis_budget"]["roster_fill_market_symbol_min"]
        >= 6
    )


def test_strategy_learning_single_low_quality_open_position_does_not_global_release(
    tmp_path,
) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[_old_flat_open_position("LOW/USDT")],
        orders=[],
        decisions=[_healthy_decision("long", executed=True)],
        shadows=[],
        memories=[],
        max_open_positions=20,
    )
    context = engine.apply_to_context({}, payload)

    pressure = payload["feedback"]["open_position_pressure"]
    assert pressure["low_quality_open_count"] == 1
    assert payload["schedule"]["active_profile"]["id"] != "loss_release"
    assert context["strategy_learning_release_pressure_active"] is False
    assert context["strategy_learning_sizing"].get("release_pressure_active") is not True


def test_strategy_learning_runtime_keeps_roster_fill_candidate_floor(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    open_positions = [
        SimpleNamespace(
            model_name="ensemble_trader",
            symbol=f"OPEN{i}/USDT",
            side="long",
            is_open=True,
            created_at=None,
            opened_at=None,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            quantity=1.0,
            entry_price=1.0,
            current_price=1.0,
        )
        for i in range(7)
    ]

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=open_positions,
        orders=[],
        decisions=[_healthy_decision("long", executed=True)],
        shadows=[],
        memories=[],
        max_open_positions=12,
    )
    context = engine.apply_to_context({}, payload)
    runtime_budget = context["strategy_learning"]["runtime"]["analysis_budget"]

    assert context["target_position_groups"] > 7
    assert context["portfolio_roster"]["policy_source"] == "strategy_learning_runtime"
    assert runtime_budget["roster_fill_market_symbol_min"] >= 6


def test_strategy_learning_structured_candidate_preferences_are_consumed(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    state_store.set_manual_active_profile("llm_structured_preference")

    llm_profile = StrategyProfile(
        profile_id="llm_structured_preference",
        version=1,
        label="结构偏好",
        status="candidate",
        source="llm_structured_candidate",
        description="用结构化偏好扩大容量并优化进退场。",
        params={
            "global_min_score_delta": -0.03,
            "position_size_multiplier": 1.02,
            "entry_filters": {
                "quality_bias": "expand",
                "missed_opportunity_bias": "relax",
                "volume_ratio_multiplier": 0.9,
                "adx_multiplier": 0.92,
            },
            "portfolio_preference": {
                "capacity_mode": "expand",
                "target_open_bias": 1.2,
                "rotation_bias": 1.4,
                "roster_fill_bias": 1.3,
                "review_bias": 1.15,
            },
            "exit_preference": {
                "winner_mode": "let_run",
                "loser_mode": "cut_faster",
                "profit_lock_bias": 1.15,
                "review_priority_bias": 1.2,
                "loss_exit_bias": 1.25,
            },
        },
    )

    open_positions = [
        _old_flat_open_position("LOW1/USDT"),
        _old_flat_open_position("LOW2/USDT"),
        _old_flat_open_position("LOW3/USDT"),
    ]

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-2.0, position_id=601)],
        open_positions=open_positions,
        orders=[],
        decisions=[_healthy_decision("long", executed=True)],
        shadows=[
            _missed_shadow("BTC/USDT", "long", 0.9),
            _missed_shadow("ETH/USDT", "long", 0.8),
        ],
        memories=[],
        strategy_events=[
            _strategy_event("capacity_block", status="blocked", reason="max_position capacity")
        ],
        reflections=[
            _reflection(position_id=602, pnl=-1.4, hold_minutes=220.0, mistake="loss hold too long")
        ],
        max_open_positions=8,
        extra_profiles=[llm_profile],
    )
    context = engine.apply_to_context({}, payload)

    active = next(
        row
        for row in payload["schedule"]["candidates"]
        if row["id"] == "llm_structured_preference"
    )
    backtest = next(
        row
        for row in payload["schedule"]["backtest"]["rows"]
        if row["profile_id"] == "llm_structured_preference"
    )
    shadow = next(
        row
        for row in payload["schedule"]["shadow_validation"]["rows"]
        if row["profile_id"] == "llm_structured_preference"
    )

    assert active["consumed_runtime_params"] == [
        "entry_filters",
        "exit_preference",
        "global_min_score_delta",
        "portfolio_preference",
        "position_size_multiplier",
    ]
    assert "portfolio_capacity_reallocation" in backtest["matched_fixes"]
    assert shadow["would_increase_entries"] is True
    assert shadow["would_release_losers"] is True
    assert shadow["would_hold_winners"] is True
    assert payload["schedule"]["active_profile"]["id"] == "llm_structured_preference"
    assert context["entry_filter_preference"]["quality_bias"] == "expand"
    assert context["portfolio_roster"]["preference"]["capacity_mode"] == "expand"
    assert context["strategy_learning"]["structured_params"]["exit_preference"]["winner_mode"] == (
        "let_run"
    )
    assert context["winner_hold_extension"] == "high"
    assert context["loss_exit_aggressiveness"] == "high"
    assert context["target_position_groups"] >= 4
    assert context["strategy_learning"]["runtime"]["analysis_budget"]["roster_fill_market_symbol_min"] >= 6
    assert "entry_preference_expand_quality_entries" in context["entry_filters"]["reason"]
    assert "portfolio_preference_expand_capacity" in context["portfolio_roster"]["policy_reason"]


def test_strategy_learning_historical_capacity_blocks_do_not_trigger_release_without_current_pressure(
    tmp_path,
) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-8.0, position_id=91)],
        open_positions=[
            {
                "model_name": "ensemble_trader",
                "symbol": "OP/USDT",
                "side": "long",
                "entry_price": 100.0,
                "current_price": 101.0,
                "quantity": 1.0,
                "unrealized_pnl": 1.0,
                "created_at": datetime.now(UTC) - timedelta(minutes=30),
            }
        ],
        orders=[],
        decisions=[_healthy_decision("hold")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "capacity_block",
                status="blocked",
                reason="max_position capacity reached earlier",
            ),
        ],
        max_open_positions=20,
    )
    context = engine.apply_to_context({}, payload)

    pressure = payload["feedback"]["open_position_pressure"]
    problem_keys = {item["key"] for item in payload["feedback"]["problems"]}

    assert "max_position_blocks" in problem_keys
    assert pressure["full_position_pressure"] is False
    assert pressure["fragmentation_pressure"] is False
    assert pressure["low_quality_open_count"] == 0
    assert payload["schedule"]["active_profile"]["id"] != "loss_release"
    assert context["strategy_learning_release_pressure_active"] is False
    assert context["strategy_learning_sizing"].get("release_pressure_active") is not True
    assert context["strategy_learning_release_pressure_detail"]["current_pressure"] is False
    assert (
        context["strategy_learning_release_pressure_detail"]["policy"]
        == "current_position_pressure_only"
    )


def test_llm_candidate_status_exposes_sanitized_cached_candidates(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    service = StrategyLearningService(state_store=state_store)

    strategy_feedback = service._compile_feedback(
        mode="paper",
        hours=168,
        rows={
            "closed_positions": [],
            "open_positions": [],
            "orders": [],
            "decisions": [_decision("long", reason="expert_integrity fallback")],
            "shadows": [],
            "memories": [],
            "strategy_events": [],
            "reflections": [],
        },
        open_positions=[],
        max_open_positions=14,
    )

    signature = service._feedback_signature(strategy_feedback)

    state_store.save(
        {
            "llm_candidate_cache": {
                "signature": signature,
                "generated_at": "2026-06-13T00:00:00+00:00",
                "candidates": [
                    {
                        "id": "candidate_1",
                        "label": "减仓探针",
                        "description": "降低仓位并启用探针",
                        "params": {"probe_fraction": 0.05},
                    }
                ],
                "model": "qwen3-14b-trade",
                "source": "llm_structured_candidate",
            }
        }
    )

    status = service._llm_candidate_status(strategy_feedback)

    assert status["cache_matches_feedback"] is True

    assert status["cache_status"] == "current"

    assert status["cached_candidates"][0]["label"] == "减仓探针"

    assert "探针" in status["cached_candidates"][0]["description"]


def test_strategy_learning_compiles_events_and_full_score_metrics(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=-4.0, position_id=10),
            _position(side="short", pnl=2.0, position_id=11),
            _position(side="short", pnl=-1.0, position_id=12),
        ],
        open_positions=[_open_position("BTC/USDT", "long", -5.0)],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[
            SimpleNamespace(
                status="completed",
                missed_opportunity=True,
                decision_action="hold",
                long_return_pct=0.9,
                short_return_pct=-0.2,
            )
        ],
        memories=[],
        strategy_events=[
            _strategy_event(
                "capacity_block",
                status="blocked",
                reason="max_position capacity reached",
            ),
            _strategy_event(
                "manual_close",
                status="executed",
                reason="user manual close",
                exclude=True,
                order_id=88,
                position_id=12,
            ),
        ],
        reflections=[],
        max_open_positions=1,
    )

    feedback = payload["feedback"]

    event_feedback = feedback["event_feedback"]

    problem_keys = {item["key"] for item in feedback["problems"]}

    assert event_feedback["max_position_blocks"] == 1

    assert event_feedback["manual_close_events"] == 1

    assert event_feedback["attribution_coverage"] == 1.0

    manual_event = next(
        row for row in event_feedback["recent_events"] if row["event_type"] == "manual_close"
    )

    assert manual_event["order_id"] == 88

    assert manual_event["position_id"] == 12

    assert manual_event["exclude_from_training"] is True

    assert "max_position_blocks" in problem_keys

    assert payload["schedule"]["active_profile"]["id"] == "loss_release"

    score = payload["schedule"]["backtest"]["rows"][0]

    assert "fee_adjusted_pnl" in score

    assert "max_drawdown" in score

    assert "consecutive_losses" in score

    assert "position_occupancy" in score


def test_strategy_learning_defensive_probe_shadow_loop_schedules_quality_recovery(
    tmp_path,
) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("long", executed=False) for _ in range(12)],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "execution_result",
                status="skipped",
                reason="Profit-First 防御探针拦截：该极小/探针开仓属于低收益质量。",
                attribution={
                    "skip_kind": "profit_first_defensive_probe_shadow",
                    "blocker": "profit_first_defensive_probe_shadow",
                },
            )
            for _ in range(4)
        ],
        max_open_positions=14,
    )
    context = engine.apply_to_context({}, payload)
    feedback = payload["feedback"]
    event_feedback = feedback["event_feedback"]
    problem_keys = {item["key"] for item in feedback["problems"]}
    schedule = payload["schedule"]
    backtest = next(
        row
        for row in schedule["backtest"]["rows"]
        if row["profile_id"] == "quality_entry_recovery"
    )
    shadow = next(
        row
        for row in schedule["shadow_validation"]["rows"]
        if row["profile_id"] == "quality_entry_recovery"
    )

    assert event_feedback["profit_first_defensive_probe_shadow_count"] == 4
    assert event_feedback["skip_kind_counts"]["profit_first_defensive_probe_shadow"] == 4
    assert "defensive_probe_shadow_loop" in problem_keys
    assert schedule["active_profile"]["id"] == "quality_entry_recovery"
    assert "defensive_probe_quality_recovery" in backtest["matched_fixes"]
    assert shadow["eligible"] is True
    assert shadow["would_restore_quality_entries"] is True
    assert shadow["probe_required"] is False
    assert context["strategy_profile_id"] == "quality_entry_recovery"
    assert context["probe_fraction"] == 0.0
    assert context["max_probe_size_pct"] == 0.0
    assert context["strategy_learning_sizing"]["quality_entry_recovery_active"] is True
    assert context["strategy_learning_sizing"]["probe_fraction"] == 0.0
    assert context["strategy_learning_sizing"]["max_probe_size_pct"] == 0.0


def test_strategy_learning_groups_fragmented_open_positions_by_symbol_and_side(
    tmp_path,
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[
            _open_position("ARB/USDT", "long", -1.0),
            _open_position("ARB/USDT:USDT", "long", -2.0),
            _open_position("ARB-USDT-SWAP", "long", -3.0),
            _open_position("YGG/USDT", "long", 0.5),
        ],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        max_open_positions=3,
    )

    pressure = payload["feedback"]["open_position_pressure"]

    assert pressure["open_part_count"] == 4

    assert pressure["open_group_count"] == 2

    assert pressure["open_count"] == 2

    assert pressure["duplicate_part_count"] == 2

    assert pressure["usage_ratio"] == 0.666667

    assert pressure["part_usage_ratio"] == 1.333333

    assert pressure["full_position_pressure"] is False

    assert pressure["fragmentation_pressure"] is True

    arb = next(row for row in pressure["release_candidates"] if row["symbol_key"] == "ARB/USDT")

    assert arb["parts"] == 3

    assert arb["unrealized_pnl"] == -6.0


def test_strategy_learning_pressure_uses_exchange_open_time_for_hold_hours(
    tmp_path,
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[_open_position_with_okx_time("MSFT/USDT:USDT", "long", 6.0)],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        max_open_positions=3,
    )

    pressure = payload["feedback"]["open_position_pressure"]
    msft = next(row for row in pressure["release_candidates"] if row["symbol_key"] == "MSFT/USDT")

    assert msft["symbol"] == "MSFT/USDT:USDT"
    assert msft["position_quality"]["hold_hours"] >= 5.9
    assert msft["position_quality"]["hold_hours"] < 6.1


def test_strategy_learning_uses_trade_reflections_and_excludes_manual_samples(
    tmp_path,
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-2.0, position_id=100)],
        open_positions=[],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(
                position_id=100,
                pnl=-3.5,
                hold_minutes=240.0,
                mistake="止损太晚",
                improvement="亏损释放要更主动",
            ),
            _reflection(
                position_id=101,
                pnl=0.4,
                hold_minutes=25.0,
                mistake="小盈过早平仓",
                improvement="优势仓位允许多跑一点",
            ),
            _reflection(
                position_id=102,
                pnl=-8.0,
                hold_minutes=10.0,
                mistake="手动平仓样本",
                source="manual_close",
            ),
        ],
        max_open_positions=14,
    )

    feedback = payload["feedback"]

    reflection = feedback["reflection_feedback"]

    problem_keys = {item["key"] for item in feedback["problems"]}

    assert reflection["total_count"] == 3

    assert reflection["training_count"] == 2

    assert reflection["excluded_manual_count"] == 1

    assert reflection["fee_adjusted_pnl"] < 0

    assert reflection["loss_sample_count"] == 1

    assert reflection["win_sample_count"] == 1

    assert reflection["top_mistakes"][0]["summary"] == "止损太晚"

    compact_reflection = StrategyLearningService(
        state_store=state_store
    )._compact_reflection_feedback(reflection)

    assert compact_reflection["top_mistakes"][0] == {"summary": "止损太晚", "count": 1}

    assert compact_reflection["top_improvements"][0] == {
        "summary": "亏损释放要更主动",
        "count": 1,
    }

    assert "reflection_negative_pnl" in problem_keys

    assert "reflection_loss_hold_too_long" in problem_keys

    assert "trade_reflection_mistakes" in problem_keys

    assert payload["schedule"]["active_profile"]["id"] != "loss_release"

    assert any(
        "reflection_loss_hold_too_long" in row["matched_fixes"]
        for row in payload["schedule"]["backtest"]["rows"]
        if row["profile_id"] == "loss_release"
    )


def test_strategy_learning_normalizes_event_reasons_and_attribution_scope(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    okx_reason = (
        'okx {"code":"1","data":[{"clOrdId":"e847386590ce4dBC62e44c817548d568",'
        '"ordId":"","sCode":"51000","sMsg":"Parameter tpTriggerPx error"}]}'
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "decision_logged",
                status="recorded",
                action="hold",
                profile_id="",
                reason="market scan hold",
            ),
            _strategy_event(
                "execution_result",
                status="failed",
                profile_id="",
                reason="missing execution result",
            ),
            _strategy_event(
                "execution_result",
                status="rejected",
                profile_id="baseline_current",
                reason=okx_reason,
            ),
        ],
        max_open_positions=14,
    )

    event_feedback = payload["feedback"]["event_feedback"]

    assert event_feedback["total_events"] == 3

    assert event_feedback["attributable_events"] == 2

    assert event_feedback["non_attributable_events"] == 1

    assert event_feedback["missing_profile_events"] == 2

    assert event_feedback["attributable_missing_profile_events"] == 1

    assert event_feedback["attributable_event_coverage"] == 0.5

    reasons = event_feedback["top_block_reasons"]

    assert any(
        row["category"] == "execution_missing_result" and "交易接口未返回执行结果" in row["reason"]
        for row in reasons
    )

    assert any(
        row["category"] == "okx_execution_error" and "tpTriggerPx 无效" in row["reason"]
        for row in reasons
    )

    assert all("clOrdId" not in row["reason"] for row in reasons)

    rejected = next(
        row for row in event_feedback["recent_events"] if row["event_status"] == "rejected"
    )

    assert rejected["reason_category"] == "okx_execution_error"

    assert "tpTriggerPx 无效" in rejected["reason_label"]


def test_strategy_learning_runtime_guard_rolls_back_bad_candidate(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    service = StrategyLearningService(engine=engine, state_store=state_store)

    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=-9.0, position_id=21),
            _position(side="short", pnl=-2.0, position_id=22),
        ],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
        ],
        max_open_positions=14,
    )

    guard = service._runtime_guard(payload, mutate=True)

    state = state_store.load()

    assert guard["should_rollback"] is True

    assert state["manual_active_profile"] == ""

    assert "balanced_probe" in state["disabled_profiles"]


def test_strategy_learning_execution_guard_rolls_back_without_global_entry_pause(
    tmp_path,
) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    service = StrategyLearningService(engine=engine, state_store=state_store)
    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="entry")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
        ],
        max_open_positions=14,
    )

    guard = service._runtime_guard(payload, mutate=True)
    payload["runtime_guard"] = guard
    context = engine.apply_to_context({"min_opportunity_score": 1.0}, payload)

    assert guard["should_rollback"] is True
    assert context["strategy_learning_entry_pause"] is False
    assert context["strategy_learning_entry_pause_reason"] == ""
    assert context["strategy_learning_execution_guard_active"] is True
    assert context["strategy_learning_recovery_probe_allowed"] is True
    assert context["strategy_learning_sizing"]["execution_guard_active"] is True


def test_strategy_learning_runtime_guard_read_only_does_not_mutate_state(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    service = StrategyLearningService(engine=engine, state_store=state_store)

    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-9.0, position_id=31)],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
        ],
        max_open_positions=14,
    )

    guard = service._runtime_guard(payload, mutate=False)

    state = state_store.load()

    assert guard["should_rollback"] is True

    assert guard["mutated"] is False

    assert state["manual_active_profile"] == "balanced_probe"

    assert "balanced_probe" not in state.get("disabled_profiles", {})


def test_strategy_learning_fallback_guard_allows_recovery_probe(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    service = StrategyLearningService(engine=engine, state_store=state_store)
    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[],
        memories=[],
        max_open_positions=14,
    )

    guard = service._runtime_guard(payload, mutate=True)
    payload["runtime_guard"] = guard
    context = engine.apply_to_context({"min_opportunity_score": 1.0}, payload)
    state = state_store.load()

    assert "fallback_dependency_guard" in guard["reasons"]
    assert guard["should_rollback"] is False
    assert guard["fallback_health_guard_active"] is True
    assert guard["recovery_probe_allowed"] is True
    assert state["manual_active_profile"] == "balanced_probe"
    assert context["strategy_learning_entry_pause"] is False
    assert context["strategy_learning_health_guard_active"] is True
    assert context["strategy_learning_recovery_probe_allowed"] is True
    assert context["strategy_learning_sizing"]["recovery_probe_allowed"] is True
    assert (
        DEFAULT_TRADING_PARAMS.entry_risk_sizing.recovery_probe_min_cap_pct
        <= context["strategy_learning_sizing"]["max_probe_size_pct"]
        <= DEFAULT_TRADING_PARAMS.entry_risk_sizing.recovery_health_probe_max_cap_pct
    )
    assert "质量驱动恢复探针" in context["strategy_learning_sizing"]["reason"]


def test_strategy_learning_execution_success_clears_stale_missing_results(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    service = StrategyLearningService(engine=engine, state_store=state_store)
    state_store.set_manual_active_profile("balanced_probe")
    now = datetime.now(UTC)

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("long", executed=True)],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "execution_result",
                status="failed",
                reason="missing execution result",
                created_at=now - timedelta(hours=4),
            ),
            _strategy_event(
                "execution_result",
                status="failed",
                reason="missing execution result",
                created_at=now - timedelta(hours=3),
            ),
            _strategy_event(
                "execution_result",
                status="failed",
                reason="missing execution result",
                created_at=now - timedelta(hours=2),
            ),
            _strategy_event(
                "execution_result",
                status="executed",
                reason="OKX 平仓已全部成交。",
                created_at=now - timedelta(minutes=20),
            ),
        ],
        max_open_positions=14,
    )

    event_feedback = payload["feedback"]["event_feedback"]
    guard = service._runtime_guard(payload, mutate=True)

    assert event_feedback["execution_errors"] == 3
    assert event_feedback["execution_successes"] == 1
    assert event_feedback["unresolved_execution_errors"] == 0
    assert event_feedback["execution_recovered_after_error"] is True
    assert "execution_error_guard" not in guard["reasons"]
    assert guard["execution_recovered_after_error"] is True
    assert guard["should_rollback"] is False
    assert state_store.load()["manual_active_profile"] == "balanced_probe"


def test_strategy_learning_okx_rule_errors_do_not_hard_pause_entries(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    service = StrategyLearningService(engine=engine, state_store=state_store)
    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("long", executed=False)],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "execution_result",
                status="failed",
                reason="Order size is below OKX minimum contract size",
            ),
            _strategy_event(
                "execution_result",
                status="failed",
                reason="OKX 51008 Insufficient USDT margin",
            ),
            _strategy_event(
                "execution_result",
                status="failed",
                reason="OKX open interest has reached the platform's limit",
            ),
        ],
        max_open_positions=14,
    )

    event_feedback = payload["feedback"]["event_feedback"]
    guard = service._runtime_guard(payload, mutate=True)
    payload["runtime_guard"] = guard
    context = engine.apply_to_context({"min_opportunity_score": 1.0}, payload)

    assert event_feedback["execution_errors"] == 3
    assert event_feedback["unresolved_execution_errors"] == 3
    assert event_feedback["unresolved_execution_guard_errors"] == 0
    assert "execution_error_guard" not in guard["reasons"]
    assert guard["should_rollback"] is False
    assert context["strategy_learning_entry_pause"] is False
    assert state_store.load()["manual_active_profile"] == "balanced_probe"


def test_strategy_learning_okx_50001_is_external_transient_not_strategy_error(
    tmp_path,
) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    service = StrategyLearningService(engine=engine, state_store=state_store)
    state_store.set_manual_active_profile("balanced_probe")
    okx_50001 = (
        'Max retries exceeded: okx {"code":"50001","data":[],'
        '"msg":"Service temporarily unavailable. Please try again later."}'
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("long", executed=False)],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "execution_result",
                status="failed",
                reason=okx_50001,
            ),
        ],
        max_open_positions=14,
    )

    event_feedback = payload["feedback"]["event_feedback"]
    guard = service._runtime_guard(payload, mutate=True)
    recent = event_feedback["recent_events"][0]

    assert event_feedback["execution_errors"] == 0
    assert event_feedback["unresolved_execution_errors"] == 0
    assert event_feedback["unresolved_execution_guard_errors"] == 0
    assert recent["reason_category"] == "okx_transient_exchange_error"
    assert "交易所服务临时不可用" in recent["reason_label"]
    assert "execution_error_guard" not in guard["reasons"]
    assert guard["should_rollback"] is False


def test_strategy_learning_auto_rollback_pressure_holds_baseline(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    now = datetime.now(UTC)

    state_store.save(
        {
            "manual_active_profile": "",
            "disabled_profiles": {
                "candidate_1": {
                    "reason": "auto_runtime_guard:recent_net_pnl_guard",
                    "updated_at": now.isoformat(),
                    "auto": True,
                    "disabled_until": (now + timedelta(hours=1)).isoformat(),
                }
            },
        }
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-9.0, position_id=33)],
        open_positions=[
            _open_position("ARB/USDT", "long", -3.0),
            _open_position("YGG/USDT", "long", -1.5),
        ],
        orders=[],
        decisions=[_healthy_decision("long", executed=True)],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(
                position_id=33,
                pnl=-9.0,
                hold_minutes=260.0,
                mistake="loss hold too long",
            )
        ],
        max_open_positions=2,
    )

    schedule = payload["schedule"]

    assert schedule["active_profile"]["id"] == "loss_release"

    assert "candidate_1" in schedule["disabled_profiles"]

    assert "亏损释放画像" in schedule["reason"]


def test_strategy_learning_baseline_manual_state_uses_auto_scheduler(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    state_store.set_manual_active_profile("baseline_current")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=-4.0, position_id=41),
            _position(side="long", pnl=-5.0, position_id=42),
        ],
        open_positions=[
            _open_position("BTC/USDT", "long", -8.0),
            _open_position("ETH/USDT", "long", -3.0),
        ],
        orders=[],
        decisions=[_decision("hold")],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(
                position_id=41,
                pnl=-4.2,
                hold_minutes=260.0,
                mistake="loss hold too long",
            )
        ],
        max_open_positions=2,
    )

    assert state_store.load()["manual_active_profile"] == ""

    assert payload["schedule"]["scheduler_mode"] == "auto"

    assert payload["schedule"]["manual_profile_id"] == ""

    assert payload["schedule"]["active_profile"]["id"] == "loss_release"


def test_strategy_learning_rollback_clears_manual_lock(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    service = StrategyLearningService(state_store=state_store)

    state_store.set_manual_active_profile("balanced_probe")

    state = service.rollback_to_baseline()

    assert state["manual_active_profile"] == ""


def test_strategy_learning_auto_schedules_valid_structured_candidate(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    llm_profile = StrategyProfile(
        profile_id="llm_loss_probe",
        version=1,
        label="LLM亏损释放探针",
        status="candidate",
        source="llm_structured_candidate",
        description="用受控参数同时处理开仓样本不足和亏损仓占位。",
        params={
            "global_min_score_delta": -0.06,
            "probe_fraction": 0.06,
            "position_size_multiplier": 0.65,
            "full_position_release": True,
            "release_losing_positions_first": True,
            "loss_exit_aggressiveness": "high",
            "expert_integrity_mode": "balanced_probe_allow_one_non_core_missing",
            "fallback_tolerance": {
                "allow_missing_non_core_experts": 1,
                "core_experts_required": ["trend_expert", "momentum_expert", "risk_expert"],
                "non_core_experts": ["sentiment_expert", "position_expert"],
            },
        },
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-4.0, position_id=51)],
        open_positions=[_open_position("BTC/USDT", "long", -7.0)],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[
            _missed_shadow("BTC/USDT", "long", 0.70),
            _missed_shadow("BTC/USDT", "long", 0.74),
            _missed_shadow("BTC/USDT", "long", 0.78),
        ],
        memories=[],
        strategy_events=[
            _strategy_event("capacity_block", status="blocked", reason="max_position capacity")
        ],
        max_open_positions=1,
        extra_profiles=[llm_profile],
    )

    schedule = payload["schedule"]

    assert schedule["scheduler_mode"] == "auto"

    assert schedule["active_profile"]["id"] == "llm_loss_probe"

    assert "结构化候选" in schedule["reason"]

    shadow = next(
        row
        for row in schedule["shadow_validation"]["rows"]
        if row["profile_id"] == "llm_loss_probe"
    )

    assert shadow["eligible"] is True

    assert shadow["would_increase_entries"] is True

    assert shadow["would_release_losers"] is True

    assert shadow["probe_required"] is True

    assert shadow["fallback_safety"] == "probe_core_required"


def test_strategy_learning_auto_disabled_profiles_expire(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    old_time = datetime.now(UTC) - timedelta(seconds=AUTO_DISABLED_PROFILE_RECONSIDER_SECONDS + 60)

    state_store.save(
        {
            "manual_active_profile": "",
            "disabled_profiles": {
                "balanced_probe": {
                    "reason": "auto_runtime_guard:recent_net_pnl_guard",
                    "updated_at": old_time.isoformat(),
                },
                "winner_hold": {
                    "reason": "manual_disable",
                    "updated_at": old_time.isoformat(),
                },
            },
        }
    )

    disabled = state_store.disabled_profiles()

    assert "balanced_probe" not in disabled

    assert "winner_hold" in disabled

    persisted = state_store.load()["disabled_profiles"]

    assert "balanced_probe" not in persisted


def test_strategy_learning_auto_disabled_probe_can_reenter_when_it_solves_overblocking(
    tmp_path,
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    now = datetime.now(UTC)

    state_store.save(
        {
            "manual_active_profile": "",
            "disabled_profiles": {
                "balanced_probe": {
                    "reason": "auto_runtime_guard:fallback_dependency_guard",
                    "updated_at": now.isoformat(),
                    "auto": True,
                    "disabled_until": (now + timedelta(hours=1)).isoformat(),
                }
            },
        }
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[
            SimpleNamespace(
                status="completed",
                missed_opportunity=True,
                decision_action="hold",
                long_return_pct=0.9,
                short_return_pct=-0.1,
            )
        ],
        memories=[],
        max_open_positions=14,
    )

    assert payload["schedule"]["scheduler_mode"] == "auto"

    assert payload["schedule"]["active_profile"]["id"] == "balanced_probe"

    assert "balanced_probe" not in payload["schedule"]["disabled_profiles"]


def test_strategy_learning_context_and_snapshot_include_dispatch_details(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback")],
        shadows=[],
        memories=[],
        max_open_positions=14,
    )

    context = engine.apply_to_context({"min_opportunity_score": 1.0}, payload)

    learning = context["strategy_learning"]

    snapshot = {
        "active_profile": learning["active_profile"],
        "runtime": learning["runtime"],
        "feedback_summary": learning["feedback_summary"],
        "rollback": learning["rollback"],
        "scheduler_mode": learning["scheduler_mode"],
        "manual_profile_id": learning["manual_profile_id"],
        "candidate_count": learning["candidate_count"],
        "dispatch_reason": learning["dispatch_reason"],
        "disabled_profiles": learning["disabled_profiles"],
        "shadow_validation": learning["shadow_validation"],
        "probe": learning["probe"],
        "backtest": learning["backtest"],
        "training_policy": learning["training_policy"],
        "exclude_from_training": True,
    }

    assert learning["scheduler_mode"] in {"auto", "manual"}

    assert isinstance(learning["shadow_validation"].get("rows"), list)

    assert learning["training_policy"]["manual_close_excluded"] is True

    assert snapshot["exclude_from_training"] is True


def test_strategy_learning_llm_prompt_exposes_structured_param_guidance(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    service = StrategyLearningService(state_store=state_store)

    feedback = service.engine.compiler.compile(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-1.6, position_id=701)],
        open_positions=[_old_flat_open_position("LOW/USDT")],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity fallback") for _ in range(8)],
        shadows=[_missed_shadow("BTC/USDT", "long", 0.9)],
        memories=[],
        strategy_events=[
            _strategy_event("capacity_block", status="blocked", reason="max_position capacity")
        ],
        reflections=[_reflection(position_id=702, pnl=-1.0, hold_minutes=160.0)],
        max_open_positions=10,
    )

    prompt = service._llm_candidate_prompt_v3(feedback)
    retry_prompt = service._llm_candidate_retry_prompt(prompt)
    rules_text = " ".join(str(item) for item in prompt["rules"])
    retry_rules_text = " ".join(str(item) for item in retry_prompt["rules"])
    structured_guidance = prompt.get("structured_param_guidance") or prompt[
        "generation_guidance"
    ].get("structured_param_guidance")

    assert "entry_filters" in prompt["allowed_params"]
    assert "portfolio_preference" in prompt["allowed_params"]
    assert "exit_preference" in prompt["allowed_params"]
    assert (
        structured_guidance["portfolio_preference"]["allowed_keys"]["capacity_mode"]
        == ["balanced", "expand", "focus"]
    )
    assert "structured params entry_filters, portfolio_preference, and exit_preference" in rules_text
    assert retry_prompt["structured_param_guidance"] == structured_guidance
    assert "structured params entry_filters, portfolio_preference, and exit_preference" in retry_rules_text


def test_strategy_learning_llm_candidate_prompt_accepts_datetimes(tmp_path, monkeypatch) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    service = StrategyLearningService(state_store=state_store)

    feedback = service.engine.compiler.compile(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "capacity_block",
                status="blocked",
                reason="max_position capacity reached",
            )
        ],
        reflections=[
            _reflection(
                position_id=71,
                pnl=-2.0,
                hold_minutes=220.0,
                mistake="loss held too long",
            )
        ],
        max_open_positions=1,
    )

    captured: dict[str, Any] = {}

    class FakeResponse:

        is_success = True

        status_code = 200

        def json(self) -> dict[str, Any]:

            return {"choices": [{"message": {"content": '{"candidates": []}'}}]}

    class FakeClient:

        def __init__(self, *args: Any, **kwargs: Any) -> None:

            pass

        async def __aenter__(self) -> FakeClient:

            return self

        async def __aexit__(self, *args: Any) -> None:

            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> FakeResponse:

            captured["body"] = json

            return FakeResponse()

    monkeypatch.setattr("services.strategy_learning.httpx.AsyncClient", FakeClient)

    import asyncio

    result = asyncio.run(
        service._call_llm_candidate_model(
            api_base="http://127.0.0.1:9",
            api_key="test-key",
            model="test-model",
            feedback=feedback,
        )
    )

    assert result == []

    prompt = json.loads(captured["body"]["messages"][1]["content"])

    assert len(json.dumps(prompt, ensure_ascii=False)) <= 9000

    recent_events = prompt["feedback_summary"]["event_feedback"].get("recent_events", [])

    assert recent_events

    assert recent_events[0]["event_type"] == "capacity_block"

    assert "created_at" not in recent_events[0]


def test_strategy_learning_llm_candidate_retries_next_model_after_http_error(
    tmp_path, monkeypatch
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    service = StrategyLearningService(state_store=state_store)

    feedback = service.engine.compiler.compile(
        mode="paper",
        window_hours=24,
        positions=[_position(side="long", pnl=-1.2)],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event("entry_block", status="blocked", reason="expert fallback")
        ],
        reflections=[_reflection(position_id=81, pnl=-1.2, mistake="entry quality weak")],
        max_open_positions=20,
    )

    monkeypatch.setattr(
        service,
        "_llm_candidate_configs",
        lambda: [
            {
                "name": "decision_maker",
                "api_base": "http://model-a/v1",
                "api_key": "key-a",
                "model": "qwen3-14b-trade",
            },
            {
                "name": "risk_expert",
                "api_base": "http://model-b/v1",
                "api_key": "key-b",
                "model": "deepseek-r1-14b-risk",
            },
        ],
    )

    calls: list[str] = []

    class FakeResponse:

        def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:

            self.status_code = status_code

            self._payload = payload or {}

            self.text = json.dumps(self._payload)

        @property
        def is_success(self) -> bool:

            return 200 <= self.status_code < 300

        def json(self) -> dict[str, Any]:

            return self._payload

    class FakeClient:

        def __init__(self, *args: Any, **kwargs: Any) -> None:

            pass

        async def __aenter__(self) -> FakeClient:

            return self

        async def __aexit__(self, *args: Any) -> None:

            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> FakeResponse:

            calls.append(json["model"])

            if json["model"] == "qwen3-14b-trade":

                return FakeResponse(502, {"error": "bad gateway"})

            return FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json_module.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "profile_id": "llm_probe_retry",
                                                "label": "重试探针",
                                                "description": "第二模型生成的受控候选。",
                                                "params": {
                                                    "probe_fraction": 0.05,
                                                    "global_min_score_delta": -0.04,
                                                },
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                },
            )

    import asyncio
    import json as json_module

    monkeypatch.setattr("services.strategy_learning.httpx.AsyncClient", FakeClient)

    profiles = asyncio.run(service._generate_llm_profiles(mode="paper", feedback=feedback))

    assert calls == ["qwen3-14b-trade", "deepseek-r1-14b-risk"]

    assert [profile.profile_id for profile in profiles] == ["llm_probe_retry"]

    cache = state_store.load()["llm_candidate_cache"]

    assert cache["model"] == "deepseek-r1-14b-risk"

    assert cache["last_error"] == ""

    assert cache["last_error_kind"] == ""

    assert cache["attempts"][0]["status"] == "failed"

    assert cache["attempts"][0]["error_kind"] == "http_error"

    assert cache["attempts"][1]["status"] == "completed"


def test_strategy_learning_llm_candidate_timeout_is_classified(tmp_path, monkeypatch) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    service = StrategyLearningService(state_store=state_store)
    feedback = service.engine.compiler.compile(
        mode="paper",
        window_hours=24,
        positions=[_position(side="long", pnl=-1.2)],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event("entry_block", status="blocked", reason="expert fallback")
        ],
        reflections=[_reflection(position_id=182, pnl=-1.2, mistake="entry quality weak")],
        max_open_positions=20,
    )
    monkeypatch.setattr(
        service,
        "_llm_candidate_configs",
        lambda: [
            {
                "name": "decision_maker",
                "api_base": "http://model-a/v1",
                "api_key": "key-a",
                "model": "qwen3-14b-trade",
            }
        ],
    )
    calls = 0

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> None:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("model took too long")

    import asyncio

    monkeypatch.setattr("services.strategy_learning.httpx.AsyncClient", FakeClient)

    profiles = asyncio.run(service._generate_llm_profiles(mode="paper", feedback=feedback))

    assert profiles == []
    assert calls == 2
    cache = state_store.load()["llm_candidate_cache"]
    assert cache["last_error_kind"] == "timeout"
    assert "timed out" in cache["last_error"]
    assert cache["attempts"][0]["error_kind"] == "timeout"


def test_strategy_learning_llm_candidate_retries_short_prompt_after_incomplete_json(
    tmp_path, monkeypatch
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    service = StrategyLearningService(state_store=state_store)

    feedback = service.engine.compiler.compile(
        mode="paper",
        window_hours=24,
        positions=[_position(side="long", pnl=-1.2)],
        open_positions=[],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity")],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event("entry_block", status="blocked", reason="expert fallback")
        ],
        reflections=[_reflection(position_id=82, pnl=-1.2, mistake="entry quality weak")],
        max_open_positions=20,
    )

    calls: list[dict[str, Any]] = []

    class FakeResponse:

        is_success = True

        status_code = 200

        def __init__(self, content: str) -> None:

            self._content = content

            self.text = content

        def json(self) -> dict[str, Any]:

            return {"choices": [{"message": {"content": self._content}}]}

    class FakeClient:

        def __init__(self, *args: Any, **kwargs: Any) -> None:

            pass

        async def __aenter__(self) -> FakeClient:

            return self

        async def __aexit__(self, *args: Any) -> None:

            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict[str, Any]
        ) -> FakeResponse:

            calls.append(json)

            if len(calls) == 1:

                return FakeResponse('{"candidates":[{"profile_id":"broken"')

            return FakeResponse(
                json_module.dumps(
                    {
                        "candidates": [
                            {
                                "profile_id": "llm_retry_short",
                                "label": "短重试",
                                "description": "短格式恢复",
                                "params": {
                                    "probe_fraction": 0.05,
                                    "global_min_score_delta": -0.03,
                                },
                            }
                        ]
                    }
                )
            )

    import asyncio
    import json as json_module

    monkeypatch.setattr("services.strategy_learning.httpx.AsyncClient", FakeClient)

    candidates = asyncio.run(
        service._call_llm_candidate_model(
            api_base="http://model/v1",
            api_key="key",
            model="qwen3-14b-trade",
            feedback=feedback,
        )
    )

    assert len(calls) == 2

    assert calls[1]["max_tokens"] <= 260

    assert candidates[0]["profile_id"] == "llm_retry_short"


def test_strategy_learning_llm_candidate_prompt_stays_under_budget(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    service = StrategyLearningService(state_store=state_store)

    noisy_events = [
        _strategy_event(
            "entry_block",
            status="blocked",
            reason=f"expert fallback {index} " + ("x" * 600),
        )
        for index in range(80)
    ]

    noisy_reflections = [
        _reflection(
            position_id=index,
            pnl=-2.0,
            hold_minutes=240.0,
            mistake="loss held too long " + ("m" * 500),
            improvement="release bad positions earlier " + ("i" * 500),
        )
        for index in range(80)
    ]

    feedback = service.engine.compiler.compile(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-1.5, position_id=index) for index in range(20)],
        open_positions=[_open_position(f"SYM{index}/USDT", "long", -index) for index in range(20)],
        orders=[],
        decisions=[_decision("long", reason="expert_integrity") for _ in range(60)],
        shadows=[],
        memories=[],
        strategy_events=noisy_events,
        reflections=noisy_reflections,
        max_open_positions=20,
    )

    prompt = service._llm_candidate_prompt_v3(feedback)

    prompt_text = json.dumps(prompt, ensure_ascii=False)

    assert len(prompt_text) <= 9000

    assert "recent_reflections" not in prompt_text

    assert "created_at" not in prompt_text


def test_strategy_learning_llm_prompt_uses_quality_recovery_for_defensive_probe_loop(
    tmp_path,
) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    service = StrategyLearningService(state_store=state_store)

    feedback = service.engine.compiler.compile(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("long", executed=False) for _ in range(8)],
        shadows=[],
        memories=[],
        strategy_events=[
            _strategy_event(
                "execution_result",
                status="skipped",
                reason="Profit-First 防御探针拦截：低收益质量。",
                attribution={
                    "skip_kind": "profit_first_defensive_probe_shadow",
                    "blocker": "profit_first_defensive_probe_shadow",
                },
            )
            for _ in range(2)
        ],
        max_open_positions=14,
    )

    prompt = service._llm_candidate_prompt_v3(feedback)
    rules_text = " ".join(str(item) for item in prompt["rules"])
    prompt_text = json.dumps(prompt, ensure_ascii=False)

    assert prompt["generation_guidance"]["require_quality_entry_recovery_candidate"] is True
    assert "quality_entry_recovery" in prompt_text
    assert "Do not force every candidate into probe mode" in rules_text
    assert "Probe first" not in rules_text


def test_strategy_learning_parses_json_after_thinking_text() -> None:

    parsed = StrategyLearningService._parse_json_object(
        '<think>reasoning</think>\nHere is JSON:\n{"candidates": []}\nextra text'
    )

    assert parsed == {"candidates": []}


def test_strategy_learning_parses_markdown_json_code_block() -> None:

    parsed = StrategyLearningService._parse_json_object(
        '```json\n{"candidates": [{"profile_id": "md_probe"}]}\n```'
    )

    assert parsed == {"candidates": [{"profile_id": "md_probe"}]}


def test_strategy_learning_auto_disabled_loss_release_stays_disabled_under_pressure(
    tmp_path,
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    now = datetime.now(UTC)

    state_store.save(
        {
            "manual_active_profile": "",
            "disabled_profiles": {
                "loss_release": {
                    "reason": "auto_runtime_guard:recent_net_pnl_guard,fallback_dependency_guard,execution_error_guard",
                    "updated_at": now.isoformat(),
                    "auto": True,
                    "disabled_until": (now + timedelta(hours=1)).isoformat(),
                }
            },
        }
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-2.0, position_id=91)],
        open_positions=[
            _open_position("ARB/USDT", "long", -3.0),
            _open_position("YGG/USDT", "long", -1.5),
        ],
        orders=[],
        decisions=[_healthy_decision("long", executed=True)],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(
                position_id=91,
                pnl=-2.2,
                hold_minutes=260.0,
                mistake="loss hold too long",
            )
        ],
        max_open_positions=2,
    )

    schedule = payload["schedule"]

    assert payload["feedback"]["decision_quality"]["model_health_recovered"] is True

    assert schedule["active_profile"]["id"] == "baseline_current"

    assert "loss_release" in schedule["disabled_profiles"]

    assert "loss_release" not in schedule["reconsidered_profiles"]


def test_strategy_learning_baseline_reason_lists_disabled_candidates(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    now = datetime.now(UTC)

    state_store.save(
        {
            "manual_active_profile": "",
            "disabled_profiles": {
                "loss_release": {
                    "reason": "manual_disable",
                    "updated_at": now.isoformat(),
                    "auto": False,
                    "disabled_until": "",
                },
                "balanced_probe": {
                    "reason": "manual_disable",
                    "updated_at": now.isoformat(),
                    "auto": False,
                    "disabled_until": "",
                },
            },
        }
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-2.0, position_id=92)],
        open_positions=[
            _open_position("ARB/USDT", "long", -3.0),
            _open_position("YGG/USDT", "long", -1.5),
        ],
        orders=[],
        decisions=[_healthy_decision("hold")],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(
                position_id=92,
                pnl=-2.2,
                hold_minutes=260.0,
                mistake="loss hold too long",
            )
        ],
        max_open_positions=2,
    )

    schedule = payload["schedule"]

    assert schedule["active_profile"]["id"] == "baseline_current"

    assert schedule["blocked_candidate_count"] == 2

    assert schedule["disabled_profile_reasons"]["loss_release"]["reason"] == "manual_disable"

    assert "候选" in schedule["reason"]

    assert "禁用" in schedule["reason"]


def test_strategy_learning_runtime_guard_keeps_candidate_when_model_recovered(
    tmp_path,
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    service = StrategyLearningService(engine=engine, state_store=state_store)

    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-1.0, position_id=93)],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("long", executed=True)],
        shadows=[
            SimpleNamespace(
                status="completed",
                missed_opportunity=True,
                decision_action="hold",
                long_return_pct=0.9,
                short_return_pct=-0.1,
            )
        ],
        memories=[],
        strategy_events=[
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
            _strategy_event("execution_error", status="failed", reason="execution failed"),
        ],
        max_open_positions=14,
    )

    guard = service._runtime_guard(payload, mutate=True)

    state = state_store.load()

    assert guard["model_health_recovered"] is True

    assert "execution_error_guard" in guard["reasons"]

    assert guard["should_rollback"] is False

    assert state["manual_active_profile"] == "balanced_probe"

    assert "balanced_probe" not in state.get("disabled_profiles", {})


def test_strategy_learning_compact_dashboard_payload_keeps_scheduler_evidence(
    tmp_path,
) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    service = StrategyLearningService(engine=engine, state_store=state_store)

    extra_profiles = [
        StrategyProfile(
            profile_id=f"llm_probe_{index}",
            version=1,
            label=f"LLM probe {index}",
            status="candidate",
            source="llm_structured_candidate",
            description="bounded probe",
            params={"probe_fraction": 0.05, "global_min_score_delta": -0.03},
        )
        for index in range(20)
    ]

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[_open_position(f"SYM{index}/USDT", "long", -1.0) for index in range(20)],
        orders=[],
        decisions=[_healthy_decision("long", executed=True)],
        shadows=[],
        memories=[],
        max_open_positions=20,
        extra_profiles=extra_profiles,
    )

    payload["detail"] = "summary"

    compact = service._compact_dashboard_payload(payload)

    assert compact["feedback"]["open_position_pressure"]["open_part_count"] == 20

    assert "model_health_recovered" in compact["feedback"]["decision_quality"]

    assert len(compact["schedule"]["candidates"]) <= 12

    assert len(compact["schedule"]["backtest"]["rows"]) <= 12

    assert len(compact["schedule"]["shadow_validation"]["rows"]) <= 12


def test_strategy_learning_context_exposes_low_quality_rebalance_queue(tmp_path) -> None:

    state_store = StrategyLearningStateStore(tmp_path / "state.json")

    engine = StrategyLearningEngine(scheduler=None)

    engine.scheduler.state_store = state_store

    state_store.set_manual_active_profile("balanced_probe")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[],
        open_positions=[
            _old_flat_open_position("ARB/USDT"),
            _old_flat_open_position("YGG/USDT"),
        ],
        orders=[],
        decisions=[_healthy_decision("hold")],
        shadows=[],
        memories=[],
        max_open_positions=2,
    )

    context = engine.apply_to_context({}, payload)

    pressure = payload["feedback"]["open_position_pressure"]

    learning = context["strategy_learning"]

    assert pressure["low_quality_open_count"] == 2

    assert pressure["release_queue"][0]["should_release"] is True

    assert context["low_quality_open_count"] == 2

    assert context["position_rebalance_queue"]

    assert learning["rebalance_queue"]


def test_strategy_learning_trade_target_is_dynamic_advisory_not_entry_gate() -> None:
    engine = StrategyLearningEngine()
    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=1.2, position_id=index) for index in range(6)],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("long", executed=index % 3 == 0) for index in range(120)],
        shadows=[],
        memories=[],
        max_open_positions=20,
    )

    totals = payload["feedback"]["totals"]
    backtest = payload["schedule"]["backtest"]["rows"][0]
    shadow = payload["schedule"]["shadow_validation"]["rows"][0]["trade_count_guard"]

    assert totals["trade_count_target"] > totals["trade_count_target_baseline"]
    assert totals["trade_count_target_is_entry_gate"] is False
    assert totals["trade_count_target_policy"] == "dynamic_advisory_learning_confidence"
    assert backtest["trade_count_target_is_entry_gate"] is False
    assert shadow["is_entry_gate"] is False


def test_strategy_learning_winner_hold_params_are_runtime_consumed(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    state_store.set_manual_active_profile("winner_hold")

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[
            _position(side="long", pnl=0.18, position_id=201),
            _position(side="short", pnl=-4.0, position_id=202),
            _position(side="long", pnl=0.15, position_id=203),
            _position(side="short", pnl=-3.5, position_id=204),
        ],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("hold")],
        shadows=[],
        memories=[],
        reflections=[
            _reflection(
                position_id=201,
                pnl=0.18,
                mistake="small win closed too early",
            ),
            _reflection(position_id=202, pnl=-4.0),
            _reflection(
                position_id=203,
                pnl=0.15,
                mistake="small win closed too early",
            ),
            _reflection(position_id=204, pnl=-3.5),
        ],
        max_open_positions=14,
    )
    context = engine.apply_to_context({}, payload)

    assert payload["schedule"]["active_profile"]["id"] == "winner_hold"
    assert payload["schedule"]["scheduler_mode"] == "manual"
    assert context["winner_hold_extension"] == "high"
    dynamic_multiplier = payload["schedule"]["active_profile"]["params"][
        "profit_lock_min_usdt_multiplier"
    ]
    assert dynamic_multiplier > 1.0
    assert context["profit_lock_min_usdt_multiplier"] == dynamic_multiplier
    assert context["pullback_lock_enabled"] is True
    runtime = context["strategy_learning"]["runtime"]
    assert runtime["profit_lock_min_usdt_multiplier"] == dynamic_multiplier
    assert runtime["winner_hold_dynamic"]["policy"] == (
        "dynamic_window_distribution_not_fixed_usdt_thresholds"
    )
    assert "profit_lock_min_usdt_multiplier" in payload["schedule"]["active_profile"][
        "consumed_runtime_params"
    ]


def test_strategy_learning_winner_hold_uses_dynamic_payoff_distribution(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store

    positions = [
        *[
            _position(side="long", pnl=pnl, position_id=3100 + index)
            for index, pnl in enumerate(
                [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, -1.35, -1.45]
            )
        ],
        *[
            _position(side="short", pnl=pnl, position_id=3200 + index)
            for index, pnl in enumerate(
                [0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, -1.30, -1.55]
            )
        ],
    ]
    reflections = [
        _reflection(position_id=3300 + index, pnl=pnl)
        for index, pnl in enumerate([0.19, 0.21, 0.23, -1.95, -2.45])
    ]

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=positions,
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("hold") for _ in range(40)],
        shadows=[],
        memories=[],
        reflections=reflections,
        max_open_positions=20,
    )
    context = engine.apply_to_context({}, payload)
    feedback = payload["feedback"]
    schedule = payload["schedule"]
    active = schedule["active_profile"]
    problem_keys = {item["key"] for item in feedback["problems"]}
    backtest = next(row for row in schedule["backtest"]["rows"] if row["profile_id"] == "winner_hold")

    assert feedback["totals"]["payoff_profile"]["triggered"] is True
    assert feedback["totals"]["payoff_profile"]["policy"] == (
        "dynamic_window_distribution_not_fixed_usdt_thresholds"
    )
    assert "small_wins_large_losses" in problem_keys
    assert schedule["active_profile"]["id"] == "winner_hold"
    assert active["params"]["payoff_repair_intensity"] > 0
    assert active["params"]["profit_lock_min_usdt_multiplier"] != 1.25
    assert active["params"]["winner_hold_dynamic"]["training"]["triggered"] is True
    assert backtest["payoff_repair_profile"]["triggered"] is True
    assert "small_wins_large_losses" in backtest["matched_fixes"]
    assert context["payoff_repair_intensity"] == active["params"]["payoff_repair_intensity"]
    assert context["profit_lock_min_usdt_multiplier"] == active["params"][
        "profit_lock_min_usdt_multiplier"
    ]
    assert context["winner_hold_dynamic"]["training"]["triggered"] is True

    service = StrategyLearningService(state_store=state_store)
    prompt = service._llm_candidate_prompt_v3(
        service.engine.compiler.compile(
            mode="paper",
            window_hours=168,
            positions=positions,
            open_positions=[],
            orders=[],
            decisions=[_healthy_decision("hold") for _ in range(40)],
            shadows=[],
            memories=[],
            reflections=reflections,
            max_open_positions=20,
        )
    )
    assert prompt["generation_guidance"]["payoff_repair_profile"]["triggered"] is True
    assert prompt["feedback_summary"]["totals"]["payoff_profile"]["triggered"] is True
    assert prompt["feedback_summary"]["reflection_feedback"]["payoff_profile"]["triggered"] is True
    assert "fixed USDT cutoffs" in " ".join(prompt["rules"])


def test_strategy_learning_rejects_structured_candidate_with_no_runtime_params(tmp_path) -> None:
    state_store = StrategyLearningStateStore(tmp_path / "state.json")
    engine = StrategyLearningEngine(scheduler=None)
    engine.scheduler.state_store = state_store
    inert_profile = StrategyProfile(
        profile_id="llm_inert_sample_target",
        version=1,
        label="sample target only",
        status="candidate",
        source="llm_structured_candidate",
        description="Only changes advisory training sample target.",
        params={"min_trade_count_target": 40},
    )

    payload = engine.build(
        mode="paper",
        window_hours=168,
        positions=[_position(side="long", pnl=-1.0, position_id=301)],
        open_positions=[],
        orders=[],
        decisions=[_healthy_decision("hold") for _ in range(20)],
        shadows=[],
        memories=[],
        max_open_positions=14,
        extra_profiles=[inert_profile],
    )

    backtest = next(
        row
        for row in payload["schedule"]["backtest"]["rows"]
        if row["profile_id"] == "llm_inert_sample_target"
    )
    shadow = next(
        row
        for row in payload["schedule"]["shadow_validation"]["rows"]
        if row["profile_id"] == "llm_inert_sample_target"
    )

    assert backtest["pass"] is False
    assert shadow["eligible"] is False
    assert backtest["consumed_runtime_params"] == []
    assert shadow["consumed_runtime_params"] == []
    assert payload["schedule"]["active_profile"]["id"] != "llm_inert_sample_target"
