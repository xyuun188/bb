from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from data_feed.feature_vector import FeatureVector
from risk_manager.engine import RiskEngine
from services.entry_profit_risk_sizing import reconcile_profit_risk_sizing
from services.execution_service import _return_entry_contract_result
from services.paper_bootstrap_canary import (
    PAPER_BOOTSTRAP_CANARY_VERSION,
    PAPER_BOOTSTRAP_LOSS_CIRCUIT_COOLDOWN_SECONDS,
    PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES,
    PAPER_BOOTSTRAP_MIN_FILL_DRIFT_RESERVE_FRACTION,
    PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION,
    PAPER_BOOTSTRAP_SIZING_VERSION,
    PaperBootstrapCanaryPolicy,
    annotate_paper_bootstrap_opportunity,
    assess_paper_canary_position_horizon,
    build_paper_canary_position_lifecycle,
    select_paper_bootstrap_candidate,
)


def _context() -> dict[str, Any]:
    def distribution(side: str, objective: float) -> dict[str, Any]:
        return {
            "side": side,
            "horizon_minutes": 10,
            "raw_expected_return_pct": objective + 0.2,
            "objective_expected_return_pct": objective,
            "lower_quantile_return_pct": objective - 0.1,
            "dispersion_pct": 0.1,
            "distribution_member_count": 128,
            "source_authority": "extra_trees_empirical_distribution",
        }

    return {
        "trading_mode": "paper",
        "entry_candidate_evidence": {
            "long": {
                "production_source_count": 0,
                "execution_cost": {"total_pct": 0.08},
            },
            "short": {
                "production_source_count": 0,
                "execution_cost": {"total_pct": 0.08},
            },
        },
        "ml_signal": {
            "paper_canary_authorized": True,
            "artifact_lifecycle": "canary",
            "model_version": "candidate-v1",
            "trained_sample_count": 1200,
            "paper_canary": {
                "state": "ready",
                "authorized": True,
                "execution_scope": "paper_only",
                "production_permission": False,
                "eligible_sides": ["long", "short"],
            },
            "predictions": [
                {
                    "return_distribution_contract": {
                        "long": distribution("long", -0.12),
                        "short": distribution("short", -0.35),
                    }
                }
            ],
        },
    }


def _decision() -> DecisionOutput:
    canary = select_paper_bootstrap_candidate(_context())
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=float(canary["confidence"]),
        reasoning="paper canary test",
        position_size_pct=0.0,
        suggested_leverage=1.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        raw_response={
            "paper_bootstrap_canary": canary,
            "pre_order_execution_facts": {
                "production_eligible": True,
                "input_fingerprint": "book-v1",
            },
        },
        feature_snapshot={
            "current_price": 100.0,
            "atr_14": 1.0,
            "volatility_20": 0.01,
        },
    )


def test_negative_shadow_objective_can_request_paper_sample_without_production_permission() -> None:
    canary = select_paper_bootstrap_candidate(_context())

    assert canary["authorized"] is True
    assert canary["selected_side"] == "long"
    assert canary["selected_observation"]["objective_expected_return_pct"] < 0
    assert canary["production_permission"] is False
    assert canary["version"] == PAPER_BOOTSTRAP_CANARY_VERSION


def test_paper_canary_opportunity_annotation_is_finite_and_observation_only() -> None:
    decision = _decision()

    score = annotate_paper_bootstrap_opportunity(decision)

    opportunity = decision.raw_response["opportunity_score"]
    assert score == pytest.approx(-0.12)
    assert opportunity["score"] == pytest.approx(-0.12)
    assert opportunity["score_kind"] == "paper_canary_objective_expected_return"
    assert opportunity["contract_lifecycle"] == "paper_bootstrap_canary"
    assert opportunity["production_eligible"] is False
    assert opportunity["production_permission"] is False
    assert opportunity["observation_only"] is True
    assert opportunity["execution_scope"] == "paper_only"


def test_paper_canary_selection_is_forbidden_in_live_mode() -> None:
    context = _context()
    context["trading_mode"] = "live"

    canary = select_paper_bootstrap_candidate(context)

    assert canary["authorized"] is False
    assert canary["reason"] == "paper_execution_mode_required"


def test_executed_paper_canary_builds_expiring_position_lifecycle() -> None:
    executed_at = datetime.now(UTC) - timedelta(minutes=11)
    decision = SimpleNamespace(
        id=123,
        symbol="BTC/USDT",
        action="long",
        is_paper=True,
        was_executed=True,
        executed_at=executed_at,
        raw_llm_response={"paper_bootstrap_canary": select_paper_bootstrap_candidate(_context())},
    )

    lifecycle = build_paper_canary_position_lifecycle(decision)
    assessment = assess_paper_canary_position_horizon(
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "execution_mode": "paper",
            "paper_canary_lifecycle": lifecycle,
        }
    )

    assert lifecycle["version"] == PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION
    assert lifecycle["horizon_minutes"] == 10
    assert assessment["authorized"] is True
    assert assessment["elapsed"] is True


def test_paper_canary_horizon_fails_closed_for_malformed_lifecycle() -> None:
    assessment = assess_paper_canary_position_horizon(
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "execution_mode": "paper",
            "paper_canary_lifecycle": {
                "version": PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION,
                "authorized": True,
                "expires_at": "not-a-time",
            },
        }
    )

    assert assessment["authorized"] is False
    assert assessment["elapsed"] is False


def test_paper_canary_horizon_reads_persisted_management_lifecycle() -> None:
    lifecycle = {
        "version": PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION,
        "kind": "paper_bootstrap_canary_position",
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "symbol": "BTC/USDT",
        "side": "long",
        "horizon_minutes": 10,
        "expires_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
    }

    assessment = assess_paper_canary_position_horizon(
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "execution_mode": "paper",
            "current_management_contract": {
                "paper_canary_lifecycle": lifecycle,
            },
        }
    )

    assert assessment["authorized"] is True
    assert assessment["elapsed"] is True


def test_ensemble_emits_paper_canary_entry_when_production_source_is_absent() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace(get=lambda _name: None))

    decision = coordinator.combine(
        FeatureVector(symbol="BTC/USDT", current_price=100.0, close=100.0),
        _context(),
        opinions={},
    )

    assert decision.action == Action.LONG
    assert decision.raw_response["paper_bootstrap_canary"]["authorized"] is True
    assert decision.raw_response["entry_permission_policy"]["execution_scope"] == "paper_only"


@pytest.mark.asyncio
async def test_paper_canary_builds_bounded_contract_that_passes_hard_risk() -> None:
    async def balance(_mode: str, _decision: DecisionOutput) -> float:
        return 500.0

    async def facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "production_eligible": True,
            "account_equity_usdt": 1000.0,
            "available_margin_usdt": 500.0,
            "target_inst_id": "BTC-USDT-SWAP",
            "contract_specs": {"BTC-USDT-SWAP": {"ctVal": "0.01", "ctMult": "1"}},
            "leverage_tiers": [{"maxLeverage": 100, "maxNotional": 10000}],
            "entry_instrument_availability": {"available": True},
        }

    async def history() -> list[Any]:
        return []

    decision = _decision()
    decision.feature_snapshot["atr_14"] = 2.443918123
    decision.raw_response["opportunity_score"] = {"execution_cost": {"total_pct": 0.15}}
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=balance,
        exchange_risk_facts=facts,
        history_provider=history,
    )

    prepared = await policy.prepare(decision, "paper", [])
    assessed = policy.assess(decision, "paper")
    hard_risk = RiskEngine().assess(decision, [], account_balance=1000.0)

    assert prepared.eligible is True
    assert assessed.eligible is True
    assert hard_risk.approved is True
    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["contract_lifecycle"] == "paper_bootstrap_canary"
    assert sizing["risk_budget_usdt"] == pytest.approx(0.5)
    assert sizing["portfolio_risk_budget_usdt"] == pytest.approx(1.0)
    assert sizing["leverage_tier_selection"]["production_eligible"] is True
    assert sizing["leverage_tier_selection"]["contract_spec"]["ctVal"] == "0.01"
    assert sizing["contract_version"] == PAPER_BOOTSTRAP_SIZING_VERSION
    assert sizing["portfolio_risk_snapshot"]["gross_notional_usdt"] == 0.0
    assert sizing["portfolio_risk_snapshot"]["scope"] == ("paper_bootstrap_canary_positions_only")
    assert sizing["entry_instrument_availability"]["available"] is True
    assert sizing["estimated_fill_drift_reserve_fraction"] == pytest.approx(
        PAPER_BOOTSTRAP_MIN_FILL_DRIFT_RESERVE_FRACTION
    )
    assert sizing["fill_notional_ceiling_usdt"] > sizing["target_notional_usdt"]
    assert sizing["target_notional_usdt"] * (
        1.0 + sizing["estimated_fill_drift_reserve_fraction"]
    ) == pytest.approx(sizing["fill_notional_ceiling_usdt"])
    assert decision.suggested_leverage == 1.0
    assert decision.stop_loss_pct > 0
    assert decision.take_profit_pct > decision.stop_loss_pct
    assert _return_entry_contract_result(decision, "paper").passed is True
    assert _return_entry_contract_result(decision, "live").passed is False

    reconciled = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=sizing["final_notional_usdt"],
        final_leverage=1.0,
        source="test_exchange_precision",
    )
    assert reconciled["eligible"] is True
    assert (
        decision.raw_response["profit_risk_sizing"]["policy_provenance"]["strategy_version"]
        == PAPER_BOOTSTRAP_SIZING_VERSION
    )


@pytest.mark.asyncio
async def test_non_canary_account_positions_do_not_consume_canary_position_slot() -> None:
    async def balance(_mode: str, _decision: DecisionOutput) -> float:
        return 500.0

    async def facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "production_eligible": True,
            "account_equity_usdt": 1000.0,
            "available_margin_usdt": 500.0,
            "target_inst_id": "BTC-USDT-SWAP",
            "contract_specs": {"BTC-USDT-SWAP": {"ctVal": "0.01", "ctMult": "1"}},
            "leverage_tiers": [{"maxLeverage": 100, "maxNotional": 10000}],
            "entry_instrument_availability": {"available": True},
        }

    async def history() -> list[Any]:
        return []

    decision = _decision()
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=balance,
        exchange_risk_facts=facts,
        history_provider=history,
    )

    result = await policy.prepare(
        decision,
        "paper",
        [
            {"symbol": "ETH/USDT", "quantity": 1.0, "is_open": True},
            {"symbol": "SOL/USDT", "quantity": 1.0, "is_open": True},
        ],
    )

    assert result.eligible is True
    guard = decision.raw_response["paper_bootstrap_canary"]["runtime_guard"]
    assert guard["open_position_count"] == 0
    assert guard["account_open_position_count"] == 2


@pytest.mark.asyncio
async def test_unsettled_canary_entry_consumes_canary_position_slot() -> None:
    async def unused_balance(_mode: str, _decision: DecisionOutput) -> float:
        raise AssertionError("open canary must block before account calls")

    async def unused_facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise AssertionError("open canary must block before exchange calls")

    now = datetime.now(UTC)

    async def history() -> list[Any]:
        return [
            SimpleNamespace(
                id=101,
                raw_llm_response={"paper_bootstrap_canary": {"authorized": True}},
                outcome=None,
                executed_at=now,
                created_at=now,
            )
        ]

    decision = _decision()
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=unused_balance,
        exchange_risk_facts=unused_facts,
        history_provider=history,
    )

    result = await policy.prepare(decision, "paper", [])

    assert result.eligible is False
    assert "paper_canary_open_position_limit_reached" in result.reason
    guard = decision.raw_response["paper_bootstrap_canary"]["runtime_guard"]
    assert guard["open_position_count"] == 1
    assert guard["account_open_position_count"] == 0


@pytest.mark.asyncio
async def test_blocked_canary_is_persisted_as_hold_with_candidate_direction_evidence() -> None:
    async def unused_balance(_mode: str, _decision: DecisionOutput) -> float:
        raise AssertionError("preflight must not request account capacity")

    async def unused_facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise AssertionError("preflight must not request exchange facts")

    now = datetime.now(UTC)

    async def history() -> list[Any]:
        return [
            SimpleNamespace(
                id=101,
                raw_llm_response={"paper_bootstrap_canary": {"authorized": True}},
                outcome=None,
                executed_at=now,
                created_at=now,
            )
        ]

    decision = _decision()
    original_action = decision.action
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=unused_balance,
        exchange_risk_facts=unused_facts,
        history_provider=history,
    )

    preflight = await policy.preflight(decision, "paper", [])
    changed = policy.demote_blocked_candidate_to_hold(decision, preflight)

    assert changed is True
    assert original_action == Action.LONG
    assert decision.action == Action.HOLD
    contract = decision.raw_response["paper_bootstrap_canary"]
    assert contract["selected_side"] == "long"
    assert contract["candidate_action"] == "long"
    assert contract["persisted_action"] == "hold"
    assert contract["runtime_authorized"] is False
    observation = decision.raw_response["paper_bootstrap_canary_observation"]
    assert observation["shadow_direction_preserved"] is True
    assert observation["exchange_submission_allowed"] is False


@pytest.mark.asyncio
async def test_paper_canary_opens_circuit_after_two_completed_losses() -> None:
    async def unused_balance(_mode: str, _decision: DecisionOutput) -> float:
        raise AssertionError("loss circuit must block before account calls")

    async def unused_facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise AssertionError("loss circuit must block before exchange calls")

    now = datetime.now(UTC)

    async def history() -> list[Any]:
        raw = {"paper_bootstrap_canary": {"authorized": True}}
        return [
            SimpleNamespace(
                raw_llm_response=raw,
                outcome="loss",
                executed_at=now,
                created_at=now,
            ),
            SimpleNamespace(
                raw_llm_response=raw,
                outcome="loss",
                executed_at=now,
                created_at=now,
            ),
        ]

    decision = _decision()
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=unused_balance,
        exchange_risk_facts=unused_facts,
        history_provider=history,
    )

    result = await policy.prepare(decision, "paper", [])

    assert result.eligible is False
    assert "paper_canary_consecutive_loss_circuit_open" in result.reason


@pytest.mark.asyncio
async def test_paper_canary_loss_circuit_allows_recovery_probe_after_cooldown() -> None:
    old_loss_at = datetime.now(UTC) - timedelta(
        seconds=PAPER_BOOTSTRAP_LOSS_CIRCUIT_COOLDOWN_SECONDS + 60
    )

    async def history() -> list[Any]:
        raw = {"paper_bootstrap_canary": {"authorized": True}}
        return [
            SimpleNamespace(
                raw_llm_response=raw,
                outcome="loss",
                executed_at=old_loss_at,
                created_at=old_loss_at,
            ),
            SimpleNamespace(
                raw_llm_response=raw,
                outcome="loss",
                executed_at=old_loss_at - timedelta(minutes=10),
                created_at=old_loss_at - timedelta(minutes=10),
            ),
        ]

    decision = _decision()
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=lambda *_args: None,
        exchange_risk_facts=lambda *_args: None,
        history_provider=history,
    )

    result = await policy.preflight(decision, "paper", [])

    assert result.eligible is True
    guard = decision.raw_response["paper_bootstrap_canary"]["runtime_guard"]
    assert guard["consecutive_loss_count"] == 2
    assert guard["loss_circuit_open"] is False


@pytest.mark.asyncio
async def test_paper_canary_daily_budget_expands_while_sample_deficit_is_large() -> None:
    executed_at = datetime.now(UTC) - timedelta(minutes=20)

    async def history() -> list[Any]:
        raw = {"paper_bootstrap_canary": {"authorized": True}}
        return [
            SimpleNamespace(
                raw_llm_response=raw,
                outcome="profit",
                executed_at=executed_at - timedelta(seconds=index),
                created_at=executed_at - timedelta(seconds=index),
            )
            for index in range(PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES)
        ]

    decision = _decision()
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=lambda *_args: None,
        exchange_risk_facts=lambda *_args: None,
        history_provider=history,
    )

    result = await policy.preflight(decision, "paper", [])

    assert result.eligible is False
    assert "paper_canary_daily_entry_budget_exhausted" in result.reason
    guard = decision.raw_response["paper_bootstrap_canary"]["runtime_guard"]
    assert guard["max_daily_entries"] == PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES
    assert guard["daily_entry_limit_source"] == (
        "authoritative_sample_deficit_over_collection_horizon"
    )


def test_canary_history_query_projects_only_runtime_guard_columns() -> None:
    statement = PaperBootstrapCanaryPolicy._history_statement()

    selected_columns = {column.key for column in statement.selected_columns}

    assert selected_columns == {
        "paper_canary_authorized",
        "outcome",
        "executed_at",
        "created_at",
    }
    assert "raw_llm_response" not in selected_columns
    assert any(
        str(criterion.left) == "ai_decisions.created_at"
        for criterion in statement.whereclause.clauses
    )


@pytest.mark.asyncio
async def test_canary_preflight_history_timeout_fails_closed_with_timing() -> None:
    history_cancelled = asyncio.Event()

    async def unused_balance(_mode: str, _decision: DecisionOutput) -> float:
        raise AssertionError("history timeout must block before account calls")

    async def unused_facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise AssertionError("history timeout must block before exchange calls")

    async def slow_history() -> list[Any]:
        try:
            await asyncio.sleep(60)
        finally:
            history_cancelled.set()
        return []

    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=unused_balance,
        exchange_risk_facts=unused_facts,
        history_provider=slow_history,
        preflight_timeout_seconds=0.1,
        history_timeout_seconds=0.05,
    )
    decision = _decision()
    started = asyncio.get_running_loop().time()

    result = await policy.preflight(decision, "paper", [])
    await asyncio.sleep(0)

    elapsed = asyncio.get_running_loop().time() - started
    contract = decision.raw_response["paper_bootstrap_canary"]
    history_timing = contract["runtime_guard"]["history_query"]
    preflight_timing = contract["runtime_preflight_timing"]
    assert result.eligible is False
    assert "paper_canary_history_timeout" in result.reason
    assert elapsed < 0.3
    assert history_cancelled.is_set() is True
    assert history_timing["status"] == "timeout"
    assert history_timing["allowed_timeout_seconds"] == pytest.approx(0.05, abs=0.01)
    assert preflight_timing["status"] == "failed_closed"
    assert preflight_timing["history_query_status"] == "timeout"
    assert preflight_timing["message_zh"]


@pytest.mark.asyncio
async def test_canary_preflight_exhausted_market_deadline_skips_history_query() -> None:
    history_called = False

    async def unused_balance(_mode: str, _decision: DecisionOutput) -> float:
        raise AssertionError("exhausted preflight must block before account calls")

    async def unused_facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise AssertionError("exhausted preflight must block before exchange calls")

    async def history() -> list[Any]:
        nonlocal history_called
        history_called = True
        return []

    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=unused_balance,
        exchange_risk_facts=unused_facts,
        history_provider=history,
    )
    decision = _decision()

    result = await policy.preflight(
        decision,
        "paper",
        [],
        deadline_monotonic=asyncio.get_running_loop().time(),
        timing_scope="market_decision_persistence",
    )

    contract = decision.raw_response["paper_bootstrap_canary"]
    assert result.eligible is False
    assert "paper_canary_preflight_budget_exhausted" in result.reason
    assert history_called is False
    assert contract["runtime_guard"]["history_query"]["status"] == "budget_exhausted"
    assert decision.raw_response["market_context_timings"][-1]["scope"] == (
        "market_decision_persistence"
    )


@pytest.mark.asyncio
async def test_canary_exchange_facts_timeout_fails_closed_before_balance() -> None:
    facts_cancelled = asyncio.Event()
    balance_called = False

    async def balance(_mode: str, _decision: DecisionOutput) -> float:
        nonlocal balance_called
        balance_called = True
        return 500.0

    async def slow_facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            await asyncio.sleep(60)
        finally:
            facts_cancelled.set()
        return {}

    async def history() -> list[Any]:
        return []

    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=balance,
        exchange_risk_facts=slow_facts,
        history_provider=history,
        prepare_timeout_seconds=0.3,
        exchange_facts_timeout_seconds=0.05,
    )
    decision = _decision()

    result = await policy.prepare(decision, "paper", [])
    await asyncio.sleep(0)

    contract = decision.raw_response["paper_bootstrap_canary"]
    prepare_timing = contract["runtime_prepare_timing"]
    assert result.eligible is False
    assert "paper_canary_exchange_facts_timeout" in result.reason
    assert facts_cancelled.is_set() is True
    assert balance_called is False
    assert prepare_timing["status"] == "failed_closed"
    assert prepare_timing["stages"][0]["stage"] == "exchange_risk_facts"
    assert prepare_timing["stages"][0]["status"] == "timeout"
