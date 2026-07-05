from __future__ import annotations

from typing import Any

from services.analysis_budget import AnalysisBudgetConfig, AnalysisBudgetPolicy
from services.trading_service import TradingService


def _normalize(symbol: Any) -> str:
    return str(symbol or "").upper()


def _group_count(open_positions: list[dict[str, Any]] | None) -> int:
    return len(
        {
            (
                str(pos.get("model_name") or "ensemble_trader"),
                _normalize(pos.get("symbol")),
                str(pos.get("side") or "").lower(),
            )
            for pos in (open_positions or [])
            if str(pos.get("side") or "").lower() in {"long", "short"}
        }
    )


def _policy(
    fast_scan: dict[tuple[str, str], dict[str, Any]] | None = None,
    *,
    config: AnalysisBudgetConfig | None = None,
) -> tuple[AnalysisBudgetPolicy, list[list[tuple[tuple[str, str], list[dict[str, Any]]]]]]:
    scans_seen: list[list[tuple[tuple[str, str], list[dict[str, Any]]]]] = []

    def portfolio_context(open_positions: list[dict[str, Any]]) -> dict[str, Any]:
        return {"position_count": len(open_positions)}

    def scanner(
        grouped_items: list[tuple[tuple[str, str], list[dict[str, Any]]]],
        _feature_vectors: dict[str, Any],
        portfolio_profit_context: dict[str, Any] | None,
        _strategy_context: dict[str, Any] | None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        assert portfolio_profit_context == {"position_count": sum(len(g[1]) for g in grouped_items)}
        scans_seen.append(grouped_items)
        return fast_scan or {}

    return (
        AnalysisBudgetPolicy(
            normalize_symbol=_normalize,
            open_position_group_counter=_group_count,
            portfolio_profit_context_provider=portfolio_context,
            position_review_scanner=scanner,
            urgent_exit_checker=lambda scan: bool(scan and scan.get("urgent")),
            config=config or AnalysisBudgetConfig(),
        ),
        scans_seen,
    )


def _position(symbol: str, side: str = "long", model_name: str = "ensemble_trader") -> dict:
    return {"symbol": symbol, "side": side, "model_name": model_name, "is_open": True}


def test_analysis_budget_expands_market_budget_when_low_risk_roster_underfilled() -> None:
    policy, scans_seen = _policy(
        config=AnalysisBudgetConfig(target_position_groups=3, roster_fill_market_symbol_min=7)
    )

    result = policy.context(
        [_position("BTC/USDT")],
        {},
        base_market_limit=8,
        run_position_analysis=False,
        run_market_analysis=True,
    )

    assert result["risk_level"] == "low"
    assert result["market_symbol_limit"] == 7
    assert result["configured_market_symbol_limit"] == 8
    assert result["market_limit_policy"] == "position_first_low_risk_underfilled"
    assert result["market_symbol_limit_is_entry_gate"] is False
    assert result["position_first_scheduling"] is True
    assert result["roster_underfilled"] is True
    assert "持仓复盘由独立 position loop 并行负责" in result["reason"]
    diagnostics = result["market_limit_diagnostics"]
    assert diagnostics["read_only"] is True
    assert diagnostics["is_entry_gate"] is False
    assert diagnostics["selected_market_symbol_limit"] == 7
    assert diagnostics["configured_market_symbol_limit"] == 8
    assert diagnostics["market_limit_policy"] == "position_first_low_risk_underfilled"
    assert diagnostics["position_group_count"] == 1
    assert diagnostics["target_position_groups"] == 3
    assert diagnostics["roster_underfilled"] is True
    assert diagnostics["market_caps"]["low_risk_open_position_cap"] == 3
    assert diagnostics["market_caps"]["roster_fill_market_symbol_min"] == 7
    assert scans_seen == []


def test_analysis_budget_underfilled_market_budget_respects_base_limit() -> None:
    policy, _scans_seen = _policy(
        config=AnalysisBudgetConfig(target_position_groups=3, roster_fill_market_symbol_min=7)
    )

    result = policy.context(
        [_position("BTC/USDT")],
        {},
        base_market_limit=2,
        run_position_analysis=False,
        run_market_analysis=True,
    )

    assert result["risk_level"] == "low"
    assert result["market_symbol_limit"] == 2
    assert result["configured_market_symbol_limit"] == 2
    assert result["market_limit_policy"] == "position_first_low_risk_underfilled"


def test_analysis_budget_without_positions_uses_dynamic_market_cap_not_full_pool() -> None:
    policy, scans_seen = _policy(
        config=AnalysisBudgetConfig(
            target_position_groups=3,
            roster_fill_market_symbol_min=36,
            market_no_position_cap=6,
        )
    )

    result = policy.context(
        [],
        {},
        base_market_limit=20,
        run_position_analysis=False,
        run_market_analysis=True,
    )

    assert result["risk_level"] == "none"
    assert result["market_symbol_limit"] == 6
    assert result["configured_market_symbol_limit"] == 20
    assert result["market_limit_policy"] == "no_position_dynamic_market_budget"
    assert result["market_symbol_limit_is_entry_gate"] is False
    assert scans_seen == []


def test_analysis_budget_high_risk_position_protects_position_review_capacity() -> None:
    fast_scan = {
        ("ensemble_trader", "BTC/USDT"): {"exit_score": 92.0, "priority_score": 92.0},
        ("ensemble_trader", "ETH/USDT"): {"exit_score": 71.0, "priority_score": 71.0},
        ("ensemble_trader", "SOL/USDT"): {"exit_score": 20.0, "priority_score": 10.0},
    }
    policy, scans_seen = _policy(fast_scan)

    result = policy.context(
        [_position("BTC/USDT"), _position("ETH/USDT"), _position("SOL/USDT")],
        {},
        base_market_limit=20,
        run_position_analysis=True,
        run_market_analysis=True,
    )

    assert result["risk_level"] == "high"
    assert result["position_max_groups"] == 8
    assert result["market_symbol_limit"] == 1
    assert result["market_limit_policy"] == "position_first_high_risk"
    assert result["forced_exit_groups"] == 2
    assert result["high_exit_groups"] == 1
    assert len(scans_seen[0]) == 3


def test_analysis_budget_medium_risk_keeps_limited_entry_exploration() -> None:
    fast_scan = {
        ("ensemble_trader", "BTC/USDT"): {"exit_score": 10.0, "priority_score": 63.0},
        ("ensemble_trader", "ETH/USDT"): {"exit_score": 20.0, "priority_score": 70.0},
        ("ensemble_trader", "SOL/USDT"): {"exit_score": 30.0, "priority_score": 65.0},
    }
    policy, _scans_seen = _policy(fast_scan, config=AnalysisBudgetConfig(target_position_groups=3))

    result = policy.context(
        [_position("BTC/USDT"), _position("ETH/USDT"), _position("SOL/USDT")],
        {},
        base_market_limit=20,
        run_position_analysis=True,
        run_market_analysis=True,
    )

    assert result["risk_level"] == "medium"
    assert result["position_max_groups"] == 6
    assert result["market_symbol_limit"] == 2
    assert result["market_limit_policy"] == "position_first_medium_risk"
    assert result["priority_groups"] == 3


def test_analysis_budget_raises_position_review_budget_when_positions_are_crowded() -> None:
    policy, _scans_seen = _policy(config=AnalysisBudgetConfig(target_position_groups=3))
    open_positions = [_position(f"SYM{i}/USDT") for i in range(13)]

    result = policy.context(
        open_positions,
        {},
        base_market_limit=20,
        run_position_analysis=True,
        run_market_analysis=True,
    )

    assert result["risk_level"] == "low"
    assert result["position_max_groups"] == 13
    assert result["total_position_groups"] == 13


def test_analysis_budget_high_load_position_review_budget_overrides_static_high_risk_cap() -> None:
    symbols = [f"SYM{i}/USDT" for i in range(25)]
    fast_scan = {("ensemble_trader", symbols[0]): {"exit_score": 92.0, "priority_score": 92.0}}
    policy, _scans_seen = _policy(fast_scan, config=AnalysisBudgetConfig(target_position_groups=3))

    result = policy.context(
        [_position(symbol) for symbol in symbols],
        {},
        base_market_limit=20,
        run_position_analysis=True,
        run_market_analysis=True,
    )

    assert result["risk_level"] == "high"
    assert result["position_max_groups"] == 14
    assert result["total_position_groups"] == 25


def test_analysis_budget_new_pair_pause_zeroes_market_scan_budget() -> None:
    policy, _scans_seen = _policy(
        config=AnalysisBudgetConfig(target_position_groups=3, roster_fill_market_symbol_min=7)
    )

    result = policy.context(
        [_position("BTC/USDT")],
        {},
        base_market_limit=20,
        run_position_analysis=True,
        run_market_analysis=True,
        new_pair_pause_reason="daily loss pause",
    )

    assert result["risk_level"] == "low"
    assert result["market_symbol_limit"] == 0
    assert result["roster_underfilled"] is True


def test_analysis_budget_uses_strategy_learning_runtime_targets() -> None:
    policy, scans_seen = _policy()
    strategy_context = {
        "strategy_profile_id": "loss_release",
        "strategy_learning": {
            "runtime": {
                "target_position_groups": 5,
                "max_open_positions": 12,
                "analysis_budget": {
                    "position_max_groups": 9,
                    "position_high_risk_max_groups": 11,
                    "position_urgent_exit_max_groups": 12,
                    "roster_fill_market_symbol_min": 18,
                    "market_low_risk_open_position_cap": 3,
                },
            }
        },
    }
    result = policy.context(
        [_position("BTC/USDT"), _position("ETH/USDT")],
        {},
        base_market_limit=2,
        run_position_analysis=True,
        run_market_analysis=True,
        strategy_context=strategy_context,
    )

    assert result["budget_source"] == "strategy_learning"
    assert result["target_position_groups"] == 5
    assert result["position_max_groups"] == 9
    assert result["market_symbol_limit"] == 2
    assert result["configured_market_symbol_limit"] == 2
    assert result["market_limit_policy"] == "position_first_low_risk_underfilled"
    diagnostics = result["market_limit_diagnostics"]
    assert diagnostics["budget_source"] == "strategy_learning"
    assert diagnostics["target_position_groups"] == 5
    assert diagnostics["position_review_caps"]["selected_position_max_groups"] == 9
    assert diagnostics["market_caps"]["low_risk_open_position_cap"] == 3
    assert "must not be treated as trade permission" in diagnostics["diagnostic_boundary"]
    assert scans_seen


def test_analysis_budget_runtime_roster_fill_cannot_lower_candidate_floor() -> None:
    policy, _scans_seen = _policy()
    strategy_context = {
        "strategy_profile_id": "candidate_2",
        "strategy_learning": {
            "runtime": {
                "target_position_groups": 12,
                "max_open_positions": 12,
                "analysis_budget": {
                    "roster_fill_market_symbol_min": 2,
                    "market_low_risk_open_position_cap": 2,
                },
            }
        },
    }

    result = policy.context(
        [_position(f"OPEN{i}/USDT") for i in range(7)],
        {},
        base_market_limit=8,
        run_position_analysis=True,
        run_market_analysis=True,
        strategy_context=strategy_context,
    )

    assert result["budget_source"] == "strategy_learning"
    assert result["roster_underfilled"] is True
    assert result["market_symbol_limit"] == 6
    assert result["configured_market_symbol_limit"] == 8
    assert result["market_limit_diagnostics"]["market_caps"]["roster_fill_market_symbol_min"] == 6


def test_analysis_budget_low_risk_market_scan_keeps_discovery_width() -> None:
    policy, _scans_seen = _policy()
    strategy_context = {
        "strategy_profile_id": "candidate_too_narrow",
        "strategy_learning": {
            "runtime": {
                "target_position_groups": 4,
                "max_open_positions": 20,
                "analysis_budget": {
                    "market_low_risk_open_position_cap": 2,
                    "roster_fill_market_symbol_min": 2,
                },
            }
        },
    }

    result = policy.context(
        [_position(f"OPEN{i}/USDT") for i in range(12)],
        {},
        base_market_limit=12,
        run_position_analysis=True,
        run_market_analysis=True,
        strategy_context=strategy_context,
    )

    assert result["risk_level"] == "low"
    assert result["roster_underfilled"] is False
    assert result["market_limit_policy"] == "position_first_low_risk"
    assert result["market_symbol_limit"] == 4
    assert result["market_symbol_limit_is_entry_gate"] is False


def test_trading_service_analysis_budget_context_delegates_to_policy() -> None:
    service = object.__new__(TradingService)
    calls: list[dict[str, Any]] = []
    feature_vector = object()
    strategy_context = {"strategy_profile_id": "loss_release"}

    class FakeBudget:
        def context(self, open_positions, feature_vectors, **kwargs):
            calls.append(
                {
                    "open_positions": open_positions,
                    "feature_vectors": feature_vectors,
                    **kwargs,
                }
            )
            return {"risk_level": "delegated", "market_symbol_limit": 3}

    service.analysis_budget = FakeBudget()

    result = service._position_review_budget_context(
        [{"symbol": "BTC/USDT"}],
        {"BTC/USDT": feature_vector},
        base_market_limit=9,
        run_position_analysis=True,
        run_market_analysis=False,
        new_pair_pause_reason="pause",
        strategy_context=strategy_context,
    )

    assert result == {"risk_level": "delegated", "market_symbol_limit": 3}
    assert calls == [
        {
            "open_positions": [{"symbol": "BTC/USDT"}],
            "feature_vectors": {"BTC/USDT": feature_vector},
            "base_market_limit": 9,
            "run_position_analysis": True,
            "run_market_analysis": False,
            "new_pair_pause_reason": "pause",
            "strategy_context": strategy_context,
        }
    ]
