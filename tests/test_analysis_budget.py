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


def test_analysis_budget_raises_market_budget_when_roster_is_underfilled() -> None:
    policy, scans_seen = _policy(
        config=AnalysisBudgetConfig(target_position_groups=3, roster_fill_market_symbol_min=7)
    )

    result = policy.context(
        [_position("BTC/USDT")],
        {},
        base_market_limit=2,
        run_position_analysis=False,
        run_market_analysis=True,
    )

    assert result["risk_level"] == "none"
    assert result["market_symbol_limit"] == 7
    assert result["roster_underfilled"] is True
    assert "roster-fill" in result["reason"]
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
    assert result["market_symbol_limit"] == 2
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
    assert result["market_symbol_limit"] == 4
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
    assert result["position_max_groups"] == 10
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


def test_trading_service_analysis_budget_context_delegates_to_policy() -> None:
    service = object.__new__(TradingService)
    calls: list[dict[str, Any]] = []
    feature_vector = object()

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
        }
    ]
