from __future__ import annotations

from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_candidate_filter import EntryCandidateFilterPolicy
from services.entry_candidate_queue import EntryCandidate
from services.trading_service import TradingService


def _decision(symbol: str, action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=action,
        confidence=0.76,
        reasoning="candidate",
        position_size_pct=0.02,
        suggested_leverage=3.0,
        raw_response={},
    )


def _candidate(symbol: str) -> EntryCandidate:
    return (symbol, "ensemble_trader", _decision(symbol), object(), None)


def test_entry_candidate_filter_runs_all_gates_before_capacity_reservation() -> None:
    candidates = [_candidate("A/USDT"), _candidate("B/USDT"), _candidate("C/USDT")]
    calls: list[tuple[str, str]] = []
    reserved: list[str] = []

    def gate_reason(decision: DecisionOutput) -> str | None:
        calls.append(("gate", decision.symbol))
        return "gate blocked" if decision.symbol == "B/USDT" else None

    def capacity_reason(
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]],
        staged_entry_counts: dict[str, dict],
    ) -> str | None:
        assert model_name == "ensemble_trader"
        assert open_positions == [{"symbol": "OPEN/USDT"}]
        calls.append(("capacity", decision.symbol))
        return "capacity blocked" if decision.symbol == "C/USDT" else None

    def reserve_capacity(
        _model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: dict[str, dict],
    ) -> None:
        calls.append(("reserve", decision.symbol))
        staged_entry_counts.setdefault("reserved", {})[decision.symbol] = True
        reserved.append(decision.symbol)

    policy = EntryCandidateFilterPolicy(
        gate_reason=gate_reason,
        market_regime_reason=lambda _decision, _context: None,
        capacity_reason=capacity_reason,
        reserve_capacity=reserve_capacity,
    )

    staged_counts: dict[str, dict] = {}
    result = policy.filter(
        candidates,
        strategy_context={"strategy": "normal"},
        market_regime_context={"mode": "mixed"},
        open_positions=[{"symbol": "OPEN/USDT"}],
        staged_entry_counts=staged_counts,
    )

    assert [candidate[0] for candidate in result.accepted_candidates] == ["A/USDT"]
    assert [
        (item.candidate[0], item.blocker, item.reason) for item in result.rejected_candidates
    ] == [
        ("B/USDT", "entry_gate", "gate blocked"),
        ("C/USDT", "entry_capacity", "capacity blocked"),
    ]
    assert result.rejected_candidates[0].annotate_raw_response is True
    assert result.rejected_candidates[1].annotate_raw_response is False
    assert reserved == ["A/USDT"]
    assert staged_counts == {"reserved": {"A/USDT": True}}
    assert calls == [
        ("gate", "A/USDT"),
        ("gate", "B/USDT"),
        ("gate", "C/USDT"),
        ("capacity", "A/USDT"),
        ("reserve", "A/USDT"),
        ("capacity", "C/USDT"),
    ]


def test_entry_candidate_filter_uses_strategy_context_for_market_regime() -> None:
    candidate = _candidate("BTC/USDT")
    contexts_seen: list[dict[str, Any] | None] = []

    def market_regime_reason(
        decision: DecisionOutput,
        context: dict[str, Any] | None,
    ) -> str | None:
        assert decision.symbol == "BTC/USDT"
        contexts_seen.append(context)
        return "regime blocked"

    policy = EntryCandidateFilterPolicy(
        gate_reason=lambda _decision: None,
        market_regime_reason=market_regime_reason,
        capacity_reason=lambda *_args: None,
        reserve_capacity=lambda *_args: None,
    )

    result = policy.filter(
        [candidate],
        strategy_context={"strategy": "portfolio_roster_build"},
        market_regime_context={"mode": "mixed"},
        open_positions=[],
        staged_entry_counts={},
    )

    assert result.accepted_candidates == []
    assert [
        (item.candidate[0], item.blocker, item.reason) for item in result.rejected_candidates
    ] == [("BTC/USDT", "market_regime", "regime blocked")]
    assert contexts_seen == [{"strategy": "portfolio_roster_build"}]


def test_trading_service_entry_candidate_filter_delegates_to_policy() -> None:
    service = object.__new__(TradingService)
    calls: list[tuple[list[EntryCandidate], dict[str, Any] | None]] = []

    class FakeFilter:
        def filter(self, candidates, *, strategy_context, **_kwargs):
            calls.append((candidates, strategy_context))
            return "filtered"

    service.entry_candidate_filter = FakeFilter()
    candidates = [_candidate("BTC/USDT")]

    assert (
        service._entry_candidate_filter_policy().filter(
            candidates,
            strategy_context={"strategy": "x"},
            market_regime_context={},
            open_positions=[],
            staged_entry_counts={},
        )
        == "filtered"
    )
    assert calls == [(candidates, {"strategy": "x"})]
