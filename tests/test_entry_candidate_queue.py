from __future__ import annotations

from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_candidate_queue import EntryCandidate, EntryCandidateQueuePolicy
from services.trading_service import TradingService


def _decision(symbol: str, score: float) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=Action.LONG,
        confidence=0.75,
        reasoning="candidate",
        position_size_pct=0.02,
        suggested_leverage=3.0,
        raw_response={"test_score": score},
    )


def _candidate(symbol: str, score: float) -> EntryCandidate:
    decision = _decision(symbol, score)
    return (symbol, "ensemble_trader", decision, object(), None)


def test_entry_candidate_queue_ranks_by_opportunity_score() -> None:
    reasons_seen: list[tuple[str, int, int]] = []

    def score_candidate(decision: DecisionOutput, strategy: dict[str, Any] | None) -> float:
        assert strategy == {"mode": "test"}
        return float(decision.raw_response["test_score"])

    def wait_sort_reason(
        decision: DecisionOutput,
        *,
        rank: int,
        candidate_count: int,
    ) -> str:
        reasons_seen.append((decision.symbol, rank, candidate_count))
        return f"wait:{rank}/{candidate_count}"

    policy = EntryCandidateQueuePolicy(score_candidate, wait_sort_reason)

    ranked = policy.ranked(
        [_candidate("ETH/USDT", 0.4), _candidate("BTC/USDT", 1.2), _candidate("SOL/USDT", 0.8)],
        {"mode": "test"},
    )

    assert [item.candidate[0] for item in ranked] == ["BTC/USDT", "SOL/USDT", "ETH/USDT"]
    assert [item.rank for item in ranked] == [1, 2, 3]
    assert [item.score for item in ranked] == [1.2, 0.8, 0.4]
    assert [item.wait_reason for item in ranked] == ["wait:1/3", "wait:2/3", "wait:3/3"]
    assert reasons_seen == [("BTC/USDT", 1, 3), ("SOL/USDT", 2, 3), ("ETH/USDT", 3, 3)]


def test_trading_service_entry_candidate_queue_delegates_to_policy() -> None:
    service = object.__new__(TradingService)
    calls: list[tuple[list[EntryCandidate], dict[str, Any]]] = []

    class FakeQueue:
        def ranked(self, candidates, strategy_context):
            calls.append((candidates, strategy_context))
            return []

    service.entry_candidate_queue = FakeQueue()
    candidates = [_candidate("BTC/USDT", 1.0)]

    assert service._entry_candidate_queue_policy().ranked(candidates, {"mode": "x"}) == []
    assert calls == [(candidates, {"mode": "x"})]
