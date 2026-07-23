from __future__ import annotations

from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_candidate_queue import EntryCandidate, EntryCandidateQueuePolicy
from services.trading_service import TradingService


def _decision(
    symbol: str,
    score: float,
    *,
    action: Action = Action.LONG,
    opportunity_score: dict[str, Any] | None = None,
) -> DecisionOutput:
    raw_response: dict[str, Any] = {"test_score": score}
    if opportunity_score is not None:
        raw_response["opportunity_score"] = opportunity_score
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=action,
        confidence=0.75,
        reasoning="candidate",
        position_size_pct=0.02,
        suggested_leverage=3.0,
        raw_response=raw_response,
    )


def _candidate(
    symbol: str,
    score: float,
    *,
    action: Action = Action.LONG,
    opportunity_score: dict[str, Any] | None = None,
) -> EntryCandidate:
    decision = _decision(symbol, score, action=action, opportunity_score=opportunity_score)
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


def test_rules_canary_queue_uses_rule_score_without_calling_profit_scorer() -> None:
    def forbidden_profit_score(
        _decision: DecisionOutput,
        _strategy: dict[str, Any] | None,
    ) -> float:
        raise AssertionError("rules-canary queue must not call the profit scorer")

    def wait_sort_reason(
        _decision: DecisionOutput,
        *,
        rank: int,
        candidate_count: int,
    ) -> str:
        return f"wait:{rank}/{candidate_count}"

    first = _candidate("BTC/USDT", 100.0)
    second = _candidate("ETH/USDT", -100.0)
    for candidate, rule_score in ((first, 0.8), (second, 1.0)):
        decision = candidate[2]
        decision.raw_response.update(
            {
                "production_trade_gate": {
                    "mode": "live_rules_canary",
                    "decision_authority": "rules",
                    "model_can_influence": False,
                },
                "live_rules_canary_signal": {
                    "production_eligible": True,
                    "score": rule_score,
                },
            }
        )

    ranked = EntryCandidateQueuePolicy(
        forbidden_profit_score,
        wait_sort_reason,
    ).ranked([first, second], None)

    assert [item.candidate[0] for item in ranked] == ["ETH/USDT", "BTC/USDT"]
    assert [item.score for item in ranked] == [1.0, 0.8]
    queue = ranked[0].candidate[2].raw_response["opportunity_score"][
        "authoritative_queue"
    ]
    assert queue["policy"] == "live_rules_canary_technical_consensus"


def test_entry_candidate_queue_strategy_context_cannot_override_return_order() -> None:
    def score_candidate(decision: DecisionOutput, strategy: dict[str, Any] | None) -> float:
        return float(decision.raw_response["test_score"])

    def wait_sort_reason(
        decision: DecisionOutput,
        *,
        rank: int,
        candidate_count: int,
    ) -> str:
        return f"wait:{rank}/{candidate_count}"

    policy = EntryCandidateQueuePolicy(score_candidate, wait_sort_reason)
    ranked = policy.ranked(
        [
            _candidate(
                "BTC/USDT",
                1.02,
                action=Action.LONG,
                opportunity_score={
                    "expected_net_return_pct": 0.90,
                    "profit_quality_ratio": 0.95,
                    "capital_efficiency_score": 0.80,
                    "tail_risk_score": 0.40,
                },
            ),
            _candidate(
                "ETH/USDT",
                0.98,
                action=Action.SHORT,
                opportunity_score={
                    "expected_net_return_pct": 0.88,
                    "profit_quality_ratio": 0.92,
                    "capital_efficiency_score": 0.78,
                    "tail_risk_score": 0.36,
                },
            ),
        ],
        {
            "position_exposure": {
                "dominant_side": "long",
                "net_ratio": 0.78,
                "long_count": 4,
                "short_count": 1,
                "long_count_share": 0.80,
                "short_count_share": 0.20,
            },
            "portfolio_roster": {
                "underfilled": True,
                "gap": 3,
                "current_position_groups": 5,
                "target_position_groups": 8,
            },
            "dynamic_position_capacity": {
                "entry_limit": 6,
                "open_group_count": 5,
                "factors": {"rotation_slots": 1},
            },
            "strategy_learning": {
                "structured_params": {
                    "portfolio_preference": {"capacity_mode": "expand"},
                }
            },
        },
    )

    assert [item.candidate[0] for item in ranked] == ["BTC/USDT", "ETH/USDT"]
    top_queue = ranked[0].candidate[2].raw_response["opportunity_score"][
        "authoritative_queue"
    ]
    assert top_queue["score"] == 1.02
    assert top_queue["policy"] == "fee_after_return_lcb_minus_expected_downside_only"
    assert "portfolio_queue" not in ranked[0].candidate[2].raw_response["opportunity_score"]


def test_entry_candidate_queue_does_not_apply_duplicate_symbol_penalty() -> None:
    def score_candidate(decision: DecisionOutput, strategy: dict[str, Any] | None) -> float:
        return float(decision.raw_response["test_score"])

    def wait_sort_reason(
        decision: DecisionOutput,
        *,
        rank: int,
        candidate_count: int,
    ) -> str:
        return f"wait:{rank}/{candidate_count}"

    policy = EntryCandidateQueuePolicy(score_candidate, wait_sort_reason)
    ranked = policy.ranked(
        [
            _candidate(
                "BTC/USDT",
                1.10,
                opportunity_score={
                    "expected_net_return_pct": 1.20,
                    "profit_quality_ratio": 1.00,
                    "capital_efficiency_score": 0.90,
                    "tail_risk_score": 0.35,
                    "strong_aligned_profit_evidence": True,
                },
            ),
            _candidate(
                "BTC/USDT",
                1.09,
                opportunity_score={
                    "expected_net_return_pct": 1.18,
                    "profit_quality_ratio": 0.98,
                    "capital_efficiency_score": 0.88,
                    "tail_risk_score": 0.35,
                    "strong_aligned_profit_evidence": True,
                },
            ),
            _candidate(
                "SOL/USDT",
                1.00,
                opportunity_score={
                    "expected_net_return_pct": 0.82,
                    "profit_quality_ratio": 0.84,
                    "capital_efficiency_score": 0.72,
                    "tail_risk_score": 0.40,
                },
            ),
        ],
        {
            "position_exposure": {
                "dominant_side": "neutral",
                "net_ratio": 0.0,
                "long_count": 2,
                "short_count": 2,
                "long_count_share": 0.5,
                "short_count_share": 0.5,
            },
            "dynamic_position_capacity": {
                "entry_limit": 4,
                "open_group_count": 3,
            },
        },
    )

    assert [item.candidate[0] for item in ranked] == ["BTC/USDT", "BTC/USDT", "SOL/USDT"]
    assert [item.score for item in ranked] == [1.10, 1.09, 1.00]
    for item in ranked:
        opportunity = item.candidate[2].raw_response["opportunity_score"]
        assert "portfolio_queue" not in opportunity
        assert opportunity["authoritative_queue"]["score"] == item.score


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
