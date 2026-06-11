"""Entry candidate filtering before execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.entry_candidate_queue import EntryCandidate

GateReason = Callable[[DecisionOutput], str | None]
MarketRegimeReason = Callable[[DecisionOutput, dict[str, Any] | None], str | None]
CapacityReason = Callable[[str, DecisionOutput, list[dict[str, Any]], dict[str, dict]], str | None]
ReserveCapacity = Callable[[str, DecisionOutput, dict[str, dict]], None]


@dataclass(frozen=True, slots=True)
class RejectedEntryCandidate:
    """An entry candidate rejected by a pre-execution policy."""

    candidate: EntryCandidate
    reason: str
    blocker: str
    annotate_raw_response: bool = False


@dataclass(frozen=True, slots=True)
class EntryCandidateFilterResult:
    """Entry candidates separated into executable and rejected groups."""

    accepted_candidates: list[EntryCandidate]
    rejected_candidates: list[RejectedEntryCandidate]


@dataclass(frozen=True, slots=True)
class EntryCandidateFilterPolicy:
    """Apply entry gate, market-regime, and capacity filters in ranking order."""

    gate_reason: GateReason
    market_regime_reason: MarketRegimeReason
    capacity_reason: CapacityReason
    reserve_capacity: ReserveCapacity

    def filter(
        self,
        candidates: list[EntryCandidate],
        *,
        strategy_context: dict[str, Any] | None,
        market_regime_context: dict[str, Any] | None,
        open_positions: list[dict[str, Any]],
        staged_entry_counts: dict[str, dict],
    ) -> EntryCandidateFilterResult:
        gate_passed: list[EntryCandidate] = []
        rejected: list[RejectedEntryCandidate] = []

        for candidate in candidates:
            decision = candidate[2]
            reason = self.gate_reason(decision)
            if reason:
                rejected.append(
                    RejectedEntryCandidate(
                        candidate=candidate,
                        reason=reason,
                        blocker="entry_gate",
                        annotate_raw_response=True,
                    )
                )
                continue
            gate_passed.append(candidate)

        accepted: list[EntryCandidate] = []
        regime_context = strategy_context or market_regime_context
        for candidate in gate_passed:
            _symbol, model_name, decision, _assessment, _decision_db_id = candidate
            reason = self.market_regime_reason(decision, regime_context)
            if reason:
                rejected.append(
                    RejectedEntryCandidate(
                        candidate=candidate,
                        reason=reason,
                        blocker="market_regime",
                    )
                )
                continue

            reason = self.capacity_reason(
                model_name,
                decision,
                open_positions,
                staged_entry_counts,
            )
            if reason:
                rejected.append(
                    RejectedEntryCandidate(
                        candidate=candidate,
                        reason=reason,
                        blocker="entry_capacity",
                    )
                )
                continue

            accepted.append(candidate)
            self.reserve_capacity(model_name, decision, staged_entry_counts)

        return EntryCandidateFilterResult(
            accepted_candidates=accepted,
            rejected_candidates=rejected,
        )
