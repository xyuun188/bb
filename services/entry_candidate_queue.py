"""Entry candidate queue ranking for round-end execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput

EntryCandidate = tuple[str, str, DecisionOutput, Any, int | None]
ScoreCandidate = Callable[[DecisionOutput, dict[str, Any] | None], float]
WaitSortReason = Callable[..., str]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


@dataclass(frozen=True, slots=True)
class RankedEntryCandidate:
    """An entry candidate with its current round ranking metadata."""

    candidate: EntryCandidate
    rank: int
    candidate_count: int
    score: float
    wait_reason: str


@dataclass(frozen=True, slots=True)
class EntryCandidateQueuePolicy:
    """Sort queued entry candidates by opportunity score."""

    score_candidate: ScoreCandidate
    wait_sort_reason: WaitSortReason

    def ranked(
        self,
        candidates: list[EntryCandidate],
        strategy_context: dict[str, Any] | None,
    ) -> list[RankedEntryCandidate]:
        runtime = self._queue_runtime_context(strategy_context)
        remaining = [
            {
                "candidate": candidate,
                "base_score": float(self.score_candidate(candidate[2], strategy_context)),
                "symbol": self._candidate_symbol(candidate),
                "side": self._candidate_side(candidate[2]),
            }
            for candidate in candidates
        ]
        candidate_count = len(remaining)
        open_side_counts = {
            "long": max(_safe_int(runtime.get("long_count"), 0), 0),
            "short": max(_safe_int(runtime.get("short_count"), 0), 0),
        }
        selected_side_counts = {"long": 0, "short": 0}
        selected_symbols: dict[str, int] = {}
        ranked: list[RankedEntryCandidate] = []
        for rank in range(1, candidate_count + 1):
            adjusted_candidates = []
            for row in remaining:
                decision = row["candidate"][2]
                queue_adjustment = self._portfolio_queue_adjustment(
                    decision,
                    runtime=runtime,
                    open_side_counts=open_side_counts,
                    selected_side_counts=selected_side_counts,
                    selected_symbols=selected_symbols,
                    rank=rank,
                    candidate_count=candidate_count,
                )
                adjusted_score = float(row["base_score"]) + _safe_float(
                    queue_adjustment.get("total_adjustment"),
                    0.0,
                )
                adjusted_candidates.append(
                    (adjusted_score, float(row["base_score"]), queue_adjustment, row)
                )
            adjusted_candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                ),
                reverse=True,
            )
            score, base_score, queue_adjustment, winner = adjusted_candidates[0]
            candidate = winner["candidate"]
            decision = candidate[2]
            self._annotate_queue_diagnostics(
                decision,
                runtime=runtime,
                queue_adjustment=queue_adjustment,
                base_score=float(base_score),
                adjusted_score=float(score),
                rank=rank,
                candidate_count=candidate_count,
            )
            wait_reason = self.wait_sort_reason(
                decision,
                rank=rank,
                candidate_count=candidate_count,
            )
            ranked.append(
                RankedEntryCandidate(
                    candidate=candidate,
                    rank=rank,
                    candidate_count=candidate_count,
                    score=float(score),
                    wait_reason=wait_reason,
                )
            )
            side = str(winner["side"] or "")
            if side in selected_side_counts:
                selected_side_counts[side] += 1
            symbol = str(winner["symbol"] or "")
            if symbol:
                selected_symbols[symbol] = selected_symbols.get(symbol, 0) + 1
            remaining = [row for row in remaining if row is not winner]
        return ranked

    @staticmethod
    def _candidate_symbol(candidate: EntryCandidate) -> str:
        symbol = str(candidate[0] or "").strip().upper()
        if symbol:
            return symbol
        decision = candidate[2]
        return str(getattr(decision, "symbol", "") or "").strip().upper()

    @staticmethod
    def _candidate_side(decision: DecisionOutput) -> str:
        action = getattr(decision, "action", None)
        if action == Action.LONG:
            return "long"
        if action == Action.SHORT:
            return "short"
        raw = _safe_dict(getattr(decision, "raw_response", None))
        text = str(raw.get("action") or raw.get("decision_action") or "").lower()
        if text in {"long", "buy"}:
            return "long"
        if text in {"short", "sell"}:
            return "short"
        return "unknown"

    @staticmethod
    def _queue_runtime_context(strategy_context: dict[str, Any] | None) -> dict[str, Any]:
        context = _safe_dict(strategy_context)
        roster = _safe_dict(context.get("portfolio_roster"))
        capacity = _safe_dict(context.get("dynamic_position_capacity"))
        exposure = _safe_dict(context.get("position_exposure"))
        learning = _safe_dict(context.get("strategy_learning"))
        structured = _safe_dict(learning.get("structured_params"))
        preference = _safe_dict(roster.get("preference"))
        if not preference:
            preference = _safe_dict(context.get("portfolio_preference"))
        if not preference:
            preference = _safe_dict(structured.get("portfolio_preference"))
        capacity_factors = _safe_dict(capacity.get("factors"))
        current_groups = max(
            _safe_int(roster.get("current_position_groups"), 0),
            _safe_int(context.get("position_group_count"), 0),
            _safe_int(capacity.get("open_group_count"), 0),
        )
        entry_limit = max(
            _safe_int(context.get("max_open_positions_entry"), 0),
            _safe_int(capacity.get("entry_limit"), 0),
            _safe_int(capacity.get("effective_limit"), 0),
            _safe_int(capacity.get("target_limit"), 0),
        )
        available_slots = max(entry_limit - current_groups, 0) if entry_limit > 0 else 0
        rotation_slots = max(
            _safe_int(context.get("rotation_slots"), 0),
            _safe_int(roster.get("rotation_slots"), 0),
            _safe_int(capacity_factors.get("rotation_slots"), 0),
            _safe_int(capacity_factors.get("strategy_rotation_slots"), 0),
        )
        target_groups = max(
            _safe_int(context.get("target_position_groups"), 0),
            _safe_int(context.get("target_open_position_groups"), 0),
            _safe_int(roster.get("target_position_groups"), 0),
        )
        roster_gap = max(
            _safe_int(roster.get("gap"), 0),
            max(target_groups - current_groups, 0),
        )
        return {
            "underfilled": bool(roster.get("underfilled")) or roster_gap > 0,
            "roster_gap": roster_gap,
            "current_groups": current_groups,
            "target_groups": target_groups,
            "entry_limit": entry_limit,
            "available_slots": available_slots,
            "rotation_slots": rotation_slots,
            "capacity_mode": str(preference.get("capacity_mode") or "balanced").lower(),
            "quality_bias": str(
                _safe_dict(context.get("entry_filter_preference")).get("quality_bias")
                or _safe_dict(context.get("entry_filters")).get("quality_bias")
                or "balanced"
            ).lower(),
            "dominant_side": str(exposure.get("dominant_side") or "neutral").lower(),
            "net_ratio": abs(_safe_float(exposure.get("net_ratio"), 0.0)),
            "long_count": _safe_int(exposure.get("long_count"), 0),
            "short_count": _safe_int(exposure.get("short_count"), 0),
            "long_count_share": _safe_float(exposure.get("long_count_share"), 0.0),
            "short_count_share": _safe_float(exposure.get("short_count_share"), 0.0),
            "release_pressure_active": bool(
                context.get("strategy_learning_release_pressure_active")
                or _safe_dict(context.get("strategy_learning_sizing")).get("release_pressure_active")
                or _safe_dict(learning.get("runtime")).get("release_pressure_active")
            ),
        }

    def _portfolio_queue_adjustment(
        self,
        decision: DecisionOutput,
        *,
        runtime: dict[str, Any],
        open_side_counts: dict[str, int],
        selected_side_counts: dict[str, int],
        selected_symbols: dict[str, int],
        rank: int,
        candidate_count: int,
    ) -> dict[str, Any]:
        raw = _safe_dict(getattr(decision, "raw_response", None))
        opportunity = _safe_dict(raw.get("opportunity_score"))
        side = self._candidate_side(decision)
        symbol = str(getattr(decision, "symbol", "") or "").strip().upper()
        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        capital_efficiency = _safe_float(opportunity.get("capital_efficiency_score"), 0.0)
        tail_risk = _safe_float(opportunity.get("tail_risk_score"), 0.0)
        if not opportunity:
            return {
                "symbol": symbol,
                "side": side,
                "rank_context": {"rank": rank, "candidate_count": candidate_count},
                "runtime": {
                    "underfilled": bool(runtime.get("underfilled")),
                    "roster_gap": max(_safe_int(runtime.get("roster_gap"), 0), 0),
                    "capacity_mode": str(runtime.get("capacity_mode") or "balanced"),
                    "quality_bias": runtime.get("quality_bias"),
                    "dominant_side": str(runtime.get("dominant_side") or "neutral"),
                    "available_slots": max(_safe_int(runtime.get("available_slots"), 0), 0),
                    "rotation_slots": max(_safe_int(runtime.get("rotation_slots"), 0), 0),
                    "release_pressure_active": bool(runtime.get("release_pressure_active")),
                },
                "signals": {},
                "projected_side_share": 0.0,
                "projected_long_count": open_side_counts.get("long", 0)
                + selected_side_counts.get("long", 0),
                "projected_short_count": open_side_counts.get("short", 0)
                + selected_side_counts.get("short", 0),
                "adjustments": {},
                "reasons": [],
                "total_adjustment": 0.0,
            }
        strong_evidence = bool(opportunity.get("strong_aligned_profit_evidence")) or (
            expected_net > 0 and profit_quality > 0.85 and tail_risk < 0.92
        )
        dominant_side = str(runtime.get("dominant_side") or "neutral")
        side_share = _safe_float(
            runtime.get(f"{side}_count_share"),
            0.0,
        )
        available_slots = max(_safe_int(runtime.get("available_slots"), 0), 0)
        rotation_slots = max(_safe_int(runtime.get("rotation_slots"), 0), 0)
        scarcity = 1.0 if available_slots <= max(rotation_slots, 1) else 0.0
        if available_slots <= 1:
            scarcity = 1.25
        capacity_mode = str(runtime.get("capacity_mode") or "balanced")
        if capacity_mode == "focus":
            scarcity = max(scarcity, 1.10)
        elif capacity_mode == "expand":
            scarcity = max(scarcity - 0.10, 0.0)
        underfilled = bool(runtime.get("underfilled"))
        roster_gap = max(_safe_int(runtime.get("roster_gap"), 0), 0)
        total_open = max(
            open_side_counts.get("long", 0) + open_side_counts.get("short", 0),
            0,
        )
        projected_long = open_side_counts.get("long", 0) + selected_side_counts.get("long", 0)
        projected_short = open_side_counts.get("short", 0) + selected_side_counts.get("short", 0)
        if side == "long":
            projected_long += 1
        elif side == "short":
            projected_short += 1
        projected_total = max(projected_long + projected_short, total_open + rank, 1)
        projected_side_share = (
            projected_long / projected_total if side == "long" else projected_short / projected_total
        )

        adjustments: dict[str, float] = {}
        reasons: list[str] = []

        if scarcity > 0:
            quality_bonus = (
                min(max(expected_net, 0.0), 6.0) * 0.035
                + min(max(profit_quality, 0.0), 2.0) * 0.07
                + min(max(capital_efficiency, 0.0), 2.0) * 0.05
            ) * scarcity
            if expected_net <= 0 or profit_quality <= 0:
                quality_bonus -= 0.10 * max(scarcity, 1.0)
            adjustments["scarce_slot_quality"] = _clamp(quality_bonus, -0.22, 0.32)
            if abs(adjustments["scarce_slot_quality"]) >= 0.01:
                reasons.append("scarce_slot_quality")

        if dominant_side in {"long", "short"} and side in {"long", "short"}:
            if side != dominant_side and expected_net > 0 and profit_quality > 0:
                balance_gap = max(
                    _safe_float(runtime.get("net_ratio"), 0.0),
                    abs(
                        _safe_float(runtime.get("long_count_share"), 0.0)
                        - _safe_float(runtime.get("short_count_share"), 0.0)
                    ),
                )
                balance_bonus = (
                    0.06 + balance_gap * 0.24 + min(max(roster_gap, 0), 6) * 0.015
                ) * (1.0 + max(scarcity, 0.0) * 0.25)
                adjustments["diversification_bonus"] = _clamp(balance_bonus, 0.0, 0.34)
                if adjustments["diversification_bonus"] >= 0.01:
                    reasons.append("diversification_bonus")
            elif side == dominant_side and not underfilled:
                crowding_penalty = (
                    max(_safe_float(runtime.get("net_ratio"), 0.0) - 0.10, 0.0) * 0.28
                    + max(side_share - 0.55, 0.0) * 0.36
                )
                adjustments["dominant_side_penalty"] = -_clamp(crowding_penalty, 0.0, 0.24)
                if abs(adjustments["dominant_side_penalty"]) >= 0.01:
                    reasons.append("dominant_side_penalty")

        if underfilled and side in {"long", "short"} and expected_net > 0 and profit_quality > 0:
            roster_fill_bonus = 0.05 + min(roster_gap, 6) * 0.02
            if capacity_mode == "expand":
                roster_fill_bonus *= 1.15
            elif capacity_mode == "focus":
                roster_fill_bonus *= 0.85
            if dominant_side not in {"long", "short"} or side != dominant_side or strong_evidence:
                adjustments["roster_fill_bonus"] = _clamp(roster_fill_bonus, 0.0, 0.24)
                if adjustments["roster_fill_bonus"] >= 0.01:
                    reasons.append("roster_fill_bonus")

        if side in {"long", "short"}:
            repeat_side_penalty = max(projected_side_share - 0.60, 0.0) * (0.45 + scarcity * 0.18)
            if underfilled and dominant_side in {"long", "short"} and side != dominant_side:
                repeat_side_penalty *= 0.60
            if strong_evidence and expected_net > 1.0:
                repeat_side_penalty *= 0.75
            adjustments["repeat_side_penalty"] = -_clamp(repeat_side_penalty, 0.0, 0.26)
            if abs(adjustments["repeat_side_penalty"]) >= 0.01:
                reasons.append("repeat_side_penalty")

        if symbol and selected_symbols.get(symbol, 0) > 0:
            duplicate_penalty = 0.18 + selected_symbols[symbol] * 0.12
            if strong_evidence and expected_net > 1.0:
                duplicate_penalty *= 0.60
            adjustments["duplicate_symbol_penalty"] = -_clamp(duplicate_penalty, 0.0, 0.42)
            reasons.append("duplicate_symbol_penalty")

        if runtime.get("release_pressure_active") and strong_evidence and expected_net > 0:
            release_bonus = 0.08 + min(max(capital_efficiency, 0.0), 2.0) * 0.04
            adjustments["release_rotation_bonus"] = _clamp(release_bonus, 0.0, 0.18)
            if adjustments["release_rotation_bonus"] >= 0.01:
                reasons.append("release_rotation_bonus")

        total_adjustment = round(sum(adjustments.values()), 6)
        return {
            "symbol": symbol,
            "side": side,
            "rank_context": {"rank": rank, "candidate_count": candidate_count},
            "runtime": {
                "underfilled": underfilled,
                "roster_gap": roster_gap,
                "capacity_mode": capacity_mode,
                "quality_bias": runtime.get("quality_bias"),
                "dominant_side": dominant_side,
                "available_slots": available_slots,
                "rotation_slots": rotation_slots,
                "release_pressure_active": bool(runtime.get("release_pressure_active")),
            },
            "signals": {
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(profit_quality, 6),
                "capital_efficiency_score": round(capital_efficiency, 6),
                "tail_risk_score": round(tail_risk, 6),
                "strong_aligned_profit_evidence": bool(strong_evidence),
            },
            "projected_side_share": round(projected_side_share, 6),
            "projected_long_count": projected_long,
            "projected_short_count": projected_short,
            "adjustments": {key: round(value, 6) for key, value in adjustments.items()},
            "reasons": reasons,
            "total_adjustment": total_adjustment,
        }

    @staticmethod
    def _annotate_queue_diagnostics(
        decision: DecisionOutput,
        *,
        runtime: dict[str, Any],
        queue_adjustment: dict[str, Any],
        base_score: float,
        adjusted_score: float,
        rank: int,
        candidate_count: int,
    ) -> None:
        raw = _safe_dict(getattr(decision, "raw_response", None))
        opportunity = _safe_dict(raw.get("opportunity_score"))
        opportunity["portfolio_queue"] = {
            "base_score": round(base_score, 6),
            "adjusted_score": round(adjusted_score, 6),
            "adjustment": round(_safe_float(queue_adjustment.get("total_adjustment"), 0.0), 6),
            "rank": int(rank),
            "candidate_count": int(candidate_count),
            "reasons": list(queue_adjustment.get("reasons") or []),
            "adjustments": _safe_dict(queue_adjustment.get("adjustments")),
            "runtime": {
                "underfilled": bool(runtime.get("underfilled")),
                "roster_gap": _safe_int(runtime.get("roster_gap"), 0),
                "capacity_mode": str(runtime.get("capacity_mode") or "balanced"),
                "available_slots": _safe_int(runtime.get("available_slots"), 0),
                "rotation_slots": _safe_int(runtime.get("rotation_slots"), 0),
                "dominant_side": str(runtime.get("dominant_side") or "neutral"),
                "release_pressure_active": bool(runtime.get("release_pressure_active")),
            },
            "projected_side_share": _safe_float(
                queue_adjustment.get("projected_side_share"),
                0.0,
            ),
            "policy": (
                "portfolio-aware queue ranking reorders close candidates by marginal"
                " expected-net value, side crowding, roster gap, and scarce-slot quality"
            ),
        }
        raw["opportunity_score"] = opportunity
        decision.raw_response = raw
