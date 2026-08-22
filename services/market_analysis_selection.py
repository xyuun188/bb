"""Select market symbols that have the highest marginal analysis value."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from services.trading_params import MarketAnalysisSelectionParams

NormalizeSymbol = Callable[[Any], str]
AdvantageScorer = Callable[[Any], float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _feature_value(feature: Any, key: str) -> Any:
    if isinstance(feature, dict):
        return feature.get(key)
    return getattr(feature, key, None)


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    return None


@dataclass(frozen=True, slots=True)
class MarketAnalysisObservation:
    observed_at: datetime
    feature_snapshot: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MarketAnalysisSelectionResult:
    selected: dict[str, Any]
    diagnostics: dict[str, Any]


class MarketAnalysisSelectionPolicy:
    """Balance ranked advantage with the incremental value of another AI review."""

    VERSION = "2026-08-22.coverage-backlog.v8"

    def __init__(
        self,
        *,
        normalize_symbol: NormalizeSymbol,
        advantage_scorer: AdvantageScorer,
        params: MarketAnalysisSelectionParams,
    ) -> None:
        self.normalize_symbol = normalize_symbol
        self.advantage_scorer = advantage_scorer
        self.params = params
        self._recent: dict[str, MarketAnalysisObservation] = {}
        self._selection_round = 0
        self.history_loaded = False

    def clear(self) -> None:
        self._recent.clear()
        self._selection_round = 0
        self.history_loaded = False

    def remember(
        self,
        symbol: str,
        feature: Any,
        *,
        observed_at: datetime | str | None = None,
    ) -> None:
        key = self.normalize_symbol(symbol)
        timestamp = _as_utc(observed_at) or datetime.now(UTC)
        if not key:
            return
        snapshot = dict(feature) if isinstance(feature, dict) else self._snapshot(feature)
        current = self._recent.get(key)
        if current is None or timestamp >= current.observed_at:
            self._recent[key] = MarketAnalysisObservation(timestamp, snapshot)

    def candidate_pool_limit(self, final_limit: int, candidate_count: int) -> int:
        final = max(0, int(final_limit or 0))
        total = max(0, int(candidate_count or 0))
        if final <= 0 or total <= 0:
            return 0
        multiplier = max(int(self.params.candidate_pool_multiplier), 1)
        minimum = max(int(getattr(self.params, "candidate_pool_minimum", 0) or 0), final)
        return min(total, max(final * multiplier, minimum))

    def select(
        self,
        candidates: dict[str, Any],
        limit: int,
        *,
        now: datetime | None = None,
    ) -> MarketAnalysisSelectionResult:
        final_limit = min(max(0, int(limit or 0)), len(candidates or {}))
        selected_at = _as_utc(now) or datetime.now(UTC)
        if final_limit <= 0:
            return MarketAnalysisSelectionResult(
                selected={},
                diagnostics=self._diagnostics([], [], final_limit, selected_at),
            )

        rows = [
            self._candidate_row(symbol, feature, rank=index + 1, now=selected_at)
            for index, (symbol, feature) in enumerate((candidates or {}).items())
        ]
        ranked = sorted(
            rows,
            key=lambda row: (
                _safe_float(row["evaluation_score"]),
                _safe_float(row["base_advantage_score"]),
                -int(row["rank_before_selection"]),
            ),
            reverse=True,
        )

        # A repeat penalty is useful for ranking, but it is not permission to
        # call a symbol again.  The cooldown is a hard scheduling contract:
        # when the current feature pool contains only recently completed
        # symbols, return an underfilled result and let the upstream rotating
        # pool provide different symbols on a later round.  Reusing a recent
        # symbol merely to keep the activity counter non-zero was the source
        # of sub-minute duplicate analyses in production.
        cooldown_excluded = [row for row in ranked if bool(row["recent"])]
        eligible_ranked = [row for row in ranked if not bool(row["recent"])]

        self._selection_round += 1
        coverage_candidates = [row for row in eligible_ranked if bool(row["coverage_due"])]
        aged_coverage_candidates = [
            row for row in coverage_candidates if not bool(row.get("never_analyzed"))
        ]
        oldest_due_age_seconds = max(
            (
                _safe_float(row.get("recent_age_seconds"), 0.0)
                for row in aged_coverage_candidates
            ),
            default=0.0,
        )
        coverage_slots = self._coverage_slot_count(
            final_limit,
            has_coverage_candidates=bool(coverage_candidates),
            # Never-analyzed symbols are already handled by the normal
            # discovery slot.  Only previously observed symbols contribute to
            # the backlog pressure that reserves an extra coverage slot.
            coverage_due_count=len(aged_coverage_candidates),
            oldest_due_age_seconds=oldest_due_age_seconds,
        )
        advantage_slots = final_limit - coverage_slots
        selected_rows = eligible_ranked[:advantage_slots]
        for row in selected_rows:
            row["selection_role"] = "advantage"
        selected_keys = {str(row["symbol_key"]) for row in selected_rows}

        coverage_ranked = sorted(
            (row for row in coverage_candidates if str(row["symbol_key"]) not in selected_keys),
            key=lambda row: (
                # Previously observed overdue symbols must converge before a
                # never-analyzed discovery row can consume the same coverage
                # slot.  Fresh discovery still remains eligible through the
                # normal advantage slot and the bounded fallback below.
                not bool(row["never_analyzed"]),
                _safe_float(row["recent_age_seconds"], float("inf")),
                _safe_float(row["evaluation_score"]),
            ),
            reverse=True,
        )
        for row in coverage_ranked[:coverage_slots]:
            row["selection_role"] = "coverage"
            selected_rows.append(row)
            selected_keys.add(str(row["symbol_key"]))

        coverage_assigned = sum(row.get("selection_role") == "coverage" for row in selected_rows)
        if coverage_assigned < coverage_slots:
            for row in reversed(selected_rows):
                if (
                    not bool(row["coverage_due"])
                    or row.get("selection_role") == "coverage"
                ):
                    continue
                row["selection_role"] = "coverage"
                coverage_assigned += 1
                if coverage_assigned >= coverage_slots:
                    break

        if len(selected_rows) < final_limit:
            advantage_assigned = sum(
                row.get("selection_role") == "advantage" for row in selected_rows
            )
            for row in eligible_ranked:
                if str(row["symbol_key"]) in selected_keys:
                    continue
                if advantage_assigned < advantage_slots:
                    row["selection_role"] = "advantage"
                    advantage_assigned += 1
                else:
                    row["selection_role"] = "fallback_fill"
                selected_rows.append(row)
                selected_keys.add(str(row["symbol_key"]))
                if len(selected_rows) >= final_limit:
                    break

        selected = {str(row["symbol"]): row["feature"] for row in selected_rows}
        diagnostics = self._diagnostics(
            rows,
            selected_rows,
            final_limit,
            selected_at,
            cooldown_excluded=cooldown_excluded,
        )
        return MarketAnalysisSelectionResult(selected=selected, diagnostics=diagnostics)

    def _coverage_slot_count(
        self,
        final_limit: int,
        *,
        has_coverage_candidates: bool,
        coverage_due_count: int = 0,
        oldest_due_age_seconds: float = 0.0,
    ) -> int:
        if not has_coverage_candidates or final_limit <= 0:
            return 0
        configured = max(int(self.params.coverage_slots), 0)
        if configured <= 0:
            return 0
        if final_limit > 1:
            # One coverage slot is enough while the queue is healthy.  Once
            # due coverage accumulates, reserve a second bounded slot so a
            # large backlog converges inside the 30-minute contract without
            # turning the market loop into a coverage-only loop.
            backlog_slots = 1 if coverage_due_count > 1 else 0
            overdue_slots = (
                1
                if oldest_due_age_seconds >= float(self.params.coverage_target_seconds) * 1.5
                else 0
            )
            return min(
                final_limit - 1,
                max(configured, configured + backlog_slots + overdue_slots),
            )
        interval = max(int(self.params.single_slot_coverage_interval), 1)
        return 1 if self._selection_round % interval == 0 else 0

    def _candidate_row(
        self,
        symbol: str,
        feature: Any,
        *,
        rank: int,
        now: datetime,
    ) -> dict[str, Any]:
        key = self.normalize_symbol(symbol)
        base_score = max(_safe_float(self.advantage_scorer(feature)), 0.0)
        observation = self._recent.get(key)
        age_seconds: float | None = None
        changes: list[dict[str, Any]] = []
        material_change = False
        if observation is not None:
            age_seconds = max((now - observation.observed_at).total_seconds(), 0.0)
            changes = self._material_changes(feature, observation.feature_snapshot)
            material_change = bool(changes)
        recent = bool(
            observation is not None
            and age_seconds is not None
            and age_seconds < float(self.params.cooldown_seconds)
        )
        recent_unchanged = bool(recent and not material_change)
        repeat_penalty_ratio = 0.0
        if recent:
            repeat_penalty_ratio = (
                self.params.material_change_repeat_penalty_ratio
                if material_change
                else self.params.unchanged_repeat_penalty_ratio
            )
        repeat_penalty_ratio = min(max(float(repeat_penalty_ratio), 0.0), 1.0)
        penalty = base_score * repeat_penalty_ratio
        evaluation_score = max(base_score - penalty, 0.0)
        coverage_due = bool(
            observation is None
            or age_seconds is None
            or age_seconds >= float(self.params.coverage_target_seconds)
        )
        if observation is None:
            status = "not_recently_analyzed"
        elif recent and material_change:
            status = "recent_material_change_cooldown"
        elif recent_unchanged:
            status = "recent_unchanged_cooldown"
        elif material_change:
            status = "material_change_after_cooldown"
        else:
            status = "cooldown_expired"
        return {
            "symbol": symbol,
            "symbol_key": key,
            "feature": feature,
            "rank_before_selection": rank,
            "base_advantage_score": round(base_score, 6),
            "repeat_penalty": round(penalty, 6),
            "repeat_penalty_ratio": round(repeat_penalty_ratio, 6),
            "evaluation_score": round(evaluation_score, 6),
            "recent_age_seconds": None if age_seconds is None else round(age_seconds, 3),
            "recent": recent,
            "recent_unchanged": recent_unchanged,
            "never_analyzed": observation is None,
            "coverage_due": coverage_due,
            "material_change": material_change,
            "material_change_reasons": changes,
            "selection_status": status,
        }

    def _material_changes(
        self,
        current: Any,
        previous: dict[str, Any],
    ) -> list[dict[str, Any]]:
        checks = (
            ("current_price", "relative", self.params.material_price_change_ratio),
            (
                "entry_activity_volume_ratio",
                "relative",
                self.params.material_volume_ratio_change_ratio,
            ),
            ("adx_14", "absolute", self.params.material_adx_change),
            ("returns_5", "absolute", self.params.material_return_change),
            (
                "volatility_20",
                "relative",
                self.params.material_volatility_change_ratio,
            ),
        )
        changes: list[dict[str, Any]] = []
        for key, kind, threshold in checks:
            if key not in previous or previous.get(key) is None:
                continue
            current_value = _safe_float(_feature_value(current, key))
            previous_value = _safe_float(previous.get(key))
            if kind == "relative":
                delta = abs(current_value - previous_value) / max(
                    abs(previous_value),
                    float(self.params.relative_change_floor),
                )
            else:
                delta = abs(current_value - previous_value)
            if delta >= float(threshold):
                changes.append(
                    {
                        "feature": key,
                        "change": round(delta, 6),
                        "threshold": round(float(threshold), 6),
                    }
                )
        return changes

    @staticmethod
    def _snapshot(feature: Any) -> dict[str, Any]:
        return {
            key: _feature_value(feature, key)
            for key in (
                "current_price",
                "entry_activity_volume_ratio",
                "adx_14",
                "returns_5",
                "volatility_20",
            )
        }

    def _diagnostics(
        self,
        rows: list[dict[str, Any]],
        selected_rows: list[dict[str, Any]],
        final_limit: int,
        selected_at: datetime,
        *,
        cooldown_excluded: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cooldown_excluded = list(cooldown_excluded or [])
        selected_keys = {str(row["symbol_key"]) for row in selected_rows}
        coverage_due_rows = [row for row in rows if bool(row["coverage_due"])]
        coverage_due_unselected = [
            row for row in coverage_due_rows if str(row["symbol_key"]) not in selected_keys
        ]
        recent_excluded = [
            row
            for row in cooldown_excluded
            if row["symbol_key"] not in selected_keys
        ]
        selected_details = [self._public_row(row) for row in selected_rows]
        coverage_selected = [
            str(row["symbol"]) for row in selected_rows if row.get("selection_role") == "coverage"
        ]
        advantage_selected = [
            str(row["symbol"]) for row in selected_rows if row.get("selection_role") == "advantage"
        ]
        cooldown_fill_selected = [
            str(row["symbol"])
            for row in selected_rows
            if row.get("selection_role") == "cooldown_fill"
        ]
        return {
            "version": self.VERSION,
            "read_only": True,
            "is_entry_gate": False,
            "candidate_count": len(rows),
            "final_limit": int(final_limit),
            "selected_count": len(selected_rows),
            "selected_symbols": [str(row["symbol"]) for row in selected_rows],
            "cooldown_excluded_count": len(cooldown_excluded),
            "cooldown_excluded_symbols": [
                str(row["symbol"]) for row in cooldown_excluded
            ],
            # Retained as a compatibility field for existing dashboard clients;
            # strict v7 never populates it because a recent symbol is never a
            # valid fill candidate.
            "cooldown_fill_count": len(cooldown_fill_selected),
            "cooldown_fill_symbols": cooldown_fill_selected,
            "cooldown_underfilled": bool(
                cooldown_excluded and len(selected_rows) < final_limit
            ),
            "underfill_reason": (
                "all_candidates_in_cooldown"
                if cooldown_excluded
                and not any(not bool(row["recent"]) for row in rows)
                and len(selected_rows) < final_limit
                else (
                    "fresh_candidate_pool_exhausted"
                    if len(selected_rows) < final_limit
                    else None
                )
            ),
            "selected": selected_details,
            "cooldown_seconds": int(self.params.cooldown_seconds),
            "unchanged_repeat_penalty_ratio": round(
                float(self.params.unchanged_repeat_penalty_ratio), 6
            ),
            "material_change_repeat_penalty_ratio": round(
                float(self.params.material_change_repeat_penalty_ratio), 6
            ),
            "coverage_target_seconds": int(self.params.coverage_target_seconds),
            "coverage_configured_slots": int(self.params.coverage_slots),
            "single_slot_coverage_interval": int(self.params.single_slot_coverage_interval),
            "selection_round": int(self._selection_round),
            "coverage_due_candidate_count": sum(bool(row["coverage_due"]) for row in rows),
            "coverage_due_symbols": [str(row["symbol"]) for row in coverage_due_rows],
            "coverage_due_unselected_count": len(coverage_due_unselected),
            "coverage_due_unselected_symbols": [
                str(row["symbol"]) for row in coverage_due_unselected
            ],
            "coverage_due_resolved_count": 0,
            "coverage_due_resolved_symbols": [],
            "never_analyzed_candidate_count": sum(bool(row["never_analyzed"]) for row in rows),
            "oldest_completed_analysis_age_seconds": round(
                max(
                    (
                        _safe_float(row["recent_age_seconds"])
                        for row in rows
                        if row["recent_age_seconds"] is not None
                    ),
                    default=0.0,
                ),
                3,
            ),
            "coverage_selected_count": len(coverage_selected),
            "coverage_selected_symbols": coverage_selected,
            "advantage_selected_count": len(advantage_selected),
            "advantage_selected_symbols": advantage_selected,
            "recent_material_change_count": sum(
                bool(row["material_change"])
                and row.get("selection_status") == "recent_material_change_cooldown"
                for row in rows
            ),
            "recent_candidate_count": sum(bool(row["recent"]) for row in rows),
            "recent_unchanged_candidate_count": sum(bool(row["recent_unchanged"]) for row in rows),
            "skipped_count": len(recent_excluded),
            "skipped_symbols": [str(row["symbol"]) for row in recent_excluded],
            "candidate_sample": [self._public_row(row) for row in rows[:12]],
            "generated_at": selected_at.isoformat(),
            "reason": (
                "Allocate expert analysis between current advantage and overdue coverage. "
                "Completed symbols stay out for the full cooldown. If the current pool is "
                "entirely recent, the result is intentionally underfilled and the upstream "
                "rotating pool must supply other symbols; no cooldown fill is permitted. "
                "This controls expert-analysis allocation only."
            ),
            "diagnostic_boundary": (
                "Analysis scheduling only; it cannot authorize entry, change rank eligibility, "
                "OKX instrument availability, profitability evidence, sizing, leverage, or risk vetoes."
            ),
        }

    @staticmethod
    def _public_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            key: row.get(key)
            for key in (
                "symbol",
                "selection_role",
                "rank_before_selection",
                "base_advantage_score",
                "repeat_penalty",
                "repeat_penalty_ratio",
                "evaluation_score",
                "recent_age_seconds",
                "recent",
                "recent_unchanged",
                "never_analyzed",
                "coverage_due",
                "material_change",
                "material_change_reasons",
                "selection_status",
            )
        }
