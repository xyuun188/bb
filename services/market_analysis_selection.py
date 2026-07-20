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

    VERSION = "2026-07-21.marginal-market-analysis-value.v1"

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
        self.history_loaded = False

    def clear(self) -> None:
        self._recent.clear()
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
        return min(total, max(final, final * max(int(self.params.candidate_pool_multiplier), 1)))

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

        discovery_slots = min(
            max(int(self.params.discovery_slots), 0),
            max(final_limit - 1, 0),
        )
        exploitation_slots = final_limit - discovery_slots
        selected_rows = ranked[:exploitation_slots]
        for row in selected_rows:
            row["selection_role"] = "advantage"
        selected_keys = {str(row["symbol_key"]) for row in selected_rows}

        discovery_candidates = [
            row
            for row in ranked
            if str(row["symbol_key"]) not in selected_keys
            and not bool(row["recent_unchanged"])
        ]
        for row in discovery_candidates[:discovery_slots]:
            row["selection_role"] = "discovery"
            selected_rows.append(row)
            selected_keys.add(str(row["symbol_key"]))

        discovery_assigned = sum(
            row.get("selection_role") == "discovery" for row in selected_rows
        )
        if discovery_assigned < discovery_slots:
            for row in reversed(selected_rows):
                if bool(row["recent_unchanged"]):
                    continue
                row["selection_role"] = "discovery"
                discovery_assigned += 1
                if discovery_assigned >= discovery_slots:
                    break

        if len(selected_rows) < final_limit:
            for row in ranked:
                if str(row["symbol_key"]) in selected_keys:
                    continue
                row["selection_role"] = "fallback_fill"
                selected_rows.append(row)
                selected_keys.add(str(row["symbol_key"]))
                if len(selected_rows) >= final_limit:
                    break

        selected = {
            str(row["symbol"]): row["feature"]
            for row in selected_rows
        }
        diagnostics = self._diagnostics(rows, selected_rows, final_limit, selected_at)
        return MarketAnalysisSelectionResult(selected=selected, diagnostics=diagnostics)

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
        penalty = (
            base_score * min(max(float(self.params.unchanged_repeat_penalty_ratio), 0.0), 1.0)
            if recent_unchanged
            else 0.0
        )
        evaluation_score = max(base_score - penalty, 0.0)
        if observation is None:
            status = "not_recently_analyzed"
        elif material_change:
            status = "material_change_bypass"
        elif recent_unchanged:
            status = "recent_unchanged_penalty"
        else:
            status = "cooldown_expired"
        return {
            "symbol": symbol,
            "symbol_key": key,
            "feature": feature,
            "rank_before_selection": rank,
            "base_advantage_score": round(base_score, 6),
            "repeat_penalty": round(penalty, 6),
            "evaluation_score": round(evaluation_score, 6),
            "recent_age_seconds": None if age_seconds is None else round(age_seconds, 3),
            "recent_unchanged": recent_unchanged,
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
    ) -> dict[str, Any]:
        selected_keys = {str(row["symbol_key"]) for row in selected_rows}
        recent_excluded = [
            row for row in rows if row["recent_unchanged"] and row["symbol_key"] not in selected_keys
        ]
        selected_details = [self._public_row(row) for row in selected_rows]
        return {
            "version": self.VERSION,
            "read_only": True,
            "is_entry_gate": False,
            "candidate_count": len(rows),
            "final_limit": int(final_limit),
            "selected_count": len(selected_rows),
            "selected_symbols": [str(row["symbol"]) for row in selected_rows],
            "selected": selected_details,
            "cooldown_seconds": int(self.params.cooldown_seconds),
            "unchanged_repeat_penalty_ratio": round(
                float(self.params.unchanged_repeat_penalty_ratio), 6
            ),
            "discovery_slots": min(
                max(int(self.params.discovery_slots), 0),
                max(int(final_limit) - 1, 0),
            ),
            "discovery_selected_symbols": [
                str(row["symbol"])
                for row in selected_rows
                if row.get("selection_role") == "discovery"
            ],
            "material_change_bypass_count": sum(bool(row["material_change"]) for row in rows),
            "recent_unchanged_candidate_count": sum(
                bool(row["recent_unchanged"]) for row in rows
            ),
            "skipped_count": len(recent_excluded),
            "skipped_symbols": [str(row["symbol"]) for row in recent_excluded],
            "candidate_sample": [self._public_row(row) for row in rows[:12]],
            "generated_at": selected_at.isoformat(),
            "reason": (
                "Rank by current market advantage, penalize only unchanged recent analyses, "
                "reserve discovery capacity when available, and allow material market changes "
                "to bypass the cooldown. This controls expert-analysis allocation only."
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
                "evaluation_score",
                "recent_age_seconds",
                "recent_unchanged",
                "material_change",
                "material_change_reasons",
                "selection_status",
            )
        }
