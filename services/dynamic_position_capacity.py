"""Dynamic open-position capacity policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from typing import Any

from config.settings import DEFAULT_MAX_OPEN_POSITIONS_PER_MODEL, settings
from services.position_quality import PositionQualityScorer

MaxOpenPositionsProvider = Callable[[], int]
CAPACITY_MIN_FLOOR_RATIO = 0.25
CAPACITY_OPERATING_FLOOR_RATIO = 0.35
CAPACITY_OPEN_BOOK_FLOOR_RATIO = 0.75
CAPACITY_ROTATION_SLOT_RATIO = 0.15
CAPACITY_RELEASE_SLOT_RATIO = 0.20
CAPACITY_LOW_QUALITY_HIGH_RATIO = 0.45
CAPACITY_LOW_QUALITY_WARN_RATIO = 0.25
CAPACITY_MAX_EXPANSION_RATIO = 1.25


@dataclass(frozen=True, slots=True)
class DynamicCapacityDecision:
    base_limit: int
    target_limit: int
    effective_limit: int
    entry_limit: int
    open_group_count: int
    low_quality_count: int
    low_quality_ratio: float
    market_confidence: float
    recent_win_rate: float
    drawdown_ratio: float
    release_candidate_count: int
    reason: str
    factors: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_limit": self.base_limit,
            "target_limit": self.target_limit,
            "effective_limit": self.effective_limit,
            "entry_limit": self.entry_limit,
            "open_group_count": self.open_group_count,
            "low_quality_count": self.low_quality_count,
            "low_quality_ratio": round(self.low_quality_ratio, 6),
            "market_confidence": round(self.market_confidence, 6),
            "recent_win_rate": round(self.recent_win_rate, 6),
            "drawdown_ratio": round(self.drawdown_ratio, 6),
            "release_candidate_count": self.release_candidate_count,
            "reason": self.reason,
            "factors": self.factors,
        }


class DynamicPositionCapacityPolicy:
    """Convert a configured hard cap into a runtime capacity envelope."""

    def __init__(
        self,
        max_open_positions_provider: MaxOpenPositionsProvider | None = None,
        *,
        quality_scorer: PositionQualityScorer | None = None,
    ) -> None:
        self.max_open_positions_provider = max_open_positions_provider or (
            lambda: settings.max_open_positions_per_model
        )
        self.quality_scorer = quality_scorer or PositionQualityScorer()

    def evaluate(
        self,
        *,
        open_positions: list[dict[str, Any]],
        strategy_context: dict[str, Any] | None = None,
        market_regime: dict[str, Any] | None = None,
        account_equity: float | None = None,
        active_strategy_profile_id: str | None = None,
    ) -> DynamicCapacityDecision:
        configured_limit = self._configured_limit()
        rows = [row for row in (open_positions or []) if row.get("is_open", True) is not False]
        open_group_count = self._open_group_count(rows)
        market_confidence = self._market_confidence(market_regime, strategy_context)
        recent_win_rate = self._recent_win_rate(strategy_context)
        drawdown_ratio = self._drawdown_ratio(strategy_context, account_equity)
        quality_rows = self._quality_rows(
            rows, active_strategy_profile_id=active_strategy_profile_id
        )
        low_quality_count = sum(1 for item in quality_rows if item[1].should_release)
        low_quality_ratio = low_quality_count / max(len(quality_rows), 1)
        release_candidate_count = sum(1 for item in quality_rows if item[1].bucket == "release_now")
        strategy_rotation_slots = self._strategy_rotation_slots(strategy_context)
        policy_rotation_slots = self._rotation_slot_count(
            open_group_count=open_group_count,
            low_quality_count=low_quality_count,
            release_candidate_count=release_candidate_count,
            configured_limit=configured_limit,
        )
        rotation_slots = max(policy_rotation_slots, strategy_rotation_slots)
        base_limit = configured_limit
        learned_target_limit = self._learned_target_limit(strategy_context)
        target_limit = self._target_limit(
            base_limit,
            open_group_count=open_group_count,
            learned_target_limit=learned_target_limit,
        )

        min_limit = self._minimum_limit(
            target_limit,
            open_group_count=open_group_count,
            rotation_slots=rotation_slots,
            base_limit=base_limit,
        )
        max_limit = max(base_limit, ceil(target_limit * CAPACITY_MAX_EXPANSION_RATIO))
        effective_limit = target_limit
        reason_codes: list[str] = []
        reasons: list[str] = [
            f"配置容量 {configured_limit} 组",
            f"运行容量 {base_limit} 组",
            f"执行目标 {target_limit} 组",
        ]
        if learned_target_limit is not None:
            reasons.append(f"学习建议 {learned_target_limit} 组")
        if learned_target_limit is not None and target_limit > base_limit:
            reasons.append(f"学习目标高于当前运行容量，允许扩展到 {target_limit} 组")
            reason_codes.append("learned_target_expansion")
        if strategy_rotation_slots > 0:
            reasons.append(f"策略学习建议预留 {strategy_rotation_slots} 个轮换槽")
            reason_codes.append("strategy_rotation_slots")
        if rotation_slots > 0:
            reasons.append(f"低质量持仓释放中，预留 {rotation_slots} 个小仓轮换槽")
            reason_codes.append("release_rotation_slots")
        factors: dict[str, Any] = {
            "configured_limit": configured_limit,
            "learned_target_limit": learned_target_limit,
            "rotation_slots": rotation_slots,
            "strategy_rotation_slots": strategy_rotation_slots,
            "policy_rotation_slots": policy_rotation_slots,
            "market_confidence": round(market_confidence, 6),
            "recent_win_rate": round(recent_win_rate, 6),
            "drawdown_ratio": round(drawdown_ratio, 6),
            "low_quality_ratio": round(low_quality_ratio, 6),
            "min_limit": min_limit,
            "max_limit": max_limit,
        }

        if low_quality_count:
            if low_quality_ratio >= CAPACITY_LOW_QUALITY_HIGH_RATIO:
                reduction = min(
                    max(ceil(target_limit * 0.30), low_quality_count),
                    max(target_limit - min_limit, 0),
                )
                effective_limit = max(target_limit - reduction, min_limit)
                reasons.append(f"低质量持仓压力 {low_quality_count} 组")
                reason_codes.append("low_quality_pressure")
            elif low_quality_ratio >= CAPACITY_LOW_QUALITY_WARN_RATIO:
                reduction = min(
                    max(1, ceil(low_quality_count / 2)),
                    max(target_limit - min_limit, 0),
                )
                effective_limit = max(target_limit - reduction, min_limit)
                reasons.append(f"低质量持仓预警 {low_quality_count} 组")
                reason_codes.append("low_quality_warn")
            else:
                reasons.append(f"低质量持仓观察 {low_quality_count} 组")
                reason_codes.append("low_quality_carry")

        if drawdown_ratio >= 0.05:
            reduction = min(ceil(target_limit * 0.30), max(target_limit - min_limit, 0))
            effective_limit = max(effective_limit - reduction, min_limit)
            reasons.append(f"当日回撤 {drawdown_ratio:.1%}")
            reason_codes.append("drawdown")
        elif drawdown_ratio >= 0.02:
            reduction = min(ceil(target_limit * 0.18), max(target_limit - min_limit, 0))
            effective_limit = max(effective_limit - reduction, min_limit)
            reasons.append(f"当日回撤观察 {drawdown_ratio:.1%}")
            reason_codes.append("drawdown_watch")

        if recent_win_rate >= 0.58 and low_quality_count == 0 and drawdown_ratio < 0.01:
            bonus = min(
                max(1, round(target_limit * min(market_confidence, 0.85) * 0.12)),
                max(max_limit - target_limit, 0),
            )
            effective_limit = min(target_limit + bonus, max_limit)
            reasons.append(f"胜率和行情清晰，扩展 {bonus} 个机会槽")
            reason_codes.append("trend_bonus")
        elif market_confidence >= 0.72 and low_quality_count == 0 and drawdown_ratio < 0.015:
            bonus = min(
                max(1, round(target_limit * (market_confidence - 0.65) * 0.10)),
                max(max_limit - target_limit, 0),
            )
            effective_limit = min(max(effective_limit, target_limit) + bonus, max_limit)
            reasons.append(f"行情方向清晰度 {market_confidence:.0%}")
            reason_codes.append("market_clear")

        if open_group_count > base_limit and low_quality_count > 0:
            effective_limit = min(effective_limit, target_limit)
            reasons.append("持仓超过运行容量，优先释放低质量旧仓")
            reason_codes.append("over_capacity_release_first")

        effective_limit = max(min(effective_limit, max_limit), min_limit)
        entry_limit = self._entry_limit(
            effective_limit=effective_limit,
            target_limit=target_limit,
            open_group_count=open_group_count,
            rotation_slots=rotation_slots,
            release_candidate_count=release_candidate_count,
        )
        if entry_limit > effective_limit:
            reasons.append(
                f"为释放低质量旧仓，额外开放 {entry_limit - effective_limit} 个轮换开仓槽"
            )
            reason_codes.append("rotation_entry_expansion")

        factors["open_group_count"] = open_group_count
        factors["low_quality_count"] = low_quality_count
        factors["release_candidate_count"] = release_candidate_count
        factors["entry_limit"] = entry_limit
        factors["reason_codes"] = reason_codes
        return DynamicCapacityDecision(
            base_limit=base_limit,
            target_limit=target_limit,
            effective_limit=effective_limit,
            entry_limit=entry_limit,
            open_group_count=open_group_count,
            low_quality_count=low_quality_count,
            low_quality_ratio=low_quality_ratio,
            market_confidence=market_confidence,
            recent_win_rate=recent_win_rate,
            drawdown_ratio=drawdown_ratio,
            release_candidate_count=release_candidate_count,
            reason="；".join(reasons),
            factors=factors,
        )

    @classmethod
    def _learned_target_limit(cls, strategy_context: dict[str, Any] | None) -> int | None:
        context = strategy_context if isinstance(strategy_context, dict) else {}
        roster = _safe_dict(context.get("portfolio_roster"))
        learning = _safe_dict(context.get("strategy_learning"))
        candidates = [
            context.get("target_open_position_groups"),
            context.get("target_position_groups"),
            roster.get("target_position_groups"),
            learning.get("target_position_groups"),
            learning.get("max_open_positions"),
        ]
        for value in candidates:
            target = int(_safe_float(value, 0.0))
            if target > 0:
                return max(target, 1)
        return None

    def _configured_limit(self) -> int:
        try:
            raw = int(float(self.max_open_positions_provider() or 0))
        except (TypeError, ValueError):
            raw = 0
        if raw <= 0:
            return DEFAULT_MAX_OPEN_POSITIONS_PER_MODEL
        return raw

    @staticmethod
    def _strategy_rotation_slots(strategy_context: dict[str, Any] | None) -> int:
        context = strategy_context if isinstance(strategy_context, dict) else {}
        candidates = [
            context.get("rotation_slots"),
            _safe_dict(context.get("portfolio_roster")).get("rotation_slots"),
            _safe_dict(context.get("strategy_learning")).get("rotation_slots"),
            _safe_dict(_safe_dict(context.get("strategy_learning")).get("runtime")).get(
                "rotation_slots"
            ),
        ]
        for value in candidates:
            slots = int(_safe_float(value, 0.0))
            if slots > 0:
                return slots
        return 0

    @staticmethod
    def _target_limit(
        base_limit: int,
        *,
        open_group_count: int,
        learned_target_limit: int | None,
    ) -> int:
        adaptive_floor = DynamicPositionCapacityPolicy._adaptive_tradeable_floor(
            base_limit,
            open_group_count=open_group_count,
        )
        if learned_target_limit is None:
            return max(base_limit, adaptive_floor)
        if learned_target_limit < base_limit and open_group_count >= base_limit:
            return base_limit
        return max(learned_target_limit, adaptive_floor)

    @staticmethod
    def _adaptive_tradeable_floor(base_limit: int, *, open_group_count: int) -> int:
        base = max(base_limit, 1)
        floor = max(1, ceil(base * CAPACITY_OPERATING_FLOOR_RATIO))
        if open_group_count > 0:
            floor = max(
                floor,
                min(base, ceil(open_group_count * CAPACITY_OPEN_BOOK_FLOOR_RATIO)),
            )
        return min(base, floor)

    @staticmethod
    def _rotation_slot_count(
        *,
        open_group_count: int,
        low_quality_count: int,
        release_candidate_count: int,
        configured_limit: int,
    ) -> int:
        if open_group_count <= 0 or low_quality_count <= 0:
            return 0
        pressure_count = max(low_quality_count, release_candidate_count, 1)
        pressure_slots = max(1, ceil(pressure_count * CAPACITY_RELEASE_SLOT_RATIO))
        envelope_slots = max(
            1,
            ceil(max(configured_limit, open_group_count, 1) * CAPACITY_ROTATION_SLOT_RATIO),
        )
        return min(pressure_slots, envelope_slots)

    @staticmethod
    def _minimum_limit(
        target_limit: int,
        *,
        open_group_count: int,
        rotation_slots: int,
        base_limit: int,
    ) -> int:
        floor = max(1, ceil(max(target_limit, 1) * CAPACITY_MIN_FLOOR_RATIO))
        if rotation_slots > 0 and open_group_count < base_limit:
            floor = max(floor, min(base_limit, open_group_count + rotation_slots))
        return min(max(base_limit, 1), floor)

    @staticmethod
    def _entry_limit(
        *,
        effective_limit: int,
        target_limit: int,
        open_group_count: int,
        rotation_slots: int,
        release_candidate_count: int,
    ) -> int:
        entry_limit = max(effective_limit, 1)
        if open_group_count >= entry_limit:
            return entry_limit
        if rotation_slots <= 0 and release_candidate_count <= 0:
            return entry_limit
        rotation_ceiling = min(
            target_limit,
            open_group_count + max(rotation_slots, release_candidate_count, 1),
        )
        return max(entry_limit, rotation_ceiling)

    def _quality_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        active_strategy_profile_id: str | None,
    ) -> list[tuple[dict[str, Any], Any]]:
        quality_rows: list[tuple[dict[str, Any], Any]] = []
        by_group: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            side = str(row.get("side") or "").strip().lower() or "unknown"
            key = (symbol, side)
            group = by_group.setdefault(key, {**row, "_parts": 0})
            group["_parts"] = int(group.get("_parts", 0)) + 1
            if not group.get("strategy_profile_id") and row.get("strategy_profile_id"):
                group["strategy_profile_id"] = row.get("strategy_profile_id")
            if not group.get("profile_id") and row.get("profile_id"):
                group["profile_id"] = row.get("profile_id")
        for group in by_group.values():
            quality = self.quality_scorer.score(
                group,
                active_strategy_profile_id=active_strategy_profile_id,
                same_symbol_side_parts=int(group.get("_parts", 1)),
            )
            quality_rows.append((group, quality))
        quality_rows.sort(
            key=lambda item: (
                item[1].score,
                _safe_float(item[0].get("unrealized_pnl"), 0.0),
                -item[1].hold_hours,
            )
        )
        return quality_rows

    @staticmethod
    def _open_group_count(rows: list[dict[str, Any]]) -> int:
        groups: set[tuple[str, str]] = set()
        for row in rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            side = str(row.get("side") or "").strip().lower() or "unknown"
            groups.add((symbol, side))
        return len(groups)

    @staticmethod
    def _market_confidence(
        market_regime: dict[str, Any] | None,
        strategy_context: dict[str, Any] | None,
    ) -> float:
        values = []
        for source in (market_regime, strategy_context):
            if not isinstance(source, dict):
                continue
            for key in ("confidence", "trend_confidence", "market_confidence"):
                value = source.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if isinstance(source.get("market_regime"), dict):
                nested = source["market_regime"]
                for key in ("confidence", "trend_confidence"):
                    value = nested.get(key)
                    if isinstance(value, (int, float)):
                        values.append(float(value))
        if not values:
            return 0.5
        return max(0.0, min(sum(values) / len(values), 1.0))

    @staticmethod
    def _recent_win_rate(strategy_context: dict[str, Any] | None) -> float:
        if not isinstance(strategy_context, dict):
            return 0.5
        for key in ("recent_win_rate", "win_rate", "today_win_rate"):
            value = strategy_context.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, min(float(value), 1.0))
        return 0.5

    @staticmethod
    def _drawdown_ratio(
        strategy_context: dict[str, Any] | None,
        account_equity: float | None,
    ) -> float:
        if not isinstance(strategy_context, dict):
            return 0.0
        equity = max(float(account_equity or 0.0), 0.0)
        risk_pnl = strategy_context.get("today_risk_pnl")
        if not isinstance(risk_pnl, (int, float)) or equity <= 0:
            return 0.0
        loss = max(-float(risk_pnl), 0.0)
        return min(loss / equity, 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
