from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput

NormalizeSymbol = Callable[[Any], str | None]
MaxOpenPositionsProvider = Callable[[], Any]
StagedEntryCounts = dict[str, dict[Any, int]]


@dataclass(frozen=True, slots=True)
class EntryCapacityPolicy:
    """Limit new entries without blocking same-symbol adds that manage an existing position."""

    normalize_symbol: NormalizeSymbol
    max_open_positions_per_model_provider: MaxOpenPositionsProvider

    def empty_staged_counts(self) -> StagedEntryCounts:
        """Return the per-round staged-entry counters used before orders are submitted."""

        return {"model_totals": {}, "symbol_side": {}, "side_totals": {}}

    def reason(
        self,
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict],
        staged_entry_counts: StagedEntryCounts,
    ) -> str | None:
        if not decision.is_entry:
            return None

        side = "long" if decision.action == Action.LONG else "short"
        symbol_key = self.normalize_symbol(decision.symbol)
        staged_symbol_side = self._stage_dict(staged_entry_counts, "symbol_side")
        staged_model_totals = self._stage_dict(staged_entry_counts, "model_totals")
        existing_same_symbol = sum(
            1
            for position in open_positions
            if self._is_effective_open_position(position)
            and position.get("model_name") == model_name
            and self.normalize_symbol(position.get("symbol")) == symbol_key
            and position.get("side") == side
        )
        staged_key = (model_name, symbol_key, side)
        existing_same_symbol += int(staged_symbol_side.get(staged_key, 0))
        is_same_symbol_add = existing_same_symbol > 0

        capacity = self._capacity_context()
        position_list_open_count = self._model_open_group_count(model_name, open_positions)
        capacity_open_count = self._capacity_open_group_count(capacity)
        live_open_count = max(position_list_open_count, capacity_open_count)
        staged_open_count = int(staged_model_totals.get(model_name, 0))
        model_open_count = live_open_count + staged_open_count
        max_open_positions = self._capacity_limit(capacity)
        if (
            not is_same_symbol_add
            and max_open_positions > 0
            and model_open_count >= max_open_positions
        ):
            capacity_detail = self._capacity_count_text(
                live_open_count,
                staged_open_count,
                max_open_positions,
                capacity_open_count=capacity_open_count,
                position_list_open_count=position_list_open_count,
            )
            return (
                "策略执行容量已达到本轮动态上限，暂停新开不同币种/方向仓位。"
                f"{capacity_detail}"
                f"{self._capacity_suffix(capacity)}"
            )
        return None

    def _capacity_context(self) -> dict[str, Any]:
        raw = self.max_open_positions_per_model_provider()
        if isinstance(raw, dict):
            return dict(raw)
        as_dict = getattr(raw, "as_dict", None)
        if callable(as_dict):
            value = as_dict()
            if isinstance(value, dict):
                return value
        effective = int(raw or 0)
        return {
            "entry_limit": effective,
            "effective_limit": effective,
            "base_limit": effective,
            "reason": "",
        }

    @classmethod
    def _capacity_limit(cls, capacity: dict[str, Any]) -> int:
        return cls._safe_non_negative_int(
            capacity.get("entry_limit") or capacity.get("effective_limit") or 0
        )

    @classmethod
    def _capacity_open_group_count(cls, capacity: dict[str, Any]) -> int:
        return cls._safe_non_negative_int(capacity.get("open_group_count") or 0)

    @staticmethod
    def _safe_non_negative_int(value: Any) -> int:
        try:
            return max(int(float(value or 0)), 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _capacity_suffix(capacity: dict[str, Any]) -> str:
        base_limit = capacity.get("base_limit")
        effective_limit = capacity.get("effective_limit")
        entry_limit = capacity.get("entry_limit")
        reason = str(capacity.get("reason") or "").strip()
        parts: list[str] = []
        if base_limit and effective_limit and int(base_limit) != int(effective_limit):
            parts.append(f"基础上限 {base_limit}，运行上限 {effective_limit}")
        if entry_limit and int(entry_limit) != int(effective_limit or 0):
            parts.append(f"开仓上限 {entry_limit}")
        readable_reason = EntryCapacityPolicy._capacity_reason_text(capacity, reason)
        if readable_reason:
            parts.append(readable_reason[:160])
        return " " + "；".join(parts) if parts else ""

    @staticmethod
    def _capacity_reason_text(capacity: dict[str, Any], reason: str) -> str:
        factors = capacity.get("factors") if isinstance(capacity.get("factors"), dict) else {}
        codes = factors.get("reason_codes") if isinstance(factors, dict) else None
        if isinstance(codes, list) and codes:
            labels = {
                "strategy_rotation_slots": "策略学习已为轮换释放预留开仓槽",
                "release_rotation_slots": "低质量持仓释放中，系统预留了小仓轮换槽",
                "rotation_entry_expansion": "开仓上限已按轮换释放策略上调",
                "low_quality_pressure": "低质量持仓压力较高，优先复盘释放旧仓",
                "low_quality_warn": "低质量持仓偏高，降低扩仓节奏",
                "drawdown": "当日回撤达到收缩区间",
                "drawdown_watch": "当日回撤进入观察区间",
            }
            return "；".join(labels.get(str(code), str(code)) for code in codes[:4])
        if "=" in reason:
            return "容量由策略学习、持仓质量和账户风险动态计算。"
        return reason

    @staticmethod
    def _capacity_count_text(
        live_open_count: int,
        staged_open_count: int,
        max_open_positions: int,
        *,
        capacity_open_count: int = 0,
        position_list_open_count: int | None = None,
    ) -> str:
        if position_list_open_count is not None and capacity_open_count > position_list_open_count:
            prefix = (
                f"容量快照 {capacity_open_count} 组，本轮持仓列表 "
                f"{position_list_open_count} 组，按较大值 {live_open_count} 组计算。"
            )
            if staged_open_count > 0:
                return (
                    f"{prefix}本轮待确认开仓 {staged_open_count} 组，"
                    f"合计占用 {live_open_count + staged_open_count} 组，"
                    f"执行容量 {max_open_positions} 组。"
                )
            return f"{prefix}执行容量 {max_open_positions} 组。"
        if staged_open_count > 0:
            return (
                f"真实持仓 {live_open_count} 组，本轮待确认开仓 {staged_open_count} 组，"
                f"合计占用 {live_open_count + staged_open_count} 组，执行容量 {max_open_positions} 组。"
            )
        return f"当前真实持仓 {live_open_count} 组，执行容量 {max_open_positions} 组。"

    @staticmethod
    def _is_effective_open_position(position: dict[str, Any]) -> bool:
        if position.get("is_open", True) is False:
            return False
        if "quantity" not in position:
            return True
        try:
            return float(position.get("quantity") or 0.0) > 1e-12
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _stage_dict(staged_entry_counts: StagedEntryCounts, key: str) -> dict[Any, int]:
        value = staged_entry_counts.get(key)
        if isinstance(value, dict):
            return value
        value = {}
        staged_entry_counts[key] = value
        return value

    def _model_open_group_count(self, model_name: str, open_positions: list[dict]) -> int:
        groups: set[tuple[str | None, str]] = set()
        for position in open_positions:
            if not self._is_effective_open_position(position):
                continue
            if position.get("model_name") != model_name:
                continue
            symbol_key = self.normalize_symbol(position.get("symbol"))
            if not symbol_key:
                continue
            side = str(position.get("side") or "unknown").lower().strip() or "unknown"
            groups.add((symbol_key, side))
        return len(groups)

    def reserve_slot(
        self,
        model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: StagedEntryCounts,
    ) -> None:
        """Reserve capacity for an entry selected earlier in the current round."""

        if not decision.is_entry:
            return

        staged_model_totals = self._stage_dict(staged_entry_counts, "model_totals")
        staged_symbol_side = self._stage_dict(staged_entry_counts, "symbol_side")
        staged_side_totals = self._stage_dict(staged_entry_counts, "side_totals")

        side = "long" if decision.action == Action.LONG else "short"
        staged_side_totals[side] = int(staged_side_totals.get(side, 0)) + 1
        staged_key = (model_name, self.normalize_symbol(decision.symbol), side)
        if staged_key not in staged_symbol_side:
            staged_model_totals[model_name] = int(staged_model_totals.get(model_name, 0)) + 1
        staged_symbol_side[staged_key] = int(staged_symbol_side.get(staged_key, 0)) + 1

    def release_slot(
        self,
        model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: StagedEntryCounts,
    ) -> None:
        """Release a staged entry when it did not become a real or pending order."""

        if not decision.is_entry:
            return

        staged_model_totals = self._stage_dict(staged_entry_counts, "model_totals")
        staged_symbol_side = self._stage_dict(staged_entry_counts, "symbol_side")
        staged_side_totals = self._stage_dict(staged_entry_counts, "side_totals")
        side = "long" if decision.action == Action.LONG else "short"
        staged_key = (model_name, self.normalize_symbol(decision.symbol), side)
        symbol_side_count = int(staged_symbol_side.get(staged_key, 0))
        if symbol_side_count <= 0:
            return

        if symbol_side_count <= 1:
            staged_symbol_side.pop(staged_key, None)
            model_total = int(staged_model_totals.get(model_name, 0)) - 1
            if model_total > 0:
                staged_model_totals[model_name] = model_total
            else:
                staged_model_totals.pop(model_name, None)
        else:
            staged_symbol_side[staged_key] = symbol_side_count - 1

        side_total = int(staged_side_totals.get(side, 0)) - 1
        if side_total > 0:
            staged_side_totals[side] = side_total
        else:
            staged_side_totals.pop(side, None)
