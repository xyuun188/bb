from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.normal_paper_trade import PAPER_TRAINING_REVIEW_GROUP_LIMIT

NormalizeSymbol = Callable[[Any], str | None]
StagedEntryCounts = dict[str, dict[Any, int]]


@dataclass(frozen=True, slots=True)
class EntryCapacityPolicy:
    """Prevent duplicate symbol exposure while tracking current-round reservations."""

    normalize_symbol: NormalizeSymbol

    def empty_staged_counts(self) -> StagedEntryCounts:
        """Return the per-round staged-entry counters used before orders are submitted."""

        return {"model_totals": {}, "symbol_side": {}, "side_totals": {}}

    def paper_training_review_groups(
        self, open_positions: list[dict[str, Any]],
    ) -> set[tuple[str, str]]:
        return {
            (str(position.get("model_name") or "ensemble_trader"), symbol)
            for position in open_positions
            if isinstance(position, dict)
            and self._is_effective_open_position(position)
            and (symbol := self.normalize_symbol(position.get("symbol")))
        }

    def paper_training_review_capacity(
        self, decision: DecisionOutput, open_positions: list[dict[str, Any]],
        *, pending_orders: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        raw = decision.raw_response
        trade = raw.get("normal_paper_trade") if isinstance(raw, dict) else None
        if not decision.is_entry or not isinstance(trade, dict):
            return None
        if trade.get("selection_reason") != "paper_training_entry":
            return None
        open_groups = self.paper_training_review_groups(open_positions)
        open_symbols = {symbol for _, symbol in open_groups}
        pending_symbols = {
            symbol for order in (pending_orders or [])
            if (symbol := self._pending_entry_symbol(order))
        }
        pending_groups = pending_symbols - open_symbols
        count = len(open_groups) + len(pending_groups)
        return {
            "open_review_groups": len(open_groups),
            "pending_entry_review_groups": len(pending_groups),
            "same_symbol_pending_entry": self.normalize_symbol(decision.symbol) in pending_symbols,
            "training_review_group_limit": PAPER_TRAINING_REVIEW_GROUP_LIMIT,
            "passed": count < PAPER_TRAINING_REVIEW_GROUP_LIMIT,
        }

    def _pending_entry_symbol(self, order: dict[str, Any]) -> str | None:
        info = order.get("info")
        info = info if isinstance(info, dict) else {}
        reduce_only = order.get("reduceOnly")
        if reduce_only in (None, ""):
            reduce_only = info.get("reduceOnly")
        side = str(order.get("side") or info.get("side") or "").lower()
        position_side = str(info.get("posSide") or "").lower()
        if (
            str(reduce_only or "").lower() in {"true", "1"}
            or (side, position_side) in {("sell", "long"), ("buy", "short")}
        ):
            return None
        return self.normalize_symbol(order.get("symbol") or info.get("instId"))

    def reason(
        self,
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict],
        staged_entry_counts: StagedEntryCounts,
    ) -> str | None:
        if not decision.is_entry:
            return None
        symbol = self.normalize_symbol(decision.symbol)
        if not symbol:
            return "交易对无法规范化，系统未创建新的模拟盘仓位。"
        for position in open_positions or []:
            if not isinstance(position, dict) or not self._is_effective_open_position(position):
                continue
            if self.normalize_symbol(position.get("symbol")) == symbol:
                return (
                    f"{symbol} 已有未平仓仓位；新观点保留为分析记录，"
                    "由持仓管理链处理现有仓位，不重复或反向开仓。"
                )
        staged_symbol_side = self._stage_dict(staged_entry_counts, "symbol_side")
        if any(key[1] == symbol and count > 0 for key, count in staged_symbol_side.items()):
            return f"{symbol} 本轮已有待提交开仓，系统不重复提交。"
        del model_name
        return None

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
