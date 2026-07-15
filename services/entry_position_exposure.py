"""Entry-side long/short exposure context."""

from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class EntryPositionExposurePolicy:
    """Summarize current and staged exposure so entries do not stack one side."""

    def context(
        self,
        open_positions: list[dict[str, Any]] | None,
        staged_entry_counts: dict[str, dict[Any, int]] | None = None,
    ) -> dict[str, Any]:
        long_notional = 0.0
        short_notional = 0.0
        long_unrealized_pnl = 0.0
        short_unrealized_pnl = 0.0
        long_count = 0
        short_count = 0

        for position in open_positions or []:
            if position.get("is_open", True) is False:
                continue
            side = str(position.get("side") or "").lower()
            if side not in {"long", "short"}:
                continue
            quantity = abs(
                _safe_float(
                    position.get("quantity") or position.get("contracts") or position.get("sz"),
                    0.0,
                )
            )
            info = position.get("info") if isinstance(position.get("info"), dict) else {}
            direct_notional = abs(
                _safe_float(
                    position.get("notional")
                    or position.get("notional_usd")
                    or position.get("notionalUsd")
                    or info.get("notionalUsd")
                    or info.get("notional"),
                    0.0,
                )
            )
            contract_size = _safe_float(
                position.get("contract_size") or position.get("contractSize") or info.get("ctVal"),
                1.0,
            )
            price = _safe_float(
                position.get("current_price")
                or position.get("markPrice")
                or position.get("lastPrice")
                or position.get("entry_price")
                or position.get("entryPrice")
                or position.get("avgPx"),
                0.0,
            )
            notional = (
                direct_notional
                if direct_notional > 0
                else quantity * max(price, 0.0) * (contract_size if contract_size > 0 else 1.0)
            )
            unrealized_pnl = _safe_float(
                position.get("unrealized_pnl")
                or position.get("unrealizedPnl")
                or position.get("upl")
                or info.get("upl")
                or info.get("unrealizedPnl"),
                0.0,
            )
            if side == "long":
                long_count += 1
                long_notional += max(notional, 0.0)
                long_unrealized_pnl += unrealized_pnl
            else:
                short_count += 1
                short_notional += max(notional, 0.0)
                short_unrealized_pnl += unrealized_pnl

        side_totals = (staged_entry_counts or {}).get("side_totals") or {}
        staged_long_count = int(side_totals.get("long", 0) or 0)
        staged_short_count = int(side_totals.get("short", 0) or 0)

        total_long_count = long_count + staged_long_count
        total_short_count = short_count + staged_short_count
        gross_notional = long_notional + short_notional
        net_notional = long_notional - short_notional
        net_ratio = net_notional / gross_notional if gross_notional > 0 else 0.0
        total_count = total_long_count + total_short_count
        long_count_share = total_long_count / total_count if total_count > 0 else 0.0
        short_count_share = total_short_count / total_count if total_count > 0 else 0.0

        dominant_side = "neutral"
        if net_notional != 0:
            dominant_side = "long" if net_notional > 0 else "short"
        elif total_long_count != total_short_count:
            dominant_side = "long" if total_long_count > total_short_count else "short"

        return {
            "long_notional": round(long_notional, 4),
            "short_notional": round(short_notional, 4),
            "long_unrealized_pnl": round(long_unrealized_pnl, 4),
            "short_unrealized_pnl": round(short_unrealized_pnl, 4),
            "total_unrealized_pnl": round(long_unrealized_pnl + short_unrealized_pnl, 4),
            "gross_notional": round(gross_notional, 4),
            "net_notional": round(net_notional, 4),
            "net_ratio": round(net_ratio, 4),
            "long_count": total_long_count,
            "short_count": total_short_count,
            "staged_long_count": staged_long_count,
            "staged_short_count": staged_short_count,
            "long_count_share": round(long_count_share, 4),
            "short_count_share": round(short_count_share, 4),
            "dominant_side": dominant_side,
        }
