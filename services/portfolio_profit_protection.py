"""Portfolio-level floating-profit context for position review."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

NormalizeSymbol = Callable[[Any], str]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class PortfolioProfitProtectionPolicy:
    """Build portfolio profit-lock context without depending on TradingService."""

    normalize_symbol: NormalizeSymbol
    default_model_name: str
    def context(self, open_positions: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a portfolio-level floating-profit context for AI lock-profit review."""

        groups: dict[tuple[str, str], dict[str, Any]] = {}
        total_unrealized = 0.0
        total_positive = 0.0
        total_notional = 0.0

        for pos in open_positions or []:
            if pos.get("is_open", True) is False:
                continue
            model_name = str(pos.get("model_name") or self.default_model_name)
            symbol = self.normalize_symbol(pos.get("symbol"))
            side = str(pos.get("side") or "").lower()
            if not symbol or side not in {"long", "short"}:
                continue
            unrealized = _safe_float(pos.get("unrealized_pnl"), 0.0)
            entry_price = _safe_float(pos.get("entry_price"), 0.0)
            quantity = abs(_safe_float(pos.get("quantity"), 0.0))
            notional = abs(entry_price * quantity)
            total_unrealized += unrealized
            total_positive += max(unrealized, 0.0)
            total_notional += notional

            key = (model_name, symbol)
            item = groups.setdefault(
                key,
                {
                    "model_name": model_name,
                    "symbol": symbol,
                    "side": side,
                    "rows": 0,
                    "quantity": 0.0,
                    "notional": 0.0,
                    "unrealized_pnl": 0.0,
                    "first_opened_at": pos.get("created_at"),
                },
            )
            item["rows"] += 1
            item["quantity"] += quantity
            item["notional"] += notional
            item["unrealized_pnl"] += unrealized
            if str(item.get("side") or "") != side:
                item["side"] = "mixed"

        active = total_positive > 0.0
        ranked = sorted(
            groups.values(),
            key=lambda item: _safe_float(item.get("unrealized_pnl"), 0.0),
            reverse=True,
        )
        focus_groups: list[dict[str, Any]] = []
        if active:
            for item in ranked:
                unrealized = _safe_float(item.get("unrealized_pnl"), 0.0)
                share = unrealized / max(total_positive, 1e-9) if unrealized > 0 else 0.0
                if unrealized > 0.0:
                    focus_groups.append(
                        {
                            **item,
                            "profit_share": round(share, 6),
                            "profit_pct": round(
                                unrealized / max(_safe_float(item.get("notional"), 0.0), 1e-9),
                                6,
                            ),
                        }
                    )
        return {
            "active": active,
            "total_unrealized_pnl": round(total_unrealized, 6),
            "total_positive_unrealized_pnl": round(total_positive, 6),
            "total_open_notional": round(total_notional, 6),
            "focus_groups": focus_groups,
            "top_groups": [
                {
                    **item,
                    "profit_share": (
                        round(
                            _safe_float(item.get("unrealized_pnl"), 0.0)
                            / max(total_positive, 1e-9),
                            6,
                        )
                        if _safe_float(item.get("unrealized_pnl"), 0.0) > 0
                        else 0.0
                    ),
                    "profit_pct": round(
                        _safe_float(item.get("unrealized_pnl"), 0.0)
                        / max(_safe_float(item.get("notional"), 0.0), 1e-9),
                        6,
                    ),
                }
                for item in ranked
            ],
            "instruction": (
                "Portfolio floating profit has reached the winner-management line. "
                "High-contribution positions must be deep-reviewed for one of: "
                "continue holding, add to winner, partial profit lock, or full close."
                if active
                else ""
            ),
        }

    def symbol_context(
        self,
        context: dict[str, Any],
        model_name: str,
        symbol: str,
        positions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(context, dict) or not context.get("active"):
            return {"active": False}
        normalized = self.normalize_symbol(symbol)
        model = str(model_name or self.default_model_name)
        focus = [
            item
            for item in context.get("focus_groups", [])
            if item.get("model_name") == model
            and self.normalize_symbol(item.get("symbol")) == normalized
        ]
        top_match = [
            item
            for item in context.get("top_groups", [])
            if item.get("model_name") == model
            and self.normalize_symbol(item.get("symbol")) == normalized
        ]
        current = focus[0] if focus else (top_match[0] if top_match else {})
        if not current and positions:
            unrealized = sum(_safe_float(p.get("unrealized_pnl"), 0.0) for p in positions)
            notional = sum(
                abs(_safe_float(p.get("entry_price"), 0.0) * _safe_float(p.get("quantity"), 0.0))
                for p in positions
            )
            current = {
                "model_name": model,
                "symbol": normalized,
                "side": str((positions[0] or {}).get("side") or ""),
                "rows": len(positions),
                "unrealized_pnl": round(unrealized, 6),
                "notional": round(notional, 6),
                "profit_pct": round(unrealized / max(notional, 1e-9), 6),
            }
        return {
            "active": True,
            "is_focus": bool(focus),
            "total_unrealized_pnl": context.get("total_unrealized_pnl"),
            "total_positive_unrealized_pnl": context.get("total_positive_unrealized_pnl"),
            "current_group": current,
            "top_groups": context.get("top_groups", []),
            "required_choice": [
                "continue_hold_with_reason",
                "add_to_winner_if_trend_continues",
                "partial_lock_profit",
                "full_close",
            ],
            "instruction": context.get("instruction") or "",
        }
