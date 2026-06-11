"""Recover useful execution reasons from persisted decision rows."""

from __future__ import annotations

from typing import Any


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class DecisionReasonRecoveryPolicy:
    """Build fallback reasons when execution output was missing or unusable."""

    def recover(self, decision: Any | None, fallback: Any = None) -> str | None:
        if decision is None:
            return None

        raw = _safe_dict(getattr(decision, "raw_llm_response", None))
        action = str(getattr(decision, "action", "") or "")
        if action in {"close_long", "close_short"}:
            return self._recover_exit_reason(decision, raw)
        if fallback:
            return str(fallback)
        return None

    def _recover_exit_reason(self, decision: Any, raw: dict[str, Any]) -> str:
        close_evidence = _safe_dict(raw.get("close_evidence"))
        action_plan = str(close_evidence.get("action_plan") or "").lower()
        plan_label = (
            "全平" if action_plan == "full_close" else "减仓" if action_plan == "reduce" else "平仓"
        )
        close_reason = str(
            close_evidence.get("reason") or getattr(decision, "reasoning", "") or ""
        ).strip()
        pnl = _safe_float(close_evidence.get("position_unrealized_pnl"), 0.0)
        if close_reason:
            return (
                f"平仓裁决已生成但本轮没有确认到 OKX 平仓订单结果：AI 建议{plan_label}，"
                f"当时估算浮动盈亏 {pnl:.4f} USDT。裁决依据：{close_reason}"
                "系统会继续以 OKX 实际仓位和执行记录为准同步；如果仓位仍存在，下一轮持仓复盘会重新评估并提交平仓。"
            )
        return (
            "平仓裁决已生成但本轮没有确认到 OKX 平仓订单结果。"
            "系统会继续以 OKX 实际仓位和执行记录为准同步；如果仓位仍存在，下一轮持仓复盘会重新评估并提交平仓。"
        )
