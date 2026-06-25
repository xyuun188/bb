"""Recover useful execution reasons from persisted decision rows."""

from __future__ import annotations

from typing import Any

from services.execution_result_classifier import ExecutionResultClassifier
from web_dashboard.api.text_sanitize import sanitize_text


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class DecisionReasonRecoveryPolicy:
    """Build fallback reasons when execution output was missing or unusable."""

    def __init__(self, classifier: ExecutionResultClassifier | None = None) -> None:
        self._classifier = classifier or ExecutionResultClassifier()

    def recover(self, decision: Any | None, fallback: Any = None) -> str | None:
        if decision is None:
            return None

        raw = _safe_dict(getattr(decision, "raw_llm_response", None))
        if not raw:
            raw = _safe_dict(getattr(decision, "raw_response", None))
        action = str(getattr(decision, "action", "") or "")
        if action in {"close_long", "close_short"}:
            return self._recover_exit_reason(decision, raw)
        if fallback:
            return str(fallback)
        return None

    def _recover_exit_reason(self, decision: Any, raw: dict[str, Any]) -> str:
        exchange_failure = self._recover_exchange_failure_reason(raw)
        if exchange_failure:
            return exchange_failure

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

    def _recover_exchange_failure_reason(self, raw: dict[str, Any]) -> str | None:
        fragments: list[str] = []
        self._append_error_fragments(fragments, raw.get("untradable_exit_execution_error"))

        execution_result = _safe_dict(raw.get("execution_result"))
        self._append_error_fragments(fragments, execution_result)
        self._append_error_fragments(fragments, execution_result.get("raw_response"))

        if not fragments:
            return None
        text = " ".join(fragment for fragment in fragments if fragment).strip()
        translated = self._classifier.translate_execution_error_text(text)
        if translated:
            return translated
        cleaned = sanitize_text(text[:500])
        return f"平仓执行失败：{cleaned or text[:500]}"

    def _append_error_fragments(self, fragments: list[str], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                fragments.append(text)
            return
        if isinstance(value, dict):
            for key in (
                "reason",
                "error",
                "raw_error",
                "message",
                "msg",
                "sMsg",
                "code",
                "sCode",
                "order_id",
                "exchange_order_id",
                "status",
            ):
                item = value.get(key)
                if item is not None:
                    self._append_error_fragments(fragments, item)
            self._append_error_fragments(fragments, value.get("raw_response"))
            self._append_error_fragments(fragments, value.get("data"))
            return
        if isinstance(value, list):
            for item in value:
                self._append_error_fragments(fragments, item)
