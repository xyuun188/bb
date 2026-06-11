"""Build hold decisions for position groups skipped by slow position review."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from services.analysis_budget import POSITION_REVIEW_FAST_EXIT_SCORE


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class FastScanHoldPlan:
    """Reasoning and raw payload for a fast-scan hold record."""

    reason: str
    raw_response: dict[str, Any]
    defer_count: int


@dataclass(frozen=True, slots=True)
class PositionReviewFastScanHoldPolicy:
    """Prepare explanatory HOLD records for position groups deferred from slow AI review."""

    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    fast_exit_score: float = POSITION_REVIEW_FAST_EXIT_SCORE

    def plan(
        self,
        scan: dict[str, Any],
        *,
        previous_defer_count: int,
        urgent_exit: bool,
        portfolio_symbol_context: dict[str, Any] | None,
        agent_skill_dicts: list[dict[str, Any]],
        agent_skill_summary: dict[str, Any],
    ) -> FastScanHoldPlan:
        priority_score = _safe_float(scan.get("priority_score"), 0.0)
        exit_score = _safe_float(scan.get("exit_score"), 0.0)
        add_score = _safe_float(scan.get("add_score"), 0.0)
        scan_reason = str(scan.get("reason") or "")
        should_count_defer = urgent_exit or exit_score >= self.fast_exit_score
        defer_count = int(previous_defer_count or 0) + 1 if should_count_defer else 0

        if exit_score >= self.fast_exit_score:
            reason = (
                "快速持仓扫描发现需要复盘的平仓/锁盈信号，"
                "但本轮慢专家名额已满，已记录并等待下一轮优先处理；"
                f"优先级 {priority_score:.1f}，退出分 {exit_score:.1f}。"
            )
            if urgent_exit:
                reason += " 该信号属于紧急退出类，下一轮会优先插队深度复盘。"
            if defer_count >= 2:
                reason += f" 已连续跳过 {defer_count} 轮，下一轮将强制插队。"
        else:
            reason = (
                "快速持仓扫描未发现必须立即交给慢专家的平仓/加仓信号，"
                f"优先级 {priority_score:.1f}。"
            )
        if scan_reason:
            reason += f" 触发项：{scan_reason}"

        portfolio_symbol_context = (
            portfolio_symbol_context if isinstance(portfolio_symbol_context, dict) else {}
        )
        if portfolio_symbol_context.get("active") and portfolio_symbol_context.get("is_focus"):
            reason += " 组合利润保护已激活；该高贡献仓位已纳入锁盈复盘关注。"

        raw_response: dict[str, Any] = {
            "analysis_type": "position_review",
            "position_fast_scan": {
                "skipped_llm": True,
                "priority_score": round(priority_score, 4),
                "exit_score": round(exit_score, 4),
                "add_score": round(add_score, 4),
                "reason": scan_reason,
            },
            "agent_skills": {
                "version": 1,
                "phases": {
                    "position_fast_scan": {
                        "phase": "position_fast_scan",
                        "recorded_at": self.clock().isoformat(),
                        "note": (
                            "快速扫描记录：退出分达到优先线表示发现平仓/锁盈复盘信号；"
                            "未达到优先线时仅作为普通轮转观察。"
                        ),
                        "skills": agent_skill_dicts,
                    },
                },
                "summary": agent_skill_summary,
            },
        }
        if portfolio_symbol_context.get("active"):
            raw_response["portfolio_profit_protection"] = portfolio_symbol_context

        return FastScanHoldPlan(
            reason=reason,
            raw_response=raw_response,
            defer_count=defer_count,
        )
