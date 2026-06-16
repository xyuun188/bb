"""Lightweight Agent/Skills layer for trading decisions.

This module does not replace the existing experts, ML models, risk engine, or
OKX executor. It standardizes their evidence and guard decisions so every
analysis record can explain which components supported, warned, or blocked a
trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.entry_signal_extraction import (
    expected_return_pct as signal_expected_return_pct,
)
from services.entry_signal_extraction import (
    first_tool_payload,
    has_signal_evidence,
    payload_side,
    signal_available,
)


@dataclass(slots=True)
class SkillResult:
    """Normalized result produced by one trading skill."""

    name: str
    label: str
    status: str
    decision: str = "neutral"
    reason: str = ""
    confidence: float | None = None
    blocks_entry: bool = False
    blocks_exit: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "status": self.status,
            "decision": self.decision,
            "reason": self.reason,
            "blocks_entry": self.blocks_entry,
            "blocks_exit": self.blocks_exit,
            "data": self.data,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


class TradingAgentSkillBook:
    """Builds standardized skill evidence for market, position, and execution."""

    def market_skills(
        self,
        *,
        new_pair_pause_reason: str | None,
        ml_signal: dict[str, Any] | None,
        local_ai_tools: dict[str, Any] | None,
        market_regime: dict[str, Any] | None,
        strategy_mode: dict[str, Any] | None,
    ) -> list[SkillResult]:
        skills = [
            self._new_pair_guard(new_pair_pause_reason),
            self._local_ml_skill(ml_signal),
            self._server_profit_skill(local_ai_tools),
            self._timeseries_skill(local_ai_tools),
            self._sentiment_skill(local_ai_tools),
            self._market_regime_skill(market_regime, strategy_mode),
        ]
        return [skill for skill in skills if skill is not None]

    def position_skills(
        self,
        *,
        position_entry_pause_reason: str | None,
        ml_signal: dict[str, Any] | None,
        local_ai_tools: dict[str, Any] | None,
        portfolio_profit_protection: dict[str, Any] | None,
    ) -> list[SkillResult]:
        skills = [
            self._position_mode_guard(position_entry_pause_reason),
            self._local_ml_skill(ml_signal),
            self._server_profit_skill(local_ai_tools),
            self._timeseries_skill(local_ai_tools),
            self._sentiment_skill(local_ai_tools),
            self._exit_advice_skill(local_ai_tools),
            self._portfolio_profit_skill(portfolio_profit_protection),
        ]
        return [skill for skill in skills if skill is not None]

    def execution_skills(
        self,
        *,
        decision: DecisionOutput,
        model_mode: str,
        override_balance: float | None,
        new_pair_pause_reason: str | None = None,
    ) -> list[SkillResult]:
        skills: list[SkillResult] = []
        if decision.is_entry and new_pair_pause_reason:
            skills.append(self._new_pair_guard(new_pair_pause_reason))
        if decision.is_entry:
            balance = self._to_float(override_balance, 0.0)
            if balance <= 0:
                skills.append(
                    SkillResult(
                        name="execution_margin_guard",
                        label="执行前保证金检查",
                        status="blocked",
                        decision="block_entry",
                        blocks_entry=True,
                        reason=(
                            "执行前没有可用于本次订单的 USDT 保证金，订单不会提交到 OKX。"
                            "通常是持仓或挂单占用过高、账户可用余额不足，或本轮仓位过大。"
                        ),
                        data={
                            "mode": model_mode,
                            "override_balance": balance,
                        },
                    )
                )
            else:
                skills.append(
                    SkillResult(
                        name="execution_margin_guard",
                        label="执行前保证金检查",
                        status="passed",
                        decision="allow",
                        reason="执行前仍有可分配保证金，本次允许进入 OKX 下单流程。",
                        data={
                            "mode": model_mode,
                            "override_balance": balance,
                        },
                    )
                )
        return skills

    def block_reason(self, skills: list[SkillResult], *, for_entry: bool = True) -> str | None:
        blockers = [
            skill for skill in skills if (skill.blocks_entry if for_entry else skill.blocks_exit)
        ]
        if not blockers:
            return None
        return (
            "；".join(skill.reason for skill in blockers if skill.reason)
            or "Agent/Skills 守门拒绝本次动作。"
        )

    def attach(
        self,
        decision: DecisionOutput,
        *,
        phase: str,
        skills: list[SkillResult],
        note: str | None = None,
    ) -> dict[str, Any]:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        existing_raw = raw.get("agent_skills")
        existing = existing_raw if isinstance(existing_raw, dict) else {}
        phases_raw = existing.get("phases")
        phases = phases_raw if isinstance(phases_raw, dict) else {}
        phases[phase] = {
            "phase": phase,
            "recorded_at": datetime.now(UTC).isoformat(),
            "note": note or "",
            "skills": [skill.to_dict() for skill in skills],
        }
        raw["agent_skills"] = {
            "version": 1,
            "phases": phases,
            "summary": self.summary(skills),
        }
        decision.raw_response = raw
        return raw

    def summary(self, skills: list[SkillResult]) -> dict[str, Any]:
        blocked = [skill for skill in skills if skill.blocks_entry or skill.blocks_exit]
        warnings = [skill for skill in skills if skill.status in {"warning", "partial"}]
        active = [skill for skill in skills if skill.status in {"active", "passed", "supported"}]
        return {
            "blocked": bool(blocked),
            "blockers": [skill.label for skill in blocked],
            "warning_count": len(warnings),
            "active_count": len(active),
        }

    def _new_pair_guard(self, reason: str | None) -> SkillResult:
        if reason:
            return SkillResult(
                name="new_pair_entry_guard",
                label="新交易对开仓守门",
                status="blocked",
                decision="block_entry",
                blocks_entry=True,
                reason=reason,
            )
        return SkillResult(
            name="new_pair_entry_guard",
            label="新交易对开仓守门",
            status="passed",
            decision="allow",
            reason="没有触发暂停新交易对的账户限制。",
        )

    def _position_mode_guard(self, reason: str | None) -> SkillResult:
        if reason:
            return SkillResult(
                name="position_mode_guard",
                label="持仓保护模式",
                status="active",
                decision="exit_reduce_hold_only",
                blocks_entry=True,
                reason=(
                    "当前账户触发新开仓限制，持仓分析只允许平仓、减仓或继续持有，"
                    f"不会执行加仓/新开仓。触发原因：{reason}"
                ),
            )
        return SkillResult(
            name="position_mode_guard",
            label="持仓保护模式",
            status="passed",
            decision="all_position_actions_available",
            reason="当前没有触发账户级新开仓限制，持仓分析可按 AI 结论处理。",
        )

    def _local_ml_skill(self, signal: dict[str, Any] | None) -> SkillResult:
        if not isinstance(signal, dict) or not signal:
            return SkillResult(
                name="local_ml_quality",
                label="本地 ML 盈亏质量",
                status="unavailable",
                decision="neutral",
                reason="本轮没有拿到本地 ML 盈亏质量预测。",
            )
        ready = bool(signal.get("ready") or signal.get("available"))
        side = str(
            signal.get("best_side") or signal.get("side") or signal.get("direction") or ""
        ).lower()
        expected = self._first_number(
            signal, "expected_return_pct", "expected_pct", "profit_edge_pct"
        )
        if side not in {"long", "short"} or expected is None:
            predictions = signal.get("predictions")
            if isinstance(predictions, list) and predictions:
                first = predictions[0] if isinstance(predictions[0], dict) else {}
                if side not in {"long", "short"}:
                    side = str(first.get("best_side") or first.get("side") or "").lower()
                if expected is None:
                    expected = self._first_number(
                        first, "best_expected_return_pct", "expected_return_pct", "profit_edge_pct"
                    )
        decision = side if side in {"long", "short"} else "neutral"
        if not ready:
            status = "learning"
            reason = "本地 ML 仍在学习或样本不足，只作为观察证据。"
        elif expected is not None and expected > 0:
            status = "supported"
            reason = f"本地 ML 预测 {decision or '中性'} 方向预期收益为正。"
        else:
            status = "warning"
            reason = "本地 ML 没有给出明确正收益优势。"
        return SkillResult(
            name="local_ml_quality",
            label="本地 ML 盈亏质量",
            status=status,
            decision=decision,
            confidence=self._first_number(signal, "confidence", "score"),
            reason=reason,
            data=self._compact(
                signal,
                [
                    "ready",
                    "available",
                    "best_side",
                    "side",
                    "expected_return_pct",
                    "profit_edge_pct",
                    "loss_probability",
                    "confidence",
                    "suggestion",
                ],
            ),
        )

    def _server_profit_skill(self, tools: dict[str, Any] | None) -> SkillResult:
        profit = self._tool_section(
            tools,
            "profit_prediction",
            "profit_model",
            "server_profit",
            "server_profit_model",
            "profit",
        )
        if not profit:
            return SkillResult(
                name="server_profit_model",
                label="服务器盈利预测",
                status="unavailable",
                decision="neutral",
                reason="本轮没有拿到服务器盈利模型预测。",
            )
        available = signal_available(profit)
        side = payload_side(profit)
        expected = self._first_number(
            profit, "expected_return_pct", "expected_net_return_pct", "profit_edge_pct"
        )
        if expected is None and has_signal_evidence(profit):
            expected = signal_expected_return_pct(profit, side)
        status = (
            "supported"
            if available and expected is not None and expected > 0
            else ("warning" if available else "unavailable")
        )
        reason = (
            f"服务器盈利模型给出 {side or '中性'}，预期收益 {expected:.4f}%."
            if expected is not None
            else "服务器盈利模型已返回，但没有明确预期收益。"
        )
        return SkillResult(
            name="server_profit_model",
            label="服务器盈利预测",
            status=status,
            decision=side if side in {"long", "short"} else "neutral",
            confidence=self._first_number(profit, "confidence", "score"),
            reason=reason,
            data=self._compact(
                profit,
                [
                    "available",
                    "status",
                    "error",
                    "path",
                    "duration_sec",
                    "latency_ms",
                    "model",
                    "backend",
                    "endpoint",
                    "best_side",
                    "side",
                    "expected_return_pct",
                    "expected_net_return_pct",
                    "profit_edge_pct",
                    "loss_probability",
                    "confidence",
                    "recommendation",
                ],
            ),
        )

    def _timeseries_skill(self, tools: dict[str, Any] | None) -> SkillResult:
        series = self._tool_section(
            tools,
            "time_series_prediction",
            "timeseries_prediction",
            "sequence_prediction",
            "timeseries",
            "time_series",
        )
        if not series:
            return SkillResult(
                name="time_series_model",
                label="时序预测",
                status="unavailable",
                decision="neutral",
                reason="本轮没有拿到时序预测。",
            )
        available = signal_available(series)
        side = payload_side(series)
        expected = self._first_number(series, "expected_return_pct", "expected_move_pct")
        if expected is None and has_signal_evidence(series):
            expected = signal_expected_return_pct(series, side)
        return SkillResult(
            name="time_series_model",
            label="时序预测",
            status=(
                "supported"
                if available and expected is not None and expected > 0
                else ("warning" if available else "unavailable")
            ),
            decision=side if side in {"long", "short"} else "neutral",
            confidence=self._first_number(series, "confidence", "score"),
            reason=(
                f"时序模型倾向 {side or '中性'}，预期变化 {expected:.4f}%."
                if expected is not None
                else "时序模型未给出明确收益方向。"
            ),
            data=self._compact(
                series,
                [
                    "available",
                    "status",
                    "error",
                    "path",
                    "duration_sec",
                    "latency_ms",
                    "model",
                    "backend",
                    "endpoint",
                    "best_side",
                    "side",
                    "direction",
                    "expected_return_pct",
                    "expected_move_pct",
                    "confidence",
                ],
            ),
        )

    def _sentiment_skill(self, tools: dict[str, Any] | None) -> SkillResult:
        sentiment = self._tool_section(
            tools,
            "sentiment_analysis",
            "sentiment_prediction",
            "sentiment_model",
            "sentiment",
        )
        if not sentiment:
            return SkillResult(
                name="sentiment_model",
                label="情绪预测",
                status="unavailable",
                decision="neutral",
                reason="本轮没有拿到情绪模型预测。",
            )
        available = signal_available(sentiment)
        side = payload_side(sentiment)
        score = self._first_number(sentiment, "score", "sentiment_score")
        return SkillResult(
            name="sentiment_model",
            label="情绪预测",
            status=(
                "supported"
                if available
                and (score is not None and abs(score) >= 0.05 or side in {"long", "short"})
                else ("warning" if available else "unavailable")
            ),
            decision=side if side in {"long", "short"} else "neutral",
            confidence=self._first_number(sentiment, "confidence"),
            reason=(
                f"情绪模型倾向 {side or '中性'}，情绪分 {score:.3f}."
                if score is not None
                else "情绪模型未给出明确分数。"
            ),
            data=self._compact(
                sentiment,
                [
                    "available",
                    "status",
                    "error",
                    "path",
                    "duration_sec",
                    "latency_ms",
                    "model",
                    "backend",
                    "endpoint",
                    "best_side",
                    "side",
                    "label",
                    "sentiment",
                    "score",
                    "sentiment_score",
                    "risk_level",
                    "confidence",
                ],
            ),
        )

    def _exit_advice_skill(self, tools: dict[str, Any] | None) -> SkillResult | None:
        advice = self._tool_section(
            tools,
            "exit_advice",
            "exit_model",
            "position_exit",
            "exit",
        )
        if not advice:
            return None
        action = str(advice.get("action") or advice.get("recommendation") or "hold").lower()
        reason = self._humanize_exit_reason(
            str(advice.get("reason") or advice.get("note") or ""),
            action,
        )
        display_advice = dict(advice)
        display_advice["reason"] = reason
        display_advice["action_label"] = self._humanize_exit_action(action)
        return SkillResult(
            name="exit_advice_model",
            label="平仓建议模型",
            status="supported" if action not in {"", "hold", "wait"} else "active",
            decision=display_advice["action_label"],
            confidence=self._first_number(advice, "confidence", "score"),
            reason=reason[:240],
            data=self._compact(
                display_advice,
                [
                    "available",
                    "status",
                    "error",
                    "path",
                    "duration_sec",
                    "latency_ms",
                    "model",
                    "backend",
                    "endpoint",
                    "action",
                    "action_label",
                    "recommendation",
                    "confidence",
                    "expected_net_pnl",
                    "expected_return_pct",
                    "reason",
                    "note",
                ],
            ),
        )

    def _humanize_exit_action(self, action: str) -> str:
        return {
            "hold": "继续持有",
            "wait": "继续观察",
            "observe": "继续观察",
            "reduce": "减仓",
            "partial_close": "部分平仓",
            "close": "平仓",
            "full_close": "全部平仓",
            "close_long": "平多",
            "close_short": "平空",
            "no_position": "无匹配持仓",
            "reduce_or_close": "减仓或平仓",
            "protect_profit": "保护利润",
            "close_if_ai_agrees": "AI确认后平仓",
            "trail_profit": "移动锁盈",
        }.get(str(action or "").lower(), str(action or "继续观察"))

    def _humanize_exit_reason(self, reason: str, action: str = "") -> str:
        text = str(reason or "").strip()
        normalized = text.lower().strip(" .")
        if not text or normalized in {
            "no trained exit pressure",
            "no exit pressure",
            "no trained close pressure",
        }:
            if str(action or "").lower() in {"hold", "wait", "observe", ""}:
                return "平仓建议模型未识别到明确的主动平仓压力，本轮倾向继续持有。"
            return "平仓建议模型已参与本轮持仓分析。"
        known = {
            "no matching open position was supplied": "本轮没有传入与该币种匹配的当前持仓，平仓建议模型不参与。",
            "this symbol/side has weak realized profile and the open position is losing": "该币种/方向历史实盘表现偏弱，且当前持仓正在亏损，建议减仓或平仓。",
            "profit exists but historical giveback/loss pressure is elevated": "当前已有浮盈，但历史回吐或亏损压力偏高，建议优先保护利润。",
            "loss is expanding beyond the local exit model tolerance": "亏损扩大到本地平仓模型容忍线之外，若 AI 也确认应优先退出。",
            "position is profitable; trail rather than cap upside immediately": "当前持仓盈利且历史盈亏质量尚可，建议移动保护利润，不急于完全限制上行空间。",
        }
        if normalized in known:
            return known[normalized]
        return text

    def _portfolio_profit_skill(self, context: dict[str, Any] | None) -> SkillResult | None:
        if not isinstance(context, dict) or not context:
            return None
        active = bool(context.get("active"))
        is_focus = bool(context.get("is_focus"))
        return SkillResult(
            name="portfolio_profit_protection",
            label="组合赢家管理",
            status="active" if active else "inactive",
            decision="focus_review" if is_focus else "observe",
            reason=(
                "账户浮盈达到赢家管理条件，该持仓属于重点复盘对象，需要判断继续拿、加仓、锁盈或全平。"
                if active and is_focus
                else "组合赢家管理未要求该持仓插队深度复盘。"
            ),
            data=self._compact(
                context,
                [
                    "active",
                    "is_focus",
                    "total_unrealized_pnl",
                    "symbol_unrealized_pnl",
                    "contribution_share",
                    "focus_rank",
                    "reason",
                ],
            ),
        )

    def _market_regime_skill(
        self,
        market_regime: dict[str, Any] | None,
        strategy_mode: dict[str, Any] | None,
    ) -> SkillResult:
        regime = market_regime if isinstance(market_regime, dict) else {}
        strategy = strategy_mode if isinstance(strategy_mode, dict) else {}
        blocked = strategy.get("blocked_directions") or []
        return SkillResult(
            name="market_regime_filter",
            label="整体行情方向过滤",
            status="active" if regime or strategy else "unavailable",
            decision=str(strategy.get("strategy") or regime.get("mode") or "neutral"),
            confidence=self._first_number(strategy, "confidence")
            or self._first_number(regime, "confidence"),
            reason=str(
                strategy.get("reason")
                or regime.get("reason")
                or "整体行情只做方向背景，不直接强制所有币同向开仓。"
            )[:260],
            data={
                "regime": self._compact(
                    regime, ["mode", "confidence", "avoid_long", "avoid_short", "reason"]
                ),
                "strategy": self._compact(
                    strategy,
                    [
                        "strategy",
                        "posture",
                        "allow_long",
                        "allow_short",
                        "blocked_directions",
                        "reason",
                    ],
                ),
                "blocked_directions": blocked,
            },
        )

    def _tool_section(self, tools: dict[str, Any] | None, *names: str) -> dict[str, Any]:
        if not isinstance(tools, dict):
            return {}
        return first_tool_payload(tools, *names)

    def _first_number(self, data: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            value = data.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _compact(self, data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        return {key: data.get(key) for key in keys if key in data}

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
