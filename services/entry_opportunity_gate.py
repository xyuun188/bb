"""Entry opportunity gate policy.

The gate owns only severe execution blockers and entry-risk downgrades.  AI
still owns the trade/no-trade decision; most weak opportunity signals are
annotated as advisory warnings instead of hard vetoes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_crowded_side_cap import EntryCrowdedSideCapPolicy
from services.entry_direction_metrics import (
    selected_entry_metrics,
    write_selected_metrics_snapshot,
)
from services.entry_loss_cooldown import EntryLossCooldownPolicy
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE

EntryOpportunityGateEvaluator = Callable[[DecisionOutput], str | None]

ENTRY_DIRECTION_HARD_CONFLICT_GAP = 0.12
ENTRY_DIRECTION_MIN_SUPPORT_SCORE = 0.02
ENTRY_MIN_NET_PROFIT_QUALITY_RATIO = 1.50
QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT = 0.18
PORTFOLIO_ROSTER_FILL_MIN_EXPECTED_PCT = 0.08
PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY = 0.66
PORTFOLIO_ROSTER_FILL_MIN_NET_PCT = 0.20
PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO = 0.25


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class EntryOpportunityGatePolicy:
    """Evaluate severe blockers before an entry reaches execution."""

    evaluator: EntryOpportunityGateEvaluator | None = None
    suspicious_symbol_policy: Any | None = None
    symbol_loss_cooldown_policy: EntryLossCooldownPolicy | None = None
    crowded_side_cap_policy: EntryCrowdedSideCapPolicy | None = None
    post_crash_rebound_guard: Any | None = None

    def gate_reason(self, decision: DecisionOutput) -> str | None:
        """Return the severe-entry-block reason, if any."""

        suspicious_reason = self._suspicious_reason(decision)
        if suspicious_reason:
            return suspicious_reason
        if self.evaluator is not None:
            return self.evaluator(decision)
        return self._evaluate(decision)

    def _suspicious_reason(self, decision: DecisionOutput) -> str | None:
        if self.suspicious_symbol_policy is None:
            return None
        return self.suspicious_symbol_policy.reason(decision.symbol)

    def _strategy_learning_entry_pause_reason(self, raw: dict[str, Any]) -> str | None:
        strategy_mode = _safe_dict(raw.get("strategy_mode"))
        context = _safe_dict(raw.get("strategy_learning_context"))
        learning = _safe_dict(context.get("strategy_learning"))
        mode_learning = _safe_dict(strategy_mode.get("strategy_learning"))
        paused = bool(
            context.get("strategy_learning_entry_pause")
            or strategy_mode.get("strategy_learning_entry_pause")
            or learning.get("entry_pause")
            or mode_learning.get("entry_pause")
        )
        if not paused:
            return None
        reason = (
            context.get("strategy_learning_entry_pause_reason")
            or strategy_mode.get("strategy_learning_entry_pause_reason")
            or learning.get("entry_pause_reason")
            or mode_learning.get("entry_pause_reason")
            or "策略学习护栏提示：已转为小仓恢复探针，不作为新开仓硬拦截。"
        )
        opportunity = _safe_dict(raw.get("opportunity_score"))
        warnings = list(_safe_list(opportunity.get("execution_advisory_warnings")))
        warnings.append(
            {
                "reason": str(reason)[:300],
                "policy": "strategy_learning_recovery_advisory",
                "blocks_entry": False,
            }
        )
        opportunity["execution_advisory_warnings"] = warnings
        opportunity["strategy_learning_pause_is_hard_gate"] = False
        raw["opportunity_score"] = opportunity
        raw["strategy_learning_entry_pause_is_hard_gate"] = False
        return str(reason)[:300]

    def _evaluate(self, decision: DecisionOutput) -> str | None:
        raw = _safe_dict(decision.raw_response)
        entry_pause_reason = self._strategy_learning_entry_pause_reason(raw)
        if entry_pause_reason:
            decision.raw_response = raw
        opportunity = _safe_dict(raw.get("opportunity_score"))
        selected_metrics = selected_entry_metrics(decision)
        confidence = max(
            float(decision.confidence or 0.0),
            _safe_float(opportunity.get("confidence"), 0.0),
        )
        advisory_warnings = list(_safe_list(opportunity.get("execution_advisory_warnings")))

        def add_advisory(reason: str, *, size_cap: float | None = None) -> None:
            item: dict[str, Any] = {"reason": reason}
            if size_cap is not None:
                original_size = float(decision.position_size_pct or 0.0)
                if original_size > size_cap > 0:
                    decision.position_size_pct = size_cap
                    item["original_position_size_pct"] = round(original_size, 6)
                    item["adjusted_position_size_pct"] = round(size_cap, 6)
            advisory_warnings.append(item)
            opportunity["execution_advisory_mode"] = "ai_decision_primary"
            opportunity["execution_advisory_policy"] = (
                "AI decides whether to open; opportunity score and local quantitative models "
                "are execution-layer risk hints and sizing downgrades, not ordinary hard vetoes."
            )
            opportunity["execution_advisory_warnings"] = advisory_warnings
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw

        symbol_loss_cooldown_reason = self._symbol_loss_cooldown_reason(decision)
        if symbol_loss_cooldown_reason:
            return symbol_loss_cooldown_reason
        crowded_side_cap_reason = self._crowded_side_cap_reason(decision)
        if crowded_side_cap_reason:
            return crowded_side_cap_reason
        post_crash_rebound_reason = self._post_crash_rebound_reason(decision)
        if post_crash_rebound_reason:
            return post_crash_rebound_reason

        score = _safe_float(opportunity.get("score"), float("nan"))
        min_score = _safe_float(
            opportunity.get("min_score_required"),
            MIN_ENTRY_OPPORTUNITY_SCORE,
        )
        if opportunity.get("historical_block"):
            opportunity["historical_block_applied_as_warning"] = True
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        if not math.isfinite(score):
            return "机会评分缺失或无效，本次不执行开仓。"

        evidence_block_reason = self._evidence_hard_block_reason(
            decision,
            raw,
            opportunity,
        )
        if evidence_block_reason:
            return evidence_block_reason

        expected_net = _safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        profit_quality_ratio = _safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        if selected_metrics.has_selected_side:
            expected_net = selected_metrics.expected_net_return_pct
            profit_quality_ratio = selected_metrics.profit_quality_ratio
            write_selected_metrics_snapshot(
                raw,
                selected_metrics,
                blocked=False,
                policy="selected_side_expected_net_quality_gate",
            )
            opportunity = _safe_dict(raw.get("opportunity_score"))
        min_profit_quality_ratio = _safe_float(
            opportunity.get("min_profit_quality_ratio_required"),
            ENTRY_MIN_NET_PROFIT_QUALITY_RATIO,
        )
        tail_risk_score = _safe_float(opportunity.get("tail_risk_score"), 0.0)
        success_probability = _safe_float(opportunity.get("success_probability"), 0.0)
        strong_support = bool(
            opportunity.get("ml_aligned") and opportunity.get("local_profit_aligned")
        )
        server_profit_conflict = bool(opportunity.get("server_profit_conflict"))
        same_side_loss_concentration = bool(opportunity.get("same_side_loss_concentration"))
        server_profit_expected = _safe_float(
            opportunity.get("server_profit_expected_return_pct"),
            0.0,
        )
        server_profit_side = str(opportunity.get("server_profit_best_side") or "")
        quant_probe = _safe_dict(raw.get("quant_profit_probe"))
        roster = _safe_dict(opportunity.get("portfolio_roster"))
        roster_underfilled = bool(roster.get("underfilled"))
        entry_side = self._entry_side(decision, opportunity)
        opposite_entry_side = self._opposite_side(entry_side)
        direction_competition = _safe_dict(opportunity.get("direction_competition"))
        direction_preferred_side = str(
            opportunity.get("direction_preferred_side")
            or direction_competition.get("preferred_side")
            or ""
        ).lower()
        direction_gap = _safe_float(direction_competition.get("score_gap"), 0.0)
        direction_side_score = _safe_float(
            _safe_dict(direction_competition.get(entry_side)).get(
                "score", opportunity.get("direction_side_score")
            ),
            0.0,
        )
        direction_opposite_score = _safe_float(
            _safe_dict(direction_competition.get(opposite_entry_side)).get(
                "score", opportunity.get("direction_opposite_score")
            ),
            0.0,
        )
        quant_profit_probe_entry = bool(
            quant_probe.get("triggered")
            and expected_net > 0
            and server_profit_expected
            >= (
                PORTFOLIO_ROSTER_FILL_MIN_EXPECTED_PCT
                if roster_underfilled
                else QUANT_PROFIT_PROBE_MIN_EXPECTED_PCT
            )
            and _safe_float(opportunity.get("server_profit_loss_probability"), 1.0)
            < (PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY if roster_underfilled else 0.58)
            and tail_risk_score < 0.95
            and not same_side_loss_concentration
        )
        roster_fill_entry = bool(
            roster_underfilled
            and expected_net >= PORTFOLIO_ROSTER_FILL_MIN_NET_PCT
            and profit_quality_ratio >= PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO
            and success_probability >= 0.48
            and tail_risk_score < 0.90
            and not same_side_loss_concentration
            and (
                opportunity.get("local_profit_aligned")
                or opportunity.get("ml_aligned")
                or opportunity.get("timeseries_aligned")
                or quant_probe.get("triggered")
            )
            and score >= max(min_score - 0.35, 0.25)
        )
        direction_supported_by_symbol = bool(
            opportunity.get("local_profit_aligned")
            or opportunity.get("ml_aligned")
            or opportunity.get("timeseries_aligned")
            or opportunity.get("expert_aligned")
            or quant_probe.get("triggered")
            or direction_side_score >= ENTRY_DIRECTION_MIN_SUPPORT_SCORE
        )

        if (
            entry_side in {"long", "short"}
            and direction_preferred_side == opposite_entry_side
            and direction_gap >= ENTRY_DIRECTION_HARD_CONFLICT_GAP
            and not (
                server_profit_side == entry_side
                and server_profit_expected >= 0.45
                and profit_quality_ratio >= 0.55
            )
        ):
            if (
                expected_net > 0
                and score >= max(min_score - 0.60, 0.95)
                and confidence >= 0.72
                and success_probability >= 0.48
                and tail_risk_score < 0.85
            ):
                opportunity["direction_conflict_softened"] = {
                    "applied": True,
                    "preferred_side": direction_preferred_side,
                    "entry_side": entry_side,
                    "direction_gap": round(direction_gap, 6),
                    "reason": (
                        "AI confidence and expected net return are high enough; direction "
                        "competition becomes a sizing warning instead of a hard veto."
                    ),
                }
                raw["opportunity_score"] = opportunity
                decision.raw_response = raw
            else:
                side_label = "做多" if entry_side == "long" else "做空"
                opposite_label = "做多" if opposite_entry_side == "long" else "做空"
                add_advisory(
                    f"开仓方向预判不支持本次{side_label}：该币种 long/short 竞争更偏向"
                    f"{opposite_label}，方向分差 {direction_gap:.2f}，本方向分 "
                    f"{direction_side_score:.2f}，相反方向分 {direction_opposite_score:.2f}。"
                    "本次作为风险提示并降低仓位，不硬拦 AI 开仓。",
                    size_cap=0.025,
                )
        if (
            entry_side in {"long", "short"}
            and not direction_supported_by_symbol
            and direction_side_score < ENTRY_DIRECTION_MIN_SUPPORT_SCORE
        ):
            side_label = "做多" if entry_side == "long" else "做空"
            add_advisory(
                f"该币种本轮缺少提前支持{side_label}的方向证据：方向分 "
                f"{direction_side_score:.2f}，未获得 ML、服务器盈利模型、时序模型或专家"
                "同向确认。本次作为仓位降级提示，不硬拦 AI 开仓。",
                size_cap=0.02,
            )

        server_profit_conflict_relief = bool(
            server_profit_conflict
            and expected_net > 0
            and score >= max(min_score - 0.55, 0.95)
            and confidence >= 0.72
            and profit_quality_ratio >= 0.20
            and success_probability >= 0.48
            and tail_risk_score < 0.90
            and (
                opportunity.get("ml_aligned")
                or opportunity.get("expert_aligned")
                or direction_preferred_side == entry_side
                or direction_side_score > 0
            )
        )
        if server_profit_conflict_relief:
            original_size = float(decision.position_size_pct or 0.0)
            if original_size > 0:
                decision.position_size_pct = min(original_size, 0.025)
            opportunity["server_profit_conflict_relief"] = {
                "applied": True,
                "original_position_size_pct": round(original_size, 6),
                "adjusted_position_size_pct": round(float(decision.position_size_pct or 0.0), 6),
                "server_profit_expected_return_pct": round(server_profit_expected, 6),
                "expected_net_return_pct": round(expected_net, 6),
                "reason": (
                    "Server profit model alone conflicts, but expected net return and AI/"
                    "direction evidence meet tiny-probe conditions."
                ),
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        if (
            server_profit_conflict
            and not server_profit_conflict_relief
            and not (
                server_profit_expected > 0
                and expected_net >= 1.20
                and profit_quality_ratio >= max(min_profit_quality_ratio, 1.35)
                and success_probability >= 0.58
                and (opportunity.get("ml_aligned") or opportunity.get("timeseries_aligned"))
            )
        ):
            side_label = self._side_label(str(opportunity.get("side") or ""))
            add_advisory(
                f"服务器盈利模型不支持本次{side_label}：该方向预期收益 "
                f"{server_profit_expected:.4f}%，模型更偏向 "
                f"{server_profit_side or '观望/未知'}。本次作为极小仓风险提示，不硬拦 AI 开仓。",
                size_cap=0.02,
            )

        contribution_adjustment = _safe_dict(opportunity.get("model_contribution_adjustment"))
        contribution_hard_caution = bool(contribution_adjustment.get("hard_caution"))
        negative_sources = contribution_adjustment.get("negative_sources")
        negative_source_count = len(negative_sources) if isinstance(negative_sources, list) else 0
        contribution_positive_expected_relief = bool(contribution_hard_caution and expected_net > 0)
        if contribution_positive_expected_relief:
            opportunity["contribution_positive_expected_relief"] = {
                "applied": True,
                "negative_source_count": negative_source_count,
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(profit_quality_ratio, 6),
                "reason": (
                    "Closed-loop contribution stats are treated as risk hints; positive expected "
                    "net return is not hard-blocked solely by one weak source."
                ),
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        if (
            contribution_hard_caution
            and not contribution_positive_expected_relief
            and not (
                expected_net >= 0.80
                and profit_quality_ratio >= max(min_profit_quality_ratio, 1.25)
                and success_probability >= 0.56
                and (
                    opportunity.get("local_profit_aligned") or opportunity.get("timeseries_aligned")
                )
            )
            and not (
                roster_fill_entry
                and negative_source_count <= 1
                and expected_net >= 0.70
                and profit_quality_ratio >= 0.55
            )
        ):
            add_advisory(
                f"闭环贡献统计显示本轮信号组合里有 {negative_source_count} 个来源最近真实净亏，"
                f"当前预期净收益 {expected_net:.2f}%、净盈亏比 {profit_quality_ratio:.2f} "
                "还不足以覆盖该风险。本次只降低仓位，不硬拦 AI 开仓。",
                size_cap=0.02,
            )
        if expected_net <= 0:
            write_selected_metrics_snapshot(
                raw,
                selected_metrics,
                blocked=True,
                policy="selected_side_expected_net_hard_gate",
            )
            decision.raw_response = raw
            side_label = self._side_label(selected_metrics.side or entry_side)
            return (
                f"实际下单方向{side_label}费后预期净收益 {expected_net:.4f}% 不为正，"
                "系统禁止提交开仓订单；下一轮会用最新行情和模型证据重新评估。"
            )
        if (
            profit_quality_ratio < min_profit_quality_ratio
            and not quant_profit_probe_entry
            and not roster_fill_entry
            and not contribution_positive_expected_relief
        ):
            add_advisory(
                f"净盈亏比 {profit_quality_ratio:.2f} 低于最低要求 {min_profit_quality_ratio:.2f}，"
                "这类机会容易形成小盈大亏，本次只降低仓位，不硬拦 AI 开仓。",
                size_cap=0.02,
            )
        if opportunity.get("weak_history_requires_stronger_edge") and not (
            opportunity.get("local_profit_aligned") or opportunity.get("ml_aligned")
        ):
            add_advisory(
                "该币种/方向近期真实盈亏偏弱，且本轮没有本地盈利模型或 ML 同向改善证据；"
                "本次按 AI 决策进入执行检查，但降低仓位。",
                size_cap=0.02,
            )
        if tail_risk_score >= 1.15 and not (
            strong_support and profit_quality_ratio >= 0.55 and success_probability >= 0.58
        ):
            return (
                f"尾部亏损风险 {tail_risk_score:.2f} 过高；这类机会可能胜率看起来不低，"
                "但单次亏损偏大，属于严重风险，本次不执行开仓。"
            )
        if tail_risk_score >= 0.95:
            add_advisory(
                f"尾部亏损风险 {tail_risk_score:.2f} 偏高，本次只降低仓位，不硬拦 AI 开仓。",
                size_cap=0.02,
            )
        if (
            profit_quality_ratio <= 0.12
            and not strong_support
            and not quant_profit_probe_entry
            and not roster_fill_entry
            and not contribution_positive_expected_relief
        ):
            add_advisory(
                f"盈利质量比 {profit_quality_ratio:.2f} 过低，相对可能亏损，预期收益不够有吸引力。"
                "本次只降低仓位，不硬拦 AI 开仓。",
                size_cap=0.015,
            )
        if (
            score <= min_score
            and not quant_profit_probe_entry
            and not roster_fill_entry
            and not contribution_positive_expected_relief
        ):
            add_advisory(
                f"机会评分 {score:.4f} 低于当前执行门槛 {min_score:.2f}；"
                "本次作为风险提示和仓位降级，不硬拦 AI 开仓。",
                size_cap=0.02,
            )
        if roster_fill_entry:
            opportunity["portfolio_roster_fill_relief"] = {
                "applied": True,
                "target_position_groups": roster.get("target_position_groups"),
                "current_position_groups": roster.get("current_position_groups"),
                "gap": roster.get("gap"),
                "expected_net_return_pct": round(expected_net, 6),
                "profit_quality_ratio": round(profit_quality_ratio, 6),
                "negative_source_count": negative_source_count,
                "reason": (
                    "Portfolio is underfilled and this signal remains positive expectancy; allow "
                    "small position to improve diversification."
                ),
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        if quant_profit_probe_entry:
            opportunity["quant_probe_execution_relief"] = {
                "applied": True,
                "score": round(score, 6),
                "min_score_required": round(min_score, 6),
                "profit_quality_ratio": round(profit_quality_ratio, 6),
                "min_profit_quality_ratio": round(min_profit_quality_ratio, 6),
                "reason": (
                    "Positive-expectancy tiny probe may enter execution checks even with low "
                    "score/profit-quality ratio, so real fills can train later filters."
                ),
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
        return None

    def _symbol_loss_cooldown_reason(self, decision: DecisionOutput) -> str | None:
        if self.symbol_loss_cooldown_policy is None:
            return None
        return self.symbol_loss_cooldown_policy.reason(decision)

    def _crowded_side_cap_reason(self, decision: DecisionOutput) -> str | None:
        if self.crowded_side_cap_policy is None:
            return None
        return self.crowded_side_cap_policy.block_reason(decision)

    def _post_crash_rebound_reason(self, decision: DecisionOutput) -> str | None:
        if self.post_crash_rebound_guard is not None:
            return self.post_crash_rebound_guard.guard_reason(decision)
        return None

    def _evidence_hard_block_reason(
        self,
        decision: DecisionOutput,
        raw: dict[str, Any],
        opportunity: dict[str, Any],
    ) -> str | None:
        evidence_score = _safe_dict(opportunity.get("evidence_score"))
        if not evidence_score.get("hard_block"):
            return None
        reasons = evidence_score.get("hard_block_reasons")
        reason_text = (
            "；".join(str(item) for item in reasons if item) if isinstance(reasons, list) else ""
        )
        effective_score = _safe_float(evidence_score.get("effective_score"), 0.0)
        raw["entry_evidence_hard_block"] = {
            "blocked": True,
            "effective_score": round(effective_score, 6),
            "tier": evidence_score.get("tier"),
            "reasons": reasons if isinstance(reasons, list) else [],
            "policy": (
                "Dynamic evidence scoring only hard-blocks severe directional conflicts; "
                "missing model data and weak evidence are handled by skipped observation or tiny probe sizing."
            ),
        }
        decision.raw_response = raw
        return (
            f"动态证据强冲突硬拦截：{reason_text or 'ML/时序/记忆出现严重反向'}。"
            f"当前有效分 {effective_score:.1f}；本次不提交开仓，等待下一轮重新评估。"
        )

    def _entry_side(self, decision: DecisionOutput, opportunity: dict[str, Any]) -> str:
        if decision.action == Action.LONG:
            return "long"
        if decision.action == Action.SHORT:
            return "short"
        return str(opportunity.get("side") or "")

    @staticmethod
    def _opposite_side(side: str) -> str:
        if side == "long":
            return "short"
        if side == "short":
            return "long"
        return ""

    @staticmethod
    def _side_label(side: str) -> str:
        if side == "long":
            return "做多"
        if side == "short":
            return "做空"
        return side or "当前方向"
