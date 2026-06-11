"""Entry profit-risk sizing policy.

This module owns the entry-side size/leverage caps that depend on expected
profit, loss budget, evidence tier, ATR stress stops, and existing same-side
winners. TradingService wires the dependencies; this policy owns the decision
math.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from config.settings import settings
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from services.entry_sizing import apply_evidence_sizing_policy

EntryProfitRiskSizingEvaluator = Callable[
    [DecisionOutput, str, list[dict[str, Any]]],
    Awaitable[None],
]
EntryBalanceProvider = Callable[[str, DecisionOutput | None], Awaitable[float | None]]

ENTRY_MIN_NET_PROFIT_QUALITY_RATIO = 1.50
ENTRY_WEAK_HISTORY_MAX_SIZE = 0.025
ENTRY_WEAK_HISTORY_MAX_LEVERAGE = 5.0
ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_SIZE = 0.045
ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_LEVERAGE = 8.0
ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE = 0.02
ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_LEVERAGE = 4.0
ENTRY_LOW_QUALITY_MAX_SIZE = 0.018
ENTRY_LOW_QUALITY_MAX_LEVERAGE = 3.0
ENTRY_SYMBOL_LOSER_SIZE_MULTIPLIER = 0.55
ENTRY_HIGH_QUALITY_MIN_NOTIONAL_BALANCE_RATIO = 0.10
ENTRY_NORMAL_MIN_NOTIONAL_BALANCE_RATIO = 0.06
ENTRY_NOTIONAL_FLOOR_MAX_SIZE_PCT = 0.12
ENTRY_HIGH_PROFIT_MIN_NOTIONAL_BALANCE_RATIO = 0.75
ENTRY_HIGH_PROFIT_MIN_LEVERAGE = 8.0
ENTRY_HIGH_PROFIT_ELITE_MIN_LEVERAGE = 10.0
ENTRY_GOOD_PROBE_MIN_NOTIONAL_BALANCE_RATIO = 0.25
ENTRY_WINNER_ADD_MIN_NOTIONAL_BALANCE_RATIO = 0.35
ENTRY_STRONG_PROBE_MIN_NOTIONAL_BALANCE_RATIO = 0.45
ENTRY_ELITE_MIN_NOTIONAL_BALANCE_RATIO = 0.60
ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK = 0.82
ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_USDT = 0.75
ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_RATIO = 0.003
PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY = 0.66
PORTFOLIO_ROSTER_FILL_MIN_NET_PCT = 0.20
PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO = 0.25
PORTFOLIO_ROSTER_FILL_NOTIONAL_BALANCE_RATIO = 0.18
ENTRY_PNL_STRUCTURE_MIN_EXPECTED_PROFIT_USDT = 1.50
ENTRY_PNL_STRUCTURE_LOW_QUALITY_MAX_LOSS_MULTIPLE = 0.65
ENTRY_PNL_STRUCTURE_NORMAL_MAX_LOSS_MULTIPLE = 1.05
ENTRY_PNL_STRUCTURE_HIGH_QUALITY_MAX_LOSS_MULTIPLE = 1.35
ENTRY_BALANCED_PROBE_MAX_LOSS_USDT = 5.0
ENTRY_STRONG_PROBE_MAX_LOSS_USDT = 9.0


def _settings_max_leverage() -> float:
    return float(settings.max_leverage or 1.0)


@dataclass(slots=True)
class EntryProfitRiskSizingPolicy:
    """Apply entry sizing caps that depend on profit/risk context."""

    evaluator: EntryProfitRiskSizingEvaluator | None = None
    allocated_order_balance: EntryBalanceProvider | None = None
    entry_low_payoff_quality: Any | None = None
    entry_stop_loss_budget: Any | None = None
    entry_stress_stop: Any | None = None
    entry_existing_winner_context: Any | None = None
    max_leverage_provider: Callable[[], float] = _settings_max_leverage
    probe_max_loss_usdt: float = ENTRY_BALANCED_PROBE_MAX_LOSS_USDT
    strong_probe_max_loss_usdt: float = ENTRY_STRONG_PROBE_MAX_LOSS_USDT

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _missing_dependencies(self) -> list[str]:
        required = {
            "allocated_order_balance": self.allocated_order_balance,
            "entry_low_payoff_quality": self.entry_low_payoff_quality,
            "entry_stop_loss_budget": self.entry_stop_loss_budget,
            "entry_stress_stop": self.entry_stress_stop,
            "entry_existing_winner_context": self.entry_existing_winner_context,
        }
        return [name for name, value in required.items() if value is None]

    async def apply(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Apply profit-risk sizing to an entry decision."""

        if self.evaluator is not None:
            await self.evaluator(decision, model_mode, open_positions or [])
            return
        missing = self._missing_dependencies()
        if missing:
            raise RuntimeError(
                "EntryProfitRiskSizingPolicy requires dependencies: " + ", ".join(missing)
            )
        await self._apply_sizing(decision, model_mode, open_positions or [])

    async def _apply_sizing(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict] | None = None,
    ) -> None:
        """Cap entry size by planned USDT loss at stop, especially during drawdown recovery."""
        if not decision.is_entry:
            return
        raw: dict[str, Any] = self._safe_dict(decision.raw_response)
        opportunity: dict[str, Any] = self._safe_dict(raw.get("opportunity_score"))
        risk_mode = str(opportunity.get("risk_mode") or "normal")
        configured_max_loss = self._safe_float(opportunity.get("max_entry_stop_loss_usdt"), 0.0)

        stop_loss_pct = max(float(decision.stop_loss_pct or 0.0), 0.0)
        leverage = max(float(decision.suggested_leverage or 1.0), 1.0)
        current_size = max(float(decision.position_size_pct or 0.0), 0.0)
        if stop_loss_pct <= 0 or current_size <= 0:
            return
        weak_history = bool(opportunity.get("weak_history_requires_stronger_edge"))
        local_expected = self._safe_float(opportunity.get("server_profit_expected_return_pct"), 0.0)
        local_aligned = bool(opportunity.get("local_profit_aligned"))
        ml_aligned = bool(opportunity.get("ml_aligned"))
        profit_quality_ratio = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        min_profit_quality_ratio = self._safe_float(
            opportunity.get("min_profit_quality_ratio_required"),
            ENTRY_MIN_NET_PROFIT_QUALITY_RATIO,
        )
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        tail_risk = self._safe_float(opportunity.get("tail_risk_score"), 0.0)
        score = self._safe_float(opportunity.get("score"), 0.0)
        min_score_required = self._safe_float(
            opportunity.get("min_score_required"),
            MIN_ENTRY_OPPORTUNITY_SCORE,
        )
        raw_expected_return = self._safe_float(
            opportunity.get("raw_expected_return_pct", opportunity.get("expected_return_pct")),
            0.0,
        )
        expected_loss_pct = self._safe_float(opportunity.get("expected_loss_pct"), 0.0)
        small_win_big_loss_penalty = self._safe_float(
            opportunity.get("small_win_big_loss_penalty"),
            0.0,
        )
        loss_probability = self._safe_float(opportunity.get("server_profit_loss_probability"), 1.0)
        contribution_adjustment = self._safe_dict(opportunity.get("model_contribution_adjustment"))
        hard_contribution_caution = bool(contribution_adjustment.get("hard_caution"))
        quant_probe = self._safe_dict(raw.get("quant_profit_probe"))
        evidence_probe = self._safe_dict(raw.get("evidence_profit_probe"))
        quant_probe_triggered = bool(quant_probe.get("triggered"))
        evidence_probe_triggered = bool(evidence_probe.get("triggered"))
        strong_probe = bool(
            quant_probe.get("strong_probe") or evidence_probe.get("high_profit_potential")
        )
        roster_fill_relief = self._safe_dict(opportunity.get("portfolio_roster_fill_relief"))
        roster = self._safe_dict(opportunity.get("portfolio_roster"))
        symbol_profit_tier = str(opportunity.get("symbol_profit_tier") or "neutral")
        roster_fill_candidate = bool(
            roster_fill_relief.get("applied")
            or quant_probe.get("roster_fill_probe")
            or (
                roster.get("underfilled")
                and (quant_probe_triggered or evidence_probe_triggered)
                and not strong_probe
            )
        )
        if quant_probe_triggered:
            loss_probability = self._safe_float(
                quant_probe.get("loss_probability"),
                loss_probability,
            )
        if evidence_probe_triggered:
            loss_probability = self._safe_float(
                evidence_probe.get("loss_probability"),
                loss_probability,
            )
        high_quality_boost = self._safe_dict(raw.get("profit_quality_position_boost"))
        high_quality_entry = bool(
            high_quality_boost.get("allow")
            or (
                expected_net > 0
                and profit_quality_ratio >= max(min_profit_quality_ratio, 0.85)
                and tail_risk < 0.88
                and (local_aligned or ml_aligned or bool(opportunity.get("timeseries_aligned")))
            )
        )
        caps: list[str] = []
        evidence_score = self._safe_dict(opportunity.get("evidence_score"))
        evidence_sizing = apply_evidence_sizing_policy(
            evidence_score=evidence_score,
            current_size=current_size,
            leverage=leverage,
        )
        current_size = evidence_sizing.position_size_pct
        leverage = evidence_sizing.leverage
        decision.position_size_pct = current_size
        decision.suggested_leverage = leverage
        caps.extend(evidence_sizing.caps)
        evidence_effective_score = evidence_sizing.effective_score
        if weak_history:
            weak_history_max_size = (
                ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_SIZE
                if high_quality_entry
                else ENTRY_WEAK_HISTORY_MAX_SIZE
            )
            weak_history_max_leverage = (
                ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_LEVERAGE
                if high_quality_entry
                else ENTRY_WEAK_HISTORY_MAX_LEVERAGE
            )
            if current_size > weak_history_max_size:
                current_size = weak_history_max_size
                decision.position_size_pct = current_size
                caps.append(
                    "近期该币种/方向真实盈亏偏弱，但当前盈利证据同向，仓位按中小仓验证"
                    if high_quality_entry
                    else "近期该币种/方向真实盈亏偏弱，仓位降为小仓验证"
                )
            if leverage > weak_history_max_leverage:
                leverage = weak_history_max_leverage
                decision.suggested_leverage = leverage
                caps.append(
                    "近期亏损方向当前证据改善，杠杆放宽但仍限制上限"
                    if high_quality_entry
                    else "近期亏损方向限制杠杆，避免单笔亏损继续放大"
                )
        if local_expected < 0 and not (local_aligned or ml_aligned):
            if current_size > ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE:
                current_size = ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE
                decision.position_size_pct = current_size
                caps.append("服务器盈利模型反向或预期为负，仅允许极小仓")
            if leverage > ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_LEVERAGE:
                leverage = ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_LEVERAGE
                decision.suggested_leverage = leverage
                caps.append("服务器盈利模型未支持该方向，降低杠杆")
        low_payoff_quality = self.entry_low_payoff_quality.is_low_payoff(
            score=score,
            min_score_required=min_score_required,
            expected_net_return_pct=expected_net,
            profit_quality_ratio=profit_quality_ratio,
            raw_expected_return_pct=raw_expected_return,
            small_win_big_loss_penalty=small_win_big_loss_penalty,
            hard_contribution_caution=hard_contribution_caution,
            evidence_score=evidence_score,
            evidence_effective_score=evidence_effective_score,
        )
        if low_payoff_quality:
            high_quality_entry = False
            if current_size > ENTRY_LOW_QUALITY_MAX_SIZE:
                current_size = ENTRY_LOW_QUALITY_MAX_SIZE
                decision.position_size_pct = current_size
                caps.append("收益质量不足或存在小盈大亏风险，仓位降为小仓验证")
            if leverage > ENTRY_LOW_QUALITY_MAX_LEVERAGE:
                leverage = ENTRY_LOW_QUALITY_MAX_LEVERAGE
                decision.suggested_leverage = leverage
                caps.append("收益质量不足或存在小盈大亏风险，杠杆降到低档")
        if symbol_profit_tier == "side_loser" and not high_quality_entry:
            loser_size_cap = max(
                ENTRY_LOW_QUALITY_MAX_SIZE, current_size * ENTRY_SYMBOL_LOSER_SIZE_MULTIPLIER
            )
            if current_size > loser_size_cap:
                current_size = loser_size_cap
                decision.position_size_pct = current_size
                caps.append("该币种同方向近期真实亏损，未达到高质量解锁前缩小仓位")
        balance = await self.allocated_order_balance(model_mode, decision)
        if balance <= 0:
            return
        stop_loss_budget = self.entry_stop_loss_budget.resolve(
            risk_mode=risk_mode,
            configured_max_loss_usdt=configured_max_loss,
            balance=balance,
            high_quality_entry=high_quality_entry,
            low_payoff_quality=low_payoff_quality,
        )
        max_loss = stop_loss_budget.max_loss_usdt
        risk_budget_boost = stop_loss_budget.risk_budget_boost
        probe_budget_guard: dict[str, Any] = {"applied": False}
        if (quant_probe_triggered or evidence_probe_triggered) and not high_quality_entry:
            probe_budget = (
                self.strong_probe_max_loss_usdt if strong_probe else self.probe_max_loss_usdt
            )
            if 0.0 < probe_budget < max_loss:
                probe_budget_guard = {
                    "applied": True,
                    "strong_probe": bool(strong_probe),
                    "previous_max_stop_loss_usdt": round(max_loss, 6),
                    "max_stop_loss_usdt": round(probe_budget, 6),
                    "reason": (
                        "\u63a2\u9488\u6863\u4f7f\u7528\u72ec\u7acb\u7684\u5c0f\u98ce\u9669\u9884\u7b97"
                        "\uff0c\u5355\u7b14\u6700\u5927\u4e8f\u635f\u9650\u5728\u63a2\u9488\u9884\u7b97\u5185"
                        "\uff0c\u4ee5\u4fdd\u6301\u6210\u4ea4\u91cf\u7684\u540c\u65f6\u63a7\u4f4f\u5355\u6b21\u635f\u5931\u3002"
                    ),
                }
                max_loss = probe_budget
        pnl_structure_guard: dict[str, Any] = {
            "applied": False,
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality_ratio, 6),
        }
        snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        atr_14 = self._safe_float(snapshot.get("atr_14"), 0.0)
        current_price_for_atr = self._safe_float(
            snapshot.get("current_price", snapshot.get("close", 0.0)),
            0.0,
        )
        atr_pct = (
            atr_14 / current_price_for_atr if atr_14 > 0 and current_price_for_atr > 0 else 0.0
        )
        stress_stop_loss_pct = self.entry_stress_stop.stress_stop_loss_pct(
            declared_stop_loss_pct=stop_loss_pct,
            expected_loss_pct=expected_loss_pct,
            tail_risk_score=tail_risk,
            raw_expected_return_pct=raw_expected_return,
            low_payoff_quality=low_payoff_quality,
            atr_pct=atr_pct,
        )

        original_size_before_floor = current_size
        original_notional = balance * current_size * leverage
        target_min_notional = 0.0
        notional_floor_ratio = 0.0
        notional_floor_reason = ""
        quality_tier = "probe" if (quant_probe_triggered or evidence_probe_triggered) else "base"
        meaningful_size_reason = ""
        timeseries_aligned = bool(opportunity.get("timeseries_aligned"))
        existing_winner = self.entry_existing_winner_context.context(
            decision,
            open_positions or [],
        )
        has_existing_winner = bool(existing_winner.get("has_winner"))
        strong_probe_quality = bool(
            strong_probe
            and expected_net >= 0.75
            and profit_quality_ratio >= 0.85
            and loss_probability <= 0.42
            and (score >= 2.8 or evidence_probe_triggered)
        )
        elite_quality = bool(
            expected_net >= 1.20
            and profit_quality_ratio >= 1.20
            and loss_probability <= 0.38
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
            and (local_aligned or ml_aligned or timeseries_aligned)
        )
        high_profit_quality = bool(
            expected_net >= 1.60
            and profit_quality_ratio >= 1.45
            and loss_probability <= 0.34
            and tail_risk <= 0.72
            and score
            >= max(
                self._safe_float(
                    opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE
                ),
                1.15,
            )
            and (local_aligned or ml_aligned or timeseries_aligned)
        )
        good_probe_quality = bool(
            (quant_probe_triggered or evidence_probe_triggered)
            and expected_net >= 0.35
            and profit_quality_ratio >= 0.20
            and loss_probability <= 0.52
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
        )
        roster_fill_quality = bool(
            roster_fill_candidate
            and expected_net >= PORTFOLIO_ROSTER_FILL_MIN_NET_PCT
            and profit_quality_ratio >= PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO
            and loss_probability <= PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY
            and tail_risk <= 0.88
            and (local_aligned or ml_aligned or timeseries_aligned or quant_probe_triggered)
        )
        if has_existing_winner and (
            strong_probe_quality
            or elite_quality
            or (expected_net >= 0.55 and profit_quality_ratio >= 0.55 and loss_probability <= 0.48)
        ):
            current_profit = self._safe_float(existing_winner.get("unrealized_pnl"), 0.0)
            profit_ratio = self._safe_float(existing_winner.get("pnl_ratio"), 0.0)
            if (
                current_profit >= ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_USDT
                or profit_ratio >= ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_RATIO
            ):
                quality_tier = "winner_add"
                notional_floor_ratio = ENTRY_WINNER_ADD_MIN_NOTIONAL_BALANCE_RATIO
                meaningful_size_reason = (
                    "已有同币种同方向持仓浮盈，且新信号仍然同向；允许用中等仓位加到赢家上，"
                    "把正确判断转成更有意义的利润。"
                )
        if quality_tier != "winner_add" and elite_quality:
            quality_tier = "elite"
            notional_floor_ratio = ENTRY_ELITE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "净收益、盈亏质量和亏损概率同时达到精选标准，使用精选仓位地板。"
            )
        if (
            high_profit_quality
            and not low_payoff_quality
            and quality_tier in {"base", "probe", "good_probe", "strong_probe", "elite"}
        ):
            quality_tier = "high_profit"
            notional_floor_ratio = max(
                notional_floor_ratio, ENTRY_HIGH_PROFIT_MIN_NOTIONAL_BALANCE_RATIO
            )
            target_leverage_floor = (
                ENTRY_HIGH_PROFIT_ELITE_MIN_LEVERAGE
                if expected_net >= 2.20
                and profit_quality_ratio >= 1.80
                and loss_probability <= 0.30
                else ENTRY_HIGH_PROFIT_MIN_LEVERAGE
            )
            if leverage < target_leverage_floor:
                leverage = min(target_leverage_floor, self.max_leverage_provider())
                decision.suggested_leverage = leverage
            meaningful_size_reason = (
                "盈利可能性较大：预期净收益、盈亏质量、亏损概率和尾部风险同时达标，"
                "允许适当提高交易数量和杠杆，把高质量机会转成更大的实际收益。"
            )
        elif (
            quality_tier not in {"winner_add", "elite"}
            and strong_probe_quality
            and not low_payoff_quality
        ):
            quality_tier = "strong_probe"
            notional_floor_ratio = ENTRY_STRONG_PROBE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "强量化探针通过净收益、盈亏质量、亏损概率和机会分联合校验，升级为有效仓位。"
            )
        elif quality_tier == "probe" and good_probe_quality and not low_payoff_quality:
            quality_tier = "good_probe"
            notional_floor_ratio = ENTRY_GOOD_PROBE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "普通量化探针质量达标，不再只做极小验证仓，抬到可产生有效收益的基础仓位。"
            )
        elif quality_tier in {"probe", "base"} and roster_fill_quality and not low_payoff_quality:
            quality_tier = "roster_fill"
            notional_floor_ratio = PORTFOLIO_ROSTER_FILL_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "当前组合低于目标持仓组数，且该信号仍为正期望；使用小而不碎的补齐仓位，"
                "让组合逐步接近 10 个独立持仓组。"
            )

        if notional_floor_ratio > 0:
            target_min_notional = balance * notional_floor_ratio
            notional_floor_reason = meaningful_size_reason
            high_quality_entry = high_quality_entry or quality_tier in {
                "elite",
                "strong_probe",
                "winner_add",
                "high_profit",
            }
            if symbol_profit_tier in {"side_winner", "symbol_winner"} and quality_tier in {
                "base",
                "probe",
                "good_probe",
                "roster_fill",
            }:
                target_min_notional *= 1.15
                notional_floor_reason = (
                    f"{notional_floor_reason} 该币种近期真实盈利，名义本金地板小幅上调。"
                    if notional_floor_reason
                    else "该币种近期真实盈利，名义本金地板小幅上调。"
                )

        if high_quality_entry and not low_payoff_quality:
            quality_reference = max(min_profit_quality_ratio, 0.85)
            quality_multiplier = min(
                max(profit_quality_ratio / max(quality_reference, 1e-12), 0.75),
                1.35,
            )
            default_floor_ratio = ENTRY_HIGH_QUALITY_MIN_NOTIONAL_BALANCE_RATIO * quality_multiplier
            if default_floor_ratio > notional_floor_ratio:
                notional_floor_ratio = default_floor_ratio
                target_min_notional = balance * notional_floor_ratio
                notional_floor_reason = (
                    "高质量盈利证据同向，按当前可用资金和信号质量动态抬高名义本金，避免无意义小盈"
                )
        elif (
            expected_net > 0
            and profit_quality_ratio >= 0.65
            and (local_aligned or ml_aligned or timeseries_aligned)
        ):
            quality_multiplier = min(max(profit_quality_ratio / 0.65, 0.75), 1.25)
            notional_floor_ratio = ENTRY_NORMAL_MIN_NOTIONAL_BALANCE_RATIO * quality_multiplier
            target_min_notional = balance * notional_floor_ratio
            notional_floor_reason = (
                "普通正收益信号获得本地模型或时序模型支持，按当前可用资金动态设置基础名义本金"
            )

        intended_notional_for_profit = max(original_notional, target_min_notional)
        expected_profit_usdt = intended_notional_for_profit * max(expected_net, 0.0) / 100.0
        if expected_net > 0 and intended_notional_for_profit > 0:
            max_loss_multiple = (
                ENTRY_PNL_STRUCTURE_LOW_QUALITY_MAX_LOSS_MULTIPLE
                if low_payoff_quality or symbol_profit_tier == "side_loser"
                else (
                    ENTRY_PNL_STRUCTURE_HIGH_QUALITY_MAX_LOSS_MULTIPLE
                    if high_quality_entry
                    or quality_tier in {"elite", "winner_add", "high_profit", "strong_probe"}
                    else ENTRY_PNL_STRUCTURE_NORMAL_MAX_LOSS_MULTIPLE
                )
            )
            structure_max_loss = max(
                ENTRY_PNL_STRUCTURE_MIN_EXPECTED_PROFIT_USDT,
                expected_profit_usdt * max_loss_multiple,
            )
            if structure_max_loss < max_loss:
                previous_max_loss = max_loss
                max_loss = max(structure_max_loss, 1.0)
                pnl_structure_guard = {
                    "applied": True,
                    "previous_max_stop_loss_usdt": round(previous_max_loss, 6),
                    "max_stop_loss_usdt": round(max_loss, 6),
                    "expected_profit_usdt": round(expected_profit_usdt, 6),
                    "expected_net_return_pct": round(expected_net, 6),
                    "max_loss_multiple": round(max_loss_multiple, 6),
                    "quality_tier": quality_tier,
                    "symbol_profit_tier": symbol_profit_tier,
                    "reason": "按预期净收益动态压缩单笔止损预算，避免小盈大亏结构。",
                }

        notional_floor_blocked = ""
        if target_min_notional > 0:
            if low_payoff_quality:
                notional_floor_blocked = "收益质量不足或小盈大亏风险偏高，不抬高仓位"
            elif tail_risk >= 0.88:
                notional_floor_blocked = "尾部风险偏高，不抬高仓位"
            elif weak_history and not high_quality_entry:
                notional_floor_blocked = "该币种/方向近期真实盈亏偏弱，普通信号不抬高仓位"
            elif local_expected < 0 and not (local_aligned or ml_aligned):
                notional_floor_blocked = (
                    "服务器盈利模型预期为负且未获得本地模型同向支持，不抬高仓位"
                )
            else:
                risk_max_size = max_loss / max(balance * leverage * stress_stop_loss_pct, 1e-12)
                floor_size = target_min_notional / max(balance * leverage, 1e-12)
                raised_size = min(
                    max(current_size, floor_size),
                    max(risk_max_size, 0.001),
                    ENTRY_NOTIONAL_FLOOR_MAX_SIZE_PCT,
                )
                if raised_size > current_size:
                    current_size = raised_size
                    decision.position_size_pct = current_size

        planned_loss = balance * current_size * leverage * stress_stop_loss_pct
        max_size_pct = max_loss / max(balance * leverage * stress_stop_loss_pct, 1e-12)
        max_size_pct = max(min(max_size_pct, current_size), 0.001)
        raw["profit_risk_sizing"] = {
            "applied": planned_loss > max_loss,
            "risk_mode": risk_mode,
            "original_position_size_pct": round(original_size_before_floor, 6),
            "position_size_pct": round(
                max_size_pct if planned_loss > max_loss else current_size, 6
            ),
            "planned_stop_loss_usdt": round(planned_loss, 6),
            "max_stop_loss_usdt": round(max_loss, 6),
            "declared_stop_loss_pct": round(stop_loss_pct, 6),
            "stress_stop_loss_pct": round(stress_stop_loss_pct, 6),
            "atr_14": round(atr_14, 8),
            "atr_pct": round(atr_pct, 8),
            "low_payoff_quality": bool(low_payoff_quality),
            "quality_caps": caps,
            "evidence_score": evidence_score,
            "high_quality_entry": bool(high_quality_entry),
            "high_profit_quality": bool(high_profit_quality),
            "quality_tier": quality_tier,
            "meaningful_size_reason": meaningful_size_reason,
            "same_side_existing_winner": existing_winner,
            "risk_budget_boost": risk_budget_boost,
            "probe_budget_guard": probe_budget_guard,
            "pnl_structure_guard": pnl_structure_guard,
            "notional_floor_applied": current_size > original_size_before_floor,
            "original_notional_usdt": round(original_notional, 6),
            "target_min_notional_usdt": round(target_min_notional, 6),
            "target_min_notional_balance_ratio": round(notional_floor_ratio, 6),
            "notional_floor_reason": notional_floor_reason,
            "notional_floor_blocked": notional_floor_blocked,
            "final_notional_usdt": round(balance * current_size * leverage, 6),
        }
        if planned_loss > max_loss:
            decision.position_size_pct = max_size_pct
            raw["profit_risk_sizing"]["capped_stop_loss_usdt"] = round(
                balance * max_size_pct * leverage * stress_stop_loss_pct,
                6,
            )
            raw["profit_risk_sizing"][
                "reason"
            ] = "position size capped by stress-stop budget to prevent small-win-big-loss structure"
        decision.raw_response = raw
