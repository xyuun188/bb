"""AI analysis budget allocation between position review and entry scanning."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from services.entry_strategy_mode import (
    PORTFOLIO_MIN_POSITION_GROUPS_TARGET,
    PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN,
)

POSITION_REVIEW_MAX_GROUPS_PER_ROUND = 6
POSITION_REVIEW_HIGH_RISK_MAX_GROUPS_PER_ROUND = 8
POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND = 14
POSITION_REVIEW_MEDIUM_LOAD_GROUP_THRESHOLD = 13
POSITION_REVIEW_HIGH_LOAD_GROUP_THRESHOLD = 25
POSITION_REVIEW_MEDIUM_LOAD_MAX_GROUPS_PER_ROUND = 10
POSITION_REVIEW_HIGH_LOAD_MAX_GROUPS_PER_ROUND = 14
POSITION_REVIEW_FAST_EXIT_SCORE = 70.0
POSITION_REVIEW_FAST_ADD_SCORE = 62.0
POSITION_REVIEW_URGENT_EXIT_MARKERS = (
    "loss_expanding",
    "loss_needs_review",
    "near_stop",
    "adverse_momentum",
    "predictive_reversal",
)
MARKET_ANALYSIS_MIN_EXPLORATION_SYMBOLS = 2
MARKET_ANALYSIS_HIGH_RISK_MIN_EXPLORATION_SYMBOLS = 1
MARKET_ANALYSIS_NO_POSITION_CAP = 6
MARKET_ANALYSIS_LOW_RISK_OPEN_POSITION_CAP = 3
MARKET_ANALYSIS_MEDIUM_RISK_CAP = 2
MARKET_ANALYSIS_HIGH_RISK_CAP = 1

NormalizeSymbol = Callable[[Any], str]
OpenPositionGroupCounter = Callable[[list[dict[str, Any]] | None], int]
PortfolioProfitContextProvider = Callable[[list[dict[str, Any]]], dict[str, Any]]
PositionReviewScanner = Callable[
    [
        list[tuple[tuple[str, str], list[dict[str, Any]]]],
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
    ],
    dict[tuple[str, str], dict[str, Any]],
]
UrgentExitChecker = Callable[[dict[str, Any] | None], bool]


@dataclass(frozen=True, slots=True)
class AnalysisBudgetConfig:
    """Tunable limits for slow AI work in one trading round."""

    position_max_groups_per_round: int = POSITION_REVIEW_MAX_GROUPS_PER_ROUND
    position_high_risk_max_groups_per_round: int = POSITION_REVIEW_HIGH_RISK_MAX_GROUPS_PER_ROUND
    position_urgent_exit_max_groups_per_round: int = (
        POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND
    )
    position_medium_load_group_threshold: int = POSITION_REVIEW_MEDIUM_LOAD_GROUP_THRESHOLD
    position_high_load_group_threshold: int = POSITION_REVIEW_HIGH_LOAD_GROUP_THRESHOLD
    position_medium_load_max_groups_per_round: int = (
        POSITION_REVIEW_MEDIUM_LOAD_MAX_GROUPS_PER_ROUND
    )
    position_high_load_max_groups_per_round: int = POSITION_REVIEW_HIGH_LOAD_MAX_GROUPS_PER_ROUND
    position_fast_exit_score: float = POSITION_REVIEW_FAST_EXIT_SCORE
    position_fast_add_score: float = POSITION_REVIEW_FAST_ADD_SCORE
    market_min_exploration_symbols: int = MARKET_ANALYSIS_MIN_EXPLORATION_SYMBOLS
    market_high_risk_min_exploration_symbols: int = (
        MARKET_ANALYSIS_HIGH_RISK_MIN_EXPLORATION_SYMBOLS
    )
    market_no_position_cap: int = MARKET_ANALYSIS_NO_POSITION_CAP
    market_low_risk_open_position_cap: int = MARKET_ANALYSIS_LOW_RISK_OPEN_POSITION_CAP
    market_medium_risk_cap: int = MARKET_ANALYSIS_MEDIUM_RISK_CAP
    market_high_risk_cap: int = MARKET_ANALYSIS_HIGH_RISK_CAP
    target_position_groups: int = PORTFOLIO_MIN_POSITION_GROUPS_TARGET
    roster_fill_market_symbol_min: int = PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN


@dataclass(frozen=True, slots=True)
class AnalysisBudgetRuntime:
    """Resolved runtime budget after applying strategy-learning overrides."""

    position_max_groups_per_round: int
    position_high_risk_max_groups_per_round: int
    position_urgent_exit_max_groups_per_round: int
    position_medium_load_group_threshold: int
    position_high_load_group_threshold: int
    position_medium_load_max_groups_per_round: int
    position_high_load_max_groups_per_round: int
    market_min_exploration_symbols: int
    market_high_risk_min_exploration_symbols: int
    market_no_position_cap: int
    market_low_risk_open_position_cap: int
    market_medium_risk_cap: int
    market_high_risk_cap: int
    target_position_groups: int
    roster_fill_market_symbol_min: int
    max_position_group_bound: int
    source: str
    strategy_profile_id: str | None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(slots=True)
class AnalysisBudgetPolicy:
    """Allocate slow AI analysis capacity for one market/position round."""

    normalize_symbol: NormalizeSymbol
    open_position_group_counter: OpenPositionGroupCounter
    portfolio_profit_context_provider: PortfolioProfitContextProvider
    position_review_scanner: PositionReviewScanner
    urgent_exit_checker: UrgentExitChecker
    config: AnalysisBudgetConfig = field(default_factory=AnalysisBudgetConfig)
    default_model_name: str = "ensemble_trader"

    def context(
        self,
        open_positions: list[dict[str, Any]],
        feature_vectors: dict[str, Any],
        *,
        base_market_limit: int,
        run_position_analysis: bool,
        run_market_analysis: bool,
        new_pair_pause_reason: str | None = None,
        strategy_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the budget context consumed by the trading loop."""

        runtime = self._runtime(strategy_context)
        base_market_limit = max(0, int(base_market_limit or 0))
        position_group_count = self.open_position_group_counter(open_positions)
        roster_underfilled = position_group_count < runtime.target_position_groups

        if not open_positions:
            market_limit, market_limit_policy = self._market_limit_without_positions(
                base_market_limit=base_market_limit,
                run_market_analysis=run_market_analysis,
                runtime=runtime,
                roster_underfilled=roster_underfilled,
                new_pair_pause_reason=new_pair_pause_reason,
            )
            return self._result(
                runtime=runtime,
                risk_level="none",
                market_symbol_limit=market_limit,
                configured_market_symbol_limit=base_market_limit,
                market_limit_policy=market_limit_policy,
                position_max_groups=runtime.position_max_groups_per_round,
                forced_exit_count=0,
                urgent_exit_count=0,
                high_exit_count=0,
                priority_count=0,
                total_position_groups=0,
                roster_underfilled=roster_underfilled,
                position_group_count=position_group_count,
                reason=(
                    (
                        "当前没有持仓需要复盘，市场分析使用动态候选预算；"
                        "该预算只控制本轮送入大模型的候选数量，不是开仓门槛。"
                    )
                    if roster_underfilled
                    else (
                        "当前没有持仓需要复盘，市场分析使用基础动态候选预算；"
                        "该预算不是开仓门槛。"
                    )
                ),
            )

        if not run_position_analysis:
            market_limit, market_limit_policy = self._market_limit_with_positions(
                base_market_limit=base_market_limit,
                run_market_analysis=run_market_analysis,
                runtime=runtime,
                risk_level="low",
                roster_underfilled=roster_underfilled,
                new_pair_pause_reason=new_pair_pause_reason,
            )
            return self._result(
                runtime=runtime,
                risk_level="low",
                market_symbol_limit=market_limit,
                configured_market_symbol_limit=base_market_limit,
                market_limit_policy=market_limit_policy,
                position_max_groups=runtime.position_max_groups_per_round,
                forced_exit_count=0,
                urgent_exit_count=0,
                high_exit_count=0,
                priority_count=0,
                total_position_groups=position_group_count,
                roster_underfilled=roster_underfilled,
                position_group_count=position_group_count,
                reason=(
                    "当前存在持仓，持仓复盘由独立 position loop 并行负责；"
                    "市场分析只使用剩余小批量候选，避免抢占大模型资源。"
                ),
            )

        grouped_items = self._group_positions(open_positions)
        portfolio_profit_context = self.portfolio_profit_context_provider(open_positions)
        fast_scan = self.position_review_scanner(
            grouped_items,
            feature_vectors,
            portfolio_profit_context,
            strategy_context,
        )
        dynamic_position_max_groups = self._dynamic_position_max_groups(len(grouped_items), runtime)

        forced_exit = [
            scan
            for scan in fast_scan.values()
            if _safe_float(scan.get("exit_score"), 0.0) >= self.config.position_fast_exit_score
        ]
        urgent_exit = [scan for scan in fast_scan.values() if self.urgent_exit_checker(scan)]
        high_exit = [
            scan for scan in fast_scan.values() if _safe_float(scan.get("exit_score"), 0.0) >= 90.0
        ]
        priority = [
            scan
            for scan in fast_scan.values()
            if _safe_float(scan.get("priority_score"), 0.0) >= self.config.position_fast_add_score
        ]

        risk_level = "low"
        position_max_groups = dynamic_position_max_groups
        if high_exit or len(forced_exit) >= 3:
            risk_level = "high"
            position_max_groups = max(
                runtime.position_high_risk_max_groups_per_round,
                dynamic_position_max_groups,
                min(
                    len(grouped_items),
                    runtime.position_urgent_exit_max_groups_per_round,
                    (
                        len(urgent_exit) + 2
                        if urgent_exit
                        else runtime.position_high_risk_max_groups_per_round
                    ),
                ),
            )
        elif forced_exit or len(priority) >= 3:
            risk_level = "medium"
            position_max_groups = max(
                dynamic_position_max_groups,
                min(len(priority) + 2, runtime.position_high_risk_max_groups_per_round),
            )

        market_limit, market_limit_policy = self._market_limit_with_positions(
            base_market_limit=base_market_limit,
            run_market_analysis=run_market_analysis,
            runtime=runtime,
            risk_level=risk_level,
            roster_underfilled=roster_underfilled,
            new_pair_pause_reason=new_pair_pause_reason,
        )

        return self._result(
            runtime=runtime,
            risk_level=risk_level,
            market_symbol_limit=market_limit,
            configured_market_symbol_limit=base_market_limit,
            market_limit_policy=market_limit_policy,
            position_max_groups=position_max_groups,
            forced_exit_count=len(forced_exit),
            urgent_exit_count=len(urgent_exit),
            high_exit_count=len(high_exit),
            priority_count=len(priority),
            total_position_groups=len(grouped_items),
            roster_underfilled=roster_underfilled,
            position_group_count=position_group_count,
            reason=(
                f"持仓优先调度：风险级别 {risk_level}，强制退出 {len(forced_exit)} 组，"
                f"紧急退出 {len(urgent_exit)} 组，高风险退出 {len(high_exit)} 组，"
                f"优先复盘 {len(priority)} 组；本轮最多复盘 "
                f"{int(position_max_groups)} 组持仓，市场新开仓候选只保留 "
                f"{max(0, int(market_limit))} 个。该数量只限制大模型调度负载，"
                "不是开仓门槛。"
                + (
                    f" 当前持仓组 {position_group_count}/"
                    f"{runtime.target_position_groups}，组合仍未补齐，但市场扫描仍受持仓优先预算约束。"
                    if roster_underfilled and risk_level != "high"
                    else ""
                )
            ),
        )

    def _market_limit_without_positions(
        self,
        *,
        base_market_limit: int,
        run_market_analysis: bool,
        runtime: AnalysisBudgetRuntime,
        roster_underfilled: bool,
        new_pair_pause_reason: str | None,
    ) -> tuple[int, str]:
        if not run_market_analysis:
            return 0, "market_disabled"
        if new_pair_pause_reason:
            return 0, "new_pair_pause"
        if base_market_limit <= 0:
            return 0, "no_market_budget"
        dynamic_cap = max(
            runtime.market_min_exploration_symbols,
            min(
                runtime.market_no_position_cap,
                max(1, math.ceil(math.sqrt(base_market_limit))),
            ),
        )
        if roster_underfilled:
            dynamic_cap = max(
                dynamic_cap,
                min(runtime.roster_fill_market_symbol_min, runtime.market_no_position_cap),
            )
        return min(base_market_limit, dynamic_cap), "no_position_dynamic_market_budget"

    def _market_limit_with_positions(
        self,
        *,
        base_market_limit: int,
        run_market_analysis: bool,
        runtime: AnalysisBudgetRuntime,
        risk_level: str,
        roster_underfilled: bool,
        new_pair_pause_reason: str | None,
    ) -> tuple[int, str]:
        if not run_market_analysis:
            return 0, "market_disabled"
        if new_pair_pause_reason:
            return 0, "new_pair_pause"
        if base_market_limit <= 0:
            return 0, "no_market_budget"
        if risk_level == "high":
            cap = max(
                runtime.market_high_risk_min_exploration_symbols,
                runtime.market_high_risk_cap,
            )
            return min(base_market_limit, cap), "position_first_high_risk"
        if risk_level == "medium":
            cap = max(runtime.market_min_exploration_symbols, runtime.market_medium_risk_cap)
            return min(base_market_limit, cap), "position_first_medium_risk"

        cap = max(
            runtime.market_min_exploration_symbols,
            min(runtime.market_low_risk_open_position_cap, math.ceil(base_market_limit * 0.15)),
        )
        policy = (
            "position_first_low_risk_underfilled"
            if roster_underfilled
            else "position_first_low_risk"
        )
        return min(base_market_limit, cap), policy

    def _group_positions(
        self, open_positions: list[dict[str, Any]]
    ) -> list[tuple[tuple[str, str], list[dict[str, Any]]]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for pos in open_positions or []:
            symbol = self.normalize_symbol(pos.get("symbol"))
            model = str(pos.get("model_name") or self.default_model_name)
            if model and symbol:
                grouped.setdefault((model, symbol), []).append(pos)
        return list(grouped.items())

    def _dynamic_position_max_groups(
        self,
        total_groups: int,
        runtime: AnalysisBudgetRuntime,
    ) -> int:
        base = max(1, int(runtime.position_max_groups_per_round))
        total = max(0, int(total_groups or 0))
        if total >= int(runtime.position_high_load_group_threshold):
            return max(
                base,
                min(total, int(runtime.position_high_load_max_groups_per_round)),
            )
        if total >= int(runtime.position_medium_load_group_threshold):
            return max(
                base,
                min(total, int(runtime.position_medium_load_max_groups_per_round)),
            )
        return base

    def _runtime(self, strategy_context: dict[str, Any] | None) -> AnalysisBudgetRuntime:
        context = _safe_dict(strategy_context)
        learning = _safe_dict(context.get("strategy_learning"))
        runtime = _safe_dict(learning.get("runtime"))
        capacity = _safe_dict(context.get("dynamic_position_capacity"))
        roster = _safe_dict(context.get("portfolio_roster"))
        open_pressure = _safe_dict(learning.get("open_position_pressure"))
        budget = _safe_dict(runtime.get("analysis_budget"))
        source = "config"

        target_position_groups = self._first_positive_int(
            context.get("target_open_position_groups"),
            context.get("target_position_groups"),
            runtime.get("target_open_position_groups"),
            runtime.get("target_position_groups"),
            learning.get("target_position_groups"),
            capacity.get("target_limit"),
            roster.get("target_position_groups"),
            default=self.config.target_position_groups,
        )
        if target_position_groups != self.config.target_position_groups:
            source = "strategy_learning"

        max_bound = self._first_positive_int(
            runtime.get("max_open_positions"),
            open_pressure.get("max_open_positions"),
            context.get("max_open_positions_base"),
            capacity.get("base_limit"),
            default=max(
                target_position_groups,
                self.config.position_high_load_max_groups_per_round,
            ),
        )
        max_bound = max(max_bound, target_position_groups, 1)

        position_max_groups = self._runtime_int(
            budget.get("position_max_groups"),
            runtime.get("position_review_max_groups"),
            context.get("position_review_max_groups"),
            default=self.config.position_max_groups_per_round,
            upper=max_bound,
        )
        if position_max_groups != self.config.position_max_groups_per_round:
            source = "strategy_learning"

        position_high_risk_max = self._runtime_int(
            budget.get("position_high_risk_max_groups"),
            runtime.get("position_high_risk_max_groups"),
            runtime.get("position_review_high_risk_max_groups"),
            default=self.config.position_high_risk_max_groups_per_round,
            upper=max_bound,
        )
        position_urgent_max = self._runtime_int(
            budget.get("position_urgent_exit_max_groups"),
            runtime.get("position_urgent_exit_max_groups"),
            runtime.get("position_review_urgent_max_groups"),
            default=self.config.position_urgent_exit_max_groups_per_round,
            upper=max_bound,
        )
        medium_load_max = self._runtime_int(
            budget.get("position_medium_load_max_groups"),
            runtime.get("position_medium_load_max_groups"),
            default=self.config.position_medium_load_max_groups_per_round,
            upper=max_bound,
        )
        high_load_max = self._runtime_int(
            budget.get("position_high_load_max_groups"),
            runtime.get("position_high_load_max_groups"),
            default=self.config.position_high_load_max_groups_per_round,
            upper=max_bound,
        )

        medium_threshold = self._runtime_int(
            budget.get("position_medium_load_group_threshold"),
            runtime.get("position_medium_load_group_threshold"),
            default=min(
                self.config.position_medium_load_group_threshold,
                max(target_position_groups + 2, 1),
            ),
            upper=max_bound * 2,
        )
        high_threshold = self._runtime_int(
            budget.get("position_high_load_group_threshold"),
            runtime.get("position_high_load_group_threshold"),
            default=min(
                self.config.position_high_load_group_threshold,
                max(target_position_groups * 2, medium_threshold + 1),
            ),
            upper=max_bound * 3,
        )

        return AnalysisBudgetRuntime(
            position_max_groups_per_round=position_max_groups,
            position_high_risk_max_groups_per_round=max(
                position_high_risk_max,
                position_max_groups,
            ),
            position_urgent_exit_max_groups_per_round=max(
                position_urgent_max,
                position_high_risk_max,
                position_max_groups,
            ),
            position_medium_load_group_threshold=medium_threshold,
            position_high_load_group_threshold=max(high_threshold, medium_threshold + 1),
            position_medium_load_max_groups_per_round=max(medium_load_max, position_max_groups),
            position_high_load_max_groups_per_round=max(high_load_max, medium_load_max),
            market_min_exploration_symbols=self._runtime_int(
                budget.get("market_min_exploration_symbols"),
                runtime.get("market_min_exploration_symbols"),
                default=self.config.market_min_exploration_symbols,
                upper=10_000,
            ),
            market_high_risk_min_exploration_symbols=self._runtime_int(
                budget.get("market_high_risk_min_exploration_symbols"),
                runtime.get("market_high_risk_min_exploration_symbols"),
                default=self.config.market_high_risk_min_exploration_symbols,
                upper=10_000,
            ),
            market_no_position_cap=self._runtime_int(
                budget.get("market_no_position_cap"),
                runtime.get("market_no_position_cap"),
                default=self.config.market_no_position_cap,
                upper=10_000,
            ),
            market_low_risk_open_position_cap=self._runtime_int(
                budget.get("market_low_risk_open_position_cap"),
                runtime.get("market_low_risk_open_position_cap"),
                default=self.config.market_low_risk_open_position_cap,
                upper=10_000,
            ),
            market_medium_risk_cap=self._runtime_int(
                budget.get("market_medium_risk_cap"),
                runtime.get("market_medium_risk_cap"),
                default=self.config.market_medium_risk_cap,
                upper=10_000,
            ),
            market_high_risk_cap=self._runtime_int(
                budget.get("market_high_risk_cap"),
                runtime.get("market_high_risk_cap"),
                default=self.config.market_high_risk_cap,
                upper=10_000,
            ),
            target_position_groups=max(1, min(target_position_groups, max_bound)),
            roster_fill_market_symbol_min=self._runtime_int(
                budget.get("roster_fill_market_symbol_min"),
                runtime.get("roster_fill_market_symbol_min"),
                default=self.config.roster_fill_market_symbol_min,
                upper=10_000,
            ),
            max_position_group_bound=max_bound,
            source=source,
            strategy_profile_id=str(
                context.get("strategy_profile_id")
                or runtime.get("profile_id")
                or _safe_dict(learning.get("active_profile")).get("id")
                or ""
            )
            or None,
        )

    @staticmethod
    def _first_positive_int(*values: Any, default: int) -> int:
        for value in values:
            parsed = _safe_int(value, 0)
            if parsed > 0:
                return parsed
        return max(1, int(default or 1))

    @staticmethod
    def _runtime_int(*values: Any, default: int, upper: int) -> int:
        selected = max(1, int(default or 1))
        for value in values:
            parsed = _safe_int(value, 0)
            if parsed > 0:
                selected = parsed
                break
        return max(1, min(selected, max(1, int(upper or 1))))

    def _result(
        self,
        *,
        runtime: AnalysisBudgetRuntime,
        risk_level: str,
        market_symbol_limit: int,
        configured_market_symbol_limit: int,
        market_limit_policy: str,
        position_max_groups: int,
        forced_exit_count: int,
        urgent_exit_count: int,
        high_exit_count: int,
        priority_count: int,
        total_position_groups: int,
        roster_underfilled: bool,
        position_group_count: int,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "risk_level": risk_level,
            "market_symbol_limit": max(0, int(market_symbol_limit)),
            "configured_market_symbol_limit": max(0, int(configured_market_symbol_limit)),
            "market_limit_policy": market_limit_policy,
            "market_symbol_limit_is_entry_gate": False,
            "position_first_scheduling": True,
            "position_max_groups": max(1, int(position_max_groups)),
            "forced_exit_groups": forced_exit_count,
            "urgent_exit_groups": urgent_exit_count,
            "high_exit_groups": high_exit_count,
            "priority_groups": priority_count,
            "total_position_groups": total_position_groups,
            "roster_underfilled": roster_underfilled,
            "position_group_count": position_group_count,
            "target_position_groups": runtime.target_position_groups,
            "budget_source": runtime.source,
            "strategy_profile_id": runtime.strategy_profile_id,
            "max_position_group_bound": runtime.max_position_group_bound,
            "reason": reason,
        }
