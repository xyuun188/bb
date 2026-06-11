"""AI analysis budget allocation between position review and entry scanning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from services.entry_strategy_mode import (
    PORTFOLIO_MIN_POSITION_GROUPS_TARGET,
    PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN,
)

POSITION_REVIEW_MAX_GROUPS_PER_ROUND = 6
POSITION_REVIEW_PRIORITY_MAX_GROUPS_PER_ROUND = 4
POSITION_REVIEW_HIGH_RISK_MAX_GROUPS_PER_ROUND = 8
POSITION_REVIEW_URGENT_EXIT_MAX_GROUPS_PER_ROUND = 14
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
MARKET_ANALYSIS_MEDIUM_RISK_CAP = 4
MARKET_ANALYSIS_HIGH_RISK_CAP = 2

NormalizeSymbol = Callable[[Any], str]
OpenPositionGroupCounter = Callable[[list[dict[str, Any]] | None], int]
PortfolioProfitContextProvider = Callable[[list[dict[str, Any]]], dict[str, Any]]
PositionReviewScanner = Callable[
    [
        list[tuple[tuple[str, str], list[dict[str, Any]]]],
        dict[str, Any],
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
    position_fast_exit_score: float = POSITION_REVIEW_FAST_EXIT_SCORE
    position_fast_add_score: float = POSITION_REVIEW_FAST_ADD_SCORE
    market_min_exploration_symbols: int = MARKET_ANALYSIS_MIN_EXPLORATION_SYMBOLS
    market_high_risk_min_exploration_symbols: int = (
        MARKET_ANALYSIS_HIGH_RISK_MIN_EXPLORATION_SYMBOLS
    )
    market_medium_risk_cap: int = MARKET_ANALYSIS_MEDIUM_RISK_CAP
    market_high_risk_cap: int = MARKET_ANALYSIS_HIGH_RISK_CAP
    target_position_groups: int = PORTFOLIO_MIN_POSITION_GROUPS_TARGET
    roster_fill_market_symbol_min: int = PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
    ) -> dict[str, Any]:
        """Build the budget context consumed by the trading loop."""

        base_market_limit = max(0, int(base_market_limit or 0))
        position_group_count = self.open_position_group_counter(open_positions)
        roster_underfilled = position_group_count < self.config.target_position_groups

        if not run_position_analysis or not open_positions:
            market_limit = base_market_limit if run_market_analysis else 0
            if run_market_analysis and roster_underfilled:
                market_limit = max(market_limit, self.config.roster_fill_market_symbol_min)
            return self._result(
                risk_level="none",
                market_symbol_limit=market_limit,
                position_max_groups=self.config.position_max_groups_per_round,
                forced_exit_count=0,
                urgent_exit_count=0,
                high_exit_count=0,
                priority_count=0,
                total_position_groups=0,
                roster_underfilled=roster_underfilled,
                position_group_count=position_group_count,
                reason=(
                    "No position-review risk needs scheduling; market analysis uses the "
                    "roster-fill candidate budget."
                    if roster_underfilled
                    else "No position-review risk needs scheduling; market analysis uses the "
                    "base candidate budget."
                ),
            )

        grouped_items = self._group_positions(open_positions)
        portfolio_profit_context = self.portfolio_profit_context_provider(open_positions)
        fast_scan = self.position_review_scanner(
            grouped_items,
            feature_vectors,
            portfolio_profit_context,
        )

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
        position_max_groups = self.config.position_max_groups_per_round
        market_limit = base_market_limit if run_market_analysis else 0
        if high_exit or len(forced_exit) >= 3:
            risk_level = "high"
            position_max_groups = max(
                self.config.position_high_risk_max_groups_per_round,
                min(
                    len(grouped_items),
                    self.config.position_urgent_exit_max_groups_per_round,
                    (
                        len(urgent_exit) + 2
                        if urgent_exit
                        else self.config.position_high_risk_max_groups_per_round
                    ),
                ),
            )
            market_limit = min(
                base_market_limit,
                max(
                    self.config.market_high_risk_min_exploration_symbols,
                    self.config.market_high_risk_cap,
                ),
            )
        elif forced_exit or len(priority) >= 3:
            risk_level = "medium"
            position_max_groups = max(
                self.config.position_max_groups_per_round,
                min(len(priority) + 2, self.config.position_high_risk_max_groups_per_round),
            )
            market_limit = min(
                base_market_limit,
                max(
                    self.config.market_min_exploration_symbols,
                    self.config.market_medium_risk_cap,
                ),
            )

        if new_pair_pause_reason:
            market_limit = 0
        elif run_market_analysis and base_market_limit > 0 and market_limit <= 0:
            market_limit = self.config.market_high_risk_min_exploration_symbols
        if (
            roster_underfilled
            and run_market_analysis
            and not new_pair_pause_reason
            and risk_level != "high"
        ):
            market_limit = max(market_limit, self.config.roster_fill_market_symbol_min)

        return self._result(
            risk_level=risk_level,
            market_symbol_limit=market_limit,
            position_max_groups=position_max_groups,
            forced_exit_count=len(forced_exit),
            urgent_exit_count=len(urgent_exit),
            high_exit_count=len(high_exit),
            priority_count=len(priority),
            total_position_groups=len(grouped_items),
            roster_underfilled=roster_underfilled,
            position_group_count=position_group_count,
            reason=(
                f"Position-review risk level {risk_level}: forced exits {len(forced_exit)}, "
                f"urgent exits {len(urgent_exit)}, high-risk exits {len(high_exit)}, "
                f"priority reviews {len(priority)}; this round reviews up to "
                f"{int(position_max_groups)} position groups and keeps "
                f"{max(0, int(market_limit))} entry-scan candidates."
                + (
                    f" Grouped positions are {position_group_count}/"
                    f"{self.config.target_position_groups}; roster-fill mode raises "
                    "entry exploration budget."
                    if roster_underfilled and risk_level != "high"
                    else ""
                )
            ),
        )

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

    def _result(
        self,
        *,
        risk_level: str,
        market_symbol_limit: int,
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
            "position_max_groups": max(1, int(position_max_groups)),
            "forced_exit_groups": forced_exit_count,
            "urgent_exit_groups": urgent_exit_count,
            "high_exit_groups": high_exit_count,
            "priority_groups": priority_count,
            "total_position_groups": total_position_groups,
            "roster_underfilled": roster_underfilled,
            "position_group_count": position_group_count,
            "target_position_groups": self.config.target_position_groups,
            "reason": reason,
        }
