"""Read-only Profit-First no-entry and losing-exit governance report."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from services.profit_first_ranking import (
    DEFAULT_RANKING_LIMIT,
    ProfitFirstRankingService,
)

DEFAULT_GOVERNANCE_HOURS = 24
DEFAULT_GOVERNANCE_LIMIT = DEFAULT_RANKING_LIMIT


class ProfitFirstGovernanceReportService:
    """Build the 24h Profit-First governance report without mutating live state."""

    def __init__(
        self,
        *,
        ranking_service_factory: Callable[[], ProfitFirstRankingService] | None = None,
    ) -> None:
        self._ranking_service_factory = ranking_service_factory or ProfitFirstRankingService

    async def report(
        self,
        *,
        hours: int = DEFAULT_GOVERNANCE_HOURS,
        limit: int = DEFAULT_GOVERNANCE_LIMIT,
    ) -> dict[str, Any]:
        ranking_report = await self._ranking_service_factory().report(hours=hours, limit=limit)
        return self.build_report(ranking_report=ranking_report, hours=hours, limit=limit)

    def build_report(
        self,
        *,
        ranking_report: dict[str, Any],
        hours: int = DEFAULT_GOVERNANCE_HOURS,
        limit: int = DEFAULT_GOVERNANCE_LIMIT,
    ) -> dict[str, Any]:
        ranking = _safe_dict(ranking_report)
        recommendations = _safe_dict(ranking.get("brain_recommendations"))
        no_entry = _safe_dict(recommendations.get("no_entry_governance"))
        losing_exit = _safe_dict(recommendations.get("losing_exit_governance"))
        coverage = _safe_dict(recommendations.get("brain_output_coverage"))
        missing_outputs = _missing_brain_outputs(coverage)
        diagnosis = str(no_entry.get("diagnosis") or "insufficient_sample")
        loss_sample_count = _safe_int(losing_exit.get("sample_count"))
        no_entry_sample_count = _safe_int(no_entry.get("sample_count"))
        no_entry_action = _no_entry_action(diagnosis)
        losing_exit_action = _losing_exit_action(losing_exit)
        status = "ready"
        if ranking.get("report_available") is False:
            status = "unavailable"
        elif missing_outputs:
            status = "incomplete"
        return {
            "report_type": "profit_first_governance",
            "status": status,
            "generated_at": datetime.now(UTC).isoformat(),
            "window_hours": max(1, min(int(hours or DEFAULT_GOVERNANCE_HOURS), 24 * 14)),
            "limit": max(1, int(limit or DEFAULT_GOVERNANCE_LIMIT)),
            "read_only": True,
            "audit_only": True,
            "live_mutation": False,
            "live_entry_mutation": False,
            "live_exit_mutation": False,
            "live_weight_mutation": False,
            "live_sizing_mutation": False,
            "can_submit_orders": False,
            "can_start_trading_service": False,
            "can_change_model_routing": False,
            "can_change_strategy_weight": False,
            "can_increase_live_size": False,
            "requires_operator_resume_gate": True,
            "summary": {
                "ranking_ready": bool(ranking.get("ranking_ready")),
                "no_entry_sample_count": no_entry_sample_count,
                "losing_exit_sample_count": loss_sample_count,
                "no_entry_diagnosis": diagnosis,
                "missing_brain_output_count": len(missing_outputs),
                "ranking_disable_count": _safe_int(
                    _safe_dict(ranking.get("summary")).get("disable_count")
                ),
                "ranking_demote_count": _safe_int(
                    _safe_dict(ranking.get("summary")).get("demote_count")
                ),
            },
            "no_entry_governance": no_entry,
            "losing_exit_governance": losing_exit,
            "brain_output_coverage": coverage,
            "missing_brain_outputs": missing_outputs,
            "next_cycle_actions": _dedupe_actions(
                [
                    no_entry_action,
                    losing_exit_action,
                    *_recommendation_texts(
                        recommendations.get("no_entry_threshold_recommendations")
                    ),
                    *_recommendation_texts(recommendations.get("exit_policy_adjustments")),
                    *_recommendation_texts(recommendations.get("size_promotion_demotion")),
                ]
            )[:20],
            "policy": {
                "window_policy": "rolling_24h_profit_first_governance",
                "no_entry_must_be_classified": True,
                "losing_exit_must_be_attributed": True,
                "recommendations_are_read_only": True,
                "live_changes_require_go_no_go_and_operator_approval": True,
                "ranking_trade_fact_policy": _safe_dict(ranking.get("policy")).get(
                    "trade_fact_policy",
                    "okx_confirmed_closed_positions_only",
                ),
            },
            "ranking_summary": _safe_dict(ranking.get("summary")),
            "trade_fact_report": _safe_dict(ranking.get("trade_fact_report")),
            "safety_note": (
                "Read-only Profit-First governance report; it does not start trading, submit "
                "orders, close positions, change model routing, or mutate live sizing."
            ),
        }


def _missing_brain_outputs(coverage: dict[str, Any]) -> list[str]:
    required = (
        "source_weights",
        "strategy_weights",
        "lane_threshold_recommendations",
        "size_promotion_demotion",
        "no_entry_threshold_recommendations",
        "exit_policy_adjustments",
        "shadow_canary_live_decisions",
    )
    return [key for key in required if coverage.get(key) is not True]


def _no_entry_action(diagnosis: str) -> str:
    if diagnosis == "system_over_conservative_review":
        return "review_no_entry_thresholds_against_positive_shadow_outcomes"
    if diagnosis == "market_unattractive_by_expected_net":
        return "keep_entries_shadow_until_expected_net_improves"
    if diagnosis == "external_data_or_model_unavailable":
        return "fix_data_model_or_okx_availability_before_threshold_changes"
    if diagnosis == "mixed_blockers_review_top_reasons":
        return "review_top_no_entry_blockers_before_tuning"
    return "collect_more_no_entry_evidence"


def _losing_exit_action(losing_exit: dict[str, Any]) -> str:
    counts = {
        str(row.get("value") or ""): _safe_int(row.get("count"))
        for row in _safe_list(losing_exit.get("attribution_counts"))
        if isinstance(row, dict)
    }
    if counts.get("position_too_small_fee_drag", 0) > 0:
        return "pause_tiny_probe_repeats_or_raise_quality_floor_for_fee_drag_regime"
    if counts.get("exit_too_early", 0) > 0:
        return "tighten_capital_release_and_profit_drawdown_exit_rules"
    if counts.get("model_false_positive", 0) > 0:
        return "demote_false_positive_model_sources_until_shadow_recovery"
    if counts:
        return "apply_losing_exit_attribution_to_next_cycle_policy_review"
    return "collect_more_losing_exit_evidence"


def _recommendation_texts(rows: Any) -> list[str]:
    texts: list[str] = []
    for row in _safe_list(rows):
        if not isinstance(row, dict):
            continue
        value = str(row.get("recommendation") or "").strip()
        if value:
            texts.append(value)
    return texts


def _dedupe_actions(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
