"""Selected entry-side metrics for execution policies.

Execution policies must evaluate the direction that will actually be submitted
(long or short).  Aggregate opportunity metrics can describe the best side in a
long/short competition, so using them for a selected order can let the opposite
side's positive expectancy hide the submitted side's negative expectancy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from core.symbols import normalize_trading_symbol
from services.normal_paper_trade import (
    NORMAL_PAPER_TRADE_VERSION,
    normalize_normal_paper_contract,
    normal_paper_trade_contract_reasons,
    normal_paper_trade_observation_contract_reasons,
)


@dataclass(frozen=True, slots=True)
class SelectedEntryMetrics:
    """Metrics resolved for the actual entry side being executed."""

    side: str
    expected_net_return_pct: float
    objective_net_return_pct: float
    profit_quality_ratio: float
    server_profit_expected_return_pct: float
    loss_probability: float
    tail_risk_score: float
    aggregate_expected_net_return_pct: float
    aggregate_profit_quality_ratio: float
    source: str
    expected_net_return_available: bool
    loss_probability_available: bool
    quality_observation: bool

    @property
    def has_selected_side(self) -> bool:
        return self.side in {"long", "short"}


def safe_dict(value: Any) -> dict[str, Any]:
    """Return a dict value or an empty mapping."""

    return value if isinstance(value, dict) else {}


def safe_float(value: Any, default: float = 0.0) -> float:
    """Parse a finite float-like value without raising."""

    try:
        if value is None:
            return default
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def entry_side_from_action(action: Action | str | None) -> str:
    """Resolve the trade side for entry actions."""

    value = action.value if isinstance(action, Action) else str(action or "")
    value = value.lower().strip()
    if value in {Action.LONG.value, "open_long", "buy"}:
        return "long"
    if value in {Action.SHORT.value, "open_short", "sell"}:
        return "short"
    return ""


def selected_side_evidence(raw: dict[str, Any], side: str) -> dict[str, Any]:
    """Return evidence for the submitted side when present."""

    evidence = safe_dict(raw.get("entry_candidate_evidence"))
    side_evidence = safe_dict(evidence.get(side))
    if side_evidence:
        return side_evidence
    direct_side = str(evidence.get("side") or "").lower()
    if direct_side == side:
        return evidence
    return {}


def selected_entry_metrics(
    decision: DecisionOutput,
    model_mode: str = "",
) -> SelectedEntryMetrics:
    """Resolve opportunity metrics for the action that will be submitted."""

    raw = safe_dict(decision.raw_response)
    opportunity = safe_dict(raw.get("opportunity_score"))
    side = entry_side_from_action(decision.action) or str(opportunity.get("side") or "").lower()
    side_evidence = selected_side_evidence(raw, side)
    aggregate_expected_net = safe_float(opportunity.get("expected_net_return_pct"), 0.0)
    aggregate_profit_quality = safe_float(opportunity.get("profit_quality_ratio"), 0.0)
    source = "opportunity_score"
    if side_evidence:
        source = "entry_candidate_evidence"
    expected_net_value = side_evidence.get(
        "expected_net_return_pct",
        opportunity.get("expected_net_return_pct"),
    )
    objective_net_value = side_evidence.get(
        "return_lcb_pct",
        opportunity.get("return_lcb_pct"),
    )
    loss_probability_value = side_evidence.get(
        "loss_probability",
        opportunity.get("server_profit_loss_probability"),
    )
    quality_observation = False

    normal_paper = safe_dict(raw.get("normal_paper_trade"))
    normalized_historical = normalize_normal_paper_contract(normal_paper)
    analysis_contract = (
        normal_paper
        if normal_paper.get("version") == NORMAL_PAPER_TRADE_VERSION
        else normalized_historical
    )
    contract_symbol = normalize_trading_symbol(analysis_contract.get("symbol"))
    decision_symbol = normalize_trading_symbol(decision.symbol)
    contract_reasons = normal_paper_trade_contract_reasons(normal_paper)
    observation_contract = False
    if (
        contract_reasons
        and str(model_mode or "").lower() == "paper"
        and normal_paper.get("selection_reason") == "paper_quality_observation"
    ):
        observation_contract = not normal_paper_trade_observation_contract_reasons(
            normal_paper
        )
    historical_contract = bool(
        normalized_historical
        and normalized_historical.get("protocol_state") == "historical_normalized"
    )
    valid_selected_contract = bool(
        str(model_mode or "").lower() == "paper"
        and (not contract_reasons or observation_contract)
        and not historical_contract
        and str(analysis_contract.get("side") or "").lower() == side
        and contract_symbol
        and contract_symbol == decision_symbol
    )
    if valid_selected_contract:
        source = "normal_paper_trade_contract"
        expected_net_value = normal_paper.get("expected_net_return_pct")
        objective_net_value = normal_paper.get("objective_net_return_pct")
        loss_probability_value = normal_paper.get("loss_probability")
        quality_observation = (
            normal_paper.get("selection_reason") == "paper_quality_observation"
        )

    expected_net_available = expected_net_value is not None
    loss_probability_available = loss_probability_value is not None
    expected_net = safe_float(expected_net_value, aggregate_expected_net)
    objective_net = safe_float(objective_net_value, 0.0)
    profit_quality = safe_float(
        side_evidence.get("profit_quality_ratio"),
        aggregate_profit_quality,
    )
    server_profit_expected = safe_float(
        side_evidence.get(
            "server_profit_expected_return_pct",
            opportunity.get("server_profit_expected_return_pct"),
        ),
        0.0,
    )
    loss_probability = safe_float(loss_probability_value, 1.0)
    tail_risk = safe_float(
        side_evidence.get("tail_risk_score", opportunity.get("tail_risk_score")),
        0.0,
    )
    return SelectedEntryMetrics(
        side=side,
        expected_net_return_pct=expected_net,
        objective_net_return_pct=objective_net,
        profit_quality_ratio=profit_quality,
        server_profit_expected_return_pct=server_profit_expected,
        loss_probability=loss_probability,
        tail_risk_score=tail_risk,
        aggregate_expected_net_return_pct=aggregate_expected_net,
        aggregate_profit_quality_ratio=aggregate_profit_quality,
        source=source,
        expected_net_return_available=expected_net_available,
        loss_probability_available=loss_probability_available,
        quality_observation=quality_observation,
    )


def write_selected_metrics_snapshot(
    raw: dict[str, Any],
    metrics: SelectedEntryMetrics,
    *,
    blocked: bool,
    policy: str,
) -> None:
    """Persist an operator-facing selected-side quality snapshot."""

    opportunity = safe_dict(raw.get("opportunity_score"))
    opportunity["selected_side_quality_gate"] = {
        "blocked": bool(blocked),
        "policy": policy,
        "side": metrics.side,
        "source": metrics.source,
        "selected_expected_net_return_pct": round(metrics.expected_net_return_pct, 6),
        "selected_objective_net_return_pct": round(
            metrics.objective_net_return_pct,
            6,
        ),
        "aggregate_expected_net_return_pct": round(
            metrics.aggregate_expected_net_return_pct,
            6,
        ),
        "selected_profit_quality_ratio": round(metrics.profit_quality_ratio, 6),
        "aggregate_profit_quality_ratio": round(metrics.aggregate_profit_quality_ratio, 6),
        "selected_server_profit_expected_return_pct": round(
            metrics.server_profit_expected_return_pct,
            6,
        ),
        "selected_loss_probability": round(metrics.loss_probability, 6),
        "selected_tail_risk_score": round(metrics.tail_risk_score, 6),
    }
    raw["opportunity_score"] = opportunity
