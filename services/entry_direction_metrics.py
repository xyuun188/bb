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


@dataclass(frozen=True, slots=True)
class SelectedEntryMetrics:
    """Metrics resolved for the actual entry side being executed."""

    side: str
    expected_net_return_pct: float
    profit_quality_ratio: float
    server_profit_expected_return_pct: float
    loss_probability: float
    tail_risk_score: float
    aggregate_expected_net_return_pct: float
    aggregate_profit_quality_ratio: float
    source: str

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


def independent_probe_expert_support(raw: dict[str, Any], side: str) -> list[str]:
    """Return independent retry experts that explicitly support the probe side."""

    opinions = raw.get("opinions")
    if not isinstance(opinions, list):
        return []
    support: list[str] = []
    for opinion in opinions:
        if not isinstance(opinion, dict):
            continue
        if not opinion.get("independent_expert_retry"):
            continue
        action = str(opinion.get("action") or "").lower()
        confidence = safe_float(opinion.get("confidence"), 0.0)
        if action == side and confidence >= 0.55:
            support.append(str(opinion.get("model_name") or opinion.get("name") or "unknown"))
    return support


def original_hold_probe_without_support(raw: dict[str, Any], side: str) -> bool:
    """Return whether this is an AI-HOLD probe without independent side support."""

    probe = safe_dict(raw.get("evidence_profit_probe"))
    if not probe.get("triggered"):
        return False
    if str(probe.get("ai_original_action") or "").lower() != Action.HOLD.value:
        return False
    probe_side = str(probe.get("side") or "").lower()
    if probe_side and probe_side != side:
        return False
    return not independent_probe_expert_support(raw, side)


def selected_entry_metrics(decision: DecisionOutput) -> SelectedEntryMetrics:
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
    expected_net = safe_float(
        side_evidence.get("expected_net_return_pct"),
        aggregate_expected_net,
    )
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
    loss_probability = safe_float(
        side_evidence.get(
            "loss_probability",
            opportunity.get("server_profit_loss_probability"),
        ),
        1.0,
    )
    tail_risk = safe_float(
        side_evidence.get("tail_risk_score", opportunity.get("tail_risk_score")),
        0.0,
    )
    if side_evidence and original_hold_probe_without_support(raw, side):
        source = "entry_candidate_evidence:original_hold_probe_conservative"
        expected_net = min(expected_net, aggregate_expected_net)
        profit_quality = min(profit_quality, aggregate_profit_quality)
        server_profit_expected = min(
            server_profit_expected,
            safe_float(opportunity.get("server_profit_expected_return_pct"), 0.0),
        )
        loss_probability = max(
            loss_probability,
            safe_float(opportunity.get("server_profit_loss_probability"), loss_probability),
        )
        tail_risk = max(
            tail_risk,
            safe_float(opportunity.get("tail_risk_score"), tail_risk),
        )
    return SelectedEntryMetrics(
        side=side,
        expected_net_return_pct=expected_net,
        profit_quality_ratio=profit_quality,
        server_profit_expected_return_pct=server_profit_expected,
        loss_probability=loss_probability,
        tail_risk_score=tail_risk,
        aggregate_expected_net_return_pct=aggregate_expected_net,
        aggregate_profit_quality_ratio=aggregate_profit_quality,
        source=source,
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
