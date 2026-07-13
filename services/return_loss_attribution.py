"""Factual attribution for negative fee-after return samples."""

from __future__ import annotations

from typing import Any


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_losing_exit_attribution(
    position_or_record: Any,
    *,
    entry_raw: dict[str, Any] | None = None,
    close_raw: dict[str, Any] | None = None,
    shadow: dict[str, Any] | None = None,
) -> str:
    """Classify only from recorded outcomes and explicit execution facts."""

    pnl = _safe_float(_row_get(position_or_record, "realized_pnl"))
    if pnl is not None and pnl >= 0:
        return ""
    entry = _safe_dict(entry_raw or _row_get(position_or_record, "entry_raw"))
    close = _safe_dict(close_raw or _row_get(position_or_record, "close_raw"))
    side = str(_row_get(position_or_record, "side") or "").lower()
    shadow_side = str(_safe_dict(shadow or _row_get(position_or_record, "shadow")).get("best_action") or "").lower()
    if shadow_side in {"long", "short"} and side in {"long", "short"} and shadow_side != side:
        return "entry_wrong_direction"

    close_evidence = _safe_dict(close.get("close_evidence"))
    explicit = str(
        close.get("loss_attribution")
        or close_evidence.get("loss_attribution")
        or _row_get(position_or_record, "loss_attribution")
        or ""
    ).strip()
    if explicit:
        return explicit
    text = " ".join(
        str(value or "")
        for value in (
            _row_get(position_or_record, "main_reason"),
            _row_get(position_or_record, "reason"),
            close.get("exit_intent"),
            close_evidence.get("reason"),
            close.get("execution_reason"),
        )
    ).lower()
    if any(marker in text for marker in ("slippage", "滑点", "execution", "成交")):
        return "execution_cost_or_slippage"
    if any(marker in text for marker in ("trend reversal", "反转", "invalidation")):
        return "market_regime_or_direction_reversal"
    if any(marker in text for marker in ("late exit", "too late", "退出过晚")):
        return "exit_too_late"
    if any(marker in text for marker in ("early exit", "too early", "退出过早")):
        return "exit_too_early"
    opportunity = _safe_dict(entry.get("opportunity_score"))
    expected = _safe_float(opportunity.get("expected_net_return_pct"))
    if expected is not None and expected > 0:
        return "fee_after_return_forecast_error"
    return "unknown_requires_review"
