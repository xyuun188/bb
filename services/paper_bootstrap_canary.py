"""Read-only recovery for positions created by retired paper canaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

PAPER_BOOTSTRAP_CANARY_VERSION = "2026-07-21.paper-normal-strategy.v1"
PAPER_BOOTSTRAP_SIZING_VERSION = "2026-07-21.paper-normal-sizing.v1"
PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION = "2026-07-19.paper-bootstrap-position-lifecycle.v2"
PAPER_BOOTSTRAP_LEGACY_CANARY_VERSIONS = frozenset({"2026-07-19.paper-bootstrap-canary.v3"})
PAPER_BOOTSTRAP_MIN_FILL_DRIFT_RESERVE_FRACTION = 0.0025
PAPER_NORMAL_TRADE_PURPOSE = "execute_normal_paper_strategy_and_learn_after_settlement"
PAPER_NORMAL_POSITION_EXIT_POLICY = "dynamic_strategy_risk_and_position_review"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _positive(value: Any) -> float:
    result = _finite(value)
    return max(result or 0.0, 0.0)


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_raw(row: Any) -> dict[str, Any]:
    return _safe_dict(
        _row_value(row, "raw_llm_response")
        or _row_value(row, "raw_response")
        or _row_value(row, "decision_learning_snapshot")
    )


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def build_paper_canary_position_lifecycle(decision: Any) -> dict[str, Any]:
    """Recover a lifecycle from an already-executed legacy canary decision."""

    raw = _row_raw(decision)
    contract = _safe_dict(raw.get("paper_bootstrap_canary"))
    action = str(_row_value(decision, "action") or "").lower()
    executed_at = _as_utc(_row_value(decision, "executed_at"))
    selected = _safe_dict(contract.get("selected_observation"))
    horizon_minutes = int(_positive(selected.get("horizon_minutes")) or 0)
    if (
        contract.get("version")
        not in {PAPER_BOOTSTRAP_CANARY_VERSION, *PAPER_BOOTSTRAP_LEGACY_CANARY_VERSIONS}
        or contract.get("purpose") == PAPER_NORMAL_TRADE_PURPOSE
        or contract.get("position_exit_policy") == PAPER_NORMAL_POSITION_EXIT_POLICY
        or contract.get("authorized") is not True
        or contract.get("requested") is not True
        or contract.get("execution_scope") != "paper_only"
        or contract.get("production_permission") is not False
        or not bool(_row_value(decision, "is_paper"))
        or not bool(_row_value(decision, "was_executed"))
        or action not in {"long", "short"}
        or executed_at is None
        or horizon_minutes <= 0
    ):
        return {}
    expires_at = executed_at + timedelta(minutes=horizon_minutes)
    return {
        "version": PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION,
        "kind": "paper_bootstrap_canary_position",
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "decision_id": _row_value(decision, "id"),
        "symbol": str(_row_value(decision, "symbol") or ""),
        "side": action,
        "executed_at": executed_at.isoformat(),
        "horizon_minutes": horizon_minutes,
        "expires_at": expires_at.isoformat(),
        "artifact_version": contract.get("artifact_version"),
        "source_contract_version": contract.get("version"),
    }


def paper_canary_position_lifecycle(position: dict[str, Any]) -> dict[str, Any]:
    direct = _safe_dict(position.get("paper_canary_lifecycle"))
    if direct:
        return direct
    management = _safe_dict(position.get("current_management_contract"))
    return _safe_dict(management.get("paper_canary_lifecycle"))


def assess_paper_canary_position_horizon(
    position: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Observe a historical horizon without granting position-exit authority."""

    lifecycle = paper_canary_position_lifecycle(position)
    current = _as_utc(now) or datetime.now(UTC)
    try:
        expires_at = datetime.fromisoformat(
            str(lifecycle.get("expires_at") or "").replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        expires_at = None
    expires_at = _as_utc(expires_at)
    position_side = str(position.get("side") or "").lower()
    lifecycle_side = str(lifecycle.get("side") or "").lower()
    position_symbol = (
        str(position.get("symbol") or "")
        .upper()
        .replace("-", "")
        .replace("/", "")
        .replace(":USDT", "")
    )
    lifecycle_symbol = (
        str(lifecycle.get("symbol") or "")
        .upper()
        .replace("-", "")
        .replace("/", "")
        .replace(":USDT", "")
    )
    horizon_minutes = int(_positive(lifecycle.get("horizon_minutes")) or 0)
    authorized = bool(
        lifecycle.get("version") == PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION
        and lifecycle.get("kind") == "paper_bootstrap_canary_position"
        and lifecycle.get("authorized") is True
        and lifecycle.get("execution_scope") == "paper_only"
        and lifecycle.get("production_permission") is False
        and str(position.get("execution_mode") or "").lower() == "paper"
        and position_side in {"long", "short"}
        and lifecycle_side == position_side
        and bool(position_symbol)
        and position_symbol == lifecycle_symbol
        and horizon_minutes > 0
        and expires_at is not None
    )
    return {
        "authorized": authorized,
        "elapsed": bool(authorized and current >= expires_at),
        "horizon_minutes": horizon_minutes,
        "expires_at": expires_at.isoformat() if expires_at is not None else None,
        "decision_id": lifecycle.get("decision_id"),
        "version": lifecycle.get("version"),
    }
