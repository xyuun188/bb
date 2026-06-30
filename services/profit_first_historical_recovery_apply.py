"""Safety helpers for applying Profit-First historical recovery packages."""

from __future__ import annotations

from typing import Any

APPROVAL_TOKEN = "PROFIT_FIRST_HISTORICAL_RECOVERY_APPROVED"


def build_historical_recovery_apply_plan(
    package: dict[str, Any],
    *,
    allowed_decision_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Return the subset of package items that may be applied to decisions."""

    allowed = {int(item) for item in allowed_decision_ids or [] if int(item) > 0}
    items = [_safe_dict(item) for item in _safe_list(package.get("items"))]
    applicable: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in items:
        item_type = str(item.get("item_type") or "")
        decision_id = _safe_int(item.get("decision_id"))
        patch = _safe_dict(item.get("proposed_raw_patch"))
        if item_type not in {"entry_decision_recovery", "exit_decision_recovery"}:
            skipped.append(
                _skip(item, reason="item_type_not_apply_supported_by_decision_raw_patch")
            )
            continue
        if decision_id <= 0:
            skipped.append(_skip(item, reason="missing_decision_id"))
            continue
        if allowed and decision_id not in allowed:
            skipped.append(_skip(item, reason="decision_id_not_in_apply_allowlist"))
            continue
        if not patch:
            skipped.append(_skip(item, reason="missing_proposed_raw_patch"))
            continue
        applicable.append(
            {
                "item_type": item_type,
                "decision_id": decision_id,
                "symbol": item.get("symbol"),
                "action": item.get("action"),
                "training_policy": item.get("training_policy") or "exclude_until_manual_trust",
                "proposed_raw_patch": patch,
            }
        )
    return {
        "applicable_items": applicable,
        "skipped_items": skipped,
        "summary": {
            "package_item_count": len(items),
            "applicable_count": len(applicable),
            "skipped_count": len(skipped),
            "allowed_decision_ids": sorted(allowed),
            "requires_backup": True,
            "requires_approval_token": True,
            "approval_token": APPROVAL_TOKEN,
        },
    }


def validate_apply_request(
    *,
    apply: bool,
    approval_token: str,
    allowed_decision_ids: list[int],
    applicable_count: int,
) -> tuple[bool, list[str]]:
    """Validate operator intent before any database mutation."""

    if not apply:
        return False, ["dry_run_only"]
    reasons: list[str] = []
    if approval_token != APPROVAL_TOKEN:
        reasons.append("approval_token_invalid")
    if not [item for item in allowed_decision_ids if int(item) > 0]:
        reasons.append("explicit_decision_id_allowlist_required")
    if applicable_count <= 0:
        reasons.append("no_applicable_items")
    return not reasons, reasons


def merge_raw_patch(raw: dict[str, Any] | None, patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a proposed raw patch into an existing raw payload."""

    result = dict(raw) if isinstance(raw, dict) else {}
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_raw_patch(result[key], value)
        else:
            result[key] = value
    return result


def _skip(item: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "item_type": item.get("item_type"),
        "decision_id": item.get("decision_id"),
        "order_id": item.get("order_id"),
        "exchange_order_id": item.get("exchange_order_id"),
        "reason": reason,
    }


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
