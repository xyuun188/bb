"""Rebalance OKX protection coverage after an exchange-confirmed exit."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from core.symbols import normalize_trading_symbol
from services.protection_order_integrity import audit_protection_order_integrity

POSITION_PROTECTION_REBALANCE_VERSION = "2026-07-28.current-position-exact-coverage.v3"
PROTECTION_VERIFY_ATTEMPTS = 4
PROTECTION_TRANSITION_VERIFY_ATTEMPTS = 12
PROTECTION_VERIFY_DELAY_SECONDS = 0.5


class PositionProtectionRebalanceError(RuntimeError):
    """An exchange-confirmed exit left protection coverage unverified."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _target_side(decision: DecisionOutput) -> str:
    if decision.action == Action.CLOSE_LONG:
        return "long"
    if decision.action == Action.CLOSE_SHORT:
        return "short"
    return ""


def _position_side(position: dict[str, Any]) -> str:
    info = _safe_dict(position.get("info"))
    return str(position.get("side") or info.get("posSide") or "").lower()


def _protection_side(order: dict[str, Any]) -> str:
    return str(order.get("position_side") or "").lower()


def _response_success(response: Any) -> bool:
    if not isinstance(response, dict) or str(response.get("code") or "") != "0":
        return False
    rows = response.get("data")
    if not isinstance(rows, list) or not rows:
        return False
    return all(isinstance(row, dict) and str(row.get("sCode") or "") == "0" for row in rows)


def _response_algo_id(response: Any) -> str:
    rows = response.get("data") if isinstance(response, dict) else None
    row = rows[0] if isinstance(rows, list) and rows else None
    return str(row.get("algoId") or "") if isinstance(row, dict) else ""


def _coverage_is_exact(before_report: dict[str, Any], after_report: dict[str, Any]) -> bool:
    """Require the same position inventory and a completely clean protection audit."""

    return bool(
        before_report.get("position_inventory_fingerprint")
        == after_report.get("position_inventory_fingerprint")
        and not after_report.get("missing_keys")
        and not after_report.get("orphan_keys")
        and not after_report.get("coverage_mismatches")
        and not after_report.get("invalid_orders")
    )


def _amend_actions(actions: list[dict[str, Any]]) -> bool:
    return bool(actions) and all(
        str(action.get("action") or "") == "amend_size" for action in actions
    )


def _replacement_actions(actions: list[dict[str, Any]]) -> bool:
    """Return whether all actions can be rebuilt as one protection group."""

    action_names = {str(action.get("action") or "") for action in actions}
    return bool(actions) and action_names.issubset(
        {"amend_size", "create_delta", "cancel"}
    ) and bool(action_names.intersection({"amend_size", "create_delta"}))


def _same_number(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return False


def _same_protection_shape(order: dict[str, Any], action: dict[str, Any]) -> bool:
    """Match an already-created OCO so a retry stays idempotent."""

    return bool(
        str(order.get("inst_id") or "").upper()
        == str(action.get("inst_id") or "").upper()
        and str(order.get("position_side") or "").lower()
        == str(action.get("position_side") or "").lower()
        and _same_number(order.get("contracts"), action.get("new_contracts"))
        and _same_number(order.get("stop_loss_price"), action.get("stop_loss_price"))
        and _same_number(order.get("take_profit_price"), action.get("take_profit_price"))
    )


def _existing_exact_coverage_subset(
    orders: list[dict[str, Any]],
    *,
    desired_contracts: Any,
) -> list[dict[str, Any]]:
    """Find a small exact subset left by an earlier acknowledged replacement."""

    try:
        desired = float(desired_contracts)
    except (TypeError, ValueError):
        return []
    if desired <= 0:
        return []
    ordered = sorted(
        [item for item in orders if _safe_positive(item.get("contracts"))],
        key=lambda item: float(item.get("contracts") or 0.0),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    remaining = desired
    for order in ordered:
        contracts = float(order.get("contracts") or 0.0)
        if contracts <= remaining + 1e-9:
            selected.append(order)
            remaining -= contracts
        if abs(remaining) <= 1e-9:
            return selected
    return []


def _safe_positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


async def _wait_for_protection_verification(
    executor: Any,
    *,
    symbol: str,
    side: str,
    expected_position_fingerprint: str,
    attempts: int = PROTECTION_VERIFY_ATTEMPTS,
) -> tuple[dict[str, Any], int]:
    """Bound OKX eventual consistency before declaring a repair unsuccessful."""

    latest: dict[str, Any] | None = None
    total_attempts = max(1, int(attempts))
    for attempt in range(total_attempts):
        if attempt:
            await asyncio.sleep(PROTECTION_VERIFY_DELAY_SECONDS)
        latest = await protection_integrity_snapshot(
            executor,
            symbol=symbol,
            side=side,
        )
        report = latest["report"]
        if report.get("position_inventory_fingerprint") != expected_position_fingerprint:
            break
        if _coverage_is_exact(
            {"position_inventory_fingerprint": expected_position_fingerprint},
            report,
        ):
            break
    assert latest is not None
    return latest, attempt + 1


async def _wait_for_protection_order_transition(
    executor: Any,
    *,
    symbol: str,
    side: str,
    expected_position_fingerprint: str,
    present_algo_ids: set[str],
    absent_algo_ids: set[str],
    attempts: int = PROTECTION_TRANSITION_VERIFY_ATTEMPTS,
) -> tuple[dict[str, Any], int]:
    """Wait for explicit algo inventory evidence without assuming API acknowledgement."""

    latest: dict[str, Any] | None = None
    total_attempts = max(1, int(attempts))
    for attempt in range(total_attempts):
        if attempt:
            await asyncio.sleep(PROTECTION_VERIFY_DELAY_SECONDS)
        latest = await protection_integrity_snapshot(
            executor,
            symbol=symbol,
            side=side,
        )
        report = latest["report"]
        if report.get("position_inventory_fingerprint") != expected_position_fingerprint:
            break
        observed_ids = {
            str(order.get("algo_id") or "")
            for order in latest.get("protection_orders", [])
            if isinstance(order, dict)
        }
        if present_algo_ids.issubset(observed_ids) and not absent_algo_ids.intersection(
            observed_ids
        ):
            break
    assert latest is not None
    return latest, attempt + 1


def _fresh_replacement_actions(
    snapshot: dict[str, Any],
    *,
    expected_position_fingerprint: str,
) -> list[dict[str, Any]]:
    """Return only a fresh, pure-amend plan suitable for create-before-cancel."""

    report = _safe_dict(snapshot.get("report"))
    if report.get("position_inventory_fingerprint") != expected_position_fingerprint:
        raise RuntimeError("Protection replacement refused after position inventory changed")
    if report.get("repair_ready") is not True:
        raise RuntimeError("Protection replacement inventory is not repair-ready")
    actions = list(report.get("repair_actions") or [])
    if not _replacement_actions(actions):
        raise RuntimeError("Protection replacement requires a fresh replacement plan")
    return actions


async def protection_integrity_snapshot(
    executor: Any,
    *,
    symbol: str,
    side: str,
    missing_protection_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one symbol/side from strict OKX-native position and order APIs."""

    normalized_symbol = normalize_trading_symbol(symbol)
    positions = [
        row
        for row in await executor.get_positions_strict(normalized_symbol)
        if _position_side(row) == side
    ]
    protection_orders = [
        row
        for row in await executor.get_position_protection_orders(normalized_symbol)
        if _protection_side(row) == side
    ]
    pending_orders = await executor.get_open_orders_strict(normalized_symbol)
    contract_specs = await executor.get_contract_specs_strict([normalized_symbol])
    report = audit_protection_order_integrity(
        positions,
        protection_orders,
        pending_orders,
        contract_specs,
        pending_snapshot_complete=True,
        missing_protection_plans=(
            {(normalized_symbol, side): dict(missing_protection_plan)}
            if isinstance(missing_protection_plan, dict) and missing_protection_plan
            else None
        ),
    )
    return {
        "report": report,
        "positions": positions,
        "protection_orders": protection_orders,
        "pending_orders": pending_orders,
        "contract_specs": contract_specs,
    }


async def apply_protection_repair_actions(
    executor: Any,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply exact-size actions and roll back completed amendments on failure."""

    applied: list[dict[str, Any]] = []
    try:
        for action in actions:
            action_name = str(action.get("action") or "")
            if action_name == "amend_size":
                response = await executor.amend_position_protection_size(
                    inst_id=str(action.get("inst_id") or ""),
                    algo_id=str(action.get("algo_id") or ""),
                    contracts=float(action.get("new_contracts") or 0.0),
                )
            elif action_name == "create_delta":
                response = await executor.create_position_protection_order(
                    inst_id=str(action.get("inst_id") or ""),
                    position_side=str(action.get("position_side") or ""),
                    okx_position_side=str(action.get("okx_position_side") or "net"),
                    contracts=float(action.get("new_contracts") or 0.0),
                    stop_loss_price=float(action.get("stop_loss_price") or 0.0),
                    take_profit_price=float(action.get("take_profit_price") or 0.0),
                )
            elif action_name == "cancel":
                response = await executor.cancel_position_protection_order(
                    inst_id=str(action.get("inst_id") or ""),
                    algo_id=str(action.get("algo_id") or ""),
                )
            else:
                raise RuntimeError(f"Unsupported protection repair action: {action_name}")
            if not _response_success(response):
                raise RuntimeError(
                    f"OKX rejected protection {action_name} for algo {action.get('algo_id')}"
                )
            applied.append(
                {
                    "action": action,
                    "response": response,
                    "applied": True,
                    "created_algo_id": (
                        _response_algo_id(response) if action_name == "create_delta" else ""
                    ),
                }
            )
    except Exception as exc:
        rollback_results: list[dict[str, Any]] = []
        for item in reversed(applied):
            rollback = _safe_dict(item["action"].get("rollback"))
            if rollback.get("action") == "cancel_created":
                algo_id = str(item.get("created_algo_id") or "")
                try:
                    response = await executor.cancel_position_protection_order(
                        inst_id=str(rollback.get("inst_id") or ""),
                        algo_id=algo_id,
                    )
                    rollback_results.append(
                        {
                            "rollback": {**rollback, "algo_id": algo_id},
                            "response": response,
                            "applied": bool(algo_id and _response_success(response)),
                        }
                    )
                except Exception as rollback_exc:  # pragma: no cover
                    rollback_results.append(
                        {
                            "rollback": {**rollback, "algo_id": algo_id},
                            "applied": False,
                            "error": str(rollback_exc),
                        }
                    )
                continue
            if rollback.get("action") != "amend_size":
                rollback_results.append(
                    {"rollback": rollback, "applied": False, "manual_required": True}
                )
                continue
            try:
                response = await executor.amend_position_protection_size(
                    inst_id=str(rollback.get("inst_id") or ""),
                    algo_id=str(rollback.get("algo_id") or ""),
                    contracts=float(rollback.get("new_contracts") or 0.0),
                )
                rollback_results.append(
                    {
                        "rollback": rollback,
                        "response": response,
                        "applied": _response_success(response),
                    }
                )
            except Exception as rollback_exc:  # pragma: no cover - defensive exchange boundary
                rollback_results.append(
                    {
                        "rollback": rollback,
                        "applied": False,
                        "error": str(rollback_exc),
                    }
                )
        error = RuntimeError(str(exc))
        error.applied_actions = applied  # type: ignore[attr-defined]
        error.rollback_results = rollback_results  # type: ignore[attr-defined]
        raise error from exc
    return applied


async def replace_stuck_protection_amendments(
    executor: Any,
    actions: list[dict[str, Any]],
    *,
    expected_position_fingerprint: str = "",
    expected_input_fingerprint: str = "",
) -> list[dict[str, Any]]:
    """Replace OCOs whose OKX amend queue is stuck without an unprotected gap."""

    if not _replacement_actions(actions):
        raise RuntimeError(
            "Protection replacement only accepts amend_size/create_delta/cancel actions"
        )

    first_action = actions[0]
    symbol = normalize_trading_symbol(str(first_action.get("inst_id") or ""))
    side = str(first_action.get("position_side") or "").lower()
    if any(
        normalize_trading_symbol(str(action.get("inst_id") or "")) != symbol
        or (
            str(action.get("position_side") or "").lower()
            and str(action.get("position_side") or "").lower() != side
        )
        for action in actions
    ):
        raise RuntimeError("Stuck-amend replacement requires one position key")
    preflight = await protection_integrity_snapshot(
        executor,
        symbol=symbol,
        side=side,
    )
    preflight_report = preflight["report"]
    if expected_position_fingerprint and (
        preflight_report.get("position_inventory_fingerprint") != expected_position_fingerprint
    ):
        raise RuntimeError("Protection replacement refused after position inventory changed")
    if expected_input_fingerprint and (
        preflight_report.get("input_fingerprint") != expected_input_fingerprint
    ):
        raise RuntimeError("Protection replacement refused stale protection inventory")
    fresh_actions = _fresh_replacement_actions(
        preflight,
        expected_position_fingerprint=(
            expected_position_fingerprint
            or str(preflight_report.get("position_inventory_fingerprint") or "")
        ),
    )
    if fresh_actions != actions:
        raise RuntimeError("Protection replacement refused a changed amend plan")

    # The audit plan only contains orders whose size changes plus any positive
    # residual.  A full replacement must also carry forward unchanged old OCOs;
    # otherwise the rebuilt group can silently lose their covered quantity.
    planned_old_algo_ids = {
        str(action.get("algo_id") or "")
        for action in actions
        if str(action.get("algo_id") or "")
    }
    replacement_plan = [
        action for action in actions if str(action.get("action") or "") != "cancel"
    ]
    for order in preflight.get("protection_orders", []):
        if not isinstance(order, dict):
            continue
        algo_id = str(order.get("algo_id") or "")
        if not algo_id or algo_id in planned_old_algo_ids:
            continue
        contracts = str(order.get("contracts") or "")
        replacement_plan.append(
            {
                "action": "amend_size",
                "reason": "preserve_unchanged_protection_in_replacement_group",
                "inst_id": str(order.get("inst_id") or first_action.get("inst_id") or ""),
                "algo_id": algo_id,
                "position_side": side,
                "okx_position_side": str(order.get("okx_position_side") or "net"),
                "old_contracts": contracts,
                "new_contracts": contracts,
                "stop_loss_price": order.get("stop_loss_price"),
                "take_profit_price": order.get("take_profit_price"),
                "rollback": {"action": "cancel_created", "inst_id": order.get("inst_id")},
            }
        )

    # A previous attempt may have received an OKX create acknowledgement but
    # timed out before the new algo became visible. Reuse an exact existing
    # shape on the next retry instead of creating another live OCO.
    available_existing = [
        order
        for order in preflight.get("protection_orders", [])
        if isinstance(order, dict)
        and str(order.get("algo_id") or "")
    ]
    desired_position = next(
        (
            _safe_dict(position).get("contracts")
            for position in preflight.get("positions", [])
            if isinstance(position, dict)
            and normalize_trading_symbol(str(position.get("symbol") or "")) == symbol
            and str(position.get("side") or "").lower() == side
        ),
        0.0,
    )
    exact_existing = _existing_exact_coverage_subset(
        available_existing,
        desired_contracts=desired_position,
    )
    if exact_existing:
        replacement_plan = [
            {
                "action": "reuse_existing",
                "reason": "reuse_acknowledged_replacement_group",
                "inst_id": str(order.get("inst_id") or first_action.get("inst_id") or ""),
                "algo_id": str(order.get("algo_id") or ""),
                "position_side": side,
                "okx_position_side": str(order.get("okx_position_side") or "net"),
                "old_contracts": str(order.get("contracts") or "0"),
                "new_contracts": str(order.get("contracts") or "0"),
                "stop_loss_price": order.get("stop_loss_price"),
                "take_profit_price": order.get("take_profit_price"),
                "rollback": {"action": "none"},
            }
            for order in exact_existing
        ]
        available_existing = [
            order
            for order in available_existing
            if str(order.get("algo_id") or "")
            not in {str(item.get("algo_id") or "") for item in exact_existing}
        ]
    reused: list[dict[str, Any]] = []
    pending_create: list[dict[str, Any]] = []
    for action in replacement_plan:
        if str(action.get("action") or "") == "reuse_existing":
            reused.append(
                {
                    "action": action,
                    "created_algo_id": str(action.get("algo_id") or ""),
                    "create_response": {"already_present": True},
                    "reused_existing": True,
                }
            )
            continue
        match = next(
            (
                order
                for order in available_existing
                if _same_protection_shape(order, action)
            ),
            None,
        )
        if match is None:
            pending_create.append(action)
            continue
        available_existing.remove(match)
        reused.append(
            {
                "action": action,
                "created_algo_id": str(match.get("algo_id") or ""),
                "create_response": {"already_present": True},
                "reused_existing": True,
            }
        )

    required_position_fingerprint = str(
        expected_position_fingerprint
        or preflight_report.get("position_inventory_fingerprint")
        or ""
    )
    created: list[dict[str, Any]] = []
    created.extend(reused)
    try:
        for action in pending_create:
            stop_loss_price = float(action.get("stop_loss_price") or 0.0)
            take_profit_price = float(action.get("take_profit_price") or 0.0)
            if stop_loss_price <= 0 or take_profit_price <= 0:
                raise RuntimeError("Stuck-amend replacement requires the original OCO prices")

            create_response = await executor.create_position_protection_order(
                inst_id=str(action.get("inst_id") or ""),
                position_side=str(action.get("position_side") or ""),
                okx_position_side=str(action.get("okx_position_side") or "net"),
                contracts=float(action.get("new_contracts") or 0.0),
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
            )
            created_algo_id = _response_algo_id(create_response)
            if not created_algo_id or not _response_success(create_response):
                raise RuntimeError(
                    f"OKX rejected replacement protection for algo {action.get('algo_id')}"
                )
            created.append(
                {
                    "action": action,
                    "created_algo_id": created_algo_id,
                    "create_response": create_response,
                }
            )
    except Exception as exc:
        rollback_results: list[dict[str, Any]] = []
        for item in reversed(created):
            try:
                response = await executor.cancel_position_protection_order(
                    inst_id=str(item["action"].get("inst_id") or ""),
                    algo_id=str(item.get("created_algo_id") or ""),
                )
                rollback_results.append(
                    {"algo_id": item.get("created_algo_id"), "response": response}
                )
            except Exception as rollback_exc:  # pragma: no cover - exchange boundary
                rollback_results.append(
                    {"algo_id": item.get("created_algo_id"), "error": str(rollback_exc)}
                )
        error = RuntimeError(str(exc))
        error.replacement_action = {  # type: ignore[attr-defined]
            "created": created,
            "rollback_results": rollback_results,
        }
        raise error from exc

    created_algo_ids = {str(item["created_algo_id"]) for item in created}
    created_snapshot, create_verification_attempts = await _wait_for_protection_order_transition(
        executor,
        symbol=symbol,
        side=side,
        expected_position_fingerprint=required_position_fingerprint,
        present_algo_ids=created_algo_ids,
        absent_algo_ids=set(),
    )
    created_orders_by_id = {
        str(order.get("algo_id") or ""): order
        for order in created_snapshot.get("protection_orders", [])
        if isinstance(order, dict)
    }
    created_group_matches = bool(
        created_snapshot["report"].get("position_inventory_fingerprint")
        == required_position_fingerprint
        and all(
            item["created_algo_id"] in created_orders_by_id
            and _same_number(
                created_orders_by_id[item["created_algo_id"]].get("contracts"),
                item["action"].get("new_contracts"),
            )
            and _same_number(
                created_orders_by_id[item["created_algo_id"]].get("stop_loss_price"),
                item["action"].get("stop_loss_price"),
            )
            and _same_number(
                created_orders_by_id[item["created_algo_id"]].get("take_profit_price"),
                item["action"].get("take_profit_price"),
            )
            for item in created
        )
    )
    if not created_group_matches:
        rollback_results: list[dict[str, Any]] = []
        for item in reversed(created):
            try:
                response = await executor.cancel_position_protection_order(
                    inst_id=str(item["action"].get("inst_id") or ""),
                    algo_id=str(item.get("created_algo_id") or ""),
                )
                rollback_results.append(
                    {"algo_id": item.get("created_algo_id"), "response": response}
                )
            except Exception as rollback_exc:  # pragma: no cover - exchange boundary
                rollback_results.append(
                    {"algo_id": item.get("created_algo_id"), "error": str(rollback_exc)}
                )
        error = RuntimeError(
            "Replacement protection group was acknowledged but not observed before stale cancel"
        )
        error.replacement_action = {  # type: ignore[attr-defined]
            "created": created,
            "rollback_results": rollback_results,
            "create_verification_attempts": create_verification_attempts,
            "created_snapshot": created_snapshot,
        }
        raise error

    cancel_results: list[dict[str, Any]] = []
    old_algo_ids = {
        str(order.get("algo_id") or "")
        for order in preflight.get("protection_orders", [])
        if isinstance(order, dict)
        and str(order.get("algo_id") or "")
        and str(order.get("algo_id") or "")
        not in {item["created_algo_id"] for item in reused}
    }
    inst_id = str(first_action.get("inst_id") or "")
    cancel_results_by_id: dict[str, dict[str, Any]] = {}
    for old_algo_id in sorted(old_algo_ids):
        try:
            cancel_response = await executor.cancel_position_protection_order(
                inst_id=inst_id,
                algo_id=old_algo_id,
            )
            if not _response_success(cancel_response):
                raise RuntimeError(f"OKX rejected stale protection cancel for algo {old_algo_id}")
        except Exception as exc:
            snapshot = await protection_integrity_snapshot(executor, symbol=symbol, side=side)
            observed_ids = {
                str(order.get("algo_id") or "")
                for order in snapshot.get("protection_orders", [])
                if isinstance(order, dict)
            }
            if (
                snapshot["report"].get("position_inventory_fingerprint")
                == required_position_fingerprint
                and old_algo_id not in observed_ids
                and created_algo_ids.issubset(observed_ids)
            ):
                cancel_response = {"already_absent": True, "error": str(exc)}
            else:
                error = RuntimeError(
                    "Stale protection group cancel failed after replacement creation; "
                    "replacement coverage was retained for safe retry"
                )
                error.replacement_action = {  # type: ignore[attr-defined]
                    "created": created,
                    "cancel_results": cancel_results,
                    "failed_old_algo_id": old_algo_id,
                    "error": str(exc),
                }
                raise error from exc
        cancel_results_by_id[old_algo_id] = cancel_response
        cancel_results.append({"algo_id": old_algo_id, "response": cancel_response})

    post_cancel, cancel_verification_attempts = await _wait_for_protection_order_transition(
        executor,
        symbol=symbol,
        side=side,
        expected_position_fingerprint=required_position_fingerprint,
        present_algo_ids=created_algo_ids,
        absent_algo_ids=old_algo_ids,
    )
    post_cancel_report = post_cancel["report"]
    post_cancel_ids = {
        str(order.get("algo_id") or "")
        for order in post_cancel.get("protection_orders", [])
        if isinstance(order, dict)
    }
    if (
        post_cancel_report.get("position_inventory_fingerprint") != required_position_fingerprint
        or old_algo_ids.intersection(post_cancel_ids)
        or not created_algo_ids.issubset(post_cancel_ids)
    ):
        raise RuntimeError("Replacement protection group transition was not observed")

    return [
        {
            "action": {**item["action"], "action": "replace_stuck_amend"},
            "created_algo_id": item["created_algo_id"],
            "create_response": item["create_response"],
            "create_verification_attempts": create_verification_attempts,
            "cancel_response": cancel_results_by_id.get(
                str(item["action"].get("algo_id") or ""),
                {"skipped": True, "reason": "new_delta_action_has_no_old_algo_id"},
            ),
            "cancel_verification_attempts": cancel_verification_attempts,
            "applied": True,
            "reused_existing": bool(item.get("reused_existing")),
        }
        for item in created
    ]


async def rebalance_current_position_protection(
    executor: Any,
    *,
    symbol: str,
    side: str,
    observation_window: str,
    missing_protection_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make active OCO coverage equal one current OKX position quantity."""

    normalized_side = str(side or "").lower()
    normalized_symbol = normalize_trading_symbol(symbol)
    generated_at = datetime.now(UTC).isoformat()
    provenance = {
        "source": "okx_native_position_and_algo_inventory",
        "observation_window": observation_window,
        "sample_count": 0,
        "generated_at": generated_at,
        "strategy_version": POSITION_PROTECTION_REBALANCE_VERSION,
        "fallback_reason": "",
    }
    if not normalized_symbol or normalized_side not in {"long", "short"}:
        return {
            "status": "not_applicable",
            "verified": True,
            "policy_provenance": provenance,
        }

    before = await protection_integrity_snapshot(
        executor,
        symbol=normalized_symbol,
        side=normalized_side,
        missing_protection_plan=missing_protection_plan,
    )
    before_report = before["report"]
    provenance["sample_count"] = int(before_report.get("position_count") or 0) + int(
        before_report.get("protection_order_count") or 0
    )
    base_report = {
        "status": "planned",
        "verified": False,
        "symbol": normalized_symbol,
        "side": normalized_side,
        "before": before_report,
        "policy_provenance": provenance,
    }
    if before_report.get("repair_ready") is not True:
        base_report["status"] = "blocked"
        raise PositionProtectionRebalanceError(
            "Post-exit protection inventory is incomplete and cannot be repaired exactly",
            base_report,
        )

    actions = list(before_report.get("repair_actions") or [])
    if not actions:
        return {
            **base_report,
            "status": "already_exact",
            "verified": True,
            "applied_actions": [],
            "after": before_report,
        }

    amend_error = ""
    fallback_reason = ""
    replacement_actions: list[dict[str, Any]] = []
    rollback_results: list[dict[str, Any]] = []
    expected_position_fingerprint = str(
        before_report.get("position_inventory_fingerprint") or ""
    )
    replacement_input_fingerprint = ""
    action_names = {str(action.get("action") or "") for action in actions}
    initial_group_replacement = {"amend_size", "create_delta"}.issubset(action_names)
    try:
        if initial_group_replacement:
            replacement_input_fingerprint = str(before_report.get("input_fingerprint") or "")
            fallback_reason = "mixed_protection_plan_create_before_cancel"
            replacement_actions = await replace_stuck_protection_amendments(
                executor,
                actions,
                expected_position_fingerprint=expected_position_fingerprint,
                expected_input_fingerprint=replacement_input_fingerprint,
            )
            applied_actions = []
        else:
            applied_actions = await apply_protection_repair_actions(executor, actions)
    except Exception as exc:
        if not initial_group_replacement and "51513" in str(exc) and _amend_actions(actions):
            applied_actions = list(getattr(exc, "applied_actions", []))
            rollback_results = list(getattr(exc, "rollback_results", []))
            amend_error = str(exc)
            fallback_reason = "okx_amend_queue_saturated"
        else:
            base_report.update(
                {
                    "status": (
                        "replacement_failed" if initial_group_replacement else "apply_failed"
                    ),
                    "applied_actions": getattr(exc, "applied_actions", []),
                    "rollback_results": getattr(exc, "rollback_results", []),
                    "fallback_reason": fallback_reason,
                    "replacement_input_fingerprint": replacement_input_fingerprint,
                    "replacement_action": getattr(exc, "replacement_action", {}),
                    "error": str(exc),
                }
            )
            raise PositionProtectionRebalanceError(
                (
                    "Protection group replacement failed with retained coverage evidence"
                    if initial_group_replacement
                    else "Post-exit protection resize failed and rollback evidence was recorded"
                ),
                base_report,
            ) from exc

    after, amend_verification_attempts = await _wait_for_protection_verification(
        executor,
        symbol=normalized_symbol,
        side=normalized_side,
        expected_position_fingerprint=expected_position_fingerprint,
    )
    after_report = after["report"]
    verified = _coverage_is_exact(before_report, after_report)
    replacement_verification_attempts = 0

    if (
        not verified
        and _amend_actions(actions)
        and after_report.get("position_inventory_fingerprint") == expected_position_fingerprint
    ):
        try:
            fresh_actions = _fresh_replacement_actions(
                after,
                expected_position_fingerprint=expected_position_fingerprint,
            )
            replacement_input_fingerprint = str(after_report.get("input_fingerprint") or "")
            if not fallback_reason:
                fallback_reason = "okx_amend_acknowledged_but_not_observed"
            replacement_actions = await replace_stuck_protection_amendments(
                executor,
                fresh_actions,
                expected_position_fingerprint=expected_position_fingerprint,
                expected_input_fingerprint=replacement_input_fingerprint,
            )
        except Exception as replacement_exc:
            base_report.update(
                {
                    "status": "replacement_failed",
                    "applied_actions": applied_actions,
                    "rollback_results": rollback_results,
                    "amend_error": amend_error,
                    "amend_verification_attempts": amend_verification_attempts,
                    "fallback_reason": fallback_reason,
                    "replacement_input_fingerprint": replacement_input_fingerprint,
                    "error": str(replacement_exc),
                    "replacement_action": getattr(replacement_exc, "replacement_action", {}),
                    "after": after_report,
                }
            )
            raise PositionProtectionRebalanceError(
                "Stuck protection amend replacement failed with rollback evidence",
                base_report,
            ) from replacement_exc

        after, replacement_verification_attempts = await _wait_for_protection_verification(
            executor,
            symbol=normalized_symbol,
            side=normalized_side,
            expected_position_fingerprint=expected_position_fingerprint,
        )
        after_report = after["report"]
        verified = _coverage_is_exact(before_report, after_report)

    positions_unchanged = bool(
        expected_position_fingerprint == after_report.get("position_inventory_fingerprint")
    )
    final_report = {
        **base_report,
        "status": "repaired" if verified else "verification_failed",
        "verified": verified,
        "positions_unchanged": positions_unchanged,
        "applied_actions": [*applied_actions, *replacement_actions],
        "rollback_results": rollback_results,
        "amend_error": amend_error,
        "amend_verification_attempts": amend_verification_attempts,
        "fallback_reason": fallback_reason,
        "replacement_input_fingerprint": replacement_input_fingerprint,
        "replacement_verification_attempts": replacement_verification_attempts,
        "after": after_report,
    }
    if not verified:
        raise PositionProtectionRebalanceError(
            "Post-exit protection coverage did not verify against the same position inventory",
            final_report,
        )
    return final_report


async def rebalance_position_protection_after_exit(
    executor: Any,
    decision: DecisionOutput,
) -> dict[str, Any]:
    """Resize protection immediately after a locally executed exit."""

    return await rebalance_current_position_protection(
        executor,
        symbol=decision.symbol,
        side=_target_side(decision),
        observation_window="immediate_post_exit_exchange_state",
    )
