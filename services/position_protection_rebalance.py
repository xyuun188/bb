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


def _same_number(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
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
    attempts: int = PROTECTION_VERIFY_ATTEMPTS,
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
    if not _amend_actions(actions):
        raise RuntimeError("Protection replacement requires a fresh amend-only plan")
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

    if not _amend_actions(actions):
        raise RuntimeError("Stuck-amend replacement only accepts amend_size actions")

    first_action = actions[0]
    symbol = normalize_trading_symbol(str(first_action.get("inst_id") or ""))
    side = str(first_action.get("position_side") or "").lower()
    if any(
        normalize_trading_symbol(str(action.get("inst_id") or "")) != symbol
        or str(action.get("position_side") or "").lower() != side
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

    applied: list[dict[str, Any]] = []
    for action in actions:
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

        required_position_fingerprint = str(
            expected_position_fingerprint
            or preflight_report.get("position_inventory_fingerprint")
            or ""
        )
        created_snapshot, create_verification_attempts = (
            await _wait_for_protection_order_transition(
                executor,
                symbol=symbol,
                side=side,
                expected_position_fingerprint=required_position_fingerprint,
                present_algo_ids={created_algo_id},
                absent_algo_ids=set(),
            )
        )
        created_report = created_snapshot["report"]
        created_ids = {
            str(order.get("algo_id") or "")
            for order in created_snapshot.get("protection_orders", [])
            if isinstance(order, dict)
        }
        created_order = next(
            (
                order
                for order in created_snapshot.get("protection_orders", [])
                if isinstance(order, dict) and str(order.get("algo_id") or "") == created_algo_id
            ),
            None,
        )
        created_order_matches = bool(
            isinstance(created_order, dict)
            and _same_number(created_order.get("contracts"), action.get("new_contracts"))
            and _same_number(created_order.get("stop_loss_price"), action.get("stop_loss_price"))
            and _same_number(
                created_order.get("take_profit_price"), action.get("take_profit_price")
            )
        )
        if (
            created_report.get("position_inventory_fingerprint") != required_position_fingerprint
            or created_algo_id not in created_ids
            or not created_order_matches
        ):
            try:
                await executor.cancel_position_protection_order(
                    inst_id=str(action.get("inst_id") or ""),
                    algo_id=created_algo_id,
                )
            except Exception as rollback_exc:  # pragma: no cover - defensive exchange boundary
                raise RuntimeError(
                    "Replacement protection was acknowledged but not observed; "
                    f"created-order rollback failed: {rollback_exc}"
                ) from rollback_exc
            raise RuntimeError(
                "Replacement protection was acknowledged but not observed before stale cancel"
            )

        try:
            cancel_response = await executor.cancel_position_protection_order(
                inst_id=str(action.get("inst_id") or ""),
                algo_id=str(action.get("algo_id") or ""),
            )
            if not _response_success(cancel_response):
                raise RuntimeError(
                    f"OKX rejected stale protection cancel for algo {action.get('algo_id')}"
                )
        except Exception as exc:
            rollback_response: Any = None
            rollback_error = ""
            try:
                rollback_response = await executor.cancel_position_protection_order(
                    inst_id=str(action.get("inst_id") or ""),
                    algo_id=created_algo_id,
                )
            except Exception as rollback_exc:  # pragma: no cover - defensive exchange boundary
                rollback_error = str(rollback_exc)
            error = RuntimeError(
                "Stale protection cancel failed after replacement creation; "
                "the replacement cancel rollback was attempted"
            )
            error.replacement_action = {  # type: ignore[attr-defined]
                "action": action,
                "created_algo_id": created_algo_id,
                "create_response": create_response,
                "rollback_response": rollback_response,
                "rollback_error": rollback_error,
            }
            raise error from exc

        post_cancel, cancel_verification_attempts = await _wait_for_protection_order_transition(
            executor,
            symbol=symbol,
            side=side,
            expected_position_fingerprint=required_position_fingerprint,
            present_algo_ids={created_algo_id},
            absent_algo_ids={str(action.get("algo_id") or "")},
        )
        post_cancel_report = post_cancel["report"]
        post_cancel_ids = {
            str(order.get("algo_id") or "")
            for order in post_cancel.get("protection_orders", [])
            if isinstance(order, dict)
        }
        if (
            post_cancel_report.get("position_inventory_fingerprint")
            != required_position_fingerprint
            or str(action.get("algo_id") or "") in post_cancel_ids
            or created_algo_id not in post_cancel_ids
        ):
            raise RuntimeError(
                "Replacement protection transition was acknowledged but not observed"
            )

        applied.append(
            {
                "action": {**action, "action": "replace_stuck_amend"},
                "created_algo_id": created_algo_id,
                "create_response": create_response,
                "create_verification_attempts": create_verification_attempts,
                "cancel_response": cancel_response,
                "cancel_verification_attempts": cancel_verification_attempts,
                "applied": True,
            }
        )
    return applied


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
    try:
        applied_actions = await apply_protection_repair_actions(executor, actions)
    except Exception as exc:
        if "51513" in str(exc) and _amend_actions(actions):
            applied_actions = list(getattr(exc, "applied_actions", []))
            rollback_results = list(getattr(exc, "rollback_results", []))
            amend_error = str(exc)
            fallback_reason = "okx_amend_queue_saturated"
        else:
            base_report.update(
                {
                    "status": "apply_failed",
                    "applied_actions": getattr(exc, "applied_actions", []),
                    "rollback_results": getattr(exc, "rollback_results", []),
                    "error": str(exc),
                }
            )
            raise PositionProtectionRebalanceError(
                "Post-exit protection resize failed and rollback evidence was recorded",
                base_report,
            ) from exc

    expected_position_fingerprint = str(before_report.get("position_inventory_fingerprint") or "")
    after, amend_verification_attempts = await _wait_for_protection_verification(
        executor,
        symbol=normalized_symbol,
        side=normalized_side,
        expected_position_fingerprint=expected_position_fingerprint,
    )
    after_report = after["report"]
    verified = _coverage_is_exact(before_report, after_report)
    replacement_input_fingerprint = ""
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
