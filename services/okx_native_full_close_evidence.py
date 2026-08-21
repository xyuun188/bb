"""Evidence contract for OKX native full closes without an order id."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core.symbols import okx_inst_id_from_symbol

NATIVE_FULL_CLOSE_ZERO_POSITION_EVIDENCE_VERSION = (
    "2026-08-20.okx-native-full-close-zero-position.v1"
)
NATIVE_FULL_CLOSE_ZERO_POSITION_SOURCE = "okx_native_full_close_zero_position_verified"
NATIVE_FULL_CLOSE_STATE_TRANSITION_EVIDENCE_VERSION = (
    "2026-08-20.okx-native-full-close-state-transition.v1"
)
NATIVE_FULL_CLOSE_STATE_TRANSITION_SOURCE = (
    "okx_native_full_close_state_transition_verified"
)
NATIVE_FULL_CLOSE_ZERO_POSITION_MAX_LAG_SECONDS = 15 * 60
NATIVE_FULL_CLOSE_ZERO_POSITION_EARLY_TOLERANCE_SECONDS = 2 * 60
NATIVE_FULL_CLOSE_QUANTITY_TOLERANCE_RATIO = 0.02
NATIVE_FULL_CLOSE_LOCAL_TRANSITION_MAX_LAG_SECONDS = 2 * 60


def build_native_full_close_zero_position_evidence(
    *,
    position: Any,
    close_order: Any,
    decision: Any,
    current_position_row: dict[str, Any],
    verified_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Build evidence only when OKX and the local close receipt agree exactly."""

    receipt_evidence = _native_close_receipt_evidence(
        position=position,
        close_order=close_order,
        decision=decision,
    )
    if receipt_evidence is None:
        return None
    position_id = int(receipt_evidence["position_id"])
    order_id = int(receipt_evidence["close_order_id"])
    decision_id = int(receipt_evidence["decision_id"])
    okx_pos_id = str(receipt_evidence["okx_pos_id"])
    inst_id = str(receipt_evidence["inst_id"])
    closed_at = receipt_evidence["closed_at"]
    if _text(current_position_row.get("posId")) != okx_pos_id:
        return None
    if _text(current_position_row.get("instId")).upper() != inst_id:
        return None
    if abs(_float(current_position_row.get("pos"))) > 1e-12:
        return None
    trade_id = _text(current_position_row.get("tradeId"))
    current_updated_at = _datetime_from_ms(current_position_row.get("uTime"))
    current_created_at = _datetime_from_ms(current_position_row.get("cTime"))
    if not trade_id or closed_at is None:
        return None
    close_identity_matches = [
        (candidate_at, (candidate_at - closed_at).total_seconds())
        for candidate_at in (current_created_at, current_updated_at)
        if candidate_at is not None
        and -NATIVE_FULL_CLOSE_ZERO_POSITION_EARLY_TOLERANCE_SECONDS
        <= (candidate_at - closed_at).total_seconds()
        <= NATIVE_FULL_CLOSE_ZERO_POSITION_MAX_LAG_SECONDS
    ]
    if not close_identity_matches:
        return None
    identity_at, close_lag = min(close_identity_matches, key=lambda item: abs(item[1]))

    verified = _aware(verified_at) or datetime.now(UTC)
    return {
        "version": NATIVE_FULL_CLOSE_ZERO_POSITION_EVIDENCE_VERSION,
        "verified": True,
        "verified_at": verified.isoformat(),
        "source": NATIVE_FULL_CLOSE_ZERO_POSITION_SOURCE,
        "verification_kind": "exact_okx_pos_id_zero",
        "identity_authoritative": True,
        "economics_authoritative": False,
        "training_eligible": False,
        "reason": "okx_native_close_has_no_order_id_but_exact_pos_id_is_zero",
        "position_id": position_id,
        "close_order_id": order_id,
        "decision_id": decision_id,
        "okx_pos_id": okx_pos_id,
        "okx_trade_id": trade_id,
        "inst_id": inst_id,
        "pos_side": _text(current_position_row.get("posSide")).lower(),
        "current_position_contracts": 0.0,
        "okx_position_identity_at": identity_at.isoformat(),
        "okx_position_created_at": _iso(current_created_at),
        "okx_position_updated_at": _iso(current_updated_at),
        "local_position_closed_at": closed_at.isoformat(),
        "close_lag_seconds": round(close_lag, 3),
        "position_quantity": receipt_evidence["position_quantity"],
        "receipt": receipt_evidence["receipt"],
    }


def build_native_full_close_state_transition_evidence(
    *,
    position: Any,
    close_order: Any,
    decision: Any,
    current_position_rows: list[dict[str, Any]],
    current_position_query_succeeded: bool,
    verified_at: datetime | None = None,
) -> dict[str, Any] | None:
    """Confirm a native close from its OKX receipt and current instrument absence.

    This contract proves only that the exact local lifecycle was flattened. It
    deliberately does not invent an OKX order id or make local PnL trainable.
    """

    if current_position_query_succeeded is not True:
        return None
    receipt_evidence = _native_close_receipt_evidence(
        position=position,
        close_order=close_order,
        decision=decision,
    )
    if receipt_evidence is None:
        return None
    inst_id = str(receipt_evidence["inst_id"])
    rows = [row for row in current_position_rows if isinstance(row, dict)]
    matching_rows = [
        row for row in rows if _text(row.get("instId")).upper() == inst_id
    ]
    if matching_rows:
        return None

    closed_at = receipt_evidence["closed_at"]
    order_time = _aware(
        getattr(close_order, "filled_at", None)
        or getattr(close_order, "created_at", None)
    )
    if closed_at is None or order_time is None:
        return None
    order_lag = abs((order_time - closed_at).total_seconds())
    if order_lag > NATIVE_FULL_CLOSE_LOCAL_TRANSITION_MAX_LAG_SECONDS:
        return None

    receipt = receipt_evidence["receipt"]
    if _text(receipt.get("snapshot_error")):
        return None
    verified = _aware(verified_at) or datetime.now(UTC)
    return {
        "version": NATIVE_FULL_CLOSE_STATE_TRANSITION_EVIDENCE_VERSION,
        "verified": True,
        "verified_at": verified.isoformat(),
        "source": NATIVE_FULL_CLOSE_STATE_TRANSITION_SOURCE,
        "verification_kind": "okx_receipt_transition_and_current_instrument_absence",
        "identity_authoritative": True,
        "economics_authoritative": False,
        "training_eligible": False,
        "reason": "okx_native_close_receipt_flattens_position_and_instrument_is_absent",
        "position_id": receipt_evidence["position_id"],
        "close_order_id": receipt_evidence["close_order_id"],
        "decision_id": receipt_evidence["decision_id"],
        "okx_pos_id": receipt_evidence["okx_pos_id"],
        "okx_trade_id": "",
        "inst_id": inst_id,
        "current_position_contracts": 0.0,
        "current_position_query_succeeded": True,
        "current_position_matching_row_count": 0,
        "local_position_closed_at": closed_at.isoformat(),
        "local_close_order_at": order_time.isoformat(),
        "close_order_lag_seconds": round(order_lag, 3),
        "position_quantity": receipt_evidence["position_quantity"],
        "receipt": receipt,
    }


def native_full_close_identity_evidence(position: Any) -> dict[str, Any] | None:
    """Return either supported identity-only native-close evidence contract."""

    raw = getattr(position, "settlement_raw", None)
    raw = raw if isinstance(raw, dict) else {}
    legacy = native_full_close_zero_position_evidence(position)
    if legacy is not None:
        return legacy
    evidence = raw.get("native_full_close_identity_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    if (
        evidence.get("version") != NATIVE_FULL_CLOSE_STATE_TRANSITION_EVIDENCE_VERSION
        or evidence.get("verified") is not True
        or evidence.get("identity_authoritative") is not True
        or evidence.get("economics_authoritative") is not False
        or evidence.get("training_eligible") is not False
        or evidence.get("current_position_query_succeeded") is not True
        or _positive_int(evidence.get("current_position_matching_row_count")) != 0
        or abs(_float(evidence.get("current_position_contracts"))) > 1e-12
    ):
        return None
    if not _evidence_matches_position(evidence, position):
        return None
    receipt = evidence.get("receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    if (
        _text(receipt.get("response_code")) != "0"
        or abs(_float(receipt.get("position_contracts_before"))) <= 0
        or abs(_float(receipt.get("position_contracts_after"))) > 1e-12
        or abs(_float(receipt.get("remaining_contracts"))) > 1e-12
    ):
        return None
    return dict(evidence)


def native_full_close_zero_position_evidence(position: Any) -> dict[str, Any] | None:
    """Return persisted evidence only when it still matches the position identity."""

    raw = getattr(position, "settlement_raw", None)
    raw = raw if isinstance(raw, dict) else {}
    evidence = raw.get("native_full_close_zero_position_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    if (
        evidence.get("version") != NATIVE_FULL_CLOSE_ZERO_POSITION_EVIDENCE_VERSION
        or evidence.get("verified") is not True
        or evidence.get("identity_authoritative") is not True
        or evidence.get("economics_authoritative") is not False
        or evidence.get("training_eligible") is not False
        or abs(_float(evidence.get("current_position_contracts"))) > 1e-12
    ):
        return None
    if not _evidence_matches_position(evidence, position):
        return None
    if not _text(evidence.get("okx_trade_id")):
        return None
    return dict(evidence)


def _native_close_receipt_evidence(
    *,
    position: Any,
    close_order: Any,
    decision: Any,
) -> dict[str, Any] | None:
    if bool(getattr(position, "is_open", True)):
        return None
    position_id = _positive_int(getattr(position, "id", None))
    order_id = _positive_int(getattr(close_order, "id", None))
    decision_id = _positive_int(getattr(decision, "id", None))
    if not position_id or not order_id or not decision_id:
        return None
    if _positive_int(getattr(close_order, "decision_id", None)) != decision_id:
        return None
    exchange_order_id = _text(getattr(close_order, "exchange_order_id", None))
    if exchange_order_id and "okx_native_full_close" not in exchange_order_id.lower():
        return None

    okx_pos_id = _text(getattr(position, "okx_pos_id", None))
    inst_id = _expected_inst_id(position)
    closed_at = _aware(getattr(position, "closed_at", None))
    if not okx_pos_id or not inst_id or closed_at is None:
        return None

    raw = getattr(decision, "raw_llm_response", None)
    raw = raw if isinstance(raw, dict) else {}
    execution = raw.get("execution_result")
    execution = execution if isinstance(execution, dict) else {}
    receipt = execution.get("raw_response")
    receipt = receipt if isinstance(receipt, dict) else {}
    response_rows = receipt.get("data")
    response_rows = response_rows if isinstance(response_rows, list) else []
    response_inst_ids = {
        _text(row.get("instId")).upper()
        for row in response_rows
        if isinstance(row, dict) and _text(row.get("instId"))
    }
    request = receipt.get("request_params")
    request = request if isinstance(request, dict) else {}
    if (
        _text(receipt.get("code")) != "0"
        or receipt.get("okx_native_close_position") is not True
        or _text(request.get("instId")).upper() != inst_id
        or inst_id not in response_inst_ids
    ):
        return None

    quantity = abs(_float(getattr(position, "quantity", None)))
    order_quantity = abs(_float(getattr(close_order, "quantity", None)))
    base_quantity = abs(_float(receipt.get("base_quantity")))
    contracts_before = abs(_float(receipt.get("position_contracts_before")))
    requested_contracts = abs(_float(receipt.get("requested_exit_contracts")))
    filled_contracts = abs(_float(receipt.get("filled_contracts")))
    contracts_after = abs(_float(receipt.get("position_contracts_after")))
    remaining_contracts = abs(_float(receipt.get("remaining_contracts")))
    if (
        quantity <= 0
        or contracts_before <= 0
        or not _close_enough(order_quantity, quantity)
        or not _close_enough(base_quantity, quantity)
        or not _close_enough(requested_contracts, contracts_before)
        or not _close_enough(filled_contracts, contracts_before)
        or contracts_after > 1e-12
        or remaining_contracts > 1e-12
    ):
        return None
    return {
        "position_id": position_id,
        "close_order_id": order_id,
        "decision_id": decision_id,
        "okx_pos_id": okx_pos_id,
        "inst_id": inst_id,
        "closed_at": closed_at,
        "position_quantity": quantity,
        "receipt": {
            "response_code": "0",
            "inst_id": inst_id,
            "base_quantity": base_quantity,
            "contract_size": abs(_float(receipt.get("contract_size"))),
            "position_contracts_before": contracts_before,
            "requested_exit_contracts": requested_contracts,
            "filled_contracts": filled_contracts,
            "position_contracts_after": contracts_after,
            "remaining_contracts": remaining_contracts,
            "snapshot_error": _text(receipt.get("snapshot_error")),
        },
    }


def _evidence_matches_position(evidence: dict[str, Any], position: Any) -> bool:
    if _positive_int(evidence.get("position_id")) != _positive_int(
        getattr(position, "id", None)
    ):
        return False
    if _text(evidence.get("okx_pos_id")) != _text(getattr(position, "okx_pos_id", None)):
        return False
    expected_inst_id = _expected_inst_id(position)
    return bool(
        expected_inst_id
        and _text(evidence.get("inst_id")).upper() == expected_inst_id
    )


def _expected_inst_id(position: Any) -> str:
    inst_id = _text(getattr(position, "okx_inst_id", None)).upper()
    if not inst_id:
        inst_id = _text(okx_inst_id_from_symbol(getattr(position, "symbol", None))).upper()
    return inst_id


def _close_enough(left: float, right: float) -> bool:
    tolerance = max(abs(left), abs(right), 1e-12) * NATIVE_FULL_CLOSE_QUANTITY_TOLERANCE_RATIO
    return abs(left - right) <= tolerance


def _positive_int(value: Any) -> int:
    try:
        result = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return result if result > 0 else 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _text(value: Any) -> str:
    return str(value or "").strip()


def _aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _datetime_from_ms(value: Any) -> datetime | None:
    timestamp = _float(value)
    if timestamp <= 0:
        return None
    return datetime.fromtimestamp(timestamp / 1000.0, UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
