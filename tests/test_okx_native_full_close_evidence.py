from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.okx_native_full_close_evidence import (
    NATIVE_FULL_CLOSE_STATE_TRANSITION_EVIDENCE_VERSION,
    NATIVE_FULL_CLOSE_ZERO_POSITION_EVIDENCE_VERSION,
    build_native_full_close_state_transition_evidence,
    build_native_full_close_zero_position_evidence,
    native_full_close_zero_position_evidence,
)


def _facts() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, dict[str, str]]:
    closed_at = datetime(2026, 8, 19, 15, 30, 18, tzinfo=UTC)
    position = SimpleNamespace(
        id=6202,
        is_open=False,
        symbol="BTC/USDT",
        quantity=0.0018,
        closed_at=closed_at,
        okx_inst_id="BTC-USDT-SWAP",
        okx_pos_id="3847030100147798017",
        settlement_raw={},
    )
    close_order = SimpleNamespace(
        id=7351,
        decision_id=364072,
        exchange_order_id=None,
        quantity=0.0018,
    )
    decision = SimpleNamespace(
        id=364072,
        raw_llm_response={
            "execution_result": {
                "raw_response": {
                    "code": "0",
                    "data": [{"instId": "BTC-USDT-SWAP", "posSide": "net"}],
                    "request_params": {"instId": "BTC-USDT-SWAP"},
                    "okx_native_close_position": True,
                    "base_quantity": 0.0018,
                    "contract_size": 0.01,
                    "position_contracts_before": 0.18,
                    "requested_exit_contracts": 0.18,
                    "filled_contracts": 0.18,
                    "position_contracts_after": 0.0,
                    "remaining_contracts": 0.0,
                }
            }
        },
    )
    current_created_at = closed_at + timedelta(seconds=97)
    current_updated_at = closed_at + timedelta(hours=5)
    current_row = {
        "instId": "BTC-USDT-SWAP",
        "posId": "3847030100147798017",
        "posSide": "net",
        "pos": "0",
        "tradeId": "4340192137",
        "cTime": str(int(current_created_at.timestamp() * 1000)),
        "uTime": str(int(current_updated_at.timestamp() * 1000)),
    }
    return position, close_order, decision, current_row


def test_native_full_close_zero_position_evidence_requires_complete_identity() -> None:
    position, close_order, decision, current_row = _facts()

    evidence = build_native_full_close_zero_position_evidence(
        position=position,
        close_order=close_order,
        decision=decision,
        current_position_row=current_row,
        verified_at=datetime(2026, 8, 20, tzinfo=UTC),
    )

    assert evidence is not None
    assert evidence["version"] == NATIVE_FULL_CLOSE_ZERO_POSITION_EVIDENCE_VERSION
    assert evidence["identity_authoritative"] is True
    assert evidence["economics_authoritative"] is False
    assert evidence["training_eligible"] is False
    assert evidence["okx_trade_id"] == "4340192137"
    assert evidence["receipt"]["position_contracts_after"] == 0.0

    position.settlement_raw = {"native_full_close_zero_position_evidence": evidence}
    assert native_full_close_zero_position_evidence(position) == evidence


def test_native_full_close_zero_position_evidence_rejects_nonzero_or_wrong_identity() -> None:
    position, close_order, decision, current_row = _facts()
    current_row["pos"] = "0.01"
    assert (
        build_native_full_close_zero_position_evidence(
            position=position,
            close_order=close_order,
            decision=decision,
            current_position_row=current_row,
        )
        is None
    )
    current_row["pos"] = "0"
    current_row["posId"] = "another-position"
    assert (
        build_native_full_close_zero_position_evidence(
            position=position,
            close_order=close_order,
            decision=decision,
            current_position_row=current_row,
        )
        is None
    )


def test_native_full_close_state_transition_requires_successful_current_query() -> None:
    position, close_order, decision, _current_row = _facts()
    close_order.filled_at = position.closed_at
    close_order.created_at = position.closed_at

    evidence = build_native_full_close_state_transition_evidence(
        position=position,
        close_order=close_order,
        decision=decision,
        current_position_rows=[],
        current_position_query_succeeded=True,
        verified_at=datetime(2026, 8, 20, tzinfo=UTC),
    )

    assert evidence is not None
    assert evidence["version"] == NATIVE_FULL_CLOSE_STATE_TRANSITION_EVIDENCE_VERSION
    assert evidence["identity_authoritative"] is True
    assert evidence["economics_authoritative"] is False
    assert evidence["training_eligible"] is False
    assert evidence["current_position_matching_row_count"] == 0
