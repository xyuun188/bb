from types import SimpleNamespace

from services.okx_position_ledger_view import build_okx_position_ledger_groups


def test_explicit_superseded_residual_is_hidden_from_history_ledger() -> None:
    residual = SimpleNamespace(
        id=10,
        model_name="okx_authoritative_sync",
        execution_mode="paper",
        symbol="INJ/USDT",
        side="long",
        quantity=12.2,
        entry_price=5.0,
        current_price=5.1,
        leverage=2.0,
        unrealized_pnl=0.0,
        realized_pnl=0.0,
        is_open=False,
        okx_inst_id="INJ-USDT-SWAP",
        okx_pos_id="inj-pos-1",
        entry_exchange_order_id="inj-entry-a",
        close_exchange_order_id=None,
        settlement_status="settlement_exception",
        settlement_source="okx_position_history_settlement",
        settlement_raw={
            "reason": "duplicate_local_open_position_for_same_okx_pos_id",
            "canonical_position_id": 11,
        },
        created_at=None,
        closed_at=None,
    )

    assert build_okx_position_ledger_groups([residual], []) == []
