from services.protection_order_integrity import audit_protection_order_integrity


def _position(symbol: str, side: str, contracts: str) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "contracts": contracts,
        "info": {"instId": f"{symbol.replace('/', '-')}-SWAP", "pos": contracts},
    }


def _protection(
    symbol: str,
    side: str,
    algo_id: str,
    contracts: str,
    *,
    created_at_ms: int,
) -> dict:
    return {
        "symbol": symbol,
        "position_side": side,
        "algo_id": algo_id,
        "contracts": contracts,
        "reduce_only": True,
        "state": "live",
        "order_type": "oco",
        "stop_loss_price": 110.0 if side == "long" else 90.0,
        "take_profit_price": 130.0 if side == "long" else 70.0,
        "created_at_ms": created_at_ms,
    }


def test_split_protection_is_not_duplicate_when_quantity_coverage_is_exact() -> None:
    report = audit_protection_order_integrity(
        [_position("IRYS/USDT", "short", "13")],
        [
            _protection("IRYS/USDT", "short", "algo-1", "5", created_at_ms=1),
            _protection("IRYS/USDT", "short", "algo-2", "8", created_at_ms=2),
        ],
        [],
        {"IRYS-USDT-SWAP": {"lotSz": "1", "minSz": "1"}},
        pending_snapshot_complete=True,
    )

    assert report["split_coverage_keys"] == [["IRYS/USDT", "short"]]
    assert report["coverage_mismatches"] == []
    assert report["repair_actions"] == []
    assert report["repair_ready"] is True


def test_oversized_split_protection_is_resized_to_exact_current_contracts() -> None:
    report = audit_protection_order_integrity(
        [_position("IRYS/USDT", "short", "13")],
        [
            _protection("IRYS/USDT", "short", "algo-old", "88", created_at_ms=1),
            _protection("IRYS/USDT", "short", "algo-new", "57", created_at_ms=2),
        ],
        [],
        {"IRYS-USDT-SWAP": {"lotSz": "1", "minSz": "1"}},
        pending_snapshot_complete=True,
    )

    assert report["coverage_mismatches"] == [
        {
            "symbol": "IRYS/USDT",
            "side": "short",
            "position_contracts": "13",
            "protection_contracts": "145",
            "order_count": 2,
        }
    ]
    amendments = [
        action for action in report["repair_actions"] if action["action"] == "amend_size"
    ]
    assert sum(float(action["new_contracts"]) for action in amendments) == 13.0
    assert {action["algo_id"] for action in amendments} == {"algo-old", "algo-new"}
    assert all(action["rollback"]["action"] == "amend_size" for action in amendments)
    assert len(report["rollback_actions"]) == len(report["repair_actions"])
    assert report["repair_ready"] is True


def test_orphan_protection_requires_complete_pending_snapshot_before_cancel() -> None:
    orphan = _protection("SOL/USDT", "long", "algo-sol", "2.43", created_at_ms=1)
    blocked = audit_protection_order_integrity(
        [],
        [orphan],
        [],
        {},
        pending_snapshot_complete=False,
    )
    ready = audit_protection_order_integrity(
        [],
        [orphan],
        [],
        {},
        pending_snapshot_complete=True,
    )

    assert blocked["repair_actions"] == []
    assert blocked["repair_ready"] is False
    assert ready["repair_actions"][0]["action"] == "cancel"
    assert ready["repair_actions"][0]["reason"] == "no_position_and_no_pending_entry"
    assert ready["rollback_actions"][0]["action"] == "manual_recreate_from_backup"
    assert ready["rollback_actions"][0]["stop_loss_price"] == 110.0
    assert ready["repair_ready"] is True


def test_pending_entry_prevents_orphan_protection_cancellation() -> None:
    report = audit_protection_order_integrity(
        [],
        [_protection("SOL/USDT", "long", "algo-sol", "2.43", created_at_ms=1)],
        [
            {
                "symbol": "SOL/USDT",
                "side": "buy",
                "reduceOnly": False,
                "info": {"instId": "SOL-USDT-SWAP"},
            }
        ],
        {},
        pending_snapshot_complete=True,
    )

    assert report["repair_actions"] == []
    assert report["repair_ready"] is False
    assert "orphan_has_pending_entry:SOL/USDT:long" in report["repair_blockers"]


def test_existing_algo_precision_can_match_residual_position_below_order_lot_step() -> None:
    report = audit_protection_order_integrity(
        [_position("ETC/USDT", "short", "0.37")],
        [_protection("ETC/USDT", "short", "algo-etc", "0.435", created_at_ms=1)],
        [],
        {"ETC-USDT-SWAP": {"lotSz": "0.1", "minSz": "0.1"}},
        pending_snapshot_complete=True,
    )

    assert report["repair_ready"] is True
    assert report["repair_blockers"] == []
    assert report["repair_actions"] == [
        {
            "action": "amend_size",
            "reason": "match_current_position_contract_coverage",
            "inst_id": "ETC-USDT-SWAP",
            "algo_id": "algo-etc",
            "old_contracts": "0.435",
            "new_contracts": "0.37",
            "rollback": {
                "action": "amend_size",
                "inst_id": "ETC-USDT-SWAP",
                "algo_id": "algo-etc",
                "new_contracts": "0.435",
            },
        }
    ]
