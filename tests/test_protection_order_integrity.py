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


def test_repair_fingerprint_ignores_observation_timestamp_only() -> None:
    position = _position("SAND/USDT", "long", "10")
    first_order = _protection("SAND/USDT", "long", "algo-sand", "1", created_at_ms=1)
    second_order = dict(first_order)
    first_order["updated_at_ms"] = 100
    second_order["updated_at_ms"] = 200
    specs = {"SAND-USDT-SWAP": {"lotSz": "1", "minSz": "1"}}

    first = audit_protection_order_integrity(
        [position], [first_order], [], specs, pending_snapshot_complete=True
    )
    second = audit_protection_order_integrity(
        [position], [second_order], [], specs, pending_snapshot_complete=True
    )

    assert first["input_fingerprint"] == second["input_fingerprint"]


def test_repair_fingerprint_changes_with_action_relevant_order_facts() -> None:
    position = _position("SAND/USDT", "long", "10")
    baseline = _protection("SAND/USDT", "long", "algo-sand", "1", created_at_ms=1)
    changed_quantity = {**baseline, "contracts": "2"}
    changed_stop = {**baseline, "stop_loss_price": 109.0}
    specs = {"SAND-USDT-SWAP": {"lotSz": "1", "minSz": "1"}}

    reports = [
        audit_protection_order_integrity(
            [position], [order], [], specs, pending_snapshot_complete=True
        )
        for order in (baseline, changed_quantity, changed_stop)
    ]

    assert len({report["input_fingerprint"] for report in reports}) == 3


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
    amendments = [action for action in report["repair_actions"] if action["action"] == "amend_size"]
    assert sum(float(action["new_contracts"]) for action in amendments) == 13.0
    assert {action["algo_id"] for action in amendments} == {"algo-old", "algo-new"}
    assert all(action["rollback"]["action"] == "amend_size" for action in amendments)
    assert len(report["rollback_actions"]) == len(report["repair_actions"])
    assert report["repair_ready"] is True


def test_undersized_protection_adds_delta_without_exposing_existing_coverage() -> None:
    report = audit_protection_order_integrity(
        [_position("SAND/USDT", "long", "10")],
        [_protection("SAND/USDT", "long", "algo-sand", "1", created_at_ms=1)],
        [],
        {"SAND-USDT-SWAP": {"lotSz": "1", "minSz": "1"}},
        pending_snapshot_complete=True,
    )

    assert report["repair_ready"] is True
    assert report["repair_actions"] == [
        {
            "action": "create_delta",
            "reason": "cover_positive_position_residual_without_increasing_existing_oco",
            "inst_id": "SAND-USDT-SWAP",
            "position_side": "long",
            "okx_position_side": "net",
            "old_contracts": "0",
            "new_contracts": "9",
            "stop_loss_price": 110.0,
            "take_profit_price": 130.0,
            "rollback": {
                "action": "cancel_created",
                "inst_id": "SAND-USDT-SWAP",
                "algo_id": None,
            },
        }
    ]


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
            "position_side": "short",
            "okx_position_side": "net",
            "old_contracts": "0.435",
            "new_contracts": "0.37",
            "stop_loss_price": 90.0,
            "take_profit_price": 70.0,
            "rollback": {
                "action": "amend_size",
                "inst_id": "ETC-USDT-SWAP",
                "algo_id": "algo-etc",
                "new_contracts": "0.435",
            },
        }
    ]


def test_missing_protection_can_be_rebuilt_only_from_complete_dynamic_plan() -> None:
    report = audit_protection_order_integrity(
        [_position("YB/USDT", "short", "760")],
        [],
        [],
        {"YB-USDT-SWAP": {"lotSz": "1", "minSz": "1"}},
        pending_snapshot_complete=True,
        missing_protection_plans={
            ("YB/USDT", "short"): {
                "stop_loss_price": 0.047,
                "take_profit_price": 0.043,
                "okx_position_side": "short",
            }
        },
    )

    assert report["repair_ready"] is True
    assert report["planned_missing_keys"] == [["YB/USDT", "short"]]
    assert report["repair_actions"] == [
        {
            "action": "create_delta",
            "reason": "restore_missing_dynamic_position_protection",
            "inst_id": "YB-USDT-SWAP",
            "position_side": "short",
            "okx_position_side": "short",
            "old_contracts": "0",
            "new_contracts": "760",
            "stop_loss_price": "0.047",
            "take_profit_price": "0.043",
            "rollback": {
                "action": "cancel_created",
                "inst_id": "YB-USDT-SWAP",
                "algo_id": None,
            },
        }
    ]


def test_missing_protection_stays_blocked_while_entry_residual_is_active() -> None:
    report = audit_protection_order_integrity(
        [_position("YB/USDT", "short", "760")],
        [],
        [{"symbol": "YB/USDT", "side": "sell", "reduceOnly": False}],
        {"YB-USDT-SWAP": {"lotSz": "1", "minSz": "1"}},
        pending_snapshot_complete=True,
        missing_protection_plans={
            ("YB/USDT", "short"): {
                "stop_loss_price": 0.047,
                "take_profit_price": 0.043,
            }
        },
    )

    assert report["repair_ready"] is False
    assert "missing_protection_has_pending_entry:YB/USDT:short" in report["repair_blockers"]
