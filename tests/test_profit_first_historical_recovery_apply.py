from __future__ import annotations

import pytest

from scripts import apply_profit_first_historical_recovery_package as apply_cli
from services.profit_first_historical_recovery_apply import (
    APPROVAL_TOKEN,
    build_historical_recovery_apply_plan,
    merge_raw_patch,
    validate_apply_request,
)


def _package() -> dict:
    return {
        "report_type": "profit_first_historical_recovery_package",
        "status": "ready",
        "items": [
            {
                "item_type": "entry_decision_recovery",
                "decision_id": 9549,
                "proposed_raw_patch": {
                    "profit_first_trade_plan": {"decision_lane": "tiny_probe"}
                },
            },
            {
                "item_type": "exit_decision_recovery",
                "decision_id": 10391,
                "proposed_raw_patch": {
                    "profit_first_exit_reference": {
                        "missing_original_exit_plan_reference": True
                    }
                },
            },
            {
                "item_type": "okx_order_quantity_review",
                "decision_id": 9299,
                "order_id": 2678,
                "exchange_order_id": "3697115296869093376",
                "proposed_raw_patch": {},
            },
            {
                "item_type": "entry_decision_recovery",
                "decision_id": 7777,
                "proposed_raw_patch": {"profit_first_trade_plan": {"decision_lane": "shadow_only"}},
            },
        ],
    }


def test_apply_plan_only_allows_whitelisted_decision_raw_patches() -> None:
    plan = build_historical_recovery_apply_plan(
        _package(),
        allowed_decision_ids=[9549, 10391],
    )

    assert plan["summary"]["applicable_count"] == 2
    assert [item["decision_id"] for item in plan["applicable_items"]] == [9549, 10391]
    assert {item["reason"] for item in plan["skipped_items"]} == {
        "item_type_not_apply_supported_by_decision_raw_patch",
        "decision_id_not_in_apply_allowlist",
    }


def test_apply_request_requires_explicit_operator_intent() -> None:
    can_apply, blockers = validate_apply_request(
        apply=True,
        approval_token="wrong",
        allowed_decision_ids=[9549],
        applicable_count=1,
    )

    assert can_apply is False
    assert blockers == ["approval_token_invalid"]

    can_apply, blockers = validate_apply_request(
        apply=True,
        approval_token=APPROVAL_TOKEN,
        allowed_decision_ids=[],
        applicable_count=1,
    )

    assert can_apply is False
    assert blockers == ["explicit_decision_id_allowlist_required"]


def test_merge_raw_patch_is_recursive_and_preserves_existing_fields() -> None:
    merged = merge_raw_patch(
        {
            "profit_risk_sizing": {
                "position_size_pct": 0.01,
                "existing": "keep",
            },
            "other": True,
        },
        {
            "profit_risk_sizing": {
                "profit_first_position_ladder": {"lane": "tiny_probe"},
            },
            "profit_first_historical_recovery": {"kind": "entry"},
        },
    )

    assert merged["profit_risk_sizing"]["position_size_pct"] == 0.01
    assert merged["profit_risk_sizing"]["existing"] == "keep"
    assert merged["profit_risk_sizing"]["profit_first_position_ladder"]["lane"] == "tiny_probe"
    assert merged["profit_first_historical_recovery"]["kind"] == "entry"
    assert merged["other"] is True


@pytest.mark.asyncio
async def test_collect_apply_preview_applies_only_decision_patches(monkeypatch, tmp_path) -> None:
    async def fake_collect_package(**kwargs):
        assert kwargs["entry_decision_ids"] == [9549]
        assert kwargs["exit_decision_ids"] == [10391]
        assert kwargs["order_ids"] == [2678]
        assert kwargs["exchange_order_ids"] == ["3697115296869093376"]
        assert kwargs["use_current_blockers"] is False
        return _package()

    applied_items: list[dict] = []

    async def fake_apply(items, *, backup_dir):
        applied_items.extend(items)
        assert backup_dir == tmp_path
        return {
            "applied": len(items),
            "applied_decision_ids": [item["decision_id"] for item in items],
            "missing_decision_ids": [],
            "backup_path": str(tmp_path / "backup.json"),
        }

    monkeypatch.setattr(apply_cli, "collect_historical_recovery_package", fake_collect_package)
    monkeypatch.setattr(apply_cli, "_apply_decision_raw_patches", fake_apply)

    report = await apply_cli.collect_apply_preview(
        entry_decision_ids=[9549],
        exit_decision_ids=[10391],
        order_ids=[2678],
        exchange_order_ids=["3697115296869093376"],
        use_current_blockers=False,
        apply=True,
        approval_token=APPROVAL_TOKEN,
        backup_dir=tmp_path,
    )

    assert report["status"] == "applied"
    assert report["read_only"] is False
    assert report["mutates_database"] is True
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False
    assert report["changes_model_routing"] is False
    assert report["changes_live_sizing"] is False
    assert report["live_mutation"] is False
    assert report["resume_allowed_by_this_apply"] is False
    assert [item["decision_id"] for item in applied_items] == [9549, 10391]
    assert report["apply_result"]["applied_decision_ids"] == [9549, 10391]
