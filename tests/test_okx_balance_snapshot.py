from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action
from services.okx_balance_snapshot import (
    balance_snapshot_recovery_reason,
    balance_snapshot_verified,
)
from services.trading_service import TradingService


def _service(age: float = 30.0) -> TradingService:
    service = TradingService.__new__(TradingService)
    service._okx_balance_snapshot_cache = {
        "paper": {
            "snapshot": {"free": 100.0, "equity": 120.0},
            "fetched_at": datetime.now(UTC) - timedelta(seconds=age),
        }
    }
    service._okx_authoritative_sync_status_payload = lambda: {
        "status": "ok",
        "fresh_success_available": True,
    }
    return service


def _facts(snapshot, pre_order: bool):
    prefix = "okx_pre_order_" if pre_order else "okx_"
    reason = f"{prefix}account_equity_missing,{prefix}available_margin_missing"
    return {
        "production_eligible": False,
        "balance_snapshot": snapshot,
        "reason": reason,
        "policy_provenance": {"fallback_reason": reason},
    }


@pytest.mark.parametrize("pre_order", [True, False])
@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {},
        {"equity": 0.0, "free": 0.0},
        {"error": {"code": "50004", "msg": "timeout"}},
        {"error": "OKX request timed out"},
        {"error": "empty", "error_kind": "empty_response"},
    ],
)
def test_entry_stages_share_bounded_balance_recovery(snapshot, pre_order):
    result = _service()._entry_facts_with_verified_balance(
        "paper", _facts(snapshot, pre_order), pre_order=pre_order
    )
    assert result["production_eligible"] is True
    assert result["available_margin_usdt"] == 100.0
    assert result["account_equity_usdt"] == 120.0
    assert result["requires_final_balance_recheck"] is True


@pytest.mark.parametrize("pre_order", [True, False])
@pytest.mark.parametrize(
    "snapshot",
    [
        {"equity": 120.0, "free": 0.0},
        {"equity": 0.0, "free": 0.0, "verified": True},
        {"error": {"code": "50113", "msg": "invalid signature"}},
        {"error": "unclassified failure"},
    ],
)
def test_entry_stages_do_not_replace_authoritative_zero_or_permanent_error(snapshot, pre_order):
    service = _service()
    facts = _facts(snapshot, pre_order)
    assert service._entry_facts_with_verified_balance(
        "paper", facts, pre_order=pre_order
    ) is facts


@pytest.mark.parametrize("pre_order", [True, False])
def test_entry_stages_reject_expired_or_other_account_cache(pre_order):
    facts = _facts({}, pre_order)
    assert _service(121)._entry_facts_with_verified_balance(
        "paper", facts, pre_order=pre_order
    ) is facts
    assert _service()._entry_facts_with_verified_balance(
        "live", facts, pre_order=pre_order
    ) is facts


@pytest.mark.parametrize("pre_order", [True, False])
def test_balance_recovery_does_not_remove_other_exchange_blockers(pre_order):
    facts = _facts({}, pre_order)
    facts["reason"] += ",contract_spec_missing"
    facts["policy_provenance"]["fallback_reason"] += ",contract_spec_missing"
    result = _service()._entry_facts_with_verified_balance(
        "paper", facts, pre_order=pre_order
    )
    assert result["production_eligible"] is False
    assert result["policy_provenance"]["fallback_reason"] == "contract_spec_missing"


def test_failed_refresh_cannot_overwrite_verified_balance():
    service = _service()
    before = service._okx_balance_snapshot_cache["paper"]
    for snapshot in ({}, None, {"free": 0.0, "equity": 0.0}, {"error": "timeout"}):
        assert service._remember_okx_balance_snapshot_for_mode("paper", snapshot) is None
        assert service._okx_balance_snapshot_cache["paper"] is before
    assert service._remember_okx_balance_snapshot_for_mode(
        "paper", {"free": 0.0, "equity": 0.0, "verified": True}
    ) is not None
    assert service.peek_okx_balance_snapshot_for_mode("paper")["free"] == 0


def test_execution_invalidation_keeps_age_bounded_fact_but_forces_refresh():
    service = _service(1)
    fetched_at = service._okx_balance_snapshot_cache["paper"]["fetched_at"]
    service._invalidate_okx_balance_snapshot_cache_for_mode("paper")
    assert service._cached_okx_balance_snapshot("paper", max_age_seconds=15) is None
    assert service.peek_okx_balance_snapshot_for_mode("paper")["free"] == 100
    assert service._okx_balance_snapshot_cache["paper"]["fetched_at"] == fetched_at


@pytest.mark.asyncio
async def test_successful_native_sizing_seeds_cache_for_pre_order_recovery():
    service = _service(200)
    native = {"free": 200.0, "equity": 250.0, "verified": True}

    class Executor:
        async def entry_risk_facts(self, _symbol, _positions):
            return {"production_eligible": True, "balance_snapshot": native}

        async def pre_order_execution_facts(self, _symbol, _side):
            return _facts({"error": {"code": "50004"}}, True)

    async def executor_provider(_mode):
        return Executor()

    service._get_okx_executor_for_mode = executor_provider
    decision = SimpleNamespace(symbol="DOGE/USDT", action=Action.SHORT)
    await service.entry_exchange_risk_facts("paper", decision, [])
    service._invalidate_okx_balance_snapshot_cache_for_mode("paper")
    result = await service.pre_order_execution_facts("paper", decision)
    assert result["production_eligible"] is True
    assert result["available_margin_usdt"] == 200
    assert result["requires_final_balance_recheck"] is True


def test_recovery_requires_known_transient_evidence():
    assert not balance_snapshot_verified({"free": 0, "equity": 0})
    assert balance_snapshot_recovery_reason({"free": 0, "equity": 0, "verified": True}) is None
    assert balance_snapshot_recovery_reason({"error": {"code": "50113", "msg": "timeout"}}) is None
