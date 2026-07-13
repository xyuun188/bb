from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from services.position_capacity_release_audit import PositionCapacityReleaseAuditService


def _position(**overrides):
    values = {
        "id": 1,
        "model_name": "ensemble_trader",
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 2.0,
        "entry_price": 100.0,
        "current_price": 103.0,
        "unrealized_pnl": 6.0,
        "entry_fee": 0.12,
        "stop_loss_price": 98.0,
        "created_at": datetime.now(UTC),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _decision(*, decision_id: int, raw: dict, executed: bool = False):
    return SimpleNamespace(
        id=decision_id,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="close_long",
        was_executed=executed,
        raw_llm_response=raw,
        created_at=datetime.now(UTC),
    )


def _dynamic_exit_policy(*, complete: bool = True) -> dict:
    return {
        "eligible": True,
        "close_fraction": 0.72,
        "hard_risk": False,
        "policy_provenance": {
            "source": "position_fee_after_pnl_peak_stop_continuation_and_portfolio_state",
            "observation_window": "current_position_review",
            "sample_count": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-07-12.dynamic-exit-return.v1",
            "fallback_reason": "" if complete else "position_economics_missing",
        },
    }


def test_capacity_audit_reports_hard_capacity_and_position_economics_only() -> None:
    service = PositionCapacityReleaseAuditService()
    report = service._summarize(
        [_position(), _position(id=2, symbol="ETH/USDT", entry_fee=0.0)],
        [
            _decision(
                decision_id=10,
                raw={
                    "position_release_policy": {
                        "forced": True,
                        "source": "position_quality_capacity_release",
                    }
                },
            )
        ],
        [],
    )

    assert report["read_only"] is True
    assert report["audit_only"] is True
    assert report["can_force_close"] is False
    assert report["open_position_count"] == 2
    assert report["position_economics_complete_count"] == 1
    assert report["position_economics_incomplete_count"] == 1
    assert report["dynamic_exit_decision_count"] == 1
    assert report["dynamic_exit_decisions"][0]["dynamic_exit_contract_complete"] is False
    assert not any("release_decision" in key or "crowded" in key for key in report)


def test_capacity_audit_requires_dynamic_exit_provenance_and_filled_order_link() -> None:
    service = PositionCapacityReleaseAuditService()
    decision = _decision(
        decision_id=20,
        raw={"dynamic_exit_policy": _dynamic_exit_policy()},
        executed=True,
    )
    filled = SimpleNamespace(decision_id=20, status="filled")

    report = service._summarize([_position()], [decision], [filled])

    assert report["dynamic_exit_decision_count"] == 1
    assert report["executed_dynamic_exit_count"] == 1
    assert report["executed_dynamic_exit_contract_gap_count"] == 0
    row = report["dynamic_exit_decisions"][0]
    assert row["close_fraction"] == 0.72
    assert row["filled_order_count"] == 1
    assert row["dynamic_exit_contract_complete"] is True


def test_capacity_audit_flags_executed_exit_without_complete_dynamic_contract() -> None:
    service = PositionCapacityReleaseAuditService()
    decision = _decision(
        decision_id=30,
        raw={"dynamic_exit_policy": _dynamic_exit_policy(complete=False)},
        executed=True,
    )

    report = service._summarize([_position()], [decision], [])

    assert report["executed_dynamic_exit_count"] == 1
    assert report["executed_dynamic_exit_contract_gap_count"] == 1
    gap = report["executed_dynamic_exit_contract_gaps"][0]
    assert gap["filled_order_count"] == 0
    assert gap["dynamic_exit_contract_complete"] is False
