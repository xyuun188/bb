from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.paper_training import (
    PAPER_TRAINING_ORDER_IDENTITY_VERSION,
    PAPER_TRAINING_POSITION_LIFECYCLE_VERSION,
    assess_paper_training_position_horizon,
    build_paper_training_position_lifecycle,
    paper_training_contract_reasons,
    paper_training_decision_id_from_client_order_id,
)
from tests.legacy_paper_contract_fixtures import build_legacy_paper_training_contract


def _contract() -> dict:
    return build_legacy_paper_training_contract(
        symbol="BTC/USDT",
        selected_side="long",
        signal_source="local_ml_observation",
        expected_net_return_pct=-0.8,
        return_lcb_pct=-1.2,
        horizon_minutes=10.0,
    )


def test_historical_contract_is_readable_but_runtime_builders_are_absent() -> None:
    import services.paper_training as module

    contract = _contract()
    assert paper_training_contract_reasons(contract) == []
    assert not hasattr(module, "build_paper_training_contract")
    assert not hasattr(module, "assess_paper_training_entry")
    assert not hasattr(module, "attach_paper_training_order_identity")

    tampered = {**contract, "production_permission": True}
    assert "paper_training_production_permission_invalid" in paper_training_contract_reasons(
        tampered
    )


def test_historical_client_order_identity_is_parse_only() -> None:
    assert paper_training_decision_id_from_client_order_id("BBPT104208") == 104208
    assert paper_training_decision_id_from_client_order_id("BBPT-not-a-number") is None
    assert paper_training_decision_id_from_client_order_id("OTHER104208") is None


def test_historical_position_lifecycle_recovers_decision_identity() -> None:
    raw = {
        "paper_training": _contract(),
        "paper_training_order_identity": {
            "version": PAPER_TRAINING_ORDER_IDENTITY_VERSION,
            "execution_scope": "paper_only",
            "production_permission": False,
            "decision_id": 104208,
            "client_order_id": "BBPT104208",
        },
    }
    lifecycle = build_paper_training_position_lifecycle(
        SimpleNamespace(
            symbol="BTC/USDT",
            action="long",
            raw_response=raw,
            is_paper=True,
            was_executed=True,
            executed_at=datetime.now(UTC) - timedelta(minutes=11),
        )
    )

    assert lifecycle["decision_id"] == 104208
    assert lifecycle["version"] == PAPER_TRAINING_POSITION_LIFECYCLE_VERSION
    position = {
        "symbol": "BTC/USDT",
        "side": "long",
        "execution_mode": "paper",
        "paper_training_lifecycle": lifecycle,
    }
    assessment = assess_paper_training_position_horizon(position)
    assert assessment["authorized"] is True
    assert assessment["elapsed"] is True
    position["execution_mode"] = "live"
    assert assess_paper_training_position_horizon(position)["authorized"] is False
