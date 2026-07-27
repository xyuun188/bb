from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.paper_bootstrap_canary import (
    PAPER_BOOTSTRAP_LEGACY_CANARY_VERSIONS,
    PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION,
    assess_paper_canary_position_horizon,
    build_paper_canary_position_lifecycle,
)


def test_retired_canary_module_has_no_new_order_policy() -> None:
    import services.paper_bootstrap_canary as module

    assert not hasattr(module, "PaperBootstrapCanaryPolicy")
    assert not hasattr(module, "annotate_paper_bootstrap_opportunity")


def test_historical_canary_position_lifecycle_remains_recoverable() -> None:
    executed_at = datetime.now(UTC) - timedelta(minutes=11)
    legacy_version = next(iter(PAPER_BOOTSTRAP_LEGACY_CANARY_VERSIONS))
    lifecycle = build_paper_canary_position_lifecycle(
        SimpleNamespace(
            id=42,
            symbol="BTC/USDT",
            action="long",
            is_paper=True,
            was_executed=True,
            executed_at=executed_at,
            raw_response={
                "paper_bootstrap_canary": {
                    "version": legacy_version,
                    "authorized": True,
                    "requested": True,
                    "execution_scope": "paper_only",
                    "production_permission": False,
                    "artifact_version": "legacy-candidate",
                    "selected_observation": {"horizon_minutes": 10},
                }
            },
        )
    )

    assert lifecycle["version"] == PAPER_BOOTSTRAP_POSITION_LIFECYCLE_VERSION
    assert lifecycle["source_contract_version"] == legacy_version
    position = {
        "symbol": "BTC/USDT",
        "side": "long",
        "execution_mode": "paper",
        "paper_canary_lifecycle": lifecycle,
    }
    assessment = assess_paper_canary_position_horizon(position)
    assert assessment["authorized"] is True
    assert assessment["elapsed"] is True

    position["execution_mode"] = "live"
    assert assess_paper_canary_position_horizon(position)["authorized"] is False


def test_normal_paper_identity_is_not_reclassified_as_legacy_canary() -> None:
    lifecycle = build_paper_canary_position_lifecycle(
        SimpleNamespace(
            id=43,
            symbol="BTC/USDT",
            action="long",
            is_paper=True,
            was_executed=True,
            executed_at=datetime.now(UTC),
            raw_response={
                "paper_bootstrap_canary": {
                    "version": next(iter(PAPER_BOOTSTRAP_LEGACY_CANARY_VERSIONS)),
                    "authorized": True,
                    "requested": True,
                    "execution_scope": "paper_only",
                    "production_permission": False,
                    "purpose": "execute_normal_paper_strategy_and_learn_after_settlement",
                    "selected_observation": {"horizon_minutes": 10},
                }
            },
        )
    )

    assert lifecycle == {}
