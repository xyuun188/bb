from datetime import UTC, datetime, timedelta

from services.local_ai_training_contract import (
    COST_MODEL_VERSION,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_VERSION,
    authoritative_cost_training_identity,
    decision_group_training_trigger,
    local_ai_training_cursor,
    training_distribution_drift,
)
from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    COST_MODEL_VERSION as CLIENT_COST_MODEL_VERSION,
)
from services.return_objective import (
    RETURN_LABEL_VERSION as CLIENT_RETURN_LABEL_VERSION,
)
from services.return_objective import (
    RETURN_OBJECTIVE_VERSION as CLIENT_RETURN_OBJECTIVE_VERSION,
)


def _shadow_sample(sample_id: int, group: str) -> dict:
    return {
        "id": sample_id,
        "decision_id": sample_id,
        "features": {
            "horizon_minutes": 15,
            "returns_5": 0.01 * sample_id,
            "returns_20": 0.02,
            "volatility_20": 0.03,
            "spread_pct": 0.01,
            "orderbook_imbalance": 0.1,
        },
        "sample_weight": 1.0,
        "correlation_weight": {"correlation_group": group},
        "profit_supervision": {
            "version": PROFIT_SUPERVISION_VERSION,
            "tasks": {
                "market_opportunity_distribution": {
                    "eligible": True,
                    "long_gross_market_return_pct": 0.2,
                    "short_gross_market_return_pct": -0.2,
                }
            },
        },
    }


def _trade_sample(sample_id: int, lifecycle: str, realized_pnl: float) -> dict:
    return {
        "id": sample_id,
        "lifecycle_key": lifecycle,
        "side": "long",
        "realized_pnl": realized_pnl,
        "features": {"horizon_minutes": 15, "spread_pct": 0.01},
        "sample_weight": 1.0,
        "profit_supervision": {
            "version": PROFIT_SUPERVISION_VERSION,
            "tasks": {
                "execution_cost_and_slippage_distribution": {
                    "eligible": True,
                    "source_authority": "okx_fills_fees_funding",
                    "total_cost_pct": 0.04,
                }
            },
        },
    }


def test_client_and_training_contract_use_the_same_v3_versions() -> None:
    assert CLIENT_RETURN_OBJECTIVE_VERSION == RETURN_OBJECTIVE_VERSION
    assert CLIENT_RETURN_LABEL_VERSION == RETURN_LABEL_VERSION
    assert CLIENT_COST_MODEL_VERSION == COST_MODEL_VERSION


def test_training_cursor_counts_independent_groups_and_includes_losses() -> None:
    loss = _trade_sample(11, "position-11", -2.5)
    assert authoritative_cost_training_identity(loss) is not None

    cursor = local_ai_training_cursor(
        shadow_samples=[
            _shadow_sample(1, "scan-1"),
            _shadow_sample(2, "scan-1"),
            _shadow_sample(3, "scan-2"),
        ],
        trade_samples=[loss, _trade_sample(12, "position-12", 1.0)],
    )

    assert cursor["completed_market_sample_count"] == 3
    assert cursor["completed_market_decision_group_count"] == 2
    assert cursor["completed_authoritative_cost_decision_group_count"] == 2
    assert cursor["completed_training_decision_group_count"] == 4
    assert (
        cursor["training_distribution_profile"]["features"]["authoritative_execution_cost_pct"][
            "count"
        ]
        == 2
    )


def test_training_trigger_requires_independent_group_evidence() -> None:
    now = datetime(2026, 7, 27, 12, tzinfo=UTC)
    no_drift = {"detected": False}

    one_new = decision_group_training_trigger(
        force=False,
        has_artifact=True,
        completed_group_count=156,
        previous_group_count=155,
        trained_at=now.isoformat(),
        now=now,
        distribution_drift=no_drift,
        batch_threshold=50,
        minimum_increment=10,
        drift_minimum_increment=10,
        maximum_interval_seconds=86400,
    )
    batch = decision_group_training_trigger(
        force=False,
        has_artifact=True,
        completed_group_count=205,
        previous_group_count=155,
        trained_at=now.isoformat(),
        now=now,
        distribution_drift=no_drift,
        batch_threshold=50,
        minimum_increment=10,
        drift_minimum_increment=10,
        maximum_interval_seconds=86400,
    )
    elapsed_without_data = decision_group_training_trigger(
        force=False,
        has_artifact=True,
        completed_group_count=155,
        previous_group_count=155,
        trained_at=(now - timedelta(days=2)).isoformat(),
        now=now,
        distribution_drift=no_drift,
        batch_threshold=50,
        minimum_increment=10,
        drift_minimum_increment=10,
        maximum_interval_seconds=86400,
    )
    daily = decision_group_training_trigger(
        force=False,
        has_artifact=True,
        completed_group_count=165,
        previous_group_count=155,
        trained_at=(now - timedelta(days=1)).isoformat(),
        now=now,
        distribution_drift=no_drift,
        batch_threshold=50,
        minimum_increment=10,
        drift_minimum_increment=10,
        maximum_interval_seconds=86400,
    )

    assert one_new["due"] is False
    assert batch["reason"] == "mature_decision_group_batch"
    assert elapsed_without_data["due"] is False
    assert daily["reason"] == "daily_minimum_increment"


def test_training_trigger_handles_drift_and_rebased_views() -> None:
    now = datetime(2026, 7, 27, 12, tzinfo=UTC)
    drift = training_distribution_drift(
        {"features": {"returns_5": {"mean": 1.0, "std": 0.1}}},
        {"features": {"returns_5": {"mean": 0.0, "std": 0.1}}},
        threshold=0.35,
    )
    drift_due = decision_group_training_trigger(
        force=False,
        has_artifact=True,
        completed_group_count=110,
        previous_group_count=100,
        trained_at=now.isoformat(),
        now=now,
        distribution_drift=drift,
        batch_threshold=50,
        minimum_increment=10,
        drift_minimum_increment=10,
        maximum_interval_seconds=86400,
    )
    rebased = decision_group_training_trigger(
        force=False,
        has_artifact=True,
        completed_group_count=90,
        previous_group_count=100,
        trained_at=now.isoformat(),
        now=now,
        distribution_drift={"detected": False},
        batch_threshold=50,
        minimum_increment=10,
        drift_minimum_increment=10,
        maximum_interval_seconds=86400,
    )

    assert drift_due["reason"] == "distribution_drift_with_new_labels"
    assert rebased["reason"] == "training_view_rebased"


def test_training_trigger_scales_batches_and_cools_down_drift_retraining() -> None:
    now = datetime(2026, 8, 17, 3, tzinfo=UTC)
    drift = {"detected": True}
    common = {
        "force": False,
        "has_artifact": True,
        "completed_group_count": 20_049,
        "previous_group_count": 20_016,
        "distribution_drift": drift,
        "batch_threshold": 50,
        "minimum_increment": 10,
        "drift_minimum_increment": 10,
        "maximum_interval_seconds": 86400,
        "batch_growth_fraction": 0.05,
        "minimum_retraining_interval_seconds": 6 * 60 * 60,
    }

    within_cooldown = decision_group_training_trigger(
        **common,
        trained_at=(now - timedelta(hours=1)).isoformat(),
        now=now,
    )
    after_cooldown = decision_group_training_trigger(
        **common,
        trained_at=(now - timedelta(hours=7)).isoformat(),
        now=now,
    )

    assert within_cooldown["reason"] == "not_due"
    assert within_cooldown["minimum_retraining_interval_elapsed"] is False
    assert within_cooldown["effective_batch_decision_group_threshold"] == 1001
    assert after_cooldown["reason"] == "distribution_drift_with_new_labels"
    assert after_cooldown["minimum_retraining_interval_elapsed"] is True
