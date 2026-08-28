from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_local_ai_tools_training_cursors as script
from services.local_ai_training_contract import (
    TRAINING_CURSOR_VERSION,
    TRAINING_DISTRIBUTION_PROFILE_VERSION,
)
from services.profit_supervision import (
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    MARKET_OPPORTUNITY_TASK,
    PROFIT_SUPERVISION_VERSION,
)


def test_cursor_process_installs_memory_guard_before_project_imports() -> None:
    source = script.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    assert "LOCAL_AI_CURSOR_MEMORY_LIMIT_BYTES" in text
    assert "resource.RLIMIT_AS" in text


def _shadow_sample(sample_id: int, group: str) -> dict:
    return {
        "id": sample_id,
        "decision_id": sample_id,
        "features": {
            "horizon_minutes": 15,
            "returns_5": 0.01 * sample_id,
            "spread_pct": 0.01,
        },
        "sample_weight": 1.0,
        "correlation_weight": {"correlation_group": group},
        "profit_supervision": {
            "version": PROFIT_SUPERVISION_VERSION,
            "tasks": {
                MARKET_OPPORTUNITY_TASK: {
                    "eligible": True,
                    "long_gross_market_return_pct": 0.2,
                    "short_gross_market_return_pct": -0.2,
                }
            },
        },
    }


def _trade_sample(sample_id: int, lifecycle: str) -> dict:
    return {
        "id": sample_id,
        "lifecycle_key": lifecycle,
        "side": "long",
        "features": {"horizon_minutes": 15, "spread_pct": 0.01},
        "sample_weight": 1.0,
        "profit_supervision": {
            "version": PROFIT_SUPERVISION_VERSION,
            "tasks": {
                COUNTERFACTUAL_EXECUTION_COST_TASK: {
                    "eligible": True,
                    "source_authority": "okx_fills_fees_funding",
                    "total_cost_pct": 0.04,
                }
            },
        },
    }


@pytest.mark.asyncio
async def test_run_once_returns_canonical_training_cursors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    async def shadow_counter() -> int:
        return 14

    async def shadow_loader() -> list[dict]:
        return [
            _shadow_sample(1, "scan-1"),
            _shadow_sample(2, "scan-1"),
            _shadow_sample(3, "scan-2"),
        ]

    async def trade_loader() -> list[dict]:
        return [
            _trade_sample(11, "position-11"),
            _trade_sample(12, "position-12"),
        ]

    def passthrough_training_payload(**payload):
        return payload

    async def fake_close_db() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(script, "close_db", fake_close_db)
    monkeypatch.setattr(script, "annotate_training_payload", passthrough_training_payload)

    result = await script.run_once(
        shadow_counter=shadow_counter,
        shadow_loader=shadow_loader,
        trade_loader=trade_loader,
    )

    assert result["trained"] is False
    assert result["reason"] == "cursor_probe_complete"
    assert result["completed_shadow_sample_count"] == 14
    assert result["completed_trade_sample_count"] == 2
    assert result["version"] == TRAINING_CURSOR_VERSION
    assert result["completed_market_sample_count"] == 3
    assert result["completed_authoritative_cost_sample_count"] == 2
    assert result["completed_market_decision_group_count"] == 2
    assert result["completed_authoritative_cost_decision_group_count"] == 2
    assert result["completed_training_decision_group_count"] == 4
    assert (
        result["training_distribution_profile"]["version"]
        == TRAINING_DISTRIBUTION_PROFILE_VERSION
    )
    assert result["training_process_isolated"] is True
    assert result["cursor_policy"] == "canonical_clean_independent_decision_group_view"
    assert closed is True


@pytest.mark.asyncio
async def test_run_once_returns_structured_error_and_closes_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    async def fail_shadow_counter() -> int:
        raise RuntimeError("cursor query failed")

    async def empty_loader() -> list[dict]:
        return []

    async def fake_close_db() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(script, "close_db", fake_close_db)

    result = await script.run_once(
        shadow_counter=fail_shadow_counter,
        shadow_loader=empty_loader,
        trade_loader=empty_loader,
    )

    assert result["trained"] is False
    assert result["reason"] == "error"
    assert result["error"] == "cursor query failed"
    assert result["training_process_isolated"] is True
    assert closed is True
