from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.model_training_coordinator import ModelTrainingCoordinatorMixin
from services.model_training_state import LOCAL_AI_TOOL_MODEL_IDS


class CursorCoordinator(ModelTrainingCoordinatorMixin):
    def __init__(self, *, age_hours, previous_fingerprint, current_fingerprint):
        self.training_calls = 0
        self._local_tools_last_completed_shadow_count = 0
        self._local_tools_active_training_run_id = None
        trained_at = (datetime.now(UTC) - timedelta(hours=age_hours)).isoformat()
        self.previous_result = {
            "cursor_count_scope": "clean_market_groups_and_settlement_fact_fingerprint",
            "completed_market_decision_group_count": 20,
            "authoritative_trade_training_probe": {
                "available": True,
                "history_row_count": 4,
                "fingerprint": previous_fingerprint,
            },
        }
        self.current_cursor = {
            "reason": "cursor_probe_complete",
            "cursor_count_scope": "clean_market_groups_and_settlement_fact_fingerprint",
            "completed_shadow_sample_count": 80,
            "completed_market_decision_group_count": 20,
            "completed_training_decision_group_count": 20,
            "authoritative_trade_training_probe": {
                "available": True,
                "history_row_count": 4,
                "fingerprint": current_fingerprint,
            },
        }
        self.state_row = {
            "sample_cursor": {"shadow": 80, "trade": 2, "decision_group": 9999},
            "last_successful_training_at": trained_at,
            "last_successful_result": self.previous_result,
        }
        self.model_training_state_store = SimpleNamespace(
            read=lambda: {"models": {LOCAL_AI_TOOL_MODEL_IDS[0]: self.state_row}}
        )
        self.local_ai_tools = SimpleNamespace(enabled=lambda: True, status=self.status)

    @staticmethod
    def _safe_int(value, default=0):
        return int(value) if value is not None else default

    async def status(self):
        return {
            "available": True,
            "model_bundle_available": True,
            "last_trained_completed_shadow_sample_count": 80,
            "last_trained_completed_trade_sample_count": 2,
            "last_trained_completed_training_decision_group_count": 9999,
            "completed_market_decision_group_count": 2000,
        }

    async def _run_local_ai_tools_training_cursor_subprocess(self):
        return dict(self.current_cursor)

    async def _run_local_ai_tools_training_subprocess(self):
        self.training_calls += 1
        return {
            "trained": True,
            "completed_trade_sample_count": 3,
            "completed_training_decision_group_count": 8,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("age_hours", "current_fingerprint", "expected_calls"),
    [(7, "same", 0), (7, "repaired", 1), (1, "repaired", 0)],
)
async def test_settlement_repairs_use_one_training_cadence(
    monkeypatch, age_hours, current_fingerprint, expected_calls
):
    monkeypatch.setattr(
        "services.okx_training_gate.okx_training_refresh_gate", lambda: {"allowed": True}
    )
    host = CursorCoordinator(
        age_hours=age_hours, previous_fingerprint="same", current_fingerprint=current_fingerprint
    )

    result = await host._maybe_train_local_ai_tools_process()

    assert host.training_calls == expected_calls
    assert result["completed_training_decision_group_count"] == 20
    assert result["completed_trade_sample_count"] == (3 if expected_calls else 2)
    assert result["authoritative_trade_training_probe"]["history_row_count"] == 4
    assert result["training_policy"]["trigger"] == (
        "authoritative_settlement_facts_changed" if expected_calls else "not_due"
    )
    if expected_calls:
        host.state_row["last_successful_result"] = result
        host.state_row["last_successful_training_at"] = datetime.now(UTC).isoformat()
        host.state_row["sample_cursor"] = {"shadow": 80, "trade": 3, "decision_group": 20}
        second = await host._maybe_train_local_ai_tools_process()
        assert second["reason"] == "not_due"
        assert host.training_calls == 1
