from ai_brain.base_model import Action, DecisionOutput
from services.entry_wick_guard import EntryAbnormalWickGuardPolicy


def _decision(
    action: Action = Action.LONG,
    *,
    snapshot: dict | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="测试开仓",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot=snapshot or {},
    )


def test_entry_wick_guard_blocks_recent_extreme_wicks_and_records_cooldown() -> None:
    calls: list[tuple[str | None, str, float]] = []
    policy = EntryAbnormalWickGuardPolicy(
        lambda symbol, reason, minutes: calls.append((symbol, reason, minutes)),
        temporary_block_minutes=8.0,
    )
    decision = _decision(
        snapshot={
            "abnormal_wick_count_72h": 2,
            "abnormal_wick_max_pct": 120.0,
            "abnormal_wick_recent_hours": 3.5,
        }
    )

    reason = policy.guard_reason(decision)

    assert reason is not None
    assert "BTC/USDT" in reason
    assert decision.raw_response["abnormal_wick_guard"] == {
        "blocked": True,
        "count_72h": 2,
        "max_wick_pct": 120.0,
        "recent_hours": 3.5,
        "rule": "recent extreme wick can fill stops far from the planned stop price",
    }
    assert len(calls) == 1
    assert calls[0][0] == "BTC/USDT"
    assert calls[0][2] == 60.0


def test_entry_wick_guard_allows_old_small_or_non_entry_signals() -> None:
    policy = EntryAbnormalWickGuardPolicy()

    assert (
        policy.guard_reason(
            _decision(
                snapshot={
                    "abnormal_wick_count_72h": 1,
                    "abnormal_wick_max_pct": 79.0,
                    "abnormal_wick_recent_hours": 1.0,
                }
            )
        )
        is None
    )
    assert (
        policy.guard_reason(
            _decision(
                snapshot={
                    "abnormal_wick_count_72h": 1,
                    "abnormal_wick_max_pct": 120.0,
                    "abnormal_wick_recent_hours": 120.0,
                }
            )
        )
        is None
    )
    assert (
        policy.guard_reason(
            _decision(
                Action.CLOSE_LONG,
                snapshot={
                    "abnormal_wick_count_72h": 2,
                    "abnormal_wick_max_pct": 120.0,
                    "abnormal_wick_recent_hours": 3.5,
                },
            )
        )
        is None
    )
