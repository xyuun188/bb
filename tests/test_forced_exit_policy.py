from ai_brain.base_model import Action, DecisionOutput
from services.forced_exit import ForcedExitPolicy


def _decision(
    *,
    model_name: str = "ensemble_trader",
    reasoning: str = "普通平仓",
    raw_response: dict | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        model_name=model_name,
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning=reasoning,
        position_size_pct=1.0,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": 100.0},
    )


def test_forced_exit_policy_detects_structured_raw_flags() -> None:
    policy = ForcedExitPolicy()

    assert policy.is_forced_exit(_decision(raw_response={"fast_risk_exit": True}))
    assert policy.is_forced_exit(_decision(raw_response={"forced_exit": True}))
    assert policy.is_forced_exit(_decision(raw_response={"close_evidence": {"hard_risk": True}}))
    assert policy.is_forced_exit(
        _decision(raw_response={"position_review_risk_alert": {"force_exit": True}})
    )


def test_forced_exit_policy_detects_model_and_reason_keywords() -> None:
    policy = ForcedExitPolicy()

    assert policy.is_forced_exit(_decision(model_name="risk_engine"))
    assert policy.is_forced_exit(_decision(reasoning="STOP LOSS triggered"))
    assert policy.is_forced_exit(_decision(reasoning="触发止损，快速风控要求离场"))


def test_forced_exit_policy_ignores_ordinary_exit() -> None:
    assert not ForcedExitPolicy().is_forced_exit(_decision())


def test_forced_exit_policy_ignores_low_quality_release_without_hard_risk() -> None:
    decision = _decision(
        raw_response={
            "forced_exit": True,
            "position_release_policy": {
                "source": "position_quality_capacity_release",
                "forced": True,
            },
            "close_evidence": {
                "forced_exit": True,
                "hard_risk": False,
                "source": "low_quality_position_release",
            },
        }
    )

    assert not ForcedExitPolicy().is_forced_exit(decision)
