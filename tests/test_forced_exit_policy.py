from ai_brain.base_model import Action, DecisionOutput
from services.forced_exit import ForcedExitPolicy


def _decision(
    *,
    model_name: str = "ensemble_trader",
    reasoning: str = "ordinary exit",
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


def test_forced_exit_policy_rejects_legacy_structured_raw_flags() -> None:
    policy = ForcedExitPolicy()

    assert not policy.is_forced_exit(_decision(raw_response={"fast_risk_exit": True}))
    assert not policy.is_forced_exit(_decision(raw_response={"forced_exit": True}))
    assert not policy.is_forced_exit(
        _decision(raw_response={"close_evidence": {"hard_risk": True}})
    )
    assert not policy.is_forced_exit(
        _decision(raw_response={"position_review_risk_alert": {"force_exit": True}})
    )


def test_forced_exit_policy_rejects_model_and_reason_keywords() -> None:
    policy = ForcedExitPolicy()

    assert not policy.is_forced_exit(_decision(model_name="risk_engine"))
    assert not policy.is_forced_exit(_decision(reasoning="STOP LOSS triggered"))


def test_forced_exit_policy_accepts_governed_planned_stop() -> None:
    decision = _decision(
        raw_response={
            "dynamic_exit_policy": {
                "eligible": True,
                "hard_risk": True,
                "planned_stop_crossed": True,
                "policy_provenance": {
                    "source": (
                        "current_position_fee_after_pnl_peak_planned_stop_and_market_returns"
                    ),
                },
            },
        }
    )

    assert ForcedExitPolicy().is_forced_exit(decision)


def test_forced_exit_policy_requires_complete_governed_contract() -> None:
    base = {
        "eligible": True,
        "hard_risk": True,
        "planned_stop_crossed": True,
        "policy_provenance": {
            "source": "current_position_fee_after_pnl_peak_planned_stop_and_market_returns"
        },
    }
    for missing in ("eligible", "hard_risk", "planned_stop_crossed", "policy_provenance"):
        policy = dict(base)
        policy.pop(missing)
        assert not ForcedExitPolicy().is_forced_exit(
            _decision(raw_response={"dynamic_exit_policy": policy})
        )
