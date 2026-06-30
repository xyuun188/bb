from __future__ import annotations

from services.profit_first_stage2 import RecentProbePnLBrakePolicy, ReleaseNetBenefitPolicy


def test_recent_probe_pnl_brake_blocks_tiny_probe_after_all_loss_window() -> None:
    decision = RecentProbePnLBrakePolicy().evaluate(
        {"decision_lane": "tiny_probe"},
        {
            "probe_loop_health": {
                "all_recent_probes_losing": True,
                "probe_closed_count": 3,
                "probe_loss_count": 3,
            }
        },
    )

    assert decision.allowed is False
    assert decision.data["skip_kind"] == "profit_first_probe_loss_brake"


def test_recent_probe_pnl_brake_allows_validated_upgrade() -> None:
    decision = RecentProbePnLBrakePolicy().evaluate(
        {"decision_lane": "validated_probe"},
        {"probe_loop_health": {"all_recent_probes_losing": True, "probe_closed_count": 5}},
    )

    assert decision.allowed is True
    assert decision.data["probe_loss_brake_bypassed_by_upgrade"] is True


def test_release_net_benefit_guard_blocks_losing_release_without_replacement() -> None:
    decision = ReleaseNetBenefitPolicy().evaluate(
        {
            "position_quality": {
                "pnl_ratio": -0.004,
                "reasons": ["stale_probe_capital_inefficient"],
            }
        }
    )

    assert decision.allowed is False
    assert decision.data["skip_kind"] == "profit_first_release_net_benefit_guard"


def test_release_net_benefit_guard_allows_hard_risk_or_stronger_replacement() -> None:
    hard_risk = ReleaseNetBenefitPolicy().evaluate(
        {
            "position_quality": {
                "pnl_ratio": -0.03,
                "reasons": ["hard_loss_pressure"],
            }
        }
    )
    replacement = ReleaseNetBenefitPolicy().evaluate(
        {
            "position_quality": {"pnl_ratio": -0.004, "reasons": ["fee_drag_dominates"]},
            "replacement_opportunity": {
                "decision_lane": "validated_probe",
                "expected_net_return_pct": 0.55,
                "profit_quality_ratio": 0.7,
            },
        }
    )

    assert hard_risk.allowed is True
    assert hard_risk.data["release_net_benefit_hard_risk"] is True
    assert replacement.allowed is True
    assert replacement.data["release_net_benefit_replacement"] is True
