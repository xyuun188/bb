from services.model_champion_policy import compare_candidate_to_champion


def _metadata(
    *,
    avg: float = 0.30,
    lcb: float = 0.12,
    profit_factor: float = 1.40,
    cvar: float = -0.20,
    drawdown: float = 0.40,
    training_data_version: str = "2026-09-15.authoritative-trade-integrity.v6",
) -> dict:
    side = {
        "avg_return_pct": avg,
        "return_lcb_pct": lcb,
        "profit_factor": profit_factor,
        "cvar_10_pct": cvar,
        "max_drawdown_pct": drawdown,
        "promotion_math_ready": True,
    }
    return {
        "training_data_version": training_data_version,
        "quality_report": {"data_quality_version": training_data_version},
        "training_data_sha256": "a" * 64,
        "walk_forward_report": {
            "sides": {
                "long": {
                    "promotion_math_ready": True,
                    "market_regime_stability": {"stable": True},
                },
                "short": {
                    "promotion_math_ready": True,
                    "market_regime_stability": {"stable": True},
                },
            },
            "folds": [
                {
                    "sides": {
                        "long": {"promotion_math_ready": True},
                        "short": {"promotion_math_ready": True},
                    }
                },
                {
                    "sides": {
                        "long": {"promotion_math_ready": True},
                        "short": {"promotion_math_ready": True},
                    }
                },
            ],
        },
        "leave_one_symbol_out_report": {
            "long": {"stable": True},
            "short": {"stable": True},
        },
        "oos_return_evaluation": {
            "long": dict(side),
            "short": dict(side),
        },
    }


def test_active_challenger_requires_strict_fee_after_improvement() -> None:
    champion = _metadata()
    challenger = _metadata(
        avg=0.35,
        lcb=0.15,
        profit_factor=1.50,
        cvar=-0.18,
        drawdown=0.35,
    )

    report = compare_candidate_to_champion(
        challenger,
        champion,
        candidate_stage="active",
        champion_stage="active",
    )

    assert report["accepted"] is True
    assert report["reason"] == "strict_fee_after_improvement"


def test_active_champion_is_retained_when_tail_risk_worsens() -> None:
    champion = _metadata()
    challenger = _metadata(
        avg=0.35,
        lcb=0.15,
        profit_factor=1.50,
        cvar=-0.25,
        drawdown=0.35,
    )

    report = compare_candidate_to_champion(
        challenger,
        champion,
        candidate_stage="active",
        champion_stage="active",
    )

    assert report["accepted"] is False
    assert "candidate_cvar_worsened" in report["blocking_reasons"]


def test_candidate_cannot_downgrade_active_champion_to_canary() -> None:
    report = compare_candidate_to_champion(
        _metadata(),
        _metadata(),
        candidate_stage="canary",
        champion_stage="active",
    )

    assert report["accepted"] is False
    assert report["blocking_reasons"] == ["candidate_lifecycle_regression"]


def test_governed_canary_to_active_upgrade_is_accepted() -> None:
    report = compare_candidate_to_champion(
        _metadata(),
        _metadata(),
        candidate_stage="active",
        champion_stage="canary",
    )

    assert report["accepted"] is True
    assert report["reason"] == "governed_lifecycle_upgrade"


def test_same_stage_challenger_must_improve_primary_metrics() -> None:
    rejected = compare_candidate_to_champion(
        _metadata(),
        _metadata(),
        candidate_stage="canary",
        champion_stage="canary",
    )
    accepted = compare_candidate_to_champion(
        _metadata(avg=0.31),
        _metadata(),
        candidate_stage="canary",
        champion_stage="canary",
    )

    assert rejected["accepted"] is False
    assert accepted["accepted"] is True


def test_same_stage_challenger_replaces_stale_training_data_contract() -> None:
    report = compare_candidate_to_champion(
        _metadata(avg=-0.10, lcb=-0.20, profit_factor=0.70),
        _metadata(training_data_version="2026-07-14.separated-profit-supervision.v4"),
        candidate_stage="canary",
        champion_stage="canary",
    )

    assert report["accepted"] is True
    assert report["reason"] == "training_data_contract_refresh"
    assert report["candidate_training_data_version"] == (
        "2026-09-15.authoritative-trade-integrity.v6"
    )
    assert report["champion_training_data_version"] == (
        "2026-07-14.separated-profit-supervision.v4"
    )


def test_shadow_challenger_can_replace_stale_non_active_champion() -> None:
    report = compare_candidate_to_champion(
        _metadata(avg=-0.10, lcb=-0.20, profit_factor=0.70),
        _metadata(training_data_version="2026-07-14.separated-profit-supervision.v4"),
        candidate_stage="shadow",
        champion_stage="canary",
    )

    assert report["accepted"] is True
    assert report["reason"] == "training_data_contract_refresh"


def test_active_candidate_requires_multiple_profitable_walk_forward_windows() -> None:
    champion = _metadata()
    challenger = _metadata(
        avg=0.35,
        lcb=0.15,
        profit_factor=1.50,
        cvar=-0.18,
        drawdown=0.35,
    )
    challenger["walk_forward_report"]["folds"] = challenger[
        "walk_forward_report"
    ]["folds"][:1]

    report = compare_candidate_to_champion(
        challenger,
        champion,
        candidate_stage="active",
        champion_stage="active",
    )

    assert report["accepted"] is False
    assert "active_candidate_cross_section_unstable" in report["blocking_reasons"]


def test_active_candidate_requires_market_regime_stability() -> None:
    champion = _metadata()
    challenger = _metadata(
        avg=0.35,
        lcb=0.15,
        profit_factor=1.50,
        cvar=-0.18,
        drawdown=0.35,
    )
    challenger["walk_forward_report"]["sides"]["short"][
        "market_regime_stability"
    ]["stable"] = False

    report = compare_candidate_to_champion(
        challenger,
        champion,
        candidate_stage="active",
        champion_stage="active",
    )

    assert report["accepted"] is False
    assert "active_candidate_cross_section_unstable" in report["blocking_reasons"]


def test_unprofitable_champion_can_be_replaced_by_clearly_better_candidate() -> None:
    # The live deadlock: the incumbent is losing money (profit factor 0.66,
    # negative average) yet blocks a candidate that improves every primary
    # metric, because its own tail metrics are marginally better.
    champion = _metadata(
        avg=-0.52,
        lcb=-1.30,
        profit_factor=0.66,
        cvar=-7.89,
        drawdown=39.31,
    )
    challenger = _metadata(
        avg=0.97,
        lcb=-0.50,
        profit_factor=1.56,
        cvar=-9.41,
        drawdown=40.48,
    )

    report = compare_candidate_to_champion(
        challenger,
        champion,
        candidate_stage="canary",
        champion_stage="canary",
    )

    assert report["accepted"] is True
    assert report["reason"] == "unhealthy_champion_controlled_takeover"
    assert report["unhealthy_champion_takeover"] is True


def test_unprofitable_champion_still_blocks_a_worse_candidate() -> None:
    champion = _metadata(
        avg=-0.52,
        lcb=-1.30,
        profit_factor=0.66,
        cvar=-7.89,
        drawdown=39.31,
    )
    challenger = _metadata(
        avg=0.20,
        lcb=-0.90,
        profit_factor=0.95,
        cvar=-7.90,
        drawdown=39.40,
    )

    report = compare_candidate_to_champion(
        challenger,
        champion,
        candidate_stage="canary",
        champion_stage="canary",
    )

    assert report["accepted"] is False
    assert "takeover_candidate_profit_factor_not_above_one" in report["blocking_reasons"]
    assert "takeover_candidate_avg_return_not_positive" not in report["blocking_reasons"]


def test_unprofitable_champion_blocks_catastrophic_tail_regression() -> None:
    champion = _metadata(
        avg=-0.52,
        lcb=-1.30,
        profit_factor=0.66,
        cvar=-7.89,
        drawdown=39.31,
    )
    challenger = _metadata(
        avg=1.20,
        lcb=-0.40,
        profit_factor=1.70,
        cvar=-14.00,
        drawdown=56.00,
    )

    report = compare_candidate_to_champion(
        challenger,
        champion,
        candidate_stage="canary",
        champion_stage="canary",
    )

    assert report["accepted"] is False
    assert "takeover_cvar_catastrophically_worsened" in report["blocking_reasons"]
    assert (
        "takeover_max_drawdown_catastrophically_worsened"
        in report["blocking_reasons"]
    )


def test_healthy_champion_is_not_displaced_without_real_improvement() -> None:
    # A profitable champion is only displaced by real improvement, and never by
    # the unhealthy-champion takeover path.
    champion = _metadata()
    no_improvement = compare_candidate_to_champion(
        _metadata(),
        champion,
        candidate_stage="canary",
        champion_stage="canary",
    )
    tail_worse = compare_candidate_to_champion(
        _metadata(avg=0.35, lcb=0.15, profit_factor=1.50, cvar=-0.40, drawdown=0.35),
        champion,
        candidate_stage="canary",
        champion_stage="canary",
    )

    assert no_improvement["accepted"] is False
    assert no_improvement.get("unhealthy_champion_takeover") is not True
    assert tail_worse["accepted"] is False
    assert "candidate_cvar_materially_worsened" in tail_worse["blocking_reasons"]
