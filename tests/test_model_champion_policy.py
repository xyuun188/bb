from services.model_champion_policy import compare_candidate_to_champion


def _metadata(
    *,
    avg: float = 0.30,
    lcb: float = 0.12,
    profit_factor: float = 1.40,
    cvar: float = -0.20,
    drawdown: float = 0.40,
    training_data_version: str = "2026-07-21.authoritative-trade-integrity.v5",
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
        "2026-07-21.authoritative-trade-integrity.v5"
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
