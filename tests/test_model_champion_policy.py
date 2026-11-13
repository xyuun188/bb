import pytest

from services.ml_prediction_contract import ENTRY_COHORT_VERSION, PREDICTION_SCORE_POLICY_VERSION
from services.model_champion_policy import (
    PAIRED_COMPARISON_VERSION,
)
from services.model_champion_policy import (
    compare_candidate_to_champion as _compare_candidate_to_champion,
)


def _entry_policy_fixture(sides: dict) -> dict:
    """Synthetic scalar policy inputs, not an outcome replay or profitability proof."""
    rows = [sides.get(side, {}) for side in ("long", "short")]
    portfolio = {
        key: sum(row[key] for row in rows) / 2 if all(key in row for row in rows) else None
        for key in ("avg_return_pct", "return_lcb_pct", "cvar_10_pct", "max_drawdown_pct")
    }
    for key in ("count", "gross_profit_pct", "gross_loss_pct"):
        portfolio[key] = sum(row[key] for row in rows) if all(key in row for row in rows) else None
    portfolio["profit_factor_status"] = (
        "no_loss_denominator" if portfolio["gross_loss_pct"] == 0 else "defined"
    )
    return {
        "version": ENTRY_COHORT_VERSION,
        "prediction_score_policy_version": PREDICTION_SCORE_POLICY_VERSION,
        "sides": sides, "portfolio": portfolio,
    }


def _paired_evaluation(candidate: dict, champion: dict) -> dict:
    return {
        "version": PAIRED_COMPARISON_VERSION,
        "ready": True,
        "sample_count": 100,
        "chronological_label_disjoint": True,
        "evaluation_data_sha256": "c" * 64,
        "candidate_training_data_sha256": candidate["training_data_sha256"],
        "champion_training_data_sha256": champion["training_data_sha256"],
        "candidate": candidate["oos_return_evaluation"],
        "champion": champion["oos_return_evaluation"],
        "candidate_entry_evaluation": _entry_policy_fixture(candidate["oos_return_evaluation"]),
        "champion_entry_evaluation": _entry_policy_fixture(champion["oos_return_evaluation"]),
    }


def compare_candidate_to_champion(candidate: dict, champion: dict, **kwargs: object) -> dict:
    """Exercise metric policy with scores from a common evaluation fixture."""
    return _compare_candidate_to_champion(
        candidate, champion,
        comparison_evaluation=_paired_evaluation(candidate, champion),
        **kwargs,
    )


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
        "count": 4,
        "avg_return_pct": avg,
        "return_lcb_pct": lcb,
        "profit_factor": profit_factor,
        "gross_profit_pct": profit_factor,
        "gross_loss_pct": 1.0,
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


def _directional_metadata() -> tuple[dict, dict]:
    champion = _metadata(
        avg=-1.0101, lcb=-1.8260, profit_factor=0.3664, cvar=-9.3241, drawdown=58.3329
    )
    champion["oos_return_evaluation"]["short"] = _metadata(
        avg=-0.0371, lcb=-0.7743, profit_factor=0.9613, cvar=-6.4525, drawdown=20.2953
    )["oos_return_evaluation"]["short"]
    candidate = _metadata(
        avg=1.7550, lcb=0.4815, profit_factor=2.5740, cvar=-6.7702, drawdown=30.9365
    )
    candidate["oos_return_evaluation"]["short"] = _metadata(
        avg=-0.8088, lcb=-1.9869, profit_factor=0.5941, cvar=-12.3683, drawdown=77.8838
    )["oos_return_evaluation"]["short"]
    return champion, candidate


def test_canary_compares_authorized_side_without_pooling_disabled_direction() -> None:
    champion, candidate = _directional_metadata()

    report = compare_candidate_to_champion(
        candidate,
        champion,
        candidate_stage="canary",
        champion_stage="canary",
        candidate_enabled_sides=["long"],
    )

    assert report["accepted"] is True
    assert report["reason"] == "challenger_quality_improved"
    assert report["comparison_sides"] == ["long"]
    assert report["candidate_metrics"]["return_lcb_pct"] == 0.4815
    assert report["champion_metrics"]["max_drawdown_pct"] == 58.3329


def test_authorized_losing_side_cannot_hide_behind_other_side_improvement() -> None:
    champion, candidate = _directional_metadata()

    report = compare_candidate_to_champion(
        candidate,
        champion,
        candidate_stage="canary",
        champion_stage="canary",
        candidate_enabled_sides=["short"],
    )

    assert report["accepted"] is False
    assert report["comparison_sides"] == ["short"]
    assert "candidate_cvar_materially_worsened" in report["blocking_reasons"]
    assert "candidate_max_drawdown_materially_worsened" in report["blocking_reasons"]


def test_unprofitable_champion_does_not_widen_tail_tolerance() -> None:
    champion, candidate = _directional_metadata()

    report = compare_candidate_to_champion(
        candidate, champion, candidate_stage="canary", champion_stage="canary"
    )

    assert report["accepted"] is False
    assert report["comparison_sides"] == ["long", "short"]
    assert "candidate_max_drawdown_materially_worsened" in report["blocking_reasons"]


def test_shadow_model_can_still_improve_without_claiming_profitability() -> None:
    report = compare_candidate_to_champion(
        _metadata(avg=-0.10, lcb=-0.20, profit_factor=0.90),
        _metadata(avg=-0.20, lcb=-0.30, profit_factor=0.80),
        candidate_stage="shadow",
        champion_stage="shadow",
    )

    assert report["accepted"] is True
    assert report["reason"] == "challenger_quality_improved"


@pytest.mark.parametrize("sides", [[], ["neutral"], ["long", "unknown"]])
def test_invalid_canary_side_scope_is_rejected(sides: list[str]) -> None:
    report = compare_candidate_to_champion(
        _metadata(),
        _metadata(),
        candidate_stage="canary",
        champion_stage="canary",
        candidate_enabled_sides=sides,
    )

    assert report["accepted"] is False
    assert report["blocking_reasons"] == ["candidate_enabled_sides_invalid"]


@pytest.mark.parametrize("stage", ["active", "live"])
def test_active_model_cannot_skip_short_side_by_declaring_long_only(stage: str) -> None:
    champion, candidate = _directional_metadata()

    report = compare_candidate_to_champion(
        candidate,
        champion,
        candidate_stage=stage,
        champion_stage=stage,
        candidate_enabled_sides=["long"],
    )

    assert report["accepted"] is False
    assert report["comparison_sides"] == ["long", "short"]
    assert "candidate_max_drawdown_worsened" in report["blocking_reasons"]


@pytest.mark.parametrize("stage", ["active", "live"])
def test_training_contract_refresh_cannot_bypass_active_champion_quality(stage: str) -> None:
    report = compare_candidate_to_champion(
        _metadata(avg=-0.10, lcb=-0.20, profit_factor=0.70),
        _metadata(training_data_version="old-training-contract"),
        candidate_stage=stage,
        champion_stage=stage,
    )

    assert report["accepted"] is False
    assert report["reason"] == "active_champion_retained"
    assert "candidate_return_lcb_not_positive" in report["blocking_reasons"]


@pytest.mark.parametrize("stage", ["active", "live"])
def test_live_alias_requires_the_same_stability_as_active(stage: str) -> None:
    candidate = _metadata()
    candidate["walk_forward_report"]["folds"] = []
    report = compare_candidate_to_champion(
        candidate,
        _metadata(),
        candidate_stage=stage,
        champion_stage="canary",
    )

    assert report["accepted"] is False
    assert report["blocking_reasons"] == ["active_candidate_cross_section_unstable"]


@pytest.mark.parametrize("missing_side", ["long", "short"])
def test_both_side_comparison_rejects_missing_direction_evidence(missing_side: str) -> None:
    candidate = _metadata(avg=0.35)
    del candidate["oos_return_evaluation"][missing_side]
    report = compare_candidate_to_champion(
        candidate,
        _metadata(),
        candidate_stage="canary",
        champion_stage="canary",
    )

    assert report["accepted"] is False
    assert "champion_comparison_metric_missing:avg_return_pct" in report["blocking_reasons"]


def test_different_historical_reports_cannot_replace_common_evaluation() -> None:
    report = _compare_candidate_to_champion(
        _metadata(avg=10, lcb=9, profit_factor=20),
        _metadata(),
        candidate_stage="shadow",
        champion_stage="shadow",
    )
    assert report["accepted"] is False
    assert "common_unfitted_holdout_unavailable" in report["blocking_reasons"]


@pytest.mark.parametrize("loss", [0.0, 2.0])
def test_comparison_pools_profits_including_loss_free_direction(loss: float) -> None:
    from services.ml_signal_service import _return_evidence

    candidate, champion = _metadata(), _metadata()
    candidate["oos_return_evaluation"] = {
        side: _return_evidence([
            {"return_pct": value, "label_timestamp": str(index)}
            for index, value in enumerate(values)
        ])
        for side, values in (("long", [-loss, 1.0]), ("short", [0.3, 6.7]))
    }
    report = compare_candidate_to_champion(
        candidate, champion, candidate_stage="shadow", champion_stage="shadow"
    )
    if loss:
        assert report["candidate_metrics"]["profit_factor"] == pytest.approx(4.0)
        assert "candidate_cvar_materially_worsened" in report["blocking_reasons"]
    else:
        assert report["candidate_metrics"]["profit_factor"] is None
        assert "candidate_profit_factor_no_loss_denominator" in report["blocking_reasons"]
    assert report["accepted"] is False


def test_paired_results_override_stale_historical_metrics() -> None:
    candidate, champion = _metadata(avg=-10), _metadata(avg=10)
    evaluation = _paired_evaluation(_metadata(avg=0.4), _metadata(avg=0.3))
    report = _compare_candidate_to_champion(
        candidate, champion,
        candidate_stage="shadow", champion_stage="shadow",
        comparison_evaluation=evaluation,
    )
    assert report["accepted"] is True
    assert report["candidate_metrics"]["avg_return_pct"] == 0.4
    assert report["champion_metrics"]["avg_return_pct"] == 0.3


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("chronological_label_disjoint", False),
        ("candidate_training_data_sha256", "wrong-candidate"),
        ("champion_training_data_sha256", "wrong-champion"),
        ("sample_count", 1),
        ("evaluation_data_sha256", ""),
    ],
)
def test_invalid_paired_evidence_preserves_champion(key: str, value: object) -> None:
    candidate, champion = _metadata(avg=0.4), _metadata()
    evaluation = {**_paired_evaluation(candidate, champion), key: value}
    report = _compare_candidate_to_champion(
        candidate, champion,
        candidate_stage="canary", champion_stage="canary",
        comparison_evaluation=evaluation,
    )
    assert report["accepted"] is False
    assert report["reason"] == "paired_comparison_unavailable"


@pytest.mark.parametrize("empty", [False, True])
def test_comparison_uses_pooled_entry_outcomes_with_empty_direction(empty: bool) -> None:
    from services.ml_signal_service import _return_evidence

    candidate, champion = _metadata(), _metadata()
    evaluation = _paired_evaluation(candidate, champion)
    for role, returns in (("candidate", [-0.2, 1.0, 0.5]), ("champion", [-0.3, 0.6, 0.4])):
        portfolio = _return_evidence([
            {"return_pct": value, "label_timestamp": str(index)}
            for index, value in enumerate([] if empty else returns)
        ])
        evaluation[f"{role}_entry_evaluation"] = {
            "version": ENTRY_COHORT_VERSION,
            "prediction_score_policy_version": PREDICTION_SCORE_POLICY_VERSION,
            "sides": {"long": _return_evidence([]), "short": portfolio},
            "portfolio": portfolio,
        }
    report = _compare_candidate_to_champion(
        candidate, champion, candidate_stage="shadow", champion_stage="shadow",
        comparison_evaluation=evaluation,
    )
    assert report["accepted"] is (not empty)
    if not empty:
        assert report["candidate_metrics"]["avg_return_pct"] == pytest.approx(1.3 / 3)
        assert report["candidate_metrics"]["profit_factor"] == pytest.approx(7.5)


@pytest.mark.parametrize("version", [None, "old-ranking-report"])
def test_ranking_evidence_cannot_replace_entry_portfolio(version) -> None:
    candidate, champion = _metadata(avg=0.4), _metadata()
    evaluation = _paired_evaluation(candidate, champion)
    evaluation["candidate_entry_evaluation"]["version"] = version
    report = _compare_candidate_to_champion(
        candidate, champion, candidate_stage="shadow", champion_stage="shadow",
        comparison_evaluation=evaluation,
    )
    assert report["accepted"] is False


@pytest.mark.parametrize("role", ["candidate", "champion"])
@pytest.mark.parametrize("policy", [None, "legacy-tree-mean-only"])
def test_comparison_rejects_outcomes_from_another_scoring_policy(role, policy) -> None:
    candidate, champion = _metadata(avg=0.4), _metadata()
    evaluation = _paired_evaluation(candidate, champion)
    evaluation[f"{role}_entry_evaluation"]["prediction_score_policy_version"] = policy
    report = _compare_candidate_to_champion(
        candidate, champion, candidate_stage="shadow", champion_stage="shadow",
        comparison_evaluation=evaluation,
    )
    assert report["accepted"] is False
    assert f"{role}_metrics" in report
    assert report[f"{role}_metrics"]["profit_factor"] is None
