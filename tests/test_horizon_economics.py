from __future__ import annotations

import pytest

from scripts import analyze_horizon_economics
from services.horizon_economics import (
    assess_baseline_stability,
    evaluate_horizon_economics,
    primary_horizon_verdict,
    required_hit_rate,
)


def _baselines(**horizons: object) -> dict[str, object]:
    return horizons


def _view(*, gross: float, cost: float, net: float, lcb: float, pf: float, n: int = 264):
    return {
        "count": n,
        "mean_selected_gross_return_pct": gross,
        "mean_selected_execution_cost_pct": cost,
        "avg_return_pct": net,
        "return_lcb_pct": lcb,
        "profit_factor": pf,
    }


def test_five_minute_horizon_cannot_clear_cost_when_edge_is_far_below_cost() -> None:
    report = evaluate_horizon_economics(
        _baselines(
            **{
                "5": {
                    "long": {
                        "nonoverlapping": _view(
                            gross=0.0111, cost=0.1530, net=-0.1419, lcb=-0.1901, pf=0.433
                        )
                    }
                }
            }
        )
    )

    row = report["rows"][0]
    assert row["viable"] is False
    assert row["gross_edge_to_cost_ratio"] == pytest.approx(0.0725, abs=1e-3)
    assert report["conclusion"] == "no_horizon_clears_cost_in_this_sample"


def test_long_horizon_with_positive_after_cost_lcb_is_reported_viable() -> None:
    report = evaluate_horizon_economics(
        _baselines(
            **{
                "240": {
                    "short": {
                        "nonoverlapping": _view(
                            gross=0.7342, cost=0.1572, net=0.5770, lcb=0.2232, pf=1.798
                        )
                    }
                }
            }
        )
    )

    row = report["rows"][0]
    assert row["viable"] is True
    assert report["viable_horizons"][0]["horizon_minutes"] == "240"
    assert report["conclusion"] == "at_least_one_horizon_clears_cost"
    assert report["production_permission"] is False
    assert report["orders_submitted"] is False


def test_positive_mean_without_positive_lcb_is_not_viable() -> None:
    report = evaluate_horizon_economics(
        _baselines(
            **{
                "60": {
                    "long": {
                        "nonoverlapping": _view(
                            gross=0.20, cost=0.15, net=0.05, lcb=-0.02, pf=1.05
                        )
                    }
                }
            }
        )
    )

    assert report["rows"][0]["viable"] is False


def test_missing_gross_or_cost_never_becomes_viable() -> None:
    report = evaluate_horizon_economics(
        _baselines(
            **{
                "15": {
                    "long": {
                        "nonoverlapping": {
                            "count": 10,
                            "avg_return_pct": 0.5,
                            "return_lcb_pct": 0.1,
                            "profit_factor": 2.0,
                        }
                    }
                }
            }
        )
    )

    assert report["rows"][0]["viable"] is False


def test_required_hit_rate_increases_with_cost() -> None:
    cheap = required_hit_rate(0.5, 0.1)
    expensive = required_hit_rate(0.5, 0.3)

    assert cheap == pytest.approx(0.6)
    assert expensive == pytest.approx(0.8)
    assert required_hit_rate(0.0, 0.1) is None
    assert required_hit_rate(0.5, -0.1) is None


def test_single_positive_aggregate_is_not_stable_across_folds() -> None:
    positive = {
        "240": {
            "short": {
                "nonoverlapping": _view(
                    gross=0.734, cost=0.157, net=0.577, lcb=0.223, pf=1.798
                )
            }
        }
    }
    negative = {
        "240": {
            "short": {
                "nonoverlapping": _view(
                    gross=-0.10, cost=0.157, net=-0.257, lcb=-0.41, pf=0.6
                )
            }
        }
    }

    report = assess_baseline_stability([positive, negative, positive])

    item = report["stability"][0]
    assert item["positive_fold_count"] == 2
    assert item["stable"] is False
    assert report["stable_horizons"] == []


def test_all_positive_folds_mark_horizon_stable() -> None:
    positive = {
        "240": {
            "short": {
                "nonoverlapping": _view(
                    gross=0.734, cost=0.157, net=0.577, lcb=0.223, pf=1.798
                )
            }
        }
    }

    report = assess_baseline_stability([positive, positive, positive])

    assert report["stability"][0]["stable"] is True
    assert report["stable_horizons"] == [
        {"horizon_minutes": "240", "side": "short"}
    ]


def test_missing_fold_baselines_never_report_stable() -> None:
    report = assess_baseline_stability([{}, {}, {}])

    assert report["stability"] == []
    assert report["stable_horizons"] == []


def test_five_minute_primary_horizon_is_infeasible_when_edge_is_below_cost() -> None:
    report = primary_horizon_verdict(
        _baselines(
            **{
                "5": {
                    "long": {
                        "nonoverlapping": _view(
                            gross=0.0111, cost=0.1530, net=-0.1419, lcb=-0.1901, pf=0.433
                        )
                    },
                    "short": {
                        "nonoverlapping": _view(
                            gross=-0.0111, cost=0.1530, net=-0.1641, lcb=-0.2107, pf=0.360
                        )
                    },
                }
            }
        ),
        primary_horizon_minutes=5,
    )

    assert report["economically_feasible"] is False
    assert "primary_horizon_gross_edge_below_execution_cost" in report["reason_codes"]
    assert "primary_horizon_fee_after_lcb_not_positive" in report["reason_codes"]


def test_primary_horizon_with_positive_lcb_is_feasible() -> None:
    report = primary_horizon_verdict(
        _baselines(
            **{
                "240": {
                    "short": {
                        "nonoverlapping": _view(
                            gross=0.734, cost=0.157, net=0.577, lcb=0.223, pf=1.798
                        )
                    }
                }
            }
        ),
        primary_horizon_minutes=240,
    )

    assert report["economically_feasible"] is True
    assert report["reason_codes"] == []


def test_missing_primary_horizon_evidence_is_not_feasible() -> None:
    report = primary_horizon_verdict({}, primary_horizon_minutes=5)

    assert report["economically_feasible"] is False
    assert report["reason_codes"] == ["primary_horizon_economics_evidence_missing"]


def test_side_labeled_outer_evidence_feeds_fold_stability(tmp_path, monkeypatch) -> None:
    import json

    evidence = _view(gross=0.734, cost=0.157, net=0.577, lcb=0.223, pf=1.798)
    negative = _view(gross=-0.734, cost=0.157, net=-0.891, lcb=-1.238, pf=0.399)
    baselines = {
        "240": {
            "long": {"nonoverlapping": negative},
            "short": {"nonoverlapping": evidence},
        }
    }
    report = {
        "research": {
            "fixed_baselines_by_horizon": baselines,
            "folds": [
                {
                    "fixed_baselines_by_horizon": baselines,
                    "outer_evidence": {
                        "all_opportunities": evidence,
                        "nonoverlapping": evidence,
                    },
                }
            ] * 3,
        }
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    output: list[str] = []
    monkeypatch.setattr(
        analyze_horizon_economics, "_format", lambda value, _indent: json.dumps(value)
    )
    monkeypatch.setattr(analyze_horizon_economics, "safe_print", output.append)
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_horizon_economics.py", "--report", str(path)],
    )

    analyze_horizon_economics._main()

    payload = json.loads(output[0])
    assert payload["baseline_stability"]["stable_horizons"] == [
        {"horizon_minutes": "240", "side": "short"}
    ]


def test_mixed_selected_outer_evidence_never_proves_side_stability(
    tmp_path, monkeypatch
) -> None:
    import json

    evidence = _view(gross=0.734, cost=0.157, net=0.577, lcb=0.223, pf=1.798)
    report = {
        "research": {
            "fixed_baselines_by_horizon": {
                "240": {
                    "short": {"nonoverlapping": evidence},
                }
            },
            "folds": [
                {
                    "selected_policy": {"horizon_minutes": 240},
                    "aggregate_selected_evidence": evidence,
                    "outer_evidence": {
                        "all_opportunities": evidence,
                        "nonoverlapping": evidence,
                    },
                }
            ] * 3,
        }
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    output: list[str] = []
    monkeypatch.setattr(
        analyze_horizon_economics, "_format", lambda value, _indent: json.dumps(value)
    )
    monkeypatch.setattr(analyze_horizon_economics, "safe_print", output.append)
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_horizon_economics.py", "--report", str(path)],
    )

    analyze_horizon_economics._main()

    payload = json.loads(output[0])
    assert payload["baseline_stability"]["stable_horizons"] == []
