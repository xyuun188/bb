from datetime import UTC, datetime, timedelta

from services.training_effectiveness_report import (
    TRAINING_EFFECTIVENESS_REPORT_VERSION,
    TrainingEffectivenessReportService,
    build_input_fingerprint,
    calculate_fee_after_return,
    calculate_metric_delta,
    classify_sample_authority,
    validate_report,
)


def test_fee_after_return_uses_the_authoritative_cost_equation():
    assert calculate_fee_after_return(10, 1.5, 0.5, 2) == 10.0
    assert calculate_fee_after_return(-10, 1.5, 0.5, -2) == -14.0


def test_input_fingerprint_is_stable_for_mapping_order():
    left = build_input_fingerprint({"b": [2, 1], "a": {"z": 3, "y": 4}})
    right = build_input_fingerprint({"a": {"y": 4, "z": 3}, "b": [2, 1]})
    assert left == right
    assert left.startswith("sha256:")


def test_metric_delta_avoids_infinite_percentage():
    result = calculate_metric_delta(0, 2, 0)
    assert result["active_vs_challenger"] == {"absolute": 2.0, "percentage": None}


def test_sample_authority_has_only_the_four_contract_values():
    assert classify_sample_authority({"shadow": True}) == "shadow_opportunity"
    assert classify_sample_authority({"counterfactual": True}) == "counterfactual_cost"
    assert classify_sample_authority({"source": "okx_position_history_realized_pnl", "outcome_complete": True}) == "okx_realized"
    assert classify_sample_authority({"excluded": True}) == "excluded"


def test_service_returns_partial_without_realized_samples_and_no_side_effects():
    calls = []

    def registry():
        calls.append("registry")
        return {"models": [{"model_id": "active-v1", "lifecycle": "active"}]}

    report = __import__("asyncio").run(
        TrainingEffectivenessReportService(registry_provider=registry).build(
            filters={"to": "2026-08-25T00:00:00Z"}, run_id="fixed-run"
        )
    )
    assert report["status"] == "partial"
    assert "no_okx_realized_samples" in report["conclusion"]["blocking_reasons"]
    assert calls == ["registry"]


def test_validate_report_rejects_future_cutoff_and_bad_costs():
    now = datetime.now(UTC).replace(microsecond=0)
    report = {
        "report_version": TRAINING_EFFECTIVENESS_REPORT_VERSION,
        "report_id": "te-fixed",
        "generated_at": now.isoformat(),
        "data_cutoff_at": (now + timedelta(minutes=1)).isoformat(),
        "status": "partial",
        "input_fingerprint": "sha256:" + "0" * 64,
        "run": {}, "versions": {}, "filters": {}, "metrics": {},
        "cost_attribution": {"gross_pnl": 1, "fee": 1, "slippage": 0, "funding_fee": 0, "fee_after_net_pnl": 99},
        "expert_contributions": [], "execution_funnel": {}, "sample_quality": {}, "conclusion": {}, "freshness": {},
    }
    errors = validate_report(report)
    assert "invalid:time_order" in errors
    assert "invalid:cost_attribution_equation" in errors
