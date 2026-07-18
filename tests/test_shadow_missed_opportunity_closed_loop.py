from types import SimpleNamespace

from services.shadow_missed_opportunity_closed_loop import summarize_shadow_missed_opportunities
from tests.paper_canary_fixtures import complete_paper_canary_raw


def test_missed_opportunities_are_return_observations_only() -> None:
    rows = [
        SimpleNamespace(
            status="completed",
            missed_opportunity=True,
            symbol="BTC/USDT",
            best_action="long",
            long_return_pct=value,
        )
        for value in (0.4, 0.8, -0.1)
    ]
    report = summarize_shadow_missed_opportunities(rows)

    observation = report["return_observations"][0]
    assert observation["sample_count"] == 3
    assert observation["return_distribution_provenance"]["source"].startswith(
        "empirical_order_statistics"
    )
    assert observation["can_authorize_entry"] is False
    assert report["usable_group_count"] == 0


def test_executed_entry_without_positive_return_contract_is_reported() -> None:
    decision = SimpleNamespace(
        id=7,
        action="long",
        was_executed=True,
        raw_llm_response={"production_return_policy": {"eligible": False}},
    )
    report = summarize_shadow_missed_opportunities([], decisions=[decision])

    assert report["summary"]["executed_return_contract_gap_count"] == 1
    assert report["global_missed_count_can_drive_entries"] is False


def test_valid_executed_canary_is_not_reported_as_a_production_contract_gap() -> None:
    decision = SimpleNamespace(
        id=8,
        action="long",
        was_executed=True,
        raw_llm_response=complete_paper_canary_raw(),
    )

    report = summarize_shadow_missed_opportunities([], decisions=[decision])

    assert report["summary"]["executed_return_contract_gap_count"] == 0


def test_malformed_executed_canary_is_reported_with_its_lifecycle() -> None:
    raw = complete_paper_canary_raw()
    raw["paper_bootstrap_canary"]["runtime_authorized"] = False
    decision = SimpleNamespace(
        id=9,
        action="long",
        was_executed=True,
        raw_llm_response=raw,
    )

    report = summarize_shadow_missed_opportunities([], decisions=[decision])

    assert report["summary"]["executed_return_contract_gap_count"] == 1
    assert report["executed_return_contract_gaps"][0]["contract_lifecycle"] == (
        "paper_bootstrap_canary"
    )
