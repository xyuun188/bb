from __future__ import annotations

import json
from pathlib import Path

from scripts.replay_profit_integrity_incidents import build_replay_report

REPLAY_FIXTURE = Path(
    "tests/fixtures/profit_integrity/2026-07-15-current-contract-incident-replay.json"
)


def test_incident_replay_fails_old_profit_evidence_at_current_boundaries() -> None:
    report = build_replay_report()

    assert report["status"] == "passed"
    assert report["win_rate_used_for_acceptance"] is False

    robo = report["incidents"]["ROBO"]
    assert robo["incident_count"] == 6
    assert robo["all_invalid_entries_quarantined"] is True
    assert all(row["before"]["maximum_recorded_short_return_pct"] > 80 for row in robo["replays"])
    assert all(row["after"]["training_sample_weight"] == 0.0 for row in robo["replays"])
    assert all(row["after"]["production_contribution"] == 0.0 for row in robo["replays"])

    icp = report["incidents"]["ICP"]
    assert icp["before"]["lower_quantile_return_pct"] > icp["before"]["expected_return_pct"]
    assert "lower_quantile_above_raw_expected" in icp["after"]["distribution_blockers"]
    assert icp["after"]["production_contribution"] == 0.0
    assert icp["after"]["authoritative_return_after_all_cost_pct"] < 0.0
    assert icp["after"]["stop_execution_slippage"]["status"] == "measured"
    assert icp["after"]["stop_execution_slippage"]["contribution_usdt"] < 0.0

    doge = report["incidents"]["DOGE"]
    assert doge["before"]["return_uncertainty_pct"] == 0.0
    assert doge["after"]["distribution_member_count"] == 1
    assert doge["after"]["dispersion_pct"] > 0.0
    assert doge["after"]["uncertainty_penalty_pct"] > 0.0


def test_replay_is_deterministic_for_the_same_immutable_fixtures() -> None:
    assert build_replay_report() == build_replay_report()


def test_committed_incident_replay_evidence_matches_current_contracts() -> None:
    expected = json.loads(REPLAY_FIXTURE.read_text(encoding="utf-8"))

    assert build_replay_report() == expected
