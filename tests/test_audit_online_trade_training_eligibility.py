from __future__ import annotations

from scripts.audit_online_trade_training_eligibility import (
    REMOTE_SCRIPT,
    _build_remote_command,
    _compact_record,
    _summarize_records,
)


def test_compact_record_projects_annotated_trade_contract() -> None:
    sample = {
        "outcome_id": "ato:test",
        "lifecycle_key": "paper|BTC|1",
        "decision_id": 7,
        "symbol": "BTC/USDT",
        "side": "long",
        "data_quality_status": "excluded",
        "quality_reasons": ["feature_availability_contract_missing"],
        "outcome_evidence_gaps": [],
        "profit_supervision": {
            "version": "v4",
            "tasks": {
                "execution_cost_and_slippage_distribution": {
                    "eligible": True,
                    "source_authority": "okx_fills_fees_funding",
                    "fee_pct": 0.1,
                    "slippage_pct": 0.2,
                    "funding_return_pct": 0.0,
                }
            },
        },
        "features": {
            "feature_availability_contract": {
                "version": "availability-v1",
                "frozen_at": "2026-10-23T00:00:00Z",
            }
        },
    }

    record = _compact_record(sample)

    assert record["cost_eligible"] is True
    assert record["cost_source_authority"] == "okx_fills_fees_funding"
    assert record["feature_contract_version"] == "availability-v1"
    assert record["feature_contract_sealed"] is True
    assert record["quality_reasons"] == ["feature_availability_contract_missing"]


def test_summary_counts_missing_and_eligible_cost_facts() -> None:
    records = [
        _compact_record(
            {
                "profit_supervision": {
                    "tasks": {
                        "execution_cost_and_slippage_distribution": {
                            "eligible": False,
                            "reason": "authoritative_execution_cost_incomplete",
                        }
                    }
                }
            }
        ),
        _compact_record(
            {
                "profit_supervision": {
                    "version": "v4",
                    "tasks": {
                        "execution_cost_and_slippage_distribution": {
                            "eligible": True,
                            "source_authority": "okx_fills_fees_funding",
                            "fee_pct": 0.1,
                            "slippage_pct": 0.2,
                            "funding_return_pct": 0.0,
                        }
                    },
                },
                "features": {
                    "feature_availability_contract": {"version": "availability-v1"}
                },
            }
        ),
    ]

    summary = _summarize_records(records)

    assert summary["cost_reason_counts"] == {
        "authoritative_execution_cost_incomplete": 1,
        "none": 1,
    }
    assert summary["cost_source_counts"] == {
        "okx_fills_fees_funding": 1,
        "missing": 1,
    }
    assert summary["feature_contract_version_counts"] == {
        "availability-v1": 1,
        "missing": 1,
    }


def test_remote_command_embeds_the_tested_projection_helpers() -> None:
    command = _build_remote_command(25, "paper")

    assert "systemd-run --quiet --wait --pipe --collect" in command
    assert "/data/bb/app/.venv/bin/python -c" in command
    assert "__HELPERS__" in REMOTE_SCRIPT
