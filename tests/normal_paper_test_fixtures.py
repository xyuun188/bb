"""Shared v6 evidence for tests that intentionally authorize normal paper entries."""

from __future__ import annotations


def paper_quality_permissions(*, source: str = "local_ml") -> dict[str, dict]:
    return {
        source: {
            "paper_execution_permission": True,
            "paper_execution_reason": "authoritative_fee_after_quality_above_break_even",
            "paper_execution_evidence_source": "test_authoritative_trade",
            "paper_execution_evidence": {
                "sample_count": 20,
                "average_return": 0.2,
                "return_lcb": 0.1,
                "profit_factor": 1.5,
                "profit_factor_above_break_even": True,
            },
            "break_even_contract": {
                "average_return_above_zero": True,
                "return_lcb_above_zero": True,
                "profit_factor_above_one": True,
            },
        }
    }
