from __future__ import annotations

from scripts.audit_online_shadow_pipeline import (
    _fingerprint,
    verify_shadow_evidence_fingerprint,
)


def _native_contract() -> tuple[dict, dict]:
    full = {
        "entry_fact": {"fact_id": "sha256:entry"},
        "result_fact": {"fact_id": "sha256:result"},
        "price_path": {"path_fingerprint": "sha256:path"},
    }
    compact = {
        "source": "shadow_entry_result_and_okx_native_1m_path",
        "executable_market_fact_verified": True,
        "entry_fact_id": "sha256:entry",
        "result_fact_id": "sha256:result",
        "path_fingerprint": "sha256:path",
    }
    compact["data_fingerprint"] = _fingerprint(
        {
            "entry_fact_id": compact["entry_fact_id"],
            "result_fact_id": compact["result_fact_id"],
            "path_fingerprint": compact["path_fingerprint"],
        }
    )
    return compact, full


def test_native_lineage_fingerprint_is_verified_without_ohlcv_evidence() -> None:
    compact, full = _native_contract()

    assert verify_shadow_evidence_fingerprint(compact, full, None) == (
        True,
        "native_lineage_verified",
    )


def test_native_lineage_mismatch_is_rejected() -> None:
    compact, full = _native_contract()
    full["result_fact"]["fact_id"] = "sha256:other"

    assert verify_shadow_evidence_fingerprint(compact, full, None) == (
        False,
        "native_lineage_mismatch",
    )


def test_historical_ohlcv_fingerprint_keeps_legacy_verification() -> None:
    evidence = {"bars": [{"open_time_ms": 1, "close": 2}]}
    contract = {
        "source": "preserved_market_klines_historical_path",
        "data_fingerprint": _fingerprint(evidence),
    }

    assert verify_shadow_evidence_fingerprint(contract, {}, evidence) == (
        True,
        "historical_evidence_verified",
    )
