from __future__ import annotations

from services.okx_error_classifier import (
    is_okx_entry_instrument_unavailable,
    is_okx_temporary_service_error,
)


def test_compliance_restriction_is_a_durable_entry_capability_failure() -> None:
    assert is_okx_entry_instrument_unavailable(
        "OKX API error [51155]: You can't trade this pair due to local compliance restrictions."
    )


def test_structured_execution_error_fields_use_the_same_classifier() -> None:
    assert is_okx_entry_instrument_unavailable(
        {
            "error_code": "51155",
            "raw_error": "Instrument suspended for trading",
        }
    )


def test_close_only_no_position_error_never_blacklists_entry_instrument() -> None:
    value = "OKX 51169: You don't have any positions in this direction."
    assert not is_okx_entry_instrument_unavailable(value)


def test_local_market_cache_error_never_blacklists_entry_instrument() -> None:
    value = "OKX SDK market is not loaded: LINEA/USDT:USDT"
    assert not is_okx_entry_instrument_unavailable(value)


def test_temporary_okx_service_error_is_not_a_durable_instrument_failure() -> None:
    value = "OKX API error [50001]: Service temporarily unavailable."
    assert is_okx_temporary_service_error(value)
    assert not is_okx_entry_instrument_unavailable(value)
