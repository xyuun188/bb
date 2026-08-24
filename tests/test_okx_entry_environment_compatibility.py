from services.okx_entry_environment_compatibility import (
    assess_okx_entry_environment_compatibility,
)


def _instrument(**overrides: str) -> dict[str, str]:
    row = {
        "instId": "BTC-USDT-SWAP",
        "uly": "BTC-USDT",
        "ctVal": "0.01",
        "ctMult": "1",
        "ctValCcy": "BTC",
        "settleCcy": "USDT",
        "ctType": "linear",
        "lotSz": "0.01",
        "minSz": "0.01",
        "tickSz": "0.1",
    }
    row.update(overrides)
    return row


def _ticker(price: str) -> dict:
    return {"data": [{"instId": "BTC-USDT-SWAP", "last": price}]}


def test_matching_live_and_execution_contract_is_compatible() -> None:
    result = assess_okx_entry_environment_compatibility(
        live_instrument=_instrument(),
        execution_instrument=_instrument(),
        live_ticker=_ticker("60000"),
        execution_ticker=_ticker("59950"),
    )

    assert result["compatible"] is True
    assert result["blockers"] == []
    assert result["price_drift_fraction"] < 0.01


def test_same_economic_contract_with_demo_underlying_alias_is_compatible() -> None:
    result = assess_okx_entry_environment_compatibility(
        live_instrument=_instrument(),
        execution_instrument=_instrument(uly="BTC1-USDT"),
        live_ticker=_ticker("60000"),
        execution_ticker=_ticker("60000"),
    )

    assert result["compatible"] is True
    assert result["blockers"] == []
    assert result["warnings"] == ["uly_alias_mismatch"]
    assert result["identity_alias_differences"] == ["uly_alias_mismatch"]


def test_contract_value_mismatch_remains_an_identity_blocker() -> None:
    result = assess_okx_entry_environment_compatibility(
        live_instrument=_instrument(),
        execution_instrument=_instrument(ctVal="1"),
        live_ticker=_ticker("60000"),
        execution_ticker=_ticker("60000"),
    )

    assert result["compatible"] is False
    assert "ctVal_mismatch" in result["blockers"]


def test_material_live_demo_price_divergence_is_blocked() -> None:
    result = assess_okx_entry_environment_compatibility(
        live_instrument=_instrument(),
        execution_instrument=_instrument(),
        live_ticker=_ticker("100"),
        execution_ticker=_ticker("97"),
    )

    assert result["compatible"] is False
    assert "environment_price_drift_exceeded" in result["blockers"]


def test_execution_step_differences_are_diagnostic_not_identity_blockers() -> None:
    result = assess_okx_entry_environment_compatibility(
        live_instrument=_instrument(lotSz="1", minSz="1", tickSz="0.1"),
        execution_instrument=_instrument(lotSz="0.1", minSz="0.1", tickSz="0.01"),
        live_ticker=_ticker("100"),
        execution_ticker=_ticker("100"),
    )

    assert result["compatible"] is True
    assert result["blockers"] == []
    assert result["operational_rule_differences"] == [
        "lotSz_mismatch",
        "minSz_mismatch",
        "tickSz_mismatch",
    ]


def test_missing_execution_fact_fails_closed() -> None:
    result = assess_okx_entry_environment_compatibility(
        live_instrument=_instrument(),
        execution_instrument={},
        live_ticker=_ticker("100"),
        execution_ticker={},
    )

    assert result["compatible"] is False
    assert "execution_instrument_missing" in result["blockers"]
    assert "environment_price_missing" in result["blockers"]
