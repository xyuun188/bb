from core.symbols import okx_inst_id_from_payload, symbol_from_okx_market, symbol_from_okx_payload


def test_okx_inst_id_wins_over_ccxt_alias_for_h_contract() -> None:
    market = {
        "symbol": "WLFI/USDT:USDT",
        "id": "H-USDT-SWAP",
        "base": "WLFI",
        "info": {
            "instId": "H-USDT-SWAP",
            "uly": "WLFI-USDT",
            "instFamily": "H-USDT",
            "ctValCcy": "H",
        },
    }

    assert symbol_from_okx_market(market) == "H/USDT"


def test_okx_order_payload_inst_id_wins_over_ccxt_alias() -> None:
    payload = {
        "symbol": "WLFI/USDT:USDT",
        "info": {"instId": "H-USDT-SWAP"},
    }

    assert symbol_from_okx_payload(payload, fallback="WLFI/USDT") == "H/USDT"


def test_okx_order_payload_does_not_parse_order_id_as_symbol() -> None:
    payload = {
        "id": "exit-1",
        "symbol": "USAR/USDT:USDT",
        "info": {"ordId": "exit-1"},
    }

    assert symbol_from_okx_payload(payload, fallback="USAR/USDT") == "USAR/USDT"


def test_okx_inst_id_from_payload_prefers_nested_okx_fact() -> None:
    payload = {
        "symbol": "WLFI/USDT:USDT",
        "native_close_fill": {"order_info": {"instId": "H-USDT-SWAP"}},
    }

    assert okx_inst_id_from_payload(payload, fallback="WLFI/USDT") == "H-USDT-SWAP"
    assert okx_inst_id_from_payload({}, fallback="BTC/USDT") == "BTC-USDT-SWAP"


def test_okx_inst_id_from_payload_strict_mode_ignores_display_fallbacks() -> None:
    payload = {
        "symbol": "SAHARA/USDT:USDT",
        "canonical_exchange_symbol": "SAHARA/USDT",
    }

    assert okx_inst_id_from_payload(payload, fallback="SPK/USDT") == "SPK-USDT-SWAP"
    assert okx_inst_id_from_payload(payload, fallback="SPK/USDT", include_fallback=False) == ""
