import json
from pathlib import Path

from core.market_facts import (
    MARKET_FACT_CONTRACT_VERSION,
    MARKET_SOURCE_CONSISTENCY_VERSION,
    build_market_fact,
    build_market_source_consistency,
    build_shadow_market_fact_contract,
    market_fact_contract_reasons,
    verify_market_fact_path,
)
from data_feed.feature_vector import build_feature_vector


def _spec(inst_id: str = "ROBO-USDT-SWAP") -> dict[str, str]:
    return {
        "instId": inst_id,
        "instType": "SWAP",
        "uly": inst_id.removesuffix("-SWAP"),
        "instFamily": inst_id.removesuffix("-SWAP"),
        "ctType": "linear",
        "ctVal": "1",
        "ctMult": "1",
        "ctValCcy": inst_id.split("-")[0],
        "settleCcy": "USDT",
        "lotSz": "1",
        "minSz": "1",
        "tickSz": "0.00001",
        "state": "live",
    }


def _snapshot(price: float, timestamp: int, *, notional: float = 10_000.0) -> dict:
    return {
        "symbol": "ROBO/USDT",
        "inst_id": "ROBO-USDT-SWAP",
        "inst_type": "SWAP",
        "source": "rest",
        "source_endpoint": "okx_rest_market_ticker",
        "source_channel": "tickers",
        "timestamp": timestamp,
        "last_price": price,
        "bid": price - 0.00001,
        "ask": price + 0.00001,
        "notional_24h_usdt": notional,
        "volume_24h_contracts": 50_000,
        "volume_24h_base": 50_000,
        "orderbook_bid_depth": 25_000,
        "orderbook_ask_depth": 25_000,
    }


def test_zero_turnover_robo_fact_is_quarantined_without_fixed_price_threshold() -> None:
    fact = build_market_fact(
        "ROBO/USDT",
        _snapshot(0.10834, 1_783_990_800_000, notional=0.0),
        contract_spec=_spec(),
    )

    assert fact["quality"]["status"] == "quarantined"
    assert "zero_notional_turnover" in fact["quality"]["reasons"]
    assert fact["native_identity"]["inst_id"] == "ROBO-USDT-SWAP"
    assert fact["fact_id"].startswith("sha256:")


def test_cross_instrument_result_cannot_share_a_shadow_path() -> None:
    entry = build_market_fact(
        "ROBO/USDT", _snapshot(0.10834, 1_783_990_800_000), contract_spec=_spec()
    )
    result_snapshot = {
        **_snapshot(0.01294, 1_783_991_400_000),
        "symbol": "ICP/USDT",
        "inst_id": "ICP-USDT-SWAP",
    }
    result = build_market_fact(
        "ICP/USDT", result_snapshot, contract_spec=_spec("ICP-USDT-SWAP")
    )
    bars = [
        [timestamp, 0.0129, 0.0131, 0.0128, 0.01295, 1000]
        for timestamp in range(1_783_990_800_000, 1_783_991_400_001, 60_000)
    ]

    path = verify_market_fact_path(entry, result, bars)
    contract = build_shadow_market_fact_contract(entry, result, path)

    assert path["identity_match"] is False
    assert "entry_result_native_identity_mismatch" in path["reasons"]
    assert contract["status"] == "quarantined"
    assert contract["assertions"]["native_instrument_identity_verified"] is False


def test_complete_same_instrument_ohlc_path_produces_clean_contract() -> None:
    entry = build_market_fact(
        "ROBO/USDT", _snapshot(0.01290, 1_783_990_800_000), contract_spec=_spec()
    )
    result = build_market_fact(
        "ROBO/USDT", _snapshot(0.01294, 1_783_991_400_000), contract_spec=_spec()
    )
    bars = [
        [timestamp, 0.01290, 0.01310, 0.01280, 0.01295, 1000]
        for timestamp in range(1_783_990_800_000, 1_783_991_400_001, 60_000)
    ]

    path = verify_market_fact_path(entry, result, bars)
    contract = build_shadow_market_fact_contract(entry, result, path)

    assert path["status"] == "clean"
    assert contract["version"] == MARKET_FACT_CONTRACT_VERSION
    assert contract["status"] == "clean"
    assert market_fact_contract_reasons(contract) == []
    assert all(contract["assertions"].values())


def test_feature_vector_carries_one_native_market_fact_after_depth_enrichment() -> None:
    ticker = {
        **_snapshot(0.0129, 1_783_990_800_000),
        "contract_spec": _spec(),
        "source": "websocket",
        "source_endpoint": "okx_ws_public",
        "source_channel": "tickers",
    }
    derivatives = {
        "orderbook_bid_depth": 25_000.0,
        "orderbook_ask_depth": 24_000.0,
    }

    vector = build_feature_vector("ROBO/USDT", ticker=ticker, derivatives=derivatives)

    assert vector.market_fact["quality"]["status"] == "clean"
    assert vector.market_fact["native_identity"]["inst_id"] == "ROBO-USDT-SWAP"
    assert vector.to_dict()["market_fact"]["fact_id"] == vector.market_fact["fact_id"]


def _source_consistency_auxiliary(timestamp: int, price: float) -> dict:
    return {
        "orderbook_fact": {
            "inst_id": "ROBO-USDT-SWAP",
            "inst_type": "SWAP",
            "source_timestamp_ms": timestamp,
            "bid": price - 0.00001,
            "ask": price + 0.00001,
            "bid_depth_usdt": 20_000.0,
            "ask_depth_usdt": 19_000.0,
        },
        "mark_price_fact": {
            "inst_id": "ROBO-USDT-SWAP",
            "inst_type": "SWAP",
            "source_timestamp_ms": timestamp,
            "price": price,
        },
        "index_price_fact": {
            "inst_id": "ROBO-USDT",
            "inst_type": "INDEX",
            "source_timestamp_ms": timestamp,
            "price": price,
        },
    }


def test_rest_ws_quotes_use_native_tick_and_one_minute_path_not_fixed_band() -> None:
    timestamp = 1_783_990_800_000
    rest = build_market_fact(
        "ROBO/USDT",
        _snapshot(0.01290, timestamp),
        contract_spec=_spec(),
    )
    websocket = build_market_fact(
        "ROBO/USDT",
        {
            **_snapshot(0.01305, timestamp),
            "source": "websocket",
            "source_endpoint": "okx_ws_public",
        },
        contract_spec=_spec(),
    )
    auxiliary = _source_consistency_auxiliary(timestamp, 0.01295)

    contract = build_market_source_consistency(
        rest,
        [websocket],
        **auxiliary,
        bars=[[timestamp, 0.01290, 0.01310, 0.01280, 0.01300, 10_000]],
    )

    assert contract["version"] == MARKET_SOURCE_CONSISTENCY_VERSION
    assert contract["quotes_overlap"] is False
    assert contract["path"]["observed_prices_within_path"] is True
    assert contract["status"] == "clean"
    assert all(contract["assertions"].values())


def test_current_candle_lag_is_warning_when_native_executable_quotes_overlap() -> None:
    timestamp = 1_783_990_859_000
    rest = build_market_fact(
        "ROBO/USDT",
        _snapshot(0.01305, timestamp),
        contract_spec=_spec(),
    )
    websocket = build_market_fact(
        "ROBO/USDT",
        {
            **_snapshot(0.01305, timestamp),
            "source": "websocket",
            "source_endpoint": "okx_ws_public",
        },
        contract_spec=_spec(),
    )
    auxiliary = _source_consistency_auxiliary(timestamp, 0.01305)
    auxiliary["orderbook_fact"].update({"bid": 0.01304, "ask": 0.01306})

    contract = build_market_source_consistency(
        rest,
        [websocket],
        **auxiliary,
        bars=[[timestamp - 59_000, 0.01290, 0.01300, 0.01280, 0.01295, 10_000]],
    )

    assert contract["status"] == "clean"
    assert contract["reasons"] == []
    assert contract["path"]["observed_prices_within_path"] is False
    assert contract["path"]["continuity_accepted_via"] == "native_executable_quote_overlap"
    assert "observed_price_outside_recent_native_path" in contract["reference_warnings"]
    assert contract["assertions"]["one_minute_path_verified"] is False
    assert contract["assertions"]["market_continuity_verified"] is True
    assert market_fact_contract_reasons(
        build_shadow_market_fact_contract(
            rest,
            {**rest, "source_consistency": contract},
            verify_market_fact_path(
                rest,
                rest,
                [[timestamp, 0.01290, 0.01310, 0.01280, 0.01305, 10_000]],
            ),
        )
    ) == []


def test_subsecond_native_quote_movement_uses_spread_bounded_reconciliation() -> None:
    timestamp = 1_783_990_859_000
    rest = build_market_fact(
        "ROBO/USDT",
        _snapshot(2.23360, timestamp),
        contract_spec=_spec(),
    )
    websocket = build_market_fact(
        "ROBO/USDT",
        {
            **_snapshot(2.23360, timestamp),
            "source": "websocket",
            "source_endpoint": "okx_ws_public",
        },
        contract_spec=_spec(),
    )
    auxiliary = _source_consistency_auxiliary(timestamp, 2.23435)
    auxiliary["orderbook_fact"].update(
        {
            "source_timestamp_ms": timestamp + 650,
            "bid": 2.23410,
            "ask": 2.23411,
        }
    )

    contract = build_market_source_consistency(
        rest,
        [websocket],
        **auxiliary,
        bars=[[timestamp, 2.23320, 2.23361, 2.23310, 2.23360, 10_000]],
    )

    assert contract["quotes_overlap"] is False
    assert contract["quotes_temporally_reconciled"] is True
    assert contract["quote_reconciliation"]["time_span_ms"] == 650
    assert contract["quote_reconciliation"]["source_skew_tolerance"] > contract[
        "quote_reconciliation"
    ]["gap"]
    assert contract["path"]["observed_prices_within_path"] is False
    assert contract["path"]["continuity_accepted_via"] == (
        "native_executable_quote_temporal_reconciliation"
    )
    assert contract["status"] == "clean"
    assert contract["reasons"] == []
    assert "observed_price_outside_recent_native_path" in contract["reference_warnings"]

    auxiliary["orderbook_fact"]["source_timestamp_ms"] = timestamp + 2_500
    stale_contract = build_market_source_consistency(
        rest,
        [websocket],
        **auxiliary,
        bars=[[timestamp, 2.23320, 2.23361, 2.23310, 2.23360, 10_000]],
    )
    assert stale_contract["quotes_temporally_reconciled"] is False
    assert stale_contract["status"] == "quarantined"
    assert "executable_quote_sources_not_reconciled" in stale_contract["reasons"]


def test_missing_index_reference_is_visible_without_corrupting_swap_path() -> None:
    timestamp = 1_783_990_800_000
    rest = build_market_fact(
        "ROBO/USDT",
        _snapshot(0.01290, timestamp),
        contract_spec=_spec(),
    )
    auxiliary = _source_consistency_auxiliary(timestamp, 0.01290)
    auxiliary["index_price_fact"] = {
        "inst_id": "ROBO-USDT",
        "inst_type": "INDEX",
        "source_timestamp_ms": 0,
        "price": 0.0,
    }

    contract = build_market_source_consistency(
        rest,
        [],
        **auxiliary,
        bars=[[timestamp, 0.01285, 0.01295, 0.01280, 0.01290, 10_000]],
    )

    assert contract["status"] == "clean"
    assert contract["path"]["observed_prices_within_path"] is True
    assert contract["reference_observations"]["index_price_available"] is False
    assert contract["reference_warnings"] == [
        "index_price_missing",
        "index_price_source_timestamp_missing",
    ]


def test_index_basis_is_not_compared_with_executable_swap_candle_path() -> None:
    timestamp = 1_783_990_800_000
    rest = build_market_fact(
        "ROBO/USDT",
        _snapshot(0.01290, timestamp),
        contract_spec=_spec(),
    )
    auxiliary = _source_consistency_auxiliary(timestamp, 0.01290)
    auxiliary["index_price_fact"]["price"] = 0.01320

    contract = build_market_source_consistency(
        rest,
        [],
        **auxiliary,
        bars=[[timestamp, 0.01285, 0.01295, 0.01280, 0.01290, 10_000]],
    )

    assert contract["status"] == "clean"
    assert contract["path"]["observed_prices_within_path"] is True
    assert contract["reference_observations"]["index_price_available"] is True


def test_robo_cross_source_jump_is_quarantined_by_native_path_reachability() -> None:
    timestamp = 1_783_990_800_000
    rest = build_market_fact(
        "ROBO/USDT",
        _snapshot(0.10834, timestamp),
        contract_spec=_spec(),
    )
    websocket = build_market_fact(
        "ROBO/USDT",
        {
            **_snapshot(0.01290, timestamp),
            "source": "websocket",
            "source_endpoint": "okx_ws_public",
        },
        contract_spec=_spec(),
    )
    auxiliary = _source_consistency_auxiliary(timestamp, 0.01295)

    contract = build_market_source_consistency(
        rest,
        [websocket],
        **auxiliary,
        bars=[[timestamp, 0.01290, 0.01310, 0.01280, 0.01300, 10_000]],
    )

    assert contract["status"] == "quarantined"
    assert "observed_price_outside_recent_native_path" in contract["reasons"]
    assert contract["assertions"]["one_minute_path_verified"] is False


def test_robo_incident_fixture_proves_demo_rest_price_was_not_on_live_okx_path() -> None:
    fixture = json.loads(
        Path(
            "tests/fixtures/profit_integrity/2026-07-14-robo-native-source-regression.json"
        ).read_text(encoding="utf-8")
    )
    spec = fixture["instrument_response"]
    candles = fixture["okx_live_history_candles_1m"]

    for incident in fixture["incident_rest_snapshots"]:
        fact = build_market_fact(
            fixture["symbol"],
            {
                "symbol": fixture["symbol"],
                "inst_id": fixture["inst_id"],
                "inst_type": "SWAP",
                "source": "rest",
                "source_endpoint": "okx_demo_rest_market_ticker",
                "source_channel": "tickers",
                "timestamp": incident["source_timestamp_ms"],
                "last_price": incident["last"],
                "bid": incident["bid"],
                "ask": incident["ask"],
                "notional_24h_usdt": incident["notional_24h_usdt"],
                "orderbook_bid_depth": 1407.47398,
                "orderbook_ask_depth": 1502.61238,
            },
            contract_spec=spec,
        )
        result = build_market_fact(
            fixture["symbol"],
            {
                "symbol": fixture["symbol"],
                "inst_id": fixture["inst_id"],
                "inst_type": "SWAP",
                "source": "rest",
                "source_endpoint": "okx_live_history_candles",
                "source_channel": "candle1m",
                "timestamp": candles[-1][0],
                "last_price": candles[-1][4],
                "bid": candles[-1][4] - 0.00001,
                "ask": candles[-1][4] + 0.00001,
                "notional_24h_usdt": 1.0,
                "orderbook_bid_depth": 1.0,
                "orderbook_ask_depth": 1.0,
            },
            contract_spec=spec,
        )
        path = verify_market_fact_path(fact, result, candles)

        assert "zero_notional_turnover" in fact["quality"]["reasons"]
        assert "entry_price_not_reachable_on_native_path" in path["reasons"]

    assert fixture["root_cause"]["public_rest_mode_before_fix"] == "paper_demo_flag_1"
