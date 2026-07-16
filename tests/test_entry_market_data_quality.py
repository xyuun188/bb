from types import SimpleNamespace

import pytest

from core.market_facts import MARKET_SOURCE_CONSISTENCY_VERSION, build_market_fact
from services.entry_market_data_quality import (
    EntryMarketDataQualityPolicy,
    MarketValueReader,
)


def _contract_spec(inst_id: str = "BTC-USDT-SWAP") -> dict[str, str]:
    return {
        "instId": inst_id,
        "instType": "SWAP",
        "uly": inst_id.removesuffix("-SWAP"),
        "instFamily": inst_id.removesuffix("-SWAP"),
        "ctType": "linear",
        "ctVal": "0.01",
        "ctMult": "1",
        "ctValCcy": inst_id.split("-")[0],
        "settleCcy": "USDT",
        "lotSz": "0.01",
        "minSz": "0.01",
        "tickSz": "0.1",
        "state": "live",
    }


def _clean_market_fact() -> dict:
    source_consistency = {
        "version": MARKET_SOURCE_CONSISTENCY_VERSION,
        "status": "clean",
        "reasons": [],
        "assertions": {
            "native_identity_verified": True,
            "executable_quotes_verified": True,
            "tick_alignment_verified": True,
            "reference_prices_verified": True,
            "market_continuity_verified": True,
        },
    }
    return build_market_fact(
        "BTC/USDT",
        {
            "symbol": "BTC/USDT",
            "inst_id": "BTC-USDT-SWAP",
            "inst_type": "SWAP",
            "source": "rest",
            "source_endpoint": "okx_rest_market_ticker",
            "source_channel": "tickers",
            "timestamp": 1_784_000_000_000,
            "last_price": 100.0,
            "bid": 99.9,
            "ask": 100.1,
            "notional_24h_usdt": 1_000_000.0,
            "volume_24h_contracts": 10_000.0,
            "volume_24h_base": 100.0,
            "orderbook_bid_depth": 1000.0,
            "orderbook_ask_depth": 1000.0,
            "market_source_consistency": source_consistency,
        },
        contract_spec=_contract_spec(),
    )


def _valid_snapshot(**kwargs):
    snapshot = {
        "current_price": 100.0,
        "close": 100.2,
        "bid": 99.9,
        "ask": 100.1,
        "returns_1": 0.001,
        "returns_5": 0.002,
        "returns_20": 0.003,
        "volatility_20": 0.01,
        "change_24h_pct": 0.2,
        "high_24h": 105.0,
        "low_24h": 95.0,
        "orderbook_bid_depth": 1000.0,
        "orderbook_ask_depth": 1000.0,
        "orderbook_imbalance": 0.1,
        "abnormal_wick_count_72h": 0,
        "abnormal_wick_max_pct": 0.0,
        "market_fact": _clean_market_fact(),
    }
    snapshot.update(kwargs)
    return snapshot


def test_market_value_reader_supports_dicts_and_objects():
    reader = MarketValueReader()

    assert reader.read({"price": 10}, "price") == 10
    assert reader.read(SimpleNamespace(price=11), "price") == 11
    assert reader.read(SimpleNamespace(), "price", 12) == 12


@pytest.mark.parametrize(
    ("snapshot", "expected", "code"),
    [
        (
            _valid_snapshot(current_price=0, close=0, bid=0, ask=0),
            "没有有效价格",
            "missing_valid_price",
        ),
        (
            _valid_snapshot(current_price=80.0, close=0.0, bid=0.0, ask=0.0),
            "24小时区间",
            "price_outside_24h_range",
        ),
        (_valid_snapshot(bid=101.0, ask=100.0), "盘口结构无效", "crossed_bid_ask"),
        (
            _valid_snapshot(orderbook_bid_depth=0.0, orderbook_imbalance=1.0),
            "盘口深度异常",
            "orderbook_depth_invalid",
        ),
        (
            _valid_snapshot(
                returns_1=0.0,
                returns_5=0.0,
                returns_20=0.0,
                volatility_20=0.0,
                change_24h_pct=1.0,
            ),
            "短周期行情特征疑似缺失",
            "short_cycle_features_missing",
        ),
    ],
)
def test_entry_market_data_quality_policy_blocks_unusable_market_data(snapshot, expected, code):
    policy = EntryMarketDataQualityPolicy()
    issue = policy.issue(snapshot, stage_label="测试阶段")

    assert issue is not None
    assert issue.code == code
    assert issue.as_dict()["exclude_from_training"] is True
    assert "测试阶段" in issue.reason
    assert expected in issue.reason
    assert policy.reason(snapshot, stage_label="测试阶段") == issue.reason


def test_entry_market_data_quality_policy_allows_consistent_market_data():
    assert EntryMarketDataQualityPolicy().reason(_valid_snapshot()) is None


def test_entry_market_data_quality_policy_fails_closed_without_native_fact():
    issue = EntryMarketDataQualityPolicy().issue(
        {key: value for key, value in _valid_snapshot().items() if key != "market_fact"}
    )

    assert issue is not None
    assert issue.code == "native_market_fact_missing"


def test_entry_market_data_quality_policy_rejects_stale_consistency_contract():
    snapshot = _valid_snapshot()
    snapshot["market_fact"]["source_consistency"]["version"] = (
        "2026-07-14.okx-source-consistency.v1"
    )

    issue = EntryMarketDataQualityPolicy().issue(snapshot)

    assert issue is not None
    assert issue.code == "native_market_fact_invalid"
    assert "source_consistency_contract_missing_or_stale" in issue.reason


def test_zero_turnover_robo_native_fact_cannot_reach_production_entry():
    snapshot = _valid_snapshot()
    snapshot["market_fact"] = build_market_fact(
        "ROBO/USDT",
        {
            "symbol": "ROBO/USDT",
            "inst_id": "ROBO-USDT-SWAP",
            "inst_type": "SWAP",
            "source": "rest",
            "source_endpoint": "okx_rest_market_ticker",
            "source_channel": "tickers",
            "timestamp": 1_784_000_000_000,
            "last_price": 0.10834,
            "bid": 0.08172,
            "ask": 0.10834,
            "notional_24h_usdt": 0.0,
            "orderbook_bid_depth": 1000.0,
            "orderbook_ask_depth": 1000.0,
        },
        contract_spec=_contract_spec("ROBO-USDT-SWAP"),
    )

    issue = EntryMarketDataQualityPolicy().issue(snapshot)

    assert issue is not None
    assert issue.code == "native_market_fact_invalid"
    assert "zero_notional_turnover" in issue.reason
    assert "zero_notional_turnover" in issue.details["market_fact_reason_codes"]


@pytest.mark.parametrize(
    ("symbol", "inst_id", "spec", "expected_reason"),
    [
        (
            "BTC/USDT",
            "ETH-USDT-SWAP",
            _contract_spec("ETH-USDT-SWAP"),
            "native_instrument_symbol_mismatch",
        ),
        ("BTC/USDT", "BTC-USDT-SWAP", None, "native_identity_missing:uly"),
    ],
)
def test_native_identity_or_contract_spec_failure_blocks_entry(
    symbol,
    inst_id,
    spec,
    expected_reason,
):
    snapshot = _valid_snapshot()
    snapshot["market_fact"] = build_market_fact(
        symbol,
        {
            "symbol": symbol,
            "inst_id": inst_id,
            "inst_type": "SWAP",
            "source": "rest",
            "source_endpoint": "okx_rest_market_ticker",
            "source_channel": "tickers",
            "timestamp": 1_784_000_000_000,
            "last_price": 100.0,
            "bid": 99.9,
            "ask": 100.1,
            "notional_24h_usdt": 1_000_000.0,
            "orderbook_bid_depth": 1000.0,
            "orderbook_ask_depth": 1000.0,
        },
        contract_spec=spec,
    )

    issue = EntryMarketDataQualityPolicy().issue(snapshot)

    assert issue is not None
    assert issue.code == "native_market_fact_invalid"
    assert expected_reason in issue.reason


def test_wide_spread_is_priced_by_execution_cost_instead_of_fixed_data_gate():
    assert EntryMarketDataQualityPolicy().reason(_valid_snapshot(bid=100.0, ask=104.0)) is None


def test_completed_indicator_close_is_not_compared_to_live_spread_as_corruption():
    assert EntryMarketDataQualityPolicy().reason(_valid_snapshot(close=130.0)) is None


def test_entry_market_data_quality_policy_handles_reader_failures():
    def broken_reader(_source, _key, _default):
        raise AttributeError("bad payload")

    policy = EntryMarketDataQualityPolicy(market_value_reader=broken_reader)
    issue = policy.issue({}, stage_label="复核")

    assert issue is not None
    assert issue.code == "market_payload_invalid"
    assert "行情数据异常" in issue.reason
