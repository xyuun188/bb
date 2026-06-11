from types import SimpleNamespace

import pytest

from services.entry_market_data_quality import (
    EntryMarketDataQualityPolicy,
    MarketValueReader,
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
    }
    snapshot.update(kwargs)
    return snapshot


def test_market_value_reader_supports_dicts_and_objects():
    reader = MarketValueReader()

    assert reader.read({"price": 10}, "price") == 10
    assert reader.read(SimpleNamespace(price=11), "price") == 11
    assert reader.read(SimpleNamespace(), "price", 12) == 12


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        ({"current_price": 0, "close": 0, "bid": 0, "ask": 0}, "没有有效价格"),
        (_valid_snapshot(close=130.0), "行情价格源分裂"),
        (_valid_snapshot(current_price=80.0, close=0.0, bid=0.0, ask=0.0), "24小时区间"),
        (_valid_snapshot(bid=100.0, ask=104.0), "盘口价差过大"),
        (_valid_snapshot(orderbook_bid_depth=0.0, orderbook_imbalance=1.0), "盘口深度异常"),
        (
            _valid_snapshot(
                returns_1=0.0,
                returns_5=0.0,
                returns_20=0.0,
                volatility_20=0.0,
                change_24h_pct=1.0,
            ),
            "短周期行情特征疑似缺失",
        ),
        (
            _valid_snapshot(abnormal_wick_count_72h=1, abnormal_wick_max_pct=88.0),
            "异常插针",
        ),
    ],
)
def test_entry_market_data_quality_policy_blocks_unusable_market_data(snapshot, expected):
    reason = EntryMarketDataQualityPolicy().reason(snapshot, stage_label="测试阶段")

    assert reason is not None
    assert "测试阶段" in reason
    assert expected in reason


def test_entry_market_data_quality_policy_allows_consistent_market_data():
    assert EntryMarketDataQualityPolicy().reason(_valid_snapshot()) is None


def test_entry_market_data_quality_policy_handles_reader_failures():
    def broken_reader(_source, _key, _default):
        raise AttributeError("bad payload")

    policy = EntryMarketDataQualityPolicy(market_value_reader=broken_reader)

    assert "行情数据异常" in (policy.reason({}, stage_label="复核") or "")
