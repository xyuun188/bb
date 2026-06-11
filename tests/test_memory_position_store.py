from types import SimpleNamespace

from services.memory_position_store import MemoryPositionStore


def _normalize(symbol):
    value = str(symbol or "")
    if "/" not in value and "-" in value:
        base, quote, *_rest = value.split("-")
        return f"{base}/{quote}"
    return value


def test_memory_position_store_removes_matching_open_position_only():
    executor = SimpleNamespace(
        _positions={
            "ensemble_trader": [
                {"symbol": "BTC-USDT-SWAP", "side": "long", "is_open": True},
                {"symbol": "BTC/USDT", "side": "short", "is_open": True},
                {"symbol": "ETH/USDT", "side": "long", "is_open": True},
                {"symbol": "BTC/USDT", "side": "long", "is_open": False},
            ]
        }
    )
    store = MemoryPositionStore(
        paper_executor_provider=lambda: executor,
        symbol_normalizer=_normalize,
    )

    store.remove_open_position("ensemble_trader", "BTC/USDT", "long")

    assert executor._positions["ensemble_trader"] == [
        {"symbol": "BTC/USDT", "side": "short", "is_open": True},
        {"symbol": "ETH/USDT", "side": "long", "is_open": True},
        {"symbol": "BTC/USDT", "side": "long", "is_open": False},
    ]


def test_memory_position_store_noops_without_paper_executor():
    store = MemoryPositionStore(
        paper_executor_provider=lambda: None,
        symbol_normalizer=lambda symbol: str(symbol or ""),
    )

    store.remove_open_position("ensemble_trader", "BTC/USDT", "long")
