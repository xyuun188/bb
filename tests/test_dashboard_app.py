from config.settings import settings
from web_dashboard.api.dashboard import (
    _dashboard_symbol_query_variants,
    _exchange_position_totals,
    _group_open_dashboard_positions,
    _opening_funnel_reason_bucket,
)
from web_dashboard.api.text_sanitize import looks_mojibake
from web_dashboard.app import DEFAULT_INDEX_HTML, create_app


def test_default_dashboard_index_text_is_not_mojibake() -> None:
    assert "AI 量化交易系统" in DEFAULT_INDEX_HTML
    assert not looks_mojibake(DEFAULT_INDEX_HTML)


def test_dashboard_cors_middleware_uses_explicit_local_origins(monkeypatch) -> None:
    monkeypatch.setattr(settings, "dashboard_cors_origins", [])
    monkeypatch.setattr(settings, "dashboard_host", "127.0.0.1")
    monkeypatch.setattr(settings, "dashboard_port", 8002)

    app = create_app()
    cors = next(
        item
        for item in app.user_middleware
        if getattr(item.cls, "__name__", "") == "CORSMiddleware"
    )

    assert cors.kwargs["allow_origins"] == [
        "http://127.0.0.1:8002",
        "http://localhost:8002",
    ]
    assert cors.kwargs["allow_credentials"] is True


def test_dashboard_symbol_query_variants_cover_historical_okx_formats() -> None:
    variants = _dashboard_symbol_query_variants({"btc/usdt", "ETH-USDT-SWAP"})

    assert "btc/usdt" in variants
    assert "BTC/USDT" in variants
    assert "btc-usdt" in variants
    assert "btc-usdt-SWAP" in variants
    assert "ETH/USDT" in variants
    assert "ETH-USDT" in variants
    assert "ETH-USDT-SWAP" in variants


def test_dashboard_exchange_position_totals_do_not_duplicate_local_fragments() -> None:
    exchange_marks = {
        ("ARB/USDT", "long"): {
            "upl": 2.5,
            "margin_used": 7.0,
            "contracts": 624.4,
            "entry_price": 0.085,
            "mark_price": 0.086,
        }
    }

    totals = _exchange_position_totals(
        exchange_marks,
        fallback_margin_by_key={("ARB/USDT", "long"): 21.0},
    )

    assert totals["open_count"] == 1
    assert totals["unrealized_pnl"] == 2.5
    assert totals["used_margin"] == 7.0


def test_dashboard_open_positions_are_grouped_by_exchange_symbol_side() -> None:
    rows = [
        {
            "id": 10,
            "model_name": "ensemble_trader",
            "mode": "paper",
            "symbol": "ARB/USDT",
            "side": "long",
            "quantity": 309.0,
            "entry_price": 0.08606,
            "current_price": 0.08612,
            "unrealized_pnl": 0.06,
            "local_quantity": 309.0,
            "local_entry_price": 0.08606,
            "local_unrealized_pnl": 0.06,
            "is_open": True,
            "db_is_open": True,
            "exchange_synced": True,
        },
        {
            "id": 11,
            "model_name": "ensemble_trader",
            "mode": "paper",
            "symbol": "ARB/USDT",
            "side": "long",
            "quantity": 311.0,
            "entry_price": 0.08568,
            "current_price": 0.08612,
            "unrealized_pnl": 0.18,
            "local_quantity": 311.0,
            "local_entry_price": 0.08568,
            "local_unrealized_pnl": 0.18,
            "is_open": True,
            "db_is_open": True,
            "exchange_synced": True,
        },
    ]
    exchange_marks = {
        ("ARB/USDT", "long"): {
            "upl": 2.7538,
            "margin_used": 8.0,
            "contracts": 624.4,
            "entry_price": 0.0856789686,
            "mark_price": 0.08612,
        }
    }

    grouped = _group_open_dashboard_positions(rows, exchange_marks, mode="paper")

    assert len(grouped) == 1
    item = grouped[0]
    assert item["split_count"] == 2
    assert item["position_ids"] == [10, 11]
    assert item["quantity"] == 624.4
    assert item["local_quantity"] == 620.0
    assert item["unrealized_pnl"] == 2.7538
    assert item["local_unrealized_pnl"] == 0.24
    assert item["pnl_source"] == "okx_position"
    assert item["can_manual_close"] is False


def test_opening_funnel_buckets_dynamic_evidence_as_evidence_gate() -> None:
    reason = (
        "候选评分未达执行标准：动态证据评分硬拦截："
        "当前有效分 17.6，低于硬拦/强冲突要求；本次不开仓。"
    )

    assert _opening_funnel_reason_bucket(reason) == "evidence_gate"
