from config.settings import settings
from web_dashboard.api.dashboard import (
    _dashboard_symbol_query_variants,
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


def test_opening_funnel_buckets_dynamic_evidence_as_evidence_gate() -> None:
    reason = (
        "候选评分未达执行标准：动态证据评分硬拦截："
        "当前有效分 17.6，低于硬拦/强冲突要求；本次不开仓。"
    )

    assert _opening_funnel_reason_bucket(reason) == "evidence_gate"
