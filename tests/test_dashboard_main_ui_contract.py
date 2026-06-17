from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from web_dashboard.api import dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_main_dashboard_removes_manual_symbol_selector() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "\u6301\u4ed3\u5b9e\u65f6\u884c\u60c5" in html
    assert "\u4fdd\u8bc1\u91d1\u5360\u6bd4" in html
    assert "\u4e0b\u5355\u4fdd\u8bc1\u91d1\u5360\u5f53\u524d\u6267\u884c\u8d26\u6237\u53ef\u7528\u4f59\u989d" in html
    assert "+ \u5e01\u79cd" not in html
    assert "price-chart-symbol" not in html
    assert "price-chart-timeframe" not in html
    assert "populatePriceChartSymbols" not in script
    assert "onPriceChartSymbolChange" not in script
    assert "symbol-selector" not in html
    assert "symbol-dropdown" not in html
    assert "fetchActiveSymbols" not in script
    assert "fetchAvailableSymbols" not in script
    assert "/api/symbols/add" not in script
    assert "/api/symbols/remove" not in script
    assert "symbol-dropdown" not in style


def test_execution_account_settings_no_allocated_balance_control() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "exec-paper-allocated-balance" not in html
    assert "exec-live-allocated-balance" not in html
    assert "\u5206\u914d\u8d44\u91d1 USDT" not in html
    assert "body.allocated_balance" not in script
    assert "readNumberInput(`exec-${mode}-allocated-balance`)" not in script
    assert "\u81ea\u52a8\u4f7f\u7528 OKX" in html


def test_dashboard_refreshes_auth_status_in_topbar() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "fetchDashboardAuthStatus();" in script
    assert "setInterval(fetchDashboardAuthStatus, 60000);" in script
    assert "fetchJSON('/api/auth/status')" in script
    assert "dashboard-current-user" in script


def test_dashboard_keeps_single_auto_scan_status_after_execution_account() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "mode-static-tag" not in html
    assert "mode-static-title" not in html
    assert "自动模式</span>" not in html
    assert html.count("自动扫描 · 系统调度") == 1
    assert html.index('id="live-model-name"') < html.index("自动扫描 · 系统调度")
    assert ".mode-btn[data-scan]" not in script


def test_execution_detail_fetches_step_timeline_and_self_check_ui_exists() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "system-self-check-panel" in html
    assert "fetchSystemSelfCheck()" in html
    assert "repairSystemSelfCheck()" in html
    assert "fetchJSON(`/api/trades/${encodeURIComponent(Number(tradeId))}`)" in script
    assert "function renderExecutionTimeline" in script
    assert "failed_step" in script
    assert "execution_steps" in script
    assert "旧记录未采集耗时" in script
    assert ".execution-timeline" in style
    assert ".self-check-card" in style


def test_analysis_timing_deduplicates_final_expert_rows() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "function analysisFinalModelTimings" in script
    assert "function analysisSharedBatchCalls" in script
    assert "const finalModelTimings = analysisFinalModelTimings(modelTimings);" in script
    assert "const sharedBatchCalls = analysisSharedBatchCalls(modelTimings);" in script
    assert "modelTimings.map(item =>" not in script
    assert "finalModelTimings.map(item =>" in script
    assert "if (item.shared_batch_call || item.batch_expert) return;" in script


def test_dashboard_market_uses_open_position_snapshot_contract() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "function marketOpenPositions" in script
    assert "market.open_positions" in script
    assert "marketPositions.length" in script
    assert "buildTickersFromPositions(marketPositions)" in script


@pytest.mark.asyncio
async def test_dashboard_tickers_do_not_fallback_to_watchlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_open_position_tickers(
        open_symbols: set[str],
        market_tickers: dict[str, Any],
        mode: str | None = None,
    ) -> dict[str, Any]:
        assert open_symbols == set()
        assert market_tickers == {"BTC/USDT": {"price": 100.0}}
        assert mode == "paper"
        return {}

    monkeypatch.setattr(
        dashboard,
        "_build_tickers_for_open_positions",
        no_open_position_tickers,
    )

    tickers = await dashboard._build_dashboard_tickers(
        set(), {"BTC/USDT": {"price": 100.0}}, "paper"
    )

    assert tickers == {}


@pytest.mark.asyncio
async def test_dashboard_tickers_return_open_position_tickers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def open_position_tickers(
        open_symbols: set[str],
        market_tickers: dict[str, Any],
        mode: str | None = None,
    ) -> dict[str, Any]:
        return {"ETH/USDT": {"price": 2500.0}}

    monkeypatch.setattr(
        dashboard,
        "_build_tickers_for_open_positions",
        open_position_tickers,
    )

    tickers = await dashboard._build_dashboard_tickers(
        {"ETH/USDT"}, {"BTC/USDT": {"price": 100.0}}, "paper"
    )

    assert tickers == {"ETH/USDT": {"price": 2500.0}}


@pytest.mark.asyncio
async def test_open_position_market_snapshot_returns_positions_and_tickers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def open_symbols(mode: str | None = None) -> set[str]:
        return {"ETH/USDT"}

    async def dashboard_tickers(
        symbols: set[str],
        market_tickers: dict[str, Any],
        mode: str | None = None,
    ) -> dict[str, Any]:
        assert symbols == {"ETH/USDT"}
        assert mode == "paper"
        return {"ETH/USDT": {"price": 2500.0}}

    async def open_positions(mode: str | None = None) -> list[dict[str, Any]]:
        assert mode == "paper"
        return [
            {
                "symbol": "ETH/USDT",
                "side": "long",
                "current_price": 2500.0,
                "is_open": True,
            }
        ]

    monkeypatch.setattr(dashboard, "_data_service", None)
    monkeypatch.setattr(dashboard, "_get_display_open_position_symbols", open_symbols)
    monkeypatch.setattr(dashboard, "_build_dashboard_tickers", dashboard_tickers)
    monkeypatch.setattr(dashboard, "_get_display_open_positions_snapshot", open_positions)

    payload = await dashboard._build_open_position_market_snapshot("paper")

    assert payload["position_symbols"] == ["ETH/USDT"]
    assert payload["tickers"]["ETH/USDT"]["price"] == 2500.0
    assert payload["open_positions"][0]["symbol"] == "ETH/USDT"
    assert payload["open_position_count"] == 1


@pytest.mark.asyncio
async def test_market_snapshot_builds_tickers_from_open_positions_when_feed_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def open_symbols(mode: str | None = None) -> set[str]:
        return {"ETH/USDT"}

    async def no_feed_tickers(
        symbols: set[str],
        market_tickers: dict[str, Any],
        mode: str | None = None,
    ) -> dict[str, Any]:
        return {}

    async def open_positions(mode: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "ETH/USDT",
                "side": "long",
                "current_price": 2500.0,
                "entry_price": 2400.0,
                "change_24h": 1.2,
                "is_open": True,
            }
        ]

    monkeypatch.setattr(dashboard, "_data_service", None)
    monkeypatch.setattr(dashboard, "_get_display_open_position_symbols", open_symbols)
    monkeypatch.setattr(dashboard, "_build_dashboard_tickers", no_feed_tickers)
    monkeypatch.setattr(dashboard, "_get_display_open_positions_snapshot", open_positions)

    payload = await dashboard._build_open_position_market_snapshot("paper")

    assert payload["open_position_count"] == 1
    assert payload["tickers"] == {
        "ETH/USDT": {
            "price": 2500.0,
            "change_24h": 1.2,
            "volume_24h": 0.0,
            "bid": 0.0,
            "ask": 0.0,
        }
    }
