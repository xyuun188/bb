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
    assert (
        "\u4e0b\u5355\u4fdd\u8bc1\u91d1\u5360\u5f53\u524d\u6267\u884c\u8d26\u6237\u53ef\u7528\u4f59\u989d"
        in html
    )
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
    assert "logoutDashboard" in script
    assert "redirectToLogin('已退出登录。')" in script


def test_dashboard_static_assets_keep_utf8_chinese_text() -> None:
    assets = [
        PROJECT_ROOT / "web_dashboard/static/index.html",
        PROJECT_ROOT / "web_dashboard/static/js/dashboard.js",
        PROJECT_ROOT / "web_dashboard/static/css/dashboard.css",
    ]
    mojibake_markers = (
        "鐧诲綍",
        "瀹炵洏",
        "鍒囨崲",
        "宸叉湁",
        "璐︽埛",
        "澶辫触",
        "棰勬湡",
        "鏀剁泭",
        "�",
    )

    for asset in assets:
        text = asset.read_text(encoding="utf-8")
        assert not any(marker in text for marker in mojibake_markers), asset


def test_live_mode_switch_requires_known_missing_okx_config() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "liveConfigured: null" in script
    assert "if (mode === 'live' && state.okxConfig?.liveConfigured === false)" in script
    assert "const knownMissing = state.okxConfig?.liveConfigured === false" in script
    assert "button.classList.toggle('needs-config', knownMissing)" in script
    assert "后端会在切换前再次校验" in script
    assert "请先配置 API Key、API Secret 和 Passphrase" in script


def test_dashboard_internal_api_requests_handle_auth_expiry() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    allowed_raw_fetches = {
        "const res = await fetch(url, { cache: 'no-store' });",
        "const res = await fetch(url, options);",
        "await fetch('/api/auth/logout', dashboardWriteOptions({",
        "const res = await fetch(`https://www.okx.com/api/v5/market/ticker?instId=${encodeURIComponent(instId)}`);",  # noqa: E501
    }

    for line in script.splitlines():
        stripped = line.strip()
        if "fetch(" in stripped:
            assert stripped in allowed_raw_fetches or "fetchWithAuth(" in stripped
    assert "fetchWithAuth('/api/settings/okx/balance'" in script
    assert "fetchWithAuth('/api/control/mode'" in script
    assert "fetchWithAuth('/api/settings/okx'" in script


def test_opportunity_score_ui_prefers_expected_net_return() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "function opportunityScorePrimaryReturn" in script
    assert "const net = Number(score.expected_net_return_pct);" in script
    assert "if (Number.isFinite(net)) return { label: '预期净收益', value: net };" in script
    assert "预期收益：${opportunityScoreValue(score.expected_return_pct, 4)}%" not in script
    assert "`预期收益 ${opportunityScoreValue(score.expected_return_pct, 4)}%`" not in script
    assert "收益来源" in script


def test_decision_detail_explains_dynamic_evidence_and_confidence() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "function dynamicEvidenceBlock" in script
    assert "动态证据评分" in script
    assert "分析信心：${evidencePercentLabel(confidence)}" in script
    assert "不等于动态证据分" in script
    assert "弱证据不是单看分析信心" in script
    assert "AI、ML、时序、情绪、服务器盈利、影子记忆和币种历史" in script
    assert "decision-evidence-summary" in script
    assert ".decision-evidence-summary" in style
    assert ".decision-evidence-components" in style


def test_agent_skill_detail_uses_readable_card_layout() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "analysis-agent-skills-grid" in script
    assert "analysis-skill-data-grid" in script
    assert "analysis-skill-data-row" in script
    assert "analysis-skill-data-key" in script
    assert "analysis-skill-data-value" in script
    assert '<div class="analysis-resolution-item analysis-skill-item">' not in script
    assert "analysis-skill-data-chip" not in script
    assert "analysis-skill-data-chip" not in style
    assert ".analysis-agent-skills-grid" in style
    assert ".analysis-skill-item" in style
    assert ".analysis-skill-data-row" in style


def test_dashboard_keeps_single_auto_scan_status_after_execution_account() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "mode-static-tag" not in html
    assert "mode-static-title" not in html
    assert "\u81ea\u52a8\u6a21\u5f0f</span>" not in html
    assert html.count("\u81ea\u52a8\u626b\u63cf \u00b7 \u7cfb\u7edf\u8c03\u5ea6") == 1
    assert html.index('id="live-model-name"') < html.index(
        "\u81ea\u52a8\u626b\u63cf \u00b7 \u7cfb\u7edf\u8c03\u5ea6"
    )
    assert ".mode-btn[data-scan]" not in script


def test_execution_detail_fetches_step_timeline_and_self_check_ui_exists() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "system-self-check-panel" in html
    assert "refreshServerMonitorPage()" in html
    assert "repairSystemSelfCheck()" in html
    assert "fetchJSON(`/api/trades/${encodeURIComponent(Number(tradeId))}`)" in script
    assert "function renderExecutionTimeline" in script
    assert "execution-reason-primary" in script
    assert "\u6267\u884c\u539f\u56e0" in script
    assert "\u6267\u884c\u6b65\u9aa4\u8bf4\u660e" in script
    assert "failed_step" in script
    assert "execution_steps" in script
    assert "\u65e7\u8bb0\u5f55\u672a\u91c7\u96c6\u8017\u65f6" in script
    assert "\\u63d0\\u793a ${Number(summary.info || 0)}" in script
    assert ".execution-timeline" in style
    assert ".self-check-card" in style
    assert ".self-check-card.info" in style
    assert ".execution-reason-primary" in style


def test_dashboard_account_buttons_use_delegated_actions() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert 'data-dashboard-user-action="create"' in html
    assert "function initDashboardUserActions" in script
    assert "dashboardUserWriteRequest" in script
    assert 'data-dashboard-user-action="edit"' in script
    assert "data-dashboard-user-action=\"${active ? 'deactivate' : 'activate'}\"" in script
    assert 'data-dashboard-user-action="delete"' in script
    assert 'onclick="openDashboardUserModal' not in script
    assert 'onclick="setDashboardUserActive' not in script
    assert 'onclick="deleteDashboardUser' not in script


def test_server_monitor_splits_model_and_platform_panels() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert 'data-server-monitor-tab="model"' in html
    assert 'data-server-monitor-tab="self-check"' in html
    assert 'data-server-monitor-tab="platform"' in html
    assert 'id="server-monitor-panel-self-check"' in html
    assert "\u5927\u6a21\u578b\u670d\u52a1\u5668" in html
    assert "\u5e73\u53f0\u670d\u52a1\u5668" in html
    assert "platform-server-overview" in html
    assert "function renderPlatformServerMonitor" in script
    assert "serverMonitorTab: 'self-check'" in script
    assert "function refreshServerMonitorPage" in script
    assert "serverMonitorRefreshInFlight" in script
    assert "platform_server" in script
    assert "const visibleServices = Array.from(" in script
    assert "services.reduce((map, service) =>" in script
    assert "'redis-server.service': 'Redis'" in script
    assert "'redis.service': 'Redis'" in script
    assert ".server-monitor-tabs" in style
    assert ".server-monitor-self-check-actions" in style
    assert ".server-monitor-panel.active" in style
    assert "const MODEL_PUBLIC_HOST = '103.85.84.147';" in script
    assert "'qwen3-14b-trade': `http://${MODEL_PUBLIC_HOST}:21840/v1`" in script
    assert "'deepseek-r1-14b-risk': `http://${MODEL_PUBLIC_HOST}:21842/v1`" in script
    assert "local_ai_tools: `http://${MODEL_PUBLIC_HOST}:21841`" in script
    assert "data.model_access_host" not in script


def test_position_history_symbol_variants_include_okx_swap_suffix() -> None:
    variants = dashboard._dashboard_symbol_query_variants({"OP/USDT"})

    assert "OP/USDT:USDT" in variants
    assert "OP-USDT-SWAP" in variants


def test_position_history_does_not_treat_split_batches_as_partial() -> None:
    script = (PROJECT_ROOT / "web_dashboard/api/dashboard.py").read_text(encoding="utf-8")

    assert "Order.symbol.in_(close_symbol_variants)" in script
    assert 'int(group.get("split_count") or 1) > 1' not in script


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
