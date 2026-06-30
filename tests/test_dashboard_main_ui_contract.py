from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from web_dashboard.api import dashboard

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


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


def test_execution_account_ui_uses_okx_equity_pnl_not_local_trade_fallback() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "account.today_total_pnl ?? account.today_equity_pnl" not in script
    assert "const todayTotalPnl = valueNumber(account.today_equity_pnl);" in script
    assert "account.cumulative_total_pnl ?? account.total_pnl" not in script
    assert "const phase3TotalPnl = valueNumber(account.phase3_equity_pnl);" in script
    assert "今日OKX权益变化" in script
    assert "三期OKX权益变化" in script


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
        PROJECT_ROOT / "web_dashboard/static/js/strategy_learning_view.js",
        PROJECT_ROOT / "web_dashboard/static/css/dashboard.css",
        PROJECT_ROOT / "web_dashboard/static/css/strategy_learning.css",
        PROJECT_ROOT / "web_dashboard/app.py",
    ]
    mojibake_markers = (
        "????",
        _u("\\u9427\\u8bf2\\u7d8d"),
        _u("\\u7039\\u70b5\\u6d0f"),
        _u("\\u9352\\u56e8\\u5d32"),
        _u("\\u5bb8\\u53c9\\u6e41"),
        _u("\\u7490\\ufe3d\\u57db"),
        _u("\\u6fb6\\u8fab\\u89e6"),
        _u("\\u68f0\\u52ec\\u6e61"),
        _u("\\u93c0\\u5241\\u6ced"),
        _u("\\u947b\\u5d89\\u5e9c"),
        _u("\\u942a\\u5b2b\\u6f98"),
        _u("\\u7ed4\\ue21a\\u5f5b"),
        _u("\\ufffd"),
    )

    for asset in assets:
        text = asset.read_text(encoding="utf-8")
        assert not any(marker in text for marker in mojibake_markers), asset


def test_dashboard_runtime_stats_do_not_regress_from_ws_packets() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")

    assert "function parsedRuntimeSeconds" in script
    assert "function updateRuntimeClock" in script
    assert "setInterval(updateRuntimeClock, 1000);" in script
    assert "updateStats(data.stats || {}, 'ws')" in script
    assert "updateStats(data, 'summary')" in script
    assert "state.lastStatsSource === 'summary'" in script
    assert "!hasRuntimeFields" in script
    assert "stats.started_at ||" in script
    assert "stats.last_heartbeat_at ||" in script
    assert 'id="status-market-stage"' in html
    assert 'id="status-position-stage"' in html
    assert "function scopedStageText" in script
    assert "function loopErrorScopeLabel" in script
    assert "return stageLabelText(stage, '', stats?.running);" in script
    assert "learning: '刷新策略学习上下文'" in script
    assert "市场分析线程：" in script
    assert "持仓复盘线程：" in script
    assert "position analysis round cancelled by hard watchdog" in script
    assert "持仓复盘整轮超时" in script
    assert "market_current_stage" in script
    assert "position_current_stage" in script
    assert "strategy_context:" in script
    assert 'id="status-okx-sync"' in html
    assert "function okxAuthoritativeSyncLabel" in script
    assert "okx_authoritative_sync" in script
    assert "OKX权威事实同步正常" in script
    assert "OKX权威事实同步异常" in script
    assert "Missing real OKX equity snapshot" in script


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


def test_position_history_uses_okx_grouped_ledger_linked_orders_modal() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "关联订单" in html
    assert 'id="position-linked-orders-modal-overlay"' in html
    assert 'id="position-linked-orders-modal-body"' in html
    assert 'colspan="11"' in html
    assert "positionLinkedOrdersByGroup" in script
    assert "linked_fills" in script
    assert "linked_order_count" in script
    assert "evidence_complete" in script
    assert "function openPositionLinkedOrdersModal" in script
    assert "function closePositionLinkedOrdersModal" in script
    assert ".js-position-linked-orders" in script
    assert ".position-ledger-summary" in style
    assert ".position-linked-orders-table-wrap" in style


def test_opportunity_score_ui_prefers_expected_net_return() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "function opportunityScorePrimaryReturn" in script
    assert "completedLocalTrade" in script
    assert "累计去重样本" in script
    assert "手动平仓不参与训练" in script
    assert "const net = Number(score.expected_net_return_pct);" in script
    assert "if (Number.isFinite(net)) return { label: '预期净收益', value: net };" in script
    assert "function opportunityScoreFormulaItems" in script
    assert "function opportunityScoreFormulaHtml" in script
    assert "净收益拆解" in script
    assert "只参与证据评分" in script
    assert "AI贡献" in script
    assert "最终净收益" in script
    assert "预期收益：${opportunityScoreValue(score.expected_return_pct, 4)}%" not in script
    assert "`预期收益 ${opportunityScoreValue(score.expected_return_pct, 4)}%`" not in script
    assert "原始" in script


def test_opportunity_score_execution_state_uses_final_status() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "function opportunityScoreExecutionState" in script
    assert "最终未执行" in script
    assert "执行检查中" in script
    assert "已进入执行队列" not in script


def test_system_audit_model_training_distinguishes_optional_sources() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "optional_source_warnings" in script
    assert "可选增强源" in script
    assert "硬故障" in script
    assert "学习观察" in script


def test_decision_detail_explains_dynamic_evidence_and_confidence() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "function decisionMetricItem" in script
    assert "decision-score-grid" in script
    assert "decision-score-reason" in script
    assert "decision-score-formula" in script
    assert "decision-score-formula-grid" in script
    opportunity_start = script.index("function opportunityScoreBlock")
    opportunity_end = script.index("function showDecisionReason", opportunity_start)
    opportunity_block = script[opportunity_start:opportunity_end]
    assert "<br>" not in opportunity_block
    assert "function dynamicEvidenceBlock" in script
    assert "动态证据评分" in script
    assert "decisionMetricItem('分析信心', evidencePercentLabel(confidence)" in script
    assert "不等于动态证据分" in script
    assert "弱证据不是单看分析信心" in script
    assert "AI、ML、时序、情绪、服务器盈利、影子记忆和币种历史" in script
    assert ".decision-score-grid" in style
    assert ".decision-score-metric" in style
    assert ".decision-score-formula" in style
    assert ".decision-score-formula-grid" in style
    assert "max-width: calc(100vw - 28px);" in style
    assert "max-height: calc(100vh - 28px);" in style
    assert "flex: 1 1 auto;" in style
    assert "overflow-x: hidden;" in style
    assert "overflow-wrap: anywhere;" in style
    assert "grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));" in style
    assert "grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));" in style
    assert ".decision-score-formula-item:nth-child(2)" in style
    assert "grid-column: 1 / -1;" in style
    assert ".modal.modal-wide .modal-body" in style
    assert ".decision-detail-stack" in style
    assert "width: min(1480px, calc(100vw - 28px));" in style
    assert ".modal.modal-wide .decision-score-grid" in style
    assert ".decision-score-formula-item:nth-child(4)" in style
    assert "decision-evidence-summary" in script
    assert "short_evidence_adjustment" in script
    assert "做空强证据放开" in script
    assert "做空保守修正" in script
    assert "decision-evidence-adjustment" in script
    assert ".decision-evidence-summary" in style
    assert ".decision-evidence-adjustment" in style
    assert ".decision-evidence-components" in style


def test_opening_funnel_profit_expectancy_is_not_exchange_error() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert (
        dashboard._opening_funnel_reason_bucket(
            "候选逻辑暂未满足执行条件：实际下单方向做多预期净收益 -0.0410% 不为正，系统禁止提交开仓订单"
        )
        == "profit_expectancy"
    )
    assert (
        dashboard._opening_funnel_reason_bucket("OKX API timeout while submitting order")
        == "execution_or_exchange"
    )
    assert (
        dashboard._opening_funnel_reason_bucket(
            "下单前价格已比分时分析下跌 0.54%，超过允许偏移 0.50%。系统已即时刷新该币种行情复核，但偏移仍过大或盘口/动量未通过复核；为避免追空，本次不执行。"
        )
        == "risk_or_precheck"
    )
    assert "profit_expectancy: '收益期望'" in script


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
    assert ".analysis-skill-head" in style
    assert ".analysis-skill-badges" in style
    assert ".analysis-skill-reason" in style
    assert ".analysis-skill-data-row" in style
    assert "overflow-wrap: anywhere" in style


def test_analysis_news_context_is_collapsed_by_default() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    news_start = script.index("function renderAnalysisNewsContext")
    news_end = script.index("function analysisExpertConfig", news_start)
    news_block = script[news_start:news_end]

    assert '<details class="analysis-news-group">' in news_block
    assert '<details class="analysis-news-group" ${directRows ?' not in news_block
    assert '<details class="analysis-news-group" ${!directRows' not in news_block
    assert "<summary>直接相关新闻<span>${directCount} 条</span></summary>" in news_block
    assert "<summary>全市场背景新闻<span>${marketCount} 条</span></summary>" in news_block


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


def test_dashboard_status_shows_split_scheduler_intervals() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "market_loop_interval_seconds" in script
    assert "position_loop_interval_seconds" in script
    assert "market_round_time_budget_seconds" in script
    assert "配置${fmtSecondsLabel(state.decisionInterval)}" in script
    assert "市场${fmtSecondsLabel(marketInterval)}" in script
    assert "持仓${fmtSecondsLabel(positionInterval)}" in script


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
    assert "function selfCheckGroupedItems" in script
    assert "function selfCheckItemHtml" in script
    assert "self-check-group-list" in script
    assert "self-check-group-grid" in script
    assert "failed_step" in script
    assert "execution_steps" in script
    assert "\u65e7\u8bb0\u5f55\u672a\u91c7\u96c6\u8017\u65f6" in script
    assert "\\u63d0\\u793a ${Number(summary.info || 0)}" in script
    assert ".execution-timeline" in style
    assert ".self-check-card" in style
    assert ".self-check-card.info" in style
    assert ".self-check-group-list" in style
    assert ".self-check-group-grid" in style
    assert ".self-check-repair-note" in style
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


def test_system_audit_displays_issue_ledger() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")

    assert 'id="system-audit-issue-ledger"' in html
    assert "function renderSystemAuditIssueLedger" in script
    assert "issue_ledger" in script
    assert "已修复" in script
    assert "未修复" in script
    assert "历史观察" in script
    assert ".system-audit-ledger-grid" in style
    assert ".system-audit-ledger-column" in style
    assert ".system-audit-ledger-item" in style
    assert ".server-monitor-self-check-actions" in style
    assert ".server-monitor-panel.active" in style
    assert "MODEL_PUBLIC_HOST" not in script
    assert "'qwen3-32b-trade': 'platform loopback 18000'" in script
    assert "'deepseek-r1-14b-risk': 'platform loopback 18002'" in script
    assert "'BB-FinQuant-Expert-14B': 'platform loopback 18003'" in script
    assert "phase3_quant_api: 'platform loopback 18001'" in script
    assert "21840" not in script and "21841" not in script and "21842" not in script
    assert "data.model_access_host" not in script


def test_server_monitor_rendering_isolated_from_numeric_format_errors() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "dashboard.js?v=20260624-threshold-governance-strip" in html
    assert "const rawDigits = Number(digits);" in script
    assert "Math.max(0, Math.min(Math.trunc(rawDigits), 6))" in script
    assert "monitorNumber(tools.completed_shadow_sample_count, monitorNumber(" not in script
    assert "monitorNumber(tools.completed_trade_sample_count, monitorNumber(" not in script
    assert "Promise.allSettled([" in script
    assert "document.getElementById('server-monitor-model-runtime')" in script
    assert "document.getElementById('server-monitor-model-panel')" not in script
    assert "模型路由不匹配" in script
    assert "刷新大模型服务器监控失败" in script
    assert "刷新系统自检失败" in script


def test_server_monitor_endpoint_labels_are_status_aware() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    label_block = script[
        script.index("function runtimeEndpointStatusLabel") : script.index(
            "function runtimeEndpointSummary"
        )
    ]
    platform_block = script[
        script.index("function renderPlatformRuntimeCard") : script.index(
            "function runtimeStatusBadge"
        )
    ]
    model_block = script[
        script.index("function renderServerModelRuntime") : script.index("// --- Formatters ---")
    ]

    assert "status_category" in label_block
    assert "category === 'auth_failed'" in label_block
    assert "category === 'auth_forbidden'" in label_block
    assert "category === 'network_error'" in label_block
    assert "return '认证失败';" in label_block
    assert "return '权限拒绝';" in label_block
    assert "return '模型不匹配';" in label_block
    assert "return '模型未就绪';" in label_block
    assert "return '业务未通过';" in label_block
    assert "runtimeEndpointStatusLabel(item, { model: true })" in platform_block
    assert "runtimeEndpointStatusLabel(item)" in platform_block
    assert "runtimeEndpointStatusLabel(item, { model: true })" in model_block
    assert "runtimeEndpointStatusLabel(item)" in model_block
    assert "item.ok ? '接口异常'" not in script
    assert "endpoint_ok ? '模型不匹配' : '不可达'" not in script


def test_system_audit_root_cause_radar_is_wired() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert 'data-page="system-audit"' in html
    assert 'id="page-system-audit"' in html
    assert "系统巡检" in html
    assert "链路节点图谱" in html
    assert "根因队列" in html
    assert "巡检模块明细" in html
    assert "system-audit-grid" in html
    assert "system-audit-card-grid" in html
    assert "只做只读巡检" in html
    assert "fetchSystemAudit()" in html
    assert "systemAuditStatus: null" in script
    assert "let systemAuditRefreshInFlight = null" in script
    assert "if (page === 'system-audit') fetchSystemAudit();" in script
    assert "if (isPageActive('system-audit'))" in script
    assert "fetchJSON('/api/system-audit/status')" in script
    assert "function renderSystemAudit" in script
    assert "function renderSystemAuditCards" in script
    assert "function renderSystemAuditNodes" in script
    assert "function renderSystemAuditRootCauses" in script
    assert "function systemAuditCardDetailOpen" in script
    assert '<details class="system-audit-card' in script
    assert "renderSystemAuditNodes(data.nodes)" in script
    assert "system-audit-node-flow" in script
    assert "system-audit-node-checks" in script
    assert "做空保守修正样本" in script
    assert "short_conservative_adjustment_samples" in script
    assert "short_released_adjustment_samples" in script
    assert "function systemAuditShadowMissedOpportunityDetails" in script
    assert (
        script.index("function systemAuditGenericDetailsHtml")
        < script.index("function systemAuditShadowMissedOpportunityDetails")
        < script.index("function systemAuditCardDetailsHtml")
    )
    generic_body = script[
        script.index("function systemAuditGenericDetailsHtml") : script.index(
            "function systemAuditShadowMissedOpportunityDetails"
        )
    ]
    assert "Object.entries(details)" in generic_body
    assert "systemAuditShadowMissedOpportunityDetails" not in generic_body
    assert "shadow_missed_opportunity" in script
    assert "Adopted missed opportunities" in script
    assert "Blocked reason counts" in script
    assert "系统巡检接口请求失败" in script
    assert "补历史仓位、重启服务、批量训练等动作必须人工确认" in script
    assert ".system-audit-grid" in style
    assert ".system-audit-section-head" in style
    assert ".system-audit-card-grid" in style
    assert ".system-audit-node-grid" in style
    assert ".system-audit-node" in style
    assert ".system-audit-card" in style
    assert ".system-audit-root-cause" in style
    assert ".system-audit-health-strip" in style
    assert "overflow-wrap: anywhere;" in style


def test_system_audit_nodes_use_state_aware_display_status() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    display_status_block = script[
        script.index("function systemAuditDisplayStatus") : script.index(
            "function systemAuditValueText"
        )
    ]
    node_block = script[
        script.index("function renderSystemAuditNodes") : script.index(
            "function renderSystemAudit()"
        )
    ]
    card_block = script[
        script.index("function renderSystemAuditCards") : script.index(
            "function renderSystemAuditNodes"
        )
    ]

    assert "if (item.display_status) return item.display_status;" in display_status_block
    assert "if (state === 'observing') return 'warning';" in display_status_block
    assert "const displayStatus = systemAuditDisplayStatus(node);" in node_block
    assert "systemAuditTone(systemAuditDisplayStatus(left))" in node_block
    assert "systemAuditStatusLabel(displayStatus)" in node_block
    assert "systemAuditTone(card.status)" in card_block
    assert "systemAuditDisplayStatus(card)" not in card_block


def test_system_audit_static_assets_keep_new_version() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")

    assert "dashboard.css?v=20260624-threshold-governance-strip" in html
    assert "dashboard.js?v=20260624-threshold-governance-strip" in html
    assert "dashboard.css?v=20260621-data-sync" not in html
    assert "dashboard.js?v=20260621-data-sync" not in html


def test_trading_settings_show_threshold_governance_catalog() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert "阈值治理与手动配置说明" in html
    assert "手动硬风控上限" in html
    assert 'id="cfg-max-position-pct"' in html
    assert 'id="cfg-max-leverage"' in html
    assert 'id="cfg-max-daily-loss-pct"' in html
    assert 'id="cfg-hard-stop-loss-pct"' in html
    assert 'id="cfg-max-open-positions-per-model"' in html
    assert 'id="cfg-max-same-symbol-positions-per-side"' in html
    assert 'id="threshold-governance-summary"' in html
    assert 'id="threshold-governance-strip"' in html
    assert 'id="threshold-governance-strip-text"' in html
    assert 'id="threshold-manual-count"' in html
    assert 'id="threshold-auto-count"' in html
    assert 'id="threshold-hard-count"' in html
    assert 'id="threshold-removed-count"' in html
    assert 'id="threshold-manual-editable"' in html
    assert 'id="threshold-service-controls"' in html
    assert 'id="threshold-auto-tunable"' in html
    assert 'id="threshold-manual-hard-guards"' in html
    assert 'id="threshold-removed-deprecated"' in html

    assert "function fetchThresholdCatalog" in script
    assert "fetchJSON('/api/settings/threshold-catalog')" in script
    assert "if (selected === 'trading') fetchTradingParams();" in script
    assert "renderThresholdCatalogList('threshold-manual-editable'" in script
    assert "renderThresholdCatalogList('threshold-auto-tunable'" in script
    assert "renderThresholdCatalogList('threshold-removed-deprecated'" in script
    assert "threshold-governance-strip-text" in script
    assert "setCount('threshold-manual-count'" in script
    assert "setCount('threshold-hard-count'" in script
    assert "body.max_position_pct = pct / 100;" in script
    assert "body.max_leverage = leverage;" in script
    assert "body.max_daily_loss_pct = pct / 100;" in script
    assert "body.hard_stop_loss_pct = pct / 100;" in script
    assert "body.max_open_positions_per_model = value;" in script
    assert "body.max_same_symbol_positions_per_side = value;" in script
    assert "硬风险上限不会自动放松" in script
    assert "自动调度项不放进手动输入框" in script

    assert ".threshold-governance-grid" in style
    assert ".threshold-governance-strip" in style
    assert ".threshold-governance-impact" in style


def test_data_collection_page_is_wired_to_api_and_safe_layout() -> None:
    html = (PROJECT_ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = (PROJECT_ROOT / "web_dashboard/static/css/dashboard.css").read_text(encoding="utf-8")

    assert 'data-page="data-collection"' in html
    assert 'id="page-data-collection"' in html
    assert "\u6570\u636e\u91c7\u96c6\u7ba1\u7406" in html
    assert "\u5916\u90e8\u4e8b\u4ef6\u91c7\u96c6\u8bbe\u7f6e" in html
    assert 'data-settings-tab="external-events"' in html
    assert 'data-settings-section="external-events"' in html
    assert html.index('data-settings-tab="models"') < html.index(
        'data-settings-tab="external-events"'
    )
    assert html.index('data-settings-tab="external-events"') < html.index(
        'data-settings-tab="vector-memory"'
    )
    assert html.index('data-settings-tab="vector-memory"') < html.index(
        'data-settings-tab="security"'
    )
    page_start = html.index('id="page-data-collection"')
    page_end = html.index('id="page-server-monitor"')
    settings_start = html.index('data-settings-section="models"')
    settings_end = html.index('data-settings-section="external-events"')
    external_start = html.index('data-settings-section="external-events"')
    external_end = html.index('data-settings-section="vector-memory"')
    vector_start = html.index('data-settings-section="vector-memory"')
    vector_end = html.index('data-settings-section="security"')
    data_page_html = html[page_start:page_end]
    model_settings_html = html[settings_start:settings_end]
    external_settings_html = html[external_start:external_end]
    vector_settings_html = html[vector_start:vector_end]
    assert "\u542f\u7528 Scrapling \u5916\u90e8\u4e8b\u4ef6\u91c7\u96c6" not in data_page_html
    assert "\u542f\u7528 Scrapling \u5916\u90e8\u4e8b\u4ef6\u91c7\u96c6" not in model_settings_html
    assert "\u542f\u7528 Scrapling \u5916\u90e8\u4e8b\u4ef6\u91c7\u96c6" in external_settings_html
    assert "applyRecommendedDataCollectionSources()" in external_settings_html
    assert 'id="data-external-source-list"' in external_settings_html
    assert 'id="data-external-sources"' not in external_settings_html
    assert 'id="data-cryptopanic-api-key"' in external_settings_html
    assert 'id="data-coinmarketcal-api-key"' in external_settings_html
    assert 'id="data-newsapi-api-key"' in external_settings_html
    assert "\u5411\u91cf\u8bb0\u5fc6\u8bbe\u7f6e" in vector_settings_html
    assert 'id="vector-memory-enabled"' in vector_settings_html
    assert 'id="vector-memory-status-panel"' in vector_settings_html
    assert "fetchDataCollectionStatus()" in html
    assert "saveDataCollectionSettings()" in html
    assert "if (page === 'data-collection') fetchDataCollectionStatus();" in script
    assert "if (isPageActive('data-collection'))" in script
    assert "fetchDataCollectionStatus({ silent: true });" in script
    assert "selected === 'external-events'" in script
    assert "selected === 'vector-memory'" in script
    assert "selected === 'models') fetchDataCollectionStatus" not in script
    assert "applyRecommendedDataCollectionSources" in script
    assert "recommended_external_event_sources" in script
    assert "renderDataCollectionSourceManager" in script
    assert "addDataCollectionSource" in script
    assert "removeDataCollectionSource" in script
    assert "data-source-editor-status" in script
    assert "source.valid === false" in script
    assert "cryptopanic_api_key" in script
    assert "groupDataCollectionSources" in script
    assert "fetchJSON('/api/data-collection/status')" in script
    assert 'id="data-collection-feature-coverage"' in html
    assert "renderDataCollectionFeatureCoverage" in script
    assert "feature_coverage" in script
    assert "缺失特征" in script
    assert "中性阻断" in script
    assert ".data-feature-coverage-grid" in style
    assert ".data-feature-row" in style
    assert "postJSON('/api/data-collection/settings', body)" in script
    assert "fetchJSON('/api/vector-memory/status')" in script
    assert "postJSON('/api/vector-memory/clear', {})" in script
    assert "postJSON('/api/vector-memory/reindex', {})" in script
    assert "清空旧索引" in html
    assert "三期新样本向量索引" in html
    assert "三期新样本" in script
    assert "启用前请先清空旧索引" in script
    assert "等待三期新样本重新索引" in script
    assert "立即刷新索引" in vector_settings_html
    assert "renderAnalysisVectorMemory" in script
    assert "\u975e\u786c\u62e6\u622a" in script
    assert "\u5f71\u54cd ${deltaLabel}" in script
    assert ".analysis-note-positive" in style
    assert ".analysis-note-warning" in style
    assert "unknown: '\u5df2\u8fde\u63a5'" in script
    assert "collectionStatusLabel" in script
    assert "readDataCollectionSources" in script
    assert ".data-collection-health-strip" in style
    assert ".settings-data-collection-card" in style
    assert ".data-source-line" in style
    assert ".data-source-editor-row" in style
    assert ".data-source-editor-status" in style
    assert "dashboard.css?v=20260624-threshold-governance-strip" in html
    assert "dashboard.js?v=20260624-threshold-governance-strip" in html
    assert "overflow-wrap: anywhere;" in style


def test_strategy_learning_candidate_lab_prevents_card_overflow() -> None:
    style = (PROJECT_ROOT / "web_dashboard/static/css/strategy_learning.css").read_text(
        encoding="utf-8"
    )

    assert "grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr));" in style
    assert "grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));" in style
    assert "grid-template-columns: repeat(auto-fit, minmax(min(100%, 142px), 1fr));" in style
    assert ".strategy-learning-profile-chips span" in style
    assert "text-overflow: clip;" in style
    assert "overflow-wrap: anywhere;" in style
    assert "white-space: normal;" in style
    assert (
        "text-overflow: ellipsis;"
        not in style[
            style.index(".strategy-learning-profile-footer") : style.index(
                ".strategy-learning-guard-state"
            )
        ]
    )


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
async def test_display_open_position_symbols_never_falls_back_to_local_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def local_symbols(_mode: str | None = None) -> set[str]:
        raise AssertionError("Phase 3 current-position display must not use local DB fallback")

    async def exchange_symbols(_mode: str | None = None) -> set[str]:
        return set()

    monkeypatch.setattr(dashboard, "_get_open_position_symbols", local_symbols)
    monkeypatch.setattr(dashboard, "_get_exchange_open_position_symbols", exchange_symbols)

    assert await dashboard._get_display_open_position_symbols("paper") == set()


def test_opening_funnel_request_failure_uses_unavailable_fallback() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    fetch_block = script[
        script.index("async function fetchOpeningFunnel") : script.index(
            "function renderOpeningFunnelUnavailable"
        )
    ]

    assert "try {" in fetch_block
    assert (
        "renderOpeningFunnelUnavailable({ detail: err?.message || '开仓漏斗接口请求失败' })"
        in fetch_block
    )


def test_local_ml_dashboard_request_failure_degrades_per_endpoint() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    fetch_block = script[
        script.index("async function fetchMLSignalDashboard") : script.index(
            "function renderMLSignalDashboard"
        )
    ]

    assert "fetchJSON('/api/ml-signal/status').catch(err => ({" in fetch_block
    assert "status: 'request_error'" in fetch_block
    assert "fetchJSON('/api/local-ai-tools/status').catch(err => ({" in fetch_block
    assert "本地 ML 状态接口请求失败" in fetch_block
    assert "本地量化工具状态接口请求失败" in fetch_block


def test_fetch_json_throws_errors_so_page_fallbacks_run() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    fetch_block = script[
        script.index("async function fetchJSON") : script.index("function redirectToLogin")
    ]

    assert "const data = await res.json().catch(() => ({}));" in fetch_block
    assert "redirectToLogin(message);" in fetch_block
    assert "throw new Error(message);" in fetch_block
    assert "throw new Error(apiErrorText(data, res.statusText || '请求失败'));" in fetch_block
    assert "throw e;" in fetch_block
    assert "return null;" not in fetch_block


def test_local_ml_loss_filter_uses_backend_model_contract() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    status_block = script[
        script.index("function localModelStatus") : script.index("function mlSampleCounts")
    ]

    assert "loss_filter: 'profit_prediction'" in status_block
    assert (
        "loss_filter: ['loss_filter', 'loss_model', 'loss_probability', 'risk_filter']"
        in status_block
    )
    assert "localModelStatus(local, 'loss_filter')" in script


def test_ml_signal_dashboard_renders_readiness_blockers() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    overview_block = script[
        script.index("function renderMLSignalOverview") : script.index(
            "function renderLocalAIToolsStatus"
        )
    ]

    assert "status.readiness || {}" in overview_block
    assert "readiness.blocking_reasons" in overview_block
    assert "allow_live_position_influence" in overview_block
    assert "readiness.metrics" in overview_block
    assert "dirty_sample_ratio" in overview_block
    assert "long_pr_auc" in overview_block
    assert "short_pr_auc" in overview_block


def test_server_monitor_gpu_summary_uses_all_cards() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    block = script[
        script.index("function serverMonitorGpuSummary") : script.index(
            "function renderServerMonitor"
        )
    ]

    assert "reduce((sum, gpu) => sum + Number(gpu.memory_used_mb || 0), 0)" in block
    assert "reduce((sum, gpu) => sum + Number(gpu.memory_total_mb || 0), 0)" in block
    assert "rows.length" in block
    assert "8" not in block
    assert "const gpu = (data.gpu?.gpus || [])[0] || {};" not in script
    assert "gpuSummary.memory_total_mb" in script
    assert "data.phase3_model_server_gpu || {}" in script
    assert "liveGpuRows.length ? liveGpuPayload : phase3GpuPayload" in script
    assert "卡汇总" in script


def test_data_collection_ui_explains_phase3_clean_training_view() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "phase3CleanCount" in script
    assert "旧数据参与训练" not in script
    assert "只使用干净训练视图" in script
    assert "冷启动待补齐，不驱动开仓" in script
    assert "缺失/过期特征默认中性阻断" in script
    assert "三期重新开始训练；旧数据禁止进入新模型训练" in script
    assert "三期相似样本记忆" in script


def test_system_audit_okx_details_renders_root_cause_and_training_policy() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    okx_block = script[
        script.index("function systemAuditOkxDetailsV2") : script.index(
            "function systemAuditCardDetailsHtml"
        )
    ]

    assert "root_cause_summary" in okx_block
    assert "training_data_policy" in okx_block
    assert "OKX root causes" in okx_block
    assert "Training data policy" in okx_block
    assert "requires_training_rebuild" in okx_block
    assert "systemAuditOkxDetailsV2(details)" in script


def test_system_audit_okx_details_renders_runtime_entry_gate() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    okx_block = script[
        script.index("function systemAuditOkxDetailsV2") : script.index(
            "function systemAuditCardDetailsHtml"
        )
    ]

    assert "runtime_okx_entry_gate" in okx_block
    assert "Runtime OKX entry gate" in okx_block
    assert "Runtime OKX sync result kinds" in okx_block
    assert "Runtime OKX sync samples" in okx_block
    assert "Entry gate" in okx_block
    assert "runtimeGate.blocker" in okx_block
    assert "Requires attention" in okx_block
    assert "runtimeSampleRows" in okx_block
    assert "last_samples" in okx_block


def test_system_audit_position_price_details_render_mismatch_root_causes() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    position_block = script[
        script.index("function systemAuditPositionPriceDetails") : script.index(
            "function systemAuditStrategySignalRootCauseDetails"
        )
    ]

    assert "root_cause_summary" in position_block
    assert "local_only_positions" in position_block
    assert "exchange_only_positions" in position_block
    assert "Position mismatch root causes" in position_block
    assert "OKX position mode counts" in position_block
    assert "OKX side inference counts" in position_block
    assert "Price/PnL split samples" in position_block
    assert "Local-only open positions" in position_block
    assert "OKX-only open positions" in position_block
    assert "okx_pos_side" in position_block
    assert "okx_raw_pos" in position_block
    assert "okx_side_inference" in position_block
    assert "Raw pos" in position_block
    assert "systemAuditPositionPriceDetails(details)" in script


def test_system_audit_strategy_signal_details_render_scheduler_diagnostics() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    signal_block = script[
        script.index("function systemAuditStrategySignalRootCauseDetails") : script.index(
            "function systemAuditCardDetailsHtml"
        )
    ]

    assert "details.scheduler" in signal_block
    assert "dynamic_capacity" in signal_block
    assert "Scheduler strategy distribution" in signal_block
    assert "Scheduler flags" in signal_block
    assert "Dynamic capacity reason codes" in signal_block
    assert "Latest scheduler samples" in signal_block
    assert "strategy_learning_context_timeout" in signal_block
    assert "systemAuditStrategySignalRootCauseDetails(details)" in script


def test_ml_signal_dashboard_renders_controlled_degraded_as_observing() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    overview_block = script[
        script.index("function renderMLSignalOverview") : script.index(
            "function renderLocalAIToolsStatus"
        )
    ]

    assert "controlledReadinessDegrade" in overview_block
    assert (
        "const readinessDisplayState = controlledReadinessDegrade ? '学习观察' : readinessState;"
        in overview_block
    )
    assert (
        "const readinessTone = allowLivePositionInfluence ? 'good' : (ready ? 'warn' : 'bad');"
        in overview_block
    )
    assert "readinessState === 'degraded' ? 'bad' : 'warn'" not in overview_block


def test_data_collection_request_failure_still_renders_error_fallback() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    fetch_block = script[
        script.index("async function fetchDataCollectionStatus") : script.index(
            "function collectionStatusTone"
        )
    ]

    assert "try {" in fetch_block
    assert "数据采集状态接口请求失败" in fetch_block
    assert "state.dataCollectionStatus = data || null;" in script
    assert "renderDataCollectionDashboard()" in script


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

    async def open_positions(
        mode: str | None = None,
        ticker_overrides: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        assert mode == "paper"
        assert ticker_overrides == {"ETH/USDT": {"price": 2500.0}}
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

    async def open_positions(
        mode: str | None = None,
        ticker_overrides: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
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


@pytest.mark.asyncio
async def test_market_snapshot_keeps_nonzero_feed_change_when_position_change_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def open_symbols(mode: str | None = None) -> set[str]:
        return {"ETH/USDT"}

    async def feed_tickers(
        symbols: set[str],
        market_tickers: dict[str, Any],
        mode: str | None = None,
    ) -> dict[str, Any]:
        return {
            "ETH/USDT": {
                "price": 2500.0,
                "change_24h_pct": -2.4,
                "volume_24h": 123.0,
                "bid": 2499.0,
                "ask": 2501.0,
            }
        }

    async def open_positions(
        mode: str | None = None,
        ticker_overrides: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        assert ticker_overrides == {
            "ETH/USDT": {
                "price": 2500.0,
                "change_24h_pct": -2.4,
                "volume_24h": 123.0,
                "bid": 2499.0,
                "ask": 2501.0,
            }
        }
        return [
            {
                "symbol": "ETH/USDT",
                "side": "long",
                "current_price": 2500.0,
                "entry_price": 2400.0,
                "change_24h": 0.0,
                "is_open": True,
            }
        ]

    monkeypatch.setattr(dashboard, "_data_service", None)
    monkeypatch.setattr(dashboard, "_get_display_open_position_symbols", open_symbols)
    monkeypatch.setattr(dashboard, "_build_dashboard_tickers", feed_tickers)
    monkeypatch.setattr(dashboard, "_get_display_open_positions_snapshot", open_positions)

    payload = await dashboard._build_open_position_market_snapshot("paper")

    assert payload["tickers"]["ETH/USDT"]["change_24h"] == -2.4
    assert payload["tickers"]["ETH/USDT"]["volume_24h"] == 123.0


def test_parse_public_tickers_accepts_internal_market_state_fields() -> None:
    parsed = dashboard._parse_public_tickers(
        {
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "last_price": 2500.0,
                "change_24h_pct": -2.4,
                "volume_24h": 123.0,
                "bid": 2499.0,
                "ask": 2501.0,
            }
        },
        {"ETH/USDT"},
    )

    assert parsed["ETH/USDT"]["price"] == 2500.0
    assert parsed["ETH/USDT"]["change_24h"] == -2.4


def test_parse_public_tickers_uses_okx_swap_base_volume_for_notional() -> None:
    parsed = dashboard._parse_public_tickers(
        {
            "PEPE/USDT": {
                "symbol": "PEPE/USDT",
                "last": 0.000002355,
                "info": {
                    "instId": "PEPE-USDT-SWAP",
                    "vol24h": "5357584.8",
                    "volCcy24h": "53575848000000",
                },
            }
        },
        {"PEPE/USDT"},
    )

    assert parsed["PEPE/USDT"]["volume_24h"] == pytest.approx(53_575_848_000_000)
    assert parsed["PEPE/USDT"]["volume_24h_contracts"] == pytest.approx(5_357_584.8)
    assert parsed["PEPE/USDT"]["notional_24h_usdt"] == pytest.approx(126_171_122.04)


def test_dashboard_js_preserves_market_change_when_position_snapshot_change_is_zero() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    position_builder = script[
        script.index("function buildTickersFromPositions") : script.index(
            "function buildPositionTickers"
        )
    ]
    market_update = script[
        script.index("function updateMarketData") : script.index("function decisionSizeTitle")
    ]

    assert "const positionChange =" in position_builder
    assert "Number(positionChange) !== 0" in position_builder
    assert "const shouldKeepMarketChange =" in market_update
    assert "Number(tickerChange) === 0" in market_update
    assert "change_24h: shouldKeepMarketChange ? marketChange" in market_update


def test_position_ticker_snapshot_uses_backend_market_contract() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    snapshot_block = script[
        script.index("async function fetchPositionTickerSnapshot") : script.index(
            "function filterTickersToOpenPositions"
        )
    ]

    assert "/api/dashboard/market" in snapshot_block
    assert "updateMarketData(data, state.accounts || [])" in snapshot_block
    assert "/api/dashboard/positions" not in snapshot_block
    assert "enrichTickersFromOKX" not in snapshot_block


def test_parse_public_tickers_uses_okx_open24h_baseline() -> None:
    parsed = dashboard._parse_public_tickers(
        {
            "ETH/USDT": {
                "symbol": "ETH/USDT",
                "last": 105.0,
                "open24h": 100.0,
            }
        },
        {"ETH/USDT"},
    )

    assert parsed["ETH/USDT"]["change_24h"] == 5.0


def test_parse_public_tickers_prefers_okx_24h_change_over_sod_utc8() -> None:
    parsed = dashboard._parse_public_tickers(
        {
            "AUCTION/USDT": {
                "symbol": "AUCTION/USDT",
                "last": 3.53,
                "percentage": -0.2261164499717354,
                "info": {
                    "instId": "AUCTION-USDT-SWAP",
                    "last": "3.53",
                    "open24h": "3.538",
                    "sodUtc8": "3.53",
                },
            }
        },
        {"AUCTION/USDT"},
    )

    assert parsed["AUCTION/USDT"]["change_24h"] == pytest.approx(-0.2261164499717354)


@pytest.mark.asyncio
async def test_open_position_tickers_use_public_change_when_market_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def public_tickers(symbols: set[str]) -> dict[str, Any]:
        return {
            "ETH/USDT": {
                "price": 2500.0,
                "change_24h": -1.7,
                "volume_24h": 321.0,
                "bid": 2499.0,
                "ask": 2501.0,
            }
        }

    async def exchange_marks(mode: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            ("ETH/USDT", "long"): {
                "mark_price": 2502.0,
                "entry_price": 2400.0,
                "quantity": 1.0,
            }
        }

    monkeypatch.setattr(dashboard, "_get_public_ticker_map", public_tickers)
    monkeypatch.setattr(dashboard, "_get_exchange_position_mark_map", exchange_marks)

    tickers = await dashboard._build_tickers_for_open_positions(
        {"ETH/USDT"},
        {},
        "paper",
    )

    assert tickers["ETH/USDT"]["price"] == 2502.0
    assert tickers["ETH/USDT"]["change_24h"] == -1.7
    assert tickers["ETH/USDT"]["volume_24h"] == 321.0


def test_analysis_pre_expert_skip_contract_is_not_reported_as_model_config_error() -> None:
    script = (PROJECT_ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")

    assert "function analysisPreExpertSkip" in script
    assert "expert_call_status" in script
    assert "行情预检未进入专家" in script
    assert "预检跳过专家" in script
    assert "不是模型故障" in script
    assert "没有消耗大模型专家" in script
    assert "pre_expert_skipped" in script
    assert "未配置 API Key" in script
    assert script.index("pre_expert_skipped") < script.index("cfg && cfg.loading")
    detail_start = script.index("async function showAnalysisReason")
    detail_end = script.index("function changeAnalysisPage", detail_start)
    assert "isFastScan" not in script[detail_start:detail_end]
    assert "try {" in script[detail_start:detail_end]
    assert "renderAnalysisReasonModal(record)" in script[detail_start:detail_end]
    assert "详情渲染失败" in script[detail_start:detail_end]
    assert "function renderAnalysisReasonModal" in script[detail_start:detail_end]


def test_dashboard_api_normalizes_market_prefilter_expert_status() -> None:
    raw = {
        "fast_prefilter": {
            "skipped_llm": True,
            "reason": "短周期行情特征疑似缺失：1/5/20周期收益率和波动率都为0。",
        }
    }

    status = dashboard._analysis_pre_expert_skip(raw)

    assert status == {
        "skipped": True,
        "kind": "market_prefilter",
        "label": "行情预检未进入专家",
        "reason": "短周期行情特征疑似缺失：1/5/20周期收益率和波动率都为0。",
    }


def test_dashboard_api_normalizes_position_fast_scan_expert_status() -> None:
    status = dashboard._analysis_pre_expert_skip({"position_fast_scan": {"skipped_llm": True}})

    assert status["skipped"] is True
    assert status["kind"] == "position_fast_scan"
    assert status["label"] == "持仓快速扫描未进入专家"
    assert "强平仓" in status["reason"]
