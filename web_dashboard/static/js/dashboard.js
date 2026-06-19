/**
 * Main dashboard logic.
 * Connects WebSocket, polls REST API, renders all components.
 */

// State
const state = {
    mode: 'paper',
    paused: false,
    scanMode: 'auto',
    liveModel: null,
    models: [],
    tickers: {},
    decisions: [],
    executions: [],
    rankings: [],
    accounts: [],
    aiExpertModels: [],
    riskEvents: [],
    riskEventsPage: 1,
    tradeMode: 'paper',
    decisionsTotal: 0,
    todayDecisionsTotal: 0,
    tradesTotal: 0,
    openPositionsTotal: 0,
    decisionInterval: 60,
    allTrades: [],
    tradesPage: 1,
    tradesPageMode: '',
    positionsPage: 1,
    positionsTotal: 0,
    positionHistoryPage: 1,
    positionHistoryTotal: 0,
    dailyPnlRecords: [],
    allDecisions: [],
    decisionsPage: 1,
    analysisRecords: [],
    analysisPage: 1,
    analysisTotal: 0,
    analysisTotalPages: 1,
    analysisView: 'market',
    modelModeMap: {},  // model_name -> execution_mode
    positionTickerSymbols: [],
    priceChartSymbol: '',
    priceChartTimeframe: '1h',
    executionAccount: null,
    okxConfig: {
        paperConfigured: true,
        liveConfigured: null,
    },
    expertMemories: [],
    tradeReflections: [],
    expertMemoryPage: 1,
    expertMemoryTotal: 0,
    tradeReflectionPage: 1,
    tradeReflectionTotal: 0,
    expertMemoryView: 'memories',
    shadowBacktests: [],
    shadowBacktestPage: 1,
    shadowBacktestTotal: 0,
    shadowBacktestStatus: '',
    mlSignalStatus: null,
    localAIToolsStatus: null,
    serverMonitorStatus: null,
    serverMonitorTab: 'self-check',
    systemSelfCheck: null,
    mlSignalRecords: [],
    mlSignalPage: 1,
    tradesTotalPages: 1,
    openingFunnel: null,
    profitAttribution: null,
    profitAttributionView: 'overview',
    profitAttributionRecordPage: 1,
};
const PAGE_SIZE = 20;
const EXPERT_MEMORY_PAGE_SIZE = 10;
const RISK_ALERT_PAGE_SIZE = 10;
const ML_SIGNAL_PAGE_SIZE = 10;
const PROFIT_ATTRIBUTION_RECORD_PAGE_SIZE = 10;
const FIXED_AI_EXPERT_FALLBACKS = [
    {
        name: 'trend_expert',
        label: '行情方向专家',
        role: 'trend_direction',
        description: '判断当前交易对更适合做多、做空、震荡观望或方向不确定，不直接决定仓位。',
    },
    {
        name: 'momentum_expert',
        label: '盈利质量专家',
        role: 'profit_quality',
        description: '判断预期净收益、亏损概率、盈亏比、手续费覆盖和小赚大亏风险。',
    },
    {
        name: 'sentiment_expert',
        label: '短线时序专家',
        role: 'short_timeseries',
        description: '判断未来 1/5/10/30 分钟路径、动量延续、反转、假突破和事件冲击。',
    },
    {
        name: 'position_expert',
        label: '持仓退出专家',
        role: 'position_exit',
        description: '只看已有仓位，判断继续拿、锁盈、减仓、全平、亏损修复或加仓条件。',
    },
    {
        name: 'risk_expert',
        label: '异常风控专家',
        role: 'risk_anomaly',
        description: '检查异常插针、流动性、极端波动、保证金、交易所限制和硬风险。',
    },
    {
        name: 'decision_maker',
        label: '最终交易员',
        role: 'final_decision',
        description: '读取专家协作结果后，以真实盈利最大化为目标做最终开仓、平仓或观望裁决。',
    },
];
let recentDecisionsRefreshTimer = null;
const okxTickerCache = {};
let positionsRequestToken = 0;
const closingPositionIds = new Set();
let closingAllPositions = false;
let serverMonitorRefreshInFlight = null;
const THEME_STORAGE_KEY = 'dashboardTheme';

function isPageActive(page) {
    return document.getElementById(`page-${page}`)?.classList.contains('active');
}

// Init on page load
document.addEventListener('DOMContentLoaded', () => {
    initThemeToggle();
    initWebSocket();
    initCharts();
    initModeButtons();
    initSidebarNav();
    initTradeTabs();
    initSettingsTabs();
    initScanModeButtons();
    initPositionActions();
    initDashboardUserActions();
    initModalActionButtons();
    initServerMonitorTabs();
    fetchDashboardSummary();
    fetchPnlHistory();
    fetchRecentDecisions();
    fetchRecentExecutions();
    fetchRiskEvents();
    fetchDashboardAuthStatus();
    setInterval(() => {
        if (isPageActive('dashboard')) {
            fetchDashboardSummary();
        }
    }, 10000);
    setInterval(fetchPnlHistory, 60000);
    setInterval(fetchRecentDecisions, 30000);
    setInterval(fetchRecentExecutions, 30000);
    setInterval(fetchTrades, 60000);
    setInterval(fetchDashboardAuthStatus, 60000);
    setInterval(() => {
        if (isPageActive('positions')) {
            fetchPositions();
        }
    }, 15000);
    setInterval(() => {
        if (isPageActive('server-monitor')) {
            refreshServerMonitorPage();
        }
    }, 15000);
    fetchDashboardAccountSettings();
    fetchModelServerSettings();
    fetchOKXSettings();
    fetchExecutionAccountSettings();
    fetchAIModels();
});

// --- WebSocket ---
function initWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WSClient(`${protocol}//${location.host}/ws`);

    ws.on('connected', () => {
        document.getElementById('ws-dot').className = 'ws-dot ws-connected';
    });
    ws.on('disconnected', () => {
        document.getElementById('ws-dot').className = 'ws-dot ws-disconnected';
    });
    ws.on('message', handleWSMessage);

    ws.connect();
    window._ws = ws;
}

function handleWSMessage(data) {
    switch (data.type) {
        case 'ticker_update':
            const incomingTickers = data.symbols || data;
            const filteredTickers = filterTickersToOpenPositions(incomingTickers);
            if (Object.keys(filteredTickers).length) {
                updateTickers(filteredTickers, { replace: true });
            }
            break;
        case 'trading_round':
            // Filter decisions/executions by current mode
            const modeIsPaper = state.mode === 'paper';
            updateDecisions((data.decisions || []).filter(d => (d.is_paper !== false) === modeIsPaper));
            updateExecutions((data.executions || []).filter(e => (e.is_paper !== false) === modeIsPaper));
            updateStats(data.stats || {});
            break;
        case 'risk_alert':
            addRiskAlert(data);
            break;
    }
}

// --- Charts ---
function initCharts() {
    const charts = new DashboardCharts();
    charts.initPnLChart('pnl-chart');
    charts.initPriceChart('price-chart');
    charts.applyTheme();
    window._charts = charts;
}

// --- Theme ---
function getStoredTheme() {
    try {
        const theme = localStorage.getItem(THEME_STORAGE_KEY);
        return theme === 'light' || theme === 'dark' ? theme : 'dark';
    } catch (_) {
        return 'dark';
    }
}

function applyDashboardTheme(theme) {
    const normalizedTheme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.dataset.theme = normalizedTheme;
    state.theme = normalizedTheme;

    const toggle = document.getElementById('theme-toggle');
    const icon = document.getElementById('theme-toggle-icon');
    const text = document.getElementById('theme-toggle-text');
    const nextLabel = normalizedTheme === 'light' ? '切换为深色模式' : '切换为浅色模式';

    if (toggle) toggle.setAttribute('aria-label', nextLabel);
    if (icon) icon.textContent = normalizedTheme === 'light' ? '🌙' : '☀️';
    if (text) text.textContent = normalizedTheme === 'light' ? '深色模式' : '浅色模式';

    if (window._charts?.applyTheme) {
        window._charts.applyTheme();
    }
}

function initThemeToggle() {
    applyDashboardTheme(document.documentElement.dataset.theme || getStoredTheme());

    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            const nextTheme = state.theme === 'light' ? 'dark' : 'light';
            try {
                localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
            } catch (_) {}
            applyDashboardTheme(nextTheme);
        });
    }

    window.addEventListener('storage', event => {
        if (event.key === THEME_STORAGE_KEY) {
            applyDashboardTheme(event.newValue || 'dark');
        }
    });
}

// --- API Calls ---
const DASHBOARD_ADMIN_KEY_STORAGE_KEYS = [
    'dashboard_admin_api_key',
    'dashboardAdminApiKey',
];

function getDashboardAdminKey() {
    for (const key of DASHBOARD_ADMIN_KEY_STORAGE_KEYS) {
        try {
            const value = sessionStorage.getItem(key);
            if (value && value.trim()) return value.trim();
        } catch (_) {}
    }
    return '';
}

function dashboardWriteOptions(options = {}) {
    const headers = { ...(options.headers || {}) };
    const adminKey = getDashboardAdminKey();
    if (adminKey && !headers.Authorization && !headers['X-Dashboard-Admin-Key']) {
        headers['X-Dashboard-Admin-Key'] = adminKey;
    }
    return { ...options, headers };
}

function apiErrorText(data, fallback = '未知错误') {
    if (!data) return fallback;
    if (typeof data === 'string') return data.trim() || fallback;
    if (typeof data !== 'object') return fallback;
    const detail = data.detail ?? data.error ?? data.message ?? data.rejection_reason;
    if (detail && typeof detail === 'object') {
        const message = String(detail.message || detail.error || detail.reason || '').trim();
        const missing = Array.isArray(detail.missing_fields) && detail.missing_fields.length
            ? `缺少：${detail.missing_fields.join('、')}`
            : '';
        return [message, missing].filter(Boolean).join('；') || fallback;
    }
    return String(detail || fallback).trim() || fallback;
}

async function fetchJSON(url) {
    try {
        const res = await fetch(url, { cache: 'no-store' });
        if (res.status === 401) {
            redirectToLogin('登录已过期，请重新登录。');
            return null;
        }
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            console.error(`Fetch failed: ${url}`, data);
            return null;
        }
        return data;
    } catch (e) {
        console.error(`Fetch failed: ${url}`, e);
        return null;
    }
}

function redirectToLogin(message = '') {
    try {
        if (message) sessionStorage.setItem('dashboard_login_notice', message);
    } catch (_) {}
    if (!location.pathname.startsWith('/login')) {
        window.location.href = '/login';
    }
}

async function fetchWithAuth(url, options = {}, expiredMessage = '登录已过期，请重新登录。') {
    const res = await fetch(url, options);
    if (res.status === 401) {
        redirectToLogin(expiredMessage);
        throw new Error(expiredMessage);
    }
    return res;
}

async function postJSON(url, body = {}) {
    const res = await fetchWithAuth(url, dashboardWriteOptions({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        throw new Error(apiErrorText(data, res.statusText || '请求失败'));
    }
    return data;
}

async function putJSON(url, body = {}) {
    const res = await fetchWithAuth(url, dashboardWriteOptions({
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        throw new Error(apiErrorText(data, res.statusText || '请求失败'));
    }
    return data;
}

async function dashboardUserWriteRequest(url, options = {}) {
    const res = await fetchWithAuth(
        url,
        dashboardWriteOptions(options),
        '登录已过期，请重新登录后再操作会员。',
    );
    const data = await res.json().catch(() => ({}));
    if (res.status === 403) {
        throw new Error('当前登录账号没有执行该操作的权限。');
    }
    if (!res.ok) {
        throw new Error(apiErrorText(data, res.statusText || '会员操作失败'));
    }
    return data;
}

function setButtonBusy(button, busy, label = '') {
    if (!button) return;
    if (busy) {
        button.dataset.originalText = button.textContent || '';
        button.disabled = true;
        if (label) button.textContent = label;
        return;
    }
    button.disabled = false;
    if (button.dataset.originalText) {
        button.textContent = button.dataset.originalText;
        delete button.dataset.originalText;
    }
}

async function logoutDashboard() {
    try {
        await fetch('/api/auth/logout', dashboardWriteOptions({
            method: 'POST',
            credentials: 'include',
        }));
    } catch (error) {
        console.debug('dashboard logout request failed', error);
    } finally {
        redirectToLogin('已退出登录。');
    }
}

async function fetchDashboardSummary() {
    const data = await fetchJSON('/api/dashboard/summary');
    if (!data) return;

    updateModeDisplay(data.mode, data.paused, data.scan_mode);
    updateExecutionAccountPanel(data.execution_account || {});
    updateAccounts(data.accounts || [], data.execution_account || null);
    updateMarketData(data.market || {}, data.accounts || []);
    updateStats(data);
    updateDashboardDecisionCounts(data);
    updateSymbolCount();
    fetchModeCounts();
}

async function fetchPnlHistory() {
    const mode = state.mode || 'paper';
    const data = await fetchJSON(`/api/dashboard/pnl-history?mode=${mode}&_=${Date.now()}`);
    if (!data || !data.history || !window._charts) return;
    window._charts.updatePnLChart(data.history);
}

async function fetchDailyPnlRecords() {
    const tbody = document.getElementById('daily-pnl-tbody');
    if (tbody) {
        tbody.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);text-align:center;padding:24px;">加载每日盈亏中...</td></tr>';
    }
    const days = Number(document.getElementById('daily-pnl-days')?.value || 30);
    const mode = state.mode || 'paper';
    const data = await fetchJSON(`/api/dashboard/daily-pnl?mode=${mode}&days=${days}&_=${Date.now()}`);
    if (!data || !Array.isArray(data.records)) {
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="9" style="color:var(--red);text-align:center;padding:24px;">每日盈亏加载失败</td></tr>';
        }
        return;
    }
    state.dailyPnlRecords = data.records;
    const subtitle = document.getElementById('daily-pnl-subtitle');
    if (subtitle) {
        subtitle.textContent = `${mode === 'live' ? '实盘' : '模拟盘'} · 北京时间 ${data.start_date || ''} 至 ${data.end_date || ''}`;
    }
    renderDailyPnlRecords(data.records);
}

function updateDecisionBadge(total) {
    const badge = document.getElementById('decision-badge');
    if (!badge) return;
    const count = Number(total) || 0;
    badge.textContent = count;
    badge.style.display = count > 0 ? '' : 'none';
}

function updateOpenPositionStat(total) {
    const count = Number(total) || 0;
    state.openPositionsTotal = count;
    const el = document.getElementById('stat-trades');
    if (el) el.textContent = count;
    updateDecisionPositionStatus();
}

function updateDecisionPositionStatus() {
    const dtEl = document.getElementById('status-decision-trade');
    if (dtEl) {
        dtEl.textContent = state.decisionsTotal + ' / ' + state.openPositionsTotal;
    }
}

function updateDashboardDecisionCounts(data) {
    if (!data) return;
    if (data.decisions_total !== undefined) {
        state.decisionsTotal = Number(data.decisions_total) || 0;
    }
    if (data.today_decisions_total !== undefined) {
        state.todayDecisionsTotal = Number(data.today_decisions_total) || 0;
        const el = document.getElementById('stat-decisions');
        if (el) el.textContent = state.todayDecisionsTotal;
    }
    updateDecisionPositionStatus();
}

async function fetchRecentDecisions() {
    const isPaper = state.mode === 'paper';
    const data = await fetchJSON(`/api/decisions?limit=5&is_paper=${isPaper}`);
    if (!data || !data.decisions) return;
    updateDecisionBadge(data.total ?? data.count);
    renderRecentDecisions(data.decisions);
}

async function fetchRecentExecutions() {
    const data = await fetchJSON(`/api/trades?limit=5&mode=${state.mode}`);
    if (!data || !data.trades) return;
    renderRecentExecutions(data.trades, data.total ?? data.count);
}

async function fetchModeCounts() {
    // Query mode-specific cumulative decisions from DB.
    // The second value in the status panel is current open positions, updated
    // from the dashboard account summary.
    const isPaper = state.mode === 'paper';
    const decData = await fetchJSON(`/api/decisions?limit=1&is_paper=${isPaper}`);

    if (decData) {
        state.decisionsTotal = decData.total ?? decData.count ?? 0;
        updateDecisionBadge(state.decisionsTotal);
    }

    updateDecisionPositionStatus();
}

function updateSymbolCount() {
    const el = document.getElementById('stat-symbols');
    if (!el) return;
    const count = Object.keys(state.tickers || {}).length;
    el.textContent = String(count);
}

async function fetchTrades() {
    const data = await fetchJSON(`/api/trades?limit=${PAGE_SIZE}&mode=${state.mode}&page=${state.tradesPage}`);
    if (!data) return;
    updateTradeTable(data.trades || [], state.mode, data.total ?? data.count);
}

async function fetchPositionTickerSnapshot() {
    const data = await fetchJSON(`/api/dashboard/positions?mode=${state.mode}&page=1&page_size=200&open_only=true`);
    if (!data || !data.positions) return;
    let tickers = buildTickersFromPositions(data.positions);
    state.positionTickerSymbols = Object.keys(tickers);
    tickers = await enrichTickersFromOKX(tickers);
    updateTickers(tickers, { replace: true });
    refreshAutoPriceChart();
}

function filterTickersToOpenPositions(tickers) {
    if (!tickers || typeof tickers !== 'object') return {};
    const open = new Set(state.positionTickerSymbols || []);
    if (!open.size) return {};
    return Object.fromEntries(
        Object.entries(tickers).filter(([symbol]) => open.has(symbol))
    );
}

function updatePositionsTable(positions, page = 1, totalPages = 1, totalItems = 0) {
    const tbody = document.getElementById('positions-tbody');
    const pagination = document.getElementById('positions-pagination');
    if (!tbody) return;

    if (!positions.length) {
        tbody.innerHTML = '<tr><td colspan="12" style="color:var(--text-muted);text-align:center;padding:24px;">暂无持仓记录</td></tr>';
        if (pagination) pagination.style.display = 'none';
        return;
    }

    tbody.innerHTML = positions.map(p => {
        const isOpen = p.is_open !== false;
        const pnl = isOpen ? (p.unrealized_pnl || 0) : (p.realized_pnl || p.unrealized_pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const tp = p.take_profit ? fmtPrice(p.take_profit) : '-';
        const sl = p.stop_loss ? fmtPrice(p.stop_loss) : '-';
        const closePrice = fmtPrice(p.current_price || p.entry_price);
        const closeTime = isOpen ? '-' : toBeijingTime(p.closed_at);
        const statusTag = p.exchange_synced === false
            ? '<span style="color:var(--red);font-weight:600;">交易所无仓位</span>'
            : isOpen
            ? '<span style="color:var(--accent-light);font-weight:600;">持有中</span>'
            : '<span style="color:var(--text-muted);">已平仓</span>';
        const rowStyle = isOpen ? '' : 'opacity:0.65;';
        return `
            <tr style="${rowStyle}">
                <td>${p.model_name || '-'}</td>
                <td>${p.symbol}</td>
                <td><span style="color:${p.side === 'long' ? 'var(--green)' : 'var(--red)'}">${sideLabel(p.side)}</span></td>
                <td>${statusTag}</td>
                <td>${p.quantity ? p.quantity.toFixed(6) : '-'}</td>
                <td>${fmtPrice(p.entry_price)}</td>
                <td>${closePrice}</td>
                <td style="color:${pnlColor};font-weight:500;">${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)}</td>
                <td>${tp}</td>
                <td>${sl}</td>
                <td>${toBeijingTime(p.opened_at)}</td>
                <td>${closeTime}</td>
            </tr>
        `;
    }).join('');
    renderPagination('positions-pagination', page, totalPages, totalItems, 'changePositionsPage');
}

// --- Render Functions ---
function updateModeDisplay(mode, paused, scanMode) {
    state.mode = mode;
    state.paused = paused;
    state.scanMode = 'auto';

    const badge = document.getElementById('mode-badge');
    if (paused) {
        badge.textContent = '已暂停';
        badge.className = 'status-badge status-paused';
    } else if (mode === 'live') {
        badge.textContent = '实盘';
        badge.className = 'status-badge status-live';
    } else {
        badge.textContent = '模拟盘';
        badge.className = 'status-badge status-paper';
    }

    document.querySelectorAll('.mode-btn[data-mode]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    updateModeButtonAvailability();

    const scanLabel = document.getElementById('scan-mode-label');
    if (scanLabel) scanLabel.textContent = '自动扫描全市场 · 智能调度';
}

function updateTickers(tickers, options = {}) {
    if (!tickers || typeof tickers !== 'object') return;
    const nextTickers = {};
    Object.entries(tickers).forEach(([sym, ticker]) => {
        const prev = state.tickers[sym] || {};
        nextTickers[sym] = { ...prev, ...ticker };
    });
    state.tickers = options.replace ? nextTickers : { ...state.tickers, ...nextTickers };

    const container = document.getElementById('ticker-list');
    const countEl = document.getElementById('ticker-count');
    if (!container) return;

    const symbols = Object.keys(state.tickers).sort((a, b) => a.localeCompare(b));
    if (countEl) countEl.textContent = symbols.length + ' 个币种';
    updateSymbolCount();

    if (!symbols.length) {
        container.innerHTML = '<div class="ticker-card"><div class="ticker-sym">---</div><div class="ticker-price" style="color:var(--text-muted)">暂无持仓币种</div></div>';
        updateAutoPriceChartTitle('');
        return;
    }

    container.innerHTML = symbols.map(sym => {
        const t = state.tickers[sym];
        const price = t.price || t.last_price || 0;
        const change = t.change_24h || t.change24h || 0;
        const isUp = change >= 0;
        return `
            <div class="ticker-card">
                <div class="ticker-sym">${sym}</div>
                <div class="ticker-price">${fmtPrice(price)}</div>
                <div class="ticker-chg ${isUp ? 'ticker-up' : 'ticker-down'}">${isUp ? '+' : ''}${fmtPct(change)}</div>
            </div>
        `;
    }).join('');
}

function valueNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function fmtMoney(value) {
    const n = valueNumber(value);
    return n === null ? '--' : n.toFixed(2);
}

function fmtRatioPct(value) {
    const n = valueNumber(value);
    return n === null ? '--' : (n * 100).toFixed(1) + '%';
}

function signedMoney(value) {
    const n = valueNumber(value) || 0;
    return `${n >= 0 ? '+' : ''}${fmtMoney(n)}`;
}

function updateModelRankings(rankings) {
    state.rankings = rankings || [];
}

function accountMoneyText(value, account = null) {
    if (account && account.balance_error) return '--';
    const number = valueNumber(value);
    return number === null ? '--' : fmtMoney(number);
}

function updateExecutionAccountPanel(account) {
    state.executionAccount = account || {};
    const container = document.getElementById('execution-account-panel');
    const liveSpan = document.getElementById('live-model-name');
    if (liveSpan) {
        liveSpan.textContent = state.executionAccount.account_name || '多专家执行账户';
    }
    if (!container) return;

    if (!account || !Object.keys(account).length) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">暂无账户</div>';
        return;
    }

    const modeLabel = account.mode === 'live' ? '实盘' : '模拟盘';
    const unrealizedPnl = valueNumber(account.unrealized_pnl) || 0;
    const totalPnl = valueNumber(account.cumulative_total_pnl ?? account.total_pnl) || 0;
    const todayRealizedPnl = valueNumber(account.today_closed_realized_pnl ?? account.today_realized_pnl) || 0;
    const todayTotalPnl = todayRealizedPnl + unrealizedPnl;
    const remainingAllocation = valueNumber(account.available_balance ?? account.okx_available_balance ?? account.remaining_allocation);
    const accountEquity = valueNumber(account.account_equity ?? account.okx_equity_balance ?? account.equity ?? account.wallet_balance);
    const positionMarginUsed = valueNumber(
        account.used_margin ?? account.okx_used_balance ?? account.position_margin_used ?? account.paper_execution_used_margin
    ) || 0;
    const balanceSource = account.balance_source || (account.balance_snapshot_stale ? 'OKX 缓存快照' : '执行账户');
    const accountBalanceLabel = account.mode === 'live' ? 'OKX 实盘' : 'OKX 模拟盘';
    const pauseNote = account.risk_paused
        ? `<div class="exec-risk-note paused">已暂停分析新交易对：${escHtml(translatePauseReason(account.risk_pause_reason || '账户触发风险限制'))}</div>`
        : '<div class="exec-risk-note">系统按 OKX 实际可用余额计算仓位；已有持仓仍会继续复盘和平仓。</div>';

    container.innerHTML = `
        <div class="exec-account-card">
            <div class="exec-account-head">
                <div>
                    <div class="exec-account-name">${escHtml(account.account_name || '多专家执行账户')}</div>
                    <div class="exec-account-mode">${modeLabel} · ${escHtml(balanceSource)}${account.balance_snapshot_stale ? ` · 缓存 ${monitorNumber(account.balance_snapshot_stale_age_seconds, 1)}秒` : ''}</div>
                </div>
                <span class="badge ${account.risk_paused ? 'badge-short' : 'badge-long'}">${account.risk_paused ? '暂停开新仓' : '可分析'}</span>
            </div>
            <div class="exec-status-grid">
                <div class="exec-status-cell"><span>${accountBalanceLabel}可交易余额</span><strong>${accountMoneyText(remainingAllocation, account)} USDT</strong></div>
                <div class="exec-status-cell"><span>${accountBalanceLabel}账户权益</span><strong>${accountMoneyText(accountEquity, account)} USDT</strong></div>
                <div class="exec-status-cell"><span>持仓保证金占用</span><strong>${accountMoneyText(positionMarginUsed, account)} USDT</strong></div>
                <div class="exec-status-cell"><span>浮动盈亏</span><strong style="color:${unrealizedPnl >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(unrealizedPnl)} USDT</strong></div>
                <div class="exec-status-cell"><span>今日总盈亏</span><strong style="color:${todayTotalPnl >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(todayTotalPnl)} USDT</strong></div>
                <div class="exec-status-cell"><span>累计盈亏</span><strong style="color:${totalPnl >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(totalPnl)} USDT</strong></div>
            </div>
            ${account.balance_error ? `<div class="exec-risk-note paused">${escHtml(account.balance_error)}</div>` : pauseNote}
        </div>
    `;
}

function updateAccounts(accounts, executionAccount = null) {
    state.accounts = accounts || [];
    const container = document.getElementById('account-list');
    const account = executionAccount || state.executionAccount || state.accounts[0];
    const totalPositions = Number(account?.open_positions ?? 0) || state.accounts.reduce((sum, a) => sum + (a.open_positions || 0), 0);
    updateOpenPositionStat(totalPositions);
    const posBadge = document.getElementById('position-badge');
    if (posBadge) {
        posBadge.textContent = totalPositions;
        posBadge.style.display = totalPositions > 0 ? '' : 'none';
    }
    if (!container) return;

    if (!account || !Object.keys(account).length) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">暂无账户</div>';
        return;
    }

    const accountEquity = valueNumber(account.account_equity ?? account.okx_equity_balance ?? account.equity ?? account.wallet_balance);
    const remainingAllocation = valueNumber(account.available_balance ?? account.okx_available_balance ?? account.remaining_allocation);
    const totalPnl = valueNumber(account.cumulative_total_pnl ?? account.total_pnl) || 0;
    const unrealizedPnl = valueNumber(account.unrealized_pnl) || 0;
    const todayRealizedPnl = valueNumber(account.today_closed_realized_pnl ?? account.today_realized_pnl) || 0;
    const todayTotalPnl = todayRealizedPnl + unrealizedPnl;
    const pnlColor = totalPnl >= 0 ? 'var(--green)' : 'var(--red)';
    const accountBalanceLabel = account.mode === 'live' ? 'OKX 实盘' : 'OKX 模拟盘';
    container.innerHTML = `
        <div class="acct-row">
            <div class="acct-main">
                <div class="acct-name">${escHtml(account.account_name || account.model_name || '多专家执行账户')}</div>
                <div style="font-size:12px;color:var(--text);font-weight:700;">${accountBalanceLabel}可交易余额 ${accountMoneyText(remainingAllocation, account)} USDT</div>
                <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">${accountBalanceLabel}账户权益 ${accountMoneyText(accountEquity, account)} | 今日总盈亏（北京时间）${signedMoney(todayTotalPnl)}</div>
                ${account.balance_error ? `<div style="font-size:10px;color:var(--red);margin-top:4px;">${escHtml(account.balance_error)}</div>` : ''}
            </div>
            <div class="acct-side">
                <div class="acct-side-label">累计盈亏</div>
                <div class="acct-side-value" style="color:${pnlColor};">${signedMoney(totalPnl)} USDT</div>
            </div>
        </div>
    `;
}

function buildTickersFromPositions(positions) {
    const tickers = {};
    (positions || []).forEach(position => {
        if (position.is_open === false || !position.symbol) return;
        const price = position.current_price || position.entry_price || 0;
        if (!price) return;
        const previous = state.tickers[position.symbol] || {};
        tickers[position.symbol] = {
            price,
            change_24h: position.change_24h ?? previous.change_24h ?? previous.change24h ?? 0,
            volume_24h: 0,
            bid: 0,
            ask: 0,
        };
    });
    return tickers;
}

function buildPositionTickers(accounts) {
    const positions = [];
    (accounts || []).forEach(account => {
        (account.positions || []).forEach(position => {
            positions.push(position);
        });
    });
    return buildTickersFromPositions(positions);
}

function marketOpenPositions(market) {
    return Array.isArray(market?.open_positions) ? market.open_positions : [];
}

function toOKXSwapInstId(symbol) {
    const base = String(symbol || '').split('/')[0].split('-')[0].split(':')[0];
    return base ? `${base}-USDT-SWAP` : '';
}

async function fetchOKXTicker(symbol) {
    const cached = okxTickerCache[symbol];
    if (cached && Date.now() - cached.ts < 5000) return cached.data;

    const instId = toOKXSwapInstId(symbol);
    if (!instId) return null;

    const res = await fetch(`https://www.okx.com/api/v5/market/ticker?instId=${encodeURIComponent(instId)}`);
    const json = await res.json();
    const raw = json?.data?.[0];
    if (!raw) return null;

    const price = Number(raw.last || 0);
    const displayOpen = Number(raw.sodUtc8 || raw.open24h || 0);
    const change = displayOpen ? ((price - displayOpen) / displayOpen * 100) : 0;
    const data = {
        price,
        change_24h: change,
        volume_24h: Number(raw.vol24h || 0),
        bid: Number(raw.bidPx || 0),
        ask: Number(raw.askPx || 0),
    };
    okxTickerCache[symbol] = { ts: Date.now(), data };
    return data;
}

async function enrichTickersFromOKX(tickers) {
    const enriched = { ...tickers };
    await Promise.all(Object.keys(enriched).map(async (symbol) => {
        try {
            const liveTicker = await fetchOKXTicker(symbol);
            if (liveTicker) enriched[symbol] = { ...enriched[symbol], ...liveTicker };
        } catch (e) {
            console.debug('OKX ticker fallback failed', symbol, e);
        }
    }));
    return enriched;
}

function updateMarketData(market, accounts = []) {
    const marketPositions = marketOpenPositions(market);
    const positionTickers = marketPositions.length
        ? buildTickersFromPositions(marketPositions)
        : buildPositionTickers(accounts);
    const marketTickers = market.tickers || {};
    const positionSymbols = new Set((market.position_symbols || []).filter(Boolean));
    state.positionTickerSymbols = Object.keys(positionTickers).length
        ? Object.keys(positionTickers)
        : Array.from(positionSymbols);
    const marketPositionTickers = Object.fromEntries(
        Object.entries(marketTickers).filter(([symbol]) => state.positionTickerSymbols.includes(symbol))
    );
    const tickers = Object.keys(positionTickers).length
        ? { ...marketPositionTickers, ...positionTickers }
        : marketPositionTickers;
    updateTickers(tickers, { replace: true });
    refreshAutoPriceChart();
}

function decisionSizeTitle(d, sizePct) {
    const orderQty = valueNumber(d.order_quantity);
    const orderPrice = valueNumber(d.order_price ?? d.execution_price);
    const leverage = Math.max(valueNumber(d.suggested_leverage) || 1, 1);
    const notional = valueNumber(d.order_notional_usdt) ?? (
        orderQty !== null && orderPrice !== null ? orderQty * orderPrice : null
    );
    const margin = notional !== null ? notional / leverage : null;
    return [
        `保证金占比 ${sizePct.toFixed(1)}%：下单保证金 / 当前执行账户可用余额。`,
        '不是成交币数量比例，也不是账户权益占比。',
        orderQty !== null ? `订单数量 ${orderQty}` : '',
        notional !== null ? `名义价值约 ${fmtMoney(notional)} USDT` : '',
        margin !== null ? `估算保证金约 ${fmtMoney(margin)} USDT` : '',
        leverage ? `杠杆 ${leverage}x` : '',
    ].filter(Boolean).join(' ');
}

function decisionSizeCell(d) {
    const sizePct = Number(d.position_size_pct || 0) * 100;
    const orderQty = valueNumber(d.order_quantity);
    const title = decisionSizeTitle(d, sizePct);
    const qtyLine = orderQty !== null ? `<small>数量 ${escHtml(String(orderQty))}</small>` : '';
    return `<span title="${escHtml(title)}">${sizePct.toFixed(1)}%</span>${qtyLine}`;
}

function decisionTimeMs(d) {
    const raw = d.created_at || d.executed_at || d.timestamp || '';
    const ms = raw ? new Date(raw).getTime() : 0;
    return Number.isFinite(ms) ? ms : 0;
}

function decisionKey(d) {
    return d.id || [
        d.model || d.model_name || '',
        d.symbol || '',
        d.action || '',
        d.created_at || d.executed_at || d.timestamp || '',
    ].join('|');
}

function renderRecentDecisions(decisions) {
    const container = document.getElementById('decision-list');
    const countEl = document.getElementById('decision-count');
    if (!container) return;

    state.decisions = (decisions || [])
        .slice()
        .sort((a, b) => decisionTimeMs(b) - decisionTimeMs(a))
        .slice(0, 5);

    if (countEl) countEl.textContent = state.decisions.length;

    if (!state.decisions.length) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">暂无 AI 决策记录</div>';
        return;
    }

    container.innerHTML = `
        <div class="mini-table-wrap">
            <table class="mini-table">
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>币种</th>
                        <th>方向</th>
                        <th>信心度</th>
                        <th title="下单保证金占当前执行账户可用余额的比例，不是成交币数量比例。">保证金占比</th>
                        <th>是否执行</th>
                    </tr>
                </thead>
                <tbody>
                    ${state.decisions.map(d => {
                        const conf = Number(d.confidence || 0);
                        const executedHtml = d.was_executed
                            ? '<span style="color:var(--green);font-weight:600;">是</span>'
                            : '<span style="color:var(--text-dim);">否</span>';
                        return `
                            <tr>
                                <td>${toBeijingTime(d.created_at)}</td>
                                <td>${escHtml(d.symbol || '-')}</td>
                                <td><span class="badge badge-${d.action || 'hold'}">${analysisActionLabel(d.action, d)}</span></td>
                                <td style="color:${conf >= 0.65 ? 'var(--green)' : 'var(--text-muted)'};font-weight:600;">${(conf * 100).toFixed(0)}%</td>
                                <td class="decision-size-cell">${decisionSizeCell(d)}</td>
                                <td>${executedHtml}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function updateDecisions(decisions) {
    const incoming = (decisions || []).filter(Boolean);
    if (!incoming.length) return;

    if (recentDecisionsRefreshTimer) {
        clearTimeout(recentDecisionsRefreshTimer);
    }
    recentDecisionsRefreshTimer = setTimeout(() => {
        recentDecisionsRefreshTimer = null;
        fetchRecentDecisions();
    }, 300);
}

function updateExecutions(executions) {
    const incoming = (executions || []).filter(Boolean);
    if (!incoming.length) return;
    fetchRecentExecutions();
}

function renderRecentExecutions(executions, total) {
    const container = document.getElementById('execution-list');
    const countEl = document.getElementById('execution-count');
    if (!container) return;

    state.executions = (executions || []).slice(0, 5);
    if (countEl) countEl.textContent = total ?? state.executions.length;

    if (!state.executions.length) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">暂无执行记录</div>';
        return;
    }

    container.innerHTML = `
        <div class="mini-table-wrap">
            <table class="mini-table">
                <thead>
                    <tr>
                        <th>执行时间</th>
                        <th>币种</th>
                        <th>方向</th>
                        <th>杠杆</th>
                        <th>数量</th>
                        <th>价格</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
                    ${state.executions.map(t => {
                        const success = t.success === true || t.status === 'filled';
                        const statusInfo = executionStatusPresentation(t, success);
                        return `
                            <tr>
                                <td>${toBeijingTime(t.filled_at || t.created_at)}</td>
                                <td>${escHtml(t.symbol || '-')}</td>
                                <td>${executionActionCell(t)}</td>
                                <td>${Number(t.leverage || 1).toFixed(1)}x</td>
                                <td>${fmtNum(t.quantity)}</td>
                                <td>${fmtPrice(t.price)}</td>
                                <td style="color:${statusInfo.color};font-weight:600;">${escHtml(statusInfo.label)}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function changeTradePage(page) {
    state.tradesPage = page;
    fetchTrades();
}

function updateStats(stats) {
    if (stats.running !== undefined) {
        document.getElementById('stat-uptime').textContent = formatUptime(stats.uptime_seconds || 0);
        updateAutoStatus(stats);
    }
}

function formatRiskAlertText(event) {
    const details = event.message || event.warning || event.reason || event.details;
    let message = '';

    if (typeof details === 'string') {
        message = details;
    } else if (details && typeof details === 'object') {
        if (typeof details.message === 'string') {
            message = details.message;
        } else if (typeof details.warning === 'string') {
            message = details.warning;
        } else if (typeof details.reason === 'string') {
            message = details.reason;
        } else {
            message = Object.entries(details)
                .filter(([, value]) => value !== null && value !== undefined && value !== '')
                .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)
                .join(' | ');
        }
    }

    if (!message) {
        message = JSON.stringify(event);
    }

    const type = event.type || event.event_type || '\u98ce\u9669';
    const symbol = event.symbol ? ` ${event.symbol}` : '';
    return `[${type}]${symbol} ${message}`;
}

function riskAlertMessage(event) {
    const details = event?.message || event?.warning || event?.reason || event?.details;
    if (typeof details === 'string') return details.trim();
    if (details && typeof details === 'object') {
        for (const key of ['message', 'warning', 'reason', 'details']) {
            if (typeof details[key] === 'string' && details[key].trim()) {
                return details[key].trim();
            }
        }
        return Object.entries(details)
            .filter(([, value]) => value !== null && value !== undefined && value !== '')
            .map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)
            .join(' | ');
    }
    return formatRiskAlertText(event);
}

function riskAlertMatch(text, pattern) {
    const match = String(text || '').match(pattern);
    return match && match[1] ? match[1].trim() : '';
}

function riskAlertSeverity(event) {
    const value = String(event?.severity || event?.level || '').toLowerCase();
    if (['critical', 'high', 'error', 'danger'].includes(value)) return 'critical';
    if (['warn', 'warning', 'medium'].includes(value)) return 'warning';
    return 'info';
}

function riskAlertTypeLabel(type) {
    const labels = {
        position_review_warning: '持仓复盘预警',
        risk_alert: '风险告警',
        circuit_breaker: '风险熔断',
        stop_loss: '止损风险',
        margin_warning: '保证金提醒',
        black_swan: '极端风险',
    };
    return labels[type] || type || '风险事件';
}

function riskAlertActionLabel(action) {
    const text = String(action || '').trim();
    const normalized = text.toLowerCase();
    if (!text) return '-';
    if (normalized === 'hold' || text.includes('观望') || text.includes('持有')) return '观望';
    if (normalized === 'long' || text.includes('做多')) return '做多';
    if (normalized === 'short' || text.includes('做空')) return '做空';
    if (normalized === 'close_long' || text.includes('平多')) return '平多';
    if (normalized === 'close_short' || text.includes('平空')) return '平空';
    return text;
}

function riskAlertSideLabel(side) {
    const value = String(side || '').toLowerCase();
    if (value === 'long') return '多头';
    if (value === 'short') return '空头';
    return side || '-';
}

function riskAlertMoneyLabel(value) {
    const n = valueNumber(value);
    if (n === null) return '--';
    const abs = Math.abs(n);
    const digits = abs > 0 && abs < 0.01 ? 4 : 2;
    return `${n >= 0 ? '+' : ''}${n.toFixed(digits)} U`;
}

function parseRiskAlert(event) {
    const message = riskAlertMessage(event);
    const normalized = message.replace(/^Position review risk alert:\s*/i, '').trim();
    const symbol = event?.symbol || riskAlertMatch(normalized, /\b([A-Z0-9-]+\/[A-Z0-9-]+)\b/);
    const reason = riskAlertMatch(message, /Reason=(.*?)(?:\.\s*Final review action=|$)/is);
    return {
        id: event?.id,
        type: event?.event_type || event?.type || 'risk_alert',
        severity: riskAlertSeverity(event),
        symbol,
        side: riskAlertMatch(normalized, /\bcurrent\s+(long|short)\b/i),
        entry: riskAlertMatch(normalized, /\bentry=([-+]?\d+(?:\.\d+)?)/i),
        quantity: riskAlertMatch(normalized, /\bqty=([-+]?\d+(?:\.\d+)?)/i),
        pnl: riskAlertMatch(normalized, /\bpnl=([-+]?\d+(?:\.\d+)?)/i),
        expertAction: riskAlertMatch(message, /Risk expert action=([^,.]+)/i),
        confidence: riskAlertMatch(message, /\bconfidence=([0-9.]+%?)/i),
        finalAction: riskAlertMatch(message, /Final review action=([^.]*)/i),
        systemAction: riskAlertMatch(message, /system_action=([^.]*)/i),
        result: riskAlertMatch(message, /result=(.*)$/is),
        reason,
        message: normalized,
        createdAt: event?.created_at,
    };
}

function riskAlertMetric(label, value, tone = 'muted') {
    if (value === undefined || value === null || value === '') return '';
    return `
        <div class="risk-alert-metric risk-alert-metric-${tone}">
            <span>${escHtml(label)}</span>
            <strong>${escHtml(value)}</strong>
        </div>
    `;
}

function riskAlertReasonHtml(reason) {
    const parts = String(reason || '')
        .split(/[；;]/)
        .map(part => part.replace(/\s+/g, ' ').trim())
        .filter(Boolean)
        .slice(0, 4);
    if (!parts.length) return '';
    return `
        <div class="risk-alert-reason">
            ${parts.map(part => {
                const [label, ...rest] = part.split(/[:：]/);
                const body = rest.join('：').trim();
                return body
                    ? `<span><em>${escHtml(label)}</em>${escHtml(body)}</span>`
                    : `<span>${escHtml(part)}</span>`;
            }).join('')}
        </div>
    `;
}

function renderRiskAlertItem(event) {
    const item = parseRiskAlert(event);
    const pnl = valueNumber(item.pnl);
    const pnlTone = pnl === null ? 'muted' : pnl < 0 ? 'bad' : pnl > 0 ? 'good' : 'muted';
    const severityLabel = item.severity === 'critical' ? '严重' : item.severity === 'warning' ? '提醒' : '记录';
    const time = toBeijingTime(item.createdAt);
    const action = riskAlertActionLabel(item.expertAction || item.finalAction || item.systemAction);
    const finalAction = riskAlertActionLabel(item.finalAction);
    const systemAction = riskAlertActionLabel(item.systemAction);
    const metrics = [
        riskAlertMetric('方向', riskAlertSideLabel(item.side), 'muted'),
        riskAlertMetric('入场', item.entry ? fmtPrice(item.entry) : '', 'muted'),
        riskAlertMetric('数量', item.quantity ? fmtNum(item.quantity) : '', 'muted'),
        riskAlertMetric('PnL', item.pnl ? riskAlertMoneyLabel(item.pnl) : '', pnlTone),
        riskAlertMetric('置信度', item.confidence, 'muted'),
    ].join('');
    const result = item.result || item.message;

    return `
        <div class="risk-alert-item risk-alert-${item.severity}" data-id="${escHtml(item.id ?? '')}" role="listitem">
            <div class="risk-alert-head">
                <div class="risk-alert-title">
                    <span class="risk-alert-dot"></span>
                    <strong>${escHtml(item.symbol || '全局风险')}</strong>
                    <span>${escHtml(riskAlertTypeLabel(item.type))}</span>
                    <em>${escHtml(severityLabel)}</em>
                </div>
                <time>${escHtml(time || '-')}</time>
            </div>
            <div class="risk-alert-metrics">${metrics}</div>
            <div class="risk-alert-flow">
                <div><span>风控专家</span><strong>${escHtml(action)}${item.confidence ? ` / ${escHtml(item.confidence)}` : ''}</strong></div>
                <div><span>复盘结论</span><strong>${escHtml(finalAction)}</strong></div>
                <div><span>系统动作</span><strong>${escHtml(systemAction)}</strong></div>
            </div>
            ${riskAlertReasonHtml(item.reason)}
            <div class="risk-alert-result">${escHtml(result || '暂无详情')}</div>
        </div>
    `;
}

function updateRiskAlertCounters(count) {
    const countEl = document.getElementById('alert-count');
    const badgeEl = document.getElementById('alert-badge');
    if (countEl) countEl.textContent = count;
    if (badgeEl) badgeEl.textContent = count;
}

function renderRiskAlertSummary(events) {
    const el = document.getElementById('risk-alert-summary');
    if (!el) return;
    const parsed = events.map(parseRiskAlert);
    const total = parsed.length;
    const critical = parsed.filter(item => item.severity === 'critical').length;
    const warning = parsed.filter(item => item.severity === 'warning').length;
    const symbols = new Set(parsed.map(item => item.symbol).filter(Boolean));
    const latest = parsed[0]?.createdAt ? toBeijingTime(parsed[0].createdAt) : '-';
    el.innerHTML = `
        <div class="risk-alert-kpi">
            <span>总告警</span>
            <strong>${total}</strong>
        </div>
        <div class="risk-alert-kpi risk-alert-kpi-critical">
            <span>严重</span>
            <strong>${critical}</strong>
        </div>
        <div class="risk-alert-kpi risk-alert-kpi-warning">
            <span>提醒</span>
            <strong>${warning}</strong>
        </div>
        <div class="risk-alert-kpi">
            <span>涉及币种</span>
            <strong>${symbols.size}</strong>
        </div>
        <div class="risk-alert-kpi risk-alert-kpi-wide">
            <span>最新时间</span>
            <strong>${escHtml(latest)}</strong>
        </div>
    `;
}

function renderRiskAlerts(events = state.riskEvents || []) {
    const container = document.getElementById('risk-alerts');
    if (!container) return;
    const paginationId = 'risk-alert-pagination';
    renderRiskAlertSummary(events);
    updateRiskAlertCounters(events.length);
    if (!events.length) {
        container.innerHTML = '<div class="risk-alert-empty">暂无告警</div>';
        renderPagination(paginationId, 1, 1, 0, 'changeRiskAlertPage');
        return;
    }
    const total = events.length;
    const totalPages = Math.max(Math.ceil(total / RISK_ALERT_PAGE_SIZE), 1);
    const page = Math.min(Math.max(Number(state.riskEventsPage || 1), 1), totalPages);
    state.riskEventsPage = page;
    const start = (page - 1) * RISK_ALERT_PAGE_SIZE;
    const pageEvents = events.slice(start, start + RISK_ALERT_PAGE_SIZE);
    container.innerHTML = pageEvents.map(renderRiskAlertItem).join('');
    renderPagination(paginationId, page, totalPages, total, 'changeRiskAlertPage');
}

function changeRiskAlertPage(page) {
    state.riskEventsPage = Math.max(1, Number(page) || 1);
    renderRiskAlerts(state.riskEvents || []);
}

function addRiskAlert(data) {
    const event = {
        ...data,
        event_type: data.event_type || data.type || 'risk_alert',
        created_at: data.created_at || new Date().toISOString(),
    };
    const seen = new Set();
    state.riskEvents = [event, ...(state.riskEvents || [])]
        .filter(item => {
            const key = item.id !== undefined && item.id !== null
                ? `id:${item.id}`
                : `${item.event_type || item.type || ''}:${item.symbol || ''}:${riskAlertMessage(item)}:${item.created_at || ''}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        })
        .slice(0, 50);
    state.riskEventsPage = 1;
    renderRiskAlerts(state.riskEvents);
    return;

    const container = document.getElementById('risk-alerts');
    const countEl = document.getElementById('alert-count');
    const badgeEl = document.getElementById('alert-badge');
    if (!container) return;

    // Clear placeholder on first real alert
    const placeholder = container.querySelector('div[style]');
    if (placeholder && placeholder.style.color === 'var(--text-muted)') {
        placeholder.remove();
    }

    const alertDiv = document.createElement('div');
    const cls = data.severity === 'critical' ? 'alert-critical' : 'alert-warning';
    alertDiv.className = 'alert-item ' + cls;
    alertDiv.style.overflowWrap = 'anywhere';
    alertDiv.style.lineHeight = '1.5';
    alertDiv.textContent = `[${data.type || data.event_type || '风险'}] ${data.message || data.details || JSON.stringify(data)}`;
    alertDiv.textContent = formatRiskAlertText(data);
    container.prepend(alertDiv);

    const count = container.querySelectorAll('.alert-item').length;
    if (countEl) countEl.textContent = count;
    if (badgeEl) badgeEl.textContent = count;

    while (container.children.length > 50) {
        container.removeChild(container.lastChild);
    }
}

async function fetchRiskEvents() {
    const data = await fetchJSON('/api/risk/events?limit=50');
    if (!data || !data.events) return;
    state.riskEvents = data.events || [];
    const totalPages = Math.max(Math.ceil(state.riskEvents.length / RISK_ALERT_PAGE_SIZE), 1);
    state.riskEventsPage = Math.min(Math.max(Number(state.riskEventsPage || 1), 1), totalPages);
    renderRiskAlerts(state.riskEvents);
    return;

    const container = document.getElementById('risk-alerts');
    if (!container) return;

    // Clear existing (but keep WS-pushed alerts that came after)
    const existingAlerts = container.querySelectorAll('.alert-item');
    if (!existingAlerts.length && data.events.length) {
        // Remove placeholder
        const placeholder = container.querySelector('div[style]');
        if (placeholder) placeholder.remove();
    }

    data.events.forEach(e => {
        // Skip if already rendered
        if (container.querySelector(`[data-id="${e.id}"]`)) return;
        const alertDiv = document.createElement('div');
        const cls = e.severity === 'critical' ? 'alert-critical' : 'alert-warning';
        alertDiv.className = 'alert-item ' + cls;
        alertDiv.style.overflowWrap = 'anywhere';
        alertDiv.style.lineHeight = '1.5';
        alertDiv.setAttribute('data-id', e.id);
        const time = toBeijingTime(e.created_at);
        alertDiv.textContent = `[${e.event_type || '风险'}] ${e.details || JSON.stringify(e)} ${time ? '— ' + time : ''}`;
        alertDiv.textContent = `${formatRiskAlertText(e)} ${time ? '- ' + time : ''}`;
        container.appendChild(alertDiv);
    });

    const count = container.querySelectorAll('.alert-item').length;
    const countEl = document.getElementById('alert-count');
    const badgeEl = document.getElementById('alert-badge');
    if (countEl) countEl.textContent = count;
    if (badgeEl) badgeEl.textContent = count;
}

// --- Trade Mode Tabs ---
function initTradeTabs() {
    // Model mode tabs (paper/live toggle in settings)
    document.querySelectorAll('#model-mode-tabs .trade-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const mode = tab.dataset.mm;
            currentModelMode = mode || 'paper';
            document.querySelectorAll('#model-mode-tabs .trade-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            fetchAIModels();
        });
    });
}

function initSettingsTabs() {
    const buttons = Array.from(document.querySelectorAll('.settings-menu-item[data-settings-tab]'));
    const sections = Array.from(document.querySelectorAll('.settings-section[data-settings-section]'));
    if (!buttons.length || !sections.length) return;

    buttons.forEach(btn => {
        btn.addEventListener('click', () => activateSettingsTab(btn.dataset.settingsTab || 'okx'));
    });
}

function initDashboardUserActions() {
    document.addEventListener('click', async event => {
        const button = event.target?.closest?.('[data-dashboard-user-action]');
        if (!button) return;
        event.preventDefault();
        const action = button.dataset.dashboardUserAction || '';
        const username = button.dataset.username || '';
        if (action === 'create') {
            openDashboardUserModal('create');
            return;
        }
        if (action === 'edit') {
            openDashboardUserModal('edit', username);
            return;
        }
        if (action === 'activate') {
            await setDashboardUserActive(username, true, button);
            return;
        }
        if (action === 'deactivate') {
            await setDashboardUserActive(username, false, button);
            return;
        }
        if (action === 'delete') {
            await deleteDashboardUser(username, button);
            return;
        }
        if (action === 'close-modal') {
            closeDashboardUserModal();
            return;
        }
        if (action === 'save-modal') {
            await saveDashboardUserModal();
        }
    });
}

function initModalActionButtons() {
    document.addEventListener('click', async event => {
        const button = event.target?.closest?.('[data-modal-action]');
        if (!button) return;
        event.preventDefault();
        const action = button.dataset.modalAction || '';
        if (action === 'close-model') {
            closeModelModal();
            return;
        }
        if (action === 'save-model') {
            await saveModelConfig();
        }
    });
}

function initServerMonitorTabs() {
    document.addEventListener('click', event => {
        const button = event.target?.closest?.('[data-server-monitor-tab]');
        if (!button) return;
        event.preventDefault();
        state.serverMonitorTab = button.dataset.serverMonitorTab || 'self-check';
        renderServerMonitor();
    });
}

// --- Sidebar Navigation ---
function activateSettingsTab(name = 'okx') {
    const selected = name || 'okx';
    document.querySelectorAll('.settings-menu-item[data-settings-tab]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.settingsTab === selected);
    });
    document.querySelectorAll('.settings-section[data-settings-section]').forEach(section => {
        section.classList.toggle('active', section.dataset.settingsSection === selected);
    });
}

function loadPageData(page) {
    if (page === 'trades') {
        const label = document.getElementById('trade-mode-label');
        if (label) label.textContent = state.mode === 'paper' ? '模拟盘' : '实盘';
        fetchTrades();
    }
    if (page === 'positions') {
        const label = document.getElementById('positions-mode-label');
        if (label) label.textContent = state.mode === 'paper' ? '模拟盘' : '实盘';
        fetchPositions();
    }
    if (page === 'position-history') {
        const label = document.getElementById('position-history-mode-label');
        if (label) label.textContent = state.mode === 'paper' ? '模拟盘' : '实盘';
        fetchPositionHistory();
    }
    if (page === 'daily-pnl') fetchDailyPnlRecords();
    if (page === 'decisions') { populateDecisionModelFilter(); fetchAllDecisions(); }
    if (page === 'opening-funnel') fetchOpeningFunnel();
    if (page === 'profit-attribution') fetchProfitAttribution();
    if (page === 'strategy-learning') fetchStrategyLearning();
    if (page === 'analysis') fetchAnalysisRecords();
    if (page === 'alerts') fetchRiskEvents();
    if (page === 'expert-memory') fetchExpertMemories();
    if (page === 'shadow-backtest') fetchShadowBacktests();
    if (page === 'ml-signal') fetchMLSignalDashboard();
    if (page === 'server-monitor') {
        refreshServerMonitorPage();
    }
    if (page === 'settings') {
        fetchDashboardAccountSettings();
        fetchModelServerSettings();
        fetchOKXSettings();
        fetchExecutionAccountSettings();
        fetchAIModels();
        fetchTradingParams();
    }
}

function activatePage(page) {
    const selected = page || 'dashboard';
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === selected);
    });
    document.querySelectorAll('.page-section').forEach(section => {
        section.classList.remove('active');
    });
    const target = document.getElementById('page-' + selected);
    if (target) target.classList.add('active');
}

function openPage(page) {
    activatePage(page);
    loadPageData(page);
}

function initSidebarNav() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => openPage(item.dataset.page));
    });
}

// --- Mode Controls ---
function initModeButtons() {
    document.querySelectorAll('.mode-btn[data-mode]').forEach(btn => {
        btn.addEventListener('click', async () => {
            const mode = btn.dataset.mode;
            if (mode === 'live' && state.okxConfig?.liveConfigured === false) {
                const message = '实盘 OKX API 未配置完整，不能切换执行账户。请先配置 API Key、API Secret 和 Passphrase。';
                alert(message);
                openPage('settings');
                activateSettingsTab('okx');
                const status = document.getElementById('execution-account-save-status');
                if (status) {
                    status.textContent = message;
                    status.style.color = 'var(--red)';
                }
                fetchOKXSettings();
                fetchExecutionAccountSettings();
                return;
            }
            const res = await fetchWithAuth('/api/control/mode', dashboardWriteOptions({
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode }),
            }));
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                const detail = err.detail && typeof err.detail === 'object' ? err.detail : {};
                const message = apiErrorText(err, res.statusText || '切换失败');
                alert('切换失败: ' + message);
                if (res.status === 409 && detail.settings_tab) {
                    openPage('settings');
                    activateSettingsTab(detail.settings_tab);
                    const status = document.getElementById('execution-account-save-status');
                    if (status) {
                        status.textContent = message;
                        status.style.color = 'var(--red)';
                    }
                    fetchOKXSettings();
                    fetchExecutionAccountSettings();
                }
                return;
            }
            state.mode = mode;
            state.positionsPage = 1;
            // Clear old WS data (belongs to previous mode)
            state.decisions = [];
            state.executions = [];
            document.getElementById('decision-list').innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">等待决策数据...</div>';
            document.getElementById('execution-list').innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">暂无成交记录</div>';
            // Refresh all data for the new mode
            await fetchDashboardSummary();
            fetchPnlHistory();
            fetchRecentDecisions();
            fetchRecentExecutions();
            fetchTrades();
            fetchAllDecisions();
            fetchAnalysisRecords();
            if (isPageActive('opening-funnel')) fetchOpeningFunnel();
            if (isPageActive('profit-attribution')) fetchProfitAttribution();
            if (isPageActive('strategy-learning')) fetchStrategyLearning();
            if (isPageActive('expert-memory')) fetchExpertMemories();
            fetchPositions();
            fetchPositionHistory();
            if (isPageActive('daily-pnl')) fetchDailyPnlRecords();
        });
    });
}

async function togglePause() {
    const endpoint = state.paused ? '/api/control/resume' : '/api/control/pause';
    await fetchWithAuth(endpoint, dashboardWriteOptions({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    }));
    await fetchDashboardSummary();
    const btn = document.getElementById('pause-btn');
    if (btn) btn.textContent = state.paused ? '恢复' : '暂停';
}

async function selectLiveModel(modelName) {
    await fetchWithAuth('/api/control/select-model', dashboardWriteOptions({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
    }));
    state.liveModel = modelName;
    await fetchDashboardSummary();
}

// ========== All Decisions Page ==========

let decisionFilterTimeout = null;

function getDecisionFilters() {
    const startDate = document.getElementById('filter-start-date')?.value || '';
    const endDate = document.getElementById('filter-end-date')?.value || '';
    const model = document.getElementById('filter-model')?.value || '';
    const action = document.getElementById('filter-action')?.value || '';
    const executed = document.getElementById('filter-executed')?.value || '';

    const params = new URLSearchParams();
    params.set('page', String(state.decisionsPage || 1));
    params.set('page_size', String(PAGE_SIZE));
    if (startDate) params.set('start_date', new Date(startDate).toISOString());
    if (endDate) params.set('end_date', new Date(endDate).toISOString());
    if (model) params.set('model_name', model);
    if (action) params.set('action', action);
    if (executed) params.set('was_executed', executed);
    // Global mode filter: paper/live
    params.set('is_paper', state.mode === 'paper' ? 'true' : 'false');
    return params.toString();
}

function onDecisionFilterChange() {
    // Debounce the fetch
    if (decisionFilterTimeout) clearTimeout(decisionFilterTimeout);
    decisionFilterTimeout = setTimeout(fetchAllDecisions, 300);
}

function resetDecisionFilters() {
    const startEl = document.getElementById('filter-start-date');
    const endEl = document.getElementById('filter-end-date');
    const modelEl = document.getElementById('filter-model');
    const actionEl = document.getElementById('filter-action');
    const execEl = document.getElementById('filter-executed');
    if (startEl) startEl.value = '';
    if (endEl) endEl.value = '';
    if (modelEl) modelEl.value = '';
    if (actionEl) actionEl.value = '';
    if (execEl) execEl.value = '';
    state.decisionsPage = 1;
    fetchAllDecisions();
}

async function fetchAllDecisions() {
    const qs = getDecisionFilters();
    const data = await fetchJSON('/api/decisions?' + qs);
    if (!data || !data.decisions) return;
    renderAllDecisions(data.decisions, data);
    updateDecisionBadge(data.total ?? data.count);
}

async function populateDecisionModelFilter() {
    const data = await fetchJSON('/api/settings/ai-models');
    if (!data) return;

    const select = document.getElementById('filter-model');
    if (!select) return;

    const allModels = (data.models || []).concat(data.legacy || []);
    if (data.execution_model) {
        allModels.push({ name: data.execution_model });
    }
    const currentVal = select.value;
    select.innerHTML = '<option value="">全部模型</option>' +
        allModels.map(m => `<option value="${escHtml(m.name)}">${escHtml(m.name)}</option>`).join('');
    if (currentVal) select.value = currentVal;
}

async function clearAllDecisions() {
    if (!confirm('确定要删除所有 AI 决策记录吗？此操作不可撤销。')) return;

    const res = await fetchWithAuth('/api/decisions', dashboardWriteOptions({
        method: 'DELETE',
        headers: { 'X-Dashboard-Confirm': 'delete-records' },
    }));
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('清除失败: ' + (err.detail || '未知错误'));
        return;
    }
    const data = await res.json();
    alert('已删除 ' + data.deleted + ' 条决策记录');
    fetchAllDecisions();
    fetchDashboardSummary();
}

function renderAllDecisions(decisions, meta = {}) {
    state.allDecisions = decisions || [];
    state.decisionsPage = Number(meta.page || state.decisionsPage || 1);
    state.decisionsTotal = Number(meta.total ?? state.allDecisions.length);

    const countEl = document.getElementById('all-decisions-count');
    if (countEl) countEl.textContent = state.decisionsTotal + ' 条';

    renderDecisionsPage(Number(meta.total_pages || 1));
}

function opportunityScoreValue(value, digits = 4) {
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(digits) : '-';
}

function opportunityScorePrimaryReturn(score) {
    if (!score || typeof score !== 'object') return { label: '预期净收益', value: null };
    const net = Number(score.expected_net_return_pct);
    if (Number.isFinite(net)) return { label: '预期净收益', value: net };
    return { label: '预期收益', value: Number(score.expected_return_pct) };
}

function opportunityScoreReturnDetail(score) {
    if (!score || typeof score !== 'object') return '';
    const items = opportunityScoreFormulaItems(score);
    if (!items.length) return '';
    return items.map(item => `${item.label} ${item.text}`).join(' / ');
}

function opportunityScoreFormulaItems(score) {
    if (!score || typeof score !== 'object') return [];
    const breakdown = score.expected_net_breakdown || {};
    const components = Array.isArray(breakdown.components) ? breakdown.components : [];
    if (components.length) {
        return components.map(component => {
            const contribution = Number(component.contribution_pct);
            const rawReturn = Number(component.raw_return_pct);
            const weight = Number(component.weight);
            const available = component.available !== false;
            const pieces = [];
            if (Number.isFinite(contribution)) pieces.push(signedPctValueLabel(contribution));
            if (Number.isFinite(rawReturn) && Number.isFinite(weight)) {
                pieces.push(`原始 ${signedPctValueLabel(rawReturn)} × 权重 ${opportunityScoreValue(weight, 2)}`);
            }
            if (!available) pieces.push('当前未参与');
            return {
                label: component.label || component.key || '收益来源',
                value: Number.isFinite(contribution) ? contribution : 0,
                text: pieces.join(' · ') || '-',
                tone: !available ? 'muted' : (contribution > 0 ? 'good' : (contribution < 0 ? 'bad' : 'muted')),
                note: component.note || '',
            };
        });
    }
    const weights = score.expected_net_weights || {};
    const weightOf = (key) => Number(weights[key] ?? 0);
    const signed = (value) => `${Number(value) >= 0 ? '+' : ''}${opportunityScoreValue(value, 4)}%`;
    const items = [];
    const ai = Number(score.ai_expected_return_contribution_pct);
    if (Number.isFinite(ai)) {
        items.push({ label: 'AI贡献', value: ai, text: signed(ai), tone: ai >= 0 ? 'good' : 'bad' });
    }
    const mlRaw = Number(score.expected_return_pct);
    const mlWeight = weightOf('local_ml_expected_return');
    const ml = Number.isFinite(mlRaw) ? mlRaw * mlWeight : NaN;
    if (Number.isFinite(ml)) {
        items.push({ label: '本地ML', value: ml, text: `${signed(ml)}（${opportunityScoreValue(mlRaw, 4)}% × ${opportunityScoreValue(mlWeight, 2)}）`, tone: ml >= 0 ? 'good' : 'bad' });
    }
    const serverRaw = Number(score.server_profit_expected_return_pct);
    const serverWeight = weightOf('server_profit_expected_return');
    const server = Number.isFinite(serverRaw) ? serverRaw * serverWeight : NaN;
    if (Number.isFinite(server)) {
        items.push({ label: '服务器盈利', value: server, text: `${signed(server)}（${opportunityScoreValue(serverRaw, 4)}% × ${opportunityScoreValue(serverWeight, 2)}）`, tone: server >= 0 ? 'good' : 'bad' });
    }
    const timeseriesRaw = Number(score.timeseries_expected_return_pct);
    const timeseriesWeight = weightOf('timeseries_expected_return');
    const timeseries = Number.isFinite(timeseriesRaw) ? timeseriesRaw * timeseriesWeight : NaN;
    if (Number.isFinite(timeseries)) {
        items.push({ label: '时序', value: timeseries, text: `${signed(timeseries)}（${opportunityScoreValue(timeseriesRaw, 4)}% × ${opportunityScoreValue(timeseriesWeight, 2)}）`, tone: timeseries >= 0 ? 'good' : 'bad' });
    }
    const fee = Number(score.fee_pct || 0);
    const slippage = Number(score.slippage_pct || 0);
    const cost = fee + slippage;
    if (Number.isFinite(cost) && cost > 0) {
        items.push({ label: '成本', value: -cost, text: `-${opportunityScoreValue(cost, 4)}%`, tone: 'bad' });
    }
    return items;
}

function opportunityScoreFormulaHtml(score) {
    const items = opportunityScoreFormulaItems(score);
    if (!items.length) return '';
    const breakdown = score.expected_net_breakdown || {};
    const net = Number(score.expected_net_return_pct);
    const modelNet = Number(score.model_expected_net_return_pct);
    const rows = items.map(item => `
        <div class="decision-score-formula-item ${escHtml(item.tone || '')}">
            <span>${escHtml(item.label)}</span>
            <strong>${escHtml(item.text)}</strong>
            ${item.note ? `<em>${escHtml(item.note)}</em>` : ''}
        </div>
    `).join('');
    const observedRows = Array.isArray(breakdown.observed_not_in_formula)
        ? breakdown.observed_not_in_formula.map(item => `
            <span>${escHtml(item.label || item.key || '观察项')}：${escHtml(item.available === false ? '未返回' : (item.aligned ? '同向观察' : '仅作证据观察'))}</span>
        `).join('')
        : '';
    const modelNetText = Number.isFinite(modelNet) ? `模型净值 ${signedPctValueLabel(modelNet)}` : '';
    const netText = Number.isFinite(net) ? `最终净收益 ${signedPctValueLabel(net)}` : '';
    return `
        <div class="decision-score-formula">
            <div class="decision-score-formula-head"><span>净收益拆解</span><em>${escHtml([modelNetText, netText].filter(Boolean).join(' · '))}</em></div>
            <div class="decision-score-formula-grid">${rows}</div>
            ${observedRows ? `<div class="decision-score-observed"><strong>只参与证据评分</strong>${observedRows}</div>` : ''}
        </div>
    `;
}

function opportunityScoreExecutionState(score, decision = null) {
    const wasExecuted = decision?.was_executed === true;
    const hasFinalSkip = decision && decision.was_executed === false && !!(decision.execution_reason || score?.selection_reason);
    const finalState = String(score?.execution_final_state || '').toLowerCase();
    if (wasExecuted) return { label: '已执行完成', tone: 'good' };
    if (hasFinalSkip || ['skipped', 'blocked'].includes(finalState) || score?.selected_for_execution === false) {
        return { label: '最终未执行', tone: 'warn' };
    }
    if (score?.selected_for_execution === true) return { label: '执行检查中', tone: 'neutral' };
    return { label: '等待排序', tone: 'neutral' };
}

function evidenceTierLabel(tier) {
    return {
        normal: '正常仓位',
        medium: '中等仓位',
        small: '小仓',
        exploration: '探索小仓',
        weak_conflict_probe: '弱冲突学习档',
        degraded_missing_probe: '模型缺失降级学习档',
        blocked: '不可交易档',
    }[String(tier || '').toLowerCase()] || tier || '-';
}

function evidenceSourceLabel(source) {
    return {
        ai: 'AI/专家',
        ml: '本地 ML',
        timeseries: '时序',
        sentiment: '情绪',
        server_profit: '服务器盈利',
        shadow_memory: '影子/记忆',
        symbol_side_history: '币种方向历史',
    }[String(source || '').toLowerCase()] || source || '-';
}

function evidenceStatusLabel(status) {
    return {
        aligned: '同向支持',
        opposite: '反向冲突',
        weak_opposite: '弱反向',
        missing: '缺失',
        neutral: '中性',
        ignored: '学习观察',
        limited_no_expert_support: '专家未同向',
        limited_single_expert_support: '仅 1 个专家同向',
        probe_derived_limited: '探针来源限分',
        probe_derived_no_expert_support: '探针无专家支持',
    }[String(status || '').toLowerCase()] || status || '-';
}

function evidenceListLabel(items) {
    return Array.isArray(items) && items.length
        ? items.map(evidenceSourceLabel).join('、')
        : '无';
}

function evidencePercentLabel(value, digits = 1) {
    const num = Number(value);
    return Number.isFinite(num) ? `${(num * 100).toFixed(digits)}%` : '-';
}

function dynamicEvidenceBlock(score, decision = null) {
    const evidence = score && typeof score === 'object' ? score.evidence_score : null;
    if (!evidence || typeof evidence !== 'object') return '';
    const rawScore = Number(evidence.score);
    const effective = Number(evidence.effective_score);
    const multiplier = Number(evidence.size_multiplier);
    const maxSize = Number(evidence.max_size_pct);
    const confidence = Number(decision?.confidence ?? score?.confidence);
    const components = Array.isArray(evidence.components) ? evidence.components : [];
    const componentRows = components.slice(0, 8).map(item => {
        const points = Number(item.points);
        const expected = Number(item.expected_return_pct ?? item.pnl);
        const sub = Number.isFinite(expected) ? ` · ${signedPctValueLabel(expected)}` : '';
        return `
            <div class="decision-evidence-component">
                <strong>${escHtml(item.label || evidenceSourceLabel(item.source))}</strong>
                <span>${escHtml(evidenceStatusLabel(item.status))}${sub}</span>
                <em>${Number.isFinite(points) ? points.toFixed(1) : '-'} 分</em>
            </div>
        `;
    }).join('');
    const waitReasons = Array.isArray(evidence.advisory_wait_reasons)
        ? evidence.advisory_wait_reasons.filter(Boolean)
        : [];
    const tier = String(evidence.tier || '').toLowerCase();
    const explanation = tier === 'weak_conflict_probe'
        ? '弱证据不是单看分析信心；它表示有效证据分只够学习/观察，而且存在反向或同向来源不足。'
        : tier === 'degraded_missing_probe'
            ? '弱证据不是单看分析信心；它表示关键模型缺失或质量不足，只能作为学习样本。'
            : '动态证据分综合 AI、ML、时序、情绪、服务器盈利、影子记忆和币种历史。';
    return `
        <div class="reason-block decision-evidence-block">
            <div class="reason-label">动态证据评分</div>
            <div class="decision-evidence-summary">
                <div><span>分析信心</span><strong>${evidencePercentLabel(confidence)}</strong><small>AI/专家最终置信度</small></div>
                <div><span>证据分</span><strong>${Number.isFinite(rawScore) ? rawScore.toFixed(1) : '-'}</strong><small>原始多来源分</small></div>
                <div><span>有效分</span><strong>${Number.isFinite(effective) ? effective.toFixed(1) : '-'}</strong><small>做空偏移/风控修正后</small></div>
                <div><span>档位</span><strong>${escHtml(evidenceTierLabel(evidence.tier))}</strong><small>仓位系数 ${Number.isFinite(multiplier) ? multiplier.toFixed(2) : '-'}</small></div>
            </div>
            <div class="decision-evidence-explain">${escHtml(explanation)}</div>
            <div class="decision-evidence-lists">
                <span>同向支持：${escHtml(evidenceListLabel(evidence.aligned_support_sources))}</span>
                <span>明确反向：${escHtml(evidenceListLabel(evidence.major_opposites))}</span>
                <span>弱反向：${escHtml(evidenceListLabel(evidence.weak_opposites))}</span>
                <span>缺失关键源：${escHtml(evidenceListLabel(evidence.missing_key_sources))}</span>
                ${Number.isFinite(maxSize) ? `<span>最大仓位参考：${(maxSize * 100).toFixed(2)}%</span>` : ''}
            </div>
            ${waitReasons.length ? `<div class="decision-evidence-wait">${waitReasons.map(item => `<span>${escHtml(analysisLocalizeText(item))}</span>`).join('')}</div>` : ''}
            ${componentRows ? `<div class="decision-evidence-components">${componentRows}</div>` : ''}
        </div>
    `;
}

function decisionMetricItem(label, value, hint = '', tone = '') {
    return `
        <div class="decision-score-metric ${escHtml(tone)}">
            <span>${escHtml(label)}</span>
            <strong>${value}</strong>
            ${hint ? `<em>${escHtml(hint)}</em>` : ''}
        </div>
    `;
}

function opportunityScoreBlock(score, decision = null) {
    if (!score || typeof score !== 'object') return '';
    const executionState = opportunityScoreExecutionState(score, decision);
    const reason = score.selection_reason || score.rule || '系统按预期净收益、方向优势、AI 信心、ML 盈亏质量、手续费、滑点、止损风险和当前敞口综合排序。';
    const primaryReturn = opportunityScorePrimaryReturn(score);
    const returnDetail = opportunityScoreReturnDetail(score);
    const formulaHtml = opportunityScoreFormulaHtml(score);
    const confidence = Number(decision?.confidence ?? score.confidence);
    const feeAndSlippage = Number(score.fee_pct || 0) + Number(score.slippage_pct || 0);
    const winRate = Number(score.win_rate || 0) * 100;
    return `
        <div class="reason-block decision-score-block">
            <div class="reason-label">盈利机会评分</div>
            <div class="decision-score-head">
                <div>
                    <strong>${opportunityScoreValue(score.score, 6)}</strong>
                    <span>${escHtml(actionLabel(score.side || '-'))} · ${primaryReturn.label} ${opportunityScoreValue(primaryReturn.value, 4)}%</span>
                </div>
                <span class="decision-score-state ${executionState.tone}">${escHtml(executionState.label)}</span>
            </div>
            <div class="decision-score-grid">
                ${decisionMetricItem('分析信心', evidencePercentLabel(confidence), 'AI/专家最终置信度，不等于动态证据分')}
                ${decisionMetricItem(primaryReturn.label, `${opportunityScoreValue(primaryReturn.value, 4)}%`, returnDetail || '综合收益估计', Number(primaryReturn.value) >= 0 ? 'good' : 'bad')}
                ${decisionMetricItem('相对反向优势', `${opportunityScoreValue(score.profit_edge_pct, 4)}%`)}
                ${decisionMetricItem('ML 胜率', `${opportunityScoreValue(winRate, 1)}%`)}
                ${decisionMetricItem('仓位 x 杠杆', opportunityScoreValue(score.size_x_leverage, 4))}
                ${decisionMetricItem('手续费+滑点', `${opportunityScoreValue(feeAndSlippage, 4)}%`)}
            </div>
            ${formulaHtml}
            <div class="decision-score-reason">
                <span>排序原因</span>
                <div>${escapeMultiline(reason)}</div>
            </div>
        </div>
        ${dynamicEvidenceBlock(score, decision)}
    `;
}

function showDecisionReason(decisionId) { 
    const decision = state.allDecisions.find(d => Number(d.id) === Number(decisionId)); 
    if (!decision) return; 
    setDecisionModalWide(false); 
 
    const title = `${decision.symbol || '-'} / ${analysisActionLabel(decision.action, decision)}`; 
    const primaryReason = decision.was_executed
        ? '该决策已执行，不属于未执行记录。'
        : (decision.execution_reason || (decision.action === 'hold' ? 'AI 选择观望，未提交订单。' : '暂无未执行原因。'));
    const executedInfo = decision.was_executed
        ? `<div class="reason-meta">执行时间：${toBeijingTime(decision.executed_at)}<br>执行价格：${fmtPrice(decision.execution_price)}</div>`
        : '';
    const aiReasoning = decision.reasoning
        ? `<div class="reason-block"><div class="reason-label">AI 分析</div><div>${escapeMultiline(decision.reasoning)}</div></div>`
        : '';
    const opportunityHtml = opportunityScoreBlock(decision.opportunity_score, decision);
    setDecisionModalWide(Boolean(decision.opportunity_score?.evidence_score));

    document.getElementById('decision-reason-title').textContent = title;
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="reason-block">
            <div class="reason-label">${decision.was_executed ? '执行状态' : '未执行原因'}</div>
            <div>${escapeMultiline(primaryReason)}</div>
            ${executedInfo}
        </div>
        ${opportunityHtml}
        ${aiReasoning}
    `;
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex';
}

function closeDecisionReasonModal() { 
    document.getElementById('decision-reason-modal-overlay').style.display = 'none'; 
    setDecisionModalWide(false); 
} 

function setDecisionModalWide(enabled) {
    const modal = document.querySelector('#decision-reason-modal-overlay .modal');
    if (!modal) return;
    modal.classList.toggle('modal-wide', Boolean(enabled));
}

function changeDecisionsPage(page) {
    state.decisionsPage = page;
    fetchAllDecisions();
}

// ========== Expert Analysis Records ==========

async function fetchAnalysisRecords() {
    const tbody = document.getElementById('analysis-tbody');
    if (tbody) {
        tbody.innerHTML = `<tr><td colspan="9" style="color:var(--text-muted);text-align:center;padding:24px;">正在加载${analysisViewLabel()}记录...</td></tr>`;
    }
    const params = new URLSearchParams();
    params.set('page', String(state.analysisPage || 1));
    params.set('page_size', String(PAGE_SIZE));
    params.set('analysis_type', state.analysisView === 'position' ? 'position' : 'market');
    params.set('include_detail', 'false');
    params.set('is_paper', state.mode === 'paper' ? 'true' : 'false');
    const data = await fetchJSON('/api/analysis-records?' + params.toString());
    if (!data || !data.records) return;
    renderAnalysisRecords(data.records, data);
    const badge = document.getElementById('analysis-badge');
    if (badge) badge.textContent = data.total ?? data.count ?? data.records.length;
}

function renderAnalysisRecords(records, meta = {}) {
    state.analysisRecords = records || [];
    state.analysisPage = Number(meta.page || state.analysisPage || 1);
    state.analysisTotal = Number(meta.total ?? state.analysisRecords.length);
    state.analysisTotalPages = Number(meta.total_pages || Math.ceil(state.analysisTotal / PAGE_SIZE) || 1);
    const countEl = document.getElementById('analysis-count');
    if (countEl) countEl.textContent = `${analysisViewLabel()} ${state.analysisTotal} 条`;
    renderAnalysisPage();
}

function analysisRecordType(record) {
    const value = String(record?.analysis_type || '').toLowerCase();
    return value === 'position' ? 'position' : 'market';
}

function analysisPositionLifecycleLabel(record) {
    if (!record || analysisRecordType(record) !== 'position') return '';
    if (record.position_lifecycle_label) return record.position_lifecycle_label;
    const status = String(record.position_lifecycle_status || '').toLowerCase();
    if (status === 'holding') return '持仓中';
    if (status === 'closed') return '已平仓';
    return '';
}

function analysisPositionLifecycleTone(record) {
    const status = String(record?.position_lifecycle_status || '').toLowerCase();
    if (status === 'holding') return 'good';
    if (status === 'closed') return 'muted';
    return 'muted';
}

function analysisIsCurrentPositionRecord(record) {
    return analysisRecordType(record) === 'position'
        && String(record.position_lifecycle_status || '').toLowerCase() === 'holding';
}

function analysisIsFastPositionScan(record) {
    return analysisRecordType(record) === 'position'
        && !!(record?.position_fast_scan && record.position_fast_scan.skipped_llm);
}

function analysisViewLabel(view = state.analysisView) {
    return view === 'position' ? '持仓分析' : '市场分析';
}

function getVisibleAnalysisRecords() {
    return state.analysisRecords || [];
}

function updateAnalysisViewControls(visibleCount = null) {
    const marketCount = state.analysisView === 'market' ? state.analysisTotal : '';
    const positionCount = state.analysisView === 'position' ? state.analysisTotal : '';
    document.querySelectorAll('[data-analysis-view]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.analysisView === state.analysisView);
    });
    const marketEl = document.getElementById('analysis-market-count');
    const positionEl = document.getElementById('analysis-position-count');
    const countEl = document.getElementById('analysis-count');
    if (marketEl) marketEl.textContent = marketCount;
    if (positionEl) positionEl.textContent = positionCount;
    if (countEl) {
        const count = visibleCount === null ? getVisibleAnalysisRecords().length : visibleCount;
        countEl.textContent = `${analysisViewLabel()} ${count} 条`;
    }
}

function setAnalysisView(view) {
    state.analysisView = view === 'position' ? 'position' : 'market';
    state.analysisPage = 1;
    fetchAnalysisRecords();
}

function analysisExpertDisplayName(name, experts = []) { 
    const direct = experts.find(e => e.expert_name === name); 
    if (direct) return direct.expert_label || direct.expert_name || name; 
    const alias = { 
        trend: 'trend_expert', 
        technical_trend: 'trend_expert', 
        trend_direction: 'trend_expert',
        momentum: 'momentum_expert', 
        short_term_momentum: 'momentum_expert', 
        profit_quality: 'momentum_expert',
        sentiment: 'sentiment_expert', 
        sentiment_news: 'sentiment_expert', 
        short_timeseries: 'sentiment_expert',
        position: 'position_expert', 
        position_manager: 'position_expert', 
        position_exit: 'position_expert',
        risk: 'risk_expert', 
        risk_guardian: 'risk_expert', 
        risk_anomaly: 'risk_expert',
        final_decision: 'decision_maker',
        decision: 'decision_maker',
    }; 
    const normalized = alias[name] || name; 
    const fallback = (FIXED_AI_EXPERT_FALLBACKS || []).find(e => e.name === normalized); 
    return fallback ? fallback.label : (name || '-'); 
} 

function analysisConsistencyLabel(value) { 
    const map = { aligned: '一致', divergent: '有分歧', neutral: '中性' }; 
    return map[value] || value || '-'; 
} 

function analysisValidationLabel(v) {
    if (v && v.validation_status === 'target_missing') return '无法验证';
    return analysisConsistencyLabel(v?.consistency);
}

function analysisConsultationLabel(status, hasMajorConflict = false) { 
    if (status === 'completed') return '已会诊'; 
    if (status === 'skipped') return '已跳过'; 
    if (status === 'failed') return '会诊失败'; 
    return hasMajorConflict ? '应会诊但未完成' : '无需会诊'; 
} 

function analysisAdjustmentLabel(value) { 
    const num = Number(value || 0); 
    if (num > 0) return `提高 ${num} 分`; 
    if (num < 0) return `降低 ${Math.abs(num)} 分`; 
    return '不调整'; 
} 

function analysisLocalizeText(text) { 
    let value = String(text || ''); 
    const replacements = [ 
        [/\btrend_expert\b/g, '行情方向专家'], 
        [/\bmomentum_expert\b/g, '盈利质量专家'], 
        [/\bsentiment_expert\b/g, '短线时序专家'], 
        [/\bposition_expert\b/g, '持仓退出专家'], 
        [/\brisk_expert\b/g, '异常风控专家'], 
        [/\bdecision_maker\b/g, '最终交易员'],
        [/\bunknown\b/g, '未知'],
        [/\bmixed\b/g, '震荡分化'],
        [/\brebound_squeeze_up\b/g, '短线普涨反弹'],
        [/\bselloff_squeeze_down\b/g, '短线普跌抛压'],
        [/\buptrend_continuation\b/g, '上行趋势延续'],
        [/\bdowntrend_continuation\b/g, '下行趋势延续'],
        [/\bbalanced\b/g, '均衡捕捉'],
        [/\bpatient\b/g, '耐心等待'],
        [/\bselective_recovery\b/g, '精选修复'],
        [/\bprofit_first_expansion\b/g, '盈利优先扩张'],
        [/\btight_selective_reentry\b/g, '严格精选再入场'],
        [/\btight_selective\b/g, '严格精选'],
        [/\bdiversified_positive_expectancy\b/g, '分散正期望'],
        [/\bnormal_capture\b/g, '常规机会捕捉'],
        [/\bloss_recovery_selective\b/g, '亏损修复精选'],
        [/\brecovery_attack\b/g, '修复进攻'],
        [/\brecovery_selective\b/g, '修复精选'],
        [/\bchop_wait\b/g, '震荡等待'],
        [/\bhard_recovery\b/g, '深度回撤修复'],
        [/\bdrawdown_clamp\b/g, '回撤收紧'],
        [/\bportfolio_roster_build\b/g, '组合队列构建'],
        [/\baligned\b/g, '一致'],
        [/\bdivergent\b/g, '有分歧'], 
        [/\bneutral\b/g, '中性'], 
        [/\bcompleted\b/g, '已会诊'], 
        [/\bskipped\b/g, '已跳过'], 
        [/\bfailed\b/g, '会诊失败'], 
        [/\bdegraded_missing_probe\b/g, '模型缺失降级探针'],
        [/\bweak_conflict_probe\b/g, '弱冲突小仓探针'],
        [/\bexploration\b/g, '探索小仓'],
        [/\bsmall\b/g, '小仓'],
        [/\bmedium\b/g, '中等仓位'],
        [/\bnormal\b/g, '正常仓位'],
        [/\bblocked\b/g, '硬风控阻断'],
        [/ML\/time-series services are unavailable/gi, 'ML/时序服务不可用'],
        [/missing model data is treated as degraded evidence/gi, '缺失模型数据按降级证据处理'],
        [/controlled tiny probe/gi, '受控极小仓探针'],
        [/hard execution veto/gi, '硬执行否决'],
        [/\bclose_long\b/g, '平多'], 
        [/\bclose_short\b/g, '平空'], 
        [/\bopen_long\b/g, '做多'], 
        [/\bopen_short\b/g, '做空'], 
        [/\blong\b/g, '做多'], 
        [/\bshort\b/g, '做空'], 
        [/\bhold\b/g, '观望'], 
    ]; 
    replacements.forEach(([pattern, label]) => { 
        value = value.replace(pattern, label); 
    }); 
    return value; 
} 

function analysisText(text, fallback = '-') {  
    const value = text === null || text === undefined || text === '' ? fallback : text;  
    return escapeMultiline(analysisLocalizeText(value));  
}  

function analysisConsultationAttemptLabel(status) {
    const map = {
        completed: '成功',
        empty_response: '空返回',
        invalid_json: '格式错误',
        call_failed: '调用失败',
    };
    return map[status] || status || '-';
}

function renderConsultationAttempts(consultation) {
    const attempts = Array.isArray(consultation?.consultation_attempts)
        ? consultation.consultation_attempts
        : [];
    if (!attempts.length) return '';
    const rows = attempts.map(item => {
        const label = item.expert_label || item.expert || '-';
        const model = item.model ? ` / ${item.model}` : '';
        const attempt = item.attempt ? `第 ${item.attempt} 次` : '';
        const status = analysisConsultationAttemptLabel(item.status);
        const message = item.message ? `：${item.message}` : '';
        return `<div class="analysis-note"><span>${escHtml(label)}${escHtml(model)} ${escHtml(attempt)}</span>${analysisText(`${status}${message}`)}</div>`;
    }).join('');
    return `<div style="margin-top:10px;">${rows}</div>`;
}

function analysisTone(actionOrStatus) {
    const value = String(actionOrStatus || '').toLowerCase();
    if (['long', 'buy', 'aligned', 'completed'].includes(value)) return 'good';
    if (['short', 'sell', 'divergent', 'failed', 'target_missing'].includes(value)) return 'bad';
    if (['close_long', 'close_short', 'neutral', 'skipped'].includes(value)) return 'warn';
    return 'muted';
}

function analysisPositionSide(record) {
    if (!record || analysisRecordType(record) !== 'position') return '';
    const direct = String(record.position_side || record.current_position_side || '').toLowerCase();
    if (direct === 'long' || direct === 'short') return direct;
    const finalAction = String(record.final_action || '').toLowerCase();
    if (finalAction === 'close_long') return 'long';
    if (finalAction === 'close_short') return 'short';
    const text = [
        record.final_reasoning || '',
        ...(record.experts || []).map(e => e.reasoning || ''),
    ].join(' ');
    if (text.includes('空单') || text.includes('空仓')) return 'short';
    if (text.includes('多单') || text.includes('多仓')) return 'long';
    return '';
}

function analysisActionLabel(action, record = null) {
    const value = String(action || '').toLowerCase();
    if (!record || analysisRecordType(record) !== 'position') return actionLabel(action);
    const side = analysisPositionSide(record);
    const reviewResult = String(record?.position_review_policy?.result || '').toLowerCase();
    const labels = {
        addLong: '\u52a0\u591a',
        addShort: '\u52a0\u7a7a',
        reverseLong: '\u53cd\u624b\u770b\u591a',
        reverseShort: '\u53cd\u624b\u770b\u7a7a',
        viewLong: '\u770b\u591a',
        viewShort: '\u770b\u7a7a',
        suggestCloseLong: '\u5efa\u8bae\u5e73\u591a',
        suggestCloseShort: '\u5efa\u8bae\u5e73\u7a7a',
        closeLong: '\u5e73\u591a',
        closeShort: '\u5e73\u7a7a',
        hold: '\u7ee7\u7eed\u89c2\u5bdf',
    };
    if (value === 'long' || value === 'open_long') {
        if (reviewResult === 'add' || side === 'long') return labels.addLong;
        if (side === 'short') return labels.reverseLong;
        return labels.viewLong;
    }
    if (value === 'short' || value === 'open_short') {
        if (reviewResult === 'add' || side === 'short') return labels.addShort;
        if (side === 'long') return labels.reverseShort;
        return labels.viewShort;
    }
    if (value === 'close_long') return side === 'long' ? labels.suggestCloseLong : labels.closeLong;
    if (value === 'close_short') return side === 'short' ? labels.suggestCloseShort : labels.closeShort;
    if (['hold', 'wait', 'none', ''].includes(value)) return labels.hold;
    return actionLabel(action);
}

function analysisPill(label, tone = 'muted') {
    return `<span class="analysis-pill analysis-pill-${tone}">${escHtml(label)}</span>`;
}

function analysisMetric(label, value, tone = 'muted') {
    return `
        <div class="analysis-metric analysis-metric-${tone}">
            <div class="analysis-metric-label">${escHtml(label)}</div>
            <div class="analysis-metric-value">${escHtml(value)}</div>
        </div>`;
}

function analysisOpportunityScoreHtml(score, record = null) {
    if (!score || typeof score !== 'object') return '';
    const executionState = opportunityScoreExecutionState(score, record);
    const rank = score.rank && score.candidate_count ? `${score.rank}/${score.candidate_count}` : '-';
    const primaryReturn = opportunityScorePrimaryReturn(score);
    const metrics = [
        ['机会分', opportunityScoreValue(score.score, 6)],
        ['排名', rank],
        ['方向', actionLabel(score.side || '-')],
        [primaryReturn.label, `${opportunityScoreValue(primaryReturn.value, 4)}%`],
        ['反向优势', `${opportunityScoreValue(score.profit_edge_pct, 4)}%`],
        ['ML胜率', `${opportunityScoreValue(Number(score.win_rate || 0) * 100, 1)}%`],
        ['执行状态', executionState.label],
    ].map(([label, value]) => `
        <div class="analysis-opportunity-metric">
            <span>${escHtml(label)}</span>
            <strong>${escHtml(value)}</strong>
        </div>
    `).join('');
    const reason = score.selection_reason || '用于把多个开仓候选按预期净收益排序，不替代 AI 对方向、仓位、杠杆和平仓的裁决。';
    const formulaHtml = opportunityScoreFormulaHtml(score);
    return `
        <div class="analysis-opportunity-card">
            <div class="analysis-opportunity-head"><span>盈利机会评分</span><em>${escHtml(executionState.label)}</em></div>
            <div class="analysis-opportunity-grid">${metrics}</div>
            ${formulaHtml}
        </div>
        <div class="analysis-note analysis-note-muted"><span>排序原因</span>${analysisText(reason)}</div>
    `;
}

function analysisSection(title, body, subtitle = '') {
    return `
        <section class="analysis-section">
            <div class="analysis-section-head">
                <div class="analysis-section-title">${escHtml(title)}</div>
                ${subtitle ? `<div class="analysis-section-subtitle">${escHtml(subtitle)}</div>` : ''}
            </div>
            ${body}
        </section>`;
}

function analysisDurationLabel(seconds) {
    const value = Number(seconds || 0);
    if (!Number.isFinite(value) || value <= 0) return '-';
    if (value < 0.1) return '<0.1秒';
    if (value < 60) return `${value.toFixed(1)}秒`;
    return `${Math.floor(value / 60)}分${(value % 60).toFixed(1)}秒`;
}

function analysisLatencyPillText(latency) {
    if (!latency || latency.duration_sec === undefined) return '';
    if (latency.shared_batch_call || latency.batch_expert) {
        const shared = Number(latency.shared_batch_duration_sec || latency.batch_duration_sec || latency.duration_sec || 0);
        return shared > 0 ? `同批共享 · 批量耗时 ${analysisDurationLabel(shared)}` : '同批共享，见批量请求';
    }
    return `耗时 ${analysisDurationLabel(latency.duration_sec)}`;
}

function analysisModelTimingText(item) {
    if (item.shared_batch_call || item.batch_expert) {
        const batchSize = Number(item.batch_model_count || 0);
        const batchText = batchSize > 1 ? ` · 同批 ${batchSize} 个专家` : '';
        const shared = Number(item.shared_batch_duration_sec || item.batch_duration_sec || item.duration_sec || 0);
        const durationText = shared > 0 ? `批量耗时 ${analysisDurationLabel(shared)}` : '批量耗时见同批请求';
        return `同批共享 · ${durationText} · ${escHtml(analysisTimingStatusLabel(item.status))}${batchText}${item.provider_model ? ` · ${escHtml(item.provider_model)}` : ''}`;
    }
    return `${analysisDurationLabel(item.duration_sec)} · ${escHtml(analysisTimingStatusLabel(item.status))}${item.provider_model ? ` · ${escHtml(item.provider_model)}` : ''}`;
}

function analysisTimingAttemptKey(item) {
    if (!item) return '';
    return [
        item.stage || '',
        item.started_at || '',
        item.provider_model || '',
        item.duration_kind || '',
        item.duration_sec || '',
    ].join('|');
}

function analysisFinalModelTimings(modelTimings) {
    const byName = new Map();
    (modelTimings || []).forEach(item => {
        if (!item || !item.name) return;
        if (item.shared_batch_call || item.batch_expert) return;
        byName.set(String(item.name), item);
    });
    return Array.from(byName.values());
}

function analysisSharedBatchCalls(modelTimings) {
    const calls = new Map();
    (modelTimings || []).forEach(item => {
        if (!item || !(item.shared_batch_call || item.batch_expert)) return;
        const key = analysisTimingAttemptKey(item);
        const current = calls.get(key) || {
            duration_sec: 0,
            expert_names: new Set(),
            provider_model: item.provider_model || '',
            started_at: item.started_at || '',
        };
        current.duration_sec = Math.max(current.duration_sec, Number(item.duration_sec || 0));
        current.provider_model = current.provider_model || item.provider_model || '';
        current.started_at = current.started_at || item.started_at || '';
        if (item.name) current.expert_names.add(String(item.name));
        calls.set(key, current);
    });
    return Array.from(calls.values()).map(call => ({
        ...call,
        expert_names: Array.from(call.expert_names),
    }));
}

function analysisStageLabel(stage) {
    const labels = {
        expert_initial: '专家初诊',
        cross_validation: '交叉验证',
        deep_consultation: '深度会诊',
        ensemble_rules: '规则汇总',
        decision_maker: '最终交易员',
    };
    return labels[String(stage || '')] || String(stage || '-');
}

function analysisTimingStatusLabel(status) {
    const labels = {
        completed: '完成',
        skipped: '跳过',
        failed: '失败',
        invalid: '无效',
        batch_fallback: '批量回退',
        partial_batch_fallback: '批量缺失',
        independent_provider: '独立专家',
        batch_format_independent: '独立专家',
        batch_timeout_independent: '独立专家',
        independent_provider_fallback: '独立调用失败，本地兜底',
        independent_provider_failed: '独立调用失败',
        circuit_breaker_fallback: '熔断兜底',
        timeout_fallback: '超时兜底',
    };
    return labels[String(status || '')] || String(status || '-');
}

function analysisExpertStatusLine(record, missingCount) {
    if (analysisIsFastPositionScan(record)) {
        return '快速扫全部持仓，未调用 5 个专家；有强信号时才进入深度复盘';
    }
    if (missingCount) return `${missingCount} 个未返回，点详情查看原因`;
    return '5 个专家都已返回';
}

function pctLabel(value, digits = 0) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '-';
    return `${(num * 100).toFixed(digits)}%`;
}

function signedPctValueLabel(value, digits = 2) {
    const num = Number(value);
    if (!Number.isFinite(num)) return '-';
    const sign = num > 0 ? '+' : '';
    return `${sign}${num.toFixed(digits)}%`;
}

function renderAnalysisMlSignal(signal) {
    if (!signal || signal.available === false) {
        return '<div class="analysis-empty">本轮没有可用的本地 ML 盈亏质量预测；AI 决策未受 ML 影响。</div>';
    }
    const predictions = Array.isArray(signal.predictions) ? signal.predictions : [];
    const rows = predictions.map(item => {
        const bestSide = item.best_side === 'long' ? '做多' : item.best_side === 'short' ? '做空' : '-';
        const expected = Number(item.best_expected_return_pct || 0);
        const edge = Number(item.profit_edge_pct || 0);
        const tone = expected > 0 && edge > 0 ? 'good' : Number(item.risk_score || 0) >= 0.55 ? 'warn' : 'muted';
        return `
            <div class="analysis-resolution-item">
                <strong>${Number(item.horizon_minutes || 0)}分钟</strong>
                <span>
                    预期收益 ${signedPctValueLabel(item.best_expected_return_pct)}
                    · 收益差 ${signedPctValueLabel(item.profit_edge_pct)}
                    · ${bestSide}
                    · 胜率辅助 ${pctLabel(item.best_win_rate)}
                    · ${analysisPill(`风险 ${pctLabel(item.risk_score)}`, tone)}
                </span>
            </div>`;
    }).join('');
    const influenceEnabled = signal.influence_enabled !== false && (signal.mode === 'entry_profit_filter' || signal.status === 'entry_profit_filter');
    const modeLabel = influenceEnabled
        ? '参与开仓过滤'
        : '学习观察中';
    const influenceReason = signal?.influence_policy?.disabled_reason
        || signal.note
        || (influenceEnabled ? 'ML 指标达标，参与开仓质量过滤和机会排序。' : 'ML 指标未达标，继续学习训练，暂不影响交易。');
    return `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">本地盈亏质量模型</div>
                <div class="analysis-card-tags">
                    ${analysisPill(modeLabel, influenceEnabled ? 'good' : 'warn')}
                    ${signal.trained_sample_count ? analysisPill(`样本 ${Number(signal.trained_sample_count)}`, 'good') : ''}
                    ${signal.model_version ? analysisPill(String(signal.model_version).slice(0, 10), 'muted') : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note"><span>模型建议</span>${analysisText(signal.suggestion || signal.note || '以预期盈亏为主，胜率仅作辅助。')}</div>
                <div class="analysis-note analysis-note-muted"><span>预测结果</span>
                    <div class="analysis-resolution-list">${rows || '<div class="analysis-empty">暂无预测明细</div>'}</div>
                </div>
                <div class="analysis-note analysis-note-muted"><span>生效方式</span>${analysisText(influenceEnabled ? 'ML 参与开仓门槛/否决和机会排序：预期收益为负会拦截，收益质量强可小幅加分；胜率只作辅助，不直接决定方向。' : 'ML 当前只学习不介入：继续预测、记录影子复盘和自动训练；达到上岗指标后自动恢复参与。')}</div>
                <div class="analysis-note analysis-note-muted"><span>上岗判断</span>${analysisText(influenceReason)}</div>
            </div>
        </div>`;
}

function analysisDecisionLabel(decision) {
    return ({
        hold: '继续持有',
        wait: '继续观察',
        observe: '继续观察',
        neutral: '中性',
        allow: '允许',
        block: '拦截',
        block_entry: '禁止开仓',
        all_position_actions_available: '持仓动作可用',
        exit_reduce_hold_only: '只允许平仓/减仓/持有',
        long: '做多',
        short: '做空',
        close: '平仓',
        reduce: '减仓',
        partial_close: '部分平仓',
        full_close: '全部平仓',
        close_long: '平多',
        close_short: '平空',
        focus_review: '重点复盘',
        no_position: '无匹配持仓',
        reduce_or_close: '减仓或平仓',
        protect_profit: '保护利润',
        close_if_ai_agrees: 'AI确认后平仓',
        trail_profit: '移动锁盈',
    }[String(decision || '').toLowerCase()] || decision || '');
}

function analysisReasonLabel(reason) {
    const text = String(reason || '').trim();
    const normalized = text.toLowerCase().replace(/[.。]+$/g, '');
    return ({
        'no trained exit pressure': '平仓建议模型未识别到明确的主动平仓压力，本轮倾向继续持有。',
        'no exit pressure': '平仓建议模型未识别到明确的主动平仓压力，本轮倾向继续持有。',
        'no trained close pressure': '平仓建议模型未识别到明确的主动平仓压力，本轮倾向继续持有。',
        'no matching open position was supplied': '本轮没有传入与该币种匹配的当前持仓，平仓建议模型不参与。',
        'this symbol/side has weak realized profile and the open position is losing': '该币种/方向历史实盘表现偏弱，且当前持仓正在亏损，建议减仓或平仓。',
        'profit exists but historical giveback/loss pressure is elevated': '当前已有浮盈，但历史回吐或亏损压力偏高，建议优先保护利润。',
        'loss is expanding beyond the local exit model tolerance': '亏损扩大到本地平仓模型容忍线之外，若 AI 也确认应优先退出。',
        'position is profitable; trail rather than cap upside immediately': '当前持仓盈利且历史盈亏质量尚可，建议移动保护利润，不急于完全限制上行空间。',
    }[normalized] || text);
}

const SKILL_DATA_LABELS = {
    available: '可用',
    model: '模型',
    backend: '后端',
    endpoint: '端点',
    path: '接口',
    duration_sec: '耗时',
    latency_ms: '延迟',
    best_side: '优选方向',
    side: '方向',
    direction: '趋势',
    label: '标签',
    sentiment: '情绪',
    score: '分数',
    sentiment_score: '情绪分',
    expected_return_pct: '预期收益',
    expected_net_return_pct: '预期净收益',
    expected_move_pct: '预期波动',
    expected_net_pnl: '预期净盈亏',
    profit_edge_pct: '收益差',
    loss_probability: '亏损概率',
    confidence: '信心',
    recommendation: '建议',
    action: '动作',
    action_label: '动作说明',
    risk_level: '风险级别',
    ready: '就绪',
    suggestion: '模型建议',
    reason: '原因',
    note: '备注',
    regime: '行情过滤',
    strategy: '调度策略',
    mode: '市场状态',
    posture: '调度姿态',
    allow_long: '允许做多',
    allow_short: '允许做空',
    avoid_long: '谨慎做多',
    avoid_short: '谨慎做空',
    blocked_directions: '受限方向',
};

const SENSITIVE_DATA_KEY_RE = /(api[_-]?key|secret|password|passphrase|token|authorization|webhook)/i;

function skillDataValueLabel(key, value) {
    if (value === null || value === undefined || value === '') return '';
    if (SENSITIVE_DATA_KEY_RE.test(String(key || ''))) return '';
    if (typeof value === 'boolean') return value ? '是' : '否';
    if (['expected_return_pct', 'expected_net_return_pct', 'expected_move_pct', 'profit_edge_pct'].includes(key)) {
        return signedPctValueLabel(value, 4);
    }
    if (key === 'loss_probability' || key === 'confidence') {
        return pctLabel(value, 1);
    }
    if (key === 'expected_net_pnl') {
        return `${signedMoney(value)} USDT`; 
    }
    if (key === 'duration_sec') return analysisDurationLabel(value);
    if (key === 'latency_ms') return `${monitorNumber(value, 0)} ms`; 
    if (['side', 'best_side', 'action'].includes(key)) return analysisDecisionLabel(value) || String(value);
    if (typeof value === 'number') return Number.isFinite(value) ? String(Number(value.toFixed ? value.toFixed(6) : value)) : '';
    if (Array.isArray(value)) return value.slice(0, 4).map(item => typeof item === 'object' ? skillDataValueLabel(key, item) : String(item)).filter(Boolean).join('、');
    if (typeof value === 'object') {
        return Object.entries(value)
            .filter(([childKey]) => !SENSITIVE_DATA_KEY_RE.test(String(childKey || '')))
            .map(([childKey, childValue]) => {
                const label = SKILL_DATA_LABELS[childKey] || childKey;
                const childText = skillDataValueLabel(childKey, childValue);
                return childText ? `${label} ${childText}` : '';
            })
            .filter(Boolean)
            .slice(0, 4)
            .join('；');
    }
    return String(value).slice(0, 160);
}

function renderSkillDataSummary(data) {
    if (!data || typeof data !== 'object') return ''; 
    const rows = Object.entries(data)
        .filter(([key]) => !SENSITIVE_DATA_KEY_RE.test(String(key || '')))
        .map(([key, value]) => {
            const label = SKILL_DATA_LABELS[key] || key;
            const text = skillDataValueLabel(key, value);
            if (!text) return ''; 
            return `
                <div class="analysis-skill-data-row">
                    <span class="analysis-skill-data-key">${escHtml(label)}</span>
                    <span class="analysis-skill-data-value">${analysisText(text)}</span>
                </div>
            `;
        })
        .filter(Boolean)
        .slice(0, 8);
    if (!rows.length) return ''; 
    return `<div class="analysis-skill-data-grid" aria-label="SkillData">${rows.join('')}</div>`;
}

function renderAnalysisAgentSkills(agentSkills) {
    if (!agentSkills || !agentSkills.phases) {
        return '<div class="analysis-empty">本条记录还没有 Agent/Skills 归因数据。新分析会逐步写入市场、持仓和执行前守门结果。</div>';
    }
    const phases = Object.values(agentSkills.phases || {});
    if (!phases.length) {
        return '<div class="analysis-empty">本条记录没有可展示的 Skills 阶段。</div>';
    }
    const phaseLabel = {
        market_prefilter: '市场预筛',
        market_analysis: '市场分析',
        position_review: '持仓分析',
        position_fast_scan: '持仓快速扫描',
        execution_precheck: '执行前检查',
    };
    const statusLabel = (status) => ({
        active: '已参与',
        passed: '通过',
        supported: '支持',
        warning: '提醒',
        partial: '部分可用',
        unavailable: '不可用',
        blocked: '拦截',
        degraded_missing_probe: '模型缺失降级探针',
        inactive: '未触发',
    }[String(status || '').toLowerCase()] || status || '-');
    const statusTone = (skill) => {
        if (skill.blocks_entry || skill.blocks_exit || skill.status === 'blocked') return 'bad';
        if (['warning', 'partial', 'unavailable'].includes(String(skill.status || ''))) return 'warn';
        if (['active', 'passed', 'supported'].includes(String(skill.status || ''))) return 'good';
        return 'muted';
    };
    const skillRows = phases.map(phase => {
        const skills = Array.isArray(phase.skills) ? phase.skills : [];
        const rows = skills.map(skill => {
            const tone = statusTone(skill);
            return `
                <div class="analysis-skill-item">
                    <div class="analysis-skill-head">
                        <strong class="analysis-skill-title">${escHtml(skill.label || skill.name || '-')}</strong>
                        <div class="analysis-skill-badges">
                            ${analysisPill(statusLabel(skill.status), tone)}
                            ${skill.decision ? analysisPill(analysisDecisionLabel(skill.decision), tone) : ''}
                            ${skill.confidence !== undefined ? analysisPill(`信心 ${(Number(skill.confidence || 0) * 100).toFixed(0)}%`, 'muted') : ''}
                        </div>
                    </div>
                    <div class="analysis-skill-body">
                        <div class="analysis-skill-reason"><span>结论</span>${analysisText(analysisReasonLabel(skill.reason || '-'))}</div>
                        ${renderSkillDataSummary(skill.data)}
                    </div>
                </div>
            `;
        }).join('');
        return `
            <div class="analysis-card analysis-final-card">
                <div class="analysis-card-head">
                    <div class="analysis-card-title">${escHtml(phaseLabel[phase.phase] || phase.phase || 'Agent/Skills')}</div>
                    <div class="analysis-card-tags">
                        ${phase.note ? analysisPill('有说明', 'muted') : ''}
                        ${analysisPill(`${skills.length} 个 Skill`, skills.length ? 'good' : 'warn')}
                    </div>
                </div>
                <div class="analysis-card-text">
                    ${phase.note ? `<div class="analysis-note analysis-note-muted"><span>阶段说明</span>${analysisText(phase.note)}</div>` : ''}
                    <div class="analysis-resolution-list">${rows || '<div class="analysis-empty">该阶段没有返回 Skill 明细。</div>'}</div>
                </div>
            </div>
        `;
    }).join('');
    return `<div class="analysis-grid analysis-agent-skills-grid">${skillRows}</div>`;
}

function unwrapAnalysisToolPayload(value) {
    if (!value || typeof value !== 'object') return {};
    const wrappedKeys = ['data', 'result', 'prediction', 'payload', 'output'];
    for (const key of wrappedKeys) {
        if (value[key] && typeof value[key] === 'object') {
            return { ...value, ...unwrapAnalysisToolPayload(value[key]) };
        }
    }
    return value;
}

function analysisToolSection(tools, aliases) {
    if (!tools || typeof tools !== 'object') return {};
    for (const key of aliases) {
        const payload = unwrapAnalysisToolPayload(tools[key]);
        if (Object.keys(payload).length) return payload;
    }
    return {};
}

function analysisToolAvailable(payload) {
    if (!payload || typeof payload !== 'object' || !Object.keys(payload).length) return false;
    if (payload.error || payload.exception) return false;
    const status = String(payload.status || '').toLowerCase();
    if (['unavailable', 'error', 'disabled', 'circuit_open', 'failed'].includes(status)) return false;
    if (payload.available === false || payload.enabled === false || payload.ok === false) return false;
    return true;
}

function analysisToolPlainStatus(payload) {
    if (!payload || typeof payload !== 'object' || !Object.keys(payload).length) return '未返回';
    const status = String(payload.status || '').toLowerCase();
    const labels = {
        returned: '已返回',
        completed: '完成',
        ok: '正常',
        supported: '支持',
        active: '已参与',
        trained_torch_sequence_model: '已训练时序模型',
        trained_text_model: '已训练情绪模型',
        heuristic_fallback_available: '启发式可用',
        unavailable: '不可用',
        error: '错误',
        disabled: '已关闭',
        circuit_open: '熔断中',
        failed: '失败',
    };
    if (status && labels[status]) return labels[status];
    if (payload.trained === false) return '学习中';
    if (payload.model || payload.backend || payload.available === true || payload.ok === true) return '已返回';
    return status || '已参与';
}

function analysisToolMetaText(payload) {
    if (!payload || typeof payload !== 'object' || !Object.keys(payload).length) return '未返回';
    const parts = [`状态 ${analysisToolPlainStatus(payload)}`];
    const duration = Number(payload.duration_sec || 0);
    if (duration > 0) parts.push(`耗时 ${analysisDurationLabel(duration)}`);
    if (payload.model || payload.backend) parts.push(`模型 ${payload.model || payload.backend}`);
    if (payload.path) parts.push(`接口 ${payload.path}`);
    return parts.join('；');
}

function analysisToolStatus(payload) {
    if (!payload || typeof payload !== 'object' || !Object.keys(payload).length) {
        return analysisPill('未返回', 'warn');
    }
    if (!analysisToolAvailable(payload)) {
        return analysisPill(analysisToolPlainStatus(payload), 'warn');
    }
    return analysisPill(analysisToolPlainStatus(payload), payload.trained === false ? 'warn' : 'good');
}

function renderAnalysisLocalAiTools(tools, analysisType = 'market') {
    if (!tools) {
        return '<div class="analysis-empty">本轮没有调用服务器量化工具。</div>';
    }
    const profit = analysisToolSection(tools, ['profit_prediction', 'profit_model', 'server_profit', 'server_profit_model', 'profit']);
    const ts = analysisToolSection(tools, ['time_series_prediction', 'timeseries_prediction', 'sequence_prediction', 'timeseries', 'time_series']);
    const sentiment = analysisToolSection(tools, ['sentiment_analysis', 'sentiment_prediction', 'sentiment_model', 'sentiment']);
    const exitAdvice = analysisToolSection(tools, ['exit_advice', 'exit_model', 'position_exit', 'exit']);
    const hasAnyToolPayload = [profit, ts, sentiment, exitAdvice].some(item => item && Object.keys(item).length > 0);
    if (tools.enabled === false && !hasAnyToolPayload) {
        return '<div class="analysis-empty">本轮没有调用服务器量化工具。</div>';
    }
    const isPositionAnalysis = ['position', 'position_review'].includes(String(analysisType || '').toLowerCase());
    const predictions = Array.isArray(ts.predictions) ? ts.predictions : [];
    const predictionRows = predictions.map(item => `
        <div class="analysis-resolution-item">
            <strong>${Number(item.horizon_minutes || item.horizon || 0)}分钟</strong>
            <span>
                预期 ${signedPctValueLabel(item.expected_return_pct)}
                ${item.direction ? ` · 方向 ${escHtml(String(item.direction))}` : ''}
                ${item.downside_risk_pct !== undefined ? ` · 下行风险 ${signedPctValueLabel(item.downside_risk_pct)}` : ''}
            </span>
        </div>
    `).join('') || (ts.available ? `
        <div class="analysis-resolution-item">
            <strong>当前窗口</strong>
            <span>
                方向 ${escHtml(ts.direction || '-')}
                · 预期波动 ${signedPctValueLabel(ts.expected_move_pct)}
                · 信心 ${pctLabel(ts.confidence)}
                ${ts.sample_count ? ` · 样本 ${Number(ts.sample_count)}` : ''}
            </span>
        </div>
    ` : '');
    const profitStatus = analysisToolStatus(profit);
    const tsStatus = analysisToolStatus(ts);
    const sentimentStatus = analysisToolStatus(sentiment);
    const exitStatus = !isPositionAnalysis
        ? analysisPill('市场分析不适用', 'muted')
        : (!analysisToolAvailable(exitAdvice)
            ? analysisToolStatus(exitAdvice)
            : analysisPill(exitAdvice.action ? '已参与' : '本轮无持仓建议', exitAdvice.action ? 'good' : 'muted'));
    return `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">服务器量化工具</div>
                <div class="analysis-card-tags">
                    ${analysisPill(tools.status || 'completed', tools.status === 'completed' ? 'good' : 'warn')}
                    ${tools.duration_sec !== undefined ? analysisPill(`耗时 ${analysisDurationLabel(tools.duration_sec)}`, Number(tools.duration_sec || 0) > 2 ? 'warn' : 'muted') : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note"><span>盈利预测</span>
                    ${profitStatus}
                    ${analysisText([
                        analysisToolMetaText(profit),
                        `最佳方向 ${profit.best_side || '-'}`,
                        `预期收益 ${signedPctValueLabel(profit.expected_return_pct)}`,
                        `收益优势 ${signedPctValueLabel(profit.profit_edge_pct)}`,
                        `质量分 ${profit.profit_quality_score ?? '-'}`,
                        `做多亏损概率 ${pctLabel(profit.long_loss_probability)}`,
                        `做空亏损概率 ${pctLabel(profit.short_loss_probability)}`,
                        `模型 ${profit.model || profit.backend || '-'}`
                    ].join('；'))}
                </div>
                <div class="analysis-note analysis-note-muted"><span>时序预测</span>
                    ${tsStatus}
                    ${analysisText(analysisToolMetaText(ts))}
                    <div class="analysis-resolution-list">${predictionRows || '<div class="analysis-empty">暂无时序预测明细</div>'}</div>
                </div>
                <div class="analysis-note analysis-note-muted"><span>情绪模型</span>
                    ${sentimentStatus}
                    ${analysisText([
                        analysisToolMetaText(sentiment),
                        `结论 ${sentiment.label || '-'}`,
                        `情绪分 ${sentiment.score ?? '-'}`,
                        `情绪预期收益 ${signedPctValueLabel(sentiment.expected_return_from_sentiment_pct)}`,
                        `风险 ${sentiment.risk_level || '-'}`,
                        `模型 ${sentiment.model || sentiment.backend || '-'}`
                    ].join('；'))}
                </div>
                ${isPositionAnalysis ? `<div class="analysis-note analysis-note-muted"><span>平仓建议</span>
                    ${exitStatus}
                    ${analysisText(exitAdvice.action ? `${analysisToolMetaText(exitAdvice)}；${exitAdvice.action_label || analysisDecisionLabel(exitAdvice.action)}，信心 ${(Number(exitAdvice.confidence || 0) * 100).toFixed(0)}%${exitAdvice.reason ? `，原因：${analysisReasonLabel(exitAdvice.reason)}` : ''}` : '本轮没有返回独立平仓建议；如果不是持仓分析记录，通常不会触发这一项。')}
                </div>` : `<div class="analysis-note analysis-note-muted"><span>平仓建议</span>${exitStatus}${analysisText('市场分析只评估开仓机会、方向和风险；平仓建议只在持仓分析中显示。')}</div>`}
                ${tools.errors ? `<div class="analysis-note analysis-note-muted"><span>部分错误</span>${analysisText(JSON.stringify(tools.errors))}</div>` : ''}
            </div>
        </div>`;
}

function renderAnalysisNewsContext(news) {
    if (!news) {
        return '<div class="analysis-empty">本轮没有新闻上下文。</div>'; 
    }
    const items = Array.isArray(news.items) ? news.items : [];
    const derivedDirectCount = items.filter(item => item && item.direct_match === true).length;
    const derivedMarketCount = items.filter(item => !item || item.direct_match !== true).length;
    const directCount = Number(news.direct_news_item_count ?? derivedDirectCount);
    const marketCount = Number(news.market_news_item_count ?? derivedMarketCount);
    const hasDirectNews = directCount > 0;
    const hasMarketNews = marketCount > 0;
    const statusTone = hasDirectNews ? 'good' : 'muted';
    const statusLabel = hasDirectNews
        ? `${directCount} 条直接相关`
        : (hasMarketNews ? `直接新闻 0 / 全市场 ${marketCount}` : '新闻中性');
    const dataNote = hasDirectNews
        ? '本轮有直接匹配到该币种的新闻，短线时序专家可以把它作为该币种的利好、利空或风险证据。'
        : (hasMarketNews
            ? '本轮没有直接匹配该币种的新闻；全市场新闻只作为大盘风险背景。无直接新闻按情绪中性处理，不阻止开仓。'
            : '本轮暂无新闻/社媒证据；情绪按中性处理，不作为开仓阻碍。');
    const renderNewsRows = (list) => list.map(item => {
        const impact = Number(item.impact_level || 1);
        const sentiment = Number(item.sentiment_score || 0);
        const tone = impact >= 4 || Math.abs(sentiment) >= 0.5 ? 'warn' : (item.direct_match ? 'good' : 'muted');
        const title = escHtml(item.title || '-');
        const source = escHtml(item.source || '-');
        const rawEventType = item.event_type || 'market_news';
        const eventType = escHtml(item.direct_match && rawEventType === 'market_news' ? 'symbol_news' : rawEventType);
        const reason = escHtml(item.match_reason || '');
        const sourceUrl = safeExternalUrl(item.url);
        const url = sourceUrl ? `<a href="${escHtml(sourceUrl)}" target="_blank" rel="noopener noreferrer">来源</a>` : '';
        return `
            <div class="analysis-resolution-item">
                <strong>${source}</strong>
                <span>
                    ${analysisPill(item.direct_match ? '直接相关' : '全市场', item.direct_match ? 'good' : 'muted')}
                    ${analysisPill(eventType, tone)}
                    ${analysisPill(`影响 ${impact}/5`, tone)}
                    ${analysisPill(`情绪 ${sentiment.toFixed(2)}`, sentiment > 0 ? 'good' : sentiment < 0 ? 'warn' : 'muted')}
                    ${url}
                    <br>${title}
                    ${reason ? `<br><span style="color:var(--text-muted);">${reason}</span>` : ''}
                </span>
            </div>`;
    }).join('');
    const directRows = renderNewsRows(items.filter(item => item && item.direct_match === true));
    const marketRows = renderNewsRows(items.filter(item => !item || item.direct_match !== true));
    const newsGroups = `
        <details class="analysis-news-group" ${directRows ? 'open' : ''}>
            <summary>直接相关新闻<span>${directCount} 条</span></summary>
            <div class="analysis-news-group-body">${directRows || '<div class="analysis-news-empty">本轮没有直接匹配该币种的新闻。</div>'}</div>
        </details>
        <details class="analysis-news-group" ${!directRows && marketRows ? 'open' : ''}>
            <summary>全市场背景新闻<span>${marketCount} 条</span></summary>
            <div class="analysis-news-group-body">${marketRows || '<div class="analysis-news-empty">暂无全市场背景新闻。</div>'}</div>
        </details>`;
    return `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">新闻与事件</div>
                <div class="analysis-card-tags">
                    ${analysisPill(statusLabel, statusTone)}
                    ${analysisPill(`新闻 ${Number(news.news_article_count || 0)}`, Number(news.news_article_count || 0) ? 'good' : 'muted')}
                    ${analysisPill(`社媒 ${Number(news.social_mention_count || 0)}`, Number(news.social_mention_count || 0) ? 'good' : 'muted')}
                    ${analysisPill(`新闻情绪 ${Number(news.news_sentiment_avg || 0).toFixed(2)}`, Number(news.news_sentiment_avg || 0) > 0 ? 'good' : Number(news.news_sentiment_avg || 0) < 0 ? 'warn' : 'muted')}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note"><span>数据说明</span>${analysisText(dataNote)}</div>
                <div class="analysis-note analysis-note-muted"><span>实际新闻</span>
                    <div class="analysis-news-groups">${newsGroups}</div>
                </div>
            </div>
        </div>`;
}

function analysisExpertConfig(name) {
    const models = Array.isArray(state.aiExpertModels) && state.aiExpertModels.length
        ? state.aiExpertModels
        : FIXED_AI_EXPERT_FALLBACKS;
    return models.find(item => item.name === name) || null;
}

function analysisMissingExpertReason(missing, record) {
    const name = missing?.expert_name || '';
    const label = missing?.expert_label || analysisExpertDisplayName(name, record?.experts || []);
    const attempted = Array.isArray(record?.attempted_experts)
        && record.attempted_experts.map(String).includes(String(name));
    const timing = missing?.latency
        || (Array.isArray(record?.model_timings)
            ? record.model_timings.find(item => item.name === name)
            : null);
    const cfg = analysisExpertConfig(name);
    const rawReason = String(missing?.reason || '').trim();
    const lowerReason = rawReason.toLowerCase();

    if (cfg && cfg.loading === true) {
        return `${label} 的系统配置还在加载中，暂时无法判断具体原因。`;
    }
    if (cfg && cfg.enabled === false) {
        return `${label} 在系统设置中已关闭，所以本轮没有发起调用。`;
    }
    if (cfg && !cfg.api_key) {
        return `${label} 未配置 API Key，所以本轮没有发起调用。`;
    }
    if (cfg && !cfg.api_base) {
        return `${label} 未配置 API URL，所以本轮没有发起调用。`;
    }
    if (cfg && !cfg.model) {
        return `${label} 未配置模型名称，所以本轮没有发起调用。`;
    }

    if (attempted || timing) {
        const status = String(timing?.status || '').toLowerCase();
        const detail = timing?.reason || rawReason;
        if (status === 'timeout_fallback' || lowerReason.includes('timeout') || lowerReason.includes('超时')) {
            return `${label} 已发起调用，但 AI 响应超时，本轮结果没有进入专家列表。`;
        }
        if (status === 'invalid' || lowerReason.includes('json') || lowerReason.includes('格式')) {
            return `${label} 已发起调用，但 AI 返回格式不符合 JSON 要求，系统已丢弃这次结果。`;
        }
        if (lowerReason.includes('401') || lowerReason.includes('unauthorized') || lowerReason.includes('invalid api key') || rawReason.includes('API Key 无效')) {
            return `${label} 已发起调用，但 API Key 无效或没有权限。`;
        }
        if (lowerReason.includes('403') || lowerReason.includes('forbidden') || lowerReason.includes('permission') || rawReason.includes('权限不足')) {
            return `${label} 已发起调用，但模型或接口权限不足。`;
        }
        if (lowerReason.includes('429') || lowerReason.includes('rate limit') || rawReason.includes('限流')) {
            return `${label} 已发起调用，但接口限流，请求被服务商拒绝。`;
        }
        if (lowerReason.includes('connect') || lowerReason.includes('connection') || lowerReason.includes('network') || rawReason.includes('连接失败')) {
            return `${label} 已发起调用，但 AI 接口连接失败，本轮没有拿到结果。`;
        }
        if (status === 'failed' || rawReason) {
            return `${label} 已发起调用，但 AI 未响应或调用失败。${detail ? `详情：${detail}` : ''}`;
        }
        return `${label} 已发起调用，但本轮没有返回可用结果，可能是 AI 未响应、超时或返回内容无效。`;
    }

    return `${label} 本轮没有发起调用。可能原因：系统设置中未启用、未配置 API Key、未配置 API URL、未配置模型名称，或服务启动时未加载该专家配置。`;
}
 
function renderAnalysisPage() {  
    const tbody = document.getElementById('analysis-tbody');
    if (!tbody) return;
    const records = getVisibleAnalysisRecords();
    updateAnalysisViewControls(records.length);

    if (!records.length) {
        tbody.innerHTML = `<tr><td colspan="9" style="color:var(--text-muted);text-align:center;padding:24px;">暂无${analysisViewLabel()}记录</td></tr>`;
        document.getElementById('analysis-pagination').style.display = 'none';
        return;
    }

    const totalPages = Number(state.analysisTotalPages || Math.ceil(state.analysisTotal / PAGE_SIZE) || 1);
    const page = Math.min(state.analysisPage, totalPages);
    const pageData = records;

    tbody.innerHTML = pageData.map(r => {
        const conf = Number(r.final_confidence || 0);
        const score = r.weighted_score === null || r.weighted_score === undefined ? '-' : Number(r.weighted_score).toFixed(2);
        const cross = r.cross_summary || {};
        const isFastScan = analysisIsFastPositionScan(r);
        const expertCount = Number(r.expert_count || (r.experts || []).length || 0); 
        const expectedCount = Number(r.expected_expert_count || 5); 
        const attemptedCount = isFastScan ? 0 : Number(r.attempted_expert_count || expectedCount);  
        const missingCount = isFastScan ? 0 : Math.max(expectedCount - expertCount, 0);  
        const hasMajorConflict = Number(cross.major_conflicts || 0) > 0;  
        const completedCross = Number(cross.completed ?? cross.total ?? 0);
        const unavailableCross = Number(cross.unavailable || 0);
        const crossText = isFastScan
            ? '快速扫描未发起交叉验证'
            : `请求 ${Number(r.cross_requested || 0)}，完成 ${completedCross}，无法验证 ${unavailableCross}，分歧 ${Number(cross.divergent || 0)}`;  
        const expertStatusLine = analysisExpertStatusLine(r, missingCount);
        const expertStatusColor = missingCount ? 'var(--yellow)' : 'var(--text-muted)';
        const expertSummary = isFastScan
            ? '快速扫描，未调用专家'
            : `发起 ${attemptedCount}/${expectedCount}，返回 ${expertCount}`;
        return ` 
        <tr> 
            <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(r.created_at)}</td> 
            <td>${escHtml(r.symbol || '-')}</td> 
            <td> 
                <strong>${escHtml(expertSummary)}</strong> 
                <div style="font-size:10px;color:${expertStatusColor};">${escHtml(expertStatusLine)}</div> 
            </td> 
            <td style="font-size:11px;color:${hasMajorConflict ? 'var(--red)' : 'var(--text-muted)'};">${crossText}</td> 
            <td style="font-size:11px;color:var(--text-muted);">${analysisConsultationLabel(r.consultation_status, hasMajorConflict)}</td>
            <td><span class="badge badge-${r.final_action || 'hold'}">${analysisActionLabel(r.final_action, r)}</span></td>
            <td style="color:${conf >= 0.65 ? 'var(--green)' : 'var(--text-muted)'};font-weight:600;">${(conf * 100).toFixed(0)}%</td>
            <td>${score}</td>
            <td>
                <button
                    type="button"
                    class="btn btn-sm js-analysis-reason"
                    data-record-id="${escHtml(r.id ?? '')}"
                    data-decision-id="${escHtml(r.decision_id ?? r.id ?? '')}"
                >查看流程</button>
            </td>
        </tr>
    `}).join('');

    renderPagination('analysis-pagination', page, totalPages, state.analysisTotal, 'changeAnalysisPage');
}

function analysisRecordKeyMatches(record, recordId, decisionId) {
    if (!record) return false;
    const wanted = [recordId, decisionId]
        .filter(value => value !== undefined && value !== null && String(value) !== '')
        .map(value => String(value));
    if (!wanted.length) return false;
    return [record.id, record.decision_id]
        .filter(value => value !== undefined && value !== null)
        .map(value => String(value))
        .some(value => wanted.includes(value));
}

function showAnalysisReasonLoading(recordId) {
    setDecisionModalWide(true);
    document.getElementById('decision-reason-title').textContent = `分析流程 ${recordId || ''}`.trim();
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="analysis-empty">正在加载专家协作流程...</div>
    `;
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex';
}

function showAnalysisReasonLoadError(recordId, message = '没有找到这条分析记录的详情。') {
    setDecisionModalWide(true);
    document.getElementById('decision-reason-title').textContent = `分析流程 ${recordId || ''}`.trim();
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="analysis-card analysis-card-warning">
            <div class="analysis-card-head">
                <div class="analysis-card-title">加载失败</div>
                ${analysisPill('请刷新后重试', 'warn')}
            </div>
            <div class="analysis-card-text">${analysisText(message)}</div>
        </div>
    `;
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex';
}

async function fetchAnalysisRecordDetail(recordId, decisionId) {
    const lookupId = decisionId || recordId;
    if (!lookupId) return null;
    const params = new URLSearchParams({
        page: '1',
        page_size: '1',
        decision_id: String(lookupId),
        include_detail: 'true',
        is_paper: state.mode === 'paper' ? 'true' : 'false',
    });
    const detailData = await fetchJSON(`/api/analysis-records?${params.toString()}`);
    const records = detailData?.records || [];
    return records.find(r => analysisRecordKeyMatches(r, recordId, decisionId)) || records[0] || null;
}

async function showAnalysisReason(recordId, decisionId = null) {
    let record = state.analysisRecords.find(r => analysisRecordKeyMatches(r, recordId, decisionId));
    if (!record || !Array.isArray(record.experts)) {
        showAnalysisReasonLoading(recordId || decisionId);
        const detailed = await fetchAnalysisRecordDetail(
            recordId || record?.id,
            decisionId || record?.decision_id || record?.id
        );
        if (detailed) {
            record = detailed;
            const idx = state.analysisRecords.findIndex(r => analysisRecordKeyMatches(r, recordId, decisionId));
            if (idx >= 0) state.analysisRecords[idx] = detailed;
        }
    }
    if (!record) {
        showAnalysisReasonLoadError(recordId || decisionId);
        return;
    }
    if (!Array.isArray(record.experts)) {
        showAnalysisReasonLoadError(recordId || decisionId, '详情接口暂未返回专家流程数据。');
        return;
    }
    setDecisionModalWide(true);
    const experts = record.experts || []; 
    const crossSummary = record.cross_summary || {};
    const isFastScan = analysisIsFastPositionScan(record);
    const expertCount = Number(record.expert_count || experts.length || 0);  
    const expectedCount = Number(record.expected_expert_count || 5);  
    const attemptedCount = isFastScan ? 0 : Number(record.attempted_expert_count || expectedCount);  
    const completedCross = Number(crossSummary.completed ?? crossSummary.total ?? 0);
    const unavailableCross = Number(crossSummary.unavailable || 0);
    const majorConflicts = Number(crossSummary.major_conflicts || 0);
    const finalConfidence = `${(Number(record.final_confidence || 0) * 100).toFixed(0)}%`;
    const tradeConfidence = `${(Number(record.trade_confidence || 0) * 100).toFixed(0)}%`;
    const positionSize = `${(Number(record.position_size_pct || 0) * 100).toFixed(1)}%`;
    const lifecycleLabel = analysisPositionLifecycleLabel(record);
    const endToEndDuration = Number((record.timing && record.timing.analysis_duration_sec) || 0);
    const totalDuration = Number(
        (record.latency_summary && record.latency_summary.stage_duration_sec)
        || endToEndDuration
        || 0
    );
    const expertSectionSubtitle = isFastScan
        ? '快速扫描，未调用专家'
        : `发起 ${attemptedCount} 个，返回 ${expertCount} 个`;
    const mlSignal = record.ml_signal || null;
    const localAiTools = record.local_ai_tools || null;
    const agentSkills = record.agent_skills || null;
    const newsContext = record.news_context || null;
    const attribution = record.decision_attribution || null;
 
    const expertsHtml = isFastScan ? `
        <div class="analysis-card analysis-card-warning">
            <div class="analysis-card-head">
                <div class="analysis-card-title">持仓快速扫描</div>
                ${analysisPill('未调用专家', 'muted')}
            </div>
            <div class="analysis-card-text">
                ${analysisText(record.flow_summary || '本轮是持仓快速扫描：系统先快速扫全部持仓，没有调用 5 个慢专家；只有出现强平仓、强加仓或高风险信号时才插队进入专家深度复盘。')}
            </div>
        </div>
    ` : experts.map(e => { 
        const targetName = e.cross_check_for 
            ? analysisExpertDisplayName(e.cross_check_for.target, experts) 
            : ''; 
        const latency = e.latency && e.latency.duration_sec !== undefined
            ? analysisPill(
                analysisLatencyPillText(e.latency),
                Number(e.latency.duration_sec || 0) > 25 ? 'warn' : 'muted'
            )
            : '';
        const cross = e.cross_check_for  
            ? `<div class="analysis-note"><span>请求 ${escHtml(targetName)} 核实</span>${analysisText(e.cross_check_for.question || '-')}</div>`  
            : '<div class="analysis-note analysis-note-muted"><span>交叉验证</span>没有提出交叉验证请求</div>';  
        return `  
            <div class="analysis-card">  
                <div class="analysis-card-head">
                    <div class="analysis-card-title">${escHtml(e.expert_label || e.expert_name || '-')}</div>
                    <div class="analysis-card-tags">
                        ${analysisPill(analysisActionLabel(e.action, record), analysisTone(e.action))}
                        ${analysisPill(`信心 ${(Number(e.confidence || 0) * 100).toFixed(0)}%`, Number(e.confidence || 0) >= 0.6 ? 'good' : 'muted')}
                        ${analysisPill(`权重 ${Number(e.weight || 0).toFixed(2)}`, 'muted')}
                        ${e.timeout_fallback ? analysisPill('超时降级', 'warn') : ''}
                        ${latency}
                    </div>
                </div>
                <div class="analysis-card-text">  
                    ${analysisText(e.reasoning || '暂无分析内容')}  
                    ${cross}  
                </div>  
            </div>`;  
    }).join('');  
 
    const missingHtml = isFastScan ? '' : (record.missing_experts || []).map(e => {
        const reason = analysisMissingExpertReason(e, record);
        const notCalled = !Array.isArray(record.attempted_experts)
            || !record.attempted_experts.map(String).includes(String(e.expert_name || ''));
        const pillText = notCalled ? '未调用' : '未返回';
        const pillTone = 'bad';
        return `  
        <div class="analysis-card analysis-card-warning">  
            <div class="analysis-card-head">
                <div class="analysis-card-title">${escHtml(e.expert_label || e.expert_name || '-')}</div>
                ${analysisPill(pillText, pillTone)}
            </div>
            <div class="analysis-card-text">${analysisText(reason)}</div>  
        </div>`;
    }).join('');  
 
    const pairValidations = (record.cross_validations || []).map(v => {  
        const names = (v.expert_pair || []).map(name => analysisExpertDisplayName(name, experts)).join(' / ');  
        const validationStatus = analysisValidationLabel(v);
        const statusTone = v.validation_status === 'target_missing' ? 'bad' : analysisTone(v.consistency);
        const validationNote = v.validation_note || v.conflict_note || '已按核实问题完成检查，未发现需要降级的矛盾。';
        const checkedEvidence = Array.isArray(v.checked_evidence) && v.checked_evidence.length
            ? `<div class="analysis-note analysis-note-muted"><span>核验依据</span>${analysisText(v.checked_evidence.join('；'))}</div>`
            : '';
        return `   
            <div class="analysis-card">   
                <div class="analysis-card-head">
                    <div class="analysis-card-title">${escHtml(names || '-')}</div>
                    <div class="analysis-card-tags">
                        ${analysisPill(validationStatus, statusTone)}
                        ${analysisPill(analysisAdjustmentLabel(v.confidence_adjustment), Number(v.confidence_adjustment || 0) > 0 ? 'good' : Number(v.confidence_adjustment || 0) < 0 ? 'warn' : 'muted')}
                    </div>
                </div>
                <div class="analysis-card-text">   
                    <div class="analysis-question">核实问题：${analysisText(v.question || '-')}</div>
                    <div class="analysis-note"><span>核验结论</span>${analysisText(validationNote)}</div>
                    ${checkedEvidence}
                </div>
            </div>`;  
    }).join('');  
 
    const consultationTitle = record.consultation
        ? (record.consultation.consultation_expert_label || analysisExpertDisplayName(record.consultation.consultation_expert || 'trend_expert', experts))
        : '';
    const consultationAttempts = record.consultation ? renderConsultationAttempts(record.consultation) : '';
    const consultation = record.consultation ? `  
        <div class="analysis-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">${escHtml(consultationTitle)}</div>
                <div class="analysis-card-tags">
                    ${analysisPill(analysisConsultationLabel(record.consultation.status, true), analysisTone(record.consultation.status))}
                    ${analysisPill(analysisAdjustmentLabel(record.consultation.confidence_adjustment), Number(record.consultation.confidence_adjustment || 0) > 0 ? 'good' : Number(record.consultation.confidence_adjustment || 0) < 0 ? 'warn' : 'muted')}
                    ${analysisPill(`建议交易：${record.consultation.should_trade === true ? '是' : record.consultation.should_trade === false ? '否' : '未说明'}`, record.consultation.should_trade === true ? 'good' : record.consultation.should_trade === false ? 'bad' : 'muted')}
                </div>
            </div>
            <div class="analysis-card-text">
                ${analysisText(record.consultation.conflict_note || record.consultation.reason || '行情方向专家完成深度会诊。')}
                ${consultationAttempts}
            </div>
        </div>  
    ` : '<div class="analysis-empty">没有重大矛盾，不需要深度会诊</div>';  
    const conflictResolution = record.conflict_resolution || {};
    const resolutionItems = Array.isArray(conflictResolution.items) ? conflictResolution.items : [];
    const resolutionHtml = `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">分歧怎么处理</div>
                <div class="analysis-card-tags">
                    ${analysisPill(`调整 ${analysisAdjustmentLabel((Number(conflictResolution.validation_adjustment || 0) * 100).toFixed(0))}`, Number(conflictResolution.validation_adjustment || 0) < 0 ? 'warn' : 'muted')}
                    ${analysisPill(conflictResolution.consultation_used ? '已会诊' : '规则消化', conflictResolution.consultation_used ? 'good' : 'muted')}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note"><span>处理结果</span>${analysisText(conflictResolution.summary || '没有需要额外处理的分歧，按专家权重和风控阈值裁决。')}</div>
                ${resolutionItems.length ? `
                    <div class="analysis-resolution-list">
                        ${resolutionItems.map(item => `
                            <div class="analysis-resolution-item">
                                <strong>${(item.expert_pair || []).map(name => escHtml(analysisExpertDisplayName(name, experts))).join(' / ') || '-'}</strong>
                                <span>${analysisText(item.resolution || item.validation_note || '-')}</span>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        </div>`;
    const decisionMaker = record.decision_maker || null;
    const decisionMakerHtml = decisionMaker ? `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">最终交易员</div>
                <div class="analysis-card-tags">
                    ${analysisPill(decisionMaker.status === 'completed' ? '已裁决' : decisionMaker.status === 'skipped' ? '已跳过' : '未完成', decisionMaker.status === 'completed' ? 'good' : decisionMaker.status === 'skipped' ? 'muted' : 'warn')}
                    ${decisionMaker.action ? analysisPill(analysisActionLabel(decisionMaker.action, record), analysisTone(decisionMaker.action)) : ''}
                    ${decisionMaker.confidence !== undefined ? analysisPill(`信心 ${(Number(decisionMaker.confidence || 0) * 100).toFixed(0)}%`, Number(decisionMaker.confidence || 0) >= 0.6 ? 'good' : 'muted') : ''}
                    ${decisionMaker.applied === true ? analysisPill('已采用', 'good') : decisionMaker.applied === false ? analysisPill('未采用', 'warn') : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note"><span>裁决说明</span>${analysisText(decisionMaker.reasoning || decisionMaker.reason || decisionMaker.guard_reason || '-')}</div>
                ${decisionMaker.provider_model ? `<div class="analysis-note analysis-note-muted"><span>模型</span>${analysisText(decisionMaker.provider_model)}</div>` : ''}
            </div>
        </div>
    ` : '<div class="analysis-empty">本轮未启用最终交易员</div>';

    const latencySummary = record.latency_summary || {};
    const timingBreakdown = Array.isArray(record.timing_breakdown) ? record.timing_breakdown : [];
    const modelTimings = Array.isArray(record.model_timings) ? record.model_timings : [];
    const finalModelTimings = analysisFinalModelTimings(modelTimings);
    const sharedBatchCalls = analysisSharedBatchCalls(modelTimings);
    const sharedBatchTimings = modelTimings.filter(item => item && (item.shared_batch_call || item.batch_expert));
    const sharedBatchFallbackTotal = sharedBatchCalls.reduce((sum, item) => sum + Number(item.duration_sec || 0), 0);
    const sharedBatchCallCount = Number(
        (latencySummary && latencySummary.shared_batch_call_count)
        || sharedBatchCalls.length
        || 0
    );
    const sharedBatchExpertCount = sharedBatchTimings.length;
    const sharedBatchDuration = Number(
        (latencySummary && (latencySummary.shared_batch_total_duration_sec || latencySummary.shared_batch_duration_sec))
        || sharedBatchFallbackTotal
    );
    const sharedBatchCount = sharedBatchCallCount;
    const stageTimingHtml = timingBreakdown.length ? `
        <div class="analysis-resolution-list">
            ${timingBreakdown.map(item => `
                <div class="analysis-resolution-item">
                    <strong>${escHtml(item.label || analysisStageLabel(item.stage))}</strong>
                    <span>
                        ${analysisDurationLabel(item.duration_sec)}
                        · ${escHtml(analysisTimingStatusLabel(item.status))}
                        ${item.slowest_model ? ` · 最慢专家 ${escHtml(analysisExpertDisplayName(item.slowest_model, experts))}` : ''}
                    </span>
                </div>
            `).join('')}
        </div>
    ` : '<div class="analysis-empty">本轮还没有分阶段耗时记录</div>';
    const sharedBatchCallRows = sharedBatchCalls.map(call => {
        const expertsText = call.expert_names
            .map(name => analysisExpertDisplayName(name, experts))
            .join('、');
        return `
            <div class="analysis-resolution-item">
                <strong>${escHtml(call.provider_model || '批量专家请求')}</strong>
                <span>
                    真实墙钟 ${analysisDurationLabel(call.duration_sec)}
                    · 覆盖 ${call.expert_names.length} 个专家${expertsText ? `：${escHtml(expertsText)}` : ''}
                </span>
            </div>`;
    }).join('');
    const finalTimingRows = finalModelTimings.map(item => `
        <div class="analysis-resolution-item">
            <strong>${escHtml(analysisExpertDisplayName(item.name, experts))}</strong>
            <span>${analysisModelTimingText(item)}</span>
        </div>
    `).join('');
    const modelTimingHtml = modelTimings.length ? `
        <div class="analysis-resolution-list">
            ${sharedBatchCount ? `
                <div class="analysis-resolution-item">
                    <strong>批量请求汇总</strong>
                    <span>
                        真实墙钟 ${analysisDurationLabel(sharedBatchDuration)}
                        · ${sharedBatchCallCount} 次模型调用覆盖 ${sharedBatchExpertCount} 个专家
                    </span>
                </div>
            ` : ''}
            ${sharedBatchCallRows}
            ${finalTimingRows}
        </div>
    ` : '<div class="analysis-empty">本轮还没有单专家耗时记录</div>';
    const timingHtml = `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">耗时拆解</div>
                <div class="analysis-card-tags">
                    ${analysisPill(`专家流程 ${analysisDurationLabel(totalDuration)}`, totalDuration > 30 ? 'warn' : 'muted')}
                    ${endToEndDuration && endToEndDuration > totalDuration + 3 ? analysisPill(`全流程 ${analysisDurationLabel(endToEndDuration)}`, endToEndDuration > 60 ? 'warn' : 'muted') : ''}
                    ${sharedBatchCount ? analysisPill(`专家批量 ${analysisDurationLabel(sharedBatchDuration)}`, sharedBatchDuration > 25 ? 'warn' : 'muted') : ''}
                    ${latencySummary.slowest_model ? analysisPill(`最慢 ${analysisExpertDisplayName(latencySummary.slowest_model.name, experts)}`, Number(latencySummary.slowest_model.duration_sec || 0) > 25 ? 'warn' : 'muted') : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note analysis-note-muted"><span>流程耗时</span>${stageTimingHtml}</div>
                ${endToEndDuration && endToEndDuration > totalDuration + 3 ? '<div class="analysis-note analysis-note-muted"><span>全流程说明</span>全流程包含调度、行情/持仓同步和写库时间；专家流程只统计模型协作阶段，避免把外部等待误算成专家耗时。</div>' : ''}
                <div class="analysis-note analysis-note-muted"><span>${sharedBatchCount ? '专家批量耗时' : '专家耗时'}</span>${modelTimingHtml}</div>
                ${sharedBatchCount ? '<div class="analysis-note analysis-note-muted"><span>耗时说明</span>批量专家按模型服务分组共享请求，同一批专家显示的是同一个共享墙钟耗时；不能把每个专家行的耗时重复相加。</div>' : ''}
            </div>
        </div>
    `;
    const highRiskReview = attribution?.high_risk_review || {};
    const highRiskStatusLabel = (() => {
        if (!highRiskReview.triggered) return '未触发';
        if (highRiskReview.approved === false) return '否决';
        if (highRiskReview.status === 'completed') return '通过';
        if (highRiskReview.status === 'error_blocked') return '调用失败，已拦截';
        if (highRiskReview.status === 'error') return highRiskReview.approved === false ? '调用失败，已拦截' : '调用失败';
        if (highRiskReview.status === 'skipped') return '配置不完整，已放行';
        return highRiskReview.status || '已触发';
    })();
    const highRiskTone = highRiskReview.approved === false
        ? 'bad'
        : (highRiskReview.status === 'error' || highRiskReview.status === 'skipped' ? 'warn' : 'good');
    const highRiskDetail = highRiskReview.triggered
        ? [
            `模型 ${highRiskReview.model || '-'}`,
            `状态 ${highRiskStatusLabel}`,
            highRiskReview.confidence !== undefined ? `复核信心 ${(Number(highRiskReview.confidence || 0) * 100).toFixed(0)}%` : '',
            `触发原因 ${(highRiskReview.reasons || []).join('、') || '-'}`,
            highRiskReview.reason ? `说明 ${highRiskReview.reason}` : '',
        ].filter(Boolean).join(' / ')
        : `未触发：只有开仓决策命中高杠杆、大仓位、专家分歧、ML 与 AI 冲突、近期该币方向亏损、本地量化谨慎或今日亏损恢复开仓时，才会调用线上高风险复核。`;
    const closeEvidence = attribution?.close_evidence || {};
    const lossRepair = closeEvidence?.loss_repair_evidence || {};
    const lossRepairDetail = lossRepair.enabled
        ? [
            lossRepair.repair_possible ? '有由亏转盈证据' : '修复证据不足',
            lossRepair.likely_expanding_loss ? '扩亏风险偏高' : '暂未判定扩亏',
            `修复分 ${Number(lossRepair.repair_score || 0)}`,
            `扩亏分 ${Number(lossRepair.expansion_score || 0)}`,
            `服务器亏损概率 ${pctLabel(lossRepair.local_loss_probability)}`,
            lossRepair.reason ? `结论：${lossRepair.reason}` : '',
        ].filter(Boolean).join(' / ')
        : (
            closeEvidence.position_loss
                ? '当前是亏损持仓，但后端未返回亏损修复评估；请等待下一轮持仓分析刷新。'
                : '当前不是亏损持仓，或本条不是持仓分析记录。'
        );

    const attributionHtml = attribution ? `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">决策归因</div>
                <div class="analysis-card-tags">
                    ${analysisPill(attribution.side_label || analysisActionLabel(record.final_action, record), analysisTone(record.final_action))}
                    ${analysisPill(attribution.executed ? '已执行' : '未执行', attribution.executed ? 'good' : 'warn')}
                    ${highRiskReview.triggered ? analysisPill(`高风险复核 ${highRiskStatusLabel}`, highRiskTone) : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-resolution-list">
                    <div class="analysis-resolution-item"><strong>AI 专家</strong><span>${analysisText(attribution.ai_experts?.summary || '-')}</span></div>
                    <div class="analysis-resolution-item"><strong>本地 ML</strong><span>${escHtml(attribution.local_ml?.available ? `${attribution.local_ml.side_label || '-'} / 预期 ${signedPctValueLabel(attribution.local_ml.expected_return_pct)} / 收益差 ${signedPctValueLabel(attribution.local_ml.profit_edge_pct)}` : '无可用预测')}</span></div>
                    <div class="analysis-resolution-item"><strong>服务器盈利模型</strong><span>${escHtml(attribution.server_profit?.available ? `${attribution.server_profit.side_label || '-'} / 预期 ${signedPctValueLabel(attribution.server_profit.expected_return_pct)} / 亏损概率 ${pctLabel(attribution.server_profit.loss_probability)}` : '无可用预测')}</span></div>
                    <div class="analysis-resolution-item"><strong>时序预测</strong><span>${escHtml(attribution.timeseries?.available ? `${attribution.timeseries.side_label || '-'} / 预期 ${signedPctValueLabel(attribution.timeseries.expected_return_pct)}` : '无可用预测')}</span></div>
                    <div class="analysis-resolution-item"><strong>情绪预测</strong><span>${escHtml(attribution.sentiment?.available ? `${attribution.sentiment.side_label || '-'} / 情绪分 ${Number(attribution.sentiment.score || 0).toFixed(3)}` : '无可用预测')}</span></div>
                    <div class="analysis-resolution-item"><strong>亏损修复评估</strong><span>${escHtml(lossRepairDetail)}</span></div>
                    <div class="analysis-resolution-item"><strong>机会评分</strong><span>${escHtml(attribution.opportunity_score ? `总分 ${Number(attribution.opportunity_score.score || 0).toFixed(4)} / 门槛 ${Number(attribution.opportunity_score.min_score_required || 0).toFixed(2)} / 净收益 ${signedPctValueLabel(attribution.opportunity_score.expected_net_return_pct)}` : '无评分')}</span></div>
                    <div class="analysis-resolution-item"><strong>高风险复核模型</strong><span>${escHtml(highRiskDetail)}</span></div>
                    <div class="analysis-resolution-item"><strong>最终原因</strong><span>${analysisText(attribution.final_reason || '-')}</span></div>
                </div>
            </div>
        </div>
    ` : '';
 
    document.getElementById('decision-reason-title').textContent =  
        `${record.symbol || '-'} / 专家协作流程`;  
    document.getElementById('decision-reason-body').innerHTML = ` 
        <div class="analysis-flow">
            <div class="analysis-summary">
                ${analysisMetric('专家返回', isFastScan ? '快速扫描' : `${expertCount}/${expectedCount}`, isFastScan || expertCount === expectedCount ? 'good' : 'warn')}
                ${mlSignal?.available ? analysisMetric('ML盈亏', `预期 ${signedPctValueLabel(mlSignal.expected_return_pct)}`, Number(mlSignal.expected_return_pct || 0) > 0 ? 'good' : 'warn') : ''}
                ${analysisMetric('交叉验证', `${completedCross}/${Number(record.cross_requested || 0)}`, unavailableCross ? 'warn' : 'good')}
                ${analysisMetric('分析耗时', analysisDurationLabel(totalDuration), totalDuration > 60 ? 'warn' : 'muted')}
                ${analysisMetric('最终方向', analysisActionLabel(record.final_action, record), analysisTone(record.final_action))}
                ${lifecycleLabel ? analysisMetric('持仓状态', lifecycleLabel, analysisPositionLifecycleTone(record)) : ''}
                ${analysisMetric('分析信心 / 仓位', `${finalConfidence} / ${positionSize}`, Number(record.final_confidence || 0) >= 0.6 ? 'good' : 'muted')}
            </div>

            ${attributionHtml ? analysisSection('决策归因面板', attributionHtml) : ''}
            ${analysisSection('Agent/Skills 守门', renderAnalysisAgentSkills(agentSkills))}
            ${analysisSection('本地ML盈亏质量', renderAnalysisMlSignal(mlSignal))}
            ${analysisSection('服务器量化工具', renderAnalysisLocalAiTools(localAiTools, record.analysis_type))}
            ${analysisSection('新闻与事件', renderAnalysisNewsContext(newsContext))}
            ${analysisSection(isFastScan ? '持仓快速扫描' : '专家初诊', `<div class="analysis-grid">${expertsHtml || '<div class="analysis-empty">无返回结果</div>'}</div>`, expertSectionSubtitle)}
            ${missingHtml ? analysisSection('未返回专家', `<div class="analysis-grid">${missingHtml}</div>`) : ''}
            ${analysisSection('交叉验证', `<div class="analysis-grid">${pairValidations || '<div class="analysis-empty">没有触发交叉验证</div>'}</div>`, `请求 ${Number(record.cross_requested || 0)} 个，完成 ${completedCross} 个，无法验证 ${unavailableCross} 个，重大矛盾 ${majorConflicts} 个`)}
            ${analysisSection('深度会诊', consultation)}
            ${analysisSection('分歧处理', resolutionHtml)}
            ${analysisSection('最终交易员', decisionMakerHtml)}
            ${analysisSection('耗时记录', timingHtml)}
            ${analysisSection('最终裁决', `
                <div class="analysis-card analysis-final-card">
                    <div class="analysis-card-head">
                        <div class="analysis-card-title">${analysisActionLabel(record.final_action, record)}</div>
                        <div class="analysis-card-tags">
                            ${analysisPill(`分析信心 ${finalConfidence}`, Number(record.final_confidence || 0) >= 0.6 ? 'good' : 'muted')}
                            ${analysisPill(`下单信心 ${tradeConfidence}`, Number(record.trade_confidence || 0) >= 0.6 ? 'good' : 'muted')}
                            ${analysisPill(`仓位 ${positionSize}`, Number(record.position_size_pct || 0) > 0 ? 'good' : 'muted')}
                            ${analysisPill(record.was_executed ? '已执行' : '未执行', record.was_executed ? 'good' : 'warn')}
                        </div>
                    </div>
                    <div class="analysis-card-text">
                        <div class="analysis-final-metrics">
                            <span>综合分：${escHtml(record.weighted_score ?? '-')}</span>
                            <span>分歧度：${escHtml(record.disagreement ?? '-')}</span>
                        </div>
                        ${analysisOpportunityScoreHtml(record.opportunity_score, record)}
                        ${record.execution_reason ? `<div class="analysis-note"><span>未执行原因</span>${analysisText(record.execution_reason)}</div>` : ''}
                        ${record.confidence_note ? `<div class="analysis-note analysis-note-muted"><span>信心说明</span>${analysisText(record.confidence_note)}</div>` : ''}
                        <div class="analysis-note analysis-note-muted"><span>裁决理由</span>${analysisText(record.final_reasoning || '-')}</div>
                    </div>
                </div>
            `)}
        </div>
    `; 
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex'; 
} 

function changeAnalysisPage(page) {
    state.analysisPage = page;
    fetchAnalysisRecords();
}

// ========== Expert Long-term Memory ==========

async function fetchExpertMemories() {
    const params = new URLSearchParams({
        page_size: EXPERT_MEMORY_PAGE_SIZE,
        memory_page: state.expertMemoryPage,
        reflection_page: state.tradeReflectionPage,
    });
    const data = await fetchJSON(`/api/expert-memories?${params.toString()}`);
    if (!data) return;
    state.expertMemories = data.memories || [];
    state.tradeReflections = data.reflections || [];
    state.expertMemoryTotal = Number(data.count || 0);
    state.tradeReflectionTotal = Number(data.reflection_count || 0);
    renderExpertMemories(data);
}

function renderExpertMemories(data = {}) {
    const memories = state.expertMemories || [];
    const reflections = state.tradeReflections || [];
    const pagination = data.pagination || {};
    const countEl = document.getElementById('expert-memory-count');
    const reflectionCountEl = document.getElementById('trade-reflection-count');
    const memoryBody = document.getElementById('expert-memory-tbody');
    const reflectionBody = document.getElementById('trade-reflection-tbody');
    const memoryTotal = Number(pagination.memory_total ?? state.expertMemoryTotal ?? memories.length);
    const reflectionTotal = Number(pagination.reflection_total ?? state.tradeReflectionTotal ?? reflections.length);
    const memoryPage = Number(pagination.memory_page || state.expertMemoryPage || 1);
    const reflectionPage = Number(pagination.reflection_page || state.tradeReflectionPage || 1);
    const memoryTotalPages = Number(pagination.memory_total_pages || Math.max(Math.ceil(memoryTotal / EXPERT_MEMORY_PAGE_SIZE), 1));
    const reflectionTotalPages = Number(pagination.reflection_total_pages || Math.max(Math.ceil(reflectionTotal / EXPERT_MEMORY_PAGE_SIZE), 1));
    if (countEl) countEl.textContent = `${memoryTotal} 条`;
    if (reflectionCountEl) reflectionCountEl.textContent = `${reflectionTotal} 条`;
    setExpertMemoryView(state.expertMemoryView || 'memories');

    if (memoryBody) {
        if (!memories.length) {
            memoryBody.innerHTML = '<tr><td colspan="8" style="color:var(--text-muted);text-align:center;padding:24px;">暂无专家记忆，平仓后会自动生成复盘经验。</td></tr>';
        } else {
            memoryBody.innerHTML = memories.map(m => {
                const adjustment = valueNumber(m.confidence_adjustment) || 0;
                const multiplier = valueNumber(m.position_size_multiplier) || 1;
                const adjColor = adjustment >= 0 ? 'var(--green)' : 'var(--red)';
                return `
                    <tr>
                        <td>${escHtml(m.expert_label || m.expert_name || '-')}</td>
                        <td>${escHtml(m.symbol || '通用')}</td>
                        <td>${memoryTypeLabel(m.memory_type)}</td>
                        <td style="max-width:180px;">${escHtml(m.market_pattern || '-')}</td>
                        <td style="max-width:360px;">${escHtml(m.lesson || '-')}</td>
                        <td>${memoryActionLabel(m.recommended_action)}</td>
                        <td>${Number(m.evidence_count || 0)} / ${Number(m.hit_count || 0)}</td>
                        <td style="color:${adjColor};white-space:nowrap;">${adjustment >= 0 ? '+' : ''}${(adjustment * 100).toFixed(1)}% · ${(multiplier * 100).toFixed(0)}%</td>
                    </tr>
                `;
            }).join('');
        }
        renderPagination('expert-memory-pagination', memoryPage, memoryTotalPages, memoryTotal, 'changeExpertMemoryPage');
    }

    if (reflectionBody) {
        if (!reflections.length) {
            reflectionBody.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center;padding:24px;">暂无复盘记录。</td></tr>';
        } else {
            reflectionBody.innerHTML = reflections.map(r => {
                const pnl = valueNumber(r.realized_pnl) || 0;
                const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
                const generatedTime = tradeReflectionTimeHtml(r.created_at);
                const generatedTimeTitle = toBeijingDateTime(r.created_at);
                return `
                    <tr>
                        <td class="trade-reflection-time" title="${escHtml(generatedTimeTitle)}">${generatedTime}</td>
                        <td>${escHtml(r.symbol || '-')}</td>
                        <td>${sideLabel(r.side)}</td>
                        <td style="color:${pnlColor};white-space:nowrap;">${signedMoney(pnl)} USDT</td>
                        <td>${Number(r.hold_minutes || 0).toFixed(1)} 分钟</td>
                        <td><div class="trade-reflection-text">${escHtml(r.mistake_summary || '-')}</div></td>
                        <td><div class="trade-reflection-text">${escHtml(r.improvement_summary || '-')}</div></td>
                    </tr>
                `;
            }).join('');
        }
        renderPagination('trade-reflection-pagination', reflectionPage, reflectionTotalPages, reflectionTotal, 'changeTradeReflectionPage');
    }
}

function changeExpertMemoryPage(page) {
    state.expertMemoryPage = Math.max(1, Number(page) || 1);
    fetchExpertMemories();
}

function changeTradeReflectionPage(page) {
    state.tradeReflectionPage = Math.max(1, Number(page) || 1);
    fetchExpertMemories();
}

function setExpertMemoryView(view) {
    const selected = view === 'reflections' ? 'reflections' : 'memories';
    state.expertMemoryView = selected;
    document.querySelectorAll('#expert-memory-tabs .trade-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.expertMemoryView === selected);
    });
    document.getElementById('expert-memory-panel-memories')?.classList.toggle('active', selected === 'memories');
    document.getElementById('expert-memory-panel-reflections')?.classList.toggle('active', selected === 'reflections');
}

// ========== Shadow Backtest ==========

async function fetchShadowBacktests() {
    const params = new URLSearchParams({
        page_size: EXPERT_MEMORY_PAGE_SIZE,
        page: state.shadowBacktestPage,
    });
    const status = state.shadowBacktestStatus || document.getElementById('shadow-backtest-status')?.value || '';
    if (status) params.set('status', status);
    const data = await fetchJSON(`/api/shadow-backtests?${params.toString()}`);
    if (!data) return;
    state.shadowBacktests = data.records || [];
    state.shadowBacktestTotal = Number(data.count || 0);
    renderShadowBacktests(data);
}

function renderShadowBacktests(data = {}) {
    const rows = state.shadowBacktests || [];
    const pagination = data.pagination || {};
    const total = Number(pagination.total ?? state.shadowBacktestTotal ?? rows.length);
    const page = Number(pagination.page || state.shadowBacktestPage || 1);
    const totalPages = Number(pagination.total_pages || Math.max(Math.ceil(total / EXPERT_MEMORY_PAGE_SIZE), 1));
    const body = document.getElementById('shadow-backtest-tbody');
    const countEl = document.getElementById('shadow-backtest-count');
    if (countEl) {
        countEl.textContent = `${total} 条（已完成 ${Number(data.completed_count || 0)}，等待 ${Number(data.pending_count || 0)}）`;
    }
    if (!body) return;
    if (!rows.length) {
        body.innerHTML = '<tr><td colspan="12" style="color:var(--text-muted);text-align:center;padding:24px;">暂无影子复盘数据</td></tr>';
    } else {
        body.innerHTML = rows.map(r => {
            const longRet = (r.long_return_pct === null || r.long_return_pct === undefined) ? null : valueNumber(r.long_return_pct);
            const shortRet = (r.short_return_pct === null || r.short_return_pct === undefined) ? null : valueNumber(r.short_return_pct);
            const longColor = longRet === null ? 'var(--text-muted)' : longRet >= 0 ? 'var(--green)' : 'var(--red)';
            const shortColor = shortRet === null ? 'var(--text-muted)' : shortRet >= 0 ? 'var(--green)' : 'var(--red)';
            const best = r.best_action || 'hold';
            const statusColor = r.status === 'completed' ? 'var(--green)' : 'var(--accent-light)';
            const conclusion = String(r.conclusion || '');
            const conclusionColor = r.missed_opportunity ? 'var(--accent-light)' : conclusion.includes('有效') ? 'var(--green)' : conclusion.includes('偏差') ? 'var(--red)' : 'var(--text)';
            const decisionNote = shadowDecisionNote(r);
            return `
                <tr>
                    <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(r.created_at)}</td>
                    <td>${escHtml(r.symbol || '-')}</td>
                    <td><span class="badge badge-${r.decision_action || 'hold'}">${escHtml(r.decision_action_label || actionLabel(r.decision_action))}</span><div style="font-size:10px;color:var(--text-muted);margin-top:4px;">${Math.round(Number(r.decision_confidence || 0) * 100)}%</div>${decisionNote ? `<div style="font-size:10px;color:var(--text-muted);line-height:1.45;margin-top:3px;">${escHtml(decisionNote)}</div>` : ''}</td>
                    <td>${Number(r.horizon_minutes || 0)} 分钟</td>
                    <td>${fmtPrice(r.entry_price)}</td>
                    <td>${r.actual_price ? fmtPrice(r.actual_price) : '-'}</td>
                    <td style="color:${longColor};white-space:nowrap;">${shadowReturnText(longRet)}</td>
                    <td style="color:${shortColor};white-space:nowrap;">${shadowReturnText(shortRet)}</td>
                    <td><span class="badge badge-${best}">${escHtml(r.best_action_label || actionLabel(best))}</span></td>
                    <td style="color:${conclusionColor};max-width:180px;">${escHtml(r.conclusion || '-')}</td>
                    <td style="color:${statusColor};white-space:nowrap;">${escHtml(r.status_label || r.status || '-')}</td>
                    <td><button class="btn btn-sm" onclick="showShadowBacktestDetail(${Number(r.id)})">查看</button></td>
                </tr>
            `;
        }).join('');
    }
    renderPagination('shadow-backtest-pagination', page, totalPages, total, 'changeShadowBacktestPage');
}

function shadowDecisionNote(row) {
    const action = String(row?.decision_action || '').toLowerCase();
    const confidence = Number(row?.decision_confidence || 0);
    const note = String(row?.decision_note || '').trim();
    if (note) return note;
    if (action === 'hold' && confidence <= 0) {
        return '当时没有形成可执行开仓信号。';
    }
    return '';
}

function changeShadowBacktestPage(page) {
    state.shadowBacktestPage = Math.max(1, Number(page) || 1);
    fetchShadowBacktests();
}

function changeShadowBacktestStatus() {
    state.shadowBacktestStatus = document.getElementById('shadow-backtest-status')?.value || '';
    state.shadowBacktestPage = 1;
    fetchShadowBacktests();
}

function shadowReturnText(value) {
    if (value === null || value === undefined) return '-';
    const n = valueNumber(value);
    if (n === null) return '-';
    return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
}

function showShadowBacktestDetail(id) {
    const row = (state.shadowBacktests || []).find(r => Number(r.id) === Number(id));
    if (!row) return;
    setDecisionModalWide(false);
    const decisionNote = shadowDecisionNote(row);
    document.getElementById('decision-reason-title').textContent = `${row.symbol || '-'} / 影子复盘`;
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="reason-block">
            <div class="reason-label">复盘结论</div>
            <div>${escapeMultiline(row.conclusion || '-')}</div>
        </div>
        <div class="reason-block">
            <div class="reason-label">当时决策</div>
            <div>${escHtml(row.decision_action_label || actionLabel(row.decision_action))}，信心度 ${Math.round(Number(row.decision_confidence || 0) * 100)}%，周期 ${Number(row.horizon_minutes || 0)} 分钟</div>
            ${decisionNote ? `<div class="reason-meta">${escHtml(decisionNote)}</div>` : ''}
            <div class="reason-meta">入场价：${fmtPrice(row.entry_price)}<br>结果价：${row.actual_price ? fmtPrice(row.actual_price) : '等待结果'}<br>做多收益：${shadowReturnText(row.long_return_pct)}<br>做空收益：${shadowReturnText(row.short_return_pct)}<br>最优方向：${escHtml(row.best_action_label || actionLabel(row.best_action))}</div>
        </div>
        <div class="reason-block">
            <div class="reason-label">说明</div>
            <div>${escapeMultiline(row.note || (row.status === 'pending' ? '还没到复盘时间。' : '暂无额外说明。'))}</div>
        </div>
    `;
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex';
}

function memoryTypeLabel(type) {
    const map = {
        loss_lesson: '亏损教训',
        profit_pattern: '盈利经验',
        flat_lesson: '打平复盘',
        shadow_missed_opportunity: '影子复盘-错过机会',
        shadow_bad_signal: '影子复盘-错误信号',
        shadow_good_signal: '影子复盘-有效信号',
        lesson: '经验',
    };
    return map[type] || type || '-';
}

function memoryActionLabel(action) {
    const map = {
        reduce_risk: '降信心/降仓位',
        keep_with_filters: '保留但需过滤',
        wait_for_better_setup: '等待更好机会',
        allow_small_probe_with_filters: '允许小仓位试探',
    };
    return map[action] || action || '-';
}

// ========== Local ML Signal Dashboard ==========

async function fetchMLSignalDashboard() {
    const [status, localToolsStatus, recordsData, contributionData] = await Promise.all([
        fetchJSON('/api/ml-signal/status'),
        fetchJSON('/api/local-ai-tools/status').catch(() => null),
        fetchJSON(`/api/analysis-records?limit=120&is_paper=${state.mode === 'paper' ? 'true' : 'false'}`),
        fetchJSON(`/api/model-contribution/stats?mode=${state.mode === 'live' ? 'live' : 'paper'}&days=7`).catch(() => null),
    ]);
    state.mlSignalStatus = status || null;
    state.localAIToolsStatus = localToolsStatus || null;
    state.modelContributionStats = contributionData || null;
    state.mlSignalRecords = (recordsData?.records || []).filter(r => r && r.ml_signal && r.ml_signal.available !== false);
    const totalPages = Math.max(Math.ceil(state.mlSignalRecords.length / ML_SIGNAL_PAGE_SIZE), 1);
    state.mlSignalPage = Math.min(Math.max(Number(state.mlSignalPage || 1), 1), totalPages);
    renderMLSignalDashboard();
}

function renderMLSignalDashboard() {
    renderLocalAIToolsStatus();
    renderMLSignalOverview();
    renderMLSignalMetrics();
    renderModelContributionStats();
    renderTrainableModels();
    renderMLSignalRecent();
}


function mlPrimaryPrediction(signal) {
    if (!signal) return null;
    const predictions = Array.isArray(signal.predictions) ? signal.predictions : [];
    if (!predictions.length) return null;
    const primary = Number(signal.primary_horizon_minutes || 0);
    return predictions.find(p => Number(p.horizon_minutes || 0) === primary) || predictions[0];
}

function mlSideLabel(side) {
    const value = String(side || '').toLowerCase();
    if (value === 'long') return '做多';
    if (value === 'short') return '做空';
    return '中性';
}

function mlSignalToneByRate(rate) {
    const value = Number(rate || 0);
    if (value >= 0.62) return 'good';
    if (value >= 0.55) return 'warn';
    return 'muted';
}

function mlMetricTone(value, good = 0.6, warn = 0.55) {
    const num = Number(value);
    if (!Number.isFinite(num)) return 'muted';
    if (num >= good) return 'good';
    if (num >= warn) return 'warn';
    return 'bad';
}

function mlMetricCard(label, value, subtitle = '', tone = 'muted') {
    return `
        <div class="ml-metric ml-metric-${tone}">
            <div class="ml-metric-label">${escHtml(label)}</div>
            <div class="ml-metric-value">${escHtml(value)}</div>
            ${subtitle ? `<div class="ml-metric-subtitle">${escHtml(subtitle)}</div>` : ''}
        </div>`;
}

function mlModelStatusPill(isReady, label = '') {
    const text = label || (isReady ? '已训练' : '未就绪');
    return `<span class="analysis-pill analysis-pill-${isReady ? 'good' : 'warn'}">${escHtml(text)}</span>`;
}

function localModelStatus(status, key) {
    const models = status?.models || {};
    const childEndpoints = status?.child_endpoints || {};
    const endpointByModel = {
        profit: 'profit_prediction',
        timeseries: 'time_series_prediction',
        deep_timeseries: 'time_series_prediction',
        sentiment: 'sentiment_analysis',
        deep_sentiment: 'sentiment_analysis',
        exit: ['exit_advice', 'exit', 'position_exit'],
    };
    const modelAliases = {
        profit: ['profit', 'profit_model', 'entry_profit', 'profit_prediction'],
        timeseries: ['timeseries', 'time_series', 'time_series_prediction'],
        deep_timeseries: ['deep_timeseries', 'timeseries', 'time_series', 'time_series_prediction'],
        sentiment: ['sentiment', 'sentiment_model', 'sentiment_analysis'],
        deep_sentiment: ['deep_sentiment', 'sentiment', 'sentiment_model', 'sentiment_analysis'],
        exit: ['exit', 'exit_advice', 'position_exit'],
    };
    const endpointAliases = Array.isArray(endpointByModel[key])
        ? endpointByModel[key]
        : [endpointByModel[key]];
    const endpoint = endpointAliases.map(name => childEndpoints[name]).find(Boolean);
    const modelReady = (modelAliases[key] || [key]).some(alias => Boolean(models[alias]));
    return Boolean(status?.service_available !== false && (modelReady || endpoint?.available));
}

function mlSampleCounts() {
    const ml = state.mlSignalStatus || {};
    const local = state.localAIToolsStatus || {};
    const autoLast = ml.auto_train_last_result || {};
    const trainingMl = Number(ml.training_shadow_sample_count || ml.sample_count || ml.trained_sample_count || 0);
    const completedMl = Number(
        ml.completed_shadow_sample_count
        || ml.total_shadow_sample_count
        || autoLast.completed_sample_count
        || Math.max(trainingMl, Number(autoLast.previous_sample_count || 0) + Number(autoLast.new_sample_count || 0))
        || trainingMl
    );
    const trainingLocal = Number(local.training_shadow_sample_count || local.shadow_sample_count || 0);
    const completedLocal = Number(
        local.completed_shadow_sample_count
        || local.total_shadow_sample_count
        || completedMl
        || trainingLocal
    );
    const trainingLocalTrade = Number(local.trade_sample_count || 0);
    const completedLocalTrade = Number(
        local.completed_trade_sample_count
        || local.total_trade_sample_count
        || trainingLocalTrade
    );
    const limit = Number(ml.training_shadow_sample_limit || local.training_shadow_sample_limit || 20000);
    const newCount = Number(
        ml.new_shadow_sample_count
        || autoLast.new_sample_count
        || Math.max(completedMl - Number(autoLast.previous_sample_count || trainingMl || 0), 0)
    );
    return {
        trainingMl,
        completedMl,
        trainingLocal,
        completedLocal,
        trainingLocalTrade,
        completedLocalTrade,
        limit,
        newCount,
    };
}

function mlWinBar(label, value, tone = 'muted') {
    const num = Number(value);
    const pct = Number.isFinite(num) ? Math.max(0, Math.min(num * 100, 100)) : 0;
    return `
        <div class="ml-bar-row">
            ${label ? `<div class="ml-bar-label">${escHtml(label)}</div>` : ''}
            <div class="ml-bar-track"><div class="ml-bar-fill ml-bar-${tone}" style="width:${pct.toFixed(1)}%;"></div></div>
            <div class="ml-bar-value">${pctLabel(value)}</div>
        </div>`;
}

function renderMLSignalMetrics() {
    const container = document.getElementById('ml-signal-metrics');
    if (!container) return;
    const status = state.mlSignalStatus || {};
    const metrics = status.metrics || {};
    if (!status.available || !Object.keys(metrics).length) {
        container.innerHTML = '<div class="analysis-empty">暂无可展示的训练指标，请先运行本地 ML 训练。</div>';
        return;
    }

    container.innerHTML = `
        <div class="ml-metrics-grid">
            ${mlMetricCard('做多 AUC', Number(metrics.long_auc || 0).toFixed(3), '越接近 1 越能区分好坏机会', mlMetricTone(metrics.long_auc))}
            ${mlMetricCard('做空 AUC', Number(metrics.short_auc || 0).toFixed(3), '当前做空信号略强于做多', mlMetricTone(metrics.short_auc))}
            ${mlMetricCard('做多准确率', pctLabel(metrics.long_accuracy, 1), '测试集表现', mlMetricTone(metrics.long_accuracy))}
            ${mlMetricCard('做空准确率', pctLabel(metrics.short_accuracy, 1), '测试集表现', mlMetricTone(metrics.short_accuracy))}
        </div>
        <div class="ml-panel">
            <div class="ml-panel-title">分层收益质量</div>
            ${mlMetricCard('做多高分组平均收益', signedPctValueLabel(metrics.top_long_avg_return_pct), '收益越高越说明模型能筛出赚钱机会', Number(metrics.top_long_avg_return_pct || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('做空高分组平均收益', signedPctValueLabel(metrics.top_short_avg_return_pct), '收益越高越说明模型能筛出赚钱机会', Number(metrics.top_short_avg_return_pct || 0) > 0 ? 'good' : 'warn')}
            ${mlWinBar('做多高分组胜率', metrics.top_long_win_rate, mlSignalToneByRate(metrics.top_long_win_rate))}
            ${mlWinBar('做多低分组胜率', metrics.bottom_long_win_rate, 'muted')}
            ${mlWinBar('做空高分组胜率', metrics.top_short_win_rate, mlSignalToneByRate(metrics.top_short_win_rate))}
            ${mlWinBar('做空低分组胜率', metrics.bottom_short_win_rate, 'muted')}
        </div>
        `;
}

function renderModelContributionStats() {
    const container = document.getElementById('model-contribution-stats');
    if (!container) return;
    const data = state.modelContributionStats || {};
    const rows = Array.isArray(data.stats) ? data.stats : [];
    if (!rows.length) {
        container.innerHTML = '<div class="analysis-empty">暂无实盘贡献样本。等有新的已平仓记录后，这里会显示每个模型到底帮你赚了还是亏了。</div>';
        return;
    }
    container.innerHTML = `
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>模型/信号</th>
                        <th>样本</th>
                        <th>真实盈亏</th>
                        <th>胜率</th>
                        <th>平均盈亏</th>
                        <th>盈亏比</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows.map(row => {
                        const pnl = Number(row.pnl || 0);
                        const tone = pnl > 0 ? 'good' : pnl < 0 ? 'bad' : 'muted';
                        return `
                            <tr>
                                <td>${escHtml(row.label || '-')}</td>
                                <td>${Number(row.count || 0)}</td>
                                <td><span class="analysis-pill analysis-pill-${tone}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)} U</span></td>
                                <td>${pctLabel(row.win_rate, 1)}</td>
                                <td>${Number(row.avg_pnl || 0).toFixed(4)} U</td>
                                <td>${Number(row.profit_factor || 0).toFixed(2)}</td>
                            </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
        <div class="analysis-note analysis-note-muted" style="margin:12px;">
            <span>说明</span>${escHtml(data.summary || '按真实已平仓盈亏统计，用于判断哪些模型应该加权，哪些应该降权。')}
        </div>`;
}

function mlDecisionAlignment(record, prediction) {
    const action = String(record?.final_action || '').toLowerCase();
    const best = String(prediction?.best_side || '').toLowerCase();
    if (!prediction || !best) return '暂无预测';
    if (['hold', 'wait', 'none', ''].includes(action)) return 'AI观望，ML未触发开仓';
    if ((action.includes('long') && best === 'long') || (action.includes('short') && best === 'short')) return '方向一致';
    return '方向不一致';
}

function renderMLSignalRecent() {
    const container = document.getElementById('ml-signal-recent');
    const countEl = document.getElementById('ml-signal-recent-count');
    if (!container) return;
    const allRows = state.mlSignalRecords || [];
    const total = allRows.length;
    const totalPages = Math.max(Math.ceil(total / ML_SIGNAL_PAGE_SIZE), 1);
    const page = Math.min(Math.max(Number(state.mlSignalPage || 1), 1), totalPages);
    state.mlSignalPage = page;
    const start = (page - 1) * ML_SIGNAL_PAGE_SIZE;
    const rows = allRows.slice(start, start + ML_SIGNAL_PAGE_SIZE);
    if (countEl) countEl.textContent = `${total} 条`;
    if (!total) {
        container.innerHTML = '<div class="analysis-empty">暂无最近 ML 预测。等待下一轮 AI 分析后会自动出现。</div>';
        renderPagination('ml-signal-pagination', 1, 1, 0, 'changeMLSignalPage');
        return;
    }
    container.innerHTML = `
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>币种</th>
                        <th>AI裁决</th>
                        <th>ML倾向</th>
                        <th>做多预期</th>
                        <th>做空预期</th>
                        <th>预期收益</th>
                        <th>过滤结论</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows.map(record => {
                        const signal = record.ml_signal || {};
                        const pred = mlPrimaryPrediction(signal) || {};
                        const longRate = Number(pred.long_win_rate || 0);
                        const shortRate = Number(pred.short_win_rate || 0);
                        const longExpected = Number(pred.long_expected_return_pct || 0);
                        const shortExpected = Number(pred.short_expected_return_pct || 0);
                        const bestRate = Number(pred.best_win_rate || 0);
                        const tone = Number(pred.best_expected_return_pct || 0) > 0 ? 'good' : Number(pred.risk_score || 0) >= 0.55 ? 'warn' : 'muted';
                        return `
                            <tr>
                                <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(record.created_at)}</td>
                                <td>${escHtml(record.symbol || '-')}</td>
                                <td><span class="badge badge-${record.final_action || 'hold'}">${escHtml(actionLabel(record.final_action))}</span></td>
                                <td><span class="analysis-pill analysis-pill-${tone}">${mlSideLabel(pred.best_side)} ${signedPctValueLabel(pred.best_expected_return_pct)}</span></td>
                                <td>${signedPctValueLabel(longExpected)}<div style="font-size:10px;color:var(--text-muted);">胜率 ${pctLabel(longRate)}</div></td>
                                <td>${signedPctValueLabel(shortExpected)}<div style="font-size:10px;color:var(--text-muted);">胜率 ${pctLabel(shortRate)}</div></td>
                                <td style="white-space:nowrap;">${signedPctValueLabel(pred.best_expected_return_pct)}</td>
                                <td style="max-width:260px;">${escHtml(mlDecisionAlignment(record, pred))}<div style="font-size:10px;color:var(--text-muted);margin-top:4px;">${escHtml(signal.suggestion || signal.note || '盈亏质量过滤')}</div></td>
                            </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>`;
    renderPagination('ml-signal-pagination', page, totalPages, total, 'changeMLSignalPage');
}

function changeMLSignalPage(page) {
    state.mlSignalPage = Math.max(1, Number(page) || 1);
    renderMLSignalRecent();
}

// ========== Server Monitor ==========

async function fetchServerMonitor() {
    const updated = document.getElementById('server-monitor-updated');
    if (updated) updated.textContent = '读取中...';
    const data = await fetchJSON('/api/server-monitor/status');
    state.serverMonitorStatus = data || null;
    renderServerMonitor();
}

async function fetchSystemSelfCheck() {
    const updated = document.getElementById('system-self-check-updated');
    const panel = document.getElementById('system-self-check-panel');
    if (updated) updated.textContent = '自检中...';
    if (panel) panel.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:16px;">正在检查模型、账户、执行链路和最近失败步骤...</div>';
    const data = await fetchJSON('/api/system/self-check');
    state.systemSelfCheck = data || null;
    renderSystemSelfCheck();
}

async function refreshServerMonitorPage() {
    if (serverMonitorRefreshInFlight) return serverMonitorRefreshInFlight;
    serverMonitorRefreshInFlight = (async () => {
        const results = await Promise.allSettled([
            fetchServerMonitor(),
            fetchSystemSelfCheck(),
        ]);
        const [monitorResult, selfCheckResult] = results;
        if (monitorResult.status === 'rejected') {
            const updated = document.getElementById('server-monitor-updated');
            const panel = document.getElementById('server-monitor-model-runtime');
            const message = monitorResult.reason?.message || String(monitorResult.reason || '监控渲染失败');
            if (updated) updated.textContent = '读取失败';
            if (panel) {
                panel.innerHTML = `<div style="color:var(--red);font-size:12px;padding:16px;">大模型服务器监控渲染失败：${escHtml(message)}</div>`;
            }
            console.error('刷新大模型服务器监控失败', monitorResult.reason);
        }
        if (selfCheckResult.status === 'rejected') {
            const updated = document.getElementById('system-self-check-updated');
            const panel = document.getElementById('system-self-check-panel');
            const message = selfCheckResult.reason?.message || String(selfCheckResult.reason || '系统自检失败');
            if (updated) updated.textContent = '自检失败';
            if (panel) {
                panel.innerHTML = `<div style="color:var(--red);font-size:12px;padding:16px;">系统自检失败：${escHtml(message)}</div>`;
            }
            console.error('刷新系统自检失败', selfCheckResult.reason);
        }
    })().finally(() => {
        serverMonitorRefreshInFlight = null;
    });
    return serverMonitorRefreshInFlight;
}

async function repairSystemSelfCheck() {
    const updated = document.getElementById('system-self-check-updated');
    if (updated) updated.textContent = '安全修复中...';
    try {
        const data = await postJSON('/api/system/self-check/repair', {});
        const actions = (data.actions || []).map(item => `${item.action}: ${item.status}`).join('；');
        alert(`安全修复已执行：${actions || '无可执行动作'}。将重新自检。`);
        await refreshServerMonitorPage();
    } catch (error) {
        alert(`安全修复失败：${error.message || error}`);
        renderSystemSelfCheck();
    }
}

function selfCheckStatusLabel(status) {
    const labels = { ok: '正常', warning: '需关注', critical: '异常', info: '提示' };
    return labels[status] || status || '-';
}

function selfCheckStatusGroupTitle(status, count) {
    const titles = {
        critical: '\u5f02\u5e38\u95ee\u9898',
        warning: '\u9700\u5173\u6ce8\u9879',
        info: '\u8fd0\u884c\u63d0\u793a',
        ok: '\u6b63\u5e38\u9879',
    };
    return `${titles[status] || selfCheckStatusLabel(status)} \u00b7 ${Number(count || 0)} \u9879`;
}

function selfCheckStatusRank(status) {
    return { critical: 0, warning: 1, info: 2, ok: 3 }[status] ?? 4;
}

function selfCheckGroupedItems(items) {
    const groups = { critical: [], warning: [], info: [], ok: [] };
    items.forEach(item => {
        const status = String(item?.status || 'info');
        if (!groups[status]) groups[status] = [];
        groups[status].push(item);
    });
    return Object.entries(groups)
        .filter(([, rows]) => rows.length)
        .sort(([left], [right]) => selfCheckStatusRank(left) - selfCheckStatusRank(right));
}

function selfCheckItemHtml(item) {
    const detailText = selfCheckDetailText(item.details);
    return `
        <div class="self-check-card ${escHtml(item.status || 'info')}">
            <div class="self-check-title">
                <span>${escHtml(item.title || item.key || '-')}</span>
                <strong>${escHtml(selfCheckStatusLabel(item.status))}</strong>
            </div>
            <div class="self-check-message">${escHtml(item.message || '-')}</div>
            ${detailText ? `<div class="self-check-details">${escHtml(detailText)}</div>` : ''}
            ${item.repairable ? '<div class="self-check-repair-note">\u53ef\u6267\u884c\u5b89\u5168\u4fee\u590d\uff1a\u6e05\u7f13\u5b58 / \u91cd\u7f6e\u7194\u65ad\uff0c\u4e0d\u4f1a\u6539\u8d44\u91d1\u548c\u8ba2\u5355\u3002</div>' : ''}
        </div>`;
}

function selfCheckDetailText(details) {
    if (!details || typeof details !== 'object' || !Object.keys(details).length) return '';
    const lines = [];
    Object.entries(details).forEach(([key, value]) => {
        if (value === null || value === undefined || value === '') return;
        if (typeof value === 'object') {
            lines.push(`${key}: ${JSON.stringify(value)}`);
        } else {
            lines.push(`${key}: ${value}`);
        }
    });
    return lines.join('\n');
}

function renderSystemSelfCheck() {
    const updated = document.getElementById('system-self-check-updated');
    const panel = document.getElementById('system-self-check-panel');
    const data = state.systemSelfCheck || {};
    if (updated) {
        updated.textContent = data.checked_at ? toBeijingTime(data.checked_at) : '等待自检';
    }
    if (!panel) return;
    const items = Array.isArray(data.items) ? data.items : [];
    if (!items.length) {
        panel.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:16px;">还没有自检结果。</div>';
        return;
    }
    const summary = data.summary || {};
    const groupedItems = selfCheckGroupedItems(items);
    const problemItems = items.filter(item => ['critical', 'warning'].includes(String(item.status || '')));
    const highlightItems = (problemItems.length ? problemItems : items).slice(0, 3);
    const summaryHtml = `
        <div class="self-check-summary">
            <div class="self-check-card self-check-overview ${data.status || 'info'}">
                <div class="self-check-title"><span>\u603b\u4f53\u72b6\u6001</span><strong>${escHtml(selfCheckStatusLabel(data.status))}</strong></div>
                <div class="self-check-message">\u5f02\u5e38 ${Number(summary.critical || 0)} \u00b7 \u9700\u5173\u6ce8 ${Number(summary.warning || 0)} \u00b7 \u63d0\u793a ${Number(summary.info || 0)} \u00b7 \u6b63\u5e38 ${Number(summary.ok || 0)}</div>
            </div>
            ${highlightItems.map(item => selfCheckItemHtml(item)).join('')}
        </div>`;
    const detailHtml = `
        <div class="self-check-group-list">
            ${groupedItems.map(([status, rows]) => `
                <section class="self-check-group self-check-group-${escHtml(status)}">
                    <div class="self-check-group-head">
                        <strong>${escHtml(selfCheckStatusGroupTitle(status, rows.length))}</strong>
                        <span>${escHtml(selfCheckStatusLabel(status))}</span>
                    </div>
                    <div class="self-check-group-grid">${rows.map(item => selfCheckItemHtml(item)).join('')}</div>
                </section>
            `).join('')}
        </div>`;
    panel.innerHTML = summaryHtml + detailHtml;
}

function monitorNumber(value, digits = 1) {
    const n = Number(value || 0);
    if (!Number.isFinite(n)) return '0';
    const rawDigits = Number(digits);
    const fractionDigits = Number.isFinite(rawDigits)
        ? Math.max(0, Math.min(Math.trunc(rawDigits), 6))
        : 1;
    return n.toLocaleString('zh-CN', {
        maximumFractionDigits: fractionDigits,
        minimumFractionDigits: 0,
    });
}

function monitorPercentTone(value) {
    const n = Number(value || 0);
    if (n >= 90) return 'bad';
    if (n >= 75) return 'warn';
    return 'good';
}

function monitorMetric(label, value, subtitle = '', pct = null) {
    const tone = pct === null ? 'good' : monitorPercentTone(pct);
    const bar = pct === null
        ? ''
        : `<div class="server-monitor-progress"><div class="server-monitor-progress-bar ${tone === 'bad' ? 'bad' : tone === 'warn' ? 'warn' : ''}" style="width:${Math.max(0, Math.min(Number(pct || 0), 100)).toFixed(1)}%;"></div></div>`;
    return `
        <div class="server-monitor-card server-monitor-${tone}">
            <div class="server-monitor-label">${escHtml(label)}</div>
            <div class="server-monitor-value">${escHtml(value)}</div>
            ${subtitle ? `<div class="server-monitor-sub">${escHtml(subtitle)}</div>` : ''}
            ${bar}
        </div>`;
}

function renderServerMonitor() {
    const updated = document.getElementById('server-monitor-updated');
    const overview = document.getElementById('server-monitor-overview');
    const runtimeEl = document.getElementById('server-monitor-model-runtime');
    const servicesEl = document.getElementById('server-monitor-services');
    const platformOverview = document.getElementById('platform-server-overview');
    const platformServices = document.getElementById('platform-server-services');
    const platformRuntime = document.getElementById('platform-server-runtime');
    const data = state.serverMonitorStatus || {};
    if (updated) {
        updated.textContent = data.checked_at ? toBeijingTime(data.checked_at) : new Date().toLocaleTimeString('zh-CN', { hour12: false });
    }
    document.querySelectorAll('[data-server-monitor-tab]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.serverMonitorTab === state.serverMonitorTab);
    });
    document.querySelectorAll('.server-monitor-panel').forEach(panel => {
        panel.classList.toggle(
            'active',
            panel.id === `server-monitor-panel-${state.serverMonitorTab}`
        );
    });
    if (!overview || !runtimeEl || !servicesEl || !platformOverview || !platformServices || !platformRuntime) return;
    renderPlatformServerMonitor(data, platformOverview, platformServices, platformRuntime);
    if (!data.available) {
        const msg = data.message || data.status || '服务器监控暂不可用';
        overview.innerHTML = monitorMetric('连接状态', '不可用', msg, 100);
        runtimeEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:16px;">服务器未返回模型运行数据。</div>';
        servicesEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:16px;">服务器未返回服务状态。</div>';
        return;
    }

    const cpu = data.cpu || {};
    const memory = data.memory || {};
    const gpu = (data.gpu?.gpus || [])[0] || {};
    const disks = data.disks || [];
    const mainDisk = disks.find(d => d.path === '/data') || disks[0] || {};
    const gpuMemPct = Number(gpu.memory_used_pct || 0);
    overview.innerHTML = [
        monitorMetric('CPU 使用率', `${monitorNumber(cpu.usage_pct)}%`, `${Number(cpu.cores || 0)} 核 · 负载 ${monitorNumber(cpu.load_1m)}/${monitorNumber(cpu.load_5m)}/${monitorNumber(cpu.load_15m)}`, cpu.usage_pct),
        monitorMetric('内存使用', `${monitorNumber(memory.used_pct)}%`, `${monitorNumber(memory.used_mb / 1024)} / ${monitorNumber(memory.total_mb / 1024)} GB`, memory.used_pct),
        monitorMetric('GPU 使用率', gpu.name ? `${monitorNumber(gpu.utilization_pct)}%` : '未检测到', gpu.name ? `${gpu.name} · ${monitorNumber(gpu.temperature_c, 0)}°C · ${monitorNumber(gpu.power_w, 0)}W` : (data.gpu?.error || 'nvidia-smi 未返回 GPU'), gpu.name ? gpu.utilization_pct : null),
        monitorMetric('显存占用', gpu.name ? `${monitorNumber(gpuMemPct)}%` : '未检测到', gpu.name ? `${monitorNumber(gpu.memory_used_mb / 1024)} / ${monitorNumber(gpu.memory_total_mb / 1024)} GB` : '', gpu.name ? gpuMemPct : null),
        monitorMetric('磁盘使用', `${monitorNumber(mainDisk.used_pct)}%`, `${mainDisk.path || '-'} · ${monitorNumber(mainDisk.used_gb)} / ${monitorNumber(mainDisk.total_gb)} GB`, mainDisk.used_pct),
        monitorMetric('服务器', data.hostname || data.host || '-', data.host ? `公网 ${data.host}` : '', null),
    ].join('');

    renderServerModelRuntime(data, runtimeEl);

    const services = data.services || [];
    if (!services.length) {
        servicesEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:16px;">没有返回模型服务状态。</div>';
        return;
    }
    servicesEl.innerHTML = `<div class="server-monitor-services">${services.map(s => `
        <div class="server-monitor-service">
            <div>
                <strong>${escHtml(s.name || '-')}</strong>
                <span>${escHtml(s.active ? '运行中' : (s.status || '未运行'))}${s.pid ? ` · PID ${escHtml(s.pid)}` : ''}${s.elapsed ? ` · 已运行 ${escHtml(s.elapsed)}` : ''}</span>
            </div>
            <span class="status-badge ${s.active ? 'status-live' : 'status-paused'}">${s.active ? 'ACTIVE' : 'DOWN'}</span>
        </div>
    `).join('')}</div>`;
}

function renderPlatformServerMonitor(data, overview, servicesEl, runtimeEl) {
    const platform = data.platform_server || {};
    if (!platform.available) {
        overview.innerHTML = monitorMetric(
            '平台服务器',
            '不可用',
            platform.message || platform.status || '平台服务器状态暂未返回',
            100
        );
        servicesEl.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:16px;">平台服务状态暂不可用。</div>';
        runtimeEl.innerHTML = renderPlatformRuntimeCard(data.platform_runtime || {});
        return;
    }
    const cpu = platform.cpu || {};
    const memory = platform.memory || {};
    const disks = Array.isArray(platform.disks) ? platform.disks : [];
    const mainDisk = disks[0] || {};
    const uptime = Number(platform.uptime_seconds);
    const uptimeText = Number.isFinite(uptime) && uptime > 0
        ? `${monitorNumber(uptime / 3600, 1)} 小时`
        : '-';
    overview.innerHTML = [
        monitorMetric('CPU 使用率', `${monitorNumber(cpu.usage_pct)}%`, `${Number(cpu.cores || 0)} 核 · 负载 ${monitorNumber(cpu.load_1m)}/${monitorNumber(cpu.load_5m)}/${monitorNumber(cpu.load_15m)}`, cpu.usage_pct),
        monitorMetric('内存使用', `${monitorNumber(memory.used_pct)}%`, `${monitorNumber(memory.used_mb / 1024)} / ${monitorNumber(memory.total_mb / 1024)} GB`, memory.used_pct),
        monitorMetric('磁盘使用', `${monitorNumber(mainDisk.used_pct)}%`, `${mainDisk.path || '-'} · ${monitorNumber(mainDisk.used_gb)} / ${monitorNumber(mainDisk.total_gb)} GB`, mainDisk.used_pct),
        monitorMetric('平台主机', platform.hostname || '-', `${platform.platform || '-'} · Python ${platform.python || '-'} · 运行 ${uptimeText}`, null),
    ].join('');
    const services = Array.isArray(platform.services) ? platform.services : [];
    const visibleServices = Array.from(
        services.reduce((map, service) => {
            const key = serviceLabel(service.name);
            const current = map.get(key);
            if (!current) {
                map.set(key, service);
                return map;
            }
            const currentScore = (current.active ? 2 : 0) + (current.pid ? 1 : 0);
            const nextScore = (service.active ? 2 : 0) + (service.pid ? 1 : 0);
            if (nextScore >= currentScore) map.set(key, service);
            return map;
        }, new Map()).values()
    );
    servicesEl.innerHTML = visibleServices.length
        ? `<div class="server-monitor-services platform-services">${visibleServices.map(service => `
            <div class="server-monitor-service">
                <div>
                    <strong>${escHtml(serviceLabel(service.name))}</strong>
                    <span>${escHtml(service.name || '-')} · ${escHtml(service.active ? '运行中' : service.status || '未运行')}${service.pid ? ` · PID ${escHtml(service.pid)}` : ''}${service.elapsed ? ` · 已运行 ${escHtml(service.elapsed)}` : ''}</span>
                </div>
                <span class="status-badge ${service.active ? 'status-live' : 'status-paused'}">${service.active ? 'ACTIVE' : 'DOWN'}</span>
            </div>
        `).join('')}</div>`
        : '<div style="color:var(--text-muted);font-size:12px;padding:16px;">没有返回平台服务状态。</div>';
    runtimeEl.innerHTML = renderPlatformRuntimeCard(data.platform_runtime || {});
}

function serviceLabel(name) {
    const labels = {
        'bb-dashboard.service': 'Dashboard 看板',
        'bb-paper-trading.service': '交易主循环',
        'bb-model-tunnels.service': '模型隧道',
        'postgresql.service': 'PostgreSQL',
        'redis-server.service': 'Redis',
        'redis.service': 'Redis',
    };
    return labels[name] || name || '-';
}

function renderPlatformRuntimeCard(platformRuntime) {
    const platformModels = Array.isArray(platformRuntime.ai_models) ? platformRuntime.ai_models : [];
    const platformTools = platformRuntime.local_ai_tools || {};
    const childEndpoints = platformTools.child_endpoints || {};
    const childRows = Object.entries(childEndpoints);
    const childAvailable = childRows.filter(([, item]) => item && (item.available || item.ok)).length;
    const modelRows = platformModels.length
        ? platformModels.map(item => `
            <div class="server-monitor-process">
                <span>${escHtml(item.label || item.name || item.model || '-')} · 平台调用 ${escHtml(item.api_base || '-')}<br><em>目标：${escHtml(item.model || '-')} · 返回：${escHtml((item.models || []).join('、') || '未返回模型名')} · ${escHtml(runtimeEndpointSummary(item) || '-')}</em></span>
                <strong>${item.available ? '正常' : (item.endpoint_ok ? '模型不匹配' : '不可达')}</strong>
            </div>
        `).join('')
        : '<div style="color:var(--text-muted);font-size:11px;">未配置平台侧模型端点。</div>';
    const childHtml = childRows.length
        ? childRows.map(([name, item]) => `
            <div class="server-monitor-process">
                <span>${escHtml(name)} · ${escHtml(item.path || '-')}<br><em>${escHtml(runtimeEndpointSummary(item) || item.error || '-')}</em></span>
                <strong>${item.available ? '正常' : (item.ok ? '接口异常' : '不可达')}</strong>
            </div>
        `).join('')
        : '<div style="color:var(--text-muted);font-size:11px;">本地量化工具子接口未返回。</div>';
    return `
        <div class="server-monitor-runtime">
            <div class="server-monitor-runtime-card">
                <strong>平台实际调用模型端点</strong>
                <div class="server-monitor-process-list">${modelRows}</div>
            </div>
            <div class="server-monitor-runtime-card">
                <strong>平台本地量化工具 ${runtimeStatusBadge(platformTools.available)}</strong>
                <div>平台调用地址：${escHtml(platformTools.api_base || '-')}</div>
                <div>健康接口：${escHtml(runtimeEndpointSummary(platformTools.health) || '-')}</div>
                <div>状态接口：${escHtml(runtimeEndpointSummary(platformTools.status) || '-')}</div>
                <div>子接口：${childRows.length ? `${childAvailable}/${childRows.length} 正常` : '-'}</div>
                <div>训练模型：${escHtml(platformTools.model_bundle_available ? '已就绪' : '未就绪/启发式可用')}</div>
                ${platformTools.status && platformTools.status.error ? `<div style="color:var(--red);">状态接口：${escHtml(platformTools.status.error)}</div>` : ''}
                <div class="server-monitor-process-list">${childHtml}</div>
            </div>
        </div>`;
}

function runtimeStatusBadge(ok) {
    return `<span class="status-badge ${ok ? 'status-live' : 'status-paused'}">${ok ? '运行中' : '异常'}</span>`;
}

function runtimeEndpointSummary(health) {
    if (!health || typeof health !== 'object') return '';
    const status = Number(health.status_code || 0);
    const latency = Number(health.latency_ms);
    const parts = [];
    parts.push(status ? `HTTP ${status}` : (health.ok ? 'HTTP 正常' : 'HTTP 未连接'));
    if (Number.isFinite(latency)) parts.push(`${monitorNumber(latency, 0)} ms`);
    if (health.error) parts.push(String(health.error));
    if (health.truncated) parts.push('响应已截断');
    return parts.join(' · ');
}

function renderServerModelRuntime(data, container) {
    const runtime = data.model_runtime || {};
    const platformRuntime = data.platform_runtime || {};
    const vllm = runtime.vllm || {};
    const vllmEndpoints = Array.isArray(runtime.vllm_endpoints) ? runtime.vllm_endpoints : [];
    const tools = runtime.local_ai_tools || {};
    const processes = data.gpu_processes || [];
    const models = Array.isArray(vllm.models) && vllm.models.length ? vllm.models.join('、') : '未返回模型名';
    const platformTools = platformRuntime.local_ai_tools || {};
    const platformToolChildren = platformTools.child_endpoints || {};
    const platformToolChildEntries = Object.entries(platformToolChildren);
    const platformToolChildAvailable = platformToolChildEntries.filter(([, item]) => item && item.available).length;
    const platformToolContract = platformTools.tunnel_contract || {};
    const platformToolContractOk = platformToolContract.ok !== false;
    const toolsAvailable = Boolean(
        tools.available || (platformToolContractOk && (platformTools.available || platformToolChildAvailable > 0))
    );
    const toolsModels = tools.models || platformTools.models || {};
    const vllmLabel = vllm.label || vllm.provider_model || 'vLLM';
    const vllmHealthLine = runtimeEndpointSummary(vllm.health);
    const toolsStatusLine = runtimeEndpointSummary(tools.status_health);
    const toolsHealthLine = runtimeEndpointSummary(tools.health);
    const platformModels = Array.isArray(platformRuntime.ai_models) ? platformRuntime.ai_models : [];
    const MODEL_PUBLIC_HOST = '103.85.84.147';
    const MODEL_PUBLIC_ENDPOINTS = {
        'qwen3-14b-trade': `http://${MODEL_PUBLIC_HOST}:21840/v1`,
        'deepseek-r1-14b-risk': `http://${MODEL_PUBLIC_HOST}:21842/v1`,
        local_ai_tools: `http://${MODEL_PUBLIC_HOST}:21841`,
    };
    const platformModelPublicUrl = (modelId, fallbackPort = '21840') => {
        return MODEL_PUBLIC_ENDPOINTS[modelId] || `http://${MODEL_PUBLIC_HOST}:${fallbackPort}/v1`; 
    };
    const configuredOrPublicModelEndpoint = (modelId, configuredBaseValue = '', fallbackPort = '21840') => {
        const configuredBase = String(configuredBaseValue || '').trim().replace(/\/$/, '');
        if (
            !configuredBase
            || configuredBase.includes('127.0.0.1')
            || configuredBase.includes('localhost')
            || configuredBase.includes(':18000')
            || configuredBase.includes(':18002')
        ) {
            return platformModelPublicUrl(modelId, fallbackPort);
        }
        return configuredBase;
    };
    const localToolsPublicUrl = () => {
        return MODEL_PUBLIC_ENDPOINTS.local_ai_tools; 
    }; 
    const vllmEndpointRows = vllmEndpoints.length
        ? `<div class="server-monitor-process-list">${vllmEndpoints.map(item => {
            const endpointModels = Array.isArray(item.models) && item.models.length ? item.models.join('、') : '未返回模型名';
            const healthLine = runtimeEndpointSummary(item.health);
            const targetModel = item.provider_model || (String(item.label || '').includes('DeepSeek') ? 'deepseek-r1-14b-risk' : '-');
            const publicPort = targetModel === 'deepseek-r1-14b-risk' ? '21842' : '21840';
            const publicEndpoint = configuredOrPublicModelEndpoint(
                targetModel,
                item.api_base || item.endpoint,
                publicPort
            );
            const state = item.available ? '模型命中' : (item.endpoint_available ? '端点正常/模型不匹配' : '不可达');
            return `
                <div class="server-monitor-process server-monitor-endpoint-row">
                    <span>${escHtml(item.label || 'vLLM')} · 内网 ${escHtml(item.endpoint || '-')} · 外网 ${escHtml(publicEndpoint)}<br><em>目标：${escHtml(targetModel)} · 返回：${escHtml(endpointModels)}${healthLine ? ` · ${escHtml(healthLine)}` : ''}</em></span>
                    <strong>${escHtml(state)}</strong>
                </div>`;
        }).join('')}</div>`
        : '<div style="color:var(--text-muted);font-size:11px;">没有返回 vLLM 端点明细。</div>'; 
    const platformRows = platformModels.length
        ? `<div class="server-monitor-process-list">${platformModels.map(item => `
            <div class="server-monitor-process">
                <span>${escHtml(item.label || item.name || item.model || '-')} · ${escHtml(item.api_base || '-')}</span>
                <strong>${item.available ? '正常' : (item.endpoint_ok ? '模型不匹配' : '不可达')}</strong>
            </div>
        `).join('')}</div>`
        : '<div style="color:var(--text-muted);font-size:11px;">未配置平台侧模型端点。</div>';
    const processRows = processes.length
        ? `<div class="server-monitor-process-list">${processes.map(p => `
            <div class="server-monitor-process">
                <span>${escHtml(p.process_name || '-')} · PID ${escHtml(p.pid || '-')}</span>
                <strong>${monitorNumber(Number(p.used_memory_mb || 0) / 1024)} GB</strong>
            </div>
        `).join('')}</div>`
        : '<div style="color:var(--text-muted);font-size:11px;">没有检测到 GPU 模型进程。</div>';

    container.innerHTML = `
        <div class="server-monitor-runtime">
            <div class="server-monitor-runtime-card">
                <strong>${escHtml(vllmLabel)} / vLLM ${runtimeStatusBadge(vllm.available)}</strong>
                <div>内网地址：${escHtml(vllm.endpoint || '-')}</div>
                <div>外网地址：${escHtml(configuredOrPublicModelEndpoint(vllm.provider_model, vllm.api_base || vllm.endpoint, vllm.provider_model === 'deepseek-r1-14b-risk' ? '21842' : '21840'))}</div>
                <div>状态：${escHtml(vllm.status || '-')}${vllmHealthLine ? ` · ${escHtml(vllmHealthLine)}` : ''}</div>
                <div>配置模型：${escHtml(vllm.provider_model || '-')}</div>
                <div>模型：${escHtml(models)}</div>
                ${vllm.error ? `<div style="color:var(--red);">错误：${escHtml(vllm.error)}</div>` : ''}
            </div>
            <div class="server-monitor-runtime-card">
                <strong>vLLM 端点列表</strong>
                ${vllmEndpointRows}
            </div>
            <div class="server-monitor-runtime-card">
                <strong>本地量化模型 ${runtimeStatusBadge(toolsAvailable)}</strong>
                <div>内网地址：${escHtml(tools.endpoint || '-')}</div>
                <div>外网地址：${escHtml(localToolsPublicUrl())}</div>
                <div>平台调用：${escHtml(platformTools.api_base || '-')}</div>
                ${platformTools.expected_platform_api_base ? `<div>平台应调用：${escHtml(platformTools.expected_platform_api_base)}</div>` : ''}
                ${platformToolContract.message ? `<div style="color:var(--accent-light);">契约提示：${escHtml(platformToolContract.message)}</div>` : ''}
                ${platformTools.config_issue ? `<div style="color:var(--red);">配置问题：${escHtml(platformTools.config_issue)}</div>` : ''}
                <div>状态接口：${escHtml(toolsStatusLine || '-')}</div>
                <div>健康接口：${escHtml(toolsHealthLine || '-')}</div>
                <div>平台子接口：${platformToolChildEntries.length ? `${platformToolChildAvailable}/${platformToolChildEntries.length} 正常` : '-'}</div>
                <div>训练时间：${tools.trained_at ? toBeijingTime(tools.trained_at) : '-'}</div>
                <div>影子样本：窗口 ${monitorNumber(tools.shadow_sample_count, 0)} / 累计 ${monitorNumber(tools.completed_shadow_sample_count, 0)}</div>
                <div>交易样本：窗口 ${monitorNumber(tools.trade_sample_count, 0)} / 累计 ${monitorNumber(tools.completed_trade_sample_count, 0)}</div>
                <div>盈利模型：${escHtml(toolsModels.profit || '未返回')}</div>
                <div>平仓模型：${escHtml(toolsModels.exit || '未返回')}</div>
                ${tools.error ? `<div style="color:var(--red);">错误：${escHtml(tools.error)}</div>` : ''}
            </div>
            <div class="server-monitor-runtime-card">
                <strong>GPU 模型进程</strong>
                ${processRows}
            </div>
            <div class="server-monitor-runtime-card">
                <strong>平台实际调用端点</strong>
                ${platformRows}
                <div>量化工具：${escHtml(platformTools.configured ? (platformTools.available ? '可访问' : '不可访问') : '未配置')}</div>
                <div>量化工具地址：${escHtml(platformTools.api_base || '-')}</div>
                ${platformTools.expected_platform_api_base ? `<div>期望地址：${escHtml(platformTools.expected_platform_api_base)}</div>` : ''}
                ${platformTools.config_issue ? `<div style="color:var(--red);">配置问题：${escHtml(platformTools.config_issue)}</div>` : ''}
                <div>训练模型：${escHtml(platformTools.model_bundle_available ? '已就绪' : '未就绪/启发式可用')}</div>
                ${platformTools.status && platformTools.status.error ? `<div style="color:var(--red);">状态接口：${escHtml(platformTools.status.error)}</div>` : ''}
                ${platformToolChildEntries.length ? `<div class="server-monitor-process-list">${platformToolChildEntries.map(([name, item]) => `
                    <div class="server-monitor-process">
                        <span>${escHtml(name)} · ${escHtml(item.path || '-')}</span>
                        <strong>${item.available ? '正常' : (item.ok ? '接口异常' : '不可达')}</strong>
                    </div>
                `).join('')}</div>` : ''}
            </div>
        </div>`;
}

// --- Formatters ---
function actionLabel(a) { 
    const map = { 
        long: '做多', 
        open_long: '做多', 
        short: '做空', 
        open_short: '做空', 
        close_long: '平多', 
        close_short: '平空', 
        hold: '观望', 
        wait: '观望', 
        none: '观望', 
        buy: '买入', 
        sell: '卖出', 
    }; 
    return map[a] || a || '未知'; 
} 

function closeStatusLabel(record) {
    const action = String(record?.action || record?.side || '').toLowerCase();
    if (!['close_long', 'close_short'].includes(action)) return '';
    if (record?.close_status_label) return record.close_status_label;
    const status = String(record?.close_status || '').toLowerCase();
    if (status === 'partial') return '部分平仓';
    if (status === 'full') return '全部平仓';
    const pct = Number(record?.position_size_pct || record?.close_ratio || 0);
    if (pct > 0 && pct < 0.999) return '部分平仓';
    if (pct >= 0.999) return '全部平仓';
    return '';
}

function executionActionCell(record) {
    const action = record?.action || record?.side || 'hold';
    const closeLabel = closeStatusLabel(record);
    const statusColor = record?.close_status === 'partial' ? 'var(--accent-light)' : 'var(--text-muted)';
    const closeHtml = closeLabel
        ? `<div style="margin-top:4px;font-size:10px;line-height:1.2;color:${statusColor};white-space:nowrap;">${escHtml(closeLabel)}</div>`
        : '';
    return `<span class="badge badge-${action || 'hold'}">${actionLabel(action)}</span>${closeHtml}`;
}

function decisionType(action) {
    if (action === 'long' || action === 'short') return 'entry';
    if (action === 'close_long' || action === 'close_short') return 'exit';
    if (action === 'hold') return 'hold';
    return 'other';
}

function decisionTypeLabel(decisionOrAction) {
    if (decisionOrAction && typeof decisionOrAction === 'object' && decisionOrAction.decision_type_label) {
        return decisionOrAction.decision_type_label;
    }
    const action = typeof decisionOrAction === 'string' ? decisionOrAction : decisionOrAction?.action;
    const map = {
        entry: '开仓决策',
        exit: '平仓决策',
        hold: '观望决策',
        other: '其他决策',
    };
    return map[decisionType(action)];
}

function sideLabel(s) {
    const map = { long: '做多', short: '做空', close_long: '平多', close_short: '平空', buy: '买入', sell: '卖出' };
    return map[s] || s || '未知';
}
function statusLabel(s) {
    const map = { filled: '已成交', rejected: '已拒绝', pending: '待成交', open: '待成交', partial: '部分成交', canceled: '已取消', cancelled: '已取消' };
    return map[s] || s || '-';
}

function executionStatusPresentation(record, explicitSuccess = null) {
    const success = explicitSuccess ?? (record?.success === true || record?.status === 'filled');
    const kind = String(record?.execution_failure_kind || record?.final_result?.status || '').toLowerCase();
    const isTransientExchange = kind === 'transient_exchange_error';
    const label = record?.execution_status_label || (success ? '执行成功' : (isTransientExchange ? '交易所临时不可用' : '执行失败'));
    const color = success ? 'var(--green)' : (isTransientExchange ? 'var(--orange)' : 'var(--red)');
    return { label, color, kind, success, isTransientExchange };
}
function fmtPrice(p) { return p ? Number(p).toFixed(4) : '0.0000'; }
function fmtPct(p) { return p ? Number(p).toFixed(2) + '%' : '0.00%'; }
function fmtNum(n) { return n ? Number(n).toFixed(4) : '0'; }
function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}时${m}分`;
}

function toBeijingTime(isoStr) {
    if (!isoStr) return '-';
    // SQLite stores UTC without timezone indicator; JS would treat it as local time.
    const text = String(isoStr).trim();
    const hasTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(text);
    const normalized = hasTimezone ? text : text.replace(' ', 'T') + 'Z';
    const d = new Date(normalized);
    if (Number.isNaN(d.getTime())) return '-';
    const parts = new Intl.DateTimeFormat('zh-CN', {
        timeZone: 'Asia/Shanghai',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
        hourCycle: 'h23',
    }).formatToParts(d);
    const lookup = Object.fromEntries(parts.map(part => [part.type, part.value]));
    return `${lookup.year}-${lookup.month}-${lookup.day} ${lookup.hour}:${lookup.minute}`;
}

function beijingDateTimeParts(isoStr) {
    if (!isoStr) return null;
    const text = String(isoStr).trim();
    const hasTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(text);
    const normalized = hasTimezone ? text : text.replace(' ', 'T') + 'Z';
    const d = new Date(normalized);
    if (Number.isNaN(d.getTime())) return null;
    const parts = new Intl.DateTimeFormat('zh-CN', {
        timeZone: 'Asia/Shanghai',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    }).formatToParts(d).reduce((acc, part) => {
        if (part.type !== 'literal') acc[part.type] = part.value;
        return acc;
    }, {});
    const date = `${parts.year}/${parts.month}/${parts.day}`;
    const time = `${parts.hour}:${parts.minute}:${parts.second}`;
    return { date, time, full: `${date} ${time}` };
}

function toBeijingDateTime(isoStr) {
    return beijingDateTimeParts(isoStr)?.full || '-';
}

function tradeReflectionTimeHtml(isoStr) {
    const parts = beijingDateTimeParts(isoStr);
    if (!parts) return '-';
    return `<span>${escHtml(parts.date)}</span><em>${escHtml(parts.time)}</em>`;
}

function shortBeijingTime(isoStr) {
    if (!isoStr) return '-';
    const text = String(isoStr).trim();
    const hasTimezone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(text);
    const normalized = hasTimezone ? text : text.replace(' ', 'T') + 'Z';
    const d = new Date(normalized);
    return d.toLocaleTimeString('zh-CN', { timeZone: 'Asia/Shanghai', hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

function loopErrorLabel(message) {
    const text = String(message || '').trim();
    if (!text) return '';
    if (text.includes('reconciliation timed out')) {
        return 'OKX 仓位/保护单同步超时，系统已跳过外部同步并继续分析，避免主循环卡死。';
    }
    if (text.includes('Invalid OK-ACCESS-KEY') || text.includes('50111')) {
        return 'OKX API Key 无效，余额/仓位同步可能失败，请检查当前模式的 OKX Key 配置。';
    }
    return text;
}

// ========== Auto Price Chart ==========

function updateAutoPriceChartTitle(symbol) {
    const titleEl = document.getElementById('price-chart-title');
    const subtitleEl = document.getElementById('price-chart-subtitle');
    if (!symbol) {
        if (titleEl) titleEl.textContent = '持仓价格走势';
        if (subtitleEl) subtitleEl.textContent = '无持仓时不加载交易对';
        return;
    }
    if (titleEl) titleEl.textContent = `${symbol} 价格走势`;
    if (subtitleEl) subtitleEl.textContent = '自动跟随当前持仓';
}

function clearPriceChart() {
    if (!window._charts?.charts?.price) return;
    const chart = window._charts.charts.price;
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update();
}

function preferredPriceChartSymbol() {
    const symbols = Object.keys(state.tickers || {});
    if (!symbols.length) return '';
    if (state.priceChartSymbol && symbols.includes(state.priceChartSymbol)) {
        return state.priceChartSymbol;
    }
    return symbols.sort((a, b) => a.localeCompare(b))[0];
}

async function refreshAutoPriceChart() {
    const symbol = preferredPriceChartSymbol();
    if (!symbol) {
        state.priceChartSymbol = '';
        updateAutoPriceChartTitle('');
        clearPriceChart();
        return;
    }
    if (state.priceChartSymbol === symbol) return;
    state.priceChartSymbol = symbol;
    updateAutoPriceChartTitle(symbol);
    await loadPriceChartKlines(symbol, state.priceChartTimeframe);
}

async function loadPriceChartKlines(symbol, timeframe) {
    if (!symbol) return;
    const encodedSymbol = encodeURIComponent(symbol);
    const data = await fetchJSON(
        `/api/market/klines/${encodedSymbol}?timeframe=${timeframe}&limit=100`
    );
    if (!data || !data.data || data.data.length === 0) return;

    if (window._charts) {
        window._charts.updatePriceChart(data.data);
    }
}

// ========== Auto Status Panel ==========

function isMojibakeText(value) {
    if (!value) return false;
    const text = String(value);
    return /[\u9352\u951b\u7ef1\u93c2\u93c8\u9422\u7039\u9477\u6d7c\u95c2\u6f36\u6401\u64c3\u93c9\u95ab\u95c6\u7ee9\u7f01\u7ece\u93b5]/.test(text);
}

function cleanStatusText(value, fallback) {
    if (value === null || value === undefined || value === '') return fallback;
    const text = String(value);
    return isMojibakeText(text) ? fallback : text;
}

function autoStatusStageLabel(stats) {
    const stage = String(stats?.current_stage || '');
    const labels = {
        idle: '\u7a7a\u95f2\uff0c\u7b49\u5f85\u4e0b\u4e00\u8f6e\u5206\u6790',
        starting: '\u51c6\u5907\u5f00\u59cb\u672c\u8f6e\u5206\u6790',
        shadow_backtests: '\u66f4\u65b0\u5f71\u5b50\u590d\u76d8',
        sync_exchange_positions: '\u540c\u6b65 OKX \u4ed3\u4f4d/\u4fdd\u62a4\u5355',
        load_open_positions: '\u8bfb\u53d6\u672c\u5730\u6301\u4ed3',
        recover_pending_exits: '\u8865\u6267\u884c\u672a\u5b8c\u6210\u5e73\u4ed3',
        select_symbols: '\u7b5b\u9009\u672c\u8f6e\u5206\u6790\u5e01\u79cd',
        fetch_features: '\u83b7\u53d6\u884c\u60c5\u6307\u6807',
        refresh_position_prices: '\u5237\u65b0\u6301\u4ed3\u4ef7\u683c',
        enforce_sl_tp: '\u68c0\u67e5\u6b62\u76c8\u6b62\u635f',
        review_open_positions: '\u590d\u76d8\u5f53\u524d\u6301\u4ed3',
        publish_results: '\u5199\u5165\u5e76\u63a8\u9001\u5206\u6790\u7ed3\u679c',
        error: '\u672c\u8f6e\u5f02\u5e38',
    };
    if (labels[stage]) return labels[stage];
    if (stage.startsWith('analyze:')) {
        return `\u6b63\u5728\u5206\u6790 ${stage.split(':').slice(1).join(':')}`;
    }
    if (stage.startsWith('execute:')) {
        return `\u6b63\u5728\u6267\u884c ${stage.split(':').slice(1).join(':')} \u8ba2\u5355`;
    }
    return cleanStatusText(
        stats?.current_stage_label,
        stats?.running ? '\u7b49\u5f85\u4e0b\u4e00\u8f6e\u5206\u6790' : '\u670d\u52a1\u672a\u8fd0\u884c'
    );
}

function updateAutoStatus(stats) {
    const scanModeEl = document.getElementById('status-scan-mode');
    if (scanModeEl) {
        scanModeEl.textContent = '\u81ea\u52a8\u626b\u63cf\u5168\u5e02\u573a (OKX)';
    }

    const modelCountEl = document.getElementById('status-model-count');
    if (modelCountEl) {
        const expertCount = state.aiExpertModels.length || FIXED_AI_EXPERT_FALLBACKS.length;
        modelCountEl.textContent = `${expertCount} / 1`;
    }

    if (stats && stats.decision_interval) {
        state.decisionInterval = stats.decision_interval;
    }

    const intervalEl = document.getElementById('status-interval');
    if (intervalEl) {
        intervalEl.textContent = `${state.decisionInterval}\u79d2/\u8f6e`;
    }

    const dtEl = document.getElementById('status-decision-trade');
    if (dtEl) updateDecisionPositionStatus();

    const stageEl = document.getElementById('status-current-stage');
    if (stageEl) {
        const stage = autoStatusStageLabel(stats);
        const seconds = Math.round(Number(stats?.round_running_seconds || 0));
        stageEl.textContent = stats?.round_active
            ? `${stage}\uff0c\u5df2\u7528 ${seconds} \u79d2`
            : stage;
    }

    const timingEl = document.getElementById('status-round-timing');
    if (timingEl) {
        const started = stats?.last_round_started_at ? shortBeijingTime(stats.last_round_started_at) : '-';
        const finished = stats?.last_round_finished_at
            ? shortBeijingTime(stats.last_round_finished_at)
            : '\u8fdb\u884c\u4e2d';
        timingEl.textContent = `\u5f00\u59cb ${started} / \u5b8c\u6210 ${finished}`;
    }

    const errRow = document.getElementById('status-loop-error-row');
    const errEl = document.getElementById('status-loop-error');
    if (errRow && errEl) {
        const err = loopErrorLabel(stats?.last_round_error);
        errRow.style.display = err ? 'flex' : 'none';
        errEl.textContent = err || '-';
    }
}

// ========== Scan Mode Buttons ==========
function initScanModeButtons() {
    state.scanMode = 'auto';
    const scanLabel = document.getElementById('scan-mode-label');
    if (scanLabel) scanLabel.textContent = '自动扫描全市场 · 智能调度';
    updateSymbolCount();
    return;
}

// ========== Dashboard Account Settings ==========
const dashboardAuthState = {
    currentUsername: '',
    users: [],
    editingUsername: '',
};

function setSettingsStatus(id, message, ok = null) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = message || '';
    if (ok === true) el.style.color = 'var(--green)';
    else if (ok === false) el.style.color = 'var(--red)';
    else el.style.color = 'var(--text-muted)';
}

async function fetchModelServerSettings() {
    const data = await fetchJSON('/api/settings/model-server');
    if (!data) {
        setSettingsStatus('model-server-status', '模型服务器配置暂时不可用', false);
        return;
    }
    if (data.detail && typeof data.configured === 'undefined') {
        setSettingsStatus('model-server-status', apiErrorText(data), false);
        return;
    }
    setInputValue('model-server-host', data.host || '');
    setInputValue('model-server-port', data.port || 22);
    setInputValue('model-server-username', data.username || '');
    const password = document.getElementById('model-server-password');
    if (password) {
        password.value = '';
        password.placeholder = data.password_configured
            ? '已保存密码，留空不修改'
            : '请输入服务器密码';
    }
    const status = data.configured
        ? `已加密保存${data.updated_at ? ' · ' + data.updated_at : ''}`
        : '未配置，请填写后保存';
    setSettingsStatus('model-server-status', status, data.configured ? true : null);
}

function readModelServerForm() {
    return {
        host: (document.getElementById('model-server-host')?.value || '').trim(),
        port: Number(document.getElementById('model-server-port')?.value || 22),
        username: (document.getElementById('model-server-username')?.value || '').trim(),
        password: document.getElementById('model-server-password')?.value || '',
    };
}

function validateModelServerForm(payload, requirePassword = false) {
    if (!payload.host) return '请填写模型服务器地址';
    if (!Number.isInteger(payload.port) || payload.port < 1 || payload.port > 65535) {
        return 'SSH 端口必须在 1 到 65535 之间';
    }
    if (!payload.username) return '请填写模型服务器用户名';
    if (requirePassword && !payload.password) return '请填写模型服务器密码';
    return '';
}

async function saveModelServerSettings() {
    const payload = readModelServerForm();
    const validation = validateModelServerForm(payload, false);
    if (validation) {
        setSettingsStatus('model-server-status', validation, false);
        return;
    }
    setSettingsStatus('model-server-status', '保存中...', null);
    try {
        await postJSON('/api/settings/model-server', payload);
        setInputValue('model-server-password', '');
        setSettingsStatus('model-server-status', '模型服务器配置已加密保存', true);
        await fetchModelServerSettings();
        if (isPageActive('server-monitor')) fetchServerMonitor();
    } catch (error) {
        setSettingsStatus('model-server-status', `保存失败: ${error.message || error}`, false);
    }
}

async function testModelServerSettings() {
    const payload = readModelServerForm();
    const validation = validateModelServerForm(payload, false);
    if (validation) {
        setSettingsStatus('model-server-status', validation, false);
        return;
    }
    const btn = document.getElementById('model-server-test-btn');
    if (btn) btn.disabled = true;
    setSettingsStatus('model-server-status', '测试连接中...', null);
    try {
        const data = await postJSON('/api/settings/model-server/test', payload);
        setSettingsStatus(
            'model-server-status',
            data.success ? '连接成功，硬件与模型监控可用' : `连接失败: ${data.message || data.status || '未知错误'}`,
            Boolean(data.success),
        );
    } catch (error) {
        setSettingsStatus('model-server-status', `测试失败: ${error.message || error}`, false);
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function fetchDashboardAccountSettings() {
    const data = await fetchJSON('/api/auth/account');
    if (!data) return;
    const current = data.current_user || {};
    dashboardAuthState.currentUsername = current.username || '';
    setText('dashboard-current-user', dashboardAuthState.currentUsername || '未登录');
    dashboardAuthState.users = Array.isArray(data.users) ? data.users : [];
    renderDashboardUsers(dashboardAuthState.users, dashboardAuthState.currentUsername);
}

async function fetchDashboardAuthStatus() {
    const data = await fetchJSON('/api/auth/status');
    if (!data) return;
    dashboardAuthState.currentUsername = data.username || dashboardAuthState.currentUsername || '';
    setText('dashboard-current-user', dashboardAuthState.currentUsername || '未登录');
}

function renderDashboardUsers(users, currentUsername) {
    const tbody = document.getElementById('dashboard-users-tbody');
    if (!tbody) return;
    if (!Array.isArray(users) || !users.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted);text-align:center;padding:18px;">暂无会员账号</td></tr>'; 
        return;
    }
    tbody.innerHTML = users.map(user => {
        const username = String(user.username || '');
        const isCurrent = username === currentUsername;
        const active = user.is_active !== false;
        const status = active
            ? '<span class="settings-status-badge active">启用</span>'
            : '<span class="settings-status-badge inactive">停用</span>';
        const action = `
            <div class="dashboard-user-actions">
                <button class="btn btn-sm" type="button" data-dashboard-user-action="edit" data-username="${escHtml(username)}">修改</button>
                ${isCurrent ? '<span style="color:var(--text-muted);font-size:11px;">当前账号</span>' : `
                    <button class="btn btn-sm" type="button" data-dashboard-user-action="${active ? 'deactivate' : 'activate'}" data-username="${escHtml(username)}">${active ? '停用' : '启用'}</button>
                    <button class="btn btn-sm btn-danger" type="button" data-dashboard-user-action="delete" data-username="${escHtml(username)}">删除</button>
                `}
            </div>
        `;
        return `
            <tr>
                <td>${escHtml(username)}</td>
                <td>${escHtml(user.masked_email || user.email || '-')}</td>
                <td>${status}</td>
                <td>${escHtml(user.last_login_at ? toBeijingTime(user.last_login_at) : '-')}</td>
                <td>${action}</td>
            </tr>
        `;
    }).join('');
}

async function createDashboardUser() {
    openDashboardUserModal('create');
}

function findDashboardUser(username) {
    return (dashboardAuthState.users || []).find(user => String(user.username || '') === String(username || '')) || null;
}

function openDashboardUserModal(mode = 'create', username = '') {
    const editing = mode === 'edit';
    const user = editing ? findDashboardUser(username) : null;
    dashboardAuthState.editingUsername = user?.username || '';
    setInputValue('dashboard-user-modal-mode', editing ? 'edit' : 'create');
    setInputValue('dashboard-user-original-username', user?.username || '');
    setInputValue('dashboard-user-username', user?.username || '');
    setInputValue('dashboard-user-email', user?.email || '');
    setInputValue('dashboard-user-password', '');
    const usernameInput = document.getElementById('dashboard-user-username');
    if (usernameInput) usernameInput.readOnly = editing;
    const activeInput = document.getElementById('dashboard-user-active');
    if (activeInput) {
        activeInput.checked = editing ? user?.is_active !== false : true;
        activeInput.disabled = editing && user?.username === dashboardAuthState.currentUsername;
    }
    const title = document.getElementById('dashboard-user-modal-title');
    if (title) title.textContent = editing ? `修改会员：${user?.username || username}` : '新增会员';
    const passwordInput = document.getElementById('dashboard-user-password');
    if (passwordInput) passwordInput.placeholder = editing ? '留空表示不修改密码' : '初始密码，至少 10 位';
    setSettingsStatus('dashboard-user-modal-status', '', null);
    const overlay = document.getElementById('dashboard-user-modal-overlay');
    if (overlay) overlay.style.display = 'flex';
}

function closeDashboardUserModal() {
    const overlay = document.getElementById('dashboard-user-modal-overlay');
    if (overlay) overlay.style.display = 'none';
}

async function saveDashboardUserModal() {
    const mode = document.getElementById('dashboard-user-modal-mode')?.value || 'create';
    const originalUsername = (document.getElementById('dashboard-user-original-username')?.value || '').trim();
    const username = (document.getElementById('dashboard-user-username')?.value || '').trim();
    const email = (document.getElementById('dashboard-user-email')?.value || '').trim();
    const password = document.getElementById('dashboard-user-password')?.value || '';
    const isActive = document.getElementById('dashboard-user-active')?.checked !== false;
    if (!username) {
        setSettingsStatus('dashboard-user-modal-status', '用户名不能为空', false);
        return;
    }
    if (mode === 'create' && !password) {
        setSettingsStatus('dashboard-user-modal-status', '新增会员必须填写初始密码', false);
        return;
    }
    if (password && password.length < 10) {
        setSettingsStatus('dashboard-user-modal-status', '密码至少 10 位', false);
        return;
    }
    if (mode === 'edit' && originalUsername === dashboardAuthState.currentUsername && !isActive) {
        setSettingsStatus('dashboard-user-modal-status', '当前账号不能停用', false);
        return;
    }
    const saveBtn = document.getElementById('dashboard-user-save-btn');
    if (saveBtn) saveBtn.disabled = true;
    setSettingsStatus('dashboard-user-modal-status', '保存中...', null);
    try {
        if (mode === 'edit') {
            const payload = { email, role: 'admin', is_active: isActive };
            if (password) payload.password = password;
            await putJSON(`/api/auth/users/${encodeURIComponent(originalUsername || username)}`, payload);
            setSettingsStatus('dashboard-users-status', '会员已更新', true);
        } else {
            await postJSON('/api/auth/users', { username, email, password, role: 'admin', is_active: isActive });
            setSettingsStatus('dashboard-users-status', '会员已新增', true);
        }
        closeDashboardUserModal();
        await fetchDashboardAccountSettings();
    } catch (error) {
        setSettingsStatus('dashboard-user-modal-status', `保存失败：${error.message || error}`, false);
    } finally {
        if (saveBtn) saveBtn.disabled = false;
    }
}

async function setDashboardUserActive(username, active, sourceButton = null) {
    if (!username) return;
    if (username === dashboardAuthState.currentUsername && !active) {
        setSettingsStatus('dashboard-users-status', '当前账号不能停用', false);
        return;
    }
    const actionText = active ? '启用' : '停用';
    if (!confirm(`${actionText}会员 ${username}？`)) return;
    setButtonBusy(sourceButton, true, `${actionText}中`);
    setSettingsStatus('dashboard-users-status', `${actionText}中...`, null);
    try {
        if (active) {
            await dashboardUserWriteRequest(`/api/auth/users/${encodeURIComponent(username)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_active: true }),
            });
        } else {
            await dashboardUserWriteRequest(`/api/auth/users/${encodeURIComponent(username)}/deactivate`, {
                method: 'POST',
            });
        }
        setSettingsStatus('dashboard-users-status', `会员已${actionText}`, true);
        await fetchDashboardAccountSettings();
    } catch (error) {
        setSettingsStatus('dashboard-users-status', `${actionText}失败：${error.message || error}`, false);
    } finally {
        setButtonBusy(sourceButton, false);
    }
}

async function deleteDashboardUser(username, sourceButton = null) {
    if (!username) return;
    if (username === dashboardAuthState.currentUsername) {
        setSettingsStatus('dashboard-users-status', '当前账号不能删除', false);
        return;
    }
    if (!confirm(`删除会员 ${username}？删除后该账号无法登录。`)) return;
    setButtonBusy(sourceButton, true, '删除中');
    setSettingsStatus('dashboard-users-status', '删除中...', null);
    try {
        await dashboardUserWriteRequest(`/api/auth/users/${encodeURIComponent(username)}`, {
            method: 'DELETE',
        });
        setSettingsStatus('dashboard-users-status', '会员已删除', true);
        await fetchDashboardAccountSettings();
    } catch (error) {
        setSettingsStatus('dashboard-users-status', `删除失败：${error.message || error}`, false);
    } finally {
        setButtonBusy(sourceButton, false);
    }
}

window.createDashboardUser = createDashboardUser;
window.openDashboardUserModal = openDashboardUserModal;
window.closeDashboardUserModal = closeDashboardUserModal;
window.saveDashboardUserModal = saveDashboardUserModal;
window.setDashboardUserActive = setDashboardUserActive;
window.deleteDashboardUser = deleteDashboardUser;

// ========== OKX Settings (split paper/live) ==========
async function fetchOKXSettings() {
    const data = await fetchJSON('/api/settings/okx');
    if (!data) return;
    const hasCredentials = (item) => Boolean(
        item && item.api_key && item.has_secret && item.has_passphrase
    );
    state.okxConfig = {
        paperConfigured: hasCredentials(data.paper),
        liveConfigured: hasCredentials(data.live),
    };
    updateModeButtonAvailability();

    // Paper account
    if (data.paper) {
        const paperKey = document.getElementById('paper-api-key');
        const paperSecret = document.getElementById('paper-api-secret');
        if (paperKey && data.paper.api_key) {
            paperKey.placeholder = '已有密钥（已隐藏）';
        }
        if (paperSecret && data.paper.has_secret) {
            paperSecret.placeholder = '已有密钥（已隐藏）';
        }
    }
    // Live account
    if (data.live) {
        const liveKey = document.getElementById('live-api-key');
        const liveSecret = document.getElementById('live-api-secret');
        if (liveKey && data.live.api_key) {
            liveKey.placeholder = '已有密钥（已隐藏）';
        }
        if (liveSecret && data.live.has_secret) {
            liveSecret.placeholder = '已有密钥（已隐藏）';
        }
    }
}

function updateModeButtonAvailability() {
    document.querySelectorAll('.mode-btn[data-mode="live"]').forEach(button => {
        const configured = state.okxConfig?.liveConfigured === true;
        const knownMissing = state.okxConfig?.liveConfigured === false;
        button.classList.toggle('needs-config', knownMissing);
        button.title = configured
            ? '切换到 OKX 实盘账户'
            : knownMissing
                ? '实盘 OKX API 未配置完整，点击后会跳转到系统设置。'
                : '正在读取 OKX 实盘配置，后端会在切换前再次校验。';
    });
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setInputValue(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = value === null || value === undefined ? '' : value;
}

function readNumberInput(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    const raw = String(el.value || '').trim();
    if (!raw) return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
}

function riskFloorFromAccount(account) {
    const accountEquity = valueNumber(account?.account_equity ?? account?.okx_equity_balance ?? account?.equity ?? account?.allocated_balance) || 0;
    const maxLossUsdt = valueNumber(account?.max_loss_usdt) || 0;
    const maxLossPct = valueNumber(account?.max_loss_pct) || 0;
    if (valueNumber(account?.risk_floor) !== null) return valueNumber(account.risk_floor);
    if (accountEquity <= 0) return 0;
    const loss = maxLossUsdt > 0 ? maxLossUsdt : accountEquity * maxLossPct;
    return Math.max(accountEquity - loss, 0);
}

function renderExecutionAccountSettings(data) {
    const paper = data?.paper || {};
    const live = data?.live || {};
    const accountName = paper.account_name || live.account_name || '多专家执行账户';
    setInputValue('exec-account-name', accountName);

    [
        ['paper', paper],
        ['live', live],
    ].forEach(([mode, account]) => {
        const displayAvailable = valueNumber(account.okx_available_balance ?? account.available_balance);
        const displayEquity = valueNumber(account.okx_equity_balance ?? account.equity ?? account.account_equity ?? account.okx_total_balance);
        const availableText = account.balance_error
            ? account.balance_error
            : `${fmtMoney(displayAvailable)} USDT`;
        setText(`${mode}-current-available`, availableText);
        setText(
            `${mode}-account-equity`,
            account.balance_error ? '-- USDT' : `${fmtMoney(displayEquity)} USDT`
        );
        setText(`${mode}-cumulative-loss`, `${fmtMoney(account.cumulative_loss ?? account.realized_loss)} USDT`);
        setText(`${mode}-cumulative-profit`, `${fmtMoney(account.cumulative_profit ?? account.realized_profit)} USDT`);

        const maxLossPct = valueNumber(account.max_loss_pct);
        setInputValue(
            `exec-${mode}-max-loss-pct`,
            maxLossPct !== null ? (maxLossPct * 100).toFixed(0) : ''
        );
        const cooldownPct = valueNumber(account.cooldown_loss_pct);
        setInputValue(
            `exec-${mode}-cooldown-loss-pct`,
            cooldownPct !== null ? (cooldownPct * 100).toFixed(0) : ''
        );
    });
}

async function fetchExecutionAccountSettings() {
    const data = await fetchJSON('/api/settings/execution-account');
    if (!data) return;
    renderExecutionAccountSettings(data);
}

async function saveExecutionAccountSettings() {
    const status = document.getElementById('execution-account-save-status');
    if (status) {
        status.textContent = '保存中...';
        status.style.color = 'var(--text-muted)';
    }

    const accountName = (document.getElementById('exec-account-name')?.value || '').trim();
    for (const mode of ['paper', 'live']) {
        const maxLossPct = readNumberInput(`exec-${mode}-max-loss-pct`);
        if (maxLossPct !== null && (maxLossPct < 0 || maxLossPct > 100)) {
            if (status) {
                status.textContent = '保存失败: 最高可亏损比例必须在 0 到 100 之间';
                status.style.color = 'var(--red)';
            }
            return;
        }
        const body = {
            mode,
            account_name: accountName,
        };
        if (maxLossPct !== null) body.max_loss_pct = maxLossPct / 100;
        const cooldownPct = readNumberInput(`exec-${mode}-cooldown-loss-pct`);
        if (cooldownPct !== null) {
            if (cooldownPct < 0 || cooldownPct > 100) {
                if (status) {
                    status.textContent = '保存失败: 冷静期触发比例必须在 0 到 100 之间';
                    status.style.color = 'var(--red)';
                }
                return;
            }
            body.cooldown_loss_pct = cooldownPct / 100;
        }
        Object.keys(body).forEach(key => {
            if (body[key] === null || body[key] === undefined || body[key] === '') delete body[key];
        });

        const res = await fetchWithAuth('/api/settings/execution-account', dashboardWriteOptions({
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        }));
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            if (status) {
                status.textContent = '保存失败: ' + apiErrorText(err);
                status.style.color = 'var(--red)';
            }
            return;
        }
    }

    if (status) {
        status.textContent = '已保存';
        status.style.color = 'var(--green)';
    }
    await fetchExecutionAccountSettings();
    await fetchDashboardSummary();
}

async function saveOKXSettings(mode) {
    const prefix = mode === 'live' ? 'live' : 'paper';
    const apiKey = document.getElementById(prefix + '-api-key').value.trim();
    const apiSecret = document.getElementById(prefix + '-api-secret').value.trim();
    const passphrase = document.getElementById(prefix + '-passphrase').value.trim();

    const body = { mode };
    if (apiKey && !apiKey.startsWith('****')) body.api_key = apiKey;
    if (apiSecret && !apiSecret.startsWith('****')) body.api_secret = apiSecret;
    if (passphrase) body.passphrase = passphrase;

    const res = await fetchWithAuth('/api/settings/okx', dashboardWriteOptions({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }));

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('保存失败: ' + (err.detail || '未知错误'));
        return;
    }
    alert(mode === 'live' ? '实盘设置已保存' : '模拟盘设置已保存');
    fetchExecutionAccountSettings();
}

async function testOKXConnection(mode) {
    const prefix = mode === 'live' ? 'live' : 'paper';
    const btn = document.getElementById('test-' + prefix + '-btn');
    const status = document.getElementById(prefix + '-conn-status');
    if (!btn || !status) return;

    btn.disabled = true;
    btn.textContent = '测试中...';
    status.textContent = '';
    status.className = '';

    const res = await fetchWithAuth('/api/settings/okx/balance', { cache: 'no-store' });
    const data = await res.json().catch(() => ({}));
    if (data && !data.error) data.error = apiErrorText(data);
    const modeError = data[`${mode}_error`];
    const modeBalance = data[mode];
    data.success = res.ok && !modeError && modeBalance !== null && modeBalance !== undefined;
    if (!data.success) data.error = modeError || data.error || apiErrorText(data);
    if (data.success) data.message = `可用余额 ${fmtMoney(modeBalance)} USDT`;

    btn.disabled = false;
    btn.textContent = '测试连接';
    if (data.success) {
        status.textContent = '连接成功';
        status.className = 'conn-ok';
        fetchExecutionAccountSettings();
    } else {
        status.textContent = '连接失败: ' + (data.error || '未知错误');
        status.className = 'conn-fail';
    }
}

// ========== AI Model CRUD ==========
let currentModelMode = 'paper';

async function testModelByName(name) {
    const btn = event && event.target;
    if (btn && btn.tagName === 'BUTTON') {
        btn.disabled = true;
        btn.textContent = '...';
    }

    const res = await fetchWithAuth('/api/settings/ai-models/test', dashboardWriteOptions({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    }));
    const data = await res.json().catch(() => ({}));
    if (data && !data.error) data.error = apiErrorText(data);

    if (btn && btn.tagName === 'BUTTON') {
        btn.disabled = false;
        btn.textContent = '🔍';
    }

    if (data.success) {
        alert('连接成功: ' + data.message);
    } else {
        alert('连接失败: ' + (data.error || '未知错误'));
    }
}

// Fixed expert model UI overrides. The older CRUD handlers remain above for
// compatibility, but these definitions are the active ones.
async function fetchAIModels() {
    const cached = localStorage.getItem('aiExpertModelsCache');
    if (cached && !state.aiExpertModels.length) {
        try {
            const cachedModels = JSON.parse(cached);
            if (Array.isArray(cachedModels) && cachedModels.length) {
                state.aiExpertModels = cachedModels;
                renderModelList(cachedModels);
            }
        } catch (_) {}
    }

    if (!state.aiExpertModels.length) {
        renderModelList(FIXED_AI_EXPERT_FALLBACKS.map(m => ({ ...m, loading: true })));
    }

    const data = await fetchJSON('/api/settings/ai-models');
    if (!data) return;

    const models = data.models || [];
    state.aiExpertModels = models;
    localStorage.setItem('aiExpertModelsCache', JSON.stringify(models));
    state.modelModeMap = {};
    models.forEach(m => { state.modelModeMap[m.name] = state.mode || 'paper'; });
    renderModelList(models);

    const balanceEl = document.getElementById('okx-balance-info');
    if (balanceEl) {
        const exec = data.execution_account?.[state.mode] || data.execution_account?.paper || {};
        const parts = [
            `执行账户: <strong>${escHtml(exec.account_name || '多专家执行账户')}</strong>`,
            `内部执行器: <strong>${escHtml(data.execution_model || 'ensemble_trader')}</strong>`,
            '余额和风控额度请在“OKX 账户”的执行账户设置中维护',
        ];
        balanceEl.innerHTML = parts.join(' | ');
    }
}

function renderModelList(models) {
    const tbody = document.getElementById('model-config-tbody');
    if (!tbody) return;

    if (!models.length) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center;padding:24px;">固定专家模型加载中...</td></tr>';
        return;
    }

    tbody.innerHTML = models.map(m => {
        const loading = m.loading === true;
        const keyState = m.api_key
            ? '<span style="color:var(--green);font-size:11px;">已设置</span>'
            : `<span style="color:var(--text-muted);font-size:11px;">${loading ? '加载中' : '未设置'}</span>`;
        const actionButtons = loading
            ? '<button class="btn btn-sm" disabled title="配置加载中">编辑</button><button class="btn btn-sm" disabled title="配置加载中">测试</button>'
            : `<button class="btn btn-sm" onclick="editModel(${jsStringAttr(m.name)})" title="编辑">编辑</button>
                <button class="btn btn-sm" onclick="testModelByName(${jsStringAttr(m.name)})" title="测试连接">测试</button>`;
        return `
        <tr>
            <td>
                <strong>${escHtml(m.label || m.name)}</strong>
                <div style="font-size:10px;color:var(--text-muted);">${escHtml(m.name)}</div>
            </td>
            <td style="font-size:11px;color:var(--text-muted);max-width:260px;">${escHtml(m.description || m.role || '-')}</td>
            <td style="font-size:11px;color:var(--text-muted);">${loading ? '读取中...' : escHtml(m.api_base || '-')}</td>
            <td>${loading ? '读取中...' : escHtml(m.model || '-')}</td>
            <td>${keyState}</td>
            <td>${actionButtons}</td>
        </tr>
    `}).join('');
}

function showAddModelForm() {
    alert('模型槽位已固定，请直接编辑列表里的专家模型。');
}

function editModel(name) {
    const m = (state.aiExpertModels || []).find(x => x.name === name);
    if (!m) { alert('模型配置还在加载，请稍后再试'); return; }

    document.getElementById('model-modal-title').textContent = `编辑 ${m.label || m.name}`;
    document.getElementById('model-edit-orig-name').value = name;
    document.getElementById('model-cfg-name').value = m.name || '';
    document.getElementById('model-cfg-api-base').value = m.api_base || '';
    document.getElementById('model-cfg-api-key').value = '';
    document.getElementById('model-cfg-api-key').placeholder = m.api_key ? '已有密钥（已隐藏），留空不变' : '请输入密钥';
    document.getElementById('model-cfg-model').value = m.model || '';
    document.getElementById('model-save-btn').textContent = '保存';
    document.getElementById('model-modal-overlay').style.display = 'flex';
}

async function saveModelConfig() {
    const origName = document.getElementById('model-edit-orig-name').value.trim();
    const body = {
        name: document.getElementById('model-cfg-name').value.trim(),
        api_base: document.getElementById('model-cfg-api-base').value.trim(),
        api_key: document.getElementById('model-cfg-api-key').value.trim(),
        model: document.getElementById('model-cfg-model').value.trim(),
        execution_mode: 'analysis',
    };

    if (!origName || !body.name) { alert('请选择要编辑的专家模型'); return; }

    const res = await fetchWithAuth(`/api/settings/ai-models/${encodeURIComponent(origName)}`, dashboardWriteOptions({
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }));

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('保存失败: ' + (err.detail || '未知错误'));
        return;
    }

    closeModelModal();
    fetchAIModels();
    alert('模型已更新');
}

async function deleteModel(name) {
    alert('固定专家模型不能删除，只能清空 Key 或修改模型配置。');
}

// ========== Record Page Overrides ==========

async function fetchPositions() {
    const requestToken = ++positionsRequestToken;
    const data = await fetchJSON(`/api/dashboard/positions?mode=${state.mode}&page=${state.positionsPage}&page_size=${PAGE_SIZE}&open_only=true`);
    if (!data) return;
    if (requestToken !== positionsRequestToken) return;
    state.positionsPage = data.page || state.positionsPage;
    state.positionsTotal = data.total || 0;
    renderOpenPositionsTable(data.positions || [], state.positionsPage, data.total_pages || 1, data.total || 0);
    const badge = document.getElementById('position-badge');
    if (badge) {
        const total = Number(data.total ?? data.count ?? 0);
        badge.textContent = total;
        badge.style.display = total > 0 ? '' : 'none';
    }
}

async function fetchPositionHistory() {
    const data = await fetchJSON(`/api/dashboard/positions?mode=${state.mode}&page=${state.positionHistoryPage}&page_size=${PAGE_SIZE}&closed_only=true`);
    if (!data) return;
    state.positionHistoryPage = data.page || state.positionHistoryPage;
    state.positionHistoryTotal = data.total || 0;
    renderClosedPositionsTable(data.positions || [], state.positionHistoryPage, data.total_pages || 1, data.total || 0);
}

function renderOpenPositionsTable(positions, page = 1, totalPages = 1, totalItems = 0) {
    const tbody = document.getElementById('positions-tbody');
    const pagination = document.getElementById('positions-pagination');
    if (!tbody) return;
    if (!positions.length) {
        tbody.innerHTML = '<tr><td colspan="11" style="color:var(--text-muted);text-align:center;padding:24px;">暂无正在持仓数据</td></tr>';
        if (pagination) pagination.style.display = 'none';
        return;
    }
    tbody.innerHTML = positions.map(p => {
        const pnl = Number(p.unrealized_pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const positionId = Number(p.id || 0);
        const splitCount = Number(p.split_count || 1);
        const canManualClose = p.can_manual_close !== false && positionId > 0;
        const closeDisabled = closingAllPositions || closingPositionIds.has(positionId) || !canManualClose;
        const quantityMeta = splitCount > 1
            ? `<div style="font-size:10px;color:var(--text-muted);margin-top:2px;">${splitCount} \u4e2a\u5206\u7247</div>`
            : '';
        const closeLabel = closingPositionIds.has(positionId) ? '平仓中...' : '平仓';
        const closeButtonAttrs = [
            'class="btn btn-sm js-close-position"',
            closeDisabled ? 'disabled' : '',
            `data-position-id="${escHtml(String(positionId))}"`,
            `data-symbol="${escHtml(p.symbol || '')}"`,
            `data-side="${escHtml(p.side || '')}"`,
            'title="手动平掉该持仓"',
        ].filter(Boolean).join(' ');
        return `
        <tr>
            <td>${escHtml(p.symbol || '-')}</td>
            <td><span style="color:${p.side === 'long' ? 'var(--green)' : 'var(--red)'}">${sideLabel(p.side)}</span></td>
            <td>${Number(p.leverage || 1).toFixed(1)}x</td>
            <td>${fmtNum(p.quantity)}${quantityMeta}</td>
            <td>${fmtPrice(p.entry_price)}</td>
            <td>${fmtPrice(p.current_price || p.entry_price)}</td>
            <td style="color:${pnlColor};font-weight:600;">${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)}</td>
            <td>${p.take_profit ? fmtPrice(p.take_profit) : '-'}</td>
            <td>${p.stop_loss ? fmtPrice(p.stop_loss) : '-'}</td>
            <td style="font-size:10px;color:var(--text-muted);">${toBeijingTime(p.opened_at)}</td>
            <td><button ${closeButtonAttrs}>${closeLabel}</button></td>
        </tr>`;
    }).join('');
    renderPagination('positions-pagination', page, totalPages, totalItems, 'changePositionsPage');
}


function renderClosedPositionsTable(positions, page = 1, totalPages = 1, totalItems = 0) {
    const tbody = document.getElementById('position-history-tbody');
    const pagination = document.getElementById('position-history-pagination');
    if (!tbody) return;
    if (!positions.length) {
        tbody.innerHTML = '<tr><td colspan="10" style="color:var(--text-muted);text-align:center;padding:24px;">\u6682\u65e0\u5386\u53f2\u6301\u4ed3\u6570\u636e</td></tr>';
        if (pagination) pagination.style.display = 'none';
        return;
    }
    tbody.innerHTML = positions.map(p => {
        const pnl = Number(p.realized_pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const statusLabel = p.close_status_label || p.position_status || (p.close_status === 'partial' ? '\u90e8\u5206\u5e73\u4ed3' : '\u5168\u90e8\u5e73\u4ed3');
        const statusColor = p.close_status === 'partial' ? 'var(--accent-light)' : 'var(--text-muted)';
        return `
        <tr>
            <td>${escHtml(p.symbol || '-')}</td>
            <td><span style="color:${p.side === 'long' ? 'var(--green)' : 'var(--red)'}">${sideLabel(p.side)}</span></td>
            <td><span style="color:${statusColor};font-weight:600;">${escHtml(statusLabel)}</span></td>
            <td>${Number(p.leverage || 1).toFixed(1)}x</td>
            <td>${fmtNum(p.quantity)}</td>
            <td>${fmtPrice(p.entry_price)}</td>
            <td>${fmtPrice(p.current_price || p.entry_price)}</td>
            <td style="color:${pnlColor};font-weight:600;">${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)}</td>
            <td style="font-size:10px;color:var(--text-muted);">${toBeijingTime(p.opened_at)}</td>
            <td style="font-size:10px;color:var(--text-muted);">${toBeijingTime(p.closed_at)}</td>
        </tr>`;
    }).join('');
    renderPagination('position-history-pagination', page, totalPages, totalItems, 'changePositionHistoryPage');
}

function initPositionActions() {
    const tbody = document.getElementById('positions-tbody');
    if (!tbody || tbody.dataset.closeHandlerAttached === '1') return;
    tbody.dataset.closeHandlerAttached = '1';
    tbody.addEventListener('click', (event) => {
        const button = event.target?.closest?.('.js-close-position');
        if (!button || !tbody.contains(button)) return;
        event.preventDefault();
        closeOpenPosition(
            Number(button.dataset.positionId || 0),
            button.dataset.symbol || '',
            button.dataset.side || ''
        );
    });
}

async function closeOpenPosition(positionId, symbol, side) {
    if (!positionId || closingPositionIds.has(positionId) || closingAllPositions) return;
    const sideText = sideLabel(side);
    if (!confirm(`\u786e\u8ba4\u624b\u52a8\u5e73\u4ed3 ${symbol || '-'} ${sideText} \u5417\uff1f`)) return;
    closingPositionIds.add(positionId);
    fetchPositions();
    try {
        const data = await postJSON(`/api/positions/${positionId}/close`, {
            reason: '\u7528\u6237\u5728\u6301\u4ed3\u8bb0\u5f55\u9875\u9762\u624b\u52a8\u70b9\u51fb\u5e73\u4ed3\u3002',
        });
        if (!data.approved) {
            alert('\u5e73\u4ed3\u672a\u6267\u884c: ' + (data.rejection_reason || '\u672a\u77e5\u539f\u56e0'));
        }
    } catch (err) {
        alert('\u5e73\u4ed3\u5931\u8d25: ' + (err.message || '\u672a\u77e5\u9519\u8bef'));
    } finally {
        closingPositionIds.delete(positionId);
        await fetchPositions();
        fetchTrades();
        fetchDashboardSummary();
    }
}

async function closeAllOpenPositions() {
    if (closingAllPositions) return;
    const count = Number(state.positionsTotal || 0);
    const suffix = count > 0 ? `\u5f53\u524d\u7ea6 ${count} \u6761\u6301\u4ed3\u3002` : '';
    const modeLabel = state.mode === 'live' ? '\u5b9e\u76d8' : '\u6a21\u62df\u76d8';
    if (!confirm(`\u786e\u8ba4\u4e00\u952e\u5e73\u6389\u5f53\u524d${modeLabel}\u5168\u90e8\u6301\u4ed3\u5417\uff1f${suffix}`)) return;
    closingAllPositions = true;
    const btn = document.getElementById('close-all-positions-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = '\u5e73\u4ed3\u4e2d...';
    }
    fetchPositions();
    try {
        const data = await postJSON('/api/positions/close-all', {
            mode: state.mode || 'paper',
            reason: '\u7528\u6237\u5728\u6301\u4ed3\u8bb0\u5f55\u9875\u9762\u70b9\u51fb\u4e00\u952e\u5e73\u4ed3\u3002',
        });
        if (data.failed > 0) {
            alert(`\u4e00\u952e\u5e73\u4ed3\u5b8c\u6210 ${data.closed || 0} \u6761\uff0c\u5931\u8d25 ${data.failed} \u6761\u3002`);
        } else {
            alert(`\u4e00\u952e\u5e73\u4ed3\u5df2\u63d0\u4ea4 ${data.closed || 0} \u6761\u3002`);
        }
    } catch (err) {
        alert('\u4e00\u952e\u5e73\u4ed3\u5931\u8d25: ' + (err.message || '\u672a\u77e5\u9519\u8bef'));
    } finally {
        closingAllPositions = false;
        if (btn) {
            btn.disabled = false;
            btn.textContent = '一键平仓';
        }
        await fetchPositions();
        fetchTrades();
        fetchDashboardSummary();
    }
}

function leverageDetailCell(item) {
    const actual = Number(item.actual_leverage ?? item.leverage ?? 1);
    return `
        <div style="font-weight:700;color:var(--text);">${actual.toFixed(1)}x</div>
    `;
}

function renderDailyPnlRecords(records) {
    const tbody = document.getElementById('daily-pnl-tbody');
    if (!tbody) return;
    if (!records.length) {
        tbody.innerHTML = '<tr><td colspan="8" style="color:var(--text-muted);text-align:center;padding:24px;">暂无每日盈亏记录</td></tr>';
        return;
    }
    tbody.innerHTML = records.map(row => {
        const realized = Number(row.realized_pnl || 0);
        const total = Number(row.total_pnl || 0);
        const cumulative = Number(row.cumulative_total_pnl ?? row.cumulative_realized_pnl ?? 0);
        const winLoss = `${Number(row.win_count || 0)}胜 / ${Number(row.loss_count || 0)}亏`;
        const symbolCount = Array.isArray(row.symbol_pnl)
            ? row.symbol_pnl.length
            : (Array.isArray(row.symbols) ? row.symbols.length : 0);
        const detailCount = Array.isArray(row.position_details) ? row.position_details.length : 0;
        return `
        <tr>
            <td style="font-weight:700;white-space:nowrap;">${escHtml(row.date || '-')}</td>
            <td style="color:var(--red);">${fmtMoney(row.realized_loss || 0)} USDT</td>
            <td style="color:var(--green);">${fmtMoney(row.realized_profit || 0)} USDT</td>
            <td style="color:${realized >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:700;">${signedMoney(realized)} USDT</td>
            <td style="color:${total >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:700;">${signedMoney(total)} USDT</td>
            <td style="color:${cumulative >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(cumulative)} USDT</td>
            <td>${Number(row.trade_count || 0)} <span style="color:var(--text-muted);font-size:10px;">${winLoss}</span></td>
            <td>
                <button class="btn btn-sm js-daily-pnl-detail" data-date="${escHtml(row.date || '')}">
                    ${detailCount ? `查看 ${detailCount} 笔` : (symbolCount ? `查看 ${symbolCount} 个币种` : '查看详情')}
                </button>
            </td>
        </tr>`;
    }).join('');
}

function openDailyPnlModal(date) {
    const row = (state.dailyPnlRecords || []).find(item => item.date === date);
    if (!row) return;
    const title = document.getElementById('daily-pnl-modal-title');
    const body = document.getElementById('daily-pnl-modal-body');
    const overlay = document.getElementById('daily-pnl-modal-overlay');
    if (!title || !body || !overlay) return;

    const details = Array.isArray(row.symbol_pnl) ? row.symbol_pnl : [];
    const positionDetails = Array.isArray(row.position_details) ? row.position_details : [];
    const total = Number(row.total_pnl || 0);
    title.textContent = `${date} 盈亏详情（北京时间）`;
    if (!details.length && !positionDetails.length) {
        const hasOverview = Number(row.trade_count || 0) > 0
            || Number(row.realized_pnl || 0) !== 0
            || Number(row.unrealized_pnl || 0) !== 0
            || Number(row.total_pnl || 0) !== 0;
        body.innerHTML = hasOverview
            ? `<div style="color:var(--text-muted);font-size:12px;padding:8px;">当日有盈亏汇总，但没有按币种拆分明细。可能是历史记录未保存 symbol_pnl，或该日只保留了总览数据。</div>
               <div class="daily-pnl-modal-summary">
                   <div>已平仓净盈亏 <strong style="color:${Number(row.realized_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(row.realized_pnl || 0)} USDT</strong></div>
                   <div>当日总盈亏 <strong style="color:${total >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(total)} USDT</strong></div>
                   <div>交易笔数 <strong>${Number(row.trade_count || 0)}</strong></div>
               </div>`
            : '<div style="color:var(--text-muted);font-size:12px;padding:8px;">当日没有已平仓交易。</div>';
        overlay.style.display = 'flex';
        return;
    }
    body.innerHTML = `
        <div class="daily-pnl-modal-summary">
            <div>已平仓净盈亏 <strong style="color:${Number(row.realized_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(row.realized_pnl || 0)} USDT</strong></div>
            <div>当日总盈亏 <strong style="color:${total >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(total)} USDT</strong></div>
            <div>交易笔数 <strong>${Number(row.trade_count || 0)}</strong></div>
        </div>
        ${positionDetails.length ? renderDailyPnlPositionDetails(positionDetails) : ''}
        <div class="table-wrap" style="margin-top:10px;">
            <table>
                <thead>
                    <tr>
                        <th>币种</th>
                        <th>净盈亏</th>
                        <th>盈利合计</th>
                        <th>亏损合计</th>
                        <th>交易数</th>
                        <th>胜 / 亏</th>
                    </tr>
                </thead>
                <tbody>
                    ${details.map(item => {
                        const pnl = Number(item.realized_pnl || 0);
                        return `
                            <tr>
                                <td style="font-weight:700;">${escHtml(item.symbol || '-')}</td>
                                <td style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:700;">${signedMoney(pnl)} USDT</td>
                                <td style="color:var(--green);">${fmtMoney(item.realized_profit || 0)} USDT</td>
                                <td style="color:var(--red);">${fmtMoney(item.realized_loss || 0)} USDT</td>
                                <td>${Number(item.trade_count || 0)}</td>
                                <td>${Number(item.win_count || 0)} / ${Number(item.loss_count || 0)}</td>
                            </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
    overlay.style.display = 'flex';
}

function renderDailyPnlPositionDetails(positionDetails) {
    return `
        <div class="table-wrap" style="margin-top:10px;">
            <table>
                <thead>
                    <tr>
                        <th>时间</th>
                        <th>币种</th>
                        <th>方向</th>
                        <th>数量</th>
                        <th>开仓价</th>
                        <th>平仓价</th>
                        <th>已实现盈亏</th>
                    </tr>
                </thead>
                <tbody>
                    ${positionDetails.map(item => {
                        const pnl = Number(item.realized_pnl || 0);
                        return `
                            <tr>
                                <td>${toBeijingTime(item.closed_at)}</td>
                                <td style="font-weight:700;">${escHtml(item.symbol || '-')}</td>
                                <td>${escHtml(item.side_label || sideLabel(item.side) || '-')}</td>
                                <td>${Number(item.quantity || 0).toFixed(6)}</td>
                                <td>${fmtPrice(item.entry_price)}</td>
                                <td>${fmtPrice(item.exit_price)}</td>
                                <td style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:700;">${signedMoney(pnl)} USDT</td>
                            </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function closeDailyPnlModal() {
    const overlay = document.getElementById('daily-pnl-modal-overlay');
    if (overlay) overlay.style.display = 'none';
}

function updateTradeTable(trades, mode, total) {
    state.allTrades = trades || [];
    state.tradesPageMode = mode || state.tradeMode;
    state.tradesTotal = Number(total ?? state.allTrades.length);
    const badge = document.getElementById('trade-badge');
    if (badge) badge.textContent = state.tradesTotal;
    renderTradePage();
}

function renderTradePage() {
    const tbody = document.getElementById('trades-tbody');
    if (!tbody) return;
    const filtered = state.tradesPageMode
        ? state.allTrades.filter(t => t.mode === state.tradesPageMode)
        : state.allTrades;
    if (!filtered.length) {
        const modeLabel = state.tradesPageMode === 'live' ? '实盘' : '模拟盘';
        tbody.innerHTML = `<tr><td colspan="9" style="color:var(--text-muted);text-align:center;padding:24px;">暂无${modeLabel}执行记录</td></tr>`;
        document.getElementById('trades-pagination').style.display = 'none';
        return;
    }
    const totalPages = Math.ceil((state.tradesTotal || filtered.length) / PAGE_SIZE);
    const page = Math.min(state.tradesPage, totalPages);
    const pageData = filtered;
    tbody.innerHTML = pageData.map(t => {
        const time = t.filled_at || t.created_at || '';
        const success = t.success === true || t.status === 'filled';
        const statusInfo = executionStatusPresentation(t, success);
        const sourceLabel = t.execution_source_label || (t.execution_source === 'okx' ? 'OKX执行' : '系统执行');
        const sourceColor = t.execution_source === 'okx' ? 'var(--accent-light)' : 'var(--text-muted)';
        return `
        <tr>
            <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(time)}</td>
            <td>${escHtml(t.display_symbol || t.symbol || '-')}</td>
            <td>${executionActionCell(t)}</td>
            <td>${leverageDetailCell(t)}</td>
            <td>${fmtNum(t.quantity)}</td>
            <td>${fmtPrice(t.price)}</td>
            <td style="color:${statusInfo.color};font-weight:600;">${escHtml(statusInfo.label)}</td>
            <td style="color:${sourceColor};font-weight:600;">${escHtml(sourceLabel)}</td>
            <td><button class="btn btn-sm" onclick="showExecutionDetail(${Number(t.id)})">查看</button></td>
        </tr>`;
    }).join('');
    renderPagination('trades-pagination', page, totalPages, state.tradesTotal || filtered.length, 'changeTradePage');
}

function renderDecisionsPage(totalPagesOverride = null) {
    const tbody = document.getElementById('all-decisions-tbody');
    if (!tbody) return;
    if (!state.allDecisions.length) {
        tbody.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center;padding:24px;">暂无决策记录</td></tr>';
        document.getElementById('decisions-pagination').style.display = 'none';
        return;
    }
    const totalPages = Number(totalPagesOverride || Math.ceil(state.decisionsTotal / PAGE_SIZE) || 1);
    const page = Math.min(state.decisionsPage, totalPages);
    const pageData = state.allDecisions;
    tbody.innerHTML = pageData.map(d => {
        const confPct = ((d.confidence || 0) * 100).toFixed(0);
        const executedHtml = d.was_executed
            ? '<span style="color:var(--green);font-weight:600;">是</span>'
            : '<span style="color:var(--text-dim);">否</span>';
        const reasonBtn = d.was_executed
            ? '<span style="color:var(--text-muted);">-</span>'
            : `<button class="btn btn-sm" onclick="showDecisionReason(${Number(d.id)})">查看</button>`;
        return `
        <tr>
            <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(d.created_at)}</td>
            <td>${escHtml(d.symbol || '-')}</td>
            <td><span class="badge badge-${d.action || 'hold'}">${analysisActionLabel(d.action, d)}</span></td>
            <td style="color:${(d.confidence || 0) >= 0.65 ? 'var(--green)' : 'var(--text-muted)'};font-weight:600;">${confPct}%</td>
            <td class="decision-size-cell">${decisionSizeCell(d)}</td>
            <td>${executedHtml}</td>
            <td>${reasonBtn}</td>
        </tr>`;
    }).join('');
    renderPagination('decisions-pagination', page, totalPages, state.decisionsTotal, 'changeDecisionsPage');
}

function changePositionsPage(page) {
    state.positionsPage = page;
    fetchPositions();
}

function changePositionHistoryPage(page) {
    state.positionHistoryPage = page;
    fetchPositionHistory();
}

// Override the older pagination renderer so record pages always show clear,
// valid controls even when legacy text in the bundle is garbled.
function renderPagination(containerId, page, totalPages, totalItems, callbackName) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const callback = safePaginationCallbackName(callbackName);
    if (!callback) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    const currentPage = Math.max(1, Number(page || 1));
    const pages = Math.max(1, Number(totalPages || 1));
    const total = Math.max(0, Number(totalItems || 0));
    if (pages <= 1) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }

    container.style.display = 'flex';
    let startP = Math.max(1, currentPage - 3);
    let endP = Math.min(pages, currentPage + 3);
    if (endP - startP < 6) {
        if (startP === 1) endP = Math.min(pages, startP + 6);
        else startP = Math.max(1, endP - 6);
    }

    let html = '';
    html += `<button onclick="${callback}(1)" ${currentPage <= 1 ? 'disabled' : ''}>首页</button>`;
    html += `<button onclick="${callback}(${currentPage - 1})" ${currentPage <= 1 ? 'disabled' : ''}>上一页</button>`;
    for (let p = startP; p <= endP; p++) {
        html += `<button onclick="${callback}(${p})" ${p === currentPage ? 'class="active"' : ''}>${p}</button>`;
    }
    html += `<button onclick="${callback}(${currentPage + 1})" ${currentPage >= pages ? 'disabled' : ''}>下一页</button>`;
    html += `<button onclick="${callback}(${pages})" ${currentPage >= pages ? 'disabled' : ''}>末页</button>`;
    html += `<span class="page-info">共 ${total} 条 / ${pages} 页</span>`;
    container.innerHTML = html;
}

function closeModelModal() {
    document.getElementById('model-modal-overlay').style.display = 'none';
}

// Execution detail renderer with readable leverage fields and sanitized failure text.
function cleanExecutionDetailText(value, fallback) {
    if (value === null || value === undefined || value === '') return fallback;
    const text = String(value);
    if (/(OKX|okx).*(code|sCode|返回码|51004|51155|51169|59670)/i.test(text)) {
        return text;
    }
    return isMojibakeText(text)
        ? '执行失败，原因文本编码异常。请以 OKX 返回码、执行记录原始响应和当前订单状态为准。'
        : text;
}

function translatePauseReason(value) {
    const text = String(value || '').trim();
    if (!text) return '账户触发风险限制';
    if (text.includes('Execution account reached max loss limit')) {
        const total = text.match(/total_pnl=([-0-9.]+)\s*USDT/i)?.[1] || '-';
        const maxLoss = text.match(/max_loss=([-0-9.]+)\s*USDT/i)?.[1] || '-';
        const pct = text.match(/\(([-0-9.]+)%\)/)?.[1] || '-';
        return `执行账户已达到最高亏损限制：当前累计盈亏 ${total} USDT，最高允许亏损 ${maxLoss} USDT（${pct}%）。暂停分析新的交易对。`;
    }
    if (text.includes('Risk circuit breaker is open')) {
        const reason = text.includes('reason=') ? text.split('reason=').pop() : '触发风险阈值';
        return `风险熔断已开启，暂停分析新的交易对。原因：${reason}`;
    }
    if (text.includes('OKX usable balance snapshot is unavailable')) {
        return '未获取到 OKX 可用余额快照，暂停分析新的交易对。';
    }
    if (text.includes('OKX equity/balance is unavailable')) {
        return '未获取到 OKX 账户权益或余额，暂停分析新的交易对。';
    }
    if (text.includes('OKX tradable balance is too low')) {
        const available = text.match(/available=([-0-9.]+)\s*USDT/i)?.[1] || '-';
        const required = text.match(/minimum_required=([-0-9.]+)\s*USDT/i)?.[1] || '-';
        return `OKX 可交易余额过低：当前可用 ${available} USDT，最低需要 ${required} USDT，暂停分析新的交易对。`;
    }
    return text;
}

function executionStepTone(status) {
    const value = String(status || '').toLowerCase();
    if (['blocked', 'failed'].includes(value)) return 'bad';
    if (['skipped', 'pending'].includes(value)) return 'warn';
    return 'ok';
}

function executionStepDuration(step) {
    if (!step || step.duration_sec === null || step.duration_sec === undefined) {
        return '旧记录未采集耗时';
    }
    return analysisDurationLabel(step.duration_sec);
}

function executionStepDataText(data) {
    if (!data || typeof data !== 'object' || !Object.keys(data).length) return '';
    const formatRuleValue = (value, suffix = '') => {
        if (value === null || value === undefined || value === '') return '';
        if (typeof value === 'number' && Number.isFinite(value)) return `${value}${suffix}`;
        return `${value}${suffix}`;
    };
    const formatOkxRules = (rules) => {
        if (!rules || typeof rules !== 'object') return '';
        const rows = [
            ['OKX\u4ea4\u6613\u5bf9', rules.okx_symbol],
            ['\u5f53\u524d\u4ef7\u683c', rules.price],
            ['\u5408\u7ea6\u9762\u503c', rules.contract_size],
            ['\u6700\u5c0f\u5f20\u6570', rules.amount_min_contracts],
            ['\u4e0b\u5355\u6b65\u8fdb', rules.amount_step_contracts],
            ['\u6700\u5c0f\u540d\u4e49\u4ef7\u503c', formatRuleValue(rules.min_notional_usdt, ' USDT')],
            ['\u53ef\u7528\u4f59\u989d', formatRuleValue(rules.available_balance_usdt, ' USDT')],
            ['\u6760\u6746', formatRuleValue(rules.leverage, 'x')],
            ['\u53ef\u627f\u53d7\u540d\u4e49\u4ef7\u503c', formatRuleValue(rules.affordable_notional_usdt, ' USDT')],
            ['\u8ba1\u5212\u540d\u4e49\u4ef7\u503c', formatRuleValue(rules.planned_notional_usdt, ' USDT')],
            ['\u8ba1\u5212\u5f20\u6570', rules.planned_contracts_raw],
            ['\u6700\u7ec8\u5f20\u6570', rules.final_contracts],
            ['\u6700\u7ec8\u5e01\u6570', rules.final_base_quantity],
            ['\u6700\u7ec8\u540d\u4e49\u4ef7\u503c', formatRuleValue(rules.final_notional_usdt, ' USDT')],
            ['\u9884\u8ba1\u4fdd\u8bc1\u91d1', formatRuleValue(rules.required_margin_usdt, ' USDT')],
            ['\u662f\u5426\u62ac\u5230\u6700\u5c0f\u5f20\u6570', rules.system_adjusted_to_min_contracts ? '\u662f' : '\u5426'],
            ['\u63d0\u4ea4\u524d\u6821\u9a8c', rules.pre_submit_valid ? '\u901a\u8fc7' : '\u672a\u901a\u8fc7'],
        ].filter(([, value]) => value !== null && value !== undefined && value !== '');
        return rows.map(([label, value]) => `${label}: ${value}`).join('\n');
    };
    const labels = {
        source: '\u6765\u6e90',
        order_status: '\u8ba2\u5355\u72b6\u6001',
        blocker: '\u62e6\u622a\u7c7b\u578b',
        execution_blocker: '\u6267\u884c\u62e6\u622a\u5668',
        system_pre_submit_rejection: '\u7cfb\u7edf\u63d0\u4ea4\u524d\u62e6\u622a',
        okx_rejection: 'OKX\u5b9e\u9645\u62d2\u7edd',
        okx_order_rules: 'OKX\u4e0b\u5355\u89c4\u5219',
        okx_code: 'OKX \u8fd4\u56de\u7801',
        min_size: '\u6700\u5c0f\u6570\u91cf',
        min_notional: '\u6700\u5c0f\u540d\u4e49\u4ef7\u503c',
        requested_qty: '\u8bf7\u6c42\u6570\u91cf',
        adjusted_qty: '\u8c03\u6574\u540e\u6570\u91cf',
        available_balance: '\u53ef\u7528\u4f59\u989d',
        required_margin: '\u6240\u9700\u4fdd\u8bc1\u91d1',
        symbol: '\u4ea4\u6613\u5bf9',
        side: '\u65b9\u5411',
    };
    return Object.entries(data)
        .filter(([, value]) => value !== null && value !== undefined && value !== '')
        .map(([key, value]) => {
            if (key === 'okx_order_rules') {
                const rulesText = formatOkxRules(value);
                return rulesText ? `${labels[key]}:\n${rulesText}` : '';
            }
            return `${labels[key] || key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`;
        })
        .filter(Boolean)
        .join('\n');
}

function executionStepPlainReason(step) {
    const stage = String(step?.stage || '');
    const status = String(step?.status || '');
    const reason = String(step?.reason || '').trim();
    if (reason) return reason;
    if (status === 'passed' || status === 'completed') {
        const labels = {
            ai_analysis: 'AI 已完成本轮交易判断。',
            strategy_arbitration: '策略调度已完成候选排序与执行裁决。',
            risk_check: '风控检查通过，允许继续提交订单。',
            exchange_submit: '订单已提交到交易所。',
            exchange_confirm: '交易所成交确认已返回。',
            local_sync: '本地订单、持仓和收益记录已同步。',
        };
        return labels[stage] || '该步骤已完成。';
    }
    if (status === 'blocked') return '该步骤拦截了订单，系统没有继续向后执行。';
    if (status === 'failed') return '该步骤执行失败，请优先查看本步骤的原因和数据。';
    if (status === 'skipped') return '前置步骤未通过，该步骤被跳过。';
    return '该步骤没有返回额外说明。';
}

function renderExecutionTimeline(steps, failedStep) {
    const rows = Array.isArray(steps) ? steps : [];
    if (!rows.length) {
        return '<div class="reason-block">该记录没有执行步骤链；如果是旧记录，只能显示订单快照。</div>';
    }
    const failedKey = failedStep ? `${failedStep.stage || ''}:${failedStep.status || ''}` : '';
    return `<div class="execution-timeline">${rows.map((step, index) => {
        const status = step.status || '';
        const key = `${step.stage || ''}:${status}`;
        const tone = key === failedKey ? 'bad' : executionStepTone(status);
        const dataText = executionStepDataText(step.data);
        return `
            <div class="execution-step ${tone}">
                <div class="execution-step-head">
                    <span>${Number(step.step_no || index + 1)}. ${escHtml(step.stage_label || stateStageLabel(step.stage))}</span>
                    <span class="execution-step-meta">${escHtml(step.status_label || stateStatusLabel(status))} · ${escHtml(executionStepDuration(step))}</span>
                </div>
                <div class="execution-step-reason">${escapeMultiline(executionStepPlainReason(step))}</div>
                <div class="execution-step-meta">发生时间：${toBeijingTime(step.at)} · 阶段：${escHtml(step.stage || '-')}</div>
                ${dataText ? `<div class="execution-step-data">${escHtml(dataText)}</div>` : ''}
            </div>`;
    }).join('')}</div>`;
}

function renderExecutionDetailModal(trade, detailData = null) {
    setDecisionModalWide(false);
    const success = trade.success === true || trade.status === 'filled';
    const fallbackSource = trade.execution_source === 'okx' ? 'OKX执行' : '系统执行';
    const sourceLabel = trade.execution_source_label && !isMojibakeText(String(trade.execution_source_label))
        ? trade.execution_source_label
        : fallbackSource;
    const closeStatus = closeStatusLabel(trade);
    const actionTitle = closeStatus
        ? `${actionLabel(trade.action || trade.side)} / ${closeStatus}`
        : actionLabel(trade.action || trade.side);
    const detail = cleanExecutionDetailText(
        detailData?.display_reason || detailData?.detail || detailData?.reason || trade.display_reason || trade.detail || trade.reason,
        success ? '订单执行成功。' : '订单执行失败，暂无详细原因。'
    );
    const decision = detailData?.decision || trade.decision || {};
    const aiReason = cleanExecutionDetailText(
        decision.reasoning || trade.reasoning || '',
        ''
    );
    const executionReason = cleanExecutionDetailText(
        detailData?.display_reason || decision.execution_reason || detailData?.reason || trade.execution_reason || trade.reason || detail,
        detail
    );
    const aiLev = Number(trade.ai_suggested_leverage ?? trade.leverage ?? 1).toFixed(1);
    const actualLev = Number(trade.actual_leverage ?? trade.leverage ?? 1).toFixed(1);
    const holdHours = Number(trade.hold_hours);
    const holdTimeHtml = Number.isFinite(holdHours) && holdHours > 0
        ? `持仓时长：${holdHours >= 1 ? `${holdHours.toFixed(2)} 小时` : `${Number(trade.hold_minutes || 0).toFixed(0)} 分钟`}<br>`
        : '';

    const finalResult = detailData?.final_result || null;
    const failedStep = detailData?.failed_step || null;
    const repairSuggestions = Array.isArray(detailData?.repair_suggestions)
        ? detailData.repair_suggestions
        : [];
    const timelineHtml = detailData
        ? renderExecutionTimeline(detailData.execution_steps, failedStep)
        : '<div class="reason-block">正在读取每一步执行耗时和失败节点...</div>';
    const detailStatusInfo = executionStatusPresentation(
        { ...trade, ...(detailData || {}), final_result: finalResult },
        success
    );
    const finalTitle = finalResult?.success
        ? '执行成功'
        : (detailStatusInfo.isTransientExchange ? '交易所临时不可用' : '执行未完成/失败');
    const reasonLabel = success
        ? '执行原因'
        : (detailStatusInfo.isTransientExchange ? '临时故障原因' : '失败原因');
    const finalHtml = finalResult ? `
        <div class="reason-block">
            <div class="reason-label">最终结果</div>
            <div class="execution-result-grid">
                <div><strong>${escHtml(finalTitle)}</strong><br><span class="reason-meta">${escHtml(finalResult.stage_label || '-')}</span></div>
                <div><strong>${escHtml(finalResult.status_label || '-')}</strong><br><span class="reason-meta">最终状态</span></div>
                <div><strong>${escHtml(analysisDurationLabel(finalResult.total_duration_sec))}</strong><br><span class="reason-meta">总耗时</span></div>
            </div>
            ${finalResult.reason ? `<div style="margin-top:8px;">${escapeMultiline(finalResult.reason)}</div>` : ''}
        </div>` : '';
    const reasonHtml = `
        <div class="reason-block execution-reason-primary">
            <div class="reason-label">${reasonLabel}</div>
            <div>${escapeMultiline(executionReason || detail)}</div>
            ${aiReason ? `<div class="reason-meta">AI 裁决依据：${escapeMultiline(aiReason)}</div>` : ''}
        </div>`;
    const failedHtml = failedStep ? `
        <div class="reason-block">
            <div class="reason-label">问题定位</div>
            <div>卡在：${escHtml(failedStep.stage_label || stateStageLabel(failedStep.stage))} / ${escHtml(failedStep.status_label || stateStatusLabel(failedStep.status))} / 耗时 ${escHtml(executionStepDuration(failedStep))}</div>
            <div style="margin-top:6px;">${escapeMultiline(failedStep.reason || '该步骤未返回详细原因。')}</div>
        </div>` : '';
    const suggestionsHtml = repairSuggestions.length ? `
        <div class="reason-block">
            <div class="reason-label">处理建议</div>
            <div>${repairSuggestions.map(item => `• ${escHtml(item)}`).join('<br>')}</div>
        </div>` : '';

    document.getElementById('decision-reason-title').textContent =
        `${trade.display_symbol || trade.symbol || '-'} / ${actionTitle} / ${detailStatusInfo.label}`;
    document.getElementById('decision-reason-body').innerHTML = `
        ${finalHtml}
        ${reasonHtml}
        ${failedHtml}
        ${suggestionsHtml}
        <div class="reason-block">
            <div class="reason-label">执行步骤说明</div>
            <div class="reason-meta" style="margin:0 0 8px;">按实际执行顺序展示：每一步包含状态、耗时、发生时间和可读原因；如果某一步失败，系统会在“问题定位”中指出卡在哪一步。</div>
            ${timelineHtml}
        </div>
        <div class="reason-block">
            <div class="reason-label">${success ? '执行补充' : '失败补充'}</div>
            <div>${escapeMultiline(detail)}</div>
        </div>
        <div class="reason-block">
            <div class="reason-label">杠杆明细</div>
            <div>
                AI建议：${aiLev}x<br>
                实际下单：${actualLev}x
            </div>
        </div>
        <div class="reason-block">
            <div class="reason-label">订单信息</div>
            <div>
                执行时间：${toBeijingTime(trade.filled_at || trade.created_at)}<br>
                ${closeStatus ? `平仓类型：${escHtml(closeStatus)}<br>` : ''}
                ${holdTimeHtml}
                数量：${fmtNum(trade.quantity)}<br>
                价格：${fmtPrice(trade.price)}<br>
                来源：${escHtml(sourceLabel)}<br>
                状态：${statusLabel(trade.status)}
            </div>
        </div>`;
}

async function showExecutionDetail(tradeId) {
    const trade = state.allTrades.find(t => Number(t.id) === Number(tradeId));
    if (!trade) return;
    renderExecutionDetailModal(trade, null);
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex';
    const detail = await fetchJSON(`/api/trades/${encodeURIComponent(Number(tradeId))}`);
    if (!detail || detail.error) {
        const body = document.getElementById('decision-reason-body');
        if (body) {
            body.innerHTML = `<div class="reason-block"><div class="reason-label">详情加载失败</div><div>${escHtml(detail?.error || '未能读取执行步骤详情。')}</div></div>` + body.innerHTML;
        }
        return;
    }
    renderExecutionDetailModal({ ...trade, ...detail }, detail);
}

// Close modal on overlay click
document.addEventListener('click', (e) => {
    const analysisReasonButton = e.target?.closest?.('.js-analysis-reason');
    if (analysisReasonButton) {
        e.preventDefault();
        showAnalysisReason(
            analysisReasonButton.dataset.recordId,
            analysisReasonButton.dataset.decisionId
        );
        return;
    }
    const dailyPnlButton = e.target?.closest?.('.js-daily-pnl-detail');
    if (dailyPnlButton) {
        e.preventDefault();
        openDailyPnlModal(dailyPnlButton.dataset.date || '');
        return;
    }
    if (e.target.id === 'decision-reason-modal-overlay') {
        closeDecisionReasonModal();
    }
    if (e.target.id === 'daily-pnl-modal-overlay') {
        closeDailyPnlModal();
    }
    if (e.target.id === 'dashboard-user-modal-overlay') {
        closeDashboardUserModal();
    }
});

function escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function jsStringAttr(value) {
    return escHtml(JSON.stringify(String(value ?? '')));
}

const PAGINATION_CALLBACKS = new Set([
    'changePositionsPage',
    'changePositionHistoryPage',
    'changeTradePage',
    'changeDecisionsPage',
    'changeAnalysisPage',
    'changeRiskAlertPage',
    'changeExpertMemoryPage',
    'changeTradeReflectionPage',
    'changeShadowBacktestPage',
    'changeMLSignalPage',
    'changeProfitAttributionRecordPage',
]);

function safePaginationCallbackName(callbackName) {
    const value = String(callbackName || '');
    return PAGINATION_CALLBACKS.has(value) ? value : '';
}

function safeExternalUrl(value) {
    const raw = String(value || '').trim();
    if (!raw || /[\u0000-\u001f\u007f\\]/.test(raw)) return '';
    try {
        const parsed = new URL(raw);
        if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return '';
        if (parsed.username || parsed.password) return '';
        return parsed.href;
    } catch (_) {
        return '';
    }
}

function escapeMultiline(str) {
    return escHtml(str || '').replace(/\n/g, '<br>');
}

function finiteInputNumberAttr(input, attrName, fallback) {
    if (!input) return fallback;
    const rawValue = input.getAttribute(attrName);
    if (rawValue === null || rawValue === '') return fallback;
    const parsed = Number(rawValue);
    return Number.isFinite(parsed) ? parsed : fallback;
}

// ========== Trading Parameters ==========

async function fetchTradingParams() {
    const data = await fetchJSON('/api/settings/thresholds');
    if (!data) return;

    const intervalInput = document.getElementById('cfg-decision-interval');
    const thresholdInput = document.getElementById('cfg-confidence-threshold');
    const totalMarginInput = document.getElementById('cfg-total-margin-limit-pct');
    const localToolsEnabledInput = document.getElementById('cfg-local-ai-tools-enabled');
    const localToolsBaseInput = document.getElementById('cfg-local-ai-tools-api-base');
    const localToolsTimeoutInput = document.getElementById('cfg-local-ai-tools-timeout');
    const localToolsBreakerFailuresInput = document.getElementById('cfg-local-ai-tools-breaker-failures');
    const localToolsBreakerCooldownInput = document.getElementById('cfg-local-ai-tools-breaker-cooldown');
    const highRiskEnabledInput = document.getElementById('cfg-high-risk-review-enabled');
    const highRiskBaseInput = document.getElementById('cfg-high-risk-review-api-base');
    const highRiskKeyInput = document.getElementById('cfg-high-risk-review-api-key');
    const highRiskModelInput = document.getElementById('cfg-high-risk-review-model');
    const highRiskTimeoutInput = document.getElementById('cfg-high-risk-review-timeout');
    const highRiskMaxTokensInput = document.getElementById('cfg-high-risk-review-max-tokens');
    const highRiskBreakerFailuresInput = document.getElementById('cfg-high-risk-review-breaker-failures');
    const highRiskBreakerCooldownInput = document.getElementById('cfg-high-risk-review-breaker-cooldown');

    if (intervalInput) intervalInput.value = data.decision_interval;
    if (thresholdInput) thresholdInput.value = data.confidence_threshold;
    if (localToolsEnabledInput) localToolsEnabledInput.checked = Boolean(data.local_ai_tools_enabled);
    if (localToolsBaseInput) localToolsBaseInput.value = data.local_ai_tools_api_base || '';
    if (localToolsTimeoutInput) localToolsTimeoutInput.value = data.local_ai_tools_timeout_seconds ?? 8.0;
    if (localToolsBreakerFailuresInput) {
        localToolsBreakerFailuresInput.value = data.local_ai_tools_circuit_breaker_failures ?? 3;
    }
    if (localToolsBreakerCooldownInput) {
        localToolsBreakerCooldownInput.value = data.local_ai_tools_circuit_breaker_cooldown_seconds ?? 45;
    }
    if (highRiskEnabledInput) highRiskEnabledInput.checked = Boolean(data.high_risk_review_enabled);
    if (highRiskBaseInput) highRiskBaseInput.value = data.high_risk_review_api_base || '';
    if (highRiskKeyInput) {
        highRiskKeyInput.value = '';
        highRiskKeyInput.placeholder = data.high_risk_review_has_api_key
            ? '已有密钥（已隐藏），留空不变'
            : '线上模型密钥';
    }
    if (highRiskModelInput) highRiskModelInput.value = data.high_risk_review_model || '';
    if (highRiskTimeoutInput) highRiskTimeoutInput.value = data.high_risk_review_timeout_seconds ?? 30;
    if (highRiskMaxTokensInput) {
        const tokenFloor = Number(data.high_risk_review_token_floor);
        const tokenCap = Number(data.high_risk_review_token_cap);
        if (Number.isFinite(tokenFloor) && tokenFloor > 0) {
            highRiskMaxTokensInput.min = String(tokenFloor);
        }
        if (Number.isFinite(tokenCap) && tokenCap >= finiteInputNumberAttr(highRiskMaxTokensInput, 'min', 1)) {
            highRiskMaxTokensInput.max = String(tokenCap);
        }
        highRiskMaxTokensInput.value = data.high_risk_review_max_tokens ?? 480;
    }
    if (highRiskBreakerFailuresInput) {
        highRiskBreakerFailuresInput.value = data.high_risk_review_circuit_breaker_failures ?? 2;
    }
    if (highRiskBreakerCooldownInput) {
        highRiskBreakerCooldownInput.value = data.high_risk_review_circuit_breaker_cooldown_seconds ?? 120;
    }
    if (totalMarginInput) {
        const pct = valueNumber(data.total_margin_limit_pct);
        totalMarginInput.value = pct !== null ? (pct * 100).toFixed(0) : '';
    }
}

async function saveTradingParams() {
    const intervalInput = document.getElementById('cfg-decision-interval');
    const thresholdInput = document.getElementById('cfg-confidence-threshold');
    const totalMarginInput = document.getElementById('cfg-total-margin-limit-pct');
    const localToolsEnabledInput = document.getElementById('cfg-local-ai-tools-enabled');
    const localToolsBaseInput = document.getElementById('cfg-local-ai-tools-api-base');
    const localToolsTimeoutInput = document.getElementById('cfg-local-ai-tools-timeout');
    const localToolsBreakerFailuresInput = document.getElementById('cfg-local-ai-tools-breaker-failures');
    const localToolsBreakerCooldownInput = document.getElementById('cfg-local-ai-tools-breaker-cooldown');
    const highRiskEnabledInput = document.getElementById('cfg-high-risk-review-enabled');
    const highRiskBaseInput = document.getElementById('cfg-high-risk-review-api-base');
    const highRiskKeyInput = document.getElementById('cfg-high-risk-review-api-key');
    const highRiskModelInput = document.getElementById('cfg-high-risk-review-model');
    const highRiskTimeoutInput = document.getElementById('cfg-high-risk-review-timeout');
    const highRiskMaxTokensInput = document.getElementById('cfg-high-risk-review-max-tokens');
    const highRiskBreakerFailuresInput = document.getElementById('cfg-high-risk-review-breaker-failures');
    const highRiskBreakerCooldownInput = document.getElementById('cfg-high-risk-review-breaker-cooldown');

    const body = {};
    if (intervalInput && intervalInput.value) {
        body.decision_interval = parseInt(intervalInput.value);
    }
    if (thresholdInput && thresholdInput.value) {
        body.confidence_threshold = parseFloat(thresholdInput.value);
    }
    if (localToolsEnabledInput) {
        body.local_ai_tools_enabled = Boolean(localToolsEnabledInput.checked);
    }
    if (localToolsBaseInput) {
        body.local_ai_tools_api_base = localToolsBaseInput.value.trim();
    }
    if (localToolsTimeoutInput && localToolsTimeoutInput.value !== '') {
        const timeout = parseFloat(localToolsTimeoutInput.value);
        if (!Number.isFinite(timeout) || timeout < 0.2 || timeout > 15) {
            alert('保存失败: 本地 AI 工具超时必须在 0.2 到 15 秒之间');
            return;
        }
        body.local_ai_tools_timeout_seconds = timeout;
    }
    if (localToolsBreakerFailuresInput && localToolsBreakerFailuresInput.value !== '') {
        const failures = parseInt(localToolsBreakerFailuresInput.value, 10);
        if (!Number.isFinite(failures) || failures < 1 || failures > 20) {
            alert('保存失败: 本地 AI 工具熔断失败次数必须在 1 到 20 之间');
            return;
        }
        body.local_ai_tools_circuit_breaker_failures = failures;
    }
    if (localToolsBreakerCooldownInput && localToolsBreakerCooldownInput.value !== '') {
        const cooldown = parseFloat(localToolsBreakerCooldownInput.value);
        if (!Number.isFinite(cooldown) || cooldown < 5 || cooldown > 3600) {
            alert('保存失败: 本地 AI 工具熔断冷却时间必须在 5 到 3600 秒之间');
            return;
        }
        body.local_ai_tools_circuit_breaker_cooldown_seconds = cooldown;
    }
    if (highRiskEnabledInput) {
        body.high_risk_review_enabled = Boolean(highRiskEnabledInput.checked);
    }
    if (highRiskBaseInput) {
        body.high_risk_review_api_base = highRiskBaseInput.value.trim();
    }
    if (highRiskKeyInput && highRiskKeyInput.value.trim() && !highRiskKeyInput.value.trim().startsWith('****')) {
        body.high_risk_review_api_key = highRiskKeyInput.value.trim();
    }
    if (highRiskModelInput) {
        body.high_risk_review_model = highRiskModelInput.value.trim();
    }
    if (highRiskTimeoutInput && highRiskTimeoutInput.value !== '') {
        const timeout = parseFloat(highRiskTimeoutInput.value);
        if (!Number.isFinite(timeout) || timeout < 5 || timeout > 120) {
            alert('保存失败: 高风险复核超时必须在 5 到 120 秒之间');
            return;
        }
        body.high_risk_review_timeout_seconds = timeout;
    }
    if (highRiskMaxTokensInput && highRiskMaxTokensInput.value !== '') {
        const maxTokens = parseInt(highRiskMaxTokensInput.value, 10);
        const tokenFloor = finiteInputNumberAttr(highRiskMaxTokensInput, 'min', 1);
        const tokenCap = finiteInputNumberAttr(highRiskMaxTokensInput, 'max', Number.MAX_SAFE_INTEGER);
        if (!Number.isFinite(maxTokens) || maxTokens < tokenFloor || maxTokens > tokenCap) {
            alert(`保存失败: 高风险复核最大输出 Token 必须在 ${tokenFloor} 到 ${tokenCap} 之间`);
            return;
        }
        body.high_risk_review_max_tokens = maxTokens;
    }
    if (highRiskBreakerFailuresInput && highRiskBreakerFailuresInput.value !== '') {
        const failures = parseInt(highRiskBreakerFailuresInput.value, 10);
        if (!Number.isFinite(failures) || failures < 1 || failures > 20) {
            alert('保存失败: 高风险复核熔断失败次数必须在 1 到 20 之间');
            return;
        }
        body.high_risk_review_circuit_breaker_failures = failures;
    }
    if (highRiskBreakerCooldownInput && highRiskBreakerCooldownInput.value !== '') {
        const cooldown = parseFloat(highRiskBreakerCooldownInput.value);
        if (!Number.isFinite(cooldown) || cooldown < 5 || cooldown > 3600) {
            alert('保存失败: 高风险复核熔断冷却时间必须在 5 到 3600 秒之间');
            return;
        }
        body.high_risk_review_circuit_breaker_cooldown_seconds = cooldown;
    }
    if (totalMarginInput && totalMarginInput.value) {
        const pct = parseFloat(totalMarginInput.value);
        if (!Number.isFinite(pct) || pct < 10 || pct > 100) {
            alert('保存失败: 总保证金占用上限必须在 10 到 100 之间');
            return;
        }
        body.total_margin_limit_pct = pct / 100;
    }

    const res = await fetchWithAuth('/api/settings/thresholds', dashboardWriteOptions({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }));

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('保存失败: ' + (err.detail || '未知错误'));
        return;
    }

    const data = await res.json();
    state.decisionInterval = data.decision_interval;
    if (totalMarginInput && data.total_margin_limit_pct !== undefined) {
        totalMarginInput.value = (Number(data.total_margin_limit_pct) * 100).toFixed(0);
    }
    alert('参数已保存，立即生效');
}

// Inline handlers in index.html need explicit window bindings in some browser shells.
window.showAnalysisReason = showAnalysisReason;
window.changeAnalysisPage = changeAnalysisPage;
window.fetchAnalysisRecords = fetchAnalysisRecords;

// Cleaner Local ML page rendering. Keep the raw model names in small technical
// text, but lead with trading-purpose language so the page is readable.
function mlFriendlyStatusLabel(ready, activeText = '已介入') {
    if (!ready) return '未就绪';
    return activeText;
}

function mlTechName(name) {
    return name || '技术模型未返回';
}

// Final readable Local ML rendering override. Earlier definitions are kept for
// compatibility, but this version makes the sample-count cap explicit.
function renderReadableTrainableModelCard(model) {
    const metrics = Array.isArray(model.metrics) && model.metrics.length
        ? `<div class="ml-model-metrics">${model.metrics.map(item => `
            <div class="ml-model-metric">
                <span>${escHtml(item.label)}</span>
                <strong>${escHtml(item.value)}</strong>
            </div>
        `).join('')}</div>`
        : '';
    return `
        <div class="ml-train-model-card">
            <div class="ml-train-model-head">
                <div>
                    <div class="ml-train-model-title">${escHtml(model.title)}</div>
                    <div class="ml-train-model-type">${escHtml(model.type || '-')}</div>
                </div>
                ${mlModelStatusPill(model.ready, model.statusLabel || (model.ready ? '可用' : '未就绪'))}
            </div>
            <div class="ml-train-model-desc">${escHtml(model.description || '-')}</div>
            <div class="ml-train-model-grid">
                <div><span>样本情况</span><strong>${escHtml(model.samples || '-')}</strong></div>
                <div><span>最近训练</span><strong>${escHtml(model.trainedAt || '-')}</strong></div>
                <div><span>当前作用</span><strong>${escHtml(model.usage || '-')}</strong></div>
            </div>
            ${metrics}
            <div class="ml-train-model-note">${escHtml(model.note || '')}</div>
        </div>`;
}

function renderMLSignalOverview() {
    const container = document.getElementById('ml-signal-overview');
    const updatedEl = document.getElementById('ml-signal-updated');
    if (!container) return;
    const status = state.mlSignalStatus || {};
    const records = state.mlSignalRecords || [];
    const latestRecord = records[0] || null;
    const latestSignal = latestRecord?.ml_signal || null;
    const latestPrediction = mlPrimaryPrediction(latestSignal);
    const ready = status.available === true;
    const influenceEnabled = status.influence_enabled === true && status.status === 'ready';
    const mode = status.mode || latestSignal?.mode || 'learning_only';
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    const samples = mlSampleCounts();
    const latestText = latestRecord
        ? `${toBeijingTime(latestRecord.created_at)} ${latestRecord.symbol || '-'}`
        : '暂无最近预测';
    const strongSignals = records.filter(r => {
        const pred = mlPrimaryPrediction(r.ml_signal) || {};
        return Number(pred.best_expected_return_pct || 0) > 0 && Number(pred.profit_edge_pct || 0) > 0;
    }).length;

    if (updatedEl) {
        updatedEl.textContent = ready
            ? `累计完成 ${samples.completedMl} 条，训练窗口 ${samples.trainingMl} 条 · ${influenceEnabled ? '已介入' : '学习中'}`
            : '模型不可用';
    }

    container.innerHTML = `
        <div class="ml-flow">
            <div class="ml-flow-step">
                <div class="ml-flow-index">1</div>
                <div><strong>累计影子复盘样本</strong><span>${samples.completedMl} 条 completed 样本，数据仍在增长</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">2</div>
                <div><strong>本次训练使用样本</strong><span>${samples.trainingMl} 条最新样本；窗口上限 ${samples.limit} 条，不等于累计总数</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">3</div>
                <div><strong>训练目标</strong><span>以预期收益和盈亏质量为主，胜率只做辅助指标</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">4</div>
                <div><strong>${influenceEnabled ? '参与开仓过滤' : '学习观察中'}</strong><span>${influenceEnabled ? '负预期机会会降权或拦截，高质量机会会加分' : '指标未达标时继续训练，不强行影响交易'}</span></div>
            </div>
        </div>
        <div class="ml-overview-grid">
            ${mlMetricCard('模型状态', ready ? (influenceEnabled ? '已介入' : '学习中') : '不可用', mode === 'entry_profit_filter' ? '盈亏质量过滤中' : '暂不强制影响交易', ready ? (influenceEnabled ? 'good' : 'warn') : 'bad')}
            ${mlMetricCard('累计完成样本', String(samples.completedMl), '数据库里已完成的影子复盘总数', samples.completedMl > samples.trainingMl ? 'good' : 'muted')}
            ${mlMetricCard('训练窗口样本', String(samples.trainingMl), `训练 ${Number(status.train_count || 0)} / 测试 ${Number(status.test_count || 0)}；窗口上限 ${samples.limit}`, 'good')}
            ${mlMetricCard('新增待消化样本', String(samples.newCount), '达到自动训练条件后会进入下一轮训练', samples.newCount >= Number(status.auto_train_min_new_samples || 500) ? 'good' : 'muted')}
            ${mlMetricCard('最近预测', latestText, latestPrediction ? `${mlSideLabel(latestPrediction.best_side)} 预期 ${signedPctValueLabel(latestPrediction.best_expected_return_pct)}` : '等待新分析', latestPrediction ? (Number(latestPrediction.best_expected_return_pct || 0) > 0 ? 'good' : 'warn') : 'muted')}
            ${mlMetricCard('正期望数量', `${strongSignals} / ${records.length}`, '最近记录里预期收益为正且有收益差的数量', strongSignals ? 'warn' : 'muted')}
            ${mlMetricCard('训练时间', trainedAt, status.version ? `版本 ${String(status.version).slice(0, 10)}` : '', 'muted')}
            ${mlMetricCard('显示说明', '20000 是窗口', '不是样本没增长，而是只拿最新窗口训练，避免老行情污染模型', 'warn')}
        </div>`;
}

function renderLocalAIToolsStatus() {
    const container = document.getElementById('local-ai-tools-status');
    const updatedEl = document.getElementById('local-ai-tools-updated');
    if (!container) return;
    const status = state.localAIToolsStatus || {};
    const models = status.models || {};
    const childEndpoints = status.child_endpoints || {};
    const available = status.available === true;
    const serviceAvailable = status.service_available !== false && (available || status.service_available === true);
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    const samples = mlSampleCounts();
    const childAvailableCount = Object.values(childEndpoints).filter(item => item && item.available).length;
    const childTotalCount = Object.keys(childEndpoints).length;
    if (updatedEl) {
        updatedEl.textContent = serviceAvailable
            ? `累计 ${samples.completedLocal} 条影子 / ${samples.completedLocalTrade} 条交易；训练窗口 ${samples.trainingLocal} / ${samples.trainingLocalTrade} 条；子接口 ${childAvailableCount}/${childTotalCount || 4}`
            : '服务不可用';
    }

    const cards = [
        {
            label: '服务状态',
            value: serviceAvailable ? '可用' : '不可用',
            subtitle: serviceAvailable ? (status.model_bundle_available === false ? '服务已连接，训练模型未就绪，子接口使用启发式/轻量模型' : '服务器量化工具已连接') : (status.error || status.message || '等待服务返回状态'),
            tone: serviceAvailable ? 'good' : 'bad',
        },
        {
            label: '真实子接口',
            value: `${childAvailableCount}/${childTotalCount || 4}`,
            subtitle: childTotalCount ? '盈利预测、时序、情绪和平仓建议探针结果' : '等待后端返回子接口探针',
            tone: childAvailableCount >= Math.max(childTotalCount, 1) ? 'good' : (childAvailableCount > 0 ? 'warn' : 'bad'),
        },
        {
            label: '累计影子复盘样本',
            value: String(samples.completedLocal),
            subtitle: '数据库里已完成的影子复盘总数，应该持续增长',
            tone: samples.completedLocal > 0 ? 'good' : 'warn',
        },
        {
            label: '本次训练使用样本',
            value: String(samples.trainingLocal || Number(status.shadow_sample_count || 0)),
            subtitle: `只取最新窗口训练，上限 ${Number(status.training_shadow_sample_limit || samples.limit || 20000)} 条`,
            tone: (samples.trainingLocal || Number(status.shadow_sample_count || 0)) > 0 ? 'good' : 'warn',
        },
        {
            label: '交易/平仓样本',
            value: `${samples.trainingLocalTrade} / ${samples.completedLocalTrade}`,
            subtitle: '训练窗口 / 累计；已按仓位去重，手动平仓不参与训练',
            tone: samples.completedLocalTrade > 0 ? 'good' : 'warn',
        },
        {
            label: '序列样本',
            value: String(Number(status.sequence_sample_count || 0)),
            subtitle: models.deep_timeseries || models.timeseries || '用于多周期行情预测',
            tone: Number(status.sequence_sample_count || 0) > 0 ? 'good' : 'warn',
        },
        {
            label: '文本情绪样本',
            value: String(Number(status.text_sentiment_sample_count || 0)),
            subtitle: models.deep_sentiment || models.sentiment || '用于新闻/公告/情绪校准',
            tone: Number(status.text_sentiment_sample_count || 0) > 0 ? 'good' : 'warn',
        },
    ];

    container.innerHTML = `
        <div class="ml-overview-grid ml-overview-grid-compact">
            ${cards.map(item => mlMetricCard(item.label, item.value, item.subtitle, item.tone)).join('')}
        </div>
        <div class="ml-purpose-grid">
            <div class="ml-purpose-card ml-purpose-good">
                <div class="ml-purpose-title">盈利预测</div>
                <div class="ml-purpose-desc">判断扣除成本后的预期收益是不是值得开仓。</div>
                <div class="ml-purpose-tech">${escHtml(models.profit || 'ExtraTrees / CatBoost-style')}</div>
            </div>
            <div class="ml-purpose-card ml-purpose-warn">
                <div class="ml-purpose-title">亏损过滤</div>
                <div class="ml-purpose-desc">识别近期容易亏损的币种、方向和行情组合。</div>
                <div class="ml-purpose-tech">${escHtml(models.loss_filter || 'Classifier')}</div>
            </div>
            <div class="ml-purpose-card ml-purpose-muted">
                <div class="ml-purpose-title">训练窗口说明</div>
                <div class="ml-purpose-desc">页面里看到的 20000 是最新样本窗口，不是累计样本停止增长。</div>
                <div class="ml-purpose-tech">累计样本看“累计影子复盘样本”。最近训练：${escHtml(trainedAt)}</div>
            </div>
        </div>`;
}

function renderTrainableModels() {
    const container = document.getElementById('ml-trainable-models');
    if (!container) return;
    const local = state.localAIToolsStatus || {};
    const ml = state.mlSignalStatus || {};
    const modelsMap = local.models || {};
    const localTrainedAt = local.trained_at ? toBeijingTime(local.trained_at) : '-';
    const mlTrainedAt = ml.trained_at ? toBeijingTime(ml.trained_at) : '-';
    const metrics = ml.metrics || {};
    const autoLast = ml.auto_train_last_result || {};
    const samples = mlSampleCounts();
    const autoTrainText = ml.auto_train_enabled
        ? `自动训练已开启；下次检查 ${ml.auto_train_next_check_at ? toBeijingTime(ml.auto_train_next_check_at) : '-'}`
        : '自动训练未开启';
    const windowText = `${samples.trainingMl} / ${samples.completedMl}（训练窗口/累计）`;
    const localWindowText = `${samples.trainingLocal || Number(local.shadow_sample_count || 0)} / ${samples.completedLocal}（训练窗口/累计）`;
    const localTradeWindowText = `${samples.trainingLocalTrade} / ${samples.completedLocalTrade}（训练窗口/累计）`;
    const models = [
        {
            title: '本地 ML 盈亏质量',
            type: '本机 ExtraTrees 盈亏过滤',
            ready: ml.available === true,
            statusLabel: ml.influence_enabled ? '已介入' : (ml.available ? '学习中' : '未训练'),
            description: '判断一笔交易是否有正期望，开仓时用于门槛、否决和机会排序。',
            samples: windowText,
            trainedAt: mlTrainedAt,
            usage: ml.influence_enabled ? '开仓过滤 + 机会排序' : '只学习，不强制影响交易',
            metrics: [
                { label: '做多 AUC', value: Number(metrics.long_auc || 0).toFixed(3) },
                { label: '做空 AUC', value: Number(metrics.short_auc || 0).toFixed(3) },
                { label: '做多高分收益', value: signedPctValueLabel(metrics.top_long_avg_return_pct) },
                { label: '做空高分收益', value: signedPctValueLabel(metrics.top_short_avg_return_pct) },
            ],
            note: autoLast.message || autoTrainText,
        },
        {
            title: '开仓盈利预测',
            type: mlTechName(modelsMap.profit),
            ready: localModelStatus(local, 'profit'),
            description: '预测做多/做空扣除成本后的预期收益，目标是净利润最大化。',
            samples: localWindowText,
            trainedAt: localTrainedAt,
            usage: '给专家和最终裁决提供收益证据',
            metrics: [
                { label: '特征数', value: String(Number(local.feature_count || 0)) },
                { label: '预测周期', value: (local.horizons || []).join('/') || '-' },
            ],
            note: '胜率不是最终目标，真正目标是扣除手续费和滑点后的实现利润。',
        },
        {
            title: '亏损风险过滤',
            type: mlTechName(modelsMap.loss_filter),
            ready: localModelStatus(local, 'loss_filter'),
            description: '识别某个币种/方向近期是否容易亏损，避免反复交易亏损组合。',
            samples: localWindowText,
            trainedAt: localTrainedAt,
            usage: '亏损概率提示 + 开仓风险过滤',
            metrics: [
                { label: '币种画像', value: String(Number(local.profile_count || 0)) },
                { label: '特征数', value: String(Number(local.feature_count || 0)) },
            ],
            note: '比如某币种近期连续亏损时，会降低开仓优先级或要求更强证据。',
        },
        {
            title: '多周期行情预测',
            type: mlTechName(modelsMap.deep_timeseries || modelsMap.timeseries),
            ready: localModelStatus(local, 'timeseries') || Boolean(local.torch_patch_status?.available),
            description: '预测未来 10/30/60 分钟收益和波动，用来辅助判断入场窗口。',
            samples: `${Number(local.sequence_sample_count || 0)} 条序列样本`,
            trainedAt: localTrainedAt,
            usage: '短周期方向 + 波动预判',
            metrics: [
                { label: '周期', value: (local.horizons || []).join('/') || '-' },
                { label: 'MAE', value: local.torch_patch_status?.train_mae_pct !== undefined ? `${Number(local.torch_patch_status.train_mae_pct).toFixed(4)}%` : '-' },
                { label: '输入维度', value: String(local.torch_patch_status?.input_dim || local.feature_count || '-') },
            ],
            note: '这部分用于辅助判断时机，不会单独决定买卖。',
        },
        {
            title: '情绪风险校准',
            type: mlTechName(modelsMap.deep_sentiment || modelsMap.sentiment),
            ready: localModelStatus(local, 'sentiment') || localModelStatus(local, 'deep_sentiment'),
            description: '学习新闻、公告和社媒情绪对收益/风险的影响。',
            samples: `${Number(local.text_sentiment_sample_count || 0)} 条文本样本`,
            trainedAt: localTrainedAt,
            usage: '新闻情绪风险 + 收益校准',
            metrics: [
                { label: 'Transformers', value: local.transformers_sentiment_backend?.available ? '可用' : '未启用' },
                { label: '库版本', value: local.transformers_sentiment_backend?.version || '-' },
            ],
            note: '文本样本越多，对突发新闻和事件风险的判断越有价值。',
        },
        {
            title: '平仓/退出建议',
            type: mlTechName(modelsMap.exit),
            ready: localModelStatus(local, 'exit'),
            description: '结合真实持仓盈亏、持仓时间和历史交易画像，判断止盈、止损、减仓或继续持有。',
            samples: localTradeWindowText,
            trainedAt: localTrainedAt,
            usage: '持仓复盘 + 平仓建议',
            metrics: [
                { label: '训练窗口', value: String(samples.trainingLocalTrade) },
                { label: '累计去重样本', value: String(samples.completedLocalTrade) },
                { label: '币种画像', value: String(Number(local.profile_count || 0)) },
            ],
            note: '它服务于已实现净利润，不是单纯追求持仓浮盈。',
        },
    ];

    container.innerHTML = `
        <div class="ml-train-summary">
            ${mlMetricCard('可训练模型', `${models.length} 个`, '覆盖开仓、亏损过滤、时序、情绪和平仓', 'good')}
            ${mlMetricCard('自动训练', ml.auto_train_enabled ? '已开启' : '未开启', autoTrainText, ml.auto_train_enabled ? 'good' : 'warn')}
            ${mlMetricCard('新增待训练样本', String(samples.newCount), autoLast.message || '等待下一次训练检查', samples.newCount >= Number(ml.auto_train_min_new_samples || 500) ? 'good' : 'muted')}
            ${mlMetricCard('样本显示说明', '窗口/累计', `训练窗口上限 ${samples.limit} 条；累计完成 ${samples.completedMl} 条`, 'warn')}
        </div>
        <div class="ml-train-model-list ml-train-model-list-clear">
            ${models.map(renderReadableTrainableModelCard).join('')}
        </div>`;
}

// ========== Profit Attribution ==========
async function fetchProfitAttribution() {
    const hoursEl = document.getElementById('profit-attribution-hours');
    const hours = hoursEl ? Number(hoursEl.value || 24) : 24;
    const mode = state.mode || 'paper';
    const data = await fetchJSON(`/api/profit-attribution?mode=${mode}&hours=${hours}&limit=200`);
    state.profitAttributionRecordPage = 1;
    state.profitAttribution = data || null;
    renderProfitAttribution();
}

function setProfitAttributionView(view) {
    const selected = view === 'records' ? 'records' : 'overview';
    state.profitAttributionView = selected;
    document.querySelectorAll('#profit-attribution-tabs .trade-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.profitAttributionView === selected);
    });
    document.getElementById('profit-attribution-panel-overview')?.classList.toggle('active', selected === 'overview');
    document.getElementById('profit-attribution-panel-records')?.classList.toggle('active', selected === 'records');
}

function changeProfitAttributionRecordPage(page) {
    state.profitAttributionRecordPage = Math.max(1, Number(page) || 1);
    renderProfitAttributionRecords(state.profitAttribution || {});
}

// ========== Opening Funnel ==========
function pctFmt(value) {
    const n = Number(value || 0);
    return `${(n * 100).toFixed(1)}%`;
}

async function fetchOpeningFunnel() {
    const hoursEl = document.getElementById('opening-funnel-hours');
    const hours = hoursEl ? Number(hoursEl.value || 24) : 24;
    const data = await fetchJSON(`/api/opening-funnel?mode=${state.mode || 'paper'}&hours=${hours}&limit=500`);
    if (!data || !data.stages) {
        renderOpeningFunnelUnavailable(data);
        return;
    }
    state.openingFunnel = data;
    renderOpeningFunnel(data);
}

function renderOpeningFunnelUnavailable(data) {
    const summaryEl = document.getElementById('opening-funnel-summary');
    const stagesEl = document.getElementById('opening-funnel-stages');
    const reasonsEl = document.getElementById('opening-funnel-reasons');
    const symbolsEl = document.getElementById('opening-funnel-symbols');
    const tbody = document.getElementById('opening-funnel-blocked-tbody');
    const updatedEl = document.getElementById('opening-funnel-updated');
    const detail = data && data.detail ? `接口返回：${data.detail}` : '后端接口暂不可用';
    if (summaryEl) {
        summaryEl.innerHTML = `
            <div class="opening-funnel-verdict opening-funnel-warn">
                <strong>开仓漏斗后端尚未加载</strong>
                <span>${escHtml(detail)}。请重启交易服务后刷新本页，新的 /api/opening-funnel 接口才会生效。</span>
            </div>`;
    }
    if (stagesEl) stagesEl.innerHTML = '';
    if (reasonsEl) reasonsEl.innerHTML = '<div class="opening-funnel-empty">等待后端接口生效。</div>';
    if (symbolsEl) symbolsEl.innerHTML = '<div class="opening-funnel-empty">等待后端接口生效。</div>';
    if (tbody) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center;padding:24px;">等待后端接口生效</td></tr>';
    }
    if (updatedEl) updatedEl.textContent = '需要重启服务';
}

function openingFunnelReasonLabel(key) {
    const labels = {
        evidence_gate: '证据评分',
        risk_or_precheck: '风控/预检',
        waiting_queue: '观望/等待',
        execution_or_exchange: '执行/交易所',
        ai_budget: 'AI预算',
        other: '其他',
        unknown: '缺少原因',
    };
    return labels[key] || key || '-';
}

function openingFunnelActionLabel(action) {
    if (action === 'long') return '做多';
    if (action === 'short') return '做空';
    return action || '-';
}

function renderOpeningFunnel(data) {
    renderOpeningFunnelSummary(data);
    renderOpeningFunnelStages(data);
    renderOpeningFunnelReasons(data);
    renderOpeningFunnelSymbols(data);
    renderOpeningFunnelBlocked(data);
    const updatedEl = document.getElementById('opening-funnel-updated');
    if (updatedEl) {
        const modeLabel = data.mode === 'live' ? '实盘' : '模拟盘';
        updatedEl.textContent = `${modeLabel} · 最近 ${data.window_hours || 24} 小时 · ${new Date().toLocaleTimeString()}`;
    }
}

function renderOpeningFunnelSummary(data) {
    const el = document.getElementById('opening-funnel-summary');
    if (!el) return;
    const scans = Number(data.market_scans || 0);
    const signals = Number(data.stages?.ai_entry_signals || 0);
    const executed = Number(data.stages?.executed_entries || 0);
    const bottleneck = data.bottleneck_label || '暂无足够数据';
    const tone = data.bottleneck === 'healthy_selective' ? 'good' : scans ? 'warn' : 'muted';
    el.innerHTML = `
        <div class="opening-funnel-verdict opening-funnel-${tone}">
            <strong>${escHtml(bottleneck)}</strong>
            <span>市场分析 ${scans} 次，AI 开仓信号 ${signals} 次，实际开仓 ${executed} 次。总开仓率 ${pctFmt(data.rates?.overall_open_rate)}。</span>
        </div>
        <div class="opening-funnel-kpis">
            <div><span>AI 给信号率</span><strong>${pctFmt(data.rates?.signal_rate)}</strong></div>
            <div><span>信号成单率</span><strong>${pctFmt(data.rates?.order_rate)}</strong></div>
            <div><span>信号执行率</span><strong>${pctFmt(data.rates?.execution_rate)}</strong></div>
            <div><span>平均信心</span><strong>${Number(data.average_confidence || 0).toFixed(2)}</strong></div>
        </div>`;
}

function renderOpeningFunnelStages(data) {
    const el = document.getElementById('opening-funnel-stages');
    if (!el) return;
    const stages = [
        ['市场扫描', data.stages?.market_scans || 0, '系统完成的新机会分析'],
        ['AI开仓信号', data.stages?.ai_entry_signals || 0, '最终裁决为做多/做空'],
        ['生成订单', data.stages?.orders_created || 0, '本地订单表有关联记录'],
        ['实际开仓', data.stages?.executed_entries || 0, '决策标记为已执行'],
    ];
    const max = Math.max(...stages.map(s => Number(s[1] || 0)), 1);
    el.innerHTML = stages.map(([label, value, desc], index) => {
        const width = Math.max(4, (Number(value || 0) / max) * 100);
        return `
            <div class="opening-funnel-stage">
                <div class="opening-funnel-stage-step">0${index + 1}</div>
                <div class="opening-funnel-stage-head">
                    <span>${escHtml(label)}</span>
                    <strong>${Number(value || 0)}</strong>
                </div>
                <div class="opening-funnel-bar"><span style="width:${width}%;"></span></div>
                <div class="opening-funnel-stage-desc">${escHtml(desc)}</div>
            </div>`;
    }).join('');
}

function renderOpeningFunnelReasons(data) {
    const el = document.getElementById('opening-funnel-reasons');
    if (!el) return;
    const buckets = data.reason_buckets || {};
    const items = Object.entries(buckets).filter(([, count]) => Number(count || 0) > 0);
    const total = items.reduce((sum, [, count]) => sum + Number(count || 0), 0);
    if (!items.length) {
        el.innerHTML = '<div class="opening-funnel-empty">没有未执行的开仓信号。</div>';
        return;
    }
    el.innerHTML = items.sort((a, b) => Number(b[1]) - Number(a[1])).map(([key, count]) => {
        const ratio = total ? Number(count || 0) / total : 0;
        return `
            <div class="opening-funnel-row opening-funnel-reason-row">
                <div><strong>${escHtml(openingFunnelReasonLabel(key))}</strong><span>${pctFmt(ratio)}</span></div>
                <div class="opening-funnel-bar"><span style="width:${Math.max(4, ratio * 100)}%;"></span></div>
                <em>${Number(count || 0)} 次拦截</em>
            </div>`;
    }).join('');
}

function renderOpeningFunnelSymbols(data) {
    const el = document.getElementById('opening-funnel-symbols');
    if (!el) return;
    const symbols = Array.isArray(data.top_symbols) ? data.top_symbols : [];
    if (!symbols.length) {
        el.innerHTML = '<div class="opening-funnel-empty">暂无币种统计。</div>';
        return;
    }
    el.innerHTML = symbols.map(item => {
        const scans = Number(item.scans || 0);
        const signals = Number(item.signals || 0);
        const executed = Number(item.executed || 0);
        const width = scans ? Math.max(4, (signals / scans) * 100) : 4;
        return `
            <div class="opening-funnel-row opening-funnel-symbol-row">
                <div><strong>${escHtml(item.symbol || '-')}</strong><span>${signals}/${scans} 信号 · ${executed} 开仓</span></div>
                <div class="opening-funnel-bar"><span style="width:${width}%;"></span></div>
                <em>信号率 ${pctFmt(item.signal_rate)}</em>
            </div>`;
    }).join('');
}

function renderOpeningFunnelBlocked(data) {
    const tbody = document.getElementById('opening-funnel-blocked-tbody');
    if (!tbody) return;
    const rows = Array.isArray(data.recent_blocked) ? data.recent_blocked : [];
    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center;padding:24px;">暂无未执行的开仓信号</td></tr>';
        return;
    }
    tbody.innerHTML = rows.map(row => `
        <tr>
            <td class="opening-funnel-time">${toBeijingTime(row.created_at)}</td>
            <td class="opening-funnel-symbol">${escHtml(row.symbol || '-')}</td>
            <td><span class="opening-funnel-side">${openingFunnelActionLabel(row.action)}</span></td>
            <td class="opening-funnel-confidence">${Number(row.confidence || 0).toFixed(2)}</td>
            <td><span class="opening-funnel-bucket">${escHtml(openingFunnelReasonLabel(row.reason_bucket))}</span></td>
            <td class="opening-funnel-reason-cell">${escHtml(row.reason || '-')}</td>
        </tr>
    `).join('');
}

window.fetchOpeningFunnel = fetchOpeningFunnel;

// Clean profit-attribution renderers. These intentionally live at the end of
// the file so they override the older profit-attribution functions above.
function renderProfitAttribution() {
    const data = state.profitAttribution || {};
    setProfitAttributionView(state.profitAttributionView || 'overview');
    renderProfitAttributionSummary(data);
    renderProfitAttributionBuckets(data);
    renderProfitAttributionState(data);
    renderProfitAttributionRecords(data);
    const updated = document.getElementById('profit-attribution-updated');
    if (updated) {
        const modeLabel = data.mode === 'live' ? '实盘' : '模拟盘';
        updated.textContent = `${modeLabel} | 最近 ${data.window_hours || 24} 小时 | ${new Date().toLocaleTimeString()}`;
    }
}

function renderProfitAttributionSummary(data) {
    const el = document.getElementById('profit-attribution-summary');
    if (!el) return;
    const summary = data.summary || {};
    const pnl = Number(summary.total_closed_pnl || 0);
    const trades = Number(summary.trade_count || 0);
    const tone = pnl > 0 ? 'good' : pnl < 0 ? 'warn' : 'muted';
    if (!trades) {
        el.innerHTML = `
            <div class="opening-funnel-verdict opening-funnel-muted">
                <strong>暂无已平仓样本</strong>
                <span>${escHtml(data.message || '最近窗口内没有可归因的交易。')}</span>
            </div>`;
        return;
    }
    el.innerHTML = `
        <div class="opening-funnel-verdict opening-funnel-${tone}">
            <strong>${signedMoney(pnl)} U</strong>
            <span>最近 ${data.window_hours || 24} 小时已平仓 ${trades} 笔，胜率 ${pctLabel(summary.win_rate, 1)}，盈亏比 ${Number(summary.profit_factor || 0).toFixed(2)}。</span>
        </div>
        <div class="opening-funnel-kpis">
            <div><span>盈利 / 亏损</span><strong>${Number(summary.win_count || 0)} / ${Number(summary.loss_count || 0)}</strong></div>
            <div><span>平均盈利</span><strong>${signedMoney(summary.avg_win || 0)} U</strong></div>
            <div><span>平均亏损</span><strong>-${fmtMoney(summary.avg_loss || 0)} U</strong></div>
            <div><span>小盈 / 大亏</span><strong>${Number(summary.small_win_count || 0)} / ${Number(summary.large_loss_count || 0)}</strong></div>
        </div>`;
}

function renderProfitAttributionBuckets(data) {
    const el = document.getElementById('profit-attribution-buckets');
    if (!el) return;
    const rows = Array.isArray(data.buckets) ? data.buckets : [];
    if (!rows.length) {
        el.innerHTML = '<div class="opening-funnel-empty">暂无归因分类。</div>';
        return;
    }
    const maxAbs = Math.max(...rows.map(row => Math.abs(Number(row.pnl || 0))), 1);
    el.innerHTML = rows.map(row => {
        const pnl = Number(row.pnl || 0);
        const width = Math.max(4, Math.abs(pnl) / maxAbs * 100);
        const color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        return `
            <div class="opening-funnel-row opening-funnel-reason-row">
                <div><strong>${escHtml(row.label || row.key || '-')}</strong><span>${Number(row.count || 0)} 笔 | 均值 ${signedMoney(row.avg_pnl || 0)} U</span></div>
                <div class="opening-funnel-bar"><span style="width:${width}%;background:${color};"></span></div>
                <em style="color:${color};">${signedMoney(pnl)} U</em>
            </div>`;
    }).join('');
}

function renderProfitAttributionState(data) {
    const el = document.getElementById('profit-attribution-state');
    if (!el) return;
    const records = Array.isArray(data.records) ? data.records : [];
    const counts = {};
    records.forEach(row => {
        const summary = row.decision_state?.summary || {};
        const label = summary.final_status
            ? `${stateStageLabel(summary.final_stage)} / ${stateStatusLabel(summary.final_status)}`
            : '无状态机记录';
        counts[label] = (counts[label] || 0) + 1;
    });
    const items = Object.entries(counts);
    if (!items.length) {
        el.innerHTML = '<div class="opening-funnel-empty">暂无状态机样本。</div>';
        return;
    }
    const max = Math.max(...items.map(([, count]) => Number(count || 0)), 1);
    el.innerHTML = items.sort((a, b) => Number(b[1]) - Number(a[1])).map(([label, count]) => `
        <div class="opening-funnel-row opening-funnel-symbol-row">
            <div><strong>${escHtml(label)}</strong><span>${Number(count || 0)} 笔</span></div>
            <div class="opening-funnel-bar"><span style="width:${Math.max(4, Number(count || 0) / max * 100)}%;"></span></div>
            <em>${Number(count || 0)}</em>
        </div>`).join('');
}

// Profit attribution evidence rendering override.
// Keep this block after the legacy renderers so the compact two-line view wins.
function profitAttributionShortText(value, maxLen = 36) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    return text.length > maxLen ? `${text.slice(0, maxLen - 3)}...` : text;
}

function sideTone(side) {
    const value = String(side || '').toLowerCase();
    if (value.includes('close_long') || value.includes('long') || value.includes('做多') || value.includes('平多')) return 'long';
    if (value.includes('close_short') || value.includes('short') || value.includes('做空') || value.includes('平空')) return 'short';
    if (value.includes('hold') || value.includes('观望') || value.includes('观察')) return 'hold';
    return 'muted';
}

function sideZh(side) {
    const value = String(side || '').toLowerCase();
    if (value === 'long') return '做多';
    if (value === 'short') return '做空';
    if (value === 'close_long') return '平多';
    if (value === 'close_short') return '平空';
    if (value === 'hold') return '观望';
    return '-';
}

function stateStageLabel(stage) {
    const labels = {
        ai_analysis: 'AI 分析',
        strategy_arbitration: '策略仲裁',
        risk_check: '风控检查',
        exchange_submit: 'OKX 提交',
        exchange_confirm: '成交确认',
        local_sync: '本地同步',
    };
    return labels[stage] || stage || '-';
}

function stateStatusLabel(status) {
    const labels = {
        pending: '处理中',
        passed: '通过',
        blocked: '拦截',
        degraded_missing_probe: '模型缺失降级探针',
        failed: '失败',
        skipped: '跳过',
        completed: '完成',
    };
    return labels[status] || status || '-';
}

function confidenceZh(value) {
    const labels = { high: '高', medium: '中', low: '低' };
    return labels[String(value || '').toLowerCase()] || '中';
}

function renderProfitAttributionRecords(data) {
    const tbody = document.getElementById('profit-attribution-tbody');
    const paginationEl = document.getElementById('profit-attribution-record-pagination');
    if (!tbody) return;
    const rows = Array.isArray(data.records) ? data.records : [];
    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="profit-attribution-empty">暂无归因数据</td></tr>';
        if (paginationEl) paginationEl.style.display = 'none';
        return;
    }
    const total = rows.length;
    const totalPages = Math.max(Math.ceil(total / PROFIT_ATTRIBUTION_RECORD_PAGE_SIZE), 1);
    const page = Math.min(Math.max(Number(state.profitAttributionRecordPage || 1), 1), totalPages);
    state.profitAttributionRecordPage = page;
    const start = (page - 1) * PROFIT_ATTRIBUTION_RECORD_PAGE_SIZE;
    const pageRows = rows.slice(start, start + PROFIT_ATTRIBUTION_RECORD_PAGE_SIZE);
    tbody.innerHTML = pageRows.map(row => {
        const pnl = Number(row.realized_pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const stateSummary = row.decision_state?.summary || {};
        const stateText = stateSummary.final_stage
            ? `${stateStageLabel(stateSummary.final_stage)} / ${stateStatusLabel(stateSummary.final_status)}`
            : '无状态机记录';
        return `
            <tr>
                <td><span class="profit-attribution-time">${toBeijingTime(row.closed_at)}</span></td>
                <td><strong class="profit-attribution-symbol">${escHtml(row.symbol || '-')}</strong></td>
                <td><span class="profit-attribution-side ${sideTone(row.side)}">${escHtml(row.side_label || sideZh(row.side))}</span></td>
                <td class="profit-attribution-pnl" style="color:${pnlColor};">${signedMoney(pnl)} U</td>
                <td class="profit-attribution-hold">${Number(row.hold_minutes || 0).toFixed(1)} 分钟</td>
                <td>${renderProfitAttributionReason(row)}</td>
                <td class="profit-attribution-evidence-cell">${renderProfitAttributionEvidence(row)}</td>
                <td>${renderProfitAttributionChain(stateText, stateSummary.final_reason || '')}</td>
            </tr>`;
    }).join('');
    renderPagination('profit-attribution-record-pagination', page, totalPages, total, 'changeProfitAttributionRecordPage');
}

function renderProfitAttributionReason(row) {
    const notes = Array.isArray(row.notes) ? row.notes.filter(Boolean) : [];
    const visibleNotes = notes.slice(0, 2);
    const extra = notes.length > visibleNotes.length ? `<span>+${notes.length - visibleNotes.length}</span>` : '';
    const noteHtml = visibleNotes.length || extra
        ? `<div class="profit-attribution-note-list">${visibleNotes.map(note => `<span>${escHtml(profitAttributionShortText(note, 28))}</span>`).join('')}${extra}</div>`
        : '';
    return `
        <div class="profit-attribution-reason-cell">
            <strong class="profit-attribution-main-reason">${escHtml(row.main_reason || '-')}</strong>
            ${noteHtml}
            <em>置信度 ${confidenceZh(row.attribution_confidence)}</em>
        </div>`;
}

function profitAttributionEvidenceScoreChip(score) {
    if (!score || typeof score !== 'object') return null;
    const rawScore = Number(score.score);
    const effective = Number(score.effective_score);
    const multiplier = Number(score.size_multiplier);
    const scoreText = Number.isFinite(rawScore) ? rawScore.toFixed(0) : '-';
    const effectiveText = Number.isFinite(effective) ? effective.toFixed(0) : '-';
    const multiplierText = Number.isFinite(multiplier) ? `x${multiplier.toFixed(2)}` : 'x-';
    const tier = String(score.tier || '');
    const tierLabel = {
        degraded_missing_probe: '模型缺失降级探针',
        weak_conflict_probe: '弱冲突小仓探针',
        exploration: '探索小仓',
        small: '小仓',
        medium: '中等仓位',
        normal: '正常仓位',
        blocked: '硬风控阻断',
    }[tier] || tier || '-';
    const title = `证据分 ${scoreText} / 有效 ${effectiveText} / 仓位 ${multiplierText} / ${tierLabel}`;
    const tone = score.hard_block ? 'bad' : tier === 'degraded_missing_probe' ? 'warn' : effective >= 80 ? 'good' : effective >= 60 ? 'warn' : 'muted';
    return {
        tone,
        text: title,
        html: `<span class="profit-attribution-evidence-chip evidence-score profit-attribution-evidence-score ${tone}" title="${escHtml(title)}"><b>证据</b><em>${escHtml(scoreText)}</em><small>${escHtml(tierLabel === '-' ? multiplierText : tierLabel)}</small></span>`,
    };
}

function profitAttributionEvidenceChip(label, side, options = {}) {
    const main = options.main || sideZh(side);
    const visibleSub = compactProfitAttributionMetric(options.sub);
    const hasMain = Boolean(main && main !== '-');
    const hasEvidence = options.available !== false && (hasMain || visibleSub);
    if (!hasEvidence) return '';
    const tone = sideTone(side || main);
    const visibleSide = compactProfitAttributionSide(hasMain ? main : '未知');
    const sub = visibleSub ? `<small>${escHtml(visibleSub)}</small>` : '';
    const text = [label, main, options.sub && options.sub !== '-' ? options.sub : ''].filter(Boolean).join(' ');
    const typeClass = options.type ? ` evidence-${String(options.type).replace(/[^a-z0-9_-]/gi, '')}` : '';
    return {
        tone,
        text,
        html: `<span class="profit-attribution-evidence-chip${typeClass} ${tone}" title="${escHtml(text)}"><b>${escHtml(label)}</b><em>${escHtml(visibleSide)}</em>${sub}</span>`,
    };
}

function compactProfitAttributionSide(value) {
    const text = String(value || '').toLowerCase();
    if (text.includes('close_long') || text.includes('平多')) return '平多';
    if (text.includes('close_short') || text.includes('平空')) return '平空';
    if (text.includes('long') || text.includes('做多')) return '多';
    if (text.includes('short') || text.includes('做空')) return '空';
    if (text.includes('hold') || text.includes('观望')) return '观望';
    if (text.includes('观察')) return '观察';
    return String(value || '-');
}

function compactProfitAttributionMetric(value) {
    const text = String(value || '').trim();
    if (!text || text === '-') return '';
    const compact = text
        .replace(/%/g, '')
        .replace(/\s*\/\s*/g, '/')
        .replace(/([+-]?\d+\.\d{2})\d+/g, (match) => {
            const num = Number(match);
            if (!Number.isFinite(num)) return match;
            return num.toFixed(Math.abs(num) >= 10 ? 1 : 2).replace(/0+$/, '').replace(/\.$/, '');
        });
    return compact.length > 13 ? `${compact.slice(0, 10)}...` : compact;
}

// Final profit-attribution evidence renderer. It uses the backend
// evidence_status object so the cell shows source coverage instead of going
// blank when AI/ML/shadow samples are not matched.
function renderProfitAttributionEvidence(record) {
    const entryDecision = record?.entry_decision || {};
    const signals = record?.signals || {};
    const shadow = record?.shadow || {};
    const evidence = record?.evidence_status || {};
    const aiConfidence = Number(evidence.ai?.confidence ?? entryDecision?.confidence);
    const aiChip = profitAttributionEvidenceStatusChip('AI', evidence.ai, {
        type: 'ai',
        side: evidence.ai?.action || entryDecision?.action,
        main: evidence.ai?.action_label || entryDecision?.action_label,
        sub: Number.isFinite(aiConfidence) && aiConfidence > 0 ? aiConfidence.toFixed(2) : '',
        available: evidence.ai?.available === true || Boolean(entryDecision?.id),
    });
    const mlChip = profitAttributionEvidenceStatusChip('ML', evidence.ml, {
        type: 'ml',
        side: signals?.ml?.side || evidence.ml?.side,
        main: profitAttributionSideLabel(signals?.ml?.side || evidence.ml?.side),
        sub: signedPctValueLabel(signals?.ml?.expected_return_pct ?? evidence.ml?.expected_return_pct),
        available: signals?.ml?.available === true || evidence.ml?.available === true,
    });
    const shadowChip = profitAttributionEvidenceStatusChip('影子', evidence.shadow, {
        type: 'shadow',
        side: shadow?.best_action || evidence.shadow?.best_action,
        main: shadow?.best_action_label || evidence.shadow?.best_action_label,
        sub: shadow?.status === 'completed'
            ? `多${signedPctValueLabel(shadow?.long_return_pct)}/空${signedPctValueLabel(shadow?.short_return_pct)}`
            : (shadow?.status || evidence.shadow?.status || ''),
        available: Boolean(shadow?.id) || evidence.shadow?.available === true,
    });
    const rows = [[aiChip, mlChip], [shadowChip]];
    const supporting = [
        profitAttributionEvidenceStatusChip('盈利', evidence.server_profit, {
            type: 'server',
            side: signals?.server_profit?.side || evidence.server_profit?.side,
            main: profitAttributionSideLabel(signals?.server_profit?.side || evidence.server_profit?.side),
            sub: signedPctValueLabel(
                signals?.server_profit?.expected_return_pct ?? evidence.server_profit?.expected_return_pct
            ),
            available: signals?.server_profit?.available === true
                || evidence.server_profit?.available === true,
        }),
        profitAttributionEvidenceStatusChip('时序', evidence.timeseries, {
            type: 'timeseries',
            side: signals?.timeseries?.side || evidence.timeseries?.side,
            main: profitAttributionSideLabel(signals?.timeseries?.side || evidence.timeseries?.side),
            sub: signedPctValueLabel(
                signals?.timeseries?.expected_return_pct ?? evidence.timeseries?.expected_return_pct
            ),
            available: signals?.timeseries?.available === true
                || evidence.timeseries?.available === true,
        }),
        profitAttributionEvidenceStatusChip('情绪', evidence.sentiment, {
            type: 'sentiment',
            side: signals?.sentiment?.side || evidence.sentiment?.side,
            main: profitAttributionSideLabel(signals?.sentiment?.side || evidence.sentiment?.side),
            sub: Number.isFinite(Number(signals?.sentiment?.score))
                ? Number(signals.sentiment.score || 0).toFixed(3)
                : signedPctValueLabel(
                    signals?.sentiment?.expected_return_pct ?? evidence.sentiment?.expected_return_pct
                ),
            available: signals?.sentiment?.available === true
                || evidence.sentiment?.available === true,
        }),
    ];
    const scoreChip = profitAttributionEvidenceScoreChip(
        entryDecision?.evidence_score || entryDecision?.opportunity_score?.evidence_score
    );
    if (scoreChip) supporting.push(scoreChip);
    const title = rows.flat().concat(supporting).map(chip => chip.text).filter(Boolean).join(' | ');
    return `
        <div class="profit-attribution-evidence-rail" title="${escHtml(title)}">
            ${rows.map(row => `<div class="profit-attribution-evidence-row">${row.map(chip => chip.html).join('')}</div>`).join('')}
        </div>`;
}

function profitAttributionEvidenceStatusChip(label, status, options = {}) {
    const sourceStatus = status || {};
    const available = sourceStatus.available === true || options.available === true;
    const side = options.side || sourceStatus.side || sourceStatus.action || sourceStatus.best_action || '';
    const main = options.main || sourceStatus.action_label || sourceStatus.best_action_label
        || profitAttributionSideLabel(side);
    const sub = compactProfitAttributionMetric(options.sub || '');
    const typeClass = options.type ? ` evidence-${String(options.type).replace(/[^a-z0-9_-]/gi, '')}` : '';
    if (!available) {
        const reason = sourceStatus.missing_reason || `${label}证据未匹配`;
        const missingLabel = profitAttributionMissingLabel(reason);
        return {
            tone: 'missing',
            text: `${label} ${reason}`,
            html: `<span class="profit-attribution-evidence-chip${typeClass} missing" title="${escHtml(reason)}"><b>${escHtml(label)}</b><em>${escHtml(missingLabel)}</em></span>`,
        };
    }
    return profitAttributionEvidenceChip(label, side, {
        ...options,
        main,
        sub,
        available: true,
    });
}

function profitAttributionMissingLabel(reason) {
    const text = String(reason || '').toLowerCase();
    if (text.includes('未保存') || text.includes('not_saved')) return '未保存';
    if (text.includes('未匹配') || text.includes('not_matched')) return '未匹配';
    if (text.includes('等待') || text.includes('pending')) return '等待';
    return '无证据';
}

function profitAttributionSideLabel(value) {
    const text = String(value || '').toLowerCase();
    if (text === 'long' || text === 'open_long') return '多';
    if (text === 'short' || text === 'open_short') return '空';
    if (text === 'close_long') return '平多';
    if (text === 'close_short') return '平空';
    if (text === 'hold' || text === 'wait' || text === 'observe') return '观察';
    return String(value || '-');
}

function renderProfitAttributionChain(stateText, reason) {
    return `
        <div class="profit-attribution-chain">
            <strong>${escHtml(stateText || '-')}</strong>
            ${reason ? `<span title="${escHtml(reason)}">${escHtml(profitAttributionShortText(reason, 42))}</span>` : ''}
        </div>`;
}
async function fetchStrategyLearning() {
    const hoursEl = document.getElementById('strategy-learning-hours');
    const hours = Number(hoursEl?.value || 168);
    const limit = hours <= 24 ? 500 : hours <= 72 ? 800 : 1000;
    try {
        const data = await fetchJSON(`/api/strategy-learning?mode=${state.mode || 'paper'}&hours=${hours}&limit=${limit}&detail=summary`);
        state.strategyLearning = data;
        renderStrategyLearning(data);
    } catch (err) {
        const summary = document.getElementById('strategy-learning-summary');
        if (summary) {
            summary.innerHTML = `<div class="opening-funnel-verdict opening-funnel-warn"><strong>\u7b56\u7565\u8c03\u5ea6\u52a0\u8f7d\u5931\u8d25</strong><span>${escHtml(err.message || err)}</span></div>`;
        }
    }
}

function renderStrategyLearning(data) {
    const summary = document.getElementById('strategy-learning-summary');
    if (!summary) return;
    const profile = data?.schedule?.active_profile || data?.active_profile || {};
    summary.innerHTML = `<div class="opening-funnel-empty">\u7b56\u7565\u63a7\u5236\u53f0\u6570\u636e\u5df2\u52a0\u8f7d\uff1a${escHtml(profile.label || profile.id || '\u5f53\u524d\u57fa\u7ebf')}</div>`;
}

function strategyLearningSetActionState(profileId, status, message = '') {
    if (!window.strategyLearningActionState) window.strategyLearningActionState = {};
    const key = String(profileId || 'auto');
    window.strategyLearningActionState[key] = {
        status,
        message,
        updatedAt: Date.now(),
    };
    if (state.strategyLearning && typeof renderStrategyLearning === 'function') {
        renderStrategyLearning(state.strategyLearning);
    }
}

async function strategyLearningWriteRequest(profileId, statusText, requestFactory) {
    strategyLearningSetActionState(profileId, 'loading', statusText);
    try {
        const res = await requestFactory();
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(apiErrorText(data, res.statusText || '请求失败'));
        strategyLearningSetActionState(profileId, 'success', '已生效，正在刷新调度状态');
        await fetchStrategyLearning();
        return data;
    } catch (error) {
        strategyLearningSetActionState(profileId, 'error', error.message || String(error));
        throw error;
    }
}

async function setStrategyLearningProfileDisabled(profileId, disabled) {
    await strategyLearningWriteRequest(
        profileId,
        disabled ? '正在禁用策略...' : '正在取消禁用...',
        () => fetchWithAuth(`/api/strategy-learning/profiles/${encodeURIComponent(profileId)}/disabled?disabled=${disabled}`, dashboardWriteOptions({ method: 'POST' })),
    );
}

async function activateStrategyLearningProfile(profileId) {
    await strategyLearningWriteRequest(
        profileId,
        '正在人工指定策略...',
        () => fetchWithAuth(`/api/strategy-learning/profiles/${encodeURIComponent(profileId)}/activate`, dashboardWriteOptions({ method: 'POST' })),
    );
}

async function clearStrategyLearningManualOverride() {
    await strategyLearningWriteRequest(
        'auto',
        '正在恢复系统自动调度...',
        () => fetchWithAuth('/api/strategy-learning/rollback', dashboardWriteOptions({ method: 'POST' })),
    );
}

async function rollbackStrategyLearning() {
    await clearStrategyLearningManualOverride();
}

