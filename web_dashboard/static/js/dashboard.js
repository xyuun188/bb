/**
 * Main dashboard logic.
 * Connects WebSocket, polls REST API, renders all components.
 */

// State
const state = {
    mode: 'paper',
    paused: false,
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
    modeDecisionsTotal: 0,
    decisionsTotal: 0,
    todayDecisionsTotal: 0,
    tradesTotal: 0,
    openPositionsTotal: 0,
    decisionInterval: null,
    allTrades: [],
    tradesPage: 1,
    tradesPageMode: '',
    positionsPage: 1,
    positionsTotal: 0,
    openPositions: [],
    protectionInventory: null,
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
    lastFreshAccountBalances: {},
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
    trainingEffectivenessReport: null,
    trainingEffectivenessFilters: { mode: 'all', side: 'all', symbol: '' },
    shadowBacktests: [],
    shadowBacktestPage: 1,
    shadowBacktestTotal: 0,
    shadowBacktestStatus: '',
    mlSignalStatus: null,
    localAIToolsStatus: null,
    modelTrainingRegistry: null,
    dataCollectionStatus: null,
    dataCollectionSettingsLoaded: false,
    dataCollectionSettingsDirty: false,
    dataCollectionSettingsSaving: false,
    serverMonitorStatus: null,
    systemAuditStatus: null,
    serverMonitorTab: 'self-check',
    systemSelfCheck: null,
    mlSignalRecords: [],
    mlSignalPage: 1,
    tradesTotalPages: 1,
    openingFunnel: null,
    profitAttribution: null,
    profitAttributionView: 'overview',
    profitAttributionRecordPage: 1,
    runtimeStartedAt: null,
    lastStatsSource: '',
    lastStatsAt: 0,
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
const closingPositionIds = new Set();
let closingAllPositions = false;
const positionLinkedOrdersByGroup = new Map();
let serverMonitorRefreshInFlight = null;
let systemAuditRefreshInFlight = null;
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
    initDataCollectionSettingsForm();
    initPositionActions();
    initDashboardUserActions();
    initModalActionButtons();
    initServerMonitorTabs();
    initPaginationControls();
    fetchDashboardSummary();
    fetchPnlHistory();
    fetchRecentDecisions();
    fetchRecentExecutions();
    fetchRiskEvents();
    fetchDashboardAuthStatus();
    setInterval(() => {
        if (!document.hidden && isPageActive('dashboard')) {
            void fetchDashboardSummary();
        }
    }, 10000);
    document.addEventListener('visibilitychange', recoverDashboardSummaryPolling);
    window.addEventListener('online', recoverDashboardSummaryPolling);
    setInterval(updateRuntimeClock, 1000);
    setInterval(() => {
        if (!document.hidden && isPageActive('dashboard')) fetchPnlHistory();
    }, 60000);
    setInterval(() => {
        if (!document.hidden && isPageActive('dashboard')) fetchRecentDecisions();
    }, 30000);
    setInterval(() => {
        if (!document.hidden && isPageActive('dashboard')) fetchRecentExecutions();
    }, 30000);
    setInterval(() => {
        if (!document.hidden && isPageActive('trades')) fetchTrades();
    }, 60000);
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
    setInterval(() => {
        if (isPageActive('data-collection')) {
            fetchDataCollectionStatus({ silent: true });
        }
    }, 60000);
    setInterval(() => {
        if (isPageActive('system-audit')) {
            fetchSystemAudit({ silent: true });
        }
    }, 60000);
    setInterval(() => {
        if (isPageActive('ml-signal')) {
            fetchMLSignalDashboard();
        }
    }, 60000);
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
            updateStats(data.stats || {}, 'ws');
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

const inflightJSONRequests = new Map();
const paginatedRequestVersions = new Map();
const JSON_REQUEST_TIMEOUT_MS = 20000;
const DASHBOARD_BALANCE_FIELDS = [
    'available_balance',
    'okx_available_balance',
    'remaining_allocation',
    'current_balance',
    'tradeable_balance',
    'account_equity',
    'okx_equity_balance',
    'equity',
    'wallet_balance',
    'used_margin',
    'okx_used_balance',
    'position_margin_used',
    'paper_execution_available_balance',
    'paper_execution_used_margin',
];

async function fetchJSON(url) {
    const requestKey = String(url);
    const existingEntry = inflightJSONRequests.get(requestKey);
    if (existingEntry) return existingEntry.promise;

    const controller = typeof AbortController === 'function' ? new AbortController() : null;
    let timedOut = false;
    const timeoutId = controller ? window.setTimeout(() => {
        timedOut = true;
        controller.abort();
    }, JSON_REQUEST_TIMEOUT_MS) : null;
    const request = (async () => {
        try {
            const res = await fetch(url, { cache: 'no-store', ...(controller ? { signal: controller.signal } : {}) });
            const data = await res.json().catch(() => ({}));
            if (res.status === 401) {
                const message = apiErrorText(data, '登录已过期，请重新登录。');
                redirectToLogin(message);
                throw new Error(message);
            }
            if (!res.ok) {
                console.error(`Fetch failed: ${url}`, data);
                throw new Error(apiErrorText(data, res.statusText || '请求失败'));
            }
            return data;
        } catch (e) {
            if (timedOut) {
                const timeoutError = new Error(`请求超过 ${JSON_REQUEST_TIMEOUT_MS / 1000} 秒，已中止并等待自动重试。`);
                console.warn(`Fetch timed out: ${url}`, timeoutError);
                throw timeoutError;
            }
            console.error(`Fetch failed: ${url}`, e);
            throw e;
        } finally {
            if (timeoutId !== null) window.clearTimeout(timeoutId);
        }
    })();
    const entry = { promise: request, controller };
    inflightJSONRequests.set(requestKey, entry);
    try {
        return await request;
    } finally {
        if (inflightJSONRequests.get(requestKey) === entry) {
            inflightJSONRequests.delete(requestKey);
        }
    }
}

function cancelInflightJSONRequest(url) {
    const requestKey = String(url);
    const entry = inflightJSONRequests.get(requestKey);
    if (!entry) return;
    inflightJSONRequests.delete(requestKey);
    if (entry.controller && !entry.controller.signal.aborted) entry.controller.abort();
}

async function fetchLatestPageJSON(requestKey, url) {
    const key = String(requestKey);
    const version = Number(paginatedRequestVersions.get(key) || 0) + 1;
    paginatedRequestVersions.set(key, version);
    const data = await fetchJSON(url);
    return paginatedRequestVersions.get(key) === version ? data : null;
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
    try {
        const data = await fetchJSON('/api/dashboard/summary');
        if (!data) return null;

        const executionAccount = preserveLastGoodAccountBalance(
            data.execution_account || {},
            data.mode,
        );
        const accounts = (data.accounts || []).map(account => (
            preserveLastGoodAccountBalance(account, data.mode)
        ));
        const normalizedData = {
            ...data,
            execution_account: executionAccount,
            accounts,
        };
        updateModeDisplay(normalizedData.mode, normalizedData.paused);
        updateExecutionAccountPanel(executionAccount);
        updateAccounts(accounts, executionAccount || null);
        updateMarketData(normalizedData.market || {}, accounts);
        updateStats(normalizedData, 'summary');
        updateDashboardDecisionCounts(normalizedData);
        updateSymbolCount();
        fetchModeCounts();
        return normalizedData;
    } catch (error) {
        console.warn('Dashboard summary refresh failed; the next poll will retry automatically.', error);
        return null;
    }
}

function preserveLastGoodAccountBalance(rawAccount, fallbackMode = 'paper') {
    if (!rawAccount || !Object.keys(rawAccount).length) return rawAccount || {};
    const account = { ...rawAccount };
    const mode = account.mode === 'live' || fallbackMode === 'live' ? 'live' : 'paper';
    const hasBalanceValue = DASHBOARD_BALANCE_FIELDS.some(field => (
        valueNumber(account[field]) !== null
    ));
    const isFresh = hasBalanceValue
        && !account.balance_error
        && account.balance_snapshot_stale !== true;

    if (isFresh) {
        state.lastFreshAccountBalances[mode] = {
            capturedAtMs: Date.now(),
            values: Object.fromEntries(
                DASHBOARD_BALANCE_FIELDS
                    .filter(field => valueNumber(account[field]) !== null)
                    .map(field => [field, account[field]]),
            ),
        };
        return account;
    }

    const cached = state.lastFreshAccountBalances[mode];
    const needsClientFallback = Boolean(account.balance_error)
        && account.balance_snapshot_stale !== true;
    let usedClientFallback = false;
    if (cached && (!hasBalanceValue || needsClientFallback)) {
        for (const [field, value] of Object.entries(cached.values)) {
            if (needsClientFallback || valueNumber(account[field]) === null) account[field] = value;
        }
        usedClientFallback = true;
    }
    if ((account.balance_error || usedClientFallback) && (hasBalanceValue || usedClientFallback)) {
        account.balance_snapshot_stale = true;
        if (usedClientFallback) {
            account.balance_snapshot_age_seconds = Math.max(
                (Date.now() - cached.capturedAtMs) / 1000,
                Number(account.balance_snapshot_age_seconds || 0),
            );
            account.balance_source = mode === 'live'
                ? 'OKX 实盘最近成功快照'
                : 'OKX 模拟盘最近成功快照';
        }
    }
    return account;
}

function recoverDashboardSummaryPolling() {
    if (document.hidden || !isPageActive('dashboard')) return;
    cancelInflightJSONRequest('/api/dashboard/summary');
    void fetchDashboardSummary();
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
    state.dailyPnlMeta = data;
    const subtitle = document.getElementById('daily-pnl-subtitle');
    if (subtitle) {
        const equityStart = data.okx_equity_series_start_date || '';
        const equityScope = data.okx_equity_series_complete
            ? '\u4e09\u671f\u5b8c\u6574 OKX \u6743\u76ca\u5feb\u7167'
            : (equityStart ? `OKX \u6743\u76ca\u5feb\u7167\u81ea ${equityStart} \u5f00\u59cb` : 'OKX \u6743\u76ca\u5feb\u7167\u672a\u7559\u5b58');
        subtitle.textContent = `${mode === 'live' ? '实盘' : '模拟盘'} · 北京时间 ${data.start_date || ''} 至 ${data.end_date || ''} · ${equityScope}`;
    }
    const headers = document.querySelectorAll('#page-daily-pnl thead th');
    if (headers.length >= 6) {
        headers[3].textContent = '\u6743\u5a01\u5df2\u7ed3\u7b97\u51c0\u76c8\u4e8f';
        headers[5].textContent = data.okx_equity_series_complete
            ? '\u4e09\u671f\u7d2f\u8ba1\u6743\u76ca\u53d8\u5316'
            : `\u81ea ${data.okx_equity_series_start_date || '--'} \u6743\u76ca\u53d8\u5316`;
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
        dtEl.textContent = state.modeDecisionsTotal + ' / ' + state.openPositionsTotal;
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
        state.modeDecisionsTotal = Number(decData.total ?? decData.count ?? 0) || 0;
        updateDecisionBadge(state.modeDecisionsTotal);
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
    const data = await fetchLatestPageJSON(
        'trades',
        `/api/trades?limit=${PAGE_SIZE}&mode=${state.mode}&page=${state.tradesPage}`,
    );
    if (!data) return;
    updateTradeTable(data.trades || [], state.mode, data.total ?? data.count);
}

async function fetchPositionTickerSnapshot() {
    const data = await fetchJSON(`/api/dashboard/market?_=${Date.now()}`);
    if (!data) return;
    updateMarketData(data, state.accounts || []);
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
function updateModeDisplay(mode, paused) {
    state.mode = mode;
    state.paused = Boolean(paused);

    const badge = document.getElementById('mode-badge');
    if (state.paused) {
        badge.textContent = '已暂停新开仓';
        badge.className = 'status-badge status-paused';
    } else if (mode === 'live') {
        badge.textContent = '实盘';
        badge.className = 'status-badge status-live';
    } else {
        badge.textContent = '模拟盘';
        badge.className = 'status-badge status-paper';
    }
    const pauseBtn = document.getElementById('pause-btn');
    if (pauseBtn) {
        pauseBtn.textContent = state.paused ? '恢复新开仓分析' : '暂停新开仓分析';
        pauseBtn.className = state.paused ? 'btn pause-btn active' : 'btn pause-btn';
        pauseBtn.title = state.paused
            ? '当前已暂停新市场分析和新开仓；已有仓位仍继续复盘和平仓。'
            : '暂停后不再分析新交易对或提交新开仓，已有仓位仍继续风控。';
    }
    const pauseBanner = document.getElementById('dashboard-pause-banner');
    if (pauseBanner) pauseBanner.hidden = !state.paused;

    document.querySelectorAll('.mode-btn[data-mode]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    updateModeButtonAvailability();

    const marketScanLabel = document.getElementById('scan-mode-label');
    if (marketScanLabel) {
        marketScanLabel.textContent = state.paused
            ? '已暂停新市场分析 · 持仓风控继续'
            : '自动扫描全市场 · 智能调度';
    }
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
        const change = t.change_24h ?? t.change24h ?? t.change_24h_pct ?? t.percentage ?? 0;
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
    const n = valueNumber(value);
    if (n === null) return '--';
    return `${n >= 0 ? '+' : ''}${fmtMoney(n)}`;
}

function signedMoneyWithUnit(value, unit = 'USDT') {
    const text = signedMoney(value);
    return text === '--' ? '--' : `${text} ${unit}`;
}

function signedMoneyColor(value) {
    const n = valueNumber(value);
    if (n === null) return 'var(--text-muted)';
    return n >= 0 ? 'var(--green)' : 'var(--red)';
}

function dailyPnlOkxSnapshotMissing(row) {
    return row?.okx_equity_source === 'okx_snapshot_missing'
        || (row?.okx_equity == null && row?.okx_equity_pnl == null);
}

function dailyPnlEquityDisplay(row, field) {
    if (dailyPnlOkxSnapshotMissing(row)) {
        const startDate = row?.okx_equity_series_start_date;
        const message = startDate && String(row?.date || '') < String(startDate)
            ? `\u672a\u7559\u5b58 ${startDate} \u4e4b\u524d\u7684\u771f\u5b9e OKX \u6743\u76ca\u5feb\u7167`
            : '\u5f53\u65e5\u771f\u5b9e OKX \u6743\u76ca\u5feb\u7167\u7f3a\u5931';
        return `<span style="color:var(--text-muted);">${escHtml(message)}</span>`;
    }
    const value = valueNumber(row?.[field]);
    return `<span style="color:${signedMoneyColor(value)};">${signedMoneyWithUnit(value)}</span>`;
}

function dailyPnlMissingSnapshotNotice(row) {
    if (!dailyPnlOkxSnapshotMissing(row)) return '';
    return `
        <div class="info-banner" style="margin:8px 0;">
            \u5f53\u65e5\u6ca1\u6709\u7559\u5b58\u771f\u5b9e OKX \u6743\u76ca\u5feb\u7167\u3002\u7cfb\u7edf\u4e0d\u4f7f\u7528\u56fa\u5b9a\u4f59\u989d\u3001\u672c\u5730\u4ea4\u6613\u76c8\u4e8f\u6216 OKX \u8d26\u5355\u53d8\u52a8\u5012\u63a8\u5386\u53f2\u8d26\u6237\u6743\u76ca\u3002\u4e0b\u65b9\u5f00\u5e73\u4ed3\u6d3b\u52a8\u53ea\u5c55\u793a OKX \u6210\u4ea4\u4e8b\u5b9e\uff0c\u76c8\u4e8f\u4ecd\u53ea\u7edf\u8ba1\u6743\u5a01\u5df2\u7ed3\u7b97\u8bb0\u5f55\u3002
        </div>
    `;
}

function updateModelRankings(rankings) {
    state.rankings = rankings || [];
}

function accountMoneyText(value, account = null) {
    if (account && account.balance_error && account.balance_snapshot_stale !== true) return '--';
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
    const phase3TotalPnl = valueNumber(account.phase3_equity_pnl);
    const phase3ObservedStart = account.phase3_equity_observed_start_date || '';
    const phase3EquityLabel = account.phase3_equity_series_complete
        ? '\u4e09\u671fOKX\u6743\u76ca\u53d8\u5316'
        : (phase3ObservedStart
            ? `\u81ea ${phase3ObservedStart} \u9996\u4e2aOKX\u5feb\u7167\u6743\u76ca\u53d8\u5316`
            : '\u4e09\u671fOKX\u6743\u76ca\u53d8\u5316\u4e0d\u53ef\u7528');
    const todayTotalPnl = valueNumber(account.today_equity_pnl);
    const remainingAllocation = valueNumber(account.available_balance ?? account.okx_available_balance ?? account.remaining_allocation);
    const accountEquity = valueNumber(account.account_equity ?? account.okx_equity_balance ?? account.equity ?? account.wallet_balance);
    const positionMarginUsed = valueNumber(
        account.used_margin ?? account.okx_used_balance ?? account.position_margin_used ?? account.paper_execution_used_margin
    ) || 0;
    const balanceSource = account.balance_source || (account.balance_snapshot_stale ? 'OKX 缓存快照' : 'OKX 权威账户');
    const accountBalanceLabel = account.mode === 'live' ? 'OKX 实盘' : 'OKX 模拟盘';
    const pauseNote = account.risk_paused
        ? `<div class="exec-risk-note paused">已暂停分析新交易对：${escHtml(translatePauseReason(account.risk_pause_reason || '账户触发风险限制'))}</div>`
        : '<div class="exec-risk-note">账户余额、权益、订单和持仓只以 OKX 实时/快照事实为准；本地不再使用固定金额或虚拟余额算账。</div>';

    container.innerHTML = `
        <div class="exec-account-card">
            <div class="exec-account-head">
                <div>
                    <div class="exec-account-name">${escHtml(account.account_name || '多专家执行账户')}</div>
                    <div class="exec-account-mode">${modeLabel} · ${escHtml(balanceSource)}${account.balance_snapshot_stale ? ` · 缓存 ${monitorNumber(account.balance_snapshot_age_seconds, 1)}秒` : ''}</div>
                </div>
                <span class="badge ${account.risk_paused ? 'badge-short' : 'badge-long'}">${account.risk_paused ? '暂停开新仓' : '可分析'}</span>
            </div>
            <div class="exec-status-grid">
                <div class="exec-status-cell"><span>${accountBalanceLabel}可交易余额</span><strong>${accountMoneyText(remainingAllocation, account)} USDT</strong></div>
                <div class="exec-status-cell"><span>${accountBalanceLabel}当前账户权益</span><strong>${accountMoneyText(accountEquity, account)} USDT</strong></div>
                <div class="exec-status-cell"><span>持仓保证金占用</span><strong>${accountMoneyText(positionMarginUsed, account)} USDT</strong></div>
                <div class="exec-status-cell"><span>浮动盈亏</span><strong style="color:${unrealizedPnl >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(unrealizedPnl)} USDT</strong></div>
                <div class="exec-status-cell"><span>今日OKX权益变化</span><strong style="color:${signedMoneyColor(todayTotalPnl)};">${signedMoneyWithUnit(todayTotalPnl)}</strong></div>
                <div class="exec-status-cell"><span>${phase3EquityLabel}</span><strong style="color:${signedMoneyColor(phase3TotalPnl)};">${signedMoneyWithUnit(phase3TotalPnl)}</strong></div>
            </div>
            ${account.balance_error
                ? `<div class="exec-risk-note paused">${escHtml(account.balance_error)}</div>`
                : account.balance_warning
                    ? `<div class="exec-risk-note">${escHtml(account.balance_warning)}</div>`
                    : pauseNote}
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
    const phase3TotalPnl = valueNumber(account.phase3_equity_pnl);
    const unrealizedPnl = valueNumber(account.unrealized_pnl) || 0;
    const todayTotalPnl = valueNumber(account.today_equity_pnl);
    const pnlColor = signedMoneyColor(phase3TotalPnl);
    const accountBalanceLabel = account.mode === 'live' ? 'OKX 实盘' : 'OKX 模拟盘';
    container.innerHTML = `
        <div class="acct-row">
            <div class="acct-main">
                <div class="acct-name">${escHtml(account.account_name || account.model_name || '多专家执行账户')}</div>
                <div style="font-size:12px;color:var(--text);font-weight:700;">${accountBalanceLabel}可交易余额 ${accountMoneyText(remainingAllocation, account)} USDT</div>
                <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">${accountBalanceLabel}当前账户权益 ${accountMoneyText(accountEquity, account)} | 今日OKX权益变化（北京时间）${signedMoney(todayTotalPnl)}</div>
                ${account.balance_error
                    ? `<div style="font-size:10px;color:var(--red);margin-top:4px;">${escHtml(account.balance_error)}</div>`
                    : account.balance_warning
                        ? `<div style="font-size:10px;color:var(--yellow);margin-top:4px;">${escHtml(account.balance_warning)}</div>`
                        : ''}
            </div>
            <div class="acct-side">
                <div class="acct-side-label">三期OKX权益变化</div>
                <div class="acct-side-value" style="color:${pnlColor};">${signedMoneyWithUnit(phase3TotalPnl)}</div>
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
        const positionChange = position.change_24h ?? position.change24h ?? position.change_24h_pct ?? position.percentage ?? null;
        const previousChange = previous.change_24h ?? previous.change24h ?? previous.change_24h_pct ?? previous.percentage ?? 0;
        const change = positionChange !== null && Number(positionChange) !== 0
            ? positionChange
            : previousChange;
        tickers[position.symbol] = {
            price,
            change_24h: change,
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
        ? Object.fromEntries(Object.entries(positionTickers).map(([symbol, ticker]) => {
            const marketTicker = marketPositionTickers[symbol] || {};
            const tickerChange = ticker.change_24h ?? ticker.change24h ?? ticker.change_24h_pct ?? ticker.percentage ?? null;
            const marketChange = marketTicker.change_24h ?? marketTicker.change24h ?? marketTicker.change_24h_pct ?? marketTicker.percentage ?? null;
            const shouldKeepMarketChange = marketChange !== null && (tickerChange === null || Number(tickerChange) === 0);
            return [
                symbol,
                {
                    ...marketTicker,
                    ...ticker,
                    change_24h: shouldKeepMarketChange ? marketChange : (tickerChange ?? marketChange ?? 0),
                    volume_24h: ticker.volume_24h || marketTicker.volume_24h || 0,
                    bid: ticker.bid || marketTicker.bid || 0,
                    ask: ticker.ask || marketTicker.ask || 0,
                },
            ];
        }))
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
                                <td><span class="badge badge-${analysisDisplayAction(d.action, d)}">${analysisActionLabel(d.action, d)}</span></td>
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
    return fetchTrades();
}

function parsedRuntimeSeconds(stats) {
    const explicit = Number(stats?.uptime_seconds || 0);
    if (Number.isFinite(explicit) && explicit > 0) return Math.floor(explicit);
    const startedAt = stats?.started_at || state.runtimeStartedAt;
    if (startedAt) {
        const startedMs = Date.parse(startedAt);
        if (Number.isFinite(startedMs)) {
            return Math.max(0, Math.floor((Date.now() - startedMs) / 1000));
        }
    }
    return 0;
}

function updateStats(stats, source = 'unknown') {
    if (!stats || typeof stats !== 'object') return;
    const now = Date.now();
    const isFullSummary = source === 'summary' || stats.uptime_source === 'split_process_heartbeat';
    const hasRuntimeFields = Boolean(
        stats.started_at ||
        stats.heartbeat_at ||
        stats.last_heartbeat_at ||
        stats.uptime_seconds ||
        stats.decision_interval
    );
    if (
        source === 'ws' &&
        state.lastStatsSource === 'summary' &&
        now - Number(state.lastStatsAt || 0) < 15000 &&
        !hasRuntimeFields
    ) {
        return;
    }
    if (stats.running !== undefined) {
        const uptimeEl = document.getElementById('stat-uptime');
        if (uptimeEl) {
            if (stats.started_at) state.runtimeStartedAt = stats.started_at;
            const uptimeSeconds = parsedRuntimeSeconds(stats);
            uptimeEl.textContent = uptimeSeconds > 0
                ? formatUptime(uptimeSeconds)
                : (stats.uptime_source === 'split_process_heartbeat' ? '\u72ec\u7acb\u8fdb\u7a0b' : formatUptime(0));
        }
        const autoStatusStats = (
            source === 'ws' &&
            state.lastStatsSource === 'summary' &&
            now - Number(state.lastStatsAt || 0) < 15000
        )
            ? {
                ...stats,
                decision_interval: state.decisionInterval,
                market_loop_interval_seconds: stats.market_loop_interval_seconds
                    || state.lastStats?.market_loop_interval_seconds,
                position_loop_interval_seconds: stats.position_loop_interval_seconds
                    || state.lastStats?.position_loop_interval_seconds,
                market_round_time_budget_seconds: stats.market_round_time_budget_seconds
                    || state.lastStats?.market_round_time_budget_seconds,
            }
            : stats;
        if (isFullSummary || hasRuntimeFields) {
            state.lastStats = { ...(state.lastStats || {}), ...stats };
        }
        updateAutoStatus(autoStatusStats);
    }
    if (isFullSummary || hasRuntimeFields) {
        state.lastStatsSource = source;
        state.lastStatsAt = now;
    }
}

function updateRuntimeClock() {
    if (!state.runtimeStartedAt) return;
    const uptimeEl = document.getElementById('stat-uptime');
    if (!uptimeEl) return;
    const seconds = parsedRuntimeSeconds({});
    if (seconds > 0) {
        uptimeEl.textContent = formatUptime(seconds);
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
        specialist_shadow_inference: '\u5df2\u8fd4\u56de\uff08\u89c2\u5bdf\u6001\uff09',
        shadow_observation: '\u5df2\u8fd4\u56de\uff08\u89c2\u5bdf\u6001\uff09',
        trained_calibrator: '\u5df2\u8fd4\u56de\uff08\u6821\u51c6\u6001\uff09',
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
    return renderRiskAlerts(state.riskEvents || []);
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

function initDataCollectionSettingsForm() {
    document.addEventListener('input', event => {
        const target = event.target;
        if (target?.matches?.('[data-data-collection-setting], #data-external-source-list input')) {
            markDataCollectionSettingsDirty();
        }
    });
    document.addEventListener('change', event => {
        const target = event.target;
        if (target?.matches?.('[data-data-collection-setting], #data-external-source-list input')) {
            markDataCollectionSettingsDirty();
        }
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
    if (selected === 'trading') fetchTradingParams();
    if (selected === 'external-events') fetchDataCollectionSettings({ silent: true });
    if (selected === 'vector-memory') refreshVectorMemoryStatus({ silent: true });
}

function loadPageData(page) {
    if (page === 'dashboard') {
        fetchDashboardSummary();
        fetchPnlHistory();
        fetchRecentDecisions();
        fetchRecentExecutions();
    }
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
    if (page === 'expert-memory') {
        fetchExpertMemories();
        fetchTrainingEffectivenessReport();
    }
    if (page === 'shadow-backtest') fetchShadowBacktests();
    if (page === 'ml-signal') fetchMLSignalDashboard();
    if (page === 'data-collection') fetchDataCollectionStatus();
    if (page === 'system-audit') fetchSystemAudit();
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
        fetchDataCollectionSettings({ silent: true });
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
            if (isPageActive('expert-memory')) {
                fetchExpertMemories();
                fetchTrainingEffectivenessReport();
            }
            fetchPositions();
            fetchPositionHistory();
            if (isPageActive('daily-pnl')) fetchDailyPnlRecords();
        });
    });
}

async function togglePause() {
    const endpoint = state.paused ? '/api/control/resume' : '/api/control/pause';
    const btn = document.getElementById('pause-btn');
    if (btn) {
        btn.disabled = true;
        btn.textContent = state.paused ? '正在恢复...' : '正在暂停...';
    }
    try {
        const response = await fetchWithAuth(endpoint, dashboardWriteOptions({
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        }));
        const data = await response.json().catch(() => null);
        if (data && data.state) {
            updateModeDisplay(data.state.mode, data.state.paused);
        }
        await fetchDashboardSummary();
    } finally {
        if (btn) btn.disabled = false;
    }
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
    const data = await fetchLatestPageJSON('decisions', '/api/decisions?' + qs);
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
    const contract = score.return_distribution_contract || {};
    const observed = Number(contract.raw_expected_return_pct);
    return {
        label: '标准分布观察值',
        value: Number.isFinite(observed) ? observed : null,
    };
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
            const contract = component.return_distribution_contract || null;
            const objective = Number(contract?.objective_expected_return_pct);
            const weight = Number(component.production_weight);
            const available = component.available !== false;
            const pieces = [];
            pieces.push(distributionSummaryText(contract));
            if (Number.isFinite(weight)) pieces.push(`生产权重 ${opportunityScoreValue(weight, 4)}`);
            if (component.production_eligible !== true) pieces.push('仅观察');
            if (!available) pieces.push('当前未参与');
            return {
                label: component.label || component.key || '收益来源',
                value: Number.isFinite(objective) ? objective : 0,
                text: pieces.join(' · ') || '-',
                tone: component.production_eligible === true && objective > 0 ? 'good' : 'muted',
                note: component.eligibility_reason || component.note || '',
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

function evidencePercentLabel(value, digits = 1) {
    const num = Number(value);
    return Number.isFinite(num) ? `${(num * 100).toFixed(digits)}%` : '-';
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
    const winRate = Number(score.diagnostic_win_rate || 0) * 100;
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
                ${decisionMetricItem('ML 胜率（诊断）', `${opportunityScoreValue(winRate, 1)}%`, '不参与评分、放行、仓位或杠杆')}
                ${decisionMetricItem('仓位 x 杠杆', opportunityScoreValue(score.size_x_leverage, 4))}
                ${decisionMetricItem('手续费+滑点', `${opportunityScoreValue(feeAndSlippage, 4)}%`)}
            </div>
            ${formulaHtml}
            <div class="decision-score-reason">
                <span>排序原因</span>
                <div>${escapeMultiline(reason)}</div>
            </div>
        </div>
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
    const needsWideModal = Boolean(
        decision.opportunity_score ||
        decision.reasoning ||
        decision.execution_reason
    );
    setDecisionModalWide(needsWideModal);

    document.getElementById('decision-reason-title').textContent = title;
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="decision-detail-stack">
        <div class="reason-block">
            <div class="reason-label">${decision.was_executed ? '执行状态' : '未执行原因'}</div>
            <div>${escapeMultiline(primaryReason)}</div>
            ${executedInfo}
        </div>
        ${opportunityHtml}
        ${aiReasoning}
        </div>
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
    return fetchAllDecisions();
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
    const data = await fetchLatestPageJSON(
        'analysis',
        '/api/analysis-records?' + params.toString(),
    );
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

function analysisPreExpertSkip(record) {
    const status = record?.expert_call_status || {};
    if (status && status.skipped) {
        return {
            skipped: true,
            kind: status.kind || '',
            label: status.label || '行情预检未进入专家',
            reason: status.reason || record?.flow_summary || '',
        };
    }
    if (analysisIsFastPositionScan(record)) {
        return {
            skipped: true,
            kind: 'position_fast_scan',
            label: '持仓快速扫描未进入专家',
            reason: record?.flow_summary || '本轮是持仓快速扫描；只有出现强信号才进入专家深度复盘。',
        };
    }
    return { skipped: false, kind: '', label: '', reason: '' };
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
        [/\bexploration\b/g, '探索小仓'],
        [/\bsmall\b/g, '小仓'],
        [/\bmedium\b/g, '中等仓位'],
        [/\bnormal\b/g, '正常仓位'],
        [/\bblocked\b/g, '硬风控阻断'],
        [/ML\/time-series services are unavailable/gi, 'ML/时序服务不可用'],
        [/missing model data is treated as degraded evidence/gi, '缺失模型数据按降级证据处理'],
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

function analysisDisplayAction(action, record = null) {
    const value = String(action || '').toLowerCase() || 'hold';
    if (!record || analysisRecordType(record) !== 'market') return value;
    if (record.was_executed !== false || !['long', 'short'].includes(value)) return value;
    const hasPositionSize = record.position_size_pct !== null && record.position_size_pct !== undefined;
    const positionSize = Number(record.position_size_pct);
    const zeroPositionSize = hasPositionSize && Number.isFinite(positionSize) && positionSize <= 0;
    return zeroPositionSize || Boolean(record.execution_reason) ? 'hold' : value;
}

function analysisActionLabel(action, record = null) {
    const originalValue = String(action || '').toLowerCase();
    const value = analysisDisplayAction(action, record);
    if (!record || analysisRecordType(record) !== 'position') {
        const observed = String(record?.observed_action || originalValue).toLowerCase();
        if (value === 'hold' && ['long', 'short'].includes(observed)) {
            return observed === 'long' ? '观望（看多观察）' : '观望（看空观察）';
        }
        return actionLabel(value);
    }
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
        ['ML胜率（诊断）', `${opportunityScoreValue(Number(score.diagnostic_win_rate || 0) * 100, 1)}%`],
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
        returned_but_governance_blocked: '已返回（治理不可交易）',
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
    const preExpertSkip = analysisPreExpertSkip(record);
    if (preExpertSkip.skipped) {
        return `${preExpertSkip.label}；没有消耗大模型专家`;
    }
    const expectedCount = Number(record?.expected_expert_count ?? 0);
    const successfulCount = Number(record?.expert_count ?? 0);
    const returnedCount = Number(record?.returned_expert_count ?? successfulCount);
    if (record?.analysis_complete === false) {
        return `有效 ${successfulCount}/${expectedCount}，实际返回 ${returnedCount}；分析证据不完整`;
    }
    if (missingCount) return `${missingCount} 个未完成，点详情查看原因`;
    return `${expectedCount} 个专家均已有效返回`;
}

function pctLabel(value, digits = 0) {
    if (value === null || value === undefined || value === '') return '-';
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

function standardizedReturnDistribution(payload, side = '') {
    if (!payload || typeof payload !== 'object') return null;
    const container = payload.return_distribution_contract;
    if (!container || typeof container !== 'object') return null;
    const selectedSide = String(side || payload.best_side || payload.side || '').toLowerCase();
    const contract = container[selectedSide];
    return contract && typeof contract === 'object' ? contract : null;
}

function distributionPctLabel(value, digits = 4) {
    if (value === null || value === undefined || value === '') return '未评估';
    const num = Number(value);
    return Number.isFinite(num) ? signedPctValueLabel(num, digits) : '未评估';
}

function distributionProbabilityLabel(value, digits = 1) {
    if (value === null || value === undefined || value === '') return '未评估';
    const num = Number(value);
    return Number.isFinite(num) ? `${(num * 100).toFixed(digits)}%` : '未评估';
}

function distributionSummaryText(contract) {
    if (!contract) return '标准收益分布缺失';
    return [
        `原始期望 ${distributionPctLabel(contract.raw_expected_return_pct)}`,
        `目标期望 ${distributionPctLabel(contract.objective_expected_return_pct)}`,
        `收益下界 ${distributionPctLabel(contract.lower_quantile_return_pct)}`,
        `离散度 ${distributionPctLabel(contract.dispersion_pct)}`,
        `尾损概率 ${distributionProbabilityLabel(contract.tail_loss_probability)}`,
        `尾损尺度 ${distributionPctLabel(contract.tail_loss_scale_pct)}`,
    ].join(' · ');
}

function renderAnalysisMlSignal(signal) {
    if (!signal || signal.available === false) {
        return '<div class="analysis-empty">本轮没有可用的本地 ML 盈亏质量预测；AI 决策未受 ML 影响。</div>';
    }
    const predictions = Array.isArray(signal.predictions) ? signal.predictions : [];
    const rows = predictions.map(item => {
        const bestSide = item.best_side === 'long' ? '做多' : item.best_side === 'short' ? '做空' : '-';
        const distribution = standardizedReturnDistribution(item, item.best_side);
        const expected = Number(distribution?.objective_expected_return_pct);
        const edge = Number(item.profit_edge_pct || 0);
        const tone = Number.isFinite(expected) && expected > 0 && edge > 0 ? 'good' : 'warn';
        return `
            <div class="analysis-resolution-item">
                <strong>${Number(item.horizon_minutes || 0)}分钟</strong>
                <span>
                    ${bestSide} · ${distributionSummaryText(distribution)}
                    · 收益差 ${distributionPctLabel(item.profit_edge_pct)}
                    · ${analysisPill(`风险 ${pctLabel(item.risk_score)}`, tone)}
                </span>
            </div>`;
    }).join('');
    const influenceEnabled = signal.prediction_eligible === true && (signal.mode === 'entry_profit_filter' || signal.status === 'entry_profit_filter');
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
        market_prefilter: '行情预检',
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
    if (String(payload.governance_status || '').toLowerCase() === 'returned_but_governance_blocked') {
        return '已返回（治理不可交易）';
    }
    const status = String(payload.status || '').toLowerCase();
    const labels = {
        returned: '已返回',
        completed: '完成',
        ok: '正常',
        supported: '支持',
        active: '已参与',
        trained_torch_sequence_model: '已训练时序模型',
        trained_text_model: '已训练情绪模型',
        artifact_unavailable: '缺少模型产物',
        unavailable: '不可用',
        timeout: '已调用未返回',
        deferred: '本轮排队未执行',
        error: '错误',
        disabled: '已关闭',
        circuit_open: '熔断中',
        failed: '失败',
    };
    labels.specialist_shadow_inference = '\u5df2\u8fd4\u56de\uff08\u89c2\u5bdf\u6001\uff09';
    labels.shadow_observation = '\u5df2\u8fd4\u56de\uff08\u89c2\u5bdf\u6001\uff09';
    labels.trained_calibrator = '\u5df2\u8fd4\u56de\uff08\u6821\u51c6\u6001\uff09';
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
    if (String(payload.governance_status || '').toLowerCase() === 'returned_but_governance_blocked') {
        return analysisPill('已返回（治理不可交易）', 'muted');
    }
    if (!analysisToolAvailable(payload)) {
        return analysisPill(analysisToolPlainStatus(payload), 'warn');
    }
    const observationOnly = payload.production_permission === false
        || ['specialist_shadow_inference', 'shadow_observation', 'trained_calibrator']
            .includes(String(payload.status || '').toLowerCase());
    return analysisPill(
        analysisToolPlainStatus(payload),
        observationOnly ? 'muted' : payload.trained === false ? 'warn' : 'good',
    );
}

function analysisLocalToolsRunStatus(status) {
    const labels = {
        completed: '全部返回',
        partial: '部分返回',
        unavailable: '全部不可用',
        error: '调用失败',
        timeout: '调用超时',
        analysis_budget_deferred: '预算未执行',
        disabled: '未启用',
        circuit_open: '暂时熔断',
    };
    const normalized = String(status || 'completed').toLowerCase();
    return labels[normalized] || analysisLocalizeText(normalized);
}

function analysisLocalToolsErrorsText(errors) {
    if (!errors || typeof errors !== 'object') return '-';
    const labels = {
        profit_prediction: '盈利预测',
        time_series_prediction: '时序预测',
        sentiment_analysis: '情绪模型',
        exit_advice: '平仓建议',
    };
    return Object.entries(errors).map(([name, raw]) => {
        const text = String(raw || '').trim();
        const readable = /ReadTimeout|读取响应超时/i.test(text)
            ? '读取服务器响应超时，该轮未取得结果'
            : /batch budget exhausted before request|inference queue exhausted the batch budget/i.test(text)
                ? '服务器量化工具本轮预算已耗尽，后续子工具未执行'
                : /exceeded the remaining batch budget/i.test(text)
                    ? '服务器量化工具调用超过本轮剩余时间'
            : /ConnectTimeout|连接.*超时/i.test(text)
                ? '连接服务器超时，该轮未取得结果'
                : /could not reach the service|无法连接服务器量化工具/i.test(text)
                    ? '无法连接服务器量化工具'
                    : analysisLocalizeText(text);
        return `${labels[name] || analysisLocalizeText(name)}：${readable}`;
    }).join('；');
}

function analysisFundingTimeLabel(value) {
    if (value === null || value === undefined || value === '') return '-';
    const text = String(value).trim();
    if (/^\d{10,16}$/.test(text)) {
        let epoch = Number(text);
        if (text.length <= 10) epoch *= 1000;
        if (text.length >= 16) epoch /= 1000;
        if (Number.isFinite(epoch)) return toBeijingTime(new Date(epoch).toISOString());
    }
    return toBeijingTime(text);
}

function analysisFundingRateLabel(value) {
    const rate = Number(value);
    if (!Number.isFinite(rate)) return '-';
    const sign = rate > 0 ? '+' : '';
    return `${sign}${(rate * 100).toFixed(6)}%`;
}

function analysisFundingEvidenceReason(reason) {
    const labels = {
        current_direction_funding_cashflow_ready: '证据完整',
        funding_data_unavailable: '资金费数据不可用',
        funding_rate_missing: '资金费率缺失',
        funding_interval_missing: '结算周期缺失',
        prediction_horizon_missing: '预测持仓周期缺失',
        next_funding_time_missing: '下次结算时间缺失',
        funding_rate_observed_at_missing: '资金费率数据时间缺失',
        funding_rate_stale: '资金费率已过期',
        funding_side_invalid: '持仓方向无效',
    };
    const key = String(reason || '').trim();
    return labels[key] || (key ? analysisReasonLabel(key) : '未说明');
}

function analysisFundingSideRisk(projection) {
    if (!projection || projection.production_eligible !== true) return '证据不可用';
    const cashflow = Number(projection.signed_cashflow_pct);
    if (!Number.isFinite(cashflow) || cashflow === 0) return '预计无资金费现金流';
    return cashflow > 0 ? '预计资金费收入' : '预计资金费支出';
}

function renderMarketFundingAnalysis(directionCompetition) {
    const direction = directionCompetition && typeof directionCompetition === 'object'
        ? directionCompetition : {};
    const projection = direction.funding_projection && typeof direction.funding_projection === 'object'
        ? direction.funding_projection : {};
    const longFunding = projection.long && typeof projection.long === 'object'
        ? projection.long : {};
    const shortFunding = projection.short && typeof projection.short === 'object'
        ? projection.short : {};
    const longDirection = direction.long && typeof direction.long === 'object'
        ? direction.long : {};
    const shortDirection = direction.short && typeof direction.short === 'object'
        ? direction.short : {};
    const evidenceComplete = projection.evidence_complete === true;
    const evidenceReason = evidenceComplete
        ? '费率、结算时间与预测周期完整'
        : [longFunding.reason, shortFunding.reason]
            .map(analysisFundingEvidenceReason)
            .filter((value, index, list) => value && list.indexOf(value) === index)
            .join('、') || '资金费证据不可用';
    const sourceTime = longFunding.funding_rate_observed_at
        || shortFunding.funding_rate_observed_at;
    const nextFundingTime = longFunding.next_funding_time || shortFunding.next_funding_time;
    const intervalMinutes = Number(
        longFunding.funding_interval_minutes ?? shortFunding.funding_interval_minutes
    );
    const intervalText = Number.isFinite(intervalMinutes) && intervalMinutes > 0
        ? `${intervalMinutes.toFixed(0)} 分钟` : '-';
    const longEdge = Number(longDirection.score);
    const shortEdge = Number(shortDirection.score);
    const preferredSide = String(direction.preferred_side || 'neutral').toLowerCase();
    return `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">市场资金费与方向净优势</div>
                <div class="analysis-card-tags">
                    ${analysisPill(evidenceComplete ? '证据完整' : '证据不可用', evidenceComplete ? 'good' : 'warn')}
                    ${analysisPill(`方向 ${preferredSide === 'long' ? '做多' : preferredSide === 'short' ? '做空' : '中性'}`, preferredSide === 'neutral' ? 'muted' : 'good')}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-resolution-list">
                    <div class="analysis-resolution-item"><strong>当前费率</strong><span>${escHtml(analysisFundingRateLabel(longFunding.funding_rate ?? shortFunding.funding_rate))} · 周期 ${escHtml(intervalText)}</span></div>
                    <div class="analysis-resolution-item"><strong>费率数据时间</strong><span>${escHtml(analysisFundingTimeLabel(sourceTime))}</span></div>
                    <div class="analysis-resolution-item"><strong>下一次结算</strong><span>${escHtml(analysisFundingTimeLabel(nextFundingTime))}</span></div>
                    <div class="analysis-resolution-item"><strong>做多预计资金费</strong><span>${escHtml(signedPctValueLabel(longFunding.signed_cashflow_pct))} · ${escHtml(analysisFundingSideRisk(longFunding))} · 预计 ${Number(longFunding.estimated_settlement_count || 0)} 次结算</span></div>
                    <div class="analysis-resolution-item"><strong>做空预计资金费</strong><span>${escHtml(signedPctValueLabel(shortFunding.signed_cashflow_pct))} · ${escHtml(analysisFundingSideRisk(shortFunding))} · 预计 ${Number(shortFunding.estimated_settlement_count || 0)} 次结算</span></div>
                    <div class="analysis-resolution-item"><strong>资金费后净优势</strong><span>做多 ${escHtml(Number.isFinite(longEdge) ? signedPctValueLabel(longEdge) : '-')} · 做空 ${escHtml(Number.isFinite(shortEdge) ? signedPctValueLabel(shortEdge) : '-')}</span></div>
                    <div class="analysis-resolution-item"><strong>证据状态</strong><span>${escHtml(evidenceReason)}</span></div>
                </div>
            </div>
        </div>`;
}

function renderPositionFundingAnalysis(dynamicExitPolicy) {
    const policy = dynamicExitPolicy && typeof dynamicExitPolicy === 'object'
        ? dynamicExitPolicy : {};
    if (!Object.keys(policy).length) {
        return '<div class="analysis-empty">本轮没有返回持仓资金费与净收益合同。</div>';
    }
    const evidenceComplete = policy.funding_evidence_eligible === true;
    const fundingIncluded = policy.funding_fee_included === true;
    const evidenceLabel = evidenceComplete
        ? '已结算资金费已完成账单与生命周期核验'
        : analysisFundingEvidenceReason(policy.funding_evidence_status || policy.funding_cost_projection_reason);
    return `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">持仓资金费与净收益</div>
                <div class="analysis-card-tags">
                    ${analysisPill(evidenceComplete ? '已结算证据完整' : '已结算证据不完整', evidenceComplete ? 'good' : 'warn')}
                    ${policy.observation_only === true ? analysisPill('只读评估', 'muted') : ''}
                    ${analysisPill(policy.eligible === true ? '退出合同可用' : '退出合同未通过', policy.eligible === true ? 'good' : 'warn')}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-resolution-list">
                    <div class="analysis-resolution-item"><strong>已结算资金费</strong><span>${escHtml(signedMoneyWithUnit(policy.settled_funding_fee ?? policy.funding_fee_usdt))} · ${fundingIncluded ? '已计入净收益' : '未核验，未计入净收益'}</span></div>
                    <div class="analysis-resolution-item"><strong>预计未来资金费</strong><span>${escHtml(signedMoneyWithUnit(policy.expected_future_funding_cashflow))} · 下次结算 ${escHtml(analysisFundingTimeLabel(policy.next_funding_time))}</span></div>
                    <div class="analysis-resolution-item"><strong>当前生命周期净收益</strong><span>${escHtml(signedMoneyWithUnit(policy.current_lifecycle_net_pnl ?? policy.lifecycle_net_pnl_usdt))}</span></div>
                    <div class="analysis-resolution-item"><strong>继续持仓预计净收益</strong><span>${escHtml(signedMoneyWithUnit(policy.projected_hold_net_pnl))}</span></div>
                    <div class="analysis-resolution-item"><strong>资金费证据状态</strong><span>${escHtml(evidenceLabel)}</span></div>
                    <div class="analysis-resolution-item"><strong>动态退出</strong><span>收益保护压力 ${Number(policy.profit_lock_pressure || 0).toFixed(4)} · 建议平仓比例 ${pctLabel(policy.close_fraction, 1)} · ${escHtml(analysisFundingEvidenceReason(policy.reason))}</span></div>
                </div>
            </div>
        </div>`;
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
    const predictionRows = predictions.map(item => {
        const distribution = standardizedReturnDistribution(item, item.best_side || item.side);
        return `
            <div class="analysis-resolution-item">
                <strong>${Number(item.horizon_minutes || item.horizon || 0)}分钟</strong>
                <span>
                    ${distributionSummaryText(distribution)}
                    ${item.direction ? ` · 方向 ${escHtml(String(item.direction))}` : ''}
                </span>
            </div>`;
    }).join('') || (ts.available ? `
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
    const profitDistribution = standardizedReturnDistribution(
        profit,
        profit.best_side || profit.side,
    );
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
                    ${analysisPill(analysisLocalToolsRunStatus(tools.status), tools.status === 'completed' ? 'good' : 'warn')}
                    ${tools.duration_sec !== undefined ? analysisPill(`耗时 ${analysisDurationLabel(tools.duration_sec)}`, Number(tools.duration_sec || 0) > 2 ? 'warn' : 'muted') : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note"><span>盈利预测</span>
                    ${profitStatus}
                    ${analysisText([
                        analysisToolMetaText(profit),
                        `最佳方向 ${profit.best_side || '-'}`,
                        distributionSummaryText(profitDistribution),
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
                        `风险 ${sentiment.risk_level || '-'}`,
                        `模型 ${sentiment.model || sentiment.backend || '-'}`
                    ].join('；'))}
                </div>
                ${isPositionAnalysis ? `<div class="analysis-note analysis-note-muted"><span>平仓建议</span>
                    ${exitStatus}
                    ${analysisText(exitAdvice.action ? `${analysisToolMetaText(exitAdvice)}；${exitAdvice.action_label || analysisDecisionLabel(exitAdvice.action)}，信心 ${(Number(exitAdvice.confidence || 0) * 100).toFixed(0)}%${exitAdvice.reason ? `，原因：${analysisReasonLabel(exitAdvice.reason)}` : ''}` : '本轮没有返回独立平仓建议；如果不是持仓分析记录，通常不会触发这一项。')}
                </div>` : `<div class="analysis-note analysis-note-muted"><span>平仓建议</span>${exitStatus}${analysisText('市场分析只评估开仓机会、方向和风险；平仓建议只在持仓分析中显示。')}</div>`}
                ${tools.errors ? `<div class="analysis-note analysis-note-muted"><span>部分错误</span>${analysisText(analysisLocalToolsErrorsText(tools.errors))}</div>` : ''}
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
        <details class="analysis-news-group">
            <summary>直接相关新闻<span>${directCount} 条</span></summary>
            <div class="analysis-news-group-body">${directRows || '<div class="analysis-news-empty">本轮没有直接匹配该币种的新闻。</div>'}</div>
        </details>
        <details class="analysis-news-group">
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
    const skipKind = missing?.skip_kind || record?.expert_call_status?.kind || '';
    const ensembleTimedOut = missing?.status === 'ensemble_timeout'
        || skipKind === 'ensemble_timeout'
        || record?.expert_call_status?.timed_out === true;

    if (missing?.status === 'called_timeout') {
        return `${label} 已发起调用，但在本轮截止时间内未返回；这不是“未调用”，系统已保留本轮超时记录。`;
    }

    if (ensembleTimedOut) {
        const timeoutReason = rawReason
            || record?.expert_call_status?.reason
            || '专家协作已经发起，但超过单币种时间上限，结果未能完整返回并落库。';
        return `${label} 已进入本轮专家协作，但整体任务超时，本条记录没有保存到完整专家结果。原因：${timeoutReason}`;
    }

    if (missing?.status === 'pre_expert_skipped' || record?.expert_call_status?.skipped === true) {
        const skipLabel = record?.expert_call_status?.label || '行情预检未进入专家';
        return `${label} 本轮没有发起调用，不是模型故障：${skipLabel}。原因：${rawReason || record?.expert_call_status?.reason || '预检阶段已确定暂不需要大模型专家。'}`;
    }

    if (cfg && cfg.loading === true) {
        return `${label} 的系统配置还在加载中，暂时无法判断具体原因。`;
    }
    if (cfg && cfg.enabled === false) {
        return `${label} 在系统设置中已关闭，所以本轮没有发起调用。`;
    }
    const keylessLoopback = cfg?.configured === true
        && cfg?.configuration_type === 'keyless_loopback';
    if (cfg && cfg.configured === false && !cfg.api_key) {
        return `${label} 未配置 API Key，所以本轮没有发起调用。`;
    }
    if (cfg && Object.hasOwn(cfg, 'api_base') && !cfg.api_base) {
        return `${label} 未配置 API URL，所以本轮没有发起调用。`;
    }
    if (cfg && Object.hasOwn(cfg, 'model') && !cfg.model) {
        return `${label} 未配置模型名称，所以本轮没有发起调用。`;
    }
    if (keylessLoopback && !attempted && !timing) {
        return `${label} 已配置为本地免 Key 模型，但本轮没有留下调用结果；请结合专家协作状态判断是否整体超时。`;
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
        const preExpertSkip = analysisPreExpertSkip(r);
        const expertCount = Number(r.expert_count || (r.experts || []).length || 0); 
        const expectedCount = Number(r.expected_expert_count ?? 0);
        const attemptedCount = preExpertSkip.skipped ? 0 : Number(r.attempted_expert_count || expectedCount);  
        const missingCount = preExpertSkip.skipped ? 0 : Math.max(expectedCount - expertCount, 0);  
        const hasMajorConflict = Number(cross.major_conflicts || 0) > 0;  
        const completedCross = Number(cross.completed ?? cross.total ?? 0);
        const expectedCross = Number(cross.expected ?? r.cross_requested ?? cross.total ?? 0);
        const expertRequestedCross = Number(cross.expert_requested || 0);
        const automaticCross = Number(cross.automatic || 0);
        const unavailableCross = Number(cross.unavailable || 0);
        const crossText = preExpertSkip.skipped
            ? '预检阶段未发起交叉验证'
            : `计划 ${expectedCross}，完成 ${completedCross}（专家请求 ${expertRequestedCross} / 自动 ${automaticCross}），无法验证 ${unavailableCross}，分歧 ${Number(cross.divergent || 0)}`;
        const expertStatusLine = analysisExpertStatusLine(r, missingCount);
        const expertStatusColor = missingCount ? 'var(--yellow)' : 'var(--text-muted)';
        const expertSummary = preExpertSkip.skipped
            ? preExpertSkip.label
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
            <td><span class="badge badge-${analysisDisplayAction(r.final_action, r)}">${analysisActionLabel(r.final_action, r)}</span></td>
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

function renderAnalysisVectorMemory(memory) {
    if (!memory || memory.enabled === false) {
        return '<div class="analysis-empty">向量记忆未启用；启用后只检索三期新样本索引。</div>';
    }
    if (memory.status && memory.status !== 'ok') {
        return `<div class="analysis-empty">向量记忆状态：${escHtml(memory.status)}${memory.error ? `；${escHtml(memory.error)}` : ''}</div>`;
    }
    const hits = Array.isArray(memory.hits) ? memory.hits : [];
    if (!hits.length) {
        return '<div class="analysis-empty">没有检索到足够相似的三期新样本。</div>';
    }
    const influence = memory.influence || {};
    const influenceLevel = String(influence.level || 'neutral');
    const influenceTone = influenceLevel === 'positive' ? 'good'
        : influenceLevel === 'negative' ? 'warn'
            : 'muted';
    const delta = Number(influence.score_delta || 0);
    const deltaLabel = delta > 0 ? `+${delta.toFixed(2)} 分` : `${delta.toFixed(2)} 分`;
    const rows = hits.map(hit => {
        const outcomeTone = Number(hit.pnl_pct || 0) > 0 ? 'good' : Number(hit.pnl_pct || 0) < 0 ? 'warn' : 'muted';
        const kindLabel = hit.kind === 'news' ? '新闻/事件' : '三期决策样本';
        const pnl = hit.pnl_pct !== null && hit.pnl_pct !== undefined
            ? `收益 ${signedPctValueLabel(hit.pnl_pct)}`
            : '无收益结果';
        return `
            <div class="analysis-resolution-item">
                <strong>${escHtml(hit.symbol || kindLabel)}</strong>
                <span>
                    ${analysisPill(kindLabel, hit.kind === 'decision' ? 'good' : 'muted')}
                    ${hit.action ? analysisPill(analysisDecisionLabel(hit.action), analysisTone(hit.action)) : ''}
                    ${analysisPill(`相似度 ${(Number(hit.score || 0) * 100).toFixed(0)}%`, Number(hit.score || 0) >= 0.45 ? 'good' : 'muted')}
                    ${analysisPill(pnl, outcomeTone)}
                    ${hit.created_at ? ` · ${toBeijingTime(hit.created_at)}` : ''}
                    <br>${analysisText(hit.text || '-')}
                </span>
            </div>`;
    }).join('');
    return `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">三期相似样本记忆</div>
                <div class="analysis-card-tags">
                    ${analysisPill(memory.backend || 'vector', 'muted')}
                    ${analysisPill(`命中 ${hits.length} 条`, hits.length ? 'good' : 'muted')}
                    ${analysisPill(`影响 ${deltaLabel}`, influenceTone)}
                    ${analysisPill('非硬拦截', 'muted')}
                    ${memory.min_score !== undefined ? analysisPill(`阈值 ${(Number(memory.min_score || 0) * 100).toFixed(0)}%`, 'muted') : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note analysis-note-muted"><span>作用</span>${analysisText('只基于三期新索引样本提醒系统避免重复新阶段已验证的亏损模式，也帮助解释为什么降仓、观望或需要更强证据。')}</div>
                <div class="analysis-note analysis-note-${influenceTone === 'good' ? 'positive' : influenceTone === 'warn' ? 'warning' : 'muted'}">
                    <span>${escHtml(influence.label || '三期相似样本影响')}</span>
                    ${analysisText(influence.reason || '三期相似样本仅作为轻量加减分因子，不直接拦截开仓。')}
                    <br>
                    ${analysisText(`命中 ${Number(influence.matched_count || hits.length)} 条，三期盈利 ${Number(influence.profit_count || 0)} 条，三期亏损 ${Number(influence.loss_count || 0)} 条，同方向亏损 ${Number(influence.same_action_loss_count || 0)} 条。`)}
                </div>
                <div class="analysis-resolution-list">${rows}</div>
            </div>
        </div>`;
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
    try {
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
        renderAnalysisReasonModal(record);
    } catch (error) {
        console.error('Failed to render analysis reason detail', error);
        showAnalysisReasonLoadError(
            recordId || decisionId,
            `详情渲染失败：${error?.message || error || '未知错误'}。请刷新后重试；如果连续出现，请检查该条记录的详情数据结构。`
        );
    }
}

function renderAnalysisReasonModal(record) {
    setDecisionModalWide(true);
    const experts = Array.isArray(record.experts) ? record.experts : [];
    const crossSummary = record.cross_summary || {};
    const preExpertSkip = analysisPreExpertSkip(record);
    const expertCount = Number(record.expert_count || experts.length || 0);  
    const expectedCount = Number(record.expected_expert_count ?? 0);
    const attemptedCount = preExpertSkip.skipped ? 0 : Number(record.attempted_expert_count || expectedCount);  
    const completedCross = Number(crossSummary.completed ?? crossSummary.total ?? 0);
    const expectedCross = Number(
        crossSummary.expected ?? record.cross_requested ?? crossSummary.total ?? 0
    );
    const expertRequestedCross = Number(crossSummary.expert_requested || 0);
    const automaticCross = Number(crossSummary.automatic || 0);
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
    const expertSectionSubtitle = preExpertSkip.skipped
        ? preExpertSkip.label
        : `发起 ${attemptedCount} 个，返回 ${expertCount} 个`;
    const mlSignal = record.ml_signal || null;
    const mlSignalPrediction = mlPrimaryPrediction(mlSignal);
    const mlSignalDistribution = standardizedReturnDistribution(
        mlSignalPrediction,
        mlSignalPrediction?.best_side,
    );
    const localAiTools = record.local_ai_tools || null;
    const agentSkills = record.agent_skills || null;
    const newsContext = record.news_context || null;
    const vectorMemory = record.vector_memory || null;
    const attribution = record.decision_attribution || null;
    const isPositionFundingAnalysis = ['position', 'position_review'].includes(
        String(record.analysis_type || '').toLowerCase()
    );
    const fundingAnalysisHtml = isPositionFundingAnalysis
        ? renderPositionFundingAnalysis(record.dynamic_exit_policy)
        : renderMarketFundingAnalysis(record.direction_competition);
 
    const expertsHtml = preExpertSkip.skipped ? `
        <div class="analysis-card analysis-card-warning">
            <div class="analysis-card-head">
                <div class="analysis-card-title">${escHtml(preExpertSkip.label)}</div>
                ${analysisPill('预检跳过专家', 'muted')}
            </div>
            <div class="analysis-card-text">
                ${analysisText(preExpertSkip.reason || record.flow_summary || '预检阶段已确定本轮暂不需要进入大模型专家。')}
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
 
    const missingHtml = preExpertSkip.skipped ? '' : (record.missing_experts || []).map(e => {
        const reason = analysisMissingExpertReason(e, record);
        const ensembleTimedOut = e?.status === 'ensemble_timeout'
            || record?.expert_call_status?.kind === 'ensemble_timeout';
        const calledTimeout = e?.status === 'called_timeout';
        const notCalled = !ensembleTimedOut && !calledTimeout && (
            !Array.isArray(record.attempted_experts)
            || !record.attempted_experts.map(String).includes(String(e.expert_name || ''))
        );
        const pillText = calledTimeout ? '已调用未返回' : notCalled ? '未调用' : '未返回';
        const pillTone = calledTimeout ? 'warn' : 'bad';
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
                        ${analysisPill('仅观察', 'muted')}
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
                    ${analysisPill('无生产权限', 'muted')}
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
                    ${analysisPill(conflictResolution.consultation_used ? '已观察复核' : '只读记录', 'muted')}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note"><span>观察结果</span>${analysisText(conflictResolution.summary || '没有专家交叉观察记录。')}</div>
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
    const decisionMakerObservationOnly = decisionMaker?.status === 'observation_only';
    const decisionMakerStatusLabel = decisionMakerObservationOnly
        ? '仅观察'
        : decisionMaker?.status === 'completed'
            ? '已裁决'
            : decisionMaker?.status === 'skipped'
                ? '已跳过'
                : '未完成';
    const decisionMakerStatusTone = decisionMakerObservationOnly
        ? 'muted'
        : decisionMaker?.status === 'completed'
            ? 'good'
            : decisionMaker?.status === 'skipped'
                ? 'muted'
                : 'warn';
    const decisionMakerHtml = decisionMaker ? `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">模型裁决记录</div>
                <div class="analysis-card-tags">
                    ${analysisPill(decisionMakerStatusLabel, decisionMakerStatusTone)}
                    ${decisionMaker.action ? analysisPill(analysisActionLabel(decisionMaker.action, record), analysisTone(decisionMaker.action)) : ''}
                    ${decisionMaker.confidence !== undefined ? analysisPill(`信心 ${(Number(decisionMaker.confidence || 0) * 100).toFixed(0)}%`, Number(decisionMaker.confidence || 0) >= 0.6 ? 'good' : 'muted') : ''}
                    ${decisionMakerObservationOnly ? analysisPill('不参与生产裁决', 'muted') : decisionMaker.applied === true ? analysisPill('已采用', 'good') : decisionMaker.applied === false ? analysisPill('未采用', 'warn') : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note"><span>${decisionMakerObservationOnly ? '权限说明' : '裁决说明'}</span>${analysisText(decisionMaker.reasoning || decisionMaker.reason || decisionMaker.guard_reason || '-')}</div>
                ${decisionMaker.provider_model ? `<div class="analysis-note analysis-note-muted"><span>模型</span>${analysisText(decisionMaker.provider_model)}</div>` : ''}
            </div>
        </div>
    ` : '<div class="analysis-empty">本轮没有模型裁决记录</div>';

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
                    <div class="analysis-resolution-item"><strong>本地 ML</strong><span>${escHtml(attribution.local_ml?.available ? `${attribution.local_ml.side_label || '-'} / ${distributionSummaryText(attribution.local_ml.return_distribution_contract)} / 收益差 ${signedPctValueLabel(attribution.local_ml.profit_edge_pct)}` : '无可用预测')}</span></div>
                    <div class="analysis-resolution-item"><strong>服务器盈利模型</strong><span>${escHtml(attribution.server_profit?.available ? `${attribution.server_profit.side_label || '-'} / ${distributionSummaryText(attribution.server_profit.return_distribution_contract)}` : '无可用预测')}</span></div>
                    <div class="analysis-resolution-item"><strong>时序预测</strong><span>${escHtml(attribution.timeseries?.available ? `${attribution.timeseries.side_label || '-'} / ${distributionSummaryText(attribution.timeseries.return_distribution_contract)}` : '无可用预测')}</span></div>
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
                ${analysisMetric('专家返回', preExpertSkip.skipped ? preExpertSkip.label : `${expertCount}/${expectedCount}`, preExpertSkip.skipped || expertCount === expectedCount ? 'good' : 'warn')}
                ${mlSignal?.available ? analysisMetric('ML目标期望', distributionPctLabel(mlSignalDistribution?.objective_expected_return_pct), mlSignalDistribution && Number(mlSignalDistribution.objective_expected_return_pct) > 0 ? 'good' : 'warn') : ''}
                ${analysisMetric('交叉验证', `${completedCross}/${expectedCross}`, unavailableCross || completedCross < expectedCross ? 'warn' : 'good')}
                ${analysisMetric('分析耗时', analysisDurationLabel(totalDuration), totalDuration > 60 ? 'warn' : 'muted')}
                ${analysisMetric('最终方向', analysisActionLabel(record.final_action, record), analysisTone(record.final_action))}
                ${lifecycleLabel ? analysisMetric('持仓状态', lifecycleLabel, analysisPositionLifecycleTone(record)) : ''}
                ${analysisMetric('分析信心 / 仓位', `${finalConfidence} / ${positionSize}`, Number(record.final_confidence || 0) >= 0.6 ? 'good' : 'muted')}
            </div>

            ${attributionHtml ? analysisSection('决策归因面板', attributionHtml) : ''}
            ${analysisSection(isPositionFundingAnalysis ? '持仓资金费' : '市场资金费', fundingAnalysisHtml)}
            ${analysisSection('Agent/Skills 守门', renderAnalysisAgentSkills(agentSkills))}
            ${analysisSection('本地ML盈亏质量', renderAnalysisMlSignal(mlSignal))}
            ${analysisSection('服务器量化工具', renderAnalysisLocalAiTools(localAiTools, record.analysis_type))}
            ${analysisSection('新闻与事件', renderAnalysisNewsContext(newsContext))}
            ${analysisSection('三期相似样本记忆', renderAnalysisVectorMemory(vectorMemory))}
            ${analysisSection(preExpertSkip.skipped ? preExpertSkip.label : '专家初诊', `<div class="analysis-grid">${expertsHtml || '<div class="analysis-empty">无返回结果</div>'}</div>`, expertSectionSubtitle)}
            ${missingHtml ? analysisSection('未返回专家', `<div class="analysis-grid">${missingHtml}</div>`) : ''}
            ${analysisSection('交叉验证', `<div class="analysis-grid">${pairValidations || '<div class="analysis-empty">没有触发交叉验证</div>'}</div>`, `计划 ${expectedCross} 个，完成 ${completedCross} 个（专家请求 ${expertRequestedCross} 个，自动核验 ${automaticCross} 个），无法验证 ${unavailableCross} 个，重大矛盾 ${majorConflicts} 个`)}
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
    return fetchAnalysisRecords();
}

// ========== Expert Long-term Memory ==========

async function fetchExpertMemories() {
    const params = new URLSearchParams({
        page_size: EXPERT_MEMORY_PAGE_SIZE,
        memory_page: state.expertMemoryPage,
        reflection_page: state.tradeReflectionPage,
    });
    const data = await fetchLatestPageJSON(
        'expert-memories',
        `/api/expert-memories?${params.toString()}`,
    );
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
    const authorityEl = document.getElementById('trade-reflection-authority');
    const authority = data.authoritative_outcome_contract || {};
    const memoryTotal = Number(pagination.memory_total ?? state.expertMemoryTotal ?? memories.length);
    const reflectionTotal = Number(pagination.reflection_total ?? state.tradeReflectionTotal ?? reflections.length);
    const memoryPage = Number(pagination.memory_page || state.expertMemoryPage || 1);
    const reflectionPage = Number(pagination.reflection_page || state.tradeReflectionPage || 1);
    const memoryTotalPages = Number(pagination.memory_total_pages || Math.max(Math.ceil(memoryTotal / EXPERT_MEMORY_PAGE_SIZE), 1));
    const reflectionTotalPages = Number(pagination.reflection_total_pages || Math.max(Math.ceil(reflectionTotal / EXPERT_MEMORY_PAGE_SIZE), 1));
    if (countEl) countEl.textContent = `${memoryTotal} 条`;
    if (reflectionCountEl) reflectionCountEl.textContent = `${reflectionTotal} 条`;
    if (authorityEl) {
        const loaded = mlOptionalNumber(authority.loaded_count);
        const complete = mlOptionalNumber(authority.complete_count);
        authorityEl.className = `trade-reflection-authority ${authority.actual_outcome_overrides_shadow === true ? 'ready' : 'blocked'}`;
        authorityEl.innerHTML = authority.actual_outcome_overrides_shadow === true
            ? `<strong>交易所结算结果优先</strong><span>已读取 ${mlSampleCountLabel(loaded)} 条 OKX 结算记录，其中 ${mlSampleCountLabel(complete)} 条完整；未成交对照权重 ${mlEvidenceValue(authority.shadow_production_weight)}。</span><em>${escHtml(authority.version || '结算规则版本缺失')}</em>`
            : '<strong>交易所结算结果暂不可用</strong><span>当前不能确认未成交对照是否应被真实成交结果覆盖。</span>';
    }
    setExpertMemoryView(state.expertMemoryView || 'memories');

    if (memoryBody) {
        if (!memories.length) {
            memoryBody.innerHTML = '<tr><td colspan="7" class="expert-memory-empty">暂无专家记忆，平仓完成权威结算后会自动生成。</td></tr>';
        } else {
            memoryBody.innerHTML = memories.map(m => {
                const presentation = expertMemoryPresentation(m);
                return `
                    <tr>
                        <td><strong class="expert-memory-expert">${escHtml(m.expert_label || m.expert_name || '-')}</strong></td>
                        <td>${presentation.marketHtml}</td>
                        <td>${presentation.sourceHtml}</td>
                        <td>${presentation.outcomeHtml}</td>
                        <td>${presentation.lessonHtml}</td>
                        <td>${presentation.statsHtml}</td>
                        <td>${presentation.usageHtml}</td>
                    </tr>
                `;
            }).join('');
        }
        renderPagination('expert-memory-pagination', memoryPage, memoryTotalPages, memoryTotal, 'changeExpertMemoryPage');
    }

    if (reflectionBody) {
        if (!reflections.length) {
            reflectionBody.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);text-align:center;padding:24px;">暂无复盘记录。</td></tr>';
        } else {
            reflectionBody.innerHTML = reflections.map(r => {
                const authoritative = r.authoritative_outcome || null;
                const authorityStatus = r.authority_status || {};
                const authoritativePnl = authoritative
                    ? mlOptionalNumber(authoritative.realized_pnl) : null;
                const fallbackPnl = mlOptionalNumber(r.realized_pnl);
                const pnl = authoritative ? authoritativePnl : null;
                const pnlColor = pnl === null ? 'var(--text-muted)' : pnl >= 0 ? 'var(--green)' : 'var(--red)';
                const generatedTime = tradeReflectionTimeHtml(r.created_at);
                const generatedTimeTitle = toBeijingDateTime(r.created_at);
                const shadowRows = Array.isArray(authoritative?.counterfactual_evidence)
                    ? authoritative.counterfactual_evidence : [];
                const actualHtml = authoritative
                    ? `<div class="trade-outcome-cell ${authoritative.complete ? 'ready' : 'blocked'}">
                        <strong>${distributionPctLabel(authoritative.net_return_after_all_cost_pct)}</strong>
                        <span>${escHtml(authoritative.outcome_id ? `结算记录 ${compactIdentifier(authoritative.outcome_id, 22)}` : '结算记录编号缺失')}</span>
                        <em>${authoritative.complete ? '交易所结算已确认' : `证据未完整：${[...new Set(authoritative.evidence_gaps || [])].map(item => escHtml(dashboardReasonText(item))).join('；') || '完整持仓记录'}`}</em>
                    </div>`
                    : `<div class="trade-outcome-cell blocked">
                        <strong>${escHtml(authorityStatus.label || '等待交易所结算')}</strong>
                        <span>${escHtml(authorityStatus.reason || '还没有拿到完整的开仓、持仓、平仓和资金费记录')}</span>
                        <em>${fallbackPnl === null ? '暂时没有可确认的盈亏' : `本地暂存 ${signedMoney(fallbackPnl)} USDT，不能代替交易所结果`}</em>
                    </div>`;
                const shadowHtml = authoritative
                    ? `<div class="trade-shadow-cell">
                        <strong>${shadowRows.length} 条路径</strong>
                        <span>生产权重 ${mlEvidenceValue(authoritative.counterfactual_production_weight)}</span>
                        <em>仅作未成交情况下的对比，不代表真实盈亏</em>
                    </div>`
                    : '<div class="trade-shadow-cell"><strong>等待交易所结算</strong><span>暂时没有完整结果</span><em>未成交对照不能代替真实成交</em></div>';
                const reflectionCopy = reflectionTextPresentation(
                    r,
                    authoritative,
                    authorityStatus,
                    pnl,
                    fallbackPnl,
                );
                return `
                    <tr>
                        <td class="trade-reflection-time" title="${escHtml(generatedTimeTitle)}">${generatedTime}</td>
                        <td>${escHtml(r.symbol || '-')}</td>
                        <td>${sideLabel(r.side)}</td>
                        <td class="trade-reflection-pnl" style="color:${pnlColor};">${pnl === null ? escHtml(authorityStatus.label || '等待交易所结算') : `${signedMoney(pnl)} USDT`}</td>
                        <td>${mlOptionalNumber(r.hold_minutes) === null ? '证据缺失' : `${mlOptionalNumber(r.hold_minutes).toFixed(1)} 分钟`}</td>
                        <td>${actualHtml}</td>
                        <td>${shadowHtml}</td>
                        <td><div class="trade-reflection-text">${reflectionCopy.conclusionHtml}</div></td>
                        <td><div class="trade-reflection-text">${reflectionCopy.improvementHtml}</div></td>
                    </tr>
                `;
            }).join('');
        }
        renderPagination('trade-reflection-pagination', reflectionPage, reflectionTotalPages, reflectionTotal, 'changeTradeReflectionPage');
    }
}

function changeExpertMemoryPage(page) {
    state.expertMemoryPage = Math.max(1, Number(page) || 1);
    return fetchExpertMemories();
}

function changeTradeReflectionPage(page) {
    state.tradeReflectionPage = Math.max(1, Number(page) || 1);
    return fetchExpertMemories();
}

function setExpertMemoryView(view) {
    const selected = ['memories', 'reflections', 'training-effectiveness'].includes(view) ? view : 'memories';
    state.expertMemoryView = selected;
    document.querySelectorAll('#expert-memory-tabs .trade-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.expertMemoryView === selected);
    });
    document.getElementById('expert-memory-panel-memories')?.classList.toggle('active', selected === 'memories');
    document.getElementById('expert-memory-panel-reflections')?.classList.toggle('active', selected === 'reflections');
    document.getElementById('expert-memory-panel-training-effectiveness')?.classList.toggle('active', selected === 'training-effectiveness');
}

function trainingEffectivenessFilters() {
    const filters = {
        mode: document.getElementById('training-effectiveness-mode')?.value || 'all',
        side: document.getElementById('training-effectiveness-side')?.value || 'all',
        symbol: document.getElementById('training-effectiveness-symbol')?.value?.trim() || '',
    };
    state.trainingEffectivenessFilters = filters;
    return filters;
}

async function fetchTrainingEffectivenessReport() {
    const query = new URLSearchParams(trainingEffectivenessFilters());
    query.set('refresh', '1');
    const report = await fetchLatestPageJSON(
        'training-effectiveness',
        `/api/training-effectiveness/report?${query.toString()}`,
    );
    if (!report) return;
    state.trainingEffectivenessReport = report;
    renderTrainingEffectiveness(report);
    if (report.refresh_state === 'running') {
        window.clearTimeout(state.trainingEffectivenessRefreshTimer);
        state.trainingEffectivenessRefreshTimer = window.setTimeout(fetchTrainingEffectivenessReport, 2500);
    }
}

function trainingEffectivenessAvailable(report) {
    return report?.status === 'complete'
        && report?.freshness?.is_stale !== true
        && Number(report?.sample_quality?.valid_sample_count || 0) > 0;
}

function trainingEffectivenessNumber(value, digits = 4) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : '不可用';
}

function trainingEffectivenessPercent(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : '不可用';
}

function trainingEffectivenessPanel(title, content, className = '') {
    return `<section class="training-effectiveness-section ${className}"><h3>${escHtml(title)}</h3>${content}</section>`;
}

const TRAINING_EFFECTIVENESS_VERSION_LABELS = {
    active: '当前模型',
    challenger: '候选模型',
    baseline: '基准模型',
    observed: '已观测样本',
};

const TRAINING_EFFECTIVENESS_LIFECYCLE_LABELS = {
    active: '运行中',
    live: '实盘可用',
    trained: '已训练',
    canary: '灰度中',
    promotion_blocked: '晋级受阻',
    inferred: '由成交样本推断',
    inferred_from_authoritative_samples: '由权威成交样本推断',
    defined: '已定义',
    missing: '未登记',
    service_unavailable: '服务不可用',
};

function trainingEffectivenessVersionLabel(key) {
    return TRAINING_EFFECTIVENESS_VERSION_LABELS[key] || key;
}

function trainingEffectivenessLifecycleLabel(value) {
    const key = String(value || '').trim().toLowerCase();
    return TRAINING_EFFECTIVENESS_LIFECYCLE_LABELS[key] || value || '未说明';
}

function trainingEffectivenessSampleEvidence(report, key) {
    const version = report.versions?.[key] || {};
    const metric = report.metrics?.[key] || {};
    const parts = [];
    const trainingCount = Number(version.sample_count || 0);
    const effectCount = Number(metric.sample_count || 0);
    if (trainingCount > 0 && key !== 'baseline') {
        parts.push(`训练周期 ${trainingCount.toLocaleString()}`);
    }
    parts.push(`效果样本 ${effectCount.toLocaleString()}`);
    if (version.artifact_available === true) parts.push('产物已生成');
    return parts.join(' · ');
}

function renderTrainingEffectivenessFreshness(report) {
    const element = document.getElementById('training-effectiveness-freshness');
    if (!element) return;
    const available = trainingEffectivenessAvailable(report);
    const generating = report.refresh_state === 'running';
    element.className = `training-effectiveness-status ${available ? 'complete' : (report.status || 'missing')}`;
    element.innerHTML = `<strong>${available ? '报告完整' : (generating ? '正在生成报告' : '暂无完整报告')}</strong><span>生成 ${escHtml(report.generated_at || '未生成')} · 截止 ${escHtml(report.data_cutoff_at || '无数据')} · ${generating ? '后台读取权威成交与模型贡献，完成后自动刷新' : `指纹 ${escHtml(report.input_fingerprint || '缺失')}`}</span>`;
}

function renderTrainingEffectivenessVersions(report) {
    const element = document.getElementById('training-effectiveness-version-comparison');
    if (!element) return;
    const versions = report.versions || {};
    const cards = ['active', 'challenger', 'baseline'].map(key => {
        const row = versions[key] || {};
        const status = row.lifecycle || row.status || 'missing';
        return `<div class="training-effectiveness-version-card ${escHtml(String(status))}"><span>${trainingEffectivenessVersionLabel(key)}</span><strong title="${escHtml(row.model_id || row.version || '未登记')}">${escHtml(row.display_name || row.model_id || row.version || '未登记')}</strong><em>${escHtml(trainingEffectivenessLifecycleLabel(status))}</em><small>${escHtml(trainingEffectivenessSampleEvidence(report, key))}</small></div>`;
    }).join('');
    element.innerHTML = trainingEffectivenessPanel('版本对照', `<div class="training-effectiveness-summary">${cards}</div><p class="training-effectiveness-source-note">训练周期数据表示模型读过的训练窗口；效果样本只统计已结算且能归因到该模型的成交。</p>`);
}

function renderTrainingEffectivenessMetrics(report) {
    const element = document.getElementById('training-effectiveness-metrics');
    if (!element) return;
    const metrics = report.metrics || {};
    const rows = ['active', 'challenger', 'baseline', 'observed'].map(key => {
        const row = metrics[key] || {};
        const label = key === 'observed' ? '已观测权威样本' : trainingEffectivenessVersionLabel(key);
        const tone = Number(row.sample_count || 0) > 0 ? 'has-data' : 'no-data';
        return `<div class="training-effectiveness-metric-card ${tone}"><span>${escHtml(label)}</span><strong>${row.sample_count ? `${trainingEffectivenessNumber(row.fee_after_net_pnl)} USDT` : '暂无效果样本'}</strong><em><b>Profit Factor</b> ${trainingEffectivenessNumber(row.profit_factor)}<b>收益下界</b> ${trainingEffectivenessNumber(row.return_lower_bound)}<b>最大回撤</b> ${trainingEffectivenessNumber(row.max_drawdown)}<b>胜率</b> ${trainingEffectivenessPercent(row.win_rate)}<b>效果样本</b> ${Number(row.sample_count || 0).toLocaleString()}</em><small>${escHtml(trainingEffectivenessSampleEvidence(report, key))}</small></div>`;
    }).join('');
    element.innerHTML = trainingEffectivenessPanel('训练前后效果', `<div class="training-effectiveness-metric-grid">${rows}</div>`);
}

function renderTrainingEffectivenessChart(report) {
    const element = document.getElementById('training-effectiveness-chart');
    if (!element) return;
    const metrics = report.metrics || {};
    const rows = ['active', 'challenger', 'baseline', 'observed']
        .map(key => ({ key, value: Number(metrics[key]?.fee_after_net_pnl), sampleCount: Number(metrics[key]?.sample_count || 0) }))
        .filter(row => Number.isFinite(row.value) && row.sampleCount > 0);
    if (!rows.length) {
        element.innerHTML = '';
        return;
    }
    const max = Math.max(...rows.map(row => Math.abs(row.value)), 1);
    const labels = { active: '当前模型', challenger: '候选模型', baseline: '基准模型', observed: '已观测样本' };
    const bars = rows.map(row => {
        const width = Math.max(3, Math.round(Math.abs(row.value) / max * 100));
        const sign = row.value >= 0 ? 'positive' : 'negative';
        return `<div class="training-effectiveness-bar-row"><span>${labels[row.key]}</span><div><i class="${sign}" style="width:${width}%"></i></div><strong>${trainingEffectivenessNumber(row.value)} USDT</strong></div>`;
    }).join('');
    element.innerHTML = trainingEffectivenessPanel('费后净收益对照', `<div class="training-effectiveness-bars">${bars}</div>`);
}

function renderTrainingEffectivenessCostAttribution(report) {
    const element = document.getElementById('training-effectiveness-cost-attribution');
    if (!element) return;
    const costs = report.cost_attribution || {};
    const items = [
        ['毛盈亏', costs.gross_pnl, 'gross'],
        ['手续费', costs.fee, 'fee'],
        ['滑点', costs.slippage, 'slippage'],
        ['资金费', costs.funding_fee, 'funding'],
    ];
    const parts = items.map(([label, value, tone]) => `<div class="training-effectiveness-cost-item ${tone}"><span>${label}</span><strong>${trainingEffectivenessNumber(value)}</strong><em>USDT</em></div>`).join('');
    element.innerHTML = trainingEffectivenessPanel('收益组成', `<div class="training-effectiveness-cost-breakdown">${parts}</div><div class="training-effectiveness-cost-total"><span>费后净收益</span><strong>${trainingEffectivenessNumber(costs.fee_after_net_pnl)} USDT</strong></div><p>计算口径：毛盈亏 − 手续费 − 滑点 + 资金费。资金费只用于完整核算；资金费贡献不能直接等同于模型预测能力。</p>`);
}

function renderTrainingEffectivenessExperts(report) {
    const element = document.getElementById('training-effectiveness-expert-contributions');
    if (!element) return;
    const rows = Array.isArray(report.expert_contributions) ? report.expert_contributions : [];
    const content = rows.length ? rows.map(row => `<div><strong>${escHtml(row.expert_label || row.expert_name || '专家')}</strong><span>费后影响 <b>${trainingEffectivenessNumber(row.net_pnl_delta)}</b> · 回撤影响 <b>${trainingEffectivenessNumber(row.drawdown_delta)}</b> · 错误开仓 <b>${trainingEffectivenessNumber(row.false_entry_delta)}</b> · 多空平衡 <b>${trainingEffectivenessNumber(row.side_balance_delta)}</b> · 样本 <b>${Number(row.sample_count || 0)}</b></span></div>`).join('') : '<div class="training-effectiveness-empty">暂时没有专家消融样本</div>';
    element.innerHTML = trainingEffectivenessPanel('专家贡献', `<div class="training-effectiveness-list">${content}</div>`);
}

function renderTrainingEffectivenessFunnel(report) {
    const element = document.getElementById('training-effectiveness-execution-funnel');
    if (!element) return;
    const funnel = report.execution_funnel || {};
    const labels = [['signals', '产生信号'], ['evidence_passed', '证据门禁'], ['risk_passed', '风险检查'], ['orders_submitted', '提交订单'], ['filled', '成交'], ['positions_opened', '建立持仓'], ['closed', '平仓'], ['settled', '结算']];
    const rows = labels.map(([key, label]) => `<div><span>${escHtml(label)}</span><strong>${Number(funnel[key] || 0).toLocaleString()}</strong><em>损失率 ${trainingEffectivenessPercent(funnel[`${key}_loss_rate`])}</em></div>`).join('');
    const scope = funnel.scope ? `<p class="training-effectiveness-source-note">${escHtml(funnel.scope)}</p>` : '';
    element.innerHTML = trainingEffectivenessPanel('交易链路漏斗', `<div class="training-effectiveness-funnel">${rows}</div>${scope}`);
}

function renderTrainingEffectivenessSamples(report) {
    const element = document.getElementById('training-effectiveness-sample-quality');
    if (!element) return;
    const quality = report.sample_quality || {};
    const counts = quality.authority_counts || {};
    element.innerHTML = trainingEffectivenessPanel('样本可信度', `<div class="training-effectiveness-metric-grid"><div><span>OKX 实际成交/结算</span><strong>${Number(counts.okx_realized || 0)}</strong></div><div><span>影子市场机会（非真实盈利）</span><strong>${Number(counts.shadow_opportunity || 0)}</strong></div><div><span>反事实成本</span><strong>${Number(counts.counterfactual_cost || 0)}</strong></div><div><span>排除异常</span><strong>${Number(counts.excluded || 0)}</strong></div></div>`);
}

function renderTrainingEffectivenessConclusion(report) {
    const element = document.getElementById('training-effectiveness-conclusion');
    if (!element) return;
    const available = trainingEffectivenessAvailable(report);
    const conclusion = report.conclusion || {};
    const blockers = Array.isArray(conclusion.blocking_reasons) ? conclusion.blocking_reasons : [];
    const blockerLabels = {
        no_okx_realized_samples: '暂无 OKX 权威成交/结算样本',
        active_version_missing: '当前 active 模型版本缺失',
        active_version_inferred: 'active 模型来自权威成交样本推断，尚未在 registry 正式登记',
        challenger_version_missing: '当前 challenger 模型版本缺失',
        insufficient_effectiveness_samples: '有效训练效果样本不足',
        report_version_mismatch: '报告版本不匹配',
        invalid: '报告结构校验失败',
    };
    const visibleBlockers = blockers.map(reason => blockerLabels[reason] || reason);
    const message = visibleBlockers.length
        ? visibleBlockers.map(reason => escHtml(reason)).join('；')
        : (available ? '没有记录阻断原因' : '报告缺失、过期、不完整或有效样本为 0');
    element.innerHTML = trainingEffectivenessPanel('审计结论', `<div class="training-effectiveness-conclusion ${available && conclusion.promotion_eligible === true ? 'complete' : 'blocked'}"><strong>${available ? (conclusion.promotion_eligible === true ? '满足晋级证据条件' : '不满足晋级条件') : '结论不可用'}</strong><span>${message}</span></div>`);
}

function renderTrainingEffectiveness(report = {}) {
    renderTrainingEffectivenessFreshness(report);
    renderTrainingEffectivenessVersions(report);
    renderTrainingEffectivenessChart(report);
    renderTrainingEffectivenessMetrics(report);
    renderTrainingEffectivenessCostAttribution(report);
    renderTrainingEffectivenessExperts(report);
    renderTrainingEffectivenessFunnel(report);
    renderTrainingEffectivenessSamples(report);
    renderTrainingEffectivenessConclusion(report);
}

// ========== Shadow Backtest ==========

async function fetchShadowBacktests() {
    const params = new URLSearchParams({
        page_size: EXPERT_MEMORY_PAGE_SIZE,
        page: state.shadowBacktestPage,
    });
    const status = state.shadowBacktestStatus || document.getElementById('shadow-backtest-status')?.value || '';
    if (status) params.set('status', status);
    const data = await fetchLatestPageJSON(
        'shadow-backtests',
        `/api/shadow-backtests?${params.toString()}`,
    );
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
    return fetchShadowBacktests();
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
        authoritative_trade_outcome: '权威成交结果',
        lesson: '经验',
    };
    return map[type] || '其他记忆';
}

function reflectionTextPresentation(row = {}, authoritative = null, authorityStatus = {}, pnl = null, fallbackPnl = null) {
    const symbol = String(row.symbol || '该标的');
    const side = sideLabel(row.side);
    const authoritativeComplete = authoritative?.complete === true;
    const authoritativePresent = Boolean(authoritative);
    const authorityGapText = Array.isArray(authoritative?.evidence_gaps)
        ? authoritative.evidence_gaps.map(item => dashboardReasonText(item)).filter(Boolean).join('；')
        : '';
    const returnPct = authoritativeComplete
        ? mlOptionalNumber(authoritative.net_return_after_all_cost_pct)
        : null;
    const authoritativePnl = authoritativeComplete
        ? mlOptionalNumber(authoritative.realized_pnl)
        : null;
    const confirmedPnl = authoritativePnl ?? pnl;
    const confirmedBasis = confirmedPnl ?? returnPct;
    const confirmedResult = confirmedBasis === null
        ? '结果待确认'
        : confirmedBasis > 0 ? '盈利' : confirmedBasis < 0 ? '亏损' : '持平';
    const confirmedResultTone = confirmedBasis === null
        ? 'neutral'
        : confirmedBasis > 0 ? 'profit' : confirmedBasis < 0 ? 'loss' : 'neutral';
    const returnText = returnPct === null
        ? ''
        : `，费后收益率 ${returnPct >= 0 ? '+' : ''}${returnPct.toFixed(4)}%`;

    if (authoritativeComplete) {
        return {
            conclusionHtml: `<div class="trade-reflection-copy ${confirmedResultTone}">
                <strong>交易所已确认：${escHtml(confirmedResult)}</strong>
                <span>${escHtml(symbol)} ${escHtml(side)} 扣除手续费、计入资金费后，结果为${escHtml(confirmedResult)}${confirmedPnl === null ? '' : `，金额 ${signedMoney(confirmedPnl)} USDT`}${returnText}。</span>
            </div>`,
            improvementHtml: `<div class="trade-reflection-copy">
                <strong>系统下一步</strong>
                <span>把这笔结果加入训练对比，重新检查${escHtml(side)}信号是否稳定；不会只凭这一笔交易决定以后只做多或只做空。</span>
            </div>`,
        };
    }

    if (authoritativePresent) {
        return {
            conclusionHtml: `<div class="trade-reflection-copy neutral">
                <strong>交易所结算还没完成</strong>
                <span>${escHtml(symbol)} ${escHtml(side)} 当前不能确认真实盈亏${authorityGapText ? `；${escHtml(authorityGapText)}` : ''}。</span>
            </div>`,
            improvementHtml: `<div class="trade-reflection-copy">
                <strong>系统下一步</strong>
                <span>等 OKX 返回完整的开仓、持仓、平仓和资金费记录，再确认真实结果；确认前不用于训练或升级模型。</span>
            </div>`,
        };
    }

    const localResult = fallbackPnl === null
        ? '暂时没有可用盈亏'
        : fallbackPnl > 0 ? '暂时盈利' : fallbackPnl < 0 ? '暂时亏损' : '暂时持平';
    const localPnlText = fallbackPnl === null ? '' : `，暂存盈亏 ${signedMoney(fallbackPnl)} USDT`;
    return {
        conclusionHtml: `<div class="trade-reflection-copy neutral">
            <strong>等待交易所结算</strong>
            <span>${escHtml(symbol)} ${escHtml(side)}${localResult ? `：${escHtml(localResult)}` : ''}${localPnlText}，当前只是本地暂存结果${fallbackPnl === null ? '' : `；复盘暂存 ${signedMoney(fallbackPnl)} USDT，不作为权威训练事实`}。</span>
        </div>`,
        improvementHtml: `<div class="trade-reflection-copy">
            <strong>系统下一步</strong>
            <span>${escHtml(authorityStatus.label || '等待 OKX 交易所结算')}；确认前不把这条记录当作训练事实或升级模型的依据。</span>
        </div>`,
    };
}

function memoryActionLabel(action) {
    const map = {
        reduce_risk: '降信心/降仓位',
        keep_with_filters: '保留但需过滤',
        wait_for_better_setup: '等待更好机会',
        observation_only_revalidate_distribution: '观察并复核分布',
    };
    return map[action] || '仅用于观察';
}

function expertMemoryPresentation(memory = {}) {
    const patternParts = String(memory.market_pattern || '').split('|').map(item => item.trim());
    const symbol = String(memory.symbol || patternParts[0] || '通用');
    const side = String(memory.side || patternParts[1] || '').toLowerCase();
    const sideText = side ? sideLabel(side) : '方向不限';
    const sideClass = ['long', 'short'].includes(side) ? `badge-${side}` : 'badge-hold';
    const typeText = memoryTypeLabel(memory.memory_type);
    const isAuthoritative = memory.memory_type === 'authoritative_trade_outcome'
        || memory.memory_source === 'authoritative_trade_outcome';
    const outcome = expertMemoryOutcome(memory);
    const actionText = memoryActionLabel(memory.recommended_action);
    const evidenceCount = Math.max(0, Number(memory.evidence_count || 0));
    const hitCount = Math.max(0, Number(memory.hit_count || 0));
    const successCount = Math.max(0, Number(memory.success_count || 0));
    const failureCount = Math.max(0, Number(memory.failure_count || 0));
    const lessonText = isAuthoritative
        ? authoritativeMemoryLesson(sideText, outcome)
        : readableMemoryLesson(memory.lesson, typeText);
    const sourceDetail = isAuthoritative ? 'OKX 完整持仓生命周期' : memorySourceDetail(memory.memory_type);
    const productionEligible = memory.production_evidence_eligible === true;

    const marketHtml = `<div class="expert-memory-market">
        <strong>${escHtml(symbol)}</strong>
        <span class="badge ${sideClass}">${escHtml(sideText)}</span>
    </div>`;
    const sourceHtml = `<div class="expert-memory-source">
        <strong>${escHtml(typeText)}</strong>
        <span>${escHtml(sourceDetail)}</span>
    </div>`;
    const outcomeHtml = outcome.available
        ? `<div class="expert-memory-outcome ${outcome.tone}">
            <strong>${escHtml(outcome.resultText)} · ${escHtml(outcome.returnText)}</strong>
            <span>净盈亏 ${escHtml(outcome.pnlText)}</span>
            ${outcome.id ? `<em title="${escHtml(outcome.id)}">结算 ID ${escHtml(compactIdentifier(outcome.id, 22))}</em>` : ''}
        </div>`
        : `<div class="expert-memory-outcome neutral">
            <strong>${escHtml(memoryResultSummary(successCount, failureCount))}</strong>
            <span>该类记忆未提供单笔结算金额</span>
        </div>`;
    const lessonHtml = `<div class="expert-memory-lesson">
        <strong>${escHtml(lessonText)}</strong>
        <span>${escHtml(actionText)}</span>
    </div>`;
    const statsHtml = `<div class="expert-memory-stats">
        <span><b>${evidenceCount}</b> 条证据</span>
        <span><b>${hitCount}</b> 次命中</span>
    </div>`;
    const usageHtml = `<div class="expert-memory-usage">
        <strong>仅观察</strong>
        <span>${productionEligible ? '可进入训练评估，不直接触发交易' : '不参与生产决策'}</span>
    </div>`;
    return { marketHtml, sourceHtml, outcomeHtml, lessonHtml, statsHtml, usageHtml };
}

function expertMemoryOutcome(memory = {}) {
    let pnl = mlOptionalNumber(memory.realized_net_pnl_usdt);
    let returnPct = mlOptionalNumber(memory.net_return_after_all_cost_pct);
    let outcomeId = String(memory.outcome_id || '').trim();
    if ((pnl === null || returnPct === null || !outcomeId) && memory.lesson) {
        const match = String(memory.lesson).match(/^Authoritative fee-after outcome\s+(.+?):\s*symbol=[^,]+,\s*side=[^,]+,\s*result=([^,]+),\s*net_return_pct=([-+0-9.eE]+),\s*realized_net_pnl_usdt=([-+0-9.eE]+)\.?$/i);
        if (match) {
            outcomeId = outcomeId || match[1].trim();
            returnPct = returnPct ?? mlOptionalNumber(match[3]);
            pnl = pnl ?? mlOptionalNumber(match[4]);
        }
    }
    if (pnl === null && returnPct === null) {
        return { available: false, id: outcomeId };
    }
    const basis = pnl ?? returnPct ?? 0;
    const resultText = basis > 0 ? '盈利' : basis < 0 ? '亏损' : '持平';
    const tone = basis > 0 ? 'profit' : basis < 0 ? 'loss' : 'neutral';
    return {
        available: true,
        id: outcomeId,
        resultText,
        tone,
        returnText: returnPct === null ? '费后收益率待补充' : `${returnPct >= 0 ? '+' : ''}${returnPct.toFixed(4)}%`,
        pnlText: pnl === null ? '待补充' : `${signedMoney(pnl)} USDT`,
    };
}

function authoritativeMemoryLesson(sideText, outcome) {
    if (!outcome.available) {
        return `该笔${sideText}已完成权威结算，当前记录用于复核同类场景。`;
    }
    return `该笔${sideText}在计入资金费并扣除交易成本后${outcome.resultText}，用于校准同类场景的收益与风险分布。`;
}

function readableMemoryLesson(rawLesson, typeText) {
    const text = String(rawLesson || '').trim();
    if (!text) return `${typeText}尚未补充可读结论。`;
    if (/^[\x00-\x7F]+$/.test(text)) return `${typeText}已记录，原始技术说明已收起。`;
    return text;
}

function memorySourceDetail(type) {
    if (String(type || '').startsWith('shadow_')) return '未成交机会的反事实复盘';
    return '历史交易与复盘记录';
}

function memoryResultSummary(successCount, failureCount) {
    if (successCount || failureCount) return `盈利 ${successCount} · 亏损 ${failureCount}`;
    return '暂无结果统计';
}

// ========== Local ML Signal Dashboard ==========

async function fetchMLSignalDashboard() {
    const [status, localToolsStatus, registryData, recordsData, contributionData] = await Promise.all([
        fetchJSON('/api/ml-signal/status').catch(err => ({
            available: false,
            status: 'request_error',
            error: err?.message || '本地 ML 状态接口请求失败',
            message: '本地 ML 状态接口请求失败，页面已保留其它诊断数据。',
        })),
        fetchJSON('/api/local-ai-tools/status').catch(err => ({
            available: false,
            service_available: false,
            status: 'request_error',
            error: err?.message || '本地量化工具状态接口请求失败',
            message: '本地量化工具状态接口请求失败，请检查 18001 或后端日志。',
        })),
        fetchJSON('/api/model-training/registry').catch(err => ({
            models: [],
            summary: {},
            error: err?.message || '模型训练注册表请求失败',
        })),
        fetchJSON(`/api/analysis-records?limit=20&include_ml_summary=true&is_paper=${state.mode === 'paper' ? 'true' : 'false'}`).catch(() => ({ records: [] })),
        fetchJSON(`/api/model-contribution/stats?mode=${state.mode === 'live' ? 'live' : 'paper'}&days=7`).catch(() => null),
    ]);
    state.mlSignalStatus = status || null;
    state.localAIToolsStatus = localToolsStatus || null;
    state.modelTrainingRegistry = registryData || null;
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
        loss_filter: 'profit_prediction',
        timeseries: 'time_series_prediction',
        deep_timeseries: 'time_series_prediction',
        sentiment: 'sentiment_analysis',
        deep_sentiment: 'sentiment_analysis',
        exit: ['exit_advice', 'exit', 'position_exit'],
    };
    const modelAliases = {
        profit: ['profit', 'profit_model', 'entry_profit', 'profit_prediction'],
        loss_filter: ['loss_filter', 'loss_model', 'loss_probability', 'risk_filter'],
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

function mlOptionalNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function mlFirstNumber(source, keys) {
    for (const key of keys) {
        if (!Object.prototype.hasOwnProperty.call(source || {}, key)) continue;
        const number = mlOptionalNumber(source[key]);
        if (number !== null) return number;
    }
    return null;
}

const DASHBOARD_REASON_TEXT = Object.freeze({
    execution_position_exceeds_model_request: '成交后实际仓位超出模型请求上限，已标记为风险合同违规，不能作为合格执行证据',
    execution_price_far_from_pre_submit_quote: '成交价与提交前权威行情相差过大，成交已隔离，禁止作为正常执行证据',
    execution_instrument_mismatch: '成交返回的 OKX 合约身份与请求合约不一致，成交已隔离',
    execution_underlying_mismatch: '成交返回的标的身份与 OKX 合约不一致，成交已隔离',
    exchange_position_underlying_mismatch: 'OKX 当前仓位的 instId 与 underlying 标的不一致，已阻断正常持仓接管',
    okx_private_position_underlying_differs_from_public_instrument: 'OKX 私有仓位标的与公共合约规格不一致，已阻断正常持仓接管',
    current_position_notional_missing: 'OKX 当前仓位缺少可验证的权威名义价值，已阻断正常持仓接管',
    exchange_position_notional_mismatch: 'OKX 当前仓位名义价值与合约张数、合约面值和标记价不一致，已阻断正常持仓接管',
    close_fill_contracts_history_mismatch: 'OKX 平仓成交数量与仓位历史记录不一致，无法确认完整平仓数量',
    close_fill_contract_history_mismatch: 'OKX 平仓成交数量与仓位历史记录不一致，无法确认完整平仓数量',
    entry_fill_contracts_history_mismatch: 'OKX 开仓成交数量与仓位历史记录不一致，无法确认完整开仓数量',
    order_fee_total_mismatch: 'OKX 成交单手续费合计与仓位历史手续费不一致，无法确认最终手续费',
    entry_fill_price_history_mismatch: 'OKX 开仓成交均价与仓位历史均价不一致',
    close_fill_price_history_mismatch: 'OKX 平仓成交均价与仓位历史均价不一致',
    entry_fill_contract_quantity_mismatch: 'OKX 开仓成交数量与合约面值换算结果不一致',
    close_fill_contract_quantity_mismatch: 'OKX 平仓成交数量与合约面值换算结果不一致',
    missing_authoritative_entry_fill_facts: '缺少 OKX 完整开仓成交明细',
    missing_authoritative_close_fill_facts: '缺少 OKX 完整平仓成交明细',
    settlement_algebra_mismatch: 'OKX 盈亏、手续费和资金费无法按同一公式对账',
    no_model: '本地 ML 尚未注册当前模型 Artifact',
    artifact_incompatible: '当前模型 Artifact 与运行时收益监督合同不兼容，已禁止加载',
    artifact_load_failed: '当前模型 Artifact 加载失败，已禁止用于运行时预测',
    disabled: '已禁用',
    degraded: '证据未达标',
    learning_only: '仅学习观察',
    shadow_ready: '影子观察就绪',
    paper_canary_ready: 'Paper Canary 生命周期；模拟盘正常交易，实盘仍未授权',
    partial_ready: '部分方向就绪',
    artifact_activation_not_production_authorized: '当前 Artifact 可参与模拟盘，尚未获得实盘权限',
    shadow_market_opportunity_distribution_missing: '缺少影子市场机会收益分布',
    counterfactual_execution_cost_distribution_missing: '缺少反事实执行成本分布',
    authoritative_realized_return_distribution_missing: '缺少权威真实成交收益分布',
    authoritative_return_distribution_missing: '缺少权威真实成交收益分布',
    authoritative_execution_cost_distribution_missing: '缺少权威执行成本分布',
    market_opportunity_distribution_unavailable: '缺少可按决策组切分的固定窗口市场机会标签',
    authoritative_execution_cost_distribution_unavailable: '缺少带入场特征的 OKX 权威手续费、滑点和资金费样本',
    chronological_market_training_identity_incomplete: '市场机会样本缺少完整时间身份',
    chronological_cost_training_identity_incomplete: '权威执行成本样本缺少完整生命周期时间身份',
    authoritative_slippage_distribution_missing: '缺少权威真实滑点分布',
    average_fee_after_return_not_positive: '平均费后收益不为正',
    empirical_return_lower_hinge_not_positive: '费后收益经验下界不为正',
    profit_factor_undefined: '缺少亏损分母，盈亏比暂时无法计算',
    profit_factor_not_above_break_even: '盈亏比没有高于自然盈亏平衡线 1',
    profit_factor_below_unity: '盈亏比低于自然盈亏平衡线 1',
    realized_net_pnl_non_positive: '权威已实现净收益不为正',
    no_trainable_samples: '没有符合干净训练契约的可训练样本',
    effective_training_weight_zero: '可训练样本的有效权重为 0',
    return_objective_report_missing: '缺少费后收益目标报告',
    paper_observation_not_healthy: '模拟盘观察尚未达到健康状态',
    walk_forward_required: '缺少按时间滚动验证',
    model_stage_not_live: '模型仍处于影子或候选阶段',
    model_stage_not_canary_eligible: '模型证据尚不满足灰度阶段要求',
    ml_readiness_blocks_live_route: '本地 ML 收益证据未达标，禁止切换生产路由',
    competition_not_live_applicable: '模型竞赛结果仍为观察数据，不能应用到生产',
    competition_baseline_missing: '缺少可比较的生产基线样本',
    baseline_missing: '缺少可比较的基线样本',
    feature_coverage_missing: '特征覆盖不完整',
    authoritative_fee_after_return_lcb_not_positive: '权威费后收益置信下界不为正',
    authoritative_profit_factor_undefined: '权威成交缺少亏损分母，盈亏比无法计算',
    authoritative_profit_factor_below_unity: '权威成交盈亏比低于自然盈亏平衡线 1',
    child_endpoint_contract_missing_or_not_ready: '量化子接口尚未返回就绪契约',
    legacy_data_paths_preserved: '旧数据路径按策略只读隔离保留，不影响当前运行',
    historical_entry_contract_incomplete_preserved: '旧版入场合同缺口已保留',
    deterministic_position_order_match: '历史仓位与订单可确定匹配，等待受控补链',
    missing_matching_entry_order: '历史仓位缺少可确认的开仓订单链接',
    missing_matching_close_order: '历史仓位缺少可确认的平仓订单链接',
    positions_history_no_matching_row: 'OKX 官方仓位历史暂未匹配到该平仓生命周期',
    official_position_history_identity_unresolved: 'OKX 平仓成交已确认，但官方仓位历史身份尚未解析',
    okx_executor_unavailable: 'OKX 执行器尚未初始化，无法读取保护证据',
    okx_protection_inventory_unavailable: 'OKX 保护单快照读取失败',
    okx_protection_evidence_unavailable: 'OKX 保护证据读取失败',
    position_risk_evidence_unavailable: '仓位风险证据读取失败',
    missing_okx_position_protection: '当前 OKX 仓位缺少止盈止损保护单',
    okx_protection_quantity_coverage_mismatch: 'OKX 保护单数量未完整覆盖当前仓位',
    invalid_okx_protection_order: 'OKX 保护单合同无效',
    profit_risk_sizing_missing: '历史决策未保存独立风险定仓合同',
    profit_risk_sizing_not_production_eligible: '历史入场风险合同当时未达到生产执行条件',
    risk_contract_version_missing: '历史入场记录未保存风险合同版本',
    independent_risk_budget_missing: '历史入场记录未保存独立风险预算',
    planned_stressed_loss_missing: '历史入场记录未保存计划压力损失',
    target_notional_missing: '历史入场记录未保存目标仓位名义价值',
    final_notional_missing: '历史入场记录未保存最终仓位名义价值',
    portfolio_stressed_loss_missing: '历史入场记录未保存组合压力损失',
    portfolio_gross_notional_missing: '历史入场记录未保存组合总名义价值',
    risk_contract_fingerprint_missing: '历史入场记录未保存风险合同指纹',
    local_position_lineage_missing: '本地仓位缺少可追溯的入场链路',
    entry_decision_risk_contract_missing: '入场决策缺少可追溯的风险合同',
    final_notional_reduced_from_dynamic_target: '最终仓位已缩减到可执行且不超风险预算的数量；调整已经完成，不是待处理异常',
    upstream_sizing_ineligible: '上游风险定仓未通过，系统已禁止提交订单',
    execution_reconciliation_inputs_incomplete: '成交校验所需的仓位、杠杆、止损或保证金数据不完整，系统已阻止继续执行',
    execution_leverage_tier_contract_missing: '缺少 OKX 杠杆档位合同，系统已阻止继续执行',
    execution_leverage_exceeds_selected_okx_tier: '计划杠杆超过 OKX 当前档位上限，系统已阻止继续执行',
    execution_notional_exceeds_authoritative_target: '实际成交名义价值超过风险合同上限，该成交已标记为风险合同违规，不能作为合格执行证据',
    execution_stressed_loss_exceeds_risk_budget: '实际成交后的压力损失超过风险预算，该成交已标记为风险合同违规，不能作为合格执行证据',
    entry_opportunity_evidence_missing: '缺少入场机会证据',
    claimed_profit_target_distribution_missing: '模型声明缺少权威盈利目标分布',
    live_ml_return_distribution_missing: '模型实盘缺少盈利分布证据',
    live_ml_profit_contract_missing_or_ineligible: '模型实盘盈利合同缺失或不合格',
    live_ml_profit_contract_provenance_incomplete: '模型实盘盈利合同来源不完整',
    opportunity_profit_distribution_ineligible: '当前机会盈利分布未达到模型实盘标准',
    live_rules_canary_contract_incomplete: '规则小仓交易合同不完整',
    live_rules_canary_gate_mode_invalid: '规则小仓交易模式不正确',
    production_trade_gate_not_open: '统一交易闸门未放行',
    rules_canary_authority_not_rules: '规则未获得小仓交易决策权',
    rules_canary_model_influence_not_disabled: '规则小仓仍受未晋升模型影响',
    rules_canary_signal_missing_or_ineligible: '规则小仓方向信号缺失或不合格',
    rules_canary_signal_authority_invalid: '规则未获得小仓方向决策权',
    rules_canary_signal_model_influence_invalid: '规则小仓方向仍受未晋升模型影响',
    rules_canary_signal_provenance_incomplete: '规则小仓方向信号来源不完整',
    rules_canary_signal_action_mismatch: '待提交方向与规则小仓信号不一致',
    rules_canary_model_shadow_contract_missing: '模型旁路观察合同缺失',
    rules_canary_signal_price_missing: '规则小仓方向缺少当前价格',
    rules_canary_signal_risk_anchor_missing: '规则小仓方向缺少 ATR 或波动率风险锚点',
    rules_canary_signal_votes_insufficient: '规则小仓方向有效技术信号不足',
    rules_canary_signal_consensus_weak: '规则小仓方向一致性不足',
    rules_canary_signal_direction_tied: '规则小仓多空方向持平',
    rules_canary_max_notional_missing: '规则小仓名义金额上限缺失',
    rules_canary_exchange_minimum_incomplete: '交易所最小合约规格不完整',
    rules_canary_notional_below_exchange_minimum: '规则小仓金额低于交易所最小下单金额',
    rules_canary_exchange_minimum_contract_mismatch: '交易所最小下单合同与规格不一致',
    rules_canary_daily_loss_budget_exhausted: '规则小仓当日亏损预算已用完',
    rules_canary_max_open_positions_missing: '规则小仓最大并发仓位数缺失',
    rules_canary_order_notional_missing: '规则小仓订单金额缺失',
    rules_canary_order_notional_above_gate_limit: '规则小仓订单金额超过闸门上限',
    rules_canary_execution_cost_incomplete: '规则小仓实时执行成本不完整',
    rules_canary_risk_sizing_ineligible: '规则小仓风险定仓不合格',
    rules_canary_risk_budget_algebra_invalid: '规则小仓风险预算计算不一致',
    rules_canary_stressed_loss_algebra_invalid: '规则小仓压力亏损计算不一致',
    rules_canary_leverage_not_one: '规则小仓必须使用 1 倍杠杆',
    live_execution_cost_missing: '缺少实时执行成本证据',
    expected_net_breakdown_missing: '缺少费后预期收益拆分',
    live_ml_profit_contract_missing: '模型实盘盈利合同缺失',
    'position size capped by stress-stop budget to prevent small-win-big-loss structure': '仓位已按压力止损预算缩小，避免小赚大亏结构',
    'timed out': '请求超时',
    TimeoutError: '请求超时',
    ReadTimeout: '读取响应超时',
    network_error: '网络连接异常',
    runtime_heartbeat_unavailable: '运行心跳不可用',
    trading_runtime_inactive: '交易服务当前未运行',
    trading_runtime_heartbeat_stale: '交易服务心跳已过期',
    ok: '当前正常',
    ready: '当前已就绪',
    active: '当前运行中',
    warning: '当前需要关注',
    blocked: '当前已阻断',
    unavailable: '当前不可用',
});

function dashboardDiagnosticDetailText(value) {
    const text = String(value || '').trim();
    if (!text) return '';
    if (/[\u3400-\u9fff]/.test(text)) return text;
    if (/timed?\s*out|timeout|ReadTimeout/i.test(text)) return '请求超时';
    if (/connection refused|could not connect|connect error|network error/i.test(text)) return '底层连接失败';
    if (/okx executor unavailable/i.test(text)) return 'OKX 执行器不可用';
    if (/OCO inventory unavailable/i.test(text)) return 'OCO 保护单快照不可用';
    if (/risk evidence store unavailable/i.test(text)) return '风险证据存储不可用';
    return '';
}

function dashboardReasonText(value) {
    const item = value && typeof value === 'object' ? value : {};
    const code = String(item.code || item.reason || (typeof value === 'string' ? value : '') || '').trim();
    const message = String(item.message || '').trim();
    const rawText = message || code;
    const [baseCode, ...detailParts] = code.split(':');
    const registeredText = DASHBOARD_REASON_TEXT[code] || DASHBOARD_REASON_TEXT[baseCode];
    if (registeredText) {
        const detailText = dashboardDiagnosticDetailText(detailParts.join(':'));
        return detailText ? `${registeredText}（${detailText}）` : registeredText;
    }
    if (code.startsWith('paper_observation_unsafe:')) {
        return `模拟盘观察存在不安全项：${code.split(':').slice(1).join(':')}`;
    }
    const sideMatch = code.match(/^(long|short)_(.+)$/);
    if (sideMatch) {
        const sideText = sideMatch[1] === 'long' ? '做多' : '做空';
        const suffix = sideMatch[2];
        const sideReasons = {
            top_return_not_above_bottom: '高分组费后收益没有高于低分组',
            top_return_lcb_not_positive: '高分组收益置信下界不为正',
            top_profit_factor_not_above_one: '高分组盈亏比没有高于自然盈亏平衡线 1',
            top_tail_loss_not_improved: '高分组尾部亏损没有改善',
            walk_forward_return_stability_failed: '费后收益在时间滚动验证中不稳定',
            market_regime_stability_failed: '费后收益未能在至少两种市场状态下稳定为正',
            walk_forward_return_evidence_not_ready: '时间滚动费后收益证据尚未就绪',
            walk_forward_fold_not_ready: '时间滚动验证分折证据尚未就绪',
            leave_one_symbol_out_stability_failed: '收益过度依赖单一币种',
            leave_one_symbol_out_not_stable: '逐币种剔除验证仍不稳定',
            oos_return_evidence_not_ready: '样本外收益证据尚未就绪',
            oos_profit_factor_undefined: '样本外盈亏比无法计算',
            oos_profit_factor_not_above_break_even: '样本外盈亏比没有高于自然盈亏平衡线 1',
            oos_return_tail_evidence_incomplete: '样本外尾部风险证据不完整',
            authoritative_profit_factor_undefined: '真实成交盈亏比无法计算',
            authoritative_profit_factor_not_above_break_even: '真实成交盈亏比没有高于自然盈亏平衡线 1',
            authoritative_return_tail_evidence_incomplete: '真实成交尾部风险证据不完整',
            authoritative_realized_return_calibration_missing: '缺少真实成交收益校准',
            authoritative_slippage_calibration_missing: '缺少真实滑点校准',
            authoritative_return_evidence_not_ready: '真实成交收益证据尚未就绪',
            return_distribution_input_missing: '缺少收益分布输入',
            return_distribution_input_version_mismatch: '收益分布输入版本不匹配',
        };
        if (sideReasons[suffix]) return `${sideText}${sideReasons[suffix]}`;
    }
    const dynamicReasons = [
        [/^position_\d+_entry_order_lineage_missing$/, '历史仓位未保存开仓订单链路'],
        [/^position_\d+_entry_order_not_loaded$/, '历史仓位关联的开仓订单已不在当前保留窗口'],
        [/^entry_order_.+_decision_not_loaded$/, '历史开仓订单关联的决策已不在当前保留窗口'],
    ];
    for (const [pattern, text] of dynamicReasons) {
        if (pattern.test(baseCode)) return text;
    }
    if (/Unterminated string/i.test(rawText)) {
        return '状态响应被截断，JSON 不完整；这属于监控读取错误，不代表模型损坏';
    }
    const englishPatterns = [
        [/^(long|short) top-score return confidence lower bound is not positive\.?$/i, '高分组收益置信下界不为正'],
        [/^(long|short) fee-after return evidence is not stable across walk-forward folds\.?$/i, '费后收益在时间滚动验证各折之间不稳定'],
        [/^(long|short) return evidence depends on at least one removed symbol\.?$/i, '收益证据过度依赖至少一个单独币种'],
        [/^(long|short) OOS Profit Factor is not above natural break-even\.?$/i, '样本外盈亏比没有高于自然盈亏平衡线 1'],
    ];
    for (const [pattern, text] of englishPatterns) {
        const match = rawText.match(pattern);
        if (match) return `${match[1].toLowerCase() === 'long' ? '做多' : '做空'}${text}`;
    }
    if (/timed?\s*out|timeout/i.test(rawText)) return '请求超时，服务没有在监控时限内返回';
    if (/connection refused|could not connect|connect error/i.test(rawText)) return '服务连接失败，请检查服务进程和接口监听状态';
    if (/unauthorized|forbidden|\b401\b|\b403\b/i.test(rawText)) return '接口鉴权失败，请检查服务凭据配置';
    if (/not found|\b404\b/i.test(rawText)) return '接口或模型资源不存在';
    if (/no trained local quant bundle/i.test(rawText)) return '本地量化模型产物尚未生成或尚未注册';
    if (rawText && /[\u3400-\u9fff]/.test(rawText)) return rawText;
    if (/^[a-z][a-z0-9_:-]*$/i.test(code)) {
        return `系统发现一项未确认的结算数据，已暂不纳入训练（诊断编号：${code}）`;
    }
    if (/[A-Za-z]{4}/.test(rawText)) return '系统返回了未中文化的异常说明，已按问题处理';
    return rawText || '原因尚未返回';
}

function mlSampleCounts() {
    const ml = state.mlSignalStatus || {};
    const local = state.localAIToolsStatus || {};
    const autoLast = ml.auto_train_last_result || {};
    const mlSupervision = ml.profit_supervision_report || {};
    const localSupervision = local.profit_supervision_report || {};
    const supervisionCount = (report, key) => {
        if (!Object.prototype.hasOwnProperty.call(report, key)) return null;
        const value = Number(report[key]);
        return Number.isFinite(value) ? value : null;
    };
    const trainingMl = mlFirstNumber(ml, ['training_shadow_sample_count']);
    const completedMl = mlFirstNumber(ml, ['completed_shadow_sample_count']);
    const trainingLocal = mlFirstNumber(local, ['training_shadow_sample_count']);
    const completedLocal = mlFirstNumber(local, ['completed_shadow_sample_count']);
    const trainingLocalTrade = mlFirstNumber(local, ['training_trade_sample_count']);
    const completedLocalTrade = mlFirstNumber(local, ['completed_trade_sample_count']);
    const configuredLimit = mlFirstNumber(ml, ['training_shadow_sample_limit'])
        ?? mlFirstNumber(local, ['training_shadow_sample_limit']);
    const limit = configuredLimit !== null && configuredLimit > 0
        ? configuredLimit
        : null;
    let newCount = mlFirstNumber(ml, ['new_shadow_sample_count'])
        ?? mlFirstNumber(local, ['new_shadow_sample_count']);
    const previousCount = mlFirstNumber(ml, [
            'last_trained_completed_shadow_sample_count',
        ])
        ?? mlFirstNumber(autoLast, [
            'last_trained_completed_shadow_sample_count',
            'last_trained_completed_sample_count',
        ])
        ?? trainingMl;
    if (newCount === null && completedMl !== null && previousCount !== null) {
        newCount = Math.max(completedMl - previousCount, 0);
    }
    return {
        trainingMl,
        completedMl,
        trainingLocal,
        completedLocal,
        trainingLocalTrade,
        completedLocalTrade,
        mlShadowMarket: supervisionCount(mlSupervision, 'shadow_market_sample_count'),
        mlShadowCost: supervisionCount(
            mlSupervision,
            'shadow_counterfactual_cost_sample_count'
        ),
        mlActualReturn: supervisionCount(
            mlSupervision,
            'actual_realized_return_sample_count'
        ),
        localShadowMarket: supervisionCount(
            localSupervision,
            'shadow_market_sample_count'
        ),
        localShadowCost: supervisionCount(
            localSupervision,
            'shadow_counterfactual_cost_sample_count'
        ),
        localActualReturn: supervisionCount(
            localSupervision,
            'actual_realized_return_sample_count'
        ),
        limit,
        newCount,
        trainedCursor: previousCount,
    };
}

function mlSampleCountLabel(value) {
    return value !== null && Number.isFinite(value) ? String(value) : '缺失 / 未评估';
}

function metricNumberLabel(value, digits = 2, missing = '未评估') {
    if (value === null || value === undefined || value === '') return missing;
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(digits) : missing;
}

function profitFactorLabel(value, digits = 2) {
    return metricNumberLabel(value, digits, '未定义');
}

function profitFactorTone(value) {
    if (value === null || value === undefined || value === '') return 'muted';
    const num = Number(value);
    if (!Number.isFinite(num)) return 'muted';
    return num > 1 ? 'good' : 'warn';
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
            ${mlMetricCard('做多影子市场收益下界', signedPctValueLabel(metrics.top_long_return_lcb_pct), '毛市场机会诊断；生产端仍需独立扣除成本和实际滑点尾部', Number(metrics.top_long_return_lcb_pct || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('做空影子市场收益下界', signedPctValueLabel(metrics.top_short_return_lcb_pct), '毛市场机会诊断；生产端仍需独立扣除成本和实际滑点尾部', Number(metrics.top_short_return_lcb_pct || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('做多影子市场 Profit Factor', profitFactorLabel(metrics.top_long_profit_factor), '仅评估影子市场机会分层，不等于实际成交 Profit Factor', profitFactorTone(metrics.top_long_profit_factor))}
            ${mlMetricCard('做空影子市场 Profit Factor', profitFactorLabel(metrics.top_short_profit_factor), '仅评估影子市场机会分层，不等于实际成交 Profit Factor', profitFactorTone(metrics.top_short_profit_factor))}
        </div>
        <div class="ml-panel">
            <div class="ml-panel-title">影子市场机会分层质量</div>
            ${mlMetricCard('做多高分组毛市场收益', signedPctValueLabel(metrics.top_long_avg_return_pct), '成本和实际滑点在生产组合层单独处理', Number(metrics.top_long_avg_return_pct || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('做空高分组毛市场收益', signedPctValueLabel(metrics.top_short_avg_return_pct), '成本和实际滑点在生产组合层单独处理', Number(metrics.top_short_avg_return_pct || 0) > 0 ? 'good' : 'warn')}
            ${mlWinBar('做多高分组胜率', metrics.top_long_win_rate, mlSignalToneByRate(metrics.top_long_win_rate))}
            ${mlWinBar('做多低分组胜率', metrics.bottom_long_win_rate, 'muted')}
            ${mlWinBar('做空高分组胜率', metrics.top_short_win_rate, mlSignalToneByRate(metrics.top_short_win_rate))}
            ${mlWinBar('做空低分组胜率', metrics.bottom_short_win_rate, 'muted')}
            <div class="ml-metrics-grid">
                ${mlMetricCard('做多 AUC（诊断）', metricNumberLabel(metrics.long_auc, 3), '仅用于观察分类器，不影响 ready 或生产权重', 'muted')}
                ${mlMetricCard('做空 AUC（诊断）', metricNumberLabel(metrics.short_auc, 3), '仅用于观察分类器，不影响 ready 或生产权重', 'muted')}
                ${mlMetricCard('做多准确率（诊断）', pctLabel(metrics.long_accuracy, 1), '不能代表收益能力', 'muted')}
                ${mlMetricCard('做空准确率（诊断）', pctLabel(metrics.short_accuracy, 1), '不能代表收益能力', 'muted')}
            </div>
        </div>
        `;
}

function renderModelContributionStats() {
    const container = document.getElementById('model-contribution-stats');
    if (!container) return;
    const data = state.modelContributionStats || {};
    const rows = Array.isArray(data.stats) ? data.stats : [];
    const lineage = data.lineage || {};
    const lineageHtml = renderModelContributionLineage(lineage);
    if (!rows.length) {
        container.innerHTML = `
            ${lineageHtml}
            <div class="analysis-empty">暂无实盘贡献样本。等有新的已平仓记录后，这里会显示每个模型到底帮你赚了还是亏了。</div>`;
        return;
    }
    container.innerHTML = `
        ${lineageHtml}
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
                                <td>${profitFactorLabel(row.profit_factor)}</td>
                            </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
        <div class="analysis-note analysis-note-muted" style="margin:12px;">
            <span>说明</span>${escHtml(data.summary || '按真实已平仓盈亏统计，用于判断哪些模型应该加权，哪些应该降权。')}
        </div>`;
}

function renderModelContributionLineage(lineage = {}) {
    const total = Number(lineage.total_closed_positions || 0);
    const matched = Number(lineage.matched_position_count || 0);
    const orders = Number(lineage.filled_order_count || 0);
    const linked = Number(lineage.orders_with_decision_id || 0);
    const loaded = Number(lineage.orders_with_loaded_decision || 0);
    const ready = Boolean(lineage.ready_for_profit_learning);
    const tone = ready ? 'positive' : (total > 0 ? 'warning' : 'muted');
    const reasonMap = {
        ok: '归因链路正常，已平仓样本可以进入模型贡献学习。',
        no_closed_positions: '最近窗口没有已平仓仓位，贡献统计等待新样本。',
        no_filled_orders_for_symbols: '有已平仓仓位，但没有找到同币种成交订单，需检查 OKX 同步/订单留存。',
        filled_orders_missing_decision_id: '有成交订单，但订单没有 decision_id，模型贡献无法回溯到当时决策。',
        linked_decisions_missing: '订单带 decision_id，但没有加载到对应 AI 决策，需检查决策表/对账链路。',
        position_order_time_or_side_mismatch: '订单和仓位存在币种、方向或时间窗口不匹配，暂不能归因。',
        partial_lineage: '只有部分平仓能归因，贡献表可观察但不能直接用于自动加权。',
    };
    const reason = reasonMap[lineage.reason] || '贡献归因链路仍在收集样本。';
    return `
        <div class="analysis-note analysis-note-${tone} model-contribution-lineage">
            <span>贡献归因链路</span>
            <div class="model-contribution-lineage-grid">
                <strong>平仓 ${total}</strong>
                <strong>成交订单 ${orders}</strong>
                <strong>带决策ID ${linked}</strong>
                <strong>已加载决策 ${loaded}</strong>
                <strong>可归因仓位 ${matched}</strong>
            </div>
            <em>${escHtml(reason)}</em>
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

function mlPredictionEconomicsHtml(record, prediction) {
    const economics = record?.prediction_economics || {};
    const tradeGate = economics.production_trade_gate || {};
    const gateMode = String(economics.execution_mode || tradeGate.mode || '').trim();
    const distribution = standardizedReturnDistribution(prediction, prediction?.best_side);
    const productionDistribution = economics.return_distribution_contract || {};
    const cost = economics.execution_cost || {};
    const breakdown = economics.cost_and_return_breakdown || {};
    const blockers = Array.isArray(economics.blockers) ? economics.blockers : [];
    const metric = (label, value) => `
        <div class="ml-prediction-metric">
            <span>${escHtml(label)}</span>
            <strong>${escHtml(value)}</strong>
        </div>`;
    const distributionMetrics = [
        metric('原始期望', distributionPctLabel(distribution?.raw_expected_return_pct)),
        metric('目标期望', distributionPctLabel(distribution?.objective_expected_return_pct)),
        metric('收益下界', distributionPctLabel(distribution?.lower_quantile_return_pct)),
        metric('不确定性', distributionPctLabel(distribution?.uncertainty_penalty_pct ?? distribution?.dispersion_pct)),
        metric('尾损概率', distributionProbabilityLabel(distribution?.tail_loss_probability)),
        metric('尾损尺度', distributionPctLabel(distribution?.tail_loss_scale_pct)),
    ];
    const costRows = [
        ['往返手续费', cost.fee_pct],
        ['实时滑点', cost.slippage_pct],
        ['盘口价差', cost.spread_pct],
        ['流动性惩罚', cost.liquidity_penalty_pct],
        ['失衡惩罚', cost.imbalance_penalty_pct],
        ['订单冲击', cost.market_impact_pct],
        ['实时总成本', cost.total_pct],
        ['反事实成本期望', breakdown.historical_counterfactual_cost_expected_pct],
        ['反事实成本不确定性', breakdown.historical_counterfactual_cost_uncertainty_pct],
        ['权威滑点尾部增量', breakdown.authoritative_slippage_tail_excess_pct],
    ];
    const contractReady = economics.available === true
        && distribution?.production_eligible === true
        && economics.production_eligible === true;
    const rulesCanary = gateMode === 'live_rules_canary';
    const headlineStatus = contractReady
        ? '生产合同完整'
        : (rulesCanary ? '规则小仓采样' : '仅观察 / 已阻断');
    return `
        <div class="ml-prediction-contract ${contractReady || rulesCanary ? 'ready' : 'blocked'}">
            <div class="ml-prediction-contract-head">
                <strong>收益分布与逐项成本</strong>
                <span>${headlineStatus}</span>
            </div>
            <div class="ml-prediction-distribution">${distributionMetrics.join('')}</div>
            <div class="ml-prediction-costs">
                ${costRows.map(([label, value]) => `<span><b>${escHtml(label)}</b>${distributionPctLabel(value)}</span>`).join('')}
            </div>
            <div class="ml-prediction-provenance">
                <span>权威成交结果优先</span>
                <span>生产净收益 ${distributionPctLabel(productionDistribution.raw_expected_return_pct ?? breakdown.net_pct)}</span>
                <span>生产收益下界 ${distributionPctLabel(productionDistribution.objective_expected_return_pct)}</span>
                <span>成本扣除次数 ${mlSampleCountLabel(mlOptionalNumber(breakdown.cost_deduction_count))}</span>
            </div>
            ${blockers.length ? `<div class="ml-prediction-blockers">阻断：${blockers.map(item => escHtml(dashboardReasonText(item))).join(' / ')}</div>` : ''}
        </div>`;
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
                        <th>收益分布与成本</th>
                        <th>生产结论</th>
                    </tr>
                </thead>
                <tbody>
                    ${rows.map(record => {
                        const signal = record.ml_signal || {};
                        const pred = mlPrimaryPrediction(signal) || {};
                        const longRate = Number(pred.long_win_rate || 0);
                        const shortRate = Number(pred.short_win_rate || 0);
                        const longDistribution = standardizedReturnDistribution(pred, 'long');
                        const shortDistribution = standardizedReturnDistribution(pred, 'short');
                        const bestDistribution = standardizedReturnDistribution(pred, pred.best_side);
                        const bestObjective = Number(bestDistribution?.objective_expected_return_pct);
                        const tone = Number.isFinite(bestObjective) && bestObjective > 0 ? 'good' : 'warn';
                        return `
                            <tr>
                                <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(record.created_at)}</td>
                                <td>${escHtml(record.symbol || '-')}</td>
                                <td><span class="badge badge-${record.final_action || 'hold'}">${escHtml(actionLabel(record.final_action))}</span></td>
                                <td><span class="analysis-pill analysis-pill-${tone}">${mlSideLabel(pred.best_side)} ${distributionPctLabel(bestDistribution?.objective_expected_return_pct)}</span></td>
                                <td>${distributionPctLabel(longDistribution?.raw_expected_return_pct)}<div style="font-size:10px;color:var(--text-muted);">目标 ${distributionPctLabel(longDistribution?.objective_expected_return_pct)} · 下界 ${distributionPctLabel(longDistribution?.lower_quantile_return_pct)} · 胜率诊断 ${pctLabel(longRate)}</div></td>
                                <td>${distributionPctLabel(shortDistribution?.raw_expected_return_pct)}<div style="font-size:10px;color:var(--text-muted);">目标 ${distributionPctLabel(shortDistribution?.objective_expected_return_pct)} · 下界 ${distributionPctLabel(shortDistribution?.lower_quantile_return_pct)} · 胜率诊断 ${pctLabel(shortRate)}</div></td>
                                <td>${mlPredictionEconomicsHtml(record, pred)}</td>
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
    return renderMLSignalRecent();
}

// ========== Data Collection Dashboard ==========

async function fetchDataCollectionSettings(options = {}) {
    const runtimeNote = document.getElementById('data-external-runtime-note');
    const saveStatus = document.getElementById('data-collection-save-status');
    if (!state.dataCollectionSettingsLoaded && runtimeNote && !options.silent) {
        runtimeNote.textContent = '正在读取线上配置...';
    }
    let data = null;
    try {
        data = await fetchJSON('/api/data-collection/settings');
    } catch (err) {
        if (saveStatus) {
            saveStatus.style.color = 'var(--red)';
            saveStatus.textContent = err?.message || '外部事件采集配置读取失败。';
        }
        return;
    }
    if (!data?.config) {
        if (saveStatus) {
            saveStatus.style.color = 'var(--red)';
            saveStatus.textContent = '外部事件采集配置未返回，已保持当前表单。';
        }
        return;
    }
    state.dataCollectionStatus = {
        ...(state.dataCollectionStatus || {}),
        checked_at: data.checked_at,
        config: data.config,
    };
    const wasLoaded = state.dataCollectionSettingsLoaded;
    fillDataCollectionSettings(data.config);
    if (!wasLoaded && state.dataCollectionSettingsLoaded && saveStatus) {
        saveStatus.style.color = 'var(--text-muted)';
        saveStatus.textContent = '线上配置已加载。';
    }
}

async function fetchDataCollectionStatus(options = {}) {
    const updated = document.getElementById('data-collection-updated');
    if (updated && !options.silent) updated.textContent = '读取中...';
    let data = null;
    try {
        data = await fetchJSON('/api/data-collection/status');
    } catch (err) {
        data = { status: 'error', detail: err?.message || '数据采集状态接口请求失败' };
    }
    if (!data) {
        if (updated) {
            updated.textContent = state.dataCollectionStatus?.checked_at
                ? `${toBeijingTime(state.dataCollectionStatus.checked_at)} · 本次刷新失败`
                : '读取失败';
        }
        renderDataCollectionDashboard({ failedRefresh: true });
        return;
    }
    state.dataCollectionStatus = data;
    renderDataCollectionDashboard();
}

function collectionStatusTone(status, enabled = true) {
    const value = String(status || '').toLowerCase();
    if (!enabled || value === 'disabled' || value === 'not_configured') return 'muted';
    if (['active', 'ok', 'ready', 'running', 'unknown', 'learning_only'].includes(value)) return 'good';
    if (['artifact_unavailable', 'shadow_ready', 'shadow', 'empty'].includes(value)) return 'warn';
    if (['missing_dependency', 'timeout', 'warning', 'degraded', 'invalid_config', 'quarantined', 'downweighted'].includes(value)) return 'warn';
    return 'bad';
}

function collectionStatusLabel(status, enabled = true) {
    if (!enabled) return '未启用';
    const map = {
        active: '运行中',
        disabled: '未启用',
        not_configured: '未配置',
        missing_dependency: '缺少依赖',
        invalid_config: '配置异常',
        timeout: '超时',
        error: '异常',
        client_not_ready: '客户端未就绪',
        invalid_status: '状态异常',
        unknown: '已连接',
        learning_only: '学习中',
        artifact_unavailable: '缺少模型产物',
        shadow_ready: '影子可用',
        shadow: '影子观察',
        canary: '灰度',
        live: '生产',
        empty: '冷启动等待',
        quarantined: '已隔离',
        downweighted: '已降权',
        ready: '可用',
        running: '运行中',
    };
    return map[String(status || '').toLowerCase()] || String(status || '未知');
}

function collectionAgeLabel(minutes) {
    const n = Number(minutes);
    if (!Number.isFinite(n)) return '无数据';
    if (n < 1) return '刚刚更新';
    if (n < 60) return `${monitorNumber(n, 1)} 分钟前`;
    if (n < 1440) return `${monitorNumber(n / 60, 1)} 小时前`;
    return `${monitorNumber(n / 1440, 1)} 天前`;
}

function collectionFreshnessTone(minutes) {
    const n = Number(minutes);
    if (!Number.isFinite(n)) return 'muted';
    if (n <= 360) return 'good';
    if (n <= 1440) return 'warn';
    return 'bad';
}

function collectionFreshnessLabel(minutes) {
    const n = Number(minutes);
    if (!Number.isFinite(n)) return '无最新数据';
    if (n <= 360) return `正常 · 最新 ${collectionAgeLabel(n)}`;
    if (n <= 1440) return `延迟 · 最新 ${collectionAgeLabel(n)}`;
    return `旧源待关注 · 最新 ${collectionAgeLabel(n)}`;
}

function collectionMetric(label, value, subtitle = '', tone = 'muted') {
    return `
        <div class="data-collection-metric data-collection-${tone}">
            <span>${escHtml(label)}</span>
            <strong>${escHtml(value)}</strong>
            ${subtitle ? `<em>${escHtml(subtitle)}</em>` : ''}
        </div>`;
}

function renderPhase3PromotionGate(promotion, localTools = {}) {
    const gate = promotion && typeof promotion === 'object' ? promotion : {};
    const canaryBlockers = Array.isArray(gate.canary_blocking_reasons)
        ? gate.canary_blocking_reasons
        : [];
    const liveBlockers = Array.isArray(gate.live_blocking_reasons)
        ? gate.live_blocking_reasons
        : [];
    return `
        <div class="data-quality-panel">
            <strong>三期模型晋升检查</strong>
            <div class="data-collection-summary data-collection-summary-compact">
                ${collectionMetric('建议阶段', collectionStatusLabel(gate.recommended_stage || localTools.model_stage || 'shadow', true), `训练模式：${collectionStatusLabel(localTools.training_mode || 'shadow', true)}`, gate.live_ml_ready ? 'good' : gate.canary_ready ? 'warn' : 'muted')}
                ${collectionMetric('灰度是否就绪', gate.canary_ready ? '是' : '否', canaryBlockers.slice(0, 2).map(dashboardReasonText).join('；') || '没有灰度阻断', gate.canary_ready ? 'good' : 'warn')}
                ${collectionMetric('生产是否就绪', gate.live_ml_ready ? '是' : '否', liveBlockers.slice(0, 3).map(dashboardReasonText).join('；') || '生产影响权限保持关闭', gate.live_ml_ready ? 'good' : 'warn')}
            </div>
        </div>`;
}

function markDataCollectionSettingsDirty(message = '外部事件采集设置有未保存修改，请点击“保存设置”。') {
    if (!state.dataCollectionSettingsLoaded || state.dataCollectionSettingsSaving) return;
    state.dataCollectionSettingsDirty = true;
    const status = document.getElementById('data-collection-save-status');
    if (status && message) {
        status.style.color = 'var(--accent-light)';
        status.textContent = message;
    }
}

function clearDataCollectionSettingsDirty() {
    state.dataCollectionSettingsDirty = false;
}

function fillDataCollectionSettings(config, options = {}) {
    if (!config || typeof config !== 'object' || !Object.keys(config).length) return;
    const firstHydration = !state.dataCollectionSettingsLoaded;
    if (!firstHydration && state.dataCollectionSettingsDirty && !options.force) {
        const note = document.getElementById('data-external-runtime-note');
        if (note) {
            const dependency = config.external_event_scraper_dependency_installed ? '依赖已安装' : '依赖未安装';
            const runtime = config.external_event_scraper_runtime_active ? '后台可运行' : '后台未运行';
            const sourceMode = config.external_event_scraper_uses_default_sources ? '使用默认源' : '使用自定义源';
            note.textContent = `${dependency} · ${runtime} · ${sourceMode} · 有未保存修改`;
        }
        return;
    }
    const enabled = document.getElementById('data-external-enabled');
    if (enabled) enabled.checked = Boolean(config.external_event_scraper_enabled);
    setInputValue('data-external-interval', config.external_event_scraper_interval_seconds);
    setInputValue('data-external-timeout', config.external_event_scraper_timeout_seconds);
    setInputValue('data-external-max-sources', config.external_event_scraper_max_sources);
    setInputValue('data-external-max-items', config.external_event_scraper_max_items_per_source);
    const apiChannels = config.api_channels || {};
    setInputValue('data-cryptopanic-api-key', apiChannels.cryptopanic?.api_key || '');
    setInputValue('data-coinmarketcal-api-key', apiChannels.coinmarketcal?.api_key || '');
    setInputValue('data-newsapi-api-key', apiChannels.newsapi?.api_key || '');
    renderDataCollectionSourceManager(
        Array.isArray(config.external_event_scraper_sources)
            ? config.external_event_scraper_sources
            : []
    );
    const note = document.getElementById('data-external-runtime-note');
    if (note) {
        const dependency = config.external_event_scraper_dependency_installed ? '依赖已安装' : '依赖未安装';
        const runtime = config.external_event_scraper_runtime_active ? '后台可运行' : '后台未运行';
        const sourceMode = config.external_event_scraper_uses_default_sources ? '使用默认源' : '使用自定义源';
        note.textContent = `${dependency} · ${runtime} · ${sourceMode}`;
    }
    state.dataCollectionSettingsLoaded = true;
    if (firstHydration || options.force) clearDataCollectionSettingsDirty();
}

function renderDataCollectionDashboard(options = {}) {
    const data = state.dataCollectionStatus || {};
    const config = data.config || {};
    const stats = data.stats || {};
    const training = data.training || {};
    const updated = document.getElementById('data-collection-updated');
    const hasError = data.status === 'error' || data.detail || data.error;
    const errorText = data.detail || data.error || '数据采集状态读取失败，请检查 Dashboard API 或登录状态。';
    if (updated && !options.failedRefresh) {
        updated.textContent = data.checked_at ? toBeijingTime(data.checked_at) : '未返回时间';
    }
    if (hasError) {
        const overview = document.getElementById('data-collection-overview');
        if (overview) {
            overview.innerHTML = `<div class="analysis-empty">${escHtml(errorText)}</div>`;
        }
        if (!data.config && !data.sources && !data.stats && !data.training) {
            return;
        }
    }
    if (!Object.keys(data).length) {
        setText('data-collection-updated', '读取失败');
        const overview = document.getElementById('data-collection-overview');
        if (overview) {
            overview.innerHTML = '<div class="analysis-empty">数据采集状态读取失败，请检查 Dashboard API 或登录状态。</div>';
        }
        return;
    }
    fillDataCollectionSettings(config, { force: options.forceSettings === true });
    renderDataCollectionOverview(data, config, stats, training);
    renderDataCollectionSources(data, stats);
    renderDataCollectionFeatureCoverage(data.feature_coverage || {});
    renderDataCollectionTraining(training);
}

function renderDataCollectionOverview(data, config, stats, training) {
    const container = document.getElementById('data-collection-overview');
    if (!container) return;
    const news = stats.news || {};
    const social = stats.social || {};
    const market = stats.market || {};
    const localTools = training.local_ai_tools || {};
    const scraplingEnabled = Boolean(config.external_event_scraper_enabled);
    const scraplingStatus = scraplingEnabled
        ? (config.external_event_scraper_dependency_installed ? 'active' : 'missing_dependency')
        : 'disabled';
    container.innerHTML = `
        <div class="data-collection-health-strip">
            ${collectionMetric(
                'Scrapling 外部事件',
                collectionStatusLabel(scraplingStatus, scraplingEnabled),
                scraplingEnabled ? `间隔 ${monitorNumber(config.external_event_scraper_interval_seconds, 0)} 秒 · 在系统设置中管理` : '默认关闭 · 在系统设置中管理',
                collectionStatusTone(scraplingStatus, scraplingEnabled)
            )}
            ${collectionMetric(
                '新闻样本',
                `${monitorNumber(news.total, 0)} 条`,
                `最新 ${collectionAgeLabel(news.age_minutes)}`,
                Number(news.total || 0) > 0 ? 'good' : 'warn'
            )}
            ${collectionMetric(
                '社媒样本',
                `${monitorNumber(social.total, 0)} 条`,
                `最新 ${collectionAgeLabel(social.age_minutes)}`,
                Number(social.total || 0) > 0 ? 'good' : 'warn'
            )}
            ${collectionMetric(
                'Ticker 快照',
                `${monitorNumber(market.ticker_count, 0)} 个`,
                `最新 ${collectionAgeLabel(market.ticker_age_minutes)}`,
                Number(market.ticker_count || 0) > 0 ? 'good' : 'warn'
            )}
            ${collectionMetric(
                '本地量化训练',
                collectionStatusLabel(localTools.status, localTools.available),
                `影子 ${monitorNumber(localTools.shadow_sample_count, 0)} · 交易 ${monitorNumber(localTools.trade_sample_count, 0)} · 文本 ${monitorNumber(localTools.text_sentiment_sample_count, 0)}`,
                collectionStatusTone(localTools.status, localTools.available)
            )}
        </div>
        <div class="data-collection-guide">
            <strong>数据采集页只看状态</strong>
            <span>Scrapling 启停和白名单源已移动到：系统设置 → 外部事件采集。</span>
        </div>`;
}

function dataFeatureStatusTone(status) {
    const value = String(status || '').toLowerCase();
    if (value === 'available' || value === 'ok') return 'good';
    if (value === 'missing' || value === 'critical' || value === 'error') return 'bad';
    if (value === 'stale' || value === 'low_confidence' || value === 'warning') return 'warn';
    return 'muted';
}

function dataFeatureStatusLabel(status) {
    const map = {
        available: '可用',
        missing: '缺失',
        stale: '过期',
        low_confidence: '低可信',
        ok: '正常',
        warning: '需关注',
        critical: '异常',
        error: '读取失败',
    };
    return map[String(status || '').toLowerCase()] || String(status || '未知');
}

function renderDataCollectionFeatureCoverage(featureCoverage) {
    const container = document.getElementById('data-collection-feature-coverage');
    if (!container) return;
    const report = featureCoverage && typeof featureCoverage === 'object' ? featureCoverage : {};
    const features = Array.isArray(report.features) ? report.features : [];
    const missing = Array.isArray(report.missing_features) ? report.missing_features : [];
    const stale = Array.isArray(report.stale_features) ? report.stale_features : [];
    const neutralized = Array.isArray(report.neutralized_features) ? report.neutralized_features : [];
    const waitingForSamples = Boolean(report.waiting_for_decision_samples || report.cold_start_safe);
    const missingTone = missing.length ? (waitingForSamples ? 'warn' : 'bad') : 'good';
    const policyText = report.display_message || '缺失/过期特征默认中性阻断，不能静默当作正常。';
    const problemFeatures = features.filter(item => {
        const status = String(item?.status || '').toLowerCase();
        return ['missing', 'stale', 'low_confidence'].includes(status);
    }).slice(0, 12);
    container.innerHTML = `
        <div class="data-feature-coverage-grid">
            ${collectionMetric('覆盖状态', dataFeatureStatusLabel(report.status), report.audit_only ? '只读报告' : '状态来源异常', dataFeatureStatusTone(report.status))}
            ${collectionMetric('缺失特征', `${monitorNumber(missing.length, 0)} 类`, waitingForSamples ? '冷启动待补齐，不驱动开仓' : (missing.slice(0, 4).join(' / ') || '暂无'), missingTone)}
            ${collectionMetric('过期特征', `${monitorNumber(stale.length, 0)} 类`, stale.slice(0, 4).join(' / ') || '暂无', stale.length ? 'warn' : 'good')}
            ${collectionMetric('中性阻断', `${monitorNumber(neutralized.length, 0)} 类`, '缺失/过期不驱动开仓', neutralized.length ? 'warn' : 'good')}
        </div>
        <div class="data-feature-policy">
            <span>${escHtml(policyText)}</span>
            <span>低可信事件只允许影子观察，不直接驱动真实开仓。</span>
        </div>
        <div class="data-feature-table">
            ${problemFeatures.length ? problemFeatures.map(item => `
                <div class="data-feature-row data-source-${dataFeatureStatusTone(item.status)}">
                    <span>${escHtml(item.label || item.key || '-')}</span>
                    <strong>${escHtml(dataFeatureStatusLabel(item.status))}</strong>
                    <em>${escHtml((item.reasons || []).slice(0, 3).join(' / ') || item.source || '')}</em>
                </div>
            `).join('') : '<div class="analysis-empty compact">暂无缺失、过期或低可信特征。</div>'}
        </div>`;
}

function renderDataCollectionSources(data, stats) {
    const container = document.getElementById('data-collection-sources');
    if (!container) return;
    const sources = Array.isArray(data.sources) ? data.sources : [];
    const byGroup = groupDataCollectionSources(sources);
    const newsSources = (stats.news?.sources || []).slice(0, 12);
    const socialPlatforms = (stats.social?.platforms || []).slice(0, 8);
    const klines = stats.market?.klines || [];
    container.innerHTML = `
        <div class="data-quality-grid">
            <div class="data-quality-panel data-source-panel">
                <strong>系统内置通道</strong>
                <div class="data-source-list">${renderDataCollectionChannelRows(byGroup.system)}</div>
            </div>
            <div class="data-quality-panel data-source-panel">
                <strong>付费 API 通道</strong>
                <div class="data-source-list">${renderDataCollectionChannelRows(byGroup.api)}</div>
            </div>
            <div class="data-quality-panel data-source-panel">
                <strong>Scrapling 白名单源</strong>
                <div class="data-source-list">${renderDataCollectionChannelRows(byGroup.scrapling)}</div>
            </div>
            <div class="data-quality-panel">
                <strong>新闻来源新鲜度</strong>
                ${collectionRows(newsSources, 'name', 'count')}
            </div>
            <div class="data-quality-panel">
                <strong>社媒来源新鲜度</strong>
                ${collectionRows(socialPlatforms, 'name', 'count')}
            </div>
            <div class="data-quality-panel">
                <strong>K 线覆盖</strong>
                ${collectionRows(klines, 'timeframe', 'rows', row => `${monitorNumber(row.symbols, 0)} 个币种 · 最新 ${collectionAgeLabel(row.age_minutes)}`)}
            </div>
        </div>`;
}

function groupDataCollectionSources(sources) {
    return sources.reduce((groups, source) => {
        const group = source.group || 'system';
        if (!groups[group]) groups[group] = [];
        groups[group].push(source);
        return groups;
    }, { system: [], api: [], scrapling: [] });
}

function renderDataCollectionChannelRows(sources) {
    if (!Array.isArray(sources) || !sources.length) {
        return '<div class="analysis-empty compact">暂无通道。</div>';
    }
    return sources.map(source => {
        const tone = collectionStatusTone(source.status, source.enabled);
        return `
            <div class="data-source-line data-source-${tone}">
                <span>${escHtml(source.name || source.key || '-')}</span>
                <strong>${escHtml(collectionStatusLabel(source.status, source.enabled))}</strong>
                <em>${escHtml(source.detail || '')}</em>
            </div>`;
    }).join('');
}

function collectionRows(rows, labelKey, valueKey, subtitleFn = null) {
    if (!Array.isArray(rows) || !rows.length) {
        return '<div class="analysis-empty compact">暂无数据。</div>';
    }
    return `
        <div class="data-collection-table">
            ${rows.map(row => `
                <div class="data-collection-row data-source-${collectionFreshnessTone(row.age_minutes)}">
                    <span>${escHtml(row[labelKey] || '-')}</span>
                    <strong>${monitorNumber(row[valueKey], 0)}</strong>
                    <em>${escHtml(subtitleFn ? subtitleFn(row) : collectionFreshnessLabel(row.age_minutes))}</em>
                </div>
            `).join('')}
        </div>`;
}

function renderDataCollectionTraining(training) {
    const container = document.getElementById('data-collection-training');
    if (!container) return;
    const quality = training.text_sentiment_quality_sample || {};
    const localTools = training.local_ai_tools || {};
    const governance = training.governance || {};
    const localGovernance = governance.local_ai_tools || localTools.governance_report || {};
    const mlGovernance = governance.local_ml_signal || {};
    const currentEpochCount = Number(
        governance.training_shadow_sample_count
        ?? localGovernance.current_epoch_trainable_sample_count
        ?? 0
    );
    const quarantinedCount = Number(
        governance.quarantined_shadow_sample_count
        ?? localGovernance.quarantined_sample_count
        ?? localGovernance.excluded_sample_count
        ?? 0
    );
    const reasons = Array.isArray(quality.top_reasons) ? quality.top_reasons : [];
    const qualitySources = Array.isArray(quality.top_sources) ? quality.top_sources : [];
    const models = localTools.models && typeof localTools.models === 'object'
        ? Object.entries(localTools.models)
        : [];
    const promotion = localTools.promotion_recommendation && typeof localTools.promotion_recommendation === 'object'
        ? localTools.promotion_recommendation
        : {};
    const canaryBlockers = Array.isArray(promotion.canary_blocking_reasons)
        ? promotion.canary_blocking_reasons
        : [];
    const liveBlockers = Array.isArray(promotion.live_blocking_reasons)
        ? promotion.live_blocking_reasons
        : [];
    const completedShadowText = Number(localTools.completed_shadow_sample_count || 0) > 0
        ? `三期完成 ${monitorNumber(localTools.completed_shadow_sample_count, 0)}`
        : '三期干净样本暂未形成';
    const completedTradeText = Number(localTools.completed_trade_sample_count || 0) > 0
        ? `三期完成 ${monitorNumber(localTools.completed_trade_sample_count, 0)}`
        : '三期干净交易样本暂未形成';
    container.innerHTML = `
        <div class="data-quality-grid">
            <div class="data-quality-panel data-governance-panel">
                <strong>训练数据治理</strong>
                <div class="data-collection-summary data-collection-summary-compact">
                    ${collectionMetric('清洗状态', trainingGovernanceStatusLabel(localGovernance.status || governance.status), trainingGovernanceSummary(localGovernance), trainingGovernanceTone(localGovernance.status || governance.status))}
                    ${collectionMetric('当前训练纪元可训练', `${monitorNumber(currentEpochCount, 0)} 条`, '只使用当前训练纪元数据', currentEpochCount > 0 ? 'good' : 'warn')}
                    ${collectionMetric('隔离样本', `${monitorNumber(quarantinedCount, 0)} 条`, '旧数据不参与三期训练', quarantinedCount ? 'warn' : 'good')}
                    ${collectionMetric('降权样本', `${monitorNumber(localGovernance.downweighted_sample_count, 0)} 条`, '仅限三期干净窗口内弱证据', Number(localGovernance.downweighted_sample_count || 0) ? 'warn' : 'good')}
                </div>
                <div class="data-governance-notes">
                    ${trainingGovernanceNotes(localGovernance, mlGovernance)}
                </div>
            </div>
            <div class="data-quality-panel">
                <strong>文本情绪样本质量</strong>
                <div class="data-collection-summary data-collection-summary-compact">
                    ${collectionMetric('抽样', `${monitorNumber(quality.sampled, 0)} 条`, '最近新闻 + 社媒', 'muted')}
                    ${collectionMetric('纳入', `${monitorNumber(quality.included, 0)} 条`, `有效率 ${monitorNumber((quality.effective_ratio || 0) * 100, 1)}%`, Number(quality.effective_ratio || 0) >= 0.55 ? 'good' : 'warn')}
                    ${collectionMetric('降权', `${monitorNumber(quality.downweighted, 0)} 条`, '弱文本/重复/低信息量', 'warn')}
                    ${collectionMetric('排除', `${monitorNumber(quality.excluded, 0)} 条`, '不进入训练', Number(quality.excluded || 0) ? 'bad' : 'good')}
                </div>
                <div class="data-chip-list">
                    ${reasons.length ? reasons.map(item => `<span>${escHtml(item.reason)} × ${monitorNumber(item.count, 0)}</span>`).join('') : '<span>暂无质量问题原因</span>'}
                </div>
                <div class="data-chip-list">
                    ${qualitySources.length ? qualitySources.map(item => `<span>${escHtml(item.source)}：${monitorNumber(item.trainable, 0)} / ${monitorNumber(item.count, 0)} 可训练</span>`).join('') : '<span>暂无来源质量分布</span>'}
                </div>
            </div>
            <div class="data-quality-panel">
                <strong>本地量化工具训练样本</strong>
                <div class="data-collection-summary data-collection-summary-compact">
                    ${collectionMetric('服务状态', collectionStatusLabel(localTools.status, localTools.available), localTools.available ? '训练接口可用' : (localTools.error || '训练接口不可用'), collectionStatusTone(localTools.status, localTools.available))}
                    ${collectionMetric('三期影子样本', `${monitorNumber(localTools.shadow_sample_count, 0)} 条`, completedShadowText, 'muted')}
                    ${collectionMetric('三期交易样本', `${monitorNumber(localTools.trade_sample_count, 0)} 条`, completedTradeText, 'muted')}
                    ${collectionMetric('文本样本', `${monitorNumber(localTools.text_sentiment_sample_count, 0)} 条`, '新闻/社媒训练输入', 'muted')}
                </div>
                <div class="data-chip-list">
                    ${models.length ? models.map(([name, ready]) => `<span>${escHtml(name)}：${ready ? '已就绪' : '学习中'}</span>`).join('') : '<span>模型状态未返回</span>'}
                </div>
            </div>
            ${renderPhase3PromotionGate(promotion, localTools)}
        </div>`;
}

function trainingGovernanceStatusLabel(status) {
    const normalized = String(status || '').toLowerCase();
    return {
        ok: '已检查',
        clean: '清洁',
        quarantined: '已隔离',
        downweighted: '已降权',
        error: '检查失败',
    }[normalized] || '待检查';
}

function trainingGovernanceTone(status) {
    const normalized = String(status || '').toLowerCase();
    if (normalized === 'error') return 'bad';
    if (normalized === 'quarantined' || normalized === 'downweighted') return 'warn';
    if (normalized === 'clean' || normalized === 'ok') return 'good';
    return 'muted';
}

function trainingGovernanceSummary(report) {
    if (!report || typeof report !== 'object') return '等待治理报告';
    return report.summary || `有效权重 ${monitorNumber((report.effective_weight_ratio || 0) * 100, 1)}%`;
}

function trainingGovernanceNotes(localReport, mlReport) {
    const notes = ['当前训练纪元已重新开始；纪元前数据禁止进入新模型训练。'];
    if (localReport?.requires_artifact_refresh || mlReport?.requires_artifact_refresh) {
        notes.push('清洗策略已生效，下一轮训练只使用当前训练纪元数据。');
    }
    const targets = localReport?.refresh_targets || mlReport?.refresh_targets || [];
    if (Array.isArray(targets) && targets.length) {
        notes.push(`刷新目标：${targets.join(' / ')}`);
    }
    return notes.map(note => `<span>${escHtml(note)}</span>`).join('');
}

async function refreshTrainingGovernance() {
    const container = document.getElementById('data-collection-training');
    if (container) {
        const previous = document.getElementById('training-governance-refreshing');
        if (previous) previous.remove();
        container.insertAdjacentHTML(
            'afterbegin',
            '<div class="analysis-empty compact" id="training-governance-refreshing">正在按清洗视图重训/重建，请稍等...</div>'
        );
    }
    try {
        const data = await postJSON('/api/data-collection/training-governance/refresh', {});
        state.dataCollectionStatus = data || null;
        renderDataCollectionDashboard();
        const message = data?.message || '训练数据治理刷新完成。';
        const updated = document.getElementById('data-collection-updated');
        if (updated) updated.textContent = message;
    } catch (err) {
        const refreshing = document.getElementById('training-governance-refreshing');
        if (refreshing) {
            refreshing.textContent = err?.message || '训练数据治理刷新失败。';
        }
    }
}

function renderDataCollectionSourceManager(sources) {
    const container = document.getElementById('data-external-source-list');
    if (!container) return;
    const rows = Array.isArray(sources) ? sources : [];
    container.innerHTML = rows.length
        ? rows.map((source, index) => dataCollectionSourceRow(source, index)).join('')
        : '<div class="analysis-empty compact">暂无自定义源。可点击“填入推荐源”或“新增源”。</div>';
}

function dataCollectionSourceRow(source = {}, index = 0) {
    const symbols = Array.isArray(source.symbols) ? source.symbols.join(',') : '';
    const status = source.status || (source.valid === false ? 'invalid' : 'active');
    const statusText = source.valid === false
        ? `无效：${source.error || '请检查 URL、名称或权重'}`
        : (status === 'over_limit' ? '有效，但超出本轮采集源数量上限' : '有效');
    const statusClass = source.valid === false
        ? 'bad'
        : (status === 'over_limit' ? 'warn' : 'good');
    return `
        <div class="data-source-editor-row" data-source-index="${index}">
            <label><span>名称</span><input class="form-input" data-source-field="name" value="${escHtml(source.name || '')}" placeholder="ethereum_blog"></label>
            <label><span>URL</span><input class="form-input" data-source-field="url" value="${escHtml(source.url || '')}" placeholder="https://example.com/news"></label>
            <label><span>币种</span><input class="form-input" data-source-field="symbols" value="${escHtml(symbols)}" placeholder="ETH,SOL"></label>
            <label><span>权重</span><input type="number" class="form-input" data-source-field="weight" min="0.2" max="1" step="0.01" value="${escHtml(source.weight ?? 0.6)}"></label>
            <span class="data-source-editor-status data-source-${statusClass}">${escHtml(statusText)}</span>
            <button class="btn btn-sm" type="button" onclick="removeDataCollectionSource(${index})">删除</button>
        </div>`;
}

function currentDataCollectionSourcesFromForm() {
    return Array.from(document.querySelectorAll('#data-external-source-list .data-source-editor-row')).map(row => {
        const read = field => row.querySelector(`[data-source-field="${field}"]`)?.value?.trim() || '';
        return {
            name: read('name') || null,
            url: read('url'),
            symbols: read('symbols').split(',').map(item => item.trim().toUpperCase()).filter(Boolean),
            weight: Number(read('weight') || 0.6),
        };
    }).filter(item => item.url);
}

function readDataCollectionSources() {
    return currentDataCollectionSourcesFromForm();
}

function addDataCollectionSource(source = {}) {
    const sources = currentDataCollectionSourcesFromForm();
    sources.push(source);
    renderDataCollectionSourceManager(sources);
    markDataCollectionSettingsDirty();
}

function removeDataCollectionSource(index) {
    const sources = currentDataCollectionSourcesFromForm();
    sources.splice(Number(index), 1);
    renderDataCollectionSourceManager(sources);
    markDataCollectionSettingsDirty();
}

async function applyRecommendedDataCollectionSources() {
    const status = document.getElementById('data-collection-save-status');
    const sourcesEl = document.getElementById('data-external-source-list');
    const config = state.dataCollectionStatus?.config || {};
    const recommended = Array.isArray(config.recommended_external_event_sources)
        ? config.recommended_external_event_sources
        : [];
    if (!sourcesEl) return;
    if (!recommended.length) {
        if (status) {
            status.style.color = 'var(--red)';
            status.textContent = '推荐源未返回，请先刷新外部事件采集状态。';
        }
        return;
    }
    renderDataCollectionSourceManager(recommended);
    const maxSourcesInput = document.getElementById('data-external-max-sources');
    if (maxSourcesInput) {
        const currentMaxSources = Number(maxSourcesInput.value || 0);
        maxSourcesInput.value = String(Math.min(32, Math.max(currentMaxSources, recommended.length)));
    }
    markDataCollectionSettingsDirty(`已填入 ${recommended.length} 个推荐源，并同步调整每轮源数量；正在保存到后端配置...`);
    await saveDataCollectionSettings({
        successMessage: `已填入并保存 ${recommended.length} 个推荐源，每轮最多源数量已同步到后端配置。`,
    });
}

async function saveDataCollectionSettings(options = {}) {
    const status = document.getElementById('data-collection-save-status');
    if (!state.dataCollectionSettingsLoaded) {
        if (status) {
            status.style.color = 'var(--red)';
            status.textContent = '线上配置尚未加载完成，已阻止保存空表单，请稍后重试。';
        }
        return;
    }
    state.dataCollectionSettingsSaving = true;
    if (status) {
        status.style.color = 'var(--text-muted)';
        status.textContent = '正在保存数据采集配置...';
    }
    try {
        const body = {
            external_event_scraper_enabled: Boolean(document.getElementById('data-external-enabled')?.checked),
            external_event_scraper_interval_seconds: readNumberInput('data-external-interval'),
            external_event_scraper_timeout_seconds: readNumberInput('data-external-timeout'),
            external_event_scraper_max_sources: readNumberInput('data-external-max-sources'),
            external_event_scraper_max_items_per_source: readNumberInput('data-external-max-items'),
            external_event_scraper_sources: readDataCollectionSources(),
            cryptopanic_api_key: document.getElementById('data-cryptopanic-api-key')?.value?.trim() || null,
            coinmarketcal_api_key: document.getElementById('data-coinmarketcal-api-key')?.value?.trim() || null,
            newsapi_api_key: document.getElementById('data-newsapi-api-key')?.value?.trim() || null,
        };
        const data = await postJSON('/api/data-collection/settings', body);
        state.dataCollectionStatus = {
            ...(state.dataCollectionStatus || {}),
            checked_at: data?.checked_at,
            config: data?.config || {},
        };
        clearDataCollectionSettingsDirty();
        fillDataCollectionSettings(data?.config || {}, { force: true });
        if (isPageActive('data-collection')) fetchDataCollectionStatus({ silent: true });
        if (status) {
            status.style.color = 'var(--green)';
            const runtime = data?.runtime_sync?.message ? ` ${data.runtime_sync.message}` : '';
            status.textContent = `${options.successMessage || data?.message || '数据采集配置已保存。'}${runtime}`;
        }
    } catch (err) {
        if (status) {
            status.style.color = 'var(--red)';
            status.textContent = err?.message || '数据采集配置保存失败。';
        }
    } finally {
        state.dataCollectionSettingsSaving = false;
    }
}

// ========== Vector Memory Settings ==========

async function refreshVectorMemoryStatus(options = {}) {
    const statusEl = document.getElementById('vector-memory-runtime-note');
    if (statusEl && !options.silent) statusEl.textContent = '读取中...';
    const data = await fetchJSON('/api/vector-memory/status');
    renderVectorMemoryStatus(data);
}

function renderVectorMemoryStatus(data) {
    const enabled = document.getElementById('vector-memory-enabled');
    const backend = document.getElementById('vector-memory-backend');
    const minScore = document.getElementById('vector-memory-min-score');
    if (enabled) enabled.checked = Boolean(data?.enabled);
    const configuredBackend = data?.configured_backend || data?.backend;
    if (backend && configuredBackend && ['auto', 'zvec', 'jsonl'].includes(String(configuredBackend))) {
        backend.value = configuredBackend;
    }
    if (minScore && minScore.value === '') minScore.value = String(data?.min_score ?? 0.18);
    const note = document.getElementById('vector-memory-runtime-note');
    if (note) {
        const autoLabel = data?.auto_reindex_enabled
            ? (data.auto_reindex_running ? '自动索引中' : '自动维护')
            : '手动维护';
        note.textContent = data
            ? `${collectionStatusLabel(data.status, data.enabled)} · ${data.backend || 'unknown'} · ${monitorNumber(data.document_count, 0)} 条 · ${autoLabel}`
            : '读取失败';
    }
    const panel = document.getElementById('vector-memory-status-panel');
    if (!panel) return;
    if (!data) {
        panel.innerHTML = '<div class="analysis-empty compact">状态读取失败</div>';
        return;
    }
    panel.innerHTML = `
        <div class="data-source-line data-source-${collectionStatusTone(data.status, data.enabled)}">
            <span>运行状态</span>
            <strong>${escHtml(collectionStatusLabel(data.status, data.enabled))}</strong>
            <em>${escHtml(data.backend || '-')}</em>
        </div>
        <div class="data-source-line data-source-muted">
            <span>三期索引样本</span>
            <strong>${monitorNumber(data.document_count, 0)} 条</strong>
            <em>${data.last_reindex_at ? `上次索引 ${toBeijingTime(data.last_reindex_at)}` : (data.auto_reindex_enabled ? '等待新样本自动索引' : '尚未索引')}</em>
        </div>
        <div class="data-source-line data-source-${data.auto_reindex_running ? 'warn' : (data.auto_reindex_enabled ? 'good' : 'muted')}">
            <span>自动维护</span>
            <strong>${data.auto_reindex_enabled ? (data.auto_reindex_running ? '索引中' : '已启用') : '未启用'}</strong>
            <em>${data.auto_reindex_enabled ? `约每 ${monitorNumber((data.auto_reindex_interval_seconds || 1800) / 60, 0)} 分钟检查；手动重建只用于立即刷新` : '关闭后需要手动重建'}</em>
        </div>
        ${data.last_error ? `<div class="data-source-line data-source-bad"><span>最近错误</span><strong>需要关注</strong><em>${escHtml(data.last_error)}</em></div>` : ''}
    `;
}

async function saveVectorMemorySettings() {
    const status = document.getElementById('vector-memory-save-status');
    if (status) {
        status.style.color = 'var(--text-muted)';
        status.textContent = '正在保存向量记忆设置...';
    }
    const body = {
        enabled: Boolean(document.getElementById('vector-memory-enabled')?.checked),
        backend: document.getElementById('vector-memory-backend')?.value || 'auto',
        min_score: readNumberInput('vector-memory-min-score'),
    };
    const data = await postJSON('/api/vector-memory/settings', body);
    renderVectorMemoryStatus(data);
    if (status) {
        status.style.color = data?.status === 'error' ? 'var(--red)' : 'var(--green)';
        status.textContent = data?.status === 'error'
            ? `保存后状态异常：${data.last_error || '未知错误'}`
            : '向量记忆设置已保存；启用前请先清空旧索引，再等待三期新样本重建。';
    }
}

async function clearVectorMemoryIndex() {
    const status = document.getElementById('vector-memory-save-status');
    if (status) {
        status.style.color = 'var(--text-muted)';
        status.textContent = '正在清空旧向量索引...';
    }
    const data = await postJSON('/api/vector-memory/clear', {});
    if (status) {
        status.style.color = data?.status === 'cleared' ? 'var(--green)' : 'var(--red)';
        status.textContent = data?.status === 'cleared'
            ? `旧索引已清空，移除 ${monitorNumber(data.removed, 0)} 条；等待三期新样本重新索引。`
            : `清空失败：${data?.error || data?.status || '未知错误'}`;
    }
    await refreshVectorMemoryStatus({ silent: true });
}

async function reindexVectorMemory() {
    const status = document.getElementById('vector-memory-save-status');
    if (status) {
        status.style.color = 'var(--text-muted)';
        status.textContent = '正在重建向量记忆索引...';
    }
    const data = await postJSON('/api/vector-memory/reindex', {});
    if (status) {
        status.style.color = data?.status === 'ok' ? 'var(--green)' : 'var(--red)';
        status.textContent = data?.status === 'ok'
            ? `已索引 ${monitorNumber(data.indexed, 0)} 条三期新样本。`
            : `重建失败：${data?.error || data?.status || '未知错误'}`;
    }
    await refreshVectorMemoryStatus({ silent: true });
}

async function searchVectorMemory() {
    const status = document.getElementById('vector-memory-search-status');
    const resultEl = document.getElementById('vector-memory-search-results');
    const query = document.getElementById('vector-memory-query')?.value?.trim() || '';
    if (!query) {
        if (status) status.textContent = '请输入检索关键词';
        return;
    }
    if (status) status.textContent = '检索中...';
    const data = await postJSON('/api/vector-memory/search', {
        query,
        top_k: 8,
        min_score: readNumberInput('vector-memory-min-score'),
    });
    const hits = Array.isArray(data?.hits) ? data.hits : [];
    if (status) status.textContent = `命中 ${hits.length} 条`;
    if (!resultEl) return;
    if (!hits.length) {
        resultEl.innerHTML = '<div class="analysis-empty compact">没有足够相似的历史案例</div>';
        return;
    }
    resultEl.innerHTML = hits.map(hit => `
        <div class="data-source-line data-source-${Number(hit.pnl_pct || 0) < 0 ? 'warn' : 'good'}">
            <span>${escHtml(hit.symbol || hit.kind || '-')}</span>
            <strong>${escHtml(hit.kind === 'news' ? '新闻/事件' : '三期决策样本')}</strong>
            <em>相似度 ${(Number(hit.score || 0) * 100).toFixed(0)}% · ${hit.action ? analysisDecisionLabel(hit.action) : '-'} · ${hit.pnl_pct !== null && hit.pnl_pct !== undefined ? signedPctValueLabel(hit.pnl_pct) : '无收益'}</em>
        </div>
    `).join('');
}

// ========== System Audit / Root Cause Radar ==========

async function fetchSystemAudit(options = {}) {
    if (systemAuditRefreshInFlight) return systemAuditRefreshInFlight;
    const updated = document.getElementById('system-audit-updated');
    if (updated && !options.silent) updated.textContent = '巡检中...';
    systemAuditRefreshInFlight = (async () => {
        try {
            const data = await fetchJSON('/api/system-audit/status');
            state.systemAuditStatus = data || null;
            renderSystemAudit();
        } catch (error) {
            const message = error?.message || String(error || '系统巡检接口请求失败');
            if (options.silent && state.systemAuditStatus) {
                if (updated) updated.textContent = '刷新失败，保留上次结果';
                return;
            }
            state.systemAuditStatus = {
                status: 'critical',
                status_label: '异常',
                checked_at: new Date().toISOString(),
                summary: { cards: 1, critical: 1, warning: 0, ok: 0, findings: 1 },
                root_causes: [{
                    key: 'system_audit_api_failed',
                    title: '系统巡检接口',
                    severity: 'critical',
                    summary: message,
                    evidence: [{ label: '接口错误', value: message }],
                    next_actions: ['先检查 Dashboard API 日志、登录状态和 /api/system-audit/status 路由。'],
                }],
                cards: [{
                    key: 'system_audit_api_failed',
                    title: '系统巡检接口',
                    status: 'critical',
                    summary: message,
                    evidence: [{ label: '接口错误', value: message }],
                    next_actions: ['先检查 Dashboard API 日志、登录状态和 /api/system-audit/status 路由。'],
                }],
                safety_note: '根因雷达当前只读巡检；补历史仓位、重启服务、批量训练等动作必须人工确认。',
            };
            renderSystemAudit();
            console.error('系统巡检刷新失败', error);
        }
    })().finally(() => {
        systemAuditRefreshInFlight = null;
    });
    return systemAuditRefreshInFlight;
}

function systemAuditStatusLabel(status) {
    const labels = { ok: '正常', warning: '需关注', critical: '异常', info: '提示' };
    return labels[String(status || '').toLowerCase()] || String(status || '未知');
}

function systemAuditTone(status) {
    const value = String(status || '').toLowerCase();
    if (value === 'critical') return 'critical';
    if (value === 'warning') return 'warning';
    if (value === 'ok') return 'ok';
    return 'info';
}

function systemAuditDisplayStatus(item) {
    if (!item || typeof item !== 'object') return 'info';
    if (item.display_status) return item.display_status;
    const state = String(item.state || '').toLowerCase();
    if (state === 'fixed') return 'ok';
    if (state === 'observing') return 'warning';
    if (state === 'unresolved') {
        const tone = systemAuditTone(item.status);
        return tone === 'ok' || tone === 'info' ? 'warning' : tone;
    }
    return item.status || 'info';
}

function systemAuditValueText(value) {
    if (value === null || value === undefined || value === '') return '-';
    if (typeof value === 'number') return monitorNumber(value, Number.isInteger(value) ? 0 : 3);
    if (typeof value === 'boolean') return value ? '是' : '否';
    if (Array.isArray(value)) {
        if (!value.length) return '无';
        return value.map(item => systemAuditValueText(item)).join('、');
    }
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
}

function systemAuditEvidenceHtml(evidence) {
    const rows = Array.isArray(evidence) ? evidence : [];
    if (!rows.length) return '<div class="system-audit-muted">暂无关键证据</div>'; 
    return `<div class="system-audit-evidence">${rows.map(item => `
        <span><b>${escHtml(item.label || '证据')}</b><em>${escHtml(systemAuditValueText(item.value))}</em></span>
    `).join('')}</div>`;
}

function systemAuditActionsHtml(actions) {
    const rows = Array.isArray(actions) ? actions.filter(Boolean) : [];
    if (!rows.length) return '<div class="system-audit-muted">暂无建议动作</div>'; 
    return `<div class="system-audit-actions">${rows.map(item => `<span>${escHtml(item)}</span>`).join('')}</div>`;
}

function systemAuditDetailValue(value) {
    if (Array.isArray(value)) {
        if (!value.length) return '无';
        const sample = value.slice(0, 3).map(item => systemAuditValueText(item)).join('；');
        return value.length > 3 ? `${sample}；另 ${value.length - 3} 条` : sample;
    }
    return systemAuditValueText(value);
}

function systemAuditDetailsHtml(details) {
    if (!details || typeof details !== 'object') return ''; 
    const rows = Object.entries(details)
        .filter(([, value]) => value !== null && value !== undefined && value !== '')
        .slice(0, 10);
    if (!rows.length) return ''; 
    return `<div class="system-audit-details">${rows.map(([key, value]) => `
        <div><span>${escHtml(key)}</span><strong>${escHtml(systemAuditDetailValue(value))}</strong></div>
    `).join('')}</div>`;
}

function systemAuditShortText(value, maxLength = 140) {
    const text = String(value ?? '').replace(/\s+/g, ' ').trim();
    if (!text) return '-';
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function systemAuditMetric(label, value, subtitle = '') {
    return `
        <div class="system-audit-detail-chip">
            <span>${escHtml(label)}</span>
            <strong>${escHtml(systemAuditValueText(value))}</strong>
            ${subtitle ? `<em>${escHtml(subtitle)}</em>` : ''}
        </div>`;
}

function systemAuditSafeValue(value) {
    if (value === null || value === undefined || value === '') return '-';
    if (typeof value === 'object') return systemAuditShortText(JSON.stringify(value), 120);
    return systemAuditShortText(value, 120);
}

function systemAuditTable(headers, rows) {
    const visibleRows = Array.isArray(rows) ? rows.filter(Boolean) : [];
    if (!visibleRows.length) return '<div class="system-audit-muted">暂无明细</div>'; 
    return `
        <div class="system-audit-table-wrap">
            <table class="system-audit-table">
                <thead><tr>${headers.map(item => `<th>${escHtml(item)}</th>`).join('')}</tr></thead>
                <tbody>${visibleRows.map(row => `
                    <tr>${row.map(item => `<td>${escHtml(systemAuditSafeValue(item))}</td>`).join('')}</tr>
                `).join('')}</tbody>
            </table>
        </div>`;
}

function systemAuditSection(title, body) {
    if (!body) return ''; 
    return `<section class="system-audit-detail-section"><h4>${escHtml(title)}</h4>${body}</section>`;
}

function systemAuditCompactList(title, rows) {
    const items = Array.isArray(rows) ? rows.filter(Boolean) : [];
    if (!items.length) return ''; 
    return `
        <div class="system-audit-compact-list">
            <strong>${escHtml(title)}</strong>
            ${items.map(item => `<span>${escHtml(systemAuditShortText(item, 180))}</span>`).join('')}
        </div>`;
}

function systemAuditTradingDetails(details) {
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('10分钟分析', details.last_10m_decisions, '交易主循环心跳')}
            ${systemAuditMetric('2小时分析', details.last_2h_decisions, '候选评估数量')}
            ${systemAuditMetric('2小时订单', details.last_2h_orders, '执行链路产出')}
            ${systemAuditMetric('当前持仓', details.open_positions, '用于判断容量')}
            ${systemAuditMetric('最新分析', details.latest_decision_at ? toBeijingTime(details.latest_decision_at) : '-', '市场/持仓分析时间')}
            ${systemAuditMetric('最新订单', details.latest_order_at ? toBeijingTime(details.latest_order_at) : '-', '下单/平仓时间')}
        </div>`;
}

function systemAuditOkxDetails(details) {
    const plans = Array.isArray(details.sample_plans) ? details.sample_plans : [];
    const rows = plans.slice(0, 5).map(item => [
        item.symbol || '-',
        sideLabel(item.side || '-'),
        item.quantity ?? '-',
        item.realized_pnl ?? '-',
        item.closed_at ? toBeijingTime(item.closed_at) : '-',
    ]);
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('对账窗口', `${details.window_days || 14} 天`, '只读 dry-run')}
            ${systemAuditMetric('缺失闭仓', details.missing_closed_positions, '不为 0 会影响收益/训练')}
        </div>
        ${systemAuditSection('缺失闭仓样例', systemAuditTable(['交易对', '方向', '数量', '盈亏', '平仓时间'], rows))}`;
}

function systemAuditMarketDataDetails(details) {
    const klines = Array.isArray(details.klines) ? details.klines : [];
    const rows = klines.map(item => [
        item.timeframe || '-',
        item.symbols ?? 0,
        item.rows ?? 0,
        item.latest_at ? toBeijingTime(item.latest_at) : '-',
        item.missing ? '缺失' : (item.stale ? '过期' : '正常'),
    ]);
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('Ticker 数量', details.ticker_count, details.ticker_stale ? '实时行情过期' : '实时行情正常')}
            ${systemAuditMetric('Ticker 最新', details.ticker_latest_at ? toBeijingTime(details.ticker_latest_at) : '-', '行情新鲜度')}
            ${systemAuditMetric('WebSocket ticker', details.websocket_ticker_age_seconds == null ? '-' : `${monitorNumber(details.websocket_ticker_age_seconds, 1)} 秒前`, details.websocket_stale ? '实时流失活' : '实时流正常')}
            ${systemAuditMetric('WebSocket 重连', details.websocket_reconnect_count ?? 0, details.websocket_connected === false ? '当前断开' : '当前连接')}
            ${systemAuditMetric('缺失周期', Array.isArray(details.missing_timeframes) ? details.missing_timeframes.join('、') || '无' : '无')}
            ${systemAuditMetric('过期周期', Array.isArray(details.stale_timeframes) ? details.stale_timeframes.join('、') || '无' : '无')}
        </div>
        ${systemAuditSection('K线覆盖', systemAuditTable(['周期', '币种数', '行数', '最新时间', '状态'], rows))}`;
}

function systemAuditStrategyDetails(details) {
    const fastLoss = Array.isArray(details.fast_loss_positions) ? details.fast_loss_positions : [];
    const fastLossRows = fastLoss.slice(0, 6).map(item => [
        item.symbol || '-',
        sideLabel(item.side || '-'),
        `${monitorNumber(item.hold_minutes || 0, 1)} 分钟`,
        item.realized_pnl ?? '-',
        item.closed_at ? toBeijingTime(item.closed_at) : '-',
    ]);
    const blockedReasons = (Array.isArray(details.top_blocked_reasons) ? details.top_blocked_reasons : [])
        .slice(0, 5)
        .map(item => `${item.count || 0} 次：${systemAuditShortText(item.reason || '-', 120)}`);
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('24小时决策', details.decision_count, '最近样本窗口')}
            ${systemAuditMetric('开仓候选', details.entry_decision_count, 'long/short 候选数')}
            ${systemAuditMetric('负净收益候选', details.negative_expected_net_count, '过高需查成本/收益')}
            ${systemAuditMetric('零净收益候选', details.zero_expected_net_count, '过高需查模型返回')}
        </div>
        ${systemAuditSection('快亏平样本', systemAuditTable(['交易对', '方向', '持仓时长', '盈亏', '平仓时间'], fastLossRows))}
        ${systemAuditCompactList('主要拦截原因', blockedReasons)}`;
}

function systemAuditModelTrainingDetails(details) {
    const localTools = details.local_ai_tools || {};
    const sourceWarnings = Array.isArray(details.source_warnings) ? details.source_warnings : [];
    const optionalSourceWarnings = Array.isArray(details.optional_source_warnings) ? details.optional_source_warnings : [];
    const criticalItems = Array.isArray(details.model_critical_items) ? details.model_critical_items : [];
    const sourceRows = sourceWarnings.slice(0, 6).map(item => [
        item.name || item.key || '-',
        collectionStatusLabel(item.status || '-', item.enabled !== false),
        item.detail || item.message || '-',
    ]);
    const optionalRows = optionalSourceWarnings.slice(0, 6).map(item => [
        item.name || item.key || '-',
        collectionStatusLabel(item.status || '-', false),
        item.detail || item.message || '未配置时只影响新闻/事件覆盖，不代表模型训练硬故障。',
    ]);
    const criticalRows = criticalItems.slice(0, 6).map(item => [
        item.model || item.title || item.key || '-',
        systemAuditStatusLabel(item.status || (item.status_code ? 'critical' : '-')),
        item.error || item.message || item.api_base || '-',
    ]);
    const modeText = details.hard_failure
        ? '硬故障'
        : (details.observing ? '学习观察' : '正常');
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('巡检判断', modeText, details.hard_failure ? '需要立即处理' : '非硬故障不进入未修复')}
            ${systemAuditMetric('本地量化工具', localTools.available ? '可用' : '不可用', collectionStatusLabel(localTools.status || '-', true))}
            ${systemAuditMetric('影子样本', localTools.shadow_sample_count ?? 0, '训练输入')}
            ${systemAuditMetric('交易样本', localTools.trade_sample_count ?? 0, '真实复盘输入')}
            ${systemAuditMetric('文本样本', localTools.text_sentiment_sample_count ?? 0, '新闻/情绪输入')}
            ${systemAuditMetric('治理状态', details.governance_status || '-', '训练数据清洗')}
        </div>
        ${systemAuditSection('硬故障数据源', systemAuditTable(['来源', '状态', '说明'], sourceRows))}
        ${systemAuditSection('可选增强源', systemAuditTable(['来源', '状态', '说明'], optionalRows))}
        ${systemAuditSection('模型服务异常', systemAuditTable(['项目', '状态', '说明'], criticalRows))}`;
}

function systemAuditPhase3ServerMigrationDetails(details) {
    const blockers = Array.isArray(details.blockers) ? details.blockers : [];
    const warnings = Array.isArray(details.warnings) ? details.warnings : [];
    const legacyPaths = Array.isArray(details.legacy_data_paths) ? details.legacy_data_paths : [];
    const forbiddenServices = Array.isArray(details.forbidden_services) ? details.forbidden_services : [];
    const migration = details.migration_manifest || {};
    const release = details.resource_release_marker || details.reset_marker || {};
    const policy = details.migration_policy || {};
    const blockerRows = blockers.slice(0, 10).map(item => [
        dashboardReasonText(item),
        systemAuditStatusLabel(item.severity || '-'),
        dashboardReasonText(item),
        item.evidence || '-',
    ]);
    const warningRows = warnings.slice(0, 8).map(item => [
        dashboardReasonText(item),
        systemAuditStatusLabel(item.severity || '-'),
        dashboardReasonText(item),
        item.evidence || '-',
    ]);
    const pathRows = legacyPaths.slice(0, 12).map(item => [
        item.path || '-',
        item.kind || '-',
        item.size_bytes ?? '-',
        Array.isArray(item.sample_children) ? item.sample_children.slice(0, 5).join(', ') : '-',
    ]);
    const serviceRows = forbiddenServices.slice(0, 12).map(item => [
        item.name || '-',
        item.unit_exists ?? '-',
        item.active ?? '-',
        item.enabled ?? '-',
        item.active_state || '-',
        item.enabled_state || '-',
    ]);
    const approvedRows = [
        ['允许迁移的类别', Array.isArray(policy.approved_categories) ? policy.approved_categories.join(', ') : '-'],
        ['允许的数据来源', Array.isArray(policy.approved_sources) ? policy.approved_sources.join(', ') : '-'],
        ['是否允许整盘复制', policy.whole_disk_copy_allowed ? '是' : '否'],
        ['旧服务器迁移后角色', policy.old_server_production_role_after_migration || '-'],
    ];
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('生产启用是否阻断', details.phase3_go_live_blocked ? '是' : '否', '三期模型服务器检查')}
            ${systemAuditMetric('远端探针', details.remote_probe_available ? '可用' : '不可用', details.error || '只读检查')}
            ${systemAuditMetric('资源释放证明', release.present ? '存在' : '缺失', details.resource_release_marker_path || details.reset_marker_path || '-')}
            ${systemAuditMetric('旧资源是否释放', release.legacy_resources_stopped ? '是' : '否', release.policy_id || details.policy_id || '-')}
            ${systemAuditMetric('三期数据根目录', details.phase3_root || '-', Array.isArray(details.missing_phase3_roots) && details.missing_phase3_roots.length ? '目录缺失' : '已隔离')}
            ${systemAuditMetric('迁移清单', migration.present ? '存在' : '缺失', details.migration_manifest_path || '-')}
            ${systemAuditMetric('迁移条目', migration.item_count ?? 0, migration.whitelist_only ? '仅允许白名单' : '未强制白名单')}
            ${systemAuditMetric('只读保留旧数据路径', details.legacy_data_path_count ?? 0, '保留但与当前运行隔离')}
            ${systemAuditMetric('旧服务残留', details.forbidden_service_count ?? 0, '正常应为 0')}
            ${systemAuditMetric('旧进程残留', details.legacy_process_count ?? 0, '正常应为 0')}
        </div>
        ${systemAuditSection('三期硬阻断', systemAuditTable(['原因', '级别', '说明', '证据'], blockerRows))}
        ${systemAuditSection('迁移观察项', systemAuditTable(['原因', '级别', '说明', '证据'], warningRows))}
        ${systemAuditSection('只读保留的旧数据路径', systemAuditTable(['路径', '类型', '大小', '内容示例'], pathRows))}
        ${systemAuditSection('禁止残留的旧服务', systemAuditTable(['服务', '单元', '运行', '启用', '运行状态', '启用状态'], serviceRows))}
        ${systemAuditSection('白名单迁移策略', systemAuditTable(['策略', '值'], approvedRows))}`;
}

function systemAuditGenericDetailsHtml(details) {
    if (!details || typeof details !== 'object') return '';
    const rows = Object.entries(details)
        .filter(([, value]) => value === null || ['string', 'number', 'boolean'].includes(typeof value))
        .slice(0, 8)
        .map(([key, value]) => systemAuditMetric(key, value));
    return rows.length ? `<div class="system-audit-detail-grid">${rows.join('')}</div>` : '';
}

function systemAuditShadowMissedOpportunityDetails(details) {
    const summary = details.summary || {};
    const blockedCounts = details.blocked_reason_counts || {};
    const observationRows = (Array.isArray(details.return_observations) ? details.return_observations : [])
        .slice(0, 8)
        .map(item => [
            item.symbol || '-',
            sideLabel(item.side || '-'),
            item.sample_count ?? 0,
            item.average_return_pct ?? '-',
            item.return_lower_hinge_pct ?? '-',
            item.observation_only === true ? 'yes' : 'no',
        ]);
    const gapRows = (Array.isArray(details.executed_return_contract_gaps)
        ? details.executed_return_contract_gaps
        : [])
        .slice(0, 8)
        .map(item => [item.decision_id ?? '-', dashboardReasonText(item.reason)]);
    const blockedReasonRows = Object.entries(blockedCounts)
        .slice(0, 8)
        .map(([reason, count]) => [dashboardReasonText(reason), count]);
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('Completed', summary.completed_count || 0, 'shadow rows')}
            ${systemAuditMetric('Missed', summary.missed_count || 0, 'observation only')}
            ${systemAuditMetric('Observed groups', summary.observe_only_count || 0, 'cannot authorize entry')}
            ${systemAuditMetric('Contract gaps', summary.executed_return_contract_gap_count || 0, 'must stay 0')}
        </div>
        ${systemAuditSection('Fee-after return observations', systemAuditTable(['Symbol', 'Side', 'Samples', 'Average return', 'Lower hinge', 'Observe only'], observationRows))}
        ${systemAuditSection('Executed return-contract gaps', systemAuditTable(['Decision', 'Reason'], gapRows))}
        ${systemAuditSection('Blocked reason counts', systemAuditTable(['Reason', 'Count'], blockedReasonRows))}`;
}

function systemAuditOkxDetailsV2(details) {
    const plans = Array.isArray(details.sample_plans) ? details.sample_plans : [];
    const rootCause = details.root_cause_summary || {};
    const trainingPolicy = details.training_data_policy || {};
    const authoritative = details.okx_authoritative_sync || {};
    const runtimeGate = details.runtime_okx_entry_gate || {};
    const dailyReport = details.daily_reconciliation_report || {};
    const rootCauseRows = Array.isArray(rootCause.root_causes)
        ? rootCause.root_causes.slice(0, 6).map(item => [
            dashboardReasonText(item),
            item.count ?? 0,
            item.training_policy || '-',
            item.action || '-',
        ])
        : [];
    const authoritativeIssueRows = (Array.isArray(authoritative.issues) ? authoritative.issues : [])
        .slice(0, 8)
        .map(item => [
            item.kind || '-',
            item.classification || '-',
            item.severity || '-',
            item.symbol || '-',
            item.side || '-',
            item.exchange_order_id || item.local_order_id || item.local_position_id || '-',
            dashboardReasonText(item),
        ]);
    const runtimeKindRows = Object.entries(runtimeGate.last_result_kinds || {})
        .slice(0, 8)
        .map(([kind, count]) => [kind, count]);
    const runtimeSampleRows = (Array.isArray(runtimeGate.last_samples) ? runtimeGate.last_samples : [])
        .slice(0, 8)
        .map(item => [
            item.kind || '-',
            item.symbol || '-',
            sideLabel(item.side || '-'),
            item.requires_attention === true ? 'yes' : 'no',
            item.exchange_order_id || '-',
            item.note || '-',
        ]);
    const runtimeGateState = runtimeGate.entry_blocked === true
        ? 'blocked'
        : (runtimeGate.entry_blocked === false ? 'open' : 'unknown');
    const dailyReportState = dailyReport.stale === true
        ? 'stale'
        : (dailyReport.status || (dailyReport.available === false ? 'missing' : 'unknown'));
    const dailyLedger = dailyReport.issue_ledger_summary || {};
    const dailyBuckets = dailyReport.attention_buckets || {};
    const dailyEntryBlockerRows = (Array.isArray(dailyReport.entry_blockers) ? dailyReport.entry_blockers : [])
        .slice(0, 8)
        .map(item => [
            dashboardReasonText(item),
            item.card_key || '-',
            item.status || '-',
            item.requires_attention === true ? 'yes' : 'no',
            item.summary || '-',
        ]);
    const dailyTrainingBlockerRows = (Array.isArray(dailyReport.training_blockers) ? dailyReport.training_blockers : [])
        .slice(0, 8)
        .map(item => [
            dashboardReasonText(item),
            item.card_key || '-',
            item.status || '-',
            item.summary || '-',
        ]);
    const rows = plans.slice(0, 5).map(item => [
        item.symbol || '-',
        sideLabel(item.side || '-'),
        item.quantity ?? '-',
        item.realized_pnl ?? '-',
        item.closed_at ? toBeijingTime(item.closed_at) : '-',
        item.exchange_order_id || '-',
    ]);
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('OKX window', `${details.window_days || 14} days`, 'read-only dry-run')}
            ${systemAuditMetric('Missing closes', details.missing_closed_positions, 'affects PnL/training if >0')}
            ${systemAuditMetric('Root cause', rootCause.status || '-', 'OKX/local mismatch class')}
            ${systemAuditMetric('Repairable', rootCause.repairable_count ?? details.repairable_count ?? 0, 'dry-run only')}
            ${systemAuditMetric('Manual review', rootCause.manual_review_count ?? details.manual_review_count ?? 0, 'needs OKX fact check')}
            ${systemAuditMetric('Unscanned', rootCause.unscanned_candidate_count ?? details.unscanned_candidate_count ?? 0, 'run full scan if >0')}
            ${systemAuditMetric('OKX API pull', authoritative.okx_pull_available === false ? 'unavailable' : 'available', 'private API facts')}
            ${systemAuditMetric('OKX positions', authoritative.okx_position_count ?? 0, 'current exchange positions')}
            ${systemAuditMetric('OKX fill orders', authoritative.okx_fill_order_count ?? 0, 'recent exchange fills')}
            ${systemAuditMetric('OKX sync issues', authoritative.issue_count ?? 0, 'manual review/repairable/skipped')}
            ${systemAuditMetric('Entry gate', runtimeGateState, dashboardReasonText(runtimeGate.blocker || runtimeGate.status || 'runtime OKX sync'))}
            ${systemAuditMetric('Daily report', dailyReportState, dailyReport.generated_at ? toBeijingTime(dailyReport.generated_at) : 'latest timer artifact')}
            ${systemAuditMetric('Can train', dailyReport.can_refresh_training === true ? 'yes' : 'no', dailyReport.training_blocked === true ? 'training blocked' : 'clean-view allowed')}
        </div>
        ${systemAuditSection('Latest OKX daily report gates', systemAuditTable(['Report status', 'Age seconds', 'Can open', 'Can train', 'Requires attention', 'Entry blockers', 'Training blockers', 'Unresolved'], [[
            dailyReportState,
            dailyReport.age_seconds ?? '-',
            dailyReport.can_open_new_entries ?? '-',
            dailyReport.can_refresh_training ?? '-',
            dailyReport.requires_attention ?? '-',
            dailyBuckets.entry ?? '-',
            dailyBuckets.training ?? '-',
            dailyLedger.unresolved ?? '-',
        ]]))}
        ${systemAuditSection('Daily entry blockers', systemAuditTable(['Code', 'Card', 'Status', 'Attention', 'Summary'], dailyEntryBlockerRows))}
        ${systemAuditSection('Daily training blockers', systemAuditTable(['Code', 'Card', 'Status', 'Summary'], dailyTrainingBlockerRows))}
        ${systemAuditSection('Runtime OKX entry gate', systemAuditTable(['Running', 'Runtime status', 'Sync status', 'Entry blocked', 'Heartbeat age', 'Fresh limit', 'Blocker', 'Reason'], [[
            runtimeGate.running ?? '-',
            runtimeGate.status || '-',
            runtimeGate.sync_status || '-',
            runtimeGate.entry_blocked,
            runtimeGate.heartbeat_age_seconds ?? '-',
            runtimeGate.heartbeat_fresh_limit_seconds ?? '-',
            dashboardReasonText(runtimeGate.blocker || '-'),
            dashboardReasonText(runtimeGate.reason || '-'),
        ]]))}
        ${systemAuditSection('Runtime OKX sync result kinds', systemAuditTable(['Kind', 'Count'], runtimeKindRows))}
        ${systemAuditSection('Runtime OKX sync samples', systemAuditTable(['Kind', 'Symbol', 'Side', 'Requires attention', 'Exchange order', 'Note'], runtimeSampleRows))}
        ${systemAuditSection('OKX authoritative sync policy', systemAuditTable(['Read-only', 'Can write DB', 'Backup required'], [[
            authoritative.read_only ?? true,
            authoritative.can_write_database ?? authoritative.apply_policy?.can_write_database ?? false,
            authoritative.apply_policy?.requires_backup ?? true,
        ]]))}
        ${systemAuditSection('OKX authoritative sync issues', systemAuditTable(['Kind', 'Class', 'Severity', 'Symbol', 'Side', 'ID', 'Reason'], authoritativeIssueRows))}
        ${systemAuditSection('Training data policy', systemAuditTable(['Policy', 'Cleanup', 'Rebuild'], [[
            trainingPolicy.policy || rootCause.training_policy || '-',
            trainingPolicy.cleanup_mode || rootCause.cleanup_mode || '-',
            trainingPolicy.requires_training_rebuild ?? rootCause.requires_training_rebuild ?? false,
        ]]))}
        ${systemAuditSection('OKX root causes', systemAuditTable(['Code', 'Count', 'Training policy', 'Action'], rootCauseRows))}
        ${systemAuditSection('Missing close samples', systemAuditTable(['Symbol', 'Side', 'Qty', 'PnL', 'Closed at', 'Exchange order'], rows))}`;
}

function systemAuditPositionPriceDetails(details) {
    const root = details.root_cause_summary || {};
    const rootRows = Object.entries(root.root_cause_counts || {})
        .slice(0, 10)
        .map(([code, count]) => [code, count]);
    const posSideRows = Object.entries(details.okx_pos_side_counts || root.okx_pos_side_counts || {})
        .slice(0, 10)
        .map(([code, count]) => [code, count]);
    const sideInferenceRows = Object.entries(details.okx_side_inference_counts || root.okx_side_inference_counts || {})
        .slice(0, 10)
        .map(([code, count]) => [code, count]);
    const splitRows = (Array.isArray(details.splits) ? details.splits : [])
        .slice(0, 8)
        .map(item => [
            item.mode || '-',
            item.symbol || '-',
            sideLabel(item.side || '-'),
            item.local_price ?? '-',
            item.okx_price ?? '-',
            item.price_gap_pct ?? '-',
            item.local_unrealized_pnl ?? '-',
            item.okx_unrealized_pnl ?? '-',
            item.okx_pos_side || '-',
            item.okx_raw_pos ?? '-',
            item.okx_side_inference || '-',
            item.root_cause || '-',
        ]);
    const localOnlyRows = (Array.isArray(details.local_only_positions) ? details.local_only_positions : [])
        .slice(0, 8)
        .map(item => [
            item.mode || '-',
            item.position_id ?? '-',
            item.symbol || '-',
            sideLabel(item.side || '-'),
            item.local_quantity ?? '-',
            item.local_price ?? '-',
            item.local_unrealized_pnl ?? '-',
        ]);
    const exchangeOnlyRows = (Array.isArray(details.exchange_only_positions) ? details.exchange_only_positions : [])
        .slice(0, 8)
        .map(item => [
            item.mode || '-',
            item.symbol || '-',
            sideLabel(item.side || '-'),
            item.okx_quantity ?? '-',
            item.okx_price ?? '-',
            item.okx_unrealized_pnl ?? '-',
            item.okx_contracts ?? '-',
            item.okx_contract_size ?? '-',
            item.okx_raw_symbol || '-',
            item.okx_pos_side || '-',
            item.okx_raw_pos ?? '-',
            item.okx_side_inference || '-',
        ]);
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('Mismatch status', root.status || '-', 'OKX/local position state')}
            ${systemAuditMetric('Mismatch total', details.mismatch_count ?? root.mismatch_count ?? 0, 'all mismatch classes')}
            ${systemAuditMetric('Price/PnL splits', details.split_count ?? root.split_count ?? 0, 'matched positions differ')}
            ${systemAuditMetric('Local only', details.local_only_count ?? root.local_only_count ?? 0, 'local open but not OKX')}
            ${systemAuditMetric('OKX only', details.exchange_only_count ?? root.exchange_only_count ?? 0, 'OKX open but not local')}
            ${systemAuditMetric('Repair mutation', details.live_repair_mutation === false ? 'disabled' : '-', 'read-only audit')}
        </div>
        ${systemAuditSection('Position mismatch root causes', systemAuditTable(['Root cause', 'Count'], rootRows))}
        ${systemAuditSection('OKX position mode counts', systemAuditTable(['posSide', 'Count'], posSideRows))}
        ${systemAuditSection('OKX side inference counts', systemAuditTable(['Inference', 'Count'], sideInferenceRows))}
        ${systemAuditSection('Price/PnL split samples', systemAuditTable(['Mode', 'Symbol', 'Side', 'Local price', 'OKX price', 'Gap %', 'Local UPL', 'OKX UPL', 'posSide', 'Raw pos', 'Inference', 'Root cause'], splitRows))}
        ${systemAuditSection('Local-only open positions', systemAuditTable(['Mode', 'Position ID', 'Symbol', 'Side', 'Qty', 'Local price', 'Local UPL'], localOnlyRows))}
        ${systemAuditSection('OKX-only open positions', systemAuditTable(['Mode', 'Symbol', 'Side', 'Qty', 'OKX price', 'OKX UPL', 'Contracts', 'ctVal', 'Raw symbol', 'posSide', 'Raw pos', 'Inference'], exchangeOnlyRows))}`;
}

function systemAuditStrategySignalRootCauseDetails(details) {
    const scheduler = details.scheduler || {};
    const capacity = scheduler.dynamic_capacity || {};
    const rootCauses = Array.isArray(details.root_causes) ? details.root_causes : [];
    const samples = Array.isArray(scheduler.latest_samples) ? scheduler.latest_samples : [];
    const topReasons = Array.isArray(scheduler.top_scheduler_reasons) ? scheduler.top_scheduler_reasons : [];
    const objectRows = (value) => Object.entries(value && typeof value === 'object' ? value : {})
        .slice(0, 10)
        .map(([key, count]) => [key, count]);
    const rootCauseRows = rootCauses.slice(0, 8).map(item => [
        item.code || '-',
        item.severity || '-',
        item.count ?? '-',
        item.message || '-',
    ]);
    const sampleRows = samples.slice(0, 8).map(item => {
        const capacityView = item.dynamic_position_capacity || {};
        return [
            item.symbol || '-',
            sideLabel(item.action || '-'),
            item.strategy || '-',
            item.posture || '-',
            item.risk_mode || '-',
            item.cache_status || '-',
            capacityView.entry_limit ?? '-',
            capacityView.open_group_count ?? '-',
            item.scheduler_reason || item.reason || '-',
        ];
    });
    const reasonRows = topReasons.slice(0, 8).map(item => [
        item.reason || '-',
        item.count ?? 0,
    ]);
    return `
        <div class="system-audit-detail-grid">
            ${systemAuditMetric('Entry candidates', details.entry_decision_count || 0, 'long/short decisions')}
            ${systemAuditMetric('High quality', details.high_quality_entry_count || 0, 'exploration/small/medium/normal')}
            ${systemAuditMetric('Scheduler samples', scheduler.sample_count || 0, 'strategy_mode coverage')}
            ${systemAuditMetric('Learning timeouts', (scheduler.flag_counts || {}).strategy_learning_context_timeout || 0, 'cache/baseline fallback')}
            ${systemAuditMetric('Capacity constrained', capacity.constrained_count || 0, 'read-only diagnosis')}
            ${systemAuditMetric('Root causes', rootCauses.length, 'no live mutation')}
        </div>
        ${systemAuditSection('Scheduler strategy distribution', systemAuditTable(['Strategy', 'Count'], objectRows(scheduler.strategy_counts)))}
        ${systemAuditSection('Scheduler flags', systemAuditTable(['Flag', 'Count'], objectRows(scheduler.flag_counts)))}
        ${systemAuditSection('Dynamic capacity reason codes', systemAuditTable(['Reason code', 'Count'], objectRows(capacity.reason_code_counts)))}
        ${systemAuditSection('Top scheduler reasons', systemAuditTable(['Reason', 'Count'], reasonRows))}
        ${systemAuditSection('Root causes', systemAuditTable(['Code', 'Severity', 'Count', 'Message'], rootCauseRows))}
        ${systemAuditSection('Latest scheduler samples', systemAuditTable(['Symbol', 'Action', 'Strategy', 'Posture', 'Risk', 'Cache', 'Entry limit', 'Open groups', 'Reason'], sampleRows))}`;
}

function systemAuditCardDetailsHtml(card) {
    const details = card.details || {};
    const key = String(card.key || '');
    if (key === 'trade_loop') return systemAuditTradingDetails(details);
    if (key === 'okx_reconciliation') return systemAuditOkxDetailsV2(details);
    if (key === 'position_price_integrity') return systemAuditPositionPriceDetails(details);
    if (key === 'market_data') return systemAuditMarketDataDetails(details);
    if (key === 'strategy_quality') return systemAuditStrategyDetails(details);
    if (key === 'strategy_signal_root_cause') return systemAuditStrategySignalRootCauseDetails(details);
    if (key === 'model_training') return systemAuditModelTrainingDetails(details);
    if (key === 'phase3_server_migration') return systemAuditPhase3ServerMigrationDetails(details);
    if (key === 'shadow_missed_opportunity') return systemAuditShadowMissedOpportunityDetails(details);
    return systemAuditGenericDetailsHtml(details);
}

function systemAuditOverviewHtml(data) {
    const status = systemAuditTone(data.status);
    const summary = data.summary || {};
    const title = status === 'ok' ? '当前未发现关键根因' : (status === 'critical' ? '发现异常根因' : '发现需关注项');
    return `
        <div class="system-audit-overview system-audit-overview-${status}">
            <div class="system-audit-hero">
                <span>整体健康度</span>
                <strong>${escHtml(data.status_label || systemAuditStatusLabel(data.status))}</strong>
                <em>${escHtml(title)}</em>
            </div>
            <div class="system-audit-health-strip">
                ${collectionMetric('巡检模块', `${monitorNumber(summary.cards || 0, 0)} 个`, '交易/对账/行情/策略/模型', 'muted')}
                ${collectionMetric('异常根因', `${monitorNumber(summary.critical || 0, 0)} 项`, '需要优先处理', summary.critical ? 'bad' : 'good')}
                ${collectionMetric('需关注项', `${monitorNumber(summary.warning || 0, 0)} 项`, '继续观察或排查', summary.warning ? 'warn' : 'good')}
                ${collectionMetric('正常项', `${monitorNumber(summary.ok || 0, 0)} 项`, '已通过只读巡检', 'good')}
            </div>
        </div>`;
}

function systemAuditCardSummaryHtml(card) {
    const evidence = Array.isArray(card.evidence) ? card.evidence.slice(0, 3) : [];
    if (!evidence.length) return '<div class="system-audit-card-summary muted">暂无关键证据</div>'; 
    return `<div class="system-audit-card-summary">${evidence.map(item => `
        <span><b>${escHtml(item.label || '证据')}</b><em>${escHtml(systemAuditValueText(item.value))}</em></span>
    `).join('')}</div>`;
}

function systemAuditCardDetailOpen(card) {
    return systemAuditTone(card.status) !== 'ok';
}

function systemAuditLedgerItemHtml(item) {
    const tone = systemAuditTone(item?.status);
    const evidence = Array.isArray(item?.evidence) ? item.evidence.slice(0, 2) : [];
    return `
        <article class="system-audit-ledger-item system-audit-ledger-item-${tone}">
            <div>
                <strong>${escHtml(item?.title || item?.key || '-')}</strong>
                <em>${escHtml(item?.state_label || systemAuditStatusLabel(item?.status))}</em>
            </div>
            <p>${escHtml(systemAuditShortText(item?.summary || '-', 150))}</p>
            ${evidence.length ? `<div class="system-audit-ledger-evidence">${evidence.map(row => `<span>${escHtml(row.label || '证据')}：${escHtml(systemAuditValueText(row.value))}</span>`).join('')}</div>` : ''}
        </article>`;
}

function renderSystemAuditIssueLedger(ledger) {
    const container = document.getElementById('system-audit-issue-ledger');
    if (!container) return;
    const data = ledger && typeof ledger === 'object' ? ledger : {};
    const summary = data.summary || {};
    const groups = [
        { key: 'unresolved', title: '未修复', hint: '当前仍需处理，优先看这里', tone: 'critical' },
        { key: 'observing', title: '历史观察', hint: '历史遗留或刚修复后观察，不重复改同一问题', tone: 'warning' },
        { key: 'fixed', title: '已修复', hint: '本轮只读巡检验证通过', tone: 'ok' },
    ];
    container.innerHTML = `
        <div class="system-audit-ledger-head">
            <div>
                <strong>问题台账</strong>
                <span>把已修复、未修复、历史遗留分开，避免重复改同一个问题。</span>
            </div>
            <em>未修复 ${monitorNumber(summary.unresolved || 0, 0)} · 历史观察 ${monitorNumber(summary.observing || 0, 0)} · 已修复 ${monitorNumber(summary.fixed || 0, 0)}</em>
        </div>
        <div class="system-audit-ledger-grid">
            ${groups.map(group => {
                const rows = Array.isArray(data[group.key]) ? data[group.key] : [];
                return `
                    <section class="system-audit-ledger-column system-audit-ledger-column-${group.tone}">
                        <div class="system-audit-ledger-column-head">
                            <strong>${escHtml(group.title)}</strong>
                            <span>${monitorNumber(rows.length, 0)} 项 · ${escHtml(group.hint)}</span>
                        </div>
                        <div class="system-audit-ledger-list">
                            ${rows.length ? rows.map(systemAuditLedgerItemHtml).join('') : '<div class="system-audit-ledger-empty">暂无</div>'}
                        </div>
                    </section>`;
            }).join('')}
        </div>`;
}

function renderSystemAuditRootCauses(rootCauses) {
    const container = document.getElementById('system-audit-root-causes');
    if (!container) return;
    const rows = Array.isArray(rootCauses) ? rootCauses : [];
    if (!rows.length) {
        container.innerHTML = '<div class="system-audit-empty">暂无关键根因，继续观察核心指标。</div>'; 
        return;
    }
    container.innerHTML = rows.map(item => {
        const tone = systemAuditTone(item.severity || item.status);
        return `
            <div class="system-audit-root-cause ${tone}">
                <div><strong>${escHtml(item.title || item.key || '-')}</strong><span>${escHtml(systemAuditStatusLabel(tone))}</span></div>
                <p>${escHtml(item.summary || '-')}</p>
                ${systemAuditEvidenceHtml(item.evidence)}
                ${systemAuditActionsHtml(item.next_actions)}
            </div>`;
    }).join('');
}

const SYSTEM_AUDIT_NODE_LABELS = Object.freeze({
    server_migration: '三期服务器迁移',
    model_server_readiness: '模型服务器就绪检查',
    phase3_stage_handoff: '三期阶段交接',
    runtime_loop: '调度与心跳',
    market_data: '行情与 K 线',
    crypto_feature_coverage: '数字货币特征覆盖',
    model_training: '模型与训练数据',
    model_expert_health: '模型/专家体检',
    model_expert_competition: '模型/专家竞赛',
    model_routing: '模型路由',
    high_risk_review_audit: '高风险独立复核',
    shadow_missed_opportunity: '影子错失机会复盘',
    strong_opportunity: '强机会识别',
    position_capacity_release: '持仓容量释放',
    strategy_decision: '策略决策质量',
    strategy_closed_loop: '策略闭环有效性',
    strategy_signal_root_cause: '策略信号根因',
    strategy_gate_contract: '策略门槛契约',
    risk_guard: '风控与守门',
    okx_execution: 'OKX 执行与历史对账',
    position_sync: '持仓同步与盈亏',
    training_data: '训练标签与样本治理',
    dashboard_observability: '页面与可观测性',
    visible_text_encoding: '中文显示与乱码',
    runtime_text_integrity: '运行时文本完整性',
});

function systemAuditNodeLabel(value) {
    const key = String(value || '');
    return SYSTEM_AUDIT_NODE_LABELS[key] || key || '无';
}

function renderSystemAuditCards(cards) {
    const container = document.getElementById('system-audit-cards');
    if (!container) return;
    const rows = Array.isArray(cards) ? cards : [];
    if (!rows.length) {
        container.innerHTML = '<div class="analysis-empty">还没有巡检卡片。</div>'; 
        return;
    }
    const sortedRows = [...rows].sort((left, right) => {
        const priority = { critical: 0, warning: 1, ok: 2 };
        return (priority[systemAuditTone(left.status)] ?? 3) - (priority[systemAuditTone(right.status)] ?? 3);
    });
    container.innerHTML = sortedRows.map(card => {
        const tone = systemAuditTone(card.status);
        const isOpen = systemAuditCardDetailOpen(card);
        return `
            <details class="system-audit-card system-audit-card-${tone}" ${isOpen ? 'open' : ''}>
                <summary class="system-audit-card-head">
                    <div><strong>${escHtml(card.title || card.key || '-')}</strong><span>${escHtml(card.summary || '-')}</span></div>
                    <em>${escHtml(systemAuditStatusLabel(card.status))}</em>
                </summary>
                ${systemAuditCardSummaryHtml(card)}
                <div class="system-audit-card-body">
                    ${systemAuditEvidenceHtml(card.evidence)}
                    ${systemAuditCardDetailsHtml(card)}
                    <div class="system-audit-card-actions">
                        <strong>建议处理</strong>
                        ${systemAuditActionsHtml(card.next_actions)}
                    </div>
                </div>
            </details>`;
    }).join('');
}

function renderSystemAuditNodes(nodes) {
    const container = document.getElementById('system-audit-nodes');
    if (!container) return;
    const rows = Array.isArray(nodes) ? nodes : [];
    if (!rows.length) {
        container.innerHTML = '<div class="analysis-empty">还没有节点图谱。</div>'; 
        return;
    }
    const sortedRows = [...rows].sort((left, right) => {
        const priority = { critical: 0, warning: 1, ok: 2 };
        return (priority[systemAuditTone(systemAuditDisplayStatus(left))] ?? 3) - (priority[systemAuditTone(systemAuditDisplayStatus(right))] ?? 3);
    });
    container.innerHTML = sortedRows.map((node, index) => {
        const displayStatus = systemAuditDisplayStatus(node);
        const tone = systemAuditTone(displayStatus);
        const checks = Array.isArray(node.checks) ? node.checks.slice(0, 4) : [];
        const upstream = Array.isArray(node.upstream) ? node.upstream : [];
        const downstream = Array.isArray(node.downstream) ? node.downstream : [];
        return `
            <article class="system-audit-node system-audit-node-${tone}">
                <div class="system-audit-node-head">
                    <span>${escHtml(node.layer || '节点')}</span>
                    <em>${escHtml(node.state_label || systemAuditStatusLabel(displayStatus))}</em>
                </div>
                <div class="system-audit-node-index">${String(index + 1).padStart(2, '0')}</div>
                <strong>${escHtml(node.title || node.key || '-')}</strong>
                <p>${escHtml(systemAuditShortText(node.summary || node.impact || '-', 180))}</p>
                <div class="system-audit-node-flow">
                    <span>上游 ${escHtml(upstream.length ? upstream.map(systemAuditNodeLabel).join('、') : '无')}</span>
                    <span>下游 ${escHtml(downstream.length ? downstream.map(systemAuditNodeLabel).join('、') : '无')}</span>
                </div>
                <div class="system-audit-node-checks">
                    ${checks.map(item => `<i>${escHtml(item)}</i>`).join('') || '<i>暂无检查项</i>'}
                </div>
            </article>`;
    }).join('');
}

function renderSystemAudit() {
    const data = state.systemAuditStatus || {};
    const updated = document.getElementById('system-audit-updated');
    const overview = document.getElementById('system-audit-overview');
    if (updated) updated.textContent = data.checked_at ? toBeijingTime(data.checked_at) : '等待巡检';
    if (!overview) return;
    if (!Object.keys(data).length) {
        overview.innerHTML = '<div class="analysis-empty">等待系统巡检结果...</div>'; 
        renderSystemAuditCards([]);
        renderSystemAuditNodes([]);
        renderSystemAuditRootCauses([]);
        renderSystemAuditIssueLedger(null);
        return;
    }
    overview.innerHTML = systemAuditOverviewHtml(data);
    renderSystemAuditIssueLedger(data.issue_ledger);
    renderSystemAuditCards(data.cards);
    renderSystemAuditNodes(data.nodes);
    renderSystemAuditRootCauses(data.root_causes);
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

function serverMonitorGpuSummary(gpuPayload) {
    const gpus = Array.isArray(gpuPayload?.gpus) ? gpuPayload.gpus : [];
    const rows = gpus.filter(gpu => Number(gpu?.memory_total_mb || 0) > 0);
    if (!rows.length) {
        return {
            available: false,
            count: 0,
            name: '',
            memory_used_mb: 0,
            memory_total_mb: 0,
            memory_used_pct: 0,
            utilization_pct: null,
            detail: gpuPayload?.error || 'nvidia-smi 未返回 GPU',
        };
    }
    const memoryUsedMb = rows.reduce((sum, gpu) => sum + Number(gpu.memory_used_mb || 0), 0);
    const memoryTotalMb = rows.reduce((sum, gpu) => sum + Number(gpu.memory_total_mb || 0), 0);
    const utilizationValues = rows
        .map(gpu => Number(gpu.utilization_pct))
        .filter(value => Number.isFinite(value));
    const utilizationPct = utilizationValues.length
        ? utilizationValues.reduce((sum, value) => sum + value, 0) / utilizationValues.length
        : null;
    const hottest = rows.reduce((best, gpu) => (
        Number(gpu.temperature_c || 0) > Number(best.temperature_c || 0) ? gpu : best
    ), rows[0]);
    const powerW = rows.reduce((sum, gpu) => sum + Number(gpu.power_w || 0), 0);
    const names = Array.from(new Set(rows.map(gpu => String(gpu.name || '').trim()).filter(Boolean)));
    return {
        available: true,
        count: rows.length,
        name: names.length === 1 ? names[0] : `${names[0] || 'GPU'} 等 ${rows.length} 张`,
        memory_used_mb: memoryUsedMb,
        memory_total_mb: memoryTotalMb,
        memory_used_pct: memoryTotalMb ? (memoryUsedMb / memoryTotalMb * 100) : 0,
        utilization_pct: utilizationPct,
        detail: `${rows.length} 张 GPU · 最高 ${monitorNumber(hottest.temperature_c, 0)}°C · 总功耗 ${monitorNumber(powerW, 0)}W`,
    };
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
    const liveGpuPayload = data.gpu || {};
    const phase3GpuPayload = data.phase3_model_server_gpu || {};
    const liveGpuRows = Array.isArray(liveGpuPayload?.gpus) ? liveGpuPayload.gpus : [];
    const gpuSummary = serverMonitorGpuSummary(liveGpuRows.length ? liveGpuPayload : phase3GpuPayload);
    const disks = data.disks || [];
    const mainDisk = disks.find(d => d.path === '/data') || disks[0] || {};
    const gpuMemPct = Number(gpuSummary.memory_used_pct || 0);
    overview.innerHTML = [
        monitorMetric('CPU 使用率', `${monitorNumber(cpu.usage_pct)}%`, `${Number(cpu.cores || 0)} 核 · 负载 ${monitorNumber(cpu.load_1m)}/${monitorNumber(cpu.load_5m)}/${monitorNumber(cpu.load_15m)}`, cpu.usage_pct),
        monitorMetric('内存使用', `${monitorNumber(memory.used_pct)}%`, `${monitorNumber(memory.used_mb / 1024)} / ${monitorNumber(memory.total_mb / 1024)} GB`, memory.used_pct),
        monitorMetric('GPU 使用率', gpuSummary.available ? `${monitorNumber(gpuSummary.utilization_pct)}%` : '未检测到', gpuSummary.available ? `${gpuSummary.name} · ${gpuSummary.detail}` : gpuSummary.detail, gpuSummary.available ? gpuSummary.utilization_pct : null),
        monitorMetric('显存占用', gpuSummary.available ? `${monitorNumber(gpuMemPct)}%` : '未检测到', gpuSummary.available ? `${monitorNumber(gpuSummary.memory_used_mb / 1024)} / ${monitorNumber(gpuSummary.memory_total_mb / 1024)} GB · ${gpuSummary.count} 卡汇总` : '', gpuSummary.available ? gpuMemPct : null),
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
                <strong>${escHtml(runtimeEndpointStatusLabel(item, { model: true }))}</strong>
            </div>
        `).join('')
        : '<div style="color:var(--text-muted);font-size:11px;">未配置平台侧模型端点。</div>';
    const childHtml = childRows.length
        ? childRows.map(([name, item]) => `
            <div class="server-monitor-process">
                <span>${escHtml(name)} · ${escHtml(item.path || '-')}<br><em>${escHtml(runtimeEndpointSummary(item) || item.error || '-')}</em></span>
                <strong>${escHtml(runtimeEndpointStatusLabel(item))}</strong>
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
<div>训练模型：${escHtml(platformTools.model_bundle_available ? '已就绪' : '缺少模型产物')}</div>
                ${platformTools.status && platformTools.status.error ? `<div style="color:var(--red);">状态接口：${escHtml(platformTools.status.error)}</div>` : ''}
                <div class="server-monitor-process-list">${childHtml}</div>
            </div>
        </div>`;
}

function runtimeStatusBadge(ok) {
    return `<span class="status-badge ${ok ? 'status-live' : 'status-paused'}">${ok ? '运行中' : '异常'}</span>`;
}

function runtimeEndpointStatusLabel(item, options = {}) {
    if (!item || typeof item !== 'object') return '未返回';
    if (item.available) return '正常';
    if (options.model && item.model_mismatch) return '模型路由不匹配';
    const statusCode = Number(item.status_code || 0);
    const category = String(item.status_category || '').toLowerCase();
    if (category === 'auth_failed' || statusCode === 401) return '认证失败';
    if (category === 'auth_forbidden' || statusCode === 403) return '权限拒绝';
    if (category === 'not_found' || statusCode === 404) return '路径不存在';
    if (category === 'server_error' || statusCode >= 500) return '服务异常/启动中';
    if (item.endpoint_ok || item.endpoint_available || item.ok) {
        if (options.model && item.model_available === false) return '模型不匹配';
        if (item.model_bundle_available === false) return '模型未就绪';
        return '业务未通过';
    }
    if (category === 'network_error' || statusCode === 0) return '不可达';
    if (statusCode > 0) return `HTTP ${statusCode}`;
    return '不可达';
}

function runtimeEndpointSummary(health) {
    if (!health || typeof health !== 'object') return '';
    const status = Number(health.status_code || 0);
    const latency = Number(health.latency_ms);
    const parts = [];
    parts.push(status ? `HTTP ${status}` : (health.ok ? 'HTTP 正常' : 'HTTP 未连接'));
    if (Number.isFinite(latency)) parts.push(`${monitorNumber(latency, 0)} ms`);
    if (health.error) parts.push(dashboardReasonText(health.error));
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
    const platformTools = platformRuntime.local_ai_tools || {};
    const platformToolChildren = platformTools.child_endpoints || {};
    const platformToolChildEntries = Object.entries(platformToolChildren);
    const platformToolChildAvailable = platformToolChildEntries.filter(([, item]) => item && item.available).length;
    const platformToolContract = platformTools.tunnel_contract || {};
    const platformToolContractOk = platformToolContract.ok !== false;
    const toolsAvailable = Boolean(
        platformToolContractOk && (tools.available || platformTools.available || platformToolChildAvailable > 0)
    );
    const toolsModels = tools.models || platformTools.models || {};
    const toolsStatusLine = runtimeEndpointSummary(tools.status_health);
    const toolsHealthLine = runtimeEndpointSummary(tools.health);
    const platformModels = Array.isArray(platformRuntime.ai_models) ? platformRuntime.ai_models : [];
    const MODEL_PUBLIC_ENDPOINTS = {
        'qwen3-14b-trade': 'platform loopback 18000',
        'deepseek-r1-14b-risk': 'platform loopback 18002',
        'BB-FinQuant-Expert-14B': 'platform loopback 18003',
        phase3_quant_api: 'platform loopback 18001',
    };
    const platformModelPublicUrl = (modelId, fallbackPort = '') => {
        return MODEL_PUBLIC_ENDPOINTS[modelId] || fallbackPort || 'platform loopback only';
    };
    const configuredOrPublicModelEndpoint = (modelId, configuredBaseValue = '', fallbackPort = '') => {
        const configuredBase = String(configuredBaseValue || '').trim().replace(/\/$/, '');
        if (
            !configuredBase
            || configuredBase.includes('127.0.0.1')
            || configuredBase.includes('localhost')
            || configuredBase.includes(':18000')
            || configuredBase.includes(':18002')
            || configuredBase.includes(':18003')
        ) {
            return platformModelPublicUrl(modelId, fallbackPort);
        }
        return configuredBase;
    };
    const localToolsPublicUrl = () => {
        return MODEL_PUBLIC_ENDPOINTS.phase3_quant_api;
    };
    const vllmRows = vllmEndpoints.length ? vllmEndpoints : [vllm];
    const vllmInstanceCards = vllmRows.map(item => {
        const label = item.label || item.provider_model || 'vLLM';
        const targetModel = item.provider_model || item.model || '';
        const healthLine = runtimeEndpointSummary(item.health);
        const modelNames = Array.isArray(item.models) && item.models.length ? item.models.join('、') : '未返回模型名';
        const mismatchLine = item.model_mismatch
            ? `<div style="color:var(--red);">模型路由不匹配：目标 ${escHtml(targetModel || '-')}，实际返回 ${escHtml(modelNames)}</div>`
            : '';
        return `
            <div class="server-monitor-runtime-card">
                <strong>${escHtml(label)} / vLLM ${runtimeStatusBadge(item.available)}</strong>
                <div>内网地址：${escHtml(item.endpoint || '-')}</div>
                <div>外网地址：${escHtml(configuredOrPublicModelEndpoint(targetModel, item.api_base || item.endpoint))}</div>
                <div>状态：${escHtml(item.status || '-')}${healthLine ? ` · ${escHtml(healthLine)}` : ''}</div>
                <div>配置模型：${escHtml(targetModel || '-')}</div>
                <div>模型：${escHtml(modelNames)}</div>
                ${mismatchLine}
                ${item.error ? `<div style="color:var(--red);">错误：${escHtml(dashboardReasonText(item.error))}</div>` : ''}
            </div>`;
    }).join('');
    const vllmEndpointRows = vllmEndpoints.length
        ? `<div class="server-monitor-process-list">${vllmEndpoints.map(item => {
            const endpointModels = Array.isArray(item.models) && item.models.length ? item.models.join('、') : '未返回模型名';
            const healthLine = runtimeEndpointSummary(item.health);
            const targetModel = item.provider_model || item.model || '-';
            const publicEndpoint = configuredOrPublicModelEndpoint(
                targetModel,
                item.api_base || item.endpoint
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
                <strong>${escHtml(runtimeEndpointStatusLabel(item, { model: true }))}</strong>
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
            ${vllmInstanceCards}
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
                ${tools.error ? `<div style="color:var(--red);">错误：${escHtml(dashboardReasonText(tools.error))}</div>` : ''}
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
<div>训练模型：${escHtml(platformTools.model_bundle_available ? '已就绪' : '缺少模型产物')}</div>
                ${platformTools.status && platformTools.status.error ? `<div style="color:var(--red);">状态接口：${escHtml(platformTools.status.error)}</div>` : ''}
                ${platformToolChildEntries.length ? `<div class="server-monitor-process-list">${platformToolChildEntries.map(([name, item]) => `
                    <div class="server-monitor-process">
                        <span>${escHtml(name)} · ${escHtml(item.path || '-')}</span>
                        <strong>${escHtml(runtimeEndpointStatusLabel(item))}</strong>
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
function compactIdentifier(value, maxLength = 20) {
    const text = String(value ?? '').trim() || '-';
    if (text === '-' || text.length <= maxLength) return text;
    const suffixLength = 5;
    const prefixLength = Math.max(6, maxLength - suffixLength - 3);
    return `${text.slice(0, prefixLength)}...${text.slice(-suffixLength)}`;
}
function identifierCell(value, className, maxLength = 20) {
    const text = String(value ?? '').trim() || '-';
    return `<span class="${className}" title="${escHtml(text)}">${escHtml(compactIdentifier(text, maxLength))}</span>`;
}
function fmtSecondsLabel(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds <= 0) return '-';
    const rounded = Math.round(seconds * 10) / 10;
    return `${rounded}${'\u79d2'}`;
}
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
    if (text.includes('exchange close-fill lookup timed out')) {
        return 'OKX 平仓成交回报查询超时：本轮不会估算平仓，系统会等待下一轮拿到真实成交回报后再补记。';
    }
    if (text.includes('exchange protection order map timed out')) {
        return 'OKX 保护单列表同步超时：本轮先使用本地止盈止损记录；若连续出现，请检查 OKX 条件单接口响应。';
    }
    if (text.includes('reconciliation timed out')) {
        return 'OKX 仓位对账整轮超时：本轮已继续使用本地持仓快照；若连续出现，请检查 OKX 网络、API Key 权限或历史成交接口响应。';
    }
    if (text.includes('position analysis round cancelled by hard watchdog')) {
        return '持仓复盘整轮超时：本轮已被保护性中断，通常是 OKX 同步、行情刷新或持仓复盘阶段累计过慢；系统会进入下一轮继续处理。';
    }
    if (text.includes('market analysis round cancelled by hard watchdog')) {
        return '市场分析曾被旧版整轮保护取消；当前版本会按具体阶段预算降级。请刷新后查看最新阶段耗时。';
    }
    if (text.includes('market analysis task cancelled during')) {
        return '市场分析任务被外部取消；请查看当前阶段和耗时拆解，系统不会把它误记为模型或 OKX 超时。';
    }
    if (text.includes('position analysis task cancelled during')) {
        return '持仓复盘任务被外部取消；请查看当前阶段和耗时拆解，系统不会把它误记为 OKX 或模型超时。';
    }
    if (text.includes('Invalid OK-ACCESS-KEY') || text.includes('50111')) {
        return 'OKX API Key 无效，余额/仓位同步可能失败，请检查当前模式的 OKX Key 配置。';
    }
    return text;
}

function loopErrorScopeLabel(stats) {
    const marketErr = loopErrorLabel(stats?.market_last_error);
    const positionErr = loopErrorLabel(stats?.position_last_error);
    const lastErr = loopErrorLabel(stats?.last_round_error);
    if (marketErr) return `市场分析线程：${marketErr}`;
    if (positionErr) return `持仓复盘线程：${positionErr}`;
    return lastErr;
}

function okxAuthoritativeSyncLabel(stats) {
    const sync = stats?.okx_authoritative_sync || {};
    const status = String(sync.status || 'pending').toLowerCase();
    const lastSuccess = sync.last_success_at ? shortBeijingTime(sync.last_success_at) : '';
    const lastFailure = sync.last_failure_at ? shortBeijingTime(sync.last_failure_at) : '';
    const error = loopErrorLabel(sync.last_error);
    const attention = Number(sync.last_requires_attention_count || 0);
    if (status === 'ok') {
        return `OKX权威事实同步正常 · 最近成功 ${lastSuccess || '-'}${attention > 0 ? ` · 待核对 ${attention}` : ''}`;
    }
    if (status === 'stale') {
        const age = fmtSecondsLabel(sync.last_success_age_seconds);
        return `OKX权威事实同步过期 · 最近成功 ${lastSuccess || '-'} · 已 ${age}，暂停新开仓`;
    }
    if (status === 'warning') {
        return `OKX权威事实同步异常 · 最近失败 ${lastFailure || '-'}${error ? ' · ' + error : ''} · 暂停新开仓`;
    }
    return `等待OKX权威事实同步 · 间隔 ${fmtSecondsLabel(sync.interval_seconds)}`;
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
    return stageLabelText(stage, '', stats?.running);
}

function stageLabelText(stageValue, fallbackLabel = '', running = true) {
    const stage = String(stageValue || '');
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
        watchdog_cancelled: '\u672c\u8f6e\u8d85\u65f6\u5df2\u4e2d\u65ad',
    };
    if (labels[stage]) return labels[stage];
    if (stage.startsWith('analyze:')) {
        return `\u6b63\u5728\u5206\u6790 ${stage.split(':').slice(1).join(':')}`;
    }
    if (stage.startsWith('execute:')) {
        return `\u6b63\u5728\u6267\u884c ${stage.split(':').slice(1).join(':')} \u8ba2\u5355`;
    }
    if (stage.startsWith('market_ai:')) {
        return `市场扫描：正在分析 ${stage.split(':').slice(1).join(':')}`;
    }
    if (stage.startsWith('strategy_context:')) {
        const contextLabels = {
            daily_perf: '读取今日绩效',
            today_side_perf: '读取今日多空表现',
            multiday_side_perf: '读取多日多空表现',
            symbol_side_perf: '读取币种方向表现',
            model_contribution_perf: '读取模型贡献表现',
            open_positions: '读取当前持仓',
            account_equity: '读取账户权益',
            learning: '刷新策略学习上下文',
        };
        const key = stage.split(':').slice(1).join(':');
        return contextLabels[key] || `构建策略上下文：${key}`;
    }
    return cleanStatusText(
        fallbackLabel,
        running ? '\u7b49\u5f85\u4e0b\u4e00\u8f6e\u5206\u6790' : '\u670d\u52a1\u672a\u8fd0\u884c'
    );
}

function scopedStageText(stats, scope) {
    const isMarket = scope === 'market';
    const active = Boolean(isMarket ? stats?.market_round_active : stats?.position_round_active);
    const stage = isMarket ? stats?.market_current_stage : stats?.position_current_stage;
    const err = loopErrorLabel(isMarket ? stats?.market_last_error : stats?.position_last_error);
    if (err) return `异常：${err}`;
    const label = stageLabelText(stage, '', stats?.running !== false);
    return active ? `运行中：${label}` : label;
}

function updateAutoStatus(stats) {
    const marketScanEl = document.getElementById('status-scan-mode');
    if (marketScanEl) {
        marketScanEl.textContent = state.paused
            ? '已暂停新市场分析；已有仓位继续复盘 / 风控 / 平仓'
            : '\u81ea\u52a8\u626b\u63cf\u5168\u5e02\u573a (OKX)';
    }

    const modelCountEl = document.getElementById('status-model-count');
    if (modelCountEl) {
        const expertCount = state.aiExpertModels.length || FIXED_AI_EXPERT_FALLBACKS.length;
        modelCountEl.textContent = `${expertCount} / 1`;
    }

    if (stats && stats.decision_interval) {
        const interval = Number(stats.decision_interval);
        state.decisionInterval = Number.isFinite(interval) && interval > 0 ? interval : null;
    }

    const intervalEl = document.getElementById('status-interval');
    if (intervalEl) {
        const marketInterval = stats?.market_loop_interval_seconds;
        const positionInterval = stats?.position_loop_interval_seconds;
        intervalEl.textContent = state.decisionInterval
            ? `配置${fmtSecondsLabel(state.decisionInterval)} / 市场${fmtSecondsLabel(marketInterval)} / 持仓${fmtSecondsLabel(positionInterval)}`
            : '\u8bfb\u53d6\u4e2d';
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

    const marketStageEl = document.getElementById('status-market-stage');
    if (marketStageEl) {
        marketStageEl.textContent = scopedStageText(stats, 'market');
    }

    const positionStageEl = document.getElementById('status-position-stage');
    if (positionStageEl) {
        positionStageEl.textContent = scopedStageText(stats, 'position');
    }

    const timingEl = document.getElementById('status-round-timing');
    if (timingEl) {
        const started = stats?.last_round_started_at ? shortBeijingTime(stats.last_round_started_at) : '-';
        const finished = stats?.last_round_finished_at
            ? shortBeijingTime(stats.last_round_finished_at)
            : '\u8fdb\u884c\u4e2d';
        timingEl.textContent = `\u5f00\u59cb ${started} / \u5b8c\u6210 ${finished}`;
    }

    const okxSyncEl = document.getElementById('status-okx-sync');
    if (okxSyncEl) {
        okxSyncEl.textContent = okxAuthoritativeSyncLabel(stats);
    }

    const errRow = document.getElementById('status-loop-error-row');
    const errEl = document.getElementById('status-loop-error');
    if (errRow && errEl) {
        const err = loopErrorScopeLabel(stats);
        errRow.style.display = err ? 'flex' : 'none';
        errEl.textContent = err || '-';
    }
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
    const body = { account_name: accountName };
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
        const configured = m.configured === true || Boolean(m.api_key);
        const keyState = configured
            ? `<span style="color:var(--green);font-size:11px;">${m.configuration_type === 'keyless_loopback' ? '已设置（本地免 Key）' : '已设置'}</span>`
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
    const data = await fetchLatestPageJSON(
        'positions',
        `/api/dashboard/positions?mode=${state.mode}&page=${state.positionsPage}&page_size=${PAGE_SIZE}&open_only=true`,
    );
    if (!data) return;
    state.positionsPage = data.page || state.positionsPage;
    state.positionsTotal = data.total || 0;
    state.openPositions = data.positions || [];
    state.protectionInventory = data.protection_inventory || null;
    renderOpenPositionsTable(data.positions || [], state.positionsPage, data.total_pages || 1, data.total || 0);
    const badge = document.getElementById('position-badge');
    if (badge) {
        const total = Number(data.total ?? data.count ?? 0);
        badge.textContent = total;
        badge.style.display = total > 0 ? '' : 'none';
    }
}

function positionProtectionInventoryWarnings(inventory) {
    const missing = Array.isArray(inventory.missing_keys) ? inventory.missing_keys : [];
    const orphan = Array.isArray(inventory.orphan_keys) ? inventory.orphan_keys : [];
    const mismatch = Array.isArray(inventory.coverage_mismatches) ? inventory.coverage_mismatches : [];
    const repairBlockers = Array.isArray(inventory.repair_blockers) ? inventory.repair_blockers : [];
    const invalidCount = mlOptionalNumber(inventory.invalid_order_count);
    return [...new Set([
        ...missing.map(key => `缺失 ${Array.isArray(key) ? key.join(' ') : key}`),
        ...orphan.map(key => `孤儿 ${Array.isArray(key) ? key.join(' ') : key}`),
        ...mismatch.map(item => `数量不一致 ${item.symbol || '-'} ${item.side || '-'}`),
        ...(invalidCount !== null && invalidCount > 0 ? [`无效保护单 ${invalidCount} 张`] : []),
        ...repairBlockers.map(item => `修复阻断：${dashboardReasonText(item)}`),
    ])];
}

async function fetchPositionHistory() {
    const data = await fetchLatestPageJSON(
        'position-history',
        `/api/dashboard/positions?mode=${state.mode}&page=${state.positionHistoryPage}&page_size=${PAGE_SIZE}&closed_only=true`,
    );
    if (!data) return;
    state.positionHistoryPage = data.page || state.positionHistoryPage;
    state.positionHistoryTotal = data.total || 0;
    const label = document.getElementById('position-history-mode-label');
    if (label) {
        label.textContent = `${state.mode === 'paper' ? '模拟盘' : '实盘'} · ${Number(data.settled_count || 0)} 已结算 / ${Number(data.pending_settlement_count || 0)} 待结算`;
    }
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
    tbody.innerHTML = positions.map((p, positionIndex) => {
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
        const riskEnvelope = p.risk_contract || {};
        const riskBlockers = Array.isArray(riskEnvelope.blockers) ? riskEnvelope.blockers : [];
        const protectionBlockers = Array.isArray(p.protection_contract?.blockers) ? p.protection_contract.blockers : [];
        const management = p.current_management_contract || {};
        const managementBlockers = Array.isArray(management.blockers) ? management.blockers : [];
        const evidenceReady = management.management_eligible === true
            && p.protection_contract?.available === true
            && !managementBlockers.length
            && !protectionBlockers.length;
        const evidenceLabel = evidenceReady
            ? (riskEnvelope.current_management_authoritative === true && riskEnvelope.available !== true
                ? '接管/OCO 完整'
                : '风险/OCO 完整')
            : '存在证据阻断';
        return `
        <tr>
            <td>${escHtml(p.symbol || '-')}</td>
            <td><span style="color:${p.side === 'long' ? 'var(--green)' : 'var(--red)'}">${sideLabel(p.side)}</span></td>
            <td>${Number(p.leverage || 1).toFixed(1)}x</td>
            <td>${fmtNum(p.quantity)}${quantityMeta}</td>
            <td>${fmtPrice(p.entry_price)}</td>
            <td>${fmtPrice(p.current_price || p.entry_price)}</td>
            <td>
                <button
                    type="button"
                    class="position-pnl-link js-open-position-pnl"
                    data-position-index="${positionIndex}"
                    aria-label="查看 ${escHtml(p.symbol || '-')} 盈亏明细"
                    title="查看盈亏、资金费和手续费"
                ><span style="color:${pnlColor};">${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)}</span></button>
            </td>
            <td>${p.take_profit ? fmtPrice(p.take_profit) : '-'}</td>
            <td>${p.stop_loss ? fmtPrice(p.stop_loss) : '-'}</td>
            <td style="font-size:10px;color:var(--text-muted);">${toBeijingTime(p.opened_at)}</td>
            <td>
                <div class="position-action-stack">
                    <button class="btn btn-sm js-position-evidence" data-position-index="${positionIndex}">查看证据</button>
                    <span class="position-evidence-state ${evidenceReady ? 'ok' : 'warn'}">${evidenceLabel}</span>
                    <button ${closeButtonAttrs}>${closeLabel}</button>
                </div>
            </td>
        </tr>`;
    }).join('');
    renderPagination('positions-pagination', page, totalPages, totalItems, 'changePositionsPage');
}


function isOfficialClosedPositionSettlement(position) {
    if (isPendingClosedPositionSettlement(position)) return false;
    const status = String(position?.settlement_status || '').trim();
    if (['reconciled', 'settled', 'okx_position_history'].includes(status)) {
        return true;
    }
    const source = String(position?.settlement_source || position?.pnl_source || '').trim();
    return source.includes('okx_position_history') || source.includes('position_settlement_snapshot');
}

function isPendingClosedPositionSettlement(position) {
    if (position?.settlement_complete === false || position?.settlement_state === 'pending') {
        return true;
    }
    const status = String(position?.settlement_status || '').trim();
    const displayState = String(position?.settlement_display_state || '').trim();
    return ['lifecycle_open', 'identity_unresolved', 'evidence_unresolved', 'pending_authority', 'stopped_waiting']
        .includes(displayState)
        || status === 'settlement_quarantined'
        || status === 'settlement_pending'
        || status === 'settlement_unresolved';
}

function closedPositionSettlementLabel(position) {
    const displayLabel = String(position?.settlement_display_label || '').trim();
    if (displayLabel) return displayLabel;
    const state = String(position?.settlement_display_state || '').trim();
    if (state === 'lifecycle_open') return 'OKX 仓位生命周期仍开放';
    if (state === 'identity_unresolved') return 'OKX 权威仓位历史身份未确认';
    if (state === 'evidence_unresolved') return '结算证据无法守恒';
    if (state === 'stopped_waiting') return '权威结算未完成，已停止自动等待';
    return '等待 OKX 权威结算';
}

function closedPositionEvidenceLabel(position) {
    if (isPendingClosedPositionSettlement(position)) {
        return '\u5e73\u4ed3\u6210\u4ea4\u5df2\u786e\u8ba4\uff0c\u5b98\u65b9\u7ed3\u7b97\u5f85\u8865\u9f50';
    }
    if (isOfficialClosedPositionSettlement(position)) {
        return '\u5df2\u5b98\u65b9\u7ed3\u7b97';
    }
    if (position?.evidence_complete === true) {
        return '\u5b8c\u6574';
    }
    return '\u8ba2\u5355\u8865\u5168\u4e2d';
}

function renderClosedPositionsTable(positions, page = 1, totalPages = 1, totalItems = 0) {
    const tbody = document.getElementById('position-history-tbody');
    const pagination = document.getElementById('position-history-pagination');
    if (!tbody) return;
    positionLinkedOrdersByGroup.clear();
    if (!positions.length) {
        tbody.innerHTML = '<tr><td colspan="12" style="color:var(--text-muted);text-align:center;padding:24px;">\u6682\u65e0\u5386\u53f2\u6301\u4ed3\u6570\u636e</td></tr>';
        if (pagination) pagination.style.display = 'none';
        return;
    }
    tbody.innerHTML = positions.map((p, index) => {
        const settlementPending = isPendingClosedPositionSettlement(p);
        const pnl = mlOptionalNumber(p.realized_pnl);
        const pnlColor = pnl === null ? 'var(--text-muted)' : (pnl >= 0 ? 'var(--green)' : 'var(--red)');
        const pnlText = settlementPending || pnl === null
            ? '\u5f85\u5b98\u65b9\u7ed3\u7b97'
            : `${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)}`;
        const statusLabel = p.close_status_label || p.position_status || (p.close_status === 'partial' ? '\u90e8\u5206\u5e73\u4ed3' : '\u5168\u90e8\u5e73\u4ed3');
        const statusColor = p.close_status === 'partial' ? 'var(--accent-light)' : 'var(--text-muted)';
        const closeOrigin = String(p.close_origin || 'unknown');
        const closeOriginLabel = p.close_origin_label || '\u6765\u6e90\u5f85\u786e\u8ba4';
        const groupId = String(p.group_id || p.id || `row-${page}-${index}`);
        const linkedFills = Array.isArray(p.linked_fills) ? p.linked_fills : [];
        positionLinkedOrdersByGroup.set(groupId, { position: p, fills: linkedFills });
        const linkedCount = Number(p.linked_order_count ?? linkedFills.length ?? 0);
        const evidenceBadge = settlementPending
            ? `<div class="position-ledger-badge warn">${escHtml(closedPositionSettlementLabel(p))}</div>`
            : (isOfficialClosedPositionSettlement(p) || p.evidence_complete === true
                ? '<div class="position-ledger-badge ok">OKX</div>'
                : '<div class="position-ledger-badge warn">\u8ba2\u5355\u8865\u5168\u4e2d</div>');
        const linkedButtonDisabled = linkedCount <= 0 ? 'disabled' : '';
        return `
        <tr>
            <td>${escHtml(p.symbol || '-')}</td>
            <td><span style="color:${p.side === 'long' ? 'var(--green)' : 'var(--red)'}">${sideLabel(p.side)}</span></td>
            <td><span style="color:${statusColor};font-weight:600;">${escHtml(statusLabel)}</span></td>
            <td><span class="position-close-origin origin-${escHtml(closeOrigin)}">${escHtml(closeOriginLabel)}</span></td>
            <td>${Number(p.leverage || 1).toFixed(1)}x</td>
            <td>${fmtNum(p.quantity)}</td>
            <td>${fmtPrice(p.entry_price)}</td>
            <td>${fmtPrice(p.current_price || p.entry_price)}</td>
            <td>
                <button
                    type="button"
                    class="position-pnl-link js-position-pnl-detail"
                    data-group-id="${escHtml(groupId)}"
                    ${settlementPending || pnl === null ? 'disabled' : ''}
                    aria-label="查看 ${escHtml(p.symbol || '-')} 已实现盈亏明细"
                    title="查看盈亏、资金费和手续费"
                ><span style="color:${pnlColor};">${pnlText}</span></button>
            </td>
            <td style="font-size:10px;color:var(--text-muted);">${toBeijingTime(p.opened_at)}</td>
            <td style="font-size:10px;color:var(--text-muted);">${toBeijingTime(p.closed_at)}</td>
            <td>
                <button class="btn btn-sm js-position-linked-orders" data-group-id="${escHtml(groupId)}" ${linkedButtonDisabled}>\u5173\u8054\u8ba2\u5355 (${linkedCount})</button>
                ${evidenceBadge}
            </td>
        </tr>`;
    }).join('');
    renderPagination('position-history-pagination', page, totalPages, totalItems, 'changePositionHistoryPage');
}

function positionFeeTotal(position) {
    const explicitFee = mlOptionalNumber(position?.fee);
    if (explicitFee !== null) return Math.abs(explicitFee);
    return Math.abs(Number(position?.entry_fee || 0)) + Math.abs(Number(position?.close_fee || 0));
}

function positionPnlBreakdownHtml(position) {
    // Match the OKX position detail popover: show ledger components only.
    const isOpenPosition = position?.is_open === true
        || String(position?.close_status || '').toLowerCase() === 'open';
    const floatingPnl = mlOptionalNumber(position?.unrealized_pnl);
    const settledRealizedPnl = mlOptionalNumber(position?.realized_pnl);
    const closeFillPnl = mlOptionalNumber(position?.close_fill_pnl);
    const fundingFee = Number(position?.funding_fee || 0);
    const fee = positionFeeTotal(position);
    const realizedPnl = isOpenPosition && floatingPnl !== null
        ? floatingPnl + fundingFee - fee
        : settledRealizedPnl;
    const secondaryPnl = isOpenPosition ? floatingPnl : closeFillPnl;
    const secondaryLabel = isOpenPosition ? '浮动盈亏' : '平仓收益';
    const realizedText = realizedPnl === null ? '待官方结算' : `${signedMoney(realizedPnl)} USDT`;
    const secondaryText = secondaryPnl === null ? '--' : `${signedMoney(secondaryPnl)} USDT`;
    const feeText = fee > 0 ? `-${fmtMoney(fee)} USDT` : '0.00 USDT';
    return `
        <div class="position-ledger-summary position-pnl-breakdown">
            <div><span>已实现盈亏</span><strong style="color:${signedMoneyColor(realizedPnl)};">${realizedText}</strong></div>
            <div><span>${secondaryLabel}</span><strong style="color:${signedMoneyColor(secondaryPnl)};">${secondaryText}</strong></div>
            <div><span>资金费</span><strong style="color:${signedMoneyColor(fundingFee)};">${signedMoney(fundingFee)} USDT</strong></div>
            <div><span>手续费</span><strong style="color:${fee > 0 ? 'var(--red)' : 'var(--text-muted)'};">${feeText}</strong></div>
        </div>`;
}

let openPositionPnlPopoverAnchor = null;

function positionPnlPopoverPosition(anchor, popover) {
    const rect = anchor?.getBoundingClientRect?.();
    const width = popover.offsetWidth;
    const height = popover.offsetHeight;
    const gap = 8;
    const margin = 12;
    if (!rect) {
        popover.style.left = `${Math.max(margin, (window.innerWidth - width) / 2)}px`;
        popover.style.top = `${Math.max(margin, (window.innerHeight - height) / 2)}px`;
        return;
    }
    const left = Math.min(
        Math.max(margin, rect.left + (rect.width - width) / 2),
        Math.max(margin, window.innerWidth - width - margin),
    );
    let top = rect.top - height - gap;
    if (top < margin) top = rect.bottom + gap;
    if (top + height > window.innerHeight - margin) {
        top = Math.max(margin, window.innerHeight - height - margin);
    }
    popover.style.left = `${Math.round(left)}px`;
    popover.style.top = `${Math.round(top)}px`;
}

function closePositionPnlPopover() {
    const popover = document.getElementById('position-pnl-popover');
    if (!popover) return;
    popover.hidden = true;
    popover.setAttribute('aria-hidden', 'true');
    popover.classList.remove('is-open');
    openPositionPnlPopoverAnchor = null;
}

function openPositionPnlPopover(positionIndex, anchor = null) {
    const position = (state.openPositions || [])[Number(positionIndex)];
    if (!position) return;
    const popover = document.getElementById('position-pnl-popover');
    const title = document.getElementById('position-pnl-popover-title');
    const body = document.getElementById('position-pnl-popover-body');
    if (!popover || !body) return;
    if (title) title.textContent = `${position.symbol || '-'} ${sideLabel(position.side)} · 盈亏明细`;
    body.innerHTML = positionPnlBreakdownHtml(position);
    openPositionPnlPopoverAnchor = anchor;
    popover.hidden = false;
    popover.setAttribute('aria-hidden', 'false');
    popover.classList.add('is-open');
    requestAnimationFrame(() => positionPnlPopoverPosition(openPositionPnlPopoverAnchor, popover));
}

function positionEvidenceValue(value, suffix = '') {
    const number = mlOptionalNumber(value);
    return number === null ? '证据缺失' : `${number.toFixed(4)}${suffix}`;
}

function positionEvidenceBlockers(blockers) {
    const values = Array.isArray(blockers)
        ? blockers.flatMap(item => String(item || '').split(',')).map(item => item.trim()).filter(Boolean)
        : [];
    return values.length
        ? `<div class="position-evidence-blockers">阻断：${values.map(item => escHtml(dashboardReasonText(item))).join(' / ')}</div>`
        : '';
}

function openPositionEvidenceModal(positionIndex) {
    const position = (state.openPositions || [])[Number(positionIndex)];
    if (!position) return;
    const riskEnvelope = position.risk_contract || {};
    const contracts = Array.isArray(riskEnvelope.contracts) ? riskEnvelope.contracts : [];
    const management = position.current_management_contract || {};
    const managementBlockers = Array.isArray(management.blockers) ? management.blockers : [];
    const protection = position.protection_contract || {};
    const protectionOrders = Array.isArray(protection.orders) ? protection.orders : [];
    const inventory = state.protectionInventory || {};
    const historicalEntryArchived = riskEnvelope.historical_entry_incomplete === true
        && riskEnvelope.current_management_authoritative === true;
    const title = document.getElementById('position-linked-orders-modal-title');
    const body = document.getElementById('position-linked-orders-modal-body');
    const overlay = document.getElementById('position-linked-orders-modal-overlay');
    if (!body || !overlay) return;
    if (title) title.textContent = `${position.symbol || '-'} ${sideLabel(position.side)} · 风险与 OCO 证据`;

    const managementActions = Array.isArray(management.allowed_actions)
        ? management.allowed_actions : [];
    const fundingGaps = Array.isArray(management.funding_evidence_gaps)
        ? management.funding_evidence_gaps : [];
    const fundingBillCount = Number(management.funding_bill_count || 0);
    const fundingInclusionLabel = fundingBillCount <= 0
        ? '\u65e0\u8d26\u5355\uff0c\u6309 0 \u5904\u7406'
        : management.funding_evidence_eligible === true
            ? '\u5df2\u7eb3\u5165\u9000\u51fa\u5224\u65ad'
            : '\u4ec5\u5c55\u793a\uff0c\u672a\u7eb3\u5165\u9000\u51fa\u5224\u65ad';
    const fundingHtml = `
        <div class="position-evidence-grid">
            <div><span>\u7d2f\u8ba1\u8d44\u91d1\u8d39</span><strong>${positionEvidenceValue(management.funding_fee_usdt, ' U')}</strong></div>
            <div><span>\u8d44\u91d1\u8d39\u8d26\u5355</span><strong>${fundingBillCount}</strong></div>
            <div><span>\u8d44\u91d1\u8d39\u9000\u51fa\u53e3\u5f84</span><strong>${fundingInclusionLabel}</strong></div>
            <div><span>\u8d44\u91d1\u8d39\u6765\u6e90</span><strong>${escHtml(management.funding_fee_source || 'none')}</strong></div>
        </div>`;
    const managementHtml = management.contract_version
        ? `
            <article class="position-evidence-contract">
                <div class="position-evidence-contract-head">
                    <strong>当前仓位接管合同</strong>
                    <span class="position-evidence-state ${management.management_eligible ? 'ok' : 'warn'}">${management.management_eligible ? '当前管理有效' : '当前管理阻断'}</span>
                </div>
                <div class="position-evidence-grid">
                    <div><span>实际入场手续费</span><strong>${positionEvidenceValue(management.entry_fee_usdt, ' U')}</strong></div>
                    <div><span>当前压力损失</span><strong>${positionEvidenceValue(management.position_stressed_loss_usdt, ' U')}</strong></div>
                    <div><span>组合压力损失</span><strong>${positionEvidenceValue(management.portfolio_stressed_loss_usdt, ' U')}</strong></div>
                    <div><span>组合总 notional</span><strong>${positionEvidenceValue(management.portfolio_gross_notional_usdt, ' U')}</strong></div>
                    <div><span>账户权益</span><strong>${positionEvidenceValue(management.account_equity_usdt, ' U')}</strong></div>
                    <div><span>组合集中压力</span><strong>${distributionProbabilityLabel(management.portfolio_concentration_pressure)}</strong></div>
                    <div><span>当前止损</span><strong>${management.stop_loss_price == null ? '证据缺失' : fmtPrice(management.stop_loss_price)}</strong></div>
                    <div><span>当前止盈</span><strong>${management.take_profit_price == null ? '证据缺失' : fmtPrice(management.take_profit_price)}</strong></div>
                    <div><span>允许扩大仓位</span><strong>${management.can_expand_position === false ? '禁止' : '证据缺失'}</strong></div>
                    <div><span>允许提高杠杆</span><strong>${management.can_increase_leverage === false ? '禁止' : '证据缺失'}</strong></div>
                    <div><span>历史入场状态</span><strong>${escHtml(dashboardReasonText(management.original_entry_contract_status || 'unavailable'))}</strong></div>
                    <div><span>允许动作</span><strong>${managementActions.length ? managementActions.map(item => escHtml(({ hold: '保持', reduce: '减仓', close: '平仓', protection_repair: '修复保护单' })[item] || dashboardReasonText(item))).join(' / ') : '未授权任何动作'}</strong></div>
                </div>
                <div class="position-evidence-provenance"><span>版本 ${escHtml(management.contract_version || '缺失')}</span><span>接管时间 ${escHtml(management.takeover_at || '缺失')}</span><span>指纹 ${escHtml(management.policy_provenance?.contract_fingerprint || '缺失')}</span></div>
                ${fundingHtml}
                ${positionEvidenceBlockers([...managementBlockers, ...fundingGaps])}
            </article>`
        : '<div class="position-evidence-empty">当前仓位尚未生成只减仓接管合同。</div>';

    const archivedDecisionIds = contracts
        .map(contract => contract.decision_id)
        .filter(Boolean);
    const riskHtml = historicalEntryArchived
        ? `<div class="position-evidence-empty">该持仓创建于独立风险合同上线前，旧版入场证据已归档${archivedDecisionIds.length ? `（历史决策 #${archivedDecisionIds.map(item => escHtml(item)).join(' / #')}）` : ''}；当前只减仓接管合同与 OKX 保护证据继续作为有效管理依据。</div>`
        : contracts.length
            ? contracts.map(contract => {
            const portfolio = contract.portfolio_risk_snapshot || {};
            const adjustments = Array.isArray(contract.adjustment_reasons)
                ? contract.adjustment_reasons : [];
            return `
                <article class="position-evidence-contract">
                    <div class="position-evidence-contract-head">
                        <strong>独立风险合同 · 决策 #${escHtml(contract.decision_id || '-')}</strong>
                        <span class="position-evidence-state ${contract.production_eligible ? 'ok' : 'warn'}">${contract.production_eligible ? '执行时有效' : '合同未通过'}</span>
                    </div>
                    <div class="position-evidence-grid">
                        <div><span>独立风险预算</span><strong>${positionEvidenceValue(contract.risk_budget_usdt, ' U')}</strong></div>
                        <div><span>计划压力损失</span><strong>${positionEvidenceValue(contract.planned_stressed_loss_usdt, ' U')}</strong></div>
                        <div><span>压力损失距离</span><strong>${distributionProbabilityLabel(contract.stressed_loss_fraction)}</strong></div>
                        <div><span>目标 notional</span><strong>${positionEvidenceValue(contract.target_notional_usdt, ' U')}</strong></div>
                        <div><span>最终 notional</span><strong>${positionEvidenceValue(contract.final_notional_usdt, ' U')}</strong></div>
                        <div><span>预期费后收益</span><strong>${distributionPctLabel(contract.expected_net_return_pct)}</strong></div>
                        <div><span>组合风险预算</span><strong>${positionEvidenceValue(contract.portfolio_risk_budget_usdt, ' U')}</strong></div>
                        <div><span>组合已占压力损失</span><strong>${positionEvidenceValue(contract.current_portfolio_stressed_loss_usdt ?? portfolio.current_stressed_loss_usdt, ' U')}</strong></div>
                        <div><span>组合剩余风险预算</span><strong>${positionEvidenceValue(contract.remaining_portfolio_risk_budget_usdt, ' U')}</strong></div>
                        <div><span>组合总 notional</span><strong>${positionEvidenceValue(portfolio.gross_notional_usdt, ' U')}</strong></div>
                        <div><span>同方向 notional</span><strong>${positionEvidenceValue(portfolio.same_side_notional_usdt, ' U')}</strong></div>
                        <div><span>方向集中度</span><strong>${distributionProbabilityLabel(portfolio.direction_concentration)}</strong></div>
                    </div>
                    <div class="position-evidence-reasons"><b>仓位调整结果</b><span>${adjustments.length ? adjustments.map(item => escHtml(dashboardReasonText(item))).join(' / ') : '目标与最终仓位一致，无需调整'}</span></div>
                    <div class="position-evidence-provenance"><span>版本 ${escHtml(contract.contract_version || '缺失')}</span><span>指纹 ${escHtml(contract.policy_provenance?.contract_fingerprint || '缺失')}</span></div>
                    ${positionEvidenceBlockers(contract.blockers)}
                </article>`;
            }).join('')
            : `<div class="position-evidence-empty">无法追溯入场订单对应的独立风险合同。${positionEvidenceBlockers(riskEnvelope.blockers)}</div>`;

    const protectionHtml = protectionOrders.length
        ? protectionOrders.map(order => `
            <article class="position-protection-order">
                <div><span>OKX algo ID</span><strong>${escHtml(order.algo_id || '证据缺失')}</strong></div>
                <div><span>状态</span><strong>${escHtml(order.state || '证据缺失')}</strong></div>
                <div><span>止损触发价</span><strong>${order.stop_loss_price == null ? '证据缺失' : fmtPrice(order.stop_loss_price)}</strong></div>
                <div><span>止盈触发价</span><strong>${order.take_profit_price == null ? '证据缺失' : fmtPrice(order.take_profit_price)}</strong></div>
                <div><span>移动触发价</span><strong>${order.trigger_price == null ? '未设置' : fmtPrice(order.trigger_price)}</strong></div>
                <div><span>保护合约数</span><strong>${positionEvidenceValue(order.contracts)}</strong></div>
                <div><span>关联订单 ID</span><strong>${escHtml(order.linked_order_id || 'OKX 原生保护单（无需本地关联）')}</strong></div>
                <div><span>唯一性</span><strong>${protection.unique ? '唯一一张' : protection.split_coverage ? `精确分片 ${protection.order_count ?? protectionOrders.length} 张` : `异常多张 ${protection.order_count ?? protectionOrders.length} 张`}</strong></div>
            </article>`).join('')
        : '<div class="position-evidence-empty">没有加载到该持仓的 OKX 原生 OCO/止盈止损保护单。</div>';

    const inventoryWarnings = positionProtectionInventoryWarnings(inventory);
    body.innerHTML = `
        <div class="position-evidence-summary">
            <div><strong>${escHtml(position.symbol || '-')}</strong><span>OKX 当前持仓</span></div>
            <div><strong>${positionEvidenceValue(position.quantity)}</strong><span>持仓数量</span></div>
            <div><strong>${positionEvidenceValue(position.unrealized_pnl, ' U')}</strong><span>当前浮盈亏</span></div>
            <div><strong>${management.management_eligible === true ? '有效' : '阻断'}</strong><span>当前接管</span></div>
            <div><strong>${riskEnvelope.available === true ? '完整' : historicalEntryArchived ? '旧版已归档' : '历史缺口'}</strong><span>入场合同</span></div>
            <div><strong>${protection.available === true ? `${protection.order_count ?? 0} 张` : '读取失败'}</strong><span>OKX 保护</span></div>
        </div>
        ${positionEvidenceBlockers([...(management.blockers || []), ...(protection.blockers || [])])}
        <section class="position-evidence-section"><h3>当前仓位接管与只减仓边界</h3>${managementHtml}</section>
        <section class="position-evidence-section"><h3>历史入场风险合同</h3>${riskHtml}</section>
        <section class="position-evidence-section"><h3>OKX OCO / 保护生命周期</h3>${protectionHtml}</section>
        <section class="position-evidence-section"><h3>全账户保护告警</h3>
            ${inventory.available === false
                ? positionEvidenceBlockers(inventory.blockers)
                : inventoryWarnings.length
                    ? `<div class="position-evidence-blockers">${inventoryWarnings.map(item => escHtml(item)).join(' / ')}</div>`
                    : '<div class="position-evidence-ok">当前快照未发现孤儿或多张保护方向。</div>'}
        </section>`;
    overlay.style.display = 'flex';
}

function openPositionLinkedOrdersModal(groupId) {
    const payload = positionLinkedOrdersByGroup.get(String(groupId));
    if (!payload) return;
    const position = payload.position || {};
    const fills = Array.isArray(payload.fills) ? payload.fills : [];
    const title = document.getElementById('position-linked-orders-modal-title');
    const body = document.getElementById('position-linked-orders-modal-body');
    const overlay = document.getElementById('position-linked-orders-modal-overlay');
    if (!body || !overlay) return;
    if (title) {
        title.textContent = `${position.symbol || '-'} ${sideLabel(position.side)} \u5173\u8054\u8ba2\u5355`;
    }
    const gaps = Array.isArray(position.evidence_gaps) ? position.evidence_gaps : [];
    const settlementPending = isPendingClosedPositionSettlement(position);
    const realizedPnlText = settlementPending
        ? '\u5f85\u5b98\u65b9\u7ed3\u7b97'
        : `${signedMoney(position.realized_pnl || 0)} USDT`;
    const evidenceHtml = `
        ${positionPnlBreakdownHtml(position)}
        <div class="position-ledger-summary">
            <div><strong>${escHtml(position.okx_inst_id || '-')}</strong><span>OKX instId</span></div>
            <div><strong>${fmtNum(position.closed_quantity ?? position.quantity)}</strong><span>\u5df2\u5e73\u6570\u91cf</span></div>
            <div><strong>${realizedPnlText}</strong><span>\u5df2\u5b9e\u73b0\u76c8\u4e8f</span></div>
            <div><strong>${closedPositionEvidenceLabel(position)}</strong><span>OKX \u8bc1\u636e</span></div>
        </div>
        ${settlementPending && position.settlement_explanation ? `<div class="position-ledger-gaps">${escHtml(position.settlement_explanation)}</div>` : ''}
        ${gaps.length ? `<div class="position-ledger-gaps">\u8bc1\u636e\u7f3a\u53e3\uff1a${gaps.map(item => escHtml(dashboardReasonText(item))).join(' / ')}</div>` : ''}`;
    if (!fills.length) {
        body.innerHTML = `${evidenceHtml}<div class="reason-block">\u6682\u65e0 OKX \u5173\u8054\u8ba2\u5355\u660e\u7ec6\uff0c\u8bf7\u5148\u6267\u884c\u4e09\u671f OKX \u8ba2\u5355/\u6210\u4ea4\u540c\u6b65\u3002</div>`;
        overlay.style.display = 'flex';
        return;
    }
    const rows = fills.map(fill => {
        const pnl = Number(fill.pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const okxBadge = fill.okx_confirmed
            ? '<span class="position-ledger-mini-badge ok">OKX</span>'
            : '<span class="position-ledger-mini-badge warn">\u672c\u5730</span>';
        return `
            <tr>
                <td>${escHtml(sideLabel(fill.side))}</td>
                <td>${fmtNum(fill.quantity)}</td>
                <td>${fmtPrice(fill.price)}</td>
                <td style="color:${pnlColor};font-weight:700;">${signedMoney(fill.pnl || 0)}</td>
                <td>${fmtNum(fill.fee)}</td>
                <td>${identifierCell(fill.order_id, 'position-linked-order-id', 18)}</td>
                <td>${identifierCell(fill.trade_id, 'position-linked-trade-id', 22)}</td>
                <td>${toBeijingTime(fill.filled_at)}</td>
                <td>${okxBadge}</td>
            </tr>`;
    }).join('');
    body.innerHTML = `
        ${evidenceHtml}
        <div class="table-wrap position-linked-orders-table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>\u65b9\u5411</th>
                        <th>\u6570\u91cf</th>
                        <th>\u4ef7\u683c</th>
                        <th>PnL</th>
                        <th>\u624b\u7eed\u8d39</th>
                        <th>\u8ba2\u5355 ID</th>
                        <th>\u6210\u4ea4 ID</th>
                        <th>\u65f6\u95f4</th>
                        <th>\u6765\u6e90</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
    overlay.style.display = 'flex';
}

function closePositionLinkedOrdersModal() {
    const overlay = document.getElementById('position-linked-orders-modal-overlay');
    if (overlay) overlay.style.display = 'none';
}

function initPositionActions() {
    const tbody = document.getElementById('positions-tbody');
    if (!tbody || tbody.dataset.closeHandlerAttached === '1') return;
    tbody.dataset.closeHandlerAttached = '1';
    tbody.addEventListener('click', (event) => {
        const pnlButton = event.target?.closest?.('.js-open-position-pnl');
        if (pnlButton && tbody.contains(pnlButton)) {
            event.preventDefault();
            openPositionPnlPopover(Number(pnlButton.dataset.positionIndex || 0), pnlButton);
            return;
        }
        const evidenceButton = event.target?.closest?.('.js-position-evidence');
        if (evidenceButton && tbody.contains(evidenceButton)) {
            event.preventDefault();
            openPositionEvidenceModal(Number(evidenceButton.dataset.positionIndex || 0));
            return;
        }
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
        const total = valueNumber(row.okx_equity_pnl ?? row.total_pnl);
        const cumulative = valueNumber(row.okx_cumulative_equity_pnl ?? row.cumulative_total_pnl);
        const winLoss = `${Number(row.win_count || 0)}胜 / ${Number(row.loss_count || 0)}亏`;
        const symbolCount = Array.isArray(row.symbol_pnl)
            ? row.symbol_pnl.length
            : (Array.isArray(row.symbols) ? row.symbols.length : 0);
        const detailCount = Array.isArray(row.order_details)
            ? row.order_details.length
            : (Array.isArray(row.position_details) ? row.position_details.length : 0);
        const orderCount = Number(row.filled_order_count ?? row.order_count ?? row.trade_count ?? 0);
        const closedCount = Number(row.closed_trade_count ?? row.trade_count ?? 0);
        const entryCount = Number(row.entry_filled_order_count || 0);
        const closeCount = Number(row.close_filled_order_count || 0);
        const pendingSettlementCount = Number(row.pending_settlement_close_count || 0);
        const orderWinLoss = `${entryCount}\u5f00/${closeCount}\u5e73 \u00b7 ${closedCount}\u5df2\u7ed3\u7b97${pendingSettlementCount ? `/${pendingSettlementCount}\u5f85\u7ed3\u7b97` : ''}`;
        return `
        <tr>
            <td style="font-weight:700;white-space:nowrap;">${escHtml(row.date || '-')}</td>
            <td style="color:var(--red);">${fmtMoney(row.realized_loss || 0)} USDT</td>
            <td style="color:var(--green);">${fmtMoney(row.realized_profit || 0)} USDT</td>
            <td style="color:${realized >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:700;">${signedMoney(realized)} USDT</td>
            <td style="font-weight:700;">${dailyPnlEquityDisplay(row, 'okx_equity_pnl')}</td>
            <td>${dailyPnlEquityDisplay(row, 'okx_cumulative_equity_pnl')}</td>
            <td>${orderCount} <span style="color:var(--text-muted);font-size:10px;">${orderWinLoss}</span></td>
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
    const orderDetails = Array.isArray(row.order_details) ? row.order_details : [];
    const orderCount = Number(row.filled_order_count ?? row.order_count ?? row.trade_count ?? 0);
    const closedCount = Number(row.closed_trade_count ?? row.trade_count ?? 0);
    const entryCount = Number(row.entry_filled_order_count || 0);
    const closeCount = Number(row.close_filled_order_count || 0);
    const pendingSettlementCount = Number(row.pending_settlement_close_count || 0);
    const total = valueNumber(row.okx_equity_pnl ?? row.total_pnl);
    const totalColor = signedMoneyColor(total);
    const snapshotNotice = dailyPnlMissingSnapshotNotice(row);
    const orderOnlyDetails = orderDetails.length && !details.length && !positionDetails.length;
    title.textContent = `${date} 盈亏详情（北京时间）`;
    if (orderOnlyDetails) {
        body.innerHTML = `
            <div class="daily-pnl-modal-summary">
                <div>\u5df2\u5e73\u4ed3\u51c0\u76c8\u4e8f <strong style="color:${Number(row.realized_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(row.realized_pnl || 0)} USDT</strong></div>
                <div>OKX\u6743\u76ca\u53d8\u5316 <strong style="color:${totalColor};">${signedMoneyWithUnit(total)}</strong></div>
                <div>\u6210\u4ea4\u8ba2\u5355 <strong>${orderCount}</strong></div>
                <div>\u5f00\u4ed3\u6210\u4ea4 <strong>${entryCount}</strong></div>
                <div>\u5e73\u4ed3\u6210\u4ea4 <strong>${closeCount}</strong></div>
                <div>\u6743\u5a01\u5df2\u7ed3\u7b97 <strong>${closedCount}</strong>${pendingSettlementCount ? ` \u00b7 \u5f85\u7ed3\u7b97 ${pendingSettlementCount}` : ''}</div>
            </div>
            ${snapshotNotice}
            ${renderDailyPnlOrderDetails(orderDetails)}
        `;
        overlay.style.display = 'flex';
        return;
    }
    if (!details.length && !positionDetails.length && !orderDetails.length) {
        const hasOverview = orderCount > 0
            || Number(row.realized_pnl || 0) !== 0
            || Number(row.unrealized_pnl || 0) !== 0
            || valueNumber(row.okx_equity_pnl ?? row.total_pnl) !== null;
        body.innerHTML = hasOverview
            ? `<div style="color:var(--text-muted);font-size:12px;padding:8px;">当日有盈亏汇总，但没有按币种拆分明细。可能是历史记录未保存 symbol_pnl，或该日只保留了总览数据。</div>
               <div class="daily-pnl-modal-summary">
                   <div>已平仓净盈亏 <strong style="color:${Number(row.realized_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(row.realized_pnl || 0)} USDT</strong></div>
                   <div>OKX权益变化 <strong style="color:${totalColor};">${signedMoneyWithUnit(total)}</strong></div>
                   <div>交易笔数 <strong>${Number(row.trade_count || 0)}</strong></div>
               </div>`
            : '<div style="color:var(--text-muted);font-size:12px;padding:8px;">当日没有已平仓交易。</div>';
        if (snapshotNotice && hasOverview) {
            body.innerHTML = snapshotNotice + body.innerHTML;
        }
        overlay.style.display = 'flex';
        return;
    }
    body.innerHTML = `
        <div class="daily-pnl-modal-summary">
            <div>已平仓净盈亏 <strong style="color:${Number(row.realized_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(row.realized_pnl || 0)} USDT</strong></div>
            <div>OKX权益变化 <strong style="color:${totalColor};">${signedMoneyWithUnit(total)}</strong></div>
            <div>交易笔数 <strong>${Number(row.trade_count || 0)}</strong></div>
        </div>
        ${snapshotNotice}
        ${orderDetails.length ? renderDailyPnlOrderDetails(orderDetails) : ''}
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

function renderDailyPnlOrderDetails(orderDetails) {
    return `
        <div class="table-wrap" style="margin-top:10px;">
            <table>
                <thead>
                    <tr>
                        <th>\u65f6\u95f4</th>
                        <th>\u5e01\u79cd</th>
                        <th>\u65b9\u5411</th>
                        <th>\u72b6\u6001</th>
                        <th>\u6570\u91cf</th>
                        <th>\u6210\u4ea4\u4ef7</th>
                        <th>OKX PnL</th>
                        <th>\u624b\u7eed\u8d39</th>
                    </tr>
                </thead>
                <tbody>
                    ${orderDetails.map(item => {
                        const pnl = valueNumber(item.okx_fill_pnl);
                        const pnlColor = pnl === null ? 'var(--text-muted)' : (pnl >= 0 ? 'var(--green)' : 'var(--red)');
                        return `
                            <tr>
                                <td>${toBeijingTime(item.time || item.filled_at || item.created_at)}</td>
                                <td style="font-weight:700;">${escHtml(item.symbol || '-')}</td>
                                <td>${escHtml(sideLabel(item.side) || '-')}</td>
                                <td>${escHtml(statusLabel(item.status) || '-')}</td>
                                <td>${fmtNum(item.quantity || 0)}</td>
                                <td>${fmtPrice(item.price)}</td>
                                <td style="color:${pnlColor};font-weight:700;">${pnl === null ? '-' : `${signedMoney(pnl)} USDT`}</td>
                                <td>${fmtMoney(item.fee || 0)} USDT</td>
                            </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
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
        const sourceLabel = t.execution_source_label || (t.execution_source === 'okx' ? 'OKX同步' : '系统执行');
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
            <td><span class="badge badge-${analysisDisplayAction(d.action, d)}">${analysisActionLabel(d.action, d)}</span></td>
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
    return fetchPositions();
}

function changePositionHistoryPage(page) {
    state.positionHistoryPage = page;
    return fetchPositionHistory();
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

    const button = (targetPage, label, options = {}) => {
        const disabled = options.disabled ? 'disabled' : '';
        const active = options.active ? 'class="active" aria-current="page"' : '';
        return `<button type="button" data-pagination-callback="${callback}" data-page="${targetPage}" ${active} ${disabled}>${label}</button>`;
    };
    let html = '';
    html += button(1, '首页', { disabled: currentPage <= 1 });
    html += button(currentPage - 1, '上一页', { disabled: currentPage <= 1 });
    for (let p = startP; p <= endP; p++) {
        html += button(p, String(p), { active: p === currentPage, disabled: p === currentPage });
    }
    html += button(currentPage + 1, '下一页', { disabled: currentPage >= pages });
    html += button(pages, '末页', { disabled: currentPage >= pages });
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
    if (/^\d{12,}$/.test(text.trim())) {
        return fallback || '未记录可读执行原因；该数字是交易所/本地订单标识，不是策略原因。';
    }
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
            ['\u8ba1\u5212\u662f\u5426\u4f4e\u4e8e\u6700\u5c0f\u5f20\u6570', rules.planned_below_minimum_contracts ? '\u662f' : '\u5426'],
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
    const fallbackSource = trade.execution_source === 'okx' ? 'OKX同步' : '系统执行';
    const sourceLabel = trade.execution_source_label && !isMojibakeText(String(trade.execution_source_label))
        ? trade.execution_source_label
        : fallbackSource;
    const closeStatus = closeStatusLabel(trade);
    const actionTitle = closeStatus
        ? `${actionLabel(trade.action || trade.side)} / ${closeStatus}`
        : actionLabel(trade.action || trade.side);
    const detail = cleanExecutionDetailText(
        detailData?.display_reason || detailData?.detail || trade.display_reason || trade.detail || trade.reason,
        success ? '订单执行成功。' : '订单执行失败，暂无详细原因。'
    );
    const decision = detailData?.decision || trade.decision || {};
    const aiReason = cleanExecutionDetailText(
        decision.reasoning || trade.reasoning || '',
        ''
    );
    const executionReason = cleanExecutionDetailText(
        detailData?.display_reason || decision.execution_reason || trade.execution_reason || trade.reason || detail,
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
    if (e.target?.closest?.('.js-open-position-pnl')) return;
    const pnlPopoverClose = e.target?.closest?.('.position-pnl-popover-close');
    if (pnlPopoverClose) {
        e.preventDefault();
        closePositionPnlPopover();
        return;
    }
    if (!e.target?.closest?.('#position-pnl-popover')) closePositionPnlPopover();
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
    const positionLinkedOrdersButton = e.target?.closest?.('.js-position-linked-orders');
    if (positionLinkedOrdersButton) {
        e.preventDefault();
        openPositionLinkedOrdersModal(positionLinkedOrdersButton.dataset.groupId || '');
        return;
    }
    const positionPnlButton = e.target?.closest?.('.js-position-pnl-detail');
    if (positionPnlButton) {
        e.preventDefault();
        openPositionLinkedOrdersModal(positionPnlButton.dataset.groupId || '');
        return;
    }
    if (e.target.id === 'decision-reason-modal-overlay') {
        closeDecisionReasonModal();
    }
    if (e.target.id === 'daily-pnl-modal-overlay') {
        closeDailyPnlModal();
    }
    if (e.target.id === 'position-linked-orders-modal-overlay') {
        closePositionLinkedOrdersModal();
    }
    if (e.target.id === 'dashboard-user-modal-overlay') {
        closeDashboardUserModal();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closePositionPnlPopover();
});
window.addEventListener('resize', () => {
    const popover = document.getElementById('position-pnl-popover');
    if (popover && !popover.hidden) positionPnlPopoverPosition(openPositionPnlPopoverAnchor, popover);
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

const PAGINATION_HANDLERS = {
    changePositionsPage,
    changePositionHistoryPage,
    changeTradePage,
    changeDecisionsPage,
    changeAnalysisPage,
    changeRiskAlertPage,
    changeExpertMemoryPage,
    changeTradeReflectionPage,
    changeShadowBacktestPage,
    changeMLSignalPage,
    changeProfitAttributionRecordPage,
};

function initPaginationControls() {
    document.addEventListener('click', async event => {
        const button = event.target?.closest?.('.pagination button[data-pagination-callback]');
        if (!button || button.disabled) return;
        const callbackName = safePaginationCallbackName(button.dataset.paginationCallback);
        const handler = PAGINATION_HANDLERS[callbackName];
        const page = Number(button.dataset.page);
        if (!handler || !Number.isInteger(page) || page < 1) return;

        event.preventDefault();
        const container = button.closest('.pagination');
        if (container?.dataset.loading === 'true') return;
        if (container) {
            container.dataset.loading = 'true';
            container.setAttribute('aria-busy', 'true');
            container.querySelectorAll('button').forEach(item => { item.disabled = true; });
        }
        try {
            await Promise.resolve(handler(page));
        } catch (error) {
            console.error(`Failed to load page ${page} for ${callbackName}`, error);
        } finally {
            if (container?.isConnected) {
                delete container.dataset.loading;
                container.removeAttribute('aria-busy');
            }
        }
    });
}

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

function thresholdCatalogSummaryHtml(data) {
    const policy = data?.policy || {};
    const notes = Array.isArray(policy.notes) ? policy.notes : [];
    const flags = [
        policy.hard_risk_auto_relax === false ? '硬风险上限不会自动放松' : '',
        policy.auto_tunable_not_rendered_as_manual_inputs ? '自动调度项不放进手动输入框' : '',
        policy.removed_fake_thresholds ? '无行为接入的假阈值已清理' : '',
    ].filter(Boolean);
    const text = [...flags, ...notes].filter(Boolean).slice(0, 4).join('；');
    return escHtml(text || '阈值治理规则已读取。');
}

function thresholdCatalogItemHtml(item) {
    const current = item?.effective_display || item?.current_display || item?.current || '-';
    const source = [item?.surface, item?.source].filter(Boolean).join(' / ');
    const up = item?.increase_effect ? `<span>调高：${escHtml(item.increase_effect)}</span>` : '';
    const down = item?.decrease_effect ? `<span>调低：${escHtml(item.decrease_effect)}</span>` : '';
    const impact = up || down ? `<div class="threshold-governance-impact">${up}${down}</div>` : '';
    const automation = item?.automation ? `<em>${escHtml(item.automation)}</em>` : '';
    const reason = item?.reason ? `<small>${escHtml(item.reason)}</small>` : '';
    return `
        <div class="threshold-governance-item">
            <div class="threshold-governance-item-head">
                <strong>${escHtml(item?.label || item?.key || '阈值')}</strong>
                <span class="threshold-governance-value">${escHtml(String(current))}</span>
            </div>
            <p>${escHtml(item?.effect || '-')}</p>
            ${impact}
            ${automation}
            ${source ? `<small>来源：${escHtml(source)}</small>` : ''}
            ${reason}
        </div>
    `;
}

function renderThresholdCatalogList(elementId, items) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const list = Array.isArray(items) ? items : [];
    if (!list.length) {
        el.innerHTML = '<div class="threshold-governance-item"><p>暂无项目</p></div>';
        return;
    }
    el.innerHTML = list.map(thresholdCatalogItemHtml).join('');
}

function renderThresholdCatalog(data) {
    const summary = document.getElementById('threshold-governance-summary');
    if (summary) summary.innerHTML = thresholdCatalogSummaryHtml(data);
    const stripText = document.getElementById('threshold-governance-strip-text');
    if (stripText) {
        stripText.textContent = '手动项只保留账户风控和服务连接参数；策略学习能自动计算的阈值不会放进手动输入框。';
    }
    const setCount = (id, label, items) => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = `${label} ${Array.isArray(items) ? items.length : 0}`;
    };
    setCount('threshold-manual-count', '手动', data?.manual_editable);
    setCount('threshold-auto-count', '自动', data?.auto_tunable);
    setCount('threshold-hard-count', '硬上限', data?.manual_hard_guards);
    setCount('threshold-removed-count', '废弃', data?.removed_or_deprecated);
    renderThresholdCatalogList('threshold-manual-editable', data?.manual_editable);
    renderThresholdCatalogList('threshold-service-controls', data?.manual_service_controls);
    renderThresholdCatalogList('threshold-auto-tunable', data?.auto_tunable);
    renderThresholdCatalogList('threshold-manual-hard-guards', data?.manual_hard_guards);
    renderThresholdCatalogList('threshold-removed-deprecated', data?.removed_or_deprecated);
}

async function fetchThresholdCatalog() {
    try {
        const data = await fetchJSON('/api/settings/threshold-catalog');
        renderThresholdCatalog(data);
    } catch (err) {
        const summary = document.getElementById('threshold-governance-summary');
        if (summary) summary.textContent = `阈值治理接口异常：${err.message || err}`;
        const stripText = document.getElementById('threshold-governance-strip-text');
        if (stripText) stripText.textContent = `阈值治理接口异常：${err.message || err}`;
    }
}

async function fetchTradingParams() {
    const data = await fetchJSON('/api/settings/thresholds');
    if (!data) return;

    const intervalInput = document.getElementById('cfg-decision-interval');
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
    await fetchThresholdCatalog();
}

async function saveTradingParams() {
    const intervalInput = document.getElementById('cfg-decision-interval');
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
    await fetchThresholdCatalog();
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

function mlEvidenceValue(value, suffix = '', missingText = '证据缺失') {
    if (value === null || value === undefined || value === '') return missingText;
    return `${value}${suffix}`;
}

function mlArtifactEvidenceMissingText(status) {
    const diagnosticCode = status.model_load_diagnostic?.code || status.status;
    if (diagnosticCode === 'artifact_incompatible') return 'Artifact 不兼容，运行时未加载';
    if (diagnosticCode === 'artifact_load_failed') return 'Artifact 加载失败';
    if (diagnosticCode === 'no_model') return '尚未注册模型 Artifact';
    return '证据缺失';
}

function mlEvidenceRow(label, value, tone = '') {
    return `<div class="ml-evidence-row ${tone}"><span>${escHtml(label)}</span><strong>${escHtml(value)}</strong></div>`;
}

function mlLocalEvidenceHtml(status) {
    const evidenceValue = (value, suffix = '') => mlEvidenceValue(
        value,
        suffix,
        mlArtifactEvidenceMissingText(status),
    );
    const quality = status.quality_report || {};
    const totals = quality.totals || {};
    const byKind = quality.by_kind || {};
    const shadow = byKind.shadow || {};
    const trade = byKind.trade || {};
    const reasons = Array.isArray(quality.top_reasons) ? quality.top_reasons : [];
    const diagnostics = quality.training_view_diagnostics || {};
    const symbolInfluence = Array.isArray(diagnostics.leave_one_symbol_out)
        ? diagnostics.leave_one_symbol_out : [];
    const registry = status.artifact_registry || {};
    const manifest = registry.manifest || {};
    const activation = status.artifact_activation_manifest
        || registry.activation_manifest
        || {};
    const walkForward = status.walk_forward_report || {};
    const symbolStability = status.leave_one_symbol_out_report || {};
    const readiness = status.readiness || {};
    const blockers = Array.isArray(readiness.blocking_reasons)
        ? readiness.blocking_reasons : [];
    const blockerSource = status._promotion_source || 'ml_signal';
    const authoritative = status.authoritative_trade_return_evidence || {};
    const tasks = status.training_task_manifest || {};
    const replay = status.replay_weight_manifest || {};
    const fingerprint = quality.market_fact_contract?.provenance?.data_fingerprint
        || diagnostics.provenance?.data_fingerprint
        || status.training_data_sha256;
    const evidenceSections = [
        `<section class="ml-evidence-panel">
            <div class="ml-evidence-head"><strong>数据版本与样本分布</strong><span>${escHtml(quality.data_quality_version || '版本缺失')}</span></div>
            ${mlEvidenceRow('数据指纹', evidenceValue(fingerprint))}
            ${mlEvidenceRow('全部 / 纳入 / 隔离', `${mlSampleCountLabel(mlOptionalNumber(totals.total))} / ${mlSampleCountLabel(mlOptionalNumber(totals.included))} / ${mlSampleCountLabel(mlOptionalNumber(totals.excluded))}`)}
            ${mlEvidenceRow('影子样本 原始 / 纳入 / 降权 / 隔离', `${mlSampleCountLabel(mlOptionalNumber(shadow.total))} / ${mlSampleCountLabel(mlOptionalNumber(shadow.included))} / ${mlSampleCountLabel(mlOptionalNumber(shadow.downweighted))} / ${mlSampleCountLabel(mlOptionalNumber(shadow.excluded))}`)}
            ${mlEvidenceRow('真实成交 纳入 / 隔离', `${mlSampleCountLabel(mlOptionalNumber(trade.included))} / ${mlSampleCountLabel(mlOptionalNumber(trade.excluded))}`)}
        </section>`,
        `<section class="ml-evidence-panel ${reasons.length ? 'warn' : ''}">
            <div class="ml-evidence-head"><strong>隔离与降权原因</strong><span>${reasons.length ? `${reasons.length} 类` : '无已报告原因'}</span></div>
            <div class="ml-evidence-list">${reasons.length
                ? reasons.slice(0, 8).map(item => mlEvidenceRow(item.reason || '未知原因', mlSampleCountLabel(mlOptionalNumber(item.count)), 'warn')).join('')
                : '<div class="ml-evidence-empty">当前 artifact 未报告隔离或降权原因。</div>'}</div>
        </section>`,
        `<section class="ml-evidence-panel">
            <div class="ml-evidence-head"><strong>单币种影响重算</strong><span>${symbolInfluence.length ? `${symbolInfluence.length} 个币种` : '未评估'}</span></div>
            <div class="ml-evidence-list">${symbolInfluence.length
                ? symbolInfluence.slice(0, 8).map(item => mlEvidenceRow(
                    item.symbol || '未知币种',
                    `样本 ${mlSampleCountLabel(mlOptionalNumber(item.sample_count))} · 去除后变化 ${distributionPctLabel(item.best_return_mean_delta_pct)}`,
                )).join('')
                : '<div class="ml-evidence-empty">缺少逐币移除重算证据，不能把符号影响显示为 0。</div>'}</div>
        </section>`,
        `<section class="ml-evidence-panel">
            <div class="ml-evidence-head"><strong>Artifact manifest</strong><span>${escHtml(status.artifact_lifecycle || '生命周期缺失')}</span></div>
            ${mlEvidenceRow('Artifact 版本', evidenceValue(registry.version || manifest.artifact_version || status.artifact_version))}
            ${mlEvidenceRow('Artifact SHA256', evidenceValue(registry.sha256 || manifest.artifact_sha256 || status.artifact_sha256))}
            ${mlEvidenceRow('训练数据 SHA256', evidenceValue(status.training_data_sha256))}
            ${mlEvidenceRow('源码 SHA256', evidenceValue(status.source_code_sha256))}
            ${mlEvidenceRow('Manifest 路径', evidenceValue(registry.manifest_path || status.artifact_manifest_path))}
        </section>`,
        `<section class="ml-evidence-panel">
            <div class="ml-evidence-head"><strong>多任务与重放池</strong><span>${escHtml(status.multitask_prediction_contract_version || '合同缺失')}</span></div>
            ${mlEvidenceRow('机会 / 入场任务样本', `${mlSampleCountLabel(mlOptionalNumber(tasks.market_opportunity?.sample_count))} / ${mlSampleCountLabel(mlOptionalNumber(tasks.entry_timing?.sample_count))}`)}
            ${mlEvidenceRow('退出 / 执行任务样本', `${mlSampleCountLabel(mlOptionalNumber(tasks.exit?.sample_count))} / ${mlSampleCountLabel(mlOptionalNumber(tasks.execution?.sample_count))}`)}
            ${mlEvidenceRow('独立决策组', mlSampleCountLabel(mlOptionalNumber(status.completed_shadow_decision_group_count)))}
            ${mlEvidenceRow('重放池有效样本量', mlEvidenceValue(mlOptionalNumber(replay.effective_sample_size)))}
            ${mlEvidenceRow('标签合同', Array.isArray(status.label_contract_versions) ? status.label_contract_versions.join('，') : '证据缺失')}
        </section>`,
        `<section class="ml-evidence-panel ${activation.live_ml_ready === true ? '' : 'warn'}">
            <div class="ml-evidence-head"><strong>晋升与激活证据</strong><span>${activation.live_ml_ready === true ? '已授权' : '未授权'}</span></div>
            ${mlEvidenceRow('激活阶段', evidenceValue(activation.activation_stage || status.artifact_lifecycle))}
            ${mlEvidenceRow('模拟盘交易权限', status.paper_trading_permission === true ? '允许' : '不可用')}
            ${mlEvidenceRow('实盘候选权限', status.live_trading_permission === true ? '允许逐笔检查' : '未晋升')}
            ${mlEvidenceRow('Walk-forward', evidenceValue(walkForward.status))}
            ${mlEvidenceRow('滚动折数', mlSampleCountLabel(mlOptionalNumber(walkForward.folds?.length)))}
            ${mlEvidenceRow('做多移除单币稳定', evidenceValue(symbolStability.long?.stable))}
            ${mlEvidenceRow('做空移除单币稳定', evidenceValue(symbolStability.short?.stable))}
            ${mlEvidenceRow('权威成交指纹', evidenceValue(authoritative.data_fingerprint))}
        </section>`,
        `<section class="ml-evidence-panel ${blockers.length ? 'bad' : ''}">
            <div class="ml-evidence-head"><strong>当前晋升阻断</strong><span>${blockerSource === 'local_ai_tools' ? 'local_ai_tools · ' : ''}${blockers.length ? `${blockers.length} 项` : '无阻断'}</span></div>
            <div class="ml-evidence-list">${blockers.length
                ? blockers.slice(0, 10).map(item => mlEvidenceRow('阻断原因', dashboardReasonText(item), 'bad')).join('')
                : mlEvidenceRow('就绪判断', dashboardReasonText(readiness.state || status.readiness_state || '证据缺失'))}</div>
        </section>`,
    ];
    return `<div class="ml-evidence-grid">${evidenceSections.join('')}</div>`;
}

function renderMLSignalOverview() {
    const container = document.getElementById('ml-signal-overview');
    const updatedEl = document.getElementById('ml-signal-updated');
    if (!container) return;
    const status = state.mlSignalStatus || {};
    const localStatus = state.localAIToolsStatus || {};
    const localPromotion = localStatus.promotion_recommendation
        || localStatus.activation_manifest?.promotion_recommendation
        || localStatus.artifact_activation_manifest?.promotion_recommendation;
    const localBlockers = localPromotion
        ? (localPromotion.live_blocking_reasons || localPromotion.active_blocking_reasons || localPromotion.blocking_reasons || [])
        : [];
    const evidenceStatus = localPromotion
        ? {
            ...status,
            ...localStatus,
            _promotion_source: 'local_ai_tools',
            _promotion_recommendation: localPromotion,
            readiness: {
                ...(status.readiness || {}),
                blocking_reasons: localBlockers,
                state: localStatus.artifact_lifecycle || localStatus.status || status.readiness?.state,
            },
            artifact_activation_manifest: localStatus.activation_manifest
                || localStatus.artifact_activation_manifest
                || status.artifact_activation_manifest,
        }
        : status;
    const records = state.mlSignalRecords || [];
    const latestRecord = records[0] || null;
    const latestSignal = latestRecord?.ml_signal || null;
    const latestPrediction = mlPrimaryPrediction(latestSignal);
    const latestDistribution = standardizedReturnDistribution(
        latestPrediction,
        latestPrediction?.best_side,
    );
    const ready = status.available === true;
    const unavailableReason = status.message || status.error || '本地 ML 模型尚未返回可用状态';
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    const samples = mlSampleCounts();
    const readiness = status.readiness || {};
    const readinessMetrics = readiness.metrics || {};
    const trainDecisionGroupCount = mlOptionalNumber(status.train_decision_group_count);
    const testDecisionGroupCount = mlOptionalNumber(status.test_decision_group_count);
    const splitEvidenceAvailable = trainDecisionGroupCount !== null && testDecisionGroupCount !== null;
    const readinessDistributionAvailable = status.available === true
        && Boolean(readinessMetrics.training_data_version);
    const dirtySampleRatioLabel = readinessDistributionAvailable
        ? distributionProbabilityLabel(readinessMetrics.dirty_sample_ratio)
        : '缺失 / 未评估';
    const quarantinedSampleCount = readinessDistributionAvailable
        ? mlOptionalNumber(readinessMetrics.quarantined_sample_count)
        : null;
    const downweightedSampleCount = readinessDistributionAvailable
        ? mlOptionalNumber(readinessMetrics.downweighted_sample_count)
        : null;
    const readinessBlockers = Array.isArray(readiness.blocking_reasons) ? readiness.blocking_reasons : [];
    const readinessState = status.readiness_state || readiness.state || status.status || 'learning_only';
    const allowLivePositionInfluence = status.live_ml_ready === true;
    const allowPaperTrading = status.paper_trading_permission === true;
    const influenceEnabled = allowLivePositionInfluence;
    const controlledReadinessDegrade = ready && !allowLivePositionInfluence && ['degraded', 'learning_only'].includes(String(readinessState || '').toLowerCase());
    const readinessDisplayState = controlledReadinessDegrade
        ? '模拟盘可用'
        : dashboardReasonText(readinessState);
    const readinessTone = allowLivePositionInfluence ? 'good' : (ready ? 'warn' : 'bad');
    const readinessReasonText = readinessBlockers.length
        ? readinessBlockers.slice(0, 4).map(dashboardReasonText).filter(Boolean).join('；')
        : Object.keys(readiness).length ? '当前没有就绪阻断项' : '就绪证据缺失';
    const prAucText = `${mlEvidenceValue(mlOptionalNumber(readinessMetrics.long_pr_auc))} / ${mlEvidenceValue(mlOptionalNumber(readinessMetrics.short_pr_auc))}`;
    const latestText = latestRecord
        ? `${toBeijingTime(latestRecord.created_at)} ${latestRecord.symbol || '-'}`
        : '暂无最近预测';
    const strongSignals = records.filter(r => {
        const pred = mlPrimaryPrediction(r.ml_signal) || {};
        const distribution = standardizedReturnDistribution(pred, pred.best_side);
        const objective = Number(distribution?.objective_expected_return_pct);
        return Number.isFinite(objective) && objective > 0 && Number(pred.profit_edge_pct || 0) > 0;
    }).length;

    if (updatedEl) {
        updatedEl.textContent = ready
            ? `市场标签 ${mlSampleCountLabel(samples.mlShadowMarket)} 条 · 反事实成本 ${mlSampleCountLabel(samples.mlShadowCost)} 条 · OKX 实际费后收益 ${mlSampleCountLabel(samples.mlActualReturn)} 条 · 模拟盘${allowPaperTrading ? '正常参与' : '不可用'} · 实盘${influenceEnabled ? '候选就绪' : '未授权'}`
            : `模型不可用 · ${unavailableReason}`;
    }

    container.innerHTML = `
        <div class="ml-flow">
            <div class="ml-flow-step">
                <div class="ml-flow-index">1</div>
                <div><strong>影子市场机会样本</strong><span>${mlSampleCountLabel(samples.mlShadowMarket)} 条，只监督同 horizon 的行情方向和幅度</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">2</div>
                <div><strong>影子反事实成本样本</strong><span>${mlSampleCountLabel(samples.mlShadowCost)} 条，只监督当时可执行成本和滑点</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">3</div>
                <div><strong>OKX 实际费后收益样本</strong><span>${mlSampleCountLabel(samples.mlActualReturn)} 条，唯一监督真实成交收益与滑点尾部</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">4</div>
                <div><strong>${allowPaperTrading ? '参与模拟盘正常决策' : '模型当前不可用'}</strong><span>${influenceEnabled ? '实盘候选已就绪，每笔订单仍执行生产门禁' : '晋升状态只阻断实盘，不阻断模拟盘采样和训练'}</span></div>
            </div>
        </div>
        <div class="ml-overview-grid">
            ${mlMetricCard('模型状态', ready ? (allowPaperTrading ? '模拟盘参与中' : '仅加载') : '不可用', ready ? (influenceEnabled ? '实盘候选已就绪' : '实盘未晋升') : unavailableReason, ready ? (allowPaperTrading ? 'good' : 'warn') : 'bad')}
            ${mlMetricCard('就绪判断', readinessDisplayState, readinessReasonText, readinessTone)}
            ${mlMetricCard('模拟盘交易权限', allowPaperTrading ? '允许' : '不可用', allowPaperTrading ? '晋升、LCB、PF 不参与模拟盘授权' : '模型制品或运行链不可用', allowPaperTrading ? 'good' : 'bad')}
            ${mlMetricCard('实盘候选权限', allowLivePositionInfluence ? '允许逐笔检查' : '未晋升', allowLivePositionInfluence ? '仍须 production_trade_gate 逐笔授权' : '不影响模拟盘分析、交易和训练', allowLivePositionInfluence ? 'good' : 'warn')}
            ${mlMetricCard('影子市场机会样本', mlSampleCountLabel(samples.mlShadowMarket), '不代表实际成交或真实费后收益', Number.isFinite(samples.mlShadowMarket) ? 'good' : 'warn')}
            ${mlMetricCard('影子反事实成本样本', mlSampleCountLabel(samples.mlShadowCost), '监督盘口、费用、资金费与反事实滑点；只作反事实对照，不能覆盖真实盈亏', Number.isFinite(samples.mlShadowCost) ? 'good' : 'warn')}
            ${mlMetricCard('OKX 实际费后收益样本', mlSampleCountLabel(samples.mlActualReturn), '只统计可信已平仓生命周期', Number.isFinite(samples.mlActualReturn) ? 'good' : 'warn')}
            ${mlMetricCard('训练/留出分组', `${mlSampleCountLabel(trainDecisionGroupCount)} / ${mlSampleCountLabel(testDecisionGroupCount)}`, '同一 decision 不得跨训练和留出集', splitEvidenceAvailable ? 'good' : 'warn')}
            ${mlMetricCard('脏样本比例', dirtySampleRatioLabel, `隔离 ${mlSampleCountLabel(quarantinedSampleCount)} / 降权 ${mlSampleCountLabel(downweightedSampleCount)}`, readinessDistributionAvailable ? readinessTone : 'warn')}
            ${mlMetricCard('PR-AUC 多/空（诊断）', prAucText, '仅观察分类器，不参与 ready、评分或晋升', 'muted')}
            ${mlMetricCard('当前新增待训练样本', mlSampleCountLabel(samples.newCount), status.sample_count_blocker || (samples.completedMl !== null && samples.trainedCursor !== null ? `当前干净完成 ${samples.completedMl} - 最近已训练游标 ${samples.trainedCursor}` : '等待当前干净样本总数与最近训练游标'), status.sample_count_blocker ? 'bad' : (Number(samples.newCount || 0) > 0 ? 'warn' : 'muted'))}
            ${mlMetricCard('最近预测', latestText, latestPrediction ? `${mlSideLabel(latestPrediction.best_side)} ${distributionSummaryText(latestDistribution)}` : '等待新分析', latestDistribution && Number(latestDistribution.objective_expected_return_pct) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('正目标期望数量', `${strongSignals} / ${records.length}`, '最近记录里标准合同目标期望为正且有收益差的数量', strongSignals ? 'warn' : 'muted')}
            ${mlMetricCard('训练时间', trainedAt, status.version ? `版本 ${String(status.version).slice(0, 10)}` : '', 'muted')}
            ${mlMetricCard('数据质量版本', readinessMetrics.training_data_version || status.quality_report?.data_quality_version || '证据缺失', `要求 ${readinessMetrics.required_training_data_version || '证据缺失'}`, readinessMetrics.training_data_version && readinessMetrics.training_data_version === readinessMetrics.required_training_data_version ? 'good' : 'warn')}
            ${mlMetricCard('训练窗口配置', samples.limit === null ? '未公开' : String(samples.limit), '这是训练数据窗口，不是收益、仓位或生产准入阈值', 'muted')}
        </div>
        ${mlLocalEvidenceHtml(evidenceStatus)}`;
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
    const childMetadataReadyFallback = Object.values(childEndpoints).filter(item => item && (item.metadata_ready || item.available)).length;
    const childMetadataReadyCount = Number.isFinite(Number(status.child_metadata_ready_count))
        ? Number(status.child_metadata_ready_count) : childMetadataReadyFallback;
    const childLiveProbeOkCount = Number.isFinite(Number(status.child_live_probe_ok_count))
        ? Number(status.child_live_probe_ok_count)
        : Object.values(childEndpoints).filter(item => item && (item.live_probe_ok || item.actual_inference_probe)).length;
    const childTotalCount = Object.keys(childEndpoints).length || 4;
    const childContractStatus = status.child_contract_status || (childLiveProbeOkCount ? 'live_probe_ok' : childMetadataReadyCount ? 'metadata_ready' : 'unavailable');
    if (updatedEl) {
        updatedEl.textContent = serviceAvailable
            ? `影子市场 ${mlSampleCountLabel(samples.localShadowMarket)} 条 · 反事实成本 ${mlSampleCountLabel(samples.localShadowCost)} 条 · OKX 实际费后收益 ${mlSampleCountLabel(samples.localActualReturn)} 条 · 子接口登记 ${childMetadataReadyCount}/${childTotalCount} · 实时探针 ${childLiveProbeOkCount}/${childTotalCount} · 状态 ${childContractStatus}`
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
            label: '子接口状态',
            value: `${childMetadataReadyCount}/${childTotalCount}`,
            subtitle: `登记 ${childMetadataReadyCount}/${childTotalCount}；实时推理探针 ${childLiveProbeOkCount}/${childTotalCount}；状态 ${childContractStatus}`,
            tone: childMetadataReadyCount >= childTotalCount && childLiveProbeOkCount >= childTotalCount ? 'good' : (childMetadataReadyCount > 0 ? 'warn' : 'bad'),
        },
        {
            label: '影子市场机会样本',
            value: mlSampleCountLabel(samples.localShadowMarket),
            subtitle: '只监督行情方向和幅度，不冒充实际收益',
            tone: Number.isFinite(samples.localShadowMarket) ? 'good' : 'warn',
        },
        {
            label: '影子反事实成本样本',
            value: mlSampleCountLabel(samples.localShadowCost),
            subtitle: '独立学习费用、资金费和盘口滑点分布',
            tone: Number.isFinite(samples.localShadowCost) ? 'good' : 'warn',
        },
        {
            label: 'OKX 实际费后收益样本',
            value: mlSampleCountLabel(samples.localActualReturn),
            subtitle: '只来自可信已平仓生命周期，并用于实际收益与滑点校准',
            tone: Number.isFinite(samples.localActualReturn) ? 'good' : 'warn',
        },
        {
            label: '序列样本',
            value: mlSampleCountLabel(mlOptionalNumber(status.sequence_sample_count)),
            subtitle: models.deep_timeseries || models.timeseries || '用于多周期行情预测',
            tone: mlOptionalNumber(status.sequence_sample_count) !== null ? 'good' : 'warn',
        },
        {
            label: '文本情绪样本',
            value: mlSampleCountLabel(mlOptionalNumber(status.text_sentiment_sample_count)),
            subtitle: models.deep_sentiment || models.sentiment || '用于新闻/公告/情绪校准',
            tone: mlOptionalNumber(status.text_sentiment_sample_count) !== null ? 'good' : 'warn',
        },
    ];
    container.innerHTML = `
        <div class="ml-overview-grid ml-overview-grid-compact">
            ${cards.map(item => mlMetricCard(item.label, item.value, item.subtitle, item.tone)).join('')}
        </div>
        <div class="ml-purpose-grid">
            <div class="ml-purpose-card ml-purpose-good">
                <div class="ml-purpose-title">盈利预测</div>
                <div class="ml-purpose-desc">先预测毛市场机会，再由生产组合层独立扣除实时成本和实际滑点尾部。</div>
                <div class="ml-purpose-tech">${escHtml(models.profit || 'ExtraTrees / CatBoost-style')}</div>
            </div>
            <div class="ml-purpose-card ml-purpose-warn">
                <div class="ml-purpose-title">亏损过滤</div>
                <div class="ml-purpose-desc">识别近期容易亏损的币种、方向和行情组合。</div>
                <div class="ml-purpose-tech">${escHtml(models.loss_filter || 'Classifier')}</div>
            </div>
            <div class="ml-purpose-card ml-purpose-muted">
                <div class="ml-purpose-title">训练窗口说明</div>
                <div class="ml-purpose-desc">页面只展示三期干净样本，旧数据不会进入新模型训练。</div>
                <div class="ml-purpose-tech">最近训练：${escHtml(trainedAt)}</div>
            </div>
        </div>`;
}

function renderTrainableModels() {
    const container = document.getElementById('ml-trainable-models');
    if (!container) return;
    const registry = state.modelTrainingRegistry || {};
    const registryModels = Array.isArray(registry.models) ? registry.models : [];
    const summary = registry.summary || {};
    const lifecycleLabels = {
        training: '训练中', trained: '已训练', inference_only: '仅推理',
        shadow_evaluating: '影子评估', promotion_blocked: '禁止晋升',
        canary: '小资金验证', live: '已介入实盘', not_trained: '未训练',
        service_unavailable: '服务不可用',
    };
    const models = registryModels.map(model => ({
        title: model.display_name || model.model_id || '-',
        type: model.model_family || '-',
        ready: Boolean(model.runtime_available && model.identity_verified),
        statusLabel: lifecycleLabels[model.lifecycle] || model.lifecycle || '未知',
        description: `任务：${model.task || '-'}；运行角色：${model.runtime_role || '-'}`,
        samples: `${mlSampleCountLabel(mlOptionalNumber(model.sample_count))} 条可追溯样本`,
        trainedAt: model.trained_at ? toBeijingTime(model.trained_at) : '-',
        usage: model.live_ml_ready ? '影响实盘交易' : (model.trainable ? '未晋升，不影响实盘' : '推理或影子评估'),
        metrics: [
            { label: '可训练', value: model.trainable ? '是' : '否' },
            { label: '产物', value: model.artifact_available ? '已验证' : '无' },
            { label: '身份', value: model.identity_verified ? '已验证' : '未验证' },
            { label: '别名代理', value: model.alias_only ? '是' : '否' },
        ],
        note: Array.isArray(model.blocking_reasons) && model.blocking_reasons.length
            ? `阻塞原因：${model.blocking_reasons.join('、')}`
            : `质量状态：${model.quality_state || '-'}`,
    }));

    if (!models.length) {
        container.innerHTML = `<div class="analysis-empty">${escHtml(registry.error || '模型训练注册表暂无数据。')}</div>`;
        return;
    }

    container.innerHTML = `
        <div class="ml-train-summary">
            ${mlMetricCard('模型总数', String(Number(summary.model_count || models.length)), `注册表版本 ${registry.version || '-'}`, 'good')}
            ${mlMetricCard('可训练模型', String(Number(summary.trainable_count || 0)), '只有这些模型允许产生项目训练产物', 'good')}
            ${mlMetricCard('仅推理/评估', String(Number(summary.inference_or_evaluation_only_count || 0)), '不会冒充持续训练', 'muted')}
            ${mlMetricCard('别名代理', String(Number(summary.alias_only_count || 0)), summary.alias_only_models?.join('、') || '无', Number(summary.alias_only_count || 0) ? 'bad' : 'good')}
            ${mlMetricCard('身份异常', String(Number(summary.identity_failure_count || 0)), summary.identity_failure_models?.join('、') || '无', Number(summary.identity_failure_count || 0) ? 'bad' : 'good')}
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
    return renderProfitAttributionRecords(state.profitAttribution || {});
}

// ========== Opening Funnel ==========
function pctFmt(value) {
    const n = Number(value || 0);
    return `${(n * 100).toFixed(1)}%`;
}

async function fetchOpeningFunnel() {
    const hoursEl = document.getElementById('opening-funnel-hours');
    const hours = hoursEl ? Number(hoursEl.value || 24) : 24;
    let data = null;
    try {
        data = await fetchJSON(`/api/opening-funnel?mode=${state.mode || 'paper'}&hours=${hours}&limit=500`);
    } catch (err) {
        renderOpeningFunnelUnavailable({ detail: err?.message || '开仓漏斗接口请求失败' });
        return;
    }
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
        profit_expectancy: '收益期望',
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
            <span>最近 ${data.window_hours || 24} 小时已平仓 ${trades} 笔，胜率 ${pctLabel(summary.win_rate, 1)}，盈亏比 ${profitFactorLabel(summary.profit_factor)}。</span>
        </div>
        <div class="opening-funnel-kpis">
            <div><span>盈利 / 亏损</span><strong>${Number(summary.win_count || 0)} / ${Number(summary.loss_count || 0)}</strong></div>
            <div><span>平均盈利</span><strong>${signedMoney(summary.avg_win || 0)} U</strong></div>
            <div><span>平均亏损</span><strong>-${fmtMoney(summary.avg_loss || 0)} U</strong></div>
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
        sub: distributionSummaryText(
            signals?.ml?.return_distribution_contract
            || evidence.ml?.return_distribution_contract,
        ),
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
            sub: distributionSummaryText(
                signals?.server_profit?.return_distribution_contract
                || evidence.server_profit?.return_distribution_contract,
            ),
            available: signals?.server_profit?.available === true
                || evidence.server_profit?.available === true,
        }),
        profitAttributionEvidenceStatusChip('时序', evidence.timeseries, {
            type: 'timeseries',
            side: signals?.timeseries?.side || evidence.timeseries?.side,
            main: profitAttributionSideLabel(signals?.timeseries?.side || evidence.timeseries?.side),
            sub: distributionSummaryText(
                signals?.timeseries?.return_distribution_contract
                || evidence.timeseries?.return_distribution_contract,
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

function renderStrategyLearningFallback(data) {
    const summary = document.getElementById('strategy-learning-summary');
    if (!summary) return;
    const production = data?.current_production_strategy || data?.schedule?.current_production_strategy || {};
    summary.innerHTML = `<div class="opening-funnel-empty">\u5f53\u524d\u751f\u4ea7\u7b56\u7565\uff1a${escHtml(production.name || production.id || '\u5408\u540c\u7f3a\u5931')}</div>`;
}

