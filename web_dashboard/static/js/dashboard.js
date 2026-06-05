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
    tradeMode: 'paper',
    availableSymbolCount: 0,
    activeSymbols: [],
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
    executionAccount: null,
    expertMemories: [],
    tradeReflections: [],
    expertMemoryPage: 1,
    expertMemoryTotal: 0,
    tradeReflectionPage: 1,
    tradeReflectionTotal: 0,
    shadowBacktests: [],
    shadowBacktestPage: 1,
    shadowBacktestTotal: 0,
    shadowBacktestStatus: '',
    mlSignalStatus: null,
    localAIToolsStatus: null,
    serverMonitorStatus: null,
    mlSignalRecords: [],
    mlSignalPage: 1,
    tradesTotalPages: 1,
    openingFunnel: null,
    profitAttribution: null,
};
const PAGE_SIZE = 20;
const EXPERT_MEMORY_PAGE_SIZE = 10;
const ML_SIGNAL_PAGE_SIZE = 10;
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
    initScanModeButtons();
    fetchDashboardSummary();
    fetchPnlHistory();
    fetchRecentDecisions();
    fetchRecentExecutions();
    fetchRiskEvents();
    setInterval(() => {
        if (isPageActive('dashboard')) {
            fetchDashboardSummary();
        }
    }, 5000);
    setInterval(fetchPnlHistory, 60000);
    setInterval(fetchRecentDecisions, 30000);
    setInterval(fetchRecentExecutions, 30000);
    setInterval(fetchTrades, 60000);
    setInterval(() => {
        if (isPageActive('positions')) {
            fetchPositions();
        }
    }, 5000);
    setInterval(() => {
        if (isPageActive('server-monitor')) {
            fetchServerMonitor();
        }
    }, 15000);
    fetchActiveSymbols();
    fetchAvailableSymbols();
    populatePriceChartSymbols();
    loadPriceChartKlines('BTC/USDT', '1h');
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
async function fetchJSON(url) {
    try {
        const res = await fetch(url, { cache: 'no-store' });
        return await res.json();
    } catch (e) {
        console.error(`Fetch failed: ${url}`, e);
        return null;
    }
}

async function fetchDashboardSummary() {
    const data = await fetchJSON('/api/dashboard/summary');
    if (!data) return;

    updateModeDisplay(data.mode, data.paused, data.scan_mode);
    updateExecutionAccountPanel(data.execution_account || {});
    updateAccounts(data.accounts || [], data.execution_account || null);
    updateMarketData(data.market || {}, data.accounts || []);
    updateAutoStatus(data);
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
    if (state.scanMode === 'auto') {
        el.textContent = state.availableSymbolCount > 0 ? state.availableSymbolCount : '...';
    } else {
        el.textContent = state.activeSymbols.length;
    }
}

async function fetchTrades() {
    const data = await fetchJSON(`/api/trades?limit=${PAGE_SIZE}&mode=${state.mode}&page=${state.tradesPage}`);
    if (!data) return;
    updateTradeTable(data.trades || [], state.mode, data.total ?? data.count);
}

async function fetchPositions() {
    const requestToken = ++positionsRequestToken;
    const data = await fetchJSON(`/api/dashboard/positions?mode=${state.mode}&page=${state.positionsPage}&page_size=${PAGE_SIZE}`);
    if (!data) return;
    if (requestToken !== positionsRequestToken) return;
    state.positionsPage = data.page || state.positionsPage;
    state.positionsTotal = data.total || 0;
    updatePositionsTable(
        data.positions || [],
        state.positionsPage,
        data.total_pages || 1,
        data.total || 0,
    );
    // Update badge
    const badge = document.getElementById('position-badge');
    if (badge) {
        const total = Number(data.total ?? data.count ?? 0);
        badge.textContent = total;
        badge.style.display = total > 0 ? '' : 'none';
    }
}

async function fetchPositionTickerSnapshot() {
    const data = await fetchJSON(`/api/dashboard/positions?mode=${state.mode}&page=1&page_size=200&open_only=true`);
    if (!data || !data.positions) return;
    let tickers = buildTickersFromPositions(data.positions);
    state.positionTickerSymbols = Object.keys(tickers);
    tickers = await enrichTickersFromOKX(tickers);
    updateTickers(tickers, { replace: true });
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
    if (scanMode) state.scanMode = scanMode;

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

    // Update scan mode toggle
    document.querySelectorAll('.mode-btn[data-scan]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.scan === state.scanMode);
    });

    // Update scan mode label
    const scanLabel = document.getElementById('scan-mode-label');
    if (scanLabel) scanLabel.textContent = state.scanMode === 'auto' ? '自动扫描全市场' : '手动选择币种';

    // Symbol selector only visible in manual mode
    const symbolBar = document.getElementById('symbol-selector');
    const symbolDropdown = document.getElementById('symbol-dropdown');

    if (symbolBar) {
        symbolBar.style.display = state.scanMode === 'manual' ? '' : 'none';
    }
    if (symbolDropdown) {
        symbolDropdown.style.display = state.scanMode === 'manual' ? '' : 'none';
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

    const symbols = Object.keys(state.tickers).sort((a, b) => a.localeCompare(b)).slice(0, 12);
    if (countEl) countEl.textContent = symbols.length + ' 个币种';

    if (!symbols.length) {
        container.innerHTML = '<div class="ticker-card"><div class="ticker-sym">---</div><div class="ticker-price" style="color:var(--text-muted)">暂无持仓币种</div></div>';
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

function updateModelRankings(rankings) {
    state.rankings = rankings;
    const container = document.getElementById('ranking-list');
    if (!container) return;

    if (!rankings.length) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">暂无排名数据</div>';
        return;
    }

    container.innerHTML = rankings.map((r, i) => {
        const posCls = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : i === 2 ? 'rank-3' : 'rank-o';
        return `
        <div class="rank-item">
            <div class="rank-pos ${posCls}">${i + 1}</div>
            <div class="rank-body">
                <div class="rank-name">${r.model_name}</div>
                <div class="rank-bar"><div class="rank-bar-inner" style="width:${Math.max(r.composite_score * 100, 2)}%"></div></div>
                <div class="rank-meta">
                    <span>胜率 ${r.win_rate.toFixed(0)}%</span>
                    <span>夏普 ${r.sharpe_ratio}</span>
                    <span>交易 ${r.total_trades}</span>
                    <span>回撤 ${r.max_drawdown.toFixed(1)}%</span>
                </div>
            </div>
            <div class="rank-pnl" style="color:${r.pnl_pct >= 0 ? 'var(--green)' : 'var(--red)'}">
                ${r.pnl_pct >= 0 ? '+' : ''}${r.pnl_pct.toFixed(2)}%
            </div>
        </div>
    `}).join('');

    const liveSpan = document.getElementById('live-model-name');
    if (liveSpan) {
        liveSpan.textContent = state.liveModel || (rankings[0]?.model_name || '未选择');
    }
}

function updateAccounts(accounts) {
    state.accounts = accounts;
    const container = document.getElementById('account-list');
    if (!container) return;

    // Update positions badge from accounts
    const totalPositions = accounts.reduce((sum, a) => sum + (a.open_positions || 0), 0);
    const posBadge = document.getElementById('position-badge');
    if (posBadge) {
        posBadge.textContent = totalPositions;
        posBadge.style.display = totalPositions > 0 ? '' : 'none';
    }

    if (!accounts.length) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">暂无账户</div>';
        return;
    }

    container.innerHTML = accounts.map(a => {
        const walletBalance = a.wallet_balance ?? a.balance ?? a.current_balance ?? 0;
        const initialBalance = a.initial_balance ?? 0;
        return `
            <div class="acct-row">
                <div>
                    <div class="acct-name">${a.model_name}</div>
                    <div style="font-size:12px;color:var(--text);font-weight:700;">当前余额 ${fmtNum(walletBalance)} USDT</div>
                    <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">初始 ${fmtNum(initialBalance)} USDT</div>
                </div>
            </div>
        `;
        return `
            <div class="acct-row">
                <div>
                    <div class="acct-name">${a.model_name}</div>
                    <div style="font-size:12px;color:var(--text);font-weight:700;">当前余额 ${fmtNum(walletBalance)} USDT</div>
                    <div style="font-size:12px;color:var(--text);font-weight:700;">剩余余额 ${fmtNum(remainingBalance)} USDT</div>
                    <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">初始 ${fmtNum(initialBal)} | 保证金 ${fmtNum(usedMargin)} | 权益 ${fmtNum(equity)}</div>
                    <div style="font-size:10px;color:${unrealizedPnl >= 0 ? 'var(--green)' : 'var(--red)'};margin-top:2px;">浮盈亏 ${fmtNum(unrealizedPnl)}</div>
                    <div style="font-size:10px;color:var(--text-muted);">初始 ${fmtNum(initialBal)} USDT</div>
                </div>
            </div>
        `;
    }).join('').replace(/<div style="font-size:10px;color:var\(--text-muted\);">[^<]*USDT<\/div>/g, '');
}

function updateAccounts(accounts) {
    state.accounts = accounts;
    const container = document.getElementById('account-list');
    if (!container) return;

    const totalPositions = accounts.reduce((sum, a) => sum + (a.open_positions || 0), 0);
    const posBadge = document.getElementById('position-badge');
    if (posBadge) {
        posBadge.textContent = totalPositions;
        posBadge.style.display = totalPositions > 0 ? '' : 'none';
    }

    if (!accounts.length) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">暂无账户</div>';
        return;
    }

    container.innerHTML = accounts.map(a => {
        const walletBalance = a.wallet_balance ?? a.balance ?? a.current_balance ?? 0;
        const initialBalance = a.initial_balance ?? 0;
        const unrealizedPnl = Number(a.display_unrealized_pnl ?? a.unrealized_pnl ?? 0);
        const pnlColor = unrealizedPnl >= 0 ? 'var(--green)' : 'var(--red)';
        return `
            <div class="acct-row">
                <div class="acct-main">
                    <div class="acct-name">${a.model_name}</div>
                    <div style="font-size:12px;color:var(--text);font-weight:700;">当前余额 ${fmtNum(walletBalance)} USDT</div>
                    <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">初始 ${fmtNum(initialBalance)} USDT</div>
                </div>
                <div class="acct-side">
                    <div class="acct-side-label">浮动收益</div>
                    <div class="acct-side-value" style="color:${pnlColor};">${unrealizedPnl >= 0 ? '+' : ''}${fmtNum(unrealizedPnl)} USDT</div>
                </div>
            </div>
        `;
    }).join('');
}

function updateAccounts(accounts) {
    state.accounts = accounts;
    const container = document.getElementById('account-list');
    if (!container) return;

    const totalPositions = accounts.reduce((sum, a) => sum + (a.open_positions || 0), 0);
    const posBadge = document.getElementById('position-badge');
    if (posBadge) {
        posBadge.textContent = totalPositions;
        posBadge.style.display = totalPositions > 0 ? '' : 'none';
    }

    if (!accounts.length) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">\u6682\u65e0\u8d26\u6237</div>';
        return;
    }

    container.innerHTML = accounts.map(a => {
        const walletBalance = a.wallet_balance ?? a.balance ?? a.current_balance ?? 0;
        const initialBalance = a.initial_balance ?? 0;
        const unrealizedPnl = Number(a.display_unrealized_pnl ?? a.unrealized_pnl ?? 0);
        const pnlColor = unrealizedPnl >= 0 ? 'var(--green)' : 'var(--red)';
        return `
            <div class="acct-row">
                <div class="acct-main">
                    <div class="acct-name">${a.model_name}</div>
                    <div style="font-size:12px;color:var(--text);font-weight:700;">当前余额 ${fmtNum(walletBalance)} USDT</div>
                    <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">初始 ${fmtNum(initialBalance)} USDT</div>
                </div>
                <div class="acct-side">
                    <div class="acct-side-label">浮动收益</div>
                    <div class="acct-side-value" style="color:${pnlColor};">${unrealizedPnl >= 0 ? '+' : ''}${fmtNum(unrealizedPnl)} USDT</div>
                </div>
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
    const pauseNote = account.risk_paused
        ? `<div class="exec-risk-note paused">已暂停分析新交易对：${escHtml(translatePauseReason(account.risk_pause_reason || '账户触发风险限制'))}</div>`
        : '<div class="exec-risk-note">系统按 OKX 实际可用余额计算仓位；已有持仓仍会继续复盘和平仓。</div>';

    container.innerHTML = `
        <div class="exec-account-card">
            <div class="exec-account-head">
                <div>
                    <div class="exec-account-name">${escHtml(account.account_name || '多专家执行账户')}</div>
                    <div class="exec-account-mode">${modeLabel} · ${escHtml(account.balance_source || '执行账户')}</div>
                </div>
                <span class="badge ${account.risk_paused ? 'badge-short' : 'badge-long'}">${account.risk_paused ? '暂停开新仓' : '可分析'}</span>
            </div>
            <div class="exec-status-grid">
                <div class="exec-status-cell"><span>剩余额度</span><strong>${fmtMoney(remainingAllocation)} USDT</strong></div>
                <div class="exec-status-cell"><span>账户权益</span><strong>${fmtMoney(accountEquity)} USDT</strong></div>
                <div class="exec-status-cell"><span>持仓保证金占用</span><strong>${fmtMoney(positionMarginUsed)} USDT</strong></div>
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
    container.innerHTML = `
        <div class="acct-row">
            <div class="acct-main">
                <div class="acct-name">${escHtml(account.account_name || account.model_name || '多专家执行账户')}</div>
                <div style="font-size:12px;color:var(--text);font-weight:700;">剩余额度 ${fmtMoney(remainingAllocation)} USDT</div>
                <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">账户权益 ${fmtMoney(accountEquity)} | 今日已平仓（北京时间）${signedMoney(todayTotalPnl)}</div>
            </div>
            <div class="acct-side">
                <div class="acct-side-label">浮动收益</div>
                <div class="acct-side-value" style="color:${pnlColor};">${signedMoney(totalPnl)} USDT</div>
            </div>
        </div>
    `;
}

function buildTickersFromPositions(positions) {
    const tickers = {};
    (positions || []).forEach(position => {
        if (position.is_open === false || !position.symbol) return;
        if (position.exchange_synced === false) return;
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
    const positionTickers = buildPositionTickers(accounts);
    const marketTickers = market.tickers || {};
    const tickers = Object.keys(marketTickers).length
        ? { ...positionTickers, ...marketTickers }
        : positionTickers;
    updateTickers(tickers, { replace: true });
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
                        <th>仓位%</th>
                        <th>是否执行</th>
                    </tr>
                </thead>
                <tbody>
                    ${state.decisions.map(d => {
                        const conf = Number(d.confidence || 0);
                        const sizePct = Number(d.position_size_pct || 0) * 100;
                        const executedHtml = d.was_executed
                            ? '<span style="color:var(--green);font-weight:600;">是</span>'
                            : '<span style="color:var(--text-dim);">否</span>';
                        return `
                            <tr>
                                <td>${toBeijingTime(d.created_at)}</td>
                                <td>${escHtml(d.symbol || '-')}</td>
                                <td><span class="badge badge-${d.action || 'hold'}">${analysisActionLabel(d.action, d)}</span></td>
                                <td style="color:${conf >= 0.65 ? 'var(--green)' : 'var(--text-muted)'};font-weight:600;">${(conf * 100).toFixed(0)}%</td>
                                <td>${sizePct.toFixed(1)}%</td>
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
                        const statusText = success ? '执行成功' : '执行失败';
                        const statusColor = success ? 'var(--green)' : 'var(--red)';
                        return `
                            <tr>
                                <td>${toBeijingTime(t.filled_at || t.created_at)}</td>
                                <td>${escHtml(t.symbol || '-')}</td>
                                <td>${executionActionCell(t)}</td>
                                <td>${Number(t.leverage || 1).toFixed(1)}x</td>
                                <td>${fmtNum(t.quantity)}</td>
                                <td>${fmtPrice(t.price)}</td>
                                <td style="color:${statusColor};font-weight:600;">${statusText}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function updateTradeTable(trades, mode, total) {
    state.allTrades = trades || [];
    state.tradesPageMode = mode || state.tradeMode;

    const badge = document.getElementById('trade-badge');
    state.tradesTotal = Number(total ?? state.allTrades.length);
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
        tbody.innerHTML = `<tr><td colspan="8" style="color:var(--text-muted);text-align:center;padding:24px;">暂无${modeLabel}交易记录</td></tr>`;
        document.getElementById('trades-pagination').style.display = 'none';
        return;
    }

    const totalPages = Math.ceil((state.tradesTotal || filtered.length) / PAGE_SIZE);
    const page = Math.min(state.tradesPage, totalPages);
    const pageData = filtered;

    tbody.innerHTML = pageData.map(t => {
        const time = t.filled_at || t.timestamp || t.created_at || '';
        const timeStr = toBeijingTime(time);
        const reason = t.reason || '-';
        return `
        <tr>
            <td>${t.model_name || '-'}</td>
            <td>${t.symbol || '-'}</td>
            <td>${executionActionCell(t)}</td>
            <td>${fmtNum(t.quantity)}</td>
            <td>${fmtPrice(t.price)}</td>
            <td>${statusLabel(t.status)}</td>
            <td title="${escHtml(reason)}" style="max-width:320px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text-muted);">${escHtml(reason)}</td>
            <td style="font-size:10px;color:var(--text-muted);">${timeStr}</td>
        </tr>
    `}).join('');

    renderPagination('trades-pagination', page, totalPages, state.tradesTotal || filtered.length, 'changeTradePage');
}

function changeTradePage(page) {
    state.tradesPage = page;
    fetchTrades();
}

function changePositionsPage(page) {
    state.positionsPage = page;
    fetchPositions();
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

function addRiskAlert(data) {
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

// --- Sidebar Navigation ---
function initSidebarNav() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            // Toggle active nav item
            document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            // Show/hide page sections
            document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
            const target = document.getElementById('page-' + page);
            if (target) target.classList.add('active');

            // Load data for the selected page
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
            if (page === 'analysis') fetchAnalysisRecords();
            if (page === 'alerts') fetchRiskEvents();
            if (page === 'expert-memory') fetchExpertMemories();
            if (page === 'shadow-backtest') fetchShadowBacktests();
            if (page === 'ml-signal') fetchMLSignalDashboard();
            if (page === 'server-monitor') fetchServerMonitor();
            if (page === 'settings') { fetchOKXSettings(); fetchExecutionAccountSettings(); fetchAIModels(); fetchTradingParams(); }
        });
    });
}

// --- Mode Controls ---
function initModeButtons() {
    document.querySelectorAll('.mode-btn[data-mode]').forEach(btn => {
        btn.addEventListener('click', async () => {
            const mode = btn.dataset.mode;
            const res = await fetch('/api/control/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                alert('切换失败: ' + (err.detail || res.statusText));
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
            if (isPageActive('expert-memory')) fetchExpertMemories();
            fetchPositions();
            fetchPositionHistory();
            if (isPageActive('daily-pnl')) fetchDailyPnlRecords();
        });
    });
}

async function togglePause() {
    const endpoint = state.paused ? '/api/control/resume' : '/api/control/pause';
    await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
    await fetchDashboardSummary();
    const btn = document.getElementById('pause-btn');
    if (btn) btn.textContent = state.paused ? '恢复' : '暂停';
}

async function selectLiveModel(modelName) {
    await fetch('/api/control/select-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
    });
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

    const res = await fetch('/api/decisions', { method: 'DELETE' });
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

function renderDecisionsPage(totalPagesOverride = null) {
    const tbody = document.getElementById('all-decisions-tbody');
    if (!tbody) return;

    if (!state.allDecisions.length) {
        tbody.innerHTML = '<tr><td colspan="10" style="color:var(--text-muted);text-align:center;padding:24px;">暂无决策记录</td></tr>';
        document.getElementById('decisions-pagination').style.display = 'none';
        return;
    }

    const totalPages = Number(totalPagesOverride || Math.ceil(state.decisionsTotal / PAGE_SIZE) || 1);
    const page = Math.min(state.decisionsPage, totalPages);
    const pageData = state.allDecisions;

    tbody.innerHTML = pageData.map(d => {
        const time = toBeijingTime(d.created_at);
        const confPct = ((d.confidence || 0) * 100).toFixed(0);
        const confColor = d.confidence >= 0.65 ? 'var(--green)' : 'var(--text-muted)';
        const sizePct = ((d.position_size_pct || 0) * 100).toFixed(1);
        const executedHtml = d.was_executed
            ? `<span style="color:var(--green);font-weight:600;">是</span>`
            : `<span style="color:var(--text-dim);">否</span>`;
        const isPaper = d.is_paper !== false;
        const modeHtml = isPaper
            ? '<span style="color:var(--accent);font-size:11px;">模拟盘</span>'
            : '<span style="color:var(--red);font-size:11px;">实盘</span>';
        const reasonLabel = d.was_executed ? '详情' : '查看';
        return `
        <tr>
            <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${time}</td>
            <td><strong>${escHtml(d.model_name)}</strong></td>
            <td>${escHtml(d.symbol)}</td>
            <td><span class="badge badge-${d.action || 'hold'}">${analysisActionLabel(d.action, d)}</span></td>
            <td><span class="badge badge-${decisionType(d.action)}">${decisionTypeLabel(d)}</span></td>
            <td style="color:${confColor};font-weight:600;">${confPct}%</td>
            <td>${sizePct}%</td>
            <td>${modeHtml}</td>
            <td>${executedHtml}</td>
            <td><button class="btn btn-sm" onclick="showDecisionReason(${Number(d.id)})">${reasonLabel}</button></td>
        </tr>
    `}).join('');

    renderPagination('decisions-pagination', page, totalPages, state.decisionsTotal, 'changeDecisionsPage');
}

function opportunityScoreValue(value, digits = 4) {
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(digits) : '-';
}

function opportunityScoreBlock(score) {
    if (!score || typeof score !== 'object') return '';
    const rank = score.rank && score.candidate_count
        ? `${score.rank} / ${score.candidate_count}`
        : '-';
    const selected = score.selected_for_execution === true
        ? '已进入执行队列'
        : (score.selected_for_execution === false ? '未进入执行队列' : '等待排序');
    const reason = score.selection_reason || score.rule || '系统按预期净收益、方向优势、AI 信心、ML 盈亏质量、手续费、滑点、止损风险和当前敞口综合排序。';
    return `
        <div class="reason-block">
            <div class="reason-label">盈利机会评分</div>
            <div>
                机会分：${opportunityScoreValue(score.score, 6)}<br>
                排名：${escHtml(rank)}<br>
                方向：${escHtml(actionLabel(score.side || '-'))}<br>
                预期收益：${opportunityScoreValue(score.expected_return_pct, 4)}%<br>
                相对反向优势：${opportunityScoreValue(score.profit_edge_pct, 4)}%<br>
                ML 胜率：${opportunityScoreValue(Number(score.win_rate || 0) * 100, 1)}%<br>
                仓位 x 杠杆：${opportunityScoreValue(score.size_x_leverage, 4)}<br>
                手续费+滑点：${opportunityScoreValue(Number(score.fee_pct || 0) + Number(score.slippage_pct || 0), 4)}%<br>
                状态：${escHtml(selected)}<br>
                原因：${escapeMultiline(reason)}
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
    const opportunityHtml = opportunityScoreBlock(decision.opportunity_score);

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
        [/\baligned\b/g, '一致'], 
        [/\bdivergent\b/g, '有分歧'], 
        [/\bneutral\b/g, '中性'], 
        [/\bcompleted\b/g, '已会诊'], 
        [/\bskipped\b/g, '已跳过'], 
        [/\bfailed\b/g, '会诊失败'], 
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

function analysisOpportunityScoreHtml(score) {
    if (!score || typeof score !== 'object') return '';
    const selected = score.selected_for_execution === true
        ? '已进入执行队列'
        : (score.selected_for_execution === false ? '未进入执行队列' : '等待排序');
    const rank = score.rank && score.candidate_count ? `${score.rank}/${score.candidate_count}` : '-';
    const text = [
        `机会分 ${opportunityScoreValue(score.score, 6)}`,
        `排名 ${rank}`,
        `方向 ${actionLabel(score.side || '-')}`,
        `预期收益 ${opportunityScoreValue(score.expected_return_pct, 4)}%`,
        `相对反向优势 ${opportunityScoreValue(score.profit_edge_pct, 4)}%`,
        `ML胜率 ${opportunityScoreValue(Number(score.win_rate || 0) * 100, 1)}%`,
        `状态 ${selected}`,
    ].join('；');
    const reason = score.selection_reason || '用于把多个开仓候选按预期净收益排序，不替代 AI 对方向、仓位、杠杆和平仓的裁决。';
    return `
        <div class="analysis-note"><span>盈利机会评分</span>${analysisText(text)}</div>
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
    if (!Number.isFinite(value) || value <= 0) return '0.0秒';
    if (value < 60) return `${value.toFixed(1)}秒`;
    return `${Math.floor(value / 60)}分${(value % 60).toFixed(1)}秒`;
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
        const rows = skills.map(skill => `
            <div class="analysis-resolution-item">
                <strong>${escHtml(skill.label || skill.name || '-')}</strong>
                <span>
                    ${analysisPill(statusLabel(skill.status), statusTone(skill))}
                    ${skill.decision ? analysisPill(analysisDecisionLabel(skill.decision), statusTone(skill)) : ''}
                    ${skill.confidence !== undefined ? analysisPill(`信心 ${(Number(skill.confidence || 0) * 100).toFixed(0)}%`, 'muted') : ''}
                    ${analysisText(analysisReasonLabel(skill.reason || '-'))}
                </span>
            </div>
        `).join('');
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
    return `<div class="analysis-grid">${skillRows}</div>`;
}

function renderAnalysisLocalAiTools(tools, analysisType = 'market') {
    if (!tools || tools.enabled === false) {
        return '<div class="analysis-empty">本轮没有调用服务器量化工具。</div>';
    }
    const profit = tools.profit_prediction || {};
    const ts = tools.time_series_prediction || {};
    const sentiment = tools.sentiment_analysis || {};
    const exitAdvice = tools.exit_advice || {};
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
    const profitStatus = profit.available === false
        ? analysisPill('未返回', 'warn')
        : analysisPill(profit.trained === false ? '学习中' : '已参与', profit.trained === false ? 'warn' : 'good');
    const tsStatus = ts.available === false
        ? analysisPill('未返回', 'warn')
        : analysisPill(ts.trained === false ? '学习中' : '已参与', ts.trained === false ? 'warn' : 'good');
    const sentimentStatus = sentiment.available === false
        ? analysisPill('未返回', 'warn')
        : analysisPill(sentiment.trained === false ? '学习中' : '已参与', sentiment.trained === false ? 'warn' : 'good');
    const exitStatus = !isPositionAnalysis
        ? analysisPill('市场分析不适用', 'muted')
        : (exitAdvice.available === false
            ? analysisPill('未返回', 'warn')
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
                    <div class="analysis-resolution-list">${predictionRows || '<div class="analysis-empty">暂无时序预测明细</div>'}</div>
                </div>
                <div class="analysis-note analysis-note-muted"><span>情绪模型</span>
                    ${sentimentStatus}
                    ${analysisText([
                        `结论 ${sentiment.label || '-'}`,
                        `情绪分 ${sentiment.score ?? '-'}`,
                        `情绪预期收益 ${signedPctValueLabel(sentiment.expected_return_from_sentiment_pct)}`,
                        `风险 ${sentiment.risk_level || '-'}`,
                        `模型 ${sentiment.model || sentiment.backend || '-'}`
                    ].join('；'))}
                </div>
                ${isPositionAnalysis ? `<div class="analysis-note analysis-note-muted"><span>平仓建议</span>
                    ${exitStatus}
                    ${analysisText(exitAdvice.action ? `${exitAdvice.action_label || analysisDecisionLabel(exitAdvice.action)}，信心 ${(Number(exitAdvice.confidence || 0) * 100).toFixed(0)}%${exitAdvice.reason ? `，原因：${analysisReasonLabel(exitAdvice.reason)}` : ''}` : '本轮没有返回独立平仓建议；如果不是持仓分析记录，通常不会触发这一项。')}
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
    const rows = items.map(item => {
        const impact = Number(item.impact_level || 1);
        const sentiment = Number(item.sentiment_score || 0);
        const tone = impact >= 4 || Math.abs(sentiment) >= 0.5 ? 'warn' : (item.direct_match ? 'good' : 'muted');
        const title = escHtml(item.title || '-');
        const source = escHtml(item.source || '-');
        const rawEventType = item.event_type || 'market_news';
        const eventType = escHtml(item.direct_match && rawEventType === 'market_news' ? 'symbol_news' : rawEventType);
        const reason = escHtml(item.match_reason || '');
        const url = item.url ? `<a href="${escHtml(item.url)}" target="_blank" rel="noopener">来源</a>` : '';
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
                    <div class="analysis-resolution-list">${rows || '<div class="analysis-empty">暂无新闻明细</div>'}</div>
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
            <td><button class="btn btn-sm" onclick="showAnalysisReason('${escHtml(r.id)}')">查看流程</button></td>
        </tr>
    `}).join('');

    renderPagination('analysis-pagination', page, totalPages, state.analysisTotal, 'changeAnalysisPage');
}

async function showAnalysisReason(recordId) { 
    let record = state.analysisRecords.find(r => String(r.id) === String(recordId)); 
    if (!record) return; 
    if (!Array.isArray(record.experts)) {
        const params = new URLSearchParams({
            page: '1',
            page_size: '1',
            decision_id: String(record.decision_id || record.id),
            include_detail: 'true',
            is_paper: state.mode === 'paper' ? 'true' : 'false',
        });
        const detailData = await fetchJSON(`/api/analysis-records?${params.toString()}`);
        const detailed = (detailData?.records || []).find(r => String(r.id) === String(recordId));
        if (detailed) {
            record = detailed;
            const idx = state.analysisRecords.findIndex(r => String(r.id) === String(recordId));
            if (idx >= 0) state.analysisRecords[idx] = detailed;
        }
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
    const totalDuration = Number(
        (record.timing && record.timing.analysis_duration_sec)
        || (record.latency_summary && record.latency_summary.stage_duration_sec)
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
                `${e.latency.shared_batch_call || e.latency.batch_expert ? '共享耗时' : '耗时'} ${analysisDurationLabel(e.latency.duration_sec)}`,
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
    const sharedBatchTimings = modelTimings.filter(item => item && (item.shared_batch_call || item.batch_expert));
    const sharedBatchDuration = Number(
        (latencySummary && latencySummary.shared_batch_duration_sec)
        || (sharedBatchTimings.length ? Math.max(...sharedBatchTimings.map(item => Number(item.duration_sec || 0))) : 0)
    );
    const sharedBatchCount = sharedBatchTimings.length;
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
    const modelTimingHtml = modelTimings.length ? `
        <div class="analysis-resolution-list">
            ${sharedBatchCount ? `
                <div class="analysis-resolution-item">
                    <strong>批量专家请求</strong>
                    <span>
                        真实墙钟 ${analysisDurationLabel(sharedBatchDuration)}
                        · 一次模型调用覆盖 ${sharedBatchCount} 个专家
                    </span>
                </div>
            ` : ''}
            ${modelTimings.map(item => `
                <div class="analysis-resolution-item">
                    <strong>${escHtml(analysisExpertDisplayName(item.name, experts))}</strong>
                    <span>
                        ${item.shared_batch_call || item.batch_expert ? '共享' : ''}
                        ${analysisDurationLabel(item.duration_sec)}
                        · ${escHtml(analysisTimingStatusLabel(item.status))}
                        ${item.provider_model ? ` · ${escHtml(item.provider_model)}` : ''}
                    </span>
                </div>
            `).join('')}
        </div>
    ` : '<div class="analysis-empty">本轮还没有单专家耗时记录</div>';
    const timingHtml = `
        <div class="analysis-card analysis-final-card">
            <div class="analysis-card-head">
                <div class="analysis-card-title">耗时拆解</div>
                <div class="analysis-card-tags">
                    ${analysisPill(`总耗时 ${analysisDurationLabel(totalDuration)}`, totalDuration > 60 ? 'warn' : 'muted')}
                    ${sharedBatchCount ? analysisPill(`专家批量 ${analysisDurationLabel(sharedBatchDuration)}`, sharedBatchDuration > 25 ? 'warn' : 'muted') : ''}
                    ${latencySummary.slowest_model ? analysisPill(`最慢 ${analysisExpertDisplayName(latencySummary.slowest_model.name, experts)}`, Number(latencySummary.slowest_model.duration_sec || 0) > 25 ? 'warn' : 'muted') : ''}
                </div>
            </div>
            <div class="analysis-card-text">
                <div class="analysis-note analysis-note-muted"><span>流程耗时</span>${stageTimingHtml}</div>
                <div class="analysis-note analysis-note-muted"><span>${sharedBatchCount ? '专家批量耗时' : '专家耗时'}</span>${modelTimingHtml}</div>
                ${sharedBatchCount ? '<div class="analysis-note analysis-note-muted"><span>耗时说明</span>5 个专家是一次批量模型请求返回，所以每个专家显示的是同一个共享墙钟耗时；不能把 5 个 30 秒相加。</div>' : ''}
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
                        ${analysisOpportunityScoreHtml(record.opportunity_score)}
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
    const targetEl = document.getElementById('expert-memory-target');
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
    if (targetEl) {
        const target = data.daily_target || {};
        targetEl.innerHTML = `每日目标：约 ${fmtMoney(target.target_usdt)} USDT。长期记忆只用于提高筛选质量，不会绕过风控追单。`;
    }

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
                return `
                    <tr>
                        <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(r.created_at)}</td>
                        <td>${escHtml(r.symbol || '-')}</td>
                        <td>${sideLabel(r.side)}</td>
                        <td style="color:${pnlColor};white-space:nowrap;">${signedMoney(pnl)} USDT</td>
                        <td>${Number(r.hold_minutes || 0).toFixed(1)} 分钟</td>
                        <td style="max-width:320px;">${escHtml(r.mistake_summary || '-')}</td>
                        <td style="max-width:320px;">${escHtml(r.improvement_summary || '-')}</td>
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
            return `
                <tr>
                    <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(r.created_at)}</td>
                    <td>${escHtml(r.symbol || '-')}</td>
                    <td><span class="badge badge-${r.decision_action || 'hold'}">${escHtml(r.decision_action_label || actionLabel(r.decision_action))}</span><div style="font-size:10px;color:var(--text-muted);margin-top:4px;">${Math.round(Number(r.decision_confidence || 0) * 100)}%</div></td>
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
    document.getElementById('decision-reason-title').textContent = `${row.symbol || '-'} / 影子复盘`;
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="reason-block">
            <div class="reason-label">复盘结论</div>
            <div>${escapeMultiline(row.conclusion || '-')}</div>
        </div>
        <div class="reason-block">
            <div class="reason-label">杠杆明细</div>
            <div>
                AI建议：${Number(trade.ai_suggested_leverage ?? trade.leverage ?? 1).toFixed(1)}x<br>
                实际下单：${Number(trade.actual_leverage ?? trade.leverage ?? 1).toFixed(1)}x
            </div>
        </div>
        <div class="reason-block">
            <div class="reason-label">当时决策</div>
            <div>${escHtml(row.decision_action_label || actionLabel(row.decision_action))}，信心度 ${Math.round(Number(row.decision_confidence || 0) * 100)}%，周期 ${Number(row.horizon_minutes || 0)} 分钟</div>
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

/*
function renderLocalAIToolsStatus() {
    const container = document.getElementById('local-ai-tools-status');
    const updatedEl = document.getElementById('local-ai-tools-updated');
    if (!container) return;
    const status = state.localAIToolsStatus || {};
    const available = status.available === true;
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    if (updatedEl) {
        updatedEl.textContent = available
            ? `已训练 ${Number(status.shadow_sample_count || 0)} 条影子样本 / ${Number(status.trade_sample_count || 0)} 条交易样本`
            : '服务不可用';
    }
    const models = status.models || {};
    container.innerHTML = `
        <div class="ml-overview-grid">
            ${mlMetricCard('服务状态', available ? '可用' : '不可用', available ? '服务器 31841 已返回训练模型状态' : (status.error || status.message || '等待服务返回'), available ? 'good' : 'bad')}
            ${mlMetricCard('训练时间', trainedAt, status.source ? `来源 ${status.source}` : '', 'muted')}
            ${mlMetricCard('影子复盘样本', String(Number(status.shadow_sample_count || 0)), '用于盈利质量、亏损过滤和时间序列训练', Number(status.shadow_sample_count || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('交易/平仓样本', String(Number(status.trade_sample_count || 0)), '用于币种亏损画像和平仓建议', Number(status.trade_sample_count || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('盈利/亏损模型', models.profit || '未返回', models.loss_filter || '', models.profit ? 'good' : 'warn')}
            ${mlMetricCard('时间序列模型', models.timeseries || '未返回', `周期 ${(status.horizons || []).join('/') || '-'}`, models.timeseries ? 'good' : 'warn')}
            ${mlMetricCard('情绪模型', models.sentiment || '未返回', '独立情绪服务/校准器状态', models.sentiment ? 'good' : 'warn')}
            ${mlMetricCard('平仓模型', models.exit || '未返回', status.objective || '以真实盈利为最终目标', models.exit ? 'good' : 'warn')}
        </div>`;
}

*/
function renderLocalAIToolsStatus() {
    const container = document.getElementById('local-ai-tools-status');
    const updatedEl = document.getElementById('local-ai-tools-updated');
    if (!container) return;
    const status = state.localAIToolsStatus || {};
    const available = status.available === true;
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    if (updatedEl) {
        updatedEl.textContent = available
            ? `已训练 ${Number(status.shadow_sample_count || 0)} 条影子样本 / ${Number(status.trade_sample_count || 0)} 条交易样本`
            : '服务不可用';
    }
    const models = status.models || {};
    container.innerHTML = `
        <div class="ml-overview-grid">
            ${mlMetricCard('服务状态', available ? '可用' : '不可用', available ? '服务器 31841 已返回训练模型状态' : (status.error || status.message || '等待服务返回'), available ? 'good' : 'bad')}
            ${mlMetricCard('训练时间', trainedAt, status.source ? `来源 ${status.source}` : '', 'muted')}
            ${mlMetricCard('影子复盘样本', String(Number(status.shadow_sample_count || 0)), '用于盈利质量、亏损过滤和周期模型训练', Number(status.shadow_sample_count || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('交易/平仓样本', String(Number(status.trade_sample_count || 0)), '用于币种亏损画像和平仓建议', Number(status.trade_sample_count || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('序列样本', String(Number(status.sequence_sample_count || 0)), models.deep_timeseries || '用于 PatchTST/TFT 风格时间序列模型', Number(status.sequence_sample_count || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('文本情绪样本', String(Number(status.text_sentiment_sample_count || 0)), models.deep_sentiment || '用于 CryptoBERT/FinBERT 风格文本情绪模型', Number(status.text_sentiment_sample_count || 0) > 0 ? 'good' : 'warn')}
            ${mlMetricCard('盈利/亏损模型', models.profit || '未返回', models.loss_filter || '', models.profit ? 'good' : 'warn')}
            ${mlMetricCard('平仓模型', models.exit || '未返回', status.objective || '以真实盈利为最终目标', models.exit ? 'good' : 'warn')}
        </div>`;
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
    return Boolean(status?.available && models[key]);
}

function renderLocalAIToolsStatus() {
    const container = document.getElementById('local-ai-tools-status');
    const updatedEl = document.getElementById('local-ai-tools-updated');
    if (!container) return;

    const status = state.localAIToolsStatus || {};
    const models = status.models || {};
    const available = status.available === true;
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    const completedShadowSamples = Number(status.completed_shadow_sample_count || status.total_shadow_sample_count || 0);
    const trainingShadowSamples = Number(status.training_shadow_sample_count || status.shadow_sample_count || 0);
    const displayCompletedShadow = completedShadowSamples || trainingShadowSamples;
    const trainingShadowLimit = Number(status.training_shadow_sample_limit || 20000);

    if (updatedEl) {
        updatedEl.textContent = available
            ? `训练使用 ${trainingShadowSamples} 条影子样本 / 累计 ${displayCompletedShadow} 条 / 交易 ${Number(status.trade_sample_count || 0)} 条`
            : '服务不可用';
    }

    const cards = [
        {
            label: '服务状态',
            value: available ? '可用' : '不可用',
            subtitle: available ? '服务器量化模型服务已连接，可给交易系统提供预测。' : (status.error || status.message || '等待服务器返回状态'),
            tone: available ? 'good' : 'bad',
        },
        {
            label: '最近训练',
            value: trainedAt,
            subtitle: status.source ? `数据来源：${status.source}` : '用于判断模型是否使用了最新样本。',
            tone: 'muted',
        },
        {
            label: '累计影子复盘样本',
            value: String(displayCompletedShadow),
            subtitle: '数据库里已经完成结果回填、可用于训练的影子复盘总数。',
            tone: displayCompletedShadow > 0 ? 'good' : 'warn',
        },
        {
            label: '本次训练使用样本',
            value: String(trainingShadowSamples),
            subtitle: `服务器每次取最近最多 ${trainingShadowLimit} 条训练；超过后这里会稳定在上限。`,
            tone: trainingShadowSamples > 0 ? 'good' : 'warn',
        },
        {
            label: '真实交易样本',
            value: String(Number(status.trade_sample_count || 0)),
            subtitle: '用于学习币种画像、历史亏损压力和平仓建议。',
            tone: Number(status.trade_sample_count || 0) > 0 ? 'good' : 'warn',
        },
        {
            label: 'K线序列样本',
            value: String(Number(status.sequence_sample_count || 0)),
            subtitle: '用于判断未来 10/30/60 分钟方向和波动。',
            tone: Number(status.sequence_sample_count || 0) > 0 ? 'good' : 'warn',
        },
        {
            label: '情绪文本样本',
            value: String(Number(status.text_sentiment_sample_count || 0)),
            subtitle: '用于新闻、公告、社媒情绪校准。',
            tone: Number(status.text_sentiment_sample_count || 0) > 0 ? 'good' : 'warn',
        },
    ];

    const purposeCards = [
        {
            title: '开仓盈利预测',
            desc: '预测做多/做空扣除成本后的预期收益，帮助系统优先选择更可能赚钱的方向。',
            detail: `技术模型：${mlTechName(models.profit)}`,
            tone: models.profit ? 'good' : 'warn',
        },
        {
            title: '亏损风险过滤',
            desc: '判断某个币种和方向是否容易亏损，用来降低反复买入亏损组合的概率。',
            detail: `技术模型：${mlTechName(models.loss_filter)}`,
            tone: models.loss_filter ? 'good' : 'warn',
        },
        {
            title: '多周期行情预测',
            desc: `按 ${(status.horizons || [10, 30, 60]).join('/')} 分钟窗口预测未来收益变化，辅助判断入场时机。`,
            detail: `技术模型：${mlTechName(models.deep_timeseries || models.timeseries)}`,
            tone: (models.deep_timeseries || models.timeseries) ? 'good' : 'warn',
        },
        {
            title: '平仓建议',
            desc: '结合真实持仓盈亏、持仓时间和历史交易画像，判断继续持有、减仓、止盈或止损。',
            detail: `技术模型：${mlTechName(models.exit)}`,
            tone: models.exit ? 'good' : 'warn',
        },
    ];

    container.innerHTML = `
        <div class="ml-overview-grid ml-overview-grid-compact">
            ${cards.map(item => mlMetricCard(item.label, item.value, item.subtitle, item.tone)).join('')}
        </div>
        <div class="ml-purpose-grid">
            ${purposeCards.map(item => `
                <div class="ml-purpose-card ml-purpose-${item.tone}">
                    <div class="ml-purpose-title">${escHtml(item.title)}</div>
                    <div class="ml-purpose-desc">${escHtml(item.desc)}</div>
                    <div class="ml-purpose-tech">${escHtml(item.detail)}</div>
                </div>
            `).join('')}
        </div>`;
}

function renderTrainableModelCard(model) {
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
                ${mlModelStatusPill(model.ready, model.statusLabel)}
            </div>
            <div class="ml-train-model-desc">${escHtml(model.description || '-')}</div>
            <div class="ml-train-model-grid">
                <div><span>训练样本</span><strong>${escHtml(model.samples || '-')}</strong></div>
                <div><span>训练时间</span><strong>${escHtml(model.trainedAt || '-')}</strong></div>
                <div><span>当前作用</span><strong>${escHtml(model.usage || '-')}</strong></div>
            </div>
            ${metrics}
            <div class="ml-train-model-note">${escHtml(model.note || '')}</div>
        </div>`;
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
        limit,
        newCount,
    };
}

function renderTrainableModels() {
    const container = document.getElementById('ml-trainable-models');
    if (!container) return;
    const local = state.localAIToolsStatus || {};
    const ml = state.mlSignalStatus || {};
    const localTrainedAt = local.trained_at ? toBeijingTime(local.trained_at) : '-';
    const mlTrainedAt = ml.trained_at ? toBeijingTime(ml.trained_at) : '-';
    const metrics = ml.metrics || {};
    const autoLast = ml.auto_train_last_result || {};
    const autoTrainText = ml.auto_train_enabled
        ? `自动训练开启；下次检查 ${ml.auto_train_next_check_at ? toBeijingTime(ml.auto_train_next_check_at) : '-'}`
        : '自动训练未开启';
    const models = [
        {
            title: '本地 ML 盈亏质量模型',
            type: '本机 ExtraTrees / 盈亏过滤',
            ready: ml.available === true,
            statusLabel: ml.influence_enabled ? '已介入' : (ml.available ? '学习中' : '未就绪'),
            description: '预测做多/做空的预期收益和收益质量，用于开仓门槛、否决和机会排序。',
            samples: `${Number(ml.sample_count || 0)} 条影子复盘`,
            trainedAt: mlTrainedAt,
            usage: ml.influence_enabled ? '开仓过滤/机会排序' : '只学习不介入',
            metrics: [
                { label: '做多 AUC', value: Number(metrics.long_auc || 0).toFixed(3) },
                { label: '做空 AUC', value: Number(metrics.short_auc || 0).toFixed(3) },
                { label: '做多高分收益', value: signedPctValueLabel(metrics.top_long_avg_return_pct) },
                { label: '做空高分收益', value: signedPctValueLabel(metrics.top_short_avg_return_pct) },
            ],
            note: autoLast.message || autoTrainText,
        },
        {
            title: '盈利预测模型',
            type: local.models?.profit || 'ExtraTreesRegressor',
            ready: localModelStatus(local, 'profit'),
            description: '预测某个币种和方向扣除成本后的预期收益，目标是净收益最大化。',
            samples: `${Number(local.shadow_sample_count || 0)} 条影子样本`,
            trainedAt: localTrainedAt,
            usage: '给专家和最终决策提供收益证据',
            metrics: [
                { label: '特征数', value: String(Number(local.feature_count || 0)) },
                { label: '周期', value: (local.horizons || []).join('/') || '-' },
            ],
            note: local.objective || '以真实盈利为目标，胜率只作为辅助。',
        },
        {
            title: '亏损过滤模型',
            type: local.models?.loss_filter || 'ExtraTreesClassifier',
            ready: localModelStatus(local, 'loss_filter'),
            description: '判断币种/方向组合的亏损概率，帮助识别历史容易亏损的交易组合。',
            samples: `${Number(local.shadow_sample_count || 0)} 条影子样本`,
            trainedAt: localTrainedAt,
            usage: '亏损概率提示/风险过滤',
            metrics: [
                { label: '币种画像', value: String(Number(local.profile_count || 0)) },
                { label: '特征数', value: String(Number(local.feature_count || 0)) },
            ],
            note: '用于避免反复买入近期亏损压力很高的币种方向。',
        },
        {
            title: '时间序列预测模型',
            type: local.models?.timeseries || 'Per-horizon ExtraTreesRegressor',
            ready: localModelStatus(local, 'timeseries'),
            description: '按 10/30/60 分钟窗口预测未来收益变化，辅助判断入场窗口。',
            samples: `${Number(local.sequence_sample_count || 0)} 条序列样本`,
            trainedAt: localTrainedAt,
            usage: '短周期方向和波动预判',
            metrics: [
                { label: '周期', value: (local.horizons || []).join('/') || '-' },
                { label: '特征数', value: String(Number(local.feature_count || 0)) },
            ],
            note: '传统 ML 时序模型，速度快，作为深度时序的稳定备份。',
        },
        {
            title: '深度时序模型',
            type: local.models?.deep_timeseries || 'Torch PatchTST/TFT-style',
            ready: Boolean(local.torch_patch_status?.available),
            description: '用 K 线序列学习未来收益和回撤结构，更适合多周期行情形态。',
            samples: `${Number(local.torch_patch_status?.samples || local.sequence_sample_count || 0)} 条序列样本`,
            trainedAt: localTrainedAt,
            usage: '深度时序辅助预测',
            metrics: [
                { label: '后端', value: local.torch_patch_status?.backend || '-' },
                { label: 'MAE', value: local.torch_patch_status?.train_mae_pct !== undefined ? `${Number(local.torch_patch_status.train_mae_pct).toFixed(4)}%` : '-' },
                { label: '输入维度', value: String(local.torch_patch_status?.input_dim || '-') },
            ],
            note: local.torch_patch_status?.reason || '当前已经可用，但仍需要更多样本继续提升泛化能力。',
        },
        {
            title: '情绪校准模型',
            type: local.models?.sentiment || 'RandomForest sentiment calibration',
            ready: localModelStatus(local, 'sentiment'),
            description: '学习新闻、社媒和事件情绪对收益/风险的影响。',
            samples: `${Number(local.text_sentiment_sample_count || 0)} 条文本情绪样本`,
            trainedAt: localTrainedAt,
            usage: '情绪风险和收益校准',
            metrics: [
                { label: '文本样本', value: String(Number(local.text_sentiment_sample_count || 0)) },
                { label: 'Transformers', value: local.transformers_sentiment_backend?.available ? '可用' : '未启用' },
            ],
            note: '当前文本样本偏少，后续接入更多新闻/公告/社媒文本后会更有价值。',
        },
        {
            title: '深度情绪模型',
            type: local.models?.deep_sentiment || 'CryptoBERT/FinBERT-ready',
            ready: localModelStatus(local, 'deep_sentiment'),
            description: '面向 CryptoBERT / FinBERT 的文本情绪模型接口，支持后续替换为更强文本模型。',
            samples: `${Number(local.text_sentiment_sample_count || 0)} 条文本情绪样本`,
            trainedAt: localTrainedAt,
            usage: '新闻文本情绪辅助',
            metrics: [
                { label: '库', value: local.transformers_sentiment_backend?.library || '-' },
                { label: '版本', value: local.transformers_sentiment_backend?.version || '-' },
            ],
            note: (local.transformers_sentiment_backend?.preferred_models || []).join(' / ') || '等待更多文本数据。',
        },
        {
            title: '平仓/退出建议模型',
            type: local.models?.exit || 'trade-profile plus live pnl rules',
            ready: localModelStatus(local, 'exit'),
            description: '根据历史交易画像、当前浮盈浮亏和持仓时间，判断小赚快跑、止损或继续持有。',
            samples: `${Number(local.trade_sample_count || 0)} 条交易/平仓样本`,
            trainedAt: localTrainedAt,
            usage: '持仓复盘和平仓建议',
            metrics: [
                { label: '交易样本', value: String(Number(local.trade_sample_count || 0)) },
                { label: '币种画像', value: String(Number(local.profile_count || 0)) },
            ],
            note: '它不是单纯追求持仓浮盈，而是服务于已实现净利润。',
        },
    ];

    container.innerHTML = `
        <div class="ml-train-summary">
            ${mlMetricCard('可训练模型', `${models.length} 个`, '包含本地 ML、盈利、亏损、时序、情绪和平仓模型', 'good')}
            ${mlMetricCard('自动训练', ml.auto_train_enabled ? '已开启' : '未开启', autoTrainText, ml.auto_train_enabled ? 'good' : 'warn')}
            ${mlMetricCard('新增样本', String(Number(autoLast.new_sample_count || 0)), autoLast.message || '等待下一次训练检查', Number(autoLast.new_sample_count || 0) >= Number(ml.auto_train_min_new_samples || 500) ? 'good' : 'muted')}
        </div>
        <div class="ml-train-model-list">
            ${models.map(renderTrainableModelCard).join('')}
        </div>`;
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
    const influenceEnabled = status.influence_enabled !== false && status.status === 'ready';
    const mode = status.mode || latestSignal?.mode || 'observe_only';
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    const strongSignals = records.filter(r => {
        const pred = mlPrimaryPrediction(r.ml_signal) || {};
        return Number(pred.best_expected_return_pct || 0) > 0 && Number(pred.profit_edge_pct || 0) > 0;
    }).length;
    const latestText = latestRecord
        ? `${toBeijingTime(latestRecord.created_at)} ${latestRecord.symbol || '-'}`
        : '暂无最近预测';
    if (updatedEl) updatedEl.textContent = ready
        ? `已训练 ${Number(status.sample_count || 0)} 条样本 · ${influenceEnabled ? '已介入' : '学习中'}`
        : '模型不可用';

    container.innerHTML = `
        <div class="ml-flow">
            <div class="ml-flow-step">
                <div class="ml-flow-index">1</div>
                <div><strong>影子复盘样本</strong><span>${Number(status.sample_count || 0)} 条已用于训练</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">2</div>
                <div><strong>提取行情特征</strong><span>${Number(status.feature_count || 0)} 个特征，周期 ${escHtml((status.horizons || []).join('/'))} 分钟</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">3</div>
                <div><strong>预测多空盈亏</strong><span>预期收益和收益差为主，胜率仅作辅助</span></div>
            </div>
            <div class="ml-flow-step">
                <div class="ml-flow-index">4</div>
                <div><strong>${influenceEnabled ? '参与开仓过滤' : '学习观察中'}</strong><span>${influenceEnabled ? '负预期拦截，强收益质量小幅加分' : '指标未达标时不介入交易，继续训练'}</span></div>
            </div>
        </div>
        <div class="ml-overview-grid">
            ${mlMetricCard('模型状态', ready ? (influenceEnabled ? '已介入' : '学习中') : '不可用', mode === 'entry_profit_filter' ? '盈亏质量过滤中' : '只学习不介入', ready ? (influenceEnabled ? 'good' : 'warn') : 'bad')}
            ${mlMetricCard('训练样本', String(Number(status.sample_count || 0)), `训练 ${Number(status.train_count || 0)} / 测试 ${Number(status.test_count || 0)}`, 'good')}
            ${mlMetricCard('最近预测', latestText, latestPrediction ? `${mlSideLabel(latestPrediction.best_side)} 预期 ${signedPctValueLabel(latestPrediction.best_expected_return_pct)}` : '等待新分析', latestPrediction ? (Number(latestPrediction.best_expected_return_pct || 0) > 0 ? 'good' : 'warn') : 'muted')}
            ${mlMetricCard('正期望数量', `${strongSignals} / ${records.length}`, '最近 120 条分析里预期收益为正且有收益差', strongSignals ? 'warn' : 'muted')}
            ${mlMetricCard('训练时间', trainedAt, status.version ? `版本 ${String(status.version).slice(0, 10)}` : '', 'muted')}
            ${mlMetricCard('生效方式', influenceEnabled ? '开仓过滤' : '自动暂停介入', influenceEnabled ? '不直接改方向；只影响开仓门槛/否决' : '继续预测、复盘和训练，达标后自动恢复', influenceEnabled ? 'good' : 'warn')}
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

function monitorNumber(value, digits = 1) {
    const n = Number(value || 0);
    if (!Number.isFinite(n)) return '0';
    return n.toLocaleString('zh-CN', {
        maximumFractionDigits: digits,
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
    const data = state.serverMonitorStatus || {};
    if (updated) {
        updated.textContent = data.checked_at ? toBeijingTime(data.checked_at) : new Date().toLocaleTimeString('zh-CN', { hour12: false });
    }
    if (!overview || !runtimeEl || !servicesEl) return;
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

function runtimeStatusBadge(ok) {
    return `<span class="status-badge ${ok ? 'status-live' : 'status-paused'}">${ok ? '运行中' : '异常'}</span>`;
}

function renderServerModelRuntime(data, container) {
    const runtime = data.model_runtime || {};
    const vllm = runtime.vllm || {};
    const tools = runtime.local_ai_tools || {};
    const processes = data.gpu_processes || [];
    const models = Array.isArray(vllm.models) && vllm.models.length ? vllm.models.join('、') : '未返回模型名';
    const toolsModels = tools.models || {};
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
                <strong>DeepSeek 14B / vLLM ${runtimeStatusBadge(vllm.available)}</strong>
                <div>地址：${escHtml(vllm.endpoint || '-')}</div>
                <div>模型：${escHtml(models)}</div>
                ${vllm.error ? `<div style="color:var(--red);">错误：${escHtml(vllm.error)}</div>` : ''}
            </div>
            <div class="server-monitor-runtime-card">
                <strong>本地量化模型 ${runtimeStatusBadge(tools.available)}</strong>
                <div>地址：${escHtml(tools.endpoint || '-')}</div>
                <div>训练时间：${tools.trained_at ? toBeijingTime(tools.trained_at) : '-'}</div>
                <div>影子样本：${monitorNumber(tools.shadow_sample_count, 0)} · 交易样本：${monitorNumber(tools.trade_sample_count, 0)}</div>
                <div>盈利模型：${escHtml(toolsModels.profit || '未返回')}</div>
                <div>平仓模型：${escHtml(toolsModels.exit || '未返回')}</div>
                ${tools.error ? `<div style="color:var(--red);">错误：${escHtml(tools.error)}</div>` : ''}
            </div>
            <div class="server-monitor-runtime-card">
                <strong>GPU 模型进程</strong>
                ${processRows}
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
    return d.toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', second:'2-digit' });
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

// ========== Pagination ==========

function renderPagination(containerId, page, totalPages, totalItems, callbackName) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (totalPages <= 1) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'flex';

    let html = '';
    html += `<button onclick="${callbackName}(1)" ${page <= 1 ? 'disabled' : ''}>首页</button>`;
    html += `<button onclick="${callbackName}(${page - 1})" ${page <= 1 ? 'disabled' : ''}>上一页</button>`;

    // Page number buttons (show max 7)
    let startP = Math.max(1, page - 3);
    let endP = Math.min(totalPages, page + 3);
    if (endP - startP < 6) {
        if (startP === 1) endP = Math.min(totalPages, startP + 6);
        else startP = Math.max(1, endP - 6);
    }
    for (let p = startP; p <= endP; p++) {
        html += `<button onclick="${callbackName}(${p})" ${p === page ? 'class="active"' : ''}>${p}</button>`;
    }

    html += `<button onclick="${callbackName}(${page + 1})" ${page >= totalPages ? 'disabled' : ''}>下一页</button>`;
    html += `<button onclick="${callbackName}(${totalPages})" ${page >= totalPages ? 'disabled' : ''}>末页</button>`;
    html += `<span class="page-info">共 ${totalItems} 条 / ${totalPages} 页</span>`;

    container.innerHTML = html;
}

// ========== Symbol Selector ==========
let availableSymbols = [];

async function fetchActiveSymbols() {
    const data = await fetchJSON('/api/symbols/active');
    if (!data) return;
    state.activeSymbols = data.symbols || [];
    renderSymbolTags(state.activeSymbols);
    updateSymbolDatalist(state.activeSymbols);
    updateSymbolCount();
}

async function fetchAvailableSymbols() {
    const data = await fetchJSON('/api/symbols/available');
    if (!data) return;
    availableSymbols = data.symbols || [];
    state.availableSymbolCount = availableSymbols.length;
    renderSymbolDropdown();
    updateSymbolCount();
}

function renderSymbolTags(symbols) {
    const container = document.getElementById('symbol-selector');
    if (!container) return;
    container.innerHTML = symbols.map(s => `
        <span class="symbol-tag">
            ${s}
            <span class="remove-sym" onclick="removeSymbol('${s}')" title="移除">×</span>
        </span>
    `).join('');
}

function updateSymbolDatalist(symbols) {
    const dl = document.getElementById('active-symbols-list');
    if (!dl) return;
    dl.innerHTML = symbols.map(s => `<option value="${s}">`).join('');
}

function renderSymbolDropdown() {
    const list = document.getElementById('symbol-dropdown-list');
    if (!list) return;
    if (!availableSymbols.length) {
        list.innerHTML = '<div class="symbol-dropdown-item" style="color:var(--text-secondary)">暂无可用币种</div>';
        return;
    }
    list.innerHTML = availableSymbols.map(s => `
        <div class="symbol-dropdown-item" onclick="addSymbol('${s.symbol}')">
            <strong>${s.symbol}</strong>
            <span style="color:var(--text-secondary);font-size:11px;margin-left:6px;">${s.type}</span>
        </div>
    `).join('');
}

function toggleSymbolDropdown() {
    const menu = document.getElementById('symbol-dropdown-menu');
    if (!menu) return;
    menu.classList.toggle('show');
    if (menu.classList.contains('show') && !availableSymbols.length) {
        fetchAvailableSymbols();
    }
}

function filterSymbolDropdown() {
    const query = (document.getElementById('symbol-search-input')?.value || '').toUpperCase();
    const list = document.getElementById('symbol-dropdown-list');
    if (!list) return;
    const items = list.querySelectorAll('.symbol-dropdown-item');
    items.forEach(item => {
        const text = item.textContent.toUpperCase();
        item.style.display = text.includes(query) ? '' : 'none';
    });
}

async function addSymbol(symbol) {
    const res = await fetch('/api/symbols/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('添加失败: ' + (err.detail || '未知错误'));
        return;
    }
    const data = await res.json();
    renderSymbolTags(data.symbols || []);
    updateSymbolDatalist(data.symbols || []);
    document.getElementById('symbol-dropdown-menu')?.classList.remove('show');
}

async function removeSymbol(symbol) {
    const res = await fetch('/api/symbols/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('移除失败: ' + (err.detail || '未知错误'));
        return;
    }
    const data = await res.json();
    renderSymbolTags(data.symbols || []);
    updateSymbolDatalist(data.symbols || []);
}

// Close dropdown on outside click
document.addEventListener('click', (e) => {
    const dd = document.getElementById('symbol-dropdown');
    if (dd && !dd.contains(e.target)) {
        document.getElementById('symbol-dropdown-menu')?.classList.remove('show');
    }
});

// ========== Price Chart Symbol Selector ==========

let priceChartCurrentSymbol = 'BTC/USDT';
let priceChartCurrentTimeframe = '1h';

async function populatePriceChartSymbols() {
    const data = await fetchJSON('/api/symbols/available');
    if (!data || !data.symbols) return;

    const select = document.getElementById('price-chart-symbol');
    if (!select) return;

    const symbols = data.symbols || [];
    select.innerHTML = symbols.map(s =>
        `<option value="${s.symbol}">${s.symbol}</option>`
    ).join('');

    if (symbols.some(s => s.symbol === priceChartCurrentSymbol)) {
        select.value = priceChartCurrentSymbol;
    } else {
        select.value = 'BTC/USDT';
        priceChartCurrentSymbol = 'BTC/USDT';
    }
}

async function onPriceChartSymbolChange() {
    const symbolSelect = document.getElementById('price-chart-symbol');
    const tfSelect = document.getElementById('price-chart-timeframe');

    if (symbolSelect) priceChartCurrentSymbol = symbolSelect.value;
    if (tfSelect) priceChartCurrentTimeframe = tfSelect.value;

    const titleEl = document.getElementById('price-chart-title');
    if (titleEl) titleEl.textContent = priceChartCurrentSymbol + ' 价格走势';

    await loadPriceChartKlines(priceChartCurrentSymbol, priceChartCurrentTimeframe);
}

async function loadPriceChartKlines(symbol, timeframe) {
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

function updateAutoStatus(stats) {
    const scanModeEl = document.getElementById('status-scan-mode');
    if (scanModeEl) {
        scanModeEl.textContent = state.scanMode === 'auto'
            ? '自动扫描全市场 (OKX)'
            : '手动选择币种';
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
        intervalEl.textContent = state.decisionInterval + '秒/轮';
    }

    const dtEl = document.getElementById('status-decision-trade');
    if (dtEl) updateDecisionPositionStatus();

    const stageEl = document.getElementById('status-current-stage');
    if (stageEl) {
        const stage = stats?.current_stage_label || (stats?.running ? '等待下一轮分析' : '服务未运行');
        const seconds = Number(stats?.round_running_seconds || 0);
        stageEl.textContent = stats?.round_active
            ? `${stage}，已用 ${seconds} 秒`
            : stage;
    }

    const timingEl = document.getElementById('status-round-timing');
    if (timingEl) {
        const started = stats?.last_round_started_at ? shortBeijingTime(stats.last_round_started_at) : '-';
        const finished = stats?.last_round_finished_at ? shortBeijingTime(stats.last_round_finished_at) : '进行中';
        timingEl.textContent = `开始 ${started} / 完成 ${finished}`;
    }

    const errRow = document.getElementById('status-loop-error-row');
    const errEl = document.getElementById('status-loop-error');
    if (errRow && errEl) {
        const err = loopErrorLabel(stats?.last_round_error);
        errRow.style.display = err ? 'flex' : 'none';
        errEl.textContent = err || '-';
    }
}

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
        scanModeEl.textContent = state.scanMode === 'auto'
            ? '\u81ea\u52a8\u626b\u63cf\u5168\u5e02\u573a (OKX)'
            : '\u624b\u52a8\u9009\u62e9\u5e01\u79cd';
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
    document.querySelectorAll('.mode-btn[data-scan]').forEach(btn => {
        btn.addEventListener('click', async () => {
            const scanMode = btn.dataset.scan;
            const res = await fetch('/api/control/scan-mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: scanMode }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                alert('切换失败: ' + (err.detail || res.statusText));
                return;
            }
            state.scanMode = scanMode;
            document.querySelectorAll('.mode-btn[data-scan]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const scanLabel = document.getElementById('scan-mode-label');
            if (scanLabel) scanLabel.textContent = scanMode === 'auto' ? '自动扫描全市场' : '手动选择币种';

            updateSymbolCount();

            const symbolBar = document.getElementById('symbol-selector');
            const symbolDropdown = document.getElementById('symbol-dropdown');
            if (symbolBar) {
                symbolBar.style.display = scanMode === 'auto' ? 'none' : '';
            }
            if (symbolDropdown) {
                symbolDropdown.style.display = scanMode === 'auto' ? 'none' : '';
            }
        });
    });
}

// ========== OKX Settings (split paper/live) ==========
async function fetchOKXSettings() {
    const data = await fetchJSON('/api/settings/okx');
    if (!data) return;

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

        const res = await fetch('/api/settings/execution-account', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            if (status) {
                status.textContent = '保存失败: ' + (err.detail || '未知错误');
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

    const res = await fetch('/api/settings/okx', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

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

    const res = await fetch('/api/settings/okx/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
    });
    const data = await res.json().catch(() => ({}));

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

async function fetchAIModels() {
    const data = await fetchJSON('/api/settings/ai-models');
    if (!data) return;

    const allModels = (data.models || []).concat(data.legacy || []);
    // Store model -> mode mapping for dashboard filtering
    state.modelModeMap = {};
    allModels.forEach(m => { state.modelModeMap[m.name] = m.execution_mode || 'paper'; });
    const models = allModels.filter(m => (m.execution_mode || 'paper') === currentModelMode);
    renderModelList(models);

    const balanceEl = document.getElementById('okx-balance-info');
    if (balanceEl && data.okx) {
        const sameModeModels = allModels.filter(m => (m.execution_mode || 'paper') === currentModelMode);
        const allocated = sameModeModels.reduce((sum, m) => sum + (m.balance || 0), 0);
        const balKey = currentModelMode === 'paper' ? 'paper_balance' : 'live_balance';
        const errKey = currentModelMode === 'paper' ? 'paper_error' : 'live_error';
        const modeLabel = currentModelMode === 'paper' ? '模拟盘' : '实盘';
        let html = '';
        if (data.okx[balKey] !== null && data.okx[balKey] !== undefined) {
            html += `OKX${modeLabel}余额: <strong>${data.okx[balKey].toFixed(2)}</strong> USDT`;
        } else if (data.okx[errKey]) {
            html += `OKX${modeLabel}: <span style="color:var(--yellow)">${data.okx[errKey]}</span>`;
        }
        if (sameModeModels.length > 0) {
            html += ` | 已分配: <strong>${allocated.toFixed(2)}</strong> USDT`;
            if (data.okx[balKey] !== null && allocated > data.okx[balKey]) {
                html += ` <span style="color:var(--red)">(超额!)</span>`;
            }
        }
        balanceEl.innerHTML = html;
    }
}

function renderModelList(models) {
    const tbody = document.getElementById('model-config-tbody');
    if (!tbody) return;

    if (!models.length) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center;padding:24px;">暂无模型配置，点击"+ 添加模型"开始</td></tr>';
        return;
    }

    tbody.innerHTML = models.map(m => {
        const mode = m.execution_mode || 'paper';
        const modeLabel = mode === 'paper' ? '模拟盘' : '实盘';
        const modeColor = mode === 'paper' ? 'var(--accent)' : 'var(--red)';
        return `
        <tr>
            <td><strong>${escHtml(m.name)}</strong></td>
            <td style="font-size:11px;color:var(--text-muted);">${escHtml(m.api_base || '-')}</td>
            <td>${escHtml(m.model || '-')}</td>
            <td>${m.balance ? fmtNum(m.balance) : '默认'}</td>
            <td><span style="color:${modeColor};font-size:11px;font-weight:500;">${modeLabel}</span></td>
            <td>
                <button class="btn btn-sm" onclick="editModel('${escHtml(m.name)}')" title="编辑">✏️</button>
                <button class="btn btn-sm" onclick="testModelByName('${escHtml(m.name)}')" title="测试连接">🔍</button>
                <button class="btn btn-sm" onclick="deleteModel('${escHtml(m.name)}')" title="删除">🗑️</button>
            </td>
        </tr>
    `}).join('');
}

function showAddModelForm() {
    document.getElementById('model-modal-title').textContent = '添加 AI 模型';
    document.getElementById('model-edit-orig-name').value = '';
    document.getElementById('model-cfg-name').value = '';
    document.getElementById('model-cfg-api-base').value = '';
    document.getElementById('model-cfg-api-key').value = '';
    document.getElementById('model-cfg-model').value = '';
    document.getElementById('model-cfg-balance').value = '';
    const modeSel = document.getElementById('model-cfg-mode');
    if (modeSel) modeSel.value = currentModelMode;
    document.getElementById('model-save-btn').textContent = '添加';
    document.getElementById('model-modal-overlay').style.display = 'flex';
}

function editModel(name) {
    fetchJSON('/api/settings/ai-models').then(data => {
        if (!data) return;
        const allModels = (data.models || []).concat(data.legacy || []);
        const m = allModels.find(x => x.name === name);
        if (!m) { alert('未找到模型: ' + name); return; }

        document.getElementById('model-modal-title').textContent = '编辑 AI 模型';
        document.getElementById('model-edit-orig-name').value = name;
        document.getElementById('model-cfg-name').value = m.name || '';
        document.getElementById('model-cfg-api-base').value = m.api_base || '';
        document.getElementById('model-cfg-api-key').value = '';
        document.getElementById('model-cfg-api-key').placeholder = m.api_key ? '已有密钥（已隐藏），留空不变' : 'API Key';
        document.getElementById('model-cfg-model').value = m.model || '';
        document.getElementById('model-cfg-balance').value = m.balance || '';
        const modeSel = document.getElementById('model-cfg-mode');
        if (modeSel) modeSel.value = m.execution_mode || 'paper';
        document.getElementById('model-save-btn').textContent = '保存';
        document.getElementById('model-modal-overlay').style.display = 'flex';
    });
}

async function saveModelConfig() {
    const origName = document.getElementById('model-edit-orig-name').value.trim();
    const isEdit = !!origName;

    const body = {
        name: document.getElementById('model-cfg-name').value.trim(),
        api_base: document.getElementById('model-cfg-api-base').value.trim(),
        api_key: document.getElementById('model-cfg-api-key').value.trim(),
        model: document.getElementById('model-cfg-model').value.trim(),
        balance: parseFloat(document.getElementById('model-cfg-balance').value) || null,
        execution_mode: document.getElementById('model-cfg-mode')?.value || 'paper',
    };

    if (!body.name) { alert('请输入模型名称'); return; }

    const url = isEdit ? `/api/settings/ai-models/${encodeURIComponent(origName)}` : '/api/settings/ai-models';
    const method = isEdit ? 'PUT' : 'POST';

    const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('保存失败: ' + (err.detail || '未知错误'));
        return;
    }

    closeModelModal();
    fetchAIModels();
    alert(isEdit ? '模型已更新' : '模型已添加');
}

async function deleteModel(name) {
    if (!confirm('确定要删除模型 "' + name + '" 吗？此操作不可撤销。')) return;

    const res = await fetch('/api/settings/ai-models/' + encodeURIComponent(name), {
        method: 'DELETE',
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert('删除失败: ' + (err.detail || '未知错误'));
        return;
    }

    fetchAIModels();
    alert('模型已删除');
}

async function testModelByName(name) {
    const btn = event && event.target;
    if (btn && btn.tagName === 'BUTTON') {
        btn.disabled = true;
        btn.textContent = '...';
    }

    const res = await fetch('/api/settings/ai-models/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    });
    const data = await res.json().catch(() => ({}));

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
            '余额和风控额度请在上方“执行账户设置”中维护',
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
            : `<button class="btn btn-sm" onclick="editModel('${escHtml(m.name)}')" title="编辑">编辑</button>
                <button class="btn btn-sm" onclick="testModelByName('${escHtml(m.name)}')" title="测试连接">测试</button>`;
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
    document.getElementById('model-cfg-api-key').placeholder = m.api_key ? '已有密钥（已隐藏），留空不变' : 'API Key';
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

    const res = await fetch(`/api/settings/ai-models/${encodeURIComponent(origName)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

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
        tbody.innerHTML = '<tr><td colspan="10" style="color:var(--text-muted);text-align:center;padding:24px;">暂无正在持仓数据</td></tr>';
        if (pagination) pagination.style.display = 'none';
        return;
    }
    tbody.innerHTML = positions.map(p => {
        const pnl = Number(p.unrealized_pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        return `
        <tr>
            <td>${escHtml(p.symbol || '-')}</td>
            <td><span style="color:${p.side === 'long' ? 'var(--green)' : 'var(--red)'}">${sideLabel(p.side)}</span></td>
            <td>${Number(p.leverage || 1).toFixed(1)}x</td>
            <td>${fmtNum(p.quantity)}</td>
            <td>${fmtPrice(p.entry_price)}</td>
            <td>${fmtPrice(p.current_price || p.entry_price)}</td>
            <td style="color:${pnlColor};font-weight:600;">${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)}</td>
            <td>${p.take_profit ? fmtPrice(p.take_profit) : '-'}</td>
            <td>${p.stop_loss ? fmtPrice(p.stop_loss) : '-'}</td>
            <td style="font-size:10px;color:var(--text-muted);">${toBeijingTime(p.opened_at)}</td>
        </tr>`;
    }).join('');
    renderPagination('positions-pagination', page, totalPages, totalItems, 'changePositionsPage');
}

function renderClosedPositionsTable(positions, page = 1, totalPages = 1, totalItems = 0) {
    const tbody = document.getElementById('position-history-tbody');
    const pagination = document.getElementById('position-history-pagination');
    if (!tbody) return;
    if (!positions.length) {
        tbody.innerHTML = '<tr><td colspan="10" style="color:var(--text-muted);text-align:center;padding:24px;">暂无历史持仓数据</td></tr>';
        if (pagination) pagination.style.display = 'none';
        return;
    }
    tbody.innerHTML = positions.map(p => {
        const pnl = Number(p.realized_pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const statusLabel = p.close_status_label || p.position_status || (p.close_status === 'partial' ? '部分平仓' : '全部平仓');
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
        tbody.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);text-align:center;padding:24px;">暂无每日盈亏记录</td></tr>';
        return;
    }
    tbody.innerHTML = records.map(row => {
        const realized = Number(row.realized_pnl || 0);
        const unrealized = Number(row.unrealized_pnl || 0);
        const total = Number(row.total_pnl || 0);
        const cumulative = Number(row.cumulative_total_pnl ?? row.cumulative_realized_pnl ?? 0);
        const winLoss = `${Number(row.win_count || 0)}胜 / ${Number(row.loss_count || 0)}亏`;
        const symbolCount = Array.isArray(row.symbol_pnl)
            ? row.symbol_pnl.length
            : (Array.isArray(row.symbols) ? row.symbols.length : 0);
        const detailDisabled = symbolCount <= 0 ? 'disabled' : '';
        return `
        <tr>
            <td style="font-weight:700;white-space:nowrap;">${escHtml(row.date || '-')}</td>
            <td style="color:${realized >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:700;">${signedMoney(realized)} USDT</td>
            <td style="color:var(--green);">${fmtMoney(row.realized_profit || 0)} USDT</td>
            <td style="color:var(--red);">${fmtMoney(row.realized_loss || 0)} USDT</td>
            <td style="color:${unrealized >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(unrealized)} USDT</td>
            <td style="color:${total >= 0 ? 'var(--green)' : 'var(--red)'};font-weight:700;">${signedMoney(total)} USDT</td>
            <td style="color:${cumulative >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(cumulative)} USDT</td>
            <td>${Number(row.trade_count || 0)} <span style="color:var(--text-muted);font-size:10px;">${winLoss}</span></td>
            <td>
                <button class="btn btn-sm" ${detailDisabled} onclick="openDailyPnlModal('${escHtml(row.date || '')}')">
                    ${symbolCount ? `查看 ${symbolCount} 个币种` : '无交易'}
                </button>
            </td>
        </tr>`;
    }).join('');
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
        const detailDisabled = symbolCount <= 0 ? 'disabled' : '';
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
                <button class="btn btn-sm" ${detailDisabled} onclick="openDailyPnlModal('${escHtml(row.date || '')}')">
                    ${symbolCount ? `查看 ${symbolCount} 个币种` : '无交易'}
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
    const total = Number(row.total_pnl || 0);
    title.textContent = `${date} 盈亏详情（北京时间）`;
    if (!details.length) {
        body.innerHTML = '<div style="color:var(--text-muted);font-size:12px;padding:8px;">当天没有已平仓交易。</div>';
        overlay.style.display = 'flex';
        return;
    }
    body.innerHTML = `
        <div class="daily-pnl-modal-summary">
            <div>已平仓净盈亏 <strong style="color:${Number(row.realized_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(row.realized_pnl || 0)} USDT</strong></div>
            <div>当日总盈亏 <strong style="color:${total >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(total)} USDT</strong></div>
            <div>交易数 <strong>${Number(row.trade_count || 0)}</strong></div>
        </div>
        <div class="table-wrap" style="margin-top:10px;">
            <table>
                <thead>
                    <tr>
                        <th>币种</th>
                        <th>净盈亏</th>
                        <th>盈利合计</th>
                        <th>亏损合计</th>
                        <th>交易数</th>
                        <th>胜/亏</th>
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
        const statusText = success ? '执行成功' : '执行失败';
        const statusColor = success ? 'var(--green)' : 'var(--red)';
        const sourceLabel = t.execution_source_label || (t.execution_source === 'okx' ? 'OKX执行' : '系统执行');
        const sourceColor = t.execution_source === 'okx' ? 'var(--accent-light)' : 'var(--text-muted)';
        return `
        <tr>
            <td style="font-size:10px;color:var(--text-muted);white-space:nowrap;">${toBeijingTime(time)}</td>
            <td>${escHtml(t.symbol || '-')}</td>
            <td>${executionActionCell(t)}</td>
            <td>${leverageDetailCell(t)}</td>
            <td>${fmtNum(t.quantity)}</td>
            <td>${fmtPrice(t.price)}</td>
            <td style="color:${statusColor};font-weight:600;">${statusText}</td>
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
        const sizePct = ((d.position_size_pct || 0) * 100).toFixed(1);
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
            <td>${sizePct}%</td>
            <td>${executedHtml}</td>
            <td>${reasonBtn}</td>
        </tr>`;
    }).join('');
    renderPagination('decisions-pagination', page, totalPages, state.decisionsTotal, 'changeDecisionsPage');
}

function showExecutionDetail(tradeId) { 
    const trade = state.allTrades.find(t => Number(t.id) === Number(tradeId)); 
    if (!trade) return; 
    setDecisionModalWide(false);
    const success = trade.success === true || trade.status === 'filled'; 
    const sourceLabel = trade.execution_source_label || (trade.execution_source === 'okx' ? 'OKX执行' : '系统执行');
    const closeStatus = closeStatusLabel(trade);
    const actionTitle = closeStatus
        ? `${actionLabel(trade.action || trade.side)} · ${closeStatus}`
        : actionLabel(trade.action || trade.side);
    const detail = trade.detail || trade.reason || (success ? '订单执行成功。' : '订单执行失败，暂无详细原因。');
    document.getElementById('decision-reason-title').textContent =
        `${trade.symbol || '-'} / ${actionTitle} / ${success ? '执行成功' : '执行失败'}`;
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="reason-block">
            <div class="reason-label">${success ? '执行详情' : '失败原因'}</div>
            <div>${escapeMultiline(detail)}</div>
        </div>
        <div class="reason-block">
            <div class="reason-label">订单信息</div>
            <div>
                执行时间：${toBeijingTime(trade.filled_at || trade.created_at)}<br>
                ${closeStatus ? `平仓类型：${escHtml(closeStatus)}<br>` : ''}
                杠杆：${Number(trade.leverage || 1).toFixed(1)}x<br>
                数量：${fmtNum(trade.quantity)}<br>
                价格：${fmtPrice(trade.price)}<br>
                来源：${escHtml(sourceLabel)}<br>
                状态：${statusLabel(trade.status)}
            </div>
        </div>`;
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex';
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
    html += `<button onclick="${callbackName}(1)" ${currentPage <= 1 ? 'disabled' : ''}>首页</button>`;
    html += `<button onclick="${callbackName}(${currentPage - 1})" ${currentPage <= 1 ? 'disabled' : ''}>上一页</button>`;
    for (let p = startP; p <= endP; p++) {
        html += `<button onclick="${callbackName}(${p})" ${p === currentPage ? 'class="active"' : ''}>${p}</button>`;
    }
    html += `<button onclick="${callbackName}(${currentPage + 1})" ${currentPage >= pages ? 'disabled' : ''}>下一页</button>`;
    html += `<button onclick="${callbackName}(${pages})" ${currentPage >= pages ? 'disabled' : ''}>末页</button>`;
    html += `<span class="page-info">共 ${total} 条 / ${pages} 页</span>`;
    container.innerHTML = html;
}

function closeModelModal() {
    document.getElementById('model-modal-overlay').style.display = 'none';
}

// Final override for execution details with readable leverage fields.
function showExecutionDetail(tradeId) {
    const trade = state.allTrades.find(t => Number(t.id) === Number(tradeId));
    if (!trade) return;
    setDecisionModalWide(false);
    const success = trade.success === true || trade.status === 'filled';
    const sourceLabel = trade.execution_source_label || (trade.execution_source === 'okx' ? 'OKX执行' : '系统执行');
    const closeStatus = closeStatusLabel(trade);
    const actionTitle = closeStatus
        ? `${actionLabel(trade.action || trade.side)} / ${closeStatus}`
        : actionLabel(trade.action || trade.side);
    const detail = trade.detail || trade.reason || (success ? '订单执行成功。' : '订单执行失败，暂无详细原因。');
    document.getElementById('decision-reason-title').textContent =
        `${trade.symbol || '-'} / ${actionTitle} / ${success ? '执行成功' : '执行失败'}`;
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="reason-block">
            <div class="reason-label">${success ? '执行详情' : '失败原因'}</div>
            <div>${escapeMultiline(detail)}</div>
        </div>
        <div class="reason-block">
            <div class="reason-label">杠杆明细</div>
            <div>
                AI建议：${Number(trade.ai_suggested_leverage ?? trade.leverage ?? 1).toFixed(1)}x<br>
                实际下单：${Number(trade.actual_leverage ?? trade.leverage ?? 1).toFixed(1)}x
            </div>
        </div>
        <div class="reason-block">
            <div class="reason-label">订单信息</div>
            <div>
                执行时间：${toBeijingTime(trade.filled_at || trade.created_at)}<br>
                ${closeStatus ? `平仓类型：${escHtml(closeStatus)}<br>` : ''}
                数量：${fmtNum(trade.quantity)}<br>
                价格：${fmtPrice(trade.price)}<br>
                来源：${escHtml(sourceLabel)}<br>
                状态：${statusLabel(trade.status)}
            </div>
        </div>`;
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex';
}

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

function showExecutionDetail(tradeId) {
    const trade = state.allTrades.find(t => Number(t.id) === Number(tradeId));
    if (!trade) return;
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
        trade.detail || trade.reason,
        success ? '订单执行成功。' : '订单执行失败，暂无详细原因。'
    );
    const aiLev = Number(trade.ai_suggested_leverage ?? trade.leverage ?? 1).toFixed(1);
    const actualLev = Number(trade.actual_leverage ?? trade.leverage ?? 1).toFixed(1);

    document.getElementById('decision-reason-title').textContent =
        `${trade.symbol || '-'} / ${actionTitle} / ${success ? '执行成功' : '执行失败'}`;
    document.getElementById('decision-reason-body').innerHTML = `
        <div class="reason-block">
            <div class="reason-label">${success ? '执行详情' : '失败原因'}</div>
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
                数量：${fmtNum(trade.quantity)}<br>
                价格：${fmtPrice(trade.price)}<br>
                来源：${escHtml(sourceLabel)}<br>
                状态：${statusLabel(trade.status)}
            </div>
        </div>`;
    document.getElementById('decision-reason-modal-overlay').style.display = 'flex';
}

// Close modal on overlay click
document.addEventListener('click', (e) => {
    if (e.target.id === 'decision-reason-modal-overlay') {
        closeDecisionReasonModal();
    }
    if (e.target.id === 'daily-pnl-modal-overlay') {
        closeDailyPnlModal();
    }
});

function escHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeMultiline(str) {
    return escHtml(str || '').replace(/\n/g, '<br>');
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
    const highRiskEnabledInput = document.getElementById('cfg-high-risk-review-enabled');
    const highRiskBaseInput = document.getElementById('cfg-high-risk-review-api-base');
    const highRiskKeyInput = document.getElementById('cfg-high-risk-review-api-key');
    const highRiskModelInput = document.getElementById('cfg-high-risk-review-model');

    if (intervalInput) intervalInput.value = data.decision_interval;
    if (thresholdInput) thresholdInput.value = data.confidence_threshold;
    if (localToolsEnabledInput) localToolsEnabledInput.checked = Boolean(data.local_ai_tools_enabled);
    if (localToolsBaseInput) localToolsBaseInput.value = data.local_ai_tools_api_base || '';
    if (localToolsTimeoutInput) localToolsTimeoutInput.value = data.local_ai_tools_timeout_seconds ?? 2.5;
    if (highRiskEnabledInput) highRiskEnabledInput.checked = Boolean(data.high_risk_review_enabled);
    if (highRiskBaseInput) highRiskBaseInput.value = data.high_risk_review_api_base || '';
    if (highRiskKeyInput) {
        highRiskKeyInput.value = '';
        highRiskKeyInput.placeholder = data.high_risk_review_has_api_key
            ? '已有密钥（已隐藏），留空不变'
            : '线上模型 API Key';
    }
    if (highRiskModelInput) highRiskModelInput.value = data.high_risk_review_model || '';
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
    const highRiskEnabledInput = document.getElementById('cfg-high-risk-review-enabled');
    const highRiskBaseInput = document.getElementById('cfg-high-risk-review-api-base');
    const highRiskKeyInput = document.getElementById('cfg-high-risk-review-api-key');
    const highRiskModelInput = document.getElementById('cfg-high-risk-review-model');

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
    if (totalMarginInput && totalMarginInput.value) {
        const pct = parseFloat(totalMarginInput.value);
        if (!Number.isFinite(pct) || pct < 10 || pct > 100) {
            alert('保存失败: 总保证金占用上限必须在 10 到 100 之间');
            return;
        }
        body.total_margin_limit_pct = pct / 100;
    }

    const res = await fetch('/api/settings/thresholds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

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

function renderLocalAIToolsStatus() {
    const container = document.getElementById('local-ai-tools-status');
    const updatedEl = document.getElementById('local-ai-tools-updated');
    if (!container) return;
    const status = state.localAIToolsStatus || {};
    const models = status.models || {};
    const available = status.available === true;
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    if (updatedEl) {
        updatedEl.textContent = available
            ? `已训练 ${Number(status.shadow_sample_count || 0)} 条影子样本 / ${Number(status.trade_sample_count || 0)} 条交易样本`
            : '服务不可用';
    }

    const cards = [
        {
            label: '服务状态',
            value: available ? '可用' : '不可用',
            subtitle: available ? '服务器模型服务已连接，可给交易系统提供预测。' : (status.error || status.message || '等待服务器返回状态'),
            tone: available ? 'good' : 'bad',
        },
        {
            label: '最近训练',
            value: trainedAt,
            subtitle: status.source ? `数据来源：${status.source}` : '用于判断模型是否使用了最新样本。',
            tone: 'muted',
        },
        {
            label: '影子复盘样本',
            value: String(Number(status.shadow_sample_count || 0)),
            subtitle: '训练盈利预测、亏损过滤和多周期收益预测。',
            tone: Number(status.shadow_sample_count || 0) > 0 ? 'good' : 'warn',
        },
        {
            label: '真实交易样本',
            value: String(Number(status.trade_sample_count || 0)),
            subtitle: '用于学习币种画像、历史亏损压力和平仓建议。',
            tone: Number(status.trade_sample_count || 0) > 0 ? 'good' : 'warn',
        },
        {
            label: 'K线序列样本',
            value: String(Number(status.sequence_sample_count || 0)),
            subtitle: '用于判断未来 10/30/60 分钟方向和波动。',
            tone: Number(status.sequence_sample_count || 0) > 0 ? 'good' : 'warn',
        },
        {
            label: '情绪文本样本',
            value: String(Number(status.text_sentiment_sample_count || 0)),
            subtitle: '用于新闻、公告、社媒情绪校准。',
            tone: Number(status.text_sentiment_sample_count || 0) > 0 ? 'good' : 'warn',
        },
    ];

    const purposeCards = [
        {
            title: '开仓盈利预测',
            desc: '预测做多/做空扣除成本后的预期收益，帮助系统优先选择更可能赚钱的方向。',
            detail: `技术模型：${mlTechName(models.profit)}`,
            tone: models.profit ? 'good' : 'warn',
        },
        {
            title: '亏损风险过滤',
            desc: '判断某个币种和方向是否容易亏损，用来降低反复买入亏损组合的概率。',
            detail: `技术模型：${mlTechName(models.loss_filter)}`,
            tone: models.loss_filter ? 'good' : 'warn',
        },
        {
            title: '多周期行情预测',
            desc: `按 ${(status.horizons || [10, 30, 60]).join('/')} 分钟窗口预测未来收益变化，辅助判断入场时机。`,
            detail: `技术模型：${mlTechName(models.deep_timeseries || models.timeseries)}`,
            tone: (models.deep_timeseries || models.timeseries) ? 'good' : 'warn',
        },
        {
            title: '平仓建议',
            desc: '结合真实持仓盈亏、持仓时间和历史交易画像，判断继续持有、减仓、止盈或止损。',
            detail: `技术模型：${mlTechName(models.exit)}`,
            tone: models.exit ? 'good' : 'warn',
        },
    ];

    container.innerHTML = `
        <div class="ml-overview-grid ml-overview-grid-compact">
            ${cards.map(item => mlMetricCard(item.label, item.value, item.subtitle, item.tone)).join('')}
        </div>
        <div class="ml-purpose-grid">
            ${purposeCards.map(item => `
                <div class="ml-purpose-card ml-purpose-${item.tone}">
                    <div class="ml-purpose-title">${escHtml(item.title)}</div>
                    <div class="ml-purpose-desc">${escHtml(item.desc)}</div>
                    <div class="ml-purpose-tech">${escHtml(item.detail)}</div>
                </div>
            `).join('')}
        </div>`;
}

function renderTrainableModelCard(model) {
    const metrics = Array.isArray(model.metrics) && model.metrics.length
        ? `<div class="ml-model-metrics">${model.metrics.map(item => `
            <div class="ml-model-metric">
                <span>${escHtml(item.label)}</span>
                <strong>${escHtml(item.value)}</strong>
            </div>
        `).join('')}</div>`
        : '';
    return `
        <div class="ml-train-model-card ml-train-model-card-clear">
            <div class="ml-train-model-head">
                <div>
                    <div class="ml-train-model-title">${escHtml(model.title)}</div>
                    <div class="ml-train-model-desc">${escHtml(model.description || '-')}</div>
                </div>
                ${mlModelStatusPill(model.ready, model.statusLabel)}
            </div>
            <div class="ml-model-purpose-row">
                <span>当前作用</span>
                <strong>${escHtml(model.usage || '-')}</strong>
            </div>
            <div class="ml-train-model-grid">
                <div><span>训练样本</span><strong>${escHtml(model.samples || '-')}</strong></div>
                <div><span>训练时间</span><strong>${escHtml(model.trainedAt || '-')}</strong></div>
                <div><span>技术模型</span><strong>${escHtml(model.type || '-')}</strong></div>
            </div>
            ${metrics}
            <div class="ml-train-model-note">${escHtml(model.note || '')}</div>
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
    const autoTrainText = ml.auto_train_enabled
        ? `自动训练已开启；下次检查 ${ml.auto_train_next_check_at ? toBeijingTime(ml.auto_train_next_check_at) : '-'}`
        : '自动训练未开启';

    const models = [
        {
            title: '本地 ML 盈亏质量',
            type: '本机 ExtraTrees 盈亏过滤',
            ready: ml.available === true,
            statusLabel: ml.influence_enabled ? '已介入' : (ml.available ? '学习中' : '未就绪'),
            description: '判断一笔交易是否有正期望，开仓时用于门槛、否决和机会排序。',
            samples: `${Number(ml.sample_count || 0)} 条影子复盘`,
            trainedAt: mlTrainedAt,
            usage: ml.influence_enabled ? '开仓过滤 + 机会排序' : '只学习，不影响交易',
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
            description: '预测做多和做空扣除成本后的预期收益，目标是净利润最大化。',
            samples: `${Number(local.shadow_sample_count || 0)} 条影子样本`,
            trainedAt: localTrainedAt,
            usage: '给专家和最终裁决提供收益证据',
            metrics: [
                { label: '特征数', value: String(Number(local.feature_count || 0)) },
                { label: '预测周期', value: (local.horizons || []).join('/') || '-' },
            ],
            note: '胜率不是目标，真正目标是扣除手续费和滑点后的实现利润。',
        },
        {
            title: '亏损风险过滤',
            type: mlTechName(modelsMap.loss_filter),
            ready: localModelStatus(local, 'loss_filter'),
            description: '识别某个币种/方向近期是否容易亏损，避免反复交易亏损组合。',
            samples: `${Number(local.shadow_sample_count || 0)} 条影子样本`,
            trainedAt: localTrainedAt,
            usage: '亏损概率提示 + 开仓风险过滤',
            metrics: [
                { label: '币种画像', value: String(Number(local.profile_count || 0)) },
                { label: '特征数', value: String(Number(local.feature_count || 0)) },
            ],
            note: '例如某币种近期连续亏损时，会降低开仓优先级或要求更强证据。',
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
            samples: `${Number(local.trade_sample_count || 0)} 条交易/平仓样本`,
            trainedAt: localTrainedAt,
            usage: '持仓复盘 + 平仓建议',
            metrics: [
                { label: '交易样本', value: String(Number(local.trade_sample_count || 0)) },
                { label: '币种画像', value: String(Number(local.profile_count || 0)) },
            ],
            note: '它服务于已实现净利润，不是单纯追求持仓浮盈。',
        },
    ];

    container.innerHTML = `
        <div class="ml-train-summary">
            ${mlMetricCard('可训练模型', `${models.length} 个`, '覆盖开仓、亏损过滤、时序、情绪和平仓', 'good')}
            ${mlMetricCard('自动训练', ml.auto_train_enabled ? '已开启' : '未开启', autoTrainText, ml.auto_train_enabled ? 'good' : 'warn')}
            ${mlMetricCard('新增样本', String(Number(autoLast.new_sample_count || 0)), autoLast.message || '等待下一次训练检查', Number(autoLast.new_sample_count || 0) >= Number(ml.auto_train_min_new_samples || 500) ? 'good' : 'muted')}
        </div>
        <div class="ml-train-model-list ml-train-model-list-clear">
            ${models.map(renderTrainableModelCard).join('')}
        </div>`;
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
    const available = status.available === true;
    const trainedAt = status.trained_at ? toBeijingTime(status.trained_at) : '-';
    const samples = mlSampleCounts();
    if (updatedEl) {
        updatedEl.textContent = available
            ? `累计 ${samples.completedLocal} 条影子样本 / 训练窗口 ${samples.trainingLocal} 条 / 交易 ${Number(status.trade_sample_count || 0)} 条`
            : '服务不可用';
    }

    const cards = [
        {
            label: '服务状态',
            value: available ? '可用' : '不可用',
            subtitle: available ? '服务器量化工具已连接' : (status.error || status.message || '等待服务返回状态'),
            tone: available ? 'good' : 'bad',
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
            value: String(Number(status.trade_sample_count || 0)),
            subtitle: '用于币种画像和平仓建议',
            tone: Number(status.trade_sample_count || 0) > 0 ? 'good' : 'warn',
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
            samples: `${Number(local.trade_sample_count || 0)} 条交易/平仓样本`,
            trainedAt: localTrainedAt,
            usage: '持仓复盘 + 平仓建议',
            metrics: [
                { label: '交易样本', value: String(Number(local.trade_sample_count || 0)) },
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
    const data = await fetchJSON(`/api/profit-attribution?mode=${mode}&hours=${hours}&limit=300&_=${Date.now()}`);
    state.profitAttribution = data || null;
    renderProfitAttribution();
}

function renderProfitAttribution() {
    const data = state.profitAttribution || {};
    renderProfitAttributionSummary(data);
    renderProfitAttributionBuckets(data);
    renderProfitAttributionState(data);
    renderProfitAttributionRecords(data);
    const updated = document.getElementById('profit-attribution-updated');
    if (updated) {
        const modeLabel = data.mode === 'live' ? '实盘' : '模拟盘';
        updated.textContent = `${modeLabel} · 最近 ${data.window_hours || 24} 小时 · ${new Date().toLocaleTimeString()}`;
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
        el.innerHTML = '<div class="opening-funnel-empty">暂无归因桶。</div>';
        return;
    }
    const maxAbs = Math.max(...rows.map(row => Math.abs(Number(row.pnl || 0))), 1);
    el.innerHTML = rows.map(row => {
        const pnl = Number(row.pnl || 0);
        const width = Math.max(4, Math.abs(pnl) / maxAbs * 100);
        const color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        return `
            <div class="opening-funnel-row opening-funnel-reason-row">
                <div><strong>${escHtml(row.label || row.key || '-')}</strong><span>${Number(row.count || 0)} 笔 · 均值 ${signedMoney(row.avg_pnl || 0)} U</span></div>
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

function renderProfitAttributionRecords(data) {
    const tbody = document.getElementById('profit-attribution-tbody');
    if (!tbody) return;
    const rows = Array.isArray(data.records) ? data.records : [];
    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="8" style="color:var(--text-muted);text-align:center;padding:24px;">暂无归因数据</td></tr>';
        return;
    }
    tbody.innerHTML = rows.map(row => {
        const pnl = Number(row.realized_pnl || 0);
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const entryDecision = row.entry_decision || {};
        const signals = row.signals || {};
        const shadow = row.shadow || {};
        const stateSummary = row.decision_state?.summary || {};
        const stateText = stateSummary.final_stage
            ? `${stateStageLabel(stateSummary.final_stage)}：${stateStatusLabel(stateSummary.final_status)}`
            : '无状态机记录';
        const evidence = [
            `AI ${escHtml(entryDecision.action_label || '-')}`,
            `ML ${sideZh(signals.ml?.side)}`,
            `盈利模型 ${sideZh(signals.server_profit?.side)}`,
            `时序 ${sideZh(signals.timeseries?.side)}`,
            shadow.best_action ? `影子 ${sideZh(shadow.best_action)}` : '',
        ].filter(Boolean).join('<br>');
        const notes = Array.isArray(row.notes) && row.notes.length
            ? `<div style="color:var(--text-muted);margin-top:4px;">${row.notes.map(escHtml).join('；')}</div>`
            : '';
        return `
            <tr>
                <td>${toBeijingTime(row.closed_at)}</td>
                <td>${escHtml(row.symbol || '-')}</td>
                <td>${escHtml(row.side_label || sideZh(row.side))}</td>
                <td style="color:${pnlColor};font-weight:700;">${signedMoney(pnl)} U</td>
                <td>${Number(row.hold_minutes || 0).toFixed(1)} 分钟</td>
                <td><strong>${escHtml(row.main_reason || '-')}</strong>${notes}<div style="color:var(--text-muted);font-size:11px;">置信度 ${confidenceZh(row.attribution_confidence)}</div></td>
                <td>${evidence || '-'}</td>
                <td><strong>${escHtml(stateText)}</strong><div style="color:var(--text-muted);font-size:11px;">${escHtml(stateSummary.final_reason || '')}</div></td>
            </tr>`;
    }).join('');
}

function sideZh(side) {
    const value = String(side || '').toLowerCase();
    if (value === 'long') return '做多';
    if (value === 'short') return '做空';
    if (value === 'hold') return '观望';
    return '-';
}

function stateStageLabel(stage) {
    const labels = {
        ai_analysis: 'AI分析',
        strategy_arbitration: '策略仲裁',
        risk_check: '风控检查',
        exchange_submit: 'OKX提交',
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

// ========== Opening Funnel ==========
function pctFmt(value) {
    const n = Number(value || 0);
    return `${(n * 100).toFixed(1)}%`;
}

function openingFunnelReasonLabel(key) {
    const labels = {
        risk_or_precheck: '风控/预检',
        waiting_queue: '候选排队',
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

async function fetchOpeningFunnel() {
    const hoursEl = document.getElementById('opening-funnel-hours');
    const hours = hoursEl ? Number(hoursEl.value || 24) : 24;
    const data = await fetchJSON(`/api/opening-funnel?mode=${state.mode || 'paper'}&hours=${hours}&limit=1000&_=${Date.now()}`);
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
    el.innerHTML = stages.map(([label, value, desc]) => {
        const width = Math.max(4, (Number(value || 0) / max) * 100);
        return `
            <div class="opening-funnel-stage">
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
            <div class="opening-funnel-row">
                <div><strong>${escHtml(openingFunnelReasonLabel(key))}</strong><span>${pctFmt(ratio)}</span></div>
                <div class="opening-funnel-bar"><span style="width:${Math.max(4, ratio * 100)}%;"></span></div>
                <em>${Number(count || 0)} 次</em>
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
            <div class="opening-funnel-row">
                <div><strong>${escHtml(item.symbol || '-')}</strong><span>${signals}/${scans} 信号 · ${executed} 开仓</span></div>
                <div class="opening-funnel-bar"><span style="width:${width}%;"></span></div>
                <em>${pctFmt(item.signal_rate)}</em>
            </div>`;
    }).join('');
}

function renderOpeningFunnelBlocked(data) {
    const tbody = document.getElementById('opening-funnel-blocked-tbody');
    if (!tbody) return;
    const rows = Array.isArray(data.recent_blocked) ? data.recent_blocked : [];
    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center;padding:24px;">暂无被挡的开仓信号</td></tr>';
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

function openingFunnelReasonLabel(key) {
    const labels = {
        risk_or_precheck: '风控/预检',
        waiting_queue: '候选排序',
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
        tbody.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center;padding:24px;">暂无被挡的开仓信号</td></tr>';
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
