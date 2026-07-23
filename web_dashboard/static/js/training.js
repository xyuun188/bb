(() => {
  const $ = (s) => document.querySelector(s);
  const payloads = {};
  const urls = {
    registry: '/api/model-training/registry',
    scheduler: '/api/model-training/scheduler',
    data: '/api/data-collection/status',
    health: '/api/model-expert-health/status',
    competition: '/api/model-expert-competition/status',
    strategy: '/api/strategy-learning',
    decisions: '/api/analysis-records?limit=12&page=1',
  };
  const unwrap = (v) => v?.data ?? v?.result ?? v ?? {};
  const first = (o, keys, fallback = null) => { for (const key of keys) if (o && o[key] !== undefined && o[key] !== null) return o[key]; return fallback; };
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (v, digits = 0) => Number.isFinite(Number(v)) ? Number(v).toLocaleString('zh-CN', { maximumFractionDigits: digits }) : '--';
  const percent = (v) => Number.isFinite(Number(v)) ? `${(Number(v) * 100).toFixed(2)}%` : '--';
  const time = (v) => { if (!v) return '--'; const d = new Date(v); return Number.isNaN(d.getTime()) ? String(v) : d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }); };
  const line = (label, value) => `<div class="row"><span class="row-label">${esc(label)}</span><span class="row-value">${esc(value ?? '--')}</span></div>`;
  const setState = (label, kind = '') => { const e = $('#page-state'); e.textContent = label; e.className = `state ${kind ? `state-${kind}` : ''}`; };

  function renderOverview() {
    const data = unwrap(payloads.data), registry = unwrap(payloads.registry), scheduler = unwrap(payloads.scheduler);
    const quality = first(data, ['training_governance', 'training', 'quality'], {}) || {};
    const status = first(scheduler, ['status', 'state'], first(registry, ['status'], '未知'));
    const trusted = first(quality, ['trainable_sample_count', 'trusted_sample_count', 'clean_sample_count'], first(registry, ['trainable_sample_count'], null));
    const pending = first(quality, ['pending_sample_count', 'waiting_sample_count'], null);
    const quarantined = first(quality, ['quarantined_count', 'isolated_sample_count'], null);
    const influence = first(registry, ['paper_influence', 'overall_influence', 'influence'], null);
    $('[data-metric="training-status"]').textContent = status ?? '未知';
    $('[data-detail="training-status"]').textContent = first(scheduler, ['message', 'reason'], '持续读取状态');
    $('[data-metric="trusted-samples"]').textContent = fmt(trusted);
    $('[data-detail="trusted-samples"]').textContent = trusted === null ? '等待治理统计' : '可以进入训练视图';
    $('[data-metric="pending-samples"]').textContent = fmt(pending);
    $('[data-detail="pending-samples"]').textContent = pending === null ? '等待成熟结果' : '尚未完成结果计算';
    $('[data-metric="quarantined-samples"]').textContent = fmt(quarantined);
    $('[data-metric="model-influence"]').textContent = influence === null ? '--' : Number(influence) <= 1 ? percent(influence) : fmt(influence, 2);
    $('[data-metric="last-training"]').textContent = time(first(scheduler, ['last_completed_at', 'last_training_at', 'updated_at'], first(registry, ['trained_at'], null)));
    $('[data-detail="last-training"]').textContent = time(first(scheduler, ['next_run_at', 'next_training_at'], null));
  }
  function renderQuality() {
    const data = unwrap(payloads.data), quality = first(data, ['training_governance', 'training', 'quality'], {}) || {}, coverage = first(data, ['feature_coverage', 'coverage'], {}) || {};
    const values = [['训练口径', first(quality, ['summary', 'message'], first(data, ['message'], '已读取数据治理状态'))], ['可训练数量', fmt(first(quality, ['trainable_sample_count', 'clean_sample_count'], null))], ['隔离数量', fmt(first(quality, ['quarantined_count', 'isolated_sample_count'], null))], ['证据完整度', percent(first(coverage, ['ratio', 'completeness'], null))], ['最近更新时间', time(first(data, ['updated_at', 'generated_at'], null))]];
    $('#data-quality').classList.remove('empty'); $('#data-quality').innerHTML = values.map(([a, b]) => line(a, b)).join('');
  }
  function renderScheduler() {
    const s = unwrap(payloads.scheduler), values = [['当前状态', first(s, ['status', 'state'], '未知')], ['最近结果', first(s, ['last_result', 'last_status', 'message'], '暂无结果')], ['最近运行', time(first(s, ['last_started_at', 'last_run_at', 'updated_at'], null))], ['下一次计划', time(first(s, ['next_run_at', 'next_training_at'], null))], ['失败原因', first(s, ['failure_reason', 'error'], '没有记录失败')]];
    $('#scheduler').classList.remove('empty'); $('#scheduler').innerHTML = values.map(([a, b]) => line(a, b)).join('');
  }
  function renderModels() {
    const registry = unwrap(payloads.registry), health = unwrap(payloads.health), competition = unwrap(payloads.competition);
    const list = first(registry, ['models', 'items', 'entries'], []) || first(health, ['models', 'items', 'entries'], []) || first(competition, ['models', 'items', 'entries'], []) || [];
    if (!list.length) { $('#model-table').innerHTML = '<tr><td colspan="7" class="empty">暂时没有模型表现记录。</td></tr>'; return; }
    $('#model-table').innerHTML = list.slice(0, 30).map((m) => { const name = first(m, ['name', 'model_name', 'id'], '未命名模型'), role = first(m, ['role', 'purpose', 'description', 'capability'], '职责待补充'), samples = first(m, ['sample_count', 'trainable_sample_count', 'samples'], null), ret = first(m, ['fee_after_return', 'net_return_after_cost', 'average_net_return_after_cost_pct'], null), dd = first(m, ['max_drawdown', 'drawdown'], null), tail = first(m, ['tail_loss', 'downside_mean', 'lower_hinge'], null), weight = first(m, ['influence', 'weight', 'effective_weight'], null), reason = first(m, ['weight_change_reason', 'reason', 'status_reason'], '暂无变化说明'); return `<tr><td><span class="model-name">${esc(name)}</span><span class="model-role">${esc(role)}</span></td><td>${esc(first(m, ['specialty', 'capability'], role))}</td><td>${fmt(samples)}</td><td class="${Number(ret) >= 0 ? 'positive' : 'negative'}">${Number.isFinite(Number(ret)) ? `${fmt(ret, 4)}%` : '--'}</td><td>${dd === null ? '--' : fmt(dd, 3)} / ${tail === null ? '--' : fmt(tail, 3)}</td><td>${weight === null ? '--' : Number(weight) <= 1 ? percent(weight) : fmt(weight, 3)}</td><td>${esc(reason)}</td></tr>`; }).join('');
  }
  function renderStrategies() {
    const s = unwrap(payloads.strategy), list = first(s, ['strategies', 'candidates', 'items'], []) || [], fallback = first(s, ['current_strategy', 'active_strategy'], null);
    const rows = list.length ? list : fallback ? [fallback] : []; if (!rows.length) { $('#strategies').textContent = '暂时没有策略状态。'; return; }
    $('#strategies').classList.remove('empty'); $('#strategies').innerHTML = rows.slice(0, 12).map((m) => line(first(m, ['name', 'strategy_name'], '当前策略'), `${first(m, ['status', 'market_regime', 'state'], '观察中')} · 费后收益 ${first(m, ['fee_after_return', 'net_return'], '--')} · ${first(m, ['reason', 'description'], '暂无说明')}`)).join('');
  }
  function renderDecisions() {
    const raw = unwrap(payloads.decisions), list = first(raw, ['items', 'records', 'decisions'], Array.isArray(raw) ? raw : []) || [];
    if (!list.length) { $('#recent-decisions').textContent = '暂时没有分析记录。'; return; }
    $('#recent-decisions').classList.remove('empty'); $('#recent-decisions').innerHTML = list.slice(0, 12).map((m) => { const action = String(first(m, ['action', 'decision', 'decision_or_action'], 'hold')).toLowerCase(), cls = action.includes('long') ? 'long' : action.includes('short') ? 'short' : 'hold'; return `<div class="decision ${cls}"><div><strong>${esc(first(m, ['symbol', 'instrument'], '未知币种'))} · ${esc(action)}</strong><p>${esc(first(m, ['reason', 'reasoning', 'execution_reason'], '暂无原因'))}</p></div><time>${esc(time(first(m, ['created_at', 'timestamp', 'analyzed_at'], null)))}</time></div>`; }).join('');
  }
  async function fetchJson(url) { const r = await fetch(url, { credentials: 'same-origin', headers: { Accept: 'application/json' } }); if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); }
  async function refresh() {
    setState('正在刷新'); $('#error-text').hidden = true;
    const results = await Promise.allSettled(Object.entries(urls).map(async ([key, url]) => [key, await fetchJson(url)])), failures = [];
    results.forEach((r) => r.status === 'fulfilled' ? payloads[r.value[0]] = r.value[1] : failures.push(r.reason?.message || '接口读取失败'));
    renderOverview(); renderQuality(); renderScheduler(); renderModels(); renderStrategies(); renderDecisions();
    $('#updated-at').textContent = `更新于 ${time(new Date())}`;
    if (failures.length) { setState('部分数据不可用', 'warn'); $('#error-text').textContent = failures.slice(0, 2).join('；'); $('#error-text').hidden = false; } else setState('运行正常', 'ok');
  }
  $('#refresh-button').addEventListener('click', refresh); refresh(); window.setInterval(refresh, 60_000);
})();
