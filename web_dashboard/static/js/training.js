(() => {
  const $ = (selector) => document.querySelector(selector);
  const payloads = {};
  const requestErrors = {};
  let refreshSequence = 0;
  const requestTimeoutMs = Object.freeze({
    ml: 30_000, registry: 45_000, scheduler: 30_000,
    data: 45_000, strategy: 20_000, decisions: 30_000,
  });
  const urls = {
    ml: '/api/ml-signal/status',
    registry: '/api/model-training/registry',
    scheduler: '/api/model-training/scheduler',
    data: '/api/data-collection/status?include_feature_coverage=false',
    // The strategy view is diagnostic only; a bounded recent window keeps it
    // from competing with trading/analysis work during page refreshes.
    strategy: '/api/strategy-learning?hours=24&limit=50&detail=summary',
    decisions: '/api/analysis-records?limit=12&page=1',
  };
  const endpointLabels = {
    ml: '本地 ML 状态', registry: '模型注册表', scheduler: '训练调度', data: '训练数据治理',
    strategy: '策略学习', decisions: '最近分析',
  };
  const reasonText = Object.freeze({
    no_model: '尚未注册当前模型 Artifact',
    artifact_incompatible: '当前 Artifact 与运行时收益监督合同不兼容，已禁止加载',
    artifact_load_failed: '当前 Artifact 加载失败，已禁止用于预测',
    degraded: '模型可用，但收益证据尚未达到生产晋升标准',
    disabled: '模型已禁用',
    learning_only: '仅学习观察，不影响生产交易',
    ready: '收益证据已达到生产就绪标准',
    shadow_ready: '影子观察就绪',
    paper_canary_ready: '模型处于 Paper Canary 生命周期；模拟盘正常交易，实盘仍未授权',
    artifact_activation_not_production_authorized: '当前 Artifact 可参与模拟盘，尚未获得实盘权限',
    average_net_return_after_all_cost_not_positive: '平均费后收益不为正',
    average_fee_after_return_not_positive: '平均费后收益不为正',
    empirical_return_lower_hinge_not_positive: '费后收益经验下界不为正',
    profit_factor_not_above_break_even: '盈亏比没有高于自然盈亏平衡线 1',
    profit_factor_below_unity: '盈亏比低于自然盈亏平衡线 1',
    profit_factor_undefined: '缺少亏损分母，盈亏比暂时无法计算',
    authoritative_execution_cost_distribution_missing: '缺少权威真实执行成本分布',
    authoritative_slippage_distribution_missing: '缺少权威真实滑点分布',
    authoritative_realized_return_distribution_missing: '缺少权威真实成交收益分布',
    model_specific_fee_after_attribution_missing: '缺少该模型独立的费后归因证据',
    high_risk_review_fee_after_evaluation_missing: '缺少高风险复核模型的费后评估',
    decision_llm_fee_after_evaluation_missing: '缺少决策模型的费后评估',
    finquant_specialization_missing: '缺少可验证的项目专用训练产物',
    realized_net_pnl_non_positive: '权威已实现净收益不为正',
    separated_supervision_distribution_unavailable: '缺少来自不同决策组的市场机会与执行成本监督样本',
    market_opportunity_distribution_unavailable: '缺少可按决策组切分的固定窗口市场机会标签',
    authoritative_execution_cost_distribution_unavailable: '缺少带入场特征的 OKX 权威手续费、滑点和资金费样本',
    chronological_market_training_identity_incomplete: '市场机会样本缺少完整的决策时间或标签时间身份',
    chronological_cost_training_identity_incomplete: '权威执行成本样本缺少完整的开仓时间、结算时间或生命周期身份',
    trained_challenger_rejected: '新挑战模型已完成训练，但费后收益未超过当前模型，继续保留当前模型',
    dynamic_exit_pressure_zero: '当前退出压力为零，继续持有',
    error: '运行异常',
    warning: '需要关注',
    unavailable: '当前不可用',
  });
  const lifecycleText = Object.freeze({
    active: '已激活', live: '已介入生产', canary: 'Paper Canary', trained: '已训练',
    training: '训练中', promotion_blocked: '晋升阻断', shadow_evaluating: '影子评估', diagnostic_timeout: '诊断查询超时',
    inference_only: '仅推理', not_trained: '未训练', not_evaluated: '未评估',
    service_unavailable: '服务不可用', unavailable: '不可用', running: '运行中',
    warning: '需要关注', error: '运行异常', ok: '正常',
  });
  const taskText = Object.freeze({
    after_cost_entry_profit_quality: '费后开仓收益质量',
    after_cost_long_short_expected_return: '多空费后预期收益',
    side_specific_loss_probability: '分方向亏损概率',
    multi_horizon_return_forecast: '多周期收益预测',
    sequence_return_forecast: '序列收益预测',
    event_sentiment_return_calibration: '事件情绪收益校准',
    position_exit_attribution: '持仓退出归因',
    pretrained_timeseries_forecast: '预训练时序预测',
    pretrained_timeseries_challenger: '预训练时序挑战模型',
    pretrained_sentiment_inference: '预训练情绪推理',
    pretrained_sentiment_challenger: '预训练情绪挑战模型',
    quant_expert_reasoning: '量化专家推理',
    trade_reasoning_fallback: '交易推理后备模型',
    risk_review: '高风险复核',
    final_decision: '最终决策',
  });
  const schedulerText = Object.freeze({
    local_ai_tools_auto_train: '本地量化工具自动训练',
    local_ml_auto_train: '本地 ML 自动训练',
  });
  const modelText = Object.freeze({
    local_ml_profit_quality: ['本地 ML 费后收益质量', '本地分类与收益回归模型'],
    local_ai_profit_prediction: ['本地 AI 收益预测', '本地多空收益回归模型'],
    local_ai_loss_filter: ['本地 AI 亏损过滤', '本地分方向亏损分类模型'],
    local_ai_timeseries: ['本地 AI 多周期时序', '本地多周期收益分布模型'],
    local_ai_sequence: ['本地 AI 序列模型', '本地深度时序模型'],
    local_ai_sentiment_calibration: ['本地 AI 情绪校准', '本地文本情绪与收益校准模型'],
    local_ai_exit_profile: ['本地 AI 退出画像', '持仓退出归因与规则模型'],
    timesfm_2_5: ['TimesFM 2.5 时序预测', '预训练时序基础模型'],
    chronos_2: ['Chronos-2 时序挑战模型', '预训练时序基础模型'],
    finbert: ['FinBERT 财经情绪', '预训练财经情绪模型'],
    finbert_tone: ['FinBERT Tone 情绪挑战模型', '预训练财经情绪模型'],
    bb_finquant_expert_14b: ['BB 量化专家 14B', '项目专用量化推理模型'],
    qwen3_14b_trade: ['Qwen3 交易决策 14B', '交易推理后备模型'],
    deepseek_r1_14b_risk: ['DeepSeek 风险复核 14B', '高风险交易复核模型'],
    deepseek_online_decision: ['线上 DeepSeek 最终决策', '托管式最终决策模型'],
  });

  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[char]));
  const unwrap = (value) => value?.data ?? value?.result ?? value ?? {};
  const present = (value) => value !== null && value !== undefined && value !== '';
  const number = (value) => present(value) && Number.isFinite(Number(value)) ? Number(value) : null;
  const fmt = (value, digits = 0) => {
    const parsed = number(value);
    return parsed === null ? '未提供' : parsed.toLocaleString('zh-CN', { maximumFractionDigits: digits });
  };
  const pct = (value, digits = 3) => {
    const parsed = number(value);
    return parsed === null ? '未提供' : `${parsed.toFixed(digits)}%`;
  };
  const ratio = (value, digits = 2) => {
    const parsed = number(value);
    return parsed === null ? '未提供' : `${(parsed * 100).toFixed(digits)}%`;
  };
  const time = (value) => {
    if (!value) return '未提供';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString('zh-CN', { hour12: false });
  };
  const row = (label, value, tone = '') => `<div class="row ${tone}"><span class="row-label">${esc(label)}</span><span class="row-value">${esc(value)}</span></div>`;
  const setText = (selector, value) => { const element = $(selector); if (element) element.textContent = value; };
  const setState = (label, kind = '') => {
    const element = $('#page-state');
    if (!element) return;
    element.textContent = label;
    element.className = `state ${kind ? `state-${kind}` : ''}`;
  };

  function localizedReason(value) {
    const item = value && typeof value === 'object' ? value : {};
    const code = String(item.code || item.reason || (typeof value === 'string' ? value : '') || '').trim();
    if (reasonText[code]) return reasonText[code];
    const sideMatch = code.match(/^(long|short)_(.+)$/);
    if (sideMatch) {
      const side = sideMatch[1] === 'long' ? '做多' : '做空';
      const suffixes = {
        walk_forward_return_stability_failed: '费后收益在时间滚动验证中不稳定',
        market_regime_stability_failed: '费后收益未能在至少两种行情状态下稳定为正',
        leave_one_symbol_out_stability_failed: '收益过度依赖单一币种',
        oos_profit_factor_not_above_break_even: '样本外盈亏比没有高于自然盈亏平衡线 1',
        oos_return_tail_evidence_incomplete: '样本外尾部风险证据不完整',
        top_return_lcb_not_positive: '高分组收益置信下界不为正',
        authoritative_return_evidence_not_ready: '真实成交收益证据尚未就绪',
      };
      if (suffixes[sideMatch[2]]) return `${side}${suffixes[sideMatch[2]]}`;
    }
    if (/^[a-z][a-z0-9_:-]*$/i.test(code)) return `系统诊断代码：${code}`;
    return String(item.message || code || '原因尚未返回');
  }

  function visibleModelState(ml) {
    const diagnostic = ml.model_load_diagnostic || {};
    const code = diagnostic.code || ml.status || 'unavailable';
    if (ml.available !== true) return localizedReason(code);
    if (ml.live_ml_ready === true) return '已生产就绪';
    if (ml.paper_trading_permission === true) return '模拟盘可用，实盘未晋升';
    if (code === 'degraded' || ml.readiness_state === 'degraded') return '模型可用，实盘证据未达标';
    return localizedReason(code);
  }

  function trainingData() {
    const root = unwrap(payloads.data);
    const training = root.training || {};
    return { root, training, governance: training.governance || {}, localTools: training.local_ai_tools || {} };
  }

  function renderOverview() {
    const ml = unwrap(payloads.ml);
    const scheduler = unwrap(payloads.scheduler);
    const { governance, localTools } = trainingData();
    const quality = localTools.quality_report || {};
    const totals = quality.totals || {};
    const trusted = number(governance.training_shadow_sample_count)
      ?? number(governance.local_ml_signal?.current_epoch_trainable_sample_count)
      ?? number(localTools.training_shadow_sample_count)
      ?? number(ml.training_shadow_sample_count)
      ?? number(ml.completed_shadow_sample_count)
      ?? number(totals.included);
    const pending = number(ml.new_shadow_sample_count);
    const quarantined = number(governance.quarantined_shadow_sample_count)
      ?? number(governance.local_ml_signal?.quarantined_sample_count)
      ?? number(ml.quality_report?.totals?.excluded)
      ?? number(totals.excluded);
    setText('[data-metric="training-status"]', visibleModelState(ml));
    setText('[data-detail="training-status"]', scheduler.status ? `调度：${lifecycleText[scheduler.status] || scheduler.status}` : '训练调度状态未提供');
    setText('[data-metric="trusted-samples"]', fmt(trusted));
    setText('[data-detail="trusted-samples"]', trusted === null ? '接口未提供可信训练样本数' : '当前训练纪元可追溯样本');
    setText('[data-metric="pending-samples"]', fmt(pending));
    setText('[data-detail="pending-samples"]', pending === null ? '接口未提供新增样本游标' : '相对最近已训练游标');
    setText('[data-metric="quarantined-samples"]', fmt(quarantined));
    setText('[data-metric="model-influence"]', ml.paper_trading_permission === true ? '模拟盘允许' : '模拟盘不可用');
    setText('[data-detail="model-influence"]', ml.live_ml_ready === true ? '实盘候选已就绪，逐笔生产门禁仍生效' : '实盘未授权；不阻断模拟盘分析和正常交易');
    setText('[data-metric="last-training"]', time(ml.trained_at));
    setText('[data-detail="last-training"]', ml.artifact_version ? `Artifact ${ml.artifact_version}` : 'Artifact 版本未提供');
    renderTrainingEvidence();
  }

  function renderTrainingEvidence() {
    const container = $('#training-evidence');
    if (!container) return;
    const ml = unwrap(payloads.ml);
    const registry = unwrap(payloads.registry);
    const scheduler = unwrap(payloads.scheduler);
    const registryRoot = ml.artifact_registry || {};
    const currentPointer = artifactPointer(registryRoot, 'current');
    const challengerPointer = artifactPointer(registryRoot, 'challenger');
    const currentManifest = artifactManifest(currentPointer);
    const challengerManifest = artifactManifest(challengerPointer);
    const activation = ml.artifact_activation_manifest || currentManifest.activation_manifest || ml.artifact_registry?.activation_manifest || {};
    const latestTrainingVersion = challengerPointer?.version
      || ml.artifact_version
      || ml.version;
    const challenger = ml.challenger_artifact_version
      || ml.latest_challenger_version
      || challengerPointer?.version
      || activation.challenger_version;
    const active = ml.active_artifact_version
      || ml.active_version
      || currentPointer?.version
      || activation.active_version
      || (ml.live_ml_ready ? ml.artifact_version : null);
    const evaluated = ml.latest_evaluation_version
      || ml.evaluation_artifact_version
      || challengerPointer?.version
      || ml.artifact_registry?.latest_evaluation_version
      || (ml.evaluation_status ? ml.artifact_version : null);
    const baseline = ml.no_model_baseline || ml.baseline || {};
    const modelSummary = registry.summary || {};
    const contributionStatus = registry.contribution_performance_status || {};
    const schedulerState = scheduler.models || scheduler.schedulers || {};
    const schedulerRows = Object.values(schedulerState).filter(item => item && typeof item === 'object');
    const lastResult = schedulerRows.map(item => item.last_result || {}).find(item => Object.keys(item).length) || {};
    const values = [
      ['最新训练版本', latestTrainingVersion || '证据缺失'],
      ['最新评估版本', evaluated || '尚未评估'],
      ['当前 active', active || '未授权 active'],
      ['候选 challenger', challenger || '暂无 challenger'],
      ['无模型基线', baseline.version || baseline.name || '规则基线待提供'],
      ['评估状态', ml.evaluation_status || ml.readiness?.state || '未提供'],
      ['训练结果', lastResult.reason || lastResult.error || (lastResult.trained ? 'trained' : 'not_due / 未触发')],
      ['下次检查', schedulerRows.map(item => item.next_check_at || item.next_run_at).find(Boolean) || '未提供'],
      ['模型注册数', modelSummary.model_count ?? '未提供'],
      ['贡献归因查询', contributionStatus.state === 'timeout' ? '已限时降级' : contributionStatus.state || '未提供'],
      ['模拟盘权限', ml.paper_trading_permission === true ? '允许' : '未允许'],
      ['实盘权限', ml.live_trading_permission === true ? '允许' : '未晋级'],
      ['数据时间', ml.trained_at || ml.checked_at || '未提供'],
    ];
    container.innerHTML = values.map(([label, value]) => evidenceItem(label, value)).join('');
    renderTrainingComparison(ml, registry);
  }

  function artifactPointer(registry, role) {
    const pointers = registry?.pointers || registry?.artifact_registry?.pointers || {};
    const pointer = pointers[role];
    return pointer && typeof pointer === 'object' && pointer.available !== false ? pointer : null;
  }

  function artifactManifest(pointer) {
    if (!pointer || typeof pointer !== 'object') return {};
    const manifest = pointer.manifest;
    return manifest && typeof manifest === 'object' ? manifest : {};
  }

  function artifactMetrics(pointer) {
    const manifest = artifactManifest(pointer);
    const source = manifest.metadata && typeof manifest.metadata === 'object'
      ? manifest.metadata
      : manifest;
    const oos = source.oos_return_evaluation && typeof source.oos_return_evaluation === 'object'
      ? source.oos_return_evaluation
      : {};
    const metrics = source.metrics && typeof source.metrics === 'object' ? source.metrics : {};
    const side = (name) => {
      const row = oos[name] && typeof oos[name] === 'object' ? oos[name] : {};
      return {
        avg_return_pct: row.avg_return_pct ?? metrics[`top_${name}_avg_return_pct`],
        return_lcb_pct: row.return_lcb_pct ?? metrics[`top_${name}_return_lcb_pct`],
        profit_factor: row.profit_factor ?? metrics[`top_${name}_profit_factor`],
        cvar_10_pct: row.cvar_10_pct ?? metrics[`top_${name}_cvar_10_pct`],
        max_drawdown_pct: row.max_drawdown_pct,
        sample_count: row.sample_count,
      };
    };
    return { long: side('long'), short: side('short'), version: pointer?.version || source.artifact_version || source.version };
  }

  function metricCell(metrics, key) {
    const sideText = (side) => {
      const value = metrics?.[side]?.[key];
      if (!present(value)) return '未提供';
      if (key === 'profit_factor') return Number(value).toFixed(2);
      if (key === 'sample_count') return fmt(value);
      return `${Number(value).toFixed(3)}%`;
    };
    return `多 ${sideText('long')} / 空 ${sideText('short')}`;
  }

  function renderTrainingComparison(ml, registry) {
    const container = $('#training-comparison');
    if (!container) return;
    const registryRoot = ml.artifact_registry || registry?.artifact_registry || registry || {};
    const activePointer = artifactPointer(registryRoot, 'current')
      || artifactPointer(registryRoot, 'active');
    const challengerPointer = artifactPointer(registryRoot, 'challenger');
    const baselineSource = ml.no_model_baseline || ml.baseline
      || registry.no_model_baseline || registry.baseline;
    const baseline = baselineSource && typeof baselineSource === 'object' ? baselineSource : null;
    const columns = [
      ['当前 active', artifactMetrics(activePointer)],
      ['候选 challenger', artifactMetrics(challengerPointer)],
      ['无模型基线', baseline ? artifactMetrics({ manifest: baseline, version: baseline.version }) : null],
    ];
    const rows = [
      ['费后平均收益', 'avg_return_pct'],
      ['收益下置信界限', 'return_lcb_pct'],
      ['Profit Factor', 'profit_factor'],
      ['CVaR10', 'cvar_10_pct'],
      ['最大回撤', 'max_drawdown_pct'],
      ['样本数', 'sample_count'],
    ];
    const header = columns.map(([label, value]) => `<th>${esc(label)}<small>${esc(value?.version || '版本未提供')}</small></th>`).join('');
    const body = rows.map(([label, key]) => `<tr><th>${esc(label)}</th>${columns.map(([, value]) => `<td>${esc(value ? metricCell(value, key) : '证据未提供')}</td>`).join('')}</tr>`).join('');
    container.innerHTML = `
      <div class="comparison-heading"><strong>样本外费后结果对照</strong><span>多空分开；缺失值不补零</span></div>
      <div class="table-wrap"><table class="comparison-table"><thead><tr><th>指标</th>${header}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  function renderQuality() {
    const ml = unwrap(payloads.ml);
    const { governance, localTools } = trainingData();
    const quality = localTools.quality_report || ml.quality_report || {};
    const totals = quality.totals || {};
    const shadow = quality.by_kind?.shadow || {};
    const trade = quality.by_kind?.trade || {};
    const supervision = ml.profit_supervision_report || quality.profit_supervision || {};
    const tasks = ml.training_task_manifest || {};
    const replay = ml.replay_weight_manifest || {};
    const policy = ml.auto_train_last_result?.training_policy || {};
    const values = [
      ['治理状态', governance.status === 'ok' ? '治理快照正常' : localizedReason(governance.status || 'unavailable')],
      ['数据质量版本', quality.data_quality_version || '未提供'],
      ['全部 / 纳入 / 隔离', `${fmt(totals.total)} / ${fmt(totals.included)} / ${fmt(totals.excluded)}`],
      ['影子样本 / 真实成交样本', `${fmt(shadow.total)} / ${fmt(trade.total)}`],
      ['有效训练权重', ratio(totals.effective_weight_ratio)],
      ['市场机会 / 反事实成本 / 权威收益', `${fmt(supervision.shadow_market_sample_count)} / ${fmt(supervision.shadow_counterfactual_cost_sample_count)} / ${fmt(supervision.actual_realized_return_sample_count)}`],
      ['隔离原因类型', fmt(Array.isArray(quality.top_reasons) ? quality.top_reasons.length : null)],
      ['训练策略', governance.training_policy || localTools.training_policy || '未提供'],
      ['机会 / 入场任务样本', `${fmt(tasks.market_opportunity?.sample_count)} / ${fmt(tasks.entry_timing?.sample_count)}`],
      ['退出 / 执行任务样本', `${fmt(tasks.exit?.sample_count)} / ${fmt(tasks.execution?.sample_count)}`],
      ['成熟独立决策组', fmt(ml.completed_shadow_decision_group_count ?? policy.completed_mature_decision_group_count)],
      ['新增独立决策组', fmt(policy.new_mature_decision_group_count)],
      ['重放池有效样本量', fmt(replay.effective_sample_size, 2)],
      ['标签合同', Array.isArray(ml.label_contract_versions) ? ml.label_contract_versions.join('，') : '未提供'],
    ];
    const container = $('#data-quality');
    container.classList.remove('empty');
    container.innerHTML = values.map(([label, value]) => row(label, value)).join('');
  }

  function renderScheduler() {
    const scheduler = unwrap(payloads.scheduler);
    const entries = Object.entries(scheduler.models || scheduler.schedulers || {})
      .sort(([left], [right]) => Number(right === 'local_ml_profit_quality') - Number(left === 'local_ml_profit_quality'));
    const lines = [
      row('当前状态', lifecycleText[scheduler.status] || scheduler.status || '未提供'),
      row('状态更新时间', time(scheduler.updated_at)),
    ];
    (entries.length ? entries : [['全局调度', scheduler]]).forEach(([id, item]) => {
      const last = item.last_result || {};
      const result = last.reason || last.error || item.last_status || last.message || '尚无完成结果';
      const label = modelText[id]?.[0] || schedulerText[id] || id;
      lines.push(row(label, `${localizedReason(result)}；最近运行 ${time(item.last_started_at || item.last_run_at || item.updated_at)}；下次 ${time(item.next_run_at || item.next_check_at)}`));
    });
    const container = $('#scheduler');
    container.classList.remove('empty');
    container.innerHTML = lines.join('');
  }

  function evidenceItem(label, value, tone = '') {
    return `<div class="evidence-item ${tone}"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
  }

  function renderReadiness() {
    const ml = unwrap(payloads.ml);
    const readiness = ml.readiness || {};
    const blockers = Array.isArray(readiness.blocking_reasons) ? readiness.blocking_reasons : [];
    const walkForward = ml.walk_forward_report || {};
    const symbol = ml.leave_one_symbol_out_report || {};
    const authoritative = ml.authoritative_trade_return_evidence || {};
    const activation = ml.artifact_activation_manifest || ml.artifact_registry?.activation_manifest || {};
    const diagnostic = ml.model_load_diagnostic || {};
    const stateTone = ml.available !== true ? 'bad' : ml.live_ml_ready === true ? 'good' : 'warn';
    const blockerHtml = blockers.length
      ? `<div class="blocker-list">${blockers.map((item) => `<div><span>${esc(item.code || '阻断')}</span><strong>${esc(localizedReason(item))}</strong></div>`).join('')}</div>`
      : `<div class="empty-line">${ml.available === true ? '当前接口没有返回晋升阻断。' : esc(localizedReason(diagnostic.code || ml.status || 'unavailable'))}</div>`;
    const container = $('#model-readiness');
    container.classList.remove('empty');
    container.innerHTML = `
      <div class="readiness-verdict ${stateTone}">
        <div><span>当前判断</span><strong>${esc(visibleModelState(ml))}</strong></div>
        <p>${esc(diagnostic.message || (ml.available === true ? '模型已加载；是否晋升只由费后收益、未来时间验证和稳定性证据决定。' : '运行时已故障关闭，不使用不可验证的模型产物。'))}</p>
      </div>
      <div class="evidence-grid">
        ${evidenceItem('Artifact 版本', ml.artifact_version || '未提供')}
        ${evidenceItem('激活阶段', lifecycleText[activation.activation_stage] || activation.activation_stage || ml.artifact_lifecycle || '未提供')}
        ${evidenceItem('模拟盘交易权限', ml.paper_trading_permission === true ? '允许' : '不可用', ml.paper_trading_permission === true ? 'good' : 'warn')}
        ${evidenceItem('实盘候选权限', ml.live_trading_permission === true ? '允许逐笔检查' : '未晋升', ml.live_trading_permission === true ? 'good' : 'warn')}
        ${evidenceItem('训练 / 留出决策组', `${fmt(ml.train_decision_group_count)} / ${fmt(ml.test_decision_group_count)}`)}
        ${evidenceItem('Walk-forward', `${walkForward.status === 'complete' ? '已完成' : walkForward.status || '未提供'} · ${fmt(walkForward.fold_count ?? walkForward.folds?.length)} 折`)}
        ${evidenceItem('做多逐币移除稳定', present(symbol.long?.stable) ? (symbol.long.stable ? '通过' : '未通过') : '未提供', symbol.long?.stable === false ? 'warn' : '')}
        ${evidenceItem('做空逐币移除稳定', present(symbol.short?.stable) ? (symbol.short.stable ? '通过' : '未通过') : '未提供', symbol.short?.stable === false ? 'warn' : '')}
        ${evidenceItem('权威成交收益样本', fmt(authoritative.sample_count))}
        ${evidenceItem('权威证据指纹', authoritative.data_fingerprint || '未提供')}
      </div>
      <div class="blocker-heading"><strong>真实晋升阻断</strong><span>${blockers.length} 项</span></div>
      ${blockerHtml}`;
  }

  function modelFeeAfter(model, ml) {
    if (model.model_id === 'local_ml_profit_quality') {
      if (ml.available !== true) return '未评估';
      const metrics = ml.readiness?.metrics || {};
      return `多 ${pct(metrics.top_long_avg_return_pct)} / 空 ${pct(metrics.top_short_avg_return_pct)}`;
    }
    const evaluated = number(model.evaluation_sample_count) > 0
      || number(model.authoritative_sample_count) > 0
      || number(model.actual_inference_count) > 0;
    if (!evaluated) return '未评估';
    const value = number(model.net_return_after_all_cost_pct)
      ?? number(model.avg_realized_net_pnl_usdt) ?? number(model.realized_net_pnl_usdt);
    return value === null ? '注册表未提供' : (present(model.net_return_after_all_cost_pct) ? pct(value) : `${fmt(value, 4)} USDT`);
  }

  function modelProfitFactor(model, ml) {
    if (model.model_id === 'local_ml_profit_quality') {
      if (ml.available !== true) return '未评估';
      const metrics = ml.readiness?.metrics || {};
      return `多 ${fmt(metrics.top_long_profit_factor, 3)} / 空 ${fmt(metrics.top_short_profit_factor, 3)}`;
    }
    const evaluated = number(model.evaluation_sample_count) > 0
      || number(model.authoritative_sample_count) > 0
      || number(model.actual_inference_count) > 0;
    if (!evaluated) return '未评估';
    return fmt(model.authoritative_profit_factor ?? model.profit_factor, 3);
  }

  function modelSampleText(model) {
    const samples = number(model.sample_count);
    if (samples === null) return '未提供';
    if (samples === 0 && model.artifact_available !== true) return '未训练或未评估';
    return fmt(samples);
  }

  function renderModels() {
    const registry = unwrap(payloads.registry);
    const ml = unwrap(payloads.ml);
    const models = Array.isArray(registry.models) ? registry.models : [];
    const table = $('#model-table');
    if (!models.length) {
      table.innerHTML = '<tr><td colspan="8" class="empty">模型注册表没有返回记录。</td></tr>';
      return;
    }
    table.innerHTML = models.map((model) => {
      const blockers = Array.isArray(model.blocking_reasons) ? model.blocking_reasons : [];
      const influence = model.live_ml_ready
        ? '模拟盘正常参与；实盘逐笔门禁'
        : model.trainable ? '模拟盘正常参与；实盘未授权' : '仅推理或评估';
      const localizedModel = modelText[model.model_id] || [];
      return `<tr>
        <td><span class="model-name">${esc(localizedModel[0] || model.display_name || model.model_id || '未命名模型')}</span><span class="model-role">${esc(localizedModel[1] || '模型类型未登记')}</span></td>
        <td>${esc(lifecycleText[model.lifecycle] || model.lifecycle || '未提供')}</td>
        <td>${esc(taskText[model.task] || model.task || taskText[model.runtime_role] || model.runtime_role || '未提供')}</td>
        <td>${esc(modelSampleText(model))}</td><td>${esc(modelFeeAfter(model, ml))}</td><td>${esc(modelProfitFactor(model, ml))}</td>
        <td>${esc(influence)}</td>
        <td class="model-reason">${esc(blockers.length ? blockers.map(localizedReason).join('；') : `质量状态：${lifecycleText[model.quality_state] || localizedReason(model.quality_state || '未提供')}`)}</td>
      </tr>`;
    }).join('');
  }

  function renderStrategies() {
    const strategy = unwrap(payloads.strategy);
    const container = $('#strategies');
    if (strategy.status === 'timeout') {
      container.classList.remove('empty');
      container.innerHTML = [
        row('当前状态', '查询已限时结束'),
        row('原因', '策略学习数据库查询超过 12 秒，已停止本次查询'),
        row('影响范围', '不影响模型、开仓、平仓或训练；页面不会重复发起'),
      ].join('');
      return;
    }
    const schedule = strategy.schedule || {};
    const production = strategy.current_production_strategy || schedule.current_production_strategy || {};
    const champion = strategy.paper_strategy_champion || {};
    const leading = schedule.leading_candidate || {};
    const values = [
      ['当前生产策略', production.name || production.id || '未提供'],
      ['生产策略状态', lifecycleText[production.status] || production.status || '未提供'],
      ['模拟盘冠军策略', champion.name || champion.strategy_id || champion.status || '尚未形成'],
      ['领先挑战策略', leading.name || leading.strategy_id || leading.id || '尚未形成'],
      ['候选 / 通过治理 / 拒绝', `${fmt(schedule.candidate_count)} / ${fmt(schedule.governed_candidate_count)} / ${fmt(schedule.rejected_candidate_count)}`],
      ['调度原因', localizedReason(schedule.reason || '原因尚未返回')],
    ];
    container.classList.remove('empty');
    container.innerHTML = values.map(([label, value]) => row(label, value)).join('');
  }

  function renderDecisions() {
    const raw = unwrap(payloads.decisions);
    const records = Array.isArray(raw.records) ? raw.records : [];
    const container = $('#recent-decisions');
    if (!records.length) {
      container.classList.add('empty');
      container.textContent = '最近分析接口没有返回记录。';
      return;
    }
    container.classList.remove('empty');
    container.innerHTML = records.slice(0, 12).map((record) => {
      const action = String(record.action || record.decision || record.decision_or_action || 'hold').toLowerCase();
      const tone = action.includes('long') ? 'long' : action.includes('short') ? 'short' : 'hold';
      const actionText = action.includes('long') ? '做多' : action.includes('short') ? '做空' : '观望';
      const rawReason = record.reason || record.reasoning || record.execution_reason || '原因未提供';
      const reason = String(rawReason).includes('dynamic_exit_pressure_zero')
        ? reasonText.dynamic_exit_pressure_zero : localizedReason(rawReason);
      return `<div class="decision ${tone}"><div><strong>${esc(record.symbol || '未知币种')} · ${actionText}</strong><p>${esc(reason)}</p></div><time>${esc(time(record.created_at || record.timestamp || record.analyzed_at))}</time></div>`;
    }).join('');
  }

  function renderEndpoint(key) {
    if (key === 'ml') {
      renderOverview(); renderReadiness();
      if (payloads.registry) renderModels();
    } else if (key === 'registry') {
      renderModels();
      renderTrainingEvidence();
    } else if (key === 'scheduler') {
      renderScheduler();
      renderTrainingEvidence();
      if (payloads.ml) renderOverview();
    } else if (key === 'data') {
      renderQuality();
      if (payloads.ml) renderOverview();
    } else if (key === 'strategy') {
      renderStrategies();
    } else if (key === 'decisions') {
      renderDecisions();
    }
  }

  function renderEndpointError(key, message) {
    const text = `${endpointLabels[key]}数据不可用：${message}`;
    if (key === 'ml') {
      setText('[data-metric="training-status"]', '读取失败');
      setText('[data-detail="training-status"]', text);
      const readiness = $('#model-readiness');
      readiness.classList.add('empty');
      readiness.textContent = text;
    } else if (key === 'registry') {
      $('#model-table').innerHTML = `<tr><td colspan="8" class="empty">${esc(text)}</td></tr>`;
    } else {
      const containers = { scheduler: '#scheduler', data: '#data-quality', strategy: '#strategies', decisions: '#recent-decisions' };
      const container = $(containers[key]);
      if (container) {
        container.classList.add('empty');
        container.textContent = text;
      }
    }
  }

  async function fetchJson(url, timeoutMs) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        credentials: 'same-origin',
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return await response.json();
    } catch (error) {
      if (error?.name === 'AbortError') throw new Error(`请求超过 ${Math.round(timeoutMs / 1000)} 秒未完成`);
      throw error;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async function refresh() {
    const sequence = ++refreshSequence;
    setState('正在刷新');
    $('#error-text').hidden = true;
    const entries = Object.entries(urls);
    const results = await Promise.allSettled(entries.map(async ([key, url]) => {
      const value = await fetchJson(url, requestTimeoutMs[key] || 30_000);
      if (sequence === refreshSequence) {
        payloads[key] = value;
        delete requestErrors[key];
        renderEndpoint(key);
      }
      return [key, value];
    }));
    if (sequence !== refreshSequence) return;
    results.forEach((result, index) => {
      const key = entries[index][0];
      if (result.status === 'rejected') {
        requestErrors[key] = result.reason?.message || '请求失败';
        if (!payloads[key]) renderEndpointError(key, requestErrors[key]);
      }
    });
    setText('#updated-at', `更新于 ${time(new Date())}`);
    const failures = Object.entries(requestErrors).map(([key, message]) => `${endpointLabels[key]}：${message}`);
    if (failures.length) {
      setState('部分数据不可用', 'warn');
      $('#error-text').textContent = failures.slice(0, 3).join('；');
      $('#error-text').hidden = false;
    } else {
      const ml = unwrap(payloads.ml);
      setState(ml.available === true ? (ml.live_ml_ready === true ? '模型已就绪' : '模型学习观察中') : '模型不可用', ml.available === true ? (ml.live_ml_ready === true ? 'ok' : 'warn') : 'error');
    }
  }

  $('#refresh-button').addEventListener('click', refresh);
  refresh();
  window.setInterval(refresh, 60_000);
})();
