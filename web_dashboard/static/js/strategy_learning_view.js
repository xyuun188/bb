function strategyLearningEsc(value) {
    return typeof escHtml === 'function'
        ? escHtml(value)
        : String(value ?? '').replace(/[&<>"']/g, ch => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
        }[ch]));
}

function strategyLearningFinite(value) {
    if (value === null || value === undefined || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
}

function strategyLearningNumber(value, digits = 2) {
    const number = strategyLearningFinite(value);
    return number === null ? '-' : number.toFixed(digits);
}

function strategyLearningMetric(label, value, hint = '', tone = 'neutral') {
    return `
        <div class="strategy-learning-metric ${strategyLearningEsc(tone)}">
            <span>${strategyLearningEsc(label)}</span>
            <strong>${strategyLearningEsc(value)}</strong>
            ${hint ? `<em>${strategyLearningEsc(hint)}</em>` : ''}
        </div>`;
}

function strategyLearningTone(value, inverse = false) {
    const number = strategyLearningFinite(value);
    if (number === null) return 'neutral';
    const good = inverse ? number <= 0 : number > 0;
    return good ? 'good' : number === 0 ? 'neutral' : 'bad';
}

function strategyLearningMetricValue(metrics, key, suffix = '') {
    const value = strategyLearningNumber(metrics?.[key]);
    return value === '-' ? value : `${value}${suffix}`;
}

function strategyLearningStatusLabel(value) {
    const labels = {
        complete: '验证完成',
        insufficient_chronological_partitions: '滚动分区不足',
        no_cost_complete_shadow_samples: '无成本完整影子样本',
        governed_dynamic_return: '动态收益治理已启用',
        shadow_validation: '候选验证中',
        insufficient_authoritative_evidence: '权威收益证据不足',
    };
    return labels[value] || value || '未评估';
}

function strategyLearningSchedulerModeLabel(value) {
    return strategyLearningStatusLabel(value || 'insufficient_authoritative_evidence');
}

function strategyLearningScopeMeta(scope) {
    const scopes = {
        side: {
            label: '全市场方向历史分区',
            description: '汇总当前窗口内所有币种的同方向费后收益；不是全市场统一开仓指令',
        },
        regime_side: {
            label: '行情状态方向历史分区',
            description: '仅在行情状态与方向同时相符时提供历史收益先验',
        },
        symbol_side: {
            label: '单币方向历史分区',
            description: '指定币种与方向的历史样本分组；不是为该币种配置的独立执行策略',
        },
    };
    return scopes[scope] || {
        label: '其他历史收益分区',
        description: '按权威费后收益数据自动形成的统计分组',
    };
}

function strategyLearningSelectorLabel(selector) {
    const parts = [];
    if (selector?.symbol) parts.push(selector.symbol);
    if (selector?.market_regime) parts.push(selector.market_regime);
    if (selector?.side) parts.push(selector.side === 'long' ? '做多' : '做空');
    return parts.join(' · ') || '动态收益分区';
}

function strategyLearningProfileTitle(profile) {
    const selector = profile?.params?.selector || {};
    const scope = strategyLearningScopeMeta(selector.scope);
    return `${scope.label} · ${strategyLearningSelectorLabel(selector)}`;
}

function strategyLearningOwnerLabel(owner) {
    const labels = {
        live_ml_profit_contract: '实时费后收益',
        dynamic_entry_risk_budget: '入场风险预算',
        dynamic_position_capacity: '仓位容量',
        dynamic_exit_policy: '动态退出',
    };
    return labels[owner] || owner;
}

function strategyLearningRejectionLabel(reason) {
    const labels = {
        insufficient_chronological_partitions: '滚动分区不足',
        walk_forward_fee_after_return_lcb_not_positive: '滚动收益率下界未转正',
        walk_forward_profit_factor_not_above_break_even: '滚动 PF 未超过盈亏平衡',
        no_cost_complete_shadow_samples: '缺少成本完整影子样本',
        shadow_fee_after_return_lcb_not_positive: '影子收益率下界未转正',
        shadow_profit_factor_not_above_break_even: '影子 PF 未超过盈亏平衡',
    };
    return labels[reason] || reason;
}

function strategyLearningReturnSummary(observation) {
    const count = Number(observation?.sample_count || 0);
    const net = strategyLearningFinite(observation?.realized_net_pnl_usdt);
    const averageReturn = strategyLearningFinite(observation?.average_net_return_pct);
    const returnLcb = strategyLearningFinite(observation?.return_lcb_pct);
    const factor = strategyLearningFinite(observation?.profit_factor);
    return `
        <div class="strategy-learning-summary-grid">
            ${strategyLearningMetric('费后净收益', net === null ? '-' : `${strategyLearningNumber(net)} U`, count ? '权威平仓' : '暂无权威样本', strategyLearningTone(net))}
            ${strategyLearningMetric('平均收益率', averageReturn === null ? '-' : `${strategyLearningNumber(averageReturn)}%`, '按实际保证金口径', strategyLearningTone(averageReturn))}
            ${strategyLearningMetric('收益率下界', returnLcb === null ? '-' : `${strategyLearningNumber(returnLcb)}%`, '动态分区最弱表现', strategyLearningTone(returnLcb))}
            ${strategyLearningMetric('Profit Factor', factor === null ? '-' : strategyLearningNumber(factor), factor === null && count ? '窗口内无亏损样本' : '收益/亏损强度', strategyLearningTone(factor === null ? null : factor - 1))}
            ${strategyLearningMetric('权威样本', String(count), count ? '成本与来源完整' : '没有把缺失当作 0')}
        </div>`;
}

function strategyLearningProductionOverview(data) {
    const schedule = data?.schedule || {};
    const runtime = schedule.runtime || {};
    const usage = data?.feedback?.runtime_prior_usage || {};
    const candidates = Array.isArray(schedule.candidates) ? schedule.candidates : [];
    const production = data?.current_production_strategy || schedule.current_production_strategy || {};
    const leading = schedule.leading_candidate || candidates[0] || null;
    const governedCount = Number(schedule.governed_candidate_count || 0);
    const historicalPriorContextEnabled = runtime.historical_prior_context_enabled === true && governedCount > 0;
    const rejectedCount = Number(schedule.rejected_candidate_count || 0);
    const governedProfiles = candidates.filter(profile => profile?.promotion?.historical_prior_context_eligible === true);
    const eligible = governedProfiles[0] || null;
    const eligibleIdentity = eligible
        ? `${strategyLearningProfileTitle(eligible)} · ${eligible.id || '-'} · v${Number(eligible.version || 0)}`
        : '';
    const leadingIdentity = leading
        ? `${strategyLearningProfileTitle(leading)} · ${leading.id || '-'} · v${Number(leading.version || 0)}`
        : '当前窗口没有可排名的历史收益分区';
    const matchedDecisionCount = Number(usage.matched_decision_count || 0);
    const owners = Array.isArray(production.execution_owners)
        ? production.execution_owners
        : Array.isArray(runtime.execution_owners) ? runtime.execution_owners : [];
    const ownerItems = owners.length ? owners : [
        'live_ml_profit_contract',
        'dynamic_entry_risk_budget',
        'dynamic_position_capacity',
        'dynamic_exit_policy',
    ];
    const stateTitle = historicalPriorContextEnabled
        ? `${governedCount} 条治理通过的历史先验可动态匹配`
        : '当前没有历史先验参与生产决策';
    const stateReason = historicalPriorContextEnabled
        ? matchedDecisionCount
            ? `当前窗口已有 ${matchedDecisionCount} 条市场决策实际匹配历史先验；它们仍不能绕过实时收益、成本和风险合同。`
            : '已有治理通过的历史先验，但当前窗口尚无市场决策实际命中对应币种、方向或行情状态。'
        : candidates.length
            ? `${candidates.length} 个历史收益分区均未完成治理，当前决策没有使用任何候选先验。`
            : '当前窗口没有足够的权威费后收益样本，生产继续使用动态费后收益执行链。';
    return `
        <div class="strategy-learning-command">
            <section class="strategy-learning-runtime-panel" aria-label="所有币种当前共同执行规则">
                <div class="strategy-learning-runtime-head">
                    <div>
                        <span>所有币种当前共同执行规则</span>
                        <strong>${strategyLearningEsc(production.name || '动态费后收益执行链')}</strong>
                    </div>
                    <span class="strategy-learning-table-pill ${production.enabled === false ? 'bad' : 'good'}">${production.enabled === false ? '已停用' : '正在运行'}</span>
                </div>
                <p class="strategy-learning-runtime-copy">每个币种在每轮决策中独立比较做多与做空的实时费后收益，不存在预先固定的逐币执行策略。</p>
                <div class="strategy-learning-command-meta">
                    <span class="strategy-learning-meta"><b>策略 ID</b>${strategyLearningEsc(production.id || '缺失')}</span>
                    <span class="strategy-learning-meta"><b>版本</b>${strategyLearningEsc(production.version || '缺失')}</span>
                    <span class="strategy-learning-meta"><b>目标</b>${strategyLearningEsc(production.objective || '缺失')}</span>
                    <span class="strategy-learning-meta"><b>所有者</b>${strategyLearningEsc(production.owner || '缺失')}</span>
                    <span class="strategy-learning-meta"><b>权威 outcome</b>${strategyLearningEsc(production.data_sources?.authoritative_trade_outcome?.status || '缺失')}</span>
                </div>
                <div class="strategy-learning-execution-chain">
                    ${ownerItems.map((owner, index) => `
                        <span class="strategy-learning-owner-step">
                            <b>${String(index + 1).padStart(2, '0')}</b>
                            <em>${strategyLearningEsc(strategyLearningOwnerLabel(owner))}</em>
                        </span>
                        ${index < ownerItems.length - 1 ? '<i aria-hidden="true">→</i>' : ''}`).join('')}
                </div>
                <div class="strategy-learning-governance-flow">
                    <span class="done"><b>实时</b><em>逐币多空收益评估</em></span>
                    <i aria-hidden="true">→</i>
                    <span class="done"><b>动态</b><em>风险预算与仓位容量</em></span>
                    <i aria-hidden="true">→</i>
                    <span class="done"><b>费后</b><em>最终收益合同与执行</em></span>
                </div>
            </section>
            <section class="strategy-learning-command-main ${historicalPriorContextEnabled ? 'prior-context-active' : 'prior-context-inactive'}" aria-label="当前历史先验上下文状态">
                <span class="strategy-learning-command-eyebrow">历史收益先验状态 · 不能授权开仓</span>
                <div class="strategy-learning-command-title">
                    <strong>${strategyLearningEsc(stateTitle)}</strong>
                    <span class="strategy-learning-status-pill ${matchedDecisionCount ? 'good' : 'warn'}">最近匹配 ${matchedDecisionCount}</span>
                </div>
                <p>${strategyLearningEsc(stateReason)}</p>
                <div class="strategy-learning-inline-alert ${historicalPriorContextEnabled ? '' : 'warn'}">
                    <strong>${historicalPriorContextEnabled ? '可匹配的治理先验示例' : '排名首位不等于正在使用'}</strong>
                    <span>${strategyLearningEsc(historicalPriorContextEnabled ? eligibleIdentity : leadingIdentity)}</span>
                    <em>${historicalPriorContextEnabled ? '实际是否使用以最近决策匹配记录为准' : '未治理分区只参与排名和继续验证'}</em>
                </div>
                <div class="strategy-learning-command-meta">
                    <span class="strategy-learning-meta"><b>调度状态</b>${strategyLearningEsc(strategyLearningSchedulerModeLabel(schedule.scheduler_mode))}</span>
                    <span class="strategy-learning-meta"><b>历史分区</b>${candidates.length}</span>
                    <span class="strategy-learning-meta ${governedCount ? 'good' : ''}"><b>治理通过</b>${governedCount}</span>
                    <span class="strategy-learning-meta ${matchedDecisionCount ? 'good' : ''}"><b>最近实际匹配</b>${matchedDecisionCount}</span>
                    <span class="strategy-learning-meta ${rejectedCount ? 'bad' : ''}"><b>未治理</b>${rejectedCount}</span>
                </div>
            </section>
        </div>`;
}

function strategyLearningRuntimeUsage(usage, governedCount) {
    const matches = Array.isArray(usage?.latest_matches) ? usage.latest_matches : [];
    const inspected = Number(usage?.inspected_decision_count || 0);
    const matchedDecisions = Number(usage?.matched_decision_count || 0);
    const matchedProfiles = Number(usage?.matched_profile_count || 0);
    return `
        <section class="strategy-learning-runtime-usage" aria-label="最近实际匹配的历史先验">
            <div class="strategy-learning-runtime-usage-head">
                <div>
                    <span>最近实际匹配记录</span>
                    <strong>哪些币种与方向真正使用过历史先验</strong>
                </div>
                <div class="strategy-learning-command-meta">
                    <span class="strategy-learning-meta"><b>检查决策</b>${inspected}</span>
                    <span class="strategy-learning-meta ${matchedDecisions ? 'good' : ''}"><b>匹配决策</b>${matchedDecisions}</span>
                    <span class="strategy-learning-meta ${matchedProfiles ? 'good' : ''}"><b>涉及先验</b>${matchedProfiles}</span>
                </div>
            </div>
            ${matches.length
                ? `<div class="strategy-learning-runtime-match-list">${matches.map(match => `
                    <div class="strategy-learning-runtime-match-row">
                        <div>
                            <strong>${strategyLearningEsc(match.symbol || '-')} · ${match.evaluated_side === 'long' ? '做多' : '做空'}</strong>
                            <span>${strategyLearningEsc(match.profile_id || '-')} · v${Number(match.profile_version || 0)} · 排名 #${Number(match.rank || 0)}</span>
                        </div>
                        <span class="strategy-learning-table-pill good">已匹配 · 只读先验</span>
                        <em>${match.matched_at ? strategyLearningEsc(toBeijingTime(match.matched_at)) : '-'}</em>
                    </div>`).join('')}</div>`
                : `<div class="strategy-learning-runtime-match-empty">
                    <strong>最近 ${inspected} 条市场决策：0 条匹配历史先验</strong>
                    <span>${governedCount ? '存在治理先验，但最近决策的币种、方向或行情状态没有命中。' : '当前治理通过为 0，因此所有币种都只运行动态费后收益执行链。'}</span>
                </div>`}
        </section>`;
}

function strategyLearningSideRows(sidePerformance) {
    return ['long', 'short'].map(side => {
        const row = sidePerformance?.[side] || {};
        const count = Number(row.sample_count || 0);
        const net = strategyLearningFinite(row.realized_net_pnl_usdt);
        const averageReturn = strategyLearningFinite(row.average_net_return_pct);
        const lcb = strategyLearningFinite(row.return_lcb_pct);
        return `
            <div class="strategy-learning-compact-head">
                <strong>${side === 'long' ? '做多' : '做空'}</strong>
                <span>${count ? `费后 ${strategyLearningNumber(net)} U · 收益率 ${strategyLearningNumber(averageReturn)}% · 下界 ${strategyLearningNumber(lcb)}%` : '暂无权威费后样本'}</span>
                <em>${count} 样本</em>
            </div>`;
    }).join('');
}

function strategyLearningCandidateState(profile, matchedProfileIds, leadingId) {
    const promotion = profile?.promotion || {};
    const governed = promotion.historical_prior_context_eligible === true;
    const matchedRecently = governed && matchedProfileIds.has(profile?.id);
    const leading = profile?.id === leadingId;
    if (matchedRecently) return { css: 'recently-matched', tone: 'good', label: '最近决策已匹配 · 只读' };
    if (governed) return { css: 'governed', tone: 'good', label: '治理通过 · 等待匹配' };
    if (leading) return { css: 'leading', tone: 'warn', label: '分区排名首位 · 未生效' };
    return { css: 'blocked', tone: 'neutral', label: '历史分区 · 未生效' };
}

function strategyLearningCandidateCard(profile, matchedProfileIds, leadingId) {
    const params = profile?.params || {};
    const historical = params.historical_return_distribution || {};
    const walkForward = profile?.backtest?.metrics || {};
    const shadow = profile?.shadow_validation?.metrics || {};
    const promotion = profile?.promotion || {};
    const reasons = Array.isArray(promotion.rejection_reasons) ? promotion.rejection_reasons : [];
    const state = strategyLearningCandidateState(profile, matchedProfileIds, leadingId);
    return `
        <article class="strategy-learning-profile-card ${state.css}">
            <div class="strategy-learning-profile-card-head">
                <div>
                    <strong>#${Number(profile?.rank || 0)} ${strategyLearningEsc(strategyLearningProfileTitle(profile))}</strong>
                    <span>${strategyLearningEsc(profile?.id || '-')} · v${Number(profile?.version || 0)}</span>
                </div>
                <span class="strategy-learning-table-pill ${state.tone}">${strategyLearningEsc(state.label)}</span>
            </div>
            <div class="strategy-learning-profile-metrics">
                ${strategyLearningMetric('历史收益率下界', strategyLearningMetricValue(historical, 'return_lcb_pct', '%'), `${Number(historical.sample_count || 0)} 条权威平仓`, strategyLearningTone(historical.return_lcb_pct))}
                ${strategyLearningMetric('滚动收益率下界', strategyLearningMetricValue(walkForward, 'return_lcb_pct', '%'), `${Number(walkForward.sample_count || 0)} 条 · ${strategyLearningStatusLabel(profile?.backtest?.status)}`, strategyLearningTone(walkForward.return_lcb_pct))}
                ${strategyLearningMetric('影子收益率下界', strategyLearningMetricValue(shadow, 'return_lcb_pct', '%'), `${Number(shadow.sample_count || 0)} 条 · ${strategyLearningStatusLabel(profile?.shadow_validation?.status)}`, strategyLearningTone(shadow.return_lcb_pct))}
                ${strategyLearningMetric('费后净收益', strategyLearningMetricValue(historical, 'realized_net_pnl_usdt', ' U'), '真实平仓', strategyLearningTone(historical.realized_net_pnl_usdt))}
                ${strategyLearningMetric('Profit Factor', strategyLearningMetricValue(historical, 'profit_factor'), historical.profit_factor == null && historical.sample_count ? '窗口内无亏损样本' : '盈亏强度', strategyLearningTone(historical.profit_factor == null ? null : historical.profit_factor - 1))}
                ${strategyLearningMetric('最大回撤', strategyLearningMetricValue(historical, 'max_drawdown'), '同口径累计', strategyLearningTone(historical.max_drawdown, true))}
                ${strategyLearningMetric('尾部收益率', strategyLearningMetricValue(historical, 'tail_loss_pct', '%'), '窗口最差样本', strategyLearningTone(historical.tail_loss_pct))}
            </div>
            <div class="strategy-learning-profile-gate ${reasons.length ? 'blocked' : 'passed'}">
                <strong>${reasons.length ? '未生效原因' : '治理证据'}</strong>
                <div class="strategy-learning-profile-chips">
                    ${reasons.length
                        ? reasons.map(reason => `<span>${strategyLearningEsc(strategyLearningRejectionLabel(reason))}</span>`).join('')
                        : '<span>滚动与成本完整影子证据均通过</span>'}
                </div>
            </div>
            <div class="strategy-learning-profile-footer">
                <span>适用范围：${strategyLearningEsc(strategyLearningSelectorLabel(params.selector || {}))}</span>
                <span>运行作用：${state.css === 'recently-matched' ? '最近决策曾匹配为只读历史先验；不能授权开仓。' : state.css === 'governed' ? '等待与决策币种、方向或行情状态匹配。' : '无运行影响，只参与排名和继续验证。'}</span>
            </div>
        </article>`;
}

function strategyLearningCandidateGroups(candidates, matchedProfileIds, leadingId) {
    const scopes = ['side', 'regime_side', 'symbol_side'];
    return `<div class="strategy-learning-candidate-groups">
        ${scopes.map(scope => {
            const group = candidates.filter(profile => profile?.params?.selector?.scope === scope);
            const meta = strategyLearningScopeMeta(scope);
            return `
                <section class="strategy-learning-candidate-group" aria-label="${strategyLearningEsc(meta.label)}">
                    <div class="strategy-learning-candidate-group-head">
                        <div>
                            <strong>${strategyLearningEsc(meta.label)}</strong>
                            <span>${strategyLearningEsc(meta.description)}</span>
                        </div>
                        <em>${group.length} 个</em>
                    </div>
                    ${group.length
                        ? `<div class="strategy-learning-profile-board">${group.map(profile => strategyLearningCandidateCard(profile, matchedProfileIds, leadingId)).join('')}</div>`
                        : '<div class="strategy-learning-candidate-group-empty">当前窗口未形成这一类历史收益分区</div>'}
                </section>`;
        }).join('')}
    </div>`;
}

function strategyLearningCandidateIndex(candidates, leading, usage) {
    const leadingText = leading ? `#${Number(leading.rank || 0)} ${strategyLearningProfileTitle(leading)}` : '无可排名候选';
    const governedCount = candidates.filter(profile => profile?.promotion?.historical_prior_context_eligible === true).length;
    const matchedCount = Number(usage?.matched_decision_count || 0);
    return `
        <div class="strategy-learning-compact-head"><strong>运行</strong><span>所有币种共同使用动态费后收益执行链</span><em>逐币逐方向评估</em></div>
        <div class="strategy-learning-compact-head"><strong>${candidates.length}</strong><span>历史收益分区</span><em>不是 ${candidates.length} 套执行策略</em></div>
        <div class="strategy-learning-compact-head"><strong>${governedCount}</strong><span>治理通过的只读先验</span><em>${governedCount ? '可按适用范围匹配' : '当前无先验参与决策'}</em></div>
        <div class="strategy-learning-compact-head"><strong>${matchedCount}</strong><span>最近实际匹配决策</span><em>${matchedCount ? '详情见页面顶部' : '当前未匹配'}</em></div>
        <div class="strategy-learning-compact-head"><strong>排名</strong><span>${strategyLearningEsc(leadingText)}</span><em>排名不等于使用</em></div>`;
}

function strategyLearningSetHtml(id, html) {
    const element = document.getElementById(id);
    if (element) element.innerHTML = html;
}

function renderStrategyLearning(data) {
    const feedback = data?.feedback || {};
    const observation = feedback.authoritative_return_observation || {};
    const schedule = data?.schedule || {};
    const runtime = schedule.runtime || {};
    const candidates = Array.isArray(schedule.candidates) ? schedule.candidates : [];
    const leading = schedule.leading_candidate || candidates[0] || null;
    const usage = feedback.runtime_prior_usage || {};
    const governedCount = Number(schedule.governed_candidate_count || 0);
    const matchedProfileIds = new Set(
        (Array.isArray(usage.latest_matches) ? usage.latest_matches : [])
            .map(match => match?.profile_id)
            .filter(Boolean),
    );
    const openCount = Number(feedback.open_position_pressure?.open_position_count || 0);
    const policy = feedback.training_policy || {};
    const shadow = feedback.shadow_feedback || {};
    const eventFeedback = feedback.event_feedback || {};
    const problems = Array.isArray(feedback.problems) ? feedback.problems : [];
    const historicalPriorContextEnabled = runtime.historical_prior_context_enabled === true && governedCount > 0;

    const updated = document.getElementById('strategy-learning-updated');
    if (updated) updated.textContent = feedback.generated_at ? toBeijingTime(feedback.generated_at) : '暂无生成时间';

    strategyLearningSetHtml('strategy-learning-summary', `
        ${strategyLearningProductionOverview(data)}
        ${strategyLearningRuntimeUsage(usage, governedCount)}
        ${strategyLearningReturnSummary(observation)}`);
    strategyLearningSetHtml('strategy-learning-problems', problems.length
        ? problems.map(item => `<div class="strategy-learning-compact-head"><strong>${strategyLearningEsc(item.code)}</strong><span>${strategyLearningEsc(item.kind)}</span><em>${Number(item.count || 0)}</em></div>`).join('')
        : '<div class="strategy-learning-empty">当前窗口没有样本隔离问题。</div>');
    strategyLearningSetHtml('strategy-learning-sides', strategyLearningSideRows(feedback.side_performance || {}));
    strategyLearningSetHtml('strategy-learning-release', `
        <div class="strategy-learning-compact-head"><strong>${openCount}</strong><span>当前持仓</span><em>${strategyLearningNumber(feedback.open_position_pressure?.unrealized_pnl_usdt)} U</em></div>`);
    strategyLearningSetHtml('strategy-learning-experts', `
        <div class="strategy-learning-guard-state ok"><strong>动态费后收益执行链</strong><span>${strategyLearningEsc((runtime.execution_owners || []).map(strategyLearningOwnerLabel).join(' · ') || '实时收益 · 风险预算 · 仓位容量 · 动态退出')}</span><em>逐次独立校验</em></div>`);
    strategyLearningSetHtml('strategy-learning-reflections', `
        <div class="strategy-learning-chip-row">
            <span>目标 ${strategyLearningEsc(policy.optimization_target || 'maximize_authoritative_fee_after_return_rate')}</span>
            <span>滚动验证 ${policy.walk_forward_required === true ? '必须' : '未声明'}</span>
            <span>成本完整影子 ${policy.cost_complete_shadow_required === true ? '必须' : '未声明'}</span>
            <span>胜率 ${strategyLearningEsc(policy.win_rate_role || 'diagnostic_only')}</span>
        </div>`);
    strategyLearningSetHtml('strategy-learning-profiles', candidates.length
        ? strategyLearningCandidateGroups(candidates, matchedProfileIds, leading?.id)
        : '<div class="strategy-learning-empty">没有权威、成本完整且具备保证金收益率口径的平仓样本，因此历史收益分区为空；这不是数值 0。</div>');
    strategyLearningSetHtml('strategy-learning-events', `
        <div class="strategy-learning-compact-head"><strong>${Number(eventFeedback.linked_event_count || 0)}</strong><span>已关联策略事件</span><em>${Number(eventFeedback.regime_linked_position_count || 0)} 个持仓带行情分区</em></div>
        <div class="strategy-learning-compact-head"><strong>${Number(shadow.cost_complete_direction_sample_count || 0)}</strong><span>成本完整影子方向样本</span><em>${Number(shadow.completed_row_count || 0)} 条完成记录</em></div>`);
    strategyLearningSetHtml('strategy-learning-guard', `
        <div class="strategy-learning-guard-state ${historicalPriorContextEnabled ? 'ok' : 'warn'}"><strong>${strategyLearningEsc(strategyLearningSchedulerModeLabel(schedule.scheduler_mode))}</strong><span>${historicalPriorContextEnabled ? '治理历史先验可按决策上下文匹配' : '当前没有历史先验参与决策'}</span><em>历史先验不能授权下单</em></div>`);
    strategyLearningSetHtml('strategy-learning-recent-events', strategyLearningCandidateIndex(candidates, leading, usage));
}
