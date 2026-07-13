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
            label: '全市场方向候选',
            description: '汇总当前窗口内所有币种的做多或做空费后收益',
        },
        regime_side: {
            label: '行情状态方向候选',
            description: '只匹配相同行情状态与方向的历史收益分区',
        },
        symbol_side: {
            label: '币种方向候选',
            description: '只匹配指定币种与做多或做空方向的历史收益分区',
        },
    };
    return scopes[scope] || {
        label: '其他收益候选',
        description: '按权威费后收益数据自动生成的历史分区',
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
    return `${scope.label.replace('候选', '')} · ${strategyLearningSelectorLabel(selector)}`;
}

function strategyLearningOwnerLabel(owner) {
    const labels = {
        return_execution_policy: '实时费后收益',
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
    const candidates = Array.isArray(schedule.candidates) ? schedule.candidates : [];
    const active = schedule.active_profile || data?.active_profile || null;
    const leading = schedule.leading_candidate || candidates[0] || null;
    const governedCount = Number(schedule.governed_candidate_count || 0);
    const influenceEnabled = runtime.production_influence_enabled === true && governedCount > 0;
    const rejectedCount = Number(schedule.rejected_candidate_count || 0);
    const governedProfiles = candidates.filter(profile => profile?.promotion?.production_influence_eligible === true);
    const selected = active || governedProfiles[0] || leading;
    const selector = selected?.params?.selector || {};
    const selectedIdentity = selected
        ? `${strategyLearningProfileTitle(selected)} · ${selected.id || '-'} · v${Number(selected.version || 0)}`
        : '当前窗口没有可排名候选';
    const owners = Array.isArray(runtime.execution_owners) ? runtime.execution_owners : [];
    const ownerItems = owners.length ? owners : [
        'return_execution_policy',
        'dynamic_entry_risk_budget',
        'dynamic_position_capacity',
        'dynamic_exit_policy',
    ];
    const stateTitle = active
        ? strategyLearningProfileTitle(active)
        : influenceEnabled
            ? `${governedCount} 个治理候选可按适用范围匹配`
            : '没有候选策略在生产生效';
    const stateReason = influenceEnabled
        ? '治理候选按币种、方向和行情状态匹配；当前实时收益、交易成本和账户风险仍须逐次通过。'
        : candidates.length
            ? `${candidates.length} 个候选均未完成治理，生产继续使用动态费后收益执行链。`
            : '当前窗口没有足够的权威费后收益样本，生产继续使用动态费后收益执行链。';
    return `
        <div class="strategy-learning-command">
            <section class="strategy-learning-command-main ${influenceEnabled ? 'production-active' : 'production-idle'}" aria-label="当前候选策略生产状态">
                <span class="strategy-learning-command-eyebrow">候选先验状态</span>
                <div class="strategy-learning-command-title">
                    <strong>${strategyLearningEsc(stateTitle)}</strong>
                    <span class="strategy-learning-status-pill ${influenceEnabled ? 'good' : 'warn'}">${influenceEnabled ? '生产匹配已启用' : '生产影响关闭'}</span>
                </div>
                <p>${strategyLearningEsc(stateReason)}</p>
                <div class="strategy-learning-inline-alert ${influenceEnabled ? '' : 'warn'}">
                    <strong>${active ? '当前全局/行情先验' : influenceEnabled ? '当前可匹配治理候选' : '当前排名首位（未生效）'}</strong>
                    <span>${strategyLearningEsc(selectedIdentity)}</span>
                    <em>${strategyLearningEsc(strategyLearningSelectorLabel(selector))}</em>
                </div>
                <div class="strategy-learning-command-meta">
                    <span class="strategy-learning-meta"><b>调度状态</b>${strategyLearningEsc(strategyLearningSchedulerModeLabel(schedule.scheduler_mode))}</span>
                    <span class="strategy-learning-meta"><b>已生成</b>${candidates.length}</span>
                    <span class="strategy-learning-meta ${governedCount ? 'good' : ''}"><b>治理通过</b>${governedCount}</span>
                    <span class="strategy-learning-meta ${rejectedCount ? 'bad' : ''}"><b>未生效</b>${rejectedCount}</span>
                </div>
            </section>
            <section class="strategy-learning-runtime-panel" aria-label="当前实际执行策略">
                <div class="strategy-learning-runtime-head">
                    <div>
                        <span>当前实际执行策略</span>
                        <strong>动态费后收益执行链</strong>
                    </div>
                    <span class="strategy-learning-table-pill good">逐次校验</span>
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
                    <span class="${candidates.length ? 'done' : ''}"><b>${candidates.length}</b><em>候选生成</em></span>
                    <i aria-hidden="true">→</i>
                    <span class="${governedCount ? 'done' : 'blocked'}"><b>${governedCount}</b><em>证据治理通过</em></span>
                    <i aria-hidden="true">→</i>
                    <span class="${influenceEnabled ? 'done' : 'blocked'}"><b>${governedCount}</b><em>生产先验可匹配</em></span>
                </div>
            </section>
        </div>`;
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

function strategyLearningCandidateState(profile, activeId, leadingId, influenceEnabled) {
    const promotion = profile?.promotion || {};
    const governed = promotion.production_influence_eligible === true;
    const productionActive = influenceEnabled && governed && profile?.id === activeId;
    const leading = profile?.id === leadingId;
    if (productionActive) return { css: 'production-active', tone: 'good', label: '生产先验生效' };
    if (governed) return { css: 'governed', tone: 'good', label: '治理通过 · 可匹配' };
    if (leading) return { css: 'leading', tone: 'warn', label: '排名首位 · 未生效' };
    return { css: 'blocked', tone: 'neutral', label: '候选 · 未生效' };
}

function strategyLearningCandidateCard(profile, activeId, leadingId, influenceEnabled) {
    const params = profile?.params || {};
    const historical = params.historical_return_distribution || {};
    const walkForward = profile?.backtest?.metrics || {};
    const shadow = profile?.shadow_validation?.metrics || {};
    const promotion = profile?.promotion || {};
    const reasons = Array.isArray(promotion.rejection_reasons) ? promotion.rejection_reasons : [];
    const state = strategyLearningCandidateState(profile, activeId, leadingId, influenceEnabled);
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
                <span>生产作用：${state.css === 'production-active' ? '历史收益先验已生效；不能绕过实时收益、成本或风险合同。' : state.css === 'governed' ? '等待与当前币种、方向或行情状态匹配。' : '无生产影响，继续收集验证证据。'}</span>
            </div>
        </article>`;
}

function strategyLearningCandidateGroups(candidates, activeId, leadingId, influenceEnabled) {
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
                        ? `<div class="strategy-learning-profile-board">${group.map(profile => strategyLearningCandidateCard(profile, activeId, leadingId, influenceEnabled)).join('')}</div>`
                        : '<div class="strategy-learning-candidate-group-empty">当前窗口未生成这一类候选</div>'}
                </section>`;
        }).join('')}
    </div>`;
}

function strategyLearningCandidateIndex(candidates, active, leading) {
    const leadingText = leading ? `#${Number(leading.rank || 0)} ${strategyLearningProfileTitle(leading)}` : '无可排名候选';
    const governedCount = candidates.filter(profile => profile?.promotion?.production_influence_eligible === true).length;
    const activeText = active
        ? strategyLearningProfileTitle(active)
        : governedCount
            ? `${governedCount} 个治理候选按适用范围动态匹配`
            : '无候选在生产生效';
    return `
        <div class="strategy-learning-compact-head"><strong>${candidates.length}</strong><span>已生成候选</span><em>按三类适用范围分组</em></div>
        <div class="strategy-learning-compact-head"><strong>${governedCount}</strong><span>治理通过候选</span><em>${governedCount ? '可作为历史先验匹配' : '当前全部未生效'}</em></div>
        <div class="strategy-learning-compact-head"><strong>生产</strong><span>${strategyLearningEsc(activeText)}</span><em>${governedCount ? '可匹配' : '关闭'}</em></div>
        <div class="strategy-learning-compact-head"><strong>排名</strong><span>${strategyLearningEsc(leadingText)}</span><em>${active && leading?.id === active?.id ? '生产生效' : '不等于生效'}</em></div>`;
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
    const active = schedule.active_profile || data?.active_profile || null;
    const leading = schedule.leading_candidate || candidates[0] || null;
    const openCount = Number(feedback.open_position_pressure?.open_position_count || 0);
    const policy = feedback.training_policy || {};
    const shadow = feedback.shadow_feedback || {};
    const eventFeedback = feedback.event_feedback || {};
    const problems = Array.isArray(feedback.problems) ? feedback.problems : [];
    const influenceEnabled = runtime.production_influence_enabled === true && Boolean(active);

    const updated = document.getElementById('strategy-learning-updated');
    if (updated) updated.textContent = feedback.generated_at ? toBeijingTime(feedback.generated_at) : '暂无生成时间';

    strategyLearningSetHtml('strategy-learning-summary', `
        ${strategyLearningProductionOverview(data)}
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
        ? strategyLearningCandidateGroups(candidates, active?.id, leading?.id, influenceEnabled)
        : '<div class="strategy-learning-empty">没有权威、成本完整且具备保证金收益率口径的平仓样本，因此候选为空；这不是数值 0。</div>');
    strategyLearningSetHtml('strategy-learning-events', `
        <div class="strategy-learning-compact-head"><strong>${Number(eventFeedback.linked_event_count || 0)}</strong><span>已关联策略事件</span><em>${Number(eventFeedback.regime_linked_position_count || 0)} 个持仓带行情分区</em></div>
        <div class="strategy-learning-compact-head"><strong>${Number(shadow.cost_complete_direction_sample_count || 0)}</strong><span>成本完整影子方向样本</span><em>${Number(shadow.completed_row_count || 0)} 条完成记录</em></div>`);
    strategyLearningSetHtml('strategy-learning-guard', `
        <div class="strategy-learning-guard-state ${influenceEnabled ? 'ok' : 'warn'}"><strong>${strategyLearningEsc(strategyLearningSchedulerModeLabel(schedule.scheduler_mode))}</strong><span>${influenceEnabled ? '历史收益先验已启用' : '没有候选策略在生产生效'}</span><em>候选不能授权下单</em></div>`);
    strategyLearningSetHtml('strategy-learning-recent-events', strategyLearningCandidateIndex(candidates, active, leading));
}
