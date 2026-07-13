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

function strategyLearningSelectorLabel(selector) {
    const parts = [];
    if (selector?.symbol) parts.push(selector.symbol);
    if (selector?.market_regime) parts.push(selector.market_regime);
    if (selector?.side) parts.push(selector.side === 'long' ? '做多' : '做空');
    return parts.join(' · ') || '动态收益分区';
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

function strategyLearningCandidateCard(profile, activeId) {
    const params = profile?.params || {};
    const historical = params.historical_return_distribution || {};
    const walkForward = profile?.backtest?.metrics || {};
    const shadow = profile?.shadow_validation?.metrics || {};
    const promotion = profile?.promotion || {};
    const governed = promotion.production_influence_eligible === true;
    const active = profile?.id && profile.id === activeId;
    const reasons = Array.isArray(promotion.rejection_reasons) ? promotion.rejection_reasons : [];
    return `
        <article class="strategy-learning-profile-card ${active ? 'active' : ''}">
            <div class="strategy-learning-profile-card-head">
                <div>
                    <strong>#${Number(profile?.rank || 0)} ${strategyLearningEsc(profile?.label || profile?.id || '收益策略候选')}</strong>
                    <span>${strategyLearningEsc(strategyLearningSelectorLabel(params.selector || {}))}</span>
                </div>
                <span class="strategy-learning-table-pill ${governed ? 'good' : 'warn'}">${governed ? '治理通过' : '影子验证'}</span>
            </div>
            <div class="strategy-learning-profile-metrics">
                ${strategyLearningMetric('历史收益率下界', strategyLearningMetricValue(historical, 'return_lcb_pct', '%'), '权威费后')}
                ${strategyLearningMetric('滚动收益率下界', strategyLearningMetricValue(walkForward, 'return_lcb_pct', '%'), profile?.backtest?.status || '未评估')}
                ${strategyLearningMetric('影子收益率下界', strategyLearningMetricValue(shadow, 'return_lcb_pct', '%'), profile?.shadow_validation?.status || '未评估')}
                ${strategyLearningMetric('费后净收益', strategyLearningMetricValue(historical, 'realized_net_pnl_usdt', ' U'), '真实平仓')}
                ${strategyLearningMetric('Profit Factor', strategyLearningMetricValue(historical, 'profit_factor'), historical.profit_factor == null && historical.sample_count ? '无亏损样本' : '')}
                ${strategyLearningMetric('最大回撤', strategyLearningMetricValue(historical, 'max_drawdown'), '同口径累计')}
                ${strategyLearningMetric('尾部收益率', strategyLearningMetricValue(historical, 'tail_loss_pct', '%'), '窗口最差样本')}
            </div>
            <div class="strategy-learning-profile-chips">
                ${reasons.length
                    ? reasons.map(reason => `<span>${strategyLearningEsc(strategyLearningRejectionLabel(reason))}</span>`).join('')
                    : '<span>滚动与成本完整影子证据均通过</span>'}
            </div>
            <p>${governed ? '可作为历史收益先验；当前实时收益、成本和账户风险合同仍须独立通过。' : '继续收集证据，不影响生产方向、仓位、杠杆或退出。'}</p>
        </article>`;
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
    const openCount = Number(feedback.open_position_pressure?.open_position_count || 0);
    const policy = feedback.training_policy || {};
    const shadow = feedback.shadow_feedback || {};
    const eventFeedback = feedback.event_feedback || {};
    const problems = Array.isArray(feedback.problems) ? feedback.problems : [];
    const influenceEnabled = runtime.production_influence_enabled === true;

    const updated = document.getElementById('strategy-learning-updated');
    if (updated) updated.textContent = feedback.generated_at ? toBeijingTime(feedback.generated_at) : '暂无生成时间';

    strategyLearningSetHtml('strategy-learning-summary', `
        <div class="opening-funnel-verdict ${influenceEnabled ? 'opening-funnel-ok' : 'opening-funnel-warn'}">
            <strong>${influenceEnabled ? '动态收益策略已受治理调度' : candidates.length ? '候选仍在滚动/影子验证' : '尚无可生成策略候选'}</strong>
            <span>${strategyLearningEsc(schedule.reason || '等待权威费后收益率证据。')}</span>
        </div>
        ${strategyLearningReturnSummary(observation)}`);

    strategyLearningSetHtml('strategy-learning-problems', problems.length
        ? problems.map(item => `<div class="strategy-learning-compact-head"><strong>${strategyLearningEsc(item.code)}</strong><span>${strategyLearningEsc(item.kind)}</span><em>${Number(item.count || 0)}</em></div>`).join('')
        : '<div class="strategy-learning-empty">当前窗口没有样本隔离问题。</div>');
    strategyLearningSetHtml('strategy-learning-sides', strategyLearningSideRows(feedback.side_performance || {}));
    strategyLearningSetHtml('strategy-learning-release', `
        <div class="strategy-learning-compact-head"><strong>${openCount}</strong><span>当前持仓</span><em>${strategyLearningNumber(feedback.open_position_pressure?.unrealized_pnl_usdt)} U</em></div>`);
    strategyLearningSetHtml('strategy-learning-experts', `
        <div class="strategy-learning-guard-state ok"><strong>执行所有权隔离</strong><span>${strategyLearningEsc((runtime.execution_owners || []).join(' · ') || '动态收益执行与风险服务')}</span><em>策略不能授权下单</em></div>`);
    strategyLearningSetHtml('strategy-learning-reflections', `
        <div class="strategy-learning-chip-row">
            <span>目标 ${strategyLearningEsc(policy.optimization_target || 'maximize_authoritative_fee_after_return_rate')}</span>
            <span>滚动验证 ${policy.walk_forward_required === true ? '必须' : '未声明'}</span>
            <span>成本完整影子 ${policy.cost_complete_shadow_required === true ? '必须' : '未声明'}</span>
            <span>胜率 ${strategyLearningEsc(policy.win_rate_role || 'diagnostic_only')}</span>
        </div>`);
    strategyLearningSetHtml('strategy-learning-profiles', candidates.length
        ? `<div class="strategy-learning-profile-board">${candidates.map(profile => strategyLearningCandidateCard(profile, active?.id)).join('')}</div>`
        : '<div class="strategy-learning-empty">没有权威、成本完整且具备保证金收益率口径的平仓样本，因此候选为空；这不是数值 0。</div>');
    strategyLearningSetHtml('strategy-learning-events', `
        <div class="strategy-learning-compact-head"><strong>${Number(eventFeedback.linked_event_count || 0)}</strong><span>已关联策略事件</span><em>${Number(eventFeedback.regime_linked_position_count || 0)} 个持仓带行情分区</em></div>
        <div class="strategy-learning-compact-head"><strong>${Number(shadow.cost_complete_direction_sample_count || 0)}</strong><span>成本完整影子方向样本</span><em>${Number(shadow.completed_row_count || 0)} 条完成记录</em></div>`);
    strategyLearningSetHtml('strategy-learning-guard', `
        <div class="strategy-learning-guard-state ${influenceEnabled ? 'ok' : 'warn'}"><strong>${strategyLearningEsc(schedule.scheduler_mode || 'unknown')}</strong><span>${influenceEnabled ? '历史收益先验已启用' : '生产影响保持关闭'}</span><em>下单权限始终为否</em></div>`);
    strategyLearningSetHtml('strategy-learning-recent-events', candidates.length
        ? candidates.map(profile => `<div class="strategy-learning-compact-head"><strong>#${Number(profile.rank || 0)}</strong><span>${strategyLearningEsc(profile.label || profile.id)}</span><em>${profile.promotion?.production_influence_eligible === true ? '通过' : '待验证'}</em></div>`).join('')
        : '<div class="strategy-learning-empty">暂无候选调度事件。</div>');
}
