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

function strategyLearningNumber(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : '-';
}

function strategyLearningMetric(label, value, hint = '', tone = 'neutral') {
    return `
        <div class="strategy-learning-metric ${strategyLearningEsc(tone)}">
            <span>${strategyLearningEsc(label)}</span>
            <strong>${strategyLearningEsc(value)}</strong>
            ${hint ? `<em>${strategyLearningEsc(hint)}</em>` : ''}
        </div>`;
}

function strategyLearningReturnSummary(observation) {
    const net = Number(observation.realized_net_pnl_usdt || 0);
    const lower = Number(observation.pnl_lower_hinge_usdt || 0);
    const factor = observation.profit_factor;
    return `
        <div class="strategy-learning-summary-grid">
            ${strategyLearningMetric('费后净收益', `${strategyLearningNumber(net)} U`, '权威平仓样本', net > 0 ? 'good' : 'bad')}
            ${strategyLearningMetric('收益下半区', `${strategyLearningNumber(lower)} U`, '左尾质量', lower > 0 ? 'good' : 'warn')}
            ${strategyLearningMetric('Profit Factor', factor == null ? '-' : strategyLearningNumber(factor), '收益/亏损强度')}
            ${strategyLearningMetric('样本数', String(Number(observation.sample_count || 0)), '只用于观察')}
        </div>`;
}

function strategyLearningSideRows(sidePerformance) {
    return ['long', 'short'].map(side => {
        const row = sidePerformance?.[side] || {};
        const net = Number(row.realized_net_pnl_usdt || 0);
        return `
            <div class="strategy-learning-compact-head">
                <strong>${side === 'long' ? '做多' : '做空'}</strong>
                <span>费后 ${strategyLearningNumber(net)} U · PF ${row.profit_factor == null ? '-' : strategyLearningNumber(row.profit_factor)}</span>
                <em>${Number(row.sample_count || 0)} 样本</em>
            </div>`;
    }).join('');
}

function strategyLearningSetHtml(id, html) {
    const element = document.getElementById(id);
    if (element) element.innerHTML = html;
}

function renderStrategyLearning(data) {
    const feedback = data?.feedback || {};
    const observation = feedback.authoritative_return_observation || {};
    const schedule = data?.schedule || {};
    const profile = schedule.active_profile || data?.active_profile || {};
    const openCount = Number(feedback.open_position_pressure?.open_position_count || 0);
    const policy = feedback.training_policy || {};

    const updated = document.getElementById('strategy-learning-updated');
    if (updated) updated.textContent = feedback.generated_at ? toBeijingTime(feedback.generated_at) : '暂无生成时间';

    strategyLearningSetHtml('strategy-learning-summary', `
        <div class="opening-funnel-verdict ${Number(observation.realized_net_pnl_usdt || 0) > 0 ? 'opening-funnel-ok' : 'opening-funnel-warn'}">
            <strong>费后收益只读观察</strong>
            <span>交易权仍只属于当前收益分布、实时成本、动态风险预算和账户状态。</span>
        </div>
        ${strategyLearningReturnSummary(observation)}`);

    strategyLearningSetHtml('strategy-learning-problems', strategyLearningReturnSummary(observation));
    strategyLearningSetHtml('strategy-learning-sides', strategyLearningSideRows(feedback.side_performance || {}));
    strategyLearningSetHtml('strategy-learning-release', `
        <div class="strategy-learning-compact-head"><strong>${openCount}</strong><span>当前持仓</span><em>不改变容量</em></div>`);
    strategyLearningSetHtml('strategy-learning-experts', `
        <div class="strategy-learning-guard-state ok"><strong>观察专用</strong><span>专家/记忆/影子不能授权交易</span><em>只读</em></div>`);
    strategyLearningSetHtml('strategy-learning-reflections', `
        <div class="strategy-learning-chip-row">
            <span>优化目标 ${strategyLearningEsc(policy.optimization_target || 'realized_fee_after_return')}</span>
            <span>成本完整 ${policy.cost_complete_samples_required === true ? '必须' : '未声明'}</span>
        </div>`);
    strategyLearningSetHtml('strategy-learning-profiles', `
        <article class="strategy-learning-profile-card active">
            <div class="strategy-learning-profile-card-head"><div><strong>${strategyLearningEsc(profile.label || '权威收益观察')}</strong><span>${strategyLearningEsc(profile.source || 'closed_position_fee_after_return')}</span></div><span class="strategy-learning-table-pill good">只读</span></div>
            <p>${strategyLearningEsc(profile.description || '不影响生产交易、仓位、杠杆或模型晋升。')}</p>
        </article>`);
    strategyLearningSetHtml('strategy-learning-events', '<div class="strategy-learning-empty">事件只作收益归因，不生成策略权限。</div>');
    strategyLearningSetHtml('strategy-learning-guard', '<div class="strategy-learning-guard-state ok"><strong>生产权限隔离</strong><span>动态收益执行合同是唯一入口</span><em>已隔离</em></div>');
    strategyLearningSetHtml('strategy-learning-recent-events', '<div class="strategy-learning-empty">没有可改写生产策略的学习事件。</div>');
}
