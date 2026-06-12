function strategyLearningText(key) {
    const labels = {
        baseline_current: '\u5f53\u524d\u57fa\u7ebf',
        balanced_probe: '\u5e73\u8861\u63a2\u9488',
        loss_release: '\u4e8f\u635f\u91ca\u653e',
        winner_hold: '\u8d62\u5bb6\u6301\u4ed3\u4f18\u5316',
        negative_realized_pnl: '\u6700\u8fd1\u51c0\u6536\u76ca\u4e3a\u8d1f',
        long_side_degraded: '\u591a\u5355\u4fa7\u8868\u73b0\u9000\u5316',
        short_side_degraded: '\u7a7a\u5355\u4fa7\u8868\u73b0\u9000\u5316',
        full_position_loss_pressure: '\u6ee1\u4ed3\u4e14\u4e8f\u635f\u4ed3\u5360\u4f4d',
        expert_fallback_overblocking: '\u4e13\u5bb6 fallback \u8fc7\u591a\u963b\u65ad\u5f00\u4ed3',
        missed_opportunities: '\u5f71\u5b50\u590d\u76d8\u663e\u793a\u9519\u8fc7\u673a\u4f1a',
        small_wins_large_losses: '\u5c0f\u76c8\u591a\u4f46\u5927\u4e8f\u5b58\u5728',
        loss_hold_too_long: '\u4e8f\u635f\u4ed3\u6301\u6709\u8fc7\u4e45',
        max_position_blocks: '\u4ed3\u4f4d\u4e0a\u9650\u963b\u65ad\u5f00\u4ed3',
        event_fallback_blocks: '\u4e8b\u4ef6\u663e\u793a fallback \u963b\u65ad',
        execution_errors: '\u6267\u884c\u5f02\u5e38',
        strategy_attribution_gap: '\u7b56\u7565\u5f52\u56e0\u4e0d\u5b8c\u6574',
        reflection_negative_pnl: '\u7b56\u7565\u590d\u76d8\u8d39\u540e\u4e8f\u635f',
        reflection_loss_hold_too_long: '\u590d\u76d8\u663e\u793a\u4e8f\u635f\u62d6\u5ef6\u8fc7\u4e45',
        reflection_small_wins_large_losses: '\u590d\u76d8\u663e\u793a\u5c0f\u76c8\u5927\u4e8f',
        trade_reflection_mistakes: '\u7b56\u7565\u590d\u76d8\u91cd\u590d\u9519\u8bef',
        recent_net_pnl_guard: '\u8fd1\u671f\u51c0\u6536\u76ca\u62a4\u680f',
        fallback_dependency_guard: 'fallback \u4f9d\u8d56\u62a4\u680f',
        execution_error_guard: '\u6267\u884c\u5f02\u5e38\u62a4\u680f',
        insufficient_trade_samples: '\u4ea4\u6613\u6837\u672c\u4e0d\u8db3',
    };
    return labels[key] || key || '-';
}

function strategyLearningShort(value, maxLen = 36) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    return text.length > maxLen ? `${text.slice(0, maxLen - 1)}...` : text;
}

function strategyLearningIsReadable(text) {
    const value = String(text || '');
    return value && !value.includes('�') && !value.includes('????') && !/[\u4e00-\u9fff]?鍙|鐢|浣|涓|鏆/.test(value);
}

function renderStrategyLearning(data) {
    data = data || {};
    renderStrategyLearningSummary(data);
    renderStrategyLearningProblems(data);
    renderStrategyLearningSides(data);
    renderStrategyLearningRelease(data);
    renderStrategyLearningExperts(data);
    renderStrategyLearningReflections(data);
    renderStrategyLearningProfiles(data);
    renderStrategyLearningEvents(data);
    renderStrategyLearningGuard(data);
    renderStrategyLearningRecentEvents(data);
    const updated = document.getElementById('strategy-learning-updated');
    if (updated) {
        updated.textContent = `${data.mode === 'live' ? '\u5b9e\u76d8' : '\u6a21\u62df\u76d8'} · \u6700\u8fd1 ${data.window_hours || 168} \u5c0f\u65f6 · ${new Date().toLocaleTimeString()}`;
    }
}

function renderStrategyLearningSummary(data) {
    const el = document.getElementById('strategy-learning-summary');
    if (!el) return;
    const feedback = data.feedback || {};
    const totals = feedback.totals || {};
    const schedule = data.schedule || {};
    const profile = schedule.active_profile || data.active_profile || {};
    const pnl = Number(totals.net_pnl || 0);
    const tone = pnl > 0 ? 'good' : pnl < 0 ? 'warn' : 'muted';
    const llm = data.llm_candidate_status || {};
    el.innerHTML = `
        <div class="opening-funnel-verdict opening-funnel-${tone}">
            <strong>${escHtml(profile.label || strategyLearningText(profile.id) || profile.id || '\u5f53\u524d\u57fa\u7ebf')}</strong>
            <span>${escHtml(strategyLearningIsReadable(schedule.reason) ? schedule.reason : '\u4f7f\u7528\u5f53\u524d\u7b56\u7565\u8c03\u5ea6\u7ed3\u679c\u3002')}</span>
        </div>
            <div class="opening-funnel-kpis strategy-learning-kpis">
            <div><span>\u8bad\u7ec3\u4ea4\u6613\u6570</span><strong>${Number(totals.training_trade_count || 0)} / ${Number(totals.trade_count_target || 0)}</strong></div>
            <div><span>\u51c0\u6536\u76ca</span><strong style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(pnl)} U</strong></div>
            <div><span>\u80dc\u7387</span><strong>${pctLabel(totals.win_rate || 0, 1)}</strong></div>
            <div><span>\u7b56\u7565\u590d\u76d8</span><strong>${Number(totals.reflection_count || 0)} / ${Number(totals.reflection_total_count || 0)}</strong></div>
            <div><span>\u4f4e\u4ea4\u6613\u91cf\u60e9\u7f5a</span><strong>${totals.low_trade_count_penalty ? '\u5df2\u542f\u7528' : '\u672a\u89e6\u53d1'}</strong></div>
            <div><span>LLM\u5019\u9009</span><strong>${llm.candidate_count || 0}${llm.last_error ? ' · \u6709\u9519\u8bef' : ''}</strong></div>
        </div>`;
}

function renderStrategyLearningProblems(data) {
    const el = document.getElementById('strategy-learning-problems');
    if (!el) return;
    const problems = data.feedback?.problems || [];
    if (!problems.length) {
        el.innerHTML = '<div class="opening-funnel-empty">\u6682\u672a\u53d1\u73b0\u9700\u8981\u8c03\u5ea6\u4ecb\u5165\u7684\u4e3b\u8981\u95ee\u9898\u3002</div>';
        return;
    }
    el.innerHTML = problems.map(item => `
        <div class="opening-funnel-row strategy-learning-problem ${escHtml(item.severity || 'medium')}">
            <div><strong>${escHtml(strategyLearningIsReadable(item.label) ? item.label : strategyLearningText(item.key))}</strong><span>${escHtml(item.key || '')}</span></div>
            <em>${escHtml(item.severity || '')}</em>
        </div>`).join('');
}

function renderStrategyLearningSides(data) {
    const el = document.getElementById('strategy-learning-sides');
    if (!el) return;
    const sides = data.feedback?.side_performance || {};
    el.innerHTML = ['long', 'short'].map(side => {
        const row = sides[side] || {};
        const pnl = Number(row.pnl || 0);
        return `
            <div class="opening-funnel-row opening-funnel-symbol-row">
                <div><strong>${side === 'long' ? '\u591a\u5355' : '\u7a7a\u5355'} · ${escHtml(row.state || 'neutral')}</strong><span>${Number(row.count || 0)} \u7b14\uff0c\u80dc\u7387 ${pctLabel(row.win_rate || 0, 1)}\uff0c\u5747\u503c ${signedMoney(row.avg_pnl || 0)} U</span></div>
                <div class="opening-funnel-bar"><span style="width:${Math.max(4, Math.min(Math.abs(pnl), 100))}%;background:${pnl >= 0 ? 'var(--green)' : 'var(--red)'};"></span></div>
                <em style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(pnl)} U</em>
            </div>`;
    }).join('');
}

function renderStrategyLearningRelease(data) {
    const el = document.getElementById('strategy-learning-release');
    if (!el) return;
    const pressure = data.feedback?.open_position_pressure || {};
    const candidates = pressure.release_candidates || [];
    const header = `<div class="opening-funnel-row"><div><strong>\u4ed3\u4f4d\u5360\u7528 ${Number(pressure.open_count || 0)} / ${Number(pressure.max_open_positions || 0)}</strong><span>\u4e8f\u635f\u4ed3 ${Number(pressure.losing_open_count || 0)}\uff0c\u6d6e\u4e8f ${signedMoney(pressure.losing_unrealized_pnl || 0)} U</span></div><em>${pressure.full_position_pressure ? '\u6ee1\u4ed3\u538b\u529b' : '\u6b63\u5e38'}</em></div>`;
    if (!candidates.length) {
        el.innerHTML = header + '<div class="opening-funnel-empty">\u6682\u65e0\u91ca\u653e\u5019\u9009\u3002</div>';
        return;
    }
    el.innerHTML = header + candidates.map(row => `
        <div class="opening-funnel-row opening-funnel-symbol-row">
            <div><strong>${escHtml(row.symbol || '-')} · ${escHtml(row.side || '-')}</strong><span>${escHtml(row.model_name || '')}</span></div>
            <em style="color:${Number(row.unrealized_pnl || 0) >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(row.unrealized_pnl || 0)} U</em>
        </div>`).join('');
}

function renderStrategyLearningExperts(data) {
    const el = document.getElementById('strategy-learning-experts');
    if (!el) return;
    const quality = data.feedback?.decision_quality || {};
    const statuses = quality.model_timing_status_counts || {};
    const missing = quality.missing_expert_counts || {};
    const statusRows = Object.entries(statuses).slice(0, 6).map(([key, count]) => `<span>${escHtml(key)}: ${Number(count || 0)}</span>`).join('');
    const missingRows = Object.entries(missing).slice(0, 6).map(([key, count]) => `<span>${escHtml(key)}: ${Number(count || 0)}</span>`).join('');
    el.innerHTML = `
        <div class="opening-funnel-row"><div><strong>\u5b8c\u6574\u6027\u62e6\u622a ${Number(quality.expert_integrity_blocks || 0)}</strong><span>fallback \u5f00\u4ed3\u7387 ${pctLabel(quality.fallback_entry_rate || 0, 1)}</span></div><em>${Number(quality.entry_signals || 0)} \u4fe1\u53f7</em></div>
        <div class="strategy-learning-chip-row">${statusRows || '<span>\u6682\u65e0\u72b6\u6001\u5f02\u5e38</span>'}</div>
        <div class="strategy-learning-chip-row">${missingRows || '<span>\u6682\u65e0\u7f3a\u5931\u4e13\u5bb6</span>'}</div>`;
}

function renderStrategyLearningReflections(data) {
    const el = document.getElementById('strategy-learning-reflections');
    if (!el) return;
    const reflection = data.feedback?.reflection_feedback || {};
    const total = Number(reflection.total_count || 0);
    const training = Number(reflection.training_count || 0);
    const pnl = Number(reflection.fee_adjusted_pnl || 0);
    const mistakes = reflection.top_mistakes || [];
    const improvements = reflection.top_improvements || [];
    const mistakeRows = mistakes.slice(0, 3).map(item => `<span>${escHtml(strategyLearningShort(item.summary, 34))}: ${Number(item.count || 0)}</span>`).join('');
    const improvementRows = improvements.slice(0, 3).map(item => `<span>${escHtml(strategyLearningShort(item.summary, 34))}: ${Number(item.count || 0)}</span>`).join('');
    if (!total) {
        el.innerHTML = '<div class="opening-funnel-empty">\u6682\u65e0\u7b56\u7565\u590d\u76d8\u6837\u672c\u3002\u4ea4\u6613\u95ed\u73af\u5b8c\u6210\u540e\u4f1a\u81ea\u52a8\u7f16\u8bd1\u6210\u7ed3\u6784\u5316\u53cd\u9988\u3002</div>';
        return;
    }
    el.innerHTML = `
        <div class="opening-funnel-row"><div><strong>\u590d\u76d8\u6837\u672c ${training} / ${total}</strong><span>\u624b\u52a8\u5e73\u4ed3\u6392\u9664 ${Number(reflection.excluded_manual_count || 0)}\uff0c\u4e0d\u8fdb\u5165\u6a21\u578b\u8bad\u7ec3</span></div><em>${escHtml(reflection.policy || '')}</em></div>
        <div class="opening-funnel-row"><div><strong>\u8d39\u540e\u590d\u76d8\u76c8\u4e8f</strong><span>\u4e8f\u635f\u5e73\u5747\u6301\u6709 ${Number(reflection.avg_loss_hold_minutes || 0).toFixed(0)} \u5206\u949f\uff0c\u76c8\u5229\u5e73\u5747\u6301\u6709 ${Number(reflection.avg_win_hold_minutes || 0).toFixed(0)} \u5206\u949f</span></div><em style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'};">${signedMoney(pnl)} U</em></div>
        <div class="opening-funnel-row"><div><strong>\u5c0f\u76c8 / \u5927\u4e8f</strong><span>${Number(reflection.small_win_count || 0)} / ${Number(reflection.large_loss_count || 0)}\uff0c\u9519\u8bef ${Number(reflection.mistake_count || 0)}\uff0c\u6539\u8fdb ${Number(reflection.improvement_count || 0)}</span></div><em>\u590d\u76d8</em></div>
        <div class="strategy-learning-chip-row">${mistakeRows || '<span>\u6682\u65e0\u91cd\u590d\u9519\u8bef</span>'}</div>
        <div class="strategy-learning-chip-row">${improvementRows || '<span>\u6682\u65e0\u6539\u8fdb\u5efa\u8bae</span>'}</div>`;
}

function renderStrategyLearningProfiles(data) {
    const el = document.getElementById('strategy-learning-profiles');
    if (!el) return;
    const schedule = data.schedule || {};
    const activeId = schedule.active_profile?.id || data.active_profile?.id;
    const candidates = schedule.candidates || [];
    const backtestRows = schedule.backtest?.rows || [];
    const shadowRows = schedule.shadow_validation?.rows || [];
    const disabled = new Set(schedule.disabled_profiles || []);
    if (!candidates.length) {
        el.innerHTML = '<div class="opening-funnel-empty">\u6682\u65e0\u7b56\u7565\u5019\u9009\u3002</div>';
        return;
    }
    el.innerHTML = candidates.map(profile => {
        const bt = backtestRows.find(row => row.profile_id === profile.id) || {};
        const sh = shadowRows.find(row => row.profile_id === profile.id) || {};
        const isActive = profile.id === activeId;
        const isDisabled = disabled.has(profile.id);
        const profileIdArg = JSON.stringify(String(profile.id || ''));
        return `
            <div class="strategy-learning-profile ${isActive ? 'active' : ''} ${isDisabled ? 'disabled' : ''}">
                <div class="strategy-learning-profile-head"><strong>${escHtml(profile.label || strategyLearningText(profile.id) || profile.id)}</strong><span>${isActive ? '\u5f53\u524d\u542f\u7528' : escHtml(profile.status || '\u5019\u9009')}</span></div>
                <p>${escHtml(profile.description || '')}</p>
                <div class="strategy-learning-profile-metrics">
                    <span>\u8bc4\u5206 ${Number(bt.score || 0).toFixed(2)}</span><span>\u8d39\u540e ${signedMoney(bt.fee_adjusted_pnl || 0)} U</span><span>\u56de\u64a4 ${signedMoney(bt.max_drawdown || 0)} U</span><span>\u8fde\u7eed\u4e8f\u635f ${Number(bt.consecutive_losses || 0)}</span><span>\u4ea4\u6613 ${Number(bt.trade_count || 0)} / ${Number(bt.trade_count_target || 0)}</span><span>\u5f71\u5b50 ${Number(sh.shadow_score || 0).toFixed(2)}</span><span>\u63a2\u9488 ${Number(profile.params?.probe_fraction || 0) ? pctLabel(profile.params.probe_fraction, 1) : '-'}</span>
                </div>
                <div class="strategy-learning-profile-actions">
                    <button class="btn btn-sm" onclick='activateStrategyLearningProfile(${profileIdArg})'>\u542f\u7528</button>
                    <button class="btn btn-sm" onclick='setStrategyLearningProfileDisabled(${profileIdArg}, ${isDisabled ? 'false' : 'true'})'>${isDisabled ? '\u6062\u590d\u5019\u9009' : '\u7981\u7528\u5019\u9009'}</button>
                </div>
            </div>`;
    }).join('');
}

function renderStrategyLearningEvents(data) {
    const el = document.getElementById('strategy-learning-events');
    if (!el) return;
    const ev = data.feedback?.event_feedback || {};
    const blocks = ev.top_block_reasons || [];
    const blockRows = blocks.slice(0, 5).map(item => `<div class="opening-funnel-row"><div><strong>${escHtml(strategyLearningShort(item.reason, 56))}</strong><span>\u963b\u65ad\u539f\u56e0</span></div><em>${Number(item.count || 0)}</em></div>`).join('');
    el.innerHTML = `
        <div class="opening-funnel-row"><div><strong>\u5f52\u56e0\u8986\u76d6 ${pctLabel(ev.attribution_coverage || 0, 1)}</strong><span>${Number(ev.total_events || 0)} \u4e2a\u4e8b\u4ef6\uff0c\u7f3a\u753b\u50cf ${Number(ev.missing_profile_events || 0)}</span></div><em>${Number(ev.manual_close_events || 0)} \u624b\u52a8</em></div>
        <div class="opening-funnel-row"><div><strong>\u6ee1\u4ed3/\u5bb9\u91cf ${Number(ev.max_position_blocks || 0)}</strong><span>fallback ${Number(ev.fallback_blocks || 0)}\uff0c\u6267\u884c\u5f02\u5e38 ${Number(ev.execution_errors || 0)}</span></div><em>\u4e8b\u4ef6</em></div>
        ${blockRows || '<div class="opening-funnel-empty">\u6682\u65e0\u963b\u65ad\u539f\u56e0\u7edf\u8ba1\u3002</div>'}`;
}

function renderStrategyLearningGuard(data) {
    const el = document.getElementById('strategy-learning-guard');
    if (!el) return;
    const guard = data.runtime_guard || {};
    const stateInfo = data.state || {};
    const disabled = Object.keys(stateInfo.disabled_profiles || {});
    const reasonRows = (guard.reasons || []).map(reason => `<span>${escHtml(strategyLearningText(reason))}</span>`).join('');
    el.innerHTML = `
        <div class="opening-funnel-row"><div><strong>${guard.should_rollback ? '\u5df2\u89e6\u53d1\u56de\u6eda\u6761\u4ef6' : '\u62a4\u680f\u6b63\u5e38'}</strong><span>\u5f53\u524d\u753b\u50cf ${escHtml(guard.profile_id || '-')}</span></div><em>${guard.should_rollback ? '\u56de\u6eda' : '\u89c2\u5bdf'}</em></div>
        <div class="strategy-learning-chip-row">${reasonRows || '<span>\u6682\u65e0\u56de\u6eda\u539f\u56e0</span>'}</div>
        <div class="opening-funnel-row"><div><strong>\u7981\u7528\u5019\u9009 ${disabled.length}</strong><span>${escHtml(disabled.slice(0, 4).join(', ') || '\u65e0')}</span></div><em>\u53ef\u4eba\u5de5\u6062\u590d</em></div>`;
}

function renderStrategyLearningRecentEvents(data) {
    const el = document.getElementById('strategy-learning-recent-events');
    if (!el) return;
    const rows = data.feedback?.event_feedback?.recent_events || [];
    if (!rows.length) {
        el.innerHTML = '<div class="opening-funnel-empty">\u6682\u65e0\u7b56\u7565\u4e8b\u4ef6\u3002\u4e0b\u4e00\u8f6e\u51b3\u7b56\u3001\u62e6\u622a\u3001\u6267\u884c\u6216\u624b\u52a8\u5e73\u4ed3\u540e\u4f1a\u81ea\u52a8\u8bb0\u5f55\u3002</div>';
        return;
    }
    el.innerHTML = rows.slice(0, 18).map(row => `
        <div class="strategy-learning-event-row ${escHtml(row.severity || 'info')}">
            <div><strong>${escHtml(row.event_type || '-')} · ${escHtml(row.event_status || '-')}</strong><span>${escHtml(row.symbol || '-')} ${escHtml(row.action || '')} · ${escHtml(row.profile_id || '\u65e0\u753b\u50cf')}</span></div>
            <p title="${escHtml(row.reason || '')}">${escHtml(strategyLearningShort(row.reason || '-', 96))}</p>
            <em>${row.exclude_from_training ? '\u4e0d\u8bad\u7ec3' : '\u53ef\u5f52\u56e0'} · O${row.order_id || '-'} / P${row.position_id || '-'}</em>
        </div>`).join('');
}
