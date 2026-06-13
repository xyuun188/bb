function strategyLearningText(key) {
    const labels = {
        baseline_current: '当前基线',
        balanced_probe: '平衡探针',
        loss_release: '亏损释放',
        winner_hold: '赢家持仓优化',
        long_side_recovery: '多单恢复',
        short_side_recovery: '空单恢复',
        negative_realized_pnl: '最近净收益为负',
        low_trade_count: '交易样本不足',
        long_side_degraded: '多单侧表现退化',
        short_side_degraded: '空单侧表现退化',
        full_position_loss_pressure: '满仓且亏损仓占位',
        expert_fallback_overblocking: '专家 fallback 过多阻断开仓',
        missed_opportunities: '影子复盘显示错过机会',
        small_wins_large_losses: '小盈多但大亏存在',
        loss_hold_too_long: '亏损仓持有过久',
        max_position_blocks: '仓位上限阻断开仓',
        event_fallback_blocks: '事件显示 fallback 阻断',
        execution_errors: '执行异常',
        strategy_attribution_gap: '策略归因不完整',
        reflection_negative_pnl: '策略复盘费后亏损',
        reflection_loss_hold_too_long: '复盘显示亏损拖延过久',
        reflection_small_wins_large_losses: '复盘显示小盈大亏',
        trade_reflection_mistakes: '策略复盘重复错误',
        recent_net_pnl_guard: '近期净收益护栏',
        fallback_dependency_guard: 'fallback 依赖护栏',
        execution_error_guard: '执行异常护栏',
        insufficient_trade_samples: '交易样本不足',
        auto: '自动调度',
        manual: '手动锁定',
        degraded: '退化',
        healthy: '健康',
        neutral: '中性',
        blocked: '阻断',
        failed: '失败',
        executed: '已执行',
        recorded: '已记录',
        info: '信息',
        warn: '警告',
        error: '错误',
    };
    return labels[key] || key || '-';
}

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

function strategyLearningMoney(value) {
    if (typeof signedMoney === 'function') return signedMoney(Number(value || 0));
    const n = Number(value || 0);
    return `${n >= 0 ? '+' : ''}${n.toFixed(2)}`;
}

function strategyLearningPct(value, digits = 1) {
    if (typeof pctLabel === 'function') return pctLabel(value || 0, digits);
    return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function strategyLearningShort(value, maxLen = 44) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    return text.length > maxLen ? `${text.slice(0, maxLen - 1)}...` : text;
}

function strategyLearningIsReadable(text) {
    const value = String(text || '');
    const badTokens = ['�', '????', '锟', '榛樿', '浜哄', '鐩堝', '妫€', '绛栫暐', '璋冨害', '鍊欓'];
    return Boolean(value.trim()) && !badTokens.some(token => value.includes(token));
}

function strategyLearningManualProfile(data) {
    const schedule = data?.schedule || {};
    const stateInfo = data?.state || {};
    const manualId = String(schedule.manual_profile_id || stateInfo.manual_active_profile || '').trim();
    return manualId === 'baseline_current' ? '' : manualId;
}

function strategyLearningIsManualLocked(data) {
    const schedule = data?.schedule || {};
    return schedule.scheduler_mode === 'manual' || Boolean(strategyLearningManualProfile(data));
}

function strategyLearningReason(data) {
    const schedule = data?.schedule || {};
    const profile = schedule.active_profile || data?.active_profile || {};
    if (strategyLearningIsReadable(schedule.reason)) return schedule.reason;
    if (strategyLearningIsManualLocked(data)) {
        return `人工锁定策略画像 ${strategyLearningManualProfile(data)}，自动调度暂不覆盖。`;
    }
    const profileId = profile.id || 'baseline_current';
    if (profileId === 'loss_release') return '检测到满仓压力、亏损仓占位或亏损拖延，自动调度到亏损释放画像。';
    if (profileId === 'balanced_probe') return '开仓样本不足、专家 fallback 拦截或影子复盘错过机会偏多，自动调度到平衡探针画像。';
    if (profileId === 'winner_hold') return '盈利仓小盈过多且大亏存在，自动调度到赢家持仓优化画像。';
    if (String(profileId).endsWith('_side_recovery')) return '某一方向近期表现退化，自动调度到方向恢复画像。';
    return '自动调度未发现需要切换的高优先级问题，使用当前基线。';
}

function strategyLearningSeverityLabel(value) {
    if (value === 'high') return '高';
    if (value === 'medium') return '中';
    if (value === 'low') return '低';
    return strategyLearningText(value);
}

function strategyLearningStatusClass(value) {
    if (['high', 'bad', 'danger', 'error', 'failed'].includes(String(value))) return 'bad';
    if (['medium', 'warn', 'blocked'].includes(String(value))) return 'warn';
    if (['good', 'ok', 'healthy', 'executed'].includes(String(value))) return 'good';
    return 'neutral';
}

function strategyLearningMetric(label, value, hint = '', tone = 'neutral') {
    return `
        <div class="strategy-learning-metric ${strategyLearningEsc(tone)}">
            <span>${strategyLearningEsc(label)}</span>
            <strong>${value}</strong>
            ${hint ? `<em>${strategyLearningEsc(hint)}</em>` : ''}
        </div>`;
}

function strategyLearningMeta(label, value, tone = '') {
    return `<span class="strategy-learning-meta ${strategyLearningEsc(tone)}"><b>${strategyLearningEsc(label)}</b>${strategyLearningEsc(value)}</span>`;
}

function strategyLearningProfileMetric(label, value, tone = '') {
    return `<span class="strategy-learning-profile-stat ${strategyLearningEsc(tone)}"><b>${strategyLearningEsc(label)}</b>${value}</span>`;
}

function strategyLearningSourceLabel(source) {
    const value = String(source || '').trim();
    if (value === 'current_system') return '当前系统';
    if (value === 'feedback_generator') return '反馈生成';
    if (value === 'llm_structured_candidate') return 'LLM结构化候选';
    return value || '-';
}

function strategyLearningFixChips(items) {
    const rows = Array.isArray(items) ? items.slice(0, 4) : [];
    if (!rows.length) return '<span>暂无命中反馈</span>';
    return rows.map(item => `<span>${strategyLearningEsc(strategyLearningText(item))}</span>`).join('');
}

function strategyLearningShadowChips(row) {
    const flags = [];
    if (row?.would_increase_entries) flags.push('增加开仓');
    if (row?.would_reduce_blocks) flags.push('减少拦截');
    if (row?.would_release_losers) flags.push('释放亏损');
    if (row?.would_hold_winners) flags.push('拿住赢家');
    if (!flags.length) flags.push('维持基线');
    flags.push(row?.fallback_safety === 'probe_core_required' ? '核心专家必须可信' : row?.fallback_safety === 'too_loose' ? 'fallback过宽' : '严格专家');
    return flags.map(flag => `<span>${strategyLearningEsc(flag)}</span>`).join('');
}

function strategyLearningTime(value) {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '-';
    return date.toLocaleString('zh-CN', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
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
        updated.textContent = `${data.mode === 'live' ? '实盘' : '模拟盘'} · 最近 ${data.window_hours || 24} 小时 · ${new Date().toLocaleTimeString()}`;
    }
}

function renderStrategyLearningSummary(data) {
    const el = document.getElementById('strategy-learning-summary');
    if (!el) return;
    const feedback = data.feedback || {};
    const totals = feedback.totals || {};
    const schedule = data.schedule || {};
    const profile = schedule.active_profile || data.active_profile || {};
    const guard = data.runtime_guard || {};
    const llm = data.llm_candidate_status || {};
    const stateInfo = data.state || {};
    const disabled = Object.keys(stateInfo.disabled_profiles || {});
    const problemCount = (feedback.problems || []).length;
    const pnl = Number(totals.net_pnl || 0);
    const manualLocked = strategyLearningIsManualLocked(data);
    const activeName = profile.label || strategyLearningText(profile.id) || profile.id || '当前基线';
    const modeLabel = manualLocked ? '手动锁定' : '自动调度中';
    const modeTone = manualLocked ? 'warn' : 'good';
    const guardTone = guard.should_rollback ? 'bad' : 'good';
    const guardLabel = guard.should_rollback ? '护栏预警' : '护栏正常';
    const source = profile.source || (profile.id === 'baseline_current' ? 'baseline' : 'feedback_generator');

    el.innerHTML = `
        <div class="strategy-learning-command">
            <div class="strategy-learning-command-main">
                <div class="strategy-learning-command-eyebrow">${data.mode === 'live' ? '实盘' : '模拟盘'} · 策略调度</div>
                <div class="strategy-learning-command-title">
                    <strong>${strategyLearningEsc(activeName)}</strong>
                    <span class="strategy-learning-status-pill ${modeTone}">${modeLabel}</span>
                </div>
                <p>${strategyLearningEsc(strategyLearningReason(data))}</p>
                <div class="strategy-learning-command-meta">
                    ${strategyLearningMeta('策略ID', profile.id || '-')}
                    ${strategyLearningMeta('控制权', manualLocked ? strategyLearningManualProfile(data) : '系统自动')}
                    ${strategyLearningMeta('来源', source)}
                    ${strategyLearningMeta('禁用候选', `${disabled.length}`)}
                    ${strategyLearningMeta('回滚护栏', guardLabel, guardTone)}
                </div>
            </div>
            <div class="strategy-learning-metric-grid">
                ${strategyLearningMetric('净收益', `${strategyLearningMoney(pnl)} U`, '训练窗口', pnl >= 0 ? 'good' : 'bad')}
                ${strategyLearningMetric('胜率', strategyLearningPct(totals.win_rate || 0, 1), `${Number(totals.win_count || 0)} / ${Number(totals.training_trade_count || 0)}`, 'neutral')}
                ${strategyLearningMetric('交易数', `${Number(totals.training_trade_count || 0)} / ${Number(totals.trade_count_target || 0)}`, totals.low_trade_count_penalty ? '低交易量惩罚' : '样本达标', totals.low_trade_count_penalty ? 'warn' : 'good')}
                ${strategyLearningMetric('策略复盘', `${Number(totals.reflection_count || 0)} / ${Number(totals.reflection_total_count || 0)}`, '训练 / 总数', 'neutral')}
                ${strategyLearningMetric('问题数', Number(problemCount || 0), '当前归因', problemCount ? 'warn' : 'good')}
                ${strategyLearningMetric('LLM候选', Number(llm.candidate_count || 0), llm.last_error ? '有错误' : '结构化', llm.last_error ? 'warn' : 'neutral')}
            </div>
        </div>`;
}

function renderStrategyLearningProblems(data) {
    const el = document.getElementById('strategy-learning-problems');
    if (!el) return;
    const problems = data.feedback?.problems || [];
    if (!problems.length) {
        el.innerHTML = '<div class="strategy-learning-empty">暂无需要调度介入的主要问题。</div>';
        return;
    }
    el.innerHTML = `<div class="strategy-learning-issue-list">${problems.slice(0, 8).map(item => {
        const label = strategyLearningIsReadable(item.label) ? item.label : strategyLearningText(item.key);
        return `
            <div class="strategy-learning-issue ${strategyLearningEsc(item.severity || 'medium')}">
                <span>${strategyLearningEsc(strategyLearningSeverityLabel(item.severity))}</span>
                <div><strong>${strategyLearningEsc(label)}</strong><em>${strategyLearningEsc(item.key || '')}</em></div>
            </div>`;
    }).join('')}</div>`;
}

function renderStrategyLearningSides(data) {
    const el = document.getElementById('strategy-learning-sides');
    if (!el) return;
    const sides = data.feedback?.side_performance || {};
    el.innerHTML = `<div class="strategy-learning-side-board">${['long', 'short'].map(side => {
        const row = sides[side] || {};
        const pnl = Number(row.pnl || 0);
        const width = Math.max(8, Math.min(Math.abs(pnl) * 3, 100));
        const stateLabel = strategyLearningText(row.state || 'neutral');
        return `
            <div class="strategy-learning-side-card ${pnl >= 0 ? 'good' : 'bad'}">
                <div><span>${side === 'long' ? '多单' : '空单'}</span><strong>${strategyLearningMoney(pnl)} U</strong></div>
                <p>${Number(row.count || 0)} 笔 · 胜率 ${strategyLearningPct(row.win_rate || 0, 1)} · 均值 ${strategyLearningMoney(row.avg_pnl || 0)} U</p>
                <div class="strategy-learning-meter"><i style="width:${width}%;"></i></div>
                <em>${strategyLearningEsc(stateLabel)}</em>
            </div>`;
    }).join('')}</div>`;
}

function renderStrategyLearningRelease(data) {
    const el = document.getElementById('strategy-learning-release');
    if (!el) return;
    const pressure = data.feedback?.open_position_pressure || {};
    const candidates = pressure.release_candidates || [];
    const header = `
        <div class="strategy-learning-compact-head">
            <strong>${Number(pressure.open_count || 0)} / ${Number(pressure.max_open_positions || 0)}</strong>
            <span>亏损仓 ${Number(pressure.losing_open_count || 0)} · ${strategyLearningMoney(pressure.losing_unrealized_pnl || 0)} U</span>
            <em>${pressure.full_position_pressure ? '满仓压力' : '正常'}</em>
        </div>`;
    if (!candidates.length) {
        el.innerHTML = header + '<div class="strategy-learning-empty">暂无释放候选。</div>';
        return;
    }
    el.innerHTML = header + `<div class="strategy-learning-mini-list">${candidates.slice(0, 6).map(row => `
        <div>
            <strong>${strategyLearningEsc(row.symbol || '-')}</strong>
            <span>${strategyLearningEsc(row.side || '-')} · ${strategyLearningEsc(row.model_name || '')}</span>
            <em class="${Number(row.unrealized_pnl || 0) >= 0 ? 'good' : 'bad'}">${strategyLearningMoney(row.unrealized_pnl || 0)} U</em>
        </div>`).join('')}</div>`;
}

function renderStrategyLearningExperts(data) {
    const el = document.getElementById('strategy-learning-experts');
    if (!el) return;
    const quality = data.feedback?.decision_quality || {};
    const statuses = quality.model_timing_status_counts || {};
    const missing = quality.missing_expert_counts || {};
    const statusRows = Object.entries(statuses).slice(0, 6).map(([key, count]) => `<span>${strategyLearningEsc(key)} <b>${Number(count || 0)}</b></span>`).join('');
    const missingRows = Object.entries(missing).slice(0, 6).map(([key, count]) => `<span>${strategyLearningEsc(key)} <b>${Number(count || 0)}</b></span>`).join('');
    el.innerHTML = `
        <div class="strategy-learning-compact-head">
            <strong>${Number(quality.expert_integrity_blocks || 0)}</strong>
            <span>完整性拦截 · fallback ${strategyLearningPct(quality.fallback_entry_rate || 0, 1)}</span>
            <em>${Number(quality.entry_signals || 0)} 信号</em>
        </div>
        <div class="strategy-learning-chip-row">${statusRows || '<span>暂无状态异常</span>'}</div>
        <div class="strategy-learning-chip-row muted">${missingRows || '<span>暂无缺失专家</span>'}</div>`;
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
    const mistakeRows = mistakes.slice(0, 3).map(item => `<span>${strategyLearningEsc(strategyLearningShort(item.summary, 32))} <b>${Number(item.count || 0)}</b></span>`).join('');
    const improvementRows = improvements.slice(0, 3).map(item => `<span>${strategyLearningEsc(strategyLearningShort(item.summary, 32))} <b>${Number(item.count || 0)}</b></span>`).join('');
    if (!total) {
        el.innerHTML = '<div class="strategy-learning-empty">暂无策略复盘样本。</div>';
        return;
    }
    el.innerHTML = `
        <div class="strategy-learning-compact-head">
            <strong class="${pnl >= 0 ? 'good' : 'bad'}">${strategyLearningMoney(pnl)} U</strong>
            <span>复盘 ${training} / ${total} · 手动排除 ${Number(reflection.excluded_manual_count || 0)}</span>
            <em>费后</em>
        </div>
        <div class="strategy-learning-split-metrics">
            <span>亏损持有 <b>${Number(reflection.avg_loss_hold_minutes || 0).toFixed(0)}m</b></span>
            <span>盈利持有 <b>${Number(reflection.avg_win_hold_minutes || 0).toFixed(0)}m</b></span>
            <span>小盈 / 大亏 <b>${Number(reflection.small_win_count || 0)} / ${Number(reflection.large_loss_count || 0)}</b></span>
        </div>
        <div class="strategy-learning-chip-row">${mistakeRows || '<span>暂无重复错误</span>'}</div>
        <div class="strategy-learning-chip-row muted">${improvementRows || '<span>暂无改进建议</span>'}</div>`;
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
    const manualLocked = strategyLearningIsManualLocked(data);
    if (!candidates.length) {
        el.innerHTML = '<div class="strategy-learning-empty">暂无策略候选。</div>';
        return;
    }
    el.innerHTML = `
        <div class="strategy-learning-profile-board">
            ${candidates.map(profile => {
                const bt = backtestRows.find(row => row.profile_id === profile.id) || {};
                const sh = shadowRows.find(row => row.profile_id === profile.id) || {};
                const isActive = profile.id === activeId;
                const isDisabled = disabled.has(profile.id);
                const profileIdArg = JSON.stringify(String(profile.id || ''));
                const fee = Number(bt.fee_adjusted_pnl || 0);
                const score = Number(bt.score || 0);
                const shadowScore = Number(sh.shadow_score || 0);
                const pass = bt.pass !== false;
                const shadowOk = sh.eligible !== false;
                const probePct = Number(profile.params?.probe_fraction || 0);
                const cardClass = [isActive ? 'active' : '', isDisabled ? 'disabled' : '', pass && shadowOk ? '' : 'failed'].filter(Boolean).join(' ');
                const activeLabel = isActive ? (manualLocked ? '手动锁定' : '自动启用') : (isDisabled ? '已禁用' : (pass && shadowOk ? '候选可用' : '未通过'));
                const statusTone = isActive ? 'good' : isDisabled ? 'bad' : pass && shadowOk ? 'neutral' : 'warn';
                const actionButton = profile.id === 'baseline_current'
                    ? `<button class="btn btn-sm" onclick="rollbackStrategyLearning()">恢复自动</button>`
                    : `<button class="btn btn-sm" ${isActive && manualLocked ? 'disabled' : ''} onclick='activateStrategyLearningProfile(${profileIdArg})'>${isActive && manualLocked ? '已锁定' : '手动启用'}</button>`;
                return `
                    <article class="strategy-learning-profile-card ${strategyLearningEsc(cardClass)}">
                        <div class="strategy-learning-profile-card-head">
                            <div>
                                <strong>${strategyLearningEsc(profile.label || strategyLearningText(profile.id) || profile.id)}</strong>
                                <span>${strategyLearningEsc(profile.id || '-')} · ${strategyLearningEsc(strategyLearningSourceLabel(profile.source))}</span>
                            </div>
                            <span class="strategy-learning-table-pill ${statusTone}">${strategyLearningEsc(activeLabel)}</span>
                        </div>
                        <p>${strategyLearningEsc(strategyLearningShort(profile.description || '', 128))}</p>
                        <div class="strategy-learning-profile-card-stats">
                            ${strategyLearningProfileMetric('回测评分', score.toFixed(2), score >= 0 ? 'good' : 'bad')}
                            ${strategyLearningProfileMetric('费后收益', `${strategyLearningMoney(fee)} U`, fee >= 0 ? 'good' : 'bad')}
                            ${strategyLearningProfileMetric('最大回撤', `${strategyLearningMoney(bt.max_drawdown || 0)} U`)}
                            ${strategyLearningProfileMetric('交易数量', `${Number(bt.trade_count || 0)} / ${Number(bt.trade_count_target || 0)}`, Number(bt.low_trade_count_penalty || 0) ? 'warn' : '')}
                            ${strategyLearningProfileMetric('影子评分', shadowScore.toFixed(2), shadowOk ? 'good' : 'bad')}
                            ${strategyLearningProfileMetric('探针仓位', probePct ? strategyLearningPct(probePct, 1) : '-')}
                        </div>
                        <div class="strategy-learning-chip-row strategy-learning-profile-chips">${strategyLearningShadowChips(sh)}</div>
                        <div class="strategy-learning-chip-row muted strategy-learning-profile-chips">${strategyLearningFixChips(bt.matched_fixes)}</div>
                        <div class="strategy-learning-profile-footer">
                            <span>${pass ? '回测通过' : '回测未通过'} · ${shadowOk ? '影子通过' : '影子未通过'} · ${sh.trade_count_guard?.low_trade_count ? '低交易量惩罚中' : '交易量约束正常'}</span>
                            <div class="strategy-learning-profile-actions">
                                ${actionButton}
                                <button class="btn btn-sm" onclick='setStrategyLearningProfileDisabled(${profileIdArg}, ${isDisabled ? 'false' : 'true'})'>${isDisabled ? '恢复' : '禁用'}</button>
                            </div>
                        </div>
                    </article>`;
            }).join('')}
        </div>`;
}

function renderStrategyLearningEvents(data) {
    const el = document.getElementById('strategy-learning-events');
    if (!el) return;
    const ev = data.feedback?.event_feedback || {};
    const blocks = ev.top_block_reasons || [];
    const blockRows = blocks.slice(0, 5).map(item => `<div><strong>${strategyLearningEsc(strategyLearningShort(item.reason, 46))}</strong><em>${Number(item.count || 0)}</em></div>`).join('');
    el.innerHTML = `
        <div class="strategy-learning-compact-head">
            <strong>${strategyLearningPct(ev.attribution_coverage || 0, 1)}</strong>
            <span>${Number(ev.total_events || 0)} 事件 · 缺画像 ${Number(ev.missing_profile_events || 0)}</span>
            <em>${Number(ev.manual_close_events || 0)} 手动</em>
        </div>
        <div class="strategy-learning-split-metrics">
            <span>满仓 <b>${Number(ev.max_position_blocks || 0)}</b></span>
            <span>fallback <b>${Number(ev.fallback_blocks || 0)}</b></span>
            <span>执行异常 <b>${Number(ev.execution_errors || 0)}</b></span>
        </div>
        <div class="strategy-learning-block-list">${blockRows || '<div>暂无阻断原因统计。</div>'}</div>`;
}

function renderStrategyLearningGuard(data) {
    const el = document.getElementById('strategy-learning-guard');
    if (!el) return;
    const guard = data.runtime_guard || {};
    const stateInfo = data.state || {};
    const disabled = Object.keys(stateInfo.disabled_profiles || {});
    const reasonRows = (guard.reasons || []).map(reason => `<span>${strategyLearningEsc(strategyLearningText(reason))}</span>`).join('');
    el.innerHTML = `
        <div class="strategy-learning-guard-state ${guard.should_rollback ? 'danger' : 'ok'}">
            <strong>${guard.should_rollback ? '已触发回滚条件' : '护栏正常'}</strong>
            <span>${strategyLearningEsc(guard.profile_id || '-')}</span>
            <em>${guard.should_rollback ? '回滚' : '观察'}</em>
        </div>
        <div class="strategy-learning-chip-row">${reasonRows || '<span>暂无回滚原因</span>'}</div>
        <div class="strategy-learning-compact-head small">
            <strong>${disabled.length}</strong>
            <span>${strategyLearningEsc(disabled.slice(0, 4).join(', ') || '无')}</span>
            <em>禁用候选</em>
        </div>`;
}

function renderStrategyLearningRecentEvents(data) {
    const el = document.getElementById('strategy-learning-recent-events');
    if (!el) return;
    const rows = data.feedback?.event_feedback?.recent_events || [];
    if (!rows.length) {
        el.innerHTML = '<div class="strategy-learning-empty">暂无策略事件。</div>';
        return;
    }
    el.innerHTML = `
        <div class="strategy-learning-event-table">
            <div class="strategy-learning-event-row header">
                <span>事件</span><span>标的 / 动作</span><span>策略</span><span>原因</span><span>训练</span>
            </div>
            ${rows.slice(0, 24).map(row => {
                const status = row.event_status || '-';
                const severity = strategyLearningStatusClass(row.severity || status);
                return `
                    <div class="strategy-learning-event-row ${severity}">
                        <span><b>${strategyLearningEsc(row.event_type || '-')}</b><em>${strategyLearningEsc(strategyLearningText(status))}</em></span>
                        <span>${strategyLearningEsc(row.symbol || '-')} · ${strategyLearningEsc(row.action || '-')}</span>
                        <span>${strategyLearningEsc(row.profile_id || '无画像')}</span>
                        <p title="${strategyLearningEsc(row.reason || '')}">${strategyLearningEsc(strategyLearningShort(row.reason || '-', 88))}</p>
                        <span>${row.exclude_from_training ? '不训练' : '可归因'} · O${row.order_id || '-'} / P${row.position_id || '-'}</span>
                    </div>`;
            }).join('')}
        </div>`;
}
