function strategyLearningText(key) {
    const labels = {
        baseline_current: '系统基线',
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
        expert_fallback_overblocking: '专家 fallback 过多拦截开仓',
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
        manual: '人工指定',
        degraded: '退化',
        healthy: '健康',
        neutral: '中性',
        blocked: '硬风控阻断',
        degraded_missing_probe: '模型缺失降级探针',
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

function strategyLearningEventReasonCategoryLabel(category) {
    const labels = {
        okx_execution_error: 'OKX执行',
        execution_missing_result: '接口回报',
        crowded_side_cap: '单边上限',
        capacity_block: '仓位容量',
        expert_fallback_block: '专家fallback',
        unknown: '未记录',
        other: '其他',
    };
    return labels[category] || strategyLearningText(category) || '其他';
}

function strategyLearningIsReadable(text) {
    const value = String(text || '');
    const badTokens = ['\u951f', '\u95ff', '\u59d2', '\u5a34', '\u6fe1', '\u7f01', '\u9420', '\u95b8'];
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

function strategyLearningActionState(profileId) {
    const store = window.strategyLearningActionState || {};
    const key = String(profileId || 'auto');
    const item = store[key];
    if (!item) return null;
    const ageMs = Date.now() - Number(item.updatedAt || 0);
    const keepMs = item.status === 'error' ? 30000 : 12000;
    if (item.status !== 'loading' && ageMs > keepMs) {
        delete store[key];
        return null;
    }
    return item;
}

function strategyLearningActionFeedback(profileId) {
    const item = strategyLearningActionState(profileId);
    if (!item) return '';
    const status = String(item.status || 'info');
    const label = status === 'loading' ? '处理中' : status === 'success' ? '已生效' : '操作失败';
    return `<div class="strategy-learning-action-feedback ${strategyLearningEsc(status)}" role="status" aria-live="polite"><strong>${label}</strong><span>${strategyLearningEsc(item.message || '')}</span></div>`;
}

function strategyLearningActionButtonAttrs(profileId) {
    const item = strategyLearningActionState(profileId);
    return item?.status === 'loading' ? 'disabled data-action-loading="true"' : '';
}

function strategyLearningActionButtonLabel(profileId, label) {
    const item = strategyLearningActionState(profileId);
    return item?.status === 'loading' ? '处理中...' : label;
}

function updateStrategyLearningAutoButton(data) {
    const button = document.getElementById('strategy-learning-auto-button');
    if (!button) return;
    const manualId = strategyLearningManualProfile(data);
    const actionState = strategyLearningActionState('auto');
    button.classList.toggle('is-loading', actionState?.status === 'loading');
    button.classList.toggle('is-success', actionState?.status === 'success');
    button.classList.toggle('is-error', actionState?.status === 'error');
    if (actionState?.status === 'loading') {
        button.disabled = true;
        button.textContent = '恢复中...';
        button.title = actionState.message || '正在恢复系统自动调度。';
        button.classList.add('btn-accent');
        return;
    }
    if (manualId) {
        button.disabled = false;
        button.textContent = '取消人工指定';
        button.title = `当前人工指定 ${manualId}，点击后恢复系统自动调度。`;
        button.classList.add('btn-accent');
    } else {
        button.disabled = true;
        button.textContent = actionState?.status === 'success' ? '已恢复自动调度' : '当前自动调度';
        button.title = '当前没有人工指定策略，系统会根据复盘、回测、影子验证和护栏自动选择策略画像。';
        button.classList.remove('btn-accent');
    }
}

function strategyLearningReason(data) {
    const schedule = data?.schedule || {};
    const profile = schedule.active_profile || data?.active_profile || {};
    if (strategyLearningIsReadable(schedule.reason)) return schedule.reason;
    if (strategyLearningIsManualLocked(data)) {
        return `人工指定策略画像 ${strategyLearningManualProfile(data)}，自动调度暂不覆盖。`;
    }
    const profileId = profile.id || 'baseline_current';
    if (profileId === 'loss_release') return '检测到满仓压力、亏损仓占位或亏损拖延，系统自动切到亏损释放画像。';
    if (profileId === 'balanced_probe') return '开仓样本不足、专家 fallback 拦截或错过机会偏多，系统自动切到平衡探针画像。';
    if (profileId === 'winner_hold') return '盈利仓小盈过多且大亏存在，系统自动切到赢家持仓优化画像。';
    if (String(profileId).endsWith('_side_recovery')) return '某一方向近期表现退化，系统自动切到方向恢复画像。';
    if (profile.source === 'llm_structured_candidate') return '结构化候选已通过约束，系统自动进入小仓探针调度。';
    return '自动调度未发现需要切换的高优先级问题，使用系统基线兜底。';
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

function strategyLearningEventTypeLabel(value) {
    const labels = {
        decision_logged: '决策记录',
        execution_attempt: '执行提交',
        execution_result: '执行结果',
        execution_error: '执行异常',
        manual_close: '手动平仓',
        risk_block: '风险阻断',
    };
    return labels[value] || strategyLearningText(value);
}

function strategyLearningActionLabel(value) {
    const labels = {
        long: '做多',
        short: '做空',
        close_long: '平多',
        close_short: '平空',
        hold: '观望',
    };
    return labels[value] || value || '-';
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

function strategyLearningCount(value, unit = '条') {
    return `${Number(value || 0)} ${unit}`;
}

function strategyLearningSourceLabel(source) {
    const value = String(source || '').trim();
    if (value === 'current_system' || value === 'baseline') return '当前系统';
    if (value === 'feedback_generator') return '反馈生成';
    if (value === 'llm_structured_candidate') return 'LLM 结构化候选';
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
    if (!flags.length) flags.push('接近基线');
    flags.push(row?.fallback_safety === 'probe_core_required' ? '核心专家必需' : row?.fallback_safety === 'too_loose' ? 'fallback 过宽' : '专家约束正常');
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
    updateStrategyLearningAutoButton(data);
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
        updated.textContent = `${data.mode === 'live' ? '实盘' : '模拟盘'} · 最近 ${data.window_hours || 24} 小时 · ${new Date().toLocaleTimeString('zh-CN', { hour12: false })}`;
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
    const disabled = Array.isArray(schedule.disabled_profiles) ? schedule.disabled_profiles : [];
    const problemCount = (feedback.problems || []).length;
    const pnl = Number(totals.net_pnl || 0);
    const manualLocked = strategyLearningIsManualLocked(data);
    const activeName = profile.label || strategyLearningText(profile.id) || profile.id || '系统基线';
    const modeLabel = manualLocked ? '人工指定中' : '自动调度中';
    const modeTone = manualLocked ? 'warn' : 'good';
    const sampleTarget = Number(totals.trade_count_target || 0);
    const sampleTargetIsEntryGate = Boolean(totals.trade_count_target_is_entry_gate);
    const sampleTargetHint = totals.low_trade_count_penalty
        ? `低于动态学习目标 ${sampleTarget} 条，只降低策略置信并扩大受控探针`
        : `动态学习目标 ${sampleTarget} 条，${sampleTargetIsEntryGate ? '会影响开仓门槛' : '不是开仓门槛'}`;
    const guardTone = guard.should_rollback ? 'bad' : 'good';
    const guardLabel = guard.should_rollback ? '护栏预警' : '护栏正常';
    const source = profile.source || (profile.id === 'baseline_current' ? 'baseline' : 'feedback_generator');
    const llmError = String(llm.last_error || '').trim();
    const llmCached = Array.isArray(llm.cached_candidates) ? llm.cached_candidates : [];
    const llmStateLabel = llm.cache_status === 'current'
        ? '缓存匹配当前反馈'
        : (llm.cache_status === 'stale' ? '缓存已过期' : '暂无缓存');
    const llmNotice = llmError
        ? `<div class="strategy-learning-inline-alert warn"><strong>\u52a8\u6001\u5019\u9009\u672a\u751f\u6548</strong><span>\u5019\u9009\u6a21\u578b\u672a\u8fd4\u56de\u53ef\u89e3\u6790 JSON\uff0c\u5f53\u524d\u4f7f\u7528\u5185\u7f6e\u53d7\u63a7\u7b56\u7565\u515c\u5e95\u3002</span><em>${strategyLearningEsc(strategyLearningShort(llmError, 96))}</em></div>`
        : '';
    const llmCacheNotice = llmCached.length
        ? `<div class="strategy-learning-inline-alert ${llm.cache_status === 'current' ? '' : 'warn'}"><strong>LLM 结构化候选</strong><span>${strategyLearningEsc(llmStateLabel)} · ${strategyLearningEsc(llm.last_model || llm.source || '-')} · ${strategyLearningEsc(strategyLearningTime(llm.cached_at))}</span><em>${llmCached.map(item => strategyLearningEsc(item.label || item.id || '-')).join(' / ')}</em></div>`
        : '';

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
                    ${strategyLearningMeta('控制权', manualLocked ? `人工指定 ${strategyLearningManualProfile(data)}` : '系统自动')}
                    ${strategyLearningMeta('来源', strategyLearningSourceLabel(source))}
                    ${strategyLearningMeta('禁用候选', `${disabled.length}`)}
                    ${strategyLearningMeta('回滚护栏', guardLabel, guardTone)}
                </div>
                ${llmNotice}
                ${llmCacheNotice}
            </div>
            <div class="strategy-learning-metric-grid">
                ${strategyLearningMetric('净收益', `${strategyLearningMoney(pnl)} U`, '训练窗口', pnl >= 0 ? 'good' : 'bad')}
                ${strategyLearningMetric('胜率', strategyLearningPct(totals.win_rate || 0, 1), `${strategyLearningCount(totals.win_count, '笔')}盈利 · ${strategyLearningCount(totals.training_trade_count, '笔')}训练`, 'neutral')}
                ${strategyLearningMetric('交易样本', strategyLearningCount(totals.training_trade_count), sampleTargetHint, totals.low_trade_count_penalty ? 'warn' : 'good')}
                ${strategyLearningMetric('复盘覆盖', strategyLearningCount(totals.reflection_count), `总复盘 ${Number(totals.reflection_total_count || 0)} 条，手动干预样本不训练`, 'neutral')}
                ${strategyLearningMetric('问题数', Number(problemCount || 0), '当前归因', problemCount ? 'warn' : 'good')}
                ${strategyLearningMetric('LLM 候选', Number(llm.candidate_count || 0), llm.last_error ? '生成有错误' : '结构化候选', llm.last_error ? 'warn' : 'neutral')}
            </div>
            <div class="strategy-learning-inline-help">交易样本目标由窗口、信号、影子机会和复盘数量动态计算，只用于学习评分置信；真正是否开仓还会经过行情质量、盈利期望、账户风险、持仓质量和硬风控共同判断。</div>
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
    const runtime = data.schedule?.runtime || {};
    const candidates = pressure.release_candidates || [];
    const openCount = Number(pressure.open_count || 0);
    const baseCapacity = Number(pressure.max_open_positions || 0);
    const learnedTarget = Number(runtime.target_position_groups || runtime.target_open_position_groups || 0);
    const rotationSlots = Number(runtime.rotation_slots || 0);
    const capacityText = [
        baseCapacity ? `基础容量参考 ${baseCapacity} 组` : '',
        learnedTarget ? `学习目标 ${learnedTarget} 组` : '',
        rotationSlots ? `轮换槽 ${rotationSlots} 个` : '',
    ].filter(Boolean).join(' · ') || '等待容量评估';
    const header = `
        <div class="strategy-learning-compact-head">
            <strong>${openCount} 组持仓</strong>
            <span>${strategyLearningEsc(capacityText)} · 亏损仓 ${Number(pressure.losing_open_count || 0)} · ${strategyLearningMoney(pressure.losing_unrealized_pnl || 0)} U</span>
            <em>${pressure.full_position_pressure ? '释放队列优先' : '容量正常'}</em>
        </div>`;
    if (!candidates.length) {
        el.innerHTML = header + '<div class="strategy-learning-empty">暂无释放候选。基础容量不是固定开仓数量，系统会再结合账户风险、行情质量、释放队列和硬风控计算实际开仓节奏。</div>'; 
        return;
    }
    el.innerHTML = header + '<div class="strategy-learning-inline-help">基础容量来自配置，学习目标和轮换槽由策略调度动态计算；低质量旧仓优先进入释放纪律，新机会只走受控探针。</div>' + `<div class="strategy-learning-mini-list">${candidates.slice(0, 6).map(row => `
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
    const lossSamples = Number(reflection.loss_sample_count || 0);
    const winSamples = Number(reflection.win_sample_count || 0);
    const mistakeRows = mistakes.slice(0, 3).map(item => `<span>${strategyLearningEsc(strategyLearningShort(item.summary, 32))} <b>${Number(item.count || 0)}</b></span>`).join('');
    const improvementRows = improvements.slice(0, 3).map(item => `<span>${strategyLearningEsc(strategyLearningShort(item.summary, 32))} <b>${Number(item.count || 0)}</b></span>`).join('');
    if (!total) {
        el.innerHTML = '<div class="strategy-learning-empty">暂无策略复盘样本。</div>';
        return;
    }
    el.innerHTML = `
        <div class="strategy-learning-compact-head">
            <strong class="${pnl >= 0 ? 'good' : 'bad'}">${strategyLearningMoney(pnl)} U</strong>
            <span>训练复盘 ${training} 条 · 总复盘 ${total} 条 · 手动排除 ${Number(reflection.excluded_manual_count || 0)} 条</span>
            <em>费后</em>
        </div>
        <div class="strategy-learning-split-metrics">
            <span>亏损持有 <b>${lossSamples ? `${Number(reflection.avg_loss_hold_minutes || 0).toFixed(0)}m` : '暂无亏损样本'}</b></span>
            <span>盈利持有 <b>${winSamples ? `${Number(reflection.avg_win_hold_minutes || 0).toFixed(0)}m` : '暂无盈利样本'}</b></span>
            <span>小盈 / 大亏 <b>小盈 ${Number(reflection.small_win_count || 0)} · 大亏 ${Number(reflection.large_loss_count || 0)}</b></span>
        </div>
        <div class="strategy-learning-chip-row">${mistakeRows || '<span>暂无重复错误</span>'}</div>
        <div class="strategy-learning-chip-row muted">${improvementRows || '<span>暂无改进建议</span>'}</div>`;
}

function renderStrategyLearningProfiles(data) {
    const el = document.getElementById('strategy-learning-profiles');
    if (!el) return;
    const schedule = data.schedule || {};
    const activeId = schedule.active_profile?.id || data.active_profile?.id;
    const llm = data.llm_candidate_status || {};
    const scheduleCandidates = Array.isArray(schedule.candidates) ? schedule.candidates : [];
    const cachedLlmCandidates = (Array.isArray(llm.cached_candidates) ? llm.cached_candidates : [])
        .filter(item => item && !scheduleCandidates.some(profile => profile.id === item.id))
        .map(item => ({ ...item, status: 'cached', source: item.source || 'llm_structured_candidate', cached_only: true }));
    const candidates = scheduleCandidates.concat(cachedLlmCandidates);
    const backtestRows = schedule.backtest?.rows || [];
    const shadowRows = schedule.shadow_validation?.rows || [];
    const disabled = new Set(schedule.disabled_profiles || []);
    const disabledReasons = schedule.disabled_profile_reasons || {};
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
                const disabledReason = disabledReasons[profile.id] || {};
                const disabledText = strategyLearningShort(disabledReason.reason || '', 96);
                const profileIdArg = JSON.stringify(String(profile.id || ''));
                const fee = Number(bt.fee_adjusted_pnl || 0);
                const score = Number(bt.score || 0);
                const shadowScore = Number(sh.shadow_score || 0);
                const pass = bt.pass !== false;
                const shadowOk = sh.eligible !== false;
                const probePct = Number(profile.params?.probe_fraction || 0);
                const cardClass = [isActive ? 'active' : '', isDisabled ? 'disabled' : '', profile.cached_only ? 'cached' : '', pass && shadowOk ? '' : 'failed'].filter(Boolean).join(' ');
                const activeLabel = profile.id === 'baseline_current'
                    ? (isActive ? '自动兜底中' : '稳定兜底')
                    : (profile.cached_only ? (llm.cache_status === 'current' ? 'LLM缓存候选' : 'LLM过期缓存') : (isActive ? (manualLocked ? '人工指定中' : '自动调度选中') : (isDisabled ? '已禁用 · 未生效' : (pass && shadowOk ? '候选可用' : '未通过验证'))));
                const statusTone = isActive ? 'good' : isDisabled ? 'bad' : profile.cached_only || pass && shadowOk ? 'neutral' : 'warn';
                let actionButton = '<button class="btn btn-sm" disabled title="系统基线只是兜底画像；没有人工指定时，调度器会在它和更合适的画像之间自动选择。">系统兜底</button>'; 
                let disableButton = '<button class="btn btn-sm" disabled title="系统基线是自动调度兜底画像，不能在控制台禁用。">兜底保留</button>'; 
                if (profile.id !== 'baseline_current') {
                    const actionState = strategyLearningActionState(profile.id);
                    const disableAction = profile.cached_only || isDisabled || (isActive && manualLocked) || !pass || !shadowOk;
                    const actionTitle = isActive && manualLocked
                        ? '这个策略已经被人工指定；要交回系统调度请点页面顶部“取消人工指定”。'
                        : profile.cached_only
                            ? '这个 LLM 候选只存在于缓存展示中，当前反馈签名刷新并通过回测/影子验证后才能人工指定。'
                            : isDisabled
                            ? '这个策略已被禁用，先取消禁用后才能人工指定。'
                            : (!pass || !shadowOk)
                                ? '这个策略还没有同时通过回测和影子验证，不能人工指定。'
                                : '人工指定后，自动调度不会覆盖它；需要交回系统调度时点页面顶部“取消人工指定”。';
                    const actionLabel = isActive && manualLocked ? '已人工指定' : '人工指定此策略';
                    const loadingAttrs = strategyLearningActionButtonAttrs(profile.id);
                    actionButton = `<button class="btn btn-sm strategy-learning-action-btn" data-profile-id="${strategyLearningEsc(profile.id || '')}" ${disableAction && actionState?.status !== 'loading' ? 'disabled' : loadingAttrs} title="${strategyLearningEsc(actionTitle)}" onclick='activateStrategyLearningProfile(${profileIdArg})'>${strategyLearningActionButtonLabel(profile.id, actionLabel)}</button>`;
                    disableButton = profile.cached_only
                        ? '<button class="btn btn-sm" disabled title="缓存展示候选需要刷新进入当前调度列表后才能禁用。">仅展示</button>'
                        : `<button class="btn btn-sm strategy-learning-action-btn" data-profile-id="${strategyLearningEsc(profile.id || '')}" ${loadingAttrs} title="${isDisabled ? '允许系统重新评估这个策略。' : '临时禁用这个策略，自动调度不会选它。'}" onclick='setStrategyLearningProfileDisabled(${profileIdArg}, ${isDisabled ? 'false' : 'true'})'>${strategyLearningActionButtonLabel(profile.id, isDisabled ? '取消禁用' : '禁用策略')}</button>`;
                }
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
                        ${isDisabled ? `<div class="strategy-learning-profile-note">禁用候选，当前没有参与自动调度${disabledText ? `；原因：${strategyLearningEsc(disabledText)}` : '。'}</div>` : ''}
                        ${profile.cached_only ? `<div class="strategy-learning-profile-note">${llm.cache_status === 'current' ? '已生成但未进入当前调度列表。' : '候选签名与当前反馈不一致，等待下一次生成刷新后再参与调度。'}</div>` : ''}
                        <div class="strategy-learning-profile-card-stats">
                            ${strategyLearningProfileMetric('回测评分', score.toFixed(2), score >= 0 ? 'good' : 'bad')}
                            ${strategyLearningProfileMetric('费后收益', `${strategyLearningMoney(fee)} U`, fee >= 0 ? 'good' : 'bad')}
                            ${strategyLearningProfileMetric('最大回撤', `${strategyLearningMoney(bt.max_drawdown || 0)} U`)}
                            ${strategyLearningProfileMetric('交易数量', `${Number(bt.trade_count || 0)} 笔 · 学习目标 ${Number(bt.trade_count_target || 0)} 笔`, Number(bt.low_trade_count_penalty || 0) ? 'warn' : '')}
                            ${strategyLearningProfileMetric('影子评分', shadowScore.toFixed(2), shadowOk ? 'good' : 'bad')}
                            ${strategyLearningProfileMetric('探针仓位', probePct ? strategyLearningPct(probePct, 1) : '-')}
                        </div>
                        <div class="strategy-learning-chip-row strategy-learning-profile-chips">${strategyLearningShadowChips(sh)}</div>
                        <div class="strategy-learning-chip-row muted strategy-learning-profile-chips">${strategyLearningFixChips(bt.matched_fixes)}</div>
                        ${strategyLearningActionFeedback(profile.id)}
                        <div class="strategy-learning-profile-footer">
                            <span>${pass ? '回测通过' : '回测未通过'} · ${shadowOk ? '影子通过' : '影子未通过'} · ${sh.trade_count_guard?.low_trade_count ? '低交易量惩罚中' : '交易量约束正常'}</span>
                            <div class="strategy-learning-profile-actions">
                                ${actionButton}
                                ${disableButton}
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
    const coverage = ev.attributable_event_coverage ?? ev.attribution_coverage ?? 0;
    const attributableEvents = Number(ev.attributable_events || 0);
    const missingAttributable = Number(ev.attributable_missing_profile_events || 0);
    const nonAttributable = Number(ev.non_attributable_events || 0);
    const blockRows = blocks.slice(0, 5).map(item => {
        const label = item.reason || item.reason_label || '-';
        return `<div title="${strategyLearningEsc(label)}"><strong>${strategyLearningEsc(strategyLearningShort(label, 56))}</strong><span>${strategyLearningEsc(strategyLearningEventReasonCategoryLabel(item.category || 'other'))}</span><em>${Number(item.count || 0)}</em></div>`;
    }).join('');
    el.innerHTML = `
        <div class="strategy-learning-compact-head">
            <strong>${strategyLearningPct(coverage || 0, 1)}</strong>
            <span>${attributableEvents} 可归因事件 · 缺画像 ${missingAttributable}</span>
            <em>${Number(ev.manual_close_events || 0)} 手动</em>
        </div>
        <div class="strategy-learning-split-metrics">
            <span>满仓 <b>${Number(ev.max_position_blocks || 0)}</b></span>
            <span>fallback <b>${Number(ev.fallback_blocks || 0)}</b></span>
            <span>执行异常 <b>${Number(ev.execution_errors || 0)}</b></span>
        </div>
        <div class="strategy-learning-chip-row muted"><span>总事件 ${Number(ev.total_events || 0)}</span><span>非策略观察 ${nonAttributable}</span><span>总覆盖 ${strategyLearningPct(ev.attribution_coverage || 0, 1)}</span></div>
        <div class="strategy-learning-block-list">${blockRows || '<div>暂无阻断原因统计。</div>'}</div>`;
}

function renderStrategyLearningGuard(data) {
    const el = document.getElementById('strategy-learning-guard');
    if (!el) return;
    const guard = data.runtime_guard || {};
    const schedule = data.schedule || {};
    const disabled = Array.isArray(schedule.disabled_profiles) ? schedule.disabled_profiles : [];
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
        <div class="strategy-learning-event-feed">
            ${rows.slice(0, 24).map(row => {
                const status = row.event_status || '-';
                const severity = strategyLearningStatusClass(row.severity || status);
                const eventType = strategyLearningEventTypeLabel(row.event_type || '-');
                const action = strategyLearningActionLabel(row.action || row.side || '-');
                const profile = row.profile_id || '无画像';
                const refs = [
                    row.order_id ? `O${row.order_id}` : '',
                    row.position_id ? `P${row.position_id}` : '',
                ].filter(Boolean).join(' / ');
                const reason = row.reason_label || row.reason || '-';
                const rawReason = row.reason && row.reason_label && row.reason !== row.reason_label ? `原始：${row.reason}` : reason;
                return `
                    <div class="strategy-learning-event-card ${severity}">
                        <div class="strategy-learning-event-main">
                            <span class="strategy-learning-event-dot"></span>
                            <div>
                                <div class="strategy-learning-event-title">
                                    <strong>${strategyLearningEsc(eventType)}</strong>
                                    <span class="strategy-learning-table-pill ${severity}">${strategyLearningEsc(strategyLearningText(status))}</span>
                                </div>
                                <p title="${strategyLearningEsc(rawReason)}">${strategyLearningEsc(reason)}</p>
                            </div>
                        </div>
                        <div class="strategy-learning-event-aside">
                            <span><b>${strategyLearningEsc(row.symbol || '-')}</b>${strategyLearningEsc(action)}</span>
                            <span>${strategyLearningEsc(profile)}</span>
                            <em>${strategyLearningEsc(strategyLearningTime(row.created_at))}</em>
                            <em>${row.exclude_from_training ? '不训练' : '可归因'}${refs ? ` · ${strategyLearningEsc(refs)}` : ''}</em>
                        </div>
                    </div>`;
            }).join('')}
        </div>`;
}
