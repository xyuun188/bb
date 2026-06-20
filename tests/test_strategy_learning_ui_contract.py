from __future__ import annotations

import re
from pathlib import Path


def test_strategy_learning_ui_explains_sample_target_as_advisory() -> None:
    script = Path("web_dashboard/static/js/strategy_learning_view.js").read_text(encoding="utf-8")

    assert "trade_count_target_is_entry_gate" in script
    assert "不是开仓门槛" in script
    assert "动态学习目标" in script
    assert not re.search(r"交易样本'.*training_trade_count.*\\s/\\s.*trade_count_target", script)


def test_strategy_learning_ui_explains_capacity_without_bare_ratio() -> None:
    script = Path("web_dashboard/static/js/strategy_learning_view.js").read_text(encoding="utf-8")

    assert "基础容量不是固定开仓数量" in script
    assert "基础容量参考" in script
    assert "学习目标" in script
    assert not re.search(r"open_count[^\\n]+\\s/\\s[^\\n]+max_open_positions", script)


def test_strategy_learning_profile_actions_show_immediate_feedback() -> None:
    script = Path("web_dashboard/static/js/strategy_learning_view.js").read_text(encoding="utf-8")
    dashboard_script = Path("web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    style = Path("web_dashboard/static/css/strategy_learning.css").read_text(encoding="utf-8")

    assert "function strategyLearningSetActionState" in dashboard_script
    assert "function strategyLearningWriteRequest" in dashboard_script
    assert "strategyLearningSetActionState(profileId, 'loading', statusText)" in dashboard_script
    assert "strategyLearningSetActionState(profileId, 'success'" in dashboard_script
    assert "strategyLearningSetActionState(profileId, 'error'" in dashboard_script
    assert "function strategyLearningActionFeedback" in script
    assert "strategyLearningActionFeedback(profile.id)" in script
    assert 'role="status" aria-live="polite"' in script
    assert "function strategyLearningActionButtonLabel" in script
    assert "const actionState = strategyLearningActionState(profile.id);" in script
    assert "const loadingAttrs = strategyLearningActionButtonAttrs(profile.id);" in script
    assert 'data-action-loading="true"' in script
    assert "处理中..." in script
    assert "恢复中..." in script
    assert "已恢复自动调度" in script
    assert "人工指定此策略" in script
    assert "已禁用 · 未生效" in script
    assert "禁用候选，当前没有参与自动调度" in script
    assert "setStrategyLearningProfileDisabled" in script
    assert "strategy-learning-action-btn" in script
    assert "await fetchStrategyLearning()" in dashboard_script
    assert ".strategy-learning-action-feedback" in style
    assert ".strategy-learning-action-feedback.loading" in style
    assert ".strategy-learning-action-feedback.success" in style
    assert ".strategy-learning-action-feedback.error" in style
    assert "#strategy-learning-auto-button.is-loading" in style
    assert "????" not in script


def test_strategy_learning_ui_classifies_llm_candidate_errors() -> None:
    script = Path("web_dashboard/static/js/strategy_learning_view.js").read_text(encoding="utf-8")

    assert "llm.last_error_kind" in script
    assert "候选模型响应超时" in script
    assert "候选模型接口请求失败" in script
    assert "候选模型返回格式不符合 JSON 结构要求" in script
    assert "候选模型未返回可解析 JSON" not in script
    assert "上次候选已过期，后台会按策略窗口自动刷新" in script
    assert "过期候选仅作观察参考，不会强行开仓" in script


def test_strategy_learning_console_uses_clear_stage_layout() -> None:
    html = Path("web_dashboard/static/index.html").read_text(encoding="utf-8")
    style = Path("web_dashboard/static/css/strategy_learning.css").read_text(encoding="utf-8")

    assert "strategy-learning-title-block" in html
    assert "strategy-learning-stage-diagnostics" in html
    assert "strategy-learning-stage-profiles" in html
    assert "strategy-learning-stage-audit" in html
    assert "诊断看板" in html
    assert "候选策略实验室" in html
    assert "执行审计轨迹" in html
    assert "strategy-learning-panel-problems" in html
    assert "strategy-learning-panel-candidates" in html
    assert "strategy-learning-panel-recent-events" in html
    assert ".strategy-learning-stage" in style
    assert ".strategy-learning-stage-head" in style
    assert ".strategy-learning-stage-kicker" in style
    assert ".strategy-learning-grid-diagnostics .strategy-learning-panel-problems" in style
    assert ".strategy-learning-panel-candidates" in style
    assert "grid-template-columns: minmax(420px, 0.88fr) minmax(520px, 1.12fr);" in style
    assert "align-items: start;" in style
    assert "columns: 2 420px;" in style
    assert "break-inside: avoid;" in style
    assert "grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr));" in style
    assert "grid-template-columns: repeat(auto-fit, minmax(min(100%, 142px), 1fr));" in style
    assert (
        ".strategy-learning-profile-card-stats .strategy-learning-profile-stat:nth-child(4)"
        in style
    )
    stats_fourth_block = style[
        style.index(
            ".strategy-learning-profile-card-stats .strategy-learning-profile-stat:nth-child(4)"
        ) : style.index(".strategy-learning-profile-stat.warn")
    ]
    assert "grid-column: 1 / -1;" not in stats_fourth_block
    assert "white-space: normal;" in style
    assert "overflow-wrap: anywhere;" in style
    assert "strategy-learning-grid > .strategy-learning-panel:nth-child" not in style
    assert "????" not in html
    assert "????" not in style
