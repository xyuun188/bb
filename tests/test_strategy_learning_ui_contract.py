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
    assert "setStrategyLearningProfileDisabled" in script
    assert "strategy-learning-action-btn" in script
    assert "await fetchStrategyLearning()" in dashboard_script
    assert ".strategy-learning-action-feedback" in style
    assert ".strategy-learning-action-feedback.loading" in style
    assert ".strategy-learning-action-feedback.success" in style
    assert ".strategy-learning-action-feedback.error" in style
    assert "#strategy-learning-auto-button.is-loading" in style
    assert "????" not in script
