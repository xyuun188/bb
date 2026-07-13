from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEW = (ROOT / "web_dashboard/static/js/strategy_learning_view.js").read_text(
    encoding="utf-8"
)
DASHBOARD = (ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
HTML = (ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
API = (ROOT / "web_dashboard/api/dashboard.py").read_text(encoding="utf-8")
STYLE = (ROOT / "web_dashboard/static/css/strategy_learning.css").read_text(
    encoding="utf-8"
)


def test_strategy_learning_ui_shows_dynamic_return_governance() -> None:
    assert "average_net_return_pct" in VIEW
    assert "return_lcb_pct" in VIEW
    assert "realized_net_pnl_usdt" in VIEW
    assert "profit_factor" in VIEW
    assert "production_influence_eligible" in VIEW
    assert "rejection_reasons" in VIEW
    assert "滚动收益率下界" in VIEW
    assert "影子收益率下界" in VIEW
    assert "动态费后收益执行链" in VIEW
    assert "leading_candidate" in VIEW
    assert "当前实际执行策略" in VIEW
    assert "没有候选策略在生产生效" in VIEW
    assert "全市场方向候选" in VIEW
    assert "行情状态方向候选" in VIEW
    assert "币种方向候选" in VIEW
    assert "排名首位 · 未生效" in VIEW
    assert "生产先验生效" in VIEW


def test_strategy_learning_ui_distinguishes_missing_values_from_zero() -> None:
    assert "value === null || value === undefined || value === ''" in VIEW
    assert "number === null ? '-'" in VIEW
    assert "这不是数值 0" in VIEW
    assert "observation_only" not in VIEW


def test_strategy_learning_ui_has_no_production_write_controls() -> None:
    forbidden = (
        "setStrategyLearningProfileDisabled",
        "activateStrategyLearningProfile",
        "clearStrategyLearningManualOverride",
        "/strategy-learning/profiles/",
        "/strategy-learning/rollback",
        "probe_fraction",
        "balanced_probe",
    )
    for token in forbidden:
        assert token not in VIEW
        assert token not in DASHBOARD
        assert token not in HTML


def test_strategy_learning_console_labels_dynamic_scheduler_contract() -> None:
    assert "动态收益策略调度" in HTML
    assert "权威收益率生成 · 滚动验证 · 成本完整影子治理" in HTML
    assert "已生成候选策略" in HTML
    assert "候选策略与生产状态" in HTML
    assert "排名不等于生效，缺失证据保持为空" in HTML
    assert "候选状态索引" in HTML
    assert "strategy_learning_view.js?v=20260713-scheduler-state-clarity" in HTML
    assert "strategy_learning.css?v=20260713-scheduler-state-clarity" in HTML
    assert 'id="strategy-learning-summary"' in HTML
    assert 'id="strategy-learning-sides"' in HTML
    assert 'id="strategy-learning-profiles"' in HTML


def test_strategy_learning_candidate_visual_state_matches_production_semantics() -> None:
    assert ".strategy-learning-profile-card.production-active" in STYLE
    assert ".strategy-learning-profile-card.leading" in STYLE
    assert ".strategy-learning-profile-card.blocked" in STYLE
    assert ".strategy-learning-profile-card.active" not in STYLE
    assert "strategyLearningCandidateGroups" in VIEW
    assert "strategyLearningCandidateIndex" in VIEW
    assert ".strategy-learning-panel-recent-events .strategy-learning-compact-head span" in STYLE


def test_strategy_learning_cache_is_invalidated_by_all_scheduler_evidence() -> None:
    assert "_strategy_learning_watermark_for_request" in API
    assert "func.max(Position.updated_at)" in API
    assert "func.max(ShadowBacktest.updated_at)" in API
    assert "func.max(StrategyLearningEvent.updated_at)" in API
    assert "await _strategy_learning_watermark_for_request(" in API
