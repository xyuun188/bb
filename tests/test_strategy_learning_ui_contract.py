from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEW = (ROOT / "web_dashboard/static/js/strategy_learning_view.js").read_text(
    encoding="utf-8"
)
DASHBOARD = (ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
HTML = (ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
API = (ROOT / "web_dashboard/api/dashboard.py").read_text(encoding="utf-8")


def test_strategy_learning_ui_shows_dynamic_return_governance() -> None:
    assert "average_net_return_pct" in VIEW
    assert "return_lcb_pct" in VIEW
    assert "realized_net_pnl_usdt" in VIEW
    assert "profit_factor" in VIEW
    assert "production_influence_eligible" in VIEW
    assert "rejection_reasons" in VIEW
    assert "滚动收益率下界" in VIEW
    assert "影子收益率下界" in VIEW
    assert "执行所有权隔离" in VIEW


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
    assert "动态策略候选" in HTML
    assert "候选与拒绝原因" in HTML
    assert "缺失证据显示为空，不伪装成数值 0" in HTML
    assert "strategy_learning_view.js?v=20260713-dynamic-return-scheduler" in HTML
    assert 'id="strategy-learning-summary"' in HTML
    assert 'id="strategy-learning-sides"' in HTML
    assert 'id="strategy-learning-profiles"' in HTML


def test_strategy_learning_cache_is_invalidated_by_all_scheduler_evidence() -> None:
    assert "_strategy_learning_watermark_for_request" in API
    assert "func.max(Position.updated_at)" in API
    assert "func.max(ShadowBacktest.updated_at)" in API
    assert "func.max(StrategyLearningEvent.updated_at)" in API
    assert "await _strategy_learning_watermark_for_request(" in API
