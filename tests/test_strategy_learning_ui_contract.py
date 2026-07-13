from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEW = (ROOT / "web_dashboard/static/js/strategy_learning_view.js").read_text(
    encoding="utf-8"
)
DASHBOARD = (ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
HTML = (ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")


def test_strategy_learning_ui_is_fee_after_return_observation_only() -> None:
    assert "authoritative_return_observation" in VIEW
    assert "realized_net_pnl_usdt" in VIEW
    assert "pnl_lower_hinge_usdt" in VIEW
    assert "profit_factor" in VIEW
    assert "生产权限隔离" in VIEW
    assert "专家/记忆/影子不能授权交易" in VIEW


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


def test_strategy_learning_console_labels_observation_boundary() -> None:
    assert "费后收益观察" in HTML
    assert "只读归因，不授权交易、仓位、杠杆或晋升" in HTML
    assert "权威收益分布" in HTML
    assert "权限隔离" in HTML
    assert "无交易、仓位、杠杆或晋升权限" in HTML
    assert 'id="strategy-learning-summary"' in HTML
    assert 'id="strategy-learning-sides"' in HTML
    assert 'id="strategy-learning-profiles"' in HTML
