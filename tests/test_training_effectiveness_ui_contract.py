from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_effectiveness_lives_inside_expert_memory_and_is_read_only():
    html = (ROOT / "web_dashboard/static/index.html").read_text(encoding="utf-8")
    script = (ROOT / "web_dashboard/static/js/dashboard.js").read_text(encoding="utf-8")
    assert 'data-expert-memory-view="training-effectiveness"' in html
    assert 'id="expert-memory-panel-training-effectiveness"' in html
    assert "fetchTrainingEffectivenessReport" in script
    assert "/api/training-effectiveness/report" in script
    assert "training-effectiveness" in script
    assert "结论不可用" in script
    assert "资金费贡献不能直接等同于模型预测能力" in script
    assert "影子市场机会（非真实盈利）" in script
    assert "page-training-effectiveness" not in html
