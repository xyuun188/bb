from __future__ import annotations

import re
from pathlib import Path


def test_strategy_learning_ui_explains_sample_target_as_advisory() -> None:
    script = Path("web_dashboard/static/js/strategy_learning_view.js").read_text(
        encoding="utf-8"
    )

    assert "trade_count_target_is_entry_gate" in script
    assert "不是开仓门槛" in script
    assert "动态学习目标" in script
    assert not re.search(r"交易样本'.*training_trade_count.*\\s/\\s.*trade_count_target", script)


def test_strategy_learning_ui_explains_capacity_without_bare_ratio() -> None:
    script = Path("web_dashboard/static/js/strategy_learning_view.js").read_text(
        encoding="utf-8"
    )

    assert "基础容量不是固定开仓数量" in script
    assert "基础容量参考" in script
    assert "学习目标" in script
    assert not re.search(r"open_count[^\\n]+\\s/\\s[^\\n]+max_open_positions", script)
