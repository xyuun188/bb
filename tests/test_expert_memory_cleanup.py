from db.repositories.memory_repo import _memory_text_usable
from scripts.cleanup_expert_memory_text import translate_lesson, translate_market_pattern


def test_expert_memory_cleanup_translates_shadow_template_to_chinese():
    lesson, usable = translate_lesson(
        (
            "AI16Z/USDT 做空 missed opportunity. "
            "当时选择观望，但 60 分钟后做空方向涨跌收益约 10.03%。 "
            "If expected net return, fee coverage and loss probability are favorable, "
            "support a small profit-quality probe."
        ),
        "momentum_expert",
    )

    assert usable is True
    assert "missed opportunity" not in lesson
    assert "If expected net return" not in lesson
    assert "AI16Z/USDT 做空机会曾被观望错过" in lesson
    assert "预期净收益" in lesson


def test_expert_memory_cleanup_translates_trade_pattern():
    lesson, usable = translate_lesson(
        (
            "BTC/USDT short under pattern [BTC/USDT short, short_term, 5.0x, large_loss] "
            "ended as loss. Next time prioritize expected net return, fee coverage, "
            "loss probability and payoff ratio over win rate."
        ),
        "momentum_expert",
    )

    assert usable is True
    assert "ended as" not in lesson
    assert "BTC/USDT 做空" in lesson
    assert "结果为亏损" in lesson
    assert "不只看胜率" in lesson
    assert translate_market_pattern("BTC/USDT short, short_term, 5.0x, large_loss") == "BTC/USDT 做空，短线持仓，5.0x，大亏"
    assert translate_market_pattern("ETH/USDT 做空, longer_hold, 3.0x, 盈利") == "ETH/USDT 做空，较长持仓，3.0x，盈利"


def test_memory_repository_rejects_damaged_or_mojibake_memory_text():
    assert _memory_text_usable("BTC/USDT 做多机会曾被观望错过。当时选择观望。") is True
    assert _memory_text_usable("该笔历史记录的原始说明已损坏，无法准确还原。") is False
    assert _memory_text_usable("褰撴椂閫夋嫨瑙傛湜锛屼絾 10 鍒嗛挓鍚庢敹鐩婁笉佳") is False
