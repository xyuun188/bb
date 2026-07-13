from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from db.repositories.memory_repo import _memory_row_usable
from services.memory_feedback import MemoryFeedbackPolicy


def _memory(**overrides):
    payload = {
        "id": 1,
        "expert_name": "momentum_expert",
        "expert_label": "Momentum",
        "symbol": "LAB/USDT",
        "side": "long",
        "memory_type": "shadow_missed_opportunity",
        "market_pattern": "LAB/USDT 做多影子复盘 10分钟，ADX=31.2，量比=1.80，5周期收益=0.42%",
        "lesson": "LAB/USDT 做多机会曾被观望错过。当方向结构与短周期动量同向，且预期净收益覆盖成本时，应进入小仓质量试单。",
        "recommended_action": "increase_confidence",
        "confidence_adjustment": 0.06,
        "position_size_multiplier": 1.0,
        "evidence_count": 3,
        "hit_count": 0,
        "success_count": 3,
        "failure_count": 0,
        "confidence_score": 0.62,
        "memory_key": "momentum|shadow|LAB/USDT|long",
        "source_position_id": None,
        "last_used_at": None,
        "is_active": True,
        "extra": {},
        "created_at": datetime(2026, 6, 22),
        "updated_at": datetime(2026, 6, 22),
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_shadow_missed_opportunity_memory_with_clean_chinese_is_usable() -> None:
    assert _memory_row_usable(_memory()) is True


def test_missed_opportunity_memory_feedback_remains_observation_only() -> None:
    feedback = MemoryFeedbackPolicy().build(
        [
            {
                "side": "long",
                "memory_type": "shadow_missed_opportunity",
                "confidence_adjustment": 0.06,
                "confidence_score": 0.62,
                "evidence_count": 12,
                "lesson": "多次观望后出现做多收益。",
            },
        ]
    )

    long_side = feedback["by_side"]["long"]
    habit = feedback["decision_habit"]["by_side"]["long"]
    assert long_side["candidate_score_bonus"] == 0.0
    assert long_side["canonical_outcome_count"] == 0
    assert habit["stance"] == "fee_after_observation_only"
    assert habit["proactive_level"] == 0.0
    assert habit["score_adjustment"] == 0.0
