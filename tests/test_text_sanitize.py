from __future__ import annotations

from web_dashboard.api import text_sanitize
from web_dashboard.api.text_sanitize import sanitize_payload, sanitize_text


def test_sanitize_text_removes_json_breaking_control_characters() -> None:
    assert sanitize_text("开仓\x00原因\x1f正常") == "开仓 原因 正常"


def test_sanitize_payload_removes_nested_control_characters() -> None:
    payload = {"reason": "分析\x00详情", "items": ["步骤\x07一"]}

    assert sanitize_payload(payload) == {"reason": "分析 详情", "items": ["步骤 一"]}


def test_unicode_escape_patterns_are_cached_without_changing_output() -> None:
    text_sanitize._u.cache_clear()
    try:
        first = sanitize_text("普通文本保持不变")
        first_cache = text_sanitize._u.cache_info()
        second = sanitize_text("另一条普通文本")
        second_cache = text_sanitize._u.cache_info()

        assert first == "普通文本保持不变"
        assert second == "另一条普通文本"
        assert first_cache.misses > 0
        assert second_cache.misses == first_cache.misses
        assert second_cache.hits > first_cache.hits
    finally:
        text_sanitize._u.cache_clear()
