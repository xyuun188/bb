from __future__ import annotations

from web_dashboard.api.text_sanitize import sanitize_payload, sanitize_text


def test_sanitize_text_removes_json_breaking_control_characters() -> None:
    assert sanitize_text("开仓\x00原因\x1f正常") == "开仓 原因 正常"


def test_sanitize_payload_removes_nested_control_characters() -> None:
    payload = {"reason": "分析\x00详情", "items": ["步骤\x07一"]}

    assert sanitize_payload(payload) == {"reason": "分析 详情", "items": ["步骤 一"]}
