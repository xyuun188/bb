from __future__ import annotations

from services.text_integrity import (
    looks_like_mojibake,
    repair_mojibake,
    sanitize_runtime_text,
)


def test_detects_common_mojibake_without_flagging_normal_text() -> None:
    assert looks_like_mojibake("鏈轰細璇勫垎") is True
    assert looks_like_mojibake("机会评分为正") is False
    assert looks_like_mojibake("BTC/USDT 123.45") is False


def test_repair_mojibake_returns_safe_result_object() -> None:
    result = repair_mojibake("鏈轰細璇勫垎")

    assert result.original == "鏈轰細璇勫垎"
    assert result.suspected is True
    assert result.method in {"deterministic_redecode", "known_replacement", "unrepairable"}
    assert result.text != ""
    assert "鏈" not in result.text or result.method == "unrepairable"


def test_repair_mojibake_does_not_mutate_clean_chinese() -> None:
    result = repair_mojibake("机会评分为正")

    assert result.text == "机会评分为正"
    assert result.changed is False
    assert result.suspected is False
    assert result.method == "unchanged"


def test_sanitize_runtime_text_recurses_nested_payloads() -> None:
    payload = {
        "reason": "机会评分为正",
        "items": ["鏈轰細璇勫垎", {"symbol": "BTC/USDT"}],
    }

    sanitized = sanitize_runtime_text(payload)

    assert sanitized["reason"] == "机会评分为正"
    assert sanitized["items"][1]["symbol"] == "BTC/USDT"
    assert isinstance(sanitized["items"][0], str)
    assert sanitized["items"][0] != ""
