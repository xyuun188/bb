from __future__ import annotations

import pytest

from core.url_safety import (
    normalize_external_http_url,
    normalize_http_base_url,
    normalize_https_webhook_url,
)


def test_normalize_http_base_url_accepts_absolute_http_urls() -> None:
    assert (
        normalize_http_base_url(
            " https://model.example.invalid/v1/ ",
            field_name="AI model API base",
        )
        == "https://model.example.invalid/v1"
    )


def test_normalize_http_base_url_allows_empty_when_requested() -> None:
    assert normalize_http_base_url("", allow_empty=True) == ""


@pytest.mark.parametrize(
    "value, message",
    [
        ("127.0.0.1:8000", "absolute http"),
        ("ftp://example.invalid", "absolute http"),
        ("http://user:password@example.invalid", "must not include credentials"),
        ("https://@example.invalid/v1", "must not include credentials"),
        ("https://example.invalid/v1?token=abc", "query strings"),
        ("https://example.invalid/v1#fragment", "fragments"),
        ("https://example.invalid\\v1", "backslashes"),
        ("https://exa mple.invalid/v1", "whitespace"),
        ("https://example.invalid/v 1", "whitespace"),
        ("https://example.invalid/v1\nX=1", "control characters"),
        ("https://example.invalid/%0aX", "encoded control"),
        ("https://example.invalid:99999/v1", "port"),
        ("https://example.invalid:/v1", "port"),
        ("https://example.invalid:0/v1", "port"),
    ],
)
def test_normalize_http_base_url_rejects_unsafe_values(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_http_base_url(value, field_name="AI model API base")


def test_normalize_external_http_url_allows_article_queries_and_fragments() -> None:
    assert (
        normalize_external_http_url(
            " https://news.example.invalid/article?id=1#quote ",
            field_name="news source URL",
        )
        == "https://news.example.invalid/article?id=1#quote"
    )
    assert normalize_external_http_url("", field_name="news source URL") == ""


@pytest.mark.parametrize(
    "value, message",
    [
        ("javascript:alert(1)", "absolute http"),
        ("data:text/html,<script>alert(1)</script>", "absolute http"),
        ("//example.invalid/article", "absolute http"),
        ("http://user:password@example.invalid/article", "credentials"),
        ("https://example.invalid\\article", "backslashes"),
        ("https://news.example.invalid/a b", "whitespace"),
        ("https://example.invalid/article\nonclick=1", "control characters"),
        ("https://example.invalid/%0dheader", "encoded control"),
        ("https://example.invalid:99999/article", "port"),
    ],
)
def test_normalize_external_http_url_rejects_unsafe_links(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_external_http_url(value, field_name="news source URL")


def test_normalize_https_webhook_url_allows_query_token() -> None:
    assert (
        normalize_https_webhook_url(
            " https://oapi.dingtalk.com/robot/send?access_token=secret-value ",
            field_name="DingTalk webhook URL",
        )
        == "https://oapi.dingtalk.com/robot/send?access_token=secret-value"
    )


@pytest.mark.parametrize(
    "value, message",
    [
        ("http://oapi.dingtalk.com/robot/send?access_token=x", "must use https"),
        ("https://oapi.dingtalk.com/robot/send#token", "fragments"),
        ("https://user:pass@oapi.dingtalk.com/robot/send", "credentials"),
        ("https://oapi.dingtalk.com:99999/robot/send?access_token=x", "port"),
        ("https://oapi.dingtalk.com/robot/send?access_token=x y", "whitespace"),
    ],
)
def test_normalize_https_webhook_url_rejects_unsafe_values(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_https_webhook_url(value, field_name="DingTalk webhook URL")
