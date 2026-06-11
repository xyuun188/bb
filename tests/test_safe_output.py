from __future__ import annotations

from io import StringIO

import httpx

from core.safe_output import (
    format_command_failure,
    redact_output,
    safe_error_text,
    safe_print,
    safe_response_error_text,
)


def test_safe_print_redacts_secret_values_before_terminal_output() -> None:
    stream = StringIO()
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"

    safe_print(
        "api_key=live-secret-value",
        f"Authorization: Bearer {token}",
        file=stream,
    )

    output = stream.getvalue()
    assert "live-secret-value" not in output
    assert "abcdefghijklmnopqrstuvwxyz" not in output
    assert "api_key=***" in output
    assert "Authorization: ***" in output


def test_format_command_failure_redacts_command_stdout_and_stderr() -> None:
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"

    message = format_command_failure(
        1,
        f"curl -H 'Authorization: Bearer {token}'",
        stdout=b'{"token": "stdout-secret-value"}',
        stderr="password=stderr-secret-value",
    )

    assert "abcdefghijklmnopqrstuvwxyz" not in message
    assert "stdout-secret-value" not in message
    assert "stderr-secret-value" not in message
    assert "Authorization: ***" in message
    assert '"token": "***"' in message
    assert "password=***" in message


def test_format_command_failure_caps_command_and_stream_output() -> None:
    message = format_command_failure(
        2,
        "python " + ("x" * 80),
        stdout="out-" + ("a" * 80),
        stderr="err-" + ("b" * 80),
        command_limit=24,
        stream_limit=16,
    )

    assert "python " + ("x" * 17) + "..." in message
    assert "out-" + ("a" * 12) + "..." in message
    assert "err-" + ("b" * 12) + "..." in message
    assert "a" * 40 not in message
    assert "b" * 40 not in message


def test_redact_output_preserves_plain_zero_and_empty_none() -> None:
    assert redact_output(0) == "0"
    assert redact_output(None) == ""


def test_safe_error_text_redacts_truncates_and_uses_fallback() -> None:
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"

    message = safe_error_text(f"Authorization: Bearer {token} " + ("x" * 120), limit=40)

    assert "abcdefghijklmnopqrstuvwxyz" not in message
    assert "Authorization: ***" in message
    assert len(message) == 43
    assert message.endswith("...")
    assert safe_error_text("", fallback="fallback-error") == "fallback-error"


def test_safe_response_error_text_handles_json_response_body() -> None:
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    response = httpx.Response(
        401,
        json={"detail": f"Authorization: Bearer {token} is invalid"},
        request=httpx.Request("POST", "https://example.invalid"),
    )

    message = safe_response_error_text(response)

    assert "abcdefghijklmnopqrstuvwxyz" not in message
    assert "Authorization: ***" in message
    assert '"detail"' in message


def test_safe_response_error_text_handles_text_response_body_with_limit() -> None:
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    response = httpx.Response(
        500,
        text=f"password=plain-secret-value Authorization: Bearer {token} " + ("x" * 120),
        request=httpx.Request("POST", "https://example.invalid"),
    )

    message = safe_response_error_text(response, limit=50)

    assert "plain-secret-value" not in message
    assert "abcdefghijklmnopqrstuvwxyz" not in message
    assert "password=***" in message
    assert "Authorization: ***" in message
    assert len(message) == 53
    assert message.endswith("...")


def test_redact_output_handles_webhook_and_telegram_token_shapes() -> None:
    telegram_token = "123456:" + ("A" * 28)
    dingtalk_token = "dingtalk-secret-value"

    output = redact_output(
        "https://api.telegram.org/bot" f"{telegram_token}/sendMessage?access_token={dingtalk_token}"
    )

    assert telegram_token not in output
    assert dingtalk_token not in output
    assert "api.telegram.org/bot***" in output
    assert "access_token=***" in output
