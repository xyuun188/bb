from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from core.logging_config import (
    RedactingFormatter,
    RedactingLogFilter,
    _add_or_update_file_handler,
    _ensure_redacting_filter,
    _ensure_redacting_formatter,
    redact_structlog_event,
)


def test_redact_structlog_event_masks_sensitive_fields_and_event_text() -> None:
    fake_bearer = "Bearer " + "abcdefghijklmnopqrstuvwxyz" + "123456"
    event = {
        "event": "request failed with password=plain-secret",
        "api_key": "direct-secret-value",
        "nested": {"token": "nested-token-value"},
        "messages": [
            "webhook=https://example.invalid/private",
            {"authorization": fake_bearer},
        ],
        "symbol": "BTC/USDT",
    }

    redacted = redact_structlog_event(None, "error", event)

    assert "plain-secret" not in str(redacted)
    assert "direct-secret-value" not in str(redacted)
    assert "nested-token-value" not in str(redacted)
    assert "example.invalid/private" not in str(redacted)
    assert "abcdefghijklmnopqrstuvwxyz" not in str(redacted)
    assert redacted["event"] == "request failed with password=***"
    assert redacted["api_key"] == "***"
    assert redacted["nested"]["token"] == "***"
    assert redacted["messages"][0] == "webhook=***"
    assert redacted["messages"][1]["authorization"] == "***"
    assert redacted["symbol"] == "BTC/USDT"


def test_redact_structlog_event_masks_tuple_and_set_context_values() -> None:
    event = {
        "event": "remote command failed",
        "context": (
            "password=tuple-secret-value",
            {"token": "tuple-token-value"},
        ),
        "tags": {"webhook=https://example.invalid/private"},
    }

    redacted = redact_structlog_event(None, "warning", event)
    rendered = str(redacted)

    assert "tuple-secret-value" not in rendered
    assert "tuple-token-value" not in rendered
    assert "example.invalid/private" not in rendered
    assert redacted["context"][0] == "password=***"
    assert redacted["context"][1]["token"] == "***"
    assert redacted["tags"] == {"webhook=***"}


def test_file_handler_setup_is_idempotent_for_same_log_path(tmp_path) -> None:
    logger = logging.getLogger("test_logging_config_idempotent")
    logger.handlers.clear()
    log_path = tmp_path / "trading.log"

    _add_or_update_file_handler(logger, log_path, logging.INFO)
    _add_or_update_file_handler(logger, log_path, logging.WARNING)

    handlers = [handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)]
    assert len(handlers) == 1
    assert handlers[0].level == logging.WARNING
    assert len([item for item in handlers[0].filters if isinstance(item, RedactingLogFilter)]) == 1
    assert isinstance(handlers[0].formatter, RedactingFormatter)


def test_file_handler_setup_updates_rotation_policy(
    tmp_path,
    monkeypatch,
) -> None:
    logger = logging.getLogger("test_logging_config_rotation_policy")
    logger.handlers.clear()
    log_path = tmp_path / "trading.log"

    monkeypatch.setattr("core.logging_config.settings.log_max_bytes", 2 * 1024 * 1024)
    monkeypatch.setattr("core.logging_config.settings.log_backup_count", 2)
    _add_or_update_file_handler(logger, log_path, logging.INFO)

    monkeypatch.setattr("core.logging_config.settings.log_max_bytes", 3 * 1024 * 1024)
    monkeypatch.setattr("core.logging_config.settings.log_backup_count", 4)
    _add_or_update_file_handler(logger, log_path, logging.INFO)

    handlers = [handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)]
    assert len(handlers) == 1
    assert handlers[0].maxBytes == 3 * 1024 * 1024
    assert handlers[0].backupCount == 4


def test_stdlib_log_filter_redacts_message_and_positional_args() -> None:
    record = logging.LogRecord(
        name="unit",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="failed request password=%s token=%s",
        args=("plain-value", "token-value"),
        exc_info=None,
    )

    RedactingLogFilter().filter(record)
    rendered = record.getMessage()

    assert "plain-value" not in rendered
    assert "token-value" not in rendered
    assert rendered == "failed request password=*** token=***"


def test_stdlib_log_filter_redacts_mapping_args() -> None:
    record = logging.LogRecord(
        name="unit",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="auth failed for %(symbol)s with %(api_key)s",
        args={"symbol": "ETH/USDT", "api_key": "mapping-value"},
        exc_info=None,
    )

    RedactingLogFilter().filter(record)
    rendered = record.getMessage()

    assert "mapping-value" not in rendered
    assert rendered == "auth failed for ETH/USDT with ***"


def test_redacting_filter_installation_is_idempotent_on_logger() -> None:
    logger = logging.getLogger("test_logging_config_filter_idempotent")
    logger.filters.clear()

    _ensure_redacting_filter(logger)
    _ensure_redacting_filter(logger)

    assert len([item for item in logger.filters if isinstance(item, RedactingLogFilter)]) == 1


def test_redacting_formatter_masks_exception_traceback_text() -> None:
    try:
        raise RuntimeError("remote failed with password=exception-secret")
    except RuntimeError as exc:
        record = logging.LogRecord(
            name="unit",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="operation failed",
            args=(),
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    rendered = RedactingFormatter("%(message)s").format(record)

    assert "exception-secret" not in rendered
    assert "RuntimeError: remote failed with password=***" in rendered


def test_redacting_formatter_installation_preserves_existing_format() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    _ensure_redacting_formatter(handler)
    record = logging.LogRecord(
        name="unit",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="token=formatter-secret",
        args=(),
        exc_info=None,
    )

    assert isinstance(handler.formatter, RedactingFormatter)
    assert handler.formatter.format(record) == "[ERROR] token=***"
