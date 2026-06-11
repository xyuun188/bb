"""
Structured logging configuration using structlog.
All modules use this for consistent, machine-parseable log output.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, MutableMapping
from contextlib import suppress
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

import structlog

from config.settings import settings
from core.secret_utils import redact_mapping, redact_text


class RedactingLogFilter(logging.Filter):
    """Redact secret-looking content in stdlib logging messages and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, Mapping):
            record.args = redact_mapping(record.args)
        try:
            rendered = record.getMessage()
        except Exception:
            record.msg = _redact_log_value(record.msg)
            if record.args:
                record.args = _redact_log_args(record.args)
            return True

        record.msg = redact_text(rendered)
        record.args = ()
        return True


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts the final rendered log line, including tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))


def _redact_log_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [_redact_log_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_log_value(item) for item in value)
    return value


def _redact_log_args(args: Any) -> Any:
    if isinstance(args, Mapping):
        return redact_mapping(args)
    if isinstance(args, tuple):
        return tuple(_redact_log_value(item) for item in args)
    return _redact_log_value(args)


def _ensure_redacting_filter(target: Any) -> None:
    filters = getattr(target, "filters", [])
    if any(isinstance(existing, RedactingLogFilter) for existing in filters):
        return
    target.addFilter(RedactingLogFilter())


def _ensure_redacting_formatter(handler: logging.Handler) -> None:
    if isinstance(handler.formatter, RedactingFormatter):
        return
    fmt = "%(message)s"
    datefmt = None
    style: Literal["%", "{", "$"] = "%"
    if handler.formatter is not None:
        style_obj = getattr(handler.formatter, "_style", None)
        fmt = getattr(style_obj, "_fmt", fmt)
        datefmt = getattr(handler.formatter, "datefmt", None)
        if isinstance(style_obj, logging.StrFormatStyle):
            style = "{"
        elif isinstance(style_obj, logging.StringTemplateStyle):
            style = "$"
    handler.setFormatter(RedactingFormatter(fmt, datefmt=datefmt, style=style))


def redact_structlog_event(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> Mapping[str, Any]:
    """Redact secret-looking fields before structured logs are rendered."""
    redacted = redact_mapping(event_dict)
    event = redacted.get("event")
    if isinstance(event, str):
        redacted["event"] = redact_text(event)
    return redacted


def _add_or_update_file_handler(
    root_logger: logging.Logger,
    log_path: Path,
    log_level: int,
) -> None:
    """Install a bounded file handler without duplicating it on repeated setup calls."""
    resolved = str(log_path.resolve())
    max_bytes = max(int(settings.log_max_bytes or 0), 1024 * 1024)
    backup_count = max(int(settings.log_backup_count or 0), 0)
    for handler in root_logger.handlers:
        if (
            isinstance(handler, RotatingFileHandler)
            and getattr(handler, "baseFilename", None) == resolved
        ):
            handler.setLevel(log_level)
            handler.maxBytes = max_bytes
            handler.backupCount = backup_count
            handler.setFormatter(RedactingFormatter("%(message)s"))
            _ensure_redacting_filter(handler)
            return

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(RedactingFormatter("%(message)s"))
    _ensure_redacting_filter(file_handler)
    root_logger.addHandler(file_handler)


def setup_logging() -> None:
    """Configure structlog with console + file output.

    Called once at application startup. After this, modules import
    `structlog.get_logger(__name__)` and log as usual.
    """

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")

    # Stdlib logging bridge
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # File handler for persistent logs. Keep it bounded so trading.log cannot
    # grow without limit during long-running market scans.
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    _ensure_redacting_filter(root_logger)
    _add_or_update_file_handler(root_logger, log_path, log_level)
    for handler in root_logger.handlers:
        _ensure_redacting_filter(handler)
        _ensure_redacting_formatter(handler)
    root_logger.setLevel(log_level)

    # Quiet noisy third-party loggers
    for noisy in (
        "ccxt",
        "urllib3",
        "websockets",
        "httpx",
        "httpcore",
        "openai",
        "langchain_openai",
        "langchain_core",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            redact_structlog_event,
            (
                structlog.dev.ConsoleRenderer()
                if sys.stdout.isatty()
                else structlog.processors.JSONRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Convenience: get a structlog logger for the given module name."""
    return structlog.get_logger(name or __name__)
