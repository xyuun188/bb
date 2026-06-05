"""
Structured logging configuration using structlog.
All modules use this for consistent, machine-parseable log output.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from config.settings import settings


def setup_logging() -> None:
    """Configure structlog with console + file output.

    Called once at application startup. After this, modules import
    `structlog.get_logger(__name__)` and log as usual.
    """

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

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
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max(int(settings.log_max_bytes or 0), 1024 * 1024),
        backupCount=max(int(settings.log_backup_count or 0), 0),
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

    # Quiet noisy third-party loggers
    for noisy in ("ccxt", "urllib3", "websockets", "httpx", "httpcore", "openai", "langchain_openai", "langchain_core"):
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
            structlog.dev.ConsoleRenderer()
            if sys.stdout.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Convenience: get a structlog logger for the given module name."""
    return structlog.get_logger(name or __name__)
