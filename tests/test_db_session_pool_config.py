from __future__ import annotations

import pytest

import db.session as session_module
from config.settings import settings


@pytest.mark.asyncio
async def test_postgres_engine_uses_configured_pool_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    previous_engine = session_module._engine
    previous_sessionmaker = session_module._sessionmaker

    def fake_create_async_engine(url: str, **kwargs: object) -> object:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(session_module, "create_async_engine", fake_create_async_engine)
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://example/bb")
    monkeypatch.setattr(settings, "database_pool_size", 16)
    monkeypatch.setattr(settings, "database_max_overflow", 24)
    monkeypatch.setattr(settings, "database_pool_timeout_seconds", 2.0)
    session_module._engine = None
    session_module._sessionmaker = None

    try:
        assert await session_module.get_engine() is not None
    finally:
        session_module._engine = previous_engine
        session_module._sessionmaker = previous_sessionmaker

    assert captured["url"] == "postgresql+asyncpg://example/bb"
    assert captured["kwargs"] == {
        "echo": False,
        "pool_size": 16,
        "max_overflow": 24,
        "pool_timeout": 2.0,
    }


@pytest.mark.asyncio
async def test_sqlite_engine_keeps_sqlite_specific_connection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    previous_engine = session_module._engine
    previous_sessionmaker = session_module._sessionmaker

    def fake_create_async_engine(url: str, **kwargs: object) -> object:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(session_module, "create_async_engine", fake_create_async_engine)
    monkeypatch.setattr(session_module, "_configure_sqlite_engine", lambda _engine: None)
    monkeypatch.setattr(settings, "database_url", "sqlite+aiosqlite:///./data/test.db")
    session_module._engine = None
    session_module._sessionmaker = None

    try:
        assert await session_module.get_engine() is not None
    finally:
        session_module._engine = previous_engine
        session_module._sessionmaker = previous_sessionmaker

    assert captured["kwargs"] == {
        "echo": False,
        "connect_args": {"check_same_thread": False, "timeout": 30.0},
    }


@pytest.mark.asyncio
async def test_postgres_history_status_migration_expands_match_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []

    class FakeConnection:
        async def execute(self, statement) -> None:
            statements.append(str(statement))

    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://example/bb")

    await session_module._ensure_okx_position_history_column_widths(FakeConnection())

    assert statements == [
        "ALTER TABLE okx_position_history ALTER COLUMN match_status TYPE VARCHAR(160)"
    ]
