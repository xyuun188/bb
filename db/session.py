from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config.settings import settings
from models.base import Base

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


async def get_engine():
    global _engine
    if _engine is None:
        is_sqlite = "sqlite" in settings.database_url
        engine_kwargs: dict[str, Any] = {"echo": False}
        if is_sqlite:
            engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30.0}
        else:
            engine_kwargs["pool_size"] = 5
            engine_kwargs["max_overflow"] = 10
        _engine = create_async_engine(settings.database_url, **engine_kwargs)
        if is_sqlite:
            _configure_sqlite_engine(_engine)
    return _engine


def _configure_sqlite_engine(engine: Any) -> None:
    """Apply SQLite pragmas to every pooled connection."""

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout = 30000")
            cursor.execute("PRAGMA journal_mode = WAL")
        finally:
            cursor.close()


async def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        engine = await get_engine()
        _sessionmaker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return _sessionmaker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    maker = await get_sessionmaker()
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_session_ctx() -> AsyncGenerator[AsyncSession, None]:
    maker = await get_sessionmaker()
    async with maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_read_session_ctx() -> AsyncGenerator[AsyncSession, None]:
    """Open a session for read-only dashboard queries without committing."""
    maker = await get_sessionmaker()
    async with maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables. Called at application startup."""
    import models.account  # noqa: F401 - register account tables in metadata
    import models.dashboard_auth  # noqa: F401 - register dashboard auth tables
    import models.decision  # noqa: F401 - register decision tables in metadata
    import models.learning  # noqa: F401 - register learning tables in metadata
    import models.market_data  # noqa: F401 - register market tables in metadata
    import models.news  # noqa: F401 - register news/social tables in metadata
    import models.risk  # noqa: F401 - register risk tables in metadata
    import models.secure_config  # noqa: F401 - register encrypted config tables
    import models.trade  # noqa: F401 - register trade tables in metadata

    engine = await get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if "sqlite" in settings.database_url:
            await conn.execute(text("PRAGMA busy_timeout = 30000"))
            await conn.execute(text("PRAGMA journal_mode = WAL"))
            result = await conn.execute(text("PRAGMA table_info(ai_decisions)"))
            columns = {row[1] for row in result.fetchall()}
            if "execution_reason" not in columns:
                await conn.execute(
                    text("ALTER TABLE ai_decisions ADD COLUMN execution_reason TEXT")
                )
            if "analysis_type" not in columns:
                await conn.execute(
                    text("ALTER TABLE ai_decisions ADD COLUMN analysis_type VARCHAR(20)")
                )
            await conn.execute(text("""
                    UPDATE ai_decisions
                    SET execution_reason = CASE
                        WHEN action = 'hold' THEN 'AI 选择观望，未提交订单。'
                        ELSE execution_reason
                    END
                    WHERE was_executed = 0
                      AND (execution_reason IS NULL OR execution_reason = '')
                    """))
            await conn.execute(text("""
                    UPDATE ai_decisions
                    SET analysis_type = CASE
                        WHEN lower(coalesce(json_extract(raw_llm_response, '$.analysis_type'), '')) IN
                             ('position', 'position_review', 'holding', 'holdings')
                             THEN 'position'
                        WHEN lower(coalesce(json_extract(raw_llm_response, '$.analysis_type'), '')) IN
                             ('market', 'market_scan', 'symbol_scan')
                             THEN 'market'
                        WHEN json_type(raw_llm_response, '$.position_review_policy') IS NOT NULL
                             OR json_type(raw_llm_response, '$.position_review') IS NOT NULL
                             OR action IN ('close_long', 'close_short')
                             THEN 'position'
                        ELSE 'market'
                    END
                    WHERE model_name = 'ensemble_trader'
                      AND (analysis_type IS NULL OR analysis_type = '')
                    """))
            result = await conn.execute(text("PRAGMA table_info(trade_reflections)"))
            reflection_columns = {row[1] for row in result.fetchall()}
            if "closed_at" not in reflection_columns:
                await conn.execute(
                    text("ALTER TABLE trade_reflections ADD COLUMN closed_at DATETIME")
                )
            await conn.execute(text("""
                    UPDATE trade_reflections
                    SET closed_at = (
                        SELECT positions.closed_at
                        FROM positions
                        WHERE positions.id = trade_reflections.position_id
                    )
                    WHERE closed_at IS NULL
                      AND position_id IS NOT NULL
                    """))
            for ddl in [
                "CREATE INDEX IF NOT EXISTS idx_ai_decisions_mode_created ON ai_decisions (is_paper, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_ai_decisions_model_mode_created ON ai_decisions (model_name, is_paper, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_ai_decisions_executed_mode_created ON ai_decisions (was_executed, is_paper, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_ai_decisions_analysis_mode_created ON ai_decisions (analysis_type, is_paper, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_orders_mode_created ON orders (execution_mode, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_orders_decision_id_created ON orders (decision_id, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_positions_mode_open_created ON positions (execution_mode, is_open, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_positions_mode_closed_created ON positions (execution_mode, is_open, closed_at DESC, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_positions_mode_symbol_side ON positions (execution_mode, symbol, side)",
                "CREATE INDEX IF NOT EXISTS idx_trade_reflections_closed_created ON trade_reflections (closed_at DESC, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_shadow_backtests_model_mode_created ON shadow_backtests (model_name, execution_mode, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_events_mode_created ON strategy_learning_events (execution_mode, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_events_profile_created ON strategy_learning_events (profile_id, execution_mode, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_events_type_status ON strategy_learning_events (event_type, event_status, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_events_symbol_action ON strategy_learning_events (symbol, action, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_profile_snapshots_profile ON strategy_profile_snapshots (execution_mode, profile_id, version)",
                "CREATE INDEX IF NOT EXISTS idx_strategy_profile_snapshots_active ON strategy_profile_snapshots (execution_mode, is_active, created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_dashboard_users_username ON dashboard_users (username)",
                "CREATE INDEX IF NOT EXISTS idx_dashboard_users_email ON dashboard_users (email)",
                "CREATE INDEX IF NOT EXISTS idx_secure_settings_key ON secure_settings (key)",
                "CREATE INDEX IF NOT EXISTS idx_secure_setting_audit_key_created ON secure_setting_audit (key, created_at DESC)",
            ]:
                await conn.execute(text(ddl))


async def close_db() -> None:
    """Dispose engine. Called at application shutdown."""
    global _engine, _sessionmaker
    if _engine:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
