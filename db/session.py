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
            # Market, position, training, and reconciliation work concurrently.
            # The old implicit 5+10 pool could queue a simple open-position read
            # for SQLAlchemy's default 30 seconds during a busy round.
            engine_kwargs["pool_size"] = max(int(settings.database_pool_size or 0), 1)
            engine_kwargs["max_overflow"] = max(int(settings.database_max_overflow or 0), 0)
            engine_kwargs["pool_timeout"] = max(
                float(settings.database_pool_timeout_seconds or 0.0),
                0.5,
            )
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
        await _lock_schema_migration_if_needed(conn)
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
        await _ensure_trade_fact_columns(conn)
        await _drop_removed_expert_memory_policy_columns(conn)
        await _ensure_ai_decision_model_health_columns(conn)
        await _ensure_shadow_backtest_training_snapshot_columns(conn)
        await _ensure_trade_fact_indexes(conn)
        await _ensure_okx_account_bill_indexes(conn)
        await _ensure_okx_position_history_column_widths(conn)
        await _ensure_okx_position_history_indexes(conn)
        if "sqlite" in settings.database_url:
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


async def _lock_schema_migration_if_needed(conn: Any) -> None:
    """Serialize PostgreSQL startup DDL across split services.

    Paper trading, dashboard, and helper scripts can start at nearly the same
    time after deployment.  Each process calls init_db(), and PostgreSQL can
    deadlock when concurrent CREATE/ALTER/CREATE INDEX statements touch the same
    tables in different orders.  A transaction-level advisory lock keeps startup
    idempotent while preserving the normal SQLite path used by tests/local runs.
    """

    if "postgresql" not in settings.database_url:
        return
    await conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('bb:init_db_schema'))"))


async def _drop_removed_expert_memory_policy_columns(conn: Any) -> None:
    """Permanently remove expert-memory fields that once changed live policy."""

    if "postgresql" in settings.database_url:
        for ddl in (
            "ALTER TABLE expert_memories DROP COLUMN IF EXISTS confidence_adjustment",
            "ALTER TABLE expert_memories DROP COLUMN IF EXISTS position_size_multiplier",
        ):
            await conn.execute(text(ddl))
        return

    if "sqlite" not in settings.database_url:
        return
    result = await conn.execute(text("PRAGMA table_info(expert_memories)"))
    existing = {str(row[1]) for row in result.fetchall()}
    for name in ("confidence_adjustment", "position_size_multiplier"):
        if name in existing:
            await conn.execute(text(f"ALTER TABLE expert_memories DROP COLUMN {name}"))


async def _ensure_trade_fact_columns(conn: Any) -> None:
    """Backfill lightweight trade fact linkage columns for existing databases."""

    if "sqlite" in settings.database_url:
        result = await conn.execute(text("PRAGMA table_info(positions)"))
        position_columns = {row[1] for row in result.fetchall()}
        sqlite_position_columns = {
            "okx_inst_id": "VARCHAR(64)",
            "okx_pos_id": "VARCHAR(100)",
            "entry_exchange_order_id": "VARCHAR(500)",
            "close_exchange_order_id": "VARCHAR(500)",
            "close_fill_pnl": "FLOAT",
            "entry_fee": "FLOAT",
            "close_fee": "FLOAT",
            "funding_fee": "FLOAT",
            "settlement_status": "VARCHAR(40)",
            "settlement_source": "VARCHAR(80)",
            "settlement_synced_at": "DATETIME",
            "settlement_raw": "JSON",
            "current_management_contract": "JSON",
        }
        for name, column_type in sqlite_position_columns.items():
            if name not in position_columns:
                await conn.execute(text(f"ALTER TABLE positions ADD COLUMN {name} {column_type}"))
        result = await conn.execute(text("PRAGMA table_info(orders)"))
        order_columns = {row[1] for row in result.fetchall()}
        sqlite_order_columns = {
            "okx_inst_id": "VARCHAR(64)",
            "okx_trade_ids": "VARCHAR(500)",
            "okx_fill_contracts": "FLOAT",
            "okx_fill_pnl": "FLOAT",
            "okx_state": "VARCHAR(40)",
            "okx_sync_status": "VARCHAR(40)",
            "okx_synced_at": "DATETIME",
            "okx_last_error": "VARCHAR(500)",
            "okx_raw_fills": "JSON",
        }
        for name, column_type in sqlite_order_columns.items():
            if name not in order_columns:
                await conn.execute(text(f"ALTER TABLE orders ADD COLUMN {name} {column_type}"))
        return

    if "postgresql" in settings.database_url:
        position_columns = await _postgres_table_columns(conn, "positions")
        postgres_position_columns = {
            "okx_inst_id": "ALTER TABLE positions ADD COLUMN okx_inst_id VARCHAR(64)",
            "okx_pos_id": "ALTER TABLE positions ADD COLUMN okx_pos_id VARCHAR(100)",
            "entry_exchange_order_id": (
                "ALTER TABLE positions ADD COLUMN entry_exchange_order_id VARCHAR(500)"
            ),
            "close_exchange_order_id": (
                "ALTER TABLE positions ADD COLUMN close_exchange_order_id VARCHAR(500)"
            ),
            "close_fill_pnl": "ALTER TABLE positions ADD COLUMN close_fill_pnl DOUBLE PRECISION",
            "entry_fee": "ALTER TABLE positions ADD COLUMN entry_fee DOUBLE PRECISION",
            "close_fee": "ALTER TABLE positions ADD COLUMN close_fee DOUBLE PRECISION",
            "funding_fee": "ALTER TABLE positions ADD COLUMN funding_fee DOUBLE PRECISION",
            "settlement_status": "ALTER TABLE positions ADD COLUMN settlement_status VARCHAR(40)",
            "settlement_source": "ALTER TABLE positions ADD COLUMN settlement_source VARCHAR(80)",
            "settlement_synced_at": (
                "ALTER TABLE positions ADD COLUMN settlement_synced_at TIMESTAMP WITH TIME ZONE"
            ),
            "settlement_raw": "ALTER TABLE positions ADD COLUMN settlement_raw JSONB",
            "current_management_contract": (
                "ALTER TABLE positions ADD COLUMN current_management_contract JSONB"
            ),
        }
        for name, ddl in postgres_position_columns.items():
            if name not in position_columns:
                await conn.execute(text(ddl))
        if "entry_exchange_order_id" in position_columns:
            await conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1
                            FROM information_schema.columns
                            WHERE table_schema = current_schema()
                              AND table_name = 'positions'
                              AND column_name = 'entry_exchange_order_id'
                              AND (
                                  character_maximum_length IS NULL
                                  OR character_maximum_length < 500
                              )
                        ) THEN
                            ALTER TABLE positions
                            ALTER COLUMN entry_exchange_order_id TYPE VARCHAR(500);
                        END IF;
                    END $$;
                    """
                )
            )

        order_columns = await _postgres_table_columns(conn, "orders")
        postgres_order_columns = {
            "okx_inst_id": "ALTER TABLE orders ADD COLUMN okx_inst_id VARCHAR(64)",
            "okx_trade_ids": "ALTER TABLE orders ADD COLUMN okx_trade_ids VARCHAR(500)",
            "okx_fill_contracts": (
                "ALTER TABLE orders ADD COLUMN okx_fill_contracts DOUBLE PRECISION"
            ),
            "okx_fill_pnl": "ALTER TABLE orders ADD COLUMN okx_fill_pnl DOUBLE PRECISION",
            "okx_state": "ALTER TABLE orders ADD COLUMN okx_state VARCHAR(40)",
            "okx_sync_status": "ALTER TABLE orders ADD COLUMN okx_sync_status VARCHAR(40)",
            "okx_synced_at": (
                "ALTER TABLE orders ADD COLUMN okx_synced_at TIMESTAMP WITH TIME ZONE"
            ),
            "okx_last_error": "ALTER TABLE orders ADD COLUMN okx_last_error VARCHAR(500)",
            "okx_raw_fills": "ALTER TABLE orders ADD COLUMN okx_raw_fills JSONB",
        }
        for name, ddl in postgres_order_columns.items():
            if name not in order_columns:
                await conn.execute(text(ddl))


async def _ensure_trade_fact_indexes(conn: Any) -> None:
    index_ddls = (
        (
            "idx_positions_okx_inst_side",
            "CREATE INDEX IF NOT EXISTS idx_positions_okx_inst_side ON positions (okx_inst_id, side)",
        ),
        (
            "idx_positions_entry_exchange_order",
            "CREATE INDEX IF NOT EXISTS idx_positions_entry_exchange_order ON positions (entry_exchange_order_id)",
        ),
        (
            "idx_positions_close_exchange_order",
            "CREATE INDEX IF NOT EXISTS idx_positions_close_exchange_order ON positions (close_exchange_order_id)",
        ),
        (
            "idx_positions_closed_scan",
            "CREATE INDEX IF NOT EXISTS idx_positions_closed_scan ON positions (is_open, closed_at DESC, id DESC)",
        ),
        (
            "idx_positions_created_scan",
            "CREATE INDEX IF NOT EXISTS idx_positions_created_scan ON positions (created_at DESC, id DESC)",
        ),
        (
            "idx_positions_open_created_scan",
            "CREATE INDEX IF NOT EXISTS idx_positions_open_created_scan ON positions (is_open, created_at DESC, id DESC)",
        ),
        (
            "idx_orders_filled_exchange_scan",
            "CREATE INDEX IF NOT EXISTS idx_orders_filled_exchange_scan ON orders (status, filled_at DESC, id DESC)",
        ),
        (
            "idx_orders_decision_side_scan",
            "CREATE INDEX IF NOT EXISTS idx_orders_decision_side_scan ON orders (decision_id, side, status, filled_at DESC)",
        ),
        (
            "idx_orders_exchange_order_id",
            "CREATE INDEX IF NOT EXISTS idx_orders_exchange_order_id ON orders (exchange_order_id)",
        ),
        (
            "idx_orders_okx_inst_id",
            "CREATE INDEX IF NOT EXISTS idx_orders_okx_inst_id ON orders (okx_inst_id)",
        ),
        (
            "idx_orders_okx_sync_status",
            "CREATE INDEX IF NOT EXISTS idx_orders_okx_sync_status ON orders (okx_sync_status, okx_synced_at DESC)",
        ),
        (
            "idx_ai_decisions_pending_entry_recent",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_pending_entry_recent "
            "ON ai_decisions (was_executed, action, created_at DESC, id DESC)",
        ),
        (
            "idx_ai_decisions_recent_scan",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_recent_scan "
            "ON ai_decisions (created_at DESC, id DESC)",
        ),
        (
            "idx_ai_decisions_strategy_learning_recent",
            "CREATE INDEX IF NOT EXISTS idx_ai_decisions_strategy_learning_recent "
            "ON ai_decisions (model_name, is_paper, created_at DESC, id DESC)",
        ),
        (
            "idx_shadow_backtests_training_completed",
            "CREATE INDEX IF NOT EXISTS idx_shadow_backtests_training_completed "
            "ON shadow_backtests (created_at DESC, id DESC) "
            "WHERE status = 'completed' "
            "AND long_return_pct IS NOT NULL AND short_return_pct IS NOT NULL",
        ),
    )
    if "postgresql" in settings.database_url:
        index_names = await _postgres_index_names(conn)
        for name, ddl in index_ddls:
            if name not in index_names:
                await conn.execute(text(ddl))
        return
    for _name, ddl in index_ddls:
        await conn.execute(text(ddl))


async def _ensure_ai_decision_model_health_columns(conn: Any) -> None:
    """Persist compact health evidence alongside each large decision payload."""

    column_ddls = (
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS model_health_timings JSONB",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS model_health_fallback_timings JSONB",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS model_health_experts JSONB",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS model_health_opinions JSONB",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS model_health_has_ml_signal BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS model_health_has_local_ml_signal BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS model_health_has_local_ai_tools BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS model_health_snapshot_version INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS decision_learning_snapshot JSONB",
        "ALTER TABLE ai_decisions ADD COLUMN IF NOT EXISTS decision_learning_snapshot_version INTEGER NOT NULL DEFAULT 0",
    )
    if "postgresql" in settings.database_url:
        for ddl in column_ddls:
            await conn.execute(text(ddl))
        await conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION bb_sync_ai_decision_model_health()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                DECLARE
                    raw JSONB := COALESCE(NEW.raw_llm_response::JSONB, '{}'::JSONB);
                BEGIN
                    NEW.model_health_timings := CASE
                        WHEN jsonb_typeof(raw -> 'model_timings') = 'array'
                        THEN raw -> 'model_timings' ELSE NULL END;
                    NEW.model_health_fallback_timings := CASE
                        WHEN jsonb_typeof(raw -> '_model_timings') = 'array'
                        THEN raw -> '_model_timings' ELSE NULL END;
                    NEW.model_health_experts := CASE
                        WHEN jsonb_typeof(raw -> 'experts') = 'array'
                        THEN raw -> 'experts' ELSE NULL END;
                    NEW.model_health_opinions := CASE
                        WHEN jsonb_typeof(raw -> 'opinions') = 'array'
                        THEN raw -> 'opinions' ELSE NULL END;
                    NEW.model_health_has_ml_signal := CASE
                        WHEN raw -> 'ml_signal' IS NULL OR raw -> 'ml_signal' IN ('null'::JSONB, '{}'::JSONB, '[]'::JSONB)
                        THEN FALSE ELSE TRUE END;
                    NEW.model_health_has_local_ml_signal := CASE
                        WHEN raw -> 'local_ml_signal' IS NULL OR raw -> 'local_ml_signal' IN ('null'::JSONB, '{}'::JSONB, '[]'::JSONB)
                        THEN FALSE ELSE TRUE END;
                    NEW.model_health_has_local_ai_tools := CASE
                        WHEN raw -> 'local_ai_tools' IS NULL OR raw -> 'local_ai_tools' IN ('null'::JSONB, '{}'::JSONB, '[]'::JSONB)
                        THEN FALSE ELSE TRUE END;
                    NEW.model_health_snapshot_version := 1;
                    NEW.decision_learning_snapshot := COALESCE(
                        (
                            SELECT jsonb_object_agg(item.key, item.value)
                            FROM jsonb_each(raw) AS item(key, value)
                            WHERE jsonb_typeof(item.value) IN ('number', 'boolean', 'null')
                               OR (
                                   jsonb_typeof(item.value) = 'string'
                                   AND length(item.value #>> '{}') <= 1024
                               )
                               OR (
                                   jsonb_typeof(item.value) IN ('object', 'array')
                                   AND pg_column_size(item.value) <= 16384
                               )
                        ),
                        '{}'::JSONB
                    );
                    NEW.decision_learning_snapshot_version := 1;
                    RETURN NEW;
                END;
                $$
                """
            )
        )
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger
                        WHERE tgname = 'trg_ai_decisions_model_health_snapshot'
                          AND tgrelid = 'ai_decisions'::regclass
                    ) THEN
                        CREATE TRIGGER trg_ai_decisions_model_health_snapshot
                        BEFORE INSERT OR UPDATE OF raw_llm_response ON ai_decisions
                        FOR EACH ROW EXECUTE FUNCTION bb_sync_ai_decision_model_health();
                    END IF;
                END;
                $$
                """
            )
        )
        # Backfill only the most recent rows needed by health reports. The trigger
        # handles all future writes without re-reading large historic payloads.
        await conn.execute(
            text(
                """
                WITH recent_unprojected AS (
                    SELECT id
                    FROM ai_decisions
                    WHERE model_health_snapshot_version < 1
                       OR decision_learning_snapshot_version < 1
                    ORDER BY created_at DESC NULLS LAST, id DESC
                    LIMIT 1500
                )
                UPDATE ai_decisions AS decision
                SET raw_llm_response = decision.raw_llm_response
                FROM recent_unprojected
                WHERE decision.id = recent_unprojected.id
                """
            )
        )
        return

    if "sqlite" not in settings.database_url:
        return
    result = await conn.execute(text("PRAGMA table_info(ai_decisions)"))
    existing = {str(row[1]) for row in result.fetchall()}
    sqlite_columns = (
        ("model_health_timings", "JSON"),
        ("model_health_fallback_timings", "JSON"),
        ("model_health_experts", "JSON"),
        ("model_health_opinions", "JSON"),
        ("model_health_has_ml_signal", "BOOLEAN NOT NULL DEFAULT 0"),
        ("model_health_has_local_ml_signal", "BOOLEAN NOT NULL DEFAULT 0"),
        ("model_health_has_local_ai_tools", "BOOLEAN NOT NULL DEFAULT 0"),
        ("model_health_snapshot_version", "INTEGER NOT NULL DEFAULT 0"),
        ("decision_learning_snapshot", "JSON"),
        ("decision_learning_snapshot_version", "INTEGER NOT NULL DEFAULT 0"),
    )
    for name, column_type in sqlite_columns:
        if name not in existing:
            await conn.execute(text(f"ALTER TABLE ai_decisions ADD COLUMN {name} {column_type}"))


async def _ensure_shadow_backtest_training_snapshot_columns(conn: Any) -> None:
    """Store a bounded training feature view beside every large shadow payload."""

    from core.training_contracts import SHADOW_LABEL_VERSION

    if "postgresql" in settings.database_url:
        await conn.execute(
            text(
                "ALTER TABLE shadow_backtests "
                "ADD COLUMN IF NOT EXISTS training_feature_snapshot JSONB"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE shadow_backtests "
                "ADD COLUMN IF NOT EXISTS training_feature_snapshot_version INTEGER NOT NULL DEFAULT 0"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE shadow_backtests "
                "ADD COLUMN IF NOT EXISTS label_version VARCHAR(80)"
            )
        )
        await conn.execute(
            text(
                "UPDATE shadow_backtests "
                "SET label_version = 'legacy-row-' || id::text "
                "WHERE label_version IS NULL OR label_version = ''"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE shadow_backtests "
                f"ALTER COLUMN label_version SET DEFAULT '{SHADOW_LABEL_VERSION}'"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE shadow_backtests "
                "ALTER COLUMN label_version SET NOT NULL"
            )
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_shadow_decision_horizon_label_version "
                "ON shadow_backtests (decision_id, horizon_minutes, label_version) "
                "WHERE decision_id IS NOT NULL"
            )
        )
        await conn.execute(
            text(
                """
                CREATE OR REPLACE FUNCTION bb_sync_shadow_training_feature_snapshot()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                DECLARE
                    raw JSONB := COALESCE(NEW.feature_snapshot::JSONB, '{}'::JSONB);
                BEGIN
                    NEW.training_feature_snapshot := COALESCE(
                        (
                            SELECT jsonb_object_agg(item.key, item.value)
                            FROM jsonb_each(
                                CASE WHEN jsonb_typeof(raw) = 'object'
                                THEN raw ELSE '{}'::JSONB END
                            ) AS item(key, value)
                            WHERE jsonb_typeof(item.value) IN ('number', 'boolean', 'null')
                               OR (
                                   jsonb_typeof(item.value) = 'string'
                                   AND length(item.value #>> '{}') <= 512
                               )
                               OR (
                                   jsonb_typeof(item.value) = 'object'
                                   AND pg_column_size(item.value) <= 2048
                               )
                        ),
                        '{}'::JSONB
                    );
                    NEW.training_feature_snapshot_version := 1;
                    RETURN NEW;
                END;
                $$
                """
            )
        )
        await conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_trigger
                        WHERE tgname = 'trg_shadow_backtests_training_feature_snapshot'
                          AND tgrelid = 'shadow_backtests'::regclass
                    ) THEN
                        CREATE TRIGGER trg_shadow_backtests_training_feature_snapshot
                        BEFORE INSERT OR UPDATE OF feature_snapshot ON shadow_backtests
                        FOR EACH ROW EXECUTE FUNCTION bb_sync_shadow_training_feature_snapshot();
                    END IF;
                END;
                $$
                """
            )
        )
        # The trainer retains up to 20,000 completed samples. Backfill all
        # historical shadows once so switching reads never shrinks that window.
        await conn.execute(
            text(
                """
                UPDATE shadow_backtests
                SET feature_snapshot = feature_snapshot
                WHERE training_feature_snapshot_version < 1
                """
            )
        )
        return

    if "sqlite" not in settings.database_url:
        return
    result = await conn.execute(text("PRAGMA table_info(shadow_backtests)"))
    existing = {str(row[1]) for row in result.fetchall()}
    for name, column_type in (
        ("training_feature_snapshot", "JSON"),
        ("training_feature_snapshot_version", "INTEGER NOT NULL DEFAULT 0"),
        ("label_version", "VARCHAR(80)"),
    ):
        if name not in existing:
            await conn.execute(text(f"ALTER TABLE shadow_backtests ADD COLUMN {name} {column_type}"))
    await conn.execute(
        text(
            "UPDATE shadow_backtests "
            "SET label_version = 'legacy-row-' || id "
            "WHERE label_version IS NULL OR label_version = ''"
        )
    )
    await conn.execute(
        text(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_shadow_decision_horizon_label_version "
            "ON shadow_backtests (decision_id, horizon_minutes, label_version) "
            "WHERE decision_id IS NOT NULL"
        )
    )


async def _ensure_okx_account_bill_indexes(conn: Any) -> None:
    index_ddls = (
        (
            "idx_okx_account_bills_mode_inst_ts",
            "CREATE INDEX IF NOT EXISTS idx_okx_account_bills_mode_inst_ts ON okx_account_bills (mode, inst_id, bill_ts DESC)",
        ),
        (
            "idx_okx_account_bills_mode_bill_ts",
            "CREATE INDEX IF NOT EXISTS idx_okx_account_bills_mode_bill_ts ON okx_account_bills (mode, bill_ts DESC)",
        ),
        (
            "idx_okx_account_bills_mode_type_ts",
            "CREATE INDEX IF NOT EXISTS idx_okx_account_bills_mode_type_ts ON okx_account_bills (mode, bill_type, bill_sub_type, bill_ts DESC)",
        ),
    )
    if "postgresql" in settings.database_url:
        index_names = await _postgres_index_names(conn)
        for name, ddl in index_ddls:
            if name not in index_names:
                await conn.execute(text(ddl))
        return
    for _name, ddl in index_ddls:
        await conn.execute(text(ddl))


async def _ensure_okx_position_history_column_widths(conn: Any) -> None:
    """Keep evolving OKX lifecycle status text from failing settlement writes."""

    if "postgresql" not in settings.database_url:
        return
    await conn.execute(
        text(
            "ALTER TABLE okx_position_history "
            "ALTER COLUMN match_status TYPE VARCHAR(160)"
        )
    )


async def _ensure_okx_position_history_indexes(conn: Any) -> None:
    index_ddls = (
        (
            "idx_okx_position_history_mode_updated",
            "CREATE INDEX IF NOT EXISTS idx_okx_position_history_mode_updated ON okx_position_history (mode, updated_at_okx DESC)",
        ),
        (
            "idx_okx_position_history_mode_inst_updated",
            "CREATE INDEX IF NOT EXISTS idx_okx_position_history_mode_inst_updated ON okx_position_history (mode, inst_id, updated_at_okx DESC)",
        ),
        (
            "idx_okx_position_history_mode_pos_id",
            "CREATE INDEX IF NOT EXISTS idx_okx_position_history_mode_pos_id ON okx_position_history (mode, pos_id)",
        ),
        (
            "idx_okx_position_history_mode_symbol",
            "CREATE INDEX IF NOT EXISTS idx_okx_position_history_mode_symbol ON okx_position_history (mode, symbol)",
        ),
    )
    if "postgresql" in settings.database_url:
        index_names = await _postgres_index_names(conn)
        for name, ddl in index_ddls:
            if name not in index_names:
                await conn.execute(text(ddl))
        return
    for _name, ddl in index_ddls:
        await conn.execute(text(ddl))


async def _postgres_table_columns(conn: Any, table_name: str) -> set[str]:
    result = await conn.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = :table_name
            """
        ),
        {"table_name": table_name},
    )
    return {str(row[0]) for row in result.fetchall()}


async def _postgres_index_names(conn: Any) -> set[str]:
    result = await conn.execute(
        text(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = current_schema()
            """
        )
    )
    return {str(row[0]) for row in result.fetchall()}


async def close_db() -> None:
    """Dispose engine. Called at application shutdown."""
    global _engine, _sessionmaker
    if _engine:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
