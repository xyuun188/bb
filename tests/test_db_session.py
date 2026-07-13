import pytest

import db.session as session_module


class _FakeConnection:
    def __init__(
        self,
        *,
        table_columns: dict[str, set[str]] | None = None,
        index_names: set[str] | None = None,
    ) -> None:
        self.statements: list[str] = []
        self.table_columns = table_columns or {}
        self.index_names = index_names or set()

    async def execute(self, statement, params=None):
        statement_text = str(statement)
        self.statements.append(statement_text)
        if "information_schema.columns" in statement_text:
            table_name = (params or {}).get("table_name", "")
            return _FakeResult([(name,) for name in self.table_columns.get(table_name, set())])
        if "pg_indexes" in statement_text:
            return _FakeResult([(name,) for name in self.index_names])
        return _FakeResult([])


class _FakeResult:
    def __init__(self, rows) -> None:
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self) -> None:
        self.rollback_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def rollback(self) -> None:
        self.rollback_count += 1


class _FakeMaker:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    def __call__(self) -> _FakeSession:
        return self.session


@pytest.mark.asyncio
async def test_read_session_ctx_does_not_rollback_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession()

    async def fake_get_sessionmaker():
        return _FakeMaker(fake_session)

    monkeypatch.setattr(session_module, "get_sessionmaker", fake_get_sessionmaker)

    async with session_module.get_read_session_ctx() as session:
        assert session is fake_session

    assert fake_session.rollback_count == 0


@pytest.mark.asyncio
async def test_read_session_ctx_rolls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession()

    async def fake_get_sessionmaker():
        return _FakeMaker(fake_session)

    monkeypatch.setattr(session_module, "get_sessionmaker", fake_get_sessionmaker)

    with pytest.raises(RuntimeError):
        async with session_module.get_read_session_ctx():
            raise RuntimeError("boom")

    assert fake_session.rollback_count == 1


@pytest.mark.asyncio
async def test_postgres_schema_init_uses_advisory_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_conn = _FakeConnection()
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )

    await session_module._lock_schema_migration_if_needed(fake_conn)

    assert fake_conn.statements == [
        "SELECT pg_advisory_xact_lock(hashtext('bb:init_db_schema'))"
    ]


@pytest.mark.asyncio
async def test_sqlite_schema_init_does_not_use_advisory_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = _FakeConnection()
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "sqlite+aiosqlite:///tmp/test.db",
    )

    await session_module._lock_schema_migration_if_needed(fake_conn)

    assert fake_conn.statements == []


@pytest.mark.asyncio
async def test_postgres_expert_memory_storage_contract_neutralizes_legacy_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection()

    await session_module._ensure_expert_memory_storage_contract(fake_conn)

    assert any(
        "confidence_adjustment DOUBLE PRECISION NOT NULL DEFAULT 0.0" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "position_size_multiplier DOUBLE PRECISION NOT NULL DEFAULT 1.0" in statement
        for statement in fake_conn.statements
    )
    update = next(
        statement
        for statement in fake_conn.statements
        if statement.lstrip().startswith("UPDATE expert_memories")
    )
    assert "confidence_adjustment = 0.0" in update
    assert "position_size_multiplier = 1.0" in update
    assert "updated_at" not in update


@pytest.mark.asyncio
async def test_sqlite_expert_memory_storage_contract_adds_neutral_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "sqlite+aiosqlite:///tmp/test.db",
    )
    fake_conn = _FakeConnection()

    await session_module._ensure_expert_memory_storage_contract(fake_conn)

    assert any(
        statement
        == "ALTER TABLE expert_memories ADD COLUMN confidence_adjustment FLOAT NOT NULL DEFAULT 0.0"
        for statement in fake_conn.statements
    )
    assert any(
        statement
        == "ALTER TABLE expert_memories ADD COLUMN position_size_multiplier FLOAT NOT NULL DEFAULT 1.0"
        for statement in fake_conn.statements
    )
    assert any(
        statement.lstrip().startswith("UPDATE expert_memories")
        for statement in fake_conn.statements
    )


@pytest.mark.asyncio
async def test_postgres_trade_fact_columns_skip_existing_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection(
        table_columns={
            "positions": {
                "okx_inst_id",
                "okx_pos_id",
                "entry_exchange_order_id",
                "close_exchange_order_id",
                "close_fill_pnl",
                "entry_fee",
                "close_fee",
                "funding_fee",
                "settlement_status",
                "settlement_source",
                "settlement_synced_at",
                "settlement_raw",
            },
            "orders": {
                "okx_inst_id",
                "okx_trade_ids",
                "okx_fill_contracts",
                "okx_fill_pnl",
                "okx_state",
                "okx_sync_status",
                "okx_synced_at",
                "okx_last_error",
                "okx_raw_fills",
            },
        }
    )

    await session_module._ensure_trade_fact_columns(fake_conn)

    assert not any(statement.startswith("ALTER TABLE") for statement in fake_conn.statements)
    assert any(
        "ALTER COLUMN entry_exchange_order_id TYPE VARCHAR(500)" in statement
        for statement in fake_conn.statements
    )


@pytest.mark.asyncio
async def test_postgres_trade_fact_columns_add_only_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection(
        table_columns={
            "positions": {
                "okx_inst_id",
                "okx_pos_id",
                "entry_exchange_order_id",
                "close_exchange_order_id",
                "close_fill_pnl",
                "entry_fee",
                "close_fee",
                "funding_fee",
                "settlement_status",
                "settlement_source",
                "settlement_synced_at",
                "settlement_raw",
            },
            "orders": {
                "okx_trade_ids",
                "okx_fill_contracts",
                "okx_fill_pnl",
                "okx_state",
                "okx_sync_status",
                "okx_synced_at",
                "okx_last_error",
                "okx_raw_fills",
            },
        }
    )

    await session_module._ensure_trade_fact_columns(fake_conn)

    alter_statements = [
        statement for statement in fake_conn.statements if statement.startswith("ALTER TABLE")
    ]
    assert alter_statements == ["ALTER TABLE orders ADD COLUMN okx_inst_id VARCHAR(64)"]


@pytest.mark.asyncio
async def test_postgres_trade_fact_indexes_skip_existing_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection(
        index_names={
            "idx_positions_okx_inst_side",
            "idx_positions_entry_exchange_order",
            "idx_positions_close_exchange_order",
            "idx_positions_closed_scan",
            "idx_positions_created_scan",
            "idx_positions_open_created_scan",
            "idx_orders_filled_exchange_scan",
            "idx_orders_decision_side_scan",
            "idx_orders_exchange_order_id",
            "idx_orders_okx_inst_id",
            "idx_orders_okx_sync_status",
            "idx_ai_decisions_pending_entry_recent",
            "idx_ai_decisions_recent_scan",
            "idx_ai_decisions_strategy_learning_recent",
            "idx_shadow_backtests_training_completed",
        }
    )

    await session_module._ensure_trade_fact_indexes(fake_conn)

    assert not any(statement.startswith("CREATE INDEX") for statement in fake_conn.statements)


@pytest.mark.asyncio
async def test_postgres_trade_fact_indexes_create_pending_entry_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection()

    await session_module._ensure_trade_fact_indexes(fake_conn)

    assert any(
        "CREATE INDEX IF NOT EXISTS idx_ai_decisions_pending_entry_recent" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "CREATE INDEX IF NOT EXISTS idx_ai_decisions_recent_scan" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "CREATE INDEX IF NOT EXISTS idx_ai_decisions_strategy_learning_recent" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "CREATE INDEX IF NOT EXISTS idx_shadow_backtests_training_completed" in statement
        for statement in fake_conn.statements
    )


@pytest.mark.asyncio
async def test_postgres_model_health_snapshot_schema_uses_trigger_and_bounded_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection()

    await session_module._ensure_ai_decision_model_health_columns(fake_conn)

    assert any(
        "ADD COLUMN IF NOT EXISTS model_health_timings JSONB" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "CREATE OR REPLACE FUNCTION bb_sync_ai_decision_model_health" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "trg_ai_decisions_model_health_snapshot" in statement
        for statement in fake_conn.statements
    )
    assert any("LIMIT 1500" in statement for statement in fake_conn.statements)
    assert any(
        "decision_learning_snapshot_version" in statement for statement in fake_conn.statements
    )


@pytest.mark.asyncio
async def test_postgres_shadow_training_snapshot_schema_uses_trigger_and_full_backfill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection()

    await session_module._ensure_shadow_backtest_training_snapshot_columns(fake_conn)

    assert any(
        "ADD COLUMN IF NOT EXISTS training_feature_snapshot JSONB" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "CREATE OR REPLACE FUNCTION bb_sync_shadow_training_feature_snapshot" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "trg_shadow_backtests_training_feature_snapshot" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "UPDATE shadow_backtests" in statement and "training_feature_snapshot_version < 1" in statement
        for statement in fake_conn.statements
    )
