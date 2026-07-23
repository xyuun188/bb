import pytest
from sqlalchemy import Text

import db.session as session_module
from models.trade import Order


class _FakeConnection:
    def __init__(
        self,
        *,
        table_columns: dict[str, set[str]] | None = None,
        table_column_specs: dict[str, dict[str, dict[str, object]]] | None = None,
        index_names: set[str] | None = None,
    ) -> None:
        self.statements: list[str] = []
        self.table_column_specs = table_column_specs or {}
        derived_columns = {
            table_name: set(specs)
            for table_name, specs in self.table_column_specs.items()
        }
        self.table_columns = table_columns or derived_columns
        self.index_names = index_names or set()

    async def execute(self, statement, params=None):
        statement_text = str(statement)
        self.statements.append(statement_text)
        if "information_schema.columns" in statement_text:
            table_name = (params or {}).get("table_name", "")
            if "character_maximum_length" in statement_text:
                specs = self.table_column_specs.get(table_name, {})
                return _FakeResult(
                    [
                        (
                            name,
                            spec.get("data_type", ""),
                            spec.get("character_maximum_length"),
                            spec.get("is_nullable", "YES"),
                            spec.get("column_default"),
                        )
                        for name, spec in specs.items()
                    ]
                )
            return _FakeResult([(name,) for name in self.table_columns.get(table_name, set())])
        if "pg_indexes" in statement_text:
            return _FakeResult([(name,) for name in self.index_names])
        if statement_text.startswith("PRAGMA table_info("):
            table_name = statement_text.removeprefix("PRAGMA table_info(").removesuffix(")")
            return _FakeResult(
                [(index, name) for index, name in enumerate(self.table_columns.get(table_name, set()))]
            )
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
async def test_postgres_drops_removed_expert_memory_policy_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection(
        table_columns={
            "expert_memories": {
                "confidence_adjustment",
                "position_size_multiplier",
            }
        }
    )

    await session_module._drop_removed_expert_memory_policy_columns(fake_conn)

    assert [
        statement
        for statement in fake_conn.statements
        if statement.startswith("ALTER TABLE")
    ] == [
        "ALTER TABLE expert_memories DROP COLUMN confidence_adjustment",
        "ALTER TABLE expert_memories DROP COLUMN position_size_multiplier",
    ]


@pytest.mark.asyncio
async def test_postgres_removed_expert_columns_skip_steady_state_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection(table_columns={"expert_memories": {"id"}})

    await session_module._drop_removed_expert_memory_policy_columns(fake_conn)

    assert not any(statement.startswith("ALTER TABLE") for statement in fake_conn.statements)


@pytest.mark.asyncio
async def test_sqlite_drops_removed_expert_memory_policy_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "sqlite+aiosqlite:///tmp/test.db",
    )
    fake_conn = _FakeConnection(
        table_columns={
            "expert_memories": {
                "id",
                "confidence_adjustment",
                "position_size_multiplier",
            }
        }
    )

    await session_module._drop_removed_expert_memory_policy_columns(fake_conn)

    assert fake_conn.statements[0] == "PRAGMA table_info(expert_memories)"
    assert set(fake_conn.statements[1:]) == {
        "ALTER TABLE expert_memories DROP COLUMN confidence_adjustment",
        "ALTER TABLE expert_memories DROP COLUMN position_size_multiplier",
    }


@pytest.mark.asyncio
async def test_schema_init_drops_removed_model_performance_snapshots_table() -> None:
    fake_conn = _FakeConnection()

    await session_module._drop_removed_model_performance_snapshots_table(fake_conn)

    assert fake_conn.statements == ["DROP TABLE IF EXISTS model_performance_snapshots"]


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
                "current_management_contract",
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
    assert any(
        "ALTER COLUMN okx_trade_ids TYPE TEXT" in statement
        for statement in fake_conn.statements
    )


def test_order_trade_ids_uses_unbounded_text_column() -> None:
    assert isinstance(Order.__table__.c.okx_trade_ids.type, Text)


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
                "current_management_contract",
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
        "ADD COLUMN model_health_timings JSONB" in statement
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
    retention_trigger = next(
        statement
        for statement in fake_conn.statements
        if "CREATE OR REPLACE FUNCTION bb_sync_ai_decision_model_health" in statement
    )
    assert "preserve_ai_decision_projections" in retention_trigger
    assert "runtime_data_retention" in retention_trigger


@pytest.mark.asyncio
async def test_postgres_model_health_snapshot_skips_steady_state_alters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    columns = {
        "model_health_timings",
        "model_health_fallback_timings",
        "model_health_experts",
        "model_health_opinions",
        "model_health_has_ml_signal",
        "model_health_has_local_ml_signal",
        "model_health_has_local_ai_tools",
        "model_health_snapshot_version",
        "decision_learning_snapshot",
        "decision_learning_snapshot_version",
    }
    fake_conn = _FakeConnection(table_columns={"ai_decisions": columns})

    await session_module._ensure_ai_decision_model_health_columns(fake_conn)

    assert not any(statement.startswith("ALTER TABLE") for statement in fake_conn.statements)


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
        "ADD COLUMN training_feature_snapshot JSONB" in statement
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


@pytest.mark.asyncio
async def test_postgres_shadow_snapshot_skips_steady_state_schema_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.training_contracts import SHADOW_LABEL_VERSION

    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection(
        table_column_specs={
            "shadow_backtests": {
                "training_feature_snapshot": {"data_type": "jsonb"},
                "training_feature_snapshot_version": {
                    "data_type": "integer",
                    "is_nullable": "NO",
                    "column_default": "0",
                },
                "label_version": {
                    "data_type": "character varying",
                    "character_maximum_length": 80,
                    "is_nullable": "NO",
                    "column_default": f"'{SHADOW_LABEL_VERSION}'::character varying",
                },
            }
        },
        index_names={"uq_shadow_decision_horizon_label_version"},
    )

    await session_module._ensure_shadow_backtest_training_snapshot_columns(fake_conn)

    assert not any(statement.startswith("ALTER TABLE") for statement in fake_conn.statements)


@pytest.mark.asyncio
async def test_postgres_runtime_retention_columns_backfill_markers_and_add_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection()

    await session_module._ensure_runtime_data_retention_columns(fake_conn)

    assert any(
        "ALTER TABLE ai_decisions ADD COLUMN runtime_payload_compaction_version"
        in statement
        for statement in fake_conn.statements
    )
    assert any(
        "ALTER TABLE shadow_backtests ADD COLUMN runtime_payload_compacted_at"
        in statement
        for statement in fake_conn.statements
    )
    assert any(
        "UPDATE ai_decisions" in statement and "_retention,source" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "idx_ai_decisions_payload_compaction" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "idx_shadow_backtests_payload_compaction" in statement
        for statement in fake_conn.statements
    )


@pytest.mark.asyncio
async def test_postgres_runtime_retention_schema_skips_steady_state_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    columns = {
        "runtime_payload_compaction_version",
        "runtime_payload_compacted_at",
    }
    fake_conn = _FakeConnection(
        table_columns={
            "ai_decisions": columns,
            "shadow_backtests": columns,
        },
        index_names={
            "idx_ai_decisions_payload_compaction",
            "idx_shadow_backtests_payload_compaction",
        },
    )

    await session_module._ensure_runtime_data_retention_columns(fake_conn)

    assert not any(statement.startswith("ALTER TABLE") for statement in fake_conn.statements)
    assert not any(statement.startswith("UPDATE ") for statement in fake_conn.statements)
    assert not any(statement.startswith("CREATE INDEX") for statement in fake_conn.statements)
    assert not any(
        statement.startswith("CREATE UNIQUE INDEX") for statement in fake_conn.statements
    )
