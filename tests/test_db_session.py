import json
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import Text, text
from sqlalchemy.ext.asyncio import create_async_engine

import db.session as session_module
from core.training_contracts import AUTHORITATIVE_TRADE_OUTCOME_VERSION
from core.training_feature_snapshot import TRAINING_FEATURE_SNAPSHOT_VERSION
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
        self.expunge_count = 0
        self.info: dict[str, object] = {}
        self.statements: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def rollback(self) -> None:
        self.rollback_count += 1

    def expunge_all(self) -> None:
        self.expunge_count += 1

    async def execute(self, statement, params=None):
        self.statements.append(str(statement))
        return _FakeResult([])


class _FakeMaker:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    def __call__(self) -> _FakeSession:
        return self.session


class _FakeConnectionOnly:
    def __init__(self) -> None:
        self.statements: list[str] = []

    async def execute(self, statement, params=None):
        self.statements.append(str(statement))


class _FakeConnectionEngine:
    def __init__(self) -> None:
        self.connection = _FakeConnectionOnly()
        self.begin_called = False

    @asynccontextmanager
    async def connect(self):
        yield self.connection

    @asynccontextmanager
    async def begin(self):
        self.begin_called = True
        yield self.connection


@pytest.mark.asyncio
async def test_read_session_ctx_rolls_back_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession()

    async def fake_get_sessionmaker():
        return _FakeMaker(fake_session)

    monkeypatch.setattr(session_module, "get_sessionmaker", fake_get_sessionmaker)

    async with session_module.get_read_session_ctx() as session:
        assert session is fake_session

    assert fake_session.rollback_count == 1
    assert fake_session.expunge_count == 1


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
    assert fake_session.expunge_count == 1


@pytest.mark.asyncio
async def test_read_session_ctx_sets_postgres_transaction_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = _FakeSession()

    async def fake_get_sessionmaker():
        return _FakeMaker(fake_session)

    monkeypatch.setattr(session_module, "get_sessionmaker", fake_get_sessionmaker)
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )

    async with session_module.get_read_session_ctx(
        statement_timeout_ms=800,
        idle_transaction_timeout_ms=1100,
    ):
        pass

    assert fake_session.statements == [
        "SELECT set_config('statement_timeout', :value, true)",
        "SELECT set_config('idle_in_transaction_session_timeout', :value, true)",
    ]
    assert fake_session.rollback_count == 1


@pytest.mark.asyncio
async def test_read_session_ctx_sets_isolation_before_transaction_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = _FakeSession()

    async def fake_get_sessionmaker():
        return _FakeMaker(fake_session)

    monkeypatch.setattr(session_module, "get_sessionmaker", fake_get_sessionmaker)
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )

    async with session_module.get_read_session_ctx(
        transaction_isolation="REPEATABLE READ READ ONLY",
    ):
        pass

    assert fake_session.statements == [
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY",
        "SELECT set_config('statement_timeout', :value, true)",
        "SELECT set_config('idle_in_transaction_session_timeout', :value, true)",
    ]
    assert fake_session.info["bb_consistent_read_snapshot_started"] is True


@pytest.mark.asyncio
async def test_read_session_ctx_rejects_untrusted_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = _FakeSession()

    async def fake_get_sessionmaker():
        return _FakeMaker(fake_session)

    monkeypatch.setattr(session_module, "get_sessionmaker", fake_get_sessionmaker)
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )

    with pytest.raises(ValueError, match="unsupported read transaction isolation"):
        async with session_module.get_read_session_ctx(
            transaction_isolation="READ COMMITTED; DROP TABLE orders",
        ):
            pass


@pytest.mark.asyncio
async def test_runtime_audit_indexes_cover_dashboard_list_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conn = _FakeConnection()
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )

    await session_module._ensure_runtime_audit_indexes(fake_conn)

    assert any("idx_ai_decisions_mode_created" in statement for statement in fake_conn.statements)
    assert any("idx_orders_mode_created" in statement for statement in fake_conn.statements)
    assert any("idx_positions_mode_closed_created" in statement for statement in fake_conn.statements)


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
async def test_read_only_db_init_verifies_connection_without_schema_migration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _FakeConnectionEngine()

    async def fake_get_engine():
        return engine

    monkeypatch.setattr(session_module, "get_engine", fake_get_engine)

    await session_module.init_db(migrate_schema=False)

    assert engine.connection.statements == ["SELECT 1"]
    assert engine.begin_called is False


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
async def test_postgres_deletes_every_incomplete_expert_memory_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection()

    await session_module._delete_non_authoritative_expert_memories(fake_conn)

    statement = fake_conn.statements[0]
    assert statement.lstrip().startswith("DELETE FROM expert_memories")
    assert "extra::jsonb ->> 'source'" in statement
    assert "'production_evidence_eligible' = 'true'::jsonb" in statement
    assert "'cost_complete' = 'true'::jsonb" in statement
    assert "outcome_id" in statement
    assert "outcome_fingerprint" in statement


@pytest.mark.asyncio
async def test_sqlite_physically_deletes_non_authoritative_expert_memories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.settings, "database_url", "sqlite+aiosqlite:///:memory:")
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    valid = {
        "source": "authoritative_trade_outcome",
        "authority_level": "okx_settlement_and_execution",
        "outcome_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
        "production_evidence_eligible": True,
        "cost_complete": True,
        "outcome_id": "ato:valid",
        "outcome_fingerprint": "fingerprint-valid",
    }
    invalid_rows = [
        {**valid, "source": "local_provisional_reflection"},
        {**valid, "source": "shadow_backtest"},
        {**valid, "outcome_fingerprint": ""},
        {**valid, "production_evidence_eligible": "true"},
    ]
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("CREATE TABLE expert_memories (id INTEGER PRIMARY KEY, extra JSON)")
            )
            for index, payload in enumerate([valid, *invalid_rows], start=1):
                await conn.execute(
                    text("INSERT INTO expert_memories (id, extra) VALUES (:id, :extra)"),
                    {"id": index, "extra": json.dumps(payload)},
                )

            await session_module._delete_non_authoritative_expert_memories(conn)
            rows = (
                await conn.execute(text("SELECT id FROM expert_memories ORDER BY id"))
            ).fetchall()

        assert rows == [(1,)]
    finally:
        await engine.dispose()


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
            "uq_orders_execution_exchange_fact",
            "idx_orders_okx_inst_id",
            "idx_orders_okx_sync_status",
            "idx_ai_decisions_pending_entry_recent",
            "idx_ai_decisions_recent_scan",
            "idx_ai_decisions_strategy_learning_recent",
            "idx_ai_decisions_market_selection",
            "idx_ai_decisions_market_learning_recent",
            "idx_shadow_backtests_training_completed",
            session_module.SHADOW_EXECUTABLE_TRAINING_INDEX_NAME,
            session_module.SHADOW_CRYPTO_TRAINING_INDEX_NAME,
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

    assert session_module.SHADOW_EXECUTABLE_TRAINING_INDEX_DDL in fake_conn.statements
    assert session_module.SHADOW_CRYPTO_TRAINING_INDEX_DDL in fake_conn.statements
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
        "CREATE INDEX IF NOT EXISTS idx_ai_decisions_market_learning_recent" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "CREATE INDEX IF NOT EXISTS idx_shadow_backtests_training_completed" in statement
        for statement in fake_conn.statements
    )


@pytest.mark.asyncio
async def test_postgres_duplicate_exchange_fact_repair_selects_canonical_order_before_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session_module.settings,
        "database_url",
        "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql",
    )
    fake_conn = _FakeConnection()

    await session_module._repair_duplicate_exchange_order_facts(fake_conn)

    assert len(fake_conn.statements) == 3
    position_repair = fake_conn.statements[0]
    assert "WITH canonical_orders AS" in position_repair
    assert "SELECT DISTINCT ON (o.execution_mode, o.exchange_order_id)" in position_repair
    assert "JOIN canonical_orders o" in position_repair
    assert "o.canonical_contracts" in position_repair
    assert "FIRST_VALUE(p.id)" in position_repair
    assert "duplicate_local_closed_position_for_same_okx_close_order" in position_repair

    exact_projection_repair = fake_conn.statements[1]
    assert "WITH keyed AS" in exact_projection_repair
    assert "string_to_array(COALESCE(p.close_exchange_order_id, ''), ',')" in exact_projection_repair
    assert "PARTITION BY" in exact_projection_repair
    assert "keyed.entry_order_ids" in exact_projection_repair
    assert "duplicate_local_closed_position_for_same_okx_lifecycle_exact_projection" in exact_projection_repair

    order_repair = fake_conn.statements[2]
    assert "PARTITION BY o.execution_mode, o.exchange_order_id" in order_repair
    assert "exchange_order_id = NULL" in order_repair


@pytest.mark.asyncio
async def test_postgres_model_health_snapshot_schema_uses_trigger_without_history_rewrite(
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
        "CREATE OR REPLACE FUNCTION bb_compact_ai_decision_learning_value" in statement
        for statement in fake_conn.statements
    )
    projection_function = next(
        statement
        for statement in fake_conn.statements
        if "CREATE OR REPLACE FUNCTION bb_project_profit_risk_sizing_snapshot"
        in statement
    )
    assert "'dynamic_leverage_decision'" in projection_function
    assert "WHEN 'portfolio_risk_snapshot'" in projection_function
    assert "'current_stressed_loss_usdt'" in projection_function
    assert "'exchange_contract_specs'" not in projection_function
    from models.decision import _PROFIT_RISK_SIZING_SNAPSHOT_KEYS

    for key in _PROFIT_RISK_SIZING_SNAPSHOT_KEYS:
        assert f"'{key}'" in projection_function
    assert any(
        "trg_ai_decisions_model_health_snapshot" in statement
        for statement in fake_conn.statements
    )
    assert not any("WITH recent_unprojected" in statement for statement in fake_conn.statements)
    assert not any("UPDATE ai_decisions AS decision" in statement for statement in fake_conn.statements)
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
    assert "bb_compact_ai_decision_learning_value" in retention_trigger
    assert "bb_project_profit_risk_sizing_snapshot" in retention_trigger
    assert "item.key = 'normal_paper_trade'" in retention_trigger
    assert "decision_learning_snapshot_version := 7" in retention_trigger


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
async def test_postgres_shadow_training_snapshot_schema_defers_history_repair(
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
        "CREATE OR REPLACE FUNCTION bb_compact_training_feature_snapshot" in statement
        for statement in fake_conn.statements
    )
    assert any(
        f"NEW.training_feature_snapshot_version := {TRAINING_FEATURE_SNAPSHOT_VERSION}"
        in statement
        and "bb_compact_training_feature_snapshot(" in statement
        for statement in fake_conn.statements
    )
    assert any(
        "trg_shadow_backtests_training_feature_snapshot" in statement
        for statement in fake_conn.statements
    )
    assert not any(
        "SET feature_snapshot" in statement for statement in fake_conn.statements
    )
    assert any(
        "bb_shadow_training_repair_progress" in statement
        and "COALESCE(MAX(id), 0)" in statement
        and "ON CONFLICT (repair_key) DO NOTHING" in statement
        for statement in fake_conn.statements
    )


class _ShadowRepairConnection(_FakeConnection):
    def __init__(self, *, progress=(0, 800, False), batch=(100, 100, 3), fail=False):
        super().__init__()
        self.progress = progress
        self.batch = batch
        self.fail = fail
        self.parameters = []

    async def execute(self, statement, params=None):
        self.statements.append(str(statement))
        self.parameters.append(params)
        if "FOR UPDATE SKIP LOCKED" in str(statement):
            return _FakeResult([] if self.progress is None else [self.progress])
        if "WITH scanned AS MATERIALIZED" in str(statement):
            if self.fail:
                raise RuntimeError("interrupted repair")
            return _FakeResult([self.batch])
        return _FakeResult([])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("progress", "batch", "completed"),
    [
        ((0, 800, False), (100, 100, 3), False),
        ((100, 800, False), (20, 800, 2), True),
        ((100, 800, False), (0, 100, 0), True),
        ((700, 800, False), (100, 800, 0), True),
    ],
)
async def test_shadow_repair_advances_bounded_window_and_checkpoint(
    progress, batch, completed
) -> None:
    conn = _ShadowRepairConnection(progress=progress, batch=batch)
    report = await session_module._repair_shadow_training_snapshot_batch(conn)

    assert report["completed"] is completed
    assert report["scanned"] == batch[0]
    assert report["repaired"] == batch[2]
    query = next(sql for sql in conn.statements if "WITH scanned AS MATERIALIZED" in sql)
    window, predicate = query.split("), repaired AS")
    assert "LIMIT :batch_size" in window
    assert "feature_snapshot::JSONB" not in window
    assert "id > :last_id AND id <= :upper_id" in window
    assert "training_market_fact_contract" in predicate
    assert "training_label_contract" in predicate
    assert "training_feature_snapshot_version" in predicate
    assert "< :target_version" in predicate
    assert "local_ai_tools_shadow" in predicate
    assert "local_ml_shadow" in predicate
    assert "shadow_settlement_cost_observation" in predicate
    assert "inst_category" in predicate
    assert "instCategory" in predicate
    assert "NOT COALESCE(" in predicate
    assert conn.parameters[-1] == {
        "repair_key": session_module._SHADOW_TRAINING_REPAIR_KEY,
        "last_id": batch[1],
        "completed": completed,
    }
    repair_parameters = conn.parameters[-2]
    assert repair_parameters["target_version"] == TRAINING_FEATURE_SNAPSHOT_VERSION
    assert any("statement_timeout = '5s'" in sql for sql in conn.statements)
    assert any("lock_timeout = '1s'" in sql for sql in conn.statements)


@pytest.mark.asyncio
@pytest.mark.parametrize("progress", [None, (800, 800, True)])
async def test_shadow_repair_skips_busy_or_completed_work(progress) -> None:
    conn = _ShadowRepairConnection(progress=progress)
    report = await session_module._repair_shadow_training_snapshot_batch(conn)

    assert report["completed"] is (progress is not None)
    assert not any("WITH scanned" in sql for sql in conn.statements)


@pytest.mark.asyncio
async def test_shadow_repair_does_not_advance_checkpoint_on_query_failure() -> None:
    conn = _ShadowRepairConnection(fail=True)
    with pytest.raises(RuntimeError, match="interrupted repair"):
        await session_module._repair_shadow_training_snapshot_batch(conn)
    assert not any(
        "UPDATE bb_shadow_training_repair_progress" in sql for sql in conn.statements
    )


@pytest.mark.asyncio
async def test_shadow_repair_worker_retries_with_smaller_transaction_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.settings, "database_url", "postgresql+asyncpg://bb")
    transaction_outcomes = []
    batch_sizes = []
    pauses = []

    class Engine:
        @asynccontextmanager
        async def begin(self):
            try:
                yield object()
            except Exception:
                transaction_outcomes.append("rollback")
                raise
            else:
                transaction_outcomes.append("commit")

    async def get_engine():
        return Engine()

    async def repair(_conn, *, batch_size):
        batch_sizes.append(batch_size)
        if len(batch_sizes) == 1:
            raise TimeoutError("slow historical payload")
        return {"completed": len(batch_sizes) == 3, "scanned": 50, "repaired": 1}

    async def sleep(seconds):
        pauses.append(seconds)

    monkeypatch.setattr(session_module, "get_engine", get_engine)
    monkeypatch.setattr(session_module, "_repair_shadow_training_snapshot_batch", repair)
    monkeypatch.setattr(session_module.asyncio, "sleep", sleep)
    await session_module.repair_shadow_training_snapshots()

    assert transaction_outcomes == ["rollback", "commit", "commit"]
    assert batch_sizes == [100, 50, 50]
    assert pauses == [5, 0.1]


@pytest.mark.asyncio
async def test_shadow_repair_worker_backs_off_after_repeated_database_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.settings, "database_url", "postgresql+asyncpg://bb")
    batch_sizes = []
    pauses = []
    attempts = 0

    class Engine:
        @asynccontextmanager
        async def begin(self):
            yield object()

    async def get_engine():
        return Engine()

    async def repair(_conn, *, batch_size):
        nonlocal attempts
        attempts += 1
        batch_sizes.append(batch_size)
        if attempts <= 8:
            raise TimeoutError("temporary database timeout")
        return {"completed": True, "scanned": 1, "repaired": 0}

    async def sleep(seconds):
        pauses.append(seconds)

    monkeypatch.setattr(session_module, "get_engine", get_engine)
    monkeypatch.setattr(session_module, "_repair_shadow_training_snapshot_batch", repair)
    monkeypatch.setattr(session_module.asyncio, "sleep", sleep)

    await session_module.repair_shadow_training_snapshots()

    assert batch_sizes == [100, 50, 25, 12, 6, 3, 1, 1, 1]
    assert pauses == [5, 10, 20, 40, 80, 160, 300, 300]


@pytest.mark.asyncio
async def test_shadow_repair_worker_propagates_shutdown_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module.settings, "database_url", "postgresql+asyncpg://bb")
    transaction_closed = []

    class Engine:
        @asynccontextmanager
        async def begin(self):
            try:
                yield object()
            finally:
                transaction_closed.append(True)

    async def get_engine():
        return Engine()

    async def repair(_conn, *, batch_size):
        raise session_module.asyncio.CancelledError

    monkeypatch.setattr(session_module, "get_engine", get_engine)
    monkeypatch.setattr(session_module, "_repair_shadow_training_snapshot_batch", repair)
    with pytest.raises(session_module.asyncio.CancelledError):
        await session_module.repair_shadow_training_snapshots()
    assert transaction_closed == [True]


def test_postgres_training_feature_snapshot_functions_are_single_ddl_statements() -> None:
    from core.training_feature_snapshot import postgres_training_feature_snapshot_functions

    statements = postgres_training_feature_snapshot_functions()

    assert len(statements) == 7
    assert [
        statement.count("CREATE OR REPLACE FUNCTION")
        for statement in statements
    ] == [1, 1, 1, 1, 1, 1, 1]
    assert [
        "bb_training_feature_storage_bytes",
        "bb_training_feature_normalize_jsonb",
        "bb_compact_market_fact",
        "bb_compact_training_market_fact_contract",
        "bb_project_specialist_shadow_evidence",
        "bb_compact_specialist_shadow",
        "bb_compact_training_feature_snapshot",
    ] == [
        next(
            line.split("FUNCTION ", 1)[1].split("(", 1)[0]
            for line in statement.splitlines()
            if "CREATE OR REPLACE FUNCTION " in line
        )
        for statement in statements
    ]


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
    assert any(
        "CREATE OR REPLACE FUNCTION bb_compact_training_feature_snapshot" in statement
        for statement in fake_conn.statements
    )
    assert not any(
        "UPDATE shadow_backtests" in statement
        and "training_feature_snapshot_version < 2" in statement
        for statement in fake_conn.statements
    )

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
