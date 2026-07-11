from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Order, Position
from services.manual_close_marker import ORPHAN_QUARANTINE_EXCHANGE_ID_PREFIX
from services.okx_trade_fact_integrity import (
    OkxTradeFactIntegrityService,
    _start_consistent_read_snapshot,
)


async def _reset_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade_fact.db').as_posix()}",
    )
    await init_db()


@pytest.mark.asyncio
async def test_trade_fact_audit_uses_postgres_consistent_read_snapshot() -> None:
    class _FakeBind:
        class dialect:
            name = "postgresql"

    class _FakeSession:
        def __init__(self) -> None:
            self.statements = []

        def get_bind(self):
            return _FakeBind()

        async def execute(self, statement) -> None:
            self.statements.append(statement)

    session = _FakeSession()

    await _start_consistent_read_snapshot(session)

    assert len(session.statements) == 1
    assert str(session.statements[0]) == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
    )


@pytest.mark.asyncio
async def test_trade_fact_audit_keeps_sqlite_test_reads_compatible() -> None:
    class _FakeBind:
        class dialect:
            name = "sqlite"

    class _FakeSession:
        def get_bind(self):
            return _FakeBind()

        async def execute(self, _statement) -> None:
            raise AssertionError("SQLite must not receive PostgreSQL transaction SQL")

    await _start_consistent_read_snapshot(_FakeSession())


def _execution_raw(
    *,
    inst_id: str = "H-USDT-SWAP",
    contracts: float = 100.0,
    contract_size: float = 0.1,
    avg_price: float = 2.44,
) -> dict:
    return {
        "execution_result": {
            "raw_response": {
                "symbol": "WLFI/USDT:USDT",
                "canonical_exchange_symbol": "WLFI/USDT",
                "info": {
                    "instId": inst_id,
                    "accFillSz": str(contracts),
                    "avgPx": str(avg_price),
                },
                "contract_size": contract_size,
                "filled_contracts": contracts,
            }
        }
    }


def _recent_filled_at(*, minutes_ago: int) -> datetime:
    return datetime.now(UTC) - timedelta(minutes=minutes_ago)


def test_legacy_position_history_projection_gap_is_not_a_runtime_blocker() -> None:
    opened_at = _recent_filled_at(minutes_ago=30)
    closed_at = _recent_filled_at(minutes_ago=10)
    position = Position(
        id=101,
        model_name="okx_authoritative_sync",
        execution_mode="paper",
        symbol="ADA/USDT",
        side="long",
        quantity=100.0,
        entry_price=0.6,
        current_price=0.64,
        realized_pnl=4.0,
        settlement_status="okx_position_history",
        settlement_source="okx_position_history",
        is_open=False,
        closed_at=closed_at,
        created_at=opened_at,
    )
    orders = [
        Order(
            id=201,
            model_name="okx_authoritative_sync",
            execution_mode="paper",
            symbol="ADA/USDT",
            side="buy",
            order_type="market",
            quantity=100.0,
            price=0.6,
            status="filled",
            exchange_order_id="ada-entry",
            filled_at=opened_at,
            created_at=opened_at,
        ),
        Order(
            id=202,
            model_name="okx_authoritative_sync",
            execution_mode="paper",
            symbol="ADA/USDT",
            side="sell",
            order_type="market",
            quantity=100.0,
            price=0.64,
            status="filled",
            exchange_order_id="ada-close",
            filled_at=closed_at,
            created_at=closed_at,
        ),
    ]

    issues = OkxTradeFactIntegrityService()._audit_position_authority_links(
        [position],
        orders,
        since=datetime.now(UTC) - timedelta(hours=24),
    )

    assert [issue.kind for issue in issues] == ["legacy_position_history_projection_gap"]
    assert issues[0].severity == "info"


@pytest.mark.asyncio
async def test_contract_count_converts_to_base_quantity_without_false_issue(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=55)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="H/USDT",
                action="long",
                confidence=0.9,
                raw_llm_response=_execution_raw(contracts=100, contract_size=0.1),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="H/USDT",
                        side="buy",
                        order_type="market",
                        quantity=10.0,
                        price=2.44,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="entry-ok",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="H/USDT",
                        side="long",
                        quantity=10.0,
                        entry_price=2.44,
                        current_price=2.44,
                        leverage=3.0,
                        is_open=True,
                        okx_inst_id="H-USDT-SWAP",
                        entry_exchange_order_id="entry-ok",
                        created_at=filled_at + timedelta(seconds=10),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
        assert report["checked_orders"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_linked_orders_are_loaded_even_when_fills_are_older_than_lookback(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        old_fill_time = datetime.now(UTC) - timedelta(hours=72)
        recent_close_time = datetime.now(UTC) - timedelta(minutes=20)
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="buy",
                        order_type="market",
                        quantity=0.0003,
                        price=62456.79,
                        status="filled",
                        exchange_order_id="old-entry-order",
                        filled_at=old_fill_time,
                        created_at=old_fill_time,
                    ),
                    Order(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="sell",
                        order_type="market",
                        quantity=0.0001,
                        price=62731.0,
                        status="filled",
                        exchange_order_id="old-close-order",
                        filled_at=old_fill_time + timedelta(minutes=15),
                        created_at=old_fill_time + timedelta(minutes=15),
                    ),
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="long",
                        quantity=0.0003,
                        entry_price=62456.79,
                        current_price=62731.0,
                        leverage=1.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.09,
                        is_open=False,
                        okx_inst_id="BTC-USDT-SWAP",
                        okx_pos_id="btc-pos",
                        entry_exchange_order_id="old-entry-order",
                        close_exchange_order_id="old-close-order",
                        created_at=old_fill_time,
                        closed_at=recent_close_time,
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
        assert report["checked_orders"] == 0
        assert report["checked_positions"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_okx_raw_fills_win_over_stale_decision_execution_price(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=35)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="AAVE/USDT",
                action="long",
                confidence=0.9,
                raw_llm_response=_execution_raw(
                    inst_id="AAVE-USDT-SWAP",
                    contracts=2,
                    contract_size=1,
                    avg_price=93.57918032786885,
                ),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="buy",
                        order_type="market",
                        quantity=2.0,
                        price=97.54,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="3694561249469370368",
                        filled_at=filled_at,
                        created_at=filled_at,
                        okx_inst_id="AAVE-USDT-SWAP",
                        okx_fill_contracts=2.0,
                        okx_sync_status="okx_confirmed",
                        okx_raw_fills={
                            "order_id": "3694561249469370368",
                            "trade_ids": ["aave-fill-1"],
                            "inst_id": "AAVE-USDT-SWAP",
                            "contracts": 2.0,
                            "contract_size": 1.0,
                            "base_quantity": 2.0,
                            "avg_price": 97.54,
                            "rows": [
                                {
                                    "instId": "AAVE-USDT-SWAP",
                                    "ordId": "3694561249469370368",
                                    "tradeId": "aave-fill-1",
                                    "fillSz": "2",
                                    "fillPx": "97.54",
                                }
                            ],
                        },
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="long",
                        quantity=2.0,
                        entry_price=97.54,
                        current_price=97.54,
                        leverage=3.0,
                        is_open=True,
                        okx_inst_id="AAVE-USDT-SWAP",
                        entry_exchange_order_id="3694561249469370368",
                        created_at=filled_at + timedelta(seconds=10),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
        assert report["checked_orders"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_order_direct_link_wins_over_time_window(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=55)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="H/USDT",
                action="long",
                confidence=0.9,
                raw_llm_response=_execution_raw(contracts=100, contract_size=0.1),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="H/USDT",
                        side="buy",
                        order_type="market",
                        quantity=10.0,
                        price=2.44,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="entry-direct",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="H/USDT",
                        side="long",
                        quantity=10.0,
                        entry_price=2.44,
                        current_price=2.44,
                        leverage=3.0,
                        is_open=True,
                        okx_inst_id="H-USDT-SWAP",
                        entry_exchange_order_id="entry-direct",
                        created_at=filled_at + timedelta(minutes=30),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_flags_symbol_quantity_price_and_notional_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=50)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="WLFI/USDT",
                action="long",
                confidence=0.9,
                raw_llm_response=_execution_raw(contracts=100, contract_size=0.1, avg_price=2.44),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="WLFI/USDT",
                        side="buy",
                        order_type="market",
                        quantity=100.0,
                        price=2.31,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="entry-bad",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="H/USDT",
                        side="long",
                        quantity=10.0,
                        entry_price=2.44,
                        current_price=2.44,
                        leverage=3.0,
                        is_open=True,
                        created_at=filled_at + timedelta(seconds=30),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()
        kinds = {issue["kind"] for issue in report["issues"]}

        assert report["status"] == "critical"
        assert report["critical_count"] >= 2
        assert "symbol_alias_mismatch" in kinds
        assert "contract_base_quantity_mismatch" in kinds
        assert "execution_price_mismatch" in kinds
        assert "notional_mismatch" in kinds
        assert "order_position_symbol_mismatch" not in kinds
        symbol_issue = next(
            issue for issue in report["issues"] if issue["kind"] == "symbol_alias_mismatch"
        )
        assert symbol_issue["symbol"] == "WLFI/USDT"
        assert symbol_issue["expected_symbol"] == "H/USDT"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_position_alignment_trusts_native_inst_id_over_dirty_display_symbols(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=30)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="SAHARA/USDT",
                action="long",
                confidence=0.9,
                raw_llm_response=_execution_raw(
                    inst_id="SPK-USDT-SWAP",
                    contracts=10,
                    contract_size=1,
                    avg_price=0.111,
                ),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SAHARA/USDT",
                        side="buy",
                        order_type="market",
                        quantity=10.0,
                        price=0.111,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="spk-entry",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SAHARA/USDT",
                        side="long",
                        quantity=10.0,
                        entry_price=0.111,
                        current_price=0.111,
                        leverage=3.0,
                        is_open=True,
                        okx_inst_id="SPK-USDT-SWAP",
                        entry_exchange_order_id="spk-entry",
                        created_at=filled_at + timedelta(seconds=10),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()
        kinds = {issue["kind"] for issue in report["issues"]}

        assert "symbol_alias_mismatch" in kinds
        assert "position_okx_inst_id_symbol_mismatch" in kinds
        assert "order_position_symbol_mismatch" not in kinds
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_orphan_quarantine_marker_is_not_required_as_okx_order_link(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        closed_at = _recent_filled_at(minutes_ago=30)
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="ICP/USDT",
                        side="sell",
                        order_type="market",
                        quantity=6.02,
                        price=2.15,
                        status="filled",
                        exchange_order_id="icp-entry",
                        okx_inst_id="ICP-USDT-SWAP",
                        okx_sync_status="okx_confirmed",
                        filled_at=closed_at - timedelta(minutes=20),
                        created_at=closed_at - timedelta(minutes=20),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="ICP/USDT",
                        side="short",
                        quantity=6.02,
                        entry_price=2.15,
                        current_price=2.15,
                        realized_pnl=0.0,
                        unrealized_pnl=0.0,
                        leverage=3.0,
                        is_open=False,
                        okx_inst_id="ICP-USDT-SWAP",
                        entry_exchange_order_id="icp-entry",
                        close_exchange_order_id=f"{ORPHAN_QUARANTINE_EXCHANGE_ID_PREFIX}49",
                        closed_at=closed_at,
                        created_at=closed_at - timedelta(minutes=20),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()
        kinds = {issue["kind"] for issue in report["issues"]}

        assert "orphan_position_quarantine_not_exchange_backed" in kinds
        assert "position_order_link_missing_local_order" not in kinds
        assert report["status"] == "ok"
        assert report["critical_count"] == 0
        assert report["warning_count"] == 0
        assert {issue["severity"] for issue in report["issues"]} == {"info"}
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_position_alignment_flags_conflicting_native_inst_ids(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=30)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="SPK/USDT",
                action="long",
                confidence=0.9,
                raw_llm_response=_execution_raw(
                    inst_id="SPK-USDT-SWAP",
                    contracts=10,
                    contract_size=1,
                    avg_price=0.111,
                ),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="buy",
                        order_type="market",
                        quantity=10.0,
                        price=0.111,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="spk-entry",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SAHARA/USDT",
                        side="long",
                        quantity=10.0,
                        entry_price=0.111,
                        current_price=0.111,
                        leverage=3.0,
                        is_open=True,
                        okx_inst_id="SAHARA-USDT-SWAP",
                        entry_exchange_order_id="spk-entry",
                        created_at=filled_at + timedelta(seconds=10),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()
        mismatch = next(
            issue for issue in report["issues"] if issue["kind"] == "order_position_symbol_mismatch"
        )

        assert report["status"] == "critical"
        assert mismatch["symbol"] == "SAHARA/USDT"
        assert mismatch["expected_symbol"] == "SPK/USDT"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_filled_close_order_without_position_is_a_warning(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=45)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="MASK/USDT",
                action="close_long",
                confidence=0.9,
                raw_llm_response=_execution_raw(
                    inst_id="MASK-USDT-SWAP",
                    contracts=20,
                    contract_size=1,
                    avg_price=1.2,
                ),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="MASK/USDT",
                    side="sell",
                    order_type="market",
                    quantity=20.0,
                    price=1.2,
                    status="filled",
                    decision_id=decision.id,
                    exchange_order_id="close-missing-position",
                    filled_at=filled_at,
                    created_at=filled_at,
                )
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "warning"
        assert report["critical_count"] == 0
        assert report["warning_count"] == 1
        assert report["issues"][0]["kind"] == "order_position_missing"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_closed_position_with_realized_pnl_requires_close_order_link(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        closed_at = _recent_filled_at(minutes_ago=20)
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="LAB/USDT",
                    side="long",
                    quantity=0.9,
                    entry_price=16.86,
                    current_price=17.44,
                    realized_pnl=0.51,
                    leverage=3.0,
                    is_open=False,
                    okx_inst_id="LAB-USDT-SWAP",
                    entry_exchange_order_id="lab-entry",
                    closed_at=closed_at,
                    created_at=closed_at - timedelta(minutes=15),
                )
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()
        kinds = {issue["kind"] for issue in report["issues"]}

        assert report["status"] == "critical"
        assert "closed_position_missing_close_order_link" in kinds
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_manual_close_marker_is_not_treated_as_exchange_backed_fact(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        closed_at = _recent_filled_at(minutes_ago=20)
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="LAB/USDT",
                    side="long",
                    quantity=0.9,
                    entry_price=16.86,
                    current_price=17.44,
                    realized_pnl=0.51,
                    leverage=3.0,
                    is_open=False,
                    okx_inst_id="LAB-USDT-SWAP",
                    entry_exchange_order_id="lab-entry",
                    close_exchange_order_id="manual_close:local-only",
                    closed_at=closed_at,
                    created_at=closed_at - timedelta(minutes=15),
                )
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()
        kinds = {issue["kind"] for issue in report["issues"]}

        assert report["status"] == "critical"
        assert "manual_close_position_fact_not_exchange_backed" in kinds
        assert "closed_position_missing_close_order_link" not in kinds
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_old_linked_position_missing_local_order_is_info_observation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        closed_at = datetime.now(UTC) - timedelta(days=10)
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="PROS/USDT",
                    side="long",
                    quantity=1.0,
                    entry_price=0.4983,
                    current_price=0.3948,
                    realized_pnl=-0.1035,
                    leverage=3.0,
                    is_open=False,
                    okx_inst_id="PROS-USDT-SWAP",
                    entry_exchange_order_id="entry-old",
                    close_exchange_order_id="close-old-missing-local-order",
                    closed_at=closed_at,
                    created_at=closed_at - timedelta(hours=1),
                )
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24 * 14).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 2
        assert report["critical_count"] == 0
        assert report["warning_count"] == 0
        assert {issue["severity"] for issue in report["issues"]} == {"info"}
        assert {issue["kind"] for issue in report["issues"]} == {
            "position_order_link_missing_local_order"
        }
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_entry_order_matches_existing_position_lifecycle_without_false_warning(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=240)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="PROS/USDT",
                action="long",
                confidence=0.9,
                raw_llm_response=_execution_raw(
                    inst_id="PROS-USDT-SWAP",
                    contracts=1,
                    contract_size=1,
                    avg_price=0.3902,
                ),
                was_executed=False,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1.0,
                        price=0.3902,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="pros-add-fill",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT",
                        side="long",
                        quantity=1.0,
                        entry_price=0.3902,
                        current_price=0.3948,
                        realized_pnl=0.0,
                        leverage=3.0,
                        is_open=True,
                        okx_inst_id="PROS-USDT-SWAP",
                        entry_exchange_order_id="pros-add-fill",
                        created_at=filled_at - timedelta(days=10),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_matches_okx_authoritative_position_without_model_name_false_warning(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=45)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="XPL/USDT",
                action="short",
                confidence=0.9,
                raw_llm_response=_execution_raw(
                    inst_id="XPL-USDT-SWAP",
                    contracts=58,
                    contract_size=10,
                    avg_price=0.097,
                ),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="XPL/USDT",
                        side="sell",
                        order_type="market",
                        quantity=580.0,
                        price=0.097,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="xpl-entry-add",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="XPL/USDT",
                        side="short",
                        quantity=580.0,
                        entry_price=0.097,
                        current_price=0.096,
                        realized_pnl=0.0,
                        leverage=3.0,
                        is_open=True,
                        okx_inst_id="XPL-USDT-SWAP",
                        entry_exchange_order_id="xpl-entry-add",
                        created_at=filled_at,
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_superseded_position_residual_is_info_not_critical(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        opened_at = _recent_filled_at(minutes_ago=120)
        closed_at = _recent_filled_at(minutes_ago=20)
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="BNB/USDT",
                        side="short",
                        quantity=0.45,
                        entry_price=553.2911,
                        current_price=554.0,
                        realized_pnl=-0.5487,
                        leverage=3.0,
                        is_open=False,
                        okx_inst_id="BNB-USDT-SWAP",
                        okx_pos_id="bnb-pos-1",
                        entry_exchange_order_id="bnb-entry-a,bnb-entry-b",
                        close_exchange_order_id="bnb-close",
                        created_at=opened_at,
                        closed_at=closed_at,
                    ),
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="BNB/USDT",
                        side="short",
                        quantity=0.0,
                        entry_price=553.29,
                        current_price=553.7,
                        realized_pnl=0.0,
                        leverage=3.0,
                        is_open=False,
                        okx_inst_id="BNB-USDT-SWAP",
                        okx_pos_id="bnb-pos-1",
                        entry_exchange_order_id=None,
                        close_exchange_order_id=None,
                        created_at=opened_at,
                        closed_at=closed_at - timedelta(minutes=3),
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BNB/USDT",
                        side="sell",
                        order_type="market",
                        quantity=0.1,
                        price=553.0,
                        status="filled",
                        exchange_order_id="bnb-entry-a",
                        filled_at=opened_at,
                        created_at=opened_at,
                        okx_inst_id="BNB-USDT-SWAP",
                        okx_sync_status="okx_confirmed",
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BNB/USDT",
                        side="sell",
                        order_type="market",
                        quantity=0.35,
                        price=553.4,
                        status="filled",
                        exchange_order_id="bnb-entry-b",
                        filled_at=opened_at + timedelta(minutes=1),
                        created_at=opened_at + timedelta(minutes=1),
                        okx_inst_id="BNB-USDT-SWAP",
                        okx_sync_status="okx_confirmed",
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BNB/USDT",
                        side="buy",
                        order_type="market",
                        quantity=0.45,
                        price=554.0,
                        status="filled",
                        exchange_order_id="bnb-close",
                        filled_at=closed_at,
                        created_at=closed_at,
                        okx_inst_id="BNB-USDT-SWAP",
                        okx_sync_status="okx_confirmed",
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()
        issues = report["issues"]

        assert report["critical_count"] == 0
        assert report["warning_count"] == 0
        assert report["status"] == "ok"
        assert {issue["kind"] for issue in issues} == {"superseded_position_residual"}
        assert {issue["severity"] for issue in issues} == {"info"}
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_complete_lifecycle_supersedes_nonzero_stale_projection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        opened_at = _recent_filled_at(minutes_ago=120)
        official_closed_at = _recent_filled_at(minutes_ago=30)
        stale_closed_at = _recent_filled_at(minutes_ago=20)
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="GRASS/USDT",
                        side="short",
                        quantity=500.0,
                        entry_price=0.42,
                        current_price=0.40,
                        realized_pnl=5.39,
                        leverage=2.0,
                        is_open=False,
                        okx_inst_id="GRASS-USDT-SWAP",
                        okx_pos_id="grass-pos-1",
                        entry_exchange_order_id="grass-entry-a,grass-entry-b",
                        close_exchange_order_id="grass-close-a,grass-close-b",
                        created_at=opened_at,
                        closed_at=official_closed_at,
                    ),
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="GRASS/USDT",
                        side="short",
                        quantity=398.5,
                        entry_price=0.42,
                        current_price=0.40,
                        realized_pnl=4.25,
                        leverage=2.0,
                        is_open=False,
                        okx_inst_id="GRASS-USDT-SWAP",
                        okx_pos_id="grass-pos-1",
                        entry_exchange_order_id="grass-entry-a,grass-entry-b",
                        close_exchange_order_id=None,
                        created_at=opened_at,
                        closed_at=stale_closed_at,
                    ),
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="GRASS/USDT",
                        side="short",
                        quantity=100.0,
                        entry_price=0.42,
                        current_price=0.40,
                        realized_pnl=1.0,
                        leverage=2.0,
                        is_open=False,
                        okx_inst_id="GRASS-USDT-SWAP",
                        okx_pos_id="grass-pos-1",
                        entry_exchange_order_id=None,
                        close_exchange_order_id="grass-close-a",
                        created_at=opened_at,
                        closed_at=stale_closed_at,
                    ),
                    *[
                        Order(
                            model_name="ensemble_trader",
                            execution_mode="paper",
                            symbol="GRASS/USDT",
                            side=side,
                            order_type="market",
                            quantity=250.0,
                            price=price,
                            status="filled",
                            exchange_order_id=order_id,
                            filled_at=filled_at,
                            created_at=filled_at,
                            okx_inst_id="GRASS-USDT-SWAP",
                            okx_sync_status="okx_confirmed",
                        )
                        for order_id, side, price, filled_at in (
                            ("grass-entry-a", "sell", 0.42, opened_at),
                            (
                                "grass-entry-b",
                                "sell",
                                0.421,
                                opened_at + timedelta(minutes=1),
                            ),
                            (
                                "grass-close-a",
                                "buy",
                                0.405,
                                official_closed_at - timedelta(minutes=1),
                            ),
                            (
                                "grass-close-b",
                                "buy",
                                0.40,
                                official_closed_at,
                            ),
                        )
                    ],
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()
        issues = report["issues"]

        assert report["critical_count"] == 0
        assert report["status"] == "ok"
        assert [issue["kind"] for issue in issues].count(
            "superseded_position_residual"
        ) == 2
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_old_open_position_is_included_for_recent_entry_order_audit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=30)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="LINK/USDT",
                action="short",
                confidence=0.9,
                raw_llm_response=_execution_raw(
                    inst_id="LINK-USDT-SWAP",
                    contracts=2,
                    contract_size=1,
                    avg_price=7.22,
                ),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="LINK/USDT",
                        side="sell",
                        order_type="market",
                        quantity=2.0,
                        price=7.22,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="link-add-fill",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="LINK/USDT",
                        side="short",
                        quantity=2.0,
                        entry_price=7.22,
                        current_price=7.18,
                        leverage=3.0,
                        is_open=True,
                        created_at=filled_at - timedelta(days=10),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
        assert report["checked_positions"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_split_exit_order_uses_weighted_child_fill_price(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=40)
        weighted_price = (10 * 3.85 + 10 * 3.85 + 4 * 3.95) / 24
        raw_response = {
            "execution_result": {
                "price": weighted_price,
                "raw_response": {
                    "symbol": "USAR/USDT:USDT",
                    "okx_symbol": "USAR/USDT:USDT",
                    "average": 3.95,
                    "price": 3.95,
                    "filled": 4.0,
                    "amount": 4.0,
                    "contract_size": 1.0,
                    "filled_contracts": 24.0,
                    "base_quantity": 24.0,
                    "split_exit_order": True,
                    "split_chunks": [
                        {
                            "order_id": "child-1",
                            "closed_contracts": 10.0,
                            "price": 3.85,
                        },
                        {
                            "order_id": "child-2",
                            "closed_contracts": 10.0,
                            "price": 3.85,
                        },
                        {
                            "order_id": "child-3",
                            "closed_contracts": 4.0,
                            "price": 3.95,
                        },
                    ],
                },
            }
        }
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="USAR/USDT",
                action="close_long",
                confidence=0.9,
                raw_llm_response=raw_response,
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="USAR/USDT",
                        side="sell",
                        order_type="market",
                        quantity=24.0,
                        price=weighted_price,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="child-1,child-2,child-3",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="USAR/USDT",
                        side="long",
                        quantity=24.0,
                        entry_price=2.31,
                        current_price=weighted_price,
                        realized_pnl=37.19755,
                        leverage=6.0,
                        is_open=False,
                        okx_inst_id="USAR-USDT-SWAP",
                        close_exchange_order_id="child-1,child-2,child-3",
                        created_at=filled_at - timedelta(hours=5),
                        closed_at=filled_at + timedelta(seconds=1),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_matching_does_not_use_absolute_one_price_floor(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _reset_db(tmp_path, monkeypatch)
    try:
        filled_at = _recent_filled_at(minutes_ago=35)
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="H/USDT",
                action="short",
                confidence=0.9,
                raw_llm_response=_execution_raw(
                    inst_id="H-USDT-SWAP",
                    contracts=117.6,
                    contract_size=10.0,
                    avg_price=0.0817,
                ),
                was_executed=True,
                created_at=filled_at - timedelta(seconds=5),
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="H/USDT",
                        side="sell",
                        order_type="market",
                        quantity=1176.0,
                        price=0.0817,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="h-entry",
                        filled_at=filled_at,
                        created_at=filled_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="ALGO/USDT",
                        side="short",
                        quantity=1210.0,
                        entry_price=0.091,
                        current_price=0.091,
                        leverage=3.0,
                        is_open=True,
                        created_at=filled_at + timedelta(seconds=8),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="H/USDT",
                        side="short",
                        quantity=1176.0,
                        entry_price=0.0817,
                        current_price=0.0817,
                        leverage=3.0,
                        is_open=True,
                        okx_inst_id="H-USDT-SWAP",
                        entry_exchange_order_id="h-entry",
                        created_at=filled_at + timedelta(seconds=10),
                    ),
                ]
            )

        report = await OkxTradeFactIntegrityService(lookback_hours=24).audit()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
    finally:
        await close_db()
