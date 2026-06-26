from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Order, Position
from services.okx_trade_fact_integrity import OkxTradeFactIntegrityService


async def _reset_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'trade_fact.db').as_posix()}",
    )
    await init_db()


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
async def test_flags_symbol_quantity_price_notional_and_position_mismatch(
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
        assert report["critical_count"] >= 3
        assert "symbol_alias_mismatch" in kinds
        assert "contract_base_quantity_mismatch" in kinds
        assert "execution_price_mismatch" in kinds
        assert "notional_mismatch" in kinds
        assert "order_position_symbol_mismatch" in kinds
        symbol_issue = next(
            issue for issue in report["issues"] if issue["kind"] == "symbol_alias_mismatch"
        )
        assert symbol_issue["symbol"] == "WLFI/USDT"
        assert symbol_issue["expected_symbol"] == "H/USDT"
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
