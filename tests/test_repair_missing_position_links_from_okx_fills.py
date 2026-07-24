from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.learning import TradeReflection
from models.trade import Order, Position
from scripts import repair_missing_position_links_from_okx_fills as repair_script
from services.okx_order_fact_sync import OKX_SYNC_EXECUTION_RESULT_CONFIRMED


def test_additional_entry_link_requires_one_exact_open_lifecycle() -> None:
    filled_at = datetime(2026, 7, 11, 18, 48, tzinfo=UTC)
    order = SimpleNamespace(
        exchange_order_id="entry-add-1",
        filled_at=filled_at,
        created_at=filled_at,
        okx_inst_id="CELO-USDT-SWAP",
        symbol="CELO/USDT",
        side="buy",
        execution_mode="paper",
        quantity=605.0,
        okx_fill_contracts=605.0,
        price=0.0708,
        fee=-0.01,
        pnl=0.0,
        okx_raw_fills={"contract_size": 1.0},
    )
    decision = SimpleNamespace(action="long")
    open_position = SimpleNamespace(
        id=4373,
        symbol="CELO/USDT",
        okx_inst_id="CELO-USDT-SWAP",
        side="long",
        execution_mode="paper",
        quantity=713.0,
        entry_exchange_order_id="entry-1,entry-2",
        close_exchange_order_id="close-1",
        created_at=datetime(2026, 7, 11, 0, 21, tzinfo=UTC),
        closed_at=None,
    )
    closed_before_fill = SimpleNamespace(
        **{
            **open_position.__dict__,
            "id": 4674,
            "closed_at": datetime(2026, 7, 11, 11, 39, tzinfo=UTC),
        }
    )

    plan = repair_script._additional_entry_link_plan(
        order,
        decision,
        [open_position, closed_before_fill],
    )

    assert plan is not None
    assert plan.position_id == 4373
    assert plan.link_kind == "entry_add"
    assert plan.okx_order_id == "entry-add-1"
    assert plan.source == repair_script.ADDITIONAL_ENTRY_LINK_REPAIR_SOURCE


def test_additional_entry_link_rejects_ambiguous_lifecycles() -> None:
    filled_at = datetime(2026, 7, 11, 18, 48, tzinfo=UTC)
    order = SimpleNamespace(
        exchange_order_id="entry-add-1",
        filled_at=filled_at,
        created_at=filled_at,
        okx_inst_id="CELO-USDT-SWAP",
        symbol="CELO/USDT",
        side="buy",
        execution_mode="paper",
    )
    position = SimpleNamespace(
        id=1,
        symbol="CELO/USDT",
        okx_inst_id="CELO-USDT-SWAP",
        side="long",
        execution_mode="paper",
        quantity=1.0,
        entry_exchange_order_id="entry-1",
        close_exchange_order_id=None,
        created_at=datetime(2026, 7, 11, 0, 0, tzinfo=UTC),
        closed_at=None,
    )
    other = SimpleNamespace(**{**position.__dict__, "id": 2})

    assert (
        repair_script._additional_entry_link_plan(
            order,
            SimpleNamespace(action="long"),
            [position, other],
        )
        is None
    )


def test_missing_position_link_repair_imports_online_runtime_bootstrap() -> None:
    source = repair_script.ROOT.joinpath(
        "scripts",
        "repair_missing_position_links_from_okx_fills.py",
    ).read_text(encoding="utf-8")

    assert "from scripts.runtime_env_bootstrap import" in source
    assert "load_runtime_env_files(project_root=ROOT)" in source
    assert "drop_privileges_to_runtime_user_if_needed(project_root=ROOT)" in source


def test_missing_position_link_repair_redirects_probe_logs_from_stdout() -> None:
    source = repair_script.ROOT.joinpath(
        "scripts",
        "repair_missing_position_links_from_okx_fills.py",
    ).read_text(encoding="utf-8")

    assert "from contextlib import redirect_stdout" in source
    assert "with redirect_stdout(sys.stderr):" in source


@pytest.mark.asyncio
async def test_missing_position_link_apply_requires_position_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["repair_missing_position_links_from_okx_fills.py", "--apply"],
    )

    with pytest.raises(SystemExit):
        await repair_script.main()


@pytest.mark.asyncio
async def test_missing_position_link_dry_run_allows_unfiltered_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_collect_plans(**kwargs):
        captured["plans"] = kwargs
        return []

    monkeypatch.setattr(
        repair_script.settings.__class__,
        "get_okx_credentials",
        lambda _self, _mode: {"api_key": "demo"},
    )
    monkeypatch.setattr(repair_script, "collect_plans", fake_collect_plans)
    monkeypatch.setattr(
        sys,
        "argv",
        ["repair_missing_position_links_from_okx_fills.py", "--days", "3"],
    )

    result = await repair_script.main()

    assert result == 0
    assert captured["plans"]["days"] == 3
    assert captured["plans"]["position_ids"] == ()


@pytest.mark.asyncio
async def test_missing_position_link_apply_uses_position_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    plan = repair_script.FillLinkPlan(
        position_id=42,
        link_kind="close",
        symbol="BTC/USDT",
        side="long",
        quantity=1.0,
        okx_order_id="okx-42",
        old_entry_exchange_order_id="entry-42",
        old_close_exchange_order_id=None,
        old_okx_inst_id=None,
        fill_timestamp=datetime(2026, 6, 26, tzinfo=UTC),
        position_reference_time=datetime(2026, 6, 26, tzinfo=UTC),
        time_delta_seconds=1.0,
        fill_quantity=1.0,
        fill_contracts=1.0,
        fill_price=100.0,
        source="okx_fills_history",
        okx_inst_id="BTC-USDT-SWAP",
    )

    async def fake_collect_plans(**kwargs):
        assert kwargs["position_ids"] == (42,)
        return [plan]

    async def fake_apply_plans(plans):
        assert plans == [plan]
        calls.append("apply_links")
        return {"applied": 1, "backup_path": "backup.json"}

    monkeypatch.setattr(
        repair_script.settings.__class__,
        "get_okx_credentials",
        lambda _self, _mode: {"api_key": "demo"},
    )
    monkeypatch.setattr(repair_script, "collect_plans", fake_collect_plans)
    monkeypatch.setattr(repair_script, "apply_plans", fake_apply_plans)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repair_missing_position_links_from_okx_fills.py",
            "--apply",
            "--position-id",
            "42",
        ],
    )

    result = await repair_script.main()

    assert result == 0
    assert calls == ["apply_links"]


@pytest.mark.asyncio
async def test_missing_position_link_apply_marks_training_quarantine(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'position-link-apply.db').as_posix()}",
    )
    await init_db()
    try:
        closed_at = datetime(2026, 6, 26, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="HOME/USDT",
                side="long",
                quantity=12.0,
                entry_price=0.031,
                current_price=0.034,
                leverage=3.0,
                realized_pnl=0.036,
                is_open=False,
                closed_at=closed_at,
                created_at=closed_at,
                entry_exchange_order_id="entry-home",
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.FillLinkPlan(
            position_id=position_id,
            link_kind="close",
            symbol="HOME/USDT",
            side="long",
            quantity=12.0,
            okx_order_id="close-home",
            old_entry_exchange_order_id="entry-home",
            old_close_exchange_order_id=None,
            old_okx_inst_id=None,
            fill_timestamp=closed_at,
            position_reference_time=closed_at,
            time_delta_seconds=1.0,
            fill_quantity=12.0,
            fill_contracts=12.0,
            fill_price=0.034,
            source="okx_fills_history",
            okx_inst_id="HOME-USDT-SWAP",
        )

        monkeypatch.setattr(repair_script, "_backup", lambda _plans: _async_path(tmp_path))

        result = await repair_script.apply_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            order = (
                await session.execute(
                    Order.__table__.select().where(Order.exchange_order_id == "close-home")
                )
            ).mappings().one()
            reflection_rows = (
                await session.execute(
                    TradeReflection.__table__.select().where(
                        TradeReflection.position_id == position_id
                    )
                )
            ).mappings().all()
    finally:
        await close_db()

    assert result["applied"] == 1
    assert result["created_order_rows"] == 1
    assert position.close_exchange_order_id == "close-home"
    assert position.okx_inst_id == "HOME-USDT-SWAP"
    assert order["symbol"] == "HOME/USDT"
    assert order["side"] == "sell"
    assert order["quantity"] == pytest.approx(12.0)
    assert order["price"] == pytest.approx(0.034)
    assert len(reflection_rows) == 1
    assert reflection_rows[0]["source"] == repair_script.REPAIR_REFLECTION_SOURCE
    assert reflection_rows[0]["expert_lessons"]["training_policy"] == "exclude_until_manual_trust"


def test_exchange_order_id_split_preserves_first_seen_order() -> None:
    assert repair_script._split_exchange_order_ids(
        "entry-1,entry-2;entry-1|entry-3"
    ) == ["entry-1", "entry-2", "entry-3"]


@pytest.mark.asyncio
async def test_additional_entry_apply_appends_without_overwriting_existing_links(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'additional-entry-link.db').as_posix()}",
    )
    await init_db()
    try:
        filled_at = datetime(2026, 7, 11, 18, 48, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="okx_authoritative_sync",
                execution_mode="paper",
                symbol="CELO/USDT",
                side="long",
                quantity=713.0,
                entry_price=0.0688,
                current_price=0.0708,
                leverage=1.0,
                is_open=True,
                created_at=datetime(2026, 7, 11, 0, 21, tzinfo=UTC),
                entry_exchange_order_id="entry-1,entry-2",
                close_exchange_order_id="close-1",
                okx_inst_id="CELO-USDT-SWAP",
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.FillLinkPlan(
            position_id=position_id,
            link_kind="entry_add",
            symbol="CELO/USDT",
            side="long",
            quantity=713.0,
            okx_order_id="entry-add-1",
            old_entry_exchange_order_id="entry-1,entry-2",
            old_close_exchange_order_id="close-1",
            old_okx_inst_id="CELO-USDT-SWAP",
            fill_timestamp=filled_at,
            position_reference_time=datetime(2026, 7, 11, 0, 21, tzinfo=UTC),
            time_delta_seconds=66_420.0,
            fill_quantity=605.0,
            fill_contracts=605.0,
            fill_price=0.0708,
            source=repair_script.ADDITIONAL_ENTRY_LINK_REPAIR_SOURCE,
            okx_inst_id="CELO-USDT-SWAP",
        )
        monkeypatch.setattr(repair_script, "_backup", lambda _plans: _async_path(tmp_path))

        result = await repair_script.apply_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            reflections = (
                await session.execute(
                    select(func.count(TradeReflection.id)).where(
                        TradeReflection.position_id == position_id
                    )
                )
            ).scalar_one()
    finally:
        await close_db()

    assert result["applied"] == 1
    assert result["created_order_rows"] == 0
    assert position.entry_exchange_order_id == "entry-1,entry-2,entry-add-1"
    assert position.close_exchange_order_id == "close-1"
    assert reflections == 1


@pytest.mark.asyncio
async def test_linked_protection_fill_order_apply_creates_order_without_closing_position(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'linked-protection-fill-order.db').as_posix()}",
    )
    await init_db()
    try:
        filled_at = datetime(2026, 6, 28, 1, 8, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="AAVE/USDT",
                side="short",
                quantity=0.22,
                entry_price=94.51,
                current_price=93.83,
                leverage=2.0,
                realized_pnl=0.0,
                is_open=True,
                okx_inst_id="AAVE-USDT-SWAP",
                entry_exchange_order_id="aave-entry-order",
                created_at=filled_at,
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.LinkedProtectionFillOrderPlan(
            local_entry_order_id=68,
            linked_exchange_order_id="aave-entry-order",
            symbol="AAVE/USDT",
            side="buy",
            model_name="ensemble_trader",
            execution_mode="paper",
            exchange_order_id="aave-protection-close",
            quantity=0.61,
            price=97.54,
            fee=0.029,
            filled_at=filled_at,
            source="okx_linked_protection_fill_missing_local_order",
            okx_inst_id="AAVE-USDT-SWAP",
            okx_algo_id="aave-oco-triggered",
            okx_source="7",
            decision_id=1717,
        )

        monkeypatch.setattr(
            repair_script,
            "_backup_linked_protection_fill_orders",
            lambda _plans: _async_path(tmp_path),
        )

        result = await repair_script.apply_linked_protection_fill_order_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            order = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.exchange_order_id == "aave-protection-close"
                    )
                )
            ).mappings().one()
            reflection_rows = (
                await session.execute(
                    TradeReflection.__table__.select().where(
                        TradeReflection.position_id == position_id
                    )
                )
            ).mappings().all()
    finally:
        await close_db()

    assert result["applied"] == 1
    assert position.is_open is True
    assert position.close_exchange_order_id is None
    assert order["symbol"] == "AAVE/USDT"
    assert order["side"] == "buy"
    assert order["quantity"] == pytest.approx(0.61)
    assert order["decision_id"] == 1717
    assert len(reflection_rows) == 1
    assert reflection_rows[0]["expert_lessons"]["repair_plan"]["okx_algo_id"] == "aave-oco-triggered"


@pytest.mark.asyncio
async def test_collect_existing_order_decision_link_plans_requires_unique_entry_decision(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'existing-order-decision-link.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 28, 23, 2, 54, tzinfo=UTC)
        closed_at = datetime(2026, 6, 29, 0, 30, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="okx_authoritative_sync",
                execution_mode="paper",
                symbol="AVAX/USDT",
                side="short",
                quantity=0.5,
                entry_price=12.0,
                current_price=12.2,
                leverage=3.0,
                realized_pnl=-0.1,
                is_open=False,
                created_at=opened_at,
                closed_at=closed_at,
                okx_inst_id="AVAX-USDT-SWAP",
                entry_exchange_order_id="entry-avax",
                close_exchange_order_id="close-avax",
            )
            session.add(position)
            session.add(
                Order(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="AVAX/USDT",
                    side="sell",
                    order_type="market",
                    quantity=0.5,
                    price=12.0,
                    status="filled",
                    fee=0.01,
                    decision_id=None,
                    exchange_order_id="entry-avax",
                    filled_at=opened_at,
                    created_at=opened_at,
                    okx_inst_id="AVAX-USDT-SWAP",
                )
            )
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="AVAX/USDT",
                    action="short",
                    confidence=0.8,
                    position_size_pct=2.0,
                    suggested_leverage=3.0,
                    is_paper=True,
                    was_executed=True,
                    executed_at=opened_at,
                    execution_price=12.0,
                    created_at=opened_at,
                    raw_llm_response={"opportunity_score": {"ml_aligned": True}},
                )
            )
            await session.flush()
            position_id = int(position.id)

        plans = await repair_script.collect_existing_order_decision_link_plans(
            days=30,
            decision_window_seconds=120,
            position_ids=(position_id,),
        )
    finally:
        await close_db()

    assert len(plans) == 1
    assert plans[0].exchange_order_id == "entry-avax"
    assert plans[0].decision_id > 0
    assert plans[0].side == "sell"


@pytest.mark.asyncio
async def test_collect_existing_order_decision_link_plans_uses_decision_only_once(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'existing-order-decision-once.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 28, 23, 2, 54, tzinfo=UTC)
        closed_at = datetime(2026, 6, 29, 0, 30, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="okx_authoritative_sync",
                execution_mode="paper",
                symbol="AVAX/USDT",
                side="short",
                quantity=1.0,
                entry_price=12.0,
                current_price=12.2,
                leverage=3.0,
                realized_pnl=-0.1,
                is_open=False,
                created_at=opened_at,
                closed_at=closed_at,
                okx_inst_id="AVAX-USDT-SWAP",
                entry_exchange_order_id="entry-near,entry-far",
                close_exchange_order_id="close-avax",
            )
            session.add(position)
            for exchange_id, offset in (("entry-near", 0), ("entry-far", 40)):
                filled_at = opened_at + timedelta(seconds=offset)
                session.add(
                    Order(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="AVAX/USDT",
                        side="sell",
                        order_type="market",
                        quantity=0.5,
                        price=12.0,
                        status="filled",
                        fee=0.01,
                        decision_id=None,
                        exchange_order_id=exchange_id,
                        filled_at=filled_at,
                        created_at=filled_at,
                        okx_inst_id="AVAX-USDT-SWAP",
                    )
                )
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="AVAX/USDT",
                    action="short",
                    confidence=0.8,
                    position_size_pct=2.0,
                    suggested_leverage=3.0,
                    is_paper=True,
                    was_executed=True,
                    executed_at=opened_at + timedelta(seconds=2),
                    execution_price=12.0,
                    created_at=opened_at + timedelta(seconds=2),
                )
            )
            await session.flush()
            position_id = int(position.id)

        plans = await repair_script.collect_existing_order_decision_link_plans(
            days=30,
            decision_window_seconds=120,
            position_ids=(position_id,),
        )
    finally:
        await close_db()

    assert [plan.exchange_order_id for plan in plans] == ["entry-near"]


@pytest.mark.asyncio
async def test_collect_existing_order_decision_link_plans_respects_exchange_order_filter(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'existing-order-decision-filter.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 28, 23, 2, 54, tzinfo=UTC)
        closed_at = datetime(2026, 6, 29, 0, 30, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="okx_authoritative_sync",
                execution_mode="paper",
                symbol="AVAX/USDT",
                side="short",
                quantity=1.0,
                entry_price=12.0,
                current_price=12.2,
                leverage=3.0,
                realized_pnl=-0.1,
                is_open=False,
                created_at=opened_at,
                closed_at=closed_at,
                okx_inst_id="AVAX-USDT-SWAP",
                entry_exchange_order_id="entry-avax-1,entry-avax-2",
                close_exchange_order_id="close-avax",
            )
            session.add(position)
            for exchange_id in ("entry-avax-1", "entry-avax-2"):
                session.add(
                    Order(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="AVAX/USDT",
                        side="sell",
                        order_type="market",
                        quantity=0.5,
                        price=12.0,
                        status="filled",
                        fee=0.01,
                        decision_id=None,
                        exchange_order_id=exchange_id,
                        filled_at=opened_at,
                        created_at=opened_at,
                        okx_inst_id="AVAX-USDT-SWAP",
                    )
                )
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="AVAX/USDT",
                    action="short",
                    confidence=0.8,
                    position_size_pct=2.0,
                    suggested_leverage=3.0,
                    is_paper=True,
                    was_executed=True,
                    executed_at=opened_at,
                    execution_price=12.0,
                    created_at=opened_at,
                )
            )
            await session.flush()
            position_id = int(position.id)

        plans = await repair_script.collect_existing_order_decision_link_plans(
            days=30,
            decision_window_seconds=120,
            position_ids=(position_id,),
            exchange_order_ids=("entry-avax-2",),
        )
    finally:
        await close_db()

    assert [plan.exchange_order_id for plan in plans] == ["entry-avax-2"]


@pytest.mark.asyncio
async def test_collect_existing_order_decision_link_plans_skips_ambiguous_decisions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'existing-order-decision-ambiguous.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 28, 23, 2, 54, tzinfo=UTC)
        closed_at = datetime(2026, 6, 29, 0, 30, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="okx_authoritative_sync",
                execution_mode="paper",
                symbol="AVAX/USDT",
                side="short",
                quantity=0.5,
                entry_price=12.0,
                current_price=12.2,
                leverage=3.0,
                realized_pnl=-0.1,
                is_open=False,
                created_at=opened_at,
                closed_at=closed_at,
                okx_inst_id="AVAX-USDT-SWAP",
                entry_exchange_order_id="entry-avax",
                close_exchange_order_id="close-avax",
            )
            session.add(position)
            session.add(
                Order(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="AVAX/USDT",
                    side="sell",
                    order_type="market",
                    quantity=0.5,
                    price=12.0,
                    status="filled",
                    fee=0.01,
                    decision_id=None,
                    exchange_order_id="entry-avax",
                    filled_at=opened_at,
                    created_at=opened_at,
                    okx_inst_id="AVAX-USDT-SWAP",
                )
            )
            for offset in (0, 10):
                session.add(
                    AIDecision(
                        model_name="ensemble_trader",
                        symbol="AVAX/USDT",
                        action="short",
                        confidence=0.8,
                        position_size_pct=2.0,
                        suggested_leverage=3.0,
                        is_paper=True,
                        was_executed=True,
                        executed_at=opened_at.replace(second=offset),
                        execution_price=12.0,
                        created_at=opened_at.replace(second=offset),
                    )
                )
            await session.flush()
            position_id = int(position.id)

        plans = await repair_script.collect_existing_order_decision_link_plans(
            days=5,
            decision_window_seconds=120,
            position_ids=(position_id,),
        )
    finally:
        await close_db()

    assert plans == []


@pytest.mark.asyncio
async def test_apply_existing_order_decision_link_updates_order_only(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'apply-existing-order-decision-link.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 28, 23, 2, 54, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="okx_authoritative_sync",
                execution_mode="paper",
                symbol="AVAX/USDT",
                side="short",
                quantity=0.5,
                entry_price=12.0,
                current_price=12.2,
                leverage=3.0,
                realized_pnl=-0.1,
                is_open=False,
                created_at=opened_at,
                closed_at=opened_at,
                okx_inst_id="AVAX-USDT-SWAP",
                entry_exchange_order_id="entry-avax",
                close_exchange_order_id="close-avax",
            )
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="AVAX/USDT",
                action="short",
                confidence=0.8,
                position_size_pct=2.0,
                suggested_leverage=3.0,
                is_paper=True,
                was_executed=True,
                executed_at=opened_at,
                execution_price=12.0,
                created_at=opened_at,
            )
            order = Order(
                model_name="okx_authoritative_sync",
                execution_mode="paper",
                symbol="AVAX/USDT",
                side="sell",
                order_type="market",
                quantity=0.5,
                price=12.0,
                status="filled",
                fee=0.01,
                decision_id=None,
                exchange_order_id="entry-avax",
                filled_at=opened_at,
                created_at=opened_at,
                okx_inst_id="AVAX-USDT-SWAP",
            )
            session.add_all([position, decision, order])
            await session.flush()
            position_id = int(position.id)
            decision_id = int(decision.id)
            order_id = int(order.id)

        plan = repair_script.ExistingOrderDecisionLinkPlan(
            position_id=position_id,
            order_id=order_id,
            exchange_order_id="entry-avax",
            symbol="AVAX/USDT",
            side="sell",
            model_name="okx_authoritative_sync",
            execution_mode="paper",
            decision_id=decision_id,
            decision_symbol="AVAX/USDT",
            decision_action="short",
            order_filled_at=opened_at,
            decision_executed_at=opened_at,
            position_created_at=opened_at,
            order_decision_delta_seconds=0.0,
            position_order_delta_seconds=0.0,
        )
        monkeypatch.setattr(
            repair_script,
            "_backup_existing_order_decision_links",
            lambda _plans: _async_path(tmp_path),
        )

        result = await repair_script.apply_existing_order_decision_link_plans([plan])

        async with get_session_ctx() as session:
            order_row = await session.get(Order, order_id)
            position_row = await session.get(Position, position_id)
    finally:
        await close_db()

    assert result["applied"] == 1
    assert order_row.decision_id == decision_id
    assert position_row.entry_exchange_order_id == "entry-avax"


@pytest.mark.asyncio
async def test_missing_position_link_apply_requires_native_okx_inst_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'position-link-native-required.db').as_posix()}",
    )
    await init_db()
    try:
        closed_at = datetime(2026, 6, 26, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="HOME/USDT",
                side="long",
                quantity=12.0,
                entry_price=0.031,
                current_price=0.034,
                leverage=3.0,
                realized_pnl=0.036,
                is_open=False,
                closed_at=closed_at,
                created_at=closed_at,
                entry_exchange_order_id="entry-home",
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.FillLinkPlan(
            position_id=position_id,
            link_kind="close",
            symbol="HOME/USDT",
            side="long",
            quantity=12.0,
            okx_order_id="close-home",
            old_entry_exchange_order_id="entry-home",
            old_close_exchange_order_id=None,
            old_okx_inst_id=None,
            fill_timestamp=closed_at,
            position_reference_time=closed_at,
            time_delta_seconds=1.0,
            fill_quantity=12.0,
            fill_contracts=12.0,
            fill_price=0.034,
            source="okx_fills_history_without_native_inst_id",
        )

        monkeypatch.setattr(repair_script, "_backup", lambda _plans: _async_path(tmp_path))

        result = await repair_script.apply_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
    finally:
        await close_db()

    assert result["applied"] == 0
    assert position.close_exchange_order_id is None
    assert position.okx_inst_id is None


@pytest.mark.asyncio
async def test_missing_position_link_apply_rejects_inst_id_conflict(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'position-link-native-conflict.db').as_posix()}",
    )
    await init_db()
    try:
        closed_at = datetime(2026, 6, 26, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="SPK/USDT",
                side="long",
                quantity=12.0,
                entry_price=0.031,
                current_price=0.034,
                leverage=3.0,
                realized_pnl=0.036,
                is_open=False,
                closed_at=closed_at,
                created_at=closed_at,
                okx_inst_id="SPK-USDT-SWAP",
                entry_exchange_order_id="entry-spk",
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.FillLinkPlan(
            position_id=position_id,
            link_kind="close",
            symbol="SAHARA/USDT",
            side="long",
            quantity=12.0,
            okx_order_id="close-sahara",
            old_entry_exchange_order_id="entry-spk",
            old_close_exchange_order_id=None,
            old_okx_inst_id="SPK-USDT-SWAP",
            fill_timestamp=closed_at,
            position_reference_time=closed_at,
            time_delta_seconds=1.0,
            fill_quantity=12.0,
            fill_contracts=12.0,
            fill_price=0.034,
            source="okx_fills_history",
            okx_inst_id="SAHARA-USDT-SWAP",
        )

        monkeypatch.setattr(repair_script, "_backup", lambda _plans: _async_path(tmp_path))

        result = await repair_script.apply_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
    finally:
        await close_db()

    assert result["applied"] == 0
    assert position.close_exchange_order_id is None
    assert position.okx_inst_id == "SPK-USDT-SWAP"


def test_open_position_close_plan_uses_okx_contract_size_for_quantity() -> None:
    opened_at = datetime(2026, 6, 27, 18, 0, tzinfo=UTC)
    fill_at = datetime(2026, 6, 27, 18, 16, 15, tzinfo=UTC)
    position = Position(
        id=29,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="MET/USDT",
        side="short",
        quantity=20.0,
        entry_price=0.17245,
        current_price=0.1733,
        leverage=3.0,
        realized_pnl=0.0,
        is_open=True,
        created_at=opened_at,
        okx_inst_id="MET-USDT-SWAP",
        entry_exchange_order_id="met-entry",
    )
    fill = repair_script.FillGroup(
        order_id="3693731464371474432",
        symbol="MET/USDT",
        side="buy",
        avg_price=0.1749,
        contracts=2.0,
        fill_pnl=-0.049,
        fee_abs=0.001,
        timestamp=fill_at,
        timestamp_ms=fill_at.timestamp() * 1000,
        rows=[],
        inst_id="MET-USDT-SWAP",
    )

    plan = repair_script._match_open_position_close_plan(
        position,
        [fill],
        existing_exchange_ids=set(),
        contract_sizes={"MET-USDT-SWAP": 10.0},
        window_seconds=1800,
    )

    assert plan is not None
    assert plan.position_id == 29
    assert plan.okx_order_id == "3693731464371474432"
    assert plan.close_side == "buy"
    assert plan.fill_contracts == pytest.approx(2.0)
    assert plan.contract_size == pytest.approx(10.0)
    assert plan.fill_quantity == pytest.approx(20.0)
    assert plan.fill_pnl == pytest.approx(-0.049)
    assert plan.contract_size_source == "okx_public_instruments"


def test_close_link_reassignment_plan_switches_to_quantity_matching_fill() -> None:
    closed_at = datetime(2026, 6, 27, 18, 58, 47, tzinfo=UTC)
    position = Position(
        id=13,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="LIT/USDT",
        side="short",
        quantity=10.0,
        entry_price=1.819,
        current_price=1.929,
        leverage=3.0,
        realized_pnl=-1.13803,
        is_open=False,
        closed_at=closed_at,
        created_at=datetime(2026, 6, 27, 16, 58, tzinfo=UTC),
        okx_inst_id="LIT-USDT-SWAP",
        close_exchange_order_id="3693817049043931138",
    )
    fills = [
        repair_script.FillGroup(
            order_id="3693817049043931138",
            symbol="LIT/USDT",
            side="buy",
            avg_price=1.929,
            contracts=3.0,
            fill_pnl=-4.11428571,
            fee_abs=0.028935,
            timestamp=closed_at,
            timestamp_ms=closed_at.timestamp() * 1000,
            rows=[],
            inst_id="LIT-USDT-SWAP",
        ),
        repair_script.FillGroup(
            order_id="3693817049043931137",
            symbol="LIT/USDT",
            side="buy",
            avg_price=1.929,
            contracts=1.0,
            fill_pnl=-1.37142857,
            fee_abs=0.009645,
            timestamp=closed_at,
            timestamp_ms=closed_at.timestamp() * 1000,
            rows=[],
            inst_id="LIT-USDT-SWAP",
        ),
    ]

    plan = repair_script._match_close_link_reassignment_plan(
        position,
        fills,
        existing_exchange_ids={"3693817049043931138"},
        contract_sizes={"LIT-USDT-SWAP": 10.0},
        window_seconds=300,
    )

    assert plan is not None
    assert plan.old_okx_order_id == "3693817049043931138"
    assert plan.new_okx_order_id == "3693817049043931137"
    assert plan.old_fill_quantity == pytest.approx(30.0)
    assert plan.new_fill_quantity == pytest.approx(10.0)
    assert plan.fill_pnl == pytest.approx(-1.37142857)


def test_open_position_close_plan_skips_already_allocated_fill() -> None:
    fill_at = datetime(2026, 6, 27, 18, 58, 47, tzinfo=UTC)
    fills = [
        repair_script.FillGroup(
            order_id="3693817049043931138",
            symbol="LIT/USDT",
            side="buy",
            avg_price=1.929,
            contracts=3.0,
            fill_pnl=-4.11428571,
            fee_abs=0.028935,
            timestamp=fill_at,
            timestamp_ms=fill_at.timestamp() * 1000,
            rows=[],
            inst_id="LIT-USDT-SWAP",
        ),
        repair_script.FillGroup(
            order_id="3693817049043931136",
            symbol="LIT/USDT",
            side="buy",
            avg_price=1.929,
            contracts=3.0,
            fill_pnl=-4.11428571,
            fee_abs=0.028935,
            timestamp=fill_at,
            timestamp_ms=fill_at.timestamp() * 1000,
            rows=[],
            inst_id="LIT-USDT-SWAP",
        ),
    ]
    position = Position(
        id=17,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="LIT/USDT",
        side="short",
        quantity=30.0,
        entry_price=1.796,
        current_price=1.7845,
        leverage=3.0,
        is_open=True,
        created_at=datetime(2026, 6, 27, 17, 9, tzinfo=UTC),
        okx_inst_id="LIT-USDT-SWAP",
    )

    plan = repair_script._match_open_position_close_plan(
        position,
        fills,
        existing_exchange_ids={"3693817049043931138"},
        contract_sizes={"LIT-USDT-SWAP": 10.0},
        window_seconds=7200,
    )

    assert plan is not None
    assert plan.okx_order_id == "3693817049043931136"
    assert plan.fill_quantity == pytest.approx(30.0)


def test_existing_okx_confirmed_close_order_closes_open_position_plan() -> None:
    opened_at = datetime(2026, 6, 28, 5, 0, tzinfo=UTC)
    closed_at = datetime(2026, 6, 28, 7, 46, tzinfo=UTC)
    position = Position(
        id=65,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="short",
        quantity=1270.0,
        entry_price=0.01049,
        current_price=0.009077,
        leverage=3.0,
        is_open=True,
        created_at=opened_at,
        okx_inst_id="ACT-USDT-SWAP",
        entry_exchange_order_id="act-entry",
    )
    close_order = Order(
        id=97,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=1270.0,
        price=0.00908,
        status="filled",
        fee=0.0057658,
        decision_id=None,
        exchange_order_id="act-close",
        filled_at=closed_at,
        created_at=closed_at,
        okx_inst_id="ACT-USDT-SWAP",
        okx_fill_contracts=127.0,
        okx_fill_pnl=1.7907,
        okx_sync_status="okx_confirmed",
    )

    plan = repair_script._match_existing_close_order_open_position_plan(
        position,
        [close_order],
        contract_sizes={"ACT-USDT-SWAP": 10.0},
    )

    assert plan is not None
    assert plan.position_id == 65
    assert plan.okx_order_id == "act-close"
    assert plan.close_side == "buy"
    assert plan.fill_quantity == pytest.approx(1270.0)
    assert plan.fill_contracts == pytest.approx(127.0)
    assert plan.contract_size == pytest.approx(10.0)
    assert plan.source == "okx_confirmed_existing_close_order"
    assert plan.fill_pnl == pytest.approx(1.7907)


def test_existing_close_order_plan_requires_okx_confirmed_status() -> None:
    opened_at = datetime(2026, 6, 28, 5, 0, tzinfo=UTC)
    closed_at = datetime(2026, 6, 28, 7, 46, tzinfo=UTC)
    position = Position(
        id=65,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="short",
        quantity=1270.0,
        entry_price=0.01049,
        is_open=True,
        created_at=opened_at,
        okx_inst_id="ACT-USDT-SWAP",
        entry_exchange_order_id="act-entry",
    )
    close_order = Order(
        id=97,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=1270.0,
        price=0.00908,
        status="filled",
        fee=0.0057658,
        exchange_order_id="act-close",
        filled_at=closed_at,
        created_at=closed_at,
        okx_inst_id="ACT-USDT-SWAP",
        okx_fill_contracts=127.0,
        okx_fill_pnl=1.7907,
        okx_sync_status=None,
    )

    plan = repair_script._match_existing_close_order_open_position_plan(
        position,
        [close_order],
        contract_sizes={"ACT-USDT-SWAP": 10.0},
    )

    assert plan is None


def test_existing_execution_result_confirmed_close_order_closes_open_position_plan() -> None:
    opened_at = datetime(2026, 7, 1, 6, 20, tzinfo=UTC)
    closed_at = datetime(2026, 7, 1, 6, 46, tzinfo=UTC)
    position = Position(
        id=65,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="short",
        quantity=1580.0,
        entry_price=0.00999,
        current_price=0.0097,
        leverage=3.0,
        is_open=True,
        created_at=opened_at,
        okx_inst_id="ACT-USDT-SWAP",
        entry_exchange_order_id="act-entry",
    )
    close_order = Order(
        id=97,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ACT/USDT",
        side="buy",
        order_type="market",
        quantity=1580.0,
        price=0.0097,
        status="filled",
        fee=0.007663,
        exchange_order_id="3703940352525967360",
        filled_at=closed_at,
        created_at=closed_at,
        okx_inst_id="ACT-USDT-SWAP",
        okx_fill_contracts=158.0,
        okx_fill_pnl=0.4582,
        okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
    )

    plan = repair_script._match_existing_close_order_open_position_plan(
        position,
        [close_order],
        contract_sizes={"ACT-USDT-SWAP": 10.0},
    )

    assert plan is not None
    assert plan.okx_order_id == "3703940352525967360"
    assert plan.fill_quantity == pytest.approx(1580.0)
    assert plan.fill_pnl == pytest.approx(0.4582)


def test_native_full_close_shared_plan_matches_split_positions() -> None:
    closed_at = datetime(2026, 6, 27, 19, 10, 45, tzinfo=UTC)
    positions = [
        Position(
            id=33,
            model_name="ensemble_trader",
            execution_mode="paper",
            symbol="SUSHI/USDT",
            side="long",
            quantity=67.5,
            entry_price=0.1588,
            current_price=0.156,
            leverage=3.0,
            realized_pnl=-0.1943595,
            is_open=False,
            closed_at=closed_at,
            created_at=datetime(2026, 6, 27, 18, 13, 40, tzinfo=UTC),
            okx_inst_id="SUSHI-USDT-SWAP",
            entry_exchange_order_id="3693726686522347520",
            close_exchange_order_id="okx_native_full_close",
        ),
        Position(
            id=34,
            model_name="ensemble_trader",
            execution_mode="paper",
            symbol="SUSHI/USDT",
            side="long",
            quantity=67.5,
            entry_price=0.1588,
            current_price=0.156,
            leverage=3.0,
            realized_pnl=-0.1943595,
            is_open=False,
            closed_at=closed_at,
            created_at=datetime(2026, 6, 27, 18, 13, 40, tzinfo=UTC),
            okx_inst_id="SUSHI-USDT-SWAP",
            entry_exchange_order_id="3693726686522347520",
            close_exchange_order_id="okx_native_full_close",
        ),
    ]
    order = Order(
        id=45,
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="SUSHI/USDT",
        side="sell",
        order_type="market",
        quantity=135.0,
        price=0.156,
        status="filled",
        fee=0.0,
        exchange_order_id=None,
        filled_at=closed_at,
        created_at=closed_at,
    )
    fill_at = datetime(2026, 6, 27, 19, 11, 1, tzinfo=UTC)
    fill = repair_script.FillGroup(
        order_id="3693841704639238144",
        symbol="SUSHI/USDT",
        side="sell",
        avg_price=0.15541926,
        contracts=135.0,
        fill_pnl=-0.4564,
        fee_abs=0.0104908,
        timestamp=fill_at,
        timestamp_ms=fill_at.timestamp() * 1000,
        rows=[],
        inst_id="SUSHI-USDT-SWAP",
    )

    plan = repair_script._match_native_full_close_shared_plan(
        positions,
        [order],
        [fill],
        contract_sizes={"SUSHI-USDT-SWAP": 1.0},
        window_seconds=1800,
    )

    assert plan is not None
    assert plan.position_ids == (33, 34)
    assert plan.close_order_id == 45
    assert plan.okx_order_id == "3693841704639238144"
    assert plan.total_quantity == pytest.approx(135.0)
    assert plan.fill_quantity == pytest.approx(135.0)
    assert plan.close_fee == pytest.approx(0.0104908)
    assert plan.fill_pnl == pytest.approx(-0.4564)


def test_native_full_close_group_allows_multiple_entry_orders() -> None:
    closed_at = datetime(2026, 6, 27, 19, 26, 18, tzinfo=UTC)
    positions = [
        Position(
            id=3,
            model_name="ensemble_trader",
            execution_mode="paper",
            symbol="JUP/USDT",
            side="long",
            quantity=130.0,
            entry_price=0.2305,
            current_price=0.2255,
            leverage=2.0,
            is_open=False,
            closed_at=closed_at,
            created_at=datetime(2026, 6, 27, 16, 41, tzinfo=UTC),
            okx_inst_id="JUP-USDT-SWAP",
            entry_exchange_order_id="entry-jup-1",
            close_exchange_order_id="okx_native_full_close",
        ),
        Position(
            id=16,
            model_name="ensemble_trader",
            execution_mode="paper",
            symbol="JUP/USDT",
            side="long",
            quantity=100.0,
            entry_price=0.2297,
            current_price=0.2255,
            leverage=2.0,
            is_open=False,
            closed_at=closed_at,
            created_at=datetime(2026, 6, 27, 17, 4, tzinfo=UTC),
            okx_inst_id="JUP-USDT-SWAP",
            entry_exchange_order_id="entry-jup-2",
            close_exchange_order_id="okx_native_full_close",
        ),
    ]

    groups = repair_script._group_native_full_close_positions(positions)

    assert len(groups) == 1
    assert [int(position.id) for position in groups[0]] == [3, 16]


@pytest.mark.asyncio
async def test_apply_open_position_close_plan_closes_position_and_quarantines_training(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'open-position-close.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 27, 18, 0, tzinfo=UTC)
        closed_at = datetime(2026, 6, 27, 18, 16, 15, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="MET/USDT",
                side="short",
                quantity=20.0,
                entry_price=0.17245,
                current_price=0.1733,
                leverage=3.0,
                realized_pnl=0.0,
                is_open=True,
                created_at=opened_at,
                okx_inst_id="MET-USDT-SWAP",
                entry_exchange_order_id="met-entry",
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.OpenPositionClosePlan(
            position_id=position_id,
            symbol="MET/USDT",
            side="short",
            close_side="buy",
            model_name="ensemble_trader",
            execution_mode="paper",
            okx_order_id="met-close",
            old_is_open=True,
            old_close_exchange_order_id=None,
            old_okx_inst_id="MET-USDT-SWAP",
            quantity=20.0,
            fill_quantity=20.0,
            fill_contracts=2.0,
            contract_size=10.0,
            contract_size_source="okx_public_instruments",
            entry_price=0.17245,
            exit_price=0.1749,
            close_fee=0.001,
            fill_pnl=-0.049,
            computed_realized_pnl=-0.05,
            old_realized_pnl=0.0,
            old_current_price=0.1733,
            fill_timestamp=closed_at,
            position_reference_time=opened_at,
            time_delta_seconds=975.0,
            source="okx_fills_history_open_position_close",
            okx_inst_id="MET-USDT-SWAP",
        )
        monkeypatch.setattr(
            repair_script,
            "_backup_open_position_closes",
            lambda _plans: _async_path(tmp_path),
        )

        result = await repair_script.apply_open_position_close_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            order = (
                await session.execute(
                    Order.__table__.select().where(Order.exchange_order_id == "met-close")
                )
            ).mappings().one()
            reflection = (
                await session.execute(
                    TradeReflection.__table__.select().where(
                        TradeReflection.position_id == position_id
                    )
                )
            ).mappings().one()
    finally:
        await close_db()

    assert result["applied"] == 1
    assert position.is_open is False
    assert position.close_exchange_order_id == "met-close"
    assert position.current_price == pytest.approx(0.1749)
    assert position.realized_pnl == pytest.approx(-0.049)
    assert position.closed_at == closed_at.replace(tzinfo=None)
    assert order["symbol"] == "MET/USDT"
    assert order["side"] == "buy"
    assert order["quantity"] == pytest.approx(20.0)
    assert order["price"] == pytest.approx(0.1749)
    assert reflection["source"] == repair_script.REPAIR_REFLECTION_SOURCE
    assert reflection["expert_lessons"]["training_policy"] == "exclude_until_manual_trust"


@pytest.mark.asyncio
async def test_apply_open_position_close_plan_reuses_existing_okx_confirmed_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'existing-close-order.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 28, 5, 0, tzinfo=UTC)
        closed_at = datetime(2026, 6, 28, 7, 46, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="ACT/USDT",
                side="short",
                quantity=1270.0,
                entry_price=0.01049,
                current_price=0.009077,
                leverage=3.0,
                realized_pnl=0.0,
                unrealized_pnl=1.79,
                is_open=True,
                created_at=opened_at,
                okx_inst_id="ACT-USDT-SWAP",
                entry_exchange_order_id="act-entry",
            )
            session.add(position)
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1270.0,
                    price=0.00908,
                    status="filled",
                    fee=0.0057658,
                    decision_id=None,
                    exchange_order_id="act-close",
                    filled_at=closed_at,
                    created_at=closed_at,
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_fill_contracts=127.0,
                    okx_fill_pnl=1.7907,
                    okx_sync_status="okx_confirmed",
                )
            )
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.OpenPositionClosePlan(
            position_id=position_id,
            symbol="ACT/USDT",
            side="short",
            close_side="buy",
            model_name="ensemble_trader",
            execution_mode="paper",
            okx_order_id="act-close",
            old_is_open=True,
            old_close_exchange_order_id=None,
            old_okx_inst_id="ACT-USDT-SWAP",
            quantity=1270.0,
            fill_quantity=1270.0,
            fill_contracts=127.0,
            contract_size=10.0,
            contract_size_source="okx_order_fill_contracts_ctVal",
            entry_price=0.01049,
            exit_price=0.00908,
            close_fee=0.0057658,
            fill_pnl=1.7907,
            computed_realized_pnl=1.7849342,
            old_realized_pnl=0.0,
            old_current_price=0.009077,
            fill_timestamp=closed_at,
            position_reference_time=opened_at,
            time_delta_seconds=9960.0,
            source="okx_confirmed_existing_close_order",
            okx_inst_id="ACT-USDT-SWAP",
        )
        monkeypatch.setattr(
            repair_script,
            "_backup_open_position_closes",
            lambda _plans: _async_path(tmp_path),
        )

        result = await repair_script.apply_open_position_close_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            order_count = (
                await session.execute(
                    select(func.count(Order.id)).where(Order.exchange_order_id == "act-close")
                )
            ).scalar_one()
            reflection = (
                await session.execute(
                    TradeReflection.__table__.select().where(
                        TradeReflection.position_id == position_id
                    )
                )
            ).mappings().one()
    finally:
        await close_db()

    assert result["applied"] == 1
    assert position.is_open is False
    assert position.close_exchange_order_id == "act-close"
    assert position.realized_pnl == pytest.approx(1.7907)
    assert position.unrealized_pnl == pytest.approx(0.0)
    assert order_count == 1
    assert reflection["expert_lessons"]["repair_plan"]["source"] == "okx_confirmed_existing_close_order"


@pytest.mark.asyncio
async def test_apply_close_link_reassignment_updates_order_and_reflection(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'close-link-reassign.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 27, 16, 58, tzinfo=UTC)
        closed_at = datetime(2026, 6, 27, 18, 58, 47, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="LIT/USDT",
                side="short",
                quantity=10.0,
                entry_price=1.819,
                current_price=1.929,
                leverage=3.0,
                realized_pnl=-1.13803,
                is_open=False,
                created_at=opened_at,
                closed_at=closed_at,
                okx_inst_id="LIT-USDT-SWAP",
                close_exchange_order_id="3693817049043931138",
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="LIT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=10.0,
                    price=1.929,
                    status="filled",
                    fee=0.028935,
                    decision_id=904,
                    exchange_order_id="3693817049043931138",
                    filled_at=closed_at,
                    created_at=closed_at,
                )
            )
            session.add(
                TradeReflection(
                    position_id=position_id,
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="LIT/USDT",
                    side="short",
                    entry_price=1.819,
                    exit_price=1.929,
                    quantity=10.0,
                    realized_pnl=-1.13803,
                    fee_estimate=0.03803,
                    hold_minutes=120.0,
                    closed_at=closed_at,
                    outcome="loss",
                    source="okx_reconcile",
                )
            )

        plan = repair_script.CloseLinkReassignmentPlan(
            position_id=position_id,
            symbol="LIT/USDT",
            side="short",
            close_side="buy",
            model_name="ensemble_trader",
            execution_mode="paper",
            old_okx_order_id="3693817049043931138",
            new_okx_order_id="3693817049043931137",
            old_fill_quantity=30.0,
            new_fill_quantity=10.0,
            old_fill_contracts=3.0,
            new_fill_contracts=1.0,
            contract_size=10.0,
            contract_size_source="okx_public_instruments",
            target_quantity=10.0,
            entry_price=1.819,
            exit_price=1.929,
            close_fee=0.009645,
            fill_pnl=-1.37142857,
            computed_realized_pnl=-1.109645,
            old_realized_pnl=-1.13803,
            fill_timestamp=closed_at,
            position_reference_time=closed_at,
            time_delta_seconds=0.0,
            source="okx_fills_history_close_link_reassignment",
            okx_inst_id="LIT-USDT-SWAP",
        )
        monkeypatch.setattr(
            repair_script,
            "_backup_close_link_reassignments",
            lambda _plans: _async_path(tmp_path),
        )

        result = await repair_script.apply_close_link_reassignment_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            order = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.exchange_order_id == "3693817049043931137"
                    )
                )
            ).mappings().one()
            reflections = (
                await session.execute(
                    TradeReflection.__table__.select().where(
                        TradeReflection.position_id == position_id
                    )
                )
            ).mappings().all()
    finally:
        await close_db()

    assert result["applied"] == 1
    assert position.close_exchange_order_id == "3693817049043931137"
    assert position.realized_pnl == pytest.approx(-1.37142857)
    assert order["quantity"] == pytest.approx(10.0)
    assert order["fee"] == pytest.approx(0.009645)
    assert {row["source"] for row in reflections} == {
        "okx_reconcile",
        repair_script.REPAIR_REFLECTION_SOURCE,
    }


@pytest.mark.asyncio
async def test_apply_native_full_close_shared_updates_split_positions_and_one_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'native-full-close-shared.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 27, 18, 13, 40, tzinfo=UTC)
        closed_at = datetime(2026, 6, 27, 19, 10, 45, tzinfo=UTC)
        fill_at = datetime(2026, 6, 27, 19, 11, 1, tzinfo=UTC)
        async with get_session_ctx() as session:
            position_ids: list[int] = []
            for _idx in range(2):
                position = Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SUSHI/USDT",
                    side="long",
                    quantity=67.5,
                    entry_price=0.1588,
                    current_price=0.156,
                    leverage=3.0,
                    realized_pnl=-0.1943595,
                    is_open=False,
                    closed_at=closed_at,
                    created_at=opened_at,
                    okx_inst_id="SUSHI-USDT-SWAP",
                    entry_exchange_order_id="3693726686522347520",
                    close_exchange_order_id="okx_native_full_close",
                )
                session.add(position)
                await session.flush()
                position_ids.append(int(position.id))
                session.add(
                    TradeReflection(
                        position_id=int(position.id),
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SUSHI/USDT",
                        side="long",
                        entry_price=0.1588,
                        exit_price=0.156,
                        quantity=67.5,
                        realized_pnl=-0.1943595,
                        fee_estimate=0.0053595,
                        hold_minutes=57.0,
                        closed_at=closed_at,
                        outcome="loss",
                        source="system_execution",
                    )
                )
            close_order = Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="SUSHI/USDT",
                side="sell",
                order_type="market",
                quantity=135.0,
                price=0.156,
                status="filled",
                fee=0.0,
                decision_id=969,
                exchange_order_id=None,
                filled_at=closed_at,
                created_at=closed_at,
            )
            session.add(close_order)
            await session.flush()
            close_order_id = int(close_order.id)

        plan = repair_script.NativeFullCloseSharedPlan(
            position_ids=tuple(position_ids),
            symbol="SUSHI/USDT",
            side="long",
            close_side="sell",
            model_name="ensemble_trader",
            execution_mode="paper",
            close_order_id=close_order_id,
            old_exchange_order_id=None,
            okx_order_id="3693841704639238144",
            total_quantity=135.0,
            fill_quantity=135.0,
            fill_contracts=135.0,
            contract_size=1.0,
            entry_price_weighted=0.1588,
            exit_price=0.15541926,
            close_fee=0.0104908,
            fill_pnl=-0.4564,
            fill_timestamp=fill_at,
            source="okx_fills_history_native_full_close_shared",
            okx_inst_id="SUSHI-USDT-SWAP",
        )
        monkeypatch.setattr(
            repair_script,
            "_backup_native_full_close_shared",
            lambda _plans: _async_path(tmp_path),
        )

        result = await repair_script.apply_native_full_close_shared_plans([plan])

        async with get_session_ctx() as session:
            positions = [
                await session.get(Position, position_id) for position_id in position_ids
            ]
            order = await session.get(Order, close_order_id)
            reflections = (
                await session.execute(
                    TradeReflection.__table__.select().where(
                        TradeReflection.position_id.in_(position_ids)
                    )
                )
            ).mappings().all()
    finally:
        await close_db()

    assert result["applied"] == 1
    assert all(position.close_exchange_order_id == "3693841704639238144" for position in positions)
    assert all(position.current_price == pytest.approx(0.15541926) for position in positions)
    assert all(position.realized_pnl == pytest.approx(-0.2282) for position in positions)
    assert order.exchange_order_id == "3693841704639238144"
    assert order.quantity == pytest.approx(135.0)
    assert order.fee == pytest.approx(0.0104908)
    assert len(reflections) == 4
    assert sum(1 for row in reflections if row["source"] == repair_script.REPAIR_REFLECTION_SOURCE) == 2


@pytest.mark.asyncio
async def test_apply_open_position_close_plan_rejects_missing_okx_inst_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'open-position-missing-inst.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 27, 18, 0, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="MET/USDT",
                side="short",
                quantity=20.0,
                entry_price=0.17245,
                current_price=0.1733,
                leverage=3.0,
                realized_pnl=0.0,
                is_open=True,
                created_at=opened_at,
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.OpenPositionClosePlan(
            position_id=position_id,
            symbol="MET/USDT",
            side="short",
            close_side="buy",
            model_name="ensemble_trader",
            execution_mode="paper",
            okx_order_id="met-close",
            old_is_open=True,
            old_close_exchange_order_id=None,
            old_okx_inst_id=None,
            quantity=20.0,
            fill_quantity=20.0,
            fill_contracts=2.0,
            contract_size=10.0,
            contract_size_source="okx_public_instruments",
            entry_price=0.17245,
            exit_price=0.1749,
            close_fee=0.001,
            fill_pnl=-0.049,
            computed_realized_pnl=-0.05,
            old_realized_pnl=0.0,
            old_current_price=0.1733,
            fill_timestamp=opened_at,
            position_reference_time=opened_at,
            time_delta_seconds=0.0,
            source="okx_fills_history_open_position_close",
            okx_inst_id="",
        )
        monkeypatch.setattr(
            repair_script,
            "_backup_open_position_closes",
            lambda _plans: _async_path(tmp_path),
        )

        result = await repair_script.apply_open_position_close_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            order_count = (
                await session.execute(select(func.count()).select_from(Order))
            ).scalar_one()
    finally:
        await close_db()

    assert result["applied"] == 0
    assert position.is_open is True
    assert position.close_exchange_order_id is None
    assert order_count == 0


@pytest.mark.asyncio
async def test_orphan_open_position_quarantine_plan_requires_okx_absence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'orphan-position-plan.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 28, 1, 0, tzinfo=UTC)
        async with get_session_ctx() as session:
            missing = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="ICP/USDT",
                side="short",
                quantity=6.02,
                entry_price=2.159,
                current_price=2.159,
                leverage=3.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                is_open=True,
                created_at=opened_at,
                okx_inst_id="ICP-USDT-SWAP",
                entry_exchange_order_id="icp-entry",
            )
            present = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="FLOKI/USDT",
                side="short",
                quantity=600000.0,
                entry_price=0.00002174,
                current_price=0.0000217,
                leverage=3.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                is_open=True,
                created_at=opened_at,
                okx_inst_id="FLOKI-USDT-SWAP",
                entry_exchange_order_id="floki-entry",
            )
            session.add_all([missing, present])
            await session.flush()
            missing_id = int(missing.id)

        monkeypatch.setattr(
            repair_script,
            "_fetch_current_okx_position_keys",
            lambda _positions: _async_value({("FLOKI-USDT-SWAP", "short")}),
        )
        monkeypatch.setattr(
            repair_script,
            "collect_open_position_close_plans",
            lambda **_kwargs: _async_value([]),
        )

        plans = await repair_script.collect_orphan_open_position_quarantine_plans(
            position_ids=(missing_id,),
        )
    finally:
        await close_db()

    assert len(plans) == 1
    assert plans[0].position_id == missing_id
    assert plans[0].okx_inst_id == "ICP-USDT-SWAP"
    assert plans[0].source == "okx_current_position_absent_no_close_fill"


@pytest.mark.asyncio
async def test_apply_orphan_open_position_quarantine_marks_non_trainable_without_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'orphan-position-apply.db').as_posix()}",
    )
    await init_db()
    try:
        opened_at = datetime(2026, 6, 28, 1, 0, tzinfo=UTC)
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="ICP/USDT",
                side="short",
                quantity=6.02,
                entry_price=2.159,
                current_price=2.159,
                leverage=3.0,
                realized_pnl=0.42,
                unrealized_pnl=-0.1,
                is_open=True,
                created_at=opened_at,
                okx_inst_id="ICP-USDT-SWAP",
                entry_exchange_order_id="icp-entry",
            )
            session.add(position)
            await session.flush()
            position_id = int(position.id)

        plan = repair_script.OrphanOpenPositionQuarantinePlan(
            position_id=position_id,
            symbol="ICP/USDT",
            side="short",
            model_name="ensemble_trader",
            execution_mode="paper",
            quantity=6.02,
            entry_price=2.159,
            old_current_price=2.159,
            old_unrealized_pnl=-0.1,
            old_realized_pnl=0.42,
            old_close_exchange_order_id=None,
            old_okx_inst_id="ICP-USDT-SWAP",
            source="okx_current_position_absent_no_close_fill",
            okx_inst_id="ICP-USDT-SWAP",
            reason="OKX current positions do not contain this local open row.",
        )
        monkeypatch.setattr(
            repair_script,
            "_backup_orphan_open_position_quarantines",
            lambda _plans: _async_path(tmp_path),
        )

        result = await repair_script.apply_orphan_open_position_quarantine_plans([plan])

        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            order_count = (
                await session.execute(select(func.count()).select_from(Order))
            ).scalar_one()
            reflection = (
                await session.execute(
                    TradeReflection.__table__.select().where(
                        TradeReflection.position_id == position_id
                    )
                )
            ).mappings().one()
    finally:
        await close_db()

    assert result["applied"] == 1
    assert position.is_open is False
    assert position.realized_pnl == pytest.approx(0.0)
    assert position.unrealized_pnl == pytest.approx(0.0)
    assert position.close_exchange_order_id == f"{repair_script.ORPHAN_QUARANTINE_CLOSE_PREFIX}{position_id}"
    assert order_count == 0
    assert reflection["source"] == repair_script.ORPHAN_QUARANTINE_REFLECTION_SOURCE
    assert reflection["expert_lessons"]["training_policy"] == "exclude_until_manual_trust"


async def _async_path(tmp_path):
    return tmp_path / "backup.json"


async def _async_value(value):
    return value
