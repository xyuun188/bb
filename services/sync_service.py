"""OKX and local-position synchronization boundary."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select

from ai_brain.base_model import Action, DecisionOutput
from config.settings import ENSEMBLE_TRADER_NAME
from core.trading_mode import mode_manager
from db.repositories.account_repo import AccountRepository
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from executor.base_executor import OrderStatus
from models.trade import Order, Position


logger = structlog.get_logger(__name__)


class OkxSyncService:
    """Owns OKX reconciliation and open-position context building."""

    def __init__(self, orchestrator: Any) -> None:
        self.orchestrator = orchestrator

    async def reconcile_positions(self, reason: str, timeout: float = 25.0) -> list[dict]:
        ts = self.orchestrator
        try:
            async def _locked_reconcile() -> list[dict]:
                async with ts._exchange_reconcile_lock:
                    return await self.reconcile_exchange_positions()

            return await asyncio.wait_for(_locked_reconcile(), timeout=timeout)
        except asyncio.TimeoutError:
            timeout_reason = (
                f"exchange position reconciliation timed out during {reason}; "
                "continuing with local position state"
            )
            ts._last_round_error = timeout_reason
            logger.warning(timeout_reason)
            return []

    async def refresh_position_prices(self, feature_vectors: dict[str, Any]) -> Any:
        """Update persisted open-position prices and unrealized PnL."""
        ts = self.orchestrator
        try:
            async with get_session_ctx() as session:
                trade_repo = TradeRepository(session)
                account_repo = AccountRepository(session)
                positions = await trade_repo.get_open_positions()
                open_context: list[dict[str, Any]] = []
                pnl_by_model: dict[str, float] = {}

                for pos in positions:
                    fv = feature_vectors.get(pos.symbol)
                    current_price = (
                        fv.current_price
                        if fv is not None and getattr(fv, "current_price", 0)
                        else pos.current_price or pos.entry_price
                    )
                    if not current_price or current_price <= 0:
                        continue

                    if pos.side == "short":
                        unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
                    else:
                        unrealized_pnl = (current_price - pos.entry_price) * pos.quantity

                    await trade_repo.update_position_price(
                        pos.id,
                        float(current_price),
                        float(unrealized_pnl),
                    )
                    open_context.append({
                        "model_name": pos.model_name,
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "current_price": float(current_price),
                        "entry_price": float(pos.entry_price or 0.0),
                        "unrealized_pnl": float(unrealized_pnl),
                        "created_at": pos.created_at,
                        "is_open": True,
                    })
                    ts._update_position_profit_peak(
                        model_name=pos.model_name,
                        symbol=pos.symbol,
                        side=pos.side,
                        current_price=float(current_price),
                        entry_price=float(pos.entry_price or 0.0),
                        unrealized_pnl=float(unrealized_pnl),
                        hold_minutes=ts._position_age_minutes(pos.created_at),
                    )
                    pnl_by_model[pos.model_name] = pnl_by_model.get(pos.model_name, 0.0) + float(unrealized_pnl)

                for model_name, unrealized_pnl in pnl_by_model.items():
                    await account_repo.update_unrealized_pnl(model_name, round(unrealized_pnl, 8))
                ts._prune_position_profit_peaks(open_context)
        except Exception as e:
            logger.warning("failed to refresh DB position prices", error=str(e))

    async def has_matching_exchange_exit_position(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool:
        ts = self.orchestrator
        """Check OKX directly before rejecting a close decision for local mismatch."""
        if not decision.is_exit:
            return True
        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        target_symbol = ts._normalize_position_symbol(decision.symbol)
        mode = ts._get_model_execution_mode(model_name)
        try:
            executor = await ts._get_okx_executor_for_mode(mode)
            positions = await asyncio.wait_for(
                executor.get_positions(decision.symbol),
                timeout=8.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "timed out checking OKX position before exit",
                model=model_name,
                symbol=decision.symbol,
            )
            return False
        except Exception as e:
            logger.warning(
                "failed to check OKX position before exit",
                model=model_name,
                symbol=decision.symbol,
                error=str(e),
            )
            return False

        for pos in positions or []:
            if ts._normalize_position_symbol(pos.get("symbol")) != target_symbol:
                continue
            if str(pos.get("side") or "").lower() != target_side:
                continue
            info = pos.get("info") if isinstance(pos.get("info"), dict) else {}
            quantity = ts._safe_float(
                pos.get("contracts")
                or pos.get("size")
                or pos.get("positionAmt")
                or info.get("pos")
                or info.get("qty"),
                0.0,
            )
            if abs(quantity) > 0:
                return True
        return False


    async def reconcile_exchange_positions(self) -> list[dict]:
        ts = self.orchestrator
        """Reconcile local paper positions with actual OKX demo positions.

        OKX attached TP/SL orders can close positions without going through the
        AI decision loop. When that happens, close the local DB position using
        the exchange fill so the dashboard and account state remain truthful.
        """
        if not ts._okx_paper:
            return []

        try:
            exchange_positions = await asyncio.wait_for(
                ts._okx_paper.get_positions_strict(),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning("timed out fetching OKX positions for reconciliation")
            return []
        except Exception as e:
            logger.warning("failed to fetch OKX positions for reconciliation", error=str(e))
            return []

        protection_by_key = await ts._fetch_exchange_protection_map(
            ts._okx_paper,
            exchange_positions,
        )

        exchange_position_keys = {
            (
                ts._normalize_position_symbol(p.get("symbol")),
                str(p.get("side") or "").lower(),
            )
            for p in exchange_positions or []
            if ts._exchange_position_is_open(p)
        }
        exchange_position_keys.discard(("", ""))
        reconciled: list[dict] = []

        try:
            async with get_session_ctx() as session:
                trade_repo = TradeRepository(session)
                account_repo = AccountRepository(session)
                positions = await trade_repo.get_open_positions()
                local_open_keys = {
                    (
                        ts._normalize_position_symbol(pos.symbol),
                        str(pos.side or "").lower(),
                    )
                    for pos in positions
                    if pos.execution_mode == "paper" and pos.is_open
                }

                for exchange_pos in exchange_positions or []:
                    if not ts._exchange_position_is_open(exchange_pos):
                        continue
                    symbol = ts._normalize_position_symbol(exchange_pos.get("symbol"))
                    side = str(exchange_pos.get("side") or "").lower()
                    if not symbol or side not in {"long", "short"}:
                        continue
                    key = (symbol, side)

                    info = exchange_pos.get("info") or {}
                    contracts = ts._safe_float(exchange_pos.get("contracts"), 0.0)
                    contract_size = ts._safe_float(exchange_pos.get("contractSize"), 1.0) or 1.0
                    quantity = abs(contracts * contract_size)
                    entry_price = ts._safe_float(
                        exchange_pos.get("entryPrice") or info.get("avgPx"),
                        0.0,
                    )
                    if quantity <= 0 or entry_price <= 0:
                        continue

                    current_price = ts._safe_float(
                        exchange_pos.get("markPrice") or exchange_pos.get("lastPrice") or entry_price,
                        entry_price,
                    )
                    leverage = ts._safe_float(exchange_pos.get("leverage") or info.get("lever"), 1.0) or 1.0
                    exchange_unrealized = ts._safe_float(exchange_pos.get("unrealizedPnl"), 0.0)
                    exchange_realized = ts._safe_float(exchange_pos.get("realizedPnl"), 0.0)
                    protection = protection_by_key.get(key, {})
                    fallback_protection = await ts._fallback_position_protection_from_decision(
                        session,
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                    )
                    stop_loss_price = protection.get("stop_loss_price") or fallback_protection.get("stop_loss_price")
                    take_profit_price = protection.get("take_profit_price") or fallback_protection.get("take_profit_price")

                    matching_local_positions = [
                        pos for pos in positions
                        if (
                            pos.execution_mode == "paper"
                            and pos.is_open
                            and ts._normalize_position_symbol(pos.symbol) == symbol
                            and str(pos.side or "").lower() == side
                        )
                    ]
                    if matching_local_positions:
                        changed = ts._sync_local_open_position_snapshot(
                            matching_local_positions,
                            exchange_quantity=quantity,
                            current_price=current_price,
                            entry_price=entry_price,
                            leverage=leverage,
                            exchange_unrealized=exchange_unrealized,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                        )
                        local_open_keys.add(key)
                        if changed:
                            reconciled.append({
                                "model_name": matching_local_positions[0].model_name,
                                "symbol": symbol,
                                "side": side,
                                "quantity": quantity,
                                "current_price": current_price,
                                "note": "OKX 持仓数量或价格已变化，本地持仓快照已同步更新。",
                            })
                        continue

                    closed_position_result = await session.execute(
                        select(Position)
                        .where(
                            Position.execution_mode == "paper",
                            Position.symbol == symbol,
                            Position.side == side,
                            Position.is_open == False,
                        )
                        .order_by(Position.created_at.desc())
                        .limit(1)
                    )
                    closed_position = closed_position_result.scalar_one_or_none()
                    if closed_position:
                        closed_position.is_open = True
                        closed_position.quantity = quantity
                        closed_position.entry_price = entry_price
                        closed_position.current_price = current_price
                        closed_position.leverage = leverage
                        closed_position.unrealized_pnl = exchange_unrealized
                        closed_position.realized_pnl = exchange_realized
                        closed_position.stop_loss_price = stop_loss_price
                        closed_position.take_profit_price = take_profit_price
                        closed_position.closed_at = None
                        closed_position.updated_at = datetime.now(timezone.utc)
                        local_open_keys.add(key)
                        reconciled.append({
                            "model_name": closed_position.model_name,
                            "symbol": symbol,
                            "side": side,
                            "entry_price": entry_price,
                            "note": "OKX 仍有持仓，本地之前误记为已平仓，已重新打开本地持仓记录。",
                        })
                        logger.warning(
                            "reopened local position still open on OKX",
                            position_id=closed_position.id,
                            symbol=symbol,
                            side=side,
                        )
                        continue

                    entry_side = "buy" if side == "long" else "sell"
                    order_result = await session.execute(
                        select(Order)
                        .where(
                            Order.execution_mode == "paper",
                            Order.symbol == symbol,
                            Order.side == entry_side,
                            Order.exchange_order_id.is_not(None),
                            Order.exchange_order_id != "",
                            Order.status.in_([
                                OrderStatus.OPEN.value,
                                OrderStatus.PENDING.value,
                                OrderStatus.PARTIAL.value,
                                OrderStatus.FILLED.value,
                            ]),
                        )
                        .order_by(Order.created_at.desc())
                        .limit(1)
                    )
                    order = order_result.scalar_one_or_none()
                    if not order:
                        continue
                    if not stop_loss_price or not take_profit_price:
                        order_fallback_protection = await ts._fallback_position_protection_from_decision(
                            session,
                            symbol=symbol,
                            side=side,
                            entry_price=entry_price,
                            order=order,
                        )
                        stop_loss_price = stop_loss_price or order_fallback_protection.get("stop_loss_price")
                        take_profit_price = take_profit_price or order_fallback_protection.get("take_profit_price")

                    opened_at = ts._datetime_from_ms(exchange_pos.get("timestamp") or info.get("cTime"))
                    order.status = OrderStatus.FILLED.value
                    order.quantity = order.quantity or quantity
                    order.price = order.price or entry_price
                    order.filled_at = order.filled_at or opened_at

                    await trade_repo.open_position({
                        "model_name": order.model_name or ENSEMBLE_TRADER_NAME,
                        "execution_mode": "paper",
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "leverage": leverage,
                        "unrealized_pnl": exchange_unrealized,
                        "realized_pnl": exchange_realized,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                    })
                    local_open_keys.add(key)
                    reconciled.append({
                        "model_name": order.model_name or ENSEMBLE_TRADER_NAME,
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "exchange_order_id": order.exchange_order_id,
                        "note": "OKX 已有持仓但本地缺失，已按执行订单补回持仓记录。",
                    })
                    logger.warning(
                        "synced missing local position from OKX",
                        symbol=symbol,
                        side=side,
                        order_id=order.exchange_order_id,
                    )

                for pos in positions:
                    if pos.execution_mode != "paper":
                        continue
                    if not pos.is_open:
                        continue
                    symbol = ts._normalize_position_symbol(pos.symbol)
                    if (symbol, str(pos.side or "").lower()) in exchange_position_keys:
                        continue

                    try:
                        await session.refresh(pos)
                    except Exception as e:
                        logger.warning(
                            "failed to refresh local position before exchange reconciliation close",
                            position_id=pos.id,
                            symbol=pos.symbol,
                            side=pos.side,
                            error=str(e),
                        )
                    if not pos.is_open:
                        logger.info(
                            "skip exchange reconciliation close; local position already closed",
                            position_id=pos.id,
                            symbol=pos.symbol,
                            side=pos.side,
                        )
                        continue

                    close_fill = await ts._find_exchange_close_fill(pos)
                    if not close_fill.get("order_id"):
                        now = datetime.now(timezone.utc)
                        opened_at = pos.created_at
                        if opened_at and opened_at.tzinfo is None:
                            opened_at = opened_at.replace(tzinfo=timezone.utc)
                        age_seconds = (now - opened_at).total_seconds() if opened_at else 0.0
                        if age_seconds < UNCONFIRMED_EXCHANGE_CLOSE_GRACE_SECONDS:
                            logger.warning(
                                "exchange position missing but close fill not found; waiting before local close",
                                position_id=pos.id,
                                symbol=pos.symbol,
                                side=pos.side,
                                age_seconds=round(age_seconds, 1),
                            )
                            continue

                        exit_price = pos.current_price or pos.entry_price
                        fresh = await ts._fresh_feature_vector_for_price_recheck(pos.symbol)
                        if fresh is not None:
                            fresh_price = ts._safe_float(
                                ts._market_value(fresh, "current_price")
                                or ts._market_value(fresh, "close")
                                or ts._market_value(fresh, "bid")
                                or ts._market_value(fresh, "ask"),
                                0.0,
                            )
                            if fresh_price > 0:
                                exit_price = fresh_price
                        if pos.side == "short":
                            gross_pnl = (float(pos.entry_price or 0.0) - float(exit_price or 0.0)) * float(pos.quantity or 0.0)
                            close_side = "buy"
                        else:
                            gross_pnl = (float(exit_price or 0.0) - float(pos.entry_price or 0.0)) * float(pos.quantity or 0.0)
                            close_side = "sell"
                        entry_fee = await ts._entry_fee_for_position(session, pos, pos.quantity)
                        realized_pnl = gross_pnl - entry_fee
                        decision_id = await ts._log_exchange_sync_close_decision(
                            session=session,
                            pos=pos,
                            exit_price=exit_price,
                            realized_pnl=realized_pnl,
                            closed_at=now,
                            reason=(
                                "OKX 已没有这笔持仓，但没有查到对应平仓成交回报；"
                                "系统按交易所仓位状态同步为平仓，并用本地开仓价与同步平仓价估算盈亏。"
                            ),
                            close_fill={
                                "estimated": True,
                                "price": exit_price,
                                "gross_pnl": gross_pnl,
                                "entry_fee": entry_fee,
                                "fee": 0.0,
                                "pnl": realized_pnl,
                                "fresh_price_used": bool(fresh is not None),
                                "note": "close fill not found; realized pnl estimated from local entry and freshest available sync price",
                            },
                        )
                        pos.is_open = False
                        pos.current_price = exit_price
                        pos.unrealized_pnl = 0.0
                        pos.realized_pnl = realized_pnl
                        pos.closed_at = now
                        await trade_repo.create_order({
                            "model_name": pos.model_name,
                            "execution_mode": pos.execution_mode,
                            "symbol": pos.symbol,
                            "side": close_side,
                            "order_type": "market",
                            "quantity": pos.quantity,
                            "price": exit_price,
                            "status": OrderStatus.FILLED.value,
                            "fee": 0.0,
                            "decision_id": decision_id,
                            "exchange_order_id": None,
                            "filled_at": now,
                        })
                        reconciled.append({
                            "model_name": pos.model_name,
                            "symbol": pos.symbol,
                            "side": pos.side,
                            "exit_price": pos.current_price,
                            "realized_pnl": realized_pnl,
                            "exchange_order_id": None,
                            "note": "OKX 已无对应持仓，未查到平仓成交回报；本地已按同步价格估算盈亏并关闭仓位。",
                        })
                        logger.warning(
                            "closed unsynced local position; no OKX open position or close fill found",
                            position_id=pos.id,
                            symbol=pos.symbol,
                            side=pos.side,
                        )
                        continue
                    exit_price = close_fill.get("price") or pos.current_price or pos.entry_price
                    if not exit_price or exit_price <= 0:
                        logger.warning(
                            "skip local close; invalid OKX close fill price",
                            position_id=pos.id,
                            symbol=pos.symbol,
                            side=pos.side,
                            exchange_order_id=close_fill.get("order_id"),
                        )
                        continue
                    close_fee = float(close_fill.get("fee") or 0.0)
                    if pos.side == "short":
                        gross_pnl = (pos.entry_price - exit_price) * pos.quantity
                        close_side = "buy"
                    else:
                        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
                        close_side = "sell"
                    entry_fee = await ts._entry_fee_for_position(session, pos, pos.quantity)
                    realized_pnl = gross_pnl - entry_fee - close_fee

                    pos.is_open = False
                    pos.current_price = exit_price
                    pos.unrealized_pnl = 0.0
                    pos.realized_pnl = realized_pnl
                    pos.closed_at = close_fill.get("timestamp") or datetime.now(timezone.utc)
                    decision_id = await ts._log_exchange_sync_close_decision(
                        session=session,
                        pos=pos,
                        exit_price=exit_price,
                        realized_pnl=realized_pnl,
                        closed_at=pos.closed_at,
                        reason=(
                            "OKX 已返回平仓成交，系统同步为平仓记录；"
                            "这通常来自 OKX 止盈止损、手动平仓或交易所侧自动平仓。"
                        ),
                        close_fill=close_fill,
                    )
                    await ts._record_trade_reflection_in_session(
                        session,
                        pos,
                        exit_price=exit_price,
                        entry_fee=entry_fee,
                        close_fee=close_fee,
                        gross_pnl=gross_pnl,
                        source="okx_reconcile",
                        decision=None,
                    )

                    existing_close_order = None
                    close_order_id = str(close_fill.get("order_id") or "")
                    if close_order_id:
                        existing_close_order_result = await session.execute(
                            select(Order.id)
                            .where(
                                Order.execution_mode == pos.execution_mode,
                                Order.exchange_order_id == close_order_id,
                            )
                            .limit(1)
                        )
                        existing_close_order = existing_close_order_result.scalar_one_or_none()

                    if not existing_close_order:
                        await trade_repo.create_order({
                            "model_name": pos.model_name,
                            "execution_mode": pos.execution_mode,
                            "symbol": pos.symbol,
                            "side": close_side,
                            "order_type": "market",
                            "quantity": pos.quantity,
                            "price": exit_price,
                            "status": OrderStatus.FILLED.value,
                            "fee": close_fee,
                            "decision_id": decision_id,
                            "exchange_order_id": close_fill.get("order_id"),
                            "filled_at": pos.closed_at,
                        })

                    released_margin = ts._position_margin(
                        pos.quantity * pos.entry_price,
                        pos.leverage,
                    )
                    await account_repo.update_balance(
                        pos.model_name,
                        released_margin + realized_pnl,
                        realized_pnl,
                    )
                    await account_repo.record_trade_result(pos.model_name, realized_pnl > 0)
                    ts._remove_memory_position(pos.model_name, pos.symbol, pos.side)

                    reconciled.append({
                        "model_name": pos.model_name,
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "exit_price": exit_price,
                        "realized_pnl": realized_pnl,
                        "gross_pnl": gross_pnl,
                        "fees": entry_fee + close_fee,
                        "exchange_order_id": close_fill.get("order_id"),
                    })

        except Exception as e:
            logger.warning("exchange position reconciliation failed", error=str(e))
            return reconciled

        if reconciled:
            logger.info("reconciled exchange-closed positions", count=len(reconciled), positions=reconciled)
        return reconciled


    async def get_open_positions_context(self) -> list[dict]:
        ts = self.orchestrator
        """Get open positions for LLM context (paper + live)."""
        positions = []

        try:
            async with get_session_ctx() as session:
                repo = TradeRepository(session)
                db_positions = await repo.get_position_records(
                    execution_mode=mode_manager.mode.value,
                    model_name=ENSEMBLE_TRADER_NAME,
                    limit=1000,
                )
                for p in db_positions:
                    if p.is_open:
                        positions.append({
                            "model_name": p.model_name,
                            "symbol": p.symbol,
                            "side": p.side,
                            "entry_price": p.entry_price,
                            "current_price": p.current_price or p.entry_price,
                            "quantity": p.quantity,
                            "leverage": p.leverage or 1.0,
                            "unrealized_pnl": p.unrealized_pnl,
                            "stop_loss": p.stop_loss_price,
                            "take_profit": p.take_profit_price,
                            "is_open": p.is_open,
                            "created_at": p.created_at,
                        })
        except Exception as e:
            logger.warning("failed to load DB positions for context", error=str(e))

        # Keep in-memory paper positions as a fallback for positions opened before
        # persistence was introduced in this process.
        if not positions and ts.paper_executor:
            positions.extend(await ts.paper_executor.get_positions())

        # OKX positions for the active execution account.
        active_okx = ts._okx_live if mode_manager.mode.value == "live" else ts._okx_paper
        if active_okx:
            try:
                okx_positions = await asyncio.wait_for(
                    active_okx.get_positions(),
                    timeout=8.0,
                )
                exchange_keys = {
                    (
                        ts._normalize_position_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    )
                    for p in (okx_positions or [])
                    if ts._exchange_position_is_open(p)
                }
                positions = [
                    p for p in positions
                    if (
                        ts._normalize_position_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    ) in exchange_keys
                ]
                existing_keys = {
                    (
                        ts._normalize_position_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    )
                    for p in positions
                }
                for p in (okx_positions or []):
                    p["model_name"] = p.get("model_name") or ENSEMBLE_TRADER_NAME
                    key = (
                        ts._normalize_position_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    )
                    if key not in existing_keys:
                        positions.append(p)
            except Exception:
                pass

        return [
            {
                "model_name": p.get("model_name", ""),
                "symbol": p.get("symbol", ""),
                "side": p.get("side", "long"),
                "entry_price": p.get("entry_price") or p.get("entryPrice") or p.get("avgPx") or 0,
                "current_price": (
                    p.get("current_price")
                    or p.get("markPrice")
                    or p.get("lastPrice")
                    or p.get("entry_price")
                    or p.get("entryPrice")
                    or p.get("avgPx")
                    or 0
                ),
                "quantity": p.get("quantity") or p.get("contracts") or p.get("sz") or 0,
                "contracts": p.get("contracts") or p.get("sz"),
                "contract_size": (
                    p.get("contract_size")
                    or p.get("contractSize")
                    or (p.get("info") or {}).get("ctVal")
                ),
                "contractSize": (
                    p.get("contractSize")
                    or p.get("contract_size")
                    or (p.get("info") or {}).get("ctVal")
                ),
                "leverage": ts._safe_float(
                    p.get("leverage") or (p.get("info") or {}).get("lever"),
                    1.0,
                ),
                "notional": (
                    p.get("notional")
                    or p.get("notional_usd")
                    or p.get("notionalUsd")
                    or (p.get("info") or {}).get("notionalUsd")
                    or (p.get("info") or {}).get("notional")
                    or (p.get("info") or {}).get("posValue")
                    or 0
                ),
                "margin": (
                    p.get("margin")
                    or p.get("initial_margin")
                    or p.get("initialMargin")
                    or p.get("margin_used")
                    or (p.get("info") or {}).get("margin")
                    or (p.get("info") or {}).get("imr")
                ),
                "initial_margin": (
                    p.get("initial_margin")
                    or p.get("initialMargin")
                    or (p.get("info") or {}).get("imr")
                ),
                "initialMargin": (
                    p.get("initialMargin")
                    or p.get("initial_margin")
                    or (p.get("info") or {}).get("imr")
                ),
                "unrealized_pnl": p.get("unrealized_pnl", p.get("unrealizedPnl", 0)),
                "stop_loss": p.get("stop_loss"),
                "take_profit": p.get("take_profit"),
                "is_open": p.get("is_open", True),
                "created_at": (
                    p.get("created_at")
                    or p.get("timestamp")
                    or p.get("opened_at")
                    or (p.get("info") or {}).get("cTime")
                    or (p.get("info") or {}).get("uTime")
                ),
                "info": p.get("info") if isinstance(p.get("info"), dict) else {},
            }
            for p in (positions or [])
        ]

