"""
OKX live/demo executor via CCXT.
Sends real orders to OKX exchange (demo or production based on settings).
Includes retry logic, rate limiting, and error handling.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from core.exceptions import (
    ExchangeAPIError,
    OrderPlacementError,
    RateLimitError,
)
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from executor.base_executor import AbstractExecutor, ExecutionResult, OrderStatus

logger = structlog.get_logger(__name__)

OKX_REST_URL = "https://{hostname}"
OKX_HOSTNAME = "www.okx.com"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
RATE_LIMIT_TOKENS = 10  # max requests per second
RATE_LIMIT_PERIOD = 1.0
OKX_REST_CALL_TIMEOUT = 10.0
EXIT_ORDER_REPLACE_AFTER_SECONDS = 20.0
ATTACHED_PROTECTION_MIN_STOP_PCT = 0.012
ATTACHED_PROTECTION_MIN_TAKE_PROFIT_PCT = 0.024
ATTACHED_PROTECTION_MIN_TRIGGER_GAP_PCT = 0.0015


def _okx_proxy_url() -> str | None:
    return (
        settings.okx_proxy
        or os.environ.get("OKX_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
    )


class TokenBucket:
    """Simple token bucket for rate limiting API requests."""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.last_update) * self.rate)
        self.last_update = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    async def wait_for_token(self) -> None:
        # Token refill is time-based, so this wait intentionally sleeps until
        # enough time has passed for another token to become available.
        while not self.consume():  # noqa: ASYNC110
            await asyncio.sleep(1.0 / self.rate)


class OKXExecutor(AbstractExecutor):
    """Executes trades on OKX via CCXT.

    In demo mode (settings.okx_demo=True), trades go to OKX demo trading environment.
    In production mode, real orders are placed.
    """

    def __init__(self, mode: str | None = None) -> None:
        self._mode_override = mode  # "paper" or "live", overrides global mode
        self._exchange: Any = None
        self._rate_limiter = TokenBucket(RATE_LIMIT_TOKENS, RATE_LIMIT_TOKENS * 2)
        self._connected = False
        self._leverage_cache: dict[tuple[str, str], tuple[float, float]] = {}

    @property
    def executor_mode(self) -> str:
        return self._mode_override or mode_manager.mode.value

    async def initialize(self) -> None:
        import ccxt.async_support as ccxt_async

        mode = self.executor_mode
        creds = settings.get_okx_credentials(mode)
        is_demo = settings.is_okx_demo(mode)

        config: dict[str, Any] = {
            "apiKey": creds["api_key"],
            "secret": creds["api_secret"],
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "defaultSubType": "linear",
                "fetchMarkets": ["swap"],
            },
        }
        if creds.get("passphrase"):
            config["password"] = creds["passphrase"]
        if is_demo:
            # OKX demo trading uses the normal REST host with this header.
            config["headers"] = {"x-simulated-trading": "1"}
        proxy_url = _okx_proxy_url()
        if proxy_url:
            config["aiohttp_proxy"] = proxy_url

        self._exchange = ccxt_async.okx(config)
        if is_demo:
            self._exchange.set_sandbox_mode(True)
        self._ensure_rest_url()

        try:
            await self._load_usdt_swap_markets()
            self._connected = True
            logger.info(
                "OKX executor initialized",
                mode=mode,
                demo=is_demo,
                markets=len(self._exchange.markets),
            )
        except Exception:
            await self.shutdown()
            raise

    def _ensure_rest_url(self) -> None:
        """Repair CCXT OKX URL fields before every exchange call."""
        if self._exchange is None:
            return
        urls = getattr(self._exchange, "urls", None)
        if not isinstance(urls, dict):
            return
        for key in ("api", "test"):
            value = urls.get(key)
            if not isinstance(value, dict):
                urls[key] = {"rest": OKX_REST_URL}
            elif not value.get("rest"):
                value["rest"] = OKX_REST_URL
        if not getattr(self._exchange, "hostname", None):
            self._exchange.hostname = OKX_HOSTNAME

    def _is_broken_rest_url_error(self, exc: Exception) -> bool:
        message = safe_error_text(exc)
        return "unsupported operand type(s) for +: 'NoneType' and 'str'" in message or (
            "NoneType" in message and "+:" in message and "str" in message
        )

    async def _load_usdt_swap_markets(self) -> None:
        """Load only live linear USDT perpetual swaps.

        OKX demo public instruments can include test/preopen contracts with missing
        fields. CCXT's full load_markets() tries to parse all of them and can fail
        before credentials are even tested. The trading system only needs live
        USDT swaps, so filter the raw instrument list before parsing.
        """
        if self._exchange is None:
            raise ExchangeAPIError("OKX exchange is not initialized")

        self._ensure_rest_url()
        response = await self._exchange.publicGetPublicInstruments({"instType": "SWAP"})
        instruments = response.get("data", []) if isinstance(response, dict) else []
        filtered = [
            item
            for item in instruments
            if item.get("instType") == "SWAP"
            and item.get("state") == "live"
            and item.get("ctType") == "linear"
            and item.get("settleCcy") == "USDT"
            and item.get("instId", "").endswith("-USDT-SWAP")
            and item.get("ctVal")
            and item.get("minSz")
            and item.get("tickSz")
        ]
        markets = self._exchange.parse_markets(filtered)
        self._exchange.set_markets(markets)
        if not self._exchange.markets:
            raise ExchangeAPIError("No OKX USDT swap markets loaded")

    async def reinitialize(self) -> None:
        """Close and re-create exchange with current settings."""
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception as exc:
                logger.debug(
                    "OKX exchange close failed during reinitialize",
                    error=safe_error_text(exc),
                )
        self._exchange = None
        self._connected = False
        await self.initialize()

    async def _with_retry(self, fn, *args, **kwargs):
        """Execute an API call with retry + rate limit handling."""
        import ccxt.async_support as ccxt_async

        last_error = None
        method_name = getattr(fn, "__name__", "")
        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limiter.wait_for_token()
                self._ensure_rest_url()
                result = await asyncio.wait_for(
                    fn(*args, **kwargs),
                    timeout=OKX_REST_CALL_TIMEOUT,
                )
                return result
            except TimeoutError as e:
                logger.warning(
                    "OKX REST call timed out",
                    method=method_name,
                    attempt=attempt,
                    timeout=OKX_REST_CALL_TIMEOUT,
                )
                raise ExchangeAPIError(
                    f"OKX REST call timed out after {OKX_REST_CALL_TIMEOUT:.0f}s: {method_name}"
                ) from e
            except ccxt_async.RateLimitExceeded as e:
                logger.warning("rate limited", attempt=attempt)
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
                last_error = e
            except ccxt_async.NetworkError as e:
                logger.warning("network error", attempt=attempt, error=safe_error_text(e))
                await asyncio.sleep(RETRY_DELAY * (2**attempt))
                last_error = e
            except ccxt_async.ExchangeError as e:
                message = safe_error_text(e)
                logger.error("exchange error", error=message)
                raise ExchangeAPIError(message) from e
            except Exception as e:
                if self._is_broken_rest_url_error(e) and attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "OKX executor REST URL state invalid; reinitializing CCXT client",
                        method=method_name,
                        attempt=attempt,
                        error=safe_error_text(e),
                    )
                    await self.reinitialize()
                    if method_name and self._exchange is not None:
                        fn = getattr(self._exchange, method_name, fn)
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))
                    last_error = e
                    continue
                raise

        raise RateLimitError(f"Max retries exceeded: {safe_error_text(last_error)}")

    async def place_order(
        self,
        decision: DecisionOutput,
        account_id: str | None = None,
        override_balance: float | None = None,
    ) -> ExecutionResult:
        if not self._connected:
            await self.initialize()

        if decision.is_hold:
            return ExecutionResult(
                order_id="hold",
                symbol=decision.symbol,
                side="hold",
                order_type="market",
                quantity=0,
                price=0,
                status=OrderStatus.REJECTED,
                raw_response={"error": "AI 选择观望，未提交 OKX 订单"},
            )

        ccxt = await self._get_ccxt()

        try:
            okx_symbol = self._to_swap_symbol(decision.symbol)
            # Map action to CCXT side
            if decision.action == Action.LONG:
                side = "buy"
            elif decision.action == Action.SHORT:
                side = "sell"
            elif decision.action == Action.CLOSE_LONG:
                side = "sell"
            elif decision.action == Action.CLOSE_SHORT:
                side = "buy"
            else:
                side = "buy"

            # Get balance for position sizing — use override for per-model allocation
            balance = 0.0
            position_value = 0.0
            if decision.is_entry:
                if override_balance is not None and override_balance > 0:
                    balance = override_balance
                else:
                    balance = await self.get_balance()
                position_value = balance * decision.position_size_pct * decision.suggested_leverage

            # Get current ticker for quantity calculation
            ticker = await self._with_retry(ccxt.fetch_ticker, okx_symbol)
            price = ticker.get("last", 0)
            market = ccxt.market(okx_symbol)
            contract_size = self._contract_size(market)
            order_quantity = 0.0
            base_quantity = 0.0
            order_resize_note = None
            target_side = None
            position_side = None
            pre_exit_contracts = 0.0
            requested_exit_contracts = 0.0
            requested_exit_fraction = 1.0
            exit_order_replace_note = None

            if decision.is_entry:
                order_quantity, base_quantity = self._entry_order_amount(
                    ccxt,
                    market,
                    position_value,
                    price,
                    balance,
                    decision.suggested_leverage,
                )
                okx_order_rules = self._entry_order_rule_snapshot(
                    market,
                    price=price,
                    balance=balance,
                    leverage=decision.suggested_leverage,
                    planned_notional_usdt=position_value,
                    final_contracts=order_quantity,
                )
            else:
                okx_order_rules = {}

            if order_quantity <= 0 and decision.is_entry:
                min_notional = self._minimum_order_notional(market, price)
                affordable_notional = balance * max(float(decision.suggested_leverage or 1.0), 1.0)
                return ExecutionResult(
                    order_id="rejected",
                    symbol=decision.symbol,
                    side=side,
                    order_type="market",
                    quantity=0,
                    price=price,
                    status=OrderStatus.REJECTED,
                    raw_response={
                        "error": (
                            "该交易对的 OKX 最小下单张数超过当前可用余额或风险预算，"
                            "系统已在提交前拦截，未向 OKX 发送无效订单。"
                        ),
                        "execution_blocker": "system_pre_submit_order_rule",
                        "system_pre_submit_rejection": True,
                        "okx_rejection": False,
                        "okx_symbol": okx_symbol,
                        "contract_size": contract_size,
                        "okx_order_rules": okx_order_rules,
                        "okx_min_order_notional_usdt": round(min_notional, 8),
                        "affordable_notional_usdt": round(affordable_notional, 8),
                        "planned_order_notional_usdt": round(position_value, 8),
                    },
                )

            if decision.is_entry:
                existing_entry = await self._find_active_entry_order(ccxt, okx_symbol, side)
                if existing_entry:
                    info = existing_entry.get("info") or {}
                    order_id = str(existing_entry.get("id") or info.get("ordId") or "")
                    filled_contracts = self._safe_float(
                        existing_entry.get("filled") or info.get("accFillSz"),
                        0.0,
                    )
                    amount_contracts = self._safe_float(
                        existing_entry.get("amount") or info.get("sz"),
                        order_quantity,
                    )
                    remaining_contracts = max(amount_contracts - filled_contracts, 0.0)
                    status = self._order_status_from_ccxt(
                        existing_entry.get("status") or info.get("state")
                    )
                    if filled_contracts > 0 and status in {OrderStatus.OPEN, OrderStatus.PENDING}:
                        status = OrderStatus.PARTIAL
                    execution_price = float(
                        existing_entry.get("average") or existing_entry.get("price") or price or 0
                    )
                    return ExecutionResult(
                        order_id=order_id or "entry_tracking",
                        symbol=decision.symbol,
                        side=side,
                        order_type="market",
                        quantity=filled_contracts * contract_size if filled_contracts > 0 else 0.0,
                        price=execution_price,
                        status=status,
                        fee=self._order_fee_cost(existing_entry),
                        exchange_order_id=order_id or None,
                        timestamp=datetime.now(UTC),
                        raw_response={
                            **existing_entry,
                            "entry_tracking": True,
                            "existing_entry_order": True,
                            "message": (
                                "OKX 已有同方向开仓委托正在挂单或追单，系统不会重复提交新的开仓单；"
                                "成交后会由 OKX 仓位同步写入本地持仓。"
                            ),
                            "request_params": {"tdMode": "cross"},
                            "okx_symbol": okx_symbol,
                            "contract_size": contract_size,
                            "order_contracts": amount_contracts,
                            "filled_contracts": filled_contracts,
                            "remaining_contracts": remaining_contracts,
                            "planned_order_contracts": order_quantity,
                            "planned_base_quantity": base_quantity,
                            "okx_order_rules": okx_order_rules,
                        },
                    )

            # For closing positions, get the actual position size
            if decision.is_exit:
                positions = await self.get_positions_strict(decision.symbol)
                target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
                matching = [
                    p
                    for p in positions
                    if p.get("side") == target_side and self._position_contracts(p) > 0
                ]
                if matching:
                    pre_exit_contracts = self._position_contracts(matching[0])
                    manual_close = bool(
                        isinstance(decision.raw_response, dict)
                        and decision.raw_response.get("manual_close")
                    )
                    min_exit_fraction = 1e-9 if manual_close else 0.05
                    requested_exit_fraction = min(
                        max(float(decision.position_size_pct or 1.0), min_exit_fraction),
                        1.0,
                    )
                    order_quantity = pre_exit_contracts * requested_exit_fraction
                    amount_min = self._amount_min(market)
                    order_quantity = self._normalize_order_contracts(
                        ccxt, market, order_quantity, amount_min
                    )
                    if amount_min > 0 and order_quantity > pre_exit_contracts:
                        order_quantity = pre_exit_contracts
                    if order_quantity <= 0:
                        return ExecutionResult(
                            order_id="rejected",
                            symbol=decision.symbol,
                            side=side,
                            order_type="market",
                            quantity=0,
                            price=price,
                            status=OrderStatus.REJECTED,
                            raw_response={"error": "平仓数量低于 OKX 最小合约数量，未提交订单。"},
                        )
                    requested_exit_contracts = min(order_quantity, pre_exit_contracts)
                    base_quantity = order_quantity * contract_size
                    position_side = self._okx_position_side(matching[0], target_side)
                else:
                    return ExecutionResult(
                        order_id="no_position",
                        symbol=decision.symbol,
                        side=side,
                        order_type="market",
                        quantity=0,
                        price=price,
                        status=OrderStatus.REJECTED,
                        raw_response={
                            "error": "OKX 当前没有对应方向的可平仓位，本轮未提交平仓订单。"
                        },
                    )

                existing_exit = await self._find_active_exit_order(ccxt, okx_symbol, side)
                if existing_exit:
                    info = existing_exit.get("info") or {}
                    order_id = str(existing_exit.get("id") or info.get("ordId") or "")
                    filled_contracts = self._safe_float(
                        existing_exit.get("filled") or info.get("accFillSz"),
                        0.0,
                    )
                    amount_contracts = self._safe_float(
                        existing_exit.get("amount") or info.get("sz"),
                        order_quantity,
                    )
                    remaining_contracts = max(amount_contracts - filled_contracts, 0.0)
                    order_status = self._order_status_from_ccxt(
                        existing_exit.get("status") or info.get("state")
                    )
                    if filled_contracts > 0 and order_status in {
                        OrderStatus.OPEN,
                        OrderStatus.PENDING,
                    }:
                        order_status = OrderStatus.PARTIAL
                    execution_price = float(
                        existing_exit.get("average") or existing_exit.get("price") or price or 0
                    )
                    order_age = self._order_age_seconds(existing_exit)
                    if order_id and order_age >= EXIT_ORDER_REPLACE_AFTER_SECONDS:
                        cancel_result = await self._cancel_stale_exit_order(
                            ccxt,
                            existing_exit,
                            okx_symbol,
                            order_id,
                            order_age,
                        )
                        if not cancel_result.get("cancel_success"):
                            return ExecutionResult(
                                order_id=order_id or "exit_tracking",
                                symbol=decision.symbol,
                                side=side,
                                order_type="market",
                                quantity=0.0,
                                price=execution_price,
                                status=order_status,
                                fee=self._order_fee_cost(existing_exit),
                                exchange_order_id=order_id or None,
                                timestamp=datetime.now(UTC),
                                raw_response={
                                    **existing_exit,
                                    "exit_tracking": True,
                                    "existing_exit_order": True,
                                    "exit_replace_attempted": True,
                                    "exit_replace_after_seconds": EXIT_ORDER_REPLACE_AFTER_SECONDS,
                                    "exit_order_age_seconds": order_age,
                                    "cancel_success": False,
                                    "cancel_error": cancel_result.get("cancel_error"),
                                    "message": (
                                        f"OKX 已有平仓委托挂单约 {order_age:.0f} 秒仍未完成，"
                                        "系统尝试撤单重提，但 OKX 未确认撤单成功。为避免重复平仓，本轮不再提交第二张平仓单；"
                                        "下一轮会继续检查该委托和仓位状态。"
                                    ),
                                    "request_params": {"tdMode": "cross", "reduceOnly": True},
                                    "okx_symbol": okx_symbol,
                                    "contract_size": contract_size,
                                    "order_contracts": amount_contracts,
                                    "filled_contracts": filled_contracts,
                                    "remaining_contracts": remaining_contracts,
                                    "position_contracts_before": pre_exit_contracts,
                                    "position_contracts_after": pre_exit_contracts,
                                },
                            )

                        await asyncio.sleep(0.5)
                        positions = await self.get_positions_strict(decision.symbol)
                        matching = [
                            p
                            for p in positions
                            if p.get("side") == target_side and self._position_contracts(p) > 0
                        ]
                        if not matching:
                            return ExecutionResult(
                                order_id=order_id,
                                symbol=decision.symbol,
                                side=side,
                                order_type="market",
                                quantity=filled_contracts * contract_size,
                                price=execution_price,
                                status=OrderStatus.FILLED,
                                fee=self._order_fee_cost(existing_exit),
                                exchange_order_id=order_id,
                                timestamp=datetime.now(UTC),
                                raw_response={
                                    **existing_exit,
                                    "exit_tracking": True,
                                    "exit_replace_attempted": True,
                                    "cancel_success": True,
                                    "message": "OKX 原平仓委托已撤销或成交，当前已没有对应方向仓位；本地等待同步确认即可。",
                                    "okx_symbol": okx_symbol,
                                    "contract_size": contract_size,
                                    "filled_contracts": filled_contracts,
                                    "position_contracts_before": pre_exit_contracts,
                                    "position_contracts_after": 0.0,
                                    "remaining_contracts": 0.0,
                                },
                            )

                        pre_exit_contracts = self._position_contracts(matching[0])
                        position_side = self._okx_position_side(matching[0], target_side)
                        leftover_contracts = max(amount_contracts - filled_contracts, 0.0)
                        if requested_exit_fraction >= 0.999:
                            order_quantity = pre_exit_contracts
                        elif leftover_contracts > 0:
                            order_quantity = min(pre_exit_contracts, leftover_contracts)
                        else:
                            order_quantity = pre_exit_contracts * requested_exit_fraction
                        order_quantity = self._normalize_order_contracts(
                            ccxt, market, order_quantity, amount_min
                        )
                        if amount_min > 0 and order_quantity > pre_exit_contracts:
                            order_quantity = pre_exit_contracts
                        if order_quantity <= 0:
                            return ExecutionResult(
                                order_id=order_id,
                                symbol=decision.symbol,
                                side=side,
                                order_type="market",
                                quantity=0.0,
                                price=execution_price,
                                status=OrderStatus.REJECTED,
                                fee=self._order_fee_cost(existing_exit),
                                exchange_order_id=order_id,
                                timestamp=datetime.now(UTC),
                                raw_response={
                                    **existing_exit,
                                    "exit_tracking": True,
                                    "exit_replace_attempted": True,
                                    "cancel_success": True,
                                    "error": "OKX 原平仓委托已撤销，但刷新后剩余可平数量低于最小下单数量，本轮未重新提交。",
                                    "okx_symbol": okx_symbol,
                                    "contract_size": contract_size,
                                    "filled_contracts": filled_contracts,
                                    "remaining_contracts": pre_exit_contracts,
                                },
                            )
                        base_quantity = order_quantity * contract_size
                        requested_exit_contracts = min(order_quantity, pre_exit_contracts)
                        exit_order_replace_note = (
                            f"OKX 原平仓委托 {order_id} 已挂单约 {order_age:.0f} 秒仍未完成，"
                            "系统已先撤掉旧委托，并按最新剩余仓位重新提交 reduce-only 市价平仓。"
                        )
                    else:
                        wait_seconds = max(EXIT_ORDER_REPLACE_AFTER_SECONDS - order_age, 0.0)
                        wait_note = (
                            f"OKX 已有平仓订单正在追单或部分成交，已等待约 {order_age:.0f} 秒；"
                            f"若约 {wait_seconds:.0f} 秒后仍未成交，系统会撤单并按最新仓位重提平仓。"
                        )
                        return ExecutionResult(
                            order_id=order_id or "exit_tracking",
                            symbol=decision.symbol,
                            side=side,
                            order_type="market",
                            quantity=0.0,
                            price=execution_price,
                            status=order_status,
                            fee=self._order_fee_cost(existing_exit),
                            exchange_order_id=order_id or None,
                            timestamp=datetime.now(UTC),
                            raw_response={
                                **existing_exit,
                                "exit_tracking": True,
                                "existing_exit_order": True,
                                "exit_replace_after_seconds": EXIT_ORDER_REPLACE_AFTER_SECONDS,
                                "exit_order_age_seconds": order_age,
                                "message": wait_note,
                                "request_params": {"tdMode": "cross", "reduceOnly": True},
                                "okx_symbol": okx_symbol,
                                "contract_size": contract_size,
                                "order_contracts": amount_contracts,
                                "filled_contracts": filled_contracts,
                                "remaining_contracts": remaining_contracts,
                                "position_contracts_before": pre_exit_contracts,
                                "position_contracts_after": pre_exit_contracts,
                            },
                        )
            params: dict[str, Any] = {"tdMode": "cross"}

            if decision.is_entry:
                quantity_leverage = self._safe_float(decision.suggested_leverage, 1.0)
                leverage_check = await self._set_leverage_if_needed(decision)
                if not leverage_check.get("ok"):
                    return ExecutionResult(
                        order_id="rejected",
                        symbol=decision.symbol,
                        side=side,
                        order_type="market",
                        quantity=0,
                        price=price,
                        status=OrderStatus.REJECTED,
                        raw_response={
                            "error": leverage_check.get("error")
                            or "OKX 杠杆设置失败，本次未开仓。",
                            "leverage_check": leverage_check,
                            "okx_symbol": okx_symbol,
                            "contract_size": contract_size,
                            "planned_order_contracts": order_quantity,
                            "planned_base_quantity": base_quantity,
                            "okx_order_rules": okx_order_rules,
                        },
                    )
                actual_leverage = self._safe_float(
                    leverage_check.get("actual_leverage")
                    or leverage_check.get("target_leverage")
                    or decision.suggested_leverage,
                    decision.suggested_leverage,
                )
                if actual_leverage > 0 and abs(actual_leverage - quantity_leverage) > 1e-9:
                    decision.suggested_leverage = actual_leverage
                    position_value = (
                        balance * decision.position_size_pct * decision.suggested_leverage
                    )
                    order_quantity, base_quantity = self._entry_order_amount(
                        ccxt,
                        market,
                        position_value,
                        price,
                        balance,
                        decision.suggested_leverage,
                    )
                    okx_order_rules = self._entry_order_rule_snapshot(
                        market,
                        price=price,
                        balance=balance,
                        leverage=decision.suggested_leverage,
                        planned_notional_usdt=position_value,
                        final_contracts=order_quantity,
                    )
                    if order_quantity <= 0:
                        return ExecutionResult(
                            order_id="rejected",
                            symbol=decision.symbol,
                            side=side,
                            order_type="market",
                            quantity=0,
                            price=price,
                            status=OrderStatus.REJECTED,
                            raw_response={
                                "error": (
                                    "杠杆按 OKX 上限回退后，该交易对最小下单张数仍超过当前可用余额或风险预算，"
                                    "系统已在提交前拦截，未向 OKX 发送无效订单。"
                                ),
                                "execution_blocker": "system_pre_submit_order_rule",
                                "system_pre_submit_rejection": True,
                                "okx_rejection": False,
                                "leverage_check": leverage_check,
                                "okx_symbol": okx_symbol,
                                "contract_size": contract_size,
                                "planned_order_contracts": order_quantity,
                                "planned_base_quantity": base_quantity,
                                "okx_order_rules": okx_order_rules,
                            },
                        )
                stop_loss_px, take_profit_px = self._attached_sl_tp_prices(
                    decision,
                    price,
                    ticker=ticker,
                )
                protection = self._format_attached_sl_tp_prices(
                    ccxt,
                    okx_symbol,
                    decision,
                    stop_loss_px,
                    take_profit_px,
                    price,
                )
                if not protection.get("ok"):
                    return ExecutionResult(
                        order_id="rejected",
                        symbol=decision.symbol,
                        side=side,
                        order_type="market",
                        quantity=0,
                        price=price,
                        status=OrderStatus.REJECTED,
                        raw_response=protection,
                    )
                params["attachAlgoOrds"] = [
                    {
                        "tpTriggerPx": protection["take_profit_price"],
                        "tpOrdPx": "-1",
                        "slTriggerPx": protection["stop_loss_price"],
                        "slOrdPx": "-1",
                    }
                ]
            elif decision.is_exit:
                if position_side:
                    params["positionSide"] = position_side
                params["reduceOnly"] = True

            # Place the market order
            logger.info(
                "placing order",
                symbol=decision.symbol,
                okx_symbol=okx_symbol,
                side=side,
                quantity=order_quantity,
                base_quantity=base_quantity,
                contract_size=contract_size,
                action=decision.action.value,
                position_side=position_side,
                mode=self.executor_mode,
                requested_leverage=decision.suggested_leverage,
            )

            try:
                order = await self._with_retry(
                    ccxt.create_order,
                    okx_symbol,
                    "market",
                    side,
                    order_quantity,
                    None,
                    params,
                )
            except ExchangeAPIError as e:
                if decision.is_entry:
                    error_text = safe_error_text(e)
                    capped_quantity = self._capped_quantity_from_position_limit_error(
                        error_text,
                        ccxt,
                        market,
                        order_quantity,
                    )
                    if capped_quantity and capped_quantity < order_quantity:
                        order_resize_note = (
                            f"OKX 当前杠杆下最大允许 {capped_quantity:g} 张，"
                            f"系统已将原计划 {order_quantity:g} 张自动缩小后重试。"
                        )
                        logger.warning(
                            "entry order size capped by OKX position limit",
                            symbol=decision.symbol,
                            okx_symbol=okx_symbol,
                            original_quantity=order_quantity,
                            capped_quantity=capped_quantity,
                            error=error_text,
                        )
                        order_quantity = capped_quantity
                        base_quantity = order_quantity * contract_size
                        order = await self._with_retry(
                            ccxt.create_order,
                            okx_symbol,
                            "market",
                            side,
                            order_quantity,
                            None,
                            params,
                        )
                    else:
                        raise
                else:
                    raise
            order = await self._confirm_market_order(ccxt, order, okx_symbol)
            filled_contracts = self._safe_float(order.get("filled"), 0.0)
            execution_price = float(order.get("average") or order.get("price") or price or 0)
            status = self._order_status_from_ccxt(
                order.get("status") or (order.get("info") or {}).get("state")
            )
            if decision.is_entry and status == OrderStatus.FILLED and filled_contracts <= 0:
                info = order.get("info") or {}
                order_id = str(order.get("id") or info.get("ordId") or "").strip()
                logger.warning(
                    "entry order reported filled without a filled quantity; tracking until OKX position sync confirms it",
                    order_id=order_id,
                    symbol=okx_symbol,
                    requested_contracts=order_quantity,
                    order_status=status.value,
                )
                return ExecutionResult(
                    order_id=order_id or "entry_fill_quantity_missing",
                    symbol=decision.symbol,
                    side=side,
                    order_type="market",
                    quantity=0.0,
                    price=execution_price,
                    status=OrderStatus.PENDING,
                    fee=self._order_fee_cost(order),
                    exchange_order_id=order_id or None,
                    timestamp=datetime.now(UTC),
                    raw_response={
                        **order,
                        "entry_tracking": True,
                        "fill_quantity_missing": True,
                        "message": (
                            "OKX 订单状态显示已完成，但没有返回大于 0 的真实成交数量；"
                            "系统不会用下单数量冒充成交数量，也不会先创建本地持仓，"
                            "等待下一轮 OKX 订单/仓位同步确认。"
                        ),
                        "request_params": params,
                        "okx_symbol": okx_symbol,
                        "contract_size": contract_size,
                        "order_contracts": order_quantity,
                        "filled_contracts": filled_contracts,
                        "planned_base_quantity": base_quantity,
                        "okx_order_rules": okx_order_rules,
                        "order_status": status.value,
                    },
                )
            filled_base_quantity = filled_contracts * contract_size

            if decision.is_exit and target_side:
                after_contracts = pre_exit_contracts
                for _ in range(5):
                    await asyncio.sleep(1.0)
                    try:
                        refreshed = await self._with_retry(
                            ccxt.fetch_order,
                            order.get("id"),
                            okx_symbol,
                        )
                        if refreshed:
                            order = {**order, **refreshed}
                            status = self._order_status_from_ccxt(
                                order.get("status") or (order.get("info") or {}).get("state")
                            )
                            filled_contracts = max(
                                filled_contracts,
                                self._safe_float(order.get("filled"), 0.0),
                            )
                            execution_price = float(
                                order.get("average")
                                or order.get("price")
                                or execution_price
                                or price
                                or 0
                            )
                    except Exception as e:
                        logger.warning(
                            "exit order refresh failed",
                            order_id=order.get("id"),
                            symbol=okx_symbol,
                            error=safe_error_text(e),
                        )

                    try:
                        after_contracts = await self._position_contracts_for_side(
                            decision.symbol,
                            target_side,
                        )
                    except Exception as e:
                        error_text = safe_error_text(e)
                        logger.warning(
                            "exit position refresh failed after order submit",
                            order_id=order.get("id"),
                            symbol=okx_symbol,
                            side=target_side,
                            error=error_text,
                        )
                        order_id = str(order.get("id") or "").strip()
                        return ExecutionResult(
                            order_id=order_id or "exit_position_snapshot_unknown",
                            symbol=decision.symbol,
                            side=side,
                            order_type="market",
                            quantity=0.0,
                            price=execution_price,
                            status=(
                                OrderStatus.PARTIAL if filled_contracts > 0 else OrderStatus.PENDING
                            ),
                            fee=self._order_fee_cost(order),
                            exchange_order_id=order_id or None,
                            timestamp=datetime.now(UTC),
                            raw_response={
                                **order,
                                "exit_tracking": True,
                                "position_snapshot_unknown": True,
                                "position_snapshot_error": error_text,
                                "message": (
                                    "OKX 平仓订单已提交，但系统暂时无法刷新 OKX 剩余仓位；"
                                    "本地不估算剩余仓位为 0，等待下一轮同步确认成交和持仓状态。"
                                ),
                                "exit_order_replace_note": exit_order_replace_note,
                                "request_params": params,
                                "okx_symbol": okx_symbol,
                                "contract_size": contract_size,
                                "order_contracts": order_quantity,
                                "order_status": status.value,
                                "filled_contracts": filled_contracts,
                                "position_contracts_before": pre_exit_contracts,
                                "position_contracts_after": None,
                                "remaining_contracts": None,
                            },
                        )
                    if pre_exit_contracts - after_contracts > max(pre_exit_contracts * 0.001, 1e-8):
                        break
                    if status in {OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                        break

                closed_contracts = max(pre_exit_contracts - after_contracts, 0.0)
                tolerance = max(pre_exit_contracts * 0.001, 1e-8)
                if closed_contracts <= tolerance:
                    order_id = str(order.get("id") or "").strip()
                    active_tracking_status = status in {
                        OrderStatus.OPEN,
                        OrderStatus.PENDING,
                        OrderStatus.PARTIAL,
                    }
                    if order_id and active_tracking_status:
                        return ExecutionResult(
                            order_id=order_id,
                            symbol=decision.symbol,
                            side=side,
                            order_type="market",
                            quantity=0.0,
                            price=execution_price,
                            status=(
                                OrderStatus.PARTIAL
                                if filled_contracts > tolerance
                                else OrderStatus.OPEN
                            ),
                            fee=self._order_fee_cost(order),
                            exchange_order_id=order_id,
                            timestamp=datetime.now(UTC),
                            raw_response={
                                **order,
                                "exit_tracking": True,
                                "message": (
                                    f"{exit_order_replace_note} 新平仓单仍在追单或等待成交；系统会继续同步。"
                                    if exit_order_replace_note
                                    else "OKX 平仓订单已提交，但仍在追单或等待成交；系统会继续同步，不会重复提交平仓单。"
                                ),
                                "exit_order_replace_note": exit_order_replace_note,
                                "request_params": params,
                                "okx_symbol": okx_symbol,
                                "contract_size": contract_size,
                                "order_contracts": order_quantity,
                                "order_status": status.value,
                                "filled_contracts": filled_contracts,
                                "position_contracts_before": pre_exit_contracts,
                                "position_contracts_after": after_contracts,
                                "remaining_contracts": max(after_contracts, 0.0),
                            },
                        )
                    return ExecutionResult(
                        order_id=order_id or "exit_not_confirmed",
                        symbol=decision.symbol,
                        side=side,
                        order_type="market",
                        quantity=0,
                        price=execution_price,
                        status=OrderStatus.REJECTED,
                        fee=(
                            order.get("fee", {}).get("cost", 0)
                            if isinstance(order.get("fee"), dict)
                            else 0
                        ),
                        exchange_order_id=order_id or None,
                        timestamp=datetime.now(UTC),
                        raw_response={
                            **order,
                            "error": (
                                "OKX 未完成这笔平仓单，成交数量为 0 或仓位没有减少；"
                                "本地不会把这笔仓位记为已平仓。"
                            ),
                            "request_params": params,
                            "okx_symbol": okx_symbol,
                            "contract_size": contract_size,
                            "order_contracts": order_quantity,
                            "order_status": status.value,
                            "filled_contracts": filled_contracts,
                            "position_contracts_before": pre_exit_contracts,
                            "position_contracts_after": after_contracts,
                            "cancel_attempted": False,
                            "cancel_success": False,
                            "cancel_error": None,
                        },
                    )

                filled_contracts = closed_contracts
                filled_base_quantity = filled_contracts * contract_size
                requested_filled = (
                    requested_exit_contracts <= 0
                    or closed_contracts + tolerance >= requested_exit_contracts
                )
                status = OrderStatus.FILLED if requested_filled else OrderStatus.PARTIAL

            return ExecutionResult(
                order_id=order.get("id", ""),
                symbol=decision.symbol,
                side=side,
                order_type="market",
                quantity=(
                    filled_base_quantity
                    if filled_base_quantity > 0
                    else (base_quantity if status == OrderStatus.FILLED else 0.0)
                ),
                price=execution_price,
                status=status,
                fee=(
                    order.get("fee", {}).get("cost", 0) if isinstance(order.get("fee"), dict) else 0
                ),
                exchange_order_id=order.get("id"),
                timestamp=datetime.now(UTC),
                raw_response={
                    **order,
                    "request_params": params,
                    "okx_symbol": okx_symbol,
                    "contract_size": contract_size,
                    "order_contracts": order_quantity,
                    "filled_contracts": filled_contracts,
                    "okx_order_rules": okx_order_rules,
                    "base_quantity": (
                        filled_base_quantity
                        if filled_base_quantity > 0
                        else (base_quantity if status == OrderStatus.FILLED else 0.0)
                    ),
                    "order_resize_note": order_resize_note,
                    "exit_order_replace_note": exit_order_replace_note,
                    **(
                        {
                            "entry_tracking": True,
                            "message": (
                                "OKX 开仓订单已提交，但仍在挂单或追单，尚未确认成交；"
                                "本地不会先创建持仓，成交后会由 OKX 仓位同步补回。"
                            ),
                        }
                        if decision.is_entry
                        and status
                        in {
                            OrderStatus.OPEN,
                            OrderStatus.PENDING,
                            OrderStatus.PARTIAL,
                        }
                        else {}
                    ),
                    **({"leverage_check": leverage_check} if decision.is_entry else {}),
                    **(
                        {
                            "exit_tracking": True,
                            "position_contracts_before": pre_exit_contracts,
                            "position_contracts_after": after_contracts,
                            "requested_exit_fraction": requested_exit_fraction,
                            "requested_exit_contracts": requested_exit_contracts,
                            "remaining_contracts": max(after_contracts, 0.0),
                            "message": (
                                f"{exit_order_replace_note} OKX 平仓已全部成交。"
                                if exit_order_replace_note
                                and status == OrderStatus.FILLED
                                and requested_exit_fraction >= 0.999
                                else (
                                    f"{exit_order_replace_note} OKX 已按计划减仓 {requested_exit_fraction:.0%}，剩余仓位继续同步追踪。"
                                    if exit_order_replace_note and status == OrderStatus.FILLED
                                    else (
                                        (
                                            "OKX 平仓已全部成交。"
                                            if requested_exit_fraction >= 0.999
                                            else f"OKX 已按计划减仓 {requested_exit_fraction:.0%}，剩余仓位继续同步追踪。"
                                        )
                                        if status == OrderStatus.FILLED
                                        else "OKX 平仓已部分成交，剩余仓位继续同步追踪。"
                                    )
                                )
                            ),
                        }
                        if decision.is_exit
                        else {}
                    ),
                },
            )

        except ExchangeAPIError as e:
            error_text = safe_error_text(e)
            if decision.is_exit and self._is_no_position_error(error_text):
                return ExecutionResult(
                    order_id="no_position",
                    symbol=decision.symbol,
                    side=side,
                    order_type="market",
                    quantity=0,
                    price=0,
                    status=OrderStatus.REJECTED,
                    raw_response={
                        "error": "OKX 提示当前没有对应方向的可平仓位，可能已被 OKX 止盈/止损、手动平仓或刚刚同步延迟；本轮未重复提交。",
                        "raw_error": error_text,
                    },
                )
            raise
        except RateLimitError:
            raise
        except Exception as e:
            error_text = safe_error_text(e)
            logger.error("order placement failed", error=error_text)
            raise OrderPlacementError(f"Failed to place order: {error_text}") from e

    async def _confirm_market_order(self, ccxt, order: dict, symbol: str) -> dict:
        """Fetch the final OKX order state after market order submission."""
        order_id = order.get("id")
        if not order_id:
            return order
        if order.get("status") in {"closed", "canceled", "cancelled", "rejected"}:
            return order

        await asyncio.sleep(0.5)
        try:
            confirmed = await self._with_retry(ccxt.fetch_order, order_id, symbol)
            if confirmed:
                return {**order, **confirmed}
        except Exception as e:
            logger.warning(
                "order confirmation failed; keeping initial order status",
                order_id=order_id,
                symbol=symbol,
                error=safe_error_text(e),
            )
        return order

    def _order_status_from_ccxt(self, status: str | None) -> OrderStatus:
        status_map = {
            "closed": OrderStatus.FILLED,
            "filled": OrderStatus.FILLED,
            "open": OrderStatus.OPEN,
            "pending": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL,
            "partial": OrderStatus.PARTIAL,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }
        return status_map.get(str(status or "").lower(), OrderStatus.OPEN)

    def _contract_size(self, market: dict[str, Any]) -> float:
        try:
            value = float(market.get("contractSize") or 1.0)
            return value if value > 0 else 1.0
        except (TypeError, ValueError):
            return 1.0

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _position_contracts(self, position: dict[str, Any]) -> float:
        info = position.get("info") or {}
        raw_size = (
            position.get("contracts")
            or position.get("size")
            or position.get("positionAmt")
            or info.get("pos")
            or info.get("qty")
            or 0
        )
        try:
            return abs(float(raw_size))
        except (TypeError, ValueError):
            return 0.0

    def _okx_position_side(self, position: dict[str, Any], target_side: str) -> str:
        """Return the OKX position side to submit for a close order."""
        info = position.get("info") or {}
        raw_side = str(info.get("posSide") or "").lower().strip()
        if raw_side in {"long", "short", "net"}:
            return raw_side
        return target_side

    async def _position_contracts_for_side(self, symbol: str, side: str) -> float:
        positions = await self.get_positions_strict(symbol)
        for position in positions or []:
            if position.get("side") == side:
                return self._position_contracts(position)
        return 0.0

    async def _find_active_exit_order(self, ccxt, okx_symbol: str, side: str) -> dict | None:
        """Return an active reduce-only close order for the same symbol and side."""
        try:
            orders = await self._with_retry(ccxt.fetch_open_orders, okx_symbol)
        except Exception as e:
            logger.warning(
                "fetch open exit orders failed",
                symbol=okx_symbol,
                side=side,
                error=safe_error_text(e),
            )
            return None

        for order in orders or []:
            info = order.get("info") or {}
            order_side = str(order.get("side") or info.get("side") or "").lower()
            if order_side != side:
                continue

            reduce_only = order.get("reduceOnly")
            if reduce_only in (None, ""):
                reduce_only = info.get("reduceOnly")
            if str(reduce_only).lower() != "true":
                continue

            status = self._order_status_from_ccxt(order.get("status") or info.get("state"))
            if status in {OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIAL}:
                return order
        return None

    async def _find_active_entry_order(self, ccxt, okx_symbol: str, side: str) -> dict | None:
        """Return an active non-reduce-only entry order for the same symbol/side."""
        try:
            orders = await self._with_retry(ccxt.fetch_open_orders, okx_symbol)
        except Exception as e:
            logger.warning(
                "fetch open entry orders failed",
                symbol=okx_symbol,
                side=side,
                error=safe_error_text(e),
            )
            return None

        for order in orders or []:
            info = order.get("info") or {}
            order_side = str(order.get("side") or info.get("side") or "").lower()
            if order_side != side:
                continue

            reduce_only = order.get("reduceOnly")
            if reduce_only in (None, ""):
                reduce_only = info.get("reduceOnly")
            if str(reduce_only).lower() == "true":
                continue

            ord_type = str(info.get("ordType") or order.get("type") or "").lower()
            if ord_type in {"oco", "conditional", "trigger"}:
                continue

            status = self._order_status_from_ccxt(order.get("status") or info.get("state"))
            if status in {OrderStatus.OPEN, OrderStatus.PENDING, OrderStatus.PARTIAL}:
                return order
        return None

    def _order_age_seconds(self, order: dict[str, Any]) -> float:
        """Best-effort age for an OKX order from CCXT normalized/raw timestamps."""
        info = order.get("info") or {}
        raw_ts = (
            order.get("timestamp")
            or info.get("cTime")
            or info.get("uTime")
            or info.get("createdTime")
            or info.get("created_at")
        )
        try:
            ts = self._safe_float(raw_ts, 0.0)
            if ts <= 0:
                raise ValueError("missing timestamp")
            if ts > 1_000_000_000_000:
                ts = ts / 1000.0
            return max(time.time() - ts, 0.0)
        except Exception as exc:
            logger.debug(
                "failed to parse OKX order timestamp",
                raw_timestamp=raw_ts,
                error=safe_error_text(exc),
            )

        raw_dt = order.get("datetime") or info.get("datetime")
        if raw_dt:
            try:
                dt = datetime.fromisoformat(str(raw_dt).replace("Z", "+00:00"))
                return max((datetime.now(UTC) - dt.astimezone(UTC)).total_seconds(), 0.0)
            except Exception as exc:
                logger.debug(
                    "failed to parse OKX order datetime",
                    raw_datetime=raw_dt,
                    error=safe_error_text(exc),
                )
        return EXIT_ORDER_REPLACE_AFTER_SECONDS

    async def _cancel_stale_exit_order(
        self,
        ccxt,
        order: dict[str, Any],
        okx_symbol: str,
        order_id: str,
        order_age: float,
    ) -> dict[str, Any]:
        """Cancel a stale close order before submitting a fresher reduce-only close."""
        try:
            await self._with_retry(ccxt.cancel_order, order_id, okx_symbol)
            logger.warning(
                "stale OKX exit order cancelled for replace",
                symbol=okx_symbol,
                order_id=order_id,
                age_seconds=order_age,
            )
            return {"cancel_success": True}
        except Exception as e:
            error_text = safe_error_text(e, limit=300)
            logger.warning(
                "failed to cancel stale OKX exit order for replace",
                symbol=okx_symbol,
                order_id=order_id,
                age_seconds=order_age,
                error=error_text,
            )
            return {"cancel_success": False, "cancel_error": error_text}

    def _order_fee_cost(self, order: dict) -> float:
        fee = order.get("fee")
        if isinstance(fee, dict):
            return self._safe_float(fee.get("cost"), 0.0)
        info_fee = (order.get("info") or {}).get("fee")
        return abs(self._safe_float(info_fee, 0.0))

    def _is_no_position_error(self, message: str) -> bool:
        lowered = str(message or "").lower()
        return (
            "51169" in lowered
            or "don't have any positions in this direction" in lowered
            or "no matching position to close" in lowered
        )

    def _entry_order_amount(
        self,
        ccxt,
        market: dict[str, Any],
        position_value: float,
        price: float,
        balance: float,
        leverage: float,
    ) -> tuple[float, float]:
        """Return (contracts, base_quantity) for OKX swap orders.

        CCXT/OKX expects perpetual swap amounts in contracts, while the rest of
        the app tracks position quantity in base coin units for PnL display.
        """
        if price <= 0 or position_value <= 0:
            return 0.0, 0.0

        contract_size = self._contract_size(market)
        planned_contracts = position_value / (price * contract_size)
        amount_min = self._amount_min(market)
        min_contracts = amount_min if amount_min > 0 else 0.0
        contracts = max(planned_contracts, min_contracts)

        # OKX enforces minSz/lotSz in contracts.  Entry sizing owns risk, but
        # exchange validity owns order shape: if the intended order is below the
        # hard minimum and the account can afford that minimum, lift to the
        # minimum before submission instead of letting OKX reject the order.
        min_notional = min_contracts * contract_size * price
        if min_contracts > 0 and planned_contracts < min_contracts:
            affordable_notional = balance * max(float(leverage or 1.0), 1.0)
            if min_notional > affordable_notional:
                return 0.0, 0.0

        try:
            contracts = float(ccxt.amount_to_precision(market["symbol"], contracts))
        except Exception as exc:
            logger.debug(
                "OKX amount precision failed for position size",
                symbol=market.get("symbol"),
                contracts=contracts,
                error=safe_error_text(exc),
            )
        contracts = self._normalize_order_contracts(ccxt, market, contracts, min_contracts)

        return contracts, contracts * contract_size

    def _entry_order_rule_snapshot(
        self,
        market: dict[str, Any],
        *,
        price: float,
        balance: float,
        leverage: float,
        planned_notional_usdt: float,
        final_contracts: float,
    ) -> dict[str, Any]:
        contract_size = self._contract_size(market)
        amount_min = self._amount_min(market)
        amount_step = self._amount_step(market)
        planned_contracts = (
            planned_notional_usdt / (price * contract_size)
            if price > 0 and contract_size > 0 and planned_notional_usdt > 0
            else 0.0
        )
        min_notional = amount_min * contract_size * price if price > 0 else 0.0
        final_notional = max(final_contracts, 0.0) * contract_size * max(price, 0.0)
        effective_leverage = max(float(leverage or 1.0), 1.0)
        return {
            "okx_symbol": market.get("symbol"),
            "price": round(max(price, 0.0), 12),
            "contract_size": round(contract_size, 12),
            "amount_min_contracts": round(amount_min, 12),
            "amount_step_contracts": round(amount_step, 12),
            "min_notional_usdt": round(min_notional, 8),
            "available_balance_usdt": round(max(balance, 0.0), 8),
            "leverage": round(effective_leverage, 6),
            "affordable_notional_usdt": round(max(balance, 0.0) * effective_leverage, 8),
            "planned_notional_usdt": round(max(planned_notional_usdt, 0.0), 8),
            "planned_contracts_raw": round(max(planned_contracts, 0.0), 12),
            "final_contracts": round(max(final_contracts, 0.0), 12),
            "final_base_quantity": round(max(final_contracts, 0.0) * contract_size, 12),
            "final_notional_usdt": round(final_notional, 8),
            "required_margin_usdt": round(final_notional / effective_leverage, 8),
            "system_adjusted_to_min_contracts": bool(
                amount_min > 0 and 0 < planned_contracts < amount_min <= final_contracts
            ),
            "pre_submit_valid": bool(
                final_contracts > 0 and (amount_min <= 0 or final_contracts >= amount_min)
            ),
        }

    def _minimum_order_notional(self, market: dict[str, Any], price: float) -> float:
        min_contracts = self._amount_min(market)
        contract_size = self._contract_size(market)
        if min_contracts <= 0 or contract_size <= 0 or price <= 0:
            return 0.0
        return min_contracts * contract_size * price

    def _capped_quantity_from_position_limit_error(
        self,
        message: str,
        ccxt,
        market: dict[str, Any],
        current_quantity: float,
    ) -> float | None:
        """Parse OKX 51004 max-position errors and return a retry quantity."""
        text = str(message or "")
        if "51004" not in text and "maximum position amount" not in text.lower():
            return None
        match = re.search(r"more than\s+([\d,]+(?:\.\d+)?)\s*\(contracts\)", text, re.IGNORECASE)
        if not match:
            return None
        try:
            max_contracts = float(match.group(1).replace(",", ""))
        except (TypeError, ValueError):
            return None
        if max_contracts <= 0 or max_contracts >= current_quantity:
            return None

        # Leave a small buffer so rounding and pending-order race conditions do not hit the cap again.
        capped = max_contracts * 0.98
        try:
            capped = float(ccxt.amount_to_precision(market["symbol"], capped))
        except Exception as exc:
            logger.debug(
                "OKX amount precision failed for capped quantity",
                symbol=market.get("symbol"),
                capped=capped,
                error=safe_error_text(exc),
            )
        min_contracts = self._amount_min(market)
        if min_contracts > 0 and capped < min_contracts:
            return None
        return capped if capped > 0 else None

    def _amount_min(self, market: dict[str, Any]) -> float:
        values = [
            self._safe_float(((market.get("limits") or {}).get("amount") or {}).get("min"), 0.0),
            self._market_info_float(market, "minSz", "min_size", "minSize"),
            self._amount_step(market),
        ]
        return max((value for value in values if value > 0), default=0.0)

    def _amount_step(self, market: dict[str, Any]) -> float:
        info_step = self._market_info_float(market, "lotSz", "stepSize", "amount_step")
        if info_step > 0:
            return info_step
        precision_amount = self._safe_float((market.get("precision") or {}).get("amount"), 0.0)
        if 0 < precision_amount <= 1:
            return precision_amount
        return 0.0

    def _market_info_float(self, market: dict[str, Any], *keys: str) -> float:
        info = market.get("info") or {}
        for key in keys:
            value = self._safe_float(info.get(key), 0.0)
            if value > 0:
                return value
        return 0.0

    def _normalize_order_contracts(
        self,
        ccxt,
        market: dict[str, Any],
        contracts: float,
        min_contracts: float | None = None,
    ) -> float:
        minimum = min_contracts if min_contracts is not None else self._amount_min(market)
        if contracts <= 0 and minimum <= 0:
            return 0.0
        target = max(float(contracts), float(minimum or 0.0))
        try:
            normalized = float(ccxt.amount_to_precision(market["symbol"], target))
        except Exception as exc:
            logger.debug(
                "OKX amount precision failed for normalized order size",
                symbol=market.get("symbol"),
                contracts=target,
                error=safe_error_text(exc),
            )
            normalized = target

        if normalized <= 0 or (minimum > 0 and normalized < minimum):
            normalized = self._ceil_to_amount_step(max(target, minimum), market)
        return normalized if normalized > 0 else 0.0

    def _ceil_to_amount_step(self, amount: float, market: dict[str, Any]) -> float:
        step = self._amount_step(market)
        if amount <= 0 or step <= 0:
            return amount
        try:
            amount_decimal = Decimal(str(amount))
            step_decimal = Decimal(str(step))
            units = (amount_decimal / step_decimal).to_integral_value(rounding=ROUND_CEILING)
            return float(units * step_decimal)
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return amount

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        ccxt = await self._get_ccxt()
        try:
            await self._with_retry(ccxt.cancel_order, order_id, symbol)
            return True
        except Exception as e:
            logger.error("cancel order failed", order_id=order_id, error=safe_error_text(e))
            return False

    def _leverage_cache_key(self, okx_symbol: str, params: dict[str, Any]) -> tuple[str, str]:
        return (okx_symbol, str(params.get("mgnMode") or params.get("tdMode") or "cross"))

    def _cache_leverage(self, okx_symbol: str, params: dict[str, Any], leverage: float) -> None:
        if leverage > 0:
            self._leverage_cache[self._leverage_cache_key(okx_symbol, params)] = (
                float(leverage),
                time.monotonic(),
            )

    async def _fetch_current_leverage(
        self,
        ccxt,
        okx_symbol: str,
        params: dict[str, Any],
        *,
        max_age_seconds: float = 45.0,
        force: bool = False,
    ) -> tuple[float, dict[str, Any] | None]:
        key = self._leverage_cache_key(okx_symbol, params)
        cached = self._leverage_cache.get(key)
        if not force and cached and time.monotonic() - cached[1] <= max_age_seconds:
            return cached[0], None
        response = await self._with_retry(ccxt.fetch_leverage, okx_symbol, params)
        actual = self._extract_verified_leverage(response)
        self._cache_leverage(okx_symbol, params, actual)
        return actual, response

    def _is_leverage_open_order_limit_error(self, error: Any) -> bool:
        text = str(error or "").lower()
        return (
            "59670" in text
            or "more than 5 open orders" in text
            or "cancel to reduce your orders to 5" in text
        )

    def _is_safe_to_cancel_for_leverage_retry(self, order: dict[str, Any]) -> bool:
        info = order.get("info") or {}
        reduce_only = order.get("reduceOnly")
        if reduce_only in (None, ""):
            reduce_only = info.get("reduceOnly")
        if str(reduce_only).lower() == "true":
            return False
        algo_id = info.get("algoId") or info.get("algoClOrdId")
        if algo_id:
            return False
        filled = self._safe_float(order.get("filled") or info.get("accFillSz"), 0.0)
        if filled > 0:
            return False
        status = self._order_status_from_ccxt(order.get("status") or info.get("state"))
        return status in {OrderStatus.OPEN, OrderStatus.PENDING}

    async def _reduce_open_orders_for_leverage_retry(self, ccxt, okx_symbol: str) -> dict[str, Any]:
        """Cancel stale non-reduce-only entry orders only when OKX blocks leverage changes."""
        try:
            orders = await self._with_retry(ccxt.fetch_open_orders, okx_symbol)
        except Exception as e:
            return {
                "checked": False,
                "cancelled": 0,
                "remaining": None,
                "error": safe_error_text(e, limit=220),
            }

        safe_orders = [
            order for order in orders or [] if self._is_safe_to_cancel_for_leverage_retry(order)
        ]
        safe_orders.sort(
            key=lambda order: self._safe_float(
                order.get("timestamp") or (order.get("info") or {}).get("cTime") or 0,
                0.0,
            )
        )
        cancel_count = max(len(orders or []) - 5, 0)
        cancelled = 0
        errors: list[str] = []
        for order in safe_orders[:cancel_count]:
            order_id = str(order.get("id") or (order.get("info") or {}).get("ordId") or "")
            if not order_id:
                continue
            try:
                await self._with_retry(ccxt.cancel_order, order_id, okx_symbol)
                cancelled += 1
            except Exception as e:
                errors.append(safe_error_text(e, limit=160))

        remaining = max(len(orders or []) - cancelled, 0)
        return {
            "checked": True,
            "open_orders": len(orders or []),
            "safe_cancel_candidates": len(safe_orders),
            "cancelled": cancelled,
            "remaining": remaining,
            "errors": errors[:3],
        }

    async def _fetch_okx_max_leverage(
        self,
        okx_symbol: str,
        params: dict[str, Any],
        requested_leverage: int,
    ) -> float:
        """Return OKX's max leverage for this swap when available."""
        ccxt = await self._get_ccxt()
        values: list[float] = []
        try:
            tiers = await self._with_retry(ccxt.fetch_market_leverage_tiers, okx_symbol)
            if isinstance(tiers, list):
                for tier in tiers:
                    if isinstance(tier, dict):
                        values.append(
                            self._safe_float(
                                tier.get("maxLeverage") or tier.get("max_leverage"), 0.0
                            )
                        )
        except Exception as e:
            logger.debug(
                "fetch leverage tiers failed",
                symbol=okx_symbol,
                error=safe_error_text(e),
            )

        try:
            market = ccxt.market(okx_symbol)
            inst_id = (market.get("info") or {}).get("instId") or okx_symbol.replace(
                "/", "-"
            ).replace(":USDT", "-SWAP")
            estimate = await self._with_retry(
                ccxt.privateGetAccountAdjustLeverageInfo,
                {
                    "instType": "SWAP",
                    "mgnMode": params.get("mgnMode", "cross"),
                    "lever": str(max(1, requested_leverage)),
                    "instId": inst_id,
                },
            )
            for item in (estimate or {}).get("data") or []:
                if isinstance(item, dict):
                    values.append(self._safe_float(item.get("maxLever"), 0.0))
        except Exception as e:
            logger.debug(
                "fetch OKX adjust leverage info failed",
                symbol=okx_symbol,
                error=safe_error_text(e),
            )

        values = [value for value in values if value > 0]
        return max(values) if values else float(settings.max_leverage)

    def _extract_verified_leverage(self, leverage_response: dict[str, Any] | None) -> float:
        """Return the leverage reported by CCXT fetch_leverage."""
        if not isinstance(leverage_response, dict):
            return 0.0
        values = [
            self._safe_float(leverage_response.get("longLeverage"), 0.0),
            self._safe_float(leverage_response.get("shortLeverage"), 0.0),
        ]
        for value in values:
            if value > 0:
                return value
        for item in leverage_response.get("info") or []:
            if isinstance(item, dict):
                value = self._safe_float(item.get("lever"), 0.0)
                if value > 0:
                    return value
        return 0.0

    async def _set_leverage_if_needed(self, decision: DecisionOutput) -> dict[str, Any]:
        """Apply leverage with a fast happy path and one safe 59670 recovery attempt."""
        ccxt = await self._get_ccxt()
        okx_symbol = self._to_swap_symbol(decision.symbol)
        params = {"mgnMode": "cross"}
        requested_leverage = int(
            max(1, min(int(round(decision.suggested_leverage)), settings.max_leverage))
        )
        max_leverage = await self._fetch_okx_max_leverage(okx_symbol, params, requested_leverage)
        leverage = int(
            max(
                1,
                min(
                    requested_leverage,
                    int(max_leverage or requested_leverage),
                    settings.max_leverage,
                ),
            )
        )
        decision.suggested_leverage = float(leverage)

        actual = 0.0
        verify_response: dict[str, Any] | None = None
        try:
            actual, verify_response = await self._fetch_current_leverage(ccxt, okx_symbol, params)
        except Exception as e:
            logger.debug(
                "fetch current leverage before set failed",
                symbol=okx_symbol,
                error=safe_error_text(e),
            )
        actual_rounded = int(round(actual or 0))
        if actual > 0 and actual_rounded == leverage:
            return {
                "ok": True,
                "skipped_set": True,
                "reason": "当前 OKX 杠杆已等于系统目标杠杆，无需重复设置。",
                "ai_requested_leverage": requested_leverage,
                "okx_max_leverage": max_leverage,
                "target_leverage": leverage,
                "actual_leverage": actual,
                "set_response": None,
                "verify_response": verify_response,
                "params": params,
            }

        set_response: dict[str, Any] | None = None
        cleanup_result: dict[str, Any] | None = None
        set_error: Exception | None = None
        try:
            set_response = await self._with_retry(ccxt.set_leverage, leverage, okx_symbol, params)
        except Exception as e:
            set_error = e
            set_error_text = safe_error_text(set_error, limit=220)
            if self._is_leverage_open_order_limit_error(set_error_text):
                cleanup_result = await self._reduce_open_orders_for_leverage_retry(ccxt, okx_symbol)
                if int(cleanup_result.get("cancelled") or 0) > 0:
                    try:
                        set_response = await self._with_retry(
                            ccxt.set_leverage, leverage, okx_symbol, params
                        )
                        set_error = None
                        set_error_text = ""
                    except Exception as retry_error:
                        set_error = retry_error
                        set_error_text = safe_error_text(set_error, limit=220)
                if set_response is None:
                    try:
                        actual, verify_response = await self._fetch_current_leverage(
                            ccxt,
                            okx_symbol,
                            params,
                            force=True,
                        )
                    except Exception as current_error:
                        logger.debug(
                            "fetch current leverage after 59670 failed",
                            symbol=okx_symbol,
                            error=safe_error_text(current_error),
                        )
                    actual_rounded = int(round(actual or 0))
                    if actual > 0 and actual_rounded == leverage:
                        return {
                            "ok": True,
                            "skipped_set": True,
                            "reason": (
                                "OKX 因未成交委托限制本轮无法调整杠杆，"
                                "但当前杠杆已等于系统目标杠杆。"
                            ),
                            "ai_requested_leverage": requested_leverage,
                            "okx_max_leverage": max_leverage,
                            "target_leverage": leverage,
                            "actual_leverage": actual,
                            "set_response": None,
                            "verify_response": verify_response,
                            "cleanup_result": cleanup_result,
                            "params": params,
                        }
                    if actual <= 0:
                        try:
                            positions = await self.get_positions_strict(decision.symbol)
                            for position in positions or []:
                                info = position.get("info") or {}
                                candidate = self._safe_float(
                                    position.get("leverage")
                                    or info.get("lever")
                                    or info.get("leverage"),
                                    0.0,
                                )
                                if candidate > 0:
                                    actual = candidate
                                    break
                        except Exception as position_error:
                            logger.debug(
                                "fetch position leverage after 59670 failed",
                                symbol=okx_symbol,
                                error=safe_error_text(position_error),
                            )
                    actual_rounded = int(round(actual or 0))
                    if actual <= 0:
                        message = (
                            f"OKX 因该交易对未成交委托数量限制，无法在本轮调整杠杆；"
                            f"目标杠杆 {leverage}x，但系统无法确认当前实际杠杆。"
                            "为避免在未知杠杆下开仓，本次不开仓。"
                        )
                        logger.warning(
                            "leverage 59670; rejecting unknown actual leverage",
                            symbol=decision.symbol,
                            okx_symbol=okx_symbol,
                            requested_leverage=requested_leverage,
                            target_leverage=leverage,
                            cleanup_result=cleanup_result,
                            error=set_error_text,
                        )
                        return {
                            "ok": False,
                            "error": message,
                            "ai_requested_leverage": requested_leverage,
                            "okx_max_leverage": max_leverage,
                            "target_leverage": leverage,
                            "actual_leverage": None,
                            "set_response": None,
                            "verify_response": verify_response,
                            "cleanup_result": cleanup_result,
                            "open_order_limit_error": set_error_text,
                            "params": params,
                        }
                    fallback_leverage = actual
                    if actual > 0 and actual_rounded > leverage:
                        message = (
                            f"OKX 当前杠杆 {actual:g}x 高于系统目标 {leverage}x；"
                            "但 OKX 因该交易对未成交委托数量限制，无法在本轮降低杠杆。"
                            "为避免使用更高杠杆放大亏损，本次不开仓。"
                        )
                        logger.warning(
                            "leverage 59670; rejecting higher existing leverage",
                            symbol=decision.symbol,
                            okx_symbol=okx_symbol,
                            requested_leverage=requested_leverage,
                            target_leverage=leverage,
                            actual_leverage=actual,
                            cleanup_result=cleanup_result,
                            error=set_error_text,
                        )
                        return {
                            "ok": False,
                            "error": message,
                            "ai_requested_leverage": requested_leverage,
                            "okx_max_leverage": max_leverage,
                            "target_leverage": leverage,
                            "actual_leverage": actual,
                            "set_response": None,
                            "verify_response": verify_response,
                            "cleanup_result": cleanup_result,
                            "open_order_limit_error": set_error_text,
                            "params": params,
                        }
                    decision.suggested_leverage = float(fallback_leverage)
                    logger.warning(
                        "leverage 59670; using safer existing leverage",
                        symbol=decision.symbol,
                        okx_symbol=okx_symbol,
                        requested_leverage=requested_leverage,
                        target_leverage=leverage,
                        fallback_leverage=fallback_leverage,
                        cleanup_result=cleanup_result,
                        error=set_error_text,
                    )
                    return {
                        "ok": True,
                        "skipped_set": True,
                        "reason": (
                            f"OKX 因该交易对未成交委托数量限制，无法在本轮调整杠杆；"
                            f"当前杠杆 {fallback_leverage:g}x 不高于系统目标 {leverage}x，"
                            "系统按较低风险杠杆继续下单。"
                        ),
                        "ai_requested_leverage": requested_leverage,
                        "okx_max_leverage": max_leverage,
                        "target_leverage": leverage,
                        "actual_leverage": fallback_leverage,
                        "set_response": None,
                        "verify_response": verify_response,
                        "cleanup_result": cleanup_result,
                        "open_order_limit_error": set_error_text,
                        "params": params,
                    }

        if set_response is None:
            set_error_text = safe_error_text(set_error, limit=220)
            message = (
                f"OKX 杠杆设置失败，本次未开仓。目标杠杆 {leverage}x。"
                f"OKX 返回：{set_error_text}"
            )
            if cleanup_result:
                message += (
                    f" 已检查未成交委托 {cleanup_result.get('open_orders', 0)} 条，"
                    f"已取消可安全取消的旧开仓委托 {cleanup_result.get('cancelled', 0)} 条。"
                )
            logger.warning(
                "set leverage failed; rejecting entry order",
                symbol=decision.symbol,
                okx_symbol=okx_symbol,
                leverage=leverage,
                error=set_error_text,
                cleanup_result=cleanup_result,
            )
            return {
                "ok": False,
                "error": message,
                "ai_requested_leverage": requested_leverage,
                "okx_max_leverage": max_leverage,
                "target_leverage": leverage,
                "actual_leverage": actual or None,
                "set_response": set_response,
                "verify_response": verify_response,
                "cleanup_result": cleanup_result,
                "params": params,
            }

        try:
            verify_response = await self._with_retry(ccxt.fetch_leverage, okx_symbol, params)
            actual = self._extract_verified_leverage(verify_response)
            self._cache_leverage(okx_symbol, params, actual)
        except Exception as e:
            error_text = safe_error_text(e, limit=220)
            message = (
                "OKX 杠杆已提交设置，但系统无法确认是否生效，本次未开仓。"
                f"目标杠杆 {leverage}x。确认查询返回：{error_text}"
            )
            logger.warning(
                "set leverage verification failed; rejecting entry order",
                symbol=decision.symbol,
                okx_symbol=okx_symbol,
                leverage=leverage,
                error=error_text,
                set_response=set_response,
            )
            return {
                "ok": False,
                "error": message,
                "ai_requested_leverage": requested_leverage,
                "okx_max_leverage": max_leverage,
                "target_leverage": leverage,
                "actual_leverage": None,
                "set_response": set_response,
                "verify_response": verify_response,
                "cleanup_result": cleanup_result,
                "params": params,
            }

        if int(round(actual or 0)) != leverage:
            message = (
                "OKX 杠杆设置未生效，本次未开仓。"
                f"目标杠杆 {leverage}x，实际查询为 {actual or 0:g}x。"
            )
            logger.warning(
                "set leverage mismatch; rejecting entry order",
                symbol=decision.symbol,
                okx_symbol=okx_symbol,
                target_leverage=leverage,
                actual_leverage=actual,
                set_response=set_response,
                verify_response=verify_response,
            )
            return {
                "ok": False,
                "error": message,
                "ai_requested_leverage": requested_leverage,
                "okx_max_leverage": max_leverage,
                "target_leverage": leverage,
                "actual_leverage": actual,
                "set_response": set_response,
                "verify_response": verify_response,
                "cleanup_result": cleanup_result,
                "params": params,
            }

        return {
            "ok": True,
            "ai_requested_leverage": requested_leverage,
            "okx_max_leverage": max_leverage,
            "target_leverage": leverage,
            "actual_leverage": actual,
            "set_response": set_response,
            "verify_response": verify_response,
            "cleanup_result": cleanup_result,
            "params": params,
        }

    def _to_swap_symbol(self, symbol: str) -> str:
        """Convert app symbols to CCXT's OKX USDT perpetual swap symbol."""
        normalized = (symbol or "").strip().upper().replace("_", "-")
        if not normalized:
            return symbol
        if ":" in normalized:
            return normalized
        if normalized.endswith("-SWAP"):
            parts = normalized.split("-")
            if len(parts) >= 3:
                return f"{parts[0]}/{parts[1]}:{parts[1]}"
        if "/" in normalized:
            base, quote = normalized.split("/", 1)
            quote = quote.split(":")[0]
            if quote == "USDT":
                return f"{base}/USDT:USDT"
        if "-" in normalized:
            parts = normalized.split("-")
            if len(parts) >= 2 and parts[1] == "USDT":
                return f"{parts[0]}/USDT:USDT"
        return symbol

    def _from_swap_symbol(self, symbol: str | None) -> str:
        """Convert OKX/CCXT swap symbols back to app symbols."""
        normalized = str(symbol or "").strip().upper()
        if not normalized:
            return ""
        normalized = normalized.split(":")[0]
        if normalized.endswith("-SWAP"):
            normalized = normalized[:-5]
        if "/" not in normalized and "-" in normalized:
            parts = normalized.split("-")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"
        return normalized

    def _attached_sl_tp_prices(
        self,
        decision: DecisionOutput,
        entry_price: float,
        *,
        ticker: dict[str, Any] | None = None,
    ) -> tuple[float, float]:
        """Convert AI stop/take percentages into OKX trigger prices."""
        refs = [self._safe_float(entry_price, 0.0)]
        ticker = ticker if isinstance(ticker, dict) else {}
        for key in ("last", "close", "bid", "ask", "high", "low"):
            value = self._safe_float(ticker.get(key), 0.0)
            if value > 0:
                refs.append(value)
        snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        for key in ("current_price", "close", "bid", "ask", "last", "last_price"):
            value = self._safe_float(snapshot.get(key), 0.0)
            if value > 0:
                refs.append(value)
        valid_refs = [value for value in refs if value > 0]
        low_ref = min(valid_refs) if valid_refs else self._safe_float(entry_price, 0.0)
        high_ref = max(valid_refs) if valid_refs else self._safe_float(entry_price, 0.0)
        primary_ref = self._safe_float(entry_price, 0.0)
        if primary_ref <= 0 and valid_refs:
            primary_ref = valid_refs[0]
        stop_pct = max(
            self._safe_float(decision.stop_loss_pct, 0.0),
            ATTACHED_PROTECTION_MIN_STOP_PCT,
        )
        stop_pct = min(stop_pct, 0.15)
        take_pct = max(
            self._safe_float(decision.take_profit_pct, 0.0),
            stop_pct * 1.8,
            ATTACHED_PROTECTION_MIN_TAKE_PROFIT_PCT,
        )
        take_pct = min(take_pct, 0.50)
        trigger_gap = max(primary_ref * ATTACHED_PROTECTION_MIN_TRIGGER_GAP_PCT, 1e-8)
        if decision.action == Action.LONG:
            stop_loss_px = low_ref * (1 - stop_pct)
            take_profit_px = high_ref * (1 + take_pct)
            if primary_ref > 0:
                stop_loss_px = min(stop_loss_px, primary_ref - trigger_gap)
                take_profit_px = max(take_profit_px, primary_ref + trigger_gap)
        else:
            stop_loss_px = high_ref * (1 + stop_pct)
            take_profit_px = low_ref * (1 - take_pct)
            if primary_ref > 0:
                stop_loss_px = max(stop_loss_px, primary_ref + trigger_gap)
                take_profit_px = min(take_profit_px, primary_ref - trigger_gap)
        return stop_loss_px, take_profit_px

    def _format_attached_sl_tp_prices(
        self,
        ccxt,
        okx_symbol: str,
        decision: DecisionOutput,
        stop_loss_px: float,
        take_profit_px: float,
        reference_price: float,
    ) -> dict[str, Any]:
        """Format attached TP/SL trigger prices with OKX market precision."""

        def fmt(value: float) -> str | None:
            if value <= 0:
                return None
            try:
                text = str(ccxt.price_to_precision(okx_symbol, value))
            except Exception as exc:
                logger.debug(
                    "OKX price precision failed for attached protection",
                    symbol=okx_symbol,
                    price=value,
                    error=safe_error_text(exc),
                )
                text = format(value, ".12g")
            try:
                parsed = Decimal(text)
            except (InvalidOperation, ValueError):
                return None
            if parsed <= 0:
                return None
            return format(parsed.normalize(), "f")

        stop_text = fmt(stop_loss_px)
        take_text = fmt(take_profit_px)
        ref = self._safe_float(reference_price, 0.0)
        stop_value = self._safe_float(stop_text, 0.0)
        take_value = self._safe_float(take_text, 0.0)
        valid = bool(stop_text and take_text)
        if valid and ref > 0:
            if decision.action == Action.LONG:
                valid = stop_value < ref < take_value
            else:
                valid = take_value < ref < stop_value
        if not valid:
            action_label = "做多" if decision.action == Action.LONG else "做空"
            return {
                "ok": False,
                "error": (
                    f"OKX 保护单价格校验失败：{action_label}开仓的止损/止盈触发价不符合交易所精度或方向约束，"
                    "本次未提交订单，等待下一轮行情重新计算。"
                ),
                "okx_symbol": okx_symbol,
                "reference_price": ref,
                "raw_stop_loss_price": stop_loss_px,
                "raw_take_profit_price": take_profit_px,
                "stop_loss_price": stop_text,
                "take_profit_price": take_text,
            }
        return {
            "ok": True,
            "okx_symbol": okx_symbol,
            "reference_price": ref,
            "raw_stop_loss_price": stop_loss_px,
            "raw_take_profit_price": take_profit_px,
            "stop_loss_price": stop_text,
            "take_profit_price": take_text,
        }

    async def get_balance(self, asset: str = "USDT") -> float:
        ccxt = await self._get_ccxt()
        try:
            balance_data = await self._with_retry(ccxt.fetch_balance)
            return float(balance_data.get(asset, {}).get("free", 0))
        except Exception as e:
            logger.error("fetch balance failed", error=safe_error_text(e))
            return 0.0

    async def get_balance_snapshot(self, asset: str = "USDT") -> dict[str, Any]:
        ccxt = await self._get_ccxt()
        try:
            balance_data = await self._with_retry(ccxt.fetch_balance)
            asset_data = balance_data.get(asset, {}) or {}
            raw_detail = {}
            info = balance_data.get("info") or {}
            for item in info.get("data", []) if isinstance(info, dict) else []:
                if not isinstance(item, dict):
                    continue
                for detail in item.get("details", []) or []:
                    if isinstance(detail, dict) and detail.get("ccy") == asset:
                        raw_detail = detail
                        break
                if raw_detail:
                    break

            def raw_float(key: str, fallback: float = 0.0) -> float:
                try:
                    return float(raw_detail.get(key) or fallback)
                except Exception:
                    return fallback

            total = float(asset_data.get("total") or 0.0)
            cash = raw_float("cashBal", total)
            equity = raw_float("eq", total)
            allocatable = equity if equity > 0 else (cash if cash > 0 else total)
            return {
                "free": float(asset_data.get("free") or 0.0),
                "used": float(asset_data.get("used") or 0.0),
                "total": total,
                "cash": cash,
                "equity": equity,
                "allocatable": allocatable,
            }
        except Exception as e:
            error_text = safe_error_text(e)
            logger.error("fetch balance snapshot failed", error=error_text)
            return {"free": 0.0, "used": 0.0, "total": 0.0, "error": error_text}

    async def get_positions(self, symbol: str | None = None) -> list[dict]:
        ccxt = await self._get_ccxt()
        try:
            symbols = [self._to_swap_symbol(symbol)] if symbol else None
            positions = await self._with_retry(ccxt.fetch_positions, symbols)
            return positions
        except Exception as e:
            logger.error("fetch positions failed", error=safe_error_text(e))
            return []

    async def get_positions_strict(self, symbol: str | None = None) -> list[dict]:
        """Fetch positions and let errors propagate so reconciliation can trust an empty result."""
        ccxt = await self._get_ccxt()
        symbols = [self._to_swap_symbol(symbol)] if symbol else None
        return await self._with_retry(ccxt.fetch_positions, symbols)

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        okx_symbol = self._to_swap_symbol(symbol) if symbol else None
        try:
            return await self.get_open_orders_strict(symbol)
        except Exception as e:
            logger.warning(
                "fetch open orders failed",
                symbol=okx_symbol,
                error=safe_error_text(e),
            )
            return []

    async def get_open_orders_strict(self, symbol: str | None = None) -> list[dict]:
        """Fetch open orders and let errors propagate for reconciliation safety."""
        ccxt = await self._get_ccxt()
        return await self._with_retry(
            ccxt.fetch_open_orders,
            self._to_swap_symbol(symbol) if symbol else None,
        )

    async def get_position_protection_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetch active OKX TP/SL algo orders that protect open positions."""
        ccxt = await self._get_ccxt()
        okx_symbol = self._to_swap_symbol(symbol) if symbol else None
        protection_orders: list[dict] = []

        for ord_type in ("oco", "conditional", "trigger"):
            try:
                orders = await self._with_retry(
                    ccxt.fetch_open_orders,
                    okx_symbol,
                    None,
                    100,
                    {"ordType": ord_type},
                )
            except Exception as e:
                logger.warning(
                    "fetch OKX protection orders failed",
                    symbol=okx_symbol,
                    ord_type=ord_type,
                    error=safe_error_text(e),
                )
                continue

            for order in orders or []:
                info = order.get("info") or {}
                state = str(order.get("status") or info.get("state") or "").lower()
                if state and state not in {"open", "live", "pending"}:
                    continue

                reduce_only = order.get("reduceOnly")
                if reduce_only in (None, ""):
                    reduce_only = info.get("reduceOnly")
                if str(reduce_only).lower() != "true":
                    continue

                take_profit = self._safe_float(
                    order.get("takeProfitPrice") or info.get("tpTriggerPx"),
                    0.0,
                )
                stop_loss = self._safe_float(
                    order.get("stopLossPrice") or info.get("slTriggerPx"),
                    0.0,
                )
                trigger_price = self._safe_float(
                    order.get("triggerPrice") or info.get("triggerPx"),
                    0.0,
                )
                if take_profit <= 0 and stop_loss <= 0 and trigger_price <= 0:
                    continue

                close_side = str(order.get("side") or info.get("side") or "").lower()
                pos_side = str(info.get("posSide") or "").lower()
                if pos_side not in {"long", "short"}:
                    pos_side = "short" if close_side == "buy" else "long"

                protection_orders.append(
                    {
                        "symbol": self._from_swap_symbol(
                            order.get("symbol") or info.get("instId") or symbol
                        ),
                        "position_side": pos_side,
                        "close_side": close_side,
                        "order_type": info.get("ordType") or order.get("type") or ord_type,
                        "take_profit_price": take_profit if take_profit > 0 else None,
                        "stop_loss_price": stop_loss if stop_loss > 0 else None,
                        "trigger_price": trigger_price if trigger_price > 0 else None,
                        "algo_id": info.get("algoId") or order.get("id"),
                        "updated_at_ms": self._safe_float(
                            info.get("uTime") or info.get("cTime"), 0.0
                        ),
                        "raw": order,
                    }
                )

        return protection_orders

    async def _get_ccxt(self):
        if self._exchange is None:
            await self.initialize()
        self._ensure_rest_url()
        return self._exchange

    async def shutdown(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
            self._connected = False
        logger.info("OKX executor shut down")
