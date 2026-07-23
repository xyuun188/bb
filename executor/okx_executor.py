"""
OKX live/demo executor via the official python-okx SDK adapter.
Sends real orders to OKX exchange (demo or production based on settings).
Includes retry logic, rate limiting, and error handling.
"""

from __future__ import annotations

import asyncio
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
from core.okx_instrument_filter import supported_usdt_swap_instruments
from core.safe_output import safe_error_text
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_symbol,
    symbol_from_okx_market,
    symbol_from_okx_payload,
)
from core.trading_mode import mode_manager
from executor.base_executor import AbstractExecutor, ExecutionResult, OrderStatus
from services.entry_profit_risk_sizing import (
    reconcile_profit_risk_sizing,
    select_okx_leverage_tier,
)
from services.exchange_position_state import parse_exchange_position_snapshot
from services.okx_native_facts import OkxNativeFactsClient
from services.okx_perpetual_sdk import OkxPerpetualSdkExchange
from services.paper_training import (
    PAPER_TRAINING_ORDER_IDENTITY_VERSION,
    is_paper_training_decision,
    paper_training_client_order_id,
)

logger = structlog.get_logger(__name__)

OKX_REST_URL = "https://{hostname}"
OKX_HOSTNAME = "www.okx.com"
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
RATE_LIMIT_TOKENS = 10  # max requests per second
RATE_LIMIT_PERIOD = 1.0
OKX_REST_CALL_TIMEOUT = 10.0
OKX_TIME_DIFFERENCE_SYNC_TIMEOUT = 3.0
EXIT_ORDER_REPLACE_AFTER_SECONDS = 20.0
OKX_CONTRACT_DELIVERY_LOCK_SECONDS = 3600.0
OKX_ENTRY_INSTRUMENT_AVAILABILITY_CACHE_SECONDS = 1800.0
OKX_ENTRY_INSTRUMENT_UNAVAILABLE_CACHE_SECONDS = 21600.0
OKX_ENTRY_INSTRUMENT_PROBE_FAILURE_CACHE_SECONDS = 30.0

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
    """Executes trades on OKX via the official python-okx SDK adapter.

    Paper and live modes use their explicitly isolated OKX credentials.
    In production mode, real orders are placed.
    """

    def __init__(self, mode: str | None = None, *, load_markets_on_initialize: bool = True) -> None:
        self._mode_override = mode  # "paper" or "live", overrides global mode
        self._exchange: Any = None
        self._rate_limiter = TokenBucket(RATE_LIMIT_TOKENS, RATE_LIMIT_TOKENS * 2)
        self._connected = False
        self._load_markets_on_initialize = load_markets_on_initialize
        self._markets_loaded = False
        self._leverage_cache: dict[tuple[str, str], tuple[float, float]] = {}
        self._contract_delivery_locks: dict[str, tuple[float, str]] = {}
        self._entry_instrument_availability_cache: dict[
            str, tuple[dict[str, Any], float, float]
        ] = {}

    @property
    def executor_mode(self) -> str:
        return self._mode_override or mode_manager.mode.value

    async def initialize(self) -> None:
        mode = self.executor_mode
        is_demo = settings.is_okx_demo(mode)

        self._exchange = OkxPerpetualSdkExchange(mode)
        self._ensure_rest_url()

        try:
            await self._sync_time_difference()
            if self._load_markets_on_initialize:
                await self._load_usdt_swap_markets()
            self._connected = True
            logger.info(
                "OKX executor initialized",
                mode=mode,
                demo=is_demo,
                markets=len(self._exchange.markets or {}),
                markets_loaded=self._markets_loaded,
            )
        except Exception:
            await self.shutdown()
            raise

    async def _sync_time_difference(self) -> None:
        if self._exchange is None:
            return
        loader = getattr(self._exchange, "load_time_difference", None)
        if not callable(loader):
            return
        try:
            self._ensure_rest_url()
            await asyncio.wait_for(
                loader(),
                timeout=OKX_TIME_DIFFERENCE_SYNC_TIMEOUT,
            )
        except Exception as exc:
            logger.warning(
                "OKX time difference sync failed",
                mode=self.executor_mode,
                error=safe_error_text(exc),
            )

    @staticmethod
    def _is_time_difference_error(error: Any) -> bool:
        text = safe_error_text(error).lower()
        return any(
            marker in text
            for marker in (
                "50102",
                "timestamp request expired",
                "invalid nonce",
                "time difference",
            )
        )

    def _ensure_rest_url(self) -> None:
        """Keep legacy URL guards harmless for the SDK-backed exchange adapter."""
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
        fields. The trading system only needs live USDT swaps, so filter the raw
        instrument list before building local market rules.
        """
        if self._exchange is None:
            raise ExchangeAPIError("OKX exchange is not initialized")

        self._ensure_rest_url()
        response = await self._exchange.publicGetPublicInstruments({"instType": "SWAP"})
        instruments = response.get("data", []) if isinstance(response, dict) else []
        filtered = supported_usdt_swap_instruments(instruments)
        markets = self._exchange.parse_markets(filtered)
        self._apply_okx_instrument_contract_rules(markets, filtered)
        self._exchange.set_markets(markets)
        if not self._exchange.markets:
            raise ExchangeAPIError("No OKX USDT swap markets loaded")
        self._markets_loaded = True

    def _apply_okx_instrument_contract_rules(
        self,
        markets: Any,
        instruments: list[Any],
    ) -> None:
        """Preserve OKX raw contract rules after local market parsing.

        Some OKX swap instruments, notably BTC/ETH, have ``ctVal`` below 1.
        If the parsed market loses that raw value, the executor can mistake
        contracts for base quantity and poison order/position reconciliation.
        """

        raw_by_inst_id = {
            str(item.get("instId") or "").strip().upper(): item
            for item in instruments
            if isinstance(item, dict) and str(item.get("instId") or "").strip()
        }
        if isinstance(markets, dict):
            iterable = markets.values()
        elif isinstance(markets, list):
            iterable = markets
        else:
            return
        for market in iterable:
            if not isinstance(market, dict):
                continue
            info = market.get("info") if isinstance(market.get("info"), dict) else {}
            inst_id = str(info.get("instId") or market.get("id") or "").strip().upper()
            raw = raw_by_inst_id.get(inst_id)
            if not raw:
                continue
            market["info"] = {**info, **raw}
            contract_size = self._safe_float(raw.get("ctVal"), 0.0)
            if contract_size > 0:
                market["contractSize"] = contract_size
                market["info"]["ctVal"] = raw.get("ctVal")
            min_size = self._safe_float(raw.get("minSz"), 0.0)
            lot_size = self._safe_float(raw.get("lotSz"), 0.0)
            max_market_size = self._safe_float(raw.get("maxMktSz"), 0.0)
            limits = dict(market.get("limits") or {})
            amount_limits = dict(limits.get("amount") or {})
            if min_size > 0:
                amount_limits["min"] = min_size
            if max_market_size > 0:
                amount_limits["max"] = max_market_size
            if amount_limits:
                limits["amount"] = amount_limits
                market["limits"] = limits
            precision = dict(market.get("precision") or {})
            if lot_size > 0:
                precision["amount"] = lot_size
                market["precision"] = precision

    async def _ensure_markets_loaded(self) -> None:
        """Load OKX swap contract rules before market/order/position operations."""
        ccxt = await self._get_ccxt()
        if self._markets_loaded and getattr(ccxt, "markets", None):
            return
        if not hasattr(ccxt, "publicGetPublicInstruments"):
            self._markets_loaded = True
            return
        await self._load_usdt_swap_markets()

    def _is_missing_market_symbol_error(self, error: Any) -> bool:
        text = str(error or "").lower()
        return "does not have market symbol" in text or "bad symbol" in text

    def _position_matches_symbol(self, position: dict[str, Any], symbol: str) -> bool:
        info = position.get("info") or {}
        requested = normalize_trading_symbol(symbol)
        actual = normalize_trading_symbol(info.get("instId") or position.get("symbol"))
        return bool(requested and actual and requested == actual)

    def _native_facts_client(self) -> OkxNativeFactsClient:
        return OkxNativeFactsClient(self)

    def _synthetic_exit_market_from_position(
        self,
        position: dict[str, Any],
        okx_symbol: str,
    ) -> dict[str, Any] | None:
        """Build minimal market rules from an existing position for native full-close."""

        from services.exchange_position_state import parse_exchange_position_snapshot

        info = position.get("info") or {}
        inst_id = str(info.get("instId") or position.get("symbol") or okx_symbol or "").strip()
        if not inst_id:
            return None
        snapshot = parse_exchange_position_snapshot(
            position,
            symbol_normalizer=normalize_trading_symbol,
        )
        if not snapshot:
            return None
        contract_size = self._safe_float(snapshot.get("contract_size"), 0.0)
        if contract_size <= 0:
            contract_size = 1.0
        min_size = self._safe_float(info.get("minSz") or info.get("lotSz"), 0.0) or 1.0
        max_market_size = self._safe_float(info.get("maxMktSz"), 0.0)
        market_info = {
            **info,
            "instId": inst_id,
            "ctVal": str(contract_size),
            "minSz": str(min_size),
            "lotSz": str(min_size),
        }
        if max_market_size > 0:
            market_info["maxMktSz"] = str(max_market_size)
        return {
            "id": inst_id,
            "symbol": str(position.get("symbol") or inst_id),
            "type": "swap",
            "swap": True,
            "linear": True,
            "contract": True,
            "contractSize": contract_size,
            "precision": {"amount": min_size},
            "limits": {"amount": {"min": min_size, "max": max_market_size or None}},
            "info": market_info,
            "synthetic_from_position": True,
        }

    async def _market_from_existing_position(
        self,
        app_symbol: str | None,
        okx_symbol: str,
    ) -> dict[str, Any] | None:
        if not app_symbol:
            return None
        positions = await self.get_positions_strict(app_symbol)
        for position in positions or []:
            market = self._synthetic_exit_market_from_position(position, okx_symbol)
            if market:
                return market
        return None

    async def _market_for_symbol(
        self,
        okx_symbol: str,
        *,
        app_symbol: str | None = None,
    ) -> dict[str, Any]:
        """Return a market, reloading OKX instruments when a new swap is missing."""
        ccxt = await self._get_ccxt()
        await self._ensure_markets_loaded()
        try:
            return ccxt.market(okx_symbol)
        except Exception as exc:
            error_text = safe_error_text(exc).lower()
            if not self._is_missing_market_symbol_error(error_text):
                raise
            logger.warning(
                "OKX market missing from cache; reloading instruments",
                symbol=okx_symbol,
                error=safe_error_text(exc),
            )
            self._markets_loaded = False
            await self._load_usdt_swap_markets()
            try:
                return ccxt.market(okx_symbol)
            except Exception as retry_exc:
                retry_error = safe_error_text(retry_exc)
                if not self._is_missing_market_symbol_error(retry_error):
                    raise
                market = await self._market_from_existing_position(app_symbol, okx_symbol)
                if market:
                    logger.warning(
                        "using existing OKX position snapshot as synthetic exit market",
                        symbol=okx_symbol,
                        app_symbol=app_symbol,
                    )
                    return market
                raise

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
        self._markets_loaded = False
        await self.initialize()

    async def _with_retry(
        self,
        fn,
        *args,
        _expected_error_codes: set[str] | frozenset[str] | None = None,
        _max_attempts: int | None = None,
        **kwargs,
    ):
        """Execute an API call with retry + rate limit handling."""
        last_error = None
        method_name = getattr(fn, "__name__", "")
        attempts = max(1, min(int(_max_attempts or MAX_RETRIES), MAX_RETRIES))
        expected_error_codes = {
            str(code).strip()
            for code in (_expected_error_codes or set())
            if str(code).strip()
        }
        for attempt in range(attempts):
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
                if attempt < attempts - 1:
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))
                    last_error = e
                    continue
                raise ExchangeAPIError(
                    f"OKX REST call timed out after {OKX_REST_CALL_TIMEOUT:.0f}s: {method_name}"
                ) from e
            except ExchangeAPIError as e:
                message = safe_error_text(e)
                error_code = self._exchange_error_code(e, message)
                if self._is_time_difference_error(message) and attempt < attempts - 1:
                    logger.warning(
                        "OKX SDK time drift detected; resyncing and retrying",
                        method=method_name,
                        attempt=attempt,
                        error=message,
                    )
                    await self._sync_time_difference()
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))
                    last_error = e
                    continue
                if self._is_rate_limit_error(message) and attempt < attempts - 1:
                    logger.warning("OKX SDK rate limited", method=method_name, attempt=attempt)
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))
                    last_error = e
                    continue
                if self._is_transient_system_error(message) and attempt < attempts - 1:
                    logger.warning(
                        "OKX SDK temporary system error; retrying",
                        method=method_name,
                        attempt=attempt,
                        error=message,
                    )
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))
                    last_error = e
                    continue
                if error_code in expected_error_codes:
                    logger.debug(
                        "OKX SDK expected capability rejection",
                        method=method_name,
                        error_code=error_code,
                    )
                else:
                    logger.error("OKX SDK exchange error", error=message)
                raise ExchangeAPIError(
                    message,
                    code=error_code or getattr(e, "code", None),
                    payload=getattr(e, "payload", None),
                ) from e
            except Exception as e:
                if self._is_broken_rest_url_error(e) and attempt < attempts - 1:
                    logger.warning(
                        "OKX executor REST URL state invalid; reinitializing SDK client",
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
                message = safe_error_text(e)
                if self._is_rate_limit_error(message) and attempt < attempts - 1:
                    logger.warning("OKX SDK call rate limited", method=method_name, attempt=attempt)
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))
                    last_error = e
                    continue
                if self._is_network_retryable_error(message) and attempt < attempts - 1:
                    logger.warning(
                        "OKX SDK network error; retrying",
                        method=method_name,
                        attempt=attempt,
                        error=message,
                    )
                    await asyncio.sleep(RETRY_DELAY * (2**attempt))
                    last_error = e
                    continue
                raise

        raise RateLimitError(f"Max retries exceeded: {safe_error_text(last_error)}")

    @staticmethod
    def _exchange_error_code(exc: BaseException, message: str = "") -> str:
        code = str(getattr(exc, "code", "") or "").strip()
        if code:
            return code
        match = re.search(r"\[(\d{5})\]", str(message or ""))
        return match.group(1) if match else ""

    @staticmethod
    def _is_rate_limit_error(message: Any) -> bool:
        text = str(message or "").lower()
        return "rate limit" in text or "too many requests" in text or "50011" in text

    @staticmethod
    def _is_network_retryable_error(message: Any) -> bool:
        text = str(message or "").lower()
        return any(
            marker in text
            for marker in (
                "timeout",
                "timed out",
                "connection",
                "network",
                "temporarily unavailable",
                "remote protocol error",
            )
        )

    @staticmethod
    def _is_transient_system_error(message: Any) -> bool:
        text = str(message or "").lower()
        return "50026" in text or "system error. try again later" in text

    async def place_order(
        self,
        decision: DecisionOutput,
        account_id: str | None = None,
        override_balance: float | None = None,
    ) -> ExecutionResult:
        if not self._connected:
            await self.initialize()
        await self._ensure_markets_loaded()

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

        side = "buy"
        okx_symbol = self._to_swap_symbol(decision.symbol)
        contract_size = 1.0
        order_quantity = 0.0
        base_quantity = 0.0
        okx_order_rules: dict[str, Any] = {}
        params: dict[str, Any] = {}
        leverage_check: dict[str, Any] | None = None
        paper_training_margin_reserve: dict[str, Any] = {}
        protection_submit_requested_at: datetime | None = None
        protection_submission: dict[str, Any] = {}

        try:
            okx_symbol = await self._resolve_swap_symbol(decision.symbol)
            # Map action to OKX order side.
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
                sizing = (
                    decision.raw_response.get("profit_risk_sizing")
                    if isinstance(decision.raw_response, dict)
                    else {}
                )
                sizing = sizing if isinstance(sizing, dict) else {}
                position_value = max(
                    self._safe_float(sizing.get("final_notional_usdt"), 0.0),
                    0.0,
                )
                paper_training_margin_reserve = (
                    self._paper_training_margin_execution_reserve(
                        decision,
                        available_balance_usdt=balance,
                        leverage=decision.suggested_leverage,
                        planned_notional_usdt=position_value,
                    )
                )
                position_value = self._safe_float(
                    paper_training_margin_reserve.get(
                        "executable_notional_usdt",
                        position_value,
                    ),
                    position_value,
                )

            exit_position_snapshot: list[dict[str, Any]] | None = None

            # Use OKX-native instId ticker for quantity calculation. If ticker is
            # unavailable, entries fail closed; exits can still use position marks.
            ticker: dict[str, Any] = {}
            try:
                ticker = await self._fetch_native_ticker(decision.symbol)
            except Exception as exc:
                if not decision.is_exit:
                    error_text = safe_error_text(exc)
                    logger.warning(
                        "OKX native ticker unavailable for entry; rejecting before submit",
                        symbol=decision.symbol,
                        okx_symbol=okx_symbol,
                        error=error_text,
                    )
                    return ExecutionResult(
                        order_id="ticker_unavailable",
                        symbol=decision.symbol,
                        side=side,
                        order_type="market",
                        quantity=0,
                        price=0,
                        status=OrderStatus.REJECTED,
                        raw_response={
                            "error": "OKX 原生行情不可用，系统已在提交前拦截，未向 OKX 发送开仓订单。",
                            "execution_blocker": "okx_native_ticker_unavailable",
                            "system_pre_submit_rejection": True,
                            "okx_rejection": False,
                            "okx_symbol": okx_symbol,
                            "raw_error": error_text,
                        },
                    )
                logger.warning(
                    "OKX native ticker unavailable for exit; using position snapshot price",
                    symbol=decision.symbol,
                    okx_symbol=okx_symbol,
                    error=safe_error_text(exc),
                )
            price = self._safe_float(ticker.get("last"), 0.0)
            market = await self._market_for_symbol(okx_symbol, app_symbol=decision.symbol)
            if price <= 0 and decision.is_exit:
                target_price_side = "long" if decision.action == Action.CLOSE_LONG else "short"
                exit_position_snapshot = await self.get_positions_strict(decision.symbol)
                for position in exit_position_snapshot:
                    if position.get("side") != target_price_side:
                        continue
                    info = position.get("info") or {}
                    price = (
                        self._safe_float(position.get("markPrice"), 0.0)
                        or self._safe_float(info.get("markPx"), 0.0)
                        or self._safe_float(position.get("lastPrice"), 0.0)
                        or self._safe_float(info.get("last"), 0.0)
                        or self._safe_float(position.get("entryPrice"), 0.0)
                        or self._safe_float(info.get("avgPx"), 0.0)
                    )
                    if price > 0:
                        break
            contract_size = self._contract_size(market)
            order_quantity = 0.0
            base_quantity = 0.0
            order_resize_note = None
            market_order_size_adjustment = None
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
                if paper_training_margin_reserve:
                    okx_order_rules["paper_training_margin_execution_reserve"] = dict(
                        paper_training_margin_reserve
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
                positions = exit_position_snapshot or await self.get_positions_strict(
                    decision.symbol
                )
                exit_position_snapshot = positions
                target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
                matching = [
                    p
                    for p in positions
                    if self._position_matches_exit_side(
                        p,
                        target_side,
                        decision_symbol=decision.symbol,
                    )
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
                    diagnostics = self._exit_position_mismatch_diagnostics(
                        positions,
                        decision_symbol=decision.symbol,
                        okx_symbol=okx_symbol,
                        target_side=target_side,
                        exit_side=side,
                        source="pre_submit_position_lookup",
                    )
                    return ExecutionResult(
                        order_id="no_position",
                        symbol=decision.symbol,
                        side=side,
                        order_type="market",
                        quantity=0,
                        price=price,
                        status=OrderStatus.REJECTED,
                        raw_response={
                            "okx_exit_position_mismatch": diagnostics,
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
                            if self._position_matches_exit_side(
                                p,
                                target_side,
                                decision_symbol=decision.symbol,
                            )
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
                    sizing = (
                        decision.raw_response.get("profit_risk_sizing")
                        if isinstance(decision.raw_response, dict)
                        else {}
                    )
                    sizing = sizing if isinstance(sizing, dict) else {}
                    position_value = max(
                        self._safe_float(sizing.get("final_notional_usdt"), 0.0),
                        0.0,
                    )
                    paper_training_margin_reserve = (
                        self._paper_training_margin_execution_reserve(
                            decision,
                            available_balance_usdt=balance,
                            leverage=decision.suggested_leverage,
                            planned_notional_usdt=position_value,
                        )
                    )
                    position_value = self._safe_float(
                        paper_training_margin_reserve.get(
                            "executable_notional_usdt",
                            position_value,
                        ),
                        position_value,
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
                    if paper_training_margin_reserve:
                        okx_order_rules[
                            "paper_training_margin_execution_reserve"
                        ] = dict(paper_training_margin_reserve)
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
                market_size_adjustment = self._entry_market_order_size_adjustment(
                    decision=decision,
                    side=side,
                    price=price,
                    okx_symbol=okx_symbol,
                    contract_size=contract_size,
                    order_quantity=order_quantity,
                    base_quantity=base_quantity,
                    okx_order_rules=okx_order_rules,
                )
                if isinstance(market_size_adjustment, ExecutionResult):
                    return market_size_adjustment
                if market_size_adjustment is not None:
                    order_quantity = market_size_adjustment["adjusted_order_contracts"]
                    base_quantity = market_size_adjustment["adjusted_base_quantity"]
                    market_order_size_adjustment = market_size_adjustment
                    okx_order_rules = dict(okx_order_rules)
                    okx_order_rules["final_contracts"] = round(order_quantity, 12)
                    okx_order_rules["final_base_quantity"] = round(base_quantity, 12)
                    okx_order_rules["final_notional_usdt"] = round(
                        order_quantity * contract_size * max(price, 0.0),
                        8,
                    )
                    okx_order_rules["market_order_within_max_size"] = True
                    okx_order_rules["pre_submit_valid"] = True
                    okx_order_rules["market_order_size_adjustment"] = market_size_adjustment
                    order_resize_note = (
                        f"OKX 单笔市价单上限为 {market_size_adjustment['amount_max_market_contracts']:g} 张，"
                        f"系统已把原计划 {market_size_adjustment['original_planned_order_contracts']:g} 张"
                        f"缩到 {order_quantity:g} 张后提交。"
                    )
                reconciled_contract = reconcile_profit_risk_sizing(
                    decision,
                    final_notional_usdt=order_quantity * contract_size * max(price, 0.0),
                    final_leverage=decision.suggested_leverage,
                    source="okx_pre_submit_order_shape",
                    execution_facts={
                        "okx_symbol": okx_symbol,
                        "price": price,
                        "contract_size": contract_size,
                        "order_contracts": order_quantity,
                        "base_quantity": base_quantity,
                        "okx_order_rules": okx_order_rules,
                        "leverage_check": leverage_check,
                    },
                )
                if reconciled_contract.get("eligible") is not True:
                    return ExecutionResult(
                        order_id="risk_contract_rejected",
                        symbol=decision.symbol,
                        side=side,
                        order_type="market",
                        quantity=0.0,
                        price=price,
                        status=OrderStatus.REJECTED,
                        raw_response={
                            "error": (
                                "The final OKX order shape does not match the authoritative "
                                "risk contract; no order was submitted."
                            ),
                            "execution_blocker": "execution_risk_contract_reconciliation",
                            "system_pre_submit_rejection": True,
                            "okx_rejection": False,
                            "reconciliation_reasons": reconciled_contract.get("reasons"),
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
                paper_training_identity = (
                    decision.raw_response.get("paper_training_order_identity")
                    if isinstance(decision.raw_response, dict)
                    else {}
                )
                paper_training_identity = (
                    paper_training_identity
                    if isinstance(paper_training_identity, dict)
                    else {}
                )
                paper_training_decision_id = paper_training_identity.get("decision_id")
                client_order_id = str(
                    paper_training_identity.get("client_order_id") or ""
                ).strip()
                if (
                    self.executor_mode == "paper"
                    and is_paper_training_decision(decision)
                    and paper_training_identity.get("version")
                    == PAPER_TRAINING_ORDER_IDENTITY_VERSION
                    and paper_training_identity.get("execution_scope") == "paper_only"
                    and paper_training_identity.get("production_permission") is False
                    and client_order_id
                    == paper_training_client_order_id(paper_training_decision_id)
                ):
                    params["clOrdId"] = client_order_id
                    okx_order_rules["client_order_identity"] = dict(
                        paper_training_identity
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
                full_close_result = await self._place_okx_native_full_close(
                    ccxt=ccxt,
                    decision=decision,
                    okx_symbol=okx_symbol,
                    market=market,
                    side=side,
                    params=params,
                    price=price,
                    contract_size=contract_size,
                    pre_exit_contracts=pre_exit_contracts,
                    target_side=target_side,
                    position_side=position_side,
                    requested_exit_fraction=requested_exit_fraction,
                    requested_exit_contracts=requested_exit_contracts,
                    exit_order_replace_note=exit_order_replace_note,
                )
                if full_close_result is not None:
                    return full_close_result
                split_exit_result = await self._place_split_exit_market_orders(
                    ccxt=ccxt,
                    decision=decision,
                    okx_symbol=okx_symbol,
                    side=side,
                    market=market,
                    params=params,
                    price=price,
                    contract_size=contract_size,
                    pre_exit_contracts=pre_exit_contracts,
                    target_side=target_side,
                    requested_exit_fraction=requested_exit_fraction,
                    requested_exit_contracts=requested_exit_contracts,
                    order_quantity=order_quantity,
                    exit_order_replace_note=exit_order_replace_note,
                )
                if split_exit_result is not None:
                    return split_exit_result
                native_reduce_result = await self._place_okx_native_reduce_market_order(
                    ccxt=ccxt,
                    decision=decision,
                    okx_symbol=okx_symbol,
                    market=market,
                    side=side,
                    params=params,
                    price=price,
                    contract_size=contract_size,
                    pre_exit_contracts=pre_exit_contracts,
                    target_side=target_side,
                    position_side=position_side,
                    position_snapshot=exit_position_snapshot or [],
                    requested_exit_fraction=requested_exit_fraction,
                    requested_exit_contracts=requested_exit_contracts,
                    order_quantity=order_quantity,
                    exit_order_replace_note=exit_order_replace_note,
                )
                if native_reduce_result is not None:
                    return native_reduce_result

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
                if decision.is_entry and params.get("attachAlgoOrds"):
                    protection_submit_requested_at = datetime.now(UTC)
                order = await self._create_order_with_client_recovery(
                    ccxt,
                    okx_symbol,
                    side,
                    order_quantity,
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
                        retry_contract = reconcile_profit_risk_sizing(
                            decision,
                            final_notional_usdt=(
                                order_quantity * contract_size * max(price, 0.0)
                            ),
                            final_leverage=decision.suggested_leverage,
                            source="okx_51004_position_limit_retry",
                            execution_facts={
                                "okx_symbol": okx_symbol,
                                "price": price,
                                "contract_size": contract_size,
                                "order_contracts": order_quantity,
                                "original_error": error_text,
                            },
                        )
                        if retry_contract.get("eligible") is not True:
                            return self._entry_exchange_rejection_result(
                                decision=decision,
                                side=side,
                                price=price,
                                exchange_error=(
                                    "OKX retry quantity failed authoritative risk reconciliation"
                                ),
                                okx_symbol=okx_symbol,
                                contract_size=contract_size,
                                order_quantity=order_quantity,
                                base_quantity=base_quantity,
                                okx_order_rules=okx_order_rules,
                                request_params=params,
                                leverage_check=leverage_check,
                            )
                        protection_submit_requested_at = datetime.now(UTC)
                        order = await self._create_order_with_client_recovery(
                            ccxt,
                            okx_symbol,
                            side,
                            order_quantity,
                            params,
                        )
                    else:
                        return self._entry_exchange_rejection_result(
                            decision=decision,
                            side=side,
                            price=price,
                            exchange_error=e,
                            okx_symbol=okx_symbol,
                            contract_size=contract_size,
                            order_quantity=order_quantity,
                            base_quantity=base_quantity,
                            okx_order_rules=okx_order_rules,
                            request_params=params,
                            leverage_check=leverage_check,
                        )
                else:
                    raise
            protection_submission = self._protection_submission_fact(
                params=params,
                order=order,
                requested_at=protection_submit_requested_at,
                acknowledged_at=datetime.now(UTC),
            )
            order = await self._confirm_market_order(ccxt, order, okx_symbol)
            result_symbol = self._execution_result_symbol(order, decision.symbol)
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
                    symbol=result_symbol,
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
                        "decision_symbol": decision.symbol,
                        "canonical_exchange_symbol": result_symbol,
                        "okx_symbol": okx_symbol,
                        "contract_size": contract_size,
                        "order_contracts": order_quantity,
                        "filled_contracts": filled_contracts,
                        "planned_base_quantity": base_quantity,
                        "okx_order_rules": okx_order_rules,
                        "order_status": status.value,
                        **(
                            {"protection_submission": protection_submission}
                            if protection_submission
                            else {}
                        ),
                    },
                )
            filled_base_quantity = filled_contracts * contract_size
            if decision.is_entry and filled_contracts > 0:
                reconcile_profit_risk_sizing(
                    decision,
                    final_notional_usdt=(
                        filled_contracts * contract_size * max(execution_price, 0.0)
                    ),
                    final_leverage=decision.suggested_leverage,
                    source="okx_confirmed_entry_fill",
                    execution_facts={
                        "okx_symbol": okx_symbol,
                        "execution_price": execution_price,
                        "contract_size": contract_size,
                        "filled_contracts": filled_contracts,
                        "order_id": order.get("id") or (order.get("info") or {}).get("ordId"),
                    },
                )

            if decision.is_exit and target_side:
                after_contracts = pre_exit_contracts
                for _ in range(5):
                    await asyncio.sleep(1.0)
                    try:
                        refreshed = await self._fetch_native_order_detail(
                            ccxt,
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
                            symbol=result_symbol,
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
                                "decision_symbol": decision.symbol,
                                "canonical_exchange_symbol": result_symbol,
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
                        symbol=result_symbol,
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
                            "decision_symbol": decision.symbol,
                            "canonical_exchange_symbol": result_symbol,
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

            final_order_id = str(
                order.get("id")
                or (order.get("info") or {}).get("ordId")
                or (order.get("info") or {}).get("clOrdId")
                or ""
            ).strip()
            return ExecutionResult(
                order_id=final_order_id,
                symbol=result_symbol,
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
                exchange_order_id=final_order_id or None,
                timestamp=datetime.now(UTC),
                raw_response={
                    **order,
                    "request_params": params,
                    "decision_symbol": decision.symbol,
                    "canonical_exchange_symbol": result_symbol,
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
                    **(
                        {"protection_submission": protection_submission}
                        if protection_submission
                        else {}
                    ),
                    **(
                        {"market_order_size_adjustment": market_order_size_adjustment}
                        if market_order_size_adjustment is not None
                        else {}
                    ),
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
                diagnostics = self._exit_position_mismatch_diagnostics(
                    exit_position_snapshot or [],
                    decision_symbol=decision.symbol,
                    okx_symbol=okx_symbol,
                    target_side=target_side or "",
                    exit_side=side,
                    source="exchange_no_position_rejection",
                )
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
                        "okx_exit_position_mismatch": diagnostics,
                        "raw_error": error_text,
                    },
                )
            if decision.is_entry:
                return self._entry_exchange_rejection_result(
                    decision=decision,
                    side=side,
                    price=0.0,
                    exchange_error=e,
                    okx_symbol=okx_symbol,
                    contract_size=contract_size,
                    order_quantity=order_quantity,
                    base_quantity=base_quantity,
                    okx_order_rules=okx_order_rules,
                    request_params=params,
                    leverage_check=leverage_check,
                )
            raise
        except RateLimitError:
            raise
        except Exception as e:
            error_text = safe_error_text(e)
            logger.error("order placement failed", error=error_text)
            raise OrderPlacementError(f"Failed to place order: {error_text}") from e

    @staticmethod
    def _protection_submission_fact(
        *,
        params: dict[str, Any],
        order: dict[str, Any],
        requested_at: datetime | None,
        acknowledged_at: datetime,
    ) -> dict[str, Any]:
        requested = params.get("attachAlgoOrds")
        if not isinstance(requested, list) or not requested:
            return {}
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        response_rows = order.get("attachAlgoOrds") or info.get("attachAlgoOrds") or []
        response_rows = [dict(row) for row in response_rows if isinstance(row, dict)]
        algo_ids = [
            str(row.get("attachAlgoId") or row.get("algoId") or "").strip()
            for row in response_rows
            if str(row.get("attachAlgoId") or row.get("algoId") or "").strip()
        ]
        confirmed = bool(algo_ids)
        return {
            "version": "2026-07-15.okx-protection-submission.v1",
            "source_authority": "local_submit_plus_okx_create_order_response",
            "client_submit_requested_at": (
                requested_at.isoformat() if requested_at is not None else None
            ),
            "exchange_acknowledged_at": acknowledged_at.isoformat(),
            "exchange_confirmation_recorded": confirmed,
            "exchange_confirmed_at": acknowledged_at.isoformat() if confirmed else None,
            "state": "confirmed" if confirmed else "submitted_unconfirmed",
            "requested_attach_algo_orders": [
                dict(row) for row in requested if isinstance(row, dict)
            ],
            "response_attach_algo_orders": response_rows,
            "algo_ids": algo_ids,
        }

    async def _place_okx_native_full_close(
        self,
        *,
        ccxt,
        decision: DecisionOutput,
        okx_symbol: str,
        market: dict[str, Any],
        side: str,
        params: dict[str, Any],
        price: float,
        contract_size: float,
        pre_exit_contracts: float,
        target_side: str,
        position_side: str | None,
        requested_exit_fraction: float,
        requested_exit_contracts: float,
        exit_order_replace_note: str | None,
    ) -> ExecutionResult | None:
        if requested_exit_fraction < 0.999 or requested_exit_contracts < pre_exit_contracts * 0.999:
            return None
        close_position = getattr(ccxt, "privatePostTradeClosePosition", None)
        if not callable(close_position):
            return None

        request_params = {
            "instId": str(
                market.get("id") or okx_symbol.replace("/", "-").replace(":USDT", "-SWAP")
            ),
            "mgnMode": str(params.get("tdMode") or params.get("marginMode") or "cross"),
            "autoCxl": params.get("autoCxl", True),
        }
        if position_side and position_side != "net":
            request_params["posSide"] = target_side

        submitted_at_ms = int(time.time() * 1000)
        try:
            response = await self._with_retry(close_position, request_params)
        except ExchangeAPIError as exc:
            logger.warning(
                "OKX native full close failed; will use reduce-only market orders",
                symbol=okx_symbol,
                side=target_side,
                error=safe_error_text(exc),
            )
            return None

        after_contracts = pre_exit_contracts
        snapshot_error: str | None = None
        tolerance = max(pre_exit_contracts * 0.001, 1e-8)
        for _ in range(8):
            await asyncio.sleep(0.75)
            try:
                after_contracts = await self._position_contracts_for_side(
                    decision.symbol,
                    target_side,
                )
            except Exception as exc:
                snapshot_error = safe_error_text(exc)
                logger.warning(
                    "OKX native full close position refresh failed",
                    symbol=okx_symbol,
                    side=target_side,
                    error=snapshot_error,
                )
                break
            if after_contracts <= tolerance:
                break

        closed_contracts = max(pre_exit_contracts - after_contracts, 0.0)
        data = response.get("data") if isinstance(response, dict) else None
        first_item = data[0] if isinstance(data, list) and data else {}
        if not isinstance(first_item, dict):
            first_item = {}
        order_id = str(first_item.get("ordId") or "").strip()
        client_order_id = str(first_item.get("clOrdId") or "").strip()
        response_code = str(response.get("code") if isinstance(response, dict) else "")
        s_code = str(first_item.get("sCode") or "")
        success_code = response_code == "0" or s_code in {"", "0"}
        raw_response = {
            **(response if isinstance(response, dict) else {"response": response}),
            "exit_tracking": True,
            "okx_native_close_position": True,
            "exit_order_replace_note": exit_order_replace_note,
            "request_params": request_params,
            "fallback_market_order_params": params,
            "okx_symbol": okx_symbol,
            "contract_size": contract_size,
            "position_contracts_before": pre_exit_contracts,
            "position_contracts_after": after_contracts,
            "requested_exit_fraction": requested_exit_fraction,
            "requested_exit_contracts": requested_exit_contracts,
            "remaining_contracts": max(after_contracts, 0.0),
            "snapshot_error": snapshot_error,
            "filled_contracts": closed_contracts,
            "base_quantity": closed_contracts * contract_size,
        }
        fill_confirmation = None
        if closed_contracts > tolerance:
            fill_confirmation = await self._native_full_close_fill_confirmation(
                ccxt=ccxt,
                inst_id=str(request_params["instId"]),
                side=side,
                submitted_at_ms=submitted_at_ms,
                expected_contracts=closed_contracts,
                contract_size=contract_size,
            )
            if fill_confirmation:
                order_id = str(fill_confirmation.get("order_id") or order_id or "").strip()
                price = self._safe_float(fill_confirmation.get("price"), price)
                raw_response["native_close_fill"] = {
                    "source": fill_confirmation.get("source"),
                    "order_id": order_id or None,
                    "price": price,
                    "fee": self._safe_float(fill_confirmation.get("fee"), 0.0),
                    "pnl": self._safe_float(fill_confirmation.get("pnl"), 0.0),
                    "contracts": self._safe_float(fill_confirmation.get("contracts"), 0.0),
                    "quantity": self._safe_float(fill_confirmation.get("quantity"), 0.0),
                    "timestamp_ms": fill_confirmation.get("timestamp_ms"),
                    "timestamp": (
                        fill_confirmation["timestamp"].isoformat()
                        if fill_confirmation.get("timestamp") is not None
                        else None
                    ),
                    "order_info": fill_confirmation.get("order_info") or {},
                }
        if not success_code and closed_contracts <= tolerance:
            logger.warning(
                "OKX native full close returned failure; will use reduce-only market orders",
                symbol=okx_symbol,
                side=target_side,
                response=safe_error_text(response, limit=300),
            )
            return None
        if closed_contracts <= tolerance:
            return ExecutionResult(
                order_id=order_id or client_order_id or "okx_native_full_close_not_confirmed",
                symbol=decision.symbol,
                side=side,
                order_type="market",
                quantity=0.0,
                price=price,
                status=OrderStatus.OPEN,
                exchange_order_id=order_id or None,
                timestamp=datetime.now(UTC),
                raw_response={
                    **raw_response,
                    "error": snapshot_error
                    or "OKX native full close was submitted but position is not flat yet.",
                },
            )
        if not order_id:
            return ExecutionResult(
                order_id=client_order_id or "okx_native_full_close_fill_pending",
                symbol=decision.symbol,
                side=side,
                order_type="market",
                quantity=closed_contracts * contract_size,
                price=price,
                status=OrderStatus.PARTIAL,
                fee=0.0,
                pnl=0.0,
                exchange_order_id=None,
                timestamp=datetime.now(UTC),
                raw_response={
                    **raw_response,
                    "requires_okx_fill_backfill": True,
                    "error": (
                        "OKX native full close flattened the exchange position, but OKX "
                        "fills-history did not expose the real ordId yet. Local ledger "
                        "will wait for authoritative OKX fill sync instead of closing with "
                        "a synthetic order id."
                    ),
                },
            )

        return ExecutionResult(
            order_id=order_id or client_order_id or "okx_native_full_close",
            symbol=decision.symbol,
            side=side,
            order_type="market",
            quantity=closed_contracts * contract_size,
            price=price,
            status=OrderStatus.FILLED if after_contracts <= tolerance else OrderStatus.PARTIAL,
            fee=self._safe_float(
                (fill_confirmation or {}).get("fee"),
                0.0,
            ),
            pnl=self._safe_float(
                (fill_confirmation or {}).get("pnl"),
                0.0,
            ),
            exchange_order_id=order_id or None,
            timestamp=(fill_confirmation or {}).get("timestamp") or datetime.now(UTC),
            raw_response=raw_response,
        )

    async def _native_full_close_fill_confirmation(
        self,
        *,
        ccxt,
        inst_id: str,
        side: str,
        submitted_at_ms: int,
        expected_contracts: float,
        contract_size: float,
    ) -> dict[str, Any] | None:
        fetch_fills = getattr(ccxt, "privateGetTradeFillsHistory", None)
        if not callable(fetch_fills) or expected_contracts <= 0:
            return None

        try:
            response = await self._with_retry(
                fetch_fills,
                {
                    "instType": "SWAP",
                    "instId": inst_id,
                    "limit": "100",
                },
            )
        except Exception as exc:
            logger.warning(
                "OKX native full close instrument fill lookup failed; trying account-wide history",
                inst_id=inst_id,
                side=side,
                error=safe_error_text(exc),
            )
            try:
                response = await self._with_retry(
                    fetch_fills,
                    {
                        "instType": "SWAP",
                        "limit": "100",
                    },
                )
            except Exception as fallback_exc:
                logger.warning(
                    "OKX native full close fill confirmation failed",
                    inst_id=inst_id,
                    side=side,
                    error=safe_error_text(fallback_exc),
                )
                return None

        rows = response.get("data", []) if isinstance(response, dict) else []
        min_timestamp = max(int(submitted_at_ms or 0) - 30_000, 0)
        groups: dict[str, dict[str, Any]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            row_inst_id = str(row.get("instId") or "").strip()
            if (
                row_inst_id
                and row_inst_id != inst_id
                and not row_inst_id.startswith(f"{inst_id}-OFF")
            ):
                continue
            if str(row.get("side") or "").lower() != str(side or "").lower():
                continue
            timestamp_ms = self._safe_float(row.get("ts") or row.get("fillTime"), 0.0)
            if timestamp_ms > 0 and timestamp_ms < min_timestamp:
                continue
            contracts = self._safe_float(row.get("fillSz") or row.get("sz"), 0.0)
            price = self._safe_float(row.get("fillPx") or row.get("price"), 0.0)
            order_id = str(row.get("ordId") or "").strip()
            if contracts <= 0 or price <= 0 or not order_id:
                continue
            group = groups.setdefault(
                order_id,
                {
                    "order_id": order_id,
                    "contracts": 0.0,
                    "price_value": 0.0,
                    "fee": 0.0,
                    "pnl": 0.0,
                    "timestamp_ms": timestamp_ms,
                    "order_info": row,
                    "source": "okx_fills_history_after_native_close",
                },
            )
            group["contracts"] += contracts
            group["price_value"] += price * contracts
            group["fee"] += abs(self._safe_float(row.get("fee"), 0.0))
            group["pnl"] += self._safe_float(row.get("fillPnl") or row.get("pnl"), 0.0)
            if timestamp_ms >= self._safe_float(group.get("timestamp_ms"), 0.0):
                group["timestamp_ms"] = timestamp_ms
                group["order_info"] = row

        candidates = []
        for group in groups.values():
            contracts = self._safe_float(group.get("contracts"), 0.0)
            if contracts <= 0:
                continue
            timestamp_ms = self._safe_float(group.get("timestamp_ms"), 0.0)
            candidates.append(
                {
                    **group,
                    "price": self._safe_float(group.get("price_value"), 0.0) / contracts,
                    "quantity": contracts * (contract_size if contract_size > 0 else 1.0),
                    "timestamp": (
                        datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
                        if timestamp_ms > 0
                        else None
                    ),
                }
            )
        if not candidates:
            return None
        tolerance = max(abs(expected_contracts) * 0.05, 1e-8)
        return sorted(
            candidates,
            key=lambda item: (
                abs(self._safe_float(item.get("contracts"), 0.0) - expected_contracts) > tolerance,
                abs(self._safe_float(item.get("contracts"), 0.0) - expected_contracts),
                -self._safe_float(item.get("timestamp_ms"), 0.0),
            ),
        )[0]

    async def _place_split_exit_market_orders(
        self,
        *,
        ccxt,
        decision: DecisionOutput,
        okx_symbol: str,
        side: str,
        market: dict[str, Any],
        params: dict[str, Any],
        price: float,
        contract_size: float,
        pre_exit_contracts: float,
        target_side: str,
        requested_exit_fraction: float,
        requested_exit_contracts: float,
        order_quantity: float,
        exit_order_replace_note: str | None,
    ) -> ExecutionResult | None:
        chunks = self._exit_market_order_slices(ccxt, market, order_quantity)
        if not chunks:
            return None

        max_market_contracts = self._amount_market_max(market)
        tolerance = max(pre_exit_contracts * 0.001, 1e-8)
        chunk_results: list[dict[str, Any]] = []
        order_ids: list[str] = []
        total_fee = 0.0
        weighted_price_total = 0.0
        weighted_contracts = 0.0
        last_order: dict[str, Any] = {}
        last_status = OrderStatus.PENDING
        after_contracts = pre_exit_contracts
        split_error: str | None = None

        for index, planned_contracts in enumerate(chunks, start=1):
            closed_so_far = max(pre_exit_contracts - after_contracts, 0.0)
            remaining_request = max(requested_exit_contracts - closed_so_far, 0.0)
            if after_contracts <= tolerance or remaining_request <= tolerance:
                break

            chunk_contracts = min(planned_contracts, remaining_request, after_contracts)
            chunk_contracts = self._clamp_exit_chunk_contracts(ccxt, market, chunk_contracts)
            if chunk_contracts <= 0:
                break

            before_chunk_contracts = after_contracts
            try:
                order = await self._with_retry(
                    ccxt.create_order,
                    okx_symbol,
                    "market",
                    side,
                    chunk_contracts,
                    None,
                    params,
                )
                order = await self._confirm_market_order(ccxt, order, okx_symbol)
            except ExchangeAPIError as exc:
                split_error = safe_error_text(exc)
                logger.warning(
                    "split exit market order rejected by OKX",
                    symbol=okx_symbol,
                    chunk_index=index,
                    chunk_contracts=chunk_contracts,
                    error=split_error,
                )
                break

            last_order = order
            last_status = self._order_status_from_ccxt(
                order.get("status") or (order.get("info") or {}).get("state")
            )
            order_id = str(order.get("id") or (order.get("info") or {}).get("ordId") or "")
            if order_id:
                order_ids.append(order_id)
            total_fee += self._order_fee_cost(order)
            order_price = self._safe_float(order.get("average") or order.get("price"), price)

            for _ in range(5):
                await asyncio.sleep(0.5)
                try:
                    after_contracts = await self._position_contracts_for_side(
                        decision.symbol,
                        target_side,
                    )
                except Exception as exc:
                    split_error = safe_error_text(exc)
                    logger.warning(
                        "split exit position refresh failed",
                        symbol=okx_symbol,
                        chunk_index=index,
                        error=split_error,
                    )
                    break
                if before_chunk_contracts - after_contracts > tolerance:
                    break
                if last_status in {OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                    break

            chunk_closed = max(before_chunk_contracts - after_contracts, 0.0)
            if chunk_closed > tolerance and order_price > 0:
                weighted_price_total += order_price * chunk_closed
                weighted_contracts += chunk_closed
            chunk_results.append(
                {
                    "index": index,
                    "order_id": order_id or None,
                    "requested_contracts": chunk_contracts,
                    "status": last_status.value,
                    "fee": self._order_fee_cost(order),
                    "price": order_price,
                    "position_contracts_before": before_chunk_contracts,
                    "position_contracts_after": after_contracts,
                    "closed_contracts": chunk_closed,
                }
            )

            if chunk_closed <= tolerance:
                break

        closed_contracts = max(pre_exit_contracts - after_contracts, 0.0)
        execution_price = (
            weighted_price_total / weighted_contracts if weighted_contracts > 0 else price
        )
        raw_response = {
            **last_order,
            "exit_tracking": True,
            "split_exit_order": True,
            "split_reason": "okx_market_order_max_contracts",
            "split_error": split_error,
            "exit_order_replace_note": exit_order_replace_note,
            "request_params": params,
            "okx_symbol": okx_symbol,
            "contract_size": contract_size,
            "order_contracts": order_quantity,
            "amount_max_market_contracts": max_market_contracts,
            "split_chunks": chunk_results,
            "position_contracts_before": pre_exit_contracts,
            "position_contracts_after": after_contracts,
            "requested_exit_fraction": requested_exit_fraction,
            "requested_exit_contracts": requested_exit_contracts,
            "remaining_contracts": max(after_contracts, 0.0),
            "message": (
                "OKX limits this symbol's single market close size, so the reduce-only "
                "close was submitted as several market orders within maxMktSz."
            ),
        }

        if closed_contracts <= tolerance:
            return ExecutionResult(
                order_id=",".join(order_ids) or "split_exit_not_confirmed",
                symbol=decision.symbol,
                side=side,
                order_type="market",
                quantity=0.0,
                price=execution_price,
                status=(
                    OrderStatus.REJECTED
                    if split_error or last_status == OrderStatus.REJECTED
                    else OrderStatus.OPEN
                ),
                fee=total_fee,
                exchange_order_id=",".join(order_ids) or None,
                timestamp=datetime.now(UTC),
                raw_response={
                    **raw_response,
                    "error": split_error
                    or "Split exit orders did not reduce the OKX position yet.",
                },
            )

        requested_filled = (
            requested_exit_contracts <= 0
            or closed_contracts + tolerance >= requested_exit_contracts
        )
        return ExecutionResult(
            order_id=",".join(order_ids) or "split_exit",
            symbol=decision.symbol,
            side=side,
            order_type="market",
            quantity=closed_contracts * contract_size,
            price=execution_price,
            status=OrderStatus.FILLED if requested_filled else OrderStatus.PARTIAL,
            fee=total_fee,
            exchange_order_id=",".join(order_ids) or None,
            timestamp=datetime.now(UTC),
            raw_response={
                **raw_response,
                "filled_contracts": closed_contracts,
                "base_quantity": closed_contracts * contract_size,
            },
        )

    async def _place_okx_native_reduce_market_order(
        self,
        *,
        ccxt,
        decision: DecisionOutput,
        okx_symbol: str,
        market: dict[str, Any],
        side: str,
        params: dict[str, Any],
        price: float,
        contract_size: float,
        pre_exit_contracts: float,
        target_side: str,
        position_side: str | None,
        position_snapshot: list[dict[str, Any]] | None,
        requested_exit_fraction: float,
        requested_exit_contracts: float,
        order_quantity: float,
        exit_order_replace_note: str | None,
    ) -> ExecutionResult | None:
        native_required = bool(
            market.get("synthetic_from_position") or str(okx_symbol).upper().endswith("-SWAP")
        )
        if not native_required:
            return None
        place_order = getattr(ccxt, "privatePostTradeOrder", None)
        if not callable(place_order):
            return None

        tolerance = max(pre_exit_contracts * 0.001, 1e-8)
        contracts = min(
            max(self._safe_float(order_quantity, 0.0), 0.0),
            max(self._safe_float(requested_exit_contracts, 0.0), 0.0),
            max(pre_exit_contracts, 0.0),
        )
        if contracts <= tolerance:
            return None

        inst_id = self._native_inst_id_for_market(market, okx_symbol)
        request_params = {
            "instId": inst_id,
            "tdMode": str(params.get("tdMode") or params.get("marginMode") or "cross"),
            "side": side,
            "ordType": "market",
            "sz": self._format_okx_number(contracts),
            "reduceOnly": "true",
        }
        if position_side and position_side != "net":
            request_params["posSide"] = target_side

        lock_reason = self._contract_delivery_lock_reason(inst_id)
        if lock_reason:
            return self._contract_delivery_rejected_result(
                decision=decision,
                side=side,
                price=price,
                request_params=request_params,
                params=params,
                okx_symbol=okx_symbol,
                reason=lock_reason,
                lock_hit=True,
            )

        try:
            response = await self._with_retry(place_order, request_params)
        except ExchangeAPIError as exc:
            error_text = safe_error_text(exc)
            if self._is_no_position_error(error_text):
                diagnostics = self._exit_position_mismatch_diagnostics(
                    position_snapshot or [],
                    decision_symbol=decision.symbol,
                    okx_symbol=okx_symbol,
                    target_side=target_side,
                    exit_side=side,
                    source="native_reduce_no_position_rejection",
                )
                return ExecutionResult(
                    order_id="no_position",
                    symbol=decision.symbol,
                    side=side,
                    order_type="market",
                    quantity=0.0,
                    price=price,
                    status=OrderStatus.REJECTED,
                    timestamp=datetime.now(UTC),
                    raw_response={
                        "error": error_text,
                        "okx_native_reduce_market_order": True,
                        "okx_exit_position_mismatch": diagnostics,
                        "request_params": request_params,
                        "fallback_market_order_params": params,
                        "okx_symbol": okx_symbol,
                        "do_not_persist_order": True,
                    },
                )
            if self._is_contract_delivery_error(error_text):
                self._remember_contract_delivery_lock(inst_id, error_text)
                return self._contract_delivery_rejected_result(
                    decision=decision,
                    side=side,
                    price=price,
                    request_params=request_params,
                    params=params,
                    okx_symbol=okx_symbol,
                    reason=error_text,
                    lock_hit=False,
                )
            logger.warning(
                "OKX native reduce-only market order failed",
                symbol=okx_symbol,
                side=target_side,
                error=error_text,
            )
            return ExecutionResult(
                order_id="native_reduce_rejected",
                symbol=decision.symbol,
                side=side,
                order_type="market",
                quantity=0.0,
                price=price,
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(UTC),
                raw_response={
                    "error": error_text,
                    "okx_native_reduce_market_order": True,
                    "request_params": request_params,
                    "fallback_market_order_params": params,
                    "okx_symbol": okx_symbol,
                    "do_not_persist_order": True,
                },
            )

        after_contracts = pre_exit_contracts
        snapshot_error: str | None = None
        for _ in range(8):
            await asyncio.sleep(0.75)
            try:
                after_contracts = await self._position_contracts_for_side(
                    decision.symbol,
                    target_side,
                )
            except Exception as exc:
                snapshot_error = safe_error_text(exc)
                logger.warning(
                    "OKX native reduce-only position refresh failed",
                    symbol=okx_symbol,
                    side=target_side,
                    error=snapshot_error,
                )
                break
            if pre_exit_contracts - after_contracts > tolerance:
                break

        closed_contracts = max(pre_exit_contracts - after_contracts, 0.0)
        data = response.get("data") if isinstance(response, dict) else None
        first_item = data[0] if isinstance(data, list) and data else {}
        if not isinstance(first_item, dict):
            first_item = {}
        order_id = str(first_item.get("ordId") or first_item.get("clOrdId") or "").strip()
        response_code = str(response.get("code") if isinstance(response, dict) else "")
        s_code = str(first_item.get("sCode") or "")
        success_code = response_code == "0" or s_code in {"", "0"}
        raw_response = {
            **(response if isinstance(response, dict) else {"response": response}),
            "exit_tracking": True,
            "okx_native_reduce_market_order": True,
            "exit_order_replace_note": exit_order_replace_note,
            "request_params": request_params,
            "fallback_market_order_params": params,
            "okx_symbol": okx_symbol,
            "canonical_exchange_symbol": normalize_trading_symbol(request_params["instId"]),
            "contract_size": contract_size,
            "order_contracts": contracts,
            "position_contracts_before": pre_exit_contracts,
            "position_contracts_after": after_contracts,
            "requested_exit_fraction": requested_exit_fraction,
            "requested_exit_contracts": requested_exit_contracts,
            "remaining_contracts": max(after_contracts, 0.0),
            "snapshot_error": snapshot_error,
            "filled_contracts": closed_contracts,
            "base_quantity": closed_contracts * contract_size,
        }
        if not success_code and closed_contracts <= tolerance:
            return ExecutionResult(
                order_id=order_id or "native_reduce_rejected",
                symbol=decision.symbol,
                side=side,
                order_type="market",
                quantity=0.0,
                price=price,
                status=OrderStatus.REJECTED,
                exchange_order_id=order_id or None,
                timestamp=datetime.now(UTC),
                raw_response={**raw_response, "do_not_persist_order": True},
            )
        if closed_contracts <= tolerance:
            return ExecutionResult(
                order_id=order_id or "native_reduce_not_confirmed",
                symbol=decision.symbol,
                side=side,
                order_type="market",
                quantity=0.0,
                price=price,
                status=OrderStatus.OPEN,
                exchange_order_id=order_id or None,
                timestamp=datetime.now(UTC),
                raw_response={
                    **raw_response,
                    "error": snapshot_error
                    or "OKX native reduce-only order was submitted but position is unchanged.",
                },
            )

        requested_filled = (
            requested_exit_contracts <= 0
            or closed_contracts + tolerance >= requested_exit_contracts
        )
        return ExecutionResult(
            order_id=order_id or "native_reduce_market",
            symbol=decision.symbol,
            side=side,
            order_type="market",
            quantity=closed_contracts * contract_size,
            price=price,
            status=OrderStatus.FILLED if requested_filled else OrderStatus.PARTIAL,
            exchange_order_id=order_id or None,
            timestamp=datetime.now(UTC),
            raw_response=raw_response,
        )

    async def _confirm_market_order(self, ccxt, order: dict, symbol: str) -> dict:
        """Fetch the final OKX-native order state after market order submission."""
        info = order.get("info") if isinstance(order.get("info"), dict) else {}
        order_id = order.get("id") or info.get("ordId") or info.get("clOrdId")
        if not order_id:
            return order

        await asyncio.sleep(0.5)
        try:
            confirmed = await self._fetch_native_order_detail(ccxt, order_id, symbol)
            if confirmed:
                return {**order, **confirmed}
        except Exception as e:
            logger.warning(
                "OKX native order confirmation failed; keeping initial order status",
                order_id=order_id,
                symbol=symbol,
                error=safe_error_text(e),
            )
            if str(order.get("status") or "").lower() in {
                "closed",
                "filled",
                "partially_filled",
                "partial",
            }:
                return {
                    **order,
                    "status": "open",
                    "filled": 0.0,
                    "okx_native_order_detail_unavailable": True,
                }
        return order

    async def _fetch_native_order_detail(
        self,
        ccxt: Any,
        order_id: Any,
        symbol: str,
    ) -> dict[str, Any] | None:
        """Read one order through OKX native instId/ordId instead of CCXT symbol aliases."""

        native_fetch = getattr(ccxt, "privateGetTradeOrder", None)
        if not callable(native_fetch):
            raise ExchangeAPIError("OKX native order detail API is unavailable")
        ord_id = str(order_id or "").strip()
        inst_id = okx_inst_id_from_symbol(symbol)
        if not ord_id or not inst_id:
            raise ExchangeAPIError(
                f"Cannot fetch OKX native order detail: instId={inst_id!r}, ordId={ord_id!r}"
            )
        response = await self._with_retry(
            native_fetch,
            {
                "instId": inst_id,
                "ordId": ord_id,
            },
        )
        rows = response.get("data") if isinstance(response, dict) else None
        row = rows[0] if isinstance(rows, list) and rows else {}
        if not isinstance(row, dict):
            return None
        return self._native_order_detail_to_execution_order(row, symbol=symbol)

    async def _create_order_with_client_recovery(
        self,
        ccxt: Any,
        symbol: str,
        side: str,
        quantity: float,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit once logically and recover an ambiguous OKX response by ``clOrdId``."""

        try:
            return await self._with_retry(
                ccxt.create_order,
                symbol,
                "market",
                side,
                quantity,
                None,
                params,
            )
        except ExchangeAPIError as submit_error:
            client_order_id = str(params.get("clOrdId") or "").strip()
            native_fetch = getattr(ccxt, "privateGetTradeOrder", None)
            inst_id = okx_inst_id_from_symbol(symbol)
            if not client_order_id or not callable(native_fetch) or not inst_id:
                raise

            for attempt in range(3):
                if attempt:
                    await asyncio.sleep(0.35 * attempt)
                try:
                    response = await self._with_retry(
                        native_fetch,
                        {
                            "instId": inst_id,
                            "clOrdId": client_order_id,
                        },
                    )
                except Exception as recovery_error:
                    logger.debug(
                        "OKX client order recovery query not ready",
                        symbol=symbol,
                        client_order_id=client_order_id,
                        attempt=attempt + 1,
                        error=safe_error_text(recovery_error),
                    )
                    continue
                rows = response.get("data") if isinstance(response, dict) else None
                row = rows[0] if isinstance(rows, list) and rows else {}
                if not isinstance(row, dict):
                    continue
                if str(row.get("clOrdId") or "").strip() != client_order_id:
                    continue
                if str(row.get("side") or "").strip().lower() != str(side).lower():
                    continue
                if not str(row.get("ordId") or "").strip():
                    continue
                logger.warning(
                    "ambiguous OKX submit recovered by client order identity",
                    symbol=symbol,
                    side=side,
                    client_order_id=client_order_id,
                    order_id=str(row.get("ordId") or ""),
                    original_error=safe_error_text(submit_error),
                )
                return self._native_order_detail_to_execution_order(row, symbol=symbol)
            raise submit_error

    def _native_order_detail_to_execution_order(
        self,
        row: dict[str, Any],
        *,
        symbol: str,
    ) -> dict[str, Any]:
        inst_id = str(row.get("instId") or okx_inst_id_from_symbol(symbol) or "").strip()
        ord_id = str(row.get("ordId") or row.get("clOrdId") or "").strip()
        side = str(row.get("side") or "").lower()
        order_type = str(row.get("ordType") or "").lower() or "market"
        state = str(row.get("state") or "").lower()
        avg_px = self._safe_float(row.get("avgPx"), 0.0)
        px = self._safe_float(row.get("px"), 0.0)
        fee = abs(self._safe_float(row.get("fee"), 0.0))
        return {
            "id": ord_id,
            "symbol": inst_id or symbol,
            "side": side,
            "type": order_type,
            "status": state,
            "amount": self._safe_float(row.get("sz"), 0.0),
            "filled": self._safe_float(row.get("accFillSz") or row.get("fillSz"), 0.0),
            "price": px or avg_px,
            "average": avg_px or px,
            "fee": {"cost": fee} if fee else {},
            "info": dict(row),
            "okx_native_order_detail": True,
            "canonical_exchange_symbol": normalize_trading_symbol(inst_id or symbol),
        }

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
        info_value = self._market_info_float(
            market,
            "ctVal",
            "contractSize",
            "contract_size",
        )
        market_value = self._safe_float(market.get("contractSize"), 0.0)
        if info_value > 0 and (
            market_value <= 0 or abs(info_value - market_value) > max(info_value, market_value) * 1e-12
        ):
            return info_value
        if market_value > 0:
            return market_value
        return info_value if info_value > 0 else 1.0

    def _native_inst_id_for_market(self, market: dict[str, Any], okx_symbol: str) -> str:
        info = market.get("info") if isinstance(market.get("info"), dict) else {}
        inst_id = str(info.get("instId") or market.get("id") or "").strip()
        if inst_id:
            return inst_id
        return str(okx_symbol or "").replace("/", "-").replace(":USDT", "-SWAP")

    async def _fetch_native_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch OKX public ticker by native instId for execution sizing."""

        inst_id = okx_inst_id_from_symbol(symbol)
        if not inst_id:
            raise ExchangeAPIError(f"Cannot resolve OKX instId for ticker: {symbol}")
        ccxt = await self._get_ccxt()
        fetch_ticker = getattr(ccxt, "publicGetMarketTicker", None)
        if not callable(fetch_ticker):
            raise ExchangeAPIError("OKX native market ticker API is unavailable")
        response = await self._with_retry(fetch_ticker, {"instId": inst_id})
        rows = response.get("data") if isinstance(response, dict) else None
        row = rows[0] if isinstance(rows, list) and rows else {}
        if not isinstance(row, dict):
            row = {}
        last = self._safe_float(row.get("last") or row.get("lastPx"), 0.0)
        if last <= 0:
            raise ExchangeAPIError(f"OKX native ticker has no positive last price: {inst_id}")
        return {
            "symbol": normalize_trading_symbol(inst_id),
            "id": inst_id,
            "last": last,
            "bid": self._safe_float(row.get("bidPx"), 0.0),
            "ask": self._safe_float(row.get("askPx"), 0.0),
            "timestamp": int(self._safe_float(row.get("ts"), 0.0)),
            "info": dict(row),
        }

    @staticmethod
    def _format_okx_number(value: float) -> str:
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return str(value)
        return format(decimal_value.normalize(), "f").rstrip("0").rstrip(".") or "0"

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _exit_position_mismatch_diagnostics(
        self,
        positions: list[dict[str, Any]] | None,
        *,
        decision_symbol: str,
        okx_symbol: str,
        target_side: str,
        exit_side: str,
        source: str,
    ) -> dict[str, Any]:
        expected_symbol = normalize_trading_symbol(decision_symbol)
        expected_inst_id = okx_inst_id_from_symbol(decision_symbol)
        rows = [
            self._exit_position_candidate_diagnostic(
                position,
                expected_symbol=expected_symbol,
                target_side=target_side,
            )
            for position in positions or []
            if isinstance(position, dict)
        ]
        matching_contracts = [
            self._safe_float(row.get("contracts"), 0.0)
            for row in rows
            if row.get("matches_expected_symbol")
            and row.get("matches_target_side")
            and self._safe_float(row.get("contracts"), 0.0) > 0
        ]
        nonzero_same_symbol_sides = sorted(
            {
                str(row.get("side") or "")
                for row in rows
                if row.get("matches_expected_symbol")
                and self._safe_float(row.get("contracts"), 0.0) > 0
            }
        )
        return {
            "source": source,
            "decision_symbol": decision_symbol,
            "normalized_decision_symbol": expected_symbol,
            "expected_okx_inst_id": expected_inst_id,
            "okx_symbol": okx_symbol,
            "target_position_side": target_side,
            "exit_order_side": exit_side,
            "positions_returned": len(rows),
            "matching_position_count": len(matching_contracts),
            "matching_contracts_total": round(sum(matching_contracts), 12),
            "nonzero_same_symbol_sides": nonzero_same_symbol_sides,
            "candidates": rows[:20],
            "candidate_limit_applied": len(rows) > 20,
        }

    def _exit_position_candidate_diagnostic(
        self,
        position: dict[str, Any],
        *,
        expected_symbol: str,
        target_side: str,
    ) -> dict[str, Any]:
        info = position.get("info") if isinstance(position.get("info"), dict) else {}
        snapshot = parse_exchange_position_snapshot(
            position,
            symbol_normalizer=normalize_trading_symbol,
        )
        symbol = normalize_trading_symbol(
            (snapshot or {}).get("symbol")
            or info.get("instId")
            or position.get("symbol")
            or ""
        )
        raw_symbol = str(
            (snapshot or {}).get("raw_symbol")
            or info.get("instId")
            or position.get("symbol")
            or ""
        ).strip()
        side = str(
            (snapshot or {}).get("side")
            or position.get("side")
            or info.get("posSide")
            or ""
        ).lower()
        contracts = abs(
            self._safe_float((snapshot or {}).get("contracts"), 0.0)
            or self._position_contracts(position)
        )
        quantity = abs(self._safe_float((snapshot or {}).get("quantity"), 0.0))
        contract_size = abs(
            self._safe_float((snapshot or {}).get("contract_size"), 0.0)
            or self._safe_float(position.get("contractSize"), 0.0)
            or self._safe_float(info.get("ctVal"), 0.0)
        )
        matches_symbol = bool(symbol and expected_symbol and symbol == expected_symbol)
        matches_side = bool(side and target_side and side == target_side)
        reasons: list[str] = []
        if not matches_symbol:
            reasons.append("symbol_mismatch")
        if not matches_side:
            reasons.append("side_mismatch")
        if contracts <= 0 and quantity <= 0:
            reasons.append("zero_contracts")
        if not reasons:
            reasons.append("matches")
        return {
            "raw_symbol": raw_symbol,
            "symbol": symbol,
            "expected_symbol": expected_symbol,
            "matches_expected_symbol": matches_symbol,
            "side": side,
            "target_side": target_side,
            "matches_target_side": matches_side,
            "contracts": contracts,
            "quantity": quantity,
            "contract_size": contract_size,
            "entry_price": self._safe_float((snapshot or {}).get("entry_price"), 0.0),
            "mark_price": self._safe_float((snapshot or {}).get("mark_price"), 0.0),
            "upl": self._safe_float((snapshot or {}).get("upl"), 0.0),
            "reason": ",".join(reasons),
        }

    def _is_contract_delivery_error(self, message: Any) -> bool:
        text = str(message or "").lower()
        return "51028" in text or "contract under delivery" in text

    def _contract_delivery_lock_reason(self, inst_id: str) -> str | None:
        key = str(inst_id or "").strip().upper()
        if not key:
            return None
        item = self._contract_delivery_locks.get(key)
        if item is None:
            return None
        locked_at, reason = item
        if time.monotonic() - locked_at > OKX_CONTRACT_DELIVERY_LOCK_SECONDS:
            self._contract_delivery_locks.pop(key, None)
            return None
        return reason

    def _remember_contract_delivery_lock(self, inst_id: str, reason: str) -> None:
        key = str(inst_id or "").strip().upper()
        if not key:
            return
        self._contract_delivery_locks[key] = (time.monotonic(), reason)

    def _contract_delivery_rejected_result(
        self,
        *,
        decision: DecisionOutput,
        side: str,
        price: float,
        request_params: dict[str, Any],
        params: dict[str, Any],
        okx_symbol: str,
        reason: str,
        lock_hit: bool,
    ) -> ExecutionResult:
        return ExecutionResult(
            order_id="contract_delivery_paused",
            symbol=decision.symbol,
            side=side,
            order_type="market",
            quantity=0.0,
            price=price,
            status=OrderStatus.REJECTED,
            timestamp=datetime.now(UTC),
            raw_response={
                "error": reason,
                "okx_contract_delivery_cooldown": True,
                "okx_contract_delivery_lock_hit": lock_hit,
                "okx_native_reduce_market_order": True,
                "request_params": request_params,
                "fallback_market_order_params": params,
                "okx_symbol": okx_symbol,
                "do_not_persist_order": True,
            },
        )

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

    def _position_matches_exit_side(
        self,
        position: dict[str, Any],
        target_side: str,
        *,
        decision_symbol: str | None = None,
    ) -> bool:
        expected_symbol = normalize_trading_symbol(decision_symbol)
        snapshot = parse_exchange_position_snapshot(
            position,
            symbol_normalizer=normalize_trading_symbol,
        )
        if snapshot:
            if expected_symbol and snapshot.get("symbol") != expected_symbol:
                return False
            side = str(snapshot.get("side") or "").lower()
            contracts = self._safe_float(snapshot.get("contracts"), 0.0)
            quantity = self._safe_float(snapshot.get("quantity"), 0.0)
            return side == target_side and max(abs(contracts), abs(quantity)) > 0
        info = position.get("info") if isinstance(position.get("info"), dict) else {}
        if expected_symbol:
            symbol = normalize_trading_symbol(info.get("instId") or position.get("symbol"))
            if not symbol or symbol != expected_symbol:
                return False
        side = str(position.get("side") or info.get("posSide") or "").lower()
        return side == target_side and self._position_contracts(position) > 0

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
            if self._position_matches_exit_side(position, side, decision_symbol=symbol):
                return self._position_contracts(position)
        return 0.0

    async def _find_active_exit_order(self, ccxt, okx_symbol: str, side: str) -> dict | None:
        """Return an active reduce-only close order for the same symbol and side."""
        try:
            orders = await self.get_open_orders_strict(okx_symbol)
        except Exception as e:
            logger.warning(
                "fetch native open exit orders failed",
                symbol=okx_symbol,
                side=side,
                error=safe_error_text(e),
            )
            raise

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
            orders = await self.get_open_orders_strict(okx_symbol)
        except Exception as e:
            logger.warning(
                "fetch native open entry orders failed",
                symbol=okx_symbol,
                side=side,
                error=safe_error_text(e),
            )
            raise

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
            await self._cancel_order_native(ccxt, order_id, okx_symbol)
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
        del balance, leverage

        contract_size = self._contract_size(market)
        planned_contracts = position_value / (price * contract_size)
        amount_min = self._amount_min(market)
        min_contracts = amount_min if amount_min > 0 else 0.0
        contracts = planned_contracts

        # Exchange minimums may reject a risk-sized order, but must never enlarge it.
        if min_contracts > 0 and planned_contracts < min_contracts:
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

    def _paper_training_margin_execution_reserve(
        self,
        decision: DecisionOutput,
        *,
        available_balance_usdt: float,
        leverage: float,
        planned_notional_usdt: float,
    ) -> dict[str, Any]:
        """Reserve current modeled execution costs without imposing a training risk cap."""

        if self.executor_mode != "paper" or not is_paper_training_decision(decision):
            return {}
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opportunity = raw.get("opportunity_score")
        opportunity = opportunity if isinstance(opportunity, dict) else {}
        sizing = raw.get("profit_risk_sizing")
        sizing = sizing if isinstance(sizing, dict) else {}
        execution_cost = opportunity.get("execution_cost")
        if not isinstance(execution_cost, dict) or not execution_cost:
            execution_cost = sizing.get("execution_cost")
        execution_cost = execution_cost if isinstance(execution_cost, dict) else {}
        fee_pct = max(self._safe_float(execution_cost.get("fee_pct"), 0.0), 0.0)
        slippage_pct = max(
            self._safe_float(execution_cost.get("slippage_pct"), 0.0),
            0.0,
        )
        total_pct = max(
            self._safe_float(execution_cost.get("total_pct"), 0.0),
            fee_pct + slippage_pct,
            0.0,
        )
        reserve_fraction = total_pct / 100.0
        balance = max(self._safe_float(available_balance_usdt, 0.0), 0.0)
        effective_leverage = max(self._safe_float(leverage, 1.0), 1.0)
        planned = max(self._safe_float(planned_notional_usdt, 0.0), 0.0)
        denominator = (1.0 / effective_leverage) + reserve_fraction
        margin_feasible = balance / denominator if denominator > 0 else 0.0
        executable = min(planned, margin_feasible)
        return {
            "version": "2026-07-22.paper-training-margin-reserve.v1",
            "execution_scope": "paper_only",
            "production_permission": False,
            "risk_cap_applied": False,
            "reserve_source": "current_size_aware_execution_cost",
            "available_balance_usdt": round(balance, 8),
            "leverage": round(effective_leverage, 8),
            "planned_notional_usdt": round(planned, 8),
            "execution_cost_pct": round(total_pct, 8),
            "execution_cost_reserve_fraction": round(reserve_fraction, 10),
            "margin_feasible_notional_usdt": round(margin_feasible, 8),
            "executable_notional_usdt": round(executable, 8),
            "reserve_usdt": round(max(planned - executable, 0.0), 8),
            "applied": executable + 1e-8 < planned,
        }

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
        amount_market_max = self._amount_market_max(market)
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
            "amount_max_market_contracts": round(amount_market_max, 12),
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
            "planned_below_minimum_contracts": bool(
                amount_min > 0 and 0 < planned_contracts < amount_min
            ),
            "market_order_within_max_size": bool(
                amount_market_max <= 0 or final_contracts <= amount_market_max
            ),
            "pre_submit_valid": bool(
                final_contracts > 0
                and (amount_min <= 0 or final_contracts >= amount_min)
                and (amount_market_max <= 0 or final_contracts <= amount_market_max)
            ),
        }

    def _minimum_order_notional(self, market: dict[str, Any], price: float) -> float:
        min_contracts = self._amount_min(market)
        contract_size = self._contract_size(market)
        if min_contracts <= 0 or contract_size <= 0 or price <= 0:
            return 0.0
        return min_contracts * contract_size * price

    def _entry_exchange_rejection_result(
        self,
        *,
        decision: DecisionOutput,
        side: str,
        price: float,
        exchange_error: Exception | str,
        okx_symbol: str,
        contract_size: float,
        order_quantity: float,
        base_quantity: float,
        okx_order_rules: dict[str, Any],
        request_params: dict[str, Any],
        leverage_check: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Return a structured rejected result when OKX still rejects an entry."""

        error_text = safe_error_text(exchange_error, limit=500)
        error_code = str(getattr(exchange_error, "code", "") or "") or None
        error_payload = getattr(exchange_error, "payload", None)

        return ExecutionResult(
            order_id="okx_rejected",
            symbol=decision.symbol,
            side=side,
            order_type="market",
            quantity=0.0,
            price=price,
            status=OrderStatus.REJECTED,
            raw_response={
                "error": (
                    "OKX 拒绝了开仓订单；系统已记录提交前规则快照、计划张数、"
                    "最终张数和交易所返回原因，供执行详情定位。"
                ),
                "raw_error": error_text,
                "okx_error_code": error_code,
                "okx_error_payload": (
                    error_payload if isinstance(error_payload, dict) else None
                ),
                "execution_blocker": "okx_exchange_rejection",
                "system_pre_submit_rejection": False,
                "okx_rejection": True,
                "okx_symbol": okx_symbol,
                "contract_size": contract_size,
                "planned_order_contracts": order_quantity,
                "planned_base_quantity": base_quantity,
                "okx_order_rules": okx_order_rules,
                "request_params": request_params,
                "leverage_check": dict(leverage_check or {}),
            },
        )

    def _entry_market_order_size_adjustment(
        self,
        *,
        decision: DecisionOutput,
        side: str,
        price: float,
        okx_symbol: str,
        contract_size: float,
        order_quantity: float,
        base_quantity: float,
        okx_order_rules: dict[str, Any],
    ) -> dict[str, Any] | ExecutionResult | None:
        max_market_contracts = self._safe_float(
            okx_order_rules.get("amount_max_market_contracts"), 0.0
        ) or self._amount_market_max(okx_order_rules)
        if max_market_contracts <= 0 or order_quantity <= max_market_contracts:
            return None
        adjusted_contracts = min(order_quantity, max_market_contracts)
        adjusted_base_quantity = adjusted_contracts * contract_size
        adjusted_notional = adjusted_base_quantity * max(price, 0.0)
        amount_min = self._safe_float(okx_order_rules.get("amount_min_contracts"), 0.0)
        min_notional = self._safe_float(okx_order_rules.get("min_notional_usdt"), 0.0)
        if adjusted_contracts <= 0 or (amount_min > 0 and adjusted_contracts + 1e-12 < amount_min):
            return self._entry_market_order_size_rejection_result(
                decision=decision,
                side=side,
                price=price,
                okx_symbol=okx_symbol,
                contract_size=contract_size,
                order_quantity=order_quantity,
                base_quantity=base_quantity,
                okx_order_rules=okx_order_rules,
                blocker="system_pre_submit_market_order_max_too_small_after_cap",
                error=(
                    "计划市价单张数超过 OKX 单笔市价单上限，但按上限缩量后低于 OKX 最小下单张数，"
                    "系统未提交无效订单。"
                ),
            )
        if min_notional > 0 and adjusted_notional + 1e-12 < min_notional:
            return self._entry_market_order_size_rejection_result(
                decision=decision,
                side=side,
                price=price,
                okx_symbol=okx_symbol,
                contract_size=contract_size,
                order_quantity=order_quantity,
                base_quantity=base_quantity,
                okx_order_rules=okx_order_rules,
                blocker="system_pre_submit_market_order_max_notional_below_min_after_cap",
                error=(
                    "计划市价单张数超过 OKX 单笔市价单上限，但按上限缩量后的名义金额低于 OKX 最小要求，"
                    "系统未提交无效订单。"
                ),
            )
        original_notional = base_quantity * max(price, 0.0)
        reduction_ratio = (
            max(0.0, 1.0 - (adjusted_notional / original_notional))
            if original_notional > 0
            else 0.0
        )
        return {
            "applied": True,
            "reason": "okx_single_market_order_max_size",
            "original_planned_order_contracts": order_quantity,
            "original_planned_base_quantity": base_quantity,
            "original_planned_notional_usdt": round(original_notional, 8),
            "adjusted_order_contracts": adjusted_contracts,
            "adjusted_base_quantity": adjusted_base_quantity,
            "adjusted_notional_usdt": round(adjusted_notional, 8),
            "amount_max_market_contracts": max_market_contracts,
            "risk_notional_reduction_ratio": round(reduction_ratio, 8),
            "okx_symbol": okx_symbol,
            "contract_size": contract_size,
        }

    def _entry_market_order_size_rejection_result(
        self,
        *,
        decision: DecisionOutput,
        side: str,
        price: float,
        okx_symbol: str,
        contract_size: float,
        order_quantity: float,
        base_quantity: float,
        okx_order_rules: dict[str, Any],
        blocker: str = "system_pre_submit_market_order_max",
        error: str | None = None,
    ) -> ExecutionResult:
        max_market_contracts = self._safe_float(
            okx_order_rules.get("amount_max_market_contracts"), 0.0
        ) or self._amount_market_max(okx_order_rules)
        return ExecutionResult(
            order_id="rejected",
            symbol=decision.symbol,
            side=side,
            order_type="market",
            quantity=0.0,
            price=price,
            status=OrderStatus.REJECTED,
            raw_response={
                "error": error
                or (
                    "计划市场单张数超过 OKX 单笔市价单上限，系统已在提交前拦截，"
                    "未向 OKX 发送必然被拒绝的订单。"
                ),
                "execution_blocker": blocker,
                "system_pre_submit_rejection": True,
                "okx_rejection": False,
                "okx_symbol": okx_symbol,
                "contract_size": contract_size,
                "planned_order_contracts": order_quantity,
                "planned_base_quantity": base_quantity,
                "amount_max_market_contracts": max_market_contracts,
                "okx_order_rules": okx_order_rules,
            },
        )

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

        capped = max_contracts
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

    def _amount_market_max(self, market: dict[str, Any]) -> float:
        info_max = self._market_info_float(
            market,
            "maxMktSz",
            "maxMarketSz",
            "maxMarketSize",
            "max_market_size",
        )
        if info_max > 0:
            return info_max
        return self._safe_float(((market.get("limits") or {}).get("amount") or {}).get("max"), 0.0)

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

    def _exit_market_order_slices(
        self,
        ccxt,
        market: dict[str, Any],
        contracts: float,
    ) -> list[float]:
        max_contracts = self._amount_market_max(market)
        if contracts <= 0 or max_contracts <= 0 or contracts <= max_contracts:
            return []

        chunks: list[float] = []
        remaining = contracts
        minimum = self._amount_min(market)
        while remaining > 1e-12:
            chunk = min(remaining, max_contracts)
            chunk = self._clamp_exit_chunk_contracts(ccxt, market, chunk)
            if chunk <= 0:
                break
            if minimum > 0 and chunk < minimum:
                break
            chunks.append(chunk)
            remaining = max(remaining - chunk, 0.0)
            if len(chunks) > 100:
                logger.warning(
                    "split exit order exceeded chunk safety limit",
                    symbol=market.get("symbol"),
                    requested_contracts=contracts,
                    max_contracts=max_contracts,
                )
                break
        return chunks if len(chunks) > 1 else []

    def _clamp_exit_chunk_contracts(self, ccxt, market: dict[str, Any], contracts: float) -> float:
        if contracts <= 0:
            return 0.0
        max_contracts = self._amount_market_max(market)
        amount = min(contracts, max_contracts) if max_contracts > 0 else contracts
        try:
            amount = float(ccxt.amount_to_precision(market["symbol"], amount))
        except Exception as exc:
            logger.debug(
                "OKX amount precision failed for split exit chunk",
                symbol=market.get("symbol"),
                contracts=amount,
                error=safe_error_text(exc),
            )
        if max_contracts > 0 and amount > max_contracts:
            amount = max_contracts
        return amount if amount > 0 else 0.0

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
            await self._cancel_order_native(ccxt, order_id, symbol)
            return True
        except Exception as e:
            logger.error("cancel order failed", order_id=order_id, error=safe_error_text(e))
            return False

    async def _cancel_order_native(
        self,
        ccxt: Any,
        order_id: Any,
        symbol: str,
    ) -> dict[str, Any]:
        """Cancel an OKX order by native instId/ordId, not CCXT symbol aliases."""

        cancel = getattr(ccxt, "privatePostTradeCancelOrder", None)
        if not callable(cancel):
            raise ExchangeAPIError("OKX native cancel-order API is unavailable")
        inst_id = okx_inst_id_from_symbol(symbol)
        ord_id = str(order_id or "").strip()
        if not inst_id or not ord_id:
            raise ExchangeAPIError(
                f"Cannot cancel OKX native order: instId={inst_id!r}, ordId={ord_id!r}"
            )
        params = {"instId": inst_id, "ordId": ord_id}
        response = await self._with_retry(cancel, params)
        self._raise_if_native_cancel_failed(response, order_id=ord_id, symbol=inst_id)
        return response if isinstance(response, dict) else {"response": response}

    def _raise_if_native_cancel_failed(
        self,
        response: Any,
        *,
        order_id: str,
        symbol: str,
    ) -> None:
        payload = response if isinstance(response, dict) else {}
        rows = payload.get("data") if isinstance(payload.get("data"), list) else []
        first = rows[0] if rows and isinstance(rows[0], dict) else {}
        code = str(payload.get("code") or "")
        s_code = str(first.get("sCode") or "")
        if code in {"", "0"} and s_code in {"", "0"}:
            return
        message = (
            first.get("sMsg")
            or payload.get("msg")
            or f"OKX native cancel failed for {symbol} {order_id}"
        )
        raise ExchangeAPIError(str(message))

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
            orders = await self.get_open_orders_strict(okx_symbol)
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
                await self._cancel_order_native(ccxt, order_id, okx_symbol)
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

    async def _fetch_applicable_okx_leverage_tier(
        self,
        okx_symbol: str,
        decision: DecisionOutput,
    ) -> dict[str, Any]:
        """Refresh and select the OKX tier for the executable position."""

        tiers = await self._fetch_okx_leverage_tiers(okx_symbol)
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        sizing = raw.get("profit_risk_sizing") if isinstance(raw, dict) else {}
        sizing = sizing if isinstance(sizing, dict) else {}
        original = sizing.get("leverage_tier_selection")
        original = original if isinstance(original, dict) else {}
        contract_spec = original.get("contract_spec")
        contract_spec = contract_spec if isinstance(contract_spec, dict) else {}
        snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        selection = select_okx_leverage_tier(
            tiers,
            target_notional_usdt=self._safe_float(
                sizing.get("final_notional_usdt"),
                0.0,
            ),
            mark_price=self._safe_float(
                original.get("mark_price") or snapshot.get("current_price") or snapshot.get("close"),
                0.0,
            ),
            contract_spec=contract_spec,
            current_position_notional_usdt=self._safe_float(
                original.get("current_position_notional_usdt"),
                0.0,
            ),
            current_position_contracts=self._safe_float(
                original.get("current_position_contracts"),
                0.0,
            ),
        )
        if selection.get("production_eligible") is not True:
            return selection
        original_max = self._safe_float(original.get("max_leverage"), 0.0)
        if original and (original.get("production_eligible") is not True or original_max < 1):
            selection = dict(selection)
            selection.update(
                {
                    "production_eligible": False,
                    "reason": "authoritative_sizing_leverage_tier_missing",
                    "max_leverage": 0.0,
                }
            )
            return selection
        if original_max >= 1:
            selection = dict(selection)
            selection["max_leverage"] = min(
                self._safe_float(selection.get("max_leverage"), 0.0),
                original_max,
            )
            selection["sizing_tier_max_leverage"] = original_max
        return selection

    async def _fetch_okx_leverage_tiers(self, okx_symbol: str) -> list[dict[str, Any]]:
        """Return unmodified OKX tier bounds plus normalized max leverage."""

        ccxt = await self._get_ccxt()
        try:
            tiers = await self._with_retry(ccxt.fetch_market_leverage_tiers, okx_symbol)
            if not isinstance(tiers, list):
                return []
            return [
                {
                    **dict(tier),
                    "maxLeverage": self._safe_float(
                        tier.get("maxLeverage") or tier.get("max_leverage"),
                        0.0,
                    ),
                }
                for tier in tiers
                if isinstance(tier, dict)
            ]
        except Exception as e:
            logger.debug(
                "fetch leverage tiers failed",
                symbol=okx_symbol,
                error=safe_error_text(e),
            )
            return []

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
        okx_symbol = await self._resolve_swap_symbol(decision.symbol)
        params = {"mgnMode": "cross"}
        requested_leverage = max(1, int(round(decision.suggested_leverage)))
        tier_selection = await self._fetch_applicable_okx_leverage_tier(okx_symbol, decision)
        max_leverage = self._safe_float(tier_selection.get("max_leverage"), 0.0)
        if tier_selection.get("production_eligible") is not True or max_leverage < 1.0:
            return {
                "ok": False,
                "error": (
                    "OKX did not return an authoritative leverage tier for the executable "
                    "position; the entry was rejected before submission."
                ),
                "ai_requested_leverage": requested_leverage,
                "okx_max_leverage": None,
                "okx_leverage_tier_selection": tier_selection,
                "target_leverage": None,
                "actual_leverage": None,
                "set_response": None,
                "verify_response": None,
                "params": params,
            }
        leverage = int(
            max(
                1,
                min(
                    requested_leverage,
                    int(max_leverage or requested_leverage),
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
                "okx_leverage_tier_selection": tier_selection,
                "target_leverage": leverage,
                "actual_leverage": actual,
                "set_response": None,
                "verify_response": verify_response,
                "params": params,
            }

        # Cross-margin leverage belongs to the existing OKX position lifecycle.
        # An add-on entry must use that accepted leverage rather than trying to
        # mutate the instrument while the position is open.
        existing_position: dict[str, Any] | None = None
        target_side = "long" if decision.action == Action.LONG else "short"
        try:
            positions = await self.get_positions_strict(decision.symbol)
            for position in positions or []:
                if self._position_matches_exit_side(
                    position,
                    target_side,
                    decision_symbol=decision.symbol,
                ):
                    existing_position = position
                    break
        except Exception as e:
            logger.debug(
                "fetch existing position before leverage mutation failed",
                symbol=okx_symbol,
                side=target_side,
                error=safe_error_text(e),
            )
        if existing_position is not None:
            info = existing_position.get("info") or {}
            position_leverage = self._safe_float(
                existing_position.get("leverage")
                or info.get("lever")
                or info.get("leverage"),
                actual,
            )
            if position_leverage <= 0:
                return {
                    "ok": False,
                    "error": (
                        "OKX 已有同方向持仓，但当前持仓快照没有返回可确认的实际杠杆；"
                        "系统不会在未知杠杆下修改现有仓位或继续加仓。"
                    ),
                    "ai_requested_leverage": requested_leverage,
                    "okx_max_leverage": max_leverage,
                    "okx_leverage_tier_selection": tier_selection,
                    "target_leverage": leverage,
                    "actual_leverage": None,
                    "existing_position": True,
                    "existing_position_side": target_side,
                    "set_response": None,
                    "verify_response": verify_response,
                    "params": params,
                }
            if position_leverage > max_leverage + 1e-8:
                return {
                    "ok": False,
                    "error": (
                        f"OKX existing position leverage {position_leverage:g}x exceeds the "
                        f"current projected-position tier maximum {max_leverage:g}x; "
                        "the add-on entry was rejected."
                    ),
                    "ai_requested_leverage": requested_leverage,
                    "okx_max_leverage": max_leverage,
                    "okx_leverage_tier_selection": tier_selection,
                    "target_leverage": leverage,
                    "actual_leverage": position_leverage,
                    "existing_position": True,
                    "existing_position_side": target_side,
                    "set_response": None,
                    "verify_response": verify_response,
                    "params": params,
                }
            decision.suggested_leverage = float(position_leverage)
            self._cache_leverage(okx_symbol, params, position_leverage)
            return {
                "ok": True,
                "skipped_set": True,
                "reason": (
                    f"OKX 已有同方向持仓，系统沿用该仓位已接受的 "
                    f"{position_leverage:g}x 杠杆，不在加仓前修改杠杆。"
                ),
                "ai_requested_leverage": requested_leverage,
                "okx_max_leverage": max_leverage,
                "okx_leverage_tier_selection": tier_selection,
                "target_leverage": position_leverage,
                "actual_leverage": position_leverage,
                "existing_position": True,
                "existing_position_side": target_side,
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
                            "okx_leverage_tier_selection": tier_selection,
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
                            "okx_leverage_tier_selection": tier_selection,
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
                            "okx_leverage_tier_selection": tier_selection,
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
                        "okx_leverage_tier_selection": tier_selection,
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
                "okx_leverage_tier_selection": tier_selection,
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
                "okx_leverage_tier_selection": tier_selection,
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
                "okx_leverage_tier_selection": tier_selection,
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
            "okx_leverage_tier_selection": tier_selection,
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

    async def _resolve_swap_symbol(self, symbol: str) -> str:
        """Return the CCXT symbol for an app symbol, honoring OKX native instIds."""

        candidate = self._to_swap_symbol(symbol)
        requested_symbol = normalize_trading_symbol(symbol)
        try:
            await self._ensure_markets_loaded()
            ccxt = await self._get_ccxt()
            try:
                market = ccxt.market(candidate)
                native_symbol = symbol_from_okx_market(market, fallback=symbol)
                if native_symbol and requested_symbol and native_symbol != requested_symbol:
                    raise ExchangeAPIError(
                        "OKX market symbol mismatch: "
                        f"requested {requested_symbol}, exchange instrument is {native_symbol}"
                    )
                return str(market.get("symbol") or candidate)
            except Exception as exc:
                if isinstance(exc, ExchangeAPIError):
                    raise
                logger.debug(
                    "OKX candidate CCXT symbol not found; trying native instId mapping",
                    symbol=symbol,
                    candidate=candidate,
                    error=safe_error_text(exc),
                )
            native = normalize_trading_symbol(symbol).replace("/", "-")
            market_id = f"{native}-SWAP" if native else ""
            by_id = (getattr(ccxt, "markets_by_id", None) or {}).get(market_id)
            markets = by_id if isinstance(by_id, list) else [by_id] if by_id else []
            for market in markets:
                if not isinstance(market, dict) or not market.get("symbol"):
                    continue
                info = market.get("info") if isinstance(market.get("info"), dict) else {}
                native_inst_id = str(info.get("instId") or market.get("id") or "").strip()
                native_symbol = symbol_from_okx_market(market, fallback=symbol)
                if native_symbol and requested_symbol and native_symbol != requested_symbol:
                    logger.warning(
                        "OKX markets_by_id returned mismatched instrument; ignoring candidate",
                        requested_symbol=requested_symbol,
                        market_id=market_id,
                        market_symbol=market.get("symbol"),
                        native_symbol=native_symbol,
                    )
                    continue
                market_symbol = normalize_trading_symbol(market.get("symbol"))
                if (
                    native_inst_id
                    and requested_symbol
                    and market_symbol
                    and market_symbol != requested_symbol
                ):
                    logger.warning(
                        "OKX markets_by_id symbol is an alias; using native instId",
                        requested_symbol=requested_symbol,
                        market_id=market_id,
                        market_symbol=market.get("symbol"),
                        native_inst_id=native_inst_id,
                    )
                    return native_inst_id
                return str(market["symbol"])
            positions = await self.get_positions_strict(symbol)
            for position in positions or []:
                info = position.get("info") or {}
                native_position_symbol = str(
                    info.get("instId") or position.get("symbol") or ""
                ).strip()
                native_symbol = symbol_from_okx_payload(
                    {"symbol": native_position_symbol, "info": info},
                    fallback=symbol,
                )
                if (
                    native_position_symbol
                    and requested_symbol
                    and native_symbol == requested_symbol
                ):
                    logger.warning(
                        "OKX swap symbol resolved from existing position snapshot",
                        symbol=symbol,
                        okx_symbol=native_position_symbol,
                    )
                    return native_position_symbol
        except ExchangeAPIError:
            raise
        except Exception as exc:
            logger.debug(
                "OKX swap symbol resolution fell back to direct conversion",
                symbol=symbol,
                candidate=candidate,
                error=safe_error_text(exc),
            )
        return candidate

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

    @staticmethod
    def _execution_result_symbol(order: dict[str, Any], decision_symbol: str) -> str:
        exchange_symbol = symbol_from_okx_payload(order, fallback=decision_symbol)
        decision_normalized = normalize_trading_symbol(decision_symbol)
        return exchange_symbol or decision_normalized or str(decision_symbol or "")

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
        stop_pct = self._safe_float(decision.stop_loss_pct, 0.0)
        take_pct = self._safe_float(decision.take_profit_pct, 0.0)
        if stop_pct <= 0 or take_pct <= 0:
            return 0.0, 0.0
        if decision.action == Action.LONG:
            stop_loss_px = low_ref * (1 - stop_pct)
            take_profit_px = high_ref * (1 + take_pct)
        else:
            stop_loss_px = high_ref * (1 + stop_pct)
            take_profit_px = low_ref * (1 - take_pct)
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
        try:
            snapshot = await self.get_balance_snapshot(asset)
            return float(snapshot.get("free") or 0.0)
        except Exception as e:
            logger.error("fetch balance failed", error=safe_error_text(e))
            return 0.0

    async def get_balance_snapshot(self, asset: str = "USDT") -> dict[str, Any]:
        ccxt = await self._get_ccxt()
        try:
            balance_data = await self._fetch_balance_without_markets(ccxt, asset)
            if asset not in balance_data and isinstance(balance_data.get("data"), list):
                balance_data = self._balance_response_to_ccxt_shape(balance_data, asset)
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
            available = float(asset_data.get("free") or 0.0) or raw_float("availEq", 0.0)
            allocatable = equity if equity > 0 else (cash if cash > 0 else total)
            return {
                "free": available,
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

    async def _fetch_balance_without_markets(
        self,
        ccxt: Any,
        asset: str,
    ) -> dict[str, Any]:
        """Fetch account balance without loading OKX instrument metadata."""

        if hasattr(ccxt, "privateGetAccountBalance"):
            return await self._with_retry(
                ccxt.privateGetAccountBalance,
                {"ccy": asset},
            )
        if hasattr(ccxt, "fetch_balance"):
            markets_before = getattr(ccxt, "markets", None)
            if markets_before is None:
                try:
                    ccxt.markets = {}
                except Exception as exc:
                    logger.debug(
                        "failed to set temporary OKX markets cache",
                        error=safe_error_text(exc),
                    )
            try:
                return await self._with_retry(ccxt.fetch_balance)
            finally:
                if markets_before is None:
                    try:
                        ccxt.markets = markets_before
                    except Exception as exc:
                        logger.debug(
                            "failed to restore OKX markets cache",
                            error=safe_error_text(exc),
                        )
        raise ExchangeAPIError("OKX balance API is unavailable on this client")

    def _balance_response_to_ccxt_shape(
        self,
        response: dict[str, Any],
        asset: str,
    ) -> dict[str, Any]:
        """Convert native OKX account balance response without loading markets."""

        raw_detail: dict[str, Any] = {}
        data = response.get("data") if isinstance(response, dict) else None
        for item in data or []:
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

        cash = raw_float("cashBal")
        equity = raw_float("eq", cash)
        used = raw_float("frozenBal")
        available = (
            raw_float("availBal")
            or raw_float("availEq")
            or raw_float("disEq")
            or max(equity - used, 0.0)
        )
        total = equity if equity > 0 else cash
        return {
            asset: {
                "free": available,
                "used": used,
                "total": total,
            },
            "info": {"data": data or []},
        }

    async def get_positions(self, symbol: str | None = None) -> list[dict]:
        try:
            return await self.get_positions_strict(symbol)
        except Exception as e:
            logger.error(
                "fetch OKX-native positions failed",
                symbol=symbol,
                error=safe_error_text(e),
            )
            return []

    async def get_positions_strict(self, symbol: str | None = None) -> list[dict]:
        """Fetch authoritative OKX-native positions and propagate failures."""
        inst_ids = [okx_inst_id_from_symbol(symbol)] if symbol else None
        return await self._native_facts_client().fetch_positions(inst_ids=inst_ids)

    async def entry_risk_facts(
        self,
        symbol: str,
        positions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return authoritative account, instrument, and leverage facts for sizing."""

        if not self._connected:
            await self.initialize()
        okx_symbol = await self._resolve_swap_symbol(symbol)
        position_symbols = [str(position.get("symbol") or "") for position in positions or []]
        requested_symbols = [symbol, *position_symbols]
        balance_snapshot, contract_specs, leverage_tiers, instrument_availability = (
            await asyncio.gather(
            self.get_balance_snapshot(),
            self._native_facts_client().fetch_contract_specs(symbols=requested_symbols),
            self._fetch_okx_leverage_tiers(okx_symbol),
            self.entry_instrument_availability(symbol, okx_symbol=okx_symbol),
            )
        )
        tier_leverages = [
            self._safe_float(tier.get("maxLeverage"), 0.0) for tier in leverage_tiers
        ]
        tier_leverages = [value for value in tier_leverages if value > 0]
        reported_max_leverage = max(tier_leverages) if tier_leverages else 0.0
        equity = max(self._safe_float(balance_snapshot.get("equity"), 0.0), 0.0)
        available_margin = max(self._safe_float(balance_snapshot.get("free"), 0.0), 0.0)
        target_inst_id = okx_inst_id_from_symbol(symbol)
        required_inst_ids = {
            inst_id
            for item in requested_symbols
            if (inst_id := okx_inst_id_from_symbol(item))
        }
        missing_specs = sorted(required_inst_ids.difference(contract_specs))
        reasons: list[str] = []
        if equity <= 0:
            reasons.append("okx_account_equity_missing")
        if available_margin <= 0:
            reasons.append("okx_available_margin_missing")
        if not leverage_tiers or reported_max_leverage < 1:
            reasons.append("okx_leverage_tiers_missing")
        if not target_inst_id or target_inst_id not in contract_specs:
            reasons.append("target_okx_contract_spec_missing")
        if missing_specs:
            reasons.append("open_position_okx_contract_spec_missing")
        if instrument_availability.get("available") is not True:
            reasons.append(
                str(
                    instrument_availability.get("reason")
                    or "okx_private_entry_instrument_availability_unverified"
                )
            )
        generated_at = datetime.now(UTC).isoformat()
        return {
            "production_eligible": not reasons,
            "account_equity_usdt": equity,
            "available_margin_usdt": available_margin,
            "reported_max_leverage": reported_max_leverage,
            "leverage_tiers": leverage_tiers,
            "margin_mode": "cross",
            "target_inst_id": target_inst_id,
            "contract_specs": contract_specs,
            "missing_contract_specs": missing_specs,
            "entry_instrument_availability": instrument_availability,
            "balance_snapshot": balance_snapshot,
            "policy_provenance": {
                "source": "okx_native_balance_contract_specs_and_leverage_tiers",
                "observation_window": "current_pre_entry_exchange_snapshot",
                "sample_count": len(contract_specs) + int(equity > 0) + len(leverage_tiers),
                "generated_at": generated_at,
                "strategy_version": "2026-07-15.okx-entry-risk-facts.v2",
                "fallback_reason": ",".join(reasons),
            },
        }

    async def entry_instrument_availability(
        self,
        symbol: str,
        *,
        okx_symbol: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Verify that this account and execution mode can address the instrument."""

        inst_id = okx_inst_id_from_symbol(symbol)
        now = time.monotonic()
        cached = self._entry_instrument_availability_cache.get(inst_id)
        if not force and cached and now - cached[1] <= cached[2]:
            return {**cached[0], "cache_hit": True}

        generated_at = datetime.now(UTC).isoformat()
        try:
            ccxt = await self._get_ccxt()
            fetch_leverage = getattr(ccxt, "fetch_leverage", None)
            if not callable(fetch_leverage):
                raise ExchangeAPIError("OKX private account leverage API is unavailable")
            resolved_symbol = okx_symbol or await self._resolve_swap_symbol(symbol)
            response = await self._with_retry(
                fetch_leverage,
                resolved_symbol,
                {"mgnMode": "cross"},
                _expected_error_codes={"51001"},
                _max_attempts=1,
            )
            result = {
                "available": True,
                "reason": "okx_private_account_instrument_verified",
                "source": "okx_private_account_leverage_info",
                "symbol": normalize_trading_symbol(symbol),
                "inst_id": inst_id,
                "mode": self.executor_mode,
                "demo": settings.is_okx_demo(self.executor_mode),
                "reported_leverage": self._extract_verified_leverage(response),
                "generated_at": generated_at,
                "cache_hit": False,
            }
            ttl = OKX_ENTRY_INSTRUMENT_AVAILABILITY_CACHE_SECONDS
        except Exception as exc:
            error_text = safe_error_text(exc, limit=220)
            error_code = self._exchange_error_code(exc, error_text)
            unavailable = error_code == "51001" or "[51001]" in error_text
            result = {
                "available": False,
                "reason": (
                    "okx_private_entry_instrument_unavailable"
                    if unavailable
                    else "okx_private_entry_instrument_probe_failed"
                ),
                "source": "okx_private_account_leverage_info",
                "symbol": normalize_trading_symbol(symbol),
                "inst_id": inst_id,
                "mode": self.executor_mode,
                "demo": settings.is_okx_demo(self.executor_mode),
                "error_code": error_code or None,
                "error": error_text,
                "generated_at": generated_at,
                "cache_hit": False,
            }
            ttl = (
                OKX_ENTRY_INSTRUMENT_UNAVAILABLE_CACHE_SECONDS
                if unavailable
                else OKX_ENTRY_INSTRUMENT_PROBE_FAILURE_CACHE_SECONDS
            )
        self._entry_instrument_availability_cache[inst_id] = (dict(result), now, ttl)
        return result

    async def entry_instrument_availability_shortlist(
        self,
        symbols: list[str],
        *,
        target_count: int,
        concurrency: int = 4,
    ) -> dict[str, Any]:
        """Probe ranked instruments in bounded batches and stop once the target is met."""

        target = max(0, int(target_count or 0))
        ordered_symbols = list(dict.fromkeys(str(symbol) for symbol in symbols if str(symbol)))
        if target <= 0 or not ordered_symbols:
            return {
                "selected_symbols": [],
                "availability": {},
                "evaluated_count": 0,
                "probed_count": 0,
                "cache_hit_count": 0,
                "skipped_after_target_count": len(ordered_symbols),
            }

        batch_size = max(1, min(int(concurrency or 1), target, len(ordered_symbols)))
        selected: list[str] = []
        availability: dict[str, dict[str, Any]] = {}
        evaluated_count = 0
        probed_count = 0
        cache_hit_count = 0

        for offset in range(0, len(ordered_symbols), batch_size):
            if len(selected) >= target:
                break
            batch = ordered_symbols[offset : offset + batch_size]
            facts_rows = await asyncio.gather(
                *(self.entry_instrument_availability(symbol) for symbol in batch)
            )
            for symbol, facts in zip(batch, facts_rows, strict=True):
                normalized_facts = dict(facts) if isinstance(facts, dict) else {}
                availability[symbol] = normalized_facts
                evaluated_count += 1
                if normalized_facts.get("cache_hit") is True:
                    cache_hit_count += 1
                else:
                    probed_count += 1
                if normalized_facts.get("available") is True and len(selected) < target:
                    selected.append(symbol)

        for symbol in ordered_symbols[evaluated_count:]:
            availability[symbol] = {
                "available": None,
                "reason": "okx_private_entry_instrument_not_probed_after_target_filled",
                "cache_hit": False,
            }

        return {
            "selected_symbols": selected,
            "availability": availability,
            "evaluated_count": evaluated_count,
            "probed_count": probed_count,
            "cache_hit_count": cache_hit_count,
            "skipped_after_target_count": max(len(ordered_symbols) - evaluated_count, 0),
        }

    async def pre_order_execution_facts(
        self,
        symbol: str,
        side: str,
    ) -> dict[str, Any]:
        """Refresh native quote, book, mark, contract and fee facts before entry."""

        if not self._connected:
            await self.initialize()
        okx_symbol = await self._resolve_swap_symbol(symbol)
        inst_id = okx_inst_id_from_symbol(symbol)
        ccxt = await self._get_ccxt()
        ticker, book, mark_response, specs, fee = await asyncio.gather(
            self._fetch_native_ticker(symbol),
            self._with_retry(ccxt.fetch_order_book, okx_symbol),
            self._with_retry(ccxt.publicGetPublicMarkPrice, {"instId": inst_id}),
            self._native_facts_client().fetch_contract_specs(symbols=[symbol]),
            self.fetch_account_fee_snapshot(),
        )
        mark_rows = mark_response.get("data") if isinstance(mark_response, dict) else []
        mark_row = mark_rows[0] if isinstance(mark_rows, list) and mark_rows else {}
        mark_row = mark_row if isinstance(mark_row, dict) else {}
        mark_price = max(self._safe_float(mark_row.get("markPx"), 0.0), 0.0)
        spec = specs.get(inst_id) if isinstance(specs, dict) else None
        spec = spec if isinstance(spec, dict) else {}
        ct_val = max(self._safe_float(spec.get("ctVal"), 0.0), 0.0)
        ct_mult = max(self._safe_float(spec.get("ctMult"), 0.0), 0.0)
        contract_value_base = ct_val * ct_mult
        bids = [
            [self._safe_float(level[0], 0.0), self._safe_float(level[1], 0.0)]
            for level in (book.get("bids") or [])
            if isinstance(level, (list, tuple)) and len(level) >= 2
        ]
        asks = [
            [self._safe_float(level[0], 0.0), self._safe_float(level[1], 0.0)]
            for level in (book.get("asks") or [])
            if isinstance(level, (list, tuple)) and len(level) >= 2
        ]
        bid = max(self._safe_float(ticker.get("bid"), 0.0), 0.0)
        ask = max(self._safe_float(ticker.get("ask"), 0.0), 0.0)
        if bid <= 0 and bids:
            bid = bids[0][0]
        if ask <= 0 and asks:
            ask = asks[0][0]
        bid_depth = sum(price * contracts * contract_value_base for price, contracts in bids)
        ask_depth = sum(price * contracts * contract_value_base for price, contracts in asks)
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
        reasons: list[str] = []
        if not inst_id:
            reasons.append("okx_pre_order_inst_id_missing")
        if max(self._safe_float(ticker.get("last"), 0.0), 0.0) <= 0:
            reasons.append("okx_pre_order_last_price_missing")
        if bid <= 0 or ask <= 0 or bid > ask:
            reasons.append("okx_pre_order_bid_ask_invalid")
        if mark_price <= 0:
            reasons.append("okx_pre_order_mark_price_missing")
        if contract_value_base <= 0:
            reasons.append("okx_pre_order_contract_spec_missing")
        if not bids or not asks or bid_depth <= 0 or ask_depth <= 0:
            reasons.append("okx_pre_order_orderbook_incomplete")
        if not fee.get("taker_fee_rate"):
            reasons.append("okx_pre_order_account_fee_missing")
        generated_at = datetime.now(UTC).isoformat()
        feature_snapshot = {
            "current_price": max(self._safe_float(ticker.get("last"), 0.0), 0.0),
            "bid": bid,
            "ask": ask,
            "mark_price": mark_price,
            "spread_pct": (
                (ask - bid) / ((ask + bid) / 2.0) * 100.0
                if bid > 0 and ask >= bid
                else 0.0
            ),
            "orderbook_bids": bids,
            "orderbook_asks": asks,
            "orderbook_bid_depth": bid_depth,
            "orderbook_ask_depth": ask_depth,
            "orderbook_imbalance": imbalance,
            "contract_value_base": contract_value_base,
            "contract_spec": spec,
            "taker_fee_rate": fee.get("taker_fee_rate"),
            "entry_fee_rate": fee.get("entry_fee_rate"),
            "exit_fee_rate": fee.get("exit_fee_rate"),
            "fee_rate_source": fee.get("fee_rate_source"),
            "fee_rate_observed_at": fee.get("fee_rate_observed_at"),
            "fee_policy_provenance": fee.get("policy_provenance"),
        }
        return {
            "production_eligible": not reasons,
            "reason": "okx_native_pre_order_execution_facts_ready"
            if not reasons
            else ",".join(reasons),
            "symbol": normalize_trading_symbol(symbol),
            "side": str(side or "").lower(),
            "okx_symbol": okx_symbol,
            "inst_id": inst_id,
            "ticker_source_timestamp_ms": ticker.get("timestamp"),
            "orderbook_source_timestamp_ms": book.get("timestamp"),
            "mark_source_timestamp_ms": self._safe_float(mark_row.get("ts"), 0.0),
            "contract_spec": spec,
            "fee_snapshot": fee,
            "feature_snapshot": feature_snapshot,
            "policy_provenance": {
                "source": "okx_native_ticker_orderbook_mark_contract_and_account_fee",
                "observation_window": "current_immediate_pre_order_refresh",
                "sample_count": len(bids) + len(asks) + int(mark_price > 0) + int(bool(spec)),
                "generated_at": generated_at,
                "strategy_version": "2026-07-15.okx-pre-order-execution-facts.v1",
                "fallback_reason": "" if not reasons else ",".join(reasons),
            },
        }

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
        """Fetch authoritative OKX-native pending orders and propagate failures."""
        inst_ids = [okx_inst_id_from_symbol(symbol)] if symbol else None
        return await self._native_facts_client().fetch_open_orders(inst_ids=inst_ids)

    async def get_position_protection_orders(self, symbol: str | None = None) -> list[dict]:
        """Fetch active OKX-native TP/SL algo orders that protect open positions."""
        inst_ids = [okx_inst_id_from_symbol(symbol)] if symbol else None
        return await self._native_facts_client().fetch_position_protection_orders(
            inst_ids=inst_ids,
        )

    async def get_contract_specs_strict(
        self,
        symbols: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch authoritative OKX contract sizing rules and propagate failures."""

        return await self._native_facts_client().fetch_contract_specs(symbols=symbols)

    async def amend_position_protection_size(
        self,
        *,
        inst_id: str,
        algo_id: str,
        contracts: float,
    ) -> dict[str, Any]:
        """Amend only the OKX algo size while preserving its dynamic prices."""

        if contracts <= 0:
            raise ExchangeAPIError("Protection amend requires positive contracts")
        ccxt = await self._get_ccxt()
        amend = getattr(ccxt, "privatePostTradeAmendAlgos", None)
        if not callable(amend):
            raise ExchangeAPIError("OKX native amend-algo API is unavailable")
        return await self._with_retry(
            amend,
            {
                "instId": str(inst_id or "").upper(),
                "algoId": str(algo_id or ""),
                "newSz": self._format_okx_number(contracts),
                "cxlOnFail": "false",
            },
        )

    async def cancel_position_protection_order(
        self,
        *,
        inst_id: str,
        algo_id: str,
    ) -> dict[str, Any]:
        """Cancel one proven orphan or zero-allocation OKX protection order."""

        ccxt = await self._get_ccxt()
        cancel = getattr(ccxt, "privatePostTradeCancelAlgos", None)
        if not callable(cancel):
            raise ExchangeAPIError("OKX native cancel-algo API is unavailable")
        return await self._with_retry(
            cancel,
            {
                "algoIds": [
                    {
                        "instId": str(inst_id or "").upper(),
                        "algoId": str(algo_id or ""),
                    }
                ]
            },
        )

    async def fetch_account_fee_snapshot(self) -> dict[str, Any]:
        """Read the current account-level SWAP taker fee from OKX."""

        exchange = await self._get_ccxt()
        response = await exchange.privateGetAccountFeeRates({"instType": "SWAP"})
        rows = response.get("data") if isinstance(response, dict) else []
        row = rows[0] if isinstance(rows, list) and rows else {}
        row = row if isinstance(row, dict) else {}
        taker_rate = 0.0
        taker_field = ""
        for field_name in ("takerU", "taker", "takerUSDC"):
            value = abs(self._safe_float(row.get(field_name), 0.0))
            if value > 0:
                taker_rate = value
                taker_field = field_name
                break
        observed_at_ms = self._safe_float(row.get("ts"), 0.0)
        observed_at = (
            datetime.fromtimestamp(observed_at_ms / 1000.0, tz=UTC).isoformat()
            if observed_at_ms > 0
            else datetime.now(UTC).isoformat()
        )
        fallback_reason = "" if taker_rate > 0 else "okx_swap_taker_fee_missing"
        return {
            "taker_fee_rate": taker_rate if taker_rate > 0 else None,
            "entry_fee_rate": taker_rate if taker_rate > 0 else None,
            "exit_fee_rate": taker_rate if taker_rate > 0 else None,
            "fee_rate_source": (
                f"okx_account_trade_fee.{taker_field}"
                if taker_field
                else "okx_account_trade_fee.missing"
            ),
            "fee_rate_observed_at": observed_at,
            "policy_provenance": {
                "source": "okx_account_trade_fee_swap",
                "observation_window": "current_account_fee_tier",
                "sample_count": int(taker_rate > 0),
                "generated_at": observed_at,
                "strategy_version": "2026-07-13.okx-account-fee-snapshot.v1",
                "fallback_reason": fallback_reason,
            },
        }

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
            self._markets_loaded = False
        logger.info("OKX executor shut down")
