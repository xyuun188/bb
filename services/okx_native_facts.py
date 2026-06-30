"""OKX-native read-only facts.

This module keeps exchange truth in OKX's own identifiers (`instId`, `ordId`,
`tradeId`, `posSide`) instead of CCXT's normalized symbols.  It is intentionally
read-only and is shared by reconciliation, fill lookup, and historical repair
guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

from core.symbols import normalize_trading_symbol, okx_inst_id_from_symbol

DEFAULT_FILL_LIMIT = 100
DEFAULT_MAX_FILL_PAGES = 1
DEFAULT_MAX_TARGET_ORDER_QUERIES = 100
DEFAULT_MAX_INSTRUMENT_QUERIES = 20
DEFAULT_MAX_ORDER_HISTORY_CONTEXT_QUERIES = 30
DEFAULT_PROTECTION_ALGO_ORDER_TYPES = ("conditional", "oco", "trigger", "move_order_stop")


@dataclass(frozen=True, slots=True)
class OkxNativeFillGroup:
    order_id: str
    trade_ids: tuple[str, ...]
    inst_id: str
    symbol: str
    side: str
    pos_side: str
    contracts: float
    avg_price: float
    fee_abs: float
    fill_pnl: float
    timestamp_ms: float
    timestamp: datetime | None
    raw_count: int
    rows: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "trade_ids": list(self.trade_ids),
            "inst_id": self.inst_id,
            "symbol": self.symbol,
            "side": self.side,
            "pos_side": self.pos_side,
            "contracts": _round(self.contracts),
            "avg_price": _round(self.avg_price),
            "fee_abs": _round(self.fee_abs),
            "fill_pnl": _round(self.fill_pnl),
            "timestamp_ms": _round(self.timestamp_ms),
            "timestamp": _iso(self.timestamp),
            "raw_count": self.raw_count,
        }

    @property
    def latest_row(self) -> dict[str, Any]:
        if not self.rows:
            return {}
        return max(
            self.rows,
            key=lambda row: _safe_float(row.get("ts") or row.get("fillTime"), 0.0),
        )


class OkxNativeFactsClient:
    """Small read-only adapter for OKX-native private facts."""

    def __init__(
        self,
        executor: Any,
        *,
        symbol_normalizer: Any = normalize_trading_symbol,
        max_instrument_queries: int = DEFAULT_MAX_INSTRUMENT_QUERIES,
    ) -> None:
        self.executor = executor
        self.symbol_normalizer = symbol_normalizer
        self.max_instrument_queries = max(1, int(max_instrument_queries or 1))

    async def fetch_fill_groups(
        self,
        *,
        symbols: Iterable[Any] | None = None,
        inst_ids: Iterable[Any] | None = None,
        order_ids: Iterable[Any] | None = None,
        since: datetime | int | float | None = None,
        side: str | None = None,
        limit: int = DEFAULT_FILL_LIMIT,
        max_pages: int = DEFAULT_MAX_FILL_PAGES,
        account_wide_fallback: bool = True,
        account_wide_only: bool = False,
        strict: bool = False,
    ) -> list[OkxNativeFillGroup]:
        ccxt = await self.executor._get_ccxt()
        fetch_fills = getattr(ccxt, "privateGetTradeFillsHistory", None)
        if not callable(fetch_fills):
            if strict:
                raise RuntimeError("OKX native fills-history API is unavailable")
            return []

        target_inst_ids = _target_inst_ids(symbols=symbols, inst_ids=inst_ids)
        target_order_ids = _target_order_ids(order_ids)
        page_limit = _limit(limit)
        page_count = _max_pages(max_pages)
        queried_account_wide = bool(account_wide_only or not target_inst_ids)
        if queried_account_wide:
            params_list = [{"instType": "SWAP", "limit": str(page_limit)}]
        else:
            params_list = [
                {"instType": "SWAP", "instId": inst_id, "limit": str(page_limit)}
                for inst_id in sorted(target_inst_ids)
            ][: self.max_instrument_queries]

        since_ms = _timestamp_ms(since)
        rows: list[dict[str, Any]] = []
        seen: set[tuple[Any, Any, Any, Any]] = set()
        fallback_needed = False
        failed_inst_ids: set[str] = set()
        last_error: Exception | None = None
        successful_read = False

        for params in params_list:
            inst_id = str(params.get("instId") or "").strip().upper()
            try:
                page_rows = await self._fetch_fill_history_pages(
                    fetch_fills,
                    params,
                    since_ms=since_ms,
                    side=side,
                    target_inst_ids=target_inst_ids,
                    page_limit=page_limit,
                    max_pages=page_count,
                )
                successful_read = True
            except Exception as exc:
                last_error = exc
                if inst_id:
                    failed_inst_ids.add(inst_id)
                continue
            for row in page_rows:
                key = _fill_row_identity(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

        if (
            account_wide_fallback
            and not queried_account_wide
            and (fallback_needed or failed_inst_ids)
        ):
            try:
                page_rows = await self._fetch_fill_history_pages(
                    fetch_fills,
                    {"instType": "SWAP", "limit": str(page_limit)},
                    since_ms=since_ms,
                    side=side,
                    target_inst_ids=target_inst_ids,
                    page_limit=page_limit,
                    max_pages=page_count,
                )
                successful_read = True
            except Exception as exc:
                if strict:
                    raise last_error or exc
                page_rows = []
            for row in page_rows:
                key = _fill_row_identity(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        elif strict and last_error is not None and not successful_read:
            raise last_error

        if target_order_ids:
            known_order_ids = {
                str(row.get("ordId") or row.get("order") or "").strip()
                for row in rows
                if str(row.get("ordId") or row.get("order") or "").strip()
            }
            for order_id in sorted(target_order_ids - known_order_ids)[
                :DEFAULT_MAX_TARGET_ORDER_QUERIES
            ]:
                try:
                    page_rows = await self._fetch_fill_history_pages(
                        fetch_fills,
                        {"instType": "SWAP", "ordId": order_id, "limit": str(page_limit)},
                        since_ms=since_ms,
                        side=side,
                        target_inst_ids=set(),
                        page_limit=page_limit,
                        max_pages=page_count,
                    )
                    successful_read = True
                except Exception as exc:
                    last_error = exc
                    if strict and not successful_read:
                        raise
                    continue
                for row in page_rows:
                    key = _fill_row_identity(row)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)

        return group_okx_native_fill_rows(
            rows,
            symbol_normalizer=self.symbol_normalizer,
        )

    async def _fetch_fill_history_pages(
        self,
        fetch_fills: Any,
        params: dict[str, Any],
        *,
        since_ms: float,
        side: str | None,
        target_inst_ids: set[str],
        page_limit: int,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        after_cursor = ""
        seen_cursors: set[str] = set()
        for _page in range(max(1, int(max_pages or 1))):
            page_params = dict(params)
            page_params["limit"] = str(page_limit)
            if since_ms > 0 and "begin" not in page_params:
                page_params["begin"] = str(int(since_ms))
            if after_cursor:
                page_params["after"] = after_cursor
            response = await self.executor._with_retry(fetch_fills, page_params)
            page_rows = _response_rows(response)
            for row in page_rows:
                if _row_matches(
                    row,
                    since_ms=since_ms,
                    side=side,
                    target_inst_ids=target_inst_ids,
                ):
                    rows.append(row)
            if len(page_rows) < page_limit:
                break
            oldest_ts = _oldest_timestamp_ms(page_rows)
            if since_ms > 0 and oldest_ts > 0 and oldest_ts < since_ms:
                break
            cursor = _oldest_fill_pagination_cursor(page_rows)
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
            after_cursor = cursor
        return rows

    async def fetch_positions(
        self,
        *,
        symbols: Iterable[Any] | None = None,
        inst_ids: Iterable[Any] | None = None,
    ) -> list[dict[str, Any]]:
        ccxt = await self.executor._get_ccxt()
        fetch_positions = getattr(ccxt, "privateGetAccountPositions", None)
        if not callable(fetch_positions):
            raise RuntimeError("OKX native positions API is unavailable")

        target_inst_ids = _target_inst_ids(symbols=symbols, inst_ids=inst_ids)
        params: dict[str, str] = {"instType": "SWAP"}
        if len(target_inst_ids) == 1:
            params["instId"] = next(iter(target_inst_ids))

        response = await self.executor._with_retry(fetch_positions, params)
        rows = [
            row
            for row in _response_rows(response)
            if _native_position_row_is_open(row)
            and (
                not target_inst_ids
                or str(row.get("instId") or "").strip().upper() in target_inst_ids
            )
        ]
        return [
            _native_position_to_ccxt_shape(row, symbol_normalizer=self.symbol_normalizer)
            for row in rows
        ]

    async def fetch_position_history_rows(
        self,
        *,
        inst_ids: Iterable[Any] | None = None,
        pos_ids: Iterable[Any] | None = None,
        since: datetime | int | float | None = None,
        limit: int = DEFAULT_FILL_LIMIT,
        max_pages: int = DEFAULT_MAX_FILL_PAGES,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch OKX-native historical position lifecycle rows.

        OKX positions-history is the preferred lifecycle source for Phase 3
        closed-position display.  Linked orders/fills still provide the
        execution details popup and training evidence.
        """

        ccxt = await self.executor._get_ccxt()
        fetch_history = getattr(ccxt, "privateGetAccountPositionsHistory", None)
        if not callable(fetch_history):
            if strict:
                raise RuntimeError("OKX native positions-history API is unavailable")
            return []

        target_inst_ids = _target_inst_ids(symbols=None, inst_ids=inst_ids)
        target_pos_ids = _target_pos_ids(pos_ids)
        page_limit = _limit(limit)
        page_count = _max_pages(max_pages)
        since_ms = _timestamp_ms(since)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        params_list: list[dict[str, Any]] = []
        if target_pos_ids:
            for chunk in _chunked(sorted(target_pos_ids), 20):
                params = {
                    "instType": "SWAP",
                    "posId": ",".join(chunk),
                    "limit": str(page_limit),
                }
                if len(target_inst_ids) == 1:
                    params["instId"] = next(iter(target_inst_ids))
                params_list.append(params)
        elif target_inst_ids:
            params_list = [
                {"instType": "SWAP", "instId": inst_id, "limit": str(page_limit)}
                for inst_id in sorted(target_inst_ids)
            ][: self.max_instrument_queries]
        else:
            params_list = [{"instType": "SWAP", "limit": str(page_limit)}]

        for params in params_list:
            try:
                page_rows = await self._fetch_position_history_pages(
                    fetch_history,
                    params,
                    since_ms=since_ms,
                    target_inst_ids=target_inst_ids,
                    target_pos_ids=target_pos_ids,
                    page_limit=page_limit,
                    max_pages=page_count,
                )
            except Exception:
                if strict:
                    raise
                continue
            for row in page_rows:
                key = _position_history_row_identity(row)
                if not key or key in seen:
                    continue
                seen.add(key)
                rows.append(row)

        return sorted(
            rows,
            key=lambda row: _safe_float(row.get("uTime") or row.get("cTime"), 0.0),
            reverse=True,
        )

    async def _fetch_position_history_pages(
        self,
        fetch_history: Any,
        params: dict[str, Any],
        *,
        since_ms: float,
        target_inst_ids: set[str],
        target_pos_ids: set[str],
        page_limit: int,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        after_cursor = ""
        seen_cursors: set[str] = set()
        for _page in range(max(1, int(max_pages or 1))):
            page_params = dict(params)
            page_params["limit"] = str(page_limit)
            if since_ms > 0 and "begin" not in page_params:
                page_params["begin"] = str(int(since_ms))
            if after_cursor:
                page_params["after"] = after_cursor
            response = await self.executor._with_retry(fetch_history, page_params)
            page_rows = _response_rows(response)
            for row in page_rows:
                if _position_history_row_matches(
                    row,
                    since_ms=since_ms,
                    target_inst_ids=target_inst_ids,
                    target_pos_ids=target_pos_ids,
                ):
                    rows.append(row)
            if len(page_rows) < page_limit:
                break
            oldest_ts = _oldest_position_history_timestamp_ms(page_rows)
            if since_ms > 0 and oldest_ts > 0 and oldest_ts < since_ms:
                break
            cursor = _oldest_position_history_pagination_cursor(page_rows)
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
            after_cursor = cursor
        return rows

    async def fetch_contract_sizes(
        self,
        *,
        symbols: Iterable[Any] | None = None,
        inst_ids: Iterable[Any] | None = None,
    ) -> dict[str, float]:
        ccxt = await self.executor._get_ccxt()
        fetch_instruments = getattr(ccxt, "publicGetPublicInstruments", None)
        if not callable(fetch_instruments):
            raise RuntimeError("OKX public instruments API is unavailable")

        target_inst_ids = _target_inst_ids(symbols=symbols, inst_ids=inst_ids)
        response = await self.executor._with_retry(fetch_instruments, {"instType": "SWAP"})
        sizes: dict[str, float] = {}
        for row in _response_rows(response):
            inst_id = str(row.get("instId") or "").strip().upper()
            if not inst_id or not inst_id.endswith("-USDT-SWAP"):
                continue
            if target_inst_ids and inst_id not in target_inst_ids:
                continue
            contract_size = _safe_float(row.get("ctVal"), 0.0)
            if contract_size > 0:
                sizes[inst_id] = contract_size
        return sizes

    async def fetch_order_history_contexts(
        self,
        *,
        fills: Iterable[OkxNativeFillGroup] | None = None,
        order_ids: Iterable[Any] | None = None,
        inst_ids_by_order_id: dict[str, str] | None = None,
        limit: int = 5,
        max_queries: int = DEFAULT_MAX_ORDER_HISTORY_CONTEXT_QUERIES,
        strict: bool = False,
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        """Fetch OKX-native order history rows around specific order ids.

        OKX may return both the triggered reduce-only close order and its
        source entry order when querying by ``ordId``.  Authoritative sync uses
        that context to recognize attached TP/SL/OCO orders generated by this
        system, instead of treating them as external manual trades.
        """

        ccxt = await self.executor._get_ccxt()
        fetch_orders = getattr(ccxt, "privateGetTradeOrdersHistory", None)
        if not callable(fetch_orders):
            if strict:
                raise RuntimeError("OKX native orders-history API is unavailable")
            return {}

        inst_lookup = dict(inst_ids_by_order_id or {})
        targets: dict[str, str] = {}
        for value in order_ids or ():
            order_id = str(value or "").strip()
            if not order_id:
                continue
            targets[order_id] = str(inst_lookup.get(order_id) or "").strip().upper()
        for fill in fills or ():
            order_id = str(getattr(fill, "order_id", "") or "").strip()
            if not order_id:
                continue
            fill_inst_id = str(getattr(fill, "inst_id", "") or "").strip().upper()
            if order_id not in targets:
                targets[order_id] = fill_inst_id
            elif not targets[order_id] and fill_inst_id:
                targets[order_id] = fill_inst_id

        contexts: dict[str, tuple[dict[str, Any], ...]] = {}
        query_limit = max(1, min(int(max_queries or 1), DEFAULT_MAX_ORDER_HISTORY_CONTEXT_QUERIES))
        for order_id, inst_id in list(targets.items())[:query_limit]:
            params = {"instType": "SWAP", "ordId": order_id, "limit": str(_limit(limit))}
            if inst_id:
                params["instId"] = inst_id
            try:
                response = await self.executor._with_retry(fetch_orders, params)
            except Exception:
                if strict:
                    raise
                continue
            contexts[order_id] = tuple(_response_rows(response))
        return contexts

    async def fetch_order_history_rows(
        self,
        *,
        inst_ids: Iterable[Any] | None = None,
        order_ids: Iterable[Any] | None = None,
        since: datetime | int | float | None = None,
        limit: int = DEFAULT_FILL_LIMIT,
        max_pages: int = DEFAULT_MAX_FILL_PAGES,
        strict: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch OKX-native order history rows for Phase 3 ledger sync.

        OKX order history is the authority for order existence/state, while
        fills-history is the authority for execution price, fee, and realized PnL.
        """

        ccxt = await self.executor._get_ccxt()
        fetch_order_methods = [
            method
            for method in (
                getattr(ccxt, "privateGetTradeOrdersHistory", None),
                getattr(ccxt, "privateGetTradeOrdersHistoryArchive", None),
            )
            if callable(method)
        ]
        if not fetch_order_methods:
            if strict:
                raise RuntimeError("OKX native orders-history API is unavailable")
            return []

        target_inst_ids = _target_inst_ids(symbols=None, inst_ids=inst_ids)
        target_order_ids = _target_order_ids(order_ids)
        page_limit = _limit(limit)
        page_count = _max_pages(max_pages)
        since_ms = _timestamp_ms(since)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        if target_order_ids:
            for order_id in sorted(target_order_ids)[:DEFAULT_MAX_TARGET_ORDER_QUERIES]:
                params = {"instType": "SWAP", "ordId": order_id, "limit": str(page_limit)}
                inst_id = _single_inst_id_for_order(order_id, target_inst_ids)
                if inst_id:
                    params["instId"] = inst_id
                for fetch_orders in fetch_order_methods:
                    try:
                        page_rows = await self._fetch_order_history_pages(
                            fetch_orders,
                            params,
                            since_ms=since_ms,
                            target_inst_ids=target_inst_ids,
                            target_order_ids=target_order_ids,
                            page_limit=page_limit,
                            max_pages=page_count,
                        )
                    except Exception:
                        if strict and fetch_orders is fetch_order_methods[-1]:
                            raise
                        continue
                    for row in page_rows:
                        key = _order_row_identity(row)
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        rows.append(row)

        if not target_order_ids:
            params_list = [
                {"instType": "SWAP", "instId": inst_id, "limit": str(page_limit)}
                for inst_id in sorted(target_inst_ids)
            ][: self.max_instrument_queries]
            if not params_list:
                params_list = [{"instType": "SWAP", "limit": str(page_limit)}]
            for params in params_list:
                for fetch_orders in fetch_order_methods:
                    try:
                        page_rows = await self._fetch_order_history_pages(
                            fetch_orders,
                            params,
                            since_ms=since_ms,
                            target_inst_ids=target_inst_ids,
                            target_order_ids=target_order_ids,
                            page_limit=page_limit,
                            max_pages=page_count,
                        )
                    except Exception:
                        if strict and fetch_orders is fetch_order_methods[-1]:
                            raise
                        continue
                    for row in page_rows:
                        key = _order_row_identity(row)
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        rows.append(row)

        return sorted(
            rows,
            key=lambda row: _safe_float(row.get("uTime") or row.get("cTime"), 0.0),
            reverse=True,
        )

    async def _fetch_order_history_pages(
        self,
        fetch_orders: Any,
        params: dict[str, Any],
        *,
        since_ms: float,
        target_inst_ids: set[str],
        target_order_ids: set[str],
        page_limit: int,
        max_pages: int,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        after_cursor = ""
        seen_cursors: set[str] = set()
        for _page in range(max(1, int(max_pages or 1))):
            page_params = dict(params)
            page_params["limit"] = str(page_limit)
            if since_ms > 0 and "begin" not in page_params:
                page_params["begin"] = str(int(since_ms))
            if after_cursor:
                page_params["after"] = after_cursor
            response = await self.executor._with_retry(fetch_orders, page_params)
            page_rows = _response_rows(response)
            for row in page_rows:
                if _order_row_matches(
                    row,
                    since_ms=since_ms,
                    target_inst_ids=target_inst_ids,
                    target_order_ids=target_order_ids,
                ):
                    rows.append(row)
            if len(page_rows) < page_limit:
                break
            oldest_ts = _oldest_order_timestamp_ms(page_rows)
            if since_ms > 0 and oldest_ts > 0 and oldest_ts < since_ms:
                break
            cursor = _oldest_order_pagination_cursor(page_rows)
            if not cursor or cursor in seen_cursors:
                break
            seen_cursors.add(cursor)
            after_cursor = cursor
        return rows

    async def fetch_open_orders(
        self,
        *,
        symbols: Iterable[Any] | None = None,
        inst_ids: Iterable[Any] | None = None,
        limit: int = DEFAULT_FILL_LIMIT,
    ) -> list[dict[str, Any]]:
        ccxt = await self.executor._get_ccxt()
        fetch_orders = getattr(ccxt, "privateGetTradeOrdersPending", None)
        if not callable(fetch_orders):
            raise RuntimeError("OKX native pending-orders API is unavailable")

        target_inst_ids = _target_inst_ids(symbols=symbols, inst_ids=inst_ids)
        params_list = [
            {"instType": "SWAP", "instId": inst_id, "limit": str(_limit(limit))}
            for inst_id in sorted(target_inst_ids)
        ][: self.max_instrument_queries]
        if not params_list:
            params_list = [{"instType": "SWAP", "limit": str(_limit(limit))}]

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for params in params_list:
            response = await self.executor._with_retry(fetch_orders, params)
            for row in _response_rows(response):
                row_inst_id = str(row.get("instId") or "").strip().upper()
                if target_inst_ids and row_inst_id and row_inst_id not in target_inst_ids:
                    continue
                order_id = str(row.get("ordId") or row.get("clOrdId") or "").strip()
                if not order_id or order_id in seen:
                    continue
                seen.add(order_id)
                rows.append(row)
        return [
            _native_order_to_ccxt_shape(row, symbol_normalizer=self.symbol_normalizer)
            for row in rows
        ]

    async def fetch_position_protection_orders(
        self,
        *,
        symbols: Iterable[Any] | None = None,
        inst_ids: Iterable[Any] | None = None,
        ord_types: Iterable[str] = DEFAULT_PROTECTION_ALGO_ORDER_TYPES,
        limit: int = DEFAULT_FILL_LIMIT,
    ) -> list[dict[str, Any]]:
        ccxt = await self.executor._get_ccxt()
        fetch_algos = getattr(ccxt, "privateGetTradeOrdersAlgoPending", None)
        if not callable(fetch_algos):
            raise RuntimeError("OKX native algo pending-orders API is unavailable")

        target_inst_ids = _target_inst_ids(symbols=symbols, inst_ids=inst_ids)
        params_list: list[dict[str, str]] = []
        for ord_type in ord_types or DEFAULT_PROTECTION_ALGO_ORDER_TYPES:
            ord_type_text = str(ord_type or "").strip()
            if not ord_type_text:
                continue
            if target_inst_ids:
                params_list.extend(
                    {
                        "instType": "SWAP",
                        "instId": inst_id,
                        "ordType": ord_type_text,
                        "limit": str(_limit(limit)),
                    }
                    for inst_id in sorted(target_inst_ids)
                )
            else:
                params_list.append(
                    {
                        "instType": "SWAP",
                        "ordType": ord_type_text,
                        "limit": str(_limit(limit)),
                    }
                )

        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for params in params_list[: self.max_instrument_queries * 3]:
            response = await self.executor._with_retry(fetch_algos, params)
            for row in _response_rows(response):
                row_inst_id = str(row.get("instId") or "").strip().upper()
                if target_inst_ids and row_inst_id not in target_inst_ids:
                    continue
                algo_id = str(row.get("algoId") or row.get("algoClOrdId") or "").strip()
                key = (row_inst_id, algo_id, str(row.get("ordType") or ""))
                if not algo_id or key in seen:
                    continue
                seen.add(key)
                rows.append(row)

        orders = [
            _native_algo_order_to_protection_shape(
                row,
                symbol_normalizer=self.symbol_normalizer,
            )
            for row in rows
        ]
        return [order for order in orders if order is not None]


def group_okx_native_fill_rows(
    rows: Iterable[dict[str, Any]],
    *,
    symbol_normalizer: Any = normalize_trading_symbol,
) -> list[OkxNativeFillGroup]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        order_id = str(row.get("ordId") or row.get("order") or "").strip()
        inst_id = str(row.get("instId") or "").strip().upper()
        symbol = symbol_normalizer(inst_id)
        side = str(row.get("side") or "").lower().strip()
        pos_side = str(row.get("posSide") or "").lower().strip()
        contracts = _safe_float(row.get("fillSz") or row.get("sz"), 0.0)
        price = _safe_float(row.get("fillPx") or row.get("price"), 0.0)
        timestamp_ms = _safe_float(row.get("ts") or row.get("fillTime"), 0.0)
        if not order_id or not inst_id or not symbol or not side or contracts <= 0 or price <= 0:
            continue
        group = groups.setdefault(
            order_id,
            {
                "order_id": order_id,
                "inst_id": inst_id,
                "symbol": symbol,
                "side": side,
                "pos_side": pos_side,
                "contracts": 0.0,
                "price_value": 0.0,
                "fee_abs": 0.0,
                "fill_pnl": 0.0,
                "timestamp_ms": 0.0,
                "trade_ids": set(),
                "rows": [],
            },
        )
        group["contracts"] += contracts
        group["price_value"] += price * contracts
        group["fee_abs"] += abs(_safe_float(row.get("fee"), 0.0))
        group["fill_pnl"] += _safe_float(row.get("fillPnl") or row.get("pnl"), 0.0)
        group["timestamp_ms"] = max(_safe_float(group.get("timestamp_ms"), 0.0), timestamp_ms)
        group["rows"].append(dict(row))
        if row.get("tradeId"):
            group["trade_ids"].add(str(row.get("tradeId")))
        if not group.get("pos_side") and pos_side:
            group["pos_side"] = pos_side

    result: list[OkxNativeFillGroup] = []
    for group in groups.values():
        contracts = _safe_float(group.get("contracts"), 0.0)
        if contracts <= 0:
            continue
        timestamp_ms = _safe_float(group.get("timestamp_ms"), 0.0)
        result.append(
            OkxNativeFillGroup(
                order_id=str(group.get("order_id") or ""),
                trade_ids=tuple(sorted(group.get("trade_ids") or set())),
                inst_id=str(group.get("inst_id") or ""),
                symbol=str(group.get("symbol") or ""),
                side=str(group.get("side") or ""),
                pos_side=str(group.get("pos_side") or ""),
                contracts=contracts,
                avg_price=_safe_float(group.get("price_value"), 0.0) / contracts,
                fee_abs=_safe_float(group.get("fee_abs"), 0.0),
                fill_pnl=_safe_float(group.get("fill_pnl"), 0.0),
                timestamp_ms=timestamp_ms,
                timestamp=_datetime_from_ms(timestamp_ms),
                raw_count=len(group.get("rows") or []),
                rows=tuple(group.get("rows") or ()),
            )
        )
    return sorted(result, key=lambda item: item.timestamp_ms, reverse=True)


def _target_inst_ids(
    *,
    symbols: Iterable[Any] | None,
    inst_ids: Iterable[Any] | None,
) -> set[str]:
    result: set[str] = set()
    for value in inst_ids or ():
        text = str(value or "").strip().upper()
        if text:
            if text.endswith("-SWAP"):
                result.add(text)
            else:
                inst_id = okx_inst_id_from_symbol(text)
                if inst_id:
                    result.add(inst_id)
    for symbol in symbols or ():
        inst_id = okx_inst_id_from_symbol(symbol)
        if inst_id:
            result.add(inst_id)
    return result


def _target_order_ids(order_ids: Iterable[Any] | None) -> set[str]:
    result: set[str] = set()
    for value in order_ids or ():
        text = str(value or "").strip()
        if text:
            result.add(text)
    return result


def _target_pos_ids(pos_ids: Iterable[Any] | None) -> set[str]:
    result: set[str] = set()
    for value in pos_ids or ():
        text = str(value or "").strip()
        if text:
            result.add(text)
    return result


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    chunk_size = max(int(size or 1), 1)
    for index in range(0, len(values), chunk_size):
        yield values[index : index + chunk_size]


def _native_position_row_is_open(row: dict[str, Any]) -> bool:
    return abs(_safe_float(row.get("pos") or row.get("qty"), 0.0)) > 0


def _native_position_to_ccxt_shape(
    row: dict[str, Any],
    *,
    symbol_normalizer: Any,
) -> dict[str, Any]:
    inst_id = str(row.get("instId") or "").strip().upper()
    pos = _safe_float(row.get("pos") or row.get("qty"), 0.0)
    pos_side = str(row.get("posSide") or "").lower().strip()
    side = pos_side
    if side == "net" and pos:
        side = "short" if pos < 0 else "long"
    if side not in {"long", "short", "net"}:
        side = "short" if pos < 0 else "long" if pos > 0 else ""
    contracts = abs(pos)
    contract_size = _safe_float(row.get("ctVal"), 0.0)
    mark_price = _safe_float(row.get("markPx") or row.get("last"), 0.0)
    entry_price = _safe_float(row.get("avgPx"), 0.0)
    return {
        "id": str(row.get("posId") or ""),
        "symbol": inst_id or symbol_normalizer(row.get("instId")),
        "side": side,
        "contracts": contracts,
        "contractSize": contract_size if contract_size > 0 else None,
        "markPrice": mark_price,
        "entryPrice": entry_price,
        "unrealizedPnl": _safe_float(row.get("upl"), 0.0),
        "leverage": _safe_float(row.get("lever"), 0.0),
        "marginMode": row.get("mgnMode"),
        "notional": _safe_float(row.get("notionalUsd") or row.get("notional"), 0.0),
        "liquidationPrice": _safe_float(row.get("liqPx"), 0.0),
        "timestamp": _safe_float(row.get("uTime") or row.get("cTime"), 0.0),
        "info": dict(row),
    }


def _native_order_to_ccxt_shape(
    row: dict[str, Any],
    *,
    symbol_normalizer: Any,
) -> dict[str, Any]:
    inst_id = str(row.get("instId") or "").strip().upper()
    order_id = str(row.get("ordId") or row.get("clOrdId") or "").strip()
    state = str(row.get("state") or "").lower().strip()
    amount = _safe_float(row.get("sz"), 0.0)
    filled = _safe_float(row.get("accFillSz") or row.get("fillSz"), 0.0)
    price = _safe_float(row.get("px"), 0.0)
    average = _safe_float(row.get("avgPx"), 0.0)
    return {
        "id": order_id,
        "clientOrderId": str(row.get("clOrdId") or ""),
        "symbol": inst_id or symbol_normalizer(row.get("instId")),
        "type": str(row.get("ordType") or "").lower(),
        "side": str(row.get("side") or "").lower(),
        "status": _native_order_status(state),
        "amount": amount,
        "filled": filled,
        "remaining": max(amount - filled, 0.0),
        "price": price,
        "average": average or None,
        "reduceOnly": _native_bool(row.get("reduceOnly")),
        "timestamp": _safe_float(row.get("cTime"), 0.0),
        "lastTradeTimestamp": _safe_float(row.get("uTime"), 0.0),
        "info": dict(row),
    }


def _native_algo_order_to_protection_shape(
    row: dict[str, Any],
    *,
    symbol_normalizer: Any,
) -> dict[str, Any] | None:
    inst_id = str(row.get("instId") or "").strip().upper()
    state = str(row.get("state") or "").lower().strip()
    if state and state not in {"live", "effective", "partially_effective", "open", "pending"}:
        return None

    take_profit = _safe_float(
        row.get("tpTriggerPx")
        or row.get("takeProfitTriggerPrice")
        or row.get("tpOrdPx")
        or row.get("tpPx"),
        0.0,
    )
    stop_loss = _safe_float(
        row.get("slTriggerPx")
        or row.get("stopLossTriggerPrice")
        or row.get("slOrdPx")
        or row.get("slPx"),
        0.0,
    )
    trigger_price = _safe_float(row.get("triggerPx") or row.get("moveTriggerPx"), 0.0)
    if take_profit <= 0 and stop_loss <= 0 and trigger_price <= 0:
        return None

    close_side = str(row.get("side") or "").lower().strip()
    pos_side = str(row.get("posSide") or "").lower().strip()
    if pos_side not in {"long", "short"}:
        pos_side = "short" if close_side == "buy" else "long" if close_side == "sell" else ""
    if pos_side not in {"long", "short"}:
        return None

    return {
        "symbol": symbol_normalizer(inst_id),
        "position_side": pos_side,
        "close_side": close_side,
        "order_type": str(row.get("ordType") or "").lower(),
        "take_profit_price": take_profit if take_profit > 0 else None,
        "stop_loss_price": stop_loss if stop_loss > 0 else None,
        "trigger_price": trigger_price if trigger_price > 0 else None,
        "algo_id": str(row.get("algoId") or row.get("algoClOrdId") or ""),
        "updated_at_ms": _safe_float(row.get("uTime") or row.get("cTime"), 0.0),
        "raw": {"info": dict(row), **dict(row)},
    }


def _native_order_status(state: str) -> str:
    return {
        "live": "open",
        "partially_filled": "partially_filled",
        "filled": "closed",
        "canceled": "canceled",
        "cancelled": "canceled",
    }.get(str(state or "").lower(), state or "open")


def _native_bool(value: Any) -> bool | None:
    text = str(value or "").lower().strip()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _row_matches(
    row: dict[str, Any],
    *,
    since_ms: float,
    side: str | None,
    target_inst_ids: set[str],
) -> bool:
    timestamp_ms = _safe_float(row.get("ts") or row.get("fillTime"), 0.0)
    if since_ms > 0 and (timestamp_ms <= 0 or timestamp_ms < since_ms):
        return False
    expected_side = str(side or "").lower().strip()
    if expected_side and str(row.get("side") or "").lower().strip() != expected_side:
        return False
    row_inst_id = str(row.get("instId") or "").strip().upper()
    if not target_inst_ids or not row_inst_id:
        return True
    return row_inst_id in target_inst_ids or any(
        row_inst_id.startswith(f"{inst_id}-OFF") for inst_id in target_inst_ids
    )


def _single_inst_id_for_order(order_id: str, target_inst_ids: set[str]) -> str:
    if len(target_inst_ids) == 1:
        return next(iter(target_inst_ids))
    return ""


def _order_row_matches(
    row: dict[str, Any],
    *,
    since_ms: float,
    target_inst_ids: set[str],
    target_order_ids: set[str],
) -> bool:
    timestamp_ms = _safe_float(row.get("cTime") or row.get("uTime"), 0.0)
    if since_ms > 0 and (timestamp_ms <= 0 or timestamp_ms < since_ms):
        return False
    row_order_id = str(row.get("ordId") or row.get("order") or "").strip()
    if target_order_ids and row_order_id not in target_order_ids:
        return False
    row_inst_id = str(row.get("instId") or "").strip().upper()
    if target_inst_ids and row_inst_id and row_inst_id not in target_inst_ids:
        return False
    return True


def _position_history_row_matches(
    row: dict[str, Any],
    *,
    since_ms: float,
    target_inst_ids: set[str],
    target_pos_ids: set[str],
) -> bool:
    timestamp_ms = _safe_float(row.get("uTime") or row.get("cTime"), 0.0)
    if since_ms > 0 and (timestamp_ms <= 0 or timestamp_ms < since_ms):
        return False
    row_pos_id = str(row.get("posId") or "").strip()
    if target_pos_ids and row_pos_id not in target_pos_ids:
        return False
    row_inst_id = str(row.get("instId") or "").strip().upper()
    if target_inst_ids and row_inst_id and row_inst_id not in target_inst_ids:
        return False
    return True


def _response_rows(response: Any) -> list[dict[str, Any]]:
    rows = response.get("data", []) if isinstance(response, dict) else []
    return [row for row in rows or [] if isinstance(row, dict)]


def _limit(value: int) -> int:
    return max(1, min(int(value or DEFAULT_FILL_LIMIT), 100))


def _max_pages(value: int) -> int:
    return max(1, min(int(value or DEFAULT_MAX_FILL_PAGES), 20))


def _oldest_timestamp_ms(rows: Iterable[dict[str, Any]]) -> float:
    timestamps = [
        _safe_float(row.get("ts") or row.get("fillTime"), 0.0)
        for row in rows
        if isinstance(row, dict)
    ]
    timestamps = [item for item in timestamps if item > 0]
    return min(timestamps) if timestamps else 0.0


def _oldest_fill_pagination_cursor(rows: Iterable[dict[str, Any]]) -> str:
    oldest_row: dict[str, Any] | None = None
    oldest_timestamp = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = _safe_float(row.get("ts") or row.get("fillTime"), 0.0)
        if timestamp <= 0:
            continue
        if oldest_row is None or timestamp < oldest_timestamp:
            oldest_row = row
            oldest_timestamp = timestamp
    if not oldest_row:
        return ""
    return str(
        oldest_row.get("billId")
        or oldest_row.get("bill_id")
        or oldest_row.get("tradeId")
        or oldest_row.get("ordId")
        or ""
    ).strip()


def _oldest_order_timestamp_ms(rows: Iterable[dict[str, Any]]) -> float:
    timestamps = [
        _safe_float(row.get("cTime") or row.get("uTime"), 0.0)
        for row in rows
        if isinstance(row, dict)
    ]
    timestamps = [item for item in timestamps if item > 0]
    return min(timestamps) if timestamps else 0.0


def _oldest_order_pagination_cursor(rows: Iterable[dict[str, Any]]) -> str:
    oldest_row: dict[str, Any] | None = None
    oldest_timestamp = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = _safe_float(row.get("cTime") or row.get("uTime"), 0.0)
        if timestamp <= 0:
            continue
        if oldest_row is None or timestamp < oldest_timestamp:
            oldest_row = row
            oldest_timestamp = timestamp
    if not oldest_row:
        return ""
    return str(oldest_row.get("ordId") or oldest_row.get("clOrdId") or "").strip()


def _oldest_position_history_timestamp_ms(rows: Iterable[dict[str, Any]]) -> float:
    timestamps = [
        _safe_float(row.get("uTime") or row.get("cTime"), 0.0)
        for row in rows
        if isinstance(row, dict)
    ]
    timestamps = [item for item in timestamps if item > 0]
    return min(timestamps) if timestamps else 0.0


def _oldest_position_history_pagination_cursor(rows: Iterable[dict[str, Any]]) -> str:
    oldest_row: dict[str, Any] | None = None
    oldest_timestamp = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = _safe_float(row.get("uTime") or row.get("cTime"), 0.0)
        if timestamp <= 0:
            continue
        if oldest_row is None or timestamp < oldest_timestamp:
            oldest_row = row
            oldest_timestamp = timestamp
    if not oldest_row:
        return ""
    return str(oldest_row.get("posId") or oldest_row.get("uTime") or "").strip()


def _fill_row_identity(row: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    return (
        row.get("billId") or row.get("bill_id"),
        row.get("ordId") or row.get("order"),
        row.get("tradeId"),
        row.get("ts") or row.get("fillTime"),
        row.get("instId"),
    )


def _order_row_identity(row: dict[str, Any]) -> str:
    return str(row.get("ordId") or row.get("clOrdId") or "").strip()


def _position_history_row_identity(row: dict[str, Any]) -> str:
    pos_id = str(row.get("posId") or "").strip()
    inst_id = str(row.get("instId") or "").strip().upper()
    u_time = str(row.get("uTime") or row.get("cTime") or "").strip()
    if pos_id or inst_id or u_time:
        return "|".join([pos_id, inst_id, u_time])
    return ""


def _timestamp_ms(value: datetime | int | float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).timestamp() * 1000.0
    return _safe_float(value, 0.0)


def _datetime_from_ms(value: Any) -> datetime | None:
    timestamp_ms = _safe_float(value, 0.0)
    if timestamp_ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000.0, UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: float) -> float:
    return round(float(value), 8)


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()
