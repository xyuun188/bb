from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from typing import Any

from core.symbols import normalize_trading_symbol


def _default_float_parser(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ExchangeCloseFillFinder:
    """Find an OKX close fill for a local position that disappeared from exchange state."""

    def __init__(
        self,
        *,
        paper_okx_provider: Callable[[], Any | None],
        float_parser: Callable[[Any, float], float] | None = None,
        datetime_from_ms_parser: Callable[[Any], Any] | None = None,
    ) -> None:
        self.paper_okx_provider = paper_okx_provider
        self.float_parser = float_parser or _default_float_parser
        self.datetime_from_ms_parser = datetime_from_ms_parser

    async def find(self, position: Any) -> dict[str, Any]:
        paper_okx = self.paper_okx_provider()
        if not paper_okx:
            return {}

        ccxt = await paper_okx._get_ccxt()
        okx_symbol = paper_okx._to_swap_symbol(position.symbol)
        contract_size = self._contract_size(ccxt, okx_symbol)
        since = self._opened_since_ms(position)
        close_side = "buy" if position.side == "short" else "sell"
        target_quantity = abs(self.float_parser(getattr(position, "quantity", 0.0), 0.0))

        candidates = []
        candidates.extend(
            await self._closed_order_candidates(
                paper_okx,
                ccxt,
                okx_symbol=okx_symbol,
                since=since,
                close_side=close_side,
                target_quantity=target_quantity,
                contract_size=contract_size,
            )
        )
        candidates.extend(
            await self._trade_candidates(
                paper_okx,
                ccxt,
                okx_symbol=okx_symbol,
                since=since,
                close_side=close_side,
                target_quantity=target_quantity,
                contract_size=contract_size,
            )
        )
        candidates.extend(
            await self._okx_fill_history_candidates(
                paper_okx,
                ccxt,
                okx_inst_id=self._okx_inst_id_for_position(position),
                since=since,
                close_side=close_side,
                target_quantity=target_quantity,
                contract_size=contract_size,
            )
        )

        candidates = [
            candidate
            for candidate in candidates
            if candidate.get("price", 0) > 0 and candidate.get("order_id")
        ]
        if not candidates:
            return {}
        return self._best_candidate(candidates, target_quantity)

    @staticmethod
    def _best_candidate(
        candidates: list[dict[str, Any]],
        target_quantity: float,
    ) -> dict[str, Any]:
        if target_quantity > 0:
            quantity_candidates = [
                candidate for candidate in candidates if float(candidate.get("quantity") or 0.0) > 0
            ]
            if quantity_candidates:
                return sorted(
                    quantity_candidates,
                    key=lambda candidate: (
                        abs(float(candidate.get("quantity") or 0.0) - target_quantity),
                        -(candidate.get("timestamp_ms") or 0),
                    ),
                )[0]
        return sorted(candidates, key=lambda candidate: candidate.get("timestamp_ms") or 0)[-1]

    def _contract_size(self, ccxt: Any, okx_symbol: str) -> float:
        try:
            market = ccxt.market(okx_symbol)
            return self.float_parser(market.get("contractSize"), 1.0) or 1.0
        except Exception:
            return 1.0

    @staticmethod
    def _okx_inst_id_for_position(position: Any) -> str:
        for attr in ("okx_inst_id", "exchange_inst_id", "inst_id", "instrument_id"):
            value = str(getattr(position, attr, "") or "").strip().upper()
            if value:
                return value
        symbol = normalize_trading_symbol(getattr(position, "symbol", ""))
        if not symbol:
            return ""
        return f"{symbol.replace('/', '-')}-SWAP"

    @staticmethod
    def _opened_since_ms(position: Any) -> int | None:
        opened_at = getattr(position, "created_at", None)
        if not opened_at:
            return None
        if opened_at.tzinfo is None:
            opened_at = opened_at.replace(tzinfo=UTC)
        return int(opened_at.timestamp() * 1000)

    async def _closed_order_candidates(
        self,
        paper_okx: Any,
        ccxt: Any,
        *,
        okx_symbol: str,
        since: int | None,
        close_side: str,
        target_quantity: float,
        contract_size: float,
    ) -> list[dict[str, Any]]:
        try:
            orders = await paper_okx._with_retry(ccxt.fetch_closed_orders, okx_symbol, since, 50)
        except Exception:
            orders = []

        min_timestamp = (since or 0) - 1000
        candidates: list[dict[str, Any]] = []
        for order in orders or []:
            info = order.get("info") or {}
            if order.get("side") != close_side:
                continue
            timestamp = self.float_parser(
                order.get("timestamp") or info.get("uTime") or info.get("cTime"),
                0.0,
            )
            if timestamp and timestamp < min_timestamp:
                continue
            reduce_raw = order.get("reduceOnly")
            if reduce_raw in (None, ""):
                reduce_raw = info.get("reduceOnly")
            is_reduce_only = str(reduce_raw).lower() == "true"
            order_type = info.get("ordType") or order.get("type")
            algo_id = info.get("algoId") or info.get("algoClOrdId")
            pnl = self.float_parser(info.get("pnl") or info.get("fillPnl"), 0.0)
            has_close_pnl = abs(pnl) > 1e-12
            if not is_reduce_only and not has_close_pnl:
                continue
            contracts = self.float_parser(
                order.get("filled")
                or order.get("amount")
                or info.get("fillSz")
                or info.get("accFillSz"),
                0.0,
            )
            quantity = contracts * contract_size
            if target_quantity > 0 and quantity > 0 and quantity < target_quantity * 0.2:
                continue
            candidates.append(
                {
                    "price": self.float_parser(
                        order.get("average") or order.get("price") or info.get("fillPx"), 0.0
                    ),
                    "fee": order_fee_cost(order),
                    "order_id": info.get("ordId") or order.get("id"),
                    "timestamp_ms": timestamp,
                    "timestamp": self._datetime_from_ms(timestamp),
                    "quantity": quantity,
                    "contracts": contracts,
                    "contract_size": contract_size,
                    "pnl": pnl,
                    "source": "closed_orders",
                    "reduce_only": is_reduce_only,
                    "order_type": order_type,
                    "algo_id": algo_id,
                    "order_info": info,
                }
            )
        return candidates

    async def _trade_candidates(
        self,
        paper_okx: Any,
        ccxt: Any,
        *,
        okx_symbol: str,
        since: int | None,
        close_side: str,
        target_quantity: float,
        contract_size: float,
    ) -> list[dict[str, Any]]:
        try:
            trades = await paper_okx._with_retry(ccxt.fetch_my_trades, okx_symbol, since, 100)
        except Exception:
            trades = []

        min_timestamp = (since or 0) - 1000
        trade_groups: dict[str, dict[str, Any]] = {}
        for trade in trades or []:
            info = trade.get("info") or {}
            if trade.get("side") != close_side:
                continue
            timestamp = self.float_parser(
                trade.get("timestamp") or info.get("ts") or info.get("fillTime"), 0.0
            )
            if timestamp and timestamp < min_timestamp:
                continue
            fill_pnl = self.float_parser(info.get("fillPnl") or info.get("pnl"), 0.0)
            contracts = self.float_parser(trade.get("amount") or info.get("fillSz"), 0.0)
            quantity = contracts * contract_size
            if abs(fill_pnl) <= 1e-12 and not (
                target_quantity > 0 and quantity >= target_quantity * 0.8
            ):
                continue
            price = self.float_parser(trade.get("price") or info.get("fillPx"), 0.0)
            if price <= 0:
                continue
            order_id = info.get("ordId") or trade.get("order") or trade.get("id")
            group = trade_groups.setdefault(
                order_id,
                {
                    "price_value": 0.0,
                    "quantity": 0.0,
                    "fee": 0.0,
                    "pnl": 0.0,
                    "timestamp_ms": timestamp,
                    "order_id": order_id,
                    "source": "my_trades",
                    "order_info": info,
                },
            )
            group.setdefault("order_type", info.get("ordType") or trade.get("type"))
            group.setdefault("algo_id", info.get("algoId") or info.get("algoClOrdId"))
            group["price_value"] += price * quantity
            group["quantity"] += quantity
            group["contracts"] = group.get("contracts", 0.0) + contracts
            group["contract_size"] = contract_size
            group["fee"] += order_fee_cost(trade)
            group["pnl"] += fill_pnl
            group["timestamp_ms"] = max(group["timestamp_ms"] or 0, timestamp or 0)

        candidates: list[dict[str, Any]] = []
        for group in trade_groups.values():
            quantity = group.get("quantity") or 0.0
            if target_quantity > 0 and quantity > 0 and quantity < target_quantity * 0.2:
                continue
            timestamp = group.get("timestamp_ms") or 0
            candidates.append(
                {
                    "price": group["price_value"] / quantity if quantity > 0 else 0.0,
                    "fee": group.get("fee") or 0.0,
                    "order_id": group.get("order_id"),
                    "timestamp_ms": timestamp,
                    "timestamp": self._datetime_from_ms(timestamp),
                    "quantity": quantity,
                    "pnl": group.get("pnl") or 0.0,
                    "source": "my_trades",
                    "order_type": group.get("order_type"),
                    "algo_id": group.get("algo_id"),
                    "order_info": group.get("order_info"),
                }
            )
        return candidates

    async def _okx_fill_history_candidates(
        self,
        paper_okx: Any,
        ccxt: Any,
        *,
        okx_inst_id: str,
        since: int | None,
        close_side: str,
        target_quantity: float,
        contract_size: float,
    ) -> list[dict[str, Any]]:
        fetch_fills = getattr(ccxt, "privateGetTradeFillsHistory", None)
        if not okx_inst_id or not callable(fetch_fills):
            return []

        params = {
            "instType": "SWAP",
            "instId": okx_inst_id,
            "limit": "100",
        }
        try:
            response = await paper_okx._with_retry(fetch_fills, params)
        except Exception:
            try:
                response = await paper_okx._with_retry(
                    fetch_fills,
                    {
                        "instType": "SWAP",
                        "limit": "100",
                    },
                )
            except Exception:
                return []

        rows = response.get("data", []) if isinstance(response, dict) else []
        min_timestamp = (since or 0) - 1000
        groups: dict[str, dict[str, Any]] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            row_inst_id = str(row.get("instId") or "").strip().upper()
            if (
                row_inst_id
                and row_inst_id != okx_inst_id
                and not row_inst_id.startswith(f"{okx_inst_id}-OFF")
            ):
                continue
            if str(row.get("side") or "").lower() != close_side:
                continue
            timestamp = self.float_parser(row.get("ts") or row.get("fillTime"), 0.0)
            if timestamp and timestamp < min_timestamp:
                continue
            price = self.float_parser(row.get("fillPx") or row.get("price"), 0.0)
            if price <= 0:
                continue
            contracts = self.float_parser(row.get("fillSz") or row.get("sz"), 0.0)
            if contracts <= 0:
                continue
            order_id = str(row.get("ordId") or row.get("order") or "").strip()
            if not order_id:
                continue
            group = groups.setdefault(
                order_id,
                {
                    "price_value": 0.0,
                    "contracts": 0.0,
                    "fee": 0.0,
                    "pnl": 0.0,
                    "timestamp_ms": timestamp,
                    "order_id": order_id,
                    "source": "okx_fills_history",
                    "order_info": row,
                },
            )
            group["price_value"] += price * contracts
            group["contracts"] += contracts
            group["fee"] += abs(self.float_parser(row.get("fee"), 0.0))
            group["pnl"] += self.float_parser(row.get("fillPnl") or row.get("pnl"), 0.0)
            group["timestamp_ms"] = max(group["timestamp_ms"] or 0, timestamp or 0)
            if timestamp >= group["timestamp_ms"]:
                group["order_info"] = row

        candidates: list[dict[str, Any]] = []
        for group in groups.values():
            contracts = float(group.get("contracts") or 0.0)
            if contracts <= 0:
                continue
            quantity_contract_size = contract_size
            quantity = contracts * quantity_contract_size
            inferred_contract_size = False
            if (
                target_quantity > 0
                and quantity_contract_size == 1.0
                and quantity > target_quantity * 1.2
            ):
                quantity_contract_size = target_quantity / contracts
                quantity = target_quantity
                inferred_contract_size = True
            if target_quantity > 0 and quantity > 0 and quantity < target_quantity * 0.2:
                continue
            timestamp = group.get("timestamp_ms") or 0
            candidates.append(
                {
                    "price": group["price_value"] / contracts,
                    "fee": group.get("fee") or 0.0,
                    "order_id": group.get("order_id"),
                    "timestamp_ms": timestamp,
                    "timestamp": self._datetime_from_ms(timestamp),
                    "quantity": quantity,
                    "contracts": contracts,
                    "contract_size": quantity_contract_size,
                    "contract_size_inferred_from_target": inferred_contract_size,
                    "pnl": group.get("pnl") or 0.0,
                    "source": "okx_fills_history",
                    "order_info": group.get("order_info"),
                }
            )
        return candidates

    def _datetime_from_ms(self, timestamp_ms: Any) -> Any | None:
        if not timestamp_ms or self.datetime_from_ms_parser is None:
            return None
        return self.datetime_from_ms_parser(timestamp_ms)


def order_fee_cost(order: dict[str, Any]) -> float:
    fee = order.get("fee")
    if isinstance(fee, dict):
        return float(fee.get("cost") or 0.0)
    info_fee = (order.get("info") or {}).get("fee")
    try:
        return abs(float(info_fee or 0.0))
    except (TypeError, ValueError):
        return 0.0
