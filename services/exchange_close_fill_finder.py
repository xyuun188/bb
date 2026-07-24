from __future__ import annotations

from collections.abc import Callable
from datetime import UTC
from typing import Any

from core.symbols import normalize_trading_symbol
from services.okx_native_facts import OkxNativeFactsClient


def _default_float_parser(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ExchangeCloseFillFinder:
    """Find OKX-native close fills for a local position missing from exchange state."""

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

        since = self._opened_since_ms(position)
        close_side = "buy" if position.side == "short" else "sell"
        target_quantity = abs(self.float_parser(getattr(position, "quantity", 0.0), 0.0))
        okx_inst_id = self._okx_inst_id_for_position(position)

        candidates = []
        candidates.extend(
            await self._okx_fill_history_candidates(
                paper_okx,
                okx_inst_id=okx_inst_id,
                since=since,
                close_side=close_side,
                target_quantity=target_quantity,
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

    async def _contract_size_from_okx_instruments(
        self,
        client: OkxNativeFactsClient,
        *,
        okx_inst_id: str,
    ) -> tuple[float, str]:
        if not okx_inst_id:
            return 0.0, ""
        contract_sizes = await client.fetch_contract_sizes(inst_ids=[okx_inst_id])
        value = self.float_parser(contract_sizes.get(okx_inst_id), 0.0)
        if value > 0:
            return value, "okx_public_instruments"
        return 0.0, ""

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

    async def _okx_fill_history_candidates(
        self,
        paper_okx: Any,
        *,
        okx_inst_id: str,
        since: int | None,
        close_side: str,
        target_quantity: float,
    ) -> list[dict[str, Any]]:
        if not okx_inst_id:
            return []

        client = OkxNativeFactsClient(paper_okx)
        okx_contract_size, okx_contract_size_source = await self._contract_size_from_okx_instruments(
            client,
            okx_inst_id=okx_inst_id,
        )
        if okx_contract_size <= 0:
            return []
        try:
            groups = await client.fetch_fill_groups(
                inst_ids=[okx_inst_id],
                since=(since or 0) - 1000,
                side=close_side,
                limit=100,
                strict=True,
            )
        except Exception:
            raise

        candidates: list[dict[str, Any]] = []
        for group in groups:
            contracts = float(group.contracts or 0.0)
            if contracts <= 0:
                continue
            quantity = contracts * okx_contract_size
            if target_quantity > 0 and quantity > 0 and quantity < target_quantity * 0.2:
                continue
            timestamp = group.timestamp_ms or 0
            candidates.append(
                {
                    "price": group.avg_price,
                    "fee": group.fee_abs,
                    "order_id": group.order_id,
                    "timestamp_ms": timestamp,
                    "timestamp": self._datetime_from_ms(timestamp),
                    "quantity": quantity,
                    "contracts": contracts,
                    "contract_size": okx_contract_size,
                    "contract_size_source": okx_contract_size_source,
                    "pnl": group.fill_pnl,
                    "source": "okx_fills_history",
                    "order_info": group.latest_row,
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
