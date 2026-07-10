from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select

from db.repositories.base import BaseRepository
from models.market_data import Kline, Ticker


def _utc_time_key(value: datetime) -> datetime:
    return value.replace(tzinfo=None) if value.tzinfo is not None else value


class MarketRepository(BaseRepository):
    """Repository for market data (Klines, Tickers)."""

    async def upsert_ticker(self, symbol: str, data: dict) -> Ticker:
        ticker = await self.find_one_by(Ticker, symbol=symbol)
        if ticker:
            for key, value in data.items():
                if hasattr(ticker, key):
                    setattr(ticker, key, value)
        else:
            ticker = Ticker(symbol=symbol, **data)
            self.session.add(ticker)
        await self.session.flush()
        return ticker

    async def upsert_kline(
        self, symbol: str, timeframe: str, open_time: datetime, data: dict
    ) -> Kline:
        kline = await self.find_one_by(
            Kline, symbol=symbol, timeframe=timeframe, open_time=open_time
        )
        if kline:
            for key, value in data.items():
                if hasattr(kline, key):
                    setattr(kline, key, value)
        else:
            kline = Kline(symbol=symbol, timeframe=timeframe, open_time=open_time, **data)
            self.session.add(kline)
        await self.session.flush()
        return kline

    async def upsert_klines_bulk(
        self,
        symbol: str,
        timeframe: str,
        rows: list[tuple[datetime, dict]],
    ) -> int:
        """Upsert one symbol/timeframe batch with a single read and flush."""

        payload_by_open_time = {
            open_time: dict(data)
            for open_time, data in rows
            if isinstance(open_time, datetime) and isinstance(data, dict)
        }
        if not payload_by_open_time:
            return 0
        existing_result = await self.session.execute(
            select(Kline).where(
                Kline.symbol == symbol,
                Kline.timeframe == timeframe,
            )
        )
        existing_by_open_time = {
            _utc_time_key(row.open_time): row
            for row in existing_result.scalars().all()
            if row.open_time is not None
        }
        for open_time, data in payload_by_open_time.items():
            kline = existing_by_open_time.get(_utc_time_key(open_time))
            if kline is None:
                self.session.add(Kline(symbol=symbol, timeframe=timeframe, open_time=open_time, **data))
                continue
            for key, value in data.items():
                if hasattr(kline, key):
                    setattr(kline, key, value)
        await self.session.flush()
        return len(payload_by_open_time)

    async def get_tickers(self, symbols: list[str] | None = None) -> list[Ticker]:
        stmt = select(Ticker)
        if symbols:
            stmt = stmt.where(Ticker.symbol.in_(symbols))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_klines(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> list[Kline]:
        result = await self.session.execute(
            select(Kline)
            .where(Kline.symbol == symbol, Kline.timeframe == timeframe)
            .order_by(Kline.open_time.desc())
            .limit(limit)
        )
        return list(result.scalars().all())[::-1]

    async def clean_old_klines(self, symbol: str, timeframe: str, keep: int = 1000) -> int:
        """Delete klines beyond the keep count for a given symbol/timeframe."""
        subquery = (
            select(Kline.id)
            .where(Kline.symbol == symbol, Kline.timeframe == timeframe)
            .order_by(Kline.open_time.desc())
            .limit(keep)
            .subquery()
        )
        stmt = delete(Kline).where(
            Kline.symbol == symbol,
            Kline.timeframe == timeframe,
            Kline.id.not_in(select(subquery.c.id)),
        )
        result = await self.session.execute(stmt)
        return result.rowcount
