from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select

from db.repositories.base import BaseRepository
from models.market_data import Kline, Ticker


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
