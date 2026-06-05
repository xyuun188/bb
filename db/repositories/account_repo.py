from __future__ import annotations

from sqlalchemy import select

from db.repositories.base import BaseRepository
from models.account import VirtualAccount


class AccountRepository(BaseRepository):
    """Repository for virtual accounts (paper trading)."""

    async def get_or_create_account(
        self, model_name: str, initial_balance: float = 100_000.0
    ) -> VirtualAccount:
        account = await self.find_one_by(VirtualAccount, model_name=model_name)
        if account is None:
            account = VirtualAccount(
                model_name=model_name,
                initial_balance=initial_balance,
                current_balance=initial_balance,
            )
            self.session.add(account)
            await self.session.flush()
        return account

    async def get_account(self, model_name: str) -> VirtualAccount | None:
        return await self.find_one_by(VirtualAccount, model_name=model_name)

    async def get_all_accounts(self) -> list[VirtualAccount]:
        result = await self.session.execute(select(VirtualAccount))
        return list(result.scalars().all())

    async def update_balance(
        self, model_name: str, balance_delta: float, realized_pnl_delta: float = 0.0
    ) -> VirtualAccount | None:
        account = await self.get_account(model_name)
        if account:
            account.current_balance += balance_delta
            account.realized_pnl += realized_pnl_delta
            await self.session.flush()
        return account

    async def update_unrealized_pnl(
        self, model_name: str, unrealized_pnl: float
    ) -> VirtualAccount | None:
        account = await self.get_account(model_name)
        if account:
            account.unrealized_pnl = unrealized_pnl
            await self.session.flush()
        return account

    async def record_trade_result(
        self, model_name: str, is_win: bool
    ) -> VirtualAccount | None:
        account = await self.get_account(model_name)
        if account:
            account.total_trades += 1
            if is_win:
                account.winning_trades += 1
            await self.session.flush()
        return account
