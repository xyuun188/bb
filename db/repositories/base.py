from __future__ import annotations

from typing import Any, Generic, TypeVar, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from models.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Generic async repository with common CRUD operations."""

    model: type[ModelType]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _id_column(self, model: type[ModelType] | None = None) -> Any:
        return cast(Any, model or self.model).id

    async def add(self, instance: ModelType) -> ModelType:
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def add_all(self, instances: list[ModelType]) -> list[ModelType]:
        self.session.add_all(instances)
        await self.session.flush()
        return instances

    async def get(self, id: int) -> ModelType | None:
        return await self.session.get(self.model, id)

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[ModelType]:
        result = await self.session.execute(
            select(self.model).order_by(self._id_column().desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def delete(self, instance: ModelType) -> None:
        await self.session.delete(instance)
        await self.session.flush()

    async def count(self, **filters: Any) -> int:
        stmt = select(func.count()).select_from(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                stmt = stmt.where(getattr(self.model, key) == value)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def find_by(
        self, _model: type[ModelType] | None = None, **kwargs: Any
    ) -> list[ModelType]:
        model = _model or self.model
        stmt = select(model)
        for key, value in kwargs.items():
            if hasattr(model, key):
                stmt = stmt.where(getattr(model, key) == value)
        result = await self.session.execute(stmt.order_by(self._id_column(model).desc()))
        return list(result.scalars().all())

    async def find_one_by(
        self, _model: type[ModelType] | None = None, **kwargs: Any
    ) -> ModelType | None:
        model = _model or self.model
        stmt = select(model)
        for key, value in kwargs.items():
            if hasattr(model, key):
                stmt = stmt.where(getattr(model, key) == value)
        result = await self.session.execute(stmt.limit(1))
        return result.scalar_one_or_none()
