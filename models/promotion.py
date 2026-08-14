"""Persistent, append-only promotion reviews."""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Index, Integer, String, Text, event, inspect
from sqlalchemy.orm import Mapped, mapped_column

from core.experiment_contracts import ExperimentContractError
from core.promotion_contracts import verify_promotion_review
from models.base import Base, TimestampMixin


class PromotionReview(Base, TimestampMixin):
    __tablename__ = "promotion_reviews"
    __table_args__ = (Index("idx_promotion_reviews_strategy_created", "strategy_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    review_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    artifact_id: Mapped[str] = mapped_column(String(160), index=True)
    strategy_id: Mapped[str] = mapped_column(String(120), index=True)
    strategy_version: Mapped[str] = mapped_column(String(120), index=True)
    from_stage: Mapped[str] = mapped_column(String(30))
    to_stage: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    review_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    review: Mapped[dict[str, Any]] = mapped_column(JSON)
    reviewer: Mapped[str | None] = mapped_column(String(160), nullable=True)
    decision_reason: Mapped[str] = mapped_column(Text, default="")


_IMMUTABLE = (
    "review_id",
    "artifact_id",
    "strategy_id",
    "strategy_version",
    "from_stage",
    "to_stage",
    "review_sha256",
    "review",
)


@event.listens_for(PromotionReview, "before_insert")
def _validate_insert(_mapper: Any, _connection: Any, target: PromotionReview) -> None:
    verify_promotion_review(target.review)
    for name in _IMMUTABLE:
        if name == "review":
            continue
        if getattr(target, name) != target.review.get(name) and name not in {"review_id", "review_sha256"}:
            raise ExperimentContractError(f"promotion registry field {name} mismatches review")
    if target.review_id != target.review.get("review_id") or target.review_sha256 != target.review.get("review_sha256"):
        raise ExperimentContractError("promotion review identity mismatch")


@event.listens_for(PromotionReview, "before_update")
def _validate_update(_mapper: Any, _connection: Any, target: PromotionReview) -> None:
    state = inspect(target)
    changed = [name for name in _IMMUTABLE if state.attrs[name].history.has_changes()]
    if changed:
        raise ExperimentContractError("promotion review immutable fields cannot be updated: " + ", ".join(changed))
    verify_promotion_review(target.review)
