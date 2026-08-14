"""Controlled promotion-review persistence with an explicit approval boundary."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.experiment_contracts import ExperimentContractError
from core.promotion_contracts import build_promotion_review, verify_promotion_review
from models.promotion import PromotionReview


class PromotionRegistry:
    """Persist evidence and approvals; never changes a live model pointer."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, review_id: str) -> PromotionReview | None:
        return (
            await self.session.execute(
                select(PromotionReview).where(PromotionReview.review_id == str(review_id or ""))
            )
        ).scalar_one_or_none()

    async def request(
        self,
        *,
        artifact_id: str,
        strategy_id: str,
        strategy_version: str,
        from_stage: str,
        to_stage: str,
        evidence: dict[str, Any],
        reviewer: str | None = None,
        review_reason: str = "",
    ) -> PromotionReview:
        review = build_promotion_review(
            artifact_id=artifact_id,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            from_stage=from_stage,
            to_stage=to_stage,
            evidence=evidence,
            reviewer=reviewer,
            review_reason=review_reason,
        )
        existing = await self.get(review["review_id"])
        if existing is not None:
            if existing.review != review:
                raise ExperimentContractError("promotion review id already contains different evidence")
            return existing
        row = PromotionReview(
            review_id=review["review_id"],
            artifact_id=review["artifact_id"],
            strategy_id=review["strategy_id"],
            strategy_version=review["strategy_version"],
            from_stage=review["from_stage"],
            to_stage=review["to_stage"],
            status="pending",
            review_sha256=review["review_sha256"],
            review=review,
            reviewer=review.get("reviewer"),
            decision_reason=review.get("review_reason", ""),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def approve(self, review_id: str, *, reviewer: str, reason: str = "") -> PromotionReview:
        row = await self._required(review_id)
        verify_promotion_review(row.review)
        if row.status != "pending":
            raise ExperimentContractError(f"promotion review is not pending: {row.status}")
        if not str(reviewer or "").strip():
            raise ExperimentContractError("promotion approval requires a reviewer")
        if row.to_stage == "live" and row.review.get("gate", {}).get("promotion_ready") is not True:
            raise ExperimentContractError("live promotion evidence gate is not passing")
        row.status = "approved"
        row.reviewer = str(reviewer).strip()
        row.decision_reason = str(reason or row.decision_reason or "approved")[:4000]
        await self.session.flush()
        return row

    async def reject(self, review_id: str, *, reviewer: str, reason: str) -> PromotionReview:
        row = await self._required(review_id)
        if row.status != "pending":
            raise ExperimentContractError(f"promotion review is not pending: {row.status}")
        if not str(reason or "").strip():
            raise ExperimentContractError("promotion rejection requires a reason")
        row.status = "rejected"
        row.reviewer = str(reviewer or "").strip() or None
        row.decision_reason = str(reason).strip()[:4000]
        await self.session.flush()
        return row

    async def rollback(self, review_id: str, *, reviewer: str, reason: str) -> PromotionReview:
        row = await self._required(review_id)
        if row.to_stage != "live" or row.status != "approved":
            raise ExperimentContractError("only an approved live review can be rolled back")
        if not str(reason or "").strip():
            raise ExperimentContractError("rollback requires a reason")
        row.status = "rolled_back"
        row.reviewer = str(reviewer or "").strip() or None
        row.decision_reason = str(reason).strip()[:4000]
        await self.session.flush()
        return row

    async def _required(self, review_id: str) -> PromotionReview:
        row = await self.get(review_id)
        if row is None:
            raise ExperimentContractError(f"unknown promotion review {review_id!r}")
        return row
