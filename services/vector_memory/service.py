"""Optional vector-memory service for historical context retrieval."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select

from config.settings import settings
from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.decision import AIDecision
from models.news import NewsArticle
from models.trade import Position
from services.vector_memory.store import VectorMemoryStore, build_vector_memory_store
from services.vector_memory.types import VectorMemoryDocument, VectorMemoryHit

logger = structlog.get_logger(__name__)


class VectorMemoryService:
    """Index and search historical cases without blocking trading decisions."""

    def __init__(self) -> None:
        self._store: VectorMemoryStore | None = None
        self._lock = asyncio.Lock()
        self._last_reindex_at: datetime | None = None
        self._last_error: str = ""

    @property
    def enabled(self) -> bool:
        return bool(settings.vector_memory_enabled)

    def _get_store(self) -> VectorMemoryStore:
        if self._store is None:
            self._store = build_vector_memory_store(
                settings.data_dir / "vector_memory",
                backend=settings.vector_memory_backend,
                dimension=int(settings.vector_memory_dimension),
                max_documents=int(settings.vector_memory_max_documents),
            )
        return self._store

    async def reset_store(self) -> None:
        """Reload the configured backend after settings change."""

        async with self._lock:
            self._store = None
            self._last_error = ""

    async def status(self) -> dict[str, Any]:
        """Return current vector memory status."""

        if not self.enabled:
            return {
                "enabled": False,
                "backend": "disabled",
                "status": "disabled",
                "document_count": 0,
                "configured_backend": settings.vector_memory_backend,
                "min_score": float(settings.vector_memory_min_score),
                "last_reindex_at": _iso(self._last_reindex_at),
                "last_error": self._last_error,
            }
        try:
            stats = self._get_store().stats()
            return {
                "enabled": True,
                "backend": stats.get("backend", "unknown"),
                "status": "ready",
                "document_count": int(stats.get("document_count") or 0),
                "configured_backend": settings.vector_memory_backend,
                "min_score": float(settings.vector_memory_min_score),
                "path": stats.get("path"),
                "last_reindex_at": _iso(self._last_reindex_at),
                "last_error": self._last_error,
            }
        except Exception as exc:
            self._last_error = safe_error_text(exc, limit=180)
            return {
                "enabled": True,
                "backend": "unknown",
                "status": "error",
                "document_count": 0,
                "configured_backend": settings.vector_memory_backend,
                "min_score": float(settings.vector_memory_min_score),
                "last_reindex_at": _iso(self._last_reindex_at),
                "last_error": self._last_error,
            }

    async def reindex_recent(self) -> dict[str, Any]:
        """Index recent decisions and news into vector memory."""

        if not self.enabled:
            return {"enabled": False, "status": "disabled", "indexed": 0}
        async with self._lock:
            try:
                documents = await self._load_recent_documents()
                indexed = self._get_store().upsert(documents)
                self._last_reindex_at = datetime.now(UTC)
                self._last_error = ""
                return {
                    "enabled": True,
                    "status": "ok",
                    "indexed": indexed,
                    "last_reindex_at": _iso(self._last_reindex_at),
                    "store": self._get_store().stats(),
                }
            except Exception as exc:
                self._last_error = safe_error_text(exc, limit=240)
                logger.warning("vector memory reindex failed", error=self._last_error)
                return {
                    "enabled": True,
                    "status": "error",
                    "indexed": 0,
                    "error": self._last_error,
                }

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        symbol: str = "",
        kind: str = "",
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """Search vector memory for similar records."""

        if not self.enabled:
            return {"enabled": False, "status": "disabled", "hits": []}
        text = str(query or "").strip()
        if not text:
            return {"enabled": True, "status": "empty_query", "hits": []}
        filters = {
            key: value
            for key, value in {
                "symbol": _normalize_symbol(symbol),
                "kind": kind,
            }.items()
            if value
        }
        try:
            hits = self._get_store().search(text, top_k=top_k, filters=filters)
        except Exception as exc:
            self._last_error = safe_error_text(exc, limit=180)
            logger.warning("vector memory search failed", error=self._last_error)
            return {"enabled": True, "status": "error", "error": self._last_error, "hits": []}
        threshold = (
            float(settings.vector_memory_min_score) if min_score is None else float(min_score)
        )
        filtered = [hit for hit in hits if hit.score >= threshold]
        return {
            "enabled": True,
            "status": "ok",
            "backend": self._get_store().backend_name,
            "hits": [_hit_payload(hit) for hit in filtered[: max(int(top_k or 8), 1)]],
            "min_score": threshold,
        }

    async def similar_decision_context(
        self, decision: AIDecision, raw: dict[str, Any]
    ) -> dict[str, Any]:
        """Return similar historical cases for one decision detail view."""

        if not self.enabled:
            return {"enabled": False, "status": "disabled", "hits": []}
        query = _decision_text(decision, raw)
        result = await self.search(
            query,
            top_k=6,
            symbol=_normalize_symbol(decision.symbol),
            min_score=float(settings.vector_memory_min_score),
        )
        result["query_summary"] = query[:320]
        result["influence"] = _influence_payload(
            result.get("hits") if isinstance(result.get("hits"), list) else [],
            action=str(decision.action or ""),
        )
        return result

    async def _load_recent_documents(self) -> list[VectorMemoryDocument]:
        async with get_session_ctx() as session:
            decision_rows = list(
                (
                    await session.execute(
                        select(AIDecision)
                        .order_by(AIDecision.id.desc())
                        .limit(int(settings.vector_memory_decision_index_limit))
                    )
                )
                .scalars()
                .all()
            )
            news_rows = list(
                (
                    await session.execute(
                        select(NewsArticle)
                        .order_by(NewsArticle.id.desc())
                        .limit(int(settings.vector_memory_news_index_limit))
                    )
                )
                .scalars()
                .all()
            )
            position_rows = list(
                (
                    await session.execute(
                        select(Position)
                        .where(Position.is_open.is_(False))
                        .order_by(Position.id.desc())
                        .limit(500)
                    )
                )
                .scalars()
                .all()
            )

        documents: list[VectorMemoryDocument] = []
        closed_pnl_by_symbol = _closed_pnl_by_symbol(position_rows)
        for decision in decision_rows:
            raw = decision.raw_llm_response if isinstance(decision.raw_llm_response, dict) else {}
            documents.append(_decision_document(decision, raw, closed_pnl_by_symbol))
        for article in news_rows:
            document = _news_document(article)
            if document:
                documents.append(document)
        return documents


_service = VectorMemoryService()


def get_vector_memory_service() -> VectorMemoryService:
    """Return the singleton vector memory service."""

    return _service


def _decision_document(
    decision: AIDecision,
    raw: dict[str, Any],
    closed_pnl_by_symbol: dict[str, float],
) -> VectorMemoryDocument:
    symbol = _normalize_symbol(decision.symbol)
    pnl_pct = decision.outcome_pnl_pct
    if pnl_pct is None and symbol in closed_pnl_by_symbol:
        pnl_pct = closed_pnl_by_symbol[symbol]
    outcome = decision.outcome or (
        "profit" if (pnl_pct or 0) > 0 else "loss" if (pnl_pct or 0) < 0 else ""
    )
    metadata = {
        "decision_id": decision.id,
        "analysis_type": decision.analysis_type or raw.get("analysis_type") or "",
        "confidence": float(decision.confidence or 0.0),
        "position_size_pct": float(decision.position_size_pct or 0.0),
        "is_paper": bool(decision.is_paper),
        "was_executed": bool(decision.was_executed),
    }
    return VectorMemoryDocument(
        id=f"decision:{decision.id}",
        kind="decision",
        text=_decision_text(decision, raw),
        symbol=symbol,
        action=str(decision.action or ""),
        outcome=str(outcome or ""),
        pnl_pct=pnl_pct,
        created_at=decision.created_at,
        source_ref=str(decision.id),
        metadata=metadata,
    )


def _news_document(article: NewsArticle) -> VectorMemoryDocument | None:
    text = " ".join(part for part in (article.title, article.summary) if part).strip()
    if not text:
        return None
    symbols = article.symbols_mentioned if isinstance(article.symbols_mentioned, dict) else {}
    symbol_text = ",".join(str(item) for item in symbols.get("symbols", [])[:6]) if symbols else ""
    return VectorMemoryDocument(
        id=f"news:{article.id}",
        kind="news",
        text=text,
        symbol=symbol_text,
        action="",
        outcome="",
        pnl_pct=None,
        created_at=article.published_at or article.fetched_at,
        source_ref=str(article.url or article.id),
        metadata={
            "source": article.source,
            "sentiment_score": float(article.sentiment_score or 0.0),
        },
    )


def _decision_text(decision: AIDecision, raw: dict[str, Any]) -> str:
    parts = [
        f"币种 {decision.symbol}",
        f"动作 {decision.action}",
        f"信心 {float(decision.confidence or 0.0):.3f}",
        f"仓位 {float(decision.position_size_pct or 0.0):.4f}",
        f"结果 {decision.outcome or ''}",
        f"收益 {decision.outcome_pnl_pct if decision.outcome_pnl_pct is not None else ''}",
        str(decision.reasoning or ""),
        str(decision.execution_reason or ""),
    ]
    opportunity = raw.get("opportunity_score") if isinstance(raw, dict) else None
    if isinstance(opportunity, dict):
        parts.extend(
            [
                f"预期收益 {opportunity.get('expected_net_return_pct', '')}",
                f"质量 {opportunity.get('quality_score', '')}",
                str(opportunity.get("selection_reason") or ""),
            ]
        )
    decision_maker = raw.get("decision_maker") if isinstance(raw, dict) else None
    if isinstance(decision_maker, dict):
        parts.append(str(decision_maker.get("reasoning") or decision_maker.get("reason") or ""))
    return "；".join(part for part in parts if str(part).strip())


def _closed_pnl_by_symbol(positions: list[Position]) -> dict[str, float]:
    result: dict[str, float] = {}
    for position in positions:
        symbol = _normalize_symbol(position.symbol)
        if symbol and symbol not in result:
            result[symbol] = float(position.realized_pnl or 0.0)
    return result


def _normalize_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper().replace("-SWAP", "")


def _iso(value: datetime | None) -> str | None:
    if not value:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC).isoformat()
    return value.astimezone(UTC).isoformat()


def _hit_payload(hit: VectorMemoryHit) -> dict[str, Any]:
    return {
        "id": hit.id,
        "score": hit.score,
        "kind": hit.kind,
        "text": hit.text[:700],
        "symbol": hit.symbol,
        "action": hit.action,
        "outcome": hit.outcome,
        "pnl_pct": hit.pnl_pct,
        "created_at": hit.created_at,
        "source_ref": hit.source_ref,
        "metadata": hit.metadata,
    }


def _influence_payload(hits: list[dict[str, Any]], *, action: str) -> dict[str, Any]:
    """Explain how similar historical cases should influence this decision."""

    if not hits:
        return {
            "score_delta": 0.0,
            "level": "neutral",
            "label": "未命中相似历史",
            "reason": "没有足够相似的历史案例，本次不调整策略评分。",
            "matched_count": 0,
            "loss_count": 0,
            "profit_count": 0,
        }
    normalized_action = str(action or "").lower()
    weighted = 0.0
    weight_total = 0.0
    loss_count = 0
    profit_count = 0
    same_action_loss_count = 0
    for hit in hits:
        score = max(float(hit.get("score") or 0.0), 0.0)
        pnl = hit.get("pnl_pct")
        try:
            pnl_value = float(pnl)
        except (TypeError, ValueError):
            pnl_value = 0.0
        if pnl_value < 0:
            loss_count += 1
        elif pnl_value > 0:
            profit_count += 1
        same_action = (
            normalized_action and str(hit.get("action") or "").lower() == normalized_action
        )
        if same_action and pnl_value < 0:
            same_action_loss_count += 1
        direction = 1.0 if pnl_value > 0 else -1.0 if pnl_value < 0 else 0.0
        action_multiplier = 1.25 if same_action else 0.75
        weighted += direction * score * action_multiplier
        weight_total += score * action_multiplier
    ratio = weighted / weight_total if weight_total > 0 else 0.0
    score_delta = round(max(min(ratio * 8.0, 6.0), -6.0), 2)
    if same_action_loss_count >= 2 or score_delta <= -2.5:
        level = "negative"
        label = "相似历史偏负向"
        reason = "过去相似案例亏损较多，建议降低仓位或要求更强证据，不作为硬拦截。"
    elif score_delta >= 2.5:
        level = "positive"
        label = "相似历史偏正向"
        reason = "过去相似案例盈利占优，可作为轻量加分，但仍需服从实时风控和收益评估。"
    else:
        level = "neutral"
        label = "相似历史中性"
        reason = "相似历史结果分化，本次仅作解释参考，不明显调整评分。"
    return {
        "score_delta": score_delta,
        "level": level,
        "label": label,
        "reason": reason,
        "matched_count": len(hits),
        "loss_count": loss_count,
        "profit_count": profit_count,
        "same_action_loss_count": same_action_loss_count,
        "is_hard_gate": False,
    }
