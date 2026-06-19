"""Storage backends for optional vector memory."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import structlog

from core.safe_output import safe_error_text
from services.vector_memory.embedding import cosine_similarity, deterministic_text_vector
from services.vector_memory.types import VectorMemoryDocument, VectorMemoryHit

logger = structlog.get_logger(__name__)

_CONTROL_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZVEC_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


class VectorMemoryStore(Protocol):
    """Minimal persistence contract used by VectorMemoryService."""

    backend_name: str

    def upsert(self, documents: Iterable[VectorMemoryDocument]) -> int:
        """Persist or update documents and return the accepted count."""

    def search(
        self, query: str, *, top_k: int = 8, filters: dict[str, str] | None = None
    ) -> list[VectorMemoryHit]:
        """Search for similar documents."""

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""


def document_to_fields(document: VectorMemoryDocument) -> dict[str, Any]:
    """Convert a document into scalar fields stored with the vector."""

    metadata = json.dumps(
        _safe_metadata(document.metadata),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "kind": _clean_scalar_text(document.kind, limit=80),
        "text": _clean_scalar_text(document.text, limit=4000),
        "symbol": _clean_scalar_text(document.symbol, limit=120),
        "action": _clean_scalar_text(document.action, limit=80),
        "outcome": _clean_scalar_text(document.outcome, limit=80),
        "pnl_pct": float(document.pnl_pct) if document.pnl_pct is not None else 0.0,
        "created_at": document.created_at.isoformat() if document.created_at else "",
        "source_ref": _clean_scalar_text(document.source_ref, limit=1000),
        "metadata_json": _clean_scalar_text(metadata, limit=3000),
    }


def fields_to_hit(doc_id: str, score: float, fields: dict[str, Any]) -> VectorMemoryHit:
    """Convert persisted scalar fields into an API-safe hit."""

    metadata: dict[str, Any] = {}
    raw_metadata = fields.get("metadata_json")
    if isinstance(raw_metadata, str) and raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata = parsed
        except json.JSONDecodeError:
            metadata = {}
    pnl_pct: float | None
    try:
        pnl_pct = float(fields.get("pnl_pct"))
    except (TypeError, ValueError):
        pnl_pct = None
    return VectorMemoryHit(
        id=str(doc_id),
        score=round(float(score), 6),
        kind=str(fields.get("kind") or ""),
        text=str(fields.get("text") or ""),
        symbol=str(fields.get("symbol") or ""),
        action=str(fields.get("action") or ""),
        outcome=str(fields.get("outcome") or ""),
        pnl_pct=pnl_pct,
        created_at=str(fields.get("created_at") or "") or None,
        source_ref=str(fields.get("source_ref") or ""),
        metadata=metadata,
    )


class JsonVectorMemoryStore:
    """Small durable fallback store used when zvec is unavailable."""

    backend_name = "jsonl"

    def __init__(self, path: Path, *, dimension: int, max_documents: int) -> None:
        self.path = path
        self.dimension = max(int(dimension), 16)
        self.max_documents = max(int(max_documents), 100)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def upsert(self, documents: Iterable[VectorMemoryDocument]) -> int:
        existing = {item["id"]: item for item in self._read_rows()}
        accepted = 0
        for document in documents:
            if not document.text.strip():
                continue
            fields = document_to_fields(document)
            existing[document.id] = {
                "id": document.id,
                "fields": fields,
                "vector": deterministic_text_vector(fields["text"], dimension=self.dimension),
            }
            accepted += 1
        rows = list(existing.values())[-self.max_documents :]
        self._write_rows(rows)
        return accepted

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        filters: dict[str, str] | None = None,
    ) -> list[VectorMemoryHit]:
        query_vector = deterministic_text_vector(query, dimension=self.dimension)
        hits: list[VectorMemoryHit] = []
        for row in self._read_rows():
            fields = row.get("fields") if isinstance(row.get("fields"), dict) else {}
            if not self._matches_filters(fields, filters):
                continue
            score = cosine_similarity(query_vector, row.get("vector") or [])
            hits.append(fields_to_hit(str(row.get("id") or ""), score, fields))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(int(top_k or 8), 1)]

    def stats(self) -> dict[str, Any]:
        rows = self._read_rows()
        return {"backend": self.backend_name, "document_count": len(rows), "path": str(self.path)}

    def _read_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get("id"):
                rows.append(parsed)
        return rows

    def _write_rows(self, rows: list[dict[str, Any]]) -> None:
        self.path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
            + ("\n" if rows else ""),
            encoding="utf-8",
        )

    @staticmethod
    def _matches_filters(fields: dict[str, Any], filters: dict[str, str] | None) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if not value:
                continue
            if str(fields.get(key) or "").upper() != str(value).upper():
                return False
        return True


class ZvecVectorMemoryStore:
    """zvec-backed vector memory store."""

    backend_name = "zvec"
    write_batch_size = 500

    def __init__(self, path: Path, *, dimension: int, max_documents: int) -> None:
        self.path = path
        self.dimension = max(int(dimension), 16)
        self.max_documents = max(int(max_documents), 100)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._zvec = self._load_zvec()
        self._collection = self._open_or_create()
        self._last_known_count = 0

    @staticmethod
    def _load_zvec() -> Any:
        import zvec

        return zvec

    def _open_or_create(self) -> Any:
        if self.path.exists():
            try:
                return self._zvec.open(str(self.path))
            except Exception as exc:
                logger.warning("zvec collection open failed", error=safe_error_text(exc))
        schema = self._zvec.CollectionSchema(
            name="bb_vector_memory",
            fields=[
                self._zvec.FieldSchema("kind", self._zvec.DataType.STRING, nullable=True),
                self._zvec.FieldSchema("text", self._zvec.DataType.STRING, nullable=True),
                self._zvec.FieldSchema("symbol", self._zvec.DataType.STRING, nullable=True),
                self._zvec.FieldSchema("action", self._zvec.DataType.STRING, nullable=True),
                self._zvec.FieldSchema("outcome", self._zvec.DataType.STRING, nullable=True),
                self._zvec.FieldSchema("pnl_pct", self._zvec.DataType.DOUBLE, nullable=True),
                self._zvec.FieldSchema("created_at", self._zvec.DataType.STRING, nullable=True),
                self._zvec.FieldSchema("source_ref", self._zvec.DataType.STRING, nullable=True),
                self._zvec.FieldSchema("metadata_json", self._zvec.DataType.STRING, nullable=True),
            ],
            vectors=[
                self._zvec.VectorSchema(
                    "embedding",
                    self._zvec.DataType.VECTOR_FP32,
                    dimension=self.dimension,
                    index_param=self._zvec.FlatIndexParam(metric_type=self._zvec.MetricType.COSINE),
                )
            ],
        )
        return self._zvec.create_and_open(str(self.path), schema)

    def upsert(self, documents: Iterable[VectorMemoryDocument]) -> int:
        docs = []
        for document in documents:
            if not document.text.strip():
                continue
            docs.append(
                self._zvec.Doc(
                    id=_zvec_safe_doc_id(document.id),
                    fields=document_to_fields(document),
                    vectors={
                        "embedding": deterministic_text_vector(
                            _clean_scalar_text(document.text, limit=4000),
                            dimension=self.dimension,
                        )
                    },
                )
            )
        if not docs:
            return 0
        for offset in range(0, len(docs), self.write_batch_size):
            self._collection.upsert(docs[offset : offset + self.write_batch_size])
        self._collection.flush()
        self._last_known_count = max(self._last_known_count, len(docs))
        return len(docs)

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        filters: dict[str, str] | None = None,
    ) -> list[VectorMemoryHit]:
        fetch_limit = _zvec_fetch_limit(top_k, filters)
        results = self._collection.query(
            self._zvec.Query(
                field_name="embedding",
                vector=deterministic_text_vector(query, dimension=self.dimension),
            ),
            topk=fetch_limit,
            include_vector=False,
        )
        hits = [
            fields_to_hit(
                doc.id, _zvec_score_to_similarity(float(doc.score or 0.0)), doc.fields or {}
            )
            for doc in results
        ]
        hits = [hit for hit in hits if _hit_matches_filters(hit, filters)]
        return hits[: max(int(top_k or 8), 1)]

    def stats(self) -> dict[str, Any]:
        try:
            raw_stats = self._collection.stats
            stats = raw_stats() if callable(raw_stats) else raw_stats
            count = int(getattr(stats, "doc_count", 0) or getattr(stats, "row_count", 0) or 0)
        except Exception:
            count = 0
        count = max(count, self._last_known_count)
        return {"backend": self.backend_name, "document_count": count, "path": str(self.path)}


def _zvec_score_to_similarity(score: float) -> float:
    """Convert zvec cosine distance-like score to higher-is-better similarity."""

    if score <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - score))


def build_vector_memory_store(
    path: Path,
    *,
    backend: str,
    dimension: int,
    max_documents: int,
) -> VectorMemoryStore:
    """Build the configured vector memory store with safe fallback."""

    normalized = str(backend or "auto").lower()
    if normalized in {"auto", "zvec"}:
        try:
            return ZvecVectorMemoryStore(
                path / "zvec",
                dimension=dimension,
                max_documents=max_documents,
            )
        except Exception as exc:
            if normalized == "zvec":
                raise
            logger.warning(
                "zvec unavailable; falling back to json vector memory", error=safe_error_text(exc)
            )
    return JsonVectorMemoryStore(
        path / "vector_memory.jsonl",
        dimension=dimension,
        max_documents=max_documents,
    )


def _zvec_fetch_limit(top_k: int, filters: dict[str, str] | None) -> int:
    requested = max(int(top_k or 8), 1)
    if not filters:
        return requested
    return min(max(requested * 20, 100), 1000)


def _hit_matches_filters(hit: VectorMemoryHit, filters: dict[str, str] | None) -> bool:
    if not filters:
        return True
    values = {
        "kind": hit.kind,
        "symbol": hit.symbol,
        "action": hit.action,
        "outcome": hit.outcome,
    }
    for key, value in filters.items():
        if not value or key not in values:
            continue
        if str(values[key] or "").upper() != str(value).upper():
            return False
    return True


def _clean_scalar_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = _CONTROL_TEXT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _clean_scalar_text(key, limit=120): _safe_metadata(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_metadata(item) for item in value[:50]]
    if isinstance(value, tuple):
        return [_safe_metadata(item) for item in value[:50]]
    if isinstance(value, str):
        return _clean_scalar_text(value, limit=1000)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _clean_scalar_text(value, limit=1000)


def _zvec_safe_doc_id(doc_id: str) -> str:
    raw = str(doc_id or "doc").strip()
    cleaned = _ZVEC_ID_RE.sub("_", raw).strip("_") or "doc"
    if cleaned == raw and len(cleaned) <= 120:
        return cleaned
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{cleaned[:96]}_{digest}"
