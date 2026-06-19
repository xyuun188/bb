from __future__ import annotations

from datetime import UTC, datetime

import pytest

from config.settings import settings
from services.vector_memory.embedding import deterministic_text_vector
from services.vector_memory.store import (
    JsonVectorMemoryStore,
    ZvecVectorMemoryStore,
    document_to_fields,
)
from services.vector_memory.types import VectorMemoryDocument


def test_deterministic_text_vector_is_stable_and_normalized() -> None:
    first = deterministic_text_vector("AI16Z/USDT 多单亏损后重复开仓", dimension=32)
    second = deterministic_text_vector("AI16Z/USDT 多单亏损后重复开仓", dimension=32)

    assert first == second
    assert len(first) == 32
    assert any(value != 0 for value in first)


def test_json_vector_memory_store_upserts_and_searches(tmp_path) -> None:
    store = JsonVectorMemoryStore(
        tmp_path / "memory.jsonl",
        dimension=32,
        max_documents=20,
    )
    indexed = store.upsert(
        [
            VectorMemoryDocument(
                id="decision:1",
                kind="decision",
                text="AI16Z/USDT 多单亏损后半小时重复开仓，最终继续亏损。",
                symbol="AI16Z/USDT",
                action="long",
                outcome="loss",
                pnl_pct=-0.42,
                created_at=datetime.now(UTC),
            ),
            VectorMemoryDocument(
                id="news:1",
                kind="news",
                text="Ethereum mainnet upgrade announcement with developer ecosystem impact.",
                symbol="ETH",
            ),
        ]
    )

    hits = store.search("AI16Z/USDT 多单 半小时 重复开仓 继续亏损", top_k=3)

    assert indexed == 2
    assert hits
    assert hits[0].id == "decision:1"
    assert hits[0].pnl_pct == pytest.approx(-0.42)


def test_json_vector_memory_store_filters_by_symbol(tmp_path) -> None:
    store = JsonVectorMemoryStore(
        tmp_path / "memory.jsonl",
        dimension=32,
        max_documents=20,
    )
    store.upsert(
        [
            VectorMemoryDocument(
                id="decision:1", kind="decision", text="BTC 强趋势突破", symbol="BTC/USDT"
            ),
            VectorMemoryDocument(
                id="decision:2", kind="decision", text="ETH 强趋势突破", symbol="ETH/USDT"
            ),
        ]
    )

    hits = store.search("强趋势突破", top_k=5, filters={"symbol": "ETH/USDT"})

    assert [hit.id for hit in hits] == ["decision:2"]


def test_vector_memory_settings_defaults_are_safe() -> None:
    assert settings.vector_memory_enabled is False
    assert settings.vector_memory_backend == "auto"
    assert settings.vector_memory_dimension >= 16


def test_vector_memory_document_fields_strip_invalid_control_text() -> None:
    fields = document_to_fields(
        VectorMemoryDocument(
            id="decision:1",
            kind="decision",
            text="AI16Z\x00 多单\n亏损\t复开",
            symbol="AI16Z/USDT",
            metadata={"reason": "bad\x00json", "items": ["ok", object()]},
        )
    )

    assert "\x00" not in fields["text"]
    assert "\n" not in fields["text"]
    assert "\x00" not in fields["metadata_json"]
    assert "AI16Z" in fields["text"]


def test_zvec_store_accepts_colon_ids_and_cleaned_text(tmp_path) -> None:
    pytest.importorskip("zvec")
    store = ZvecVectorMemoryStore(tmp_path / "zvec", dimension=32, max_documents=20)

    indexed = store.upsert(
        [
            VectorMemoryDocument(
                id="decision:103966",
                kind="decision",
                text="CRCL/USDT\x00 做空 亏损后相似复盘",
                symbol="CRCL/USDT",
                action="short",
            )
        ]
    )
    hits = store.search("CRCL 做空 亏损 复盘", top_k=3)

    assert indexed == 1
    assert hits
    assert hits[0].kind == "decision"


def test_zvec_store_filters_slash_symbols_without_query_expression(tmp_path) -> None:
    pytest.importorskip("zvec")
    store = ZvecVectorMemoryStore(tmp_path / "zvec-filter", dimension=32, max_documents=20)
    store.upsert(
        [
            VectorMemoryDocument(
                id="decision:crcl",
                kind="decision",
                text="CRCL/USDT 做空 弱证据 亏损复盘",
                symbol="CRCL/USDT",
                action="short",
            ),
            VectorMemoryDocument(
                id="decision:btc",
                kind="decision",
                text="BTC/USDT 做多 强趋势 盈利复盘",
                symbol="BTC/USDT",
                action="long",
            ),
        ]
    )

    hits = store.search("弱证据 亏损复盘", top_k=3, filters={"symbol": "CRCL/USDT"})

    assert hits
    assert {hit.symbol for hit in hits} == {"CRCL/USDT"}


def test_zvec_store_writes_large_batches(tmp_path) -> None:
    pytest.importorskip("zvec")
    store = ZvecVectorMemoryStore(tmp_path / "zvec-large", dimension=24, max_documents=2000)

    indexed = store.upsert(
        [
            VectorMemoryDocument(
                id=f"decision:{idx}",
                kind="decision",
                text=f"批量历史样本 {idx} BTC ETH 风险收益",
                symbol="BTC/USDT",
            )
            for idx in range(1100)
        ]
    )
    stats = store.stats()

    assert indexed == 1100
    assert stats["document_count"] >= 1100
