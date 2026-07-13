from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.settings import settings
from services.vector_memory.embedding import deterministic_text_vector
from services.vector_memory.service import _decision_document, _hit_payload, _influence_payload
from services.vector_memory.store import (
    JsonVectorMemoryStore,
    ZvecVectorMemoryStore,
    document_to_fields,
    fields_to_hit,
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


def test_json_vector_memory_store_clear_removes_documents(tmp_path) -> None:
    store = JsonVectorMemoryStore(tmp_path / "memory.jsonl", dimension=32, max_documents=100)
    indexed = store.upsert(
        [
            VectorMemoryDocument(
                id="decision:1",
                kind="decision",
                text="BTC long profitable phase3 sample",
                symbol="BTC/USDT",
                action="long",
                outcome="win",
            ),
            VectorMemoryDocument(
                id="news:1",
                kind="news",
                text="Fresh phase3 market event",
                symbol="BTC",
                action="",
                outcome="",
            ),
        ]
    )

    assert indexed == 2
    assert store.stats()["document_count"] == 2
    assert store.clear() == 2
    assert store.stats()["document_count"] == 0


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
    assert settings.vector_memory_auto_reindex_enabled is True
    assert settings.vector_memory_auto_reindex_interval_seconds >= 300


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


def test_vector_memory_keeps_missing_pnl_as_missing() -> None:
    fields = document_to_fields(
        VectorMemoryDocument(
            id="decision:hold",
            kind="decision",
            text="ETHW/USDT 观望 未成交",
            symbol="ETHW/USDT",
            action="hold",
            pnl_pct=None,
        )
    )

    assert fields["pnl_pct"] is None


def test_vector_memory_decision_document_does_not_invent_unfilled_pnl() -> None:
    class Decision:
        id = 101
        symbol = "ETHW/USDT"
        action = "hold"
        confidence = 0.0
        position_size_pct = 0.0
        is_paper = True
        was_executed = False
        outcome = None
        outcome_pnl_pct = None
        analysis_type = "market"
        created_at = datetime.now(UTC)
        reasoning = "保持观望"
        execution_reason = "未提交订单"

    document = _decision_document(Decision(), {})

    assert document.pnl_pct is None
    assert document.outcome == ""
    assert document.metadata["has_realized_outcome"] is False
    assert document.metadata["pnl_source"] == ""


def test_vector_memory_hit_payload_hides_legacy_unverified_pnl() -> None:
    hit = fields_to_hit(
        "decision:legacy",
        0.99,
        {
            "kind": "decision",
            "text": "ETHW/USDT 观望 未提交订单",
            "symbol": "ETHW/USDT",
            "action": "hold",
            "outcome": "",
            "pnl_pct": -0.08407744,
            "metadata_json": "{}",
        },
    )

    payload = _hit_payload(hit)

    assert payload["pnl_pct"] is None
    assert payload["metadata"] == {}


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


def test_zvec_store_quarantines_invalid_existing_path_before_create(tmp_path, monkeypatch) -> None:
    path = tmp_path / "zvec-invalid"
    path.mkdir()
    (path / "broken").write_text("not a valid collection", encoding="utf-8")

    class FakeOption:
        def __init__(self, read_only=0):
            self.read_only = read_only

    class FakeSchema:
        def __init__(self, *args, **kwargs):
            pass

    class FakeField:
        def __init__(self, *args, **kwargs):
            pass

    class FakeVector:
        def __init__(self, *args, **kwargs):
            pass

    class FakeFlatIndex:
        def __init__(self, *args, **kwargs):
            pass

    class FakeCollection:
        def __init__(self):
            self.stats = type("Stats", (), {"doc_count": 0})()

    class FakeZvec:
        CollectionOption = FakeOption
        CollectionSchema = FakeSchema
        FieldSchema = FakeField
        VectorSchema = FakeVector
        FlatIndexParam = FakeFlatIndex

        class DataType:
            STRING = "string"
            DOUBLE = "double"
            VECTOR_FP32 = "vector"

        class MetricType:
            COSINE = "cosine"

        def open(self, _path, _option):
            raise RuntimeError("path validate failed")

        def create_and_open(self, create_path, _schema):
            assert not Path(create_path).exists()
            Path(create_path).mkdir()
            return FakeCollection()

    monkeypatch.setattr(ZvecVectorMemoryStore, "_load_zvec", staticmethod(lambda: FakeZvec()))

    store = ZvecVectorMemoryStore(path, dimension=24, max_documents=20)
    stats = store.stats()

    assert stats["backend"] == "zvec"
    assert path.exists()
    assert any(item.name.startswith("zvec-invalid.corrupt-") for item in tmp_path.iterdir())


def test_zvec_store_read_only_opens_existing_collection_and_rejects_writes(tmp_path) -> None:
    pytest.importorskip("zvec")
    path = tmp_path / "zvec-readonly"
    writer = ZvecVectorMemoryStore(path, dimension=24, max_documents=20)
    writer.upsert(
        [
            VectorMemoryDocument(
                id="decision:readonly",
                kind="decision",
                text="BTC/USDT 做多 盈利 复盘",
                symbol="BTC/USDT",
                action="long",
            )
        ]
    )

    reader = ZvecVectorMemoryStore(path, dimension=24, max_documents=20, read_only=True)
    hits = reader.search("BTC 做多 盈利", top_k=3)

    assert hits
    with pytest.raises(RuntimeError):
        reader.upsert(
            [
                VectorMemoryDocument(
                    id="decision:blocked-write",
                    kind="decision",
                    text="should fail",
                )
            ]
        )


def test_vector_memory_influence_is_observation_only() -> None:
    influence = _influence_payload(
        [
            {"score": 0.72, "action": "long", "pnl_pct": -0.8},
            {"score": 0.61, "action": "long", "pnl_pct": -0.3},
            {"score": 0.42, "action": "short", "pnl_pct": 0.5},
        ],
        action="long",
    )

    assert influence["level"] == "observation_only"
    assert influence["score_delta"] == 0.0
    assert influence["realized_outcome_count"] == 3
    assert influence["loss_count"] == 2
    assert influence["is_hard_gate"] is False
    assert influence["production_permission"] is False


def test_vector_memory_influence_ignores_missing_pnl_hits() -> None:
    influence = _influence_payload(
        [
            {"score": 0.99, "action": "hold", "pnl_pct": None},
            {"score": 0.88, "action": "long"},
        ],
        action="long",
    )

    assert influence["level"] == "observation_only"
    assert influence["score_delta"] == 0.0
    assert influence["loss_count"] == 0
    assert influence["profit_count"] == 0
    assert influence["realized_outcome_count"] == 0
    assert influence["production_permission"] is False


def test_vector_memory_auto_reindex_due_for_empty_or_stale_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from services.vector_memory.service import VectorMemoryService

    service = VectorMemoryService(data_dir=tmp_path)
    monkeypatch.setattr(settings, "vector_memory_auto_reindex_enabled", True)
    monkeypatch.setattr(settings, "vector_memory_auto_reindex_interval_seconds", 1800)

    assert service._auto_reindex_due(0) is True

    service._write_reset_marker(
        reason="test_phase3_reset",
        reset_at=datetime.now(UTC),
    )
    assert service._auto_reindex_due(0) is False

    service._last_reindex_at = None
    assert service._auto_reindex_due(12) is True

    service._last_reindex_at = datetime.now(UTC)
    assert service._auto_reindex_due(12) is False

    service._last_reindex_at = datetime.now(UTC) - timedelta(seconds=1801)
    assert service._auto_reindex_due(12) is True


@pytest.mark.asyncio
async def test_vector_memory_clear_index_writes_phase3_reset_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from services.vector_memory.service import VectorMemoryService

    monkeypatch.setattr(settings, "vector_memory_backend", "jsonl")
    monkeypatch.setattr(settings, "vector_memory_enabled", True)
    service = VectorMemoryService(data_dir=tmp_path)
    service._get_store().upsert(
        [
            VectorMemoryDocument(
                id="decision:phase3-old",
                kind="decision",
                text="old vector memory sample",
            )
        ]
    )

    result = await service.clear_index(reason="test_clear")
    status = await service.status()

    assert result["status"] == "cleared"
    assert result["removed"] == 1
    assert result["reset_at"]
    assert service._load_reset_at() is not None
    assert status["document_count"] == 0
    assert status["reset_at"] == result["reset_at"]
    assert status["auto_reindex_due"] is False


@pytest.mark.asyncio
async def test_vector_memory_clear_index_cancels_running_auto_reindex(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from services.vector_memory.service import VectorMemoryService

    monkeypatch.setattr(settings, "vector_memory_backend", "jsonl")
    monkeypatch.setattr(settings, "vector_memory_enabled", True)
    monkeypatch.setattr(settings, "vector_memory_auto_reindex_enabled", True)
    service = VectorMemoryService(data_dir=tmp_path)
    started = asyncio.Event()

    async def slow_auto_reindex() -> None:
        started.set()
        await asyncio.sleep(60)

    service._auto_reindex_task = asyncio.create_task(slow_auto_reindex())
    service._auto_reindex_started_at = datetime.now(UTC)
    await started.wait()

    result = await service.clear_index(reason="test_clear_cancels_auto_reindex")
    status = await service.status()

    assert result["status"] == "cleared"
    assert service._auto_reindex_task is None
    assert status["auto_reindex_running"] is False
    assert status["auto_reindex_due"] is False


@pytest.mark.asyncio
async def test_vector_memory_recovers_once_from_zvec_path_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from services.vector_memory.service import VectorMemoryService

    class BrokenStore:
        backend_name = "zvec"

        def stats(self):
            raise RuntimeError("path validate failed: path[/data/bb/app/data/vector_memory/zvec] exists")

        def search(self, *_args, **_kwargs):
            raise RuntimeError("path validate failed: path[/data/bb/app/data/vector_memory/zvec] exists")

    monkeypatch.setattr(settings, "vector_memory_backend", "jsonl")
    monkeypatch.setattr(settings, "vector_memory_enabled", True)
    monkeypatch.setattr(settings, "vector_memory_auto_reindex_enabled", False)
    service = VectorMemoryService(data_dir=tmp_path)
    service._store = BrokenStore()  # type: ignore[assignment]

    status = await service.status()

    assert status["status"] == "ready"
    assert status["backend"] == "jsonl"
    assert status["store_recovered"] is True
    assert status["last_error"] == ""

    service._store = BrokenStore()  # type: ignore[assignment]
    result = await service.search("BTC 做多", top_k=3)

    assert result["status"] == "ok"
    assert result["backend"] == "jsonl"
    assert result["store_recovered"] is True
    assert result["hits"] == []
