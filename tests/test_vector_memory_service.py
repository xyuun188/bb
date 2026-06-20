from __future__ import annotations

from datetime import UTC, datetime, timedelta

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


def test_vector_memory_influence_is_explainable_soft_score() -> None:
    influence = _influence_payload(
        [
            {"score": 0.72, "action": "long", "pnl_pct": -0.8},
            {"score": 0.61, "action": "long", "pnl_pct": -0.3},
            {"score": 0.42, "action": "short", "pnl_pct": 0.5},
        ],
        action="long",
    )

    assert influence["level"] == "negative"
    assert influence["score_delta"] < 0
    assert influence["same_action_loss_count"] == 2
    assert influence["is_hard_gate"] is False
    assert "硬拦截" in influence["reason"] or "不作为硬拦截" in influence["reason"]


def test_vector_memory_influence_ignores_missing_pnl_hits() -> None:
    influence = _influence_payload(
        [
            {"score": 0.99, "action": "hold", "pnl_pct": None},
            {"score": 0.88, "action": "long"},
        ],
        action="long",
    )

    assert influence["level"] == "neutral"
    assert influence["score_delta"] == 0.0
    assert influence["loss_count"] == 0
    assert influence["profit_count"] == 0


def test_vector_memory_auto_reindex_due_for_empty_or_stale_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.vector_memory.service import VectorMemoryService

    service = VectorMemoryService()
    monkeypatch.setattr(settings, "vector_memory_auto_reindex_enabled", True)
    monkeypatch.setattr(settings, "vector_memory_auto_reindex_interval_seconds", 1800)

    assert service._auto_reindex_due(0) is True

    service._last_reindex_at = None
    assert service._auto_reindex_due(12) is True

    service._last_reindex_at = datetime.now(UTC)
    assert service._auto_reindex_due(12) is False

    service._last_reindex_at = datetime.now(UTC) - timedelta(seconds=1801)
    assert service._auto_reindex_due(12) is True
