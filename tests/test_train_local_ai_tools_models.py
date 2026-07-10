from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.learning import ShadowBacktest
from scripts import train_local_ai_tools_models as train_script
from scripts.train_local_ai_tools_models import (
    _build_auth_headers,
    _compact_local_ai_tools_features,
    _merge_trade_samples,
    _normalize_base_url,
    _position_settlement_metadata,
    _post_training_payload,
)
from services.phase3_boundary import PHASE3_CLEAN_START_UTC


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    db_path = tmp_path / "local-ai-tools-shadow-training.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()


def test_local_ai_tools_autotrain_task_persists_artifacts() -> None:
    command = Path("scripts/run_local_ai_tools_autotrain.cmd").read_text(encoding="utf-8")

    assert "--persist-artifact" in command
    assert "--confirm-phase3-rebuild" in command


@pytest.mark.asyncio
async def test_local_ai_tools_shadow_loader_uses_clean_compact_feature_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(train_script, "_LOCAL_AI_TOOLS_SHADOW_READ_PAGE_SIZE", 1)
    created_at = PHASE3_CLEAN_START_UTC + timedelta(minutes=5)
    async with get_session_ctx() as session:
        session.add_all(
            [
                ShadowBacktest(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="OLD/USDT",
                    analysis_type="market",
                    decision_action="long",
                    decision_confidence=0.7,
                    feature_snapshot={"current_price": 10.0},
                    status="completed",
                    due_at=PHASE3_CLEAN_START_UTC,
                    long_return_pct=0.2,
                    short_return_pct=-0.2,
                    created_at=PHASE3_CLEAN_START_UTC - timedelta(seconds=1),
                ),
                ShadowBacktest(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    analysis_type="market",
                    decision_action="long",
                    decision_confidence=0.7,
                    feature_snapshot={
                        "current_price": 100.0,
                        "rsi_14": 53.0,
                        "returns_1": 0.01,
                        "unused_llm_context": {"transcript": "x" * 100_000},
                    },
                    raw_llm_response={"unused_full_response": "x" * 100_000},
                    status="completed",
                    due_at=created_at + timedelta(minutes=10),
                    horizon_minutes=10,
                    long_return_pct=0.2,
                    short_return_pct=-0.2,
                    best_action="long",
                    created_at=created_at,
                ),
                ShadowBacktest(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ETH/USDT",
                    analysis_type="market",
                    decision_action="short",
                    decision_confidence=0.6,
                    feature_snapshot={"current_price": 10.0, "rsi_14": 47.0},
                    status="completed",
                    due_at=created_at + timedelta(minutes=11),
                    horizon_minutes=10,
                    long_return_pct=-0.1,
                    short_return_pct=0.1,
                    best_action="short",
                    created_at=created_at + timedelta(minutes=1),
                ),
            ]
        )

    try:
        samples = await train_script._load_shadow_samples(limit=10)
    finally:
        await close_db()

    assert len(samples) == 2
    assert samples[0]["symbol"] == "BTC/USDT"
    assert samples[0]["features"]["current_price"] == 100.0
    assert samples[0]["features"]["rsi_14"] == 53.0
    assert "unused_llm_context" not in samples[0]["features"]
    assert samples[1]["symbol"] == "ETH/USDT"


def test_local_ai_tools_training_headers_use_bearer_token() -> None:
    assert _build_auth_headers("  local-secret-token  ") == {
        "Authorization": "Bearer local-secret-token"
    }
    assert _build_auth_headers("") == {}


def test_local_ai_tools_training_base_url_validation() -> None:
    assert _normalize_base_url(" http://127.0.0.1:8001/ ") == "http://127.0.0.1:8001"

    with pytest.raises(RuntimeError, match="absolute http"):
        _normalize_base_url("127.0.0.1:8001")

    with pytest.raises(RuntimeError, match="credentials"):
        _normalize_base_url("http://user:password@127.0.0.1:8001")

    with pytest.raises(RuntimeError, match="LOCAL_AI_TOOLS_API_BASE is empty"):
        _normalize_base_url("")


def test_local_ai_tools_training_merges_trade_samples_without_duplicate_positions() -> None:
    reflection_samples = [
        {
            "source": "trade_reflection",
            "id": 11,
            "position_id": 7,
            "realized_pnl": -1.2,
            "hold_minutes": 3.0,
        },
        {"source": "trade_reflection", "id": 12, "position_id": 8, "realized_pnl": 2.4},
    ]
    closed_position_samples = [
        {
            "source": "closed_position",
            "id": 7,
            "position_id": 7,
            "realized_pnl": -1.2,
            "raw_llm_response": {
                "profit_first_trade_plan": {
                    "decision_lane": "tiny_probe",
                    "position_size_pct": 0.01,
                }
            },
        },
        {"source": "closed_position", "id": 9, "position_id": 9, "realized_pnl": 0.4},
    ]

    merged = _merge_trade_samples(reflection_samples, closed_position_samples)

    assert [item["source"] for item in merged] == ["closed_position", "trade_reflection", "closed_position"]
    assert [item["position_id"] for item in merged] == [7, 8, 9]
    assert merged[0]["hold_minutes"] == 3.0
    assert merged[0]["raw_llm_response"]["profit_first_trade_plan"]["decision_lane"] == "tiny_probe"


def test_closed_position_training_sample_preserves_settlement_truth_sources() -> None:
    position = SimpleNamespace(
        settlement_source="okx_position_history_settlement",
        settlement_status="reconciled",
        close_fill_pnl=1.2,
        entry_fee=0.1,
        close_fee=0.2,
        funding_fee=-0.03,
        settlement_raw={
            "fee_source": "okx_positions_history.fee",
            "funding_fee_source": "okx_positions_history.fundingFee",
            "official_realized_pnl": 0.87,
            "formula": "realizedPnl = closeFillPnl - fee - fundingFee",
        },
    )

    metadata = _position_settlement_metadata(position)

    assert metadata["pnl_source"] == "okx_position_history_settlement"
    assert metadata["settlement_status"] == "reconciled"
    assert metadata["settlement_source"] == "okx_position_history_settlement"
    assert metadata["close_fill_pnl"] == 1.2
    assert metadata["entry_fee"] == 0.1
    assert metadata["close_fee"] == 0.2
    assert metadata["funding_fee"] == -0.03
    assert metadata["fee_source"] == "okx_positions_history.fee"
    assert metadata["funding_fee_source"] == "okx_positions_history.fundingFee"
    assert metadata["official_realized_pnl"] == 0.87
    assert metadata["settlement_formula"] == "realizedPnl = closeFillPnl - fee - fundingFee"


@pytest.mark.asyncio
async def test_local_ai_tools_training_post_preserves_phase3_training_policy() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"trained": True}, request=request)

    result = await _post_training_payload(
        "https://local-ai-tools.test/",
        {
            "shadow_samples": [],
            "training_mode": "walk_forward",
            "model_stage": "shadow",
            "evaluation_policy": {
                "promotion_flow": "shadow_to_canary_to_live",
                "live_mutation": False,
                "requires_walk_forward": False,
            },
        },
        request_timeout=3.0,
        transport=httpx.MockTransport(handler),
    )

    assert result == {"trained": True}
    assert captured_payload["training_mode"] == "walk_forward"
    assert captured_payload["model_stage"] == "shadow"
    assert captured_payload["evaluation_policy"] == {
        "promotion_flow": "shadow_to_canary_to_live",
        "live_mutation": False,
        "requires_walk_forward": False,
    }


@pytest.mark.asyncio
async def test_local_ai_tools_training_post_preserves_promotion_recommendation() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"trained": True}, request=request)

    promotion = {
        "policy": "phase3_shadow_to_canary_to_live",
        "recommended_stage": "canary",
        "canary_ready": True,
        "live_ready": False,
    }
    result = await _post_training_payload(
        "https://local-ai-tools.test/",
        {"shadow_samples": [], "promotion_recommendation": promotion},
        request_timeout=3.0,
        transport=httpx.MockTransport(handler),
    )

    assert result == {"trained": True}
    assert captured_payload["promotion_recommendation"] == promotion


def test_local_ai_tools_shadow_features_are_compacted_for_training_payload() -> None:
    features = {
        "symbol": "BTC/USDT",
        "current_price": "100.5",
        "rsi_14": 55,
        "returns_1": 0.12,
        "volume_ratio": 1.8,
        "decision_confidence": 0.73,
        "horizon_minutes": 10,
        "close_sequence": list(range(120)),
        "recent_headlines": ["x" * 300 for _ in range(20)],
        "local_ai_tools_shadow": {
            "status": "completed",
            "time_series_prediction": {
                "model": "local-timeseries-ensemble-v1",
                "expected_return_pct": 0.1,
                "timesfm_shadow_expected_return_pct": 0.42,
                "timesfm_shadow_side": "long",
                "timesfm_shadow_horizon_step": 2,
                "chronos_shadow_expected_return_pct": 0.21,
                "chronos_shadow_side": "long",
                "chronos_shadow_horizon_step": 2,
                "specialist_inference_active": True,
                "professional_model_shadow": {
                    "kind": "timeseries",
                    "primary_model": "google/timesfm-2.5-200m-pytorch",
                    "challenger_model": "amazon/chronos-2",
                    "artifacts_ready": True,
                    "actual_inference": True,
                    "baseline_response": True,
                    "live_mutation": False,
                    "shadow_result": {
                        "model": "timesfm-2.5-shadow-challenger",
                        "actual_inference": True,
                        "expected_return_pct": 0.42,
                        "best_side": "long",
                        "confidence": 0.73,
                        "sequence_length": 60,
                        "raw_predictions": list(range(100)),
                    },
                    "primary_shadow_result": {
                        "model": "timesfm-2.5-primary",
                        "actual_inference": True,
                        "expected_return_pct": 0.42,
                        "best_side": "long",
                        "confidence": 0.73,
                        "sequence_length": 60,
                        "raw_predictions": list(range(100)),
                    },
                    "challenger_shadow_result": {
                        "model": "chronos-2-shadow-challenger",
                        "actual_inference": True,
                        "expected_return_pct": 0.21,
                        "best_side": "long",
                        "confidence": 0.62,
                        "sequence_length": 60,
                        "raw_predictions": list(range(100)),
                    },
                },
                "raw_huge_payload": list(range(1000)),
            },
        },
        "raw_llm_response": {"huge": ["unused"] * 1000},
        "opinions": [{"unused": True}],
        "nested_context": {"unused": True},
    }

    compact = _compact_local_ai_tools_features(features)

    assert compact["symbol"] == "BTC/USDT"
    assert compact["current_price"] == 100.5
    assert compact["rsi_14"] == 55.0
    assert compact["close_sequence"] == list(range(40, 120))
    assert len(compact["recent_headlines"]) == 12
    assert all(len(text) == 220 for text in compact["recent_headlines"])
    shadow = compact["local_ai_tools_shadow"]["time_series_prediction"]
    assert shadow["timesfm_shadow_expected_return_pct"] == 0.42
    assert shadow["timesfm_shadow_side"] == "long"
    assert shadow["chronos_shadow_expected_return_pct"] == 0.21
    assert shadow["chronos_shadow_side"] == "long"
    assert shadow["professional_model_shadow"]["primary_model"] == (
        "google/timesfm-2.5-200m-pytorch"
    )
    assert shadow["professional_model_shadow"]["challenger_model"] == "amazon/chronos-2"
    assert shadow["professional_model_shadow"]["shadow_result"] == {
        "model": "timesfm-2.5-shadow-challenger",
        "actual_inference": True,
        "expected_return_pct": 0.42,
        "best_side": "long",
        "confidence": 0.73,
        "sequence_length": 60.0,
    }
    assert shadow["professional_model_shadow"]["primary_shadow_result"] == {
        "model": "timesfm-2.5-primary",
        "actual_inference": True,
        "expected_return_pct": 0.42,
        "best_side": "long",
        "confidence": 0.73,
        "sequence_length": 60.0,
    }
    assert shadow["professional_model_shadow"]["challenger_shadow_result"] == {
        "model": "chronos-2-shadow-challenger",
        "actual_inference": True,
        "expected_return_pct": 0.21,
        "best_side": "long",
        "confidence": 0.62,
        "sequence_length": 60.0,
    }
    assert "raw_huge_payload" not in shadow
    assert "raw_predictions" not in shadow["professional_model_shadow"]["shadow_result"]
    assert "raw_predictions" not in shadow["professional_model_shadow"]["primary_shadow_result"]
    assert "raw_predictions" not in shadow["professional_model_shadow"]["challenger_shadow_result"]
    assert "raw_llm_response" not in compact
    assert "opinions" not in compact
    assert "nested_context" not in compact


@pytest.mark.asyncio
async def test_local_ai_tools_training_post_sends_auth_header() -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"trained": True}, request=request)

    result = await _post_training_payload(
        "https://local-ai-tools.test/",
        {"shadow_samples": []},
        request_timeout=3.0,
        auth_token="test-local-tools-key",
        transport=httpx.MockTransport(handler),
    )

    assert result == {"trained": True}
    assert captured["url"] == "https://local-ai-tools.test/train"
    assert captured["authorization"] == "Bearer test-local-tools-key"


@pytest.mark.asyncio
async def test_local_ai_tools_training_post_preserves_quality_report() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"trained": True}, request=request)

    quality_report = {"data_quality_version": "test", "totals": {"excluded": 2}}
    result = await _post_training_payload(
        "https://local-ai-tools.test/",
        {"shadow_samples": [], "quality_report": quality_report},
        request_timeout=3.0,
        transport=httpx.MockTransport(handler),
    )

    assert result == {"trained": True}
    assert captured_payload["quality_report"] == quality_report


@pytest.mark.asyncio
async def test_local_ai_tools_training_post_preserves_governance_report() -> None:
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json={"trained": True}, request=request)

    governance_report = {
        "cleanup_mode": "quarantine_not_delete",
        "excluded_sample_count": 3,
    }
    result = await _post_training_payload(
        "https://local-ai-tools.test/",
        {"shadow_samples": [], "governance_report": governance_report},
        request_timeout=3.0,
        transport=httpx.MockTransport(handler),
    )

    assert result == {"trained": True}
    assert captured_payload["governance_report"] == governance_report


@pytest.mark.asyncio
async def test_train_local_ai_tools_cli_defaults_to_phase3_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def load_shadow_samples(_limit: int) -> list[dict[str, object]]:
        return [
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "features": {
                    "symbol": "BTC/USDT",
                    "current_price": 100.0,
                    "returns_1": 0.01,
                    "returns_5": 0.02,
                    "returns_20": 0.03,
                },
                "long_return_pct": 0.2,
                "short_return_pct": -0.1,
            }
        ]

    async def empty_samples(_limit: int | None = None) -> list[dict[str, object]]:
        return []

    async def completed_shadow_count() -> int:
        return 1

    async def completed_trade_count() -> int:
        return 0

    async def post_training_payload(
        base_url: str,
        payload: dict[str, object],
        *,
        request_timeout: float,
    ) -> dict[str, object]:
        captured["base_url"] = base_url
        captured["payload"] = payload
        captured["request_timeout"] = request_timeout
        return {"trained": False, "reason": "phase3_preflight_no_artifact_write"}

    async def fail_quarantine(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("default preflight must not quarantine training rows")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_local_ai_tools_models.py",
            "--base-url",
            "http://127.0.0.1:8001",
            "--shadow-limit",
            "1",
            "--trade-limit",
            "1",
            "--sequence-limit",
            "1",
            "--text-limit",
            "1",
        ],
    )
    monkeypatch.setattr(train_script, "_load_shadow_samples", load_shadow_samples)
    monkeypatch.setattr(train_script, "_load_trade_reflection_samples", empty_samples)
    monkeypatch.setattr(train_script, "_load_closed_position_samples", empty_samples)
    monkeypatch.setattr(train_script, "_load_sequence_samples", empty_samples)
    monkeypatch.setattr(train_script, "_load_text_sentiment_samples", empty_samples)
    monkeypatch.setattr(train_script, "_completed_shadow_sample_count", completed_shadow_count)
    monkeypatch.setattr(train_script, "_completed_trade_sample_count", completed_trade_count)
    monkeypatch.setattr(
        train_script,
        "okx_training_refresh_gate",
        lambda: {
            "allowed": True,
            "reason": "okx_daily_reconciliation_allows_training_refresh",
            "can_refresh_training": True,
        },
    )
    monkeypatch.setattr(train_script, "quarantine_dirty_shadow_samples", fail_quarantine)
    monkeypatch.setattr(train_script, "_post_training_payload", post_training_payload)
    monkeypatch.setattr(train_script, "safe_print", lambda *_args, **_kwargs: None)

    await train_script._main()

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert captured["base_url"] == "http://127.0.0.1:8001"
    assert captured["request_timeout"] == 180.0
    assert payload["persist_artifact"] is False
    assert payload["confirm_phase3_rebuild"] is False
    assert payload["okx_daily_reconciliation_gate"]["allowed"] is True
    assert payload["training_quarantine"] == {
        "skipped": True,
        "reason": "phase3_preflight_no_quarantine_writes",
    }
    assert payload["evaluation_policy"]["phase"] == "phase3_model_factory"
    assert payload["evaluation_policy"]["live_mutation"] is False


@pytest.mark.asyncio
async def test_train_local_ai_tools_cli_rejects_unconfirmed_artifact_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_local_ai_tools_models.py", "--persist-artifact"],
    )

    with pytest.raises(SystemExit, match="--persist-artifact requires --confirm-phase3-rebuild"):
        await train_script._main()


@pytest.mark.asyncio
async def test_train_local_ai_tools_cli_blocks_when_okx_gate_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_local_ai_tools_models.py", "--base-url", "http://127.0.0.1:8001"],
    )
    monkeypatch.setattr(
        train_script,
        "okx_training_refresh_gate",
        lambda: {
            "allowed": False,
            "reason": "okx_daily_reconciliation_training_blocked",
            "can_refresh_training": False,
        },
    )

    async def post_training_payload(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("training payload must not be posted when OKX gate blocks")

    monkeypatch.setattr(train_script, "_post_training_payload", post_training_payload)

    with pytest.raises(SystemExit, match="OKX daily reconciliation blocks"):
        await train_script._main()


@pytest.mark.asyncio
async def test_local_ai_tools_training_auth_failure_is_actionable_and_redacted() -> None:
    leaked_token = "abcdefghijklmnopqrstuvwxyz123456"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"detail": f"Authorization: Bearer {leaked_token} is invalid"},
            request=request,
        )

    with pytest.raises(RuntimeError) as exc_info:
        await _post_training_payload(
            "http://127.0.0.1:8001",
            {"shadow_samples": []},
            request_timeout=3.0,
            auth_token=leaked_token,
            transport=httpx.MockTransport(handler),
        )

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "LOCAL_AI_TOOLS_API_KEY" in message
    assert "/data/trade_ai/local_ai_tools.env" in message
    assert leaked_token not in message
    assert "Authorization: ***" in message
