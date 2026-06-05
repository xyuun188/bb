"""Client for optional local AI quant tools hosted beside the local LLM."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from enum import Enum
from decimal import Decimal
from typing import Any

import httpx
import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


class LocalAIToolsClient:
    """Fetch profit, time-series, and sentiment signals without blocking trading."""

    def __init__(self) -> None:
        self._timeout = max(float(settings.local_ai_tools_timeout_seconds or 2.5), 0.2)

    def enabled(self) -> bool:
        return bool(settings.local_ai_tools_enabled and settings.local_ai_tools_api_base)

    async def enrich(self, features: Any, ml_signal: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self.enrich_with_context(features, ml_signal=ml_signal)

    async def enrich_with_context(
        self,
        features: Any,
        ml_signal: dict[str, Any] | None = None,
        open_positions: list[dict[str, Any]] | None = None,
        include_exit_advice: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled():
            return {"enabled": False, "status": "disabled"}

        payload = self._feature_payload(features)
        if isinstance(ml_signal, dict):
            payload["local_ml_signal"] = ml_signal
        if open_positions:
            payload["open_positions"] = open_positions
        payload = self._json_safe(payload)

        started = datetime.now(timezone.utc)
        calls = [
            ("profit_prediction", self._post("/profit/predict", payload)),
            ("time_series_prediction", self._post("/timeseries/deep/predict", payload)),
            ("sentiment_analysis", self._post("/sentiment/deep/analyze", payload)),
        ]
        if include_exit_advice and open_positions:
            calls.append(("exit_advice", self._post("/exit/advise", payload)))
        results = await asyncio.gather(*(call for _, call in calls), return_exceptions=True)
        data: dict[str, Any] = {
            "enabled": True,
            "status": "completed",
            "api_base": settings.local_ai_tools_api_base,
            "started_at": started.isoformat(),
            "duration_sec": round((datetime.now(timezone.utc) - started).total_seconds(), 3),
        }
        errors: dict[str, str] = {}
        for (name, _), item in zip(calls, results):
            if isinstance(item, Exception):
                errors[name] = str(item)[:180]
                data[name] = {"available": False, "error": str(item)[:180]}
            else:
                data[name] = self._normalize_signal(name, item)
        if errors:
            data["status"] = "partial" if len(errors) < len(calls) else "unavailable"
            data["errors"] = errors
        return data

    def _normalize_signal(self, name: str, item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {"value": item}
        normalized = dict(item)
        if name == "time_series_prediction":
            side = str(normalized.get("best_side") or normalized.get("side") or "").lower()
            direction = str(
                normalized.get("direction")
                or normalized.get("forecast_direction")
                or normalized.get("trend")
                or ""
            ).lower()
            if side not in {"long", "short"}:
                if direction == "up":
                    side = "long"
                elif direction == "down":
                    side = "short"
            if side in {"long", "short"}:
                normalized["best_side"] = side
                normalized["side"] = side
            if "expected_return_pct" not in normalized and "expected_move_pct" in normalized:
                normalized["expected_return_pct"] = normalized.get("expected_move_pct")
        elif name == "sentiment_analysis":
            side = str(normalized.get("best_side") or normalized.get("side") or "").lower()
            label = str(normalized.get("label") or normalized.get("sentiment") or "").lower()
            score = self._to_float(normalized.get("score", normalized.get("sentiment_score")), 0.0)
            if side not in {"long", "short"}:
                if label in {"positive", "bullish"} or score > 0:
                    side = "long"
                elif label in {"negative", "bearish"} or score < 0:
                    side = "short"
            if side in {"long", "short"}:
                normalized["best_side"] = side
                normalized["side"] = side
            if "expected_return_pct" not in normalized and "expected_return_from_sentiment_pct" in normalized:
                normalized["expected_return_pct"] = normalized.get("expected_return_from_sentiment_pct")
        elif name == "exit_advice":
            action = str(normalized.get("action") or normalized.get("recommendation") or "hold").lower()
            normalized["action"] = action or "hold"
            reason = str(normalized.get("reason") or normalized.get("note") or "").strip()
            normalized["reason"] = self._humanize_exit_reason(reason, action)
            normalized["action_label"] = self._humanize_exit_action(action)
        return normalized

    def _humanize_exit_action(self, action: str) -> str:
        return {
            "hold": "继续持有",
            "wait": "继续观察",
            "observe": "继续观察",
            "reduce": "减仓",
            "partial_close": "部分平仓",
            "close": "平仓",
            "full_close": "全部平仓",
            "close_long": "平多",
            "close_short": "平空",
            "no_position": "无匹配持仓",
            "reduce_or_close": "减仓或平仓",
            "protect_profit": "保护利润",
            "close_if_ai_agrees": "AI确认后平仓",
            "trail_profit": "移动锁盈",
        }.get(str(action or "").lower(), str(action or "继续观察"))

    def _humanize_exit_reason(self, reason: str, action: str = "") -> str:
        text = str(reason or "").strip()
        normalized = text.lower().strip(" .")
        if not text or normalized in {
            "no trained exit pressure",
            "no exit pressure",
            "no trained close pressure",
        }:
            if str(action or "").lower() in {"hold", "wait", "observe", ""}:
                return "平仓建议模型未识别到明确的主动平仓压力，本轮倾向继续持有。"
            return "平仓建议模型已参与本轮持仓分析。"
        known = {
            "no matching open position was supplied": "本轮没有传入与该币种匹配的当前持仓，平仓建议模型不参与。",
            "this symbol/side has weak realized profile and the open position is losing": "该币种/方向历史实盘表现偏弱，且当前持仓正在亏损，建议减仓或平仓。",
            "profit exists but historical giveback/loss pressure is elevated": "当前已有浮盈，但历史回吐或亏损压力偏高，建议优先保护利润。",
            "loss is expanding beyond the local exit model tolerance": "亏损扩大到本地平仓模型容忍线之外，若 AI 也确认应优先退出。",
            "position is profitable; trail rather than cap upside immediately": "当前持仓盈利且历史盈亏质量尚可，建议移动保护利润，不急于完全限制上行空间。",
        }
        if normalized in known:
            return known[normalized]
        return text

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def status(self) -> dict[str, Any]:
        if not self.enabled():
            return {"available": False, "status": "disabled"}
        try:
            return await self._get("/models/status")
        except Exception as exc:
            return {"available": False, "status": "error", "error": str(exc)[:180]}

    async def train(
        self,
        shadow_samples: list[dict[str, Any]],
        trade_samples: list[dict[str, Any]],
        sequence_samples: list[dict[str, Any]] | None = None,
        text_sentiment_samples: list[dict[str, Any]] | None = None,
        *,
        source: str = "local_trading_system_auto",
    ) -> dict[str, Any]:
        if not self.enabled():
            return {"trained": False, "reason": "disabled"}
        payload = {
            "source": source,
            "shadow_samples": shadow_samples,
            "trade_samples": trade_samples,
            "sequence_samples": sequence_samples or [],
            "text_sentiment_samples": text_sentiment_samples or [],
        }
        payload = self._json_safe(payload)
        return await self._post("/train", payload, timeout=max(self._timeout, 180.0))

    async def _get(self, path: str) -> dict[str, Any]:
        base = str(settings.local_ai_tools_api_base or "").rstrip("/")
        if not base:
            raise RuntimeError("local AI tools API base is empty")
        headers = {}
        if settings.local_ai_tools_api_key:
            headers["Authorization"] = f"Bearer {settings.local_ai_tools_api_key}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{base}{path}", headers=headers)
            response.raise_for_status()
            parsed = response.json()
            return parsed if isinstance(parsed, dict) else {"value": parsed}

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        base = str(settings.local_ai_tools_api_base or "").rstrip("/")
        if not base:
            raise RuntimeError("local AI tools API base is empty")
        headers = {}
        if settings.local_ai_tools_api_key:
            headers["Authorization"] = f"Bearer {settings.local_ai_tools_api_key}"
        async with httpx.AsyncClient(timeout=timeout or self._timeout) as client:
            response = await client.post(f"{base}{path}", json=payload, headers=headers)
            response.raise_for_status()
            parsed = response.json()
            return parsed if isinstance(parsed, dict) else {"value": parsed}

    def _feature_payload(self, features: Any) -> dict[str, Any]:
        if hasattr(features, "to_dict"):
            snapshot = features.to_dict()
            headlines = getattr(features, "recent_headlines", None)
            if headlines:
                snapshot["recent_headlines"] = list(headlines)[:20]
        else:
            snapshot = dict(features or {})
        return {
            "symbol": snapshot.get("symbol") or getattr(features, "symbol", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": snapshot,
        }

    def _json_safe(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
