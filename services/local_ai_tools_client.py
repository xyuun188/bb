"""Client for optional local AI quant tools hosted beside the local LLM."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from time import monotonic
from typing import Any

import httpx
import structlog

from config.settings import settings
from core.safe_output import safe_error_text, safe_response_error_text
from core.url_safety import normalize_http_base_url
from services.entry_signal_extraction import (
    enrich_signal_payload,
    payload_side,
    unwrap_tool_payload,
)
from services.model_promotion_policy import (
    build_phase3_promotion_recommendation,
    build_return_objective_report,
    load_latest_paper_observation_report,
)
from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    RETURN_DISTRIBUTION_CONTRACT_VERSION,
    RETURN_DISTRIBUTION_INPUT_VERSION,
    standardized_return_distribution,
    validate_return_distribution_contract,
)

logger = structlog.get_logger(__name__)

_AUTH_FAILURE_STATUS_CODES = {401, 403}
_ERROR_EXCERPT_LIMIT = 700
_MIN_REQUEST_TIMEOUT_SECONDS = 0.2
_MAX_REQUEST_TIMEOUT_SECONDS = 15.0
_MAX_CIRCUIT_BREAKER_FAILURES = 20
_MAX_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 3600.0
_STATUS_CACHE_TTL_SECONDS = 8.0
_MAX_TIMESERIES_SEQUENCE_LENGTH = 80
_HTTP_MAX_KEEPALIVE_CONNECTIONS = 4
_HTTP_MAX_CONNECTIONS = 8


class LocalAIToolsClient:
    """Fetch profit, time-series, and sentiment signals without blocking trading."""

    def __init__(self) -> None:
        self._timeout = 2.5
        self._failure_threshold = 3
        self._cooldown_seconds = 45.0
        self._refresh_runtime_settings()
        self._failure_count = 0
        self._circuit_open_until: datetime | None = None
        self._last_failure: str = ""
        self._last_success_at: datetime | None = None
        self._status_cache: tuple[float, dict[str, Any]] | None = None
        self._inference_lock = asyncio.Lock()
        self._http_client: httpx.AsyncClient | None = None
        self._http_client_base: str = ""

    def _request_timeout(self) -> float:
        return min(max(self._timeout, 0.5), _MAX_REQUEST_TIMEOUT_SECONDS)

    @staticmethod
    def _is_soft_timeout_failure(reason: str) -> bool:
        lowered = str(reason or "").lower()
        return (
            "readtimeout" in lowered
            or "timed out" in lowered
            or "timeout" in lowered
            or "超时" in lowered
        )

    @staticmethod
    def _request_error_message(exc: httpx.RequestError) -> str:
        if isinstance(exc, httpx.ReadTimeout):
            return "服务器量化工具读取响应超时"
        if isinstance(exc, httpx.ConnectTimeout):
            return "连接服务器量化工具超时"
        if isinstance(exc, httpx.PoolTimeout):
            return "等待服务器量化工具连接池超时"
        if isinstance(exc, httpx.WriteTimeout):
            return "向服务器量化工具发送请求超时"
        if isinstance(exc, httpx.ConnectError):
            return "无法连接服务器量化工具"
        return f"服务器量化工具请求失败：{safe_error_text(exc)}"

    def _refresh_runtime_settings(self) -> None:
        self._timeout = min(
            max(
                float(settings.local_ai_tools_timeout_seconds or 2.5), _MIN_REQUEST_TIMEOUT_SECONDS
            ),
            _MAX_REQUEST_TIMEOUT_SECONDS,
        )
        self._failure_threshold = min(
            max(int(settings.local_ai_tools_circuit_breaker_failures or 3), 1),
            _MAX_CIRCUIT_BREAKER_FAILURES,
        )
        self._cooldown_seconds = min(
            max(
                float(settings.local_ai_tools_circuit_breaker_cooldown_seconds or 45.0),
                self._timeout,
            ),
            _MAX_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
        )

    def enabled(self) -> bool:
        self._refresh_runtime_settings()
        return bool(settings.local_ai_tools_enabled and settings.local_ai_tools_api_base)

    def service_configured(self) -> bool:
        self._refresh_runtime_settings()
        return bool(settings.local_ai_tools_api_base)

    async def enrich(
        self, features: Any, ml_signal: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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
        circuit_open = self._circuit_open_payload()
        if circuit_open:
            return circuit_open

        payload = self._feature_payload(features)
        if isinstance(ml_signal, dict):
            payload["local_ml_signal"] = ml_signal
        if include_exit_advice and open_positions:
            payload["open_positions"] = self._position_payload(features, open_positions)
        payload = self._json_safe(payload)
        request_timeout = self._request_timeout()

        started = datetime.now(UTC)
        tool_specs = [
            ("profit_prediction", "/profit/predict"),
            ("time_series_prediction", "/timeseries/predict"),
            ("sentiment_analysis", "/sentiment/deep/analyze"),
        ]
        if include_exit_advice and open_positions:
            tool_specs.append(("exit_advice", "/exit/advise"))

        async def call_tool(name: str, path: str) -> dict[str, Any]:
            tool_started = datetime.now(UTC)
            try:
                result = await self._post(path, payload, request_timeout=request_timeout)
            except Exception as exc:
                return {
                    "available": False,
                    "status": "error",
                    "error": safe_error_text(exc, limit=180),
                    "path": path,
                    "duration_sec": round(
                        max((datetime.now(UTC) - tool_started).total_seconds(), 0.0001),
                        4,
                    ),
                }
            item = self._normalize_signal(name, result)
            item.setdefault("available", True)
            item.setdefault("status", "returned")
            item.setdefault("path", path)
            item["duration_sec"] = round(
                max((datetime.now(UTC) - tool_started).total_seconds(), 0.0001),
                4,
            )
            return item

        # Market and position analysis run independently. Without one shared
        # client-side queue, their batches can overlap and overload the quant
        # service. The three market inference routes are independent and run
        # concurrently inside the one admitted batch; optional exit advice stays
        # sequential because it is only needed for position review.
        results: list[dict[str, Any]] = []
        async with self._inference_lock:
            core_specs = tool_specs[:3]
            results.extend(
                await asyncio.gather(
                    *(call_tool(name, path) for name, path in core_specs)
                )
            )
            for name, path in tool_specs[len(core_specs) :]:
                results.append(await call_tool(name, path))
        data: dict[str, Any] = {
            "enabled": True,
            "status": "completed",
            "api_base": self._public_api_base(),
            "started_at": started.isoformat(),
            "duration_sec": round((datetime.now(UTC) - started).total_seconds(), 3),
        }
        errors: dict[str, str] = {}
        for (name, _path), item in zip(tool_specs, results, strict=False):
            if isinstance(item, Exception):
                error = safe_error_text(item, limit=180)
                errors[name] = error
                data[name] = {"available": False, "status": "error", "error": error}
            else:
                if item.get("status") == "error" or item.get("available") is False:
                    errors[name] = safe_error_text(
                        item.get("error"),
                        limit=180,
                        fallback="local AI tools request failed",
                    )
                data[name] = item
        if errors:
            data["status"] = "partial" if len(errors) < len(tool_specs) else "unavailable"
            data["errors"] = errors
        if errors and len(errors) == len(tool_specs):
            error_summary = "; ".join(errors.values())
            self._record_failure(
                error_summary,
                open_circuit=not self._is_soft_timeout_failure(error_summary),
            )
            data.update(self._breaker_fields())
        else:
            self._record_success()
            data.update(self._breaker_fields())
        return data

    def _normalize_signal(self, name: str, item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {"value": item}
        normalized = enrich_signal_payload(name, unwrap_tool_payload(item) or dict(item))
        if name == "profit_prediction":
            normalized = self._attach_return_distribution_contract(normalized)
        elif name == "time_series_prediction":
            normalized = self._attach_return_distribution_contract(normalized)
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
        elif name == "exit_advice":
            reported_action = str(
                normalized.get("action") or normalized.get("recommendation") or "hold"
            ).lower()
            normalized["reported_action"] = reported_action or "hold"
            normalized["action"] = "hold"
            normalized["action_label"] = "继续观察"
            normalized["reason"] = "本地退出模型仅提供观察画像，生产平仓由动态退出契约独占。"
            normalized["production_permission"] = False
            normalized["live_mutation"] = False
        return self._attach_model_metadata(name, normalized)

    def _attach_return_distribution_contract(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(payload)
        input_version = str(normalized.get("return_distribution_input_version") or "")
        raw_inputs = normalized.get("return_distribution_inputs")
        inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
        contracts: dict[str, dict[str, Any]] = {}
        boundary_blockers: list[str] = []
        if input_version != RETURN_DISTRIBUTION_INPUT_VERSION:
            boundary_blockers.append("return_distribution_input_version_mismatch")
        for side in ("long", "short"):
            item = inputs.get(side)
            distribution_input = item if isinstance(item, dict) else {}
            if not distribution_input:
                boundary_blockers.append(f"{side}_return_distribution_input_missing")
                continue
            contract = standardized_return_distribution(
                side=side,
                horizon_minutes=distribution_input.get("horizon_minutes"),
                raw_expected_return_pct=distribution_input.get(
                    "raw_expected_return_pct"
                ),
                median_return_pct=distribution_input.get("median_return_pct"),
                lower_quantile_return_pct=distribution_input.get(
                    "lower_quantile_return_pct"
                ),
                upper_quantile_return_pct=distribution_input.get(
                    "upper_quantile_return_pct"
                ),
                dispersion_pct=distribution_input.get("dispersion_pct"),
                tail_loss_probability=distribution_input.get(
                    "tail_loss_probability"
                ),
                tail_loss_scale_pct=distribution_input.get("tail_loss_scale_pct"),
                distribution_member_count=distribution_input.get(
                    "distribution_member_count"
                ),
                return_semantics=str(
                    distribution_input.get("return_semantics") or ""
                ),
                source_authority=str(
                    distribution_input.get("source_authority") or ""
                ),
                objective_version=str(
                    distribution_input.get("objective_version") or ""
                ),
                label_version=str(distribution_input.get("label_version") or ""),
                cost_model_version=str(
                    distribution_input.get("cost_model_version") or ""
                ),
                profit_supervision_version=str(
                    distribution_input.get("profit_supervision_version") or ""
                ),
            )
            if input_version != RETURN_DISTRIBUTION_INPUT_VERSION:
                contract["blockers"] = list(
                    dict.fromkeys(
                        [
                            *(contract.get("blockers") or []),
                            "return_distribution_input_version_mismatch",
                        ]
                    )
                )
                contract["production_eligible"] = False
            contracts[side] = contract

        normalized["return_distribution_contract_version"] = (
            RETURN_DISTRIBUTION_CONTRACT_VERSION
        )
        normalized["return_distribution_contract"] = {
            "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
            **contracts,
        }
        validations = {
            side: validate_return_distribution_contract(
                contract,
                side=side,
                return_semantics="gross_market_opportunity_before_execution",
                profit_supervision_version=PROFIT_SUPERVISION_VERSION,
            )
            for side, contract in contracts.items()
        }
        ready_sides = [
            side
            for side, validation in validations.items()
            if validation.get("eligible") is True
        ]
        reported_side = payload_side(normalized)
        selected_side = reported_side if reported_side in ready_sides else ""
        if not selected_side and ready_sides:
            selected_side = max(
                ready_sides,
                key=lambda side: self._to_float(
                    contracts[side].get("objective_expected_return_pct"),
                    float("-inf"),
                ),
            )
        if selected_side:
            normalized["best_side"] = selected_side
            normalized["side"] = selected_side
            normalized["raw_expected_return_pct"] = contracts[selected_side].get(
                "raw_expected_return_pct"
            )
            normalized["objective_expected_return_pct"] = contracts[
                selected_side
            ].get("objective_expected_return_pct")

        quality = normalized.get("prediction_quality")
        quality = dict(quality) if isinstance(quality, dict) else {}
        remote_ready = bool(
            quality.get("production_eligible") is True
            and quality.get("anomalous") is not True
        )
        contract_ready = len(ready_sides) == 2
        remote_blockers = quality.get("blockers")
        remote_blockers = remote_blockers if isinstance(remote_blockers, list) else []
        all_blockers = [
            *boundary_blockers,
            *remote_blockers,
        ]
        for validation in validations.values():
            all_blockers.extend(validation.get("blockers") or [])
        all_blockers = list(dict.fromkeys(str(item) for item in all_blockers if item))
        production_eligible = bool(remote_ready and contract_ready and not all_blockers)
        normalized["prediction_quality"] = {
            **quality,
            "production_eligible": production_eligible,
            "anomalous": not production_eligible,
            "reason": (
                "standardized_return_distribution_ready"
                if production_eligible
                else all_blockers[0]
                if all_blockers
                else str(quality.get("reason") or "remote_prediction_quality_blocked")
            ),
            "blockers": all_blockers,
        }
        predictions = normalized.get("predictions")
        if isinstance(predictions, list):
            normalized["predictions"] = [
                self._attach_return_distribution_contract(item)
                if isinstance(item, dict)
                and item.get("return_distribution_inputs")
                else item
                for item in predictions
            ]
        return normalized

    def _attach_model_metadata(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "profit_prediction": {
                "primary_model": "profit_v1_baseline",
                "challenger_model": None,
                "model_version": "local_ai_tools.v1",
            },
            "time_series_prediction": {
                "primary_model": "timeseries_v1_baseline",
                "challenger_model": None,
                "model_version": "local_ai_tools.v1",
            },
            "sentiment_analysis": {
                "primary_model": "sentiment_v1_baseline",
                "challenger_model": None,
                "model_version": "local_ai_tools.v1",
            },
            "exit_advice": {
                "primary_model": "exit_profile_observer_v2",
                "challenger_model": None,
                "model_version": "local_ai_tools.v2",
            },
        }.get(name, {})
        for key, value in defaults.items():
            payload.setdefault(key, value)
        payload.setdefault("route_mode", "shadow_observation")
        payload.setdefault("fallback_reason", "")
        payload["feature_coverage"] = self._feature_coverage_payload(payload)
        return payload

    def _feature_coverage_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        raw = payload.get("feature_coverage")
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, (int, float)):
            ratio = min(max(float(raw), 0.0), 1.0)
            return {"ratio": round(ratio, 6), "status": "reported"}
        return {"ratio": None, "status": "not_reported"}

    def _to_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    async def status(self) -> dict[str, Any]:
        enabled_for_trading = self.enabled()
        if not self.service_configured():
            return {
                "available": False,
                "service_available": False,
                "enabled_for_trading": False,
                "status": "not_configured",
            }
        circuit_open = self._circuit_open_payload()
        if circuit_open:
            return {
                "available": False,
                "enabled_for_trading": enabled_for_trading,
                **circuit_open,
            }
        cached = self._read_status_cache()
        if cached is not None:
            return cached
        try:
            status = await self._get(
                "/models/status",
                request_timeout=self._request_timeout(),
            )
            status.setdefault("api_base", self._public_api_base())
            status_ok = True
            status_error = ""
        except Exception as exc:
            status = {"api_base": self._public_api_base()}
            status_ok = False
            status_error = safe_error_text(exc, limit=180)

        try:
            health = await self._get(
                "/health",
                request_timeout=self._request_timeout(),
            )
            health_ok = True
            health_error = ""
        except Exception as exc:
            health = {}
            health_ok = False
            health_error = safe_error_text(exc, limit=180)

        child_endpoints = self._metadata_child_endpoints(status, health)
        child_available = any(
            bool(item.get("available") or item.get("ok"))
            for item in child_endpoints.values()
            if isinstance(item, dict)
        )
        model_bundle_available = bool(status.get("available"))
        service_available = bool(status_ok or health_ok or child_available)

        status["model_bundle_available"] = model_bundle_available
        status["service_available"] = service_available
        status["health_available"] = health_ok
        status["enabled_for_trading"] = enabled_for_trading
        status["health"] = health
        status["child_endpoints"] = child_endpoints
        # Phase 3 separates service reachability from persisted model-bundle readiness.
        # Reachability and persisted artifact readiness are separate states.
        status["available"] = bool(service_available)
        status.setdefault("api_base", self._public_api_base())
        if status_error:
            status["status_error"] = status_error
        if health_error:
            status["health_error"] = health_error
        for key in (
            "trained_at",
            "objective_name",
            "objective_version",
            "label_name",
            "label_version",
            "cost_model_version",
            "training_cost_policy",
            "profit_supervision_version",
            "profit_supervision_report",
            "shadow_sample_count",
            "train_shadow_sample_count",
            "holdout_shadow_sample_count",
            "train_decision_group_count",
            "holdout_decision_group_count",
            "trade_sample_count",
            "sequence_sample_count",
            "text_sentiment_sample_count",
            "completed_shadow_sample_count",
            "completed_trade_sample_count",
            "training_mode",
            "model_stage",
            "route_mode",
            "live_mutation",
            "live_trading_mutation",
            "artifact_persisted",
            "training_data_sha256",
            "source_code_sha256",
            "return_distribution_input_version",
            "promotion_recommendation",
            "governance_report",
            "quality_report",
        ):
            if key not in status and key in health:
                status[key] = health.get(key)
        if "models" not in status:
            if isinstance(health.get("models"), dict):
                status["models"] = health["models"]
            elif isinstance(health.get("model_status"), dict):
                status["models"] = health["model_status"]
        if model_bundle_available and not str(status.get("status") or "").strip():
            status["status"] = "ready"
        if service_available and not model_bundle_available:
            status.setdefault(
                "message",
                "Local AI tools service is available; trained bundle is not ready yet.",
            )
            status["status"] = "artifact_unavailable"
        if service_available and not enabled_for_trading:
            status.setdefault(
                "message",
                "Local AI tools service is online; trading influence is disabled by configuration.",
            )
            status["status"] = "connected_trading_disabled"
        if service_available:
            self._record_success()
            status.update(self._breaker_fields())
            return self._write_status_cache(status)

        error = status_error or "local AI tools service is unavailable"
        self._record_failure(error, open_circuit=False)
        return self._write_status_cache(
            {
                "available": False,
                "status": "error",
                "error": error,
                "api_base": self._public_api_base(),
                "model_bundle_available": False,
                "service_available": False,
                "enabled_for_trading": enabled_for_trading,
                "child_endpoints": child_endpoints,
                **self._breaker_fields(),
            }
        )

    @staticmethod
    def _metadata_child_endpoints(
        status: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Read child readiness without executing expensive inference routes."""

        for source in (status, health):
            raw = source.get("child_endpoints")
            if not isinstance(raw, dict):
                continue
            return {
                str(name): dict(item)
                for name, item in raw.items()
                if str(name).strip() and isinstance(item, dict)
            }
        return {}

    async def train(
        self,
        shadow_samples: list[dict[str, Any]],
        trade_samples: list[dict[str, Any]],
        sequence_samples: list[dict[str, Any]] | None = None,
        text_sentiment_samples: list[dict[str, Any]] | None = None,
        *,
        source: str = "local_trading_system_auto",
        completed_shadow_sample_count: int | None = None,
        completed_trade_sample_count: int | None = None,
        raw_trade_sample_count: int | None = None,
        trainable_trade_sample_count: int | None = None,
        quarantined_trade_sample_count: int | None = None,
        trade_sample_cursor_policy: str = "clean_training_view_only",
        quality_report: dict[str, Any] | None = None,
        governance_report: dict[str, Any] | None = None,
        training_mode: str = "shadow",
        paper_observation_report: dict[str, Any] | None = None,
        promotion_recommendation: dict[str, Any] | None = None,
        persist_artifact: bool = False,
        confirm_phase3_rebuild: bool = False,
    ) -> dict[str, Any]:
        if not self.enabled():
            return {"trained": False, "reason": "disabled"}
        circuit_open = self._circuit_open_payload()
        if circuit_open:
            return {"trained": False, "reason": "circuit_open", **circuit_open}
        effective_paper_observation = (
            paper_observation_report or load_latest_paper_observation_report()
        )
        return_objective_report = build_return_objective_report(
            trade_samples=trade_samples,
            shadow_samples=shadow_samples,
        )
        profit_supervision_report = (
            quality_report.get("profit_supervision", {})
            if isinstance(quality_report, dict)
            else {}
        )
        effective_promotion = promotion_recommendation or build_phase3_promotion_recommendation(
            training_mode=training_mode,
            quality_report=quality_report or {},
            governance_report=governance_report or {},
            paper_observation_report=effective_paper_observation,
            completed_shadow_sample_count=int(completed_shadow_sample_count or 0),
            completed_trade_sample_count=int(completed_trade_sample_count or 0),
            return_objective_report=return_objective_report,
        )
        payload = {
            "source": source,
            "shadow_samples": shadow_samples,
            "trade_samples": trade_samples,
            "sequence_samples": sequence_samples or [],
            "text_sentiment_samples": text_sentiment_samples or [],
            "completed_shadow_sample_count": completed_shadow_sample_count,
            "completed_trade_sample_count": completed_trade_sample_count,
            "raw_trade_sample_count": raw_trade_sample_count,
            "trainable_trade_sample_count": trainable_trade_sample_count,
            "quarantined_trade_sample_count": quarantined_trade_sample_count,
            "trade_sample_cursor_policy": trade_sample_cursor_policy,
            "quality_report": quality_report or {},
            "governance_report": governance_report or {},
            "training_mode": training_mode,
            "return_objective_report": return_objective_report,
            "profit_supervision_report": profit_supervision_report,
            "promotion_recommendation": effective_promotion,
            "persist_artifact": bool(persist_artifact),
            "confirm_phase3_rebuild": bool(confirm_phase3_rebuild),
        }
        payload = self._json_safe(payload)
        try:
            result = await self._post(
                "/train",
                payload,
                request_timeout=max(self._timeout, 180.0),
            )
        except Exception as exc:
            error = safe_error_text(exc, limit=180)
            self._record_failure(error)
            logger.warning(
                "local AI tools training request failed",
                reason="request_failed",
                error=error,
                shadow_sample_count=len(shadow_samples),
                trade_sample_count=len(trade_samples),
                quality_report=quality_report or {},
            )
            return {
                "trained": False,
                "reason": "request_failed",
                "error": error,
                **self._breaker_fields(),
            }
        self._record_success()
        return result

    def _circuit_open_payload(self) -> dict[str, Any] | None:
        now = datetime.now(UTC)
        if self._circuit_open_until is None:
            return None
        if now >= self._circuit_open_until:
            self._circuit_open_until = None
            return None
        remaining = max((self._circuit_open_until - now).total_seconds(), 0.0)
        return {
            "enabled": True,
            "available": False,
            "status": "circuit_open",
            "api_base": self._public_api_base(),
            "failure_count": self._failure_count,
            "cooldown_remaining_sec": round(remaining, 1),
            "circuit_open_until": self._circuit_open_until.isoformat(),
            "error": self._last_failure[:180],
        }

    def _breaker_fields(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "failure_count": self._failure_count,
            "circuit_breaker_threshold": self._failure_threshold,
        }
        if self._last_success_at is not None:
            data["last_success_at"] = self._last_success_at.isoformat()
        if self._circuit_open_until is not None:
            data["circuit_open_until"] = self._circuit_open_until.isoformat()
        return data

    def _read_status_cache(self) -> dict[str, Any] | None:
        if self._status_cache is None:
            return None
        cached_at, payload = self._status_cache
        age = monotonic() - cached_at
        if age > _STATUS_CACHE_TTL_SECONDS:
            self._status_cache = None
            return None
        data = copy.deepcopy(payload)
        data["status_cache"] = {
            "hit": True,
            "age_seconds": round(max(age, 0.0), 2),
            "ttl_seconds": _STATUS_CACHE_TTL_SECONDS,
        }
        return data

    def _write_status_cache(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = copy.deepcopy(payload)
        data["status_cache"] = {
            "hit": False,
            "age_seconds": 0.0,
            "ttl_seconds": _STATUS_CACHE_TTL_SECONDS,
        }
        self._status_cache = (monotonic(), copy.deepcopy(data))
        return data

    def _record_success(self) -> None:
        self._failure_count = 0
        self._circuit_open_until = None
        self._last_failure = ""
        self._last_success_at = datetime.now(UTC)

    def _record_failure(self, reason: str, *, open_circuit: bool = True) -> None:
        self._failure_count += 1
        self._last_failure = safe_error_text(
            reason,
            limit=180,
            fallback="local AI tools request failed",
        )
        if not open_circuit or self._failure_count < self._failure_threshold:
            return
        self._circuit_open_until = datetime.now(UTC) + timedelta(seconds=self._cooldown_seconds)
        logger.warning(
            "local AI tools circuit breaker opened",
            failure_count=self._failure_count,
            threshold=self._failure_threshold,
            cooldown_seconds=self._cooldown_seconds,
            reason=self._last_failure,
        )

    async def _get(self, path: str, request_timeout: float | None = None) -> dict[str, Any]:
        base = self._api_base()
        try:
            client = await self._shared_http_client(base)
            response = await client.get(
                f"{base}{path}",
                headers=self._auth_headers(),
                timeout=request_timeout or self._timeout,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(self._request_error_message(exc)) from exc
        return self._parse_response(response, path)

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        base = self._api_base()
        timeout_seconds = request_timeout or self._timeout
        timeout: float | httpx.Timeout = timeout_seconds
        if path == "/train":
            # The scheduler/lease owns the overall training deadline. Large clean
            # datasets must not fail while the request body is still streaming.
            timeout = httpx.Timeout(
                connect=timeout_seconds,
                read=None,
                write=None,
                pool=timeout_seconds,
            )
        try:
            client = await self._shared_http_client(base)
            response = await client.post(
                f"{base}{path}",
                json=payload,
                headers=self._auth_headers(),
                timeout=timeout,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(self._request_error_message(exc)) from exc
        return self._parse_response(response, path)

    def _api_base(self) -> str:
        try:
            return normalize_http_base_url(
                settings.local_ai_tools_api_base,
                field_name="local AI tools API base",
            )
        except ValueError as exc:
            raise RuntimeError(safe_error_text(exc)) from exc

    def _public_api_base(self) -> str:
        try:
            return self._api_base()
        except RuntimeError:
            return "invalid_config"

    def _auth_headers(self) -> dict[str, str]:
        key = str(settings.local_ai_tools_api_key or "").strip()
        headers: dict[str, str] = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def _shared_http_client(self, base: str) -> httpx.AsyncClient:
        if self._http_client is not None and self._http_client_base != base:
            await self._http_client.aclose()
            self._http_client = None
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=_HTTP_MAX_KEEPALIVE_CONNECTIONS,
                    max_connections=_HTTP_MAX_CONNECTIONS,
                    keepalive_expiry=30.0,
                )
            )
            self._http_client_base = base
        return self._http_client

    async def close(self) -> None:
        client = self._http_client
        self._http_client = None
        self._http_client_base = ""
        if client is not None:
            await client.aclose()

    def _parse_response(self, response: httpx.Response, path: str) -> dict[str, Any]:
        if not response.is_success:
            detail = self._response_error_excerpt(response)
            if response.status_code in _AUTH_FAILURE_STATUS_CODES:
                message = (
                    f"local AI tools request {path} was rejected with HTTP "
                    f"{response.status_code}; check LOCAL_AI_TOOLS_API_KEY on both sides"
                )
            else:
                message = f"local AI tools request {path} failed with HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise RuntimeError(message)
        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError(f"local AI tools request {path} returned invalid JSON") from exc
        return dict(parsed) if isinstance(parsed, Mapping) else {"value": parsed}

    def _response_error_excerpt(self, response: httpx.Response) -> str:
        return safe_response_error_text(response, limit=_ERROR_EXCERPT_LIMIT)

    def _feature_payload(self, features: Any) -> dict[str, Any]:
        if hasattr(features, "to_dict"):
            snapshot = features.to_dict()
            headlines = getattr(features, "recent_headlines", None)
            if headlines:
                snapshot["recent_headlines"] = list(headlines)[:20]
        else:
            snapshot = dict(features or {})
        for key in ("close_sequence", "volume_sequence", "recent_closes", "recent_volumes"):
            if key in snapshot:
                snapshot[key] = self._compact_numeric_sequence(
                    snapshot.get(key),
                    limit=_MAX_TIMESERIES_SEQUENCE_LENGTH,
                )
        if snapshot.get("close_sequence"):
            snapshot["sequence_length"] = len(snapshot["close_sequence"])
        return {
            "symbol": snapshot.get("symbol") or getattr(features, "symbol", ""),
            "timestamp": datetime.now(UTC).isoformat(),
            "features": snapshot,
        }

    def _compact_numeric_sequence(self, value: Any, *, limit: int) -> list[float]:
        if not isinstance(value, (list, tuple)):
            return []
        values: list[float] = []
        for item in value[-limit:]:
            number = self._to_float(item, default=float("nan"))
            if number == number and abs(number) != float("inf"):
                values.append(float(number))
        return values

    def _position_payload(
        self,
        features: Any,
        open_positions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        symbol = str(getattr(features, "symbol", "") or "").upper().replace("/", "-")
        compact: list[dict[str, Any]] = []
        for pos in open_positions:
            if not isinstance(pos, dict):
                continue
            pos_symbol = str(pos.get("symbol") or "").upper().replace("/", "-")
            if symbol and pos_symbol and pos_symbol != symbol:
                continue
            compact.append(
                {
                    "symbol": pos.get("symbol"),
                    "side": pos.get("side"),
                    "entry_price": pos.get("entry_price"),
                    "current_price": pos.get("current_price"),
                    "quantity": pos.get("quantity") or pos.get("contracts"),
                    "leverage": pos.get("leverage"),
                    "notional": pos.get("notional"),
                    "margin": pos.get("margin") or pos.get("initial_margin"),
                    "unrealized_pnl": pos.get("unrealized_pnl"),
                    "unrealized_pnl_pct": pos.get("unrealized_pnl_pct") or pos.get("pnl_pct"),
                    "created_at": pos.get("created_at"),
                }
            )
        return compact[:4]

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
