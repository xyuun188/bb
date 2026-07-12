"""Shadow backtest lifecycle and memory generation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from config.settings import FIXED_AI_MODEL_SLOTS, settings
from core.safe_output import safe_error_text
from db.repositories.memory_repo import MemoryRepository
from db.session import get_session_ctx
from services.runtime_entry_filters import default_entry_filters
from services.shadow_training_quarantine import quarantine_completed_shadow_row

logger = structlog.get_logger(__name__)

SHADOW_BACKTEST_HORIZONS_MINUTES = (10, 30, 60)
SHADOW_MISSED_OPPORTUNITY_THRESHOLD = 0.004

LatestPriceProvider = Callable[[str], Awaitable[float]]
SymbolNormalizer = Callable[[str | None], str]
FloatParser = Callable[[Any, float], float]
SessionFactory = Callable[[], Any]
RepositoryFactory = Callable[[Any], Any]

_SHADOW_TOOL_NAMES = (
    "profit_prediction",
    "time_series_prediction",
    "sentiment_analysis",
    "exit_advice",
)
_SHADOW_TOOL_KEYS = (
    "available",
    "status",
    "model",
    "primary_model",
    "challenger_model",
    "model_version",
    "route_mode",
    "fallback_reason",
    "best_side",
    "side",
    "direction",
    "expected_return_pct",
    "expected_move_pct",
    "adjusted_expected_return_pct",
    "loss_probability",
    "profit_quality_score",
    "confidence",
    "specialist_inference_active",
    "specialist_primary_model",
    "specialist_challenger_model",
    "timesfm_shadow_expected_return_pct",
    "timesfm_shadow_expected_move_pct",
    "timesfm_shadow_side",
    "timesfm_shadow_confidence",
    "chronos_shadow_expected_return_pct",
    "chronos_shadow_expected_move_pct",
    "chronos_shadow_side",
    "chronos_shadow_confidence",
)
_SHADOW_PROFESSIONAL_KEYS = (
    "kind",
    "primary_model",
    "challenger_model",
    "artifacts_ready",
    "actual_inference",
    "baseline_response",
    "activation_blocker",
    "promotion_flow",
    "live_mutation",
)


def side_label(side: str) -> str:
    side_value = str(side).lower()
    if side_value == "long":
        return "做多"
    if side_value == "short":
        return "做空"
    return str(side)


def _safe_shadow_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _compact_shadow_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        number = _safe_shadow_number(value)
        return round(number, 8) if number is not None else None
    if isinstance(value, str):
        return value.strip()[:160]
    return None


def _compact_professional_shadow(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {}
    for key in _SHADOW_PROFESSIONAL_KEYS:
        if key not in value:
            continue
        item = _compact_shadow_value(value.get(key))
        if item is not None:
            compact[key] = item
    def compact_result_payload(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        compact_result = {}
        for key in (
            "model",
            "available",
            "actual_inference",
            "reason",
            "expected_return_pct",
            "expected_move_pct",
            "best_side",
            "direction",
            "confidence",
            "horizon_step",
            "sequence_length",
            "prediction_count",
        ):
            item = _compact_shadow_value(result.get(key))
            if item is not None:
                compact_result[key] = item
        return compact_result

    shadow_result = compact_result_payload(value.get("shadow_result"))
    if shadow_result:
        compact["shadow_result"] = shadow_result
    primary_shadow_result = compact_result_payload(value.get("primary_shadow_result"))
    if primary_shadow_result:
        compact["primary_shadow_result"] = primary_shadow_result
    challenger_shadow_result = compact_result_payload(value.get("challenger_shadow_result"))
    if challenger_shadow_result:
        compact["challenger_shadow_result"] = challenger_shadow_result
    predictions = value.get("predictions")
    if isinstance(predictions, dict):
        compact_predictions = {}
        for slot, prediction in list(predictions.items())[:4]:
            if not isinstance(prediction, dict):
                continue
            compact_prediction = {}
            for key in ("available", "reason", "score", "label", "text_count"):
                item = _compact_shadow_value(prediction.get(key))
                if item is not None:
                    compact_prediction[key] = item
            if compact_prediction:
                compact_predictions[str(slot)[:80]] = compact_prediction
        if compact_predictions:
            compact["predictions"] = compact_predictions
    return compact


def compact_local_ai_tools_shadow(local_ai_tools_context: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only auditable shadow evidence needed for later walk-forward scoring."""

    if not isinstance(local_ai_tools_context, dict):
        return {}
    compact: dict[str, Any] = {
        "status": str(local_ai_tools_context.get("status") or "")[:60],
        "captured_at": datetime.now(UTC).isoformat(),
    }
    for tool_name in _SHADOW_TOOL_NAMES:
        tool = local_ai_tools_context.get(tool_name)
        if not isinstance(tool, dict):
            continue
        item = {}
        for key in _SHADOW_TOOL_KEYS:
            if key not in tool:
                continue
            value = _compact_shadow_value(tool.get(key))
            if value is not None:
                item[key] = value
        professional = _compact_professional_shadow(tool.get("professional_model_shadow"))
        if professional:
            item["professional_model_shadow"] = professional
        if item:
            compact[tool_name] = item
    return compact if any(key in compact for key in _SHADOW_TOOL_NAMES) else {}


@dataclass(slots=True)
class ShadowBacktestService:
    """Record delayed market outcomes and convert strong results into memory."""

    latest_price_provider: LatestPriceProvider
    symbol_normalizer: SymbolNormalizer
    float_parser: FloatParser
    session_factory: SessionFactory = get_session_ctx
    repository_factory: RepositoryFactory = MemoryRepository
    horizons_minutes: tuple[int, ...] = SHADOW_BACKTEST_HORIZONS_MINUTES
    missed_opportunity_threshold: float = SHADOW_MISSED_OPPORTUNITY_THRESHOLD
    fixed_model_slots: list[dict[str, Any]] = field(
        default_factory=lambda: list(FIXED_AI_MODEL_SLOTS)
    )

    async def create(
        self,
        decision_id: int | None,
        decision: DecisionOutput,
        feature_vector: Any,
        execution_mode: str,
        analysis_type: str = "market",
        local_ai_tools_context: dict[str, Any] | None = None,
    ) -> None:
        """Record pending shadow samples for market-analysis decisions."""
        if analysis_type != "market":
            return
        entry_price = self.float_parser(
            getattr(feature_vector, "current_price", 0.0)
            or getattr(feature_vector, "close", 0.0)
            or (decision.feature_snapshot or {}).get("current_price"),
            0.0,
        )
        if entry_price <= 0:
            return

        now = datetime.now(UTC)
        try:
            async with self.session_factory() as session:
                repo = self.repository_factory(session)
                feature_snapshot = (
                    decision.feature_snapshot or getattr(feature_vector, "to_dict", lambda: {})()
                )
                if not isinstance(feature_snapshot, dict):
                    feature_snapshot = {}
                else:
                    feature_snapshot = dict(feature_snapshot)
                local_ai_shadow = compact_local_ai_tools_shadow(local_ai_tools_context)
                if local_ai_shadow:
                    feature_snapshot["local_ai_tools_shadow"] = local_ai_shadow
                for horizon in self.horizons_minutes:
                    await repo.create_shadow_backtest(
                        {
                            "decision_id": decision_id,
                            "model_name": decision.model_name,
                            "execution_mode": execution_mode,
                            "symbol": decision.symbol,
                            "analysis_type": analysis_type,
                            "decision_action": decision.action.value,
                            "decision_confidence": float(decision.confidence or 0.0),
                            "entry_price": entry_price,
                            "feature_snapshot": feature_snapshot,
                            "raw_llm_response": (
                                decision.raw_response
                                if isinstance(decision.raw_response, dict)
                                else {}
                            ),
                            "status": "pending",
                            "due_at": now + timedelta(minutes=int(horizon)),
                            "horizon_minutes": int(horizon),
                        }
                    )
        except Exception as exc:
            logger.debug(
                "failed to create shadow backtests",
                symbol=decision.symbol,
                error=safe_error_text(exc),
            )

    async def update_due(self, limit: int = 200) -> int:
        """Complete due samples without holding a database session during OKX reads."""
        try:
            async with self.session_factory() as session:
                repo = self.repository_factory(session)
                rows = await repo.get_due_shadow_backtests(limit=max(1, int(limit or 1)))
            if not rows:
                return 0

            # Price collection can wait on an exchange request.  Keep it outside the
            # ORM context so low-priority shadow maintenance cannot exhaust the pool.
            price_cache: dict[str, float] = {}
            completions: dict[int, dict[str, Any]] = {}
            for row in rows:
                row_id = int(getattr(row, "id", 0) or 0)
                if row_id <= 0:
                    continue
                symbol = self.symbol_normalizer(row.symbol) or row.symbol
                if symbol not in price_cache:
                    price_cache[symbol] = await self.latest_price_provider(symbol)
                actual_price = self.float_parser(price_cache.get(symbol), 0.0)
                entry_price = self.float_parser(row.entry_price, 0.0)
                if actual_price <= 0 or entry_price <= 0:
                    continue

                long_return = (actual_price - entry_price) / entry_price
                short_return = (entry_price - actual_price) / entry_price
                threshold = max(
                    float(settings.shadow_memory_min_return_pct or 0.40) / 100.0,
                    self.missed_opportunity_threshold,
                )
                best_action = "hold"
                if long_return >= threshold and long_return >= short_return:
                    best_action = "long"
                elif short_return >= threshold and short_return > long_return:
                    best_action = "short"

                decision_action = str(row.decision_action or "hold")
                missed = decision_action == "hold" and best_action in {"long", "short"}
                completions[row_id] = {
                    "actual_price": actual_price,
                    "long_return": long_return,
                    "short_return": short_return,
                    "best_action": best_action,
                    "missed": missed,
                    "threshold": threshold,
                    "note": self._completion_note(
                        decision_action,
                        best_action,
                        int(row.horizon_minutes),
                        long_return,
                        short_return,
                        missed,
                    ),
                }

            if not completions:
                return 0

            async with self.session_factory() as session:
                repo = self.repository_factory(session)
                reload_rows = getattr(repo, "get_pending_shadow_backtests_by_ids", None)
                if callable(reload_rows):
                    writable_rows = await reload_rows(list(completions))
                else:
                    # Keep isolated test doubles and external repository adapters working.
                    writable_rows = rows
                completed_count = 0
                for row in writable_rows:
                    completion = completions.get(int(getattr(row, "id", 0) or 0))
                    if completion is None:
                        continue
                    await repo.complete_shadow_backtest(
                        row,
                        actual_price=completion["actual_price"],
                        long_return_pct=completion["long_return"] * 100,
                        short_return_pct=completion["short_return"] * 100,
                        best_action=completion["best_action"],
                        missed_opportunity=completion["missed"],
                        note=completion["note"],
                    )
                    completed_count += 1
                    quarantine_result = quarantine_completed_shadow_row(row)
                    if quarantine_result.get("applied"):
                        logger.info(
                            "shadow backtest quarantined from training",
                            shadow_backtest_id=getattr(row, "id", None),
                            symbol=getattr(row, "symbol", None),
                            reasons=quarantine_result.get("reasons"),
                        )
                        continue
                    if settings.shadow_memory_enabled:
                        await self._record_memory_in_session(
                            repo,
                            row,
                            long_return=completion["long_return"],
                            short_return=completion["short_return"],
                            best_action=completion["best_action"],
                            threshold=completion["threshold"],
                        )
            logger.info("shadow backtests updated", count=completed_count)
            return completed_count
        except Exception as exc:
            logger.debug("failed to update shadow backtests", error=safe_error_text(exc))
            return 0

    def _completion_note(
        self,
        decision_action: str,
        best_action: str,
        horizon_minutes: int,
        long_return: float,
        short_return: float,
        missed: bool,
    ) -> str:
        if missed:
            return (
                f"当时观望，但 {horizon_minutes} 分钟后"
                f"{side_label(best_action)}方向收益约"
                f"{max(long_return, short_return) * 100:.2f}%。"
            )
        if (
            decision_action in {"long", "short"}
            and decision_action != best_action
            and best_action != "hold"
        ):
            return f"实际更优方向是 {side_label(best_action)}，用于后续复盘。"
        return ""

    async def _record_memory_in_session(
        self,
        repo: Any,
        row: Any,
        *,
        long_return: float,
        short_return: float,
        best_action: str,
        threshold: float,
    ) -> None:
        """Turn shadow backtest outcomes into small, reusable expert memories."""
        decision_action = str(getattr(row, "decision_action", "") or "hold")
        symbol = str(getattr(row, "symbol", "") or "")
        horizon = int(getattr(row, "horizon_minutes", 0) or 0)
        if not symbol or horizon <= 0:
            return

        if decision_action == "hold" and best_action in {"long", "short"}:
            realized = long_return if best_action == "long" else short_return
            if realized < threshold:
                return
            memory_type = "shadow_missed_opportunity"
            side = best_action
            confidence_adjustment = 0.04
            position_size_multiplier = 1.04
            success_count = 1
            failure_count = 0
            outcome_text = (
                f"当时选择观望，但 {horizon} 分钟后"
                f"{side_label(side)}方向涨跌收益约 {realized * 100:.2f}%。"
            )
            recommended = "allow_small_probe_with_filters"
        elif decision_action in {"long", "short"}:
            realized = long_return if decision_action == "long" else short_return
            side = decision_action
            if realized >= threshold:
                memory_type = "shadow_good_signal"
                confidence_adjustment = 0.025
                position_size_multiplier = 1.02
                success_count = 1
                failure_count = 0
                outcome_text = (
                    f"影子复盘显示：{side_label(side)}信号在 {horizon} 分钟后"
                    f"收益约 {realized * 100:.2f}%，该形态短线有效。"
                )
                recommended = "keep_with_filters"
            elif realized <= -threshold:
                memory_type = "shadow_bad_signal"
                confidence_adjustment = -0.06
                position_size_multiplier = 0.78
                success_count = 0
                failure_count = 1
                opposite = "short" if side == "long" else "long"
                opposite_return = short_return if opposite == "short" else long_return
                outcome_text = (
                    f"影子复盘显示：{side_label(side)}信号在 {horizon} 分钟后"
                    f"亏损约 {abs(realized) * 100:.2f}%，而"
                    f"{side_label(opposite)}方向收益约 {opposite_return * 100:.2f}%。"
                )
                recommended = "reduce_risk"
            else:
                return
        else:
            return

        feature_snapshot = getattr(row, "feature_snapshot", None) or {}
        pattern = self._memory_pattern(feature_snapshot, symbol, side, horizon)
        labels = {slot["name"]: slot.get("label", slot["name"]) for slot in self.fixed_model_slots}
        for expert_name, lesson in self._expert_lessons(
            symbol=symbol,
            side=side,
            memory_type=memory_type,
            outcome_text=outcome_text,
        ).items():
            await repo.upsert_memory(
                {
                    "expert_name": expert_name,
                    "expert_label": labels.get(expert_name, expert_name),
                    "symbol": symbol,
                    "side": side,
                    "memory_type": memory_type,
                    "market_pattern": pattern,
                    "lesson": lesson,
                    "recommended_action": recommended,
                    "confidence_adjustment": confidence_adjustment,
                    "position_size_multiplier": position_size_multiplier,
                    "evidence_count": 1,
                    "success_count": success_count,
                    "failure_count": failure_count,
                    "confidence_score": 0.52,
                    "memory_key": (
                        f"{expert_name}|shadow|{symbol}|{side}|{memory_type}|"
                        f"{horizon}m|{self._feature_bucket(feature_snapshot)}"
                    ),
                    "extra": {
                        "source": "shadow_backtest",
                        "shadow_backtest_id": getattr(row, "id", None),
                        "decision_id": getattr(row, "decision_id", None),
                        "decision_action": decision_action,
                        "best_action": best_action,
                        "horizon_minutes": horizon,
                        "entry_price": getattr(row, "entry_price", None),
                        "actual_price": getattr(row, "actual_price", None),
                        "long_return_pct": long_return * 100,
                        "short_return_pct": short_return * 100,
                    },
                }
            )

    def _expert_lessons(
        self,
        *,
        symbol: str,
        side: str,
        memory_type: str,
        outcome_text: str,
    ) -> dict[str, str]:
        label = side_label(side)
        if memory_type == "shadow_missed_opportunity":
            return {
                "trend_expert": (
                    f"{symbol} {label}机会曾被观望错过。{outcome_text}"
                    "当方向结构、ADX、均线和 MACD 同向时，可以提高方向支持，但不能直接决定仓位。"
                ),
                "momentum_expert": (
                    f"{symbol} {label}机会曾被观望错过。{outcome_text}"
                    "如果预期净收益、手续费覆盖和亏损概率都合格，可以支持小仓位盈利质量试单。"
                ),
                "sentiment_expert": (
                    f"{symbol} {label}机会曾被观望错过。{outcome_text}"
                    "如果 1/5/10/30 分钟路径和事件冲击风险有利，可以支持更早执行。"
                ),
                "risk_expert": (
                    f"{symbol} {label}机会曾被观望错过。{outcome_text}"
                    "没有硬风险时，优先用仓位和杠杆控制风险，不要直接否决交易。"
                ),
            }
        if memory_type == "shadow_good_signal":
            return {
                "trend_expert": (
                    f"{symbol} {label}信号被影子复盘验证有效。{outcome_text}"
                    "下次出现相似方向结构时，可以适当提高方向信心。"
                ),
                "momentum_expert": (
                    f"{symbol} {label}信号被影子复盘验证有效。{outcome_text}"
                    "当扣费后预期净收益和盈亏质量仍为正时，可以支持执行。"
                ),
                "sentiment_expert": (
                    f"{symbol} {label}信号被影子复盘验证有效。{outcome_text}"
                    "短周期路径延续相似时，可以支持当前执行时机。"
                ),
                "risk_expert": (
                    f"{symbol} {label}信号被影子复盘验证有效。{outcome_text}"
                    "没有硬风险时，可以允许小仓位执行。"
                ),
            }
        return {
            "trend_expert": (
                f"{symbol} {label}信号在影子复盘中表现偏弱。{outcome_text}"
                "下次必须先看到趋势延续，再提高方向信心。"
            ),
            "momentum_expert": (
                f"{symbol} {label}信号在影子复盘中表现偏弱。{outcome_text}"
                "追单前要检查预期净收益、手续费覆盖和盈亏比是否过弱。"
            ),
            "sentiment_expert": (
                f"{symbol} {label}信号在影子复盘中表现偏弱。{outcome_text}"
                "执行前要确认短周期路径是否已经反转。"
            ),
            "risk_expert": (
                f"{symbol} {label}信号在影子复盘中表现偏弱。{outcome_text}"
                "相似条件下需要降低仓位/杠杆，必要时阻止新开仓。"
            ),
        }

    def _memory_pattern(
        self,
        feature_snapshot: dict[str, Any],
        symbol: str,
        side: str,
        horizon: int,
    ) -> str:
        return (
            f"{symbol} {side_label(side)}影子复盘 {horizon}分钟，"
            f"ADX={self.float_parser(feature_snapshot.get('adx_14'), 0.0):.1f}，"
            f"量比={self.float_parser(feature_snapshot.get('volume_ratio'), 0.0):.2f}，"
            f"5周期收益={self.float_parser(feature_snapshot.get('returns_5'), 0.0) * 100:.2f}%，"
            f"盘口倾斜={self.float_parser(feature_snapshot.get('orderbook_imbalance'), 0.0):.2f}"
        )

    def _feature_bucket(self, feature_snapshot: dict[str, Any]) -> str:
        adx = self.float_parser(feature_snapshot.get("adx_14"), 0.0)
        volume_ratio = self.float_parser(feature_snapshot.get("volume_ratio"), 0.0)
        returns_5 = self.float_parser(feature_snapshot.get("returns_5"), 0.0)
        imbalance = self.float_parser(feature_snapshot.get("orderbook_imbalance"), 0.0)
        entry_filters = default_entry_filters(reason="shadow_backtest_bucket")
        adx_bucket = (
            "adx_hi"
            if adx >= 25
            else "adx_mid" if adx >= entry_filters.min_entry_adx else "adx_low"
        )
        volume_bucket = (
            "vol_hi"
            if volume_ratio >= 1.2
            else "vol_ok" if volume_ratio >= entry_filters.min_entry_volume_ratio else "vol_low"
        )
        momentum_bucket = (
            "mom_up" if returns_5 > 0.002 else "mom_down" if returns_5 < -0.002 else "mom_flat"
        )
        book_bucket = (
            "bid_wall" if imbalance > 0.12 else "ask_wall" if imbalance < -0.12 else "book_flat"
        )
        return f"{adx_bucket}|{volume_bucket}|{momentum_bucket}|{book_bucket}"
