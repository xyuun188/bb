"""Local ML profit-quality model built from shadow backtest outcomes.

The model is intentionally used as an observation signal first. It predicts
statistical long/short profit quality from market features, but does not
execute trades by itself.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sqlalchemy import func, select

from core.model_artifact_safety import dump_trusted_joblib, load_trusted_joblib
from core.safe_output import safe_error_text
from db.session import get_read_session_ctx
from models.learning import ShadowBacktest
from services.ml_readiness import build_ml_readiness_report, disabled_ml_readiness
from services.shadow_training_quarantine import quarantine_dirty_shadow_samples
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import (
    annotate_samples,
    assess_shadow_sample,
    governance_report,
    quality_report,
)

logger = structlog.get_logger(__name__)

MODEL_DIR = Path("data/ml_signal")
MODEL_PATH = MODEL_DIR / "winrate_model.joblib"
METADATA_PATH = MODEL_DIR / "winrate_model_metadata.json"
_LOCAL_ML_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
AUTO_TRAIN_CHECK_INTERVAL_SECONDS = _LOCAL_ML_PARAMS.auto_train_check_interval_seconds
AUTO_TRAIN_MIN_INTERVAL_SECONDS = _LOCAL_ML_PARAMS.auto_train_min_interval_seconds
AUTO_TRAIN_MIN_NEW_SAMPLES = _LOCAL_ML_PARAMS.auto_train_min_new_samples
AUTO_TRAIN_LEARNING_ONLY_INTERVAL_SECONDS = (
    _LOCAL_ML_PARAMS.auto_train_learning_only_interval_seconds
)
AUTO_TRAIN_LEARNING_ONLY_MIN_NEW_SAMPLES = _LOCAL_ML_PARAMS.auto_train_learning_only_min_new_samples
TRAINING_SHADOW_SAMPLE_LIMIT = _LOCAL_ML_PARAMS.training_shadow_sample_limit
TRAINING_BALANCED_RECENT_CANDIDATE_SHARE = 0.60
TRAINING_BALANCED_NON_HOLD_CANDIDATE_SHARE = 1.00
TRAINING_BALANCED_BEST_TRADE_CANDIDATE_SHARE = 1.25
TRAINING_MIN_NON_HOLD_SHARE = 0.25
TRAINING_MIN_BEST_TRADE_SHARE = 1.00

FEATURE_KEYS = [
    "change_24h_pct",
    "spread_pct",
    "rsi_14",
    "rsi_7",
    "macd",
    "macd_signal",
    "macd_diff",
    "stoch_k",
    "adx_14",
    "bb_width",
    "bb_pct",
    "atr_pct",
    "volume_ratio",
    "returns_1",
    "returns_5",
    "returns_20",
    "volatility_20",
    "price_vs_sma20",
    "price_vs_sma50",
    "funding_rate",
    "log_volume_24h",
    "log_open_interest_value",
    "orderbook_imbalance",
    "orderbook_depth_ratio",
    "news_sentiment_avg",
    "social_sentiment_avg",
    "social_mention_count",
    "news_article_count",
    "decision_confidence",
    "horizon_minutes",
]

WIN_RETURN_THRESHOLD_PCT = _LOCAL_ML_PARAMS.win_return_threshold_pct
_EXECUTION_COST_PARAMS = DEFAULT_TRADING_PARAMS.execution_cost
ROUND_TRIP_COST_PCT = _EXECUTION_COST_PARAMS.local_ml_round_trip_cost_pct
TAIL_LOSS_THRESHOLD_PCT = _EXECUTION_COST_PARAMS.local_ml_tail_loss_threshold_pct
MIN_PROFIT_EDGE_PCT = _LOCAL_ML_PARAMS.min_profit_edge_pct
MIN_PROFIT_SIGNAL_WIN_RATE = _LOCAL_ML_PARAMS.min_profit_signal_win_rate
MIN_TRAINING_SAMPLES = _LOCAL_ML_PARAMS.min_training_samples
ML_INFLUENCE_MIN_SAMPLE_COUNT = _LOCAL_ML_PARAMS.influence_min_sample_count
ML_INFLUENCE_MIN_TEST_COUNT = _LOCAL_ML_PARAMS.influence_min_test_count
ML_INFLUENCE_MIN_AUC = _LOCAL_ML_PARAMS.influence_min_auc
ML_INFLUENCE_MIN_PR_AUC = _LOCAL_ML_PARAMS.influence_min_pr_auc
ML_INFLUENCE_MIN_ACCURACY = _LOCAL_ML_PARAMS.influence_min_accuracy
READINESS_MAX_DIRTY_SAMPLE_RATIO = _LOCAL_ML_PARAMS.readiness_max_dirty_sample_ratio
READINESS_MAX_MODEL_AGE_SECONDS = _LOCAL_ML_PARAMS.readiness_max_model_age_seconds
ML_INFLUENCE_MIN_TOP_RETURN_PCT = WIN_RETURN_THRESHOLD_PCT


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _feature_row_from_snapshot(
    snapshot: dict[str, Any],
    *,
    decision_confidence: float = 0.0,
    horizon_minutes: int = 10,
) -> dict[str, float]:
    price = _safe_float(snapshot.get("current_price") or snapshot.get("close"), 0.0)
    atr = _safe_float(snapshot.get("atr_14"), 0.0)
    bid_depth = _safe_float(snapshot.get("orderbook_bid_depth"), 0.0)
    ask_depth = _safe_float(snapshot.get("orderbook_ask_depth"), 0.0)
    depth_total = max(bid_depth + ask_depth, 1e-9)
    values = {
        "change_24h_pct": _safe_float(snapshot.get("change_24h_pct")),
        "spread_pct": _safe_float(snapshot.get("spread_pct")),
        "rsi_14": _safe_float(snapshot.get("rsi_14"), 50.0),
        "rsi_7": _safe_float(snapshot.get("rsi_7"), 50.0),
        "macd": _safe_float(snapshot.get("macd")),
        "macd_signal": _safe_float(snapshot.get("macd_signal")),
        "macd_diff": _safe_float(snapshot.get("macd_diff")),
        "stoch_k": _safe_float(snapshot.get("stoch_k"), 50.0),
        "adx_14": _safe_float(snapshot.get("adx_14")),
        "bb_width": _safe_float(snapshot.get("bb_width")),
        "bb_pct": _safe_float(snapshot.get("bb_pct"), 0.5),
        "atr_pct": atr / price if price > 0 else 0.0,
        "volume_ratio": _safe_float(snapshot.get("volume_ratio"), 1.0),
        "returns_1": _safe_float(snapshot.get("returns_1")),
        "returns_5": _safe_float(snapshot.get("returns_5")),
        "returns_20": _safe_float(snapshot.get("returns_20")),
        "volatility_20": _safe_float(snapshot.get("volatility_20")),
        "price_vs_sma20": _safe_float(snapshot.get("price_vs_sma20")),
        "price_vs_sma50": _safe_float(snapshot.get("price_vs_sma50")),
        "funding_rate": _safe_float(snapshot.get("funding_rate")),
        "log_volume_24h": math.log10(max(_safe_float(snapshot.get("volume_24h")), 0.0) + 1.0),
        "log_open_interest_value": math.log10(
            max(_safe_float(snapshot.get("open_interest_value")), 0.0) + 1.0
        ),
        "orderbook_imbalance": _safe_float(snapshot.get("orderbook_imbalance")),
        "orderbook_depth_ratio": (bid_depth - ask_depth) / depth_total,
        "news_sentiment_avg": _safe_float(snapshot.get("news_sentiment_avg")),
        "social_sentiment_avg": _safe_float(snapshot.get("social_sentiment_avg")),
        "social_mention_count": _safe_float(snapshot.get("social_mention_count")),
        "news_article_count": _safe_float(snapshot.get("news_article_count")),
        "decision_confidence": _safe_float(decision_confidence),
        "horizon_minutes": float(horizon_minutes),
    }
    return {key: float(values.get(key, 0.0)) for key in FEATURE_KEYS}


def _feature_row_from_feature_vector(
    features: Any,
    *,
    horizon_minutes: int,
    decision_confidence: float = 0.0,
) -> dict[str, float]:
    snapshot = features.to_dict() if hasattr(features, "to_dict") else dict(features or {})
    return _feature_row_from_snapshot(
        snapshot,
        decision_confidence=decision_confidence,
        horizon_minutes=horizon_minutes,
    )


def _make_classifier(y: pd.Series) -> Pipeline:
    if int(y.nunique()) < 2:
        estimator = DummyClassifier(strategy="prior")
    else:
        estimator = RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def _make_regressor(y: pd.Series) -> Pipeline:
    if int(y.nunique()) < 2:
        estimator = DummyRegressor(strategy="mean")
    else:
        estimator = RandomForestRegressor(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def _positive_proba(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["model"]
    proba = model.predict_proba(x)
    classes = list(getattr(classifier, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(x), dtype=float)


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    try:
        if int(pd.Series(y_true).nunique()) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except (TypeError, ValueError):
        return None


def _safe_pr_auc(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    try:
        if int(pd.Series(y_true).nunique()) < 2:
            return None
        return float(average_precision_score(y_true, y_score))
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(min(float(value), high), low)


def _bucket_return(y_return: pd.Series, scores: np.ndarray, top: bool) -> float | None:
    if len(scores) < 10:
        return None
    count = max(int(len(scores) * 0.20), 1)
    order = np.argsort(scores)
    idx = order[-count:] if top else order[:count]
    return float(pd.Series(y_return).iloc[idx].mean())


def _bucket_win_rate(y_win: pd.Series, scores: np.ndarray, top: bool) -> float | None:
    if len(scores) < 10:
        return None
    count = max(int(len(scores) * 0.20), 1)
    order = np.argsort(scores)
    idx = order[-count:] if top else order[:count]
    return float(pd.Series(y_win).iloc[idx].mean())


def _profit_quality_score(expected_return_pct: float, win_rate: float, edge_pct: float) -> float:
    """Score signal quality by expected PnL first, using win rate only as a sanity check."""
    expected_component = max(expected_return_pct, 0.0)
    edge_component = max(edge_pct, 0.0) * 0.5
    win_penalty = max(0.45 - win_rate, 0.0) * 0.05
    return expected_component + edge_component - win_penalty


def _net_return_pct(raw_return_pct: float) -> float:
    """Approximate executable net return after round-trip fee/slippage costs."""
    return _safe_float(raw_return_pct) - ROUND_TRIP_COST_PCT


def _side_influence_status(metadata: dict[str, Any], side: str) -> dict[str, Any]:
    metrics = _safe_dict(metadata.get("metrics"))
    sample_count = int(metadata.get("sample_count") or 0)
    test_count = int(metadata.get("test_count") or 0)
    auc = _safe_float(metrics.get(f"{side}_auc"), 0.0)
    pr_auc = _safe_float(metrics.get(f"{side}_pr_auc"), None)
    accuracy = _safe_float(metrics.get(f"{side}_accuracy"), 0.0)
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), 0.0)
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), 0.0)
    top_win = _safe_float(metrics.get(f"top_{side}_win_rate"), 0.0)
    bottom_win = _safe_float(metrics.get(f"bottom_{side}_win_rate"), 0.0)

    hard_reasons: list[str] = []
    maturity_reasons: list[str] = []
    if sample_count < ML_INFLUENCE_MIN_SAMPLE_COUNT:
        maturity_reasons.append(f"样本数 {sample_count} < {ML_INFLUENCE_MIN_SAMPLE_COUNT}")
    if test_count < ML_INFLUENCE_MIN_TEST_COUNT:
        maturity_reasons.append(f"测试样本 {test_count} < {ML_INFLUENCE_MIN_TEST_COUNT}")
    if auc < ML_INFLUENCE_MIN_AUC:
        hard_reasons.append(f"AUC {auc:.3f} < {ML_INFLUENCE_MIN_AUC:.2f}")
    if pr_auc is None:
        hard_reasons.append("PR-AUC missing")
    elif pr_auc < ML_INFLUENCE_MIN_PR_AUC:
        hard_reasons.append(f"PR-AUC {pr_auc:.3f} < {ML_INFLUENCE_MIN_PR_AUC:.2f}")
    if accuracy < ML_INFLUENCE_MIN_ACCURACY:
        hard_reasons.append(f"准确率 {accuracy:.3f} < {ML_INFLUENCE_MIN_ACCURACY:.2f}")
    if top_return <= ML_INFLUENCE_MIN_TOP_RETURN_PCT:
        hard_reasons.append(
            f"高分组平均收益 {top_return:.3f}% <= {ML_INFLUENCE_MIN_TOP_RETURN_PCT:.2f}%"
        )
    if top_win <= bottom_win:
        hard_reasons.append(f"高分组胜率 {top_win:.3f} 未优于低分组 {bottom_win:.3f}")

    reliable = not hard_reasons and not maturity_reasons
    advisory = not hard_reasons and sample_count >= MIN_TRAINING_SAMPLES and test_count >= 40
    influence_weight = 1.0 if reliable else 0.35 if advisory else 0.0
    reasons = hard_reasons + maturity_reasons
    status = "active" if reliable else "advisory" if advisory else "learning_only"
    return {
        "enabled": reliable,
        "advisory_enabled": advisory,
        "influence_weight": round(influence_weight, 4),
        "status": status,
        "side": side,
        "auc": round(auc, 4),
        "pr_auc": None if pr_auc is None else round(pr_auc, 4),
        "accuracy": round(accuracy, 4),
        "top_avg_return_pct": round(top_return, 4),
        "bottom_avg_return_pct": round(bottom_return, 4),
        "top_win_rate": round(top_win, 4),
        "bottom_win_rate": round(bottom_win, 4),
        "reasons": reasons,
    }


def _influence_policy(metadata: dict[str, Any]) -> dict[str, Any]:
    long_status = _side_influence_status(metadata, "long")
    short_status = _side_influence_status(metadata, "short")
    enabled = bool(long_status.get("enabled") or short_status.get("enabled"))
    advisory_enabled = bool(
        enabled or long_status.get("advisory_enabled") or short_status.get("advisory_enabled")
    )
    disabled_reasons: list[str] = []
    if not long_status.get("enabled"):
        disabled_reasons.append("做多：" + "；".join(long_status.get("reasons") or ["未达标"]))
    if not short_status.get("enabled"):
        disabled_reasons.append("做空：" + "；".join(short_status.get("reasons") or ["未达标"]))
    return {
        "enabled": enabled,
        "advisory_enabled": advisory_enabled,
        "mode": (
            "entry_profit_filter"
            if enabled
            else "advisory" if advisory_enabled else "learning_only"
        ),
        "status": "active" if enabled else "advisory" if advisory_enabled else "learning_only",
        "long": long_status,
        "short": short_status,
        "disabled_reason": "；".join(disabled_reasons) if disabled_reasons else "",
        "rule": (
            "ML 指标完全达标时按完整权重参与；样本成熟度不足但 AUC/收益分层有效时，"
            "只按小权重参与 expected_net 和证据解释，不作为硬否决；硬指标不达标时继续学习观察。"
        ),
    }


@dataclass(frozen=True)
class ShadowTrainingRow:
    id: int
    created_at: datetime | None
    symbol: str
    analysis_type: str
    decision_action: str
    decision_confidence: float
    feature_snapshot: Any
    due_at: datetime | None
    horizon_minutes: int
    long_return_pct: float | None
    short_return_pct: float | None
    best_action: str | None
    missed_opportunity: bool


def _shadow_training_columns() -> tuple[Any, ...]:
    return (
        ShadowBacktest.id,
        ShadowBacktest.created_at,
        ShadowBacktest.symbol,
        ShadowBacktest.analysis_type,
        ShadowBacktest.decision_action,
        ShadowBacktest.decision_confidence,
        ShadowBacktest.feature_snapshot,
        ShadowBacktest.due_at,
        ShadowBacktest.horizon_minutes,
        ShadowBacktest.long_return_pct,
        ShadowBacktest.short_return_pct,
        ShadowBacktest.best_action,
        ShadowBacktest.missed_opportunity,
    )


def _shadow_training_row_from_mapping(mapping: Any) -> ShadowTrainingRow:
    return ShadowTrainingRow(
        id=int(mapping.get("id") or 0),
        created_at=mapping.get("created_at"),
        symbol=str(mapping.get("symbol") or ""),
        analysis_type=str(mapping.get("analysis_type") or ""),
        decision_action=str(mapping.get("decision_action") or ""),
        decision_confidence=_safe_float(mapping.get("decision_confidence"), 0.0),
        feature_snapshot=mapping.get("feature_snapshot"),
        due_at=mapping.get("due_at"),
        horizon_minutes=int(mapping.get("horizon_minutes") or 10),
        long_return_pct=mapping.get("long_return_pct"),
        short_return_pct=mapping.get("short_return_pct"),
        best_action=mapping.get("best_action"),
        missed_opportunity=bool(mapping.get("missed_opportunity")),
    )


def _shadow_row_id(row: Any) -> Any:
    return getattr(row, "id", id(row))


def _shadow_sort_key(row: Any) -> tuple[datetime, int]:
    created_at = getattr(row, "created_at", None)
    if not isinstance(created_at, datetime):
        created_at = datetime.fromtimestamp(0, UTC)
    elif created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at.astimezone(UTC), int(getattr(row, "id", 0) or 0)


def _shadow_action(row: Any, field: str) -> str:
    return str(getattr(row, field, "") or "").lower().strip()


def _shadow_decision_confidence(row: Any) -> float:
    return _safe_float(getattr(row, "decision_confidence", 0.0), 0.0) or 0.0


def _shadow_is_low_confidence_hold(row: Any) -> bool:
    threshold = DEFAULT_TRADING_PARAMS.training_data_quality.very_low_confidence_threshold
    return _shadow_action(row, "decision_action") == "hold" and (
        _shadow_decision_confidence(row) < threshold
    )


def _shadow_is_trainable_trade_opportunity(row: Any) -> bool:
    if _shadow_action(row, "best_action") not in {"long", "short"}:
        return False
    if _shadow_action(row, "decision_action") not in {"long", "short"}:
        return False
    return not assess_shadow_sample(_shadow_quality_sample(row)).exclude_from_training


def _shadow_quality_sample(row: Any) -> dict[str, Any]:
    return {
        "symbol": getattr(row, "symbol", ""),
        "analysis_type": getattr(row, "analysis_type", ""),
        "decision_action": getattr(row, "decision_action", ""),
        "decision_confidence": _shadow_decision_confidence(row),
        "horizon_minutes": int(getattr(row, "horizon_minutes", 10) or 10),
        "features": _parse_json(getattr(row, "feature_snapshot", None)),
        "long_return_pct": _safe_float(getattr(row, "long_return_pct", None), None),
        "short_return_pct": _safe_float(getattr(row, "short_return_pct", None), None),
        "label_timestamp": getattr(row, "due_at", None),
        "best_action": getattr(row, "best_action", ""),
        "missed_opportunity": bool(getattr(row, "missed_opportunity", False)),
    }


def _shadow_quality_rank(row: Any) -> tuple[int, float, int]:
    """Prefer trainable directional samples before recent low-confidence holds."""

    assessment = assess_shadow_sample(_shadow_quality_sample(row))
    action = _shadow_action(row, "decision_action")
    best_action = _shadow_action(row, "best_action")
    directional = int(action in {"long", "short"})
    best_trade = int(best_action in {"long", "short"})
    missed_trade = int(bool(getattr(row, "missed_opportunity", False)) and best_trade)
    action_score = directional * 4 + best_trade * 3 + missed_trade
    trainable_score = 0 if assessment.exclude_from_training else 10
    return (
        trainable_score + action_score,
        float(assessment.weight),
        int(getattr(row, "id", 0) or 0),
    )


def _sort_shadow_quality_first(rows: list[Any]) -> list[Any]:
    return sorted(
        rows,
        key=lambda row: (_shadow_quality_rank(row), _shadow_sort_key(row)),
        reverse=True,
    )


def select_shadow_training_rows(rows: list[Any], *, limit: int) -> list[Any]:
    """Select a trade-opportunity shadow window for the profit-quality model.

    The local ML artifact is used to judge long/short profit quality. Rows whose
    hindsight ``best_action`` is still hold are useful for audit, but they dilute
    the directional profit labels and keep readiness degraded. Sample-count gates
    should block live influence when there are not enough trade-opportunity rows.
    """

    capped_limit = max(int(limit or TRAINING_SHADOW_SAMPLE_LIMIT), 1)
    deduped: dict[Any, Any] = {}
    for row in rows:
        deduped.setdefault(_shadow_row_id(row), row)
    recent = sorted(deduped.values(), key=_shadow_sort_key, reverse=True)
    trainable_best_trade = [row for row in recent if _shadow_is_trainable_trade_opportunity(row)]
    if len(trainable_best_trade) <= capped_limit:
        return sorted(trainable_best_trade, key=_shadow_sort_key, reverse=True)

    selected: list[Any] = []
    selected_ids: set[Any] = set()

    def add_from(candidates: list[Any], target_count: int, *, quality_first: bool = False) -> None:
        source = _sort_shadow_quality_first(candidates) if quality_first else candidates
        for candidate in source:
            if len(selected) >= capped_limit or len(selected) >= target_count:
                return
            candidate_id = _shadow_row_id(candidate)
            if candidate_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate_id)

    non_hold_target = int(capped_limit * TRAINING_MIN_NON_HOLD_SHARE)
    best_trade_target = int(capped_limit * TRAINING_MIN_BEST_TRADE_SHARE)
    non_hold = [
        row
        for row in trainable_best_trade
        if _shadow_action(row, "decision_action") in {"long", "short"}
    ]

    add_from(non_hold, min(non_hold_target, len(non_hold)), quality_first=True)
    add_from(
        trainable_best_trade,
        min(best_trade_target, len(trainable_best_trade)),
        quality_first=True,
    )
    return sorted(selected[:capped_limit], key=_shadow_sort_key, reverse=True)


def _training_window_composition(frame: pd.DataFrame) -> dict[str, Any]:
    def counts(column: str) -> dict[str, int]:
        if column not in frame:
            return {}
        return {
            str(key): int(value)
            for key, value in Counter(
                str(item or "unknown").lower().strip() or "unknown"
                for item in frame[column].tolist()
            ).most_common()
        }

    sample_count = int(len(frame))
    weight_total = float(
        frame.get("sample_weight", pd.Series([1.0] * len(frame))).astype(float).sum()
    )
    return {
        "sample_count": sample_count,
        "decision_action_counts": counts("decision_action"),
        "best_action_counts": counts("best_action"),
        "data_quality_status_counts": counts("data_quality_status"),
        "effective_weight": round(weight_total, 4),
        "effective_weight_ratio": round(weight_total / max(sample_count, 1), 4),
    }


def build_training_frame(rows: list[Any]) -> pd.DataFrame:
    data: list[dict[str, Any]] = []
    for row in rows:
        snapshot = _parse_json(getattr(row, "feature_snapshot", None))
        if not snapshot:
            continue
        raw_long_return = getattr(row, "long_return_pct", None)
        raw_short_return = getattr(row, "short_return_pct", None)
        if raw_long_return is None or raw_short_return is None:
            continue
        quality_sample = {
            "symbol": getattr(row, "symbol", ""),
            "analysis_type": getattr(row, "analysis_type", ""),
            "decision_action": getattr(row, "decision_action", ""),
            "decision_confidence": _safe_float(getattr(row, "decision_confidence", 0.0)),
            "horizon_minutes": int(getattr(row, "horizon_minutes", 10) or 10),
            "features": snapshot,
            "long_return_pct": _safe_float(raw_long_return),
            "short_return_pct": _safe_float(raw_short_return),
            "label_timestamp": getattr(row, "due_at", None),
            "best_action": getattr(row, "best_action", ""),
            "missed_opportunity": bool(getattr(row, "missed_opportunity", False)),
        }
        assessment = assess_shadow_sample(quality_sample)
        if assessment.exclude_from_training:
            continue
        long_return = _net_return_pct(_safe_float(raw_long_return))
        short_return = _net_return_pct(_safe_float(raw_short_return))
        feature_row: dict[str, Any] = dict(
            _feature_row_from_snapshot(
                snapshot,
                decision_confidence=_safe_float(getattr(row, "decision_confidence", 0.0)),
                horizon_minutes=int(getattr(row, "horizon_minutes", 10) or 10),
            )
        )
        feature_row.update(
            {
                "id": int(getattr(row, "id", 0) or 0),
                "symbol": str(getattr(row, "symbol", "") or ""),
                "decision_action": str(getattr(row, "decision_action", "") or ""),
                "best_action": str(getattr(row, "best_action", "") or ""),
                "missed_opportunity": bool(getattr(row, "missed_opportunity", False)),
                "raw_long_return_pct": _safe_float(raw_long_return),
                "raw_short_return_pct": _safe_float(raw_short_return),
                "long_return_pct": long_return,
                "short_return_pct": short_return,
                "long_tail_loss": int(long_return < -TAIL_LOSS_THRESHOLD_PCT),
                "short_tail_loss": int(short_return < -TAIL_LOSS_THRESHOLD_PCT),
                "long_win": int(long_return > WIN_RETURN_THRESHOLD_PCT),
                "short_win": int(short_return > WIN_RETURN_THRESHOLD_PCT),
                "sample_weight": assessment.weight,
                "data_quality_status": assessment.status,
                "data_quality_score": assessment.score,
                "quality_reasons": list(assessment.reasons),
            }
        )
        data.append(feature_row)
    return pd.DataFrame(data)


def shadow_training_quality_report(rows: list[Any]) -> dict[str, Any]:
    """Assess all candidate shadow rows, including rows excluded before fitting."""

    samples: list[dict[str, Any]] = []
    for row in rows:
        snapshot = _parse_json(getattr(row, "feature_snapshot", None))
        raw_long_return = getattr(row, "long_return_pct", None)
        raw_short_return = getattr(row, "short_return_pct", None)
        sample = {
            "symbol": getattr(row, "symbol", ""),
            "analysis_type": getattr(row, "analysis_type", ""),
            "decision_action": getattr(row, "decision_action", ""),
            "decision_confidence": _safe_float(getattr(row, "decision_confidence", 0.0)),
            "horizon_minutes": int(getattr(row, "horizon_minutes", 10) or 10),
            "features": snapshot,
            "long_return_pct": None if raw_long_return is None else _safe_float(raw_long_return),
            "short_return_pct": None if raw_short_return is None else _safe_float(raw_short_return),
            "label_timestamp": getattr(row, "due_at", None),
            "best_action": getattr(row, "best_action", ""),
            "missed_opportunity": bool(getattr(row, "missed_opportunity", False)),
        }
        samples.append(sample)
    annotated = annotate_samples(samples, "shadow")
    report = quality_report({"shadow": annotated})
    return {
        "quality_report": report,
        "governance_report": governance_report(report),
    }


def train_from_frame(
    frame: pd.DataFrame,
    *,
    min_samples: int = MIN_TRAINING_SAMPLES,
    completed_sample_count: int | None = None,
    training_quality_report: dict[str, Any] | None = None,
    persist_artifact: bool = True,
) -> dict[str, Any]:
    if len(frame) < min_samples:
        raise ValueError(f"训练样本不足：{len(frame)} < {min_samples}")

    frame = frame.sort_values("id").reset_index(drop=True)
    split = max(int(len(frame) * _LOCAL_ML_PARAMS.train_split_ratio), 1)
    if len(frame) - split < _LOCAL_ML_PARAMS.min_test_rows:
        split = max(len(frame) - _LOCAL_ML_PARAMS.min_test_rows, 1)

    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    x_train = train[FEATURE_KEYS]
    x_test = test[FEATURE_KEYS]
    train_weights = train.get("sample_weight", pd.Series([1.0] * len(train))).astype(float)

    long_classifier = _make_classifier(train["long_win"])
    short_classifier = _make_classifier(train["short_win"])
    long_regressor = _make_regressor(train["long_return_pct"])
    short_regressor = _make_regressor(train["short_return_pct"])

    long_classifier.fit(x_train, train["long_win"], model__sample_weight=train_weights)
    short_classifier.fit(x_train, train["short_win"], model__sample_weight=train_weights)
    long_regressor.fit(x_train, train["long_return_pct"], model__sample_weight=train_weights)
    short_regressor.fit(x_train, train["short_return_pct"], model__sample_weight=train_weights)

    long_scores = _positive_proba(long_classifier, x_test)
    short_scores = _positive_proba(short_classifier, x_test)
    long_expected_scores = long_regressor.predict(x_test)
    short_expected_scores = short_regressor.predict(x_test)
    long_pred = (long_scores >= 0.50).astype(int)
    short_pred = (short_scores >= 0.50).astype(int)

    now = datetime.now(UTC).isoformat()
    completed_count = int(completed_sample_count or len(frame))
    frame_quality_report = training_quality_report or quality_report(
        {
            "shadow": [
                {
                    "data_quality_status": row.get("data_quality_status", "included"),
                    "sample_weight": row.get("sample_weight", 1.0),
                    "quality_reasons": row.get("quality_reasons", []),
                }
                for row in frame.to_dict("records")
            ]
        }
    )
    metadata = {
        "version": now,
        "trained_at": now,
        "sample_count": int(len(frame)),
        "completed_shadow_sample_count": completed_count,
        "last_trained_completed_shadow_sample_count": completed_count,
        "training_shadow_sample_count": int(len(frame)),
        "training_window_composition": _training_window_composition(frame),
        "quality_report": frame_quality_report,
        "governance_report": governance_report(frame_quality_report),
        "training_shadow_sample_limit": TRAINING_SHADOW_SAMPLE_LIMIT,
        "training_sample_note": "sample_count is the latest training window, not the all-time total.",
        "training_cursor_note": "last_trained_completed_shadow_sample_count is the cumulative cursor used for auto-training.",
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "feature_count": len(FEATURE_KEYS),
        "horizons": sorted(int(v) for v in frame["horizon_minutes"].dropna().unique().tolist()),
        "win_return_threshold_pct": WIN_RETURN_THRESHOLD_PCT,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "tail_loss_threshold_pct": TAIL_LOSS_THRESHOLD_PCT,
        "training_objective": (
            "Predict executable net return after round-trip fee/slippage cost; "
            "win rate is auxiliary and tail-loss samples are tracked for risk."
        ),
        "metrics": {
            "long_auc": _safe_auc(test["long_win"], long_scores),
            "short_auc": _safe_auc(test["short_win"], short_scores),
            "long_pr_auc": _safe_pr_auc(test["long_win"], long_scores),
            "short_pr_auc": _safe_pr_auc(test["short_win"], short_scores),
            "long_accuracy": (
                float(accuracy_score(test["long_win"], long_pred)) if len(test) else None
            ),
            "short_accuracy": (
                float(accuracy_score(test["short_win"], short_pred)) if len(test) else None
            ),
            "top_long_avg_return_pct": _bucket_return(
                test["long_return_pct"], long_expected_scores, top=True
            ),
            "bottom_long_avg_return_pct": _bucket_return(
                test["long_return_pct"], long_expected_scores, top=False
            ),
            "top_long_win_rate": _bucket_win_rate(test["long_win"], long_scores, top=True),
            "bottom_long_win_rate": _bucket_win_rate(test["long_win"], long_scores, top=False),
            "top_short_avg_return_pct": _bucket_return(
                test["short_return_pct"], short_expected_scores, top=True
            ),
            "bottom_short_avg_return_pct": _bucket_return(
                test["short_return_pct"], short_expected_scores, top=False
            ),
            "top_short_win_rate": _bucket_win_rate(test["short_win"], short_scores, top=True),
            "bottom_short_win_rate": _bucket_win_rate(test["short_win"], short_scores, top=False),
        },
        "score_bucket_diagnostics": _score_bucket_diagnostics(
            test,
            long_expected_scores=long_expected_scores,
            short_expected_scores=short_expected_scores,
        ),
        "feature_keys": FEATURE_KEYS,
        "mode": "entry_profit_filter",
        "training_run_mode": "persist" if persist_artifact else "dry_run",
        "artifact_persisted": bool(persist_artifact),
        "note": "本地 ML 以预期盈亏和收益质量为主，胜率仅作为辅助过滤；用于开仓门槛/否决，不直接决定交易方向。",
    }

    bundle = {
        "long_classifier": long_classifier,
        "short_classifier": short_classifier,
        "long_regressor": long_regressor,
        "short_regressor": short_regressor,
        "metadata": metadata,
        "feature_keys": FEATURE_KEYS,
    }
    if persist_artifact:
        dump_trusted_joblib(bundle, MODEL_PATH, trusted_root=MODEL_DIR)
        METADATA_PATH.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return metadata


class MLSignalService:
    """Lazy loader and inference wrapper for the local profit-quality model."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._bundle: dict[str, Any] | None = None
        self._loaded_mtime: float | None = None
        self._train_lock = asyncio.Lock()
        self._training = False
        self._last_check_at: str | None = None
        self._next_check_at: str | None = None
        self._last_train_started_at: str | None = None
        self._last_train_finished_at: str | None = None
        self._last_train_result: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        self._ensure_loaded()
        auto_status = self._auto_train_status()
        if not self._bundle:
            readiness = disabled_ml_readiness(
                "no_model",
                "ML model artifact is not available.",
            )
            return {
                "available": False,
                "status": "no_model",
                "readiness_state": readiness["state"],
                "readiness": readiness,
                "allow_live_position_influence": False,
                "model_path": str(self.model_path),
                "message": "本地 ML 盈亏质量模型尚未训练。",
                **auto_status,
            }
        metadata = _safe_dict(self._bundle.get("metadata"))
        influence = _influence_policy(metadata)
        readiness = build_ml_readiness_report(metadata, influence)
        allow_live_position_influence = bool(readiness.get("allow_live_position_influence"))
        advisory_enabled = bool(
            influence.get("advisory_enabled") and readiness.get("state") == "shadow_ready"
        )
        model_note = metadata.get("note")
        training_count = int(metadata.get("sample_count") or 0)
        return {
            "available": True,
            "model_path": str(self.model_path),
            **metadata,
            "training_shadow_sample_count": int(
                metadata.get("training_shadow_sample_count") or training_count
            ),
            "training_shadow_sample_limit": int(
                metadata.get("training_shadow_sample_limit") or TRAINING_SHADOW_SAMPLE_LIMIT
            ),
            "training_sample_note": metadata.get("training_sample_note")
            or "sample_count is the latest training window, not the all-time total.",
            "status": (
                "ready"
                if allow_live_position_influence
                else str(readiness.get("state") or influence.get("status") or "learning_only")
            ),
            "mode": (
                "entry_profit_filter"
                if allow_live_position_influence
                else (
                    "advisory"
                    if advisory_enabled
                    else str(readiness.get("state") or "learning_only")
                )
            ),
            "readiness_state": readiness.get("state"),
            "readiness": readiness,
            "allow_live_position_influence": allow_live_position_influence,
            "influence_enabled": allow_live_position_influence,
            "advisory_enabled": advisory_enabled,
            "influence_policy": influence,
            "model_note": model_note,
            "note": (
                "ML 指标达标，当前允许参与开仓过滤、加分和机会排序。"
                if allow_live_position_influence
                else (
                    "ML 硬指标有效但样本成熟度不足，当前按小权重提供收益解释，不做硬否决。"
                    if influence.get("advisory_enabled")
                    else "ML 指标未达标，当前只学习不介入；继续预测、影子复盘和自动训练，达标后自动恢复。"
                )
            ),
            **auto_status,
        }

    async def maybe_auto_train(self, *, force: bool = False) -> dict[str, Any]:
        """Retrain in the background when enough fresh shadow samples exist."""
        if self._train_lock.locked():
            return {
                "trained": False,
                "reason": "training_in_progress",
                "message": "本地 ML 模型正在训练中，本次跳过重复训练。",
            }

        async with self._train_lock:
            now = datetime.now(UTC)
            self._last_check_at = now.isoformat()
            self._next_check_at = None
            try:
                completed_count = await self._completed_shadow_sample_count()
                metadata = self._current_metadata()
                last_sample_count = int(metadata.get("sample_count") or 0)
                last_completed_count = int(
                    metadata.get("last_trained_completed_shadow_sample_count")
                    or metadata.get("completed_shadow_sample_count")
                    or metadata.get("completed_sample_count")
                    or last_sample_count
                    or 0
                )
                influence = _influence_policy(metadata) if metadata else {"enabled": False}
                readiness = (
                    build_ml_readiness_report(metadata, influence)
                    if metadata
                    else disabled_ml_readiness(
                        "no_metadata",
                        "ML model metadata is not available.",
                    )
                )
                learning_only = not bool(readiness.get("allow_live_position_influence"))
                min_interval_seconds = (
                    AUTO_TRAIN_LEARNING_ONLY_INTERVAL_SECONDS
                    if learning_only
                    else AUTO_TRAIN_MIN_INTERVAL_SECONDS
                )
                min_new_samples = (
                    AUTO_TRAIN_LEARNING_ONLY_MIN_NEW_SAMPLES
                    if learning_only
                    else AUTO_TRAIN_MIN_NEW_SAMPLES
                )
                trained_at = self._parse_datetime(
                    metadata.get("trained_at") or metadata.get("version")
                )
                age_seconds = (
                    (now - trained_at).total_seconds()
                    if trained_at is not None
                    else min_interval_seconds
                )
                new_samples = max(completed_count - last_completed_count, 0)
                training_policy = {
                    "learning_only": learning_only,
                    "readiness_state": readiness.get("state"),
                    "readiness_blocking_reasons": readiness.get("blocking_reasons") or [],
                    "min_interval_seconds": min_interval_seconds,
                    "min_new_samples": min_new_samples,
                    "min_training_samples": MIN_TRAINING_SAMPLES,
                    "cursor_source": "last_trained_completed_shadow_sample_count",
                    "promotion_requires_readiness": True,
                    "candidate_artifact_persisted": False,
                    "persist_artifact_only_when_readiness_allows_live_influence": True,
                }
                if completed_count < MIN_TRAINING_SAMPLES:
                    result = {
                        "trained": False,
                        "reason": "not_enough_samples",
                        "completed_sample_count": completed_count,
                        "last_trained_sample_count": last_sample_count,
                        "last_trained_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "training_policy": training_policy,
                        "message": f"本地 ML 自动训练样本不足：{completed_count} < {MIN_TRAINING_SAMPLES}。继续累计影子复盘样本。",
                    }
                    self._last_train_result = result
                    return result

                last_result = _safe_dict(self._last_train_result)
                last_result_finished_at = self._parse_datetime(self._last_train_finished_at)
                last_result_completed_count = int(
                    last_result.get("completed_sample_count")
                    or last_result.get("last_trained_completed_sample_count")
                    or 0
                )
                last_result_age_seconds = (
                    (now - last_result_finished_at).total_seconds()
                    if last_result_finished_at is not None
                    else min_interval_seconds
                )
                if (
                    not force
                    and last_result.get("reason") == "candidate_readiness_rejected"
                    and last_result_age_seconds < min_interval_seconds
                    and completed_count - last_result_completed_count < min_new_samples
                ):
                    result = {
                        "trained": False,
                        "reason": "not_due_after_candidate_rejection",
                        "completed_sample_count": completed_count,
                        "last_candidate_completed_sample_count": last_result_completed_count,
                        "last_candidate_rejected_at": self._last_train_finished_at,
                        "new_sample_count": max(completed_count - last_result_completed_count, 0),
                        "model_age_seconds": round(age_seconds, 1),
                        "last_candidate_age_seconds": round(last_result_age_seconds, 1),
                        "training_policy": training_policy,
                        "message": (
                            "上一轮候选 ML 模型未通过 readiness 晋级门，且尚未达到重新评估间隔"
                            "或新增样本门槛；继续保留当前线上 artifact。"
                        ),
                    }
                    self._last_train_result = result
                    return result

                should_train = force or (
                    age_seconds >= min_interval_seconds or new_samples >= min_new_samples
                )
                if not should_train:
                    result = {
                        "trained": False,
                        "reason": "not_due",
                        "completed_sample_count": completed_count,
                        "last_trained_sample_count": last_sample_count,
                        "last_trained_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "model_age_seconds": round(age_seconds, 1),
                        "training_policy": training_policy,
                        "message": (
                            f"未达到自动训练条件：需要距离上次训练至少 {min_interval_seconds // 3600} 小时，"
                            f"或新增 completed 影子复盘样本不少于 {min_new_samples} 条。"
                        ),
                    }
                    self._last_train_result = result
                    return result

                self._training = True
                self._last_train_started_at = datetime.now(UTC).isoformat()
                quarantine_result = await self._quarantine_dirty_training_samples()
                completed_count = await self._completed_shadow_sample_count()
                new_samples = max(completed_count - last_completed_count, 0)
                if completed_count < MIN_TRAINING_SAMPLES:
                    result = {
                        "trained": False,
                        "reason": "not_enough_clean_samples",
                        "completed_sample_count": completed_count,
                        "last_trained_sample_count": last_sample_count,
                        "last_trained_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "training_policy": training_policy,
                        "training_quarantine": quarantine_result,
                        "message": (
                            f"自动隔离脏样本后，干净影子复盘样本不足："
                            f"{completed_count} < {MIN_TRAINING_SAMPLES}。继续累计可训练样本。"
                        ),
                    }
                    self._last_train_result = result
                    return result
                rows = await load_shadow_training_rows(limit=TRAINING_SHADOW_SAMPLE_LIMIT)
                quality_state = shadow_training_quality_report(rows)
                frame = build_training_frame(rows)
                candidate_metadata = await asyncio.to_thread(
                    train_from_frame,
                    frame,
                    completed_sample_count=completed_count,
                    training_quality_report=quality_state["quality_report"],
                    persist_artifact=False,
                )
                candidate_influence = _influence_policy(candidate_metadata)
                candidate_readiness = build_ml_readiness_report(
                    candidate_metadata,
                    candidate_influence,
                )
                candidate_summary = {
                    "sample_count": int(candidate_metadata.get("sample_count") or 0),
                    "test_count": int(candidate_metadata.get("test_count") or 0),
                    "trained_at": candidate_metadata.get("trained_at"),
                    "training_run_mode": candidate_metadata.get("training_run_mode"),
                    "artifact_persisted": bool(candidate_metadata.get("artifact_persisted")),
                    "metrics": _safe_dict(candidate_metadata.get("metrics")),
                    "training_window_composition": _safe_dict(
                        candidate_metadata.get("training_window_composition")
                    ),
                    "quality_totals": _safe_dict(
                        _safe_dict(candidate_metadata.get("quality_report")).get("totals")
                    ),
                }
                if not bool(candidate_readiness.get("allow_live_position_influence")):
                    result = {
                        "trained": False,
                        "reason": "candidate_readiness_rejected",
                        "completed_sample_count": completed_count,
                        "previous_sample_count": last_sample_count,
                        "previous_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "sample_count": int(candidate_metadata.get("sample_count") or 0),
                        "last_trained_completed_sample_count": last_completed_count,
                        "training_quarantine": quarantine_result,
                        "training_policy": training_policy,
                        "candidate": candidate_summary,
                        "candidate_readiness": candidate_readiness,
                        "candidate_influence_policy": candidate_influence,
                        "artifact_persisted": False,
                        "message": (
                            "候选 ML 模型未通过 readiness 晋级门，已拒绝替换线上 artifact；"
                            "继续使用上一版模型或保持学习观察。"
                        ),
                    }
                    self._last_train_result = result
                    return result

                trained_metadata = await asyncio.to_thread(
                    train_from_frame,
                    frame,
                    completed_sample_count=completed_count,
                    training_quality_report=quality_state["quality_report"],
                    persist_artifact=True,
                )
                self._bundle = None
                self._loaded_mtime = None
                self._ensure_loaded()
                result = {
                    "trained": True,
                    "reason": "trained",
                    "completed_sample_count": completed_count,
                    "previous_sample_count": last_sample_count,
                    "previous_completed_sample_count": last_completed_count,
                    "new_sample_count": new_samples,
                    "sample_count": int(trained_metadata.get("sample_count") or 0),
                    "last_trained_completed_sample_count": int(
                        trained_metadata.get("last_trained_completed_shadow_sample_count")
                        or completed_count
                    ),
                    "training_quarantine": quarantine_result,
                    "training_policy": training_policy,
                    "candidate": candidate_summary,
                    "candidate_readiness": candidate_readiness,
                    "candidate_influence_policy": candidate_influence,
                    "artifact_persisted": bool(trained_metadata.get("artifact_persisted")),
                    "trained_at": trained_metadata.get("trained_at"),
                    "message": "本地 ML 盈亏质量模型已自动完成训练并热加载。",
                }
                self._last_train_result = result
                return result
            except Exception as exc:
                error = safe_error_text(exc, limit=160)
                result = {
                    "trained": False,
                    "reason": "error",
                    "error": error,
                    "message": f"本地 ML 自动训练失败，继续使用上一版模型：{error}",
                }
                self._last_train_result = result
                return result
            finally:
                finished = datetime.now(UTC)
                if self._training:
                    self._last_train_finished_at = finished.isoformat()
                self._training = False
                self._next_check_at = datetime.fromtimestamp(
                    finished.timestamp() + AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
                    tz=UTC,
                ).isoformat()

    def predict(self, features: Any, *, horizons: tuple[int, ...] = (10, 30)) -> dict[str, Any]:
        self._ensure_loaded()
        if not self._bundle:
            readiness = disabled_ml_readiness(
                "no_model",
                "ML model artifact is not available.",
            )
            return {
                "available": False,
                "status": "no_model",
                "readiness_state": readiness["state"],
                "readiness": readiness,
                "allow_live_position_influence": False,
                "message": "本地 ML 盈亏质量模型尚未训练，当前分析不使用 ML 辅助信号。",
            }
        metadata = _safe_dict(self._bundle.get("metadata"))
        influence = _influence_policy(metadata)
        readiness = build_ml_readiness_report(metadata, influence)
        allow_live_position_influence = bool(readiness.get("allow_live_position_influence"))
        advisory_enabled = bool(
            influence.get("advisory_enabled") and readiness.get("state") == "shadow_ready"
        )

        predictions = []
        for horizon in horizons:
            row = _feature_row_from_feature_vector(features, horizon_minutes=horizon)
            x = pd.DataFrame([row], columns=FEATURE_KEYS)
            long_win_rate = float(_positive_proba(self._bundle["long_classifier"], x)[0])
            short_win_rate = float(_positive_proba(self._bundle["short_classifier"], x)[0])
            long_expected = float(self._bundle["long_regressor"].predict(x)[0])
            short_expected = float(self._bundle["short_regressor"].predict(x)[0])
            best_side = "long" if long_expected >= short_expected else "short"
            best_win = long_win_rate if best_side == "long" else short_win_rate
            best_expected = long_expected if best_side == "long" else short_expected
            profit_edge = abs(long_expected - short_expected)
            profit_quality = _profit_quality_score(best_expected, best_win, profit_edge)
            side_influence = _safe_dict(influence.get(best_side))
            risk_score = _clamp(
                max(-best_expected, 0.0) / max(WIN_RETURN_THRESHOLD_PCT, 1e-9)
                + max(MIN_PROFIT_SIGNAL_WIN_RATE - best_win, 0.0)
            )
            predictions.append(
                {
                    "horizon_minutes": int(horizon),
                    "long_win_rate": round(long_win_rate, 4),
                    "short_win_rate": round(short_win_rate, 4),
                    "long_expected_return_pct": round(long_expected, 4),
                    "short_expected_return_pct": round(short_expected, 4),
                    "best_side": best_side,
                    "best_win_rate": round(best_win, 4),
                    "best_expected_return_pct": round(best_expected, 4),
                    "profit_edge_pct": round(profit_edge, 4),
                    "profit_quality_score": round(profit_quality, 4),
                    "profit_signal": bool(
                        allow_live_position_influence
                        and side_influence.get("enabled")
                        and best_expected > WIN_RETURN_THRESHOLD_PCT
                        and profit_edge >= MIN_PROFIT_EDGE_PCT
                        and best_win >= MIN_PROFIT_SIGNAL_WIN_RATE
                    ),
                    "risk_score": round(risk_score, 4),
                    "ml_influence_enabled": bool(
                        allow_live_position_influence and side_influence.get("enabled")
                    ),
                }
            )

        primary = predictions[0] if predictions else {}
        return {
            "available": True,
            "status": (
                "entry_profit_filter"
                if allow_live_position_influence
                else (
                    "advisory"
                    if advisory_enabled
                    else str(readiness.get("state") or "learning_only")
                )
            ),
            "mode": (
                "entry_profit_filter"
                if allow_live_position_influence
                else (
                    "advisory"
                    if advisory_enabled
                    else str(readiness.get("state") or "learning_only")
                )
            ),
            "readiness_state": readiness.get("state"),
            "readiness": readiness,
            "allow_live_position_influence": allow_live_position_influence,
            "influence_enabled": allow_live_position_influence,
            "advisory_enabled": advisory_enabled,
            "influence_policy": influence,
            "model_version": metadata.get("version"),
            "trained_sample_count": int(metadata.get("sample_count") or 0),
            "primary_horizon_minutes": primary.get("horizon_minutes"),
            "long_win_rate": primary.get("long_win_rate"),
            "short_win_rate": primary.get("short_win_rate"),
            "expected_return_pct": primary.get("best_expected_return_pct"),
            "profit_edge_pct": primary.get("profit_edge_pct"),
            "profit_quality_score": primary.get("profit_quality_score"),
            "profit_signal": primary.get("profit_signal"),
            "risk_score": primary.get("risk_score"),
            "suggestion": self._suggestion(primary, influence),
            "predictions": predictions,
            "note": (
                "ML 当前指标达标，参与开仓门槛/否决和机会排序；不直接决定交易方向。"
                if influence.get("enabled")
                else (
                    "ML 当前为建议权重模式：参与 expected_net 解释和轻量排序，不作为硬否决。"
                    if influence.get("advisory_enabled")
                    else "ML 当前处于学习观察中：继续预测、影子复盘和自动训练，但不影响开仓过滤、加分或机会排序。"
                )
            ),
        }

    def _ensure_loaded(self) -> None:
        try:
            if not self.model_path.exists():
                self._bundle = None
                self._loaded_mtime = None
                return
            mtime = self.model_path.stat().st_mtime
            if self._bundle is not None and self._loaded_mtime == mtime:
                return
            self._bundle = load_trusted_joblib(
                self.model_path,
                trusted_root=MODEL_DIR,
                expected_type=dict,
            )
            self._loaded_mtime = mtime
        except Exception as exc:
            logger.warning(
                "failed to load ML signal model",
                path=str(self.model_path),
                error=safe_error_text(exc),
            )
            self._bundle = None
            self._loaded_mtime = None

    def _auto_train_status(self) -> dict[str, Any]:
        return {
            "auto_train_enabled": True,
            "auto_train_check_interval_seconds": AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
            "auto_train_min_interval_seconds": AUTO_TRAIN_MIN_INTERVAL_SECONDS,
            "auto_train_min_new_samples": AUTO_TRAIN_MIN_NEW_SAMPLES,
            "auto_train_learning_only_interval_seconds": AUTO_TRAIN_LEARNING_ONLY_INTERVAL_SECONDS,
            "auto_train_learning_only_min_new_samples": AUTO_TRAIN_LEARNING_ONLY_MIN_NEW_SAMPLES,
            "auto_training": self._training,
            "auto_train_last_check_at": self._last_check_at,
            "auto_train_next_check_at": self._next_check_at,
            "auto_train_last_started_at": self._last_train_started_at,
            "auto_train_last_finished_at": self._last_train_finished_at,
            "auto_train_last_result": self._last_train_result,
        }

    def _current_metadata(self) -> dict[str, Any]:
        self._ensure_loaded()
        if self._bundle:
            metadata = self._bundle.get("metadata") or {}
            if isinstance(metadata, dict):
                return metadata
        try:
            if METADATA_PATH.exists():
                parsed = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.debug(
                "failed to read ML signal metadata",
                path=str(METADATA_PATH),
                error=safe_error_text(exc),
            )
        return {}

    async def _completed_shadow_sample_count(self) -> int:
        return await count_shadow_training_rows()

    async def completed_shadow_sample_count(self) -> int:
        """Return completed shadow samples through a public dashboard boundary."""

        return await self._completed_shadow_sample_count()

    async def _quarantine_dirty_training_samples(
        self,
        *,
        only_newer_than_id: int | None = None,
    ) -> dict[str, Any]:
        return await quarantine_dirty_shadow_samples(
            batch_size=_LOCAL_ML_PARAMS.auto_quarantine_batch_size,
            max_batches=_LOCAL_ML_PARAMS.auto_quarantine_max_batches,
            only_newer_than_id=only_newer_than_id,
        )

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            return None

    def _suggestion(self, primary: dict[str, Any], influence: dict[str, Any] | None = None) -> str:
        if not primary:
            return "暂无 ML 预测。"
        if isinstance(influence, dict) and not influence.get("enabled"):
            if influence.get("advisory_enabled"):
                return "ML 样本成熟度不足但排序有效，当前仅按小权重辅助收益解释。"
            return "ML 当前评估未达标，自动降级为学习观察；继续训练，暂不介入交易决策。"
        win = float(primary.get("best_win_rate") or 0.0)
        expected = float(primary.get("best_expected_return_pct") or 0.0)
        edge = float(primary.get("profit_edge_pct") or 0.0)
        side = "做多" if primary.get("best_side") == "long" else "做空"
        if (
            expected > WIN_RETURN_THRESHOLD_PCT
            and edge >= MIN_PROFIT_EDGE_PCT
            and win >= MIN_PROFIT_SIGNAL_WIN_RATE
        ):
            return f"ML 盈亏期望支持{side}，胜率仅作辅助；可作为开仓质量加分。"
        if expected <= 0:
            return "ML 预期盈亏为负，后续可用于提高入场门槛。"
        if win >= 0.62 and expected <= WIN_RETURN_THRESHOLD_PCT:
            return "ML 胜率不低但预期收益不足，不应仅因胜率高而加分。"
        if edge < MIN_PROFIT_EDGE_PCT:
            return "ML 多空预期收益差距不明显，信号中性。"
        if win < MIN_PROFIT_SIGNAL_WIN_RATE:
            return "ML 预期收益尚可但胜率过低，需更强单币种确认。"
        return "ML 盈亏质量信号中性，暂不改变 AI 决策。"


async def load_shadow_training_rows(limit: int = TRAINING_SHADOW_SAMPLE_LIMIT) -> list[Any]:
    async with get_read_session_ctx() as session:
        safe_limit = max(int(limit or TRAINING_SHADOW_SAMPLE_LIMIT), 1)
        recent_limit = max(int(safe_limit * TRAINING_BALANCED_RECENT_CANDIDATE_SHARE), 1)
        non_hold_limit = max(
            int(safe_limit * TRAINING_BALANCED_NON_HOLD_CANDIDATE_SHARE),
            int(safe_limit * TRAINING_MIN_NON_HOLD_SHARE),
        )
        best_trade_limit = max(
            int(safe_limit * TRAINING_BALANCED_BEST_TRADE_CANDIDATE_SHARE),
            int(safe_limit * TRAINING_MIN_BEST_TRADE_SHARE),
        )
        base_filters = (
            ShadowBacktest.status == "completed",
            ShadowBacktest.long_return_pct.is_not(None),
            ShadowBacktest.short_return_pct.is_not(None),
        )
        order_by = (ShadowBacktest.created_at.desc(), ShadowBacktest.id.desc())
        columns = _shadow_training_columns()

        async def load_rows(stmt: Any) -> list[ShadowTrainingRow]:
            return [
                _shadow_training_row_from_mapping(row)
                for row in (await session.execute(stmt)).mappings().all()
            ]

        recent_rows = await load_rows(
            select(*columns).where(*base_filters).order_by(*order_by).limit(recent_limit)
        )
        non_hold_rows = await load_rows(
            select(*columns)
            .where(
                *base_filters,
                ShadowBacktest.decision_action.in_(["long", "short"]),
            )
            .order_by(*order_by)
            .limit(non_hold_limit)
        )
        best_trade_rows = await load_rows(
            select(*columns)
            .where(
                *base_filters,
                ShadowBacktest.best_action.in_(["long", "short"]),
            )
            .order_by(*order_by)
            .limit(best_trade_limit)
        )
        return select_shadow_training_rows(
            [*recent_rows, *non_hold_rows, *best_trade_rows],
            limit=safe_limit,
        )


async def count_shadow_training_rows() -> int:
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(func.count(ShadowBacktest.id)).where(
                ShadowBacktest.status == "completed",
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            )
        )
        return int(result.scalar() or 0)


def _top_counts(values: list[Any], *, limit: int = 8) -> dict[str, int]:
    normalized = []
    for value in values:
        text = str(value or "unknown").strip().lower() or "unknown"
        normalized.append(text)
    return dict(Counter(normalized).most_common(limit))


def _flatten_quality_reasons(values: list[Any]) -> list[str]:
    reasons: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            reasons.extend(str(item) for item in value if str(item or "").strip())
        elif str(value or "").strip():
            reasons.append(str(value))
    return reasons


def _bucket_indices(scores: np.ndarray, *, top: bool) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=int)
    count = max(int(len(scores) * 0.20), 1)
    order = np.argsort(scores)
    return order[-count:] if top else order[:count]


def _bucket_segment_summary(
    test: pd.DataFrame,
    scores: np.ndarray,
    *,
    side: str,
    top: bool,
) -> dict[str, Any]:
    idx = _bucket_indices(scores, top=top)
    bucket = test.iloc[idx].copy() if len(idx) else test.iloc[:0].copy()
    score_values = pd.Series(scores).iloc[idx] if len(idx) else pd.Series([], dtype=float)
    return_col = f"{side}_return_pct"
    win_col = f"{side}_win"
    reasons = _flatten_quality_reasons(
        bucket.get("quality_reasons", pd.Series([], dtype=object)).tolist()
    )
    return {
        "count": int(len(bucket)),
        "avg_model_score": None if bucket.empty else float(score_values.mean()),
        "avg_return_pct": None if bucket.empty else float(bucket[return_col].mean()),
        "win_rate": None if bucket.empty else float(bucket[win_col].mean()),
        "avg_sample_weight": (
            None
            if bucket.empty
            else float(bucket.get("sample_weight", pd.Series([1.0] * len(bucket))).mean())
        ),
        "action_counts": _top_counts(
            bucket.get("decision_action", pd.Series(["unknown"] * len(bucket))).tolist()
        ),
        "best_action_counts": _top_counts(
            bucket.get("best_action", pd.Series(["unknown"] * len(bucket))).tolist()
        ),
        "horizon_counts": _top_counts(
            bucket.get("horizon_minutes", pd.Series(["unknown"] * len(bucket))).tolist()
        ),
        "data_quality_status_counts": _top_counts(
            bucket.get("data_quality_status", pd.Series(["unknown"] * len(bucket))).tolist()
        ),
        "top_quality_reasons": [
            {"reason": reason, "count": count} for reason, count in Counter(reasons).most_common(8)
        ],
    }


def _score_bucket_diagnostics(
    test: pd.DataFrame,
    *,
    long_expected_scores: np.ndarray,
    short_expected_scores: np.ndarray,
) -> dict[str, Any]:
    return {
        "long": {
            "top": _bucket_segment_summary(test, long_expected_scores, side="long", top=True),
            "bottom": _bucket_segment_summary(test, long_expected_scores, side="long", top=False),
        },
        "short": {
            "top": _bucket_segment_summary(test, short_expected_scores, side="short", top=True),
            "bottom": _bucket_segment_summary(test, short_expected_scores, side="short", top=False),
        },
    }
