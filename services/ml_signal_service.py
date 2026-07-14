"""Local ML profit-quality model built from shadow backtest outcomes.

The model is intentionally used as an observation signal first. It predicts
statistical long/short profit quality from market features, but does not
execute trades by itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sqlalchemy import and_, func, or_, select

from config.settings import settings
from core.model_artifact_safety import dump_trusted_joblib, load_trusted_joblib
from core.safe_output import safe_error_text
from db.session import get_read_session_ctx
from models.learning import ShadowBacktest
from services.artifact_retirement_audit import (
    PHASE3_ARTIFACT_POLICY_ID,
    PHASE3_REQUIRED_PROMOTION_FLOW,
    PHASE3_REQUIRED_TRAINING_POLICY,
)
from services.dynamic_policy_values import empirical_policy_value
from services.execution_cost_model import execution_cost_estimate
from services.ml_readiness import build_ml_readiness_report, disabled_ml_readiness
from services.model_artifact_registry import ModelArtifactRegistry, ResolvedModelArtifact
from services.model_training_state import (
    LOCAL_ML_MODEL_IDS,
    ModelTrainingStateStore,
)
from services.phase3_boundary import PHASE3_CLEAN_START_UTC
from services.return_objective import (
    COST_MODEL_VERSION,
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
    return_distribution_summary,
    risk_adjusted_expected_return,
)
from services.shadow_training_quarantine import quarantine_dirty_shadow_samples
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import (
    annotate_samples,
    artifact_bound_governance_report,
    assess_shadow_sample,
    governance_report,
    quality_report,
)

logger = structlog.get_logger(__name__)

MODEL_DIR = Path("data/ml_signal")
MODEL_PATH = MODEL_DIR / "net_return_model.joblib"
METADATA_PATH = MODEL_DIR / "net_return_model_metadata.json"
ML_SIGNAL_ARTIFACT_REGISTRY = ModelArtifactRegistry(
    root=Path(settings.data_dir) / "model_artifacts",
    model_id="local_ml_profit_quality",
)
MODEL_TRAINING_STATE_STORE = ModelTrainingStateStore(
    Path(settings.data_dir) / "model_training_scheduler_state.json"
)
LOCAL_ML_TRAINING_SCHEDULER_ID = "local_ml_auto_train"
AUTO_TRAIN_RETRY_INTERVAL_SECONDS = 5 * 60
AUTO_TRAIN_LEASE_STALE_SECONDS = 60 * 60


def _training_source_code_version() -> str:
    digest = hashlib.sha256()
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("training_data_quality.py"),
        Path(__file__).with_name("ml_readiness.py"),
        Path(__file__).with_name("return_objective.py"),
        Path(__file__).with_name("model_artifact_registry.py"),
    )
    for path in source_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return f"source-sha256:{digest.hexdigest()}"
_LOCAL_ML_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
AUTO_TRAIN_CHECK_INTERVAL_SECONDS = _LOCAL_ML_PARAMS.auto_train_check_interval_seconds

FEATURE_KEYS = [
    "abnormal_wick_count_72h",
    "abnormal_wick_max_pct",
    "abnormal_wick_recent_hours",
    "change_24h_pct",
    "spread_pct",
    "rsi_14",
    "rsi_7",
    "macd",
    "macd_signal",
    "macd_diff",
    "ema_12_gap_pct",
    "ema_26_gap_pct",
    "stoch_k",
    "adx_14",
    "bb_width",
    "bb_pct",
    "atr_pct",
    "entry_activity_volume_ratio",
    "volume_ratio",
    "returns_1",
    "returns_5",
    "returns_20",
    "volatility_20",
    "price_vs_sma20",
    "price_vs_sma50",
    "sector_relative_strength",
    "indicator_price_gap_pct",
    "liquidation_risk_score",
    "whale_txn_count",
    "exchange_inflow",
    "funding_rate",
    "log_notional_24h_usdt",
    "log_volume_24h",
    "log_open_interest_value",
    "orderbook_imbalance",
    "orderbook_depth_ratio",
    "sentiment_data_available",
    "direct_sentiment_data_available",
    "news_sentiment_avg",
    "social_sentiment_avg",
    "social_mention_count",
    "news_article_count",
    "direct_news_item_count",
    "market_news_item_count",
    "sequence_length",
    "decision_confidence",
    "horizon_minutes",
]

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
    ema_12 = _safe_float(snapshot.get("ema_12"), 0.0)
    ema_26 = _safe_float(snapshot.get("ema_26"), 0.0)
    bid_depth = _safe_float(snapshot.get("orderbook_bid_depth"), 0.0)
    ask_depth = _safe_float(snapshot.get("orderbook_ask_depth"), 0.0)
    depth_total = max(bid_depth + ask_depth, 1e-9)
    values = {
        "abnormal_wick_count_72h": _safe_float(snapshot.get("abnormal_wick_count_72h")),
        "abnormal_wick_max_pct": _safe_float(snapshot.get("abnormal_wick_max_pct")),
        "abnormal_wick_recent_hours": _safe_float(
            snapshot.get("abnormal_wick_recent_hours"), 9999.0
        ),
        "change_24h_pct": _safe_float(snapshot.get("change_24h_pct")),
        "spread_pct": _safe_float(snapshot.get("spread_pct")),
        "rsi_14": _safe_float(snapshot.get("rsi_14"), 50.0),
        "rsi_7": _safe_float(snapshot.get("rsi_7"), 50.0),
        "macd": _safe_float(snapshot.get("macd")),
        "macd_signal": _safe_float(snapshot.get("macd_signal")),
        "macd_diff": _safe_float(snapshot.get("macd_diff")),
        "ema_12_gap_pct": ((price - ema_12) / price * 100.0) if price > 0 and ema_12 > 0 else 0.0,
        "ema_26_gap_pct": ((price - ema_26) / price * 100.0) if price > 0 and ema_26 > 0 else 0.0,
        "stoch_k": _safe_float(snapshot.get("stoch_k"), 50.0),
        "adx_14": _safe_float(snapshot.get("adx_14")),
        "bb_width": _safe_float(snapshot.get("bb_width")),
        "bb_pct": _safe_float(snapshot.get("bb_pct"), 0.5),
        "atr_pct": atr / price if price > 0 else 0.0,
        "entry_activity_volume_ratio": _safe_float(
            snapshot.get("entry_activity_volume_ratio"),
            _safe_float(snapshot.get("volume_ratio"), 1.0),
        ),
        "volume_ratio": _safe_float(snapshot.get("volume_ratio"), 1.0),
        "returns_1": _safe_float(snapshot.get("returns_1")),
        "returns_5": _safe_float(snapshot.get("returns_5")),
        "returns_20": _safe_float(snapshot.get("returns_20")),
        "volatility_20": _safe_float(snapshot.get("volatility_20")),
        "price_vs_sma20": _safe_float(snapshot.get("price_vs_sma20")),
        "price_vs_sma50": _safe_float(snapshot.get("price_vs_sma50")),
        "sector_relative_strength": _safe_float(snapshot.get("sector_relative_strength")),
        "indicator_price_gap_pct": _safe_float(snapshot.get("indicator_price_gap_pct")),
        "liquidation_risk_score": _safe_float(snapshot.get("liquidation_risk_score")),
        "whale_txn_count": _safe_float(snapshot.get("whale_txn_count")),
        "exchange_inflow": _safe_float(snapshot.get("exchange_inflow")),
        "funding_rate": _safe_float(snapshot.get("funding_rate")),
        "log_notional_24h_usdt": math.log10(
            max(_safe_float(snapshot.get("notional_24h_usdt")), 0.0) + 1.0
        ),
        "log_volume_24h": math.log10(max(_safe_float(snapshot.get("volume_24h")), 0.0) + 1.0),
        "log_open_interest_value": math.log10(
            max(_safe_float(snapshot.get("open_interest_value")), 0.0) + 1.0
        ),
        "orderbook_imbalance": _safe_float(snapshot.get("orderbook_imbalance")),
        "orderbook_depth_ratio": (bid_depth - ask_depth) / depth_total,
        "sentiment_data_available": 1.0 if snapshot.get("sentiment_data_available") else 0.0,
        "direct_sentiment_data_available": (
            1.0 if snapshot.get("direct_sentiment_data_available") else 0.0
        ),
        "news_sentiment_avg": _safe_float(snapshot.get("news_sentiment_avg")),
        "social_sentiment_avg": _safe_float(snapshot.get("social_sentiment_avg")),
        "social_mention_count": _safe_float(snapshot.get("social_mention_count")),
        "news_article_count": _safe_float(snapshot.get("news_article_count")),
        "direct_news_item_count": _safe_float(snapshot.get("direct_news_item_count")),
        "market_news_item_count": _safe_float(snapshot.get("market_news_item_count")),
        "sequence_length": _safe_float(snapshot.get("sequence_length")),
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


def _optional_positive_proba(model: Any, x: pd.DataFrame, *, default: float = 0.0) -> np.ndarray:
    if model is None:
        return np.full(len(x), float(default), dtype=float)
    try:
        return _positive_proba(model, x)
    except Exception as exc:
        logger.debug(
            "failed to score optional ML probability model",
            error=safe_error_text(exc),
        )
        return np.full(len(x), float(default), dtype=float)


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


def _bucket_return_summary(
    y_return: pd.Series,
    scores: np.ndarray,
    *,
    top: bool,
    tail_loss_threshold_pct: float,
) -> dict[str, Any]:
    if not len(scores):
        return return_distribution_summary(
            [],
            tail_loss_threshold_pct=tail_loss_threshold_pct,
        )
    count = max(int(math.sqrt(len(scores))), 1)
    order = np.argsort(scores)
    idx = order[-count:] if top else order[:count]
    return return_distribution_summary(
        pd.Series(y_return).iloc[idx].astype(float).tolist(),
        tail_loss_threshold_pct=tail_loss_threshold_pct,
    )


def _bucket_win_rate(y_win: pd.Series, scores: np.ndarray, top: bool) -> float | None:
    if not len(scores):
        return None
    count = max(int(math.sqrt(len(scores))), 1)
    order = np.argsort(scores)
    idx = order[-count:] if top else order[:count]
    return float(pd.Series(y_win).iloc[idx].mean())


def _regression_prediction_distribution(
    model: Pipeline,
    x: pd.DataFrame,
) -> dict[str, np.ndarray]:
    expected = np.asarray(model.predict(x), dtype=float)
    named_steps = getattr(model, "named_steps", {})
    getter = getattr(named_steps, "get", None)
    estimator = getter("model") if callable(getter) else None
    imputer = getter("imputer") if callable(getter) else None
    trees = list(getattr(estimator, "estimators_", []) or [])
    if not trees or imputer is None:
        return {
            "expected": expected,
            "median": expected.copy(),
            "lower_quantile": expected.copy(),
            "std": np.zeros(len(expected), dtype=float),
        }
    transformed = imputer.transform(x)
    tree_predictions = np.asarray([tree.predict(transformed) for tree in trees], dtype=float)
    ordered_tree_predictions = np.sort(tree_predictions, axis=0)
    lower_tail_count = max(int(math.sqrt(len(ordered_tree_predictions))), 1)
    return {
        "expected": expected,
        "median": np.median(tree_predictions, axis=0),
        "lower_quantile": np.median(ordered_tree_predictions[:lower_tail_count], axis=0),
        "std": np.std(tree_predictions, axis=0),
    }


def _risk_adjusted_expected_scores(
    distribution: dict[str, np.ndarray],
    tail_loss_scores: np.ndarray,
    *,
    tail_loss_scale_pct: float,
) -> np.ndarray:
    return np.asarray(
        [
            risk_adjusted_expected_return(
                expected_return_pct=float(distribution["expected"][index]),
                lower_quantile_return_pct=float(distribution["lower_quantile"][index]),
                tail_loss_probability=float(tail_loss_scores[index]),
                tail_loss_scale_pct=tail_loss_scale_pct,
            )["objective_net_return_pct"]
            for index in range(len(distribution["expected"]))
        ],
        dtype=float,
    )


def _profit_quality_score(
    objective_return_pct: float,
    lower_quantile_return_pct: float,
    edge_pct: float,
    tail_loss_probability: float,
    tail_loss_scale_pct: float,
) -> float:
    """Score fee-after return quality without win-rate input."""

    expected_component = max(objective_return_pct, 0.0)
    lower_bound_component = max(lower_quantile_return_pct, 0.0)
    edge_component = max(edge_pct, 0.0)
    tail_penalty = _clamp(tail_loss_probability) * max(tail_loss_scale_pct, 0.0)
    return expected_component + lower_bound_component + edge_component - tail_penalty


def _net_return_pct(raw_return_pct: float) -> float:
    """Compatibility helper for values already expressed after costs."""
    return _safe_float(raw_return_pct)


def _cost_complete_shadow_returns(
    snapshot: dict[str, Any],
    *,
    horizon_minutes: int,
    long_gross_return_pct: float,
    short_gross_return_pct: float,
) -> tuple[float, float, dict[str, Any]] | None:
    execution_cost = execution_cost_estimate(snapshot)
    funding_rate = _safe_float(snapshot.get("funding_rate"), float("nan"))
    funding_interval_minutes = _safe_float(
        snapshot.get("funding_interval_minutes"),
        float("nan"),
    )
    if not math.isfinite(funding_interval_minutes):
        funding_interval_hours = _safe_float(
            snapshot.get("funding_interval_hours"),
            float("nan"),
        )
        if math.isfinite(funding_interval_hours):
            funding_interval_minutes = funding_interval_hours * 60.0
    if (
        not execution_cost.production_eligible
        or not math.isfinite(funding_rate)
        or not math.isfinite(funding_interval_minutes)
        or funding_interval_minutes <= 0
    ):
        return None
    funding_drag = funding_rate * 100.0 * horizon_minutes / funding_interval_minutes
    long_net = (
        long_gross_return_pct
        - execution_cost.fee_pct
        - execution_cost.slippage_pct
        - funding_drag
    )
    short_net = (
        short_gross_return_pct
        - execution_cost.fee_pct
        - execution_cost.slippage_pct
        + funding_drag
    )
    return long_net, short_net, execution_cost.to_dict()


def _side_influence_status(metadata: dict[str, Any], side: str) -> dict[str, Any]:
    metrics = _safe_dict(metadata.get("metrics"))
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), 0.0)
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), 0.0)
    top_return_lcb = _safe_float(metrics.get(f"top_{side}_return_lcb_pct"), None)
    top_profit_factor = _safe_float(metrics.get(f"top_{side}_profit_factor"), None)
    top_tail_loss = _safe_float(metrics.get(f"top_{side}_tail_loss_rate"), None)
    bottom_tail_loss = _safe_float(metrics.get(f"bottom_{side}_tail_loss_rate"), None)

    hard_reasons: list[str] = []
    if (
        metadata.get("objective_name") != RETURN_OBJECTIVE_NAME
        or metadata.get("objective_version") != RETURN_OBJECTIVE_VERSION
        or metadata.get("label_version") != RETURN_LABEL_VERSION
    ):
        hard_reasons.append("artifact objective/label version is not fee-after-return v1")
    if top_return <= bottom_return:
        hard_reasons.append(
            f"高分组平均收益 {top_return:.3f}% 未优于低分组 {bottom_return:.3f}%"
        )
    if top_return_lcb is None or top_return_lcb <= 0:
        hard_reasons.append("高分组费后收益置信下界未大于 0")
    if top_profit_factor is None or top_profit_factor <= 1.0:
        hard_reasons.append("高分组 Profit Factor 未大于 1")
    if (
        top_tail_loss is None
        or bottom_tail_loss is None
        or top_tail_loss > bottom_tail_loss
    ):
        hard_reasons.append("高分组尾部损失率缺失或劣于低分组")

    reliable = not hard_reasons
    advisory = False
    influence_weight = 1.0 if reliable else 0.0
    reasons = hard_reasons
    status = "active" if reliable else "learning_only"
    return {
        "enabled": reliable,
        "advisory_enabled": advisory,
        "influence_weight": round(influence_weight, 4),
        "status": status,
        "side": side,
        "top_avg_return_pct": round(top_return, 4),
        "bottom_avg_return_pct": round(bottom_return, 4),
        "top_return_lcb_pct": None if top_return_lcb is None else round(top_return_lcb, 4),
        "top_profit_factor": (
            None if top_profit_factor is None else round(top_profit_factor, 4)
        ),
        "top_tail_loss_rate": None if top_tail_loss is None else round(top_tail_loss, 4),
        "bottom_tail_loss_rate": (
            None if bottom_tail_loss is None else round(bottom_tail_loss, 4)
        ),
        "diagnostics": {
            "auc": _safe_float(metrics.get(f"{side}_auc"), None),
            "pr_auc": _safe_float(metrics.get(f"{side}_pr_auc"), None),
            "accuracy": _safe_float(metrics.get(f"{side}_accuracy"), None),
            "top_win_rate": _safe_float(metrics.get(f"top_{side}_win_rate"), None),
            "bottom_win_rate": _safe_float(metrics.get(f"bottom_{side}_win_rate"), None),
        },
        "reasons": reasons,
        "policy": "fee_after_return_lcb_without_fixed_sample_or_return_threshold",
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
            "只有费后收益、收益置信下界、Profit Factor、尾部损失、样本成熟度和数据质量"
            "可以控制生产影响；胜率、AUC、PR-AUC 和 Accuracy 仅作诊断。"
        ),
    }


@dataclass(frozen=True)
class ShadowTrainingRow:
    id: int
    decision_id: int | None
    created_at: datetime | None
    symbol: str
    analysis_type: str
    decision_action: str
    decision_confidence: float
    feature_snapshot: Any
    due_at: datetime | None
    horizon_minutes: int
    label_version: str
    long_return_pct: float | None
    short_return_pct: float | None
    best_action: str | None
    missed_opportunity: bool


_TRAINING_FEATURE_SNAPSHOT_KEYS = (
    "abnormal_wick_count_72h",
    "abnormal_wick_max_pct",
    "abnormal_wick_recent_hours",
    "adx_14",
    "atr_14",
    "bb_pct",
    "bb_width",
    "change_24h_pct",
    "close",
    "current_price",
    "direct_news_item_count",
    "direct_sentiment_data_available",
    "ema_12",
    "ema_26",
    "entry_activity_volume_ratio",
    "exchange_inflow",
    "feature_at",
    "feature_timestamp",
    "funding_rate",
    "high_24h",
    "indicator_price_gap_pct",
    "liquidation_risk_score",
    "low_24h",
    "macd",
    "macd_diff",
    "macd_signal",
    "market_data_quality",
    "market_news_item_count",
    "news_article_count",
    "news_sentiment_avg",
    "notional_24h_usdt",
    "observed_at",
    "open_interest_value",
    "orderbook_ask_depth",
    "orderbook_bid_depth",
    "orderbook_imbalance",
    "price_reconciliation_warning",
    "price_vs_sma20",
    "price_vs_sma50",
    "returns_1",
    "returns_20",
    "returns_5",
    "rsi_14",
    "rsi_7",
    "sector_relative_strength",
    "sentiment_data_available",
    "sequence_length",
    "social_mention_count",
    "social_sentiment_avg",
    "spread_pct",
    "stale",
    "stoch_k",
    "ticker_stale",
    "training_quality_reason",
    "training_market_fact_contract",
    "training_label_contract",
    "volatility_20",
    "volume_24h",
    "volume_ratio",
    "whale_txn_count",
)
_TRAINING_FEATURE_COLUMN_PREFIX = "training_feature__"
def _shadow_training_columns() -> tuple[Any, ...]:
    return (
        ShadowBacktest.id,
        ShadowBacktest.decision_id,
        ShadowBacktest.created_at,
        ShadowBacktest.symbol,
        ShadowBacktest.analysis_type,
        ShadowBacktest.decision_action,
        ShadowBacktest.decision_confidence,
        ShadowBacktest.training_feature_snapshot,
        ShadowBacktest.due_at,
        ShadowBacktest.horizon_minutes,
        ShadowBacktest.label_version,
        ShadowBacktest.long_return_pct,
        ShadowBacktest.short_return_pct,
        ShadowBacktest.best_action,
        ShadowBacktest.missed_opportunity,
    )


def _shadow_training_row_from_mapping(mapping: Any) -> ShadowTrainingRow:
    feature_snapshot = _parse_json(mapping.get("training_feature_snapshot"))
    return ShadowTrainingRow(
        id=int(mapping.get("id") or 0),
        decision_id=int(mapping.get("decision_id") or 0) or None,
        created_at=mapping.get("created_at"),
        symbol=str(mapping.get("symbol") or ""),
        analysis_type=str(mapping.get("analysis_type") or ""),
        decision_action=str(mapping.get("decision_action") or ""),
        decision_confidence=_safe_float(mapping.get("decision_confidence"), 0.0),
        feature_snapshot=feature_snapshot,
        due_at=mapping.get("due_at"),
        horizon_minutes=int(mapping.get("horizon_minutes") or 10),
        label_version=str(mapping.get("label_version") or ""),
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


def _shadow_is_trainable_trade_opportunity(row: Any) -> bool:
    action = _shadow_action(row, "decision_action")
    best_action = _shadow_action(row, "best_action")
    if action in {"long", "short"}:
        return not assess_shadow_sample(_shadow_quality_sample(row)).exclude_from_training
    missed = bool(getattr(row, "missed_opportunity", False)) and best_action in {"long", "short"}
    if not missed:
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


def select_shadow_training_rows(rows: list[Any]) -> list[Any]:
    """Select the latest quality-governed chronological training window."""

    deduped: dict[Any, Any] = {}
    for row in rows:
        deduped.setdefault(_shadow_row_id(row), row)
    recent = sorted(deduped.values(), key=_shadow_sort_key, reverse=True)
    trainable_rows = [row for row in recent if _shadow_is_trainable_trade_opportunity(row)]

    return trainable_rows


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
    missed_count = (
        int(frame.get("missed_opportunity", pd.Series([], dtype=bool)).astype(bool).sum())
        if sample_count
        else 0
    )
    directional_count = (
        int(frame.get("decision_action", pd.Series([], dtype=str)).isin(["long", "short"]).sum())
        if sample_count
        else 0
    )
    return {
        "sample_count": sample_count,
        "decision_action_counts": counts("decision_action"),
        "best_action_counts": counts("best_action"),
        "data_quality_status_counts": counts("data_quality_status"),
        "directional_decision_count": directional_count,
        "missed_opportunity_count": missed_count,
        "missed_opportunity_share": round(missed_count / max(sample_count, 1), 4),
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
            "id": int(getattr(row, "id", 0) or 0),
            "decision_id": int(getattr(row, "decision_id", 0) or 0) or None,
            "label_version": str(getattr(row, "label_version", "") or ""),
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
        horizon_minutes = int(getattr(row, "horizon_minutes", 10) or 10)
        cost_complete_returns = _cost_complete_shadow_returns(
            snapshot,
            horizon_minutes=horizon_minutes,
            long_gross_return_pct=_safe_float(raw_long_return),
            short_gross_return_pct=_safe_float(raw_short_return),
        )
        if cost_complete_returns is None:
            continue
        long_return, short_return, execution_cost = cost_complete_returns
        feature_row: dict[str, Any] = dict(
            _feature_row_from_snapshot(
                snapshot,
                decision_confidence=_safe_float(getattr(row, "decision_confidence", 0.0)),
                horizon_minutes=horizon_minutes,
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
                "execution_cost": execution_cost,
                "sample_weight": assessment.weight,
                "data_quality_status": assessment.status,
                "data_quality_score": assessment.score,
                "quality_reasons": list(assessment.reasons),
            }
        )
        data.append(feature_row)
    frame = pd.DataFrame(data)
    if frame.empty:
        return frame
    tail_policy: dict[str, Any] = {}
    for side in ("long", "short"):
        returns = frame[f"{side}_return_pct"].astype(float)
        boundary = empirical_policy_value(
            f"{side}_tail_loss_boundary_pct",
            returns[returns < 0].tolist(),
            selector="lower_hinge",
            observation_window="current_cost_complete_training_window",
        )
        threshold = float(boundary.value) if boundary.value is not None else 0.0
        frame[f"{side}_tail_loss"] = (returns < threshold).astype(int)
        frame[f"{side}_win"] = (returns > 0.0).astype(int)
        tail_policy[side] = boundary.to_dict()
    frame.attrs["tail_loss_policy"] = tail_policy
    return frame


def shadow_training_quality_report(rows: list[Any]) -> dict[str, Any]:
    """Assess all candidate shadow rows, including rows excluded before fitting."""

    samples: list[dict[str, Any]] = []
    for row in rows:
        snapshot = _parse_json(getattr(row, "feature_snapshot", None))
        raw_long_return = getattr(row, "long_return_pct", None)
        raw_short_return = getattr(row, "short_return_pct", None)
        sample = {
            "id": int(getattr(row, "id", 0) or 0),
            "decision_id": int(getattr(row, "decision_id", 0) or 0) or None,
            "label_version": str(getattr(row, "label_version", "") or ""),
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
    completed_sample_count: int | None = None,
    training_quality_report: dict[str, Any] | None = None,
    persist_artifact: bool = True,
) -> dict[str, Any]:
    if len(frame) <= 1:
        raise ValueError("训练收益分布不足，无法形成非空训练集和留出集")

    tail_policy = dict(frame.attrs.get("tail_loss_policy") or {})
    frame = frame.sort_values("id").reset_index(drop=True)
    tail_scales: dict[str, float] = {}
    for side in ("long", "short"):
        boundary = _safe_float(_safe_dict(tail_policy.get(side)).get("value"), float("nan"))
        if not math.isfinite(boundary):
            negatives = frame.loc[
                frame[f"{side}_return_pct"].astype(float) < 0,
                f"{side}_return_pct",
            ].tolist()
            generated = empirical_policy_value(
                f"{side}_tail_loss_boundary_pct",
                negatives,
                selector="lower_hinge",
                observation_window="current_cost_complete_training_window",
            )
            tail_policy[side] = generated.to_dict()
            boundary = float(generated.value) if generated.value is not None else 0.0
        frame[f"{side}_tail_loss"] = (
            frame[f"{side}_return_pct"].astype(float) < boundary
        ).astype(int)
        frame[f"{side}_win"] = (frame[f"{side}_return_pct"].astype(float) > 0.0).astype(int)
        tail_scales[side] = max(abs(boundary), 1e-12)
    split = len(frame) // 2

    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    x_train = train[FEATURE_KEYS]
    x_test = test[FEATURE_KEYS]
    train_weights = train.get("sample_weight", pd.Series([1.0] * len(train))).astype(float)

    long_classifier = _make_classifier(train["long_win"])
    short_classifier = _make_classifier(train["short_win"])
    long_tail_classifier = _make_classifier(train["long_tail_loss"])
    short_tail_classifier = _make_classifier(train["short_tail_loss"])
    long_regressor = _make_regressor(train["long_return_pct"])
    short_regressor = _make_regressor(train["short_return_pct"])

    long_classifier.fit(x_train, train["long_win"], model__sample_weight=train_weights)
    short_classifier.fit(x_train, train["short_win"], model__sample_weight=train_weights)
    long_tail_classifier.fit(
        x_train, train["long_tail_loss"], model__sample_weight=train_weights
    )
    short_tail_classifier.fit(
        x_train, train["short_tail_loss"], model__sample_weight=train_weights
    )
    long_regressor.fit(x_train, train["long_return_pct"], model__sample_weight=train_weights)
    short_regressor.fit(x_train, train["short_return_pct"], model__sample_weight=train_weights)

    long_scores = _positive_proba(long_classifier, x_test)
    short_scores = _positive_proba(short_classifier, x_test)
    long_tail_scores = _positive_proba(long_tail_classifier, x_test)
    short_tail_scores = _positive_proba(short_tail_classifier, x_test)
    long_distribution = _regression_prediction_distribution(long_regressor, x_test)
    short_distribution = _regression_prediction_distribution(short_regressor, x_test)
    long_expected_scores = _risk_adjusted_expected_scores(
        long_distribution,
        long_tail_scores,
        tail_loss_scale_pct=tail_scales["long"],
    )
    short_expected_scores = _risk_adjusted_expected_scores(
        short_distribution,
        short_tail_scores,
        tail_loss_scale_pct=tail_scales["short"],
    )
    return_buckets = {
        "long": {
            "top": _bucket_return_summary(
                test["long_return_pct"],
                long_expected_scores,
                top=True,
                tail_loss_threshold_pct=tail_scales["long"],
            ),
            "bottom": _bucket_return_summary(
                test["long_return_pct"],
                long_expected_scores,
                top=False,
                tail_loss_threshold_pct=tail_scales["long"],
            ),
        },
        "short": {
            "top": _bucket_return_summary(
                test["short_return_pct"],
                short_expected_scores,
                top=True,
                tail_loss_threshold_pct=tail_scales["short"],
            ),
            "bottom": _bucket_return_summary(
                test["short_return_pct"],
                short_expected_scores,
                top=False,
                tail_loss_threshold_pct=tail_scales["short"],
            ),
        },
    }

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
        "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "phase": "phase3_model_factory",
        "version": now,
        "trained_at": now,
        "sample_count": int(len(frame)),
        "completed_shadow_sample_count": completed_count,
        "phase3_clean_completed_shadow_sample_count": completed_count,
        "last_trained_completed_shadow_sample_count": completed_count,
        "last_trained_phase3_shadow_sample_count": completed_count,
        "training_shadow_sample_count": int(len(frame)),
        "training_window_composition": _training_window_composition(frame),
        "quality_report": frame_quality_report,
        "market_fact_contract": _safe_dict(
            frame_quality_report.get("market_fact_contract")
        ),
        "governance_report": artifact_bound_governance_report(
            frame_quality_report,
            persist_artifact=persist_artifact,
        ),
        "training_window_policy": "all_current_clean_cost_complete_samples",
        "training_cursor_note": "last_trained_completed_shadow_sample_count is the cumulative cursor used for auto-training.",
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "feature_count": len(FEATURE_KEYS),
        "horizons": sorted(int(v) for v in frame["horizon_minutes"].dropna().unique().tolist()),
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "positive_net_return_boundary_pct": 0.0,
        "positive_return_boundary_policy": "fee_after_profitability_math_boundary",
        "tail_loss_policy": tail_policy,
        "tail_loss_scale_pct": tail_scales,
        "training_cost_policy": "per_sample_live_spread_fee_and_funding_complete",
        "prediction_distribution": {
            "lower_bound": "tree_prediction_lower_hinge",
            "uncertainty_source": "random_forest_tree_empirical_order_statistics",
            "tail_risk_source": "tail_loss_classifier_diagnostic_risk_penalty",
        },
        "training_objective": (
            "Directly regress executable fee-after net return by side. Rank with the "
            "conditional expected return minus model-disagreement and tail-loss "
            "penalties. Classification metrics are diagnostics only."
        ),
        "metrics": {
            "long_auc": _safe_auc(test["long_win"], long_scores),
            "short_auc": _safe_auc(test["short_win"], short_scores),
            "long_pr_auc": _safe_pr_auc(test["long_win"], long_scores),
            "short_pr_auc": _safe_pr_auc(test["short_win"], short_scores),
            "top_long_avg_return_pct": return_buckets["long"]["top"]["avg_return_pct"],
            "bottom_long_avg_return_pct": return_buckets["long"]["bottom"]["avg_return_pct"],
            "top_long_median_return_pct": return_buckets["long"]["top"]["median_return_pct"],
            "top_long_return_lcb_pct": return_buckets["long"]["top"]["return_lcb_pct"],
            "top_long_profit_factor": return_buckets["long"]["top"]["profit_factor"],
            "top_long_cvar_10_pct": return_buckets["long"]["top"]["cvar_10_pct"],
            "top_long_win_rate": _bucket_win_rate(test["long_win"], long_scores, top=True),
            "bottom_long_win_rate": _bucket_win_rate(test["long_win"], long_scores, top=False),
            "top_long_tail_loss_rate": return_buckets["long"]["top"]["tail_loss_rate"],
            "bottom_long_tail_loss_rate": return_buckets["long"]["bottom"]["tail_loss_rate"],
            "top_short_avg_return_pct": return_buckets["short"]["top"]["avg_return_pct"],
            "bottom_short_avg_return_pct": return_buckets["short"]["bottom"]["avg_return_pct"],
            "top_short_median_return_pct": return_buckets["short"]["top"]["median_return_pct"],
            "top_short_return_lcb_pct": return_buckets["short"]["top"]["return_lcb_pct"],
            "top_short_profit_factor": return_buckets["short"]["top"]["profit_factor"],
            "top_short_cvar_10_pct": return_buckets["short"]["top"]["cvar_10_pct"],
            "top_short_win_rate": _bucket_win_rate(test["short_win"], short_scores, top=True),
            "bottom_short_win_rate": _bucket_win_rate(test["short_win"], short_scores, top=False),
            "top_short_tail_loss_rate": return_buckets["short"]["top"]["tail_loss_rate"],
            "bottom_short_tail_loss_rate": return_buckets["short"]["bottom"]["tail_loss_rate"],
        },
        "score_bucket_diagnostics": _score_bucket_diagnostics(
            test,
            long_expected_scores=long_expected_scores,
            short_expected_scores=short_expected_scores,
        ),
        "feature_keys": FEATURE_KEYS,
        "mode": "entry_profit_filter",
        "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
        "trade_sample_cursor_policy": PHASE3_REQUIRED_TRAINING_POLICY,
        "training_mode": "walk_forward",
        "model_stage": "shadow",
        "evaluation_policy": {
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
            "requires_walk_forward": True,
            "phase": "phase3_model_factory",
        },
        "training_run_mode": "persist" if persist_artifact else "dry_run",
        "artifact_persisted": bool(persist_artifact),
        "note": "本地 ML 直接优化费后预期收益及左尾风险；胜率仅作为诊断，不参与开仓、评分、权重或晋升。",
    }

    bundle = {
        "long_classifier": long_classifier,
        "short_classifier": short_classifier,
        "long_tail_classifier": long_tail_classifier,
        "short_tail_classifier": short_tail_classifier,
        "long_regressor": long_regressor,
        "short_regressor": short_regressor,
        "metadata": metadata,
        "feature_keys": FEATURE_KEYS,
    }
    if persist_artifact:
        if MODEL_PATH != MODEL_DIR / "net_return_model.joblib" or METADATA_PATH != (
            MODEL_DIR / "net_return_model_metadata.json"
        ):
            dump_trusted_joblib(bundle, MODEL_PATH, trusted_root=MODEL_DIR)
            METADATA_PATH.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            resolved = ML_SIGNAL_ARTIFACT_REGISTRY.persist_joblib(
                bundle,
                metadata,
                parent_model_identity=(
                    "sklearn RandomForest/Dummy classifier-regressor pipelines"
                ),
                code_version=_training_source_code_version(),
            )
            metadata.clear()
            metadata.update(resolved.manifest)
    return metadata


class MLSignalService:
    """Lazy loader and inference wrapper for the local profit-quality model."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        artifact_registry: ModelArtifactRegistry | None = None,
        training_state_store: ModelTrainingStateStore | None = None,
    ) -> None:
        self._explicit_model_path = model_path
        self.artifact_registry = artifact_registry or ML_SIGNAL_ARTIFACT_REGISTRY
        self.training_state_store = training_state_store or MODEL_TRAINING_STATE_STORE
        self.model_path = model_path or (
            self.artifact_registry.model_root / "unregistered-model.joblib"
        )
        self.metadata_path = METADATA_PATH if model_path is not None else (
            self.artifact_registry.model_root / "unregistered-metadata.json"
        )
        self._bundle: dict[str, Any] | None = None
        self._loaded_mtime: float | None = None
        self._loaded_pointer_mtime_ns: int | None = None
        self._resolved_artifact: ResolvedModelArtifact | None = None
        self._train_lock = asyncio.Lock()
        self._training = False
        self._last_check_at: str | None = None
        self._next_check_at: str | None = None
        self._last_train_started_at: str | None = None
        self._last_train_finished_at: str | None = None
        self._last_train_result: dict[str, Any] | None = None
        self._active_training_run_id: str | None = None

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
                "artifact_registry": self._artifact_registry_status(),
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
        phase3_counts = self._phase3_sample_count_status(metadata)
        return {
            "available": True,
            "model_path": str(self.model_path),
            "artifact_registry": self._artifact_registry_status(),
            **metadata,
            "training_shadow_sample_count": int(
                metadata.get("training_shadow_sample_count") or training_count
            ),
            "training_window_policy": metadata.get("training_window_policy")
            or "all_current_clean_cost_complete_samples",
            **phase3_counts,
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

    @staticmethod
    def _phase3_cursor_from_metadata(metadata: dict[str, Any], completed_count: int) -> int:
        """Return a trained cursor on the current Phase 3 clean-sample scale."""

        value = metadata.get("last_trained_completed_shadow_sample_count")
        try:
            cursor = int(value)
        except (TypeError, ValueError):
            return 0
        return cursor if 0 <= cursor <= completed_count else 0

    def _phase3_sample_count_status(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Expose authoritative clean-training counters without inferred fallbacks."""

        try:
            completed_count = int(metadata.get("phase3_clean_completed_shadow_sample_count") or 0)
        except (TypeError, ValueError):
            completed_count = 0
        if completed_count <= 0:
            try:
                completed_count = int(metadata.get("completed_shadow_sample_count") or 0)
            except (TypeError, ValueError):
                completed_count = 0
        completed_count = max(completed_count, 0)
        trained_cursor = self._phase3_cursor_from_metadata(metadata, completed_count)
        new_count = max(completed_count - trained_cursor, 0)
        return {
            "phase3_clean_completed_shadow_sample_count": completed_count,
            "phase3_clean_trainable_shadow_sample_count": completed_count,
            "last_trained_phase3_shadow_sample_count": trained_cursor,
            "phase3_new_shadow_sample_count": new_count,
            "new_shadow_sample_count": new_count,
            "phase3_sample_cursor_policy": "phase3_clean_training_view_only",
        }

    async def maybe_auto_train(self, *, force: bool = False) -> dict[str, Any]:
        """Run one cross-process single-flight training check."""

        lease_attempt = self.training_state_store.try_acquire_lease(
            scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
            stale_after_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
        )
        if not lease_attempt.acquired or lease_attempt.lease is None:
            return {
                "trained": False,
                "reason": lease_attempt.reason,
                "recovered_stale_lease": lease_attempt.recovered_stale_lease,
            }
        lease = lease_attempt.lease
        self._active_training_run_id = lease.run_id
        now = datetime.now(UTC)
        try:
            self.training_state_store.heartbeat(
                scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
                model_ids=LOCAL_ML_MODEL_IDS,
                interval_seconds=AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
            )
            self.training_state_store.record_check(
                scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
                model_ids=LOCAL_ML_MODEL_IDS,
                run_id=lease.run_id,
                force=force,
            )
        except Exception:
            self._active_training_run_id = None
            lease.release()
            raise
        try:
            result = await self._maybe_auto_train_process(force=force)
            failed = str(result.get("reason") or "") in {
                "error",
                "load_samples_error",
                "timeout",
            }
            delay = (
                AUTO_TRAIN_RETRY_INTERVAL_SECONDS
                if failed
                else AUTO_TRAIN_CHECK_INTERVAL_SECONDS
            )
            next_check = datetime.now(UTC) + timedelta(seconds=delay)
            self.training_state_store.finish_check(
                scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
                model_ids=LOCAL_ML_MODEL_IDS,
                run_id=lease.run_id,
                result=result,
                next_check_at=next_check,
            )
            return result
        except asyncio.CancelledError:
            self.training_state_store.record_exception(
                scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
                model_ids=LOCAL_ML_MODEL_IDS,
                run_id=lease.run_id,
                error="training_cancelled",
                next_check_at=now + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
            )
            raise
        except Exception as exc:
            error = safe_error_text(exc, limit=180)
            self.training_state_store.record_exception(
                scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
                model_ids=LOCAL_ML_MODEL_IDS,
                run_id=lease.run_id,
                error=error,
                next_check_at=now + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
            )
            raise
        finally:
            self._active_training_run_id = None
            lease.release()

    async def _maybe_auto_train_process(self, *, force: bool = False) -> dict[str, Any]:
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
                last_completed_count = self._phase3_cursor_from_metadata(
                    metadata,
                    completed_count,
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
                new_samples = max(completed_count - last_completed_count, 0)
                training_policy = {
                    "learning_only": learning_only,
                    "readiness_state": readiness.get("state"),
                    "readiness_blocking_reasons": readiness.get("blocking_reasons") or [],
                    "trigger": "new_cost_complete_authoritative_sample_or_forced_rebuild",
                    "cursor_source": "phase3_clean_training_view",
                    "promotion_requires_readiness": True,
                    "candidate_artifact_persisted": False,
                    "persist_artifact_only_when_readiness_allows_live_influence": False,
                    "persist_latest_artifact_even_when_readiness_blocks_live_influence": True,
                }
                if completed_count <= 1:
                    result = {
                        "trained": False,
                        "reason": "training_distribution_unavailable",
                        "completed_sample_count": completed_count,
                        "last_trained_sample_count": last_sample_count,
                        "last_trained_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "training_policy": training_policy,
                        "message": "本地 ML 尚无法形成非空训练集和留出集，继续收集成本完整样本。",
                    }
                    self._last_train_result = result
                    return result

                should_train = force or not metadata or new_samples > 0
                if not should_train:
                    result = {
                        "trained": False,
                        "reason": "not_due",
                        "completed_sample_count": completed_count,
                        "last_trained_sample_count": last_sample_count,
                        "last_trained_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "training_policy": training_policy,
                        "message": "没有新增成本完整权威样本，当前 artifact 无需重复训练。",
                    }
                    self._last_train_result = result
                    return result

                self._training = True
                self._last_train_started_at = datetime.now(UTC).isoformat()
                if self._active_training_run_id:
                    self.training_state_store.start_run(
                        scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
                        model_ids=LOCAL_ML_MODEL_IDS,
                        run_id=self._active_training_run_id,
                        trigger_reason="forced" if force else "training_due",
                        sample_cursor={"shadow": completed_count},
                        timeout_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
                    )
                quarantine_result = await self._quarantine_dirty_training_samples()
                completed_count = await self._completed_shadow_sample_count()
                new_samples = max(completed_count - last_completed_count, 0)
                if completed_count <= 1:
                    result = {
                        "trained": False,
                        "reason": "clean_training_distribution_unavailable",
                        "completed_sample_count": completed_count,
                        "last_trained_sample_count": last_sample_count,
                        "last_trained_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "training_policy": training_policy,
                        "training_quarantine": quarantine_result,
                        "message": "自动隔离后无法形成非空训练集和留出集，继续累计成本完整样本。",
                    }
                    self._last_train_result = result
                    return result
                rows = await load_shadow_training_rows()
                quality_state = shadow_training_quality_report(rows)
                frame = build_training_frame(rows)
                if len(frame) <= 1:
                    result = {
                        "trained": False,
                        "reason": "cost_complete_training_distribution_unavailable",
                        "completed_sample_count": completed_count,
                        "cost_complete_sample_count": int(len(frame)),
                        "training_policy": training_policy,
                        "training_quarantine": quarantine_result,
                        "message": "成本完整样本无法形成训练集和留出集，本轮不训练也不回退旧成本。",
                    }
                    self._last_train_result = result
                    return result
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
                trained_metadata = await asyncio.to_thread(
                    train_from_frame,
                    frame,
                    completed_sample_count=completed_count,
                    training_quality_report=quality_state["quality_report"],
                    persist_artifact=True,
                )
                trained_influence = _influence_policy(trained_metadata)
                trained_readiness = build_ml_readiness_report(
                    trained_metadata,
                    trained_influence,
                )
                self._bundle = None
                self._loaded_mtime = None
                self._ensure_loaded()
                allow_live_position_influence = bool(
                    trained_readiness.get("allow_live_position_influence")
                )
                result = {
                    "trained": True,
                    "reason": (
                        "trained" if allow_live_position_influence else "trained_learning_only"
                    ),
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
                    "readiness": trained_readiness,
                    "readiness_state": trained_readiness.get("state"),
                    "allow_live_position_influence": allow_live_position_influence,
                    "influence_enabled": allow_live_position_influence,
                    "influence_policy": trained_influence,
                    "artifact_persisted": bool(trained_metadata.get("artifact_persisted")),
                    "trained_at": trained_metadata.get("trained_at"),
                    "message": (
                        "本地 ML 盈亏质量模型已自动完成训练、替换为最新 artifact 并热加载；"
                        "当前已允许参与开仓过滤与收益排序。"
                        if allow_live_position_influence
                        else (
                            "本地 ML 盈亏质量模型已自动完成训练、替换为最新 artifact 并热加载；"
                            "当前仍处于学习观察/降级状态，暂不参与实盘影响。"
                        )
                    ),
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
        tail_scales = _safe_dict(metadata.get("tail_loss_scale_pct"))
        long_tail_scale = max(_safe_float(tail_scales.get("long"), 0.0), 0.0)
        short_tail_scale = max(_safe_float(tail_scales.get("short"), 0.0), 0.0)

        predictions = []
        for horizon in horizons:
            row = _feature_row_from_feature_vector(features, horizon_minutes=horizon)
            x = pd.DataFrame([row], columns=FEATURE_KEYS)
            long_win_rate = float(_positive_proba(self._bundle["long_classifier"], x)[0])
            short_win_rate = float(_positive_proba(self._bundle["short_classifier"], x)[0])
            long_distribution = _regression_prediction_distribution(
                self._bundle["long_regressor"], x
            )
            short_distribution = _regression_prediction_distribution(
                self._bundle["short_regressor"], x
            )
            raw_long_expected = float(long_distribution["expected"][0])
            raw_short_expected = float(short_distribution["expected"][0])
            long_lower_quantile = float(long_distribution["lower_quantile"][0])
            short_lower_quantile = float(short_distribution["lower_quantile"][0])
            long_tail_model = self._bundle.get("long_tail_classifier")
            short_tail_model = self._bundle.get("short_tail_classifier")
            long_tail_loss_probability = (
                float(
                    _optional_positive_proba(
                        long_tail_model,
                        x,
                        default=0.0,
                    )[0]
                )
                if long_tail_model is not None
                else None
            )
            short_tail_loss_probability = (
                float(
                    _optional_positive_proba(
                        short_tail_model,
                        x,
                        default=0.0,
                    )[0]
                )
                if short_tail_model is not None
                else None
            )
            long_objective = risk_adjusted_expected_return(
                expected_return_pct=raw_long_expected,
                lower_quantile_return_pct=long_lower_quantile,
                tail_loss_probability=long_tail_loss_probability,
                tail_loss_scale_pct=long_tail_scale,
            )
            short_objective = risk_adjusted_expected_return(
                expected_return_pct=raw_short_expected,
                lower_quantile_return_pct=short_lower_quantile,
                tail_loss_probability=short_tail_loss_probability,
                tail_loss_scale_pct=short_tail_scale,
            )
            long_expected = long_objective["objective_net_return_pct"]
            short_expected = short_objective["objective_net_return_pct"]
            best_side = "long" if long_expected >= short_expected else "short"
            best_win = long_win_rate if best_side == "long" else short_win_rate
            best_expected = long_expected if best_side == "long" else short_expected
            best_tail_loss_probability = (
                long_tail_loss_probability
                if best_side == "long"
                else short_tail_loss_probability
            )
            best_lower_quantile = (
                long_lower_quantile if best_side == "long" else short_lower_quantile
            )
            profit_edge = abs(long_expected - short_expected)
            profit_quality = _profit_quality_score(
                best_expected,
                best_lower_quantile,
                profit_edge,
                float(best_tail_loss_probability or 0.0),
                long_tail_scale if best_side == "long" else short_tail_scale,
            )
            side_influence = _safe_dict(influence.get(best_side))
            downside = max(-best_expected, 0.0) + max(-best_lower_quantile, 0.0)
            return_scale = abs(best_expected) + abs(best_lower_quantile)
            risk_score = _clamp(
                downside / max(return_scale, 1e-9)
                + float(best_tail_loss_probability or 0.0)
            )
            predictions.append(
                {
                    "horizon_minutes": int(horizon),
                    "long_win_rate": round(long_win_rate, 4),
                    "short_win_rate": round(short_win_rate, 4),
                    "long_tail_loss_probability": (
                        None
                        if long_tail_loss_probability is None
                        else round(long_tail_loss_probability, 4)
                    ),
                    "short_tail_loss_probability": (
                        None
                        if short_tail_loss_probability is None
                        else round(short_tail_loss_probability, 4)
                    ),
                    "tail_loss_threshold_pct": round(
                        long_tail_scale if best_side == "long" else short_tail_scale,
                        4,
                    ),
                    "long_raw_expected_return_pct": round(raw_long_expected, 4),
                    "short_raw_expected_return_pct": round(raw_short_expected, 4),
                    "long_lower_quantile_return_pct": round(long_lower_quantile, 4),
                    "short_lower_quantile_return_pct": round(short_lower_quantile, 4),
                    "long_uncertainty_penalty_pct": round(
                        long_objective["uncertainty_penalty_pct"], 4
                    ),
                    "short_uncertainty_penalty_pct": round(
                        short_objective["uncertainty_penalty_pct"], 4
                    ),
                    "long_tail_loss_penalty_pct": round(
                        long_objective["tail_loss_penalty_pct"], 4
                    ),
                    "short_tail_loss_penalty_pct": round(
                        short_objective["tail_loss_penalty_pct"], 4
                    ),
                    "long_expected_return_pct": round(long_expected, 4),
                    "short_expected_return_pct": round(short_expected, 4),
                    "best_side": best_side,
                    "best_win_rate": round(best_win, 4),
                    "best_tail_loss_probability": (
                        None
                        if best_tail_loss_probability is None
                        else round(best_tail_loss_probability, 4)
                    ),
                    "best_expected_return_pct": round(best_expected, 4),
                    "profit_edge_pct": round(profit_edge, 4),
                    "profit_quality_score": round(profit_quality, 4),
                    "profit_signal": bool(
                        allow_live_position_influence
                        and side_influence.get("enabled")
                        and best_expected > 0.0
                        and best_lower_quantile > 0.0
                        and profit_edge > 0.0
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
            trusted_root = MODEL_DIR
            if self._explicit_model_path is None:
                pointer_mtime_ns = (
                    self.artifact_registry.current_path.stat().st_mtime_ns
                    if self.artifact_registry.current_path.exists()
                    else None
                )
                if (
                    self._bundle is not None
                    and self._resolved_artifact is not None
                    and self._loaded_pointer_mtime_ns == pointer_mtime_ns
                    and self.model_path.exists()
                    and self._loaded_mtime == self.model_path.stat().st_mtime
                ):
                    return
                current = self.artifact_registry.resolve_current()
                if current is None:
                    self._bundle = None
                    self._loaded_mtime = None
                    self._loaded_pointer_mtime_ns = pointer_mtime_ns
                    self._resolved_artifact = None
                    return
                self.model_path = current.model_path
                self.metadata_path = current.metadata_path
                trusted_root = self.artifact_registry.model_root
                self._loaded_pointer_mtime_ns = pointer_mtime_ns
                self._resolved_artifact = current
            if not self.model_path.exists():
                self._bundle = None
                self._loaded_mtime = None
                return
            mtime = self.model_path.stat().st_mtime
            if self._bundle is not None and self._loaded_mtime == mtime:
                return
            self._bundle = load_trusted_joblib(
                self.model_path,
                trusted_root=trusted_root,
                expected_type=dict,
            )
            metadata = _safe_dict(self._bundle.get("metadata"))
            if (
                metadata.get("objective_name") != RETURN_OBJECTIVE_NAME
                or metadata.get("objective_version") != RETURN_OBJECTIVE_VERSION
                or metadata.get("label_version") != RETURN_LABEL_VERSION
            ):
                raise ValueError(
                    "refusing local ML artifact with legacy or unknown return objective"
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
            self._loaded_pointer_mtime_ns = None
            self._resolved_artifact = None

    def _artifact_registry_status(self) -> dict[str, Any]:
        current = self._resolved_artifact
        if current is None:
            return self.artifact_registry.status()
        return {
            "available": True,
            "model_id": current.model_id,
            "registry_version": current.manifest.get("artifact_registry_version"),
            "version": current.version,
            "model_path": str(current.model_path),
            "manifest_path": str(current.manifest_path),
            "sha256": current.sha256,
            "manifest": current.manifest,
        }

    def _auto_train_status(self) -> dict[str, Any]:
        persistent = self.training_state_store.read()
        models = persistent.get("models") if isinstance(persistent.get("models"), dict) else {}
        row = models.get(LOCAL_ML_MODEL_IDS[0]) if isinstance(models, dict) else {}
        row = row if isinstance(row, dict) else {}
        return {
            "auto_train_enabled": True,
            "auto_train_check_interval_seconds": AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
            "auto_train_trigger": "new_cost_complete_authoritative_sample_or_forced_rebuild",
            "auto_train_distribution_requirement": (
                "non_empty_train_and_holdout_from_cost_complete_samples"
            ),
            "auto_training": row.get("state") == "running",
            "auto_train_last_check_at": row.get("last_check_at") or self._last_check_at,
            "auto_train_next_check_at": row.get("next_check_at") or self._next_check_at,
            "auto_train_last_started_at": row.get("last_started_at")
            or self._last_train_started_at,
            "auto_train_last_finished_at": row.get("last_finished_at")
            or self._last_train_finished_at,
            "auto_train_last_result": row.get("last_result") or self._last_train_result,
            "auto_train_persistent_state": row,
            "model_training_scheduler_state": persistent,
        }

    def _current_metadata(self) -> dict[str, Any]:
        self._ensure_loaded()
        if self._bundle:
            metadata = self._bundle.get("metadata") or {}
            if isinstance(metadata, dict):
                return metadata
        try:
            if self.metadata_path.exists():
                parsed = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.debug(
                "failed to read ML signal metadata",
                path=str(self.metadata_path),
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
        expected = float(primary.get("best_expected_return_pct") or 0.0)
        edge = float(primary.get("profit_edge_pct") or 0.0)
        lower_quantile = float(
            primary.get(f"{primary.get('best_side')}_lower_quantile_return_pct") or 0.0
        )
        tail_probability = float(primary.get("best_tail_loss_probability") or 0.0)
        side = "做多" if primary.get("best_side") == "long" else "做空"
        if expected > 0.0 and edge > 0.0 and lower_quantile > 0.0:
            return f"ML 费后收益分布支持{side}，可作为开仓收益质量证据。"
        if expected <= 0:
            return "ML 风险调整后的费后预期收益为负，应阻止该方向获得模型加分。"
        if lower_quantile <= 0:
            return "ML 平均费后收益为正但置信下界未转正，继续 shadow 验证。"
        if tail_probability * max(abs(lower_quantile), 0.0) >= max(expected, 0.0):
            return "ML 费后收益为正但动态左尾损失预算已覆盖预期收益，不能晋升或放大风险。"
        if edge <= 0.0:
            return "ML 多空预期收益差距不明显，信号中性。"
        return "ML 盈亏质量信号中性，暂不改变 AI 决策。"


async def load_shadow_training_rows() -> list[Any]:
    base_filters = (
        ShadowBacktest.status == "completed",
        ShadowBacktest.created_at >= PHASE3_CLEAN_START_UTC,
        ShadowBacktest.long_return_pct.is_not(None),
        ShadowBacktest.short_return_pct.is_not(None),
        or_(
            ShadowBacktest.decision_action.in_(["long", "short"]),
            and_(
                ShadowBacktest.missed_opportunity.is_(True),
                ShadowBacktest.best_action.in_(["long", "short"]),
            ),
        ),
    )
    order_by = (ShadowBacktest.created_at.desc(), ShadowBacktest.id.desc())
    columns = _shadow_training_columns()

    async with get_read_session_ctx() as session:
        stmt = select(*columns).where(*base_filters).order_by(*order_by)
        result = await session.execute(stmt)
        rows = [
            _shadow_training_row_from_mapping(row)
            for row in result.mappings().all()
        ]
    return select_shadow_training_rows(rows)


async def count_shadow_training_rows() -> int:
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(func.count(ShadowBacktest.id)).where(
                ShadowBacktest.status == "completed",
                ShadowBacktest.created_at >= PHASE3_CLEAN_START_UTC,
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
        "tail_loss_rate": (
            None
            if bucket.empty or f"{side}_tail_loss" not in bucket
            else float(bucket[f"{side}_tail_loss"].mean())
        ),
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
