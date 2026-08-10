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
from sklearn.linear_model import LogisticRegression, Ridge
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
)
from services.dynamic_policy_values import empirical_policy_value
from services.ml_readiness import build_ml_readiness_report, disabled_ml_readiness
from services.ml_training_contract import (
    DECISION_GROUP_PARTITION_VERSION,
    MIN_TRAINING_DECISION_GROUP_COUNT,
    MIN_TRAINING_SAMPLE_COUNT,
    RANDOM_FOREST_MIN_SAMPLES_LEAF,
    decision_group_partition_errors,
)
from services.model_artifact_registry import ModelArtifactRegistry, ResolvedModelArtifact
from services.model_champion_policy import compare_candidate_to_champion
from services.model_strategy_blueprint import build_model_strategy_blueprint
from services.model_training_state import (
    LOCAL_ML_MODEL_IDS,
    ModelTrainingStateStore,
)
from services.profit_supervision import (
    AUTHORITATIVE_REALIZED_RETURN_TASK,
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    MARKET_OPPORTUNITY_TASK,
    PROFIT_SUPERVISION_VERSION,
    authoritative_trade_calibration,
    profit_supervision_report,
    select_trade_calibration,
)
from services.profit_training_contract import PROFIT_TRAINING_TARGET
from services.return_objective import (
    COST_MODEL_VERSION,
    RETURN_DISTRIBUTION_CONTRACT_VERSION,
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
    return_distribution_summary,
    risk_adjusted_expected_return,
    standardized_return_distribution,
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
from services.training_epoch import (
    CURRENT_TRAINING_EPOCH_POLICY,
    load_training_epoch_start,
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
LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX = "BB_LOCAL_ML_AUTO_TRAIN_RESULT_JSON="
AUTO_TRAIN_RETRY_INTERVAL_SECONDS = 5 * 60
AUTO_TRAIN_LEASE_STALE_SECONDS = 2 * 60 * 60
MIN_ACTIVE_WALK_FORWARD_FOLDS = 2
FEATURE_CONTRACT_VERSION = "2026-07-27.global-point-in-time-features.v1"
MULTITASK_PREDICTION_CONTRACT_VERSION = "2026-07-27.paper-multitask-prediction.v1"
REPLAY_WEIGHT_POLICY_VERSION = "2026-07-27.recency-tail-hard-example.v1"
_TRAINING_CANDIDATE_CACHE: dict[
    int,
    tuple[dict[str, Any], dict[str, Any], str],
] = {}


def _training_source_code_version() -> str:
    digest = hashlib.sha256()
    source_paths = (
        Path(__file__),
        Path(__file__).parents[1] / "core" / "training_contracts.py",
        Path(__file__).with_name("training_data_quality.py"),
        Path(__file__).with_name("ml_readiness.py"),
        Path(__file__).with_name("return_objective.py"),
        Path(__file__).with_name("model_artifact_registry.py"),
        Path(__file__).with_name("trading_params.py"),
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
    "regime_trending",
    "regime_volatile",
    "regime_ranging",
    "liquidity_regime_high",
    "news_shock_present",
    "major_asset",
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


def _market_regime_label(features: dict[str, Any]) -> str:
    explicit = (
        str(
            features.get("market_regime")
            or features.get("regime")
            or features.get("market_state")
            or ""
        )
        .strip()
        .lower()
    )
    if explicit:
        return explicit[:80]
    volatility = abs(_safe_float(features.get("volatility_20")))
    if volatility <= 0:
        price = max(
            _safe_float(features.get("current_price")),
            _safe_float(features.get("close")),
        )
        atr = abs(_safe_float(features.get("atr_14")))
        volatility = atr / price if price > 0 else 0.0
    returns_20 = abs(_safe_float(features.get("returns_20")))
    return "volatile" if volatility >= 0.03 else "trending" if returns_20 >= 0.01 else "ranging"


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
    regime = _market_regime_label(snapshot)
    symbol = str(snapshot.get("symbol") or snapshot.get("instrument") or "").upper()
    base_asset = symbol.split("/")[0].split("-")[0]
    notional_24h = _safe_float(snapshot.get("notional_24h_usdt"), 0.0)
    direct_news_count = _safe_float(snapshot.get("direct_news_item_count"), 0.0)
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
        "regime_trending": 1.0 if regime == "trending" else 0.0,
        "regime_volatile": 1.0 if regime == "volatile" else 0.0,
        "regime_ranging": 1.0 if regime == "ranging" else 0.0,
        "liquidity_regime_high": 1.0 if notional_24h >= 100_000_000.0 else 0.0,
        "news_shock_present": 1.0 if direct_news_count > 0.0 else 0.0,
        "major_asset": 1.0 if base_asset in {"BTC", "ETH", "SOL", "BNB"} else 0.0,
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
    elif len(y) < 250:
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
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
    elif len(y) < 250:
        estimator = Ridge(alpha=1.0)
    else:
        estimator = RandomForestRegressor(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
            random_state=42,
            n_jobs=1,
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


def _configure_single_row_inference(bundle: dict[str, Any]) -> None:
    """Avoid process-wide joblib fan-out for latency-sensitive single-row scoring."""

    for value in bundle.values():
        named_steps = getattr(value, "named_steps", None)
        if not isinstance(named_steps, dict):
            continue
        estimator = named_steps.get("model")
        if estimator is not None and hasattr(estimator, "n_jobs"):
            estimator.n_jobs = 1


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


def _safe_accuracy(y_true: pd.Series, y_score: np.ndarray) -> float | None:
    try:
        truth = pd.Series(y_true).astype(int)
        if not len(truth):
            return None
        predictions = (np.asarray(y_score, dtype=float) >= 0.5).astype(int)
        if len(predictions) != len(truth):
            return None
        return float((truth.to_numpy() == predictions).mean())
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
) -> dict[str, Any]:
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
            "upper_quantile": expected.copy(),
            "std": np.zeros(len(expected), dtype=float),
            "member_count": 0,
            "source_authority": "regressor_point_prediction_without_members",
        }
    transformed = imputer.transform(x)
    tree_predictions = np.asarray([tree.predict(transformed) for tree in trees], dtype=float)
    ordered_tree_predictions = np.sort(tree_predictions, axis=0)
    lower_tail_count = max(int(math.sqrt(len(ordered_tree_predictions))), 1)
    return {
        "expected": expected,
        "median": np.median(tree_predictions, axis=0),
        "lower_quantile": np.median(ordered_tree_predictions[:lower_tail_count], axis=0),
        "upper_quantile": np.median(ordered_tree_predictions[-lower_tail_count:], axis=0),
        "std": np.std(tree_predictions, axis=0),
        "member_count": len(trees),
        "source_authority": "random_forest_tree_empirical_distribution",
    }


def _standardized_model_return_distribution(
    distribution: dict[str, Any],
    index: int,
    *,
    side: str,
    horizon_minutes: int,
    tail_loss_probability: float | None,
    tail_loss_scale_pct: float,
) -> dict[str, Any]:
    return standardized_return_distribution(
        side=side,
        horizon_minutes=horizon_minutes,
        raw_expected_return_pct=distribution["expected"][index],
        median_return_pct=distribution["median"][index],
        lower_quantile_return_pct=distribution["lower_quantile"][index],
        upper_quantile_return_pct=distribution["upper_quantile"][index],
        dispersion_pct=distribution["std"][index],
        tail_loss_probability=tail_loss_probability,
        tail_loss_scale_pct=tail_loss_scale_pct,
        distribution_member_count=distribution.get("member_count"),
        return_semantics="gross_market_opportunity_before_execution",
        source_authority=str(distribution.get("source_authority") or ""),
        objective_version=RETURN_OBJECTIVE_VERSION,
        label_version=RETURN_LABEL_VERSION,
        cost_model_version=COST_MODEL_VERSION,
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )


def _optional_regression_value(
    bundle: dict[str, Any],
    model_key: str,
    x: pd.DataFrame,
    *,
    index: int = 0,
) -> float | None:
    model = bundle.get(model_key)
    if model is None:
        return None
    try:
        value = float(np.asarray(model.predict(x), dtype=float)[index])
    except Exception as exc:
        logger.debug(
            "failed to score optional multitask regression head",
            model_key=model_key,
            error=safe_error_text(exc),
        )
        return None
    return max(value, 0.0) if math.isfinite(value) else None


def _multitask_side_prediction(
    *,
    side: str,
    horizon_minutes: int,
    model_version: str,
    calibration_version: str,
    return_contract: dict[str, Any],
    cost_distribution: dict[str, Any],
    win_probability: float,
    tail_loss_probability: float | None,
    expected_mfe_pct: float | None,
    expected_mae_pct: float | None,
) -> dict[str, Any]:
    gross_expected = _safe_float(return_contract.get("raw_expected_return_pct"), 0.0)
    gross_q10 = _safe_float(return_contract.get("lower_quantile_return_pct"), gross_expected)
    gross_q50 = _safe_float(return_contract.get("median_return_pct"), gross_expected)
    gross_q90 = _safe_float(return_contract.get("upper_quantile_return_pct"), gross_expected)
    cost_expected = max(float(cost_distribution["expected"][0]), 0.0)
    cost_q10 = max(float(cost_distribution["lower_quantile"][0]), 0.0)
    cost_q90 = max(float(cost_distribution["upper_quantile"][0]), 0.0)
    ordered_quantiles = sorted(
        (
            gross_q10 - cost_q90,
            gross_q50 - cost_expected,
            gross_q90 - cost_q10,
        )
    )
    return {
        "version": MULTITASK_PREDICTION_CONTRACT_VERSION,
        "side": side,
        "expected_net_return_pct": gross_expected - cost_expected,
        "return_q10": ordered_quantiles[0],
        "return_q50": ordered_quantiles[1],
        "return_q90": ordered_quantiles[2],
        "loss_probability": 1.0 - _clamp(win_probability),
        "tail_loss_probability": (
            _clamp(tail_loss_probability)
            if tail_loss_probability is not None
            else None
        ),
        "expected_execution_cost_pct": cost_expected,
        "expected_mfe_pct": expected_mfe_pct,
        "expected_mae_pct": expected_mae_pct,
        "prediction_horizon": int(horizon_minutes),
        "model_version": model_version,
        "calibration_version": calibration_version,
        "return_semantics": "net_after_counterfactual_execution_cost",
        "quantile_monotonic": True,
    }


def _actual_calibration_ready(profile: dict[str, Any]) -> bool:
    realized = _safe_dict(profile.get(PROFIT_TRAINING_TARGET))
    slippage = _safe_dict(profile.get("slippage_pct"))
    required_values = (
        realized.get("expected"),
        realized.get("lower_hinge"),
        slippage.get("expected"),
        slippage.get("upper_hinge"),
    )
    return bool(
        int(_safe_float(realized.get("count"), 0.0) or 0) > 0
        and int(_safe_float(slippage.get("count"), 0.0) or 0) > 0
        and all(math.isfinite(_safe_float(value, float("nan"))) for value in required_values)
    )


def _distribution_ready_at(
    distribution: dict[str, np.ndarray],
    index: int,
) -> bool:
    expected = float(distribution["expected"][index])
    lower = float(distribution["lower_quantile"][index])
    upper = float(distribution["upper_quantile"][index])
    std = float(distribution["std"][index])
    numerical_resolution = float(np.finfo(float).eps) * max(
        abs(expected),
        abs(lower),
        abs(upper),
        1.0,
    )
    return bool(
        all(math.isfinite(value) for value in (expected, lower, upper, std))
        and (upper - lower > numerical_resolution or std > numerical_resolution)
    )


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
    if metadata.get("profit_supervision_version") != PROFIT_SUPERVISION_VERSION:
        hard_reasons.append("artifact separated profit supervision contract is missing")
    calibration = _safe_dict(metadata.get("actual_trade_calibration"))
    profiles = _safe_dict(calibration.get("profiles"))
    global_profile = _safe_dict(profiles.get(f"*|{side}"))
    actual_return_distribution = _safe_dict(global_profile.get(PROFIT_TRAINING_TARGET))
    slippage_distribution = _safe_dict(global_profile.get("slippage_pct"))
    if int(actual_return_distribution.get("count") or 0) <= 0:
        hard_reasons.append("authoritative realized return calibration is missing")
    if int(slippage_distribution.get("count") or 0) <= 0:
        hard_reasons.append("authoritative slippage tail calibration is missing")
    if top_return <= bottom_return:
        hard_reasons.append(f"高分组平均收益 {top_return:.3f}% 未优于低分组 {bottom_return:.3f}%")
    if top_return_lcb is None or top_return_lcb <= 0:
        hard_reasons.append("高分组费后收益置信下界未大于 0")
    if top_profit_factor is None or top_profit_factor <= 1.0:
        hard_reasons.append("高分组 Profit Factor 未大于 1")
    if top_tail_loss is None or bottom_tail_loss is None or top_tail_loss > bottom_tail_loss:
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
        "top_profit_factor": (None if top_profit_factor is None else round(top_profit_factor, 4)),
        "top_tail_loss_rate": None if top_tail_loss is None else round(top_tail_loss, 4),
        "bottom_tail_loss_rate": (None if bottom_tail_loss is None else round(bottom_tail_loss, 4)),
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


def _activation_gated_policy(
    influence: dict[str, Any],
    readiness: dict[str, Any],
    artifact: ResolvedModelArtifact | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    activation = _safe_dict(artifact.activation_manifest if artifact is not None else None)
    stage = str(activation.get("activation_stage") or "unregistered")
    activation_blockers = activation.get("blocking_reasons")
    activation_blockers = activation_blockers if isinstance(activation_blockers, list) else []
    manifest_authorized = bool(
        stage in {"canary", "active"}
        and activation.get("live_ml_ready") is True
        and activation.get("readiness_state") in {"ready", "partial_ready"}
        and not activation_blockers
    )
    if manifest_authorized and readiness.get("live_ml_ready") is True:
        live_sides = set(readiness.get("live_enabled_sides") or [])
        effective_influence = {
            **influence,
            "enabled": bool(live_sides),
            "live_enabled_sides": sorted(live_sides),
        }
        for side in ("long", "short"):
            side_policy = _safe_dict(influence.get(side))
            if side_policy:
                side_enabled = bool(side in live_sides and side_policy.get("enabled"))
                effective_influence[side] = {
                    **side_policy,
                    "enabled": side_enabled,
                    "advisory_enabled": False,
                    "influence_weight": 1.0 if side_enabled else 0.0,
                }
        return effective_influence, readiness

    gated_influence = {
        **influence,
        "enabled": False,
        "advisory_enabled": False,
        "influence_weight": 0.0,
        "activation_stage": stage,
        "live_ml_ready": False,
        "ungated_return_evidence_enabled": bool(influence.get("enabled")),
    }
    for side in ("long", "short"):
        side_policy = _safe_dict(influence.get(side))
        if side_policy:
            gated_influence[side] = {
                **side_policy,
                "enabled": False,
                "advisory_enabled": False,
                "influence_weight": 0.0,
            }
    gated_blockers = list(readiness.get("blocking_reasons") or [])
    activation_blocker = {
        "code": (
            "artifact_current_readiness_revalidation_failed"
            if manifest_authorized
            else "artifact_activation_not_production_authorized"
        ),
        "message": (
            "The current artifact no longer passes production readiness revalidation."
            if manifest_authorized
            else "The atomic artifact activation manifest does not authorize production influence."
        ),
        "actual": stage,
        "required": "canary_or_active_activation_with_ready_return_evidence",
    }
    if not any(
        isinstance(item, dict) and item.get("code") == activation_blocker["code"]
        for item in gated_blockers
    ):
        gated_blockers.append(activation_blocker)
    gated_readiness = {
        **readiness,
        "state": (
            "shadow_ready"
            if stage == "shadow" and readiness.get("live_ml_ready")
            else readiness.get("state") or "promotion_blocked"
        ),
        "live_ml_ready": False,
        "live_enabled_sides": [],
        "blocking_reasons": gated_blockers,
        "artifact_activation": activation,
    }
    return gated_influence, gated_readiness


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
        "id": int(getattr(row, "id", 0) or 0),
        "decision_id": int(getattr(row, "decision_id", 0) or 0) or None,
        "label_version": str(getattr(row, "label_version", "") or ""),
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


def _training_distribution_profile(frame: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    columns = (
        "returns_5",
        "returns_20",
        "volatility_20",
        "spread_pct",
        "orderbook_imbalance",
        "long_return_pct",
        "short_return_pct",
        "long_execution_cost_pct",
        "short_execution_cost_pct",
    )
    for column in columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        profile[column] = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return {
        "version": "2026-07-27.training-distribution-profile.v1",
        "features": profile,
    }


def _training_distribution_drift(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    current_features = _safe_dict(current.get("features"))
    previous_features = _safe_dict(_safe_dict(previous).get("features"))
    shifts: dict[str, float] = {}
    for key, current_row_value in current_features.items():
        current_row = _safe_dict(current_row_value)
        previous_row = _safe_dict(previous_features.get(key))
        if not previous_row:
            continue
        current_mean = _safe_float(current_row.get("mean"), float("nan"))
        previous_mean = _safe_float(previous_row.get("mean"), float("nan"))
        scale = max(
            _safe_float(current_row.get("std"), 0.0),
            _safe_float(previous_row.get("std"), 0.0),
            1e-9,
        )
        if math.isfinite(current_mean) and math.isfinite(previous_mean):
            shifts[key] = abs(current_mean - previous_mean) / scale
    maximum_shift = max(shifts.values(), default=0.0)
    threshold = float(_LOCAL_ML_PARAMS.distribution_drift_threshold)
    return {
        "version": "2026-07-27.standardized-mean-shift.v1",
        "detected": bool(shifts and maximum_shift >= threshold),
        "threshold": threshold,
        "maximum_shift": round(maximum_shift, 6),
        "feature_shifts": {key: round(value, 6) for key, value in shifts.items()},
    }


def _apply_training_replay_weights(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    weighted = train.copy()
    if weighted.empty:
        return weighted, {
            "version": REPLAY_WEIGHT_POLICY_VERSION,
            "sample_count": 0,
            "effective_sample_size": 0.0,
        }
    timestamps = pd.to_datetime(weighted["label_timestamp"], utc=True, errors="raise")
    latest = timestamps.max()
    ages_days = (latest - timestamps).dt.total_seconds().clip(lower=0.0) / 86400.0
    half_life_days = max(float(_LOCAL_ML_PARAMS.replay_half_life_days), 1.0)
    minimum_recency = max(
        min(float(_LOCAL_ML_PARAMS.replay_minimum_recency_weight), 1.0),
        0.01,
    )
    recency = np.maximum(np.power(0.5, ages_days / half_life_days), minimum_recency)
    base = pd.to_numeric(
        weighted.get("sample_weight", pd.Series([1.0] * len(weighted))),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    tail = (
        weighted.get("long_tail_loss", pd.Series([0] * len(weighted))).astype(bool)
        | weighted.get("short_tail_loss", pd.Series([0] * len(weighted))).astype(bool)
    )
    decision_action = weighted.get("decision_action", pd.Series([""] * len(weighted))).astype(str)
    best_action = weighted.get("best_action", pd.Series([""] * len(weighted))).astype(str)
    confidence = pd.to_numeric(
        weighted.get("decision_confidence", pd.Series([0.0] * len(weighted))),
        errors="coerce",
    ).fillna(0.0)
    hard_error = (confidence >= 0.7) & (decision_action != best_action)
    cost_consumed_edge = (
        (
            pd.to_numeric(weighted["long_return_pct"], errors="coerce") > 0.0
        )
        & (
            pd.to_numeric(weighted["long_return_pct"], errors="coerce")
            - pd.to_numeric(weighted["long_execution_cost_pct"], errors="coerce")
            <= 0.0
        )
    ) | (
        (
            pd.to_numeric(weighted["short_return_pct"], errors="coerce") > 0.0
        )
        & (
            pd.to_numeric(weighted["short_return_pct"], errors="coerce")
            - pd.to_numeric(weighted["short_execution_cost_pct"], errors="coerce")
            <= 0.0
        )
    )
    multiplier = (
        1.0
        + tail.astype(float) * 0.25
        + hard_error.astype(float) * 0.25
        + cost_consumed_edge.astype(float) * 0.15
    )
    final_weights = (base * recency * multiplier).clip(lower=0.01, upper=2.0)
    weighted["sample_weight"] = final_weights
    weighted["replay_recency_weight"] = recency
    weighted["replay_tail_emphasis"] = tail.astype(int)
    weighted["replay_hard_error_emphasis"] = hard_error.astype(int)
    weighted["replay_cost_error_emphasis"] = cost_consumed_edge.astype(int)
    total = float(final_weights.sum())
    squared_total = float(np.square(final_weights).sum())
    effective = total * total / squared_total if squared_total > 0.0 else 0.0
    return weighted, {
        "version": REPLAY_WEIGHT_POLICY_VERSION,
        "validation_and_test_resampling": False,
        "recency_half_life_days": half_life_days,
        "minimum_recency_weight": minimum_recency,
        "tail_multiplier": 1.25,
        "hard_error_multiplier": 1.25,
        "cost_consumed_edge_multiplier": 1.15,
        "sample_count": int(len(weighted)),
        "raw_weight_total": round(float(base.sum()), 6),
        "final_weight_total": round(total, 6),
        "effective_sample_size": round(effective, 6),
        "tail_sample_count": int(tail.sum()),
        "hard_error_sample_count": int(hard_error.sum()),
        "cost_consumed_edge_sample_count": int(cost_consumed_edge.sum()),
    }


def _fingerprint_value(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return normalized.astimezone(UTC).isoformat()
    if isinstance(value, np.generic):
        return _fingerprint_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _training_data_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(str(column) for column in frame.columns)
    records = [
        {column: _fingerprint_value(row.get(column)) for column in columns}
        for row in frame.to_dict("records")
    ]
    records.sort(
        key=lambda row: (
            str(row.get("label_timestamp") or ""),
            str(row.get("decision_group") or ""),
            int(_safe_float(row.get("horizon_minutes"), 0.0) or 0),
            int(_safe_float(row.get("id"), 0.0) or 0),
        )
    )
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compact_integer_ranges(values: list[Any]) -> list[list[int]]:
    ordered = sorted({int(number) for value in values if (number := _safe_float(value, 0.0)) > 0})
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


def _chronological_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    if ordered["decision_group"].isna().any():
        raise ValueError("decision_group cannot be missing for chronological evaluation")
    if "label_timestamp" in ordered:
        ordered["_evaluation_timestamp"] = pd.to_datetime(
            ordered["label_timestamp"],
            utc=True,
            errors="coerce",
        )
    else:
        ordered["_evaluation_timestamp"] = pd.NaT
    if ordered["_evaluation_timestamp"].isna().any():
        raise ValueError("label_timestamp is required for chronological evaluation")
    ordered["_evaluation_id"] = pd.to_numeric(
        ordered.get("id", pd.Series(range(len(ordered)))),
        errors="coerce",
    ).fillna(0)
    return ordered.sort_values(
        ["_evaluation_timestamp", "_evaluation_id"],
        na_position="last",
        kind="stable",
    ).drop(columns=["_evaluation_timestamp", "_evaluation_id"])


def _decision_group_availability(
    frame: pd.DataFrame,
) -> tuple[list[str], dict[str, dict[str, pd.Timestamp]]]:
    ordered = _chronological_frame(frame)
    label_timestamps = pd.to_datetime(
        ordered["label_timestamp"],
        utc=True,
        errors="raise",
    )
    horizons = pd.to_numeric(ordered["horizon_minutes"], errors="coerce")
    if horizons.isna().any() or (horizons <= 0).any():
        raise ValueError("positive horizon is required for chronological evaluation")
    inferred_decisions = label_timestamps - pd.to_timedelta(horizons, unit="m")
    if "decision_timestamp" in ordered:
        explicit_decisions = pd.to_datetime(
            ordered["decision_timestamp"],
            utc=True,
            errors="coerce",
        )
        decision_timestamps = explicit_decisions.fillna(inferred_decisions)
    else:
        decision_timestamps = inferred_decisions
    working = ordered.assign(
        _label_timestamp=label_timestamps,
        _decision_timestamp=decision_timestamps,
    )
    bounds: dict[str, dict[str, pd.Timestamp]] = {}
    for group, rows in working.groupby(working["decision_group"].astype(str)):
        bounds[str(group)] = {
            "start": rows["_label_timestamp"].min(),
            "end": rows["_label_timestamp"].max(),
            "decision_start": rows["_decision_timestamp"].min(),
            "decision_end": rows["_decision_timestamp"].max(),
        }
    groups = sorted(
        bounds,
        key=lambda group: (
            bounds[group]["decision_start"],
            bounds[group]["decision_end"],
            group,
        ),
    )
    return groups, bounds


def _walk_forward_side_scores(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    side: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    x_train = train[FEATURE_KEYS]
    x_validation = validation[FEATURE_KEYS]
    weights = train.get("sample_weight", pd.Series([1.0] * len(train))).astype(float)
    return_column = f"{side}_return_pct"
    cost_column = f"{side}_execution_cost_pct"
    net_training_returns = train[return_column].astype(float) - train[cost_column].astype(float)
    tail_policy = empirical_policy_value(
        f"{side}_walk_forward_tail_loss_boundary_pct",
        net_training_returns[net_training_returns < 0].tolist(),
        selector="lower_hinge",
        observation_window="walk_forward_training_groups_only",
    )
    tail_boundary = float(tail_policy.value) if tail_policy.value is not None else 0.0
    tail_scale = max(abs(tail_boundary), float(np.finfo(float).eps))
    tail_labels = (net_training_returns < tail_boundary).astype(int)
    market_model = _make_regressor(train[return_column])
    cost_model = _make_regressor(train[cost_column])
    tail_model = _make_classifier(tail_labels)
    market_model.fit(x_train, train[return_column], model__sample_weight=weights)
    cost_model.fit(x_train, train[cost_column], model__sample_weight=weights)
    tail_model.fit(x_train, tail_labels, model__sample_weight=weights)
    scores = (
        np.asarray(market_model.predict(x_validation), dtype=float)
        - np.asarray(cost_model.predict(x_validation), dtype=float)
        - _positive_proba(tail_model, x_validation) * tail_scale
    )
    return scores, {
        **tail_policy.to_dict(),
        "scale_pct": tail_scale,
        "training_decision_group_count": int(train["decision_group"].nunique()),
    }


def _top_scored_return_rows(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    side: str,
) -> list[dict[str, Any]]:
    if not len(scores):
        return []
    rows = [
        {
            "symbol": str(row.get("symbol") or ""),
            "market_regime": str(row.get("market_regime") or "unknown"),
            "decision_group": str(row.get("decision_group") or ""),
            "label_timestamp": _fingerprint_value(row.get("label_timestamp")),
            "return_pct": float(row[f"{side}_return_pct"])
            - float(row[f"{side}_execution_cost_pct"]),
            "gross_market_return_pct": float(row[f"{side}_return_pct"]),
            "execution_cost_pct": float(row[f"{side}_execution_cost_pct"]),
            "score": float(scores[index]),
        }
        for index, (_, row) in enumerate(frame.iterrows())
    ]
    return _select_top_return_rows(rows)


def _all_scored_return_rows(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    side: str,
) -> list[dict[str, Any]]:
    return [
        {
            "symbol": str(row.get("symbol") or ""),
            "market_regime": str(row.get("market_regime") or "unknown"),
            "decision_group": str(row.get("decision_group") or ""),
            "label_timestamp": _fingerprint_value(row.get("label_timestamp")),
            "return_pct": float(row[f"{side}_return_pct"])
            - float(row[f"{side}_execution_cost_pct"]),
            "gross_market_return_pct": float(row[f"{side}_return_pct"]),
            "execution_cost_pct": float(row[f"{side}_execution_cost_pct"]),
            "score": float(scores[index]),
        }
        for index, (_, row) in enumerate(frame.iterrows())
    ]


def _select_top_return_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    count = max(int(math.sqrt(len(rows))), 1)
    return sorted(rows, key=lambda row: float(row["score"]))[-count:]


def _max_drawdown(returns: list[float]) -> float | None:
    if not returns:
        return None
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in returns:
        equity += float(value)
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return float(drawdown)


def _return_evidence(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("label_timestamp") or ""),
            str(row.get("decision_group") or ""),
        ),
    )
    returns = [float(row["return_pct"]) for row in ordered]
    tail_policy = empirical_policy_value(
        "oos_tail_loss_boundary_pct",
        [value for value in returns if value < 0],
        selector="lower_hinge",
        observation_window="current_oos_evidence_only",
    )
    tail_boundary = float(tail_policy.value) if tail_policy.value is not None else 0.0
    summary = return_distribution_summary(
        returns,
        tail_loss_threshold_pct=abs(tail_boundary),
    )
    profit_factor_value = _safe_float(summary.get("profit_factor"), None)
    return_lcb = _safe_float(summary.get("return_lcb_pct"), None)
    cvar_value = _safe_float(summary.get("cvar_10_pct"), None)
    max_drawdown = _max_drawdown(returns)
    return {
        **summary,
        "tail_loss_policy": tail_policy.to_dict(),
        "tail_loss_scale_pct": abs(tail_boundary),
        "max_drawdown_pct": max_drawdown,
        "promotion_math_ready": bool(
            return_lcb is not None
            and return_lcb > 0.0
            and profit_factor_value is not None
            and profit_factor_value > 1.0
            and cvar_value is not None
            and max_drawdown is not None
        ),
    }


def _market_regime_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        regime = str(row.get("market_regime") or "").strip().lower()
        if regime and regime != "unknown":
            grouped.setdefault(regime, []).append(row)
    reports = {
        regime: _return_evidence(_select_top_return_rows(regime_rows))
        for regime, regime_rows in sorted(grouped.items())
    }
    stable = bool(len(reports) >= 2) and all(
        report.get("promotion_math_ready") is True for report in reports.values()
    )
    return {
        "stable": stable,
        "observed_regime_count": len(reports),
        "required_regime_count": 2,
        "regimes": reports,
        "blocking_reasons": (
            []
            if stable
            else [
                (
                    "insufficient_market_regime_coverage"
                    if len(reports) < 2
                    else "market_regime_fee_after_return_unstable"
                )
            ]
        ),
    }


def _authoritative_trade_return_evidence(
    trade_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    side_rows: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for sample in trade_samples:
        tasks = _safe_dict(_safe_dict(sample.get("profit_supervision")).get("tasks"))
        realized = _safe_dict(tasks.get(AUTHORITATIVE_REALIZED_RETURN_TASK))
        side = str(realized.get("side") or sample.get("side") or "").lower()
        value = _safe_float(realized.get(PROFIT_TRAINING_TARGET), float("nan"))
        if (
            realized.get("eligible") is not True
            or side not in side_rows
            or not math.isfinite(value)
        ):
            continue
        side_rows[side].append(
            {
                "symbol": str(sample.get("symbol") or ""),
                "decision_group": str(
                    sample.get("lifecycle_key")
                    or sample.get("position_id")
                    or sample.get("id")
                    or ""
                ),
                "label_timestamp": _fingerprint_value(
                    sample.get("label_timestamp")
                    or sample.get("closed_at")
                    or sample.get("updated_at")
                ),
                "return_pct": float(value),
                "score": float(value),
            }
        )
    sides = {side: _return_evidence(rows) for side, rows in side_rows.items()}
    fingerprint_payload = {
        side: [
            {
                key: row.get(key)
                for key in ("symbol", "decision_group", "label_timestamp", "return_pct")
            }
            for row in rows
        ]
        for side, rows in side_rows.items()
    }
    return {
        "version": "2026-07-15.authoritative-trade-return-evidence.v1",
        "source_authority": "okx_position_history_profit_supervision",
        "sides": sides,
        "sample_count": sum(len(rows) for rows in side_rows.values()),
        "data_fingerprint": hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }


def _leave_one_symbol_out_stability(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    symbols = sorted({str(row.get("symbol") or "") for row in rows if row.get("symbol")})
    reports = []
    for symbol in symbols:
        remaining = [row for row in rows if str(row.get("symbol") or "") != symbol]
        selected = _select_top_return_rows(remaining)
        reports.append(
            {
                "excluded_symbol": symbol,
                "remaining_symbol_count": len(
                    {str(row.get("symbol") or "") for row in remaining if row.get("symbol")}
                ),
                "evidence": _return_evidence(selected),
            }
        )
    return {
        "version": "2026-07-15.leave-one-symbol-out.v1",
        "evaluated_symbol_count": len(symbols),
        "rows": reports,
        "stable": bool(reports) and all(row["evidence"]["promotion_math_ready"] for row in reports),
        "policy": "recompute_oos_fee_after_return_evidence_after_each_symbol_removal",
    }


def _walk_forward_return_report(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    ordered = _chronological_frame(frame)
    groups, group_bounds = _decision_group_availability(ordered)
    version = "2026-07-15.expanding-decision-group-walk-forward.v1"
    if len(groups) <= 1:
        return {
            "version": version,
            "status": "insufficient_chronological_decision_groups",
            "folds": [],
            "decision_group_disjoint": False,
            "chronological_label_disjoint": False,
            "model_refit_per_fold": True,
        }
    validation_candidates = [
        group
        for group in groups
        if any(
            group_bounds[prior]["end"] < group_bounds[group]["decision_start"]
            for prior in groups
            if group_bounds[prior]["decision_start"] < group_bounds[group]["decision_start"]
        )
    ]
    if not validation_candidates:
        return {
            "version": version,
            "status": "insufficient_purged_chronological_decision_groups",
            "folds": [],
            "decision_group_count": len(groups),
            "decision_group_disjoint": False,
            "chronological_label_disjoint": False,
            "model_refit_per_fold": True,
            "chronological": True,
        }
    validation_fold_count = max(
        int(math.ceil(math.log10(len(validation_candidates) + 1))),
        1,
    )
    group_blocks = [
        [str(value) for value in block.tolist()]
        for block in np.array_split(
            np.asarray(validation_candidates, dtype=object),
            validation_fold_count,
        )
        if len(block)
    ]
    folds: list[dict[str, Any]] = []
    oos_rows = {"long": [], "short": []}
    for index, validation_groups in enumerate(group_blocks, start=1):
        validation_decision_start = min(
            group_bounds[group]["decision_start"] for group in validation_groups
        )
        training_set = {
            group for group in groups if group_bounds[group]["end"] < validation_decision_start
        }
        validation_set = set(validation_groups)
        if training_set & validation_set:
            raise ValueError("walk-forward decision groups overlap")
        train = ordered[ordered["decision_group"].astype(str).isin(training_set)].copy()
        validation = ordered[ordered["decision_group"].astype(str).isin(validation_set)].copy()
        side_reports: dict[str, Any] = {}
        for side in ("long", "short"):
            scores, fold_tail_policy = _walk_forward_side_scores(
                train,
                validation,
                side=side,
            )
            selected_rows = _top_scored_return_rows(
                validation,
                scores,
                side=side,
            )
            oos_rows[side].extend(_all_scored_return_rows(validation, scores, side=side))
            side_reports[side] = {
                **_return_evidence(selected_rows),
                "training_tail_loss_policy": fold_tail_policy,
            }
        folds.append(
            {
                "fold": index,
                "training_decision_group_count": len(training_set),
                "validation_decision_group_count": len(validation_set),
                "validation_start": _fingerprint_value(validation.iloc[0].get("label_timestamp")),
                "validation_end": _fingerprint_value(validation.iloc[-1].get("label_timestamp")),
                "training_label_end": _fingerprint_value(
                    max(group_bounds[group]["end"] for group in training_set)
                ),
                "validation_decision_start": _fingerprint_value(validation_decision_start),
                "label_timestamp_overlap_count": 0,
                "purged_training_decision_group_count": sum(
                    1
                    for group in groups
                    if group_bounds[group]["decision_start"] < validation_decision_start
                    and group not in training_set
                ),
                "decision_group_overlap_count": 0,
                "sides": side_reports,
            }
        )
    side_reports = {}
    for side in ("long", "short"):
        evidence = _return_evidence(_select_top_return_rows(oos_rows[side]))
        side_reports[side] = {
            **evidence,
            "leave_one_symbol_out": _leave_one_symbol_out_stability(oos_rows[side]),
            "market_regime_stability": _market_regime_stability(oos_rows[side]),
        }
    return {
        "version": version,
        "status": "complete" if folds else "insufficient_chronological_decision_groups",
        "folds": folds,
        "fold_count": len(folds),
        "decision_group_count": len(groups),
        "decision_group_disjoint": all(row["decision_group_overlap_count"] == 0 for row in folds),
        "chronological_label_disjoint": all(
            row["label_timestamp_overlap_count"] == 0
            and row["training_label_end"] < row["validation_decision_start"]
            for row in folds
        ),
        "model_refit_per_fold": True,
        "chronological": True,
        "sides": side_reports,
        "stable": len(folds) >= MIN_ACTIVE_WALK_FORWARD_FOLDS
        and all(
            evidence["promotion_math_ready"]
            and evidence["leave_one_symbol_out"]["stable"]
            and evidence["market_regime_stability"]["stable"]
            and all(fold["sides"][side]["promotion_math_ready"] for fold in folds)
            for side, evidence in side_reports.items()
        ),
    }


@dataclass(frozen=True)
class DecisionGroupPartition:
    train: pd.DataFrame
    holdout: pd.DataFrame
    report: dict[str, Any]


def _champion_comparison_inputs(
    current_artifact: ResolvedModelArtifact | None,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    if not isinstance(current_artifact, ResolvedModelArtifact):
        return None, None, []
    manifest = current_artifact.manifest
    eligibility_errors = [
        *local_ml_artifact_compatibility_errors(manifest),
        *decision_group_partition_errors(manifest.get("decision_group_partition")),
    ]
    if not eligibility_errors:
        try:
            bundle = load_trusted_joblib(
                current_artifact.model_path,
                trusted_root=current_artifact.model_path.parent,
                expected_type=dict,
            )
        except Exception:
            eligibility_errors.append("artifact_bundle_unloadable")
        else:
            eligibility_errors.extend(
                local_ml_artifact_compatibility_errors(
                    _safe_dict(bundle.get("metadata")),
                    bundle=bundle,
                )
            )
    if eligibility_errors:
        return None, None, sorted(set(eligibility_errors))
    stage = str(_safe_dict(current_artifact.activation_manifest).get("activation_stage") or "")
    return manifest, stage or None, []


_REQUIRED_LOCAL_ML_BUNDLE_KEYS = (
    "long_regressor",
    "short_regressor",
    "long_cost_regressor",
    "short_cost_regressor",
)


def local_ml_artifact_compatibility_errors(
    metadata: dict[str, Any],
    *,
    bundle: dict[str, Any] | None = None,
) -> list[str]:
    """Return stable diagnostics for artifacts the current runtime cannot use."""

    errors: list[str] = []
    expected_values = {
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_version": RETURN_LABEL_VERSION,
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "multitask_prediction_contract_version": MULTITASK_PREDICTION_CONTRACT_VERSION,
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            errors.append(f"artifact_{key}_mismatch")
    if bundle is not None:
        errors.extend(
            f"artifact_model_component_missing:{key}"
            for key in _REQUIRED_LOCAL_ML_BUNDLE_KEYS
            if key not in bundle
        )
        bundle_feature_keys = bundle.get("feature_keys")
        if not isinstance(bundle_feature_keys, (list, tuple)):
            errors.append("artifact_feature_keys_missing")
        elif list(bundle_feature_keys) != FEATURE_KEYS:
            errors.append("artifact_feature_keys_mismatch")
    return errors


def decision_group_partition(frame: pd.DataFrame) -> DecisionGroupPartition:
    """Build the only valid chronological train/holdout partition."""

    if frame.empty:
        report = {
            "version": DECISION_GROUP_PARTITION_VERSION,
            "ready": False,
            "reason": "cost_complete_training_distribution_unavailable",
            "sample_count": 0,
            "decision_group_count": 0,
            "candidate_training_decision_group_count": 0,
            "purged_training_decision_group_count": 0,
            "purged_training_sample_count": 0,
            "train_sample_count": 0,
            "train_decision_group_count": 0,
            "holdout_sample_count": 0,
            "holdout_decision_group_count": 0,
            "minimum_train_sample_count": MIN_TRAINING_SAMPLE_COUNT,
            "minimum_train_decision_group_count": MIN_TRAINING_DECISION_GROUP_COUNT,
            "holdout_decision_start": None,
            "training_label_end": None,
            "decision_group_overlap_count": 0,
            "chronological_label_disjoint": False,
        }
        return DecisionGroupPartition(frame.copy(), frame.copy(), report)

    group_column = "decision_group"
    if group_column not in frame:
        raise ValueError("decision_group is required for leakage-free evaluation")
    ordered = _chronological_frame(frame).reset_index(drop=True)
    ordered_groups, group_bounds = _decision_group_availability(ordered)
    boundary = len(ordered_groups) // 2
    candidate_train_groups = set(ordered_groups[:boundary])
    holdout_groups = set(ordered_groups[boundary:])
    holdout_decision_start = (
        min(group_bounds[group]["decision_start"] for group in holdout_groups)
        if holdout_groups
        else None
    )
    train_groups = {
        group
        for group in candidate_train_groups
        if holdout_decision_start is not None
        and group_bounds[group]["end"] < holdout_decision_start
    }
    train = ordered[ordered[group_column].astype(str).isin(train_groups)].copy()
    holdout = ordered[ordered[group_column].astype(str).isin(holdout_groups)].copy()
    purged_training_groups = candidate_train_groups - train_groups
    purged_training_sample_count = int(
        ordered[group_column].astype(str).isin(purged_training_groups).sum()
    )
    overlap_count = len(train_groups & holdout_groups)
    training_label_end = (
        max(group_bounds[group]["end"] for group in train_groups) if train_groups else None
    )
    chronological_label_disjoint = bool(
        training_label_end is not None
        and holdout_decision_start is not None
        and training_label_end < holdout_decision_start
    )
    if not holdout_groups or holdout.empty:
        reason = "decision_group_holdout_unavailable"
    elif (
        len(train_groups) < MIN_TRAINING_DECISION_GROUP_COUNT
        or len(train) < MIN_TRAINING_SAMPLE_COUNT
    ):
        reason = "decision_group_training_partition_immature"
    elif overlap_count or not chronological_label_disjoint:
        reason = "decision_group_partition_overlap"
    else:
        reason = "ready"
    report = {
        "version": DECISION_GROUP_PARTITION_VERSION,
        "ready": reason == "ready",
        "reason": reason,
        "sample_count": int(len(ordered)),
        "decision_group_count": len(ordered_groups),
        "candidate_training_decision_group_count": len(candidate_train_groups),
        "purged_training_decision_group_count": len(purged_training_groups),
        "purged_training_sample_count": purged_training_sample_count,
        "train_sample_count": int(len(train)),
        "train_decision_group_count": len(train_groups),
        "holdout_sample_count": int(len(holdout)),
        "holdout_decision_group_count": len(holdout_groups),
        "minimum_train_sample_count": MIN_TRAINING_SAMPLE_COUNT,
        "minimum_train_decision_group_count": MIN_TRAINING_DECISION_GROUP_COUNT,
        "holdout_decision_start": _fingerprint_value(holdout_decision_start),
        "training_label_end": _fingerprint_value(training_label_end),
        "decision_group_overlap_count": overlap_count,
        "chronological_label_disjoint": chronological_label_disjoint,
    }
    if report["ready"] and decision_group_partition_errors(report):
        raise ValueError("ready decision-group partition violates its training contract")
    return DecisionGroupPartition(train, holdout, report)


def _mature_shadow_multitask_labels(
    snapshot: dict[str, Any],
    *,
    expected_label_version: str,
) -> dict[str, Any]:
    contract = _safe_dict(snapshot.get("training_label_contract"))
    if (
        not contract
        or str(contract.get("version") or "") != expected_label_version
        or contract.get("label_maturity_status") != "matured"
    ):
        return {}
    result: dict[str, Any] = {
        "label_contract_version": str(contract.get("version") or ""),
        "label_maturity_status": "matured",
    }
    for key in ("long_mfe_pct", "long_mae_pct", "short_mfe_pct", "short_mae_pct"):
        value = _safe_float(contract.get(key), float("nan"))
        if not math.isfinite(value) or value < 0.0:
            return {}
        result[key] = value
    for key in (
        "long_stop_loss_triggered",
        "short_stop_loss_triggered",
        "long_take_profit_triggered",
        "short_take_profit_triggered",
    ):
        value = contract.get(key)
        if not isinstance(value, bool):
            return {}
        result[key] = int(value)
    for key in ("long_first_touch", "short_first_touch"):
        value = str(contract.get(key) or "").lower()
        if value not in {"none", "stop_loss", "take_profit", "path_uncertain"}:
            return {}
        result[key] = value
    return result


def build_training_frame(rows: list[Any]) -> pd.DataFrame:
    data: list[dict[str, Any]] = []
    annotated_by_id = {
        int(sample.get("id") or 0): sample
        for sample in annotate_samples(
            [_shadow_quality_sample(row) for row in rows],
            "shadow",
        )
    }
    for row in rows:
        snapshot = _parse_json(getattr(row, "feature_snapshot", None))
        if not snapshot:
            continue
        raw_long_return = getattr(row, "long_return_pct", None)
        raw_short_return = getattr(row, "short_return_pct", None)
        if raw_long_return is None or raw_short_return is None:
            continue
        sample_id = int(getattr(row, "id", 0) or 0)
        quality_sample = annotated_by_id.get(sample_id, {})
        if not quality_sample or quality_sample.get("exclude_from_training"):
            continue
        label_version = str(getattr(row, "label_version", "") or "")
        multitask_labels = _mature_shadow_multitask_labels(
            snapshot,
            expected_label_version=label_version,
        )
        if not multitask_labels:
            continue
        horizon_minutes = int(getattr(row, "horizon_minutes", 10) or 10)
        supervision = _safe_dict(quality_sample.get("profit_supervision"))
        tasks = _safe_dict(supervision.get("tasks"))
        market_task = _safe_dict(tasks.get(MARKET_OPPORTUNITY_TASK))
        cost_task = _safe_dict(tasks.get(COUNTERFACTUAL_EXECUTION_COST_TASK))
        if market_task.get("eligible") is not True or cost_task.get("eligible") is not True:
            continue
        long_return = _safe_float(
            market_task.get("long_gross_market_return_pct"),
            float("nan"),
        )
        short_return = _safe_float(
            market_task.get("short_gross_market_return_pct"),
            float("nan"),
        )
        long_cost = _safe_float(cost_task.get("long_total_cost_pct"), float("nan"))
        short_cost = _safe_float(cost_task.get("short_total_cost_pct"), float("nan"))
        if not all(
            math.isfinite(value) for value in (long_return, short_return, long_cost, short_cost)
        ):
            continue
        feature_row: dict[str, Any] = dict(
            _feature_row_from_snapshot(
                {
                    **snapshot,
                    "symbol": str(getattr(row, "symbol", "") or snapshot.get("symbol") or ""),
                },
                decision_confidence=_safe_float(getattr(row, "decision_confidence", 0.0)),
                horizon_minutes=horizon_minutes,
            )
        )
        feature_row.update(
            {
                "id": sample_id,
                "decision_id": int(getattr(row, "decision_id", 0) or 0) or None,
                "decision_group": _safe_dict(quality_sample.get("correlation_weight")).get(
                    "correlation_group"
                ),
                "decision_timestamp": getattr(row, "created_at", None),
                "label_timestamp": getattr(row, "due_at", None) or getattr(row, "created_at", None),
                "symbol": str(getattr(row, "symbol", "") or ""),
                "market_regime": _market_regime_label(snapshot),
                "liquidity_regime": (
                    "high"
                    if feature_row["liquidity_regime_high"] == 1.0
                    else "low"
                ),
                "asset_group": "major" if feature_row["major_asset"] == 1.0 else "alt",
                "news_regime": (
                    "news_shock"
                    if feature_row["news_shock_present"] == 1.0
                    else "no_news_shock"
                ),
                "decision_action": str(getattr(row, "decision_action", "") or ""),
                "best_action": str(getattr(row, "best_action", "") or ""),
                "missed_opportunity": bool(getattr(row, "missed_opportunity", False)),
                "raw_long_return_pct": _safe_float(raw_long_return),
                "raw_short_return_pct": _safe_float(raw_short_return),
                "long_return_pct": long_return,
                "short_return_pct": short_return,
                "long_execution_cost_pct": long_cost,
                "short_execution_cost_pct": short_cost,
                "execution_cost": cost_task,
                "profit_supervision": supervision,
                "sample_weight": _safe_float(quality_sample.get("sample_weight"), 0.0),
                "data_quality_status": quality_sample.get("data_quality_status"),
                "data_quality_score": quality_sample.get("data_quality_score"),
                "quality_reasons": list(quality_sample.get("quality_reasons") or []),
                "sample_source": "shadow_market_label",
                "training_tasks": ["market_opportunity", "entry_timing"],
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                **multitask_labels,
            }
        )
        data.append(feature_row)
    frame = pd.DataFrame(data)
    if frame.empty:
        return frame
    tail_policy: dict[str, Any] = {}
    for side in ("long", "short"):
        returns = frame[f"{side}_return_pct"].astype(float) - frame[
            f"{side}_execution_cost_pct"
        ].astype(float)
        boundary = empirical_policy_value(
            f"{side}_tail_loss_boundary_pct",
            returns[returns < 0].tolist(),
            selector="lower_hinge",
            observation_window="current_shadow_market_opportunity_training_window",
        )
        threshold = float(boundary.value) if boundary.value is not None else 0.0
        frame[f"{side}_tail_loss"] = (returns < threshold).astype(int)
        frame[f"{side}_win"] = (returns > 0.0).astype(int)
        tail_policy[side] = boundary.to_dict()
    frame.attrs["tail_loss_policy"] = tail_policy
    frame.attrs["profit_supervision_version"] = PROFIT_SUPERVISION_VERSION
    frame.attrs["feature_contract_version"] = FEATURE_CONTRACT_VERSION
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


def _persist_training_bundle(
    bundle: dict[str, Any],
    metadata: dict[str, Any],
    *,
    source_code_version: str,
) -> dict[str, Any]:
    if MODEL_PATH != MODEL_DIR / "net_return_model.joblib" or METADATA_PATH != (
        MODEL_DIR / "net_return_model_metadata.json"
    ):
        dump_trusted_joblib(bundle, MODEL_PATH, trusted_root=MODEL_DIR)
        METADATA_PATH.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        resolved = ML_SIGNAL_ARTIFACT_REGISTRY.persist_candidate_joblib(
            bundle,
            metadata,
            parent_model_identity=("sklearn RandomForest/Dummy classifier-regressor pipelines"),
            code_version=source_code_version,
        )
        metadata.clear()
        metadata.update(resolved.manifest)
    return metadata


def persist_cached_training_candidate(
    candidate_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Persist the exact dry-run fit so auto-training never fits one window twice."""

    cached = _TRAINING_CANDIDATE_CACHE.pop(id(candidate_metadata), None)
    if cached is None:
        return None
    bundle, cached_metadata, source_code_version = cached
    if cached_metadata is not candidate_metadata:
        return None
    candidate_metadata.update(
        {
            "training_run_mode": "persist",
            "artifact_persisted": True,
            "governance_report": artifact_bound_governance_report(
                _safe_dict(candidate_metadata.get("quality_report")),
                persist_artifact=True,
            ),
        }
    )
    bundle["metadata"] = candidate_metadata
    return _persist_training_bundle(
        bundle,
        candidate_metadata,
        source_code_version=source_code_version,
    )


def train_from_frame(
    frame: pd.DataFrame,
    *,
    completed_sample_count: int | None = None,
    training_quality_report: dict[str, Any] | None = None,
    trade_samples: list[dict[str, Any]] | None = None,
    persist_artifact: bool = True,
) -> dict[str, Any]:
    required_multitask_columns = {
        "label_contract_version",
        "label_maturity_status",
        "long_mfe_pct",
        "long_mae_pct",
        "short_mfe_pct",
        "short_mae_pct",
        "long_stop_loss_triggered",
        "short_stop_loss_triggered",
        "long_take_profit_triggered",
        "short_take_profit_triggered",
        "long_first_touch",
        "short_first_touch",
    }
    missing_multitask_columns = sorted(required_multitask_columns - set(frame.columns))
    if missing_multitask_columns:
        raise ValueError(
            "mature multitask market labels are required: "
            + ", ".join(missing_multitask_columns)
        )
    if not frame["label_maturity_status"].eq("matured").all():
        raise ValueError("unmatured market labels cannot enter training")
    partition = decision_group_partition(frame)
    if not partition.report["ready"]:
        raise ValueError(
            f"{partition.report['reason']}: "
            f"train_groups={partition.report['train_decision_group_count']}, "
            f"train_samples={partition.report['train_sample_count']}, "
            f"holdout_groups={partition.report['holdout_decision_group_count']}, "
            f"holdout_samples={partition.report['holdout_sample_count']}"
        )
    frame = _chronological_frame(frame).reset_index(drop=True)
    tail_policy: dict[str, Any] = {}
    tail_scales: dict[str, float] = {}
    for side in ("long", "short"):
        training_net_returns = partition.train[f"{side}_return_pct"].astype(
            float
        ) - partition.train[f"{side}_execution_cost_pct"].astype(float)
        negatives = training_net_returns[training_net_returns < 0].tolist()
        generated = empirical_policy_value(
            f"{side}_tail_loss_boundary_pct",
            negatives,
            selector="lower_hinge",
            observation_window="chronological_training_partition_only",
        )
        tail_policy[side] = generated.to_dict()
        boundary = float(generated.value) if generated.value is not None else 0.0
        net_returns = frame[f"{side}_return_pct"].astype(float) - frame[
            f"{side}_execution_cost_pct"
        ].astype(float)
        frame[f"{side}_tail_loss"] = (net_returns < boundary).astype(int)
        frame[f"{side}_win"] = (net_returns > 0.0).astype(int)
        tail_scales[side] = max(abs(boundary), float(np.finfo(float).eps))
    train = frame.loc[partition.train.index].copy()
    test = frame.loc[partition.holdout.index].copy()
    train, replay_weight_manifest = _apply_training_replay_weights(train)
    training_data_sha256 = _training_data_sha256(frame)
    source_code_version = _training_source_code_version()
    source_code_sha256 = source_code_version.removeprefix("source-sha256:")
    walk_forward_report = _walk_forward_return_report(frame)
    strategy_replay_holdout = {
        "version": "2026-07-21.strategy-replay-holdout.v1",
        "source": "artifact_disjoint_test_partition",
        "sample_count": int(len(test)),
        "decision_group_count": int(test["decision_group"].nunique()),
        "shadow_source_id_ranges": _compact_integer_ranges(test["id"].tolist()),
        "label_start": _fingerprint_value(test["label_timestamp"].min()),
        "label_end": _fingerprint_value(test["label_timestamp"].max()),
        "training_data_sha256": training_data_sha256,
    }
    x_train = train[FEATURE_KEYS]
    x_test = test[FEATURE_KEYS]
    train_weights = train.get("sample_weight", pd.Series([1.0] * len(train))).astype(float)

    long_classifier = _make_classifier(train["long_win"])
    short_classifier = _make_classifier(train["short_win"])
    long_tail_classifier = _make_classifier(train["long_tail_loss"])
    short_tail_classifier = _make_classifier(train["short_tail_loss"])
    long_regressor = _make_regressor(train["long_return_pct"])
    short_regressor = _make_regressor(train["short_return_pct"])
    long_cost_regressor = _make_regressor(train["long_execution_cost_pct"])
    short_cost_regressor = _make_regressor(train["short_execution_cost_pct"])
    long_mfe_regressor = _make_regressor(train["long_mfe_pct"])
    long_mae_regressor = _make_regressor(train["long_mae_pct"])
    short_mfe_regressor = _make_regressor(train["short_mfe_pct"])
    short_mae_regressor = _make_regressor(train["short_mae_pct"])

    long_classifier.fit(x_train, train["long_win"], model__sample_weight=train_weights)
    short_classifier.fit(x_train, train["short_win"], model__sample_weight=train_weights)
    long_tail_classifier.fit(x_train, train["long_tail_loss"], model__sample_weight=train_weights)
    short_tail_classifier.fit(x_train, train["short_tail_loss"], model__sample_weight=train_weights)
    long_regressor.fit(x_train, train["long_return_pct"], model__sample_weight=train_weights)
    short_regressor.fit(x_train, train["short_return_pct"], model__sample_weight=train_weights)
    long_cost_regressor.fit(
        x_train,
        train["long_execution_cost_pct"],
        model__sample_weight=train_weights,
    )
    short_cost_regressor.fit(
        x_train,
        train["short_execution_cost_pct"],
        model__sample_weight=train_weights,
    )
    long_mfe_regressor.fit(x_train, train["long_mfe_pct"], model__sample_weight=train_weights)
    long_mae_regressor.fit(x_train, train["long_mae_pct"], model__sample_weight=train_weights)
    short_mfe_regressor.fit(x_train, train["short_mfe_pct"], model__sample_weight=train_weights)
    short_mae_regressor.fit(x_train, train["short_mae_pct"], model__sample_weight=train_weights)

    long_scores = _positive_proba(long_classifier, x_test)
    short_scores = _positive_proba(short_classifier, x_test)
    long_tail_scores = _positive_proba(long_tail_classifier, x_test)
    short_tail_scores = _positive_proba(short_tail_classifier, x_test)
    long_distribution = _regression_prediction_distribution(long_regressor, x_test)
    short_distribution = _regression_prediction_distribution(short_regressor, x_test)
    long_cost_distribution = _regression_prediction_distribution(
        long_cost_regressor,
        x_test,
    )
    short_cost_distribution = _regression_prediction_distribution(
        short_cost_regressor,
        x_test,
    )
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
    actual_trade_calibration = authoritative_trade_calibration(trade_samples or [])
    authoritative_return_evidence = _authoritative_trade_return_evidence(trade_samples or [])
    supervision_report = profit_supervision_report(
        frame.to_dict("records"),
        trade_samples or [],
    )
    supervision_report = {
        **supervision_report,
        "actual_trade_calibration_fingerprint": actual_trade_calibration.get("data_fingerprint"),
    }
    trade_samples = trade_samples or []
    execution_task_count = sum(
        1
        for sample in trade_samples
        if _safe_dict(
            _safe_dict(_safe_dict(sample.get("profit_supervision")).get("tasks")).get(
                COUNTERFACTUAL_EXECUTION_COST_TASK
            )
        ).get("eligible")
        is True
    )
    exit_task_count = sum(
        1
        for sample in trade_samples
        if _safe_dict(
            _safe_dict(_safe_dict(sample.get("profit_supervision")).get("tasks")).get(
                AUTHORITATIVE_REALIZED_RETURN_TASK
            )
        ).get("eligible")
        is True
    )
    task_manifest = {
        "market_opportunity": {
            "source": "mature_shadow_market_labels",
            "sample_count": int(len(train)),
            "decision_group_count": int(train["decision_group"].nunique()),
            "targets": [
                "gross_return_pct",
                "mfe_pct",
                "mae_pct",
                "tail_loss",
                "stop_loss_triggered",
                "take_profit_triggered",
                "first_touch",
            ],
        },
        "entry_timing": {
            "source": "mature_shadow_market_labels_with_counterfactual_cost",
            "sample_count": int(len(train)),
            "decision_group_count": int(train["decision_group"].nunique()),
            "targets": ["fee_after_fixed_horizon_return_pct", "mfe_pct", "mae_pct"],
        },
        "exit": {
            "source": "authoritative_okx_settled_positions",
            "sample_count": exit_task_count,
            "trained_in_shared_artifact": False,
        },
        "execution": {
            "source": "authoritative_okx_order_and_settlement_facts",
            "sample_count": execution_task_count,
            "trained_in_shared_artifact": True,
        },
    }
    frame_quality_report = {
        **frame_quality_report,
        "profit_supervision": supervision_report,
    }
    metadata = {
        "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "phase": "phase3_model_factory",
        "version": now,
        "trained_at": now,
        "training_epoch_started_at": load_training_epoch_start().isoformat(),
        "pre_epoch_data_training_allowed": False,
        "sample_count": int(len(train)),
        "completed_shadow_sample_count": completed_count,
        "last_trained_completed_shadow_sample_count": completed_count,
        "training_shadow_sample_count": int(len(train)),
        "training_trade_sample_count": len(trade_samples or []),
        "completed_trade_sample_count": len(trade_samples or []),
        "last_trained_completed_trade_sample_count": len(trade_samples or []),
        "training_window_composition": _training_window_composition(train),
        "training_task_manifest": task_manifest,
        "training_sample_sources": {
            "shadow_market_label": int(len(frame)),
            "authoritative_okx_trade": len(trade_samples),
        },
        "replay_weight_manifest": replay_weight_manifest,
        "training_distribution_profile": _training_distribution_profile(frame),
        "quality_report": frame_quality_report,
        "market_fact_contract": _safe_dict(frame_quality_report.get("market_fact_contract")),
        "governance_report": artifact_bound_governance_report(
            frame_quality_report,
            persist_artifact=persist_artifact,
        ),
        "training_window_policy": "all_current_clean_separated_supervision_samples",
        "training_cursor_note": "last_trained_completed_shadow_sample_count is the cumulative cursor used for auto-training.",
        "test_count": int(len(test)),
        "feature_count": len(FEATURE_KEYS),
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "label_contract_versions": sorted(
            str(value)
            for value in frame["label_contract_version"].dropna().unique().tolist()
        ),
        "multitask_prediction_contract_version": MULTITASK_PREDICTION_CONTRACT_VERSION,
        "horizons": sorted(int(v) for v in frame["horizon_minutes"].dropna().unique().tolist()),
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "profit_supervision_report": supervision_report,
        "actual_trade_calibration": actual_trade_calibration,
        "authoritative_trade_return_evidence": authoritative_return_evidence,
        "positive_net_return_boundary_pct": 0.0,
        "positive_return_boundary_policy": "fee_after_profitability_math_boundary",
        "tail_loss_policy": tail_policy,
        "tail_loss_scale_pct": tail_scales,
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "evaluation_group_policy": "chronological_disjoint_decision_groups",
        "decision_group_partition": partition.report,
        "strategy_replay_holdout": strategy_replay_holdout,
        "training_data_sha256": training_data_sha256,
        "source_code_sha256": source_code_sha256,
        "walk_forward_report": walk_forward_report,
        "leave_one_symbol_out_report": {
            side: _safe_dict(_safe_dict(walk_forward_report.get("sides")).get(side)).get(
                "leave_one_symbol_out"
            )
            for side in ("long", "short")
        },
        "oos_return_evaluation": {
            side: {
                key: value
                for key, value in _safe_dict(
                    _safe_dict(walk_forward_report.get("sides")).get(side)
                ).items()
                if key != "leave_one_symbol_out"
            }
            for side in ("long", "short")
        },
        "train_decision_group_count": int(train["decision_group"].nunique()),
        "test_decision_group_count": int(test["decision_group"].nunique()),
        "completed_shadow_decision_group_count": int(frame["decision_group"].nunique()),
        "last_trained_completed_shadow_decision_group_count": int(
            frame["decision_group"].nunique()
        ),
        "prediction_distribution": {
            "lower_bound": "tree_prediction_lower_hinge",
            "uncertainty_source": "random_forest_tree_empirical_order_statistics",
            "tail_risk_source": "tail_loss_classifier_diagnostic_risk_penalty",
        },
        "training_objective": (
            "Regress shadow gross market opportunity and counterfactual execution cost "
            "as separate tasks. Authoritative OKX trade outcomes calibrate realized net "
            "return and slippage; classification metrics are diagnostics only."
        ),
        "counterfactual_cost_holdout": {
            "long_expected_pct": float(long_cost_distribution["expected"].mean()),
            "long_lower_quantile_pct": float(long_cost_distribution["lower_quantile"].mean()),
            "short_expected_pct": float(short_cost_distribution["expected"].mean()),
            "short_lower_quantile_pct": float(short_cost_distribution["lower_quantile"].mean()),
        },
        "metrics": {
            "long_auc": _safe_auc(test["long_win"], long_scores),
            "short_auc": _safe_auc(test["short_win"], short_scores),
            "long_pr_auc": _safe_pr_auc(test["long_win"], long_scores),
            "short_pr_auc": _safe_pr_auc(test["short_win"], short_scores),
            "long_accuracy": _safe_accuracy(test["long_win"], long_scores),
            "short_accuracy": _safe_accuracy(test["short_win"], short_scores),
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
        "training_policy": CURRENT_TRAINING_EPOCH_POLICY,
        "trade_sample_cursor_policy": CURRENT_TRAINING_EPOCH_POLICY,
        "training_mode": "walk_forward",
        "model_stage": "candidate",
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "training_run_mode": "persist" if persist_artifact else "dry_run",
        "artifact_persisted": bool(persist_artifact),
        "artifact_activation_manifest": {
            "status": "not_activated",
            "activation_stage": "candidate",
            "live_ml_ready": False,
        },
        "live_promotion_manifest": {
            "status": "not_issued",
            "reason": "candidate_requires_independent_shadow_and_return_readiness",
            "live_ml_ready": False,
        },
        "note": "本地 ML 直接优化费后预期收益及左尾风险；胜率仅作为诊断，不参与开仓、评分、权重或晋升。",
    }

    bundle = {
        "long_classifier": long_classifier,
        "short_classifier": short_classifier,
        "long_tail_classifier": long_tail_classifier,
        "short_tail_classifier": short_tail_classifier,
        "long_regressor": long_regressor,
        "short_regressor": short_regressor,
        "long_cost_regressor": long_cost_regressor,
        "short_cost_regressor": short_cost_regressor,
        "long_mfe_regressor": long_mfe_regressor,
        "long_mae_regressor": long_mae_regressor,
        "short_mfe_regressor": short_mfe_regressor,
        "short_mae_regressor": short_mae_regressor,
        "metadata": metadata,
        "feature_keys": FEATURE_KEYS,
    }
    if persist_artifact:
        _persist_training_bundle(
            bundle,
            metadata,
            source_code_version=source_code_version,
        )
    else:
        _TRAINING_CANDIDATE_CACHE.clear()
        _TRAINING_CANDIDATE_CACHE[id(metadata)] = (bundle, metadata, source_code_version)
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
        self.metadata_path = (
            METADATA_PATH
            if model_path is not None
            else (self.artifact_registry.model_root / "unregistered-metadata.json")
        )
        self._bundle: dict[str, Any] | None = None
        self._loaded_mtime: float | None = None
        self._loaded_pointer_mtime_ns: int | None = None
        self._resolved_artifact: ResolvedModelArtifact | None = None
        self._load_diagnostic: dict[str, Any] | None = None
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
            diagnostic = self._model_unavailable_diagnostic()
            readiness = disabled_ml_readiness(
                diagnostic["code"],
                diagnostic["message"],
            )
            artifact_metadata = _safe_dict(
                self._resolved_artifact.manifest if self._resolved_artifact is not None else None
            )
            activation = _safe_dict(
                self._resolved_artifact.activation_manifest
                if self._resolved_artifact is not None
                else None
            )
            return {
                **artifact_metadata,
                "available": False,
                "status": diagnostic["code"],
                "readiness_state": readiness["state"],
                "readiness": readiness,
                "live_ml_ready": False,
                "paper_analysis_permission": False,
                "paper_trading_permission": False,
                "live_trading_permission": False,
                "model_path": str(self.model_path),
                "artifact_registry": self._artifact_registry_status(),
                "artifact_lifecycle": activation.get("activation_stage")
                or artifact_metadata.get("artifact_lifecycle")
                or "unregistered",
                "artifact_activation_manifest": activation,
                "model_load_diagnostic": diagnostic,
                "strategy_blueprint": build_model_strategy_blueprint(
                    metadata=None,
                    readiness=readiness,
                    activation=None,
                ),
                "message": diagnostic["message"],
                **auto_status,
            }
        metadata = _safe_dict(self._bundle.get("metadata"))
        influence = _influence_policy(metadata)
        readiness = build_ml_readiness_report(metadata, influence)
        influence, readiness = _activation_gated_policy(
            influence,
            readiness,
            self._resolved_artifact,
        )
        live_ml_ready = bool(readiness.get("live_ml_ready"))
        advisory_enabled = bool(
            influence.get("advisory_enabled") and readiness.get("state") == "shadow_ready"
        )
        model_note = metadata.get("note")
        training_count = int(metadata.get("sample_count") or 0)
        training_epoch_counts = self._training_epoch_sample_count_status(metadata)
        activation = _safe_dict(
            self._resolved_artifact.activation_manifest
            if self._resolved_artifact is not None
            else None
        )
        strategy_blueprint = build_model_strategy_blueprint(
            metadata=metadata,
            readiness=readiness,
            activation=activation,
            artifact_version=self._artifact_version(metadata),
        )
        return {
            "available": True,
            "model_path": str(self.model_path),
            "artifact_registry": self._artifact_registry_status(),
            **metadata,
            "artifact_lifecycle": activation.get("activation_stage") or "unregistered",
            "artifact_activation_manifest": activation,
            "strategy_blueprint": strategy_blueprint,
            "training_shadow_sample_count": int(
                metadata.get("training_shadow_sample_count") or training_count
            ),
            "training_window_policy": metadata.get("training_window_policy")
            or "all_current_clean_cost_complete_samples",
            **training_epoch_counts,
            "status": (
                "ready"
                if live_ml_ready
                else str(readiness.get("state") or influence.get("status") or "learning_only")
            ),
            "mode": (
                "entry_profit_filter"
                if live_ml_ready
                else "paper_model"
            ),
            "readiness_state": readiness.get("state"),
            "readiness": readiness,
            "live_ml_ready": live_ml_ready,
            "paper_analysis_permission": True,
            "paper_trading_permission": True,
            "live_trading_permission": live_ml_ready,
            "advisory_enabled": advisory_enabled,
            "influence_policy": influence,
            "model_note": model_note,
            "note": (
                "模型已获得实盘候选资格；实盘订单仍需逐笔通过生产门禁。"
                if live_ml_ready
                else "模型可正常参与模拟盘分析和交易；晋升状态只阻断实盘权限。"
            ),
            **auto_status,
        }

    def strategy_blueprint(self) -> dict[str, Any]:
        """Return the current declarative paper strategy generated by the artifact."""

        self._ensure_loaded()
        return self._loaded_strategy_blueprint()

    def _loaded_strategy_blueprint(self) -> dict[str, Any]:
        if not self._bundle:
            diagnostic = self._model_unavailable_diagnostic()
            readiness = disabled_ml_readiness(
                diagnostic["code"],
                diagnostic["message"],
            )
            return build_model_strategy_blueprint(
                metadata=None,
                readiness=readiness,
                activation=None,
            )
        metadata = _safe_dict(self._bundle.get("metadata"))
        influence = _influence_policy(metadata)
        readiness = build_ml_readiness_report(metadata, influence)
        _influence, readiness = _activation_gated_policy(
            influence,
            readiness,
            self._resolved_artifact,
        )
        activation = _safe_dict(
            self._resolved_artifact.activation_manifest
            if self._resolved_artifact is not None
            else None
        )
        return build_model_strategy_blueprint(
            metadata=metadata,
            readiness=readiness,
            activation=activation,
            artifact_version=self._artifact_version(metadata),
        )

    def _artifact_version(self, metadata: dict[str, Any]) -> str | None:
        return (
            str(
                getattr(self._resolved_artifact, "version", None)
                or metadata.get("artifact_version")
                or metadata.get("version")
                or ""
            )
            or None
        )

    def rollback_to_strategy_model(self, target_version: str | None) -> dict[str, Any]:
        """Restore the registered predecessor only when it matches the strategy target."""

        target = str(target_version or "").strip()
        if not target:
            return {"rolled_back": False, "reason": "rollback_target_missing"}
        current = self.artifact_registry.resolve_current()
        if current is not None and current.version == target:
            return {
                "rolled_back": False,
                "reason": "target_model_already_current",
                "model_version": current.version,
            }
        rollback = self.artifact_registry.resolve_rollback()
        if rollback is None or rollback.version != target:
            return {
                "rolled_back": False,
                "reason": "registered_rollback_model_does_not_match_strategy_target",
                "target_model_version": target,
                "registered_rollback_model_version": (
                    rollback.version if rollback is not None else None
                ),
            }
        restored = self.artifact_registry.rollback_current()
        self._bundle = None
        self._loaded_mtime = None
        self._loaded_pointer_mtime_ns = None
        self._resolved_artifact = None
        self._ensure_loaded()
        return {
            "rolled_back": True,
            "reason": "previous_strategy_model_restored",
            "model_version": restored.version,
        }

    @staticmethod
    def _trained_cursor_from_metadata(metadata: dict[str, Any], completed_count: int) -> int:
        """Return an explicit trained cursor on the current-epoch sample scale."""

        value = metadata.get("last_trained_completed_shadow_sample_count")
        try:
            cursor = int(value)
        except (TypeError, ValueError):
            return 0
        return cursor if 0 <= cursor <= completed_count else 0

    def _training_epoch_sample_count_status(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Expose only current-epoch training counters."""

        try:
            completed_count = int(metadata.get("completed_shadow_sample_count") or 0)
        except (TypeError, ValueError):
            completed_count = 0
        completed_count = max(completed_count, 0)
        trained_cursor = self._trained_cursor_from_metadata(metadata, completed_count)
        new_count = max(completed_count - trained_cursor, 0)
        return {
            "training_policy": "current_training_epoch_only",
            "training_epoch_started_at": metadata.get("training_epoch_started_at"),
            "pre_epoch_data_training_allowed": False,
            "completed_shadow_sample_count": completed_count,
            "training_shadow_sample_count": completed_count,
            "last_trained_completed_shadow_sample_count": trained_cursor,
            "new_shadow_sample_count": new_count,
            "sample_cursor_policy": "current_training_epoch_only",
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
                AUTO_TRAIN_RETRY_INTERVAL_SECONDS if failed else AUTO_TRAIN_CHECK_INTERVAL_SECONDS
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
                trade_samples = await load_authoritative_trade_training_samples()
                completed_trade_count = len(trade_samples)
                metadata = self._current_metadata()
                cursor_metadata = self._training_cursor_metadata(metadata)
                last_sample_count = int(cursor_metadata.get("sample_count") or 0)
                last_completed_count = self._trained_cursor_from_metadata(
                    cursor_metadata,
                    completed_count,
                )
                last_completed_trade_count = int(
                    cursor_metadata.get("last_trained_completed_trade_sample_count") or 0
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
                readiness_metrics = _safe_dict(readiness.get("metrics"))
                training_data_version = str(
                    readiness_metrics.get("training_data_version") or ""
                ).strip()
                required_training_data_version = str(
                    readiness_metrics.get("required_training_data_version") or ""
                ).strip()
                training_data_contract_stale = bool(
                    required_training_data_version
                    and training_data_version != required_training_data_version
                )
                training_data_contract_stale = bool(
                    training_data_contract_stale
                    or (
                        metadata
                        and metadata.get("feature_contract_version")
                        != FEATURE_CONTRACT_VERSION
                    )
                    or (
                        metadata
                        and metadata.get("multitask_prediction_contract_version")
                        != MULTITASK_PREDICTION_CONTRACT_VERSION
                    )
                )
                learning_only = not bool(readiness.get("live_ml_ready"))
                new_samples = max(completed_count - last_completed_count, 0)
                new_trade_samples = max(
                    completed_trade_count - last_completed_trade_count,
                    0,
                )
                trigger_rows = await load_shadow_training_rows()
                trigger_frame = build_training_frame(trigger_rows)
                completed_decision_group_count = (
                    int(trigger_frame["decision_group"].nunique())
                    if not trigger_frame.empty
                    else 0
                )
                previous_group_count = int(
                    cursor_metadata.get(
                        "last_trained_completed_shadow_decision_group_count"
                    )
                    or _safe_dict(
                        cursor_metadata.get("decision_group_partition")
                    ).get("decision_group_count")
                    or 0
                )
                new_decision_group_count = max(
                    completed_decision_group_count - previous_group_count,
                    0,
                )
                current_distribution_profile = _training_distribution_profile(
                    trigger_frame
                )
                distribution_drift = _training_distribution_drift(
                    current_distribution_profile,
                    _safe_dict(cursor_metadata.get("training_distribution_profile")),
                )
                last_trained_at = self._parse_datetime(cursor_metadata.get("trained_at"))
                seconds_since_training = (
                    max((now - last_trained_at).total_seconds(), 0.0)
                    if last_trained_at is not None
                    else None
                )
                batch_due = bool(
                    new_decision_group_count
                    >= _LOCAL_ML_PARAMS.batch_decision_group_threshold
                )
                interval_due = bool(
                    new_decision_group_count
                    >= _LOCAL_ML_PARAMS.minimum_decision_group_increment
                    and seconds_since_training is not None
                    and seconds_since_training
                    >= _LOCAL_ML_PARAMS.maximum_training_interval_seconds
                )
                drift_due = bool(
                    distribution_drift.get("detected") is True
                    and new_decision_group_count
                    >= _LOCAL_ML_PARAMS.drift_minimum_decision_group_increment
                )
                trigger_reason = (
                    "forced"
                    if force
                    else "initial_artifact"
                    if not metadata
                    else "training_data_contract_changed"
                    if training_data_contract_stale
                    else "mature_decision_group_batch"
                    if batch_due
                    else "daily_minimum_increment"
                    if interval_due
                    else "distribution_drift_with_new_labels"
                    if drift_due
                    else "not_due"
                )
                training_policy = {
                    "learning_only": learning_only,
                    "readiness_state": readiness.get("state"),
                    "readiness_blocking_reasons": readiness.get("blocking_reasons") or [],
                    "trigger": trigger_reason,
                    "training_data_contract_stale": training_data_contract_stale,
                    "training_data_version": training_data_version or None,
                    "required_training_data_version": required_training_data_version or None,
                    "cursor_source": "current_training_epoch",
                    "promotion_requires_readiness": True,
                    "candidate_artifact_persisted": False,
                    "persist_artifact_only_when_readiness_allows_live_influence": False,
                    "persist_latest_artifact_even_when_readiness_blocks_live_influence": True,
                    "completed_mature_decision_group_count": completed_decision_group_count,
                    "last_trained_mature_decision_group_count": previous_group_count,
                    "new_mature_decision_group_count": new_decision_group_count,
                    "batch_decision_group_threshold": (
                        _LOCAL_ML_PARAMS.batch_decision_group_threshold
                    ),
                    "minimum_decision_group_increment": (
                        _LOCAL_ML_PARAMS.minimum_decision_group_increment
                    ),
                    "maximum_training_interval_seconds": (
                        _LOCAL_ML_PARAMS.maximum_training_interval_seconds
                    ),
                    "seconds_since_last_successful_training": seconds_since_training,
                    "distribution_drift": distribution_drift,
                }
                should_train = (
                    force
                    or not metadata
                    or training_data_contract_stale
                    or batch_due
                    or interval_due
                    or drift_due
                )
                if not should_train:
                    result = {
                        "trained": False,
                        "reason": "not_due",
                        "completed_sample_count": completed_count,
                        "last_trained_sample_count": last_sample_count,
                        "last_trained_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "completed_trade_sample_count": completed_trade_count,
                        "new_trade_sample_count": new_trade_samples,
                        "completed_decision_group_count": completed_decision_group_count,
                        "new_decision_group_count": new_decision_group_count,
                        "training_policy": training_policy,
                        "message": (
                            "尚未达到独立决策组批次、每日最小增量或漂移训练条件，"
                            "继续使用当前 paper champion。"
                        ),
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
                        trigger_reason=trigger_reason,
                        sample_cursor={
                            "shadow": completed_count,
                            "trade": completed_trade_count,
                        },
                        timeout_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
                    )
                quarantine_result = await self._quarantine_dirty_training_samples()
                completed_count = await self._completed_shadow_sample_count()
                new_samples = max(completed_count - last_completed_count, 0)
                rows = await load_shadow_training_rows()
                quality_state = shadow_training_quality_report(rows)
                frame = build_training_frame(rows)
                partition = decision_group_partition(frame)
                if not partition.report["ready"]:
                    partition_report = partition.report
                    result = {
                        "trained": False,
                        "reason": partition_report["reason"],
                        "completed_sample_count": completed_count,
                        "cost_complete_sample_count": int(len(frame)),
                        "decision_group_count": partition_report["decision_group_count"],
                        "train_sample_count": partition_report["train_sample_count"],
                        "train_decision_group_count": partition_report[
                            "train_decision_group_count"
                        ],
                        "holdout_sample_count": partition_report["holdout_sample_count"],
                        "holdout_decision_group_count": partition_report[
                            "holdout_decision_group_count"
                        ],
                        "purged_training_decision_group_count": partition_report[
                            "purged_training_decision_group_count"
                        ],
                        "purged_training_sample_count": partition_report[
                            "purged_training_sample_count"
                        ],
                        "minimum_train_sample_count": partition_report[
                            "minimum_train_sample_count"
                        ],
                        "minimum_train_decision_group_count": partition_report[
                            "minimum_train_decision_group_count"
                        ],
                        "last_trained_sample_count": last_sample_count,
                        "last_trained_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "completed_trade_sample_count": completed_trade_count,
                        "new_trade_sample_count": new_trade_samples,
                        "training_policy": training_policy,
                        "training_quarantine": quarantine_result,
                        "decision_group_partition": partition_report,
                        "message": (
                            "成本完整样本尚未形成满足时间隔离和最低拟合要求的训练/留出分区；"
                            "本轮按等待数据成熟处理，不写模型产物。"
                        ),
                    }
                    self._last_train_result = result
                    return result
                candidate_metadata = await asyncio.to_thread(
                    train_from_frame,
                    frame,
                    completed_sample_count=completed_count,
                    training_quality_report=quality_state["quality_report"],
                    trade_samples=trade_samples,
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
                    persist_cached_training_candidate,
                    candidate_metadata,
                )
                if trained_metadata is None:
                    trained_metadata = await asyncio.to_thread(
                        train_from_frame,
                        frame,
                        completed_sample_count=completed_count,
                        training_quality_report=quality_state["quality_report"],
                        trade_samples=trade_samples,
                        persist_artifact=True,
                    )
                trained_influence = _influence_policy(trained_metadata)
                trained_readiness = build_ml_readiness_report(
                    trained_metadata,
                    trained_influence,
                )
                production_authorized = bool(
                    trained_readiness.get("live_ml_ready")
                    and trained_readiness.get("state") in {"ready", "partial_ready"}
                    and trained_readiness.get("live_enabled_sides")
                    and not trained_readiness.get("blocking_reasons")
                )
                paper_canary = _safe_dict(trained_readiness.get("paper_canary"))
                paper_canary_authorized = bool(
                    not production_authorized
                    and paper_canary.get("authorized") is True
                    and paper_canary.get("state") == "ready"
                    and paper_canary.get("eligible_sides")
                    and not paper_canary.get("blocking_reasons")
                )
                live_enabled_sides = (
                    list(trained_readiness.get("live_enabled_sides") or [])
                    if production_authorized
                    else []
                )
                activation_stage = (
                    "active"
                    if production_authorized
                    and trained_readiness.get("state") == "ready"
                    and set(live_enabled_sides) == {"long", "short"}
                    else "canary" if production_authorized or paper_canary_authorized else "shadow"
                )
                current_artifact = None
                resolve_current = getattr(self.artifact_registry, "resolve_current", None)
                if callable(resolve_current):
                    current_artifact = resolve_current()
                (
                    champion_manifest,
                    champion_stage,
                    champion_eligibility_errors,
                ) = _champion_comparison_inputs(
                    current_artifact
                    if isinstance(current_artifact, ResolvedModelArtifact)
                    else None
                )
                champion_comparison = compare_candidate_to_champion(
                    trained_metadata,
                    champion_manifest,
                    candidate_stage=activation_stage,
                    champion_stage=champion_stage,
                )
                if champion_eligibility_errors:
                    champion_comparison = {
                        **champion_comparison,
                        "replaced_ineligible_champion": True,
                        "ineligible_champion_version": current_artifact.version,
                        "ineligible_champion_errors": sorted(set(champion_eligibility_errors)),
                    }
                if champion_comparison.get("accepted") is not True:
                    reject_candidate = getattr(self.artifact_registry, "reject_candidate", None)
                    rejected = (
                        reject_candidate(champion_comparison)
                        if callable(reject_candidate)
                        else None
                    )
                    current_activation = _safe_dict(
                        current_artifact.activation_manifest
                        if isinstance(current_artifact, ResolvedModelArtifact)
                        else None
                    )
                    result = {
                        "trained": True,
                        "reason": "trained_challenger_rejected",
                        "challenger_rejected": True,
                        "challenger_version": (
                            rejected.version
                            if isinstance(rejected, ResolvedModelArtifact)
                            else None
                        ),
                        "champion_retained": True,
                        "champion_version": (
                            current_artifact.version
                            if isinstance(current_artifact, ResolvedModelArtifact)
                            else None
                        ),
                        "champion_comparison": champion_comparison,
                        "completed_sample_count": completed_count,
                        "previous_sample_count": last_sample_count,
                        "previous_completed_sample_count": last_completed_count,
                        "new_sample_count": new_samples,
                        "completed_trade_sample_count": completed_trade_count,
                        "previous_completed_trade_sample_count": last_completed_trade_count,
                        "new_trade_sample_count": new_trade_samples,
                        "sample_count": int(trained_metadata.get("sample_count") or 0),
                        "training_quarantine": quarantine_result,
                        "training_policy": training_policy,
                        "candidate": candidate_summary,
                        "candidate_readiness": candidate_readiness,
                        "readiness": readiness,
                        "readiness_state": readiness.get("state"),
                        "live_ml_ready": bool(readiness.get("live_ml_ready")),
                        "artifact_persisted": True,
                        "artifact_version": (
                            current_artifact.version
                            if isinstance(current_artifact, ResolvedModelArtifact)
                            else None
                        ),
                        "artifact_activation_stage": current_activation.get("activation_stage"),
                        "strategy_blueprint": build_model_strategy_blueprint(
                            metadata=_safe_dict(
                                current_artifact.manifest
                                if isinstance(current_artifact, ResolvedModelArtifact)
                                else None
                            ),
                            readiness=readiness,
                            activation=current_activation,
                            artifact_version=(
                                current_artifact.version
                                if isinstance(current_artifact, ResolvedModelArtifact)
                                else None
                            ),
                        ),
                        "paper_canary_authorized": bool(
                            current_activation.get("paper_canary_authorized")
                        ),
                        "live_enabled_sides": list(
                            current_activation.get("live_enabled_sides") or []
                        ),
                        "trained_at": trained_metadata.get("trained_at"),
                        "message": (
                            "本地 ML challenger 已完成训练，但候选费后收益未满足非退化门槛，"
                            "保留当前 champion；训练游标已记录，下一轮继续使用新增样本。"
                        ),
                    }
                    self._last_train_result = result
                    return result
                common_activation_evidence = {
                    "return_evidence_report": trained_readiness,
                    "paper_canary_report": paper_canary,
                    "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
                    "champion_comparison": champion_comparison,
                }
                shadow_activation = {
                    **common_activation_evidence,
                    "activation_stage": "shadow",
                    "readiness_state": trained_readiness.get("state"),
                    "live_ml_ready": False,
                    "paper_canary_authorized": False,
                    "live_enabled_sides": [],
                    "blocking_reasons": (trained_readiness.get("blocking_reasons") or []),
                    "lifecycle_path": ["candidate", "shadow"],
                }
                shadow_activation["strategy_blueprint"] = build_model_strategy_blueprint(
                    metadata=trained_metadata,
                    readiness=trained_readiness,
                    activation=shadow_activation,
                    artifact_version=trained_metadata.get("version"),
                )
                activated_artifact = self.artifact_registry.promote_candidate(shadow_activation)
                if activation_stage in {"canary", "active"}:
                    canary_activation = {
                        **common_activation_evidence,
                        "activation_stage": "canary",
                        "readiness_state": (
                            trained_readiness.get("state")
                            if production_authorized
                            else "paper_canary_ready"
                        ),
                        "live_ml_ready": production_authorized,
                        "paper_canary_authorized": paper_canary_authorized,
                        "live_enabled_sides": live_enabled_sides,
                        "blocking_reasons": [],
                        "lifecycle_path": ["candidate", "shadow", "canary"],
                    }
                    canary_activation["strategy_blueprint"] = build_model_strategy_blueprint(
                        metadata=trained_metadata,
                        readiness=trained_readiness,
                        activation=canary_activation,
                        artifact_version=trained_metadata.get("version"),
                    )
                    activated_artifact = self.artifact_registry.transition_current(
                        canary_activation
                    )
                if activation_stage == "active":
                    active_activation = {
                        **common_activation_evidence,
                        "activation_stage": "active",
                        "readiness_state": "ready",
                        "live_ml_ready": True,
                        "paper_canary_authorized": False,
                        "live_enabled_sides": live_enabled_sides,
                        "blocking_reasons": [],
                        "lifecycle_path": [
                            "candidate",
                            "shadow",
                            "canary",
                            "active",
                        ],
                    }
                    active_activation["strategy_blueprint"] = build_model_strategy_blueprint(
                        metadata=trained_metadata,
                        readiness=trained_readiness,
                        activation=active_activation,
                        artifact_version=trained_metadata.get("version"),
                    )
                    activated_artifact = self.artifact_registry.transition_current(
                        active_activation
                    )
                self._bundle = None
                self._loaded_mtime = None
                self._ensure_loaded()
                live_ml_ready = production_authorized
                result = {
                    "trained": True,
                    "reason": (
                        "trained_active_activated"
                        if activation_stage == "active"
                        else (
                            "trained_canary_activated"
                            if production_authorized
                            else (
                                "trained_paper_canary_activated"
                                if paper_canary_authorized
                                else "trained_shadow_activated"
                            )
                        )
                    ),
                    "completed_sample_count": completed_count,
                    "previous_sample_count": last_sample_count,
                    "previous_completed_sample_count": last_completed_count,
                    "new_sample_count": new_samples,
                    "completed_trade_sample_count": completed_trade_count,
                    "previous_completed_trade_sample_count": last_completed_trade_count,
                    "new_trade_sample_count": new_trade_samples,
                    "sample_count": int(trained_metadata.get("sample_count") or 0),
                    "last_trained_completed_sample_count": int(
                        trained_metadata.get("last_trained_completed_shadow_sample_count")
                        or completed_count
                    ),
                    "training_quarantine": quarantine_result,
                    "training_policy": training_policy,
                    "candidate": candidate_summary,
                    "champion_comparison": champion_comparison,
                    "candidate_readiness": candidate_readiness,
                    "candidate_influence_policy": candidate_influence,
                    "readiness": trained_readiness,
                    "readiness_state": trained_readiness.get("state"),
                    "live_ml_ready": live_ml_ready,
                    "influence_policy": trained_influence,
                    "artifact_persisted": bool(trained_metadata.get("artifact_persisted")),
                    "artifact_version": activated_artifact.version,
                    "artifact_activation_stage": activation_stage,
                    "strategy_blueprint": self._loaded_strategy_blueprint(),
                    "paper_canary_authorized": paper_canary_authorized,
                    "paper_canary": paper_canary,
                    "live_enabled_sides": live_enabled_sides,
                    "trained_at": trained_metadata.get("trained_at"),
                    "message": (
                        "本地 ML 候选已通过双边费后收益证据，并按 shadow → canary → "
                        "active 顺序原子激活为统一生产模型。"
                        if activation_stage == "active"
                        else (
                            "本地 ML 候选已通过单边费后收益证据并原子激活为 canary，"
                            "仅允许证据达标方向影响生产。"
                            if production_authorized
                            else (
                                "本地 ML 候选已通过数据治理与时间滚动完整性检查，原子激活为"
                                "模拟盘 Paper Canary；可参与正常模拟盘分析和交易，但不拥有实盘权限。"
                                if paper_canary_authorized
                                else "本地 ML 候选已完成完整性验证并原子激活为 shadow；"
                                "生产影响保持关闭，等待收益证据达标。"
                            )
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

    def predict_strategy_replay_batch(
        self,
        feature_rows: list[dict[str, Any]],
        *,
        horizon_minutes: int = 10,
    ) -> list[dict[str, Any]]:
        """Return compact model decisions for historical strategy replay."""

        self._ensure_loaded()
        if not self._bundle or not feature_rows:
            return [
                {
                    "available": False,
                    "model_version": None,
                    "predictions": [],
                }
                for _row in feature_rows
            ]
        metadata = _safe_dict(self._bundle.get("metadata"))
        frame = pd.DataFrame(
            [
                _feature_row_from_feature_vector(
                    features,
                    horizon_minutes=horizon_minutes,
                )
                for features in feature_rows
            ],
            columns=FEATURE_KEYS,
        )
        long_distribution = _regression_prediction_distribution(
            self._bundle["long_regressor"],
            frame,
        )
        short_distribution = _regression_prediction_distribution(
            self._bundle["short_regressor"],
            frame,
        )
        long_cost_distribution = _regression_prediction_distribution(
            self._bundle["long_cost_regressor"],
            frame,
        )
        short_cost_distribution = _regression_prediction_distribution(
            self._bundle["short_cost_regressor"],
            frame,
        )
        long_tail_model = self._bundle.get("long_tail_classifier")
        short_tail_model = self._bundle.get("short_tail_classifier")
        long_tail_probabilities = (
            _optional_positive_proba(long_tail_model, frame, default=0.0)
            if long_tail_model is not None
            else np.asarray([float("nan")] * len(frame), dtype=float)
        )
        short_tail_probabilities = (
            _optional_positive_proba(short_tail_model, frame, default=0.0)
            if short_tail_model is not None
            else np.asarray([float("nan")] * len(frame), dtype=float)
        )
        tail_scales = _safe_dict(metadata.get("tail_loss_scale_pct"))
        long_tail_scale = max(_safe_float(tail_scales.get("long"), 0.0), 0.0)
        short_tail_scale = max(_safe_float(tail_scales.get("short"), 0.0), 0.0)
        model_version = str(
            self._artifact_version(metadata)
            or metadata.get("artifact_version")
            or metadata.get("version")
            or ""
        )
        results: list[dict[str, Any]] = []
        for index, features in enumerate(feature_rows):
            long_tail_probability = (
                None
                if not math.isfinite(float(long_tail_probabilities[index]))
                else float(long_tail_probabilities[index])
            )
            short_tail_probability = (
                None
                if not math.isfinite(float(short_tail_probabilities[index]))
                else float(short_tail_probabilities[index])
            )
            long_contract = _standardized_model_return_distribution(
                long_distribution,
                index,
                side="long",
                horizon_minutes=horizon_minutes,
                tail_loss_probability=long_tail_probability,
                tail_loss_scale_pct=long_tail_scale,
            )
            short_contract = _standardized_model_return_distribution(
                short_distribution,
                index,
                side="short",
                horizon_minutes=horizon_minutes,
                tail_loss_probability=short_tail_probability,
                tail_loss_scale_pct=short_tail_scale,
            )
            long_objective = _safe_float(
                long_contract.get("objective_expected_return_pct"),
                float("nan"),
            )
            short_objective = _safe_float(
                short_contract.get("objective_expected_return_pct"),
                float("nan"),
            )
            long_rank = (
                long_objective
                if long_contract.get("production_eligible") is True
                else float("-inf")
            )
            short_rank = (
                short_objective
                if short_contract.get("production_eligible") is True
                else float("-inf")
            )
            if not math.isfinite(long_rank) and not math.isfinite(short_rank):
                long_rank = float(long_distribution["expected"][index])
                short_rank = float(short_distribution["expected"][index])
            best_side = "long" if long_rank >= short_rank else "short"
            symbol = str(features.get("symbol") or "")
            actual_calibration = {
                side: select_trade_calibration(
                    _safe_dict(metadata.get("actual_trade_calibration")),
                    symbol=symbol,
                    side=side,
                )
                for side in ("long", "short")
            }
            long_cost_ready = _distribution_ready_at(long_cost_distribution, index)
            short_cost_ready = _distribution_ready_at(short_cost_distribution, index)
            results.append(
                {
                    "available": True,
                    "model_version": model_version,
                    "predictions": [
                        {
                            "horizon_minutes": horizon_minutes,
                            "best_side": best_side,
                            "actual_trade_calibration_ready": _actual_calibration_ready(
                                _safe_dict(actual_calibration.get(best_side))
                            ),
                            "return_distribution_contract": {
                                "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
                                "long": long_contract,
                                "short": short_contract,
                            },
                            "counterfactual_execution_cost_distribution": {
                                "long": {
                                    "distribution_ready": long_cost_ready,
                                },
                                "short": {
                                    "distribution_ready": short_cost_ready,
                                },
                            },
                        }
                    ],
                }
            )
        return results

    def predict(
        self,
        features: Any,
        *,
        horizons: tuple[int, ...] = (5, 15, 60, 240),
    ) -> dict[str, Any]:
        self._ensure_loaded()
        if not self._bundle:
            diagnostic = self._model_unavailable_diagnostic()
            readiness = disabled_ml_readiness(
                diagnostic["code"],
                diagnostic["message"],
            )
            return {
                "available": False,
                "status": diagnostic["code"],
                "readiness_state": readiness["state"],
                "readiness": readiness,
                "live_ml_ready": False,
                "model_load_diagnostic": diagnostic,
                "strategy_blueprint": build_model_strategy_blueprint(
                    metadata=None,
                    readiness=readiness,
                    activation=None,
                ),
                "message": diagnostic["message"],
            }
        metadata = _safe_dict(self._bundle.get("metadata"))
        influence = _influence_policy(metadata)
        readiness = build_ml_readiness_report(metadata, influence)
        influence, readiness = _activation_gated_policy(
            influence,
            readiness,
            self._resolved_artifact,
        )
        live_ml_ready = bool(readiness.get("live_ml_ready"))
        advisory_enabled = bool(
            influence.get("advisory_enabled") and readiness.get("state") == "shadow_ready"
        )
        tail_scales = _safe_dict(metadata.get("tail_loss_scale_pct"))
        long_tail_scale = max(_safe_float(tail_scales.get("long"), 0.0), 0.0)
        short_tail_scale = max(_safe_float(tail_scales.get("short"), 0.0), 0.0)
        model_version = str(
            self._artifact_version(metadata)
            or metadata.get("artifact_version")
            or metadata.get("version")
            or ""
        )
        calibration_version = str(
            _safe_dict(metadata.get("actual_trade_calibration")).get("data_fingerprint")
            or "uncalibrated_time_series_v1"
        )
        feature_symbol = (
            str(features.get("symbol") or "")
            if isinstance(features, dict)
            else str(getattr(features, "symbol", "") or "")
        )

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
            long_cost_distribution = _regression_prediction_distribution(
                self._bundle["long_cost_regressor"], x
            )
            short_cost_distribution = _regression_prediction_distribution(
                self._bundle["short_cost_regressor"], x
            )
            long_mfe = _optional_regression_value(self._bundle, "long_mfe_regressor", x)
            long_mae = _optional_regression_value(self._bundle, "long_mae_regressor", x)
            short_mfe = _optional_regression_value(self._bundle, "short_mfe_regressor", x)
            short_mae = _optional_regression_value(self._bundle, "short_mae_regressor", x)
            raw_long_expected = float(long_distribution["expected"][0])
            raw_short_expected = float(short_distribution["expected"][0])
            long_lower_quantile = float(long_distribution["lower_quantile"][0])
            short_lower_quantile = float(short_distribution["lower_quantile"][0])
            long_cost_distribution_ready = _distribution_ready_at(
                long_cost_distribution,
                0,
            )
            short_cost_distribution_ready = _distribution_ready_at(
                short_cost_distribution,
                0,
            )
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
            long_return_contract = _standardized_model_return_distribution(
                long_distribution,
                0,
                side="long",
                horizon_minutes=int(horizon),
                tail_loss_probability=long_tail_loss_probability,
                tail_loss_scale_pct=long_tail_scale,
            )
            short_return_contract = _standardized_model_return_distribution(
                short_distribution,
                0,
                side="short",
                horizon_minutes=int(horizon),
                tail_loss_probability=short_tail_loss_probability,
                tail_loss_scale_pct=short_tail_scale,
            )
            multitask_prediction = {
                "version": MULTITASK_PREDICTION_CONTRACT_VERSION,
                "long": _multitask_side_prediction(
                    side="long",
                    horizon_minutes=int(horizon),
                    model_version=model_version,
                    calibration_version=calibration_version,
                    return_contract=long_return_contract,
                    cost_distribution=long_cost_distribution,
                    win_probability=long_win_rate,
                    tail_loss_probability=long_tail_loss_probability,
                    expected_mfe_pct=long_mfe,
                    expected_mae_pct=long_mae,
                ),
                "short": _multitask_side_prediction(
                    side="short",
                    horizon_minutes=int(horizon),
                    model_version=model_version,
                    calibration_version=calibration_version,
                    return_contract=short_return_contract,
                    cost_distribution=short_cost_distribution,
                    win_probability=short_win_rate,
                    tail_loss_probability=short_tail_loss_probability,
                    expected_mfe_pct=short_mfe,
                    expected_mae_pct=short_mae,
                ),
            }
            long_market_distribution_ready = bool(long_return_contract.get("production_eligible"))
            short_market_distribution_ready = bool(short_return_contract.get("production_eligible"))
            long_objective_expected = _safe_float(
                long_return_contract.get("objective_expected_return_pct"),
                float("nan"),
            )
            short_objective_expected = _safe_float(
                short_return_contract.get("objective_expected_return_pct"),
                float("nan"),
            )
            long_rank = long_objective_expected if long_market_distribution_ready else float("-inf")
            short_rank = (
                short_objective_expected if short_market_distribution_ready else float("-inf")
            )
            if not math.isfinite(long_rank) and not math.isfinite(short_rank):
                long_rank = raw_long_expected
                short_rank = raw_short_expected
            best_side = "long" if long_rank >= short_rank else "short"
            best_win = long_win_rate if best_side == "long" else short_win_rate
            best_objective_expected = (
                long_objective_expected if best_side == "long" else short_objective_expected
            )
            best_raw_expected = raw_long_expected if best_side == "long" else raw_short_expected
            best_scoring_expected = (
                best_objective_expected
                if math.isfinite(best_objective_expected)
                else best_raw_expected
            )
            best_tail_loss_probability = (
                long_tail_loss_probability if best_side == "long" else short_tail_loss_probability
            )
            best_lower_quantile = (
                long_lower_quantile if best_side == "long" else short_lower_quantile
            )
            selected_market_distribution_ready = (
                long_market_distribution_ready
                if best_side == "long"
                else short_market_distribution_ready
            )
            selected_cost_distribution_ready = (
                long_cost_distribution_ready
                if best_side == "long"
                else short_cost_distribution_ready
            )
            actual_calibration = {
                "long": select_trade_calibration(
                    _safe_dict(metadata.get("actual_trade_calibration")),
                    symbol=feature_symbol,
                    side="long",
                ),
                "short": select_trade_calibration(
                    _safe_dict(metadata.get("actual_trade_calibration")),
                    symbol=feature_symbol,
                    side="short",
                ),
            }
            selected_actual_calibration_ready = _actual_calibration_ready(
                _safe_dict(actual_calibration.get(best_side))
            )
            selected_return_contract = (
                long_return_contract if best_side == "long" else short_return_contract
            )
            profit_edge = abs(
                (
                    long_objective_expected
                    if math.isfinite(long_objective_expected)
                    else raw_long_expected
                )
                - (
                    short_objective_expected
                    if math.isfinite(short_objective_expected)
                    else raw_short_expected
                )
            )
            profit_quality = _profit_quality_score(
                best_scoring_expected,
                best_lower_quantile,
                profit_edge,
                float(best_tail_loss_probability or 0.0),
                long_tail_scale if best_side == "long" else short_tail_scale,
            )
            side_influence = _safe_dict(influence.get(best_side))
            downside = max(-best_scoring_expected, 0.0) + max(
                -best_lower_quantile,
                0.0,
            )
            return_scale = abs(best_scoring_expected) + abs(best_lower_quantile)
            risk_score = _clamp(
                downside / max(return_scale, 1e-9) + float(best_tail_loss_probability or 0.0)
            )
            paper_prediction_eligible = bool(
                best_side in {"long", "short"}
                and math.isfinite(best_raw_expected)
                and math.isfinite(
                    _safe_float(
                        _safe_dict(multitask_prediction.get(best_side)).get(
                            "expected_execution_cost_pct"
                        ),
                        float("nan"),
                    )
                )
            )
            predictions.append(
                {
                    "horizon_minutes": int(horizon),
                    "long_win_rate": round(long_win_rate, 4),
                    "short_win_rate": round(short_win_rate, 4),
                    "return_distribution_contract_version": (RETURN_DISTRIBUTION_CONTRACT_VERSION),
                    "return_distribution_contract": {
                        "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
                        "long": long_return_contract,
                        "short": short_return_contract,
                    },
                    "multitask_prediction": multitask_prediction,
                    "counterfactual_execution_cost_distribution": {
                        "long": {
                            "expected_pct": round(float(long_cost_distribution["expected"][0]), 4),
                            "upper_tail_pct": round(
                                float(long_cost_distribution["upper_quantile"][0]), 4
                            ),
                            "uncertainty_pct": round(float(long_cost_distribution["std"][0]), 4),
                            "source_authority": ("shadow_counterfactual_live_microstructure"),
                            "distribution_ready": long_cost_distribution_ready,
                        },
                        "short": {
                            "expected_pct": round(float(short_cost_distribution["expected"][0]), 4),
                            "upper_tail_pct": round(
                                float(short_cost_distribution["upper_quantile"][0]), 4
                            ),
                            "uncertainty_pct": round(float(short_cost_distribution["std"][0]), 4),
                            "source_authority": ("shadow_counterfactual_live_microstructure"),
                            "distribution_ready": short_cost_distribution_ready,
                        },
                        "source_authority": "shadow_counterfactual_live_microstructure",
                    },
                    "actual_trade_calibration": actual_calibration,
                    "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
                    "return_semantics": "gross_market_opportunity_before_execution",
                    "best_side": best_side,
                    "best_win_rate": round(best_win, 4),
                    "profit_edge_pct": round(profit_edge, 4),
                    "profit_quality_score": round(profit_quality, 4),
                    "profit_signal": bool(
                        live_ml_ready
                        and side_influence.get("enabled")
                        and selected_market_distribution_ready
                        and selected_cost_distribution_ready
                        and selected_actual_calibration_ready
                        and best_objective_expected > 0.0
                        and best_lower_quantile > 0.0
                        and profit_edge > 0.0
                    ),
                    "risk_score": round(risk_score, 4),
                    "paper_prediction_eligible": paper_prediction_eligible,
                    "ml_prediction_eligible": bool(
                        live_ml_ready
                        and side_influence.get("enabled")
                        and selected_market_distribution_ready
                        and selected_cost_distribution_ready
                        and selected_actual_calibration_ready
                    ),
                    "selected_return_distribution_blockers": list(
                        selected_return_contract.get("blockers") or []
                    ),
                    "actual_trade_calibration_ready": (selected_actual_calibration_ready),
                }
            )

        primary = predictions[0] if predictions else {}
        primary_side = str(primary.get("best_side") or "")
        primary_cost_distribution = _safe_dict(
            _safe_dict(primary.get("counterfactual_execution_cost_distribution")).get(primary_side)
        )
        primary_return_distribution = _safe_dict(
            _safe_dict(primary.get("return_distribution_contract")).get(primary_side)
        )
        current_prediction_ready = bool(
            primary
            and primary_side in {"long", "short"}
            and primary_side in set(readiness.get("live_enabled_sides") or [])
            and primary_return_distribution.get("version") == RETURN_DISTRIBUTION_CONTRACT_VERSION
            and primary_return_distribution.get("production_eligible") is True
            and primary_cost_distribution.get("distribution_ready") is True
            and primary.get("actual_trade_calibration_ready") is True
        )
        live_prediction_influence = bool(live_ml_ready and current_prediction_ready)
        paper_ml_ready = bool(
            primary and primary.get("paper_prediction_eligible") is True
        )
        prediction_contract_complete = bool(
            paper_ml_ready
            and primary_return_distribution.get("production_eligible") is True
        )
        structural_blockers = list(
            primary_return_distribution.get("blockers") or []
        )
        if not paper_ml_ready:
            structural_blockers.append("paper_prediction_contract_incomplete")
        production_blockers = [
            *structural_blockers,
            *(
                []
                if primary_cost_distribution.get("distribution_ready") is True
                else ["counterfactual_execution_cost_distribution_incomplete"]
            ),
            *(
                []
                if primary.get("actual_trade_calibration_ready") is True
                else ["authoritative_actual_trade_calibration_incomplete"]
            ),
            *([] if live_ml_ready else ["model_not_promoted_for_live"]),
        ]
        activation = _safe_dict(
            self._resolved_artifact.activation_manifest
            if self._resolved_artifact is not None
            else None
        )
        paper_canary = _safe_dict(readiness.get("paper_canary"))
        paper_canary_authorized = bool(
            activation.get("activation_stage") == "canary"
            and activation.get("paper_canary_authorized") is True
            and activation.get("live_ml_ready") is not True
            and paper_canary.get("authorized") is True
        )
        strategy_blueprint = build_model_strategy_blueprint(
            metadata=metadata,
            readiness=readiness,
            activation=activation,
            artifact_version=self._artifact_version(metadata),
        )
        return {
            "available": True,
            "route_mode": ("live" if live_prediction_influence else "paper_analysis"),
            "paper_ml_ready": paper_ml_ready,
            "live_ml_ready": live_ml_ready,
            "production_permission": live_prediction_influence,
            "objective_name": metadata.get("objective_name"),
            "objective_version": metadata.get("objective_version"),
            "label_name": metadata.get("label_name"),
            "label_version": metadata.get("label_version"),
            "training_cost_policy": metadata.get("training_cost_policy"),
            "artifact_persisted": metadata.get("artifact_persisted") is True,
            "artifact_lifecycle": _safe_dict(
                self._resolved_artifact.activation_manifest
                if self._resolved_artifact is not None
                else None
            ).get("activation_stage")
            or "unregistered",
            "paper_canary_authorized": paper_canary_authorized,
            "paper_canary": paper_canary,
            "strategy_blueprint": strategy_blueprint,
            "return_distribution_contract_version": (RETURN_DISTRIBUTION_CONTRACT_VERSION),
            "prediction_quality": {
                "contract_complete": prediction_contract_complete,
                "paper_eligible": prediction_contract_complete,
                "production_eligible": live_prediction_influence,
                "anomalous": not prediction_contract_complete,
                "reason": (
                    "separated_market_cost_and_actual_calibration_ready"
                    if live_prediction_influence
                    else "standardized_return_distribution_ready_for_paper"
                    if prediction_contract_complete
                    else "current_prediction_contract_incomplete"
                ),
                "blockers": list(dict.fromkeys(structural_blockers)),
                "production_blockers": list(dict.fromkeys(production_blockers)),
            },
            "status": (
                "entry_profit_filter"
                if live_prediction_influence
                else (
                    "advisory"
                    if advisory_enabled
                    else str(readiness.get("state") or "learning_only")
                )
            ),
            "mode": (
                "entry_profit_filter"
                if live_prediction_influence
                else (
                    "advisory"
                    if advisory_enabled
                    else str(readiness.get("state") or "learning_only")
                )
            ),
            "readiness_state": readiness.get("state"),
            "readiness": readiness,
            "prediction_eligible": live_prediction_influence,
            "advisory_enabled": advisory_enabled,
            "influence_policy": influence,
            "model_version": strategy_blueprint.get("model_version"),
            "trained_sample_count": int(metadata.get("sample_count") or 0),
            "primary_horizon_minutes": primary.get("horizon_minutes"),
            "long_win_rate": primary.get("long_win_rate"),
            "short_win_rate": primary.get("short_win_rate"),
            "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
            "return_semantics": "gross_market_opportunity_before_execution",
            "return_distribution_contract": primary.get("return_distribution_contract"),
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
        resolved_in_attempt: ResolvedModelArtifact | None = None
        try:
            trusted_root = MODEL_DIR
            if self._explicit_model_path is None:
                pointer_mtime_ns = (
                    self.artifact_registry.current_path.stat().st_mtime_ns
                    if self.artifact_registry.current_path.exists()
                    else None
                )
                if (
                    self._resolved_artifact is not None
                    and self._loaded_pointer_mtime_ns == pointer_mtime_ns
                    and self.model_path.exists()
                    and self._loaded_mtime == self.model_path.stat().st_mtime
                    and (
                        self._bundle is not None
                        or _safe_dict(self._load_diagnostic).get("code")
                        == "artifact_incompatible"
                    )
                ):
                    return
                current = self.artifact_registry.resolve_current()
                if current is None:
                    self._bundle = None
                    self._loaded_mtime = None
                    self._loaded_pointer_mtime_ns = pointer_mtime_ns
                    self._resolved_artifact = None
                    self._load_diagnostic = {
                        "code": "no_model",
                        "message": "本地 ML 尚未注册当前模型 Artifact。",
                        "details": [],
                    }
                    return
                resolved_in_attempt = current
                self.model_path = current.model_path
                self.metadata_path = current.metadata_path
                trusted_root = self.artifact_registry.model_root
                self._loaded_pointer_mtime_ns = pointer_mtime_ns
                self._resolved_artifact = current
            if not self.model_path.exists():
                self._bundle = None
                self._loaded_mtime = None
                self._load_diagnostic = {
                    "code": "artifact_load_failed" if self._resolved_artifact else "no_model",
                    "message": (
                        "当前已注册模型 Artifact 文件不存在，运行时已禁止使用。"
                        if self._resolved_artifact
                        else "本地 ML 模型 Artifact 文件不存在。"
                    ),
                    "details": ["artifact_model_file_missing"],
                }
                return
            mtime = self.model_path.stat().st_mtime
            if self._loaded_mtime == mtime and (
                self._bundle is not None
                or _safe_dict(self._load_diagnostic).get("code") == "artifact_incompatible"
            ):
                return
            self._bundle = load_trusted_joblib(
                self.model_path,
                trusted_root=trusted_root,
                expected_type=dict,
            )
            _configure_single_row_inference(self._bundle)
            metadata = _safe_dict(self._bundle.get("metadata"))
            compatibility_errors = local_ml_artifact_compatibility_errors(
                metadata,
                bundle=self._bundle,
            )
            if compatibility_errors:
                self._bundle = None
                self._loaded_mtime = mtime
                self._load_diagnostic = {
                    "code": "artifact_incompatible",
                    "message": "当前已注册模型与运行时收益监督合同不兼容，已禁止加载。",
                    "details": compatibility_errors,
                }
                logger.warning(
                    "refusing incompatible ML signal artifact",
                    path=str(self.model_path),
                    errors=compatibility_errors,
                )
                return
            self._loaded_mtime = mtime
            self._load_diagnostic = None
        except Exception as exc:
            error_text = safe_error_text(exc)
            logger.warning(
                "failed to load ML signal model",
                path=str(self.model_path),
                error=error_text,
            )
            self._bundle = None
            self._loaded_mtime = None
            self._loaded_pointer_mtime_ns = None
            self._resolved_artifact = resolved_in_attempt
            self._load_diagnostic = {
                "code": "artifact_load_failed",
                "message": "当前模型 Artifact 加载失败，运行时已禁止使用。",
                "details": [error_text],
            }

    def _model_unavailable_diagnostic(self) -> dict[str, Any]:
        return self._load_diagnostic or {
            "code": "no_model",
            "message": "本地 ML 尚未注册当前模型 Artifact。",
            "details": [],
        }

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
            "activation_manifest": current.activation_manifest,
        }

    def _auto_train_status(self) -> dict[str, Any]:
        persistent = self.training_state_store.read()
        models = persistent.get("models") if isinstance(persistent.get("models"), dict) else {}
        row = models.get(LOCAL_ML_MODEL_IDS[0]) if isinstance(models, dict) else {}
        row = row if isinstance(row, dict) else {}
        return {
            "auto_train_enabled": True,
            "auto_train_check_interval_seconds": AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
            "auto_train_trigger": (
                "50_new_mature_decision_groups_or_10_after_24h_or_drift_with_10"
            ),
            "auto_train_distribution_requirement": (
                "chronological_purged_holdout_with_minimum_fit_distribution"
            ),
            "auto_training": row.get("state") == "running",
            "auto_train_last_check_at": row.get("last_check_at") or self._last_check_at,
            "auto_train_next_check_at": row.get("next_check_at") or self._next_check_at,
            "auto_train_last_started_at": row.get("last_started_at") or self._last_train_started_at,
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

    def _training_cursor_metadata(
        self,
        current_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Advance scheduling past an evaluated but rejected challenger."""

        selected = dict(current_metadata)
        resolve_challenger = getattr(self.artifact_registry, "resolve_challenger", None)
        if not callable(resolve_challenger):
            return selected
        try:
            challenger = resolve_challenger()
        except Exception as exc:
            logger.warning(
                "failed to resolve rejected ML challenger cursor",
                error=safe_error_text(exc),
            )
            return selected
        if not isinstance(challenger, ResolvedModelArtifact):
            return selected
        challenger_metadata = _safe_dict(challenger.manifest)
        current_cursor = int(selected.get("last_trained_completed_shadow_sample_count") or 0)
        challenger_cursor = int(
            challenger_metadata.get("last_trained_completed_shadow_sample_count") or 0
        )
        current_trade_cursor = int(selected.get("last_trained_completed_trade_sample_count") or 0)
        challenger_trade_cursor = int(
            challenger_metadata.get("last_trained_completed_trade_sample_count") or 0
        )
        current_group_cursor = int(
            selected.get("last_trained_completed_shadow_decision_group_count") or 0
        )
        challenger_group_cursor = int(
            challenger_metadata.get(
                "last_trained_completed_shadow_decision_group_count"
            )
            or 0
        )
        if (challenger_group_cursor, challenger_cursor, challenger_trade_cursor) > (
            current_group_cursor,
            current_cursor,
            current_trade_cursor,
        ):
            return challenger_metadata
        return selected

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
        side_key = str(primary.get("best_side") or "")
        distribution = _safe_dict(
            _safe_dict(primary.get("return_distribution_contract")).get(side_key)
        )
        expected = _safe_float(
            distribution.get("objective_expected_return_pct"),
            0.0,
        )
        edge = float(primary.get("profit_edge_pct") or 0.0)
        lower_quantile = _safe_float(
            distribution.get("lower_quantile_return_pct"),
            0.0,
        )
        tail_probability = _safe_float(
            distribution.get("tail_loss_probability"),
            0.0,
        )
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
    epoch_start = load_training_epoch_start()
    base_filters = (
        ShadowBacktest.status == "completed",
        ShadowBacktest.created_at >= epoch_start,
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
        rows = [_shadow_training_row_from_mapping(row) for row in result.mappings().all()]
    return select_shadow_training_rows(rows)


async def count_shadow_training_rows() -> int:
    epoch_start = load_training_epoch_start()
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(func.count(ShadowBacktest.id)).where(
                ShadowBacktest.status == "completed",
                ShadowBacktest.created_at >= epoch_start,
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            )
        )
        return int(result.scalar() or 0)


async def load_authoritative_trade_training_samples() -> list[dict[str, Any]]:
    """Load the clean OKX lifecycle view used only for realized calibration."""

    from scripts.train_local_ai_tools_models import _load_trade_samples

    annotated = annotate_samples(
        await _load_trade_samples(),
        "trade",
    )
    return [sample for sample in annotated if not sample.get("exclude_from_training")]


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
