"""Local ML profit-quality model built from shadow backtest outcomes.

The model is intentionally used as an observation signal first. It predicts
statistical long/short profit quality from market features, but does not
execute trades by itself.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from core.model_artifact_safety import dump_trusted_joblib, load_trusted_joblib
from core.safe_output import safe_error_text
from db.repositories.memory_repo import MemoryRepository
from db.session import get_session_ctx

logger = structlog.get_logger(__name__)

MODEL_DIR = Path("data/ml_signal")
MODEL_PATH = MODEL_DIR / "winrate_model.joblib"
METADATA_PATH = MODEL_DIR / "winrate_model_metadata.json"
AUTO_TRAIN_CHECK_INTERVAL_SECONDS = 30 * 60
AUTO_TRAIN_MIN_INTERVAL_SECONDS = 6 * 60 * 60
AUTO_TRAIN_MIN_NEW_SAMPLES = 500

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

WIN_RETURN_THRESHOLD_PCT = 0.05
ROUND_TRIP_COST_PCT = 0.12
TAIL_LOSS_THRESHOLD_PCT = 0.18
MIN_PROFIT_EDGE_PCT = 0.02
MIN_PROFIT_SIGNAL_WIN_RATE = 0.0
MIN_TRAINING_SAMPLES = 200
ML_INFLUENCE_MIN_SAMPLE_COUNT = 1000
ML_INFLUENCE_MIN_TEST_COUNT = 200
ML_INFLUENCE_MIN_AUC = 0.53
ML_INFLUENCE_MIN_ACCURACY = 0.52
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
    accuracy = _safe_float(metrics.get(f"{side}_accuracy"), 0.0)
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), 0.0)
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), 0.0)
    top_win = _safe_float(metrics.get(f"top_{side}_win_rate"), 0.0)
    bottom_win = _safe_float(metrics.get(f"bottom_{side}_win_rate"), 0.0)

    reasons: list[str] = []
    if sample_count < ML_INFLUENCE_MIN_SAMPLE_COUNT:
        reasons.append(f"样本数 {sample_count} < {ML_INFLUENCE_MIN_SAMPLE_COUNT}")
    if test_count < ML_INFLUENCE_MIN_TEST_COUNT:
        reasons.append(f"测试样本 {test_count} < {ML_INFLUENCE_MIN_TEST_COUNT}")
    if auc < ML_INFLUENCE_MIN_AUC:
        reasons.append(f"AUC {auc:.3f} < {ML_INFLUENCE_MIN_AUC:.2f}")
    if accuracy < ML_INFLUENCE_MIN_ACCURACY:
        reasons.append(f"准确率 {accuracy:.3f} < {ML_INFLUENCE_MIN_ACCURACY:.2f}")
    if top_return <= ML_INFLUENCE_MIN_TOP_RETURN_PCT:
        reasons.append(
            f"高分组平均收益 {top_return:.3f}% <= {ML_INFLUENCE_MIN_TOP_RETURN_PCT:.2f}%"
        )
    if top_win <= bottom_win:
        reasons.append(f"高分组胜率 {top_win:.3f} 未优于低分组 {bottom_win:.3f}")

    enabled = not reasons
    return {
        "enabled": enabled,
        "status": "active" if enabled else "learning_only",
        "side": side,
        "auc": round(auc, 4),
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
    disabled_reasons: list[str] = []
    if not long_status.get("enabled"):
        disabled_reasons.append("做多：" + "；".join(long_status.get("reasons") or ["未达标"]))
    if not short_status.get("enabled"):
        disabled_reasons.append("做空：" + "；".join(short_status.get("reasons") or ["未达标"]))
    return {
        "enabled": enabled,
        "mode": "entry_profit_filter" if enabled else "learning_only",
        "status": "active" if enabled else "learning_only",
        "long": long_status,
        "short": short_status,
        "disabled_reason": "；".join(disabled_reasons) if disabled_reasons else "",
        "rule": (
            "ML 只有在样本数、测试样本、AUC、准确率、高分组收益和分组胜率同时达标时，"
            "才允许影响开仓过滤、加分和机会排序；否则继续预测、记录和训练，但不介入交易。"
        ),
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
                "raw_long_return_pct": _safe_float(raw_long_return),
                "raw_short_return_pct": _safe_float(raw_short_return),
                "long_return_pct": long_return,
                "short_return_pct": short_return,
                "long_tail_loss": int(long_return < -TAIL_LOSS_THRESHOLD_PCT),
                "short_tail_loss": int(short_return < -TAIL_LOSS_THRESHOLD_PCT),
                "long_win": int(long_return > WIN_RETURN_THRESHOLD_PCT),
                "short_win": int(short_return > WIN_RETURN_THRESHOLD_PCT),
            }
        )
        data.append(feature_row)
    return pd.DataFrame(data)


def train_from_frame(
    frame: pd.DataFrame, *, min_samples: int = MIN_TRAINING_SAMPLES
) -> dict[str, Any]:
    if len(frame) < min_samples:
        raise ValueError(f"训练样本不足：{len(frame)} < {min_samples}")

    frame = frame.sort_values("id").reset_index(drop=True)
    split = max(int(len(frame) * 0.75), 1)
    if len(frame) - split < 40:
        split = max(len(frame) - 40, 1)

    train = frame.iloc[:split].copy()
    test = frame.iloc[split:].copy()
    x_train = train[FEATURE_KEYS]
    x_test = test[FEATURE_KEYS]

    long_classifier = _make_classifier(train["long_win"])
    short_classifier = _make_classifier(train["short_win"])
    long_regressor = _make_regressor(train["long_return_pct"])
    short_regressor = _make_regressor(train["short_return_pct"])

    long_classifier.fit(x_train, train["long_win"])
    short_classifier.fit(x_train, train["short_win"])
    long_regressor.fit(x_train, train["long_return_pct"])
    short_regressor.fit(x_train, train["short_return_pct"])

    long_scores = _positive_proba(long_classifier, x_test)
    short_scores = _positive_proba(short_classifier, x_test)
    long_expected_scores = long_regressor.predict(x_test)
    short_expected_scores = short_regressor.predict(x_test)
    long_pred = (long_scores >= 0.50).astype(int)
    short_pred = (short_scores >= 0.50).astype(int)

    now = datetime.now(UTC).isoformat()
    metadata = {
        "version": now,
        "trained_at": now,
        "sample_count": int(len(frame)),
        "training_shadow_sample_count": int(len(frame)),
        "training_shadow_sample_limit": 20000,
        "training_sample_note": "sample_count is the latest training window, not the all-time total.",
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
        "feature_keys": FEATURE_KEYS,
        "mode": "entry_profit_filter",
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
    dump_trusted_joblib(bundle, MODEL_PATH, trusted_root=MODEL_DIR)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
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
            return {
                "available": False,
                "status": "no_model",
                "model_path": str(self.model_path),
                "message": "本地 ML 盈亏质量模型尚未训练。",
                **auto_status,
            }
        metadata = _safe_dict(self._bundle.get("metadata"))
        influence = _influence_policy(metadata)
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
                metadata.get("training_shadow_sample_limit") or 20000
            ),
            "training_sample_note": metadata.get("training_sample_note")
            or "sample_count is the latest training window, not the all-time total.",
            "status": "ready" if influence.get("enabled") else "learning_only",
            "mode": influence.get("mode"),
            "influence_enabled": bool(influence.get("enabled")),
            "influence_policy": influence,
            "model_note": model_note,
            "note": (
                "ML 指标达标，当前允许参与开仓过滤、加分和机会排序。"
                if influence.get("enabled")
                else "ML 指标未达标，当前只学习不介入；继续预测、影子复盘和自动训练，达标后自动恢复。"
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
                trained_at = self._parse_datetime(
                    metadata.get("trained_at") or metadata.get("version")
                )
                age_seconds = (
                    (now - trained_at).total_seconds()
                    if trained_at is not None
                    else AUTO_TRAIN_MIN_INTERVAL_SECONDS
                )
                new_samples = max(completed_count - last_sample_count, 0)

                should_train = force or (
                    age_seconds >= AUTO_TRAIN_MIN_INTERVAL_SECONDS
                    or new_samples >= AUTO_TRAIN_MIN_NEW_SAMPLES
                )
                if not should_train:
                    result = {
                        "trained": False,
                        "reason": "not_due",
                        "completed_sample_count": completed_count,
                        "last_trained_sample_count": last_sample_count,
                        "new_sample_count": new_samples,
                        "model_age_seconds": round(age_seconds, 1),
                        "message": (
                            "未达到自动训练条件：需要距离上次训练至少 6 小时，"
                            "或新增 completed 影子复盘样本不少于 500 条。"
                        ),
                    }
                    self._last_train_result = result
                    return result

                self._training = True
                self._last_train_started_at = datetime.now(UTC).isoformat()
                rows = await load_shadow_training_rows(limit=20000)
                frame = build_training_frame(rows)
                trained_metadata = await asyncio.to_thread(train_from_frame, frame)
                self._bundle = None
                self._loaded_mtime = None
                self._ensure_loaded()
                result = {
                    "trained": True,
                    "reason": "trained",
                    "completed_sample_count": completed_count,
                    "previous_sample_count": last_sample_count,
                    "new_sample_count": new_samples,
                    "sample_count": int(trained_metadata.get("sample_count") or 0),
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
            return {
                "available": False,
                "status": "no_model",
                "message": "本地 ML 盈亏质量模型尚未训练，当前分析不使用 ML 辅助信号。",
            }
        metadata = _safe_dict(self._bundle.get("metadata"))
        influence = _influence_policy(metadata)

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
                        influence.get("enabled")
                        and side_influence.get("enabled")
                        and best_expected > WIN_RETURN_THRESHOLD_PCT
                        and profit_edge >= MIN_PROFIT_EDGE_PCT
                        and best_win >= MIN_PROFIT_SIGNAL_WIN_RATE
                    ),
                    "risk_score": round(risk_score, 4),
                    "ml_influence_enabled": bool(
                        influence.get("enabled") and side_influence.get("enabled")
                    ),
                }
            )

        primary = predictions[0] if predictions else {}
        return {
            "available": True,
            "status": "entry_profit_filter" if influence.get("enabled") else "learning_only",
            "mode": "entry_profit_filter" if influence.get("enabled") else "learning_only",
            "influence_enabled": bool(influence.get("enabled")),
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
                else "ML 当前处于学习观察中：继续预测、影子复盘和自动训练，但不影响开仓过滤、加分或机会排序。"
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
        async with get_session_ctx() as session:
            repo = MemoryRepository(session)
            return await repo.count_shadow_backtests(status="completed")

    async def completed_shadow_sample_count(self) -> int:
        """Return completed shadow samples through a public dashboard boundary."""

        return await self._completed_shadow_sample_count()

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


async def load_shadow_training_rows(limit: int = 20000) -> list[Any]:
    async with get_session_ctx() as session:
        repo = MemoryRepository(session)
        rows = await repo.list_shadow_backtests(status="completed", limit=limit, offset=0)
        return rows
