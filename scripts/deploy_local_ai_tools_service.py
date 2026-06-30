"""Deploy the Phase 3 quant API to the configured model server."""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
import textwrap
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

PHASE3_ROOT = "/data/BB"
PHASE3_API_PORT = 8101
PHASE3_SERVICE_NAME = "bb-phase3-quant-api.service"
PHASE3_APP_DIR = f"{PHASE3_ROOT}/services/phase3_quant_api"
PHASE3_SYSTEMD_DIR = f"{PHASE3_ROOT}/services/systemd"
PHASE3_LOG_DIR = f"{PHASE3_ROOT}/logs/services"
PHASE3_MODEL_DIR = f"{PHASE3_ROOT}/models/local_ai_tools"
PHASE3_RUNTIME_DIR = f"{PHASE3_ROOT}/runtime/phase3_quant_api"
PHASE3_ENV_FILE = f"{PHASE3_ROOT}/env/phase3.env"
PHASE3_PYTHON_BIN = f"{PHASE3_ROOT}/envs/phase3-quant/bin/python"
PHASE3_POLICY_ID = "phase3_quant_api_shadow_contract_v2_2026_06_27"

SERVICE_CODE = r'''
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


PHASE3_ROOT = Path(os.environ.get("BB_PHASE3_ROOT", "/data/BB"))
PHASE3_API_PORT = int(os.environ.get("PHASE3_QUANT_API_PORT", "8101"))
MODEL_DIR = Path(
    os.environ.get(
        "LOCAL_AI_TOOLS_MODEL_DIR",
        str(PHASE3_ROOT / "models" / "local_ai_tools"),
    )
)
BUNDLE_PATH = MODEL_DIR / "local_quant_models.joblib"
METADATA_PATH = MODEL_DIR / "local_quant_models_metadata.json"
PHASE3_VALIDATION_REPORT_PATH = (
    PHASE3_ROOT / "reports" / "inventory" / "phase3_model_validation_latest.json"
)
PHASE3_DOWNLOAD_REPORT_PATH = (
    PHASE3_ROOT / "reports" / "inventory" / "phase3_model_download_manifest_latest.json"
)
PHASE3_ARTIFACT_POLICY_ID = "phase3_clean_training_artifact_v1"
PHASE3_REQUIRED_TRAINING_POLICY = "clean_training_view_only"
PHASE3_REQUIRED_PROMOTION_FLOW = "shadow_to_canary_to_live"
LOCAL_REVIEW_DISABLED_DETAIL = (
    "Local AI tools do not provide high-risk trade review. "
    "Configure HIGH_RISK_REVIEW_* in the trading app to an online reviewer."
)

FEATURE_KEYS = [
    "change_24h_pct", "spread_pct", "rsi_14", "rsi_7", "macd", "macd_signal",
    "macd_diff", "stoch_k", "adx_14", "bb_width", "bb_pct", "atr_pct",
    "volume_ratio", "returns_1", "returns_5", "returns_20", "volatility_20",
    "price_vs_sma20", "price_vs_sma50", "funding_rate", "log_volume_24h",
    "log_open_interest_value", "orderbook_imbalance", "orderbook_depth_ratio",
    "news_sentiment_avg", "social_sentiment_avg", "social_mention_count",
    "news_article_count", "decision_confidence", "horizon_minutes",
]
SENTIMENT_KEYS = ["news_sentiment_avg", "social_sentiment_avg", "social_mention_count", "news_article_count"]
ROUND_TRIP_COST_PCT = float(os.environ.get("LOCAL_AI_TOOLS_ROUND_TRIP_COST_PCT", "0.12"))
TAIL_LOSS_THRESHOLD_PCT = float(os.environ.get("LOCAL_AI_TOOLS_TAIL_LOSS_THRESHOLD_PCT", "0.18"))
MIN_TIMESERIES_SEQUENCE_LENGTH = int(
    os.environ.get("LOCAL_AI_TOOLS_MIN_TIMESERIES_SEQUENCE_LENGTH", "30")
)
LOCAL_AI_TOOLS_API_KEY = os.environ.get("LOCAL_AI_TOOLS_API_KEY", "").strip()
ERROR_TEXT_LIMIT = 180
SECRET_TEXT_RE = re.compile(
    r"(Authorization\s*:\s*Bearer\s+)[^\s,;\"']+"
    r"|((?:api[_-]?key|secret|password|passphrase|token|webhook)"
    r"\s*[:=]\s*)[^\s,;\"']+",
    re.IGNORECASE,
)
LOCAL_AI_TOOLS_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "LOCAL_AI_TOOLS_CORS_ORIGINS",
        "http://127.0.0.1:8002,http://localhost:8002",
    ).split(",")
    if origin.strip()
]
ALLOW_UNAUTHENTICATED_LOOPBACK = os.environ.get(
    "LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}

_BUNDLE_CACHE: dict[str, Any] | None = None
_BUNDLE_MTIME: float | None = None
_TRANSFORMER_MODEL_CACHE: dict[str, Any] = {}


def safe_error(value: Any, limit: int = ERROR_TEXT_LIMIT) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    def repl(match: re.Match[str]) -> str:
        auth_prefix = match.group(1)
        key_prefix = match.group(2)
        if auth_prefix:
            return auth_prefix + "***"
        if key_prefix:
            return key_prefix + "***"
        return "***"

    redacted = SECRET_TEXT_RE.sub(repl, text)
    if limit and len(redacted) > limit:
        return redacted[:limit] + "..."
    return redacted


def _cache_get_or_load(key: str, loader):
    if key not in _TRANSFORMER_MODEL_CACHE:
        _TRANSFORMER_MODEL_CACHE[key] = loader()
    return _TRANSFORMER_MODEL_CACHE[key]


def _is_loopback_request(request: Request) -> bool:
    client_host = (request.client.host if request.client else "") or ""
    return client_host in {"127.0.0.1", "::1", "localhost"}


def require_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    if LOCAL_AI_TOOLS_API_KEY:
        expected = f"Bearer {LOCAL_AI_TOOLS_API_KEY}"
        if authorization == expected:
            return
        raise HTTPException(status_code=401, detail="Invalid local AI tools API key.")
    if ALLOW_UNAUTHENTICATED_LOOPBACK and _is_loopback_request(request):
        return
    raise HTTPException(
        status_code=401,
        detail="LOCAL_AI_TOOLS_API_KEY is required for non-loopback access.",
    )


app = FastAPI(
    title="Trade Local AI Tools",
    version="1.0.0",
    dependencies=[Depends(require_api_key)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_AI_TOOLS_CORS_ORIGINS,
    allow_credentials=bool(LOCAL_AI_TOOLS_API_KEY),
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeatureRequest(BaseModel):
    symbol: str | None = None
    features: dict[str, Any] = {}
    local_ml_signal: dict[str, Any] | None = None
    open_positions: list[dict[str, Any]] | None = None


class TrainRequest(BaseModel):
    shadow_samples: list[dict[str, Any]] = []
    trade_samples: list[dict[str, Any]] = []
    sequence_samples: list[dict[str, Any]] = []
    text_sentiment_samples: list[dict[str, Any]] = []
    source: str = "local_trading_system"
    completed_shadow_sample_count: int | None = None
    completed_trade_sample_count: int | None = None
    quality_report: dict[str, Any] = {}
    governance_report: dict[str, Any] = {}
    training_mode: str = "shadow"
    model_stage: str = "shadow"
    evaluation_policy: dict[str, Any] = {}
    promotion_recommendation: dict[str, Any] = {}
    persist_artifact: bool = False
    confirm_phase3_rebuild: bool = False


def f(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(features.get(key, default) or default)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def net_return_pct(value: float) -> float:
    return f({"value": value}, "value") - ROUND_TRIP_COST_PCT


def feature_row(features: dict[str, Any], *, horizon_minutes: int | None = None) -> dict[str, float]:
    price = f(features, "current_price", f(features, "close", 0.0))
    atr = f(features, "atr_14")
    bid_depth = f(features, "orderbook_bid_depth")
    ask_depth = f(features, "orderbook_ask_depth")
    total_depth = max(bid_depth + ask_depth, 1e-9)
    volume_24h = max(f(features, "volume_24h"), 0.0)
    oi_value = max(f(features, "open_interest_value"), 0.0)
    values = {
        "change_24h_pct": f(features, "change_24h_pct"),
        "spread_pct": f(features, "spread_pct"),
        "rsi_14": f(features, "rsi_14", 50.0),
        "rsi_7": f(features, "rsi_7", 50.0),
        "macd": f(features, "macd"),
        "macd_signal": f(features, "macd_signal"),
        "macd_diff": f(features, "macd_diff"),
        "stoch_k": f(features, "stoch_k", 50.0),
        "adx_14": f(features, "adx_14"),
        "bb_width": f(features, "bb_width"),
        "bb_pct": f(features, "bb_pct", 0.5),
        "atr_pct": atr / price if price > 0 else 0.0,
        "volume_ratio": f(features, "volume_ratio", 1.0),
        "returns_1": f(features, "returns_1"),
        "returns_5": f(features, "returns_5"),
        "returns_20": f(features, "returns_20"),
        "volatility_20": f(features, "volatility_20"),
        "price_vs_sma20": f(features, "price_vs_sma20"),
        "price_vs_sma50": f(features, "price_vs_sma50"),
        "funding_rate": f(features, "funding_rate"),
        "log_volume_24h": math.log10(volume_24h + 1.0),
        "log_open_interest_value": math.log10(oi_value + 1.0),
        "orderbook_imbalance": f(features, "orderbook_imbalance"),
        "orderbook_depth_ratio": (bid_depth - ask_depth) / total_depth,
        "news_sentiment_avg": f(features, "news_sentiment_avg"),
        "social_sentiment_avg": f(features, "social_sentiment_avg"),
        "social_mention_count": f(features, "social_mention_count"),
        "news_article_count": f(features, "news_article_count"),
        "decision_confidence": f(features, "decision_confidence"),
        "horizon_minutes": float(horizon_minutes if horizon_minutes is not None else f(features, "horizon_minutes", 10.0)),
    }
    return {key: float(values.get(key, 0.0)) for key in FEATURE_KEYS}


def model_x(features: dict[str, Any], *, horizon_minutes: int | None = None) -> list[float]:
    row = feature_row(features, horizon_minutes=horizon_minutes)
    return [row[key] for key in FEATURE_KEYS]


def _make_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _make_classifier(y: list[int]) -> Pipeline:
    unique = set(int(v) for v in y)
    if len(unique) < 2:
        from sklearn.dummy import DummyClassifier
        estimator = DummyClassifier(strategy="prior")
    else:
        estimator = ExtraTreesClassifier(
            n_estimators=240,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


def _trusted_model_artifact_path(path: Path) -> Path:
    root = MODEL_DIR.resolve(strict=False)
    if not root.is_absolute():
        raise ValueError("Model directory must be absolute.")
    target = Path(path).resolve(strict=False)
    if target.suffix != ".joblib":
        raise ValueError("Model artifact must use .joblib suffix.")
    if not target.is_relative_to(root):
        raise ValueError("Model artifact path escapes trusted model directory.")
    return target


def load_trusted_joblib_bundle(path: Path) -> dict[str, Any]:
    target = _trusted_model_artifact_path(path)
    value = joblib.load(target)
    if not isinstance(value, dict):
        raise ValueError("Model artifact must contain a dictionary bundle.")
    return value


def dump_trusted_joblib_bundle(bundle: dict[str, Any], path: Path) -> Path:
    if not isinstance(bundle, dict):
        raise ValueError("Model artifact must be a dictionary bundle.")
    target = _trusted_model_artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}.",
            suffix=".tmp",
            dir=str(target.parent),
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        joblib.dump(bundle, tmp_path)
        os.replace(tmp_path, target)
        return target
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def load_bundle() -> dict[str, Any] | None:
    global _BUNDLE_CACHE, _BUNDLE_MTIME
    try:
        if not BUNDLE_PATH.exists():
            return None
        mtime = BUNDLE_PATH.stat().st_mtime
        if _BUNDLE_CACHE is not None and _BUNDLE_MTIME == mtime:
            return _BUNDLE_CACHE
        _BUNDLE_CACHE = load_trusted_joblib_bundle(BUNDLE_PATH)
        _BUNDLE_MTIME = mtime
        return _BUNDLE_CACHE
    except Exception:
        _BUNDLE_CACHE = None
        _BUNDLE_MTIME = None
        return None


def predict_proba_positive(model: Pipeline, x: list[list[float]]) -> float:
    try:
        estimator = model.named_steps["model"]
        proba = model.predict_proba(x)
        classes = list(getattr(estimator, "classes_", []))
        if 1 in classes:
            return float(proba[0][classes.index(1)])
        return 0.0
    except Exception:
        return 0.0


def symbol_key(symbol: str | None) -> str:
    value = str(symbol or "").upper().split(":")[0]
    if value.endswith("-SWAP"):
        value = value[:-5]
    if "/" not in value and "-" in value:
        parts = value.split("-")
        if len(parts) >= 2:
            value = f"{parts[0]}/{parts[1]}"
    return value


def _train_profiles(trade_samples: list[dict[str, Any]]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for row in trade_samples:
        if bool(row.get("exclude_from_training")):
            continue
        symbol = symbol_key(row.get("symbol"))
        side = str(row.get("side") or "").lower()
        if not symbol or side not in {"long", "short"}:
            continue
        key = f"{symbol}|{side}"
        bucket = profile.setdefault(key, {
            "symbol": symbol,
            "side": side,
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
            "largest_profit": 0.0,
            "largest_loss": 0.0,
            "avg_hold_minutes": 0.0,
        })
        pnl = f(row, "realized_pnl")
        hold = f(row, "hold_minutes")
        bucket["count"] += 1
        bucket["pnl"] += pnl
        bucket["avg_hold_minutes"] += hold
        if pnl >= 0:
            bucket["wins"] += 1
            bucket["profit"] += pnl
            bucket["largest_profit"] = max(bucket["largest_profit"], pnl)
        else:
            bucket["losses"] += 1
            bucket["loss"] += abs(pnl)
            bucket["largest_loss"] = max(bucket["largest_loss"], abs(pnl))
    for bucket in profile.values():
        count = max(int(bucket.get("count") or 0), 1)
        bucket["avg_pnl"] = bucket["pnl"] / count
        bucket["avg_hold_minutes"] = bucket["avg_hold_minutes"] / count
        bucket["win_rate"] = bucket["wins"] / count
        bucket["profit_factor"] = bucket["profit"] / max(bucket["loss"], 1e-9)
        bucket["avg_profit"] = bucket["profit"] / max(bucket["wins"], 1)
        bucket["avg_loss"] = bucket["loss"] / max(bucket["losses"], 1)
        bucket["payoff_ratio"] = bucket["avg_profit"] / max(bucket["avg_loss"], 1e-9)
        bucket["small_win_big_loss_risk"] = clamp(
            (bucket["avg_loss"] - bucket["avg_profit"]) / max(bucket["avg_loss"] + bucket["avg_profit"], 1e-9),
            0.0,
            1.0,
        )
        bucket["loss_pressure"] = clamp((bucket["loss"] - bucket["profit"]) / max(bucket["loss"] + bucket["profit"], 1e-9), 0.0, 1.0)
    return profile


def side_scores(features: dict[str, Any]) -> tuple[float, float]:
    returns_1 = f(features, "returns_1")
    returns_5 = f(features, "returns_5")
    returns_20 = f(features, "returns_20")
    macd_diff = f(features, "macd_diff")
    price_vs_sma20 = f(features, "price_vs_sma20")
    price_vs_sma50 = f(features, "price_vs_sma50")
    rsi = f(features, "rsi_14", 50.0)
    stoch = f(features, "stoch_k", 50.0)
    orderbook = f(features, "orderbook_imbalance")
    funding = f(features, "funding_rate")
    volume_ratio = f(features, "volume_ratio", 1.0)
    adx = f(features, "adx_14")

    momentum = returns_1 * 0.22 + returns_5 * 0.34 + returns_20 * 0.44
    trend = price_vs_sma20 * 0.35 + price_vs_sma50 * 0.35 + macd_diff * 25.0
    oscillator_long = clamp((rsi - 45.0) / 25.0, -1.0, 1.0) + clamp((stoch - 50.0) / 35.0, -1.0, 1.0)
    oscillator_short = -oscillator_long
    participation = clamp(volume_ratio / 1.5, 0.0, 2.0) * clamp(adx / 25.0, 0.0, 2.0)
    flow = orderbook * 0.45 - funding * 8.0

    long_score = momentum * 70.0 + trend * 26.0 + oscillator_long * 0.11 + flow + participation * 0.08
    short_score = -momentum * 70.0 - trend * 26.0 + oscillator_short * 0.11 - flow + participation * 0.08
    return long_score, short_score


def _safe_sequence(values: Any, limit: int = 80) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for item in values[-limit:]:
        try:
            value = float(item)
            if math.isfinite(value):
                out.append(value)
        except Exception:
            continue
    return out


def sequence_features(close_sequence: Any, volume_sequence: Any | None = None) -> list[float]:
    closes = _safe_sequence(close_sequence)
    volumes = _safe_sequence(volume_sequence or [])
    if len(closes) < 4:
        closes = [0.0, 0.0, 0.0, 0.0]
    last = closes[-1] if abs(closes[-1]) > 1e-9 else 1.0
    returns = []
    for window in (1, 3, 5, 10, 20, 40):
        if len(closes) > window and abs(closes[-window - 1]) > 1e-9:
            returns.append((closes[-1] - closes[-window - 1]) / closes[-window - 1] * 100.0)
        else:
            returns.append(0.0)
    diffs = np.diff(np.array(closes[-40:], dtype=float))
    volatility = float(np.std(diffs / max(abs(last), 1e-9)) * 100.0) if len(diffs) else 0.0
    drawdown = (min(closes[-40:]) - max(closes[-40:])) / max(abs(last), 1e-9) * 100.0 if closes else 0.0
    vol_ratio = 1.0
    if len(volumes) >= 10:
        recent = float(np.mean(volumes[-5:]))
        base = float(np.mean(volumes[-30:])) if len(volumes) >= 30 else float(np.mean(volumes))
        vol_ratio = recent / max(base, 1e-9)
    return returns + [volatility, drawdown, vol_ratio, float(len(closes))]


def sequence_deep_features(close_sequence: Any, volume_sequence: Any | None = None, length: int = 60) -> list[float]:
    closes = _safe_sequence(close_sequence, limit=length)
    volumes = _safe_sequence(volume_sequence or [], limit=length)
    if len(closes) < 2:
        closes = [0.0, 0.0]
    last = closes[-1] if abs(closes[-1]) > 1e-9 else 1.0
    returns = [0.0]
    for prev, cur in zip(closes[:-1], closes[1:]):
        base = prev if abs(prev) > 1e-9 else last
        returns.append((cur - base) / base)
    if len(returns) < length:
        returns = [0.0] * (length - len(returns)) + returns
    else:
        returns = returns[-length:]
    if volumes:
        vol_base = float(np.mean(volumes)) if volumes else 1.0
        vol_base = vol_base if abs(vol_base) > 1e-9 else 1.0
        vol_values = [(v / vol_base) - 1.0 for v in volumes]
    else:
        vol_values = []
    if len(vol_values) < length:
        vol_values = [0.0] * (length - len(vol_values)) + vol_values
    else:
        vol_values = vol_values[-length:]
    patch_stats: list[float] = []
    patch_size = 10
    for start in range(0, length, patch_size):
        patch = np.array(returns[start:start + patch_size], dtype=float)
        patch_stats.extend([
            float(np.mean(patch)),
            float(np.std(patch)),
            float(patch[-1] - patch[0]) if len(patch) else 0.0,
        ])
    return [float(x) for x in returns + vol_values + patch_stats]


def _train_sequence_model(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = []
    for sample in samples or []:
        if bool(sample.get("exclude_from_training")):
            continue
        x = sequence_features(sample.get("close_sequence"), sample.get("volume_sequence"))
        y = f(sample, "future_return_pct")
        if not x:
            continue
        rows.append((x, y, sample.get("timeframe") or "unknown"))
    if len(rows) < 120:
        return None
    model = _make_regressor()
    model.fit([x for x, _, _ in rows], [y for _, y, _ in rows])
    timeframes: dict[str, int] = {}
    for _, _, timeframe in rows:
        timeframes[str(timeframe)] = timeframes.get(str(timeframe), 0) + 1
    return {"model": model, "samples": len(rows), "timeframes": timeframes}


def _train_torch_patch_model(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        import torch
        from torch import nn
    except Exception as exc:
        return {"available": False, "reason": f"torch_unavailable: {safe_error(exc, 120)}"}

    rows = []
    for sample in samples or []:
        if bool(sample.get("exclude_from_training")):
            continue
        x = sequence_deep_features(sample.get("close_sequence"), sample.get("volume_sequence"))
        y = f(sample, "future_return_pct")
        if x:
            rows.append((x, y))
    if len(rows) < 180:
        return {"available": False, "reason": "not_enough_sequence_samples", "samples": len(rows)}

    X = np.array([x for x, _ in rows], dtype=np.float32)
    y = np.array([[target] for _, target in rows], dtype=np.float32)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std

    torch.set_num_threads(max(min(os.cpu_count() or 2, 8), 1))
    xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    model = nn.Sequential(
        nn.Linear(X.shape[1], 96),
        nn.GELU(),
        nn.Dropout(0.05),
        nn.Linear(96, 48),
        nn.GELU(),
        nn.Linear(48, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    epochs = 120 if len(rows) < 1000 else 80
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_mae = float(torch.mean(torch.abs(model(xt) - yt)).item())
    return {
        "available": True,
        "backend": "torch_patch_mlp_cpu",
        "samples": len(rows),
        "input_dim": int(X.shape[1]),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "mean": mean.astype(float).tolist()[0],
        "std": std.astype(float).tolist()[0],
        "train_mae_pct": round(train_mae, 5),
    }


def _predict_torch_patch_model(model_info: dict[str, Any], close_sequence: Any, volume_sequence: Any | None = None) -> float | None:
    if not model_info or model_info.get("available") is not True:
        return None
    try:
        import torch
        from torch import nn
        x = np.array([sequence_deep_features(close_sequence, volume_sequence)], dtype=np.float32)
        mean = np.array(model_info.get("mean") or [], dtype=np.float32).reshape(1, -1)
        std = np.array(model_info.get("std") or [], dtype=np.float32).reshape(1, -1)
        if mean.shape != x.shape or std.shape != x.shape:
            return None
        x = (x - mean) / (std + 1e-6)
        net = nn.Sequential(
            nn.Linear(x.shape[1], 96),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        net.load_state_dict(model_info["state_dict"])
        net.eval()
        with torch.no_grad():
            return float(net(torch.tensor(x, dtype=torch.float32))[0, 0].item())
    except Exception:
        return None


def _timeseries_close_sequence(features: dict[str, Any]) -> tuple[list[float], str, str]:
    source = ""
    raw: Any = []
    for key in ("close_sequence", "recent_closes", "closes"):
        candidate = features.get(key)
        if candidate:
            raw = candidate
            source = key
            break
    closes = _safe_sequence(raw, limit=512)
    if len(closes) < MIN_TIMESERIES_SEQUENCE_LENGTH:
        return closes, "not_enough_real_close_sequence", source or "missing"
    return closes, "", source


def _load_timesfm_model(model_dir: str):
    def loader():
        from transformers import AutoModelForTimeSeriesPrediction

        model = AutoModelForTimeSeriesPrediction.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        model.eval()
        return model

    return _cache_get_or_load(f"timesfm::{model_dir}", loader)


def _load_chronos2_pipeline(model_dir: str):
    def loader():
        from chronos import Chronos2Pipeline

        return Chronos2Pipeline.from_pretrained(model_dir)

    return _cache_get_or_load(f"chronos2::{model_dir}", loader)


def _prediction_values(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "float"):
            value = value.float()
        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        pass
    if isinstance(value, (int, float)):
        number = float(value)
        return [number] if math.isfinite(number) else []
    if not isinstance(value, (list, tuple)):
        return []
    rows = list(value)
    while rows and isinstance(rows[0], (list, tuple)):
        if rows and all(isinstance(item, (int, float)) for item in rows):
            break
        rows = list(rows[0])
    out = []
    for item in rows:
        try:
            number = float(item)
            if math.isfinite(number):
                out.append(number)
        except Exception:
            continue
    return out


def _extract_timesfm_mean_predictions(output: Any) -> list[float]:
    candidates = []
    if isinstance(output, dict):
        candidates.extend([
            output.get("mean_predictions"),
            output.get("prediction_outputs"),
            output.get("predictions"),
            output.get("full_predictions"),
        ])
    else:
        candidates.extend([
            getattr(output, "mean_predictions", None),
            getattr(output, "prediction_outputs", None),
            getattr(output, "predictions", None),
            getattr(output, "full_predictions", None),
        ])
    for candidate in candidates:
        values = _prediction_values(candidate)
        if values:
            return values
    return []


def _chronos_prediction_values(value: Any) -> list[float]:
    if value is None:
        return []
    tensor_values = _chronos_tensor_prediction_values(value)
    if tensor_values:
        return tensor_values
    if isinstance(value, dict):
        for key in (
            "median",
            "mean",
            "prediction",
            "predictions",
            "forecast",
            "forecast_values",
        ):
            values = _prediction_values(value.get(key))
            if values:
                return values
    if isinstance(value, list):
        if value and all(isinstance(item, dict) for item in value):
            for key in (
                "median",
                "mean",
                "prediction",
                "predictions",
                "forecast",
                "forecast_values",
                "target",
            ):
                collected = []
                for item in value:
                    values = _prediction_values(item.get(key))
                    if values:
                        collected.extend(values)
                if collected:
                    return collected
        for item in value:
            values = _chronos_prediction_values(item)
            if values:
                return values
    try:
        if hasattr(value, "to_dict"):
            records = value.to_dict("records")
            values = _chronos_prediction_values(records)
            if values:
                return values
        columns = list(getattr(value, "columns", []) or [])
        for name in ("median", "mean", "prediction", "forecast", "target"):
            if name in columns:
                values = _prediction_values(value[name])
                if values:
                    return values
    except Exception:
        pass
    return _prediction_values(value)


def _chronos_tensor_prediction_values(value: Any) -> list[float]:
    """Extract the median forecast path from Chronos tensor-style outputs."""
    try:
        item = value
        if hasattr(item, "detach"):
            item = item.detach()
        if hasattr(item, "cpu"):
            item = item.cpu()
        if hasattr(item, "float"):
            item = item.float()
        if hasattr(item, "numpy"):
            array = item.numpy()
        else:
            return []
        arr = np.asarray(array, dtype=float)
        if arr.ndim >= 3:
            # Chronos direct predict returns (n_variates, n_quantiles, horizon).
            arr = arr[0, arr.shape[1] // 2, :]
        elif arr.ndim == 2:
            arr = arr[arr.shape[0] // 2, :] if arr.shape[0] > 1 else arr[0, :]
        elif arr.ndim != 1:
            return []
        return [float(item) for item in arr.ravel().tolist() if math.isfinite(float(item))]
    except Exception:
        return []


def _run_chronos2_shadow(features: dict[str, Any]) -> dict[str, Any]:
    chain = _specialist_model_chain("timeseries")
    closes, reason, sequence_source = _timeseries_close_sequence(features)
    if reason:
        return {
            "available": False,
            "kind": "timeseries",
            "model": "chronos-2-shadow-primary",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": False,
            "reason": reason,
            "sequence_length": len(closes),
            "sequence_source": sequence_source,
            "minimum_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        }
    try:
        import pandas as pd

        model_dir = PHASE3_ROOT / "models" / "timeseries" / "amazon--chronos-2"
        pipeline = _load_chronos2_pipeline(model_dir.as_posix())
        horizon_step = int(
            max(
                1,
                f(
                    features,
                    "horizon_steps",
                    f(features, "forecast_horizon_steps", f(features, "horizon_minutes", 1.0)),
                ),
            )
        )
        end_timestamp = pd.Timestamp.utcnow()
        try:
            end_timestamp = end_timestamp.tz_localize(None)
        except (AttributeError, TypeError):
            pass
        history = pd.DataFrame(
            {
                "id": [str(features.get("symbol") or "series")] * len(closes),
                "timestamp": pd.date_range(
                    end=end_timestamp,
                    periods=len(closes),
                    freq=str(features.get("chronos_freq") or "min"),
                ),
                "target": np.asarray(closes, dtype=np.float64),
            }
        )
        try:
            forecast = pipeline.predict_df(
                history,
                prediction_length=max(horizon_step, 1),
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
                validate_inputs=False,
                freq=str(features.get("chronos_freq") or "min"),
            )
            predictions = _chronos_prediction_values(forecast)
        except Exception:
            forecast = pipeline.predict(
                [np.asarray(closes, dtype=np.float32)],
                prediction_length=max(horizon_step, 1),
                limit_prediction_length=False,
            )
            predictions = _chronos_prediction_values(forecast)
        if not predictions:
            return {
                "available": False,
                "kind": "timeseries",
                "model": "chronos-2-shadow-primary",
                "primary_model": chain.get("primary_model"),
                "challenger_model": chain.get("challenger_model"),
                "artifacts_ready": bool(chain.get("artifacts_ready")),
                "actual_inference": False,
                "reason": "chronos_empty_prediction",
                "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
                "live_mutation": False,
            }
        horizon_index = min(horizon_step, len(predictions)) - 1
        last_close = closes[-1]
        forecast_price = float(predictions[horizon_index])
        expected_move_pct = (
            (forecast_price - last_close) / max(abs(last_close), 1e-9) * 100.0
            if math.isfinite(forecast_price)
            else 0.0
        )
        recent = np.array(closes[-80:], dtype=float)
        diff = np.diff(recent)
        realized_vol_pct = (
            float(np.std(diff / max(abs(last_close), 1e-9)) * 100.0)
            if len(diff)
            else 0.0
        )
        confidence = clamp(abs(expected_move_pct) / max(realized_vol_pct * 2.5, 0.35), 0.0, 1.0)
        direction = "up" if expected_move_pct > 0 else "down" if expected_move_pct < 0 else "flat"
        return {
            "available": True,
            "kind": "timeseries",
            "model": "chronos-2-shadow-primary",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": True,
            "sequence_length": len(closes),
            "sequence_source": sequence_source,
            "minimum_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
            "horizon_step": horizon_step,
            "forecast_price": round(forecast_price, 8),
            "last_close": round(float(last_close), 8),
            "expected_move_pct": round(expected_move_pct, 6),
            "expected_return_pct": round(expected_move_pct, 6),
            "direction": direction,
            "best_side": "long" if direction == "up" else "short" if direction == "down" else "hold",
            "confidence": round(confidence, 6),
            "realized_vol_pct": round(realized_vol_pct, 6),
            "prediction_count": len(predictions),
            "adapter": "chronos_2_pipeline_adapter",
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        }
    except Exception as exc:
        return {
            "available": False,
            "kind": "timeseries",
            "model": "chronos-2-shadow-primary",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": False,
            "reason": safe_error(exc, 220),
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        }


def _run_timesfm_shadow(features: dict[str, Any]) -> dict[str, Any]:
    chain = _specialist_model_chain("timeseries")
    closes, reason, sequence_source = _timeseries_close_sequence(features)
    if reason:
        return {
            "available": False,
            "kind": "timeseries",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": False,
            "reason": reason,
            "sequence_length": len(closes),
            "sequence_source": sequence_source,
            "minimum_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        }
    try:
        import torch

        model_dir = (
            PHASE3_ROOT
            / "models"
            / "timeseries"
            / "google--timesfm-2.5-200m-transformers"
        )
        model = _load_timesfm_model(model_dir.as_posix())
        series = torch.tensor(closes, dtype=torch.float32)
        output = None
        errors = []
        with torch.no_grad():
            for past_values in ([series], getattr(series, "reshape", lambda *_: series)(1, -1)):
                try:
                    output = model(past_values=past_values)
                    break
                except Exception as exc:
                    errors.append(safe_error(exc, 120))
        predictions = _extract_timesfm_mean_predictions(output)
        if not predictions:
            return {
                "available": False,
                "kind": "timeseries",
                "primary_model": chain.get("primary_model"),
                "challenger_model": chain.get("challenger_model"),
                "artifacts_ready": bool(chain.get("artifacts_ready")),
                "actual_inference": False,
                "reason": "timesfm_empty_prediction" if not errors else "; ".join(errors[-2:]),
                "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
                "live_mutation": False,
            }
        horizon_step = int(
            max(
                1,
                f(
                    features,
                    "horizon_steps",
                    f(features, "forecast_horizon_steps", f(features, "horizon_minutes", 1.0)),
                ),
            )
        )
        horizon_index = min(horizon_step, len(predictions)) - 1
        last_close = closes[-1]
        forecast_price = float(predictions[horizon_index])
        expected_move_pct = (
            (forecast_price - last_close) / max(abs(last_close), 1e-9) * 100.0
            if math.isfinite(forecast_price)
            else 0.0
        )
        recent = np.array(closes[-80:], dtype=float)
        diff = np.diff(recent)
        realized_vol_pct = (
            float(np.std(diff / max(abs(last_close), 1e-9)) * 100.0)
            if len(diff)
            else 0.0
        )
        confidence = clamp(abs(expected_move_pct) / max(realized_vol_pct * 2.5, 0.35), 0.0, 1.0)
        direction = "up" if expected_move_pct > 0 else "down" if expected_move_pct < 0 else "flat"
        return {
            "available": True,
            "kind": "timeseries",
            "model": "timesfm-2.5-shadow-challenger",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": True,
            "sequence_length": len(closes),
            "sequence_source": sequence_source,
            "minimum_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
            "horizon_step": horizon_step,
            "forecast_price": round(forecast_price, 8),
            "last_close": round(float(last_close), 8),
            "expected_move_pct": round(expected_move_pct, 6),
            "expected_return_pct": round(expected_move_pct, 6),
            "direction": direction,
            "best_side": "long" if direction == "up" else "short" if direction == "down" else "hold",
            "confidence": round(confidence, 6),
            "realized_vol_pct": round(realized_vol_pct, 6),
            "prediction_count": len(predictions),
            "adapter": "timesfm_transformers_adapter",
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        }
    except Exception as exc:
        return {
            "available": False,
            "kind": "timeseries",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": False,
            "reason": safe_error(exc, 220),
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        }


def _attach_timeseries_specialist_shadow(
    payload: dict[str, Any],
    *,
    features: dict[str, Any],
) -> dict[str, Any]:
    chain = _specialist_model_chain("timeseries")
    primary_shadow = _run_chronos2_shadow(features)
    challenger_shadow = _run_timesfm_shadow(features)
    active = bool(primary_shadow.get("available") or challenger_shadow.get("available"))
    specialist_shadow = primary_shadow if primary_shadow.get("available") else challenger_shadow
    chain = dict(chain)
    chain["actual_inference"] = active
    payload["specialist_primary_model"] = chain.get("primary_model")
    payload["specialist_challenger_model"] = chain.get("challenger_model")
    payload["specialist_artifacts_ready"] = bool(chain.get("artifacts_ready"))
    payload["specialist_inference_active"] = active
    payload["specialist_model_chain"] = chain
    payload["chronos_shadow_expected_move_pct"] = primary_shadow.get("expected_move_pct")
    payload["chronos_shadow_expected_return_pct"] = primary_shadow.get("expected_return_pct")
    payload["chronos_shadow_side"] = primary_shadow.get("best_side")
    payload["chronos_shadow_confidence"] = primary_shadow.get("confidence")
    payload["chronos_shadow_horizon_step"] = primary_shadow.get("horizon_step")
    payload["timesfm_shadow_expected_move_pct"] = challenger_shadow.get("expected_move_pct")
    payload["timesfm_shadow_expected_return_pct"] = challenger_shadow.get("expected_return_pct")
    payload["timesfm_shadow_side"] = challenger_shadow.get("best_side")
    payload["timesfm_shadow_confidence"] = challenger_shadow.get("confidence")
    payload["timesfm_shadow_horizon_step"] = challenger_shadow.get("horizon_step")
    payload["professional_model_shadow"] = {
        "kind": "timeseries",
        "primary_model": chain.get("primary_model"),
        "challenger_model": chain.get("challenger_model"),
        "artifacts_ready": bool(chain.get("artifacts_ready")),
        "actual_inference": active,
        "baseline_model": payload.get("model"),
        "baseline_response": True,
        "activation_blocker": "walk_forward_required",
        "shadow_result": specialist_shadow,
        "primary_shadow_result": primary_shadow,
        "challenger_shadow_result": challenger_shadow,
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
    }
    payload["fallback_reason"] = (
        "specialist_timeseries_shadow_only"
        if active
        else "specialist_timeseries_adapter_not_promoted"
    )
    payload["note"] = (
        "TimesFM specialist inference is shadow-only and cannot mutate live routing."
        if active
        else payload.get("note")
        or "Timeseries specialist adapters remain blocked until preflight and walk-forward pass."
    )
    payload.pop("shadow_payload", None)
    return with_model_metadata(
        "time_series_prediction",
        payload,
        features=features,
        challenger_model=str(chain.get("challenger_model") or ""),
        fallback_reason=payload.get("fallback_reason") or "",
    )


def _text_value(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    platform = str(row.get("platform") or "")
    symbols = " ".join(str(s) for s in (row.get("symbols") or [])[:8])
    return " ".join(part for part in (platform, symbols, text) if part).strip()


def _train_text_sentiment_model(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [
        (_text_value(sample), f(sample, "sentiment_score"))
        for sample in samples or []
        if not bool(sample.get("exclude_from_training"))
    ]
    rows = [(text, score) for text, score in rows if text]
    if len(rows) < 80:
        return None
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=1)),
        ("model", Ridge(alpha=1.2)),
    ])
    model.fit([text for text, _ in rows], [score for _, score in rows])
    return {"model": model, "samples": len(rows)}


def _probe_transformers_sentiment_backend() -> dict[str, Any]:
    try:
        import transformers
        return {
            "available": True,
            "library": "transformers",
            "version": getattr(transformers, "__version__", "unknown"),
            "preferred_models": ["ProsusAI/finbert", "ElKulako/cryptobert"],
            "mode": "optional_runtime_backend",
        }
    except Exception as exc:
        return {"available": False, "reason": f"transformers_unavailable: {safe_error(exc, 120)}"}


def _public_torch_patch_status(model_info: dict[str, Any] | None) -> dict[str, Any]:
    info = model_info or {}
    return {
        "available": bool(info.get("available")),
        "backend": info.get("backend"),
        "samples": int(info.get("samples") or 0),
        "input_dim": int(info.get("input_dim") or 0),
        "train_mae_pct": info.get("train_mae_pct"),
        "reason": info.get("reason"),
    }


def _feature_coverage(features: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(features, dict):
        return {"ratio": 0.0, "present": 0, "total": len(FEATURE_KEYS), "status": "missing"}
    present = 0
    for key in FEATURE_KEYS:
        value = features.get(key)
        if value is not None and str(value).strip() != "":
            present += 1
    total = max(len(FEATURE_KEYS), 1)
    return {
        "ratio": round(present / total, 6),
        "present": present,
        "total": total,
        "status": "reported",
    }


def _shadow_payload(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "available",
        "trained",
        "primary_model",
        "challenger_model",
        "model_version",
        "route_mode",
        "fallback_reason",
        "best_side",
        "side",
        "action",
        "expected_return_pct",
        "adjusted_expected_return_pct",
        "loss_probability",
        "profit_quality_score",
        "expected_move_pct",
        "confidence",
        "urgency",
        "feature_coverage",
        "specialist_primary_model",
        "specialist_challenger_model",
        "specialist_artifacts_ready",
        "specialist_inference_active",
        "specialist_model_chain",
        "professional_model_shadow",
    ]
    shadow = {
        "tool": tool,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
    }
    for key in keys:
        if key in payload:
            shadow[key] = payload.get(key)
    return shadow


def with_model_metadata(
    tool: str,
    payload: dict[str, Any],
    *,
    features: dict[str, Any] | None = None,
    challenger_model: str | None = None,
    fallback_reason: str = "",
) -> dict[str, Any]:
    defaults = {
        "profit_prediction": "profit_v1_baseline",
        "time_series_prediction": "timeseries_v1_baseline",
        "sentiment_analysis": "sentiment_v1_baseline",
        "exit_advice": "exit_v1_rules",
    }
    model_name = str(payload.get("model") or defaults.get(tool) or "local_ai_tools")
    payload.setdefault("primary_model", model_name)
    payload.setdefault("challenger_model", challenger_model)
    payload.setdefault("model_version", f"{model_name}.v1")
    payload.setdefault(
        "route_mode",
        "shadow_candidate" if bool(payload.get("trained")) else "shadow_observation",
    )
    payload.setdefault("fallback_reason", fallback_reason)
    payload.setdefault("feature_coverage", _feature_coverage(features or {}))
    payload.setdefault("promotion_flow", PHASE3_REQUIRED_PROMOTION_FLOW)
    payload.setdefault("live_mutation", False)
    if not isinstance(payload.get("shadow_payload"), dict):
        payload["shadow_payload"] = _shadow_payload(tool, payload)
    return payload


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _phase3_inventory_status() -> dict[str, Any]:
    validation = _read_json_file(PHASE3_VALIDATION_REPORT_PATH)
    download = _read_json_file(PHASE3_DOWNLOAD_REPORT_PATH)
    rows = validation.get("models") if isinstance(validation.get("models"), list) else []
    model_status = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_status.append(
            {
                "slot": row.get("slot") or row.get("role") or "",
                "repo_id": row.get("repo_id") or row.get("model") or "",
                "status": row.get("status") or ("ok" if row.get("required_any_ok") else "unknown"),
                "path": row.get("path") or row.get("target") or "",
            }
        )
    ok_count = sum(1 for row in model_status if row.get("status") == "ok")
    downloaded_rows = download.get("models") if isinstance(download.get("models"), list) else []
    downloaded_count = len(downloaded_rows) if downloaded_rows else ok_count
    validation_all_ok = bool(model_status) and ok_count == len(model_status)
    if validation.get("all_ok") is not None:
        validation_all_ok = bool(validation.get("all_ok"))
    return {
        "downloaded_model_count": downloaded_count,
        "validated_model_count": ok_count,
        "validation_all_ok": validation_all_ok,
        "imports_ok": bool(validation.get("imports_ok", validation_all_ok)),
        "torch_cuda_visible": bool(validation.get("torch_cuda_visible", True)),
        "model_status": model_status,
        "download_manifest_path": PHASE3_DOWNLOAD_REPORT_PATH.as_posix(),
        "validation_report_path": PHASE3_VALIDATION_REPORT_PATH.as_posix(),
    }


SPECIALIST_MODEL_CHAINS = {
    "timeseries": [
        {
            "slot": "timeseries_primary",
            "role": "primary",
            "repo_id": "amazon/chronos-2",
            "purpose": "online_primary_time_series_forecast",
        },
        {
            "slot": "timeseries_challenger",
            "role": "challenger",
            "repo_id": "google/timesfm-2.5-200m-transformers",
            "purpose": "shadow_challenger_time_series_forecast",
        },
        {
            "slot": "timeseries_fallback",
            "role": "fallback",
            "repo_id": "ibm-granite/granite-timeseries-ttm-r2",
            "purpose": "fallback_time_series_regime_check",
        },
    ],
    "sentiment": [
        {
            "slot": "sentiment_primary",
            "role": "primary",
            "repo_id": "ProsusAI/finbert",
            "purpose": "finance_sentiment_primary",
        },
        {
            "slot": "sentiment_challenger",
            "role": "challenger",
            "repo_id": "yiyanghkust/finbert-tone",
            "purpose": "finance_sentiment_challenger",
        },
    ],
}

SPECIALIST_ADAPTER_REQUIREMENTS = {
    "timeseries_primary": {
        "adapter": "chronos_2_transformers_adapter",
        "required_imports": ["torch", "transformers"],
        "optional_imports": ["chronos"],
        "requires_walk_forward": True,
    },
    "timeseries_challenger": {
        "adapter": "timesfm_transformers_adapter",
        "required_imports": ["torch", "transformers"],
        "optional_imports": ["timesfm"],
        "requires_walk_forward": True,
    },
    "timeseries_fallback": {
        "adapter": "granite_ttm_transformers_adapter",
        "required_imports": ["torch", "transformers"],
        "optional_imports": [],
        "requires_walk_forward": True,
    },
    "sentiment_primary": {
        "adapter": "finbert_transformers_adapter",
        "required_imports": ["torch", "transformers"],
        "optional_imports": [],
        "requires_walk_forward": True,
    },
    "sentiment_challenger": {
        "adapter": "finbert_tone_transformers_adapter",
        "required_imports": ["torch", "transformers"],
        "optional_imports": [],
        "requires_walk_forward": True,
    },
}
IMPLEMENTED_SPECIALIST_ADAPTERS = {
    "timeseries_primary",
    "timeseries_challenger",
    "sentiment_primary",
    "sentiment_challenger",
}


def _import_state(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
        return {
            "module": module_name,
            "available": True,
            "version": str(getattr(module, "__version__", "")),
        }
    except Exception as exc:
        return {
            "module": module_name,
            "available": False,
            "error": safe_error(exc, 160),
        }


def _specialist_adapter_preflight(kind: str | None = None) -> dict[str, Any]:
    chain_names = [kind] if kind in SPECIALIST_MODEL_CHAINS else sorted(SPECIALIST_MODEL_CHAINS)
    chains = {name: _specialist_model_chain(name) for name in chain_names}
    rows = []
    blocked_reasons: set[str] = set()

    for chain_name, chain in chains.items():
        for model in chain.get("models", []):
            if not isinstance(model, dict):
                continue
            slot = str(model.get("slot") or "")
            req = SPECIALIST_ADAPTER_REQUIREMENTS.get(slot, {})
            required_imports = [
                _import_state(name) for name in req.get("required_imports", [])
            ]
            optional_imports = [
                _import_state(name) for name in req.get("optional_imports", [])
            ]
            required_imports_ready = all(item.get("available") for item in required_imports)
            artifact_ready = bool(model.get("artifact_ready"))
            adapter_code_ready = slot in IMPLEMENTED_SPECIALIST_ADAPTERS
            row_blockers = []
            if not artifact_ready:
                row_blockers.append("specialist_artifact_not_ready")
            if not required_imports_ready:
                row_blockers.append("specialist_required_import_missing")
            if not adapter_code_ready:
                row_blockers.append("specialist_adapter_not_implemented")
            if bool(req.get("requires_walk_forward", True)):
                row_blockers.append("walk_forward_required")
            blocked_reasons.update(row_blockers)
            rows.append(
                {
                    "kind": chain_name,
                    "slot": slot,
                    "repo_id": model.get("repo_id"),
                    "role": model.get("role"),
                    "adapter": req.get("adapter", ""),
                    "artifact_ready": artifact_ready,
                    "required_imports": required_imports,
                    "optional_imports": optional_imports,
                    "required_imports_ready": required_imports_ready,
                    "adapter_code_ready": adapter_code_ready,
                    "shadow_inference_ready": (
                        artifact_ready and required_imports_ready and adapter_code_ready
                    ),
                    "requires_walk_forward": bool(req.get("requires_walk_forward", True)),
                    "blocked_reasons": row_blockers,
                }
            )

    return {
        "ok": True,
        "service": "phase3_quant_api",
        "root": PHASE3_ROOT.as_posix(),
        "policy": "phase3_specialist_adapter_preflight",
        "stage": "preflight_only",
        "live_mutation": False,
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "all_artifacts_ready": bool(rows) and all(row["artifact_ready"] for row in rows),
        "all_required_imports_ready": bool(rows)
        and all(row["required_imports_ready"] for row in rows),
        "any_shadow_inference_ready": any(row["shadow_inference_ready"] for row in rows),
        "blocked_reasons": sorted(blocked_reasons),
        "chains": chains,
        "adapters": rows,
    }


def _specialist_model_chain(kind: str) -> dict[str, Any]:
    inventory = _phase3_inventory_status()
    by_slot = {
        str(row.get("slot") or ""): row
        for row in inventory.get("model_status", [])
        if isinstance(row, dict)
    }
    models = []
    for expected in SPECIALIST_MODEL_CHAINS.get(kind, []):
        row = by_slot.get(expected["slot"], {})
        status = str(row.get("status") or "missing")
        models.append({**expected, "status": status, "artifact_ready": status == "ok"})
    primary = next((row for row in models if row.get("role") == "primary"), {})
    challenger = next((row for row in models if row.get("role") == "challenger"), {})
    required = [row for row in models if row.get("role") in {"primary", "challenger"}]
    artifacts_ready = bool(required) and all(bool(row.get("artifact_ready")) for row in required)
    return {
        "kind": kind,
        "primary_model": primary.get("repo_id", ""),
        "challenger_model": challenger.get("repo_id", ""),
        "artifacts_ready": artifacts_ready,
        "actual_inference": False,
        "activation_gate": "specialist_adapter_and_walk_forward_required",
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
        "models": models,
    }


def _attach_specialist_shadow(
    tool: str,
    payload: dict[str, Any],
    *,
    kind: str,
    features: dict[str, Any],
    fallback_reason: str,
) -> dict[str, Any]:
    chain = _specialist_model_chain(kind)
    payload["specialist_primary_model"] = chain.get("primary_model")
    payload["specialist_challenger_model"] = chain.get("challenger_model")
    payload["specialist_artifacts_ready"] = bool(chain.get("artifacts_ready"))
    payload["specialist_inference_active"] = False
    payload["specialist_model_chain"] = chain
    payload["professional_model_shadow"] = {
        "kind": kind,
        "primary_model": chain.get("primary_model"),
        "challenger_model": chain.get("challenger_model"),
        "artifacts_ready": bool(chain.get("artifacts_ready")),
        "actual_inference": False,
        "baseline_model": payload.get("model"),
        "baseline_response": True,
        "activation_blocker": "specialist_adapter_and_walk_forward_required",
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
    }
    payload["fallback_reason"] = fallback_reason
    payload.pop("shadow_payload", None)
    return with_model_metadata(
        tool,
        payload,
        features=features,
        challenger_model=str(chain.get("challenger_model") or ""),
        fallback_reason=fallback_reason,
    )


def _attach_baseline_only_shadow(
    tool: str,
    payload: dict[str, Any],
    *,
    kind: str,
    features: dict[str, Any],
    fallback_reason: str,
) -> dict[str, Any]:
    chain = _specialist_model_chain(kind)
    payload["specialist_primary_model"] = chain.get("primary_model")
    payload["specialist_challenger_model"] = chain.get("challenger_model")
    payload["specialist_artifacts_ready"] = bool(chain.get("artifacts_ready"))
    payload["specialist_inference_active"] = False
    payload["specialist_model_chain"] = chain
    payload["professional_model_shadow"] = {
        "kind": kind,
        "primary_model": chain.get("primary_model"),
        "challenger_model": chain.get("challenger_model"),
        "artifacts_ready": bool(chain.get("artifacts_ready")),
        "actual_inference": False,
        "baseline_model": payload.get("model"),
        "baseline_response": True,
        "activation_blocker": fallback_reason,
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
    }
    payload["fallback_reason"] = fallback_reason
    return with_model_metadata(
        tool,
        payload,
        features=features,
        challenger_model=str(chain.get("challenger_model") or ""),
        fallback_reason=fallback_reason,
    )


def _text_items_from_features(features: dict[str, Any], limit: int = 12) -> list[str]:
    raw_items = (
        features.get("recent_headlines")
        or features.get("headlines")
        or features.get("news_headlines")
        or features.get("texts")
        or []
    )
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        return []
    items = []
    for raw in raw_items[:limit]:
        text = str(raw or "").strip()
        if text:
            items.append(text[:512])
    return items


def _sentiment_score_from_label(label: str, score: float) -> float:
    normalized = str(label or "").strip().lower()
    if normalized == "positive":
        return abs(score)
    if normalized == "negative":
        return -abs(score)
    return 0.0


def _load_transformer_classifier(model_dir: str):
    def loader():
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BertConfig,
            BertForSequenceClassification,
            BertTokenizer,
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        except Exception:
            vocab_path = Path(model_dir) / "vocab.txt"
            if not vocab_path.exists():
                raise
            tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir,
                local_files_only=True,
            )
        except Exception as exc:
            config_path = Path(model_dir) / "config.json"
            if "model_type" not in str(exc) or not config_path.exists():
                raise
            config = BertConfig.from_json_file(config_path.as_posix())
            config.model_type = "bert"
            model = BertForSequenceClassification.from_pretrained(
                model_dir,
                config=config,
                local_files_only=True,
            )
        model.eval()
        return tokenizer, model

    return _cache_get_or_load(model_dir, loader)


def _predict_transformer_sentiment(model_dir: str, texts: list[str]) -> dict[str, Any]:
    if not texts:
        return {"available": False, "reason": "no_text_inputs"}
    try:
        import torch

        tokenizer, model = _load_transformer_classifier(model_dir)
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded)
            probabilities = torch.softmax(output.logits, dim=-1)
        id2label = getattr(model.config, "id2label", {}) or {}
        rows = []
        scores = []
        for index, text in enumerate(texts):
            probs = probabilities[index]
            best_index = int(torch.argmax(probs).item())
            confidence = float(probs[best_index].item())
            label = str(id2label.get(best_index) or id2label.get(str(best_index)) or best_index)
            signed = _sentiment_score_from_label(label, confidence)
            scores.append(signed)
            rows.append(
                {
                    "label": label,
                    "confidence": round(confidence, 6),
                    "signed_score": round(signed, 6),
                    "text_preview": text[:120],
                }
            )
        avg_score = float(sum(scores) / max(len(scores), 1))
        return {
            "available": True,
            "text_count": len(texts),
            "score": round(avg_score, 6),
            "label": "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral",
            "rows": rows,
        }
    except Exception as exc:
        return {"available": False, "reason": safe_error(exc, 220)}


def _run_finbert_shadow(features: dict[str, Any]) -> dict[str, Any]:
    texts = _text_items_from_features(features)
    chain = _specialist_model_chain("sentiment")
    model_dirs = {
        "sentiment_primary": PHASE3_ROOT
        / "models"
        / "sentiment"
        / "ProsusAI--finbert",
        "sentiment_challenger": PHASE3_ROOT
        / "models"
        / "sentiment"
        / "yiyanghkust--finbert-tone",
    }
    predictions = {}
    for slot, path in model_dirs.items():
        predictions[slot] = _predict_transformer_sentiment(path.as_posix(), texts)
    available = any(item.get("available") for item in predictions.values())
    primary = predictions.get("sentiment_primary", {})
    challenger = predictions.get("sentiment_challenger", {})
    score_values = [
        float(item.get("score"))
        for item in (primary, challenger)
        if item.get("available") and item.get("score") is not None
    ]
    avg_score = sum(score_values) / len(score_values) if score_values else 0.0
    disagreement = (
        abs(float(primary.get("score") or 0.0) - float(challenger.get("score") or 0.0))
        if primary.get("available") and challenger.get("available")
        else None
    )
    return {
        "available": available,
        "kind": "sentiment",
        "text_count": len(texts),
        "primary_model": chain.get("primary_model"),
        "challenger_model": chain.get("challenger_model"),
        "artifacts_ready": bool(chain.get("artifacts_ready")),
        "actual_inference": available,
        "score": round(avg_score, 6),
        "label": "positive" if avg_score > 0.05 else "negative" if avg_score < -0.05 else "neutral",
        "disagreement": round(disagreement, 6) if disagreement is not None else None,
        "predictions": predictions,
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = load_bundle()
    metadata = {}
    if bundle and isinstance(bundle.get("metadata"), dict):
        metadata = bundle["metadata"]
    payload = {
        "ok": True,
        "service": "phase3_quant_api",
        "root": PHASE3_ROOT.as_posix(),
        "server_role": "dedicated_cryptocurrency_quant_model_server",
        "storage_policy": "new model/cache/training/runtime/log data under /data/BB",
        "legacy_policy": "old data preserved in place but not referenced by Phase 3 runtime",
        "port": PHASE3_API_PORT,
        "policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
        "live_trading_mutation": False,
        "route_mode": "shadow_observation",
        "tools": ["profit", "timeseries", "sentiment", "exit", "train"],
        "trained_models_available": bool(bundle),
        "trained_at": metadata.get("trained_at"),
        "shadow_sample_count": metadata.get("shadow_sample_count", 0),
        "trade_sample_count": metadata.get("trade_sample_count", 0),
        "completed_shadow_sample_count": metadata.get("completed_shadow_sample_count", 0),
        "completed_trade_sample_count": metadata.get("completed_trade_sample_count", 0),
        "quality_report": metadata.get("quality_report", {}),
        "governance_report": metadata.get("governance_report", {}),
        "review_backend": "disabled_use_trading_app_online_model",
        "model_dir": MODEL_DIR.as_posix(),
    }
    payload.update(_phase3_inventory_status())
    payload["specialist_model_chains"] = {
        "timeseries": _specialist_model_chain("timeseries"),
        "sentiment": _specialist_model_chain("sentiment"),
    }
    return payload


@app.get("/models/status")
def local_models_status() -> dict[str, Any]:
    bundle = load_bundle()
    if not bundle:
        return {
            "available": False,
            "message": "No trained local quant bundle found; heuristic fallback is active.",
            "model_path": str(BUNDLE_PATH),
            "specialist_adapter_preflight": _specialist_adapter_preflight(),
        }
    return {
        "available": True,
        "model_path": str(BUNDLE_PATH),
        "specialist_adapter_preflight": _specialist_adapter_preflight(),
        **(bundle.get("metadata") or {}),
    }


@app.get("/specialists/preflight")
def specialist_preflight(kind: str | None = None) -> dict[str, Any]:
    return _specialist_adapter_preflight(kind)


@app.post("/train")
def train(req: TrainRequest) -> dict[str, Any]:
    rows = []
    for sample in req.shadow_samples or []:
        if bool(sample.get("exclude_from_training")):
            continue
        features = sample.get("features") or {}
        horizon = int(sample.get("horizon_minutes") or features.get("horizon_minutes") or 10)
        raw_long_return = f(sample, "long_return_pct")
        raw_short_return = f(sample, "short_return_pct")
        long_return = net_return_pct(raw_long_return)
        short_return = net_return_pct(raw_short_return)
        if not features:
            continue
        rows.append({
            "x": model_x(features, horizon_minutes=horizon),
            "symbol": symbol_key(sample.get("symbol") or features.get("symbol")),
            "horizon": horizon,
            "raw_long_return": raw_long_return,
            "raw_short_return": raw_short_return,
            "long_return": long_return,
            "short_return": short_return,
            "best_side": "long" if long_return >= short_return else "short",
            "lossy_long": int(long_return < -TAIL_LOSS_THRESHOLD_PCT),
            "lossy_short": int(short_return < -TAIL_LOSS_THRESHOLD_PCT),
            "sample_weight": max(0.0, min(f(sample, "sample_weight", 1.0), 1.0)),
        })
    if len(rows) < 200:
        return {
            "trained": False,
            "reason": "not_enough_shadow_samples",
            "shadow_sample_count": len(rows),
            "message": "Need at least 200 completed shadow samples to train.",
        }

    X = [r["x"] for r in rows]
    long_y = [r["long_return"] for r in rows]
    short_y = [r["short_return"] for r in rows]
    long_loss_y = [r["lossy_long"] for r in rows]
    short_loss_y = [r["lossy_short"] for r in rows]
    sample_weights = [max(0.0, float(r.get("sample_weight") or 0.0)) for r in rows]

    long_return_model = _make_regressor()
    short_return_model = _make_regressor()
    long_loss_model = _make_classifier(long_loss_y)
    short_loss_model = _make_classifier(short_loss_y)
    long_return_model.fit(X, long_y, model__sample_weight=sample_weights)
    short_return_model.fit(X, short_y, model__sample_weight=sample_weights)
    long_loss_model.fit(X, long_loss_y, model__sample_weight=sample_weights)
    short_loss_model.fit(X, short_loss_y, model__sample_weight=sample_weights)

    horizon_models: dict[int, dict[str, Any]] = {}
    for horizon in sorted({int(r["horizon"]) for r in rows}):
        h_rows = [r for r in rows if int(r["horizon"]) == horizon]
        if len(h_rows) < 80:
            continue
        hX = [r["x"] for r in h_rows]
        net_y = [max(r["long_return"], r["short_return"], key=abs) for r in h_rows]
        h_weights = [max(0.0, float(r.get("sample_weight") or 0.0)) for r in h_rows]
        model = _make_regressor()
        model.fit(hX, net_y, model__sample_weight=h_weights)
        horizon_models[horizon] = {"model": model, "samples": len(h_rows)}

    deep_sequence_model = _train_sequence_model(req.sequence_samples or [])
    torch_patch_model = _train_torch_patch_model(req.sequence_samples or [])

    sentiment_model = None
    sentiment_samples = []
    for sample in req.shadow_samples or []:
        if bool(sample.get("exclude_from_training")):
            continue
        features = sample.get("features") or {}
        if not features:
            continue
        sentiment_samples.append((
            [feature_row(features).get(key, 0.0) for key in SENTIMENT_KEYS],
            max(net_return_pct(f(sample, "long_return_pct")), net_return_pct(f(sample, "short_return_pct"))),
            max(0.0, min(f(sample, "sample_weight", 1.0), 1.0)),
        ))
    if len(sentiment_samples) >= 200:
        sentiment_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=180,
                max_depth=8,
                min_samples_leaf=10,
                random_state=43,
                n_jobs=-1,
            )),
        ])
        sentiment_model.fit(
            [x for x, _, _ in sentiment_samples],
            [y for _, y, _ in sentiment_samples],
            model__sample_weight=[weight for _, _, weight in sentiment_samples],
        )
    text_sentiment_model = _train_text_sentiment_model(req.text_sentiment_samples or [])
    transformers_sentiment_backend = _probe_transformers_sentiment_backend()

    trainable_trade_samples = [
        sample for sample in (req.trade_samples or []) if not bool(sample.get("exclude_from_training"))
    ]
    profiles = _train_profiles(trainable_trade_samples)
    evaluation_policy = req.evaluation_policy or {
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
        "requires_walk_forward": True,
        "phase": "phase3_model_factory",
    }
    evaluation_policy.setdefault("promotion_flow", PHASE3_REQUIRED_PROMOTION_FLOW)
    evaluation_policy.setdefault("live_mutation", False)
    evaluation_policy.setdefault("requires_walk_forward", True)
    evaluation_policy.setdefault("phase", "phase3_model_factory")
    metadata = {
        "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "phase": "phase3_model_factory",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": req.source,
        "shadow_sample_count": len(rows),
        "completed_shadow_sample_count": int(req.completed_shadow_sample_count or len(rows)),
        "last_trained_completed_shadow_sample_count": int(
            req.completed_shadow_sample_count or len(rows)
        ),
        "trade_sample_count": len(trainable_trade_samples),
        "completed_trade_sample_count": int(
            req.completed_trade_sample_count or len(trainable_trade_samples)
        ),
        "last_trained_completed_trade_sample_count": int(
            req.completed_trade_sample_count or len(trainable_trade_samples)
        ),
        "sequence_sample_count": int((deep_sequence_model or {}).get("samples") or 0),
        "text_sentiment_sample_count": int((text_sentiment_model or {}).get("samples") or 0),
        "torch_patch_available": bool((torch_patch_model or {}).get("available")),
        "torch_patch_status": _public_torch_patch_status(torch_patch_model),
        "transformers_sentiment_backend": transformers_sentiment_backend,
        "feature_count": len(FEATURE_KEYS),
        "horizons": sorted(horizon_models),
        "profile_count": len(profiles),
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "tail_loss_threshold_pct": TAIL_LOSS_THRESHOLD_PCT,
        "quality_report": req.quality_report or {},
        "governance_report": req.governance_report or {},
        "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
        "trade_sample_cursor_policy": PHASE3_REQUIRED_TRAINING_POLICY,
        "training_mode": str(req.training_mode or "shadow"),
        "model_stage": str(req.model_stage or "shadow"),
        "evaluation_policy": evaluation_policy,
        "artifact_persisted": bool(req.persist_artifact and req.confirm_phase3_rebuild),
        "preflight_only": not bool(req.persist_artifact and req.confirm_phase3_rebuild),
        "persist_artifact_requested": bool(req.persist_artifact),
        "confirm_phase3_rebuild": bool(req.confirm_phase3_rebuild),
        "promotion_recommendation": req.promotion_recommendation or {},
        "training_objective": "Predict executable net return after estimated fees/slippage; win rate is auxiliary.",
        "models": {
            "profit": "ExtraTreesRegressor long/short expected return",
            "loss_filter": "ExtraTreesClassifier side-specific loss probability",
            "timeseries": "Per-horizon ExtraTreesRegressor",
            "deep_timeseries": (
                "Torch PatchTST/TFT-style sequence model"
                if (torch_patch_model or {}).get("available")
                else ("Sequence ExtraTreesRegressor PatchTST/TFT-style input" if deep_sequence_model else "not enough kline sequences")
            ),
            "sentiment": "RandomForest sentiment calibration" if sentiment_model else "heuristic fallback",
            "deep_sentiment": (
                "Transformers-ready text sentiment + TF-IDF Ridge model"
                if (transformers_sentiment_backend or {}).get("available") and text_sentiment_model
                else ("TF-IDF Ridge text sentiment model" if text_sentiment_model else "not enough text samples")
            ),
            "exit": "trade-profile plus live pnl rules",
        },
        "objective": "Maximize expected realized net profit; win rate is auxiliary only.",
    }
    bundle = {
        "metadata": metadata,
        "feature_keys": FEATURE_KEYS,
        "long_return_model": long_return_model,
        "short_return_model": short_return_model,
        "long_loss_model": long_loss_model,
        "short_loss_model": short_loss_model,
        "horizon_models": horizon_models,
        "deep_sequence_model": deep_sequence_model,
        "torch_patch_model": torch_patch_model,
        "sentiment_model": sentiment_model,
        "text_sentiment_model": text_sentiment_model,
        "transformers_sentiment_backend": transformers_sentiment_backend,
        "profiles": profiles,
    }
    if not req.persist_artifact:
        return {
            "trained": False,
            "reason": "phase3_preflight_no_artifact_write",
            **metadata,
        }
    if not req.confirm_phase3_rebuild:
        return {
            "trained": False,
            "reason": "phase3_rebuild_confirmation_required",
            **metadata,
        }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump_trusted_joblib_bundle(bundle, BUNDLE_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    global _BUNDLE_CACHE, _BUNDLE_MTIME
    _BUNDLE_CACHE = bundle
    _BUNDLE_MTIME = BUNDLE_PATH.stat().st_mtime
    return {"trained": True, **metadata}


@app.post("/profit/predict")
def profit_predict(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    bundle = load_bundle()
    profile_key_long = f"{symbol_key(req.symbol or features.get('symbol'))}|long"
    profile_key_short = f"{symbol_key(req.symbol or features.get('symbol'))}|short"
    if bundle:
        try:
            x = [model_x(features)]
            long_expected = float(bundle["long_return_model"].predict(x)[0])
            short_expected = float(bundle["short_return_model"].predict(x)[0])
            long_loss_prob = predict_proba_positive(bundle["long_loss_model"], x)
            short_loss_prob = predict_proba_positive(bundle["short_loss_model"], x)
            profiles = bundle.get("profiles") or {}
            long_profile = profiles.get(profile_key_long, {})
            short_profile = profiles.get(profile_key_short, {})
            long_profile_penalty = float(long_profile.get("loss_pressure") or 0.0) * 0.18
            short_profile_penalty = float(short_profile.get("loss_pressure") or 0.0) * 0.18
            adjusted_long = long_expected - long_loss_prob * 0.22 - long_profile_penalty
            adjusted_short = short_expected - short_loss_prob * 0.22 - short_profile_penalty
            best_side = "long" if adjusted_long >= adjusted_short else "short"
            best_expected = adjusted_long if best_side == "long" else adjusted_short
            edge = abs(adjusted_long - adjusted_short)
            loss_prob = long_loss_prob if best_side == "long" else short_loss_prob
            quality = max(best_expected, 0.0) + edge * 0.45 - loss_prob * 0.18
            return _attach_baseline_only_shadow("profit_prediction", {
                "available": True,
                "trained": True,
                "model": "local-profit-trained-v2",
                "symbol": req.symbol,
                "best_side": best_side,
                "long_expected_return_pct": round(long_expected, 4),
                "short_expected_return_pct": round(short_expected, 4),
                "adjusted_long_return_pct": round(adjusted_long, 4),
                "adjusted_short_return_pct": round(adjusted_short, 4),
                "expected_return_pct": round(best_expected, 4),
                "adjusted_expected_return_pct": round(best_expected, 4),
                "profit_edge_pct": round(edge, 4),
                "profit_quality_score": round(quality, 4),
                "long_loss_probability": round(long_loss_prob, 4),
                "short_loss_probability": round(short_loss_prob, 4),
                "loss_probability": round(loss_prob, 4),
                "symbol_side_profile": {
                    "long": long_profile,
                    "short": short_profile,
                },
                "note": "Trained profit-first model: expected return and loss probability drive the score; win rate is not the objective.",
            }, kind="profit", features=features, fallback_reason="profit_specialist_pending_phase3_clean_rebuild")
        except Exception as exc:
            fallback_error = safe_error(exc)
    else:
        fallback_error = None

    long_score, short_score = side_scores(features)
    atr_pct = abs(f(features, "atr_14")) / max(f(features, "current_price", f(features, "close", 0.0)), 1e-9)
    volatility = abs(f(features, "volatility_20"))
    spread = abs(f(features, "spread_pct"))
    liquidity_penalty = max(0.0, spread - 0.03) * 0.8
    risk_penalty = volatility * 0.45 + atr_pct * 0.35 + liquidity_penalty

    long_expected = long_score - risk_penalty
    short_expected = short_score - risk_penalty
    best_side = "long" if long_expected >= short_expected else "short"
    best_expected = long_expected if best_side == "long" else short_expected
    edge = abs(long_expected - short_expected)
    loss_prob = clamp((risk_penalty + max(-best_expected, 0.0)) / 2.0, 0.0, 1.0)
    quality = max(best_expected, 0.0) + edge * 0.35 - risk_penalty * 0.5
    return _attach_baseline_only_shadow("profit_prediction", {
        "available": True,
        "trained": False,
        "model": "local-profit-heuristic-v1",
        "symbol": req.symbol,
        "best_side": best_side,
        "long_expected_return_pct": round(long_expected, 4),
        "short_expected_return_pct": round(short_expected, 4),
        "expected_return_pct": round(best_expected, 4),
        "adjusted_expected_return_pct": round(best_expected, 4),
        "profit_edge_pct": round(edge, 4),
        "profit_quality_score": round(quality, 4),
        "loss_probability": round(loss_prob, 4),
        "long_loss_probability": round(clamp((risk_penalty + max(-long_expected, 0.0)) / 2.0, 0.0, 1.0), 4),
        "short_loss_probability": round(clamp((risk_penalty + max(-short_expected, 0.0)) / 2.0, 0.0, 1.0), 4),
        "risk_penalty": round(risk_penalty, 4),
        "fallback_error": fallback_error,
        "note": "Profit-first local signal; win rate is not used as the primary objective.",
    }, kind="profit", features=features, fallback_reason=fallback_error or "trained_profit_model_unavailable")


@app.post("/timeseries/predict")
def timeseries_predict(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    bundle = load_bundle()
    if bundle:
        predictions = []
        try:
            for horizon, item in (bundle.get("horizon_models") or {}).items():
                move = float(item["model"].predict([model_x(features, horizon_minutes=int(horizon))])[0])
                predictions.append({
                    "horizon_minutes": int(horizon),
                    "expected_move_pct": round(move, 4),
                    "direction": "up" if move > 0 else "down" if move < 0 else "flat",
                    "samples": int(item.get("samples") or 0),
                })
            if predictions:
                primary = sorted(predictions, key=lambda r: abs(float(r["expected_move_pct"])), reverse=True)[0]
                confidence = clamp(abs(float(primary["expected_move_pct"])) / 0.8, 0.0, 1.0)
                best_side = "long" if primary["direction"] == "up" else "short" if primary["direction"] == "down" else "hold"
                return with_model_metadata("time_series_prediction", {
                    "available": True,
                    "trained": True,
                    "model": "local-timeseries-trained-v2",
                    "architecture": "tree_horizon_ensemble",
                    "symbol": req.symbol,
                    "best_side": best_side,
                    "side": best_side,
                    "direction": primary["direction"],
                    "expected_move_pct": primary["expected_move_pct"],
                    "expected_return_pct": primary["expected_move_pct"],
                    "confidence": round(confidence, 4),
                    "predictions": predictions,
                }, features=features)
        except Exception:
            pass

    returns = np.array([f(features, "returns_1"), f(features, "returns_5"), f(features, "returns_20")], dtype=float)
    weights = np.array([0.20, 0.35, 0.45], dtype=float)
    forecast = float(np.dot(returns, weights))
    vol = max(abs(f(features, "volatility_20")), 1e-6)
    confidence = clamp(abs(forecast) / (vol * 1.8 + 1e-6), 0.0, 1.0)
    direction = "up" if forecast > 0 else "down" if forecast < 0 else "flat"
    best_side = "long" if direction == "up" else "short" if direction == "down" else "hold"
    return with_model_metadata("time_series_prediction", {
        "available": True,
        "trained": False,
        "model": "local-timeseries-ensemble-v1",
        "architecture": "lightweight_momentum_fallback",
        "symbol": req.symbol,
        "best_side": best_side,
        "side": best_side,
        "direction": direction,
        "expected_move_pct": round(forecast * 100.0, 4),
        "expected_return_pct": round(forecast * 100.0, 4),
        "confidence": round(confidence, 4),
    }, features=features, fallback_reason="trained_timeseries_model_unavailable")


@app.post("/timeseries/deep/predict")
def deep_timeseries_predict(req: FeatureRequest) -> dict[str, Any]:
    """Sequence time-series service slot with PatchTST/TFT-style inputs."""
    features = req.features or {}
    bundle = load_bundle()
    torch_patch_model = (bundle or {}).get("torch_patch_model") or {}
    sequence_model = (bundle or {}).get("deep_sequence_model") or {}
    close_sequence, sequence_reason, sequence_source = _timeseries_close_sequence(features)
    volume_sequence = features.get("volume_sequence") or features.get("recent_volumes")
    try:
        torch_expected = (
            None
            if sequence_reason
            else _predict_torch_patch_model(torch_patch_model, close_sequence, volume_sequence)
        )
        if torch_expected is not None:
            confidence = clamp(abs(torch_expected) / 0.8, 0.0, 1.0)
            direction = "up" if torch_expected > 0 else "down" if torch_expected < 0 else "flat"
            best_side = "long" if direction == "up" else "short" if direction == "down" else "hold"
            return _attach_timeseries_specialist_shadow({
                "available": True,
                "trained": True,
                "model": "local-torch-patch-timeseries-v1",
                "architecture": "torch_patch_mlp_tft_patchtst_style",
                "symbol": req.symbol,
                "best_side": best_side,
                "side": best_side,
                "direction": direction,
                "expected_move_pct": round(torch_expected, 4),
                "expected_return_pct": round(torch_expected, 4),
                "confidence": round(confidence, 4),
                "sample_count": int(torch_patch_model.get("samples") or 0),
                "train_mae_pct": torch_patch_model.get("train_mae_pct"),
                "endpoint": "timeseries_deep",
                "model_family": "PatchTST/TFT-style torch sequence model",
                "status": "trained_torch_sequence_model",
                "sequence_length": len(close_sequence),
                "sequence_source": sequence_source,
            }, features=features)
        model = sequence_model.get("model")
        if model and not sequence_reason:
            expected = float(model.predict([sequence_features(close_sequence, volume_sequence)])[0])
            confidence = clamp(abs(expected) / 0.8, 0.0, 1.0)
            direction = "up" if expected > 0 else "down" if expected < 0 else "flat"
            best_side = "long" if direction == "up" else "short" if direction == "down" else "hold"
            return _attach_timeseries_specialist_shadow({
                "available": True,
                "trained": True,
                "model": "local-sequence-timeseries-v1",
                "architecture": "sequence_extra_trees_patchtst_tft_style",
                "symbol": req.symbol,
                "best_side": best_side,
                "side": best_side,
                "direction": direction,
                "expected_move_pct": round(expected, 4),
                "expected_return_pct": round(expected, 4),
                "confidence": round(confidence, 4),
                "sample_count": int(sequence_model.get("samples") or 0),
                "timeframes": sequence_model.get("timeframes") or {},
                "endpoint": "timeseries_deep",
                "model_family": "PatchTST/TFT-style sequence model",
                "status": "trained_sequence_model",
                "sequence_length": len(close_sequence),
                "sequence_source": sequence_source,
            }, features=features)
    except Exception:
        pass
    base = timeseries_predict(req)
    base.update(
        {
            "endpoint": "timeseries_deep",
            "model_family": "Chronos-2/TimesFM shadow-ready time-series chain",
            "status": (
                "trained_horizon_fallback" if base.get("trained") else "heuristic_fallback"
            ),
            "note": (
                "Chronos-2/TimesFM artifacts are audited separately; this response remains "
                "baseline-only until specialist adapters pass walk-forward gates."
            ),
            "sequence_input_status": sequence_reason or "real_sequence_ready",
            "sequence_length": len(close_sequence),
            "sequence_source": sequence_source,
            "minimum_sequence_length": MIN_TIMESERIES_SEQUENCE_LENGTH,
        }
    )
    return _attach_timeseries_specialist_shadow(
        base,
        features=features,
    )


@app.post("/sentiment/analyze")
def sentiment_analyze(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    news = f(features, "news_sentiment_avg")
    social = f(features, "social_sentiment_avg")
    mentions = f(features, "social_mention_count")
    articles = f(features, "news_article_count")
    score = news * 0.55 + social * 0.45
    trained_expected = None
    bundle = load_bundle()
    try:
        sentiment_model = bundle.get("sentiment_model") if bundle else None
        if sentiment_model:
            x = [[feature_row(features).get(key, 0.0) for key in SENTIMENT_KEYS]]
            trained_expected = float(sentiment_model.predict(x)[0])
            score = score * 0.55 + clamp(trained_expected / 1.5, -1.0, 1.0) * 0.45
    except Exception:
        trained_expected = None
    text_score = None
    try:
        text_model = (bundle or {}).get("text_sentiment_model") or {}
        model = text_model.get("model")
        texts = features.get("recent_headlines") or features.get("headlines") or []
        if model and isinstance(texts, list) and texts:
            text_blob = " ".join(str(t) for t in texts[:12] if t)
            if text_blob.strip():
                text_score = float(model.predict([text_blob])[0])
                score = score * 0.5 + clamp(text_score, -1.0, 1.0) * 0.5
    except Exception:
        text_score = None
    if mentions <= 0 and articles <= 0 and abs(score) < 0.03:
        label = "neutral"
        risk = "unknown"
    elif score > 0.08:
        label = "positive"
        risk = "normal"
    elif score < -0.08:
        label = "negative"
        risk = "elevated"
    else:
        label = "neutral"
        risk = "normal"
    best_side = "long" if label == "positive" else "short" if label == "negative" else "hold"
    expected_from_sentiment = round(trained_expected, 4) if trained_expected is not None else None
    return with_model_metadata("sentiment_analysis", {
        "available": True,
        "trained": trained_expected is not None,
        "model": "local-sentiment-trained-v2" if trained_expected is not None else "local-sentiment-light-v1",
        "architecture": "finbert_cryptobert_ready_calibrator" if trained_expected is not None else "lexicon_feature_fallback",
        "symbol": req.symbol,
        "best_side": best_side,
        "side": best_side,
        "label": label,
        "score": round(score, 4),
        "expected_return_pct": expected_from_sentiment,
        "expected_return_from_sentiment_pct": expected_from_sentiment,
        "text_sentiment_score": round(text_score, 4) if text_score is not None else None,
        "risk_level": risk,
        "mentions": int(mentions),
        "articles": int(articles),
    }, features=features, fallback_reason="" if trained_expected is not None else "trained_sentiment_model_unavailable")


@app.post("/sentiment/deep/analyze")
def deep_sentiment_analyze(req: FeatureRequest) -> dict[str, Any]:
    """Independent text sentiment service slot for CryptoBERT/FinBERT style models."""
    features = req.features or {}
    base = sentiment_analyze(req)
    specialist_shadow = _run_finbert_shadow(features)
    base.update(
        {
            "endpoint": "sentiment_deep",
            "model_family": "FinBERT shadow-ready sentiment chain",
            "status": (
                "specialist_shadow_inference"
                if specialist_shadow.get("available")
                else "trained_text_model"
                if base.get("text_sentiment_score") is not None
                else ("trained_calibrator" if base.get("trained") else "feature_fallback")
            ),
            "note": (
                "FinBERT specialist inference is shadow-only and cannot mutate live routing."
                if specialist_shadow.get("available")
                else "FinBERT artifacts are audited separately; this response remains baseline-only "
                "until specialist adapters pass evaluation gates."
            ),
        }
    )
    payload = _attach_specialist_shadow(
        "sentiment_analysis",
        base,
        kind="sentiment",
        features=features,
        fallback_reason=(
            "specialist_sentiment_shadow_only"
            if specialist_shadow.get("available")
            else "specialist_sentiment_adapter_not_promoted"
        ),
    )
    payload["specialist_inference_active"] = bool(specialist_shadow.get("available"))
    payload["professional_model_shadow"].update(specialist_shadow)
    payload["professional_model_shadow"]["baseline_response"] = True
    payload.pop("shadow_payload", None)
    return with_model_metadata(
        "sentiment_analysis",
        payload,
        features=features,
        challenger_model=str(payload.get("specialist_challenger_model") or ""),
        fallback_reason=payload.get("fallback_reason") or "",
    )


@app.post("/exit/advise")
def exit_advise(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    symbol = symbol_key(req.symbol or features.get("symbol"))
    bundle = load_bundle()
    profiles = (bundle or {}).get("profiles") or {}
    positions = []
    for pos in req.open_positions or []:
        if symbol_key(pos.get("symbol")) == symbol:
            positions.append(pos)
    if not positions:
        return with_model_metadata("exit_advice", {
            "available": True,
            "trained": bool(bundle),
            "model": "local-exit-advisor-v1",
            "symbol": req.symbol,
            "action": "hold",
            "no_matching_position": True,
            "reason": "本轮没有传入与该币种匹配的当前持仓，平仓建议模型不参与。",
        }, features=features, fallback_reason="no_matching_open_position")
    advices = []
    for pos in positions:
        side = str(pos.get("side") or "").lower()
        pnl_pct = f(pos, "unrealized_pnl_pct", f(pos, "pnl_pct"))
        unrealized = f(pos, "unrealized_pnl")
        hold = f(pos, "hold_minutes")
        profile = profiles.get(f"{symbol}|{side}", {})
        loss_pressure = float(profile.get("loss_pressure") or 0.0)
        profit_factor = float(profile.get("profit_factor") or 0.0)
        payoff_ratio = float(profile.get("payoff_ratio") or 0.0)
        small_win_big_loss_risk = float(profile.get("small_win_big_loss_risk") or 0.0)
        avg_loss = float(profile.get("avg_loss") or 0.0)
        action = "hold"
        urgency = 0.25
        reason = "平仓建议模型未识别到明确的主动平仓压力，本轮倾向继续持有。"
        if unrealized < 0 and (loss_pressure >= 0.55 or profit_factor < 0.8 or small_win_big_loss_risk >= 0.45):
            action = "reduce_or_close"
            urgency = 0.82 if abs(unrealized) >= max(avg_loss * 0.35, 3.0) else 0.74
            reason = (
                "当前仓位处于亏损，且该币种/方向历史真实交易存在亏损压力或小赚大亏结构；"
                "若短线修复证据不足，建议减仓或平仓压缩亏损。"
            )
        elif pnl_pct >= 0.006 and (loss_pressure >= 0.45 or small_win_big_loss_risk >= 0.40):
            action = "protect_profit"
            urgency = 0.72
            reason = (
                "当前已有浮盈，但历史画像显示回吐/亏损压力偏高；"
                "建议优先保护利润，避免盈利仓拖成亏损仓。"
            )
        elif pnl_pct <= -0.012:
            action = "reduce_or_close"
            urgency = 0.72
            reason = "亏损扩大到本地平仓模型容忍线之外，建议减仓或平仓压缩尾部亏损。"
        elif pnl_pct >= 0.012 and profit_factor >= 1.2:
            action = "trail_profit"
            urgency = 0.52
            reason = "当前持仓盈利且历史盈亏质量尚可，建议移动保护利润，不急于完全限制上行空间。"
        advices.append({
            "side": side,
            "unrealized_pnl": round(unrealized, 4),
            "pnl_pct": round(pnl_pct, 5),
            "hold_minutes": round(hold, 2),
            "action": action,
            "urgency": round(urgency, 3),
            "reason": reason,
            "profile": profile,
            "payoff_ratio": round(payoff_ratio, 4),
            "small_win_big_loss_risk": round(small_win_big_loss_risk, 4),
        })
    top = sorted(advices, key=lambda r: float(r["urgency"]), reverse=True)[0]
    return with_model_metadata("exit_advice", {
        "available": True,
        "trained": bool(bundle),
        "model": "local-exit-advisor-v1",
        "symbol": req.symbol,
        "action": top["action"],
        "urgency": top["urgency"],
        "reason": top["reason"],
        "advices": advices,
    }, features=features)


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [],
        "disabled": LOCAL_REVIEW_DISABLED_DETAIL,
    }


@app.post("/v1/chat/completions")
def chat_completions(_payload: dict[str, Any]) -> Any:
    raise HTTPException(status_code=410, detail=LOCAL_REVIEW_DISABLED_DETAIL)
'''


def sh(value: str | int | float) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def render_phase3_quant_api_service() -> str:
    """Render the Phase 3 quant API systemd unit rooted under /data/BB."""

    env_bin = PurePosixPath(PHASE3_PYTHON_BIN).parent.as_posix()
    return (
        textwrap.dedent(
            f"""
            [Unit]
            Description=BB Phase 3 Quant API - local_ai_tools v2 shadow contracts
            After=network-online.target
            Wants=network-online.target

            [Service]
            Type=simple
            User=root
            WorkingDirectory={PHASE3_APP_DIR}
            Environment=PATH={env_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
            Environment=BB_PHASE3_ROOT={PHASE3_ROOT}
            Environment=PHASE3_QUANT_API_PORT={PHASE3_API_PORT}
            Environment=LOCAL_AI_TOOLS_MODEL_DIR={PHASE3_MODEL_DIR}
            Environment=LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true
            Environment=LOCAL_AI_TOOLS_CORS_ORIGINS=http://127.0.0.1:8002,http://localhost:8002,http://127.0.0.1:18001
            Environment=LOCAL_AI_TOOLS_ROUND_TRIP_COST_PCT=0.12
            Environment=LOCAL_AI_TOOLS_TAIL_LOSS_THRESHOLD_PCT=0.18
            EnvironmentFile=-{PHASE3_ENV_FILE}
            LimitNOFILE=65535
            ExecStart={PHASE3_PYTHON_BIN} -m uvicorn local_ai_tools_api:app --host 127.0.0.1 --port {PHASE3_API_PORT} --timeout-keep-alive 5
            Restart=always
            RestartSec=5
            StandardOutput=append:{PHASE3_LOG_DIR}/phase3_quant_api.log
            StandardError=append:{PHASE3_LOG_DIR}/phase3_quant_api.err.log

            [Install]
            WantedBy=multi-user.target
            """
        ).strip()
        + "\n"
    )


def render_phase3_deploy_plan() -> dict[str, Any]:
    return {
        "policy_id": PHASE3_POLICY_ID,
        "phase3_root": PHASE3_ROOT,
        "service_name": PHASE3_SERVICE_NAME,
        "app_dir": PHASE3_APP_DIR,
        "systemd_dir": PHASE3_SYSTEMD_DIR,
        "log_dir": PHASE3_LOG_DIR,
        "model_dir": PHASE3_MODEL_DIR,
        "runtime_dir": PHASE3_RUNTIME_DIR,
        "env_file": PHASE3_ENV_FILE,
        "python_bin": PHASE3_PYTHON_BIN,
        "port": PHASE3_API_PORT,
        "health_url": f"http://127.0.0.1:{PHASE3_API_PORT}/health",
        "shadow_only": True,
        "live_mutation": False,
        "promotion_flow": "shadow_to_canary_to_live",
        "legacy_root_used": False,
    }


def _upload_text(ssh, remote_path: str, content: str, *, mode: int = 0o644) -> None:
    directory = posixpath.dirname(remote_path)
    run_remote_text(ssh, f"mkdir -p {sh(directory)}", timeout=30, check=True)
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "w") as remote:
            remote.write(content)
        sftp.chmod(remote_path, mode)
    finally:
        sftp.close()


def _remote_preflight_command() -> str:
    return " && ".join(
        [
            f"test -x {sh(PHASE3_PYTHON_BIN)}",
            f"mkdir -p {sh(PHASE3_APP_DIR)} {sh(PHASE3_SYSTEMD_DIR)} {sh(PHASE3_LOG_DIR)} "
            f"{sh(PHASE3_MODEL_DIR)} {sh(PHASE3_RUNTIME_DIR)} {sh(f'{PHASE3_ROOT}/manifests')} "
            f"{sh(PurePosixPath(PHASE3_ENV_FILE).parent.as_posix())}",
            f"touch {sh(PHASE3_ENV_FILE)}",
            f"chmod 600 {sh(PHASE3_ENV_FILE)}",
            f"{PHASE3_PYTHON_BIN} - <<'PY'\n"
            "import fastapi, joblib, numpy, sklearn, uvicorn\n"
            "print('phase3_quant_api_deps_ok')\n"
            "PY",
        ]
    )


def _stop_legacy_8101_holder_command() -> str:
    """Stop the old ad-hoc 8101 inventory API before systemd owns the port."""

    return "\n".join(
        [
            "set -euo pipefail",
            f"new_service={sh(PHASE3_SERVICE_NAME)}",
            f"new_app={sh(PHASE3_APP_DIR + '/local_ai_tools_api.py')}",
            "holders=$(ss -ltnp 'sport = :8101' 2>/dev/null | sed -n 's/.*pid=\\([0-9][0-9]*\\).*/\\1/p' | sort -u || true)",
            "for pid in ${holders}; do",
            "  [ -n \"${pid}\" ] || continue",
            "  cmdline=$(tr '\\0' ' ' < /proc/${pid}/cmdline 2>/dev/null || true)",
            "  unit=$(systemctl status ${pid} --no-pager 2>/dev/null | sed -n 's/^.*CGroup: \\/system.slice\\/\\([^ ]*\\.service\\).*$/\\1/p' | head -1 || true)",
            "  if printf '%s' \"${cmdline}\" | grep -F \"$new_app\" >/dev/null; then",
            "    continue",
            "  fi",
            "  if [ -n \"${unit}\" ] && [ \"${unit}\" != \"$new_service\" ]; then",
            "    systemctl stop \"${unit}\" || true",
            "    systemctl disable \"${unit}\" || true",
            "  fi",
            "  if kill -0 \"${pid}\" 2>/dev/null; then",
            "    kill \"${pid}\" || true",
            "    sleep 2",
            "  fi",
            "  if kill -0 \"${pid}\" 2>/dev/null; then",
            "    kill -9 \"${pid}\" || true",
            "  fi",
            "done",
        ]
    )


def _remote_smoke_command() -> str:
    return (
        f"{PHASE3_PYTHON_BIN} - <<'PY'\n"
        "import json\n"
        "import urllib.request\n"
        "\n"
        f"BASE = 'http://127.0.0.1:{PHASE3_API_PORT}'\n"
        f"ENV_FILE = {PHASE3_ENV_FILE!r}\n"
        "\n"
        "def api_key():\n"
        "    try:\n"
        "        for raw_line in open(ENV_FILE, encoding='utf-8'):\n"
        "            line = raw_line.strip()\n"
        "            if line.startswith('LOCAL_AI_TOOLS_API_KEY='):\n"
        "                return line.split('=', 1)[1].strip().strip(chr(34)).strip(chr(39))\n"
        "    except FileNotFoundError:\n"
        "        pass\n"
        "    return ''\n"
        "\n"
        "def get(path):\n"
        "    headers = {}\n"
        "    key = api_key()\n"
        "    if key:\n"
        "        headers['Authorization'] = 'Bearer ' + key\n"
        "    request = urllib.request.Request(BASE + path, headers=headers)\n"
        "    with urllib.request.urlopen(request, timeout=8) as response:\n"
        "        return json.loads(response.read(256000).decode('utf-8'))\n"
        "\n"
        "def post(path, payload):\n"
        "    data = json.dumps(payload).encode('utf-8')\n"
        "    headers = {'Content-Type': 'application/json'}\n"
        "    key = api_key()\n"
        "    if key:\n"
        "        headers['Authorization'] = 'Bearer ' + key\n"
        "    request = urllib.request.Request(\n"
        "        BASE + path,\n"
        "        data=data,\n"
        "        headers=headers,\n"
        "        method='POST',\n"
        "    )\n"
        "    with urllib.request.urlopen(request, timeout=8) as response:\n"
        "        return json.loads(response.read(256000).decode('utf-8'))\n"
        "\n"
        "features = {\n"
        "    'current_price': 100.0,\n"
        "    'close': 100.0,\n"
        "    'returns_1': 0.01,\n"
        "    'returns_5': 0.02,\n"
        "    'returns_20': 0.03,\n"
        "    'rsi_14': 55.0,\n"
        "    'volume_ratio': 1.1,\n"
        "}\n"
        "health = get('/health')\n"
        "profit = post('/profit/predict', {'symbol': 'BTC/USDT', 'features': features})\n"
        "exit_advice = post('/exit/advise', {'symbol': 'BTC/USDT', 'features': features, 'open_positions': []})\n"
        "assert health.get('service') == 'phase3_quant_api', health\n"
        "assert health.get('root') == '/data/BB', health\n"
        "assert health.get('live_mutation') is False, health\n"
        "assert profit.get('shadow_payload', {}).get('tool') == 'profit_prediction', profit\n"
        "assert profit.get('live_mutation') is False, profit\n"
        "assert 'adjusted_expected_return_pct' in profit, profit\n"
        "assert 'loss_probability' in profit, profit\n"
        "assert exit_advice.get('action') == 'hold', exit_advice\n"
        "assert exit_advice.get('no_matching_position') is True, exit_advice\n"
        "print(json.dumps({\n"
        "    'event': 'phase3_quant_api_smoke_ok',\n"
        "    'health': health,\n"
        "    'profit_contract': {\n"
        "        'shadow_payload': bool(profit.get('shadow_payload')),\n"
        "        'live_mutation': profit.get('live_mutation'),\n"
        "        'promotion_flow': profit.get('promotion_flow'),\n"
        "    },\n"
        "    'exit_contract': {\n"
        "        'action': exit_advice.get('action'),\n"
        "        'no_matching_position': exit_advice.get('no_matching_position'),\n"
        "    },\n"
        "}, ensure_ascii=False, indent=2, sort_keys=True))\n"
        "PY"
    )


def deploy_phase3_quant_api(*, plan_only: bool = False, start: bool = True) -> None:
    safe_print(json.dumps(render_phase3_deploy_plan(), ensure_ascii=False, indent=2, sort_keys=True))
    if plan_only:
        return

    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        run_remote_text(ssh, _remote_preflight_command(), timeout=180, check=True)
        _upload_text(ssh, f"{PHASE3_APP_DIR}/local_ai_tools_api.py", SERVICE_CODE)
        staged_service_path = f"{PHASE3_SYSTEMD_DIR}/{PHASE3_SERVICE_NAME}"
        _upload_text(ssh, staged_service_path, render_phase3_quant_api_service())
        _upload_text(
            ssh,
            f"{PHASE3_ROOT}/manifests/phase3_quant_api_manifest.json",
            json.dumps(render_phase3_deploy_plan(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        run_remote_text(
            ssh,
            f"install -m 0644 {sh(staged_service_path)} /etc/systemd/system/{sh(PHASE3_SERVICE_NAME)} && "
            "systemctl daemon-reload",
            timeout=60,
            check=True,
        )
        if not start:
            safe_print("Phase 3 quant API installed but not started.")
            return
        run_remote_text(
            ssh,
            _stop_legacy_8101_holder_command(),
            timeout=60,
            check=True,
            max_output_chars=20_000,
        )
        run_remote_text(
            ssh,
            f"systemctl enable {sh(PHASE3_SERVICE_NAME)} && "
            f"systemctl restart {sh(PHASE3_SERVICE_NAME)}",
            timeout=90,
            check=True,
        )
        safe_print(
            run_remote_text(
                ssh,
                f"systemctl is-active {sh(PHASE3_SERVICE_NAME)} && sleep 2 && "
                + _remote_smoke_command(),
                timeout=180,
                check=True,
                max_output_chars=80_000,
            )
        )
    finally:
        ssh.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--install-only",
        action="store_true",
        help="Install files and systemd unit without restarting the Phase 3 quant API.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    deploy_phase3_quant_api(plan_only=bool(args.plan_only), start=not bool(args.install_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
