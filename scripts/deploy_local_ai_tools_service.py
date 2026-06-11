"""Deploy the lightweight local AI tools API to the configured server."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

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


MODEL_DIR = Path("/data/trade_ai/models")
BUNDLE_PATH = MODEL_DIR / "local_quant_models.joblib"
METADATA_PATH = MODEL_DIR / "local_quant_models_metadata.json"
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
ROUND_TRIP_COST_PCT = 0.12
TAIL_LOSS_THRESHOLD_PCT = 0.18
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


def _text_value(row: dict[str, Any]) -> str:
    text = str(row.get("text") or "").strip()
    platform = str(row.get("platform") or "")
    symbols = " ".join(str(s) for s in (row.get("symbols") or [])[:8])
    return " ".join(part for part in (platform, symbols, text) if part).strip()


def _train_text_sentiment_model(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    rows = [(_text_value(sample), f(sample, "sentiment_score")) for sample in samples or []]
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


@app.get("/health")
def health() -> dict[str, Any]:
    bundle = load_bundle()
    metadata = {}
    if bundle and isinstance(bundle.get("metadata"), dict):
        metadata = bundle["metadata"]
        return {
            "ok": True,
            "service": "trade-local-ai-tools",
            "tools": ["profit", "timeseries", "sentiment", "exit", "train"],
            "trained_models_available": bool(bundle),
            "trained_at": metadata.get("trained_at"),
            "shadow_sample_count": metadata.get("shadow_sample_count", 0),
            "trade_sample_count": metadata.get("trade_sample_count", 0),
            "review_backend": "disabled_use_trading_app_online_model",
        }


@app.get("/models/status")
def local_models_status() -> dict[str, Any]:
    bundle = load_bundle()
    if not bundle:
        return {
            "available": False,
            "message": "No trained local quant bundle found; heuristic fallback is active.",
            "model_path": str(BUNDLE_PATH),
        }
    return {
        "available": True,
        "model_path": str(BUNDLE_PATH),
        **(bundle.get("metadata") or {}),
    }


@app.post("/train")
def train(req: TrainRequest) -> dict[str, Any]:
    rows = []
    for sample in req.shadow_samples or []:
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

    long_return_model = _make_regressor()
    short_return_model = _make_regressor()
    long_loss_model = _make_classifier(long_loss_y)
    short_loss_model = _make_classifier(short_loss_y)
    long_return_model.fit(X, long_y)
    short_return_model.fit(X, short_y)
    long_loss_model.fit(X, long_loss_y)
    short_loss_model.fit(X, short_loss_y)

    horizon_models: dict[int, dict[str, Any]] = {}
    for horizon in sorted({int(r["horizon"]) for r in rows}):
        h_rows = [r for r in rows if int(r["horizon"]) == horizon]
        if len(h_rows) < 80:
            continue
        hX = [r["x"] for r in h_rows]
        net_y = [max(r["long_return"], r["short_return"], key=abs) for r in h_rows]
        model = _make_regressor()
        model.fit(hX, net_y)
        horizon_models[horizon] = {"model": model, "samples": len(h_rows)}

    deep_sequence_model = _train_sequence_model(req.sequence_samples or [])
    torch_patch_model = _train_torch_patch_model(req.sequence_samples or [])

    sentiment_model = None
    sentiment_samples = []
    for sample in req.shadow_samples or []:
        features = sample.get("features") or {}
        if not features:
            continue
        sentiment_samples.append((
            [feature_row(features).get(key, 0.0) for key in SENTIMENT_KEYS],
            max(net_return_pct(f(sample, "long_return_pct")), net_return_pct(f(sample, "short_return_pct"))),
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
        sentiment_model.fit([x for x, _ in sentiment_samples], [y for _, y in sentiment_samples])
    text_sentiment_model = _train_text_sentiment_model(req.text_sentiment_samples or [])
    transformers_sentiment_backend = _probe_transformers_sentiment_backend()

    profiles = _train_profiles(req.trade_samples or [])
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": req.source,
        "shadow_sample_count": len(rows),
        "trade_sample_count": len(req.trade_samples or []),
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
            return {
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
                "profit_edge_pct": round(edge, 4),
                "profit_quality_score": round(quality, 4),
                "long_loss_probability": round(long_loss_prob, 4),
                "short_loss_probability": round(short_loss_prob, 4),
                "symbol_side_profile": {
                    "long": long_profile,
                    "short": short_profile,
                },
                "note": "Trained profit-first model: expected return and loss probability drive the score; win rate is not the objective.",
            }
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
    quality = max(best_expected, 0.0) + edge * 0.35 - risk_penalty * 0.5
    return {
        "available": True,
        "trained": False,
        "model": "local-profit-heuristic-v1",
        "symbol": req.symbol,
        "best_side": best_side,
        "long_expected_return_pct": round(long_expected, 4),
        "short_expected_return_pct": round(short_expected, 4),
        "expected_return_pct": round(best_expected, 4),
        "profit_edge_pct": round(edge, 4),
        "profit_quality_score": round(quality, 4),
        "risk_penalty": round(risk_penalty, 4),
        "fallback_error": fallback_error,
        "note": "Profit-first local signal; win rate is not used as the primary objective.",
    }


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
                return {
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
                }
        except Exception:
            pass

    returns = np.array([f(features, "returns_1"), f(features, "returns_5"), f(features, "returns_20")], dtype=float)
    weights = np.array([0.20, 0.35, 0.45], dtype=float)
    forecast = float(np.dot(returns, weights))
    vol = max(abs(f(features, "volatility_20")), 1e-6)
    confidence = clamp(abs(forecast) / (vol * 1.8 + 1e-6), 0.0, 1.0)
    direction = "up" if forecast > 0 else "down" if forecast < 0 else "flat"
    best_side = "long" if direction == "up" else "short" if direction == "down" else "hold"
    return {
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
    }


@app.post("/timeseries/deep/predict")
def deep_timeseries_predict(req: FeatureRequest) -> dict[str, Any]:
    """Sequence time-series service slot with PatchTST/TFT-style inputs."""
    features = req.features or {}
    bundle = load_bundle()
    torch_patch_model = (bundle or {}).get("torch_patch_model") or {}
    sequence_model = (bundle or {}).get("deep_sequence_model") or {}
    try:
        close_sequence = features.get("close_sequence") or features.get("recent_closes")
        volume_sequence = features.get("volume_sequence") or features.get("recent_volumes")
        if not close_sequence:
            close = f(features, "current_price", f(features, "close", 0.0))
            returns = [f(features, "returns_20"), f(features, "returns_5"), f(features, "returns_1")]
            close_sequence = [close * (1.0 - r / 100.0) for r in returns if close > 0] + [close]
        torch_expected = _predict_torch_patch_model(torch_patch_model, close_sequence, volume_sequence)
        if torch_expected is not None:
            confidence = clamp(abs(torch_expected) / 0.8, 0.0, 1.0)
            direction = "up" if torch_expected > 0 else "down" if torch_expected < 0 else "flat"
            best_side = "long" if direction == "up" else "short" if direction == "down" else "hold"
            return {
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
            }
        model = sequence_model.get("model")
        if model:
            expected = float(model.predict([sequence_features(close_sequence, volume_sequence)])[0])
            confidence = clamp(abs(expected) / 0.8, 0.0, 1.0)
            direction = "up" if expected > 0 else "down" if expected < 0 else "flat"
            best_side = "long" if direction == "up" else "short" if direction == "down" else "hold"
            return {
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
            }
    except Exception:
        pass
    base = timeseries_predict(req)
    base["endpoint"] = "timeseries_deep"
    base["model_family"] = "PatchTST/TFT-compatible"
    base["status"] = "trained_horizon_fallback" if base.get("trained") else "heuristic_fallback"
    base["note"] = "Sequence model unavailable for this request; using trained horizon ensemble."
    return base


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
    return {
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
    }


@app.post("/sentiment/deep/analyze")
def deep_sentiment_analyze(req: FeatureRequest) -> dict[str, Any]:
    """Independent text sentiment service slot for CryptoBERT/FinBERT style models."""
    base = sentiment_analyze(req)
    base["endpoint"] = "sentiment_deep"
    base["model_family"] = "CryptoBERT/FinBERT-style text model"
    base["status"] = "trained_text_model" if base.get("text_sentiment_score") is not None else ("trained_calibrator" if base.get("trained") else "feature_fallback")
    base["note"] = "Uses an independently trained local text sentiment model when headlines/text are supplied."
    return base


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
        return {
            "available": True,
            "trained": bool(bundle),
            "model": "local-exit-advisor-v1",
            "symbol": req.symbol,
            "action": "no_position",
            "reason": "本轮没有传入与该币种匹配的当前持仓，平仓建议模型不参与。",
        }
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
            action = "close_if_ai_agrees"
            urgency = 0.72
            reason = "亏损扩大到本地平仓模型容忍线之外，若 AI 也确认应优先退出。"
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
    return {
        "available": True,
        "trained": bool(bundle),
        "model": "local-exit-advisor-v1",
        "symbol": req.symbol,
        "action": top["action"],
        "urgency": top["urgency"],
        "reason": top["reason"],
        "advices": advices,
    }


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


def main() -> None:
    ssh = connect_remote_ssh(ROOT, timeout=15)
    try:
        run_remote_text(
            ssh,
            "mkdir -p /data/trade_ai/tools /data/trade_ai/logs /data/trade_ai/systemd && "
            "touch /data/trade_ai/local_ai_tools.env && chmod 600 /data/trade_ai/local_ai_tools.env",
            timeout=120,
        )
        sftp = ssh.open_sftp()
        with sftp.file("/data/trade_ai/tools/local_ai_tools_api.py", "w") as remote:
            remote.write(SERVICE_CODE)
        sftp.close()
        python_bin = "/home/linux/anaconda3/envs/trade_ml/bin/python"
        env_bin = "/home/linux/anaconda3/envs/trade_ml/bin"
        service = (
            textwrap.dedent(
                """
            [Unit]
            Description=Trade Local AI Tools API
            After=network-online.target qwen3-32b-main.service
            Wants=network-online.target

            [Service]
            User=linux
            WorkingDirectory=/data/trade_ai/tools
            Environment=PATH=__ENV_BIN__:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
            Environment=LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true
            Environment=LOCAL_AI_TOOLS_CORS_ORIGINS=http://127.0.0.1:8002,http://localhost:8002
            EnvironmentFile=-/data/trade_ai/local_ai_tools.env
            ExecStart=__PYTHON_BIN__ -m uvicorn local_ai_tools_api:app --host 0.0.0.0 --port 8001
            Restart=always
            RestartSec=5
            StandardOutput=append:/data/trade_ai/logs/local_ai_tools_api.log
            StandardError=append:/data/trade_ai/logs/local_ai_tools_api.err.log

            [Install]
            WantedBy=multi-user.target
            """
            )
            .strip()
            .replace("__ENV_BIN__", env_bin)
            .replace("__PYTHON_BIN__", python_bin)
            + "\n"
        )
        remote_service_path = "/data/trade_ai/systemd/local-ai-tools.service"
        with ssh.open_sftp().file(remote_service_path, "w") as remote:
            remote.write(service)
        run_remote_text(
            ssh,
            f"sudo install -m 0644 {remote_service_path} /etc/systemd/system/local-ai-tools.service && "
            "sudo systemctl daemon-reload && "
            "sudo systemctl enable local-ai-tools.service && "
            "sudo systemctl restart local-ai-tools.service",
            timeout=120,
        )
        safe_print(
            run_remote_text(
                ssh,
                "systemctl is-active local-ai-tools.service && "
                "sleep 2 && curl -s http://127.0.0.1:8001/health",
                timeout=120,
            )
        )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
