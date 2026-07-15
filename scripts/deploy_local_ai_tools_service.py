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

import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
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
ARTIFACT_REGISTRY_VERSION = "2026-07-15.local-ai-tools.v2"
ARTIFACT_ACTIVATION_MANIFEST_VERSION = "2026-07-15.local-ai-tools-activation.v2"
ARTIFACT_MODEL_ID = "local_ai_tools_quant_bundle"
VERSIONS_ROOT = MODEL_DIR / "versions"
CANDIDATE_POINTER_PATH = MODEL_DIR / "candidate.json"
CURRENT_POINTER_PATH = MODEL_DIR / "current.json"
ROLLBACK_POINTER_PATH = MODEL_DIR / "rollback.json"
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
RETURN_OBJECTIVE_NAME = "maximize_expected_realized_net_return_after_cost"
RETURN_OBJECTIVE_VERSION = "2026-07-14.separated-supervision.v2"
RETURN_LABEL_NAME = "separated_market_cost_and_realized_return_tasks"
RETURN_LABEL_VERSION = "2026-07-14.separated-supervision.v2"
COST_MODEL_VERSION = "okx_live_cost_and_authoritative_slippage_distribution_v2"
PROFIT_SUPERVISION_VERSION = "2026-07-14.separated-profit-supervision.v1"
RETURN_DISTRIBUTION_INPUT_VERSION = "2026-07-15.model-return-distribution-input.v1"
MARKET_OPPORTUNITY_TASK = "market_opportunity_distribution"
EXECUTION_COST_TASK = "execution_cost_and_slippage_distribution"
AUTHORITATIVE_REALIZED_RETURN_TASK = "authoritative_realized_return_distribution"
EVALUATION_REPORT_FIELDS = (
    "walk_forward_report",
    "leave_one_symbol_out_report",
    "oos_return_evaluation",
    "authoritative_trade_return_evidence",
)
COMPACT_SEQUENCE_SERIES_FORMAT = "compact_native_kline_series.v1"
TIMESERIES_MODEL_INPUT_ROWS = int(
    os.environ.get("LOCAL_AI_TOOLS_TIMESERIES_MODEL_INPUT_ROWS", "30")
)
TIMESERIES_PRIMARY_REPO_ID = os.environ.get(
    "LOCAL_AI_TOOLS_TIMESERIES_PRIMARY_MODEL",
    "google/timesfm-2.5-200m-pytorch",
).strip() or "google/timesfm-2.5-200m-pytorch"
TIMESERIES_LEGACY_TIMESFM_REPO_ID = "google/timesfm-2.5-200m-transformers"
TIMESERIES_CHRONOS_REPO_ID = "amazon/chronos-2"
TIMESERIES_FALLBACK_REPO_ID = "ibm-granite/granite-timeseries-ttm-r2"
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
_CURRENT_POINTER_MTIME_NS: int | None = None
_CURRENT_MODEL_MTIME_NS: int | None = None
_TRANSFORMER_MODEL_CACHE: dict[str, Any] = {}
_STATUS_METADATA_KEYS = (
    "artifact_policy_id",
    "phase",
    "trained_at",
    "source",
    "shadow_sample_count",
    "train_shadow_sample_count",
    "holdout_shadow_sample_count",
    "train_decision_group_count",
    "holdout_decision_group_count",
    "purged_holdout_decision_group_count",
    "completed_shadow_sample_count",
    "last_trained_completed_shadow_sample_count",
    "trade_sample_count",
    "completed_trade_sample_count",
    "last_trained_completed_trade_sample_count",
    "sequence_sample_count",
    "text_sentiment_sample_count",
    "torch_patch_available",
    "torch_patch_status",
    "transformers_sentiment_backend",
    "feature_count",
    "horizons",
    "profile_count",
    "training_data_sha256",
    "source_code_sha256",
    "objective_name",
    "objective_version",
    "label_name",
    "label_version",
    "cost_model_version",
    "training_cost_policy",
    "profit_supervision_version",
    "profit_supervision_report",
    "market_fact_contract",
    "quality_report",
    "governance_report",
    "return_objective_report",
    "training_policy",
    "trade_sample_cursor_policy",
    "training_mode",
    "requested_model_stage",
    "model_stage",
    "evaluation_policy",
    "artifact_persisted",
    "preflight_only",
    "persist_artifact_requested",
    "confirm_phase3_rebuild",
    "promotion_recommendation",
    "training_objective",
    "models",
    "objective",
    "artifact_registry_version",
    "artifact_model_id",
    "artifact_version",
    "artifact_lifecycle",
    "production_influence_authorized",
    "artifact_activation_manifest",
    "live_promotion_manifest",
    "walk_forward_report",
    "leave_one_symbol_out_report",
    "oos_return_evaluation",
    "authoritative_trade_return_evidence",
    "evaluation_report_hashes",
    "artifact_return_evidence_sha256",
)


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
    return_objective_report: dict[str, Any] = {}
    profit_supervision_report: dict[str, Any] = {}
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


def cost_complete_net_returns(
    sample: dict[str, Any],
    features: dict[str, Any],
    *,
    horizon_minutes: int,
    long_gross_return_pct: float,
    short_gross_return_pct: float,
) -> tuple[float, float, dict[str, float]] | None:
    spread_pct = f(features, "spread_pct", float("nan"))
    fee_pct = f(sample, "round_trip_fee_pct", f(features, "round_trip_fee_pct", float("nan")))
    funding_rate = f(features, "funding_rate", float("nan"))
    funding_interval_minutes = f(features, "funding_interval_minutes", float("nan"))
    if not math.isfinite(funding_interval_minutes):
        funding_interval_hours = f(features, "funding_interval_hours", float("nan"))
        if math.isfinite(funding_interval_hours):
            funding_interval_minutes = funding_interval_hours * 60.0
    if (
        not math.isfinite(spread_pct)
        or spread_pct <= 0
        or not math.isfinite(fee_pct)
        or fee_pct <= 0
        or not math.isfinite(funding_rate)
        or not math.isfinite(funding_interval_minutes)
        or funding_interval_minutes <= 0
    ):
        return None
    slippage_pct = spread_pct / 2.0
    funding_drag_pct = funding_rate * 100.0 * horizon_minutes / funding_interval_minutes
    return (
        long_gross_return_pct - fee_pct - slippage_pct - funding_drag_pct,
        short_gross_return_pct - fee_pct - slippage_pct + funding_drag_pct,
        {
            "spread_pct": spread_pct,
            "fee_pct": fee_pct,
            "slippage_pct": slippage_pct,
            "funding_drag_pct": funding_drag_pct,
        },
    )


def empirical_lower_hinge(values: list[float]) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return 0.0
    lower = ordered[: len(ordered) // 2 + (len(ordered) % 2)]
    middle = len(lower) // 2
    if len(lower) % 2:
        return lower[middle]
    return (lower[middle - 1] + lower[middle]) / 2.0


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _parsed_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_text(value: Any) -> str:
    parsed = _parsed_timestamp(value)
    return parsed.isoformat() if parsed is not None else ""


def _chronological_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(_parsed_timestamp(row.get("label_timestamp")) is None for row in rows):
        raise ValueError("label_timestamp is required for chronological evaluation")
    if any(not str(row.get("decision_group") or "").strip() for row in rows):
        raise ValueError("decision_group is required for chronological evaluation")
    if any(not str(row.get("symbol") or "").strip() for row in rows):
        raise ValueError("native symbol identity is required for evaluation")
    return sorted(
        rows,
        key=lambda row: (
            _parsed_timestamp(row.get("label_timestamp")),
            str(row.get("decision_group") or ""),
            int(row.get("id") or 0),
            int(row.get("horizon") or 0),
        ),
    )


def _decision_group_availability(
    rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, datetime]]]:
    bounds: dict[str, dict[str, datetime]] = {}
    for row in _chronological_rows(rows):
        group = str(row["decision_group"])
        timestamp = _parsed_timestamp(row["label_timestamp"])
        if timestamp is None:
            raise ValueError("label_timestamp is required for group availability")
        horizon_minutes = int(row.get("horizon") or 0)
        if horizon_minutes <= 0:
            raise ValueError("positive horizon is required for group availability")
        decision_timestamp = _parsed_timestamp(row.get("decision_timestamp"))
        if decision_timestamp is None:
            decision_timestamp = timestamp - timedelta(minutes=horizon_minutes)
        current = bounds.setdefault(
            group,
            {
                "start": timestamp,
                "end": timestamp,
                "decision_start": decision_timestamp,
                "decision_end": decision_timestamp,
            },
        )
        current["start"] = min(current["start"], timestamp)
        current["end"] = max(current["end"], timestamp)
        current["decision_start"] = min(
            current["decision_start"],
            decision_timestamp,
        )
        current["decision_end"] = max(
            current["decision_end"],
            decision_timestamp,
        )
    groups = sorted(
        bounds,
        key=lambda group: (
            bounds[group]["decision_start"],
            bounds[group]["decision_end"],
            group,
        ),
    )
    return groups, bounds


def _predict_positive_probabilities(model: Pipeline, x: list[list[float]]) -> np.ndarray:
    try:
        probabilities = np.asarray(model.predict_proba(x), dtype=float)
        named_steps = getattr(model, "named_steps", {})
        estimator = named_steps.get("model") if hasattr(named_steps, "get") else model
        classes = list(getattr(estimator, "classes_", []))
        if probabilities.ndim == 2 and 1 in classes:
            return probabilities[:, classes.index(1)]
    except Exception:
        pass
    return np.zeros(len(x), dtype=float)


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


def _profit_factor(returns: list[float]) -> float | None:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    gross_loss = abs(float(values[values < 0].sum()))
    if gross_loss <= np.finfo(float).eps:
        return None
    return float(values[values > 0].sum()) / gross_loss


def _return_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("label_timestamp") or ""),
            str(row.get("decision_group") or ""),
        ),
    )
    values = np.asarray(
        [float(row["return_pct"]) for row in ordered],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "avg_return_pct": None,
            "median_return_pct": None,
            "return_lcb_pct": None,
            "profit_factor": None,
            "cvar_10_pct": None,
            "max_drawdown_pct": None,
            "tail_loss_rate": None,
            "tail_loss_policy": {
                "source": "oos_negative_return_lower_hinge",
                "value": None,
                "observation_window": "current_oos_evidence_only",
            },
            "promotion_math_ready": False,
            "return_semantics": "net_return_after_counterfactual_execution_cost",
        }
    negatives = values[values < 0].tolist()
    tail_boundary = empirical_lower_hinge(negatives) if negatives else None
    mean = float(values.mean())
    if values.size == 1:
        return_lcb = mean
    else:
        standard_error = float(values.std(ddof=1)) / math.sqrt(values.size)
        return_lcb = mean - 1.645 * standard_error
    tail_cutoff = float(np.quantile(values, 0.10))
    cvar_values = values[values <= tail_cutoff]
    cvar_value = float(cvar_values.mean()) if cvar_values.size else tail_cutoff
    profit_factor_value = _profit_factor(values.tolist())
    max_drawdown = _max_drawdown(values.tolist())
    return {
        "count": int(values.size),
        "avg_return_pct": mean,
        "median_return_pct": float(np.median(values)),
        "return_lcb_pct": return_lcb,
        "profit_factor": profit_factor_value,
        "cvar_10_pct": cvar_value,
        "max_drawdown_pct": max_drawdown,
        "tail_loss_rate": (
            float((values < float(tail_boundary)).mean())
            if tail_boundary is not None
            else None
        ),
        "tail_loss_policy": {
            "source": "oos_negative_return_lower_hinge",
            "value": tail_boundary,
            "observation_window": "current_oos_evidence_only",
        },
        "promotion_math_ready": bool(
            return_lcb > 0.0
            and profit_factor_value is not None
            and profit_factor_value > 1.0
            and math.isfinite(cvar_value)
            and max_drawdown is not None
        ),
        "return_semantics": "net_return_after_counterfactual_execution_cost",
        "cost_deduction_count": 1,
    }


def _select_top_return_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    selected_count = max(int(math.sqrt(len(rows))), 1)
    return sorted(rows, key=lambda row: float(row["score"]))[-selected_count:]


def _leave_one_symbol_out_stability(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    symbols = sorted(
        {str(row.get("symbol") or "") for row in rows if row.get("symbol")}
    )
    reports = []
    for symbol in symbols:
        remaining = [
            row for row in rows if str(row.get("symbol") or "") != symbol
        ]
        reports.append(
            {
                "excluded_symbol": symbol,
                "remaining_symbol_count": len(
                    {
                        str(row.get("symbol") or "")
                        for row in remaining
                        if row.get("symbol")
                    }
                ),
                "evidence": _return_evidence(_select_top_return_rows(remaining)),
            }
        )
    return {
        "version": "2026-07-15.leave-one-symbol-out.v1",
        "evaluated_symbol_count": len(symbols),
        "rows": reports,
        "stable": bool(reports)
        and all(row["evidence"]["promotion_math_ready"] for row in reports),
        "policy": "recompute_oos_fee_after_return_evidence_after_each_symbol_removal",
    }


def _fit_walk_forward_side(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    side: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    net_key = f"{side}_net_return"
    return_key = f"{side}_return"
    cost_key = f"{side}_execution_cost"
    negatives = [float(row[net_key]) for row in train_rows if row[net_key] < 0]
    tail_boundary = empirical_lower_hinge(negatives) if negatives else 0.0
    tail_scale = abs(float(tail_boundary))
    x_train = [row["x"] for row in train_rows]
    x_validation = [row["x"] for row in validation_rows]
    weights = [float(row["sample_weight"]) for row in train_rows]
    market_model = _make_regressor(len(train_rows))
    cost_model = _make_regressor(len(train_rows))
    tail_labels = [int(row[net_key] < tail_boundary) for row in train_rows]
    tail_model = _make_classifier(tail_labels)
    market_model.fit(
        x_train,
        [row[return_key] for row in train_rows],
        model__sample_weight=weights,
    )
    cost_model.fit(
        x_train,
        [row[cost_key] for row in train_rows],
        model__sample_weight=weights,
    )
    tail_model.fit(x_train, tail_labels, model__sample_weight=weights)
    scores = (
        np.asarray(market_model.predict(x_validation), dtype=float)
        - np.asarray(cost_model.predict(x_validation), dtype=float)
        - _predict_positive_probabilities(tail_model, x_validation) * tail_scale
    )
    evaluated_rows = [
        {
            "symbol": str(row.get("symbol") or ""),
            "decision_group": str(row.get("decision_group") or ""),
            "label_timestamp": str(row.get("label_timestamp") or ""),
            "return_pct": float(row[net_key]),
            "gross_market_return_pct": float(row[return_key]),
            "execution_cost_pct": float(row[cost_key]),
            "score": float(scores[index]),
        }
        for index, row in enumerate(validation_rows)
    ]
    return evaluated_rows, {
        "source": "walk_forward_training_net_negative_return_lower_hinge",
        "value": tail_boundary if negatives else None,
        "scale_pct": tail_scale,
        "observation_window": "walk_forward_training_groups_only",
        "training_decision_group_count": len(
            {str(row["decision_group"]) for row in train_rows}
        ),
    }


def _walk_forward_return_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = _chronological_rows(rows)
    groups, group_bounds = _decision_group_availability(ordered)
    version = "2026-07-15.expanding-decision-group-walk-forward.v1"
    if len(groups) <= 1:
        return {
            "version": version,
            "status": "insufficient_chronological_decision_groups",
            "folds": [],
            "decision_group_disjoint": False,
            "model_refit_per_fold": True,
            "chronological": True,
        }
    validation_candidates = [
        group
        for group in groups
        if any(
            group_bounds[prior]["end"]
            < group_bounds[group]["decision_start"]
            for prior in groups
            if group_bounds[prior]["decision_start"]
            < group_bounds[group]["decision_start"]
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
    blocks = [
        [str(value) for value in block.tolist()]
        for block in np.array_split(
            np.asarray(validation_candidates, dtype=object),
            validation_fold_count,
        )
        if len(block)
    ]
    folds = []
    oos_rows: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for fold_number, validation_groups in enumerate(blocks, start=1):
        validation_decision_start = min(
            group_bounds[group]["decision_start"]
            for group in validation_groups
        )
        training_set = {
            group
            for group in groups
            if group_bounds[group]["end"] < validation_decision_start
        }
        training_label_end = max(
            group_bounds[group]["end"] for group in training_set
        )
        purged_training_groups = [
            group
            for group in groups
            if group_bounds[group]["decision_start"] < validation_decision_start
            and group not in training_set
        ]
        validation_set = set(validation_groups)
        if training_set & validation_set:
            raise ValueError("walk-forward decision groups overlap")
        train_rows = [
            row for row in ordered if str(row["decision_group"]) in training_set
        ]
        validation_rows = [
            row for row in ordered if str(row["decision_group"]) in validation_set
        ]
        side_reports = {}
        for side in ("long", "short"):
            evaluated, tail_policy = _fit_walk_forward_side(
                train_rows,
                validation_rows,
                side=side,
            )
            oos_rows[side].extend(evaluated)
            side_reports[side] = {
                **_return_evidence(_select_top_return_rows(evaluated)),
                "training_tail_loss_policy": tail_policy,
            }
        folds.append(
            {
                "fold": fold_number,
                "training_decision_group_count": len(training_set),
                "validation_decision_group_count": len(validation_set),
                "validation_start": validation_rows[0]["label_timestamp"],
                "validation_end": validation_rows[-1]["label_timestamp"],
                "training_label_end": training_label_end.isoformat(),
                "validation_decision_start": validation_decision_start.isoformat(),
                "label_timestamp_overlap_count": 0,
                "purged_training_decision_group_count": len(
                    purged_training_groups
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
        }
    return {
        "version": version,
        "status": "complete" if folds else "insufficient_chronological_decision_groups",
        "folds": folds,
        "fold_count": len(folds),
        "decision_group_count": len(groups),
        "decision_group_disjoint": all(
            fold["decision_group_overlap_count"] == 0 for fold in folds
        ),
        "chronological_label_disjoint": all(
            fold["label_timestamp_overlap_count"] == 0
            and fold["training_label_end"] < fold["validation_decision_start"]
            for fold in folds
        ),
        "model_refit_per_fold": True,
        "chronological": True,
        "sides": side_reports,
        "stable": bool(folds)
        and all(
            report["promotion_math_ready"]
            and report["leave_one_symbol_out"]["stable"]
            and all(
                fold["sides"][side]["promotion_math_ready"]
                for fold in folds
            )
            for side, report in side_reports.items()
        ),
    }


def _authoritative_trade_return_evidence(
    trade_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    side_rows: dict[str, list[dict[str, Any]]] = {"long": [], "short": []}
    for sample in trade_samples:
        supervision = sample.get("profit_supervision") or {}
        tasks = supervision.get("tasks") or {}
        realized = tasks.get(AUTHORITATIVE_REALIZED_RETURN_TASK) or {}
        side = str(realized.get("side") or sample.get("side") or "").lower()
        value = f(realized, "realized_net_return_pct", float("nan"))
        if (
            supervision.get("version") != PROFIT_SUPERVISION_VERSION
            or realized.get("eligible") is not True
            or side not in side_rows
            or not math.isfinite(value)
        ):
            continue
        side_rows[side].append(
            {
                "symbol": symbol_key(sample.get("symbol")),
                "decision_group": str(
                    sample.get("lifecycle_key")
                    or sample.get("position_id")
                    or sample.get("id")
                    or ""
                ),
                "label_timestamp": _timestamp_text(
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
                for key in (
                    "symbol",
                    "decision_group",
                    "label_timestamp",
                    "return_pct",
                )
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
        "data_fingerprint": canonical_sha256(fingerprint_payload),
    }


def _evaluation_report_hashes(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        field: canonical_sha256(metadata.get(field) or {})
        for field in EVALUATION_REPORT_FIELDS
    }


def _production_return_evidence_blockers(metadata: dict[str, Any]) -> list[str]:
    blockers = []
    if not _is_sha256(metadata.get("training_data_sha256")):
        blockers.append("training_data_fingerprint_invalid")
    if not _is_sha256(metadata.get("source_code_sha256")):
        blockers.append("source_code_fingerprint_invalid")
    if metadata.get("time_split_policy") != "chronological_disjoint_decision_groups":
        blockers.append("chronological_decision_group_policy_missing")
    governance = metadata.get("governance_report") or {}
    if (
        not str(governance.get("quality_fingerprint") or "")
        or governance.get("artifact_quality_fingerprint")
        != governance.get("quality_fingerprint")
        or governance.get("artifact_matches_quality") is not True
        or governance.get("requires_artifact_refresh") is True
    ):
        blockers.append("artifact_quality_fingerprint_mismatch")
    market_fact_contract = metadata.get("market_fact_contract") or {}
    market_fact_provenance = market_fact_contract.get("provenance") or {}
    market_fact_assertions = market_fact_contract.get("assertions") or {}
    try:
        market_fact_violation_count = int(
            market_fact_contract.get("violation_count")
        )
    except (TypeError, ValueError):
        market_fact_violation_count = None
    if (
        market_fact_contract.get("status") != "clean"
        or market_fact_violation_count != 0
        or not _is_sha256(market_fact_provenance.get("data_fingerprint"))
        or any(
            market_fact_assertions.get(name) is not True
            for name in (
                "native_instrument_identity_verified",
                "same_contract_price_path_verified",
                "executable_market_fact_verified",
            )
        )
    ):
        blockers.append("market_fact_contract_not_clean")
    expected_hashes = _evaluation_report_hashes(metadata)
    if metadata.get("evaluation_report_hashes") != expected_hashes:
        blockers.append("evaluation_report_hash_mismatch")
    if metadata.get("artifact_return_evidence_sha256") != canonical_sha256(
        expected_hashes
    ):
        blockers.append("artifact_return_evidence_hash_mismatch")
    walk_forward = metadata.get("walk_forward_report") or {}
    folds = list(walk_forward.get("folds") or [])
    if (
        walk_forward.get("status") != "complete"
        or walk_forward.get("decision_group_disjoint") is not True
        or walk_forward.get("chronological_label_disjoint") is not True
        or walk_forward.get("model_refit_per_fold") is not True
        or not folds
    ):
        blockers.append("walk_forward_evidence_incomplete")
    loso_report = metadata.get("leave_one_symbol_out_report") or {}
    oos_report = metadata.get("oos_return_evaluation") or {}
    authoritative_sides = (
        (metadata.get("authoritative_trade_return_evidence") or {}).get("sides")
        or {}
    )
    walk_sides = walk_forward.get("sides") or {}
    for side in ("long", "short"):
        if (walk_sides.get(side) or {}).get("promotion_math_ready") is not True:
            blockers.append(f"{side}_walk_forward_return_evidence_not_ready")
        if any(
            ((fold.get("sides") or {}).get(side) or {}).get(
                "promotion_math_ready"
            )
            is not True
            for fold in folds
        ):
            blockers.append(f"{side}_walk_forward_fold_not_ready")
        if (loso_report.get(side) or {}).get("stable") is not True:
            blockers.append(f"{side}_leave_one_symbol_out_not_stable")
        for scope, evidence in (
            ("oos", oos_report.get(side) or {}),
            ("authoritative", authoritative_sides.get(side) or {}),
        ):
            if evidence.get("profit_factor") is None:
                blockers.append(f"{side}_{scope}_profit_factor_undefined")
            if evidence.get("promotion_math_ready") is not True:
                blockers.append(f"{side}_{scope}_return_evidence_not_ready")
            if any(
                evidence.get(field) is None
                for field in (
                    "return_lcb_pct",
                    "cvar_10_pct",
                    "max_drawdown_pct",
                )
            ):
                blockers.append(f"{side}_{scope}_tail_evidence_incomplete")
    return list(dict.fromkeys(blockers))


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


def _dynamic_min_samples_leaf(sample_count: int) -> int:
    observed_count = max(int(sample_count or 0), 1)
    return max(int(math.log2(max(observed_count, 2))), 1)


def _make_regressor(sample_count: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=260,
            max_depth=12,
            min_samples_leaf=_dynamic_min_samples_leaf(sample_count),
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
            min_samples_leaf=_dynamic_min_samples_leaf(len(y)),
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


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Artifact registry JSON must be an object: {path}")
    return value


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _resolve_artifact_pointer(
    pointer_path: Path,
    *,
    role: str,
    deserialize_bundle: bool = False,
) -> dict[str, Any] | None:
    if not pointer_path.exists():
        return None
    pointer = read_json_object(pointer_path)
    if pointer.get("artifact_registry_version") != ARTIFACT_REGISTRY_VERSION:
        raise ValueError("Unsupported local AI artifact registry pointer version.")
    if pointer.get("pointer_role") != role:
        raise ValueError("Local AI artifact pointer role mismatch.")
    if pointer.get("model_id") != ARTIFACT_MODEL_ID:
        raise ValueError("Local AI artifact pointer model identity mismatch.")
    version = _required_text(pointer, "version")
    version_root = (VERSIONS_ROOT / version).resolve(strict=True)
    manifest_path = (MODEL_DIR / _required_text(pointer, "manifest_path")).resolve(
        strict=True
    )
    manifest_path.relative_to(version_root)
    if sha256_file(manifest_path) != _required_text(pointer, "manifest_sha256"):
        raise ValueError("Local AI artifact manifest hash verification failed.")
    manifest = read_json_object(manifest_path)
    if manifest.get("artifact_registry_version") != ARTIFACT_REGISTRY_VERSION:
        raise ValueError("Local AI artifact manifest registry version mismatch.")
    if manifest.get("artifact_model_id") != ARTIFACT_MODEL_ID:
        raise ValueError("Local AI artifact manifest model identity mismatch.")
    if manifest.get("artifact_version") != version:
        raise ValueError("Local AI artifact manifest version mismatch.")
    model_path = (version_root / _required_text(manifest, "model_relative_path")).resolve(
        strict=True
    )
    metadata_path = (
        version_root / _required_text(manifest, "metadata_relative_path")
    ).resolve(strict=True)
    model_path.relative_to(version_root)
    metadata_path.relative_to(version_root)
    artifact_hash = _required_text(pointer, "artifact_sha256")
    if artifact_hash != manifest.get("artifact_sha256") or sha256_file(model_path) != artifact_hash:
        raise ValueError("Local AI artifact model hash verification failed.")
    metadata_hash = _required_text(pointer, "metadata_sha256")
    if metadata_hash != manifest.get("metadata_sha256") or sha256_file(metadata_path) != metadata_hash:
        raise ValueError("Local AI artifact metadata hash verification failed.")
    metadata = read_json_object(metadata_path)
    expected_report_hashes = _evaluation_report_hashes(metadata)
    if metadata.get("evaluation_report_hashes") != expected_report_hashes:
        raise ValueError("Local AI artifact evaluation report hash mismatch.")
    if metadata.get("artifact_return_evidence_sha256") != canonical_sha256(
        expected_report_hashes
    ):
        raise ValueError("Local AI artifact return evidence hash mismatch.")
    for field, expected in (
        ("artifact_registry_version", ARTIFACT_REGISTRY_VERSION),
        ("artifact_model_id", ARTIFACT_MODEL_ID),
        ("artifact_version", version),
        ("artifact_sha256", artifact_hash),
    ):
        if metadata.get(field) != expected:
            raise ValueError(f"Local AI artifact metadata {field} mismatch.")
    for field in (
        "training_data_sha256",
        "source_code_sha256",
        "objective_name",
        "objective_version",
        "label_name",
        "label_version",
        "cost_model_version",
        "profit_supervision_version",
        "time_split_policy",
        "model_stage",
        "market_fact_contract",
        "governance_report",
        "evaluation_report_hashes",
        "artifact_return_evidence_sha256",
    ):
        if metadata.get(field) != manifest.get(field):
            raise ValueError(f"Local AI artifact metadata/manifest {field} mismatch.")
    activation = None
    if role in {"current", "rollback"}:
        activation_path = (
            MODEL_DIR / _required_text(pointer, "activation_manifest_path")
        ).resolve(strict=True)
        activation_path.relative_to(version_root)
        if sha256_file(activation_path) != _required_text(
            pointer, "activation_manifest_sha256"
        ):
            raise ValueError("Local AI activation manifest hash verification failed.")
        activation = read_json_object(activation_path)
        if (
            activation.get("activation_manifest_version")
            != ARTIFACT_ACTIVATION_MANIFEST_VERSION
            or activation.get("artifact_model_id") != ARTIFACT_MODEL_ID
            or activation.get("artifact_version") != version
            or activation.get("artifact_sha256") != artifact_hash
        ):
            raise ValueError("Local AI activation identity mismatch.")
        if activation.get("training_data_sha256") != manifest.get(
            "training_data_sha256"
        ):
            raise ValueError("Local AI activation training-data identity mismatch.")
        if activation.get("source_code_sha256") != manifest.get("source_code_sha256"):
            raise ValueError("Local AI activation source-code identity mismatch.")
        if activation.get("artifact_return_evidence_sha256") != metadata.get(
            "artifact_return_evidence_sha256"
        ):
            raise ValueError("Local AI activation return-evidence identity mismatch.")
        stage = activation.get("activation_stage")
        production_authorized = activation.get("production_influence_authorized") is True
        if stage == "shadow" and production_authorized:
            raise ValueError("Shadow local AI artifact has production authorization.")
        if stage in {"canary", "live"} and not production_authorized:
            raise ValueError("Production local AI activation is not authorized.")
        if stage in {"canary", "live"}:
            evidence_blockers = _production_return_evidence_blockers(metadata)
            if evidence_blockers:
                raise ValueError(
                    "Production local AI activation return evidence is not ready: "
                    + ",".join(evidence_blockers)
                )
            if activation.get("return_evidence_ready") is not True:
                raise ValueError("Production local AI activation evidence was not authorized.")
        if stage not in {"shadow", "canary", "live"}:
            raise ValueError("Local AI activation stage is invalid.")
    bundle = None
    if deserialize_bundle:
        bundle = load_trusted_joblib_bundle(model_path)
        embedded = bundle.get("metadata") if isinstance(bundle, dict) else None
        if not isinstance(embedded, dict):
            raise ValueError("Local AI artifact bundle metadata is missing.")
        for field in (
            "artifact_registry_version",
            "artifact_model_id",
            "artifact_version",
            "training_data_sha256",
            "source_code_sha256",
            "objective_version",
            "label_version",
            "profit_supervision_version",
        ):
            if embedded.get(field) != metadata.get(field):
                raise ValueError(f"Local AI artifact bundle {field} mismatch.")
    return {
        "role": role,
        "pointer": pointer,
        "version": version,
        "version_root": version_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "activation_manifest": activation,
        "bundle": bundle,
    }


def persist_candidate_bundle(
    bundle: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not _is_sha256(metadata.get("training_data_sha256")):
        raise ValueError("Local AI training_data_sha256 is invalid.")
    if not _is_sha256(metadata.get("source_code_sha256")):
        raise ValueError("Local AI source_code_sha256 is invalid.")
    for field in (
        "objective_name",
        "objective_version",
        "label_name",
        "label_version",
        "cost_model_version",
        "profit_supervision_version",
        "time_split_policy",
    ):
        _required_text(metadata, field)
    for field in EVALUATION_REPORT_FIELDS:
        if not isinstance(metadata.get(field), dict):
            raise ValueError(f"Local AI {field} is required for candidate persistence.")
    expected_report_hashes = _evaluation_report_hashes(metadata)
    if metadata.get("evaluation_report_hashes") != expected_report_hashes:
        raise ValueError("Local AI evaluation report hashes are invalid.")
    if metadata.get("artifact_return_evidence_sha256") != canonical_sha256(
        expected_report_hashes
    ):
        raise ValueError("Local AI combined return-evidence hash is invalid.")
    created_at = datetime.now(timezone.utc)
    version = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
    version_root = VERSIONS_ROOT / version
    version_root.mkdir(parents=True, exist_ok=False)
    model_path = version_root / "model.joblib"
    metadata_path = version_root / "model_metadata.json"
    manifest_path = version_root / "manifest.json"
    registry_metadata = {
        **metadata,
        "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
        "artifact_model_id": ARTIFACT_MODEL_ID,
        "artifact_version": version,
        "artifact_lifecycle": "candidate",
        "model_stage": "candidate",
        "production_influence_authorized": False,
    }
    persisted_bundle = dict(bundle)
    persisted_bundle["metadata"] = registry_metadata
    dump_trusted_joblib_bundle(persisted_bundle, model_path)
    artifact_hash = sha256_file(model_path)
    registry_metadata["artifact_sha256"] = artifact_hash
    registry_metadata["artifact_size_bytes"] = model_path.stat().st_size
    write_json_atomic(metadata_path, registry_metadata)
    metadata_hash = sha256_file(metadata_path)
    manifest = {
        **registry_metadata,
        "created_at": created_at.isoformat(),
        "artifact_sha256": artifact_hash,
        "metadata_sha256": metadata_hash,
        "model_relative_path": "model.joblib",
        "metadata_relative_path": "model_metadata.json",
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        CANDIDATE_POINTER_PATH,
        {
            "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
            "pointer_role": "candidate",
            "model_id": ARTIFACT_MODEL_ID,
            "version": version,
            "manifest_path": str(manifest_path.relative_to(MODEL_DIR)),
            "artifact_sha256": artifact_hash,
            "metadata_sha256": metadata_hash,
            "manifest_sha256": sha256_file(manifest_path),
            "updated_at": created_at.isoformat(),
        },
    )
    candidate = _resolve_artifact_pointer(
        CANDIDATE_POINTER_PATH,
        role="candidate",
        deserialize_bundle=True,
    )
    if candidate is None:
        raise ValueError("Local AI candidate artifact did not resolve after persistence.")
    return candidate


def activate_candidate_shadow(return_evidence: dict[str, Any]) -> dict[str, Any]:
    candidate = _resolve_artifact_pointer(
        CANDIDATE_POINTER_PATH,
        role="candidate",
        deserialize_bundle=False,
    )
    if candidate is None:
        raise ValueError("Local AI candidate artifact is not registered.")
    activation_path = candidate["version_root"] / "activation-shadow.json"
    evidence_blockers = _production_return_evidence_blockers(candidate["metadata"])
    activation = {
        "activation_manifest_version": ARTIFACT_ACTIVATION_MANIFEST_VERSION,
        "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
        "artifact_model_id": ARTIFACT_MODEL_ID,
        "artifact_version": candidate["version"],
        "artifact_sha256": candidate["manifest"]["artifact_sha256"],
        "training_data_sha256": candidate["manifest"].get("training_data_sha256"),
        "source_code_sha256": candidate["manifest"].get("source_code_sha256"),
        "artifact_return_evidence_sha256": candidate["metadata"].get(
            "artifact_return_evidence_sha256"
        ),
        "activation_stage": "shadow",
        "production_influence_authorized": False,
        "return_evidence_report": return_evidence,
        "return_evidence_ready": not evidence_blockers,
        "return_evidence_blockers": evidence_blockers,
        "activated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(activation_path, activation)
    if CURRENT_POINTER_PATH.exists():
        current = _resolve_artifact_pointer(CURRENT_POINTER_PATH, role="current")
        if current is None:
            raise ValueError("Local AI current pointer disappeared during activation.")
        write_json_atomic(
            ROLLBACK_POINTER_PATH,
            {
                **current["pointer"],
                "pointer_role": "rollback",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        _resolve_artifact_pointer(ROLLBACK_POINTER_PATH, role="rollback")
    write_json_atomic(
        CURRENT_POINTER_PATH,
        {
            **candidate["pointer"],
            "pointer_role": "current",
            "activation_manifest_path": str(activation_path.relative_to(MODEL_DIR)),
            "activation_manifest_sha256": sha256_file(activation_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    current = _resolve_artifact_pointer(
        CURRENT_POINTER_PATH,
        role="current",
        deserialize_bundle=True,
    )
    if current is None:
        raise ValueError("Local AI shadow activation did not produce a current artifact.")
    CANDIDATE_POINTER_PATH.unlink(missing_ok=True)
    return current


def rollback_current_artifact() -> dict[str, Any]:
    rollback = _resolve_artifact_pointer(ROLLBACK_POINTER_PATH, role="rollback")
    if rollback is None:
        raise ValueError("Local AI rollback artifact is not registered.")
    current_pointer = read_json_object(CURRENT_POINTER_PATH)
    write_json_atomic(
        CURRENT_POINTER_PATH,
        {
            **rollback["pointer"],
            "pointer_role": "current",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    restored = _resolve_artifact_pointer(CURRENT_POINTER_PATH, role="current")
    if restored is None:
        raise ValueError("Local AI rollback did not restore a current artifact.")
    write_json_atomic(
        ROLLBACK_POINTER_PATH,
        {
            **current_pointer,
            "pointer_role": "rollback",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    _resolve_artifact_pointer(ROLLBACK_POINTER_PATH, role="rollback")
    return restored


def load_bundle() -> dict[str, Any] | None:
    global _BUNDLE_CACHE, _CURRENT_POINTER_MTIME_NS, _CURRENT_MODEL_MTIME_NS
    model_mtime_ns: int | None = None
    pointer_mtime_ns = (
        CURRENT_POINTER_PATH.stat().st_mtime_ns
        if CURRENT_POINTER_PATH.exists()
        else None
    )
    try:
        current = _resolve_artifact_pointer(
            CURRENT_POINTER_PATH,
            role="current",
            deserialize_bundle=False,
        )
        if current is None:
            _BUNDLE_CACHE = None
            _CURRENT_POINTER_MTIME_NS = pointer_mtime_ns
            _CURRENT_MODEL_MTIME_NS = None
            return None
        model_mtime_ns = current["model_path"].stat().st_mtime_ns
        if (
            _CURRENT_POINTER_MTIME_NS == pointer_mtime_ns
            and _CURRENT_MODEL_MTIME_NS == model_mtime_ns
        ):
            return _BUNDLE_CACHE
        current = _resolve_artifact_pointer(
            CURRENT_POINTER_PATH,
            role="current",
            deserialize_bundle=True,
        )
        candidate = current["bundle"] if current else None
        metadata = candidate.get("metadata") if isinstance(candidate, dict) else None
        if not isinstance(metadata, dict) or (
            metadata.get("objective_name") != RETURN_OBJECTIVE_NAME
            or metadata.get("objective_version") != RETURN_OBJECTIVE_VERSION
            or metadata.get("label_version") != RETURN_LABEL_VERSION
            or metadata.get("profit_supervision_version") != PROFIT_SUPERVISION_VERSION
            or not all(
                key in candidate
                for key in (
                    "long_return_model",
                    "short_return_model",
                    "long_cost_model",
                    "short_cost_model",
                )
            )
        ):
            raise ValueError("local quant artifact separated supervision rejected")
        activation = current.get("activation_manifest") or {}
        runtime_metadata = {
            **metadata,
            "artifact_lifecycle": activation.get("activation_stage") or "unregistered",
            "model_stage": activation.get("activation_stage") or "unregistered",
            "production_influence_authorized": bool(
                activation.get("production_influence_authorized")
            ),
            "artifact_activation_manifest": activation,
        }
        candidate = {**candidate, "metadata": runtime_metadata}
        _BUNDLE_CACHE = candidate
        _CURRENT_POINTER_MTIME_NS = pointer_mtime_ns
        _CURRENT_MODEL_MTIME_NS = model_mtime_ns
        return _BUNDLE_CACHE
    except Exception:
        _BUNDLE_CACHE = None
        _CURRENT_POINTER_MTIME_NS = pointer_mtime_ns
        _CURRENT_MODEL_MTIME_NS = model_mtime_ns if model_mtime_ns is not None else -1
        return None


def _file_stat(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError as exc:
        return {
            "exists": False,
            "error": safe_error(exc),
        }
    return {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _read_metadata_file() -> dict[str, Any]:
    try:
        current = _resolve_artifact_pointer(CURRENT_POINTER_PATH, role="current")
        return current["metadata"] if current else {}
    except Exception:
        return {}


def _status_metadata() -> dict[str, Any]:
    metadata = _read_metadata_file()
    if not metadata and _BUNDLE_CACHE and isinstance(_BUNDLE_CACHE.get("metadata"), dict):
        metadata = _BUNDLE_CACHE["metadata"]
    return {
        key: metadata.get(key)
        for key in _STATUS_METADATA_KEYS
        if key in metadata
    }


def _model_artifact_status() -> dict[str, Any]:
    pointer_rows = {}
    resolved_rows = {}
    for role, path in (
        ("candidate", CANDIDATE_POINTER_PATH),
        ("current", CURRENT_POINTER_PATH),
        ("rollback", ROLLBACK_POINTER_PATH),
    ):
        try:
            resolved = _resolve_artifact_pointer(path, role=role)
            resolved_rows[role] = resolved
            pointer_rows[role] = {
                "available": resolved is not None,
                "version": resolved.get("version") if resolved else None,
                "error": None if resolved else f"{role}_artifact_not_registered",
            }
        except Exception as exc:
            resolved_rows[role] = None
            pointer_rows[role] = {
                "available": False,
                "version": None,
                "error": safe_error(exc),
            }
    current = resolved_rows["current"]
    metadata = current["metadata"] if current else {}
    activation = current["activation_manifest"] if current else {}
    model_bundle_available = current is not None and bool(metadata)
    activation_stage = str((activation or {}).get("activation_stage") or "unregistered")
    return {
        "available": model_bundle_available,
        "model_bundle_available": model_bundle_available,
        "trained_models_available": model_bundle_available,
        "status": activation_stage if model_bundle_available else "artifact_unavailable",
        "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
        "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
        "artifact_model_id": ARTIFACT_MODEL_ID,
        "artifact_version": current.get("version") if current else None,
        "artifact_lifecycle": activation_stage,
        "production_influence_authorized": bool(
            (activation or {}).get("production_influence_authorized")
        ),
        "model_path": str(current["model_path"]) if current else None,
        "metadata_path": str(current["metadata_path"]) if current else None,
        "bundle_file": _file_stat(current["model_path"]) if current else {"exists": False},
        "metadata_file": (
            _file_stat(current["metadata_path"]) if current else {"exists": False}
        ),
        "metadata_loaded": bool(metadata),
        "metadata_source": "verified_current_pointer" if metadata else "missing",
        "activation_manifest": activation or {},
        "artifact_pointers": pointer_rows,
        **metadata,
        "artifact_lifecycle": activation_stage,
        "model_stage": activation_stage if model_bundle_available else "candidate",
        "artifact_activation_manifest": activation or {},
        "production_influence_authorized": bool(
            (activation or {}).get("production_influence_authorized")
        ),
    }


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


def regression_prediction_distribution(model: Pipeline, x: list[list[float]]) -> dict[str, Any]:
    """Return a current tree-prediction distribution without a fixed cutoff."""

    expected = float(model.predict(x)[0])
    named_steps = getattr(model, "named_steps", {})
    estimator = named_steps.get("model") if hasattr(named_steps, "get") else None
    imputer = named_steps.get("imputer") if hasattr(named_steps, "get") else None
    trees = list(getattr(estimator, "estimators_", []) or [])
    if not trees or imputer is None:
        return {
            "expected": expected,
            "median": expected,
            "lower_bound": expected,
            "upper_bound": expected,
            "std": 0.0,
            "spread": 0.0,
            "sample_count": 0,
            "distribution_ready": False,
            "source_authority": "regressor_point_prediction_without_members",
        }
    transformed = imputer.transform(x)
    values = np.asarray([float(tree.predict(transformed)[0]) for tree in trees], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "expected": expected,
            "median": expected,
            "lower_bound": expected,
            "upper_bound": expected,
            "std": 0.0,
            "spread": 0.0,
            "sample_count": 0,
            "distribution_ready": False,
            "source_authority": "regressor_point_prediction_without_members",
        }
    ordered = np.sort(values)
    lower_tail_count = max(int(math.sqrt(ordered.size)), 1)
    spread = float(ordered[-1] - ordered[0])
    numerical_resolution = float(np.finfo(float).eps) * max(
        abs(float(ordered[0])),
        abs(float(ordered[-1])),
        1.0,
    )
    return {
        "expected": expected,
        "median": float(np.median(values)),
        "lower_bound": float(np.median(ordered[:lower_tail_count])),
        "upper_bound": float(np.median(ordered[-lower_tail_count:])),
        "std": float(np.std(values)),
        "spread": spread,
        "sample_count": int(values.size),
        "distribution_ready": spread > numerical_resolution,
        "source_authority": "extra_trees_empirical_distribution",
    }


def execution_cost_distribution_contract(
    distribution: dict[str, Any],
) -> dict[str, Any]:
    """Expose one stable counterfactual-cost contract to return composition."""

    return {
        "expected_pct": distribution.get("expected"),
        "upper_tail_pct": distribution.get("upper_bound"),
        "uncertainty_pct": distribution.get("std"),
        "distribution_member_count": distribution.get("sample_count"),
        "distribution_ready": distribution.get("distribution_ready") is True,
        "source_authority": "shadow_counterfactual_live_microstructure",
    }


def model_return_distribution_input(
    distribution: dict[str, Any],
    *,
    side: str,
    horizon_minutes: int,
    tail_loss_probability: Any,
    tail_loss_scale_pct: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    def finite_or_none(value: Any) -> float | None:
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        except (TypeError, ValueError):
            return None

    values = {
        "raw_expected_return_pct": finite_or_none(distribution.get("expected")),
        "median_return_pct": finite_or_none(distribution.get("median")),
        "lower_quantile_return_pct": finite_or_none(distribution.get("lower_bound")),
        "upper_quantile_return_pct": finite_or_none(distribution.get("upper_bound")),
        "dispersion_pct": finite_or_none(distribution.get("std")),
        "tail_loss_probability": finite_or_none(tail_loss_probability),
        "tail_loss_scale_pct": finite_or_none(tail_loss_scale_pct),
    }
    try:
        member_count = int(distribution.get("sample_count") or 0)
    except (TypeError, ValueError):
        member_count = 0
    blockers: list[str] = []
    if distribution.get("distribution_ready") is not True:
        blockers.append(
            "current_tree_prediction_distribution_degenerate"
            if member_count > 0
            else "current_tree_prediction_distribution_missing"
        )
    blockers.extend(
        f"{field}_missing"
        for field, value in values.items()
        if value is None
    )
    if side not in {"long", "short"}:
        blockers.append("distribution_side_invalid")
    if int(horizon_minutes or 0) <= 0:
        blockers.append("distribution_horizon_missing")
    if member_count <= 0:
        blockers.append("distribution_members_missing")
    expected = values["raw_expected_return_pct"]
    median = values["median_return_pct"]
    lower = values["lower_quantile_return_pct"]
    upper = values["upper_quantile_return_pct"]
    dispersion = values["dispersion_pct"]
    tail_probability = values["tail_loss_probability"]
    tail_scale = values["tail_loss_scale_pct"]
    if expected is not None and lower is not None and lower > expected:
        blockers.append("lower_quantile_above_raw_expected")
    if lower is not None and median is not None and lower > median:
        blockers.append("lower_quantile_above_median")
    if median is not None and upper is not None and median > upper:
        blockers.append("median_above_upper_quantile")
    if dispersion is not None and dispersion < 0:
        blockers.append("return_dispersion_negative")
    if tail_probability is not None and not 0.0 <= tail_probability <= 1.0:
        blockers.append("tail_loss_probability_out_of_bounds")
    if tail_scale is not None and tail_scale < 0:
        blockers.append("tail_loss_scale_negative")
    for field, expected_version in (
        ("objective_version", RETURN_OBJECTIVE_VERSION),
        ("label_version", RETURN_LABEL_VERSION),
        ("cost_model_version", COST_MODEL_VERSION),
        ("profit_supervision_version", PROFIT_SUPERVISION_VERSION),
    ):
        if str(metadata.get(field) or "") != expected_version:
            blockers.append(f"return_distribution_{field}_mismatch")
    if not str(distribution.get("source_authority") or "").strip():
        blockers.append("return_distribution_source_authority_missing")
    blockers = list(dict.fromkeys(blockers))
    return {
        "side": side,
        "horizon_minutes": int(horizon_minutes),
        **values,
        "distribution_member_count": member_count,
        "return_semantics": "gross_market_opportunity_before_execution",
        "source_authority": distribution.get("source_authority"),
        "objective_version": metadata.get("objective_version"),
        "label_version": metadata.get("label_version"),
        "cost_model_version": metadata.get("cost_model_version"),
        "profit_supervision_version": metadata.get("profit_supervision_version"),
        "production_eligible": not blockers,
        "blockers": blockers,
    }


def unavailable_return_distribution_inputs(
    *,
    horizon_minutes: int,
    source_authority: str,
) -> dict[str, dict[str, Any]]:
    missing_distribution = {
        "expected": None,
        "median": None,
        "lower_bound": None,
        "upper_bound": None,
        "std": None,
        "sample_count": 0,
        "source_authority": source_authority,
    }
    return {
        side: model_return_distribution_input(
            missing_distribution,
            side=side,
            horizon_minutes=horizon_minutes,
            tail_loss_probability=None,
            tail_loss_scale_pct=None,
            metadata={},
        )
        for side in ("long", "short")
    }


def symbol_key(symbol: str | None) -> str:
    value = str(symbol or "").upper().split(":")[0]
    if value.endswith("-SWAP"):
        value = value[:-5]
    if "/" not in value and "-" in value:
        parts = value.split("-")
        if len(parts) >= 2:
            value = f"{parts[0]}/{parts[1]}"
    return value


def _weighted_empirical_distribution(values: list[tuple[Any, Any]]) -> dict[str, Any]:
    pairs = []
    for raw_value, raw_weight in values:
        value = f({"value": raw_value}, "value", float("nan"))
        weight = max(f({"weight": raw_weight}, "weight", 0.0), 0.0)
        if math.isfinite(value) and weight > 0:
            pairs.append((value, weight))
    if not pairs:
        return {
            "count": 0,
            "effective_sample_size": 0.0,
            "expected": None,
            "median": None,
            "lower_hinge": None,
            "upper_hinge": None,
        }
    pairs.sort(key=lambda item: item[0])
    total = sum(weight for _value, weight in pairs)
    square_total = sum(weight * weight for _value, weight in pairs)

    def quantile(fraction: float) -> float:
        target = total * fraction
        cumulative = 0.0
        for value, weight in pairs:
            cumulative += weight
            if cumulative >= target:
                return value
        return pairs[-1][0]

    return {
        "count": len(pairs),
        "effective_sample_size": total * total / square_total if square_total > 0 else 0.0,
        "expected": sum(value * weight for value, weight in pairs) / total,
        "median": quantile(0.5),
        "lower_hinge": quantile(0.25),
        "upper_hinge": quantile(0.75),
    }


def _train_profiles(trade_samples: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in trade_samples:
        if bool(row.get("exclude_from_training")):
            continue
        supervision = row.get("profit_supervision") or {}
        if supervision.get("version") != PROFIT_SUPERVISION_VERSION:
            continue
        tasks = supervision.get("tasks") or {}
        realized = tasks.get(AUTHORITATIVE_REALIZED_RETURN_TASK) or {}
        if realized.get("eligible") is not True:
            continue
        symbol = symbol_key(row.get("symbol"))
        side = str(realized.get("side") or row.get("side") or "").lower()
        if not symbol or side not in {"long", "short"}:
            continue
        for key in (f"{symbol}|{side}", f"*|{side}"):
            buckets.setdefault(key, []).append(row)

    profiles: dict[str, Any] = {}
    for key, rows in buckets.items():
        side = key.rsplit("|", 1)[-1]

        def task_pairs(task_name: str, field: str) -> list[tuple[Any, Any]]:
            pairs = []
            for row in rows:
                tasks = (row.get("profit_supervision") or {}).get("tasks") or {}
                task = tasks.get(task_name) or {}
                if task.get("eligible") is True:
                    pairs.append((task.get(field), row.get("sample_weight", 1.0)))
            return pairs

        profiles[key] = {
            "source_authority": "okx_position_history",
            "symbol": key.rsplit("|", 1)[0],
            "side": side,
            "net_return_after_cost_pct": _weighted_empirical_distribution(
                task_pairs(AUTHORITATIVE_REALIZED_RETURN_TASK, "realized_net_return_pct")
            ),
            "execution_cost_pct": _weighted_empirical_distribution(
                task_pairs(EXECUTION_COST_TASK, "total_cost_pct")
            ),
            "slippage_pct": _weighted_empirical_distribution(
                task_pairs(EXECUTION_COST_TASK, "slippage_pct")
            ),
            "stop_loss_slippage_pct": _weighted_empirical_distribution(
                task_pairs(AUTHORITATIVE_REALIZED_RETURN_TASK, "stop_loss_slippage_pct")
            ),
            "hold_minutes": _weighted_empirical_distribution(
                task_pairs(AUTHORITATIVE_REALIZED_RETURN_TASK, "hold_minutes")
            ),
        }
    return profiles


def _profile_for_side(
    profiles: dict[str, Any],
    *,
    symbol: str,
    side: str,
) -> dict[str, Any]:
    exact = f"{symbol_key(symbol)}|{side}"
    global_key = f"*|{side}"
    if exact in profiles:
        return {**(profiles.get(exact) or {}), "profile_source": "symbol_side"}
    if global_key in profiles:
        return {**(profiles.get(global_key) or {}), "profile_source": "global_side"}
    return {
        "profile_source": "missing",
        "fallback_reason": "authoritative_trade_calibration_missing",
    }


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


def _compact_sequence_series(
    sample: dict[str, Any],
) -> tuple[list[float], list[float]] | None:
    if sample.get("sequence_format") != COMPACT_SEQUENCE_SERIES_FORMAT:
        return None
    closes = sample.get("close_sequence")
    volumes = sample.get("volume_sequence")
    if not isinstance(closes, list) or not isinstance(volumes, list):
        return None
    if len(closes) != len(volumes):
        return None
    parsed_closes: list[float] = []
    parsed_volumes: list[float] = []
    for raw_close, raw_volume in zip(closes, volumes):
        try:
            close = float(raw_close)
            volume = float(raw_volume)
        except Exception:
            return None
        if not math.isfinite(close) or close <= 0:
            return None
        if not math.isfinite(volume) or volume < 0:
            return None
        parsed_closes.append(close)
        parsed_volumes.append(volume)
    expected_count = max(len(parsed_closes) - 31, 0)
    if int(f(sample, "observation_count", -1.0)) != expected_count:
        return None
    if str(sample.get("label_name") or "") != "gross_market_move_pct":
        return None
    if not str(sample.get("label_version") or "").strip():
        return None
    return parsed_closes, parsed_volumes


def _iter_sequence_training_windows(
    samples: list[dict[str, Any]],
):
    """Expand compact native series lazily on the model server."""

    for sample in samples or []:
        if bool(sample.get("exclude_from_training")):
            continue
        compact = _compact_sequence_series(sample)
        if sample.get("sequence_format") == COMPACT_SEQUENCE_SERIES_FORMAT:
            if compact is None:
                continue
        else:
            yield sample
            continue
        closes, volumes = compact
        for idx in range(30, len(closes) - 1):
            start = max(0, idx - 59)
            current_close = closes[start : idx + 1]
            current_volume = volumes[start : idx + 1]
            current_price = current_close[-1]
            future_return = (closes[idx + 1] - current_price) / current_price * 100.0
            yield {
                "symbol": sample.get("symbol"),
                "timeframe": sample.get("timeframe"),
                "close_sequence": current_close,
                "volume_sequence": current_volume,
                "future_return_pct": future_return,
                "long_return_pct": future_return,
                "short_return_pct": -future_return,
            }


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
    for sample in _iter_sequence_training_windows(samples):
        x = sequence_features(sample.get("close_sequence"), sample.get("volume_sequence"))
        future_move = f(sample, "future_return_pct")
        long_return = f(sample, "long_return_pct", future_move)
        short_return = f(sample, "short_return_pct", -future_move)
        if not x:
            continue
        rows.append((x, long_return, short_return, sample.get("timeframe") or "unknown"))
    if len(rows) <= 1:
        return None
    long_model = _make_regressor(len(rows))
    short_model = _make_regressor(len(rows))
    long_model.fit([x for x, _, _, _ in rows], [y for _, y, _, _ in rows])
    short_model.fit([x for x, _, _, _ in rows], [y for _, _, y, _ in rows])
    timeframes: dict[str, int] = {}
    for _, _, _, timeframe in rows:
        timeframes[str(timeframe)] = timeframes.get(str(timeframe), 0) + 1
    return {
        "long_model": long_model,
        "short_model": short_model,
        "samples": len(rows),
        "timeframes": timeframes,
    }


def _train_torch_patch_model(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        import torch
        from torch import nn
    except Exception as exc:
        return {"available": False, "reason": f"torch_unavailable: {safe_error(exc, 120)}"}

    rows = []
    for sample in _iter_sequence_training_windows(samples):
        x = sequence_deep_features(sample.get("close_sequence"), sample.get("volume_sequence"))
        future_move = f(sample, "future_return_pct")
        long_return = f(sample, "long_return_pct", future_move)
        short_return = f(sample, "short_return_pct", -future_move)
        if x:
            rows.append((x, long_return, short_return))
    if len(rows) <= 1:
        return {"available": False, "reason": "sequence_distribution_unavailable", "samples": len(rows)}

    X = np.array([x for x, _, _ in rows], dtype=np.float32)
    y = np.array([[long_y, short_y] for _, long_y, short_y in rows], dtype=np.float32)
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
        nn.Linear(48, 2),
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


def _predict_torch_patch_model(
    model_info: dict[str, Any],
    close_sequence: Any,
    volume_sequence: Any | None = None,
) -> tuple[float, float] | None:
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
            nn.Linear(48, 2),
        )
        net.load_state_dict(model_info["state_dict"])
        net.eval()
        with torch.no_grad():
            prediction = net(torch.tensor(x, dtype=torch.float32))[0]
            return float(prediction[0].item()), float(prediction[1].item())
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
    if len(closes) < TIMESERIES_MODEL_INPUT_ROWS:
        return closes, "not_enough_real_close_sequence", source or "missing"
    return closes, "", source


def _rolling_forecast_quality(
    closes: list[float],
    forecast_price: float,
    horizon_step: int,
) -> dict[str, Any]:
    """Build a scale-aware forecast interval from the current rolling distribution."""

    prices = np.asarray(closes, dtype=float)
    if (
        prices.size < 2
        or not np.all(np.isfinite(prices))
        or np.any(prices <= 0)
        or not math.isfinite(forecast_price)
        or forecast_price <= 0
    ):
        return {
            "production_eligible": False,
            "anomalous": True,
            "reason": "invalid_forecast_price_scale",
            "threshold_source": "rolling_horizon_empirical_order_statistics",
            "sample_count": 0,
        }

    effective_horizon = min(max(int(horizon_step), 1), prices.size - 1)
    historical_returns = (
        (prices[effective_horizon:] - prices[:-effective_horizon])
        / prices[:-effective_horizon]
        * 100.0
    )
    historical_returns = historical_returns[np.isfinite(historical_returns)]
    if historical_returns.size == 0:
        return {
            "production_eligible": False,
            "anomalous": True,
            "reason": "rolling_horizon_distribution_unavailable",
            "threshold_source": "rolling_horizon_empirical_order_statistics",
            "sample_count": 0,
        }

    ordered = np.sort(historical_returns)
    tail_count = max(int(math.sqrt(ordered.size)), 1)
    lower_index = min(tail_count - 1, ordered.size - 1)
    upper_index = max(ordered.size - tail_count, lower_index)
    lower_bound = float(ordered[lower_index])
    upper_bound = float(ordered[upper_index])
    predicted_return = float((forecast_price - prices[-1]) / prices[-1] * 100.0)
    anomalous = predicted_return < lower_bound or predicted_return > upper_bound
    rank = int(np.searchsorted(ordered, predicted_return, side="right"))
    empirical_cdf = (rank + 0.5) / (ordered.size + 1.0)
    distribution_confidence = max(
        0.0,
        min(2.0 * min(empirical_cdf, 1.0 - empirical_cdf), 1.0),
    )
    return {
        "production_eligible": not anomalous,
        "anomalous": anomalous,
        "reason": (
            "outside_dynamic_rolling_forecast_interval"
            if anomalous
            else "within_dynamic_rolling_forecast_interval"
        ),
        "threshold_source": "rolling_horizon_empirical_order_statistics",
        "threshold_policy": "tail_count_is_square_root_of_current_rolling_sample_count",
        "sample_count": int(ordered.size),
        "effective_horizon_step": int(effective_horizon),
        "lower_return_bound_pct": round(lower_bound, 6),
        "upper_return_bound_pct": round(upper_bound, 6),
        "predicted_return_pct": round(predicted_return, 6),
        "distribution_confidence": round(distribution_confidence, 6),
    }


def _load_timesfm_model(model_dir: str):
    def loader():
        official_error = ""
        try:
            import timesfm

            model_ref = model_dir if Path(model_dir).exists() else TIMESERIES_PRIMARY_REPO_ID
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_ref)
            if hasattr(model, "compile"):
                model.compile(
                    timesfm.ForecastConfig(
                        max_context=1024,
                        max_horizon=256,
                        normalize_inputs=True,
                        use_continuous_quantile_head=True,
                        force_flip_invariance=True,
                        infer_is_positive=True,
                        fix_quantile_crossing=True,
                    )
                )
            return {"backend": "timesfm", "model": model}
        except Exception as exc:
            official_error = safe_error(exc, 160)

        from transformers import AutoModelForTimeSeriesPrediction

        model = AutoModelForTimeSeriesPrediction.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        model.eval()
        return {
            "backend": "transformers",
            "model": model,
            "official_backend_error": official_error,
        }

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


def _timesfm_model_dir() -> Path:
    candidates = [
        PHASE3_ROOT / "models" / "timeseries" / "google--timesfm-2.5-200m-pytorch",
        PHASE3_ROOT / "models" / "timeseries" / "google--timesfm-2.5-200m-transformers",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _timesfm_forecast_values(loaded_model: Any, closes: list[float], horizon_step: int) -> tuple[list[float], str, str]:
    backend = "transformers"
    model = loaded_model
    official_backend_error = ""
    if isinstance(loaded_model, dict):
        backend = str(loaded_model.get("backend") or backend)
        official_backend_error = str(loaded_model.get("official_backend_error") or "")
        model = loaded_model.get("model")

    if model is None:
        return [], backend, official_backend_error

    if backend == "timesfm" and hasattr(model, "forecast"):
        point_forecast, _quantile_forecast = model.forecast(
            horizon=max(horizon_step, 1),
            inputs=[np.asarray(closes, dtype=np.float32)],
        )
        predictions = _prediction_values(point_forecast)
        if predictions:
            return predictions, backend, official_backend_error

    import torch

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
    if predictions:
        return predictions, backend, official_backend_error
    error = official_backend_error or "; ".join(errors[-2:])
    return [], backend, error


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
            "model": "chronos-2-shadow-challenger",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": False,
            "reason": reason,
            "sequence_length": len(closes),
            "sequence_source": sequence_source,
            "model_input_rows": TIMESERIES_MODEL_INPUT_ROWS,
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
                "model": "chronos-2-shadow-challenger",
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
        prediction_quality = _rolling_forecast_quality(closes, forecast_price, horizon_step)
        confidence = float(prediction_quality.get("distribution_confidence") or 0.0)
        direction = "up" if expected_move_pct > 0 else "down" if expected_move_pct < 0 else "flat"
        return {
            "available": True,
            "kind": "timeseries",
            "model": "chronos-2-shadow-challenger",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": True,
            "sequence_length": len(closes),
            "sequence_source": sequence_source,
            "model_input_rows": TIMESERIES_MODEL_INPUT_ROWS,
            "horizon_step": horizon_step,
            "forecast_price": round(forecast_price, 8),
            "last_close": round(float(last_close), 8),
            "expected_move_pct": round(expected_move_pct, 6),
            "expected_return_pct": round(expected_move_pct, 6),
            "direction": direction,
            "best_side": "long" if direction == "up" else "short" if direction == "down" else "hold",
            "confidence": round(confidence, 6),
            "prediction_quality": prediction_quality,
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
            "model": "chronos-2-shadow-challenger",
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
            "model_input_rows": TIMESERIES_MODEL_INPUT_ROWS,
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        }
    try:
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
        model_dir = _timesfm_model_dir()
        loaded_model = _load_timesfm_model(model_dir.as_posix())
        predictions, backend, backend_error = _timesfm_forecast_values(
            loaded_model,
            closes,
            horizon_step,
        )
        if not predictions:
            return {
                "available": False,
                "kind": "timeseries",
                "model": "timesfm-2.5-primary",
                "primary_model": chain.get("primary_model"),
                "challenger_model": chain.get("challenger_model"),
                "artifacts_ready": bool(chain.get("artifacts_ready")),
                "actual_inference": False,
                "reason": "timesfm_empty_prediction" if not backend_error else backend_error,
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
        prediction_quality = _rolling_forecast_quality(closes, forecast_price, horizon_step)
        confidence = float(prediction_quality.get("distribution_confidence") or 0.0)
        direction = "up" if expected_move_pct > 0 else "down" if expected_move_pct < 0 else "flat"
        return {
            "available": True,
            "kind": "timeseries",
            "model": "timesfm-2.5-primary",
            "primary_model": chain.get("primary_model"),
            "challenger_model": chain.get("challenger_model"),
            "artifacts_ready": bool(chain.get("artifacts_ready")),
            "actual_inference": True,
            "sequence_length": len(closes),
            "sequence_source": sequence_source,
            "model_input_rows": TIMESERIES_MODEL_INPUT_ROWS,
            "horizon_step": horizon_step,
            "forecast_price": round(forecast_price, 8),
            "last_close": round(float(last_close), 8),
            "expected_move_pct": round(expected_move_pct, 6),
            "expected_return_pct": round(expected_move_pct, 6),
            "direction": direction,
            "best_side": "long" if direction == "up" else "short" if direction == "down" else "hold",
            "confidence": round(confidence, 6),
            "prediction_quality": prediction_quality,
            "realized_vol_pct": round(realized_vol_pct, 6),
            "prediction_count": len(predictions),
            "adapter": "timesfm_official_adapter"
            if backend == "timesfm"
            else "timesfm_transformers_adapter",
            "backend": backend,
            "model_dir": model_dir.as_posix(),
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        }
    except Exception as exc:
        return {
            "available": False,
            "kind": "timeseries",
            "model": "timesfm-2.5-primary",
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
    primary_shadow = _run_timesfm_shadow(features)
    challenger_shadow = _run_chronos2_shadow(features)
    active = bool(primary_shadow.get("available") or challenger_shadow.get("available"))
    specialist_shadow = primary_shadow if primary_shadow.get("available") else challenger_shadow
    payload["specialist_response_applied"] = False
    payload["specialist_applied_model"] = None
    chain = dict(chain)
    chain["actual_inference"] = active
    payload["specialist_primary_model"] = chain.get("primary_model")
    payload["specialist_challenger_model"] = chain.get("challenger_model")
    payload["specialist_artifacts_ready"] = bool(chain.get("artifacts_ready"))
    payload["specialist_inference_active"] = active
    payload["specialist_model_chain"] = chain
    payload["timesfm_shadow_expected_move_pct"] = primary_shadow.get("expected_move_pct")
    payload["timesfm_shadow_expected_return_pct"] = primary_shadow.get("expected_return_pct")
    payload["timesfm_shadow_side"] = primary_shadow.get("best_side")
    payload["timesfm_shadow_confidence"] = primary_shadow.get("confidence")
    payload["timesfm_shadow_horizon_step"] = primary_shadow.get("horizon_step")
    payload["chronos_shadow_expected_move_pct"] = challenger_shadow.get("expected_move_pct")
    payload["chronos_shadow_expected_return_pct"] = challenger_shadow.get("expected_return_pct")
    payload["chronos_shadow_side"] = challenger_shadow.get("best_side")
    payload["chronos_shadow_confidence"] = challenger_shadow.get("confidence")
    payload["chronos_shadow_horizon_step"] = challenger_shadow.get("horizon_step")
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
        "TimesFM is the primary time-series evidence model; Chronos/Granite remain comparison and fallback."
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
    if len(rows) <= 1:
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
        "horizon_minutes",
        "objective_name",
        "objective_version",
        "label_name",
        "label_version",
        "training_cost_policy",
        "artifact_persisted",
        "prediction_quality",
        "return_semantics",
        "return_distribution_input_version",
        "return_distribution_inputs",
        "fallback_reason",
        "best_side",
        "side",
        "action",
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
        "exit_advice": "exit_profile_observer_v2",
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
    payload.setdefault("production_permission", False)
    if payload.get("trained") is True:
        bundle = load_bundle()
        metadata = bundle.get("metadata") if isinstance(bundle, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        for key in (
            "objective_name",
            "objective_version",
            "label_name",
            "label_version",
            "training_cost_policy",
            "artifact_persisted",
            "artifact_lifecycle",
            "model_stage",
            "training_mode",
            "profit_supervision_version",
            "production_influence_authorized",
        ):
            if key in metadata:
                payload.setdefault(key, metadata.get(key))
        distribution_inputs = payload.get("return_distribution_inputs")
        distribution_inputs = (
            distribution_inputs if isinstance(distribution_inputs, dict) else {}
        )
        activation_stage = str(
            metadata.get("artifact_lifecycle")
            or metadata.get("model_stage")
            or "candidate"
        ).lower()
        live_authorized = bool(
            activation_stage == "live"
            and metadata.get("production_influence_authorized") is True
            and not _production_return_evidence_blockers(metadata)
        )
        if not live_authorized:
            for side in ("long", "short"):
                distribution_input = distribution_inputs.get(side)
                if not isinstance(distribution_input, dict):
                    continue
                blockers = list(distribution_input.get("blockers") or [])
                blockers.append("artifact_activation_not_production_authorized")
                distribution_input["blockers"] = list(dict.fromkeys(blockers))
                distribution_input["production_eligible"] = False
            payload["route_mode"] = f"{activation_stage}_observation"
            payload["live_mutation"] = False
            payload["production_permission"] = False
            payload["promotion_ready"] = False
        else:
            payload["route_mode"] = "live"
            payload["live_mutation"] = True
            payload["live_influence"] = True
            payload["production_permission"] = True
            payload["promotion_ready"] = True
        distribution_inputs_ready = all(
            isinstance(distribution_inputs.get(side), dict)
            and distribution_inputs[side].get("production_eligible") is True
            for side in ("long", "short")
        )
        contract_ready = bool(
            payload.get("objective_name") == RETURN_OBJECTIVE_NAME
            and payload.get("objective_version") == RETURN_OBJECTIVE_VERSION
            and payload.get("label_name") == RETURN_LABEL_NAME
            and payload.get("label_version") == RETURN_LABEL_VERSION
            and payload.get("training_cost_policy")
            == "separated_market_opportunity_and_execution_cost_tasks"
            and payload.get("profit_supervision_version")
            == PROFIT_SUPERVISION_VERSION
            and payload.get("return_semantics")
            == "gross_market_opportunity_before_execution"
            and payload.get("return_distribution_input_version")
            == RETURN_DISTRIBUTION_INPUT_VERSION
            and distribution_inputs_ready
            and payload.get("artifact_persisted") is True
        )
        prediction_quality = payload.get("prediction_quality")
        if not isinstance(prediction_quality, dict):
            prediction_quality = {
                "production_eligible": False,
                "anomalous": True,
                "reason": "current_prediction_distribution_missing",
            }
            payload["prediction_quality"] = prediction_quality
        if not contract_ready:
            prediction_quality["production_eligible"] = False
            prediction_quality["anomalous"] = True
            contract_blockers = list(
                dict.fromkeys(
                    [
                        *(prediction_quality.get("blockers") or []),
                        *(
                            ["artifact_activation_not_production_authorized"]
                            if not live_authorized
                            else []
                        ),
                        "runtime_return_artifact_contract_incomplete",
                    ]
                )
            )
            prediction_quality["blockers"] = contract_blockers
            prediction_quality["reason"] = contract_blockers[0]
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
            "repo_id": TIMESERIES_PRIMARY_REPO_ID,
            "purpose": "online_primary_time_series_forecast",
        },
        {
            "slot": "timeseries_challenger",
            "role": "challenger",
            "repo_id": TIMESERIES_CHRONOS_REPO_ID,
            "purpose": "shadow_challenger_time_series_forecast",
        },
        {
            "slot": "timeseries_fallback",
            "role": "fallback",
            "repo_id": TIMESERIES_FALLBACK_REPO_ID,
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
        "adapter": "timesfm_official_primary_adapter",
        "required_imports": ["torch", "transformers"],
        "optional_imports": ["timesfm"],
        "requires_walk_forward": True,
    },
    "timeseries_challenger": {
        "adapter": "chronos_2_transformers_adapter",
        "required_imports": ["torch", "transformers"],
        "optional_imports": ["chronos"],
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
    artifact_status = _model_artifact_status()
    payload = {
        "ok": True,
        "service": "phase3_quant_api",
        "root": PHASE3_ROOT.as_posix(),
        "server_role": "dedicated_cryptocurrency_quant_model_server",
        "storage_policy": "new model/cache/training/runtime/log data under /data/BB",
        "port": PHASE3_API_PORT,
        "policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "live_mutation": False,
        "live_trading_mutation": False,
        "route_mode": "shadow_observation",
        "tools": ["profit", "timeseries", "sentiment", "exit", "train"],
        "review_backend": "disabled_use_trading_app_online_model",
        "model_dir": MODEL_DIR.as_posix(),
        "status_endpoint_uses_metadata_only": True,
    }
    payload.update(artifact_status)
    payload.setdefault("trained_at", None)
    payload.setdefault("shadow_sample_count", 0)
    payload.setdefault("trade_sample_count", 0)
    payload.setdefault("completed_shadow_sample_count", 0)
    payload.setdefault("completed_trade_sample_count", 0)
    payload.setdefault("quality_report", {})
    payload.setdefault("governance_report", {})
    payload.setdefault("return_objective_report", {})
    payload.setdefault("profit_supervision_report", {})
    payload.update(_phase3_inventory_status())
    payload["specialist_model_chains"] = {
        "timeseries": _specialist_model_chain("timeseries"),
        "sentiment": _specialist_model_chain("sentiment"),
    }
    return payload


@app.get("/models/status")
def local_models_status() -> dict[str, Any]:
    artifact_status = _model_artifact_status()
    message = ""
    if not artifact_status.get("available"):
        if artifact_status.get("status") == "metadata_missing":
            message = "Trained bundle exists but metadata is missing; rebuild training artifacts."
        else:
            message = "No trained local quant bundle found; return inference is unavailable."
    return {
        **artifact_status,
        "message": message,
        "specialist_adapter_preflight": _specialist_adapter_preflight(),
        "status_endpoint_uses_metadata_only": True,
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
        if not features:
            continue
        supervision = sample.get("profit_supervision") or {}
        if supervision.get("version") != PROFIT_SUPERVISION_VERSION:
            continue
        tasks = supervision.get("tasks") or {}
        market_task = tasks.get(MARKET_OPPORTUNITY_TASK) or {}
        cost_task = tasks.get(EXECUTION_COST_TASK) or {}
        if market_task.get("eligible") is not True or cost_task.get("eligible") is not True:
            continue
        long_return = f(
            market_task,
            "long_gross_market_return_pct",
            float("nan"),
        )
        short_return = f(
            market_task,
            "short_gross_market_return_pct",
            float("nan"),
        )
        long_cost = f(cost_task, "long_total_cost_pct", float("nan"))
        short_cost = f(cost_task, "short_total_cost_pct", float("nan"))
        if not all(
            math.isfinite(value)
            for value in (long_return, short_return, long_cost, short_cost)
        ):
            continue
        sample_weight = max(0.0, f(sample, "sample_weight", 1.0))
        if sample_weight <= 0.0:
            continue
        correlation = sample.get("correlation_weight") or {}
        decision_group = str(correlation.get("correlation_group") or "").strip()
        if not decision_group:
            decision_identity = sample.get("decision_id") or sample.get("id")
            decision_group = (
                f"shadow_decision:{decision_identity}" if decision_identity else ""
            )
        rows.append({
            "x": model_x(features, horizon_minutes=horizon),
            "id": int(sample.get("id") or 0),
            "symbol": symbol_key(sample.get("symbol") or features.get("symbol")),
            "horizon": horizon,
            "decision_group": decision_group,
            "decision_timestamp": _timestamp_text(sample.get("decision_timestamp")),
            "label_timestamp": _timestamp_text(sample.get("label_timestamp")),
            "raw_long_return": long_return,
            "raw_short_return": short_return,
            "long_return": long_return,
            "short_return": short_return,
            "long_execution_cost": long_cost,
            "short_execution_cost": short_cost,
            "long_net_return": long_return - long_cost,
            "short_net_return": short_return - short_cost,
            "best_side": "long" if long_return >= short_return else "short",
            "execution_cost": cost_task,
            "features": features,
            "sample_weight": sample_weight,
        })
    if len(rows) <= 1:
        return {
            "trained": False,
            "reason": "separated_supervision_distribution_unavailable",
            "shadow_sample_count": len(rows),
            "message": "Need market-opportunity and execution-cost tasks from separate decision groups.",
        }

    try:
        rows = _chronological_rows(rows)
    except ValueError as exc:
        return {
            "trained": False,
            "reason": "chronological_training_identity_incomplete",
            "shadow_sample_count": len(rows),
            "message": str(exc),
        }
    ordered_groups, group_bounds = _decision_group_availability(rows)
    if len(ordered_groups) <= 1:
        return {
            "trained": False,
            "reason": "decision_group_holdout_unavailable",
            "shadow_sample_count": len(rows),
            "decision_group_count": len(ordered_groups),
        }
    split = len(ordered_groups) // 2
    holdout_candidates = ordered_groups[split:]
    holdout_decision_start = min(
        group_bounds[group]["decision_start"] for group in holdout_candidates
    )
    train_groups = {
        group
        for group in ordered_groups[:split]
        if group_bounds[group]["end"] < holdout_decision_start
    }
    holdout_groups = set(holdout_candidates)
    purged_holdout_group_count = split - len(train_groups)
    train_rows = [row for row in rows if str(row["decision_group"]) in train_groups]
    holdout_rows = [row for row in rows if str(row["decision_group"]) in holdout_groups]
    if not train_rows or not holdout_rows or train_groups & holdout_groups:
        return {
            "trained": False,
            "reason": "decision_group_holdout_unavailable",
            "shadow_sample_count": len(rows),
        }

    long_tail_boundary = empirical_lower_hinge(
        [row["long_net_return"] for row in train_rows if row["long_net_return"] < 0]
    )
    short_tail_boundary = empirical_lower_hinge(
        [row["short_net_return"] for row in train_rows if row["short_net_return"] < 0]
    )
    for row in rows:
        row["lossy_long"] = int(row["long_net_return"] < long_tail_boundary)
        row["lossy_short"] = int(row["short_net_return"] < short_tail_boundary)

    X = [r["x"] for r in train_rows]
    long_y = [r["long_return"] for r in train_rows]
    short_y = [r["short_return"] for r in train_rows]
    long_cost_y = [r["long_execution_cost"] for r in train_rows]
    short_cost_y = [r["short_execution_cost"] for r in train_rows]
    long_loss_y = [r["lossy_long"] for r in train_rows]
    short_loss_y = [r["lossy_short"] for r in train_rows]
    sample_weights = [
        max(0.0, float(r.get("sample_weight") or 0.0)) for r in train_rows
    ]

    long_return_model = _make_regressor(len(train_rows))
    short_return_model = _make_regressor(len(train_rows))
    long_cost_model = _make_regressor(len(train_rows))
    short_cost_model = _make_regressor(len(train_rows))
    long_loss_model = _make_classifier(long_loss_y)
    short_loss_model = _make_classifier(short_loss_y)
    long_return_model.fit(X, long_y, model__sample_weight=sample_weights)
    short_return_model.fit(X, short_y, model__sample_weight=sample_weights)
    long_cost_model.fit(X, long_cost_y, model__sample_weight=sample_weights)
    short_cost_model.fit(X, short_cost_y, model__sample_weight=sample_weights)
    long_loss_model.fit(X, long_loss_y, model__sample_weight=sample_weights)
    short_loss_model.fit(X, short_loss_y, model__sample_weight=sample_weights)

    horizon_models: dict[int, dict[str, Any]] = {}
    for horizon in sorted({int(r["horizon"]) for r in train_rows}):
        h_rows = [r for r in train_rows if int(r["horizon"]) == horizon]
        hX = [r["x"] for r in h_rows]
        long_horizon_y = [r["long_return"] for r in h_rows]
        short_horizon_y = [r["short_return"] for r in h_rows]
        h_weights = [max(0.0, float(r.get("sample_weight") or 0.0)) for r in h_rows]
        long_model = _make_regressor(len(h_rows))
        short_model = _make_regressor(len(h_rows))
        long_model.fit(hX, long_horizon_y, model__sample_weight=h_weights)
        short_model.fit(hX, short_horizon_y, model__sample_weight=h_weights)
        horizon_models[horizon] = {
            "long_model": long_model,
            "short_model": short_model,
            "samples": len(h_rows),
        }

    deep_sequence_model = _train_sequence_model(req.sequence_samples or [])
    torch_patch_model = _train_torch_patch_model(req.sequence_samples or [])

    sentiment_model = None
    sentiment_samples = []
    for row in train_rows:
        features = row["features"]
        sentiment_samples.append(
            (
                [feature_row(features).get(key, 0.0) for key in SENTIMENT_KEYS],
                row["long_return"],
                row["short_return"],
                row["sample_weight"],
            )
        )
    if len(sentiment_samples) > 1:
        sentiment_leaf_size = _dynamic_min_samples_leaf(len(sentiment_samples))

        def make_sentiment_regressor(random_state: int) -> Pipeline:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=180,
                    max_depth=8,
                    min_samples_leaf=sentiment_leaf_size,
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ])

        sentiment_model = {
            "long_model": make_sentiment_regressor(43),
            "short_model": make_sentiment_regressor(44),
        }
        sentiment_model["long_model"].fit(
            [x for x, _, _, _ in sentiment_samples],
            [long_y for _, long_y, _, _ in sentiment_samples],
            model__sample_weight=[weight for _, _, _, weight in sentiment_samples],
        )
        sentiment_model["short_model"].fit(
            [x for x, _, _, _ in sentiment_samples],
            [short_y for _, _, short_y, _ in sentiment_samples],
            model__sample_weight=[weight for _, _, _, weight in sentiment_samples],
        )
    text_sentiment_model = _train_text_sentiment_model(req.text_sentiment_samples or [])
    transformers_sentiment_backend = _probe_transformers_sentiment_backend()

    trainable_trade_samples = [
        sample for sample in (req.trade_samples or []) if not bool(sample.get("exclude_from_training"))
    ]
    walk_forward_report = _walk_forward_return_report(rows)
    leave_one_symbol_out_report = {
        side: (walk_forward_report.get("sides") or {}).get(side, {}).get(
            "leave_one_symbol_out",
            {},
        )
        for side in ("long", "short")
    }
    oos_return_evaluation = {
        side: {
            key: value
            for key, value in (walk_forward_report.get("sides") or {}).get(
                side,
                {},
            ).items()
            if key != "leave_one_symbol_out"
        }
        for side in ("long", "short")
    }
    authoritative_trade_return_evidence = _authoritative_trade_return_evidence(
        trainable_trade_samples
    )
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
    fingerprint_payload = {
        "shadow": [
            {
                "id": row.get("id"),
                "symbol": row.get("symbol"),
                "decision_group": row.get("decision_group"),
                "label_timestamp": row.get("label_timestamp"),
                "horizon": row.get("horizon"),
                "long_return_pct": row.get("long_return"),
                "short_return_pct": row.get("short_return"),
                "long_execution_cost_pct": row.get("long_execution_cost"),
                "short_execution_cost_pct": row.get("short_execution_cost"),
                "sample_weight": row.get("sample_weight"),
                "feature_vector": row.get("x"),
            }
            for row in rows
        ],
        "trades": sorted([
            {
                "id": sample.get("id"),
                "position_id": sample.get("position_id"),
                "realized_pnl": sample.get("realized_pnl"),
                "net_return_after_cost_pct": sample.get("net_return_after_cost_pct"),
                "sample_weight": sample.get("sample_weight"),
                "profit_supervision": sample.get("profit_supervision") or {},
            }
            for sample in trainable_trade_samples
        ], key=lambda row: (str(row.get("position_id") or ""), str(row.get("id") or ""))),
        "sequence": sorted([
            {
                "symbol": sample.get("symbol"),
                "timeframe": sample.get("timeframe"),
                "sequence_format": sample.get("sequence_format"),
                "observation_count": sample.get("observation_count"),
                "first_open_time": sample.get("first_open_time"),
                "last_open_time": sample.get("last_open_time"),
                "feature_timestamp": sample.get("feature_timestamp"),
                "label_timestamp": sample.get("label_timestamp"),
                "long_return_pct": sample.get("long_return_pct"),
                "short_return_pct": sample.get("short_return_pct"),
                "close_sequence": sample.get("close_sequence"),
                "volume_sequence": sample.get("volume_sequence"),
                "training_sample_fingerprint": (
                    sample.get("training_sample_contract") or {}
                ).get("sample_fingerprint"),
            }
            for sample in (req.sequence_samples or [])
            if not bool(sample.get("exclude_from_training"))
        ], key=lambda row: (
            str(row.get("label_timestamp") or ""),
            str(row.get("symbol") or ""),
            str(row.get("timeframe") or ""),
        )),
        "text_sentiment": sorted(
            [
                {
                    "id": sample.get("id"),
                    "label_timestamp": sample.get("label_timestamp"),
                    "text": _text_value(sample),
                    "sentiment_score": f(sample, "sentiment_score"),
                    "sample_weight": sample.get("sample_weight"),
                }
                for sample in (req.text_sentiment_samples or [])
                if not bool(sample.get("exclude_from_training"))
                and _text_value(sample)
            ],
            key=lambda row: (
                str(row.get("label_timestamp") or ""),
                str(row.get("id") or ""),
                str(row.get("text") or ""),
            ),
        ),
    }
    training_data_sha256 = canonical_sha256(fingerprint_payload)
    source_path = Path(str(globals().get("__file__") or ""))
    if source_path.is_file():
        source_code_sha256 = sha256_file(source_path)
    else:
        source_digest = hashlib.sha256()
        for function in (
            train,
            _walk_forward_return_report,
            _fit_walk_forward_side,
            _return_evidence,
            _leave_one_symbol_out_stability,
            _authoritative_trade_return_evidence,
            timeseries_predict,
            deep_timeseries_predict,
            sentiment_analyze,
            profit_predict,
        ):
            source_digest.update(function.__code__.co_code)
        source_code_sha256 = source_digest.hexdigest()
    metadata = {
        "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "phase": "phase3_model_factory",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": req.source,
        "shadow_sample_count": len(rows),
        "train_shadow_sample_count": len(train_rows),
        "holdout_shadow_sample_count": len(holdout_rows),
        "train_decision_group_count": len(train_groups),
        "holdout_decision_group_count": len(holdout_groups),
        "purged_holdout_decision_group_count": purged_holdout_group_count,
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
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "profit_supervision_report": req.profit_supervision_report or {},
        "market_fact_contract": (req.quality_report or {}).get(
            "market_fact_contract",
            {},
        ),
        "tail_loss_policy": {
            "long": {
                "source": "chronological_training_net_negative_return_lower_hinge",
                "value": long_tail_boundary,
                "observation_window": "chronological_training_groups_only",
            },
            "short": {
                "source": "chronological_training_net_negative_return_lower_hinge",
                "value": short_tail_boundary,
                "observation_window": "chronological_training_groups_only",
            },
        },
        "tail_loss_scale_pct": {
            "long": abs(float(long_tail_boundary)),
            "short": abs(float(short_tail_boundary)),
        },
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "training_data_sha256": training_data_sha256,
        "source_code_sha256": source_code_sha256,
        "time_split_policy": "chronological_disjoint_decision_groups",
        "walk_forward_report": walk_forward_report,
        "leave_one_symbol_out_report": leave_one_symbol_out_report,
        "oos_return_evaluation": oos_return_evaluation,
        "authoritative_trade_return_evidence": authoritative_trade_return_evidence,
        "quality_report": req.quality_report or {},
        "governance_report": req.governance_report or {},
        "return_objective_report": req.return_objective_report or {},
        "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
        "trade_sample_cursor_policy": PHASE3_REQUIRED_TRAINING_POLICY,
        "training_mode": str(req.training_mode or "shadow"),
        "requested_model_stage": str(req.model_stage or "shadow"),
        "model_stage": "candidate",
        "evaluation_policy": evaluation_policy,
        "artifact_persisted": bool(req.persist_artifact and req.confirm_phase3_rebuild),
        "preflight_only": not bool(req.persist_artifact and req.confirm_phase3_rebuild),
        "persist_artifact_requested": bool(req.persist_artifact),
        "confirm_phase3_rebuild": bool(req.confirm_phase3_rebuild),
        "promotion_recommendation": req.promotion_recommendation or {},
        "artifact_activation_manifest": {
            "status": "not_activated",
            "activation_stage": "candidate",
            "production_influence_authorized": False,
        },
        "live_promotion_manifest": {
            "status": "not_issued",
            "reason": "candidate_requires_independent_shadow_and_return_readiness",
            "production_influence_authorized": False,
        },
        "training_objective": (
            "Predict shadow market opportunity and counterfactual execution cost as "
            "separate tasks; calibrate realized net return and slippage only from "
            "authoritative OKX lifecycles."
        ),
        "models": {
            "profit": "ExtraTreesRegressor long/short gross market opportunity",
            "execution_cost": "ExtraTreesRegressor long/short counterfactual execution cost",
            "loss_filter": "ExtraTreesClassifier side-specific loss probability",
            "timeseries": "Per-horizon long/short ExtraTreesRegressor return distributions",
            "deep_timeseries": (
                "Torch PatchTST/TFT-style sequence model"
                if (torch_patch_model or {}).get("available")
                else ("Sequence ExtraTreesRegressor PatchTST/TFT-style input" if deep_sequence_model else "not enough kline sequences")
            ),
            "sentiment": "Side-specific RandomForest sentiment return calibration" if sentiment_model else "heuristic fallback",
            "deep_sentiment": (
                "Transformers-ready text sentiment + TF-IDF Ridge model"
                if (transformers_sentiment_backend or {}).get("available") and text_sentiment_model
                else ("TF-IDF Ridge text sentiment model" if text_sentiment_model else "not enough text samples")
            ),
            "exit": "trade-profile plus live pnl rules",
        },
        "objective": RETURN_OBJECTIVE_NAME,
    }
    metadata["evaluation_report_hashes"] = _evaluation_report_hashes(metadata)
    metadata["artifact_return_evidence_sha256"] = canonical_sha256(
        metadata["evaluation_report_hashes"]
    )
    return_evidence_blockers = _production_return_evidence_blockers(metadata)
    metadata["live_promotion_manifest"] = {
        "status": "not_issued",
        "reason": (
            "candidate_return_evidence_not_ready"
            if return_evidence_blockers
            else "candidate_requires_independent_shadow_activation"
        ),
        "blocking_reasons": return_evidence_blockers,
        "production_influence_authorized": False,
    }
    bundle = {
        "metadata": metadata,
        "feature_keys": FEATURE_KEYS,
        "long_return_model": long_return_model,
        "short_return_model": short_return_model,
        "long_cost_model": long_cost_model,
        "short_cost_model": short_cost_model,
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
    candidate = persist_candidate_bundle(bundle, metadata)
    current = activate_candidate_shadow(
        {
            "walk_forward_report": walk_forward_report,
            "leave_one_symbol_out_report": leave_one_symbol_out_report,
            "oos_return_evaluation": oos_return_evaluation,
            "authoritative_trade_return_evidence": authoritative_trade_return_evidence,
            "evaluation_report_hashes": metadata["evaluation_report_hashes"],
            "blocking_reasons": return_evidence_blockers,
            "promotion_recommendation": req.promotion_recommendation or {},
            "training_mode": str(req.training_mode or "shadow"),
        }
    )
    global _BUNDLE_CACHE, _CURRENT_POINTER_MTIME_NS, _CURRENT_MODEL_MTIME_NS
    _BUNDLE_CACHE = None
    _CURRENT_POINTER_MTIME_NS = None
    _CURRENT_MODEL_MTIME_NS = None
    loaded = load_bundle()
    if loaded is None:
        raise ValueError("Local AI shadow artifact failed post-activation load verification.")
    return {
        "trained": True,
        **current["metadata"],
        "artifact_version": current["version"],
        "artifact_activation_stage": "shadow",
        "production_influence_authorized": False,
        "candidate_version": candidate["version"],
    }


@app.post("/profit/predict")
def profit_predict(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    bundle = load_bundle()
    if bundle:
        try:
            x = [model_x(features)]
            metadata = bundle.get("metadata") or {}
            horizon_minutes = int(feature_row(features)["horizon_minutes"])
            long_distribution = regression_prediction_distribution(bundle["long_return_model"], x)
            short_distribution = regression_prediction_distribution(bundle["short_return_model"], x)
            long_cost_distribution = regression_prediction_distribution(
                bundle["long_cost_model"], x
            )
            short_cost_distribution = regression_prediction_distribution(
                bundle["short_cost_model"], x
            )
            long_expected = float(long_distribution["expected"])
            short_expected = float(short_distribution["expected"])
            long_loss_prob = predict_proba_positive(bundle["long_loss_model"], x)
            short_loss_prob = predict_proba_positive(bundle["short_loss_model"], x)
            tail_scales = metadata.get("tail_loss_scale_pct") or {}
            long_tail_scale = tail_scales.get("long")
            short_tail_scale = tail_scales.get("short")
            profiles = bundle.get("profiles") or {}
            profile_symbol = req.symbol or features.get("symbol") or ""
            long_profile = _profile_for_side(
                profiles,
                symbol=profile_symbol,
                side="long",
            )
            short_profile = _profile_for_side(
                profiles,
                symbol=profile_symbol,
                side="short",
            )
            long_lower_bound = float(long_distribution["lower_bound"])
            short_lower_bound = float(short_distribution["lower_bound"])
            long_return_input = model_return_distribution_input(
                long_distribution,
                side="long",
                horizon_minutes=horizon_minutes,
                tail_loss_probability=long_loss_prob,
                tail_loss_scale_pct=long_tail_scale,
                metadata=metadata,
            )
            short_return_input = model_return_distribution_input(
                short_distribution,
                side="short",
                horizon_minutes=horizon_minutes,
                tail_loss_probability=short_loss_prob,
                tail_loss_scale_pct=short_tail_scale,
                metadata=metadata,
            )
            return_input_blockers = list(
                dict.fromkeys(
                    [
                        *(long_return_input.get("blockers") or []),
                        *(short_return_input.get("blockers") or []),
                    ]
                )
            )
            best_side = "long" if long_expected >= short_expected else "short"
            best_expected = long_expected if best_side == "long" else short_expected
            edge = abs(long_expected - short_expected)
            loss_prob = long_loss_prob if best_side == "long" else short_loss_prob
            best_lower_bound = (
                long_lower_bound if best_side == "long" else short_lower_bound
            )
            quality = best_lower_bound
            actual_calibration_ready = all(
                int((profile.get("net_return_after_cost_pct") or {}).get("count") or 0) > 0
                and int((profile.get("slippage_pct") or {}).get("count") or 0) > 0
                for profile in (long_profile, short_profile)
            )
            prediction_blockers = list(return_input_blockers)
            if not long_cost_distribution["distribution_ready"] or not short_cost_distribution[
                "distribution_ready"
            ]:
                prediction_blockers.append(
                    "counterfactual_execution_cost_distribution_not_ready"
                )
            if not actual_calibration_ready:
                prediction_blockers.append("actual_trade_calibration_not_ready")
            if not all(
                math.isfinite(value)
                for value in (
                    long_expected,
                    short_expected,
                    long_lower_bound,
                    short_lower_bound,
                )
            ):
                prediction_blockers.append("return_distribution_values_not_finite")
            prediction_blockers = list(dict.fromkeys(prediction_blockers))
            prediction_ready = not prediction_blockers
            return _attach_baseline_only_shadow("profit_prediction", {
                "available": True,
                "trained": True,
                "model": "local-profit-trained-v2",
                "symbol": req.symbol,
                "best_side": best_side,
                "horizon_minutes": int(feature_row(features)["horizon_minutes"]),
                "profit_edge_pct": round(edge, 4),
                "profit_quality_score": round(quality, 4),
                "return_semantics": "gross_market_opportunity_before_execution",
                "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
                "return_distribution_input_version": (
                    RETURN_DISTRIBUTION_INPUT_VERSION
                ),
                "return_distribution_inputs": {
                    "long": long_return_input,
                    "short": short_return_input,
                },
                "counterfactual_execution_cost_distribution": {
                    "long": execution_cost_distribution_contract(
                        long_cost_distribution
                    ),
                    "short": execution_cost_distribution_contract(
                        short_cost_distribution
                    ),
                    "source_authority": "shadow_counterfactual_live_microstructure",
                },
                "actual_trade_calibration": {
                    "long": long_profile,
                    "short": short_profile,
                    "source_authority": "okx_position_history",
                },
                "long_loss_probability": round(long_loss_prob, 4),
                "short_loss_probability": round(short_loss_prob, 4),
                "loss_probability": round(loss_prob, 4),
                "prediction_quality": {
                    "production_eligible": prediction_ready,
                    "anomalous": not prediction_ready,
                    "reason": (
                        "separated_market_cost_and_actual_calibration_ready"
                        if prediction_ready
                        else prediction_blockers[0]
                    ),
                    "source": "current_extra_trees_prediction_distribution",
                    "blockers": prediction_blockers,
                    "long": long_distribution,
                    "short": short_distribution,
                },
                "symbol_side_profile": {
                    "long": long_profile,
                    "short": short_profile,
                },
                "note": "Shadow models predict gross market opportunity and counterfactual cost separately; only OKX lifecycles calibrate realized return and slippage.",
            }, kind="profit", features=features, fallback_reason="profit_specialist_pending_phase3_clean_rebuild")
        except Exception as exc:
            fallback_error = safe_error(exc)
    else:
        fallback_error = None

    fallback_reason = fallback_error or "trained_profit_model_unavailable"
    horizon_minutes = int(feature_row(features)["horizon_minutes"])
    return _attach_baseline_only_shadow("profit_prediction", {
        "available": False,
        "trained": False,
        "model": "local-profit-artifact-required-v3",
        "symbol": req.symbol,
        "best_side": "hold",
        "side": "hold",
        "horizon_minutes": horizon_minutes,
        "return_semantics": "gross_market_opportunity_before_execution",
        "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
        "return_distribution_inputs": unavailable_return_distribution_inputs(
            horizon_minutes=horizon_minutes,
            source_authority="artifact_unavailable",
        ),
        "prediction_quality": {
            "production_eligible": False,
            "anomalous": True,
            "reason": fallback_reason,
            "blockers": [fallback_reason],
        },
        "fallback_error": fallback_error,
        "note": "A persisted governed artifact is required before return inference.",
    }, kind="profit", features=features, fallback_reason=fallback_reason)


@app.post("/timeseries/predict")
def timeseries_predict(req: FeatureRequest) -> dict[str, Any]:
    features = req.features or {}
    bundle = load_bundle()
    if bundle:
        predictions = []
        try:
            metadata = bundle.get("metadata") or {}
            tail_scales = metadata.get("tail_loss_scale_pct") or {}
            profiles = bundle.get("profiles") or {}
            profile_symbol = req.symbol or features.get("symbol") or ""
            actual_calibration = {
                "long": _profile_for_side(
                    profiles,
                    symbol=profile_symbol,
                    side="long",
                ),
                "short": _profile_for_side(
                    profiles,
                    symbol=profile_symbol,
                    side="short",
                ),
                "source_authority": "okx_position_history",
            }
            for horizon, item in (bundle.get("horizon_models") or {}).items():
                x = [model_x(features, horizon_minutes=int(horizon))]
                long_distribution = regression_prediction_distribution(item["long_model"], x)
                short_distribution = regression_prediction_distribution(item["short_model"], x)
                long_cost_distribution = regression_prediction_distribution(
                    bundle["long_cost_model"],
                    x,
                )
                short_cost_distribution = regression_prediction_distribution(
                    bundle["short_cost_model"],
                    x,
                )
                long_return = float(long_distribution["expected"])
                short_return = float(short_distribution["expected"])
                long_loss_probability = predict_proba_positive(
                    bundle["long_loss_model"],
                    x,
                )
                short_loss_probability = predict_proba_positive(
                    bundle["short_loss_model"],
                    x,
                )
                best_side = "long" if long_return >= short_return else "short"
                return_distribution_inputs = {
                    "long": model_return_distribution_input(
                        long_distribution,
                        side="long",
                        horizon_minutes=int(horizon),
                        tail_loss_probability=long_loss_probability,
                        tail_loss_scale_pct=tail_scales.get("long"),
                        metadata=metadata,
                    ),
                    "short": model_return_distribution_input(
                        short_distribution,
                        side="short",
                        horizon_minutes=int(horizon),
                        tail_loss_probability=short_loss_probability,
                        tail_loss_scale_pct=tail_scales.get("short"),
                        metadata=metadata,
                    ),
                }
                return_input_blockers = list(
                    dict.fromkeys(
                        [
                            *(return_distribution_inputs["long"].get("blockers") or []),
                            *(return_distribution_inputs["short"].get("blockers") or []),
                        ]
                    )
                )
                return_inputs_ready = bool(
                    return_distribution_inputs["long"].get("production_eligible")
                    is True
                    and return_distribution_inputs["short"].get(
                        "production_eligible"
                    )
                    is True
                    and not return_input_blockers
                )
                predictions.append({
                    "horizon_minutes": int(horizon),
                    "prediction_sample_count": min(
                        int(long_distribution["sample_count"]),
                        int(short_distribution["sample_count"]),
                    ),
                    "prediction_distribution_ready": bool(
                        return_inputs_ready
                    ),
                    "prediction_distribution_blockers": return_input_blockers,
                    "return_distribution_input_version": (
                        RETURN_DISTRIBUTION_INPUT_VERSION
                    ),
                    "return_distribution_inputs": return_distribution_inputs,
                    "counterfactual_execution_cost_distribution": {
                        "long": execution_cost_distribution_contract(
                            long_cost_distribution
                        ),
                        "short": execution_cost_distribution_contract(
                            short_cost_distribution
                        ),
                        "source_authority": (
                            "shadow_counterfactual_live_microstructure"
                        ),
                    },
                    "actual_trade_calibration": actual_calibration,
                    "best_side": best_side,
                    "direction": "up" if best_side == "long" else "down",
                    "samples": int(item.get("samples") or 0),
                    "return_semantics": "gross_market_opportunity_before_execution",
                })
            eligible_predictions = [
                item
                for item in predictions
                if item.get("prediction_distribution_ready") is True
            ]
            if eligible_predictions:
                primary = max(
                    eligible_predictions,
                    key=lambda r: max(
                        float(
                            r["return_distribution_inputs"]["long"][
                                "lower_quantile_return_pct"
                            ]
                        ),
                        float(
                            r["return_distribution_inputs"]["short"][
                                "lower_quantile_return_pct"
                            ]
                        ),
                    ),
                )
                best_side = str(primary["best_side"])
                edge = abs(
                    float(
                        primary["return_distribution_inputs"]["long"][
                            "raw_expected_return_pct"
                        ]
                    )
                    - float(
                        primary["return_distribution_inputs"]["short"][
                            "raw_expected_return_pct"
                        ]
                    )
                )
                confidence = clamp(edge / 0.8, 0.0, 1.0)
                primary_cost_distribution = primary[
                    "counterfactual_execution_cost_distribution"
                ]
                actual_calibration_ready = all(
                    int(
                        (profile.get("net_return_after_cost_pct") or {}).get("count")
                        or 0
                    )
                    > 0
                    and int((profile.get("slippage_pct") or {}).get("count") or 0)
                    > 0
                    for profile in (
                        actual_calibration["long"],
                        actual_calibration["short"],
                    )
                )
                cost_distribution_ready = all(
                    (primary_cost_distribution.get(side) or {}).get(
                        "distribution_ready"
                    )
                    is True
                    for side in ("long", "short")
                )
                prediction_blockers = list(
                    primary.get("prediction_distribution_blockers") or []
                )
                if not cost_distribution_ready:
                    prediction_blockers.append(
                        "counterfactual_execution_cost_distribution_not_ready"
                    )
                if not actual_calibration_ready:
                    prediction_blockers.append("actual_trade_calibration_not_ready")
                prediction_blockers = list(dict.fromkeys(prediction_blockers))
                payload = with_model_metadata("time_series_prediction", {
                    "available": True,
                    "trained": True,
                    "model": "local-timeseries-trained-v2",
                    "architecture": "tree_horizon_ensemble",
                    "symbol": req.symbol,
                    "best_side": best_side,
                    "side": best_side,
                    "direction": primary["direction"],
                    "horizon_minutes": primary["horizon_minutes"],
                    "profit_edge_pct": round(edge, 4),
                    "return_semantics": "gross_market_opportunity_before_execution",
                    "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
                    "return_distribution_input_version": primary[
                        "return_distribution_input_version"
                    ],
                    "return_distribution_inputs": primary[
                        "return_distribution_inputs"
                    ],
                    "counterfactual_execution_cost_distribution": (
                        primary_cost_distribution
                    ),
                    "actual_trade_calibration": actual_calibration,
                    "confidence": round(confidence, 4),
                    "predictions": predictions,
                    "prediction_quality": {
                        "production_eligible": primary[
                            "prediction_distribution_ready"
                        ]
                        and cost_distribution_ready
                        and actual_calibration_ready
                        and not prediction_blockers,
                        "anomalous": not (
                            primary["prediction_distribution_ready"]
                            and cost_distribution_ready
                            and actual_calibration_ready
                            and not prediction_blockers
                        ),
                        "reason": (
                            "separated_market_cost_and_actual_calibration_ready"
                            if primary["prediction_distribution_ready"]
                            and cost_distribution_ready
                            and actual_calibration_ready
                            and not prediction_blockers
                            else prediction_blockers[0]
                            if prediction_blockers
                            else (
                                "current_tree_prediction_distribution_degenerate"
                                if primary["prediction_sample_count"] > 0
                                else "current_tree_prediction_distribution_missing"
                            )
                        ),
                        "source": "current_horizon_extra_trees_prediction_distribution",
                        "sample_count": primary["prediction_sample_count"],
                        "blockers": prediction_blockers,
                    },
                }, features=features)
                return _attach_timeseries_specialist_shadow(payload, features=features)
        except Exception:
            pass

    horizon_minutes = int(feature_row(features)["horizon_minutes"])
    return with_model_metadata("time_series_prediction", {
        "available": False,
        "trained": False,
        "model": "local-timeseries-artifact-required-v3",
        "architecture": "persisted_distribution_artifact_required",
        "symbol": req.symbol,
        "best_side": "hold",
        "side": "hold",
        "direction": "flat",
        "horizon_minutes": horizon_minutes,
        "return_semantics": "gross_market_opportunity_before_execution",
        "return_distribution_input_version": RETURN_DISTRIBUTION_INPUT_VERSION,
        "return_distribution_inputs": unavailable_return_distribution_inputs(
            horizon_minutes=horizon_minutes,
            source_authority="artifact_unavailable",
        ),
        "prediction_quality": {
            "production_eligible": False,
            "anomalous": True,
            "reason": "trained_timeseries_model_unavailable",
            "blockers": ["trained_timeseries_model_unavailable"],
        },
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
    metadata = (bundle or {}).get("metadata") or {}
    horizon_minutes = int(feature_row(features)["horizon_minutes"])
    try:
        torch_expected = (
            None
            if sequence_reason
            else _predict_torch_patch_model(torch_patch_model, close_sequence, volume_sequence)
        )
        if torch_expected is not None:
            long_expected, short_expected = torch_expected
            best_side = "long" if long_expected >= short_expected else "short"
            edge = abs(long_expected - short_expected)
            confidence = clamp(edge / 0.8, 0.0, 1.0)
            direction = "up" if best_side == "long" else "down"
            point_distributions = {
                "long": {
                    "expected": long_expected,
                    "median": long_expected,
                    "lower_bound": None,
                    "upper_bound": None,
                    "std": None,
                    "sample_count": 0,
                    "source_authority": "torch_point_prediction_without_distribution",
                },
                "short": {
                    "expected": short_expected,
                    "median": short_expected,
                    "lower_bound": None,
                    "upper_bound": None,
                    "std": None,
                    "sample_count": 0,
                    "source_authority": "torch_point_prediction_without_distribution",
                },
            }
            return_distribution_inputs = {
                side: model_return_distribution_input(
                    point_distributions[side],
                    side=side,
                    horizon_minutes=horizon_minutes,
                    tail_loss_probability=None,
                    tail_loss_scale_pct=None,
                    metadata=metadata,
                )
                for side in ("long", "short")
            }
            prediction_blockers = list(
                dict.fromkeys(
                    [
                        *(return_distribution_inputs["long"].get("blockers") or []),
                        *(return_distribution_inputs["short"].get("blockers") or []),
                    ]
                )
            )
            return _attach_timeseries_specialist_shadow({
                "available": True,
                "trained": True,
                "model": "local-torch-patch-timeseries-v1",
                "architecture": "torch_patch_mlp_tft_patchtst_style",
                "symbol": req.symbol,
                "best_side": best_side,
                "side": best_side,
                "direction": direction,
                "horizon_minutes": horizon_minutes,
                "return_semantics": "gross_market_opportunity_before_execution",
                "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
                "return_distribution_input_version": (
                    RETURN_DISTRIBUTION_INPUT_VERSION
                ),
                "return_distribution_inputs": return_distribution_inputs,
                "profit_edge_pct": round(edge, 4),
                "confidence": round(confidence, 4),
                "sample_count": int(torch_patch_model.get("samples") or 0),
                "train_mae_pct": torch_patch_model.get("train_mae_pct"),
                "endpoint": "timeseries_deep",
                "model_family": "PatchTST/TFT-style torch sequence model",
                "status": "trained_torch_sequence_model",
                "sequence_length": len(close_sequence),
                "sequence_source": sequence_source,
                "prediction_quality": {
                    "production_eligible": False,
                    "anomalous": True,
                    "reason": prediction_blockers[0],
                    "blockers": prediction_blockers,
                },
            }, features=features)
        long_model = sequence_model.get("long_model")
        short_model = sequence_model.get("short_model")
        if long_model and short_model and not sequence_reason:
            x = [sequence_features(close_sequence, volume_sequence)]
            long_distribution = regression_prediction_distribution(long_model, x)
            short_distribution = regression_prediction_distribution(short_model, x)
            long_expected = float(long_distribution["expected"])
            short_expected = float(short_distribution["expected"])
            best_side = "long" if long_expected >= short_expected else "short"
            edge = abs(long_expected - short_expected)
            confidence = clamp(edge / 0.8, 0.0, 1.0)
            direction = "up" if best_side == "long" else "down"
            tail_scales = metadata.get("tail_loss_scale_pct") or {}
            return_distribution_inputs = {
                "long": model_return_distribution_input(
                    long_distribution,
                    side="long",
                    horizon_minutes=horizon_minutes,
                    tail_loss_probability=None,
                    tail_loss_scale_pct=tail_scales.get("long"),
                    metadata=metadata,
                ),
                "short": model_return_distribution_input(
                    short_distribution,
                    side="short",
                    horizon_minutes=horizon_minutes,
                    tail_loss_probability=None,
                    tail_loss_scale_pct=tail_scales.get("short"),
                    metadata=metadata,
                ),
            }
            prediction_blockers = list(
                dict.fromkeys(
                    [
                        *(return_distribution_inputs["long"].get("blockers") or []),
                        *(return_distribution_inputs["short"].get("blockers") or []),
                    ]
                )
            )
            return _attach_timeseries_specialist_shadow({
                "available": True,
                "trained": True,
                "model": "local-sequence-timeseries-v1",
                "architecture": "sequence_extra_trees_patchtst_tft_style",
                "symbol": req.symbol,
                "best_side": best_side,
                "side": best_side,
                "direction": direction,
                "horizon_minutes": horizon_minutes,
                "return_semantics": "gross_market_opportunity_before_execution",
                "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
                "return_distribution_input_version": (
                    RETURN_DISTRIBUTION_INPUT_VERSION
                ),
                "return_distribution_inputs": return_distribution_inputs,
                "profit_edge_pct": round(edge, 4),
                "confidence": round(confidence, 4),
                "sample_count": int(sequence_model.get("samples") or 0),
                "timeframes": sequence_model.get("timeframes") or {},
                "endpoint": "timeseries_deep",
                "model_family": "PatchTST/TFT-style sequence model",
                "status": "trained_sequence_model",
                "sequence_length": len(close_sequence),
                "sequence_source": sequence_source,
                "prediction_quality": {
                    "production_eligible": False,
                    "anomalous": True,
                    "reason": prediction_blockers[0],
                    "blockers": prediction_blockers,
                },
            }, features=features)
    except Exception:
        pass
    base = timeseries_predict(req)
    base.update(
        {
            "endpoint": "timeseries_deep",
            "model_family": "TimesFM primary with Chronos/Granite comparison time-series chain",
            "status": (
                "trained_horizon_fallback" if base.get("trained") else "heuristic_fallback"
            ),
            "note": (
                "TimesFM is evaluated as the primary specialist time-series evidence source; "
                "Chronos and Granite remain comparison/fallback models."
            ),
            "sequence_input_status": sequence_reason or "real_sequence_ready",
            "sequence_length": len(close_sequence),
            "sequence_source": sequence_source,
            "model_input_rows": TIMESERIES_MODEL_INPUT_ROWS,
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
    trained_long_expected = None
    trained_short_expected = None
    bundle = load_bundle()
    try:
        sentiment_model = bundle.get("sentiment_model") if bundle else None
        if isinstance(sentiment_model, dict):
            x = [[feature_row(features).get(key, 0.0) for key in SENTIMENT_KEYS]]
            trained_long_expected = float(sentiment_model["long_model"].predict(x)[0])
            trained_short_expected = float(sentiment_model["short_model"].predict(x)[0])
            trained_expected = max(trained_long_expected, trained_short_expected)
            return_edge = trained_long_expected - trained_short_expected
            score = score * 0.35 + clamp(return_edge / 1.5, -1.0, 1.0) * 0.65
    except Exception:
        trained_expected = None
        trained_long_expected = None
        trained_short_expected = None
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
        "return_calibration_observation_only": bool(trained_expected is not None),
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
    observations = []
    for pos in positions:
        side = str(pos.get("side") or "").lower()
        pnl_pct = f(pos, "unrealized_pnl_pct", f(pos, "pnl_pct"))
        unrealized = f(pos, "unrealized_pnl")
        hold = f(pos, "hold_minutes")
        profile = profiles.get(f"{symbol}|{side}", {})
        observations.append({
            "side": side,
            "unrealized_pnl": round(unrealized, 4),
            "pnl_pct": round(pnl_pct, 5),
            "hold_minutes": round(hold, 2),
            "profile": profile,
            "production_permission": False,
        })
    return with_model_metadata("exit_advice", {
        "available": True,
        "trained": bool(bundle),
        "model": "local-exit-profile-observer-v2",
        "symbol": req.symbol,
        "action": "hold",
        "reason": "本地退出画像只提供观察事实；生产平仓由动态退出契约独占。",
        "observations": observations,
        "live_mutation": False,
        "production_permission": False,
    }, features=features, fallback_reason="dynamic_exit_policy_owns_production_exit")


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
            "    sudo systemctl stop \"${unit}\" || true",
            "    sudo systemctl disable \"${unit}\" || true",
            "  fi",
            "  if kill -0 \"${pid}\" 2>/dev/null; then",
            "    sudo kill \"${pid}\" || true",
            "    sleep 2",
            "  fi",
            "  if kill -0 \"${pid}\" 2>/dev/null; then",
            "    sudo kill -9 \"${pid}\" || true",
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
        "def read_json(response):\n"
        "    payload = response.read(4 * 1024 * 1024 + 1)\n"
        "    if len(payload) > 4 * 1024 * 1024:\n"
        "        raise RuntimeError('phase3_quant_api_response_exceeds_4mb')\n"
        "    return json.loads(payload.decode('utf-8'))\n"
        "\n"
        "def get(path):\n"
        "    headers = {}\n"
        "    key = api_key()\n"
        "    if key:\n"
        "        headers['Authorization'] = 'Bearer ' + key\n"
        "    request = urllib.request.Request(BASE + path, headers=headers)\n"
        "    with urllib.request.urlopen(request, timeout=8) as response:\n"
        "        return read_json(response)\n"
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
        "        return read_json(response)\n"
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
        "assert health.get('artifact_lifecycle') == 'shadow', health\n"
        "assert health.get('production_influence_authorized') is False, health\n"
        "assert health.get('artifact_activation_manifest', {}).get('activation_stage') == 'shadow', health\n"
        "assert health.get('artifact_activation_manifest', {}).get('production_influence_authorized') is False, health\n"
        "assert profit.get('trained') is True, profit\n"
        "assert profit.get('shadow_payload', {}).get('tool') == 'profit_prediction', profit\n"
        "assert profit.get('live_mutation') is False, profit\n"
        "assert profit.get('production_permission') is False, profit\n"
        "assert profit.get('promotion_ready') is False, profit\n"
        "assert profit.get('prediction_quality', {}).get('production_eligible') is False, profit\n"
        "assert profit.get('return_distribution_input_version') == '2026-07-15.model-return-distribution-input.v1', profit\n"
        "assert set((profit.get('return_distribution_inputs') or {})) == {'long', 'short'}, profit\n"
        "assert all(item.get('production_eligible') is False for item in (profit.get('return_distribution_inputs') or {}).values()), profit\n"
        "assert 'loss_probability' in profit, profit\n"
        "assert exit_advice.get('action') == 'hold', exit_advice\n"
        "assert exit_advice.get('no_matching_position') is True, exit_advice\n"
        "print(json.dumps({\n"
        "    'event': 'phase3_quant_api_smoke_ok',\n"
        "    'health_contract': {\n"
        "        'artifact_version': health.get('artifact_version'),\n"
        "        'artifact_lifecycle': health.get('artifact_lifecycle'),\n"
        "        'production_influence_authorized': health.get('production_influence_authorized'),\n"
        "        'training_data_sha256': health.get('training_data_sha256'),\n"
        "        'source_code_sha256': health.get('source_code_sha256'),\n"
        "        'return_evidence_ready': health.get('artifact_activation_manifest', {}).get('return_evidence_ready'),\n"
        "        'return_evidence_blockers': health.get('artifact_activation_manifest', {}).get('return_evidence_blockers'),\n"
        "    },\n"
        "    'profit_contract': {\n"
        "        'shadow_payload': bool(profit.get('shadow_payload')),\n"
        "        'live_mutation': profit.get('live_mutation'),\n"
        "        'promotion_flow': profit.get('promotion_flow'),\n"
        "        'production_eligible': profit.get('prediction_quality', {}).get('production_eligible'),\n"
        "        'production_permission': profit.get('production_permission'),\n"
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
            f"sudo install -m 0644 {sh(staged_service_path)} /etc/systemd/system/{sh(PHASE3_SERVICE_NAME)} && "
            "sudo systemctl daemon-reload",
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
            f"sudo systemctl enable {sh(PHASE3_SERVICE_NAME)} && "
            f"sudo systemctl restart {sh(PHASE3_SERVICE_NAME)}",
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
