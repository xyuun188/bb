from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal

from services.trading_params import DEFAULT_TRADING_PARAMS

_QUALITY_PARAMS = DEFAULT_TRADING_PARAMS.training_data_quality
DATA_QUALITY_VERSION = "2026-06-20.v2"
QualityStatus = Literal["included", "downweighted", "excluded"]
SampleKind = Literal["shadow", "trade", "sequence", "text_sentiment"]


@dataclass(frozen=True)
class SampleQualityAssessment:
    status: QualityStatus
    score: float
    weight: float
    reasons: tuple[str, ...]
    version: str = DATA_QUALITY_VERSION

    @property
    def exclude_from_training(self) -> bool:
        return self.status == "excluded"

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_quality_version": self.version,
            "data_quality_status": self.status,
            "data_quality_score": round(self.score, 4),
            "sample_weight": round(self.weight, 4),
            "exclude_from_training": self.exclude_from_training,
            "exclude_from_training_reason": (
                ";".join(self.reasons) if self.exclude_from_training else ""
            ),
            "quality_reasons": list(self.reasons),
        }


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _features(sample: dict[str, Any]) -> dict[str, Any]:
    value = sample.get("features")
    return value if isinstance(value, dict) else {}


def _market_data_quality_issue(sample: dict[str, Any], features: dict[str, Any]) -> str:
    for container in (sample, features):
        if not isinstance(container, dict):
            continue
        issue = container.get("market_data_quality")
        if isinstance(issue, dict):
            code = _safe_str(issue.get("code"))
            if code:
                return code
        reason = _safe_str(container.get("training_quality_reason"))
        if reason.startswith("market_data_quality:"):
            return reason.split(":", 1)[1]
    return ""


def _final_assessment(
    score: float, reasons: list[str], *, exclude: bool = False
) -> SampleQualityAssessment:
    if exclude:
        status: QualityStatus = "excluded"
        weight = 0.0
        score = min(score, _QUALITY_PARAMS.excluded_score_cap)
    elif score < _QUALITY_PARAMS.include_score_threshold or reasons:
        status = "downweighted"
        weight = max(
            _QUALITY_PARAMS.downweighted_min_weight,
            min(score, _QUALITY_PARAMS.downweighted_max_weight),
        )
    else:
        status = "included"
        weight = 1.0
    return SampleQualityAssessment(
        status=status, score=max(0.0, min(score, 1.0)), weight=weight, reasons=tuple(reasons)
    )


def assess_shadow_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    features = _features(sample)
    reasons: list[str] = []
    score = 1.0
    exclude = False

    quality_issue = _market_data_quality_issue(sample, features)
    if quality_issue:
        return _final_assessment(
            0.0,
            [f"market_data_quality:{quality_issue}"],
            exclude=True,
        )

    if not features:
        return _final_assessment(0.0, ["missing_features"], exclude=True)

    long_return = _safe_float(sample.get("long_return_pct"), None)
    short_return = _safe_float(sample.get("short_return_pct"), None)
    if long_return is None or short_return is None:
        return _final_assessment(0.0, ["missing_outcome_returns"], exclude=True)
    if (
        abs(long_return) > _QUALITY_PARAMS.abnormal_shadow_return_abs_pct
        or abs(short_return) > _QUALITY_PARAMS.abnormal_shadow_return_abs_pct
    ):
        return _final_assessment(0.05, ["abnormal_outcome_return"], exclude=True)

    action = _safe_str(sample.get("decision_action")).lower()
    best_action = _safe_str(sample.get("best_action")).lower()
    if action == "hold":
        if bool(sample.get("missed_opportunity")) and best_action in {"long", "short"}:
            score -= _QUALITY_PARAMS.hold_missed_opportunity_penalty
            reasons.append("hold_missed_opportunity_downweighted")
        else:
            score -= _QUALITY_PARAMS.hold_observation_penalty
            reasons.append("hold_observation_downweighted")

    confidence = _safe_float(sample.get("decision_confidence"), 0.0) or 0.0
    if confidence < _QUALITY_PARAMS.very_low_confidence_threshold:
        score -= _QUALITY_PARAMS.very_low_confidence_penalty
        reasons.append("very_low_decision_confidence")

    horizon = int(_safe_float(sample.get("horizon_minutes"), 0.0) or 0)
    if horizon <= 0 or horizon > _QUALITY_PARAMS.max_horizon_minutes:
        score -= _QUALITY_PARAMS.invalid_horizon_penalty
        reasons.append("invalid_horizon_minutes")

    current_price = _safe_float(features.get("current_price") or features.get("close"), None)
    if current_price is None or current_price <= 0:
        score -= _QUALITY_PARAMS.invalid_price_penalty
        reasons.append("missing_or_invalid_price_feature")

    spread = abs(_safe_float(features.get("spread_pct"), 0.0) or 0.0)
    if spread > _QUALITY_PARAMS.abnormal_spread_pct:
        exclude = True
        reasons.append("abnormal_spread_feature")
    elif spread > _QUALITY_PARAMS.wide_spread_pct:
        score -= _QUALITY_PARAMS.wide_spread_penalty
        reasons.append("wide_spread_feature")

    return _final_assessment(score, reasons, exclude=exclude)


def assess_trade_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    reasons: list[str] = []
    score = 1.0
    exclude = False

    source = _safe_str(sample.get("source")).lower()
    model_name = _safe_str(sample.get("model_name")).lower()
    if source in set(_QUALITY_PARAMS.manual_trade_sources) or "manual" in model_name:
        return _final_assessment(0.0, ["manual_or_test_trade"], exclude=True)

    side = _safe_str(sample.get("side")).lower()
    if side not in {"long", "short"}:
        return _final_assessment(0.0, ["invalid_trade_side"], exclude=True)

    quantity = _safe_float(sample.get("quantity"), 0.0) or 0.0
    if quantity <= 0:
        return _final_assessment(0.0, ["non_positive_quantity"], exclude=True)

    entry_price = _safe_float(sample.get("entry_price"), 0.0) or 0.0
    exit_price = _safe_float(sample.get("exit_price"), 0.0) or 0.0
    if entry_price <= 0 or exit_price < 0:
        score -= _QUALITY_PARAMS.invalid_trade_price_penalty
        reasons.append("invalid_trade_price")

    hold_minutes = _safe_float(sample.get("hold_minutes"), 0.0) or 0.0
    pnl = _safe_float(sample.get("realized_pnl"), 0.0) or 0.0
    fee = abs(_safe_float(sample.get("fee_estimate"), 0.0) or 0.0)
    if hold_minutes < _QUALITY_PARAMS.fast_loss_exit_minutes and pnl < 0:
        score -= _QUALITY_PARAMS.fast_loss_exit_penalty
        reasons.append("fast_loss_exit_requires_review")
    elif hold_minutes <= 0:
        score -= _QUALITY_PARAMS.missing_hold_duration_penalty
        reasons.append("missing_hold_duration")

    if fee > 0 and abs(pnl) <= fee * _QUALITY_PARAMS.fee_dominated_multiple:
        score -= _QUALITY_PARAMS.fee_dominated_penalty
        reasons.append("fee_dominated_trade")

    outcome = _safe_str(sample.get("outcome")).lower()
    if outcome not in {"profit", "loss", "flat", "win"}:
        score -= _QUALITY_PARAMS.unknown_outcome_penalty
        reasons.append("missing_or_unknown_outcome")

    return _final_assessment(score, reasons, exclude=exclude)


def assess_sequence_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    reasons: list[str] = []
    score = 1.0
    closes = sample.get("close_sequence") or []
    if not isinstance(closes, list) or len(closes) < _QUALITY_PARAMS.min_sequence_length:
        return _final_assessment(0.0, ["short_price_sequence"], exclude=True)
    numeric_closes = [_safe_float(value, None) for value in closes]
    if any(value is None or value <= 0 for value in numeric_closes):
        return _final_assessment(0.0, ["invalid_price_sequence"], exclude=True)
    future_return = _safe_float(sample.get("future_return_pct"), None)
    if future_return is None:
        return _final_assessment(0.0, ["missing_future_return"], exclude=True)
    if abs(future_return) > _QUALITY_PARAMS.abnormal_future_return_abs_pct:
        return _final_assessment(0.05, ["abnormal_future_return"], exclude=True)
    timeframe = _safe_str(sample.get("timeframe"))
    if timeframe not in set(_QUALITY_PARAMS.allowed_sequence_timeframes):
        score -= _QUALITY_PARAMS.unknown_timeframe_penalty
        reasons.append("unknown_timeframe")
    return _final_assessment(score, reasons)


def assess_text_sentiment_sample(sample: dict[str, Any]) -> SampleQualityAssessment:
    text = _safe_str(sample.get("text"))
    if len(text) < _QUALITY_PARAMS.min_text_length:
        return _final_assessment(0.0, ["empty_or_too_short_text"], exclude=True)
    score = 1.0
    reasons: list[str] = []
    platform = _safe_str(sample.get("platform")).lower()
    if platform in {"", "unknown"}:
        score -= _QUALITY_PARAMS.unknown_text_source_penalty
        reasons.append("unknown_text_source")
    sentiment = _safe_float(sample.get("sentiment_score"), None)
    if sentiment is None:
        score -= _QUALITY_PARAMS.missing_sentiment_penalty
        reasons.append("missing_sentiment_score")
    return _final_assessment(score, reasons)


ASSESSORS = {
    "shadow": assess_shadow_sample,
    "trade": assess_trade_sample,
    "sequence": assess_sequence_sample,
    "text_sentiment": assess_text_sentiment_sample,
}


def annotate_sample(sample: dict[str, Any], kind: SampleKind) -> dict[str, Any]:
    assessment = ASSESSORS[kind](sample)
    annotated = dict(sample)
    annotated.update(assessment.as_dict())
    return annotated


def annotate_samples(samples: list[dict[str, Any]], kind: SampleKind) -> list[dict[str, Any]]:
    return [annotate_sample(sample, kind) for sample in samples]


def trainable_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sample for sample in samples if not bool(sample.get("exclude_from_training"))]


def _sample_action(sample: dict[str, Any]) -> str:
    return (
        _safe_str(sample.get("decision_action"))
        or _safe_str(sample.get("best_action"))
        or _safe_str(sample.get("side"))
        or "unknown"
    ).lower()


def _sample_source(sample: dict[str, Any], kind: str) -> str:
    source = _safe_str(sample.get("source"))
    platform = _safe_str(sample.get("platform"))
    if kind == "text_sentiment" and platform and source in {"", "news", "social"}:
        return platform
    return source or platform


def quality_report(samples_by_kind: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_kind: dict[str, Any] = {}
    totals = Counter()
    reason_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    timeframe_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    trainable_weight_total = 0.0
    for kind, samples in samples_by_kind.items():
        status_counts: Counter[str] = Counter()
        weight_total = 0.0
        kind_action_counts: Counter[str] = Counter()
        kind_trainable_action_counts: Counter[str] = Counter()
        kind_timeframe_counts: Counter[str] = Counter()
        kind_trainable_timeframe_counts: Counter[str] = Counter()
        kind_source_counts: Counter[str] = Counter()
        kind_trainable_source_counts: Counter[str] = Counter()
        for sample in samples:
            status = _safe_str(sample.get("data_quality_status")) or "unknown"
            trainable = status != "excluded"
            status_counts[status] += 1
            totals[status] += 1
            weight = float(_safe_float(sample.get("sample_weight"), 0.0) or 0.0)
            weight_total += weight
            if trainable:
                trainable_weight_total += weight
            action = _sample_action(sample)
            kind_action_counts[action] += 1
            action_counts[f"{kind}:{action}"] += 1
            if trainable:
                kind_trainable_action_counts[action] += 1
            timeframe = _safe_str(sample.get("timeframe") or sample.get("horizon_minutes"))
            if timeframe:
                kind_timeframe_counts[timeframe] += 1
                timeframe_counts[f"{kind}:{timeframe}"] += 1
                if trainable:
                    kind_trainable_timeframe_counts[timeframe] += 1
            source = _sample_source(sample, kind)
            if source:
                kind_source_counts[source] += 1
                source_counts[f"{kind}:{source}"] += 1
                if trainable:
                    kind_trainable_source_counts[source] += 1
            for reason in sample.get("quality_reasons") or []:
                reason_key = f"{kind}:{reason}"
                reason_counts[reason_key] += 1
        total = len(samples)
        by_kind[kind] = {
            "total": total,
            "included": int(status_counts.get("included", 0)),
            "downweighted": int(status_counts.get("downweighted", 0)),
            "excluded": int(status_counts.get("excluded", 0)),
            "effective_weight": round(weight_total, 4),
            "effective_weight_ratio": round(weight_total / max(total, 1), 4),
            "actions": dict(kind_action_counts),
            "trainable_actions": dict(kind_trainable_action_counts),
            "timeframes": dict(kind_timeframe_counts),
            "trainable_timeframes": dict(kind_trainable_timeframe_counts),
            "sources": dict(kind_source_counts),
            "trainable_sources": dict(kind_trainable_source_counts),
        }
    total_count = sum(len(samples) for samples in samples_by_kind.values())
    return {
        "data_quality_version": DATA_QUALITY_VERSION,
        "policy": asdict(DEFAULT_TRADING_PARAMS.training_data_quality),
        "by_kind": by_kind,
        "totals": {
            "total": total_count,
            "included": int(totals.get("included", 0)),
            "downweighted": int(totals.get("downweighted", 0)),
            "excluded": int(totals.get("excluded", 0)),
            "effective_weight": round(trainable_weight_total, 4),
            "effective_weight_ratio": round(trainable_weight_total / max(total_count, 1), 4),
        },
        "top_reasons": [
            {"reason": reason, "count": count} for reason, count in reason_counts.most_common(20)
        ],
        "top_actions": [
            {"action": action, "count": count} for action, count in action_counts.most_common(20)
        ],
        "top_timeframes": [
            {"timeframe": timeframe, "count": count}
            for timeframe, count in timeframe_counts.most_common(20)
        ],
        "top_sources": [
            {"source": source, "count": count} for source, count in source_counts.most_common(20)
        ],
    }


def annotate_training_payload(
    *,
    shadow_samples: list[dict[str, Any]],
    trade_samples: list[dict[str, Any]],
    sequence_samples: list[dict[str, Any]],
    text_sentiment_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    annotated_shadow = annotate_samples(shadow_samples, "shadow")
    annotated_trade = annotate_samples(trade_samples, "trade")
    annotated_sequence = annotate_samples(sequence_samples, "sequence")
    annotated_text = annotate_samples(text_sentiment_samples, "text_sentiment")
    return {
        "shadow_samples": trainable_samples(annotated_shadow),
        "trade_samples": trainable_samples(annotated_trade),
        "sequence_samples": trainable_samples(annotated_sequence),
        "text_sentiment_samples": trainable_samples(annotated_text),
        "quality_report": quality_report(
            {
                "shadow": annotated_shadow,
                "trade": annotated_trade,
                "sequence": annotated_sequence,
                "text_sentiment": annotated_text,
            }
        ),
    }
