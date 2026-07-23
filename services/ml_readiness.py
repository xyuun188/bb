from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
)
from services.training_data_quality import (
    DATA_QUALITY_VERSION,
    MARKET_FACT_CONTRACT_VERSION,
)


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        result = float(value)
        return (
            result if result == result and result not in {float("inf"), float("-inf")} else default
        )
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _reason(
    code: str,
    message: str,
    *,
    actual: Any = None,
    required: Any = None,
) -> dict[str, Any]:
    payload = {"code": code, "message": message}
    if actual is not None:
        payload["actual"] = actual
    if required is not None:
        payload["required"] = required
    return payload


def _side_label(side: str) -> str:
    return {"long": "做多", "short": "做空"}.get(str(side), str(side))


def _quality_totals(metadata: dict[str, Any]) -> dict[str, Any]:
    quality = _safe_dict(metadata.get("quality_report"))
    return _safe_dict(quality.get("totals"))


def _market_fact_contract_blockers(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    contract = _safe_dict(metadata.get("market_fact_contract"))
    blockers: list[dict[str, Any]] = []
    if contract.get("version") != MARKET_FACT_CONTRACT_VERSION:
        blockers.append(
            _reason(
                "artifact_market_fact_contract_missing_or_stale",
                "模型产物没有绑定当前要求的原生行情事实契约。",
                actual=contract.get("version") or "missing",
                required=MARKET_FACT_CONTRACT_VERSION,
            )
        )
        return blockers

    status = str(contract.get("status") or "").strip().lower()
    violation_count = contract.get("violation_count")
    if status != "clean" or violation_count != 0:
        blockers.append(
            _reason(
                "artifact_market_fact_contract_violated",
                "模型训练数据仍包含未解决的原生行情事实违规项。",
                actual={"status": status or "missing", "violation_count": violation_count},
                required={"status": "clean", "violation_count": 0},
            )
        )

    assertions = _safe_dict(contract.get("assertions"))
    required_assertions = (
        "native_instrument_identity_verified",
        "same_contract_price_path_verified",
        "executable_market_fact_verified",
    )
    failed_assertions = [name for name in required_assertions if assertions.get(name) is not True]
    if failed_assertions:
        blockers.append(
            _reason(
                "artifact_market_fact_assertions_incomplete",
                "模型产物的行情事实断言不完整。",
                actual=failed_assertions,
                required=list(required_assertions),
            )
        )

    provenance = _safe_dict(contract.get("provenance"))
    required_provenance = (
        "source",
        "observation_window",
        "generated_at",
        "strategy_version",
        "fallback_reason",
        "data_fingerprint",
    )
    missing_provenance = [
        name
        for name in required_provenance
        if name not in provenance or (name != "fallback_reason" and not provenance.get(name))
    ]
    if not any(name in provenance for name in ("sample_count", "effective_sample_size")):
        missing_provenance.append("sample_count/effective_sample_size")
    if missing_provenance:
        blockers.append(
            _reason(
                "artifact_market_fact_provenance_incomplete",
                "模型产物的行情事实来源信息不完整。",
                actual=missing_provenance,
                required=list(required_provenance) + ["sample_count/effective_sample_size"],
            )
        )
    return blockers


def _side_metric_blockers(metrics: dict[str, Any], side: str) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    side_text = _side_label(side)
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), 0.0) or 0.0
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), 0.0) or 0.0
    top_return_lcb = _safe_float(metrics.get(f"top_{side}_return_lcb_pct"), None)
    top_profit_factor = _safe_float(metrics.get(f"top_{side}_profit_factor"), None)
    top_tail_loss = _safe_float(metrics.get(f"top_{side}_tail_loss_rate"), None)
    bottom_tail_loss = _safe_float(metrics.get(f"bottom_{side}_tail_loss_rate"), None)
    if top_return <= bottom_return:
        blockers.append(
            _reason(
                f"{side}_top_return_not_above_bottom",
                f"{side_text}高分组费后收益没有高于低分组。",
                actual=round(top_return, 4),
                required=round(bottom_return, 4),
            )
        )
    if top_return_lcb is None or top_return_lcb <= 0:
        blockers.append(
            _reason(
                f"{side}_top_return_lcb_not_positive",
                f"{side_text}高分组收益置信下界不为正。",
                actual=None if top_return_lcb is None else round(top_return_lcb, 4),
                required=0.0,
            )
        )
    if top_profit_factor is None or top_profit_factor <= 1.0:
        blockers.append(
            _reason(
                f"{side}_top_profit_factor_not_above_one",
                f"{side_text}高分组盈亏比没有高于自然盈亏平衡线 1。",
                actual=None if top_profit_factor is None else round(top_profit_factor, 4),
                required=1.0,
            )
        )
    if (
        top_tail_loss is None
        or bottom_tail_loss is None
        or top_tail_loss > bottom_tail_loss
    ):
        blockers.append(
            _reason(
                f"{side}_top_tail_loss_not_improved",
                f"{side_text}高分组尾部亏损率缺失，或比低分组更差。",
                actual=None if top_tail_loss is None else round(top_tail_loss, 4),
                required=None if bottom_tail_loss is None else round(bottom_tail_loss, 4),
            )
        )
    return blockers


def _side_profit_quality_diagnostics(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    score_bucket_diagnostics = _safe_dict(metadata.get("score_bucket_diagnostics"))
    side_buckets = _safe_dict(score_bucket_diagnostics.get(side))
    top_bucket = _safe_dict(side_buckets.get("top"))
    bottom_bucket = _safe_dict(side_buckets.get("bottom"))
    top_return = _safe_float(metrics.get(f"top_{side}_avg_return_pct"), None)
    bottom_return = _safe_float(metrics.get(f"bottom_{side}_avg_return_pct"), None)
    top_return_lcb = _safe_float(metrics.get(f"top_{side}_return_lcb_pct"), None)
    top_profit_factor = _safe_float(metrics.get(f"top_{side}_profit_factor"), None)
    top_win = _safe_float(metrics.get(f"top_{side}_win_rate"), None)
    bottom_win = _safe_float(metrics.get(f"bottom_{side}_win_rate"), None)
    top_tail_loss = _safe_float(metrics.get(f"top_{side}_tail_loss_rate"), None)
    bottom_tail_loss = _safe_float(metrics.get(f"bottom_{side}_tail_loss_rate"), None)
    if top_tail_loss is None:
        top_tail_loss = _safe_float(top_bucket.get("tail_loss_rate"), None)
    if bottom_tail_loss is None:
        bottom_tail_loss = _safe_float(bottom_bucket.get("tail_loss_rate"), None)
    spread = (
        None
        if top_return is None or bottom_return is None
        else round(top_return - bottom_return, 6)
    )
    diagnosis: list[str] = []
    if spread is not None and spread <= 0:
        diagnosis.append("top_score_bucket_not_better_than_bottom")
    if top_return_lcb is None or top_return_lcb <= 0:
        diagnosis.append("top_score_return_lcb_not_positive")
    if top_profit_factor is None or top_profit_factor <= 1.0:
        diagnosis.append("top_score_profit_factor_not_above_one")
    if top_tail_loss is not None and bottom_tail_loss is not None and top_tail_loss > bottom_tail_loss:
        diagnosis.append("top_score_tail_loss_worse_than_bottom")
    return {
        "side": side,
        "training_target": "fee_after_realized_return_quality",
        "top_avg_return_pct": top_return,
        "bottom_avg_return_pct": bottom_return,
        "top_bottom_return_spread_pct": spread,
        "top_return_lcb_pct": top_return_lcb,
        "top_profit_factor": top_profit_factor,
        "top_win_rate": top_win,
        "bottom_win_rate": bottom_win,
        "top_tail_loss_rate": top_tail_loss,
        "bottom_tail_loss_rate": bottom_tail_loss,
        "top_bucket": top_bucket,
        "bottom_bucket": bottom_bucket,
        "diagnosis": diagnosis,
    }


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _artifact_evidence_blockers(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for field in ("training_data_sha256", "source_code_sha256"):
        if not _is_sha256(metadata.get(field)):
            blockers.append(
                _reason(
                    f"artifact_{field}_missing_or_invalid",
                    f"模型产物字段 {field} 缺失或无效。",
                    actual=metadata.get(field) or "missing",
                    required="sha256",
                )
            )
    if metadata.get("evaluation_group_policy") != "chronological_disjoint_decision_groups":
        blockers.append(
            _reason(
                "artifact_evaluation_group_policy_invalid",
                "模型评估没有使用按时间隔离且互不重叠的决策分组。",
                actual=metadata.get("evaluation_group_policy") or "missing",
                required="chronological_disjoint_decision_groups",
            )
        )
    for field in ("train_decision_group_count", "test_decision_group_count"):
        if int(_safe_float(metadata.get(field), 0.0) or 0) <= 0:
            blockers.append(
                _reason(
                    f"artifact_{field}_missing",
                    f"模型产物字段 {field} 没有提供非空数据分组。",
                )
            )
    governance = _safe_dict(metadata.get("governance_report"))
    if (
        not governance.get("quality_fingerprint")
        or governance.get("artifact_quality_fingerprint")
        != governance.get("quality_fingerprint")
        or governance.get("artifact_matches_quality") is not True
    ):
        blockers.append(
            _reason(
                "artifact_quality_fingerprint_mismatch",
                "模型产物没有绑定到完全一致的受治理干净训练视图。",
                actual={
                    "quality_fingerprint": governance.get("quality_fingerprint"),
                    "artifact_quality_fingerprint": governance.get(
                        "artifact_quality_fingerprint"
                    ),
                    "artifact_matches_quality": governance.get(
                        "artifact_matches_quality"
                    ),
                },
            )
        )
    walk_forward = _safe_dict(metadata.get("walk_forward_report"))
    if (
        walk_forward.get("status") != "complete"
        or walk_forward.get("decision_group_disjoint") is not True
        or walk_forward.get("chronological_label_disjoint") is not True
        or walk_forward.get("model_refit_per_fold") is not True
        or not list(walk_forward.get("folds") or [])
    ):
        blockers.append(
            _reason(
                "artifact_walk_forward_incomplete",
                "模型产物缺少按时间滚动且每折重新训练的完整验证证据。",
            )
        )
    authoritative = _safe_dict(metadata.get("authoritative_trade_return_evidence"))
    if (
        authoritative.get("version")
        != "2026-07-15.authoritative-trade-return-evidence.v1"
        or not _is_sha256(authoritative.get("data_fingerprint"))
    ):
        blockers.append(
            _reason(
                "authoritative_trade_return_evidence_missing",
                "模型产物缺少带数据指纹的权威成交收益证据。",
            )
        )
    return blockers


def _side_artifact_evidence_blockers(
    metadata: dict[str, Any],
    side: str,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    side_text = _side_label(side)
    walk_forward = _safe_dict(metadata.get("walk_forward_report"))
    walk_side = _safe_dict(_safe_dict(walk_forward.get("sides")).get(side))
    folds = list(walk_forward.get("folds") or [])
    fold_side_evidence = [
        _safe_dict(_safe_dict(fold).get("sides")).get(side)
        for fold in folds
        if isinstance(fold, dict)
    ]
    if (
        walk_side.get("promotion_math_ready") is not True
        or len(folds) < 2
        or not fold_side_evidence
        or any(
            not isinstance(evidence, dict)
            or evidence.get("promotion_math_ready") is not True
            for evidence in fold_side_evidence
        )
    ):
        blockers.append(
            _reason(
                f"{side}_walk_forward_return_stability_failed",
                f"{side_text}费后收益证据在时间滚动验证各折之间不稳定。",
            )
        )
    regime_stability = _safe_dict(walk_side.get("market_regime_stability"))
    if regime_stability.get("stable") is not True:
        blockers.append(
            _reason(
                f"{side}_market_regime_stability_failed",
                "fee-after return evidence is not stable across at least two market regimes",
                actual=regime_stability,
                required="two profitable market-regime windows",
            )
        )
    loso = _safe_dict(
        _safe_dict(metadata.get("leave_one_symbol_out_report")).get(side)
    )
    if loso.get("stable") is not True:
        blockers.append(
            _reason(
                f"{side}_leave_one_symbol_out_stability_failed",
                f"{side_text}收益证据依赖至少一个单独币种，泛化稳定性不足。",
            )
        )
    oos = _safe_dict(_safe_dict(metadata.get("oos_return_evaluation")).get(side))
    oos_profit_factor = _safe_float(oos.get("profit_factor"), None)
    if oos_profit_factor is None:
        blockers.append(
            _reason(
                f"{side}_oos_profit_factor_undefined",
                f"{side_text}样本外盈亏比因缺少亏损分母而无法计算。",
            )
        )
    elif oos_profit_factor <= 1.0:
        blockers.append(
            _reason(
                f"{side}_oos_profit_factor_not_above_break_even",
                f"{side_text}样本外盈亏比没有高于自然盈亏平衡线 1。",
                actual=oos_profit_factor,
                required=1.0,
            )
        )
    if (
        oos.get("promotion_math_ready") is not True
        or _safe_float(oos.get("return_lcb_pct"), None) is None
        or _safe_float(oos.get("cvar_10_pct"), None) is None
        or _safe_float(oos.get("max_drawdown_pct"), None) is None
    ):
        blockers.append(
            _reason(
                f"{side}_oos_return_tail_evidence_incomplete",
                f"{side_text}样本外收益下界、尾部风险或回撤证据不完整。",
            )
        )
    return blockers


_PAPER_CANARY_GLOBAL_EXEMPTIONS = frozenset(
    {
        "authoritative_trade_return_evidence_missing",
        "authoritative_return_supervision_missing",
    }
)


def _paper_canary_side_blockers(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    side: str,
) -> list[dict[str, Any]]:
    """Require a useful, governed ranker without pretending it is profitable.

    Paper bootstrap exists to collect the authoritative fills needed by the
    production gate. It therefore requires clean data, complete chronological
    evaluation and a better top score bucket, but deliberately does not require
    a positive fee-after LCB or profit factor before the first paper sample.
    """

    blockers = [
        item
        for item in _side_metric_blockers(metrics, side)
        if item.get("code")
        in {
            f"{side}_top_return_not_above_bottom",
            f"{side}_top_tail_loss_not_improved",
        }
    ]
    walk_forward = _safe_dict(metadata.get("walk_forward_report"))
    walk_side = _safe_dict(_safe_dict(walk_forward.get("sides")).get(side))
    fold_sides = [
        _safe_dict(_safe_dict(fold).get("sides")).get(side)
        for fold in list(walk_forward.get("folds") or [])
        if isinstance(fold, dict)
    ]
    if (
        not walk_side
        or not fold_sides
        or any(not isinstance(item, dict) or not item for item in fold_sides)
    ):
        blockers.append(
            _reason(
                f"{side}_paper_canary_walk_forward_evidence_missing",
                f"{_side_label(side)}缺少可用于模拟盘采样的逐折时间滚动证据。",
            )
        )
    oos = _safe_dict(_safe_dict(metadata.get("oos_return_evaluation")).get(side))
    if not oos or any(
        _safe_float(oos.get(field), None) is None
        for field in ("return_lcb_pct", "cvar_10_pct", "max_drawdown_pct")
    ):
        blockers.append(
            _reason(
                f"{side}_paper_canary_oos_distribution_incomplete",
                f"{_side_label(side)}样本外收益与尾部风险分布不完整，不能进入模拟盘采样。",
            )
        )
    return blockers


def _paper_canary_readiness(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    global_blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    paper_global_blockers = [
        item
        for item in global_blockers
        if item.get("code") not in _PAPER_CANARY_GLOBAL_EXEMPTIONS
    ]
    side_blockers = {
        side: _paper_canary_side_blockers(metadata, metrics, side)
        for side in ("long", "short")
    }
    eligible_sides = [
        side for side in ("long", "short") if not side_blockers[side]
    ]
    authorized = bool(not paper_global_blockers and eligible_sides)
    blockers = list(paper_global_blockers)
    if not eligible_sides:
        blockers.extend(side_blockers["long"])
        blockers.extend(side_blockers["short"])
    return {
        "version": "2026-07-17.paper-bootstrap-readiness.v1",
        "state": "ready" if authorized else "blocked",
        "authorized": authorized,
        "execution_scope": "paper_only",
        "production_permission": False,
        "eligible_sides": eligible_sides,
        "blocking_reasons": blockers,
        "side_blocking_reasons": side_blockers,
        "promotion_requirements": (
            "paper canary collects version-bound realized cost and return evidence; "
            "live promotion still requires positive walk-forward, OOS and authoritative fee-after LCB"
        ),
    }


def build_ml_readiness_report(
    metadata: dict[str, Any],
    influence: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    metrics = _safe_dict(metadata.get("metrics"))
    quality = _safe_dict(metadata.get("quality_report"))
    totals = _quality_totals(metadata)
    sample_count = int(metadata.get("sample_count") or 0)
    test_count = int(metadata.get("test_count") or 0)
    total_samples = int(totals.get("total") or sample_count or 0)
    excluded_count = int(totals.get("excluded") or 0)
    downweighted_count = int(totals.get("downweighted") or 0)
    contamination_downweighted_count = totals.get("contamination_downweighted")
    if contamination_downweighted_count is None:
        contamination_downweighted_count = downweighted_count
    contamination_downweighted_count = int(contamination_downweighted_count or 0)
    benign_downweighted_count = int(totals.get("benign_downweighted") or 0)
    dirty_count = excluded_count + contamination_downweighted_count
    dirty_ratio = dirty_count / max(total_samples, 1)
    trained_at = _parse_datetime(metadata.get("trained_at") or metadata.get("version"))
    age_seconds = None
    if trained_at is not None:
        age_seconds = max(((now or datetime.now(UTC)) - trained_at).total_seconds(), 0.0)
    data_quality_version = quality.get("data_quality_version")
    objective_name = metadata.get("objective_name")
    objective_version = metadata.get("objective_version")
    label_version = metadata.get("label_version")
    training_cost_policy = str(metadata.get("training_cost_policy") or "")
    profit_supervision_version = metadata.get("profit_supervision_version")
    profit_supervision_report = _safe_dict(metadata.get("profit_supervision_report"))
    actual_trade_calibration = _safe_dict(metadata.get("actual_trade_calibration"))
    actual_trade_profiles = _safe_dict(actual_trade_calibration.get("profiles"))
    tail_loss_policy = _safe_dict(metadata.get("tail_loss_policy"))
    tail_loss_scales = _safe_dict(metadata.get("tail_loss_scale_pct"))

    global_blockers: list[dict[str, Any]] = [
        *_market_fact_contract_blockers(metadata),
        *_artifact_evidence_blockers(metadata),
    ]
    if objective_name != RETURN_OBJECTIVE_NAME or objective_version != RETURN_OBJECTIVE_VERSION:
        global_blockers.append(
            _reason(
                "artifact_objective_version_mismatch",
                "模型产物没有使用要求的费后收益目标版本。",
                actual=f"{objective_name or 'unknown'}@{objective_version or 'unknown'}",
                required=f"{RETURN_OBJECTIVE_NAME}@{RETURN_OBJECTIVE_VERSION}",
            )
        )
    if label_version != RETURN_LABEL_VERSION:
        global_blockers.append(
            _reason(
                "artifact_return_label_version_mismatch",
                "模型产物没有使用要求的费后收益标签契约。",
                actual=label_version or "unknown",
                required=RETURN_LABEL_VERSION,
            )
        )
    if training_cost_policy != "separated_market_opportunity_and_execution_cost_tasks":
        global_blockers.append(
            _reason(
                "artifact_cost_policy_incomplete",
                "模型产物没有把市场机会与执行成本分开建模。",
                actual=training_cost_policy or "missing",
                required="separated_market_opportunity_and_execution_cost_tasks",
            )
        )
    if (
        profit_supervision_version != PROFIT_SUPERVISION_VERSION
        or profit_supervision_report.get("version") != PROFIT_SUPERVISION_VERSION
    ):
        global_blockers.append(
            _reason(
                "profit_supervision_contract_missing",
                "模型产物没有携带市场机会与执行成本分离的收益监督契约。",
                actual=profit_supervision_version or "missing",
                required=PROFIT_SUPERVISION_VERSION,
            )
        )
    for field, code in (
        ("shadow_market_sample_count", "shadow_market_supervision_missing"),
        (
            "shadow_counterfactual_cost_sample_count",
            "counterfactual_cost_supervision_missing",
        ),
        ("actual_realized_return_sample_count", "authoritative_return_supervision_missing"),
    ):
        if int(_safe_float(profit_supervision_report.get(field), 0.0) or 0) <= 0:
            global_blockers.append(
                _reason(
                    code,
                    f"分离监督报告缺少字段 {field}。",
                    actual=profit_supervision_report.get(field),
                    required="non-empty authoritative distribution",
                )
            )
    for side in ("long", "short"):
        side_policy = _safe_dict(tail_loss_policy.get(side))
        scale = _safe_float(tail_loss_scales.get(side), None)
        required_provenance = {
            "source",
            "observation_window",
            "sample_count",
            "generated_at",
            "strategy_version",
            "fallback_reason",
        }
        if not required_provenance.issubset(side_policy) or scale is None or scale <= 0:
            global_blockers.append(
                _reason(
                    f"{side}_dynamic_tail_policy_incomplete",
                    f"{_side_label(side)}动态尾部亏损策略元数据不完整。",
                    actual={"policy": side_policy, "scale_pct": scale},
                    required="complete empirical policy provenance and positive artifact scale",
                )
            )
    if sample_count <= 0:
        global_blockers.append(
            _reason(
                "training_distribution_missing",
                "缺少训练集收益分布。",
                actual=sample_count,
            )
        )
    if test_count <= 0:
        global_blockers.append(
            _reason(
                "holdout_distribution_missing",
                "缺少留出集收益分布。",
                actual=test_count,
            )
        )
    side_blockers = {
        side: [
            *_side_metric_blockers(metrics, side),
            *_side_artifact_evidence_blockers(metadata, side),
        ]
        for side in ("long", "short")
    }
    for side in ("long", "short"):
        profile = _safe_dict(actual_trade_profiles.get(f"*|{side}"))
        actual_slippage = _safe_dict(profile.get("slippage_pct"))
        if int(_safe_float(actual_slippage.get("count"), 0.0) or 0) <= 0:
            side_blockers[side].append(
                _reason(
                    f"{side}_authoritative_slippage_calibration_missing",
                    f"{_side_label(side)}缺少权威真实滑点校准。",
                )
            )
    profit_quality_diagnostics = {
        side: _side_profit_quality_diagnostics(metadata, metrics, side)
        for side in ("long", "short")
    }
    side_enabled = {
        side: not bool(blockers) and bool(_safe_dict(influence.get(side)).get("enabled", True))
        for side, blockers in side_blockers.items()
    }
    if data_quality_version != DATA_QUALITY_VERSION:
        global_blockers.append(
            _reason(
                "training_data_version_stale",
                "模型使用了旧版数据质量契约训练。",
                actual=data_quality_version or "unknown",
                required=DATA_QUALITY_VERSION,
            )
        )
    if age_seconds is None:
        global_blockers.append(
            _reason(
                "model_training_timestamp_missing",
                "模型缺少有效的训练时间。",
            )
        )

    paper_canary = _paper_canary_readiness(
        metadata,
        metrics,
        global_blockers,
    )

    live_enabled_sides = [side for side, enabled in side_enabled.items() if enabled]
    partial_live_influence_allowed = bool(
        not global_blockers and live_enabled_sides and influence.get("enabled")
    )
    blockers = (
        global_blockers
        if partial_live_influence_allowed
        else [
            *global_blockers,
            *side_blockers["long"],
            *side_blockers["short"],
        ]
    )
    maturity_blocked = any(
        item["code"] in {"training_distribution_missing", "holdout_distribution_missing"}
        for item in blockers
    )
    if partial_live_influence_allowed:
        state = "ready" if len(live_enabled_sides) == 2 else "partial_ready"
    elif maturity_blocked:
        state = "learning_only"
    elif blockers:
        state = "degraded"
    elif influence.get("advisory_enabled"):
        state = "shadow_ready"
    else:
        state = "learning_only"

    return {
        "state": state,
        "live_ml_ready": partial_live_influence_allowed,
        "paper_canary": paper_canary,
        "live_enabled_sides": live_enabled_sides,
        "side_blocking_reasons": side_blockers,
        "blocking_reasons": blockers,
        "profit_quality_diagnostics": profit_quality_diagnostics,
        "next_training_conditions": {
            "trigger": "new_authoritative_cost_complete_sample_or_data_contract_change",
        },
        "thresholds": {
            "min_top_return_lcb_pct": 0.0,
            "min_top_profit_factor": 1.0,
            "threshold_policy": "profitability_math_boundaries_and_empirical_confidence_intervals",
        },
        "policy_provenance": {
            "source": "separated_market_cost_and_authoritative_trade_distributions",
            "observation_window": "artifact_train_holdout_and_okx_trade_calibration",
            "sample_count": sample_count,
            "test_sample_count": test_count,
            "generated_at": (now or datetime.now(UTC)).isoformat(),
            "strategy_version": "2026-07-14.separated-ml-readiness.v2",
            "fallback_reason": "" if sample_count > 0 and test_count > 0 else "distribution_missing",
        },
        "metrics": {
            "sample_count": sample_count,
            "test_count": test_count,
            "quarantined_sample_count": excluded_count,
            "downweighted_sample_count": downweighted_count,
            "benign_downweighted_sample_count": benign_downweighted_count,
            "contamination_downweighted_sample_count": contamination_downweighted_count,
            "dirty_sample_ratio": round(dirty_ratio, 4),
            "long_auc": _safe_float(metrics.get("long_auc"), None),
            "short_auc": _safe_float(metrics.get("short_auc"), None),
            "long_pr_auc": _safe_float(metrics.get("long_pr_auc"), None),
            "short_pr_auc": _safe_float(metrics.get("short_pr_auc"), None),
            "long_accuracy": _safe_float(metrics.get("long_accuracy"), None),
            "short_accuracy": _safe_float(metrics.get("short_accuracy"), None),
            "top_long_avg_return_pct": _safe_float(metrics.get("top_long_avg_return_pct"), None),
            "top_long_return_lcb_pct": _safe_float(
                metrics.get("top_long_return_lcb_pct"), None
            ),
            "top_long_profit_factor": _safe_float(
                metrics.get("top_long_profit_factor"), None
            ),
            "bottom_long_avg_return_pct": _safe_float(
                metrics.get("bottom_long_avg_return_pct"), None
            ),
            "top_long_bottom_return_spread_pct": profit_quality_diagnostics["long"].get(
                "top_bottom_return_spread_pct"
            ),
            "top_long_tail_loss_rate": _safe_float(
                metrics.get("top_long_tail_loss_rate"), None
            ),
            "bottom_long_tail_loss_rate": _safe_float(
                metrics.get("bottom_long_tail_loss_rate"), None
            ),
            "top_short_avg_return_pct": _safe_float(metrics.get("top_short_avg_return_pct"), None),
            "top_short_return_lcb_pct": _safe_float(
                metrics.get("top_short_return_lcb_pct"), None
            ),
            "top_short_profit_factor": _safe_float(
                metrics.get("top_short_profit_factor"), None
            ),
            "bottom_short_avg_return_pct": _safe_float(
                metrics.get("bottom_short_avg_return_pct"), None
            ),
            "top_short_bottom_return_spread_pct": profit_quality_diagnostics["short"].get(
                "top_bottom_return_spread_pct"
            ),
            "top_short_tail_loss_rate": _safe_float(
                metrics.get("top_short_tail_loss_rate"), None
            ),
            "bottom_short_tail_loss_rate": _safe_float(
                metrics.get("bottom_short_tail_loss_rate"), None
            ),
            "trained_at": trained_at.isoformat() if trained_at else None,
            "model_age_seconds": None if age_seconds is None else round(age_seconds, 1),
            "training_data_version": data_quality_version,
            "required_training_data_version": DATA_QUALITY_VERSION,
            "objective_name": objective_name,
            "objective_version": objective_version,
            "required_objective_name": RETURN_OBJECTIVE_NAME,
            "required_objective_version": RETURN_OBJECTIVE_VERSION,
            "label_version": label_version,
            "required_label_version": RETURN_LABEL_VERSION,
            "profit_supervision_version": profit_supervision_version,
            "required_profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        },
    }


def disabled_ml_readiness(reason_code: str, message: str) -> dict[str, Any]:
    return {
        "state": "disabled",
        "live_ml_ready": False,
        "paper_canary": {
            "version": "2026-07-17.paper-bootstrap-readiness.v1",
            "state": "blocked",
            "authorized": False,
            "execution_scope": "paper_only",
            "production_permission": False,
            "eligible_sides": [],
            "blocking_reasons": [_reason(reason_code, message)],
            "side_blocking_reasons": {"long": [], "short": []},
        },
        "blocking_reasons": [_reason(reason_code, message)],
        "next_training_conditions": {
            "trigger": "new_authoritative_cost_complete_sample_or_data_contract_change",
        },
        "thresholds": {
            "min_top_return_lcb_pct": 0.0,
            "min_top_profit_factor": 1.0,
            "threshold_policy": "profitability_math_boundaries_and_empirical_confidence_intervals",
        },
        "policy_provenance": {
            "source": "artifact_holdout_fee_after_return_distribution",
            "observation_window": "artifact_train_and_holdout_windows",
            "sample_count": 0,
            "test_sample_count": 0,
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy_version": "2026-07-12.ml-readiness-return-lcb.v1",
            "fallback_reason": reason_code,
        },
        "metrics": {
            "sample_count": 0,
            "test_count": 0,
            "quarantined_sample_count": 0,
            "downweighted_sample_count": 0,
            "dirty_sample_ratio": 0.0,
            "training_data_version": None,
            "required_training_data_version": DATA_QUALITY_VERSION,
        },
    }
