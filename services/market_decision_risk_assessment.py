"""Risk-engine adapter for market-analysis decisions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

AccountBalanceProvider = Callable[[str], Awaitable[float]]
FalsePositiveChecker = Callable[[DecisionOutput, str | None, Any], Awaitable[bool]]

REQUIRED_ENTRY_EXPERTS = {
    "trend_expert",
    "momentum_expert",
    "sentiment_expert",
    "position_expert",
    "risk_expert",
}
CORE_ENTRY_EXPERTS = {"trend_expert", "momentum_expert", "risk_expert"}
BALANCED_PROBE_OPTIONAL_EXPERTS = {"sentiment_expert", "position_expert"}
UNTRUSTED_EXPERT_TIMING_STATUSES = {
    "batch_fallback",
    "partial_batch_fallback",
    "circuit_breaker_fallback",
    "fast_prefilter",
    "failed",
    "invalid",
    "timeout",
    "timeout_fallback",
    "independent_provider_fallback",
    "independent_provider_failed",
}
BALANCED_PROBE_EXPERT_INTEGRITY_MODE = "balanced_probe_allow_one_non_core_missing"
BALANCED_PROBE_MAX_POSITION_SIZE_PCT = 0.018


@dataclass(frozen=True, slots=True)
class MarketDecisionRiskAssessmentPolicy:
    """Assess market-analysis decisions through risk engine with rescue checks."""

    risk_engine: Any
    account_balance_provider: AccountBalanceProvider
    false_positive_checker: FalsePositiveChecker

    async def assess(
        self,
        *,
        decision: DecisionOutput,
        model_name: str,
        open_positions: list[dict[str, Any]],
        feature_vector: Any,
        strategy_mode_context: dict[str, Any] | None = None,
    ) -> Any:
        """Run risk assessment and apply price-action false-positive override."""

        model_positions = [
            position for position in open_positions if position.get("model_name") == model_name
        ]
        assessment = self.risk_engine.assess(
            decision,
            current_positions=model_positions,
            account_balance=await self.account_balance_provider(model_name),
            headlines=getattr(feature_vector, "recent_headlines", []),
            sentiment_scores=[],
            price_change_1m=getattr(feature_vector, "returns_1", 0.0),
            volume_ratio=getattr(feature_vector, "volume_ratio", 1.0),
            adx_14=getattr(feature_vector, "adx_14", None),
        )
        expert_block_reason = expert_analysis_entry_block_reason(
            decision,
            strategy_mode_context=strategy_mode_context,
        )
        if expert_block_reason:
            assessment.approved = False
            assessment.decision = None
            assessment.rejection_reason = expert_block_reason
            return assessment
        if not assessment.approved and await self.false_positive_checker(
            decision,
            assessment.rejection_reason,
            assessment,
        ):
            assessment.approved = True
            assessment.decision = decision
            assessment.rejection_reason = ""
        return assessment


def expert_analysis_entry_block_reason(
    decision: DecisionOutput,
    strategy_mode_context: dict[str, Any] | None = None,
) -> str | None:
    """Block market entries unless required experts returned real analysis.

    In the normal profile all five entry experts are required.  A scheduled
    balanced probe may tolerate exactly one missing non-core expert, but only
    when trend, momentum, and risk experts are all trusted and the entry is
    capped to a tiny probe size.
    """

    if not decision.is_entry:
        return None
    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
    expert_integrity_mode = _expert_integrity_mode(strategy_mode_context)
    model_timings = raw.get("model_timings")
    if not isinstance(model_timings, list) or not model_timings:
        return (
            "专家分析完整性保护[expert_integrity]：未记录 5 个专家的大模型分析耗时，"
            "本次新开仓不执行。"
        )

    trusted_by_name: set[str] = set()
    untrusted: list[str] = []
    seen: set[str] = set()
    for item in model_timings:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name not in REQUIRED_ENTRY_EXPERTS:
            continue
        seen.add(name)
        status = str(item.get("status") or "").lower()
        provider = str(item.get("provider_model") or "").lower()
        fallback_flag = bool(item.get("batch_expert_fallback") or item.get("fallback"))
        if status == "completed" and provider != "local_fast_prefilter" and not fallback_flag:
            trusted_by_name.add(name)
        elif status in UNTRUSTED_EXPERT_TIMING_STATUSES or "fallback" in status:
            untrusted.append(f"{name}:{status or 'unknown'}")

    missing = sorted(REQUIRED_ENTRY_EXPERTS - trusted_by_name)
    if not missing:
        return None
    if _allow_balanced_probe_missing_expert(
        decision,
        raw,
        missing=missing,
        trusted_by_name=trusted_by_name,
        expert_integrity_mode=expert_integrity_mode,
    ):
        return None

    missing_seen = sorted(REQUIRED_ENTRY_EXPERTS - seen)
    details = untrusted[:5] or [f"{name}:missing" for name in missing_seen[:5]]
    return (
        "专家分析完整性保护[expert_integrity]：新开仓必须拿到 5 个专家的真实大模型分析；"
        f"当前缺少可信专家 {', '.join(missing)}"
        f"（异常：{'; '.join(details) if details else '无耗时记录'}），本次不开仓。"
    )


def _expert_integrity_mode(strategy_mode_context: dict[str, Any] | None) -> str:
    strategy_mode = strategy_mode_context if isinstance(strategy_mode_context, dict) else {}
    learning = strategy_mode.get("strategy_learning")
    runtime = learning.get("runtime") if isinstance(learning, dict) else {}
    return str(
        strategy_mode.get("expert_integrity_mode")
        or (runtime if isinstance(runtime, dict) else {}).get("expert_integrity_mode")
        or "strict_all_required"
    )


def _allow_balanced_probe_missing_expert(
    decision: DecisionOutput,
    raw: dict[str, Any],
    *,
    missing: list[str],
    trusted_by_name: set[str],
    expert_integrity_mode: str,
) -> bool:
    """Allow a tiny probe only when one non-core expert is missing."""

    if expert_integrity_mode != BALANCED_PROBE_EXPERT_INTEGRITY_MODE:
        return False
    if len(missing) != 1:
        return False
    missing_name = missing[0]
    if missing_name not in BALANCED_PROBE_OPTIONAL_EXPERTS:
        return False
    if not CORE_ENTRY_EXPERTS.issubset(trusted_by_name):
        return False

    original_size = float(decision.position_size_pct or 0.0)
    if original_size > 0:
        decision.position_size_pct = min(original_size, BALANCED_PROBE_MAX_POSITION_SIZE_PCT)
    raw["expert_integrity_probe"] = {
        "applied": True,
        "mode": expert_integrity_mode,
        "missing_expert": missing_name,
        "trusted_experts": sorted(trusted_by_name),
        "original_position_size_pct": round(original_size, 6),
        "adjusted_position_size_pct": round(float(decision.position_size_pct or 0.0), 6),
        "policy": (
            "one non-core expert may be missing only in balanced_probe mode; "
            "core experts remain required"
        ),
    }
    decision.raw_response = raw
    return True
