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
UNTRUSTED_EXPERT_TIMING_STATUSES = {
    "batch_fallback",
    "partial_batch_fallback",
    "circuit_breaker_fallback",
    "fast_prefilter",
    "failed",
    "invalid",
    "timeout",
}


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
        expert_block_reason = expert_analysis_entry_block_reason(decision)
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


def expert_analysis_entry_block_reason(decision: DecisionOutput) -> str | None:
    """Block market entries unless all required experts returned real analysis."""

    if not decision.is_entry:
        return None
    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
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

    missing_seen = sorted(REQUIRED_ENTRY_EXPERTS - seen)
    details = untrusted[:5] or [f"{name}:missing" for name in missing_seen[:5]]
    return (
        "专家分析完整性保护[expert_integrity]：新开仓必须拿到 5 个专家的真实大模型分析；"
        f"当前缺少可信专家 {', '.join(missing)}"
        f"（异常：{'; '.join(details) if details else '无耗时记录'}），本次不开仓。"
    )
