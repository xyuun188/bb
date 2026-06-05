"""
LLM-based trading agent using OpenAI-compatible API.
Uses LangChain for structured prompting and output parsing.

Each instance carries its own name and API config, enabling
multiple independently-configured LLM agents to trade side-by-side.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput
from ai_brain.prompts import (
    build_user_prompt,
    build_expert_user_prompt,
    build_close_prompt,
    build_decision_maker_user_prompt,
    build_batch_experts_user_prompt,
    DECISION_MAKER_SYSTEM_PROMPT,
    get_compact_role_system_prompt,
    get_role_system_prompt,
)
from config.settings import settings
from core.exceptions import LLMResponseParseError, ModelInferenceError

logger = structlog.get_logger(__name__)

# Global semaphore limits active LLM calls. A single local 32B model cannot
# reliably answer five expert prompts at once, so the default is intentionally
# lower than the number of experts.
_LLM_SEMAPHORE = asyncio.Semaphore(max(int(settings.ai_llm_concurrency or 5), 1))
_LLM_CALL_DELAY = max(float(settings.ai_llm_call_delay_seconds or 0.0), 0.0)

ROLE_TO_CROSS_TARGET = {
    "trend_direction": "trend",
    "profit_quality": "momentum",
    "short_timeseries": "sentiment",
    "position_exit": "position",
    "risk_anomaly": "risk",
    "technical_trend": "trend",
    "short_term_momentum": "momentum",
    "sentiment_news": "sentiment",
    "position_manager": "position",
    "risk_guardian": "risk",
}
VALID_CROSS_TARGETS = set(ROLE_TO_CROSS_TARGET.values())


def _snapshot_float(snapshot: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(snapshot.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _trend_aligned(decision: DecisionOutput) -> bool:
    snapshot = decision.feature_snapshot or {}
    price_vs_sma20 = _snapshot_float(snapshot, "price_vs_sma20")
    price_vs_sma50 = _snapshot_float(snapshot, "price_vs_sma50")
    if decision.action == Action.LONG:
        return price_vs_sma20 > 0 and price_vs_sma50 > 0
    if decision.action == Action.SHORT:
        return price_vs_sma20 < 0 and price_vs_sma50 < 0
    return False


def _entry_filters_pass(decision: DecisionOutput) -> bool:
    snapshot = decision.feature_snapshot or {}
    volume_ratio = _snapshot_float(snapshot, "volume_ratio", 1.0)
    adx_14 = _snapshot_float(snapshot, "adx_14")
    confirmations = [
        volume_ratio >= settings.min_entry_volume_ratio,
        adx_14 >= settings.min_entry_adx,
        _trend_aligned(decision),
    ]
    return sum(1 for ok in confirmations if ok) >= 2


def _directional_edge(snapshot: dict[str, Any]) -> tuple[Action, int, list[str]]:
    """Score the market snapshot and return the stronger directional edge."""
    long_score = 0
    short_score = 0
    long_reasons: list[str] = []
    short_reasons: list[str] = []

    price_vs_sma20 = _snapshot_float(snapshot, "price_vs_sma20")
    price_vs_sma50 = _snapshot_float(snapshot, "price_vs_sma50")
    macd_diff = _snapshot_float(snapshot, "macd_diff")
    ema_12 = _snapshot_float(snapshot, "ema_12")
    ema_26 = _snapshot_float(snapshot, "ema_26")
    rsi_14 = _snapshot_float(snapshot, "rsi_14", 50.0)
    stoch_k = _snapshot_float(snapshot, "stoch_k", 50.0)
    returns_5 = _snapshot_float(snapshot, "returns_5")
    returns_20 = _snapshot_float(snapshot, "returns_20")
    bb_pct = _snapshot_float(snapshot, "bb_pct", 0.5)
    adx_14 = _snapshot_float(snapshot, "adx_14")
    volume_ratio = _snapshot_float(snapshot, "volume_ratio", 1.0)

    if price_vs_sma20 > 0:
        long_score += 1
        long_reasons.append("价格站上 SMA20")
    elif price_vs_sma20 < 0:
        short_score += 1
        short_reasons.append("价格跌破 SMA20")

    if price_vs_sma50 > 0:
        long_score += 1
        long_reasons.append("价格站上 SMA50")
    elif price_vs_sma50 < 0:
        short_score += 1
        short_reasons.append("价格跌破 SMA50")

    if macd_diff > 0:
        long_score += 1
        long_reasons.append("MACD 动能偏多")
    elif macd_diff < 0:
        short_score += 1
        short_reasons.append("MACD 动能偏空")

    if ema_12 > 0 and ema_26 > 0:
        if ema_12 > ema_26:
            long_score += 1
            long_reasons.append("EMA12 高于 EMA26")
        elif ema_12 < ema_26:
            short_score += 1
            short_reasons.append("EMA12 低于 EMA26")

    if returns_5 > 0 and returns_20 > 0:
        long_score += 1
        long_reasons.append("短中期收益为正")
    elif returns_5 < 0 and returns_20 < 0:
        short_score += 1
        short_reasons.append("短中期收益为负")

    if 52 <= rsi_14 <= 72:
        long_score += 1
        long_reasons.append("RSI 处于多头但未极端区间")
    elif 28 <= rsi_14 <= 48:
        short_score += 1
        short_reasons.append("RSI 处于空头但未极端区间")

    if stoch_k >= 55:
        long_score += 1
        long_reasons.append("随机指标偏强")
    elif stoch_k <= 45:
        short_score += 1
        short_reasons.append("随机指标偏弱")

    if 0.55 <= bb_pct <= 0.88:
        long_score += 1
        long_reasons.append("价格位于布林中上轨")
    elif 0.12 <= bb_pct <= 0.45:
        short_score += 1
        short_reasons.append("价格位于布林中下轨")

    if adx_14 >= settings.min_entry_adx:
        if long_score >= short_score:
            long_score += 1
            long_reasons.append("ADX 支持趋势交易")
        else:
            short_score += 1
            short_reasons.append("ADX 支持趋势交易")

    if volume_ratio >= settings.min_entry_volume_ratio:
        if long_score >= short_score:
            long_score += 1
            long_reasons.append("成交量达到入场要求")
        else:
            short_score += 1
            short_reasons.append("成交量达到入场要求")

    if rsi_14 > 78 and bb_pct > 0.9:
        long_score -= 2
    if rsi_14 < 22 and bb_pct < 0.1:
        short_score -= 2

    if long_score >= short_score:
        return Action.LONG, long_score - short_score, long_reasons[:4]
    return Action.SHORT, short_score - long_score, short_reasons[:4]


def _format_expert_memories(memories: list[dict[str, Any]]) -> str:
    lines = ["专家长期记忆：只作为风险教训和筛选参考，不是强制交易指令。"]
    for idx, item in enumerate(memories[: max(int(settings.expert_memory_per_prompt or 4), 1)], start=1):
        lesson = str(item.get("lesson") or "").strip()
        pattern = str(item.get("market_pattern") or "").strip()
        action = str(item.get("recommended_action") or "reduce_risk").strip()
        adjustment = item.get("confidence_adjustment")
        multiplier = item.get("position_size_multiplier")
        evidence = item.get("evidence_count")
        parts = []
        if pattern:
            parts.append(f"场景={pattern[:60]}")
        if lesson:
            parts.append(f"教训={lesson[:100]}")
        parts.append(f"建议={action}")
        if adjustment is not None:
            parts.append(f"信心调整={float(adjustment):+.0%}")
        if multiplier is not None:
            parts.append(f"仓位系数={float(multiplier):.2f}")
        if evidence is not None:
            parts.append(f"证据={int(evidence)}")
        lines.append(f"{idx}. " + "；".join(parts))
    return "\n".join(lines)


def _format_daily_target(target: dict[str, Any]) -> str:
    target_usdt = float(target.get("target_usdt") or 0.0)
    target_currency = str(target.get("target_currency") or "USDT").upper()
    target_cny = float(target.get("target_cny") or 0.0)
    today = float(target.get("today_realized_pnl") or 0.0)
    gap = float(target.get("gap_usdt") or 0.0)
    target_label = (
        f"{target_usdt:.2f} USDT"
        if target_currency == "USDT"
        else f"{target_cny:.0f} CNY / {target_usdt:.2f} USDT"
    )
    return (
        "每日目标上下文："
        f"目标约 {target_label}，"
        f"今日已实现 {today:.2f} USDT，差距 {gap:.2f} USDT。"
        "此目标只能帮助优先选择高质量机会，不能放松风控、追单或无依据提高杠杆。"
    )


def _format_market_regime(regime: dict[str, Any]) -> str:
    if not isinstance(regime, dict) or not regime:
        return ""
    mode = str(regime.get("mode") or "unknown")
    confidence = float(regime.get("confidence") or 0.0)
    reason = str(regime.get("reason") or "")
    soft_bias = []
    if regime.get("avoid_long"):
        soft_bias.append("long side needs stronger symbol-specific confirmation")
    if regime.get("avoid_short"):
        soft_bias.append("short side needs stronger symbol-specific confirmation")
    bias_text = ", ".join(soft_bias) if soft_bias else "no directional bias"
    return (
        "Market regime forecast: "
        f"mode={mode}, confidence={confidence:.0%}, {bias_text}. "
        f"{reason} "
        "This is background only, not a long/short ban. Judge every symbol independently and actively compare both long and short expected profit. "
        "If a symbol has clear downside momentum, weak rebounds, negative order-book pressure, or better short expected return, a short is valid even in a rebound regime."
    )


def _format_strategy_mode(strategy: dict[str, Any]) -> str:
    if not isinstance(strategy, dict) or not strategy:
        return ""
    blocked = strategy.get("blocked_directions")
    if isinstance(blocked, (list, tuple, set)):
        blocked_text = ",".join(str(item) for item in blocked) or "none"
    else:
        blocked_text = str(blocked or "none")
    exposure = strategy.get("position_exposure")
    exposure_text = ""
    if isinstance(exposure, dict) and exposure:
        exposure_text = (
            "Current exposure: "
            f"long_notional={float(exposure.get('long_notional') or 0.0):.2f}, "
            f"short_notional={float(exposure.get('short_notional') or 0.0):.2f}, "
            f"net_ratio={float(exposure.get('net_ratio') or 0.0):.2f}, "
            f"long_count={int(exposure.get('long_count') or 0)}, "
            f"short_count={int(exposure.get('short_count') or 0)}, "
            f"dominant_side={exposure.get('dominant_side') or 'neutral'}. "
        )
    return (
        "Execution strategy mode: "
        f"strategy={strategy.get('strategy') or 'unknown'}, "
        f"posture={strategy.get('posture') or 'balanced'}, "
        f"allow_long={bool(strategy.get('allow_long', True))}, "
        f"allow_short={bool(strategy.get('allow_short', True))}, "
        f"blocked_directions={blocked_text}. "
        f"{exposure_text}"
        f"Reason: {strategy.get('reason') or ''} "
        "Market regime is a soft bias only: it must not block a side and must not create global same-side entries. "
        "Always evaluate long and short independently for this symbol; choose short when downside expected profit is better. "
        "Portfolio exposure is information for sizing and hedging, not a hard ban; add to a side when this symbol's expected profit justifies it. "
        "Each symbol should be judged on trend, momentum, risk/reward, and execution quality without waiting for a perfect checklist. "
        "Final goal is realized net profit maximization: do not chase losses, but act decisively on high-quality symbol-specific opportunities."
    )


def _format_local_ai_tools(tools: dict[str, Any]) -> str:
    if not isinstance(tools, dict) or not tools or not tools.get("enabled"):
        return ""
    status = str(tools.get("status") or "unknown")
    parts = [f"Local AI quant tools: status={status}."]

    profit = tools.get("profit_prediction")
    if isinstance(profit, dict) and profit.get("available", True) is not False:
        best_side = profit.get("best_side") or profit.get("side") or profit.get("direction")
        expected = profit.get("expected_return_pct", profit.get("best_expected_return_pct"))
        edge = profit.get("profit_edge_pct", profit.get("edge_pct"))
        quality = profit.get("profit_quality_score", profit.get("score"))
        parts.append(
            "Profit model: "
            f"best_side={best_side or 'unknown'}, "
            f"expected_return_pct={_fmt_num(expected, 4)}, "
            f"edge_pct={_fmt_num(edge, 4)}, "
            f"quality={_fmt_num(quality, 4)}."
        )

    ts = tools.get("time_series_prediction")
    if isinstance(ts, dict) and ts.get("available", True) is not False:
        trend = ts.get("trend") or ts.get("direction") or ts.get("forecast_direction")
        move = ts.get("expected_move_pct", ts.get("forecast_return_pct"))
        confidence = ts.get("confidence", ts.get("score"))
        parts.append(
            "Time-series model: "
            f"direction={trend or 'unknown'}, "
            f"expected_move_pct={_fmt_num(move, 4)}, "
            f"confidence={_fmt_num(confidence, 3)}."
        )

    sentiment = tools.get("sentiment_analysis")
    if isinstance(sentiment, dict) and sentiment.get("available", True) is not False:
        label = sentiment.get("label") or sentiment.get("sentiment") or sentiment.get("direction")
        score = sentiment.get("score", sentiment.get("sentiment_score"))
        risk = sentiment.get("risk_level", sentiment.get("risk"))
        parts.append(
            "Sentiment model: "
            f"label={label or 'neutral'}, score={_fmt_num(score, 3)}, risk={risk or 'unknown'}."
        )

    exit_advice = tools.get("exit_advice")
    if isinstance(exit_advice, dict) and exit_advice.get("available", True) is not False:
        action = exit_advice.get("action") or "unknown"
        urgency = exit_advice.get("urgency")
        reason = exit_advice.get("reason") or ""
        parts.append(
            "Exit model: "
            f"action={action}, urgency={_fmt_num(urgency, 3)}, reason={reason[:180]}."
        )

    parts.append(
        "Use these as profit-first evidence: expected realized return and downside risk matter more than raw win rate. "
        "If local tools are unavailable or weak, fall back to market features and expert judgment."
    )
    return " ".join(parts)


def _format_entry_candidate_evidence(evidence: dict[str, Any]) -> str:
    if not isinstance(evidence, dict) or not evidence.get("enabled"):
        return ""

    def side_line(side: str) -> str:
        item = evidence.get(side)
        if not isinstance(item, dict):
            return f"{side}=unavailable"
        return (
            f"{side}: score={_fmt_num(item.get('score'), 3)}, "
            f"min_ref={_fmt_num(item.get('min_score_reference'), 3)}, "
            f"expected_net={_fmt_num(item.get('expected_net_return_pct'), 4)}%, "
            f"loss_prob={_fmt_num(float(item.get('loss_probability') or 0) * 100, 1)}%, "
            f"profit_quality={_fmt_num(item.get('profit_quality_ratio'), 3)}, "
            f"tail_risk={_fmt_num(item.get('tail_risk_score'), 3)}, "
            f"high_profit={bool(item.get('high_profit_potential'))}, "
            f"history={str(item.get('historical_reason') or '')[:90]}, "
            f"recommendation={item.get('recommendation') or 'unknown'}"
        )

    return (
        "Pre-AI entry candidate evidence: "
        f"preferred_side_by_evidence={evidence.get('preferred_side_by_evidence') or 'neutral'}, "
        f"feature_score={_fmt_num(evidence.get('feature_opportunity_score'), 2)}. "
        f"{side_line('long')} | {side_line('short')}. "
        "This evidence is for AI judgment, not a hard execution veto. Compare long/short expected net profit, "
        "loss probability, payoff quality, realized history and tail risk before choosing action, size and leverage. "
        "If high_profit=true and the thesis is clear, larger size and higher leverage are allowed; otherwise keep risk small."
    )


def _format_portfolio_profit_protection(context: dict[str, Any]) -> str:
    if not isinstance(context, dict) or not context.get("active"):
        return ""
    current = context.get("current_group") if isinstance(context.get("current_group"), dict) else {}
    top_groups = context.get("top_groups") if isinstance(context.get("top_groups"), list) else []
    top_text = "; ".join(
        f"{item.get('symbol')} pnl={_fmt_num(item.get('unrealized_pnl'), 2)}U share={_fmt_num(float(item.get('profit_share') or 0) * 100, 1)}%"
        for item in top_groups[:3]
        if isinstance(item, dict)
    )
    return (
        "Portfolio profit protection: ACTIVE. "
        f"total_unrealized={_fmt_num(context.get('total_unrealized_pnl'), 2)}U, "
        f"positive_unrealized={_fmt_num(context.get('total_positive_unrealized_pnl'), 2)}U, "
        f"threshold={_fmt_num(context.get('threshold_usdt'), 2)}U. "
        f"Current symbol contribution: {current.get('symbol') or 'unknown'} "
        f"pnl={_fmt_num(current.get('unrealized_pnl'), 2)}U, "
        f"profit_pct={_fmt_num(float(current.get('profit_pct') or 0) * 100, 2)}%, "
        f"focus={bool(context.get('is_focus'))}. "
        f"Top contributors: {top_text or 'none'}. "
        "For this position review, explicitly choose continue_hold, partial_lock_profit, or full_close. "
        "Hold is valid only if continuation edge is still better than locking realized profit now."
    )


def _apply_aggressive_hold_policy(
    decision: DecisionOutput,
    symbol_positions: list[dict],
) -> None:
    if not decision.is_hold:
        return

    snapshot = decision.feature_snapshot or {}
    edge_action, edge, reasons = _directional_edge(snapshot)
    if edge < 2:
        return

    existing_same_symbol = [p for p in symbol_positions if p.get("side") in ("long", "short")]
    if existing_same_symbol:
        current_side = existing_same_symbol[0].get("side")
        if current_side == "long" and edge_action == Action.SHORT and edge >= 3:
            decision.action = Action.CLOSE_LONG
            decision.position_size_pct = 1.0
            decision.confidence = max(decision.confidence, 0.62)
            decision.reasoning += " [进攻型改写：观望但空头反转信号较强，改为平多]"
        elif current_side == "short" and edge_action == Action.LONG and edge >= 3:
            decision.action = Action.CLOSE_SHORT
            decision.position_size_pct = 1.0
            decision.confidence = max(decision.confidence, 0.62)
            decision.reasoning += " [进攻型改写：观望但多头反转信号较强，改为平空]"
        elif ((current_side == "long" and edge_action == Action.LONG)
              or (current_side == "short" and edge_action == Action.SHORT)) and edge >= 5:
            decision.action = edge_action
            decision.position_size_pct = 0.03
            decision.confidence = max(decision.confidence, 0.55)
            decision.reasoning += " [进攻型改写：观望但同向趋势延续很强，允许小幅加仓]"
        return

    decision.action = edge_action
    decision.confidence = max(decision.confidence, 0.52 + min(edge, 5) * 0.03)
    decision.position_size_pct = 0.04 if edge < 4 else 0.07
    decision.suggested_leverage = 5.0 if decision.confidence < 0.68 else (10.0 if decision.confidence >= 0.78 else 7.0)
    decision.stop_loss_pct = min(max(decision.stop_loss_pct or 0.03, 0.02), 0.06)
    decision.take_profit_pct = max(decision.take_profit_pct or 0.06, decision.stop_loss_pct * 1.6)
    decision.reasoning += (
        " [进攻型改写：原始输出为观望，但技术边际足够清楚，"
        f"改为{edge_action.value}试探；依据：{'、'.join(reasons)}]"
    )


def _leverage_cap_for_entry(decision: DecisionOutput) -> float:
    if decision.confidence < 0.68:
        return min(5.0, settings.max_leverage)
    if decision.confidence < 0.78:
        return min(10.0, settings.max_leverage)
    return settings.max_leverage


def _apply_entry_leverage_policy(decision: DecisionOutput) -> None:
    if not decision.is_entry:
        return

    if decision.confidence < 0.50:
        decision.action = Action.HOLD
        decision.position_size_pct = 0.0
        decision.suggested_leverage = 1.0
        decision.reasoning += " [置信度低于 0.50，未达到试探开仓要求，改为观望]"
        return

    original_leverage = decision.suggested_leverage
    leverage_cap = _leverage_cap_for_entry(decision)
    if decision.confidence >= 0.78:
        min_leverage = 10.0
    elif decision.confidence >= 0.68:
        min_leverage = 5.0
    else:
        min_leverage = 1.0
    decision.suggested_leverage = min(max(decision.suggested_leverage, min_leverage), leverage_cap)

    if original_leverage != decision.suggested_leverage:
        if decision.confidence < 0.68:
            decision.reasoning += " [杠杆规则：普通信号最高 5x]"
        elif decision.confidence < 0.78:
            decision.reasoning += " [杠杆规则：高质量信号使用 5-10x]"
        else:
            decision.reasoning += " [杠杆规则：强信号最低 10x，最高使用系统最大杠杆]"


def _extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM output that may contain markdown or extra text.

    Tries multiple strategies in order:
    1. Direct JSON parse
    2. Extract from ```json ... ``` code fence
    3. Find first { and last } and parse that substring
    """
    text = _strip_qwen_thinking(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from code fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise LLMResponseParseError(f"Could not extract valid JSON from: {text[:300]}")


def _normalize_cross_check(value: Any, own_role: str) -> dict[str, str] | None:
    """Validate and normalize an expert cross-check request."""
    if not isinstance(value, dict):
        return None

    target = str(value.get("target", "")).strip().lower()
    aliases = {
        "trend_expert": "trend",
        "technical_trend": "trend",
        "trend_direction": "trend",
        "momentum_expert": "momentum",
        "short_term_momentum": "momentum",
        "profit_quality": "momentum",
        "sentiment_expert": "sentiment",
        "sentiment_news": "sentiment",
        "short_timeseries": "sentiment",
        "position_expert": "position",
        "position_manager": "position",
        "position_exit": "position",
        "risk_expert": "risk",
        "risk_guardian": "risk",
        "risk_anomaly": "risk",
    }
    target = aliases.get(target, target)
    own_target = ROLE_TO_CROSS_TARGET.get(own_role)
    question = str(value.get("question", "")).strip()

    if target not in VALID_CROSS_TARGETS or target == own_target:
        return None
    if len(question) < 12:
        return None
    return {"target": target, "question": question[:500]}


def _fmt_num(value: Any, digits: int = 4) -> str: 
    try: 
        return f"{float(value):.{digits}f}" 
    except (TypeError, ValueError): 
        return "0" 


def _is_reasoning_model(model: str | None) -> bool:
    name = str(model or "").lower()
    return name.startswith(("o1", "o3", "o4"))


def _is_qwen3_model(model: str | None) -> bool:
    return "qwen3" in str(model or "").lower()


def _uses_thinking_tags(model: str | None) -> bool:
    name = str(model or "").lower()
    return "qwen3" in name or "deepseek-r1" in name


def _backup_model_names(model: str | None) -> list[str]:
    """Provider-compatible backups used only when the configured model fails."""
    current = str(model or "").strip()
    if current.startswith("qwen3-") and current.endswith("-trade"):
        return []
    candidates = ["qwen3-max", "deepseek-v3", "claude-opus-4-7"]
    return [m for m in candidates if m and m != current][:2]


def _message_content_text(response: Any) -> str:
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, str):
        return _strip_qwen_thinking(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return _strip_qwen_thinking("\n".join(p for p in parts if p).strip())
    return _strip_qwen_thinking(str(content or ""))


def _strip_qwen_thinking(text: str) -> str:
    """Remove Qwen3 thinking blocks before JSON parsing."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(text or ""), flags=re.IGNORECASE).strip()
    if cleaned.startswith("<think>") and "{" in cleaned:
        cleaned = cleaned[cleaned.find("{"):].strip()
    return cleaned


def _sentiment_signal_is_empty(features: "FeatureVector") -> bool:
    """True when the sentiment expert has no actual sentiment evidence to analyze."""
    headlines = getattr(features, "recent_headlines", None) or []
    news_items = [
        item for item in (getattr(features, "recent_news_items", None) or [])
        if isinstance(item, dict) and item.get("title")
    ]
    sources = getattr(features, "news_sources", None) or []
    try:
        mention_count = int(float(getattr(features, "social_mention_count", 0) or 0))
    except (TypeError, ValueError):
        mention_count = 0
    try:
        article_count = int(float(getattr(features, "news_article_count", 0) or 0))
    except (TypeError, ValueError):
        article_count = 0
    news_sentiment = abs(_snapshot_float(features.to_dict(), "news_sentiment_avg"))
    social_sentiment = abs(_snapshot_float(features.to_dict(), "social_sentiment_avg"))
    return (
        not headlines
        and not news_items
        and not sources
        and mention_count <= 0
        and article_count <= 0
        and news_sentiment < 0.03
        and social_sentiment < 0.03
    )


def _news_availability_note(features: "FeatureVector") -> str:
    direct_count = int(getattr(features, "direct_news_item_count", 0) or 0)
    market_count = int(getattr(features, "market_news_item_count", 0) or 0)
    direct_items = [
        item for item in (getattr(features, "recent_news_items", None) or [])
        if isinstance(item, dict) and item.get("direct_match") is True and item.get("title")
    ]
    if direct_items and direct_count <= 0:
        direct_count = len(direct_items)
    if direct_count > 0:
        titles = " | ".join(str(item.get("title") or "")[:70] for item in direct_items[:3])
        return (
            f"direct_news={direct_count}; "
            f"direct_news_titles={titles or 'listed in news_items'}; "
            "news_policy=direct symbol news exists, sentiment expert must assess its positive/negative/risk impact and must not say no direct news"
        )
    if market_count > 0:
        return (
            f"direct_news=0; market_background_news={market_count}; "
            "news_policy=no direct symbol news, treat sentiment as neutral for entry and use market news only as broad risk context"
        )
    return "direct_news=0; market_background_news=0; news_policy=no news, treat sentiment as neutral; do not block entry if technical/ML/time-series edge is strong"


def _build_compact_feature_context(features: "FeatureVector", role: str) -> str:
    """Role-filtered feature text to reduce expert prompt tokens."""
    base = (
        f"symbol={features.symbol}; price={_fmt_num(features.current_price)}; "
        f"change24h={_fmt_num(features.change_24h_pct, 2)}%; "
        f"bid={_fmt_num(features.bid)}; ask={_fmt_num(features.ask)}; "
        f"spread={_fmt_num(getattr(features, 'spread_pct', 0), 4)}%; "
        f"abnormal_wick_count72h={int(getattr(features, 'abnormal_wick_count_72h', 0) or 0)}; "
        f"abnormal_wick_max={_fmt_num(getattr(features, 'abnormal_wick_max_pct', 0), 2)}%; "
        f"abnormal_wick_recent_h={_fmt_num(getattr(features, 'abnormal_wick_recent_hours', 9999), 1)}"
    )
    derivatives = (
        f"funding={_fmt_num(getattr(features, 'funding_rate', 0), 6)}; "
        f"oi={_fmt_num(getattr(features, 'open_interest_value', 0), 2)}; "
        f"ob_imbalance={_fmt_num(getattr(features, 'orderbook_imbalance', 0), 3)}"
    )

    if role in {"technical_trend", "trend_direction"}:
        return (
            f"{base}; rsi14={_fmt_num(features.rsi_14, 1)}; macd_diff={_fmt_num(features.macd_diff, 6)}; "
            f"ema12={_fmt_num(features.ema_12)}; ema26={_fmt_num(features.ema_26)}; "
            f"adx14={_fmt_num(features.adx_14, 1)}; bb_pct={_fmt_num(features.bb_pct, 2)}; "
            f"bb_width={_fmt_num(features.bb_width, 4)}; price_vs_sma20={_fmt_num(features.price_vs_sma20, 4)}; "
            f"price_vs_sma50={_fmt_num(features.price_vs_sma50, 4)}; {derivatives}"
        )
    if role in {"short_term_momentum", "profit_quality"}:
        return (
            f"{base}; returns1={_fmt_num(features.returns_1, 4)}; returns5={_fmt_num(features.returns_5, 4)}; "
            f"returns20={_fmt_num(features.returns_20, 4)}; volume_ratio={_fmt_num(features.volume_ratio, 2)}; "
            f"vol20={_fmt_num(features.volatility_20, 4)}; atr14={_fmt_num(features.atr_14, 4)}; "
            f"bb_pct={_fmt_num(features.bb_pct, 2)}; rsi14={_fmt_num(features.rsi_14, 1)}; "
            f"adx14={_fmt_num(features.adx_14, 1)}; {derivatives}"
        )
    if role in {"sentiment_news", "short_timeseries"}:
        headlines = " | ".join((features.recent_headlines or [])[:3])
        news_items = getattr(features, "recent_news_items", None) or []
        news_detail = " | ".join(
            f"{item.get('source','-')}:{item.get('event_type','news')}:impact{item.get('impact_level',1)}:"
            f"sent{_fmt_num(item.get('sentiment_score', 0), 2)}:{item.get('title','')[:80]}"
            for item in news_items[:4]
            if isinstance(item, dict)
        )
        return (
            f"{base}; returns1={_fmt_num(features.returns_1, 4)}; returns5={_fmt_num(features.returns_5, 4)}; "
            f"returns20={_fmt_num(features.returns_20, 4)}; volume_ratio={_fmt_num(features.volume_ratio, 2)}; "
            f"vol20={_fmt_num(features.volatility_20, 4)}; atr14={_fmt_num(features.atr_14, 4)}; "
            f"news_sent={_fmt_num(features.news_sentiment_avg, 3)}; "
            f"social_sent={_fmt_num(features.social_sentiment_avg, 3)}; "
            f"mentions={features.social_mention_count}; articles={getattr(features, 'news_article_count', 0)}; "
            f"{_news_availability_note(features)}; "
            f"sentiment_available={getattr(features, 'sentiment_data_available', False)}; "
            f"sources={','.join(getattr(features, 'news_sources', [])[:3]) or 'none'}; "
            f"headlines={headlines or 'none'}; news_items={news_detail or 'none'}"
        )
    if role in {"position_manager", "position_exit"}:
        return (
            f"{base}; returns5={_fmt_num(features.returns_5, 4)}; returns20={_fmt_num(features.returns_20, 4)}; "
            f"rsi14={_fmt_num(features.rsi_14, 1)}; volume_ratio={_fmt_num(features.volume_ratio, 2)}; "
            f"vol20={_fmt_num(features.volatility_20, 4)}; news_sent={_fmt_num(features.news_sentiment_avg, 3)}; "
            f"{derivatives}"
        )
    if role in {"risk_guardian", "risk_anomaly"}:
        return (
            f"{base}; volume24h={_fmt_num(features.volume_24h, 2)}; volume_ratio={_fmt_num(features.volume_ratio, 2)}; "
            f"vol20={_fmt_num(features.volatility_20, 4)}; atr14={_fmt_num(features.atr_14, 4)}; "
            f"adx14={_fmt_num(features.adx_14, 1)}; news_sent={_fmt_num(features.news_sentiment_avg, 3)}; "
            f"social_sent={_fmt_num(features.social_sentiment_avg, 3)}; {derivatives}; "
            f"bid_depth={_fmt_num(getattr(features, 'orderbook_bid_depth', 0), 2)}; "
            f"ask_depth={_fmt_num(getattr(features, 'orderbook_ask_depth', 0), 2)}"
        )
    if role == "final_decision":
        return (
            f"{base}; rsi14={_fmt_num(features.rsi_14, 1)}; macd_diff={_fmt_num(features.macd_diff, 6)}; "
            f"adx14={_fmt_num(features.adx_14, 1)}; volume_ratio={_fmt_num(features.volume_ratio, 2)}; "
            f"returns1={_fmt_num(features.returns_1, 4)}; returns5={_fmt_num(features.returns_5, 4)}; "
            f"returns20={_fmt_num(features.returns_20, 4)}; bb_pct={_fmt_num(features.bb_pct, 2)}; "
            f"price_vs_sma20={_fmt_num(features.price_vs_sma20, 4)}; "
            f"price_vs_sma50={_fmt_num(features.price_vs_sma50, 4)}; "
            f"news_sent={_fmt_num(features.news_sentiment_avg, 3)}; "
            f"social_sent={_fmt_num(features.social_sentiment_avg, 3)}; "
            f"{_news_availability_note(features)}; {derivatives}"
        )
    return features.to_llm_context()


def _calibrate_sentiment_decision(features: "FeatureVector", decision: DecisionOutput) -> DecisionOutput:
    """Keep sentiment text consistent with structured direct-news fields."""
    if decision.model_name != "sentiment_expert":
        return decision
    direct_items = [
        item for item in (getattr(features, "recent_news_items", None) or [])
        if isinstance(item, dict) and item.get("direct_match") is True and item.get("title")
    ]
    direct_count = int(getattr(features, "direct_news_item_count", 0) or 0)
    if direct_items and direct_count <= 0:
        direct_count = len(direct_items)
    if direct_count <= 0:
        return decision

    reasoning = str(decision.reasoning or "")
    conflict_patterns = ("无直接新闻", "没有直接新闻", "无直接相关新闻", "缺少直接新闻")
    if not any(pattern in reasoning for pattern in conflict_patterns):
        return decision

    news_sentiment = float(getattr(features, "news_sentiment_avg", 0.0) or 0.0)
    social_sentiment = float(getattr(features, "social_sentiment_avg", 0.0) or 0.0)
    titles = "；".join(str(item.get("title") or "")[:28] for item in direct_items[:2])
    direction = "偏利好" if news_sentiment > 0.12 else ("偏利空/风险" if news_sentiment < -0.12 else "整体中性")
    calibrated = (
        f"有{direct_count}条直接相关新闻，新闻情绪{news_sentiment:.2f}，社媒{social_sentiment:.2f}，"
        f"影响{direction}；代表新闻：{titles or '见新闻明细'}。"
    )
    decision.reasoning = calibrated[:150]
    raw = dict(decision.raw_response or {})
    raw["reasoning_before_news_calibration"] = reasoning
    raw["news_calibrated"] = True
    raw["direct_news_item_count"] = direct_count
    raw["news_sentiment_avg"] = news_sentiment
    decision.raw_response = raw
    return decision


BATCH_EXPERT_SYSTEM_PROMPT = """You are a five-expert crypto trading committee in one local model call.
Return ONLY one complete compact JSON object with key "experts".
No markdown, no code fences, no <think>, no prose outside JSON.
Each reasoning field must be concise but useful: 45-80 Chinese characters covering evidence, main risk, profit quality, and action rationale."""


class LLMAgent(AbstractAIModel):
    """Trading agent backed by an LLM via OpenAI-compatible API.

    Sends structured market context + prompts, receives JSON trading decisions.

    Each instance can have a unique name and API config, enabling multiple
    independently-configured LLM agents to trade in parallel.
    """

    name: str = "llm_agent"  # default, overridable per instance

    def __init__(self, name: str | None = None, api_config: dict | None = None) -> None: 
        if name is not None: 
            self.name = name 
        self._api_config = api_config  # {"api_base": ..., "api_key": ..., "model": ...} 
        self._role = (api_config or {}).get("role", "") 
        self._label = (api_config or {}).get("label", self.name) 
        self.weight = float((api_config or {}).get("weight", 1.0) or 1.0) 
        self._base_url = "" 
        self._api_key = "" 
        self._model_name = "" 
        self._llm: ChatOpenAI | None = None 
        self._max_retries = 1 

    async def initialize(self) -> None: 
        if self._api_config: 
            self._base_url = self._api_config.get("api_base") or settings.ai_api_base 
            self._api_key = self._api_config.get("api_key") or settings.ai_api_key 
            self._model_name = self._api_config.get("model") or settings.ai_model 
        else: 
            # Backward-compatible fallback to global settings 
            self._base_url = settings.ai_api_base 
            self._api_key = settings.ai_api_key 
            self._model_name = settings.ai_model 

        self._llm = self._create_llm(self._model_name) 
        logger.info("llm agent initialized", name=self.name, model=self._model_name, base=self._base_url) 

    def _create_llm(self, model: str, max_completion_tokens_override: int | None = None) -> ChatOpenAI:
        reasoning_model = _is_reasoning_model(model)
        configured_timeout = (
            settings.ai_decision_maker_timeout_seconds
            if self._role == "final_decision"
            else settings.ai_expert_timeout_seconds
        )
        request_timeout = max(
            float(configured_timeout or 0.0),
            45.0 if reasoning_model or _uses_thinking_tags(model) else 30.0,
        )
        configured_max_tokens = (
            settings.ai_decision_maker_max_completion_tokens
            if self._role == "final_decision"
            else settings.ai_expert_max_completion_tokens
        )
        max_completion_tokens = max(int(max_completion_tokens_override or configured_max_tokens or 0), 180)
        kwargs: dict[str, Any] = {
            "base_url": self._base_url,
            "api_key": self._api_key,
            "model": model,
            "timeout": request_timeout,
            "max_retries": 1,
        }
        if reasoning_model:
            kwargs["temperature"] = None
            kwargs["reasoning_effort"] = "low"
            kwargs["max_completion_tokens"] = max_completion_tokens
        else:
            kwargs["temperature"] = 0.2 if self._role else 0.3
            kwargs["max_tokens"] = max_completion_tokens
        if _uses_thinking_tags(model):
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return ChatOpenAI(**kwargs)

    async def decide(
        self, features: "FeatureVector", context: dict[str, Any]
    ) -> DecisionOutput:
        expert_mode = bool(context.get("expert_mode"))
        decision_maker_mode = bool(context.get("decision_maker_mode"))
        usage_stage = (
            "decision_maker"
            if decision_maker_mode
            else ("expert" if expert_mode else "direct_decision")
        )
        if (
            expert_mode
            and not decision_maker_mode
            and self._role == "sentiment_news"
            and _sentiment_signal_is_empty(features)
        ):
            return self._empty_sentiment_fast_path(features)

        if self._llm is None:
            await self.initialize()

        # Build the user prompt from features. Expert mode receives a compact
        # role-filtered context to keep multi-expert token cost under control.
        if decision_maker_mode:
            feature_text = _build_compact_feature_context(features, "final_decision")
        elif expert_mode:
            feature_text = _build_compact_feature_context(features, self._role)
        else:
            feature_text = features.to_llm_context()

        # In ensemble mode, expert models analyze the shared execution account.
        position_model_name = context.get("position_model_name", self.name)
        all_positions = context.get("open_positions", [])
        open_positions = [p for p in all_positions if p.get("model_name") == position_model_name]
        positions_text = ""
        if open_positions:
            lines = []
            relevant_positions = [
                p for p in open_positions
                if p.get("symbol") == features.symbol
            ] or open_positions[:3]
            for pos in relevant_positions[:3]:
                lines.append(
                    f"  - {pos['side'].upper()} {pos['symbol']}: "
                    f"entry={pos.get('entry_price', '?')}, "
                    f"current={pos.get('current_price', '?')}, "
                    f"pnl={_fmt_num(pos.get('unrealized_pnl', 0), 2)} USD, "
                    f"sl={pos.get('stop_loss') or 'none'}, "
                    f"tp={pos.get('take_profit') or 'none'}"
                )
            positions_text = "\n".join(lines)
        memories_text = ""
        expert_memories = (context.get("expert_memories") or {}).get(self.name, [])
        if expert_mode and expert_memories:
            memories_text = _format_expert_memories(expert_memories)
        target_text = ""
        daily_target = context.get("daily_target") if isinstance(context, dict) else {}
        if (
            expert_mode
            and isinstance(daily_target, dict)
            and daily_target.get("enabled") is not False
            and float(daily_target.get("target_usdt") or 0.0) > 0
        ):
            target_text = _format_daily_target(daily_target)
        regime_text = ""
        if expert_mode and context.get("market_regime"):
            regime_text = _format_market_regime(context.get("market_regime") or {})
        strategy_text = ""
        if expert_mode and context.get("strategy_mode"):
            strategy_text = _format_strategy_mode(context.get("strategy_mode") or {})
        local_tools_text = ""
        if (
            expert_mode
            and context.get("local_ai_tools")
            and context.get("local_ai_tools_prompt_enabled", True)
        ):
            local_tools_text = _format_local_ai_tools(context.get("local_ai_tools") or {})
        entry_candidate_text = ""
        if expert_mode and context.get("entry_candidate_evidence"):
            entry_candidate_text = _format_entry_candidate_evidence(
                context.get("entry_candidate_evidence") or {}
            )
        portfolio_profit_text = ""
        if expert_mode and context.get("portfolio_profit_protection"):
            portfolio_profit_text = _format_portfolio_profit_protection(
                context.get("portfolio_profit_protection") or {}
            )

        # If we have open positions for this symbol, ask specifically about closing
        symbol_positions = [p for p in open_positions if p.get("symbol") == features.symbol]
        if decision_maker_mode:
            user_prompt = build_decision_maker_user_prompt(feature_text, context)
        elif symbol_positions and context.get("review_positions", False) and not expert_mode:
            pos_desc = "\n".join(
                f"{p['side'].upper()}: entry={p.get('entry_price')}, current={features.current_price}, "
                f"pnl={features.current_price - p.get('entry_price', 0):.4f}"
                for p in symbol_positions
            )
            user_prompt = build_close_prompt(feature_text, pos_desc)
        elif expert_mode:
            user_prompt = build_expert_user_prompt(
                self._role,
                feature_text,
                "\n".join(part for part in (
                    positions_text,
                    memories_text,
                    target_text,
                    regime_text,
                    strategy_text,
                    entry_candidate_text,
                    local_tools_text,
                    portfolio_profit_text,
                ) if part),
                settings.confidence_threshold,
            )
        else:
            user_prompt = build_user_prompt(feature_text, positions_text, settings.confidence_threshold)

        messages = [
            SystemMessage(
                content=(
                    DECISION_MAKER_SYSTEM_PROMPT
                    if decision_maker_mode
                    else (
                        get_compact_role_system_prompt(self._role)
                        if expert_mode
                        else get_role_system_prompt(self._role, settings.confidence_threshold)
                    )
                )
            ),
            HumanMessage(content=(f"{user_prompt}\n/no_think" if _uses_thinking_tags(self._model_name) else user_prompt)),
        ]

        # Retry configured model first. In expert mode, try provider-compatible
        # backup AI models before falling back to conservative local rules.
        models_to_try = [self._model_name]
        if expert_mode and not decision_maker_mode:
            models_to_try.extend(_backup_model_names(self._model_name))

        last_error = ""
        primary_model = self._model_name
        for model_name in models_to_try:
            llm = self._llm if model_name == primary_model else self._create_llm(model_name)
            for attempt in range(self._max_retries + 1):
                try:
                    async with _LLM_SEMAPHORE:
                        if _LLM_CALL_DELAY:
                            await asyncio.sleep(_LLM_CALL_DELAY)
                        response = await llm.ainvoke(messages)
                    content = _message_content_text(response)
                    if not content.strip():
                        raise LLMResponseParseError(f"模型 {model_name} 返回空内容")

                    parsed = _extract_json(content)
                    parsed["provider_model"] = model_name
                    if model_name != primary_model:
                        parsed["fallback_from"] = primary_model

                    decision = self._decision_from_parsed(parsed, features, context)
                    if model_name != primary_model:
                        decision.reasoning += f" [备用模型：{primary_model} 无有效输出，改用 {model_name}]"

                    if not context.get("expert_mode") and not decision_maker_mode:
                        _apply_aggressive_hold_policy(decision, symbol_positions)

                    logger.info(
                        "llm decision",
                        name=self.name,
                        provider_model=model_name,
                        symbol=features.symbol,
                        action=decision.action.value,
                        confidence=decision.confidence,
                    )
                    return decision

                except LLMResponseParseError as e:
                    last_error = str(e)
                    logger.warning(
                        "llm parse error",
                        name=self.name,
                        model=model_name,
                        attempt=attempt,
                        error=last_error,
                    )
                except Exception as e:
                    err_msg = str(e)
                    if "model_dump" in err_msg or "JSONDecodeError" in err_msg or "Expecting value" in err_msg:
                        err_msg = f"API proxy returned empty or invalid response (模型名 '{model_name}' 可能不被该代理支持)"
                    last_error = err_msg
                    logger.error(
                        "llm api error",
                        name=self.name,
                        model=model_name,
                        attempt=attempt,
                        error=err_msg,
                    )

        if expert_mode and not decision_maker_mode:
            logger.warning("expert local fallback used", name=self.name, error=last_error)
        return self._local_expert_fallback(features, context, last_error)

    async def decide_batch_experts(
        self,
        features: "FeatureVector",
        context: dict[str, Any],
        expert_names: list[str],
    ) -> dict[str, DecisionOutput]:
        if self._llm is None:
            await self.initialize()
        if self._llm is None:
            raise ModelInferenceError(f"LLM agent {self.name} is not initialized")

        role_by_name = {
            "trend_expert": "trend_direction",
            "momentum_expert": "profit_quality",
            "sentiment_expert": "short_timeseries",
            "position_expert": "position_exit",
            "risk_expert": "risk_anomaly",
        }
        feature_text = _build_compact_feature_context(features, "final_decision")
        user_prompt = build_batch_experts_user_prompt(feature_text, context)
        messages = [
            SystemMessage(content=BATCH_EXPERT_SYSTEM_PROMPT),
            HumanMessage(content=(f"{user_prompt}\n/no_think" if _uses_thinking_tags(self._model_name) else user_prompt)),
        ]

        async with _LLM_SEMAPHORE:
            if _LLM_CALL_DELAY:
                await asyncio.sleep(_LLM_CALL_DELAY)
            batch_llm = self._create_llm(
                self._model_name,
                max_completion_tokens_override=settings.ai_batch_expert_max_completion_tokens,
            )
            response = await batch_llm.ainvoke(messages)
        content = _message_content_text(response)
        parsed = _extract_json(content)
        experts_payload = parsed.get("experts") if isinstance(parsed, dict) else None
        if not isinstance(experts_payload, dict):
            raise LLMResponseParseError("batch experts response missing experts object")

        decisions: dict[str, DecisionOutput] = {}
        for name in expert_names:
            payload = experts_payload.get(name)
            if not isinstance(payload, dict):
                fallback = self._local_expert_fallback(features, {**context, "expert_mode": True}, f"batch missing {name}")
                fallback.model_name = name
                fallback.raw_response = {
                    **(fallback.raw_response or {}),
                    "provider_model": self._model_name,
                    "batch_expert_fallback": True,
                }
                decisions[name] = fallback
                continue

            payload = dict(payload)
            payload["provider_model"] = self._model_name
            payload["batch_expert"] = True
            payload["batch_source_model"] = self.name
            payload["cross_check_for"] = _normalize_cross_check(
                payload.get("cross_check_for"),
                role_by_name.get(name, ""),
            )
            decision = DecisionOutput(
                model_name=name,
                symbol=features.symbol,
                action=Action.from_string(str(payload.get("action", "hold"))),
                confidence=min(max(float(payload.get("confidence", 0.5) or 0.5), 0.0), 1.0),
                reasoning=str(payload.get("reasoning") or "暂无分析内容。")[:150],
                position_size_pct=min(max(float(payload.get("position_size_pct", 0.0) or 0.0), 0.0), 1.0),
                suggested_leverage=min(
                    max(float(payload.get("suggested_leverage", 1.0) or 1.0), 1.0),
                    settings.max_leverage,
                ),
                stop_loss_pct=min(max(float(payload.get("stop_loss_pct", 0.05) or 0.05), 0.01), 0.15),
                take_profit_pct=min(max(float(payload.get("take_profit_pct", 0.10) or 0.10), 0.02), 0.50),
                cross_check_for=payload.get("cross_check_for"),
                raw_response=payload,
                feature_snapshot=features.to_dict(),
            )
            decision = _calibrate_sentiment_decision(features, decision)
            decisions[name] = decision
        return decisions

        raise ModelInferenceError(f"LLM agent {self.name} failed: {last_error or 'unknown error'}")

    def _empty_sentiment_fast_path(self, features: "FeatureVector") -> DecisionOutput:
        """Return immediately when sentiment/news data is genuinely empty."""
        snapshot = features.to_dict()
        reason = (
            "新闻情绪快路径：本轮没有新闻、社媒提及或有效情绪分数，"
            "情绪按中性处理，不作为开仓阻碍；若技术面、ML和时序信号足够强，仍可开仓。"
        )
        logger.info(
            "sentiment expert fast path",
            name=self.name,
            symbol=features.symbol,
            reason="empty_sentiment_signal",
        )
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.10,
            reasoning=reason,
            position_size_pct=0.0,
            suggested_leverage=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            cross_check_for={
                "target": "trend",
                "question": "新闻为空时按中性处理，请核实技术/量价/ML边际是否足够独立支持开仓。",
            },
            raw_response={
                "provider_model": "local_empty_sentiment_fast_path",
                "local_fast_path": True,
                "reason": reason,
            },
            feature_snapshot=snapshot,
        )

    def _decision_from_parsed(
        self,
        parsed: dict[str, Any],
        features: "FeatureVector",
        context: dict[str, Any],
    ) -> DecisionOutput:
        cross_check_for = _normalize_cross_check(
            parsed.get("cross_check_for"),
            self._role,
        )
        parsed["cross_check_for"] = cross_check_for

        decision = DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.from_string(str(parsed.get("action", "hold"))),
            confidence=min(max(float(parsed.get("confidence", 0.5) or 0.5), 0.0), 1.0),
            reasoning=str(parsed.get("reasoning") or "暂无分析内容。"),
            position_size_pct=min(
                max(float(parsed.get("position_size_pct", 0.0) or 0.0), 0.0), 1.0
            ),
            suggested_leverage=min(
                max(float(parsed.get("suggested_leverage", 1.0) or 1.0), 1.0),
                settings.max_leverage,
            ),
            stop_loss_pct=min(
                max(float(parsed.get("stop_loss_pct", 0.05) or 0.05), 0.01), 0.15
            ),
            take_profit_pct=min(
                max(float(parsed.get("take_profit_pct", 0.10) or 0.10), 0.02), 0.50
            ),
            cross_check_for=cross_check_for,
            raw_response=parsed,
            feature_snapshot=features.to_dict(),
        )

        if False and decision.is_entry and decision.confidence < settings.confidence_threshold:
            if decision.confidence >= 0.45 and decision.position_size_pct > 0:
                decision.position_size_pct = min(decision.position_size_pct, 0.06)
                decision.reasoning += " [小仓位试探：置信度低于常规阈值，已限制仓位]"
            else:
                decision.action = Action.HOLD
                decision.position_size_pct = 0.0
                decision.reasoning += " [置信度低于 0.45 或仓位为 0，改为观望]"

        return decision

    def _local_expert_fallback(
        self,
        features: "FeatureVector",
        context: dict[str, Any],
        error: str,
    ) -> DecisionOutput:
        action = Action.HOLD
        confidence = 0.35
        size = 0.0
        leverage = 1.0
        stop_loss = 0.05
        take_profit = 0.10
        cross_check_for: dict[str, str] | None = None
        reason = "AI 模型暂时无有效输出，使用本地保守规则兜底。"

        snapshot = features.to_dict()
        edge_action, edge, edge_reasons = _directional_edge(snapshot)
        volume_ratio = _snapshot_float(snapshot, "volume_ratio", 1.0)
        adx_14 = _snapshot_float(snapshot, "adx_14")
        volatility = _snapshot_float(snapshot, "volatility_20")
        abnormal_wick_count = int(_snapshot_float(snapshot, "abnormal_wick_count_72h"))
        abnormal_wick_max = _snapshot_float(snapshot, "abnormal_wick_max_pct")
        abnormal_wick_recent = _snapshot_float(snapshot, "abnormal_wick_recent_hours", 9999.0)
        sentiment = (_snapshot_float(snapshot, "news_sentiment_avg") + _snapshot_float(snapshot, "social_sentiment_avg")) / 2

        if self._role in {"technical_trend", "trend_direction"}:
            confidence = min(0.42 + edge * 0.04, 0.62)
            if edge >= 3 and adx_14 >= settings.min_entry_adx:
                action = edge_action
                size = 0.03
                leverage = 1.0
            reason = f"趋势规则兜底：{'、'.join(edge_reasons) if edge_reasons else '趋势证据不足'}，ADX={adx_14:.1f}。"
            cross_check_for = {
                "target": "momentum",
                "question": "请核实当前成交量和短线动量是否支持趋势判断。",
            }
        elif self._role in {"short_term_momentum", "profit_quality"}:
            returns_1 = _snapshot_float(snapshot, "returns_1")
            returns_5 = _snapshot_float(snapshot, "returns_5")
            if returns_1 > 0 and returns_5 > 0 and volume_ratio >= settings.min_entry_volume_ratio:
                action = Action.LONG
                confidence = 0.52
                size = 0.03
            elif returns_1 < 0 and returns_5 < 0 and volume_ratio >= settings.min_entry_volume_ratio:
                action = Action.SHORT
                confidence = 0.52
                size = 0.03
            else:
                confidence = 0.38
            reason = f"动量规则兜底：1周期收益={returns_1:.4f}，5周期收益={returns_5:.4f}，量比={volume_ratio:.2f}。"
            cross_check_for = {
                "target": "trend",
                "question": "请核实短线动量方向是否与主要趋势一致。",
            }
        elif self._role in {"sentiment_news", "short_timeseries"}:
            if sentiment >= 0.35:
                action = Action.LONG
                confidence = 0.46
                size = 0.02
            elif sentiment <= -0.35:
                action = Action.SHORT
                confidence = 0.46
                size = 0.02
            else:
                confidence = 0.30
            reason = f"情绪规则兜底：新闻/社媒综合情绪={sentiment:.2f}，缺少强事件驱动。"
            cross_check_for = {
                "target": "risk",
                "question": "请核实是否存在未覆盖的突发风险事件。",
            }
        elif self._role in {"position_manager", "position_exit"}:
            symbol_positions = [
                p for p in context.get("open_positions", [])
                if p.get("symbol") == features.symbol
            ]
            if symbol_positions:
                side = symbol_positions[0].get("side")
                severe_reversal = edge >= 4 and volume_ratio >= settings.min_entry_volume_ratio
                if side == "long" and edge_action == Action.SHORT and severe_reversal:
                    action = Action.CLOSE_LONG
                    confidence = 0.62
                    size = 0.5
                elif side == "short" and edge_action == Action.LONG and severe_reversal:
                    action = Action.CLOSE_SHORT
                    confidence = 0.62
                    size = 0.5
                else:
                    confidence = 0.48
                reason = (
                    "持仓规则兜底：已有持仓，小幅浮亏不直接平仓；"
                    "只有趋势强反转且量能确认时才先减仓。"
                )
            else:
                confidence = 0.42
                reason = "持仓规则兜底：当前无持仓，未发现足够清晰的新开仓管理优势。"
            cross_check_for = {
                "target": "trend",
                "question": "请核实当前趋势是否足以支撑开仓或继续持仓。",
            }
        elif self._role in {"risk_guardian", "risk_anomaly"}:
            risk_flags = []
            if volume_ratio < settings.min_entry_volume_ratio:
                risk_flags.append("成交量不足")
            if adx_14 < settings.min_entry_adx:
                risk_flags.append("趋势强度不足")
            if volatility > 0.05:
                risk_flags.append("波动偏高")
            if abnormal_wick_count > 0 and abnormal_wick_max >= 80 and abnormal_wick_recent <= 96:
                risk_flags.append(f"近72小时异常插针{abnormal_wick_count}次，最大{abnormal_wick_max:.1f}%")
            confidence = 0.62 if risk_flags else 0.40
            reason = f"风控规则兜底：{('、'.join(risk_flags) if risk_flags else '未发现硬性否决项')}。"
            cross_check_for = {
                "target": "sentiment",
                "question": "请核实是否存在会放大风险的新闻或情绪冲击。",
            } if risk_flags else None

        raw = {
            "local_fallback": True,
            "provider_model": self._model_name,
            "error": error,
            "cross_check_for": cross_check_for,
        }
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=action,
            confidence=min(max(confidence, 0.0), 0.75),
            reasoning=reason,
            position_size_pct=size,
            suggested_leverage=leverage,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            cross_check_for=cross_check_for,
            raw_response=raw,
            feature_snapshot=snapshot,
        )

    async def reinitialize(self) -> None: 
        """Recreate the ChatOpenAI instance with current config."""
        self._llm = None
        await self.initialize()
        logger.info("llm agent reinitialized", name=self.name)

    async def shutdown(self) -> None:
        self._llm = None
