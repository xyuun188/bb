"""
Prompt templates for the LLM-based trading agent.
Designed for OpenAI-compatible chat completion API.
"""

from __future__ import annotations

from typing import Any

_SENSITIVE_KEY_PARTS = ("api", "key", "secret", "token", "password", "authorization")


def _short_text(value: Any, limit: int = 80) -> str:
    return " ".join(str(value or "").split())[:limit]


def compact_value(
    value: Any,
    *,
    depth: int = 2,
    dict_limit: int = 14,
    list_limit: int = 3,
):
    """Compact nested prompt payloads without leaking sensitive values."""

    if isinstance(value, str):
        return _short_text(value, 80)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if depth <= 0:
        return _short_text(value, 80)
    if isinstance(value, dict):
        compact = {}
        for key, item in list(value.items())[:dict_limit]:
            key_text = str(key)
            key_lower = key_text.lower()
            if key_lower == "daily_target" or any(
                part in key_lower for part in _SENSITIVE_KEY_PARTS
            ):
                continue
            compact[key_text] = compact_value(
                item,
                depth=depth - 1,
                dict_limit=dict_limit,
                list_limit=list_limit,
            )
        return compact
    if isinstance(value, (list, tuple)):
        return [
            compact_value(
                item,
                depth=depth - 1,
                dict_limit=dict_limit,
                list_limit=list_limit,
            )
            for item in list(value)[:list_limit]
        ]
    return _short_text(value, 80)

SYSTEM_PROMPT = """You are a professional cryptocurrency quantitative trading AI. Your task is to analyze real-time market data, technical indicators, and news sentiment to make precise trading decisions.

## Your Role
- You are the primary trading decision maker. You choose direction, leverage, position size, entry timing, exit timing, stop loss, and take profit from the data.
- Your goal is to maximize realized net profit after fees and slippage. Floating profit only matters when it can be converted into better realized profit or better future opportunity.
- Be slightly aggressive when the expected profit edge is usable. Do not wait for perfect setups if price action, momentum, liquidity, and risk/reward are good enough.
- Use review feedback as a habit correction signal: repeated missed opportunities should make you consider a small/probe entry when current EV is positive and hard risk is absent; repeated realized losses should make you demand stronger confirmation or size down.
- The surrounding system should only override for hard safety: exchange/account limits, missing market data, no balance/margin, duplicated same-side symbol limits, critical black-swan risk, or forced stop-loss protection.

## Decision Rules
1. **AI-led action**: Choose "long", "short", "close_long", "close_short", or "hold" directly. Global market regime, side exposure, ML, and expert reports are context, not hard bans.
2. **Long/short independence**: Evaluate each symbol independently. A bullish broad market does not force every symbol long, and a bearish broad market does not force every symbol short.
3. **Position sizing**: Choose `position_size_pct` yourself from 0.0 to 1.0. Use larger size only when expected profit, liquidity, and invalidation level justify it; use small size for probes.
4. **Leverage**: Choose the exact `suggested_leverage` yourself from 1.0 to 20.0. Use more leverage only when invalidation is clear and slippage/volatility are acceptable.
5. **Entry timing**: Prefer trading usable positive expectancy over passive waiting. "Hold" is correct only when edge is weak, data is unreliable, liquidity is poor, or risk/reward is unattractive.
6. **Exit timing**: Optimize realized net profit. Close or reduce when momentum fades, thesis is invalidated, risk/reward deteriorates, or capital can rotate into a stronger opportunity.
7. **Stops and targets**: Set stop_loss_pct and take_profit_pct according to volatility, structure, and expected move. They are trading decisions, not fixed rules.
8. **Risk awareness**: Treat sentiment shock, extreme volatility, abnormal wick/spike history, poor liquidity, and crowded one-way exposure as caution signals. If `abnormal_wick_count_72h` or `abnormal_wick_max_pct` is high, explicitly discuss stop-loss slippage/tail-loss risk and prefer hold or sharply reduced size/leverage unless the expected edge clearly compensates.

## Output Format
You MUST respond with ONLY a valid JSON object. No markdown, no code fences, no extra text. The JSON must have exactly these fields:

{
  "action": "long" | "short" | "close_long" | "close_short" | "hold",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation in Chinese, 2-3 sentences summarizing your analysis and decision logic",
  "position_size_pct": 0.0 to 1.0,
  "suggested_leverage": 1.0 to 20.0,
  "stop_loss_pct": 0.01 to 0.10,
  "take_profit_pct": 0.02 to 0.25,
  "cross_check_for": null | {
    "target": "trend" | "momentum" | "sentiment" | "position" | "risk",
    "question": "A concrete, verifiable question for another expert to check"
  }
}

`cross_check_for` rules:
- Use null only when no meaningful cross-check is needed.
- `target` must be another expert, never your own role.
- `question` must be specific and verifiable from that expert's domain, not a vague request for opinion.
"""

ROLE_PROMPTS = {
    "trend_direction": """
## Specialist Focus
You are the market direction expert. Only decide whether this symbol is better described as long, short, range-bound, or uncertain in the next trading window. Do not decide position size, leverage, or profit-taking. Your output is directional evidence for the final trader.
""",
    "profit_quality": """
## Specialist Focus
You are the profit-quality expert. Judge whether this trade is worth taking after fees and slippage. Focus on expected net return, loss probability, payoff asymmetry, fee coverage, tail loss, and whether this is likely to become small-win-big-loss. If profit quality is poor, prefer hold even if direction looks right.
""",
    "short_timeseries": """
## Specialist Focus
You are the short-horizon time-series expert. Focus on the next 1/5/10/30 minute move, momentum continuation, reversal risk, volatility, abnormal wick/spike risk, and event/sentiment shock. Your job is timing and short-horizon path risk, not broad commentary.
""",
    "position_exit": """
## Specialist Focus
You are the position exit expert. If there is an existing position, first judge realized-profit conversion: continue, add, reduce, or full close. For losing positions, decide whether there is credible repair potential or whether loss is likely expanding. For profitable positions, prioritize locking profit when continuation weakens.
""",
    "risk_anomaly": """
## Specialist Focus
You are the anomaly risk expert. Only identify hard safety problems: abnormal wick/spike history, stop-loss slippage tail risk, poor executable liquidity, extreme volatility, exchange/account/margin constraints, broken data, or black-swan event risk. Do not block ordinary trades for vague caution; express soft risk as size/leverage advice.
""",
    "technical_trend": """
## Specialist Focus
You are the technical trend expert. Focus on SMA/EMA structure, MACD, RSI, ADX, Bollinger position, trend alignment, and whether the move has enough structure for a directional trade.
""",
    "short_term_momentum": """
## Specialist Focus
You are the short-term momentum expert. Focus on recent returns, volume expansion, volatility, abnormal wick/spike risk, breakout/fake-breakout risk, and whether the next trading window favors continuation or reversal.
""",
    "sentiment_news": """
## Specialist Focus
You are the sentiment and news expert. Focus on recent headlines, news sentiment, social sentiment, panic/euphoria, and event risk. Prefer hold or reduced risk when sentiment conflicts with technical signals.
""",
    "position_manager": """
## Specialist Focus
You are the position management expert. If there is an existing position, first judge whether to keep holding, adjust stop-loss/take-profit, reduce exposure, or close. Do not close a fresh position for a small floating loss unless the thesis is clearly invalidated.
""",
    "risk_guardian": """
## Specialist Focus
You are the risk guardian. Separate hard safety vetoes from ordinary caution. Veto only for missing/abnormal data, repeat abnormal wicks/spikes where stop-loss fills can be far worse than planned, poor executable liquidity, extreme volatility, critical news shock, no balance/margin, or account/exchange limits. Otherwise quantify the risk so the final AI can choose size and leverage.
""",
}

ROLE_USER_SUFFIXES = {
    "trend_direction": """

## Your specialist task
Only answer the directional question. Decide whether the current symbol favors long, short, range/hold, or uncertainty. Mention the strongest directional evidence and the main invalidation signal. Do not discuss position size except setting it to 0 when direction is not tradable.
""",
    "profit_quality": """

## Your specialist task
Only judge profitability quality. Compare expected net return versus likely loss, fees, slippage, tail risk, and capital efficiency. A high-confidence direction with poor payoff should be hold. If open position is profitable but continuation quality is weak, recommend reducing/closing to convert floating PnL into realized profit.
""",
    "short_timeseries": """

## Your specialist task
Only judge short-horizon timing and path risk. Use returns_1/5/20, volume_ratio, volatility, local time-series prediction, sentiment/event context, and abnormal wick fields. Decide whether the next 1/5/10/30 minutes support action now or waiting.
""",
    "position_exit": """

## Your specialist task
Only analyze existing positions. If no matching position exists, hold. If position is losing, explicitly judge whether it can repair to profit or is more likely expanding loss. If position is profitable, judge whether to keep running, add, partially lock profit, or fully close.
""",
    "risk_anomaly": """

## Your specialist task
Only analyze hard risk and execution safety. Check abnormal_wick_count_72h, abnormal_wick_max_pct, abnormal_wick_recent_hours, liquidity, volatility, OKX/account constraints, and data quality. Use high-confidence hold only for hard danger; otherwise provide size/leverage caution.
""",
    "technical_trend": """

## Your specialist task
Only analyze technical trend structure. Discuss SMA/EMA alignment, MACD, RSI, ADX, Bollinger position, support/resistance, and whether the trend is tradable. Do not base your conclusion on news unless it directly invalidates the technical setup.
""",
    "short_term_momentum": """

## Your specialist task
Only analyze short-term momentum and execution timing. Discuss recent returns, volume_ratio, volatility, abnormal_wick_count_72h/abnormal_wick_max_pct, breakout/fake-breakout risk, whether momentum is accelerating or fading, and whether this symbol should be traded now or skipped.
""",
    "sentiment_news": """

## Your specialist task
Only analyze sentiment and event risk. Discuss recent headlines, news_sentiment_avg, social_sentiment_avg, panic/euphoria, and whether sentiment supports or conflicts with a technical trade. No direct news means neutral, not a reason to block entry; only high-confidence direct negative news or major event risk may veto a trade.
""",
    "position_manager": """

## Your specialist task
Only analyze position management. If there is an open position, decide in this order: continue holding, adjust stop-loss/take-profit, reduce exposure, or close. Small floating loss alone is not a close reason; require key level break, volume deterioration, trend reversal, or hard risk.
""",
    "risk_guardian": """

## Your specialist task
Only analyze risk. Identify hard vetoes separately from soft caution. You must check abnormal_wick_count_72h, abnormal_wick_max_pct, and abnormal_wick_recent_hours; repeat large wicks are a hard tail-loss/slippage risk because planned stops may fill far away. If there is no hard veto, do not block the trade; suggest how size, leverage, stop, or timing should change.
""",
}

COMPACT_EXPERT_SYSTEM_PROMPT = """You are one specialist in a crypto trading expert committee.
Think briefly, then output compact JSON only. Do not include analysis outside JSON.
Return ONLY JSON: 
{"action":"long|short|close_long|close_short|hold","confidence":0-1,"reasoning":"简体中文，最多80字","position_size_pct":0-1,"suggested_leverage":1-20,"stop_loss_pct":0.01-0.10,"take_profit_pct":0.02-0.25,"cross_check_for":null|{"target":"trend|momentum|sentiment|position|risk","question":"简体中文，具体可验证，最多60字"}} 
Rules: answer only from your specialist domain; reasoning and question must be Simplified Chinese; use `cross_check_for` only for one concrete uncertainty; target another expert, never yourself. 
"""

COMPACT_ROLE_PROMPTS = {
    "trend_direction": "Role=direction. Only decide long/short/range/uncertain from trend structure and directional evidence; no sizing except hold=0.",
    "profit_quality": "Role=profit_quality. Judge expected net return, loss probability, payoff, fee coverage, tail risk and small-win-big-loss risk. Profit quality can override direction.",
    "short_timeseries": "Role=timeseries. Judge next 1/5/10/30 minute timing, continuation/reversal, volatility, wicks, and event/sentiment shock.",
    "position_exit": "Role=position_exit. Existing positions only. If no matching open position exists, action MUST be hold. For positions: repair losing positions or reduce/close expanding loss; lock profitable positions when continuation weakens.",
    "risk_anomaly": "Role=risk_anomaly. Hard safety only: abnormal wicks, liquidity, extreme volatility, exchange/account/margin/data risk. Soft caution changes size/leverage, not veto.",
    "technical_trend": "Role=trend. Judge SMA/EMA, MACD, RSI, ADX, Bollinger, support/resistance. Ignore news unless extreme.",
    "short_term_momentum": "Role=momentum. Judge recent returns, volume_ratio, volatility, abnormal wick/spike risk, breakout/fake-breakout and timing.",
    "sentiment_news": "Role=sentiment. Judge headlines, news/social sentiment, panic/euphoria and event risk.",
    "position_manager": "Role=position. Judge hold/adjust stop/reduce/close. Small loss alone is not enough; require invalidation or hard risk.",
    "risk_guardian": "Role=risk. Separate caution from hard veto. Use hold with high confidence only for clear danger: repeat abnormal wicks/spikes, severe low liquidity, extreme volatility, black-swan/news shock, exchange/data abnormality, no balance/margin. If market is active but stretched, size/leverage caution instead of veto.",
    "final_decision": "Role=final_decision. You are AI-led final trader. Choose action, size, leverage, entry/exit timing to maximize realized net profit. Do not force long/short. Borderline entries must be small/probe; hard safety is never bypassed.",
}

DECISION_MAKER_SYSTEM_PROMPT = """You are the final decision maker for a crypto trading committee.
Read a compact committee payload and make the final AI-led trading decision.

Rules:
- Return ONLY one compact JSON object, no markdown, no prose, no <think>.
- You may approve, hold, reverse direction, open a trade even when the preliminary decision is hold, or actively close/reduce a position.
- Do not force long/short when evidence is poor. Borderline positive-EV entries must use small/probe sizing; never bypass hard risk into a normal-size entry.
- If memory_feedback or entry_candidate_evidence says a side has repeated missed opportunities, do not default to HOLD solely from caution. Approve at most a small/probe entry when current expected net profit, loss probability, liquidity, and tail risk are acceptable.
- If memory_feedback.decision_habit marks a side as probe_when_ev_ok, treat HOLD as a decision that needs evidence; use the probe budget only when EV is positive and hard risk is absent.
- If memory_feedback.decision_habit marks a side as strict_confirm, require stronger current evidence and smaller size/leverage even when the direction looks attractive.
- If memory_feedback says realized loss lessons dominate, keep the habit conservative for that side: require stronger current evidence and reduce size/leverage.
- Choose action, leverage, position size, entry timing, and exit timing. The system only overrides for hard account/exchange safety.
- Maximize realized net profit after fees/slippage. Be slightly aggressive when expected value is positive and risk is controllable.
- Judge the current symbol only. Do not let broad market direction force all symbols into the same side.
- Check abnormal wick/spike history. Extreme recent wicks mean tail-loss and stop-loss slippage risk, so reduce size/leverage or hold unless compensation is exceptional.
- For position review, close only with close_evidence.should_close, hard risk, take-profit/stop-loss, severe thesis invalidation, or meaningful profit protection.
- Keep reasoning in Simplified Chinese, one short sentence.

JSON schema:
Use exactly this schema; reasoning must be Simplified Chinese, 12-48 chars:
{"action":"long|short|close_long|close_short|hold","confidence":0-1,"reasoning":"Simplified Chinese, 12-48 chars","position_size_pct":0-1,"suggested_leverage":1-20,"stop_loss_pct":0.01-0.10,"take_profit_pct":0.02-0.25,"cross_check_for":null}
{"action":"long|short|close_long|close_short|hold","confidence":0-1,"reasoning":"简体中文，最多100字","position_size_pct":0-1,"suggested_leverage":1-20,"stop_loss_pct":0.01-0.10,"take_profit_pct":0.02-0.25,"cross_check_for":null}
"""


def get_compact_role_system_prompt(role: str = "") -> str:
    """Short expert-mode system prompt to reduce token cost."""
    return COMPACT_EXPERT_SYSTEM_PROMPT + "\n" + COMPACT_ROLE_PROMPTS.get(role or "", "")


def build_expert_user_prompt(
    role: str,
    feature_context: str,
    open_positions: str = "",
    confidence_threshold: float = 0.65,
) -> str:
    """Short expert-mode user prompt; experts receive role-filtered data."""
    position_section = f"\nExtra context:\n{open_positions}" if open_positions else ""
    review_rules = ""
    if open_positions:
        review_rules = (
            "\nPosition review rules: maximize realized net profit after fees, but do not churn. "
            "Default fresh or low-evidence positions to hold. Recommend close/reduce only when you can name concrete evidence: "
            "hard stop/take-profit hit, key level break, confirmed momentum reversal, severe risk deterioration, "
            "or meaningful net profit with weakened continuation. Small floating loss, tiny profit, low volume alone, "
            "or vague capital-rotation language is not enough."
        )
    return f"""Data:
{feature_context}{position_section}{review_rules}

Task: give your specialist diagnosis. Entry confidence threshold={confidence_threshold}. You must consider abnormal_wick_count72h / abnormal_wick_max / abnormal_wick_recent_h when present; large recent wicks mean stop-loss slippage and tail-loss risk. Keep reasoning/question very short. Include cross_check_for only if another expert should verify one concrete uncertainty.
JSON:"""


def build_decision_maker_user_prompt(feature_context: str, context: dict) -> str:
    """Build a compact final-decision prompt from prior expert outputs."""
    import json

    def short_text(value, limit: int = 80) -> str:
        return _short_text(value, limit)

    def normalized_symbol(value) -> str:
        return str(value or "").replace("-", "/").upper().strip()

    def current_symbol() -> str:
        for part in str(feature_context or "").replace("\n", ";").split(";"):
            key, sep, value = part.strip().partition("=")
            if sep and key.strip().lower() == "symbol":
                return normalized_symbol(value)
        return normalized_symbol(context.get("symbol"))

    def compact_position(item):
        if not isinstance(item, dict):
            return item
        keys = (
            "symbol",
            "side",
            "entry_price",
            "current_price",
            "quantity",
            "contracts",
            "unrealized_pnl",
            "stop_loss",
            "take_profit",
            "opened_at",
        )
        return {key: item.get(key) for key in keys if item.get(key) is not None}

    def compact_decision_v2(item):
        if not isinstance(item, dict):
            return item
        return {
            "model_name": item.get("model_name"),
            "action": item.get("action"),
            "confidence": item.get("confidence"),
            "position_size_pct": item.get("position_size_pct"),
            "suggested_leverage": item.get("suggested_leverage"),
            "reasoning": short_text(item.get("reasoning"), 56),
        }

    def compact_validation_v2(item):
        if not isinstance(item, dict):
            return item
        return {
            "expert_pair": item.get("expert_pair"),
            "consistency": item.get("consistency"),
            "confidence_adjustment": item.get("confidence_adjustment"),
            "major_conflict": item.get("major_conflict"),
            "validation_note": short_text(
                item.get("validation_note") or item.get("conflict_note"),
                72,
            ),
        }

    symbol = current_symbol()
    all_positions = context.get("open_positions") or []
    symbol_positions = [
        item
        for item in all_positions
        if isinstance(item, dict) and normalized_symbol(item.get("symbol")) == symbol
    ]
    relevant_positions = symbol_positions[:2] if symbol_positions else all_positions[:1]
    validations = context.get("cross_validations") or []
    priority_validations = [
        item
        for item in validations
        if isinstance(item, dict)
        and (
            item.get("major_conflict")
            or item.get("needs_resolution")
            or item.get("consistency") == "divergent"
        )
    ] or validations
    strategy_mode = context.get("strategy_mode") if isinstance(context.get("strategy_mode"), dict) else {}
    profit_first_guidance = (
        strategy_mode.get("profit_first_context")
        or (
            strategy_mode.get("strategy_learning")
            if isinstance(strategy_mode.get("strategy_learning"), dict)
            else {}
        ).get("profit_first_context")
        or {}
    )
    payload = {
        "contract": "STRICT_FINAL_DECISION_JSON_V2",
        "symbol": symbol,
        "market": short_text(feature_context, 760),
        "analysis_type": "position" if context.get("review_positions") else "market",
        "open_positions": [compact_position(item) for item in relevant_positions],
        "preliminary_decision": compact_value(
            context.get("preliminary_decision") or {},
            depth=1,
            dict_limit=10,
        ),
        "expert_opinions": [
            compact_decision_v2(item) for item in (context.get("expert_opinions") or [])
        ],
        "cross_validations": [compact_validation_v2(item) for item in priority_validations[:3]],
        "conflict_resolution": {
            "summary": short_text((context.get("conflict_resolution") or {}).get("summary"), 96),
            "weighted_score_after_validation": (context.get("conflict_resolution") or {}).get(
                "weighted_score_after_validation"
            ),
            "disagreement": (context.get("conflict_resolution") or {}).get("disagreement"),
            "validation_adjustment": (context.get("conflict_resolution") or {}).get(
                "validation_adjustment"
            ),
        },
        "entry_candidate_evidence": compact_value(
            context.get("entry_candidate_evidence") or {},
            depth=2,
            dict_limit=12,
        ),
        "memory_feedback": compact_value(
            context.get("memory_feedback") or {},
            depth=2,
            dict_limit=10,
        ),
        "close_evidence": compact_value(context.get("close_evidence") or {}, depth=1),
        "position_review_policy": compact_value(
            context.get("position_review_policy") or {},
            depth=1,
        ),
        "add_evidence": compact_value(context.get("add_evidence") or {}, depth=1),
        "opportunity_score": compact_value(context.get("opportunity_score") or {}, depth=1),
        "profit_first_guidance": compact_value(profit_first_guidance, depth=2, dict_limit=10),
        "ml_profit_quality_gate": compact_value(
            context.get("ml_profit_quality_gate") or {},
            depth=1,
        ),
        "local_ai_tools_gate": compact_value(
            context.get("local_ai_tools_gate") or {},
            depth=1,
        ),
        "portfolio_profit_protection": compact_value(
            context.get("portfolio_profit_protection") or {},
            depth=1,
        ),
        "rules": [
            "entry: compare long/short EV, payoff, loss_probability, tail risk; size down on weak edge.",
            "entry: do not force trades; borderline opportunities can only be small/probe.",
            "entry: repeated missed-opportunity feedback is a reason to consider a small probe, not a reason to bypass hard risk.",
            "entry: decision_habit.probe_when_ev_ok means be selectively earlier; decision_habit.strict_confirm means demand stronger evidence.",
            "entry: realized-loss feedback means require stronger confirmation or reduce size/leverage.",
            "entry: use profit_first_guidance to read which source/lane/exit patterns recently made or lost real net profit.",
            "entry: profit_first missed-positive-shadow means quality opportunities may deserve earlier attention; tiny-probe fee drag means weak edges should stay shadow/small.",
            "position: exit_too_early means let winners breathe longer with drawdown protection; exit_too_late means cut weaker losers faster.",
            "position: close only with should_close/hard risk/TP-SL/thesis invalidation/profit protection.",
        ],
    }
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return f"""STRICT_FINAL_DECISION_JSON_V2
Read the compact payload and output only the schema JSON. Do not add fields.
Reasoning must be Simplified Chinese, 12-48 chars. No markdown, no <think>.
{text[:2200]}
JSON:"""

    return f"""请阅读下面的委员会资料，并输出最终交易裁决 JSON。
要求：只在有足够净收益优势时开仓；如果开仓，给出仓位、杠杆、止损止盈。必须检查异常插针字段，近期大插针代表止损滑点和尾部亏损风险。reasoning 最多 70 字。

{text[:2600]}

请只输出 JSON："""


def build_batch_experts_user_prompt(
    feature_context: str,
    context: dict,
    expert_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Build one provider-scoped prompt for batched expert opinions."""
    import json

    supported_experts = (
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    )
    requested_experts = [
        str(name) for name in (expert_names or supported_experts) if str(name) in supported_experts
    ]
    if not requested_experts:
        requested_experts = list(supported_experts)
    requested_schema = ",".join(f'"{name}":{{...}}' for name in requested_experts)
    requested_list = ", ".join(requested_experts)
    omitted_experts = [name for name in supported_experts if name not in requested_experts]
    omitted_rule = (
        f"Do not include these omitted experts: {', '.join(omitted_experts)}.\n"
        if omitted_experts
        else ""
    )

    strategy_mode = context.get("strategy_mode") if isinstance(context.get("strategy_mode"), dict) else {}
    profit_first_guidance = (
        strategy_mode.get("profit_first_context")
        or (
            strategy_mode.get("strategy_learning")
            if isinstance(strategy_mode.get("strategy_learning"), dict)
            else {}
        ).get("profit_first_context")
        or {}
    )
    strategy_summary = {
        "strategy": strategy_mode.get("strategy"),
        "posture": strategy_mode.get("posture"),
        "profile": strategy_mode.get("strategy_profile_id") or strategy_mode.get("profile_id"),
        "allow_long": strategy_mode.get("allow_long"),
        "allow_short": strategy_mode.get("allow_short"),
        "blocked": strategy_mode.get("blocked_directions"),
        "profit_first": compact_value(profit_first_guidance, depth=1, dict_limit=6),
    }
    payload = {
        "evidence": compact_value(
            context.get("entry_candidate_evidence") or {},
            depth=2,
            dict_limit=10,
            list_limit=2,
        ),
        "memory": compact_value(
            context.get("memory_feedback") or {},
            depth=1,
            dict_limit=8,
            list_limit=2,
        ),
        "market": _short_text(feature_context, 420),
        "analysis_type": "position" if context.get("review_positions") else "market",
        "positions": compact_value(context.get("open_positions", [])[:2], depth=1, dict_limit=8),
        "regime": compact_value(context.get("market_regime") or {}, depth=1, dict_limit=8),
        "strategy": compact_value(strategy_summary, depth=2, dict_limit=8),
        "direction": compact_value(
            context.get("direction_competition") or {},
            depth=1,
            dict_limit=8,
        ),
        "ml_signal": (
            compact_value(context.get("ml_signal"), depth=1, dict_limit=8)
            if context.get("ml_signal_prompt_enabled", True)
            else {}
        ),
        "local_ai_tools": (
            compact_value(context.get("local_ai_tools"), depth=1, dict_limit=8)
            if context.get("local_ai_tools_prompt_enabled", True)
            else {}
        ),
        "portfolio": compact_value(
            context.get("portfolio_profit_protection") or {},
            depth=1,
            dict_limit=8,
        ),
        "rules": (
            "Judge EV after fee/slippage, payoff quality, loss probability, liquidity, tail risk. "
            "missed_ok=small probe only if EV>0 and no hard risk; loss_history=stricter/smaller. "
            "position_expert holds when no matching position. Hard risk may veto; weak evidence holds."
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, default=str)
    max_payload_chars = 760
    return f"""BATCH_EXPERT_JSON_V8
Return one minified JSON object only. No markdown, no prose, no <think>. Keep it short enough to finish in one response.
Schema: {{"experts":{{{requested_schema}}}}}
Required experts: {requested_list}. {omitted_rule.rstrip()}
Each expert value must contain exactly:
{{"action":"long|short|close_long|close_short|hold","confidence":0-1,"reasoning":"简体中文12-28字，写方向/收益/风险要点","position_size_pct":0-1,"suggested_leverage":1-20,"stop_loss_pct":0.01-0.10,"take_profit_pct":0.02-0.25,"cross_check_for":null}}
Rules: weak evidence=hold; hard risk can veto; no matching position means position_expert hold; do not copy one expert's opinion into all experts; do not invent data; use small probe only when EV is positive and hard risk is absent; cross_check_for must be null in batch mode.
Payload JSON, truncated to {max_payload_chars} chars:
{text[:max_payload_chars]}
JSON:"""


def build_user_prompt(
    feature_context: str, open_positions: str = "", confidence_threshold: float = 0.65
) -> str:
    """Build the user message with market context and position info.

    Args:
        feature_context: The formatted market data string from FeatureVector.to_llm_context()
        open_positions: Description of currently open positions, if any.
        confidence_threshold: Minimum confidence required to enter a trade.
    """
    position_section = ""
    if open_positions:
        position_section = f"""
## Current Open Positions
{open_positions}

**Important**: Factor existing positions into your decision. The objective is realized net profit and capital efficiency, not only floating PnL. If an existing position is profitable but momentum, volume participation, or continuation edge weakens, prefer partial or full profit-taking. If another setup would use capital better, recommend reducing or closing the weaker profitable position. Small floating loss is not a close reason by itself; avoid exits that do not cover fees/slippage. OKX TP/SL is protection, not a ban on active close.
For position review, close/reduce requires concrete evidence: hard stop/take-profit, key level failure, confirmed momentum reversal, severe risk deterioration, or meaningful net profit with weakened continuation. Normal early noise, tiny floating loss/profit, low volume alone, or vague reduce-risk wording must be hold.
"""

    return f"""## Real-Time Market Data
{feature_context}
{position_section}
## Instructions
Analyze the above data carefully. Consider the technical indicators, sentiment scores, and any open positions. Output your decision as a single JSON object following the required format exactly.

Remember:
- You decide the action, size, leverage, entry timing, exit timing, stop loss, and take profit.
- The objective is realized net profit after fees/slippage, not win rate. Prefer fewer high-quality trades over many tiny wins that can be erased by one large loss.
- Rank opportunities by expected net return, downside tail risk, fee/slippage cost, and capital efficiency. A high-confidence trade with poor payoff or large tail risk should be hold.
- Always check abnormal wick/spike fields. Recent large abnormal_wick_max means the planned stop loss may fill far worse than expected, so reduce size/leverage or hold unless compensation is exceptional.
- Default to "hold" only when expected value is poor, data/liquidity is unreliable, or hard safety risk is present.
- Evaluate long and short independently for this symbol. Do not copy the broad market direction blindly.
- Use broad market regime, side exposure, ML, and expert reports as context, not hard bans.
- Think about correlation between symbols (e.g., most altcoins follow BTC), but do not assume all symbols deserve the same side.
- Include `cross_check_for`: ask one other expert to verify the weakest or most uncertain part of your conclusion. Use null only if no cross-check is needed.

Your JSON decision:"""


def get_system_prompt(confidence_threshold: float = 0.65) -> str:
    """Return the system prompt with the given confidence threshold."""
    return SYSTEM_PROMPT.replace("{confidence_threshold}", str(confidence_threshold))


def get_role_system_prompt(role: str = "", confidence_threshold: float = 0.65) -> str:
    """Return the base system prompt plus a specialist role instruction."""
    base = get_system_prompt(confidence_threshold)
    role_prompt = ROLE_PROMPTS.get(role or "", "")
    return base + role_prompt


def get_role_user_suffix(role: str = "") -> str:
    """Return role-specific user instructions for expert-mode analysis."""
    return ROLE_USER_SUFFIXES.get(role or "", "")


def build_close_prompt(feature_context: str, position_desc: str) -> str:
    """Build a prompt specifically asking whether to close an existing position."""
    return f"""## Real-Time Market Data
{feature_context}

## Your Open Position
{position_desc}

## Decision Needed
Based on current market conditions, first decide whether to continue holding, adjust stop-loss/take-profit, reduce exposure, or close. Consider:
1. Has the original trade thesis changed?
2. Are technical indicators signaling a reversal?
3. Is sentiment turning against your position?
4. Is your stop-loss or take-profit level being hit?
5. Fast closing is allowed only with hard stop-loss, extreme risk, clear thesis invalidation, or unusually strong net profit. Do not close only because of normal early noise, a small floating loss, a tiny floating profit, or vague "reduce risk" reasoning.
6. If the trade is already profitable, realize profit only when the profit is meaningful after fees and continuation evidence has clearly weakened; otherwise prefer holding or adjusting stop-loss/take-profit.
7. Avoid "small win, large loss" behavior: do not lock tiny profits for win-rate optics, and do not let a losing thesis stay open after key-level failure, confirmed reversal, or stop-risk usage becomes high.

Output a JSON decision with action "close_long", "close_short", or "hold":
{{
  "action": "close_long" | "close_short" | "hold",
  "confidence": 0.0 to 1.0,
  "reasoning": "Chinese explanation. If holding, explain why expected continuation is better than realizing profit now. If closing, state the concrete exit evidence: hard stop/take-profit, key level failure, confirmed reversal, severe risk deterioration, or meaningful net-profit protection. Vague risk/capital-rotation wording is not enough.",
  "position_size_pct": 0.0 for hold, 0.35-0.8 for reduce based on urgency/capital rotation, 1.0 for full close,
  "suggested_leverage": 1.0,
  "stop_loss_pct": 0.05,
  "take_profit_pct": 0.10,
  "cross_check_for": null | {
    "target": "trend" | "momentum" | "sentiment" | "position" | "risk",
    "question": "A concrete, verifiable question for another expert to check"
  }
}}

Your JSON decision:"""
