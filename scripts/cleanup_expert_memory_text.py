from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("data/trading.db")
MOJIBAKE_MARKERS = (
    "閿",
    "缁",
    "閹",
    "閸",
    "楠",
    "娴",
    "鈧",
    "鎾",
    "閫",
    "嫨",
    "瑙",
    "傛",
    "澧",
    "鍫",
    "鐟",
    "闁",
    "锟",
    "褰",
    "鏃",
    "涓",
    "鏆",
    "姣",
    "锛",
    "絾",
    "鍒",
    "挓",
    "鍚",
    "庢",
    "敹",
    "鐩",
    "婁",
)
DAMAGED_MARKERS = (
    "原始说明已损坏",
    "raw note is damaged",
    "无法准确还原",
)
ENGLISH_MARKERS = (
    "missed opportunity",
    "signal validated by shadow replay",
    "signal looked weak in shadow replay",
    "under pattern",
    "ended as",
    "Next time",
    "Check abnormal",
    "Shadow replay:",
    "If expected net return",
    "When directional structure",
)
CHINESE_ADVICE_STRINGS = (
    "当方向结构、ADX、均线和 MACD 同向时，可以提高方向支持，但不能直接决定仓位。",
    "下次出现相似方向结构时，可以适当提高方向信心。",
    "下次必须先看到趋势延续，再提高方向信心。",
    "如果预期净收益、手续费覆盖和亏损概率都合格，可以支持小仓位盈利质量试单。",
    "优先看预期净收益、手续费覆盖、亏损概率和盈亏比，不只看胜率。",
    "当扣费后预期净收益和盈亏质量仍为正时，可以支持执行。",
    "追单前要检查预期净收益、手续费覆盖和盈亏比是否过弱。",
    "如果 1/5/10/30 分钟路径和事件冲击风险有利，可以支持更早执行。",
    "核对 1/5/10/30 分钟路径、延续风险、反转风险和事件冲击后，再判断执行时机。",
    "短周期路径延续相似时，可以支持当前执行时机。",
    "执行前要确认短周期路径是否已经反转。",
    "检查是否该锁盈、亏损能否修复、亏损是否扩大，以及是否值得加仓或减仓。",
    "没有硬风险时，优先用仓位和杠杆控制风险，不要直接否决交易。",
    "没有硬风险时，可以允许小仓位执行。",
    "相似条件下需要降低仓位/杠杆，必要时阻止新开仓。",
    "检查异常插针、流动性、极端波动、保证金限制和交易所约束；没有硬风险时优先用仓位和杠杆控风险。",
    "作为历史教训参考，不直接强制开仓或平仓。",
)


def side_zh(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"long", "做多"}:
        return "做多"
    if text in {"short", "做空"}:
        return "做空"
    return str(value or "").strip() or "当前方向"


def outcome_zh(value: str | None) -> str:
    return {
        "loss": "亏损",
        "profit": "盈利",
        "flat": "打平",
        "good": "有效",
        "bad": "偏弱",
    }.get(str(value or "").strip().lower(), str(value or "").strip() or "未知")


def text_score(value: str) -> int:
    return sum(value.count(marker) for marker in MOJIBAKE_MARKERS)


def repair_mojibake(value: str | None) -> str:
    text = str(value or "")
    if not text:
        return ""
    best = text
    best_score = text_score(text)
    if best_score <= 0:
        return text
    for codec in ("gbk", "cp936"):
        try:
            candidate = text.encode(codec).decode("utf-8")
        except UnicodeError:
            continue
        candidate_score = text_score(candidate)
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


def repair_mojibake_fragment(value: str | None) -> str:
    text = str(value or "")
    if not text or text_score(text) <= 0:
        return text
    best = text
    best_score = text_score(text)
    for encode_errors in ("strict", "ignore", "replace"):
        for decode_errors in ("strict", "ignore", "replace"):
            try:
                candidate = text.encode("gbk", errors=encode_errors).decode("utf-8", errors=decode_errors)
            except UnicodeError:
                continue
            candidate_score = text_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
    return best


def damaged(value: str | None) -> bool:
    text = str(value or "")
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in DAMAGED_MARKERS)


def has_mojibake(value: str | None) -> bool:
    return text_score(str(value or "")) >= 2


def has_english_template(value: str | None) -> bool:
    text = str(value or "")
    return any(marker in text for marker in ENGLISH_MARKERS)


def expert_advice(expert_name: str, text: str = "") -> str:
    name = str(expert_name or "")
    shadow_missed = "机会曾被观望错过" in text or "missed opportunity" in text
    shadow_good = "信号被影子复盘验证有效" in text or "signal validated by shadow replay" in text
    shadow_bad = "信号在影子复盘中表现偏弱" in text or "signal looked weak in shadow replay" in text
    if name == "trend_expert" or "directional structure" in text:
        if shadow_good:
            return "下次出现相似方向结构时，可以适当提高方向信心。"
        if shadow_bad:
            return "下次必须先看到趋势延续，再提高方向信心。"
        return "当方向结构、ADX、均线和 MACD 同向时，可以提高方向支持，但不能直接决定仓位。"
    if name == "momentum_expert" or "expected net return" in text:
        if shadow_missed:
            return "如果预期净收益、手续费覆盖和亏损概率都合格，可以支持小仓位盈利质量试单。"
        if shadow_good:
            return "当扣费后预期净收益和盈亏质量仍为正时，可以支持执行。"
        if shadow_bad:
            return "追单前要检查预期净收益、手续费覆盖和盈亏比是否过弱。"
        return "优先看预期净收益、手续费覆盖、亏损概率和盈亏比，不只看胜率。"
    if name == "sentiment_expert" or "1/5/10/30" in text:
        if shadow_missed:
            return "如果 1/5/10/30 分钟路径和事件冲击风险有利，可以支持更早执行。"
        if shadow_good:
            return "短周期路径延续相似时，可以支持当前执行时机。"
        if shadow_bad:
            return "执行前要确认短周期路径是否已经反转。"
        return "核对 1/5/10/30 分钟路径、延续风险、反转风险和事件冲击后，再判断执行时机。"
    if name == "position_expert":
        return "检查是否该锁盈、亏损能否修复、亏损是否扩大，以及是否值得加仓或减仓。"
    if name == "risk_expert" or "abnormal" in text or "hard risk" in text:
        if shadow_missed:
            return "没有硬风险时，优先用仓位和杠杆控制风险，不要直接否决交易。"
        if shadow_good:
            return "没有硬风险时，可以允许小仓位执行。"
        if shadow_bad:
            return "相似条件下需要降低仓位/杠杆，必要时阻止新开仓。"
        return "检查异常插针、流动性、极端波动、保证金限制和交易所约束；没有硬风险时优先用仓位和杠杆控风险。"
    return "作为历史教训参考，不直接强制开仓或平仓。"


def strip_known_suffix(text: str) -> str:
    suffix_markers = (
        " When directional structure",
        " When trend structure",
        " If expected net return",
        " If 1/5/10/30",
        " If there is no hard risk",
        " Raise directional confidence",
        " Support execution",
        " Support execution timing",
        " Allow small size",
        " Require trend continuation",
        " Check whether expected net return",
        " Check whether the short-horizon path",
        " Reduce size/leverage",
        " Next time",
        " Check abnormal",
    )
    result = text
    for marker in suffix_markers:
        if marker in result:
            result = result.split(marker, 1)[0]
    return result.strip()


def strip_existing_chinese_advice(text: str) -> str:
    result = text
    for advice in CHINESE_ADVICE_STRINGS:
        result = result.replace(advice, "")
    return re.sub(r"\s+", " ", result).strip()


def translate_shadow_replay_body(text: str) -> str:
    result = text.strip()
    m = re.search(
        r"Shadow replay:\s*(?P<side>做多|做空|long|short)\s+signal returned\s+"
        r"(?P<pct>[-+]?\d+(?:\.\d+)?)%\s+after\s+(?P<horizon>\d+)\s+minutes",
        result,
    )
    if m:
        return f"影子复盘显示：{side_zh(m.group('side'))}信号在 {m.group('horizon')} 分钟后收益约 {m.group('pct')}%。"
    m = re.search(
        r"Shadow replay:\s*(?P<side>做多|做空|long|short)\s+signal lost\s+"
        r"(?P<pct>[-+]?\d+(?:\.\d+)?)%\s+after\s+(?P<horizon>\d+)\s+minutes,\s+while\s+"
        r"(?P<opp>做多|做空|long|short)\s+returned\s+(?P<opp_pct>[-+]?\d+(?:\.\d+)?)%",
        result,
    )
    if m:
        return (
            f"影子复盘显示：{side_zh(m.group('side'))}信号在 {m.group('horizon')} 分钟后亏损约 {m.group('pct')}%，"
            f"而{side_zh(m.group('opp'))}方向收益约 {m.group('opp_pct')}%。"
        )
    return strip_known_suffix(result)


def translate_market_pattern(value: str | None) -> str:
    text = repair_mojibake(value).strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split(",")]
    if len(parts) >= 4:
        first = parts[0]
        match = re.match(r"^(?P<symbol>\S+)\s+(?P<side>long|short)$", first)
        if match:
            speed_map = {
                "ultra_short": "极短持仓",
                "short_term": "短线持仓",
                "longer_hold": "较长持仓",
            }
            level_map = {
                "large_loss": "大亏",
                "small_loss": "小亏",
                "profit": "盈利",
                "flat": "打平",
            }
            speed = speed_map.get(parts[1], parts[1])
            level = level_map.get(parts[3], parts[3])
            return f"{match.group('symbol')} {side_zh(match.group('side'))}，{speed}，{parts[2]}，{level}"
    replacements = {
        " long,": " 做多，",
        " short,": " 做空，",
        ", ultra_short,": "，极短持仓，",
        ", short_term,": "，短线持仓，",
        ", longer_hold,": "，较长持仓，",
        ", large_loss": "，大亏",
        ", small_loss": "，小亏",
        ", profit": "，盈利",
        ", flat": "，打平",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def translate_lesson(value: str | None, expert_name: str) -> tuple[str, bool]:
    original = str(value or "").strip()
    text = repair_mojibake(original).strip()
    if not text:
        return "", True
    if damaged(text):
        return text, False

    m = re.match(
        r"^(?P<symbol>\S+)\s+(?P<side>做多|做空|long|short)\s+missed opportunity\.\s*(?P<body>.*)$",
        text,
    )
    if m:
        body = strip_existing_chinese_advice(strip_known_suffix(repair_mojibake_fragment(m.group("body"))))
        return (
            f"{m.group('symbol')} {side_zh(m.group('side'))}机会曾被观望错过。"
            f"{body} {expert_advice(expert_name, text)}"
        ), True

    m = re.match(
        r"^(?P<symbol>\S+)\s+(?P<side>做多|做空)\s*机会曾被观望错过。\s*(?P<body>.*)$",
        text,
    )
    if m:
        body = strip_existing_chinese_advice(strip_known_suffix(repair_mojibake_fragment(m.group("body"))))
        return (
            f"{m.group('symbol')} {side_zh(m.group('side'))}机会曾被观望错过。"
            f"{body} {expert_advice(expert_name, text)}"
        ), True

    m = re.match(
        r"^(?P<symbol>\S+)\s+(?P<side>做多|做空|long|short)\s+signal validated by shadow replay\.\s*(?P<body>.*)$",
        text,
    )
    if m:
        body = translate_shadow_replay_body(strip_existing_chinese_advice(strip_known_suffix(repair_mojibake_fragment(m.group("body")))))
        return (
            f"{m.group('symbol')} {side_zh(m.group('side'))}信号被影子复盘验证有效。"
            f"{body} {expert_advice(expert_name, text)}"
        ), True

    m = re.match(
        r"^(?P<symbol>\S+)\s+(?P<side>做多|做空)\s*信号被影子复盘验证有效。\s*(?P<body>.*)$",
        text,
    )
    if m:
        body = translate_shadow_replay_body(strip_existing_chinese_advice(strip_known_suffix(repair_mojibake_fragment(m.group("body")))))
        return (
            f"{m.group('symbol')} {side_zh(m.group('side'))}信号被影子复盘验证有效。"
            f"{body} {expert_advice(expert_name, text)}"
        ), True

    m = re.match(
        r"^(?P<symbol>\S+)\s+(?P<side>做多|做空|long|short)\s+signal looked weak in shadow replay\.\s*(?P<body>.*)$",
        text,
    )
    if m:
        body = translate_shadow_replay_body(strip_existing_chinese_advice(strip_known_suffix(repair_mojibake_fragment(m.group("body")))))
        return (
            f"{m.group('symbol')} {side_zh(m.group('side'))}信号在影子复盘中表现偏弱。"
            f"{body} {expert_advice(expert_name, text)}"
        ), True

    m = re.match(
        r"^(?P<symbol>\S+)\s+(?P<side>做多|做空)\s*信号在影子复盘中表现偏弱。\s*(?P<body>.*)$",
        text,
    )
    if m:
        body = translate_shadow_replay_body(strip_existing_chinese_advice(strip_known_suffix(repair_mojibake_fragment(m.group("body")))))
        return (
            f"{m.group('symbol')} {side_zh(m.group('side'))}信号在影子复盘中表现偏弱。"
            f"{body} {expert_advice(expert_name, text)}"
        ), True

    m = re.match(
        r"^(?P<symbol>\S+)\s+(?P<side>long|short|做多|做空)\s+under pattern\s+\[(?P<pattern>.*?)\]\s+ended as\s+(?P<outcome>loss|profit|flat)\.\s*(?P<body>.*)$",
        text,
    )
    if m:
        pattern = translate_market_pattern(m.group("pattern"))
        return (
            f"{m.group('symbol')} {side_zh(m.group('side'))}在场景[{pattern}]下结果为{outcome_zh(m.group('outcome'))}。"
            f"{expert_advice(expert_name, text)}"
        ), True

    m = re.match(
        r"^(?P<symbol>\S+)\s+(?P<side>long|short|做多|做空)\s+held\s+(?P<minutes>[-+]?\d+(?:\.\d+)?)\s+minutes\s+and\s+ended as\s+(?P<outcome>loss|profit|flat)\.\s*(?P<body>.*)$",
        text,
    )
    if m:
        return (
            f"{m.group('symbol')} {side_zh(m.group('side'))}持仓 {m.group('minutes')} 分钟后结果为{outcome_zh(m.group('outcome'))}。"
            f"{expert_advice(expert_name, text)}"
        ), True

    if text.startswith("Next time") or text.startswith("Check abnormal"):
        return expert_advice(expert_name, text), True

    if has_mojibake(text):
        return text, False
    if has_english_template(text):
        return text, False
    return text, True


def clean_reflection_text(value: str | None) -> str:
    text = repair_mojibake(value).strip()
    if not text:
        return ""
    if damaged(text) or has_mojibake(text):
        return "该笔复盘原文已损坏，已不作为专家记忆依据；请以成交记录和 OKX 订单状态为准。"
    return text


def backup_db(db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.stem}.backup_before_expert_memory_cleanup_{stamp}{db_path.suffix}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def cleanup(db_path: Path, apply: bool) -> dict[str, int | str]:
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    backup_path = backup_db(db_path) if apply else None
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    stats = {
        "memories_seen": 0,
        "memories_updated": 0,
        "memories_deactivated": 0,
        "reflections_seen": 0,
        "reflections_updated": 0,
    }
    try:
        rows = con.execute(
            "select id, expert_name, lesson, market_pattern, is_active from expert_memories"
        ).fetchall()
        for row in rows:
            stats["memories_seen"] += 1
            lesson, usable = translate_lesson(row["lesson"], row["expert_name"])
            pattern = translate_market_pattern(row["market_pattern"])
            should_deactivate = not usable
            changed = (
                lesson != (row["lesson"] or "")
                or pattern != (row["market_pattern"] or "")
                or (should_deactivate and int(row["is_active"] or 0) != 0)
            )
            if not changed:
                continue
            stats["memories_updated"] += 1
            if should_deactivate:
                stats["memories_deactivated"] += 1
            if apply:
                con.execute(
                    """
                    update expert_memories
                    set lesson = ?, market_pattern = ?, is_active = case when ? then 0 else is_active end
                    where id = ?
                    """,
                    (lesson, pattern, 1 if should_deactivate else 0, row["id"]),
                )

        rows = con.execute(
            "select id, mistake_summary, improvement_summary from trade_reflections"
        ).fetchall()
        for row in rows:
            stats["reflections_seen"] += 1
            mistake = clean_reflection_text(row["mistake_summary"])
            improvement = clean_reflection_text(row["improvement_summary"])
            if mistake == (row["mistake_summary"] or "") and improvement == (row["improvement_summary"] or ""):
                continue
            stats["reflections_updated"] += 1
            if apply:
                con.execute(
                    "update trade_reflections set mistake_summary = ?, improvement_summary = ? where id = ?",
                    (mistake, improvement, row["id"]),
                )
        if apply:
            con.commit()
            stats["backup_path"] = str(backup_path)
    finally:
        con.close()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean expert memory mojibake and English templates.")
    parser.add_argument("--db", default=str(DB_PATH), help="SQLite database path.")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Without this flag the script only reports counts.")
    args = parser.parse_args()
    stats = cleanup(Path(args.db), apply=bool(args.apply))
    for key, value in stats.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
