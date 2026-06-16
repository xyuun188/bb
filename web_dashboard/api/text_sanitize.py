"""Display text cleanup helpers for dashboard APIs."""

from __future__ import annotations

import re
from typing import Any

MOJIBAKE_MARKERS = (
    "\u9351",
    "\u95c4",
    "\u9422",
    "锛",
    "绋",
    "鎵",
    "鍗",
    "骞",
    "浠",
    "褰",
    "撳",
    "墠",
    "鏃",
    "堕",
    "涓",
    "鍙",
    "瑙",
    "绯",
    "璁",
    "搴",
    "閫",
    "鎸",
    "鍦",
    "妯",
    "浜",
    "鐩",
    "椋",
    "楂",
    "鍚",
    "鍒",
    "寮",
    "淇",
    "惧",
    "獟",
    "€",
    "\ue1da",
    "閺",
    "閸",
    "瑜",
    "娴",
    "顫",
    "澧",
    "鑰",
)


EXACT_REPLACEMENTS = {
    "\u9351\u5fce\u7ca8\u93ba\u3224\u62e1": "减仓探针",
    "\u95c4\u5d84\u7d86\u6d60\u64b2\u7d85\u9a9e\u8dfa\u60ce\u9422\u3126\u5e30\u95bd\u5818\u20ac?": "降低仓位并启用探针",
    "鍒嗘澶с€佹尝鍔ㄥ紓甯告垨娌℃湁鐩堝埄妯″瀷鍚屽悜纭锛屼繚鎸?1.15+ 楂橀棬妲涖€?": "分歧大、波动异常或没有盈利模型同向确认，保持 1.15+ 高门槛。",
    "ML 涓庢湇鍔″櫒鐩堝埄妯″瀷鍚屽悜涓旈鏈熸敹鐩婁负姝ｏ紝鍏佽 0.75+ 灏忎粨寮€浠撱€?": "ML 与服务器盈利模型同向且预期收益为正，允许 0.75+ 小仓开仓。",
    "ML 鎴栨湇鍔″櫒鐩堝埄妯″瀷涓?AI 鏂瑰悜鍚屽悜涓旈鏈熸敹鐩婁负姝ｏ紝鍏佽 0.85+ 灏忎粨寮€浠撱€?": "ML 或服务器盈利模型与 AI 方向同向且预期收益为正，允许 0.85+ 小仓开仓。",
    "涓撳鏂瑰悜涓€鑷翠笖棰勬湡鏀剁泭涓烘锛屽厑璁?0.90+ 寮€浠撱€?": "专家方向一致且预期收益为正，允许 0.90+ 开仓。",
    "浠婂ぉ杩樻病鏈夎甯佺鏂瑰悜鐨勭湡瀹炲钩浠撹褰曘€?": "今天还没有该币种方向的真实平仓记录。",
    "ML 褰撳墠杈炬爣锛屽弬涓庢満浼氳瘎鍒嗐€?": "ML 当前达标，参与机会评分。",
    "ML 褰撳墠澶勪簬瀛︿範瑙傚療涓垨璇ユ柟鍚戞湭杈炬爣锛屾湰娆℃満浼氳瘎鍒嗕笉浣跨敤 ML 鍔犲噺鍒嗐€?": "ML 当前处于学习观察中，或该方向未达标，本次机会评分不使用 ML 加减分。",
    "鏈轰細璇勫垎鎺掑悕闈犲墠锛屾湰杞繘鍏ユ墽琛岄槦鍒椼€?": "机会评分排名靠前，本轮进入执行队列。",
    "鏈轰細璇勫垎鎺掑悕闈犲墠锛屾湰杞凡杩涘叆鎵ц闃熷垪锛屾鍦ㄨ繘琛屼笅鍗曞墠妫€鏌ュ拰鎻愪氦璁㈠崟銆?": "机会评分排名靠前，本轮已进入执行队列，正在进行下单前检查和提交订单。",
    "AI 閫夋嫨瑙傛湜锛屾湭鎻愪氦璁㈠崟銆?": "AI 选择观望，未提交订单。",
    "椋庢帶寮曟搸鎷掔粷璇ュ喅绛栥€?": "风控引擎拒绝该决策。",
    "椋庢帶寮曟搸鎷掔粷璇ュ妯″瀷瑁佸喅銆?": "风控引擎拒绝该多模型裁决。",
    "璇ュ竵绉嶈繖涓柟鍚戜粖澶╃殑鐪熷疄鐩堜簭琛ㄧ幇鍋忓急": "该币种这个方向今天的真实盈亏表现偏弱",
    "璇ュ竵绉嶆渶杩戞粴鍔ㄧ湡瀹炰簭鎹熻繃澶?": "该币种最近滚动真实亏损过大",
    "璇ュ竵绉嶄粖澶╃疮璁＄湡瀹炰簭鎹熻秴杩囬檺鍒?": "该币种今天累计真实亏损超过限制",
    "璇ュ竵绉嶈繖涓柟鍚戠殑鐪熷疄浜忔崯宸茬粡瓒呰繃闄愬埗": "该币种这个方向的真实亏损已经超过限制",
    "璇ュ竵绉嶆鍦ㄨ鍙︿竴鏉″垎鏋愭祦绋嬪鐞嗭紝鏈寮€浠撴墽琛岃烦杩囷紝绛夊緟涓嬩竴杞噸鏂拌瘎浼般€?": "该币种正在被另一条分析流程处理，本次开仓执行跳过，等待下一轮重新评估。",
    "璁㈠崟宸叉垚浜ゃ€?": "订单已成交。",
    "鏈墽琛岋細": "未执行：",
}


PARTIAL_REPLACEMENTS = (
    ("骞崇┖", "平空"),
    ("骞冲", "平多"),
    ("鍋氬", "做多"),
    ("鍋氱┖", "做空"),
    ("瑙傛湜", "观望"),
    ("浠婂ぉ杩樻病", "今天还没有"),
    (
        "璇ュ竵绉嶈繖涓柟鍚戜粖澶╃殑鐪熷疄鐩堜簭琛ㄧ幇鍋忓急",
        "该币种这个方向今天的真实盈亏表现偏弱",
    ),
    ("鏈轰細璇勫垎鎺掑悕闈犲墠", "机会评分排名靠前"),
    ("鏈疆", "本轮"),
    ("寮€浠?", "开仓"),
    ("骞充粨", "平仓"),
    ("瑙傛湜", "观望"),
    ("鍋氬", "做多"),
    ("鍋氱┖", "做空"),
    ("鐩堝埄", "盈利"),
    ("浜忔崯", "亏损"),
)


def _business_rewrite(text: str) -> str | None:
    """Recover high-frequency trading messages whose UTF-8 text was stored as mojibake."""
    if "澶氭ā鍨" in text and "瑙傛湜" in text:
        return "多模型裁决结果为观望，未提交订单。"
    if "鎸佷粨澶嶇洏" in text and ("缁" in text or "寔鏈" in text):
        return "持仓复盘结论为继续持有或暂不加仓，未提交订单。"
    if "OKX 宸茶繑鍥炲钩浠撴垚浜" in text:
        return "OKX 已返回平仓成交，系统同步为平仓记录；这通常来自 OKX 止盈/止损、手动平仓或交易所侧自动平仓。"
    if "OKX 宸叉病鏈夎繖绗旀寔浠" in text:
        return "OKX 已没有这笔持仓，但没有查到对应平仓成交回报；系统按交易所仓位状态同步为平仓，盈亏暂按 0 记录。"
    if "OKX 宸叉棤瀵瑰簲鎸佷粨" in text:
        return "OKX 已无对应持仓，本地未同步持仓已关闭。"
    if "OKX 宸叉湁鎸佷粨" in text and "鏈湴" in text:
        return "OKX 已有持仓，本地缺失，系统已按执行订单补回持仓记录。"
    if "OKX 浠嶆湁鎸佷粨" in text and "閲嶆柊" in text:
        return "OKX 仍有持仓，本地之前误记为已平仓，系统已重新打开本地持仓记录。"
    if "OKX 鎸佷粨鏁伴噺" in text:
        return "OKX 持仓数量或价格已变化，本地持仓已同步更新。"
    if "蹇" in text and "鎺цЕ" in text and ("姝㈡崯" in text or "浜忔崯" in text):
        pct_match = re.search(r"(\d+(?:\.\d+)?)%", text)
        pct = f"当前进度约 {pct_match.group(1)}%，" if pct_match else ""
        return f"快速风控触发：{pct}亏损接近或触及止损风险，系统优先提交平仓控制单笔亏损。"
    if (
        ("1-5" in text or "鐭" in text)
        and ("蹇" in text or "鎺цЕ" in text)
        and ("鍙嶅悜" in text or "娉㈠姩" in text)
    ):
        return "快速风控触发：1-5 分钟短线波动明显反向，系统优先减仓或平仓控制风险。"
    if "蹇" in text and "鎺цЕ" in text and "姝㈢泩" in text:
        return "快速风控触发：价格已经触及本地记录的止盈位，系统优先提交平仓锁定结果。"
    if "鐩堝埄淇濇姢瑙﹀彂" in text:
        return "盈利保护触发：持仓已有浮盈，但利润开始明显回撤，系统优先保护已获得收益。"
    if "骞充粨淇濇姢" in text:
        return "平仓保护：当前信号还不足以主动平仓，系统继续持有并等待更明确的止盈、止损或趋势反转信号。"
    if "涓嬪崟鍓" in text and "鏈€鏂颁环鏍" in text:
        return "下单前没有重新拿到最新价格，系统不使用过期行情盲目下单，本次跳过。"
    if "涓嬪崟鍓" in text and "涓婃定" in text:
        return "下单前价格已经较分析时上涨超过允许偏移。为避免追高，本次不执行。"
    if "涓嬪崟鍓" in text and "涓嬭穼" in text:
        return "下单前价格已经较分析时下跌超过允许偏移。为避免追空，本次不执行。"
    if "涓嬪崟鍓" in text and "琛屾儏" in text:
        return "下单前价格较分析时波动过大，行情变化太快，本次不执行，等待下一轮重新判断。"
    if "鍚屼竴鏉" in text and "钩浠" in text and "璁" in text:
        return "同一条平仓决策已经生成过订单，为避免重复平仓，本次重复进入执行流程已跳过。"
    if "鍚屼竴鏉" in text and "紑浠" in text and "璁" in text:
        return "同一条开仓决策已经生成过订单，为避免重复开仓，本次重复进入执行流程已跳过。"
    if "OKX 骞充粨" in text and "閮ㄥ垎" in text:
        return "OKX 平仓已部分成交，系统会继续同步最终成交结果，不会重复提交平仓单。"
    if "OKX 骞充粨璁㈠崟" in text and ("杩藉崟" in text or "绛夊緟" in text):
        return "OKX 平仓订单正在追单或等待成交，系统不会重复提交平仓单。"
    if "璁㈠崟宸叉垚浜" in text:
        return "订单已成交。"
    if "OKX 褰撳墠鍙" in text and "浣欓" in text:
        return "OKX 当前可用余额不足，订单未提交。请检查 OKX 余额、保证金占用和账户风控状态。"
    if "浜ゆ槗鎺ュ彛鏈" in text and "OKX 璁" in text:
        return "交易接口未返回执行结果，系统没有拿到 OKX 订单号，也没有生成本地订单；本次裁决已按未执行处理。"
    if "OKX 涓嬪崟" in text and "瓒呮椂" in text:
        return "OKX 下单或确认超时，系统没有拿到最终订单结果；本次裁决已按未执行处理。"
    if "浠婃棩鐩堜" in text and "淇濇姢" in text:
        return "今日盈利回落触发目标保护线，系统将降低新开仓节奏，优先守住已实现利润；已有持仓仍会继续复盘、止盈止损和平仓处理。"
    if "鍖椾含鏃堕棿浠婃棩" in text:
        return "北京时间今日同向参与表现已用于调整专家权重。"
    if "鏆傛棤瓒冲" in text:
        return "暂无足够历史样本，使用基础权重。"
    if "杩戞湡璁板繂" in text and "鎻愰珮" in text:
        return "近期记忆中成功样本较多或正向教训更稳定，专家权重已提高。"
    if "杩戞湡璁板繂" in text and "闄嶅埌" in text:
        return "近期记忆提示该专家相关场景亏损偏多，专家权重已降低。"
    if "鍘嗗彶鏍锋湰" in text:
        return "历史样本未显示明显优劣，保持基础权重。"
    if "瀹為檯鏇翠紭鏂瑰悜" in text:
        return "实际更优方向已记录，用于后续复盘。"
    if "褰撴椂閫夋嫨瑙傛湜" in text:
        return "当时选择观望，但后续行情出现可复盘机会，已记录为影子复盘样本。"
    return None


def looks_mojibake(value: str | None) -> bool:
    text = str(value or "")
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def _repair_by_redecode(text: str) -> str:
    if not looks_mojibake(text):
        return text
    best = text
    best_score = _mojibake_score(text)
    for encoding in ("gbk", "cp936"):
        try:
            candidate = text.encode(encoding, errors="ignore").decode("utf-8", errors="replace")
        except UnicodeError:
            continue
        score = _mojibake_score(candidate)
        if score < best_score:
            best = candidate
            best_score = score
    return best


def _mojibake_score(text: str) -> int:
    return (
        sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
        + text.count("�") * 3
        + text.count("?")
    )


def sanitize_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not value:
        return value
    if value in EXACT_REPLACEMENTS:
        return EXACT_REPLACEMENTS[value]
    text = value
    for src, dst in PARTIAL_REPLACEMENTS:
        text = text.replace(src, dst)
    business = _business_rewrite(text)
    if business:
        return business
    text = _repair_by_redecode(text)
    if text in EXACT_REPLACEMENTS:
        return EXACT_REPLACEMENTS[text]
    business = _business_rewrite(text)
    if business:
        return business
    for src, dst in EXACT_REPLACEMENTS.items():
        text = text.replace(src, dst)
    for src, dst in PARTIAL_REPLACEMENTS:
        text = text.replace(src, dst)
    text = text.replace("銆?", "。").replace("锛?", "，").replace("锛屾", "，").replace("锛", "，")
    text = text.replace("�?", "。").replace("€?", "。")
    if text.endswith("。?"):
        text = text[:-1]
    if looks_mojibake(text) and _mojibake_score(text) >= 6:
        return "该笔历史记录的原始说明已损坏，无法准确还原；请以当前执行状态、成交价格和 OKX 订单状态为准。"
    return text


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_payload(item) for key, item in value.items()}
    return value
