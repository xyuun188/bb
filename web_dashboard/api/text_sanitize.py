"""Display text cleanup helpers for dashboard APIs.

This module deliberately keeps legacy mojibake samples so old database records
can be repaired.  The damaged samples are stored as Unicode escape strings, not
as visible mojibake, so source files and dashboard output stay readable.
"""

from __future__ import annotations

import re
from typing import Any

_CONTROL_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _u(escaped: str) -> str:
    """Decode an ASCII Unicode-escape literal into the original damaged text."""

    return escaped.encode("ascii").decode("unicode_escape")


MOJIBAKE_MARKERS = (
    _u("\\u9351"),
    _u("\\u95c4"),
    _u("\\u9422"),
    _u("\\u951b"),
    _u("\\u7ecb"),
    _u("\\u93b5"),
    _u("\\u9357"),
    _u("\\u9a9e"),
    _u("\\u6d60"),
    _u("\\u8930"),
    _u("\\u64b3"),
    _u("\\u58a0"),
    _u("\\u93c3"),
    _u("\\u5815"),
    _u("\\u6d93"),
    _u("\\u9359"),
    _u("\\u7459"),
    _u("\\u7eef"),
    _u("\\u7481"),
    _u("\\u6434"),
    _u("\\u95ab"),
    _u("\\u93b8"),
    _u("\\u9366"),
    _u("\\u59af"),
    _u("\\u6d5c"),
    _u("\\u9429"),
    _u("\\u690b"),
    _u("\\u6942"),
    _u("\\u935a"),
    _u("\\u9352"),
    _u("\\u5bee"),
    _u("\\u6dc7"),
    _u("\\u60e7"),
    _u("\\u735f"),
    _u("\\u20ac"),
    _u("\\ue1da"),
    _u("\\u95ba"),
    _u("\\u95b8"),
    _u("\\u745c"),
    _u("\\u5a34"),
    _u("\\u986b"),
    _u("\\u6fa7"),
    _u("\\u9470"),
)


EXACT_REPLACEMENTS = {
    _u("\\u9351\\u5fce\\u7ca8\\u93ba\\u3224\\u62e1"): "减仓探针",
    _u(
        "\\u95c4\\u5d84\\u7d86\\u6d60\\u64b2\\u7d85\\u9a9e\\u8dfa\\u60ce"
        "\\u9422\\u3126\\u5e30\\u95bd\\u5818\\u20ac?"
    ): "降低仓位并启用探针",
    _u(
        "\\u9352\\u55d8\\ue120\\u6fb6\\u0441\\u20ac\\u4f79\\u5c1d\\u9354\\u3125"
        "\\u7d13\\u752f\\u544a\\u57a8\\u5a0c\\u2103\\u6e41\\u9429\\u581d\\u57c4"
        "\\u59af\\u2033\\u7037\\u935a\\u5c7d\\u609c\\u7ead\\ue1bf\\ue17b\\u951b"
        "\\u5c7c\\u7e5a\\u93b8?1.15+ \\u6942\\u6a40\\u68ec\\u59b2\\u6d96\\u20ac?"
    ): "分歧大、波动异常或没有盈利模型同向确认，保持 1.15+ 高门槛。",
    _u(
        "ML \\u6d93\\u5ea2\\u6e47\\u9354\\u2033\\u6ad2\\u9429\\u581d\\u57c4"
        "\\u59af\\u2033\\u7037\\u935a\\u5c7d\\u609c\\u6d93\\u65c8\\ue569\\u93c8"
        "\\u71b8\\u6579\\u9429\\u5a41\\u8d1f\\u59dd\\uff4f\\u7d1d\\u934f\\u4f7d"
        "\\ue18f 0.75+ \\u704f\\u5fce\\u7ca8\\u5bee\\u20ac\\u6d60\\u64b1\\u20ac?"
    ): "ML 与服务器盈利模型同向且预期收益为正，允许 0.75+ 小仓开仓。",
    _u(
        "ML \\u93b4\\u6828\\u6e47\\u9354\\u2033\\u6ad2\\u9429\\u581d\\u57c4"
        "\\u59af\\u2033\\u7037\\u6d93?AI \\u93c2\\u7470\\u609c\\u935a\\u5c7d"
        "\\u609c\\u6d93\\u65c8\\ue569\\u93c8\\u71b8\\u6579\\u9429\\u5a41\\u8d1f"
        "\\u59dd\\uff4f\\u7d1d\\u934f\\u4f7d\\ue18f 0.85+ \\u704f\\u5fce\\u7ca8"
        "\\u5bee\\u20ac\\u6d60\\u64b1\\u20ac?"
    ): "ML 或服务器盈利模型与 AI 方向同向且预期收益为正，允许 0.85+ 小仓开仓。",
    _u(
        "\\u6d93\\u64b3\\ue18d\\u93c2\\u7470\\u609c\\u6d93\\u20ac\\u9477"
        "\\u7fe0\\u7b16\\u68f0\\u52ec\\u6e61\\u93c0\\u5241\\u6ced\\u6d93\\u70d8"
        "\\ue11c\\u951b\\u5c7d\\u5391\\u7481?0.90+ \\u5bee\\u20ac\\u6d60\\u64b1\\u20ac?"
    ): "专家方向一致且预期收益为正，允许 0.90+ 开仓。",
    _u(
        "\\u6d60\\u5a42\\u3049\\u6769\\u6a3b\\u75c5\\u93c8\\u590e\\ue1da"
        "\\u752f\\u4f7a\\ue752\\u93c2\\u7470\\u609c\\u9428\\u52ed\\u6e61\\u7039"
        "\\u70b2\\u94a9\\u6d60\\u64b9\\ue187\\u8930\\u66d8\\u20ac?"
    ): "今天还没有该币种方向的真实平仓记录。",
    _u(
        "ML \\u8930\\u64b3\\u58a0\\u6748\\u70ac\\u7223\\u951b\\u5c7d\\u5f2c"
        "\\u6d93\\u5ea2\\u6e80\\u6d7c\\u6c33\\u760e\\u9352\\u55d0\\u20ac?"
    ): "ML 当前达标，参与机会评分。",
    _u(
        "ML \\u8930\\u64b3\\u58a0\\u6fb6\\u52ea\\u7c2c\\u701b\\ufe3f\\u7bc4"
        "\\u7459\\u509a\\u7642\\u6d93\\ue15f\\u57a8\\u7487\\u30e6\\u67df\\u935a"
        "\\u621e\\u6e6d\\u6748\\u70ac\\u7223\\u951b\\u5c7e\\u6e70\\u5a06\\u2103"
        "\\u6e80\\u6d7c\\u6c33\\u760e\\u9352\\u55d5\\u7b09\\u6d63\\u8de8\\u6564"
        " ML \\u9354\\u72b2\\u567a\\u9352\\u55d0\\u20ac?"
    ): "ML 当前处于学习观察中，或该方向未达标，本次机会评分不使用 ML 加减分。",
    _u(
        "\\u93c8\\u8f70\\u7d30\\u7487\\u52eb\\u578e\\u93ba\\u6391\\u6095"
        "\\u95c8\\u72b2\\u58a0\\u951b\\u5c7e\\u6e70\\u675e\\ue1bf\\u7e58\\u934f"
        "\\u30e6\\u58bd\\u741b\\u5c84\\u69e6\\u9352\\u693c\\u20ac?"
    ): "机会评分排名靠前，本轮进入执行队列。",
    _u(
        "\\u93c8\\u8f70\\u7d30\\u7487\\u52eb\\u578e\\u93ba\\u6391\\u6095"
        "\\u95c8\\u72b2\\u58a0\\u951b\\u5c7e\\u6e70\\u675e\\ue1bc\\u51e1\\u6769"
        "\\u6d98\\u53c6\\u93b5\\u0446\\ue511\\u95c3\\u71b7\\u57aa\\u951b\\u5c7e"
        "\\ue11c\\u9366\\u3128\\u7e58\\u741b\\u5c7c\\u7b05\\u9357\\u66de\\u58a0"
        "\\u59ab\\u20ac\\u93cc\\u30e5\\u62f0\\u93bb\\u612a\\u6c26\\u7481\\u3220"
        "\\u5d1f\\u9286?"
    ): "机会评分排名靠前，本轮已进入执行队列，正在进行下单前检查和提交订单。",
    _u(
        "AI \\u95ab\\u590b\\u5ae8\\u7459\\u509b\\u6e5c\\u951b\\u5c7e\\u6e6d\\u93bb\\u612a\\u6c26\\u7481\\u3220\\u5d1f\\u9286?"
    ): "AI 选择观望，未提交订单。",
    _u(
        "\\u690b\\u5ea2\\u5e36\\u5bee\\u66df\\u6438\\u93b7\\u6394\\u7cb7\\u7487\\u30e5\\u5585\\u7edb\\u6825\\u20ac?"
    ): "风控引擎拒绝该决策。",
    _u(
        "\\u690b\\u5ea2\\u5e36\\u5bee\\u66df\\u6438\\u93b7\\u6394\\u7cb7\\u7487\\u30e5\\ue63f\\u59af\\u2033\\u7037\\u7441\\u4f78\\u5585\\u9286?"
    ): "风控引擎拒绝该多模型裁决。",
    _u(
        "\\u7487\\u30e5\\u7af5\\u7ec9\\u5d88\\u7e56\\u6d93\\ue045\\u67df\\u935a\\u621c\\u7c96\\u6fb6\\u2543\\u6b91\\u942a\\u71b7\\u7584\\u9429\\u581c\\u7c2d\\u741b\\u3127\\u5e47\\u934b\\u5fd3\\u6025"
    ): "该币种这个方向今天的真实盈亏表现偏弱",
    _u(
        "\\u7487\\u30e5\\u7af5\\u7ec9\\u5d86\\u6e36\\u6769\\u621e\\u7cb4\\u9354\\u3127\\u6e61\\u7039\\u70b0\\u7c2d\\u93b9\\u71bb\\u7e43\\u6fb6?"
    ): "该币种最近滚动真实亏损过大",
    _u(
        "\\u7487\\u30e5\\u7af5\\u7ec9\\u5d84\\u7c96\\u6fb6\\u2543\\u75ae\\u7481\\uff04\\u6e61\\u7039\\u70b0\\u7c2d\\u93b9\\u71bb\\u79f4\\u6769\\u56ec\\u6aba\\u9352?"
    ): "该币种今天累计真实亏损超过限制",
    _u(
        "\\u7487\\u30e5\\u7af5\\u7ec9\\u5d88\\u7e56\\u6d93\\ue045\\u67df\\u935a\\u6220\\u6b91\\u942a\\u71b7\\u7584\\u6d5c\\u5fd4\\u5d2f\\u5bb8\\u832c\\u7ca1\\u74d2\\u5470\\u7e43\\u95c4\\u612c\\u57d7"
    ): "该币种这个方向的真实亏损已经超过限制",
    _u(
        "\\u7487\\u30e5\\u7af5\\u7ec9\\u5d86\\ue11c\\u9366\\u3128\\ue766\\u9359"
        "\\ufe3f\\u7af4\\u93c9\\u2033\\u578e\\u93cb\\u612d\\u7966\\u7ecb\\u5b2a"
        "\\ue629\\u941e\\u55ed\\u7d1d\\u93c8\\ue101\\ue0bc\\u5bee\\u20ac\\u6d60"
        "\\u64b4\\u58bd\\u741b\\u5c83\\u70e6\\u6769\\u56f7\\u7d1d\\u7edb\\u590a"
        "\\u7ddf\\u6d93\\u5b29\\u7af4\\u675e\\ue1c0\\u5678\\u93c2\\u62cc\\u760e"
        "\\u6d7c\\u822c\\u20ac?"
    ): "该币种正在被另一条分析流程处理，本次开仓执行跳过，等待下一轮重新评估。",
    _u("\\u7481\\u3220\\u5d1f\\u5bb8\\u53c9\\u579a\\u6d5c\\u3083\\u20ac?"): "订单已成交。",
    _u("\\u93c8\\ue045\\u58bd\\u741b\\u5c8b\\u7d30"): "未执行：",
}


PARTIAL_REPLACEMENTS = (
    (_u("\\u9a9e\\u5d07\\u2516"), "平空"),
    (_u("\\u9a9e\\u51b2\\ue63f"), "平多"),
    (_u("\\u934b\\u6c2c\\ue63f"), "做多"),
    (_u("\\u934b\\u6c31\\u2516"), "做空"),
    (_u("\\u7459\\u509b\\u6e5c"), "观望"),
    (_u("\\u6d60\\u5a42\\u3049\\u6769\\u6a3b\\u75c5"), "今天还没有"),
    (
        _u(
            "\\u7487\\u30e5\\u7af5\\u7ec9\\u5d88\\u7e56\\u6d93\\ue045\\u67df\\u935a\\u621c\\u7c96\\u6fb6\\u2543\\u6b91\\u942a\\u71b7\\u7584\\u9429\\u581c\\u7c2d\\u741b\\u3127\\u5e47\\u934b\\u5fd3\\u6025"
        ),
        "该币种这个方向今天的真实盈亏表现偏弱",
    ),
    (
        _u("\\u93c8\\u8f70\\u7d30\\u7487\\u52eb\\u578e\\u93ba\\u6391\\u6095\\u95c8\\u72b2\\u58a0"),
        "机会评分排名靠前",
    ),
    (_u("\\u93c8\\ue103\\u7586"), "本轮"),
    (_u("\\u5bee\\u20ac\\u6d60?"), "开仓"),
    (_u("\\u9a9e\\u5145\\u7ca8"), "平仓"),
    (_u("\\u9429\\u581d\\u57c4"), "盈利"),
    (_u("\\u6d5c\\u5fd4\\u5d2f"), "亏损"),
)


def _business_rewrite(text: str) -> str | None:
    """Recover high-frequency trading messages whose text was stored damaged."""

    if _u("\\u6fb6\\u6c2d\\u0101\\u9368") in text and _u("\\u7459\\u509b\\u6e5c") in text:
        return "多模型裁决结果为观望，未提交订单。"
    if _u("\\u93b8\\u4f77\\u7ca8\\u6fb6\\u5d87\\u6d0f") in text and (
        _u("\\u7f01") in text or _u("\\u5bd4\\u93c8") in text
    ):
        return "持仓复盘结论为继续持有或暂不加仓，未提交订单。"
    if _u("OKX \\u5bb8\\u8336\\u7e51\\u9365\\u70b2\\u94a9\\u6d60\\u64b4\\u579a\\u6d5c") in text:
        return "OKX 已返回平仓成交，系统同步为平仓记录；这通常来自 OKX 止盈/止损、手动平仓或交易所侧自动平仓。"
    if _u("OKX \\u5bb8\\u53c9\\u75c5\\u93c8\\u590e\\u7e56\\u7ed7\\u65c0\\u5bd4\\u6d60") in text:
        return "OKX 已没有这笔持仓，但没有查到对应平仓成交回报；系统按交易所仓位状态同步为平仓，盈亏暂按 0 记录。"
    if _u("OKX \\u5bb8\\u53c9\\u68e4\\u7035\\u7470\\u7c32\\u93b8\\u4f77\\u7ca8") in text:
        return "OKX 已无对应持仓，本地未同步持仓已关闭。"
    if (
        _u("OKX \\u5bb8\\u53c9\\u6e41\\u93b8\\u4f77\\u7ca8") in text
        and _u("\\u93c8\\ue100\\u6e74") in text
    ):
        return "OKX 已有持仓，本地缺失，系统已按执行订单补回持仓记录。"
    if (
        _u("OKX \\u6d60\\u5d86\\u6e41\\u93b8\\u4f77\\u7ca8") in text
        and _u("\\u95b2\\u5d86\\u67ca") in text
    ):
        return "OKX 仍有持仓，本地之前误记为已平仓，系统已重新打开本地持仓记录。"
    if _u("OKX \\u93b8\\u4f77\\u7ca8\\u93c1\\u4f34\\u567a") in text:
        return "OKX 持仓数量或价格已变化，本地持仓已同步更新。"
    if (
        _u("\\u8e47") in text
        and _u("\\u93ba\\u0446\\u0415") in text
        and (_u("\\u59dd\\u3221\\u5d2f") in text or _u("\\u6d5c\\u5fd4\\u5d2f") in text)
    ):
        pct_match = re.search(r"(\d+(?:\.\d+)?)%", text)
        pct = f"当前进度约 {pct_match.group(1)}%，" if pct_match else ""
        return f"快速风控触发：{pct}亏损接近或触及止损风险，系统优先提交平仓控制单笔亏损。"
    if (
        ("1-5" in text or _u("\\u942d") in text)
        and (_u("\\u8e47") in text or _u("\\u93ba\\u0446\\u0415") in text)
        and (_u("\\u9359\\u5d85\\u609c") in text or _u("\\u5a09\\u3220\\u59e9") in text)
    ):
        return "快速风控触发：1-5 分钟短线波动明显反向，系统优先减仓或平仓控制风险。"
    if (
        _u("\\u8e47") in text
        and _u("\\u93ba\\u0446\\u0415") in text
        and _u("\\u59dd\\u3222\\u6ce9") in text
    ):
        return "快速风控触发：价格已经触及本地记录的止盈位，系统优先提交平仓锁定结果。"
    if _u("\\u9429\\u581d\\u57c4\\u6dc7\\u6fc7\\u59e2\\u7459\\ufe40\\u5f42") in text:
        return "盈利保护触发：持仓已有浮盈，但利润开始明显回撤，系统优先保护已获得收益。"
    if _u("\\u9a9e\\u5145\\u7ca8\\u6dc7\\u6fc7\\u59e2") in text:
        return "平仓保护：当前信号还不足以主动平仓，系统继续持有并等待更明确的止盈、止损或趋势反转信号。"
    if (
        _u("\\u6d93\\u5b2a\\u5d1f\\u9353") in text
        and _u("\\u93c8\\u20ac\\u93c2\\u9881\\u73af\\u93cd") in text
    ):
        return "下单前没有重新拿到最新价格，系统不使用过期行情盲目下单，本次跳过。"
    if _u("\\u6d93\\u5b2a\\u5d1f\\u9353") in text and _u("\\u6d93\\u5a43\\u5b9a") in text:
        return "下单前价格已经较分析时上涨超过允许偏移。为避免追高，本次不执行。"
    if _u("\\u6d93\\u5b2a\\u5d1f\\u9353") in text and _u("\\u6d93\\u5b2d\\u7a7c") in text:
        return "下单前价格已经较分析时下跌超过允许偏移。为避免追空，本次不执行。"
    if _u("\\u6d93\\u5b2a\\u5d1f\\u9353") in text and _u("\\u741b\\u5c7e\\u510f") in text:
        return "下单前价格较分析时波动过大，行情变化太快，本次不执行，等待下一轮重新判断。"
    if (
        _u("\\u935a\\u5c7c\\u7af4\\u93c9") in text
        and _u("\\u94a9\\u6d60") in text
        and _u("\\u7481") in text
    ):
        return "同一条平仓决策已经生成过订单，为避免重复平仓，本次重复进入执行流程已跳过。"
    if (
        _u("\\u935a\\u5c7c\\u7af4\\u93c9") in text
        and _u("\\u7d11\\u6d60") in text
        and _u("\\u7481") in text
    ):
        return "同一条开仓决策已经生成过订单，为避免重复开仓，本次重复进入执行流程已跳过。"
    if _u("OKX \\u9a9e\\u5145\\u7ca8") in text and _u("\\u95ae\\u3125\\u578e") in text:
        return "OKX 平仓已部分成交，系统会继续同步最终成交结果，不会重复提交平仓单。"
    if _u("OKX \\u9a9e\\u5145\\u7ca8\\u7481\\u3220\\u5d1f") in text and (
        _u("\\u6769\\u85c9\\u5d1f") in text or _u("\\u7edb\\u590a\\u7ddf") in text
    ):
        return "OKX 平仓订单正在追单或等待成交，系统不会重复提交平仓单。"
    if _u("\\u7481\\u3220\\u5d1f\\u5bb8\\u53c9\\u579a\\u6d5c") in text:
        return "订单已成交。"
    if _u("OKX \\u8930\\u64b3\\u58a0\\u9359") in text and _u("\\u6d63\\u6b13") in text:
        return "OKX 当前可用余额不足，订单未提交。请检查 OKX 余额、保证金占用和账户风控状态。"
    if _u("\\u6d5c\\u3086\\u69d7\\u93ba\\u30e5\\u5f5b\\u93c8") in text and "OKX 订" in text:
        return "交易接口未返回执行结果，系统没有拿到 OKX 订单号，也没有生成本地订单；本次裁决已按未执行处理。"
    if _u("OKX \\u6d93\\u5b2a\\u5d1f") in text and _u("\\u74d2\\u546e\\u6902") in text:
        return "OKX 下单或确认超时，系统没有拿到最终订单结果；本次裁决已按未执行处理。"
    if _u("\\u6d60\\u5a43\\u68e9\\u9429\\u581c") in text and _u("\\u6dc7\\u6fc7\\u59e2") in text:
        return "今日盈利回落触发目标保护线，系统将降低新开仓节奏，优先守住已实现利润；已有持仓仍会继续复盘、止盈止损和平仓处理。"
    if _u("\\u9356\\u693e\\u542b\\u93c3\\u5815\\u68ff\\u6d60\\u5a43\\u68e9") in text:
        return "北京时间今日同向参与表现已用于调整专家权重。"
    if _u("\\u93c6\\u509b\\u68e4\\u74d2\\u51b2") in text:
        return "暂无足够历史样本，使用基础权重。"
    if (
        _u("\\u6769\\u621e\\u6e61\\u7481\\u677f\\u7e42") in text
        and _u("\\u93bb\\u6130\\u73ee") in text
    ):
        return "近期记忆中成功样本较多或正向教训更稳定，专家权重已提高。"
    if (
        _u("\\u6769\\u621e\\u6e61\\u7481\\u677f\\u7e42") in text
        and _u("\\u95c4\\u5d85\\u57cc") in text
    ):
        return "近期记忆提示该专家相关场景亏损偏多，专家权重已降低。"
    if _u("\\u9358\\u55d7\\u5f76\\u93cd\\u950b\\u6e70") in text:
        return "历史样本未显示明显优劣，保持基础权重。"
    if _u("\\u7039\\u70ba\\u6aaf\\u93c7\\u7fe0\\u7d2d\\u93c2\\u7470\\u609c") in text:
        return "实际更优方向已记录，用于后续复盘。"
    if _u("\\u8930\\u64b4\\u6902\\u95ab\\u590b\\u5ae8\\u7459\\u509b\\u6e5c") in text:
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
        + text.count("\ufffd") * 3
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
    text = (
        text.replace(_u("\\u9286?"), "。")
        .replace(_u("\\u951b?"), "，")
        .replace(_u("\\u951b\\u5c7e"), "，")
        .replace(_u("\\u951b"), "，")
    )
    text = text.replace("\ufffd?", "。").replace(_u("\\u20ac?"), "。")
    if text.endswith("。?"):
        text = text[:-1]
    text = _CONTROL_TEXT_RE.sub(" ", text)
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
