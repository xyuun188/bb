"""
Black swan event detection via extreme news keywords and price movements.

This module only classifies risk. The trading service may still reconfirm
price-action alerts with a fresh market snapshot before blocking a new entry,
because exchange feature snapshots can occasionally contain bad 1m returns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

BLACK_SWAN_KEYWORDS = [
    r"hack(ed|ing)?",
    r"exploit(ed)?",
    r"rug.?pull",
    r"ban(ning|ned)?\s+(china|us|sec)",
    r"lawsuit",
    r"sec\s+(sues|charges|investigat)",
    r"exchange\s+(halt|suspends|freeze)",
    r"insolven(t|cy)",
    r"bank\s+run",
    r"depe?g",
    r"audit\s+(fail|issue)",
    r"bridge\s+(hack|exploit)",
    r"flash\s+crash",
    r"emergency\s+(withdraw|meeting)",
    r"delist(ing|ed)?",
]

FLASH_CRASH_THRESHOLD = -0.15
EXTREME_VOLUME_SPIKE = 5.0


@dataclass
class BlackSwanResult:
    triggered: bool
    severity: str = "none"  # none, warn, critical
    reason: str = ""
    keywords_matched: list[str] | None = None
    recommended_action: str = "none"  # none, close_all, reduce_only
    source: str = "none"  # none, sentiment, price_action, combined


class BlackSwanDetector:
    """Detect extreme market conditions from news and price action."""

    def __init__(self) -> None:
        self._compiled_patterns = [re.compile(kw, re.IGNORECASE) for kw in BLACK_SWAN_KEYWORDS]

    def check_sentiment(
        self, headlines: list[str], sentiment_scores: list[float]
    ) -> BlackSwanResult:
        matched: list[str] = []
        for headline in headlines:
            for pattern in self._compiled_patterns:
                if pattern.search(headline):
                    matched.append(headline)
                    break

        if len(matched) >= 3:
            return BlackSwanResult(
                triggered=True,
                severity="critical",
                reason=f"检测到 {len(matched)} 条重大风险新闻，可能影响市场稳定。",
                keywords_matched=matched,
                recommended_action="close_all",
                source="sentiment",
            )
        if matched:
            avg_sent = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            if avg_sent < -0.5:
                return BlackSwanResult(
                    triggered=True,
                    severity="critical",
                    reason=f"检测到重大风险关键词，且新闻情绪极端负面（情绪值 {avg_sent:.2f}）。",
                    keywords_matched=matched,
                    recommended_action="close_all",
                    source="sentiment",
                )
            return BlackSwanResult(
                triggered=True,
                severity="warn",
                reason=f"检测到潜在重大风险新闻：{matched[0][:80]}...",
                keywords_matched=matched,
                recommended_action="reduce_only",
                source="sentiment",
            )

        return BlackSwanResult(triggered=False)

    def check_price_action(self, price_change_1m: float, volume_ratio: float) -> BlackSwanResult:
        """Detect extreme short-term moves from feature data."""
        if price_change_1m < FLASH_CRASH_THRESHOLD:
            severity = "critical" if price_change_1m < -0.25 else "warn"
            return BlackSwanResult(
                triggered=True,
                severity=severity,
                reason=(
                    f"检测到 1 分钟异常暴跌 {price_change_1m * 100:.1f}%，"
                    "可能是真实闪崩、插针，也可能是行情特征数据异常。"
                ),
                recommended_action="close_all" if severity == "critical" else "reduce_only",
                source="price_action",
            )

        if price_change_1m < -0.05 and volume_ratio > EXTREME_VOLUME_SPIKE:
            return BlackSwanResult(
                triggered=True,
                severity="warn",
                reason=(
                    f"检测到放量急跌：1 分钟跌幅 {price_change_1m * 100:.1f}%，"
                    f"成交量约为平时 {volume_ratio:.1f} 倍。"
                ),
                recommended_action="reduce_only",
                source="price_action",
            )

        return BlackSwanResult(triggered=False)

    def check_combined(
        self,
        headlines: list[str],
        sentiment_scores: list[float],
        price_change_1m: float,
        volume_ratio: float,
    ) -> BlackSwanResult:
        sentiment_result = self.check_sentiment(headlines, sentiment_scores)
        price_result = self.check_price_action(price_change_1m, volume_ratio)
        reason = "；".join(r for r in (sentiment_result.reason, price_result.reason) if r)

        if sentiment_result.severity == "critical" or price_result.severity == "critical":
            source = (
                "combined"
                if sentiment_result.triggered and price_result.triggered
                else (
                    sentiment_result.source if sentiment_result.triggered else price_result.source
                )
            )
            return BlackSwanResult(
                triggered=True,
                severity="critical",
                reason=reason,
                keywords_matched=sentiment_result.keywords_matched,
                recommended_action="close_all",
                source=source,
            )
        if sentiment_result.triggered or price_result.triggered:
            source = (
                "combined"
                if sentiment_result.triggered and price_result.triggered
                else (
                    sentiment_result.source if sentiment_result.triggered else price_result.source
                )
            )
            return BlackSwanResult(
                triggered=True,
                severity="warn",
                reason=reason,
                keywords_matched=sentiment_result.keywords_matched,
                recommended_action="reduce_only",
                source=source,
            )
        return BlackSwanResult(triggered=False)
