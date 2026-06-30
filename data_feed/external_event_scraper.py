"""Optional external event scraper powered by Scrapling.

This module is intentionally kept outside the trading hot path.  It only fetches
administrator-configured public HTTPS pages and converts them into normalized
news-like dictionaries that can be persisted as text sentiment training samples.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import inspect
import ipaddress
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse

import structlog

from config.settings import settings
from core.safe_output import safe_error_text
from core.url_safety import normalize_external_http_url

logger = structlog.get_logger(__name__)

SCRAPLING_SOURCE_PREFIX = "scrapling:"
EXTERNAL_EVENT_MAX_SOURCES_LIMIT = 32
_MAX_HTML_CHARS = 600_000
_ANCHOR_SCAN_LIMIT = 250
_BLOCKED_HOSTS = {"localhost", "localhost.localdomain"}
_EVENT_HINTS = (
    "announcement",
    "list",
    "delist",
    "airdrop",
    "token",
    "upgrade",
    "mainnet",
    "hack",
    "exploit",
    "security",
    "partnership",
    "launch",
    "staking",
    "etf",
    "sec",
    "regulatory",
)
_DEFAULT_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("BTC", "Bitcoin"),
    "ETH": ("ETH", "Ethereum"),
    "SOL": ("SOL", "Solana"),
    "BNB": ("BNB", "Binance"),
    "XRP": ("XRP", "Ripple"),
    "DOGE": ("DOGE", "Dogecoin"),
    "ADA": ("ADA", "Cardano"),
    "AVAX": ("AVAX", "Avalanche"),
    "DOT": ("DOT", "Polkadot"),
    "LINK": ("LINK", "Chainlink"),
    "UNI": ("UNI", "Uniswap"),
    "SUI": ("SUI", "Sui"),
    "APT": ("APT", "Aptos"),
    "ARB": ("ARB", "Arbitrum"),
    "OP": ("OP", "Optimism"),
    "TON": ("TON", "Toncoin"),
    "MATIC": ("MATIC", "Polygon", "POL"),
    "POL": ("POL", "Polygon", "MATIC"),
    "NEAR": ("NEAR", "Near"),
    "TIA": ("TIA", "Celestia"),
    "EIGEN": ("EIGEN", "EigenLayer", "EigenCloud"),
    "KCS": ("KCS", "KuCoin"),
}
DEFAULT_EXTERNAL_EVENT_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "name": "binance_announcements",
        "url": "https://www.binance.com/en/support/announcement/c-48",
        "weight": 0.88,
        "category": "exchange",
        "description": "Binance 官方公告，覆盖上币、下架、维护和产品更新。",
    },
    {
        "name": "coinbase_blog",
        "url": "https://www.coinbase.com/blog",
        "weight": 0.72,
        "category": "exchange",
        "description": "Coinbase 官方博客，覆盖资产、合规和机构采用事件。",
    },
    {
        "name": "ethereum_blog",
        "url": "https://blog.ethereum.org/",
        "symbols": ["ETH"],
        "weight": 0.72,
        "category": "project",
        "description": "Ethereum 官方博客，覆盖协议升级和生态核心事件。",
    },
    {
        "name": "solana_news",
        "url": "https://solana.com/news",
        "symbols": ["SOL"],
        "weight": 0.70,
        "category": "project",
        "description": "Solana 官方新闻，覆盖生态、升级和合作事件。",
    },
)
RECOMMENDED_EXTERNAL_EVENT_SOURCES: tuple[dict[str, Any], ...] = DEFAULT_EXTERNAL_EVENT_SOURCES + (
    {
        "name": "okx_latest_announcements",
        "url": "https://www.okx.com/en-us/help/section/announcements-latest-announcements",
        "symbols": ["BTC", "ETH", "OKB"],
        "weight": 0.88,
        "category": "exchange",
        "description": "OKX 最新公告，覆盖交易、充提、维护和产品事件。",
    },
    {
        "name": "okx_new_listings",
        "url": "https://www.okx.com/en-us/help/section/announcements-new-listings",
        "weight": 0.90,
        "category": "exchange",
        "description": "OKX 新币上线公告，适合捕捉短线催化。",
    },
    {
        "name": "kucoin_announcements",
        "url": "https://www.kucoin.com/announcement",
        "symbols": ["KCS"],
        "weight": 0.82,
        "category": "exchange",
        "description": "KuCoin 官方公告，覆盖上币、下架和产品调整。",
    },
    {
        "name": "kraken_asset_listings",
        "url": "https://blog.kraken.com/category/product/asset-listings",
        "weight": 0.78,
        "category": "exchange",
        "description": "Kraken 资产上线博客，适合补充海外主流交易所上币事件。",
    },
    {
        "name": "kraken_status",
        "url": "https://status.kraken.com/",
        "weight": 0.74,
        "category": "exchange",
        "description": "Kraken 状态页，覆盖交易、充提和维护异常。",
    },
    {
        "name": "avalanche_blog",
        "url": "https://www.avax.network/about/blog",
        "symbols": ["AVAX"],
        "weight": 0.72,
        "category": "project",
        "description": "Avalanche 官方博客，覆盖生态和链上项目事件。",
    },
    {
        "name": "chainlink_blog",
        "url": "https://chain.link/blog",
        "symbols": ["LINK"],
        "weight": 0.70,
        "category": "project",
        "description": "Chainlink 官方博客，覆盖预言机、RWA 和机构合作。",
    },
    {
        "name": "uniswap_blog",
        "url": "https://blog.uniswap.org/",
        "symbols": ["UNI", "ETH"],
        "weight": 0.68,
        "category": "project",
        "description": "Uniswap 官方博客，覆盖 DEX、治理和协议更新。",
    },
    {
        "name": "base_blog",
        "url": "https://blog.base.org/",
        "symbols": ["ETH"],
        "weight": 0.66,
        "category": "project",
        "description": "Base 官方博客，覆盖 L2 生态和应用增长。",
    },
    {
        "name": "optimism_blog",
        "url": "https://www.optimism.io/blog",
        "symbols": ["OP", "ETH"],
        "weight": 0.68,
        "category": "project",
        "description": "Optimism 官方博客，覆盖 Superchain、治理和升级。",
    },
    {
        "name": "arbitrum_foundation_blog",
        "url": "https://blog.arbitrum.foundation/",
        "symbols": ["ARB", "ETH"],
        "weight": 0.68,
        "category": "project",
        "description": "Arbitrum Foundation 博客，覆盖 L2 生态和治理事件。",
    },
    {
        "name": "sui_blog",
        "url": "https://blog.sui.io/",
        "symbols": ["SUI"],
        "weight": 0.68,
        "category": "project",
        "description": "Sui 官方博客，覆盖生态、协议和合作事件。",
    },
    {
        "name": "aptos_currents",
        "url": "https://aptosnetwork.com/currents",
        "symbols": ["APT"],
        "weight": 0.68,
        "category": "project",
        "description": "Aptos 官方动态，覆盖生态和协议更新。",
    },
    {
        "name": "polygon_blog",
        "url": "https://polygon.technology/blog",
        "symbols": ["POL", "MATIC"],
        "weight": 0.70,
        "category": "project",
        "description": "Polygon 官方博客，覆盖 POL/MATIC、RWA、支付和生态事件。",
    },
    {
        "name": "near_blog",
        "url": "https://www.near.org/blog",
        "symbols": ["NEAR"],
        "weight": 0.68,
        "category": "project",
        "description": "NEAR 官方博客，覆盖 AI、链抽象和生态更新。",
    },
    {
        "name": "celestia_blog",
        "url": "https://blog.celestia.org/",
        "symbols": ["TIA"],
        "weight": 0.66,
        "category": "project",
        "description": "Celestia 官方博客，覆盖模块化链和 DA 事件。",
    },
    {
        "name": "eigencloud_blog",
        "url": "https://blog.eigencloud.xyz/",
        "symbols": ["EIGEN", "ETH"],
        "weight": 0.66,
        "category": "project",
        "description": "EigenCloud/EigenLayer 博客，覆盖再质押和 AVS 生态。",
    },
    {
        "name": "certik_blog",
        "url": "https://www.certik.com/blog",
        "weight": 0.72,
        "category": "security",
        "description": "CertiK 安全博客，覆盖攻击、漏洞和风险复盘。",
    },
    {
        "name": "slowmist_medium",
        "url": "https://slowmist.medium.com/",
        "weight": 0.70,
        "category": "security",
        "description": "SlowMist 安全事件源，覆盖被盗、钓鱼和链上攻击分析。",
    },
    {
        "name": "chainalysis_blog",
        "url": "https://www.chainalysis.com/blog/",
        "weight": 0.64,
        "category": "security",
        "description": "Chainalysis 博客，覆盖监管、犯罪资金流和重大风险事件。",
    },
    {
        "name": "stellar_blog",
        "url": "https://stellar.org/blog",
        "symbols": ["XLM"],
        "weight": 0.62,
        "category": "project",
        "description": "Stellar official blog for ecosystem, payment, and protocol events.",
    },
    {
        "name": "filecoin_blog",
        "url": "https://filecoin.io/blog/",
        "symbols": ["FIL"],
        "weight": 0.62,
        "category": "project",
        "description": "Filecoin official blog for storage, ecosystem, and protocol updates.",
    },
    {
        "name": "aave_blog",
        "url": "https://aave.com/blog",
        "symbols": ["AAVE", "ETH"],
        "weight": 0.64,
        "category": "defi",
        "description": "Aave official blog for DeFi, governance, and protocol updates.",
    },
    {
        "name": "circle_blog",
        "url": "https://www.circle.com/blog",
        "symbols": ["USDC"],
        "weight": 0.62,
        "category": "stablecoin",
        "description": "Circle blog for USDC, payments, institutional, and policy events.",
    },
    {
        "name": "tether_news",
        "url": "https://tether.io/en/news/",
        "symbols": ["USDT"],
        "weight": 0.62,
        "category": "stablecoin",
        "description": "Tether official news for USDT, reserves, chain support, and policy events.",
    },
    {
        "name": "sec_press_releases",
        "url": "https://www.sec.gov/newsroom/press-releases",
        "symbols": ["BTC", "ETH"],
        "weight": 0.70,
        "category": "regulatory",
        "description": "SEC public press releases for crypto asset enforcement and ETF events.",
    },
    {
        "name": "cftc_press_releases",
        "url": "https://www.cftc.gov/PressRoom/PressReleases",
        "symbols": ["BTC", "ETH"],
        "weight": 0.66,
        "category": "regulatory",
        "description": "CFTC public press releases for digital asset derivatives and enforcement.",
    },
)

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript|svg|canvas)\b[^>]*>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title\b[^>]*>(?P<value>.*?)</title>", flags=re.IGNORECASE | re.DOTALL)
_META_RE = re.compile(r"<meta\b(?P<attrs>[^>]*)>", flags=re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(
    r"(?P<key>[A-Za-z_:.-]+)\s*=\s*(?P<value>\"[^\"]*\"|'[^']*'|[^\s>]+)",
    flags=re.DOTALL,
)
_ANCHOR_RE = re.compile(
    r"<a\b(?P<attrs>[^>]*)>(?P<text>.*?)</a>",
    flags=re.IGNORECASE | re.DOTALL,
)
_WHITESPACE_RE = re.compile(r"\s+")


class AsyncFetcherLike(Protocol):
    """Subset of Scrapling's AsyncFetcher used by this project."""

    @classmethod
    def get(cls, url: str, **kwargs: Any) -> Any:
        """Return an awaitable Scrapling response."""


@dataclass(frozen=True)
class ExternalEventSource:
    name: str
    url: str
    symbols: tuple[str, ...] = ()
    weight: float = 0.60
    category: str = "project"
    description: str = ""


def _load_async_fetcher() -> type[AsyncFetcherLike] | None:
    try:
        from scrapling.fetchers import AsyncFetcher
    except Exception as exc:
        logger.debug(
            "scrapling unavailable; external event scraping disabled", error=safe_error_text(exc)
        )
        return None
    return AsyncFetcher


def _source_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = str(parsed.hostname or "external").lower().removeprefix("www.")
    return re.sub(r"[^a-z0-9_.-]+", "_", host)[:36] or "external"


def _reject_private_or_local_host(host: str) -> None:
    lowered = host.strip("[]").lower()
    if lowered in _BLOCKED_HOSTS or lowered.endswith((".local", ".internal", ".lan")):
        raise ValueError("external event scraper source host must be public.")
    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("external event scraper source IP must be globally routable.")


def _normalize_source(raw: dict[str, Any]) -> ExternalEventSource:
    raw_url = str(raw.get("url") or "").strip()
    url = normalize_external_http_url(
        raw_url,
        field_name="external event scraper source URL",
        allow_empty=False,
        max_length=500,
    )
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("external event scraper sources must use HTTPS.")
    if not parsed.hostname:
        raise ValueError("external event scraper source URL must include a hostname.")
    _reject_private_or_local_host(parsed.hostname)
    name = str(raw.get("name") or _source_name_from_url(url)).strip()
    name = re.sub(r"[^A-Za-z0-9_.:-]+", "_", name)[:36] or _source_name_from_url(url)
    symbols_raw = raw.get("symbols") or []
    if isinstance(symbols_raw, str):
        symbols = tuple(part.strip().upper() for part in symbols_raw.split(",") if part.strip())
    elif isinstance(symbols_raw, Sequence):
        symbols = tuple(str(part).strip().upper() for part in symbols_raw if str(part).strip())
    else:
        symbols = ()
    try:
        weight = float(raw.get("weight", 0.60))
    except (TypeError, ValueError):
        weight = 0.60
    category = str(raw.get("category") or "project").strip().lower()
    if category not in {"exchange", "project", "security", "macro", "media"}:
        category = "project"
    description = str(raw.get("description") or "").strip()
    return ExternalEventSource(
        name=name,
        url=url,
        symbols=symbols,
        weight=max(min(weight, 1.0), 0.2),
        category=category,
        description=description[:180],
    )


def configured_external_event_sources() -> list[ExternalEventSource]:
    raw_sources = settings.external_event_scraper_sources or list(DEFAULT_EXTERNAL_EVENT_SOURCES)
    sources: list[ExternalEventSource] = []
    for raw in raw_sources[: max(int(settings.external_event_scraper_max_sources or 1), 1)]:
        if not isinstance(raw, dict):
            continue
        try:
            sources.append(_normalize_source(raw))
        except ValueError as exc:
            logger.warning("external event scraper source rejected", error=safe_error_text(exc))
    return sources


def configured_external_event_source_diagnostics() -> list[dict[str, Any]]:
    """Return validation status for every configured external event source."""

    raw_sources = settings.external_event_scraper_sources or list(DEFAULT_EXTERNAL_EVENT_SOURCES)
    diagnostics: list[dict[str, Any]] = []
    limit = max(int(settings.external_event_scraper_max_sources or 1), 1)
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            diagnostics.append(
                {
                    "index": index,
                    "enabled": index < limit,
                    "valid": False,
                    "status": "invalid",
                    "error": "采集源必须是包含 URL 的对象。",
                }
            )
            continue
        try:
            source = _normalize_source(raw)
        except ValueError as exc:
            diagnostics.append(
                {
                    "index": index,
                    "name": str(raw.get("name") or "").strip(),
                    "url": str(raw.get("url") or "").strip(),
                    "symbols": raw.get("symbols") or [],
                    "weight": raw.get("weight", 0.60),
                    "enabled": index < limit,
                    "valid": False,
                    "status": "invalid",
                    "error": safe_error_text(exc, limit=180),
                }
            )
            continue
        diagnostics.append(
            {
                **_source_diagnostic_payload(source),
                "index": index,
                "enabled": index < limit,
                "valid": True,
                "status": "active" if index < limit else "over_limit",
                "error": "",
            }
        )
    return diagnostics


def _source_diagnostic_payload(source: ExternalEventSource) -> dict[str, Any]:
    return {
        "name": source.name,
        "url": source.url,
        "symbols": list(source.symbols),
        "weight": source.weight,
        "category": source.category,
        "description": source.description,
    }


class ExternalEventScraper:
    """Fetch administrator-approved event pages through Scrapling when enabled."""

    def __init__(
        self,
        *,
        sources: list[ExternalEventSource] | None = None,
        fetcher: type[AsyncFetcherLike] | None = None,
    ) -> None:
        self._sources = sources
        self._fetcher = fetcher
        self._last_fetch_at: datetime | None = None
        self._last_articles: list[dict[str, Any]] = []
        self._seen_keys: set[str] = set()
        self._symbol_aliases: dict[str, set[str]] = {
            base: {base, *aliases} for base, aliases in _DEFAULT_SYMBOL_ALIASES.items()
        }

    def set_tracked_symbols(self, symbols: list[str]) -> None:
        for symbol in symbols:
            base = str(symbol or "").split("/")[0].split("-")[0].upper()
            if base:
                self._symbol_aliases.setdefault(base, {base}).add(base)

    async def fetch_all(self) -> list[dict[str, Any]]:
        if not settings.external_event_scraper_enabled and self._sources is None:
            return []
        now = datetime.now(UTC)
        interval = max(int(settings.external_event_scraper_interval_seconds or 0), 60)
        if self._last_fetch_at and (now - self._last_fetch_at).total_seconds() < interval:
            return list(self._last_articles)

        sources = (
            self._sources if self._sources is not None else configured_external_event_sources()
        )
        if not sources:
            return []
        fetcher = self._fetcher or _load_async_fetcher()
        if fetcher is None:
            return []

        semaphore = asyncio.Semaphore(min(len(sources), 2))

        async def guarded_fetch(source: ExternalEventSource) -> list[dict[str, Any]]:
            async with semaphore:
                return await self._fetch_source(fetcher, source)

        results = await asyncio.gather(
            *(guarded_fetch(source) for source in sources),
            return_exceptions=True,
        )
        articles: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug("external event source failed", error=safe_error_text(result))
            else:
                articles.extend(result)
        self._last_fetch_at = now
        self._last_articles = articles[: self._max_items_total()]
        logger.info("external events fetched", count=len(self._last_articles))
        return list(self._last_articles)

    async def _fetch_source(
        self,
        fetcher: type[AsyncFetcherLike],
        source: ExternalEventSource,
    ) -> list[dict[str, Any]]:
        timeout = max(float(settings.external_event_scraper_timeout_seconds or 0.0), 1.0)
        try:
            response_or_awaitable = fetcher.get(
                source.url,
                headers={"User-Agent": "CangXiaoQuant/1.0 event-research"},
                impersonate="chrome",
                timeout=timeout,
            )
            response = (
                await asyncio.wait_for(response_or_awaitable, timeout=timeout + 1.0)
                if inspect.isawaitable(response_or_awaitable)
                else response_or_awaitable
            )
        except Exception as exc:
            logger.debug(
                "external event fetch failed",
                source=source.name,
                error=safe_error_text(exc),
            )
            return []

        status = int(getattr(response, "status", getattr(response, "status_code", 0)) or 0)
        if status and not 200 <= status < 300:
            logger.debug("external event fetch non-2xx", source=source.name, status=status)
            return []

        html_text = self._response_text(response)
        if not html_text:
            return []
        return self._extract_articles(source, html_text)

    def _extract_articles(
        self, source: ExternalEventSource, html_text: str
    ) -> list[dict[str, Any]]:
        html_text = html_text[:_MAX_HTML_CHARS]
        meta = self._extract_meta(html_text)
        page_title = self._first_text(
            meta,
            ("og:title", "twitter:title", "title"),
            fallback=self._extract_title(html_text),
        )
        page_summary = self._first_text(
            meta,
            ("og:description", "twitter:description", "description"),
            fallback=self._clean_text(html_text)[:700],
        )
        published_at = self._first_text(
            meta,
            ("article:published_time", "date", "datepublished", "pubdate"),
        )
        articles = self._extract_anchor_articles(source, html_text, page_summary)
        if page_title:
            articles.insert(
                0,
                self._build_article(
                    source=source,
                    title=page_title,
                    summary=page_summary,
                    url=source.url,
                    published_at=published_at,
                ),
            )
        deduped: list[dict[str, Any]] = []
        for article in articles:
            if not article.get("title"):
                continue
            key = self._dedup_key(article)
            if key in self._seen_keys:
                continue
            self._seen_keys.add(key)
            deduped.append(article)
            if len(deduped) >= max(int(settings.external_event_scraper_max_items_per_source), 1):
                break
        return deduped

    def _extract_anchor_articles(
        self,
        source: ExternalEventSource,
        html_text: str,
        page_summary: str,
    ) -> list[dict[str, Any]]:
        articles: list[dict[str, Any]] = []
        source_host = urlparse(source.url).hostname or ""
        for match in _ANCHOR_RE.finditer(html_text):
            if len(articles) >= _ANCHOR_SCAN_LIMIT:
                break
            attrs = self._parse_attrs(match.group("attrs"))
            href = str(attrs.get("href") or "").strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            url = urljoin(source.url, href)
            try:
                url = normalize_external_http_url(
                    url,
                    field_name="external event link URL",
                    allow_empty=False,
                    max_length=500,
                )
            except ValueError:
                continue
            if urlparse(url).hostname != source_host:
                continue
            title = self._clean_text(match.group("text"))
            if not self._looks_like_event_title(title, source):
                continue
            articles.append(
                self._build_article(
                    source=source,
                    title=title,
                    summary=page_summary,
                    url=url,
                    published_at=None,
                )
            )
        return articles

    def _looks_like_event_title(self, title: str, source: ExternalEventSource) -> bool:
        if len(title) < 16 or len(title) > 220:
            return False
        lowered = title.lower()
        if any(term in lowered for term in ("login", "sign up", "privacy", "cookie")):
            return False
        if source.symbols or self._extract_symbols(title):
            return True
        return any(hint in lowered for hint in _EVENT_HINTS)

    def _build_article(
        self,
        *,
        source: ExternalEventSource,
        title: str,
        summary: str,
        url: str,
        published_at: str | None,
    ) -> dict[str, Any]:
        text = f"{title} {summary}"
        symbols = list(source.symbols) or self._extract_symbols(text)
        return {
            "source": f"{SCRAPLING_SOURCE_PREFIX}{source.name}"[:50],
            "title": title[:260],
            "summary": summary[:900],
            "url": url,
            "symbols_mentioned": symbols,
            "published_at": published_at,
            "source_weight": source.weight,
            "source_category": source.category,
            "event_type": "external_event",
            "impact_level": self._impact_level(source, text, symbols),
            "sentiment_score": self._lexicon_sentiment(text),
        }

    def _response_text(self, response: Any) -> str:
        value = getattr(response, "body", None)
        if callable(value):
            value = value()
        if value is None:
            value = getattr(response, "text", "")
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value or "")

    def _extract_meta(self, html_text: str) -> dict[str, str]:
        meta: dict[str, str] = {}
        for match in _META_RE.finditer(html_text):
            attrs = self._parse_attrs(match.group("attrs"))
            key = str(attrs.get("property") or attrs.get("name") or "").strip().lower()
            value = str(attrs.get("content") or "").strip()
            if key and value:
                meta[key] = html.unescape(value)
        return meta

    def _parse_attrs(self, attrs_text: str) -> dict[str, str]:
        attrs: dict[str, str] = {}
        for match in _ATTR_RE.finditer(attrs_text):
            key = match.group("key").lower()
            value = match.group("value").strip("\"'")
            attrs[key] = html.unescape(value)
        return attrs

    def _extract_title(self, html_text: str) -> str:
        match = _TITLE_RE.search(html_text)
        return self._clean_text(match.group("value")) if match else ""

    def _first_text(
        self,
        values: dict[str, str],
        keys: tuple[str, ...],
        *,
        fallback: str = "",
    ) -> str:
        for key in keys:
            value = self._clean_text(values.get(key, ""))
            if value:
                return value
        return self._clean_text(fallback)

    def _clean_text(self, value: str) -> str:
        text = _SCRIPT_STYLE_RE.sub(" ", str(value or ""))
        text = _TAG_RE.sub(" ", text)
        return _WHITESPACE_RE.sub(" ", html.unescape(text)).strip()

    def _extract_symbols(self, text: str) -> list[str]:
        found: list[str] = []
        upper = str(text or "").upper()
        for base, aliases in self._symbol_aliases.items():
            for alias in sorted(aliases, key=len, reverse=True):
                pattern = rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])"
                if re.search(pattern, upper):
                    found.append(base)
                    break
        return found

    def _lexicon_sentiment(self, text: str) -> float:
        lowered = str(text or "").lower()
        positive = ("rally", "surge", "launch", "upgrade", "partnership", "listing", "approve")
        negative = ("hack", "exploit", "delist", "lawsuit", "ban", "suspend", "breach")
        score = sum(1 for word in positive if word in lowered) - sum(
            1 for word in negative if word in lowered
        )
        if score == 0:
            return 0.0
        return max(min(score / 3.0, 1.0), -1.0)

    def _impact_level(
        self,
        source: ExternalEventSource,
        text: str,
        symbols: list[str],
    ) -> int:
        lowered = str(text or "").lower()
        if source.category == "security":
            return 3 if any(term in lowered for term in ("hack", "exploit", "breach")) else 2
        if any(term in lowered for term in ("listing", "delist", "mainnet", "upgrade", "sec")):
            return 3
        return 2 if symbols else 1

    def _dedup_key(self, article: dict[str, Any]) -> str:
        url = str(article.get("url") or "")
        title = str(article.get("title") or "")
        return hashlib.sha256(f"{url}\n{title}".lower().encode("utf-8")).hexdigest()

    def _max_items_total(self) -> int:
        sources = max(int(settings.external_event_scraper_max_sources or 1), 1)
        per_source = max(int(settings.external_event_scraper_max_items_per_source or 1), 1)
        return sources * per_source
