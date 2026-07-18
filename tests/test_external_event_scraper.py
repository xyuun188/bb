from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from config.settings import Settings, settings
from data_feed.external_event_scraper import (
    EXTERNAL_EVENT_MAX_SOURCES_LIMIT,
    RECOMMENDED_EXTERNAL_EVENT_SOURCES,
    ExternalEventScraper,
    ExternalEventSource,
    _normalize_source,
    configured_external_event_source_diagnostics,
)
from db.session import close_db, get_session_ctx, init_db
from models.news import NewsArticle
from services import external_event_service as external_event_service_module
from services.external_event_service import (
    ExternalEventService,
    load_external_event_settings_from_env,
)


class FakeResponse:
    status = 200

    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")


class FakeFetcher:
    html = """
    <html>
      <head>
        <title>Ethereum Foundation Blog</title>
        <meta property="og:description" content="ETH ecosystem upgrades and launch news">
      </head>
      <body>
        <a href="/2026/06/upgrade">Ethereum Dencun upgrade launch details</a>
        <a href="/privacy">Privacy Policy</a>
      </body>
    </html>
    """

    @classmethod
    def get(cls, _url: str, **_kwargs: Any) -> FakeResponse:
        return FakeResponse(cls.html)


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    db_path = tmp_path / "external-events.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()


@pytest.mark.asyncio
async def test_external_event_scraper_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)

    scraper = ExternalEventScraper(fetcher=FakeFetcher)

    assert await scraper.fetch_all() == []


def test_external_event_source_rejects_non_public_or_non_https_urls() -> None:
    with pytest.raises(ValueError):
        _normalize_source({"name": "bad", "url": "http://example.com/news"})

    with pytest.raises(ValueError):
        _normalize_source({"name": "bad", "url": "https://127.0.0.1/news"})


def test_recommended_external_event_sources_fit_runtime_limit() -> None:
    names = {str(source.get("name") or "") for source in RECOMMENDED_EXTERNAL_EVENT_SOURCES}

    assert len(RECOMMENDED_EXTERNAL_EVENT_SOURCES) <= EXTERNAL_EVENT_MAX_SOURCES_LIMIT
    assert {
        "stellar_blog",
        "filecoin_blog",
        "aave_blog",
        "circle_blog",
        "tether_news",
        "sec_press_releases",
        "cftc_press_releases",
    }.issubset(names)


def test_external_event_runtime_settings_reload_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "EXTERNAL_EVENT_SCRAPER_ENABLED=true",
                "EXTERNAL_EVENT_SCRAPER_INTERVAL_SECONDS=600",
                "EXTERNAL_EVENT_SCRAPER_TIMEOUT_SECONDS=5",
                "EXTERNAL_EVENT_SCRAPER_MAX_SOURCES=2",
                "EXTERNAL_EVENT_SCRAPER_MAX_ITEMS_PER_SOURCE=4",
                'EXTERNAL_EVENT_SCRAPER_SOURCES=[{"name":"ethereum_blog","url":"https://blog.ethereum.org/","symbols":["ETH"],"weight":0.72}]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        external_event_service_module,
        "_project_env_path",
        lambda: env_path,
    )
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    loaded = load_external_event_settings_from_env()

    assert loaded == {"enabled": True, "source_count": 1, "interval_seconds": 600}
    assert settings.external_event_scraper_enabled is True
    assert settings.external_event_scraper_sources[0]["name"] == "ethereum_blog"


def test_external_event_sources_recover_legacy_json_wrapped_in_url() -> None:
    legacy_value = (
        '[{\\"name\\":\\"binance_announcements\\",'
        '\\"url\\":\\"https://www.binance.com/en/support/announcement/c-48\\",'
        '\\"weight\\":0.88},{\\"name\\":\\"ethereum_blog\\",'
        '\\"url\\":\\"https://blog.ethereum.org/\\",'
        '\\"symbols\\":[\\"ETH\\"],\\"weight\\":0.72}]'
    )

    cfg = Settings(_env_file=None, external_event_scraper_sources={"url": legacy_value})  # type: ignore[call-arg]

    assert len(cfg.external_event_scraper_sources) == 2
    assert cfg.external_event_scraper_sources[0]["name"] == "binance_announcements"
    assert cfg.external_event_scraper_sources[0]["url"].startswith("https://www.binance.com/")
    assert cfg.external_event_scraper_sources[1]["symbols"] == ["ETH"]


def test_external_event_runtime_reload_recovers_legacy_json_wrapped_in_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "EXTERNAL_EVENT_SCRAPER_ENABLED=true",
                "EXTERNAL_EVENT_SCRAPER_MAX_SOURCES=4",
                (
                    'EXTERNAL_EVENT_SCRAPER_SOURCES={"url":"'
                    '[{\\"name\\":\\"binance_announcements\\",'
                    '\\"url\\":\\"https://www.binance.com/en/support/announcement/c-48\\"},'
                    '{\\"name\\":\\"ethereum_blog\\",'
                    '\\"url\\":\\"https://blog.ethereum.org/\\",'
                    '\\"symbols\\":[\\"ETH\\"]}]'
                    '"}'
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        external_event_service_module,
        "_project_env_path",
        lambda: env_path,
    )
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    loaded = load_external_event_settings_from_env()

    assert loaded["enabled"] is True
    assert loaded["source_count"] == 2
    assert settings.external_event_scraper_sources[0]["name"] == "binance_announcements"
    assert settings.external_event_scraper_sources[1]["url"] == "https://blog.ethereum.org/"


def test_external_event_source_diagnostics_exposes_invalid_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "external_event_scraper_sources",
        [
            {
                "name": "ethereum_blog",
                "url": "https://blog.ethereum.org/",
                "symbols": ["ETH"],
                "weight": 0.72,
            },
            {"name": "broken", "url": "https://example.com/" + ("x" * 520)},
        ],
    )
    monkeypatch.setattr(settings, "external_event_scraper_max_sources", 4)

    diagnostics = configured_external_event_source_diagnostics()

    assert diagnostics[0]["valid"] is True
    assert diagnostics[0]["status"] == "active"
    assert diagnostics[1]["valid"] is False
    assert diagnostics[1]["status"] == "invalid"
    assert "too long" in diagnostics[1]["error"]


@pytest.mark.asyncio
async def test_external_event_scraper_parses_meta_and_event_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "external_event_scraper_max_items_per_source", 4)
    source = ExternalEventSource(
        name="ethereum_blog",
        url="https://blog.ethereum.org/",
        symbols=("ETH",),
        weight=0.72,
    )
    scraper = ExternalEventScraper(sources=[source], fetcher=FakeFetcher)

    articles = await scraper.fetch_all()

    assert len(articles) == 2
    assert articles[0]["source"] == "scrapling:ethereum_blog"
    assert articles[0]["title"] == "Ethereum Foundation Blog"
    assert articles[0]["symbols_mentioned"] == ["ETH"]
    assert articles[1]["url"] == "https://blog.ethereum.org/2026/06/upgrade"
    assert "upgrade launch" in articles[1]["title"]
    report = scraper.last_source_reports[0]
    assert report["name"] == "ethereum_blog"
    assert report["status"] == "ok"
    assert report["http_status"] == 200
    assert report["response_chars"] == len(FakeFetcher.html)
    assert report["parsed_article_count"] == 2
    assert report["emitted_article_count"] == 2
    assert report["error"] is None


@pytest.mark.asyncio
async def test_external_event_service_persists_articles_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "external_event_scraper_max_items_per_source", 4)
    source = ExternalEventSource(
        name="ethereum_blog",
        url="https://blog.ethereum.org/",
        symbols=("ETH",),
        weight=0.72,
    )
    health_path = tmp_path / "external-event-source-health.json"
    service = ExternalEventService(
        scraper=ExternalEventScraper(sources=[source], fetcher=FakeFetcher),
        health_path=health_path,
    )

    try:
        first = await service.collect_once()
        second = await service.collect_once()
        async with get_session_ctx() as session:
            rows = list((await session.execute(select(NewsArticle))).scalars().all())
    finally:
        await close_db()

    assert first == {"fetched": 2, "stored": 2, "skipped": 0}
    assert second == {"fetched": 2, "stored": 0, "skipped": 2}
    assert len(rows) == 2
    assert rows[0].source == "scrapling:ethereum_blog"
    health = external_event_service_module.load_external_event_source_health(health_path)
    assert health["healthy_source_count"] == 1
    assert health["configured_source_count"] == 1
    assert health["sources"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_external_event_health_snapshot_uses_collector_that_finished_the_round(
    tmp_path: Path,
) -> None:
    health_path = tmp_path / "external-event-source-health.json"

    class ReplacedDuringFetchScraper:
        last_source_reports = [
            {
                "name": "ethereum_blog",
                "status": "ok",
                "http_status": 200,
            }
        ]

        async def fetch_all(self) -> list[dict[str, Any]]:
            service.scraper = ExternalEventScraper()
            return []

    service = ExternalEventService(
        scraper=ReplacedDuringFetchScraper(),  # type: ignore[arg-type]
        health_path=health_path,
    )

    result = await service.collect_once()

    assert result == {"fetched": 0, "stored": 0, "skipped": 0}
    health = external_event_service_module.load_external_event_source_health(health_path)
    assert health["configured_source_count"] == 1
    assert health["healthy_source_count"] == 1
