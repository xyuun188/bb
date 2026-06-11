from datetime import UTC, datetime

from services.position_time import PositionTimeParser


def test_position_time_parser_converts_exchange_milliseconds():
    parser = PositionTimeParser()

    parsed = parser.datetime_from_ms(1_780_000_000_000)

    assert parsed.tzinfo == UTC
    assert int(parsed.timestamp() * 1000) == 1_780_000_000_000


def test_position_time_parser_falls_back_to_now_for_bad_exchange_timestamp():
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    parser = PositionTimeParser(now_provider=lambda: now)

    assert parser.datetime_from_ms("bad") == now


def test_position_time_parser_accepts_datetime_iso_seconds_and_milliseconds():
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    parser = PositionTimeParser(now_provider=lambda: now)

    assert parser.position_age_minutes(datetime(2026, 6, 8, 11, 30, tzinfo=UTC)) == 30.0
    assert parser.position_age_minutes("2026-06-08T11:45:00Z") == 15.0
    assert (
        parser.position_age_minutes(int(datetime(2026, 6, 8, 11, 50, tzinfo=UTC).timestamp()))
        == 10.0
    )
    assert (
        parser.position_age_minutes(
            int(datetime(2026, 6, 8, 11, 55, tzinfo=UTC).timestamp() * 1000)
        )
        == 5.0
    )


def test_position_time_parser_treats_future_naive_datetime_as_beijing_time():
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    parser = PositionTimeParser(now_provider=lambda: now)

    age = parser.position_age_minutes(datetime(2026, 6, 8, 19, 0))

    assert age == 60.0


def test_position_time_parser_returns_none_for_unusable_position_time():
    parser = PositionTimeParser()

    assert parser.position_age_minutes(None) is None
    assert parser.position_age_minutes("not-a-time") is None
    assert parser.position_age_minutes(object()) is None
