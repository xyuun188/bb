from __future__ import annotations

import asyncio
import json

import pytest

from core.exceptions import WebSocketConnectionError
from data_feed import okx_ws_client
from data_feed.okx_ws_client import OKXWebSocketClient
from services import okx_perpetual_sdk
from services.okx_perpetual_sdk import OkxPublicWebSocketSdkStream


@pytest.mark.asyncio
async def test_sdk_stream_recv_propagates_consumer_failure() -> None:
    stream = OkxPublicWebSocketSdkStream()

    async def fail_consumer() -> None:
        await asyncio.sleep(0)
        raise ConnectionError("consumer disconnected")

    consume_task = asyncio.create_task(fail_consumer())
    stream._consume_task = consume_task

    with pytest.raises(ConnectionError, match="consumer disconnected"):
        await stream.recv()

    assert consume_task.done()
    assert isinstance(consume_task.exception(), ConnectionError)


@pytest.mark.asyncio
async def test_sdk_stream_cancelled_recv_does_not_steal_next_message() -> None:
    stream = OkxPublicWebSocketSdkStream()
    consumer_released = asyncio.Event()
    consume_task = asyncio.create_task(consumer_released.wait())
    stream._consume_task = consume_task

    first_recv = asyncio.create_task(stream.recv())
    await asyncio.sleep(0)
    first_recv.cancel()
    await asyncio.gather(first_recv, return_exceptions=True)

    stream._on_message("next-message")
    assert await asyncio.wait_for(stream.recv(), timeout=0.2) == "next-message"

    await stream.close()
    assert consume_task.done()


@pytest.mark.asyncio
async def test_sdk_stream_coalesces_ticker_backlog_to_latest_payload() -> None:
    stream = OkxPublicWebSocketSdkStream()
    consumer_released = asyncio.Event()
    consume_task = asyncio.create_task(consumer_released.wait())
    stream._consume_task = consume_task

    def ticker(last: str, timestamp: str) -> str:
        return json.dumps(
            {
                "arg": {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
                "data": [{"last": last, "ts": timestamp}],
            }
        )

    stream._on_message(ticker("64000", "1000"))
    stream._on_message(ticker("64001", "1100"))
    stream._on_message(ticker("64002", "1200"))

    received = json.loads(await asyncio.wait_for(stream.recv(), timeout=0.2))

    assert received["data"][0]["last"] == "64002"
    assert stream._ticker_ready.empty()
    assert stream._ticker_messages == {}
    assert stream._coalesced_ticker_messages == 2

    await stream.close()
    assert consume_task.done()


@pytest.mark.asyncio
async def test_sdk_stream_rate_limits_each_ticker_instrument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = OkxPublicWebSocketSdkStream()
    consumer_released = asyncio.Event()
    consume_task = asyncio.create_task(consumer_released.wait())
    stream._consume_task = consume_task

    def ticker(last: str, timestamp: str) -> str:
        return json.dumps(
            {
                "arg": {"channel": "tickers", "instId": "ETH-USDT-SWAP"},
                "data": [{"last": last, "ts": timestamp}],
            }
        )

    monkeypatch.setattr(okx_perpetual_sdk, "OKX_WS_TICKER_EMIT_INTERVAL_SECONDS", 0.1)
    stream._on_message(ticker("1800", "1000"))
    first = json.loads(await asyncio.wait_for(stream.recv(), timeout=0.2))
    stream._on_message(ticker("1801", "1100"))
    stream._on_message(ticker("1802", "1200"))

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(stream.recv(), timeout=0.05)

    await asyncio.sleep(0.1)
    latest = json.loads(await asyncio.wait_for(stream.recv(), timeout=0.2))

    assert first["data"][0]["last"] == "1800"
    assert latest["data"][0]["last"] == "1802"

    await stream.close()
    assert consume_task.done()


@pytest.mark.asyncio
async def test_ws_listener_closes_failed_stream_before_reconnect_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedStream:
        def __init__(self) -> None:
            self.closed = False

        async def recv(self) -> str:
            raise ConnectionError("stream failed")

        async def close(self) -> None:
            self.closed = True

    stream = FailedStream()
    client = OKXWebSocketClient()
    client._running = True
    client._ws = stream
    reconnect_calls = 0

    async def stop_during_backoff(_seconds: float) -> None:
        client._running = False

    async def unexpected_connect() -> None:
        nonlocal reconnect_calls
        reconnect_calls += 1

    monkeypatch.setattr(okx_ws_client.asyncio, "sleep", stop_during_backoff)
    monkeypatch.setattr(client, "connect", unexpected_connect)

    await client.listen()

    assert stream.closed is True
    assert client._ws is None
    assert reconnect_calls == 0


@pytest.mark.asyncio
async def test_ws_listener_reconnects_when_ping_succeeds_but_tickers_are_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PongOnlyStream:
        def __init__(self) -> None:
            self.closed = False

        async def recv(self) -> str:
            return "pong"

        async def close(self) -> None:
            self.closed = True

    stream = PongOnlyStream()
    client = OKXWebSocketClient()
    client._running = True
    client._ws = stream
    client._connected_at = (
        okx_ws_client.time.time() - okx_ws_client.WS_TICKER_STALE_RECONNECT_SECONDS - 1
    )

    async def stop_during_backoff(_seconds: float) -> None:
        client._running = False

    monkeypatch.setattr(okx_ws_client.asyncio, "sleep", stop_during_backoff)

    await client.listen()

    assert stream.closed is True
    assert client._ws is None


@pytest.mark.asyncio
async def test_ws_listener_reconnects_after_repeated_receive_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutStream:
        def __init__(self) -> None:
            self.closed = False
            self.sent: list[str] = []

        async def recv(self) -> str:
            raise TimeoutError

        async def send(self, payload: str) -> None:
            self.sent.append(payload)

        async def close(self) -> None:
            self.closed = True

    stream = TimeoutStream()
    client = OKXWebSocketClient()
    client._running = True
    client._ws = stream

    async def stop_during_backoff(_seconds: float) -> None:
        client._running = False

    monkeypatch.setattr(okx_ws_client.asyncio, "sleep", stop_during_backoff)

    await client.listen()

    assert stream.sent == ["ping"]
    assert stream.closed is True
    assert client._ws is None


@pytest.mark.asyncio
async def test_ws_listener_keeps_retrying_after_reconnect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedStream:
        async def recv(self) -> str:
            raise ConnectionError("stream failed")

        async def close(self) -> None:
            return None

    client = OKXWebSocketClient()
    client._running = True
    client._ws = FailedStream()
    connect_attempts = 0

    class RecoveredStream:
        async def recv(self) -> str:
            client._running = False
            return "{}"

        async def close(self) -> None:
            return None

    async def reconnect() -> None:
        nonlocal connect_attempts
        connect_attempts += 1
        if connect_attempts == 1:
            raise WebSocketConnectionError("temporary reconnect failure")
        client._ws = RecoveredStream()

    async def immediate_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client, "connect", reconnect)
    monkeypatch.setattr(okx_ws_client.asyncio, "sleep", immediate_sleep)

    await client.listen()

    assert connect_attempts == 2
    assert client._reconnect_count == 2
    assert client._running is False
