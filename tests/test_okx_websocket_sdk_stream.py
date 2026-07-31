from __future__ import annotations

import asyncio

import pytest

from data_feed import okx_ws_client
from data_feed.okx_ws_client import OKXWebSocketClient
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
