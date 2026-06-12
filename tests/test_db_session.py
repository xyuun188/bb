import pytest

import db.session as session_module


class _FakeSession:
    def __init__(self) -> None:
        self.rollback_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def rollback(self) -> None:
        self.rollback_count += 1


class _FakeMaker:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    def __call__(self) -> _FakeSession:
        return self.session


@pytest.mark.asyncio
async def test_read_session_ctx_does_not_rollback_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession()

    async def fake_get_sessionmaker():
        return _FakeMaker(fake_session)

    monkeypatch.setattr(session_module, "get_sessionmaker", fake_get_sessionmaker)

    async with session_module.get_read_session_ctx() as session:
        assert session is fake_session

    assert fake_session.rollback_count == 0


@pytest.mark.asyncio
async def test_read_session_ctx_rolls_back_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_session = _FakeSession()

    async def fake_get_sessionmaker():
        return _FakeMaker(fake_session)

    monkeypatch.setattr(session_module, "get_sessionmaker", fake_get_sessionmaker)

    with pytest.raises(RuntimeError):
        async with session_module.get_read_session_ctx():
            raise RuntimeError("boom")

    assert fake_session.rollback_count == 1
