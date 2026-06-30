from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import runtime_env_bootstrap
from scripts.runtime_env_bootstrap import (
    RUNTIME_USER_DROPPED_ENV,
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)


def test_runtime_env_bootstrap_loads_project_then_runtime_env(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "app"
    project_root.mkdir()
    runtime_env = tmp_path / "bb-runtime.env"
    (project_root / ".env").write_text(
        "DATABASE_URL=postgresql+asyncpg://project\n"
        "BB_SECURE_SETTINGS_KEY=project-key\n"
        "PROJECT_ONLY=yes\n",
        encoding="utf-8",
    )
    runtime_env.write_text(
        "DATABASE_URL=postgresql+asyncpg://runtime\n"
        "BB_SECURE_SETTINGS_KEY='runtime-key'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("BB_SECURE_SETTINGS_KEY", raising=False)
    monkeypatch.delenv("PROJECT_ONLY", raising=False)

    loaded = load_runtime_env_files(project_root=project_root, runtime_env_path=runtime_env)

    assert loaded["DATABASE_URL"] == "postgresql+asyncpg://runtime"
    assert loaded["BB_SECURE_SETTINGS_KEY"] == "runtime-key"
    assert loaded["PROJECT_ONLY"] == "yes"


def test_runtime_env_bootstrap_ignores_unreadable_runtime_env(monkeypatch, tmp_path) -> None:
    project_root = tmp_path / "app"
    project_root.mkdir()
    runtime_env = tmp_path / "bb-runtime.env"
    (project_root / ".env").write_text("PROJECT_ONLY=yes\n", encoding="utf-8")
    runtime_env.write_text("DATABASE_URL=postgresql+asyncpg://runtime\n", encoding="utf-8")

    original_read_text = Path.read_text

    def fake_read_text(path: Path, *args, **kwargs) -> str:
        if path == runtime_env:
            raise PermissionError("permission denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.delenv("PROJECT_ONLY", raising=False)

    loaded = load_runtime_env_files(project_root=project_root, runtime_env_path=runtime_env)

    assert loaded == {"PROJECT_ONLY": "yes"}


def test_runtime_env_bootstrap_drops_root_online_script_to_runtime_user(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[tuple[object, str, object]] = []
    account = SimpleNamespace(pw_gid=1001, pw_uid=1001, pw_dir="/home/bb")

    monkeypatch.delenv(RUNTIME_USER_DROPPED_ENV, raising=False)
    monkeypatch.setattr(runtime_env_bootstrap, "_is_posix", lambda: True)
    monkeypatch.setattr(runtime_env_bootstrap, "_effective_uid", lambda: 0)
    monkeypatch.setattr(runtime_env_bootstrap, "_lookup_user", lambda user: account)
    monkeypatch.setattr(
        runtime_env_bootstrap,
        "_drop_to_user",
        lambda acct, *, user, project_root: calls.append((acct, user, project_root)),
    )

    dropped = drop_privileges_to_runtime_user_if_needed(project_root=Path("/data/bb/app"))

    assert dropped is True
    assert calls == [(account, "bb", Path("/data/bb/app"))]
    assert runtime_env_bootstrap.os.environ[RUNTIME_USER_DROPPED_ENV] == "bb"


def test_runtime_env_bootstrap_does_not_drop_local_root_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(RUNTIME_USER_DROPPED_ENV, raising=False)
    monkeypatch.setattr(runtime_env_bootstrap, "_is_posix", lambda: True)
    monkeypatch.setattr(runtime_env_bootstrap, "_effective_uid", lambda: 0)
    monkeypatch.setattr(
        runtime_env_bootstrap,
        "_drop_to_user",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not drop")),
    )

    dropped = drop_privileges_to_runtime_user_if_needed(project_root=tmp_path)

    assert dropped is False
