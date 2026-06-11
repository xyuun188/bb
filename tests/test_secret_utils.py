from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from config.settings import Settings
from core.secret_utils import (
    is_masked_secret,
    mask_secret,
    redact_mapping,
    redact_text,
    secret_fingerprint,
    secret_state,
)


def test_redact_text_masks_common_secret_assignments() -> None:
    text = (
        "api_key=live-secret-value password: p@ssw0rd "
        '"token": "abc.def.ghi" webhook=https://example.invalid/hook/123'
    )

    redacted = redact_text(text)

    assert "live-secret-value" not in redacted
    assert "p@ssw0rd" not in redacted
    assert "abc.def.ghi" not in redacted
    assert "example.invalid/hook/123" not in redacted
    assert "api_key=***" in redacted
    assert "password: ***" in redacted
    assert '"token": "***"' in redacted
    assert "webhook=***" in redacted


def test_redact_text_keeps_bearer_prefix_but_hides_token() -> None:
    fake_token = "abcdefghijklmnopqrstuvwxyz" + "123456"
    redacted = redact_text(f"Authorization: Bearer {fake_token}")

    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert redacted == "Authorization: ***"


def test_redact_text_masks_url_embedded_credentials() -> None:
    text = (
        "failed to call http://user:password@127.0.0.1:8001/v1 "
        "and postgresql://admin:secret@db.internal/trading"
    )

    redacted = redact_text(text)

    assert "user:password" not in redacted
    assert "admin:secret" not in redacted
    assert "http://***@127.0.0.1:8001/v1" in redacted
    assert "postgresql://***@db.internal/trading" in redacted


def test_redact_mapping_recurses_into_nested_payloads() -> None:
    payload = {
        "api_key": "real-secret-value",
        "nested": {"password": "nested-secret"},
        "events": [
            {"message": "token=event-secret-value"},
            "webhook=https://example.invalid/private",
        ],
        "safe": "BTC/USDT",
    }

    redacted = redact_mapping(payload)

    assert redacted["api_key"] == "***"
    assert redacted["nested"]["password"] == "***"
    assert redacted["events"][0]["message"] == "token=***"
    assert redacted["events"][1] == "webhook=***"
    assert redacted["safe"] == "BTC/USDT"


def test_redact_mapping_recurses_into_tuple_and_set_payloads() -> None:
    payload = {
        "tuple_context": (
            "password=tuple-secret-value",
            {"token": "tuple-token-value"},
        ),
        "set_context": {"api_key=set-secret-value"},
        "frozen_context": frozenset({"webhook=https://example.invalid/private"}),
    }

    redacted = redact_mapping(payload)
    rendered = str(redacted)

    assert "tuple-secret-value" not in rendered
    assert "tuple-token-value" not in rendered
    assert "set-secret-value" not in rendered
    assert "example.invalid/private" not in rendered
    assert redacted["tuple_context"][0] == "password=***"
    assert redacted["tuple_context"][1]["token"] == "***"
    assert redacted["set_context"] == {"api_key=***"}
    assert redacted["frozen_context"] == frozenset({"webhook=***"})


def test_secret_state_and_mask_detection_do_not_expose_secret_material() -> None:
    assert mask_secret("abcdef", show_last=2) == "***ef"
    assert is_masked_secret("***ef") is True
    assert secret_state("abcdef") == "configured"
    assert secret_state("") == "missing"


def test_secret_fingerprint_is_stable_and_non_revealing() -> None:
    first = secret_fingerprint("sk-sensitive-value-123456", length=16)
    second = secret_fingerprint("sk-sensitive-value-123456", length=16)

    assert first == second
    assert len(first) == 16
    assert "sensitive" not in first
    assert secret_fingerprint("") == ""


def test_settings_defaults_do_not_hardcode_remote_model_endpoints() -> None:
    cfg = Settings(_env_file=None)  # type: ignore[call-arg]

    assert cfg.ai_api_base == ""
    assert cfg.ai_api_key == ""
    assert cfg.local_ai_tools_enabled is False
    assert cfg.local_ai_tools_api_base == ""
    assert cfg.high_risk_review_api_base == ""
    assert cfg.high_risk_review_api_key == ""


def test_dashboard_cors_defaults_are_local_and_explicit() -> None:
    cfg = Settings(_env_file=None)  # type: ignore[call-arg]

    origins = cfg.dashboard_allowed_origins()

    assert "*" not in origins
    assert origins == ["http://127.0.0.1:8002", "http://localhost:8002"]


def test_dashboard_cors_origins_parse_from_env_strings() -> None:
    comma_cfg = Settings(  # type: ignore[call-arg]
        _env_file=None,
        dashboard_cors_origins=cast(Any, "https://dash.example.invalid, http://localhost:8002"),
    )
    json_cfg = Settings(  # type: ignore[call-arg]
        _env_file=None,
        dashboard_cors_origins=cast(Any, '["https://dash.example.invalid"]'),
    )

    assert comma_cfg.dashboard_allowed_origins() == [
        "https://dash.example.invalid",
        "http://localhost:8002",
    ]
    assert json_cfg.dashboard_allowed_origins() == ["https://dash.example.invalid"]


def test_update_env_file_ignores_masked_secret_and_rejects_invalid_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AI_API_KEY=real-value\nDASHBOARD_HOST=127.0.0.1\n",
        encoding="utf-8",
    )

    class TmpSettings(Settings):
        @property
        def project_root(self) -> Path:
            return tmp_path

    settings = TmpSettings(_env_file=None)  # type: ignore[call-arg]

    settings.update_env_file({"AI_API_KEY": "***masked", "DASHBOARD_HOST": "127.0.0.2"})

    text = env_path.read_text(encoding="utf-8")
    assert "AI_API_KEY=real-value" in text
    assert "DASHBOARD_HOST=127.0.0.2" in text

    with pytest.raises(ValueError, match="Invalid .env key"):
        settings.update_env_file({"bad-key": "value"})
    with pytest.raises(ValueError, match="Invalid newline"):
        settings.update_env_file({"DASHBOARD_HOST": "127.0.0.1\nMALICIOUS=1"})


def test_update_env_file_creates_missing_env_file_without_secret_placeholders(
    tmp_path: Path,
) -> None:
    class TmpSettings(Settings):
        @property
        def project_root(self) -> Path:
            return tmp_path

    settings = TmpSettings(_env_file=None)  # type: ignore[call-arg]
    env_path = tmp_path / ".env"

    settings.update_env_file(
        {
            "AI_API_KEY": "***masked",
            "DASHBOARD_HOST": "127.0.0.2",
            "LOCAL_AI_TOOLS_ENABLED": "true",
        }
    )

    text = env_path.read_text(encoding="utf-8")
    assert "AI_API_KEY" not in text
    assert "DASHBOARD_HOST=127.0.0.2" in text
    assert "LOCAL_AI_TOOLS_ENABLED=true" in text


def test_update_env_file_does_not_create_file_for_only_masked_secret_updates(
    tmp_path: Path,
) -> None:
    class TmpSettings(Settings):
        @property
        def project_root(self) -> Path:
            return tmp_path

    settings = TmpSettings(_env_file=None)  # type: ignore[call-arg]

    settings.update_env_file({"AI_API_KEY": "***masked"})

    assert not (tmp_path / ".env").exists()


def test_update_env_file_quotes_complex_values_for_dotenv_round_trip(
    tmp_path: Path,
) -> None:
    class TmpSettings(Settings):
        @property
        def project_root(self) -> Path:
            return tmp_path

    env_path = tmp_path / ".env"
    settings = TmpSettings(_env_file=None)  # type: ignore[call-arg]

    settings.update_env_file(
        {
            "AI_MODEL": 'qwen3 32b "trade"#safe',
            "DASHBOARD_CORS_ORIGINS": '["https://dash.example.invalid/path#frag"]',
            "DASHBOARD_HOST": "127.0.0.1",
        }
    )

    text = env_path.read_text(encoding="utf-8")
    assert 'AI_MODEL="qwen3 32b \\"trade\\"#safe"' in text
    assert "DASHBOARD_CORS_ORIGINS=" in text
    assert "DASHBOARD_HOST=127.0.0.1" in text

    loaded = TmpSettings(_env_file=env_path)  # type: ignore[call-arg]
    assert loaded.ai_model == 'qwen3 32b "trade"#safe'
    assert loaded.dashboard_allowed_origins() == ["https://dash.example.invalid/path#frag"]
