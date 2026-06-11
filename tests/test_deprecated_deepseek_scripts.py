from __future__ import annotations

from pathlib import Path

import pytest

from scripts import (
    configure_32b_review_service,
    continue_deploy_deepseek14_services,
    deploy_deepseek_32b_main_service,
    redeploy_server_14b_kronos_architecture,
    start_deepseek_32b_main_service,
    test_start_32b_review_service,
)

ROOT = Path(__file__).resolve().parents[1]


def test_deprecated_deepseek_deploy_entrypoint_fails_closed(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        deploy_deepseek_32b_main_service.main()

    assert exc_info.value.code == 2
    assert "no longer an allowed main LLM service" in capsys.readouterr().err


def test_deprecated_deepseek_start_entrypoint_fails_closed(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        start_deepseek_32b_main_service.main()

    assert exc_info.value.code == 2
    assert "must not be started as the main LLM service" in capsys.readouterr().err


def test_deprecated_deepseek14_redeploy_entrypoint_fails_closed(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        redeploy_server_14b_kronos_architecture.main()

    assert exc_info.value.code == 2
    assert "must not replace the approved Qwen3-32B-AWQ" in capsys.readouterr().err


def test_deprecated_deepseek14_continue_entrypoint_fails_closed(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        continue_deploy_deepseek14_services.main()

    assert exc_info.value.code == 2
    assert "continuation deploy path is deprecated" in capsys.readouterr().err


def test_deprecated_local_32b_review_config_fails_closed(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        configure_32b_review_service.main()

    assert exc_info.value.code == 2
    assert "High-risk review must use the online model" in capsys.readouterr().err


def test_deprecated_local_32b_review_start_fails_closed(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        test_start_32b_review_service.main()

    assert exc_info.value.code == 2
    assert "must use the online model configured" in capsys.readouterr().err


def test_local_ai_tools_service_fix_depends_on_current_qwen3_main_service() -> None:
    source = (ROOT / "scripts" / "fix_local_ai_tools_service_path.py").read_text(encoding="utf-8")

    assert "qwen3-32b-main.service" in source
    assert "qwen3-14b.service" not in source
