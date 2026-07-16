from __future__ import annotations

import pytest

from scripts import (
    configure_32b_review_service,
    continue_deploy_deepseek14_services,
    deploy_deepseek_32b_main_service,
    fix_local_ai_tools_service_path,
    redeploy_server_14b_kronos_architecture,
    start_deepseek_32b_main_service,
    test_start_32b_review_service,
)


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
    assert "must not replace the verified Phase 3 service identities" in capsys.readouterr().err


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


def test_legacy_local_ai_tools_service_fix_is_retired() -> None:
    with pytest.raises(RuntimeError, match="local-ai-tools.service") as exc_info:
        fix_local_ai_tools_service_path.main()

    assert "bb-phase3-quant-api.service" in str(exc_info.value)
