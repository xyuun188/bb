from __future__ import annotations

import pytest

from scripts import deploy_qwen3_32b_main_service as deploy_32b
from scripts import download_phase3_decision_model as download_32b
from scripts import start_qwen3_32b_main_service as start_32b


@pytest.mark.parametrize(
    ("entrypoint", "message"),
    (
        (lambda: download_32b.main([]), download_32b.RETIRED_MESSAGE),
        (deploy_32b.main, deploy_32b.RETIRED_MESSAGE),
        (start_32b.main, start_32b.RETIRED_MESSAGE),
    ),
)
def test_obsolete_qwen3_32b_entrypoints_refuse_execution(entrypoint, message: str) -> None:
    with pytest.raises(RuntimeError, match="Qwen3-32B") as exc_info:
        entrypoint()

    assert str(exc_info.value) == message
    assert "migrate_phase3_model_service_identity.py" in message
