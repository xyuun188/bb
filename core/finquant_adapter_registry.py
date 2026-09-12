"""Load the deployable FinQuant adapter registry source."""

from pathlib import Path

_REMOTE_REGISTRY_PATH = Path(__file__).with_name("finquant_adapter_registry_remote.py")
REMOTE_REGISTRY_TOOL_CODE = _REMOTE_REGISTRY_PATH.read_text(encoding="utf-8")
