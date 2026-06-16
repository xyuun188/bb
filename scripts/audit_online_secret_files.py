#!/usr/bin/env python3
"""Audit and optionally delete sensitive server-info text files from the online app dir."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"
SENSITIVE_NAME_MARKERS = (
    "服务器信息",
    "服务器资料",
    "大模型服务器",
    "平台服务器",
    "server_info",
    "model_server_info",
)


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _remote_script(*, remote_app_dir: str, delete: bool) -> str:
    markers_json = json.dumps(SENSITIVE_NAME_MARKERS, ensure_ascii=False)
    return textwrap.dedent(
        f"""
        import json
        import os
        from pathlib import Path

        root = Path({remote_app_dir!r}).resolve()
        markers = tuple(json.loads({markers_json!r}))
        delete = {str(bool(delete))}
        if not str(root).startswith('/data/bb/app'):
            raise SystemExit('refusing to scan outside /data/bb/app')
        matches = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_parts = Path(dirpath).relative_to(root).parts
            if any(part in {{'.git', '.venv', 'venv', '__pycache__'}} for part in rel_parts):
                dirnames[:] = []
                continue
            for name in filenames:
                lower = name.lower()
                if not lower.endswith('.txt'):
                    continue
                if any(marker.lower() in lower or marker in name for marker in markers):
                    path = Path(dirpath) / name
                    matches.append(str(path))
        for path in matches:
            print(('delete ' if delete else 'found ') + path)
            if delete:
                Path(path).unlink(missing_ok=True)
        print(json.dumps({{'count': len(matches), 'deleted': bool(delete)}}, ensure_ascii=False))
        """
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default=REMOTE_APP_DIR)
    parser.add_argument("--delete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = _remote_script(remote_app_dir=args.remote_app_dir, delete=args.delete)
    command = f"python3 - <<'PY'\n{script}\nPY"
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        output = run_remote_text(ssh, command, timeout=60, check=True)
        safe_print(output)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
