from pathlib import Path

from scripts.security_secret_scan import ROOT, scan_file, should_skip


def _rules_for(path: Path) -> set[str]:
    return {finding.rule for finding in scan_file(path)}


def test_security_scan_skips_local_secret_containers_by_name() -> None:
    assert should_skip(ROOT / ".env")
    assert should_skip(ROOT / "服务器资料.md")
    assert should_skip(ROOT / "config" / "模拟交易秘钥.yaml")
    assert should_skip(ROOT / "config" / "交易密钥.json")
    assert should_skip(ROOT / "config" / "服务器密码.toml")


def test_security_scan_flags_hardcoded_public_ip_urls(tmp_path: Path) -> None:
    source = tmp_path / "config.py"
    public_url = "http://" + ".".join(["175", "155", "64", "171"]) + ":31840/v1"
    source.write_text(
        f'AI_API_BASE = "{public_url}"\n',
        encoding="utf-8",
    )

    assert "hardcoded-public-ip-url" in _rules_for(source)


def test_security_scan_allows_loopback_and_private_healthcheck_urls(tmp_path: Path) -> None:
    source = tmp_path / "health.py"
    source.write_text(
        "\n".join(
            [
                'LOCAL = "http://127.0.0.1:8000/v1/models"',
                'LOCALHOST = "http://localhost:8001/health"',
                'PRIVATE = "http://10.0.0.8:8000/v1/models"',
                'PRIVATE_172 = "http://172.16.0.8:8000/v1/models"',
                'PRIVATE_192 = "http://192.168.1.8:8000/v1/models"',
            ]
        ),
        encoding="utf-8",
    )

    assert "hardcoded-public-ip-url" not in _rules_for(source)
