#!/usr/bin/env python3
"""Scan source files for common secret leakage patterns.

The scanner intentionally skips local secret containers such as .env, database
files, logs, and server credential text files. Its job is to catch secrets that
were accidentally hardcoded into source or printed to logs.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.safe_output import safe_print  # noqa: E402

SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".html",
    ".css",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".md",
}

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    ".codex-memory",
    ".claude",
    ".rtk",
    ".tmp",
    "data",
    "logs",
    "__pycache__",
}

SKIP_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "PROJECT_MEMORY.md",
}

SKIP_FILE_SUBSTRINGS = (
    "服务器资料",
    "模拟交易秘钥",
    "交易秘钥",
    "秘钥",
    "密钥",
    "密码",
)


@dataclass(frozen=True)
class Rule:
    name: str
    severity: str
    pattern: re.Pattern[str]
    message: str


@dataclass(frozen=True)
class Finding:
    severity: str
    rule: str
    path: Path
    line_no: int
    message: str
    snippet: str


RULES: tuple[Rule, ...] = (
    Rule(
        "private-key-block",
        "critical",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "Private key material must never be committed.",
    ),
    Rule(
        "aws-access-key",
        "critical",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "AWS-style access key detected in source.",
    ),
    Rule(
        "openai-style-token",
        "high",
        re.compile(r"\bsk-[A-Za-z0-9_\-]{24,}\b"),
        "OpenAI-style API token detected in source.",
    ),
    Rule(
        "hardcoded-secret-assignment",
        "high",
        re.compile(
            r"(?i)\b(api[_-]?key|api[_-]?secret|secret|password|passphrase|token)\b"
            r"\s*[:=]\s*['\"]([^'\"\n]{12,})['\"]"
        ),
        "Potential hardcoded secret assignment detected.",
    ),
    Rule(
        "secret-output",
        "high",
        re.compile(
            r"(?i)\b(print|logger\.(debug|info|warning|error|exception))\s*\("
            r".*\b(api[_-]?key|api[_-]?secret|secret|password|passphrase|token)\b"
        ),
        "Potential secret logging or printing detected.",
    ),
    Rule(
        "hardcoded-bearer-token",
        "high",
        re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{24,}"),
        "Hardcoded bearer token detected.",
    ),
    Rule(
        "hardcoded-public-ip-url",
        "high",
        re.compile(
            r"(?i)\bhttps?://"
            r"(?!(?:127\.0\.0\.1|localhost|0\.0\.0\.0|10\.|192\.168\.|"
            r"172\.(?:1[6-9]|2[0-9]|3[01])\.))"
            r"\d{1,3}(?:\.\d{1,3}){3}(?::\d{2,5})?(?:/[^\s'\"<>)]*)?"
        ),
        "Hardcoded public IP URL detected; use environment or local server-info config.",
    ),
)

PLACEHOLDER_RE = re.compile(
    r"(?i)^(local|none|null|changeme|example|placeholder|test|dummy|\*+|x+|<.*>|sk-\.\.\.)$"
)


def should_skip(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    if any(part in SKIP_DIRS for part in rel_parts[:-1]):
        return True
    name = path.name
    if name in SKIP_FILE_NAMES:
        return True
    path_text = str(path)
    if any(token in path_text for token in SKIP_FILE_SUBSTRINGS):
        return True
    return path.suffix.lower() not in SOURCE_SUFFIXES


def mask_snippet(line: str) -> str:
    text = line.strip()
    text = re.sub(r"(sk-)[A-Za-z0-9_\-]{8,}", r"\1***", text)
    text = re.sub(r"(AKIA)[0-9A-Z]{8,}", r"\1***", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]{8,}", r"\1***", text)
    text = re.sub(
        r"(?i)(api[_-]?key|api[_-]?secret|secret|password|passphrase|token)(\s*[:=]\s*)"
        r"(['\"])([^'\"]+)(['\"])",
        r"\1\2\3***\5",
        text,
    )
    return text[:220]


def is_placeholder_match(match: re.Match[str]) -> bool:
    if match.lastindex and match.lastindex >= 2:
        value = match.group(2).strip()
        return bool(PLACEHOLDER_RE.match(value))
    return False


def scan_file(path: Path) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [
            Finding(
                severity="medium",
                rule="scan-read-error",
                path=path,
                line_no=0,
                message=f"Could not read file: {exc}",
                snippet="",
            )
        ]

    findings: list[Finding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for rule in RULES:
            match = rule.pattern.search(line)
            if not match:
                continue
            if rule.name == "hardcoded-secret-assignment" and is_placeholder_match(match):
                continue
            findings.append(
                Finding(
                    severity=rule.severity,
                    rule=rule.name,
                    path=path,
                    line_no=line_no,
                    message=rule.message,
                    snippet=mask_snippet(line),
                )
            )
    return findings


def iter_files(paths: list[Path]) -> list[Path]:
    selected: list[Path] = []
    for item in paths:
        path = item if item.is_absolute() else ROOT / item
        if path.is_file():
            if not should_skip(path):
                selected.append(path)
            continue
        for candidate in path.rglob("*"):
            if candidate.is_file() and not should_skip(candidate):
                selected.append(candidate)
    return sorted(set(selected))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan")
    parser.add_argument(
        "--fail-on",
        choices=["critical", "high", "medium", "low"],
        default="high",
        help="Lowest severity that should fail the scan.",
    )
    args = parser.parse_args()

    severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    fail_level = severity_order[args.fail_on]
    files = iter_files([Path(p) for p in args.paths])
    findings = [finding for path in files for finding in scan_file(path)]

    if not findings:
        safe_print(f"source safety scan ok: scanned {len(files)} files")
        return 0

    for finding in findings:
        rel = finding.path.relative_to(ROOT)
        safe_print(
            f"{finding.severity.upper()} {finding.rule} {rel}:{finding.line_no} - {finding.message}"
        )
        if finding.snippet:
            safe_print(f"  {finding.snippet}")

    should_fail = any(severity_order[finding.severity] >= fail_level for finding in findings)
    return 1 if should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
