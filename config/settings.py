"""
Central configuration using pydantic-settings.
All modules import settings from here. Single source of truth.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from dotenv import dotenv_values

from core.secret_utils import is_masked_secret, is_sensitive_key, redact_mapping

ENSEMBLE_TRADER_NAME = "ensemble_trader"
DECISION_MAKER_NAME = "decision_maker"
ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
ENV_SIMPLE_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@,+-]*$")
RUNTIME_ENV_REFRESH_MIN_SECONDS = 2.0

FIXED_AI_MODEL_SLOTS: list[dict[str, Any]] = [
    {
        "name": "trend_expert",
        "label": "行情方向专家",
        "role": "trend_direction",
        "description": "只判断当前币种短线方向：做多、做空、震荡或不确定；不负责仓位和风控。",
        "weight": 0.18,
    },
    {
        "name": "momentum_expert",
        "label": "盈利质量专家",
        "role": "profit_quality",
        "description": "判断这笔交易值不值得做：预期净收益、亏损概率、盈亏比、手续费覆盖和小赚大亏风险。",
        "weight": 0.30,
    },
    {
        "name": "sentiment_expert",
        "label": "短线时序专家",
        "role": "short_timeseries",
        "description": "判断未来 1/5/10/30 分钟方向、动量延续、回撤风险、假突破和事件/情绪冲击。",
        "weight": 0.20,
    },
    {
        "name": "position_expert",
        "label": "持仓退出专家",
        "role": "position_exit",
        "description": "只看已有仓位：浮盈落袋、亏损修复、加仓、减仓、全平和资金轮转。",
        "weight": 0.20,
    },
    {
        "name": "risk_expert",
        "label": "异常风控专家",
        "role": "risk_anomaly",
        "description": "只负责异常插针、流动性、极端波动、保证金、交易所限制和硬风险拦截。",
        "weight": 0.12,
    },
    {
        "name": DECISION_MAKER_NAME,
        "label": "最终交易员",
        "role": "final_decision",
        "description": "读取专家、交叉验证、本地模型和风控证据后，只做最终交易动作确认或否决。",
        "weight": 0.0,
    },
]


class TradingMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


def _format_env_value(value: str) -> str:
    """Format one-line dotenv values without changing simple existing output."""
    if value == "" or ENV_SIMPLE_VALUE_RE.fullmatch(value):
        return value
    return json.dumps(value, ensure_ascii=False)


def _split_top_level_commas(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote = ""
    escape = False
    for idx, char in enumerate(value):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            parts.append(value[start:idx].strip())
            start = idx + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _split_legacy_key_value(value: str) -> tuple[str, str] | None:
    quote = ""
    escape = False
    depth = 0
    for idx, char in enumerate(value):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(depth - 1, 0)
        elif char == ":" and depth == 0:
            return value[:idx].strip(), value[idx + 1 :].strip()
    return None


def _parse_scalar_env_value(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        parsed = _parse_complex_env_value(text)
        if parsed is not text:
            return parsed
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return text.strip("'\"")
    if number.is_integer() and not re.search(r"[.eE]", text):
        return int(number)
    return number


def _parse_legacy_mapping(text: str) -> dict[str, Any] | None:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return {}
    result: dict[str, Any] = {}
    for item in _split_top_level_commas(inner):
        pair = _split_legacy_key_value(item)
        if pair is None:
            return None
        key, raw_value = pair
        key = str(_parse_scalar_env_value(key)).strip()
        if not key:
            return None
        result[key] = _parse_scalar_env_value(raw_value)
    return result


def _parse_legacy_sequence(text: str) -> list[Any] | None:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return []
    return [_parse_scalar_env_value(item) for item in _split_top_level_commas(inner)]


def _parse_complex_env_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    if text.startswith("{") and text.endswith("}"):
        parsed_map = _parse_legacy_mapping(text)
        if parsed_map is not None:
            return parsed_map
    if text.startswith("[") and text.endswith("]"):
        parsed_list = _parse_legacy_sequence(text)
        if parsed_list is not None:
            return parsed_list
    return text


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OKX Exchange (separate credentials for paper/demo vs live/real) ---
    # Paper trading (OKX demo/sandbox)
    okx_paper_api_key: str = ""
    okx_paper_api_secret: str = ""
    okx_paper_passphrase: str = ""
    # Live trading (OKX real exchange)
    okx_live_api_key: str = ""
    okx_live_api_secret: str = ""
    okx_live_passphrase: str = ""
    # Backward compatibility: old unified fields
    okx_api_key: str = ""
    okx_api_secret: str = ""
    okx_passphrase: str = ""
    okx_demo: bool = True
    okx_proxy: str = ""

    # --- AI API (OpenAI-compatible) ---
    # Per-model AI configurations (list of dicts from AI_MODELS env var)
    # Each element: {"name": "gpt-5.4", "api_base": "...", "api_key": "...", "model": "...", "balance": 1000}
    ai_models: Annotated[list[dict], NoDecode] = Field(default_factory=list)
    # Backward compatibility: fallback for single-model setups
    ai_api_base: str = ""
    ai_api_key: str = ""
    ai_model: str = "qwen3-32b-trade"
    local_ai_tools_enabled: bool = False
    local_ai_tools_api_base: str = ""
    local_ai_tools_api_key: str = ""
    local_ai_tools_timeout_seconds: float = 8.0
    local_ai_tools_circuit_breaker_failures: int = 3
    local_ai_tools_circuit_breaker_cooldown_seconds: float = 45.0
    high_risk_review_enabled: bool = True
    high_risk_review_api_base: str = ""
    high_risk_review_api_key: str = ""
    high_risk_review_model: str = "deepseek-reasoner"
    high_risk_review_timeout_seconds: float = 30.0
    high_risk_review_max_tokens: int = 480
    high_risk_review_circuit_breaker_failures: int = 2
    high_risk_review_circuit_breaker_cooldown_seconds: float = 120.0

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./data/trading.db"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    use_fakeredis: bool = True

    # --- Trading Parameters ---
    trading_mode: TradingMode = TradingMode.PAPER
    scan_mode: str = "auto"  # "auto" = scan all OKX pairs, "manual" = only settings.symbols
    symbols: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )
    initial_virtual_balance: float = 100_000.0
    model_initial_balances: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    execution_account_name: str = "多专家执行账户"
    execution_account_balances: Annotated[dict[str, float], NoDecode] = Field(default_factory=dict)
    execution_account_max_loss_pct: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=dict
    )
    execution_account_max_loss_usdt: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=dict
    )
    execution_account_cooldown_loss_pct: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: {"paper": 0.5, "live": 0.5}
    )
    decision_interval_seconds: int = 60
    confidence_threshold: float = 0.50
    auto_scan_symbol_limit: int = 20
    market_analysis_watchdog_seconds: int = 180
    position_analysis_watchdog_seconds: int = 180
    max_auto_trades_per_round: int = 3
    max_open_positions_per_model: int = 20
    max_same_symbol_positions_per_side: int = 2
    min_entry_adx: float = 15.0
    min_entry_volume_ratio: float = 0.2
    daily_profit_target_usdt: float = 0.0
    daily_profit_target_cny: float = 0.0
    cny_per_usdt_assumption: float = 7.2
    expert_memory_enabled: bool = True
    expert_memory_per_prompt: int = 4
    shadow_memory_enabled: bool = True
    shadow_memory_min_return_pct: float = 0.40
    ai_llm_concurrency: int = 2
    ai_llm_call_delay_seconds: float = 0.15
    ai_expert_timeout_seconds: float = 30.0
    ai_decision_maker_timeout_seconds: float = 20.0
    ai_expert_max_completion_tokens: int = 360
    ai_decision_maker_max_completion_tokens: int = 320
    ai_batch_experts_enabled: bool = True
    ai_batch_expert_max_completion_tokens: int = 900
    ai_batch_expert_timeout_seconds: float = 35.0
    ai_batch_expert_circuit_breaker_seconds: float = 0.0
    ai_batch_expert_format_failure_circuit_breaker_seconds: float = 180.0
    strategy_learning_llm_candidates_enabled: bool = True
    strategy_learning_min_trade_count_target: int = 8
    strategy_learning_llm_candidate_interval_seconds: int = 21600
    strategy_learning_llm_candidate_timeout_seconds: float = 20.0
    strategy_learning_llm_candidate_max_tokens: int = 360
    ai_market_fast_prefilter_enabled: bool = True
    ai_market_fast_prefilter_min_expected_return_pct: float = 0.03
    ai_market_fast_prefilter_max_loss_probability: float = 0.58
    sentiment_blocking_timeout_seconds: float = 6.0
    cryptopanic_api_key: str = ""
    coinmarketcal_api_key: str = ""
    newsapi_api_key: str = ""
    external_event_scraper_enabled: bool = False
    external_event_scraper_sources: Annotated[list[dict[str, Any]], NoDecode] = Field(
        default_factory=list
    )
    external_event_scraper_interval_seconds: int = 900
    external_event_scraper_timeout_seconds: float = 6.0
    external_event_scraper_max_sources: int = 6
    external_event_scraper_max_items_per_source: int = 8
    vector_memory_enabled: bool = False
    vector_memory_backend: str = "auto"
    vector_memory_dimension: int = 96
    vector_memory_max_documents: int = 20000
    vector_memory_decision_index_limit: int = 1200
    vector_memory_news_index_limit: int = 1500
    vector_memory_min_score: float = 0.18
    vector_memory_auto_reindex_enabled: bool = True
    vector_memory_auto_reindex_interval_seconds: int = 1800

    # --- Risk Management ---
    max_position_pct: float = 0.25
    max_total_margin_pct: float = 0.0  # 0 = use legacy max_position_pct * 3
    max_leverage: float = 20.0
    max_daily_loss_pct: float = 0.05
    max_slippage_pct: float = 0.005
    hard_stop_loss_pct: float = 0.05
    trailing_stop_activation: float = 0.03
    trailing_stop_distance: float = 0.015

    # --- Web Dashboard ---
    dashboard_port: int = 8002
    dashboard_host: str = "127.0.0.1"
    dashboard_cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    dashboard_admin_api_key: str = ""
    dashboard_inline_enabled: bool = True
    dashboard_auth_enabled: bool = False
    dashboard_auth_username: str = ""
    dashboard_auth_password_hash: str = ""
    dashboard_auth_email: str = ""
    dashboard_session_secret: str = ""
    dashboard_session_ttl_seconds: int = 43200
    dashboard_auth_cookie_secure: bool = False
    system_audit_history_enabled: bool = True
    system_audit_history_interval_seconds: int = 300
    system_audit_history_max_records: int = 500

    # --- Notifications ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    dingtalk_webhook_url: str = ""

    # --- Logging ---
    log_level: str = "INFO"
    log_file: str = "./data/trading.log"
    log_max_bytes: int = 200 * 1024 * 1024
    log_backup_count: int = 5

    # --- Derived paths ---
    _runtime_env_last_mtime: float = 0.0
    _runtime_env_last_checked: float = 0.0

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def data_dir(self) -> Path:
        p = self.project_root / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_paper_trading(self) -> bool:
        return self.trading_mode == TradingMode.PAPER

    @property
    def is_live_trading(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    def get_okx_credentials(self, mode: str | None = None) -> dict[str, str]:
        """Return OKX credentials for the given mode ("paper" or "live").

        Paper may use legacy unified fields for backward compatibility.
        Live must use live-specific fields so demo/global credentials cannot
        accidentally unlock real-exchange mode.
        """
        m = mode or self.trading_mode.value
        if m == "paper":
            key = self.okx_paper_api_key or self.okx_api_key
            secret = self.okx_paper_api_secret or self.okx_api_secret
            passphrase = self.okx_paper_passphrase or self.okx_passphrase
        else:  # live
            key = self.okx_live_api_key
            secret = self.okx_live_api_secret
            passphrase = self.okx_live_passphrase
        result = {"api_key": key, "api_secret": secret}
        if passphrase:
            result["passphrase"] = passphrase
        return result

    def is_okx_demo(self, mode: str | None = None) -> bool:
        """Paper mode always uses demo/sandbox; live always uses real."""
        m = mode or self.trading_mode.value
        return m == "paper"

    @field_validator("symbols", mode="before")
    @classmethod
    def parse_symbols(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            parsed = _parse_complex_env_value(v)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator("dashboard_cors_origins", mode="before")
    @classmethod
    def parse_dashboard_cors_origins(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            if not v.strip():
                return []
            parsed = _parse_complex_env_value(v)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    @field_validator("external_event_scraper_sources", mode="before")
    @classmethod
    def parse_external_event_scraper_sources(cls, v: Any) -> list[dict[str, Any]]:
        if isinstance(v, str):
            if not v.strip():
                return []
            parsed = _parse_complex_env_value(v)
            if isinstance(parsed, dict):
                return [dict(parsed)]
            if isinstance(parsed, list):
                sources: list[dict[str, Any]] = []
                for item in parsed:
                    if isinstance(item, dict):
                        sources.append(dict(item))
                    elif str(item or "").strip():
                        sources.append({"url": str(item).strip()})
                return sources
            return [{"url": item.strip()} for item in v.split(",") if item.strip()]
        if isinstance(v, dict):
            return [dict(v)]
        if isinstance(v, list):
            return [dict(item) for item in v if isinstance(item, dict)]
        return []

    def dashboard_allowed_origins(self) -> list[str]:
        """Return explicit Dashboard CORS origins without wildcard credentials."""
        if self.dashboard_cors_origins:
            return list(dict.fromkeys(self.dashboard_cors_origins))
        port = int(self.dashboard_port or 8002)
        host = str(self.dashboard_host or "127.0.0.1").strip()
        origins = [f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
        wildcard_hosts = {"0.0.0.0", "::"}  # noqa: S104 - rejected, not bound.
        if host and host not in {"127.0.0.1", "localhost"} and host not in wildcard_hosts:
            origins.append(f"http://{host}:{port}")
        return list(dict.fromkeys(origins))

    @field_validator("model_initial_balances", mode="before")
    @classmethod
    def parse_model_balances(cls, v: Any) -> dict[str, float]:
        if isinstance(v, str):
            parsed = _parse_complex_env_value(v)
            if isinstance(parsed, dict):
                return {str(k): float(val) for k, val in parsed.items() if val is not None}
            return {}
        if isinstance(v, dict):
            return {str(k): float(val) for k, val in v.items() if val is not None}
        return {}

    @field_validator(
        "execution_account_balances",
        "execution_account_max_loss_pct",
        "execution_account_max_loss_usdt",
        "execution_account_cooldown_loss_pct",
        mode="before",
    )
    @classmethod
    def parse_execution_account_maps(cls, v: Any) -> dict[str, float]:
        if isinstance(v, str):
            parsed = _parse_complex_env_value(v)
            if not isinstance(parsed, dict):
                return {}
            return {str(k): float(val) for k, val in parsed.items() if val is not None}
        if isinstance(v, dict):
            return {str(k): float(val) for k, val in v.items() if val is not None}
        return {}

    @field_validator("ai_models", mode="before")
    @classmethod
    def parse_ai_models(cls, v: Any) -> list[dict]:
        if isinstance(v, str):
            if not v.strip():
                return []
            parsed = _parse_complex_env_value(v)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [dict(item) for item in parsed if isinstance(item, dict)]
            return []
        if isinstance(v, dict):
            return [v]
        if isinstance(v, list):
            return [dict(item) for item in v if isinstance(item, dict)]
        return []

    def get_fixed_ai_models(self, include_empty: bool = True) -> list[dict[str, Any]]:
        """Return AI model configs merged into the fixed expert slots."""
        configured_by_name = {
            str(m.get("name", "")): dict(m)
            for m in self.ai_models
            if isinstance(m, dict) and m.get("name")
        }
        fixed_names = {slot["name"] for slot in FIXED_AI_MODEL_SLOTS}
        has_fixed_config = any(name in configured_by_name for name in fixed_names)
        legacy_cfg = None
        if not has_fixed_config:
            legacy_cfg = next(
                (dict(m) for m in self.ai_models if isinstance(m, dict) and m.get("api_key")),
                None,
            )
        result: list[dict[str, Any]] = []
        for slot in FIXED_AI_MODEL_SLOTS:
            cfg = dict(configured_by_name.get(slot["name"], legacy_cfg or {}))
            merged = {
                **slot,
                "api_base": cfg.get("api_base", self.ai_api_base),
                "api_key": cfg.get("api_key") or self.ai_api_key,
                "model": cfg.get("model", self.ai_model),
                "enabled": bool(cfg.get("enabled", True)),
            }
            if "balance" in cfg:
                merged["balance"] = cfg["balance"]
            if include_empty or merged.get("api_key"):
                result.append(merged)
        return result

    def set_fixed_ai_model(self, name: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Update one fixed expert slot and persist it in settings.ai_models."""
        slots_by_name = {slot["name"]: slot for slot in FIXED_AI_MODEL_SLOTS}
        if name not in slots_by_name:
            raise ValueError(f"Unknown fixed AI model slot: {name}")

        current = {m.get("name"): dict(m) for m in self.ai_models if isinstance(m, dict)}
        fixed_names = set(slots_by_name)
        has_fixed_config = any(slot_name in current for slot_name in fixed_names)
        if not has_fixed_config:
            legacy_cfg = next(
                (dict(m) for m in self.ai_models if isinstance(m, dict) and m.get("api_key")),
                {},
            )
            for slot in FIXED_AI_MODEL_SLOTS:
                current[slot["name"]] = {
                    **legacy_cfg,
                    "name": slot["name"],
                    "role": slot["role"],
                    "label": slot["label"],
                    "weight": slot["weight"],
                }
        existing = current.get(name, {})
        slot = slots_by_name[name]
        updated = {
            **existing,
            "name": name,
            "role": slot["role"],
            "label": slot["label"],
            "weight": slot["weight"],
            "api_base": str(
                updates.get("api_base", existing.get("api_base", self.ai_api_base)) or ""
            ).strip(),
            "api_key": str(updates.get("api_key", existing.get("api_key", "")) or "").strip(),
            "model": str(updates.get("model", existing.get("model", self.ai_model)) or "").strip(),
            "enabled": bool(updates.get("enabled", existing.get("enabled", True))),
        }
        current[name] = updated

        ordered = []
        for fixed in FIXED_AI_MODEL_SLOTS:
            cfg = current.get(fixed["name"])
            if cfg:
                ordered.append(cfg)
        self.ai_models = ordered
        return updated

    def get_initial_balance(self, model_name: str) -> float:
        """Return initial balance for a specific model.

        Looks up model_initial_balances dict first, falls back to global default.
        """
        if model_name == ENSEMBLE_TRADER_NAME:
            return self.execution_account_balances.get(
                "paper",
                self.model_initial_balances.get(model_name, self.initial_virtual_balance),
            )
        return self.model_initial_balances.get(model_name, self.initial_virtual_balance)

    def get_execution_account_config(self, mode: str = "paper") -> dict[str, Any]:
        """Return the mode-specific execution account quota and risk settings."""
        mode = "live" if mode == "live" else "paper"
        allocated = self.execution_account_balances.get(
            mode,
            self.initial_virtual_balance if mode == "paper" else 0.0,
        )
        max_loss_pct = self.execution_account_max_loss_pct.get(mode, self.max_daily_loss_pct)
        max_loss_usdt = self.execution_account_max_loss_usdt.get(
            mode,
            allocated * max_loss_pct if allocated > 0 else 0.0,
        )
        return {
            "mode": mode,
            "account_name": self.execution_account_name,
            "internal_model_name": ENSEMBLE_TRADER_NAME,
            "allocated_balance": allocated,
            "max_loss_pct": max_loss_pct,
            "max_loss_usdt": max_loss_usdt,
            "cooldown_loss_pct": self.execution_account_cooldown_loss_pct.get(mode, 0.5),
        }

    def to_safe_dict(self) -> dict[str, Any]:
        """Export settings with secrets masked, for logging/debugging."""
        return redact_mapping(self.model_dump())

    def update_env_file(self, updates: dict[str, Any]) -> None:
        """Write key=value updates to the .env file, preserving existing lines."""
        env_path = self.project_root / ".env"

        updates_str: dict[str, str] = {}
        for raw_key, raw_value in updates.items():
            key = str(raw_key or "").strip()
            if raw_value is None:
                value = ""
            elif isinstance(raw_value, (dict, list, tuple)):
                value = json.dumps(raw_value, ensure_ascii=False)
            else:
                value = str(raw_value)
            if not ENV_KEY_RE.fullmatch(key):
                raise ValueError(f"Invalid .env key: {key!r}")
            if "\n" in value or "\r" in value:
                raise ValueError(f"Invalid newline in .env value for {key}")
            if is_sensitive_key(key) and is_masked_secret(value):
                continue
            updates_str[key] = value
        if not updates_str:
            return

        env_path.parent.mkdir(parents=True, exist_ok=True)
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        updated_keys = set()
        new_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue
            if "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in updates_str:
                    new_lines.append(f"{key}={_format_env_value(updates_str[key])}")
                    updated_keys.add(key)
                    continue
            new_lines.append(line)

        # Append keys not found in existing file
        for k, v in updates_str.items():
            if k not in updated_keys:
                new_lines.append(f"{k}={_format_env_value(v)}")

        tmp_path = env_path.with_name(f"{env_path.name}.tmp")
        tmp_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        os.replace(tmp_path, env_path)

    def refresh_runtime_env(self, *, force: bool = False) -> bool:
        """Refresh safe runtime knobs from .env for long-lived worker processes."""
        now = time.monotonic()
        if (
            not force
            and now - self._runtime_env_last_checked < RUNTIME_ENV_REFRESH_MIN_SECONDS
        ):
            return False
        self._runtime_env_last_checked = now

        env_path = self.project_root / ".env"
        if not env_path.exists():
            return False
        try:
            mtime = env_path.stat().st_mtime
        except OSError:
            return False
        if not force and mtime <= self._runtime_env_last_mtime:
            return False

        values = dotenv_values(env_path)
        changed = False

        def update_int(key: str, attr: str, min_value: int, max_value: int) -> None:
            nonlocal changed
            raw = values.get(key)
            if raw is None:
                return
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                return
            value = min(max(value, min_value), max_value)
            if getattr(self, attr) != value:
                setattr(self, attr, value)
                changed = True

        update_int("DECISION_INTERVAL_SECONDS", "decision_interval_seconds", 10, 3600)
        update_int(
            "MARKET_ANALYSIS_WATCHDOG_SECONDS",
            "market_analysis_watchdog_seconds",
            60,
            1800,
        )
        update_int(
            "POSITION_ANALYSIS_WATCHDOG_SECONDS",
            "position_analysis_watchdog_seconds",
            60,
            1800,
        )
        self._runtime_env_last_mtime = mtime
        return changed

    def update_symbols(self, new_symbols: list[str]) -> None:
        """Update symbols at runtime and persist to .env."""
        self.symbols = new_symbols
        self.update_env_file({"SYMBOLS": json.dumps(new_symbols)})


# Singleton
settings = Settings()
