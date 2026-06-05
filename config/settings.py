"""
Central configuration using pydantic-settings.
All modules import settings from here. Single source of truth.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ENSEMBLE_TRADER_NAME = "ensemble_trader"
DECISION_MAKER_NAME = "decision_maker"

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


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


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
    ai_models: list[dict] = Field(default_factory=list)
    # Backward compatibility: fallback for single-model setups
    ai_api_base: str = "http://175.155.64.171:31840/v1"
    ai_api_key: str = "local"
    ai_model: str = "deepseek-r1-distill-qwen-32b-trade"
    local_ai_tools_enabled: bool = True
    local_ai_tools_api_base: str = "http://175.155.64.171:31841"
    local_ai_tools_api_key: str = ""
    local_ai_tools_timeout_seconds: float = 2.5
    high_risk_review_enabled: bool = True
    high_risk_review_api_base: str = "http://175.155.64.171:31841/v1"
    high_risk_review_api_key: str = "local"
    high_risk_review_model: str = "deepseek-r1-distill-qwen-32b-trade"

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///./data/trading.db"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"
    use_fakeredis: bool = True

    # --- Trading Parameters ---
    trading_mode: TradingMode = TradingMode.PAPER
    scan_mode: str = "auto"  # "auto" = scan all OKX pairs, "manual" = only settings.symbols
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    initial_virtual_balance: float = 100_000.0
    model_initial_balances: dict[str, float] = Field(default_factory=dict)
    execution_account_name: str = "多专家执行账户"
    execution_account_balances: dict[str, float] = Field(default_factory=dict)
    execution_account_max_loss_pct: dict[str, float] = Field(default_factory=dict)
    execution_account_max_loss_usdt: dict[str, float] = Field(default_factory=dict)
    execution_account_cooldown_loss_pct: dict[str, float] = Field(
        default_factory=lambda: {"paper": 0.5, "live": 0.5}
    )
    decision_interval_seconds: int = 60
    confidence_threshold: float = 0.50
    auto_scan_symbol_limit: int = 20
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
    ai_expert_max_completion_tokens: int = 520
    ai_decision_maker_max_completion_tokens: int = 720
    ai_batch_experts_enabled: bool = True
    ai_batch_expert_max_completion_tokens: int = 1100
    ai_batch_expert_timeout_seconds: float = 90.0
    ai_batch_expert_circuit_breaker_seconds: float = 0.0
    ai_market_fast_prefilter_enabled: bool = True
    ai_market_fast_prefilter_min_expected_return_pct: float = 0.03
    ai_market_fast_prefilter_max_loss_probability: float = 0.58
    sentiment_blocking_timeout_seconds: float = 6.0
    cryptopanic_api_key: str = ""
    coinmarketcal_api_key: str = ""
    newsapi_api_key: str = ""

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
    dashboard_host: str = "0.0.0.0"

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

        Falls back to old unified fields if new split fields are empty.
        Paper always uses sandbox/demo; live always uses real exchange.
        """
        m = mode or self.trading_mode.value
        if m == "paper":
            key = self.okx_paper_api_key or self.okx_api_key
            secret = self.okx_paper_api_secret or self.okx_api_secret
            passphrase = self.okx_paper_passphrase or self.okx_passphrase
        else:  # live
            key = self.okx_live_api_key or self.okx_api_key
            secret = self.okx_live_api_secret or self.okx_api_secret
            passphrase = self.okx_live_passphrase or self.okx_passphrase
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
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return [s.strip() for s in v.split(",")]
        return v

    @field_validator("model_initial_balances", mode="before")
    @classmethod
    def parse_model_balances(cls, v: Any) -> dict[str, float]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {}
        return v or {}

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
            try:
                parsed = json.loads(v)
            except (json.JSONDecodeError, TypeError):
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
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return []
        return v or []

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
                "api_key": cfg.get("api_key", ""),
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
            "api_base": str(updates.get("api_base", existing.get("api_base", self.ai_api_base)) or "").strip(),
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
        return self.model_initial_balances.get(
            model_name, self.initial_virtual_balance
        )

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
        d = self.model_dump()
        for key in ("okx_api_secret", "okx_api_key", "okx_paper_api_key", "okx_paper_api_secret",
                     "okx_live_api_key", "okx_live_api_secret",
                     "ai_api_key", "local_ai_tools_api_key", "high_risk_review_api_key",
                     "telegram_bot_token", "dingtalk_webhook_url"):
            if d.get(key):
                d[key] = d[key][:8] + "***" if len(d[key]) > 8 else "***"
        # Mask api_key in each ai_models entry
        if d.get("ai_models"):
            safe_models = []
            for m in d["ai_models"]:
                mc = dict(m)
                if mc.get("api_key") and len(mc["api_key"]) > 8:
                    mc["api_key"] = mc["api_key"][:8] + "***"
                elif mc.get("api_key"):
                    mc["api_key"] = "***"
                safe_models.append(mc)
            d["ai_models"] = safe_models
        return d

    def update_env_file(self, updates: dict[str, Any]) -> None:
        """Write key=value updates to the .env file, preserving existing lines."""
        env_path = self.project_root / ".env"
        if not env_path.exists():
            return

        updates_str = {str(k): str(v) for k, v in updates.items()}
        lines = env_path.read_text(encoding="utf-8").splitlines()
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
                    new_lines.append(f"{key}={updates_str[key]}")
                    updated_keys.add(key)
                    continue
            new_lines.append(line)

        # Append keys not found in existing file
        for k, v in updates_str.items():
            if k not in updated_keys:
                new_lines.append(f"{k}={v}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    def update_symbols(self, new_symbols: list[str]) -> None:
        """Update symbols at runtime and persist to .env."""
        self.symbols = new_symbols
        self.update_env_file({"SYMBOLS": json.dumps(new_symbols)})


# Singleton
settings = Settings()
