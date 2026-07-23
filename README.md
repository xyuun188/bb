# AI Crypto Quantitative Trading System

全自动 AI 量化交易机器人，支持多模型竞争、模拟盘/实盘切换、Web 可视化面板。

## Architecture

```
[Data Feed] -> [AI Brain (multi-model)] -> [Risk Manager] -> [Executor]
     |                                                          |
     +-----------> [Redis Pub/Sub] <-> [Web Dashboard :8002] <-+
                              |
                         [SQLite/PostgreSQL]
```

## Quick Start

### 1. Install Dependencies

```bash
cd D:\BB
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Edit `.env` file with your API keys:
- `OKX_API_KEY` / `OKX_API_SECRET` — OKX demo trading keys
- `AI_API_KEY` — OpenAI-compatible API key (uses yjxapi.top)

### 3. Initialize Database

```bash
python scripts/init_db.py
```

### 4. Run Paper Trading (Recommended First)

```bash
python scripts/run_paper_trading.py
```

Then open: **http://localhost:8002**

### 5. Run Live Trading (Demo)

```bash
python scripts/run_live_trading.py
```

## Modules

| Module | Description |
|--------|-------------|
| `config/` | Central configuration via pydantic-settings |
| `core/` | Exceptions, logging, trading mode manager |
| `models/` | SQLAlchemy ORM models (8 tables) |
| `db/` | Async session factory + repository pattern |
| `data_feed/` | OKX WebSocket/REST, news fetcher, technical indicators |
| `ai_brain/` | LLM agent, FinBERT sentiment, XGBoost classifier |
| `executor/` | Paper executor (virtual) + OKX executor (live/demo) |
| `risk_manager/` | Position limits, stop-loss, black swan, circuit breaker |
| `backtest/` | Backtrader integration for strategy validation |
| `services/` | Trading orchestration, model competition, notifications |
| `workers/` | Main event loop, data collector, model evaluator |
| `web_dashboard/` | FastAPI + WebSocket dashboard on port 8002 |

## AI Models

Three models compete in paper trading:

1. **LLM Agent** (`llm_agent`) — GPT-4 based decision via yjxapi.top
2. **FinBERT Sentiment** (`finbert_sentiment`) — Sentiment-driven baseline
3. **XGBoost** (`xgboost`) — ML classifier on technical + sentiment features

The best model (by composite score) is auto-promoted to live trading.

## API Endpoints (port 8002)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | System status |
| GET | `/api/dashboard/summary` | Aggregate dashboard data |
| GET | `/api/models` | Model rankings |
| GET | `/api/trades` | Trade history |
| GET | `/api/positions` | Open positions |
| POST | `/api/control/mode` | Switch paper/live |
| WS | `/ws` | Real-time updates |

## Risk Controls

- Max position size: 20% per symbol (configurable)
- Max leverage: 3x
- Daily loss limit: 5% (circuit breaker)
- Hard stop-loss: 5%
- Trailing stop: activates at 3% profit, trails at 1.5%
- Black swan detection: extreme sentiment keywords + flash crash detection
