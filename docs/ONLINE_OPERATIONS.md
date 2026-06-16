# Online Operations

This project is deployed on the online server under `/data/bb/app`. The trading engine runs as `bb-paper-trading.service`, while the Dashboard can run separately as `bb-dashboard.service`. The online services keep using the ignored `/data/bb/app/.env`, `/etc/bb/bb-runtime.env`, and PostgreSQL socket connection.

## Local database access

Local tools should read the online PostgreSQL database through an SSH tunnel. Keep this process running in a terminal:

```powershell
python scripts/start_online_db_tunnel.py
```

The local ignored `.env` should use:

```dotenv
DATABASE_URL=postgresql+asyncpg://bb:<password>@127.0.0.1:15432/bb_trading
```

Do not start the trading app locally. The local tunnel is only for database access and scripts that need to inspect the online database.

## Deploy code changes

After changing code locally, sync source files to the online server and restart the services:

```powershell
python scripts/sync_to_online_server.py --split-services
```

The sync script uploads source files to `/data/bb/app`, skips secrets/data/logs/virtualenvs/Git metadata, fixes ownership to `bb:bb`, purges sensitive server-info `.txt` files, ensures `/etc/bb/bb-runtime.env` exists with `BB_SECURE_SETTINGS_KEY`, `DASHBOARD_AUTH_ENABLED=true`, `DASHBOARD_INLINE_ENABLED=false`, and Redis settings, restarts `bb-paper-trading.service` plus `bb-dashboard.service`, and checks the dashboard responds on the server.

Useful options:

```powershell
python scripts/sync_to_online_server.py --dry-run
python scripts/sync_to_online_server.py --skip-restart
python scripts/sync_to_online_server.py --include-tests
```
