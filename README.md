# IB Options Screener Bot

SPY options IV (Implied Volatility) monitoring bot with Telegram alerts.

## Two Versions

### 1. `offline_bot/` - Local Development
- Connects to **TWS desktop app** (port 7497)
- Stores data in **ClickHouse** database
- Full historical data support
- For local testing and development

### 2. `headless_server_bot/` - Production Server
- Runs on **DigitalOcean** ($12/month)
- Uses **IB Gateway** in Docker (headless)
- No database (in-memory only)
- Telegram alerts only

## Features
- Monitors SPY options (strikes 600-800, 3 expirations)
- IV smile visualization
- Anomaly detection alerts
- Telegram commands: `/ivc`, `/ivp`, `/status`

## Quick Start (Server)

```bash
cd headless_server_bot
cp .env.example .env
nano .env  # Add your IB credentials
docker compose up -d
```

Then connect via VNC for 2FA: `open vnc://YOUR_IP:5900`

## Quick Start (Local)

```bash
cd offline_bot
pip install -r requirements.txt
cp .env.example .env
nano .env
python bot.py
```

Requires TWS running locally and ClickHouse server.
