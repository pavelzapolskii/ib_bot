# Headless Server Bot (DigitalOcean + IB Gateway)

Production version for headless servers. Uses IB Gateway in Docker, no database.

## Requirements
- Docker & Docker Compose
- DigitalOcean droplet (or any VPS with 2GB+ RAM)
- IB account with API access

## Setup

1. Clone and enter directory:
```bash
git clone https://github.com/YOUR_USERNAME/ib_bot.git
cd ib_bot/headless_server_bot
```

2. Create .env file:
```bash
cp .env.example .env
nano .env
```

3. Start containers:
```bash
docker compose pull
docker compose build
docker compose up -d
```

4. Connect via VNC for 2FA:
```bash
open vnc://YOUR_SERVER_IP:5900
```
Approve 2FA on IB Mobile app.

5. Verify:
```bash
docker compose ps
docker compose logs -f ib-bot
```

## Features
- Runs in Docker with IB Gateway
- No database (in-memory only)
- Telegram alerts for IV anomalies
- Commands: /ivc, /ivp, /status

## Cost
- DigitalOcean: $12/month (2GB RAM, 1 CPU)
