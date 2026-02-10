# Offline Bot (TWS + ClickHouse)

Local version that connects to TWS desktop app and stores data in ClickHouse.

## Requirements
- TWS or IB Gateway running locally
- ClickHouse server
- Python 3.11+

## Setup

1. Install ClickHouse:
```bash
brew install clickhouse
clickhouse server
```

2. Create database:
```bash
clickhouse client < sql/init/01_create_tables.sql
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Copy and edit .env:
```bash
cp .env.example .env
nano .env
```

5. Run:
```bash
python bot.py
```

## Features
- Connects to TWS on port 7497
- Stores all IV data in ClickHouse
- Historical data retrieval
- IV smile charts with history
