#!/usr/bin/env python3
"""Test IB and Telegram connections."""

import os
import sys
import time
import threading
from dotenv import load_dotenv

load_dotenv()

def test_telegram():
    """Test Telegram bot connection."""
    print("\n[1/3] Testing Telegram connection...")
    try:
        import telebot
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not token or not chat_id:
            print("  ❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
            return False

        bot = telebot.TeleBot(token)
        bot.send_message(int(chat_id), "🔧 Connection test successful!")
        print("  ✅ Telegram: Connected! Check your Telegram for a test message.")
        return True
    except Exception as e:
        print(f"  ❌ Telegram error: {e}")
        return False

def test_clickhouse():
    """Test ClickHouse connection."""
    print("\n[2/3] Testing ClickHouse connection...")
    try:
        from connection import get_clickhouse_connection

        with get_clickhouse_connection() as client:
            result = client.execute("SELECT 1")
            if result == [(1,)]:
                print("  ✅ ClickHouse: Connected!")

                # Check if table exists
                tables = client.execute("SHOW TABLES FROM options")
                if any('greeks' in str(t) for t in tables):
                    print("  ✅ ClickHouse: options.greeks table exists")
                else:
                    print("  ⚠️  ClickHouse: options.greeks table not found")
                return True
    except Exception as e:
        print(f"  ❌ ClickHouse error: {e}")
        return False

def test_ib():
    """Test Interactive Brokers connection."""
    print("\n[3/3] Testing IB TWS/Gateway connection...")

    try:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        class TestApp(EWrapper, EClient):
            def __init__(self):
                EClient.__init__(self, self)
                self.connected = False
                self.next_order_id = None

            def nextValidId(self, orderId):
                self.next_order_id = orderId
                self.connected = True
                print(f"  ✅ IB: Connected! Next valid order ID: {orderId}")

            def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
                if errorCode in [2104, 2106, 2158]:  # Market data farm messages
                    print(f"  ℹ️  IB Info: {errorString}")
                elif errorCode == 502:
                    print(f"  ❌ IB: Cannot connect - TWS/Gateway not running or API not enabled")
                elif errorCode == 504:
                    print(f"  ❌ IB: Not connected")
                else:
                    print(f"  ⚠️  IB Error {errorCode}: {errorString}")

        host = os.getenv('IB_HOST', '127.0.0.1')
        port = int(os.getenv('IB_PORT', 7497))
        client_id = int(os.getenv('IB_CLIENT_ID', 22))

        print(f"  Connecting to {host}:{port} (clientId={client_id})...")

        app = TestApp()
        app.connect(host, port, clientId=client_id)

        # Start the client thread
        thread = threading.Thread(target=app.run, daemon=True)
        thread.start()

        # Wait for connection
        timeout = 10
        start = time.time()
        while not app.connected and time.time() - start < timeout:
            time.sleep(0.5)

        if app.connected:
            app.disconnect()
            return True
        else:
            print(f"  ❌ IB: Connection timeout after {timeout} seconds")
            print("\n  Please check:")
            print("    1. TWS or IB Gateway is running")
            print("    2. API is enabled (File → Global Configuration → API → Settings)")
            print("    3. Socket port matches IB_PORT in .env")
            print("    4. 'Enable ActiveX and Socket Clients' is checked")
            return False

    except Exception as e:
        print(f"  ❌ IB error: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("IB Options Bot - Connection Test")
    print("=" * 50)

    telegram_ok = test_telegram()
    clickhouse_ok = test_clickhouse()
    ib_ok = test_ib()

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"  Telegram:   {'✅ OK' if telegram_ok else '❌ FAILED'}")
    print(f"  ClickHouse: {'✅ OK' if clickhouse_ok else '❌ FAILED'}")
    print(f"  IB TWS:     {'✅ OK' if ib_ok else '❌ FAILED'}")
    print("=" * 50)

    if all([telegram_ok, clickhouse_ok, ib_ok]):
        print("\n🎉 All connections successful! You can run the bot with:")
        print("   source venv/bin/activate && python bot_tests.py")
    else:
        print("\n⚠️  Some connections failed. Please fix the issues above.")
        sys.exit(1)
