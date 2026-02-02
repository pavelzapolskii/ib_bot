import os
import telebot
from telebot import types
from dotenv import load_dotenv

import re
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import MarketDataTypeEnum
import threading
import datetime
import matplotlib.pyplot as plt
import numpy as np
import math

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID'))
IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
IB_PORT = int(os.getenv('IB_PORT', 7497))
IB_CLIENT_ID = int(os.getenv('IB_CLIENT_ID', 22))

user_command_messages = {}


def round_down_to_closest_10(n):
    return math.floor(n * 10.0) / 10

def round_up_to_closest_10(n):
    return math.ceil(n * 10.0) / 10

def calculate_days_to_expiry(exp_str):
    """Calculate days to expiry from YYYYMMDD string"""
    exp_date = datetime.datetime.strptime(str(exp_str), '%Y%m%d')
    today = datetime.datetime.now()
    return (exp_date - today).days

def calculate_forward_price(spot, days_to_expiry, rate=0.045):
    """Calculate forward price: F = S * e^(r*T)"""
    T = days_to_expiry / 365.0
    return spot * math.exp(rate * T)

def get_atm_strikes_option2(current_prices):
    """
    Option 2: Calculate ATM strikes using forward price
    Returns dict: {(symbol, expiration): {'spot': x, 'forward': y, 'atm_strike': z, 'days': d}}
    """
    atm_data = {}
    rate = 0.045  # ~4.5% risk-free rate

    for symbol in symbols:
        spot = current_prices.get(symbol)
        if not spot:
            continue

        for exp in expiration_dates[symbol]:
            days = calculate_days_to_expiry(exp)
            forward = calculate_forward_price(spot, days, rate)
            atm_strike = round(forward)  # Round to nearest integer strike

            atm_data[(symbol, exp)] = {
                'spot': spot,
                'forward': forward,
                'atm_strike': atm_strike,
                'days': days
            }

    return atm_data

# Telegram bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
chat_id = TELEGRAM_CHAT_ID

# ETF options: GLD (Gold), SLV (Silver), SPY (S&P 500)
symbols = [
    # 'GLD',
    'SLV',
    # 'SPY',
]

# Strike range offsets (will be applied to current price)
strike_offsets = {
    'GLD': 100,  # ±100 from current price
    'SLV': 25,   # ±25 from current price
    'SPY': 50,   # ±50 from current price
}

# Expiration dates - 3 tenors
expiration_dates = {
    'GLD': ['20260220', '20260320', '20260417'],  # Feb 20, Mar 20, Apr 17 2026
    'SLV': ['20260220', '20260320', '20260417'],  # Feb 20, Mar 20, Apr 17 2026
    'SPY': ['20260220', '20260320', '20260417'],  # Feb 20, Mar 20, Apr 17 2026
}

# These will be populated dynamically after fetching current prices
strike_ranges = {}
contracts = []


class PriceFetcher(EWrapper, EClient):
    """Helper class to fetch current stock prices before building option contracts"""
    def __init__(self):
        EClient.__init__(self, self)
        self.prices = {}
        self.price_event = threading.Event()
        self.reqId_to_symbol = {}  # Map reqId -> symbol

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158]:
            print(f"PriceFetcher Error {reqId} {errorCode}: {errorString}")

    def tickPrice(self, reqId, tickType, price, attrib):
        # tickType 4 = LAST price
        if tickType == 4 and price > 0:
            symbol = self.reqId_to_symbol.get(reqId)
            if symbol:
                self.prices[symbol] = price
                print(f"Got price for {symbol}: ${price:.2f}")
                if len(self.prices) == len(self.reqId_to_symbol):
                    self.price_event.set()

    def nextValidId(self, orderId):
        pass


def fetch_current_prices(host, port, client_id):
    """Fetch current prices for ETFs"""
    print("Fetching current ETF prices...")

    fetcher = PriceFetcher()
    fetcher.connect(host, port, clientId=client_id)

    api_thread = threading.Thread(target=fetcher.run, daemon=True)
    api_thread.start()
    time.sleep(2)  # Wait for connection

    # Request market data for each symbol
    for i, symbol in enumerate(symbols):
        fetcher.reqId_to_symbol[i] = symbol  # Map reqId to symbol
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        fetcher.reqMktData(i, contract, "", False, False, [])

    # Wait for prices (max 10 seconds)
    fetcher.price_event.wait(timeout=10)
    fetcher.disconnect()
    time.sleep(1)

    return fetcher.prices


def build_strike_ranges(prices):
    """Build strike ranges based on FORWARD prices (longest tenor)"""
    global strike_ranges

    for symbol in symbols:
        if symbol in prices:
            spot_price = prices[symbol]
            offset = strike_offsets[symbol]

            # Use longest tenor (Apr 17) for forward calculation
            longest_exp = expiration_dates[symbol][-1]  # Last expiration
            days_to_exp = calculate_days_to_expiry(longest_exp)
            forward_price = calculate_forward_price(spot_price, days_to_exp)

            # Round forward to integer for strike center
            mid = int(round(forward_price))

            # Build range: forward mid ± offset
            strike_ranges[symbol] = np.arange(mid - offset, mid + offset + 1, 1)
            print(f"{symbol}: spot=${spot_price:.2f}, fwd=${forward_price:.2f} ({days_to_exp}d), range={mid-offset}-{mid+offset} ({len(strike_ranges[symbol])} strikes)")
        else:
            # Fallback if price fetch failed - should not happen
            print(f"ERROR: Could not get price for {symbol}!")
            raise Exception(f"Failed to fetch price for {symbol}. Check TWS connection.")


def build_contracts():
    """Build option contracts list after strike ranges are set"""
    global contracts
    contracts = []

    for symbol in symbols:
        for expiration_date in expiration_dates[symbol]:
            for strike in strike_ranges[symbol]:
                for right in ['C', 'P']:
                    contract = Contract()
                    contract.symbol = symbol
                    contract.secType = "OPT"
                    contract.exchange = "SMART"
                    contract.currency = "USD"
                    contract.lastTradeDateOrContractMonth = expiration_date
                    contract.strike = float(strike)
                    contract.right = right
                    contracts.append(contract)

def seconds_to_now(d):
    return (datetime.datetime.now() - d).total_seconds()

def is_smile(x):
    prev_dec = True
    for i in range(1, len(x)):
        if x[i-1] > x[i]:
            if not prev_dec and x[i-1] - x[i] > 0.01:
                return False
        else:
            prev_dec = False
    return True


class IBApp(EWrapper, EClient):
    def __init__(self, contracts_list, strike_ranges_dict):
        EClient.__init__(self, self)
        self.contracts = contracts_list
        self.strike_ranges = strike_ranges_dict
        self.contract_details = {}
        self.connected_event = threading.Event()

        # Store bid/ask IV per contract
        self.volatilities = {
            c.symbol + "_" + c.right + "_" + c.lastTradeDateOrContractMonth: {
                strike: {'bid_vol': None, 'ask_vol': None}
                for strike in self.strike_ranges[c.symbol]
            }
            for c in self.contracts
        }

        self.last_update = dict()

    def nextValidId(self, orderId):
        """Called when connection is ready"""
        super().nextValidId(orderId)
        print(f"Connected! Next valid order ID: {orderId}")
        self.connected_event.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handle errors - filter out info messages"""
        if errorCode in [2104, 2106, 2158]:  # Market data farm messages (info, not errors)
            print(f"Info: {errorString}")
        elif errorCode == 200:
            print(f"Contract error {reqId}: {errorString}")
        elif errorCode == 10091:
            print(f"Subscription needed {reqId}: {errorString}")
        else:
            print(f"Error {reqId} {errorCode}: {errorString}")

    def historicalData(self, reqId, bar):
        bot.send_message(chat_id, str(bar))

    def tickGeneric(self, tickerId, field, value):
        super().tickGeneric(tickerId, field, value)

    def marketDataType(self, reqId, marketDataType):
        super().marketDataType(reqId, marketDataType)

    def tickOptionComputation(self, tickerId, field, tickAttrib, impliedVolatility, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        super().tickOptionComputation(tickerId, field, tickAttrib, impliedVolatility, delta, optPrice, pvDividend, gamma, vega, theta, undPrice)

        # Throttle updates: only process every 60 seconds per contract/field
        key = (tickerId, field)
        if key in self.last_update and seconds_to_now(self.last_update[key]) < 60:
            return
        self.last_update[key] = datetime.datetime.now()

        if impliedVolatility is None or undPrice is None:
            return

        contract = self.contract_details.get(tickerId)
        if not contract:
            return

        symbol_key = contract.symbol + "_" + contract.right + "_" + contract.lastTradeDateOrContractMonth
        strike = contract.strike

        # Store IV from IB (in-memory only, no database)
        if field == 10:  # BID
            self.volatilities[symbol_key][strike]['bid_vol'] = impliedVolatility
        elif field == 11:  # ASK
            self.volatilities[symbol_key][strike]['ask_vol'] = impliedVolatility
        else:
            return

        print(f"Updated: {contract.symbol} {contract.right} {strike} IV={impliedVolatility:.4f}" if impliedVolatility else "")


if __name__ == '__main__':
    print("=" * 50)
    print("IB Options Monitor Bot - GLD/SLV/SPY ETF Options")
    print("=" * 50)

    # Step 1: Fetch current prices for ETFs
    prices = fetch_current_prices(IB_HOST, IB_PORT, IB_CLIENT_ID)

    # Step 2: Build strike ranges based on forward prices
    build_strike_ranges(prices)

    # Step 3: Build option contracts
    build_contracts()

    print(f"\nMonitoring {len(contracts)} option contracts")
    print(f"Symbols: {symbols}")
    for sym in symbols:
        print(f"  {sym}: {len(strike_ranges[sym])} strikes x {len(expiration_dates[sym])} expirations")
    print(f"\nConnecting to {IB_HOST}:{IB_PORT}...")

    # Create app and connect (use different client ID to avoid conflict)
    app = IBApp(contracts, strike_ranges)
    app.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID + 1)

    # Start API thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Wait for connection to be ready
    print("Waiting for connection...")
    if not app.connected_event.wait(timeout=10):
        print("ERROR: Connection timeout! Check TWS is running and API is enabled.")
        exit(1)

    print("Connection established!")
    time.sleep(1)

    # Request market data type:
    # 1 = REALTIME (requires subscription)
    # 3 = DELAYED (15-min delay, free)
    # 4 = FROZEN (last available)
    app.reqMarketDataType(1)  # REALTIME
    time.sleep(1)

    print("Subscribing to market data...")
    for i, contract in enumerate(app.contracts):
        app.reqMktData(i, contract, "232", False, False, [])
        app.contract_details[i] = contract

    print(f"Subscribed to {len(app.contracts)} contracts")
    print("Starting Telegram bot...")

    # Telegram command handlers
    @bot.message_handler(commands=['ivc'])
    def send_underlying_menu_calls(message):
        user_command_messages[message.chat.id] = message.text
        # Show symbol selection first
        markup = types.InlineKeyboardMarkup()
        for sym in symbols:
            markup.add(types.InlineKeyboardButton(sym, callback_data=f'SELECT_C_{sym}'))
        bot.send_message(message.chat.id, "Select underlying (Calls):", reply_markup=markup)

    @bot.message_handler(commands=['ivp'])
    def send_underlying_menu_puts(message):
        user_command_messages[message.chat.id] = message.text
        # Show symbol selection first
        markup = types.InlineKeyboardMarkup()
        for sym in symbols:
            markup.add(types.InlineKeyboardButton(sym, callback_data=f'SELECT_P_{sym}'))
        bot.send_message(message.chat.id, "Select underlying (Puts):", reply_markup=markup)

    @bot.message_handler(commands=['status'])
    def send_status(message):
        status_text = f"Bot is running!\nMonitoring: {', '.join(symbols)}\nContracts: {len(contracts)}"
        bot.send_message(message.chat.id, status_text)

    @bot.message_handler(commands=['menu', 'help', 'start'])
    def send_menu(message):
        """Show all available commands"""
        menu_text = """📋 *IB Options Monitor Bot*
_GLD/SLV/SPY ETF Options_

*Commands:*

📊 *Market Data*
/ivc - IV smile for Calls (select underlying & expiry)
/ivp - IV smile for Puts (select underlying & expiry)
/atm - Show ATM (50Δ) strikes for all tenors

ℹ️ *Info*
/status - Bot status & contract count
/menu - Show this menu

*Monitored Underlyings:*
• GLD (Gold ETF) - ±100 strikes
• SLV (Silver ETF) - ±25 strikes
• SPY (S&P 500 ETF) - ±50 strikes

*Expirations:* Feb 20, Mar 20, Apr 17
"""
        bot.send_message(message.chat.id, menu_text, parse_mode='Markdown')

    @bot.message_handler(commands=['atm'])
    def send_atm_strikes(message):
        """Show ATM (50 delta) strikes using forward price calculation (Option 2)"""
        try:
            # Fetch current prices (use different client_id to avoid conflict)
            current_prices = fetch_current_prices(IB_HOST, IB_PORT, IB_CLIENT_ID + 100)

            if not current_prices:
                bot.send_message(message.chat.id, "Could not fetch current prices. Make sure IB Gateway is connected.")
                return

            # Calculate ATM strikes using forward price
            atm_data = get_atm_strikes_option2(current_prices)

            response = "📊 *ATM Strikes (Forward Price)*\n"
            response += f"_Rate: 4.5%_\n\n"

            for symbol in symbols:
                spot = current_prices.get(symbol)
                if not spot:
                    response += f"*{symbol}*: No price data\n\n"
                    continue

                response += f"*{symbol}* (Spot: ${spot:.2f})\n"

                for exp in expiration_dates[symbol]:
                    data = atm_data.get((symbol, exp))
                    if data:
                        # Format expiration nicely (20260220 -> Feb 20)
                        exp_date = datetime.datetime.strptime(str(exp), '%Y%m%d')
                        exp_str = exp_date.strftime('%b %d')

                        response += f"  {exp_str}: ATM={data['atm_strike']} (Fwd=${data['forward']:.2f}, DTE={data['days']})\n"

                response += "\n"

            bot.send_message(message.chat.id, response, parse_mode='Markdown')

        except Exception as e:
            bot.send_message(message.chat.id, f"Error calculating ATM strikes: {str(e)}")

    @bot.callback_query_handler(func=lambda call: True)
    def handler(call):
        bot.answer_callback_query(call.id)
        argument = call.data

        # Handle symbol selection (SELECT_C_GLD or SELECT_P_SLV)
        if argument.startswith('SELECT_'):
            parts = argument.split('_')
            option_type = parts[1]  # C or P
            symbol = parts[2]       # GLD, SLV, SPY

            markup = types.InlineKeyboardMarkup()
            for exp in expiration_dates[symbol]:
                markup.add(types.InlineKeyboardButton(str(exp), callback_data=f'{symbol}_{option_type}_{exp}'))
            type_name = "Calls" if option_type == 'C' else "Puts"
            bot.send_message(call.message.chat.id, f"Select expiration ({symbol} {type_name}):", reply_markup=markup)
            return

        # Handle expiration selection (GLD_C_20260220)
        # Get current IV data (no historical data without database)
        for symbol_key, data in app.volatilities.items():
            if symbol_key == argument:
                strks = list(data.keys())
                bidvol = np.array([data[s]['bid_vol'] for s in strks])
                askvol = np.array([data[s]['ask_vol'] for s in strks])
                break
        else:
            bot.send_message(call.message.chat.id, "No data found for this selection.")
            return

        # Filter out None values
        valid_bids = [v for v in bidvol if v is not None]
        valid_asks = [v for v in askvol if v is not None]

        all_valid = valid_bids + valid_asks
        if not all_valid:
            bot.send_message(call.message.chat.id, "No data available yet. Please wait for market data.")
            return

        min_value = round_down_to_closest_10(min(all_valid))
        max_value = round_up_to_closest_10(max(all_valid))

        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 6))
        plt.cla()
        plt.scatter(strks, bidvol, label='Bid IV', marker='o')
        plt.scatter(strks, askvol, label='Ask IV', marker='x')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Strike')
        plt.ylabel('Implied Volatility')
        plt.ylim([min_value, max_value])
        plt.title(f'IV Smile: {argument}')
        plt.savefig('vol.png', dpi=100, bbox_inches='tight')
        plt.close()

        bot.send_photo(chat_id, open('vol.png', 'rb'))

    # Start Telegram polling in separate thread
    polling_thread = threading.Thread(target=bot.polling, daemon=True)
    polling_thread.start()

    print("Bot is running! Use /ivc, /ivp, /atm, /status, /menu in Telegram")
    print("Press Ctrl+C to stop")

    # Anomaly detection loop
    last_anom = dict()
    while True:
        time.sleep(30)

        for name, vols in app.volatilities.items():
            strks = list(sorted(list(vols.keys())))
            mid_vols = []
            real_strks = []

            for strk in strks:
                if vols[strk]['bid_vol'] is None or vols[strk]['ask_vol'] is None:
                    continue
                mid_vols.append((vols[strk]['bid_vol'] + vols[strk]['ask_vol']) / 2)
                real_strks.append(strk)

            if mid_vols and not is_smile(mid_vols):
                if name not in last_anom or seconds_to_now(last_anom[name]) > 60 * 30:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(real_strks, mid_vols)
                    plt.grid(True)
                    plt.xlabel('Strike')
                    plt.ylabel('Mid IV')
                    plt.ylim([round_down_to_closest_10(min(mid_vols)), round_up_to_closest_10(max(mid_vols))])
                    plt.title(f'Anomaly in {name}')
                    plt.savefig('vol.png', dpi=100, bbox_inches='tight')
                    plt.close()

                    bot.send_photo(chat_id, open('vol.png', 'rb'))
                    bot.send_message(chat_id, f'Anomaly detected in {name}')
                    last_anom[name] = datetime.datetime.now()
