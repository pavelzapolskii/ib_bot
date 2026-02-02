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


def find_anomalies_spread_aware(vol_data):
    """
    Find anomaly points in the IV smile with spread-awareness.

    Filters:
    1. Only consider strikes with spread < 5% (in IV terms)
    2. Anomaly spike must be > 1.5x the local spread
    3. Arbitrage detection: bid at K > ask at K±1

    Returns list of (strike_idx, reason, signal_strength) tuples.
    signal_strength: 'STRONG' for arbitrage, 'NORMAL' for shape break
    """
    anomalies = []

    # Extract data
    n = len(vol_data)
    if n < 3:
        return anomalies

    # Pre-calculate spreads for all strikes
    spreads = []
    for d in vol_data:
        bid_vol = d['bid_vol']
        ask_vol = d['ask_vol']
        if bid_vol and ask_vol and bid_vol > 0:
            spread = ask_vol - bid_vol
            spread_pct = spread / ((bid_vol + ask_vol) / 2)  # Spread as % of mid
        else:
            spread = None
            spread_pct = None
        spreads.append({'spread': spread, 'spread_pct': spread_pct})

    mid_vols = [d['mid_vol'] for d in vol_data]

    # Check for arbitrage opportunities (bid at K > ask at K±1)
    for i in range(n):
        bid_vol_i = vol_data[i]['bid_vol']
        spread_i = spreads[i]['spread_pct']

        # Skip if spread is too wide (> 5%)
        if spread_i is None or spread_i > 0.05:
            continue

        # Check against previous strike (K-1)
        if i > 0:
            ask_vol_prev = vol_data[i-1]['ask_vol']
            spread_prev = spreads[i-1]['spread_pct']
            if ask_vol_prev and spread_prev is not None and spread_prev <= 0.05:
                if bid_vol_i > ask_vol_prev:
                    diff = (bid_vol_i - ask_vol_prev) * 100
                    anomalies.append((i, f"ARBITRAGE: Bid IV > Ask IV at K-1 by {diff:.1f}%", 'STRONG'))

        # Check against next strike (K+1)
        if i < n - 1:
            ask_vol_next = vol_data[i+1]['ask_vol']
            spread_next = spreads[i+1]['spread_pct']
            if ask_vol_next and spread_next is not None and spread_next <= 0.05:
                if bid_vol_i > ask_vol_next:
                    diff = (bid_vol_i - ask_vol_next) * 100
                    anomalies.append((i, f"ARBITRAGE: Bid IV > Ask IV at K+1 by {diff:.1f}%", 'STRONG'))

    # Shape-based anomaly detection with spread threshold
    prev_dec = True
    for i in range(1, n):
        spread_i = spreads[i]['spread']
        spread_prev = spreads[i-1]['spread']

        # Skip if spread data missing or too wide
        if spread_i is None or spread_prev is None:
            continue
        if spreads[i]['spread_pct'] is not None and spreads[i]['spread_pct'] > 0.05:
            continue
        if spreads[i-1]['spread_pct'] is not None and spreads[i-1]['spread_pct'] > 0.05:
            continue

        # Local spread = average of adjacent spreads
        local_spread = (spread_i + spread_prev) / 2
        min_threshold = local_spread * 1.5  # Spike must be > 1.5x local spread

        if mid_vols[i-1] > mid_vols[i]:
            # IV is decreasing
            drop = mid_vols[i-1] - mid_vols[i]
            if not prev_dec and drop > 0.01 and drop > min_threshold:
                # Was increasing, now decreasing significantly - anomaly at i-1
                anomalies.append((i-1, f"IV spike: +{drop*100:.1f}% (>{min_threshold*100:.1f}% threshold)", 'NORMAL'))
        else:
            # IV is increasing
            rise = mid_vols[i] - mid_vols[i-1]
            if prev_dec and rise > 0.01 and rise > min_threshold:
                # Was decreasing, now increasing significantly
                anomalies.append((i, f"IV dip: +{rise*100:.1f}% (>{min_threshold*100:.1f}% threshold)", 'NORMAL'))
            prev_dec = False

    # Deduplicate (same index might have multiple reasons)
    seen_indices = set()
    unique_anomalies = []
    for idx, reason, strength in anomalies:
        if idx not in seen_indices or strength == 'STRONG':
            if idx in seen_indices:
                # Replace with stronger signal
                unique_anomalies = [(i, r, s) for i, r, s in unique_anomalies if i != idx]
            unique_anomalies.append((idx, reason, strength))
            seen_indices.add(idx)

    return unique_anomalies


def calculate_intrinsic_value(strike, underlying_price, option_type):
    """Calculate intrinsic value of an option"""
    if option_type == 'C':
        return max(0, underlying_price - strike)
    else:  # Put
        return max(0, strike - underlying_price)


def calculate_time_value_annualized(option_price, intrinsic_value, underlying_price, days_to_expiry):
    """
    Calculate annualized time value as a percentage.
    Time Value = Option Price - Intrinsic Value
    Annualized = (Time Value / Underlying Price) * (365 / DTE) * 100
    """
    if days_to_expiry <= 0:
        return 0

    time_value = option_price - intrinsic_value
    if time_value < 0:
        time_value = 0

    annualized = (time_value / underlying_price) * (365 / days_to_expiry) * 100
    return annualized


class IBApp(EWrapper, EClient):
    def __init__(self, contracts_list, strike_ranges_dict):
        EClient.__init__(self, self)
        self.contracts = contracts_list
        self.strike_ranges = strike_ranges_dict
        self.contract_details = {}
        self.connected_event = threading.Event()

        # Store bid/ask IV and prices per contract
        self.volatilities = {
            c.symbol + "_" + c.right + "_" + c.lastTradeDateOrContractMonth: {
                strike: {
                    'bid_vol': None, 'ask_vol': None,
                    'bid_price': None, 'ask_price': None,
                    'underlying_price': None
                }
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

        # Store IV and prices from IB (in-memory only, no database)
        if field == 10:  # BID
            self.volatilities[symbol_key][strike]['bid_vol'] = impliedVolatility
            self.volatilities[symbol_key][strike]['bid_price'] = optPrice
            self.volatilities[symbol_key][strike]['underlying_price'] = undPrice
        elif field == 11:  # ASK
            self.volatilities[symbol_key][strike]['ask_vol'] = impliedVolatility
            self.volatilities[symbol_key][strike]['ask_price'] = optPrice
            self.volatilities[symbol_key][strike]['underlying_price'] = undPrice
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
/calc - Calculate best trades (sell put vol / buy call insurance)

ℹ️ *Info*
/status - Bot status & contract count
/menu - Show this menu

*Calc Options:*
• Best Put to Sell Vol - highest IV/spread for OTM puts
• Best Call Insurance - lowest IV/spread for ITM calls

*Anomaly Signals:*
🔥 ARBITRAGE - Bid IV > Ask IV at adjacent strike (strong)
⚠️ Shape Anomaly - IV spike > 1.5x local spread
_Only alerts when spread < 5%_

*Monitored Underlyings:*
• GLD (Gold ETF) - ±100 strikes
• SLV (Silver ETF) - ±25 strikes
• SPY (S&P 500 ETF) - ±50 strikes

*Expirations:* Feb 20, Mar 20, Apr 17
"""
        bot.send_message(message.chat.id, menu_text, parse_mode='Markdown')

    @bot.message_handler(commands=['calc'])
    def send_calc_menu(message):
        """Show calculation options menu"""
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton("📉 Best Put to Sell Vol", callback_data='CALC_SELLPUT'))
        markup.add(types.InlineKeyboardButton("📈 Best Call Insurance", callback_data='CALC_BUYCALL'))
        bot.send_message(message.chat.id, "Select calculation:", reply_markup=markup)

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

        # Handle CALC options
        if argument.startswith('CALC_'):
            calc_type = argument.split('_')[1]

            if calc_type == 'SELLPUT':
                # Best Put to Sell Vol: highest IV/spread for OTM puts
                msg = "📉 *Best Puts to Sell Vol*\n"
                msg += "_Metric: IV / Spread (higher = better)_\n"
                msg += "_Filter: OTM puts only (strike < underlying)_\n\n"

                results = []
                for name, vols in app.volatilities.items():
                    parts = name.split('_')
                    symbol = parts[0]
                    option_type = parts[1]
                    expiration = parts[2]

                    # Only process puts
                    if option_type != 'P':
                        continue

                    dte = calculate_days_to_expiry(expiration)
                    exp_date = datetime.datetime.strptime(str(expiration), '%Y%m%d')
                    exp_str = exp_date.strftime('%b %d')

                    for strike, data in vols.items():
                        bid_vol = data.get('bid_vol')
                        ask_vol = data.get('ask_vol')
                        und_price = data.get('underlying_price')

                        if bid_vol is None or ask_vol is None or und_price is None:
                            continue

                        # OTM put = strike < underlying price
                        if strike >= und_price:
                            continue

                        spread = ask_vol - bid_vol
                        mid_vol = (bid_vol + ask_vol) / 2
                        spread_pct = spread / mid_vol if mid_vol > 0 else 999

                        # Skip wide spreads (> 10%)
                        if spread_pct > 0.10:
                            continue

                        # Metric: IV / spread (higher is better for selling vol)
                        if spread > 0:
                            metric = mid_vol / spread
                        else:
                            metric = 0

                        results.append({
                            'symbol': symbol,
                            'strike': strike,
                            'expiration': exp_str,
                            'dte': dte,
                            'mid_vol': mid_vol,
                            'spread': spread,
                            'spread_pct': spread_pct,
                            'metric': metric,
                            'und_price': und_price
                        })

                # Sort by metric (highest first)
                results.sort(key=lambda x: x['metric'], reverse=True)

                # Show top 5
                for i, r in enumerate(results[:5]):
                    otm_pct = (r['und_price'] - r['strike']) / r['und_price'] * 100
                    msg += f"*{i+1}. {r['symbol']} {r['strike']}P {r['expiration']}*\n"
                    msg += f"   IV: {r['mid_vol']*100:.1f}% | Spread: {r['spread']*100:.1f}% ({r['spread_pct']*100:.0f}%)\n"
                    msg += f"   OTM: {otm_pct:.1f}% | DTE: {r['dte']} | Score: {r['metric']:.1f}\n\n"

                if not results:
                    msg += "_No valid OTM puts with tight spreads found_"

                bot.send_message(call.message.chat.id, msg, parse_mode='Markdown')
                return

            elif calc_type == 'BUYCALL':
                # Best Call Insurance: lowest IV/spread for ITM calls
                msg = "📈 *Best Calls for Insurance*\n"
                msg += "_Metric: IV × Spread (lower = better)_\n"
                msg += "_Filter: ITM calls only (strike < underlying)_\n\n"

                results = []
                for name, vols in app.volatilities.items():
                    parts = name.split('_')
                    symbol = parts[0]
                    option_type = parts[1]
                    expiration = parts[2]

                    # Only process calls
                    if option_type != 'C':
                        continue

                    dte = calculate_days_to_expiry(expiration)
                    exp_date = datetime.datetime.strptime(str(expiration), '%Y%m%d')
                    exp_str = exp_date.strftime('%b %d')

                    for strike, data in vols.items():
                        bid_vol = data.get('bid_vol')
                        ask_vol = data.get('ask_vol')
                        und_price = data.get('underlying_price')

                        if bid_vol is None or ask_vol is None or und_price is None:
                            continue

                        # ITM call = strike < underlying price
                        if strike >= und_price:
                            continue

                        spread = ask_vol - bid_vol
                        mid_vol = (bid_vol + ask_vol) / 2
                        spread_pct = spread / mid_vol if mid_vol > 0 else 999

                        # Skip wide spreads (> 10%)
                        if spread_pct > 0.10:
                            continue

                        # Metric: IV * spread (lower is better for buying insurance)
                        metric = mid_vol * spread

                        results.append({
                            'symbol': symbol,
                            'strike': strike,
                            'expiration': exp_str,
                            'dte': dte,
                            'mid_vol': mid_vol,
                            'spread': spread,
                            'spread_pct': spread_pct,
                            'metric': metric,
                            'und_price': und_price
                        })

                # Sort by metric (lowest first)
                results.sort(key=lambda x: x['metric'])

                # Show top 5
                for i, r in enumerate(results[:5]):
                    itm_pct = (r['und_price'] - r['strike']) / r['und_price'] * 100
                    msg += f"*{i+1}. {r['symbol']} {r['strike']}C {r['expiration']}*\n"
                    msg += f"   IV: {r['mid_vol']*100:.1f}% | Spread: {r['spread']*100:.1f}% ({r['spread_pct']*100:.0f}%)\n"
                    msg += f"   ITM: {itm_pct:.1f}% | DTE: {r['dte']} | Score: {r['metric']*10000:.2f}\n\n"

                if not results:
                    msg += "_No valid ITM calls with tight spreads found_"

                bot.send_message(call.message.chat.id, msg, parse_mode='Markdown')
                return

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
            vol_data = []  # Store full data for anomaly reporting

            for strk in strks:
                if vols[strk]['bid_vol'] is None or vols[strk]['ask_vol'] is None:
                    continue
                mid_vol = (vols[strk]['bid_vol'] + vols[strk]['ask_vol']) / 2
                mid_vols.append(mid_vol)
                real_strks.append(strk)
                vol_data.append({
                    'strike': strk,
                    'bid_vol': vols[strk]['bid_vol'],
                    'ask_vol': vols[strk]['ask_vol'],
                    'mid_vol': mid_vol,
                    'bid_price': vols[strk].get('bid_price'),
                    'ask_price': vols[strk].get('ask_price'),
                    'underlying_price': vols[strk].get('underlying_price')
                })

            # Use spread-aware anomaly detection
            anomalies = find_anomalies_spread_aware(vol_data)

            if anomalies:
                if name not in last_anom or seconds_to_now(last_anom[name]) > 60 * 30:
                    # Parse name to get option type and expiration
                    parts = name.split('_')
                    symbol = parts[0]
                    option_type = parts[1]
                    expiration = parts[2]

                    # Calculate DTE
                    dte = calculate_days_to_expiry(expiration)

                    # Separate strong (arbitrage) from normal anomalies
                    strong_anomalies = [(i, r, s) for i, r, s in anomalies if s == 'STRONG']
                    normal_anomalies = [(i, r, s) for i, r, s in anomalies if s == 'NORMAL']

                    # Create plot with highlighted anomalies
                    plt.figure(figsize=(12, 7))
                    plt.scatter(real_strks, mid_vols, label='Mid IV', color='blue', alpha=0.6)

                    # Highlight normal anomaly points (yellow X)
                    if normal_anomalies:
                        normal_strks = [real_strks[idx] for idx, _, _ in normal_anomalies]
                        normal_vols = [mid_vols[idx] for idx, _, _ in normal_anomalies]
                        plt.scatter(normal_strks, normal_vols, color='orange', s=150, marker='X', label='Shape Anomaly', zorder=5)

                    # Highlight strong anomaly points (red star)
                    if strong_anomalies:
                        strong_strks = [real_strks[idx] for idx, _, _ in strong_anomalies]
                        strong_vols = [mid_vols[idx] for idx, _, _ in strong_anomalies]
                        plt.scatter(strong_strks, strong_vols, color='red', s=200, marker='*', label='ARBITRAGE', zorder=6)

                    # Add annotations for all anomalies
                    for idx, reason, strength in anomalies:
                        color = 'red' if strength == 'STRONG' else 'orange'
                        plt.annotate(f'K={real_strks[idx]}',
                                   (real_strks[idx], mid_vols[idx]),
                                   textcoords="offset points", xytext=(0,10),
                                   ha='center', fontsize=9, color=color, fontweight='bold' if strength == 'STRONG' else 'normal')

                    plt.grid(True)
                    plt.legend()
                    plt.xlabel('Strike')
                    plt.ylabel('Mid IV')
                    plt.ylim([round_down_to_closest_10(min(mid_vols)), round_up_to_closest_10(max(mid_vols))])

                    # Title indicates if there's arbitrage
                    if strong_anomalies:
                        plt.title(f'⚠️ ARBITRAGE in {name} (DTE={dte})')
                    else:
                        plt.title(f'Anomaly in {name} (DTE={dte})')
                    plt.savefig('vol.png', dpi=100, bbox_inches='tight')
                    plt.close()

                    # Build detailed message
                    if strong_anomalies:
                        msg = f"🔥 *ARBITRAGE detected in {name}*\n"
                    else:
                        msg = f"🚨 *Anomaly detected in {name}*\n"
                    msg += f"_DTE: {dte} days_\n\n"

                    for idx, reason, strength in anomalies:
                        data = vol_data[idx]
                        strk = data['strike']
                        bid_vol = data['bid_vol']
                        ask_vol = data['ask_vol']
                        bid_price = data['bid_price']
                        ask_price = data['ask_price']
                        und_price = data['underlying_price']

                        # Calculate spread
                        iv_spread = (ask_vol - bid_vol) * 100
                        iv_spread_pct = (ask_vol - bid_vol) / ((bid_vol + ask_vol) / 2) * 100

                        prefix = "🔥" if strength == 'STRONG' else "⚠️"
                        msg += f"{prefix} *Strike {strk}:* {reason}\n"
                        msg += f"  IV: Bid={bid_vol*100:.1f}% / Ask={ask_vol*100:.1f}% (spread={iv_spread:.1f}%, {iv_spread_pct:.1f}%)\n"

                        # Calculate time value if we have prices
                        if bid_price and ask_price and und_price and dte > 0:
                            intrinsic = calculate_intrinsic_value(strk, und_price, option_type)
                            bid_tv = bid_price - intrinsic
                            ask_tv = ask_price - intrinsic
                            bid_tv_ann = calculate_time_value_annualized(bid_price, intrinsic, und_price, dte)
                            ask_tv_ann = calculate_time_value_annualized(ask_price, intrinsic, und_price, dte)
                            price_spread = ask_price - bid_price
                            msg += f"  Underlying: ${und_price:.2f}\n"
                            msg += f"  Price: Bid=${bid_price:.2f} / Ask=${ask_price:.2f} (spread=${price_spread:.2f})\n"
                            msg += f"  Intrinsic: ${intrinsic:.2f}\n"
                            msg += f"  Time Value: Bid=${bid_tv:.2f} / Ask=${ask_tv:.2f}\n"
                            msg += f"  Time Value (ann): Bid={bid_tv_ann:.1f}% / Ask={ask_tv_ann:.1f}%\n"
                        msg += "\n"

                    bot.send_photo(chat_id, open('vol.png', 'rb'))
                    bot.send_message(chat_id, msg, parse_mode='Markdown')
                    last_anom[name] = datetime.datetime.now()
