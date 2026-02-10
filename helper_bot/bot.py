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
import pytz

# Load environment variables from .env file
load_dotenv()


def get_market_status():
    """
    Check if US stock market is currently open.
    Returns dict with status info.
    NYSE/NASDAQ hours: 9:30 AM - 4:00 PM ET, Mon-Fri (excluding holidays)
    """
    # Major US market holidays for 2025-2026
    holidays = [
        # 2025
        datetime.date(2025, 1, 1),   # New Year's Day
        datetime.date(2025, 1, 20),  # MLK Day
        datetime.date(2025, 2, 17),  # Presidents Day
        datetime.date(2025, 4, 18),  # Good Friday
        datetime.date(2025, 5, 26),  # Memorial Day
        datetime.date(2025, 6, 19),  # Juneteenth
        datetime.date(2025, 7, 4),   # Independence Day
        datetime.date(2025, 9, 1),   # Labor Day
        datetime.date(2025, 11, 27), # Thanksgiving
        datetime.date(2025, 12, 25), # Christmas
        # 2026
        datetime.date(2026, 1, 1),   # New Year's Day
        datetime.date(2026, 1, 19),  # MLK Day
        datetime.date(2026, 2, 16),  # Presidents Day
        datetime.date(2026, 4, 3),   # Good Friday
        datetime.date(2026, 5, 25),  # Memorial Day
        datetime.date(2026, 6, 19),  # Juneteenth
        datetime.date(2026, 7, 3),   # Independence Day (observed)
        datetime.date(2026, 9, 7),   # Labor Day
        datetime.date(2026, 11, 26), # Thanksgiving
        datetime.date(2026, 12, 25), # Christmas
    ]

    et = pytz.timezone('America/New_York')
    now_et = datetime.datetime.now(et)
    today = now_et.date()

    # Check if weekend
    if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
        next_open = today + datetime.timedelta(days=(7 - now_et.weekday()))
        return {
            'is_open': False,
            'reason': 'Weekend',
            'current_time_et': now_et.strftime('%H:%M ET'),
            'next_open': next_open.strftime('%A, %b %d'),
            'message': f"📅 Market closed (Weekend)\n⏰ Current time: {now_et.strftime('%H:%M ET')}\n📆 Opens: Monday {next_open.strftime('%b %d')} at 9:30 AM ET"
        }

    # Check if holiday
    if today in holidays:
        next_open = today + datetime.timedelta(days=1)
        while next_open in holidays or next_open.weekday() >= 5:
            next_open += datetime.timedelta(days=1)
        return {
            'is_open': False,
            'reason': 'Holiday',
            'current_time_et': now_et.strftime('%H:%M ET'),
            'next_open': next_open.strftime('%A, %b %d'),
            'message': f"🎄 Market closed (Holiday)\n⏰ Current time: {now_et.strftime('%H:%M ET')}\n📆 Opens: {next_open.strftime('%A, %b %d')} at 9:30 AM ET"
        }

    # Check trading hours (9:30 AM - 4:00 PM ET)
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if now_et < market_open:
        mins_until = int((market_open - now_et).total_seconds() / 60)
        hours = mins_until // 60
        mins = mins_until % 60
        return {
            'is_open': False,
            'reason': 'Pre-market',
            'current_time_et': now_et.strftime('%H:%M ET'),
            'opens_in': f"{hours}h {mins}m",
            'message': f"🌅 Market not yet open (Pre-market)\n⏰ Current time: {now_et.strftime('%H:%M ET')}\n⏳ Opens in: {hours}h {mins}m (9:30 AM ET)"
        }

    if now_et > market_close:
        next_open = today + datetime.timedelta(days=1)
        while next_open in holidays or next_open.weekday() >= 5:
            next_open += datetime.timedelta(days=1)
        return {
            'is_open': False,
            'reason': 'After-hours',
            'current_time_et': now_et.strftime('%H:%M ET'),
            'next_open': next_open.strftime('%A, %b %d'),
            'message': f"🌙 Market closed (After-hours)\n⏰ Current time: {now_et.strftime('%H:%M ET')}\n📆 Opens: {next_open.strftime('%A, %b %d')} at 9:30 AM ET"
        }

    # Market is open
    mins_until_close = int((market_close - now_et).total_seconds() / 60)
    hours = mins_until_close // 60
    mins = mins_until_close % 60
    return {
        'is_open': True,
        'reason': 'Open',
        'current_time_et': now_et.strftime('%H:%M ET'),
        'closes_in': f"{hours}h {mins}m",
        'message': f"✅ Market is OPEN\n⏰ Current time: {now_et.strftime('%H:%M ET')}\n⏳ Closes in: {hours}h {mins}m (4:00 PM ET)"
    }

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

# Strike step size per symbol (SLV has $0.50 strikes)
strike_steps = {
    'GLD': 1.0,
    'SLV': 0.5,
    'SPY': 1.0,
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

# ============================================
# SELL PUT SCANNER CONFIG
# ============================================
# Semi-liquid S&P 500 stocks for put selling
SELLPUT_SYMBOLS = ['JPM', 'V', 'JNJ', 'PG', 'KO']

# Tenor range: 0.5y to 1.5y (180-545 days)
SELLPUT_MIN_DTE = 180
SELLPUT_MAX_DTE = 545

# ITM filter: Strike >= Spot * 1.10 (10% ITM)
SELLPUT_ITM_THRESHOLD = 1.10


class PriceFetcher(EWrapper, EClient):
    """Helper class to fetch current stock prices before building option contracts"""
    def __init__(self):
        EClient.__init__(self, self)
        self.prices = {}
        self.price_event = threading.Event()
        self.connected_event = threading.Event()
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
        """Called when connection is ready"""
        print(f"PriceFetcher connected, orderId={orderId}")
        self.connected_event.set()


def fetch_current_prices(host, port, client_id):
    """Fetch current prices for ETFs"""
    print(f"Fetching current ETF prices from {host}:{port}...")

    fetcher = PriceFetcher()
    fetcher.connect(host, port, clientId=client_id)

    api_thread = threading.Thread(target=fetcher.run, daemon=True)
    api_thread.start()

    # Wait for connection to be ready
    print("Waiting for PriceFetcher connection...")
    if not fetcher.connected_event.wait(timeout=15):
        print("ERROR: PriceFetcher connection timeout!")
        fetcher.disconnect()
        return {}

    print("PriceFetcher connected!")
    time.sleep(0.5)

    # Request market data for each symbol
    for i, symbol in enumerate(symbols):
        fetcher.reqId_to_symbol[i] = symbol  # Map reqId to symbol
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        fetcher.reqMktData(i, contract, "", False, False, [])

    # Wait for prices (max 15 seconds)
    fetcher.price_event.wait(timeout=15)
    fetcher.disconnect()
    time.sleep(1)

    return fetcher.prices


# ============================================
# SELL PUT SCANNER
# ============================================
class SellPutScanner(EWrapper, EClient):
    """Scan for best deep ITM puts to sell"""
    def __init__(self):
        EClient.__init__(self, self)
        self.connected_event = threading.Event()
        self.prices = {}  # symbol -> price
        self.option_chains = {}  # symbol -> list of expirations
        self.option_data = []  # List of put option data
        self.chain_event = threading.Event()
        self.data_event = threading.Event()
        self.reqId_to_symbol = {}
        self.reqId_to_option = {}
        self.expected_options = 0
        self.received_options = 0

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158]:
            print(f"SellPutScanner Error {reqId} {errorCode}: {errorString}")
        # Count errors as received to avoid hanging
        if errorCode in [200, 10091, 101, 504, 354]:
            if reqId in self.reqId_to_option:
                self.received_options += 1
                if self.received_options >= self.expected_options:
                    self.data_event.set()

    def nextValidId(self, orderId):
        print(f"SellPutScanner connected, orderId={orderId}")
        self.connected_event.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        # tickType 4 = LAST price
        if tickType == 4 and price > 0:
            symbol = self.reqId_to_symbol.get(reqId)
            if symbol:
                self.prices[symbol] = price
                print(f"  {symbol}: ${price:.2f}")

    def securityDefinitionOptionParameter(self, reqId, exchange, underlyingConId,
                                           tradingClass, multiplier, expirations, strikes):
        """Receive option chain parameters"""
        symbol = self.reqId_to_symbol.get(reqId)
        if symbol and exchange == "SMART":
            # Filter expirations to 0.5y - 1.5y range
            valid_exps = []
            for exp in expirations:
                try:
                    exp_date = datetime.datetime.strptime(exp, '%Y%m%d')
                    dte = (exp_date - datetime.datetime.now()).days
                    if SELLPUT_MIN_DTE <= dte <= SELLPUT_MAX_DTE:
                        valid_exps.append(exp)
                except:
                    pass

            if valid_exps:
                self.option_chains[symbol] = {
                    'expirations': sorted(valid_exps),
                    'strikes': sorted(list(strikes)),
                    'multiplier': multiplier,
                    'tradingClass': tradingClass
                }
                print(f"  {symbol}: {len(valid_exps)} valid expirations, {len(strikes)} strikes")

    def securityDefinitionOptionParameterEnd(self, reqId):
        """Option chain request complete"""
        self.chain_event.set()

    def tickOptionComputation(self, reqId, field, tickAttrib, impliedVolatility,
                               delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        """Receive option greeks and IV"""
        if reqId not in self.reqId_to_option:
            return

        opt_info = self.reqId_to_option[reqId]

        # field 10 = BID, 11 = ASK, 12 = LAST
        if field == 10:  # BID - this is what we care about for selling
            if impliedVolatility and impliedVolatility > 0 and optPrice and optPrice > 0:
                opt_info['bid_price'] = optPrice
                opt_info['bid_iv'] = impliedVolatility
                opt_info['delta'] = delta
                opt_info['gamma'] = gamma
                opt_info['theta'] = theta
                opt_info['vega'] = vega
                opt_info['undPrice'] = undPrice
                self.received_options += 1

                if self.received_options >= self.expected_options:
                    self.data_event.set()


def calculate_sellput_score(opt):
    """
    Calculate quality score for selling a put option.
    Higher score = better candidate.

    Factors:
    - Time Value Annualized (higher = more premium)
    - ITM Depth (deeper = safer)
    - IV (higher = more premium)
    - Delta (closer to -1 = safer)
    - Theta (higher absolute = faster decay = good for seller)
    """
    try:
        strike = opt['strike']
        und_price = opt['undPrice']
        bid_price = opt['bid_price']
        bid_iv = opt['bid_iv']
        delta = opt['delta']
        theta = opt['theta']
        dte = opt['dte']

        if not all([strike, und_price, bid_price, bid_iv, delta, dte]):
            return 0

        # Intrinsic value for ITM put: max(0, strike - undPrice)
        intrinsic = max(0, strike - und_price)

        # Time value = bid price - intrinsic
        time_value = bid_price - intrinsic
        if time_value <= 0:
            return 0

        # Time Value Annualized (% of strike per year)
        tv_annualized = (time_value / strike) * (365 / dte) * 100

        # ITM Depth (% above current price)
        itm_depth = (strike - und_price) / und_price * 100

        # Delta safety: want delta close to -1 (safer)
        # |delta| > 0.85 is good
        delta_safety = abs(delta) if delta else 0

        # Theta benefit: higher |theta| = faster decay = good for seller
        # Normalize theta (typically -0.01 to -0.10 for deep ITM)
        theta_benefit = abs(theta) * 100 if theta else 0

        # IV component (higher = more premium)
        iv_component = bid_iv * 100

        # Composite score
        # Weight: TV_ann (40%), ITM_depth (20%), Delta (15%), Theta (15%), IV (10%)
        score = (
            tv_annualized * 0.40 +
            itm_depth * 0.20 +
            delta_safety * 100 * 0.15 +
            theta_benefit * 0.15 +
            iv_component * 0.10
        )

        return score
    except Exception as e:
        print(f"Score calc error: {e}")
        return 0


def scan_sellput_opportunities(host, port, client_id, progress_callback=None):
    """
    Scan for best deep ITM puts to sell.
    Returns list of opportunities sorted by score.
    """
    scanner = SellPutScanner()
    scanner.connect(host, port, clientId=client_id)

    api_thread = threading.Thread(target=scanner.run, daemon=True)
    api_thread.start()

    if not scanner.connected_event.wait(timeout=15):
        print("ERROR: SellPutScanner connection timeout!")
        scanner.disconnect()
        return []

    print("SellPutScanner connected!")
    time.sleep(0.5)

    # Step 1: Get current stock prices
    if progress_callback:
        progress_callback("📊 Fetching stock prices...")
    print("Fetching stock prices...")

    for i, symbol in enumerate(SELLPUT_SYMBOLS):
        scanner.reqId_to_symbol[i] = symbol
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        scanner.reqMktData(i, contract, "", True, False, [])  # Snapshot mode
        time.sleep(0.2)

    time.sleep(3)  # Wait for prices

    if not scanner.prices:
        print("ERROR: No stock prices received!")
        scanner.disconnect()
        return []

    # Step 2: Get option chains
    if progress_callback:
        progress_callback("📋 Fetching option chains...")
    print("Fetching option chains...")

    for i, symbol in enumerate(SELLPUT_SYMBOLS):
        if symbol not in scanner.prices:
            continue

        req_id = 100 + i
        scanner.reqId_to_symbol[req_id] = symbol

        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        scanner.reqSecDefOptParams(req_id, symbol, "", "STK", contract.conId if hasattr(contract, 'conId') else 0)
        time.sleep(0.3)

    # Wait for chains
    scanner.chain_event.wait(timeout=15)
    time.sleep(1)

    if not scanner.option_chains:
        print("ERROR: No option chains received!")
        scanner.disconnect()
        return []

    # Step 3: Request option data for deep ITM puts
    if progress_callback:
        progress_callback("💰 Scanning deep ITM puts...")
    print("Scanning deep ITM puts...")

    req_id = 1000
    options_to_request = []

    for symbol, chain in scanner.option_chains.items():
        spot = scanner.prices.get(symbol, 0)
        if spot <= 0:
            continue

        itm_threshold = spot * SELLPUT_ITM_THRESHOLD

        # Get strikes that are >= 10% ITM
        itm_strikes = [s for s in chain['strikes'] if s >= itm_threshold]

        # Limit to top 5 strikes per symbol to avoid rate limits
        itm_strikes = sorted(itm_strikes)[:5]

        for exp in chain['expirations'][:3]:  # Limit to 3 expirations per symbol
            for strike in itm_strikes:
                options_to_request.append({
                    'symbol': symbol,
                    'expiration': exp,
                    'strike': strike,
                    'spot': spot,
                    'req_id': req_id
                })
                req_id += 1

    print(f"Requesting data for {len(options_to_request)} options...")
    scanner.expected_options = len(options_to_request)

    # Request option market data
    for opt in options_to_request:
        contract = Contract()
        contract.symbol = opt['symbol']
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = opt['expiration']
        contract.strike = float(opt['strike'])
        contract.right = "P"  # Puts only
        contract.multiplier = "100"

        exp_date = datetime.datetime.strptime(opt['expiration'], '%Y%m%d')
        dte = (exp_date - datetime.datetime.now()).days

        scanner.reqId_to_option[opt['req_id']] = {
            'symbol': opt['symbol'],
            'strike': opt['strike'],
            'expiration': opt['expiration'],
            'dte': dte,
            'spot': opt['spot'],
            'bid_price': None,
            'bid_iv': None,
            'delta': None,
            'gamma': None,
            'theta': None,
            'vega': None,
            'undPrice': None
        }

        scanner.reqMktData(opt['req_id'], contract, "232", True, False, [])  # Snapshot with greeks
        time.sleep(0.15)  # Rate limiting

        if (opt['req_id'] - 1000 + 1) % 20 == 0:
            print(f"  Requested {opt['req_id'] - 1000 + 1}/{len(options_to_request)}...")
            time.sleep(1)

    # Wait for data
    if progress_callback:
        progress_callback("⏳ Processing data...")
    scanner.data_event.wait(timeout=60)
    time.sleep(1)

    # Step 4: Calculate scores and rank
    results = []
    for req_id, opt in scanner.reqId_to_option.items():
        if opt['bid_price'] and opt['bid_iv']:
            score = calculate_sellput_score(opt)
            opt['score'] = score
            results.append(opt)

    # Sort by score (highest first)
    results.sort(key=lambda x: x.get('score', 0), reverse=True)

    scanner.disconnect()
    time.sleep(0.5)

    return results


class IVSmileFetcher(EWrapper, EClient):
    """Fetch fresh IV smile data for a specific symbol/expiration"""
    def __init__(self, symbol, expiration, option_type, strikes):
        EClient.__init__(self, self)
        self.symbol = symbol
        self.expiration = expiration
        self.option_type = option_type
        self.strikes = strikes
        self.iv_data = {s: {'bid_vol': None, 'ask_vol': None, 'bid_price': None, 'ask_price': None, 'underlying_price': None} for s in strikes}
        self.received_count = 0
        self.expected_count = len(strikes) * 2  # bid and ask for each strike
        self.done_event = threading.Event()
        self.connected_event = threading.Event()  # Wait for connection
        self.reqId_to_strike = {}
        self.active_req_ids = []  # Track active subscriptions for cleanup

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode not in [2104, 2106, 2158]:
            print(f"IVSmileFetcher Error {reqId} {errorCode}: {errorString}")
        # If we get an error for a contract, count it as received to avoid hanging
        if errorCode in [200, 10091, 101, 504]:  # No security / subscription needed / max tickers / not connected
            self.received_count += 2  # Count both bid and ask as done
            if self.received_count >= self.expected_count:
                self.done_event.set()

    def nextValidId(self, orderId):
        """Called when connection is ready"""
        print(f"IVSmileFetcher connected, orderId={orderId}")
        self.connected_event.set()

    def tickOptionComputation(self, tickerId, field, tickAttrib, impliedVolatility, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        strike = self.reqId_to_strike.get(tickerId)
        if strike is None:
            return

        if impliedVolatility is not None and impliedVolatility > 0:
            if field == 10:  # BID
                self.iv_data[strike]['bid_vol'] = impliedVolatility
                self.iv_data[strike]['bid_price'] = optPrice
                self.iv_data[strike]['underlying_price'] = undPrice
                self.received_count += 1
            elif field == 11:  # ASK
                self.iv_data[strike]['ask_vol'] = impliedVolatility
                self.iv_data[strike]['ask_price'] = optPrice
                self.iv_data[strike]['underlying_price'] = undPrice
                self.received_count += 1

        if self.received_count >= self.expected_count:
            self.done_event.set()

    def cancel_all_market_data(self):
        """Cancel all active market data subscriptions"""
        for req_id in self.active_req_ids:
            try:
                self.cancelMktData(req_id)
            except:
                pass
        self.active_req_ids = []


def fetch_iv_smile(host, port, client_id, symbol, expiration, option_type, strikes):
    """Fetch fresh IV smile for a specific symbol/expiration using snapshot requests"""
    print(f"Fetching fresh IV smile for {symbol} {option_type} {expiration} ({len(strikes)} strikes)...")

    fetcher = IVSmileFetcher(symbol, expiration, option_type, strikes)
    fetcher.connect(host, port, clientId=client_id)

    api_thread = threading.Thread(target=fetcher.run, daemon=True)
    api_thread.start()

    # Wait for connection to be ready (max 10 seconds)
    print("Waiting for IVSmileFetcher connection...")
    if not fetcher.connected_event.wait(timeout=10):
        print("ERROR: IVSmileFetcher connection timeout!")
        fetcher.disconnect()
        raise Exception("Failed to connect to IB for fresh IV data")

    print("IVSmileFetcher connected!")
    time.sleep(0.5)

    # Request market data type (realtime)
    fetcher.reqMarketDataType(1)
    time.sleep(0.5)

    # Request market data for each strike using SNAPSHOT mode (no subscription)
    # Snapshot=True means IB sends data once and auto-cancels (no ticker limit issue)
    for i, strike in enumerate(strikes):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "OPT"
        contract.exchange = "SMART"
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = expiration
        contract.strike = float(strike)
        contract.right = option_type

        fetcher.reqId_to_strike[i] = strike
        fetcher.active_req_ids.append(i)
        # Use snapshot=True to avoid holding ticker slots
        fetcher.reqMktData(i, contract, "232", True, False, [])

        # Rate limiting - slower to avoid overwhelming IB
        if (i + 1) % 10 == 0:
            time.sleep(3.0)  # Longer pause every 10 requests
        else:
            time.sleep(0.25)  # 250ms between each request

    # Wait for data (max 90 seconds for larger strike ranges)
    fetcher.done_event.wait(timeout=90)

    # Cancel any remaining subscriptions (in case snapshot didn't work)
    fetcher.cancel_all_market_data()
    time.sleep(1)

    fetcher.disconnect()
    time.sleep(1)

    return fetcher.iv_data


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

            # Round forward to nearest strike step
            step = strike_steps.get(symbol, 1.0)
            mid = round(forward_price / step) * step

            # Build range: forward mid ± offset
            strike_ranges[symbol] = np.arange(mid - offset, mid + offset + step, step)
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


def generate_iv_smile_plot(app, symbol_key, filename='iv_smile.png'):
    """
    Generate IV smile plot (bid/ask) for a given symbol key.
    Returns True if successful, False otherwise.
    """
    if symbol_key not in app.volatilities:
        return False

    data = app.volatilities[symbol_key]
    strks = list(sorted(data.keys()))
    bidvol = np.array([data[s]['bid_vol'] for s in strks])
    askvol = np.array([data[s]['ask_vol'] for s in strks])

    # Filter out None values for y-limits
    valid_bids = [v for v in bidvol if v is not None]
    valid_asks = [v for v in askvol if v is not None]

    all_valid = valid_bids + valid_asks
    if not all_valid:
        return False

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
    plt.title(f'IV Smile: {symbol_key}')
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

    return True


class IBApp(EWrapper, EClient):
    def __init__(self, contracts_list, strike_ranges_dict):
        EClient.__init__(self, self)
        self.contracts = contracts_list
        self.strike_ranges = strike_ranges_dict
        self.contract_details = {}
        self.connected_event = threading.Event()
        self.subscriptions_paused = False
        self.subscription_lock = threading.Lock()

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
        self.tick_count = 0

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
        elif errorCode in [1100, 1101, 1102, 2110]:
            # Connection-related errors
            # 1100: Connectivity lost
            # 1101: Connectivity restored (receiving historical data)
            # 1102: Connectivity restored (all data)
            # 2110: Connectivity between IB and TWS lost
            print(f"Connection issue {errorCode}: {errorString}")
            try:
                bot.send_message(chat_id, f"⚠️ *IB Connection Issue*\nCode: {errorCode}\n{errorString}", parse_mode='Markdown')
            except:
                pass
        else:
            print(f"Error {reqId} {errorCode}: {errorString}")

    def connectionClosed(self):
        """Called when connection to TWS/Gateway is closed"""
        print("Connection closed!")
        try:
            bot.send_message(chat_id, "🔴 *Bot Disconnected*\nIB connection has been closed.", parse_mode='Markdown')
        except:
            pass

    def pause_subscriptions(self):
        """Cancel all market data subscriptions to free up ticker slots"""
        with self.subscription_lock:
            if self.subscriptions_paused:
                return
            print("Pausing all market data subscriptions...")
            for i in range(len(self.contracts)):
                try:
                    self.cancelMktData(i)
                except:
                    pass
                if (i + 1) % 50 == 0:
                    time.sleep(0.5)
            self.subscriptions_paused = True
            time.sleep(2)  # Give IB time to process cancellations
            print("All subscriptions paused.")

    def resume_subscriptions(self):
        """Re-subscribe to all market data"""
        with self.subscription_lock:
            if not self.subscriptions_paused:
                return
            print("Resuming market data subscriptions...")
            for i, contract in enumerate(self.contracts):
                try:
                    self.reqMktData(i, contract, "232", False, False, [])
                except:
                    pass
                if (i + 1) % 40 == 0:
                    time.sleep(2)
            self.subscriptions_paused = False
            print("All subscriptions resumed.")

    def fetch_fresh_iv_smile(self, symbol, expiration, option_type, strikes):
        """Fetch fresh IV smile using THIS connection (after pausing main subscriptions)"""
        print(f"Fetching fresh IV for {symbol} {option_type} {expiration} ({len(strikes)} strikes)...")

        # Check if connected
        if not self.isConnected():
            print("ERROR: Not connected to IB! Cannot fetch fresh IV.")
            return {s: {'bid_vol': None, 'ask_vol': None, 'bid_price': None, 'ask_price': None, 'underlying_price': None} for s in strikes}

        # Cancel any leftover subscriptions from previous fetch
        base_req_id = 10000
        if hasattr(self, '_last_fresh_iv_count') and self._last_fresh_iv_count > 0:
            print(f"Cancelling {self._last_fresh_iv_count} previous fresh IV subscriptions...")
            for i in range(self._last_fresh_iv_count):
                try:
                    self.cancelMktData(base_req_id + i)
                except:
                    pass
            time.sleep(1.0)  # Wait for cancellations to process

        # Use high request IDs to avoid collision with main subscriptions (0-999)
        fresh_iv_data = {s: {'bid_vol': None, 'ask_vol': None, 'bid_price': None, 'ask_price': None, 'underlying_price': None} for s in strikes}
        self.fresh_iv_strikes = {}  # Map reqId -> strike
        self.fresh_iv_data = fresh_iv_data
        self.fresh_iv_received = 0
        self.fresh_iv_expected = len(strikes) * 2
        self.fresh_iv_done = threading.Event()
        self._last_fresh_iv_count = len(strikes)  # Track for next cleanup

        # Request IV data for each strike
        for i, strike in enumerate(strikes):
            req_id = base_req_id + i
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "OPT"
            contract.exchange = "SMART"
            contract.currency = "USD"
            contract.lastTradeDateOrContractMonth = expiration
            contract.strike = float(strike)
            contract.right = option_type

            self.fresh_iv_strikes[req_id] = strike
            self.reqMktData(req_id, contract, "232", False, False, [])

            # Rate limiting - more conservative to avoid "Max tickers" error
            if (i + 1) % 20 == 0:
                print(f"  Requested {i + 1}/{len(strikes)} strikes...")
                time.sleep(2.0)  # Longer pause every 20 requests
            else:
                time.sleep(0.15)  # 150ms between each request

        print(f"All {len(strikes)} requests sent, waiting for data...")

        # Wait for data (max 60 seconds)
        got_data = self.fresh_iv_done.wait(timeout=60)

        # Cancel all fresh IV subscriptions
        for i in range(len(strikes)):
            req_id = base_req_id + i
            try:
                self.cancelMktData(req_id)
            except:
                pass

        time.sleep(0.5)

        valid_count = sum(1 for s in strikes if fresh_iv_data[s]['bid_vol'] is not None or fresh_iv_data[s]['ask_vol'] is not None)
        print(f"Fresh IV fetch: {self.fresh_iv_received} ticks, {valid_count}/{len(strikes)} strikes with data")

        # Cleanup
        self.fresh_iv_strikes = {}

        return fresh_iv_data

    def historicalData(self, reqId, bar):
        bot.send_message(chat_id, str(bar))

    def tickGeneric(self, tickerId, field, value):
        super().tickGeneric(tickerId, field, value)

    def marketDataType(self, reqId, marketDataType):
        super().marketDataType(reqId, marketDataType)

    def tickOptionComputation(self, tickerId, field, tickAttrib, impliedVolatility, delta, optPrice, pvDividend, gamma, vega, theta, undPrice):
        super().tickOptionComputation(tickerId, field, tickAttrib, impliedVolatility, delta, optPrice, pvDividend, gamma, vega, theta, undPrice)
        self.tick_count += 1

        # Handle fresh IV fetch requests (reqId >= 10000)
        if tickerId >= 10000 and hasattr(self, 'fresh_iv_strikes') and tickerId in self.fresh_iv_strikes:
            strike = self.fresh_iv_strikes[tickerId]
            if impliedVolatility is not None and impliedVolatility > 0:
                if field == 10:  # BID
                    self.fresh_iv_data[strike]['bid_vol'] = impliedVolatility
                    self.fresh_iv_data[strike]['bid_price'] = optPrice
                    self.fresh_iv_data[strike]['underlying_price'] = undPrice
                    self.fresh_iv_received += 1
                elif field == 11:  # ASK
                    self.fresh_iv_data[strike]['ask_vol'] = impliedVolatility
                    self.fresh_iv_data[strike]['ask_price'] = optPrice
                    self.fresh_iv_data[strike]['underlying_price'] = undPrice
                    self.fresh_iv_received += 1

                # Print progress
                if self.fresh_iv_received <= 3 or self.fresh_iv_received % 20 == 0:
                    print(f"  Fresh IV: {self.fresh_iv_received}/{self.fresh_iv_expected} (strike={strike}, field={field})")

                if self.fresh_iv_received >= self.fresh_iv_expected:
                    self.fresh_iv_done.set()
            return

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

    try:
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
        BATCH_SIZE = 40  # Number of requests before pausing
        BATCH_SLEEP = 2  # Seconds to sleep between batches

        for i, contract in enumerate(app.contracts):
            app.reqMktData(i, contract, "232", False, False, [])
            app.contract_details[i] = contract

            # Sleep after every BATCH_SIZE requests to avoid overwhelming IB
            if (i + 1) % BATCH_SIZE == 0:
                print(f"  Subscribed to {i + 1}/{len(app.contracts)} contracts, pausing...")
                time.sleep(BATCH_SLEEP)

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
            connected = "✅ Connected" if app.isConnected() else "❌ Disconnected"
            market = get_market_status()
            market_icon = "🟢" if market['is_open'] else "🔴"
            status_text = f"""📊 *Bot Status*
IB Connection: {connected}
Market: {market_icon} {market['status']}
Monitoring: {', '.join(symbols)}
Contracts: {len(contracts)}
Ticks received: {app.tick_count if hasattr(app, 'tick_count') else 'N/A'}"""
            bot.send_message(message.chat.id, status_text, parse_mode='Markdown')

        @bot.message_handler(commands=['market', 'market_status'])
        def send_market_status(message):
            """Check if US stock market is currently open"""
            status = get_market_status()
            bot.send_message(message.chat.id, status['message'], parse_mode='Markdown')

        @bot.message_handler(commands=['menu', 'help', 'start'])
        def send_menu(message):
            """Show all available commands"""
            menu_text = """📋 *IB Options Monitor Bot*
    _Options Helper Bot_

    *Commands:*

    📊 *ETF IV Data*
    /ivc - IV smile for Calls
    /ivp - IV smile for Puts
    /atm - Show ATM (50Δ) strikes
    /calc - Calculate best trades

    💰 *Stock Put Scanner*
    /sellput - Find best deep ITM puts to sell
    _(JPM, V, JNJ, PG, KO | 0.5-1.5y tenor | ≥10% ITM)_

    ℹ️ *Info*
    /status - Bot status
    /market - Check if market is open
    /menu - Show this menu

    *Sell Put Score:*
    • Time Value (ann) × 40%
    • ITM Depth × 20%
    • Delta safety × 15%
    • Theta (decay) × 15%
    • IV × 10%
    """
            bot.send_message(message.chat.id, menu_text, parse_mode='Markdown')

        @bot.message_handler(commands=['calc'])
        def send_calc_menu(message):
            """Show asset selection for calculations"""
            markup = types.InlineKeyboardMarkup()
            markup.add(types.InlineKeyboardButton("🥇 GLD (Gold)", callback_data='CALC_ASSET_GLD'))
            markup.add(types.InlineKeyboardButton("🥈 SLV (Silver)", callback_data='CALC_ASSET_SLV'))
            markup.add(types.InlineKeyboardButton("📊 All Assets", callback_data='CALC_ASSET_ALL'))
            bot.send_message(message.chat.id, "Select asset for calculation:", reply_markup=markup)

        @bot.message_handler(commands=['sellput'])
        def send_sellput_scan(message):
            """Scan for best deep ITM puts to sell"""
            # Check market status first
            market = get_market_status()
            if not market['is_open']:
                bot.send_message(message.chat.id,
                    f"⚠️ Market is closed ({market['status']})\n"
                    "Scan works best during market hours for accurate pricing.",
                    parse_mode='Markdown')

            status_msg = bot.send_message(message.chat.id,
                f"🔍 *Scanning best puts to sell...*\n"
                f"Symbols: {', '.join(SELLPUT_SYMBOLS)}\n"
                f"Tenor: {SELLPUT_MIN_DTE}-{SELLPUT_MAX_DTE} days\n"
                f"ITM: ≥{int((SELLPUT_ITM_THRESHOLD-1)*100)}%\n\n"
                f"⏳ This may take 30-60 seconds...",
                parse_mode='Markdown')

            def update_status(text):
                try:
                    bot.edit_message_text(
                        f"🔍 *Scanning best puts to sell...*\n{text}",
                        message.chat.id, status_msg.message_id,
                        parse_mode='Markdown')
                except:
                    pass

            try:
                results = scan_sellput_opportunities(
                    IB_HOST, IB_PORT, IB_CLIENT_ID + 200,
                    progress_callback=update_status
                )

                if not results:
                    bot.send_message(message.chat.id,
                        "❌ No valid put options found.\n"
                        "Check IB connection and market status.")
                    return

                # Build response message with top 10
                msg = "📈 *Best Deep ITM Puts to Sell*\n"
                msg += f"_Sorted by composite score_\n"
                msg += f"_ITM ≥{int((SELLPUT_ITM_THRESHOLD-1)*100)}%, DTE {SELLPUT_MIN_DTE}-{SELLPUT_MAX_DTE}d_\n\n"

                for i, opt in enumerate(results[:10]):
                    symbol = opt['symbol']
                    strike = opt['strike']
                    exp = opt['expiration']
                    dte = opt['dte']
                    bid = opt['bid_price']
                    iv = opt['bid_iv'] * 100 if opt['bid_iv'] else 0
                    delta = opt['delta'] if opt['delta'] else 0
                    theta = opt['theta'] if opt['theta'] else 0
                    spot = opt['spot']
                    score = opt.get('score', 0)

                    # Calculate ITM depth
                    itm_pct = (strike - spot) / spot * 100

                    # Calculate time value
                    intrinsic = max(0, strike - spot)
                    time_value = bid - intrinsic if bid else 0
                    tv_ann = (time_value / strike) * (365 / dte) * 100 if dte > 0 else 0

                    # Format expiration
                    exp_date = datetime.datetime.strptime(exp, '%Y%m%d')
                    exp_str = exp_date.strftime('%b %d')

                    msg += f"*{i+1}. {symbol} ${strike}P {exp_str}*\n"
                    msg += f"   Spot: ${spot:.2f} | ITM: +{itm_pct:.1f}%\n"
                    msg += f"   Bid: ${bid:.2f} | IV: {iv:.1f}%\n"
                    msg += f"   TV(ann): {tv_ann:.1f}% | DTE: {dte}d\n"
                    msg += f"   Δ: {delta:.2f} | θ: {theta:.3f}\n"
                    msg += f"   📊 Score: {score:.1f}\n\n"

                msg += "_Higher score = better candidate_\n"
                msg += "_Score = TV×0.4 + ITM×0.2 + Δ×0.15 + θ×0.15 + IV×0.1_"

                bot.send_message(message.chat.id, msg, parse_mode='Markdown')

            except Exception as e:
                print(f"SellPut scan error: {e}")
                bot.send_message(message.chat.id, f"❌ Scan error: {str(e)}")

        @bot.message_handler(commands=['atm'])
        def send_atm_strikes(message):
            """Show ATM (50 delta) strikes using forward price calculation (Option 2)"""
            try:
                # Fetch current prices (use different client_id to avoid conflict)
                current_prices = fetch_current_prices(IB_HOST, IB_PORT, IB_CLIENT_ID + 100)

                if not current_prices:
                    status = get_market_status()
                    msg = "❌ Could not fetch current prices.\n\n"
                    msg += status['message']
                    msg += "\n\n_Use /market to check market hours_"
                    bot.send_message(message.chat.id, msg, parse_mode='Markdown')
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

            # Handle CALC asset selection
            if argument.startswith('CALC_ASSET_'):
                asset = argument.split('_')[2]  # GLD, SLV, or ALL
                markup = types.InlineKeyboardMarkup()
                markup.add(types.InlineKeyboardButton("📉 Best Put to Sell Vol", callback_data=f'CALCTYPE_{asset}_SELLPUT'))
                markup.add(types.InlineKeyboardButton("📈 Best Call Insurance", callback_data=f'CALCTYPE_{asset}_BUYCALL'))
                asset_name = "All Assets" if asset == "ALL" else asset
                bot.send_message(call.message.chat.id, f"Select calculation for *{asset_name}*:", reply_markup=markup, parse_mode='Markdown')
                return

            # Handle CALC type selection
            if argument.startswith('CALCTYPE_'):
                parts = argument.split('_')
                asset_filter = parts[1]  # GLD, SLV, or ALL
                calc_type = parts[2]  # SELLPUT or BUYCALL

                if calc_type == 'SELLPUT':
                    # Best Put to Sell Vol: deep ITM puts (strike > forward), higher strike = better
                    asset_label = "All Assets" if asset_filter == "ALL" else asset_filter
                    msg = f"📉 *Best Puts to Sell Vol ({asset_label})*\n"
                    msg += "_Metric: (IV / Spread) × (1 + ITM%)_\n"
                    msg += "_Filter: Deep ITM puts, higher strike = better_\n\n"

                    results = []
                    for name, vols in app.volatilities.items():
                        name_parts = name.split('_')
                        symbol = name_parts[0]
                        option_type = name_parts[1]
                        expiration = name_parts[2]

                        # Filter by asset if not ALL
                        if asset_filter != 'ALL' and symbol != asset_filter:
                            continue

                        # Only process puts
                        if option_type != 'P':
                            continue

                        dte = calculate_days_to_expiry(expiration)
                        exp_date = datetime.datetime.strptime(str(expiration), '%Y%m%d')
                        exp_str = exp_date.strftime('%b %d')

                        # Calculate forward price for this expiration
                        first_strike_data = next((d for d in vols.values() if d.get('underlying_price')), None)
                        if not first_strike_data:
                            continue
                        spot = first_strike_data.get('underlying_price')
                        forward = calculate_forward_price(spot, dte)

                        for strike, data in vols.items():
                            bid_vol = data.get('bid_vol')
                            ask_vol = data.get('ask_vol')
                            und_price = data.get('underlying_price')
                            bid_price = data.get('bid_price')
                            ask_price = data.get('ask_price')

                            if bid_vol is None or ask_vol is None or und_price is None:
                                continue

                            # ITM put = strike > forward price
                            if strike <= forward:
                                continue

                            spread = ask_vol - bid_vol
                            mid_vol = (bid_vol + ask_vol) / 2
                            spread_pct = spread / mid_vol if mid_vol > 0 else 999

                            # Skip wide spreads (> 10%)
                            if spread_pct > 0.10:
                                continue

                            # Distance from forward (ITM percentage) - higher strike = more ITM for puts
                            itm_pct = (strike - forward) / forward * 100

                            # Metric: (IV / spread) × (1 + ITM%/100)
                            # Deeper ITM (higher strike) = higher bonus
                            if spread > 0:
                                base_metric = mid_vol / spread
                                metric = base_metric * (1 + itm_pct / 100)
                            else:
                                metric = 0

                            # Calculate time value annualized
                            tv_ann_bid = None
                            tv_ann_ask = None
                            if bid_price and ask_price and dte > 0:
                                intrinsic = calculate_intrinsic_value(strike, und_price, 'P')
                                tv_ann_bid = calculate_time_value_annualized(bid_price, intrinsic, und_price, dte)
                                tv_ann_ask = calculate_time_value_annualized(ask_price, intrinsic, und_price, dte)

                            results.append({
                                'symbol': symbol,
                                'strike': strike,
                                'expiration': exp_str,
                                'expiration_raw': expiration,
                                'dte': dte,
                                'mid_vol': mid_vol,
                                'spread': spread,
                                'spread_pct': spread_pct,
                                'metric': metric,
                                'und_price': und_price,
                                'forward': forward,
                                'itm_pct': itm_pct,
                                'bid_price': bid_price,
                                'ask_price': ask_price,
                                'tv_ann_bid': tv_ann_bid,
                                'tv_ann_ask': tv_ann_ask
                            })

                    # Sort by metric (highest first)
                    results.sort(key=lambda x: x['metric'], reverse=True)

                    # Show top 5
                    for i, r in enumerate(results[:5]):
                        msg += f"*{i+1}. {r['symbol']} {r['strike']}P {r['expiration']}*\n"
                        msg += f"   IV: {r['mid_vol']*100:.1f}% | Spread: {r['spread']*100:.1f}% ({r['spread_pct']*100:.0f}%)\n"
                        if r['bid_price'] and r['ask_price']:
                            msg += f"   Price: ${r['bid_price']:.2f} / ${r['ask_price']:.2f}\n"
                        if r['tv_ann_bid'] is not None and r['tv_ann_ask'] is not None:
                            msg += f"   TV(ann): {r['tv_ann_bid']:.1f}% / {r['tv_ann_ask']:.1f}%\n"
                        msg += f"   ITM: {r['itm_pct']:.1f}% (Fwd=${r['forward']:.2f}) | DTE: {r['dte']} | Score: {r['metric']:.1f}\n\n"

                    if not results:
                        msg += "_No valid ITM puts with tight spreads found_"

                    bot.send_message(call.message.chat.id, msg, parse_mode='Markdown')
                    return

                elif calc_type == 'BUYCALL':
                    # Best Call Insurance: deep ITM calls (strike < forward), lower strike = better
                    asset_label = "All Assets" if asset_filter == "ALL" else asset_filter
                    msg = f"📈 *Best Calls for Insurance ({asset_label})*\n"
                    msg += "_Metric: (IV × Spread) / (1 + ITM%)_\n"
                    msg += "_Filter: Deep ITM calls, lower strike = better_\n\n"

                    results = []
                    for name, vols in app.volatilities.items():
                        name_parts = name.split('_')
                        symbol = name_parts[0]
                        option_type = name_parts[1]
                        expiration = name_parts[2]

                        # Filter by asset if not ALL
                        if asset_filter != 'ALL' and symbol != asset_filter:
                            continue

                        # Only process calls
                        if option_type != 'C':
                            continue

                        dte = calculate_days_to_expiry(expiration)
                        exp_date = datetime.datetime.strptime(str(expiration), '%Y%m%d')
                        exp_str = exp_date.strftime('%b %d')

                        # Calculate forward price for this expiration
                        first_strike_data = next((d for d in vols.values() if d.get('underlying_price')), None)
                        if not first_strike_data:
                            continue
                        spot = first_strike_data.get('underlying_price')
                        forward = calculate_forward_price(spot, dte)

                        for strike, data in vols.items():
                            bid_vol = data.get('bid_vol')
                            ask_vol = data.get('ask_vol')
                            und_price = data.get('underlying_price')
                            bid_price = data.get('bid_price')
                            ask_price = data.get('ask_price')

                            if bid_vol is None or ask_vol is None or und_price is None:
                                continue

                            # ITM call = strike < forward price
                            if strike >= forward:
                                continue

                            spread = ask_vol - bid_vol
                            mid_vol = (bid_vol + ask_vol) / 2
                            spread_pct = spread / mid_vol if mid_vol > 0 else 999

                            # Skip wide spreads (> 10%)
                            if spread_pct > 0.10:
                                continue

                            # Distance from forward (ITM percentage) - lower strike = more ITM for calls
                            itm_pct = (forward - strike) / forward * 100

                            # Metric: (IV × spread) / (1 + ITM%/100)
                            # Deeper ITM (lower strike) = lower metric (better - cheaper insurance)
                            base_metric = mid_vol * spread
                            metric = base_metric / (1 + itm_pct / 100)

                            # Calculate time value annualized
                            tv_ann_bid = None
                            tv_ann_ask = None
                            if bid_price and ask_price and dte > 0:
                                intrinsic = calculate_intrinsic_value(strike, und_price, 'C')
                                tv_ann_bid = calculate_time_value_annualized(bid_price, intrinsic, und_price, dte)
                                tv_ann_ask = calculate_time_value_annualized(ask_price, intrinsic, und_price, dte)

                            results.append({
                                'symbol': symbol,
                                'strike': strike,
                                'expiration': exp_str,
                                'expiration_raw': expiration,
                                'dte': dte,
                                'mid_vol': mid_vol,
                                'spread': spread,
                                'spread_pct': spread_pct,
                                'metric': metric,
                                'und_price': und_price,
                                'forward': forward,
                                'itm_pct': itm_pct,
                                'bid_price': bid_price,
                                'ask_price': ask_price,
                                'tv_ann_bid': tv_ann_bid,
                                'tv_ann_ask': tv_ann_ask
                            })

                    # Sort by metric (lowest first)
                    results.sort(key=lambda x: x['metric'])

                    # Show top 5
                    for i, r in enumerate(results[:5]):
                        msg += f"*{i+1}. {r['symbol']} {r['strike']}C {r['expiration']}*\n"
                        msg += f"   IV: {r['mid_vol']*100:.1f}% | Spread: {r['spread']*100:.1f}% ({r['spread_pct']*100:.0f}%)\n"
                        if r['bid_price'] and r['ask_price']:
                            msg += f"   Price: ${r['bid_price']:.2f} / ${r['ask_price']:.2f}\n"
                        if r['tv_ann_bid'] is not None and r['tv_ann_ask'] is not None:
                            msg += f"   TV(ann): {r['tv_ann_bid']:.1f}% / {r['tv_ann_ask']:.1f}%\n"
                        msg += f"   ITM: {r['itm_pct']:.1f}% (Fwd=${r['forward']:.2f}) | DTE: {r['dte']} | Score: {r['metric']*10000:.2f}\n\n"

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
            # Parse the argument to get symbol, option_type, expiration
            parts = argument.split('_')
            if len(parts) != 3:
                bot.send_message(call.message.chat.id, "Invalid selection format.")
                return

            symbol = parts[0]
            option_type = parts[1]
            expiration = parts[2]

            # Check if we have strike ranges for this symbol
            if symbol not in strike_ranges:
                bot.send_message(call.message.chat.id, f"No strike range defined for {symbol}.")
                return

            strikes = list(strike_ranges[symbol])

            # Check connection first
            if not app.isConnected():
                bot.send_message(call.message.chat.id, "❌ Not connected to IB. Connection may have dropped. Bot will auto-reconnect.")
                return

            # Send "fetching" message
            bot.send_message(call.message.chat.id, f"🔄 Fetching fresh IV data for {symbol} {option_type} {expiration}...")

            try:
                # Fetch fresh IV using MAIN app connection (no need to pause - uses different reqIds)
                iv_data = app.fetch_fresh_iv_smile(symbol, expiration, option_type, strikes)

                # Update the main app's volatilities with fresh data
                symbol_key = f"{symbol}_{option_type}_{expiration}"
                if symbol_key in app.volatilities:
                    for strike, data in iv_data.items():
                        if strike in app.volatilities[symbol_key]:
                            if data['bid_vol'] is not None:
                                app.volatilities[symbol_key][strike]['bid_vol'] = data['bid_vol']
                                app.volatilities[symbol_key][strike]['bid_price'] = data['bid_price']
                            if data['ask_vol'] is not None:
                                app.volatilities[symbol_key][strike]['ask_vol'] = data['ask_vol']
                                app.volatilities[symbol_key][strike]['ask_price'] = data['ask_price']
                            if data['underlying_price'] is not None:
                                app.volatilities[symbol_key][strike]['underlying_price'] = data['underlying_price']

                # Extract data for plotting
                strks = strikes
                bidvol = np.array([iv_data[s]['bid_vol'] for s in strks])
                askvol = np.array([iv_data[s]['ask_vol'] for s in strks])

            except Exception as e:
                status = get_market_status()
                msg = f"❌ Error fetching data: {str(e)}\n\n"
                msg += status['message']
                msg += "\n\n_Use /market to check market hours_"
                bot.send_message(call.message.chat.id, msg, parse_mode='Markdown')
                return

            # Filter out None values
            valid_bids = [v for v in bidvol if v is not None]
            valid_asks = [v for v in askvol if v is not None]

            all_valid = valid_bids + valid_asks
            if not all_valid:
                status = get_market_status()
                msg = "❌ No IV data received.\n\n"
                msg += status['message']
                msg += "\n\n_Use /market to check market hours_"
                bot.send_message(call.message.chat.id, msg, parse_mode='Markdown')
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
            plt.title(f'IV Smile: {argument} (Fresh)')
            plt.savefig('vol.png', dpi=100, bbox_inches='tight')
            plt.close()

            # Count how many data points we got
            bid_count = sum(1 for v in bidvol if v is not None)
            ask_count = sum(1 for v in askvol if v is not None)

            bot.send_photo(call.message.chat.id, open('vol.png', 'rb'), caption=f"✅ Fresh data: {bid_count} bids, {ask_count} asks")

        # Start Telegram polling in separate thread with auto-restart
        def telegram_polling():
            while True:
                try:
                    print("Starting Telegram polling...")
                    bot.infinity_polling(timeout=30, long_polling_timeout=30)
                except Exception as e:
                    print(f"Telegram polling error: {e}")
                    print("Restarting Telegram polling in 5 seconds...")
                    time.sleep(5)

        polling_thread = threading.Thread(target=telegram_polling, daemon=True)
        polling_thread.start()

        print("Bot is running! Use /ivc, /ivp, /atm, /calc, /status, /menu in Telegram")
        print("Press Ctrl+C to stop")

        # Keep the bot running
        while True:
            time.sleep(30)

    except KeyboardInterrupt:
        print("\nShutting down...")
        bot.send_message(chat_id, "🛑 *Bot Stopped*\nManual shutdown (Ctrl+C)", parse_mode='Markdown')
    except Exception as e:
        print(f"Fatal error: {e}")
        try:
            bot.send_message(chat_id, f"🔴 *Bot Crashed*\nError: {str(e)}", parse_mode='Markdown')
        except:
            pass
        raise
