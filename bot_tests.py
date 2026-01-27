import os
import telebot
from telebot import types
from dotenv import load_dotenv

import re
import time
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import MarketDataTypeEnum
import threading
import datetime
import matplotlib.pyplot as plt
import numpy as np
from connection import get_sql
from black_scholes import call_delta, put_delta, call_gamma, call_vega, call_theta, put_theta
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

def last_hours_smile(option_type, underlying, expiration, hours):
    df = get_sql('get_last_smile.sql', option_type=option_type, underlying=underlying, expiration=expiration, hours=hours)

    option_dict = {}
    for index, row in df.iterrows():
        if row['strike'] not in option_dict:
            option_dict[row['strike']] = {'bid_vol': None, 'ask_vol': None}
        dt = row.data_type
        if dt == 'BID':
            option_dict[row['strike']]['bid_vol'] = row['iv']
        if dt == 'ASK':
            option_dict[row['strike']]['ask_vol'] = row['iv']

    return option_dict

# Telegram bot
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
chat_id = TELEGRAM_CHAT_ID

# SPY only
symbols = ['SPY']

# Strike ranges - SPY 600 to 800
strike_ranges = {
    'SPY': np.arange(600, 805, 5),  # 600, 605, 610, ... 800 (41 strikes)
}

# Expiration dates - 3 tenors
expiration_dates = {
    'SPY': ['20260220', '20260417', '20260515'],  # Feb 20, Apr 17, May 15
}

# Build contracts list
contracts = []
for symbol in symbols:
    for expiration_date in expiration_dates[symbol]:
        for strike in strike_ranges[symbol]:
            for right in ['C', 'P']:  # Call and Put
                contract = Contract()
                contract.symbol = symbol
                contract.secType = "OPT"
                contract.exchange = "SMART"
                contract.currency = "USD"
                contract.lastTradeDateOrContractMonth = expiration_date
                contract.strike = strike
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
    def __init__(self):
        EClient.__init__(self, self)
        self.contracts = contracts
        self.strike_ranges = strike_ranges
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

        # Load historical data
        for symbol in symbols:
            for expiration_date in expiration_dates[symbol]:
                for right in ['C', 'P']:
                    last = last_hours_smile(right, symbol, expiration_date, 0)
                    self.volatilities[symbol + "_" + right + "_" + expiration_date].update(last)

        self.last_insert = dict()

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

        # Throttle: only insert every 60 seconds per contract/field
        key = (tickerId, field)
        if key in self.last_insert and seconds_to_now(self.last_insert[key]) < 60:
            return
        self.last_insert[key] = datetime.datetime.now()

        if impliedVolatility is None or undPrice is None:
            return

        contract = self.contract_details.get(tickerId)
        if not contract:
            return

        symbol_key = contract.symbol + "_" + contract.right + "_" + contract.lastTradeDateOrContractMonth
        strike = contract.strike

        # Store IV from IB
        if field == 10:  # BID
            self.volatilities[symbol_key][strike]['bid_vol'] = impliedVolatility
            data_type = 'BID'
        elif field == 11:  # ASK
            self.volatilities[symbol_key][strike]['ask_vol'] = impliedVolatility
            data_type = 'ASK'
        else:
            return

        # Calculate mid IV for Greeks
        bid_vol = self.volatilities[symbol_key][strike]['bid_vol']
        ask_vol = self.volatilities[symbol_key][strike]['ask_vol']

        if bid_vol is not None and ask_vol is not None:
            mid_vol = (bid_vol + ask_vol) / 2

            # Calculate time to expiration in years
            expiration = datetime.datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d')
            expiration = expiration + datetime.timedelta(hours=23)
            tau = (expiration - datetime.datetime.now()).total_seconds() / (365 * 24 * 60 * 60)

            if tau > 0:
                # Calculate Greeks using mid_vol
                r = 0.05  # risk-free rate
                try:
                    if contract.right == 'C':
                        calc_delta = call_delta(strike, undPrice, tau, mid_vol, r, 0)
                        calc_theta = call_theta(strike, undPrice, tau, mid_vol, r, 0)
                    else:
                        calc_delta = put_delta(strike, undPrice, tau, mid_vol, r, 0)
                        calc_theta = put_theta(strike, undPrice, tau, mid_vol, r, 0)

                    calc_gamma = call_gamma(strike, undPrice, tau, mid_vol, r, 0)
                    calc_vega = call_vega(strike, undPrice, tau, mid_vol, r, 0)
                except:
                    calc_delta = delta
                    calc_gamma = gamma
                    calc_vega = vega
                    calc_theta = theta
            else:
                calc_delta = delta
                calc_gamma = gamma
                calc_vega = vega
                calc_theta = theta
        else:
            # Use IB's Greeks if we don't have both bid/ask
            calc_delta = delta
            calc_gamma = gamma
            calc_vega = vega
            calc_theta = theta

        # Store to database - only IV from IB, Greeks calculated with mid_vol
        iv_db = 'NULL' if impliedVolatility is None else impliedVolatility
        delta_db = 'NULL' if calc_delta is None else calc_delta
        gamma_db = 'NULL' if calc_gamma is None else calc_gamma
        vega_db = 'NULL' if calc_vega is None else calc_vega
        theta_db = 'NULL' if calc_theta is None else calc_theta
        price_db = 'NULL' if optPrice is None else optPrice

        get_sql(
            'test_update.sql',
            timestamp=int(datetime.datetime.now().timestamp()),
            option_type=contract.right,
            expiration=contract.lastTradeDateOrContractMonth,
            strike=int(contract.strike),
            underlying=contract.symbol,
            data_type=data_type,
            price=price_db,
            underlying_price=undPrice,
            iv=iv_db,
            mikhail_iv='NULL',  # Not used anymore
            delta=delta_db,
            gamma=gamma_db,
            vega=vega_db,
            theta=theta_db
        )

        print(f"Saved: {contract.symbol} {contract.right} {strike} {data_type} IV={impliedVolatility:.4f}" if impliedVolatility else "")


if __name__ == '__main__':
    print("=" * 50)
    print("IB Options Monitor Bot - SPY Only")
    print("=" * 50)
    print(f"Monitoring {len(contracts)} option contracts")
    print(f"Symbols: {symbols}")
    print(f"Strikes: {list(strike_ranges['SPY'])}")
    print(f"Connecting to {IB_HOST}:{IB_PORT}...")

    # Create app and connect
    app = IBApp()
    app.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)

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
    app.reqMarketDataType(1)  # FROZEN - last available data
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
        # For SPY only, go directly to expiration
        markup = types.InlineKeyboardMarkup()
        for exp in expiration_dates['SPY']:
            markup.add(types.InlineKeyboardButton(str(exp), callback_data=f'SPY_C_{exp}'))
        bot.send_message(message.chat.id, "Select expiration (SPY Calls):", reply_markup=markup)

    @bot.message_handler(commands=['ivp'])
    def send_underlying_menu_puts(message):
        user_command_messages[message.chat.id] = message.text
        # For SPY only, go directly to expiration
        markup = types.InlineKeyboardMarkup()
        for exp in expiration_dates['SPY']:
            markup.add(types.InlineKeyboardButton(str(exp), callback_data=f'SPY_P_{exp}'))
        bot.send_message(message.chat.id, "Select expiration (SPY Puts):", reply_markup=markup)

    @bot.message_handler(commands=['status'])
    def send_status(message):
        bot.send_message(message.chat.id, f"Bot is running!\nMonitoring: SPY\nContracts: {len(contracts)}")

    @bot.callback_query_handler(func=lambda call: True)
    def handler(call):
        bot.answer_callback_query(call.id)
        argument = call.data  # e.g., SPY_C_20260220

        command_text = user_command_messages.get(call.message.chat.id, "")
        match = re.search(r'hour=(\d+)', command_text)
        hour = int(match.group(1)) if match else 0

        if hour == 0:
            for symbol_key, data in app.volatilities.items():
                if symbol_key == argument:
                    strks = list(data.keys())
                    bidvol = np.array([data[s]['bid_vol'] for s in strks])
                    askvol = np.array([data[s]['ask_vol'] for s in strks])
                    break
            else:
                bot.send_message(call.message.chat.id, "No data found for this selection.")
                return
        else:
            parts = argument.split('_')
            historical_data = last_hours_smile(parts[1], parts[0], parts[2], hour)
            strks = list(strike_ranges['SPY'])
            bidvol = np.array([historical_data.get(s, {}).get('bid_vol') for s in strks])
            askvol = np.array([historical_data.get(s, {}).get('ask_vol') for s in strks])

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

    print("Bot is running! Use /ivc, /ivp, /status in Telegram")
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
