import pandas as pd
import hashlib
import hmac
import time
import requests
import numpy as np
from urllib.parse import urlparse, urlencode
from typing import List, Any, Optional, Dict
import datetime
import os
import logging
import json
import math
import csv

pd.set_option('future.no_silent_downcasting', True)

# --- Configuration ---
MODE = 'fronttest'  # or 'real'
COINEX_ACCESS_ID = "57E8A3627B2745C2A58E2A9ACA20275B"
COINEX_SECRET_KEY = "3776028B7005930ED6BCA650541ADC492EEFA514743D750F"
LOG_FILE_PATH = "trading_bot_real_Arty_enhanced2.log"
LOG_LEVEL = logging.INFO
TRADE_LOG_CSV_PATH = "trade_log_arty_strategy_enhanced2.csv"
ANALYTICS_CSV_PATH = "trade_analytics_arty_strategy_enhanced2.csv"
CONDITION_LOG_CSV_PATH = "condition_log_arty_strategy_enhanced2.csv"  # New CSV path for condition logs

# --- Global Trades List for Aggregated Analytics ---
global_trades = []  # This list aggregates all closed trade logs

# --- Logging Configuration ---
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(LOG_FILE_PATH)
                    ])
logger = logging.getLogger(__name__)

# --- CoinEx API Client with Precision Handling ---
class RequestsClientWithPrecision(object):
    HEADERS = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "X-COINEX-KEY": "",
        "X-COINEX-SIGN": "",
        "X-COINEX-TIMESTAMP": "",
    }

    def __init__(self, access_id=None, secret_key=None):
        self.access_id = access_id
        self.secret_key = secret_key
        self.api_url = "https://api.coinex.com/v2"  # Define API URL here
        self.headers = self.HEADERS.copy()
        if access_id:
            self.headers["X-COINEX-KEY"] = access_id
        self.market_info = {}

    def format_symbol(self, symbol: str) -> str:
        return symbol.upper()

    def log_error(self, message: str):
        logger.error(message)

    def gen_sign(self, method, request_path, body, timestamp):
        prepared_str = f"{method}{request_path}{body}{timestamp}"
        signature = hmac.new(
            bytes(self.secret_key, 'latin-1'),
            msg=bytes(prepared_str, 'latin-1'),
            digestmod=hashlib.sha256
        ).hexdigest().lower()
        return signature

    def get_common_headers(self, signed_str, timestamp):
        headers = self.HEADERS.copy()
        headers["X-COINEX-KEY"] = self.access_id
        headers["X-COINEX-SIGN"] = signed_str
        headers["X-COINEX-TIMESTAMP"] = timestamp
        return headers

    def request(self, method, url, params={}, data=""):
        req = urlparse(url)
        request_path = req.path
        if method.upper() == "GET" and params:
            params = {k: v for k, v in params.items() if v is not None}
            request_path = request_path + "?" + urlencode(params)
        timestamp = str(int(time.time() * 1000))
        body = data if method.upper() != "GET" else ""

        signed_str = ""
        if self.secret_key:
            signed_str = self.gen_sign(method, request_path, body, timestamp)
            headers = self.get_common_headers(signed_str, timestamp)
        else:
            headers = self.headers

        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() in ["POST", "PUT"]:
                response = requests.request(method.upper(), url, data=data, headers=headers)
            elif method.upper() == "DELETE":
                response = requests.delete(url, params=params, headers=headers)
            else:
                raise ValueError("Unsupported HTTP method")
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error: {e}")
            raise

    def get_price_precision(self, symbol: str) -> int:
        market = self.format_symbol(symbol)
        market_info = self.get_market_info(market)
        if market_info is not None:
            precision = market_info.get("quote_precision")
            if precision is not None:
                return int(precision)
            else:
                self.log_error(f"Could not retrieve price precision for {market}. Using default price precision 5.")
                return 5
        else:
            self.log_error(f"Could not retrieve market info for {market}. Using default price precision 5.")
            return 5

    def get_minimum_amount(self, symbol: str) -> Optional[float]:
        market = self.format_symbol(symbol)
        market_info = self.get_market_info(market)
        if market_info is not None:
            min_amount_str = market_info.get("min_amount")
            if min_amount_str:
                return float(min_amount_str)
            else:
                self.log_error(f"Minimum amount info not provided for {market}.")
                return None
        else:
            self.log_error(f"Could not retrieve market info for {market} to get minimum amount.")
            return None

    def get_market_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        market = self.format_symbol(symbol)
        if market not in self.market_info:
            try:
                market_url = f"{self.api_url}/futures/market"
                response = requests.get(f"{market_url}?market={market}")
                response.raise_for_status()
                data = response.json()
                if data.get("code") == 0 and data.get("data"):
                    for m in data["data"]:
                        name = m.get("market")
                        if name:
                            self.market_info[name.upper()] = m
                else:
                    self.log_error(f"Error fetching market info: {data}")
                    return None
            except Exception as e:
                self.log_error(f"Error fetching market info: {e}")
                return None
        return self.market_info.get(market)

    def adjust_values(self, symbol: str, entry_price: float, quantity: float, sl_price: float, tp_price: float, tp_price_partial: float = None) -> tuple:
        try:
            price_precision = self.get_price_precision(symbol)
            min_amount = self.get_minimum_amount(symbol)
            amount_precision_decimals = 8
            if min_amount is not None and min_amount > 0:
                amount_precision_decimals = max(0, -math.floor(math.log10(min_amount)))
            adjusted_entry_price = round(entry_price, price_precision)
            adjusted_sl_price = round(sl_price, price_precision)
            adjusted_tp_price = round(tp_price, price_precision)
            adjusted_quantity = round(quantity, amount_precision_decimals)
            adjusted_tp_price_partial = None
            if tp_price_partial is not None:
                adjusted_tp_price_partial = round(tp_price_partial, price_precision)
            return adjusted_entry_price, adjusted_quantity, adjusted_sl_price, adjusted_tp_price, adjusted_tp_price_partial
        except Exception as e:
            logger.error(f"Error adjusting values for {symbol}: {e}")
            return None, None, None, None, None

    def use_real_data_in_simulation(self):
        return False

# Initialize API Client
request_client = RequestsClientWithPrecision(access_id=COINEX_ACCESS_ID, secret_key=COINEX_SECRET_KEY)

# --- Helper Functions ---
def convert_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp) / 1000).strftime('%Y-%m-%d %H:%M:%S')

def get_historical_klines(symbol: str, interval: str = '1min', limit: int = 1000) -> Optional[List[List[Any]]]:
    max_retries = 3
    backoff_factor = 0.5
    for attempt in range(max_retries):
        try:
            market = symbol
            api_url = request_client.api_url
            url = f"{api_url}/futures/kline?market={market}&period={interval}&limit={limit}"
            response = request_client.request("GET", url)
            data = response.json()
            if data.get("code") == 0 and data.get("data"):
                return data["data"]
            else:
                logger.warning(f"Attempt {attempt + 1}: Error fetching klines for {symbol}: {data}")
                time.sleep(backoff_factor * (2 ** attempt))
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}: Error fetching klines for {symbol}: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    logger.error(f"Failed to fetch klines for {symbol} after {max_retries} attempts")
    return None

def fetch_ohlcv(symbol, timeframe, limit=1000, mode="backtest", data_source="coinex",
                csv_file_path_higher="{symbol}_30m_data.csv", csv_file_path_lower="{symbol}_1m_data.csv",
                higher_timeframe='4hour', lower_timeframe='3min'):
    if data_source == "csv":
        if timeframe == "higher":
            csv_path = csv_file_path_higher.format(symbol=symbol)
        elif timeframe == "lower":
            csv_path = csv_file_path_lower.format(symbol=symbol)
        else:
            logger.error(f"Invalid timeframe for CSV: {timeframe}")
            return pd.DataFrame()
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
            df.sort_index(inplace=True)
            return df
        else:
            logger.error(f"CSV file not found for {symbol}: {csv_path}")
            return pd.DataFrame()
    elif data_source == "coinex":
        if timeframe == "higher":
            period = higher_timeframe
        else:
            period = lower_timeframe
        ohlcv_data = get_historical_klines(symbol, period, limit=limit)
        if ohlcv_data:
            df_data = []
            for item in ohlcv_data:
                df_data.append([
                    int(item['created_at']),
                    float(item['open']), float(item['close']),
                    float(item['high']), float(item['low']),
                    float(item['volume'])
                ])
            df = pd.DataFrame(df_data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
        else:
            return pd.DataFrame()
    else:
        logger.error(f"Error: Invalid data_source: {data_source}. Choose 'coinex' or 'csv'.")
        return pd.DataFrame()

# --- Trading Functions ---
def place_market_order(symbol, side, amount, entry_price=None, sl_price=None, tp_price=None):
    if MODE == 'fronttest':
        logger.info(f"[SIMULATED ORDER] ({symbol}) Placing {side} market order: amount={amount}, entry={entry_price}, SL={sl_price}, TP={tp_price}")
        return {'order_id': 'simulated_order_id'}
    else:
        order_path = "/futures/order/market"
        order_url = f"{request_client.api_url}{order_path}"
        order_data = {
            "market": symbol,
            "side": side,
            "amount": str(amount)
        }
        try:
            response_order = request_client.request("POST", order_url, data=json.dumps(order_data))
            result_order = response_order.json()
            if result_order.get("code") == 0:
                order_id = result_order["data"]["order_id"]
                logger.info(f"[REAL ORDER] ({symbol}) Market order placed - Order ID: {order_id}, Side: {side}, Amount: {amount}")
                return {'order_id': order_id}
            else:
                logger.error(f"[REAL ORDER ERROR] ({symbol}) Error placing market order: {result_order}")
                return None
        except Exception as e:
            logger.error(f"[REAL ORDER EXCEPTION] ({symbol}) Exception placing market order: {e}")
            return None

def set_leverage(symbol, leverage_value):
    if MODE != 'real':
        logger.info(f"[SIMULATED - Not in real mode] ({symbol}) Setting leverage to {leverage_value}x for {symbol}")
        return True
    leverage_path = "/futures/position/adjust_leverage"
    leverage_url = f"{request_client.api_url}{leverage_path}"
    leverage_data = {
        "market": symbol,
        "leverage": int(leverage_value)
    }
    try:
        response_leverage = request_client.request("POST", leverage_url, data=json.dumps(leverage_data))
        result_leverage = response_leverage.json()
        if result_leverage.get("code") == 0:
            logger.info(f"[REAL ACTION] ({symbol}) Leverage set to {leverage_value}x: {result_leverage['data']}")
            return True
        else:
            logger.error(f"[REAL ACTION ERROR] ({symbol}) Error setting leverage: {result_leverage}")
            return False
    except Exception as e:
        logger.error(f"[REAL ACTION EXCEPTION] ({symbol}) Exception setting leverage: {e}")
        return False

def set_take_profit(symbol, tp_price, take_profit_type="latest_price"):
    if MODE != 'real':
        logger.info(f"[SIMULATED - Not in real mode] ({symbol}) Setting take profit at {tp_price} (type: {take_profit_type})")
        return True
    tp_path = "/futures/position/set_take_profit"
    tp_url = f"{request_client.api_url}{tp_path}"
    tp_data = {
        "market": symbol,
        "price_type": "latest_price",
        "price": str(tp_price)
    }
    try:
        response_tp = request_client.request("POST", tp_url, data=json.dumps(tp_data))
        result_tp = response_tp.json()
        if result_tp.get("code") == 0:
            logger.info(f"[REAL ACTION] ({symbol}) Take profit set at {tp_price} (type: {take_profit_type}): {result_tp['data']}")
            return True
        else:
            logger.error(f"[REAL ACTION ERROR] ({symbol}) Error setting take profit: {result_tp}")
            return False
    except Exception as e:
        logger.error(f"[REAL ACTION EXCEPTION] ({symbol}) Exception setting take profit: {e}")
        return False

def set_stop_loss(symbol, sl_price, stop_loss_type="latest_price"):
    if MODE != 'real':
        logger.info(f"[SIMULATED - Not in real mode] ({symbol}) Setting stop loss at {sl_price} (type: {stop_loss_type})")
        return True
    sl_path = "/futures/position/set_stop_loss"
    sl_url = f"{request_client.api_url}{sl_path}"
    sl_data = {
        "market": symbol,
        "price_type": "latest_price",
        "price": str(sl_price)
    }
    try:
        response_sl = request_client.request("POST", sl_url, data=json.dumps(sl_data))
        result_sl = response_sl.json()
        if result_sl.get("code") == 0:
            logger.info(f"[REAL ACTION] ({symbol}) Stop loss set at {sl_price} (type: {stop_loss_type}): {result_sl['data']}")
            return True
        else:
            logger.error(f"[REAL ACTION ERROR] ({symbol}) Error setting stop loss: {result_sl}")
            return False
    except Exception as e:
        logger.error(f"[REAL ACTION EXCEPTION] ({symbol}) Exception setting stop loss: {e}")
        return False

def get_futures_balance_real_mode():
    request_path = "/futures/balance"
    url = f"{request_client.api_url}{request_path}"
    try:
        response = request_client.request("GET", url)
        data = response.json()
        if data.get("code") == 0:
            balance_data = data.get("data") or {}
            available_balance = float(balance_data.get("USDT", {}).get("available", 0))
            logger.info(f"[REAL BALANCE FETCH] Fetched USDT balance: {available_balance}")
            return available_balance
        else:
            logger.error(f"[REAL BALANCE FETCH ERROR] Error fetching futures balance: {data}")
            return None
    except Exception as e:
        logger.error(f"[REAL BALANCE FETCH EXCEPTION] Exception fetching futures balance: {e}")
        return None

# --- Strategy Core Functions ---
def calculate_smma(series, window):
    if not isinstance(series, pd.Series):
        logger.error(f"Expected pd.Series, but got {type(series)} in calculate_smma") # Debug: Check series type
        return pd.Series([np.nan] * len(series)) if isinstance(series, (list, tuple, np.ndarray)) else np.nan # Handle non-series input

    smma = np.full(len(series), np.nan)
    if len(series) < window:
        return pd.Series(smma, index=series.index) # Return NaN series if not enough data

    first_smma = series[:window].mean()
    smma[window - 1] = first_smma
    for i in range(window, len(series)):
        smma[i] = (smma[i - 1] * (window - 1) + series.iloc[i]) / window
    return pd.Series(smma, index=series.index)

def calculate_adx(df, period=14):
    df = df.copy()

    # True Range (TR)
    df['High-Low'] = df['high'] - df['low']
    df['High-PrevClose'] = abs(df['high'] - df['close'].shift(1))
    df['Low-PrevClose'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    # Directional Movement (+DM and -DM)
    df['High-High'] = df['high'] - df['high'].shift(1)
    df['Low-Low'] = df['low'].shift(1) - df['low']
    df['+DM'] = np.where((df['High-High'] > df['Low-Low']) & (df['High-High'] > 0), df['High-High'], 0)
    df['-DM'] = np.where((df['Low-Low'] > df['High-High']) & (df['Low-Low'] > 0), df['Low-Low'], 0)

    # Smooth TR, +DM, and -DM using Wilder's Smoothing
    df['Smoothed_TR'] = np.nan
    df['Smoothed_+DM'] = np.nan
    df['Smoothed_-DM'] = np.nan

    df.loc[df.index[period-1], 'Smoothed_TR'] = df['TR'][:period].sum()
    df.loc[df.index[period-1], 'Smoothed_+DM'] = df['+DM'][:period].sum()
    df.loc[df.index[period-1], 'Smoothed_-DM'] = df['-DM'][:period].sum()

    for i in range(period, len(df)):
        df.loc[df.index[i], 'Smoothed_TR'] = (
            df['Smoothed_TR'].iloc[i-1] - (df['Smoothed_TR'].iloc[i-1] / period) + df['TR'].iloc[i]
        )
        df.loc[df.index[i], 'Smoothed_+DM'] = (
            df['Smoothed_+DM'].iloc[i-1] - (df['Smoothed_+DM'].iloc[i-1] / period) + df['+DM'].iloc[i]
        )
        df.loc[df.index[i], 'Smoothed_-DM'] = (
            df['Smoothed_-DM'].iloc[i-1] - (df['Smoothed_-DM'].iloc[i-1] / period) + df['-DM'].iloc[i]
        )

    # Directional Indicators (+DI and -DI)
    df['+DI'] = (df['Smoothed_+DM'] / df['Smoothed_TR']) * 100
    df['-DI'] = (df['Smoothed_-DM'] / df['Smoothed_TR']) * 100

    # Directional Index (DX)
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100

    # Average Directional Index (ADX)
    df['ADX'] = np.nan
    df.loc[df.index[period-1], 'ADX'] = df['DX'].iloc[period-1]
    for i in range(period, len(df)):
        df.loc[df.index[i], 'ADX'] = (
            (df['ADX'].iloc[i-1] * (period - 1) + df['DX'].iloc[i]) / period
        )

    return df[['ADX', '+DI', '-DI']] # Return ADX, +DI, -DI as DataFrame


def is_strong_uptrend_adx_di(df: pd.DataFrame, adx_period=14, di_diff_threshold=5) -> pd.Series:
    """
    Checks for strong uptrend based on ADX and Directional Indicators.
    (rest of docstring remains the same)
    """
    adx_data = calculate_adx(df, period=adx_period)
    df = pd.concat([df, adx_data], axis=1) # Combine ADX data with original DataFrame

    if len(df) < 2: # Need at least two rows to compare current and previous ADX
        return pd.Series([False] * len(df), index=df.index)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Use .item() to extract scalar values for comparison
    condition = (
        latest['ADX'].iloc[0] > previous['ADX'].iloc[0] and
        latest['+DI'].iloc[0] > latest['-DI'].iloc[0] and
        (latest['+DI'].iloc[0] - latest['-DI'].iloc[0]) > di_diff_threshold
    )
    return pd.Series([condition] * len(df), index=df.index) # Return Series of boolean with same index

def is_strong_downtrend_adx_di(df: pd.DataFrame, adx_period=14, di_diff_threshold=5) -> pd.Series:
    """
    Checks for strong downtrend based on ADX and Directional Indicators.
    (rest of docstring remains the same)
    """
    adx_data = calculate_adx(df, period=adx_period)
    df = pd.concat([df, adx_data], axis=1) # Combine ADX data with original DataFrame

    if len(df) < 2: # Need at least two rows to compare current and previous ADX
        return pd.Series([False] * len(df), index=df.index)

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Use .item() to extract scalar values for comparison
    condition = (
        latest['ADX'].iloc[0] > previous['ADX'].iloc[0] and
        latest['-DI'].iloc[0] > latest['+DI'].iloc[0] and
        (latest['-DI'].iloc[0] - latest['+DI'].iloc[0]) > di_diff_threshold
    )
    return pd.Series([condition] * len(df), index=df.index) # Return Series of boolean with same index

def is_ranging_market_adx(df: pd.DataFrame, adx_period=14, adx_threshold=20) -> pd.Series:
    """
    Checks if the market is ranging based on ADX value.
    (rest of docstring remains the same)
    """
    adx_data = calculate_adx(df, period=adx_period)
    df = pd.concat([df, adx_data], axis=1) # Combine ADX data with original DataFrame

    # Use .item() to extract scalar value for comparison
    latest_adx = df['ADX'].iloc[0]
    condition = latest_adx < adx_threshold
    return pd.Series([condition] * len(df), index=df.index) # Return Series of boolean with same index

def calculate_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.rolling(window).mean()
    ema_down = down.rolling(window).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, window=14):
    true_range = pd.DataFrame()
    true_range['HL'] = df['high'] - df['low']
    true_range['HC'] = abs(df['high'] - df['close'].shift(1))
    true_range['LC'] = abs(df['low'] - df['close'].shift(1))
    true_range['TR'] = true_range[['HL', 'HC', 'LC']].max(axis=1)
    atr = true_range['TR'].rolling(window).mean()
    return atr

def initialize_condition_log_csv():
    if not os.path.exists(CONDITION_LOG_CSV_PATH):
        with open(CONDITION_LOG_CSV_PATH, mode='w', newline='') as csvfile:
            condition_writer = csv.writer(csvfile)
            condition_writer.writerow([
                'Timestamp', 'Symbol', 'Close_Price',
                'SMMA_21', 'SMMA_50', 'SMMA_200', 'RSI', 'ATR', 'Body_Size', 'Body_Percent',
                'Long_Cond1', 'Long_Cond2', 'Long_Cond3', 'Long_Cond4', 'Long_Cond5', 'Long_Cond6', 'Long_Cond7', 'Long_Cond_PriceDist', 'Long_Cond_AvoidTouch200', 'Long_Cond_AvoidLongShadow',  'Long_Signal',
                'Short_Cond1', 'Short_Cond2', 'Short_Cond3', 'Short_Cond4', 'Short_Cond5', 'Short_Cond6', 'Short_Cond7', 'Short_Cond_PriceDist', 'Short_Cond_AvoidTouch200',  'Short_Cond_AvoidLongShadow',  'Short_Signal',
                'lookback_window_touch_200', 'price_distance_multiplier', 'correction_lookback'
            ])

def log_condition_to_csv(condition_data):
    with open(CONDITION_LOG_CSV_PATH, mode='a', newline='') as csvfile:
        condition_writer = csv.writer(csvfile)
        condition_writer.writerow([
            condition_data.get('timestamp'),
            condition_data.get('symbol'),
            condition_data.get('close_price'),
            condition_data.get('smma_21'),
            condition_data.get('smma_50'),
            condition_data.get('smma_200'),
            condition_data.get('rsi'),
            condition_data.get('atr'),
            condition_data.get('body_size'),
            condition_data.get('body_percent'),
            condition_data.get('long_cond1'),
            condition_data.get('long_cond2'),
            condition_data.get('long_cond3'),
            condition_data.get('long_cond4'),
            condition_data.get('long_cond5'),
            condition_data.get('long_cond6'),
            condition_data.get('long_cond7'),
            condition_data.get('long_cond_price_distance'),
            condition_data.get('long_cond_avoid_touch_200'),
            condition_data.get('long_cond_avoid_long_shadow'),
            condition_data.get('long_signal'),
            condition_data.get('short_cond1'),
            condition_data.get('short_cond2'),
            condition_data.get('short_cond3'),
            condition_data.get('short_cond4'),
            condition_data.get('short_cond5'),
            condition_data.get('short_cond6'),
            condition_data.get('short_cond7'),
            condition_data.get('short_cond_price_distance'),
            condition_data.get('short_cond_avoid_touch_200'),
            condition_data.get('short_cond_avoid_long_shadow'),
            condition_data.get('short_signal'),
            condition_data.get('lookback_window_touch_200'),
            condition_data.get('price_distance_multiplier'),
            condition_data.get('correction_lookback')
        ])


def generate_signals(df, symbol, lookback_window_touch_200=20, price_distance_multiplier=1.5, correction_lookback=5):
    """
    Generate trading signals using enhanced Arty's strategy with
    the 'correction-then-break' filter on SMMA 50.
    """
    if df.empty: # Handle empty DataFrame at the beginning
        return df

    # Calculate SMMAs
    df['SMMA_21'] = calculate_smma(df['close'], 21)
    df['SMMA_50'] = calculate_smma(df['close'], 50)
    df['SMMA_200'] = calculate_smma(df['close'], 200)
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['close'], window=14)
    # Calculate ATR
    df['ATR'] = calculate_atr(df, window=14)
    # Candle metrics
    df['body_size'] = (df['close'] - df['open']).abs()
    df['candle_size'] = (df['high'] - df['low']).abs()
    df['body_percent'] = df['body_size'] / df['candle_size']

    # Price distance from 200 SMMA filter
    price_distance_from_200_smma = df['close'] - df['SMMA_200']
    atr_price_distance = df['ATR'] * price_distance_multiplier

    # --- Calculate ADX, +DI, -DI ---
    adx_data = calculate_adx(df, period=14) # Default period 14, can be configurable
    df = pd.concat([df, adx_data], axis=1) # Combine ADX data with original DataFrame
    # --- New ADX/DI based conditions ---
    long_cond_strong_uptrend_adx_di = is_strong_uptrend_adx_di(df) # Using default parameters for now
    short_cond_strong_downtrend_adx_di = is_strong_downtrend_adx_di(df)
    neutral_cond_ranging_market_adx = is_ranging_market_adx(df)


    # --- Long Conditions (Original) ---
    long_cond1 = df['close'] > df['SMMA_200']
    long_cond2 = df['SMMA_50'] > df['SMMA_200']
    long_cond3 = df['SMMA_21'] > df['SMMA_50']
    long_cond4 = (df['SMMA_21'] > df['SMMA_50']) & (df['SMMA_50'] > df['SMMA_200'])
    long_cond5 = (df['close'] > df['SMMA_50']) 
    long_cond6 = (df['close'] > df['open']) & (df['body_percent'] > 0.6) & ((df["body_size"] / df['close']) > 0.0005)
    long_cond7 = df['low'] < df['SMMA_50']

    long_cond_price_distance = price_distance_from_200_smma > atr_price_distance
    recent_touch_200_smma_long = df['low'].rolling(window=lookback_window_touch_200).min() <= df['SMMA_200']
    long_cond_avoid_touch_200 = ~recent_touch_200_smma_long.fillna(False)


    # --- NEW CONDITION: AVOID LONG SHADOW ON SMMA 50 BREAKOUT ---
    df['body_size'] = df['body_size'].replace(0, 1e-9) # Prevent division by zero
    long_cond_avoid_long_shadow = ~(((df['open'] - df['low']) / df['body_size'] > 1.5) & (df['close'] > df['SMMA_50']))
    long_cond_avoid_long_shadow = long_cond_avoid_long_shadow.fillna(True) # handle NaN when body_size is zero

    # --- NEW CONDITION: AVOID TOUCHING SMMA 21 ---
    long_cond_avoid_touch_smma21 = ~(df['close'] <= df['SMMA_21'])
    long_cond_avoid_touch_smma21 = long_cond_avoid_touch_smma21.fillna(True)

    # --- Short Conditions (Original) ---
    short_cond1 = df['close'] < df['SMMA_200']
    short_cond2 = df['SMMA_50'] < df['SMMA_200']
    short_cond3 = df['SMMA_21'] < df['SMMA_50']
    short_cond4 = (df['SMMA_21'] < df['SMMA_50']) & (df['SMMA_50'] < df['SMMA_200'])
    short_cond5 = (df['close'] < df['SMMA_50'])
    short_cond6 = (df['close'] < df['open']) & (df['body_percent'] > 0.6) & ((df["body_size"] / df['close']) > 0.0005)
    short_cond7 = df['high'] > df['SMMA_50']
    short_cond_price_distance = price_distance_from_200_smma < -atr_price_distance
    recent_touch_200_smma_short = df['high'].rolling(window=lookback_window_touch_200).max() >= df['SMMA_200']
    short_cond_avoid_touch_200 = ~recent_touch_200_smma_short.fillna(False)


    # --- NEW CONDITION: AVOID LONG SHADOW ON SMMA 50 BREAKOUT ---
    df['body_size'] = df['body_size'].replace(0, 1e-9) # Prevent division by zero
    short_cond_avoid_long_shadow = ~(((df['open'] - df['high']) / df['body_size'] > 1.5) & (df['close'] < df['SMMA_50']))
    short_cond_avoid_long_shadow = short_cond_avoid_long_shadow.fillna(True) # handle NaN when body_size is zero

    # --- DEBUGGING PRINTS FOR CONDITIONS ---
    # print(f"--- Long Conditions for {symbol} ({df.index[-1] if not df.empty else 'No data'}) ---") # Handle empty df
    # print(f"Cond1: {long_cond1.iloc[-1] if not df.empty else 'No data'}, Cond2: {long_cond2.iloc[-1] if not df.empty else 'No data'}, Cond3: {long_cond3.iloc[-1] if not df.empty else 'No data'}, Cond4: {long_cond4.iloc[-1] if not df.empty else 'No data'}, Cond5: {long_cond5.iloc[-1] if not df.empty else 'No data'}, Cond6: {long_cond6.iloc[-1] if not df.empty else 'No data'}, Cond7: {long_cond7}, PriceDist: {long_cond_price_distance.iloc[-1] if not df.empty else 'No data'}, AvoidTouch200: {long_cond_avoid_touch_200.iloc[-1] if not df.empty else 'No data'}, CorrectionFilter: {long_cond_correction_filter.iloc[-1] if not df.empty else 'No data'}, AvoidLongShadow: {long_cond_avoid_long_shadow.iloc[-1] if not df.empty else 'No data'}, AvoidTouchSMMA21: {long_cond_avoid_touch_smma21.iloc[-1] if not df.empty else 'No data'}")

    df['Long_Signal'] = (
        long_cond1 & long_cond2 & long_cond3 & long_cond4 &
        long_cond5 & long_cond6 & long_cond7 &
        long_cond_price_distance & long_cond_avoid_touch_200 &
        long_cond_avoid_long_shadow & long_cond_strong_uptrend_adx_di
    )

    # print(f"--- Short Conditions for {symbol} ({df.index[-1] if not df.empty else 'No data'}) ---") # Handle empty df
    # print(f"Cond1: {short_cond1.iloc[-1] if not df.empty else 'No data'}, Cond2: {short_cond2.iloc[-1] if not df.empty else 'No data'}, Cond3: {short_cond3.iloc[-1] if not df.empty else 'No data'}, Cond4: {short_cond4.iloc[-1] if not df.empty else 'No data'}, Cond5: {short_cond5.iloc[-1] if not df.empty else 'No data'}, Cond6: {short_cond6.iloc[-1] if not df.empty else 'No data'}, Cond7: {short_cond7}, PriceDist: {short_cond_price_distance.iloc[-1] if not df.empty else 'No data'}, AvoidTouch200: {short_cond_avoid_touch_200.iloc[-1] if not df.empty else 'No data'}, CorrectionFilter: {short_cond_correction_filter.iloc[-1] if not df.empty else 'No data'}, AvoidLongShadow: {short_cond_avoid_long_shadow.iloc[-1] if not df.empty else 'No data'}, AvoidTouchSMMA21: {short_cond_avoid_touch_smma21.iloc[-1] if not df.empty else 'No data'}")


    df['Short_Signal'] = (
        short_cond1 & short_cond2 & short_cond3 & short_cond4 &
        short_cond5 & short_cond6 & short_cond7 &
        short_cond_price_distance & short_cond_avoid_touch_200 &
        short_cond_avoid_long_shadow & short_cond_strong_downtrend_adx_di
    )

    # --- DEBUGGING PRINTS FOR INDICATORS ---
    # print(f"--- Indicators for {symbol} ({df.index[-1] if not df.empty else 'No data'}) ---") # Handle empty df
    # if not df.empty:
    #     print(df[['close', 'SMMA_21', 'SMMA_50', 'SMMA_200', 'RSI', 'ATR', 'body_size', 'body_percent']].tail(1)) # Only print last row

    # --- Log conditions and indicators to CSV ---
    condition_log_data = {
        'timestamp': df.index[-1] if not df.empty else datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # Use current time if df is empty
        'symbol': symbol,
        'close_price': df['close'].iloc[-1] if not df.empty else 'No data',
        'smma_21': df['SMMA_21'].iloc[-1] if not df.empty else 'No data',
        'smma_50': df['SMMA_50'].iloc[-1] if not df.empty else 'No data',
        'smma_200': df['SMMA_200'].iloc[-1] if not df.empty else 'No data',
        'rsi': df['RSI'].iloc[-1] if not df.empty else 'No data',
        'atr': df['ATR'].iloc[-1] if not df.empty else 'No data',
        'body_size': df['body_size'].iloc[-1] if not df.empty else 'No data',
        'body_percent': df['body_percent'].iloc[-1] if not df.empty else 'No data',
        'long_cond1': long_cond1.iloc[-1] if not df.empty else 'No data',
        'long_cond2': long_cond2.iloc[-1] if not df.empty else 'No data',
        'long_cond3': long_cond3.iloc[-1] if not df.empty else 'No data',
        'long_cond4': long_cond4.iloc[-1] if not df.empty else 'No data',
        'long_cond5': long_cond5.iloc[-1] if not df.empty else 'No data',
        'long_cond6': long_cond6.iloc[-1] if not df.empty else 'No data',
        'long_cond7': long_cond7,
        'long_cond_price_distance': long_cond_price_distance.iloc[-1] if not df.empty else 'No data',
        'long_cond_avoid_touch_200': long_cond_avoid_touch_200.iloc[-1] if not df.empty else 'No data',
        'long_cond_avoid_long_shadow': long_cond_avoid_long_shadow.iloc[-1] if not df.empty else 'No data',
        'long_cond_avoid_touch_smma21': long_cond_avoid_touch_smma21.iloc[-1] if not df.empty else 'No data',
        'long_signal': df['Long_Signal'].iloc[-1] if not df.empty else 'No data',
        'short_cond1': short_cond1.iloc[-1] if not df.empty else 'No data',
        'short_cond2': short_cond2.iloc[-1] if not df.empty else 'No data',
        'short_cond3': short_cond3.iloc[-1] if not df.empty else 'No data',
        'short_cond4': short_cond4.iloc[-1] if not df.empty else 'No data',
        'short_cond5': short_cond5.iloc[-1] if not df.empty else 'No data',
        'short_cond6': short_cond6.iloc[-1] if not df.empty else 'No data',
        'short_cond7': short_cond7,
        'short_cond_price_distance': short_cond_price_distance.iloc[-1] if not df.empty else 'No data',
        'short_cond_avoid_touch_200': short_cond_avoid_touch_200.iloc[-1] if not df.empty else 'No data',
        'short_cond_avoid_long_shadow': short_cond_avoid_long_shadow.iloc[-1] if not df.empty else 'No data',
        'short_signal': df['Short_Signal'].iloc[-1] if not df.empty else 'No data',
        'lookback_window_touch_200': lookback_window_touch_200,
        'price_distance_multiplier': price_distance_multiplier,
        'correction_lookback': correction_lookback
    }
    log_condition_to_csv(condition_log_data)


    return df

def calculate_position_size(balance, risk_percent, entry_price, stop_loss_atr_value, leverage):
    risk_amount = balance * risk_percent
    return risk_amount / (stop_loss_atr_value * leverage)

# --- CSV Trade Logging Functions ---
def initialize_trade_log_csv():
    if not os.path.exists(TRADE_LOG_CSV_PATH):
        with open(TRADE_LOG_CSV_PATH, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Trade ID', 'Symbol', 'Entry Time', 'Entry Price', 'Side', 'Amount', 'Stop Loss', 'Take Profit', 'Status', 'Close Time', 'Profit/Loss', 'Bars Held'])

def log_trade_to_csv(trade_data):
    with open(TRADE_LOG_CSV_PATH, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            trade_data.get('trade_id'),
            trade_data.get('symbol'),
            trade_data.get('entry_time'),
            trade_data.get('entry_price'),
            trade_data.get('side'),
            trade_data.get('amount'),
            trade_data.get('stop_loss'),
            trade_data.get('take_profit'),
            trade_data.get('status'),
            trade_data.get('close_time'),
            trade_data.get('profit_loss', ''),
            trade_data.get('bars_held', '')
        ])

def initialize_analytics_csv():
    if not os.path.exists(ANALYTICS_CSV_PATH):
        with open(ANALYTICS_CSV_PATH, mode='w', newline='') as csvfile:
            analytics_writer = csv.writer(csvfile)
            analytics_writer.writerow(['Timestamp', 'Total Trades', 'Win Rate (%)', 'Loss Rate (%)', 'Profit Factor', 'Average Win', 'Average Loss', 'Net Profit'])

def log_analytics_to_csv(analytics_data):
    with open(ANALYTICS_CSV_PATH, mode='a', newline='') as csvfile:
        analytics_writer = csv.writer(csvfile)
        analytics_writer.writerow([
            analytics_data.get('timestamp'),
            analytics_data.get('total_trades'),
            analytics_data.get('win_rate_percent'),
            analytics_data.get('loss_rate_percent'),
            analytics_data.get('profit_factor'),
            analytics_data.get('average_win'),
            analytics_data.get('average_loss'),
            analytics_data.get('net_profit')
        ])

def calculate_trade_analytics(trades):
    if not trades:
        return {}
    wins = [trade['profit_loss'] for trade in trades if trade['profit_loss'] > 0]
    losses = [trade['profit_loss'] for trade in trades if trade['profit_loss'] <= 0]
    total_trades = len(trades)
    win_rate_percent = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    loss_rate_percent = (len(losses) / total_trades) * 100 if total_trades > 0 else 0
    total_profit = sum(wins)
    total_loss = abs(sum(losses))
    net_profit = total_profit - total_loss
    profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0)
    average_win = np.mean(wins) if wins else 0
    average_loss = np.mean(losses) if losses else 0
    return {
        'total_trades': total_trades,
        'win_rate_percent': win_rate_percent,
        'loss_rate_percent': loss_rate_percent,
        'profit_factor': profit_factor,
        'average_win': average_win,
        'average_loss': average_loss,
        'net_profit': net_profit
    }

def process_strategy_logic(symbol, df, df_4h, symbol_data, risk_percentage, risk_reward, leverage,
                           atr_stop_loss_multiplier=2.0, partial_tp_factor=2.0,
                           trailing_stop_atr_multiplier=1.0, lookback_window_touch_200=20,
                           price_distance_multiplier=1.5, correction_lookback=5):
    try:
        df = generate_signals(df, symbol, lookback_window_touch_200, price_distance_multiplier, correction_lookback)

        # --- DEBUGGING PRINTS FOR 4H TREND FILTER ---
        if df_4h is not None and not df_4h.empty:
            df_4h['SMMA_200_4h'] = calculate_smma(df_4h['close'], 200)
            current_4h_smma_200 = df_4h['SMMA_200_4h'].iloc[-1]
            current_4h_close = df_4h['close'].iloc[-1]
            four_hour_uptrend = current_4h_close > current_4h_smma_200
            four_hour_downtrend = current_4h_close < current_4h_smma_200
            # print(f"--- 4H Trend for {symbol} ({df.index[-1]}) ---") # Add timestamp for context from 3min chart for alignment
            # print(f"4H Uptrend: {four_hour_uptrend}, 4H Downtrend: {four_hour_downtrend}, 4H SMMA_200: {current_4h_smma_200:.2f}, 4H Close: {current_4h_close:.2f}")

        else:
            four_hour_uptrend = True
            four_hour_downtrend = True
            logger.warning(f"4-Hour data unavailable for {symbol}. Trend filter bypassed.")

        if df.empty: # Added check for empty DataFrame before strategy execution
            logger.warning(f"No data fetched for {symbol}. Skipping strategy logic.")
            return

        last_row = df.iloc[-1]
        if not symbol_data['in_trade']:
            # Long entry
            if last_row['Long_Signal']: # Removed 4h trend condition for debugging, add back if needed: and four_hour_uptrend
                entry_price = last_row['close']
                atr_value = last_row['ATR']
                stop_loss_price = entry_price - (atr_value * atr_stop_loss_multiplier)
                take_profit_price = entry_price + (atr_value * risk_reward * atr_stop_loss_multiplier)
                partial_tp_price = entry_price + (atr_value * partial_tp_factor * atr_stop_loss_multiplier)
                position_size = calculate_position_size(
                    symbol_data['balance'], risk_percentage,
                    entry_price, atr_value * atr_stop_loss_multiplier, leverage
                )
                adjusted_entry, adjusted_size, adjusted_sl, adjusted_tp, adjusted_tp_partial = request_client.adjust_values(
                    symbol, entry_price, position_size, stop_loss_price, take_profit_price, partial_tp_price
                )
                if adjusted_size is not None and adjusted_entry is not None and place_market_order(symbol, 'buy', adjusted_size, adjusted_entry, adjusted_sl, adjusted_tp): # Check for None values
                    trade_id = f"{symbol}-{int(time.time())}"
                    entry_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    trade_log_data = {
                        'trade_id': trade_id,
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'entry_price': adjusted_entry,
                        'side': 'long',
                        'amount': adjusted_size,
                        'stop_loss': adjusted_sl,
                        'take_profit': adjusted_tp,
                        'status': 'OPEN',
                        'close_time': None,
                        'profit_loss': None,
                        'bars_held': 0
                    }
                    log_trade_to_csv(trade_log_data)
                    symbol_data.update({
                        'in_trade': True,
                        'direction': 'long',
                        'entry_price': adjusted_entry,
                        'position_size': adjusted_size,
                        'stop_loss': adjusted_sl,
                        'take_profit': adjusted_tp,
                        'atr_value_sl': atr_value * atr_stop_loss_multiplier,
                        'trailing_stop_active': False,
                        'trailing_stop_price': None,
                        'partial_tp_price': adjusted_tp_partial,
                        'partial_tp_hit': False,
                        'current_trade_id': trade_id,
                        'entry_bar_index': len(df)
                    })
                    if MODE == 'real':
                        set_stop_loss(symbol, adjusted_sl)
                        set_take_profit(symbol, adjusted_tp)
                    logger.info(f"[TRADE OPEN - SIMULATION] ({symbol}) Long opened at {adjusted_entry}, size={adjusted_size}, SL={adjusted_sl}, TP={adjusted_tp}, Partial TP={adjusted_tp_partial}, Trade ID={trade_id}")
            # Short entry
            elif last_row['Short_Signal']: # Removed 4h trend condition for debugging, add back if needed: and four_hour_downtrend
                entry_price = last_row['close']
                atr_value = last_row['ATR']
                stop_loss_price = entry_price + (atr_value * atr_stop_loss_multiplier)
                take_profit_price = entry_price - (atr_value * risk_reward * atr_stop_loss_multiplier)
                partial_tp_price = entry_price - (atr_value * partial_tp_factor * atr_stop_loss_multiplier)
                position_size = calculate_position_size(
                    symbol_data['balance'], risk_percentage,
                    entry_price, atr_value * atr_stop_loss_multiplier, leverage
                )
                adjusted_entry, adjusted_size, adjusted_sl, adjusted_tp, adjusted_tp_partial = request_client.adjust_values(
                    symbol, entry_price, position_size, stop_loss_price, take_profit_price, partial_tp_price
                )
                if adjusted_size is not None and adjusted_entry is not None and place_market_order(symbol, 'sell', adjusted_size, adjusted_entry, adjusted_sl, adjusted_tp): # Check for None values
                    trade_id = f"{symbol}-{int(time.time())}"
                    entry_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    trade_log_data = {
                        'trade_id': trade_id,
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'entry_price': adjusted_entry,
                        'side': 'short',
                        'amount': adjusted_size,
                        'stop_loss': adjusted_sl,
                        'take_profit': adjusted_tp,
                        'status': 'OPEN',
                        'close_time': None,
                        'profit_loss': None,
                        'bars_held': 0
                    }
                    log_trade_to_csv(trade_log_data)
                    symbol_data.update({
                        'in_trade': True,
                        'direction': 'short',
                        'entry_price': adjusted_entry,
                        'position_size': adjusted_size,
                        'stop_loss': adjusted_sl,
                        'take_profit': adjusted_tp,
                        'atr_value_sl': atr_value * atr_stop_loss_multiplier,
                        'trailing_stop_active': False,
                        'trailing_stop_price': None,
                        'partial_tp_price': adjusted_tp_partial,
                        'partial_tp_hit': False,
                        'current_trade_id': trade_id,
                        'entry_bar_index': len(df)
                    })
                    if MODE == 'real':
                        set_stop_loss(symbol, adjusted_sl)
                        set_take_profit(symbol, adjusted_tp)
                    logger.info(f"[TRADE OPEN - SIMULATION] ({symbol}) Short opened at {adjusted_entry}, size={adjusted_size}, SL={adjusted_sl}, TP={adjusted_tp}, Partial TP={adjusted_tp_partial}, Trade ID={trade_id}")
    except Exception as e:
        logger.error(f"Strategy logic error in process_strategy_logic: {str(e)}") # More specific error message

def close_position(symbol, symbol_data, current_price, df):
    if MODE == 'fronttest':
        if symbol_data['in_trade']:
            side_to_close = 'sell' if symbol_data['direction'] == 'long' else 'buy'
            profit = 0
            if symbol_data['direction'] == 'long':
                profit = (current_price - symbol_data['entry_price']) * symbol_data['position_size']
            elif symbol_data['direction'] == 'short':
                profit = (symbol_data['entry_price'] - current_price) * symbol_data['position_size']
            symbol_data['balance'] += profit
            close_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            bars_held = len(df) - symbol_data['entry_bar_index']
            trade_log_data = {
                'trade_id': symbol_data['current_trade_id'],
                'symbol': symbol,
                'entry_time': symbol_data['entry_time'], # Use stored entry time
                'entry_price': symbol_data['entry_price'], # Use stored entry price
                'side': symbol_data['direction'], # Use stored side
                'amount': symbol_data['position_size'], # Use stored position size
                'stop_loss': symbol_data['stop_loss'], # Use stored stop loss
                'take_profit': symbol_data['take_profit'], # Use stored take profit
                'status': 'CLOSED',
                'close_time': close_time,
                'profit_loss': profit,
                'bars_held': bars_held
            }
            log_trade_to_csv(trade_log_data)
            global_trades.append(trade_log_data)
            symbol_data['trades'].append(trade_log_data)
            logger.info(f"[TRADE CLOSED - SIMULATION] ({symbol}) {symbol_data['direction'].capitalize()} closed at {current_price}, PNL={profit:.2f}, Balance={symbol_data['balance']:.2f}, Bars Held={bars_held}, Trade ID={symbol_data['current_trade_id']}")
            symbol_data.update({
                'in_trade': False,
                'direction': None,
                'entry_price': 0,
                'position_size': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'atr_value_sl': 0,
                'trailing_stop_active': False,
                'trailing_stop_price': None,
                'partial_tp_price': None,
                'partial_tp_hit': False,
                'current_trade_id': None,
                'entry_bar_index': 0
            })
            place_market_order(symbol, side_to_close, symbol_data['position_size'])
    else:
        try:
            if place_market_order(symbol, 'sell' if symbol_data['direction'] == 'long' else 'buy', symbol_data['position_size']):
                profit = ((symbol_data['entry_price'] - symbol_data['stop_loss']) * symbol_data['position_size']
                          if symbol_data['direction'] == 'short'
                          else (symbol_data['take_profit'] - symbol_data['entry_price']) * symbol_data['position_size'])
                symbol_data['balance'] += profit
                close_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                bars_held = len(df) - symbol_data['entry_bar_index']
                trade_log_data = {
                    'trade_id': symbol_data['current_trade_id'],
                    'symbol': symbol,
                    'entry_time': symbol_data['entry_time'], # Use stored entry time
                    'entry_price': symbol_data['entry_price'], # Use stored entry price
                    'side': symbol_data['direction'], # Use stored side
                    'amount': symbol_data['position_size'], # Use stored position size
                    'stop_loss': symbol_data['stop_loss'], # Use stored stop loss
                    'take_profit': symbol_data['take_profit'], # Use stored take profit
                    'status': 'CLOSED',
                    'close_time': close_time,
                    'profit_loss': profit,
                    'bars_held': bars_held
                }
                log_trade_to_csv(trade_log_data)
                global_trades.append(trade_log_data)
                symbol_data['trades'].append(trade_log_data)
                symbol_data.update({
                    'in_trade': False,
                    'direction': None,
                    'entry_price': 0,
                    'position_size': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'atr_value_sl': 0,
                    'trailing_stop_active': False,
                    'trailing_stop_price': None,
                    'partial_tp_price': None,
                    'partial_tp_hit': False,
                    'current_trade_id': None,
                    'entry_bar_index': 0
                })
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")

def simulate_tp_sl_check(symbol, symbol_data, current_price, df, trailing_stop_atr_multiplier=1.0, partial_tp_reduction_factor=0.5):
    if symbol_data['in_trade'] and MODE == 'fronttest':
        if symbol_data['direction'] == 'long':
            if not symbol_data['partial_tp_hit'] and symbol_data['partial_tp_price'] is not None and current_price >= symbol_data['partial_tp_price']:
                logger.info(f"[SIMULATION - PARTIAL TP HIT] ({symbol}) Long Partial TP hit at {current_price}, Partial TP={symbol_data['partial_tp_price']}")
                partial_tp_amount = symbol_data['position_size'] * partial_tp_reduction_factor
                reduced_position_size = symbol_data['position_size'] * (1 - partial_tp_reduction_factor)
                profit_partial_tp = (current_price - symbol_data['entry_price']) * partial_tp_amount
                symbol_data['balance'] += profit_partial_tp
                symbol_data['position_size'] = reduced_position_size
                symbol_data['partial_tp_hit'] = True
                symbol_data['trailing_stop_active'] = True
                symbol_data['trailing_stop_price'] = current_price - (symbol_data['atr_value_sl'] * trailing_stop_atr_multiplier)
                logger.info(f"[SIMULATION - PARTIAL TP ACTION] ({symbol}) Position reduced to {symbol_data['position_size']}, Trailing stop activated at {symbol_data['trailing_stop_price']}")
            if symbol_data['trailing_stop_active']:
                trailing_stop_level = symbol_data['trailing_stop_price']
                if current_price <= trailing_stop_level:
                    logger.info(f"[SIMULATION - TRAILING STOP HIT] ({symbol}) Long Trailing Stop hit at {current_price}, TS Level={trailing_stop_level}")
                    close_position(symbol, symbol_data, current_price, df)
                else:
                    new_trailing_stop = max(trailing_stop_level, current_price - (symbol_data['atr_value_sl'] * trailing_stop_atr_multiplier))
                    symbol_data['trailing_stop_price'] = new_trailing_stop
            if not symbol_data['trailing_stop_active']:
                if current_price <= symbol_data['stop_loss']:
                    logger.info(f"[SIMULATION - SL HIT] ({symbol}) Long SL hit at {current_price}, SL={symbol_data['stop_loss']}")
                    close_position(symbol, symbol_data, current_price, df)
                elif current_price >= symbol_data['take_profit']:
                    logger.info(f"[SIMULATION - TP HIT] ({symbol}) Long TP hit at {current_price}, TP={symbol_data['take_profit']}")
                    close_position(symbol, symbol_data, current_price, df)
        elif symbol_data['direction'] == 'short':
            if not symbol_data['partial_tp_hit'] and symbol_data['partial_tp_price'] is not None and current_price <= symbol_data['partial_tp_price']:
                logger.info(f"[SIMULATION - PARTIAL TP HIT] ({symbol}) Short Partial TP hit at {current_price}, Partial TP={symbol_data['partial_tp_price']}")
                partial_tp_amount = symbol_data['position_size'] * partial_tp_reduction_factor
                reduced_position_size = symbol_data['position_size'] * (1 - partial_tp_reduction_factor)
                profit_partial_tp = (symbol_data['entry_price'] - current_price) * partial_tp_amount
                symbol_data['balance'] += profit_partial_tp
                symbol_data['position_size'] = reduced_position_size
                symbol_data['partial_tp_hit'] = True
                symbol_data['trailing_stop_active'] = True
                symbol_data['trailing_stop_price'] = current_price + (symbol_data['atr_value_sl'] * trailing_stop_atr_multiplier)
                logger.info(f"[SIMULATION - PARTIAL TP ACTION] ({symbol}) Position reduced to {symbol_data['position_size']}, Trailing stop activated at {symbol_data['trailing_stop_price']}")
            if symbol_data['trailing_stop_active']:
                trailing_stop_level = symbol_data['trailing_stop_price']
                if current_price >= trailing_stop_level:
                    logger.info(f"[SIMULATION - TRAILING STOP HIT] ({symbol}) Short Trailing Stop hit at {current_price}, TS Level={trailing_stop_level}")
                    close_position(symbol, symbol_data, current_price, df)
                else:
                    new_trailing_stop = min(trailing_stop_level, current_price + (symbol_data['atr_value_sl'] * trailing_stop_atr_multiplier))
                    symbol_data['trailing_stop_price'] = new_trailing_stop
            if not symbol_data['trailing_stop_active']:
                if current_price >= symbol_data['stop_loss']:
                    logger.info(f"[SIMULATION - SL HIT] ({symbol}) Short SL hit at {current_price}, SL={symbol_data['stop_loss']}")
                    close_position(symbol, symbol_data, current_price, df)
                elif current_price <= symbol_data['take_profit']:
                    logger.info(f"[SIMULATION - TP HIT] ({symbol}) Short TP hit at {current_price}, TP={symbol_data['take_profit']}")
                    close_position(symbol, symbol_data, current_price, df)

def run_strategy(config):
    symbols = config['symbols']
    timeframe = config['timeframe']
    timeframe_4h = '4hour'
    risk_percent = config['risk_percentage']
    risk_reward = config['risk_reward']
    leverage = config['leverage']
    atr_stop_loss_multiplier = config.get('atr_stop_loss_multiplier', 2.0)
    partial_tp_factor = config.get('partial_tp_factor', 2.0)
    trailing_stop_atr_multiplier = config.get('trailing_stop_atr_multiplier', 1.0)
    lookback_window_touch_200 = config.get('lookback_window_touch_200', 20)
    price_distance_multiplier = config.get('price_distance_multiplier', 1.5)
    correction_lookback = config.get('correction_lookback', 5)

    symbol_data = {sym: {
        'in_trade': False,
        'direction': None,
        'entry_price': 0,
        'position_size': 0,
        'stop_loss': 0,
        'take_profit': 0,
        'atr_value_sl': 0,
        'trailing_stop_active': False,
        'trailing_stop_price': None,
        'partial_tp_price': None,
        'partial_tp_hit': False,
        'balance': config['initial_capital'],
        'trades': [],
        'current_trade_id': None,
        'entry_bar_index': 0,
        'entry_time': None # Store entry time
    } for sym in symbols}

    initialize_trade_log_csv()
    initialize_analytics_csv()
    initialize_condition_log_csv() # Initialize condition log CSV

    if MODE == 'real':
        for symbol in symbols:
            if not set_leverage(symbol, leverage):
                logger.error(f"Failed to set leverage for {symbol}. Exiting.")
                return

    analytics_interval = 60 * 60  # 1 hour
    last_analytics_time = time.time()

    while True:
        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=1000)
                if df.empty:
                    continue

                # --- DEBUGGING PRINTS FOR OHLCV DATA ---
                # print(f"--- OHLCV Data for {symbol} ({timeframe}) ---")
                # print(df.tail())

                df_4h = fetch_ohlcv(symbol=symbol, timeframe=timeframe_4h, limit=1000)
                if df_4h.empty:
                    print(f"Warning: 4h data empty for {symbol}") # Just warn, don't skip main logic
                else:
                    # --- DEBUGGING PRINTS FOR 4H OHLCV DATA ---
                    # print(f"--- 4H OHLCV Data for {symbol} ({timeframe_4h}) ---")
                    # print(df_4h.tail())
                    ...

                if MODE == 'real':
                    current_balance = get_futures_balance_real_mode()
                    if current_balance is not None:
                        symbol_data[symbol]['balance'] = current_balance
                    else:
                        logger.error(f"Could not update balance for {symbol}. Skipping strategy logic.")
                        continue
                process_strategy_logic(
                    symbol=symbol,
                    df=df,
                    df_4h=df_4h,
                    symbol_data=symbol_data[symbol],
                    risk_percentage=risk_percent,
                    risk_reward=risk_reward,
                    leverage=leverage,
                    atr_stop_loss_multiplier=atr_stop_loss_multiplier,
                    partial_tp_factor=partial_tp_factor,
                    trailing_stop_atr_multiplier=trailing_stop_atr_multiplier,
                    lookback_window_touch_200=lookback_window_touch_200,
                    price_distance_multiplier=price_distance_multiplier,
                    correction_lookback=correction_lookback
                )
                simulate_tp_sl_check(
                    symbol=symbol,
                    symbol_data=symbol_data[symbol],
                    current_price=df['close'].iloc[-1],
                    df=df,
                    trailing_stop_atr_multiplier=trailing_stop_atr_multiplier
                )
                if time.time() - last_analytics_time >= analytics_interval:
                    analytics = calculate_trade_analytics(global_trades)
                    analytics_data = {
                        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_trades': analytics.get('total_trades', 0),
                        'win_rate_percent': analytics.get('win_rate_percent', 0),
                        'loss_rate_percent': analytics.get('loss_rate_percent', 0),
                        'profit_factor': analytics.get('profit_factor', 0),
                        'average_win': analytics.get('average_win', 0),
                        'average_loss': analytics.get('average_loss', 0),
                        'net_profit': analytics.get('net_profit', 0)
                    }
                    log_analytics_to_csv(analytics_data)
                    logger.info(f"[GLOBAL ANALYTICS] - {analytics_data}")
                    last_analytics_time = time.time()
                time.sleep(config['request_delay'])
            except Exception as e:
                logger.error(f"Main loop error: {str(e)}")
                time.sleep(10)

if __name__ == "__main__":
    config = {
        "symbols": ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOGEUSDT"], # Reduced symbols for easier debugging
        "timeframe": "5min", # Reduced timeframe for faster iteration
        "initial_capital": 100,
        "risk_percentage": 0.01,
        "risk_reward": 3,
        "leverage": 20,
        "request_delay": 20,
        "mode": "fronttest",
        "atr_stop_loss_multiplier": 2.0,
        "partial_tp_factor": 2.0,
        "trailing_stop_atr_multiplier": 1.0,
        "lookback_window_touch_200": 20,
        "price_distance_multiplier": 1.5,
        "correction_lookback": 5
    }
    if config["mode"] == "real":
        logger.info("Starting LIVE TRADING with Enhanced Arty's Strategy + Correction Filter (Global Analytics)")
        logger.warning("Ensure proper risk management settings and thoroughly backtest before live trading!")
    elif config["mode"] == "fronttest":
        logger.info("Starting FRONTTEST SIMULATION with Enhanced Arty's Strategy + Correction Filter (Global Analytics)")
        run_strategy(config)