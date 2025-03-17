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
MODE = 'fronttest'  # or 'fronttest'
COINEX_ACCESS_ID = "57E8A3627B2745C2A58E2A9ACA20275B"
COINEX_SECRET_KEY = "3776028B7005930ED6BCA650541ADC492EEFA514743D750F"
LOG_FILE_PATH = "trading_bot_real_BB.log"  # Updated log file name
LOG_LEVEL = logging.INFO
TRADE_LOG_CSV_PATH = "trade_log_BB_strategy.csv"  # Trade log CSV file
ANALYTICS_CSV_PATH = "trade_analytics_BB_strategy.csv"  # Analytics CSV file
CONDITION_LOG_CSV_PATH = "condition_log_BB_strategy.csv"  # New CSV path for condition logs


# --- Global Trades List for Aggregated Analytics ---
global_trades = []  # All closed trade logs from every symbol

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
        self.market_info = {}  # Initialize market_info cache

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

def get_historical_klines(symbol: str, interval: str = '1min', limit: int = 100) -> Optional[List[List[Any]]]:
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
        order_path = "/futures/order"
        order_url = f"{request_client.api_url}{order_path}"
        order_data = {
        "market": symbol,
        "market_type": "FUTURES",
        "side": side,
        "type": 'market',
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

    leverage_path = "/futures/adjust-position-leverage"
    leverage_url = f"{request_client.api_url}{leverage_path}"
    leverage_data = {
        "market": symbol,
        "market_type": "FUTURES",
        "margin_mode": "isolated",
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

    tp_path = "/futures/set-position-take-profit"
    tp_url = f"{request_client.api_url}{tp_path}"
    tp_data = {
        "market": symbol,
        "market_type": "FUTURES",
        "take_profit_type": "latest_price",
        "take_profit_price": str(tp_price)
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

    sl_path = "/futures/set-position-stop-loss"
    sl_url = f"{request_client.api_url}{sl_path}"
    sl_data = {
        "market": symbol,
        "market_type": "FUTURES",
        "stop_loss_type": "latest_price",
        "stop_loss_price": str(sl_price)
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
    request_path = "/assets/futures/balance"
    url = f"{request_client.api_url}{request_path}"
    try:
        response = request_client.request("GET", url)
        data = response.json()
        if data.get("code") == 0:
            balance_data = data.get("data") or []
            for asset in balance_data:
                if asset.get("ccy") == "USDT":
                    available_balance = float(asset.get("available", 0))
                    # logger.info(f"[REAL BALANCE FETCH] Fetched USDT balance: {available_balance}")
                    return available_balance
            logger.warning("[REAL BALANCE FETCH] USDT balance not found in API response, returning 0.")
            return 0.0
        else:
            logger.error(f"[REAL BALANCE FETCH ERROR] Error fetching futures balance: {data}")
            return None
    except Exception as e:
        logger.error(f"[REAL BALANCE FETCH EXCEPTION] Exception fetching futures balance: {e}")
        return None

# --- Bollinger Bands Strategy Core Functions ---
def calculate_bollinger_bands(df, window=20, multiplier=2):
    df['MA'] = df['close'].rolling(window=window).mean()
    df['STD'] = df['close'].rolling(window=window).std()
    df['upper'] = df['MA'] + multiplier * df['STD']
    df['lower'] = df['MA'] - multiplier * df['STD']
    return df

def check_long_entry_signal(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    Long signal:
      - Previous candle is red and closed below the lower Bollinger Band.
      - Current candle is green, opens below the lower band, and closes within the bands.
    Returns:
      - tuple: (signal_bool, condition_dict)
    """
    signal = False
    condition_dict = {}
    if len(df) < 2:
        return signal, condition_dict

    df_bb = calculate_bollinger_bands(df.copy(), window=20, multiplier=2)
    candle1 = df_bb.iloc[-2]
    candle2 = df_bb.iloc[-1]

    cond1 = (candle1['close'] < candle1['open']) and (candle1['close'] < candle1['lower'])
    cond2 = (candle2['close'] > candle2['open']) and (candle2['open'] < candle2['lower']) and (candle2['lower'] <= candle2['close'] <= candle2['upper'])
    signal = cond1 and cond2

    condition_dict = {
        'timestamp': candle2.name.isoformat(),
        'symbol': df.name if hasattr(df, 'name') else 'UNKNOWN',  # Symbol name if available
        'close_price': candle2['close'],
        'ma': candle2['MA'],
        'upper_bb': candle2['upper'],
        'lower_bb': candle2['lower'],
        'std_dev': candle2['STD'],
        'long_cond1': cond1,
        'long_cond2': cond2,
        'long_signal': signal
    }
    return signal, condition_dict

def check_short_entry_signal(df: pd.DataFrame) -> tuple[bool, dict]:
    """
    Short signal:
      - Previous candle is green and closed above the upper Bollinger Band.
      - Current candle is red, opens above the upper band, and closes within the bands.
    Returns:
      - tuple: (signal_bool, condition_dict)
    """
    signal = False
    condition_dict = {}
    if len(df) < 2:
        return signal, condition_dict

    df_bb = calculate_bollinger_bands(df.copy(), window=20, multiplier=2)
    candle1 = df_bb.iloc[-2]
    candle2 = df_bb.iloc[-1]

    cond1 = (candle1['close'] > candle1['open']) and (candle1['close'] > candle1['upper'])
    cond2 = (candle2['close'] < candle2['open']) and (candle2['open'] > candle2['upper']) and (candle2['lower'] <= candle2['close'] <= candle2['upper'])
    signal = cond1 and cond2

    condition_dict = {
        'timestamp': candle2.name.isoformat(),
        'symbol': df.name if hasattr(df, 'name') else 'UNKNOWN', # Symbol name if available
        'close_price': candle2['close'],
        'ma': candle2['MA'],
        'upper_bb': candle2['upper'],
        'lower_bb': candle2['lower'],
        'std_dev': candle2['STD'],
        'short_cond1': cond1,
        'short_cond2': cond2,
        'short_signal': signal
    }
    return signal, condition_dict

def calculate_smma(series, window):
    """
    Calculates Smoothed Moving Average (SMMA) for a given series.
    """
    smma = np.full(len(series), np.nan)
    first_smma = series[:window].mean()
    smma[window - 1] = first_smma
    for i in range(window, len(series)):
        smma[i] = (smma[i - 1] * (window - 1) + series.iloc[i]) / window
    return pd.Series(smma, index=series.index)

def get_current_timeframe_trend(df):
    """
    Determine the trend on the current timeframe (e.g., 15min) based on a 50-period Smoothed Moving Average (SMMA).
    Returns 'bullish', 'bearish', or 'neutral'.
    Trend is bullish if price is above SMMA50, bearish if below.
    """
    if df is None or df.empty or len(df) < 50: # Need at least 50 periods for SMMA
        return 'neutral'  # Not enough data to determine trend

    smma_period = 50 # Using 50 period SMMA as requested
    # Calculate SMMA using the custom function
    df['SMMA_50'] = calculate_smma(df['close'], smma_period)

    current_candle = df.iloc[-1]
    current_smma = current_candle['SMMA_50']
    current_close = current_candle['close']

    if current_close > current_smma:
        return 'bullish'  # Price is above 50 SMMA, bullish trend
    elif current_close < current_smma:
        return 'bearish'  # Price is below 50 SMMA, bearish trend
    else:
        return 'neutral'  # Price is around 50 SMMA, neutral trend


def initialize_condition_log_csv():
    if not os.path.exists(CONDITION_LOG_CSV_PATH):
        with open(CONDITION_LOG_CSV_PATH, mode='w', newline='') as csvfile:
            condition_writer = csv.writer(csvfile)
            condition_writer.writerow([
                'Timestamp', 'Symbol', 'Close_Price', 'MA', 'Upper_BB', 'Lower_BB', 'STD_Dev',
                'Long_Cond1', 'Long_Cond2', 'Long_Signal',
                'Short_Cond1', 'Short_Cond2', 'Short_Signal',
                'Current_TF_Trend' # Changed Log Column Name
            ])

def log_condition_to_csv(condition_data, current_tf_trend): # Changed parameter name
    with open(CONDITION_LOG_CSV_PATH, mode='a', newline='') as csvfile:
        condition_writer = csv.writer(csvfile)
        condition_writer.writerow([
            condition_data.get('timestamp'),
            condition_data.get('symbol'),
            condition_data.get('close_price'),
            condition_data.get('ma'),
            condition_data.get('upper_bb'),
            condition_data.get('lower_bb'),
            condition_data.get('std_dev'),
            condition_data.get('long_cond1', ''),  # Handle cases where keys might not exist
            condition_data.get('long_cond2', ''),
            condition_data.get('long_signal', ''),
            condition_data.get('short_cond1', ''),
            condition_data.get('short_cond2', ''),
            condition_data.get('short_signal', ''),
            current_tf_trend # Changed variable name
        ])


def process_strategy_logic(symbol, df, symbol_data, risk_percentage, risk_reward, leverage): # Removed df_4h
    """
    Implements the Bollinger Bands strategy with trend filter (SMMA 50 on current timeframe),
    SL based on last 5 candles, and TP validation against Bollinger Bands.
    """
    try:
        # Determine trend using SMMA 50 on current timeframe (e.g., 15min)
        current_tf_trend = get_current_timeframe_trend(df) # Using df directly for trend

        # Check for long entry signal
        long_signal, long_condition_data = check_long_entry_signal(df)
        log_condition_to_csv(long_condition_data, current_tf_trend) # Log long condition in every iteration

        if not symbol_data['in_trade']: # Only check for entry if not already in a trade
            if current_tf_trend in ['bullish', 'neutral'] and long_signal: # Consider neutral trend for long entries as per original logic


                signal_candle = df.iloc[-1]
                entry_price = signal_candle['close']

                # Calculate SL as lowest low of last 5 candles
                if len(df) >= 5:
                    stop_loss = df['low'].iloc[-5:].min()
                else:
                    stop_loss = signal_candle['low']

                risk = entry_price - stop_loss
                take_profit = entry_price + risk_reward * risk

                # --- NEW CONDITION: Check TP against upper band ---
                df_bb = calculate_bollinger_bands(df.copy(), window=20, multiplier=2)
                current_upper_band = df_bb['upper'].iloc[-1]
                if take_profit > current_upper_band:
                    logger.info(f"[{symbol}] Long entry skipped: TP {take_profit:.2f} > Upper BB {current_upper_band:.2f}")
                    return
                # ------------------------------------------------

                risk_amount = symbol_data['balance'] * risk_percentage
                position_size = risk_amount / risk if risk > 0 else 0

                adjusted_entry, adjusted_size, adjusted_sl, adjusted_tp, _ = request_client.adjust_values(
                    symbol, entry_price, position_size, stop_loss, take_profit
                )

                if adjusted_size is None or adjusted_size <= 0:
                    logger.error(f"[{symbol}] Invalid position size calculated for long. Skipping trade.")
                    return

                if place_market_order(symbol, 'buy', adjusted_size, adjusted_entry, adjusted_sl, adjusted_tp):
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
                        'bars_held': 0,
                    }
                    log_trade_to_csv(trade_log_data)
                    symbol_data.update({
                        'in_trade': True,
                        'direction': 'long',
                        'entry_price': adjusted_entry,
                        'position_size': adjusted_size,
                        'stop_loss': adjusted_sl,
                        'take_profit': adjusted_tp,
                        'current_trade_id': trade_id,
                        'entry_bar_index': len(df)
                    })
                    if MODE == 'real':
                        set_stop_loss(symbol, adjusted_sl)
                        set_take_profit(symbol, adjusted_tp)
                    logger.info(f"[TRADE OPEN] ({symbol}) Long opened at {adjusted_entry}, size={adjusted_size}, SL={adjusted_sl}, TP={adjusted_tp}, Trade ID={trade_id}, Current TF Trend={current_tf_trend}") # Changed log message

            # Check for short entry signal
            short_signal, short_condition_data = check_short_entry_signal(df)
            log_condition_to_csv(short_condition_data, current_tf_trend) # Log short condition in every iteration
            if current_tf_trend in ['bearish', 'neutral'] and short_signal and not symbol_data['in_trade']: # Consider neutral trend for short entries as per original logic


                signal_candle = df.iloc[-1]
                entry_price = signal_candle['close']

                # Calculate SL as highest high of last 5 candles
                if len(df) >= 5:
                    stop_loss = df['high'].iloc[-5:].max()
                else:
                    stop_loss = signal_candle['high']

                risk = stop_loss - entry_price
                take_profit = entry_price - risk_reward * risk

                # --- NEW CONDITION: Check TP against lower band ---
                df_bb = calculate_bollinger_bands(df.copy(), window=20, multiplier=2)
                current_lower_band = df_bb['lower'].iloc[-1]
                if take_profit < current_lower_band:
                    logger.info(f"[{symbol}] Short entry skipped: TP {take_profit:.2f} < Lower BB {current_lower_band:.2f}")
                    return
                # ------------------------------------------------

                risk_amount = symbol_data['balance'] * risk_percentage
                position_size = risk_amount / risk if risk > 0 else 0

                adjusted_entry, adjusted_size, adjusted_sl, adjusted_tp, _ = request_client.adjust_values(
                    symbol, entry_price, position_size, stop_loss, take_profit
                )

                if adjusted_size is None or adjusted_size <= 0:
                    logger.error(f"[{symbol}] Invalid position size calculated for short. Skipping trade.")
                    return

                if place_market_order(symbol, 'sell', adjusted_size, adjusted_entry, adjusted_sl, adjusted_tp):
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
                        'bars_held': 0,
                    }
                    log_trade_to_csv(trade_log_data)
                    symbol_data.update({
                        'in_trade': True,
                        'direction': 'short',
                        'entry_price': adjusted_entry,
                        'position_size': adjusted_size,
                        'stop_loss': adjusted_sl,
                        'take_profit': adjusted_tp,
                        'current_trade_id': trade_id,
                        'entry_bar_index': len(df)
                    })
                    if MODE == 'real':
                        set_stop_loss(symbol, adjusted_sl)
                        set_take_profit(symbol, adjusted_tp)
                    logger.info(f"[TRADE OPEN] ({symbol}) Short opened at {adjusted_entry}, size={adjusted_size}, SL={adjusted_sl}, TP={adjusted_tp}, Trade ID={trade_id}, Current TF Trend={current_tf_trend}") # Changed log message

    except Exception as e:
        logger.error(f"Strategy logic error: {str(e)}")

def close_position(symbol, symbol_data, current_price, df):
    if MODE == 'fronttest':
        if symbol_data['in_trade']:
            side_to_close = 'sell' if symbol_data['direction'] == 'long' else 'buy'
            if symbol_data['direction'] == 'long':
                profit = (current_price - symbol_data['entry_price']) * symbol_data['position_size']
            else:
                profit = (symbol_data['entry_price'] - current_price) * symbol_data['position_size']
            symbol_data['balance'] += profit
            close_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            bars_held = len(df) - symbol_data['entry_bar_index']
            trade_log_data = {
                'trade_id': symbol_data['current_trade_id'],
                'symbol': symbol,
                'entry_time': symbol_data['entry_time'], #Keep entry time from symbol_data if you want to log it
                'entry_price': symbol_data['entry_price'], #Keep entry price
                'side': symbol_data['direction'], #Keep direction
                'amount': symbol_data['position_size'], #Keep position size
                'stop_loss': symbol_data['stop_loss'], #Keep stop loss
                'take_profit': symbol_data['take_profit'], #Keep take profit
                'status': 'CLOSED',
                'close_time': close_time,
                'profit_loss': profit,
                'bars_held': bars_held
            }
            log_trade_to_csv(trade_log_data)
            # Append closed trade to global trades for aggregated analytics
            global_trades.append(trade_log_data)
            # Also update per-symbol trade history if needed
            symbol_data['trades'].append(trade_log_data)
            logger.info(f"[TRADE CLOSED] ({symbol}) {symbol_data['direction'].capitalize()} closed at {current_price}, PNL={profit:.2f}, Balance={symbol_data['balance']:.2f}, Bars Held={bars_held}, Trade ID={symbol_data['current_trade_id']}")
            symbol_data.update({
                'in_trade': False,
                'direction': None,
                'entry_price': 0,
                'position_size': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'current_trade_id': None,
                'entry_bar_index': 0
            })
            place_market_order(symbol, side_to_close, symbol_data['position_size'])
    else:
        try:
            if place_market_order(symbol, 'sell' if symbol_data['direction'] == 'long' else 'buy', symbol_data['position_size']):
                profit = (symbol_data['take_profit'] - symbol_data['entry_price']) * symbol_data['position_size']  # Conceptual profit for live mode
                symbol_data['balance'] += profit
                close_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                bars_held = len(df) - symbol_data['entry_bar_index']
                trade_log_data = {
                    'trade_id': symbol_data['current_trade_id'],
                    'symbol': symbol,
                    'entry_time': symbol_data['entry_time'], #Keep entry time from symbol_data if you want to log it
                    'entry_price': symbol_data['entry_price'], #Keep entry price
                    'side': symbol_data['direction'], #Keep direction
                    'amount': symbol_data['position_size'], #Keep position size
                    'stop_loss': symbol_data['stop_loss'], #Keep stop loss
                    'take_profit': symbol_data['take_profit'], #Keep take profit
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
                    'current_trade_id': None,
                    'entry_bar_index': 0
                })
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")

def simulate_tp_sl_check(symbol, symbol_data, current_price, df):
    """For simulation, check if current price has reached TP or SL for either long or short trades."""
    if symbol_data['in_trade'] and MODE == 'fronttest':
        if symbol_data['direction'] == 'long':
            if current_price <= symbol_data['stop_loss']:
                logger.info(f"[SIMULATION - SL HIT] ({symbol}) Long SL hit at {current_price}")
                close_position(symbol, symbol_data, current_price, df)
            elif current_price >= symbol_data['take_profit']:
                logger.info(f"[SIMULATION - TP HIT] ({symbol}) Long TP hit at {current_price}")
                close_position(symbol, symbol_data, current_price, df)
        elif symbol_data['direction'] == 'short':
            if current_price >= symbol_data['stop_loss']:
                logger.info(f"[SIMULATION - SL HIT] ({symbol}) Short SL hit at {current_price}")
                close_position(symbol, symbol_data, current_price, df)
            elif current_price <= symbol_data['take_profit']:
                logger.info(f"[SIMULATION - TP HIT] ({symbol}) Short TP hit at {current_price}")
                close_position(symbol, symbol_data, current_price, df)

# --- CSV Trade Logging Functions (already defined above) ---
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

# --- Trade Analytics Functions ---
def initialize_analytics_csv():
    # Here, we initialize a CSV for global aggregated analytics
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

# --- Modified Main Execution Flow ---
def run_strategy(config):
    symbols = config['symbols']
    timeframe = config['timeframe']          # Lower timeframe (e.g. "15min")
    risk_percent = config['risk_percentage']
    risk_reward = config['risk_reward']
    leverage = config['leverage']

    # Initialize per-symbol data
    symbol_data = {sym: {
        'in_trade': False,
        'direction': None,
        'entry_price': 0,
        'position_size': 0,
        'stop_loss': 0,
        'take_profit': 0,
        'balance': config['initial_capital'],
        'trades': [],
        'current_trade_id': None,
        'entry_bar_index': 0
    } for sym in symbols}

    initialize_trade_log_csv()
    initialize_analytics_csv()
    initialize_condition_log_csv() # Initialize condition log CSV


    if MODE == 'real':
        for symbol in symbols:
            if not set_leverage(symbol, leverage):
                logger.error(f"Failed to set leverage for {symbol}. Exiting.")
                return

    analytics_interval = 60 * 60  # 1 hour interval for analytics logging
    last_analytics_time = time.time()

    while True:
        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=1000) # Fetching data for trading timeframe (e.g., 15min)
                if df.empty:
                    continue
                df.name = symbol # Set symbol name to df for logging

                # df_4h = fetch_ohlcv(symbol=symbol, timeframe=timeframe_4h, limit=250) # No longer needed for trend

                if MODE == 'real':
                    current_balance = get_futures_balance_real_mode()
                    if current_balance is not None:
                        symbol_data[symbol]['balance'] = current_balance
                    else:
                        logger.error(f"Could not update balance for {symbol}. Skipping.")
                        continue

                process_strategy_logic(symbol, df, symbol_data[symbol], risk_percent, risk_reward, leverage) # Removed df_4h from parameter

                simulate_tp_sl_check(symbol, symbol_data[symbol], df['close'].iloc[-1], df)

                # Instead of per-symbol analytics, aggregate global trades for analytics.
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
                logger.error(f"Main loop error for {symbol}: {str(e)}")
                time.sleep(10)

if __name__ == "__main__":
    config = {
        "symbols": ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","ADAUSDT","DOGEUSDT"], # Reduced symbols for easier debugging
        "timeframe": "15min",  # Lower timeframe for trading signal and trend now
        "initial_capital": 100,
        "risk_percentage": 0.01,  # 1% risk per trade
        "risk_reward": 3,         # 3:1 risk-reward ratio
        "leverage": 20,
        "request_delay": 20,
        "mode": "fronttest"  # Set to 'real' for live trading
    }

    if config["mode"] == "real":
        logger.info("Starting LIVE TRADING with BB Strategy (Long & Short) - Global Analytics")
        logger.warning("Ensure proper risk management settings and thoroughly backtest before live trading!")
    elif config["mode"] == "fronttest":
        logger.info("Starting FRONTTEST SIMULATION with BB Strategy (Long & Short) - Global Analytics")

    run_strategy(config)