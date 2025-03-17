import pandas as pd
import hashlib
import hmac
import time
import requests
from urllib.parse import urlparse, urlencode
from typing import List, Any, Optional, Dict
import datetime
import os
import logging
import json
import math
import psutil

# --- Configuration ---
MODE = 'fronttest'  # Set to 'real' for live trading, 'backtest' or 'fronttest' for simulation
COINEX_ACCESS_ID="57E8A3627B2745C2A58E2A9ACA20275B" # Replace with your actual Access ID if using CoinEx data source
COINEX_SECRET_KEY="3776028B7005930ED6BCA650541ADC492EEFA514743D750F" # Replace with your actual Secret Key if using CoinEx data source
LOG_FILE_PATH = "trading_bot_real.log"
LOG_LEVEL = logging.INFO

# --- Logging Configuration ---
log_file_path = LOG_FILE_PATH
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(log_file_path)
                    ])
logger = logging.getLogger(__name__)

# --- CoinEx API Client with Precision and Rate Limit Handling ---
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
        self.api_url = "https://api.coinex.com/v2"
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
        max_retries = 3
        retry_delay = 1  # seconds, initial delay
        for attempt in range(max_retries):
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
                if response.status_code == 429:  # Rate Limit Error
                    logger.warning(f"Rate limit hit. Retry attempt {attempt + 1}/{max_retries} in {retry_delay} seconds.")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"HTTP error on attempt {attempt + 1}/{max_retries}: {e}")
                    raise  # Re-raise for outer handling if not rate limit
        logger.error(f"Failed request after {max_retries} attempts.")
        raise Exception("Max retries reached for API request") # Raise exception if max retries exceeded

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
                response = self.request("GET", f"{market_url}?market={market}")
                data = response.json()
                if data.get("code") == 0 and data.get("data"):
                    for m in data["data"]:
                        name = m.get("market")
                        if name:
                            self.market_info[name.upper()] = m
                            if not self.use_real_data_in_simulation():
                                if 'last_price' not in self.market_info[name.upper()]:
                                    self.market_info[name.upper()]['last_price'] = 30000.0
                else:
                    self.log_error(f"Error fetching market info: {data}")
                    return None
            except requests.exceptions.RequestException as e:
                self.log_error(f"Error fetching market info: {e}")
                return None
            except Exception as e:
                self.log_error(f"General error fetching market info: {e}")
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
                time.sleep(backoff_factor * (2 ** attempt))
        except Exception as e:
            time.sleep(backoff_factor * (2 ** attempt))
    logger.error(f"Failed to fetch klines for {symbol} after {max_retries} attempts")
    return None

def fetch_ohlcv(
    symbol,
    timeframe,
    limit=100,
    mode="backtest",
    data_source="coinex",
    csv_file_path_higher="{symbol}_30m_data.csv",
    csv_file_path_lower="{symbol}_1m_data.csv",
    higher_timeframe=None,
    lower_timeframe=None,
):
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
            period_seconds = higher_timeframe
        else:
            period_seconds = lower_timeframe

        ohlcv_data = get_historical_klines(symbol, period_seconds, limit=limit)
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
def place_market_order(symbol, side, amount):
    if MODE != 'real':
        logger.info(f"[SIMULATED ORDER - Not in real mode] ({symbol}) Placing market order: {side} {amount} {symbol}")
        return {'order_id': 'simulated_order_id'}

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
                    logger.info(f"[REAL BALANCE FETCH] Fetched USDT balance: {available_balance}")
                    return available_balance
            logger.warning("[REAL BALANCE FETCH] USDT balance not found in API response, returning 0.")
            return 0.0
        else:
            logger.error(f"[REAL BALANCE FETCH ERROR] Error fetching futures balance: {data}")
            return None
    except Exception as e:
        logger.error(f"[REAL BALANCE FETCH EXCEPTION] Exception fetching futures balance: {e}")
        return None


# --- Strategy Core ---
def run_strategy(
    symbols,
    higher_timeframe,
    lower_timeframe,
    initial_capital,
    risk_reward_ratio,
    mode="backtest",
    strategy_mode="long",
    risk_percentage_per_trade=0.01,
    leverage_value=3,
    data_source="coinex",
    csv_file_path_higher=None,
    csv_file_path_lower=None,
    historical_data_limit=1000,
    request_delay=0.5,
):
    if mode == "real":
        logger.info(f"===== REAL TRADING MODE ACTIVATED! =====")
        logger.warning(f"===== LIVE TRADING - EXERCISE EXTREME CAUTION! =====")
        logger.warning(f"===== DOUBLE CHECK API KEYS AND PERMISSIONS! =====") # Added Warning
        if MODE != 'real':
            logger.error("Trading mode in config.ini is NOT set to 'real', but 'real' mode was requested in run_strategy. Exiting for safety.")
            return

    logger.info(f"Running Strategy for symbols: {symbols}, Strategy: {strategy_mode}, "
                f"Higher TF={higher_timeframe}, Lower TF={lower_timeframe}, Leverage: {leverage_value}x, "
                f"Data Source: {data_source.upper()}, Mode: {mode.upper()}")

    symbol_data_all = {}

    for symbol in symbols:
        symbol_data = {
            'in_trade': False,
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'trade_direction': None,
            'position_size_leveraged': 0,
            'balance': initial_capital,
            'trades': []
        }
        symbol_data_all[symbol] = symbol_data

        if mode == "backtest":
            logger.info(f"\n--- Backtesting Symbol: {symbol} ---")
        elif mode == "fronttest":
            ...
        elif mode == "real":
            logger.info(f"\n--- Real Trading Symbol: {symbol} ---")

    if mode in ["backtest", "fronttest"]:
        for symbol in symbols:
            df_higher_data = fetch_ohlcv(
                symbol, "higher", limit=historical_data_limit, mode=mode,
                data_source=data_source,
                csv_file_path_higher=csv_file_path_higher,
                csv_file_path_lower=csv_file_path_lower,
                higher_timeframe=higher_timeframe,
                lower_timeframe=lower_timeframe,
            )
            df_lower_data = fetch_ohlcv(
                symbol, "lower", limit=historical_data_limit, mode=mode,
                data_source=data_source,
                csv_file_path_higher=csv_file_path_higher,
                csv_file_path_lower=csv_file_path_lower,
                higher_timeframe=higher_timeframe,
                lower_timeframe=lower_timeframe,
            )

            if df_higher_data.empty or df_lower_data.empty:
                logger.error(f"Error fetching data for {symbol} in {mode} mode. Skipping.")
                continue

            for i in range(len(df_higher_data)):
                current_higher_candle = df_higher_data.iloc[i]
                process_strategy_logic(symbol, current_higher_candle, df_higher_data, df_lower_data, symbol_data_all, strategy_mode, risk_reward_ratio, risk_percentage_per_trade, leverage_value, mode)

            if mode == "backtest":
                wins = 0
                losses = 0
                total_profit = 0
                for trade in symbol_data_all[symbol]['trades']:
                    if trade['status'] == 'take_profit':
                        wins += 1
                    elif trade['status'] == 'stopped_out':
                        losses += 1
                    if 'profit' in trade:
                        total_profit += trade['profit']

                win_rate_percentage = (wins / len(symbol_data_all[symbol]['trades'])) * 100 if symbol_data_all[symbol]['trades'] else 0

                logger.info(f"\n--- Backtesting Results for Symbol: {symbol} ---")
                logger.info(f"Strategy Mode: {strategy_mode}, Higher TF={higher_timeframe}, Lower TF={lower_timeframe}, "
                            f"Leverage: {leverage_value}x, Data Source: {data_source.upper()}")
                logger.info(f"Initial Capital: {initial_capital:.2f}, Final Balance: {symbol_data_all[symbol]['balance']:.2f}, "
                            f"Total Profit: {total_profit:.2f}")
                logger.info(f"Total Trades: {len(symbol_data_all[symbol]['trades'])}, Win Rate: {win_rate_percentage:.2f}% "
                            f"({wins} wins, {losses} losses)")

            elif mode == "fronttest":
                logger.info(f"\n--- Fronttest Summary for Symbol: {symbol} ---")
                logger.info(f"Final Balance: {symbol_data_all[symbol]['balance']:.2f}")


    elif mode == "real":
        logger.info(f"Starting real-time strategy loop for symbols: {symbols}")
        symbol_data_realtime = {
            sym: {
                'higher_data': pd.DataFrame(),
                'lower_data' : pd.DataFrame(),
                'last_higher_update': None,
                'last_lower_update': None,
                'in_trade': False,
                'balance': initial_capital, # Initial capital for reference
                'trades': [],
                'trade_direction': None,
                'position_size_leveraged': 0,
                'last_higher_timestamp': None,
                'last_lower_timestamp': None
            } for sym in symbols
        }

        request_count = 0
        start_time = time.time()

        while True:
            for symbol in symbols:
                current_iteration_start = time.time()
                elapsed_time = time.time() - start_time
                if request_count >= 58 and elapsed_time < 60:
                    sleep_time = 60 - elapsed_time + 1
                    logger.warning(f"Approaching rate limit. Sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    start_time = time.time()
                    request_count = 0

                try:
                    # --- Fetch Higher Timeframe Data ---
                    if symbol_data_realtime[symbol]['higher_data'].empty:
                        df_higher_latest = fetch_ohlcv(
                            symbol, "higher", limit=historical_data_limit, mode=mode,
                            data_source=data_source,
                            csv_file_path_higher=csv_file_path_higher,
                            csv_file_path_lower=csv_file_path_lower,
                            higher_timeframe=higher_timeframe,
                            lower_timeframe=lower_timeframe,
                        )
                    else:
                        df_higher_latest = fetch_ohlcv(
                            symbol, "higher", limit=2, mode=mode,
                            data_source=data_source,
                            csv_file_path_higher=csv_file_path_higher,
                            csv_file_path_lower=csv_file_path_lower,
                            higher_timeframe=higher_timeframe,
                            lower_timeframe=lower_timeframe,
                        )
                    request_count += 1

                    # --- Fetch Lower Timeframe Data ---
                    if symbol_data_realtime[symbol]['lower_data'].empty:
                        df_lower_latest = fetch_ohlcv(
                            symbol, "lower", limit=100, mode=mode, # Adjusted limit for lower TF if needed
                            data_source=data_source,
                            csv_file_path_higher=csv_file_path_higher,
                            csv_file_path_lower=csv_file_path_lower,
                            higher_timeframe=higher_timeframe,
                            lower_timeframe=lower_timeframe,
                        )
                    else:
                        df_lower_latest = fetch_ohlcv(
                            symbol, "lower", limit=100, mode=mode, # Adjusted limit for lower TF if needed
                            data_source=data_source,
                            csv_file_path_higher=csv_file_path_higher,
                            csv_file_path_lower=csv_file_path_lower,
                            higher_timeframe=higher_timeframe,
                            lower_timeframe=lower_timeframe,
                        )
                    request_count += 1


                except Exception as e:
                    logger.error(f"Data fetch failed for {symbol}: {e}")
                    continue

                if not df_higher_latest.empty:
                    # --- Update Higher Timeframe Data ---
                    if symbol_data_realtime[symbol]['higher_data'].empty:
                        symbol_data_realtime[symbol]['higher_data'] = df_higher_latest
                    else:
                        if symbol_data_realtime[symbol]['last_higher_timestamp'] is not None:
                            new_candles_higher = df_higher_latest[df_higher_latest.index > symbol_data_realtime[symbol]['last_higher_timestamp']]
                        else:
                            new_candles_higher = df_higher_latest

                        if not new_candles_higher.empty:
                            old_higher_data = symbol_data_realtime[symbol]['higher_data']
                            symbol_data_realtime[symbol]['higher_data'] = pd.DataFrame()
                            df_higher_appended = pd.concat([old_higher_data, new_candles_higher])
                            del old_higher_data

                            df_higher_deduplicated = df_higher_appended[~df_higher_appended.index.duplicated(keep='last')]
                            symbol_data_realtime[symbol]['higher_data'] = df_higher_deduplicated.iloc[-historical_data_limit:]
                            symbol_data_realtime[symbol]['last_higher_timestamp'] = symbol_data_realtime[symbol]['higher_data'].index[-1]
                        else:
                            logger.debug(f"({symbol}) No new higher timeframe candles to add.")


                    latest_higher_candle = symbol_data_realtime[symbol]['higher_data'].iloc[-1]

                    # --- Update Lower Timeframe Data ---
                    if symbol_data_realtime[symbol]['lower_data'].empty:
                        symbol_data_realtime[symbol]['lower_data'] = df_lower_latest
                    else:
                        if symbol_data_realtime[symbol]['last_lower_timestamp'] is not None:
                            new_candles_lower = df_lower_latest[df_lower_latest.index > symbol_data_realtime[symbol]['last_lower_timestamp']]
                        else:
                            new_candles_lower = df_lower_latest

                        if not new_candles_lower.empty:
                            old_lower_data = symbol_data_realtime[symbol]['lower_data']
                            symbol_data_realtime[symbol]['lower_data'] = pd.DataFrame()
                            df_lower_appended = pd.concat([old_lower_data, new_candles_lower])
                            del old_lower_data

                            df_lower_deduplicated = df_lower_appended[~df_lower_appended.index.duplicated(keep='last')]
                            symbol_data_realtime[symbol]['lower_data'] = df_lower_deduplicated.iloc[-100:]
                            symbol_data_realtime[symbol]['last_lower_timestamp'] = symbol_data_realtime[symbol]['lower_data'].index[-1]
                        else:
                             logger.debug(f"({symbol}) No new lower timeframe candles to add.")


                    if symbol_data_realtime[symbol]['last_higher_update'] is None or latest_higher_candle.name > symbol_data_realtime[symbol]['last_higher_update']:
                        symbol_data_realtime[symbol]['last_higher_update'] = latest_higher_candle.name
                        process_strategy_logic_realtime(symbol, latest_higher_candle, symbol_data_realtime, strategy_mode, risk_reward_ratio, risk_percentage_per_trade, leverage_value, request_client)

                time_since_last = time.time() - current_iteration_start
                if time_since_last < request_delay:
                    time.sleep(request_delay - time_since_last)

                process = psutil.Process(os.getpid())
                memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                logger.debug(f"({symbol}) - Memory Usage: {memory_usage_mb:.2f} MB")


def process_strategy_logic(symbol, current_higher_candle, df_higher_data, df_lower_data, symbol_data_all, strategy_mode, risk_reward_ratio, risk_percentage_per_trade, leverage_value, mode):
    request_client = RequestsClientWithPrecision(access_id=COINEX_ACCESS_ID, secret_key=COINEX_SECRET_KEY)

    if strategy_mode in ["long", "both"]:
        resistance_level = df_higher_data['high'].iloc[max(0, df_higher_data.index.get_loc(current_higher_candle.name)-20):df_higher_data.index.get_loc(current_higher_candle.name)].max() if df_higher_data.index.get_loc(current_higher_candle.name) > 0 else df_higher_data['high'].iloc[max(0, df_higher_data.index.get_loc(current_higher_candle.name)-20):df_higher_data.index.get_loc(current_higher_candle.name)].max()
        is_higher_breakout = current_higher_candle['close'] > resistance_level

        if is_higher_breakout and not symbol_data_all[symbol]['in_trade']:
            logger.info(f"({symbol}) {current_higher_candle.name} - Higher TF Breakout at {current_higher_candle['close']}")
            start_time_lower = current_higher_candle.name
            df_lower_after_breakout = df_lower_data[df_lower_data.index > start_time_lower]

            if df_lower_after_breakout.empty:
                logger.info(f"({symbol}) No lower TF data after breakout candle.")
                return

            pullback_candles = []
            for j in range(len(df_lower_after_breakout)):
                current_lower_candle = df_lower_after_breakout.iloc[j]
                if j > 0 and current_lower_candle['close'] < df_lower_after_breakout.iloc[j-1]['close']:
                    pullback_candles.append(current_lower_candle)
                elif pullback_candles:
                    last_pullback_candle = pullback_candles[-1]
                    entry_signal_candle = current_lower_candle

                    if entry_signal_candle['close'] > last_pullback_candle['high']:
                        entry_price = entry_signal_candle['close']
                        stop_loss = last_pullback_candle['low']

                        risk_amount = symbol_data_all[symbol]['balance'] * risk_percentage_per_trade
                        stop_loss_distance = (entry_price - stop_loss)
                        position_size = risk_amount / stop_loss_distance if risk_amount > 0 else 0
                        position_size_leveraged = risk_amount / (stop_loss_distance * leverage_value)

                        if position_size_leveraged > 0:
                            take_profit = entry_price + (risk_reward_ratio * risk_amount)

                            adjusted_entry_price, adjusted_quantity, adjusted_sl_price, adjusted_tp_price, _ = request_client.adjust_values(
                                symbol, entry_price, position_size_leveraged, stop_loss, take_profit
                            )

                            if adjusted_entry_price is None:
                                logger.error(f"({symbol}) Error adjusting values. Skipping trade.")
                                return

                            place_market_order(symbol, 'buy', adjusted_quantity)
                            set_stop_loss(symbol, adjusted_sl_price)
                            set_take_profit(symbol, adjusted_tp_price)
                            order_id = 'simulated_order_id'

                            trade = {
                                'order_id': order_id,
                                'entry_time': entry_signal_candle.name,
                                'entry_price': adjusted_entry_price,
                                'stop_loss': adjusted_sl_price,
                                'take_profit': adjusted_tp_price,
                                'status': 'open',
                                'direction': 'long',
                                'position_size': adjusted_quantity,
                                'leverage': leverage_value
                            }
                            symbol_data_all[symbol]['trades'].append(trade)
                            symbol_data_all[symbol]['in_trade'] = True
                            symbol_data_all[symbol]['trade_direction'] = 'long'
                            symbol_data_all[symbol]['position_size_leveraged'] = adjusted_quantity
                            logger.info(f"  ({symbol}) {entry_signal_candle.name} - Entry LONG at {adjusted_entry_price}, "
                                         f"SL: {adjusted_sl_price}, TP: {adjusted_tp_price}, "
                                         f"Size: {adjusted_quantity:.4f}, Leverage: {leverage_value}x, Order ID: {order_id}")
                            return

    if strategy_mode in ["short", "both"]:
        support_level = df_higher_data['low'].iloc[max(0, df_higher_data.index.get_loc(current_higher_candle.name)-20):df_higher_data.index.get_loc(current_higher_candle.name)].min() if df_higher_data.index.get_loc(current_higher_candle.name) > 0 else df_higher_data['low'].iloc[max(0, df_higher_data.index.get_loc(current_higher_candle.name)-20):df_higher_data.index.get_loc(current_higher_candle.name)].min()
        is_higher_breakdown = current_higher_candle['close'] < support_level

        if is_higher_breakdown and not symbol_data_all[symbol]['in_trade']:
            logger.info(f"({symbol}) {current_higher_candle.name} - Higher TF Breakdown at {current_higher_candle['close']}")
            start_time_lower = current_higher_candle.name
            df_lower_after_breakdown = df_lower_data[df_lower_data.index > start_time_lower]

            if df_lower_after_breakdown.empty:
                logger.info(f"({symbol}) No lower TF data after breakdown candle.")
                return

            rally_candles = []
            for j in range(len(df_lower_after_breakdown)):
                current_lower_candle = df_lower_after_breakdown.iloc[j]
                if j > 0 and current_lower_candle['close'] > df_lower_after_breakdown.iloc[j-1]['close']:
                    rally_candles.append(current_lower_candle)
                elif rally_candles:
                    last_rally_candle = rally_candles[-1]
                    entry_signal_candle = current_lower_candle

                    if entry_signal_candle['close'] < last_rally_candle['low']:
                        entry_price = entry_signal_candle['close']
                        stop_loss = last_rally_candle['high']

                        risk_amount = symbol_data_all[symbol]['balance'] * risk_percentage_per_trade
                        stop_loss_distance = (stop_loss - entry_price)
                        position_size = risk_amount / stop_loss_distance if risk_amount > 0 else 0
                        position_size_leveraged = risk_amount / (stop_loss_distance * leverage_value)


                        if position_size_leveraged > 0:
                            take_profit = entry_price - (risk_reward_ratio * risk_amount)

                            adjusted_entry_price, adjusted_quantity, adjusted_sl_price, adjusted_tp_price, _ = request_client.adjust_values(
                                symbol, entry_price, position_size_leveraged, stop_loss, take_profit
                            )
                            if adjusted_entry_price is None:
                                logger.error(f"({symbol}) Error adjusting values. Skipping trade.")
                                return

                            place_market_order(symbol, 'sell', adjusted_quantity)
                            set_stop_loss(symbol, adjusted_sl_price)
                            set_take_profit(symbol, adjusted_tp_price)
                            order_id = 'simulated_order_id'

                            trade = {
                                'order_id': order_id,
                                'entry_time': entry_signal_candle.name,
                                'entry_price': adjusted_entry_price,
                                'stop_loss': adjusted_sl_price,
                                'take_profit': adjusted_tp_price,
                                'status': 'open',
                                'direction': 'short',
                                'position_size': adjusted_quantity,
                                'leverage': leverage_value
                            }
                            symbol_data_all[symbol]['trades'].append(trade)
                            symbol_data_all[symbol]['in_trade'] = True
                            symbol_data_all[symbol]['trade_direction'] = 'short'
                            symbol_data_all[symbol]['position_size_leveraged'] = adjusted_quantity
                            logger.info(f"  ({symbol}) {entry_signal_candle.name} - Entry SHORT at {adjusted_entry_price}, "
                                         f"SL: {adjusted_sl_price}, TP: {adjusted_tp_price}, "
                                         f"Size: {adjusted_quantity:.4f}, Leverage: {leverage_value}x, Order ID: {order_id}")
                            return

    if symbol_data_all[symbol]['in_trade']:
        idx = df_lower_data.index.get_indexer([current_higher_candle.name], method='nearest')[0]
        last_lower_data_point = df_lower_data.iloc[idx]
        current_price = last_lower_data_point['close']
        trade = symbol_data_all[symbol]['trades'][-1]

        if symbol_data_all[symbol]['trade_direction'] == 'long':
            if current_price <= symbol_data_all[symbol]['stop_loss']:
                profit = (symbol_data_all[symbol]['stop_loss'] - symbol_data_all[symbol]['entry_price']) * symbol_data_all[symbol]['position_size_leveraged']
                symbol_data_all[symbol]['balance'] += profit
                trade.update({
                    'status': 'stopped_out',
                    'exit_time': last_lower_data_point.name,
                    'exit_price': symbol_data_all[symbol]['stop_loss'],
                    'profit': profit
                })
                symbol_data_all[symbol]['in_trade'] = False
                symbol_data_all[symbol]['trade_direction'] = None
                logger.info(f"  ({symbol}) {last_lower_data_point.name} - LONG Stop-Loss Hit at {symbol_data_all[symbol]['stop_loss']}, "
                             f"Profit: {profit:.2f}, Balance: {symbol_data_all[symbol]['balance']:.2f}")

            elif current_price >= symbol_data_all[symbol]['take_profit']:
                profit = (symbol_data_all[symbol]['take_profit'] - symbol_data_all[symbol]['entry_price']) * symbol_data_all[symbol]['position_size_leveraged']
                symbol_data_all[symbol]['balance'] += profit
                trade.update({
                    'status': 'take_profit',
                    'exit_time': last_lower_data_point.name,
                    'exit_price': symbol_data_all[symbol]['take_profit'],
                    'profit': profit
                })
                symbol_data_all[symbol]['in_trade'] = False
                symbol_data_all[symbol]['trade_direction'] = None
                logger.info(f"  ({symbol}) {last_lower_data_point.name} - LONG Take-Profit Hit at {symbol_data_all[symbol]['take_profit']}, "
                             f"Profit: {profit:.2f}, Balance: {symbol_data_all[symbol]['balance']:.2f}")

        elif symbol_data_all[symbol]['trade_direction'] == 'short':
            if current_price >= symbol_data_all[symbol]['stop_loss']:
                profit = (symbol_data_all[symbol]['entry_price'] - symbol_data_all[symbol]['stop_loss']) * symbol_data_all[symbol]['position_size_leveraged']
                symbol_data_all[symbol]['balance'] += profit
                trade.update({
                    'status': 'stopped_out',
                    'exit_time': last_lower_data_point.name,
                    'exit_price': symbol_data_all[symbol]['stop_loss'],
                    'profit': profit
                })
                symbol_data_all[symbol]['in_trade'] = False
                symbol_data_all[symbol]['trade_direction'] = None
                logger.info(f"  ({symbol}) {last_lower_data_point.name} - SHORT Stop-Loss Hit at {symbol_data_all[symbol]['stop_loss']}, "
                             f"Profit: {profit:.2f}, Balance: {symbol_data_all[symbol]['balance']:.2f}")

            elif current_price <= symbol_data_all[symbol]['take_profit']:
                profit = (symbol_data_all[symbol]['entry_price'] - symbol_data_all[symbol]['take_profit']) * symbol_data_all[symbol]['position_size_leveraged']
                symbol_data_all[symbol]['balance'] += profit
                trade.update({
                    'status': 'take_profit',
                    'exit_time': last_lower_data_point.name,
                    'exit_price': symbol_data_all[symbol]['take_profit'],
                    'profit': profit
                })
                symbol_data_all[symbol]['in_trade'] = False
                symbol_data_all[symbol]['trade_direction'] = None
                logger.info(f"  ({symbol}) {last_lower_data_point.name} - SHORT Take-Profit Hit at {symbol_data_all[symbol]['take_profit']}, "
                             f"Profit: {profit:.2f}, Balance: {symbol_data_all[symbol]['balance']:.2f}")


def process_strategy_logic_realtime(symbol, latest_higher_candle, symbol_data_realtime, strategy_mode, risk_reward_ratio, risk_percentage_per_trade, leverage_value, request_client):

    df_higher_data = symbol_data_realtime[symbol]['higher_data']
    df_lower_data = symbol_data_realtime[symbol]['lower_data']

    if strategy_mode in ["long", "both"]:
        resistance_level = df_higher_data['high'].iloc[:-1].max() if len(df_higher_data) > 1 else latest_higher_candle['high']
        is_higher_breakout = latest_higher_candle['close'] > resistance_level

        if is_higher_breakout and not symbol_data_realtime[symbol]['in_trade']:
            logger.info(f"({symbol}) {latest_higher_candle.name} - Higher TF Breakout at {latest_higher_candle['close']}")
            start_time_lower = latest_higher_candle.name
            df_lower_after_breakout = df_lower_data[df_lower_data.index > start_time_lower]

            if df_lower_after_breakout.empty:
                logger.info(f"({symbol}) No lower TF data after breakout candle.")
                return
            else:
                pullback_candles = []
                for j in range(len(df_lower_after_breakout)):
                    current_lower_candle = df_lower_after_breakout.iloc[j]
                    if j > 0 and current_lower_candle['close'] < df_lower_after_breakout.iloc[j-1]['close']:
                        pullback_candles.append(current_lower_candle)
                    elif pullback_candles:
                        last_pullback_candle = pullback_candles[-1]
                        entry_signal_candle = current_lower_candle

                        if entry_signal_candle['close'] > last_pullback_candle['high']:
                            entry_price = entry_signal_candle['close']
                            stop_loss = last_pullback_candle['low']

                            real_balance = get_futures_balance_real_mode()
                            if real_balance is None:
                                logger.error(f"({symbol}) Failed to fetch real balance. Skipping trade.")
                                return
                            risk_amount = real_balance * risk_percentage_per_trade

                            stop_loss_distance = (entry_price - stop_loss)
                            position_size = risk_amount / stop_loss_distance if risk_amount > 0 else 0
                            position_size_leveraged = risk_amount / (stop_loss_distance * leverage_value)

                            if position_size_leveraged > 0:
                                take_profit = entry_price + (risk_reward_ratio * risk_amount)

                                adjusted_entry_price, adjusted_quantity, adjusted_sl_price, adjusted_tp_price, _ = request_client.adjust_values(
                                    symbol, entry_price, position_size_leveraged, stop_loss, take_profit
                                )

                                if adjusted_entry_price is None:
                                    logger.error(f"({symbol}) Error adjusting values. Skipping trade.")
                                    return

                                logger.debug(f"({symbol}) Adjusted Values - Entry: {adjusted_entry_price}, Quantity: {adjusted_quantity}, SL: {adjusted_sl_price}, TP: {adjusted_tp_price}") # Log adjusted values

                                if MODE == "real":
                                    if set_leverage(symbol, leverage_value):
                                        order_result = place_market_order(symbol, 'buy', adjusted_quantity)
                                        if order_result and order_result.get('order_id'):
                                            set_stop_loss(symbol, adjusted_sl_price)
                                            set_take_profit(symbol, adjusted_tp_price)
                                            order_id = order_result['order_id']
                                        else:
                                            logger.error(f"({symbol}) Real LONG order placement failed. Skipping trade.")
                                            return
                                    else:
                                        logger.error(f"({symbol}) Failed to set leverage in real mode. Skipping trade.")
                                        return
                                else:
                                    place_market_order(symbol, 'buy', adjusted_quantity)
                                    set_stop_loss(symbol, adjusted_sl_price)
                                    set_take_profit(symbol, adjusted_tp_price)
                                    order_id = 'simulated_order_id'

                                trade = {
                                    'order_id': order_id,
                                    'entry_time': entry_signal_candle.name,
                                    'entry_price': adjusted_entry_price,
                                    'stop_loss': adjusted_sl_price,
                                    'take_profit': adjusted_tp_price,
                                    'status': 'open',
                                    'direction': 'long',
                                    'position_size': adjusted_quantity,
                                    'leverage': leverage_value
                                }
                                symbol_data_realtime[symbol]['trades'].append(trade)
                                symbol_data_realtime[symbol]['in_trade'] = True
                                symbol_data_realtime[symbol]['trade_direction'] = 'long'
                                symbol_data_realtime[symbol]['position_size_leveraged'] = adjusted_quantity
                                logger.info(f"  ({symbol}) {entry_signal_candle.name} - Entry LONG at {adjusted_entry_price}, "
                                             f"SL: {adjusted_sl_price}, TP: {adjusted_tp_price}, "
                                             f"Size: {adjusted_quantity:.4f}, Leverage: {leverage_value}x, Order ID: {order_id}")
                                return
        else:
            return

    if strategy_mode in ["short", "both"]:
        support_level = df_higher_data['low'].iloc[:-1].min() if len(df_higher_data) > 1 else latest_higher_candle['low']
        is_higher_breakdown = latest_higher_candle['close'] < support_level

        if is_higher_breakdown and not symbol_data_realtime[symbol]['in_trade']:
            logger.info(f"({symbol}) {latest_higher_candle.name} - Higher TF Breakdown at {latest_higher_candle['close']}")
            start_time_lower = latest_higher_candle.name
            df_lower_after_breakdown = df_lower_data[df_lower_data.index > start_time_lower]

            if df_lower_after_breakdown.empty:
                logger.info(f"({symbol}) No lower TF data after breakdown candle.")
                return
            else:
                rally_candles = []
                for j in range(len(df_lower_after_breakdown)):
                    current_lower_candle = df_lower_after_breakout.iloc[j] # Corrected variable name
                    if j > 0 and current_lower_candle['close'] > df_lower_after_breakdown.iloc[j-1]['close']:
                        rally_candles.append(current_lower_candle)
                    elif rally_candles:
                        last_rally_candle = rally_candles[-1]
                        entry_signal_candle = current_lower_candle

                        if entry_signal_candle['close'] < last_rally_candle['low']:
                            entry_price = entry_signal_candle['close']
                            stop_loss = last_rally_candle['high']

                            real_balance = get_futures_balance_real_mode()
                            if real_balance is None:
                                logger.error(f"({symbol}) Failed to fetch real balance. Skipping trade.")
                                return
                            risk_amount = real_balance * risk_percentage_per_trade

                            stop_loss_distance = (stop_loss - entry_price)
                            position_size = risk_amount / stop_loss_distance if risk_amount > 0 else 0
                            position_size_leveraged = risk_amount / (stop_loss_distance * leverage_value)


                            if position_size_leveraged > 0:
                                take_profit = entry_price - (risk_reward_ratio * risk_amount)

                                adjusted_entry_price, adjusted_quantity, adjusted_sl_price, adjusted_tp_price, _ = request_client.adjust_values(
                                    symbol, entry_price, position_size_leveraged, stop_loss, take_profit
                                )
                                if adjusted_entry_price is None:
                                    logger.error(f"({symbol}) Error adjusting values. Skipping trade.")
                                    return

                                logger.debug(f"({symbol}) Adjusted Values - Entry: {adjusted_entry_price}, Quantity: {adjusted_quantity}, SL: {adjusted_sl_price}, TP: {adjusted_tp_price}") # Log adjusted values


                                if MODE == "real":
                                    if set_leverage(symbol, leverage_value):
                                        order_result = place_market_order(symbol, 'sell', adjusted_quantity)
                                        if order_result and order_result.get('order_id'):
                                            set_stop_loss(symbol, adjusted_sl_price)
                                            set_take_profit(symbol, adjusted_tp_price)
                                            order_id = order_result['order_id']
                                        else:
                                            logger.error(f"({symbol}) Real SHORT order placement failed. Skipping trade.")
                                            return
                                    else:
                                        logger.error(f"({symbol}) Failed to set leverage in real mode. Skipping trade.")
                                        return
                                elif MODE in ["backtest", "fronttest"]:
                                    place_market_order(symbol, 'sell', adjusted_quantity)
                                    set_stop_loss(symbol, adjusted_sl_price)
                                    set_take_profit(symbol, adjusted_tp_price)
                                    order_id = 'simulated_order_id'

                                trade = {
                                    'order_id': order_id,
                                    'entry_time': entry_signal_candle.name,
                                    'entry_price': adjusted_entry_price,
                                    'stop_loss': adjusted_sl_price,
                                    'take_profit': adjusted_tp_price,
                                    'status': 'open',
                                    'direction': 'short',
                                    'position_size': adjusted_quantity,
                                    'leverage': leverage_value
                                }
                                symbol_data_realtime[symbol]['trades'].append(trade)
                                symbol_data_realtime[symbol]['in_trade'] = True
                                symbol_data_realtime[symbol]['trade_direction'] = 'short'
                                symbol_data_realtime[symbol]['position_size_leveraged'] = adjusted_quantity
                                logger.info(f"  ({symbol}) {entry_signal_candle.name} - Entry SHORT at {adjusted_entry_price}, "
                                             f"SL: {adjusted_sl_price}, TP: {adjusted_tp_price}, "
                                             f"Size: {adjusted_quantity:.4f}, Leverage: {leverage_value}x, Order ID: {order_id}")
                                return
        else:
            return


    if symbol_data_realtime[symbol]['in_trade']:
        current_price = latest_higher_candle['close']
        trade = symbol_data_realtime[symbol]['trades'][-1]

        if symbol_data_realtime[symbol]['trade_direction'] == 'long':
            if current_price <= symbol_data_realtime[symbol]['stop_loss']:
                profit = (symbol_data_realtime[symbol]['stop_loss'] - symbol_data_realtime[symbol]['entry_price']) * symbol_data_realtime[symbol]['position_size_leveraged']
                symbol_data_realtime['balance'] += profit
                if MODE == 'real':
                    place_market_order(symbol, 'sell', symbol_data_realtime[symbol]['position_size_leveraged'])
                trade.update({
                    'status': 'stopped_out',
                    'exit_time': latest_higher_candle.name,
                    'exit_price': symbol_data_realtime[symbol]['stop_loss'],
                    'profit': profit
                })
                symbol_data_realtime[symbol]['in_trade'] = False
                symbol_data_realtime['trade_direction'] = None
                logger.info(f"  ({symbol}) {latest_higher_candle.name} - LONG Stop-Loss Hit at {symbol_data_realtime[symbol]['stop_loss']}, "
                             f"Profit: {profit:.2f}")

            elif current_price >= symbol_data_realtime[symbol]['take_profit']:
                profit = (symbol_data_realtime[symbol]['take_profit'] - symbol_data_realtime[symbol]['entry_price']) * symbol_data_realtime[symbol]['position_size_leveraged']
                symbol_data_realtime['balance'] += profit
                if MODE == 'real':
                    place_market_order(symbol, 'sell', symbol_data_realtime[symbol]['position_size_leveraged'])
                trade.update({
                    'status': 'take_profit',
                    'exit_time': latest_higher_candle.name,
                    'exit_price': symbol_data_realtime[symbol]['take_profit'],
                    'profit': profit
                })
                symbol_data_realtime[symbol]['in_trade'] = False
                symbol_data_realtime['trade_direction'] = None
                logger.info(f"  ({symbol}) {latest_higher_candle.name} - LONG Take-Profit Hit at {symbol_data_realtime[symbol]['take_profit']}, "
                             f"Profit: {profit:.2f}")

        elif symbol_data_realtime[symbol]['trade_direction'] == 'short':
            if current_price >= symbol_data_realtime[symbol]['stop_loss']:
                profit = (symbol_data_realtime[symbol]['entry_price'] - symbol_data_realtime[symbol]['stop_loss']) * symbol_data_realtime[symbol]['position_size_leveraged']
                symbol_data_realtime['balance'] += profit
                if MODE == 'real':
                    place_market_order(symbol, 'buy', symbol_data_realtime[symbol]['position_size_leveraged'])
                trade.update({
                    'status': 'stopped_out',
                    'exit_time': latest_higher_candle.name,
                    'exit_price': symbol_data_realtime[symbol]['stop_loss'],
                    'profit': profit
                })
                symbol_data_realtime[symbol]['in_trade'] = False
                symbol_data_realtime['trade_direction'] = None
                logger.info(f"  ({symbol}) {latest_higher_candle.name} - SHORT Stop-Loss Hit at {symbol_data_realtime[symbol]['stop_loss']}, "
                             f"Profit: {profit:.2f}")

            elif current_price <= symbol_data_realtime[symbol]['take_profit']:
                profit = (symbol_data_realtime[symbol]['entry_price'] - symbol_data_realtime[symbol]['take_profit']) * symbol_data_realtime[symbol]['position_size_leveraged']
                symbol_data_realtime['balance'] += profit
                if MODE == 'real':
                    place_market_order(symbol, 'buy', symbol_data_realtime[symbol]['position_size_leveraged'])
                trade.update({
                    'status': 'take_profit',
                    'exit_time': latest_higher_candle.name,
                    'exit_price': symbol_data_realtime[symbol]['take_profit'],
                    'profit': profit
                })
                symbol_data_realtime[symbol]['in_trade'] = False
                symbol_data_realtime['trade_direction'] = None
                logger.info(f"  ({symbol}) {latest_higher_candle.name} - SHORT Take-Profit Hit at {symbol_data_realtime[symbol]['take_profit']}, "
                             f"Profit: {profit:.2f}")


if __name__ == "__main__":
    symbols = [
                "BTCUSDT", "ETHUSDT", "XRPUSDT", "BCHUSDT", "LTCUSDT",
                "EOSUSDT", "ADAUSDT", "TRXUSDT", "XLMUSDT", "DOGEUSDT",
                "LINKUSDT", "DOTUSDT", "BNBUSDT", "XMRUSDT", "DASHUSDT",
                "ETCUSDT", "ZECUSDT", "VETUSDT", "SOLUSDT", "USTCUSDT",
                ]
    config_strategy = {
        "symbols": symbols,
        "higher_timeframe": '30min',
        "lower_timeframe": '1min',
        "initial_capital": 40,
        "risk_reward_ratio": 4,
        "mode": 'real',
        "strategy_mode": 'both',
        "risk_percentage_per_trade": 0.01,
        "leverage_value": 10,
        "data_source": 'coinex',
        "historical_data_limit": 500,
        "request_delay": 2.2,
        'csv_file_path_higher':"BTCUSDT_30m_data.csv",
        'csv_file_path_lower':"BTCUSDT_1m_data.csv"
    }

    run_strategy(**config_strategy)