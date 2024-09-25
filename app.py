import streamlit as st
import sqlite3
import yfinance as yf
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import random
import json
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import pytz
import math
import pywt
import numpy as np
from scipy.fft import fft, ifft
import threading
import functools

# ---------------------------
# 1. Configure Logging
# ---------------------------

logging.basicConfig(
    filename='trading_bot.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO  # Adjust as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)

# ---------------------------
# 2. Global Lock and Retry Decorator
# ---------------------------

# Create a global lock for database operations
db_lock = threading.Lock()

def retry_on_lock(max_retries=5, initial_delay=0.5, backoff_factor=2):
    """
    Decorator to retry a function if a sqlite3.OperationalError occurs due to database lock.
    Implements exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if 'database is locked' in str(e):
                        logging.warning(f"Database is locked. Retrying in {delay} seconds... (Attempt {retries + 1}/{max_retries})")
                        time.sleep(delay)
                        retries += 1
                        delay *= backoff_factor
                    else:
                        logging.error(f"OperationalError in {func.__name__}: {e}")
                        raise
            logging.error(f"Max retries exceeded for {func.__name__}.")
        return wrapper
    return decorator

# ---------------------------
# 3. Database Functions
# ---------------------------

@retry_on_lock(max_retries=5, initial_delay=0.5, backoff_factor=2)
def init_db():
    """Initializes the SQLite database with necessary tables."""
    with db_lock:
        try:
            with sqlite3.connect(r'G:\trade bots\trading_bot.db', timeout=30) as conn:
                c = conn.cursor()
                # Create trades table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        bot_name TEXT,
                        action TEXT,
                        ticker TEXT,
                        quantity INTEGER,
                        price REAL,
                        total_value REAL,
                        gain_loss REAL
                    )
                ''')
                # Create bots table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS bots (
                        name TEXT PRIMARY KEY,
                        balance REAL,
                        initial_balance REAL,
                        portfolio TEXT,
                        strategy TEXT,
                        trade_frequency_minutes INTEGER,
                        last_trade_time TEXT
                    )
                ''')
                conn.commit()
            logging.info("Database initialized.")
        except sqlite3.OperationalError as e:
            logging.error(f"OperationalError in init_db: {e}")
            raise

@retry_on_lock(max_retries=5, initial_delay=0.5, backoff_factor=2)
def save_trade_to_db(trade: dict):
    """Saves a trade record to the database."""
    with db_lock:
        try:
            with sqlite3.connect(r'G:\trade bots\trading_bot.db', timeout=30) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO trades (timestamp, bot_name, action, ticker, quantity, price, total_value, gain_loss)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade['timestamp'],
                    trade['bot_name'],
                    trade['action'],
                    trade['ticker'],
                    trade['quantity'],
                    trade['price'],
                    trade['total_value'],
                    trade['gain_loss']
                ))
                conn.commit()
            logging.info(f"Trade saved: {trade}")
        except sqlite3.OperationalError as e:
            logging.error(f"OperationalError in save_trade_to_db: {e}")
            raise

@retry_on_lock(max_retries=5, initial_delay=0.5, backoff_factor=2)
def save_bot_to_db(bot):
    """Saves or updates a TradingBot instance in the database."""
    with db_lock:
        try:
            with sqlite3.connect(r'G:\trade bots\trading_bot.db', timeout=30) as conn:
                c = conn.cursor()
                portfolio_json = json.dumps(bot.portfolio)
                # Convert last_trade_time to string without timezone for storage
                last_trade_time_str = bot.last_trade_time.strftime("%Y-%m-%d %H:%M:%S") if bot.last_trade_time else ''
                c.execute('''
                    INSERT OR REPLACE INTO bots (name, balance, initial_balance, portfolio, strategy, trade_frequency_minutes, last_trade_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    bot.name,
                    bot.balance,
                    bot.initial_balance,
                    portfolio_json,
                    bot.strategy_name,
                    bot.trade_frequency_minutes,
                    last_trade_time_str
                ))
                conn.commit()
            logging.info(f"Bot '{bot.name}' saved to database.")
        except sqlite3.OperationalError as e:
            logging.error(f"OperationalError in save_bot_to_db: {e}")
            raise

@retry_on_lock(max_retries=5, initial_delay=0.5, backoff_factor=2)
def delete_bot_from_db(bot_name: str):
    """Deletes a TradingBot from the database by name."""
    with db_lock:
        try:
            with sqlite3.connect(r'G:\trade bots\trading_bot.db', timeout=30) as conn:
                c = conn.cursor()
                c.execute('DELETE FROM bots WHERE name=?', (bot_name,))
                conn.commit()
            logging.info(f"Bot '{bot_name}' deleted from database.")
        except sqlite3.OperationalError as e:
            logging.error(f"OperationalError in delete_bot_from_db: {e}")
            raise

def load_bots_from_db() -> List['TradingBot']:
    """Loads all TradingBot instances from the database."""
    with db_lock:
        try:
            with sqlite3.connect(r'G:\trade bots\trading_bot.db', timeout=30) as conn:
                c = conn.cursor()
                c.execute('SELECT * FROM bots')
                bots_data = c.fetchall()
        except sqlite3.OperationalError as e:
            logging.error(f"OperationalError in load_bots_from_db: {e}")
            bots_data = []
    
    bots = []
    eastern = pytz.timezone('US/Eastern')  # Define timezone once
    for bot_data in bots_data:
        name, balance, initial_balance, portfolio_json, strategy_name, trade_freq_minutes, last_trade_time_str = bot_data
        portfolio = json.loads(portfolio_json)
        if last_trade_time_str:
            naive_datetime = datetime.strptime(last_trade_time_str, "%Y-%m-%d %H:%M:%S")
            last_trade_time = eastern.localize(naive_datetime)
        else:
            last_trade_time = eastern.localize(datetime(1970, 1, 1))  # Or another suitable past date
        bot = TradingBot(
            name=name,
            initial_capital=initial_balance,
            strategy_name=strategy_name,
            trade_frequency_minutes=trade_freq_minutes
        )
        bot.balance = balance
        bot.portfolio = portfolio
        bot.last_trade_time = last_trade_time
        bots.append(bot)
    logging.info(f"Loaded {len(bots)} bots from database.")
    return bots

# ---------------------------
# 4. Data Fetching Functions
# ---------------------------

@st.cache_data(ttl=300)
def get_stock_universe() -> List[str]:
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]  # For compatibility with yfinance
        logging.info("Fetched S&P 500 tickers.")
        return tickers
    except Exception as e:
        logging.error(f"Error fetching stock universe: {e}")
        return []

@st.cache_data(ttl=300)
def get_current_prices(tickers: List[str]) -> Dict[str, Optional[float]]:
    """Fetches current prices for a list of tickers using yfinance's batch functionality."""
    try:
        # yfinance can handle multiple tickers separated by space or comma
        data = yf.download(' '.join(tickers), period="1d", group_by='ticker', threads=True, progress=False)
        current_prices = {}
        for ticker in tickers:
            if ticker in data and 'Close' in data[ticker]:
                # Handle possible FutureWarning by using iloc
                current_prices[ticker] = data[ticker]['Close'].iloc[-1]
            else:
                current_prices[ticker] = None
        logging.info("Fetched current prices for all tickers.")
        return current_prices
    except Exception as e:
        logging.error(f"Error fetching current prices: {e}")
        return {ticker: None for ticker in tickers}

@st.cache_data(ttl=300)
def get_daily_performance(tickers: List[str]) -> Dict[str, Optional[float]]:
    """Fetches the daily performance percentage of a list of tickers."""
    try:
        data = yf.download(' '.join(tickers), period="2d", group_by='ticker', threads=True, progress=False)
        performance = {}
        for ticker in tickers:
            if ticker in data and 'Close' in data[ticker]:
                hist = data[ticker]['Close']
                if len(hist) >= 2:
                    yesterday_close = hist.iloc[-2]
                    today_close = hist.iloc[-1]
                    performance[ticker] = (today_close - yesterday_close) / yesterday_close
                else:
                    performance[ticker] = 0.0
            else:
                performance[ticker] = 0.0
        logging.info("Fetched daily performance for all tickers.")
        return performance
    except Exception as e:
        logging.error(f"Error fetching daily performance: {e}")
        return {ticker: 0.0 for ticker in tickers}

def validate_tickers(tickers: List[str]) -> List[str]:
    """Validates tickers by checking if current price data is available."""
    current_prices = get_current_prices(tickers)
    valid_tickers = [ticker for ticker, price in current_prices.items() if price is not None]
    invalid_tickers = [ticker for ticker, price in current_prices.items() if price is None]
    for ticker in invalid_tickers:
        logging.warning(f"Ticker '{ticker}' is invalid or data is unavailable.")
    return valid_tickers

@st.cache_data(ttl=300)
def get_historical_prices(ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Retrieves historical stock prices for a given ticker over a specified number of days."""
    try:
        stock = yf.Ticker(ticker)
        eastern = pytz.timezone('US/Eastern')
        end_date = datetime.now(eastern)
        start_date = end_date - timedelta(days=days)
        data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if not data.empty:
            logging.info(f"Fetched historical data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return data
        else:
            logging.warning(f"No historical data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return None
    except Exception as e:
        logging.error(f"Error fetching historical prices for {ticker}: {e}")
        return None

# ---------------------------
# 5. Trading Strategies
# ---------------------------

class BaseStrategy:
    """Base class for trading strategies."""
    def execute(self, bot: 'TradingBot', ticker: str, current_prices: Dict[str, Optional[float]]):
        pass

class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy: Buys when price is below mean, sells when above."""
    def execute(self, bot: 'TradingBot', ticker: str, current_prices: Dict[str, Optional[float]]):
        prices = get_historical_prices(ticker, days=10)
        if prices is not None and not prices.empty:
            mean_price = prices['Close'].mean()
            current_price = current_prices.get(ticker)
            logging.info(f"{bot.name} - {ticker}: Current Price = {current_price}, Mean Price = {mean_price}")
            if current_price and current_price < mean_price * 0.98:
                quantity = random.randint(5, 15)
                success = bot.buy_stock(ticker, quantity, current_prices)
                if success:
                    logging.info(f"{bot.name} - {ticker}: Bought {quantity} shares at ${current_price:.2f}")
            elif current_price and current_price > mean_price * 1.02:
                quantity = bot.portfolio.get(ticker, 0)
                if quantity > 0:
                    success = bot.sell_stock(ticker, quantity, current_prices)
                    if success:
                        logging.info(f"{bot.name} - {ticker}: Sold {quantity} shares at ${current_price:.2f}")
        else:
            logging.warning(f"{bot.name} - {ticker}: No historical data available.")

class MomentumStrategy(BaseStrategy):
    """Momentum Strategy: Buys on upward trends, sells on downward trends."""
    def execute(self, bot: 'TradingBot', ticker: str, current_prices: Dict[str, Optional[float]]):
        prices = get_historical_prices(ticker, days=7)
        if prices is not None and not prices.empty:
            prices['Returns'] = prices['Close'].pct_change()
            momentum = prices['Returns'].rolling(window=3).mean().iloc[-1]
            logging.info(f"{bot.name} - {ticker}: Momentum = {momentum}")
            if momentum > 0.01:
                success = bot.buy_stock(ticker, 5, current_prices)
                if success:
                    logging.info(f"{bot.name} - {ticker}: Bought 5 shares at ${current_prices.get(ticker):.2f}")
            elif momentum < -0.01:
                quantity = bot.portfolio.get(ticker, 0)
                if quantity > 0:
                    success = bot.sell_stock(ticker, 5, current_prices)
                    if success:
                        logging.info(f"{bot.name} - {ticker}: Sold 5 shares at ${current_prices.get(ticker):.2f}")
        else:
            logging.warning(f"{bot.name} - {ticker}: No historical data available.")

class MovingAverageCrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy: Buys on MA5 crossing above MA15, sells on opposite."""
    def execute(self, bot: 'TradingBot', ticker: str, current_prices: Dict[str, Optional[float]]):
        prices = get_historical_prices(ticker, days=15)
        if prices is not None and not prices.empty:
            prices['MA5'] = prices['Close'].rolling(window=5).mean()
            prices['MA15'] = prices['Close'].rolling(window=15).mean()
            if len(prices) >= 15:
                if prices['MA5'].iloc[-1] > prices['MA15'].iloc[-1] and prices['MA5'].iloc[-2] <= prices['MA15'].iloc[-2]:
                    success = bot.buy_stock(ticker, 10, current_prices)
                    if success:
                        logging.info(f"{bot.name} - {ticker}: MA5 crossed above MA15. Bought 10 shares.")
                elif prices['MA5'].iloc[-1] < prices['MA15'].iloc[-1] and prices['MA5'].iloc[-2] >= prices['MA15'].iloc[-2]:
                    quantity = bot.portfolio.get(ticker, 0)
                    if quantity >= 10:
                        success = bot.sell_stock(ticker, 10, current_prices)
                        if success:
                            logging.info(f"{bot.name} - {ticker}: MA5 crossed below MA15. Sold 10 shares.")
        else:
            logging.warning(f"{bot.name} - {ticker}: No historical data available.")

class WaveletTrendStrategy(BaseStrategy):
    """Wavelet Trend Strategy: Uses wavelet decomposition to identify trends and execute trades based on trend direction."""
    def execute(self, bot: 'TradingBot', ticker: str, current_prices: Dict[str, Optional[float]]):
        prices = get_historical_prices(ticker, days=30)
        if prices is not None and not prices.empty:
            close_prices = prices['Close'].tolist()
            bulk_trend, surface_movements = wavelet_decomposition(close_prices, wavelet='db4', max_requested_level=4)
            current_trend = bulk_trend[-1]
            logging.info(f"{bot.name} - {ticker}: Current Bulk Trend = {current_trend}")
            if current_trend > 0:
                success = bot.buy_stock(ticker, 10, current_prices)
                if success:
                    logging.info(f"{bot.name} - {ticker}: Positive Trend detected. Bought 10 shares.")
            elif current_trend < 0:
                quantity = bot.portfolio.get(ticker, 0)
                if quantity >= 10:
                    success = bot.sell_stock(ticker, 10, current_prices)
                    if success:
                        logging.info(f"{bot.name} - {ticker}: Negative Trend detected. Sold 10 shares.")
        else:
            logging.warning(f"{bot.name} - {ticker}: No historical data available.")

class FourierCycleStrategy(BaseStrategy):
    """Fourier Cycle Strategy: Uses Fourier analysis to identify dominant cycles and trade based on cycle predictions."""
    def execute(self, bot: 'TradingBot', ticker: str, current_prices: Dict[str, Optional[float]]):
        prices = get_historical_prices(ticker, days=60)
        if prices is not None and not prices.empty:
            signal = prices['Close'].tolist()
            n = len(signal)
            yf_vals = fft(signal)
            xf = np.linspace(0.0, 1.0/(2.0*(prices.index[1] - prices.index[0]).total_seconds()), n//2)
            amplitudes = 2.0/n * np.abs(yf_vals[:n//2])
            dominant_freq = xf[np.argmax(amplitudes[1:]) + 1]  # Exclude the zero frequency
            logging.info(f"{bot.name} - {ticker}: Dominant Frequency = {dominant_freq}")
            # Simple trading rule: If current price is above the reconstructed signal, sell; else buy
            reconstructed = ifft(yf_vals).real
            current_price = signal[-1]
            reconstructed_price = reconstructed[-1]
            if current_price > reconstructed_price * 1.01:
                quantity = bot.portfolio.get(ticker, 0)
                if quantity >= 10:
                    success = bot.sell_stock(ticker, 10, current_prices)
                    if success:
                        logging.info(f"{bot.name} - {ticker}: Price above cycle. Sold 10 shares.")
            elif current_price < reconstructed_price * 0.99:
                success = bot.buy_stock(ticker, 10, current_prices)
                if success:
                    logging.info(f"{bot.name} - {ticker}: Price below cycle. Bought 10 shares.")
        else:
            logging.warning(f"{bot.name} - {ticker}: No historical data available.")

class VolatilityAdjustedStrategy(BaseStrategy):
    """Volatility Adjusted Strategy: Adjusts trading behavior based on market volatility."""
    def execute(self, bot: 'TradingBot', ticker: str, current_prices: Dict[str, Optional[float]]):
        prices = get_historical_prices(ticker, days=20)
        if prices is not None and not prices.empty:
            volatility = prices['Close'].rolling(window=10).std().iloc[-1]
            current_price = current_prices.get(ticker)
            logging.info(f"{bot.name} - {ticker}: Current Volatility = {volatility}")
            # Define volatility thresholds
            historical_volatility = prices['Close'].rolling(window=10).std().mean()
            high_volatility = historical_volatility * 1.5
            low_volatility = historical_volatility * 0.5
            if volatility > high_volatility:
                # High volatility: Smaller trades to minimize risk
                success = bot.buy_stock(ticker, 5, current_prices)
                if success:
                    logging.info(f"{bot.name} - {ticker}: High volatility detected. Bought 5 shares.")
            elif volatility < low_volatility:
                # Low volatility: Larger trades to capitalize on stability
                success = bot.buy_stock(ticker, 15, current_prices)
                if success:
                    logging.info(f"{bot.name} - {ticker}: Low volatility detected. Bought 15 shares.")
            else:
                # Moderate volatility: Standard trade size
                success = bot.buy_stock(ticker, 10, current_prices)
                if success:
                    logging.info(f"{bot.name} - {ticker}: Moderate volatility detected. Bought 10 shares.")
        else:
            logging.warning(f"{bot.name} - {ticker}: No historical data available.")

# ---------------------------
# 6. TradingBot Class
# ---------------------------

class TradingBot:
    """Represents a trading bot with a specific strategy."""
    def __init__(self, name: str, initial_capital: float, strategy_name: str, trade_frequency_minutes: int = 1):
        self.name = name
        self.balance = initial_capital
        self.initial_balance = initial_capital
        self.portfolio: Dict[str, int] = {}  # {ticker: number_of_shares}
        self.strategy_name = strategy_name
        self.trade_frequency_minutes = trade_frequency_minutes
        eastern = pytz.timezone('US/Eastern')
        # Initialize last_trade_time as an offset-aware datetime far in the past
        self.last_trade_time = eastern.localize(datetime(1970, 1, 1))  # January 1, 1970
        self.strategy = self.initialize_strategy(strategy_name)
        self.gain_loss = 0
        self.trade_history: List[Dict] = []

    def initialize_strategy(self, strategy_name: str) -> BaseStrategy:
        """Initializes the strategy based on the strategy name."""
        strategies = {
            'Mean Reversion': MeanReversionStrategy(),
            'Momentum': MomentumStrategy(),
            'Moving Average Crossover': MovingAverageCrossoverStrategy(),
            'Wavelet Trend': WaveletTrendStrategy(),
            'Fourier Cycle': FourierCycleStrategy(),
            'Volatility Adjusted': VolatilityAdjustedStrategy()
            # Add more strategies here
        }
        return strategies.get(strategy_name, BaseStrategy())

    def decide_trade(self, stock_universe: List[str], current_prices: Dict[str, Optional[float]]):
        """Determines whether the bot should execute a trade based on trade frequency and market hours."""
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern)
        logging.info(f"Bot '{self.name}' checking trade conditions at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if current_time - self.last_trade_time >= timedelta(minutes=self.trade_frequency_minutes):
            if is_market_open(current_time):
                logging.info(f"Market is open. Bot '{self.name}' is executing its strategy.")
                for ticker in stock_universe:
                    self.strategy.execute(self, ticker, current_prices)
                self.last_trade_time = current_time
                save_bot_to_db(self)
            else:
                logging.info(f"Market is closed. Bot '{self.name}' will wait for the market to open.")

    def buy_stock(self, ticker: str, quantity: int, current_prices: Dict[str, Optional[float]]):
        """Buys a specified quantity of a stock if sufficient balance is available."""
        price = current_prices.get(ticker)
        if price is None:
            logging.warning(f"{self.name} attempted to buy {ticker}, but price is unavailable.")
            return False
        total_cost = price * quantity
        logging.info(f"{self.name} - Attempting to buy {quantity} shares of {ticker} at ${price:.2f} each. Total Cost: ${total_cost:.2f}")
        if self.balance >= total_cost:
            self.balance -= total_cost
            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + quantity
            self.calculate_gain_loss(current_prices)
            trade = {
                'timestamp': datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S"),
                'bot_name': self.name,
                'action': 'BUY',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'total_value': total_cost,
                'gain_loss': self.gain_loss
            }
            self.trade_history.append(trade)
            save_trade_to_db(trade)
            logging.info(f"{self.name} successfully bought {quantity} shares of {ticker} at ${price:.2f} each.")
            return True
        else:
            logging.warning(f"{self.name} has insufficient balance to buy {quantity} shares of {ticker}. Current Balance: ${self.balance:.2f}")
            return False

    def sell_stock(self, ticker: str, quantity: int, current_prices: Dict[str, Optional[float]]):
        """Sells a specified quantity of a stock if holdings are sufficient."""
        current_holding = self.portfolio.get(ticker, 0)
        logging.info(f"{self.name} holds {current_holding} shares of {ticker}. Attempting to sell {quantity} shares.")
        if current_holding >= quantity:
            price = current_prices.get(ticker)
            if price is None:
                logging.warning(f"{self.name} attempted to sell {ticker}, but price is unavailable.")
                return False
            total_return = price * quantity
            logging.info(f"{self.name} - Attempting to sell {quantity} shares of {ticker} at ${price:.2f} each. Total Return: ${total_return:.2f}")
            self.balance += total_return
            self.portfolio[ticker] -= quantity
            if self.portfolio[ticker] == 0:
                del self.portfolio[ticker]
            self.calculate_gain_loss(current_prices)
            trade = {
                'timestamp': datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S"),
                'bot_name': self.name,
                'action': 'SELL',
                'ticker': ticker,
                'quantity': quantity,
                'price': price,
                'total_value': total_return,
                'gain_loss': self.gain_loss
            }
            self.trade_history.append(trade)
            save_trade_to_db(trade)
            logging.info(f"{self.name} successfully sold {quantity} shares of {ticker} at ${price:.2f} each.")
            return True
        else:
            logging.warning(f"{self.name} does not have enough shares of {ticker} to sell. Current Holdings: {current_holding} shares.")
            return False

    def is_bankrupt(self) -> bool:
        """Checks if the bot is bankrupt."""
        return self.balance <= 0 and not self.portfolio

    def calculate_gain_loss(self, current_prices: Dict[str, Optional[float]]) -> float:
        """Calculates the gain or loss relative to the initial capital."""
        total_assets = self.balance
        for ticker, shares in self.portfolio.items():
            price = current_prices.get(ticker)
            if price:
                total_assets += price * shares
        self.gain_loss = total_assets - self.initial_balance
        logging.info(f"{self.name}: Calculated Gain/Loss = ${self.gain_loss:.2f}")
        return self.gain_loss

# ---------------------------
# 7. Helper Functions
# ---------------------------

def get_average_buy_price(bot_name: str, ticker: str) -> Optional[float]:
    """Calculates the average buy price for a specific ticker in a bot's portfolio."""
    with db_lock:
        try:
            with sqlite3.connect(r'G:\trade bots\trading_bot.db', timeout=30) as conn:
                query = """
                    SELECT price, quantity 
                    FROM trades 
                    WHERE bot_name = ? AND ticker = ? AND action = 'BUY'
                """
                buys = pd.read_sql_query(query, conn, params=(bot_name, ticker))
        except sqlite3.OperationalError as e:
            logging.error(f"OperationalError in get_average_buy_price: {e}")
            return None
    if not buys.empty:
        total_cost = (buys['price'] * buys['quantity']).sum()
        total_quantity = buys['quantity'].sum()
        if total_quantity > 0:
            return total_cost / total_quantity
    return None

def is_market_open(current_time: datetime) -> bool:
    """Checks if the stock market is currently open, including pre-market and after-hours."""
    eastern = pytz.timezone('US/Eastern')
    market_open = current_time.replace(hour=4, minute=0, second=0, microsecond=0)
    pre_market_close = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    regular_market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    after_hours_close = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
    
    # Check if today is a weekday
    if current_time.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Determine if within any of the trading sessions
    if market_open <= current_time < pre_market_close:
        return True  # Pre-market
    elif pre_market_close <= current_time < regular_market_close:
        return True  # Regular market
    elif regular_market_close <= current_time < after_hours_close:
        return True  # After-hours
    else:
        return False

def get_bot_trade_history(bot_name: str) -> pd.DataFrame:
    """Fetches trade history for a specific bot."""
    with db_lock:
        try:
            with sqlite3.connect(r'G:\trade bots\trading_bot.db', timeout=30) as conn:
                trades = pd.read_sql_query("SELECT * FROM trades WHERE bot_name = ? ORDER BY timestamp", conn, params=(bot_name,))
        except sqlite3.OperationalError as e:
            logging.error(f"OperationalError in get_bot_trade_history: {e}")
            trades = pd.DataFrame()
    return trades

def wavelet_decomposition(data, wavelet='db4', max_requested_level=4):
    """Performs wavelet decomposition and returns bulk trend and surface movements."""
    try:
        wavelet_obj = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(len(data), wavelet_obj.dec_len)
        actual_level = min(max_requested_level, max_level)
        if actual_level < max_requested_level:
            logging.warning(
                f"Requested wavelet decomposition level {max_requested_level} is too high for data length {len(data)} "
                f"and wavelet '{wavelet}'. Using level {actual_level} instead."
            )
        coeffs = pywt.wavedec(data, wavelet, level=actual_level)
        # Reconstruct the bulk trend by zeroing out detail coefficients
        bulk_trend = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], wavelet)
        # Ensure bulk_trend is the same length as data
        bulk_trend = bulk_trend[:len(data)]
        # Surface movements are the residuals
        surface_movements = np.array(data) - bulk_trend
        return bulk_trend, surface_movements
    except Exception as e:
        logging.error(f"Error in wavelet decomposition: {e}")
        # Return the original data and zero residuals in case of error
        return np.array(data), np.zeros_like(data)

# ---------------------------
# 8. Streamlit App Functions
# ---------------------------

def display_performance_graph(bots: List[TradingBot]):
    """Displays the cumulative gain/loss performance graph for all bots."""
    performance_data = []
    for bot in bots:
        trades = get_bot_trade_history(bot.name)
        if not trades.empty:
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            trades = trades.sort_values('timestamp')
            trades['Cumulative Gain/Loss'] = trades['gain_loss'].cumsum()
            performance_data.append(go.Scatter(
                x=trades['timestamp'],
                y=trades['Cumulative Gain/Loss'],
                mode='lines+markers',
                name=bot.name
            ))

    if performance_data:
        fig = go.Figure(data=performance_data)
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Cumulative Gain/Loss ($)",
            hovermode="x unified",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No performance data available.")

def display_bots_overview(bots: List[TradingBot]):
    """Displays an overview table of all bots."""
    st.subheader("🤖 Bots' Overview")
    if bots:
        bots_overview = []
        for bot in bots:
            bots_overview.append({
                'Bot Name': bot.name,
                'Strategy': bot.strategy_name,
                'Balance ($)': f"{bot.balance:.2f}",
                'Total Gain/Loss ($)': f"{bot.gain_loss:.2f}"
            })
        bots_df = pd.DataFrame(bots_overview)
        st.table(bots_df)
    else:
        st.write("No bots added yet.")

def display_trade_history():
    """Displays the overall trade history."""
    st.header("📊 Overall Trade History")
    with db_lock:
        try:
            with sqlite3.connect(r'G:\trade bots\trading_bot.db', timeout=30) as conn:
                trades = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100", conn)
        except sqlite3.OperationalError as e:
            logging.error(f"OperationalError in display_trade_history: {e}")
            trades = pd.DataFrame()
    if not trades.empty:
        trades_display = trades[['timestamp', 'bot_name', 'action', 'ticker', 'quantity', 'price', 'total_value', 'gain_loss']]
        trades_display = trades_display.rename(columns={
            'timestamp': 'Timestamp',
            'bot_name': 'Bot Name',
            'action': 'Action',
            'ticker': 'Ticker',
            'quantity': 'Quantity',
            'price': 'Price ($)',
            'total_value': 'Total Value ($)',
            'gain_loss': 'Gain/Loss ($)'
        })
        st.dataframe(trades_display)
    else:
        st.write("No trades executed yet.")

def display_bot_details(bots: List[TradingBot]):
    """Displays detailed information for a selected bot."""
    st.subheader("🔍 Bot Details")

    if not bots:
        st.write("No bots available. Please add a bot first.")
        return

    # Dropdown to select a bot
    bot_names = [bot.name for bot in bots]
    selected_bot_name = st.selectbox("Select a Bot", options=bot_names)

    # Fetch the selected bot
    selected_bot = next((bot for bot in bots if bot.name == selected_bot_name), None)

    if selected_bot:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("**Bot Overview**")
            st.write(f"**Name:** {selected_bot.name}")
            st.write(f"**Strategy:** {selected_bot.strategy_name}")
            st.write(f"**Initial Capital:** ${selected_bot.initial_balance:.2f}")
            st.write(f"**Current Balance:** ${selected_bot.balance:.2f}")
            st.write(f"**Total Gain/Loss:** ${selected_bot.gain_loss:.2f}")

        with col2:
            st.markdown("**Portfolio**")
            if selected_bot.portfolio:
                portfolio_data = []
                tickers = list(selected_bot.portfolio.keys())
                current_prices = get_current_prices(tickers)
                for ticker, qty in selected_bot.portfolio.items():
                    price = current_prices.get(ticker, None)
                    total_value = price * qty if isinstance(price, float) else None
                    avg_buy_price = get_average_buy_price(selected_bot.name, ticker)
                    gain_loss = ((price - avg_buy_price) * qty) if isinstance(price, float) and avg_buy_price else None
                    portfolio_data.append({
                        'Ticker': ticker,
                        'Quantity': qty,
                        'Current Price ($)': f"{price:.2f}" if isinstance(price, float) else 'N/A',
                        'Total Value ($)': f"{total_value:.2f}" if isinstance(total_value, float) else 'N/A',
                        'Gain/Loss ($)': f"{gain_loss:.2f}" if isinstance(gain_loss, float) else 'N/A'
                    })
                portfolio_df = pd.DataFrame(portfolio_data)
                st.table(portfolio_df)
            else:
                st.write("No holdings currently.")

        st.markdown("---")
        st.markdown("**Performance Over Time**")
        performance_data = get_bot_trade_history(selected_bot.name)
        if not performance_data.empty:
            performance_data['timestamp'] = pd.to_datetime(performance_data['timestamp'])
            performance_data = performance_data.sort_values('timestamp')
            performance_data['Cumulative Gain/Loss'] = performance_data['gain_loss'].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=performance_data['timestamp'],
                y=performance_data['Cumulative Gain/Loss'],
                mode='lines+markers',
                name='Cumulative Gain/Loss'
            ))
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Cumulative Gain/Loss ($)",
                hovermode="x unified",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No trade history available for this bot.")

        st.markdown("---")
        st.markdown("**Portfolio Allocation**")
        if selected_bot.portfolio:
            labels = list(selected_bot.portfolio.keys())
            values = []
            for ticker in labels:
                qty = selected_bot.portfolio[ticker]
                price = current_prices.get(ticker, 0)
                total = qty * price if isinstance(price, float) else 0
                values.append(total)
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("No holdings to display allocation.")

        st.markdown("---")
        st.markdown("**Individual Stock Performance**")
        if selected_bot.portfolio:
            selected_ticker = st.selectbox("Select a Ticker for Detailed Performance", options=list(selected_bot.portfolio.keys()))
            historical_data = get_historical_prices(selected_ticker, days=60)
            if historical_data is not None and not historical_data.empty:
                historical_data['timestamp'] = historical_data.index
                fig_stock = go.Figure()
                fig_stock.add_trace(go.Scatter(
                    x=historical_data['timestamp'],
                    y=historical_data['Close'],
                    mode='lines',
                    name='Close Price'
                ))
                fig_stock.update_layout(
                    title=f"{selected_ticker} Price Over Time",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_stock, use_container_width=True)
            else:
                st.write(f"No historical data available for {selected_ticker}.")
        else:
            st.write("No holdings to display stock performance.")

        st.markdown("---")
        st.markdown("**Detailed Trade History**")
        trades = get_bot_trade_history(selected_bot.name)
        if not trades.empty:
            trades_display = trades[['timestamp', 'action', 'ticker', 'quantity', 'price', 'total_value', 'gain_loss']]
            trades_display = trades_display.rename(columns={
                'timestamp': 'Timestamp',
                'action': 'Action',
                'ticker': 'Ticker',
                'quantity': 'Quantity',
                'price': 'Price ($)',
                'total_value': 'Total Value ($)',
                'gain_loss': 'Gain/Loss ($)'
            })
            st.dataframe(trades_display)
        else:
            st.write("No trades executed yet.")

def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="🤖 Trading Bot Simulator", layout="wide", page_icon="📊")
    st.title("🤖 Trading Bot Simulator")

    # Initialize the database
    init_db()

    # Load bots from the database into session state
    if 'bots' not in st.session_state:
        st.session_state.bots = load_bots_from_db()

    # Sidebar for adding new bots
    st.sidebar.header("📈 Add New Bot")
    bot_name = st.sidebar.text_input("Bot Name", key="bot_name_input")
    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)", 
        min_value=1000.0, 
        value=10000.0, 
        step=1000.0, 
        key="initial_capital_input"
    )
    strategy_option = st.sidebar.selectbox(
        "Select Strategy", [
            'Mean Reversion', 
            'Momentum', 
            'Moving Average Crossover', 
            'Wavelet Trend', 
            'Fourier Cycle', 
            'Volatility Adjusted'
        ], 
        key="strategy_option_selectbox"
    )
    trade_frequency_minutes = st.sidebar.selectbox(
        "Trade Frequency (Minutes)", 
        [1, 5, 10, 15, 30, 60], 
        key="trade_freq_selectbox"
    )

    if st.sidebar.button("➕ Add Bot", key="add_bot_button"):
        if bot_name:
            if any(bot.name == bot_name for bot in st.session_state.bots):
                st.sidebar.error("Bot with this name already exists.")
            else:
                bot = TradingBot(
                    name=bot_name,
                    initial_capital=initial_capital,
                    strategy_name=strategy_option,
                    trade_frequency_minutes=trade_frequency_minutes
                )
                st.session_state.bots.append(bot)
                save_bot_to_db(bot)
                st.sidebar.success(f"Bot '{bot_name}' added successfully!")
                logging.info(f"Bot '{bot_name}' added with strategy '{strategy_option}' and initial capital ${initial_capital:.2f}.")
        else:
            st.sidebar.error("Please enter a bot name.")

    # Fetch stock data
    stock_universe = get_stock_universe()
    valid_stock_universe = validate_tickers(stock_universe)
    current_prices = get_current_prices(valid_stock_universe)
    daily_performance = get_daily_performance(valid_stock_universe)

    # **Recalculate gain_loss for each bot after fetching current prices**
    for bot in st.session_state.bots:
        bot.calculate_gain_loss(current_prices)

    # Create tabs
    tabs = st.tabs(["📊 Overview", "🔍 Bot Details"])

    with tabs[0]:
        # Display the performance graph once
        st.subheader("📈 Bots' Performance")
        display_performance_graph(st.session_state.bots)

        # Display the bots' overview once
        display_bots_overview(st.session_state.bots)

        # Display overall trade history once
        display_trade_history()

    with tabs[1]:
        display_bot_details(st.session_state.bots)

    # Automatic trading cycle execution
    if valid_stock_universe:
        for bot in st.session_state.bots:
            bot.decide_trade(valid_stock_universe, current_prices)
    else:
        st.error("No valid tickers available to trade.")
        logging.error("No valid tickers available to trade.")

    # Auto-refresh every minute to simulate continuous trading
    # Ensure it's only initialized once per session
    if 'refresh_initialized' not in st.session_state:
        st_autorefresh(interval=60 * 1000, key="trading_refresh")
        st.session_state['refresh_initialized'] = True

if __name__ == "__main__":
    main()
