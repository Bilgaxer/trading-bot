import ccxt
import time
import requests
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import ta
from datetime import datetime
import json
from db_helper import DatabaseHelper
import os
from dotenv import load_dotenv

print("Initializing trading bot...")

# Load environment variables
load_dotenv()

# Initialize database connection
db = DatabaseHelper()
db.connect()

# Trading pair configuration
symbol = 'BTC/USDT'
timeframe = '5m'  # Changed from 1m to 5m
news_check_interval = 600  # 10 minutes
last_news_check = 0
last_sentiment_score = 0
last_summary_time = 0
SUMMARY_INTERVAL = 1800  # 30 minutes in seconds

# Price history for charting
price_history = []
MAX_HISTORY_LENGTH = 100

# Exchange setup for futures market data
exchange = ccxt.binance({
    'apiKey': 'Glun3rOW3f825CECPvnOPKbsVwDOpniL4fzsyjPoeq5s827qhGojZnxOQltJTx1U',
    'secret': 'vwvlFJHu5RTZhvBjmwtVa9gWXAeCDLrw1ONjgxyX08izEiCHK7v4IjKoh8lNRKMv',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True,
        'recvWindow': 60000
    }
})

# Test basic market data access
try:
    print("Testing market data access...")
    markets = exchange.load_markets()
    print("Successfully loaded market data")
except Exception as e:
    print(f"Error loading markets: {e}")

# Test account access
try:
    print("\nTesting account access...")
    balance = exchange.fetch_balance()
    print("Successfully connected to Binance Futures")
    print(f"USDT Balance: {balance['USDT']['free'] if 'USDT' in balance else 0:.2f}")
except Exception as e:
    print(f"Error accessing account: {e}")

# Test futures-specific endpoints
try:
    print("\nTesting futures endpoints...")
    positions = exchange.fetch_positions([symbol])
    print("Successfully accessed futures positions")
except Exception as e:
    print(f"Error accessing futures positions: {e}")

# Set leverage to 10x
try:
    print("\nSetting leverage...")
    exchange.fapiPrivatePostLeverage({
        'symbol': 'BTCUSDT',
        'leverage': 10
    })
    print("Successfully set leverage to 10x")
except Exception as e:
    print(f"Error setting leverage: {e}")
    print("Continuing with paper trading mode only...")

print("\nStarting main trading loop...")

# Trading state variables
active_position = {
    'side': None,  # 'long' or 'short'
    'entry_price': None,
    'position_size': 0,
    'num_scales': 0,
    'avg_entry_price': None,
    'trailing_stop': None
}

# Trading parameters
MAX_SCALES = 3  # Maximum number of scaling entries
SCALE_INTERVAL = 0.02  # 2% price movement for each scale
TRAILING_STOP_INITIAL = 0.02  # 2% initial trailing stop
TRAILING_STOP_STEP = 0.005  # 0.5% step for trailing stop
RISK_PER_TRADE = 0.01  # 1% risk per trade
DONCHIAN_PERIOD = 20  # Period for Donchian Channel
VOLUME_THRESHOLD = 1.3  # Volume must be 1.3x average
POSITION_SIZE_BOOST = 1.2  # Increase position size by 20% if secondary conditions met
TP_PERCENTAGE = 0.007  # 0.7% take profit
INITIAL_STOP_PERCENTAGE = 0.004  # 0.4% initial stop loss

# Volume-based position sizing thresholds
VOLUME_THRESHOLD_1 = 1.3  # Initial position threshold
VOLUME_THRESHOLD_2 = 1.7  # Add to position threshold
VOLUME_THRESHOLD_3 = 2.0  # Max position threshold
POSITION_SIZE_1 = 0.005  # 0.5% of capital for initial position
POSITION_SIZE_2 = 0.01   # 1% of capital for second addition
POSITION_SIZE_3 = 0.05   # 5% of capital max position size
MAX_POSITION_SIZE = 0.05  # 5% of total capital cap per trade

# FinBERT setup
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Paper trading simulation state
paper_trading = {
    'balance': 117.39,  # Starting with 117.39 USDT (equivalent to 10,000 INR)
    'initial_balance': 117.39,
    'position': {
        'side': None,  # 'long' or 'short'
        'size': 0.0,
        'entry_price': 0.0,
        'leverage': 10,
        'unrealized_pnl': 0.0,
        'stop_loss': 0.0,
        'take_profit': 0.0
    },
    'trades': [],
    'total_pnl': 0.0,
    'win_trades': 0,
    'loss_trades': 0
}

def calculate_vwap(df):
    """Calculate VWAP (Volume Weighted Average Price)"""
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    return df['vwap']

def calculate_emas(df):
    """Calculate EMA-9 and EMA-21"""
    df['ema9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema21'] = ta.trend.ema_indicator(df['close'], window=21)
    return df['ema9'], df['ema21']

def check_volume_spike(df):
    """Check if current volume is 1.05x the 20-period average"""
    volume_ma = df['volume'].rolling(window=20).mean()
    return df['volume'].iloc[-1] > volume_ma.iloc[-1] * 1.05

def calculate_donchian_channel(df, period=DONCHIAN_PERIOD):
    """Calculate Donchian Channel"""
    df['donchian_high'] = df['high'].rolling(window=period).max()
    df['donchian_low'] = df['low'].rolling(window=period).min()
    df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
    return df['donchian_high'], df['donchian_low'], df['donchian_mid']

def calculate_supertrend(df, period=10, multiplier=3):
    """Calculate SuperTrend indicator"""
    hl2 = (df['high'] + df['low']) / 2
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
    
    # Calculate SuperTrend
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # Initialize SuperTrend
    supertrend = [True] * len(df)
    final_upperband = [upperband.iloc[0]] * len(df)
    final_lowerband = [lowerband.iloc[0]] * len(df)
    
    for i in range(1, len(df)):
        curr_close = df['close'].iloc[i]
        prev_close = df['close'].iloc[i-1]
        
        if curr_close > final_upperband[i-1]:
            supertrend[i] = True
        elif curr_close < final_lowerband[i-1]:
            supertrend[i] = False
        else:
            supertrend[i] = supertrend[i-1]
            
            if supertrend[i] and curr_close <= final_upperband[i-1]:
                final_upperband[i] = min(upperband.iloc[i], final_upperband[i-1])
            if not supertrend[i] and curr_close >= final_lowerband[i-1]:
                final_lowerband[i] = max(lowerband.iloc[i], final_lowerband[i-1])
    
    df['supertrend'] = supertrend
    df['supertrend_upper'] = final_upperband
    df['supertrend_lower'] = final_lowerband
    return df['supertrend']

def fetch_price_data():
    """Fetch price data and calculate indicators"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate indicators
        df['vwap'] = calculate_vwap(df)
        df['donchian_high'], df['donchian_low'], df['donchian_mid'] = calculate_donchian_channel(df)
        df['supertrend'] = calculate_supertrend(df)
        df['atr'] = calculate_atr(df)
        
        return df
    except Exception as e:
        print(f"Error fetching price data: {e}")
        return None

def calculate_rsi(df, periods=14):
    # Calculate RSI
    close_delta = df['close'].diff()
    
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # Calculate the EWMA
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    
    # Calculate RS and RSI
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    return rsi

def calculate_atr(df):
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

def calculate_atr_threshold(df):
    atr = calculate_atr(df)
    median_atr = atr.median()
    return median_atr * 1.05  # Lowered from 1.1 to 1.05 (5% above median)

def check_primary_conditions(df):
    """Check primary trading conditions"""
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    
    # Check Donchian Channel breakout
    donchian_breakout_up = prev_price <= df['donchian_high'].iloc[-2] and current_price > df['donchian_high'].iloc[-1]
    donchian_breakout_down = prev_price >= df['donchian_low'].iloc[-2] and current_price < df['donchian_low'].iloc[-1]
    
    # Check volume condition
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1]
    
    # Determine position size based on volume
    position_size = 0
    if volume_ratio >= VOLUME_THRESHOLD_3:
        position_size = POSITION_SIZE_3
    elif volume_ratio >= VOLUME_THRESHOLD_2:
        position_size = POSITION_SIZE_2
    elif volume_ratio >= VOLUME_THRESHOLD_1:
        position_size = POSITION_SIZE_1
    
    return {
        'long': donchian_breakout_up and volume_ratio >= VOLUME_THRESHOLD_1,
        'short': donchian_breakout_down and volume_ratio >= VOLUME_THRESHOLD_1,
        'volume_spike': volume_ratio >= VOLUME_THRESHOLD_1,
        'volume_ratio': volume_ratio,
        'position_size': position_size
    }

def check_secondary_conditions(df):
    """Check secondary confirmation conditions"""
    current_price = df['close'].iloc[-1]
    
    # VWAP condition
    above_vwap = current_price > df['vwap'].iloc[-1]
    
    # SuperTrend condition
    supertrend_bullish = df['supertrend'].iloc[-1]
    
    return {
        'long': above_vwap or supertrend_bullish,
        'short': (not above_vwap) or (not supertrend_bullish)
    }

def save_bot_data(df, last_price, atr_value):
    try:
        # Calculate performance metrics
        total_trades = len(paper_trading['trades'])
        win_rate = (paper_trading['win_trades'] / total_trades * 100) if total_trades > 0 else 0
        roi = ((paper_trading['balance'] / paper_trading['initial_balance']) - 1) * 100

        # Get recent trades (ensure they're serializable)
        recent_trades = []
        for trade in paper_trading['trades'][-10:]:
            recent_trades.append({
                'side': str(trade['side']),
                'entry': float(trade['entry']),
                'exit': float(trade['exit']),
                'pnl': float(trade['pnl']),
                'reason': str(trade['reason']),
                'timestamp': str(trade['timestamp'])
            })

        # Get trading conditions (convert numpy bools to Python bools)
        long_conditions = check_primary_conditions(df)
        short_conditions = check_primary_conditions(df)
        
        # Convert numpy bools to Python bools
        long_conditions = {k: bool(v) for k, v in long_conditions.items()}
        short_conditions = {k: bool(v) for k, v in short_conditions.items()}

        # Current values for display (ensure all values are native Python types)
        current_values = {
            'price': float(last_price),
            'vwap': float(df['vwap'].iloc[-1]),
            'donchian_high': float(df['donchian_high'].iloc[-1]),
            'donchian_low': float(df['donchian_low'].iloc[-1]),
            'donchian_mid': float(df['donchian_mid'].iloc[-1]),
            'volume_spike_ratio': float(df['volume'].iloc[-1] / df['volume'].rolling(window=20).mean().iloc[-1]),
            'atr': float(atr_value),
            'atr_threshold': float(df['atr'].rolling(window=20).median().iloc[-1]),
            'supertrend': bool(df['supertrend'].iloc[-1])
        }

        # Position data (ensure all values are native Python types)
        position_data = {
            'side': str(paper_trading['position']['side']) if paper_trading['position']['side'] else None,
            'size': float(paper_trading['position']['size']),
            'entry_price': float(paper_trading['position']['entry_price']),
            'leverage': int(paper_trading['position']['leverage']),
            'unrealized_pnl': float(paper_trading['position']['unrealized_pnl']),
            'stop_loss': float(paper_trading['position']['stop_loss']),
            'take_profit': float(paper_trading['position']['take_profit'])
        }

        # Convert DataFrame to list of dictionaries for price history
        price_history = []
        for idx, row in df.iterrows():
            price_history.append({
                'timestamp': str(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'vwap': float(row['vwap']),
                'donchian_high': float(row['donchian_high']),
                'donchian_low': float(row['donchian_low']),
                'donchian_mid': float(row['donchian_mid']),
                'supertrend': bool(row['supertrend'])
            })

        data = {
            'balance': float(paper_trading['balance']),
            'initial_balance': float(paper_trading['initial_balance']),
            'total_pnl': float(paper_trading['total_pnl']),
            'win_trades': int(paper_trading['win_trades']),
            'total_trades': int(total_trades),
            'win_rate': float(win_rate),
            'roi': float(roi),
            'recent_trades': recent_trades,
            'position': position_data,
            'trading_conditions': {
                'long_conditions': long_conditions,
                'short_conditions': short_conditions,
                'current_values': current_values
            },
            'market_data': {
                'current_price': float(last_price),
                'current_atr': float(atr_value),
                'funding_rate': float(current_funding_rate)
            },
            'performance_summary': {
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'daily_pnl': float(calculate_daily_pnl()),
                'avg_trade_pnl': float(calculate_avg_trade_pnl()),
                'best_trade': float(get_best_trade()),
                'worst_trade': float(get_worst_trade())
            },
            'price_history': price_history  # Add price history to the data
        }

        # Save to database
        db.save_bot_data(data)
        
    except Exception as e:
        print(f"Error saving bot data: {e}")
        import traceback
        traceback.print_exc()  # Print full error traceback for debugging

def open_position(side, size, price, stop_loss=None, take_profit=None):
    """Open a new position with updated risk management"""
    if paper_trading['position']['side'] is not None:
        return False
    
    # Calculate position size based on risk
    position_size = calculate_position_size(price)
    
    # Check secondary conditions for position size boost
    df = fetch_price_data()
    secondary_conditions = check_secondary_conditions(df)
    if secondary_conditions[side]:
        position_size *= POSITION_SIZE_BOOST
    
    # Set stop loss and take profit
    if stop_loss is None:
        stop_loss = price * (1 - INITIAL_STOP_PERCENTAGE if side == 'long' else 1 + INITIAL_STOP_PERCENTAGE)
    
    if take_profit is None:
        take_profit = price * (1 + TP_PERCENTAGE if side == 'long' else 1 - TP_PERCENTAGE)
    
    paper_trading['position'] = {
        'side': side,
        'size': position_size,
        'entry_price': price,
        'leverage': 10,
        'unrealized_pnl': 0.0,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'trailing_activated': False
    }
    
    return True

def update_trailing_stop(current_price):
    """Update trailing stop based on SuperTrend"""
    position = paper_trading['position']
    if position['side'] is None:
        return
    
    df = fetch_price_data()
    supertrend = df['supertrend'].iloc[-1]
    
    # Update stop loss based on SuperTrend flip
    if position['side'] == 'long' and not supertrend:
        close_position(current_price, 'SuperTrend turned bearish')
    elif position['side'] == 'short' and supertrend:
        close_position(current_price, 'SuperTrend turned bullish')
    
    # Normal trailing stop logic
    profit_pct = ((current_price / position['entry_price'] - 1) * 100 * 
                  (1 if position['side'] == 'long' else -1))
    
    if profit_pct >= 0.3:  # Start trailing at 0.3%
        position['trailing_activated'] = True
        
        if position['side'] == 'long':
            new_stop = current_price * 0.997  # 0.3% below current price
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
        else:
            new_stop = current_price * 1.003  # 0.3% above current price
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop

# Global variables for funding rate checks
last_funding_check = 0
FUNDING_CHECK_INTERVAL = 300  # 5 minutes
current_funding_rate = 0

def update_funding_rate():
    global last_funding_check, current_funding_rate
    current_time = time.time()
    
    if current_time - last_funding_check >= FUNDING_CHECK_INTERVAL:
        try:
            current_funding_rate = get_funding_rate()
            print(f"[INFO] Updated funding rate: {current_funding_rate:.6f}")
            last_funding_check = current_time
        except Exception as e:
            print(f"Error updating funding rate: {e}")

def get_funding_rate():
    try:
        # For testing, return a neutral funding rate
        return 0.0001
        # funding_rates = exchange.fetch_funding_rate(symbol)
        # return funding_rates['fundingRate']
    except Exception as e:
        print(f"Error fetching funding rate: {e}")
        return 0.0  # Return neutral funding rate on error

def fetch_news():
    try:
        url = "https://cryptopanic.com/api/v1/posts/?auth_token=1fb24bfbfc7b965303497fe1703b10663f986aa5&currencies=BTC"
        res = requests.get(url)
        if res.status_code != 200:
            print(f"Error fetching news: Status code {res.status_code}")
            return []
        data = res.json()
        if not data or 'results' not in data:
            print("No news data available")
            return []
        headlines = [x['title'] for x in data['results']]
        return headlines
    except requests.exceptions.RequestException as e:
        print(f"Network error while fetching news: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing news data: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in fetch_news: {e}")
        return []

# Constants for sentiment check
SENTIMENT_TIMEOUT = 1200  # 20 minutes in seconds
last_sentiment_update = 0

def is_sentiment_valid():
    global last_sentiment_update
    current_time = time.time()
    return (current_time - last_sentiment_update) < SENTIMENT_TIMEOUT

def analyze_sentiment(news_list):
    global last_sentiment_update
    if not news_list:
        return 0
    sentiments = []
    for news in news_list:
        tokens = tokenizer(news, return_tensors='pt', truncation=True, padding=True)
        output = model(**tokens)
        scores = softmax(output.logits.detach().numpy()[0])
        sentiments.append(scores[2] - scores[0])  # Positive - Negative
    avg_sentiment = np.mean(sentiments)
    last_sentiment_update = time.time()  # Update the timestamp
    return avg_sentiment

def update_position_pnl(current_price):
    if paper_trading['position']['side'] is None:
        return
    
    size = paper_trading['position']['size']
    entry = paper_trading['position']['entry_price']
    leverage = paper_trading['position']['leverage']
    
    if paper_trading['position']['side'] == 'long':
        paper_trading['position']['unrealized_pnl'] = (current_price - entry) * size * leverage
    else:  # short
        paper_trading['position']['unrealized_pnl'] = (entry - current_price) * size * leverage
    
    # Check for stop loss or take profit hits
    if paper_trading['position']['side'] == 'long':
        if current_price <= paper_trading['position']['stop_loss']:
            close_position(current_price, 'Stop loss hit')
        elif current_price >= paper_trading['position']['take_profit']:
            close_position(current_price, 'Take profit hit')
    else:  # short
        if current_price >= paper_trading['position']['stop_loss']:
            close_position(current_price, 'Stop loss hit')
        elif current_price <= paper_trading['position']['take_profit']:
            close_position(current_price, 'Take profit hit')

def close_position(price, reason=''):
    if paper_trading['position']['side'] is None:
        return
    
    pnl = paper_trading['position']['unrealized_pnl']
    paper_trading['total_pnl'] += pnl
    paper_trading['balance'] += pnl
    
    # Update win/loss count
    if pnl > 0:
        paper_trading['win_trades'] += 1
    
    print(f"[PAPER] Closed {paper_trading['position']['side']} position at {price:.2f} USDT")
    print(f"[PAPER] PnL: {pnl:.2f} USDT ({reason})")
    print(f"[PAPER] New Balance: {paper_trading['balance']:.2f} USDT")
    
    # Record the trade with timestamp
    paper_trading['trades'].append({
        'side': paper_trading['position']['side'],
        'entry': paper_trading['position']['entry_price'],
        'exit': price,
        'pnl': pnl,
        'reason': reason,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Reset position
    paper_trading['position'] = {
        'side': None,
        'size': 0.0,
        'entry_price': 0.0,
        'leverage': 10,
        'unrealized_pnl': 0.0,
        'stop_loss': 0.0,
        'take_profit': 0.0
    }

def calculate_position_size(current_price, volume_ratio):
    """Calculate position size based on volume ratio"""
    if volume_ratio >= VOLUME_THRESHOLD_3:
        return (paper_trading['balance'] * POSITION_SIZE_3) / current_price
    elif volume_ratio >= VOLUME_THRESHOLD_2:
        return (paper_trading['balance'] * POSITION_SIZE_2) / current_price
    elif volume_ratio >= VOLUME_THRESHOLD_1:
        return (paper_trading['balance'] * POSITION_SIZE_1) / current_price
    return 0

def calculate_scale_size(current_price):
    try:
        # For testing, return a fixed reduced size
        return 0.001 * (0.7 ** active_position['num_scales'])
        # balance = exchange.fetch_balance()
        # usdt_balance = balance['USDT']['free']
        # scale_factor = 0.7 ** active_position['num_scales']
        # risk_amount = usdt_balance * RISK_PER_TRADE * scale_factor
        # position_size = (risk_amount * 10) / current_price
        # return position_size
    except Exception as e:
        print(f"Error calculating scale size: {e}")
        return 0.001

def check_scale_conditions(df):
    if not active_position['side'] or active_position['num_scales'] >= MAX_SCALES:
        return False

    last_price = df['close'].iloc[-1]
    rsi = calculate_rsi(df).iloc[-1]
    volume_spike = check_volume_spike(df)

    if active_position['side'] == 'long':
        # Scale into longs if price is higher and RSI is not overbought
        price_condition = last_price > active_position['entry_price'] * (1 + SCALE_INTERVAL * active_position['num_scales'])
        return price_condition and rsi < 70 and volume_spike

    elif active_position['side'] == 'short':
        # Scale into shorts if price is lower and RSI is not oversold
        price_condition = last_price < active_position['entry_price'] * (1 - SCALE_INTERVAL * active_position['num_scales'])
        return price_condition and rsi > 30 and volume_spike

    return False

def execute_scale_in(side, current_price):
    try:
        scale_size = calculate_scale_size(current_price)
        
        # Simulate order execution
        # if side == 'long':
        #     order = exchange.create_market_buy_order(symbol, scale_size)
        # else:
        #     order = exchange.create_market_sell_order(symbol, scale_size)

        # Update position tracking
        total_size = active_position['position_size'] + scale_size
        avg_price = ((active_position['avg_entry_price'] * active_position['position_size']) + 
                    (current_price * scale_size)) / total_size
        
        active_position['position_size'] = total_size
        active_position['avg_entry_price'] = avg_price
        active_position['num_scales'] += 1

        print(f"[TEST] Scaled {side} position: Size={scale_size:.6f} BTC, Avg Entry={avg_price:.2f}, Scale #{active_position['num_scales']}")
        
        # Update trailing stop with new position size
        update_trailing_stop(current_price)

    except Exception as e:
        print(f"Error scaling position: {e}")

def display_performance_summary():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_trades = len(paper_trading['trades'])
    win_rate = (paper_trading['win_trades'] / total_trades * 100) if total_trades > 0 else 0
    
    print("\n" + "="*50)
    print(f"Performance Summary at {current_time}")
    print("="*50)
    print(f"Initial Balance: {paper_trading['initial_balance']:.2f} USDT")
    print(f"Current Balance: {paper_trading['balance']:.2f} USDT")
    print(f"Total PnL: {paper_trading['total_pnl']:.2f} USDT")
    print(f"Return: {((paper_trading['balance'] / paper_trading['initial_balance']) - 1) * 100:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {paper_trading['win_trades']}")
    print(f"Losing Trades: {total_trades - paper_trading['win_trades']}")
    print(f"Win Rate: {win_rate:.2f}%")
    
    if paper_trading['position']['side']:
        print("\nCurrent Position:")
        print(f"Side: {paper_trading['position']['side']}")
        print(f"Size: {paper_trading['position']['size']:.6f} BTC")
        print(f"Entry Price: {paper_trading['position']['entry_price']:.2f} USDT")
        print(f"Unrealized PnL: {paper_trading['position']['unrealized_pnl']:.2f} USDT")
    print("="*50 + "\n")

def log_sentiment(sentiment_score, news_list):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open('sentiment_log.txt', 'a') as f:
            f.write(f"\n{timestamp} - Sentiment Score: {sentiment_score:.3f}\n")
            if news_list:
                f.write("Recent Headlines:\n")
                for headline in news_list[:5]:  # Log last 5 headlines
                    f.write(f"- {headline}\n")
            f.write("-" * 50 + "\n")
    except Exception as e:
        print(f"Error logging sentiment: {e}")

def calculate_daily_pnl():
    try:
        today_trades = [t for t in paper_trading['trades'] 
                       if datetime.strptime(t['timestamp'], "%Y-%m-%d %H:%M:%S").date() == datetime.now().date()]
        return sum(t['pnl'] for t in today_trades)
    except Exception as e:
        print(f"Error calculating daily PnL: {e}")
        return 0.0

def calculate_avg_trade_pnl():
    try:
        total_trades = len(paper_trading['trades'])
        if total_trades == 0:
            return 0.0
        return sum(t['pnl'] for t in paper_trading['trades']) / total_trades
    except Exception as e:
        print(f"Error calculating average trade PnL: {e}")
        return 0.0

def get_best_trade():
    try:
        if not paper_trading['trades']:
            return 0.0
        return max(t['pnl'] for t in paper_trading['trades'])
    except Exception as e:
        print(f"Error getting best trade: {e}")
        return 0.0

def get_worst_trade():
    try:
        if not paper_trading['trades']:
            return 0.0
        return min(t['pnl'] for t in paper_trading['trades'])
    except Exception as e:
        print(f"Error getting worst trade: {e}")
        return 0.0

while True:
    try:
        current_time = time.time()
        df = fetch_price_data()
        if df is None:
            print("Error fetching price data, retrying...")
            time.sleep(5)
            continue
            
        atr = calculate_atr(df)
        atr_threshold = calculate_atr_threshold(df)
        update_funding_rate()
        last_price = df['close'].iloc[-1]
        
        # Update paper trading position PnL
        update_position_pnl(last_price)
        
        # Save data for dashboard (more frequent updates when position is active)
        if paper_trading['position']['side'] is not None:
            save_bot_data(df, last_price, atr.iloc[-1])
            # Display active position info
            print(f"\n[ACTIVE POSITION] {paper_trading['position']['side'].upper()}")
            print(f"Current Price: {last_price:.2f} USDT")
            print(f"Entry Price: {paper_trading['position']['entry_price']:.2f} USDT")
            print(f"Position Size: {paper_trading['position']['size']:.6f} BTC")
            print(f"Unrealized PnL: {paper_trading['position']['unrealized_pnl']:.2f} USDT")
            print(f"Stop Loss: {paper_trading['position']['stop_loss']:.2f} USDT")
            print(f"Take Profit: {paper_trading['position']['take_profit']:.2f} USDT")
            print(f"Time in Trade: {(datetime.now() - datetime.strptime(paper_trading['trades'][-1]['timestamp'], '%Y-%m-%d %H:%M:%S')).seconds} seconds")
            print("-" * 50)
        else:
            # Regular interval updates when no position
            if current_time - last_summary_time >= SUMMARY_INTERVAL:
                save_bot_data(df, last_price, atr.iloc[-1])
                display_performance_summary()
                last_summary_time = current_time
        
        # Get trading conditions
        long_conditions = check_primary_conditions(df)
        short_conditions = check_primary_conditions(df)
        
        # Get current price once for all conditions
        current_price = df['close'].iloc[-1]
        
        # Display trading conditions
        print("\n=== Trading Conditions ===")
        print(f"Current Price: {current_price:.2f} USDT")
        print(f"Volume Ratio: {long_conditions['volume_ratio']:.2f}x")
        print("Long Entry Conditions:")
        print("Primary Conditions:")
        print(f"  {'YES' if long_conditions['long'] else 'NO '} Price crosses above Donchian High")
        print(f"  {'YES' if long_conditions['volume_spike'] else 'NO '} Volume > {VOLUME_THRESHOLD_1}x 20MA")
        
        # Get secondary conditions
        secondary = check_secondary_conditions(df)
        print("\nSecondary Conditions:")
        print(f"  {'YES' if current_price > df['vwap'].iloc[-1] else 'NO '} Price Above VWAP")
        print(f"  {'YES' if df['supertrend'].iloc[-1] else 'NO '} SuperTrend Bullish")
        
        print("\nShort Entry Conditions:")
        print("Primary Conditions:")
        print(f"  {'YES' if short_conditions['short'] else 'NO '} Price crosses below Donchian Low")
        print(f"  {'YES' if short_conditions['volume_spike'] else 'NO '} Volume > {VOLUME_THRESHOLD_1}x 20MA")
        
        print("\nSecondary Conditions:")
        print(f"  {'YES' if current_price < df['vwap'].iloc[-1] else 'NO '} Price Below VWAP")
        print(f"  {'YES' if not df['supertrend'].iloc[-1] else 'NO '} SuperTrend Bearish")
        print("=" * 30)
        
        if paper_trading['position']['side']:
            print(f"\n[PAPER] Active {paper_trading['position']['side']} position:")
            print(f"Size: {paper_trading['position']['size']:.6f} BTC")
            print(f"Entry: {paper_trading['position']['entry_price']:.2f} USDT")
            print(f"Current PnL: {paper_trading['position']['unrealized_pnl']:.2f} USDT")
            print(f"Stop Loss: {paper_trading['position']['stop_loss']:.2f} USDT")
            print(f"Take Profit: {paper_trading['position']['take_profit']:.2f} USDT")
            print("[PAPER] Position active - waiting for exit signals")
        else:
            # Trading logic for opening new positions
            if long_conditions['long'] and long_conditions['volume_spike']:
                entry_price = last_price
                position_size = calculate_position_size(entry_price, long_conditions['volume_ratio'])
                if position_size > 0:
                    stop_loss = entry_price * 0.996
                    take_profit = entry_price * 1.007
                    open_position('long', position_size, entry_price, stop_loss, take_profit)

            if short_conditions['short'] and short_conditions['volume_spike']:
                entry_price = last_price
                position_size = calculate_position_size(entry_price, short_conditions['volume_ratio'])
                if position_size > 0:
                    stop_loss = entry_price * 1.004
                    take_profit = entry_price * 0.993
                    open_position('short', position_size, entry_price, stop_loss, take_profit)

    except Exception as e:
        print(f"Error in main loop: {e}")

    # Shorter sleep when position is active
    sleep_time = 5 if paper_trading['position']['side'] is not None else 60
    time.sleep(sleep_time)
