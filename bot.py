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
timeframe = '1m'
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

def save_bot_data(df, last_price, rsi_value, atr_value):
    try:
        # Calculate performance metrics
        total_trades = len(paper_trading['trades'])
        win_rate = (paper_trading['win_trades'] / total_trades * 100) if total_trades > 0 else 0
        roi = ((paper_trading['balance'] / paper_trading['initial_balance']) - 1) * 100

        # Get recent trades
        recent_trades = []
        for trade in paper_trading['trades'][-10:]:
            trade_data = {
                'side': trade['side'],
                'entry': float(trade['entry']),
                'exit': float(trade['exit']),
                'pnl': float(trade['pnl']),
                'reason': trade['reason'],
                'timestamp': trade.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }
            recent_trades.append(trade_data)

        # Convert volume spike to native Python boolean
        volume_spike = bool(check_volume_spike(df))

        # Calculate trading conditions
        recent_high = df['high'].iloc[-5:].max()
        recent_low = df['low'].iloc[-5:].min()
        breakout = last_price > recent_high
        breakdown = last_price < recent_low
        sentiment_condition_long = (not is_sentiment_valid()) or (last_sentiment_score > 0.05)
        sentiment_condition_short = (not is_sentiment_valid()) or (last_sentiment_score < -0.05)
        funding_condition_long = current_funding_rate < 0.005
        funding_condition_short = current_funding_rate > -0.005
        atr_condition = atr_value > calculate_atr_threshold(df)

        # Track all conditions
        trading_conditions = {
            'long_conditions': {
                'breakout': bool(breakout),
                'volume_spike': bool(volume_spike),
                'funding_rate': bool(funding_condition_long),
                'atr_threshold': bool(atr_condition),
                'sentiment': bool(sentiment_condition_long)
            },
            'short_conditions': {
                'breakdown': bool(breakdown),
                'volume_spike': bool(volume_spike),
                'funding_rate': bool(funding_condition_short),
                'atr_threshold': bool(atr_condition),
                'sentiment': bool(sentiment_condition_short)
            },
            'current_values': {
                'price': float(last_price),
                'recent_high': float(recent_high),
                'recent_low': float(recent_low),
                'volume_spike_ratio': float(df['volume'].iloc[-1] / df['volume'].rolling(window=20).mean().iloc[-1]),
                'funding_rate': float(current_funding_rate),
                'atr': float(atr_value),
                'atr_threshold': float(calculate_atr_threshold(df)),
                'sentiment_score': float(last_sentiment_score)
            }
        }

        data = {
            'balance': float(paper_trading['balance']),
            'initial_balance': float(paper_trading['initial_balance']),
            'total_pnl': float(paper_trading['total_pnl']),
            'win_trades': int(paper_trading['win_trades']),
            'total_trades': total_trades,
            'win_rate': float(win_rate),
            'roi': float(roi),
            'recent_trades': recent_trades,
            'position': {
                'side': paper_trading['position']['side'],
                'size': float(paper_trading['position']['size']),
                'entry_price': float(paper_trading['position']['entry_price']),
                'leverage': int(paper_trading['position']['leverage']),
                'unrealized_pnl': float(paper_trading['position']['unrealized_pnl']),
                'stop_loss': float(paper_trading['position']['stop_loss']),
                'take_profit': float(paper_trading['position']['take_profit'])
            },
            'market_data': {
                'current_price': float(last_price),
                'current_rsi': float(rsi_value),
                'current_atr': float(atr_value),
                'sentiment_score': float(last_sentiment_score),
                'funding_rate': float(current_funding_rate),
                'volume_spike': volume_spike
            },
            'trading_conditions': trading_conditions,
            'performance_summary': {
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'daily_pnl': float(sum(t['pnl'] for t in recent_trades)),
                'avg_trade_pnl': float(sum(t['pnl'] for t in paper_trading['trades']) / total_trades if total_trades > 0 else 0),
                'best_trade': float(max((t['pnl'] for t in paper_trading['trades']), default=0)),
                'worst_trade': float(min((t['pnl'] for t in paper_trading['trades']), default=0)),
                'avg_win_size': float(sum(t['pnl'] for t in paper_trading['trades'] if t['pnl'] > 0) / paper_trading['win_trades'] if paper_trading['win_trades'] > 0 else 0),
                'avg_loss_size': float(sum(t['pnl'] for t in paper_trading['trades'] if t['pnl'] < 0) / (total_trades - paper_trading['win_trades']) if (total_trades - paper_trading['win_trades']) > 0 else 0)
            }
        }

        # Add price history
        if not df.empty:
            history = df.tail(100).copy()
            history['rsi'] = calculate_rsi(df).tail(100)
            history['atr'] = calculate_atr(df).tail(100)
            history['timestamp'] = history['timestamp'].astype(str)
            
            # Convert numeric columns to native Python types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'atr']
            for col in numeric_columns:
                if col in history:
                    history[col] = history[col].astype(float)
            
            # Convert DataFrame to records and ensure all values are JSON serializable
            records = history.to_dict('records')
            for record in records:
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.floating)):
                        record[key] = float(value)
                    elif isinstance(value, np.bool_):
                        record[key] = bool(value)
            data['price_history'] = records

        # Save to MongoDB
        db.save_bot_data(data)
        
    except Exception as e:
        print(f"Error saving bot data: {e}")
        import traceback
        traceback.print_exc()

def fetch_price_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

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

def check_volume_spike(df):
    avg_vol = df['volume'].rolling(window=20).mean()
    return df['volume'].iloc[-1] > 1.05 * avg_vol.iloc[-1]  # Lowered from 1.1 to 1.05

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

def open_position(side, size, price, stop_loss, take_profit):
    paper_trading['position'] = {
        'side': side,
        'size': size,
        'entry_price': price,
        'leverage': 10,
        'unrealized_pnl': 0.0,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'open_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    print(f"[PAPER] Opened {side} position: Size={size:.6f} BTC at {price:.2f} USDT")
    print(f"[PAPER] Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")

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

def calculate_position_size(current_price):
    # Risk 1% of paper trading balance
    risk_amount = paper_trading['balance'] * 0.01
    position_size = (risk_amount * 10) / current_price  # 10x leverage
    return position_size

def update_trailing_stop(current_price):
    if not active_position['side'] or not active_position['trailing_stop']:
        return

    if active_position['side'] == 'long':
        new_stop = current_price * (1 - TRAILING_STOP_INITIAL)
        if new_stop > active_position['trailing_stop']:
            active_position['trailing_stop'] = new_stop
            # try:
            #     # Update stop loss order
            #     open_orders = exchange.fetch_open_orders(symbol)
            #     for order in open_orders:
            #         if order['type'] == 'stop':
            #             exchange.cancel_order(order['id'], symbol)
                
            #     exchange.create_order(
            #         symbol,
            #         'stop',
            #         'sell',
            #         active_position['position_size'],
            #         new_stop,
            #         {'stopPrice': new_stop}
            #     )
            print(f"Trailing stop updated to: {new_stop:.2f}")
            # except Exception as e:
            #     print(f"Error updating trailing stop: {e}")

    elif active_position['side'] == 'short':
        new_stop = current_price * (1 + TRAILING_STOP_INITIAL)
        if new_stop < active_position['trailing_stop']:
            active_position['trailing_stop'] = new_stop
            # try:
            #     # Update stop loss order
            #     open_orders = exchange.fetch_open_orders(symbol)
            #     for order in open_orders:
            #         if order['type'] == 'stop':
            #             exchange.cancel_order(order['id'], symbol)
                
            #     exchange.create_order(
            #         symbol,
            #         'stop',
            #         'buy',
            #         active_position['position_size'],
            #         new_stop,
            #         {'stopPrice': new_stop}
            #     )
            print(f"Trailing stop updated to: {new_stop:.2f}")
            # except Exception as e:
            #     print(f"Error updating trailing stop: {e}")

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

while True:
    try:
        current_time = time.time()
        df = fetch_price_data()
        rsi = calculate_rsi(df)
        atr = calculate_atr(df)
        atr_threshold = calculate_atr_threshold(df)
        volume_spike = check_volume_spike(df)
        update_funding_rate()  # Update funding rate every 5 minutes
        last_price = df['close'].iloc[-1]
        
        # Update paper trading position PnL
        update_position_pnl(last_price)
        
        # Save data for dashboard
        save_bot_data(df, last_price, rsi.iloc[-1], atr.iloc[-1])
        
        # Check news and sentiment every 10 minutes
        if current_time - last_news_check > news_check_interval:
            news_list = fetch_news()
            last_sentiment_score = analyze_sentiment(news_list)
            log_sentiment(last_sentiment_score, news_list)  # Log sentiment and headlines
            print(f"News Sentiment Score: {last_sentiment_score:.3f}")
            last_news_check = current_time
        
        # Display performance summary every 30 minutes
        if current_time - last_summary_time >= SUMMARY_INTERVAL:
            display_performance_summary()
            last_summary_time = current_time
        
        # Calculate position size for new trades
        position_size = calculate_position_size(last_price)

        # Support & resistance (extended to 5-minute window)
        recent_high = df['high'].iloc[-5:].max()  # Changed from -20 to -5 for tighter range
        recent_low = df['low'].iloc[-5:].min()    # Changed from -20 to -5 for tighter range

        breakout = last_price > recent_high
        breakdown = last_price < recent_low

        # Calculate trading conditions
        sentiment_condition_long = (not is_sentiment_valid()) or (last_sentiment_score > 0.05)
        sentiment_condition_short = (not is_sentiment_valid()) or (last_sentiment_score < -0.05)
        funding_condition_long = current_funding_rate < 0.005
        funding_condition_short = current_funding_rate > -0.005
        atr_condition = atr.iloc[-1] > atr_threshold

        print(f"\n[PAPER] Current Price: {last_price:.2f} USDT")
        print(f"[PAPER] ATR: {atr.iloc[-1]:.2f} (Threshold: {atr_threshold:.2f})")
        print(f"[PAPER] Balance: {paper_trading['balance']:.2f} USDT")
        
        # Display trading conditions
        print("\n=== Trading Conditions ===")
        print("Long Entry Conditions:")
        print(f"  {'YES' if breakout else 'NO '} Breakout: Price > Recent High ({recent_high:.2f})")
        print(f"  {'YES' if volume_spike else 'NO '} Volume Spike: {df['volume'].iloc[-1] / df['volume'].rolling(window=20).mean().iloc[-1]:.2f}x")
        print(f"  {'YES' if funding_condition_long else 'NO '} Funding Rate: {current_funding_rate:.4%} < 0.5%")
        print(f"  {'YES' if atr_condition else 'NO '} ATR: {atr.iloc[-1]:.2f} > {atr_threshold:.2f}")
        print(f"  {'YES' if sentiment_condition_long else 'NO '} Sentiment: {last_sentiment_score:.3f} > 0.05")
        
        print("\nShort Entry Conditions:")
        print(f"  {'YES' if breakdown else 'NO '} Breakdown: Price < Recent Low ({recent_low:.2f})")
        print(f"  {'YES' if volume_spike else 'NO '} Volume Spike: {df['volume'].iloc[-1] / df['volume'].rolling(window=20).mean().iloc[-1]:.2f}x")
        print(f"  {'YES' if funding_condition_short else 'NO '} Funding Rate: {current_funding_rate:.4%} > -0.5%")
        print(f"  {'YES' if atr_condition else 'NO '} ATR: {atr.iloc[-1]:.2f} > {atr_threshold:.2f}")
        print(f"  {'YES' if sentiment_condition_short else 'NO '} Sentiment: {last_sentiment_score:.3f} < -0.05")
        print("=" * 30)
        
        if paper_trading['position']['side']:
            print(f"[PAPER] Active {paper_trading['position']['side']} position:")
            print(f"Size: {paper_trading['position']['size']:.6f} BTC")
            print(f"Entry: {paper_trading['position']['entry_price']:.2f} USDT")
            print(f"Current PnL: {paper_trading['position']['unrealized_pnl']:.2f} USDT")
            print(f"Stop Loss: {paper_trading['position']['stop_loss']:.2f} USDT")
            print(f"Take Profit: {paper_trading['position']['take_profit']:.2f} USDT")
            print("[PAPER] Position active - waiting for exit signals")
        else:
            # Trading logic for opening new positions
            if (breakout and 
                volume_spike and 
                current_funding_rate < 0.005 and 
                atr.iloc[-1] > atr_threshold and 
                sentiment_condition_long):  # Modified sentiment condition
                
                entry_price = last_price
                stop_loss = entry_price * (1 - 0.02)  # 2% stop loss
                take_profit = entry_price * 1.06  # 6% take profit
                open_position('long', position_size, entry_price, stop_loss, take_profit)

            if (breakdown and 
                volume_spike and 
                current_funding_rate > -0.005 and 
                atr.iloc[-1] > atr_threshold and 
                sentiment_condition_short):  # Modified sentiment condition
                
                entry_price = last_price
                stop_loss = entry_price * (1 + 0.02)  # 2% stop loss
                take_profit = entry_price * 0.94  # 6% take profit
                open_position('short', position_size, entry_price, stop_loss, take_profit)

    except Exception as e:
        print(f"Error in main loop: {e}")
        
    time.sleep(60)
