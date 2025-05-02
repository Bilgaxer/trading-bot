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

# --- Trading state variables and config (must be global) ---
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

breakout_state = {
    'breakout_bar': None,  # Price level where breakout occurred
    'breakout_time': None,  # Time of breakout
    'breakout_side': None,  # 'long' or 'short'
    'last_stop_hit': None,  # Time of last stop loss hit
    'last_stop_price': None  # Price where stop was hit
}

TP_MULTIPLIER = 1.4  # k value for ATR-based take profit (adjust as needed)
ATR_MULTIPLIER = 0.3  # 0.3 × ATR for VWAP offset
VOLUME_THRESHOLD = 1.3  # Initial entry threshold (0.5% position)
POSITION_SIZE_1 = 0.005  # 0.5% of capital for initial position
POSITION_SIZE_2 = 0.01   # 1% of capital for second addition
POSITION_SIZE_3 = 0.05   # 5% of capital max position size
MAX_POSITION_SIZE = 0.05  # 5% of total capital cap per trade
TP_PERCENTAGE = 0.007  # 0.7% take profit
REENTRY_TIME_LIMIT = 300  # 5 minutes to consider re-entry after stop loss
INITIAL_STOP_PERCENTAGE = 0.004  # 0.4% initial stop loss
POSITION_SIZE_BOOST = 1.5  # 50% boost when secondary conditions are met

# --- Utility functions and class definitions below ---
# (leave all your indicator, calculation, and helper functions here)

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
    """Check primary trading conditions with breakout confirmation"""
    current_price = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    vwap = df['vwap'].iloc[-1]
    atr = df['atr'].iloc[-1]
    
    # Check volume condition
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'].iloc[-1] / volume_ma.iloc[-1]
    
    # Calculate VWAP + ATR thresholds
    vwap_atr_long = vwap + (ATR_MULTIPLIER * atr)
    vwap_atr_short = vwap - (ATR_MULTIPLIER * atr)
    
    # Check SuperTrend condition
    supertrend_bullish = df['supertrend'].iloc[-1]
    
    # Track breakout conditions
    if current_price > vwap_atr_long and prev_close <= vwap_atr_long:
        breakout_state['breakout_bar'] = vwap_atr_long
        breakout_state['breakout_time'] = time.time()
        breakout_state['breakout_side'] = 'long'
    elif current_price < vwap_atr_short and prev_close >= vwap_atr_short:
        breakout_state['breakout_bar'] = vwap_atr_short
        breakout_state['breakout_time'] = time.time()
        breakout_state['breakout_side'] = 'short'
    
    # Check if price stays above/below breakout level
    breakout_confirmed = False
    if breakout_state['breakout_bar'] is not None:
        if breakout_state['breakout_side'] == 'long':
            breakout_confirmed = current_price > breakout_state['breakout_bar']
        else:
            breakout_confirmed = current_price < breakout_state['breakout_bar']
    
    # Determine position size based on volume
    position_size = 0
    if volume_ratio >= 2.0:
        position_size = POSITION_SIZE_3
    elif volume_ratio >= 1.7:
        position_size = POSITION_SIZE_2
    elif volume_ratio >= VOLUME_THRESHOLD:
        position_size = POSITION_SIZE_1
    
    # Check secondary conditions
    above_vwap = current_price > vwap
    below_vwap = current_price < vwap
    
    return {
        'long': current_price > vwap_atr_long and volume_ratio >= VOLUME_THRESHOLD and breakout_confirmed,
        'short': current_price < vwap_atr_short and volume_ratio >= VOLUME_THRESHOLD and breakout_confirmed,
        'volume_spike': volume_ratio >= VOLUME_THRESHOLD,
        'volume_ratio': volume_ratio,
        'position_size': position_size,
        'vwap': vwap,
        'atr': atr,
        'secondary': {
            'long': above_vwap or supertrend_bullish,
            'short': below_vwap or (not supertrend_bullish)
        }
    }

def check_secondary_conditions(df):
    """Check secondary confirmation conditions"""
    current_price = df['close'].iloc[-1]
    vwap = df['vwap'].iloc[-1]
    atr = df['atr'].iloc[-1]
    
    # VWAP condition
    above_vwap = current_price > vwap
    below_vwap = current_price < vwap
    
    # SuperTrend condition
    supertrend_bullish = df['supertrend'].iloc[-1]
    
    return {
        'long': above_vwap or supertrend_bullish,
        'short': below_vwap or (not supertrend_bullish)
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

        # Get trading conditions and convert numpy bool values to Python bool values
        conditions = check_primary_conditions(df)
        conditions_serializable = {
            'long': bool(conditions['long']),
            'short': bool(conditions['short']),
            'volume_spike': bool(conditions['volume_spike']),
            'volume_ratio': float(conditions['volume_ratio']),
            'position_size': float(conditions['position_size']),
            'vwap': float(conditions['vwap']),
            'atr': float(conditions['atr']),
            'secondary': {
                'long': bool(conditions['secondary']['long']),
                'short': bool(conditions['secondary']['short'])
            }
        }
        
        # Current values for display
        current_values = {
            'price': float(last_price),
            'vwap': float(df['vwap'].iloc[-1]),
            'volume': float(df['volume'].iloc[-1]),
            'volume_ma': float(df['volume'].rolling(window=20).mean().iloc[-1]),
            'volume_spike_ratio': float(df['volume'].iloc[-1] / df['volume'].rolling(window=20).mean().iloc[-1]),
            'atr': float(atr_value),
            'supertrend': bool(df['supertrend'].iloc[-1])
        }

        # Position data
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
        # Calculate all indicators needed for each row
        obv, obv_sma = calculate_obv(df)
        keltner_upper, keltner_lower = calculate_keltner_channel(df)
        ema9, ema21 = calculate_emas(df)
        atr = calculate_atr(df)
        # Calculate pivots for each row (useful for today)
        pivots = [calculate_pivot_levels(df.iloc[:i+1]) for i in range(len(df))]
        pivot_vals = [p[0] for p in pivots]
        r1_vals = [p[1] for p in pivots]
        s1_vals = [p[2] for p in pivots]
        # For 15m EMA, just use the last value for all rows (since we can't fetch 15m data for each row)
        ema_15m_now, ema_15m_prev = 0, 0
        try:
            ema_15m_now, ema_15m_prev = fetch_15m_ema()
        except Exception:
            pass
        for idx, row in df.iterrows():
            price_history.append({
                'timestamp': str(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'vwap': float(row['vwap']),
                'supertrend': bool(row['supertrend']),
                'supertrend_upper': float(row['supertrend_upper']) if 'supertrend_upper' in row else None,
                'supertrend_lower': float(row['supertrend_lower']) if 'supertrend_lower' in row else None,
                'atr': float(atr.iloc[idx]) if hasattr(atr, 'iloc') else float(atr),
                'ema9': float(ema9.iloc[idx]) if hasattr(ema9, 'iloc') else float(ema9),
                'ema21': float(ema21.iloc[idx]) if hasattr(ema21, 'iloc') else float(ema21),
                'obv': float(obv.iloc[idx]) if hasattr(obv, 'iloc') else float(obv),
                'obv_sma': float(obv_sma.iloc[idx]) if hasattr(obv_sma, 'iloc') else float(obv_sma),
                'keltner_upper': float(keltner_upper.iloc[idx]) if hasattr(keltner_upper, 'iloc') else float(keltner_upper),
                'keltner_lower': float(keltner_lower.iloc[idx]) if hasattr(keltner_lower, 'iloc') else float(keltner_lower),
                'ema_15m_now': float(ema_15m_now),
                'ema_15m_prev': float(ema_15m_prev),
                'pivot': float(pivot_vals[idx]),
                'r1': float(r1_vals[idx]),
                's1': float(s1_vals[idx]),
                'strong_candle': bool(is_strong_candle(df.iloc[:idx+1]))
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
                'long_conditions': conditions_serializable,
                'short_conditions': conditions_serializable,
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
            'price_history': price_history
        }

        # Save to database
        db.save_bot_data(data)
        
    except Exception as e:
        print(f"Error saving bot data: {e}")
        import traceback
        traceback.print_exc()  # Print full error traceback for debugging

def calculate_position_size(current_price, volume_ratio):
    """Calculate position size based on volume ratio:
    - 1.3x volume = 0.5% capital (initial position)
    - 1.7x volume = 1% capital (add to position)
    - 2.0x volume = 5% capital (max position size)
    """
    if volume_ratio >= 2.0:
        return (paper_trading['balance'] * 0.05) / current_price  # 5% max position
    elif volume_ratio >= 1.7:
        return (paper_trading['balance'] * 0.01) / current_price  # 1% position
    elif volume_ratio >= 1.3:
        return (paper_trading['balance'] * 0.005) / current_price  # 0.5% initial position
    return 0

def open_position(side, size, price, stop_loss=None, take_profit=None):
    """Open a new position with updated risk management"""
    if paper_trading['position']['side'] is not None:
        return False
    
    # Calculate position size based on volume ratio
    position_size = size  # Size is already calculated based on volume ratio
    
    # Get latest data for ATR-based stop loss and take profit
    df = fetch_price_data()
    atr = df['atr'].iloc[-1]
    
    # Set stop loss using Low - 0.3 × ATR for longs, High + 0.3 × ATR for shorts
    if stop_loss is None:
        if side == 'long':
            stop_loss = df['low'].iloc[-1] - (0.3 * atr)
        else:
            stop_loss = df['high'].iloc[-1] + (0.3 * atr)
    
    # ATR-based take profit
    if take_profit is None:
        if side == 'long':
            take_profit = price + TP_MULTIPLIER * atr
        else:
            take_profit = price - TP_MULTIPLIER * atr
    
    paper_trading['position'] = {
        'side': side,
        'size': position_size,
        'entry_price': price,
        'leverage': 10,
        'unrealized_pnl': 0.0,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'trailing_activated': False,
        'partial_tp_taken': False,  # Track if partial TP was taken
        'breakeven_moved': False    # Track if SL moved to breakeven
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
    position = paper_trading['position']
    if position['side'] is None:
        return
    
    size = position['size']
    entry = position['entry_price']
    leverage = position['leverage']
    
    if position['side'] == 'long':
        paper_trading['position']['unrealized_pnl'] = (current_price - entry) * size * leverage
    else:  # short
        paper_trading['position']['unrealized_pnl'] = (entry - current_price) * size * leverage
    
    # Move SL to breakeven at +0.3% profit
    if not position.get('breakeven_moved', False):
        profit_pct = ((current_price / entry - 1) * 100) if position['side'] == 'long' else ((entry / current_price - 1) * 100)
        if profit_pct >= 0.3:
            paper_trading['position']['stop_loss'] = entry
            paper_trading['position']['breakeven_moved'] = True
            print("[INFO] Stop loss moved to breakeven!")
    
    # Partial take profit logic
    if not position.get('partial_tp_taken', False):
        if (position['side'] == 'long' and current_price >= position['take_profit']) or \
           (position['side'] == 'short' and current_price <= position['take_profit']):
            # Check SuperTrend and EMA9/EMA21
            df = fetch_price_data()
            supertrend = df['supertrend'].iloc[-1]
            ema9 = ta.trend.ema_indicator(df['close'], window=9).iloc[-1]
            ema21 = ta.trend.ema_indicator(df['close'], window=21).iloc[-1]
            if (position['side'] == 'long' and supertrend and ema9 > ema21) or \
               (position['side'] == 'short' and not supertrend and ema9 < ema21):
                # Take 50% profit, leave 50% running
                paper_trading['position']['size'] *= 0.5
                paper_trading['position']['partial_tp_taken'] = True
                print("[INFO] Partial take profit: 50% closed, 50% left to run.")
                # Set trailing stop for remaining
                paper_trading['position']['trailing_activated'] = True
                # Set trailing stop at 0.3% from current price
                if position['side'] == 'long':
                    paper_trading['position']['stop_loss'] = current_price * 0.997
                else:
                    paper_trading['position']['stop_loss'] = current_price * 1.003
                # Update take profit to a very high/low value so it doesn't trigger again
                paper_trading['position']['take_profit'] = 1e10 if position['side'] == 'long' else -1e10
            else:
                # If not, close full position
                close_position(current_price, 'Take profit hit')
                return
    
    # Trailing stop for remaining 50%
    if position.get('trailing_activated', False):
        if position['side'] == 'long':
            new_stop = current_price * 0.997
            if new_stop > position['stop_loss']:
                paper_trading['position']['stop_loss'] = new_stop
        else:
            new_stop = current_price * 1.003
            if new_stop < position['stop_loss']:
                paper_trading['position']['stop_loss'] = new_stop
    
    # Forced exit on SuperTrend flip or 15m EMA slope <= 0
    df = fetch_price_data()
    supertrend = df['supertrend'].iloc[-1]
    ema_now, ema_prev = fetch_15m_ema()
    ema_slope = ema_now - ema_prev
    if (position['side'] == 'long' and (not supertrend or ema_slope <= 0)) or \
       (position['side'] == 'short' and (supertrend or ema_slope >= 0)):
        print("[INFO] SuperTrend flip or 15m EMA slope flat/negative. Exiting remaining position.")
        close_position(current_price, 'SuperTrend/EMA exit')
        return
    
    # Check for stop loss or take profit hits (for remaining position)
    if position['side'] == 'long':
        if current_price <= position['stop_loss']:
            close_position(current_price, 'Stop loss hit')
        # take_profit is now set to a very high value after partial TP
    else:  # short
        if current_price >= position['stop_loss']:
            close_position(current_price, 'Stop loss hit')
        # take_profit is now set to a very low value after partial TP

def close_position(price, reason=''):
    """Close position and handle re-entry tracking"""
    if paper_trading['position']['side'] is None:
        return
    
    # Track stop loss hits for potential re-entry
    if reason == 'Stop loss hit':
        breakout_state['last_stop_hit'] = time.time()
        breakout_state['last_stop_price'] = price
    
    pnl = paper_trading['position']['unrealized_pnl']
    paper_trading['total_pnl'] += pnl
    paper_trading['balance'] += pnl
    
    # Update win/loss count
    if pnl > 0:
        paper_trading['win_trades'] += 1
    
    print(f"[PAPER] Closed {paper_trading['position']['side']} position at {price:.2f} USDT")
    print(f"[PAPER] PnL: {pnl:.2f} USDT ({reason})")
    print(f"[PAPER] New Balance: {paper_trading['balance']:.2f} USDT")
    
    # Record the trade
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

def display_status_update(df, conditions, last_price):
    """Display a status update with current conditions and metrics"""
    print("\n" + "="*50)
    print(f"Status Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    print(f"Current Price: ${last_price:.2f}")
    print(f"Current Balance: ${paper_trading['balance']:.2f}")
    print(f"Total PnL: ${paper_trading['total_pnl']:.2f}")
    print(f"Win Rate: {(paper_trading['win_trades'] / len(paper_trading['trades'])*100 if paper_trading['trades'] else 0):.1f}%")
    
    print("\nTrading Conditions:")
    print(f"Volume Ratio: {conditions['volume_ratio']:.2f}x (Threshold: 1.3x)")
    print(f"VWAP: ${conditions['vwap']:.2f}")
    print(f"ATR: ${conditions['atr']:.2f}")
    print(f"TP Target (ATR-based): {paper_trading['position']['take_profit']:.2f}")
    
    print("\nLong Conditions Met:" if conditions['long'] and conditions['volume_spike'] else "\nLong Conditions Not Met:")
    print(f"✓ Price > VWAP + 0.3×ATR") if conditions['long'] else print("✗ Price > VWAP + 0.3×ATR")
    print(f"✓ Volume > 1.3x 20MA") if conditions['volume_spike'] else print("✗ Volume > 1.3x 20MA")
    print(f"✓ Secondary (VWAP/SuperTrend)") if conditions['secondary']['long'] else print("✗ Secondary (VWAP/SuperTrend)")
    
    print("\nShort Conditions Met:" if conditions['short'] and conditions['volume_spike'] else "\nShort Conditions Not Met:")
    print(f"✓ Price < VWAP - 0.3×ATR") if conditions['short'] else print("✗ Price < VWAP - 0.3×ATR")
    print(f"✓ Volume > 1.3x 20MA") if conditions['volume_spike'] else print("✗ Volume > 1.3x 20MA")
    print(f"✓ Secondary (VWAP/SuperTrend)") if conditions['secondary']['short'] else print("✗ Secondary (VWAP/SuperTrend)")
    
    # Print secondary boosters
    print("\nSecondary Boosters (Long):")
    booster_names = [
        "OBV Slope Up",
        "Keltner Breakout",
        "15m EMA Up",
        "Pivot Bias (Below R1)",
        "Strong Candle Body"
    ]
    votes_long = check_secondary_boosters(df, 'long')
    print(f"Votes: {votes_long}/5")
    boosters_long = []
    obv, obv_sma = calculate_obv(df)
    boosters_long.append("OBV Slope Up" if obv.iloc[-1] > obv_sma.iloc[-1] else None)
    keltner_upper, _ = calculate_keltner_channel(df)
    boosters_long.append("Keltner Breakout" if df['close'].iloc[-1] > keltner_upper.iloc[-1] else None)
    ema_now, ema_prev = fetch_15m_ema()
    boosters_long.append("15m EMA Up" if ema_now > ema_prev else None)
    _, r1, _ = calculate_pivot_levels(df)
    boosters_long.append("Pivot Bias (Below R1)" if df['close'].iloc[-1] < r1 else None)
    boosters_long.append("Strong Candle Body" if is_strong_candle(df) else None)
    for name, met in zip(booster_names, boosters_long):
        print(f"✓ {name}" if met else f"✗ {name}")
    print("\nSecondary Boosters (Short):")
    votes_short = check_secondary_boosters(df, 'short')
    print(f"Votes: {votes_short}/5")
    boosters_short = []
    obv, obv_sma = calculate_obv(df)
    boosters_short.append("OBV Slope Down" if obv.iloc[-1] < obv_sma.iloc[-1] else None)
    _, keltner_lower = calculate_keltner_channel(df)
    boosters_short.append("Keltner Breakout" if df['close'].iloc[-1] < keltner_lower.iloc[-1] else None)
    ema_now, ema_prev = fetch_15m_ema()
    boosters_short.append("15m EMA Down" if ema_now < ema_prev else None)
    _, _, s1 = calculate_pivot_levels(df)
    boosters_short.append("Pivot Bias (Above S1)" if df['close'].iloc[-1] > s1 else None)
    boosters_short.append("Strong Candle Body" if is_strong_candle(df) else None)
    for name, met in zip([
        "OBV Slope Down",
        "Keltner Breakout",
        "15m EMA Down",
        "Pivot Bias (Above S1)",
        "Strong Candle Body"
    ], boosters_short):
        print(f"✓ {name}" if met else f"✗ {name}")
    
    # Show advanced trade management events
    pos = paper_trading['position']
    if pos['side']:
        print(f"\nActive {pos['side'].upper()} Position:")
        print(f"Entry: ${pos['entry_price']:.2f}")
        print(f"Size: {pos['size']:.6f} BTC")
        print(f"Unrealized PnL: ${pos['unrealized_pnl']:.2f}")
        print(f"Stop Loss: {pos['stop_loss']:.2f}")
        print(f"Take Profit: {pos['take_profit']:.2f}")
        if pos.get('partial_tp_taken', False):
            print("[PARTIAL TP] 50% position closed, 50% running.")
        if pos.get('breakeven_moved', False):
            print("[BREAKEVEN] Stop loss moved to entry price.")
        if pos.get('trailing_activated', False):
            print("[TRAILING STOP] Trailing stop active at 0.3%.")
    print("="*50)

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_sma'] = pd.Series(obv).rolling(window=7).mean()  # 5-10 period SMA
    return df['obv'], df['obv_sma']

def calculate_keltner_channel(df, ema_period=20, atr_period=20, mult=1.2):
    ema = ta.trend.ema_indicator(df['close'], window=ema_period)
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=atr_period)
    df['keltner_upper'] = ema + mult * atr
    df['keltner_lower'] = ema - mult * atr
    return df['keltner_upper'], df['keltner_lower']

def fetch_15m_ema():
    ohlcv_15m = exchange.fetch_ohlcv(symbol, '15m', limit=30)
    df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    ema_20 = ta.trend.ema_indicator(df_15m['close'], window=20)
    return ema_20.iloc[-1], ema_20.iloc[-2]

def calculate_pivot_levels(df):
    today = df['timestamp'].dt.date.iloc[-1]
    today_df = df[df['timestamp'].dt.date == today]
    if today_df.empty:
        today_df = df.iloc[-20:]
    high = today_df['high'].max()
    low = today_df['low'].min()
    close = today_df['close'].iloc[-1]
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def is_strong_candle(df):
    body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    rng = df['high'].iloc[-1] - df['low'].iloc[-1]
    return rng > 0 and (body / rng) >= 0.6

def check_secondary_boosters(df, side):
    votes = 0
    # OBV slope
    obv, obv_sma = calculate_obv(df)
    if side == 'long' and obv.iloc[-1] > obv_sma.iloc[-1]:
        votes += 1
    if side == 'short' and obv.iloc[-1] < obv_sma.iloc[-1]:
        votes += 1
    # Keltner breakout
    keltner_upper, keltner_lower = calculate_keltner_channel(df)
    if side == 'long' and df['close'].iloc[-1] > keltner_upper.iloc[-1]:
        votes += 1
    if side == 'short' and df['close'].iloc[-1] < keltner_lower.iloc[-1]:
        votes += 1
    # 15m EMA trend
    ema_now, ema_prev = fetch_15m_ema()
    if side == 'long' and ema_now > ema_prev:
        votes += 1
    if side == 'short' and ema_now < ema_prev:
        votes += 1
    # Pivot bias
    pivot, r1, s1 = calculate_pivot_levels(df)
    if side == 'long' and df['close'].iloc[-1] < r1:
        votes += 1
    if side == 'short' and df['close'].iloc[-1] > s1:
        votes += 1
    # Strong candle body
    if is_strong_candle(df):
        votes += 1
    return votes

# --- Main bot logic: only runs when executing this file directly ---
if __name__ == "__main__":
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

    while True:
        try:
            current_time = time.time()
            df = fetch_price_data()
            if df is None:
                print("Error fetching price data, retrying...")
                time.sleep(5)
                continue

            atr = calculate_atr(df)
            update_funding_rate()
            last_price = df['close'].iloc[-1]

            # Update paper trading position PnL
            update_position_pnl(last_price)

            # Get trading conditions
            conditions = check_primary_conditions(df)

            # Save data for dashboard - different intervals based on position status
            if paper_trading['position']['side'] is not None:
                # Save data every minute when position is active
                if current_time - last_summary_time >= 60:  # 60 seconds = 1 minute
                    save_bot_data(df, last_price, atr.iloc[-1] if hasattr(atr, 'iloc') else atr)
                    display_status_update(df, conditions, last_price)
                    last_summary_time = current_time
            else:
                # Save data every 5 minutes when no position is active
                if current_time - last_summary_time >= 300:  # 300 seconds = 5 minutes
                    save_bot_data(df, last_price, atr.iloc[-1] if hasattr(atr, 'iloc') else atr)
                    display_status_update(df, conditions, last_price)
                    last_summary_time = current_time

            # Trading logic for opening new positions or scaling in
            volume_ratio = conditions['volume_ratio']
            entry_price = last_price
            current_position = paper_trading['position']
            current_size = current_position['size'] if current_position['side'] else 0.0
            capital = paper_trading['balance']
            max_position = (capital * 0.05) / entry_price

            # Check for re-entry after stop loss
            can_reenter = False
            if (breakout_state['last_stop_hit'] is not None and 
                current_time - breakout_state['last_stop_hit'] <= REENTRY_TIME_LIMIT):
                # Check if price has reclaimed breakout level
                if (breakout_state['breakout_side'] == 'long' and 
                    last_price > breakout_state['breakout_bar']):
                    can_reenter = True
                elif (breakout_state['breakout_side'] == 'short' and 
                      last_price < breakout_state['breakout_bar']):
                    can_reenter = True

            # --- Secondary boosters ---
            secondary_votes_long = check_secondary_boosters(df, 'long')
            secondary_votes_short = check_secondary_boosters(df, 'short')

            # If 2+ boosters, go all in and tighten stop
            use_max_size_long = secondary_votes_long >= 2
            use_max_size_short = secondary_votes_short >= 2

            if conditions['long'] and conditions['volume_spike']:
                if current_position['side'] is None:
                    # No position, check for normal entry or re-entry
                    if volume_ratio >= 1.3 and (can_reenter or breakout_state['last_stop_hit'] is None):
                        if use_max_size_long:
                            position_size = (capital * 0.05) / entry_price
                            print(f"\n[BOOSTED] 2+ secondary boosters met. Going all in (5% of capital)!")
                            # Tighter stop: Low - 0.15 × ATR
                            df = fetch_price_data()
                            atr = df['atr'].iloc[-1]
                            stop_loss = df['low'].iloc[-1] - (0.15 * atr)
                            open_position('long', position_size, entry_price, stop_loss)
                        else:
                            position_size = (capital * 0.005) / entry_price
                            open_position('long', position_size, entry_price)
                elif current_position['side'] == 'long':
                    # Scale in at 1.7x and 2.0x
                    if volume_ratio >= 2.0 and current_size < max_position:
                        add_size = max_position - current_size
                        if add_size > 0:
                            print(f"\n[TRADE SIGNAL] Scaling in to max at {entry_price:.2f} USDT")
                            print(f"Adding: {add_size:.6f} BTC (to reach 5% of capital)")
                            paper_trading['position']['size'] += add_size
                    elif volume_ratio >= 1.7 and current_size < (capital * 0.015) / entry_price:
                        target_size = (capital * 0.015) / entry_price
                        add_size = target_size - current_size
                        if add_size > 0:
                            print(f"\n[TRADE SIGNAL] Scaling in at {entry_price:.2f} USDT")
                            print(f"Adding: {add_size:.6f} BTC (to reach 1.5% of capital)")
                            paper_trading['position']['size'] += add_size

            if conditions['short'] and conditions['volume_spike']:
                if current_position['side'] is None:
                    # No position, check for normal entry or re-entry
                    if volume_ratio >= 1.3 and (can_reenter or breakout_state['last_stop_hit'] is None):
                        if use_max_size_short:
                            position_size = (capital * 0.05) / entry_price
                            print(f"\n[BOOSTED] 2+ secondary boosters met. Going all in (5% of capital)!")
                            # Tighter stop: High + 0.15 × ATR
                            df = fetch_price_data()
                            atr = df['atr'].iloc[-1]
                            stop_loss = df['high'].iloc[-1] + (0.15 * atr)
                            open_position('short', position_size, entry_price, stop_loss)
                        else:
                            position_size = (capital * 0.005) / entry_price
                            open_position('short', position_size, entry_price)
                elif current_position['side'] == 'short':
                    # Scale in at 1.7x and 2.0x
                    if volume_ratio >= 2.0 and current_size < max_position:
                        add_size = max_position - current_size
                        if add_size > 0:
                            print(f"\n[TRADE SIGNAL] Scaling in to max at {entry_price:.2f} USDT")
                            print(f"Adding: {add_size:.6f} BTC (to reach 5% of capital)")
                            paper_trading['position']['size'] += add_size
                    elif volume_ratio >= 1.7 and current_size < (capital * 0.015) / entry_price:
                        target_size = (capital * 0.015) / entry_price
                        add_size = target_size - current_size
                        if add_size > 0:
                            print(f"\n[TRADE SIGNAL] Scaling in at {entry_price:.2f} USDT")
                            print(f"Adding: {add_size:.6f} BTC (to reach 1.5% of capital)")
                            paper_trading['position']['size'] += add_size

        except Exception as e:
            print("Exception in main trading loop:", e)
            import traceback
            traceback.print_exc()
        # Shorter sleep when position is active
        sleep_time = 5 if paper_trading['position']['side'] is not None else 60
        time.sleep(sleep_time)
