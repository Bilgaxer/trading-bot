import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time
from db_helper import DatabaseHelper
from bot import (
    calculate_vwap, calculate_supertrend,
    calculate_atr, calculate_obv, calculate_keltner_channel,
    calculate_pivot_levels, is_strong_candle, fetch_5m_ema,
    check_secondary_boosters
)
import numpy as np
import pytz

# Streamlit page config
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better mobile view and dark mode
st.markdown("""
    <style>
    .stMetric {
        background-color: var(--background-color, #f0f2f6);
        color: var(--text-color, #222222);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: 600;
        transition: background-color 0.3s, color 0.3s;
    }
    .st-dark .stMetric {
        background-color: #1E1E1E !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        font-weight: 700;
    }
    .stMetricLabel, .stMetricValue, .stMetricDelta {
        color: inherit !important;
    }
    @media (max-width: 768px) {
        .stMetric {
            margin-bottom: 5px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Add password protection
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()

# Dashboard title
st.title("Crypto Trading Dashboard")

# Create columns for key metrics
col1, col2, col3, col4 = st.columns(4)

# Initialize database connection
@st.cache_resource
def init_db():
    db = DatabaseHelper()
    db.connect()
    return db

# Function to load and parse bot data
@st.cache_data(ttl=30)  # Cache data for 30 seconds by default
def load_bot_data():
    try:
        print("Initializing database connection...")
        db = init_db()
        print("Getting bot data from database...")
        data = db.get_bot_data()
        if not data:
            print("No trading data available from database")
            st.error("No trading data available")
            return None
        print(f"Data loaded successfully: {len(str(data))} bytes")
        return to_native_types(data)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

def to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

def main():
    print("Entering main function...")
    st.write("Starting main execution...")
    
    # Add refresh controls to sidebar
    with st.sidebar:
        st.write("Loading sidebar...")
        st.subheader("Refresh Settings")
        
        # Load current data to check position status
        print("Loading bot data...")
        data = load_bot_data()
        if data is None:
            st.error("Failed to load bot data")
            print("Bot data is None")
            return
        
        print("Bot data loaded successfully")
        has_active_position = data and data['position']['side'] is not None
        
        # Set fixed refresh interval to 10 seconds
        refresh_interval = 10
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        # Auto-refresh status
        st.write(f"Auto-refreshing every {refresh_interval} seconds")
        if has_active_position:
            st.info("Active position detected")
        
        # Mobile view toggle
        st.toggle('Mobile View', key='mobile_view')
        
        # Chart indicator toggles
        st.subheader("Chart Indicators")
        show_ut_bot = st.toggle('UT Bot Alert', value=True)
        show_trail_stop = st.toggle('UT Bot Trail Stop', value=True)
        show_price = st.toggle('Price', value=True)

        st.subheader("Chart Display Settings")
        max_candles = len(data['price_history']) if 'price_history' in data else 100
        num_candles = st.slider('Candles to display', min_value=20, max_value=max_candles, value=min(100, max_candles), step=10)

    # Load and display data
    data = load_bot_data()
    
    if data:
        # Update key metrics with proper formatting
        with col1:
            st.metric(
                "Current Balance",
                f"${data['balance']:.2f}",
                delta=f"${data['total_pnl']:.2f}"
            )
        
        with col2:
            daily_pnl = data['performance_summary']['daily_pnl']
            daily_pnl_color = "green" if daily_pnl >= 0 else "red"
            st.metric(
                "Daily PnL",
                f"${daily_pnl:.2f}",
                delta_color=daily_pnl_color
            )
        
        with col3:
            st.metric(
                "Total PnL",
                f"${data['total_pnl']:.2f}",
                delta=f"{((data['balance'] / data['initial_balance']) - 1) * 100:.1f}%"
            )

        with col4:
            st.metric(
                "Current Price",
                f"${data['market_data']['current_price']:.2f}"
            )

        # Display trading conditions
        display_trading_conditions(data)

        # Display market data
        display_market_data(data)

        # Display secondary boosters
        if 'price_history' in data:
            df = pd.DataFrame(data['price_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            display_secondary_boosters(df)

        # Display trade management events
        display_trade_management(data)

        # Create price chart with mobile-friendly layout
        if 'price_history' in data:
            df = pd.DataFrame(data['price_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
            
            # Only show the last num_candles
            if len(df) > num_candles:
                df = df.iloc[-num_candles:]
            
            # Set x and y axis ranges for visible candles
            x_range = [df['timestamp'].iloc[0], df['timestamp'].iloc[-1]]
            y_min = min(df['low'].min(), df['close'].min(), df['open'].min())
            y_max = max(df['high'].max(), df['close'].max(), df['open'].max())
            y_margin = (y_max - y_min) * 0.05
            y_range = [y_min - y_margin, y_max + y_margin]
            
            # Make chart more mobile-friendly
            chart_height = 600 if st.session_state.get('mobile_view', False) else 800
            
            fig = make_subplots(rows=2, cols=1,
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.7, 0.3])

            # Price chart
            if show_price and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                fig.add_trace(
                    go.Candlestick(
                        x=df['timestamp'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ),
                    row=1, col=1
                )

            # UT Bot Alert signals
            if show_ut_bot and 'ut_position' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=[row.close if ut == 1 else None for ut, row in zip(df['ut_position'], df.itertuples())],
                        name='UT Bot Buy',
                        mode='markers',
                        marker=dict(color='lime', size=10, symbol='triangle-up'),
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=[row.close if ut == -1 else None for ut, row in zip(df['ut_position'], df.itertuples())],
                        name='UT Bot Sell',
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                    ),
                    row=1, col=1
                )

            # UT Bot Trailing Stop
            if show_trail_stop and 'trail_stop' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['trail_stop'],
                        name='UT Bot Trail Stop',
                        line=dict(color='orange', width=1, dash='dot')
                    ),
                    row=1, col=1
                )

            # Volume subplot if enabled
            if 'volume' in df.columns:
                fig.add_trace(
                    go.Bar(x=df['timestamp'], y=df['volume'], name='Volume'),
                    row=2, col=1
                )
                # Add volume MA
                volume_ma = df['volume'].rolling(window=20).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=volume_ma * 1.3,
                        name='Volume Threshold',
                        line=dict(color='red', dash='dash')
                    ),
                    row=2, col=1
                )
            else:
                st.warning("Missing volume data for volume chart.")

            # Update layout for better mobile viewing
            fig.update_layout(
                height=chart_height,
                title_text="Trading Activity",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range)
            )

            st.plotly_chart(fig, use_container_width=True)

        # Display current position
        if data['position']['side']:
            st.subheader("Current Position")
            position_cols = st.columns([1, 1, 1])
            
            with position_cols[0]:
                st.write(f"Side: {data['position']['side'].upper()}")
                st.write(f"Size: {data['position']['size']:.6f} BTC")
            
            with position_cols[1]:
                st.write(f"Entry: ${data['position']['entry_price']:.2f}")
                st.write(f"PnL: ${data['position']['unrealized_pnl']:.2f}")
            
            with position_cols[2]:
                st.write(f"Stop: ${data['position']['stop_loss']:.2f}")
                st.write(f"Target: ${data['position']['take_profit']:.2f}")

        # Performance metrics with proper formatting
        st.subheader("Performance")
        perf_cols = st.columns(5)
        
        with perf_cols[0]:
            st.metric("Win Rate", f"{data['performance_summary'].get('win_rate', 0):.1f}%")
        
        with perf_cols[1]:
            st.metric("Avg Trade", f"${data['performance_summary']['avg_trade_pnl']:.2f}")
        
        with perf_cols[2]:
            st.metric("Best", f"${data['performance_summary']['best_trade']:.2f}")
        
        with perf_cols[3]:
            st.metric("Worst", f"${data['performance_summary']['worst_trade']:.2f}")

        with perf_cols[4]:
            total_trades = data['performance_summary'].get('total_trades', 0)
            st.metric("Total Trades", f"{total_trades}")

        # Recent trades
        st.subheader("Recent Trades")
        if data['recent_trades']:
            trades_df = pd.DataFrame(data['recent_trades'])
            if 'entry_time' not in trades_df.columns:
                trades_df['entry_time'] = '-'
            if 'exit_time' not in trades_df.columns:
                trades_df['exit_time'] = '-'
            trades_df['Entry Time'] = trades_df['entry_time']
            trades_df['Exit Time'] = trades_df['exit_time']
            trades_df = trades_df[['side', 'entry', 'exit', 'pnl', 'reason', 'Entry Time', 'Exit Time']]
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.write("No recent trades")

        # Last update time in sidebar
        st.sidebar.write(f"Last Updated: {data['performance_summary']['last_update']}")

def display_trading_conditions(data):
    """Display current trading conditions"""
    st.subheader("Trading Conditions")
    current_values = data.get('trading_conditions', {}).get('current_values', {})
    ut_signal = current_values.get('ut_signal', 0)
    score = current_values.get('score', None)
    supertrend = current_values.get('supertrend', None)
    st.markdown(f"**UT Bot Alert:** {'Long' if ut_signal == 1 else 'Short' if ut_signal == -1 else 'Neutral'}")
    st.markdown(f"**Score:** {score if score is not None else 'N/A'}")
    st.markdown(f"**SuperTrend:** {'Bullish' if supertrend else 'Bearish' if supertrend is not None else 'N/A'}")
    st.markdown("**Decision Logic:**")
    st.markdown("- Score >= 2: Full Position\n- Score >= 1: Half Position\n- Otherwise: No Trade")

def display_market_data(data):
    """Display current market data with safe data access"""
    st.subheader("Market Data")
    
    # Create three columns for price, volume, and indicators
    col1, col2, col3 = st.columns(3)
    
    # Safely get nested values with defaults
    market_data = data.get('market_data', {})
    trading_conditions = data.get('trading_conditions', {})
    current_values = trading_conditions.get('current_values', {})
    
    with col1:
        st.markdown("**Price Data**")
        st.markdown(f"Current Price: {market_data.get('current_price', 'N/A')} USDT")
        st.markdown(f"VWAP: {current_values.get('vwap', 'N/A')} USDT")
        st.markdown(f"ATR: {current_values.get('atr', 'N/A')} USDT")
        st.markdown(f"ATR Stop Distance: {current_values.get('atr_stop_distance', 'N/A')} USDT")
    
    with col2:
        st.markdown("**Volume Data**")
        st.markdown(f"Current Volume: {current_values.get('volume', 'N/A')}")
        st.markdown(f"20-period Volume MA: {current_values.get('volume_ma', 'N/A')}")
        st.markdown(f"Volume Ratio: {current_values.get('volume_spike_ratio', 'N/A')}x")
        st.markdown(f"Volume Spike: {'Yes' if current_values.get('volume_spike', False) else 'No'}")
    
    with col3:
        st.markdown("**Indicators**")
        supertrend = current_values.get('supertrend')
        st.markdown(f"SuperTrend: {'Bullish' if supertrend else 'Bearish'}")
        st.markdown(f"EMA-9 > EMA-21: {'Yes' if current_values.get('ema9_above_ema21', False) else 'No'}")
        st.markdown(f"UT Bot Signal: {current_values.get('ut_signal', 'N/A')}")
        st.markdown(f"Trading Score: {current_values.get('score', 'N/A')}")
        
        # Safely calculate VWAP bands
        vwap = current_values.get('vwap')
        atr = current_values.get('atr')
        if vwap is not None and atr is not None:
            try:
                st.markdown(f"VWAP + 0.3×ATR: {vwap + (0.3 * atr):.2f} USDT")
                st.markdown(f"VWAP - 0.3×ATR: {vwap - (0.3 * atr):.2f} USDT")
            except (TypeError, ValueError):
                st.markdown("VWAP + 0.3×ATR: N/A")
                st.markdown("VWAP - 0.3×ATR: N/A")
        else:
            st.markdown("VWAP + 0.3×ATR: N/A")
            st.markdown("VWAP - 0.3×ATR: N/A")

def display_secondary_boosters(df):
    st.subheader("Secondary Booster Conditions")
    booster_names = [
        "OBV Slope",
        "Keltner Breakout",
        "5m EMA Trend",
        "Pivot Bias",
        "Strong Candle Body"
    ]
    # Long boosters
    votes_long = check_secondary_boosters(df, 'long')
    obv, obv_sma = calculate_obv(df)
    keltner_upper, _ = calculate_keltner_channel(df)
    ema_now, ema_prev = fetch_5m_ema(df)
    _, r1, _ = calculate_pivot_levels(df)
    boosters_long = [
        obv.iloc[-1] > obv_sma.iloc[-1],
        df['close'].iloc[-1] > keltner_upper.iloc[-1],
        (ema_now is not None and ema_prev is not None and ema_now > ema_prev),
        df['close'].iloc[-1] < r1,
        is_strong_candle(df)
    ]
    # Short boosters
    votes_short = check_secondary_boosters(df, 'short')
    _, keltner_lower = calculate_keltner_channel(df)
    ema_now, ema_prev = fetch_5m_ema(df)
    _, _, s1 = calculate_pivot_levels(df)
    boosters_short = [
        obv.iloc[-1] < obv_sma.iloc[-1],
        df['close'].iloc[-1] < keltner_lower.iloc[-1],
        (ema_now is not None and ema_prev is not None and ema_now < ema_prev),
        df['close'].iloc[-1] > s1,
        is_strong_candle(df)
    ]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Long Boosters**")
        st.markdown(f"Votes: **{votes_long}/5**")
        for name, met in zip(booster_names, boosters_long):
            st.markdown(f"{'✅' if met else '❌'} {name}")
    with col2:
        st.markdown("**Short Boosters**")
        st.markdown(f"Votes: **{votes_short}/5**")
        for name, met in zip(booster_names, boosters_short):
            st.markdown(f"{'✅' if met else '❌'} {name}")

def display_trade_management(data):
    st.subheader("Trade Management Events")
    pos = data.get('position', {})
    if not pos or not pos.get('side'):
        st.info("No active position.")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Side:** {pos['side'].upper()}")
        st.markdown(f"**Entry:** ${pos['entry_price']:.2f}")
        st.markdown(f"**Size:** {pos['size']:.6f} BTC")
        st.markdown(f"**Unrealized PnL:** ${pos['unrealized_pnl']:.2f}")
    with col2:
        st.markdown(f"**Stop Loss:** ${pos['stop_loss']:.2f}")
        st.markdown(f"**Take Profit (ATR-based):** ${pos['take_profit']:.2f}")
        events = []
        if pos.get('partial_tp_taken', False):
            events.append('🟢 **Partial TP:** 50% closed, 50% running')
        if pos.get('breakeven_moved', False):
            events.append('🟡 **Breakeven:** SL moved to entry')
        if pos.get('trailing_activated', False):
            events.append('🔵 **Trailing Stop:** Active at 0.3%')
        if pos.get('supertrend_flip_exit', False):
            events.append('🔴 **SuperTrend Flip:** Exit triggered')
        if pos.get('atr_stop_exit', False):
            events.append('🔴 **ATR Stop:** Exit triggered')
        if not events:
            events.append('No advanced management events yet.')
        for e in events:
            st.markdown(e)

if __name__ == "__main__":
    main()
    # Add auto-refresh
    time.sleep(10)
    st.rerun() 