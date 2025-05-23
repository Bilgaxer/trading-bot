import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
import os
from db_helper import DatabaseHelper

# Import needed functions from bot.py
from bot import (
    calculate_obv,
    calculate_keltner_channel,
    fetch_15m_ema,
    calculate_pivot_levels,
    is_strong_candle,
    check_secondary_boosters
)

# Streamlit page config
st.set_page_config(
    page_title="Crypto Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

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
st.title("Crypto Trading Bot Dashboard")

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
        db = init_db()
        data = db.get_bot_data()
        if not data:
            st.error("No trading data available")
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def main():
    # Add refresh controls to sidebar
    with st.sidebar:
        st.subheader("Refresh Settings")
        
        # Load current data to check position status
        data = load_bot_data()
        has_active_position = data and data['position']['side'] is not None
        
        if has_active_position:
            refresh_interval = st.slider(
                "Auto-refresh interval (seconds)",
                min_value=10,
                max_value=60,
                value=30,
                step=10,
                key="refresh_interval",
                help="Active position detected - updating every minute"
            )
        else:
            refresh_interval = st.slider(
                "Auto-refresh interval (seconds)",
                min_value=60,
                max_value=300,
                value=300,
                step=60,
                key="refresh_interval",
                help="No active position - updating every 5 minutes"
            )
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        # Auto-refresh status
        st.write(f"Next auto-refresh in: {refresh_interval} seconds")
        if has_active_position:
            st.info("Active position detected - faster updates enabled")
        
        # Set up auto-refresh (for local dashboard)
        if "refresh_time" not in st.session_state:
            st.session_state.refresh_time = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.refresh_time >= refresh_interval:
            st.session_state.refresh_time = current_time
            st.rerun()
        
        # Mobile view toggle
        st.toggle('Mobile View', key='mobile_view')

    # Load and display data
    data = load_bot_data()
    
    if data:
        # Update key metrics
        with col1:
            st.metric(
                "Current Balance",
                f"${data['balance']:.2f}",
                f"{data['roi']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Total PnL",
                f"${data['total_pnl']:.2f}"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{data['win_rate']:.1f}%",
                f"{data['win_trades']}/{data['total_trades']} trades"
            )
        
        with col4:
            st.metric(
                "Current Price",
                f"${data['market_data']['current_price']:.2f}",
            )

        # Display trading conditions
        display_trading_conditions(data)

        # Display market data
        display_market_data(data)

        # Display secondary boosters
        if 'price_history' in data:
            df = pd.DataFrame(data['price_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            display_secondary_boosters(df)

        # Display trade management events
        display_trade_management(data)

        # Create price chart with mobile-friendly layout
        if 'price_history' in data:
            df = pd.DataFrame(data['price_history'])
            if df.empty or not all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']):
                st.warning("Price history data is missing or malformed. Chart cannot be displayed.")
            else:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    st.warning(f"Could not parse timestamps: {e}")
                chart_height = 600 if st.session_state.get('mobile_view', False) else 800
                fig = make_subplots(rows=2, cols=1,
                                  shared_xaxes=True,
                                  vertical_spacing=0.05,
                                  row_heights=[0.7, 0.3])
                # Price chart with trades and indicators
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
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
                else:
                    st.warning("Missing OHLC data for price chart.")
                # Add Donchian Channels
                if 'donchian_high' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['donchian_high'],
                            name='Donchian High',
                            line=dict(color='green', width=1, dash='dash')
                        ),
                        row=1, col=1
                    )
                if 'donchian_low' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['donchian_low'],
                            name='Donchian Low',
                            line=dict(color='red', width=1, dash='dash')
                        ),
                        row=1, col=1
                    )
                if 'donchian_mid' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['donchian_mid'],
                            name='Donchian Mid',
                            line=dict(color='gray', width=1)
                        ),
                        row=1, col=1
                    )
                # Add VWAP
                if 'vwap' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['vwap'],
                            name='VWAP',
                            line=dict(color='purple', width=1)
                        ),
                        row=1, col=1
                    )
                # Add trades to the chart
                for trade in data.get('recent_trades', []):
                    try:
                        trade_time = pd.to_datetime(trade['timestamp'])
                        fig.add_trace(
                            go.Scatter(
                                x=[trade_time],
                                y=[trade['entry']],
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-up' if trade['side'] == 'long' else 'triangle-down',
                                    size=12,
                                    color='green' if trade['side'] == 'long' else 'red'
                                ),
                                name=f"{trade['side'].capitalize()} Entry"
                            ),
                            row=1, col=1
                        )
                    except Exception as e:
                        st.warning(f"Could not plot trade marker: {e}")
                # Volume subplot with spike threshold
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
                fig.update_layout(
                    height=chart_height,
                    title_text="Trading Activity",
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=30, b=10)
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

        # Performance metrics
        st.subheader("Performance")
        perf_cols = st.columns(4)
        
        with perf_cols[0]:
            st.metric("Daily PnL", f"${data['performance_summary']['daily_pnl']:.2f}")
        
        with perf_cols[1]:
            st.metric("Avg Trade", f"${data['performance_summary']['avg_trade_pnl']:.2f}")
        
        with perf_cols[2]:
            st.metric("Best", f"${data['performance_summary']['best_trade']:.2f}")
        
        with perf_cols[3]:
            st.metric("Worst", f"${data['performance_summary']['worst_trade']:.2f}")

        # Recent trades table
        if data['recent_trades']:
            st.subheader("Recent Trades")
            trades_df = pd.DataFrame(data['recent_trades'])
            st.dataframe(
                trades_df,
                use_container_width=True,
                hide_index=True
            )

        # Last update time in sidebar
        st.sidebar.write(f"Last Updated: {data['performance_summary']['last_update']}")
        
        # Show local runtime information
        st.sidebar.write("---")
        st.sidebar.write("Running in Local Mode")
        st.sidebar.write(f"Server Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def display_trading_conditions(data):
    """Display current trading conditions"""
    st.subheader("Trading Conditions")
    
    # Create two columns for long and short conditions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Long Entry Conditions**")
        st.markdown("Primary Conditions:")
        st.markdown(f"- Price > VWAP + 0.3×ATR: {'✅' if data['trading_conditions']['long_conditions']['long'] else '❌'}")
        st.markdown(f"- Volume > 1.3x 20MA: {'✅' if data['trading_conditions']['long_conditions']['volume_spike'] else '❌'}")
        
        st.markdown("Secondary Conditions:")
        st.markdown(f"- Price Above VWAP or SuperTrend Bullish: {'✅' if data['trading_conditions']['long_conditions']['secondary']['long'] else '❌'}")
    
    with col2:
        st.markdown("**Short Entry Conditions**")
        st.markdown("Primary Conditions:")
        st.markdown(f"- Price < VWAP - 0.3×ATR: {'✅' if data['trading_conditions']['short_conditions']['short'] else '❌'}")
        st.markdown(f"- Volume > 1.3x 20MA: {'✅' if data['trading_conditions']['short_conditions']['volume_spike'] else '❌'}")
        
        st.markdown("Secondary Conditions:")
        st.markdown(f"- Price Below VWAP or SuperTrend Bearish: {'✅' if data['trading_conditions']['short_conditions']['secondary']['short'] else '❌'}")

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
    
    with col2:
        st.markdown("**Volume Data**")
        st.markdown(f"Current Volume: {current_values.get('volume', 'N/A')}")
        st.markdown(f"20-period Volume MA: {current_values.get('volume_ma', 'N/A')}")
        st.markdown(f"Volume Ratio: {current_values.get('volume_spike_ratio', 'N/A')}x")
    
    with col3:
        st.markdown("**Indicators**")
        supertrend = current_values.get('supertrend')
        st.markdown(f"SuperTrend: {'Bullish' if supertrend else 'Bearish'}")
        
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
        "15m EMA Trend",
        "Pivot Bias",
        "Strong Candle Body"
    ]
    # Long boosters
    votes_long = check_secondary_boosters(df, 'long')
    obv, obv_sma = calculate_obv(df)
    keltner_upper, _ = calculate_keltner_channel(df)
    ema_now, ema_prev = fetch_15m_ema()
    _, r1, _ = calculate_pivot_levels(df)
    boosters_long = [
        obv.iloc[-1] > obv_sma.iloc[-1],
        df['close'].iloc[-1] > keltner_upper.iloc[-1],
        ema_now > ema_prev,
        df['close'].iloc[-1] < r1,
        is_strong_candle(df)
    ]
    # Short boosters
    votes_short = check_secondary_boosters(df, 'short')
    _, keltner_lower = calculate_keltner_channel(df)
    _, _, s1 = calculate_pivot_levels(df)
    boosters_short = [
        obv.iloc[-1] < obv_sma.iloc[-1],
        df['close'].iloc[-1] < keltner_lower.iloc[-1],
        ema_now < ema_prev,
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
        if not events:
            events.append('No advanced management events yet.')
        for e in events:
            st.markdown(e)

if __name__ == "__main__":
    main() 