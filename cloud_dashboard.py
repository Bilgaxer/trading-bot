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
    calculate_pivot_levels, is_strong_candle, fetch_15m_ema,
    check_secondary_boosters
)

# Streamlit page config
st.set_page_config(
    page_title="Crypto Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better mobile view
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
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
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

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
        
        # Mobile view toggle
        st.toggle('Mobile View', key='mobile_view')
        
        # Chart indicator toggles
        st.subheader("Chart Indicators")
        show_vwap = st.toggle('VWAP', value=True)
        show_supertrend = st.toggle('SuperTrend', value=True)
        show_supertrend_bands = st.toggle('SuperTrend Bands', value=True)
        show_atr = st.toggle('ATR', value=False)
        show_ema9 = st.toggle('EMA-9', value=False)
        show_ema21 = st.toggle('EMA-21', value=False)
        show_obv = st.toggle('OBV (On-Balance Volume)', value=False)
        show_obv_sma = st.toggle('OBV SMA', value=False)
        show_keltner = st.toggle('Keltner Channel', value=False)
        show_ema15m = st.toggle('15m EMA', value=False)
        show_pivots = st.toggle('Pivot/R1/S1', value=False)
        show_strong_candle = st.toggle('Strong Candle', value=False)
        show_volume = st.toggle('Volume', value=True)

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
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Make chart more mobile-friendly
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

            # Add VWAP if enabled
            if show_vwap and 'vwap' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['vwap'],
                        name='VWAP',
                        line=dict(color='purple', width=1)
                    ),
                    row=1, col=1
                )

            # Add SuperTrend if enabled
            if show_supertrend and 'supertrend' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=[row['close'] if st else None for st, row in zip(df['supertrend'], df.itertuples())],
                        name='SuperTrend',
                        line=dict(color='blue', width=2, dash='dot')
                    ),
                    row=1, col=1
                )

            # Add SuperTrend Bands if enabled
            if show_supertrend_bands and 'supertrend_upper' in df.columns and 'supertrend_lower' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['supertrend_upper'],
                        name='SuperTrend Upper',
                        line=dict(color='blue', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['supertrend_lower'],
                        name='SuperTrend Lower',
                        line=dict(color='blue', width=1, dash='dash')
                    ),
                    row=1, col=1
                )

            # Add ATR if enabled
            if show_atr and 'atr' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['atr'],
                        name='ATR',
                        line=dict(color='orange', width=1, dash='dot')
                    ),
                    row=2, col=1
                )

            # Add EMA-9 if enabled
            if show_ema9 and 'ema9' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['ema9'],
                        name='EMA-9',
                        line=dict(color='green', width=1)
                    ),
                    row=1, col=1
                )

            # Add EMA-21 if enabled
            if show_ema21 and 'ema21' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['ema21'],
                        name='EMA-21',
                        line=dict(color='red', width=1)
                    ),
                    row=1, col=1
                )

            # Add OBV if enabled
            if show_obv and 'obv' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['obv'],
                        name='OBV',
                        line=dict(color='brown', width=1)
                    ),
                    row=2, col=1
                )

            # Add OBV SMA if enabled
            if show_obv_sma and 'obv_sma' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['obv_sma'],
                        name='OBV SMA',
                        line=dict(color='gray', width=1, dash='dot')
                    ),
                    row=2, col=1
                )

            # Add Keltner Channel if enabled
            if show_keltner and 'keltner_upper' in df.columns and 'keltner_lower' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['keltner_upper'],
                        name='Keltner Upper',
                        line=dict(color='magenta', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['keltner_lower'],
                        name='Keltner Lower',
                        line=dict(color='magenta', width=1, dash='dash')
                    ),
                    row=1, col=1
                )

            # Add 15m EMA if enabled
            if show_ema15m and 'ema_15m_now' in df.columns and 'ema_15m_prev' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['ema_15m_now'],
                        name='15m EMA (now)',
                        line=dict(color='black', width=1, dash='dot')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['ema_15m_prev'],
                        name='15m EMA (prev)',
                        line=dict(color='black', width=1, dash='dash')
                    ),
                    row=1, col=1
                )

            # Add Pivot, R1, S1 if enabled
            if show_pivots and 'pivot' in df.columns and 'r1' in df.columns and 's1' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['pivot'],
                        name='Pivot',
                        line=dict(color='teal', width=1, dash='dot')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['r1'],
                        name='R1',
                        line=dict(color='teal', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['s1'],
                        name='S1',
                        line=dict(color='teal', width=1, dash='dash')
                    ),
                    row=1, col=1
                )

            # Add Strong Candle if enabled
            if show_strong_candle and 'strong_candle' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=[row['close'] if sc else None for sc, row in zip(df['strong_candle'], df.itertuples())],
                        name='Strong Candle',
                        mode='markers',
                        marker=dict(color='orange', size=8, symbol='star'),
                    ),
                    row=1, col=1
                )

            # Add trades to the chart
            for trade in data['recent_trades']:
                if 'timestamp' in trade and 'entry' in trade:
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['timestamp']],
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

            # Volume subplot if enabled
            if show_volume and 'volume' in df.columns:
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
                )
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