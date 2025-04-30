import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
import os
from db_helper import DatabaseHelper

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
        refresh_interval = st.slider(
            "Auto-refresh interval (seconds)",
            min_value=10,
            max_value=300,
            value=30,
            step=10,
            key="refresh_interval"
        )
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        # Auto-refresh status
        st.write(f"Next auto-refresh in: {refresh_interval} seconds")
        
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
        st.subheader("Trading Conditions")
        conditions = data.get('trading_conditions', {})
        current_values = conditions.get('current_values', {})
        
        # Create two columns for long and short conditions
        long_col, short_col = st.columns(2)
        
        with long_col:
            st.markdown("### Long Entry Conditions")
            
            # Primary Conditions
            st.markdown("#### Primary Conditions")
            price_above_donchian = current_values.get('price', 0) > current_values.get('donchian_high', 0)
            volume_spike = current_values.get('volume_spike_ratio', 0) > 1.3
            
            st.markdown(f"**{'✅' if price_above_donchian else '❌'}** Price Above Donchian High")
            st.markdown(f"**{'✅' if volume_spike else '❌'}** Volume > 1.3x 20MA")
            
            # Secondary Conditions
            st.markdown("#### Secondary Conditions")
            price_above_vwap = current_values.get('price', 0) > current_values.get('vwap', 0)
            supertrend_bullish = current_values.get('supertrend', False)
            
            st.markdown(f"**{'✅' if price_above_vwap else '❌'}** Price Above VWAP")
            st.markdown(f"**{'✅' if supertrend_bullish else '❌'}** SuperTrend Bullish")
            
            # Show current values
            st.markdown("#### Current Values")
            st.write(f"Price: ${current_values.get('price', 0):.2f}")
            st.write(f"Donchian High: ${current_values.get('donchian_high', 0):.2f}")
            st.write(f"VWAP: ${current_values.get('vwap', 0):.2f}")
            st.write(f"Volume Ratio: {current_values.get('volume_spike_ratio', 0):.2f}x")
        
        with short_col:
            st.markdown("### Short Entry Conditions")
            
            # Primary Conditions
            st.markdown("#### Primary Conditions")
            price_below_donchian = current_values.get('price', 0) < current_values.get('donchian_low', 0)
            
            st.markdown(f"**{'✅' if price_below_donchian else '❌'}** Price Below Donchian Low")
            st.markdown(f"**{'✅' if volume_spike else '❌'}** Volume > 1.3x 20MA")
            
            # Secondary Conditions
            st.markdown("#### Secondary Conditions")
            price_below_vwap = current_values.get('price', 0) < current_values.get('vwap', 0)
            supertrend_bearish = not current_values.get('supertrend', False)
            
            st.markdown(f"**{'✅' if price_below_vwap else '❌'}** Price Below VWAP")
            st.markdown(f"**{'✅' if supertrend_bearish else '❌'}** SuperTrend Bearish")
            
            # Show current values
            st.markdown("#### Current Values")
            st.write(f"Price: ${current_values.get('price', 0):.2f}")
            st.write(f"Donchian Low: ${current_values.get('donchian_low', 0):.2f}")
            st.write(f"VWAP: ${current_values.get('vwap', 0):.2f}")
            st.write(f"Volume Ratio: {current_values.get('volume_spike_ratio', 0):.2f}x")

        # Display technical indicators
        st.subheader("Market Data")
        tech_cols = st.columns(5)
        
        with tech_cols[0]:
            st.metric("Donchian High", f"${current_values.get('donchian_high', 0):.2f}")
        
        with tech_cols[1]:
            st.metric("Donchian Low", f"${current_values.get('donchian_low', 0):.2f}")
        
        with tech_cols[2]:
            st.metric("VWAP", f"${current_values.get('vwap', 0):.2f}")
        
        with tech_cols[3]:
            st.metric("Volume Ratio", f"{current_values.get('volume_spike_ratio', 0):.2f}x")
            
        with tech_cols[4]:
            st.metric("ATR", f"${data['market_data'].get('current_atr', 0):.2f}")

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

            # Add Donchian Channels
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['donchian_high'],
                    name='Donchian High',
                    line=dict(color='green', width=1, dash='dash')
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['donchian_low'],
                    name='Donchian Low',
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=1, col=1
            )

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
            for trade in data['recent_trades']:
                # Entry point
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

            # Volume subplot with spike threshold
            fig.add_trace(
                go.Bar(x=df['timestamp'], y=df['volume'], name='Volume'),
                row=2, col=1
            )
            
            # Add volume MA
            volume_ma = df['volume'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=volume_ma * 1.3,  # Volume spike threshold (1.3x)
                    name='Volume Threshold',
                    line=dict(color='red', dash='dash')
                ),
                row=2, col=1
            )

            # Update layout for better mobile viewing
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

if __name__ == "__main__":
    main() 