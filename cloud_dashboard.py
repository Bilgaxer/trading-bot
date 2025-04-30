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
            long_conditions = conditions.get('long_conditions', {})
            
            # Display each condition with status
            for condition, status in long_conditions.items():
                if status:
                    st.markdown(f"**YES** {condition.replace('_', ' ').title()}")
                else:
                    st.markdown(f"**NO** {condition.replace('_', ' ').title()}")
            
            # Show current values relevant to long conditions
            st.markdown("#### Current Values")
            st.write(f"Price: ${current_values.get('price', 0):.2f}")
            st.write(f"Recent High: ${current_values.get('recent_high', 0):.2f}")
            st.write(f"Volume Spike Ratio: {current_values.get('volume_spike_ratio', 0):.2f}x")
            st.write(f"Funding Rate: {current_values.get('funding_rate', 0):.4%}")
            st.write(f"ATR: {current_values.get('atr', 0):.2f}")
            st.write(f"ATR Threshold: {current_values.get('atr_threshold', 0):.2f}")
            st.write(f"Sentiment Score: {current_values.get('sentiment_score', 0):.3f}")
        
        with short_col:
            st.markdown("### Short Entry Conditions")
            short_conditions = conditions.get('short_conditions', {})
            
            # Display each condition with status
            for condition, status in short_conditions.items():
                if status:
                    st.markdown(f"**YES** {condition.replace('_', ' ').title()}")
                else:
                    st.markdown(f"**NO** {condition.replace('_', ' ').title()}")
            
            # Show current values relevant to short conditions
            st.markdown("#### Current Values")
            st.write(f"Price: ${current_values.get('price', 0):.2f}")
            st.write(f"Recent Low: ${current_values.get('recent_low', 0):.2f}")
            st.write(f"Volume Spike Ratio: {current_values.get('volume_spike_ratio', 0):.2f}x")
            st.write(f"Funding Rate: {current_values.get('funding_rate', 0):.4%}")
            st.write(f"ATR: {current_values.get('atr', 0):.2f}")
            st.write(f"ATR Threshold: {current_values.get('atr_threshold', 0):.2f}")
            st.write(f"Sentiment Score: {current_values.get('sentiment_score', 0):.3f}")

        # Create price chart with mobile-friendly layout
        if 'price_history' in data:
            df = pd.DataFrame(data['price_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Make chart more mobile-friendly
            chart_height = 600 if st.session_state.get('mobile_view', False) else 800
            
            fig = make_subplots(rows=3, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              row_heights=[0.5, 0.25, 0.25])

            # Price chart with trades
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

            # RSI subplot
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['rsi'], name='RSI'),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # Volume subplot
            fig.add_trace(
                go.Bar(x=df['timestamp'], y=df['volume'], name='Volume'),
                row=3, col=1
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

        # Display technical indicators
        st.subheader("Market Data")
        tech_cols = st.columns(4)
        
        with tech_cols[0]:
            st.metric("RSI", f"{data['market_data']['current_rsi']:.2f}")
        
        with tech_cols[1]:
            st.metric("ATR", f"{data['market_data']['current_atr']:.2f}")
        
        with tech_cols[2]:
            st.metric("Sentiment", f"{data['market_data']['sentiment_score']:.3f}")
        
        with tech_cols[3]:
            st.metric("Funding", f"{data['market_data']['funding_rate']:.4%}")

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