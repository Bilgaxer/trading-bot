import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import time
import os

# Set page config
st.set_page_config(
    page_title="Crypto Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Dashboard title
st.title("Crypto Trading Bot Dashboard")

# Create columns for key metrics
col1, col2, col3, col4 = st.columns(4)

# Function to load and parse bot data
def load_bot_data():
    try:
        with open('bot_data.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error("Bot data file not found. Make sure the bot is running.")
        return None
    except json.JSONDecodeError:
        st.error("Error reading bot data. The file might be corrupted.")
        return None

# Main dashboard update
def main():
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

        # Create price chart
        if 'price_history' in data:
            df = pd.DataFrame(data['price_history'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
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
                
                # Exit point
                if 'exit' in trade:
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['timestamp']],
                            y=[trade['exit']],
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                size=12,
                                color='red' if trade['side'] == 'long' else 'green'
                            ),
                            name=f"{trade['side'].capitalize()} Exit"
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

            # Update layout
            fig.update_layout(
                height=800,
                title_text="Trading Activity",
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # Display current position
        if data['position']['side']:
            st.subheader("Current Position")
            position_col1, position_col2, position_col3 = st.columns(3)
            
            with position_col1:
                st.write(f"Side: {data['position']['side'].upper()}")
                st.write(f"Size: {data['position']['size']:.6f} BTC")
            
            with position_col2:
                st.write(f"Entry Price: ${data['position']['entry_price']:.2f}")
                st.write(f"Current PnL: ${data['position']['unrealized_pnl']:.2f}")
            
            with position_col3:
                st.write(f"Stop Loss: ${data['position']['stop_loss']:.2f}")
                st.write(f"Take Profit: ${data['position']['take_profit']:.2f}")

        # Display recent trades
        if data['recent_trades']:
            st.subheader("Recent Trades")
            trades_df = pd.DataFrame(data['recent_trades'])
            st.dataframe(trades_df)

        # Display technical indicators
        st.subheader("Technical Indicators")
        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
        
        with tech_col1:
            st.metric("RSI", f"{data['market_data']['current_rsi']:.2f}")
        
        with tech_col2:
            st.metric("ATR", f"{data['market_data']['current_atr']:.2f}")
        
        with tech_col3:
            st.metric("Sentiment", f"{data['market_data']['sentiment_score']:.3f}")
            
        with tech_col4:
            st.metric("Funding Rate", f"{data['market_data']['funding_rate']:.4%}")

        # Display performance metrics
        st.subheader("Performance Metrics")
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Daily PnL", f"${data['performance_summary']['daily_pnl']:.2f}")
        
        with perf_col2:
            st.metric("Avg Trade PnL", f"${data['performance_summary']['avg_trade_pnl']:.2f}")
        
        with perf_col3:
            st.metric("Best Trade", f"${data['performance_summary']['best_trade']:.2f}")
        
        with perf_col4:
            st.metric("Worst Trade", f"${data['performance_summary']['worst_trade']:.2f}")

        # Last update time
        st.sidebar.write(f"Last Updated: {data['performance_summary']['last_update']}")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(5)
        st.rerun() 