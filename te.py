import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- Main Data Fetching and Caching Function ---

# Add a 'run_date' parameter to the signature. It won't be used in the function,
# but its presence makes the cache automatically invalidate each day.
@st.cache_data(ttl=3600)
def fetch_technical_data(ticker, benchmark_ticker, run_date, years=3):
    """
    Fetches, calculates, and returns a DataFrame of technical metrics.
    The 'run_date' parameter ensures the cache is invalidated daily.
    """
    try:
        # Fetch historical data using yfinance
        start_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        
        df = yf.download(ticker, start=start_date, auto_adjust=True)
        bench_df = yf.download(benchmark_ticker, start=start_date, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(0)
        if isinstance(bench_df.columns, pd.MultiIndex):
            bench_df.columns = bench_df.columns.droplevel(0)

        if df.empty or len(df) < 200:
            st.error(f"Could not retrieve sufficient historical data for **{ticker}**.")
            return None
        if bench_df.empty:
            st.error(f"Could not retrieve historical data for benchmark **{benchmark_ticker}**.")
            return None

        # --- All calculations below this line remain the same ---
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        bench_df = bench_df[['Close']].rename(columns={'Close': 'Benchmark_Close'})
        latest_price = df['Close'].iloc[-1]
        
        # Performance
        perf_1m = (latest_price / df['Close'].iloc[-21] - 1) * 100 if len(df) > 21 else np.nan
        perf_3m = (latest_price / df['Close'].iloc[-63] - 1) * 100 if len(df) > 63 else np.nan
        perf_6m = (latest_price / df['Close'].iloc[-126] - 1) * 100 if len(df) > 126 else np.nan
        perf_12m = (latest_price / df['Close'].iloc[-252] - 1) * 100 if len(df) > 252 else np.nan

        # Moving Averages
        for w in [50, 100, 200]:
            df[f'{w}DMA'] = df['Close'].rolling(window=w).mean()
            df[f'Dist_{w}DMA_pct'] = (df['Close'] - df[f'{w}DMA']) / df[f'{w}DMA'] * 100
        
        # ... (The rest of your existing, correct calculations) ...
        merged = df[['Close']].merge(bench_df, left_index=True, right_index=True, how='left')
        # (further calculations...)
        
        # --- Assemble Final Dictionary ---
        latest_stats = {
            'Latest_Date': df.index[-1].strftime('%Y-%m-%d'),
            'Latest_Price': latest_price,
            '1M_Return_%': perf_1m,
            '3M_Return_%': perf_3m,
            # ... (all your other stats)
        }
        
        return pd.DataFrame([latest_stats]).T.rename(columns={0: 'Value'})

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# --- Streamlit App Interface ---
st.set_page_config(page_title="Stock Technical Dashboard", page_icon="📊", layout="wide")
st.title("📊 Technical Analysis Dashboard")
st.caption("Data is sourced from Yahoo Finance. If a valid ticker fails, please try again after a few minutes.")

col1, col2 = st.columns(2)
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., BAJAJ-AUTO.NS):", value="BAJAJ-AUTO.NS")
with col2:
    benchmark_ticker = st.text_input("Enter Benchmark Ticker (e.g., ^NSEI):", value="^NSEI")

if st.button("📈 Generate Technical Snapshot"):
    with st.spinner("Fetching and analyzing data..."):
        # --- CHANGE HERE: Pass the current date to the function ---
        today_date_str = datetime.today().strftime('%Y-%m-%d')
        data_df = fetch_technical_data(stock_ticker, benchmark_ticker, run_date=today_date_str)
    
    if data_df is not None and not data_df.empty:
        st.success(f"✅ Analysis complete for **{stock_ticker}**")
        latest_date_str = data_df.loc['Latest_Date', 'Value']
        st.info(f"All metrics are calculated as of the market close on **{latest_date_str}**.")
        st.dataframe(data_df.style.format(precision=2, na_rep="-"), use_container_width=True)
